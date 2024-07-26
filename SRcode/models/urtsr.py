from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as FV

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


# code from NDSR
class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)
        # depth_wise_conv(input_size + hidden_size, 4 * hidden_size, kernel_size)

    def forward(self, input_, prev_state=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size).to(input_.device),
                torch.zeros(state_size).to(input_.device)
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class URTSR(BaseModel):
    def __init__(self, scale: int, history_frames: int = 2, batch_size: int = 1, crop_size: Optional[int] = None):
        super().__init__(scale=scale, down_and_up=3)

        self.pre_lstm = None

        # LR bilinear upsampled
        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unshuffle = nn.PixelUnshuffle(scale)
        self.conv_lr = nn.Conv2d(3 * scale ** 2, 32, kernel_size=3, stride=1, padding=1)

        # history
        self.conv_history = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # ConvLSTM
        self.conv_lstm = ConvLSTM(input_size=32 + 32 * history_frames, hidden_size=64, kernel_size=3)

    def forward(self, x, his):
        # LR
        x = self.bilinear(x)
        x = self.unshuffle(x)
        x = self.conv_lr(x)

        # History
        his = torch.unbind(his, dim=1)
        h = []
        for his_frame in his:
            h.append(self.conv_history(his_frame))
        h = torch.cat(h, 1)

        x = torch.cat([x, h], dim=1)

        # Conv LSTM
        self.pre_lstm = self.conv_lstm(x, self.pre_lstm)
        x = self.pre_lstm[0]

        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    crop_size = None
    model = URTSR(scale=2, history_frames=2, batch_size=batch_size, crop_size=crop_size).to(device)

    input_data = (batch_size, 3, 1920, 1080)
    his = (batch_size, 2, 3, 1920, 1080)
    input_size = (input_data, his)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == "__main__":
    main()
