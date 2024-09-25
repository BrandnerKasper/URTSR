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


# code from STSS
class LWGatedConv2D(nn.Module):
    def __init__(self, in_cha, out_cha, kernel, stride, pad):
        super(LWGatedConv2D, self).__init__()

        self.conv_feature = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=kernel, stride=stride, padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=in_cha, out_channels=1, kernel_size=kernel, stride=stride, padding=pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv_feature(x)
        mask = self.conv_mask(x)

        return x1 * mask


class DownLWGated(nn.Module):
    def __init__(self, in_cha, out_cha):
        super().__init__()
        self.down_sample = LWGatedConv2D(in_cha, in_cha, kernel=3, stride=2, pad=1)
        self.conv_1 = LWGatedConv2D(in_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = LWGatedConv2D(out_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_cha, out_cha, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                nn.Conv2d(in_cha, in_cha // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_cha // 2, out_cha, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_cha, in_cha // 2, kernel_size=2, stride=2, padding=1)
            self.conv = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_cha, out_cha, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NSRRD(BaseModel):
    """
    Neural Super Resolution for Radiance Demodulation
    https://github.com/Riga2/NSRD
    """
    def __init__(self, scale: int, history_frames: int = 2):
        super().__init__(scale=scale, down_and_up=3)

        self.pre_lstm = None

        # LR bilinear upsampled
        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unshuffle = nn.PixelUnshuffle(scale)
        self.conv_lr = nn.Conv2d(3 * scale ** 2, 32, kernel_size=3, stride=1, padding=1)

        # history
        self.conv_history = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # ConvLSTM
        self.conv_lstm = ConvLSTM(input_size=32 + 32 * history_frames, hidden_size=24, kernel_size=3)

        # UNet
        self.down_1 = DownLWGated(24, 24)
        self.down_2 = DownLWGated(24, 32)
        self.down_3 = DownLWGated(32, 32)

        self.up_1 = Up(64, 32)
        self.up_2 = Up(56, 24)
        self.up_3 = Up(48, 24)

        self.conv_out = nn.Sequential(
            nn.Conv2d(24, 3 * self.scale ** 2, kernel_size=1),
            nn.PixelShuffle(scale)
        )

    def reset(self):
        self.pre_lstm = None

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
        pre_lstm = self.conv_lstm(x, self.pre_lstm)
        x1 = pre_lstm[0]
        self.pre_lstm = tuple(tensor.detach() for tensor in pre_lstm)

        # U-Net from STSS
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x = self.down_3(x3)

        x = self.up_1(x, x3)
        x = self.up_2(x, x2)
        x = self.up_3(x, x1)

        x = self.conv_out(x)

        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NSRRD(scale=2, history_frames=2).to(device)

    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    his = (batch_size, 2, 3, 1920, 1080)
    input_size = (input_data, his)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == "__main__":
    main()
