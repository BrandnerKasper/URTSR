from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as FV

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


class GatedConv(nn.Module):
    def __init__(self, in_cha, out_cha, kernel, stride, pad):
        super(GatedConv, self).__init__()
        self.conv_feature = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=kernel, stride=stride,
                                      padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=in_cha, out_channels=1, kernel_size=kernel, stride=stride, padding=pad),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        x = self.conv_feature(x)
        mask = self.conv_mask(mask)

        return x * mask


# DELETE ONCE WE USE MASKS
class GatedConv2(nn.Module):
    def __init__(self, in_cha, out_cha, kernel, stride, pad):
        super(GatedConv2, self).__init__()
        self.conv_feature = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=kernel, stride=stride,
                                      padding=pad)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(in_channels=in_cha, out_channels=1, kernel_size=kernel, stride=stride, padding=pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv_feature(x)
        mask = self.conv_mask(x)

        return x1 * mask


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


# Residual Channel Attention Block (RCAB) - copied from NDSR
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, stride=1, padding=1))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def pad_or_crop_to_size(input_t: torch.Tensor, size: (int, int)) -> torch.Tensor:
    _, _, height, width = input_t.size()
    target_height, target_width = size
    if height < target_height:
        pad_height = max(0, target_height - height)
        input_t = FV.pad(input_t, [0, 0, 0, pad_height], padding_mode="edge")
    else:
        input_t = input_t[:, :target_height, :]
    if width < target_width:
        pad_width = max(0, target_width - width)
        input_t = FV.pad(input_t, [0, 0, pad_width, 0], padding_mode="edge")
    else:
        input_t = input_t[:, :, :target_width]
    return input_t


class NDSR(BaseModel):
    def __init__(self, scale: int, buffer_cha: int = 0, history_frames: int = 2, batch_size: int = 1, crop_size: Optional[int] = None):
        super(NDSR, self).__init__(scale=scale, down_and_up=2, do_two=False)
        self.crop_size = crop_size
        if self.crop_size is None:
            self.pre_sr = torch.randn(batch_size, 3, 2160, 3840).to(device='cuda')
            self.pre_lstm = torch.randn(batch_size, 2, 64, 1080, 1920).to(device='cuda')
        else:
            self.pre_sr = torch.randn(batch_size, 3, crop_size*2, crop_size*2).to(device='cuda')
            self.pre_lstm = torch.randn(batch_size, 2, 64, crop_size, crop_size).to(device='cuda')

        self.conv_in = nn.Conv2d(3 + buffer_cha, 32, kernel_size=3, stride=1, padding=1)

        # Warping
        self.conv_gate = GatedConv2(3, 32, kernel=3, stride=1, pad=1)

        # prev SR frame
        self.unshuffle = nn.PixelUnshuffle(scale)
        self.conv_pre_sr = nn.Conv2d(3 * scale ** 2, 32, kernel_size=3, stride=1, padding=1)

        # ConvLSTM
        self.conv_lstm = ConvLSTM(input_size=32 + 32 * history_frames + 32, hidden_size=64, kernel_size=3)

        # U-Net for reconstruction
        self.pool = nn.MaxPool2d(2)
        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.encoder_1 = nn.Sequential(
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16)
        )

        self.encoder_2 = nn.Sequential(
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16)
        )

        self.center = nn.Sequential(
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16)
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16),
            RCAB(32, 64, 3, 16)
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 3 * scale**2, 3),
            nn.PixelShuffle(scale)
        )

    def reset_pre_val(self, batch_size: int, crop_size: int) -> None:
        self.crop_size = crop_size
        if crop_size is None:
            self.pre_sr = torch.randn(batch_size, 3, 2160, 3840).to(device='cuda')
            self.pre_lstm = torch.randn(batch_size, 2, 64, 1080, 1920).to(device='cuda')
        else:
            self.pre_sr = torch.randn(batch_size, 3, crop_size * 2, crop_size * 2).to(device='cuda')
            self.pre_lstm = torch.randn(batch_size, 2, 64, crop_size, crop_size).to(device='cuda')

    def forward(self, x, his): #buffers, his, masks, pre_sr, pre_lstm):
        # append buffers if valid
        # if torch.is_tensor(buffers):
        #     x = torch.cat([x, buffers], 1)
        # LR
        x = self.conv_in(x)

        # history:
        his = torch.unbind(his, dim=1)
        # masks = torch.unbind(masks, dim=1)
        h = []
        # for his_frame, mask in zip(his, masks):
        #     h.append(self.conv_gate(his_frame, mask))
        for his_frame in his:
            h.append(self.conv_gate(his_frame))
        h = torch.cat(h, 1)

        # TODO: prev frame -> Dataloader
        pre_sr = self.unshuffle(self.pre_sr)
        pre_sr = self.conv_pre_sr(pre_sr)

        # ConvLSTM
        x = torch.cat([x, h, pre_sr], dim=1)
        # TODO: prev state -> Dataloader
        pre_lstm = torch.unbind(self.pre_lstm, dim=1)
        pre_lstm = self.conv_lstm(x, pre_lstm)
        x1 = self.encoder_1(pre_lstm[0])
        x2 = self.pool(x1)
        x2 = self.encoder_2(x2)
        x3 = self.pool(x2)
        x3 = self.center(x3)
        res = self.bilinear(x3)
        res = self.decoder_1(torch.cat([res, x2], dim=1))
        res = self.bilinear(res)
        res = self.decoder_2(torch.cat([res, x1], dim=1))
        res = self.upsampling(res)
        if self.crop_size is None:
            res = pad_or_crop_to_size(res, (2160, 3840))
        else:
            res = pad_or_crop_to_size(res, (2*self.crop_size, 2*self.crop_size))
        # save old states
        self.pre_sr = res.detach()
        self.pre_lstm = torch.stack(pre_lstm, 1).detach()

        return res


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NDSR(scale=2, buffer_cha=0, history_frames=2).to(device)
    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    # buffer = (batch_size, 6, 1920, 1080)
    his = (batch_size, 2, 3, 1920, 1080)
    # masks = (batch_size, 2, 3, 1920, 1080)
    # pre_sr = (batch_size, 3, 3840, 2160)
    # pre_lstm = (batch_size, 2, 64, 1920, 1080)

    input_size = (input_data, his) #(input_data, buffer, his, masks, pre_sr, pre_lstm) # for now we do not use warping or buffers, therefore no masks as well

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == "__main__":
    main()
