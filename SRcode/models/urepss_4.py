from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

    def forward(self, x):
        x_1 = self.conv_feature(x)
        mask = self.conv_mask(x)

        return x_1 * mask


class GatedConvBlock(nn.Module):
    def __init__(self, in_cha, out_cha):
        super().__init__()
        self.conv_1 = GatedConv(in_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = GatedConv(out_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        return x


class GatedDownConvBlock(nn.Module):
    def __init__(self, in_cha, out_cha):
        super().__init__()
        self.down_sample = GatedConv(in_cha, in_cha, kernel=3, stride=2, pad=1)
        self.conv_1 = GatedConv(in_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = GatedConv(out_cha, out_cha, kernel=3, stride=1, pad=1)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class UpConvBlock(nn.Module):
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


class URepSS_04(BaseModel):
    def __init__(self, scale: int = 2, history_frames: int = 2, buffer_cha: int = 13):
        super(URepSS_04, self).__init__(scale=scale, down_and_up=3)

        # Encoder
        self.down_sample = nn.PixelUnshuffle(scale)
        self.conv_in = GatedConvBlock((3 + buffer_cha)*4, 24)
        self.down_1 = GatedDownConvBlock(24, 24)
        self.down_2 = GatedDownConvBlock(24, 32)

        # History encoder
        self.history_encoder = nn.Sequential(
            nn.PixelUnshuffle(scale),
            GatedConvBlock((history_frames * 3) * 4, 24),
            GatedDownConvBlock(24, 24),
            GatedDownConvBlock(24, 32)
        )

        # Bottom layer with multiple rep blocks
        self.bottom_layer = self.bottom_layer = Attention(dim=64, num_heads=8, bias=True)

        # Decoder
        self.up_1 = UpConvBlock(64 + 24, 32)
        self.up_2 = UpConvBlock(32 + 24, 24)
        self.conv_out = GatedConvBlock(24, 48)
        self.up_sample = nn.PixelShuffle(scale * scale)

    def forward(self, x, his, buf=None):
        # Setup
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bilinear")

        if buf is not None:
            x = torch.cat([x, buf], 1)
        his = torch.cat(torch.unbind(his, 1), 1)

        # Encoder
        x = self.down_sample(x)
        x1 = self.conv_in(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)

        # History encoder
        his = self.history_encoder(his)

        # Bottom Layer
        x3 = torch.cat([x3, his], 1)
        x3 = self.bottom_layer(x3)

        # Decoder
        x = self.up_1(x3, x2)
        x = self.up_2(x, x1)
        x = self.conv_out(x)

        x = self.up_sample(x)
        x = x + x_up

        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = URepSS_04(scale=2, history_frames=2, buffer_cha=5).to(device)

    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    his = (batch_size, 2, 3, 1920, 1080)
    feature = (batch_size, 5, 1920, 1080)
    input_size = (input_data, his, feature)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
