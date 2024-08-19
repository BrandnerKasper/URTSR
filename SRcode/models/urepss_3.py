import numbers
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if bias:
            self.body = WithBias_LayerNorm(dim)
        else:
            self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class GatedCNNBlock(nn.Module):
    def __init__(self, in_out_channels, mid_channels=96, kernel_size=7,
                 act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Conv2d(in_out_channels, mid_channels, kernel_size=1)
        self.act = act_layer()
        self.conv = nn.Conv2d(int(mid_channels/2), int(mid_channels/2), kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=int(mid_channels/2))
        self.fc2 = nn.Conv2d(int(mid_channels/2), in_out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = x
        a, b = self.fc1(x).chunk(2, dim=1)
        a = self.conv(a)
        b = self.act(b)
        x = self.fc2(a * b)
        return x + shortcut


class BaseBlock(nn.Module):
    def __init__(self, dim, up_factor: int = 2, down_factor: int = 6):
        super(BaseBlock, self).__init__()
        self.norm_1 = LayerNorm(dim)
        self.down_gate = GatedCNNBlock(dim, int(dim // down_factor))
        self.norm_2 = LayerNorm(dim)
        self.up_gate = GatedCNNBlock(dim, int(up_factor * dim))

    def forward(self, x):
        x += self.down_gate(self.norm_1(x))
        x += self.up_gate(self.norm_2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_cha, out_cha, compress_fac):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=in_cha, out_channels=out_cha, kernel_size=3, stride=1, padding=1)
        self.body = BaseBlock(dim=out_cha, down_factor=compress_fac)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.body(x)
        return x


class DownConvBlock(nn.Module):
    def __init__(self, in_cha, out_cha, compress_fac):
        super().__init__()
        self.down_sample = nn.Conv2d(in_channels=in_cha, out_channels=in_cha, kernel_size=3, stride=2, padding=1)
        self.body = ConvBlock(in_cha=in_cha, out_cha=out_cha, compress_fac=compress_fac)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.body(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_cha, out_cha, compress_fac):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.body = ConvBlock(in_cha=in_cha, out_cha=out_cha, compress_fac=compress_fac)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.body(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

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


class URepSS_03(BaseModel):
    def __init__(self, scale: int = 2, history_frames: int = 2, buffer_cha: int = 13):
        super(URepSS_03, self).__init__(scale=scale, down_and_up=3)

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
        self.pixel_shuffle = nn.PixelShuffle(scale * scale)

        # Encoder
        self.conv_block_in = ConvBlock((3 + buffer_cha) * 4, 24, compress_fac=6)
        self.down_conv_block_1 = DownConvBlock(24, 24, compress_fac=6)
        self.down_conv_block_2 = DownConvBlock(24, 32, compress_fac=4)

        # History encoder
        self.history_encoder = nn.Sequential(
            self.pixel_unshuffle,
            ConvBlock((history_frames * 3) * 4, 24, compress_fac=6),
            DownConvBlock(24, 24, compress_fac=6),
            DownConvBlock(24, 32, compress_fac=4)
        )

        # Bottom layer with cross attention btw history features and lr frame feature
        self.bottom_layer = CrossAttention(dim=32, num_heads=8, bias=True)

        # Decoder
        self.up_1 = UpConvBlock(32 + 24, 32, compress_fac=4)
        self.up_2 = UpConvBlock(32 + 24, 24, compress_fac=6)
        self.conv_out = ConvBlock(24, 48, compress_fac=6)

    def forward(self, x, his, buf=None):
        # Setup
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bilinear")

        if buf is not None:
            x = torch.cat([x, buf], 1)
        his = torch.cat(torch.unbind(his, 1), 1)

        # Encoder
        x = self.pixel_unshuffle(x)
        x1 = self.conv_block_in(x)
        x2 = self.down_conv_block_1(x1)
        x3 = self.down_conv_block_2(x2)

        # History encoder
        his = self.history_encoder(his)

        # Bottom Layer
        x3 = self.bottom_layer(x3, his)

        # Decoder
        x = self.up_1(x3, x2)
        x = self.up_2(x, x1)
        x = self.conv_out(x)

        x = self.pixel_shuffle(x)
        x = x + x_up

        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = URepSS_03(scale=2, history_frames=2, buffer_cha=5).to(device)

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
