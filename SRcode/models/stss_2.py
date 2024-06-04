import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


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


class Stss2(BaseModel):
    def __init__(self, scale: int, buffer_cha: int = 0, history_cha: int = 3 * 3):
        super(Stss2, self).__init__(scale=scale, down_and_up=3, do_two=False)

        self.conv_in = nn.Sequential(
            LWGatedConv2D(3 + buffer_cha*2, 24, kernel=3, stride=1, pad=1),
            nn.ReLU(inplace=True),
            LWGatedConv2D(24, 24, kernel=3, stride=1, pad=1),
            nn.ReLU(inplace=True)
        )

        self.down_1 = DownLWGated(24, 24)
        self.down_2 = DownLWGated(24, 32)
        self.down_3 = DownLWGated(32, 32)

        self.attention = Attention(dim=32, num_heads=8, bias=True)

        self.his_1 = nn.Sequential(
            nn.Conv2d(history_cha, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.his_2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.his_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up_1 = Up(64 + 32, 32)
        self.up_2 = Up(56, 24)
        self.up_3 = Up(48, 24)

        self.conv_out = nn.Sequential(
            nn.Conv2d(24, 3*2 * self.scale ** 2, kernel_size=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x, buffer, his):
        if torch.is_tensor(buffer):
            x = torch.cat([x, buffer], 1)
        x1 = self.conv_in(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        # Attention!
        x4 = self.attention(x4)

        his = torch.cat(torch.unbind(his, 1), 1)
        his = self.his_1(his)
        his = self.his_2(his)
        his = self.his_3(his)

        x4 = torch.cat([x4, his], 1)

        x = self.up_1(x4, x3)
        x = self.up_2(x, x2)
        x = self.up_3(x, x1)
        x = self.conv_out(x)

        x = torch.split(x, dim=1, split_size_or_sections=3)

        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Stss2(scale=2, buffer_cha=3, history_cha=3*3).to(device)
    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    buffer = (batch_size, 2*3, 1920, 1080) # 3 for SS, 3 for ESS
    his = (batch_size, 3, 3, 1920, 1080)
    input_size = (input_data, his, buffer)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
