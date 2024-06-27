import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


def depth_wise_conv(in_feats, out_feats, kernel, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_feats, in_feats, kernel_size=kernel, padding=(kernel // 2), groups=in_feats, bias=bias),
        nn.Conv2d(in_feats, out_feats, kernel_size=1)
    )


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )

    def forward(self, x):
        return self.depthwise(x)


class SEUnit(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEUnit, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CU, self).__init__()
        self.dwconv7 = DepthwiseConv2d(in_channels, 7, padding=3)
        self.dwconv5 = DepthwiseConv2d(in_channels, 5, padding=2)
        self.dwconv3 = DepthwiseConv2d(in_channels, 3, padding=1)
        self.se_unit = SEUnit(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.dwconv7(x)
        x2 = self.dwconv5(x)
        x3 = self.dwconv3(x)
        x_multi_scale = x1 + x2 + x3

        x_se = self.se_unit(x_multi_scale)

        x_out = self.conv1x1(x_se)
        x_out += x

        return x_out


class EVRModule(nn.Module):
    def __init__(self, in_cha: int = 32, cu_units: int = 2) -> None:
        super(EVRModule, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=in_cha, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=1)
        cus = []
        for i in range(cu_units):
            cus.append(CU(64, 64))
        self.cu_units = nn.Sequential(*cus)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_out = nn.Conv2d(48, 32, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_in(x)
        x = self.conv_2(x1)
        x = self.conv_4(x)
        x = self.cu_units(x)
        x = self.pixel_shuffle(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_out(x)
        return x


class EVRNet(BaseModel):
    def __init__(self, scale: int = 2) -> None:
        super(EVRNet, self).__init__(scale=scale, down_and_up=1, do_two=False)
        # Feature
        self.conv_feature = nn.Conv2d(3, 32, 3, stride=1, padding=1)

        # Alignment
        self.alignment = EVRModule(9, 5)
        # Differential
        self.differential = EVRModule(32, 2)
        # Fusion
        self.fusion = EVRModule(32, 2)

        # hidden
        self.conv_hidden = nn.Conv2d(32, 3, 3, stride=1, padding=1)

        # upsample
        self.conv_up = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.Conv2d(8, 3, 3, stride=1, padding=1)
        )

    def forward(self, x, x_prev, y_prev):
        x_concat = torch.cat([x, x_prev, y_prev], dim=1)
        x_feature = self.conv_feature(x)
        x_align = self.alignment(x_concat)
        x = self.differential(x_feature - x_align)
        x = self.fusion(x_feature + x)
        x_hidden = self.conv_hidden(x)
        x = self.conv_up(x)

        return x, x_hidden


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EVRNet(scale=2).to(device)
    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    prev_input_data = (batch_size, 3, 1920, 1080)
    latent_input_date = (batch_size, 3, 1920, 1080)
    input_size = (input_data, prev_input_data, latent_input_date)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == "__main__":
    main()
