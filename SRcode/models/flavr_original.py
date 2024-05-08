import torch
import torch.nn as nn

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


class SEGating(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.pool(x)
        mask = self.attn_layer(mask)
        return x * mask


class Flavr_Original(BaseModel):
    def __init__(self, scale: int, frame_number: int = 4):
        super(Flavr_Original, self).__init__(scale=scale, down_and_up=3)

        self.conv_in = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_in = SEGating(64)

        self.down_1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_down_1 = SEGating(128)

        self.down_2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_down_2 = SEGating(256)

        self.down_3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_down_3 = SEGating(512)

        self.bottom = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_bottom = SEGating(512)

        self.up_1 = nn.Sequential(
            nn.ConvTranspose3d(1024, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.gate_up_1 = SEGating(256)

        self.up_2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.gate_up_2 = SEGating(128)

        self.up_3 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.gate_up_3 = SEGating(64)

        self.top = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.gate_top = SEGating(64)

        self.feature_fuse = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2, True)
        )

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3 * int(frame_number/2) * self.scale ** 2, kernel_size=7, stride=1, padding=0),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # x = torch.stack(x, dim=2)
        x1 = self.conv_in(x)
        x1 = self.gate_in(x1)

        x2 = self.down_1(x1)
        x2 = self.gate_down_1(x2)

        x3 = self.down_2(x2)
        x3 = self.gate_down_2(x3)

        x4 = self.down_3(x3)
        x4 = self.gate_down_3(x4)

        x = self.bottom(x4)
        x = self.gate_bottom(x)

        x = self.up_1(torch.cat((x, x4), dim=1))
        x = self.gate_up_1(x)

        x = self.up_2(torch.cat((x, x3), dim=1))
        x = self.gate_up_2(x)

        x = self.up_3(torch.cat((x, x2), dim=1))
        x = self.gate_up_3(x)

        x = self.top(torch.cat((x, x1), dim=1))
        x = self.gate_top(x)

        x = self.feature_fuse(torch.cat(torch.unbind(x, 2), 1)) # we unbind the depth dimension and stack the rgb layers
        x = self.outconv(x)

        x = torch.split(x, dim=1, split_size_or_sections=3)

        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Flavr_Original(scale=2, frame_number=4).to(device)
    batch_size = 1
    input_size = (batch_size, 3, 4, 1920, 1080)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
