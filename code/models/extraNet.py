import torch
import torch.nn as nn

from .basemodel import BaseModel


class ExtraNet(BaseModel):
    def __init__(self, scale: int):
        super(ExtraNet, self).__init__(scale=scale, down_and_up=3)

        self.conv_in = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1), # the original paper uses 18 as first in_channels!
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.down_1 = nn.Sequential(
            # downsampling at the beginning
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.down_2 = nn.Sequential(
            # downsampling at the beginning
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.down_3 = nn.Sequential(
            # downsampling at the beginning
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), # this is hacked since we do not yet use motion vectors from previous frames where we would get 32 more layers!
            nn.ReLU(inplace=True)
        )

        self.up_1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.after_up_1 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up_2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.after_up_2 = nn.Sequential(
            nn.Conv2d(56, 28, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.Conv2d(28, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.up_3 = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)

        self.after_up_3 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.sub_pixel_conv = nn.Sequential(
            nn.Conv2d(24, 3 * self.scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale)
        )
        # self.conv_out = nn.Conv2d(24, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_in(x)

        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        x = self.up_1(x4)
        x = self.after_up_1(torch.cat((x, x3), dim=1))
        x = self.up_2(x)
        x = self.after_up_2(torch.cat((x, x2), dim=1))
        x = self.up_3(x)
        x = self.after_up_3(torch.cat((x, x1), dim=1))

        x = self.sub_pixel_conv(x)
        return x
