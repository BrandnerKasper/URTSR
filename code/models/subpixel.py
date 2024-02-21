import torch.nn as nn

from .basemodel import BaseModel


class SubPixelNN(BaseModel):
    def __init__(self, scale: int):
        super(SubPixelNN, self).__init__(scale=scale)

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU()
        )

        # Sub-pixel convolution
        self.sub_pixel_conv = nn.Sequential(
            nn.Conv2d(256, 3 * self.scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.sub_pixel_conv(x)
        return x
