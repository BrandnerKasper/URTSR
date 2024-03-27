import torch
import torch.nn as nn

from .basemodel import BaseModel


class Flavr(BaseModel):
    def __init__(self, scale: int, frame_number: int = 4):
        super(Flavr, self).__init__(scale=scale, down_and_up=3)

        self.conv_in = nn.Sequential(
            nn.Conv3d(3*frame_number, 64)
        )

    def forward(self, x):
        pass
