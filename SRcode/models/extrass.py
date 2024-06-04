from typing import Optional

import torch
import torch.nn as nn

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


def space_to_depth(input_tensor, block_size):
    # Get the dimensions of the input tensor
    batch_size, channels, height, width = input_tensor.shape

    # Check that height and width are divisible by block_size
    assert height % block_size == 0, "Height must be divisible by block_size"
    assert width % block_size == 0, "Width must be divisible by block_size"

    # Calculate the new dimensions
    new_height = height // block_size
    new_width = width // block_size
    new_channels = channels * (block_size ** 2)

    # Rearrange the tensor
    output_tensor = input_tensor.reshape(batch_size, channels, new_height, block_size, new_width, block_size)
    output_tensor = output_tensor.permute(0, 1, 3, 5, 2, 4).reshape(batch_size, new_channels, new_height, new_width)

    return output_tensor


def depth_to_space(input_tensor, block_size):
    # Get the dimensions of the input tensor
    batch_size, channels, height, width = input_tensor.shape

    # Calculate the new dimensions
    new_channels = channels // (block_size ** 2)
    new_height = height * block_size
    new_width = width * block_size

    # Check that the channels are divisible by block_size^2
    assert channels % (block_size ** 2) == 0, "Channels must be divisible by block_size squared"

    # Rearrange the tensor
    output_tensor = input_tensor.reshape(batch_size, block_size, block_size, new_channels, height, width)
    output_tensor = output_tensor.permute(0, 3, 4, 1, 5, 2).reshape(batch_size, new_channels, new_height, new_width)

    return output_tensor


class FRNet(nn.Module):
    def __init__(self, history_cha: int = 3*3):
        super(FRNet, self).__init__()

        # Define the down-sampling blocks
        self.down1 = self.conv_block(history_cha, 22)
        self.down2 = self.conv_block(22, 24)
        self.down3 = self.conv_block(24, 36)
        self.down4 = self.conv_block(36, 48)

        # Define the up-sampling blocks
        self.up1 = self.up_conv_block(48, 36)
        self.up2 = self.up_conv_block(36, 24)
        self.up3 = self.up_conv_block(24, 22)
        self.up4 = self.up_conv_block(22, 5)

        # Define the blending layer
        self.blend = nn.Sequential(
            nn.Conv2d(48, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Down-sampling path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Blending the down-sampled features
        # blend_input = torch.cat((x4, x2_d4), dim=1)
        # x = self.blend(x4)

        # Up-sampling path
        x = self.up1(x4)
        x = self.up2(x + x3)
        x = self.up3(x + x2)
        x = self.up4(x + x1)

        return x


class ExtraSS(BaseModel):
    def __init__(self, scale: int, batch_size: int, crop_size: Optional[int] = None, buffer_cha: int = 9, history_cha: int = 3*3):
        super(ExtraSS, self).__init__(scale=scale, down_and_up=4)

        if crop_size is None:
            self.hr_input = torch.randn(batch_size, 3, 3840, 2176).to(device='cuda')
        else:
            self.hr_input = torch.randn(batch_size, 3, crop_size, crop_size).to(device='cuda')
        self.extra = False
        self.fr = FRNet(history_cha=history_cha)

        # Encoder for the SS forward pass
        self.ss_red_1 = nn.Conv2d(3 + buffer_cha, 22, kernel_size=3, stride=1, padding=1)
        self.ss_down_1 = nn.Sequential(
            nn.Conv2d(22, 22, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(22, 22, kernel_size=3, stride=2, padding=1)
        )
        self.ss_red_2 = nn.Conv2d(22, 32, kernel_size=3, stride=1, padding=1)
        self.ss_down_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        self.ss_red_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.ss_down_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        )
        self.ss_red_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.ss_down_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.ss_red_5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Encoder for the ESS forward pass
        self.ess_red_1 = nn.Conv2d(3 + buffer_cha + 12 + 5, 22, kernel_size=3, stride=1, padding=1)
        self.ess_down_1 = nn.Sequential(
            nn.Conv2d(22, 22, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(22, 22, kernel_size=3, stride=2, padding=1)
        )
        self.ess_red_2 = nn.Conv2d(22, 32, kernel_size=3, stride=1, padding=1)
        self.ess_down_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        self.ess_red_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.ess_down_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        )
        self.ess_red_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.ess_down_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.ess_red_5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Shared Decoder
        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )
        self.up_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        )
        self.up_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 22, kernel_size=2, stride=2)
        )

        self.finish_1 = nn.Sequential(
            nn.Conv2d(22, 46, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.finish_2 = nn.Sequential(
            nn.Conv2d(68, 12, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor, his: torch.Tensor):
        if torch.is_tensor(features):
            x = torch.cat([x, features], dim=1)
        # SS Encoder forward pass
        if not self.extra:
            x1 = self.ss_red_1(x)
            x2 = self.ss_down_1(x1)
            x2 = self.ss_red_2(x2)
            x3 = self.ss_down_2(x2)
            x3 = self.ss_red_3(x3)
            x4 = self.ss_down_3(x3)
            x4 = self.ss_red_4(x4)
            x5 = self.ss_down_4(x4)
            x5 = self.ss_red_5(x5)
        # ESS Encoder forward pass
        else:
            # Use SS frame as additonal input
            hr = space_to_depth(self.hr_input, 2)
            # Use FRNet as additional input
            his = torch.cat(torch.unbind(his, 1), 1)
            his = self.fr(his)
            x = torch.cat([x, hr, his], dim=1)
            x1 = self.ess_red_1(x)
            x2 = self.ess_down_1(x1)
            x2 = self.ess_red_2(x2)
            x3 = self.ess_down_2(x2)
            x3 = self.ess_red_3(x3)
            x4 = self.ess_down_3(x3)
            x4 = self.ess_red_4(x4)
            x5 = self.ess_down_4(x4)
            x5 = self.ess_red_5(x5)
        # Shared Decoder
        x = self.up_1(x5)
        x = self.up_2(x+x4)
        x = self.up_3(x+x3)
        x = self.up_4(x+x2)
        x = self.finish_1(x+x1)
        x = torch.cat([x, x1], dim=1)
        x = self.finish_2(x)
        x = depth_to_space(x, 2)
        # Safe the SS output as input for the ESS pass
        if not self.extra:
            self.hr_input = x
            self.extra = True
        else:
            self.extra = False
        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    input_data = (batch_size, 3, 1920, 1088)
    buffer = (batch_size, 16, 1920, 1088)
    his = (batch_size, 4, 3, 1920, 1088)
    input_size = (input_data, buffer, his)
    model = ExtraSS(scale=2, batch_size=1, buffer_cha=16, history_cha=4*3).to(device)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
