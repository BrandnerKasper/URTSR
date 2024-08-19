from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_cha, out_cha):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_cha, out_cha, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_cha, out_cha, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        return x


class DownConvBlock(nn.Module):
    def __init__(self, in_cha, out_cha):
        super().__init__()
        self.down_sample = nn.Conv2d(in_cha, in_cha, kernel_size=3, stride=2, padding=1)
        self.conv_1 = nn.Conv2d(in_cha, out_cha, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_cha, out_cha, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        return x


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


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepBlock, self).__init__()
        # Define the layers used during training
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Identity layer (skip connection)
        self.identity = nn.Identity() if in_channels == out_channels else None

        # Flag to check if the block has been reparameterized
        self.reparam_mode = False

    def forward(self, x):
        if self.reparam_mode:
            # During inference, use the reparameterized convolution
            return F.relu(self.reparam_conv(x))
        else:
            # During training, use the combination of convolutions and identity
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.identity is not None:
                out += self.identity(x)
            out = self.bn(out)
            return F.relu(out)

    def reparameterize(self):
        kernel3x3 = self.conv3x3.weight.data
        kernel1x1 = self.conv1x1.weight.data

        if self.identity is not None:
            # Create an identity kernel for the 3x3 convolution
            identity_kernel = torch.zeros_like(kernel3x3)
            for i in range(identity_kernel.size(0)):
                identity_kernel[i, i, 1, 1] = 1.0
        else:
            identity_kernel = 0

        # Combine kernels
        combined_kernel = kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1]) + identity_kernel
        bias = self.bn.bias.data - self.bn.running_mean.data / torch.sqrt(self.bn.running_var.data + self.bn.eps)
        weight = combined_kernel * (self.bn.weight.data / torch.sqrt(self.bn.running_var.data + self.bn.eps)).view(-1,
                                                                                                                   1, 1,
                                                                                                                   1)

        # Create the reparameterized convolution
        self.reparam_conv = nn.Conv2d(in_channels=self.conv3x3.in_channels,
                                      out_channels=self.conv3x3.out_channels,
                                      kernel_size=3, padding=1, bias=True)

        # Set the weights and bias
        self.reparam_conv.weight.data = weight
        self.reparam_conv.bias.data = bias

        # Remove the original layers and set the flag
        del self.conv3x3, self.conv1x1, self.bn, self.identity
        self.reparam_mode = True


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


class URepSS(BaseModel):
    def __init__(self, scale: int = 2, history_frames: int = 2, buffer_cha: int = 13, num_blocks=3, num_channels: int = 64):
        super(URepSS, self).__init__(scale=scale, down_and_up=3)

        # Encoder
        self.down_sample = nn.PixelUnshuffle(scale)
        self.conv_in = ConvBlock((3 + buffer_cha)*4, 24)
        self.down_1 = DownConvBlock(24, 24)
        self.down_2 = DownConvBlock(24, 32)

        # History encoder
        self.history_encoder = nn.Sequential(
            nn.PixelUnshuffle(scale),
            GatedConvBlock((history_frames * 3) * 4, 24),
            GatedDownConvBlock(24, 24),
            GatedDownConvBlock(24, 32)
        )

        # Bottom layer with multiple rep blocks
        # layers = [RepBlock(64, num_channels)]
        # for _ in range(num_blocks - 2):
        #     layers.append(RepBlock(num_channels, num_channels))
        # layers.append(RepBlock(num_channels, 64))
        # self.bottom_layer = nn.Sequential(*layers)

        # Decoder
        self.up_1 = UpConvBlock(64 + 24, 32)
        self.up_2 = UpConvBlock(32 + 24, 24)
        self.conv_out = ConvBlock(24, 48)
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
        # x3 = self.bottom_layer(x3)

        # Decoder
        x = self.up_1(x3, x2)
        x = self.up_2(x, x1)
        x = self.conv_out(x)

        x = self.up_sample(x)
        x = x + x_up

        return x

    # def restructure_bottom_layer(self):
    #     for layer in self.bottom_layer:
    #         if isinstance(layer, RepBlock):
    #             layer.reparameterize()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = URepSS(scale=2, history_frames=2, buffer_cha=5, num_blocks=6, num_channels=64).to(device)
    # model.restructure_bottom_layer()

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
