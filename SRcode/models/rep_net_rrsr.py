from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


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


class RepNetRRSR(BaseModel):
    def __init__(self, scale: int, history_frames: int = 2, buffer_cha: int = 5, num_blocks=3, num_channels=64):
        super().__init__(scale=scale, down_and_up=1)
        self.downsample = nn.PixelUnshuffle(scale)
        layers = [RepBlock(3*4 + history_frames*3*4 + buffer_cha*4, num_channels)]
        for _ in range(num_blocks - 2):
            layers.append(RepBlock(num_channels, num_channels))
        layers.append(RepBlock(num_channels, 3 * scale * scale * scale * scale))
        self.body = nn.Sequential(*layers)
        self.upsample = nn.PixelShuffle(scale * scale)

    def forward(self, x, his, buf):
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bilinear")
        his = torch.cat(torch.unbind(his, 1), 1)
        x = torch.cat([x, his, buf], dim=1)
        x = self.downsample(x)
        x = self.body(x)
        x = self.upsample(x)
        x = x + x_up
        return x

    def reparameterize_all(self):
        # Iterate through the layers in the body and call reparameterize on each RepBlock
        for layer in self.body:
            if isinstance(layer, RepBlock):
                layer.reparameterize()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RepNetRRSR(scale=2, history_frames=2, buffer_cha=5).to(device)
    model.reparameterize_all()

    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    his = (batch_size, 2, 3, 1920, 1080)
    feature = (batch_size, 5, 1920, 1080)
    input_size = (input_data, his, feature)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == "__main__":
    main()
