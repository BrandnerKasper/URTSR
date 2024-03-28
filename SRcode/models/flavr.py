import torch
import torch.nn as nn
from torchinfo import summary
import time

from basemodel import BaseModel


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


class Flavr(BaseModel):
    def __init__(self, scale: int, frame_number: int = 4):
        super(Flavr, self).__init__(scale=scale, down_and_up=3)

        self.conv_in = nn.Sequential(
            nn.Conv3d(frame_number, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_in = SEGating(16)

        self.down_1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_down_1 = SEGating(32)

        self.down_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_down_2 = SEGating(64)

        self.down_3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
        )
        self.gate_down_3 = SEGating(128)

        self.up_1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.gate_up_1 = SEGating(64)

        self.up_2 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.gate_up_2 = SEGating(32)

        self.up_3 = nn.Sequential(
            nn.ConvTranspose3d(64, 16, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.gate_up_3 = SEGating(16)

        self.up_4 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.gate_up_4 = SEGating(32)

        self.sub_pixel_conv_out = nn.Sequential(
            nn.ConvTranspose3d(32, int(frame_number/2), kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            # nn.PixelShuffle(self.scale),
        )

    def forward(self, x):
        x1 = self.conv_in(x)
        x1 = self.gate_in(x1)

        x2 = self.down_1(x1)
        x2 = self.gate_down_1(x2)

        x3 = self.down_2(x2)
        x3 = self.gate_down_2(x3)

        x = self.down_3(x3)
        x = self.gate_down_3(x)

        x = self.up_1(x)
        x = self.gate_up_1(x)

        x = self.up_2(torch.cat((x, x3), dim=1))
        x = self.gate_up_2(x)

        x = self.up_3(torch.cat((x, x2), dim=1))
        x = self.gate_up_3(x)

        x = self.up_4(torch.cat((x, x1), dim=1))
        x = self.gate_up_4(x)

        x = self.sub_pixel_conv_out(x)
        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Flavr(scale=2, frame_number=4).to(device)
    batch_size = 1
    input_size = (batch_size, 4, 3, 1920, 1080)

    # summary(model, input_size=input_size)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate dummy input
    input_data = torch.randn(input_size).to(device)

    # Warm-up GPU to ensure accurate timing
    torch.cuda.synchronize()
    start = time.time()

    # Run forward pass
    with torch.no_grad():
        output = model(input_data)

    # Measure forward pass time
    torch.cuda.synchronize()
    end = time.time()
    forward_pass_time = end - start

    # Measure VRAM usage
    vram_usage = torch.cuda.memory_allocated(device) / 1024 / 1024  # Convert bytes to MB

    print(f"Forward pass time: {forward_pass_time} seconds")
    print(f"VRAM usage: {vram_usage} MB")


if __name__ == '__main__':
    main()
