import torch
import torch.nn as nn
try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


class SRCNN(BaseModel):
    def __init__(self, scale):
        super(SRCNN, self).__init__(scale=scale)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, stride=1) # k // 2
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x, **kwargs):
        # Upscaling my lr image by times 2 to have the same tensor size than the hr image
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode="bilinear")
        # Input shape: (batch_size, channels, height, width)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)
        # Output shape: (batch_size, 3, height, width)
        return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRCNN(scale=2).to(device)
    batch_size = 1
    input_size = (batch_size, 3, 1920, 1080)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
