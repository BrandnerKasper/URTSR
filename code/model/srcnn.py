import torch.nn as nn
import torch.nn.functional


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, stride=1) # k // 2
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        # Upscaling my lr image by times 2 to have the same tensor size than the hr image
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        # Input shape: (batch_size, channels, height, width)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)
        # Output shape: (batch_size, 3, height, width)
        return x

