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


class ExtraSS(BaseModel):
    def __init__(self, scale: int):
        super(ExtraSS, self).__init__(scale=scale)

        self.hr_input = torch.randn(1, 3, 3840, 2160) # abstract batch and crop size for this to work

        self.down_1 = nn.Sequential(
            nn.Conv2d(3 + 12, 22, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(22, 22, kernel_size=3, stride=2, padding=1)
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(22, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        )
        self.down_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        )
        self.down_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        )

        pass

    def forward(self, x):
        pass


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExtraSS(scale=2).to(device)
    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    feature = (batch_size, 9, 1920, 1080)
    his = (batch_size, 3, 2, 1920, 1080)
    input_size = (input_data, feature, his)

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
