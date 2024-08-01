import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


# this is just used for checking inference speed
class Bicubic(BaseModel):
    def __init__(self, scale: int = 2):
        super(Bicubic, self).__init__(scale=scale)

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Bicubic(scale=2).to(device)
    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    input_size = input_data

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == '__main__':
    main()
