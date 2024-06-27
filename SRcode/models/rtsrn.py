import torch
import torch.nn as nn

try:
    from basemodel import BaseModel
except ImportError:
    from models.basemodel import BaseModel


# https://github.com/eduardzamfir/NTIRE23-RTSR/tree/master
class RealTimeSRNet(BaseModel):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self, num_channels=3, num_feats=64, num_blocks=5, upscale=2) -> None:
        super(RealTimeSRNet, self).__init__(scale=upscale, down_and_up=0, do_two=False)

        self.head = nn.Sequential(
            nn.Conv2d(num_channels, num_feats, 3, padding=1)
        )

        body = []
        for i in range(num_blocks):
            body.append(nn.Conv2d(num_feats, num_feats, 3, padding=1))
            if i < num_blocks - 1:
                body.append(nn.ReLU(True))

        self.body = nn.Sequential(*body)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        res = self.head(x)
        out = self.body(res)
        out = self.upsample(res + out)
        return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RealTimeSRNet().to(device)
    batch_size = 1
    input_data = (batch_size, 3, 1920, 1080)
    input_size = input_data

    model.summary(input_size)
    model.measure_inference_time(input_size)
    model.measure_vram_usage(input_size)


if __name__ == "__main__":
    main()
