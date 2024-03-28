import torch
from torchinfo import summary
from models.subpixel import SubPixelNN
from models.extraNet import ExtraNet
from models.flavr import Flavr


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Flavr(scale=2, frame_number=4).to(device)
    batch_size = 1
    input_size = (batch_size, 4, 3, 1920, 1080)

    summary(model, input_size=input_size)


if __name__ == '__main__':
    main()
