import torch
from torchinfo import summary
from models.subpixel import SubPixelNN
from models.extraNet import ExtraNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExtraNet(2).to(device)
    batch_size = 1
    input_size = (batch_size, 3, 1920, 1080)

    summary(model, input_size=input_size)


if __name__ == '__main__':
    main()
