import torch
from torchinfo import summary

from model.subpixel import SubPixelNN


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SubPixelNN(2).to(device)
    batch_size = 1
    summary(model, input_size=(batch_size, 3, 1920, 1080))


if __name__ == '__main__':
    main()
