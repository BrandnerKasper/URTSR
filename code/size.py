import torch
from torchinfo import summary

from model.subpixel import SubPixelNN
from model.extraNet import ExtraNet


def track_vram_usage(model, input_tensor):
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()

    with torch.no_grad():
        output = model(input_tensor)

    torch.cuda.synchronize()

    vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to megabytes
    print(f"VRAM usage during forward pass: {vram_usage:.2f} MB")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExtraNet(2).to(device)
    batch_size = 1
    input_size = (batch_size, 3, 1920, 1080)

    summary(model, input_size=input_size)

    input_tensor = torch.randn(*input_size).to(device)
    track_vram_usage(model, input_tensor)


if __name__ == '__main__':
    main()
