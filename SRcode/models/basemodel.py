import torch
import torch.nn as nn
from torchinfo import summary


class BaseModel(nn.Module):
    def __init__(self, scale: int, down_and_up: int = 0):
        super(BaseModel, self).__init__()
        self.scale = scale
        self.down_and_up = down_and_up

    # for evaluation only
    def summary(self, input_size) -> None:
        summary(self, input_size)

    def measure_inference_time(self, input_size) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Move the model to GPU if available
        self.to(device)

        # eval mode
        self.eval()
        for k, v in self.named_parameters():
            v.requires_grad = False

        # Generate dummy input
        input_data = torch.randn(input_size).to(device)

        # GPU warmp up
        print("Warm up ...")
        with torch.no_grad():
            for _ in range(10):
                _ = self(input_data)

        print("Start timing ...")
        torch.cuda.synchronize()
        iterations = 10
        with torch.no_grad():
            total = 0
            for i in range(iterations):
                start.record()
                _ = self(input_data)
                end.record()
                torch.cuda.synchronize()
                total += start.elapsed_time(end)
        average = total / iterations
        print(f"Average forward pass time {average:.2f} ms")

    def measure_vram_usage(self, input_size) -> None:
        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # eval mode
        self.eval()
        for k, v in self.named_parameters():
            v.requires_grad = False

        # Generate dummy input
        input_data = torch.randn(input_size).to(device)

        with torch.no_grad():
            _ = self(input_data)
        print("Memory allocated (peak):", torch.cuda.max_memory_allocated() / 1024 ** 2, "MB")