import torch
import torchvision.transforms.functional as FV
import torch.nn as nn
from pytorch_msssim import ssim
from torch import Tensor


# import lpips


# Upscale
def upscale(lr_tensor: torch.Tensor, scale_factor: int, upscale_mode: str = 'bicubic') -> torch.Tensor:
    return nn.functional.interpolate(lr_tensor, scale_factor=scale_factor, mode=upscale_mode).squeeze(0)


# Padding & Cropping
def pad_to_divisible(image_tensor: torch.Tensor, factor: int) -> Tensor:
    if factor == 0:
        return image_tensor
    _, _, height, width = image_tensor.size()
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    padded_image = FV.pad(image_tensor, [0, 0, pad_width, pad_height], padding_mode="edge")
    return padded_image


def pad_or_crop_to_target(input_t: torch.Tensor, target_t: torch.Tensor) -> Tensor:
    _, height, width = input_t.size()
    _, target_height, target_width = target_t.size()
    if height < target_height:
        pad_height = max(0, target_height - height)
        input_t = FV.pad(input_t, [0, 0, 0, pad_height], padding_mode="edge")
    else:
        input_t = input_t[:, :target_height, :]
    if width < target_width:
        pad_width = max(0, target_width - width)
        input_t = FV.pad(input_t, [0, 0, pad_width, 0], padding_mode="edge")
    else:
        input_t = input_t[:, :, :target_width]
    return input_t


# Metrics
class Metrics:
    def __init__(self, psnr_values: list[float], ssim_values: list[float]):
        self.psnr_values = psnr_values
        self.ssim_values = ssim_values
        self.average_psnr: float = 0
        self.average_ssim: float = 0
        self.calc_average()

    def calc_average(self):
        if len(self.psnr_values) == 0:
            self.average_psnr = 0
        else:
            self.average_psnr = sum(self.psnr_values) / len(self.psnr_values)
        if len(self.ssim_values) == 0:
            self.ssim_values = 0
        else:
            self.average_ssim = sum(self.ssim_values) / len(self.ssim_values)

    def __add__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError(f"Can't add type {other} to a Metric instance.")

        assert len(self.psnr_values) == len(other.psnr_values), \
            f"Metrics have different amount of psnr values self: {len(self.psnr_values)}, other: {len(other.psnr_values)}"
        psnr_values = []
        for i in range(len(other.psnr_values)):
            value = self.psnr_values[i] + other.psnr_values[i]
            value = round(value, 2)
            psnr_values.append(value)

        assert len(self.ssim_values) == len(other.ssim_values), \
            f"Metrics have different amount of ssim values self: {len(self.ssim_values)}, other: {len(other.ssim_values)}"
        ssim_values = []
        for i in range(len(other.ssim_values)):
            value = self.ssim_values[i] + other.ssim_values[i]
            value = round(value, 2)
            ssim_values.append(value)

        return Metrics(psnr_values, ssim_values)

    def __truediv__(self, divisor):
        if not isinstance(divisor, (int, float)):
            raise TypeError(f"Unsupported type for division: {type(divisor)}")
        psnr_values = []
        for value in self.psnr_values:
            psnr_values.append(value/divisor)
        ssim_values = []
        for value in self.ssim_values:
            ssim_values.append(value/divisor)

        return Metrics(psnr_values, ssim_values)

    def __eq__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError(f"Can't compare type {other} to a Metric instance.")
        return ((self.psnr_values, self.ssim_values, self.average_psnr, self.average_ssim) ==
                (other.psnr_values, other.ssim_values, other.average_psnr, other.average_ssim))

    def __str__(self):
        psnr_values_str = ", ".join(f"{val:.2f}" for val in self.psnr_values)
        ssim_values_str = ", ".join(f"{val:.2f}" for val in self.ssim_values)
        return (f"Metrics:\n"
                f"  PSNR Values: {psnr_values_str}\n"
                f"  SSIM Values: {ssim_values_str}\n"
                f"  Average PSNR: {self.average_psnr:.2f}\n"
                f"  Average SSIM: {self.average_ssim:.2f}\n")


def calculate_psnr(input_t: torch.Tensor, target_t: torch.Tensor, data_range: float = 1.0) -> float:
    mse = nn.functional.mse_loss(input_t, target_t)
    psnr_value = 10 * torch.log10((data_range ** 2) / mse)
    return psnr_value.item()


def calculate_ssim(img1_t: torch.Tensor, img2_t: torch.Tensor) -> float:
    """Calculate SSIM (structural similarity) for RGB images."""
    assert img1_t.shape == img2_t.shape, f'Image shapes are different: {img1_t.shape}, {img2_t.shape}.'

    # Ensure the tensors have the same dtype
    img1_t = img1_t.float()
    img2_t = img2_t.float()

    # Add a batch dimension to the tensors
    img1_t = img1_t.unsqueeze(0)
    img2_t = img2_t.unsqueeze(0)

    # Calculate SSIM for the entire RGB image
    ssim_value = ssim(img1_t, img2_t, data_range=1.0).item()

    return ssim_value


def calculate_metrics(img_tensor1: torch.Tensor, img_tensor2: torch.Tensor, mode: str) -> Metrics:
    match mode:
        case "single":  # Single Image Pair and only Spatial SR
            psnr_value = calculate_psnr(img_tensor1, img_tensor2)
            ssim_value = calculate_ssim(img_tensor1, img_tensor2)
            return Metrics([psnr_value], [ssim_value])
        case "multi":  # Multi Image Pair and Spatial + Temporal SR
            # img_tensor_list_1 = torch.unbind(img_tensor1, 1) # example tensor dim (8, 2, 3, 1920, 1080)
            # img_tensor_list_2 = torch.unbind(img_tensor2, 1)
            psnr_values, ssim_values = [], []
            for i in range(len(img_tensor1)):
                psnr_value = calculate_psnr(img_tensor1[i], img_tensor2[i])
                ssim_value = calculate_ssim(img_tensor1[i], img_tensor2[i])
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
            return Metrics(psnr_values, ssim_values)
        case _:
            raise Exception(f"mode {mode} is not supported!")


def main() -> None:
    m1 = Metrics([0.3, 0.7], [0.8, 0.6])
    m2 = Metrics([0.1, 0.2], [0.2, 0.1])
    m3 = m1 + m2
    m4 = m3 / 2
    print(f"M1: {m1}\n M2: {m2}\n M3: {m3}\n M4: {m4}")


if __name__ == "__main__":
    main()
