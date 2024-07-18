from typing import Union

import torch
import torchvision.transforms.functional as FV
import torch.nn as nn
from pytorch_msssim import ssim
from torch import Tensor


# Upscale
def upscale(lr_tensor: torch.Tensor, scale_factor: int, upscale_mode: str = 'bicubic') -> torch.Tensor:
    return nn.functional.interpolate(lr_tensor, scale_factor=scale_factor, mode=upscale_mode)


# Padding & Cropping
def pad_to_divisible(image_tensor: torch.Tensor, factor: int) -> torch.Tensor:
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
    def __init__(self, psnr_value: float = 0.0, ssim_value: float = 0.0, lpips_value: float = 0.0):
        self.psnr_value = round(psnr_value, 2)
        self.ssim_value = round(ssim_value, 2)
        self.lpips_value = round(lpips_value, 4)

    def __add__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError(f"Can't add type {other} to a Metric instance.")

        return Metrics(self.psnr_value + other.psnr_value, self.ssim_value + other.ssim_value, self.lpips_value + other.lpips_value)

    def __truediv__(self, divisor):
        if not isinstance(divisor, (int, float)):
            raise TypeError(f"Unsupported type for division: {type(divisor)}")

        return Metrics(self.psnr_value / divisor, self.ssim_value / divisor, self.lpips_value / divisor)

    def __eq__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError(f"Can't compare type {other} to a Metric instance.")

        return (self.psnr_value, self.ssim_value, self.lpips_value) == (other.psnr_value, other.ssim_value, other.lpips_value)

    def __str__(self):
        return (f"Metrics:\n"
                f"  PSNR: {self.psnr_value}\n"
                f"  SSIM: {self.ssim_value}\n"
                f"  LPIPS: {self.lpips_value}\n")


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


def calculate_lpips(img: torch.Tensor, label: torch.Tensor, lpips_model: nn.Module) -> float:
    lpips_t = lpips_model(img * 2 - 1, label * 2 - 1)
    lpips_score = lpips_t.item()
    return lpips_score


def calculate_metrics(img_tensor1: torch.Tensor, img_tensor2: torch.Tensor, lpips_model: nn.Module) -> Metrics:
    psnr_value = calculate_psnr(img_tensor1, img_tensor2)
    ssim_value = calculate_ssim(img_tensor1, img_tensor2)
    lpips_value = calculate_lpips(img_tensor1, img_tensor2, lpips_model)
    return Metrics(psnr_value, ssim_value, lpips_value)


def main() -> None:
    m1 = Metrics(0.7, 0.8)
    m2 = Metrics(0.1, 0.1)
    m3 = m1 + m2
    m4 = m3 / 2
    print(f"M1: {m1}\n M2: {m2}\n M3: {m3}\n M4: {m4}")


if __name__ == "__main__":
    main()
