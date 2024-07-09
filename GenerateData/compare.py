import os

import cv2
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torchvision import transforms
from tqdm import tqdm


# Metrics
class Metrics:
    def __init__(self, psnr_value: float = 0.0, ssim_value: float = 0.0):
        self.psnr_value = round(psnr_value, 2)
        self.ssim_value = round(ssim_value, 2)

    def __add__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError(f"Can't add type {other} to a Metric instance.")

        return Metrics(self.psnr_value + other.psnr_value, self.ssim_value + other.ssim_value)

    def __truediv__(self, divisor):
        if not isinstance(divisor, (int, float)):
            raise TypeError(f"Unsupported type for division: {type(divisor)}")

        return Metrics(self.psnr_value / divisor, self.ssim_value / divisor)

    def __eq__(self, other):
        if not isinstance(other, Metrics):
            raise TypeError(f"Can't compare type {other} to a Metric instance.")

        return (self.psnr_value, self.ssim_value) == (other.psnr_value, other.ssim_value)

    def __str__(self):
        return (f"Metrics:\n"
                f"  PSNR: {self.psnr_value}\n"
                f"  SSIM: {self.ssim_value}\n")


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


# If we calculate the metrics for frame extrapolation we get a list of tensors!
def calculate_metrics(img_tensor1: torch.Tensor, img_tensor2: torch.Tensor) -> Metrics:
    psnr_value = calculate_psnr(img_tensor1, img_tensor2)
    ssim_value = calculate_ssim(img_tensor1, img_tensor2)
    return Metrics(psnr_value, ssim_value)


def compare_images(lr_path: str, hr_path: str) -> Metrics:
    transform = transforms.ToTensor()
    lr_frame = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
    lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
    lr_frame = transform(lr_frame)
    lr_frame = nn.functional.interpolate(lr_frame.unsqueeze(0), scale_factor=2, mode="bilinear").squeeze(0)

    hr_frame = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
    hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
    hr_frame = transform(hr_frame)

    return calculate_metrics(lr_frame, hr_frame)


def compare_sequence(lr_folder: str, hr_folder: str) -> Metrics:
    total_metrics = Metrics(0, 0)
    files = os.listdir(hr_folder)
    count = 0
    for file in tqdm(files, f"Comparing sequence {lr_folder} with {hr_folder}"):
        total_metrics += compare_images(f"{lr_folder}/{file}", f"{hr_folder}/{file}")
        count += 1
    return total_metrics / count


def main() -> None:
    # m1 = Metrics(0.3333, 0.8)
    # m2 = Metrics(0.1, 0.7676)
    # m3 = m1 + m2
    # m4 = m3 / 2
    # print(f"M1: {m1}\n M2: {m2}\n M3: {m3}\n M4: {m4}")

    # print(compare_images("LR_new/13/0073.png", "HR_new/13/0073.png"))
    print(compare_sequence("LR_new/04", "HR_new/04"))


if __name__ == "__main__":
    main()
