import torch
import torchvision.transforms.functional as FV
import torch.nn as nn
from pytorch_msssim import ssim
# import lpips


# Upscale
def upscale(lr_tensor, scale_factor: int, upscale_mode: str = 'bicubic'):
    return nn.functional.interpolate(lr_tensor, scale_factor=scale_factor, mode=upscale_mode).squeeze(0)


# Padding
def pad_to_divisible(image_tensor, factor):
    _, _, height, width = image_tensor.size()
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    padded_image = FV.pad(image_tensor, [0, 0, pad_width, pad_height], padding_mode="edge")
    return padded_image


def pad_or_crop_to_target(image_tensor, target_tensor):
    _, height, width = image_tensor.size()
    _, target_height, target_width = target_tensor.size()
    if height < target_height:
        pad_height = max(0, target_height - height)
        image_tensor = FV.pad(image_tensor, [0, 0, 0, pad_height], padding_mode="edge")
    else:
        image_tensor = image_tensor[:, :target_height, :]
    if width < target_width:
        pad_width = max(0, target_width - width)
        image_tensor = FV.pad(image_tensor, [0, 0, pad_width, 0], padding_mode="edge")
    else:
        image_tensor = image_tensor[:, :, :target_width]
    return image_tensor


# Metrics
def calculate_psnr(original, reconstructed, data_range=1.0):
    mse = nn.functional.mse_loss(original, reconstructed)
    psnr_value = 10 * torch.log10((data_range ** 2) / mse)
    return psnr_value.item()


def calculate_ssim(img_tensor, img2_tensor):
    """Calculate SSIM (structural similarity) for RGB images."""
    assert img_tensor.shape == img2_tensor.shape, f'Image shapes are different: {img_tensor.shape}, {img2_tensor.shape}.'

    # Ensure the tensors have the same dtype
    img_tensor = img_tensor.float()
    img2_tensor = img2_tensor.float()

    # Add a batch dimension to the tensors
    img_tensor = img_tensor.unsqueeze(0)
    img2_tensor = img2_tensor.unsqueeze(0)

    # Calculate SSIM for the entire RGB image
    ssim_value = ssim(img_tensor, img2_tensor, data_range=1.0).item()

    return ssim_value


def calculate_metrics(img1, img2):
    psnr_value = calculate_psnr(img1, img2)
    ssim_value = calculate_ssim(img1, img2)
    return psnr_value, ssim_value
