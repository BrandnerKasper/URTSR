from dataloader import CustomDataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import utils
from pytorch_msssim import ssim

from model.srcnn import SRCNN
from model.subpixel import SubPixelNN
from model.extraNet import ExtraNet


def calculate_psnr(original, reconstructed, data_range=1.0):
    mse = F.mse_loss(original, reconstructed)
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


def upscale(lr_tensor, scale_factor: int, upscale_mode: str = 'bicubic'):
    return F.interpolate(lr_tensor, scale_factor=scale_factor, mode=upscale_mode).squeeze(0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loading and preparing data
    transform = transforms.ToTensor()
    evaluate_dataset = CustomDataset(root='dataset/Set14', transform=transform, pattern="x2")

    # Loading model
    model_path = "pretrained_models/extranet.pth"
    model = ExtraNet(2).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_bilinear = (0.0, 0.0)
    total_net = (0.0, 0.0)
    total_bicubic = (0.0, 0.0)

    for i in range(len(evaluate_dataset)):
        filename = evaluate_dataset.get_filename(i)
        lr_image, hr_image = evaluate_dataset.__getitem__(i)
        lr_image, hr_image = lr_image.to(device), hr_image.to(device)
        print(f"HR dim: {hr_image.size()}")

        lr_image_model = utils.pad_to_divisible(lr_image.unsqueeze(0), 8)
        lr_image_bi = utils.pad_to_divisible(lr_image.unsqueeze(0), 2)

        with torch.no_grad():
            output_image = model(lr_image_model).squeeze(0)
            output_image = utils.pad_or_crop_to_target(output_image, hr_image)
            output_image = torch.clamp(output_image, min=0.0, max=1.0)

        bilinear_image = upscale(lr_image_bi, 2, "bilinear")
        bilinear_image = utils.pad_or_crop_to_target(bilinear_image, hr_image)

        bicubic_image = upscale(lr_image_bi, 2, "bicubic")
        bicubic_image = utils.pad_or_crop_to_target(bicubic_image, hr_image)

        # Calc Metrics for BILINEAR, NET and BICUBIC
        bilinear_values = calculate_metrics(hr_image, bilinear_image)
        net_values = calculate_metrics(hr_image, output_image)
        bicubic_values = calculate_metrics(hr_image, bicubic_image)
        print(f"{filename}: "
              f"PSNR | bilinear {bilinear_values[0]:.2f} dB | network {net_values[0]:.2f} dB | bicubic {bicubic_values[0]:.2f} dB || "
              f"SSIM | bilinear {bilinear_values[1]:.2f} | network {net_values[1]:.2f} | bicubic {bicubic_values[1]:.2f}")

        # Calc total
        total_bilinear = tuple(x + y for x, y in zip(total_bilinear, bilinear_values))
        total_net = tuple(x + y for x, y in zip(total_net, net_values))
        total_bicubic = tuple(x + y for x, y in zip(total_bicubic, bicubic_values))

    # Calc average
    length = len(evaluate_dataset)
    average_bilinear = tuple(x / length for x in total_bilinear)
    average_net = tuple(x / length for x in total_net)
    average_bicubic = tuple(x / length for x in total_bicubic)
    print(f"Average (PSNR, SSIM) over dataset "
          f"| Bilinear: ({average_bilinear[0]:.2f} dB, {average_bilinear[1]:.2f})"
          f"| Net: ({average_net[0]:.2f} dB, {average_net[1]:.2f})"
          f"| Bicubic ({average_bicubic[0]:.2f} dB, {average_bicubic[1]:.2f})")


if __name__ == '__main__':
    main()
