from dataloader import CustomDataset
from torchvision import transforms
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from model.srcnn import SRCNN
from model.subpixel import SubPixelNN


def psnr(original, reconstructed, max_val=1.0):
    mse = F.mse_loss(original, reconstructed)
    psnr_value = 10 * torch.log10((max_val ** 2) / mse)
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


def upscale(lr_tensor, scale_factor: int, upscale_mode: str = 'bicubic'):
    return F.interpolate(lr_tensor.unsqueeze(0), scale_factor=scale_factor, mode=upscale_mode).squeeze(0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loading and preparing data
    transform = transforms.ToTensor()
    evaluate_dataset = CustomDataset(root='dataset/matrix', transform=transform)

    # Loading model
    model_path = "pretrained_models/subpnn_model_e100.pth"
    model = SubPixelNN(2).to(device)

    # input_size = (3, 1920, 1080)  # Assuming your input is a 3-channel image
    # print(f"Number of parameters: {summary(model, input_size=input_size, device='cuda')}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_bilinear = (0.0, 0.0)
    total_net = (0.0, 0.0)
    total_bicubic = (0.0, 0.0)
    for lr_image, hr_image in evaluate_dataset:
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)
        with torch.no_grad():
            output = model(lr_image.unsqueeze(0)).squeeze(0)
        bilinear = upscale(lr_image, 2, "bilinear")
        bicubic = upscale(lr_image, 2, "bicubic")

        # Calc PSNR for BILINEAR, NET and BICUBIC
        psnr_value_bilinear = psnr(hr_image, bilinear)
        psnr_value_net = psnr(hr_image, output)
        psnr_value_bicubic = psnr(hr_image, bicubic)
        print(f"PSNR "
              f"| Bilinear: {psnr_value_bilinear:.2f} dB "
              f"| Net: {psnr_value_net:.2f} dB "
              f"| Bicubic: {psnr_value_bicubic:.2f} dB")

        # Calc SSIM for BILINEAR, NET and BICUBIC
        ssim_value_bilinear = calculate_ssim(hr_image, bilinear)
        ssim_value_net = calculate_ssim(hr_image, output)
        ssim_value_bicubic = calculate_ssim(hr_image, bicubic)
        print(f"SSIM "
              f"| Bilinear: {ssim_value_bilinear:.2f} "
              f"| Net: {ssim_value_net:.2f} "
              f"| Bicubic: {ssim_value_bicubic:.2f}")

        # Calc total
        total_bilinear = (total_bilinear[0] + psnr_value_bilinear, total_bilinear[1] + ssim_value_bilinear)
        total_net = (total_net[0] + psnr_value_net, total_net[1] + ssim_value_net)
        total_bicubic = (total_bicubic[0] + psnr_value_bicubic, total_bicubic[1] + ssim_value_bicubic)

    # Calc average
    n = len(evaluate_dataset)
    average_bilinear = tuple(x/n for x in total_bilinear)
    average_net = tuple(x/n for x in total_net)
    average_bicubic = tuple(x/n for x in total_bicubic)
    print(f"Average (PSNR, SSIM) over dataset "
          f"| Bilinear: ({average_bilinear[0]:.2f} dB, {average_bilinear[1]:.2f})"
          f"| Net: ({average_net[0]:.2f} dB, {average_net[1]:.2f})"
          f"| Bicubic ({average_bicubic[0]:.2f} dB, {average_bicubic[1]:.2f})")


if __name__ == '__main__':
    main()
