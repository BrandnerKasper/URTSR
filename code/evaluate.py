from dataloader import CustomDataset
from torchvision import transforms
import torch
import torch.nn.functional as F
from model.srcnn import SRCNN
import numpy as np
from scipy.ndimage import convolve


def psnr(original, reconstructed, max_val=1.0):
    mse = F.mse_loss(original, reconstructed)
    psnr_value = 10 * torch.log10((max_val ** 2) / mse)
    return psnr_value.item()


def calculate_ssim(img, img2, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    # if input_order not in ['HWC', 'CHW']:<
    #     raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    # img = reorder_image(img, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)
    img = np.array(img, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)

    # img = img.astype(np.float64)
    # img2 = img2.astype(np.float64)

    # if test_y_channel:
    #     img = to_y_channel(img)
    #     img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _ssim(img1, img2, data_range=None, kernel_size=11, sigma=1.5):
    K1 = 0.01
    K2 = 0.03

    if data_range is None:
        data_range = np.max([np.max(img1), np.max(img2)])

    window = _create_gaussian_window(kernel_size, sigma)
    window = window / np.sum(window)

    mu1 = convolve(img1, window)
    mu2 = convolve(img2, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 * img1, window) - mu1_sq
    sigma2_sq = convolve(img2 * img2, window) - mu2_sq
    sigma12 = convolve(img1 * img2, window) - mu1_mu2

    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    return np.mean(ssim_map)


def _create_gaussian_window(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    window = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return window / np.sum(window)


def upscale(lr_tensor, scale_factor: int, upscale_mode: str = 'bicubic'):
    return F.interpolate(lr_tensor.unsqueeze(0), scale_factor=scale_factor, mode=upscale_mode).squeeze(0)


def main() -> None:
    # Loading and preparing data
    transform = transforms.ToTensor()
    evaluate_dataset = CustomDataset(root='dataset/evaluate', transform=transform)

    # Loading model
    model_path = "pretrained_models/srcnn_model_e10.pth"
    model = SRCNN()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_bilinear = (0.0, 0.0)
    total_net = (0.0, 0.0)
    total_bicubic = (0.0, 0.0)
    for lr_image, hr_image in evaluate_dataset:
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
