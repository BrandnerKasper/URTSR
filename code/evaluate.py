from dataloader import CustomDataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import utils

from models.srcnn import SRCNN
from models.subpixel import SubPixelNN
from models.extraNet import ExtraNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loading and preparing data
    transform = transforms.ToTensor()
    evaluate_dataset = CustomDataset(root='dataset/Set14', transform=transform, pattern="x2")

    # Loading models
    model_path = "pretrained_models/srcnn_e100_x2_bs4_cs256.pth"
    model = SRCNN(2).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_bilinear = (0.0, 0.0)
    total_net = (0.0, 0.0)
    total_bicubic = (0.0, 0.0)

    for i in range(len(evaluate_dataset)):
        filename = evaluate_dataset.get_filename(i)
        lr_image, hr_image = evaluate_dataset.__getitem__(i)
        lr_image, hr_image = lr_image.to(device), hr_image.to(device)

        lr_image_model = utils.pad_to_divisible(lr_image.unsqueeze(0), 2)
        lr_image_bi = utils.pad_to_divisible(lr_image.unsqueeze(0), 2)

        with torch.no_grad():
            output_image = model(lr_image_model).squeeze(0)
            output_image = utils.pad_or_crop_to_target(output_image, hr_image)
            output_image = torch.clamp(output_image, min=0.0, max=1.0)

        bilinear_image = utils.upscale(lr_image_bi, 2, "bilinear")
        bilinear_image = utils.pad_or_crop_to_target(bilinear_image, hr_image)

        bicubic_image = utils.upscale(lr_image_bi, 2, "bicubic")
        bicubic_image = utils.pad_or_crop_to_target(bicubic_image, hr_image)

        # Calc Metrics for BILINEAR, NET and BICUBIC
        bilinear_values = utils.calculate_metrics(hr_image, bilinear_image)
        net_values = utils.calculate_metrics(hr_image, output_image)
        bicubic_values = utils.calculate_metrics(hr_image, bicubic_image)
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
