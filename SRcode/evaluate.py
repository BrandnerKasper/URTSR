import yaml
from torchvision import transforms
import torch
from utils import utils
import argparse

from data.dataloader import CustomDataset
from data.config import load_yaml_into_config, Config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SR network based on a pretrained model file.")
    parser.add_argument('file_path', type=str, nargs='?', default='pretrained_models/extranet_4.pth',
                        help="Path to the pretrained model .pth file")
    args = parser.parse_args()
    return args


def get_config_from_pretrained_model(name: str) -> Config:
    config_path = f"configs/{name}.yaml"
    return load_yaml_into_config(config_path)


def save_results(results: dict, name: str) -> None:
    yaml_text = yaml.dump(results, sort_keys=False)
    file = open(f"results/{name}.yaml", "w")
    file.write(yaml_text)
    file.close()


def evaluate(pretrained_model_path: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)
    with open(f"configs/{model_name}.yaml", "r") as file:
        results: dict = yaml.safe_load(file)

    # Datasets to evaluate:
    datasets = ["Set5", "Set14", "Urban100"]

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()

    for dataset in datasets:
        # Loading and preparing data
        dataset_path = f"dataset/{dataset}"
        evaluate_dataset = CustomDataset(root=dataset_path, pattern="")

        # Evaluate
        total = utils.Metrics()
        for i in range(len(evaluate_dataset)):
            filename = evaluate_dataset.get_filename(i)
            lr_image, hr_image = evaluate_dataset.__getitem__(i)
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)

            lr_image_model = utils.pad_to_divisible(lr_image.unsqueeze(0), 2 ** model.down_and_up)

            with torch.no_grad():
                output_image = model(lr_image_model).squeeze(0)
                output_image = utils.pad_or_crop_to_target(output_image, hr_image)
                output_image = torch.clamp(output_image, min=0.0, max=1.0)

            # Calc Metrics
            values = utils.calculate_metrics(hr_image, output_image)
            print(f"{filename}: {values}")

            # Calc total
            total += values

        # Calc average
        length = len(evaluate_dataset)
        average = total / length
        print("\n")
        print(f"Average {average} over dataset {dataset}")
        print("\n")

        # Generate result
        results[dataset] = {
            "PSNR": round(average.psnr, 2),
            "SSIM": round(average.ssim, 2),
        }

    save_results(results, model_name)


def evaluate_trad(config_path: str) -> None:
    # Setup
    name = config_path.split('/')[-1].split('.')[0]
    with open(config_path, "r") as file:
        results: dict = yaml.safe_load(file)

    # Datasets to evaluate:
    datasets = ["Set5", "Set14", "Urban100"]

    for dataset in datasets:
        # Loading and preparing data
        dataset_path = f"dataset/{dataset}"
        transform = transforms.ToTensor()
        evaluate_dataset = CustomDataset(root=dataset_path, transform=transform, pattern="")

        # Evaluate
        total = utils.Metrics()
        for i in range(len(evaluate_dataset)):
            filename = evaluate_dataset.get_filename(i)
            lr_image, hr_image = evaluate_dataset.__getitem__(i)

            lr_image_model = utils.pad_to_divisible(lr_image.unsqueeze(0), 2)

            with torch.no_grad():
                output_image = utils.upscale(lr_image_model, 2, upscale_mode=name).squeeze(0)
                output_image = utils.pad_or_crop_to_target(output_image, hr_image)
                output_image = torch.clamp(output_image, min=0.0, max=1.0)

            # Calc Metrics
            values = utils.calculate_metrics(hr_image, output_image)
            print(f"{filename}: {values}")

            # Calc total
            total += values

        # Calc average
        length = len(evaluate_dataset)
        average = total / length
        print("\n")
        print(f"Average {average} over dataset {dataset}")
        print("\n")

        # Save result
        results[dataset] = {
            "PSNR": round(average.psnr, 2),
            "SSIM": round(average.ssim, 2),
        }

    # Save to file
    yaml_text = yaml.dump(results, sort_keys=False)
    file = open(f"results/{name}.yaml", "w")
    file.write(yaml_text)
    file.close()


def main() -> None:
    args = parse_arguments()
    file_path = args.file_path
    evaluate(file_path)


if __name__ == '__main__':
    main()
