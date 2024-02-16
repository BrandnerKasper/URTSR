import yaml
from dataloader import CustomDataset
from torchvision import transforms
import torch
import utils

from config import load_yaml_into_config


def evaluate(config_path: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = load_yaml_into_config(config_path)
    print(config)
    with open(config_path, "r") as file:
        results: dict = yaml.safe_load(file)

    # Datasets to evaluate:
    datasets = ["Set5", "Set14", "Urban100"]

    # Loading model
    model = config.model.to(device)
    file = config.filename
    model_path = f"pretrained_models/{file}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for dataset in datasets:
        # Loading and preparing data
        dataset_path = f"dataset/{dataset}"
        transform = transforms.ToTensor()
        evaluate_dataset = CustomDataset(root=dataset_path, transform=transform, pattern="x2")

        # Evaluate
        total = utils.Metrics()
        for i in range(len(evaluate_dataset)):
            filename = evaluate_dataset.get_filename(i)
            lr_image, hr_image = evaluate_dataset.__getitem__(i)
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)

            lr_image_model = utils.pad_to_divisible(lr_image.unsqueeze(0), 8)

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

        # Save result
        results[dataset] = {
            "PSNR": round(average.psnr, 2),
            "SSIM": round(average.ssim, 2),
        }

    # Save to file
    yaml_text = yaml.dump(results, sort_keys=False)
    file = open(f"results/{file}.yaml", "w")
    file.write(yaml_text)
    file.close()


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
        evaluate_dataset = CustomDataset(root=dataset_path, transform=transform, pattern="x2")

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
    config_path = "configs/extranet.yaml"
    evaluate(config_path)


if __name__ == '__main__':
    main()
