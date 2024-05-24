import yaml
from torchvision import transforms
import torch
from tqdm import tqdm

from utils import utils
import argparse
from torch.utils.data import DataLoader

from data.dataloader import SingleImagePair, MultiImagePair
from config import load_yaml_into_config, Config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SR network based on a pretrained model file.")
    parser.add_argument('file_path', type=str, nargs='?', default='pretrained_models/stss.pth',
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
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)

    # decide if Single Image or Multi Image Pair
    if isinstance(config.val_dataset, SingleImagePair):
        evaluate_single_image_dataset(pretrained_model_path)
    elif isinstance(config.val_dataset, MultiImagePair):
        evaluate_multi_image_dataset(pretrained_model_path)
    else:
        evaluate_stss_image_dataset(pretrained_model_path)


def evaluate_single_image_dataset(pretrained_model_path: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)
    with open(f"configs/{model_name}.yaml", "r") as file:
        results: dict = yaml.safe_load(file)

    # Datasets to evaluate:
    datasets = ["Set5", "Set14", "Urban100"]  # this is additional for SISR!

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()

    for dataset in datasets:
        # Loading and preparing data
        dataset_path = f"dataset/{dataset}"
        evaluate_dataset = SingleImagePair(root=dataset_path, pattern="")

        # Evaluate
        total = utils.Metrics([], [])
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
            "PSNR": round(average.average_psnr, 2),
            "SSIM": round(average.average_ssim, 2),
        }

    save_results(results, model_name)


def evaluate_multi_image_dataset(pretrained_model_path: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)

    with open(f"configs/{model_name}.yaml", "r") as file:
        results: dict = yaml.safe_load(file)

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()

    val_loader = DataLoader(dataset=config.val_dataset, batch_size=1, shuffle=False, num_workers=config.number_workers)

    total_metrics = utils.Metrics([0, 0], [0, 0])  # TODO: abstract number of values based on second dim of tensor [8, 2, 3, 1920, 1080]

    for lr_image, hr_image in tqdm(val_loader, desc=f"Evaluating on {config.dataset}", dynamic_ncols=True):
        lr_image = [img.to(device) for img in lr_image]
        hr_image = [img.to(device) for img in hr_image]
        lr_image = torch.stack(lr_image, dim=2)
        with torch.no_grad():
            output_image = model(lr_image)
            output_image = [torch.clamp(img, min=0.0, max=1.0) for img in output_image]
        # Calc PSNR and SSIM
        metrics = utils.calculate_metrics(hr_image, output_image, "multi")
        total_metrics += metrics

    # PSNR & SSIM
    average_metric = total_metrics / len(val_loader)
    print("\n")
    print(average_metric)

    # Write results
    # TODO: abstract number of values based on number of frames
    results["PSNR"] = {
        "Frame_0": round(average_metric.psnr_values[0], 2),
        "Frame_1": round(average_metric.psnr_values[1], 2),
        "Average": round(average_metric.average_psnr, 2)
    }
    results["SSIM"] = {
        "Frame_0": round(average_metric.ssim_values[0], 2),
        "Frame_1": round(average_metric.ssim_values[1], 2),
        "Average": round(average_metric.average_ssim, 2)
    }

    # Save results
    save_results(results, model_name)


def evaluate_stss_image_dataset(pretrained_model_path: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)

    with open(f"configs/{model_name}.yaml", "r") as file:
        results: dict = yaml.safe_load(file)

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()

    val_loader = DataLoader(dataset=config.val_dataset, batch_size=1, shuffle=False, num_workers=config.number_workers)
    total_ss_metrics = utils.Metrics([0], [0])
    total_ess_metrics = utils.Metrics([0], [0])
    for ss, ess in tqdm(val_loader, desc=f"Evaluating on {config.dataset}", dynamic_ncols=True):
        # forward pass for SS
        lr_image = ss[0].to(device)  # shared
        ss_feature_images = [img.to(device) for img in ss[1]]
        ss_feature_images = torch.cat(ss_feature_images, dim=1)
        history_images = [img.to(device) for img in ss[2]]
        history_images = torch.stack(history_images, dim=2)  # shared
        ss_hr_image = ss[3].to(device)

        # forward pass for ESS
        ess_feature_images = [img.to(device) for img in ess[1]]
        ess_feature_images = torch.cat(ess_feature_images, dim=1)
        ess_hr_image = ess[3].to(device)

        with torch.no_grad():
            # SS frame
            ss_output = model(lr_image, ss_feature_images, history_images)
            ss_output = torch.clamp(ss_output, min=0.0, max=1.0)
            # ESS frame
            ess_output = model(lr_image, ess_feature_images, history_images)
            ess_output = torch.clamp(ess_output, min=0.0, max=1.0)

        # Calc PSNR and SSIM
        # SS frame
        ss_metric = utils.calculate_metrics(ss_hr_image, ss_output, "single")
        total_ss_metrics += ss_metric
        # ESS frame
        ess_metric = utils.calculate_metrics(ess_hr_image, ess_output, "single")
        total_ess_metrics += ess_metric

    # PSNR & SSIM
    average_ss_metric = total_ss_metrics / len(val_loader)
    average_ess_metric = total_ess_metrics / len(val_loader)
    average_metric = (average_ss_metric + average_ess_metric) / 2
    print("\n")
    print(f"SS {average_ss_metric}")
    print(f"ESS {average_ess_metric}")
    print(f"Total {average_metric}")
    # Write results
    # TODO: abstract number of values based on number of frames
    results["PSNR"] = {
        "Frame_0": round(average_ss_metric.average_psnr, 2),
        "Frame_1": round(average_ess_metric.average_psnr, 2),
        "Average": round(average_metric.average_psnr, 2)
    }
    results["SSIM"] = {
        "Frame_0": round(average_ss_metric.average_ssim, 2),
        "Frame_1": round(average_ess_metric.average_ssim, 2),
        "Average": round(average_metric.average_ssim, 2)
    }
    # Save results
    save_results(results, model_name)


# TODO: Rework
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
        evaluate_dataset = SingleImagePair(root=dataset_path, transform=transform, pattern="")

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
