import yaml
from torchvision import transforms
import torch
from tqdm import tqdm

from utils import utils
import argparse
from torch.utils.data import DataLoader

from data.dataloader import SingleImagePair, MultiImagePair, STSSCrossValidation2, STSSImagePair, DiskMode
from config import load_yaml_into_config, Config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SR network based on a pretrained model file.")
    parser.add_argument('file_path', type=str, nargs='?', default='pretrained_models/extraSS_All.pth',
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


def init_dataset(name: str, extra: bool, history: int, buffers: dict[str, bool]) -> STSSImagePair:
    match name:
        case "ue_data_npz":
            return STSSImagePair(root=f"dataset/ue_data_npz/val", scale=2, extra=extra, history=history,
                                 buffers=buffers, last_frame_idx=299, crop_size=None,
                                 use_hflip=False, use_rotation=False, digits=4, disk_mode=DiskMode.NPZ)
        case _:
            raise ValueError(f"The dataset '{name}' is not a valid dataset.")


def evaluate(pretrained_model_path: str) -> None:
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
    extra = config.extra

    val_dataset = init_dataset(config.dataset, config.extra, config.history, config.buffers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=config.number_workers)
    total_ss_metrics = utils.Metrics([0], [0])
    total_ess_metrics = utils.Metrics([0], [0])
    for ss, ess in tqdm(val_loader, desc=f"Evaluating on {config.dataset}", dynamic_ncols=True):
        # prepare data
        # SS
        lr_image = ss[0].to(device)  # shared
        lr_image = utils.pad_to_divisible(lr_image, 2 ** model.down_and_up)
        ss_feature_images = [img.to(device) for img in ss[1]]
        if ss_feature_images:
            ss_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ss_feature_images]
            ss_feature_images = torch.cat(ss_feature_images, dim=1)
        history_images = [img.to(device) for img in ss[2]]
        if history_images:
            history_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in history_images]
            history_images = torch.stack(history_images, dim=2)  # shared
        ss_hr_image = ss[3].to(device)
        ss_hr_image = utils.pad_to_divisible(ss_hr_image, 2 ** model.down_and_up * config.scale)
        # ESS
        if extra:
            ess_feature_images = [img.to(device) for img in ess[1]]
            if ess_feature_images:
                ess_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ess_feature_images]
                ess_feature_images = torch.cat(ess_feature_images, dim=1)
            ess_hr_image = ess[3].to(device)
            ess_hr_image = utils.pad_to_divisible(ess_hr_image, 2 ** model.down_and_up * config.scale)

        with torch.no_grad():
            # forward pass
            # depending on the network we perform two forward passes or only one
            if model.do_two:
                # SS
                ss_output = model(lr_image, ss_feature_images, history_images)
                ss_output = torch.clamp(ss_output, min=0.0, max=1.0)
                # ESS
                if extra:
                    ess_output = model(lr_image, ess_feature_images, history_images)
                    ess_output = torch.clamp(ess_output, min=0.0, max=1.0)
            else:
                if extra:
                    if ss_feature_images and ess_feature_images:
                        feature_images = torch.cat([ss_feature_images, ess_feature_images], 1)
                    else:
                        feature_images = []
                else:
                    feature_images = ss_feature_images
                if extra:
                    ss_output, ess_output = model(lr_image, feature_images, history_images)
                    ss_output = torch.clamp(ss_output, min=0.0, max=1.0)
                    ess_output = torch.clamp(ess_output, min=0.0, max=1.0)
                else:
                    ss_output = model(lr_image, feature_images, history_images)
                    ss_output = torch.clamp(ss_output, min=0.0, max=1.0)

        # Calc PSNR and SSIM
        # SS frame
        ss_metric = utils.calculate_metrics(ss_hr_image, ss_output, "single")
        total_ss_metrics += ss_metric
        if extra:
            # ESS frame
            ess_metric = utils.calculate_metrics(ess_hr_image, ess_output, "single")
            total_ess_metrics += ess_metric

    # PSNR & SSIM
    average_ss_metric = total_ss_metrics / len(val_loader)
    if extra:
        average_ess_metric = total_ess_metrics / len(val_loader)
        average_metric = (average_ss_metric + average_ess_metric) / 2
    else:
        average_metric = average_ss_metric
    print("\n")
    if extra:
        print(f"SS {average_ss_metric}")
        print(f"ESS {average_ess_metric}")
    print(f"Total {average_metric}")
    # Write results
    # TODO: abstract number of values based on number of frames
    if extra:
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
    else:
        results["PSNR"] = {
            "Frame_0": round(average_metric.average_psnr, 2)
        }
        results["SSIM"] = {
            "Frame_0": round(average_metric.average_ssim, 2)
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
