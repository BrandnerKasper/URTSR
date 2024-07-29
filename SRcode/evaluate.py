import yaml
from lpips import lpips
from torchvision import transforms
import torch
from tqdm import tqdm

from utils import utils
import argparse
from torch.cuda import amp
from torch.utils.data import DataLoader

from data.dataloader import SingleImagePair, MultiImagePair, STSSImagePair, DiskMode, VSR, SISR, \
    SimpleSTSS
from config import load_yaml_into_config, Config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SR network based on a pretrained model file.")
    parser.add_argument('file_path', type=str, nargs='?', default='pretrained_models/URTSR/urtsr_01.pth',
                        help="Path to the pretrained model .pth file")
    args = parser.parse_args()
    return args


def get_config_from_pretrained_model(subfolder: str, name: str) -> Config:
    config_path = f"configs/{subfolder}/{name}.yaml"
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


def evaluate_trad() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loading and preparing data
    dataset_path = "dataset/UE_data/val"
    upscale_mode = "bicubic"
    scale = 2
    eval_alex_model = lpips.LPIPS(net='alex').cuda()

    eval_dataset = SISR(root=dataset_path, scale=scale)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, num_workers=8)
    sequence_length = eval_dataset.sequence_length
    sequence_names = eval_dataset.sequence_names
    metrics = {}

    total_metric = utils.Metrics()
    count = 0
    sequences = 0
    for lr, hr in tqdm(eval_loader, desc=f"Eval on ue data", dynamic_ncols=True):
        lr = lr.to(device)
        hr = hr.to(device).squeeze(0)
        if upscale_mode == "bilinear":
            res = utils.upscale(lr, scale, upscale_mode).squeeze(0)
        else:
            res = utils.upscale(lr, scale, upscale_mode).squeeze(0).squeeze(0)
        metric = utils.calculate_metrics(hr, torch.clamp(res, min=0.0, max=1.0), eval_alex_model)
        total_metric += metric
        if count == sequence_length - 1:
            metrics[sequence_names[sequences]] = total_metric / sequence_length
            total_metric = utils.Metrics()
            count = 0
            sequences += 1
        else:
            count += 1

    # Printing
    for k, v in metrics.items():
        print(f"Sequence {k}: {v}")


def eval_trad_stss_simple():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "//media/tobiasbrandner/Data/STSS/Lewis/test"
    upscale_mode = "bilinear"
    scale = 2
    eval_alex_model = lpips.LPIPS(net='alex').cuda()

    eval_dataset = SimpleSTSS(root=path, scale=scale)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, num_workers=8)
    total_metric = utils.Metrics()

    for lr, hr in tqdm(eval_loader, desc=f"Eval on stss data", dynamic_ncols=True):
        lr = lr.to(device)
        hr = hr.to(device).squeeze(0)
        if upscale_mode == "bilinear":
            res = utils.upscale(lr, scale, upscale_mode).squeeze(0)
        else:
            res = utils.upscale(lr, scale, upscale_mode).squeeze(0).squeeze(0)
        metric = utils.calculate_metrics(hr, torch.clamp(res, min=0.0, max=1.0), eval_alex_model)
        total_metric += metric

    # Printing
    total_metric = total_metric / len(eval_loader)
    print(total_metric)


def evaluate_vsr(pretrained_model_path: str) -> None:
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    sub_folder = pretrained_model_path.split('/')[1]
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(sub_folder, model_name)
    print(config)

    with open(f"configs/{sub_folder}/{model_name}.yaml", "r") as file:
        results: dict = yaml.safe_load(file)

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    extra = config.extra

    val_dataset = config.val_dataset
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=config.number_workers)
    eval_alex_model = lpips.LPIPS(net='alex').cuda()

    total_metrics = utils.Metrics(0, 0, 0)

    for lr_image, history_images, hr_image in tqdm(val_loader, desc=f"Validation", dynamic_ncols=True):
        # prepare data
        lr_image = lr_image.to(device)
        history_images = [img.to(device) for img in history_images]
        history_images = torch.stack(history_images, dim=1)
        hr_image = hr_image.to(device)

        with torch.no_grad():
            # forward pass
            with amp.autocast():
                output = model(lr_image, history_images)
            output = torch.clamp(output, min=0.0, max=1.0)

        # Calc PSNR and SSIM
        # SS frame
        metric = utils.calculate_metrics(hr_image.squeeze(0), output.squeeze(0), eval_alex_model)
        total_metrics += metric

    # PSNR & SSIM
    average_metric = total_metrics / len(val_loader)
    print("\n")
    print(f"Total {average_metric}")

    # writing results
    results["VAL SEQUENCE"] = config.sequence + 6
    results["PSNR"] = average_metric.psnr_value
    results["SSIM"] = average_metric.ssim_value
    results["LPIPS"] = average_metric.lpips_value

    # Save results
    save_results(results, model_name)


def main() -> None:
    args = parse_arguments()
    file_path = args.file_path
    evaluate_vsr(file_path)
    # evaluate_trad()
    # eval_trad_stss_simple()


if __name__ == '__main__':
    main()
