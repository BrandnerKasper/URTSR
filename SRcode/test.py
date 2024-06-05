import os
from torchvision import transforms
import torch
from tqdm import tqdm
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import argparse

from data.dataloader import SingleImagePair, MultiImagePair, STSSImagePair, DiskMode, STSSCrossValidation, \
    STSSCrossValidation2
from config import load_yaml_into_config, Config
from utils import utils


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SR network based on a pretrained model file.")
    parser.add_argument('file_path', type=str, nargs='?', default='pretrained_models/extraSS_All.pth',
                        help="Path to the pretrained model .pth file")
    args = parser.parse_args()
    return args


def get_config_from_pretrained_model(name: str) -> Config:
    config_path = f"configs/{name}.yaml"
    return load_yaml_into_config(config_path)


def generate_directory(path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)


def init_dataset(name: str, extra: bool, history: int, buffers: dict[str, bool]) -> STSSImagePair:
    match name:
        case "ue_data_npz":
            return STSSImagePair(root=f"dataset/ue_data_npz/test", scale=2, extra=extra, history=history,
                                 buffers=buffers, last_frame_idx=299, crop_size=None,
                                 use_hflip=False, use_rotation=False, digits=4, disk_mode=DiskMode.NPZ)
        case _:
            raise ValueError(f"The dataset '{name}' is not a valid dataset.")


def test(model_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)

    save_path = f"results/{model_name}"

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    extra = config.extra

    test_dataset = init_dataset(config.dataset, config.extra, config.history, config.buffers)

    counter = 0
    for idx in tqdm(range(len(test_dataset)), "Generating sequence.."):
        if counter % 2 == 1:
            print("skipped\n")
            counter += 1
            continue
        counter += 1
        filename = test_dataset.get_filename(idx)
        subfolder = test_dataset.get_path(idx).split("/")[0]  # we want to retrieve the sub folder of the val sequences
        generate_directory(f"{save_path}/{subfolder}")
        print(f"Filename: {filename}\n")
        ss, ess = test_dataset.__getitem__(idx)
        # prepare data
        # SS
        lr_image = ss[0].unsqueeze(0).to(device)  # shared
        lr_image = utils.pad_to_divisible(lr_image, 2 ** model.down_and_up)
        ss_feature_images = [img.unsqueeze(0).to(device) for img in ss[1]]
        if ss_feature_images:
            ss_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ss_feature_images]
            ss_feature_images = torch.cat(ss_feature_images, dim=1)
        history_images = [img.unsqueeze(0).to(device) for img in ss[2]]
        if history_images:
            history_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in history_images]
            history_images = torch.stack(history_images, dim=2)  # shared
        # ESS
        if extra:
            ess_feature_images = [img.unsqueeze(0).to(device) for img in ess[1]]
            if ess_feature_images:
                ess_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ess_feature_images]
                ess_feature_images = torch.cat(ess_feature_images, dim=1)
        # forward pass for SS
        # lr_image = ss[0].unsqueeze(0).to(device)  # shared
        # lr_image = utils.pad_to_divisible(lr_image, 2 ** model.down_and_up)
        # ss_feature_images = [img.unsqueeze(0).to(device) for img in ss[1]]
        # ss_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ss_feature_images]
        # ss_feature_images = torch.cat(ss_feature_images, dim=1)
        # history_images = [img.unsqueeze(0).to(device) for img in ss[2]]
        # history_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in history_images]
        # history_images = torch.stack(history_images, dim=2)  # shared

        # forward pass for ESS
        # ess_feature_images = [img.unsqueeze(0).to(device) for img in ess[1]]
        # ess_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ess_feature_images]
        # ess_feature_images = torch.cat(ess_feature_images, dim=1)

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
        # Safe generated images into a folder
        ss_frame = F.to_pil_image(ss_output.squeeze(0))
        ss_frame.save(f"{save_path}/{subfolder}/{filename}.png")
        if extra:
            filename = int(filename) + 1
            ess_frame = F.to_pil_image(ess_output.squeeze(0))
            ess_frame.save(f"{save_path}/{subfolder}/{filename:04d}.png")


def main() -> None:
    args = parse_arguments()
    file_path = args.file_path
    test(file_path)


if __name__ == '__main__':
    main()
