import os
from torchvision import transforms
import torch
from tqdm import tqdm
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

from data.dataloader import SingleImagePair, MultiImagePair, STSSImagePair, DiskMode, STSSCrossValidation, \
    STSSCrossValidation2
from config import load_yaml_into_config, Config
from utils import utils


def get_config_from_pretrained_model(name: str) -> Config:
    config_path = f"configs/{name}.yaml"
    return load_yaml_into_config(config_path)


def generate_directory(path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)


def test() -> None:
    pretrained_model_path = "pretrained_models/stss2.pth"
    save_path = "results/stss2_cross"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    path = "dataset/ue_data_npz/test"

    # test_dataset = MultiImagePair(root=path, number_of_frames=3, last_frame_idx=299,
    #                      transform=transforms.ToTensor(), crop_size=None, scale=2,
    #                      use_hflip=False, use_rotation=False, digits=4, disk_mode=DiskMode.NPZ)
    test_dataset = STSSCrossValidation2(root="dataset/STSS_val_lewis_png", scale=2, number_of_frames=3, crop_size=None, use_hflip=False, use_rotation=False)


    counter = 0
    for idx in tqdm(range(len(test_dataset)), "Generating sequence.."):
        if counter % 2 == 1:
            # print("skipped\n")
            counter += 1
            continue
        counter += 1
        filename = test_dataset.get_filename(idx)
        subfolder = test_dataset.get_path(idx).split("/")[0] # we want to retrieve the sub folder of the val sequences
        generate_directory(f"{save_path}/{subfolder}")
        # print(f"Filename: {filename}\n")
        lr_images, hr_image = test_dataset.__getitem__(idx)
        with torch.no_grad():
            lr_images = [img.unsqueeze(0) for img in lr_images]
            lr_images = [img.to(device) for img in lr_images]
            lr_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in lr_images]
            lr_image = lr_images[0]
            history_images = torch.stack(lr_images, dim=2)
            output_image = model(lr_image, history_images)
            output_image = [torch.clamp(img, min=0.0, max=1.0) for img in output_image]
        # Safe generated images into a folder
        for i in range(len(output_image)):
            frame = F.to_pil_image(output_image[i].squeeze(0))
            # generate the right filename
            filename = int(filename) + i
            # print(f"Save file at {save_path}/{subfolder}/{filename:04d}.png")
            frame.save(f"{save_path}/{subfolder}/{filename:04d}.png")


def test_stss_image_dataset() -> None:
    pretrained_model_path = "pretrained_models/stss.pth"
    save_path = "results/stss"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    path = "dataset/ue_data/test"

    test_dataset = STSSImagePair(root=path, scale=2, history=2, last_frame_idx=299, crop_size=None,
                         use_hflip=False, use_rotation=False, digits=4)
    # test_dataset = STSSCrossValidation(root="dataset/STSS_val_lewis_png", scale=2, history=2, crop_size=None,
    #                                 use_hflip=False, use_rotation=False)

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
        # forward pass for SS
        lr_image = ss[0].unsqueeze(0).to(device)  # shared
        lr_image = utils.pad_to_divisible(lr_image, 2 ** model.down_and_up)
        ss_feature_images = [img.unsqueeze(0).to(device) for img in ss[1]]
        ss_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ss_feature_images]
        ss_feature_images = torch.cat(ss_feature_images, dim=1)
        history_images = [img.unsqueeze(0).to(device) for img in ss[2]]
        history_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in history_images]
        history_images = torch.stack(history_images, dim=2)  # shared

        # forward pass for ESS
        ess_feature_images = [img.unsqueeze(0).to(device) for img in ess[1]]
        ess_feature_images = [utils.pad_to_divisible(img, 2 ** model.down_and_up) for img in ess_feature_images]
        ess_feature_images = torch.cat(ess_feature_images, dim=1)

        with torch.no_grad():
            # SS frame
            ss_output = model(lr_image, ss_feature_images, history_images)
            ss_output = torch.clamp(ss_output, min=0.0, max=1.0)
            # ESS frame
            ess_output = model(lr_image, ess_feature_images, history_images)
            ess_output = torch.clamp(ess_output, min=0.0, max=1.0)
        # Safe generated images into a folder
        ss_frame = F.to_pil_image(ss_output.squeeze(0))
        ss_frame.save(f"{save_path}/{subfolder}/{filename}.png")
        filename = int(filename) + 1
        ess_frame = F.to_pil_image(ess_output.squeeze(0))
        ess_frame.save(f"{save_path}/{subfolder}/{filename:04d}.png")


def main() -> None:
    test()
    # test_stss_image_dataset()


if __name__ == '__main__':
    main()
