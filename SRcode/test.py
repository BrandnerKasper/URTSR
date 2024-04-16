import yaml
from torchvision import transforms
import torch
from tqdm import tqdm
import numpy as np
from enum import Enum, auto
from PIL import Image
import cv2
import os

from utils import utils
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

from data.dataloader import SingleImagePair, MultiImagePair
from SRcode.config import load_yaml_into_config, Config


def get_config_from_pretrained_model(name: str) -> Config:
    config_path = f"configs/{name}.yaml"
    return load_yaml_into_config(config_path)


class DiskMode(Enum):
    PIL = 1
    CV2 = 2
    PT = 3
    NPZ = 4


def load_image_from_disk(mode: DiskMode, path: str, transform: transforms.ToTensor) -> torch.Tensor:
    match mode:
        case DiskMode.PIL:
            # Load with PIL Image
            return transform(Image.open(f"{path}.png").convert('RGB'))
        case DiskMode.CV2:
            # Load the image with CV2
            img = cv2.imread(f"{path}.png", cv2.IMREAD_UNCHANGED)
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return transform(rgb_image)
        case DiskMode.PT:
            # Load the pytorch pt tensor
            return torch.load(f"{path}.pt")
        case DiskMode.NPZ:
            # Load the compressed numpy .npz file to tensor
            img = np.load(f"{path}.png")
            return torch.from_numpy(next(iter(img.values())))
        case _:
            raise ValueError(f"The mode {mode} is not a valid mode with {path}!")


def init_filenames(path: str) -> list[str]:
    lr_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(path) if filename.endswith(".png") and filename.isnumeric()]
    return sorted(set(lr_filenames))


def test() -> None:
    pretrained_model_path = "pretrained_models/flavr.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    path = "dataset/matrix/val/LR/01"
    filenames = init_filenames(path)

    counter = 0
    img_counter = 0
    for file in filenames:
        if counter % 2 == 1:
            continue
        lr_image = load_image_from_disk(DiskMode.CV2, f"{path}/{file}.png", transform=transforms.ToTensor())
        with torch.no_grad():
            output_image = model(lr_image)
            output_image = torch.clamp(output_image, min=0.0, max=1.0)
        # Safe generated images into a folder
        output_images = torch.unbind(output_image, 1)
        for frame in output_images:
            frame = F.to_pil_image(frame)
            frame.save(f"results/{img_counter:04d}.png")
            img_counter += 1


def main() -> None:
    test()


if __name__ == '__main__':
    main()
