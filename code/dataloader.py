import random
import timeit
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import os
import time
from torchvision import transforms
import torch
import torchvision.transforms.functional as F


def init_filenames(root_hr: str, root_lr: str, pattern: str):
    # Extract common part of the filenames (e.g., '0001')
    hr_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(root_hr) if
                    filename.endswith(".png")]
    # Remove pattern
    if pattern:
        lr_filenames = [os.path.splitext(filename.replace(pattern, ''))[0] for filename in os.listdir(root_lr)
                        if filename.endswith(".png")]
    else:
        lr_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(root_lr) if
                        filename.endswith(".png")]

    # Ensure matching filenames in HR and LR
    for lr_filename, hr_filename in zip(sorted(set(lr_filenames)), sorted(set(hr_filenames))):
        assert lr_filename == hr_filename, f"Filenames were not equal: lr filename {lr_filename} != hr filename {hr_filename}"

    return sorted(set(hr_filenames))


def get_random_crop_pair(lr_tensor: torch.Tensor, hr_tensor: torch.Tensor, patch_size: int, scale: int) -> (torch.Tensor, torch.Tensor):
    lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(lr_tensor, output_size=(patch_size, patch_size))
    hr_i, hr_j, hr_h, hr_w = lr_i * scale, lr_j * scale, lr_h * scale, lr_w * scale

    lr_tensor_patch = F.crop(lr_tensor, lr_i, lr_j, lr_h, lr_w)
    hr_tensor_patch = F.crop(hr_tensor, hr_i, hr_j, hr_h, hr_w)

    return lr_tensor_patch, hr_tensor_patch


class CustomDataset(Dataset):
    def __init__(self, root: str, transform=transforms.ToTensor(), pattern: str = None,
                 patch_size: int = None, scale: int = 1):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.transform = transform
        self.pattern = pattern
        self.patch_size = patch_size
        self.scale = scale
        self.filenames = init_filenames(self.root_hr, self.root_lr, self.pattern)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        common_filename = self.filenames[idx]
        if self.pattern:
            lr_path = os.path.join(self.root_lr, common_filename + self.pattern + ".png")
        else:
            lr_path = os.path.join(self.root_lr, common_filename + ".png")
        hr_path = os.path.join(self.root_hr, common_filename + ".png")

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = self.transform(lr_image)
        hr_image = self.transform(hr_image)

        if self.patch_size:
            lr_image, hr_image = get_random_crop_pair(lr_image, hr_image, self.patch_size, self.scale)

        return lr_image, hr_image

    def get_filename(self, idx):
        path = self.filenames[idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename


def main() -> None:
    # measuring time of utils fcts
    transform = transforms.ToTensor()
    root_hr = "dataset/DIV2K/HR"
    root_lr = "dataset/DIV2K/LR"
    pattern = "x2"
    # Use a lambda function to pass the function with its arguments to timeit
    execution_time_filenames = timeit.timeit(lambda: init_filenames(root_hr, root_lr, pattern), number=1)
    print(f"Execution time of filenames: {execution_time_filenames} seconds")

    dataset = CustomDataset(root="dataset/DIV2K", pattern=pattern, patch_size=256, scale=2)

    for lr_image, hr_image in dataset:
        lr_image = F.to_pil_image(lr_image)
        hr_image = F.to_pil_image(hr_image)

        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(lr_image)
        axes[0].set_title('LR image')
        axes[1].imshow(hr_image)
        axes[1].set_title('HR image')
        plt.show()


if __name__ == '__main__':
    main()
