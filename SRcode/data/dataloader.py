import random
import timeit

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import os
import time
from torchvision import transforms
import torch
import torchvision.transforms.functional as F


def get_random_crop_pair(lr_tensor: torch.Tensor, hr_tensor: torch.Tensor, patch_size: int, scale: int) \
        -> (torch.Tensor, torch.Tensor):
    lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(lr_tensor, output_size=(patch_size, patch_size))
    hr_i, hr_j, hr_h, hr_w = lr_i * scale, lr_j * scale, lr_h * scale, lr_w * scale

    lr_tensor_patch = F.crop(lr_tensor, lr_i, lr_j, lr_h, lr_w)
    hr_tensor_patch = F.crop(hr_tensor, hr_i, hr_j, hr_h, hr_w)

    return lr_tensor_patch, hr_tensor_patch


def flip_image_horizontal(img: torch.Tensor) -> torch.Tensor:
    return F.hflip(img)


def flip_image_vertical(img: torch.Tensor) -> torch.Tensor:
    return F.vflip(img)


def rotate_image(img: torch.Tensor, angle: int) -> torch.Tensor:
    return F.rotate(img, angle)


class SingleImagePair(Dataset):
    def __init__(self, root: str, transform=transforms.ToTensor(), pattern: str = None,
                 crop_size: int = None, scale: int = 2,
                 use_hflip: bool = False, use_rotation: bool = False):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.transform = transform
        self.pattern = pattern
        self.crop_size = crop_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rotation = use_rotation
        self.filenames = self.init_filenames()

    def init_filenames(self) -> list[str]:
        # Extract common part of the filenames (e.g., '0001')
        hr_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(self.root_hr) if
                        filename.endswith(".png")]
        # Remove pattern
        if self.pattern:
            lr_filenames = [os.path.splitext(filename.replace(self.pattern, ''))[0] for filename in
                            os.listdir(self.root_lr)
                            if filename.endswith(".png")]
        else:
            lr_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(self.root_lr) if
                            filename.endswith(".png")]

        # Ensure matching filenames in HR and LR
        for lr_filename, hr_filename in zip(sorted(set(lr_filenames)), sorted(set(hr_filenames))):
            assert lr_filename == hr_filename, f"Filenames were not equal: lr filename {lr_filename} != hr filename {hr_filename}"

        return sorted(set(hr_filenames))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        common_filename = self.filenames[idx]
        if self.pattern:
            lr_path = os.path.join(self.root_lr, common_filename + self.pattern + ".pt")
        else:
            lr_path = os.path.join(self.root_lr, common_filename + ".pt")
        hr_path = os.path.join(self.root_hr, common_filename + ".pt")

        # Load from npz file
        # lr_image = np.load(lr_path)
        # lr_image = torch.from_numpy(next(iter(lr_image.values())))
        # hr_image = np.load(hr_path)
        # hr_image = torch.from_numpy(next(iter(hr_image.values())))

        # Load from pt file
        lr_image = torch.load(lr_path)
        hr_image = torch.load(hr_path)

        # Randomly crop image
        if self.crop_size:
            lr_image, hr_image = get_random_crop_pair(lr_image, hr_image, self.crop_size, self.scale)

        # Augment image by h and v flip and rot by 90
        lr_image, hr_image = self.augment(lr_image, hr_image)

        return lr_image, hr_image

    def augment(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Apply random horizontal flip
        if self.use_hflip:
            if random.random() > 0.5:
                lr_image = flip_image_horizontal(lr_image)
                hr_image = flip_image_horizontal(hr_image)

        # Apply random rotation by v flipping and rot of 90
        if self.use_rotation:
            if random.random() > 0.5:
                lr_image = flip_image_vertical(lr_image)
                hr_image = flip_image_vertical(hr_image)
        if self.use_rotation:
            if random.random() > 0.5:
                angle = -90  # for clockwise rotation like BasicSR
                lr_image = rotate_image(lr_image, angle)
                hr_image = rotate_image(hr_image, angle)
        return lr_image, hr_image

    def get_filename(self, idx: int) -> str:
        path = self.filenames[idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename

    def display_item(self, idx: int) -> None:
        lr_image, hr_image = self.__getitem__(idx)
        lr_image = F.to_pil_image(lr_image)
        hr_image = F.to_pil_image(hr_image)

        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(lr_image)
        axes[0].set_title('LR image')
        axes[1].imshow(hr_image)
        axes[1].set_title('HR image')
        plt.show()


class MultiImagePair(Dataset):
    def __init__(self, root: str, number_of_frames: int = 4, last_frame_idx: int = 100,
                 transform=transforms.ToTensor(), crop_size: int = None, scale: int = 4,
                 use_hflip: bool = False, use_rotation: bool = False):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR/X4")
        self.number_of_frames = number_of_frames
        self.last_frame_idx = last_frame_idx
        self.transform = transform
        self.crop_size = crop_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rotation = use_rotation
        self.filenames = self.init_filenames()

    def init_filenames(self) -> list[str]:
        filenames = []
        for directory in os.listdir(self.root_hr):
            for file in os.listdir(os.path.join(self.root_hr, directory)):
                file = os.path.splitext(file)[0]
                # we want a list of images for which # of frames is always possible to retrieve
                # therefore the first (and last) couple of frames need to be excluded
                if self.number_of_frames - 2 < int(file) < self.last_frame_idx - int(self.number_of_frames / 2):
                    filenames.append(os.path.join(directory, file))
        return sorted(set(filenames))

    def __len__(self) -> int:
        return len(self.filenames)

    # only works for extrapolation atm
    def __getitem__(self, idx: int) -> (list[torch.Tensor], list[torch.Tensor]):
        path = self.filenames[idx]
        folder = path.split("/")[0]
        filename = path.split("/")[-1]

        # lr frames = [current, current - 1, ..., current -n], where n = # of frames
        lr_frames = []
        for i in range(self.number_of_frames):
            # Extract the numeric part
            file = int(filename) - i
            # Generate right file name pattern
            file = f"{file:08d}"  # Ensure 8 digit format
            # Put folder and file name back together and load the tensor
            file = f"{self.root_lr}/{folder}/{file}.png"
            file = self.transform(Image.open(file).convert('RGB'))
            lr_frames.append(file)

        # hr frames = [current, current + 1, ..., current + n], where n = # of frames / 2
        hr_frames = []
        for i in range(int(self.number_of_frames / 2)):
            # Extract the numeric part
            file = int(filename) + i
            # Generate right file name pattern
            file = f"{file:08d}"  # Ensure 8 digit format
            # Put folder and file name back together and load the tensor
            file = f"{self.root_hr}/{folder}/{file}.png"
            file = self.transform(Image.open(file).convert('RGB'))
            hr_frames.append(file)

        # Randomly crop image
        if self.crop_size:
            lr_frames, hr_frames = self.get_random_crop_pair(lr_frames, hr_frames)
        # Augment image by h and v flip and rot by 90
        lr_frames, hr_frames = self.augment(lr_frames, hr_frames)

        return lr_frames, hr_frames

    def get_filename(self, idx: int) -> str:
        path = self.filenames[idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename

    def display_item(self, idx: int) -> None:
        lr_frames, hr_frames = self.__getitem__(idx)

        num_lr_frames = len(lr_frames)
        num_hr_frames = len(hr_frames)

        # Create a single plot with LR images on the left and HR images on the right
        fig, axes = plt.subplots(1, num_lr_frames + num_hr_frames, figsize=(15, 5))

        # Display LR frames
        for i, lr_frame in enumerate(lr_frames):
            lr_image = F.to_pil_image(lr_frame)
            axes[i].imshow(lr_image)
            axes[i].set_title(f'LR image {-i}')

        # Display HR frames
        for i, hr_frame in enumerate(hr_frames):
            hr_image = F.to_pil_image(hr_frame)
            axes[num_lr_frames + i].imshow(hr_image)
            axes[num_lr_frames + i].set_title(f'HR image {i}')

        plt.tight_layout()
        plt.show()

    def get_random_crop_pair(self, lr_frames: list[torch.Tensor], hr_frames: list[torch.Tensor]) \
            -> (list[torch.Tensor], list[torch.Tensor]):
        lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(lr_frames[0], output_size=(self.crop_size, self.crop_size))
        hr_i, hr_j, hr_h, hr_w = lr_i * self.scale, lr_j * self.scale, lr_h * self.scale, lr_w * self.scale

        lr_frame_patches = []
        for lr_frame in lr_frames:
            lr_frame_patches.append(F.crop(lr_frame, lr_i, lr_j, lr_h, lr_w))
        hr_frame_patches = []
        for hr_frame in hr_frames:
            hr_frame_patches.append(F.crop(hr_frame, hr_i, hr_j, hr_h, hr_w))

        return lr_frame_patches, hr_frame_patches

    def augment(self, lr_frames: list[torch.Tensor], hr_frames: list[torch.Tensor]) \
            -> (list[torch.Tensor], list[torch.Tensor]):
        # Apply random horizontal flip
        if self.use_hflip:
            if random.random() > 0.5:
                for i in range(len(lr_frames)):
                    lr_frames[i] = flip_image_horizontal(lr_frames[i])
                for i in range(len(hr_frames)):
                    hr_frames[i] = flip_image_horizontal(hr_frames[i])

        # Apply random rotation by v flipping and rot of 90
        if self.use_rotation:
            if random.random() > 0.5:
                for i in range(len(lr_frames)):
                    lr_frames[i] = flip_image_vertical(lr_frames[i])
                for i in range(len(hr_frames)):
                    hr_frames[i] = flip_image_vertical(hr_frames[i])
        if self.use_rotation:
            if random.random() > 0.5:
                angle = -90  # for clockwise rotation like BasicSR
                for i in range(len(lr_frames)):
                    lr_frames[i] = rotate_image(lr_frames[i], angle)
                for i in range(len(hr_frames)):
                    hr_frames[i] = rotate_image(hr_frames[i], angle)
        return lr_frames, hr_frames


def main() -> None:
    div2k_dataset = SingleImagePair(root="../dataset/DIV2K/train", pattern="x2")
    div2k_dataset.display_item(0)

    reds_dataset = MultiImagePair(root="../dataset/Reds/train", crop_size=96, use_hflip=True, use_rotation=True)
    reds_dataset.display_item(888)


if __name__ == '__main__':
    main()
