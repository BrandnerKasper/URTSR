import random
import timeit
from enum import Enum, auto

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import cv2
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
            img = np.load(f"{path}.npz")
            return torch.from_numpy(next(iter(img.values())))
        case _:
            raise ValueError(f"The mode {mode} is not a valid mode with {path}!")


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
            lr_path = os.path.join(self.root_lr, common_filename + self.pattern)
        else:
            lr_path = os.path.join(self.root_lr, common_filename)
        hr_path = os.path.join(self.root_hr, common_filename)

        lr_image = load_image_from_disk(DiskMode.PT, lr_path, self.transform)
        hr_image = load_image_from_disk(DiskMode.PT, hr_path, self.transform)

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
                 use_hflip: bool = False, use_rotation: bool = False, digits: int = 4, disk_mode=DiskMode.CV2):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.number_of_frames = number_of_frames
        self.last_frame_idx = last_frame_idx
        self.transform = transform
        self.crop_size = crop_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rotation = use_rotation
        self.digits = digits
        self.disk_mode = disk_mode
        self.filenames = self.init_filenames()

    def init_filenames(self) -> list[str]:
        filenames = []
        for directory in os.listdir(self.root_hr):
            for file in os.listdir(os.path.join(self.root_hr, directory)):
                file = os.path.splitext(file)[0]
                # we want a list of images for which # of frames is always possible to retrieve
                # therefore the first (and last) couple of frames need to be excluded
                if self.number_of_frames - 2 < int(file) < self.last_frame_idx:
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
            file = f"{file:0{self.digits}d}"  # Ensure 4/8 digit format
            # Put folder and file name back together and load the tensor
            file = f"{self.root_lr}/{folder}/{file}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            lr_frames.append(file)

        # hr frames = [current, current + 1, ..., current + n], where n = # of frames / 2
        hr_frames = []
        for i in range(int(self.number_of_frames / 2)):
            # Extract the numeric part
            file = int(filename) + i
            # Generate right file name pattern
            file = f"{file:0{self.digits}d}"  # Ensure 4/8 digit format
            # Put folder and file name back together and load the tensor
            file = f"{self.root_hr}/{folder}/{file}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
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

    def get_path(self, idx: int) -> str:
        return self.filenames[idx]

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
            axes[i].set_title(f'LR image {self.get_filename(idx-i)}')

        # Display HR frames
        for i, hr_frame in enumerate(hr_frames):
            hr_image = F.to_pil_image(hr_frame)
            axes[num_lr_frames + i].imshow(hr_image)
            axes[num_lr_frames + i].set_title(f'HR image {self.get_filename(idx+i)}')

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


class STSSImagePair(Dataset):
    def __init__(self, root: str, history: int = 4, last_frame_idx: int = 299,
                 transform=transforms.ToTensor(), crop_size: int = None, scale: int = 2,
                 use_hflip: bool = False, use_rotation: bool = False, digits: int = 4, disk_mode=DiskMode.CV2):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.history = history
        self.last_frame_idx = last_frame_idx
        self.transform = transform
        self.crop_size = crop_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rotation = use_rotation
        self.digits = digits
        self.disk_mode = disk_mode
        self.filenames = self.init_filenames()
        self.feature_names = []
        self.history_names = []

    def init_filenames(self) -> list[str]:
        filenames = []
        for directory in os.listdir(self.root_hr):
            for file in os.listdir(os.path.join(self.root_hr, directory)):
                file = os.path.splitext(file)[0]
                # we want a list of images for which # of frames is always possible to retrieve
                # therefore the first (and last) couple of frames need to be excluded
                if self.history - 1 < int(file) < self.last_frame_idx:
                    filenames.append(os.path.join(directory, file))
        return sorted(set(filenames))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor):
        path = self.filenames[idx]
        folder = path.split("/")[0]
        filename = path.split("/")[-1]

        # lr frame
        file = f"{self.root_lr}/{folder}/{filename}"
        lr_frame = load_image_from_disk(self.disk_mode, file, self.transform)

        # features: basecolor, metallic, roughness, depth, normal, velocity
        feature_frames = []
        file = f"{self.root_lr}/{folder}/{filename}.basecolor"
        self.feature_names.append(f"{filename}.basecolor")
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        feature_frames.append(file)
        file = f"{self.root_lr}/{folder}/{filename}.metallic"
        self.feature_names.append(f"{filename}.metallic")
        file = cv2.imread(f"{file}.png", cv2.IMREAD_GRAYSCALE)
        file = self.transform(file)
        feature_frames.append(file)
        file = f"{self.root_lr}/{folder}/{filename}.roughness"
        self.feature_names.append(f"{filename}.roughness")
        file = cv2.imread(f"{file}.png", cv2.IMREAD_GRAYSCALE)
        file = self.transform(file)
        feature_frames.append(file)
        file = f"{self.root_lr}/{folder}/{filename}.depth_10"
        self.feature_names.append(f"{filename}.depth_10")
        file = cv2.imread(f"{file}.png", cv2.IMREAD_GRAYSCALE)
        file = self.transform(file)
        feature_frames.append(file)
        file = f"{self.root_lr}/{folder}/{filename}.normal_vector"
        self.feature_names.append(f"{filename}.normal_vector")
        file = cv2.imread(f"{file}.png", cv2.IMREAD_GRAYSCALE)
        file = self.transform(file)
        feature_frames.append(file)
        file = f"{self.root_lr}/{folder}/{filename}.velocity_log"
        self.feature_names.append(f"{filename}.velocity_log")
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        file = file[0:2]
        feature_frames.append(file)

        # 3 previous history frames [current - 2, current -4, current -6]
        history_frames = []
        for i in range(self.history):
            # Extract the numeric part
            file = int(filename) - (i + 1) * 2
            # Generate right file name pattern
            file = f"{file:0{self.digits}d}"  # Ensure 4/8 digit format
            self.history_names.append(file)
            # Put folder and file name back together and load the tensor
            file = f"{self.root_lr}/{folder}/{file}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            history_frames.append(file)

        # hr frame
        file = f"{self.root_hr}/{folder}/{filename}"
        hr_frame = load_image_from_disk(self.disk_mode, file, self.transform)

        # Randomly crop the images
        if self.crop_size:
            lr_frame, feature_frames, history_frames, hr_frame = self.get_random_crop_pair(lr_frame, feature_frames, history_frames, hr_frame)

        # Augment images
        lr_frame, feature_frames, history_frames, hr_frame = self.augment(lr_frame, feature_frames, history_frames, hr_frame)

        return lr_frame, feature_frames, history_frames, hr_frame

    def get_filename(self, idx: int) -> str:
        path = self.filenames[idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename

    def get_path(self, idx: int) -> str:
        return self.filenames[idx]

    def display_item(self, idx: int) -> None:
        lr_frame, feature_frames, history_frames, hr_frame = self.__getitem__(idx)

        # Create a single plot with LR images on the left and HR images on the right
        fig, axes = plt.subplots(2, 6, figsize=(20, 12))
        axes = axes.flatten()

        # Display LR frame
        lr_image = F.to_pil_image(lr_frame)
        axes[0].imshow(lr_image)
        axes[0].set_title(f"LR frame {self.get_filename(idx)}")

        # Display feature frames
        for i, feature_frame in enumerate(feature_frames):
            feature_frame = F.to_pil_image(feature_frame)
            axes[i+1].imshow(feature_frame)
            axes[i+1].set_title(f'Feature frame {self.feature_names[i]}')

        # Display feature frames
        for i, history_frame in enumerate(history_frames):
            history_frame = F.to_pil_image(history_frame)
            axes[i + 7].imshow(history_frame)
            axes[i + 7].set_title(f'History frame {self.history_names[i]}')

        # Display HR frame
        hr_image = F.to_pil_image(hr_frame)
        axes[10].imshow(hr_image)
        axes[10].set_title(f"HR frame {self.get_filename(idx)}")

        plt.tight_layout()
        plt.show()

    def get_random_crop_pair(self, lr_frame: torch.Tensor, feature_frames: list[torch.Tensor],
                             history_frames: list[torch.Tensor], hr_frame: torch.Tensor) \
            -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor):
        lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(lr_frame, output_size=(self.crop_size, self.crop_size))
        hr_i, hr_j, hr_h, hr_w = lr_i * self.scale, lr_j * self.scale, lr_h * self.scale, lr_w * self.scale

        # crop lr frame
        lr_frame = F.crop(lr_frame, lr_i, lr_j, lr_h, lr_w)

        # crop features
        for i, feature_frame in enumerate(feature_frames):
            feature_frames[i] = F.crop(feature_frame, lr_i, lr_j, lr_h, lr_w)

        # crop history frames
        for i, history_frame in enumerate(history_frames):
            history_frames[i] = F.crop(history_frame, lr_i, lr_j, lr_h, lr_w)

        # crop hr frame
        hr_frame = F.crop(hr_frame, hr_i, hr_j, hr_h, hr_w)

        return lr_frame, feature_frames, history_frames, hr_frame

    def augment(self, lr_frame: torch.Tensor, feature_frames: list[torch.Tensor],
                history_frames: list[torch.Tensor], hr_frame: torch.Tensor) \
            -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor):
        # Apply random horizontal flip
        if self.use_hflip:
            if random.random() > 0.5:
                # hflip lr frame
                lr_frame = flip_image_horizontal(lr_frame)
                # hflip feature frames
                for i, feature_frame in enumerate(feature_frames):
                    feature_frames[i] = flip_image_horizontal(feature_frame)
                # hflip history frames
                for i, history_frame in enumerate(history_frames):
                    history_frames[i] = flip_image_horizontal(history_frame)
                # hflip hr frame
                hr_frame = flip_image_horizontal(hr_frame)

        # Apply random rotation by v flipping and rot of 90
        if self.use_rotation:
            if random.random() > 0.5:
                # vflip lr frame
                lr_frame = flip_image_vertical(lr_frame)
                # vflip feature frames
                for i, feature_frame in enumerate(feature_frames):
                    feature_frames[i] = flip_image_vertical(feature_frame)
                # vflip history frames
                for i, history_frame in enumerate(history_frames):
                    history_frames[i] = flip_image_vertical(history_frame)
                # vflip hr frame
                hr_frame = flip_image_vertical(hr_frame)
        if self.use_rotation:
            if random.random() > 0.5:
                angle = -90  # for clockwise rotation like BasicSR
                # rotate lr frame
                lr_frame = rotate_image(lr_frame, angle)
                # rotate feature frames
                for i, feature_frame in enumerate(feature_frames):
                    feature_frames[i] = rotate_image(feature_frame, angle)
                # rotate history frames
                for i, history_frame in enumerate(history_frames):
                    history_frames[i] = rotate_image(history_frame, angle)
                # rotate hr frame
                hr_frame = rotate_image(hr_frame, angle)

        return lr_frame, feature_frames, history_frames, hr_frame


def main() -> None:
    # div2k_dataset = SingleImagePair(root="../dataset/DIV2K/train", pattern="x2")
    # div2k_dataset.display_item(0)
    #
    # reds_dataset = MultiImagePair(root="../dataset/Reds/train", scale=4,
    #                               crop_size=96, use_hflip=True, use_rotation=True, digits=8)
    # reds_dataset.display_item(888)

    # matrix_dataset = MultiImagePair(root="../dataset/ue_data/train", scale=2, number_of_frames=4, last_frame_idx=299,
    #                                 crop_size=None, use_hflip=False, use_rotation=False, digits=4)
    # matrix_dataset.display_item(42)

    stss_data = STSSImagePair(root="../dataset/ue_data/train", scale=2, history=3, last_frame_idx=299,
                              crop_size=512, use_hflip=True, use_rotation=True, digits=4)
    stss_data.display_item(42)


if __name__ == '__main__':
    main()
