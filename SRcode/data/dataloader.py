import math
import random
import timeit
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import time
from torchvision import transforms
import torch
import torchvision.transforms.functional as FV
import torch.nn.functional as F
import multiprocessing

# Set the start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def get_random_crop_pair(lr_tensor: torch.Tensor, hr_tensor: torch.Tensor, patch_size: int, scale: int) \
        -> (torch.Tensor, torch.Tensor):
    lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(lr_tensor, output_size=(patch_size, patch_size))
    hr_i, hr_j, hr_h, hr_w = lr_i * scale, lr_j * scale, lr_h * scale, lr_w * scale

    lr_tensor_patch = FV.crop(lr_tensor, lr_i, lr_j, lr_h, lr_w)
    hr_tensor_patch = FV.crop(hr_tensor, hr_i, hr_j, hr_h, hr_w)

    return lr_tensor_patch, hr_tensor_patch


def flip_image_horizontal(img: torch.Tensor) -> torch.Tensor:
    return FV.hflip(img)


def flip_image_vertical(img: torch.Tensor) -> torch.Tensor:
    return FV.vflip(img)


def rotate_image(img: torch.Tensor, angle: int) -> torch.Tensor:
    return FV.rotate(img, angle)


class DiskMode(Enum):
    PIL = 1
    CV2 = 2
    PT = 3
    NPZ = 4


def load_image_from_disk(mode: DiskMode, path: str, transform=transforms.ToTensor(),
                         read_mode=cv2.IMREAD_UNCHANGED) -> torch.Tensor:
    match mode:
        case DiskMode.PIL:
            # Load with PIL Image
            return transform(Image.open(f"{path}.png").convert('RGB'))
        case DiskMode.CV2:
            # Load the image with CV2
            img = cv2.imread(f"{path}.png", read_mode)
            # Convert BGR to RGB
            if read_mode == cv2.IMREAD_UNCHANGED:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return transform(img)
        case DiskMode.PT:
            # Load the pytorch pt tensor
            return torch.load(f"{path}.pt")
        case DiskMode.NPZ:
            # Load the compressed numpy .npz file to tensor
            img = np.load(f"{path}.npz")
            return torch.from_numpy(next(iter(img.values())))
        case _:
            raise ValueError(f"The mode {mode} is not a valid mode with {path}!")


# Recursive function to count values in a list containing lists
def count_values(list_instance: list):
    count = 0
    for item in list_instance:
        if isinstance(item, list):
            count += count_values(item)  # Recursively count values
        else:
            count += 1  # Increment count for non-list item
    return count


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
        lr_image = FV.to_pil_image(lr_image)
        hr_image = FV.to_pil_image(hr_image)

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
        for i in range(math.ceil(self.number_of_frames / 2)):
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
            lr_image = FV.to_pil_image(lr_frame)
            axes[i].imshow(lr_image)
            axes[i].set_title(f'LR image {self.get_filename(idx - i)}')

        # Display HR frames
        for i, hr_frame in enumerate(hr_frames):
            hr_image = FV.to_pil_image(hr_frame)
            axes[num_lr_frames + i].imshow(hr_image)
            axes[num_lr_frames + i].set_title(f'HR image {self.get_filename(idx + i)}')

        plt.tight_layout()
        plt.show()

    def get_random_crop_pair(self, lr_frames: list[torch.Tensor], hr_frames: list[torch.Tensor]) \
            -> (list[torch.Tensor], list[torch.Tensor]):
        lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(lr_frames[0],
                                                                  output_size=(self.crop_size, self.crop_size))
        hr_i, hr_j, hr_h, hr_w = lr_i * self.scale, lr_j * self.scale, lr_h * self.scale, lr_w * self.scale

        lr_frame_patches = []
        for lr_frame in lr_frames:
            lr_frame_patches.append(FV.crop(lr_frame, lr_i, lr_j, lr_h, lr_w))
        hr_frame_patches = []
        for hr_frame in hr_frames:
            hr_frame_patches.append(FV.crop(hr_frame, hr_i, hr_j, hr_h, hr_w))

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


# warping
def warp_img(image: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    # move pixel values back to be between -0.5 and 0.5
    mv = (mv - 0.5)
    mv_r = mv[0, :, :]
    mv_g = mv[1, :, :]
    mv_g = mv_g * -1

    mv = torch.stack([mv_r, mv_g], dim=-1)
    image = image.unsqueeze(0).cuda()
    mv = mv.unsqueeze(0).cuda()
    mv = mv.permute(0, 3, 1, 2)
    return warp(image, mv).squeeze(0).cpu()


def warp(x, flow, mode='bilinear', padding_mode='zeros'):
    """ Modified warp to forward warp image x based on motion vector flow

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """
    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    x = x.float()
    grid = grid.float()
    flow = flow.float()

    # Ensure the grid is within the range [-1, 1]
    grid[:, :, 0] = torch.clamp(grid[:, :, 0], -1, 1)
    grid[:, :, 1] = torch.clamp(grid[:, :, 1], -1, 1)

    # add flow to grid and reshape to nhw2
    grid = (grid - flow).permute(0, 2, 3, 1) # we use - for forward warping

    output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    output = torch.clamp(output, min=0, max=1)
    return output


def generate_error_mask(pre_warped: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
    # Ensure the images are of the same size
    if pre_warped.shape != warped.shape:
        raise ValueError("Ground truth and super-resolved images must have the same dimensions.")

    # Convert images to float32 for precise difference calculation
    pre_warped = pre_warped.float()
    warped = warped.float()

    # Calculate the absolute difference
    error = torch.abs(warped - pre_warped)

    # Normalize the error to the range [0, 1]
    error_min = error.min()
    error_max = error.max()
    error_norm = (error - error_min) / (error_max - error_min)
    # Define the weights for RGB to Grayscale conversion
    weights = torch.tensor([0.299, 0.587, 0.114])

    # Ensure weights have the same shape as the input tensor channels
    weights = weights.view(3, 1, 1)

    # Convert RGB to Grayscale
    mask = (error_norm * weights).sum(dim=0)
    # If you want to apply a threshold, uncomment the following line:
    # error_norm = torch.where(error_norm < 0.3, torch.tensor(0.0), error_norm)
    # We turn the mask to greyscale:


    return mask.unsqueeze(0) # general for masking it is more interesting to use the error_norm here


class STSSImagePair(Dataset):
    def __init__(self, root: str, extra: bool = True, history: int = 3, buffers: dict[str, bool] = None, last_frame_idx: int = 299,
                 transform=transforms.ToTensor(), crop_size: int = None, scale: int = 2,
                 use_hflip: bool = False, use_rotation: bool = False, digits: int = 4, disk_mode=DiskMode.CV2):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.extra = extra
        self.history = history
        self.buffers = buffers
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
                # therefore the first couple of frames (and the last frame) need to be excluded
                if self.history * 2 - 1 < int(file) <= self.last_frame_idx - self.extra:
                    filenames.append(os.path.join(directory, file))
        return sorted(set(filenames))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> (list[torch.Tensor], list[torch.Tensor]):
        path = self.filenames[idx]
        folder = path.split("/")[0]
        filename = path.split("/")[-1]

        # Generate SS frame
        # lr frame
        file = f"{self.root_lr}/{folder}/{filename}"
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        ss_lr_frame = file
        ss_mask = torch.ones(ss_lr_frame.size())
        ss_mask_name = f"{filename}_mask"

        # features: basecolor, depth, metallic, nov, roughness, world normal, world position
        buffer_frames, buffer_frame_names = self.load_buffers(folder, filename)

        # previous history frames e.g. [current - 2, current -4, current -6]
        history_frames = []
        history_frames_names = []
        for i in range(self.history):
            # Extract the numeric part
            h_filename = int(filename) - (i + 1) * 2
            # Generate right file name pattern
            h_filename = f"{h_filename:0{self.digits}d}"  # Ensure 4/8 digit format
            history_frames_names.append(h_filename)
            # Put folder and file name back together and load the tensor
            file = f"{self.root_lr}/{folder}/{h_filename}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            # warp history frames
            counter = int(filename) - int(h_filename)
            for j in range(counter+1): # we need to warp from t_0 to t_1
                mv_filename = int(filename) - counter + j + 1
                mv_filename = f"{mv_filename:0{self.digits}d}"  # Ensure 4/8 digit format
                mv = self.load_mv(folder, mv_filename)
                file = warp_img(file, mv)
            history_frames.append(file)

        # hr frame
        file = f"{self.root_hr}/{folder}/{filename}"
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        hr_frame = file

        ss = [ss_lr_frame, ss_mask, buffer_frames, history_frames, hr_frame]
        self.ss_names = [filename, ss_mask_name, buffer_frame_names, history_frames_names, filename]

        ess = []
        self.ess_names = []
        if self.extra:
            # Generate ESS frame
            # get the future frames for extrapolation
            ess_filename = int(filename) + 1
            ess_filename = f"{ess_filename:0{self.digits}d}"  # Ensure 4/8 digit format

            # lr frame -> we warp the frame based on the next motion vector
            mv = self.load_mv(folder, ess_filename)
            ess_lr_frame = warp_img(ss_lr_frame, mv)
            # generate a mask btw. the warped and not warped image
            ess_mask = generate_error_mask(ss_lr_frame, ess_lr_frame)
            ess_mask_name = f"{filename}_{ess_filename}_mask"

            # features: basecolor, depth, metallic, nov, roughness, velocity, world normal, world position
            buffer_frames, buffer_frame_names = self.load_buffers(folder, ess_filename)

            # history frames -> for now use same history frames as the SS frame

            # hr frame
            file = f"{self.root_hr}/{folder}/{ess_filename}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            hr_frame = file

            ess = [ess_lr_frame, ess_mask, buffer_frames, history_frames, hr_frame]
            self.ess_names = [filename, ess_mask_name, buffer_frame_names, history_frames_names, ess_filename]

        # Randomly crop the images
        if self.crop_size:
            self.get_random_crop_pair(ss, ess)

        # # Augment images
        self.augment(ss, ess)

        return ss, ess

    def load_mv(self, folder: str, filename: str) -> torch.Tensor:
        path = f"{self.root_lr}/{folder}/{filename}.velocity"
        return load_image_from_disk(self.disk_mode, path)

    def load_buffers(self, folder: str, filename: str) -> Optional[Tuple[list[torch.Tensor], list[str]]]:
        if self.buffers is None:
            return
        feature_frames = []
        feature_frames_names = []
        # features: basecolor, depth, metallic, nov, roughness, velocity, world normal, world position
        if self.buffers["BASE_COLOR"]:
            file = f"{self.root_lr}/{folder}/{filename}.basecolor"
            feature_frames_names.append(f"{filename}.basecolor")
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            feature_frames.append(file)
        if self.buffers["DEPTH"]:  # for now we use log depth files
            file = f"{self.root_lr}/{folder}/{filename}.depth_log"
            feature_frames_names.append(f"{filename}.depth_log")
            file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
            feature_frames.append(file)
        if self.buffers["METALLIC"]:
            file = f"{self.root_lr}/{folder}/{filename}.metallic"
            feature_frames_names.append(f"{filename}.metallic")
            file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
            feature_frames.append(file)
        if self.buffers["NOV"]:
            file = f"{self.root_lr}/{folder}/{filename}.normal_vector"
            feature_frames_names.append(f"{filename}.normal_vector")
            file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
            feature_frames.append(file)
        if self.buffers["ROUGHNESS"]:
            file = f"{self.root_lr}/{folder}/{filename}.roughness"
            feature_frames_names.append(f"{filename}.roughness")
            file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
            feature_frames.append(file)
        # if self.buffers["VELOCITY"]:  # we use velocity log for now
        #     file = f"{self.root_lr}/{folder}/{filename}.velocity_log"
        #     feature_frames_names.append(f"{filename}.velocity_log")
        #     file = load_image_from_disk(self.disk_mode, file, self.transform)
        #     # file = file[0:2] # actually we only need R and G from this buffer, but for now we use RGB
        #     feature_frames.append(file)
        if self.buffers["WORLD_NORMAL"]:
            file = f"{self.root_lr}/{folder}/{filename}.world_normal"
            feature_frames_names.append(f"{filename}.world_normal")
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            feature_frames.append(file)
        if self.buffers["WORLD_POSITION"]:
            file = f"{self.root_lr}/{folder}/{filename}.world_position"
            feature_frames_names.append(f"{filename}.world_position")
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            feature_frames.append(file)
        return feature_frames, feature_frames_names

    def get_filename(self, idx: int) -> str:
        path = self.filenames[idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename

    def get_path(self, idx: int) -> str:
        return self.filenames[idx]

    def display_item(self, idx: int) -> None:
        ss, ess = self.__getitem__(idx)

        # Create two plots: one for the SS frame and one for the ESS frame
        # Display SS frame
        values = count_values(ss)
        fig_ss, axes_ss = plt.subplots(2, math.ceil(values / 2), figsize=(20, 12))
        fig_ss.suptitle('SS frames')

        axes_ss = axes_ss.flatten()
        counter = 0

        # Display LR frame
        lr_image = FV.to_pil_image(ss[0])
        axes_ss[counter].imshow(lr_image)
        axes_ss[counter].set_title(f"LR frame {self.ss_names[0]}")
        counter += 1

        # Display feature frames
        for i, feature_frame in enumerate(ss[2]):
            feature_frame = FV.to_pil_image(feature_frame)
            axes_ss[counter].imshow(feature_frame)
            axes_ss[counter].set_title(f'Feature frame {self.ss_names[2][i]}')
            counter += 1

        # Display feature frames
        for i, history_frame in enumerate(ss[3]):
            history_frame = FV.to_pil_image(history_frame)
            axes_ss[counter].imshow(history_frame)
            axes_ss[counter].set_title(f'History frame {self.ss_names[3][i]}')
            counter += 1

        # Display HR frame
        hr_image = FV.to_pil_image(ss[4])
        axes_ss[counter].imshow(hr_image)
        axes_ss[counter].set_title(f"HR frame {self.ss_names[4]}")

        # Display ESS frame
        if self.extra:
            values = count_values(ess) + 1
            fig_ess, axes_ess = plt.subplots(2, math.ceil(values / 2), figsize=(20, 12))
            fig_ess.suptitle('ESS frames')

            axes_ess = axes_ess.flatten()
            counter = 0

            # Display LR frame
            lr_image = FV.to_pil_image(ess[0])
            axes_ess[counter].imshow(lr_image)
            axes_ess[counter].set_title(f"warped LR frame {self.ess_names[0]}")
            counter += 1

            # Display Mask
            mask_image = FV.to_pil_image(ess[1])
            axes_ess[counter].imshow(mask_image, cmap="grey")
            axes_ess[counter].set_title(f"Mask {self.ess_names[1]}")
            counter += 1

            # Display feature frames
            for i, feature_frame in enumerate(ess[2]):
                feature_frame = FV.to_pil_image(feature_frame)
                axes_ess[counter].imshow(feature_frame)
                axes_ess[counter].set_title(f'Feature frame {self.ess_names[2][i]}')
                counter += 1

            # Display feature frames
            for i, history_frame in enumerate(ess[3]):
                history_frame = FV.to_pil_image(history_frame)
                axes_ess[counter].imshow(history_frame)
                axes_ess[counter].set_title(f'History frame {self.ess_names[3][i]}')
                counter += 1

            # Display HR frame
            hr_image = FV.to_pil_image(ess[4])
            axes_ess[counter].imshow(hr_image)
            axes_ess[counter].set_title(f"HR frame {self.ess_names[4]}")

        plt.tight_layout()
        plt.show()

    def get_random_crop_pair(self, ss: list[torch.Tensor], ess: list[torch.Tensor]) \
            -> None:
        lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(ss[0],
                                                                  output_size=(self.crop_size, self.crop_size))
        hr_i, hr_j, hr_h, hr_w = lr_i * self.scale, lr_j * self.scale, lr_h * self.scale, lr_w * self.scale

        # Crop SS frame
        # crop lr frame
        ss[0] = FV.crop(ss[0], lr_i, lr_j, lr_h, lr_w)
        # crop the mask
        ss[1] = FV.crop(ss[1], lr_i, lr_j, lr_h, lr_w)
        # crop features
        for i, feature_frame in enumerate(ss[2]):
            ss[2][i] = FV.crop(feature_frame, lr_i, lr_j, lr_h, lr_w)
        # crop history frames
        for i, history_frame in enumerate(ss[3]):
            ss[3][i] = FV.crop(history_frame, lr_i, lr_j, lr_h, lr_w)
        # crop hr frame
        ss[4] = FV.crop(ss[4], hr_i, hr_j, hr_h, hr_w)

        if not self.extra:
            return
        # Crop ESS frame
        # crop lr frame
        ess[0] = FV.crop(ess[0], lr_i, lr_j, lr_h, lr_w)
        # crop the mask
        ess[1] = FV.crop(ess[1], lr_i, lr_j, lr_h, lr_w)
        # crop features
        for i, feature_frame in enumerate(ess[2]):
            ess[2][i] = FV.crop(feature_frame, lr_i, lr_j, lr_h, lr_w)
        # crop history frames -> shared
        # crop hr frame
        ess[4] = FV.crop(ess[4], hr_i, hr_j, hr_h, hr_w)

    def augment(self, ss: list[torch.Tensor], ess: list[torch.Tensor]) \
            -> None:
        # Augment SS & ESS frame
        # Apply random horizontal flip
        if self.use_hflip:
            if random.random() > 0.5:
                # SS
                # hflip ss lr frame
                ss[0] = flip_image_horizontal(ss[0])
                # hflip mask
                ss[1] = flip_image_horizontal(ss[1])
                # hflip ss feature frames
                for i, feature_frame in enumerate(ss[2]):
                    ss[2][i] = flip_image_horizontal(feature_frame)
                # hflip ss history frames -> ess shared
                for i, history_frame in enumerate(ss[3]):
                    ss[3][i] = flip_image_horizontal(history_frame)
                # hflip ss hr frame
                ss[4] = flip_image_horizontal(ss[4])
                # ESS
                if self.extra:
                    ess[0] = flip_image_horizontal(ess[0])
                    ess[1] = flip_image_horizontal(ess[1])
                    for i, feature_frame in enumerate(ess[2]):
                        ess[2][i] = flip_image_horizontal(feature_frame)
                    ess[4] = flip_image_horizontal(ess[4])

        # Apply random rotation by v flipping and rot of 90
        if self.use_rotation:
            if random.random() > 0.5:
                # SS
                # vflip ss lr frame
                ss[0] = flip_image_vertical(ss[0])
                # vlip mask
                ss[1] = flip_image_vertical(ss[1])
                # vflip ss feature frames
                for i, feature_frame in enumerate(ss[2]):
                    ss[2][i] = flip_image_vertical(feature_frame)
                # vflip ss history frames -> ess shared
                for i, history_frame in enumerate(ss[3]):
                    ss[3][i] = flip_image_vertical(history_frame)
                # vflip ss hr frame
                ss[4] = flip_image_vertical(ss[4])
                # ESS
                if self.extra:
                    ess[0] = flip_image_vertical(ess[0])
                    ess[1] = flip_image_vertical(ess[1])
                    for i, feature_frame in enumerate(ess[2]):
                        ess[2][i] = flip_image_vertical(feature_frame)
                    ess[4] = flip_image_vertical(ess[4])

        if self.use_rotation:
            if random.random() > 0.5:
                angle = -90  # for clockwise rotation like BasicSR
                # SS
                # rotate ss lr frame
                ss[0] = rotate_image(ss[0], angle)
                # rotate mask
                ss[1] = rotate_image(ss[1].unsqueeze(0), angle).squeeze(0)
                # rotate ss feature frames
                for i, feature_frame in enumerate(ss[2]):
                    ss[2][i] = rotate_image(feature_frame, angle)
                # rotate ss history frames -> ess shared
                for i, history_frame in enumerate(ss[3]):
                    ss[3][i] = rotate_image(history_frame, angle)
                # rotate ss hr frame
                ss[4] = rotate_image(ss[4], angle)
                # ESS
                if self.extra:
                    ess[0] = rotate_image(ess[0], angle)
                    ess[1] = rotate_image(ess[1].unsqueeze(0), angle).squeeze(0)
                    for i, feature_frame in enumerate(ess[2]):
                        ess[2][i] = rotate_image(feature_frame, angle)
                    ess[4] = rotate_image(ess[4], angle)


class STSSCrossValidation(Dataset):
    def __init__(self, root: str, history: int = 3,
                 transform=transforms.ToTensor(), crop_size: int = None, scale: int = 2,
                 use_hflip: bool = False, use_rotation: bool = False, disk_mode=DiskMode.CV2):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.history = history
        self.transform = transform
        self.crop_size = crop_size
        self.scale = scale
        self.use_hflip = use_hflip
        self.use_rotation = use_rotation
        self.disk_mode = disk_mode
        self.filenames = self.init_filenames()
        self.ss_names: list[str]
        self.ess_names: list[str]

    def init_filenames(self) -> list[str]:
        filenames = []
        for file in os.listdir(self.root_lr):
            file = os.path.splitext(file)[0]
            if len(file.split(".")) < 2:
                filenames.append(os.path.join(self.root_lr, file))
        filenames.sort()
        # remove the first entries for history
        for i in range(self.history):
            filenames.pop(0)
        # remove last entry for ess
        filenames.pop()
        return filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> (list[torch.Tensor], list[torch.Tensor]):
        filename = self.get_filename(idx)

        # Generate SS frame
        # lr frame
        file = f"{self.root_lr}/{filename}"
        lr_frame_name = filename
        lr_frame = load_image_from_disk(self.disk_mode, file, self.transform)

        # features: basecolor, depth, metallic, nov, roughness, velocity
        feature_frames = []
        feature_frames_names = []
        # basecolor
        file = f"{self.root_lr}/{filename}.basecolor"
        feature_frames_names.append(f"{filename}.basecolor")
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        feature_frames.append(file)
        # depth
        file = f"{self.root_lr}/{filename}.depth_log"
        feature_frames_names.append(f"{filename}.depth_log")
        file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
        feature_frames.append(file)
        # metallic
        file = f"{self.root_lr}/{filename}.metallic"
        feature_frames_names.append(f"{filename}.metallic")
        file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
        feature_frames.append(file)
        # roughness
        file = f"{self.root_lr}/{filename}.roughness"
        feature_frames_names.append(f"{filename}.roughness")
        file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
        feature_frames.append(file)
        # world normal
        file = f"{self.root_lr}/{filename}.world_normal"
        feature_frames_names.append(f"{filename}.world_normal")
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        feature_frames.append(file)

        # 2 previous history frames
        history_frames = []
        history_frames_names = []
        # edge cases because of STSS val dataset..
        if idx == 0:
            # h1
            h1_name = "0007"
            h1 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h1_name}", self.transform)
            history_frames.append(h1)
            history_frames_names.append(h1_name)
            # h2
            h2_name = "0005"
            h2 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h2_name}", self.transform)
            history_frames.append(h2)
            history_frames_names.append(h2_name)
        elif idx == 1:
            # h1
            h1_name = self.get_filename(idx - 1)
            h1 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h1_name}", self.transform)
            history_frames.append(h1)
            history_frames_names.append(h1_name)
            # h2
            h2_name = "0007"
            h2 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h2_name}", self.transform)
            history_frames.append(h2)
            history_frames_names.append(h2_name)
        else:
            for i in range(self.history):
                file = self.get_filename(idx - (i + 1))
                history_frames_names.append(file)
                file = f"{self.root_lr}/{file}"
                file = load_image_from_disk(self.disk_mode, file, self.transform)
                history_frames.append(file)

        # hr frame
        file = f"{self.root_hr}/{filename}"
        hr_frame_name = filename
        hr_frame = load_image_from_disk(self.disk_mode, file, self.transform)

        ss = [lr_frame, feature_frames, history_frames, hr_frame]
        self.ss_names = [lr_frame_name, feature_frames_names, history_frames_names, hr_frame_name]

        # Generate ESS frame
        # lr frame -> we use the same as the SS frame

        # get the future frames for extrapolation
        if idx == len(self.filenames) - 1:
            ess_filename = "3599"
        else:
            ess_filename = self.get_filename(idx + 1)

        # features: basecolor, depth, metallic, nov, roughness, velocity
        feature_frames = []
        feature_frames_names = []
        # basecolor
        file = f"{self.root_lr}/{ess_filename}.basecolor"
        feature_frames_names.append(f"{ess_filename}.basecolor")
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        feature_frames.append(file)
        # depth
        file = f"{self.root_lr}/{ess_filename}.depth_log"
        feature_frames_names.append(f"{ess_filename}.depth_log")
        file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
        feature_frames.append(file)
        # metallic
        file = f"{self.root_lr}/{ess_filename}.metallic"
        feature_frames_names.append(f"{ess_filename}.metallic")
        file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
        feature_frames.append(file)
        # roughness
        file = f"{self.root_lr}/{ess_filename}.roughness"
        feature_frames_names.append(f"{ess_filename}.roughness")
        file = load_image_from_disk(self.disk_mode, file, self.transform, cv2.IMREAD_GRAYSCALE)
        feature_frames.append(file)
        # world normal
        file = f"{self.root_lr}/{ess_filename}.world_normal"
        feature_frames_names.append(f"{ess_filename}.world_normal")
        file = load_image_from_disk(self.disk_mode, file, self.transform)
        feature_frames.append(file)

        # history frames -> for now use same history frames as the SS frame

        # hr frame
        file = f"{self.root_hr}/{ess_filename}"
        hr_frame = load_image_from_disk(self.disk_mode, file, self.transform)
        hr_frame_name = ess_filename

        ess = [lr_frame, feature_frames, history_frames, hr_frame]
        self.ess_names = [lr_frame_name, feature_frames_names, history_frames_names, hr_frame_name]

        # Randomly crop the images
        if self.crop_size:
            self.get_random_crop_pair(ss, ess)

        # # Augment images
        self.augment(ss, ess)

        return ss, ess

    def get_filename(self, idx: int) -> str:
        path = self.filenames[idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename

    def get_path(self, idx: int) -> str:
        return self.filenames[idx]

    def display_item(self, idx: int) -> None:
        ss, ess = self.__getitem__(idx)

        # Create two plots: one for the SS frame and one for the ESS frame
        # Display SS frame
        fig_ss, axes_ss = plt.subplots(2, 6, figsize=(20, 12))
        fig_ss.suptitle('SS frames')

        axes_ss = axes_ss.flatten()

        # Display LR frame
        lr_image = FV.to_pil_image(ss[0])
        axes_ss[0].imshow(lr_image)
        axes_ss[0].set_title(f"LR frame {self.ss_names[0]}")

        # Display feature frames
        for i, feature_frame in enumerate(ss[1]):
            feature_frame = FV.to_pil_image(feature_frame)
            axes_ss[i + 1].imshow(feature_frame)
            axes_ss[i + 1].set_title(f'Feature frame {self.ss_names[1][i]}')

        # Display feature frames
        for i, history_frame in enumerate(ss[2]):
            history_frame = FV.to_pil_image(history_frame)
            axes_ss[i + 7].imshow(history_frame)
            axes_ss[i + 7].set_title(f'History frame {self.ss_names[2][i]}')

        # Display HR frame
        hr_image = FV.to_pil_image(ss[3])
        axes_ss[10].imshow(hr_image)
        axes_ss[10].set_title(f"HR frame {self.ss_names[3]}")

        # Display ESS frame
        fig_ess, axes_ess = plt.subplots(2, 6, figsize=(20, 12))
        fig_ess.suptitle('ESS frames')

        axes_ess = axes_ess.flatten()

        # Display LR frame
        lr_image = FV.to_pil_image(ess[0])
        axes_ess[0].imshow(lr_image)
        axes_ess[0].set_title(f"LR frame {self.ess_names[0]}")

        # Display feature frames
        for i, feature_frame in enumerate(ess[1]):
            feature_frame = FV.to_pil_image(feature_frame)
            axes_ess[i + 1].imshow(feature_frame)
            axes_ess[i + 1].set_title(f'Feature frame {self.ess_names[1][i]}')

        # Display feature frames
        for i, history_frame in enumerate(ess[2]):
            history_frame = FV.to_pil_image(history_frame)
            axes_ess[i + 7].imshow(history_frame)
            axes_ess[i + 7].set_title(f'History frame {self.ess_names[2][i]}')

        # Display HR frame
        hr_image = FV.to_pil_image(ess[3])
        axes_ess[10].imshow(hr_image)
        axes_ess[10].set_title(f"HR frame {self.ess_names[3]}")

        plt.tight_layout()
        plt.show()

    def get_random_crop_pair(self, ss: list[torch.Tensor], ess: list[torch.Tensor]) \
            -> None:
        lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(ss[0],
                                                                  output_size=(self.crop_size, self.crop_size))
        hr_i, hr_j, hr_h, hr_w = lr_i * self.scale, lr_j * self.scale, lr_h * self.scale, lr_w * self.scale

        # Crop SS frame
        # crop lr frame
        ss[0] = FV.crop(ss[0], lr_i, lr_j, lr_h, lr_w)
        # crop features
        for i, feature_frame in enumerate(ss[1]):
            ss[1][i] = FV.crop(feature_frame, lr_i, lr_j, lr_h, lr_w)
        # crop history frames
        for i, history_frame in enumerate(ss[2]):
            ss[2][i] = FV.crop(history_frame, lr_i, lr_j, lr_h, lr_w)
        # crop hr frame
        ss[3] = FV.crop(ss[3], hr_i, hr_j, hr_h, hr_w)

        # Crop ESS frame
        # crop lr frame
        ess[0] = FV.crop(ess[0], lr_i, lr_j, lr_h, lr_w)
        # crop features
        for i, feature_frame in enumerate(ess[1]):
            ess[1][i] = FV.crop(feature_frame, lr_i, lr_j, lr_h, lr_w)
        # crop history frames -> shared
        # crop hr frame
        ess[3] = FV.crop(ess[3], hr_i, hr_j, hr_h, hr_w)

    def augment(self, ss: list[torch.Tensor], ess: list[torch.Tensor]) \
            -> None:
        # Augment SS & ESS frame
        # Apply random horizontal flip
        if self.use_hflip:
            if random.random() > 0.5:
                # hflip ss, ess lr frame
                ss[0] = flip_image_horizontal(ss[0])
                ess[0] = flip_image_horizontal(ess[0])
                # hflip ss, ess feature frames
                for i, feature_frame in enumerate(ss[1]):
                    ss[1][i] = flip_image_horizontal(feature_frame)
                for i, feature_frame in enumerate(ess[1]):
                    ess[1][i] = flip_image_horizontal(feature_frame)
                # hflip ss history frames -> ess shared
                for i, history_frame in enumerate(ss[2]):
                    ss[2][i] = flip_image_horizontal(history_frame)
                # hflip ss,ess hr frame
                ss[3] = flip_image_horizontal(ss[3])
                ess[3] = flip_image_horizontal(ess[3])

        # Apply random rotation by v flipping and rot of 90
        if self.use_rotation:
            if random.random() > 0.5:
                # vflip ss, ess lr frame
                ss[0] = flip_image_vertical(ss[0])
                ess[0] = flip_image_vertical(ess[0])
                # vflip ss, ess feature frames
                for i, feature_frame in enumerate(ss[1]):
                    ss[1][i] = flip_image_vertical(feature_frame)
                for i, feature_frame in enumerate(ess[1]):
                    ess[1][i] = flip_image_vertical(feature_frame)
                # vflip ss history frames -> ess shared
                for i, history_frame in enumerate(ss[2]):
                    ss[2][i] = flip_image_vertical(history_frame)
                # vflip ss, ess hr frame
                ss[3] = flip_image_vertical(ss[3])
                ess[3] = flip_image_vertical(ess[3])
        if self.use_rotation:
            if random.random() > 0.5:
                angle = -90  # for clockwise rotation like BasicSR
                # rotate ss, ess lr frame
                ss[0] = rotate_image(ss[0], angle)
                ess[0] = rotate_image(ess[0], angle)
                # rotate ss, ess feature frames
                for i, feature_frame in enumerate(ss[1]):
                    ss[1][i] = rotate_image(feature_frame, angle)
                for i, feature_frame in enumerate(ess[1]):
                    ess[1][i] = rotate_image(feature_frame, angle)
                # rotate ss history frames -> ess shared
                for i, history_frame in enumerate(ss[2]):
                    ss[2][i] = rotate_image(history_frame, angle)
                # rotate ss, ess hr frame
                ss[3] = rotate_image(ss[3], angle)
                ess[3] = rotate_image(ess[3], angle)


class STSSCrossValidation2(Dataset):
    def __init__(self, root: str, number_of_frames: int = 4,
                 transform=transforms.ToTensor(), crop_size: int = None, scale: int = 4,
                 use_hflip: bool = False, use_rotation: bool = False, digits: int = 4, disk_mode=DiskMode.CV2):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.number_of_frames = number_of_frames
        self.history = number_of_frames - 1
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
        for file in os.listdir(self.root_lr):
            file = os.path.splitext(file)[0]
            if len(file.split(".")) < 2:
                filenames.append(os.path.join(self.root_lr, file))
        filenames.sort()
        # remove the first entries for history
        for i in range(self.history):
            filenames.pop(0)
        # remove last entry for ess
        filenames.pop()
        return filenames

    def __len__(self) -> int:
        return len(self.filenames)

    # only works for extrapolation atm
    def __getitem__(self, idx: int) -> (list[torch.Tensor], list[torch.Tensor]):

        lr_frames = []

        # edge cases because of STSS val dataset..
        if idx == 0:
            filename = self.get_filename(idx)
            file = f"{self.root_lr}/{filename}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            lr_frames.append(file)
            # h1
            h1_name = "0007"
            h1 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h1_name}", self.transform)
            lr_frames.append(h1)
            # h2
            h2_name = "0005"
            h2 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h2_name}", self.transform)
            lr_frames.append(h2)
        elif idx == 1:
            filename = self.get_filename(idx)
            file = f"{self.root_lr}/{filename}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            lr_frames.append(file)
            # h1
            h1_name = self.get_filename(idx - 1)
            h1 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h1_name}", self.transform)
            lr_frames.append(h1)
            # h2
            h2_name = "0007"
            h2 = load_image_from_disk(self.disk_mode, f"{self.root_lr}/{h2_name}", self.transform)
            lr_frames.append(h2)
        else:
            for i in range(self.number_of_frames):
                filename = self.get_filename(idx - i)
                file = f"{self.root_lr}/{filename}"
                file = load_image_from_disk(self.disk_mode, file, self.transform)
                lr_frames.append(file)

        hr_frames = []

        # edge case for STSS val dataset
        if idx == len(self.filenames) - 1:
            filename = self.get_filename(idx)
            file = f"{self.root_hr}/{filename}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            hr_frames.append(file)
            filename = "3599"
            file = f"{self.root_hr}/{filename}"
            file = load_image_from_disk(self.disk_mode, file, self.transform)
            hr_frames.append(file)
        else:
            for i in range(math.ceil(self.number_of_frames / 2)):
                filename = self.get_filename(idx + i)
                file = f"{self.root_hr}/{filename}"
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
            lr_image = FV.to_pil_image(lr_frame)
            axes[i].imshow(lr_image)
            axes[i].set_title(f'LR image {self.get_filename(idx - i)}')

        # Display HR frames
        for i, hr_frame in enumerate(hr_frames):
            hr_image = FV.to_pil_image(hr_frame)
            axes[num_lr_frames + i].imshow(hr_image)
            axes[num_lr_frames + i].set_title(f'HR image {self.get_filename(idx + i)}')

        plt.tight_layout()
        plt.show()

    def get_random_crop_pair(self, lr_frames: list[torch.Tensor], hr_frames: list[torch.Tensor]) \
            -> (list[torch.Tensor], list[torch.Tensor]):
        lr_i, lr_j, lr_h, lr_w = transforms.RandomCrop.get_params(lr_frames[0],
                                                                  output_size=(self.crop_size, self.crop_size))
        hr_i, hr_j, hr_h, hr_w = lr_i * self.scale, lr_j * self.scale, lr_h * self.scale, lr_w * self.scale

        lr_frame_patches = []
        for lr_frame in lr_frames:
            lr_frame_patches.append(FV.crop(lr_frame, lr_i, lr_j, lr_h, lr_w))
        hr_frame_patches = []
        for hr_frame in hr_frames:
            hr_frame_patches.append(FV.crop(hr_frame, hr_i, hr_j, hr_h, hr_w))

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
    # div2k_dataset = SingleImagePair(root="../dataset/DIV2K/train", pattern="x2")
    # div2k_dataset.display_item(0)
    #
    # reds_dataset = MultiImagePair(root="../dataset/Reds/train", scale=4,
    #                               crop_size=96, use_hflip=True, use_rotation=True, digits=8)
    # reds_dataset.display_item(888)

    # matrix_dataset = MultiImagePair(root="../dataset/ue_data_npz/train", scale=2, number_of_frames=3, last_frame_idx=299,
    #                                 crop_size=None, use_hflip=False, use_rotation=False, digits=4, disk_mode=DiskMode.NPZ)
    # matrix_dataset.display_item(42)

    buffers = {"BASE_COLOR": False, "DEPTH": False, "METALLIC": False, "NOV": False, "ROUGHNESS": False,
               "WORLD_NORMAL": False, "WORLD_POSITION": False}
    stss_data = STSSImagePair(root="../dataset/ue_data_npz/test", scale=2, extra=True, history=3, buffers=buffers,
                              last_frame_idx=299,
                              crop_size=512, use_hflip=True, use_rotation=True, digits=4, disk_mode=DiskMode.NPZ)
    for i in range(120):
        stss_data.display_item(i*10)

    # cross_val = STSSCrossValidation(root="../dataset/STSS_val_lewis_png", scale=2, history=2, crop_size=None,
    #                                 use_hflip=False, use_rotation=False)
    # cross_val.display_item(0)

    # cross_val2 = STSSCrossValidation2(root="../dataset/STSS_val_lewis_png", scale=2, number_of_frames=3, crop_size=None,
    #                                   use_hflip=False, use_rotation=False)
    # cross_val2.display_item(len(cross_val2)-1)


if __name__ == '__main__':
    main()
