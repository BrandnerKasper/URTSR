import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess dataset .png files into sub-images by factor and converting into .pt files.")

    parser.add_argument('--folder_path', type=str, default='dataset/DIV2K/train/LR',
                        help="Path to the dataset folder containing .png files")
    parser.add_argument('--factor', type=int, default=2,
                        help="Dividing the images into sub-images by factor times 2. "
                             "Ex: factor 2 = 4 sub-images, 3 = 9 sub-images..")
    parser.add_argument('--safe_folder_path', type=str, default='dataset/pre',
                        help="Path to the folder in which the sub-images should be saved")
    parser.add_argument('--file_format', choices=['pt', 'npz'], default='pt',
                        help="Output file format ('pt' or 'npz').")

    args = parser.parse_args()
    return args


def convert_image_to_pt_file(path: str) -> None:
    transform = transforms.ToTensor()
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img)
    pt_path = path.split(".")[0]
    torch.save(img_tensor, f"{pt_path}.pt")


def convert_image_to_npz_file(path: str) -> None:
    transform = transforms.ToTensor()
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img).numpy()
    np_path = path.split(".")[0]
    np.savez_compressed(f"{np_path}.npz", img_tensor)


def create_pt_files(folder_path: str) -> None:
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            convert_image_to_pt_file(f"{folder_path}/{filename}")


def create_compressed_npz_files(folder_path: str) -> None:
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            convert_image_to_npz_file(f"{folder_path}/{filename}")


def create_sub_images(folder_path: str, factor: int, safe_path: str) -> None:
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            img = Image.open(f"{folder_path}/{filename}").convert('RGB')
            # Get image size
            width, height = img.size

            # Adjust width and height to be divisible by factor
            width = width - (width % factor)
            height = height - (height % factor)

            # Calculate subimage width and height
            subimage_width = width // factor
            subimage_height = height // factor

            # Generate subimages
            n = 1
            for i in range(0, width, subimage_width):
                for j in range(0, height, subimage_height):
                    # Define the region to crop
                    box = (i, j, i + subimage_width, j + subimage_height)

                    # Crop the subimage
                    subimage = img.crop(box)

                    # Save the PIL image as a .png file
                    name = filename.split(".")[0]
                    # If name contains "x2" in their name
                    if "x2" in name:
                        # Adjust name so that "namex2" is turned into "name_{n}x2"
                        name = name.replace("x2", "")
                        name = f"{name}_{n}x2"
                    else:
                        # Else just attach _{n} to name
                        name = f"{name}_{n}"

                    # Save the file in the new folder
                    subimage.save(f"{safe_path}/{name}.png", format='PNG')
                    n += 1


def main() -> None:
    args = parse_arguments()
    folder_path = args.folder_path
    factor = args.factor
    safe_folder_path = args.safe_folder_path
    file_format = args.file_format

    # Create safe folder path if it does not exist
    if not os.path.exists(safe_folder_path):
        os.makedirs(safe_folder_path)
        print(f"New directory created: {safe_folder_path}")
    else:
        print(f"Directory {safe_folder_path} already exists.")

    # Create sub images
    create_sub_images(folder_path, factor, safe_folder_path)

    # Create pt or npz files
    match file_format:
        case 'pt':
            create_pt_files(safe_folder_path)
        case 'npz':
            create_compressed_npz_files(safe_folder_path)


if __name__ == "__main__":
    main()
