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
    parser.add_argument('folder_path', type=str, nargs='?', default='dataset/Set5',
                        help="Path to the dataset folder containing .png files")
    parser.add_argument('factor', type=int, default=2,
                        help="Dividing the images into sub-images by factor times 2. "
                             "Ex: factor 2 = 4 sub- images, 3 = 9 sub-images..")
    parser.add_argument('file_format', choices=['pt', 'npz'], default='pt', help="Output file format ('pt' or 'npz').")
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


def preprocess_images_to_pt_files(folder_path: str) -> None:
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            convert_image_to_pt_file(f"{folder_path}/{filename}")


def preprocess_images_to_compressed_npz_files(folder_path: str) -> None:
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            convert_image_to_npz_file(f"{folder_path}/{filename}")


def preprocess_images_to_sub_images(folder_path: str, factor: int, file_format: str) -> None:
    # Iterate through ALL subfolders inside the folder path and convert every image inside the subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in tqdm(files, desc="Preprocessing..", unit="image"):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(root, filename)).convert('RGB')
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

                        # Save the file in a new folder in dataset called "sub_Original name of folder"
                        save_folder = os.path.join(folder_path, f"sub_{dirs}")
                        os.makedirs(save_folder, exist_ok=True)
                        save_path = os.path.join(save_folder, f"{name}.png")

                        subimage.save(save_path, format='PNG')

                        # Choose conversion format
                        if file_format == 'pt':
                            # Save images as .pt file
                            convert_image_to_pt_file(f"{save_path}.png")
                        elif file_format == 'npz':
                            # Save images as .npz file
                            convert_image_to_npz_file(f"{save_path}.png")
                        else:
                            print(f"No conversion format for {save_path}!", file=sys.stderr)

                        n += 1


def main() -> None:
    args = parse_arguments()
    folder_path = args.folder_path
    factor = args.factor
    file_format = args.file_format

    preprocess_images_to_sub_images(folder_path, factor, file_format)


if __name__ == "__main__":
    main()
