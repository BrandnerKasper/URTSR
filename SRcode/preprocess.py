import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset .png files into .pt files")
    parser.add_argument('folder_path', type=str, nargs='?', default='dataset/Set5/LR',
                        help="Path to the dataset folder containing .png files")
    parser.add_argument('file_format', choices=['pt', 'npz'], help="Output file format ('pt' or 'npz').")
    args = parser.parse_args()
    return args


def preprocess_images_to_pt_files(folder_path: str) -> None:
    transform = transforms.ToTensor()
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            img = Image.open(f"{folder_path}/{filename}").convert('RGB')
            img_tensor = transform(img)
            name = filename.split(".")[0]
            torch.save(img_tensor, f"{folder_path}/{name}.pt")


def preprocess_images_to_compressed_npz_files(folder_path: str) -> None:
    transform = transforms.ToTensor()
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            img = Image.open(f"{folder_path}/{filename}").convert('RGB')
            img_tensor = transform(img).numpy()
            name = filename.split(".")[0]
            np.savez_compressed(f"{folder_path}/{name}.npz", img_tensor)


def main() -> None:
    args = parse_arguments()
    folder_path = args.folder_path
    file_format = args.file_format
    match file_format:
        case 'pt':
            preprocess_images_to_pt_files(folder_path)
        case 'npz':
            preprocess_images_to_compressed_npz_files(folder_path)
        case _:
            print(f"No valid file format {file_format}!", file=sys.stderr)


if __name__ == "__main__":
    main()
