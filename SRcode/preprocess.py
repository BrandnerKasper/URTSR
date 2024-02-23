import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def preprocess_images_to_pt_files(folder_path: str) -> None:
    transform = transforms.ToTensor()
    for filename in tqdm(os.listdir(folder_path), "Preprocessing.."):
        if filename.endswith(".png"):
            img = Image.open(f"{folder_path}/{filename}")
            img_tensor = transform(img)
            name = filename.split(".")[0]
            torch.save(img_tensor, f"{folder_path}/{name}.pt")


def main() -> None:
    path = "dataset/DIV2K/val/LR"
    preprocess_images_to_pt_files(path)


if __name__ == "__main__":
    main()
