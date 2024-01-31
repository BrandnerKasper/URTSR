import random
import timeit
from torch.utils.data import Dataset
from PIL import Image
import os
import time
from torchvision import transforms


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


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, pattern: str = None):
        self.root_hr = os.path.join(root, "HR")
        self.root_lr = os.path.join(root, "LR")
        self.transform = transform
        self.pattern = pattern
        self.filenames = init_filenames(self.root_hr, self.root_lr, self.pattern)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        common_filename = self.filenames[idx]
        hr_path = os.path.join(self.root_hr, common_filename + ".png")
        if self.pattern:
            lr_path = os.path.join(self.root_lr, common_filename + self.pattern + ".png")
        else:
            lr_path = os.path.join(self.root_lr, common_filename + ".png")

        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

    def get_filename(self, idx):
        path = self.filenames[idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename


def main() -> None:
    # measuring time of utils fcts
    transform = transforms.ToTensor()
    root_hr = "dataset/matrix/HR"
    root_lr = "dataset/matrix/LR"
    pattern = ""
    # Use a lambda function to pass the function with its arguments to timeit
    execution_time_filenames = timeit.timeit(lambda: init_filenames(root_hr, root_lr, pattern), number=1)
    print(f"Execution time of filenames: {execution_time_filenames} seconds")


if __name__ == '__main__':
    main()
