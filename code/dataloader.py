from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_hr = root + "/HR"
        self.root_lr = root + "/LR"
        self.transform = transform
        self.hr_images = os.listdir(self.root_hr)
        self.lr_images = os.listdir(self.root_lr)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.root_hr, self.hr_images[idx])
        lr_path = os.path.join(self.root_lr, self.lr_images[idx])

        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image
