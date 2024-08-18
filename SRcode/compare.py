import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm
from utils import utils
from lpips import lpips


def load_image_from_disk(path: str, transform=transforms.ToTensor(),
                         read_mode=cv2.IMREAD_UNCHANGED) -> torch.Tensor:
    # Load the image with CV2
    img = cv2.imread(f"{path}.png", read_mode)
    # Convert BGR to RGB
    if read_mode == cv2.IMREAD_UNCHANGED:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)


class Compare(Dataset):
    def __init__(self, root: str, compare: str, transform=transforms.ToTensor(), sequence_length: int = 300):
        self.root_gt = os.path.join(root, "gt")
        self.root_com = os.path.join(root, compare)
        self.transform = transform
        self.sequence_length = sequence_length
        self.filenames, self.sequence_names = self.init_filenames()

    def init_filenames(self) -> (list[list[str]], list[str]):
        filenames = []
        sequence_names = []
        for directory in os.listdir(self.root_gt):
            sequence_names.append(directory)
            sub_filenames = []
            for file in os.listdir(os.path.join(self.root_gt, directory)):
                file = os.path.splitext(file)[0]
                sub_filenames.append(file)
            filenames.append(sorted(set(sub_filenames)))
        return filenames, sequence_names

    def __len__(self) -> int:
        length = 0
        for sequence in self.filenames:
            length += len(sequence)
        return length

    def get(self, idx) -> str:
        sub_idx = math.floor(idx / self.sequence_length)
        idx = idx - self.sequence_length * sub_idx
        return f"{self.sequence_names[sub_idx]}/{self.filenames[sub_idx][idx]}"

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        common_filename = self.get(idx)

        com_path = os.path.join(self.root_com, common_filename)
        gt_path = os.path.join(self.root_gt, common_filename)

        com_frame = load_image_from_disk(com_path, self.transform)
        gt_frame = load_image_from_disk(gt_path, self.transform)

        return com_frame, gt_frame

    def get_filename(self, idx: int) -> str:
        path = self.get(idx)
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        return filename


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compare_dataset = Compare("dataset/Ref", "FSR")
    compare_loader = DataLoader(dataset=compare_dataset, batch_size=1, shuffle=False, num_workers=12)
    eval_alex_model = lpips.LPIPS(net='alex')#.cuda()

    total_metrics = utils.Metrics(0, 0, 0)
    sequence_length = compare_dataset.sequence_length
    sequence_names = compare_dataset.sequence_names
    count = 0
    sequences = 0
    metrics = {}

    for com, gt in tqdm(compare_loader, desc=f"Compare", dynamic_ncols=True):
        # com = com.to(device)
        # gt = gt.to(device)

        metric = utils.calculate_metrics(gt, com, eval_alex_model)
        total_metrics += metric
        if count == sequence_length - 1:
            metrics[sequence_names[sequences]] = total_metrics / sequence_length
            total_metrics = utils.Metrics()
            count = 0
            sequences += 1
        else:
            count += 1

        # Printing
    for k, v in metrics.items():
        print(f"Sequence {k}: {v}")


if __name__ == '__main__':
    main()
