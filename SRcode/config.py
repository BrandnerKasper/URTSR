import yaml
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms

from typing import Optional

from models.basemodel import BaseModel
from models.srcnn import SRCNN
from models.subpixel import SubPixelNN
from models.extraNet import ExtraNet
from models.flavr import Flavr

from data.dataloader import SingleImagePair, MultiImagePair


def create_yaml(filename: str, model: str, epochs: int, scale: int, batch_size: int,
                crop_size: int,  use_hflip: bool, use_rotation: bool,
                number_workers: int, learning_rate: float, criterion: str, optimizer: dict, scheduler: dict,
                dataset: str):

    data = {
        "MODEL": model,
        "EPOCHS": epochs,
        "SCALE": scale,
        "BATCH_SIZE": batch_size,
        "CROP_SIZE": crop_size,
        "USE_HFLIP": use_hflip,
        "USE_ROTATION": use_rotation,
        "NUMBER_WORKERS": number_workers,
        "LEARNING_RATE": learning_rate,
        "CRITERION": criterion,
        "OPTIMIZER": optimizer,
        "SCHEDULER": scheduler,
        "DATASET": dataset,
    }

    yaml_text = yaml.dump(data, sort_keys=False)
    file = open(f"{filename}.yaml", "w")
    file.write(yaml_text)
    file.close()
    print(yaml.load(yaml_text, Loader=yaml.Loader))
    return yaml_text


def init_model(model_name: str, scale: int) -> BaseModel:
    match model_name:
        case "SRCNN":
            return SRCNN(scale=scale)
        case "SubPixel":
            return SubPixelNN(scale=scale)
        case "ExtraNet":
            return ExtraNet(scale=scale)
        case "Flavr":
            return Flavr(scale=scale)
        case _:
            raise ValueError(f"The model '{model_name}' is not a valid model.")


def init_criterion(criterion_name: str) -> _Loss:
    match criterion_name:
        case "L1":
            return nn.L1Loss()
        case _:
            raise ValueError(f"The criterion '{criterion_name}' is not a valid criterion.")


def init_optimizer(optimizer_data: dict, model: nn.Module, learning_rate: float) -> optim.Optimizer:
    optimizer_name = optimizer_data["NAME"]

    match optimizer_name:
        case "Adam":
            beta1, beta2 = optimizer_data["BETA1"], optimizer_data["BETA2"]
            return optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        case _:
            raise ValueError(f"The optimizer '{optimizer_name}' is not a valid optimizer.")


def init_scheduler(scheduler_data: dict, optimizer: optim.Optimizer, epochs: int) -> Optional[lr_scheduler.LRScheduler]:
    scheduler_name = scheduler_data["NAME"]

    match scheduler_name:
        case "Cosine":
            min_learning_rate = scheduler_data["MIN_LEARNING_RATE"]
            return lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=min_learning_rate)
        case None:
            return None
        case _:
            raise ValueError(f"The scheduler '{scheduler_name}' is not a valid scheduler.")


def init_dataset(name: str, crop_size: int, use_hflip: bool, use_rotation: bool) -> (Dataset, Dataset):
    root = f"dataset/{name}"
    match name:
        case "DIV2K":
            train = SingleImagePair(root=f"{root}/train", transform=transforms.ToTensor(), pattern="x2",
                                    crop_size=crop_size, scale=2,
                                    use_hflip=use_hflip, use_rotation=use_rotation)
            val = SingleImagePair(root=f"{root}/val", transform=transforms.ToTensor(), pattern="x2",
                                  crop_size=None, scale=2,
                                  use_hflip=False, use_rotation=False)
            return train, val
        case "Reds": #TODO abstract number_of_frames
            train = MultiImagePair(root=f"{root}/train", number_of_frames=4, last_frame_idx=100,
                                  transform=transforms.ToTensor(), crop_size=crop_size, scale=4,
                                  use_hflip=use_hflip, use_rotation=use_rotation)
            val = MultiImagePair(root=f"{root}/val", number_of_frames=4, last_frame_idx=100,
                                  transform=transforms.ToTensor(), crop_size=None, scale=4,
                                  use_hflip=False, use_rotation=False, digits=8)
            return train, val
        case "Matrix":
            train = MultiImagePair(root=f"{root}/train", number_of_frames=4, last_frame_idx=1499,
                                   transform=transforms.ToTensor(), crop_size=crop_size, scale=2,
                                   use_hflip=use_hflip, use_rotation=use_rotation)
            val = MultiImagePair(root=f"{root}/val", number_of_frames=4, last_frame_idx=599,
                                   transform=transforms.ToTensor(), crop_size=crop_size, scale=2,
                                   use_hflip=use_hflip, use_rotation=use_rotation, digits=4)
            return train, val
        case _:
            raise ValueError(f"The dataset '{name}' is not a valid dataset.")


class Config:
    def __init__(self, filename: str, model: str, epochs: int, scale: int, batch_size: int,
                 crop_size: int, use_hflip: bool, use_rotation: bool, number_workers: int,
                 learning_rate: float, criterion: str, optimizer: dict, scheduler: dict,
                 start_decay_epoch: Optional[int],
                 dataset: str):
        self.filename: str = filename
        self.model: BaseModel = init_model(model, scale)
        self.epochs: int = epochs
        self.scale: int = scale
        self.batch_size: int = batch_size
        self.crop_size: int = crop_size
        self.use_hflip: bool = use_hflip
        self.use_rotation: bool = use_rotation
        self.number_workers: int = number_workers
        self.learning_rate: float = learning_rate
        self.criterion: _Loss = init_criterion(criterion)
        self.optimizer: optim.Optimizer = init_optimizer(optimizer, self.model, self.learning_rate)
        self.scheduler: Optional[lr_scheduler.LRScheduler] = init_scheduler(scheduler, self.optimizer, self.epochs)
        self.start_decay_epoch: Optional[int] = start_decay_epoch
        self.dataset: str = dataset
        self.train_dataset, self.val_dataset = init_dataset(dataset, crop_size, use_hflip, use_rotation)

    def __str__(self):
        return f"Config:\n" \
               f"  Filename: {self.filename}\n" \
               f"  Model: {self.model.__class__.__name__}\n" \
               f"  Epochs: {self.epochs}\n" \
               f"  Scale: {self.scale}\n" \
               f"  Batch Size: {self.batch_size}\n" \
               f"  Crop Size: {self.crop_size}\n" \
               f"  Use HFlip: {self.use_hflip}\n"\
               f"  Use Rotation: {self.use_rotation}\n" \
               f"  Number of Workers: {self.number_workers}\n" \
               f"  Learning Rate: {self.learning_rate}\n" \
               f"  Criterion: {self.criterion.__class__.__name__}\n" \
               f"  Optimizer: {self.optimizer.__class__.__name__}\n" \
               f"  Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}\n" \
               f"  Start Decay Epoch: {self.start_decay_epoch if self.start_decay_epoch else 'None'}\n" \
               f"  Dataset: {self.dataset}"


def test_yaml_creation() -> None:
    optimizer = {
        "NAME": "Adam",
        "BETA1": 0.9,
        "BETA2": 0.99,
    }
    scheduler = {
        "NAME": "Cosine",
        "MIN_LEARNING_RATE": 1e-6,
        "START_DECAY_EPOCH": 20,
    }
    create_yaml("config", "ExtraNet", 150, 2, 1,
                256, True, True, 8,
                0.001, "L1", optimizer, scheduler,
                "DIV2K")


def load_yaml_into_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
        model_name = config_dict["MODEL"]
        epochs = config_dict["EPOCHS"]
        scale = config_dict["SCALE"]
        batch_size = config_dict["BATCH_SIZE"]
        crop_size = config_dict["CROP_SIZE"]
        use_hflip = config_dict["USE_HFLIP"]
        use_rotation = config_dict["USE_ROTATION"]
        number_workers = config_dict["NUMBER_WORKERS"]
        learning_rate = config_dict["LEARNING_RATE"]
        criterion = config_dict["CRITERION"]
        optimizer = config_dict["OPTIMIZER"]
        scheduler = config_dict["SCHEDULER"]
        start_decay_epoch = scheduler["START_DECAY_EPOCH"]
        dataset = config_dict["DATASET"]
        filename = file_path.split('/')[-1].split('.')[0]

    return Config(filename, model_name, epochs, scale, batch_size,
                  crop_size, use_hflip, use_rotation, number_workers,
                  learning_rate, criterion, optimizer, scheduler, start_decay_epoch,
                  dataset)


def main() -> None:
    # test_yaml_creation()
    yaml_config = load_yaml_into_config("configs/extranet.yaml")
    print(yaml_config)


if __name__ == '__main__':
    main()
