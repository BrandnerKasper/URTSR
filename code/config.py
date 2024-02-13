import yaml
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from typing import Optional

from models.srcnn import SRCNN
from models.subpixel import SubPixelNN
from models.extraNet import ExtraNet


def create_yaml(filename: str, model: str, epochs: int, scale: int, batch_size: int, crop_size: int,  number_workers: int,
                learning_rate: float, criterion: str, optimizer: dict, scheduler: dict,
                train_dataset: str, val_dataset: str):

    data = {
        "MODEL": model,
        "EPOCHS": epochs,
        "SCALE": scale,
        "BATCH_SIZE": batch_size,
        "CROP_SIZE": crop_size,
        "NUMBER_WORKERS": number_workers,
        "LEARNING_RATE": learning_rate,
        "CRITERION": criterion,
        "OPTIMIZER": optimizer,
        "SCHEDULER": scheduler,
        "TRAIN_DATASET": train_dataset,
        "VAL_DATASET": val_dataset
    }

    yaml_text = yaml.dump(data, sort_keys=False)
    file = open(f"{filename}.yaml", "w")
    file.write(yaml_text)
    file.close()
    print(yaml.load(yaml_text, Loader=yaml.Loader))
    return yaml_text


def init_model(model_name: str, scale: int) -> nn.Module:
    match model_name:
        case "SRCNN":
            return SRCNN(scale=scale)
        case "SubPixel":
            return SubPixelNN(scale=scale)
        case "ExtraNet":
            return ExtraNet(scale=scale)
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


def init_scheduler(scheduler_data: dict) -> Optional[lr_scheduler.LRScheduler]:
    if scheduler_data is None:
        return None

    scheduler_name = scheduler_data["NAME"]

    match scheduler_name:
        case "Cosine":

            return


class Config:
    def __init__(self, filename: str, model: str, epochs: int, scale: int, batch_size: int, crop_size: int,  number_workers: int,
                learning_rate: float, criterion: str, optimizer: dict, scheduler: dict,
                train_dataset: str, val_dataset: str):
        self.filename: str = filename
        self.model: nn.Module = init_model(model, scale)
        self.epochs: int = epochs
        self.scale: int = scale
        self.batch_size: int = batch_size
        self.crop_size: int = crop_size
        self.number_workers: int = number_workers
        self.learning_rate: float = learning_rate
        self.criterion: _Loss = init_criterion(criterion)
        self.optimizer: optim.Optimizer = init_optimizer(optimizer, self.model, self.learning_rate)
        self.scheduler = init_scheduler(scheduler)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


def main() -> None:
    optimizer = {
        "NAME": "Adam",
        "BETA1": 0.9,
        "BETA2": 0.99,
    }

    scheduler = {
        "NAME": "Cosine",
        "START_DECAY_EPOCH": 20
    }

    create_yaml("config", "ExtraSS", 100, 2, 1, 0, 8,
                0.001, "L1", optimizer, scheduler,
                "DIV2K/train", "DIV2K/val")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
