import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from data.dataloader import SingleImagePair, MultiImagePair
from config import load_yaml_into_config, create_comment_from_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SR network based on a config file.")
    parser.add_argument('file_path', type=str, nargs='?', default='configs/flavr.yaml', help="Path to the config file")
    args = parser.parse_args()
    return args


def save_model(filename: str, model: nn.Module) -> None:
    model_path = "pretrained_models/" + filename + ".pth"
    torch.save(model.state_dict(), model_path)


def train(filepath: str):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = load_yaml_into_config(filepath)
    print(config)
    filename = config.filename.split('.')[0]
    writer = SummaryWriter(filename_suffix=create_comment_from_config(config), comment=filename) #log_dir="runs"
    # Hyperparameters
    batch_size = config.batch_size
    epochs = config.epochs
    num_workers = config.number_workers
    start_decay_epoch = config.start_decay_epoch

    # Model details
    model = config.model.to(device)
    criterion = config.criterion
    optimizer = config.optimizer
    scheduler = config.scheduler

    # Loading and preparing data
    # train data
    train_dataset = config.train_dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # val data
    val_dataset = config.val_dataset
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training & Validation Loop
    for epoch in tqdm(range(epochs), desc='Train & Validate', dynamic_ncols=True):
        # train loop
        total_loss = 0.0
        for lr_image, hr_image in tqdm(train_loader, desc=f'Training, Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            optimizer.zero_grad()
            output = model(lr_image)
            loss = criterion(output, hr_image)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # scheduler update if we have one
        if scheduler is not None:
            if epoch > start_decay_epoch:
                scheduler.step()
        # Loss
        average_loss = total_loss / len(train_loader)
        print("\n")
        print(f"Loss: {average_loss:.4f}\n")
        # Log loss to TensorBoard
        writer.add_scalar('Train/Loss', average_loss, epoch)

        # val loop
        if (epoch + 1) % 10 != 0:
            continue

        if isinstance(val_dataset, SingleImagePair):
            total_metrics = utils.Metrics([0], [0])
        else:
            total_metrics = utils.Metrics([0, 0], [0, 0]) # TODO: abstract number of values based on second dim of tensor [8, 2, 3, 1920, 1080]

        for lr_image, hr_image in tqdm(val_loader, desc=f"Validation, Epoch {epoch + 1}/{epochs}", dynamic_ncols=True):
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            with torch.no_grad():
                output_image = model(lr_image)
                output_image = torch.clamp(output_image, min=0.0, max=1.0)
            # Calc PSNR and SSIM
            metrics = utils.calculate_metrics(hr_image, output_image)
            total_metrics += metrics
        # PSNR & SSIM
        average_metric = total_metrics / len(val_loader)
        print("\n")
        print(average_metric)
        # Log PSNR & SSIM to TensorBoard
        # PSNR
        for i in range(average_metric.psnr_values):
            writer.add_scalar(f"Validation/PSNR_{i}", average_metric.psnr_values[i], epoch)
        # SSIM
        for i in range(average_metric.ssim_values):
            writer.add_scalar(f"Validation/SSIM_{i}", average_metric.ssim_values[i], epoch)

    # End Log
    writer.close()
    # Save trained models
    save_model(filename, model)


def main() -> None:
    args = parse_arguments()
    file_path = args.file_path
    train(file_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
