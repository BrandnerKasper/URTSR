import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataloader import CustomDataset
from config import load_yaml_into_config


def main() -> None:
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_yaml_into_config("config.yaml")
    batch_size = config.batch_size
    epochs = config.epochs
    num_workers = config.number_workers
    crop_size = config.crop_size
    scale = config.scale
    start_decay_epoch = config.start_decay_epoch

    # Model details
    model = config.model.to(device)
    criterion = config.criterion
    optimizer = config.optimizer
    scheduler = config.scheduler

    # Loading and preparing data
    transform = transforms.ToTensor()
    # train data
    train_data_path = config.train_dataset
    train_dataset = CustomDataset(root=train_data_path, transform=transform, pattern="x2", crop_size=crop_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val data
    val_data_path = config.val_dataset
    val_dataset = CustomDataset(val_data_path, transform=transform, pattern="x2", crop_size=crop_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training & Validation Loop
    for epoch in tqdm(range(epochs), desc='Train & Validate', dynamic_ncols=True):
        # train loop
        total_loss = 0.0
        for lr_image, hr_image in tqdm(train_loader, desc=f'Training, Epoch {epoch+1}/{epochs}', dynamic_ncols=True):
            input, target = lr_image.to(device), hr_image.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # scheduler update if we have one
        if scheduler is not None:
            if epoch > start_decay_epoch:
                scheduler.step()
            average_loss = total_loss / len(train_loader)
            print("\n")
            print(f"Loss: {average_loss:.4f}\n")

        # val loop
        if (epoch+1) % 10 != 0:
            continue
        total_metrics = (0, 0)
        for lr_image, hr_image in tqdm(val_loader, desc=f"Validation, Epoch {epoch+1}/{epochs}", dynamic_ncols=True):
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            with torch.no_grad():
                output_image = model(lr_image)
                output_image = torch.clamp(output_image, min=0.0, max=1.0)
            metrics = utils.calculate_metrics(hr_image, output_image)
            total_metrics = tuple(x + y for x, y in zip(total_metrics, metrics))

        average_metric = tuple(x / len(val_loader) for x in total_metrics)
        print("\n")
        print(f"PSNR: {average_metric[0]:.2f} db, SSIM: {average_metric[1]:.4f}\n")

    # Save trained models
    filename = config.filename
    model_str = f"{filename}_e{epochs}_x{scale}_bs{batch_size}_ps{crop_size}"
    model_path = "pretrained_models/" + model_str + ".pth"
    torch.save(model.state_dict(), model_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
