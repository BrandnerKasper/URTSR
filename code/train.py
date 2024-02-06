import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataloader import CustomDataset
from model.srcnn import SRCNN
from model.subpixel import SubPixelNN
from model.extraNet import ExtraNet


def main() -> None:
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    batch_size = 8
    epochs = 150
    num_workers = 8
    patch_size = 256
    scale = 2
    beta1, beta2 = 0.9, 0.999
    decay_rate = 20

    # Model details
    model = ExtraNet(scale_factor=scale).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler = CosineAnnealingLR(optimizer, T_max=decay_rate)

    # Loading and preparing data
    transform = transforms.ToTensor()
    # train data
    train_dataset = CustomDataset(root='dataset/DIV2K/train', transform=transform, pattern="x2", patch_size=patch_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val data
    val_dataset = CustomDataset("dataset/DIV2K/val", transform=transform, pattern="x2", patch_size=patch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training  & Validation Loop
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

        scheduler.step()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")

        # val loop
        total_metrics = (0, 0)
        for lr_image, hr_image in tqdm(val_loader, desc=f"Validation, Epoch {epoch+1}/{epochs}", dynamic_ncols=True):
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            with torch.no_grad():
                output_image = model(lr_image)
                output_image = torch.clamp(output_image, min=0.0, max=1.0)
            metrics = utils.calculate_metrics(hr_image, output_image)
            total_metrics = tuple(x + y for x, y in zip(total_metrics, metrics))

        average_metric = tuple(x / len(val_loader) for x in total_metrics)
        print(f"Epoch [{epoch + 1}/{epochs}], PSNR: {average_metric[0]:.2f} db, SSIM: {average_metric[1]:.4f}")

    # Save trained model
    model_str = f"extranet_e{epochs}_x{scale}_bs{batch_size}_ps{patch_size}_a{(beta1, beta2)}_c_{decay_rate}"
    model_path = "pretrained_models/" + model_str + ".pth"
    torch.save(model.state_dict(), model_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
