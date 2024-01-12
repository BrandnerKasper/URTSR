import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import CustomDataset
from model.srcnn import SRCNN
from model.subpixel import SubPixelNN


def main() -> None:
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    batch_size = 1
    epochs = 5
    num_workers = 16

    # Model details
    # model = SRCNN().to(device)
    model = SubPixelNN(scale_factor=2).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loading and preparing data
    transform = transforms.ToTensor()

    train_dataset = CustomDataset(root='dataset/train', transform=transform, pattern="x2")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training Loop
    for epoch in tqdm(range(epochs), desc='Training', dynamic_ncols=True):
        total_loss = 0.0
        for lr_image, hr_image in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', dynamic_ncols=True):
            input, target = lr_image.to(device), hr_image.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'pretrained_models/subpnn_model.pth')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
