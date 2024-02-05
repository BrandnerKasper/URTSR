import torch
from model.srcnn import SRCNN
from model.subpixel import SubPixelNN
from model.extraNet import ExtraNet
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from dataloader import CustomDataset
import utils


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading and preparing data
    transform = transforms.ToTensor()
    test_dataset = CustomDataset(root='dataset/Set14', transform=transform, pattern="x2")

    # Load model
    model_path = "pretrained_models/extranet.pth"
    model = ExtraNet(2).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    for lr_image, hr_image in test_dataset:
        lr_image, hr_image = lr_image.to(device), hr_image.to(device)
        lr_image_model = utils.pad_to_divisible(lr_image.unsqueeze(0), 8)
        with torch.no_grad():
            output = model(lr_image_model).squeeze(0)
            output = utils.pad_or_crop_to_target(output, hr_image)
            output = torch.clamp(output, min=0.0, max=1.0)
            print(f"output: {output}")
        lr_image = F.to_pil_image(lr_image)
        hr_image = F.to_pil_image(hr_image)
        output = F.to_pil_image(output)

        # Display the original and transformed images side by side
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 3, figsize=(50, 50))

        # Display the original image
        axes[0].imshow(lr_image)
        axes[0].set_title('LR image')
        axes[1].imshow(output)
        axes[1].set_title('Network image')
        axes[2].imshow(hr_image)
        axes[2].set_title('HR image')

        plt.show()


if __name__ == '__main__':
    main()
