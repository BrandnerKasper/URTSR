import torch
from model.srcnn import SRCNN
from model.subpixel import SubPixelNN
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from dataloader import CustomDataset


def main() -> None:
    model_path = "pretrained_models/subpnn_model_e100.pth"

    # Loading and preparing data
    transform = transforms.ToTensor()
    test_dataset = CustomDataset(root='dataset/test', transform=transform)

    model = SubPixelNN(2)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    for lr_image, hr_image in test_dataset:
        with torch.no_grad():
            output = model(lr_image.unsqueeze(0))

        lr_image = F.to_pil_image(lr_image)
        hr_image = F.to_pil_image(hr_image)
        srcnn_image = F.to_pil_image(output.squeeze(0))

        # Display the original and transformed images side by side
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 3, figsize=(50, 50))

        # Display the original image
        axes[0].imshow(lr_image)
        axes[0].set_title('LR image')
        axes[1].imshow(srcnn_image)
        axes[1].set_title('Srcnn image')
        axes[2].imshow(hr_image)
        axes[2].set_title('HR image')

        plt.show()


if __name__ == '__main__':
    main()
