import torch
from model.srcnn import SRCNN
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def main() -> None:
    model_path = "pretrained_models/srcnn_model.pth"

    lr_image = Image.open("dataset/test/LR/0010x2.png").convert('RGB')
    hr_image = Image.open("dataset/test/HR/0010.png").convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    transformed_lr_tensor = transform(lr_image).unsqueeze(0)
    transformed_hr_tensor = transform(hr_image).unsqueeze(0)

    model = SRCNN()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(transformed_lr_tensor)

    lr_image = F.to_pil_image(transformed_lr_tensor.squeeze(0).clamp(0.0, 1.0))
    hr_image = F.to_pil_image(transformed_hr_tensor.squeeze(0).clamp(0.0, 1.0))
    srcnn_image = F.to_pil_image(output.squeeze(0).clamp(0.0, 1.0))

    # Display the original and transformed images side by side
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 30))

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
