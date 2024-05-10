import os
from torchvision import transforms
import torch
from tqdm import tqdm
import torchvision.transforms.functional as F

from data.dataloader import SingleImagePair, MultiImagePair, DiskMode
from config import load_yaml_into_config, Config


def get_config_from_pretrained_model(name: str) -> Config:
    config_path = f"configs/{name}.yaml"
    return load_yaml_into_config(config_path)


def generate_directory(path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)


def test() -> None:
    pretrained_model_path = "pretrained_models/flavr_original.pth"
    save_path = "results/test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = pretrained_model_path.split('/')[-1].split('.')[0]
    config = get_config_from_pretrained_model(model_name)
    print(config)

    # Loading model
    model = config.model.to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    path = "dataset/ue_data/test"

    test_dataset = MultiImagePair(root=path, number_of_frames=4, last_frame_idx=299,
                         transform=transforms.ToTensor(), crop_size=None, scale=2,
                         use_hflip=False, use_rotation=False, digits=4, disk_mode=DiskMode.CV2)

    counter = 0
    for idx in tqdm(range(len(test_dataset)), "Generating sequence.."):
        if counter % 2 == 1:
            # print("skipped\n")
            counter += 1
            continue
        counter += 1
        filename = test_dataset.get_filename(idx)
        subfolder = test_dataset.get_path(idx).split("/")[0] # we want to retrieve the sub folder of the val sequences
        generate_directory(f"{save_path}/{subfolder}")
        # print(f"Filename: {filename}\n")
        lr_image, hr_image = test_dataset.__getitem__(idx)
        with torch.no_grad():
            lr_image = lr_image.unsqueeze(0)
            lr_image = lr_image.to(device)
            output_image = model(lr_image)
            output_image = torch.clamp(output_image, min=0.0, max=1.0)
        # Safe generated images into a folder
        output_images = torch.unbind(output_image, 1)
        for i in range(len(output_images)):
            frame = F.to_pil_image(output_images[i].squeeze(0))
            # generate the right filename
            filename = int(filename) + i
            # print(f"Save file at {save_path}/{subfolder}/{filename:04d}.png")
            frame.save(f"{save_path}/{subfolder}/{filename:04d}.png")


def main() -> None:
    test()


if __name__ == '__main__':
    main()
