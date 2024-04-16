import yaml
from torchvision import transforms
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

from data.dataloader import SingleImagePair, MultiImagePair
from SRcode.config import load_yaml_into_config, Config


def get_config_from_pretrained_model(name: str) -> Config:
    config_path = f"configs/{name}.yaml"
    return load_yaml_into_config(config_path)


def test() -> None:
    pretrained_model_path = "pretrained_models/flavr.pth"
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
    path = "dataset/matrix/val"

    test_dataset = MultiImagePair(root=path, number_of_frames=4, last_frame_idx=599,
                         transform=transforms.ToTensor(), crop_size=None, scale=2,
                         use_hflip=False, use_rotation=False, digits=4)

    counter = 0
    img_counter = 3 # we need history frames (number of frames -1)
    for idx in tqdm(range(len(test_dataset)), "Generating sequence.."):
        if counter % 2 == 1:
            print("skipped\n")
            counter += 1
            continue
        counter += 1
        filename = test_dataset.get_filename(idx)
        print(f"Filename: {filename}\n")
        lr_image, hr_image = test_dataset.__getitem__(idx)
        with torch.no_grad():
            lr_image = lr_image.unsqueeze(0)
            lr_image = lr_image.to(device)
            output_image = model(lr_image)
            output_image = torch.clamp(output_image, min=0.0, max=1.0)
        # Safe generated images into a folder
        output_images = torch.unbind(output_image, 1)
        for frame in output_images:
            frame = F.to_pil_image(frame.squeeze(0))
            frame.save(f"{save_path}/{img_counter:04d}.png")
            img_counter += 1


def main() -> None:
    test()


if __name__ == '__main__':
    main()
