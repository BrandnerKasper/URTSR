import cv2
import torch
from torchvision import transforms
import numpy as np
import os

from tqdm import tqdm


def convert_hr_frame_to_npz_file(path: str, filename: str, safe_path: str) -> None:
    transform = transforms.ToTensor()

    file = os.path.splitext(filename)[0]

    # Load lr frame
    hr_frame = cv2.imread(f"{path}/{file}.png", cv2.IMREAD_UNCHANGED)
    hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
    hr_tensor = transform(hr_frame).numpy()

    filename = filename.replace(".png", "")
    np.savez_compressed(f"{safe_path}/{filename}.npz", hr_tensor)


def convert_lr_frames_to_npz_file(path: str, filename: str, safe_path: str) -> None:
    transform = transforms.ToTensor()

    file = os.path.splitext(filename)[0]

    # Load lr frame
    lr_frame = cv2.imread(f"{path}/{file}.png", cv2.IMREAD_UNCHANGED)
    lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
    lr_tensor = transform(lr_frame)
    # Load basecolor
    basecolor_frame = cv2.imread(f"{path}/{file}.basecolor.png", cv2.IMREAD_UNCHANGED)
    basecolor_frame = cv2.cvtColor(basecolor_frame, cv2.COLOR_BGR2RGB)
    basecolor_tensor = transform(basecolor_frame)
    # Load depth (log)
    depth_log_frame = cv2.imread(f"{path}/{file}.depth_log.png", cv2.IMREAD_GRAYSCALE)
    depth_log_tensor = transform(depth_log_frame)
    # Load metallic
    metallic_frame = cv2.imread(f"{path}/{file}.metallic.png", cv2.IMREAD_GRAYSCALE)
    metallic_tensor = transform(metallic_frame)
    # Normal Vector
    nov_frame = cv2.imread(f"{path}/{file}.normal_vector.png", cv2.IMREAD_GRAYSCALE)
    nov_tensor = transform(nov_frame)
    # Roughness
    roughness_frame = cv2.imread(f"{path}/{file}.roughness.png", cv2.IMREAD_GRAYSCALE)
    roughness_tensor = transform(roughness_frame)
    # velocity (log)
    velocity_log_frame = cv2.imread(f"{path}/{file}.velocity_log.png", cv2.IMREAD_UNCHANGED)
    velocity_log_frame = cv2.cvtColor(velocity_log_frame, cv2.COLOR_BGR2RGB)
    velocity_log_tensor = transform(velocity_log_frame)
    # world_normal
    world_normal_frame = cv2.imread(f"{path}/{file}.world_normal.png", cv2.IMREAD_UNCHANGED)
    world_normal_frame = cv2.cvtColor(world_normal_frame, cv2.COLOR_BGR2RGB)
    world_normal_tensor = transform(world_normal_frame)
    # world_position
    world_position_frame = cv2.imread(f"{path}/{file}.world_normal.png", cv2.IMREAD_UNCHANGED)
    world_position_frame = cv2.cvtColor(world_position_frame, cv2.COLOR_BGR2RGB)
    world_position_tensor = transform(world_position_frame)

    # Now all buffers are loaded, so we stack them into one
    buffer_tensor = torch.cat([lr_tensor, basecolor_tensor, depth_log_tensor, metallic_tensor, nov_tensor,
                               roughness_tensor, velocity_log_tensor, world_normal_tensor, world_position_tensor],
                              dim=0).numpy()

    filename = filename.replace(".png", "")
    np.savez_compressed(f"{safe_path}/{filename}.npz", buffer_tensor)


def create_folder_if_not_exists(path):
    os.makedirs(path, exist_ok=True)


def convert_lr_to_npz() -> None:
    path = "ue_data"
    safe_path = "ue_data_npz"
    # "01", "02", "03",
    train = ["04", "05", "06", "07", "08", "09", "10", "11", "12"]
    # "13"
    val = ["14", "15", "16"]
    test = ["17", "18", "19", "20"]

    print("Start train...")
    # train
    # for fol in train:
    #     p = f"{path}/train/HR/{fol}"
    #     s_p = f"{safe_path}/train/HR/{fol}"
    #     create_folder_if_not_exists(s_p)
    #     for i in range(300):
    #         filename = f"{i:0{4}d}.png"
    #         convert_hr_frame_to_npz_file(p, filename, s_p)
    # for fol in train:
    #     p = f"{path}/train/LR/{fol}"
    #     s_p = f"{safe_path}/train/LR/{fol}"
    #     create_folder_if_not_exists(s_p)
    #     for i in range(300):
    #         filename = f"{i:0{4}d}.png"
    #         convert_lr_frames_to_npz_file(p, filename, s_p)

    print("Start val...")
    # val
    # for fol in val:
    #     p = f"{path}/val/HR/{fol}"
    #     s_p = f"{safe_path}/val/HR/{fol}"
    #     create_folder_if_not_exists(s_p)
    #     for i in range(300):
    #         filename = f"{i:0{4}d}.png"
    #         convert_hr_frame_to_npz_file(p, filename, s_p)
    for fol in val:
        p = f"{path}/val/LR/{fol}"
        s_p = f"{safe_path}/val/LR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_lr_frames_to_npz_file(p, filename, s_p)

    print("Start test...")
    # test
    for fol in test:
        p = f"{path}/test/HR/{fol}"
        s_p = f"{safe_path}/test/HR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_hr_frame_to_npz_file(p, filename, s_p)
    for fol in test:
        p = f"{path}/test/LR/{fol}"
        s_p = f"{safe_path}/test/LR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_lr_frames_to_npz_file(p, filename, s_p)


def convert_mvs_npz(folder_path: str, safe_path: str) -> None:
    transform = transforms.ToTensor()
    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file
    for file_name in tqdm(files, f"generate {folder_path}"):
        mv = cv2.imread(f"{folder_path}/{file_name}", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        mv = cv2.cvtColor(mv, cv2.COLOR_BGR2RGB)
        mv_r = mv[:, :, 0]  # Red channel
        mv_g = mv[:, :, 1]  # Blue channel
        # mv_g = mv_g * -1
        mv = np.stack([mv_r, mv_g], axis=-1)
        # mv = mv - 0.5
        mv = transform(mv).numpy()
        file_name = file_name.replace(".exr", "")
        np.savez_compressed(f"{safe_path}/{file_name}.npz", mv)


def main() -> None:
    path = "MVs"
    safe_path = "MVs_npz"

    folders = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
               "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

    for folder in tqdm(folders, "Generating..."):
        p = f"{path}/{folder}"
        s_p = f"{safe_path}/{folder}"
        create_folder_if_not_exists(s_p)
        convert_mvs_npz(p, s_p)


if __name__ == "__main__":
    main()
