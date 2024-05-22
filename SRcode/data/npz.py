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
    np.savez_compressed(f"{safe_path}/{file}.npz", lr_tensor)
    # Load basecolor
    basecolor_frame = cv2.imread(f"{path}/{file}.basecolor.png", cv2.IMREAD_UNCHANGED)
    basecolor_frame = cv2.cvtColor(basecolor_frame, cv2.COLOR_BGR2RGB)
    basecolor_tensor = transform(basecolor_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.basecolor.npz", basecolor_tensor)
    # Load depth (log)
    depth_log_frame = cv2.imread(f"{path}/{file}.depth_log.png", cv2.IMREAD_GRAYSCALE)
    depth_log_tensor = transform(depth_log_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.depth_log.npz", depth_log_tensor)
    # Load metallic
    metallic_frame = cv2.imread(f"{path}/{file}.metallic.png", cv2.IMREAD_GRAYSCALE)
    metallic_tensor = transform(metallic_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.metallic.npz", metallic_tensor)
    # Normal Vector
    nov_frame = cv2.imread(f"{path}/{file}.normal_vector.png", cv2.IMREAD_GRAYSCALE)
    nov_tensor = transform(nov_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.normal_vector.npz", nov_tensor)
    # Roughness
    roughness_frame = cv2.imread(f"{path}/{file}.roughness.png", cv2.IMREAD_GRAYSCALE)
    roughness_tensor = transform(roughness_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.roughness.npz", roughness_tensor)
    # velocity (log)
    velocity_log_frame = cv2.imread(f"{path}/{file}.velocity_log.png", cv2.IMREAD_UNCHANGED)
    velocity_log_frame = cv2.cvtColor(velocity_log_frame, cv2.COLOR_BGR2RGB)
    velocity_log_tensor = transform(velocity_log_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.velocity_log.npz", velocity_log_tensor)
    # world_normal
    world_normal_frame = cv2.imread(f"{path}/{file}.world_normal.png", cv2.IMREAD_UNCHANGED)
    world_normal_frame = cv2.cvtColor(world_normal_frame, cv2.COLOR_BGR2RGB)
    world_normal_tensor = transform(world_normal_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.world_normal.npz", world_normal_tensor)
    # world_position
    world_position_frame = cv2.imread(f"{path}/{file}.world_position.png", cv2.IMREAD_UNCHANGED)
    world_position_frame = cv2.cvtColor(world_position_frame, cv2.COLOR_BGR2RGB)
    world_position_tensor = transform(world_position_frame).numpy()
    np.savez_compressed(f"{safe_path}/{file}.world_position.npz", world_position_tensor)


def create_folder_if_not_exists(path):
    os.makedirs(path, exist_ok=True)


def main() -> None:
    path = "../dataset/ue_data"
    safe_path = "../dataset/ue_data_npz"
    train = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    val = ["13", "14", "15", "16"]
    test = ["17", "18", "19", "20"]

    # train
    for fol in tqdm(train, desc='Convert train HR folders..'):
        p = f"{path}/train/HR/{fol}"
        s_p = f"{safe_path}/train/HR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_hr_frame_to_npz_file(p, filename, s_p)
    for fol in tqdm(train, desc='Convert train LR folders..'):
        p = f"{path}/train/LR/{fol}"
        s_p = f"{safe_path}/train/LR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_lr_frames_to_npz_file(p, filename, s_p)

    # val
    for fol in tqdm(val, desc='Convert val HR folders..'):
        p = f"{path}/val/HR/{fol}"
        s_p = f"{safe_path}/val/HR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_hr_frame_to_npz_file(p, filename, s_p)
    for fol in tqdm(val, desc='Convert val LR folders..'):
        p = f"{path}/val/LR/{fol}"
        s_p = f"{safe_path}/val/LR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_lr_frames_to_npz_file(p, filename, s_p)

    # test
    for fol in tqdm(test, desc='Convert test HR folder'):
        p = f"{path}/test/HR/{fol}"
        s_p = f"{safe_path}/test/HR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_hr_frame_to_npz_file(p, filename, s_p)
    for fol in tqdm(test, desc='Convert test LR folder'):
        p = f"{path}/test/LR/{fol}"
        s_p = f"{safe_path}/test/LR/{fol}"
        create_folder_if_not_exists(s_p)
        for i in range(300):
            filename = f"{i:0{4}d}.png"
            convert_lr_frames_to_npz_file(p, filename, s_p)


if __name__ == "__main__":
    main()
