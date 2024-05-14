import cv2
import os
from tqdm import tqdm


def generate_videos(path: str, name: str, save_path: str) -> None:
    generate_hr_video(path, name, save_path)
    generate_lr_videos(path, name, save_path)


def generate_hr_video(path: str, name: str, save_path: str) -> None:
    path += "/HR"
    generate_video(path, f"{name}", save_path)


def generate_lr_videos(path: str, name, save_path: str) -> None:
    path += "/LR"
    sub_names = [#"basecolor", "metallic", "normal_vector", "roughness", "depth_10", "depth_log"
                 #"velocity", "velocity_log",
                    "world_normal", "world_position"]
    generate_video(path, f"{name}", save_path)

    # for sub_name in sub_names:
    #     generate_sub_video(path, f"{name}_{sub_name}", save_path, sub_name)


def generate_video(path: str, name: str, save_path: str) -> None:
    # Get all PNG files in the directory
    path = f"{path}/{name}"
    images = sorted([img for img in os.listdir(path) if img.endswith(".png") and img.count(".") == 1])
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    print(f"Height {height}, Width {width} and Layers {layers}")

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Adjust codec as needed
    generate_directory(save_path)
    video = cv2.VideoWriter(os.path.join(save_path, f"{name}.mp4"), fourcc, 60, (width, height))

    for img in tqdm(images, desc=f"Generating video.."):
        video.write(cv2.imread(os.path.join(path, img)))

    cv2.destroyAllWindows()
    video.release()


def generate_sub_video(path: str, name: str, save_path: str, sub_type: str) -> None:
    # Get all PNG files in the directory
    images = sorted([img for img in os.listdir(path) if img.endswith(f"{sub_type}.png")])
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    print(f"Height {height}, Width {width} and Layers {layers}")

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Adjust codec as needed
    video = cv2.VideoWriter(os.path.join(save_path, f"{name}.mp4"), fourcc, 60, (width, height))

    for img in tqdm(images, desc=f"Generating sub video {sub_type}.."):
        video.write(cv2.imread(os.path.join(path, img)))

    cv2.destroyAllWindows()
    video.release()


def generate_network_videos() -> None:
    path = "../results/test"
    names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    for name in names:
        print(f"Videos for {name}..")
        generate_video(path, name, f"../videos/{name}")


def generate_train_videos() -> None:
    path = "../dataset/ue_data/train"
    names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    for name in names:
        print(f"Generate video for {name}..")
        generate_hr_video(path, name, f"../videos/HR/{name}")

        # 30 fps video
        generate_lr_30_video(name, path, f"../videos/LR/{name}")


def generate_lr_30_video(name, path, save_path: str) -> None:
    path = f"{path}/LR/{name}"
    images = sorted([img for img in os.listdir(path) if img.endswith(".png") and img.count(".") == 1])
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    print(f"Height {height}, Width {width} and Layers {layers}")
    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Adjust codec as needed
    generate_directory(save_path)
    video = cv2.VideoWriter(os.path.join(save_path, f"{name}.mp4"), fourcc, 30, (width, height))
    counter = 0
    for img in tqdm(images, desc=f"Generating video.."):
        if counter % 2 == 0:
            video.write(cv2.imread(os.path.join(path, img)))
        counter += 1
    cv2.destroyAllWindows()
    video.release()


def generate_directory(path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)


def main() -> None:
    generate_train_videos()


if __name__ == "__main__":
    main()
