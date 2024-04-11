import cv2
import os
from tqdm import tqdm


def generate_videos(name: str) -> None:
    path = f"../dataset/matrix/train"
    save_path = f"../videos/{name}"

    # generate_hr_video(path, name, save_path)

    generate_lr_videos(path, name, save_path)


def generate_hr_video(path: str, name: str, save_path: str) -> None:
    path += f"/HR/{name}"
    name += "_hr"
    generate_video(path, name, save_path)


def generate_lr_videos(path: str, name, save_path: str) -> None:
    path += f"/LR/{name}"
    sub_names = [#"basecolor", "metallic", "normal_vector", "roughness", "depth_10", "depth_log"
                 #"velocity", "velocity_log",
                    "world_normal", "world_position"]
    # generate_video(path, f"{name}_lr", save_path)

    for sub_name in sub_names:
        generate_sub_video(path, f"{name}_{sub_name}", save_path, sub_name)


def generate_video(path: str, name: str, save_path: str) -> None:
    # Get all PNG files in the directory
    images = sorted([img for img in os.listdir(path) if img.endswith(".png") and img.count(".") == 1])
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    print(f"Height {height}, Width {width} and Layers {layers}")

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Adjust codec as needed
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


def main() -> None:
    names = ["empire_state", "flat_iron", "ny_by_night", "ny_in_person"]
    for name in names:
        print(f"Videos for {name}..")
        generate_videos(name)


if __name__ == "__main__":
    main()
