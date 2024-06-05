import cv2
import os
from tqdm import tqdm


def generate_directory(path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)


def generate_video(path: str, name: str, save_path: str, lr: bool = False) -> None:
    # Get all PNG files in the directory
    path = f"{path}/{name}"
    images = sorted([img for img in os.listdir(path) if img.endswith(".png") and img.count(".") == 1])
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    print(f"Height {height}, Width {width} and Layers {layers}")

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Adjust codec as needed
    if lr:
        video = cv2.VideoWriter(os.path.join(save_path, f"{name}.mp4"), fourcc, 30, (width, height))
    else:
        video = cv2.VideoWriter(os.path.join(save_path, f"{name}.mp4"), fourcc, 60, (width, height))

    if lr:
        counter = 0
        for img in tqdm(images, desc=f"Generating video.."):
            if counter % 2 == 1:
                counter += 1
                continue
            counter += 1
            video.write(cv2.imread(os.path.join(path, img)))
    else:
        for img in tqdm(images, desc=f"Generating video.."):
            video.write(cv2.imread(os.path.join(path, img)))

    cv2.destroyAllWindows()
    video.release()


def generate_network_videos() -> None:
    path = "../results/extraSS_All"
    names = ["17", "18", "19", "20"]
    generate_directory("../videos/extraSS_All")
    for name in names:
        print(f"Videos for {name}..")
        generate_video(path, name, f"../videos/extraSS_All")


def generate_videos(file_path: str, save_name: str) -> None:
    generate_directory(f"../videos/{save_name}")
    for directory in os.listdir(file_path):
        print(f"Generate video for {directory}..")
        generate_video(file_path, directory, save_name)


def main() -> None:
    generate_videos("../results/extraSS_All", "test")


if __name__ == "__main__":
    main()
