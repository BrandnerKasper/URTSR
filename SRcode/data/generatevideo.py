import cv2
import os
from tqdm import tqdm


def main() -> None:
    image_folder = "dataset/matrix/LR"
    video_name = "matrix_LR_worldposition.mp4"

    # Get all PNG files in the directory
    images = sorted([img for img in os.listdir(image_folder) if img.endswith("worldposition.png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    print(f"Height {height}, Width {width} and Layers {layers}")

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Adjust codec as needed
    video = cv2.VideoWriter(video_name, fourcc, 60, (width, height))

    for img in tqdm(images, desc=f"Generating video.."):
        video.write(cv2.imread(os.path.join(image_folder, img)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    main()
