import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def viz_exr(file: str) -> None:
    mv_path = file

    mv = cv2.imread(mv_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mv = cv2.cvtColor(mv, cv2.COLOR_BGR2RGB)

    # Display r and g channels
    mv_r = mv[:, :, 0]  # Red channel
    mv_g = mv[:, :, 1]  # Blue channel

    # Display the R channel
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mv_r, cmap='Reds')
    plt.title('Red Channel')
    plt.colorbar()

    # Display the G channel
    plt.subplot(1, 2, 2)
    plt.imshow(mv_g, cmap='Greens')
    plt.title('Green Channel')
    plt.colorbar()

    plt.show()


def convert_exr_to_png(exr_path: str, png_path: str) -> None:
    # Read the EXR image
    exr_image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)

    # Ensure the image has RGB channels
    if exr_image.shape[2] == 4:  # if the image has an alpha channel
        exr_image = exr_image[:, :, :3]

    # Apply tone mapping to convert HDR to LDR
    tonemap = cv2.createTonemapDrago(1.0, 0.7)
    ldr_image = tonemap.process(exr_image)

    # Normalize to 0-255 range and convert to uint8
    ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

    # Convert from BGR to RGB (since OpenCV uses BGR by default)
    ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(ldr_image)
    plt.title("Tone Mapped EXR to PNG")
    plt.show()

    # Convert the numpy array to a PIL image and save as PNG
    img = Image.fromarray(ldr_image)
    img.save(png_path)
    print(f"Image saved as {png_path}")


def viz_npz() -> None:
    path = "MVs_npz/01/0100.velocity.npz"
    mv = np.load(path)
    mv = next(iter(mv.values()))
    # Display r and g channels
    mv_r = mv[0, :, :]  # Red channel
    mv_g = mv[1, :, :]  # Blue channel

    # Display the R channel
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mv_r, cmap='Reds')
    plt.title('Red Channel')
    plt.colorbar()

    # Display the G channel
    plt.subplot(1, 2, 2)
    plt.imshow(mv_g, cmap='Greens')
    plt.title('Green Channel')
    plt.colorbar()

    plt.show()


def main() -> None:
    # viz_npz()
    # viz_exr("LR_Test/0050.FinalImagevelocity.exr")
    viz_exr("LR_new/03/0050.velocity.exr")
    # lr_exr_path = "HDR_Test/0275.FinalImage.exr"
    # hr_exr_path = "HDR_Test_HD/0275.FinalImage.exr"
    # convert_exr_to_png(lr_exr_path, "test.png")


if __name__ == "__main__":
    main()
