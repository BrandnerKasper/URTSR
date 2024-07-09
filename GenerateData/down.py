import os

import cv2


def downsample_image(input_path: str, output_path: str, output_size: tuple) -> None:
    # Read the image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Resize the image using bicubic interpolation
    downsampled_image = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)

    # Save the downsampled image
    cv2.imwrite(output_path, downsampled_image)
    print(f"Image saved as {output_path}")


def downsample_all(folder_path: str, output_folder_path: str, size: tuple) -> None:
    files = os.listdir(folder_path)
    for file in files:
        downsample_image(f"{folder_path}/{file}", f"{output_folder_path}/{file}", size)


def generate_lr_images(folder_path: str, output_folder_path: str, size: tuple) -> None:
    files = os.listdir(folder_path)
    for file in files:
        generate_blurry_aliased_image(f"{folder_path}/{file}", f"{output_folder_path}/{file}", size)


def generate_blurry_aliased_image(image_path, output_path, target_resolution=(1920, 1080), blur_radius=0.5):
    # Read the high-resolution image
    high_res_image = cv2.imread(image_path)

    # Calculate the intermediate downscaled size to introduce aliasing
    intermediate_scale_factor = 2  # Intermediate scale factor (can be adjusted)
    intermediate_size = (
        high_res_image.shape[1] // intermediate_scale_factor, high_res_image.shape[0] // intermediate_scale_factor)

    # Initial downsampling with nearest-neighbor interpolation to introduce aliasing
    aliased_image = cv2.resize(high_res_image, intermediate_size, interpolation=cv2.INTER_NEAREST)

    # Apply Gaussian blur to the aliased image
    # blurred_image = cv2.GaussianBlur(aliased_image, (0, 0), blur_radius)

    # Final downsampling to the target resolution
    low_res_image = cv2.resize(aliased_image, target_resolution, interpolation=cv2.INTER_LINEAR)

    # Save the downsampled image
    cv2.imwrite(output_path, low_res_image)
    print(f"Image saved as {output_path}")


def main() -> None:
    in_path = "HR_new/01"
    out_path = "LR_down2/01"
    size = (1920, 1080)
    generate_lr_images(in_path, out_path, size)


if __name__ == "__main__":
    main()
    # downsample_image("compare/0050.8K.png", "compare/0050.8k_down.png", (3840, 2160))
