import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV


def motion_diff() -> None:
    # Load the two images
    vel_0 = cv2.imread("../dataset/ue_data/train/LR/01/0008.velocity_log.png", cv2.IMREAD_GRAYSCALE)
    vel_1 = cv2.imread("../dataset/ue_data/train/LR/01/0009.velocity_log.png", cv2.IMREAD_GRAYSCALE)

    img = cv2.imread("../dataset/ue_data/train/LR/01/0009.png", cv2.IMREAD_GRAYSCALE)

    # Compute the absolute difference between the two images
    diff = cv2.absdiff(vel_1, vel_0)
    mask = diff / 255
    mask = 1 - mask

    cv2.imshow("Vel0", vel_0)
    cv2.imshow("Vel1", vel_1)
    cv2.imshow("mask", mask)

    cv2.imshow("img", img)

    img_warped = img - diff
    cv2.imshow("img2", img_warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def motion_edge_diff() -> None:
    # Load the two images
    vel_0 = cv2.imread("../dataset/ue_data/train/LR/01/0008.velocity_log.png", cv2.IMREAD_GRAYSCALE)
    vel_1 = cv2.imread("../dataset/ue_data/train/LR/01/0022.velocity_log.png", cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection to both images
    edges1 = cv2.Canny(vel_0, 100, 200)
    edges2 = cv2.Canny(vel_1, 100, 200)

    # Compute the absolute difference between the edge images0
    diff_edges = cv2.absdiff(edges1, edges2)

    # Load the LR image and try to warp it
    img = cv2.imread("../dataset/ue_data/train/LR/01/0009.png", cv2.IMREAD_GRAYSCALE)

    mask = diff_edges / 255
    mask = 1 - mask

    img_warped = img * mask
    img_as_int = img_warped.astype(int)
    #img_copy = np.copy(img)
    #for i in range(0, diff_edges.shape[0]):
    #    for j in range(0, diff_edges.shape[1]):
    #        if diff_edges[i, j] == 255:
    #            img_copy[i, j] = 0
    #cv2.imshow("Marion tries shit", img_copy)

    # Display the difference in edges
    cv2.imshow("Difference in Edges", diff_edges)
    cv2.imshow("Original Image", img)
    # cv2.imshow("Warped Images", img_as_int)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = mask * 255
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("original.png", img)
    cv2.imwrite("warped.png", img_warped)


def warping() -> None:
    # Load the images
    image = cv2.imread('../dataset/ue_data/test/LR/18/0251.png').astype(np.float32) / 255.0
    current_motion = cv2.imread('../dataset/ue_data/test/LR/18/0251.velocity_log.png', cv2.IMREAD_UNCHANGED)
    next_motion = cv2.imread('../dataset/ue_data/test/LR/18/0252.velocity_log.png', cv2.IMREAD_UNCHANGED)

    # Normalize motion vectors (assuming they are in the range [0, 255] in the image)
    current_motion = (current_motion / 255.0 - 0.5) * 2
    next_motion = (next_motion / 255.0 - 0.5) * 2

    # Display the normalized motion vectors for debugging
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Normalized Current Motion Vector')
    plt.imshow(current_motion, cmap='viridis')

    plt.subplot(1, 2, 2)
    plt.title('Normalized Next Motion Vector')
    plt.imshow(next_motion, cmap='viridis')
    plt.show()

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Create a meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute new pixel locations using current motion vectors
    new_x = (x + next_motion[:, :, 0] * width).astype(np.float32)
    new_y = (y + next_motion[:, :, 1] * height).astype(np.float32)

    # Clip the new coordinates to be within the image boundaries
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)

    # Warp the image using remap
    warped_image = cv2.remap(image, new_x, new_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Display the images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title('Current Motion Vector')
    plt.imshow(current_motion)

    plt.subplot(1, 3, 3)
    plt.title('Warped Image')
    plt.imshow(warped_image)

    plt.show()

    # Save the warped image for further inspection
    cv2.imwrite('warped_image.png', (warped_image* 255).astype(np.uint8))


def warping_2() -> None:
    # Load the images
    image = cv2.imread('../dataset/ue_data/test/LR/18/0251.png').astype(np.float32) / 255.0
    current_motion = cv2.imread('../dataset/ue_data/test/LR/18/0251.velocity_log.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    next_motion = cv2.imread('../dataset/ue_data/test/LR/18/0252.velocity_log.png', cv2.IMREAD_UNCHANGED).astype(np.float32)

    # Normalize motion vectors (assuming they are in the range [0, 255] in the image)
    # Normalize to the range [-1, 1] assuming the maximum motion vector displacement is represented by 255
    current_motion = (current_motion / 255.0 - 0.5) * 2
    next_motion = (next_motion / 255.0 - 0.5) * 2

    # Resize motion vectors to match the shape [H, W, 2]
    current_motion = current_motion[..., :2]
    next_motion = next_motion[..., :2]

    # Convert the image and motion vectors to PyTorch tensors
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)  # Shape: [1, 3, H, W]
    motion_tensor = torch.from_numpy(current_motion.transpose(2, 0, 1)).unsqueeze(0)  # Shape: [1, 2, H, W]

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Create a normalized grid for pixel coordinates
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)  # Shape: [1, H, W, 2]

    # Move the motion_tensor to the correct device
    motion_tensor = motion_tensor.to(grid.device)

    # Permute the motion_tensor to match the shape of grid
    motion_tensor_permuted = motion_tensor.permute(0, 2, 3, 1)  # Shape: [1, H, W, 2]

    # Add the motion vectors to the grid
    warped_grid = grid + motion_tensor_permuted

    # Warp the image using grid_sample
    warped_image = F.grid_sample(image_tensor, warped_grid, mode='bilinear', padding_mode='border', align_corners=False)

    # Convert the warped image tensor back to a numpy array
    warped_image_np = warped_image.squeeze().permute(1, 2, 0).numpy()

    # Display the images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title('Current Motion Vector')
    plt.imshow(current_motion[..., 0], cmap='coolwarm')  # Display one channel of the motion vector

    plt.subplot(1, 3, 3)
    plt.title('Warped Image')
    plt.imshow(warped_image_np)

    plt.show()

    # Save the warped image for further inspection
    warped_image_255 = (warped_image_np * 255).astype(np.uint8)
    cv2.imwrite('warped_image_pytorch.png', warped_image_255)


def warp(x_np, flo_np):
    x = torch.tensor(x_np).unsqueeze(0).cuda()
    flo = torch.tensor(flo_np).unsqueeze(0).cuda()

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    flo = flo.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size(), device=x.device)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def load_npz(path: str) -> torch.Tensor:
    img = np.load(f"{path}.npz")
    return torch.from_numpy(next(iter(img.values())))


def warping_3(image_path, current_motion_path, next_motion_path):
    # Load the images
    image = load_npz(image_path).numpy()
    current_motion = load_npz(current_motion_path).numpy()
    next_motion = load_npz(next_motion_path)

    # Display the normalized motion vectors for debugging
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Normalized Current Motion Vector')
    plt.imshow(current_motion, cmap='viridis')

    plt.subplot(1, 2, 2)
    plt.title('Normalized Next Motion Vector')
    plt.imshow(next_motion, cmap='viridis')
    plt.show()

    # Normalize motion vectors (assuming they are in the range [0, 255] in the image)
    current_motion2 = current_motion - 0.7373
    current_motion3 = current_motion2 * 2
    next_motion = (next_motion / 255.0 - 0.5) * 2

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Create a meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute new pixel locations using current motion vectors
    new_x = (x + current_motion3[:, :, 0] * width).astype(np.float32)
    new_y = (y + current_motion3[:, :, 1] * height).astype(np.float32)

    # Clip the new coordinates to be within the image boundaries
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)

    # Warp the image using remap
    warped_image = cv2.remap(image, new_x, new_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Display the images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    image = FV.to_pil_image(image)
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title('Current Motion Vector')
    plt.imshow(current_motion)

    plt.subplot(1, 3, 3)
    plt.title('Warped Image')
    plt.imshow(warped_image)

    plt.show()


def normalize_channel(channel):
    channel_min = channel.min()
    channel_max = channel.max()
    normalized_channel = (channel - channel_min) / (channel_max - channel_min)
    return normalized_channel


def main() -> None:
    image_path = 'test/0037.png'
    current_mv_path = "test/0037.velocity.png"
    # current_mv_log_path = "test/0037.velocity_log.png"
    next_mv_path = "test/0038.velocity.png"
    next_mv_log_path = "test/0038.velocity_log.png"

    image = cv2.imread(image_path)
    current_mv = cv2.imread(current_mv_path)
    # current_mv_log = cv2.imread(current_mv_log_path)

    # Convert images from BGR to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    current_mv_rgb = cv2.cvtColor(current_mv, cv2.COLOR_BGR2RGB)
    # current_mv_log_rgb = cv2.cvtColor(current_mv_log, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to be between 0 and 1
    image_rgb = image_rgb / 255.0
    current_mv_rgb = current_mv_rgb / 255.0
    # current_mv_log_rgb = current_mv_log_rgb / 255.0
    # current_mv_log_rgb = (current_mv_log_rgb - 0.5) * 2

    # Separate RGB channels for image_rgb
    image_r = image_rgb[:, :, 0]
    image_g = image_rgb[:, :, 1]
    image_b = image_rgb[:, :, 2]

    current_mv_r = current_mv_rgb[:, :, 0]
    current_mv_g = current_mv_rgb[:, :, 1]
    current_mv_b = current_mv_rgb[:, :, 2]

    # current_mv_log_r = current_mv_log_rgb[:, :, 0]
    # current_mv_log_g = current_mv_log_rgb[:, :, 0]
    # current_mv_log_b = current_mv_log_rgb[:, :, 0]

    # normalize the values
    current_mv_r = normalize_channel(current_mv_r)
    current_mv_g = normalize_channel(current_mv_g)
    current_mv_b = np.full((1080, 1920), 0.5)

    normalized_image = np.stack([current_mv_r, current_mv_g, current_mv_b], axis=-1)
    cv2.imwrite("test/0037.norm_mv.png", (normalized_image*255).astype(np.uint8))



    # Plot the images in one plot
    plt.figure(figsize=(15, 15))

    # Original image RGB channels
    plt.subplot(5, 3, 1)
    plt.imshow(image_r, cmap='Reds')
    plt.title('Image - Red Channel')
    plt.axis('off')

    plt.subplot(5, 3, 2)
    plt.imshow(image_g, cmap='Greens')
    plt.title('Image - Green Channel')
    plt.axis('off')

    plt.subplot(5, 3, 3)
    plt.imshow(image_b, cmap='Blues')
    plt.title('Image - Blue Channel')
    plt.axis('off')

    # Current MV RGB channels
    plt.subplot(5, 3, 4)
    plt.imshow(current_mv_r, cmap='Reds')
    plt.title('Current MV - Red Channel')
    plt.axis('off')

    plt.subplot(5, 3, 5)
    plt.imshow(current_mv_g, cmap='Greens')
    plt.title('Current MV - Green Channel')
    plt.axis('off')

    plt.subplot(5, 3, 6)
    plt.imshow(current_mv_b, cmap='Blues')
    plt.title('Current MV - Blue Channel')
    plt.axis('off')

    # mv normalized
    plt.subplot(5, 3, 7)
    plt.imshow(normalized_image)
    plt.title("Current MV normalized")

    # Plotting the original image for reference
    plt.subplot(5, 3, 10)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.figure(figsize=(15, 10))

    # Display the normalized_image
    plt.imshow(normalized_image)
    plt.title('Normalized Image')
    plt.axis('off')

    plt.show()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
