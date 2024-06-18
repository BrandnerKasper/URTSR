import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from dataloader import DiskMode, load_image_from_disk


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


def warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """
    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    x = x.float()
    grid = grid.float()
    flow = flow.float()

    # Ensure the grid is within the range [-1, 1]
    grid[:, :, 0] = torch.clamp(grid[:, :, 0], -1, 1)
    grid[:, :, 1] = torch.clamp(grid[:, :, 1], -1, 1)

    # add flow to grid and reshape to nhw2
    grid = (grid - flow).permute(0, 2, 3, 1) # we use - for forward warping

    output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    output = torch.clamp(output, min=0, max=1)
    return output


def warp_img(image: np.ndarray, mv: np.ndarray) -> np.ndarray:
    # move pixel values to be between -1 and 1
    mv = (mv - 0.5) * 1
    print(mv.shape)
    mv_r = mv[0, :, :]
    mv_g = mv[1, :, :]
    # mv_r = mv_r * -1
    mv_g = mv_g * -1

    mv = np.stack([mv_r, mv_g], axis=-1)
    # mv = mv * -1

    image = torch.tensor(image).unsqueeze(0).cuda()
    mv = torch.tensor(mv).unsqueeze(0).cuda()

    # image = image.permute(0, 3, 1, 2)
    mv = mv.permute(0, 3, 1, 2)

    return np.transpose(warp(image, mv).squeeze(0).cpu().numpy(), (1, 2, 0))


def custom_warp(image: torch.Tensor, mv: torch.Tensor):
    image = image.float()
    mv = mv.float()
    b, c, h, w = image.size()

    # Create a grid of coordinates normalized to [-1, 1]
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
    grid = torch.stack((grid_x, grid_y), 2).cuda()  # Shape: (H, W, 2)

    # Squeeze out the first dimension
    mv_squeezed = mv.squeeze(0)  # Shape: [2, 1080, 1920]
    # Transpose the axes to get the desired shape [1080, 1920, 2]
    mv_transposed = mv_squeezed.permute(1, 2, 0)
    # Apply motion vectors to the grid
    grid += mv_transposed

    # Ensure the grid is within the range [-1, 1]
    grid[:, :, 0] = torch.clamp(grid[:, :, 0], -1, 1)
    grid[:, :, 1] = torch.clamp(grid[:, :, 1], -1, 1)


    # # Transpose the axes to get the desired shape [1080, 1920, 2]
    # mv_transposed = mv_squeezed.permute(1, 2, 0)
    # vgrid = grid + mv_transposed

    # # Ensure the grid is within the range [-1, 1]
    # vgrid[:, :, 0] = torch.clamp(vgrid[:, :, 0], -1, 1)
    # vgrid[:, :, 1] = torch.clamp(vgrid[:, :, 1], -1, 1)

    # display_image(vgrid.squeeze(0).cpu().numpy(), "VGRID")

    # vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(w - 1, 1) - 1.0
    #
    # vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(h - 1, 1) - 1.0

    vgrid = grid.unsqueeze(0)

    return F.grid_sample(image, vgrid, 'nearest', align_corners=True)


def mv_zoom_windmill() -> None:
    image_path = '../dataset/ue_data_npz/test/LR/17/0082'
    next_mv_path = '../dataset/ue_data_npz/test/LR/17/0083.velocity'

    image = load_image_from_disk(DiskMode.NPZ, image_path)

    mv = load_image_from_disk(DiskMode.NPZ, next_mv_path)

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

    warped = warp_img(image, mv)
    display_image(warped, "Warped")
    save_image(warped, "warped.png")


def move_all_directions() -> None:
    image_path = 'MV/0060.FinalImage.png'
    current_mv_path = "MV/0060.FinalImageM_Velocity.exr"
    next_mv_path = "MV/0061.FinalImageM_Velocity.exr"
    prev_mv_path = "MV/0059.FinalImageM_Velocity.exr"

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0

    mv = cv2.imread(current_mv_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
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

    warped = warp_img(image, mv)
    display_image(warped, "Warped")
    save_image(warped, "warped.png")


def display_image(image: np.ndarray, name: str) -> None:
    plt.figure(figsize=(15, 15))
    # Original image RGB channels
    plt.imshow(image)
    plt.title(f"Image - {name}")
    plt.axis('off')

    plt.show()


def save_image(img: np.ndarray, name: str) -> None:
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{name}", img)


def mv_mask(ocmv, mv, gate = 0.1):
    delta = ocmv - mv
    mask = torch.where(torch.abs(delta) < gate, False, True)
    x, y = mask[0, :, :], mask[1, :, :]
    mask = torch.where(((x) | (y)), 1, 0)
    return mask


def generate_mask() -> None:
    # Load the motion vector images (assuming they are in a compatible format)
    current_mv_path = "Style_EXR/0083.FinalImagevelocity.exr"
    previous_mv_path = "Style_EXR/0082.FinalImagevelocity.exr"

    current_mv = cv2.imread(current_mv_path, cv2.IMREAD_UNCHANGED)
    previous_mv = cv2.imread(previous_mv_path, cv2.IMREAD_UNCHANGED)

    # Ensure the motion vector images are of the same type and shape
    if current_mv.dtype != previous_mv.dtype or current_mv.shape != previous_mv.shape:
        raise ValueError("Motion vector images must have the same shape and data type.")

    # Calculate the absolute difference between the current and previous motion vectors
    motion_diff = np.abs(current_mv - previous_mv)

    # Sum the differences across the channels if the motion vector has multiple channels
    if motion_diff.ndim == 3:
        motion_diff = np.sum(motion_diff, axis=-1)

    # Normalize the difference for visualization (optional)
    motion_diff_normalized = (motion_diff - np.min(motion_diff)) / (np.max(motion_diff) - np.min(motion_diff))

    # Set a threshold to create the motion mask
    threshold = 0.1  # You may need to adjust this value
    motion_mask = (motion_diff_normalized > threshold).astype(np.uint8)

    # Display the motion mask
    plt.imshow(motion_mask, cmap='gray')
    plt.title('Motion Mask')
    plt.colorbar()
    plt.show()


def generate_error_mask(gt_image, sr_image):
    """
    Generates an error mask between the ground truth and super-resolved images.

    Parameters:
    gt_image (numpy.ndarray): Ground truth image.
    sr_image (numpy.ndarray): Super-resolved image.

    Returns:
    numpy.ndarray: Error mask.
    """
    # Ensure the images are of the same size
    if gt_image.shape != sr_image.shape:
        raise ValueError("Ground truth and super-resolved images must have the same dimensions.")

    # Convert images to float32 for precise difference calculation
    gt_image = gt_image.astype(np.float32)
    sr_image = sr_image.astype(np.float32)

    # Calculate the absolute difference
    error = np.abs(gt_image - sr_image)

    # Normalize the error to the range [0, 1]
    error_norm = cv2.normalize(error, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return 1 - error_norm


def visualize_error_mask(gt_image, sr_image, error_mask):
    """
    Visualizes the ground truth, super-resolved, and error mask images.

    Parameters:
    gt_image (numpy.ndarray): Ground truth image.
    sr_image (numpy.ndarray): Super-resolved image.
    error_mask (numpy.ndarray): Error mask.
    """
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 3, 1)
    plt.title('Ground Truth')
    gt_image = np.transpose(gt_image, (1, 2, 0))
    plt.imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Super-Resolved')
    sr_image = np.transpose(sr_image, (1, 2, 0))
    plt.imshow(cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Error Mask')
    error_mask = np.transpose(error_mask, (1, 2, 0))
    plt.imshow(error_mask, cmap='hot')
    plt.colorbar()
    plt.axis('off')

    plt.show()


def error_mask_test() -> None:
    # gt_path_sr = "../dataset/ue_data_npz/test/HR/17/0086"
    # gt_path_ess = "../dataset/ue_data_npz/test/HR/17/0087"
    #
    # model_path_sr = "../results/flavr/17/0086"
    # model_path_ess = "../results/flavr/17/0087"
    #
    # ss_gt = load_image_from_disk(DiskMode.NPZ, gt_path_ess)
    # ss_model = load_image_from_disk(DiskMode.CV2, model_path_ess)
    #
    # ss_gt = ss_gt.numpy()
    # ss_model = ss_model.numpy()

    # ss_error_mask = generate_error_mask(ss_gt, ss_model)
    # error_mask = np.transpose(ss_error_mask, (1, 2, 0))
    # save_image(error_mask, "error.png")
    # visualize_error_mask(ss_gt, ss_model, ss_error_mask)

    warped_path = "warped"
    warped = load_image_from_disk(DiskMode.CV2, warped_path)
    gt_path = '../dataset/ue_data_npz/test/LR/17/0083'
    gt = load_image_from_disk(DiskMode.NPZ, gt_path)

    warped = warped.numpy()
    gt = gt.numpy()
    error = generate_error_mask(gt, warped)
    error_t = np.transpose(error, (1, 2, 0))
    save_image(error_t, "warped_error.png")
    visualize_error_mask(gt, warped, error)


def main() -> None:
    error_mask_test()
    # move_all_directions()
    # mv_zoom_windmill()
    # generate_mask()

    # image_path = 'Test2/FinalImage.0010.png'
    # current_mv_path = "Test2/FinalImageM_Velocity.0010.exr"
    # next_mv_path = "Test2/FinalImageM_Velocity.0011.exr"
    #
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image / 255.0
    #
    # mv = cv2.imread(current_mv_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # mv = cv2.cvtColor(mv, cv2.COLOR_BGR2RGB)
    #
    # # Display r and g channels
    # mv_r = mv[:, :, 0]  # Red channel
    # mv_g = mv[:, :, 1]  # Blue channel
    #
    # # Display the R channel
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(mv_r, cmap='Reds')
    # plt.title('Red Channel')
    # plt.colorbar()
    #
    # # Display the G channel
    # plt.subplot(1, 2, 2)
    # plt.imshow(mv_g, cmap='Greens')
    # plt.title('Green Channel')
    # plt.colorbar()
    #
    # plt.show()
    #
    # display_image(mv, "EXR")
    #
    # warped = warp_img(image, mv)
    #
    # display_image(warped, "warped vel")
    # save_image(warped, "test.png")

    # # Plot the images in one plot
    # plt.figure(figsize=(15, 15))
    #
    # # Original image RGB channels
    # plt.subplot(5, 3, 1)
    # plt.imshow(image_r, cmap='Reds')
    # plt.title('Image - Red Channel')
    # plt.axis('off')
    #
    # plt.subplot(5, 3, 2)
    # plt.imshow(image_g, cmap='Greens')
    # plt.title('Image - Green Channel')
    # plt.axis('off')
    #
    # plt.subplot(5, 3, 3)
    # plt.imshow(image_b, cmap='Blues')
    # plt.title('Image - Blue Channel')
    # plt.axis('off')
    #
    # # Current MV RGB channels
    # plt.subplot(5, 3, 4)
    # plt.imshow(current_mv_r, cmap='Reds')
    # plt.title('Current MV - Red Channel')
    # plt.axis('off')
    #
    # plt.subplot(5, 3, 5)
    # plt.imshow(current_mv_g, cmap='Greens')
    # plt.title('Current MV - Green Channel')
    # plt.axis('off')
    #
    # plt.subplot(5, 3, 6)
    # plt.imshow(current_mv_b, cmap='Blues')
    # plt.title('Current MV - Blue Channel')
    # plt.axis('off')
    #
    # # Current MV LOG RGB channels
    # plt.subplot(5, 3, 7)
    # plt.imshow(current_mv_log_r, cmap='Reds')
    # plt.title('Current MV LOG - Red Channel')
    # plt.axis('off')
    #
    # plt.subplot(5, 3, 8)
    # plt.imshow(current_mv_log_g, cmap='Greens')
    # plt.title('Current MV LOG - Green Channel')
    # plt.axis('off')
    #
    # plt.subplot(5, 3, 9)
    # plt.imshow(current_mv_log_b, cmap='Blues')
    # plt.title('Current MV LOG - Blue Channel')
    # plt.axis('off')
    #
    # # mv normalized
    # plt.subplot(5, 3, 10)
    # plt.imshow(normalized_image)
    # plt.title("Current MV normalized")
    #
    # # mv log normalized
    # plt.subplot(5, 3, 11)
    # plt.imshow(norm_mv_log)
    # plt.title("Current MV LOG normalized")
    #
    # # Plotting the original image for reference
    # plt.subplot(5, 3, 12)
    # plt.imshow(image_rgb)
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # plt.figure(figsize=(15, 10))
    #
    # # Display the normalized_image
    # warped_c = np.transpose(warped_c, (1, 2, 0))
    # warped_c = (warped_c * 255).astype(np.uint8)
    # warped_c = cv2.cvtColor(warped_c, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("test/0037.warped.png", warped_c)
    # plt.imshow(warped_c)
    # plt.title('Warped Image')
    # plt.axis('off')
    #
    # plt.show()


if __name__ == "__main__":
    main()
