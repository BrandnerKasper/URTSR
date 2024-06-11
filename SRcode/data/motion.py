import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import OpenEXR

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

    print(flow)
    # normalize flow to [-1, 1]
    # flow = torch.cat([
    #     flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
    #     flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    x = x.float()
    grid = grid.float()
    flow = flow.float()

    # add flow to grid and reshape to nhw2
    grid = (grid - flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default for PyTorch version < 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def backward_warp_motion(pre: torch.Tensor, motion: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    # see: https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
    # input image is: [batch, channel, height, width]
    # st = time.time()
    index_batch, number_channels, height, width = pre.size()
    grid_x = torch.arange(width).view(1, -1).repeat(height, 1)
    grid_y = torch.arange(height).view(-1, 1).repeat(1, width)
    grid_x = grid_x.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    grid_y = grid_y.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    #
    grid = torch.cat((grid_x, grid_y), 1).float().cuda()
    # grid is: [batch, channel (2), height, width]
    vgrid = grid - motion
    # Grid values must be normalised positions in [-1, 1]
    vgrid_x = vgrid[:, 0, :, :]
    vgrid_y = vgrid[:, 1, :, :]
    vgrid[:, 0, :, :] = (vgrid_x / width) * 2.0 - 1.0
    vgrid[:, 1, :, :] = (vgrid_y / height) * 2.0 - 1.0
    # swapping grid dimensions in order to match the input of grid_sample.
    # that is: [batch, output_height, output_width, grid_pos (2)]
    vgrid = vgrid.permute((0, 2, 3, 1))
    warped = F.grid_sample(pre, vgrid, align_corners=True)

    # return warped
    oox, ooy = torch.split((vgrid < -1) | (vgrid > 1), 1, dim=3)
    oo = (oox | ooy).permute(0, 3, 1, 2)
    # ed = time.time()
    # print('warp {}'.format(ed-st))
    return torch.where(oo, cur, warped)


def warp_img(image: np.ndarray, mv: np.ndarray) -> np.ndarray:
    # move pixel values to be between -1 and 1
    mv = (mv - 0.5) * 2
    print(mv.shape)
    mv_r = mv[:, :, 0]
    mv_g = mv[:, :, 1]
    mv_r = mv_r * -1
    # mv_g = mv_g * -1

    mv = np.stack([mv_r, mv_g], axis=-1)

    image = torch.tensor(image).unsqueeze(0).cuda()
    mv = torch.tensor(mv).unsqueeze(0).cuda()

    image = image.permute(0, 3, 1, 2)
    mv = mv.permute(0, 3, 1, 2)

    return np.transpose(warp(image, mv).squeeze(0).cpu().numpy(), (1, 2, 0))


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
    current_mv_path = 'Test2/FinalImageM_Velocity.0011.exr'
    previous_mv_path = 'Test2/FinalImageM_Velocity.0010.exr'

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
    threshold = 0.02  # You may need to adjust this value
    motion_mask = (motion_diff_normalized > threshold).astype(np.uint8)

    # Display the motion mask
    plt.imshow(motion_mask, cmap='gray')
    plt.title('Motion Mask')
    plt.colorbar()
    plt.show()


def main() -> None:

    generate_mask()

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
