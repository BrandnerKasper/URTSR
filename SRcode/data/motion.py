import cv2
import numpy as np


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


def main() -> None:
    motion_diff()
    # motion_edge_diff()


if __name__ == "__main__":
    main()
