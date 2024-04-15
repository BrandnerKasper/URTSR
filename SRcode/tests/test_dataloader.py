import unittest

from data.dataloader import *


class TestDataLoader(unittest.TestCase):

    def test_random_crop_pair(self):
        # Input images
        lr_image = torch.tensor([[[1, 2], [3, 4]]])
        hr_image = torch.tensor([[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]])

        # Display input images
        print(f"LR image:\n{lr_image.numpy()}")
        print(f"HR image:\n{hr_image.numpy()}")

        # Parameters for random crop
        patch_size = 1
        scale = 2

        # Perform random crop
        lr_patch, hr_patch = get_random_crop_pair(lr_image, hr_image, patch_size, scale)

        # Extract the value from LR patch
        lr_patch_value = lr_patch.item()

        # Create the expected HR patch
        expected_hr_patch = torch.tensor([[[lr_patch_value, lr_patch_value], [lr_patch_value, lr_patch_value]]])

        # Assert value equality
        self.assertTrue(torch.equal(expected_hr_patch, hr_patch),
                        f"Expected:\n{expected_hr_patch.numpy()}\nActual:\n{hr_patch.numpy()}")

        # Display LR and HR patches along with expected HR patch
        print(f"LR patch:\n{lr_patch.numpy()}")
        print(f"HR patch:\n{hr_patch.numpy()}")
        print(f"Expected HR patch:\n{expected_hr_patch.numpy()}")

        # Assert shape equality
        expected_lr_patch_shape = (1, 1, 1)
        expected_hr_patch_shape = (1, 2, 2)
        self.assertEqual(lr_patch.shape, expected_lr_patch_shape,
                         f"Lr patch {lr_patch.shape} does not have expected shape {expected_lr_patch_shape}!")
        self.assertEqual(hr_patch.shape, expected_hr_patch_shape,
                         f"HR patch {hr_patch.shape} does not have expected shape {expected_hr_patch_shape}!")
        print(f"Actual shapes: ({lr_patch.shape}, {hr_patch.shape}")
        print(f"Expected shapes: ({expected_lr_patch_shape}, {expected_hr_patch_shape}")

    def test_flip_image_horizontal(self):
        img = torch.tensor([[[1, 2], [3, 4]]])
        flipped_img = flip_image_horizontal(img)
        expected_flipped_img = torch.tensor([[[2, 1], [4, 3]]])

        # Check if flip is as expected
        self.assertTrue(torch.equal(flipped_img, expected_flipped_img),
                        f"Flipped image {flipped_img} and expected {expected_flipped_img}")
        print(f"Img {img.numpy()}")
        print(f"Flipped img {flipped_img.numpy()}")
        print(f"Expected Img {expected_flipped_img.numpy()}")

    def test_flip_image_vertical(self):
        img = torch.tensor([[[1, 2], [3, 4]]])
        flipped_img = flip_image_vertical(img)
        expected_flipped_img = torch.tensor([[[3, 4], [1, 2]]])

        # Check if flip is as expected
        self.assertTrue(torch.equal(flipped_img, expected_flipped_img),
                        f"Flipped image {flipped_img} and expected {expected_flipped_img}")
        print(f"Img {img.numpy()}")
        print(f"Flipped img {flipped_img.numpy()}")
        print(f"Expected Img {expected_flipped_img.numpy()}")

    def test_rotate_image(self):
        img = torch.tensor([[[1, 2], [3, 4]]])
        rotated_img = rotate_image(img, 90)
        expected_rotated_img = torch.tensor([[[2, 4], [1, 3]]])

        # Check if the rotated image is as expected
        self.assertTrue(torch.equal(rotated_img, expected_rotated_img),
                        f"Rotated image {rotated_img} and expected {expected_rotated_img}")
        print(f"Img {img.numpy()}")
        print(f"Rotated img {rotated_img.numpy()}")
        print(f"Expected img {expected_rotated_img.numpy()}")

    def test_div2k_dataset(self):
        root = "../dataset/DIV2K/train"
        # DIV2K datast contains a single folder of 800 train images (ranging from 0001 to 0800)
        # and 100 validation images (ranging from 0801 to 0900)
        div2k_dataset = SingleImagePair(root=root, scale=2, pattern="x2", crop_size=128, use_hflip=False, use_rotation=False)
        div2k_dataset.display_item(0)

        first_entry_filename = div2k_dataset.get_filename(0)
        self.assertEqual(first_entry_filename, "0001", f"filename: {first_entry_filename}, expected 0001")

    def test_reds_dataset(self):
        root = "../dataset/Reds/train"
        # Reds dataset contains folders of exactly 100 images starting from 00000000 to 00000099
        reds_dataset = MultiImagePair(root=root, scale=4, crop_size=96, use_hflip=True, use_rotation=True, digits=8)
        reds_dataset.display_item(0)

        first_entry_filename = reds_dataset.get_filename(0)
        self.assertEqual(first_entry_filename, "00000003", f"filename: {first_entry_filename}, expected 00000003")

    def test_matrix_dataset(self):
        root = "../dataset/matrix/train"
        # Matrix dataset contains 4 sequences of 1500 images starting from 0000 to 1499
        matrix_dataset = MultiImagePair(root=root, scale=2, number_of_frames=4,
                                        last_frame_idx=1499,
                                        crop_size=256, use_hflip=False, use_rotation=False, digits=4)
        matrix_dataset.display_item(0)

        first_entry_filename = matrix_dataset.get_filename(0)
        self.assertEqual(first_entry_filename, "0003", f"filename: {first_entry_filename}, expected 0003")


if __name__ == '__main__':
    unittest.main()
