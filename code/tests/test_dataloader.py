import unittest
from code.dataloader import *
import numpy as np


class TestDataLoader(unittest.TestCase):

    def test_random_crop(self):
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

    def test_rotate_image(self):
        img = torch.tensor([[[1, 2], [3, 4]]])
        rotated_img = rotate_image(img, 90)
        expected_rotated_img = torch.tensor([[[2, 4], [1, 3]]])

        # Check if the rotated image is as expected
        self.assertTrue(torch.equal(rotated_img, expected_rotated_img),
                        f"Rotated image {rotated_img} and expected {expected_rotated_img}")


if __name__ == '__main__':
    unittest.main()
