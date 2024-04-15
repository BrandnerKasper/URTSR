import unittest

import torch

from utils.utils import *


class TestMetrics(unittest.TestCase):
    def test_add_single_image_pair(self):
        m1 = Metrics([0.3], [0.8])
        m2 = Metrics([0.1], [0.2])
        m3 = m1+m2
        m0 = Metrics([0.4], [1.0])
        self.assertEqual(m3, m0, f"Got {m3} but expected {m0}")

    def test_add_multi_image_pair(self):
        m1 = Metrics([0.3, 0.7], [0.8, 0.6])
        m2 = Metrics([0.1, 0.2], [0.2, 0.1])
        m3 = m1+m2
        m0 = Metrics([0.4, 0.9], [1.0, 0.7])
        self.assertEqual(m3, m0, f"Got {m3} but expected {m0}")

    def test_divide_single_image_pair(self):
        m1 = Metrics([0.3], [0.8])
        m2 = m1 / 2
        m0 = Metrics([0.15], [0.4])
        self.assertEqual(m2, m0, f"Got{m2} but expected {m0}")

    def test_divide_multi_image_pair(self):
        m1 = Metrics([0.3, 0.7], [0.8, 0.6])
        m2 = m1 / 2
        m0 = Metrics([0.15, 0.35], [0.4, 0.3])
        self.assertEqual(m2, m0, f"Got {m2} but expected {m0}")

    def test_calculate_metrics_single_image_pair(self):
        img_tensor_1 = torch.rand(8, 3, 1920, 1080) # assume we have batch size of 8
        img_tensor_2 = torch.rand(8, 3, 1920, 1080)
        m1 = calculate_metrics(img_tensor_1, img_tensor_2)
        print(m1)
        self.assertEqual(len(m1.psnr_values), 1, f"Expected one entry, but got {len(m1.psnr_values)} entries")

    def test_calculate_metrics_multi_image_pair(self):
        img_tensor_1 = torch.rand(8, 2, 3, 1920, 1080) # assume we have batch size of 8
        img_tensor_2 = torch.rand(8, 2, 3, 1920, 1080) # second dimension assumes how many images we get from network
        m1 = calculate_metrics(img_tensor_1, img_tensor_2)
        print(m1)
        self.assertEqual(len(m1.psnr_values), 2, f"Expected two entries, but got {len(m1.psnr_values)} entries")


if __name__ == '__main__':
    unittest.main()
