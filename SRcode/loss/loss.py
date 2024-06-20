import torch
import torch.nn as nn
import torch.nn.functional as F


def census_transform(img, kernel_size=3):
    """
    Calculates the census transform of an image of shape [N x C x H x W] with batch size N, number of channels C,
    height H and width W. If C > 1, the census transform is applied independently on each channel.

    :param img: input image as torch.Tensor of shape [H x C x H x W]
    :return: census transform of img
    """
    assert len(img.size()) == 4
    if kernel_size != 3:
        raise NotImplementedError

    n, c, h, w = img.size()

    census = torch.zeros((n, c, h - 2, w - 2), dtype=torch.uint8, device=img.device)

    cp = img[:, :, 1:h - 1, 1:w - 1]
    offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

    # do the pixel comparisons
    for u, v in offsets:
        census = (census << 1) | (img[:, :, v:v + h - 2, u:u + w - 2] >= cp).byte()

    return torch.nn.functional.pad(census.float() / 255, (1, 1, 1, 1), mode='reflect')


class CensusTransform(torch.nn.Module):
    """
    Calculates the census transform of an image of shape [N x C x H x W] with batch size N, number of channels C,
    height H and width W. If C > 1, the census transform is applied independently on each channel.

    :param img: input image as torch.Tensor of shape [H x C x H x W]
    :return: census transform of img
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self._kernel_size = kernel_size

    def forward(self, x):
        x = census_transform(x, self._kernel_size)
        return x


class CensusLoss(nn.Module):
    def __init__(self, kernel_size=3):
        super(CensusLoss, self).__init__()
        self.census_transform = CensusTransform(kernel_size)

    def forward(self, predicted, target):
        census_predicted = self.census_transform(predicted)
        census_target = self.census_transform(target)
        loss = F.mse_loss(census_predicted, census_target)
        return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff**2 + self.epsilon**2)
        return torch.mean(loss)


# loss from the following paper https://openaccess.thecvf.com/content/WACV2023/papers/Jin_Enhanced_Bi-Directional_Motion_Estimation_for_Video_Frame_Interpolation_WACV_2023_paper.pdf
class EBMELoss(nn.Module):
    def __init__(self):
        super(EBMELoss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss()
        self.census_loss = CensusLoss()

    def forward(self, x, y):
        return self.charbonnier_loss(x, y) + 0.1 * self.census_loss(x, y)
