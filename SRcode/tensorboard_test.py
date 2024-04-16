import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader

from data.dataloader import *

writer = SummaryWriter(filename_suffix="AAA", comment="BBB")

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

img_batch = torch.zeros((4, 2, 3, 100, 100))
img_list = torch.unbind(img_batch, dim=1)
for i in range(len(img_list)):
    writer.add_images(f'my_image_batch_{i}', img_list[i], 0)
    writer.close()

root = "dataset/matrix/val"
# Matrix dataset contains 4 sequences of 1500 images starting from 0000 to 1499
matrix_dataset = MultiImagePair(root=root, scale=2, number_of_frames=4,
                                        last_frame_idx=1499,
                                        crop_size=256, use_hflip=False, use_rotation=False, digits=4)
matrix_dataloader = DataLoader(dataset=matrix_dataset, batch_size=8, shuffle=False, num_workers=8)

counter = 0
print("Entering loop")
for lr_image, hr_image in matrix_dataloader:
    if counter < 42:
        counter += 1
        continue
    lr_list = torch.unbind(lr_image, 1)
    hr_list = torch.unbind(hr_image, 1)
    for i in range(len(lr_list)):
        writer.add_images(f"Images/lr/{i}", lr_list[i], 0)
    for i in range(len(hr_list)):
        writer.add_images(f"Images/hr/{i}", hr_list[i], 0)
    break
