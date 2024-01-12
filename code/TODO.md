# TODO

-[ ] [crop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html) images into smaller parts so more can be processed in parallel (batch size > 1)
-[ ] play with subpixel super res 
- [ ] SRCNN should have a psnr score of ~32 dB after 100 epochs (while gradually reducing learning rate from 0.01 to 0.0001)