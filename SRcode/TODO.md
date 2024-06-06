# TODO

Just some notes and todos what I'd like to consider for my thesis.

## Network

### ExtraNet:
- [x] Spatial SR for Single Image Super Resolution (SISR)
- [ ] Spatial SR for Real-Time Super Resolution (RTSR) -> multiple input frames
- [ ] Temporal SR

### ExtraSS
- [ ] Spatial SR for Single Image Super Resolution (SISR)
- [x] Spatial SR for Real-Time Super Resolution (RTSR) -> multiple input frames
- [ ] Temporal SR
- [x] maybe other dataloader since only 1 frame is generated per forward pass

### Space-time Supersampling (STSS):
- [ ] Spatial SR for Single Image Super Resolution (SISR)
- [x] Spatial SR for Real-Time Super Resolution (RTSR) -> multiple input frames
- [ ] Temporal SR
- [x] find out how to train STSS correctly (two forward passes: 1. SS frame, 2. ESS frame)
- [ ] implement their loss
- [x] for now generate two images based on one forward pass and have an output of 6
- [ ] later try it with two different forward passes but update the weights (backward) after each forward pass and NOT together
- [x] instead of the hole inpaint function use a small attention layer from [here](https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py)
- [x] train without buffers on my data set and cross validate on Lewis scene and look how good it performs

### Flow-Agnostic Video Representations for Fast Frame Interpolation (FLAVR)
- [ ] Spatial SR for Single Image Super Resolution (SISR)
- [x] Spatial SR for Real-Time Super Resolution (RTSR) -> multiple input frames
- [x] Temporal SR

### Efficient Video Restoration (EVRNet)
- [ ] Spatial SR for Single Image Super Resolution (SISR)
- [ ] Spatial SR for Real-Time Super Resolution (RTSR) -> multiple input frames
- [ ] Temporal SR

### General:
- [ ] Look into fraction based values (DLSS, XESS, FSR -> quality modes, like 1.3, 1.5, 1.7, 2.0, 3.0)
- [x] U-net architectures up/down-sample an image multiple times, look into what is best practice when the input image sizes are not dividable by 2^x where x the number of up/down-sampling steps
- [x] adjust training details to ExtraNet's parameter (Cosine learning rate decay, beta 1 and 2 of adam)
- [x] add a 'how often is the image divided' number to every model so it can be abstracted for train, evaluate and test
- [x] checkout the formula in BasicSR for calculating the epochs amount based on iterations/dataset.size (check if batch size influences sth here)
- [ ] abstract config and train so it can be trained on SISR and MISR (includes Spatial SR as well as Temporal SR)
- [ ] abstract code base so that networks can be trained with different amount of buffer data

### Training
At the moment training is quite slow as we need to read 4 png images per get_item for LR (1080p) and two for HR (4k)
- [x] check how Eduard handled reading such big images from disk -> cv2 imread
- [ ] cv2 still too slow, try to compress all training and val data into npz and pt files and see how much faster it would be (XTX PC!)
- [ ] find more ways to speed up the loading from disk thing..

### Timing:
A forward pass of our network should at max take 33.3ms.
If it takes two forward passes (one for SS, one for ESS) it only should take 16.6ms.. 

| FPS       | Forward Pass Time (ms) |
|-----------|-------------------------|
| 24        | 41.6                    |
| 30        | 33.3                    |
| 60        | 16.6                    |
| 120       | 8.3                     |


## Data Loader

- [x] test if preparing data structure is faster than reading from disk for every item
- [x] get name of lr/hr image
- [x] [crop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html) images into smaller parts so more can be processed in parallel (batch size > 1)
- [x] crop/pad image to hr output size if needed (padding for image when there dimensions are not dividable by 2^x where x the number of up/down-sampling steps)

For training:
- [x] flip images (hflip)
- [x] rotate images (by vflip + 90 degree = 0, 90, 180, 270) -> total augmentation times 8 (800 * 8 = 6400)

Random Cropping images and batching them does not increase the speed of my training (bottleneck is most likely the loading image from path process)
Boost Performance by:
- [ ] preprocessing the image pairs as tensor in one .npz file and load them
- [x] preprocessing the image pairs as tensor in one .pt file and load them
- [x] use a similiar logic like BasicSR by using a Custom Data Sampler
- [x] use Prefetcher to load data while processing data (again similiar to BasicSR)

Make a similiar data loader to STSS
- [x] data loader loads lr frame, features, history frames and hr frame
- [x] crop and augment all of these files
- [x] regarding training of STSS we want to load the data for SS frame and ESS frame together but make two seperate forward passes
- [x] mind the 30 fps for history data when init filenames!
- [x] Make multi Image dataloader as well as STSS dataloader 

Abstract data loader so that:
- [ ] the multi image dataloader can load multiple images as well as the buffers
- [ ] the stss dataloader can load optionally load history and buffers

### Video
- [ ] Load and process the [Reds](https://seungjunnah.github.io/Datasets/reds.html) dataset with x4 for simplicity

### Preprocess

- [x] Divide images into sub images and save them as .png and pt/npz files
- [x] For the config file add a file type option, based on that the dataloader loads, .png, .pt or .npz files

*png for size, npz for speed -> roughly 20% increase of speed to 20% increase of size*

- [x] Play around with loading image data for depth for example and min/max normalize the values and safe it back into an image
- [x] Load a motion vector image and play around if you can visualize the motion vector data (-x/-y m/s, x,y m/s) into a color range which makes negative values visible
- [x] for depth try to see if we can put it on a logarithmic scale

*logaritmic scale for depth not bad, because objects don't disappear in the fog, but might lose detail, again tradeoff here*

- [x] honestly for motion vectors we can try logarithmic scale as well -> looks good to be honest :9
- [ ] warp lr frames based on motion vector data, the idea: we move the pixel in image space (x,y) based on the values of the R (x) and G (y) channel values 
-> (this data needs to be normalized again into values of -1 to 1, atm we have RGB so in range of 0 to 1)
- [ ] take warped lr images into the network instead of just lr images
- [ ] also warp the history frames!
- [ ] based on the buffers generate a mask and add this mask to the gated convolution!!

## General

- [x] abstract loading different networks based on model path / string
- [x] only use DIV2K to train and evaluate/test
- [x] add a validation to the training script
- [x] make it easy to train with no patchsize (crop/pad if input images are not suitable for down-/up-sampling multiple times) maybe add a variable into the model files for that?
- [ ] safe the time it took to train the model in hours:min inside the result file
- [x] play around with tensor board to visualize training process (loss, SSIM, PSNR, images?)
- [x] abstract get item fct for loaders to either take ".png", ".pth" or ".npz"

AFTER the extrapolated frames are nearer to the values of the SS frame:
- [ ] boost the values of both with [**RepRB**](https://arxiv.org/pdf/2404.16484) Convolution
- [ ] add a [**texture consistency loss**](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Exploring_Motion_Ambiguity_and_Alignment_for_High-Quality_Video_Frame_Interpolation_CVPR_2023_paper.pdf) in addition to the L1 loss
- [ ] try the auto encoder [stuff](https://arxiv.org/pdf/2111.06377)

### Config

```yaml
MODEL: Flavr
EPOCHS: 150
SCALE: 2
BATCH_SIZE: 8
CROP_SIZE: 256
USE_HFLIP: True
USE_ROTATION: True
NUMBER_WORKERS: 8
LEARNING_RATE: 0.001
CRITERION: L1
OPTIMIZER:
  NAME: Adam
  BETA1: 0.9
  BETA2: 0.99
SCHEDULER:
  NAME: Cosine
  MIN_LEARNING_RATE: 1.0e-06
  START_DECAY_EPOCH: 20
DATASET: matrix
```

- [x] valid config file -> yaml parser -> make class called config, load from yaml
- [x] model_name, scheduler, optimizer -> class object return
- [x] add if the train/val dataset is either Single Image Pair or Multi Image Pair
Support only certain datasets:
- [x] DIV2K
- [ ] REDS
- [x] Urban 100
- [x] Set 5 & 14
- [x] Matrix

### Nice to know

- SRCNN should have a psnr score of ~32 dB after 100 epochs (while gradually reducing learning rate from 0.01 to 0.0001)
- train networks for x2 from scratch use x3, x4 the pretrained x2 network
- Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR), also has a multi-scale deep super-resolution (MDSR), with different scale based blocks that only train on data designed for its upscaling proportions (x2, x3, x4)
- Rough estimation on CNN architecture's # of params: # of layers = depth, # of feature channels = width, memory allocated is O(depth * width) and # params is O(depth * width^2)


## Dataset

We want to create our own dataset in Unreal Engine 5.3:
-[x] LR frames + associated HR frames
-[x] LR depth buffer
-[x] LR geometry buffer
-[x] LR motion vectors
-[ ] last drawn LR and/or upsampled HR frames
- upsampled HR frames from
- [ ] DLSS
- [ ] FSR
- [ ] XeSS
- [ ] (TSR)

Generate a dataset from the matrix scene:
- [x] one for training
- [x] one for testing/evaluating
- [x] use png files for buffers and frames
- [ ] generate very high quality renders (SSAA quality) so for 4k we need 16k for example

Generate a higher quality dataset for matrix:
- not sure how big we should make the dataset
- more scenes are better than one big data set
- 900 frames for LR (+buffers) and HR are around 30GB
- [ ] 8-20 training scenes
- [x] 2-5 test scenes

Generate a more divers datasaet:
- [x] matrix
- [x] electric dreams env
- [ ] electric dreams lighting issue LR and HR not identical!
- [x] light issue fixed (post process + very high lumen value of directional light)
- [ ] redo electric dreams env sequences!!!
- [x] hillside
- [x] ancient valley
- 12 sequences for training, 4 for val
- each sequence around 300 frames with 60 fps -> 5 sec

For testing:
- [x] stylized content (asian village, medieval)
- 4 sequences

### Typical render resolutions

4k

| Input       | Scale | Target        |
|-------------|-------|---------------|
| 2953 x 1661 | 1.3   | 3840 x 2160   |
| 2560 x 1440 | 1.5   | 3840 x 2160   |
| 2258 x 1270 | 1.7   | 3840 x 2160   |
| 1920 x 1080 | 2     | 3840 x 2160   |
| 1280 x 720  | 3     | 3840 x 2160   |
| 960 x 540   | 4     | 3840 x 2160   |

WQHD

| Input       | Scale | Target        |
|-------------|-------|---------------|
| 1969 x 1107 | 1.3   | 2560 x 1440   |
| 1706 x 960  | 1.5   | 2560 x 1440   |
| 1505 x 807  | 1.7   | 2560 x 1440   |
| 1280 x 720  | 2     | 2560 x 1440   |
| 853 x 480   | 3     | 2560 x 1440   |
| 640 x 360   | 4     | 2560 x 1440   |

Full HD

| Input      | Scale | Target        |
|------------|-------|---------------|
| 1476 x 830 | 1.3   | 1920 x 1080   |
| 1280 x 720 | 1.5   | 1920 x 1080   |
| 1129 x 635 | 1.7   | 1920 x 1080   |
| 960 x 520  | 2     | 1920 x 1080   |
| 640 x 360  | 3     | 1920 x 1080   |
| 480 x 270  | 4     | 1920 x 1080   |

HD

| Input     | Scale | Target        |
|-----------|-------|---------------|
| 984 x 553 | 1.3   | 1280 x 720    |
| 853 x 480 | 1.5   | 1280 x 720    |
| 752 x 423 | 1.7   | 1280 x 720    |
| 640 x 360 | 2     | 1280 x 720    |
| 426 x 240 | 3     | 1280 x 720    |
| 320 x 180 | 4     | 1280 x 720    |


## Evaluation

### General
- [x] Load model based on pretrained model filename
- [x] Clean up evalaute.py by removing the comparison of bilinear, bicubic and the network
- [x] Load config files into pandas dataframe
- [x] Save dataframe into csv file
- [ ] fix eval script to work with Multi Image Pair
- [ ] abstract eval script to work with Single Images as well as Multi Images

Single Image Super Resolution (SISR) eval:
- [x] Set5
- [x] Set14
- [x] Urban100

Video Image Super Resolution with frame gen eval:
- [ ] Reds for frame generation
- [ ] Matrix data set (our own)
- [ ] maybe STSS net

### Results
After calling evaluate.py with config.yaml we want to create a result.yaml file containing:
- [x] the train details (same as config.yaml)
- [x] results in form of for each dataset PSNR and SSIM value
- [x] write a results.py script, which generates a csv file based on all result files
- [x] add "pseudo" entry results for bilinear and bicubic (will have None for model details only name)
- [ ] add a test script for spatial and temporal SR so that we safe the Spatial Temporal SR images from a test sequence and display it as a video

### Frame Generation
- [x] write a script which generates a video based on pngs
- [ ] write a test script which simulates a 30 fps 1080p image flow to generate a 60 fps 4k image flow
- [ ] abstract the test script to safe the images per sequence in sub folders beginning with the right index

### Metrics
We want to evaluate on:
- [x] PSNR
- [x] SSIM all channels -> mean
- [ ] SSIM only y-channel
- [ ] make PSNR and SSIM work with multiple output images from network to gt at once

Calc PSNR, SSIM in Multi Image:
- [x] for two images
- [x] plus the average
- [ ] abstract the number of images based on the second dimension of the output tensor ([8, 2, 3, 1920, 1080])

Against:
- [x] Bilinear
- [x] Bicubic
- [ ] DLSS
- [ ] FSR
- [ ] XeSS
- [ ] TSR


## Code Quality
Add Tests for:
- [ ] train.py
- [ ] evaluate.py
- [ ] test.py
- [x] datalaoder.py
- [ ] config.py
- [x] utils.py
- [ ] results.py
- [ ] size.py

### Documentation:
Add documentation for fcts in form of javadoc strings:
- [ ] train.py
- [ ] evaluate.py
- [ ] test.py
- [ ] datalaoder.py
- [ ] config.py
- [ ] utils.py
- [ ] results.py
- [ ] size.py