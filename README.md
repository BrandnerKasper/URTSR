# Neural Image Super-Resolution in Gaming
This is the repository for the master thesis related to super-resolution in the gaming context.

## Related Work

### Image Super-Resolution
This is a list of relevant SR papers which are trained for bicubic degradation. There are different degradation models which capture different aspects of the super-resolution tasks, e.g. different blur kernels and noise levels. However, in this context, it is only neccessary to focus on the traditional Single Image Super-Resolution (SISR) setting.

Given the motivation of this thesis, transformer-based methods are neglected as they struggle with efficient inference for large input images (SR from 1080p -> 4K).

Basics (only CNN based):
- [SRCNN](https://arxiv.org/abs/1501.00092)
- [Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158)
- [EDSR](https://arxiv.org/abs/1707.02921)
- [RCAN](https://arxiv.org/abs/1807.02758)


Efficiency Focus:
- [IMDN](https://arxiv.org/pdf/1909.11856.pdf)
- [RFDN](https://arxiv.org/pdf/2009.11551.pdf)
- [RLFN](https://arxiv.org/pdf/2205.07514.pdf)
- [LAPAR](https://papers.nips.cc/paper/2020/file/eaae339c4d89fc102edd9dbdb6a28915-Paper.pdf)
- [RepSR](https://arxiv.org/pdf/2205.05671.pdf)
- [ELAN](https://arxiv.org/pdf/2203.05568.pdf) 
- [NTIRE 2023 Real-Time 4K SR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.pdf)
- [NTIRE 2023 Efficient SR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Li_NTIRE_2023_Challenge_on_Efficient_Super-Resolution_Methods_and_Results_CVPRW_2023_paper.pdf)

### Neural Super-Sampling
- [Neural supersampling for real-time rendering](https://dl.acm.org/doi/abs/10.1145/3386569.3392376)
- [Efficient neural supersampling on a noval gaming dataset](http://openaccess.thecvf.com/content/ICCV2023/papers/Mercier_Efficient_Neural_Supersampling_on_a_Novel_Gaming_Dataset_ICCV_2023_paper.pdf)
- [FuseSR](https://arxiv.org/abs/2310.09726)
- [SSR](https://arxiv.org/pdf/2301.01036.pdf)
- [Space-time Supersampling](https://arxiv.org/pdf/2312.10890v1.pdf)

### Temporal Super-Resolution
- [TemporalSR](https://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/supplementary/AcrossScalesAndDimensions_ECCV2020.pdf)

## Code Frameworks
- [PyTorch](https://pytorch.org)
- [Basics of PyTorch](https://pytorch.org/tutorials/)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
