# Neural Image Super-Resolution in Gaming
This is the repository for the master thesis related to super-resolution in the gaming context.

## Related Work

### Image Super-Resolution
This is a list of relevant SR papers which are trained for bicubic degradation. There are different degradation models which capture different aspects of the super-resolution tasks, e.g. different blur kernels and noise levels. However, in this context, it is only neccessary to focus on the traditional Single Image Super-Resolution (SISR) setting.

Given the motivation of this thesis, transformer-based methods are neglected as they struggle with efficient inference for large input images (SR from 1080p -> 4K).

Basics (only CNN based):
- SRCNN [Link](https://arxiv.org/abs/1501.00092)
- FSRCN [Link](https://arxiv.org/abs/1608.00367)
- **Sub-Pixel Convolution** [Link](https://arxiv.org/abs/1609.05158) x
- **EDSR** [Link](https://arxiv.org/abs/1707.02921)
- **RCAN** [Link](https://arxiv.org/abs/1807.02758)
- HAN [Link](https://arxiv.org/pdf/2008.08767.pdf)


Efficiency Focus:
- IMDN [Link](https://arxiv.org/pdf/1909.11856.pdf)
- **RFDN** [Link](https://arxiv.org/pdf/2009.11551.pdf)
- **RLFN** [Link](https://arxiv.org/pdf/2205.07514.pdf)
- LAPAR [Link](https://papers.nips.cc/paper/2020/file/eaae339c4d89fc102edd9dbdb6a28915-Paper.pdf)
- **RepSR** [Link](https://arxiv.org/pdf/2205.05671.pdf)
- ELAN [Link](https://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV_2022_ELAN.pdf)
- SAFMN [Link](https://arxiv.org/pdf/2302.13800.pdf)
- ShuffleMixer [Link](https://arxiv.org/pdf/2205.15175.pdf)
- NTIRE 2023 Real-Time 4K SR [Link](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.pdf) x
- NTIRE 2023 Efficient SR [Link](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Li_NTIRE_2023_Challenge_on_Efficient_Super-Resolution_Methods_and_Results_CVPRW_2023_paper.pdf)

### Neural Super-Sampling
- Neural supersampling for real-time rendering [Link](https://dl.acm.org/doi/abs/10.1145/3386569.3392376)
- **Efficient neural supersampling on a noval gaming dataset** [Link](http://openaccess.thecvf.com/content/ICCV2023/papers/Mercier_Efficient_Neural_Supersampling_on_a_Novel_Gaming_Dataset_ICCV_2023_paper.pdf) x
- **FuseSR** [Link](https://arxiv.org/abs/2310.09726)
- SSR [Link](https://arxiv.org/pdf/2301.01036.pdf)
- **ExtraNet** [Link](https://dl.acm.org/doi/abs/10.1145/3478513.3480531)
- **Space-time Supersampling** [Link](https://arxiv.org/pdf/2312.10890v1.pdf)
- **ExtraSS: Frame Extrapolation** [Link](https://dl.acm.org/doi/pdf/10.1145/3610548.3618224) x

### Multi-Frame Super-Resolution
- MFSR [Link](https://dl.acm.org/doi/pdf/10.1145/3306346.3323024)
- DBSR [Link](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf)
- RBSR [Link](https://arxiv.org/abs/2306.17595)
### Temporal Super-Resolution
- TemporalSR [Link](https://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/supplementary/AcrossScalesAndDimensions_ECCV2020.pdf)

## Code Frameworks
- [PyTorch](https://pytorch.org) x
- [Basics of PyTorch](https://pytorch.org/tutorials/) x
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
