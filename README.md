# L2G Lensless Imaging

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/ArXiv-2512.00488-b31b1b.svg)](https://arxiv.org/abs/2512.00488)
[![Project](https://img.shields.io/badge/Project-Page-0066CC)](https://y1248.github.io/pub_homepage/l2g/index.html)

**Paper**:
Large-field-of-view lensless imaging with miniaturized sensors.

**Authors**:
Yu Ren, Xiaoling Zhang, Xu Zhan, Xiangdong Ma, Yunqi Wang, Edmund Y. Lam, and Tianjiao Zeng.

**Abstract**:
Lensless cameras replace bulky optics with thin modulation masks, enabling compact imaging systems. However, existing methods rely on an idealized model that assumes a globally shift-invariant point spread function (PSF) and sufficiently large sensors. In reality, the PSF varies spatially across the field of view (FOV), and finite sensor boundaries truncate modulated light—effects that intensify as sensors shrink, degrading peripheral reconstruction quality and limiting the effective FOV. We address these limitations through a local-to-global hierarchical framework grounded in a locally shift-invariant convolution model that explicitly accounts for PSF variation and sensor truncation. Patch-wise learned deconvolution first adaptively estimates local PSFs and reconstructs regions independently. A hierarchical enhancement network then progressively expands its receptive field—from small patches through intermediate blocks to the full image—integrating fine local details with global contextual information. Experiments on public datasets show that our method achieves superior reconstruction quality over a larger effective FOV with significantly reduced sensor sizes. Under extreme miniaturization—sensors reduced to 8\% of the original area—we achieve improvements of 2 dB (PSNR) and 5\% (SSIM), with particularly notable gains in structural fidelity.

## Results
### Global-wise Deconv v.s. Patch-wise Deconv
![](./imgs/demo_results_deconv.png)

### Comparison with other representative methods 
![](./imgs/demo_results_full.png)

## Note

Currently, this work is under review, and the associated agreement document will be made available soon.

## Todo

Due to the ongoing review process, the code, particularly those reconstruction modules, is not yet available for open-source distribution. It still requires detailed documentation and code cleanup. These contents are still in progress before the code can be made available.

## Done

We have open-sourced the part of code for the utility toolkit used in our research process, which contains some fundamental operations during processing lensless imaging data, as its structure is relatively clearer and requires minimal documentation.

## Acknowledgements

This study involves two publicly available datasets: *[DiffuserCam](https://waller-lab.github.io/LenslessLearning/dataset.html)* and *[PhlatCam](https://siddiquesalman.github.io/flatnet/)*. We express our gratitude to all the authors who make these resources publicly available.