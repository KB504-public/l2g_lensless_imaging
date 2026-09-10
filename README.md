# L2G Lensless Imaging

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/ArXiv-2512.00488-b31b1b.svg)](https://arxiv.org/abs/2512.00488)
<!-- [![Project](https://img.shields.io/badge/Project-Page-0066CC)](https://y1248.github.io/pub_homepage/l2g/index.html) -->

**Paper**:
Lensless imaging under limited measurement support via local-to-global reconstruction

**Authors**:
Yu Ren, Tianjiao Zeng, Xu Zhan, Xiangdong Ma, Yunqi Wang, Xiaoling Zhang, and Jun Shi.

**Abstract**:
Lensless cameras enable compact imaging by replacing bulky optics with thin modulation masks. However, most existing reconstruction methods typically assume a globally shift-invariant point spread function (PSF) and complete measurement acquisition, neglecting spatial PSF variation and measurement truncation that become increasingly important under limited measurement support. To address these challenges, we propose a local-to-global reconstruction method. A model-motivated local measurement formulation is first introduced to characterize spatially varying image formation when only limited measurement support is available. Guided by this formulation, locality-constrained patch-wise deconvolution assigns independently learnable inverse filters to prescribed scene regions, while a hierarchical enhancement network progressively aggregates contextual information from local patches to the full image. Experiments on public datasets and a lensless imaging prototype demonstrate improved reconstruction fidelity over larger usable reconstruction regions under restricted measurement support. When only 8\% of the original measurement area is retained, the proposed method improves PSNR by over 2 dB and SSIM by about 6\% over the strongest baseline, demonstrating its potential for compact lensless imaging with limited sensor coverage.

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