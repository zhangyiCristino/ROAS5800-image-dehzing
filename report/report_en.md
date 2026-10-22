> Public repository edition: source images, saved output images and binary PDF/PPT are not redistributed. The measured summary/CSV is included; run the README workflow to regenerate the full evaluation record and image browser.

# Local Image Dehazing: Three Recent Pretrained Systems

ROAS5800 Assignment 1 · Reference draft (unsigned) · 19 October 2026

## Abstract

This assignment implements a local image-dehazing interface and compares three released pretrained systems: PGL-Net-T, HNDiff-FocalNet and PixRestore-S. All 352 aligned pairs in the source release's RW2AH test split were evaluated with common RGB preprocessing on an RTX 4060 Laptop GPU. PGL-Net-T provided the highest mean PSNR (22.82 dB), while PGL-Net-T was fastest (28.17 ms median forward time). The experiment compares deployable systems; it does not reproduce original paper protocols or establish a global SOTA ranking.

## 1. Task and motivation

The task is to recover a clear RGB image from a single hazy image. A common idealized scattering model is `I(x) = J(x)t(x) + A(1-t(x))`, where I is the hazy image, J is clear scene radiance, t is transmission and A is atmospheric light. Haze weakens contrast and changes colour, motivating a comparison of quality and computational cost. This project implements three recent low-level vision algorithms and a common Gradio interface. The work comprises integration, shared preprocessing, evaluation and user interaction; no model was trained or proposed here. Robot perception and Real-to-Sim reconstruction are possible future applications, not evaluated outcomes.

## 2. Three algorithms

- **PGL-Net-T [1]:** a lightweight deterministic network that separates global distribution correction from local structural refinement. Physics-Inspired Affine Fusion supplies global conditioning, and Degradation-Aware Modulation adapts local detail restoration. The released RW2AH Tiny checkpoint contains 0.78 M parameters.
- **HNDiff-FocalNet [2]:** Latent HNDiff uses haze-noise diffusion to obtain a clean latent prior for the FocalNet restoration backbone. Both released RW2AH checkpoints are loaded and all four reverse steps, `[4,3,2,1]`, are retained. The loaded pair has 14.42 M parameters; the 7.82 M count of a different paper configuration does not describe this implementation.
- **PixRestore-S [3]:** a unified restoration system using a pixel-space transformer and DINOv2 feature conditioning. The released S checkpoint uses one sampling step. Its 53.70 M loaded parameters include the DINOv2 ViT-S/14 encoder; it is a general restoration checkpoint rather than an RW2AH-specialized model.

## 3. Interface and implementation

Users upload a hazy image, select one or more methods, and compare input/output sliders. A same-size aligned clear reference is optional; PSNR and SSIM are shown only when it is provided. A downloadable ZIP contains the effective input, restored PNGs, a CSV and run settings. Models run locally without a cloud API. Only one model is resident at a time to reduce memory use. Official model definitions and pretrained weights are retained; custom code supplies adapters, evaluation and the interface.

## 4. Evaluation protocol

The dataset is RW2AH [4], downloaded from the release linked by PGL-Net. **All 352 paired test images** were included, with source filenames and SHA-256 hashes verified. Input and reference were converted to RGB and downsampled identically with PIL BICUBIC, preserving aspect ratio and limiting the longest side to 512; smaller images were not enlarged. Bottom/right reflection padding to multiples of 64 was removed after inference, without an additional evaluation crop.

PGL-Net-T and HNDiff used FP32, whereas PixRestore-S used CUDA BF16 autocast. The fixed seed was 0. Each image was run three times; quality was computed from the first saved, clipped and rounded uint8 RGB output. PSNR used data range 1 after division by 255. SSIM used an 11 × 11 Gaussian window, sigma 1.5, population covariance and all RGB channels. No Y conversion, border shaving or histogram matching was applied. Quality scores are arithmetic means of per-image scores. GPU time is the median over three complete forwards per image, then the median across images; loading, warm-up, transfers and file/UI operations are excluded. Peak memory is the maximum PyTorch allocated tensor memory, not total GPU usage.

Hardware/software: RTX 4060 Laptop 8 GB, Ryzen 9 7845HX, approximately 16 GB system RAM; Windows, Python 3.12.10 and PyTorch 2.8.0+cu128. HNDiff uses common full-image inference here, rather than its official 384-pixel overlapping-tile evaluation.

## 5. Results and discussion

| Method | PSNR (dB) ↑ | SSIM ↑ | MAE ↓ | Median GPU (ms) ↓ | Peak allocated (MiB) | Parameters (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated | 19.01 | 0.5817 | 0.0991 | — | — | — |
| PGL-Net-T | 22.82 | 0.6489 | 0.0619 | 28.17 | 175.8 | 0.78 |
| HNDiff-FocalNet | 22.36 | 0.6640 | 0.0655 | 247.89 | 271.5 | 14.42 |
| PixRestore-S | 18.60 | 0.5706 | 0.1025 | 69.59 | 314.3 | 53.70 |

PGL-Net-T obtained the highest mean PSNR (22.82 dB), 3.81 dB above untreated inputs. HNDiff-FocalNet obtained the highest mean SSIM (0.6640). PGL-Net-T had the lowest median GPU forward time (28.17 ms); HNDiff took 8.8 times the median time of PGL-Net-T. PixRestore-S fell below untreated inputs by 0.41 dB PSNR and 0.0111 SSIM on average. These observations apply to the released checkpoints under this local protocol.

The original local submission illustrated two fixed test IDs `1-0010-02` and `1-0013-01`, not a representative sample of all scenes. They support visual inspection alongside the full-set metrics. The numerical ordering must not be attributed to architecture alone: PGL-Net and HNDiff use dataset-specific weights, while PixRestore-S uses a general multi-task checkpoint, and inference precisions differ. The resized full-image protocol also differs from the original papers. No ablation isolates the role of individual modules.

The fixed-seed HNDiff repeats were not byte-identical for 327 of 352 inputs. The maximum absolute difference across repeated outputs was 1 on the uint8 0-255 scale; the first output remains the predeclared quality output. Fixed seeds therefore do not imply bit-exact GPU repeatability. PSNR/SSIM do not directly measure perceptual naturalness, geometric consistency or robot success. There were no robot, reconstruction or downstream task experiments.

## 6. Conclusion and reproducibility

The project provides a working three-method low-level vision UI and a complete local comparison on the source release's 352 RW2AH test pairs. Under the chosen deployment settings, PGL-Net-T obtained the best mean PSNR and PGL-Net-T the shortest median forward time. The result supports this checkpoint choice on this dataset and protocol, not a universal ranking. Future work can test whether restoration helps fixed feature matching or reconstruction pipelines.

After preparing the assets and dataset (see the repository README), run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m dehaze.evaluation --manifest data/manifest_full.json --output results_full --max-side 512 --runs 3 --seed 0
```

Evidence: `results_full/evaluation.json`, `summary.json`, `per_image.csv`, saved result images, dataset manifest, and `docs/REPRODUCIBILITY.md`. This is an editable reference; add your own name/student ID and verify the course submission rules before submission.

## References

1. Qu et al. Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling. arXiv:2606.25732, 2026. [Paper](https://arxiv.org/abs/2606.25732); [official code/data](https://github.com/sc-30-bit/PGL-Net).
2. He et al. HNDiff: Haze-Noise Diffusion for Image Dehazing. ECCV 2026; arXiv:2608.10995. [Paper](https://arxiv.org/abs/2608.10995); [official code/data](https://github.com/Jin-Ting-He/HNDiff).
3. Sun et al. PixRestore: Unified Image Restoration via Pixel Diffusion Transformer. arXiv:2608.16793, 2026. [Paper](https://arxiv.org/abs/2608.16793); [official code/data](https://github.com/csslc/PixRestore).
4. Fang et al. Guided Real Image Dehazing using YCbCr Color Space. AAAI 2025; arXiv:2412.17496. RW2AH dataset. [Paper](https://arxiv.org/abs/2412.17496); [official code/data](https://github.com/fiwy0527/AAAI25_SGDN).
