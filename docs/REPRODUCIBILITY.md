# Reproducibility and interpretation

This project compares three released pretrained image-restoration systems through a shared inference wrapper and an English Gradio interface. The recorded experiment was run on 19 October 2026. The contribution of this assignment is the integration, interface, common evaluation and analysis; the model architectures and pretrained weights belong to their original authors.

## What the published results cover

The main experiment uses **all 352 aligned test pairs** in the RW2AH release linked by the PGL-Net authors. The pairing, original archive member names and source-image SHA-256 hashes are recorded in `data/manifest_full.json`. It uses the held-out `test` members only. The 12-pair manifest is an earlier pilot selection and is not the source of the full-test means.

The full test produced 1,056 restored images, plus untreated inputs and aligned references. Each method was run three times per image. Quality is measured on the first output; the repeats measure timing and output consistency. Published tables contain measured local results, rather than values copied from the papers.

| Method | Test pairs | Mean PSNR (dB) ↑ | Mean SSIM ↑ | Median GPU forward (ms) ↓ | Maximum allocated tensor memory (MiB) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untreated | 352 | 19.0072 | 0.5817 | — | — |
| PGL-Net-T | 352 | 22.8203 | 0.6489 | 28.17 | 175.82 |
| HNDiff-FocalNet | 352 | 22.3631 | 0.6640 | 247.89 | 271.50 |
| PixRestore-S | 352 | 18.5979 | 0.5706 | 69.59 | 314.33 |

The machine used an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM, AMD Ryzen 9 7845HX, 16 GB RAM, Windows, Python 3.12.10 and PyTorch 2.8.0+cu128. Dependency versions are pinned in `requirements.txt`. Timing and memory are specific to this machine and configuration; allocated tensor memory is not total GPU memory usage.

PGL-Net-T achieved the highest mean PSNR and lowest latency among the tested models. HNDiff-FocalNet achieved the highest mean SSIM. PixRestore-S scored below the untreated input on both metrics in this experiment. This negative result is retained: a general restoration model is not guaranteed to improve dehazing on this test set under these settings.

## Common image and metric protocol

1. Decode each aligned input/reference pair as RGB. Reject pairs with different original dimensions.
2. Downsample both once using the same aspect-preserving PIL BICUBIC rule, with maximum side 512 pixels. Never upscale. Set the dimensions to `max(1, round(original_dimension * scale))`.
3. Pass the same prepared RGB uint8 image to all three adapters. Reflect-pad the bottom and right sides to multiples of 64; remove only that padding after inference. Keep the resized dimensions for evaluation.
4. Convert input ranges for each model: PGL-Net `[0, 1]`, HNDiff `[-0.5, 0.5]`, PixRestore `[-1, 1]`. Convert outputs to `[0, 1]`, clamp, multiply by 255, round and save as RGB uint8.
5. Compute metrics on the exact saved uint8 outputs, converted to float64 RGB by dividing by 255. PSNR uses `data_range=1`. SSIM uses `channel_axis=-1`, Gaussian weights, sigma 1.5, window 11 and population covariance (`use_sample_covariance=False`). SSIM averages its valid Gaussian-window interior as implemented by scikit-image. There is no Y-channel conversion, additional border shaving, histogram matching or post-hoc alignment.
6. Average per-image PSNR, SSIM and MAE with equal image weight. Mean PSNR is not PSNR computed from pooled MSE. With no aligned clear reference, the UI omits PSNR/SSIM.

The full CLI protocol is `--max-side 512 --runs 3 --seed 0`. GPU timing covers the complete model forward, including the feature encoder and all configured diffusion/flow steps. It excludes model loading, first-shape warmup, image I/O and transfers. The reported dataset time is the median of per-image medians over three runs. Inference is sequential, batch size one, and the manager holds one model at a time.

## Released model configurations

| System | Configuration used | Loaded parameters | Precision |
| --- | --- | ---: | --- |
| PGL-Net-T | Tiny, released RW2AH weights | 780,283 | FP32 |
| HNDiff-FocalNet | Released RW2AH restoration and diffusion weights; reverse steps `[4, 3, 2, 1]` | 14,420,392 | FP32 |
| PixRestore-S | Released S checkpoint; one linear-schedule step; DINOv2 ViT-S/14 with 448-pixel encoder input | 53,701,997 including encoder | CUDA BF16 autocast when supported; otherwise FP32 |

HNDiff's loaded parameter count differs from the paper's 7.82M configuration. The wrapper preserves all four reverse steps, but uses the common full-image protocol rather than the author's 384-pixel overlapping-patch evaluation wrapper. PixRestore's parameter and timing totals include DINOv2; its restoration network alone has 31,645,421 parameters and the encoder has 22,056,576.

HNDiff produced slightly different repeated uint8 outputs for 327 of the 352 images, with maximum absolute channel difference of one intensity level. PGL-Net-T and PixRestore-S repeats were byte-identical in the recorded run. A fixed seed therefore does not imply bit-exact repeatability for every GPU operation in this environment.

## Preparing a clone and reproducing the run

From the repository root on Windows with Python 3.12 installed:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe scripts/prepare_assets.py
.\.venv\Scripts\python.exe scripts/prepare_assets.py --verify-only
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --count 352 --output-dir data/rw2ah_test352 --manifest data/manifest_full.json --budget-mb 700 --workers 4
.\.venv\Scripts\python.exe -m dehaze.evaluation --manifest data/manifest_full.json --output results_full --max-side 512 --runs 3 --seed 0
.\.venv\Scripts\python.exe scripts/validate_results.py --manifest data/manifest_full.json --results results_full --expected-count 352
```

Asset preparation downloads the required model definitions and checkpoints from author-controlled sources, applies the documented HNDiff import-only adaptation, and checks the expected file sizes and SHA-256 values. It is separate from inference; the inference backend does not use remote model loaders. Network availability and the original authors' hosting limits can affect preparation. Review the upstream terms listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) before using downloaded assets.

Do not run UI inference concurrently with the timed evaluation. After evaluation has completed and saved its images, run `run.ps1` and use **Full test results** to inspect them. The repository contains the recorded numerical tables and eight processed saved example groups in `site/samples/`; the complete test-image and output collections are not bundled. The full-results image browser requires the locally regenerated evaluation outputs; the single-image interface accepts the user's own image after model preparation.

The dataset preparation script uses the original public archive and its test-member listing. It supports bounded downloads and reuses verified local files. Model assets total about 268 MiB; test imagery and evaluation outputs require additional disk space. Training data are not required for this experiment.

## Source identity and validation evidence

`third_party/SOURCE_MANIFEST.json` records the files used in the original local assignment. The SHA-256 entries identify those exact files, including the three HNDiff files after removal of unused `from thop import profile` imports. They are the final integrity criterion when fetching assets for this repository.

PGL-Net, HNDiff and PixRestore were originally retrieved as unpinned official snapshots. The full commit IDs listed in the historical audit describe the official trees inspected later; they must not be presented as proven original download commits. The download preparation metadata may use those commits as stable retrieval locations only when the resulting files match the recorded hashes. DINOv2 was taken from the official `7764ea0` snapshot. See the source manifest and download metadata for exact per-file records.

The original local validation checked 704 source-image hashes, 1,408 saved result-image pixel hashes, 20 independently recomputed metric records, aggregate consistency, full output coverage and source/checkpoint integrity. `results_full/validation.json` records that original validation. It does not assert that a fresh lightweight clone already contains the excluded images or weights. Re-run validation after generating those files locally. The original local assignment retains the complete PDF, PPTX and image-bearing submission package; the public repository provides Markdown reference material and numerical evidence.

## Interpretation limits and Physical AI connection

This is a comparison of released pretrained systems under a shared local input and metric protocol. PGL-Net and HNDiff use RW2AH-specific checkpoints; PixRestore uses a general restoration checkpoint. Training data, training compute, inference precision and author evaluation wrappers are not equalized. Resizing can change apparent haze and absolute scores. These results are neither a reproduction of the original paper tables nor a global SOTA ranking.

Dehazing can be studied as a perception preprocessing component for outdoor robots or autonomous driving, including a future Real2Sim2Real pipeline. This assignment measures image restoration only. It does not evaluate detection, depth, localization, planning, control, simulator reconstruction or real robot performance. Any claim of downstream benefit requires a separate controlled experiment with the relevant task and failure cases.
