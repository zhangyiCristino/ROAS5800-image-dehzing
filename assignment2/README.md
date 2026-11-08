# Assignment 2: image dehazing with spatial, frequency and codebook priors

This project surveys **SFMN (TIP 2025), SFHformer (ECCV 2024) and IPC-Dehaze
(CVPR 2025)** and evaluates the official pretrained releases on the same 352
held-out RW2AH pairs at their original 640×512 resolution.

**[Assignment 2 website](https://zhangyicristino.github.io/ROAS5800-image-dehzing/assignment2/)** ·
**[Assignment 1 website](https://zhangyicristino.github.io/ROAS5800-image-dehzing/assignment1.html)** ·
**[Project portal](https://zhangyicristino.github.io/ROAS5800-image-dehzing/)**

Assignment 1 retains PGL-Net-T, HNDiff-FocalNet and PixRestore-S, its own UI,
code and maximum-side-512 evaluation. These are separate experiments. Neither
assignment table establishes a controlled six-model ranking.

## Deliverables

- [Two-page report](report/50063117-YiZhang-HW2.pdf).
- [PowerPoint with an English five-minute script and Chinese explanatory notes](presentation/50063117-YiZhang-HW2.pptx).
- [Chinese beginner guide](docs/beginner-guide.md).
- [Verified full-test summary](experiments/rw2ah_full/summary.csv) and per-model
  `results.json` / `per_image.csv` records.
- [Exact protocol](experiments/rw2ah_full/protocol.json),
  [hardware](experiments/rw2ah_full/hardware.json) and
  [independent output audit](experiments/rw2ah_full/independent_validation.json).

The public website provides eight actual archived comparisons, method
selection, before/after dividers, measured scores and downloadable ZIPs. New
uploads there are input previews. New-image GPU processing is available in
the local application below.

## Run the local GPU interface

On the prepared computer, double-click **`启动Assignment2.bat`** in the repository
root, or run:

```powershell
.\run_assignment2.ps1
```

The application opens at **http://127.0.0.1:7861**. Assignment 1 continues to use
port 7860 and its original launcher.

For a fresh clone, install Python 3.12, then prepare Assignment 2 once:

```powershell
cd assignment2
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\run.ps1
```

The setup script creates an Assignment 2 environment, installs the tested CUDA
12.8 PyTorch build and dependencies, and downloads source/weights from the
original authors. Exact source commits and SHA-256 values are retained in
`sources/*.json` and `assets.json`. A changed download fails verification
rather than being silently substituted.

Upload a hazy RGB image, optionally add an aligned same-size clean reference,
select methods and click **Compare methods**. All selected networks execute
locally. Outputs, per-image metrics and a comparison ZIP appear after inference.
PSNR/SSIM are shown only with a reference. Each new comparison uses one recorded
GPU forward after warmup; the full benchmark uses three per image.

The **Full test results** tab shows the published numerical reference. Its full
352-image browser becomes available when all local source and output images
exist. Click **Refresh full-test results** after completing a local evaluation.
New-image inference is blocked while a timed benchmark process is alive.

## Reproduce the complete test

First prepare the shared original test pairs from the repository root, using
the existing author-linked downloader:

```powershell
# From the repository root, using the Assignment 2 environment.
.\assignment2\.venv\Scripts\python.exe scripts/prepare_dataset.py --count 352 --output-dir data/rw2ah_test352 --manifest data/manifest_full.json --budget-mb 700 --workers 4
cd assignment2
.\.venv\Scripts\python.exe -m dehaze.evaluate
.\.venv\Scripts\python.exe scripts/validate_and_prepare_results.py
```

Inputs remain at 640×512. All methods use FP32 with TF32 disabled and identical
mirror padding removed before metrics. SFMN runs both stages. IPC-Dehaze uses
both official checkpoints and all eight inference iterations, including its
encoder, codebook, predictor/critic and decoder. The strictly loaded SFHformer
release has 4.75M parameters, distinct from the 3.87M paper-table configuration.

All 1,056 restored PNGs are retained locally. The validator checks every output
hash and dimension, source and weight hashes, repeats and complete iteration
count, then independently recalculates per-image and aggregate quality.
It does not mark a partial run complete. A fresh clone contains numerical
reference records without the full output PNGs, so the default evaluation
computes those images again. When all 352 output files for a method are present
and hash-verified, it can reuse that completed method. Use
`--output experiments/rw2ah_repeat` to force a separate fresh evaluation.

Model loading, transfers, disk I/O and metrics are excluded from GPU forward
times. Memory is the allocated-tensor peak, not total driver/device occupancy.
The complete FP32 iterative test can take substantially longer than a single
interactive comparison.

## Interpreting the survey

The selected papers report advanced performance or quality/efficiency
tradeoffs within different publication-time protocols. SFMN and SFHformer
report SOTS reference metrics; IPC-Dehaze reports perceptual metrics on real
haze datasets. Those numbers are kept separate from local measurements.

The first two releases use OTS weights, while IPC-Dehaze uses its released
real-world model. The local experiment tests deployment and cross-dataset
generalization, not equal-budget training or a universal SOTA ranking.
No training-from-scratch or downstream detector result is claimed.

[Sources, figure/table anchors and terminology](docs/SOURCES.md) document the
underlying papers. On 8 November 2026, the student confirmed that none of the
three selected papers was introduced in the lectures; see the
[eligibility record](docs/eligibility.json). All three also differ from the
Assignment 1 trio.

## Validation and third-party materials

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The UI tests cover reference alignment, native/limited image sizes, quality
without a reference, perfect matches and output invalidation. Actual GPU
evaluation and browser/API checks supplement these tests.

Official downloaded source code, weights and full dataset/output images are
excluded from Git; provenance and numerical records are retained. Original
licenses and source notices stay in downloaded directories. Paper figures in
the presentation/guide retain source labels and scientific legends.
