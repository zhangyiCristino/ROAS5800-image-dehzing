# Assignment 3 · Physics-Guided Fourier Side-Refiner

[Interactive demo](https://zhangyiCristino.github.io/ROAS5800-image-dehzing/assignment3/) · [Recorded benchmark](results/summary.json) · [Track 2 image-level measurements](results/track2_per_image.csv)

The site compares five actual GPU outputs with hazy inputs, the frozen PGL-Net-T outputs and clean references. The site uses compressed display copies. Metrics are computed from the native, quantized RGB outputs. Run the model locally to process new images or reproduce the 352-image benchmark.

## Quick start

Use Python 3.12. Commands below start in the cloned repository root. Prefer the tested CUDA environment when rerunning the full benchmark.

```sh
git clone https://github.com/zhangyiCristino/ROAS5800-image-dehzing.git
cd ROAS5800-image-dehzing/assignment3
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS. For the tested NVIDIA CUDA 12.8 build, install the CUDA wheels first. Use PyTorch's regular installer for CPU or other supported configurations.

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Download [the original PGL-Net-T RW²AH checkpoint](https://drive.google.com/file/d/1Sl7cCSK3peAeOzEXWvv2_3m9cJFvrvJx/view) as `rw2ah_pglnet_t.pk`. The setup script downloads a pinned upstream model definition, verifies both official assets by SHA-256 and stages the five demo pairs already present in this coursework repo. The project-trained Track 2 refiner checkpoint is included.

```sh
python prepare.py --checkpoint /path/to/rw2ah_pglnet_t.pk
python run_ta_test.py --mode demo --model side
```

The demo writes five PNGs to `ta_demo_out/` and prints five per-image scores. On the five showcase scenes, the expected mean is **25.8770 dB / 0.7754 SSIM**. The default picks CUDA when available. Add `--device cpu` to use CPU explicitly.

To try new images, put RGB-convertible PNGs in `my_images/hazy/`. For scoring, add same-size aligned references with the same filenames under `my_images/gt/`.

```sh
python run_ta_test.py --mode demo --model side --input_dir my_images --output_dir outputs
```

## Reproduce the 352-image test

Use the [official RW²AH archive](https://drive.google.com/file/d/1RmqMw-105j0Berybl8EAdRwVGlnfG3mT/view). Put the native test images in `hazy/` and `clear/` inside `../data/rw2ah_test352`, or use the repository's test-only fetcher from `assignment3/` while the Python environment is active.

```sh
python ../scripts/prepare_dataset.py --count 352 --output-dir ../data/rw2ah_test352 --manifest data/test_sources.json --budget-mb 700 --workers 4
python prepare.py
python run_ta_test.py --mode test --model side
python run_ta_test.py --mode test --model base
```

For a test directory elsewhere, run `python prepare.py --test-data /path/to/rw2ah_test352`. Test mode requires all **352** pairs and prints mean RGB PSNR and SSIM. With the provided refiner and verified upstream checkpoint, the RTX 4060 Laptop run produced **22.5885 dB / 0.6467 SSIM**. The frozen base baseline in the archived benchmark was **22.5934 / 0.6465**.

## Retrain the Track 2 refiner

Place native training images under `data/rw2ah_train/hazy/` and train references under `data/rw2ah_train/gt/`, or pass `--train-data` to `prepare.py`. The script reconstructs the exact recorded split order from `split_ids.json`. The base remains frozen. The following commands cache frozen predictions and features, run the 60-epoch refiner training schedule, and save the best validation-PSNR weights to a new directory.

```sh
python prepare.py
python tools/cache_side_data.py --manifest data/manifests/train_1304.jsonl --cache cache/side1406
python tools/cache_side_data.py --manifest data/manifests/disjoint/val.jsonl --cache cache/side_val
python tools/train_side_refiner.py --track 2 --train-cache cache/side1406 --train-manifest data/manifests/train_1304.jsonl --epochs 60 --crop 256 --batch 4 --lr 3e-4 --no-test --out outputs/retrained_track2
python run_ta_test.py --mode test --model side --ckpt outputs/retrained_track2/best_val_ema.pth
```

Training was performed on a CUDA GPU using bfloat16 and EMA. The reported checkpoint was selected at epoch 5. The frozen base and optional test-time averaging provide deployment references. Geo8 TTA results in the benchmark record belong to the frozen base.

## Evaluation protocol

- Among the 1,304 Track 2 training pairs, 399 share a byte-identical clear reference with the test set. The frozen backbone uses the released RW²AH checkpoint.
- The historical Track 1 split has 905 training pairs with zero train/test clear-reference SHA-256 overlap. The historical refiner had 55,004 parameters and used the same released frozen backbone. The public inference entry and weights target the 84,465-parameter Track 2 model.
- Track 2 changed mean PSNR by −0.0049 dB and mean SSIM by +0.0002 against the frozen base. The architecture experiment does not establish a material aggregate quality improvement.

[Original PGL-Net implementation](https://github.com/sc-30-bit/PGL-Net) · [RW²AH dataset authors](https://github.com/fiwy0527/AAAI25_SGDN) · [Third-party provenance](../THIRD_PARTY_NOTICES.md)
