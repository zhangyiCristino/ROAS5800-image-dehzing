"""Complete, resumable held-out evaluation with independently auditable outputs."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from PIL import Image
import PIL
import psutil
from skimage.metrics import structural_similarity
import skimage
import torch
from .models import load_model, final_output, mirror_pad, ROOT
from .recorded import verified_completed_summary

METHODS = ['SFMN', 'SFHformer', 'IPC-Dehaze']

PROTOCOL = {
    'dataset': 'All 352 paired RW2AH test images, original release pairing',
    'input': 'Original resolution, PIL RGB conversion, no resizing or geometric crop',
    'padding': 'All models: official IPC mirror concatenation to strictly next multiple of 32; remove all padding before metrics',
    'range': 'RGB input float32 uint8/255. No normalization, sharpening, color correction, or ground-truth-dependent tuning',
    'output': 'Clip to [0,1], multiply by 255, numpy.round, uint8 RGB PNG. Metrics use the saved PNG values',
    'psnr': '10 log10(255^2 / mean RGB squared error), full image, no border shave',
    'ssim': 'skimage RGB, channel_axis=-1, data_range=255, Gaussian weights, sigma=1.5, win_size=11, use_sample_covariance=False',
    'aggregation': 'Arithmetic mean of per-image PSNR and SSIM; each image has equal weight',
    'weights': 'Official released OTS weights for SFMN and SFHformer. Official real-world release for IPC-Dehaze. These are cross-dataset tests, not matched training-budget comparisons',
    'precision': 'All FP32, TF32 disabled, autocast disabled, batch 1, one loaded model at a time',
    'inference': 'SFMN both stages and output index 3; full SFHformer; IPC predictor and critic, all 8 official inference iterations, including VQGAN encoding and decoding',
    'randomness': 'Torch CPU/CUDA seeds reset to 0 before each forward. Save first output and compare quantized repeat outputs',
    'timing': 'Two unrecorded warmup forwards per model; CUDA event timing of full padded model inference, transfers/load/PNG/metrics excluded. Per-image median of 3 forwards, then dataset median. CPU wall forward time also retained',
    'memory': 'torch.cuda.max_memory_allocated during each forward, including loaded model/input/output tensors. Does not equal total driver/device use',
    'scope': 'Independent laptop generalization/deployment evaluation. Do not substitute these scores for paper-reported benchmark values or claim global SOTA',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()

def metrics(pred, ref):
    if pred.dtype != np.uint8 or ref.dtype != np.uint8 or pred.shape != ref.shape:
        raise ValueError('Metrics require identically shaped uint8 RGB arrays')
    error = pred.astype(np.float64) - ref.astype(np.float64)
    mse = float(np.mean(error**2))
    if mse == 0:
        raise ValueError('Unexpected perfect match: inspect data pairing and model output')
    return {'psnr': float(10 * np.log10(255**2 / mse)),
            'ssim': float(structural_similarity(ref, pred, channel_axis=-1, data_range=255,
                        gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False)),
            'mae_255': float(np.mean(np.abs(error)))}

def prepare_manifest():
    candidates = [(ROOT.parent / 'data/manifest_full.json', ROOT.parent),
                  (ROOT.parent / 'hw1/data/manifest_full.json', ROOT.parent / 'hw1'),
                  (ROOT / 'data/manifest_full.json', ROOT)]
    selected = next(((p, base) for p, base in candidates if p.is_file()), None)
    if selected is None:
        raise FileNotFoundError('Prepare the shared RW2AH test data from the repository root first')
    original, source_root = selected
    data = json.loads(original.read_text(encoding='utf-8'))
    pairs = []
    dimensions = Counter()
    for pair in data['pairs']:
        record = {'id': pair['id']}
        sizes = []
        for role in ('hazy', 'clear'):
            path = source_root / pair[role]
            actual = sha(path)
            expected = pair[role + '_source']['sha256']
            if actual != expected:
                raise ValueError('Source hash mismatch: ' + str(path))
            record[role] = str(path.resolve())
            record[role + '_sha256'] = actual
            with Image.open(path) as image:
                sizes.append(image.size)
        if sizes[0] != sizes[1]:
            raise ValueError('Unaligned dimensions: ' + pair['id'])
        record['size_wh'] = list(sizes[0])
        dimensions[str(sizes[0])] += 1
        pairs.append(record)
    if len(pairs) != 352 or len({p['id'] for p in pairs}) != 352:
        raise ValueError('RW2AH test must have 352 unique pairs')
    result = {'dataset': 'RW2AH', 'split': 'test', 'count': 352,
              'original_manifest_sha256': sha(original), 'source': data['sources'],
              'dimensions': dict(dimensions), 'verified_source_hashes': True, 'pairs': pairs}
    write_json(ROOT / 'data/manifest_rw2ah.json', result)
    return result

def summarize(rows):
    return {'count': len(rows), 'psnr_mean': statistics.mean(r['psnr'] for r in rows),
            'ssim_mean': statistics.mean(r['ssim'] for r in rows),
            'mae_255_mean': statistics.mean(r['mae_255'] for r in rows),
            'gpu_ms_median': statistics.median(r['gpu_ms_median'] for r in rows) if 'gpu_ms_median' in rows[0] else None,
            'gpu_ms_mean': statistics.mean(r['gpu_ms_median'] for r in rows) if 'gpu_ms_median' in rows[0] else None,
            'peak_MiB_max': max(r['peak_MiB'] for r in rows) if 'peak_MiB' in rows[0] else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--output', type=Path, default=ROOT / 'experiments/rw2ah_full')
    args = ap.parse_args()
    if args.runs < 1:
        raise ValueError('runs must be positive')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest()
    pairs = manifest['pairs']
    protocol = dict(PROTOCOL, timing=PROTOCOL['timing'].replace('3 forwards', str(args.runs) + ' forwards'))
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'hardware.json', {'platform': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'gpu_total_MiB': torch.cuda.get_device_properties(0).total_memory/2**20,
        'cpu': 'AMD Ryzen 9 7845HX with Radeon Graphics (Windows registry)',
        'ram_GiB': psutil.virtual_memory().total/2**30,
        'numpy': np.__version__, 'Pillow': PIL.__version__, 'skimage': skimage.__version__})
    baseline = []
    for p in pairs:
        baseline.append({'id': p['id'], **metrics(rgb(p['hazy']), rgb(p['clear']))})
    write_json(out / 'Hazy_input.json', {'rows': baseline, 'summary': summarize(baseline)})
    summaries = {'Hazy input': summarize(baseline)}
    print(json.dumps({'manifest_verified': True, 'count': len(pairs), 'dimensions': manifest['dimensions'],
                      'baseline': summaries['Hazy input']}), flush=True)
    for method in args.methods:
        method_dir = out / method
        method_dir.mkdir(exist_ok=True)
        completed = method_dir / 'results.json'
        saved_summary = verified_completed_summary(method_dir)
        if saved_summary is not None:
            summaries[method] = saved_summary
            print(json.dumps({'reused_verified_complete': method}), flush=True)
            continue
        net, model_record = load_model(method)
        write_json(method_dir / 'model.json', model_record)
        first = torch.from_numpy(rgb(pairs[0]['hazy'])).permute(2,0,1).unsqueeze(0).cuda().float()/255
        first, _ = mirror_pad(first)
        with torch.inference_mode():
            for _ in range(2):
                torch.manual_seed(0)
                warm = final_output(method, net(first))
        del first, warm
        torch.cuda.synchronize()
        rows = []
        begin_model = time.perf_counter()
        for index, pair in enumerate(pairs):
            h_arr, reference = rgb(pair['hazy']), rgb(pair['clear'])
            x = torch.from_numpy(h_arr).permute(2,0,1).unsqueeze(0).cuda().float()/255
            x, (h, w) = mirror_pad(x)
            times, wall_times, peaks, quant_hashes = [], [], [], []
            saved_output = None
            for repeat in range(args.runs):
                torch.manual_seed(0)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                with torch.inference_mode():
                    y = final_output(method, net(x))
                stop.record()
                stop.synchronize()
                wall_times.append((time.perf_counter() - wall)*1000)
                times.append(start.elapsed_time(stop))
                peaks.append(torch.cuda.max_memory_allocated()/2**20)
                y = y[..., :h, :w]
                if not torch.isfinite(y).all():
                    raise ValueError(f'Nonfinite output: {method} {pair["id"]}')
                q = (y[0].float().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8')
                quant_hashes.append(hashlib.sha256(q.tobytes()).hexdigest())
                if repeat == 0:
                    saved_output = q
                del y
            output_file = method_dir / (pair['id'] + '.png')
            Image.fromarray(saved_output).save(output_file)
            row = {'id': pair['id'], 'width': w, 'height': h, **metrics(saved_output, reference),
                   'gpu_ms': times, 'gpu_ms_median': statistics.median(times),
                   'wall_ms': wall_times, 'peak_MiB': max(peaks),
                   'repeat_uint8_equal': len(set(quant_hashes)) == 1,
                   'output_sha256': sha(output_file)}
            rows.append(row)
            del x
            if index % 20 == 0 or index == len(pairs) - 1:
                status = {'method': method, 'done': index + 1, 'total': 352,
                          'elapsed_s': time.perf_counter() - begin_model,
                          'current_psnr_mean': statistics.mean(r['psnr'] for r in rows)}
                write_json(out / 'status.json', status)
                print(json.dumps(status), flush=True)
                write_json(method_dir / 'partial.json', {'model': model_record, 'rows': rows, 'complete': False})
        summary = summarize(rows)
        summary['elapsed_s_including_IO_metrics'] = time.perf_counter() - begin_model
        summary['repeat_equal_count'] = sum(r['repeat_uint8_equal'] for r in rows)
        record = {'method': method, 'model': model_record, 'summary': summary, 'rows': rows,
                  'manifest_sha256': sha(ROOT / 'data/manifest_rw2ah.json'),
                  'complete': len(rows) == 352, 'finished_utc': datetime.now(timezone.utc).isoformat()}
        write_json(completed, record)
        with (method_dir / 'per_image.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id','width','height','psnr','ssim','gpu_ms_median','peak_MiB','repeat_uint8_equal','output_sha256'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        summaries[method] = summary
        print(json.dumps({'method_complete': method, 'summary': summary}), flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        write_json(out / 'summary.json', summaries)
    complete = all(method in summaries and summaries[method]['count'] == 352 for method in METHODS)
    write_json(out / 'status.json', {'complete': complete, 'methods': list(summaries), 'pairs': 352})
    write_json(out / 'summary.json', summaries)

if __name__ == '__main__':
    main()
