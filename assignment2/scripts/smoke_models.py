"""Strict-load and native-resolution GPU smoke tests, with real held-out input."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from dehaze.models import load_model, final_output, mirror_pad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['SFMN', 'SFHformer', 'IPC-Dehaze'])
    ap.add_argument('--max-side', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = ROOT / 'experiments' / 'smoke'
    output.mkdir(parents=True, exist_ok=True)
    original = Image.open(ROOT.parent / 'data/rw2ah_test352/hazy/1-0010-02.png').convert('RGB')
    if args.max_side and max(original.size) > args.max_side:
        scale = args.max_side / max(original.size)
        original = original.resize(tuple(round(x * scale) for x in original.size), Image.Resampling.BICUBIC)
    rgb = np.array(original)
    for method in args.methods:
        started = time.perf_counter()
        try:
            net, record = load_model(method)
            x = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).cuda().float() / 255
            x, (h, w) = mirror_pad(x)
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                y = final_output(method, net(x))[..., :h, :w]
            torch.cuda.synchronize()
            record.update({'input_wh': list(original.size), 'padded_shape': list(x.shape),
                           'first_forward_s': time.perf_counter() - begin,
                           'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
                           'finite': bool(torch.isfinite(y).all()),
                           'range': [float(y.min()), float(y.max())]})
            result = (y[0].float().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8')
            Image.fromarray(result).save(output / (method + '.png'))
            record['success'] = record['finite'] and result.shape == rgb.shape
            del net, x, y
        except Exception as error:
            record = {'method': method, 'success': False, 'error': repr(error), 'traceback': traceback.format_exc()}
        record['elapsed_s'] = time.perf_counter() - started
        (output / (method + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
