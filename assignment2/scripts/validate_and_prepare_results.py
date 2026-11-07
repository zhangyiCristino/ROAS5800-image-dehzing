"""Verify every saved output and independently recalculate full-test quality."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/rw2ah_full'
METHODS = ['SFMN', 'SFHformer', 'IPC-Dehaze']

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def array(p):
    with Image.open(p) as image:
        if image.size != (640,512):
            raise AssertionError('Unexpected dimensions: '+str(p))
        return np.asarray(image.convert('RGB')).astype(np.float64) / 255

def main():
    status=json.loads((EXP/'status.json').read_text(encoding='utf-8'))
    if not status.get('complete'):
        raise RuntimeError('Full experiment has not completed')
    manifest=json.loads((ROOT/'data/manifest_rw2ah.json').read_text(encoding='utf-8'))
    summary=json.loads((EXP/'summary.json').read_text(encoding='utf-8'))
    provenance=ROOT/'docs/implementation-provenance.json'
    if provenance.is_file():
        implementation=json.loads(provenance.read_text(encoding='utf-8'))['public_code_sha256']
        implementation_scope='documented public data/resume adaptation'
    else:
        implementation=json.loads((EXP/'implementation_hashes.json').read_text(encoding='utf-8'))
        implementation_scope='original evaluated implementation'
    for file,expected_hash in implementation.items():
        assert digest(ROOT/file)==expected_hash,'Implementation hash mismatch: '+file
    pairs=manifest['pairs']
    for pair in pairs:
        for role in ('hazy','clear'):
            candidate=Path(pair[role])
            if not candidate.is_absolute():pair[role]=str((ROOT/candidate).resolve())
    expected={p['id'] for p in pairs}
    assert len(pairs)==352 and len(expected)==352
    raw={m:json.loads((EXP/m/'results.json').read_text(encoding='utf-8')) for m in METHODS}
    baseline=json.loads((EXP/'Hazy_input.json').read_text(encoding='utf-8'))
    rows={m:{r['id']:r for r in raw[m]['rows']} for m in METHODS}
    base={r['id']:r for r in baseline['rows']}
    recalculated={m:[] for m in ['Hazy input']+METHODS}
    counts={m:0 for m in METHODS}
    for m in METHODS:
        assert raw[m]['complete'] and set(rows[m])==expected
        assert raw[m]['model']['strict_load']
        for weight in raw[m]['model']['weights']:
            assert digest(ROOT/weight['file'])==weight['sha256']
        assert raw[m]['model']['iterations']==(8 if m=='IPC-Dehaze' else 1)
    for index,p in enumerate(pairs):
        assert digest(p['hazy'])==p['hazy_sha256']
        assert digest(p['clear'])==p['clear_sha256']
        ref=array(p['clear'])
        for method in ['Hazy input']+METHODS:
            path=Path(p['hazy']) if method=='Hazy input' else EXP/method/(p['id']+'.png')
            pred=array(path)
            if method!='Hazy input':
                row=rows[method][p['id']]
                assert digest(path)==row['output_sha256']
                assert len(row['gpu_ms'])==3 and all(t>0 and math.isfinite(t) for t in row['gpu_ms'])
                assert statistics.median(row['gpu_ms'])==row['gpu_ms_median']
                assert row['repeat_uint8_equal']
                assert len(row['wall_ms'])==3 and all(t>0 and math.isfinite(t) for t in row['wall_ms'])
                assert 0<row['peak_MiB']<8187
            else:
                row=base[p['id']]
            # Independent normalized-array PSNR expression and SSIM scale.
            mse=float(np.sum((ref-pred)**2)/ref.size)
            psnr=-10*math.log10(mse)
            ssim=float(structural_similarity(ref,pred,data_range=1,channel_axis=2,
                       gaussian_weights=True,sigma=1.5,win_size=11,use_sample_covariance=False))
            assert abs(psnr-row['psnr'])<1e-9
            assert abs(ssim-row['ssim'])<1e-9
            recalculated[method].append((psnr,ssim))
            if method!='Hazy input' and psnr>base[p['id']]['psnr']:
                counts[method]+=1
        if index%80==0:
            print(json.dumps({'validated_pairs':index+1,'total':352}),flush=True)
    for method,values in recalculated.items():
        assert abs(statistics.mean(x[0] for x in values)-summary[method]['psnr_mean'])<1e-9
        assert abs(statistics.mean(x[1] for x in values)-summary[method]['ssim_mean'])<1e-9
        if method!='Hazy input':
            timings=[r['gpu_ms_median'] for r in rows[method].values()]
            assert statistics.median(timings)==summary[method]['gpu_ms_median']
            assert statistics.mean(timings)==summary[method]['gpu_ms_mean']
            assert max(r['peak_MiB'] for r in rows[method].values())==summary[method]['peak_MiB_max']
            assert summary[method]['repeat_equal_count']==352
    ordered=sorted(pairs,key=lambda p:base[p['id']]['psnr'])
    chosen=[ordered[len(ordered)//4],ordered[3*len(ordered)//4]]
    assets=ROOT/'assets'
    (assets/'local').mkdir(parents=True,exist_ok=True)
    cases=[]
    for p in chosen:
        files=[]
        labels=['Hazy input']+METHODS+['Clear reference']
        origins=[Path(p['hazy'])]+[EXP/m/(p['id']+'.png') for m in METHODS]+[Path(p['clear'])]
        for label,origin in zip(labels,origins):
            dest=assets/'local'/(p['id']+'-'+label.replace(' ','_')+'.png')
            with Image.open(origin) as im:
                im.convert('RGB').save(dest)
            files.append(str(dest.relative_to(ROOT)))
        cases.append({'id':p['id'],'images':files,
                      'scores':{m:base[p['id']]['psnr'] if m=='Hazy input' else rows[m][p['id']]['psnr'] for m in labels[:-1]}})
    cover=pairs[100]
    coverfiles={}
    for role in ('hazy','clear'):
        dest=assets/'local'/('cover-'+role+'.png')
        with Image.open(cover[role]) as im: im.convert('RGB').save(dest)
        coverfiles[role]=str(dest.relative_to(ROOT))
    examples={'selection':'25th and 75th percentile ranks of baseline PSNR, independent of model rankings',
              'cases':cases,'cover':coverfiles}
    (assets/'local_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    validation={'passed':True,'count_pairs':352,'restored_outputs_verified':352*3,
                'quality_recomputed_for_all_inputs_and_outputs':True,'source_and_output_hashes_verified':True,
                'official_weight_hashes_verified':True,'strict_weight_loading':True,
                'implementation_hashes_verified':True,'implementation_hashes_scope':implementation_scope,
                'timing_and_memory_aggregates_verified':True,
                'full_ipc_iterations':8,'three_equal_quantized_repeats_per_image':True,
                'psnr_improved_image_counts':counts,'example_ids':[p['id'] for p in chosen]}
    (EXP/'independent_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    with (EXP/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['method','count','PSNR_dB','SSIM','GPU_median_ms','peak_allocated_MiB'])
        for m,s in summary.items(): writer.writerow([m,s['count'],s['psnr_mean'],s['ssim_mean'],s['gpu_ms_median'],s['peak_MiB_max']])
    print(json.dumps(validation),flush=True)

if __name__=='__main__': main()
