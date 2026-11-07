"""Assignment 2 local GPU UI; Assignment 1 remains a separate application."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import threading
import time
import zipfile
from uuid import uuid4

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
for key in ('NO_PROXY','no_proxy'):
    hosts=[x.strip() for x in os.environ.get(key,'').split(',') if x.strip()]
    os.environ[key]=','.join(dict.fromkeys(hosts+['127.0.0.1','localhost','::1']))

import gradio as gr
import numpy as np
from PIL import Image
import psutil
import torch
from dehaze.models import ROOT,load_model,final_output,mirror_pad
from dehaze.evaluate import metrics

METHODS=['SFMN','SFHformer','IPC-Dehaze']
EXP=ROOT/'experiments/rw2ah_full'
OUTPUT=ROOT/'outputs'
GPU_LOCK=threading.Lock()
GENERATION=defaultdict(int)
GEN_LOCK=threading.Lock()
HEADERS=['Method','Image size','GPU forward (ms)','Peak (MiB)','PSNR (dB)','SSIM']
SAMPLE_IDS=['1-0010-02','12-0002-05','2-0019-03','4-0005-07',
            '8-0016-01','9-0002-03','10-0001-04','11-0001-02']

CSS='''
.gradio-container {width:100%!important;min-width:0!important;max-width:1450px!important;margin:auto!important;font-family:Arial,'Microsoft YaHei',sans-serif!important}
.gradio-container .main,.gradio-container main,.gradio-container .column,.gradio-container .row {min-width:0!important;max-width:100%}
.gradio-container .table-wrap {max-width:100%;overflow-x:auto}
#intro h1 {font-size:32px;letter-spacing:-.5px;margin-bottom:7px}
#intro p {font-size:16px;color:#526474}
#compare {min-height:48px;font-size:17px}
@media(max-width:700px){.gradio-container .row{flex-wrap:wrap!important}.gradio-container .row>.block,.gradio-container .row>.column{flex-basis:100%!important;min-width:0!important}#intro h1{font-size:25px}#sample-gallery{max-width:100%}}
footer {display:none!important}
'''

def session_key(request):
    return getattr(request,'session_hash',None) or 'direct-test'

def invalidate(request:gr.Request):
    with GEN_LOCK: GENERATION[session_key(request)]+=1
    return (None,None,None,[], 'Input changed. Select methods and run a new comparison.',gr.update(value=None,visible=False))

def benchmark_running():
    """Check live processes, not a stale lock/status file."""
    for proc in psutil.process_iter(['cmdline','cwd']):
        try:
            cmd=proc.info.get('cmdline') or []
            if any(a in ('dehaze.evaluate','dehaze.evaluation') for a in cmd) and proc.is_running():
                return True
        except (psutil.Error,OSError): pass
    return False

def prepare_pair(hazy,reference,max_side):
    if hazy is None: raise ValueError('Upload a hazy image first.')
    hazy=np.asarray(hazy)
    if hazy.ndim!=3 or hazy.shape[2] not in (3,4): raise ValueError('Use an RGB image.')
    if hazy.dtype!=np.uint8: raise ValueError('Use an 8-bit PNG, JPG or WebP image.')
    hazy=hazy[:,:,:3].copy()
    if min(hazy.shape[:2])<32: raise ValueError('Use an image at least 32 pixels on each side.')
    if hazy.shape[0]*hazy.shape[1]>40_000_000: raise ValueError('Use an image below 40 megapixels.')
    if reference is not None:
        reference=np.asarray(reference)[:,:,:3].copy()
        if reference.dtype!=np.uint8 or reference.shape!=hazy.shape:
            raise ValueError('The clean reference must match the original hazy image dimensions.')
    cap=0 if max_side=='Original size' else int(max_side.split()[0])
    if cap==0 and max(hazy.shape[:2])>1536:
        raise ValueError('For images above 1536 pixels, choose a 512 px or 256 px maximum side.')
    h,w=hazy.shape[:2]
    scale=min(1,cap/max(h,w)) if cap else 1
    size=(max(32,round(w*scale)),max(32,round(h*scale)))
    # Extremely narrow images are rejected rather than silently distorted.
    if scale<1 and min(round(w*scale),round(h*scale))<32:
        raise ValueError('The selected size would make an image side smaller than 32 pixels.')
    if size!=(w,h):
        hazy=np.array(Image.fromarray(hazy).resize(size,Image.Resampling.BICUBIC))
        if reference is not None: reference=np.array(Image.fromarray(reference).resize(size,Image.Resampling.BICUBIC))
    return hazy,reference

def number(value,digits=2):
    return '—' if value is None else ('∞' if np.isposinf(value) else f'{value:.{digits}f}')

def quality(prediction,reference):
    if reference is None: return {'psnr':None,'ssim':None}
    if np.array_equal(prediction,reference):
        return {'psnr':float('inf'),'ssim':1.0,'mae_255':0.0,'perfect_match':True}
    return metrics(prediction,reference)

def verify_model_assets(method):
    manifest=ROOT/'assets.json'
    if not manifest.is_file():
        raise ValueError('Model assets are not prepared. Run Assignment 2 setup first.')
    expected=json.loads(manifest.read_text(encoding='utf-8'))['models'][method]
    for item in expected['weights']:
        file=(ROOT/item['file'].replace('\\','/')).resolve()
        if not file.is_relative_to(ROOT.resolve()) or not file.is_file():
            raise ValueError('A required model file is missing. Run Assignment 2 setup first.')
        if hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('A model file failed verification. Restore the official assets before running.')

def compare(hazy,reference,selected,max_side,request:gr.Request):
    if not selected: raise gr.Error('Select at least one method.')
    if any(m not in METHODS for m in selected): raise gr.Error('Unknown method selection.')
    if benchmark_running():
        raise gr.Error('The full timed evaluation is running. Browse saved results now, or run new inference after it finishes.')
    try: hazy,reference=prepare_pair(hazy,reference,max_side)
    except ValueError as error: raise gr.Error(str(error))
    if not torch.cuda.is_available(): raise gr.Error('CUDA is unavailable. This application is configured for the tested NVIDIA GPU.')
    key=session_key(request)
    with GEN_LOCK: generation=GENERATION[key]
    images={};rows=[];records=[]
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+hashlib.sha256(hazy.tobytes()).hexdigest()[:8]+'-'+uuid4().hex[:6]
    dest=OUTPUT/run_id
    dest.mkdir(parents=True,exist_ok=True)
    Image.fromarray(hazy).save(dest/'hazy-input.png')
    if reference is not None: Image.fromarray(reference).save(dest/'clean-reference.png')
    h,w=hazy.shape[:2]
    try:
        with GPU_LOCK:
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            for method in METHODS:
                if method not in selected: continue
                with GEN_LOCK:
                    if generation!=GENERATION[key]: return (gr.skip(),)*6
                verify_model_assets(method)
                net,model_record=load_model(method)
                x=torch.from_numpy(hazy.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/255
                x,_=mirror_pad(x)
                with torch.inference_mode():
                    torch.manual_seed(0)
                    warm=final_output(method,net(x))
                del warm
                torch.manual_seed(0)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode(): y=final_output(method,net(x))
                stop.record();stop.synchronize()
                gpu_ms=start.elapsed_time(stop)
                peak=torch.cuda.max_memory_allocated()/2**20
                y=y[...,:h,:w]
                if not torch.isfinite(y).all(): raise ValueError('Nonfinite model output')
                result=(y[0].float().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8')
                metric=quality(result,reference)
                file=dest/(method+'.png');Image.fromarray(result).save(file)
                record={'method':method,'width':w,'height':h,'gpu_ms':gpu_ms,'peak_MiB':peak,
                        **metric,'model':model_record,'output_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
                records.append(record);images[method]=(hazy,result)
                rows.append([method,f'{w} × {h}',number(gpu_ms),number(peak,1),number(metric['psnr'],3),number(metric['ssim'],4)])
                if record.get('perfect_match'): record['psnr']=None
                del net,x,y
                gc.collect();torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        gc.collect();torch.cuda.empty_cache()
        raise gr.Error('This image does not fit the available GPU memory. Choose a smaller maximum side.')
    except Exception as error:
        gc.collect();torch.cuda.empty_cache()
        raise gr.Error(f'Comparison failed: {type(error).__name__}: {error}')
    settings={'assignment':2,'run_utc':datetime.now(timezone.utc).isoformat(),'input_size':[w,h],
              'max_side':max_side,'seed':0,'precision':'FP32, TF32 off','timing_runs':1,
              'timing':'One complete GPU forward after one warmup; excludes loading, transfers and files',
              'reference_present':reference is not None,'methods':records}
    (dest/'run.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    with (dest/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(HEADERS);writer.writerows(rows)
    archive=dest/'comparison.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in dest.iterdir():
            if f!=archive and f.is_file(): z.write(f,f.name)
    with GEN_LOCK:
        if generation!=GENERATION[key]: return (gr.skip(),)*6
    status=f'Completed {len(selected)} method(s) at {w} × {h}. GPU forward time excludes loading and image I/O.'
    if reference is None: status+=' Add an aligned reference to measure PSNR and SSIM.'
    return (*[images.get(m) for m in METHODS],rows,status,gr.update(value=str(archive),visible=True))

def manifest_pairs():
    path=ROOT/'data/manifest_rw2ah.json'
    if not path.exists(): return []
    data=json.loads(path.read_text(encoding='utf-8'))
    pairs=data.get('pairs',[])
    result=[]
    for pair in pairs:
        record=dict(pair)
        for role in ('hazy','clear'):
            path=Path(record[role])
            if not path.is_absolute():record[role]=str((ROOT/path).resolve())
        result.append(record)
    return result

def saved_status():
    try:
        status=json.loads((EXP/'status.json').read_text(encoding='utf-8'))
        if not status.get('complete'):
            return [],[],f'Full evaluation in progress: {status.get("method","preparing")}, {status.get("done",0)}/352 images.'
        summaries=json.loads((EXP/'summary.json').read_text(encoding='utf-8'))
        rows=[]
        for m in ['Hazy input']+METHODS:
            s=summaries[m]
            rows.append([m,s['count'],number(s['psnr_mean'],3),number(s['ssim_mean'],4),
                         number(s['gpu_ms_median']),number(s['peak_MiB_max'],1)])
        pairs=manifest_pairs()
        available=[]
        for pair in pairs:
            if all((EXP/m/(pair['id']+'.png')).is_file() for m in METHODS) and Path(pair['hazy']).is_file() and Path(pair['clear']).is_file():
                available.append(pair['id'])
        if len(available)==352:
            message='Complete local RW2AH test: 352 pairs, original 640 × 512, three timing forwards per image.'
        else:
            message='Published reference measurements. Complete local image files are not available yet; run the documented preparation/evaluation workflow.'
        return rows,available,message
    except (OSError,ValueError,KeyError):
        return [],[],'No complete local benchmark is available yet.'

def refresh_saved():
    rows,choices,message=saved_status()
    return rows,gr.update(choices=choices,value=choices[0] if choices else None),message

def saved_image(image_id):
    if not image_id: return (None,None,None,[],None)
    mapping={p['id']:p for p in manifest_pairs()}
    if image_id not in mapping: raise gr.Error('Choose a valid recorded test image.')
    pair=mapping[image_id]
    rows=[];sliders=[]
    for method in METHODS:
        data=json.loads((EXP/method/'results.json').read_text(encoding='utf-8'))
        row=next(r for r in data['rows'] if r['id']==image_id)
        file=EXP/method/(image_id+'.png')
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=row['output_sha256']:
            raise gr.Error('A recorded output is missing or does not match its hash.')
        sliders.append((pair['hazy'],str(file)))
        rows.append([method,'640 × 512',number(row['gpu_ms_median']),number(row['peak_MiB'],1),
                     number(row['psnr'],3),number(row['ssim'],4)])
    return (*sliders,rows,pair['clear'])

def create_ui():
    with gr.Blocks(title='ROAS5800 · Assignment 2') as demo:
        gr.Markdown('# Assignment 2 · Image Dehazing Comparison\n'
            'SFMN, SFHformer and IPC-Dehaze. Process new images on this computer or browse the complete recorded test.',elem_id='intro')
        gr.Markdown('[Assignment 1 website](https://zhangyicristino.github.io/ROAS5800-image-dehzing/assignment1.html) '
                    ' · [Assignment 2 website](https://zhangyicristino.github.io/ROAS5800-image-dehzing/assignment2/)')
        with gr.Tab('Compare an image'):
            with gr.Row():
                with gr.Column(scale=2):
                    hazy=gr.Image(label='Hazy input',type='numpy',image_mode='RGB',format='png',height=300)
                    with gr.Accordion('Clean reference (optional)',open=False):
                        reference=gr.Image(label='Aligned clean image, same original dimensions',type='numpy',image_mode='RGB',format='png',height=230)
                with gr.Column(scale=1):
                    methods=gr.CheckboxGroup(METHODS,value=METHODS,label='Methods')
                    cap=gr.Dropdown(['256 px','512 px','Original size'],value='512 px',label='Maximum image side')
                    gr.Markdown('Smaller images run faster. IPC-Dehaze retains all eight predictor-critic rounds. '
                                'Images and results stay on this computer.')
                    run=gr.Button('Compare methods',variant='primary',elem_id='compare')
                    clear=gr.Button('Reset')
            pairs={p['id']:p for p in manifest_pairs()}
            examples=[pairs[i] for i in SAMPLE_IDS if i in pairs and Path(pairs[i]['hazy']).exists()]
            if examples:
                gallery=gr.Gallery(value=[(p['hazy'],p['id']) for p in examples],label='RW2AH test inputs: click to load, then run a comparison',
                                   columns=8,rows=1,height=140,object_fit='cover',allow_preview=False,elem_id='sample-gallery')
                def select_example(event:gr.SelectData):
                    index=event.index[0] if isinstance(event.index,(list,tuple)) else event.index
                    pair=examples[int(index)]
                    return pair['hazy'],pair['clear']
                gallery.select(select_example,outputs=[hazy,reference],queue=False,api_name=False)
            status=gr.Markdown('Upload an image and click Compare methods. New model results appear after inference.')
            with gr.Row():
                sliders=[gr.ImageSlider(label=m,height=310,format='png') for m in METHODS]
            measured=gr.Dataframe(headers=HEADERS,interactive=False,label='This image: one GPU forward after warmup',wrap=True)
            download=gr.DownloadButton(label='Download this comparison ZIP',visible=False,elem_id='comparison-download')
            outputs=[*sliders,measured,status,download]
            event=run.click(compare,inputs=[hazy,reference,methods,cap],outputs=outputs,concurrency_id='gpu',concurrency_limit=1,api_name='compare')
            for component in [hazy,reference,methods,cap]:
                component.change(invalidate,outputs=outputs,queue=False,cancels=[event],api_name=False)
            clear.click(lambda:(None,None,METHODS,'512 px',None,None,None,[],'Ready for a new image.',gr.update(value=None,visible=False)),
                        outputs=[hazy,reference,methods,cap,*outputs],queue=False,cancels=[event],api_name=False)
        with gr.Tab('Full test results'):
            initial_rows,choices,initial_message=saved_status()
            full_message=gr.Markdown(initial_message)
            full_table=gr.Dataframe(value=initial_rows,headers=['Method','Pairs','Mean PSNR (dB)','Mean SSIM','Median GPU (ms)','Peak (MiB)'],interactive=False)
            gr.Markdown('Original 640 × 512 images. FP32, seed 0, no resizing. Paper-reported results and this local test use different protocols.')
            refresh=gr.Button('Refresh full-test results')
            choice=gr.Dropdown(choices=choices,value=choices[0] if choices else None,label='Recorded image')
            with gr.Row(): saved_sliders=[gr.ImageSlider(label=m,height=300,format='png') for m in METHODS]
            saved_metrics=gr.Dataframe(headers=HEADERS,interactive=False,label='Recorded three-forward timing and quality')
            saved_ref=gr.Image(label='Aligned clear reference',height=240,interactive=False,format='png')
            refresh.click(refresh_saved,outputs=[full_table,choice,full_message],api_name='refresh_benchmark')
            choice.change(saved_image,inputs=choice,outputs=[*saved_sliders,saved_metrics,saved_ref],api_name='saved_image')
        with gr.Tab('Methods and protocol'):
            gr.Markdown('''### Three published methods
- **SFMN (TIP 2025):** two-stage spatial/frequency modulation. [Official project](https://github.com/it-hao/SFMN)
- **SFHformer (ECCV 2024):** spatial and Fourier features. [Official project](https://github.com/deng-ai-lab/SFHformer)
- **IPC-Dehaze (CVPR 2025):** iterative code prediction and criticism. [Official project](https://github.com/Jiayi-Fu/IPC-Dehaze)

### Interpreting results
The application uses official pretrained releases. SFMN and SFHformer use OTS weights, while IPC-Dehaze uses its release targeting real haze. The papers use different synthetic training pipelines; IPC-Dehaze also learns a high-quality codebook. The full test evaluates transfer to RW2AH rather than a common training budget. PSNR and SSIM require an aligned reference. More visible contrast can coexist with larger reference error.

GPU timings exclude first loading, transfers, file operations and metrics. The full test uses three recorded forwards per image; a new comparison uses one forward after warmup. Memory is PyTorch allocated-tensor peak. The SFHformer release measured here has 4.75M parameters, differing from the paper's 3.87M configuration.

Assignment 1 remains separate and uses PGL-Net-T, HNDiff-FocalNet and PixRestore-S at its original evaluation settings. Avoid interpreting the two assignment tables as one controlled six-model comparison.
''')
    return demo

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=7861);ap.add_argument('--open',action='store_true')
    args=ap.parse_args()
    demo=create_ui()
    demo.queue(default_concurrency_limit=1,max_size=8).launch(server_name='127.0.0.1',server_port=args.port,
        inbrowser=args.open,css=CSS,theme=gr.themes.Soft(primary_hue='teal',neutral_hue='slate'),footer_links=[],
        allowed_paths=[str(p) for p in (ROOT,ROOT.parent/'data',ROOT.parent/'hw1/data') if p.is_dir()],
        max_file_size='32mb',show_error=True)
