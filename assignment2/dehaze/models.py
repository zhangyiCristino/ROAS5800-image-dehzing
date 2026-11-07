"""Inference-only adapters. Official network definitions remain unchanged."""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

def module_from_file(name: str, path: Path, omit=()):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    source = path.read_text(encoding='utf-8-sig')
    for line in omit:
        if line not in source:
            raise ValueError('Expected optional import not found: ' + line)
        source = source.replace(line, '')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module

def checkpoint(path: Path, field: str | None = None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if field:
        state = state[field]
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state

class IPCNet(nn.Module):
    def __init__(self, weights: Path, iterations=8):
        super().__init__()
        root = ROOT / 'sources' / 'ipc_dehaze' / 'basicsr'
        # Bypass BasicSR's training/UI initializers. Relative imports still load
        # the author's original architecture, registry and decoder files.
        for name, directory in [('basicsr', root), ('basicsr.archs', root / 'archs'),
                                ('basicsr.utils', root / 'utils')]:
            if name not in sys.modules:
                package = types.ModuleType(name)
                package.__path__ = [str(directory)]
                sys.modules[name] = package
        arch = importlib.import_module('basicsr.archs.dehazeToken_arch')
        self.decode = importlib.import_module('basicsr.utils.parallel_decode').decode_critic
        self.predictor = arch.DehazeTokenNet(codebook_params=[64, 1024, 256],
                                            blk_depth=16, LQ_stage=True,
                                            predictor_name='swinLayer')
        self.critic = arch.Critic()
        self.predictor.load_state_dict(checkpoint(weights / 'predictor.pth', 'params'), strict=True)
        self.critic.load_state_dict(checkpoint(weights / 'critic.pth', 'params'), strict=True)
        self.iterations = iterations

    def forward(self, image):
        net = self.predictor
        enc_feats = net.vqgan.multiscale_encoder(image.detach())[::-1]
        feat_to_quant = net.vqgan.before_quant(enc_feats[0])
        b, _, h, w = feat_to_quant.shape
        # Retain the encoder/quantizer computation in official inference.py.
        _, _, _lq_tokens = net.vqgan.encode(image)
        mask_tokens = -torch.ones(b, h*w, device=image.device, dtype=torch.long)
        def to_logits(seq, h=0, w=0, critic=False):
            return self.critic(seq, h, w) if critic else net.transformer(seq, critic)
        def to_feats(seq):
            return net.vqgan.quantize.get_codebook_entry(seq.long())
        output_tokens, _ = self.decode(mask_tokens, feat_to_quant, to_logits,
                                       to_feats, num_iter=self.iterations)
        quant = to_feats(output_tokens[:, -1, :].reshape(b, 1, h, w))
        x = net.vqgan.after_quant(quant)
        for i in range(net.max_depth):
            cur_res = net.gt_res // 2**net.max_depth * 2**i
            x = net.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, 1)
            x = net.vqgan.decoder_group[i](x)
        return net.vqgan.out_conv(x)

def load_model(method: str, device='cuda', split='ots'):
    if method == 'SFMN':
        source = ROOT / 'sources' / 'sfmn' / 'test_code'
        mod = module_from_file('official_sfmn', source / 'model' / 'sfmn.py')
        net = mod.Dehaze()
        weights = source / 'ckpt' / ('sfmn_' + split + '.pt')
        net.load_state_dict(checkpoint(weights), strict=True)
        paths = [weights]
    elif method == 'SFHformer':
        source = ROOT / 'sources' / 'sfhformer' / 'dehazing'
        mod = module_from_file('official_sfhformer', source / 'models' / 'sfhformer_haze.py',
                               omit=('from torchstat import stat', 'from matplotlib import pyplot as plt'))
        net = mod.sfhformer_haze()
        weights = source / ('sfhformer_haze_' + split + '.pth')
        net.load_state_dict(checkpoint(weights, 'state_dict'), strict=True)
        paths = [weights]
    elif method == 'IPC-Dehaze':
        weights = ROOT / 'checkpoints' / 'ipc_dehaze'
        net = IPCNet(weights, iterations=8)
        paths = [weights / 'predictor.pth', weights / 'critic.pth']
    else:
        raise ValueError(method)
    metadata = {'method': method, 'precision': 'FP32', 'strict_load': True,
                'parameters': sum(p.numel() for p in net.parameters()),
                'checkpoint_split': split if method != 'IPC-Dehaze' else 'official real-world release',
                'iterations': 8 if method == 'IPC-Dehaze' else 1,
                'weights': [{'file': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    return net.to(device).eval(), metadata

def final_output(method: str, output):
    return output[3] if method == 'SFMN' else output

def mirror_pad(image):
    """Exactly IPC-Dehaze's next-multiple-of-32 mirror-concatenation padding."""
    h, w = image.shape[-2:]
    hp = (h // 32 + 1) * 32 - h
    wp = (w // 32 + 1) * 32 - w
    image = torch.cat([image, torch.flip(image, [2])], 2)[..., :h+hp, :]
    image = torch.cat([image, torch.flip(image, [3])], 3)[..., :w+wp]
    return image, (h, w)
