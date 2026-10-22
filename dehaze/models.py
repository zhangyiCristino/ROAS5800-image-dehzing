"""Offline, serialized inference for the three released dehazing systems.

Prepare checkpoints and model definitions with scripts/prepare_assets.py first.
No downloads, training, torch.hub calls, or unrestricted pickle loads occur here.
See third_party/SOURCE_MANIFEST.json for provenance and source adaptations.
"""
from __future__ import annotations

import os

# Set before importing model definitions containing optional compilation decorators.
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["XFORMERS_DISABLED"] = "1"

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
import time
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import torch

METHODS = ("PGL-Net-T", "HNDiff-FocalNet", "PixRestore-S")
ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_HASHES = {
    "pgl_net/rw2ah_pglnet_t.pk": "1ea04c0d993cc2e5aa6a36a6c77f5ac7719d4faa1fede551c3bc8db23f89526e",
    "hndiff/final_FocalNet.pth": "ed02d0ad8bf9f3cbdb14ca40678fe0cc3b4fa011085de4ce86958786363e9d27",
    "hndiff/final_dm_FocalNet.pth": "144c6b5949b99fb34d90c04138c3555f32ddc0d7b2607d7c7476c8dbb67bfb42",
    "pixrestore/ema_model.safetensors": "13c10ca30fa865977de393457e731db9f2469159dbd69c85fb3c79b3edf06895",
    "pixrestore/dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}


def prepare_image(image: np.ndarray, max_side: int | None = 512) -> tuple[np.ndarray, dict]:
    """Validate RGB uint8 and downsample by a common, aspect-preserving rule.

    Returns the resized image WITHOUT padding. Evaluation should apply this same
    function to aligned input/reference images, then call restore(max_side=None).
    Sizes in all metadata are [width, height]. No upscaling or cropping is used.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a numpy uint8 HWC RGB array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have nonempty shape (height, width, 3)")
    if max_side is not None:
        if isinstance(max_side, bool) or not isinstance(max_side, (int, np.integer)) or max_side < 1:
            raise ValueError("max_side must be a positive integer or None")
        max_side = int(max_side)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width)) if max_side is not None else 1.0
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    if (new_width, new_height) != (width, height):
        result = np.array(Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BICUBIC))
    else:
        result = np.ascontiguousarray(image).copy()
    return result, {
        "original_size": [width, height], "processed_size": [new_width, new_height],
        "max_side": max_side, "scale": scale,
        "resize": "PIL BICUBIC; round(width*scale), round(height*scale); downsample only",
    }


def _pad_image(image: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    bottom, right = (-height) % 64, (-width) % 64
    # NumPy reflect handles padding larger than a dimension, including 1px axes.
    padded = np.pad(image, ((0, bottom), (0, right), (0, 0)), mode="reflect")
    return np.ascontiguousarray(padded), {
        "padded_size": [width + right, height + bottom],
        "padding": {"left": 0, "top": 0, "right": right, "bottom": bottom, "mode": "reflect"},
    }


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _checkpoint(root: Path, relative: str) -> Path:
    path = root / "checkpoints" / relative
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}. Run python scripts/prepare_assets.py first.")
    if _sha256(path) != _CHECKPOINT_HASHES[relative]:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch; rerun scripts/prepare_assets.py: {path}")
    return path


def _import_local(root: Path, directory: str, name: str):
    path = str(root / "third_party" / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    module = importlib.import_module(name)
    # Do not silently reuse a same-named package installed elsewhere.
    if not Path(module.__file__).resolve().is_relative_to(Path(path).resolve()):
        raise RuntimeError(f"Import name collision for {name}: {module.__file__}")
    return module


def _safe_torch_load(path: Path, allowed: list, expected_globals: set[str]) -> dict:
    actual = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if actual - expected_globals:
        raise RuntimeError(f"Unreviewed checkpoint globals in {path.name}: {sorted(actual)}")
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def _dino_features(self, image, masks=None):
    """Same feature traversal attached by official PixRestore vision.load_dinov2."""
    features = {}
    tokens = self.prepare_tokens_with_masks(image, masks)
    for index, block in enumerate(self.blocks):
        tokens = block(tokens)
        features[index] = tokens[:, 1:]
    return features, self.norm(tokens)[:, 1:]


@dataclass
class _Bundle:
    method: str
    models: tuple
    precision: str
    parameters: int
    forward: Any
    details: dict = field(default_factory=dict)
    warmed_shapes: set = field(default_factory=set)


def _load_bundle(root: Path, method: str, device: torch.device) -> _Bundle:
    if method == "PGL-Net-T":
        source = _import_local(root, "pgl_net", "PGLNet")
        model = source.pglnet_t()
        path = _checkpoint(root, "pgl_net/rw2ah_pglnet_t.pk")
        state = _safe_torch_load(
            path, [np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType],
            {"numpy.core.multiarray.scalar", "numpy.dtype"},
        )
        model.load_state_dict(state["model"], strict=True)
        del state
        model.to(device).eval().requires_grad_(False)
        return _Bundle(method, (model,), "float32", sum(p.numel() for p in model.parameters()),
                       lambda x: model(x), {"input_range": "[0,1]", "checkpoint_dataset": "RW2AH", "steps": 1})

    if method == "HNDiff-FocalNet":
        source = _import_local(root, "hndiff", "FocalNet.model.FocalNetPrior")
        diffusion_source = _import_local(root, "hndiff", "FocalNet.model.HazeDiffusionUnet")
        model, diffusion = source.FocalNetPriorV5(), diffusion_source.HazeDiffusion()
        for net, filename, key in (
            (model, "final_FocalNet.pth", "model_state"),
            (diffusion, "final_dm_FocalNet.pth", "model_dm_state"),
        ):
            state = _safe_torch_load(_checkpoint(root, "hndiff/" + filename),
                                     [argparse.Namespace], {"argparse.Namespace"})
            net.load_state_dict({k.removeprefix("module."): v for k, v in state[key].items()}, strict=True)
            del state
            net.to(device).eval().requires_grad_(False)
        reverse_steps = diffusion.time_stamps_list.tolist()
        if reverse_steps != [4, 3, 2, 1]:
            raise RuntimeError(f"Unexpected HNDiff reverse schedule: {reverse_steps}")
        return _Bundle(method, (model, diffusion), "float32",
                       sum(p.numel() for net in (model, diffusion) for p in net.parameters()),
                       lambda x: model(x, diffusion(x))[2], {
                           "input_range": "[-0.5,0.5]", "checkpoint_dataset": "RW2AH",
                           "reverse_steps": reverse_steps, "steps": 4,
                           "parameter_note": "Released checkpoint pair has 14.42M parameters; not the paper's 7.82M configuration.",
                           "inference_protocol": "Full padded image; four reverse steps, then restoration. Does not use official 384px overlap-patch evaluation wrapper.",
                       })

    if method == "PixRestore-S":
        from safetensors.torch import load_file
        source = _import_local(root, "pixrestore", "pixrestore")
        vision = _import_local(root, "pixrestore", "pixrestore.vision")
        dino_source = _import_local(root, "dinov2", "dinov2.models.vision_transformer")
        cfg = json.loads((root / "checkpoints/pixrestore/config.json").read_text(encoding="utf-8"))
        state = load_file(str(_checkpoint(root, "pixrestore/ema_model.safetensors")))
        # Exact architecture flag inference used by official inference.py.
        has_lq = any(key.startswith("lq_degradation_encoder.") for key in state)
        gate_weight, count, hidden = state.get("layer_gates.weight"), len(cfg.get("encoder_layers") or []), int(cfg.get("hidden_size") or 0)
        if gate_weight is not None and hidden > 0 and count > 1:
            has_lq |= int(gate_weight.shape[1]) == hidden * (count + 4)
        if has_lq:
            cfg["use_lq_degradation_token"] = True
        cfg["register_local_gate_logit_scale"] = "local_gate_logit_scale" in state
        config = SimpleNamespace(**cfg)
        model = source.LightningDiT_PixelDiffusion(
            input_size=config.resolution, patch_size=config.patch_size, in_channels=6, out_channels=3,
            hidden_size=config.hidden_size, depth=config.depth, num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio, z_dims=config.encoder_dim if config.use_venc else None,
            encdim_ratio=config.encoder_dim_ratio, use_qknorm=config.use_qknorm,
            use_swiglu=config.use_swiglu, use_rope=config.use_rope, use_rmsnorm=config.use_rmsnorm,
            adain_single=True, num_fused_layers=len(config.encoder_layers), pca_dim=config.bottleneck_dim,
            use_bottleneck_patch_embed=True, use_dino_layer_router=config.use_dino_layer_router,
            dino_layer_router_mode=config.dino_layer_router_mode,
            same_condition_layer_weights=config.same_condition_layer_weights,
            gate_temperature=config.gate_temperature, feature_norm=config.feature_norm,
            use_lq_degradation_token=bool(getattr(config, "use_lq_degradation_token", False)),
            register_local_gate_logit_scale=config.register_local_gate_logit_scale,
        )
        model.load_state_dict(state, strict=True)
        del state
        # Same kwargs as official DINO hub _make_dinov2_model(..., pretrained=False),
        # constructed locally so there is no hub import or network fallback.
        encoder = dino_source.vit_small(img_size=518, patch_size=14, init_values=1.0,
                                       ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
                                       interpolate_antialias=False, interpolate_offset=0.1)
        state = _safe_torch_load(_checkpoint(root, "pixrestore/dinov2_vits14_pretrain.pth"), [], set())
        encoder.load_state_dict(state, strict=True)
        del state
        encoder.forward_with_features = MethodType(_dino_features, encoder)
        for net in (model, encoder):
            net.to(device).eval().requires_grad_(False)
        flow = source.PixelDiffusion(
            flow_ratio=config.flow_ratio, time_dist=config.time_distribution, alpha=config.alpha,
            z_start="noise", cfg_ratio=config.cfg_ratio, cfg_scale=config.cfg_scale,
            image_size=config.resolution, channels=3, interp_type="lin", uncond_type="zero",
            norm_p=config.adaptive_loss_power, accelerator=SimpleNamespace(device=device),
            t_start=0, t_end=1, use_cos=False, args=config,
        )
        def forward(x):
            features = vision.extract_layers(encoder, x, config.encoder_layers, config.encoder_input_size)
            return flow.sample_multistep_fm(model, x, venc_fea=features, n_steps=1, schedule="linear")
        bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        return _Bundle(method, (model, encoder, flow), "bfloat16 autocast (float32 weights)" if bf16 else "float32",
                       sum(p.numel() for net in (model, encoder) for p in net.parameters()), forward,
                       {"input_range": "[-1,1]", "checkpoint_dataset": "released unified restoration checkpoint",
                        "steps": 1, "encoder_input_size": config.encoder_input_size,
                        "restoration_parameters": sum(p.numel() for p in model.parameters()),
                        "encoder_parameters": sum(p.numel() for p in encoder.parameters())})
    raise ValueError(f"Unknown method: {method}. Choose one of {METHODS}")


class ModelManager:
    """One process-wide cached system, with loading/inference serialized by a lock.

    CUDA is preferred; explicit device='cpu' supports a slower FP32 fallback.
    A first call for each padded shape performs one unmeasured warmup. Metadata
    separates warmup and model load from the timed forward. close() releases it.
    """
    _lock = threading.RLock()
    _active: _Bundle | None = None
    _active_key: tuple | None = None

    def __init__(self, device: str | torch.device | None = None, root: str | Path | None = None):
        self.root = Path(root).resolve() if root is not None else ROOT
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Only CUDA and CPU devices are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; use device='cpu'")
            if self.device.index is None:
                self.device = torch.device("cuda:0")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    @staticmethod
    def list_methods() -> list[str]:
        return list(METHODS)

    @classmethod
    def _release(cls):
        cls._active = None
        cls._active_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        """Release the shared model cache after pending inference completes."""
        with self._lock:
            self._release()

    def restore(self, image: np.ndarray, method: str, max_side: int | None = 512,
                seed: int = 0) -> tuple[np.ndarray, dict]:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}. Choose one of {METHODS}")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        queue_start = time.perf_counter()
        with self._lock:
            started = time.perf_counter()
            pixels, preprocessing = prepare_image(image, max_side)
            padded, padding = _pad_image(pixels)
            key = (str(self.root), str(self.device), method)
            cache_hit = type(self)._active_key == key and type(self)._active is not None
            load_started = time.perf_counter()
            if not cache_hit:
                self._release()
                bundle = _load_bundle(self.root, method, self.device)
                type(self)._active = bundle
                type(self)._active_key = key
            bundle = type(self)._active
            model_load_ms = (time.perf_counter() - load_started) * 1000 if not cache_hit else 0.0
            x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
            if method == "PixRestore-S":
                x = x / 127.5 - 1
            else:
                x = x / 255.0
                if method == "HNDiff-FocalNet":
                    x = x - 0.5
            shape = tuple(x.shape)
            devices = [self.device.index] if self.device.type == "cuda" else []
            warmup_ms, warmup_runs, gpu_ms = 0.0, 0, None
            use_bf16 = bundle.precision.startswith("bfloat16")
            def infer():
                context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
                with context:
                    return bundle.forward(x)
            def reset_rng():
                torch.random.default_generator.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.default_generators[self.device.index].manual_seed(int(seed))
            with torch.inference_mode(), torch.random.fork_rng(devices=devices):
                if shape not in bundle.warmed_shapes:
                    reset_rng()
                    warm_started = time.perf_counter()
                    warm = infer()
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    warmup_ms = (time.perf_counter() - warm_started) * 1000
                    warmup_runs = 1
                    del warm
                    bundle.warmed_shapes.add(shape)
                reset_rng()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start_event.record(torch.cuda.current_stream(self.device))
                forward_start = time.perf_counter()
                output = infer()
                if self.device.type == "cuda":
                    end_event.record(torch.cuda.current_stream(self.device))
                    end_event.synchronize()
                    gpu_ms = start_event.elapsed_time(end_event)
                forward_ms = (time.perf_counter() - forward_start) * 1000
                peak_allocated = torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None
                peak_reserved = torch.cuda.max_memory_reserved(self.device) / 2**20 if self.device.type == "cuda" else None
                if tuple(output.shape) != tuple(x.shape) or not bool(torch.isfinite(output).all().item()):
                    raise RuntimeError(f"Invalid output from {method}: shape={tuple(output.shape)}")
                # Convert the official output range, clamp to display range, then
                # round to nearest uint8 (NumPy ties-to-even) after removing padding.
                output = output[0, :, :pixels.shape[0], :pixels.shape[1]].float()
                if method == "HNDiff-FocalNet":
                    output = output + 0.5
                elif method == "PixRestore-S":
                    output = (output + 1) * 0.5
                raw_min, raw_max = float(output.min()), float(output.max())
                result = np.round(output.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                del output, x
            metadata = {
                "method": method, "device": str(self.device), "precision": bundle.precision,
                "seed": int(seed), "batch_size": 1, "parameters": bundle.parameters,
                "model_cache_hit": cache_hit, "model_load_ms": model_load_ms,
                "warmup_runs": warmup_runs, "warmup_ms": warmup_ms,
                "gpu_ms": gpu_ms, "forward_ms": forward_ms,
                "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
                "queue_ms": (started - queue_start) * 1000,
                "output_min_before_clamp": raw_min, "output_max_before_clamp": raw_max,
                "output_conversion": "official range -> [0,1]; clamp; multiply 255; numpy round; uint8",
                "timing_scope": "elapsed_ms includes preprocessing, cold loading, warmup, forward, transfer and output conversion; excludes lock wait and file I/O. gpu_ms is CUDA-event complete model forward only (includes DINO / full HNDiff pipeline), excludes warmup and input/output transfer. forward_ms is synchronized forward wall time.",
                "memory_scope": "Peak PyTorch process tensor allocation during timed forward, including model/input; excludes warmup, CUDA driver/context and other applications. Reserved memory may retain warmup allocator blocks.",
                **preprocessing, **padding, **bundle.details,
            }
            metadata["elapsed_ms"] = (time.perf_counter() - started) * 1000
            return np.ascontiguousarray(result), metadata
