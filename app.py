"""Local ROAS5800 Assignment 1 interface for three published dehazing methods."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
# Gradio checks its own loopback server at startup. Keep local requests local
# when the desktop session has a network proxy configured. Process scope only.
for _key in ("NO_PROXY", "no_proxy"):
    _hosts = [item.strip() for item in os.environ.get(_key, "").split(",") if item.strip()]
    for _host in ("127.0.0.1", "localhost", "::1"):
        if _host not in _hosts:
            _hosts.append(_host)
    os.environ[_key] = ",".join(_hosts)

import gradio as gr
import numpy as np
from PIL import Image

from dehaze.evaluation import compute_metrics, prepare_pair
from dehaze.models import ModelManager
from dehaze.benchmark import (
    BenchmarkUnavailable, IMAGE_HEADERS, SUMMARY_HEADERS,
    image_result, load_benchmark, overview,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
METHODS = ["PGL-Net-T", "HNDiff-FocalNet", "PixRestore-S"]
SLUGS = {"PGL-Net-T": "pgl_net_t", "HNDiff-FocalNet": "hndiff_focalnet", "PixRestore-S": "pixrestore_s"}
MANAGER = ModelManager()
LOGGER = logging.getLogger("dehaze.ui")

HEADERS = ["Method", "Image size", "Inference time (ms)", "Memory (MiB)", "PSNR (dB)", "SSIM"]
CSS = """
.gradio-container {max-width: 1420px !important; margin: auto !important;
  font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;}
#intro h1 {font-size: 32px; letter-spacing: -0.6px; margin-bottom: 6px;}
#intro p {color: #526474; font-size: 16px;}
#run-button {min-height: 49px; font-size: 17px;}
.result-image {min-height: 290px;}
footer {display: none !important;}
"""


def english_locale_js():
    """Set Gradio's own controls to English within this app only.

    The pinned Gradio frontend exports its locale setter from a hashed module.
    Resolve that installed asset rather than hard-code a version-specific name.
    """
    assets = Path(gr.__file__).resolve().parent / "templates/frontend/assets"
    for module in sorted(assets.glob("core-*.js")):
        if "as changeLocale" in module.read_text(encoding="utf-8"):
            module_url = json.dumps("/assets/" + module.name)
            return f'''async () => {{
                const {{ changeLocale }} = await import({module_url});
                changeLocale("en");
                document.documentElement.lang = "en";
                // Gradio captures some toolbar labels before its load callback.
                const labels = new Map([["下载", "Download"], ["撤销", "Undo"]]);
                const relabel = (node) => {{
                    if (node.nodeType !== Node.ELEMENT_NODE) return;
                    for (const element of [node, ...node.querySelectorAll("[aria-label], [title]")]) {{
                        for (const attribute of ["aria-label", "title"]) {{
                            const translated = labels.get(element.getAttribute(attribute));
                            if (translated) element.setAttribute(attribute, translated);
                        }}
                    }}
                }};
                const container = document.querySelector(".gradio-container");
                if (!container) return;
                relabel(container);
                new MutationObserver(records => {{
                    for (const record of records) {{
                        if (record.type === "attributes") relabel(record.target);
                        for (const node of record.addedNodes) relabel(node);
                    }}
                }}).observe(container, {{subtree:true, childList:true, attributes:true,
                                       attributeFilter:["aria-label", "title"]}});
            }}'''
    raise RuntimeError("The English locale control was not found. Use the Gradio version in requirements.txt.")


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if np.isposinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def _metrics(prediction: np.ndarray, reference: np.ndarray | None) -> dict:
    return compute_metrics(prediction, reference) if reference is not None else {"psnr": None, "ssim": None}


def _json_safe(value):
    """Keep exported metadata valid JSON, including exact-match PSNR infinity."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def compare(image, reference, methods, max_side, seed, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a hazy image or select a sample below.")
    if not methods:
        raise gr.Error("Select at least one method.")
    if any(method not in METHODS for method in methods):
        raise gr.Error("Invalid method selection. Please select a listed method.")
    try:
        processed, target = prepare_pair(image, reference, int(max_side))
    except (TypeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    job = OUTPUTS / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    job.mkdir(parents=True, exist_ok=False)
    Image.fromarray(processed).save(job / "input.png")
    if target is not None:
        Image.fromarray(target).save(job / "reference.png")
    width, height = processed.shape[1], processed.shape[0]
    try:
        baseline = _metrics(processed, target)
    except ValueError as error:
        raise gr.Error("Quality metrics require aligned images with the same dimensions, at least 11 pixels on each side after resizing.") from error
    rows = [["Hazy input", f"{width} × {height}", "—", "—",
             _format_number(baseline.get("psnr")), _format_number(baseline.get("ssim"), 4)]]
    records = [{"method": "Unprocessed", "psnr": baseline.get("psnr"), "ssim": baseline.get("ssim")}]
    sliders = {method: None for method in METHODS}
    failures = []
    for index, method in enumerate(methods):
        progress((index, len(methods)), desc=f"Processing: {method}")
        try:
            restored, metadata = MANAGER.restore(processed, method, max_side=None, seed=int(seed))
            if restored.shape != processed.shape:
                raise RuntimeError("The model output dimensions do not match the input.")
            quality = _metrics(restored, target)
            Image.fromarray(restored).save(job / f"{SLUGS[method]}.png")
            sliders[method] = (processed, restored)
            record = {"method": method, **metadata, **quality}
            records.append(record)
            inference_ms = metadata.get("gpu_ms")
            if inference_ms is None:
                inference_ms = metadata.get("forward_ms")
            rows.append([method, f"{width} × {height}", _format_number(inference_ms),
                         _format_number(metadata.get("peak_allocated_mib")),
                         _format_number(quality.get("psnr")), _format_number(quality.get("ssim"), 4)])
        except Exception as error:
            LOGGER.exception("Inference failed for %s", method)
            failures.append(method)
            records.append({"method": method, "error": str(error)})
            rows.append([method, f"{width} × {height}", "Failed", "—", "—", "—"])
    progress(1, desc="Results ready")

    serializable = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "original_size": [int(image.shape[1]), int(image.shape[0])],
        "processed_size": [width, height], "max_side": int(max_side), "seed": int(seed),
        "has_reference": target is not None, "results": records,
        "protocol": "Common RGB bicubic downsampling without upsampling. Restore common-sized inputs, pad internally then crop only padding. RGB PSNR/SSIM compare exported uint8 predictions with the common-sized reference. GPU timing excludes loading and UI. Different model precisions are recorded.",
    }
    (job / "metadata.json").write_text(json.dumps(_json_safe(serializable), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    with (job / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    archive = job / "dehazing_results.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(job.iterdir()):
            if path.is_file() and path != archive:
                bundle.write(path, arcname=path.name)
    completed = len(methods) - len(failures)
    status = f"Completed {completed}/{len(methods)} methods at {width} × {height}. Drag each divider to compare the input (left) and output (right)."
    if target is None:
        status += " Add an aligned clean reference of the same size to calculate quality metrics."
    if failures:
        status += " Failed: " + ", ".join(failures) + ". Check the server log or retry at a smaller image size."
    return *(sliders[method] for method in METHODS), rows, str(archive), status


def load_examples():
    manifest_path = ROOT / "data" / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = []
    for pair in manifest.get("pairs", [])[:3]:
        hazy, clear = ROOT / pair["hazy"], ROOT / pair["clear"]
        if hazy.exists() and clear.exists():
            examples.append([str(hazy), str(clear)])
    return examples


def refresh_benchmark(identifier=None, preset=None):
    try:
        snapshot = load_benchmark(ROOT)
        if preset in {"typical", "hard"}:
            identifier = snapshot[preset]
        if identifier not in snapshot["ids"]:
            identifier = snapshot["typical"]
        return (*overview(snapshot),
                gr.update(choices=snapshot["ids"], value=identifier, interactive=True),
                *image_result(snapshot, identifier))
    except BenchmarkUnavailable as error:
        LOGGER.warning("Full results unavailable: %s", error)
        return (f"### Full test results unavailable\n{error}", [], "", "", None,
                gr.update(choices=[], value=None, interactive=False),
                "Complete the evaluation, then refresh to browse saved results.", None, None, None, None, None, [])


def browse_benchmark(identifier):
    try:
        return image_result(load_benchmark(ROOT), identifier)
    except BenchmarkUnavailable as error:
        return f"{error} Click \"Refresh results\" to try again.", None, None, None, None, None, []


def _build_single_image_tab():
    gr.Markdown("Upload a photo and run the selected methods. Metrics on this tab apply only to the current image.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            image = gr.Image(label="Hazy image", sources=["upload"], type="numpy", format="png", height=300)
            with gr.Accordion("Add a clean reference for quality metrics (optional)", open=False):
                reference = gr.Image(label="Clean reference · aligned and the same size", sources=["upload"], type="numpy", format="png", height=220)
        with gr.Column(scale=2):
            methods = gr.CheckboxGroup(METHODS, value=METHODS, label="Methods")
            max_side = gr.Radio([256, 512], value=512, label="Maximum image side", info="Preserves aspect ratio. Smaller images run faster.")
            seed = gr.Number(value=0, precision=0, minimum=0, maximum=2147483647, label="Random seed", visible=False)
            run = gr.Button("Compare methods", variant="primary", elem_id="run-button")
            gr.Markdown("The first run loads model weights. All processing is local.")
            with gr.Accordion("About the methods", open=False):
                gr.Markdown("**PGL-Net-T**: a lightweight network for global correction and local detail restoration.\n\n**HNDiff-FocalNet**: four-step diffusion restoration guided by a haze formation prior.\n\n**PixRestore-S**: one-step unified restoration conditioned on visual features.")
    examples = load_examples()
    if examples:
        gr.Examples(examples, inputs=[image, reference], label="Real test samples · click to load", examples_per_page=3, cache_examples=False)
    status = gr.Markdown("Upload an image or select a sample, then click \"Compare methods\".")
    with gr.Row():
        outputs = [gr.ImageSlider(label=method, type="numpy", format="png", height=290,
                                  interactive=False, buttons=["download", "fullscreen"], elem_classes="result-image")
                   for method in METHODS]
    gr.Markdown("### Current image results\nTime measures one model forward pass, excluding initial loading. An aligned clean reference is required for RGB PSNR and SSIM; higher scores indicate a closer match. Dataset averages are shown on the \"Full test results\" tab.")
    metrics = gr.Dataframe(headers=HEADERS, datatype=["str"] * len(HEADERS), type="array", interactive=False, wrap=True, label="Current image comparison")
    download = gr.File(label="Download this comparison · images, metrics and settings", interactive=False)
    run.click(compare, inputs=[image, reference, methods, max_side, seed],
              outputs=[*outputs, metrics, download, status], api_name="compare", concurrency_limit=1)


def build_ui():
    initial = refresh_benchmark()
    with gr.Blocks(title="Image Dehazing Comparison · ROAS5800", analytics_enabled=False) as demo:
        gr.Markdown("# Image Dehazing Comparison\nExplore full test results or upload a photo to compare three methods.", elem_id="intro")
        with gr.Tabs(selected="benchmark"):
            with gr.Tab("Full test results", id="benchmark"):
                with gr.Row():
                    with gr.Column(scale=5):
                        full_heading = gr.Markdown(initial[0])
                    with gr.Column(scale=1, min_width=180):
                        refresh = gr.Button("Refresh results", variant="secondary")
                full_table = gr.Dataframe(value=initial[1], headers=SUMMARY_HEADERS,
                                          datatype=["str"] * len(SUMMARY_HEADERS), type="array",
                                          interactive=False, wrap=True, label="Dataset averages and inference cost")
                insight = gr.Markdown(initial[2])
                with gr.Accordion("Evaluation settings and interpretation", open=False):
                    protocol = gr.Markdown(initial[3])
                files = gr.File(value=initial[4], file_count="multiple", interactive=False,
                                label="Download results · summary CSV, per-image CSV and full records")
                gr.Markdown("### Browse saved test images\nView results for any test image. Drag each divider to compare the input (left) and output (right).")
                with gr.Row():
                    saved_id = gr.Dropdown(choices=initial[5]["choices"], value=initial[5]["value"],
                                           interactive=initial[5]["interactive"], label="Test image ID", scale=4)
                    typical = gr.Button("Typical example", scale=1)
                    hard = gr.Button("Challenging example", scale=1)
                gr.Markdown("Examples use the median and lowest per-image PSNR gain of the method with the highest mean PSNR.")
                caption = gr.Markdown(initial[6])
                with gr.Row():
                    saved_input = gr.Image(value=initial[7], label="Hazy input", type="filepath", height=220,
                                           interactive=False, buttons=["download", "fullscreen"])
                    saved_reference = gr.Image(value=initial[8], label="Clean reference", type="filepath", height=220,
                                               interactive=False, buttons=["download", "fullscreen"])
                with gr.Row():
                    saved_outputs = [gr.ImageSlider(value=initial[9 + index], label=method, type="filepath",
                                                     format="png", height=270, interactive=False,
                                                     buttons=["download", "fullscreen"], elem_classes="result-image")
                                     for index, method in enumerate(METHODS)]
                image_table = gr.Dataframe(value=initial[12], headers=IMAGE_HEADERS,
                                           datatype=["str"] * len(IMAGE_HEADERS), type="array",
                                           interactive=False, wrap=True, label="Selected test image · per-image metrics")
                image_components = [caption, saved_input, saved_reference, *saved_outputs, image_table]
                full_components = [full_heading, full_table, insight, protocol, files, saved_id, *image_components]
                refresh.click(refresh_benchmark, inputs=[saved_id], outputs=full_components, api_name="refresh_benchmark")
                saved_id.input(browse_benchmark, inputs=[saved_id], outputs=image_components, api_name="browse_benchmark")
                typical.click(lambda: refresh_benchmark(preset="typical"), outputs=full_components, api_name=False)
                hard.click(lambda: refresh_benchmark(preset="hard"), outputs=full_components, api_name=False)
            with gr.Tab("Single-image comparison", id="single"):
                _build_single_image_tab()
        demo.load(refresh_benchmark, inputs=[saved_id], outputs=full_components, api_name=False)
        gr.Markdown("ROAS5800 Assignment 1 · Comparison of released pretrained methods. Sources, evaluation protocol and limitations are documented in the project README.")
    return demo.queue(default_concurrency_limit=1, max_size=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="Open the local page in the default browser")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=args.open,
                share=False, show_error=True, theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
                css=CSS, js=english_locale_js(), footer_links=[], max_file_size="32mb",
                allowed_paths=[str(ROOT / "data"), str(OUTPUTS), str(ROOT / "results_full")])


if __name__ == "__main__":
    main()
