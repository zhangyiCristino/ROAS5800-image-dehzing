# Single-image dehazing UI

This directory contains the standalone English assignment UI and eight real saved test examples. It supports sample selection, method selection, input/output sliders, measured per-image results and ZIP downloads. The first example loads by default.

**Public website:** https://zhangyicristino.github.io/ROAS5800-image-dehzing/

From the repository root:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory site
```

Visit **http://127.0.0.1:8765/**. Use an HTTP server rather than opening `index.html` directly so the browser can load `samples/manifest.json`.

Saved results come from the completed 19 October 2026 RW2AH evaluation at maximum side 512, seed 0 and three timing runs per image. They are not simulated or recomputed when clicked. Each ZIP contains the processed input, clean reference, all three saved model outputs, original metric CSV and run settings. Parameter and timing differences are documented in the main repository.

New uploads are preview-only: they clear saved outputs and measurements. Use the repository's Gradio application for real new-image inference. This website has no model backend. GitHub Pages serves it through `.github/workflows/pages.yml`; changes to this directory on `main` trigger deployment.

The example photographs and model outputs retain their original attribution and applicable rights; see [third-party notices](../THIRD_PARTY_NOTICES.md). The website manifest records the original evaluation identity and output pixel hashes.
