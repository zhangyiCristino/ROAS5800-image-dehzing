# Third-party sources and terms

The original authors retain ownership of their model definitions, checkpoints and datasets. This repository contributes the shared inference adapters, evaluation, interface and assignment analysis. It does not claim authorship of PGL-Net, HNDiff, PixRestore, DINOv2 or RW2AH.

The public Git repository does not bundle third-party model source trees, pretrained checkpoints, original dataset photos or image-bearing PDF/PPTX files. `scripts/prepare_assets.py` retrieves the model assets directly from their upstream sources and verifies their recorded hashes. The original local submission's source manifest is retained as a provenance record; statements in that historical manifest about files being “bundled” describe that local submission, not the lightweight Git checkout.

## Model sources

| Component | Author sources | License information observed on 19 October 2026 |
| --- | --- | --- |
| PGL-Net / PGL-Net-T | [Official repository](https://github.com/sc-30-bit/PGL-Net), [paper](https://arxiv.org/abs/2606.25732) | The inspected official tree contained no `LICENSE`, `COPYING` or `NOTICE`, and its README contained no explicit license declaration. This project does not assign an MIT, Apache or other license to that code or its checkpoints. |
| HNDiff / FocalNet configuration | [Official repository](https://github.com/Jin-Ting-He/HNDiff), [paper](https://arxiv.org/abs/2608.10995) | The inspected official tree contained no such license file or explicit README license declaration. This project does not grant a new license to that code or its checkpoints. |
| PixRestore-S | [Official repository](https://github.com/csslc/PixRestore), [paper](https://arxiv.org/abs/2608.16793), [official model card](https://huggingface.co/VCLab-PolyU/PixRestore) | The README declares Apache-2.0 and the model card metadata says `apache-2.0`. An actual repository `LICENSE` file was unavailable in the inspected tree; this record does not claim that a complete license file was supplied there. |
| DINOv2 ViT-S/14 | [Official repository](https://github.com/facebookresearch/dinov2), [paper](https://arxiv.org/abs/2304.07193), [official LICENSE](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | Apache License 2.0. The preparation script preserves the upstream license and source-file copyright headers for the downloaded subset. |

The absence of a license in an inspected snapshot is a record of that inspection, not a statement about every past or future release. Availability for download alone does not establish permission to redistribute or use assets for every purpose. Consult the upstream authors for applicable terms where they are unclear. This repository supplies source provenance and download instructions rather than re-licensing third-party material.

## Checkpoint origins

| Local prepared path | Author download source |
| --- | --- |
| `checkpoints/pgl_net/rw2ah_pglnet_t.pk` | [PGL-Net author's checkpoint file](https://drive.google.com/file/d/1Sl7cCSK3peAeOzEXWvv2_3m9cJFvrvJx/view) |
| `checkpoints/hndiff/final_FocalNet.pth` and `final_dm_FocalNet.pth` | [HNDiff author's checkpoint folder](https://drive.google.com/drive/folders/1FOXemD4AK2U71ADdBikSFGHD1q9A_FXq), FocalNet/RW2AH configuration |
| `checkpoints/pixrestore/ema_model.safetensors` and `config.json` | [Official PixRestore-S model files](https://huggingface.co/VCLab-PolyU/PixRestore/tree/main/pixrestore-s) |
| `checkpoints/pixrestore/dinov2_vits14_pretrain.pth` | [Meta's official ViT-S/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |

Exact expected bytes and SHA-256 values are recorded in the preparation metadata and `third_party/SOURCE_MANIFEST.json`. No checkpoint is trained or updated by this assignment.

## Source adaptations and snapshot identity

- PGL-Net's `PGLNet.py` preserves the original model definition and numerical forward.
- In HNDiff, only unused `from thop import profile` imports are removed from `FocalNetPrior.py`, `HazeDiffusionUnet.py` and `LatentEncoder.py`. The numerical forward and four reverse steps remain unchanged. The local evaluation wrapper uses the documented common full-image protocol.
- PixRestore's original flow, model and vision definitions are preserved. The local backend constructs the original DINOv2 architecture and loads the local checkpoint strictly, avoiding its remote loader. Optional compilation and xFormers are disabled in the tested Windows setup.
- The minimal DINOv2 models/layers import closure preserves original definitions and copyright headers.

For PGL-Net, HNDiff and PixRestore, the original experiment used unpinned official snapshots retrieved on 19 October 2026. The manifest's exact file hashes identify what was run. Later audit commits were PGL-Net `dfb418adf6fd2dd55ad6f4bec72128f618d50eeb`, HNDiff `d5f2385dae1f4de80f7192d9edd73947ddcbecd5` and PixRestore `909ca06e614cf72e218d52b4fa41077c603d872b`; these are not asserted to be the original retrieval commits. The DINOv2 source snapshot was `7764ea0`.

## Dataset attribution

RW2AH (Real-World Well-Aligned Haze) is associated with Fang et al.'s AAAI 2025 work: [author repository](https://github.com/fiwy0527/AAAI25_SGDN), [paper](https://arxiv.org/abs/2412.17496). The assignment uses the test split from the [archive linked by the PGL-Net authors](https://drive.google.com/file/d/1RmqMw-105j0Berybl8EAdRwVGlnfG3mT/view). It retains the source pairing and records file hashes in the dataset manifest. Original image files are fetched from that source rather than redistributed in this Git repository. Dataset use remains subject to the authors' applicable terms.

## Citation and project licensing

When describing the experiment, cite the model papers and implementations above, the DINOv2 encoder and the RW2AH dataset. Describe the assignment's own work as integration, UI development and a local released-system comparison. The upstream READMEs provide the authors' preferred citation entries.

No blanket license in this repository grants rights to all referenced third-party assets. Upstream terms remain applicable to downloaded files. The presence of public project code does not itself assign a new software license to it.
