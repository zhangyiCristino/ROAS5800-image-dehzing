> Public text outline. The original editable PPTX remains in the local course submission.

# Single-image dehazing

ROAS5800 Assignment 1

Three recent pretrained systems on a laptop GPU

中文讲解：本作业整合三个公开预训练方法，在统一设置下比较去雾质量与本机运行成本。


---

## 2. Haze removal for camera images

- Haze reduces scene contrast and obscures structure.
- Compare three low-level restoration algorithms in one UI.
- Measure image fidelity and local inference cost.
- Simplified scattering model: I(x) = J(x)t(x) + A[1-t(x)].
- Robot perception motivates the task. Robot performance remains untested.

中文讲解：去雾是作业允许的低层视觉任务。经典散射模型将雾图 I 表示为清晰图 J、透射率 t 和大气光 A 的组合，但真实非均匀雾可偏离此简化模型。机器人相机提供动机，本作业只验证图像恢复，没有机器人任务或 Real-to-Sim 收益实验。

来源：
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- https://arxiv.org/abs/2606.25732
- https://github.com/sc-30-bit/PGL-Net

---

## 3. PGL-Net-T: global and local restoration

- PAF aligns global feature distributions.
- DAM adapts local restoration to degradation.
- RW2AH Tiny checkpoint, 0.78M parameters, FP32.
- Direct feedforward inference.

中文讲解：PGL-Net 使用物理启发的全局分布校正和局部细节恢复。此处严格采用 Tiny 的 RW2AH 权重，不将 Small 或其他数据集结果混入。

来源：
- https://arxiv.org/abs/2606.25732
- https://github.com/sc-30-bit/PGL-Net
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 4. HNDiff: a diffusion prior for FocalNet

- Haze-conditioned latent diffusion estimates a restoration prior.
- FocalNet restores the image with that prior.
- Full four-step reverse process.
- Two RW2AH checkpoints, 14.42M stored parameters, FP32.

中文讲解：HNDiff 用雾条件潜空间扩散产生先验，再由 FocalNet 恢复图像。本地推理保留 4、3、2、1 四个反向步。14.42M 为加载的两个模块总参数，不等同论文对活跃参数的计数。

来源：
- https://arxiv.org/abs/2608.10995
- https://github.com/Jin-Ting-He/HNDiff
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 5. PixRestore-S: pixel-space conditional flow

- Conditional flow matching operates on RGB pixels without a VAE.
- DINOv2 features condition the restoration network.
- Released S checkpoint with one sampling step.
- 53.70M parameters including DINO, BF16 autocast.

中文讲解：PixRestore 在像素空间学习条件流，DINOv2 提供多层视觉特征。本次采用单步 S 检查点，53.70M 包含冻结的 DINO 编码器。它是通用多任务检查点，训练数据不与另两种方法相等。

来源：
- https://arxiv.org/abs/2608.16793
- https://github.com/csslc/PixRestore
- https://huggingface.co/VCLab-PolyU/PixRestore
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 6. A local interface for reproducible comparison

- Upload a hazy image and an optional aligned reference.
- Select one or all three methods.
- Inspect restoration sliders and measured metrics.
- Export PNG predictions, CSV metrics and JSON settings.
- Screenshot: single-image example 1-0010-02.

中文讲解：界面允许上传雾图，按需加入清晰参考图，选择模型后查看滑动对比和指标，并导出图像及实验设置。截图只是单张样例，不能替代完整测试均值。

来源：
- Local application: hw1/app.py
- Screenshot: ../docs/ui_full_results_en.png

---

## 7. RW2AH test protocol

| Setting | Value |
| --- | --- |
| Test set | All 352 aligned RW2AH test pairs |
| Resolution | Longest side 512, aspect ratio preserved |
| Metrics | RGB PSNR and Gaussian SSIM on saved uint8 outputs |
| Aggregation | Equal-weight mean over images |
| Timing | 3 runs per image, median GPU forward time |
| Execution | Batch 1, seed 0, RTX 4060 Laptop 8 GB |

中文讲解：最终比较覆盖全部 352 对 test 图像。共同缩放改变了绝对分数，所以这里报告本项目的统一协议，而不是论文表格复现。HNDiff 采用共同全图流程，没有照搬官方 384 重叠裁块。

来源：
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 8. Restoration quality across the test split

| Method | PSNR (dB) | SSIM | PSNR gain (dB) |
| --- | --- | --- | --- |
| Untreated | 19.01 | 0.5817 | 0.00 |
| PGL-Net-T | 22.82 | 0.6489 | +3.81 |
| HNDiff-FocalNet | 22.36 | 0.6640 | +3.36 |
| PixRestore-S | 18.60 | 0.5706 | -0.41 |

PGL-Net-T leads PSNR. HNDiff leads SSIM.
PixRestore-S falls below the untreated baseline.

中文讲解：本表所有分数都来自同一次 352 图统一评测，没有拼接论文原表。PGL-Net-T 的平均 PSNR 最高，HNDiff-FocalNet 的 SSIM 最高。PixRestore-S 的平均 PSNR 18.60 与 SSIM 0.5706 均低于未处理图像 19.01 与 0.5817，不能写成所有模型均改善。训练分布和当前推理协议不同可能影响比较，但实验未单独定位原因。结论只支持这组检查点及协议下的判断，不能证明全领域排名。

来源：
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 9. Visual comparison on a typical test case

Example 12-0002-05, selected at the median PSNR gain of PGL-Net-T.

Untreated: 23.29 dB, SSIM 0.8497
PGL-Net-T: 26.76 dB, SSIM 0.8692
HNDiff-FocalNet: 26.56 dB, SSIM 0.8679
PixRestore-S: 23.08 dB, SSIM 0.8190

中文讲解：样例 12-0002-05 按 PGL-Net-T 的 PSNR 提升排序，选取中位数样例。所有小图来自保存的真实输出，展示整图，不合成局部结构。示例不代表完整数据分布。

来源：
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 10. A challenging case exposes remaining error

Example 2-0019-03, lowest PSNR gain of PGL-Net-T.

Untreated: 18.69 dB, SSIM 0.7535
PGL-Net-T: 16.31 dB, SSIM 0.6897
HNDiff-FocalNet: 17.63 dB, SSIM 0.7258
PixRestore-S: 18.77 dB, SSIM 0.7429

中文讲解：样例 2-0019-03 是 PGL-Net-T 在本测试集上相对原图 PSNR 提升最小的样例。展示它可以揭示平均分隐藏的失败情况。这里不根据一张图声称所有方法都有同一种视觉缺陷。

来源：
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 11. Inference cost and limitations

| Method | Params (M) | GPU time (ms) | Peak tensor MiB |
| --- | --- | --- | --- |
| PGL-Net-T | 0.78 | 28.17 | 175.82 |
| HNDiff-FocalNet | 14.42 | 247.89 | 271.50 |
| PixRestore-S | 53.70 | 69.59 | 314.33 |

- PGL-Net-T: dataset-specific checkpoint and remaining scene errors.
- HNDiff: four reverse steps add latency. Across repeated runs, 327/352 outputs differ by at most one uint8 level.
- PixRestore-S: lower mean PSNR and SSIM than untreated images in this protocol.
- Mixed precision, resized evaluation and unequal training limit comparisons.

中文讲解：参数计数包含载入模块。时间为每张图三次推理中位数再对图取中位数，显存为全部图中的 PyTorch 张量分配峰值，不含驱动上下文。FP32 与 BF16 不同，不能称为同精度架构效率实验。PGL 依赖领域权重，HNDiff 四步增加延迟。HNDiff 在327/352图像的重复运行中存在最大1灰度级的输出差异，其余两个方法全部逐像素相同。PixRestore-S 的完整测试平均 PSNR 和 SSIM 均低于原图，训练目标和协议差异是需要进一步排查的因素，当前未做归因实验。

来源：
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json

---

## 12. References and scope

1. Qu et al. PGL-Net. arXiv:2606.25732, 2026.
2. He et al. HNDiff. arXiv:2608.10995, 2026.
3. Sun et al. PixRestore. arXiv:2608.16793, 2026.
4. RW2AH dataset. arXiv:2412.17496.

Scope: released-system comparison under the local 512-pixel protocol. No equal-training-data claim, full-field SOTA claim or robot-task evaluation.

中文讲解：三个近期方法、统一界面和本机测试共同构成本作业。该比较没有统一训练数据或重训练，不报告全领域 SOTA。机器人、三维重建与 Real-to-Sim 仅作为后续方向，未加入当前实验结论。

来源：
- https://arxiv.org/abs/2606.25732
- https://github.com/sc-30-bit/PGL-Net
- https://arxiv.org/abs/2608.10995
- https://github.com/Jin-Ting-He/HNDiff
- https://arxiv.org/abs/2608.16793
- https://github.com/csslc/PixRestore
- https://huggingface.co/VCLab-PolyU/PixRestore
- https://arxiv.org/abs/2412.17496
- https://github.com/fiwy0527/AAAI25_SGDN
- Local experiment: hw1/results_full/summary.json
- Per-image results and saved predictions: hw1/results_full/evaluation.json