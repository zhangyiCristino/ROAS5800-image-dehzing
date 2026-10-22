> Public repository edition: source images, saved output images and binary PDF/PPT are not redistributed. The measured summary/CSV is included; run the README workflow to regenerate the full evaluation record and image browser.

# 三种近期图像去雾方法的本机比较：参考报告

> ROAS5800 Assignment 1；参考稿，供理解实验后修改使用。实验日期：2026-10-19。未填写姓名和学号。
> 公开版只包含可编辑的参考文字；英文稿见 [report_en.md](report_en.md)。完整 PDF/PPT 与数据集照片保留在本地课程提交材料中。

## 摘要

本作业围绕单幅图像去雾，实现 PGL-Net-T、HNDiff-FocalNet 和 PixRestore-S 三个公开预训练系统的共同界面，并在本机 RTX 4060 Laptop 8 GB 显卡上完成 RW2AH 来源版本全部 **352 对测试图像**的统一评测。在最长边 512、固定输入处理和共同 RGB 指标下，PGL-Net-T 的平均 PSNR 最高，为 **22.82 dB**，比未处理输入高 **3.81 dB**；PGL-Net-T 的 GPU 前向耗时中位数最低，为 **28.17 ms**。该实验比较现成系统的部署表现，没有重新训练模型，也没有复现各论文原始评测协议，结果不等于全领域 SOTA 排名。

## 1. 问题、动机与作业范围

输入是一张有雾 RGB 图像，输出是恢复后的 RGB 图像。常用的理想化大气散射模型是 `I(x) = J(x)t(x) + A(1-t(x))`，其中 I 是有雾图像，J 是清晰场景辐射，t 是透射率，A 是大气光。雾会降低对比度并改变颜色，因此需要同时考察恢复质量和运行成本。本作业选择三种算法，覆盖轻量前馈网络、物理引导扩散和统一像素空间恢复三类思路，并提供共同 UI。

本项目自己完成的工作是：官方网络和权重接入、统一输入/输出适配、离线本机推理、交互界面、批量评测与结果导出。三种原算法及其预训练权重属于原作者，不能写成“本人提出三个新模型”或“从零训练复现”。机器人感知和 Real-to-Sim 可作为后续研究动机，但本作业没有机器人或三维重建实验。

## 2. 三种方法与实现配置

| 标准名称 | 主要思路 | 本项目实际配置 | 局限 |
| --- | --- | --- | --- |
| PGL-Net-T [1] | 将全局分布校正与局部结构细化分开；PAF 提供全局条件，DAM 调整局部特征 | RW2AH Tiny 权重；FP32；0.78 M 参数 | 数据集专用模型；不能由这一组实验推断所有场景的泛化能力 |
| HNDiff-FocalNet [2] | 用雾-噪声扩散获取清晰潜空间先验，辅助 FocalNet 恢复 | RW2AH 两个配套权重；FP32；完整四个反向步骤 `[4,3,2,1]`；合计 14.42 M 参数 | 需要多步前向；本项目全图推理与作者 384 重叠裁块协议不同 |
| PixRestore-S [3] | 无 VAE 的像素空间 Transformer，结合 DINOv2 特征进行统一恢复 | 官方 S 权重；DINOv2 ViT-S/14；单步；BF16 autocast；含编码器 53.70 M 参数 | 通用多任务权重，与两个 RW2AH 专用模型的训练条件不同 |

注意：HNDiff 的本地两个检查点实际共 **14.42 M** 参数，不应套用论文其他配置的 **7.82 M**。PixRestore 的参数量和计时均包含 DINOv2 特征提取。公开日期和论文领先结果只说明方法新近，报告中不应声称三个方法同时是全部数据集的第一。

## 3. 界面与运行流程

1. 上传有雾图像或选择内置真实样例。
2. 选择一个或多个算法，设置图像最长边，启动比较。
3. 查看输入/输出滑动对比及运行时间。
4. 如有同尺寸、对齐的清晰参考图，可上传后计算 PSNR/SSIM；没有参考图时不显示虚构质量分数。
5. 下载结果 ZIP，保存实际输入、恢复 PNG、CSV 和运行配置。

推理不调用云端 API，同一时间仅驻留一个模型。源码来源、适配修改、权重下载地址及 SHA-256 保存在 [`REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)。

## 4. 数据与统一实验协议

### 4.1 数据范围

使用 RW2AH（Real-World Well-Aligned Haze）[4]，来自 PGL-Net 官方链接的公开发布版本。此次纳入该版本 **全部 352 对 test 图像**，不是此前排序前 12 对的试运行。原文件名、压缩包成员路径、原始尺寸和 SHA-256 见 [`data/manifest_full.json`](../data/manifest_full.json)，评测前验证源文件哈希。

“完成来源版本全部 test”不等于“复现论文原表”：本作业使用统一缩小和全图推理，各论文原始分辨率、切块方式、精度、指标实现可能不同。

### 4.2 输入与推理

- 输入和参考图转 RGB，必须原始尺寸完全相同。
- 两者使用同一个 PIL BICUBIC 等比例缩小，最长边最多 512；不放大小图、不额外裁切。
- 所有模型接收同一个有效输入；右侧和底部反射填充到 64 的倍数，输出移除填充，不拉回原始尺寸。
- PGL 输入范围 `[0,1]`，HNDiff 为 `[-0.5,0.5]`，PixRestore 为 `[-1,1]`；由适配器转换。
- PGL/HNDiff 为 FP32，PixRestore 为 CUDA BF16 autocast；这是实际部署设置比较，不是等精度的纯架构比较。
- batch size 为 1，固定 seed 为 0；每张图完整运行三次，第一张输出用于质量计算，三次完整 GPU 前向耗时的中位数用于该图计时。

### 4.3 指标与统计

质量指标针对**实际保存和 UI 显示的 uint8 RGB 输出**计算：输出裁到有效范围并四舍五入量化，然后除以 255 转 float64。PSNR 的 data range 为 1；SSIM 使用 11 × 11 高斯窗口、sigma=1.5、总体协方差、RGB 三通道，不转 Y、不剃边、不做直方图匹配或事后对齐。另保存归一化 RGB MAE。

全数据集质量分数为逐图分数的等权算术平均，不是先汇总所有 MSE 再换算 PSNR。表中 GPU 耗时为“逐图三次中位数”在 352 张图上的中位数；不含模型加载、预热、数据传输、读写文件及 UI。显存为整组图像中 PyTorch 张量分配峰值，不等于显卡总占用。

硬件软件：Ryzen 9 7845HX、约 16 GB 内存、RTX 4060 Laptop 8 GB，Windows，Python 3.12.10，PyTorch 2.8.0+cu128。

## 5. 完整测试结果

| Method | PSNR (dB) ↑ | SSIM ↑ | MAE ↓ | Median GPU (ms) ↓ | Peak allocated (MiB) | Parameters (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated | 19.01 | 0.5817 | 0.0991 | — | — | — |
| PGL-Net-T | 22.82 | 0.6489 | 0.0619 | 28.17 | 175.8 | 0.78 |
| HNDiff-FocalNet | 22.36 | 0.6640 | 0.0655 | 247.89 | 271.5 | 14.42 |
| PixRestore-S | 18.60 | 0.5706 | 0.1025 | 69.59 | 314.3 | 53.70 |

在该协议下，PGL-Net-T 的平均 PSNR 最高；HNDiff-FocalNet 的平均 SSIM 最高；PGL-Net-T 的 GPU 前向耗时中位数最低。HNDiff 的该耗时约为 PGL-Net-T 的 **8.8 倍**。**PixRestore-S 此次平均 PSNR 和 SSIM 均低于未处理原图**，分别低 0.41 dB 和 0.0111；不能用方法“新近”或论文的领先结果掩盖本地负结果。这些是本机真实测量，不使用论文表格数字替代本地结果。

### 定性样图

下面两个样例固定为原测试 ID `1-0010-02` 和 `1-0013-01`，用于说明界面实际处理的图像，不代表全部 352 对的视觉表现。比较应同时查看对比度、颜色、天空区域和建筑边缘，避免把更强的锐化或更高的饱和度直接等同于更准确恢复。

公开仓库不分发数据集照片。复现评测后，可在 `results_full/images/` 或 UI 中查看上述两张图片。

### 结果应如何解释

首先可以比较“是否优于未处理输入”，再比较恢复质量和时间成本。不能把结果差异全部归因于某个模块：PGL/HNDiff 使用 RW2AH 专用预训练权重，而 PixRestore 使用通用多任务权重；训练数据、训练预算和推理精度没有对齐。本作业也没有做消融实验来单独验证 PAF、DAM、扩散先验或 DINO 特征的贡献。

HNDiff 在 **327/352** 张图像的固定种子重复推理中未达到逐字节相同；重复输出的最大绝对差为 **1**（uint8 的 0-255 尺度），质量仍按预先规定使用第一次输出。这说明固定 seed 不保证 GPU 输出完全一致，不应把小量数值差异误写成不同实验条件。

## 6. 局限与后续方向

1. 仅使用一个来源版本的真实雾图像测试划分；没有重新训练、跨数据集测试或统计显著性检验。
2. 最长边 512 的缩小、共同填充和 HNDiff 全图推理会改变绝对分数，不能把本地成绩标成论文原表复现。
3. 三个系统训练条件和精度不同，结论只适用于当前权重及部署设置。
4. PSNR/SSIM 不能直接证明视觉自然度、跨视角一致性、三维几何准确度或机器人成功率。
5. 可在后续将输出接到固定的特征匹配或重建流程，测相机注册率与几何误差；这是待验证计划，本作业未完成。

## 7. 结论

本项目完成三个图像去雾算法的本机运行、共同 UI 和来源版本全部 352 对测试图像的受控比较。在当前协议下，PGL-Net-T 的平均 PSNR 最好，PGL-Net-T 的 GPU 前向耗时中位数最短。该结果可支持课程范围内的模型选择和质量/成本讨论，不能推出全领域方法排名或 Real-to-Sim/机器人收益。

## 8. 复现与证据路径

按仓库 README 准备模型和数据后，在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m dehaze.evaluation --manifest data/manifest_full.json --output results_full --max-side 512 --runs 3 --seed 0
```

- 逐图记录及每次运行元数据：`results_full/evaluation.json`（本地重新评测后生成）。
- 汇总：[`summary.json`](../results_full/summary.json)、[`summary.csv`](../results_full/summary.csv)。
- 逐图表：[`per_image.csv`](../results_full/per_image.csv)。
- 实际保存的输入、参考图和三个方法输出：`results_full/images/`。
- 方法、检查点来源和改动：[`REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)。
- 安装和 UI 操作：[`README.md`](../README.md)。

## 参考文献与官方实现

1. Qu et al. Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling. arXiv:2606.25732, 2026. [论文](https://arxiv.org/abs/2606.25732)；[官方代码/数据](https://github.com/sc-30-bit/PGL-Net)。
2. He et al. HNDiff: Haze-Noise Diffusion for Image Dehazing. ECCV 2026; arXiv:2608.10995. [论文](https://arxiv.org/abs/2608.10995)；[官方代码/数据](https://github.com/Jin-Ting-He/HNDiff)。
3. Sun et al. PixRestore: Unified Image Restoration via Pixel Diffusion Transformer. arXiv:2608.16793, 2026. [论文](https://arxiv.org/abs/2608.16793)；[官方代码/数据](https://github.com/csslc/PixRestore)。
4. Fang et al. Guided Real Image Dehazing using YCbCr Color Space. AAAI 2025; arXiv:2412.17496. RW2AH dataset. [论文](https://arxiv.org/abs/2412.17496)；[官方代码/数据](https://github.com/fiwy0527/AAAI25_SGDN)。

## 使用这份参考稿时的核对说明

- 本文主体是可修改参考稿；提交前用自己的表述解释方法和结果，并填写课程要求的署名。
- “本项目实现”指接入、封装、UI 与评测，不指原作者网络和训练方法。
- 本地结果支持表中的质量和成本结论；论文方法描述引用原论文；机器人与重建收益仍需要证据。
- 中文参考稿的结构为问题与方法、统一协议、结果、局限和复现，便于从同一证据整理两页英文报告和 PPT。

| 论断 | 证据 | 状态 |
| --- | --- | --- |
| 已完成来源版本 352 对 test 的比较 | 完整 manifest、evaluation.json 的 completion_status、1408 条逐图记录（含未处理输入） | 本地结果支持 |
| 表中质量与运行成本 | summary.json、per_image.csv、实际保存输出与运行元数据 | 本地结果支持 |
| 三种原算法的设计原理 | 原论文与官方模型定义，引用 [1]-[3] | 来源支持 |
| 去雾提高 Real-to-Sim 重建或机器人成功率 | 本作业没有此类实验 | 尚需验证，不能当作本作业结论 |
