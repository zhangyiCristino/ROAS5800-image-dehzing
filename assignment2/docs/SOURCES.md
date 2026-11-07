# Sources, eligibility and terminology

This is an Assignment 2 survey of three published deep-learning dehazing methods,
with an additional controlled laptop evaluation. No new algorithm is claimed.

## Selected papers

1. **SFMN.** Hao Shen, Henghui Ding, Yulun Zhang, Zhong-Qiu Zhao and Xudong Jiang.
   *Spatial Frequency Modulation Network for Efficient Image Dehazing.* IEEE
   Transactions on Image Processing, 34, 2025. Full author-hosted accepted paper:
   <https://www.haoshen.site/resources/ShenHao_SFMN_TIP_2025.pdf>.
   Official source and included OTS weights: <https://github.com/it-hao/SFMN>.
   Figure 3 (PDF p.4) supports the two-stage architecture. Table I (p.7) reports
   SOTS-Indoor 41.44 dB / 0.995 and SOTS-Outdoor 37.72 dB / 0.991.

2. **SFHformer.** Xingyu Jiang, Xiuhui Zhang, Ning Gao and Yue Deng.
   *When Fast Fourier Transform Meets Transformer for Image Restoration.* ECCV
   2024, pp.381-402. Official conference paper:
   <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06190.pdf>.
   Official source and included OTS weights: <https://github.com/deng-ai-lab/SFHformer>.
   Figure 2 (PDF p.6) supports the architecture. Table 1 (p.10) reports
   SOTS-Indoor 43.03 dB / 0.9966 and SOTS-Outdoor 38.83 dB / 0.9951.
   Important implementation boundary: the released dehazing network loaded
   strictly in this project has 4,752,899 trainable parameters, while Table 1
   lists 3.87M. Report the measured release configuration, not the paper's size.

3. **IPC-Dehaze.** Jiayi Fu, Siyu Liu, Zikun Liu, Chun-Le Guo, Hyunhee Park,
   Ruiqi Wu, Guoqing Wang and Chongyi Li. *Iterative Predictor-Critic Code
   Decoding for Real-World Image Dehazing.* CVPR 2025, pp.12700-12709.
   Official conference paper:
   <https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Iterative_Predictor-Critic_Code_Decoding_for_Real-World_Image_Dehazing_CVPR_2025_paper.pdf>.
   Official source: <https://github.com/Jiayi-Fu/IPC-Dehaze>.
   Official weight folder linked by that source:
   <https://drive.google.com/drive/folders/1i5DS0jjTlm1nSXSAPd4Qp5_7dLSva_1E>.
   Figure 2 (PDF p.3) supports the architecture. Table 1 (p.5) reports RTTS
   MUSIQ 59.60 vs 55.23 for RIDCP, and TOPIQ 0.500 vs 0.412. These are
   no-reference perceptual metrics, not PSNR or a common test with the other
   two papers. The official inference default uses eight iterations. Paper
   Section 4.1 states that codebook learning uses DIV2K/Flickr2K with Real-ESRGAN
   degradations, while predictor/critic training uses RIDCP's synthetic paired
   data pipeline. “Real-world release” denotes its target application, not
   training on paired real-haze photographs.

## Scope of the term SOTA

These papers report state-of-the-art results or favorable performance/efficiency
tradeoffs under their respective publication-time evaluations. The report does
not assert a current universal 2026 ranking or that one method wins every task.
SFMN emphasizes efficient modulation, SFHformer combines spatial and Fourier
features, and IPC-Dehaze uses a generative codebook with iterative criticism.

Potential newer candidates were screened: CIM-D's author page still advertised
code as coming soon; Inf-Dehaze's official README disclosed that reconstructed
code might differ from the paper. Neither supplies stronger reproducibility
evidence for this laptop task than the selected official releases.

## Assignment exclusions

The submitted Assignment 1 PDF establishes PGL-Net-T, HNDiff-FocalNet and
PixRestore-S as the prior three methods. They are excluded. SGDN is also kept
out of the selected trio because its dataset paper appears in that report's
references. The lecture-note exclusion list has not been supplied, so no claim
of completed lecture-note deduplication is made.

## Dataset

RW2AH, from Wenxuan Fang et al., *Guided Real Image Dehazing Using YCbCr Color
Space*, AAAI 2025, 39(3), 2906-2914, DOI: 10.1609/aaai.v39i3.32297.
Authors' dataset source: <https://github.com/fiwy0527/AAAI25_SGDN>.
Only all 352 original test pairs are used. Existing local bytes are verified
against their retained release SHA-256 manifest. Dataset reuse does not make
SGDN one of the three surveyed algorithms.

## Terminology ledger

| Canonical term | Meaning |
|---|---|
| SFMN | Spatial Frequency Modulation Network |
| SFHformer | Spatial-frequency hybrid Transformer, exact author model spelling |
| IPC-Dehaze | Iterative Predictor-Critic dehazing framework |
| CSM / FM / CLM | Cross-Scale Modulator / Frequency Modulator / Cross-Level Modulator (SFMN) |
| LGPM / MCFN | Local-Global Perception Mixer / Multi-kernel ConvFFN (SFHformer) |
| VQGAN | Vector-quantized generative adversarial network |
| RW2AH | Real-World Well-Aligned Haze dataset |
| PSNR | Peak signal-to-noise ratio, dB, larger is better |
| SSIM | Structural similarity, larger is better |
| MUSIQ / TOPIQ | No-reference image quality metrics used by IPC-Dehaze's paper |
| MiB | 2^20 bytes; measured tensor-allocation peak, not full device memory use |

## Planned presentation

Ten slides, nine in the five-minute narrative and one reference appendix.
The story covers the problem and prior directions, the three mechanisms,
paper-reported results, measured native-resolution deployment, visual behavior,
and limitations/future work. Native result tables remain editable. Chinese
speaker notes supply an English timed script plus Chinese explanatory material.
Main prose and slide labels use English.
