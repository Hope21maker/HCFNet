# HCFNet: Hierarchical and Complementary Feature Embedding for Remote Sensing Change Captioning

> Official repository of **HCFNet**, a hierarchical and complementary feature embedding framework for **Remote Sensing Change Captioning (RS-CC)**.


## Main Contributions

- **Hierarchical representation for RS-CC**  
  Instead of using a single fused change representation, HCFNet decomposes change features into multiple **Hierarchical Cognitive-Vectors (HCoV)** to model multi-granular scene changes.

- **Change-Cognitive Visual Parser (CCVP)**  
  A dedicated parser is designed to hierarchically decompose the difference feature into branch-wise cognitive vectors.

- **Hierarchical Cognition Mining Loss (HCMLoss)**  
  Change masks are used as auxiliary supervision to improve the **hierarchy** and **complementarity** of HCoV.

- **Change-Cognitive Language Decoder (CCLD)**  
  A progressive language decoder is introduced to fuse HCoV into each decoding stage for more accurate caption generation.

- **Change-Sensitive Metric (CSM)**  
  A new evaluation metric is proposed to better assess semantic correctness in RS-CC beyond lexical overlap.

---

## Method Overview

HCFNet consists of four key components:

### 1. Change-Cognitive Visual Parser (CCVP)

CCVP takes the fused difference feature as input and decomposes it into multiple **Hierarchical Cognitive-Vectors (HCoV)** through parallel branches with different generalized mean pooling settings. This allows the model to capture change information at different granularities.

### 2. Hierarchical Cognition Mining Loss (HCMLoss)

HCMLoss is introduced to regularize the HCoV with two complementary objectives:

- **Change Intensity Loss**, which encourages samples with similar change intensity to be closer in the feature space.
- **Orthogonality Branch Loss**, which reduces redundancy among branches and enhances feature complementarity.

### 3. Change-Cognitive Language Decoder (CCLD)

Instead of decoding from a single collapsed change feature, CCLD progressively injects HCoV into each decoding layer through a **Change-Cognitive Attention (CCA)** mechanism, leading to better semantic grounding and more accurate descriptions.

### 4. Change-Sensitive Metric (CSM)

Traditional automatic metrics may assign similar scores to semantically different captions if they share similar word patterns. To better evaluate RS-CC models, CSM introduces level-based semantic penalties according to:

- [**Spatial Scale**](Spatial%20Scale.json)
- [**Change Extent**](Change%20Extent.json)

The values provided in this JSON file denote the specific change intensity, which correspond to the discrete levels defined in the formula $L_d(w) \in \{0, 1, 2, 3\}$.
This makes the evaluation more sensitive to semantic correctness in change description.



## Experimental Results

### Results on LEVIR-CC

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr-D | S\*_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RSICCFormer | 84.72 | 76.27 | 68.87 | 62.77 | 39.61 | 74.12 | 134.12 | 77.65 |
| Prompt-CC | 83.66 | 75.73 | 69.10 | 63.54 | 38.82 | 73.72 | 136.44 | 78.13 |
| Chg2Cap | 86.14 | 78.08 | 70.66 | 64.39 | 40.03 | 75.12 | 136.61 | 79.03 |
| RSCaMa | 85.79 | 77.99 | 71.04 | 65.24 | 39.91 | 75.24 | 136.56 | 79.24 |
| SEN | 85.10 | 77.05 | 70.01 | 64.09 | 39.59 | 74.57 | 136.02 | 78.68 |
| SFEN | 85.20 | 77.01 | 70.96 | 64.67 | 40.12 | 75.22 | 136.47 | 79.12 |
| KCFI | 86.34 | 77.31 | 70.89 | 65.30 | 39.42 | 75.47 | 138.25 | 79.61 |
| MV-CC | 86.37 | 79.01 | 72.03 | 66.22 | 40.20 | 75.73 | 138.28 | 80.11 |
| CCExpert | 86.65 | 78.47 | 71.31 | 65.49 | 41.82 | 76.55 | 143.32 | 81.80 |
| SAT-Cap | 86.14 | 78.19 | 71.44 | 65.82 | 40.51 | 75.37 | 140.23 | 80.48 |
| Change3D | 85.81 | 77.81 | 70.57 | 64.38 | 40.03 | 75.12 | 138.29 | 79.46 |
| FST-Net | 86.76 | 78.82 | 71.71 | 65.67 | 40.51 | 76.15 | 140.04 | 80.59 |
| **HCFNet (Ours)** | **87.27** | **79.27** | **72.18** | **66.28** | **42.27** | **77.01** | **143.61** | **82.79** |

### Results on WHU-CDC

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr-D | S\*_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DUDA | 79.04 | 69.53 | 61.57 | 55.64 | 34.29 | 68.98 | 121.85 | 70.19 |
| MCCFormers-S | 82.14 | 76.29 | 71.08 | 66.51 | 43.50 | 79.76 | 148.88 | 84.66 |
| MCCFormers-D | 73.29 | 67.88 | 64.03 | 60.96 | 39.69 | 73.67 | 134.92 | 77.31 |
| RSICCformer-C | 78.25 | 72.82 | 68.57 | 65.14 | 44.35 | 76.50 | 143.44 | 82.36 |
| RSICCformer | 80.05 | 74.24 | 69.61 | 66.54 | 42.65 | 73.91 | 133.44 | 79.14 |
| PSNet | 81.26 | 73.25 | 65.78 | 60.32 | 36.97 | 71.60 | 130.52 | 74.85 |
| Prompt-CC | 81.12 | 73.96 | 67.22 | 61.45 | 36.99 | 71.88 | 134.50 | 76.21 |
| Chg2Cap | 78.93 | 72.64 | 67.20 | 62.71 | 41.46 | 77.95 | 144.18 | 81.58 |
| SparseFocus | 81.17 | 72.90 | 66.06 | 60.27 | 37.34 | 72.63 | 134.64 | 76.22 |
| SEN | 80.60 | 74.64 | 67.69 | 61.97 | 36.76 | 71.70 | 133.57 | 76.00 |
| DiffusionRSCC | 75.32 | 70.15 | 66.40 | 63.76 | 40.18 | 73.80 | 127.96 | 76.43 |
| Mask Approx Net | 81.34 | 75.68 | 71.16 | 67.73 | 43.89 | 75.41 | 135.31 | 80.59 |
| CTMTNet | 83.56 | 77.66 | 72.76 | 69.00 | 45.39 | 79.23 | 149.40 | 85.76 |
| **HCFNet (Ours)** | **85.24** | **78.89** | **74.02** | **69.24** | **47.56** | **80.18** | **153.39** | **89.05** |

### Results under the proposed CSM metric

| Method | LEVIR CSM-1 | LEVIR CSM-2 | LEVIR CSM-3 | LEVIR CSM-4 | WHU CSM-1 | WHU CSM-2 | WHU CSM-3 | WHU CSM-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RSICCFormer | 73.81 | 70.20 | 63.52 | 57.75 | 74.75 | 72.06 | 68.14 | 66.13 |
| Prompt-CC | 74.92 | 71.53 | 65.65 | 60.39 | 75.53 | 73.58 | 69.35 | 67.13 |
| Chg2Cap | 74.87 | 71.49 | 65.51 | 60.31 | 71.67 | 70.66 | 66.66 | 65.69 |
| RSCaMa | 75.47 | 72.05 | 65.88 | 60.99 | 75.43 | 72.87 | 69.57 | 67.42 |
| MV-CC | 75.94 | 72.40 | 65.39 | 60.05 | 74.90 | 72.93 | 69.67 | 67.80 |
| Change3D | 75.41 | 71.86 | 64.85 | 59.01 | 73.86 | 71.76 | 68.12 | 66.05 |
| **HCFNet (Ours)** | **77.74** | **74.18** | **67.40** | **62.13** | **76.70** | **75.12** | **72.06** | **69.95** |

---

## Qualitative Results

### LEVIR-CC qualitative comparison

<p align="center">
  <img src="figs/F3.png" alt="LEVIR-CC qualitative results" width="85%">
</p>

### WHU-CDC qualitative comparison

<p align="center">
  <img src="figs/F4.png" alt="WHU-CDC qualitative results" width="85%">
</p>

### Feature heatmap visualization

<p align="center">
  <img src="figs/F5.png" alt="LEVIR-CC heatmap" width="85%">
</p>

<p align="center">
  <img src="figs/F6.png" alt="WHU-CDC heatmap" width="85%">
</p>

These qualitative examples show that HCFNet can better focus on semantically meaningful changed regions and generate more accurate descriptions than previous RS-CC methods.

---

## Datasets

We evaluate HCFNet on two benchmark RS-CC datasets:

### LEVIR-CC
- 10,077 bi-temporal image pairs
- Image size: `256 × 256`
- Spatial resolution: `0.5 m/pixel`
- 5 captions per image pair
- Total captions: 50,385

### WHU-CDC
- 7,434 bi-temporal image pairs
- Image size: `256 × 256`
- Spatial resolution: `0.075 m/pixel`
- Total captions: 37,170

Please follow the official dataset terms and licenses when downloading and using these datasets.



## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{hcfnet,
  title={HCFNet: Hierarchical and Complementary Feature Embedding for Remote Sensing Change Captioning},
  author={ },
  journal={ },
  year={2026}
}
