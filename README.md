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

The dataset used in this paper is shared via Baidu Netdisk.

### https://pan.baidu.com/s/18hqUuWmUtuZ2kRb25rDFpw?pwd=xytc  

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
