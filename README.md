# HCFNet CSM Evaluation Code

This repository provides the Change-Sensitive Metric (CSM) evaluation code and word-level rule files used in the paper:

> **HCFNet: Hierarchical and Complementary Feature Embedding for Remote Sensing Change Captioning**

CSM is a reference-based evaluation metric designed for remote sensing change captioning. In addition to measuring the matching degree between predicted captions and ground-truth captions, CSM further penalizes inconsistencies in two semantic dimensions:

* Spatial Scale
* Change Extent

## Quick Start

This code only depends on the Python standard library. Python 3.10 or later is recommended.

```bash
python csm_batch.py
```

By default, this command evaluates the WHU-CDC example results provided in this repository. The expected output is as follows:

```text
Prediction items : 744
Evaluated samples: 744
Invalid items    : 0
Missing refs     : 0

CSM-1: 76.7174
CSM-2: 74.8664
CSM-3: 72.1313
CSM-4: 70.3291
```

The files used in this example are:

* Prediction results: `CapRjson/WHU.json`
* Ground-truth annotations: `GTjson/whuCCcaptions.json`
* CSM rules: `CSM Level Rules/Spatial Scale.json` and `CSM Level Rules/Change Extent.json`

## Repository Structure

```text
HCFNet-main/
├── CapRjson/
│   └── WHU.json
│  
├── CSM Level Rules/
│   ├── Change Extent.json
│   └── Spatial Scale.json
├── GTjson/
│   ├── LevirCCcaptions.json
│   ├── SECOND-CC-AUG.json
│   └── whuCCcaptions.json
├── csm_batch.py
└── README.md
```

## Prediction File Format

The prediction file can be a JSON array:

```json
[
  {
    "filename": "test_000001",
    "sentence": "a new building has appeared"
  },
  {
    "filename": "test_000002.png",
    "sentence": "some roads have been constructed"
  }
]
```

It can also be a JSON object containing a `results` array:

```json
{
  "results": [
    {
      "filename": "test_000001.png",
      "sentence": "a new building has appeared"
    }
  ]
}
```

JSON Lines format is also supported, where each line is a prediction object. Each prediction item must contain the `filename` and `sentence` fields. The program matches prediction results with ground-truth annotations according to `filename`. When the `.png` suffix is missing from the prediction filename, the program will automatically add it when necessary.

## Ground-Truth Annotation File Format

The ground-truth annotation file follows the Karpathy/COCO-style structure:

```json
{
  "images": [
    {
      "filename": "test_000001.png",
      "sentences": [
        {
          "tokens": ["a", "new", "building", "has", "appeared"],
          "raw": "a new building has appeared"
        }
      ]
    }
  ]
}
```

## Supported Dataset Files

The CSM rules contain 780 words collected from LEVIR-CC, WHU-CDC, and SECOND-CC. When evaluating different datasets, the corresponding ground-truth caption annotation file should be provided:

| Dataset   | Ground-Truth Annotation File |
| --------- | ---------------------------- |
| LEVIR-CC  | `LevirCCcaptions.json`       |
| WHU-CDC   | `whuCCcaptions.json`         |
| SECOND-CC | `SECOND-CC-AUG.json`         |

## Datasets

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

### SECOND-CC
- 6,041 bi-temporal image pairs
-  Image size: `256 × 256`
- Spatial resolution: approximately `0.5–3 m/pixel`
- 5 captions per image pair
- Total captions: 30,205


The dataset used in this paper is shared via Baidu Netdisk.

 https://pan.baidu.com/s/1UEmoXix5mRVO2tmfTnYTUQ?pwd=3ti4

Please follow the official dataset terms and licenses when downloading and using these datasets.