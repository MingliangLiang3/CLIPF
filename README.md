# CLIPF: Contrastive Language-Image Pre-training with Word Frequency Masking

[![WACV 2026](https://img.shields.io/badge/WACV-2026-blue)](https://openaccess.thecvf.com/content/WACV2026/html/Liang_Frequency_Is_What_You_Need_Considering_Word_Frequency_When_Text_WACV_2026_paper.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **[Frequency Is What You Need: Considering Word Frequency When Text Masking Benefits Vision-Language Model Pre-training](https://openaccess.thecvf.com/content/WACV2026/html/Liang_Frequency_Is_What_You_Need_Considering_Word_Frequency_When_Text_WACV_2026_paper.html)**
> Mingliang Liang, Martha Larson — *WACV 2026*

CLIPF introduces a **frequency-based text masking strategy** for efficient vision-language model (VLM) pre-training. We show that the optimal masking strategy changes across training, and that **word frequency distribution** is the key factor behind effective text masking — outperforming syntax masking while being significantly simpler. CLIPF is especially effective when text tokens are heavily limited.

---

## Table of Contents

- [Key Idea](#key-idea)
- [Results](#results)
- [Installation](#installation)
- [Word Frequency Dictionary](#word-frequency-dictionary)
- [Pre-training](#pre-training)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Key Idea

Text masking quality matters. We find that:

- The **optimal masking strategy shifts** throughout the course of training.
- **Word frequency** is the core signal behind prior methods such as syntax masking.
- A simple **frequency-based masking** approach (CLIPF) matches or surpasses more complex alternatives, especially under **aggressive token budgets**.

---

## Results

### Zero-shot ImageNet-1K (ViT-B/16, 75% image masking, 30 epochs on CC12M)

Top-1 zero-shot accuracy on ImageNet-1K, before and after unmasking fine-tuning.

| Model  | Masking    | Image Tokens | Text Tokens | Pre-train | Fine-tune |
|--------|------------|:------------:|:-----------:|:---------:|:---------:|
| CLIP   | —          | 197          | 32          | 36.6      | —         |
| CLIPF  | frequency  | 98           | 8           | **39.8**  | **41.0**  |
| FLIP   | —          | 49           | 32          | 32.0      | 33.7      |
| CLIPA  | truncation | 49           | 16          | 32.8      | 32.8      |
| CLIPA  | random     | 49           | 16          | 33.7      | 34.3      |
| CLIPA  | block      | 49           | 16          | 34.3      | 34.8      |
| CLIPA  | syntax     | 49           | 16          | 32.2      | 34.4      |
| **CLIPF**  | **frequency**  | **49**   | **16**      | **35.5**  | **36.0**  |
| CLIPA  | truncation | 49           | 8           | 25.4      | 28.4      |
| CLIPA  | random     | 49           | 8           | 34.5      | 36.9      |
| CLIPA  | block      | 49           | 8           | 35.5      | 37.9      |
| CLIPA  | syntax     | 49           | 8           | 28.5      | 35.0      |
| **CLIPF**  | **frequency**  | **49**   | **8**       | **36.6**  | **39.3**  |
| CLIPA  | truncation | 49           | 6           | 15.3      | 23.2      |
| CLIPA  | random     | 49           | 6           | 26.9      | 34.6      |
| CLIPA  | block      | 49           | 6           | 28.6      | 35.9      |
| CLIPA  | syntax     | 49           | 6           | 25.2      | 32.6      |
| **CLIPF**  | **frequency**  | **49**   | **6**       | **30.3**  | **37.8**  |
| CLIPA  | truncation | 49           | 4           | 5.3       | 19.8      |
| CLIPA  | random     | 49           | 4           | 14.0      | 27.1      |
| CLIPA  | block      | 49           | 4           | 18.7      | 26.6      |
| CLIPA  | syntax     | 49           | 4           | 14.2      | 24.6      |
| **CLIPF**  | **frequency**  | **49**   | **4**       | 17.0      | **30.9**  |

### Scaling to LAION-400M

Pre-trained for 6 epochs on LAION-400M (112×112), then fine-tuned on 128M samples at 224×224 without masking. Trained on **4×H100** with `amp_bf16`.

| Method | GPU Hours | Samples Seen    | Image Size | Masking | Image Tokens | Text Tokens | Pre-train | Fine-tune |
|--------|:---------:|-----------------|:----------:|:-------:|:------------:|:-----------:|:---------:|:---------:|
| CLIPF  | 270       | 2.56B + 128M    | 112×112    | 50%     | 25           | 16          | 57.5      | 61.6      |
| CLIPF  | 300       | 2.56B + 128M    | 112×112    | 0%      | 49           | 16          | 59.8      | 63.0      |

---

## Installation

CLIPF builds on [OpenCLIP](https://github.com/mlfoundations/open_clip). Follow OpenCLIP's installation instructions first, then install the additional dependencies:

```bash
git clone https://github.com/ml-liang/CLIPF.git
cd CLIPF
pip install -r requirements-training.txt
```

---

## Word Frequency Dictionary

A word-frequency dictionary is required for frequency-based masking. You can either generate one or download a precomputed file.

**Generate from scratch:**

```bash
python tests/data_counter.py
```

**Download precomputed files:**

Precomputed frequency dictionaries are available on [Google Drive](https://drive.google.com/drive/folders/1AIoC7APPqg3H_J50_j_0ES41CvUQoKst?usp=sharing).

---

## Pre-training

The `--reduction-mask` argument controls the text masking strategy. Supported values:

| Value       | Description                        |
|-------------|------------------------------------|
| `frequency` | Word-frequency masking (CLIPF)     |
| `syntax`    | Syntax-based masking               |
| `random`    | Random token masking               |
| `block`     | Block masking                      |
| `simple`    | Simple truncation                  |

**Example: CC12M pre-training with frequency masking**

```bash
torchrun --nproc_per_node=4 -m training.main \
  --train-data "/data/cc12m/cc12m-train-{0000..2175}.tar" \
  --train-num-samples 10968539 \
  --imagenet-val "/data/imagenet/validation/" \
  --dataset-type webdataset \
  --model ViT-B-16 \
  --batch-size 896 \
  --aug-cfg scale="(0.50, 1.0)" \
  --force-patch-dropout 0.75 \
  --force-text-dropout 0.75 \
  --reduction-mask "frequency" \
  --mask-probability-file "../data/cc12m/cc12m_fq_1e6_words.json" \
  --lr 1e-3 \
  --wd 0.2 \
  --epochs 30 \
  --precision amp \
  --workers 4
```

---

## Fine-tuning

Fine-tune on the full dataset without image or text masking:

```bash
torchrun --nproc_per_node=4 -m training.main \
  --train-data "/data/cc12m/cc12m-train-{0000..2175}.tar" \
  --train-num-samples 10968539 \
  --imagenet-val "/data/imagenet/validation/" \
  --dataset-type webdataset \
  --model ViT-B-16 \
  --pretrained "/path/to/checkpoints/epoch_K.pt" \
  --batch-size 160 \
  --aug-cfg scale="(0.50, 1.0)" \
  --lr 1e-5 \
  --lr-warmup-epochs 0.1 \
  --epochs 1 \
  --precision amp \
  --workers 4
```

---

## Evaluation

We use [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) for standardized zero-shot evaluation.

---

## Acknowledgements

- [OpenCLIP](https://github.com/mlfoundations/open_clip) — base pre-training framework
- [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) — evaluation suite

---

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{liang2026clipf,
    author    = {Mingliang Liang and Martha Larson},
    title     = {Frequency Is What You Need: Considering Word Frequency When Text Masking Benefits Vision-Language Model Pre-training},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2026},
}
```
