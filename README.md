## Frequency is what you need: Word-frequency Masking benefits Vision-Language Model Pre-training (WACV 2026)

This repository provides code/configs for **CLIPF**: *Contrastive Language-Image Pre-training with Word Frequency Masking*, a frequency-based text masking strategy for efficient VLM pre-training.

**Paper:** accepted to **WACV 2026**.

### Key idea
Text masking quality matters, and **the optimal masking strategy changes across training**. We show that **word frequency distribution** is a key factor behind prior masking methods (including syntax masking).  
Our **frequency-based masking (CLIPF)** is especially effective when **text tokens are limited**.

---

## Results

### Zero-shot ImageNet-1K (ViT-B/16, 75% image masking, 30 epochs on CC12M)
Below reports **top-1 zero-shot accuracy** on ImageNet-1K before/after unmasking tuning.

| Models | Masking    | Image Tokens | Text Tokens |    pre-train    |    fine-tune    |
|--------|------------|--------------|-------------|-----------------|-----------------|
| CLIP   | ✘          | 197          | 32          | 36.6            | ✘               |
| CLIPF  | frequency  | 98           | 8           | **39.8**        | **41.0**        |
| FLIP   | ✘          | 49           | 32          | 32.0            | 33.7            |
| CLIPA  | truncation | 49           | 16          | 32.8            | 32.8            |
| CLIPA  | random     | 49           | 16          | 33.7            | 34.3            |
| CLIPA  | block      | 49           | 16          | 34.3            | 34.8            |
| CLIPA  | syntax     | 49           | 16          | 32.2            | 34.4            |
| CLIPF  | frequency  | 49           | 16          | **35.5**        | **36.0**        |
| CLIPA  | truncation | 49           | 8           | 25.4            | 28.4            |
| CLIPA  | random     | 49           | 8           | 34.5            | 36.9            |
| CLIPA  | block      | 49           | 8           | 35.5            | 37.9            |
| CLIPA  | syntax     | 49           | 8           | 28.5            | 35.0            |
| CLIPF  | frequency  | 49           | 8           | **36.6**        | **39.3**        |
| CLIPA  | truncation | 49           | 6           | 15.3            | 23.2            |
| CLIPA  | random     | 49           | 6           | 26.9            | 34.6            |
| CLIPA  | block      | 49           | 6           | 28.6            | 35.9            |
| CLIPA  | syntax     | 49           | 6           | 25.2            | 32.6            |
| CLIPF  | frequency  | 49           | 6           | **30.3**        | **37.8**        |
| CLIPA  | truncation | 49           | 4           | 5.3             | 19.8            |
| CLIPA  | random     | 49           | 4           | 14.0            | 27.1            |
| CLIPA  | block      | 49           | 4           | **18.7**        | 26.6            |
| CLIPA  | syntax     | 49           | 4           | 14.2            | 24.6            |
| CLIPF  | frequency  | 49           | 4           | 17.0            | **30.9**        |

### Scaling to LAION400M
Pre-train **6 epochs** on LAION400M (112×112, 50% image masking), then fine-tune **0.4 epoch** at 224×224 without masking. Trained on **4×H100** with `amp_bf16`.

| Method | GPU Hours | Sample Seen | Image Size | Masking Ratio | Image Token | Text Token | Before Tuning | After Tuning |
|--------|-----------|-------------|------------|---------------|-------------|-----------|---------------|--------------|
| CLIPF  | 270       | 2.56B + 128M| 112 × 112  | 50%           | 25          | 16        | 57.5          | 61.6         |

---

## Getting Started

### 1) Installation
This repo builds on **OpenCLIP**:
- https://github.com/mlfoundations/open_clip

Follow OpenCLIP’s installation instructions first.

---

## Word Frequency Dictionary

You can **generate** the word-frequency dictionary using:
- `tests/data_counter.py`

Or **download precomputed files** here:
[mask-probability-file](https://drive.google.com/drive/folders/1AIoC7APPqg3H_J50_j_0ES41CvUQoKst?usp=sharing)

---

## Pre-training (OpenCLIP)

Supported `--reduction-mask` values:
- `simple` `random` `shuffle` `syntax` `frequency`

Example: CC12M pre-training with **frequency masking**:

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

Fine-tune on the full dataset without image/text masking:

```
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

## Evaluation

We use CLIP_benchmark for standardized evaluation: [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark/tree/main)

## Acknowledgements
	•	OpenCLIP: https://github.com/mlfoundations/open_clip
	•	CLIP_benchmark: https://github.com/LAION-AI/CLIP_benchmark
