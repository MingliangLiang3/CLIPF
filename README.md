# Frequency is what you need: Word-frequency Masking benefits Vision-Language Model Pre-training

# Abstract
Vision Language Models (VLMs) can be trained more efficiently if training sets can be reduced in size. Recent work has shown the benefits of masking text during VLM training using a variety of approaches: truncation, random masking, block masking and syntax masking. In this paper, we show that the best masking strategy changes over training epochs and that, given sufficient training epochs. We analyze existing text masking approaches including syntax masking, which is currently the state of the art, and identify the word frequency distribution as important in determining their success. Experiments on a large range of data sets demonstrate that syntax masking is outperformed by other approaches, given sufficient epochs, and that our proposed frequency-based approach, called Contrastive Language-Image Pre-training with Word Frequency Masking (CLIPF) has numerous advantages. The benefits are particularly evident as the number of input tokens decreases.
# Results and Pre-trained Models

We will pre-train the models based on the following code and settings [open_clip](https://github.com/mlfoundations/open_clip)

**Zero-shot classification accuracy on ImageNet-1K.**

| Models | Masking    | Image Tokens | Text Tokens | CC3M pre-train | CC3M fine-tune | CC12M pre-train | CC12M fine-tune |
|--------|------------|--------------|-------------|----------------|----------------|-----------------|-----------------|
| CLIP   | ✘          | 197          | 32          | 18.6           | ✘              | 36.6            | ✘               |
| FLIP   | ✘          | 49           | 32          | 14.1           | 14.2           | 32.0            | 33.7            |
| CLIPA  | truncation | 49           | 16          | 13.8           | 13.8           | 32.8            | 32.8            |
| CLIPA  | random     | 49           | 16          | 13.9           | 13.9           | 33.7            | 34.3            |
| CLIPA  | block      | 49           | 16          | 13.9           | 13.9           | 34.3            | 34.8            |
| CLIPA  | syntax     | 49           | 16          | 13.3           | 12.8           | 32.2            | 34.4            |
| CLIPF  | frequency  | 49           | 16          | **14.0**       | **14.0**       | **35.5**        | **36.0**        |
| CLIPA  | truncation | 49           | 8           | 10.8           | 12.0           | 25.4            | 28.4            |
| CLIPA  | random     | 49           | 8           | **17.6**       | 17.4           | 34.5            | 36.9            |
| CLIPA  | block      | 49           | 8           | 16.2           | 16.6           | 35.5            | 37.9            |
| CLIPA  | syntax     | 49           | 8           | 17.2           | **17.5**       | 28.5            | 35.0            |
| CLIPF  | frequency  | 49           | 8           | 16.8           | 17.0           | **36.6**        | **39.3**        |
| CLIPA  | truncation | 49           | 6           | 8.4            | 9.4            | 15.3            | 23.2            |
| CLIPA  | random     | 49           | 6           | 12.8           | 17.9           | 26.9            | 34.6            |
| CLIPA  | block      | 49           | 6           | 12.9           | 17.0           | 28.6            | 35.9            |
| CLIPA  | syntax     | 49           | 6           | 12.2           | 15.7           | 25.2            | 32.6            |
| CLIPF  | frequency  | 49           | 6           | **14.4**       | **18.2**       | **30.3**        | **37.8**        |
| CLIPA  | truncation | 49           | 4           | 3.8            | 8.2            | 5.3             | 19.8            |
| CLIPA  | random     | 49           | 4           | 5.4            | 14.6           | 14.0            | 27.1            |
| CLIPA  | block      | 49           | 4           | 7.5            | 14.5           | **18.7**        | 26.6            |
| CLIPA  | syntax     | 49           | 4           | 8.9            | 13.0           | 14.2            | 24.6            |
| CLIPF  | frequency  | 49           | 4           | **10.9**       | **16.0**       | 17.0            | **30.9**        |

**Comparison of CLIPF, CLIP, and CLIPA for zero-shot classification on ImageNet-1K.**
CLIP is the baseline model without image and text masking during pre-training. 
The main difference between CLIP and FLIP/CLIPA/CLIPF is whether or not image and text masking is applied during pre-training.
Specifically, FLIP does not use text masking, CLIPA employs various text masking strategies, and CLIPF uses word frequency text masking. 
All models use a ViT-B/16 image encoder backbone, pre-trained on **CC3M** and **CC12M** for 30 epochs with 75\% image masking, achieving approximately $4\times$ speedup compared to training without masking. 
The table reports the top-1 accuracy of zero-shot classification on ImageNet-1K before and after unmasking tuning.
Columns represent models, text masking methods, the number of image and text tokens, and top-1 accuracy for pre-training and fine-tuning on CC3M and CC12M datasets.


# Scaling to the larger Dataset.

We pre-trained the model for 6 epochs on the LAION400M dataset with a small image size (112x112) and 50% image masking ratios with ViT-B/16 as the image encoder. Then we fine-tuned the model on a normal image size of 224x224 without image and text masking for 0.4 epoch. For the LAION400M dataset, we successfully downloaded 297 million. We pre-trained and fine-tuned the models on 4 H100 GPUs with amp_bf16 precision. 

| Method | GPU Hours | Sample Seen | Image Size | Masking Ratio | Image Token | Text Token | Before Tuning | After Tuning |
|--------|-----------|-------------|------------|---------------|-------------|-----------|---------------|--------------|
| CLIPF  | 270       | 2.56B + 128M| 112 × 112  | 50%           | 25          | 16        | 57.5          | 61.6         |



# Word Frequency Count
Generate a word frequency dictionary using the script *tests/data_counter.py*. 

# Pre-training

Follow the instruction of [OpenCLIP](https://github.com/mlfoundations/open_clip) to pre-train the model.
Pre-training the models by different text masking strategies by "reduction-mask" parameters. The parameters are ('simple', 'random', 'shuffle', 'syntax', 'frequency')

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    -m training.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \ 
    --imagenet-val /data/imagenet/validation/
    --model ViT-B-16 \
    --batch-size 896 \
    --aug-cfg scale='(0.50, 1.0)' \
    --force-patch-dropout=0.75 \
    --force-text-dropout=0.75 \
    --reduction-mask="frequency" \
    --mask-probability-file="../data/cc12m/cc12m_fq_1e6_words.json" \
    --lr 1e-3 \
    --wd 0.2 \
    --epochs 30 \
    --precision amp \
    --workers 4 

```

# Fine-tuning
Fine-tuning the model on the entire dataset.

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    -m training.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --imagenet-val /data/imagenet/validation/ \
    --dataset-type webdataset \
    --model=ViT-B-16 \
    --pretrained /path/to/checkpoints/epoch_K.pt
    --batch-size 160 \
    --aug-cfg scale='(0.50, 1.0)' \
    --lr 1e-5 \
    --lr-warmup-epochs 0.1 \
    --epochs=1 \
    --precision amp \
    --workers 4 

```

# Evaluation

We use [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark/tree/main) to evaluate CLIP and WFPP on a standard set of datasets on different tasks.
