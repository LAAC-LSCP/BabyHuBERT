# BabyHuBERT: Pre-training and Finetuning Examples

[![Paper](https://img.shields.io/badge/arXiv-2509.15001-B31B1B.svg)](https://arxiv.org/abs/2509.15001)

This repository provides sample implementations for the **pre-training** and **finetuning** pipelines of
**[BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings](https://arxiv.org/abs/2509.15001)**.

If you want to use the finetuned version of BabyHuBERT on the Voice Type Classification task please look at [VTC2.0](https://github.com/LAAC-LSCP/VTC)

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Pre-training Usage](#pre-training-usage)

   * [Dataset Preparation](#dataset-preparation)
   * [Compute Specification](#compute-specification)
   * [BabyHuBERT-1 (First Iteration)](#-babyhubert-1--first-iteration)
   * [BabyHuBERT-2 (Second Iteration)](#-babyhubert-2--second-iteration)
4. [Finetuning Usage](#finetuning-usage)

   * [Step 1: Configure Model](#step-1-configure-model)
   * [Step 2: Run Finetuning](#step-2-run-finetuning)
5. [Citation](#-citation)

---

## Overview

**BabyHuBERT** extends **HuBERT**’s self-supervised learning framework to child-centered multilingual long-form recordings.
It follows the same two-stage pre-training procedure as HuBERT, starting from **[WavLM-base-plus](https://arxiv.org/abs/2110.13900)** features, and is implemented using the [torchaudio HuBERT example](https://github.com/pytorch/audio/tree/main/examples/hubert).

---

## Requirements

Before running the pre-training or finetuning pipelines, install the dependencies below:

```bash
pip install uv

# Create and activate the pretraining environment
uv venv .venv-pretrain
source .venv-pretrain/bin/activate

# Install the pretraining dependencies
uv sync
```

For the finetuning environment:

```bash
git clone https://github.com/arxaqapi/segma.git
cd segma

# Create and activate the finetuning environment
uv venv .venv-finetuning
source .venv-finetuning/bin/activate

# Install the finetuning dependencies
uv sync
```

---

## Pre-training Usage

The **HuBERT** model architecture requires two iterations of pre-training.
**BabyHuBERT** follows this same two-stage process.

---

### Dataset Preparation

* [`preprocess_samples.py`](./preprocess_samples.py):
  Adjusts the distribution of sample durations by merging segments that overlap or are separated by less than 2 seconds.

* [`archive_samples.py`](./archive_samples.py):
  Generates training set archives, sharded into **32 archives** for distributed training.

---

### Compute Specification

All SLURM scripts follow the naming format:
`launch_*.sh`

#### Preprocessing Steps (`preprocess.py`)

1. **Generate Features** (`-gf`)
   → 32 separate jobs, each using **1×A100 GPU**.

2. **K-means Clustering** (`-lk`)
   → Single job requiring **1 TB+ RAM**.

3. **Generate Labels** (`-gl`)
   → 32 separate **CPU** jobs.

#### Training Setup

Training was conducted on **32×H100 GPUs**, distributed across **8 nodes (4 GPUs per node)**.

---

Use correct environment
```bash
source .venv-pretrain/bin/activate
```

---

## 🔹 BabyHuBERT-1 — First Iteration
### Preprocess

```bash
srun uv run preprocess.py -gf -lk -gl \
  --num-shards-kmeans 6 \
  --feat-type wavlm-base-plus \
  --layer-index 6 \
  --num-rank 32 \
  --num-cluster 500
```

### Train

```bash
srun uv run train.py \
  --dataset longforms \
  --dataset-path ./exp_iter/data/wavlm-base-plus_1_7 \
  --exp-dir ./exp_iter2_B175 \
  --feature-type hubert \
  --num-class 500 \
  --max-updates 400000 \
  --seconds-per-batch 175 \
  --learning-rate 0.0005 \
  --gpus 4 \
  --num-nodes 8
```

---

## 🔹 BabyHuBERT-2 — Second Iteration

### Preprocess

```bash
srun uv run preprocess.py -gf -lk -gl \
  --num-shards-kmeans 6 \
  --feat-type baby-hubert-175s \
  --layer-index 7 \
  --num-rank 32 \
  --num-cluster 500
```

### Train

```bash
srun uv run train.py \
  --dataset longforms \
  --dataset-path ./exp_iter2_B175/data/baby-hubert-175s_1_7 \
  --exp-dir ./exp_iter3_B175 \
  --feature-type hubert \
  --num-class 500 \
  --max-updates 400000 \
  --seconds-per-batch 175 \
  --learning-rate 0.0005 \
  --gpus 4 \
  --num-nodes 8
```

---

## Finetuning Usage

Finetuning is performed using the **[segma](https://github.com/arxaqapi/segma)** library.

---

Use correct environment
```bash
source .venv-finetuning/bin/activate
```

---

### Step 1: Configure Model

Modify the config file:
[`segma/src/segma/config/train_surgical_hubert_hydra.yml`](https://github.com/arxaqapi/segma/blob/main/src/segma/config/train_surgical_hubert_hydra.yml)

Choose the HuBERT model checkpoint to finetune:

```yaml
# HuBERT-base
wav_encoder: hubert_base

# BabyHuBERT-1
wav_encoder: "path/to/BabyHuBERT-1-checkpoint"

# BabyHuBERT-2
wav_encoder: "path/to/BabyHuBERT-2-checkpoint"
```

---

### Step 2: Run Finetuning

```bash
# Set environment variables
run_id="BabyHuBERT2VTC"
config_model="train_surgical_hubert_hydra.yml"
user_path="/path/to/checkpoint"
segma_path="/path/to/segma"

# Launch finetuning
srun uv run $segma_path/scripts/auto_train.py \
  --auto-resume \
  --all-weights \
  --run-id $run_id \
  --output $user_path/checkpoints/ \
  --config $user_path/checkpoints/$run_id/config.yml
```

---

## 📖 Citation

To cite this work, please use the following bibtex.

```bibtex
@misc{charlot2025babyhubertmultilingualselfsupervisedlearning,
    title={BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings}, 
    author={Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
    year={2025},
    eprint={2509.15001},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.15001}, 
}
```


---


