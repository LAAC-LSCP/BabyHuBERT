---
icon: lucide/home
---

# BabyHuBERT

BabyHuBERT is an audio encoder model trained by reconstructing multilingual child-centered long-form audio recordings. BabyHuBERT as is yields alternate representations of audio recordings. The purpose of this model is to be used as a starting point for training models tailored to specific tasks using longform recordings. Refer to [Derived models](#derived-models) for BabyHuBERT models trained on a specific task.

BabyHuBERT is an audio encoder model trained by reconstructing multilingual child-centered long-form audio recordings. From the raw audios of longform recordings, the model extracts richer representations than usual models such as whisper, trained on clean adult well spelled speech that does not include child speech. BabyHuBERT is a generalistic model, using it as is will only give you alternative representations of an audio. Refer to [Derived models](#derived-models) for models trained on a specific task.

## What BabyHuBERT is *not*

- **Not the voice type classifier** — it is a generalistic model that extracts richer representations from longform recordings. To extract voice type segments refer to [VTC2.0](../index.md).

## What BabyHuBERT *is*

!!! tip "tdlr"
    BabyHuBERT is a model trained on 13 000 hours of **adult** and **child** speech from child-centered long-form audio recordings across 40 languages using the HuBERT training recipe and it's base architecture.

### Description

BabyHuBERT is a speech representation model designed for analyzing child-centered daylong recordings—audio collected from children’s everyday environments, which typically include background noise, overlapping speakers, and highly variable child speech. Unlike most existing models trained on clean adult speech, BabyHuBERT is trained directly on more than 13,000 hours of such naturalistic recordings across over 40 languages. This allows it to better capture the acoustic characteristics of real-world language input to children. The model learns general patterns in the audio without relying on manual annotations, and can then be adapted to specific research tasks. One key application is voice type classification, where recordings are automatically segmented into categories such as the target child, other children, and adult speakers. When fine-tuned for this task, BabyHuBERT outperforms previous approaches and reaches performance close to that of human annotators, while also working well across a wide range of languages, including those that are typically underrepresented in speech technology.

### Advanced description

BabyHuBERT is a self-supervised speech representation model tailored to the acoustic and linguistic properties of child-centered daylong recordings. It is pre-trained on ~13k hours of multilingual audio (40+ languages) using a HuBERT-style masked prediction objective, with adaptations for long-form, noisy input. Because raw recordings contain ~80% non-speech, the pipeline applies voice activity detection to extract speech-centered segments, extends short segments with surrounding context, and concatenates nearby segments (capped at ~30s), reducing non-speech exposure to ~8% while preserving realistic background conditions. Pre-training follows a two-iteration clustering procedure: (1) BabyHuBERT-1 uses k-means (k=500) targets derived from intermediate representations of the WavLM-base-plus model; (2) BabyHuBERT-2 re-computes targets from an internal transformer layer of BabyHuBERT-1, refining discrete units as in standard HuBERT. Training uses the HuBERT-base architecture with masked span prediction over cluster assignments, large-batch training (~400k steps on multi-GPU), and emphasizes robustness to noise and speaker variability.

For downstream voice type classification (VTC), the model is fine-tuned on the BabyTrain-2025 dataset using a multi-label setup. A lightweight classification head with four independent sigmoid outputs (KCHI, OCH, MAL, FEM) is added on top of the final transformer layer, enabling overlapping speaker predictions. Fine-tuning updates the transformer encoder while keeping the convolutional feature extractor frozen, and is trained with short fixed-length segments (~4s), dropout regularization, and low learning rates (~1e-5). Performance is evaluated using F1-score across classes. Empirically, full fine-tuning of the encoder is critical, frozen encoder finetuning u nderperforms due to the need for fine-grained acoustic discrimination in noisy, multi-speaker environments. Compared to prior models (adult-trained HuBERT and English-only wav2vec 2.0 variants), BabyHuBERT benefits from both scale and multilingual diversity in pre-training, yielding strong gains across datasets and narrowing the gap to human annotation performance.

## Ethics statement

!!! warning "Ethics surrounding BabyHuBERT data"
To know more about the BabyHuBERT ethics please refer to the [one pager](one_pager.md). By accepting the [License](license.md) you accept to join the darcle list if you derive a model from BabyHuBERT, refer to the [flowchart](flowchart.md) for more information about the process.

## How to access model

1. Read the [License](license.md)
2. Go on the BabyHuBERT repo : [coml/BabyHuBERT](https://huggingface.co/coml/BabyHuBERT)
3. Fill the required fields and accept the license
4. Download the model

## Derived models

| Model | Task |
| ------ | ------ |
| [VTC2.0](https://arxaqapi.github.io/LAAC-LSCP-documentation/) | Voice Type Classification |
| [Addressee](https://labicquette.github.io/addressee-documentation/) | Addressee classification |
| [BabAR](https://marvinlvn.github.io/projects/babar/) | Phoneme recognition |

## How to cite

```bibtex
@misc{charlot2026babyhubertmultilingualselfsupervisedlearning,
    title={BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings}, 
    author={Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
    year={2026},
    eprint={2509.15001},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.15001}, 
}
```
