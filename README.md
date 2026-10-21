# Multi-Objective Aligned Small Language Model Framework for SUD Patient Dialogue Generation

## Abstract

Substance Use Disorder (SUD) counseling requires patient responses that reflect underlying cognitive states such as beliefs, coping strategies, and readiness for change. Although large language models (LLMs) can generate fluent text, they often fail to produce cognitively coherent and clinically realistic patient behavior, especially under ethical and data-scarce clinical settings. Moreover, deploying frontier-scale LLMs in healthcare applications presents practical challenges including high computational cost, latency, privacy concerns, and limited deployability in resource-constrained environments, motivating the need for cognitively aligned small language models (SLMs). We propose a cognitively grounded framework for SUD patient dialogue generation that explicitly models and aligns latent cognitive components with patient histories and counselor questions. Our pipeline consists of two stages: cognitive component detection and cognitive component–aligned dialogue generation. To enable effective learning with smaller models, we combine knowledge distillation from high-capacity teacher models, preference optimization from human-annotations, and attention-guided reward shaping. Extensive evaluations using automatic scores like BERTScore, ROUGE, METEOR and BLEU, and LLM-as-judge hit-metrics against both human and teacher-model references show that cognitively informed fine-tuning substantially improves cognitive realization and alignment over a generic instruction-tuned baselines and mental health domain specific SLMs, with particularly strong gains for open-ended cognitive components.

## Framework

![Two-stage framework for cognitive component detection and cognitively aligned SUD patient dialogue generation](assets/moa-framework.png)

1. **Cognitive Component Detector (CCD):** extracts cognitive components from patient histories using supervised fine-tuning (SFT) and direct preference optimization (DPO).
2. **Cognitive Component-Aligned Dialogue Generator (CADG):** generates patient responses from the history, counselor question, and detected components using SFT, DPO, and group relative policy optimization (GRPO) with attention and length rewards.

The framework is evaluated using BERTScore, ROUGE, METEOR, BLEU, and LLM-as-judge alignment measures against human and teacher-model references.

## Code and documentation

- [CCD](CCD/README.md): detection scripts.
- [CADG](CADG/README.md): dialogue generation scripts.
- [DPO environment](dpo_env.yml) and [GRPO environment](grpo_env.yml): Conda dependency definitions.

Intended for LLM alignment research.
