---
name: dissertation_gec
description: Error-Type-Aware, CEFR-Adaptive Feedback for L2 Writing — MSc Computational Linguistics dissertation project.
---

# dissertation_gec

MSc Computational Linguistics dissertation: **Error-Type-Aware, CEFR-Adaptive Feedback for L2 Writing**.

## Stack
- Python 3.11+
- PyTorch, Transformers, PEFT/LoRA
- ERRANT, spaCy, language-tool-python
- pandas, scikit-learn, matplotlib
- Streamlit (demo UI)

## Datasets
- W&I+LOCNESS v2.1 (BEA-2019)
- JFLEG

## Conventions
- Random seed: `42` everywhere (numpy, torch, random, transformers).
- All structured outputs written as **JSONL**.
- Academic text uses **Harvard** referencing style.
- Code is modular: **one file per concern**.
- Log everything for reproducibility (config, seeds, versions, metrics).

## Project goals
1. Ingest and normalise W&I+LOCNESS and JFLEG into unified JSONL.
2. Type errors with ERRANT and aggregate by CEFR level.
3. Fine-tune a correction model with LoRA adapters.
4. Adapt feedback to learner CEFR level (scaffolded vs. concise).
5. Evaluate with ERRANT P/R/F0.5 and GLEU (JFLEG).
6. Ship a Streamlit demo.

## Reproducibility
- Pin dependencies in `pyproject.toml` / `requirements.txt`.
- Log run configs and git commit hash per experiment.
- Deterministic seeding across all libraries.
