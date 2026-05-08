# Error-Type-Aware, CEFR-Adaptive Grammatical Error Correction for L2 Writing

This repository contains the code, configurations and trained adapter weights accompanying the master's dissertation *Error-Type-Aware, CEFR-Adaptive Grammatical Error Correction for L2 Writing* (Derrie-Ann Vincent, 2026).

The system extends a Flan-T5 base model with a low-rank adaptation (LoRA) adapter trained on the W&I+LOCNESS v2.1 corpus, conditions correction behaviour on CEFR proficiency level (A, B, C and a native control band N), and supports both direct correction and indirect metalinguistic feedback. Evaluation is performed with ERRANT against the W&I+LOCNESS development split and with GLEU against the JFLEG fluency benchmark.

## Repository Structure

```
.
├── app/                     # Streamlit demo application
├── annotate/                # ERRANT annotation utilities
├── data/                    # Preprocessing scripts (raw/processed data excluded)
│   ├── summarise_datasets.py
│   ├── normalise_to_jsonl.py
│   ├── build_records.py
│   ├── split_dev.py
│   └── extract_pairs.py
├── eval/                    # Evaluation scripts
├── experiments/             # Training and selection scripts
├── feedback/                # Indirect-feedback generation modules
├── models/                  # LoRA adapter and tagger wrappers
│   └── lora_flan_t5_base/   # Trained adapter weights
├── selector/                # Dev-tune CEFR-aware selector
├── writing/                 # Dissertation source (Markdown)
├── REPRODUCING_RESULTS.md   # End-to-end reproduction guide
├── requirements.txt
├── config.py
└── README.md
```

## Quick Start

```bash
git clone https://github.com/Derrie-Vincent-Comp-Ling/cefr-adaptive-gec.git
cd cefr-adaptive-gec
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For the full reproduction pipeline (preprocessing, training, evaluation, figures), see [`REPRODUCING_RESULTS.md`](REPRODUCING_RESULTS.md).

## Data

The W&I+LOCNESS v2.1 and JFLEG corpora are **not** redistributed in this repository because of their respective licensing terms. To run the preprocessing pipeline you will need to download both corpora from their original sources and place them under `data/raw/`:

- **W&I+LOCNESS v2.1**: download from the [BEA-2019 shared task page](https://www.cl.cam.ac.uk/research/nl/bea2019st/) and place the per-level M2 files under `data/raw/wi_locness/`.
- **JFLEG**: download from the [JFLEG release page](https://github.com/keisks/jfleg) and place the source and reference files under `data/raw/jfleg/`.

Once both corpora are in place, run the five preprocessing scripts in `data/` in order, as described in Section 3.4 of the dissertation, to produce the processed JSONL artifacts.

## Citing

If you use this code or the trained adapter weights in academic work, please cite the dissertation:

> Vincent, D.-A. (2026) *Error-Type-Aware, CEFR-Adaptive Grammatical Error Correction for L2 Writing*. Master's dissertation.

## License

The code in this repository is released under the [MIT License](LICENSE). The trained LoRA adapter weights under `models/lora_flan_t5_base/` are released under the same terms. The underlying base model (Flan-T5) and the W&I+LOCNESS and JFLEG corpora are subject to their own respective licenses; please consult those projects directly for terms of use.
