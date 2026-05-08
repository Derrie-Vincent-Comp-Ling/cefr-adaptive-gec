"""Build sentence-level JSONL records with CEFR bands from W&I+LOCNESS.

Loads raw JSON essays, splits into sentences with spaCy, maps CEFR labels
to bands (A/B/C/N), and writes one record per sentence.

Outputs:
    data/processed/train_records.jsonl
    data/processed/val_records.jsonl
    results/cefr_distribution.json
"""
from __future__ import annotations

import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    SEED,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "build_records.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

MIN_TOKENS = 3  # skip sentences with fewer tokens


# ---------------------------------------------------------------------------
# CEFR mapping
# ---------------------------------------------------------------------------
def map_cefr_band(raw_cefr: str) -> str:
    """Map fine-grained CEFR labels to coarse bands.

    A1, A2, A2.i … → 'A'
    B1, B1.ii, B2 … → 'B'
    C1, C2 …        → 'C'
    N                → 'N'  (native / LOCNESS)
    """
    raw = raw_cefr.strip().upper()
    if raw.startswith("A"):
        return "A"
    if raw.startswith("C"):
        return "C"
    if raw == "N":
        return "N"
    return "B"  # everything else (B1, B2, …)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_essays(json_path: Path) -> list[dict]:
    """Load a W&I+LOCNESS JSON file (one JSON object per line)."""
    essays = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                essays.append(json.loads(line))
    return essays


def build_sentence_records(
    essays: list[dict],
    nlp,
    split_name: str,
    file_cefr: str | None = None,
) -> list[dict]:
    """Split essays into sentences with spaCy and build records.

    Parameters
    ----------
    essays : list of dicts with keys 'text', 'id', 'cefr'
    nlp : spaCy Language model
    split_name : 'train' or 'val'
    file_cefr : override CEFR if the file is per-level (e.g. 'N' for N.dev)

    Returns
    -------
    list of sentence-level dicts
    """
    records = []
    for essay in essays:
        essay_id = essay.get("id", "unk")
        raw_cefr = file_cefr if file_cefr else essay.get("cefr", "B")
        cefr_band = map_cefr_band(raw_cefr)
        doc = nlp(essay["text"])
        for si, sent in enumerate(doc.sents):
            tokens = [tok.text for tok in sent if not tok.is_space]
            if len(tokens) < MIN_TOKENS:
                continue
            tok_form = " ".join(tokens)
            detok_form = sent.text.strip()
            records.append(
                {
                    "id": f"{split_name}_{essay_id}_{si:04d}",
                    "essay_id": essay_id,
                    "cefr_band": cefr_band,
                    "tok_form": tok_form,
                    "detok_form": detok_form,
                }
            )
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    log.info("seed=%d", SEED)

    nlp = spacy.load("en_core_web_sm")
    log.info("Loaded spaCy model: %s", nlp.meta["name"])

    json_dir = DATA_RAW_DIR / "wi_locness" / "wi+locness" / "json"

    # ---- TRAIN splits: A.train, B.train, C.train ----
    train_files = {
        "A.train.json": None,   # CEFR is inside each record
        "B.train.json": None,
        "C.train.json": None,
    }
    train_records: list[dict] = []
    for fname, override_cefr in train_files.items():
        path = json_dir / fname
        if not path.exists():
            log.warning("Missing %s — skipping", path)
            continue
        essays = load_essays(path)
        log.info("Loaded %d essays from %s", len(essays), fname)
        recs = build_sentence_records(essays, nlp, "train", override_cefr)
        train_records.extend(recs)
    log.info("Total train sentence records: %d", len(train_records))

    # ---- VAL splits: A.dev, B.dev, C.dev, N.dev ----
    val_files = {
        "A.dev.json": None,
        "B.dev.json": None,
        "C.dev.json": None,
        "N.dev.json": "N",  # native essays have no per-record CEFR label
    }
    val_records: list[dict] = []
    for fname, override_cefr in val_files.items():
        path = json_dir / fname
        if not path.exists():
            log.warning("Missing %s — skipping", path)
            continue
        essays = load_essays(path)
        log.info("Loaded %d essays from %s", len(essays), fname)
        recs = build_sentence_records(essays, nlp, "val", override_cefr)
        val_records.extend(recs)
    log.info("Total val sentence records: %d", len(val_records))

    # ---- Write JSONL ----
    write_jsonl(train_records, DATA_PROCESSED_DIR / "train_records.jsonl")
    write_jsonl(val_records, DATA_PROCESSED_DIR / "val_records.jsonl")

    # ---- CEFR distribution ----
    train_dist = dict(sorted(Counter(r["cefr_band"] for r in train_records).items()))
    val_dist = dict(sorted(Counter(r["cefr_band"] for r in val_records).items()))

    dist = {"seed": SEED, "train": train_dist, "val": val_dist}
    dist_path = RESULTS_DIR / "cefr_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(dist, f, indent=2)
    log.info("CEFR distribution saved to %s", dist_path)

    print("\n=== CEFR distribution (train) ===")
    for band, count in train_dist.items():
        print(f"  {band}: {count}")
    print(f"  TOTAL: {sum(train_dist.values())}")

    print("\n=== CEFR distribution (val) ===")
    for band, count in val_dist.items():
        print(f"  {band}: {count}")
    print(f"  TOTAL: {sum(val_dist.values())}")

    # ---- Show a few samples ----
    print("\n=== Sample records (train, first 3) ===")
    for r in train_records[:3]:
        print(json.dumps(r, indent=2))

    print("\n=== Sample records (val, first 3) ===")
    for r in val_records[:3]:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
