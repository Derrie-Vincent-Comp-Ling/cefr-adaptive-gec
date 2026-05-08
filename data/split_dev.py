"""Split val_records.jsonl into dev-tune (80%) and dev-eval (20%) at essay level.

Uses GroupShuffleSplit so all sentences from one essay stay in the same split.
Stratified by essay_id with random_state=42.

Outputs:
    data/processed/dev_tune.jsonl
    data/processed/dev_eval.jsonl
    logs/split_ids.json
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED_DIR, LOGS_DIR, SEED, ensure_dirs

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "split_dev.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def print_dist(records: list[dict], label: str) -> dict:
    dist = dict(sorted(Counter(r["cefr_band"] for r in records).items()))
    print(f"\n=== CEFR distribution ({label}) ===")
    for band, count in dist.items():
        print(f"  {band}: {count}")
    print(f"  TOTAL: {sum(dist.values())}")
    return dist


def main() -> None:
    ensure_dirs()
    log.info("seed=%d", SEED)

    # 1) Load val records
    val_path = DATA_PROCESSED_DIR / "val_records.jsonl"
    records = load_jsonl(val_path)
    log.info("Loaded %d records from %s", len(records), val_path)

    # Extract essay_id groups
    essay_ids = [r["essay_id"] for r in records]

    # 2) GroupShuffleSplit: 80% dev-tune, 20% dev-eval
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tune_idx, eval_idx = next(gss.split(records, groups=essay_ids))

    tune_records = [records[i] for i in tune_idx]
    eval_records = [records[i] for i in eval_idx]

    log.info("dev-tune: %d sentences, dev-eval: %d sentences",
             len(tune_records), len(eval_records))

    # 3) Save JSONL
    write_jsonl(tune_records, DATA_PROCESSED_DIR / "dev_tune.jsonl")
    write_jsonl(eval_records, DATA_PROCESSED_DIR / "dev_eval.jsonl")

    # 4) Save essay IDs for reproducibility
    tune_essays = sorted(set(r["essay_id"] for r in tune_records))
    eval_essays = sorted(set(r["essay_id"] for r in eval_records))
    split_ids = {
        "seed": SEED,
        "dev_tune_essays": tune_essays,
        "dev_eval_essays": eval_essays,
        "dev_tune_essay_count": len(tune_essays),
        "dev_eval_essay_count": len(eval_essays),
        "dev_tune_sentence_count": len(tune_records),
        "dev_eval_sentence_count": len(eval_records),
    }
    ids_path = LOGS_DIR / "split_ids.json"
    with open(ids_path, "w") as f:
        json.dump(split_ids, f, indent=2)
    log.info("Essay IDs saved to %s", ids_path)

    # Verify no essay overlap
    overlap = set(tune_essays) & set(eval_essays)
    if overlap:
        log.error("OVERLAP detected: %d essays in both splits!", len(overlap))
    else:
        log.info("No essay overlap between splits (verified)")

    # 5) Print counts and distributions
    print(f"\ndev-tune: {len(tune_records)} sentences from {len(tune_essays)} essays")
    print(f"dev-eval: {len(eval_records)} sentences from {len(eval_essays)} essays")

    tune_dist = print_dist(tune_records, "dev-tune")
    eval_dist = print_dist(eval_records, "dev-eval")

    # Also save distributions
    split_ids["dev_tune_cefr"] = tune_dist
    split_ids["dev_eval_cefr"] = eval_dist
    with open(ids_path, "w") as f:
        json.dump(split_ids, f, indent=2)


if __name__ == "__main__":
    main()
