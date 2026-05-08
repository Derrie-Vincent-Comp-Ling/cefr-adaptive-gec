"""Run the LanguageTool baseline on dev-tune and save predictions.

Outputs:
    results/lt_dev_tune_preds.jsonl
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, SEED, ensure_dirs
from models.lt_wrapper import LTWrapper

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "run_lt_baseline.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ensure_dirs()
    log.info("seed=%d", SEED)

    # 1) Load dev-tune
    dev_tune_path = DATA_PROCESSED_DIR / "dev_tune.jsonl"
    records = load_jsonl(dev_tune_path)
    log.info("Loaded %d records from %s", len(records), dev_tune_path)

    sentences = [r["tok_form"] for r in records]

    # 2) Run LT baseline
    lt = LTWrapper(language="en-GB")
    log.info("Running LanguageTool on %d sentences …", len(sentences))
    t0 = time.time()
    results = lt.batch_correct(sentences, log_every=500)
    elapsed = time.time() - t0
    log.info("Done in %.1f s (%.1f sent/s)", elapsed, len(sentences) / elapsed)
    lt.close()

    # 3) Save predictions
    preds_path = RESULTS_DIR / "lt_dev_tune_preds.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for rec, res in zip(records, results):
            pred = {
                "id": rec["id"],
                "original": rec["tok_form"],
                "corrected": res["corrected"],
                "edit_ratio": res["edit_ratio"],
            }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    log.info("Predictions saved to %s", preds_path)

    # 4) Summary statistics
    n_changed = sum(
        1 for rec, res in zip(records, results)
        if rec["tok_form"] != res["corrected"]
    )
    mean_ratio = sum(r["edit_ratio"] for r in results) / len(results)

    print(f"\n=== LanguageTool Baseline Results ===")
    print(f"  Total sentences:   {len(sentences)}")
    print(f"  Sentences changed: {n_changed} ({100 * n_changed / len(sentences):.1f}%)")
    print(f"  Mean edit ratio:   {mean_ratio:.4f}")
    print(f"  Time elapsed:      {elapsed:.1f}s")
    print(f"  Predictions:       {preds_path}")


if __name__ == "__main__":
    main()
