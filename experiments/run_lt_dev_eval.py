"""Run LanguageTool baseline on the held-out dev_eval split.

Mirrors run_lt_baseline.py but targets dev_eval.jsonl instead of dev_tune.

Outputs:
    results/lt_dev_eval_preds.jsonl
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "lt_dev_eval.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    ensure_dirs()
    log.info("seed=%d", SEED)

    # Load dev_eval records
    dev_eval_path = DATA_PROCESSED_DIR / "dev_eval.jsonl"
    with open(dev_eval_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d records from %s", len(records), dev_eval_path)

    sentences = [r.get("source", r.get("tok_form", "")) for r in records]
    ids = [rec.get("id", f"eval_{i}") for i, rec in enumerate(records)]
    cefrs = [rec.get("cefr_band", "UNK") for rec in records]

    # Run LT
    lt = LTWrapper(language="en-US")
    t0 = time.time()
    results = lt.batch_correct(sentences, log_every=200)
    elapsed = time.time() - t0
    lt.close()

    log.info("Done in %.1f s (%.1f sent/s)", elapsed, len(sentences) / elapsed)

    # Save
    out_path = RESULTS_DIR / "lt_dev_eval_preds.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec_id, cefr, sent, res in zip(ids, cefrs, sentences, results):
            pred = {
                "id": rec_id,
                "original": sent,
                "corrected": res["corrected"],
                "edit_ratio": res["edit_ratio"],
                "engine": res["engine"],
                "cefr_band": cefr,
            }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    log.info("Predictions saved to %s", out_path)

    # Summary
    n_changed = sum(1 for s, r in zip(sentences, results) if s != r["corrected"])
    mean_ratio = sum(r["edit_ratio"] for r in results) / len(results)
    print(f"\n=== LT Baseline Results (dev_eval) ===")
    print(f"  Total sentences:   {len(results)}")
    print(f"  Sentences changed: {n_changed} ({100 * n_changed / len(results):.1f}%)")
    print(f"  Mean edit ratio:   {mean_ratio:.4f}")
    print(f"  Time elapsed:      {elapsed:.1f}s")
    print(f"  Predictions:       {out_path}")


if __name__ == "__main__":
    main()
