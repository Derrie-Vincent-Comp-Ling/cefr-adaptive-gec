"""Recompute hybrid-selector statistics from the prediction JSONL files.

Reads:
    results/hybrid_dev_tune_preds.jsonl
    results/hybrid_dev_eval_preds.jsonl

Writes:
    results/selector_stats.json

Run:
    cd ~/Desktop/dissertation_gec
    python3 app/recompute_selector_stats.py
"""
from __future__ import annotations

import json
import operator
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def analyse(fn: Path) -> dict:
    with fn.open() as f:
        recs = [json.loads(line) for line in f if line.strip()]

    total = len(recs)
    changed = sum(1 for r in recs if not operator.eq(r["original"], r["corrected"]))
    mean_edit_ratio = sum(r["edit_ratio"] for r in recs) / total if total else 0.0

    engine_sentences = Counter()
    total_edits = 0
    cefr = Counter()

    for r in recs:
        total_edits += r.get("n_edits_selected", 0)
        for engine in r.get("engines_used") or []:
            engine_sentences[engine] += 1
        cefr[r.get("cefr_band", "UNK")] += 1

    return {
        "total_sentences": total,
        "sentences_changed": changed,
        "sentences_changed_pct": round(100 * changed / total, 2) if total else 0.0,
        "mean_edit_ratio": round(mean_edit_ratio, 6),
        "total_edits_selected": total_edits,
        "engine_contributions_sentences": dict(engine_sentences),
        "engine_contributions_pct_of_partition": {
            e: round(100 * n / total, 2) for e, n in engine_sentences.items()
        },
        "engine_contributions_pct_of_changed": {
            e: round(100 * n / changed, 2) if changed else 0.0
            for e, n in engine_sentences.items()
        },
        "cefr_distribution": dict(cefr),
    }


def main() -> None:
    out = {
        "dev_tune": analyse(RESULTS / "hybrid_dev_tune_preds.jsonl"),
        "dev_eval": analyse(RESULTS / "hybrid_dev_eval_preds.jsonl"),
    }
    out_path = RESULTS / "selector_stats.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
