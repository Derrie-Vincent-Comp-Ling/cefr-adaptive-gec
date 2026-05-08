"""Evaluate hybrid (and individual engine) predictions with ERRANT F0.5.

Loads gold references from dev_pairs.jsonl, matches to each engine's
predictions by source text, and computes corpus-level and per-type
ERRANT P/R/F0.5.

Produces:
    results/eval_summary.json   — corpus-level metrics per engine
    results/eval_by_type.json   — per-error-type metrics per engine
    Console comparison table

Reference:
    Bryant, C., Felice, M. and Briscoe, T. (2017) 'Automatic annotation and
    evaluation of error types for grammatical error correction', ACL.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, SEED, ensure_dirs
from annotate.errant_pipe import errant_f05, errant_score_by_type

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "evaluate_hybrid.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ensure_dirs()
    log.info("Evaluation — seed=%d", SEED)

    # 1) Load gold references
    dev_pairs = load_jsonl(DATA_PROCESSED_DIR / "dev_pairs.jsonl")
    src_to_target: dict[str, str] = {}
    for r in dev_pairs:
        src_to_target[r["source"]] = r["target"]
    log.info("Loaded %d gold pairs", len(src_to_target))

    # 2) Define all engines to evaluate (including hybrid)
    engines = {
        "languagetool":      RESULTS_DIR / "lt_dev_tune_preds.jsonl",
        "t5-small-gec":      RESULTS_DIR / "tagger_dev_tune_preds.jsonl",
        "lora-flan-t5-base": RESULTS_DIR / "lora_dev_tune_preds.jsonl",
        "hybrid":            RESULTS_DIR / "hybrid_dev_tune_preds.jsonl",
    }

    corpus_results: dict[str, dict] = {}
    type_results: dict[str, dict] = {}

    for engine_name, preds_path in engines.items():
        if not preds_path.exists():
            log.warning("Skipping %s — %s not found", engine_name, preds_path)
            continue

        preds = load_jsonl(preds_path)
        log.info("Loaded %d predictions for %s", len(preds), engine_name)

        # Match to gold
        originals, predictions, references = [], [], []
        matched = 0
        for p in preds:
            orig = p["original"]
            if orig in src_to_target:
                originals.append(orig)
                predictions.append(p["corrected"])
                references.append(src_to_target[orig])
                matched += 1

        log.info("  %s: matched %d / %d to gold", engine_name, matched, len(preds))

        if matched == 0:
            continue

        # Corpus-level F0.5
        log.info("  Computing ERRANT F0.5 for %s (%d pairs) …", engine_name, matched)
        scores = errant_f05(originals, predictions, references)
        corpus_results[engine_name] = scores
        log.info("  %s: P=%.4f R=%.4f F0.5=%.4f", engine_name,
                 scores["precision"], scores["recall"], scores["f05"])

        # Per-type scoring
        log.info("  Computing per-type scores for %s …", engine_name)
        by_type = errant_score_by_type(originals, predictions, references)
        type_results[engine_name] = by_type

    # 3) Save results
    summary_path = RESULTS_DIR / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(corpus_results, f, indent=2)
    log.info("Corpus-level results saved to %s", summary_path)

    by_type_path = RESULTS_DIR / "eval_by_type.json"
    with open(by_type_path, "w") as f:
        json.dump(type_results, f, indent=2)
    log.info("Per-type results saved to %s", by_type_path)

    # 4) Print comparison table
    print("\n" + "=" * 72)
    print("  ERRANT Corpus-Level Evaluation (dev_tune)")
    print("=" * 72)
    header = f"  {'Engine':<22s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>8s} {'R':>8s} {'F0.5':>8s}"
    print(header)
    print("  " + "-" * 68)
    for eng, s in corpus_results.items():
        print(f"  {eng:<22s} {s['TP']:>6d} {s['FP']:>6d} {s['FN']:>6d} "
              f"{s['precision']:>8.4f} {s['recall']:>8.4f} {s['f05']:>8.4f}")
    print("=" * 72)

    # Highlight improvement
    if "hybrid" in corpus_results:
        h = corpus_results["hybrid"]
        others = {k: v for k, v in corpus_results.items() if k != "hybrid"}
        if others:
            best_other_name = max(others, key=lambda k: others[k]["f05"])
            best_other = others[best_other_name]
            delta = h["f05"] - best_other["f05"]
            print(f"\n  Hybrid F0.5: {h['f05']:.4f}")
            print(f"  Best single engine ({best_other_name}): {best_other['f05']:.4f}")
            print(f"  Delta: {delta:+.4f} ({delta/best_other['f05']*100:+.1f}%)" if best_other['f05'] > 0 else f"  Delta: {delta:+.4f}")

    # 5) Print top error types for hybrid vs best single engine
    if "hybrid" in type_results:
        print(f"\n{'=' * 72}")
        print("  Per-Error-Type F0.5 — Hybrid vs Best Single Engine")
        print(f"{'=' * 72}")

        hybrid_types = type_results["hybrid"]
        # Find top 15 types by hybrid F0.5
        sorted_types = sorted(hybrid_types.items(), key=lambda x: -x[1]["f05"])

        header = f"  {'Error Type':<20s} {'Hybrid':>10s}"
        for eng in [e for e in corpus_results if e != "hybrid"]:
            header += f" {eng:>14s}"
        print(header)
        print("  " + "-" * (20 + 10 + 14 * len([e for e in corpus_results if e != "hybrid"])))

        for etype, hscores in sorted_types[:20]:
            row = f"  {etype:<20s} {hscores['f05']:>10.4f}"
            for eng in [e for e in corpus_results if e != "hybrid"]:
                eng_types = type_results.get(eng, {})
                eng_s = eng_types.get(etype, {})
                eng_f = eng_s.get("f05", 0.0)
                row += f" {eng_f:>14.4f}"
            print(row)

        if len(sorted_types) > 20:
            print(f"  … ({len(sorted_types)} types total)")


if __name__ == "__main__":
    main()
