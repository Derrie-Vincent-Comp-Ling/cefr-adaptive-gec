"""Compute per-engine, per-error-type precision priors on dev-tune.

Loads gold references from dev_pairs.jsonl, matches them to each engine's
predictions by source text, then uses ERRANT to compute precision per error type.

Engines:
    - languagetool      (results/lt_dev_tune_preds.jsonl)
    - t5-small-gec      (results/tagger_dev_tune_preds.jsonl)
    - lora-flan-t5-base (results/lora_dev_tune_preds.jsonl)

Outputs:
    results/engine_priors.json
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, SEED, ensure_dirs
from annotate.errant_pipe import get_edits

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "compute_priors.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _edits_to_typed_sets(edits: list[dict]) -> dict[str, set[tuple]]:
    """Group edits by type as sets of hashable tuples."""
    grouped: dict[str, set[tuple]] = defaultdict(set)
    for e in edits:
        if e["type"] == "noop":
            continue
        key = (e["o_start"], e["o_end"], e["o_str"], e["c_str"], e["type"])
        grouped[e["type"]].add(key)
    return grouped


def compute_precision_by_type(
    originals: list[str],
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute per-error-type precision for a single engine.

    Returns {error_type: precision_float}.
    """
    type_tp: Counter = Counter()
    type_fp: Counter = Counter()

    for orig, pred, ref in zip(originals, predictions, references):
        pred_typed = _edits_to_typed_sets(get_edits(orig, pred))
        ref_typed = _edits_to_typed_sets(get_edits(orig, ref))

        all_pred_types = set(pred_typed.keys())
        for etype in all_pred_types:
            p_set = pred_typed.get(etype, set())
            r_set = ref_typed.get(etype, set())
            type_tp[etype] += len(p_set & r_set)
            type_fp[etype] += len(p_set - r_set)

    precisions = {}
    for etype in sorted(set(type_tp.keys()) | set(type_fp.keys())):
        tp = type_tp[etype]
        fp = type_fp[etype]
        precisions[etype] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0

    return precisions


def main() -> None:
    ensure_dirs()
    log.info("seed=%d", SEED)

    # 1) Load gold references (source → target mapping)
    dev_pairs = load_jsonl(DATA_PROCESSED_DIR / "dev_pairs.jsonl")
    src_to_target: dict[str, str] = {}
    src_to_cefr: dict[str, str] = {}
    for r in dev_pairs:
        src_to_target[r["source"]] = r["target"]
        src_to_cefr[r["source"]] = r.get("cefr_band", "UNK")
    log.info("Loaded %d gold pairs from dev_pairs.jsonl", len(src_to_target))

    # 2) Define engines and their prediction files
    engines = {
        "languagetool": RESULTS_DIR / "lt_dev_tune_preds.jsonl",
        "t5-small-gec": RESULTS_DIR / "tagger_dev_tune_preds.jsonl",
        "lora-flan-t5-base": RESULTS_DIR / "lora_dev_tune_preds.jsonl",
    }

    priors: dict[str, dict[str, float]] = {}

    for engine_name, preds_path in engines.items():
        if not preds_path.exists():
            log.warning("Predictions not found for %s at %s — skipping", engine_name, preds_path)
            continue

        preds = load_jsonl(preds_path)
        log.info("Loaded %d predictions for %s", len(preds), engine_name)

        # Match predictions to gold targets by original text
        originals, predictions_text, references_text = [], [], []
        matched, unmatched = 0, 0

        for p in preds:
            orig = p["original"]
            if orig in src_to_target:
                originals.append(orig)
                predictions_text.append(p["corrected"])
                references_text.append(src_to_target[orig])
                matched += 1
            else:
                unmatched += 1

        log.info("  %s: matched=%d, unmatched=%d", engine_name, matched, unmatched)

        if matched == 0:
            log.warning("  No matches for %s — skipping", engine_name)
            continue

        # Compute precision by error type
        log.info("  Computing ERRANT precision by type for %s (%d pairs) …", engine_name, matched)
        engine_priors = compute_precision_by_type(originals, predictions_text, references_text)
        priors[engine_name] = engine_priors
        log.info("  %s: %d error types found", engine_name, len(engine_priors))

    # 3) Save priors
    priors_path = RESULTS_DIR / "engine_priors.json"
    with open(priors_path, "w") as f:
        json.dump(priors, f, indent=2)
    log.info("Priors saved to %s", priors_path)

    # 4) Print summary table: top 10 error types per engine by precision
    for engine_name, type_prec in priors.items():
        sorted_types = sorted(type_prec.items(), key=lambda x: -x[1])
        print(f"\n=== {engine_name}: Top 10 error types by precision ===")
        print(f"  {'Error Type':<25s} {'Precision':>10s}")
        print(f"  {'-' * 25} {'-' * 10}")
        for etype, prec in sorted_types[:10]:
            print(f"  {etype:<25s} {prec:>10.4f}")
        print(f"  … ({len(type_prec)} types total)")

    # Also print a cross-engine comparison for common error types
    all_types = set()
    for tp in priors.values():
        all_types.update(tp.keys())
    common_types = [t for t in all_types if all(t in priors[e] for e in priors)]
    common_types.sort()

    if common_types:
        print(f"\n=== Cross-engine precision for {len(common_types)} shared error types ===")
        header = f"  {'Error Type':<25s}"
        for e in priors:
            header += f" {e:>18s}"
        print(header)
        print(f"  {'-' * 25}" + f" {'-' * 18}" * len(priors))
        for etype in common_types[:15]:
            row = f"  {etype:<25s}"
            for e in priors:
                row += f" {priors[e].get(etype, 0.0):>18.4f}"
            print(row)
        if len(common_types) > 15:
            print(f"  … ({len(common_types)} shared types total)")


if __name__ == "__main__":
    main()
