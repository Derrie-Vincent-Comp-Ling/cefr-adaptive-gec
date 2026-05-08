"""ERRANT-based edit extraction and scoring utilities.

Provides:
    get_edits(original, corrected)       — extract edits between two sentences
    errant_f05(origs, preds, refs)       — compute corpus-level P/R/F0.5
    errant_score_by_type(origs, preds, refs) — per-error-type P/R/F0.5

Requires:
    pip install errant spacy
    python -m spacy download en_core_web_sm

Reference:
    Bryant, C., Felice, M. and Briscoe, T. (2017) 'Automatic annotation and
    evaluation of error types for grammatical error correction', ACL.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import errant
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEED

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton annotator (loading spaCy model is slow, do it once)
# ---------------------------------------------------------------------------
_ANNOTATOR = None


def _get_annotator():
    """Lazily initialise the ERRANT annotator."""
    global _ANNOTATOR
    if _ANNOTATOR is None:
        log.info("Loading ERRANT annotator (en_core_web_sm) …")
        _ANNOTATOR = errant.load("en")
        log.info("ERRANT annotator ready")
    return _ANNOTATOR


# ---------------------------------------------------------------------------
# Edit extraction
# ---------------------------------------------------------------------------
def get_edits(original: str, corrected: str) -> list[dict[str, Any]]:
    """Extract ERRANT edits between an original and corrected sentence.

    Returns a list of dicts with keys:
        o_start, o_end  — token span in the original
        o_str           — original token(s)
        c_str           — corrected token(s)
        type            — ERRANT error type (e.g. 'R:VERB:FORM', 'M:DET')
    """
    annotator = _get_annotator()
    orig_parse = annotator.parse(original)
    corr_parse = annotator.parse(corrected)
    alignment = annotator.annotate(orig_parse, corr_parse)

    edits = []
    for e in alignment:
        edits.append({
            "o_start": e.o_start,
            "o_end": e.o_end,
            "o_str": e.o_str,
            "c_str": e.c_str,
            "type": e.type,
        })
    return edits


# ---------------------------------------------------------------------------
# Helpers: convert edits to hashable tuples for set-based scoring
# ---------------------------------------------------------------------------
def _edits_to_set(edits: list[dict]) -> set[tuple]:
    """Convert a list of edit dicts to a set of hashable tuples for comparison."""
    return {
        (e["o_start"], e["o_end"], e["o_str"], e["c_str"], e["type"])
        for e in edits
        if e["type"] != "noop"
    }


def _edits_to_typed_sets(edits: list[dict]) -> dict[str, set[tuple]]:
    """Group edits by error type, each as a set of hashable tuples."""
    grouped: dict[str, set[tuple]] = defaultdict(set)
    for e in edits:
        if e["type"] == "noop":
            continue
        key = (e["o_start"], e["o_end"], e["o_str"], e["c_str"], e["type"])
        grouped[e["type"]].add(key)
    return grouped


# ---------------------------------------------------------------------------
# Corpus-level ERRANT F0.5
# ---------------------------------------------------------------------------
def errant_f05(
    originals: list[str],
    predictions: list[str],
    references: list[str],
) -> dict[str, Any]:
    """Compute corpus-level ERRANT precision, recall, and F0.5.

    Parameters
    ----------
    originals   : list of original (source) sentences
    predictions : list of system-corrected sentences
    references  : list of gold-corrected sentences

    Returns
    -------
    dict with keys: TP, FP, FN, precision, recall, f05
    """
    tp_total, fp_total, fn_total = 0, 0, 0

    for orig, pred, ref in zip(originals, predictions, references):
        pred_edits = _edits_to_set(get_edits(orig, pred))
        ref_edits = _edits_to_set(get_edits(orig, ref))

        tp = len(pred_edits & ref_edits)
        fp = len(pred_edits - ref_edits)
        fn = len(ref_edits - pred_edits)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    beta = 0.5
    f05 = (
        (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": tp_total,
        "FP": fp_total,
        "FN": fn_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f05": round(f05, 4),
    }


# ---------------------------------------------------------------------------
# Per-error-type ERRANT scoring
# ---------------------------------------------------------------------------
def errant_score_by_type(
    originals: list[str],
    predictions: list[str],
    references: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-error-type ERRANT P/R/F0.5.

    Returns a dict keyed by error type (e.g. 'R:VERB:FORM'), each with
    TP, FP, FN, precision, recall, f05.
    """
    type_tp: Counter = Counter()
    type_fp: Counter = Counter()
    type_fn: Counter = Counter()

    all_types: set[str] = set()

    for orig, pred, ref in zip(originals, predictions, references):
        pred_typed = _edits_to_typed_sets(get_edits(orig, pred))
        ref_typed = _edits_to_typed_sets(get_edits(orig, ref))

        all_types.update(pred_typed.keys())
        all_types.update(ref_typed.keys())

        for etype in all_types:
            p_set = pred_typed.get(etype, set())
            r_set = ref_typed.get(etype, set())
            type_tp[etype] += len(p_set & r_set)
            type_fp[etype] += len(p_set - r_set)
            type_fn[etype] += len(r_set - p_set)

    results = {}
    beta = 0.5
    for etype in sorted(all_types):
        tp = type_tp[etype]
        fp = type_fp[etype]
        fn = type_fn[etype]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (
            (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
            if (p + r) > 0
            else 0.0
        )
        results[etype] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f05": round(f, 4),
        }

    return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
def main() -> None:
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Three test pairs: (original, corrected/predicted, gold reference)
    test_cases = [
        {
            "original":  "I has a apple .",
            "predicted": "I have an apple .",
            "reference": "I have an apple .",
        },
        {
            "original":  "She go to school yesterday .",
            "predicted": "She went to school yesterday .",
            "reference": "She went to school yesterday .",
        },
        {
            "original":  "The informations is incorrect .",
            "predicted": "The information is incorrect .",
            "reference": "The information is incorrect .",
        },
    ]

    origs = [t["original"] for t in test_cases]
    preds = [t["predicted"] for t in test_cases]
    refs = [t["reference"] for t in test_cases]

    # 1) Show edits for each pair
    print("=== Edit extraction ===")
    for i, t in enumerate(test_cases):
        edits = get_edits(t["original"], t["predicted"])
        print(f"\nPair {i + 1}: '{t['original']}' → '{t['predicted']}'")
        for e in edits:
            print(f"  [{e['o_start']}:{e['o_end']}] '{e['o_str']}' → '{e['c_str']}' ({e['type']})")

    # 2) Corpus-level F0.5
    print("\n=== Corpus-level ERRANT F0.5 ===")
    scores = errant_f05(origs, preds, refs)
    print(json.dumps(scores, indent=2))

    # 3) Per-error-type
    print("\n=== Per-error-type ERRANT F0.5 ===")
    by_type = errant_score_by_type(origs, preds, refs)
    for etype, s in by_type.items():
        print(f"  {etype:20s}  P={s['precision']:.2f}  R={s['recall']:.2f}  F0.5={s['f05']:.2f}  (TP={s['TP']} FP={s['FP']} FN={s['FN']})")


if __name__ == "__main__":
    main()
