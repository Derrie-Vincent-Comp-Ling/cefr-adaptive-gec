"""Hybrid engine selector: edit-level fusion using per-type precision priors.

For each sentence, extracts ERRANT edits from all three engines' corrections,
scores each edit by the originating engine's precision prior for that error
type, then greedily selects non-overlapping edits in descending score order.
Selected edits are applied to the original tokenisation to produce a hybrid
correction.

Algorithm:
    1. Parse predictions from all engines, indexed by original sentence.
    2. For each sentence, run ERRANT on (original, engine_correction) for
       each engine to get typed edits.
    3. Assign each edit a score = engine_prior[error_type] (default 0.0).
    4. Pool all candidate edits, sort by score descending.
    5. Greedily accept edits that do not overlap in original-token span
       with any already-accepted edit.
    6. Apply accepted edits (right-to-left) to produce the hybrid output.

Engines:
    - languagetool      (results/lt_dev_tune_preds.jsonl)
    - t5-small-gec      (results/tagger_dev_tune_preds.jsonl)
    - lora-flan-t5-base (results/lora_dev_tune_preds.jsonl)

Inputs:
    results/engine_priors.json
    results/*_dev_tune_preds.jsonl   (or dev_eval)

Outputs:
    results/hybrid_dev_tune_preds.jsonl
    results/hybrid_dev_eval_preds.jsonl  (if eval predictions exist)

Reference:
    Bryant, C., Felice, M. and Briscoe, T. (2017) 'Automatic annotation and
    evaluation of error types for grammatical error correction', ACL.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from Levenshtein import ratio as lev_ratio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, LOGS_DIR, SEED, ensure_dirs
from annotate.errant_pipe import get_edits

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "hybrid_selector.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# Minimum precision threshold: ignore edits from engines with prior below this
MIN_PRECISION = 0.10


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Core: edit-level hybrid selection
# ---------------------------------------------------------------------------
def _extract_candidate_edits(
    original: str,
    engine_corrections: dict[str, str],
    priors: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Extract scored candidate edits from all engines for one sentence.

    Returns a list of edit dicts augmented with 'engine' and 'score' keys,
    sorted by score descending.
    """
    candidates: list[dict[str, Any]] = []

    for engine_name, corrected in engine_corrections.items():
        if corrected == original:
            continue  # engine made no changes

        edits = get_edits(original, corrected)
        engine_prior = priors.get(engine_name, {})

        for e in edits:
            if e["type"] == "noop":
                continue
            score = engine_prior.get(e["type"], 0.0)
            candidates.append({
                **e,
                "engine": engine_name,
                "score": score,
            })

    # Sort by score descending, then by span length ascending (prefer smaller)
    candidates.sort(key=lambda c: (-c["score"], c["o_end"] - c["o_start"]))
    return candidates


def _select_non_overlapping(candidates: list[dict]) -> list[dict]:
    """Greedily select non-overlapping edits by descending score.

    Two edits overlap if their original-token spans [o_start, o_end) intersect.
    Insertions (o_start == o_end) overlap only with edits that cover the same
    insertion point.
    """
    selected: list[dict] = []
    occupied: set[int] = set()  # occupied token positions

    for c in candidates:
        if c["score"] < MIN_PRECISION:
            continue  # skip low-confidence edits

        # Check overlap with already-selected edits
        span = set(range(c["o_start"], max(c["o_end"], c["o_start"] + 1)))
        if span & occupied:
            continue

        selected.append(c)
        occupied.update(span)

    return selected


def _apply_edits(original: str, selected_edits: list[dict]) -> str:
    """Apply selected edits to the original sentence via ERRANT token spans.

    Edits are applied right-to-left on spaCy tokens to avoid index shifts.
    The result is reconstructed by joining tokens with spaces (matching
    ERRANT's tokenisation convention).
    """
    import spacy

    # We need the same tokenisation ERRANT uses
    from annotate.errant_pipe import _get_annotator
    annotator = _get_annotator()
    orig_doc = annotator.parse(original)
    tokens = [tok.text for tok in orig_doc]

    # Sort edits right-to-left by o_start (descending) for safe in-place edits
    edits_sorted = sorted(selected_edits, key=lambda e: -e["o_start"])

    for e in edits_sorted:
        replacement = e["c_str"].split() if e["c_str"] else []
        tokens[e["o_start"]:e["o_end"]] = replacement

    return " ".join(tokens)


def hybrid_correct_sentence(
    original: str,
    engine_corrections: dict[str, str],
    priors: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Produce a hybrid correction for a single sentence.

    Returns dict with corrected text, selected edits, and metadata.
    """
    candidates = _extract_candidate_edits(original, engine_corrections, priors)
    selected = _select_non_overlapping(candidates)

    if selected:
        corrected = _apply_edits(original, selected)
    else:
        # No edits above threshold — fall back to the engine with highest
        # average prior (LoRA typically)
        corrected = original

    edit_r = lev_ratio(original, corrected)

    # Track which engines contributed
    engines_used = list({e["engine"] for e in selected})
    type_counts = defaultdict(int)
    for e in selected:
        type_counts[e["type"]] += 1

    return {
        "corrected": corrected,
        "edit_ratio": round(edit_r, 6),
        "n_edits_selected": len(selected),
        "n_candidates": len(candidates),
        "engines_used": sorted(engines_used),
        "edit_types": dict(type_counts),
    }


# ---------------------------------------------------------------------------
# main: run hybrid selection on dev_tune (and optionally dev_eval)
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    log.info("Hybrid selector — seed=%d, MIN_PRECISION=%.2f", SEED, MIN_PRECISION)

    # 1) Load priors
    priors_path = RESULTS_DIR / "engine_priors.json"
    with open(priors_path) as f:
        priors = json.load(f)
    log.info("Loaded priors for %d engines from %s", len(priors), priors_path)

    # 2) Define engine prediction files
    engine_files = {
        "languagetool": RESULTS_DIR / "lt_dev_tune_preds.jsonl",
        "t5-small-gec": RESULTS_DIR / "tagger_dev_tune_preds.jsonl",
        "lora-flan-t5-base": RESULTS_DIR / "lora_dev_tune_preds.jsonl",
    }

    # Process each split that has predictions
    for split_tag, suffix in [("dev_tune", "dev_tune"), ("dev_eval", "dev_eval")]:
        # Build per-split file paths
        split_files = {}
        for ename, base_path in engine_files.items():
            p = Path(str(base_path).replace("dev_tune", suffix))
            if p.exists():
                split_files[ename] = p

        if len(split_files) < 2:
            log.info("Skipping %s — fewer than 2 engine prediction files found", split_tag)
            continue

        log.info("=== Processing %s with %d engines ===", split_tag, len(split_files))

        # Load predictions, indexed by original text
        engine_preds: dict[str, dict[str, str]] = defaultdict(dict)
        all_ids: dict[str, str] = {}  # original → id
        all_cefr: dict[str, str] = {}  # original → cefr_band
        originals_ordered: list[str] = []  # preserve order from first engine

        first_engine = True
        for ename, fpath in split_files.items():
            records = load_jsonl(fpath)
            log.info("  Loaded %d predictions for %s", len(records), ename)

            for r in records:
                orig = r["original"]
                engine_preds[orig][ename] = r["corrected"]
                if first_engine:
                    originals_ordered.append(orig)
                    all_ids[orig] = r.get("id", "")
                    all_cefr[orig] = r.get("cefr_band", "UNK")
            first_engine = False

        log.info("  %d unique sentences to process", len(originals_ordered))

        # 3) Run hybrid selection
        hybrid_preds: list[dict] = []
        engine_contribution: defaultdict[str, int] = defaultdict(int)
        total_edits = 0

        for i, orig in enumerate(originals_ordered):
            corrections = engine_preds[orig]
            result = hybrid_correct_sentence(orig, corrections, priors)

            pred = {
                "id": all_ids.get(orig, f"hybrid_{i}"),
                "original": orig,
                "corrected": result["corrected"],
                "edit_ratio": result["edit_ratio"],
                "engine": "hybrid",
                "engines_used": result["engines_used"],
                "n_edits_selected": result["n_edits_selected"],
                "cefr_band": all_cefr.get(orig, "UNK"),
            }
            hybrid_preds.append(pred)

            for eng in result["engines_used"]:
                engine_contribution[eng] += 1
            total_edits += result["n_edits_selected"]

            if (i + 1) % 500 == 0 or (i + 1) == len(originals_ordered):
                log.info("  [%d/%d] sentences processed", i + 1, len(originals_ordered))

        # 4) Save
        out_path = RESULTS_DIR / f"hybrid_{suffix}_preds.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in hybrid_preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        log.info("  Saved %d hybrid predictions to %s", len(hybrid_preds), out_path)

        # 5) Summary statistics
        n_changed = sum(1 for p in hybrid_preds if p["original"] != p["corrected"])
        mean_ratio = (
            sum(p["edit_ratio"] for p in hybrid_preds) / len(hybrid_preds)
            if hybrid_preds else 0.0
        )
        mean_edits = total_edits / len(hybrid_preds) if hybrid_preds else 0.0

        print(f"\n=== Hybrid selector: {split_tag} ===")
        print(f"  Sentences:       {len(hybrid_preds)}")
        print(f"  Changed:         {n_changed} ({100 * n_changed / len(hybrid_preds):.1f}%)")
        print(f"  Mean edit ratio: {mean_ratio:.4f}")
        print(f"  Mean edits/sent: {mean_edits:.2f}")
        print(f"  Total edits:     {total_edits}")
        print(f"  Engine contributions (sentences where engine contributed):")
        for eng in sorted(engine_contribution.keys()):
            cnt = engine_contribution[eng]
            print(f"    {eng:<20s}: {cnt:>5d} ({100 * cnt / len(hybrid_preds):.1f}%)")

        # CEFR breakdown
        cefr_counts: defaultdict[str, list[float]] = defaultdict(list)
        for p in hybrid_preds:
            cefr_counts[p["cefr_band"]].append(p["edit_ratio"])
        print(f"  CEFR breakdown:")
        for band in sorted(cefr_counts.keys()):
            ratios = cefr_counts[band]
            avg_r = sum(ratios) / len(ratios)
            print(f"    {band}: n={len(ratios)}, mean_edit_ratio={avg_r:.4f}")


if __name__ == "__main__":
    main()
