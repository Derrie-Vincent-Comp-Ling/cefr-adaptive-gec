"""Run hybrid selector + ERRANT evaluation on the held-out dev_eval split.

Prerequisite: lt_dev_eval_preds.jsonl, tagger_dev_eval_preds.jsonl,
and lora_dev_eval_preds.jsonl must all exist in results/.

Run AFTER:
    python experiments/run_lt_dev_eval.py
    python experiments/run_tagger_dev_eval.py
    (lora_dev_eval_preds.jsonl already exists from Colab)

Outputs:
    results/hybrid_dev_eval_preds.jsonl
    results/eval_dev_eval_summary.json
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, SEED, ensure_dirs
from annotate.errant_pipe import errant_f05, get_edits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "dev_eval_full.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

MIN_PRECISION = 0.10


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hybrid_correct(original, engine_corrections, priors):
    """Edit-level hybrid selection (same logic as selector/hybrid_selector.py)."""
    candidates = []
    for engine_name, corrected in engine_corrections.items():
        if corrected == original:
            continue
        edits = get_edits(original, corrected)
        engine_prior = priors.get(engine_name, {})
        for e in edits:
            if e["type"] == "noop":
                continue
            score = engine_prior.get(e["type"], 0.0)
            candidates.append({**e, "engine": engine_name, "score": score})
    candidates.sort(key=lambda c: (-c["score"], c["o_end"] - c["o_start"]))

    selected = []
    occupied = set()
    for c in candidates:
        if c["score"] < MIN_PRECISION:
            continue
        span = set(range(c["o_start"], max(c["o_end"], c["o_start"] + 1)))
        if span & occupied:
            continue
        selected.append(c)
        occupied.update(span)

    if selected:
        from annotate.errant_pipe import _get_annotator
        annotator = _get_annotator()
        orig_doc = annotator.parse(original)
        tokens = [tok.text for tok in orig_doc]
        for e in sorted(selected, key=lambda e: -e["o_start"]):
            replacement = e["c_str"].split() if e["c_str"] else []
            tokens[e["o_start"]:e["o_end"]] = replacement
        corrected = " ".join(tokens)
    else:
        corrected = original

    from Levenshtein import ratio as lev_ratio
    return {
        "corrected": corrected,
        "edit_ratio": round(lev_ratio(original, corrected), 6),
        "engines_used": sorted({e["engine"] for e in selected}),
        "n_edits": len(selected),
    }


def main():
    ensure_dirs()
    log.info("Dev-eval full pipeline — seed=%d", SEED)

    # Load priors
    with open(RESULTS_DIR / "engine_priors.json") as f:
        priors = json.load(f)

    # Check all prediction files exist
    engine_files = {
        "languagetool": RESULTS_DIR / "lt_dev_eval_preds.jsonl",
        "t5-small-gec": RESULTS_DIR / "tagger_dev_eval_preds.jsonl",
        "lora-flan-t5-base": RESULTS_DIR / "lora_dev_eval_preds.jsonl",
    }
    for name, path in engine_files.items():
        if not path.exists():
            log.error("Missing %s — run the baseline first: %s", name, path)
            sys.exit(1)

    # Load predictions indexed by original
    engine_preds = defaultdict(dict)
    originals_ordered = []
    all_ids = {}
    all_cefr = {}

    first = True
    for ename, fpath in engine_files.items():
        records = load_jsonl(fpath)
        log.info("Loaded %d predictions for %s", len(records), ename)
        for r in records:
            orig = r["original"]
            engine_preds[orig][ename] = r["corrected"]
            if first:
                originals_ordered.append(orig)
                all_ids[orig] = r.get("id", "")
                all_cefr[orig] = r.get("cefr_band", "UNK")
        first = False

    log.info("%d unique sentences", len(originals_ordered))

    # Run hybrid
    log.info("Running hybrid selection …")
    hybrid_preds = []
    for i, orig in enumerate(originals_ordered):
        result = hybrid_correct(orig, engine_preds[orig], priors)
        hybrid_preds.append({
            "id": all_ids.get(orig, f"hybrid_{i}"),
            "original": orig,
            "corrected": result["corrected"],
            "edit_ratio": result["edit_ratio"],
            "engine": "hybrid",
            "engines_used": result["engines_used"],
            "cefr_band": all_cefr.get(orig, "UNK"),
        })
        if (i + 1) % 200 == 0:
            log.info("  [%d/%d]", i + 1, len(originals_ordered))

    out = RESULTS_DIR / "hybrid_dev_eval_preds.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in hybrid_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("Saved %d hybrid predictions to %s", len(hybrid_preds), out)

    # Load gold
    dev_pairs = load_jsonl(DATA_PROCESSED_DIR / "dev_pairs.jsonl")
    src_to_target = {r["source"]: r["target"] for r in dev_pairs}
    log.info("Loaded %d gold pairs", len(src_to_target))

    # Evaluate all 4 systems
    all_pred_files = {
        **engine_files,
        "hybrid": out,
    }

    results = {}
    for ename, fpath in all_pred_files.items():
        preds = load_jsonl(fpath)
        origs, predictions, refs = [], [], []
        for p in preds:
            if p["original"] in src_to_target:
                origs.append(p["original"])
                predictions.append(p["corrected"])
                refs.append(src_to_target[p["original"]])

        log.info("%s: matched %d / %d to gold", ename, len(origs), len(preds))
        if not origs:
            continue

        log.info("Computing ERRANT F0.5 for %s …", ename)
        scores = errant_f05(origs, predictions, refs)
        results[ename] = scores
        log.info("  %s: P=%.4f R=%.4f F0.5=%.4f", ename,
                 scores["precision"], scores["recall"], scores["f05"])

    # Save
    summary_path = RESULTS_DIR / "eval_dev_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print(f"\n{'=' * 72}")
    print("  ERRANT Corpus-Level Evaluation (dev_eval — HELD OUT)")
    print(f"{'=' * 72}")
    print(f"  {'Engine':<22s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>8s} {'R':>8s} {'F0.5':>8s}")
    print("  " + "-" * 68)
    for eng, s in results.items():
        print(f"  {eng:<22s} {s['TP']:>6d} {s['FP']:>6d} {s['FN']:>6d} "
              f"{s['precision']:>8.4f} {s['recall']:>8.4f} {s['f05']:>8.4f}")
    print(f"{'=' * 72}")

    if "hybrid" in results:
        h = results["hybrid"]
        others = {k: v for k, v in results.items() if k != "hybrid"}
        if others:
            best = max(others, key=lambda k: others[k]["f05"])
            delta = h["f05"] - others[best]["f05"]
            print(f"\n  Hybrid F0.5: {h['f05']:.4f}")
            print(f"  Best single ({best}): {others[best]['f05']:.4f}")
            print(f"  Delta: {delta:+.4f}")


if __name__ == "__main__":
    main()
