"""CEFR-adaptive feedback generator.

Takes an original sentence, a corrected sentence, and the learner's CEFR
level, then produces structured, pedagogically appropriate feedback for
each detected error.

Pipeline:
    1. Extract ERRANT edits between original and corrected.
    2. For each edit, look up the CEFR-appropriate template.
    3. Return a FeedbackResult with per-edit items and a formatted summary.

CEFR adaptation rationale (Ellis, 2009; Bitchener & Storch, 2016):
    A — Direct corrective feedback with metalinguistic explanation.
    B — Metalinguistic feedback with grammar terminology.
    C — Indirect coded feedback (error code + correction).
    N — Minimal flagging (error code only).

Usage:
    from feedback.feedback_gen import generate_feedback

    result = generate_feedback(
        original="I has a apple .",
        corrected="I have an apple .",
        cefr="A",
    )
    print(result.summary)

Reference:
    Ellis, R. (2009) 'A typology of written corrective feedback types',
    ELT Journal, 63(2), pp. 97-107.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotate.errant_pipe import get_edits
from feedback.templates import get_feedback

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FeedbackItem:
    """A single piece of feedback for one ERRANT edit."""
    error_type: str
    original_span: str       # erroneous token(s)
    corrected_span: str      # corrected token(s)
    o_start: int             # token offset start (ERRANT)
    o_end: int               # token offset end (ERRANT)
    feedback_text: str       # CEFR-adapted explanation
    cefr_level: str          # A/B/C/N


@dataclass
class FeedbackResult:
    """Full feedback for one sentence."""
    original: str
    corrected: str
    cefr: str
    n_errors: int
    items: list[FeedbackItem] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable feedback summary."""
        if not self.items:
            if self.original == self.corrected:
                return "No errors detected. Well done!"
            return f"Corrected: {self.corrected}"

        lines = []
        # Header adapts to CEFR
        if self.cefr == "A":
            lines.append(f"We found {self.n_errors} thing(s) to fix in your sentence:")
            lines.append(f"  Original:  {self.original}")
            lines.append(f"  Corrected: {self.corrected}")
            lines.append("")
        elif self.cefr == "B":
            lines.append(f"{self.n_errors} error(s) detected:")
            lines.append(f"  Original:  {self.original}")
            lines.append(f"  Corrected: {self.corrected}")
            lines.append("")
        elif self.cefr == "C":
            lines.append(f"{self.n_errors} error(s):")
            lines.append("")
        else:  # N
            lines.append(f"{self.n_errors} error(s):")
            lines.append("")

        for i, item in enumerate(self.items, 1):
            if self.cefr in ("A", "B"):
                lines.append(f"  {i}. [{item.error_type}] {item.feedback_text}")
            else:
                lines.append(f"  {i}. {item.feedback_text}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict for JSONL output."""
        return {
            "original": self.original,
            "corrected": self.corrected,
            "cefr": self.cefr,
            "n_errors": self.n_errors,
            "items": [asdict(it) for it in self.items],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def generate_feedback(
    original: str,
    corrected: str,
    cefr: str = "B",
) -> FeedbackResult:
    """Generate CEFR-adaptive feedback for a corrected sentence.

    Parameters
    ----------
    original  : the learner's original sentence
    corrected : the corrected sentence (from hybrid selector or any engine)
    cefr      : learner's CEFR level ('A', 'B', 'C', or 'N')

    Returns
    -------
    FeedbackResult with per-edit items and a formatted summary.
    """
    cefr = cefr.upper()
    if cefr not in ("A", "B", "C", "N"):
        cefr = "B"

    edits = get_edits(original, corrected)
    # Filter out noop edits
    real_edits = [e for e in edits if e["type"] != "noop"]

    items: list[FeedbackItem] = []
    for e in real_edits:
        fb_text = get_feedback(
            error_type=e["type"],
            cefr=cefr,
            original=e["o_str"] if e["o_str"] else "(nothing)",
            corrected=e["c_str"] if e["c_str"] else "(remove)",
        )
        items.append(FeedbackItem(
            error_type=e["type"],
            original_span=e["o_str"],
            corrected_span=e["c_str"],
            o_start=e["o_start"],
            o_end=e["o_end"],
            feedback_text=fb_text,
            cefr_level=cefr,
        ))

    return FeedbackResult(
        original=original,
        corrected=corrected,
        cefr=cefr,
        n_errors=len(items),
        items=items,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def generate_feedback_batch(
    records: list[dict],
    cefr_key: str = "cefr_band",
    original_key: str = "original",
    corrected_key: str = "corrected",
    default_cefr: str = "B",
) -> list[FeedbackResult]:
    """Generate feedback for a batch of prediction records.

    Each record should have at minimum 'original' and 'corrected' fields.
    CEFR level is read from cefr_key if present, else defaults to default_cefr.
    """
    results: list[FeedbackResult] = []
    for i, r in enumerate(records):
        orig = r.get(original_key, "")
        corr = r.get(corrected_key, "")
        cefr = r.get(cefr_key, default_cefr)
        if not cefr or cefr == "UNK":
            cefr = default_cefr

        fb = generate_feedback(orig, corr, cefr)
        results.append(fb)

        if (i + 1) % 500 == 0:
            log.info("  Generated feedback for %d / %d sentences", i + 1, len(records))

    return results


# ---------------------------------------------------------------------------
# CLI: demo + batch mode
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Demo: show same sentence at all four CEFR levels
    test_pairs = [
        ("I has a apple .", "I have an apple ."),
        ("She go to school yesterday .", "She went to school yesterday ."),
        ("He speaked to the informations desk .", "He spoke to the information desk ."),
        ("I am agree with this opinion .", "I agree with this opinion ."),
    ]

    for orig, corr in test_pairs:
        print("=" * 72)
        print(f"Original:  {orig}")
        print(f"Corrected: {corr}")
        for level in ("A", "B", "C", "N"):
            print(f"\n--- CEFR {level} ---")
            result = generate_feedback(orig, corr, level)
            print(result.summary)
    print("=" * 72)

    # If hybrid predictions exist, run batch on first 5
    from config import RESULTS_DIR
    hybrid_path = RESULTS_DIR / "hybrid_dev_tune_preds.jsonl"
    if hybrid_path.exists():
        print(f"\n\n{'=' * 72}")
        print("Batch demo: first 5 hybrid predictions")
        print(f"{'=' * 72}")

        with open(hybrid_path) as f:
            records = [json.loads(line) for line in f if line.strip()][:5]

        for r in records:
            cefr = r.get("cefr_band", "B")
            if cefr == "UNK":
                cefr = "B"
            result = generate_feedback(r["original"], r["corrected"], cefr)
            print(f"\n{'-' * 60}")
            print(result.summary)


if __name__ == "__main__":
    main()
