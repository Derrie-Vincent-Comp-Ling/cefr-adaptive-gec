"""Generate publication-ready figures for the dissertation.

Produces PNG images showing:
    1. CEFR-adaptive feedback comparison (all 4 levels for one sentence)
    2. Hybrid vs single-engine correction comparison
    3. ERRANT edit-level provenance (which engine contributed each edit)
    4. Corpus-level evaluation results table

Run:
    cd ~/Desktop/dissertation_gec
    python app/generate_screenshots.py

Outputs saved to: results/plots/
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, RESULTS_PLOTS_DIR, SEED, ensure_dirs
from annotate.errant_pipe import get_edits
from feedback.feedback_gen import generate_feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ensure_dirs()

CEFR_COLOURS = {"A": "#10b981", "B": "#3b82f6", "C": "#8b5cf6", "N": "#6b7280"}
CEFR_LABELS = {"A": "A — Beginner", "B": "B — Intermediate", "C": "C — Advanced", "N": "N — Proficient"}
TYPE_COLOURS = {"R": "#ef4444", "M": "#f59e0b", "U": "#3b82f6"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


# ===================================================================
# Figure 1: CEFR Feedback Comparison
# ===================================================================
def fig_cefr_comparison():
    original = "I has a apple ."
    corrected = "I have an apple ."

    fig, axes = plt.subplots(4, 1, figsize=(10, 8.5))
    fig.suptitle("CEFR-Adaptive Feedback Comparison", fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94,
             f"Original: \"{original}\"   →   Corrected: \"{corrected}\"",
             ha="center", fontsize=11, style="italic", color="#374151")

    for idx, level in enumerate(["A", "B", "C", "N"]):
        ax = axes[idx]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fb = generate_feedback(original, corrected, level)

        # Header bar
        colour = CEFR_COLOURS[level]
        rect = mpatches.FancyBboxPatch(
            (0.0, 0.0), 10.0, 1.0,
            boxstyle="round,pad=0.05",
            facecolor="white", edgecolor=colour, linewidth=2.5
        )
        ax.add_patch(rect)

        # Level badge
        badge = mpatches.FancyBboxPatch(
            (0.1, 0.15), 1.8, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=colour, edgecolor="none"
        )
        ax.add_patch(badge)
        ax.text(1.0, 0.5, CEFR_LABELS[level], ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

        # Feedback text
        items_text = []
        for i, item in enumerate(fb.items, 1):
            items_text.append(f"{i}. [{item.error_type}] {item.feedback_text}")

        text = "\n".join(items_text) if items_text else "No errors detected."
        ax.text(2.2, 0.5, text, ha="left", va="center", fontsize=9,
                color="#1f2937", wrap=True,
                transform=ax.transData,
                fontfamily="sans-serif")

    plt.subplots_adjust(top=0.90, bottom=0.02, hspace=0.25)
    out = RESULTS_PLOTS_DIR / "fig_cefr_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


# ===================================================================
# Figure 2: Corpus-Level Evaluation Bar Chart
# ===================================================================
def fig_eval_barchart():
    summary_path = RESULTS_DIR / "eval_summary.json"
    if not summary_path.exists():
        log.warning("eval_summary.json not found — skipping bar chart")
        return

    with open(summary_path) as f:
        data = json.load(f)

    engines = list(data.keys())
    precision = [data[e]["precision"] for e in engines]
    recall = [data[e]["recall"] for e in engines]
    f05 = [data[e]["f05"] for e in engines]

    # Nicer labels
    labels = {
        "languagetool": "LanguageTool",
        "t5-small-gec": "T5-small",
        "lora-flan-t5-base": "LoRA flan-t5",
        "hybrid": "Hybrid",
    }
    engine_labels = [labels.get(e, e) for e in engines]
    colours_p = ["#93c5fd", "#93c5fd", "#93c5fd", "#3b82f6"]
    colours_r = ["#fca5a5", "#fca5a5", "#fca5a5", "#ef4444"]
    colours_f = ["#86efac", "#86efac", "#86efac", "#22c55e"]

    x = range(len(engines))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_p = ax.bar([i - width for i in x], precision, width, label="Precision",
                     color=colours_p, edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x, recall, width, label="Recall",
                     color=colours_r, edgecolor="white", linewidth=0.5)
    bars_f = ax.bar([i + width for i in x], f05, width, label="F0.5",
                     color=colours_f, edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(engine_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("ERRANT Corpus-Level Evaluation (dev-tune, n=3,022)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, max(max(precision), max(recall), max(f05)) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = RESULTS_PLOTS_DIR / "fig_eval_barchart.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


# ===================================================================
# Figure 3: Per-Error-Type Heatmap (top 15 types)
# ===================================================================
def fig_error_type_heatmap():
    by_type_path = RESULTS_DIR / "eval_by_type.json"
    if not by_type_path.exists():
        log.warning("eval_by_type.json not found — skipping heatmap")
        return

    with open(by_type_path) as f:
        data = json.load(f)

    engines = ["languagetool", "t5-small-gec", "lora-flan-t5-base", "hybrid"]
    engine_labels = ["LanguageTool", "T5-small", "LoRA flan-t5", "Hybrid"]

    # Get top 15 types by hybrid F0.5
    hybrid_types = data.get("hybrid", {})
    sorted_types = sorted(hybrid_types.items(), key=lambda x: -x[1]["f05"])
    top_types = [t[0] for t in sorted_types[:15]]

    # Build matrix
    matrix = []
    for etype in top_types:
        row = []
        for eng in engines:
            eng_data = data.get(eng, {})
            val = eng_data.get(etype, {}).get("f05", 0.0)
            row.append(val)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(8, 7))

    import numpy as np
    mat = np.array(matrix)
    im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0, vmax=max(0.5, mat.max()))

    ax.set_xticks(range(len(engine_labels)))
    ax.set_xticklabels(engine_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(top_types)))
    ax.set_yticklabels(top_types, fontsize=9, fontfamily="monospace")

    # Annotate cells
    for i in range(len(top_types)):
        for j in range(len(engines)):
            val = mat[i, j]
            colour = "white" if val > 0.25 else "#1f2937"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=colour, fontweight="bold" if val > 0.2 else "normal")

    ax.set_title("Per-Error-Type F0.5 — Top 15 Types", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="F0.5", shrink=0.8)

    out = RESULTS_PLOTS_DIR / "fig_error_type_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


# ===================================================================
# Figure 4: Engine Contribution Pie Chart
# ===================================================================
def fig_engine_contributions():
    hybrid_path = RESULTS_DIR / "hybrid_dev_tune_preds.jsonl"
    if not hybrid_path.exists():
        log.warning("hybrid predictions not found — skipping pie chart")
        return

    with open(hybrid_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    from collections import Counter
    engine_counts = Counter()
    no_edit_count = 0
    for r in records:
        used = r.get("engines_used", [])
        if not used:
            no_edit_count += 1
        for eng in used:
            engine_counts[eng] += 1

    labels_map = {
        "languagetool": "LanguageTool",
        "t5-small-gec": "T5-small",
        "lora-flan-t5-base": "LoRA flan-t5",
    }
    labels = [labels_map.get(e, e) for e in engine_counts.keys()]
    sizes = list(engine_counts.values())
    labels.append("No edit")
    sizes.append(no_edit_count)

    colours = ["#3b82f6", "#f59e0b", "#10b981", "#e5e7eb"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colours[:len(sizes)],
        startangle=90, pctdistance=0.75,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title("Hybrid Selector: Engine Contributions\n(sentences where engine contributed at least one edit)",
                 fontsize=12, fontweight="bold")

    out = RESULTS_PLOTS_DIR / "fig_engine_contributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


# ===================================================================
# Figure 5: CEFR-level feedback demo (rich text style)
# ===================================================================
def fig_feedback_demo():
    """Generate a detailed multi-sentence feedback figure."""
    test_cases = [
        ("She go to school yesterday .", "She went to school yesterday ."),
        ("He speaked to the informations desk .", "He spoke to the information desk ."),
    ]

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("System Demo: CEFR-Adaptive Feedback Examples",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = GridSpec(len(test_cases), 2, figure=fig, hspace=0.4, wspace=0.15)

    for row, (orig, corr) in enumerate(test_cases):
        # Left column: Beginner (A)
        ax_a = fig.add_subplot(gs[row, 0])
        ax_a.axis("off")
        ax_a.set_xlim(0, 10)
        ax_a.set_ylim(0, 3)

        fb_a = generate_feedback(orig, corr, "A")
        rect = mpatches.FancyBboxPatch(
            (0, 0), 10, 3, boxstyle="round,pad=0.1",
            facecolor="#ecfdf5", edgecolor="#10b981", linewidth=2
        )
        ax_a.add_patch(rect)
        ax_a.text(0.3, 2.6, "CEFR A — Beginner", fontsize=11,
                  fontweight="bold", color="#065f46")
        ax_a.text(0.3, 2.2, f'"{orig}"', fontsize=9, style="italic", color="#374151")
        ax_a.text(0.3, 1.85, f'→ "{corr}"', fontsize=9, color="#065f46", fontweight="bold")

        y = 1.4
        for i, item in enumerate(fb_a.items, 1):
            text = f"{i}. [{item.error_type}] {item.feedback_text}"
            # Wrap long text
            if len(text) > 80:
                text = text[:77] + "…"
            ax_a.text(0.3, y, text, fontsize=8, color="#1f2937", wrap=True)
            y -= 0.5

        # Right column: Advanced (C)
        ax_c = fig.add_subplot(gs[row, 1])
        ax_c.axis("off")
        ax_c.set_xlim(0, 10)
        ax_c.set_ylim(0, 3)

        fb_c = generate_feedback(orig, corr, "C")
        rect = mpatches.FancyBboxPatch(
            (0, 0), 10, 3, boxstyle="round,pad=0.1",
            facecolor="#f5f3ff", edgecolor="#8b5cf6", linewidth=2
        )
        ax_c.add_patch(rect)
        ax_c.text(0.3, 2.6, "CEFR C — Advanced", fontsize=11,
                  fontweight="bold", color="#5b21b6")
        ax_c.text(0.3, 2.2, f'"{orig}"', fontsize=9, style="italic", color="#374151")
        ax_c.text(0.3, 1.85, f'→ "{corr}"', fontsize=9, color="#5b21b6", fontweight="bold")

        y = 1.4
        for i, item in enumerate(fb_c.items, 1):
            text = f"{i}. {item.feedback_text}"
            if len(text) > 80:
                text = text[:77] + "…"
            ax_c.text(0.3, y, text, fontsize=8, color="#1f2937")
            y -= 0.5

    out = RESULTS_PLOTS_DIR / "fig_feedback_demo.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


# ===================================================================
# Main
# ===================================================================
def main():
    log.info("Generating dissertation figures …")
    fig_cefr_comparison()
    fig_eval_barchart()
    fig_error_type_heatmap()
    fig_engine_contributions()
    fig_feedback_demo()
    log.info("All figures saved to %s", RESULTS_PLOTS_DIR)
    print(f"\nDone — {5} figures saved to {RESULTS_PLOTS_DIR}/")


if __name__ == "__main__":
    main()
