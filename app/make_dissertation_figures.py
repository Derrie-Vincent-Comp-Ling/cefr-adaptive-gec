"""Standalone figure-generation script for the dissertation.

Produces publication-ready PNG figures using only the pre-computed JSON
summaries in results/ — no errant/torch/Java dependencies needed.

Figures produced (all saved to results/plots/):
    fig_eval_barchart.png           — Corpus-level P/R/F0.5 bar chart
    fig_eval_held_out_barchart.png  — Held-out (dev-eval) P/R/F0.5 bar chart
    fig_error_type_heatmap.png      — Per-error-type F0.5 heatmap, top 15 types
    fig_engine_contributions.png    — Selector engine-contribution bar chart
    fig_selector_summary.png        — Selector behaviour summary panel
    fig_engine_priors_heatmap.png   — Engine precision priors (complementarity view)

Run:
    cd ~/Desktop/dissertation_gec
    python3 app/make_dissertation_figures.py

Inputs required in results/:
    eval_summary.json
    eval_dev_eval_summary.json
    eval_by_type.json
    engine_priors.json
    selector_stats.json               (produced by scripts/recompute_selector_stats.py)
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

ENGINE_LABELS = {
    "languagetool": "LanguageTool",
    "t5-small-gec": "T5-small",
    "lora-flan-t5-base": "LoRA flan-t5",
    "hybrid": "Hybrid",
}


def _barchart(summary_path: Path, title: str, out: Path) -> None:
    with summary_path.open() as f:
        data = json.load(f)
    engines = list(data.keys())
    labels = [ENGINE_LABELS.get(e, e) for e in engines]
    precision = [data[e]["precision"] for e in engines]
    recall = [data[e]["recall"] for e in engines]
    f05 = [data[e]["f05"] for e in engines]

    colours_p = ["#93c5fd"] * (len(engines) - 1) + ["#3b82f6"]
    colours_r = ["#fca5a5"] * (len(engines) - 1) + ["#ef4444"]
    colours_f = ["#86efac"] * (len(engines) - 1) + ["#22c55e"]

    x = np.arange(len(engines))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_p = ax.bar(x - width, precision, width, label="Precision",
                    color=colours_p, edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x, recall, width, label="Recall",
                    color=colours_r, edgecolor="white", linewidth=0.5)
    bars_f = ax.bar(x + width, f05, width, label="F0.5",
                    color=colours_f, edgecolor="white", linewidth=0.5)
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, max(max(precision), max(recall), max(f05)) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


def fig_eval_barcharts() -> None:
    _barchart(
        RESULTS / "eval_summary.json",
        "ERRANT Corpus-Level Evaluation — Dev-Tune (n = 3,022 matched pairs)",
        PLOTS / "fig_eval_barchart.png",
    )
    _barchart(
        RESULTS / "eval_dev_eval_summary.json",
        "ERRANT Corpus-Level Evaluation — Held-Out Dev-Eval (n = 692 matched pairs)",
        PLOTS / "fig_eval_held_out_barchart.png",
    )


def fig_error_type_heatmap() -> None:
    with (RESULTS / "eval_by_type.json").open() as f:
        data = json.load(f)

    engines = ["languagetool", "t5-small-gec", "lora-flan-t5-base", "hybrid"]
    labels = [ENGINE_LABELS[e] for e in engines]

    hybrid_types = data.get("hybrid", {})
    ranked = sorted(hybrid_types.items(), key=lambda x: -x[1]["f05"])
    top = [t[0] for t in ranked[:15]]

    matrix = np.array([
        [data.get(e, {}).get(t, {}).get("f05", 0.0) for e in engines]
        for t in top
    ])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto",
                   vmin=0, vmax=max(0.5, matrix.max()))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=9, fontfamily="monospace")
    for i in range(len(top)):
        for j in range(len(engines)):
            val = matrix[i, j]
            colour = "white" if val > 0.25 else "#1f2937"
            weight = "bold" if val > 0.2 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=colour, fontweight=weight)
    ax.set_title("Per-Error-Type F0.5 — Top 15 ERRANT Types (Dev-Tune)")
    fig.colorbar(im, ax=ax, label="F0.5", shrink=0.8)
    out = PLOTS / "fig_error_type_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


def fig_engine_priors_heatmap() -> None:
    with (RESULTS / "engine_priors.json").open() as f:
        priors = json.load(f)
    engines = ["languagetool", "t5-small-gec", "lora-flan-t5-base"]
    labels = [ENGINE_LABELS[e] for e in engines]
    all_types = sorted({t for eng in engines for t in priors.get(eng, {})})
    # Limit to types where at least one engine has prior > 0
    active_types = [t for t in all_types if any(priors.get(e, {}).get(t, 0) > 0 for e in engines)]

    matrix = np.array([
        [priors.get(e, {}).get(t, 0.0) for e in engines]
        for t in active_types
    ])

    fig, ax = plt.subplots(figsize=(7, max(6, 0.18 * len(active_types))))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_yticks(range(len(active_types)))
    ax.set_yticklabels(active_types, fontsize=7, fontfamily="monospace")
    for i in range(len(active_types)):
        for j in range(len(engines)):
            val = matrix[i, j]
            if val > 0:
                colour = "white" if val > 0.55 else "#0f172a"
                weight = "bold" if val >= 0.67 else "normal"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=colour, fontweight=weight)
    ax.set_title("Empirical Precision Priors per Engine × Error Type\n(Dev-Tune; used by the hybrid selector)")
    fig.colorbar(im, ax=ax, label="Precision", shrink=0.7)
    out = PLOTS / "fig_engine_priors_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


def fig_engine_contributions() -> None:
    stats_path = RESULTS / "selector_stats.json"
    if not stats_path.exists():
        log.warning("selector_stats.json not found; run scripts/recompute_selector_stats.py first")
        return
    with stats_path.open() as f:
        stats = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, split_name in zip(axes, ("dev_tune", "dev_eval")):
        d = stats[split_name]
        engines = list(d["engine_contributions_sentences"].keys())
        counts = [d["engine_contributions_sentences"][e] for e in engines]
        labels = [ENGINE_LABELS.get(e, e) for e in engines]
        colours = ["#3b82f6", "#f59e0b", "#10b981"][:len(engines)]

        bars = ax.bar(labels, counts, color=colours, edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            h = bar.get_height()
            pct_partition = 100 * count / d["total_sentences"]
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.02,
                    f"{count}\n({pct_partition:.1f}% of partition)",
                    ha="center", va="bottom", fontsize=9)
        pretty = {"dev_tune": "Dev-Tune", "dev_eval": "Dev-Eval"}[split_name]
        ax.set_title(f"{pretty} Engine Contributions "
                     f"({d['sentences_changed']}/{d['total_sentences']} sentences changed)")
        ax.set_ylabel("Sentences with ≥ 1 edit from engine")
        ax.set_ylim(0, max(counts) * 1.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Hybrid Selector: Per-Engine Sentence Contributions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOTS / "fig_engine_contributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


def fig_selector_summary() -> None:
    stats_path = RESULTS / "selector_stats.json"
    with stats_path.open() as f:
        stats = json.load(f)

    dt = stats["dev_tune"]
    de = stats["dev_eval"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    rows = [
        ["Metric", "Dev-Tune", "Dev-Eval"],
        ["Total sentences", f"{dt['total_sentences']:,}", f"{de['total_sentences']:,}"],
        ["Sentences changed",
         f"{dt['sentences_changed']:,} ({dt['sentences_changed_pct']}%)",
         f"{de['sentences_changed']:,} ({de['sentences_changed_pct']}%)"],
        ["Mean edit ratio",
         f"{dt['mean_edit_ratio']:.4f}",
         f"{de['mean_edit_ratio']:.4f}"],
        ["Total edits selected",
         f"{dt['total_edits_selected']:,}",
         "—"],
        ["LanguageTool sentences",
         f"{dt['engine_contributions_sentences'].get('languagetool', 0):,} "
         f"({dt['engine_contributions_pct_of_partition'].get('languagetool', 0)}%)",
         f"{de['engine_contributions_sentences'].get('languagetool', 0):,} "
         f"({de['engine_contributions_pct_of_partition'].get('languagetool', 0)}%)"],
        ["T5-small sentences",
         f"{dt['engine_contributions_sentences'].get('t5-small-gec', 0):,} "
         f"({dt['engine_contributions_pct_of_partition'].get('t5-small-gec', 0)}%)",
         f"{de['engine_contributions_sentences'].get('t5-small-gec', 0):,} "
         f"({de['engine_contributions_pct_of_partition'].get('t5-small-gec', 0)}%)"],
        ["LoRA flan-t5 sentences",
         f"{dt['engine_contributions_sentences'].get('lora-flan-t5-base', 0):,} "
         f"({dt['engine_contributions_pct_of_partition'].get('lora-flan-t5-base', 0)}%)",
         f"{de['engine_contributions_sentences'].get('lora-flan-t5-base', 0):,} "
         f"({de['engine_contributions_pct_of_partition'].get('lora-flan-t5-base', 0)}%)"],
    ]
    tbl = ax.table(cellText=rows, loc="center", cellLoc="left",
                   colWidths=[0.38, 0.31, 0.31])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1e3a8a")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    ax.set_title("Hybrid Selector: Behavioural Statistics", pad=20)
    out = PLOTS / "fig_selector_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", out)


if __name__ == "__main__":
    fig_eval_barcharts()
    fig_error_type_heatmap()
    fig_engine_priors_heatmap()
    fig_engine_contributions()
    fig_selector_summary()
    log.info("All figures written to %s", PLOTS)
