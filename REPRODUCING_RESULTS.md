# Reproducing the Dissertation Results

This guide shows how to reproduce every numeric claim, table, and figure in
Sections 6 of the dissertation directly from the raw prediction files and
evaluation summaries committed to this repository. The instructions also cover
capturing live screenshots of the interactive Streamlit demonstrator.

All commands below assume you are at the repository root
(`~/Desktop/dissertation_gec`) and using a Python 3.11 environment with the
packages pinned in `requirements.txt` installed.

## 1. Environment setup

```bash
cd ~/Desktop/dissertation_gec
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The figure-generation scripts depend only on `matplotlib`, `numpy`, and the
standard library, so they run even without the ERRANT / LanguageTool / PyTorch
stack installed. The selector-statistics script additionally requires nothing
beyond the standard library.

## 2. Recomputing the Section 6.4 selector statistics

The selector-behaviour figures and the table in Section 6.4 are driven by
`results/selector_stats.json`, which is produced from the raw prediction JSONL
files. To regenerate it, run:

```bash
python3 app/recompute_selector_stats.py
```

Expected output (an excerpt):

```
{
  "dev_tune": {
    "total_sentences": 3500,
    "sentences_changed": 922,
    "sentences_changed_pct": 26.34,
    "mean_edit_ratio": 0.995002,
    "total_edits_selected": 1225,
    "engine_contributions_sentences": {
      "languagetool": 637,
      "t5-small-gec": 300,
      "lora-flan-t5-base": 85
    },
    ...
  },
  "dev_eval": {
    "total_sentences": 827,
    "sentences_changed": 190,
    ...
  }
}
```

Every row of Table 6.4 in the dissertation corresponds to one of these
fields.

## 3. Regenerating all figures

With `results/selector_stats.json` in place, regenerate every figure used in
Sections 5 and 6:

```bash
python3 app/make_dissertation_figures.py
```

This writes the following PNG files to `results/plots/`:

| Output file | Appears in |
|-------------|------------|
| `fig_eval_barchart_v2.png` | Figure 6.1 (corpus-level dev-tune P/R/F0.5) |
| `fig_eval_held_out_barchart.png` | Figure 6.2 (held-out dev-eval P/R/F0.5) |
| `fig_error_type_heatmap.png` | Figure 6.3 (per-error-type F0.5, top 15) |
| `fig_engine_priors_heatmap.png` | Figure 6.4 (engine precision priors) |
| `fig_engine_contributions_v2.png` | Figure 6.5 (selector engine contributions) |
| `fig_selector_summary.png` | Appendix / supplementary summary panel |

The feedback-demo figures used in Section 5 (`fig_cefr_comparison.png` and
`fig_feedback_demo.png`) are produced by the original
`app/generate_screenshots.py` script, which additionally requires the ERRANT
and feedback-generation modules. If you only wish to regenerate those, run:

```bash
python3 app/generate_screenshots.py
```

## 4. Re-running the full evaluation (optional, longer)

The numbers in Tables 6.1 and 6.2 are produced by the evaluation pipeline from
the raw engine predictions. To fully reproduce them, run:

```bash
# 1. Produce engine predictions (slow; requires torch + LanguageTool server)
python3 experiments/run_languagetool.py
python3 experiments/run_tagger.py
python3 experiments/run_lora.py

# 2. Compute priors from dev-tune predictions
python3 selector/compute_priors.py

# 3. Run the hybrid selector
python3 selector/hybrid_selector.py --split dev-tune
python3 selector/hybrid_selector.py --split dev-eval

# 4. Score everything with ERRANT
python3 eval/evaluate_hybrid.py --split dev-tune
python3 eval/evaluate_hybrid.py --split dev-eval
```

The resulting `results/eval_summary.json` and
`results/eval_dev_eval_summary.json` should match the values reported in
Tables 6.1 and 6.2 byte for byte under the fixed random seed of 42.

## 5. Capturing screenshots of the Streamlit demonstrator

The "figure-style" tool renderings in Figures 5.1 and 5.2 show the feedback
layer's output; to additionally capture *live* screenshots of the Streamlit
web interface itself, follow these steps.

### Option A — manual capture (recommended)

1. Start the Streamlit server:

   ```bash
   streamlit run app/demo.py
   ```

2. Open the URL Streamlit prints (usually `http://localhost:8501`) in a
   browser.

3. For each CEFR level (A, B, C, N), enter a learner sentence with a known
   error (for example, `"She go to school yesterday ."`), select the level
   from the sidebar, choose *Hybrid* as the correction mode, and take a
   screenshot of the full page.

4. Save the screenshots as
   `writing/screenshots/streamlit_level_{A|B|C|N}.png`.

### Option B — scripted capture with Playwright (if you prefer automation)

Install Playwright once:

```bash
pip install playwright
playwright install chromium
```

Then run the provided script in a separate terminal *after* starting Streamlit
as in step 1 above:

```bash
python3 app/capture_streamlit_screenshots.py
```

This produces four PNGs in `writing/screenshots/` corresponding to the four
CEFR levels.

Once the screenshots are in place, the figure references already inserted in
Section 5 (`../results/plots/fig_cefr_comparison.png` and
`../results/plots/fig_feedback_demo.png`) can be supplemented or replaced by
pointing at the screenshot files instead.

## 6. Verifying the numeric claims in the dissertation

A small sanity-check script that re-derives every number that appears in the
results tables is provided:

```bash
python3 app/recompute_selector_stats.py > /tmp/selector.json
python3 -c "
import json
s = json.load(open('results/eval_summary.json'))
e = json.load(open('results/eval_dev_eval_summary.json'))
print('Hybrid dev-tune precision: {:.4f} (expected 0.3039)'.format(s['hybrid']['precision']))
print('Hybrid dev-eval precision: {:.4f} (expected 0.4420)'.format(e['hybrid']['precision']))
print('Hybrid dev-tune F0.5:      {:.4f} (expected 0.1683)'.format(s['hybrid']['f05']))
print('Hybrid dev-eval F0.5:      {:.4f} (expected 0.2079)'.format(e['hybrid']['f05']))
"
```

Every number in Sections 6.1, 6.3, and 6.4 should match the values printed
by this script or by `recompute_selector_stats.py`.

## File manifest

Scripts added for this reproduction workflow:

- `app/recompute_selector_stats.py` — recomputes selector-behaviour statistics
- `app/make_dissertation_figures.py` — regenerates all Section 5 and 6 figures
- `app/capture_streamlit_screenshots.py` — captures Streamlit screenshots

Inputs required (all already present in `results/`):

- `eval_summary.json`, `eval_dev_eval_summary.json`, `eval_by_type.json`
- `engine_priors.json`
- `hybrid_dev_tune_preds.jsonl`, `hybrid_dev_eval_preds.jsonl`
