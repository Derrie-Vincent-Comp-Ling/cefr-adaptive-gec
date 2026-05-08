"""Streamlit demo — Error-Type-Aware, CEFR-Adaptive Feedback for L2 Writing.

Run with:
    cd ~/Desktop/dissertation_gec
    streamlit run app/demo.py

The demo lets users:
    1. Paste or type L2 English text.
    2. Select their CEFR level (A / B / C / N).
    3. Choose which GEC engines to use.
    4. View the corrected text with CEFR-adaptive feedback.
    5. Compare feedback across all four CEFR levels.

Engines are loaded lazily on first use and cached via st.cache_resource.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup (so imports work regardless of cwd)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, SEED
from annotate.errant_pipe import get_edits
from feedback.feedback_gen import generate_feedback, FeedbackResult
from feedback.templates import get_feedback as get_feedback_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CEFR-Adaptive GEC Feedback",
    page_icon="✏️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        max-width: 1100px;
        margin: 0 auto;
    }
    .error-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: 600;
        margin-right: 4px;
    }
    .error-replacement { background-color: #fde8e8; color: #991b1b; }
    .error-missing     { background-color: #fef3c7; color: #92400e; }
    .error-unnecessary { background-color: #dbeafe; color: #1e40af; }
    .feedback-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .feedback-card-a { border-left-color: #10b981; }
    .feedback-card-b { border-left-color: #3b82f6; }
    .feedback-card-c { border-left-color: #8b5cf6; }
    .feedback-card-n { border-left-color: #6b7280; }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
    }
    .correction-highlight {
        background-color: #fef3c7;
        padding: 2px 4px;
        border-radius: 3px;
        text-decoration: line-through;
        color: #991b1b;
    }
    .correction-fix {
        background-color: #d1fae5;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 600;
        color: #065f46;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Engine loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LanguageTool…")
def load_lt():
    from models.lt_wrapper import LTWrapper
    return LTWrapper(language="en-US")


@st.cache_resource(show_spinner="Loading T5-small…")
def load_tagger():
    from models.tagger_wrapper import TaggerWrapper
    return TaggerWrapper(device="cpu", seed=SEED)


@st.cache_resource(show_spinner="Loading LoRA flan-t5-base…")
def load_lora():
    from models.lora_wrapper import LoRAWrapper
    adapter_path = PROJECT_ROOT / "models" / "lora_flan_t5_base" / "adapter"
    if not adapter_path.exists():
        return None
    return LoRAWrapper(adapter_path=adapter_path, device="cpu", seed=SEED)


def load_priors() -> dict:
    priors_path = RESULTS_DIR / "engine_priors.json"
    if priors_path.exists():
        with open(priors_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Hybrid selection (simplified for single-sentence use)
# ---------------------------------------------------------------------------
MIN_PRECISION = 0.10


def hybrid_correct_single(
    original: str,
    engine_corrections: dict[str, str],
    priors: dict[str, dict[str, float]],
) -> dict:
    """Run edit-level hybrid selection on a single sentence."""
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

    # Greedy non-overlapping selection
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

    # Apply edits
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

    engines_used = sorted({e["engine"] for e in selected})
    return {
        "corrected": corrected,
        "n_edits": len(selected),
        "engines_used": engines_used,
        "selected_edits": selected,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def error_type_badge(etype: str) -> str:
    prefix = etype.split(":")[0]
    cls = {
        "R": "error-replacement",
        "M": "error-missing",
        "U": "error-unnecessary",
    }.get(prefix, "error-replacement")
    return f'<span class="error-tag {cls}">{etype}</span>'


def render_diff(original: str, corrected: str) -> str:
    """Render a simple word-level diff as HTML."""
    orig_words = original.split()
    corr_words = corrected.split()

    # Use a simple LCS-based diff
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, orig_words, corr_words)
    parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.append(" ".join(orig_words[i1:i2]))
        elif op == "replace":
            parts.append(f'<span class="correction-highlight">{" ".join(orig_words[i1:i2])}</span>')
            parts.append(f'<span class="correction-fix">{" ".join(corr_words[j1:j2])}</span>')
        elif op == "delete":
            parts.append(f'<span class="correction-highlight">{" ".join(orig_words[i1:i2])}</span>')
        elif op == "insert":
            parts.append(f'<span class="correction-fix">{" ".join(corr_words[j1:j2])}</span>')
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Header
    st.title("✏️ CEFR-Adaptive GEC Feedback")
    st.markdown(
        "**Error-Type-Aware, CEFR-Adaptive Feedback for L2 Writing**  \n"
        "Enter a sentence, select your proficiency level, and receive "
        "personalised grammar corrections."
    )

    st.divider()

    # Sidebar: configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        cefr = st.selectbox(
            "CEFR level",
            options=["A", "B", "C", "N"],
            index=0,
            help=(
                "A = Beginner (detailed, scaffolded feedback)\n\n"
                "B = Intermediate (grammar terminology)\n\n"
                "C = Advanced (concise, coded feedback)\n\n"
                "N = Proficient/Native (minimal flagging)"
            ),
        )

        st.divider()

        engine_mode = st.radio(
            "Correction engine",
            options=["Hybrid (recommended)", "LanguageTool only", "LoRA only"],
            index=0,
            help="Hybrid fuses edits from all available engines using precision priors.",
        )

        st.divider()

        show_comparison = st.checkbox(
            "Show all CEFR levels",
            value=False,
            help="Display feedback at all four CEFR levels for comparison.",
        )

        show_edits = st.checkbox(
            "Show ERRANT edits",
            value=False,
            help="Display raw ERRANT edit annotations.",
        )

        st.divider()
        st.caption(
            "MSc Computational Linguistics Dissertation  \n"
            "Seed: 42 | ERRANT 3.0 | spaCy 3.7.5"
        )

    # Main input
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.subheader("Input")
        text = st.text_area(
            "Type or paste your text:",
            height=200,
            placeholder="Example: I has been living in London since five years and I am agree that the transport is very good.",
        )

    # Example sentences
    st.markdown("**Try an example:**")
    examples = [
        "I has been living in London since five years.",
        "She go to the university yesterday and buyed some books.",
        "The informations that he gived me was very usefull.",
        "I am agree with this opinion because it is more better.",
        "He speaked to the informations desk about his luggages.",
    ]
    example_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with example_cols[i]:
            if st.button(f"Ex {i+1}", key=f"ex_{i}", use_container_width=True,
                         help=ex):
                st.session_state["input_text"] = ex
                st.rerun()

    # Handle example selection
    if "input_text" in st.session_state:
        text = st.session_state.pop("input_text")

    if not text or not text.strip():
        st.info("Enter some text above and press **Ctrl+Enter** to get feedback.")
        return

    # Run correction
    with st.spinner("Analysing…"):
        t0 = time.time()

        # Split into sentences (simple split on '. ' or use spaCy)
        sentences = [s.strip() for s in text.strip().split("\n") if s.strip()]
        if not sentences:
            return

        priors = load_priors()
        all_feedback: list[FeedbackResult] = []
        all_hybrid_info: list[dict] = []

        for sent in sentences:
            engine_corrections = {}

            if engine_mode == "Hybrid (recommended)":
                # Load available engines
                lt = load_lt()
                lt_result = lt.correct(sent)
                engine_corrections["languagetool"] = lt_result["corrected"]

                lora = load_lora()
                if lora is not None:
                    lora_result = lora.correct(sent)
                    engine_corrections["lora-flan-t5-base"] = lora_result["corrected"]

                # Run hybrid selection
                hybrid = hybrid_correct_single(sent, engine_corrections, priors)
                corrected = hybrid["corrected"]
                all_hybrid_info.append(hybrid)

            elif engine_mode == "LanguageTool only":
                lt = load_lt()
                result = lt.correct(sent)
                corrected = result["corrected"]
                all_hybrid_info.append({
                    "corrected": corrected,
                    "n_edits": 0,
                    "engines_used": ["languagetool"],
                    "selected_edits": [],
                })

            else:  # LoRA only
                lora = load_lora()
                if lora is None:
                    st.error("LoRA adapter not found. Place it in models/lora_flan_t5_base/adapter/")
                    return
                result = lora.correct(sent)
                corrected = result["corrected"]
                all_hybrid_info.append({
                    "corrected": corrected,
                    "n_edits": 0,
                    "engines_used": ["lora-flan-t5-base"],
                    "selected_edits": [],
                })

            fb = generate_feedback(sent, corrected, cefr)
            all_feedback.append(fb)

        elapsed = time.time() - t0

    # Display results
    with col_output:
        st.subheader("Corrected Text")
        for fb in all_feedback:
            if fb.original != fb.corrected:
                st.markdown(render_diff(fb.original, fb.corrected),
                            unsafe_allow_html=True)
            else:
                st.success(fb.corrected)

    st.divider()

    # Metrics row
    total_errors = sum(fb.n_errors for fb in all_feedback)
    engines_used = set()
    for h in all_hybrid_info:
        engines_used.update(h.get("engines_used", []))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Errors Found", total_errors)
    with m2:
        st.metric("Sentences", len(sentences))
    with m3:
        st.metric("Engines Used", len(engines_used))
    with m4:
        st.metric("Time", f"{elapsed:.1f}s")

    st.divider()

    # Feedback section
    st.subheader(f"Feedback — CEFR Level {cefr}")

    cefr_labels = {"A": "Beginner", "B": "Intermediate", "C": "Advanced", "N": "Proficient"}
    cefr_colours = {"A": "feedback-card-a", "B": "feedback-card-b",
                    "C": "feedback-card-c", "N": "feedback-card-n"}

    for fb in all_feedback:
        if fb.n_errors == 0:
            st.success(f"**✓** {fb.original}  — No errors detected.")
            continue

        for item in fb.items:
            badge = error_type_badge(item.error_type)
            card_class = cefr_colours.get(cefr, "feedback-card-b")
            st.markdown(
                f'<div class="feedback-card {card_class}">'
                f'{badge} {item.feedback_text}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Optional: ERRANT edits
    if show_edits:
        st.divider()
        st.subheader("ERRANT Edit Details")
        for i, fb in enumerate(all_feedback):
            if fb.n_errors == 0:
                continue
            edits = get_edits(fb.original, fb.corrected)
            real = [e for e in edits if e["type"] != "noop"]
            if real:
                st.markdown(f"**Sentence {i+1}:** {fb.original}")
                for e in real:
                    engine_src = ""
                    for se in all_hybrid_info[i].get("selected_edits", []):
                        if se["o_start"] == e["o_start"] and se["o_end"] == e["o_end"]:
                            engine_src = f" ← *{se['engine']}* (prior: {se['score']:.2f})"
                            break
                    st.markdown(
                        f"- `[{e['o_start']}:{e['o_end']}]` "
                        f"'{e['o_str']}' → '{e['c_str']}' "
                        f"(**{e['type']}**){engine_src}"
                    )

    # Optional: CEFR comparison
    if show_comparison:
        st.divider()
        st.subheader("CEFR Level Comparison")

        for fb in all_feedback:
            if fb.n_errors == 0:
                continue

            st.markdown(f"**Original:** {fb.original}")

            tabs = st.tabs(["A — Beginner", "B — Intermediate", "C — Advanced", "N — Proficient"])
            for tab, level in zip(tabs, ["A", "B", "C", "N"]):
                with tab:
                    comp_fb = generate_feedback(fb.original, fb.corrected, level)
                    for item in comp_fb.items:
                        badge = error_type_badge(item.error_type)
                        card_class = cefr_colours.get(level, "feedback-card-b")
                        st.markdown(
                            f'<div class="feedback-card {card_class}">'
                            f'{badge} {item.feedback_text}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
