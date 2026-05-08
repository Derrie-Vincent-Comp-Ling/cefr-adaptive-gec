"""Capture screenshots of the Streamlit demonstrator at each CEFR level.

Prerequisites (run once):
    pip install playwright
    playwright install chromium

Usage:
    # In one terminal:
    streamlit run app/demo.py

    # In another terminal (wait until Streamlit is ready):
    python3 app/capture_streamlit_screenshots.py

Outputs:
    writing/screenshots/streamlit_level_A.png
    writing/screenshots/streamlit_level_B.png
    writing/screenshots/streamlit_level_C.png
    writing/screenshots/streamlit_level_N.png

Notes:
    - The script interacts with the Streamlit page via label-based selectors,
      so if you change the sidebar labels in app/demo.py you will need to
      update the SELECTORS dictionary below.
    - Increase STABILISE_MS if you see truncated output (some models take a
      few seconds to run inference on the first call).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

STREAMLIT_URL = "http://localhost:8501"
TEST_SENTENCE = "She go to school yesterday ."
STABILISE_MS = 6000

SELECTORS = {
    "text_input": "textarea",           # First textarea in the page
    "cefr_radio_label": "CEFR level",   # Label of the radio group
    "mode_label": "Correction mode",    # Label of the correction mode group
    "hybrid_option": "Hybrid",
    "submit_button": "Correct",
}

OUT_DIR = Path(__file__).resolve().parents[1] / "writing" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.stderr.write(
            "Playwright is not installed. Run:\n"
            "    pip install playwright\n"
            "    playwright install chromium\n"
        )
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1800})
        page = context.new_page()

        for level in ("A", "B", "C", "N"):
            log.info("Capturing CEFR level %s ...", level)
            page.goto(STREAMLIT_URL, wait_until="networkidle")

            # Type sentence
            page.fill(SELECTORS["text_input"], TEST_SENTENCE)

            # Select CEFR level (assumes a radio group labelled "CEFR level")
            page.get_by_label(level, exact=True).check()

            # Select Hybrid correction mode
            try:
                page.get_by_label(SELECTORS["hybrid_option"], exact=True).check()
            except Exception:
                pass  # default is already hybrid

            # Click the submit button
            page.get_by_role("button", name=SELECTORS["submit_button"]).click()

            # Wait for the inference to stabilise
            page.wait_for_timeout(STABILISE_MS)

            out = OUT_DIR / f"streamlit_level_{level}.png"
            page.screenshot(path=str(out), full_page=True)
            log.info("Saved %s", out)

        browser.close()


if __name__ == "__main__":
    main()
