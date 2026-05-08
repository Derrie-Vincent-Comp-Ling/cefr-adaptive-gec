"""LanguageTool baseline wrapper for grammatical error correction.

Provides a simple correct() / batch_correct() interface using
language_tool_python with English rules.
"""
from __future__ import annotations

import logging
from typing import Any

import language_tool_python
from Levenshtein import ratio as lev_ratio

log = logging.getLogger(__name__)


class LTWrapper:
    """Thin wrapper around LanguageTool for GEC."""

    ENGINE = "languagetool"

    def __init__(self, language: str = "en-US") -> None:
        """Initialise LanguageTool.

        Defaults to en-US for compatibility with LanguageTool 6.4+.
        Falls back to 'en' if the requested variant fails.
        """
        log.info("Initialising LanguageTool (%s) …", language)
        try:
            self.tool = language_tool_python.LanguageTool(language)
        except Exception as e:
            log.warning("Failed with %s (%s), falling back to 'en'", language, e)
            self.tool = language_tool_python.LanguageTool("en")
        self.language = language
        log.info("LanguageTool ready")

    # ------------------------------------------------------------------
    # Single sentence
    # ------------------------------------------------------------------
    def correct(self, sentence: str) -> dict[str, Any]:
        """Return a dict with engine, corrected text, and edit_ratio.

        edit_ratio is the Levenshtein *similarity* ratio (1.0 = identical).
        """
        try:
            corrected = self.tool.correct(sentence)
        except Exception as e:
            log.warning("LT error on sentence (returning original): %s", e)
            corrected = sentence
        edit_r = lev_ratio(sentence, corrected)
        return {
            "engine": self.ENGINE,
            "corrected": corrected,
            "edit_ratio": round(edit_r, 6),
        }

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def batch_correct(
        self,
        sentences: list[str],
        log_every: int = 500,
    ) -> list[dict[str, Any]]:
        """Process a list of sentences with a progress counter."""
        results: list[dict[str, Any]] = []
        total = len(sentences)
        for i, sent in enumerate(sentences, 1):
            results.append(self.correct(sent))
            if i % log_every == 0 or i == total:
                log.info("  [LT] %d / %d sentences processed", i, total)
        return results

    def close(self) -> None:
        self.tool.close()
