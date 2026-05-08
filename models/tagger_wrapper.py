"""Tagger-style baseline wrapper for grammatical error correction.

Approach:
    Uses T5-small (Raffel et al., 2020) with the prompt 'Fix grammar: {sentence}'
    and num_beams=2 to produce conservative seq2seq corrections.

    This serves as a neural baseline alongside the rule-based LanguageTool baseline.
    T5-small was chosen over GECToR because the gotutiyan/gector package requires
    transformers>=5.x which conflicts with the project's pinned transformers==4.44.0
    needed for PEFT/LoRA fine-tuning downstream. T5-small is natively supported
    by the pinned stack and provides a comparable conservative correction baseline.

    Reference: Raffel, C. et al. (2020) 'Exploring the limits of transfer learning
    with a unified text-to-text transformer', JMLR, 21(140), pp. 1–67.

Exposes the same interface as LTWrapper: correct() and batch_correct().
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from Levenshtein import ratio as lev_ratio
from transformers import T5ForConditionalGeneration, AutoTokenizer

log = logging.getLogger(__name__)


class TaggerWrapper:
    """T5-small seq2seq baseline for GEC."""

    ENGINE = "t5-small-gec"

    def __init__(
        self,
        model_name: str = "t5-small",
        device: str = "cpu",
        seed: int = 42,
        num_beams: int = 2,
        max_length: int = 256,
    ) -> None:
        self.device = device
        self.seed = seed
        self.num_beams = num_beams
        self.max_length = max_length
        self.engine_name = self.ENGINE

        torch.manual_seed(seed)
        log.info("Loading %s for GEC baseline …", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        log.info("%s ready on %s", model_name, device)

    # ------------------------------------------------------------------
    # Single sentence
    # ------------------------------------------------------------------
    def correct(self, sentence: str) -> dict[str, Any]:
        """Return dict with engine, corrected text, and edit_ratio."""
        prompt = f"Fix grammar: {sentence}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        edit_r = lev_ratio(sentence, corrected)
        return {
            "engine": self.engine_name,
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
                log.info("  [T5] %d / %d sentences processed", i, total)
        return results
