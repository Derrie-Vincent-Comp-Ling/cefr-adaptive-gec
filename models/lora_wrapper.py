"""LoRA flan-t5-base wrapper for grammatical error correction.

Loads the PEFT LoRA adapter trained in Phase 3c on top of
google/flan-t5-base and exposes the same correct() / batch_correct()
interface as the other engine wrappers.

Reference:
    Hu, E.J. et al. (2022) 'LoRA: Low-Rank Adaptation of Large Language
    Models', ICLR 2022.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from Levenshtein import ratio as lev_ratio
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

log = logging.getLogger(__name__)

# Default adapter path (relative to project root)
_DEFAULT_ADAPTER = Path(__file__).resolve().parent / "lora_flan_t5_base" / "adapter"


class LoRAWrapper:
    """LoRA flan-t5-base GEC engine."""

    ENGINE = "lora-flan-t5-base"
    PREFIX = "Fix grammatical errors: "

    def __init__(
        self,
        adapter_path: str | Path = _DEFAULT_ADAPTER,
        base_model: str = "google/flan-t5-base",
        device: str = "cpu",
        seed: int = 42,
        num_beams: int = 4,
        max_length: int = 256,
    ) -> None:
        self.device = device
        self.seed = seed
        self.num_beams = num_beams
        self.max_length = max_length
        adapter_path = Path(adapter_path)

        torch.manual_seed(seed)
        log.info("Loading base model %s …", base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        log.info("Loading LoRA adapter from %s …", adapter_path)
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.eval()
        self.model.to(device)
        log.info("LoRA flan-t5-base ready on %s", device)

    # ------------------------------------------------------------------
    # Single sentence
    # ------------------------------------------------------------------
    def correct(self, sentence: str) -> dict[str, Any]:
        """Return dict with engine, corrected text, and edit_ratio."""
        prompt = f"{self.PREFIX}{sentence}"
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
                log.info("  [LoRA] %d / %d sentences processed", i, total)
        return results
