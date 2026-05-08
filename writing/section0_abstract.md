# Abstract

## Background

Grammatical Error Correction (GEC) is an established Natural Language Processing task used in automated writing support for Computer-Assisted Language Learning (CALL). However, current GEC systems often over-correct learners' work, producing fluent rewrites that treat all errors uniformly and do not adapt feedback to learner proficiency, limiting their pedagogical effectiveness for L2 writing instruction.

## Aims

This dissertation presents an error-type-aware, CEFR-adaptive GEC system that improves both correction precision and the relevance of feedback to the learner's language level. It addresses two questions: (1) whether selecting each correction from whichever system is most accurate for its error type outperforms any single system used alone; and (2) what error types each correction approach handles best, and how these differences can be exploited.

## Method

A hybrid system combines a rule-based checker (LanguageTool), a general-purpose language model (T5-small), and a task-adapted model (LoRA flan-t5-base). A hybrid selector routes each edit to whichever engine has the highest empirical precision for that ERRANT error type, and feedback is rendered at four CEFR registers (A to N). Training used 33,391 sentences from W&I+LOCNESS, with external validation on JFLEG.

## Key Results

On 827 previously unseen sentences containing 1,200 gold errors, the hybrid achieved precision 0.4420, recall 0.0667, and F₀.₅ 0.2079 — a 248% improvement over the best individual engine (LanguageTool, F₀.₅=0.0598). Per-error-type analysis confirmed the three engines are genuinely complementary: each attains highest precision on a distinct subset of ERRANT categories that the hybrid selector successfully exploits.

**Keywords:** Grammatical Error Correction; Computer-Assisted Language Learning; Error-Type-Aware Systems; CEFR Adaptation; Hybrid Neural–Symbolic Methods.
