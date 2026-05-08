# 4. Methodology

## 4.1 Research Design Overview

This project adopts a mixed-method experimental design that combines automated corpus-based evaluation with a planned human judgement study. The automated component uses the ERRANT scoring toolkit to measure correction quality through precision (the proportion of system-produced edits that are correct), recall (the proportion of true errors the system successfully corrects), and their weighted harmonic mean F0.5 (Bryant, Felice and Briscoe, 2017). The human component complements these quantitative metrics by assessing the pedagogical adequacy of the CEFR-adaptive explanations through Likert-scale ratings. The two evaluation axes map directly onto the two central research questions: whether error-type-aware engine selection improves correction precision (RQ1), and whether CEFR-adapted feedback is perceived as more useful than uniform feedback (RQ2).

Five system conditions are compared, as summarised in Table 4.1. Three are individual correction systems representing distinct paradigms — rule-based pattern matching, a general-purpose neural language model, and a task-specific fine-tuned model — and two are composite systems that combine outputs from the individual systems. The ablation condition (Hybrid-NoPrior) applies the same fusion algorithm as the full hybrid but replaces precision priors with uniform weights, isolating the contribution of error-type-aware routing.

**Table 4.1. System conditions evaluated in this project.**

| ID | System | Engine(s) | Description |
|----|--------|-----------|-------------|
| S1 | LanguageTool | language-tool-python 2.8 (en-US) | Rule-based baseline; deterministic pattern matching |
| S2 | T5-small-GEC | t5-small (Raffel et al., 2020) | Vanilla seq2seq baseline; prompt "Fix grammar: {s}" |
| S3 | LoRA-flan-t5-base | flan-t5-base + LoRA adapter | Fine-tuned with PEFT; prompt "Fix grammatical errors: {s}" |
| S4 | Hybrid | All three engines | Edit-level fusion with per-type precision priors |
| S5 | Hybrid-NoPrior | All three engines | Ablation: uniform prior (p = 0.5) for all types |

## 4.2 Correction Engine Configuration

### 4.2.1 LanguageTool Baseline (S1)

LanguageTool (Naber, 2003) is an open-source rule-based grammar checker employing hand-crafted XML pattern rules and a part-of-speech tagger. The system was initialised with the en-US language variant via the language-tool-python 2.8 wrapper (`models/lt_wrapper.py`). English-US was selected over en-GB for compatibility with LanguageTool 6.4, which exhibited a rule-activation failure under the en-GB locale. Each input sentence was corrected independently through the `correct()` method, which applies all matching rules and returns the corrected string. No rule filtering or category disabling was applied, preserving the full rule set as a genuine out-of-the-box baseline.

### 4.2.2 T5-small Neural Baseline (S2)

The T5-small model (Raffel et al., 2020), with 60 million parameters, served as a vanilla neural baseline. The input prompt format was "Fix grammar: {sentence}", following the task-prefix convention of the T5 text-to-text framework, in which every task is cast as a string-to-string transformation cued by an instruction prefix. Generation used beam search — a decoding procedure that maintains a small number of partial candidate sequences at each step rather than committing greedily to a single best word — with two beams and a maximum output length of 256 tokens (`models/tagger_wrapper.py`). This model was not fine-tuned on GEC data; its purpose is to measure how much an off-the-shelf language model can achieve through prompt-based correction alone. T5-small was preferred over the GECToR tagger architecture (Omelianchuk et al., 2020) because the gotutiyan/gector package requires transformers version 5.x, which conflicts with the project's pinned transformers 4.44.0 needed for PEFT and LoRA compatibility.

### 4.2.3 LoRA Fine-Tuned Model (S3)

The primary neural correction engine is a flan-t5-base model (Chung et al., 2022) fine-tuned with Low-Rank Adaptation (LoRA; Hu et al., 2022). Flan-t5-base was selected over its larger siblings for practical trainability on a single free-tier Google Colab GPU (NVIDIA Tesla T4 with 15 GB of video memory), which imposes a strict ceiling on the number of model parameters that can be updated simultaneously. LoRA addresses this constraint by freezing the 250-million-parameter base model and learning instead a small pair of low-rank matrices that approximate the required weight updates; only these low-rank matrices are trained, which reduces the memory and compute footprint by two orders of magnitude while retaining most of the accuracy of full fine-tuning.

The LoRA configuration was defined as follows:

**Table 4.2. LoRA hyperparameters for flan-t5-base fine-tuning.**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Rank (r) | 16 | Balances expressiveness and parameter count |
| Alpha (α) | 32 | Scaling factor; α/r = 2 is a common ratio |
| Dropout | 0.05 | Light regularisation to prevent overfitting |
| Target modules | q, v | Query and value projections (standard LoRA practice) |
| Task type | SEQ_2_SEQ_LM | Matches the encoder–decoder architecture |
| Learning rate | 5 × 10⁻⁵ | AdamW with linear warmup |
| Batch size | 8 | Reduced from 16 to fit Colab T4 memory |
| Epochs | 5 | Selected after monitoring training loss convergence |
| Max source length | 128 tokens | Covers >99% of sentences in W&I+LOCNESS |
| Max target length | 128 tokens | Symmetric with source length |
| Precision | FP16 | Mixed-precision training for memory efficiency |
| Seed | 42 | Deterministic across all libraries |

The input prompt prefix was "Fix grammatical errors: ", consistent with the instruction-following format of flan-T5. Training used the AdamW optimiser with a linear warm-up schedule — in which the learning rate rises gradually from zero over the first few update steps before reverting to a standard linear decay — and mixed-precision FP16 arithmetic, which stores activations in 16-bit floating-point format to halve memory use relative to full FP32. Training was conducted on the 34,308 source–target pairs from `data/processed/train_pairs.jsonl` using the Hugging Face Seq2SeqTrainer with the PEFT 0.12.0 library. Training loss converged to approximately 0.0000 after five epochs in 4,681 seconds. The resulting adapter weights (1.8 MB) were saved to `models/lora_flan_t5_base/adapter/` and loaded at inference time via PEFT's `PeftModel.from_pretrained` method. Inference used beam search with four beams (`models/lora_wrapper.py`).

## 4.3 Hybrid Constrained Selection Algorithm

The hybrid selector (`selector/hybrid_selector.py`) implements an edit-level fusion strategy that selects individual corrections from whichever engine has the highest empirically measured precision for the relevant error type. This approach draws on the observation that different GEC architectures exhibit complementary strengths: rule-based systems tend to excel at well-defined closed-class errors (e.g., preposition selection, determiner insertion), while neural models capture more complex morphosyntactic patterns (e.g., verb tense sequencing, subject–verb agreement across clausal boundaries).

The algorithm proceeds in four stages. First, for each input sentence, the system obtains corrected outputs from all available engines. Second, ERRANT is applied to each (original, corrected) pair to extract a set of typed edits, where each edit is a tuple consisting of the original token span, the replacement tokens, and an ERRANT error-type code. These codes follow a two-part convention: an operation prefix — M for missing-token insertions, U for unnecessary-token deletions, and R for replacements — followed by a linguistic category such as DET (determiner), PREP (preposition), or VERB:SVA (subject–verb agreement). Thus R:VERB:SVA denotes the replacement of a verb form because of a subject–verb agreement mismatch, and M:DET denotes the insertion of a missing determiner. Third, each candidate edit is assigned a precision score by looking up the originating engine's empirical precision for that error type in a pre-computed prior table (`results/engine_priors.json`). These priors were estimated on the dev-tune partition by running each engine on 3,022 matched sentence pairs and computing, for every error type, the ratio of true-positive edits to all edits proposed by that engine for that type. Edits whose precision score falls below a minimum threshold of 0.10 are discarded, filtering out error types for which no engine has demonstrated reliable performance.

Fourth, the surviving candidates are sorted in descending order of precision score and selected greedily under a non-overlap constraint: an edit is accepted only if its original-token span does not intersect with any previously accepted edit's span. This constraint prevents contradictory corrections from different engines from being applied to the same tokens. When all candidates have been considered, the accepted edits are applied to the original sentence in right-to-left order to avoid index-shift errors, producing the hybrid-corrected output.

The prior table reveals clear complementarities among engines. LanguageTool achieves precision of 1.0 on R:PRON and U:CONJ, 0.67 on M:PREP and R:PREP, and 0.80 on U:DET. The LoRA model achieves precision of 1.0 on R:DET, R:PRON, R:VERB:INFL, R:VERB:TENSE, and U:PREP. T5-small, despite lower overall precision, leads on R:ADJ:FORM (1.0) and R:VERB:INFL (1.0). The hybrid selector exploits these complementarities by routing each error type to its most precise engine, which is the mechanism that produces its substantially higher corpus-level precision relative to any individual engine.

## 4.4 Ablation Design

To quantify the contribution of error-type-aware precision priors, an ablation condition (S5, Hybrid-NoPrior) replaces the empirical prior lookup with a uniform prior of 0.5 for all error types and engines. All other algorithmic components — ERRANT edit extraction, non-overlap constraint, right-to-left application — remain identical. Comparing S4 against S5 isolates the effect of informed engine routing from the structural benefit of edit-level fusion itself. A statistically significant difference in F0.5 between S4 and S5 would confirm that the precision priors, rather than simple candidate pooling, are responsible for the observed improvement.

## 4.5 CEFR-Adaptive Explanation Evaluation

### 4.5.1 Feedback Conditions

The feedback module (`feedback/feedback_gen.py`) generates explanations for each ERRANT-typed edit using templates stored in `feedback/templates.py`. Four feedback styles correspond to the four CEFR levels, grounded in the corrective feedback typology of Ellis (2009) and the scaffolding principles of Bitchener and Storch (2016):

- **Level A (Direct corrective feedback):** Explicit correction shown alongside a plain-language rule explanation and, where applicable, an illustrative example. Metalinguistic terminology is avoided.
- **Level B (Metalinguistic feedback):** Grammar terminology is introduced (e.g., "subject–verb agreement", "determiner"), the rule is explained, and the correction is provided.
- **Level C (Indirect coded feedback):** Concise feedback using linguistic terminology (e.g., "SVA error", "prepositional collocation mismatch"). The learner is expected to understand the error category.
- **Level N (Minimal flagging):** The error type and correction are shown with no further explanation, assuming the writer can self-diagnose.

### 4.5.2 Planned Evaluation Protocol

A Likert-scale evaluation is designed to assess perceived usefulness of the adaptive explanations. A sample of 120 sentences is drawn from the dev-eval partition, stratified by CEFR level (30 per level) and filtered to include only sentences where the hybrid system made at least one correction. Each sentence is presented to raters in three conditions: (i) feedback matched to the sentence's CEFR level (adaptive), (ii) feedback at a fixed intermediate level B (uniform), and (iii) the corrected sentence alone with no explanation (correction-only). Raters judge each condition on a five-point Likert scale across three dimensions: clarity ("I understand why this correction was made"), helpfulness ("This feedback would help me avoid the same error"), and appropriateness ("The level of detail is right for this learner").

This design yields 120 × 3 = 360 stimulus–condition pairs. A within-subjects design with counterbalanced presentation order controls for item effects by ensuring every rater sees every condition while varying the order in which they are presented. Inter-rater reliability — the extent to which independent raters produce consistent judgements — will be assessed using Krippendorff's alpha (Krippendorff, 2011), a chance-corrected agreement statistic suitable for ordinal data and variable numbers of raters, with a threshold of α ≥ 0.67 adopted as the minimum for acceptable agreement.

## 4.6 Metrics and Statistical Tests

### 4.6.1 Automated Correction Metrics

The primary correction metric is ERRANT F0.5 (Bryant, Felice and Briscoe, 2017), implemented in `annotate/errant_pipe.py`. F0.5 weights precision twice as heavily as recall (β = 0.5), reflecting the pedagogical principle that incorrect corrections are more harmful to learners than missed errors (Chollampatt and Ng, 2018). Precision, recall, and F0.5 are computed at the corpus level by aggregating true-positive, false-positive, and false-negative edit counts across all sentences before computing ratios, following the BEA-2019 shared-task protocol (Bryant et al., 2019). Per-error-type F0.5 is additionally reported to identify the error categories most and least improved by the hybrid approach.

The Levenshtein edit ratio (Levenshtein, 1966) is reported as a secondary measure of correction conservativeness, computed as the character-level similarity between the original and corrected strings, where the underlying Levenshtein distance counts the minimum number of single-character insertions, deletions, or substitutions needed to transform one string into the other. Values near 1.0 indicate minimal surface change, which in a GEC context is desirable because it suggests the system is making targeted corrections rather than rewriting sentences wholesale.

For JFLEG evaluation, the GLEU metric (Napoles et al., 2015) is used, which computes a modified BLEU score penalising both under-correction and over-correction relative to multiple reference corrections.

### 4.6.2 Statistical Testing

Differences in F0.5 between system conditions are assessed using McNemar's test on per-sentence binary outcomes (correct or incorrect edit match), following Chollampatt and Ng (2018). McNemar's test is the appropriate non-parametric test for comparing two classifiers on the same items because it conditions on the discordant cases — those sentences where exactly one of the two systems produced the correct edit — thereby controlling for item-level difficulty. For the Likert-scale data from the explanation evaluation, the Wilcoxon signed-rank test is applied to paired adaptive-versus-uniform ratings within each CEFR level; this is the non-parametric analogue of the paired t-test and makes no assumption of normality, which is appropriate for ordinal rating data. Effect sizes are reported as the rank-biserial correlation (r). All tests use α = 0.05 with Bonferroni correction for multiple comparisons.

## 4.7 Reproducibility Statement

All experiments were conducted with a fixed random seed of 42, set in Python's `random` module, NumPy, PyTorch (including CUDA deterministic mode), and the Hugging Face Transformers library. The full software stack is pinned in `requirements.txt`: Python 3.11, PyTorch 2.3.0, Transformers 4.44.0, PEFT 0.12.0, ERRANT 3.0.0, spaCy 3.7.5, and language-tool-python 2.8. Data splits are fully deterministic via `GroupShuffleSplit` with the logged essay-level partition in `logs/split_ids.json`. The LoRA adapter weights are archived at `models/lora_flan_t5_base/adapter/`. All intermediate outputs — normalised JSONL files, prediction files, engine priors, and evaluation metrics — are stored as JSONL or JSON in the `results/` and `data/processed/` directories, enabling independent verification of every pipeline stage.

## References

Bitchener, J. and Storch, N. (2016) *Written Corrective Feedback for L2 Development*. Bristol: Multilingual Matters.

Bryant, C., Felice, M. and Briscoe, T. (2017) 'Automatic annotation and evaluation of error types for grammatical error correction', in *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Vancouver: Association for Computational Linguistics, pp. 793–805.

Bryant, C. et al. (2019) 'The BEA-2019 shared task on grammatical error correction', in *Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications*. Florence: Association for Computational Linguistics, pp. 52–75.

Chollampatt, S. and Ng, H.T. (2018) 'A reassessment of reference-based grammatical error correction metrics', in *Proceedings of the 27th International Conference on Computational Linguistics*. Santa Fe: Association for Computational Linguistics, pp. 2730–2741.

Chung, H.W. et al. (2022) 'Scaling instruction-finetuned language models', *arXiv preprint arXiv:2210.11416*.

Ellis, R. (2009) 'A typology of written corrective feedback types', *ELT Journal*, 63(2), pp. 97–107.

Hu, E.J. et al. (2022) 'LoRA: Low-Rank Adaptation of Large Language Models', in *Proceedings of the Tenth International Conference on Learning Representations*. Virtual: OpenReview.net.

Krippendorff, K. (2011) *Computing Krippendorff's Alpha-Reliability*. Departmental Paper. Philadelphia: Annenberg School for Communication, University of Pennsylvania.

Levenshtein, V.I. (1966) 'Binary codes capable of correcting deletions, insertions, and reversals', *Soviet Physics Doklady*, 10, pp. 707–710.

Naber, D. (2003) *A Rule-Based Style and Grammar Checker*. Diploma thesis, Bielefeld University.

Napoles, C. et al. (2015) 'Ground truth for grammatical error correction metrics', in *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*. Beijing: Association for Computational Linguistics, pp. 588–593.

Omelianchuk, K. et al. (2020) 'GECToR — grammatical error correction: tag, not rewrite', in *Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications*. Online: Association for Computational Linguistics, pp. 163–170.

Raffel, C. et al. (2020) 'Exploring the limits of transfer learning with a unified text-to-text transformer', *Journal of Machine Learning Research*, 21(140), pp. 1–67.
