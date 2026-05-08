# 2. Context and Related Work

This section reviews the theoretical and empirical foundations underlying the present work on error-type-aware, CEFR-adaptive feedback for L2 writing. We begin with the evolution of grammatical error correction (GEC) systems, before examining recent advances in parameter-efficient fine-tuning, system combination strategies, and the integration of learner proficiency. Finally, we situate our work within theories of corrective feedback and second language acquisition, highlighting the gap between current GEC systems and learner-centred pedagogical requirements.

## 2.1 Grammatical Error Correction

Grammatical error correction has evolved substantially over the past two decades, transitioning from rule-based systems to statistical machine learning approaches, and most recently to neural and pre-trained language models. This evolution reflects broader trends in natural language processing and has culminated in systems of considerable sophistication, yet significant tensions remain between precision and recall—particularly relevant for educational applications.

### Rule-Based and Early Statistical Approaches

Early GEC systems were predominantly rule-based, employing hand-crafted linguistic rules to identify and correct errors. While interpretable and capable of achieving high precision on well-defined error classes, these systems suffered from limited recall and poor generalisability to unseen error phenomena. The shift toward statistical approaches in the early 2000s leveraged machine translation (MT) and language modelling techniques, with systems such as those developed by Gamon (2010) and Dahlmeier and Ng (2011) applying phrase-based SMT to the error correction task, framing it as translating erroneous text to correct text.

The critical turning point came with the establishment of shared tasks and standardised evaluation frameworks. The CoNLL-2013 and CoNLL-2014 shared tasks (Ng et al., 2013; Ng et al., 2014) provided the GEC community with a benchmark corpus of student essays annotated with corrections and established evaluation protocols. These shared tasks catalysed rapid progress and enabled systematic comparison of approaches. Following CoNLL-2014, the BEA-2019 shared task (Bryant et al., 2019) built upon these foundations with a larger, more diverse corpus spanning multiple proficiency levels and error types, further advancing the field's understanding of GEC challenges.

### Neural Approaches and Seq2seq Paradigms

The advent of neural sequence-to-sequence models transformed GEC in the mid-2010s. Junczys-Dowmunt et al. (2018), reframing GEC as a low-resource machine translation task, demonstrated that attention-based encoder-decoder architectures could substantially outperform earlier statistical approaches, particularly when combined with pre-trained word embeddings and language-model rescoring. These models, however, required careful engineering of beam search, language model fusion, and confidence estimation to achieve competitive performance. A parallel line of work showed that large-scale synthetic error generation was essential for neural GEC to reach competitive performance on standard benchmarks: Grundkiewicz et al. (2019) used unsupervised pre-training on synthetically corrupted text, and Stahlberg and Kumar (2021) refined this with tagged corruption models that replicate the distribution of human learner errors.

A paradigm shift occurred with the application of pre-trained transformer models to GEC. Rather than learning error correction from scratch, transfer learning from models pre-trained on large corpora—such as BERT (Devlin et al., 2019)—substantially reduced the data requirements and improved generalisation. Early applications of BERT to GEC adapted the model for both sequence labelling and generation tasks, though sequence labelling—casting GEC as a token classification problem—proved more efficient and effective.

The GECToR approach (Omelianchuk et al., 2020) represents a significant advancement in this direction. Rather than generating complete corrected sentences, GECToR frames GEC as a sequence tagging problem: each token is assigned a tag indicating the type of edit required (deletion, insertion of a specific word, replacement with a specific word, or no change). This edit-based formulation offers several advantages: it is computationally more efficient than generation-based approaches, provides explicit edit operations that enhance interpretability, and proves amenable to iterative correction (applying multiple passes to refine outputs). GECToR achieved state-of-the-art performance on the CoNLL-2014 benchmark and has become a widely adopted baseline in subsequent GEC research.

### The ERRANT Framework and Evaluation

Crucial to progress in GEC has been the development of the ERRANT (ERRor ANnotation Toolkit) framework by Bryant et al. (2017). ERRANT provides a standardised, fine-grained error taxonomy that classifies corrections at the word level into categories such as verb form, subject-verb agreement, noun number, spelling, and punctuation. Beyond annotation, ERRANT enables precise evaluation through alignment algorithms that match gold corrections to system outputs at the edit level, computing precision and recall metrics that account for partial credit and error categorisation.

ERRANT addresses a critical limitation of earlier evaluation: previous approaches used sentence-level scoring (BLEU, METEOR), which treated all corrections equally. By contrast, ERRANT's error-type-aware evaluation permits fine-grained analysis of system performance across error categories, revealing that different systems exhibit markedly different strengths and weaknesses. This insight—that systems are complementary across error types—motivates the system combination approaches discussed in Section 2.3.

### Current State-of-the-Art and the Precision-Recall Trade-off

Contemporary GEC systems, particularly those based on pre-trained language models such as T5 (Raffel et al., 2020), have achieved F0.5 scores (which weight precision twice as heavily as recall) exceeding 70% on the CoNLL-2014 test set. This F0.5 emphasis reflects a deliberate choice in the evaluation framework: in educational contexts, false positives (flagging correct text as erroneous) are often considered more harmful than false negatives (missing actual errors), as they can undermine learner confidence and introduce unwarranted corrections.

However, this precision-recall trade-off represents a central challenge for GEC systems deployed in educational settings. While high precision is desirable to avoid spurious corrections, it often comes at the cost of reduced recall: systems optimised for F0.5 detect only a subset of the errors present in a text. The present work hypothesises that precision requirements should adapt to learner proficiency—advanced learners may benefit from strict, high-precision feedback to encourage self-correction and strategic error management, whilst lower-proficiency learners may benefit from more comprehensive error detection with slightly lower precision to maximise exposure to their characteristic errors. This perspective is explored further in Section 2.4.

## 2.2 Parameter-Efficient Fine-Tuning

The success of pre-trained language models has necessitated development of efficient adaptation strategies. Full fine-tuning of large models on task-specific data is computationally expensive and can lead to catastrophic forgetting. Parameter-efficient fine-tuning (PEFT) methods address these challenges by adapting models whilst updating only a small fraction of parameters.

### Transfer Learning in Pre-trained Models

Transfer learning in NLP, as formalised by Devlin et al. (2019) with BERT, demonstrated that large-scale unsupervised pre-training on language modelling tasks yields representations useful for diverse downstream tasks. Subsequent models such as RoBERTa (Liu et al., 2019), ELECTRA (Clark et al., 2020), and XLNet (Yang et al., 2019) refined the pre-training objective and architecture, whilst models like GPT-2 (Radford et al., 2019) and T5 (Raffel et al., 2020) demonstrated the benefits of scaling to billions of parameters.

Critically, Raffel et al. (2020) showed that a unified transformer-based architecture—the T5 model—could be effectively adapted to diverse NLP tasks through task-specific fine-tuning, including both understanding tasks (classification, question answering) and generation tasks (translation, summarisation). Their work established the principle that large pre-trained models exhibit strong few-shot and fine-tuning capabilities, reducing data requirements for task-specific training.

However, full fine-tuning of billion-parameter models presents practical challenges: storage of multiple model copies, computational cost of backpropagation through all layers, and risk of overfitting on small task-specific datasets. These considerations motivate parameter-efficient alternatives.

### Low-Rank Adaptation (LoRA)

Hu et al. (2022) introduced Low-Rank Adaptation (LoRA), a method that dramatically reduces the number of trainable parameters whilst maintaining performance. Rather than fine-tuning all weight matrices in a pre-trained model, LoRA decomposes weight updates as a product of two low-rank matrices: W' = W + ΔW, where ΔW = AB^T, with A and B being low-rank matrices. By setting the rank r to a small value (typically 8–32), the number of trainable parameters is reduced from O(d_model^2) to O(r × d_model), often achieving 99% parameter reduction.

Hu et al. demonstrated that LoRA achieves comparable performance to full fine-tuning across diverse tasks and model scales, including the 175-billion-parameter GPT-3. Critically, LoRA permits efficient multi-task adaptation: a single pre-trained model can be combined with different low-rank adapters for different tasks, and adapters can be efficiently swapped or composed. This property is relevant for the present work, where separate adapters could be trained for different error types or proficiency levels.

Variants of LoRA have since been proposed, including QLoRA (Dettmers et al., 2023), which quantises the base model to 4-bit precision whilst maintaining LoRA trainability, further reducing memory requirements. Other PEFT approaches—such as prompt tuning (Lester et al., 2021) and prefix tuning (Li and Liang, 2021)—offer alternative parameter-reduction strategies, though LoRA has proven particularly effective and is now standard practice.

### Instruction-Tuned Models and Flan-T5

A complementary development has been the emergence of instruction-tuned models. Chung et al. (2022) introduced Flan-T5, which fine-tunes T5 on a large collection of diverse NLP tasks phrased as natural language instructions. This multi-task instruction tuning improves zero-shot and few-shot performance on unseen tasks and aligns model outputs more closely with human preferences.

For GEC, instruction tuning offers particular advantages: a model can be prompted to "correct the grammatical errors in the following sentence" rather than relying on task-specific architectural choices (such as edit-based tagging). This generality facilitates integration of multiple error types and proficiency levels into a single model through prompting, potentially enabling a single model to serve different user profiles without separate training.

### Applications to Educational NLP

The application of PEFT to educational NLP remains relatively underexplored. Most GEC systems in the literature employ either full fine-tuning or inference from base models with no task-specific adaptation. However, PEFT methods offer significant advantages for educational applications: they enable efficient deployment on resource-constrained environments (schools, low-resource regions), permit rapid adaptation to institution-specific writing conventions or learner populations, and allow composition of multiple error-type-specific or proficiency-level-specific adapters. The present work leverages LoRA to efficiently fine-tune models for error-type-specific correction and proficiency-adaptive feedback generation.

## 2.3 System Combination in GEC

A consistent finding in GEC research is that different systems exhibit complementary strengths: a system excelling at verb-form errors may perform poorly on punctuation, whilst another exhibits the inverse pattern. This observation motivates system combination approaches, which integrate outputs of multiple GEC systems to achieve performance exceeding any single system.

### Ensemble and Reranking Methods

Early system combination in GEC employed ensemble methods. Susanto et al. (2014) demonstrated that simple voting schemes—selecting the most frequently proposed correction for each token—substantially improve performance over individual systems. They further explored edit-distance and decision-tree-based reranking, which learn weights for combining system outputs based on local context.

Subsequent work has refined these approaches. Rozovskaya and Roth (2016) developed a learning-to-rank framework for GEC system combination, treating combination as a classification problem where candidate corrections are ranked by a model trained on held-out data. This approach achieves higher performance than voting on some benchmarks but requires additional annotated data.

More recent approaches exploit the structure of edit operations. Choshen and Abend (2018) demonstrated that single-reference evaluation in GEC systematically under-counts valid corrections, motivating multi-reference evaluation and, in turn, the edit-level combination methods that treat system outputs as sets of token-level edit operations and select among competing edits using learned preferences. This fine-grained approach outperforms coarser sentence-level selection, as it permits mixing edits from different systems within a single sentence.

More recent system-combination work has extended these approaches. Kantor et al. (2019) trained a learning-based combiner over features extracted from multiple GEC outputs; Qorib, Na and Ng (2022) proposed a simple edit-level weighted voting scheme that is competitive with learned combiners; and Tarnavskyi, Chernodub and Omelianchuk (2022) applied ensembling and knowledge distillation to tag-based GEC models. None of these approaches, however, uses empirical per-ERRANT-category precision as a direct inference-time weight — the gap that the present work addresses.

### Complementarity and Error-Type-Specific Strengths

The complementarity of systems across error types has been empirically demonstrated using ERRANT's fine-grained evaluation. For instance, a rule-based system may excel at deterministic errors (subject-verb agreement) whilst a neural system excels at complex, context-dependent errors (word choice). This heterogeneity motivates error-type-aware combination strategies.

However, the literature reveals a significant gap: whilst ERRANT enables precise measurement of error-type-specific performance, existing combination methods do not explicitly leverage error-type information for routing decisions. Current approaches treat all errors equally or use learning-based weighting that may capture error-type effects indirectly but lacks explicit interpretability. The present work addresses this gap by developing a combination method that assigns error-type-specific precision priors to different systems, explicitly routing corrections of particular error types to systems known to perform well on those error types.

### The Complementarity Hypothesis

The hypothesis underlying system combination in GEC is that different architectures, training objectives, and feature sets lead to complementary strengths. A system trained on synthetic data may generalise differently from one trained on authentic learner errors. A rule-based system captures linguistic constraints precisely but may miss errors not anticipated by rule writers. A neural system trained with a generation objective differs fundamentally from one using a tagging objective. The present work extends this hypothesis: within a single system, different error types can be corrected via different strategies (rule-based for deterministic errors, neural for context-dependent errors), and the optimal strategy may depend on learner proficiency.

## 2.4 CEFR and Learner Proficiency

The Common European Framework of Reference for Languages (CEFR) provides a standardised six-level classification of language proficiency (A1, A2, B1, B2, C1, C2), spanning from absolute beginner to near-native fluency. CEFR is widely used in language education across Europe and increasingly globally, and has become a standard reference point for curriculum design, assessment, and learner tracking.

### The CEFR Framework

The CEFR (Council of Europe, 2001) defines proficiency levels through detailed descriptors of communicative competence across four modalities (listening, reading, writing, speaking). For writing in particular, the framework specifies characteristic features of each level: A1 writers produce simple, isolated phrases; A2 writers compose short, connected texts on familiar topics; B1 writers produce clear, connected texts on topics of personal interest; B2 writers produce well-structured texts; C1 writers produce clear, well-structured, detailed text; and C2 writers are indistinguishable from native speakers.

Beyond communicative competence, the CEFR provides implicit profiling of error characteristics at each level. Lower-level writers (A1–A2) exhibit frequent, diverse errors across foundational categories: basic verb conjugation, subject-verb agreement, article usage, and word order. Intermediate writers (B1–B2) show more sophisticated error patterns: irregular verb forms, subtle agreement errors, complex preposition usage, and stylistic issues. Advanced writers (C1–C2) exhibit rare, fine-grained errors in register, collocations, and subtle grammatical nuances. Importantly, lower-level writers' errors are often more deterministic and rule-governed, whilst higher-level writers' errors are more context-dependent and subtle.

### Computational CEFR Classification

Classification of text or learner proficiency according to CEFR levels has become an established task in NLP. Yannakoudakis et al. (2011) developed a system for automatically grading ESOL texts using hand-engineered lexical and grammatical features derived from linguistic analysis. Vajjala and Lõo (2014) extended this line of work by training classifiers for automatic CEFR level prediction on learner text, demonstrating that feature-based models could reliably distinguish adjacent CEFR levels. Subsequently, neural approaches have improved classification accuracy, with pre-trained models such as BERT adapted for this task to achieve substantial improvements over hand-engineered feature sets.

Critically, these classification systems rely on error patterns: lower-level writing contains more frequent errors across more error categories, with characteristic error types. Thus, CEFR level is a useful proxy for learner proficiency and can inform educational feedback strategies.

### The W&I+LOCNESS Corpus

The Cambridge English Write & Improve (W&I) dataset, augmented with the LOCNESS corpus, provides a large collection of learner essays annotated with both corrections (for error identification and correction) and CEFR proficiency levels. This dataset, introduced by Bryant et al. (2019) and now standard in GEC research, enables training of proficiency-aware systems and evaluation across proficiency levels.

Analysis of W&I+LOCNESS reveals clear proficiency-dependent error patterns. Lower-level learners produce more frequent errors in agreement, verb forms, and article usage—errors that are highly predictable from local context. Advanced learners' errors are distributed more evenly across categories and often involve subtle semantic or stylistic issues. This heterogeneity motivates the hypothesis that GEC systems should adapt to proficiency level.

### Why Proficiency-Awareness Matters

From a pedagogical perspective, proficiency level should influence feedback provision. Lower-level learners benefit from comprehensive, explicit feedback on frequent error types, as this promotes awareness of foundational structures. Advanced learners, conversely, may be overwhelmed by feedback on every minor error and may benefit from selective, high-precision feedback on salient or high-impact error types. Furthermore, lower-level learners' errors are more likely to be deterministic (permitting high-precision correction), whilst advanced learners' errors require nuanced judgement.

Current GEC systems are predominantly proficiency-agnostic: they apply the same error-detection thresholds and correction strategies to all learners. The present work hypothesises that explicitly adapting precision and feedback scope to proficiency level will improve both correction quality and pedagogical effectiveness.

## 2.5 Corrective Feedback in Second Language Acquisition

Corrective feedback (CF)—explicit or implicit correction of learner errors—has been extensively studied in second language acquisition (SLA) research. A vast empirical literature examines the effectiveness of different CF types and explores mechanisms by which CF promotes learning. However, this rich pedagogical foundation has only partially informed computational GEC systems.

### Direct and Indirect Feedback

Ellis (2009) provides a comprehensive review of CF in SLA, distinguishing between direct feedback (explicitly providing the correct form) and indirect feedback (indicating that an error has occurred without providing the correction). Meta-analyses indicate that direct feedback is generally more effective for immediate error correction and explicit awareness, particularly for rule-governed errors. Indirect feedback requires greater learner engagement and metacognitive processing but may promote deeper learning and error internalisation over time.

Current GEC systems typically provide direct feedback exclusively: the corrected form is presented alongside or instead of the original. However, a system capable of generating indirect feedback (e.g., "Agreement error detected here" or "Check the verb form") could offer greater pedagogical flexibility. The present work extends GEC systems to optionally generate indirect feedback through metalinguistic annotations.

### Metalinguistic Feedback and Feedback Typology

Bitchener and Storch (2016) develop a fine-grained typology of written corrective feedback, distinguishing among direct correction, metalinguistic feedback (explicit rule statement, e.g., "Verbs must agree with the subject"), focused feedback (addressing selected error types only), and unfocused feedback (addressing all errors). Their review establishes that metalinguistic feedback, particularly when combined with opportunities for revision, promotes sustained learning gains.

The distinction between focused and unfocused feedback aligns naturally with proficiency levels: advanced learners may benefit from focused feedback targeting specific error types (high precision), whilst lower-level learners may require more comprehensive error identification (lower precision, higher recall). This theoretical foundation motivates the present work's adaptation of feedback comprehensiveness to proficiency level.

### Scaffolding and Adaptive Support

Vygotsky (1978) introduced the concept of the "zone of proximal development"—the space between a learner's independent capability and their capability with expert guidance. Scaffolding, developed by Wood et al. (1976) and systematically applied to language learning by Aljaafreh and Lantolf (1994), describes the provision of graduated support that is gradually withdrawn as learner competence increases.

Aljaafreh and Lantolf (1994) identified a hierarchy of implicit-to-explicit feedback: simple indication of error presence, indication of error location, request for self-correction, providing information about the nature of the error, and explicit correction. Critically, they found that lower-proficiency learners require more explicit feedback, whilst higher-proficiency learners respond better to implicit cues prompting self-correction.

This finding directly informs the present work: by varying feedback explicitness and comprehensiveness according to proficiency level, we approximate a computational implementation of scaffolding, with the system providing maximal support to lower-level learners and encouraging greater autonomy for advanced learners.

### Computational Implementations of Adaptive Feedback

Despite the rich SLA literature on corrective feedback, most computational GEC systems implement a single, fixed feedback strategy: a corrected sentence presented to the user. Few systems adapt feedback to learner characteristics. Notable exceptions include adaptive language-learning platforms and intelligent tutoring systems, which implement explicit pedagogical models. However, academic GEC research has largely overlooked the adaptation dimension.

The present work bridges this gap by operationalising key pedagogical principles from SLA research: error-type awareness (leveraging ERRANT's fine-grained error taxonomy), proficiency-aware precision and recall adaptation (based on CEFR levels and W&I+LOCNESS), and feedback type variation (direct vs. indirect, explicit vs. implicit). Specifically, the system maps CEFR levels to feedback configurations: A1–A2 learners receive comprehensive error identification with direct corrections; B1–B2 learners receive focused feedback on frequent error types with metalinguistic annotations; C1–C2 learners receive high-precision feedback with indirect cues encouraging self-correction.

---

## References

Aljaafreh, A. and Lantolf, J.P., 1994. Negative feedback as regulation and second language learning in the zone of proximal development. *The Modern Language Journal*, 78(4), pp.465-483.

Bitchener, J. and Storch, N., 2016. *Written Corrective Feedback for L2 Development*. Bristol: Multilingual Matters.

Bryant, C., Felice, M. and Briscoe, T., 2017. Automatic annotation and evaluation of error types for grammatical error correction. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 793-805. Vancouver: ACL.

Bryant, C., Felice, M., Andersen, Ø.E. and Briscoe, T., 2019. The BEA-2019 shared task on grammatical error correction. In *Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications*, pp. 52-75. Florence: ACL.

Choshen, L. and Abend, O., 2018. Inherent biases in reference-based evaluation for grammatical error correction. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 632-642. Melbourne: ACL.

Chung, H.W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S. and Webson, A., 2022. Scaling instruction-finetuned language models. *arXiv preprint arXiv:2210.11416*.

Clark, K., Luong, M.T., Le, Q.V. and Manning, C.D., 2020. ELECTRA: Pre-training text encoders as discriminators rather than generators. In *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)*.

Council of Europe, 2001. *Common European Framework of Reference for Languages: Learning, Teaching, Assessment*. Cambridge: Cambridge University Press.

Dahlmeier, D. and Ng, H.T., 2011. Correcting semantic collocation errors with L1-induced paraphrases. In *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 107-117. Edinburgh: ACL.

Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L., 2023. QLoRA: Efficient finetuning of quantized LLMs. *arXiv preprint arXiv:2305.14314*.

Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)*, pp. 4171-4186. Minneapolis: ACL.

Ellis, R., 2009. A typology of written corrective feedback types. *ELT Journal*, 63(2), pp.97-107.

Gamon, M., 2010. Using mostly native data to correct errors in learners' writing. In *Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2010)*, pp. 163-171. Los Angeles: ACL.

Grundkiewicz, R., Junczys-Dowmunt, M. and Heafield, K., 2019. Neural grammatical error correction systems with unsupervised pre-training on synthetic data. In *Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications*, pp. 252-263. Florence: ACL.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W., 2022. LoRA: Low-rank adaptation of large language models. In *Proceedings of the 10th International Conference on Learning Representations (ICLR 2022)*.

Junczys-Dowmunt, M., Grundkiewicz, R., Guha, S. and Heafield, K., 2018. Approaching neural grammatical error correction as a low-resource machine translation task. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2018)*, pp. 595-606. New Orleans: ACL.

Kantor, Y., Katz, Y., Choshen, L., Cohen-Karlik, E., Liberman, N., Toledo, A., Menczel, A. and Slonim, N., 2019. Learning to combine grammatical error corrections. In *Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2019)*, pp. 139-148. Florence: ACL.

Lester, B., Al-Rfou, R. and Constant, N., 2021. The power of scale for parameter-efficient prompt tuning. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3045-3059. ACL.

Li, X.L. and Liang, P., 2021. Prefix-tuning: Optimizing continuous prompts for generation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)*, pp. 4582-4597. ACL.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. and Stoyanov, V., 2019. RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

Ng, H.T., Wu, S.M., Wu, Y., Hadiwinoto, C. and Tetreault, J., 2013. The CoNLL-2013 shared task on grammatical error correction. In *Proceedings of the Seventeenth Conference on Computational Natural Language Learning: Shared Task*, pp. 1-12. Sofia: ACL.

Ng, H.T., Wu, S.M., Briscoe, T., Hadiwinoto, C., Susanto, R.H. and Bryant, C., 2014. The CoNLL-2014 shared task on grammatical error correction. In *Proceedings of the Eighteenth Conference on Computational Natural Language Learning: Shared Task*, pp. 1-14. Baltimore: ACL.

Omelianchuk, K., Atrasevych, V., Chernodub, A. and Skurzhanskyi, O., 2020. GECToR – grammatical error correction: Tag, not rewrite. In *Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2020)*, pp. 163-170. ACL.

Qorib, M.R., Na, S.-H. and Ng, H.T., 2022. Frustratingly easy system combination for grammatical error correction. In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2022)*, pp. 1964-1974. Seattle: ACL.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I., 2019. Language models are unsupervised multitask learners. *OpenAI Technical Report*.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu, P.J., 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), pp.1-67.

Rozovskaya, A. and Roth, D., 2016. Grammatical error correction: Machine translation and classifiers. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 2205-2215. Berlin: ACL.

Stahlberg, F. and Kumar, S., 2021. Synthetic data generation for grammatical error correction with tagged corruption models. In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2021)*, pp. 37-47. ACL.

Susanto, R.H., Phandi, P. and Ng, H.T., 2014. System combination for grammatical error correction. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 951-962. Doha: ACL.

Tarnavskyi, M., Chernodub, A. and Omelianchuk, K., 2022. Ensembling and knowledge distilling of large sequence taggers for grammatical error correction. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3842-3852. Dublin: ACL.

Vajjala, S. and Lõo, K., 2014. Automatic CEFR level prediction for Estonian learner text. In *Proceedings of the 3rd Workshop on NLP for Computer-Assisted Language Learning (NLP4CALL)*, pp. 113-127. Uppsala: LiU Electronic Press.

Vygotsky, L.S., 1978. *Mind in Society: The Development of Higher Psychological Processes*. Cambridge, MA: Harvard University Press.

Wood, D., Bruner, J.S. and Ross, G., 1976. The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry*, 17(2), pp.89-100.

Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R.R. and Le, Q.V., 2019. XLNet: Generalized autoregressive pretraining for language understanding. In *Advances in Neural Information Processing Systems (NeurIPS 32)*, pp. 5753-5763.

Yannakoudakis, H., Briscoe, T. and Medlock, B., 2011. A new dataset and method for automatically grading ESOL texts. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (ACL-HLT 2011)*, pp. 180-189. Portland, Oregon: ACL.
