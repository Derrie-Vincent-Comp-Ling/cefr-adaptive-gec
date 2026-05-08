"""CEFR-adaptive feedback templates for ERRANT error types.

Each error type maps to a dict with four CEFR-level keys: A, B, C, N.
Templates use Python format-string placeholders:
    {original}  — the original (erroneous) token(s)
    {corrected} — the corrected token(s)
    {type}      — the ERRANT error type code

CEFR adaptation strategy (following Ellis, 2009; Bitchener & Storch, 2016):
    A (beginner)     — Direct corrective feedback: shows correction explicitly,
                        explains rule in plain language, gives an example.
    B (intermediate) — Metalinguistic feedback: names the grammar concept,
                        explains why the correction is needed.
    C (advanced)     — Indirect coded feedback: uses linguistic terminology,
                        expects learner to understand the category.
    N (proficient)   — Minimal flagging: just identifies the error type,
                        assumes the writer can self-correct.

Reference:
    Ellis, R. (2009) 'A typology of written corrective feedback types',
    ELT Journal, 63(2), pp. 97-107.
    Bitchener, J. and Storch, N. (2016) Written Corrective Feedback for
    L2 Development. Bristol: Multilingual Matters.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# ERRANT type prefixes:
#   M: = Missing     (token should be inserted)
#   R: = Replacement (token should be replaced)
#   U: = Unnecessary (token should be deleted)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict[str, str]] = {

    # -----------------------------------------------------------------------
    # DETERMINERS
    # -----------------------------------------------------------------------
    "M:DET": {
        "A": "You need to add a word like 'a', 'an', or 'the' before a noun here. "
             "Try: '{corrected}'.",
        "B": "A determiner (article) is missing. English nouns usually need 'a', 'an', "
             "or 'the' before them. Correction: '{corrected}'.",
        "C": "Missing determiner. Insert '{corrected}' — English requires articles "
             "before most singular countable nouns.",
        "N": "Missing determiner → '{corrected}'.",
    },
    "R:DET": {
        "A": "The word '{original}' should be '{corrected}'. We use 'a' before consonant "
             "sounds and 'an' before vowel sounds.",
        "B": "Wrong determiner: '{original}' → '{corrected}'. Check whether the noun "
             "needs a definite ('the') or indefinite ('a'/'an') article.",
        "C": "Determiner error: '{original}' → '{corrected}'. Review definiteness and "
             "article selection.",
        "N": "Determiner: '{original}' → '{corrected}'.",
    },
    "U:DET": {
        "A": "The word '{original}' is not needed here. Some nouns don't need 'a', 'an', "
             "or 'the' — for example, uncountable nouns and plural generalisations.",
        "B": "Unnecessary determiner '{original}'. This noun does not require an article "
             "in this context (e.g. uncountable nouns or generic plurals).",
        "C": "Superfluous determiner '{original}'. Remove — zero article is required here.",
        "N": "Remove unnecessary determiner '{original}'.",
    },

    # -----------------------------------------------------------------------
    # PREPOSITIONS
    # -----------------------------------------------------------------------
    "M:PREP": {
        "A": "A preposition is missing. You need to add '{corrected}' here. "
             "Prepositions are words like 'in', 'on', 'at', 'to', 'for'.",
        "B": "Missing preposition. Add '{corrected}'. The verb or noun here requires "
             "a specific preposition.",
        "C": "Missing preposition '{corrected}' — check collocational requirements.",
        "N": "Missing preposition → '{corrected}'.",
    },
    "R:PREP": {
        "A": "The word '{original}' should be '{corrected}'. Different verbs and "
             "expressions need different prepositions in English.",
        "B": "Wrong preposition: '{original}' → '{corrected}'. This is a fixed "
             "collocation — the correct preposition must be memorised.",
        "C": "Preposition error: '{original}' → '{corrected}'. Prepositional "
             "collocation mismatch.",
        "N": "Preposition: '{original}' → '{corrected}'.",
    },
    "U:PREP": {
        "A": "The word '{original}' is not needed here. In English, some verbs are "
             "followed directly by their object without a preposition.",
        "B": "Unnecessary preposition '{original}'. The verb here is transitive and "
             "takes a direct object without a preposition.",
        "C": "Superfluous preposition '{original}'. Remove — verb is transitive here.",
        "N": "Remove unnecessary preposition '{original}'.",
    },

    # -----------------------------------------------------------------------
    # VERBS — Subject-Verb Agreement
    # -----------------------------------------------------------------------
    "R:VERB:SVA": {
        "A": "The verb '{original}' doesn't match the subject. Change it to "
             "'{corrected}'. Remember: 'I am', 'he/she is', 'they are'; "
             "'he goes', 'they go'.",
        "B": "Subject-verb agreement error: '{original}' → '{corrected}'. The verb "
             "must agree with the subject in number (singular/plural) and person.",
        "C": "SVA error: '{original}' → '{corrected}'. Check subject number/person "
             "agreement.",
        "N": "SVA: '{original}' → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # VERBS — Tense
    # -----------------------------------------------------------------------
    "R:VERB:TENSE": {
        "A": "The verb '{original}' is in the wrong tense. It should be "
             "'{corrected}'. Check: is the action in the past, present, or future?",
        "B": "Verb tense error: '{original}' → '{corrected}'. The tense should match "
             "the time reference in the sentence.",
        "C": "Tense error: '{original}' → '{corrected}'. Review temporal consistency.",
        "N": "Tense: '{original}' → '{corrected}'.",
    },
    "M:VERB:TENSE": {
        "A": "A verb form is missing here. Add '{corrected}' to show the correct tense.",
        "B": "Missing tense marker. Add '{corrected}' to form the correct verb tense.",
        "C": "Missing tense morphology → '{corrected}'.",
        "N": "Missing tense → '{corrected}'.",
    },
    "U:VERB:TENSE": {
        "A": "The word '{original}' is not needed here. The tense is already shown "
             "by another word in the sentence.",
        "B": "Unnecessary tense marker '{original}'. The tense is already expressed "
             "elsewhere in the verb phrase.",
        "C": "Superfluous tense marker '{original}'. Remove — redundant in this "
             "verb phrase.",
        "N": "Remove unnecessary tense marker '{original}'.",
    },

    # -----------------------------------------------------------------------
    # VERBS — Form
    # -----------------------------------------------------------------------
    "R:VERB:FORM": {
        "A": "The verb '{original}' is in the wrong form. It should be '{corrected}'. "
             "Check if you need the base form, -ing form, or past participle.",
        "B": "Verb form error: '{original}' → '{corrected}'. After certain verbs, "
             "prepositions, or auxiliaries, a specific verb form is required "
             "(infinitive, gerund, or participle).",
        "C": "Verb form: '{original}' → '{corrected}'. Check subcategorisation frame.",
        "N": "Verb form: '{original}' → '{corrected}'.",
    },
    "M:VERB:FORM": {
        "A": "A verb is missing here. Add '{corrected}' to complete the sentence.",
        "B": "Missing verb form. Add '{corrected}' — the clause requires a verb "
             "in this position.",
        "C": "Missing verb form → '{corrected}'.",
        "N": "Missing verb form → '{corrected}'.",
    },
    "U:VERB:FORM": {
        "A": "The word '{original}' is extra here. Remove it to make the sentence "
             "correct.",
        "B": "Unnecessary verb form '{original}'. This creates a double marking or "
             "redundant verb phrase.",
        "C": "Superfluous verb form '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # VERBS — Inflection
    # -----------------------------------------------------------------------
    "R:VERB:INFL": {
        "A": "The verb '{original}' is spelled wrong. The correct form is "
             "'{corrected}'. Some verbs have irregular past forms.",
        "B": "Verb inflection error: '{original}' → '{corrected}'. This verb has "
             "an irregular conjugation pattern.",
        "C": "Inflection error: '{original}' → '{corrected}'. Irregular morphology.",
        "N": "Inflection: '{original}' → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # VERBS — General
    # -----------------------------------------------------------------------
    "R:VERB": {
        "A": "The verb '{original}' should be '{corrected}'. Check that you're "
             "using the right word here.",
        "B": "Wrong verb: '{original}' → '{corrected}'. A different verb is more "
             "appropriate in this context.",
        "C": "Verb selection: '{original}' → '{corrected}'.",
        "N": "Verb: '{original}' → '{corrected}'.",
    },
    "M:VERB": {
        "A": "A verb is missing. Add '{corrected}' to make the sentence complete.",
        "B": "Missing verb. Insert '{corrected}' — every clause needs a main verb.",
        "C": "Missing verb → '{corrected}'.",
        "N": "Missing verb → '{corrected}'.",
    },
    "U:VERB": {
        "A": "The word '{original}' is not needed here. Remove it.",
        "B": "Unnecessary verb '{original}'. Remove to avoid redundancy.",
        "C": "Superfluous verb '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # NOUNS
    # -----------------------------------------------------------------------
    "R:NOUN": {
        "A": "The word '{original}' should be '{corrected}'.",
        "B": "Wrong noun: '{original}' → '{corrected}'. Check the meaning "
             "and appropriateness.",
        "C": "Noun selection: '{original}' → '{corrected}'.",
        "N": "Noun: '{original}' → '{corrected}'.",
    },
    "R:NOUN:NUM": {
        "A": "The word '{original}' should be '{corrected}'. Check if you need "
             "the singular (one) or plural (more than one) form.",
        "B": "Noun number error: '{original}' → '{corrected}'. The noun should be "
             "singular or plural to match the context.",
        "C": "Number: '{original}' → '{corrected}'. Singular/plural mismatch.",
        "N": "Number: '{original}' → '{corrected}'.",
    },
    "R:NOUN:POSS": {
        "A": "The word '{original}' should be '{corrected}'. To show that something "
             "belongs to someone, add an apostrophe + s ('s).",
        "B": "Possessive form error: '{original}' → '{corrected}'. Use the possessive "
             "case (apostrophe + s) to show ownership.",
        "C": "Possessive: '{original}' → '{corrected}'.",
        "N": "Possessive: '{original}' → '{corrected}'.",
    },
    "R:NOUN:INFL": {
        "A": "The word '{original}' is not a real English word. The correct form is "
             "'{corrected}'. Some nouns have irregular plural forms.",
        "B": "Noun inflection error: '{original}' → '{corrected}'. This noun has "
             "an irregular plural form.",
        "C": "Inflection: '{original}' → '{corrected}'. Irregular nominal morphology.",
        "N": "Inflection: '{original}' → '{corrected}'.",
    },
    "U:NOUN": {
        "A": "The word '{original}' is extra here. Remove it.",
        "B": "Unnecessary noun '{original}'. Remove to avoid redundancy.",
        "C": "Superfluous noun '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # PRONOUNS
    # -----------------------------------------------------------------------
    "R:PRON": {
        "A": "The word '{original}' should be '{corrected}'. Make sure you're "
             "using the right pronoun (he/him, she/her, they/them, etc.).",
        "B": "Pronoun error: '{original}' → '{corrected}'. Check the pronoun's "
             "case (subject vs. object), number, or reference.",
        "C": "Pronoun: '{original}' → '{corrected}'. Check case/reference.",
        "N": "Pronoun: '{original}' → '{corrected}'.",
    },
    "M:PRON": {
        "A": "A pronoun is missing here. Add '{corrected}'.",
        "B": "Missing pronoun. Insert '{corrected}' — the clause requires a pronoun.",
        "C": "Missing pronoun → '{corrected}'.",
        "N": "Missing pronoun → '{corrected}'.",
    },
    "U:PRON": {
        "A": "The word '{original}' is not needed here. Remove it.",
        "B": "Unnecessary pronoun '{original}'. Remove — the referent is already clear.",
        "C": "Superfluous pronoun '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # ADJECTIVES
    # -----------------------------------------------------------------------
    "R:ADJ": {
        "A": "The word '{original}' should be '{corrected}'.",
        "B": "Adjective error: '{original}' → '{corrected}'.",
        "C": "Adjective: '{original}' → '{corrected}'.",
        "N": "Adjective: '{original}' → '{corrected}'.",
    },
    "R:ADJ:FORM": {
        "A": "The word '{original}' should be '{corrected}'. Check the comparative "
             "or superlative form (big → bigger → biggest).",
        "B": "Adjective form error: '{original}' → '{corrected}'. Check comparative/ "
             "superlative morphology.",
        "C": "Adjective form: '{original}' → '{corrected}'.",
        "N": "Adj form: '{original}' → '{corrected}'.",
    },
    "U:ADJ": {
        "A": "The word '{original}' is extra here. Remove it.",
        "B": "Unnecessary adjective '{original}'. Remove.",
        "C": "Superfluous adjective '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # ADVERBS
    # -----------------------------------------------------------------------
    "R:ADV": {
        "A": "The word '{original}' should be '{corrected}'.",
        "B": "Adverb error: '{original}' → '{corrected}'.",
        "C": "Adverb: '{original}' → '{corrected}'.",
        "N": "Adverb: '{original}' → '{corrected}'.",
    },
    "U:ADV": {
        "A": "The word '{original}' is not needed here. Remove it.",
        "B": "Unnecessary adverb '{original}'. Remove.",
        "C": "Superfluous adverb '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # SPELLING
    # -----------------------------------------------------------------------
    "R:SPELL": {
        "A": "The word '{original}' is misspelled. The correct spelling is "
             "'{corrected}'.",
        "B": "Spelling error: '{original}' → '{corrected}'.",
        "C": "Spelling: '{original}' → '{corrected}'.",
        "N": "Spelling: '{original}' → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # ORTHOGRAPHY (capitalisation, whitespace)
    # -----------------------------------------------------------------------
    "R:ORTH": {
        "A": "The word '{original}' has a capitalisation or spacing error. "
             "It should be '{corrected}'.",
        "B": "Orthography error: '{original}' → '{corrected}'. Check capitalisation "
             "and spacing.",
        "C": "Orthography: '{original}' → '{corrected}'.",
        "N": "Orth: '{original}' → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # MORPHOLOGY (general)
    # -----------------------------------------------------------------------
    "R:MORPH": {
        "A": "The form of the word '{original}' is wrong. It should be "
             "'{corrected}'. Check the ending of the word.",
        "B": "Morphological error: '{original}' → '{corrected}'. The word form "
             "(suffix, prefix, or derivation) is incorrect.",
        "C": "Morphology: '{original}' → '{corrected}'. Derivational error.",
        "N": "Morphology: '{original}' → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # PUNCTUATION
    # -----------------------------------------------------------------------
    "R:PUNCT": {
        "A": "The punctuation '{original}' should be '{corrected}'.",
        "B": "Punctuation error: '{original}' → '{corrected}'.",
        "C": "Punctuation: '{original}' → '{corrected}'.",
        "N": "Punct: '{original}' → '{corrected}'.",
    },
    "M:PUNCT": {
        "A": "A punctuation mark is missing. Add '{corrected}' here.",
        "B": "Missing punctuation. Add '{corrected}'.",
        "C": "Missing punctuation → '{corrected}'.",
        "N": "Missing punct → '{corrected}'.",
    },
    "U:PUNCT": {
        "A": "The punctuation '{original}' is not needed here. Remove it.",
        "B": "Unnecessary punctuation '{original}'. Remove.",
        "C": "Superfluous punctuation '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # CONJUNCTIONS
    # -----------------------------------------------------------------------
    "U:CONJ": {
        "A": "The word '{original}' is not needed here. Remove it.",
        "B": "Unnecessary conjunction '{original}'. Remove.",
        "C": "Superfluous conjunction '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },
    "M:CONJ": {
        "A": "A connecting word is missing. Add '{corrected}' to link the ideas.",
        "B": "Missing conjunction. Add '{corrected}'.",
        "C": "Missing conjunction → '{corrected}'.",
        "N": "Missing conjunction → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # WORD ORDER
    # -----------------------------------------------------------------------
    "R:WO": {
        "A": "The words are in the wrong order. It should be '{corrected}' "
             "instead of '{original}'.",
        "B": "Word order error: '{original}' → '{corrected}'. English has a "
             "relatively fixed word order (Subject-Verb-Object).",
        "C": "Word order: '{original}' → '{corrected}'.",
        "N": "Word order: '{original}' → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # CONTRACTIONS
    # -----------------------------------------------------------------------
    "R:CONTR": {
        "A": "'{original}' should be written as '{corrected}'.",
        "B": "Contraction error: '{original}' → '{corrected}'.",
        "C": "Contraction: '{original}' → '{corrected}'.",
        "N": "Contraction: '{original}' → '{corrected}'.",
    },

    # -----------------------------------------------------------------------
    # PARTICLES
    # -----------------------------------------------------------------------
    "R:PART": {
        "A": "The word '{original}' should be '{corrected}'.",
        "B": "Particle error: '{original}' → '{corrected}'.",
        "C": "Particle: '{original}' → '{corrected}'.",
        "N": "Particle: '{original}' → '{corrected}'.",
    },
    "U:PART": {
        "A": "The word '{original}' is extra here. Remove it.",
        "B": "Unnecessary particle '{original}'. Remove.",
        "C": "Superfluous particle '{original}'. Remove.",
        "N": "Remove '{original}'.",
    },

    # -----------------------------------------------------------------------
    # OTHER / CATCH-ALL
    # -----------------------------------------------------------------------
    "R:OTHER": {
        "A": "'{original}' should be changed to '{corrected}'.",
        "B": "Error: '{original}' → '{corrected}'.",
        "C": "'{original}' → '{corrected}'.",
        "N": "'{original}' → '{corrected}'.",
    },
    "M:OTHER": {
        "A": "Something is missing. Add '{corrected}' here.",
        "B": "Missing element. Add '{corrected}'.",
        "C": "Insert '{corrected}'.",
        "N": "Insert '{corrected}'.",
    },
    "U:OTHER": {
        "A": "The word(s) '{original}' should be removed.",
        "B": "Unnecessary: remove '{original}'.",
        "C": "Remove '{original}'.",
        "N": "Remove '{original}'.",
    },
}


# ---------------------------------------------------------------------------
# Fallback template generator
# ---------------------------------------------------------------------------
def get_feedback(
    error_type: str,
    cefr: str,
    original: str,
    corrected: str,
) -> str:
    """Return a feedback string for a given error type and CEFR level.

    Falls back to generic templates if the specific error type is not covered.
    """
    cefr = cefr.upper()
    if cefr not in ("A", "B", "C", "N"):
        cefr = "B"  # default to intermediate

    # Try exact match
    if error_type in TEMPLATES and cefr in TEMPLATES[error_type]:
        return TEMPLATES[error_type][cefr].format(
            original=original, corrected=corrected, type=error_type
        )

    # Fall back to prefix-based generic (M:/R:/U: + OTHER)
    prefix = error_type.split(":")[0] if ":" in error_type else "R"
    fallback_key = f"{prefix}:OTHER"
    if fallback_key in TEMPLATES and cefr in TEMPLATES[fallback_key]:
        return TEMPLATES[fallback_key][cefr].format(
            original=original, corrected=corrected, type=error_type
        )

    # Ultimate fallback
    if cefr == "A":
        return f"'{original}' should be changed to '{corrected}'."
    elif cefr == "B":
        return f"Error ({error_type}): '{original}' → '{corrected}'."
    elif cefr == "C":
        return f"{error_type}: '{original}' → '{corrected}'."
    else:
        return f"'{original}' → '{corrected}'."
