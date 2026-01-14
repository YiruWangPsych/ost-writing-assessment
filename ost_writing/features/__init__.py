"""
Feature extraction modules organized by linguistic category.

Categories:
    - surface: Word count, sentence length, complex sentences
    - lexical: Lexical diversity, vocabulary sophistication, polysemy
    - syntactic: Sentence type diversity, syntactic simplicity, information density
    - accuracy: Error detection, context-sensitive word count
    - readability: Flesch-Kincaid grade level, text ease
    - cohesion: Referential cohesion, deep cohesion
    - variability: Word length variance, syllable variance
    - clausal: T-unit metrics
"""

from .surface import word_count, avg_sentence_length, num_complex_sentences
from .lexical import lexical_diversity, vocabulary_sophistication, polysemy_word
from .syntactic import sentence_type_diversity, syntactic_simplicity, information_density
from .accuracy import error_count, context_sensitive_count
from .readability import flesch_kincaid_grade_level, text_ease
from .cohesion import referential_cohesion, deep_cohesion
from .variability import word_length_variance, syllable_variance
from .clausal import num_t_units, mean_length_t_unit, dependent_clauses_per_t_unit

__all__ = [
    "word_count", "avg_sentence_length", "num_complex_sentences",
    "lexical_diversity", "vocabulary_sophistication", "polysemy_word",
    "sentence_type_diversity", "syntactic_simplicity", "information_density",
    "error_count", "context_sensitive_count",
    "flesch_kincaid_grade_level", "text_ease",
    "referential_cohesion", "deep_cohesion",
    "word_length_variance", "syllable_variance",
    "num_t_units", "mean_length_t_unit", "dependent_clauses_per_t_unit",
]
