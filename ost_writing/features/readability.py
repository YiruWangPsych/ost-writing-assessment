"""
Readability features.

Features:
    - flesch_kincaid_grade_level: U.S. grade level estimate
    - text_ease: Composite text easability score (Coh-Metrix analog)
"""

from nltk.tokenize import word_tokenize, sent_tokenize

from ..utils.text_processing import count_syllables


def flesch_kincaid_grade_level(text: str) -> float:
    """
    Calculate Flesch-Kincaid Grade Level.
    
    FK Grade = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    
    Args:
        text: Input text string.
        
    Returns:
        Estimated U.S. grade level required to comprehend the text.
        
    References:
        Kincaid, J. P., et al. (1975). Derivation of new readability formulas.
    """
    sentences = sent_tokenize(text)
    words = [t for t in word_tokenize(text) if t.isalpha()]
    
    if not sentences or not words:
        return 0.0
    
    word_count = len(words)
    sentence_count = len(sentences)
    syllable_count = sum(count_syllables(w) for w in words)
    
    grade = (0.39 * (word_count / sentence_count) + 
             11.8 * (syllable_count / word_count) - 15.59)
    
    return max(0.0, grade)


def text_ease(text: str) -> float:
    """
    Calculate composite text easability score.
    
    Combines multiple readability factors following Coh-Metrix text
    easability principles. Higher scores indicate easier text.
    
    Components weighted:
        - Grade level ease (25%)
        - Syntactic simplicity (25%)
        - Word concreteness (20%)
        - Referential cohesion (15%)
        - Deep cohesion (15%)
    
    Args:
        text: Input text string.
        
    Returns:
        Text ease score (0-1 scale).
        
    References:
        Graesser, A. C., et al. (2011). Coh-Metrix: Providing multilevel
        analyses of text characteristics.
    """
    from .syntactic import syntactic_simplicity
    from .cohesion import referential_cohesion, deep_cohesion
    
    # Grade level component (inverted: lower grade = easier)
    fk_grade = flesch_kincaid_grade_level(text)
    grade_ease = 1.0 - min(1.0, max(0.0, (fk_grade - 3) / 15.0))
    
    # Get component scores
    synt_simp = syntactic_simplicity(text)
    ref_coh = referential_cohesion(text)
    deep_coh = deep_cohesion(text)
    
    # Word concreteness placeholder (simplified)
    word_concrete = 0.5  # Neutral default
    
    # Weighted combination
    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    components = [grade_ease, synt_simp, word_concrete, ref_coh, deep_coh]
    
    return sum(w * c for w, c in zip(weights, components))
