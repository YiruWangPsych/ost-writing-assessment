"""
Text variability features.

Features:
    - word_length_variance: Variance in word lengths
    - syllable_variance: Variance in syllable counts
"""

import numpy as np
from nltk.tokenize import word_tokenize

from ..utils.text_processing import count_syllables


def word_length_variance(text: str) -> float:
    """
    Calculate variance in word lengths (characters).
    
    Higher variance indicates greater variability in word complexity,
    analogous to Coh-Metrix DESWLsyd measure.
    
    Args:
        text: Input text string.
        
    Returns:
        Variance of word lengths.
        
    References:
        McNamara, D. S., et al. (2014). Automated evaluation of text
        and discourse with Coh-Metrix.
    """
    words = [t for t in word_tokenize(text) if t.isalpha()]
    
    if not words:
        return 0.0
    
    lengths = [len(w) for w in words]
    return float(np.var(lengths))


def syllable_variance(text: str) -> float:
    """
    Calculate variance in syllable counts per word.
    
    Measures phonological complexity variation across words.
    
    Args:
        text: Input text string.
        
    Returns:
        Variance of syllable counts.
    """
    words = [t for t in word_tokenize(text) if t.isalpha()]
    
    if not words:
        return 0.0
    
    syllables = [count_syllables(w) for w in words]
    return float(np.var(syllables))
