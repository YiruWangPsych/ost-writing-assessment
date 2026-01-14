"""
Text cohesion features.

Features:
    - referential_cohesion: Word overlap between adjacent sentences
    - deep_cohesion: Logical connective density
"""

from nltk.tokenize import word_tokenize, sent_tokenize

from ..utils.nlp_setup import get_stopwords


def referential_cohesion(text: str) -> float:
    """
    Measure content word overlap between adjacent sentences.
    
    Uses Jaccard similarity to compute overlap, approximating
    Coh-Metrix PCREFz (referential cohesion component).
    
    Args:
        text: Input text string.
        
    Returns:
        Mean overlap score across sentence pairs (0-1 scale).
        
    References:
        McNamara, D. S., et al. (2014). Automated evaluation of text
        and discourse with Coh-Metrix.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0.0
    
    stop_words = get_stopwords()
    overlaps = []
    
    for i in range(len(sentences) - 1):
        tokens1 = {t.lower() for t in word_tokenize(sentences[i]) 
                   if t.isalpha() and t.lower() not in stop_words}
        tokens2 = {t.lower() for t in word_tokenize(sentences[i + 1]) 
                   if t.isalpha() and t.lower() not in stop_words}
        
        if not tokens1 or not tokens2:
            continue
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        if union > 0:
            overlaps.append(intersection / union)
    
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def deep_cohesion(text: str) -> float:
    """
    Measure logical connective density.
    
    Counts connectives that signal causal, temporal, and contrastive
    relationships, normalized by word count. Approximates Coh-Metrix
    PCDCz (deep cohesion component).
    
    Args:
        text: Input text string.
        
    Returns:
        Connective density (connectives per word).
        
    References:
        Halliday, M. A. K., & Hasan, R. (1976). Cohesion in English.
    """
    connectives = [
        # Causal
        "because", "therefore", "thus", "consequently", "so",
        # Contrastive
        "although", "though", "however", "but", "despite",
        # Temporal
        "first", "second", "next", "then", "finally", "last",
        # Additive
        "furthermore", "moreover", "additionally", "also",
    ]
    
    text_lower = text.lower()
    words = [t for t in word_tokenize(text) if t.isalpha()]
    
    if not words:
        return 0.0
    
    conn_count = sum(text_lower.count(f" {c} ") for c in connectives)
    return conn_count / len(words)
