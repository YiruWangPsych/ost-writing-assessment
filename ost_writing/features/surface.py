"""
Surface-level text features.

Features:
    - word_count: Total word count
    - avg_sentence_length: Mean words per sentence
    - num_complex_sentences: Sentences exceeding threshold length
"""

from nltk.tokenize import word_tokenize, sent_tokenize


def word_count(text: str) -> int:
    """
    Count total words (alphabetic tokens only).
    
    Args:
        text: Input text string.
        
    Returns:
        Number of alphabetic words.
    """
    tokens = word_tokenize(text)
    return len([t for t in tokens if t.isalpha()])


def avg_sentence_length(text: str) -> float:
    """
    Calculate mean sentence length in words.
    
    Args:
        text: Input text string.
        
    Returns:
        Average words per sentence. Returns 0.0 if no sentences found.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    
    lengths = [len([t for t in word_tokenize(s) if t.isalpha()]) for s in sentences]
    return sum(lengths) / len(lengths)


def num_complex_sentences(text: str, threshold: int = 15) -> int:
    """
    Count sentences exceeding word count threshold.
    
    Following Hunt (1965), sentences with more than 15 words are
    considered syntactically complex.
    
    Args:
        text: Input text string.
        threshold: Minimum word count for complex classification (default: 15).
        
    Returns:
        Number of complex sentences.
        
    References:
        Hunt, K. W. (1965). Grammatical structures written at three grade levels.
    """
    sentences = sent_tokenize(text)
    count = 0
    for sent in sentences:
        words = [t for t in word_tokenize(sent) if t.isalpha()]
        if len(words) > threshold:
            count += 1
    return count
