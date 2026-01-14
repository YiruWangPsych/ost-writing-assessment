"""
Lexical richness features.

Features:
    - lexical_diversity: Type-Token Ratio (TTR)
    - vocabulary_sophistication: WordNet-based depth measure
    - polysemy_word: Count of polysemous words
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from ..utils.nlp_setup import get_stopwords


def lexical_diversity(text: str) -> float:
    """
    Calculate Type-Token Ratio (TTR).
    
    TTR = unique_types / total_tokens
    
    Args:
        text: Input text string.
        
    Returns:
        TTR value between 0 and 1. Returns 0.0 if no tokens.
        
    References:
        Templin, M. C. (1957). Certain language skills in children.
    """
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def vocabulary_sophistication(text: str) -> float:
    """
    Measure vocabulary sophistication using WordNet synset depth.
    
    Deeper synsets in the WordNet hierarchy indicate more specific,
    sophisticated vocabulary. Normalized to 0-1 scale.
    
    Args:
        text: Input text string.
        
    Returns:
        Sophistication score (0-1 scale).
        
    References:
        Miller, G. A. (1995). WordNet: A lexical database for English.
    """
    stop_words = get_stopwords()
    tokens = [t.lower() for t in word_tokenize(text) 
              if t.isalpha() and t.lower() not in stop_words]
    
    if not tokens:
        return 0.0
    
    depths = []
    for token in tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            token_depths = [s.min_depth() for s in synsets]
            depths.append(sum(token_depths) / len(token_depths))
    
    if not depths:
        return 0.0
    
    avg_depth = sum(depths) / len(depths)
    return min(1.0, avg_depth / 10.0)


def polysemy_word(text: str) -> int:
    """
    Count words with multiple WordNet senses.
    
    Polysemous words (>1 sense) indicate vocabulary that can convey
    nuanced meanings depending on context.
    
    Args:
        text: Input text string.
        
    Returns:
        Count of polysemous words.
    """
    stop_words = get_stopwords()
    tokens = [t.lower() for t in word_tokenize(text) 
              if t.isalpha() and t.lower() not in stop_words]
    
    count = 0
    for token in tokens:
        synsets = wordnet.synsets(token)
        if len(synsets) > 1:
            count += 1
    
    return count
