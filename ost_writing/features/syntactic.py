"""
Syntactic complexity features.

Features:
    - sentence_type_diversity: Variety of sentence constructions
    - syntactic_simplicity: Inverse of syntactic complexity (Coh-Metrix PCSYNz analog)
    - information_density: Content word ratio
"""

from nltk.tokenize import word_tokenize, sent_tokenize

from ..utils.nlp_setup import get_stopwords


def sentence_type_diversity(text: str) -> float:
    """
    Measure variety in sentence constructions.
    
    Classifies sentences into types (question, exclamation, complex,
    compound, simple) and returns proportion of unique types.
    
    Args:
        text: Input text string.
        
    Returns:
        Diversity score (0-1 scale based on 5 possible types).
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    
    types = set()
    complex_starters = ("if", "when", "although", "because", "while", "since", "after", "before")
    
    for sent in sentences:
        sent_stripped = sent.strip()
        if sent_stripped.endswith("?"):
            types.add("question")
        elif sent_stripped.endswith("!"):
            types.add("exclamation")
        elif any(sent_stripped.lower().startswith(w) for w in complex_starters):
            types.add("complex")
        elif "," in sent and len(word_tokenize(sent)) > 10:
            types.add("compound")
        else:
            types.add("simple")
    
    return len(types) / 5.0


def syntactic_simplicity(text: str) -> float:
    """
    Estimate syntactic simplicity based on sentence length.
    
    Shorter sentences indicate simpler syntax. This measure approximates
    Coh-Metrix PCSYNz (syntactic simplicity component).
    
    Args:
        text: Input text string.
        
    Returns:
        Simplicity score (0-1 scale). Higher = simpler syntax.
        
    References:
        Graesser, A. C., et al. (2004). Coh-Metrix. Behavior Research Methods.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    
    lengths = [len(word_tokenize(s)) for s in sentences]
    avg_len = sum(lengths) / len(lengths)
    
    # Transform: shorter sentences = higher simplicity
    return 1.0 / (1.0 + avg_len / 15.0) if avg_len > 0 else 0.0


def information_density(text: str) -> float:
    """
    Calculate ratio of content words to total words.
    
    Content words (nouns, verbs, adjectives, adverbs) carry semantic
    information, while function words serve grammatical purposes.
    
    Args:
        text: Input text string.
        
    Returns:
        Density ratio (0-1 scale).
    """
    stop_words = get_stopwords()
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
    
    if not tokens:
        return 0.0
    
    content_words = [t for t in tokens if t not in stop_words]
    return len(content_words) / len(tokens)
