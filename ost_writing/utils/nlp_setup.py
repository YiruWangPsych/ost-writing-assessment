"""
NLP model initialization and resource management.

This module handles lazy loading of SpaCy models and NLTK resources
to avoid repeated initialization overhead.
"""

import nltk
from nltk.corpus import stopwords

_nlp = None
_stopwords = None


def get_nlp():
    """
    Load SpaCy English model (lazy initialization).
    
    Returns:
        spacy.Language: Loaded SpaCy model.
    """
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_stopwords():
    """
    Load NLTK English stopwords (lazy initialization).
    
    Returns:
        set: Set of English stopwords.
    """
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stopwords = set(stopwords.words("english"))
    return _stopwords


def ensure_nltk_data():
    """Download required NLTK data packages if not present."""
    packages = ["punkt", "averaged_perceptron_tagger", "wordnet", "stopwords"]
    for pkg in packages:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}" if pkg in ["wordnet", "stopwords"] else f"taggers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
