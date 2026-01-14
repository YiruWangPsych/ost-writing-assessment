"""
Writing accuracy features.

Features:
    - error_count: Grammatical error detection using language_tool_python
    - context_sensitive_count: Words used in non-primary sense
"""

import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet

from ..utils.nlp_setup import get_nlp

# Lazy loading for language_tool_python
_language_tool = None


def _get_language_tool():
    """Lazy load language_tool_python to avoid slow startup."""
    global _language_tool
    if _language_tool is None:
        import language_tool_python
        _language_tool = language_tool_python.LanguageTool('en-US')
    return _language_tool


def error_count(text: str) -> int:
    """
    Count grammatical and usage errors using language_tool_python.
    
    This function uses LanguageTool, a rule-based grammar checker,
    to identify grammatical errors, spelling mistakes, and style issues.
    
    Args:
        text: Input text string.
        
    Returns:
        Total error count detected by LanguageTool.
        
    Note:
        Requires language_tool_python package and Java Runtime Environment.
        First call may be slow due to LanguageTool initialization.
    """
    if not text or len(text) < 10:
        return 0
    
    tool = _get_language_tool()
    matches = tool.check(text)
    return len(matches)


def context_sensitive_count(text: str) -> int:
    """
    Count words where usage differs from primary WordNet sense.
    
    This measure indicates vocabulary depth, as words used in secondary
    or specialized senses suggest more nuanced language use.
    
    Args:
        text: Input text string.
        
    Returns:
        Count of context-sensitive word usages.
    """
    nlp = get_nlp()
    doc = nlp(text)
    
    count = 0
    for token in doc:
        if not token.is_alpha:
            continue
        
        synsets = wordnet.synsets(token.text.lower())
        if synsets:
            primary_lemma = synsets[0].lemmas()[0].name().lower()
            if primary_lemma != token.text.lower():
                count += 1
    
    return count
