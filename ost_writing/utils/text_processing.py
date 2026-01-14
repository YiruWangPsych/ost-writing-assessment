"""
Text processing utilities.

Common functions shared across feature extraction modules.
"""


def count_syllables(word: str) -> int:
    """
    Estimate syllable count using vowel-group heuristic.
    
    This approximation follows the method described in Kincaid et al. (1975).
    
    Args:
        word: Input word string.
        
    Returns:
        Estimated number of syllables (minimum 1).
        
    References:
        Kincaid, J. P., et al. (1975). Derivation of new readability formulas.
    """
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    
    # Adjust for silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    
    return max(1, count)
