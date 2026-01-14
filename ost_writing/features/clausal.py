"""
Clausal complexity features based on T-units.

Features:
    - num_t_units: Count of T-units (minimal terminable units)
    - mean_length_t_unit: Average T-unit length (MLT)
    - dependent_clauses_per_t_unit: Subordination ratio

A T-unit consists of a main clause plus any subordinate clauses
attached to it. This operationalization follows Hunt (1970).

References:
    Hunt, K. W. (1970). Syntactic maturity in schoolchildren and adults.
    Lu, X. (2010). Automatic analysis of syntactic complexity in L2 writing.
"""

from ..utils.nlp_setup import get_nlp


def _extract_t_units(text: str) -> tuple:
    """
    Extract T-units and dependent clause count from text.
    
    Uses SpaCy dependency parsing to identify main clauses and
    subordinate structures.
    
    Args:
        text: Input text string.
        
    Returns:
        Tuple of (list of T-unit lengths, total dependent clause count).
    """
    nlp = get_nlp()
    doc = nlp(text)
    
    t_unit_lengths = []
    dep_clause_count = 0
    
    for sent in doc.sents:
        current_t_unit = []
        sent_dep_clauses = 0
        
        for token in sent:
            current_t_unit.append(token)
            
            # Identify dependent clause markers
            if token.dep_ in ("advcl", "relcl", "ccomp", "xcomp", "acl"):
                sent_dep_clauses += 1
            
            # T-unit boundary: coordinating conjunction with clausal complement
            if (token.dep_ == "cc" and 
                token.head.pos_ == "VERB" and 
                any(child.dep_ == "conj" and child.pos_ == "VERB" 
                    for child in token.head.children)):
                if current_t_unit:
                    t_unit_lengths.append(len(current_t_unit))
                    dep_clause_count += sent_dep_clauses
                    current_t_unit = []
                    sent_dep_clauses = 0
        
        # Final T-unit in sentence
        if current_t_unit:
            t_unit_lengths.append(len(current_t_unit))
            dep_clause_count += sent_dep_clauses
    
    return t_unit_lengths, dep_clause_count


def num_t_units(text: str) -> int:
    """
    Count T-units (minimal terminable units) in text.
    
    Args:
        text: Input text string.
        
    Returns:
        Number of T-units.
    """
    t_unit_lengths, _ = _extract_t_units(text)
    return len(t_unit_lengths)


def mean_length_t_unit(text: str) -> float:
    """
    Calculate mean length of T-unit (MLT).
    
    MLT is a widely-used measure of syntactic complexity in writing
    development research.
    
    Args:
        text: Input text string.
        
    Returns:
        Average words per T-unit. Returns 0.0 if no T-units found.
        
    References:
        Lu, X. (2010). Automatic analysis of syntactic complexity.
    """
    t_unit_lengths, _ = _extract_t_units(text)
    
    if not t_unit_lengths:
        return 0.0
    
    return sum(t_unit_lengths) / len(t_unit_lengths)


def dependent_clauses_per_t_unit(text: str) -> float:
    """
    Calculate dependent clause ratio (DC/T).
    
    Measures syntactic subordination complexity. Higher values
    indicate more embedded clausal structures.
    
    Args:
        text: Input text string.
        
    Returns:
        Dependent clauses per T-unit ratio.
    """
    t_unit_lengths, dep_count = _extract_t_units(text)
    
    if not t_unit_lengths:
        return 0.0
    
    return dep_count / len(t_unit_lengths)
