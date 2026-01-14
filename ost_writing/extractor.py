"""
Main feature extraction interface.

Provides FeatureExtractor class for extracting all 20 linguistic
features from text samples.
"""

from typing import Dict, List, Union
import pandas as pd
from tqdm import tqdm

from .utils.nlp_setup import ensure_nltk_data, get_nlp
from .features import (
    word_count, avg_sentence_length, num_complex_sentences,
    lexical_diversity, vocabulary_sophistication, polysemy_word,
    sentence_type_diversity, syntactic_simplicity, information_density,
    error_count, context_sensitive_count,
    flesch_kincaid_grade_level, text_ease,
    referential_cohesion, deep_cohesion,
    word_length_variance, syllable_variance,
    num_t_units, mean_length_t_unit, dependent_clauses_per_t_unit,
)


class FeatureExtractor:
    """
    Extract 20 linguistic features from text samples.
    
    Features are organized into 8 categories:
        - Surface: word_count, avg_sentence_length, num_complex_sentences
        - Lexical: lexical_diversity, vocabulary_sophistication, polysemy_word
        - Syntactic: sentence_type_diversity, syntactic_simplicity, information_density
        - Accuracy: error_count, context_sensitive_count
        - Readability: flesch_kincaid_grade_level, text_ease
        - Cohesion: referential_cohesion, deep_cohesion
        - Variability: word_length_variance, syllable_variance
        - Clausal: num_t_units, mean_length_t_unit, dependent_clauses_per_t_unit
    
    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract("Sample text here.")
        >>> df = extractor.extract_batch(["Text 1", "Text 2"])
    """
    
    FEATURE_NAMES = [
        "word_count", "avg_sentence_length", "num_complex_sentences",
        "lexical_diversity", "vocabulary_sophistication", "polysemy_word",
        "sentence_type_diversity", "syntactic_simplicity", "information_density",
        "error_count", "context_sensitive_count",
        "flesch_kincaid_grade_level", "text_ease",
        "referential_cohesion", "deep_cohesion",
        "word_length_variance", "syllable_variance",
        "num_t_units", "mean_length_t_unit", "dependent_clauses_per_t_unit",
    ]
    
    def __init__(self, prefix: str = "OST_"):
        """
        Initialize feature extractor.
        
        Args:
            prefix: Prefix for feature names in output (default: "OST_").
                    Set to "" for no prefix.
        """
        self.prefix = prefix
        ensure_nltk_data()
        # Pre-load SpaCy model
        get_nlp()
    
    def extract(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Extract all features from a single text.
        
        Args:
            text: Input text string.
            
        Returns:
            Dictionary mapping feature names to values.
        """
        if not text or not isinstance(text, str):
            return {f"{self.prefix}{name}": 0 for name in self.FEATURE_NAMES}
        
        features = {
            "word_count": word_count(text),
            "avg_sentence_length": avg_sentence_length(text),
            "num_complex_sentences": num_complex_sentences(text),
            "lexical_diversity": lexical_diversity(text),
            "vocabulary_sophistication": vocabulary_sophistication(text),
            "polysemy_word": polysemy_word(text),
            "sentence_type_diversity": sentence_type_diversity(text),
            "syntactic_simplicity": syntactic_simplicity(text),
            "information_density": information_density(text),
            "error_count": error_count(text),
            "context_sensitive_count": context_sensitive_count(text),
            "flesch_kincaid_grade_level": flesch_kincaid_grade_level(text),
            "text_ease": text_ease(text),
            "referential_cohesion": referential_cohesion(text),
            "deep_cohesion": deep_cohesion(text),
            "word_length_variance": word_length_variance(text),
            "syllable_variance": syllable_variance(text),
            "num_t_units": num_t_units(text),
            "mean_length_t_unit": mean_length_t_unit(text),
            "dependent_clauses_per_t_unit": dependent_clauses_per_t_unit(text),
        }
        
        if self.prefix:
            features = {f"{self.prefix}{k}": v for k, v in features.items()}
        
        return features
    
    def extract_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from multiple texts.
        
        Args:
            texts: List of text strings.
            show_progress: Show progress bar (default: True).
            
        Returns:
            DataFrame with one row per text and columns for each feature.
        """
        iterator = tqdm(texts, desc="Extracting features") if show_progress else texts
        results = [self.extract(text) for text in iterator]
        return pd.DataFrame(results)
    
    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from texts in a DataFrame column.
        
        Args:
            df: Input DataFrame.
            text_column: Name of column containing text.
            show_progress: Show progress bar (default: True).
            
        Returns:
            Original DataFrame with feature columns appended.
        """
        texts = df[text_column].tolist()
        features_df = self.extract_batch(texts, show_progress)
        return pd.concat([df.reset_index(drop=True), features_df], axis=1)
