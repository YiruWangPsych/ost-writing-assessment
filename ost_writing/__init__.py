"""
OST Writing Assessment Toolkit
==============================

An open-source toolkit for extracting linguistic features from written texts.

Example usage:
    >>> from ost_writing import FeatureExtractor
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract("Sample text here.")
"""

from .extractor import FeatureExtractor

__version__ = "0.1.0"
__all__ = ["FeatureExtractor"]
