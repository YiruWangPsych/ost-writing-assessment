#!/usr/bin/env python
"""
Example usage of OST Writing Assessment Toolkit.

This script demonstrates:
    1. Single text feature extraction
    2. Batch processing from files
    3. DataFrame integration
"""

import os

from ost_writing import FeatureExtractor


def main():
    # Initialize extractor
    extractor = FeatureExtractor(prefix="OST_")
    
    # Example 1: Single text extraction
    print("=" * 60)
    print("Example 1: Single Text Feature Extraction")
    print("=" * 60)
    
    sample_text = """
    A valuable experience that I learned during my college career was 
    when I got my first 'bad grade.' I received a C, before all I got 
    were A's. The grade was completely my fault, because the class was 
    relatively easy. I learned to check my grades no matter how much 
    anxiety it gives me.
    """
    
    features = extractor.extract(sample_text)
    
    print("\nExtracted features (showing first 5):")
    for i, (key, value) in enumerate(features.items()):
        if i >= 5:
            break
        print(f"  {key}: {value}")
    
    # Example 2: Batch processing from files
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing from Files")
    print("=" * 60)
    
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    
    if os.path.exists(sample_dir):
        texts = []
        filenames = []
        
        for filename in sorted(os.listdir(sample_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(sample_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                filenames.append(filename)
        
        if texts:
            df = extractor.extract_batch(texts, show_progress=True)
            df.insert(0, "filename", filenames)
            
            print(f"\nProcessed {len(texts)} files.")
            print("\nFeature summary (first 5 features):")
            print(df[["filename"] + list(df.columns[1:6])].to_string(index=False))
            
            # Save results
            output_path = os.path.join(sample_dir, "extracted_features.csv")
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
        else:
            print("No .txt files found in sample_data directory.")
    else:
        print("sample_data directory not found.")
    
    # Example 3: Using without prefix
    print("\n" + "=" * 60)
    print("Example 3: Extraction Without Prefix")
    print("=" * 60)
    
    extractor_no_prefix = FeatureExtractor(prefix="")
    features_simple = extractor_no_prefix.extract(sample_text)
    
    print("\nFeature names without prefix:")
    print(list(features_simple.keys())[:5], "...")


if __name__ == "__main__":
    main()
