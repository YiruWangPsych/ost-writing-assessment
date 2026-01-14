# OST Writing Assessment Toolkit

Python toolkit for extracting linguistic features from written texts. Uses NLTK, spaCy, and LanguageTool to compute 20 indicators for writing quality research.

## Features (20 indicators)

| Category | Features |
|----------|----------|
| Surface | `word_count`, `avg_sentence_length`, `num_complex_sentences` |
| Lexical | `lexical_diversity`, `vocabulary_sophistication`, `polysemy_word` |
| Syntactic | `sentence_type_diversity`, `syntactic_simplicity`, `information_density` |
| Accuracy | `error_count`, `context_sensitive_count` |
| Readability | `flesch_kincaid_grade_level`, `text_ease` |
| Cohesion | `referential_cohesion`, `deep_cohesion` |
| Variability | `word_length_variance`, `syllable_variance` |
| Clausal | `num_t_units`, `mean_length_t_unit`, `dependent_clauses_per_t_unit` |

See [docs/feature_definitions.md](docs/feature_definitions.md) for detailed definitions.

## Installation

```bash
# Create virtual environment (python <= 3.13 )
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package in editable mode for potential corrections 
pip install -e .

# Download language resources
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"
```

**Note**: `error_count` requires Java (for LanguageTool). Check with `java -version`.

## Quick Start

### Single text
```python
from ost_writing import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract("College education provides valuable experiences...")
print(features)
```

### Batch extraction
```python
texts = ["First essay...", "Second essay..."]
df = extractor.extract_batch(texts)
df.to_csv("features.csv", index=False)
```

### Run example
```bash
python examples/example_usage.py
```

## Feature Selection Analysis

The `analysis/` directory contains the VIF-guided elastic net feature selection pipeline from the accompanying paper.

```bash
cd analysis
python feature_selection.py
```

Key results:
- Full model: 13 features, R² = .511, Adjusted R² = .421
- Native speakers (n=47): 6 features, Adjusted R² = .525
- Non-native speakers (n=38): 3 features, Adjusted R² = .259

See `analysis/README.md` for details.

## Repository Structure

```
ost-writing-assessment/
├── ost_writing/           # Core feature extraction
│   ├── features/          # Feature modules by category
│   ├── utils/             # NLP setup and text processing
│   └── extractor.py       # Main FeatureExtractor class
├── analysis/              # Feature selection pipeline
│   ├── feature_selection.py
│   └── README.md
├── examples/
│   ├── example_usage.py
│   └── sample_data/       # 10 de-identified writing samples
├── data/                  # Analysis dataset
├── docs/                  # Feature definitions
├── requirements.txt
└── pyproject.toml
```

## Data

- `data/Writing_Assessment_Cleaned_Dataset_Tool_Specify.csv`: Pre-extracted features for 85 college students
- `examples/sample_data/`: 10 de-identified writing samples (6 non-native, 4 native speakers) across quality levels

## Citation

```
[To be added upon publication]
```

## License

MIT License. See [LICENSE](LICENSE) for details.
