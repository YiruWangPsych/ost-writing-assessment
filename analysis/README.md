# Feature Selection Analysis

VIF-guided elastic net feature selection pipeline for writing quality prediction.

## Method

1. **Feature Selection**: Top 15 features by |correlation| with target → VIF < 10 filtering → 13 features
2. **Model**: Elastic Net with 10-fold CV, L1 ratios [0.1, 0.3, 0.5, 0.7, 0.9]
3. **Validation**: Bootstrap (1,000 iterations), selection threshold |β| > 0.05

## Results

| Model | n | Features | R² | Adj R² |
|-------|---|----------|-----|--------|
| Full | 85 | 13 | .511 | .421 |
| Native | 47 | 6 | .590 | .528 |
| Non-native | 38 | 3 | .322 | .262 |

Bootstrap R² 95% CI: [.392, .709]  
Features with ≥80% selection: 8

## Usage

```bash
python feature_selection.py
```

## Files

- `feature_selection.py`: Main analysis script
- `../data/Writing_Assessment_Cleaned_Dataset_Tool_Specify.csv`: Input data
