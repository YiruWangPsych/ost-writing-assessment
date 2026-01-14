"""
Feature Selection Pipeline for Writing Quality Prediction

VIF-guided elastic net with bootstrap validation.
Implements the methodology from the accompanying paper.

Usage:
    python feature_selection.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import os

np.random.seed(42)


def calculate_vif(X):
    """Calculate VIF for standardized features."""
    X_scaled = StandardScaler().fit_transform(X)
    vif = []
    for i in range(X_scaled.shape[1]):
        try:
            v = variance_inflation_factor(X_scaled, i)
            v = v if np.isfinite(v) and v < 1e10 else np.inf
        except:
            v = np.inf
        vif.append({'feature': X.columns[i], 'VIF': v})
    return pd.DataFrame(vif).sort_values('VIF', ascending=False)


def iterative_vif_removal(df, features, threshold=10):
    """Remove features with VIF > threshold iteratively."""
    current = features.copy()
    removed = []
    while True:
        X = df[current].dropna()
        if X.shape[1] < 2:
            break
        vif_df = calculate_vif(X)
        if vif_df.iloc[0]['VIF'] <= threshold:
            break
        feat = vif_df.iloc[0]['feature']
        removed.append((feat, vif_df.iloc[0]['VIF']))
        current.remove(feat)
    return current, removed


def main():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 
                             'Writing_Assessment_Cleaned_Dataset_Tool_Specify.csv')
    df = pd.read_csv(data_path)
    
    target = 'Writing_Quality_Sum_Score'
    ost_features = [col for col in df.columns if col.startswith('OST_')]
    y = df[target]
    
    print(f"Dataset: N = {len(df)}")
    print(f"Target: M = {y.mean():.2f}, SD = {y.std():.2f}")
    print(f"OST features: {len(ost_features)}")
    
    # Feature selection: Top 15 by |correlation| -> VIF < 10
    correlations = [(f, abs(df[f].corr(y)), df[f].corr(y)) for f in ost_features]
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 features by |r|:")
    for i, (f, _, r) in enumerate(correlations[:15], 1):
        print(f"  {i:2d}. {f:40s} r = {r:+.3f}")
    
    top_15 = [f for f, _, _ in correlations[:15]]
    final_features, removed = iterative_vif_removal(df, top_15.copy(), threshold=10)
    
    print(f"\nRemoved {len(removed)} features:")
    for f, v in removed:
        print(f"  {f} (VIF = {v:.1f})")
    print(f"Final: {len(final_features)} features")
    
    # Fit model
    X = df[final_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=np.logspace(-3, 1, 30),
        cv=10, max_iter=5000, random_state=42
    )
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    n, p = len(y), len(final_features)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = mean_squared_error(y, y_pred)
    aic = n * np.log(mse) + 2 * p
    bic = n * np.log(mse) + p * np.log(n)
    
    print(f"\n{'='*50}")
    print("FULL MODEL")
    print(f"{'='*50}")
    print(f"N = {n}, Features = {p}")
    print(f"R² = {r2:.3f}, Adjusted R² = {adj_r2:.3f}")
    print(f"AIC = {aic:.1f}, BIC = {bic:.1f}")
    
    # VIF diagnostics
    vif_final = calculate_vif(X)
    print(f"VIF: mean = {vif_final['VIF'].mean():.2f}, max = {vif_final['VIF'].max():.2f}")
    
    # Coefficients
    print("\nCoefficients:")
    coef_df = pd.DataFrame({'feature': final_features, 'coef': model.coef_})
    coef_df = coef_df.sort_values('coef', key=abs, ascending=False)
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:45s} β = {row['coef']:+.3f}")
    
    # Bootstrap validation (threshold |β| > 0.05)
    print(f"\n{'='*50}")
    print("BOOTSTRAP VALIDATION (1,000 iterations)")
    print(f"{'='*50}")
    
    THRESH = 0.05
    n_boot = 1000
    boot_r2 = []
    selection_count = {f: 0 for f in final_features}
    
    for i in range(n_boot):
        idx = np.random.choice(len(y), len(y), replace=True)
        m = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_, max_iter=5000)
        m.fit(X_scaled[idx], y.iloc[idx])
        boot_r2.append(r2_score(y.iloc[idx], m.predict(X_scaled[idx])))
        for j, f in enumerate(final_features):
            if abs(m.coef_[j]) > THRESH:
                selection_count[f] += 1
    
    boot_r2 = np.array(boot_r2)
    print(f"R² 95% CI: [{np.percentile(boot_r2, 2.5):.3f}, {np.percentile(boot_r2, 97.5):.3f}]")
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='r2')
    print(f"CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    stability = pd.DataFrame([{'feature': f, 'freq': c/n_boot} for f, c in selection_count.items()])
    stability = stability.sort_values('freq', ascending=False)
    
    print(f"\nFeature stability (|β| > {THRESH}):")
    for _, row in stability.iterrows():
        mark = "*" if row['freq'] >= 0.80 else ""
        print(f"  {row['feature']:45s} {row['freq']*100:5.1f}% {mark}")
    print(f"Features ≥80%: {sum(stability['freq'] >= 0.80)}")
    
    # Group analysis
    print(f"\n{'='*50}")
    print("GROUP-SPECIFIC ANALYSIS")
    print(f"{'='*50}")
    
    native = df['Primary_Language'] == 0
    nonnative = df['Primary_Language'] == 1
    
    t, p_val = stats.ttest_ind(df.loc[native, target], df.loc[nonnative, target])
    print(f"Native (n={native.sum()}): M = {df.loc[native, target].mean():.2f}")
    print(f"Non-native (n={nonnative.sum()}): M = {df.loc[nonnative, target].mean():.2f}")
    print(f"t({len(df)-2}) = {t:.3f}, p = {p_val:.3f}")
    
    # Native model (top 6)
    native_df = df[native]
    y_n = native_df[target]
    corrs_n = [(f, abs(native_df[f].corr(y_n))) for f in ost_features]
    corrs_n.sort(key=lambda x: x[1], reverse=True)
    native_feats = [f for f, _ in corrs_n[:6]]
    
    X_n = StandardScaler().fit_transform(native_df[native_feats])
    m_n = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9], alphas=np.logspace(-2,1,20), 
                       cv=5, max_iter=5000, random_state=42)
    m_n.fit(X_n, y_n)
    r2_n = r2_score(y_n, m_n.predict(X_n))
    adj_r2_n = 1 - (1 - r2_n) * (len(y_n) - 1) / (len(y_n) - 6 - 1)
    
    print(f"\nNative: k = 6, R² = {r2_n:.3f}, Adj R² = {adj_r2_n:.3f}")
    for f, c in sorted(zip(native_feats, m_n.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {f:40s} β = {c:+.3f}")
    
    # Non-native model (3 features)
    nn_feats = ['OST_flesch_kincaid_grade_level', 'OST_context_sensitive_count', 'OST_error_count']
    nn_df = df[nonnative]
    y_nn = nn_df[target]
    
    X_nn = StandardScaler().fit_transform(nn_df[nn_feats])
    m_nn = ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9], alphas=np.logspace(-2,1,20),
                        cv=5, max_iter=5000, random_state=42)
    m_nn.fit(X_nn, y_nn)
    r2_nn = r2_score(y_nn, m_nn.predict(X_nn))
    adj_r2_nn = 1 - (1 - r2_nn) * (len(y_nn) - 1) / (len(y_nn) - 3 - 1)
    
    print(f"\nNon-native: k = 3, R² = {r2_nn:.3f}, Adj R² = {adj_r2_nn:.3f}")
    for f, c in sorted(zip(nn_feats, m_nn.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {f:40s} β = {c:+.3f}")
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Full model: {p} features, R² = {r2:.3f}, Adj R² = {adj_r2:.3f}")
    print(f"Native: 6 features, Adj R² = {adj_r2_n:.3f}")
    print(f"Non-native: 3 features, Adj R² = {adj_r2_nn:.3f}")


if __name__ == '__main__':
    main()
