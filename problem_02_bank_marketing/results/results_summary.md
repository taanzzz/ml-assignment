# Results — Problem 02 (Bank Marketing, Logistic Regression)

Test set: 9,043 rows (stratified 20% split, random_state=42). Positive class share: 11.7%.

| Model | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | ROC-AUC (test) | ROC-AUC (5-fold CV) |
|---|---|---|---|---|---|---|
| Baseline | 0.9012 | 0.6445 | 0.3478 | 0.4518 | 0.9056 | 0.9066 |
| Class-balanced | 0.8457 | 0.4182 | 0.8147 | 0.5527 | 0.9079 | 0.9096 |

Hyper-parameter search over `C ∈ {0.01, 0.1, 1, 10}` → best `C = 0.1` (CV AUC 0.9072, test AUC 0.9061): regularisation strength barely matters here.

### Strongest learned effects (balanced model coefficients)
- Positive (→ subscribe): `poutcome=success` +1.82, `month=mar` +1.73, `duration` +1.53, `month=oct` +1.28, `month=sep` +0.98
- Negative (→ not subscribe): `month=jan` −1.30, `contact=unknown` −1.09, `month=jul` −1.07, `month=nov` −0.98, `month=aug` −0.90

### Figures in this folder (produced by the notebook run)
- `target_balance.png` — imbalanced target (88.3% no / 11.7% yes)
- `rates_by_category.png` — term-deposit rate inside each categorical feature
- `numeric_correlation.png` — correlation matrix of numeric features
- `confusion_matrices.png` — baseline vs class-balanced confusion matrices
- `roc_curves.png` — ROC curve comparison
- `coefficients.png` — strongest positive/negative drivers
