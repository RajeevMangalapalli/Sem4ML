# Prints from `svm_tuning.py`

```
Fitting 50 folds for each of 95 candidates, totalling 4750 fits
Grid Search Results

Best params : {'C': 1, 'gamma': 1, 'kernel': 'rbf'}
Best BA (CV): 0.9033

Top-10 configurations (by mean Balanced Accuracy):
Rank  Params                                                  Mean BA     Std
-----------------------------------------------------------------------------
1     {'C': 10, 'gamma': 1, 'kernel': 'rbf'}                   0.9033  0.0312
2     {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}             0.9033  0.0312
3     {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}            0.9033  0.0312
4     {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}                0.9033  0.0312
5     {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}             0.9033  0.0312
6     {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}               0.9033  0.0312
7     {'C': 100, 'gamma': 1, 'kernel': 'rbf'}                  0.9033  0.0312
8     {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}                 0.9033  0.0312
9     {'C': 1, 'gamma': 1, 'kernel': 'rbf'}                    0.9033  0.0312
10    {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}              0.9033  0.0312
LOOCV — Best Params vs Baseline
Computing LOOCV for baseline …
Computing LOOCV for best params …

Metric                 Baseline      Tuned        Δ
--------------------------------------------------
Accuracy                 0.7903     0.8118  +0.0215
Precision                1.0000     1.0000  +0.0000
Recall                   0.7845     0.8066  +0.0221
F1 Score                 0.8793     0.8930  +0.0137
Balanced Accuracy        0.8923     0.9033  +0.0110

```
