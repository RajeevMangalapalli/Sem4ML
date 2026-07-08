# Prints from `random_forest_tuning.py`

```
Fitting 10 folds for each of 288 candidates, totalling 2880 fits
Grid Search Results

Best params : {'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Best BA (CV): 0.7462

Top-10 configurations (by mean Balanced Accuracy):
Rank  Params                                                                 Mean BA     Std
--------------------------------------------------------------------------------------------
1     {'criterion': 'log_loss', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}  0.7462  0.2070
2     {'criterion': 'log_loss', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}  0.7462  0.2070
3     {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}  0.7462  0.2070
4     {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}  0.7462  0.2070
5     {'criterion': 'log_loss', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}  0.7462  0.2070
6     {'criterion': 'log_loss', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}  0.7462  0.2070
7     {'criterion': 'log_loss', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}  0.7462  0.2070
8     {'criterion': 'log_loss', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}  0.7462  0.2070
9     {'criterion': 'log_loss', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}  0.7462  0.2070
10    {'criterion': 'log_loss', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}  0.7462  0.2070

LOOCV — Best Params vs Baseline
Computing LOOCV for baseline …
Computing LOOCV for best params …

Metric                 Baseline      Tuned        Δ
--------------------------------------------------
Accuracy                 0.8011     0.8011  +0.0000
Precision                0.9865     0.9865  +0.0000
Recall                   0.8066     0.8066  +0.0000
F1 Score                 0.8875     0.8875  +0.0000
Balanced Accuracy        0.7033     0.7033  +0.0000

```
