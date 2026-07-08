# Prints from `naive_bayes.py`

```
Class distribution:
success
1    181
0      5
Name: count, dtype: int64
Minority class ratio: 2.69%

============================================================
Leave-One-Out Cross Validation
============================================================
Metric                    Score
--------------------------------
Accuracy                 0.9624
Precision                0.9888
Recall                   0.9724
F1 Score                 0.9805
Balanced Accuracy        0.7862

============================================================
Repeated Stratified K-Fold CV
============================================================
Metric                     Mean        Std
------------------------------------------
Accuracy                0.9624    0.0257
Precision               0.9890    0.0135
Recall                  0.9724    0.0240
F1 Score                0.9804    0.0135
Balanced Accuracy       0.7862    0.2441

============================================================
Final Model
============================================================
Model trained on full dataset.
Use LOOCV metrics as the primary performance estimate.

```
