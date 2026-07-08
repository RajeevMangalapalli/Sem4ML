# Prints from `svm.py`

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
Accuracy                 0.7903
Precision                1.0000
Recall                   0.7845
F1 Score                 0.8793
Balanced Accuracy        0.8923

============================================================
Repeated Stratified K-Fold CV
============================================================
Metric                     Mean        Std
------------------------------------------
Accuracy                0.8380    0.0962
Precision               0.9916    0.0126
Recall                  0.8421    0.1103
F1 Score                0.9062    0.0587
Balanced Accuracy       0.7660    0.1828

============================================================
Final Model
============================================================
Model trained on full dataset.
Use LOOCV metrics as the primary performance estimate.

```
