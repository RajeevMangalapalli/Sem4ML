# Prints from `random_forest.py`

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
Accuracy                 0.8011
Precision                0.9865
Recall                   0.8066
F1 Score                 0.8875
Balanced Accuracy        0.7033

============================================================
Repeated Stratified K-Fold CV
============================================================
Metric                     Mean        Std
------------------------------------------
Accuracy                0.8688    0.0874
Precision               0.9894    0.0132
Recall                  0.8760    0.1019
F1 Score                0.9254    0.0524
Balanced Accuracy       0.7430    0.1985

============================================================
Final Model
============================================================
Model trained on full dataset.
Use LOOCV metrics as the primary performance estimate.

```
