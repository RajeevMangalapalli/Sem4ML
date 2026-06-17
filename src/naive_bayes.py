from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA = PROCESSED_DATA_DIR / "processed_data.csv"


df = pd.read_csv(PROCESSED_DATA)

X = df.drop(columns=["success", "name", "flight_number"])
y = df["success"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
balanced_accuracy_scores = []

print("5-Fold Cross Validation")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]

    model.fit(X_train_fold, y_train_fold)

    y_pred = model.predict(X_val_fold)

    acc = accuracy_score(y_val_fold, y_pred)
    prec = precision_score(y_val_fold, y_pred, zero_division=0)
    rec = recall_score(y_val_fold, y_pred, zero_division=0)
    f1 = f1_score(y_val_fold, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_val_fold, y_pred)

    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)
    balanced_accuracy_scores.append(bal_acc)

    print(f"\nFold {fold}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")

print("\n=== Average Cross-Validation Results ===")
print(f"Accuracy         : {sum(accuracy_scores) / len(accuracy_scores):.4f}")
print(f"Precision        : {sum(precision_scores) / len(precision_scores):.4f}")
print(f"Recall           : {sum(recall_scores) / len(recall_scores):.4f}")
print(f"F1 Score         : {sum(f1_scores) / len(f1_scores):.4f}")
print(
    f"Balanced Accuracy: {sum(balanced_accuracy_scores) / len(balanced_accuracy_scores):.4f}"
)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

print("\n=== Final Held-Out Test Set Results ===")
print(f"Accuracy         : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision        : {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall           : {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1 Score         : {f1_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_test_pred):.4f}")


folds = np.arange(1, len(accuracy_scores) + 1)
width = 0.15

plt.figure(figsize=(14, 6))

plt.bar(folds - 2 * width, accuracy_scores, width, label="Accuracy")
plt.bar(folds - width, precision_scores, width, label="Precision")
plt.bar(folds, recall_scores, width, label="Recall")
plt.bar(folds + width, f1_scores, width, label="F1 Score")
plt.bar(folds + 2 * width, balanced_accuracy_scores, width, label="Balanced Accuracy")

plt.title("5-Fold Cross Validation Metrics (Naive Bayes)")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.xticks(folds, [f"Fold {i}" for i in folds])
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
