from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA = PROCESSED_DATA_DIR / "processed_data.csv"


# 1. Load data
df = pd.read_csv(PROCESSED_DATA)

# 2. Drop identifiers to prevent memorization/leakage
# Also consider dropping 'failures' if it's a post-event outcome
X = df.drop(columns=["success", "name", "flight_number"])
y = df["success"]

# 3. Convert valid categorical features ('rocket_name', 'launchpad_name') to dummies
X = pd.get_dummies(X, drop_first=True)

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize Model
model = LogisticRegression(max_iter=1000, class_weight="balanced")

# 6. Proceed with KFold Cross Validation...
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics across folds
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

print("5-Fold Cross Validation")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    # Split the training data into fold-specific training and validation sets
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train the model on this specific fold's training subset
    model.fit(X_train_fold, y_train_fold)

    # Predict on the fold's validation subset
    y_pred = model.predict(X_val_fold)

    # Calculate metrics (zero_division=0 handles folds without positive predictions gracefully)
    acc = accuracy_score(y_val_fold, y_pred)
    prec = precision_score(y_val_fold, y_pred, zero_division=0)
    rec = recall_score(y_val_fold, y_pred, zero_division=0)
    f1 = f1_score(y_val_fold, y_pred, zero_division=0)

    # Save fold performance
    accuracy_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)

    print(f"\nFold {fold}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

# 7. Compute and display average cross-validation results
print("\n=== Average Cross-Validation Results ===")
print(f"Accuracy : {sum(accuracy_scores) / len(accuracy_scores):.4f}")
print(f"Precision: {sum(precision_scores) / len(precision_scores):.4f}")
print(f"Recall   : {sum(recall_scores) / len(recall_scores):.4f}")
print(f"F1 Score : {sum(f1_scores) / len(f1_scores):.4f}")

# 8. Train on the full training set and evaluate on the separate test set
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

print("\n=== Final Held-Out Test Set Results ===")
print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, y_test_pred, zero_division=0):.4f}")


folds = np.arange(1, len(accuracy_scores) + 1)
width = 0.2

plt.figure(figsize=(12, 6))

plt.bar(folds - 1.5 * width, accuracy_scores, width, label="Accuracy")
plt.bar(folds - 0.5 * width, precision_scores, width, label="Precision")
plt.bar(folds + 0.5 * width, recall_scores, width, label="Recall")
plt.bar(folds + 1.5 * width, f1_scores, width, label="F1 Score")

plt.title("5-Fold Cross Validation Metrics")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.xticks(folds, [f"Fold {i}" for i in folds])
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
