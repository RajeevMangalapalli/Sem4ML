from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    make_scorer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA = DATA_DIR / "processed" / "processed_data.csv"

df = pd.read_csv(PROCESSED_DATA)

X = df.drop(columns=["success", "name", "flight_number"])
y = df["success"]

X = pd.get_dummies(X, drop_first=True)

# Class distribution
print("Class distribution:")
print(y.value_counts())
print(f"Minority class ratio: {y.value_counts(normalize=True).min():.2%}")

# 5 minority samples
# LOOCV
print("\n" + "=" * 60)
print("Leave-One-Out Cross Validation")
print("=" * 60)

loo = LeaveOneOut()
model = GaussianNB()

y_loo_pred = np.zeros(len(y), dtype=int)

for train_idx, test_idx in loo.split(X):
    model = GaussianNB()

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]

    model.fit(X_train, y_train)

    y_loo_pred[test_idx] = model.predict(X_test)

loo_metrics = {
    "Accuracy": accuracy_score(y, y_loo_pred),
    "Precision": precision_score(y, y_loo_pred, zero_division=0),
    "Recall": recall_score(y, y_loo_pred, zero_division=0),
    "F1 Score": f1_score(y, y_loo_pred, zero_division=0),
    "Balanced Accuracy": balanced_accuracy_score(y, y_loo_pred),
}

print(f"{'Metric':<20} {'Score':>10}")
print("-" * 32)

for metric, value in loo_metrics.items():
    print(f"{metric:<20} {value:>10.4f}")

# Repeated Stratified K-Fold
# 5 folds x 20 repeats
print("\n" + "=" * 60)
print("Repeated Stratified K-Fold CV")
print("=" * 60)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}

cv_results = cross_validate(GaussianNB(), X, y, cv=rskf, scoring=scoring)

print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
print("-" * 42)

for label, key in [
    ("Accuracy", "test_accuracy"),
    ("Precision", "test_precision"),
    ("Recall", "test_recall"),
    ("F1 Score", "test_f1"),
    ("Balanced Accuracy", "test_balanced_accuracy"),
]:
    scores = cv_results[key]

    print(f"{label:<20}{scores.mean():>10.4f}{scores.std():>10.4f}")

# Train final model on all available data
print("\n" + "=" * 60)
print("Final Model")
print("=" * 60)

final_model = GaussianNB()
final_model.fit(X, y)

print("Model trained on full dataset.")
print("Use LOOCV metrics as the primary performance estimate.")

# Visualization
minority_mask = (y == y.value_counts().idxmin()).values
majority_mask = ~minority_mask

metric_names = list(loo_metrics.keys())
metric_values = list(loo_metrics.values())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall metrics
axes[0].bar(
    metric_names,
    metric_values,
)

axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Score")
axes[0].set_title("LOOCV Overall Metrics (GaussianNB)")
axes[0].grid(axis="y", alpha=0.3)

for i, value in enumerate(metric_values):
    axes[0].text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )

# Per-sample correctness
correct = (y_loo_pred == y.values).astype(int)
sample_idx = np.arange(len(y))

axes[1].scatter(
    sample_idx[majority_mask],
    correct[majority_mask],
    label="Majority class",
    alpha=0.5,
    s=25,
)

axes[1].scatter(
    sample_idx[minority_mask],
    correct[minority_mask],
    label="Minority class",
    color="red",
    marker="*",
    s=120,
    zorder=5,
)

axes[1].set_title("LOOCV Per-Sample Correctness\n(1 = Correct, 0 = Incorrect)")
axes[1].set_xlabel("Sample Index")
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Wrong", "Correct"])
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
