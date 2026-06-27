# Hyperparameter tuning for SVM: C, gamma, kernel
#
# Strategy:
#   - GridSearchCV with RepeatedStratifiedKFold (same CV as svm.py)
#   - Scoring: balanced_accuracy (professor's recommended metric)
#   - After grid search, re-evaluate best params via LOOCV for a fair
#     comparison with the baseline svm.py numbers
#
# Search space:
#   - kernel: rbf (baseline), poly, sigmoid, linear
#   - C: regularisation strength — smaller = softer margin
#   - gamma: RBF/poly/sigmoid kernel coefficient (not used by linear)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA = DATA_DIR / "processed" / "processed_data.csv"

df = pd.read_csv(PROCESSED_DATA)

X = df.drop(columns=["success"])
y = df["success"]

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# ── Grid Search ────────────────────────────────────────────────────────────────

param_grid = [
    {
        "kernel": ["rbf", "poly", "sigmoid"],
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    },
    {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1, 10, 100],
    },
]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

grid_search = GridSearchCV(
    SVC(class_weight="balanced", random_state=42),
    param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,         # refit best model on full data after search
    return_train_score=False,
)

grid_search.fit(X_scaled, y)


print("Grid Search Results")
print()
print(f"Best params : {grid_search.best_params_}")
print(f"Best BA (CV): {grid_search.best_score_:.4f}")

# Top-10 configurations
results_df = pd.DataFrame(grid_search.cv_results_)
top10 = (
    results_df[["params", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
print("\nTop-10 configurations (by mean Balanced Accuracy):")
print(f"{'Rank':<6}{'Params':<55}{'Mean BA':>8}{'Std':>8}")
print("-" * 77)
for i, row in top10.iterrows():
    print(f"{i+1:<6}{str(row['params']):<55}{row['mean_test_score']:>8.4f}{row['std_test_score']:>8.4f}")

# ── LOOCV with best params ─────────────────────────────────────────────────────
# Grid search CV gives the tuning estimate; LOOCV gives the comparable
# out-of-sample estimate on the full dataset (matches baseline evaluation).

print("LOOCV — Best Params vs Baseline")


best_params = grid_search.best_params_

def loocv_metrics(params):
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in loo.split(X_scaled):
        m = SVC(**params, class_weight="balanced", random_state=42)
        m.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = m.predict(X_scaled.iloc[test_idx])
    return {
        "Accuracy":          accuracy_score(y, y_pred),
        "Precision":         precision_score(y, y_pred, zero_division=0),
        "Recall":            recall_score(y, y_pred, zero_division=0),
        "F1 Score":          f1_score(y, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
    }, y_pred

baseline_params = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
print("Computing LOOCV for baseline …")
baseline_metrics, _ = loocv_metrics(baseline_params)
print("Computing LOOCV for best params …")
tuned_metrics, y_tuned_pred = loocv_metrics(best_params)

print(f"\n{'Metric':<20} {'Baseline':>10} {'Tuned':>10} {'Δ':>8}")
print("-" * 50)
for metric in baseline_metrics:
    b = baseline_metrics[metric]
    t = tuned_metrics[metric]
    print(f"{metric:<20} {b:>10.4f} {t:>10.4f} {t-b:>+8.4f}")

# ── Visualisation ──────────────────────────────────────────────────────────────
minority_mask = (y == y.value_counts().idxmin()).values
majority_mask  = ~minority_mask
sample_idx     = np.arange(len(y))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SVM Hyperparameter Tuning", fontsize=13)

# 1. Baseline vs Tuned metric comparison
metric_names  = list(baseline_metrics.keys())
baseline_vals = list(baseline_metrics.values())
tuned_vals    = list(tuned_metrics.values())
x_pos         = np.arange(len(metric_names))
width         = 0.35

axes[0].bar(x_pos - width/2, baseline_vals, width, label="Baseline (rbf, C=1)")
axes[0].bar(x_pos + width/2, tuned_vals,    width, label=f"Tuned {best_params}")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(metric_names, rotation=15, ha="right")
axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel("Score (LOOCV)")
axes[0].set_title("Baseline vs Best Params")
axes[0].legend(fontsize=7)
axes[0].grid(axis="y", alpha=0.3)
for i, (b, t) in enumerate(zip(baseline_vals, tuned_vals)):
    axes[0].text(i - width/2, b + 0.02, f"{b:.3f}", ha="center", fontsize=7)
    axes[0].text(i + width/2, t + 0.02, f"{t:.3f}", ha="center", fontsize=7)

# 2. Per-sample correctness for tuned model
correct_tuned = (y_tuned_pred == y.values).astype(int)
axes[1].scatter(sample_idx[majority_mask], correct_tuned[majority_mask],
                label="Majority", alpha=0.5, s=25)
axes[1].scatter(sample_idx[minority_mask], correct_tuned[minority_mask],
                label="Minority", color="red", marker="*", s=120, zorder=5)
axes[1].set_title("LOOCV Per-Sample Correctness\n(Tuned Model)")
axes[1].set_xlabel("Sample Index")
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Wrong", "Correct"])
axes[1].grid(alpha=0.3)
axes[1].legend()

# 3. Top-20 CV configurations scatter (mean BA vs std)
top20 = results_df.nlargest(20, "mean_test_score")
kernels  = [p["kernel"] for p in top20["params"]]
colors   = {"rbf": "steelblue", "linear": "orange", "poly": "green", "sigmoid": "red"}
for kernel in colors:
    mask = [k == kernel for k in kernels]
    axes[2].scatter(
        top20["std_test_score"].values[mask],
        top20["mean_test_score"].values[mask],
        label=kernel, color=colors[kernel], s=60, alpha=0.8,
    )
axes[2].set_xlabel("Std Balanced Accuracy")
axes[2].set_ylabel("Mean Balanced Accuracy")
axes[2].set_title("Top-20 Grid Configs\n(mean vs std, colour = kernel)")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
