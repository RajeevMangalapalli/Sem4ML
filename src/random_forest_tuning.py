# Parameters that meaningfully influence model outcome:
#   - n_estimators:  number of trees — more trees stabilise predictions
#     but increase computation
#   - max_depth:     maximum tree depth — limits overfitting; None = expand
#     until leaves are pure (or constrained by min_samples_split/leaf)
#   - min_samples_split: minimum samples required to split an internal node —
#     higher values regularise more aggressively
#   - min_samples_leaf:   minimum samples required at a leaf node —
#     prevents the model from learning patterns on too few samples
#   - max_features:  number of features considered at each split —
#     controls diversity across trees; "sqrt" is the default
#   - criterion:     split quality measure — gini vs entropy
#
# Excluded (no meaningful impact on model outcome):
#   - bootstrap:     whether to bootstrap samples when building trees;
#     False would make every tree identical, defeating the ensemble purpose
#   - max_samples:   fraction or number of samples to draw for each tree;
#     only relevant if bootstrap=True; defaults to using all samples
#   - max_leaf_nodes: alternative depth control — limits total leaf count
#     instead of depth; redundant when max_depth is tuned
#   - min_impurity_decrease: split threshold on impurity reduction;
#     negligible effect compared to min_samples_split/leaf
#   - ccp_alpha:     cost-complexity pruning parameter; primarily used
#     for post-hoc tree reduction, rarely tuned in Random Forest
#   - warm_start:    reuses the previous fit to add more estimators;
#     purely a computational convenience
#   - n_jobs:        number of parallel jobs for fitting and prediction;
#     solely computational, does not affect model decisions
#   - verbose:       controls logging verbosity during fitting;
#     purely diagnostic
#   - oob_score:     whether to compute out-of-bag estimates during
#     training; only affects evaluation, not the fitted model
#   - random_state:  seed for reproducibility; not a tunable parameter

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import persist

STEM = Path(__file__).stem
persist.begin(STEM)

from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
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

# Grid Search

param_grid = {
    "n_estimators": [100, 300],
    # "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5],
    # "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2],
    # "max_features": ["sqrt", "log2"],
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy", "log_loss"],
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=1,
    verbose=1,
    refit=True,
    return_train_score=False,
)

grid_search.fit(X, y)

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
print(f"{'Rank':<6}{'Params':<70}{'Mean BA':>8}{'Std':>8}")
print("-" * 92)
for i, row in top10.iterrows():
    print(
        f"{i + 1:<6}{str(row['params']):<70}{row['mean_test_score']:>8.4f}{row['std_test_score']:>8.4f}"
    )

# LOOCV with best params
# Grid search CV gives the tuning estimate; LOOCV gives the comparable
# out-of-sample estimate on the full dataset (matches baseline evaluation).

print("\nLOOCV — Best Params vs Baseline")


def loocv_metrics(params):
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in loo.split(X):
        m = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=42,
        )
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = m.predict(X.iloc[test_idx])
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
    }, y_pred


best_params = grid_search.best_params_

baseline_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "gini",
}
print("Computing LOOCV for baseline …")
baseline_metrics, _ = loocv_metrics(baseline_params)
print("Computing LOOCV for best params …")
tuned_metrics, y_tuned_pred = loocv_metrics(best_params)

print(f"\n{'Metric':<20} {'Baseline':>10} {'Tuned':>10} {'Δ':>8}")
print("-" * 50)
for metric in baseline_metrics:
    b = baseline_metrics[metric]
    t = tuned_metrics[metric]
    print(f"{metric:<20} {b:>10.4f} {t:>10.4f} {t - b:>+8.4f}")

# Visualisation
minority_mask = (y == y.value_counts().idxmin()).values
majority_mask = ~minority_mask
sample_idx = np.arange(len(y))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Random Forest Hyperparameter Tuning", fontsize=13)

# 1. Baseline vs Tuned metric comparison
metric_names = list(baseline_metrics.keys())
baseline_vals = list(baseline_metrics.values())
tuned_vals = list(tuned_metrics.values())
x_pos = np.arange(len(metric_names))
width = 0.35

axes[0].bar(x_pos - width / 2, baseline_vals, width, label="Baseline (default params)")
axes[0].bar(x_pos + width / 2, tuned_vals, width, label=f"Tuned {best_params}")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(metric_names, rotation=15, ha="right")
axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel("Score (LOOCV)")
axes[0].set_title("Baseline vs Best Params")
axes[0].legend(fontsize=7)
axes[0].grid(axis="y", alpha=0.3)
for i, (b, t) in enumerate(zip(baseline_vals, tuned_vals)):
    axes[0].text(i - width / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=7)
    axes[0].text(i + width / 2, t + 0.02, f"{t:.3f}", ha="center", fontsize=7)

# 2. Per-sample correctness for tuned model
correct_tuned = (y_tuned_pred == y.values).astype(int)
axes[1].scatter(
    sample_idx[majority_mask],
    correct_tuned[majority_mask],
    label="Majority",
    alpha=0.5,
    s=25,
)
axes[1].scatter(
    sample_idx[minority_mask],
    correct_tuned[minority_mask],
    label="Minority",
    color="red",
    marker="*",
    s=120,
    zorder=5,
)
axes[1].set_title("LOOCV Per-Sample Correctness\n(Tuned Model)")
axes[1].set_xlabel("Sample Index")
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Wrong", "Correct"])
axes[1].grid(alpha=0.3)
axes[1].legend()

# 3. Top-20 CV configurations scatter (mean BA vs std)
top20 = results_df.nlargest(20, "mean_test_score")
max_features_vals = [str(p["max_features"]) for p in top20["params"]]
# colors = {"sqrt": "steelblue", "log2": "orange"}
colors = {"sqrt": "steelblue", "log2": "orange", "None": "green"}
for mf in colors:
    mask = [v == mf for v in max_features_vals]
    axes[2].scatter(
        top20["std_test_score"].values[mask],
        top20["mean_test_score"].values[mask],
        label=f"max_features={mf}",
        color=colors[mf],
        s=60,
        alpha=0.8,
    )
axes[2].set_xlabel("Std Balanced Accuracy")
axes[2].set_ylabel("Mean Balanced Accuracy")
axes[2].set_title("Top-20 Grid Configs\n(mean vs std, colour = max_features)")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

md = [f"# Plot data — `{STEM}.py`\n"]
md.append(f"## Best params\n\n`{grid_search.best_params_}`\n")
md.append(f"Best BA (CV): **{grid_search.best_score_:.4f}**\n")
md.append("\n## Top-10 configurations (by mean Balanced Accuracy)\n")
md.append(persist.md_table(
    ["Rank", "Params", "Mean BA", "Std"],
    [
        [i + 1, str(row["params"]), f"{row['mean_test_score']:.4f}", f"{row['std_test_score']:.4f}"]
        for i, row in top10.iterrows()
    ],
))
md.append("\n## LOOCV — Baseline vs Best params\n")
md.append(f"Baseline params: `{baseline_params}`\n")
md.append(f"Tuned params: `{best_params}`\n")
md.append(persist.md_table(
    ["Metric", "Baseline", "Tuned", "Δ"],
    [
        [m, f"{baseline_metrics[m]:.4f}", f"{tuned_metrics[m]:.4f}",
         f"{tuned_metrics[m] - baseline_metrics[m]:+.4f}"]
        for m in baseline_metrics
    ],
))
md.append("\n## LOOCV per-sample correctness (Tuned model, 1=correct, 0=wrong)\n")
correct_tuned = (y_tuned_pred == y.values).astype(int)
md.append(persist.md_table(
    ["Sample index", "True label", "Predicted", "Correct", "Class"],
    [
        [i, int(y.values[i]), int(y_tuned_pred[i]), int(correct_tuned[i]),
         "minority" if minority_mask[i] else "majority"]
        for i in range(len(y))
    ],
))
md.append("\n## Top-20 grid configs (mean vs std, colour = max_features)\n")
md.append(persist.md_table(
    ["Params", "Mean BA", "Std BA", "max_features"],
    [
        [str(p), f"{r['mean_test_score']:.4f}", f"{r['std_test_score']:.4f}",
         str(p["max_features"])]
        for _, r in top20.iterrows()
        for p in [r["params"]]
    ],
))
persist.save_plot_data(STEM, "".join(md))

persist.end()
