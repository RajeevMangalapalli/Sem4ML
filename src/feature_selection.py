"""
Primary method: mRMR - model‑agnostic, uses mutual information
only.  No classifier bias, no CV inside the selection loop.

Wrapper methods (forward, backward, fwd+bwd) are parameterised by
classifier so that the optimal subset can be found *per model* (NB, SVM,
…).  CV strategy matches naive_bayes.py / svm.py (RepeatedStratifiedKFold
for wrapper evaluation; final LOOCV on the full dataset).

Key design choices:
  - mRMR scaling:  features are StandardScaler‑ed before MI computation
    because sklearn's MI estimators use k‑NN, which is scale‑sensitive.
  - Wrapper scaling: per‑fold via Pipeline so scaling never leaks.
  - Dynamic n_splits: RSKF adapts to the minority‑class count so wrappers
    never hit an empty‑fold error.
  - Hold‑out test set:  used for unbiased final evaluation.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import persist

STEM = Path(__file__).stem
persist.begin(STEM)

from sklearn.model_selection import (
    train_test_split,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    make_scorer,
)

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
TARGET = "success"
MAX_FEATURES = 5
RANDOM_STATE = 42

LOOCV = LeaveOneOut()


# Data loading
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET, "name"], errors="ignore")
    y = df[TARGET]
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def get_classifier(name):
    clf_map = {
        "nb": GaussianNB(),
        "svm": SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
        "lr": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    if name not in clf_map:
        raise ValueError(f"Unknown classifier '{name}'.  Choose: {list(clf_map)}")
    return clf_map[name]


def needs_scaling(name):
    """GaussianNB needs no scaling; SVM (RBF) and LR benefit from it."""
    return name in {"svm", "lr"}


def make_cv(y, n_repeats=5):
    """Return RSKF with n_splits bounded by the minority‑class count.

    If the training subset has < 2 minority samples we fall back to 2‑fold
    so that balanced‑accuracy can still be computed (one sample in each
    fold).
    """
    n_minority = y.value_counts().min()
    n_splits = max(2, min(5, n_minority))
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )


# Subset evaluation helper (used by every wrapper method)
def evaluate_subset(X, y, subset, clf, cv, scale=False):
    """Return mean **balanced accuracy** over all RSKF folds.

    Scaling (when enabled) is wrapped in a Pipeline so that ``fit`` /
    ``transform`` stays inside each fold — no leakage.
    """
    if len(subset) == 0:
        return 0.0

    X_sub = X[list(subset)]
    estimator = make_pipeline(StandardScaler(), clf) if scale else clf

    cv_results = cross_validate(
        estimator,
        X_sub,
        y,
        cv=cv,
        scoring={"ba": make_scorer(balanced_accuracy_score)},
        n_jobs=-1,
        error_score="raise",
    )
    return cv_results["test_ba"].mean()


# 1.  mRMR  (filter — model‑agnostic, no classifier, no CV)
def mrmr_selection(X, y, n_features):
    """Minimum‑Redundancy Maximum‑Relevance (greedy forward).

    * Relevance:   D(f) = MI(f, y)           ← ``mutual_info_classif``
    * Redundancy:  R(f) = mean_{s in S} MI(f, s)  ← ``mutual_info_regression``
    * Criterion:   max  D(f) − R(f)

    Features are scaled *once* before MI because both sklearn estimators
    internally use k‑NN distance computations.
    """
    features = list(X.columns)
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)

    # Relevance
    mi_target = dict(
        zip(features, mutual_info_classif(X_s, y, random_state=RANDOM_STATE))
    )

    # Redundancy (pairwise MI)
    mi_pairs = {}
    for f in features:
        mi_pairs[f] = dict(
            zip(
                features,
                mutual_info_regression(X_s, X_s[f], random_state=RANDOM_STATE),
            )
        )

    selected = []
    remaining = features.copy()

    for _ in range(n_features):
        best_score, best_feature = -np.inf, None
        for f in remaining:
            relevance = mi_target[f]
            redundancy = (
                np.mean([mi_pairs[f][s] for s in selected]) if selected else 0.0
            )
            score = relevance - redundancy
            if score > best_score:
                best_score, best_feature = score, f
        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected, {"mi_target": mi_target, "order": selected.copy()}


# 2.  Forward selection  (wrapper)
def forward_selection(X, y, clf, cv, max_features=MAX_FEATURES, scale=False):
    """Greedy forward wrapper — starts empty, adds best feature each step."""
    features = list(X.columns)
    selected = []
    best_score = 0.0

    while len(selected) < max_features:
        candidates = {
            f: evaluate_subset(X, y, selected + [f], clf, cv, scale)
            for f in features
            if f not in selected
        }
        best_candidate = max(candidates, key=candidates.get)
        candidate_score = candidates[best_candidate]
        if candidate_score > best_score:
            selected.append(best_candidate)
            best_score = candidate_score
        else:
            break

    return selected, best_score


# 3.  Backward elimination  (wrapper)
def backward_elimination(X, y, clf, cv, scale=False):
    """Greedy backward wrapper — starts with all features, removes worst."""
    selected = list(X.columns)
    best_score = evaluate_subset(X, y, selected, clf, cv, scale)

    while len(selected) > 1:
        scores = {
            f: evaluate_subset(X, y, [x for x in selected if x != f], clf, cv, scale)
            for f in selected
        }
        best_to_remove = max(scores, key=scores.get)
        if scores[best_to_remove] >= best_score:
            selected.remove(best_to_remove)
            best_score = scores[best_to_remove]
        else:
            break

    return selected, best_score


# 4.  Forward + Backward  (wrapper)
def forward_backward_selection(X, y, clf, cv, max_features=MAX_FEATURES, scale=False):
    """Alternate forward and backward passes until the selected set stabilises."""
    selected, _ = forward_selection(X, y, clf, cv, max_features, scale)

    for _ in range(20):  # safety cap
        prev = set(selected)

        # Backward pass
        selected, best_score = backward_elimination(X[selected], y, clf, cv, scale)

        # Forward pass
        candidates = {
            f: evaluate_subset(X, y, selected + [f], clf, cv, scale)
            for f in X.columns
            if f not in selected and len(selected) < max_features
        }
        if candidates:
            best_add = max(candidates, key=candidates.get)
            if candidates[best_add] > best_score:
                selected.append(best_add)

        if set(selected) == prev:
            break

    final_score = evaluate_subset(X, y, selected, clf, cv, scale)
    return selected, final_score


# Final evaluation helpers
def evaluate_loocv(X, y, subset, clf, scale=False):
    """LOOCV on a fixed feature subset — matches naive_bayes.py / svm.py."""
    X_sub = X[list(subset)]
    y_pred = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in LOOCV.split(X_sub):
        X_tr, X_te = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
        y_tr = y.iloc[train_idx]

        if scale:
            sc = StandardScaler()
            X_tr = pd.DataFrame(
                sc.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
            )
            X_te = pd.DataFrame(
                sc.transform(X_te), columns=X_te.columns, index=X_te.index
            )

        clf.fit(X_tr, y_tr)
        y_pred[test_idx] = clf.predict(X_te)

    return {
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


def evaluate_holdout(X_tr, y_tr, X_te, y_te, subset, clf, scale=False):
    """Train on training set, predict on hold‑out test set."""
    X_tr_sub = X_tr[list(subset)]
    X_te_sub = X_te[list(subset)]

    if scale:
        sc = StandardScaler()
        X_tr_sub = pd.DataFrame(
            sc.fit_transform(X_tr_sub), columns=X_tr_sub.columns, index=X_tr_sub.index
        )
        X_te_sub = pd.DataFrame(
            sc.transform(X_te_sub), columns=X_te_sub.columns, index=X_te_sub.index
        )

    clf.fit(X_tr_sub, y_tr)
    y_pred = clf.predict(X_te_sub)

    return {
        "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
    }


# Printing helpers
def _hr(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _metrics(metrics: dict, indent=4):
    for k, v in metrics.items():
        print(f"{' ' * indent}{k:<19s} {v:.4f}")


# Plotting
def plot_wrapper_results(all_results, mrmr_subset):
    n = len(all_results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    methods = ["Forward", "Backward", "Fwd+Bwd"]
    colors = ["steelblue", "tomato", "mediumseagreen"]

    for ax, (name, res) in zip(axes, all_results.items()):
        ba = [res["forward"][1], res["backward"][1], res["forward_backward"][1]]
        subsets = [res["forward"][0], res["backward"][0], res["forward_backward"][0]]

        x = np.arange(len(methods))
        bars = ax.bar(x, ba, color=colors)
        ax.set_title(name.upper(), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Balanced Accuracy (RSKF)")
        ax.grid(axis="y", alpha=0.3)

        for bar, val, sub in zip(bars, ba, subsets):
            label = ", ".join(sub[:2]) + ("…" if len(sub) > 2 else "")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}\n({label})",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        f"Wrapper Feature Selection  —  CV: RSKF (per classifier)\n"
        f"mRMR subset: {', '.join(mrmr_subset[:3])}…",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# Main
if __name__ == "__main__":
    X, y = load_data()
    print(f"Features       : {len(X.columns)}  ({', '.join(X.columns)})")
    print(f"Samples        : {len(X)}")
    print(f"Classes        : {dict(y.value_counts())}")
    print(
        f"Imbalance      : {y.value_counts(normalize=True).max():.1%} / "
        f"{y.value_counts(normalize=True).min():.1%}"
    )

    # ── Train / validation / test split ──────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain/val      : {len(X_tv)}  ({dict(y_tv.value_counts())})")
    print(f"Test           : {len(X_test)}  ({dict(y_test.value_counts())})")

    # Dynamic CV for wrappers (adapts to minority count in X_tv)
    cv = make_cv(y_tv)
    print(f"Inner CV       : {cv}")

    # ── 1.  mRMR  (model‑agnostic filter) ────────────────────────────
    _hr("mRMR — filter (model‑agnostic, MI‑based)")
    t0 = time.time()
    mrmr_subset, mrmr_info = mrmr_selection(X_tv, y_tv, n_features=MAX_FEATURES)
    print(f"  Selected     : {mrmr_subset}")
    print(f"  MI with target:")
    for f in mrmr_info["order"]:
        print(f"    {f:<35s}  MI = {mrmr_info['mi_target'][f]:.4f}")
    print(f"  Time         : {time.time() - t0:.1f}s")

    # ── 2.  Wrapper methods per classifier ───────────────────────────
    classifiers = [
        ("nb", get_classifier("nb"), False),
        ("svm", get_classifier("svm"), True),
        ("lr", get_classifier("lr"), True),
    ]

    all_wrapper = {}

    for name, clf, scale in classifiers:
        _hr(f"Wrapper Methods  —  classifier: {name.upper()}")

        t0 = time.time()
        fwd, fwd_ba = forward_selection(X_tv, y_tv, clf, cv, MAX_FEATURES, scale)
        print(f"  Forward            : {fwd}")
        print(f"    BA (RSKF) = {fwd_ba:.4f}  [{time.time() - t0:.1f}s]")

        t0 = time.time()
        bwd, bwd_ba = backward_elimination(X_tv, y_tv, clf, cv, scale)
        print(f"  Backward           : {bwd}")
        print(f"    BA (RSKF) = {bwd_ba:.4f}  [{time.time() - t0:.1f}s]")

        t0 = time.time()
        fb, fb_ba = forward_backward_selection(X_tv, y_tv, clf, cv, MAX_FEATURES, scale)
        print(f"  Forward+Backward   : {fb}")
        print(f"    BA (RSKF) = {fb_ba:.4f}  [{time.time() - t0:.1f}s]")

        all_wrapper[name] = {
            "forward": (fwd, fwd_ba),
            "backward": (bwd, bwd_ba),
            "forward_backward": (fb, fb_ba),
        }

    # ── 3.  Best subset per classifier ───────────────────────────────
    _hr("Best subset per classifier")
    best_subsets = {"mrmr": (mrmr_subset, "mRMR (filter)")}
    for name, _, _ in classifiers:
        best_method = max(all_wrapper[name], key=lambda k: all_wrapper[name][k][1])
        subset, ba = all_wrapper[name][best_method]
        best_subsets[name] = (subset, best_method)
        print(f"  {name.upper():>3s}  {best_method:<20s}  BA={ba:.4f}  →  {subset}")

    # ── 4.  Hold‑out evaluation ──────────────────────────────────────
    _hr("Hold‑out test evaluation")
    for label, (subset, method) in best_subsets.items():
        if label == "mrmr":
            for c_name, c_clf, c_scale in classifiers:
                m = evaluate_holdout(X_tv, y_tv, X_test, y_test, subset, c_clf, c_scale)
                print(f"\n  mRMR subset × {c_name.upper()}  ({len(subset)} features)")
                _metrics(m)
        else:
            c_clf = get_classifier(label)
            c_scale = needs_scaling(label)
            m = evaluate_holdout(X_tv, y_tv, X_test, y_test, subset, c_clf, c_scale)
            print(f"\n  {label.upper()} ({method})  ({len(subset)} features)")
            _metrics(m)

    # ── 5.  LOOCV on full data (mRMR subset) ─────────────────────────
    _hr("LOOCV — mRMR subset × each classifier (full dataset)")
    for c_name, c_clf, c_scale in classifiers:
        t0 = time.time()
        m = evaluate_loocv(X, y, mrmr_subset, c_clf, c_scale)
        print(f"\n  {c_name.upper()}  ({len(X)} samples, {len(mrmr_subset)} features)")
        _metrics(m)
        print(f"    [{time.time() - t0:.1f}s]")

    # ── 6.  Plot ─────────────────────────────────────────────────────
    plot_wrapper_results(all_wrapper, mrmr_subset)

    # ── 7.  Persist plot data ────────────────────────────────────────
    md = [f"# Plot data — `{STEM}.py`\n"]
    md.append("## Wrapper feature-selection results (Balanced Accuracy, RSKF)\n")
    md.append(persist.md_table(
        ["Classifier", "Method", "Balanced Accuracy", "Selected features"],
        [
            [name.upper(), method, f"{res[1]:.4f}", ", ".join(res[0])]
            for name, resdict in all_wrapper.items()
            for method, res in resdict.items()
        ],
    ))
    md.append("\n## mRMR selected subset\n")
    md.append(", ".join(mrmr_subset) + "\n")
    md.append("\n## MI with target (mRMR order)\n")
    md.append(persist.md_table(
        ["Feature", "MI with target"],
        [[f, f"{mrmr_info['mi_target'][f]:.4f}"] for f in mrmr_info["order"]],
    ))
    persist.save_plot_data(STEM, "".join(md))

    persist.end()
