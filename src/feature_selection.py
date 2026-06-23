from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "processed_data.csv"
TARGET = "success"
MAX_FEATURES = 5
CV = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)


def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# Quality criterion: stratified 4-fold CV
# Returns dict with accuracy and raw counts (correct/total) for overall,
# failure (class 0), and success (class 1).
# Selection functions use only overall_acc for comparisons.
def train_and_test(X, y, feature_subset):
    cols = list(feature_subset)
    clf = LogisticRegression(max_iter=1000)
    correct_all = total_all = 0
    correct_0 = total_0 = 0
    correct_1 = total_1 = 0

    for train_idx, val_idx in CV.split(X, y):
        X_tr = X.iloc[train_idx][cols]
        X_val = X.iloc[val_idx][cols]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)

        correct_all += (preds == y_val.values).sum()
        total_all += len(y_val)

        mask_0 = (y_val == 0).values
        mask_1 = (y_val == 1).values
        correct_0 += (preds[mask_0] == y_val.values[mask_0]).sum()
        total_0 += mask_0.sum()
        correct_1 += (preds[mask_1] == y_val.values[mask_1]).sum()
        total_1 += mask_1.sum()

    return {
        "overall": (correct_all / total_all, correct_all, total_all),
        "failure": (correct_0 / total_0, correct_0, total_0),
        "success": (correct_1 / total_1, correct_1, total_1),
    }


def _score(X, y, subset):
    """Classwise average accuracy (macro) — used for comparisons inside selection functions."""
    r = train_and_test(X, y, subset)
    return (r["failure"][0] + r["success"][0]) / 2


# 1. Exhaustive (brute-force) search
#    Try all 2^p - 1 non-empty subsets. Practical only for small p.
def exhaustive_search(X, y):
    features = list(X.columns)
    best_score = -1
    best_subset = None

    for size in range(1, len(features) + 1):
        for subset in combinations(features, size):
            score = _score(X, y, subset)
            if score > best_score:
                best_score = score
                best_subset = subset

    return list(best_subset), train_and_test(X, y, list(best_subset))


# 2. Statistics-based: mRMR (Minimum-Redundancy Maximum-Relevance)
#    Relevance:  D(S, y) = (1/|S|) * sum_{x_j in S} I(x_j; y)
#    Redundancy: R(S)    = (1/|S|^2) * sum_{x_j, x_k in S} I(x_j; x_k)
#    mRMR criterion: max_S [ (1/|S|) * sum I(f_i; c) - (1/|S|^2) * sum I(f_i; f_j) ]
#    Solved greedily (no classifier needed).
def mrmr_selection(X, y, n_features):
    features = list(X.columns)

    mi_target = dict(zip(features, mutual_info_classif(X, y, discrete_features=False)))

    mi_pairs = {}
    for f in features:
        mi_pairs[f] = dict(zip(features, mutual_info_regression(X, X[f])))

    selected = []
    remaining = features.copy()

    for _ in range(n_features):
        best_score = -np.inf
        best_feature = None

        for f in remaining:
            relevance = mi_target[f]
            redundancy = (
                np.mean([mi_pairs[f][s] for s in selected]) if selected else 0.0
            )
            score = relevance - redundancy

            if score > best_score:
                best_score = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected, train_and_test(X, y, selected)


# 3. Forward selection
#    Start with empty set. Greedily add the feature with highest CV accuracy.
#    Stop when no improvement or MAX_FEATURES reached.
def forward_selection(X, y, max_features=MAX_FEATURES):
    features = list(X.columns)
    selected = []
    best_score = 0.0

    while len(selected) < max_features:
        scores = {
            f: _score(X, y, selected + [f]) for f in features if f not in selected
        }

        best_addition = max(scores, key=scores.get)

        if scores[best_addition] > best_score:
            selected.append(best_addition)
            best_score = scores[best_addition]
        else:
            break

    return selected, train_and_test(X, y, selected)


# 4. Backward elimination
#    Start with the full feature set. Greedily remove the feature whose
#    removal hurts CV accuracy least. Stop when any removal degrades results.
def backward_elimination(X, y):
    selected = list(X.columns)
    best_score = _score(X, y, selected)

    while len(selected) > 1:
        scores = {f: _score(X, y, [x for x in selected if x != f]) for f in selected}

        best_removal = max(scores, key=scores.get)

        if scores[best_removal] >= best_score:
            selected.remove(best_removal)
            best_score = scores[best_removal]
        else:
            break

    return selected, train_and_test(X, y, selected)


# 5. Forward selection + Backward elimination (combined)
#    Alternate passes until the selected set stabilises.
def forward_backward_selection(X, y, max_features=MAX_FEATURES):
    selected, _ = forward_selection(X, y, max_features)

    while True:
        prev = set(selected)

        selected, _ = backward_elimination(X[selected], y)

        best_score = _score(X, y, selected)
        candidates = {
            f: _score(X, y, selected + [f])
            for f in X.columns
            if f not in selected and len(selected) < max_features
        }

        if candidates:
            best_add = max(candidates, key=candidates.get)
            if candidates[best_add] > best_score:
                selected.append(best_add)

        if set(selected) == prev:
            break

    return selected, train_and_test(X, y, selected)


# Feature ranking helpers
def rank_features(X, y, feature_subset, top_n=3):
    """Evaluate features individually, ranked by classwise avg.
    If the selected subset has fewer than top_n features, the remaining
    slots are filled with the best features from the full feature pool."""
    pool = list(feature_subset)
    if len(pool) < top_n:
        pool += [f for f in X.columns if f not in pool]

    rows = []
    for f in pool:
        s = train_and_test(X, y, [f])
        cw = (s["failure"][0] + s["success"][0]) / 2
        rows.append(
            {
                "feature": f,
                "scores": s,
                "classwise_avg": cw,
                "selected": f in feature_subset,
            }
        )
    rows.sort(key=lambda r: r["classwise_avg"], reverse=True)
    return rows[:top_n]


def print_ranking(ranking):
    for i, r in enumerate(ranking, 1):
        s = r["scores"]
        ov_acc, ov_c, ov_t = s["overall"]
        f_acc, f_c, f_t = s["failure"]
        s_acc, s_c, s_t = s["success"]
        cw = r["classwise_avg"]
        tag = "" if r["selected"] else "  (not in selected subset)"
        print(f"  #{i} {r['feature']}{tag}")
        print(f"     overall acc   : {ov_acc:.4f}  ({ov_c}/{ov_t})")
        print(f"     failure acc   : {f_acc:.4f}  ({f_c}/{f_t})  (class 0)")
        print(f"     success acc   : {s_acc:.4f}  ({s_c}/{s_t})  (class 1)")
        print(f"     classwise avg : {cw:.4f}")


def plot_rankings(all_rankings, method_names):
    """Bar chart: one subplot per method, top-3 features × 3 metrics."""
    metrics = ["overall", "failure", "success"]
    labels = ["Overall", "Failure (class 0)", "Success (class 1)"]
    colors = ["steelblue", "tomato", "mediumseagreen"]
    n_methods = len(method_names)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    bar_width = 0.22
    for ax, name, ranking in zip(axes, method_names, all_rankings):
        features = [r["feature"] for r in ranking]
        x = np.arange(len(features))
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            vals = [r["scores"][metric][0] for r in ranking]
            ax.bar(x + i * bar_width, vals, bar_width, label=label, color=color)
        ax.set_title(name)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(features, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Top-3 feature ranking per method (individual CV accuracy)", fontsize=12
    )
    plt.tight_layout()
    plt.show()


# Main
def print_result(subset, scores):
    ov_acc, ov_c, ov_t = scores["overall"]
    f_acc, f_c, f_t = scores["failure"]
    s_acc, s_c, s_t = scores["success"]
    cw_avg = (f_acc + s_acc) / 2
    print(f"  Selected          : {subset}")
    print(f"  CV accuracy       : {ov_acc:.4f}  ({ov_c}/{ov_t})")
    print(f"  CV acc failure    : {f_acc:.4f}  ({f_c}/{f_t})  (class 0)")
    print(f"  CV acc success    : {s_acc:.4f}  ({s_c}/{s_t})  (class 1)")
    print(f"  CV classwise avg  : {cw_avg:.4f}\n")


if __name__ == "__main__":
    import time

    X, y = load_data()
    print(f"Features: {list(X.columns)}")
    print(f"Samples:  {len(X)}  (class distribution: {dict(y.value_counts())})\n")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # print("=== 1. Exhaustive search ===")
    # t0 = time.time()
    # ex_subset, ex_scores = exhaustive_search(X_tv, y_tv)
    # print(f"  Time: {time.time() - t0:.2f}s")
    # print_result(ex_subset, ex_scores)

    print("=== 2. mRMR (statistics-based) ===")
    t0 = time.time()
    mrmr_subset, mrmr_scores = mrmr_selection(X_tv, y_tv, n_features=5)
    print(f"  Time: {time.time() - t0:.2f}s")
    print_result(mrmr_subset, mrmr_scores)

    print("=== 3. Forward selection ===")
    t0 = time.time()
    fwd_subset, fwd_scores = forward_selection(X_tv, y_tv)
    print(f"  Time: {time.time() - t0:.2f}s")
    print_result(fwd_subset, fwd_scores)

    print("=== 4. Backward elimination ===")
    t0 = time.time()
    bwd_subset, bwd_scores = backward_elimination(X_tv, y_tv)
    print(f"  Time: {time.time() - t0:.2f}s")
    print_result(bwd_subset, bwd_scores)

    print("=== 5. Forward + Backward selection ===")
    t0 = time.time()
    fb_subset, fb_scores = forward_backward_selection(X_tv, y_tv)
    print(f"  Time: {time.time() - t0:.2f}s")
    print_result(fb_subset, fb_scores)

    results = {
        #'exhaustive':       (ex_subset,   ex_scores),
        "mRMR": (mrmr_subset, mrmr_scores),
        "forward": (fwd_subset, fwd_scores),
        "backward": (bwd_subset, bwd_scores),
        "forward+backward": (fb_subset, fb_scores),
    }
    best_method = max(
        results,
        key=lambda k: (results[k][1]["failure"][0] + results[k][1]["success"][0]) / 2,
    )
    best_subset, best_scores = results[best_method]

    print(f"=== Best method on CV: {best_method} ===")
    print_result(best_subset, best_scores)

    # Top-3 feature ranking per method
    print("=== Top-3 feature ranking per method ===\n")
    all_rankings = []
    for name, (subset, _) in results.items():
        top3 = rank_features(X_tv, y_tv, subset, top_n=3)
        all_rankings.append(top3)
        print(f"-- {name} --")
        print_ranking(top3)
        print()

    plot_rankings(all_rankings, list(results.keys()))
