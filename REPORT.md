# Project Report: SpaceX Launch Outcome Prediction

A scientific-presentation reference document.
Suggested slide layout: **Motivation → Data → Method → Results & Discussion → Outlook → Summary**.

---

## 1. Motivation

> "We chose this dataset because it presents an interesting real world classification
> problem with a relatively small dataset of only 205 samples. The project allows us to
> explore whether it is possible to build a reliable and accurate machine learning model
> despite limited training data."

SpaceX is one of the most active launch providers in the world and publishes a rich,
public record of every mission. Predicting whether a given launch will **succeed or
fail** from pre-/per-launch attributes (rocket type, launch pad, reuse count, crew size,
payload count, etc.) is a genuine binary classification problem with real-world value
for risk assessment, insurance, and mission planning.

The challenge that makes this problem scientifically interesting is **data scarcity
combined with extreme class imbalance**: failures are rare. This project investigates
whether standard classifiers — Naive Bayes, Support Vector Machines, and Random Forests —
can still produce a *reliable* model under those constraints, and how much feature
selection and hyperparameter tuning help.

---

## 2. Data

### 2.1 Sources

Raw data lives in `data/raw/` and was collected from the public SpaceX / r/SpaceX
REST API (the same source used by the well-known IBM "Data Science" Capstone project):

| File | Rows | Content |
|---|---|---|
| `spacex_launches.csv` | 205 | One row per launch (flight number, name, date, rocket, launchpad, success flag, failure count, crew/payload counts, reuse, landing info, free-text details) |
| `spacex_rockets.csv` | 4 | Rocket metadata (Falcon 1, Falcon 9, Falcon Heavy, Starship) — cost per launch, success rate, dimensions, mass, active flag |
| `spacex_starlink.csv` | 3526 | Starlink satellite orbital parameters (explored but **not used** for the classification target) |

The target variable is `success` (1 = launch reached orbit, 0 = failure).

### 2.2 Preprocessing (`src/data_preprocessing (1).py`)

The preprocessing pipeline:

1. **Merge** launch data with rocket metadata on `rocket_id` to enrich each launch with
   rocket-level features (cost, success rate, height, diameter, mass, active flag).
2. **Drop corrupted rows**: rows where `success == 0` **and** `failures == 0` are
   removed. These are missions whose outcome was never properly recorded — a mixture of
   *scheduled/upcoming launches* still carrying placeholder zeros (flight numbers 188+
   with duplicated numbers and all-zero fields) and one real launch (`Starlink 3-1`,
   July 2022) with missing outcome data. **19 rows were dropped.**
3. **Drop non-predictive / leaky columns**: `flight_number`, `date_utc`, `rocket_id`,
   `failures` (perfect inverse of target), `launchpad_id`, `details` (free text),
   `landing_type`, `name`, `landing_success` (a near-perfect post-hoc correlate of
   success — would leak the label).
4. **Output**: `data/processed/processed_data.csv` — **186 rows, 12 columns**:
   `rocket_name, launchpad_name, success, crew_count, payloads_count, cores_reused,
   rocket_active, rocket_cost_per_launch, rocket_success_rate, rocket_height_m,
   rocket_diameter_m, rocket_mass_kg`.

> **Result of the drop**: the class distribution went from 181 vs 24 (88.3% / 11.7%)
> to **181 vs 5 (97.3% / 2.7%)** — dropping the corrupted rows *worsened* the imbalance
> but removed label noise. This trade-off is a central theme of the project.

### 2.3 The corrupted-data decision (presentation highlight)

The 19 corrupted rows presented three options:

| Option | Consequence |
|---|---|
| Keep as failures (`success=0`) | Adds 19 fake minority samples → less imbalance (181:24) but **label noise** (upcoming launches aren't real failures) |
| Impute the real outcome | Not feasible — most rows are future launches with no known outcome |
| **Drop them** (chosen) | Clean labels but **extreme imbalance** (181:5, 2.69% minority) |

The team chose to drop them, prioritising label correctness over class balance, and
compensated for the resulting imbalance algorithmically (`class_weight='balanced'`,
balanced-accuracy scoring, stratified CV).

### 2.4 EDA findings (`notebooks/notebook1.ipynb`)

- **Class imbalance**: 181 successes vs 5 failures — only 2.69% minority.
- **Rocket distribution**: Falcon 9 dominates; Falcon 1 has the lowest success rate
  (early failures); Falcon Heavy has a perfect record but few flights.
- **Launchpad distribution**: CCSFS SLC 40 and KSC LC 39A carry most launches; VAFB
  SLC 4E and Kwajalein Atoll are smaller.
- **Correlation matrix** (raw scales + normalised): the strongest correlates of
  `success` were `landing_success` (dropped later as a leak), `cores_reused`, and
  `failures` (the perfect inverse). Most numeric features are **not strongly linearly
  related** to the target.
- **Scatter/histogram matrix**: payloads_count is heavily right-skewed (one outlier at
  16, the STP-2 rideshare); crew missions are very rare (2–4 crew); reused cores appear
  only alongside successes. Few unique feature combinations → lots of overlapping
  points → cross-validation is necessary and complex feature engineering is unlikely to
  help much.

---

## 3. Method

### 3.1 Feature engineering

Categorical features (`rocket_name`, `launchpad_name`) are one-hot encoded with
`drop_first=True` → **14 features** after encoding:
`crew_count, payloads_count, cores_reused, rocket_active, rocket_cost_per_launch,
rocket_success_rate, rocket_height_m, rocket_diameter_m, rocket_mass_kg,
rocket_name_Falcon 9, rocket_name_Falcon Heavy, launchpad_name_KSC LC 39A,
launchpad_name_Kwajalein Atoll, launchpad_name_VAFB SLC 4E`.

### 3.2 Feature selection (`src/feature_selection.py`)

A dedicated script compares filter and wrapper methods. The primary method is
**mRMR (minimum-Redundancy Maximum-Relevance)** — a model-agnostic filter using mutual
information (`mutual_info_classif` for relevance, `mutual_info_regression` for
redundancy), with features standardised once beforehand because sklearn's MI estimators
rely on k-NN distances.

Wrapper methods (forward selection, backward elimination, forward+backward) are run
**per classifier** (GaussianNB, SVM-RBF, Logistic Regression) using
`RepeatedStratifiedKFold` (splits auto-bounded by minority-class count) and balanced
accuracy as the wrapper objective. A held-out 20% test set and LOOCV on the full set
give unbiased final estimates.

**mRMR top-5 subset** (by MI with target):

| Feature | MI with target |
|---|---|
| crew_count | 0.1430 |
| rocket_name_Falcon Heavy | 0.1413 |
| rocket_cost_per_launch | 0.0457 |
| launchpad_name_KSC LC 39A | 0.0408 |
| cores_reused | 0.0234 |

The wrappers all converged on a **single feature — `rocket_active`** — with balanced
accuracy 0.8715 (RSKF). This is essentially the model learning "active rockets succeed,
inactive ones (Falcon 1) failed": a strong but almost trivial signal.

### 3.3 Validation strategy

Two complementary schemes are used throughout, motivated by the 5-sample minority class:

- **LOOCV (Leave-One-Out)**: every one of the 5 failures is tested in isolation under
  its own model — no fold can be degenerate. Used as the **primary** performance
  estimate.
- **Repeated Stratified 5-Fold × 20** (100 evaluations): stratification guarantees each
  fold has ≥1 failure; repeats give a mean ± std distribution.

### 3.4 Class-imbalance handling

- `class_weight='balanced'` on SVM and Random Forest (inversely weights the rare class;
  ≈1:36 weight ratio for 181:5).
- `StandardScaler` for SVM (RBF kernel is distance-based; unscaled features like
  payloads_count=16 would dominate).
- **Balanced accuracy** as the headline metric (arithmetic mean of per-class recall) —
  recommended for imbalanced data and far more informative than raw accuracy, which is
  inflated by the 97% majority.

### 3.5 Models

| Script | Model | Notes |
|---|---|---|
| `naive_bayes.py` | GaussianNB | Probabilistic baseline; no scaling needed |
| `svm.py` | SVC, RBF, `class_weight=balanced` | Distance-based, scaled |
| `svm_tuning.py` | GridSearchCV over C, gamma, kernel | 95 configs × 50 folds = 4750 fits |
| `random_forest.py` | RandomForest, 100 trees, `class_weight=balanced` | Ensemble baseline |
| `random_forest_tuning.py` | GridSearchCV over 6 RF params | 288 configs × 10 folds = 2880 fits |

A `persist.py` helper tees stdout and patches `plt.show` so every run's prints and
plots are captured to `src/output/*-prints.md` and `*-plot-data.md` (plus PNGs),
making the project fully reproducible and auditable without re-running anything.

---

## 4. Results & Discussion

### 4.1 Baseline models (LOOCV, primary metric = Balanced Accuracy)

| Model | Accuracy | Precision | Recall | F1 | **Balanced Acc.** | Minority correct |
|---|---|---|---|---|---|---|
| GaussianNB | 0.9624 | 0.9888 | 0.9724 | 0.9805 | **0.7862** | 3 / 5 |
| SVM (RBF, C=1, γ=scale) | 0.7903 | 1.0000 | 0.7845 | 0.8793 | **0.8923** | 5 / 5 |
| Random Forest (100 trees) | 0.8011 | 0.9865 | 0.8066 | 0.8875 | **0.7033** | 3 / 5 |

**Key observations**

- **Naive Bayes has the highest raw accuracy (0.9624) but a poor balanced accuracy
  (0.7862)** — it misclassifies 2 of the 5 failures (samples 23 and 33) as successes.
  Its high accuracy is almost entirely driven by getting the 181 majority samples right.
- **SVM is the best failure detector**: it correctly identifies **all 5 failures**
  (recall on the minority class = 1.0) at the cost of flagging ~39 successes as failures
  (majority recall 0.78). This yields the **highest balanced accuracy (0.8923)** and a
  perfect precision of 1.0 (every predicted failure is a real failure).
- **Random Forest** also misses 2 of 5 failures and has the lowest balanced accuracy
  (0.7033) — the ensemble's majority voting tends to drown out the rare class even with
  `class_weight='balanced'`.
- The RSKF standard deviations on balanced accuracy are large (NB: 0.244, RF: 0.199,
  SVM: 0.183), reflecting the instability inherent to a 5-sample minority class.

### 4.2 SVM hyperparameter tuning (`svm_tuning.py`)

Grid over `kernel ∈ {rbf, poly, sigmoid, linear}`, `C ∈ {0.01…100}`,
`gamma ∈ {scale, auto, 0.001…1}`; CV = RSKF 5×10, scored on balanced accuracy.

- **Best CV balanced accuracy: 0.9033** — the entire top-10 consists of **RBF** kernels
  with `C ∈ {1,10,100}` and various `gamma`, all tying at 0.9033 ± 0.031. Linear/poly/
  sigmoid did not reach the top tier.
- **Best params chosen by the grid**: `{C: 1, gamma: 1, kernel: rbf}`.

LOOCV baseline vs tuned:

| Metric | Baseline (γ=scale) | Tuned (γ=1) | Δ |
|---|---|---|---|
| Accuracy | 0.7903 | 0.8118 | +0.0215 |
| Precision | 1.0000 | 1.0000 | +0.0000 |
| Recall | 0.7845 | 0.8066 | +0.0221 |
| F1 | 0.8793 | 0.8930 | +0.0137 |
| **Balanced Acc.** | 0.8923 | **0.9033** | +0.0110 |

Tuning gives a **small but consistent improvement**: +0.011 balanced accuracy and two
more correctly classified majority samples, while still detecting all 5 failures.

### 4.3 Random Forest tuning (`random_forest_tuning.py`)

Grid over `n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
criterion` (288 configs × 10 folds).

- **Best CV balanced accuracy: 0.7462** — many configurations tie at this value (std
  0.207), and `min_samples_leaf=2` recurs throughout the top-10.
- **Best params**: `{gini, max_depth=5, max_features=sqrt, min_samples_leaf=1,
  min_samples_split=2, n_estimators=100}`.
- **LOOCV: tuned = baseline exactly (Δ = 0.0000 on every metric)** — tuning did not
  improve the Random Forest's out-of-sample performance. The model still misclassifies
  failures 23 and 33 and retains balanced accuracy 0.7033. This suggests the RF is
  fundamentally limited on this data, not parameter-starved.

### 4.4 Feature-selection discussion

- The mRMR subset (`crew_count, rocket_name_Falcon Heavy, rocket_cost_per_launch,
  launchpad_name_KSC LC 39A, cores_reused`) gave a **higher hold-out accuracy but a
  much worse balanced accuracy (~0.47–0.49)** than the all-feature models — on the
  held-out test set with only 1 minority sample, the mRMR subset mostly predicted
  "success" and got the single failure wrong. LOOCV with the mRMR subset was stronger
  for NB (0.786) but worse for SVM/LR (0.667).
- The wrapper methods all collapsed to the single feature `rocket_active` with RSKF
  balanced accuracy 0.8715 — a degenerate but very stable rule ("if the rocket is still
  active, predict success"). This is the cleanest illustration that, with 5 failures,
  the only robust signal the wrappers can exploit is the Falcon 1 vs Falcon 9/Heavy
  distinction.

### 4.5 Which failures are hard?

Across all baselines, **failures at sample indices 23 and 33** (CRS-7, June 2015 and
Amos-6, September 2016 — both Falcon 9 launches from CCSFS SLC 40) are consistently
misclassified. They look identical to the 181 successes on every available feature, so
no model trained on these features can reliably tell them apart. The other 3 failures
(Falcon 1 missions, indices 0–2) are trivially separable because they used the inactive
Falcon 1 rocket.

### 4.6 Feature-selection hold-out caveat

The hold-out test split contained only **1 minority sample out of 38** (stratified
20%), so the hold-out balanced-accuracy numbers (~0.47–0.49) are dominated by that
single sample and should be treated as illustrative rather than precise. LOOCV, which
tests every failure, is the more trustworthy estimate.

---

## 5. Outlook

1. **More / better features.** The two indistinguishable failures (CRS-7, Amos-6) show
   that the current feature set lacks the discriminative power to catch Falcon 9
   failures. Candidate additions: weather/upper-wind data at launch, vehicle
   serial/block number, payload mass, orbit class, static-fire outcome, fairing-reuse
   flag, days since last launch.
2. **Address the imbalance directly.** SMOTE / ADASYN oversampling of the minority
   class, or anomaly-detection framing (one-class SVM, Isolation Forest) treating
   failures as anomalies, could be more appropriate than `class_weight` alone.
3. **Collect more failure data.** The fundamental bottleneck is n=5. Augmenting with
   failures from other providers or with SpaceX's later 2022–2024 record (the corrupted
   rows, once their true outcomes are known) would materially help.
4. **Cost-sensitive learning.** Optimise for a custom cost matrix where a missed
   failure is far more expensive than a false alarm — more aligned with the real-world
   risk-assessment use case than balanced accuracy.
5. **Probabilistic calibration.** For SVM and RF, calibrate predicted probabilities
   (Platt/ isotonic) so the model's confidence is usable downstream.
6. **Interpretability.** SHAP/permutation importance on the tuned SVM and RF would
   clarify whether the models rely on anything beyond `rocket_active`.

---

## 6. Summary

- **Goal**: binary classification of SpaceX launch success from 186 labelled launches
  (after dropping 19 corrupted rows from an initial 205).
- **Hardship**: extreme class imbalance — 181 successes vs 5 failures (2.69% minority).
- **Approach**: one-hot encoding (14 features); mRMR filter + per-classifier wrapper
  selection; GaussianNB, SVM-RBF, and Random Forest baselines; grid-search tuning for
  SVM and RF; LOOCV (primary) + Repeated Stratified K-Fold for validation;
  `class_weight='balanced'` and balanced accuracy to handle imbalance.
- **Best model**: **tuned SVM (RBF, C=1, γ=1)** — LOOCV balanced accuracy **0.9033**,
  detecting **all 5 failures** with perfect precision. Tuning improved the baseline SVM
  by +0.011 balanced accuracy.
- **Random Forest** could not be improved by tuning (balanced accuracy stuck at 0.7033,
  3/5 failures detected). **Naive Bayes** had the highest raw accuracy but the worst
  failure recall among the three.
- **Lesson**: with only 5 minority samples, raw accuracy is misleading; balanced
  accuracy reveals that the problem is essentially "detect the rare Falcon 1 failures"
  plus two genuinely indistinguishable Falcon 9 failures. A reliable model *is
  achievable* - but its reliability is bounded by the information content of the
  features, not by the choice of algorithm.
