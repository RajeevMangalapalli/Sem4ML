# Prints from `feature_selection.py`

```
Features       : 14  (crew_count, payloads_count, cores_reused, rocket_active, rocket_cost_per_launch, rocket_success_rate, rocket_height_m, rocket_diameter_m, rocket_mass_kg, rocket_name_Falcon 9, rocket_name_Falcon Heavy, launchpad_name_KSC LC 39A, launchpad_name_Kwajalein Atoll, launchpad_name_VAFB SLC 4E)
Samples        : 186
Classes        : {1: np.int64(181), 0: np.int64(5)}
Imbalance      : 97.3% / 2.7%

Train/val      : 148  ({1: np.int64(144), 0: np.int64(4)})
Test           : 38  ({1: np.int64(37), 0: np.int64(1)})
Inner CV       : RepeatedStratifiedKFold(n_repeats=5, n_splits=np.int64(4), random_state=42)

============================================================
  mRMR — filter (model‑agnostic, MI‑based)
============================================================
  Selected     : ['crew_count', 'rocket_name_Falcon Heavy', 'rocket_cost_per_launch', 'launchpad_name_KSC LC 39A', 'cores_reused']
  MI with target:
    crew_count                           MI = 0.1430
    rocket_name_Falcon Heavy             MI = 0.1413
    rocket_cost_per_launch               MI = 0.0457
    launchpad_name_KSC LC 39A            MI = 0.0408
    cores_reused                         MI = 0.0234
  Time         : 0.4s

============================================================
  Wrapper Methods  —  classifier: NB
============================================================
  Forward            : ['rocket_active']
    BA (RSKF) = 0.8715  [4.4s]
  Backward           : ['rocket_mass_kg']
    BA (RSKF) = 0.8681  [5.8s]
  Forward+Backward   : ['rocket_active']
    BA (RSKF) = 0.8715  [2.4s]

============================================================
  Wrapper Methods  —  classifier: SVM
============================================================
  Forward            : ['rocket_active']
    BA (RSKF) = 0.8715  [2.1s]
  Backward           : ['launchpad_name_Kwajalein Atoll']
    BA (RSKF) = 0.8715  [8.5s]
  Forward+Backward   : ['rocket_active']
    BA (RSKF) = 0.8715  [3.0s]

============================================================
  Wrapper Methods  —  classifier: LR
============================================================
  Forward            : ['rocket_active']
    BA (RSKF) = 0.8715  [2.2s]
  Backward           : ['launchpad_name_Kwajalein Atoll']
    BA (RSKF) = 0.8715  [9.4s]
  Forward+Backward   : ['rocket_active']
    BA (RSKF) = 0.8715  [3.3s]

============================================================
  Best subset per classifier
============================================================
   NB  forward               BA=0.8715  →  ['rocket_active']
  SVM  forward               BA=0.8715  →  ['rocket_active']
   LR  forward               BA=0.8715  →  ['rocket_active']

============================================================
  Hold‑out test evaluation
============================================================

  mRMR subset × NB  (5 features)
    balanced_accuracy   0.4730
    accuracy            0.9211
    precision           0.9722
    recall              0.9459
    f1                  0.9589

  mRMR subset × SVM  (5 features)
    balanced_accuracy   0.4865
    accuracy            0.9474
    precision           0.9730
    recall              0.9730
    f1                  0.9730

  mRMR subset × LR  (5 features)
    balanced_accuracy   0.4865
    accuracy            0.9474
    precision           0.9730
    recall              0.9730
    f1                  0.9730

  NB (forward)  (1 features)
    balanced_accuracy   0.4865
    accuracy            0.9474
    precision           0.9730
    recall              0.9730
    f1                  0.9730

  SVM (forward)  (1 features)
    balanced_accuracy   0.4865
    accuracy            0.9474
    precision           0.9730
    recall              0.9730
    f1                  0.9730

  LR (forward)  (1 features)
    balanced_accuracy   0.4865
    accuracy            0.9474
    precision           0.9730
    recall              0.9730
    f1                  0.9730

============================================================
  LOOCV — mRMR subset × each classifier (full dataset)
============================================================

  NB  (186 samples, 5 features)
    balanced_accuracy   0.7862
    accuracy            0.9624
    precision           0.9888
    recall              0.9724
    f1                  0.9805
    [0.6s]

  SVM  (186 samples, 5 features)
    balanced_accuracy   0.6674
    accuracy            0.7312
    precision           0.9852
    recall              0.7348
    f1                  0.8418
    [1.3s]

  LR  (186 samples, 5 features)
    balanced_accuracy   0.6674
    accuracy            0.7312
    precision           0.9852
    recall              0.7348
    f1                  0.8418
    [3.0s]

```
