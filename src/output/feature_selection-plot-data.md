# Plot data — `feature_selection.py`
## Wrapper feature-selection results (Balanced Accuracy, RSKF)
| Classifier | Method | Balanced Accuracy | Selected features |
| --- | --- | --- | --- |
| NB | forward | 0.8715 | rocket_active |
| NB | backward | 0.8681 | rocket_mass_kg |
| NB | forward_backward | 0.8715 | rocket_active |
| SVM | forward | 0.8715 | rocket_active |
| SVM | backward | 0.8715 | launchpad_name_Kwajalein Atoll |
| SVM | forward_backward | 0.8715 | rocket_active |
| LR | forward | 0.8715 | rocket_active |
| LR | backward | 0.8715 | launchpad_name_Kwajalein Atoll |
| LR | forward_backward | 0.8715 | rocket_active |

## mRMR selected subset
crew_count, rocket_name_Falcon Heavy, rocket_cost_per_launch, launchpad_name_KSC LC 39A, cores_reused

## MI with target (mRMR order)
| Feature | MI with target |
| --- | --- |
| crew_count | 0.1430 |
| rocket_name_Falcon Heavy | 0.1413 |
| rocket_cost_per_launch | 0.0457 |
| launchpad_name_KSC LC 39A | 0.0408 |
| cores_reused | 0.0234 |
