# Plot data — `svm_tuning.py`
## Best params

`{'C': 1, 'gamma': 1, 'kernel': 'rbf'}`
Best BA (CV): **0.9033**

## Top-10 configurations (by mean Balanced Accuracy)
| Rank | Params | Mean BA | Std |
| --- | --- | --- | --- |
| 1 | {'C': 10, 'gamma': 1, 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 2 | {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 3 | {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 4 | {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 5 | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 6 | {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 7 | {'C': 100, 'gamma': 1, 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 8 | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 9 | {'C': 1, 'gamma': 1, 'kernel': 'rbf'} | 0.9033 | 0.0312 |
| 10 | {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'} | 0.9033 | 0.0312 |

## LOOCV — Baseline vs Best params
Baseline params: `{'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}`
Tuned params: `{'C': 1, 'gamma': 1, 'kernel': 'rbf'}`
| Metric | Baseline | Tuned | Δ |
| --- | --- | --- | --- |
| Accuracy | 0.7903 | 0.8118 | +0.0215 |
| Precision | 1.0000 | 1.0000 | +0.0000 |
| Recall | 0.7845 | 0.8066 | +0.0221 |
| F1 Score | 0.8793 | 0.8930 | +0.0137 |
| Balanced Accuracy | 0.8923 | 0.9033 | +0.0110 |

## LOOCV per-sample correctness (Tuned model, 1=correct, 0=wrong)
| Sample index | True label | Predicted | Correct | Class |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1 | minority |
| 1 | 0 | 0 | 1 | minority |
| 2 | 0 | 0 | 1 | minority |
| 3 | 1 | 0 | 0 | majority |
| 4 | 1 | 0 | 0 | majority |
| 5 | 1 | 0 | 0 | majority |
| 6 | 1 | 1 | 1 | majority |
| 7 | 1 | 0 | 0 | majority |
| 8 | 1 | 1 | 1 | majority |
| 9 | 1 | 0 | 0 | majority |
| 10 | 1 | 1 | 1 | majority |
| 11 | 1 | 0 | 0 | majority |
| 12 | 1 | 0 | 0 | majority |
| 13 | 1 | 0 | 0 | majority |
| 14 | 1 | 0 | 0 | majority |
| 15 | 1 | 0 | 0 | majority |
| 16 | 1 | 0 | 0 | majority |
| 17 | 1 | 0 | 0 | majority |
| 18 | 1 | 0 | 0 | majority |
| 19 | 1 | 0 | 0 | majority |
| 20 | 1 | 1 | 1 | majority |
| 21 | 1 | 0 | 0 | majority |
| 22 | 1 | 0 | 0 | majority |
| 23 | 0 | 0 | 1 | minority |
| 24 | 1 | 0 | 0 | majority |
| 25 | 1 | 1 | 1 | majority |
| 26 | 1 | 0 | 0 | majority |
| 27 | 1 | 0 | 0 | majority |
| 28 | 1 | 0 | 0 | majority |
| 29 | 1 | 0 | 0 | majority |
| 30 | 1 | 1 | 1 | majority |
| 31 | 1 | 0 | 0 | majority |
| 32 | 1 | 0 | 0 | majority |
| 33 | 0 | 0 | 1 | minority |
| 34 | 1 | 1 | 1 | majority |
| 35 | 1 | 1 | 1 | majority |
| 36 | 1 | 1 | 1 | majority |
| 37 | 1 | 1 | 1 | majority |
| 38 | 1 | 1 | 1 | majority |
| 39 | 1 | 1 | 1 | majority |
| 40 | 1 | 1 | 1 | majority |
| 41 | 1 | 1 | 1 | majority |
| 42 | 1 | 1 | 1 | majority |
| 43 | 1 | 1 | 1 | majority |
| 44 | 1 | 1 | 1 | majority |
| 45 | 1 | 1 | 1 | majority |
| 46 | 1 | 1 | 1 | majority |
| 47 | 1 | 1 | 1 | majority |
| 48 | 1 | 1 | 1 | majority |
| 49 | 1 | 1 | 1 | majority |
| 50 | 1 | 1 | 1 | majority |
| 51 | 1 | 1 | 1 | majority |
| 52 | 1 | 0 | 0 | majority |
| 53 | 1 | 1 | 1 | majority |
| 54 | 1 | 1 | 1 | majority |
| 55 | 1 | 1 | 1 | majority |
| 56 | 1 | 0 | 0 | majority |
| 57 | 1 | 1 | 1 | majority |
| 58 | 1 | 1 | 1 | majority |
| 59 | 1 | 0 | 0 | majority |
| 60 | 1 | 1 | 1 | majority |
| 61 | 1 | 1 | 1 | majority |
| 62 | 1 | 1 | 1 | majority |
| 63 | 1 | 1 | 1 | majority |
| 64 | 1 | 0 | 0 | majority |
| 65 | 1 | 1 | 1 | majority |
| 66 | 1 | 1 | 1 | majority |
| 67 | 1 | 0 | 0 | majority |
| 68 | 1 | 1 | 1 | majority |
| 69 | 1 | 1 | 1 | majority |
| 70 | 1 | 1 | 1 | majority |
| 71 | 1 | 0 | 0 | majority |
| 72 | 1 | 0 | 0 | majority |
| 73 | 1 | 1 | 1 | majority |
| 74 | 1 | 1 | 1 | majority |
| 75 | 1 | 1 | 1 | majority |
| 76 | 1 | 1 | 1 | majority |
| 77 | 1 | 0 | 0 | majority |
| 78 | 1 | 1 | 1 | majority |
| 79 | 1 | 1 | 1 | majority |
| 80 | 1 | 1 | 1 | majority |
| 81 | 1 | 1 | 1 | majority |
| 82 | 1 | 1 | 1 | majority |
| 83 | 1 | 1 | 1 | majority |
| 84 | 1 | 0 | 0 | majority |
| 85 | 1 | 1 | 1 | majority |
| 86 | 1 | 1 | 1 | majority |
| 87 | 1 | 1 | 1 | majority |
| 88 | 1 | 1 | 1 | majority |
| 89 | 1 | 1 | 1 | majority |
| 90 | 1 | 1 | 1 | majority |
| 91 | 1 | 1 | 1 | majority |
| 92 | 1 | 1 | 1 | majority |
| 93 | 1 | 1 | 1 | majority |
| 94 | 1 | 1 | 1 | majority |
| 95 | 1 | 1 | 1 | majority |
| 96 | 1 | 0 | 0 | majority |
| 97 | 1 | 1 | 1 | majority |
| 98 | 1 | 1 | 1 | majority |
| 99 | 1 | 1 | 1 | majority |
| 100 | 1 | 1 | 1 | majority |
| 101 | 1 | 1 | 1 | majority |
| 102 | 1 | 1 | 1 | majority |
| 103 | 1 | 1 | 1 | majority |
| 104 | 1 | 1 | 1 | majority |
| 105 | 1 | 0 | 0 | majority |
| 106 | 1 | 1 | 1 | majority |
| 107 | 1 | 1 | 1 | majority |
| 108 | 1 | 1 | 1 | majority |
| 109 | 1 | 1 | 1 | majority |
| 110 | 1 | 1 | 1 | majority |
| 111 | 1 | 1 | 1 | majority |
| 112 | 1 | 1 | 1 | majority |
| 113 | 1 | 1 | 1 | majority |
| 114 | 1 | 1 | 1 | majority |
| 115 | 1 | 1 | 1 | majority |
| 116 | 1 | 1 | 1 | majority |
| 117 | 1 | 1 | 1 | majority |
| 118 | 1 | 1 | 1 | majority |
| 119 | 1 | 1 | 1 | majority |
| 120 | 1 | 1 | 1 | majority |
| 121 | 1 | 1 | 1 | majority |
| 122 | 1 | 1 | 1 | majority |
| 123 | 1 | 1 | 1 | majority |
| 124 | 1 | 1 | 1 | majority |
| 125 | 1 | 1 | 1 | majority |
| 126 | 1 | 1 | 1 | majority |
| 127 | 1 | 1 | 1 | majority |
| 128 | 1 | 1 | 1 | majority |
| 129 | 1 | 1 | 1 | majority |
| 130 | 1 | 1 | 1 | majority |
| 131 | 1 | 1 | 1 | majority |
| 132 | 1 | 1 | 1 | majority |
| 133 | 1 | 1 | 1 | majority |
| 134 | 1 | 1 | 1 | majority |
| 135 | 1 | 1 | 1 | majority |
| 136 | 1 | 1 | 1 | majority |
| 137 | 1 | 1 | 1 | majority |
| 138 | 1 | 1 | 1 | majority |
| 139 | 1 | 1 | 1 | majority |
| 140 | 1 | 1 | 1 | majority |
| 141 | 1 | 1 | 1 | majority |
| 142 | 1 | 1 | 1 | majority |
| 143 | 1 | 1 | 1 | majority |
| 144 | 1 | 1 | 1 | majority |
| 145 | 1 | 1 | 1 | majority |
| 146 | 1 | 1 | 1 | majority |
| 147 | 1 | 1 | 1 | majority |
| 148 | 1 | 1 | 1 | majority |
| 149 | 1 | 1 | 1 | majority |
| 150 | 1 | 1 | 1 | majority |
| 151 | 1 | 1 | 1 | majority |
| 152 | 1 | 1 | 1 | majority |
| 153 | 1 | 1 | 1 | majority |
| 154 | 1 | 1 | 1 | majority |
| 155 | 1 | 1 | 1 | majority |
| 156 | 1 | 1 | 1 | majority |
| 157 | 1 | 1 | 1 | majority |
| 158 | 1 | 1 | 1 | majority |
| 159 | 1 | 1 | 1 | majority |
| 160 | 1 | 1 | 1 | majority |
| 161 | 1 | 1 | 1 | majority |
| 162 | 1 | 0 | 0 | majority |
| 163 | 1 | 1 | 1 | majority |
| 164 | 1 | 1 | 1 | majority |
| 165 | 1 | 1 | 1 | majority |
| 166 | 1 | 1 | 1 | majority |
| 167 | 1 | 1 | 1 | majority |
| 168 | 1 | 1 | 1 | majority |
| 169 | 1 | 1 | 1 | majority |
| 170 | 1 | 1 | 1 | majority |
| 171 | 1 | 1 | 1 | majority |
| 172 | 1 | 1 | 1 | majority |
| 173 | 1 | 1 | 1 | majority |
| 174 | 1 | 1 | 1 | majority |
| 175 | 1 | 1 | 1 | majority |
| 176 | 1 | 1 | 1 | majority |
| 177 | 1 | 1 | 1 | majority |
| 178 | 1 | 1 | 1 | majority |
| 179 | 1 | 1 | 1 | majority |
| 180 | 1 | 1 | 1 | majority |
| 181 | 1 | 1 | 1 | majority |
| 182 | 1 | 1 | 1 | majority |
| 183 | 1 | 1 | 1 | majority |
| 184 | 1 | 1 | 1 | majority |
| 185 | 1 | 1 | 1 | majority |

## Top-20 grid configs (mean vs std, colour = kernel)
| Params | Mean BA | Std BA | kernel |
| --- | --- | --- | --- |
| {'C': 1, 'gamma': 1, 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 10, 'gamma': 1, 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 100, 'gamma': 1, 'kernel': 'rbf'} | 0.9033 | 0.0312 | rbf |
| {'C': 100, 'gamma': 'scale', 'kernel': 'poly'} | 0.8692 | 0.1050 | poly |
| {'C': 100, 'gamma': 'auto', 'kernel': 'poly'} | 0.8684 | 0.1188 | poly |
| {'C': 10, 'gamma': 'auto', 'kernel': 'sigmoid'} | 0.8623 | 0.1273 | sigmoid |
| {'C': 10, 'gamma': 'scale', 'kernel': 'sigmoid'} | 0.8523 | 0.1414 | sigmoid |
| {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} | 0.8506 | 0.1239 | rbf |
| {'C': 10, 'gamma': 0.1, 'kernel': 'sigmoid'} | 0.8137 | 0.1822 | sigmoid |
| {'C': 100, 'gamma': 0.01, 'kernel': 'sigmoid'} | 0.8137 | 0.1822 | sigmoid |
| {'C': 1, 'kernel': 'linear'} | 0.8137 | 0.1822 | linear |
| {'C': 0.1, 'gamma': 1, 'kernel': 'poly'} | 0.8106 | 0.1940 | poly |
| {'C': 100, 'gamma': 0.1, 'kernel': 'poly'} | 0.8106 | 0.1940 | poly |
