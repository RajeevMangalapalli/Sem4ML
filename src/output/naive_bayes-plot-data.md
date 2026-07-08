# Plot data — `naive_bayes.py`
## LOOCV overall metrics (GaussianNB)
| Metric | Score |
| --- | --- |
| Accuracy | 0.9624 |
| Precision | 0.9888 |
| Recall | 0.9724 |
| F1 Score | 0.9805 |
| Balanced Accuracy | 0.7862 |

## Repeated Stratified K-Fold CV (5×20)
| Metric | Mean | Std |
| --- | --- | --- |
| Accuracy | 0.9624 | 0.0257 |
| Precision | 0.9890 | 0.0135 |
| Recall | 0.9724 | 0.0240 |
| F1 Score | 0.9804 | 0.0135 |
| Balanced Accuracy | 0.7862 | 0.2441 |

## LOOCV per-sample correctness (1=correct, 0=wrong)
| Sample index | True label | Predicted | Correct | Class |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1 | minority |
| 1 | 0 | 0 | 1 | minority |
| 2 | 0 | 0 | 1 | minority |
| 3 | 1 | 0 | 0 | majority |
| 4 | 1 | 0 | 0 | majority |
| 5 | 1 | 1 | 1 | majority |
| 6 | 1 | 1 | 1 | majority |
| 7 | 1 | 1 | 1 | majority |
| 8 | 1 | 1 | 1 | majority |
| 9 | 1 | 1 | 1 | majority |
| 10 | 1 | 1 | 1 | majority |
| 11 | 1 | 1 | 1 | majority |
| 12 | 1 | 1 | 1 | majority |
| 13 | 1 | 1 | 1 | majority |
| 14 | 1 | 1 | 1 | majority |
| 15 | 1 | 1 | 1 | majority |
| 16 | 1 | 1 | 1 | majority |
| 17 | 1 | 1 | 1 | majority |
| 18 | 1 | 1 | 1 | majority |
| 19 | 1 | 1 | 1 | majority |
| 20 | 1 | 1 | 1 | majority |
| 21 | 1 | 1 | 1 | majority |
| 22 | 1 | 1 | 1 | majority |
| 23 | 0 | 1 | 0 | minority |
| 24 | 1 | 1 | 1 | majority |
| 25 | 1 | 1 | 1 | majority |
| 26 | 1 | 1 | 1 | majority |
| 27 | 1 | 1 | 1 | majority |
| 28 | 1 | 1 | 1 | majority |
| 29 | 1 | 1 | 1 | majority |
| 30 | 1 | 1 | 1 | majority |
| 31 | 1 | 1 | 1 | majority |
| 32 | 1 | 1 | 1 | majority |
| 33 | 0 | 1 | 0 | minority |
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
| 52 | 1 | 1 | 1 | majority |
| 53 | 1 | 1 | 1 | majority |
| 54 | 1 | 0 | 0 | majority |
| 55 | 1 | 1 | 1 | majority |
| 56 | 1 | 1 | 1 | majority |
| 57 | 1 | 1 | 1 | majority |
| 58 | 1 | 1 | 1 | majority |
| 59 | 1 | 1 | 1 | majority |
| 60 | 1 | 1 | 1 | majority |
| 61 | 1 | 1 | 1 | majority |
| 62 | 1 | 1 | 1 | majority |
| 63 | 1 | 1 | 1 | majority |
| 64 | 1 | 1 | 1 | majority |
| 65 | 1 | 1 | 1 | majority |
| 66 | 1 | 1 | 1 | majority |
| 67 | 1 | 1 | 1 | majority |
| 68 | 1 | 1 | 1 | majority |
| 69 | 1 | 1 | 1 | majority |
| 70 | 1 | 1 | 1 | majority |
| 71 | 1 | 1 | 1 | majority |
| 72 | 1 | 1 | 1 | majority |
| 73 | 1 | 1 | 1 | majority |
| 74 | 1 | 1 | 1 | majority |
| 75 | 1 | 1 | 1 | majority |
| 76 | 1 | 0 | 0 | majority |
| 77 | 1 | 1 | 1 | majority |
| 78 | 1 | 1 | 1 | majority |
| 79 | 1 | 1 | 1 | majority |
| 80 | 1 | 0 | 0 | majority |
| 81 | 1 | 1 | 1 | majority |
| 82 | 1 | 1 | 1 | majority |
| 83 | 1 | 1 | 1 | majority |
| 84 | 1 | 1 | 1 | majority |
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
| 96 | 1 | 1 | 1 | majority |
| 97 | 1 | 1 | 1 | majority |
| 98 | 1 | 1 | 1 | majority |
| 99 | 1 | 1 | 1 | majority |
| 100 | 1 | 1 | 1 | majority |
| 101 | 1 | 1 | 1 | majority |
| 102 | 1 | 1 | 1 | majority |
| 103 | 1 | 1 | 1 | majority |
| 104 | 1 | 1 | 1 | majority |
| 105 | 1 | 1 | 1 | majority |
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
| 162 | 1 | 1 | 1 | majority |
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
