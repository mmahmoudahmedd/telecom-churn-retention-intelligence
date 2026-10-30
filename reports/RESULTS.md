# Model results

Selected model: **xgboost**; validation F1 threshold: **0.375**.

7,043 rows; churn prevalence 26.5%; split sizes {'train': 4225, 'validation': 1409, 'test': 1409}.

| Model | Test ROC AUC | Test average precision | Precision | Recall | F1 | Brier | Lift at 10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy | 0.500 | 0.265 | 0.000 | 0.000 | 0.000 | 0.195 | 1.176 |
| logistic | 0.843 | 0.634 | 0.651 | 0.553 | 0.598 | 0.138 | 2.805 |
| random_forest | 0.841 | 0.644 | 0.648 | 0.473 | 0.547 | 0.137 | 2.886 |
| xgboost | 0.845 | 0.660 | 0.576 | 0.693 | 0.629 | 0.136 | 2.939 |

Winner selected using validation average precision, never test performance. Comparator thresholds are 0.5.
SHAP explains XGBoost. Retention values are hypothetical; no causal uplift or achieved savings are claimed.

See README.md for assumptions, provenance, methodology and limitations.