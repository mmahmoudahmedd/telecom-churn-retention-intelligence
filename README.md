# Telecom Churn & Retention Intelligence

An end-to-end local portfolio project using IBM's public Telco sample. This is a demonstration, not a Vodafone deployment or evidence of achieved savings.

## Run on Windows

From this project folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe churn.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. The first training run downloads the CSV if absent. Subsequent runs reuse the local data. Training regenerates only this project's generated artifacts and reports. No accounts or API keys required. Load only trusted local model files; joblib is not a safe interchange format for untrusted inputs.

Score a new CSV with the same feature schema: `.venv\Scripts\python.exe score.py input.csv scores.csv`. Target labels are optional and ignored. Existing output files are refused. Retention scores use the documented default assumptions; the dashboard lets you explore alternatives on held-out data. Unknown categorical values are accepted by the encoder but should prompt data-quality review.

## Method

Seed 42, stratified 60/20/20 train/validation/test split; saved customer IDs make split membership auditable. Unique customer IDs are checked. Customer ID and target are excluded by an explicit 19-feature allowlist, which also excludes any added churn score/reason/status fields. TotalCharges blanks become missing. Median/mode imputation, scaling and unknown-safe one-hot encoding are fitted on training data only inside each model pipeline. No SMOTE or sampling before splitting.

Dummy prior, logistic regression, random forest and XGBoost use fixed documented configurations in churn.py. Select by validation average precision (reported as PR AUC); select the winner's threshold by validation F1. The exact trained winner is retained without refitting. Report all test comparators once, without using test results for model selection. Comparator thresholds are 0.5. Metrics include ROC AUC, average precision, Brier score, precision/recall/F1, confusion matrices and top-decile lift. Test performance is a single split estimate, not cross-validation uncertainty. F1 threshold is a technical default, not an economically optimal operating policy.

## Retention assumptions

Expected incremental net value = churn probability × assumed rescue rate among would-be churners × monthly charges × horizon × contribution margin − cost per contacted customer.

Defaults: 6 months, 40% margin, 20% rescue rate, cost 15 in dataset currency units. Rank positive-value customers, capped by contact capacity. Cost applies to every contacted customer. Assumes homogeneous treatment efficacy, no discounting, no adverse effect, no intervention costs beyond the entered amount, and that the unspecified churn label horizon aligns with the scenario. These assumptions are unverified. The data contain no randomized offers or response outcomes; causal uplift and actual ROI cannot be estimated. Validate probability calibration, campaign effects and economics with current data and a randomized pilot before operational use. No outreach is sent.

## Explainability and limitations

SHAP global and local contributions explain the XGBoost comparator on the first 200 held-out customers, in raw log-odds units. They do not explain the selected model when it is different and do not establish causality. Saved NPZ contains values, base values and transformed inputs. Known limitations: sample data, no observation timestamp, no prospective feature-availability guarantee, no temporal validation, no campaign outcome data, possible demographic proxies and no production drift monitoring. Gender and senior status are included for the demo; fairness and feature-policy review are required before deployment. Do not interpret the dataset as representative of Vodafone or Egypt.

## Outputs

- reports/metrics.json: metrics, model choice, threshold, source hash, split counts and assumptions
- reports/retention_queue.csv: held-out scenario scores, including labels solely for audit
- reports/evaluation.png: ROC and precision-recall curves
- reports/shap_summary.png, shap_importance.csv, shap_local.csv: explanations
- artifacts/model.joblib, splits.json, shap_values.npz: reproducibility artifacts
- requirements-lock.txt: exact installed environment
- reports/validation.txt: test run evidence

## Data provenance

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). IBM's repository is archived and carries Apache-2.0 licensing; upstream LICENSE is included as data/UPSTREAM_LICENSE. Preserve upstream attribution and review terms before redistributing. Download timestamp and SHA-256 are recorded in data/provenance.json. This project does not claim ownership of the sample data.
