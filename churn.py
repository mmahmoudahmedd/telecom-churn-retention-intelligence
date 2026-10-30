"""Reproducible IBM Telco churn experiment and retention scenario."""
from pathlib import Path
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, RocCurveDisplay,
    PrecisionRecallDisplay)
import joblib
from xgboost import XGBClassifier
from retention import retention

ROOT = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
NUM = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
CAT = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']
FEATURES = NUM + CAT

def clean(df):
    """Stateless coercion; learned imputation happens only inside fitted pipelines."""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required features: {sorted(missing)}')
    out = df[FEATURES].copy()
    for col in NUM:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    for col in CAT:
        out[col] = out[col].astype('object').where(out[col].notna(), np.nan)
    return out

def preprocessor():
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), NUM),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT)
    ])

def metrics(y, p, threshold):
    pred = p >= threshold
    k = max(1, int(np.ceil(len(y)*.1)))
    top = np.argsort(-p, kind='stable')[:k]
    return dict(roc_auc=float(roc_auc_score(y,p)), pr_auc=float(average_precision_score(y,p)),
        brier=float(brier_score_loss(y,p)), precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred)), f1=float(f1_score(y,pred)),
        confusion_matrix=confusion_matrix(y,pred).tolist(),
        lift_at_10pct=float(np.asarray(y)[top].mean()/np.mean(y)))

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap
    for folder in ['data', 'artifacts', 'reports']:
        (ROOT/folder).mkdir(exist_ok=True)
    raw = ROOT/'data/telco.csv'
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
        (ROOT/'data/provenance.json').write_text(json.dumps(dict(url=URL,
            downloaded_utc=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),indent=2))
    df = pd.read_csv(raw)
    if df.customerID.duplicated().any() or not df.Churn.isin(['Yes','No']).all():
        raise ValueError('Invalid IDs or labels')
    X, y = clean(df), df.Churn.eq('Yes').astype(int)
    train, test = train_test_split(np.arange(len(df)), test_size=.2, stratify=y, random_state=42)
    train, val = train_test_split(train, test_size=.25, stratify=y.iloc[train], random_state=42)
    # Descriptive exploration uses training rows only, keeping the holdout untouched.
    train_eda = df.iloc[train].copy()
    train_eda['churn'] = y.iloc[train].to_numpy()
    train_eda.groupby('Contract').agg(customers=('churn','size'),churn_rate=('churn','mean')).to_csv(ROOT/'reports/train_contract_summary.csv')
    X.iloc[train].isna().sum().rename('missing_training_rows').to_csv(ROOT/'reports/train_missingness.csv')
    models = {
        'dummy': DummyClassifier(strategy='prior'),
        'logistic': LogisticRegression(max_iter=1500, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=250, min_samples_leaf=8, n_jobs=2, random_state=42),
        'xgboost': XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.04,
            subsample=.85, colsample_bytree=.85, eval_metric='logloss', n_jobs=2, random_state=42)
    }
    fitted, validation = {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([('preprocess', preprocessor()), ('model', estimator)])
        pipe.fit(X.iloc[train], y.iloc[train])
        p = pipe.predict_proba(X.iloc[val])[:,1]
        fitted[name] = pipe
        validation[name] = metrics(y.iloc[val], p, .5)
        print(name, validation[name], flush=True)
    winner = max(validation, key=lambda n: validation[n]['pr_auc'])
    pipe = fitted[winner]
    vp = pipe.predict_proba(X.iloc[val])[:,1]
    thresholds = np.linspace(.05,.95,181)
    threshold = float(max(thresholds, key=lambda t: f1_score(y.iloc[val],vp>=t)))
    # No refit: threshold and SHAP remain tied to the exact validated model.
    test_results = {n:metrics(y.iloc[test], m.predict_proba(X.iloc[test])[:,1], threshold if n==winner else .5)
                    for n,m in fitted.items()}
    p = pipe.predict_proba(X.iloc[test])[:,1]
    scores = df.iloc[test][['customerID','MonthlyCharges','Contract','tenure']].copy()
    scores['actual_churn'] = y.iloc[test].to_numpy()
    scores['churn_probability'] = p
    scores['expected_net_value'] = retention(scores.MonthlyCharges, p)
    scores['eligible'] = scores.expected_net_value > 0
    scores.sort_values('expected_net_value',ascending=False).to_csv(ROOT/'reports/retention_queue.csv',index=False)
    # Descriptive subgroup audit; small cells and uncertainty limit interpretation.
    audit = df.iloc[test][['gender','SeniorCitizen']].copy()
    audit['actual'] = y.iloc[test].to_numpy(); audit['predicted'] = p >= threshold
    audits = []
    for field in ['gender','SeniorCitizen']:
        for value, group in audit.groupby(field):
            audits.append(dict(field=field,group=str(value),n=len(group),
                prevalence=float(group.actual.mean()),selection_rate=float(group.predicted.mean()),
                recall=float(recall_score(group.actual,group.predicted,zero_division=0)),
                precision=float(precision_score(group.actual,group.predicted,zero_division=0))))
    pd.DataFrame(audits).to_csv(ROOT/'reports/subgroup_audit.csv',index=False)
    manifest = {s:df.iloc[idx].customerID.tolist() for s,idx in [('train',train),('validation',val),('test',test)]}
    (ROOT/'artifacts/splits.json').write_text(json.dumps(manifest,indent=2))
    result = dict(selected_model=winner, threshold=threshold, selection='Validation average precision; validation F1 threshold',
        validation=validation, test=test_results, rows=len(df), churn_rate=float(y.mean()),
        splits={s:len(v) for s,v in manifest.items()}, data_url=URL,
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        assumptions=dict(months=6,margin=.4,success=.2,cost=15))
    (ROOT/'reports/metrics.json').write_text(json.dumps(result,indent=2))
    lines = ['# Model results', '', f'Selected model: **{winner}**; validation F1 threshold: **{threshold:.3f}**.',
             '', f'{len(df):,} rows; churn prevalence {y.mean():.1%}; split sizes {result["splits"]}.', '',
             '| Model | Test ROC AUC | Test average precision | Precision | Recall | F1 | Brier | Lift at 10% |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in test_results.items():
        lines.append('| '+name+' | '+' | '.join(f'{m[k]:.3f}' for k in
            ['roc_auc','pr_auc','precision','recall','f1','brier','lift_at_10pct'])+' |')
    lines += ['', 'Winner selected using validation average precision, never test performance. Comparator thresholds are 0.5.',
              'SHAP explains XGBoost. Retention values are hypothetical; no causal uplift or achieved savings are claimed.',
              '', 'See README.md for assumptions, provenance, methodology and limitations.']
    (ROOT/'reports/RESULTS.md').write_text('\n'.join(lines))
    joblib.dump(dict(pipeline=pipe,threshold=threshold,name=winner), ROOT/'artifacts/model.joblib')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for name, model in fitted.items():
        prob = model.predict_proba(X.iloc[test])[:,1]
        RocCurveDisplay.from_predictions(y.iloc[test],prob,name=name,ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y.iloc[test],prob,name=name,ax=axes[1])
    fig.tight_layout(); fig.savefig(ROOT/'reports/evaluation.png',dpi=140); plt.close(fig)
    # Explain the XGBoost comparator, regardless of which model wins selection.
    xp = fitted['xgboost']
    joblib.dump(xp, ROOT/'artifacts/xgboost.joblib')
    transformed = xp.named_steps['preprocess'].transform(X.iloc[test[:200]])
    names = xp.named_steps['preprocess'].get_feature_names_out()
    explanation = shap.TreeExplainer(xp.named_steps['model'])(transformed)
    importance = pd.DataFrame({'feature':names, 'mean_abs_shap':np.abs(explanation.values).mean(axis=0)})
    importance.sort_values('mean_abs_shap',ascending=False).to_csv(ROOT/'reports/shap_importance.csv',index=False)
    shap.summary_plot(explanation.values, transformed, feature_names=names,show=False,max_display=15)
    plt.tight_layout(); plt.savefig(ROOT/'reports/shap_summary.png',bbox_inches='tight',dpi=140); plt.close()
    pd.DataFrame(explanation.values,columns=names,index=df.iloc[test[:200]].customerID).to_csv(ROOT/'reports/shap_local.csv')
    np.savez(ROOT/'artifacts/shap_values.npz',values=explanation.values,base_values=explanation.base_values,
             data=transformed,feature_names=names)
    print('COMPLETE',winner,threshold,test_results[winner],flush=True)

if __name__ == '__main__':
    main()
