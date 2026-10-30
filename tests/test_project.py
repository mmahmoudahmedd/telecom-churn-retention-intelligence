import json
import numpy as np
import pandas as pd
import pytest
from churn import ROOT, clean, preprocessor, retention, FEATURES

def test_retention_economics():
    assert retention([100],[.5],6,.4,.2,15)[0] == pytest.approx(9)
    assert retention([100],[0])[0] == -15
    with pytest.raises(ValueError): retention([100],[1.1])
    with pytest.raises(ValueError): retention([100],[.5],success=2)

def test_whitelist_and_missing_charges():
    df = pd.read_csv(ROOT/'data/telco.csv')
    df['Churn Score'] = 99
    x = clean(df)
    assert list(x.columns) == FEATURES
    assert 'customerID' not in x and 'Churn' not in x
    assert x.TotalCharges.isna().sum() == 11
    with pytest.raises(ValueError): clean(df.drop(columns=['tenure']))

def test_train_only_imputation_and_unknown_category():
    x = clean(pd.read_csv(ROOT/'data/telco.csv').iloc[:30])
    prep = preprocessor().fit(x)
    median = prep.named_transformers_['num'].named_steps['impute'].statistics_.copy()
    unseen = x.iloc[:1].copy(); unseen['Contract']='never_seen'; unseen['TotalCharges']=np.nan
    assert np.isfinite(prep.transform(unseen)).all()
    np.testing.assert_array_equal(median,prep.named_transformers_['num'].named_steps['impute'].statistics_)

def test_split_disjoint_complete():
    splits = json.loads((ROOT/'artifacts/splits.json').read_text())
    sets = [set(v) for v in splits.values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert len(set.union(*sets)) == 7043

def test_dashboard():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    app.slider[0].set_value(12).run()
    assert not app.exception

def test_model_roundtrip_and_shap_additivity():
    import joblib
    from scipy.special import expit
    bundle = joblib.load(ROOT/'artifacts/model.joblib')
    queue = pd.read_csv(ROOT/'reports/retention_queue.csv')
    raw = pd.read_csv(ROOT/'data/telco.csv').set_index('customerID').loc[queue.customerID].reset_index()
    p = bundle['pipeline'].predict_proba(clean(raw))[:,1]
    np.testing.assert_allclose(p,queue.churn_probability,atol=1e-7)
    saved = np.load(ROOT/'artifacts/shap_values.npz')
    assert saved['values'].shape == saved['data'].shape
    implied = expit(saved['values'].sum(axis=1)+saved['base_values'])
    xgb = joblib.load(ROOT/'artifacts/xgboost.joblib')
    predicted = xgb.named_steps['model'].predict_proba(saved['data'])[:,1]
    np.testing.assert_allclose(implied,predicted,atol=1e-5)
