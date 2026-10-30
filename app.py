from pathlib import Path
import json
import pandas as pd
import streamlit as st
from retention import retention

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Telecom Retention Intelligence',page_icon='📡',layout='wide')
st.title('Telecom Churn & Retention Intelligence')
st.caption('IBM public sample • Portfolio demonstration • No Vodafone customer data')
if not (ROOT/'reports/metrics.json').exists():
    st.info('Run python churn.py first.'); st.stop()
report = json.loads((ROOT/'reports/metrics.json').read_text())
queue = pd.read_csv(ROOT/'reports/retention_queue.csv')
overview, campaign, explain = st.tabs(['Model performance','Retention planner','Explainability'])
with overview:
    a,b,c = st.columns(3)
    a.metric('Selected model',report['selected_model'])
    b.metric('Test ROC AUC',f"{report['test'][report['selected_model']]['roc_auc']:.3f}")
    c.metric('Test average precision',f"{report['test'][report['selected_model']]['pr_auc']:.3f}")
    st.write('Model chosen using validation average precision. Threshold chosen using validation F1.')
    st.dataframe(pd.DataFrame(report['test']).T.drop(columns=['confusion_matrix']),width='stretch')
    st.image(str(ROOT/'reports/evaluation.png'))
    st.write('Confusion matrix [actual rows / predicted columns, no churn then churn]')
    st.dataframe(report['test'][report['selected_model']]['confusion_matrix'])
    st.subheader('Descriptive subgroup audit')
    st.caption('Held-out groups; these rates do not establish fairness. Review sample sizes, uncertainty and feature policy before deployment.')
    st.dataframe(pd.read_csv(ROOT/'reports/subgroup_audit.csv'),width='stretch')
with campaign:
    st.warning('Scenario estimates, not measured campaign uplift. Risk does not establish that an offer changes behavior.')
    a,b,c,d = st.columns(4)
    months = a.slider('Value horizon (months)',1,24,6)
    margin = b.slider('Contribution margin',0.,1.,.4)
    success = c.slider('Assumed rescue rate among would-be churners',0.,1.,.2)
    cost = d.number_input('Cost per contacted customer (dataset currency)',min_value=0.,value=15.)
    capacity = st.slider('Contact capacity',1,len(queue),100)
    queue['expected_net_value'] = retention(queue.MonthlyCharges,queue.churn_probability,months,margin,success,cost)
    candidates = queue[queue.expected_net_value>0].sort_values('expected_net_value',ascending=False).head(capacity)
    st.metric('Estimated total net value',f'{candidates.expected_net_value.sum():,.2f}')
    st.caption('p(churn) × rescue rate × monthly charges × months × margin − contact cost. Cost applies to every contact; no discounting or cannibalization modeled.')
    st.dataframe(candidates.drop(columns=['actual_churn','eligible']),width='stretch')
    st.download_button('Download scenario queue',candidates.drop(columns=['actual_churn','eligible']).to_csv(index=False),'retention_scenario.csv','text/csv')
with explain:
    st.write('SHAP explains the XGBoost comparator in log-odds units, even if another model wins. These are associations, not causal reasons.')
    st.image(str(ROOT/'reports/shap_summary.png'))
    st.dataframe(pd.read_csv(ROOT/'reports/shap_importance.csv').head(20),width='stretch')
    local = pd.read_csv(ROOT/'reports/shap_local.csv',index_col=0)
    customer = st.selectbox('Explain a held-out sample customer',local.index)
    row = local.loc[customer]
    st.bar_chart(row.loc[row.abs().nlargest(10).index])
