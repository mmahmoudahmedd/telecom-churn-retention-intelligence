"""Score an IBM-schema CSV with the locally trained winner; never overwrite input."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from churn import ROOT, clean, retention

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve()==args.output.resolve():
        parser.error('Output already exists; choose a new path.')
    df = pd.read_csv(args.input)
    x = clean(df)
    if x.MonthlyCharges.isna().any() or (x.MonthlyCharges<0).any():
        parser.error('Retention scoring requires valid non-negative MonthlyCharges.')
    model = joblib.load(ROOT/'artifacts/model.joblib')
    result = df[['customerID']].copy() if 'customerID' in df else pd.DataFrame({'row':df.index})
    result['churn_probability'] = model['pipeline'].predict_proba(x)[:,1]
    result['predicted_churn'] = result.churn_probability >= model['threshold']
    result['expected_net_value'] = retention(x.MonthlyCharges,result.churn_probability)
    result.sort_values('expected_net_value',ascending=False).to_csv(args.output,index=False,mode='x')
    print(f'Scored {len(result)} rows to {args.output}')

if __name__ == '__main__': main()
