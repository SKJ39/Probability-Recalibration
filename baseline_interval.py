"""Recompute the Nov 9 uncalibrated Logistic Regression validation result and cluster-bootstrap 95% interval."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/'validation'/'spambase'
with np.load(R/'prediction_arrays'/'logistic_regression_isotonic_100.npz') as z:
    y=z['y_eval'].astype(int)
    p=z['original'].astype(float)
m=pd.read_csv(R/'split_manifest.csv')
validation=m.loc[m['split'].eq('validation')].copy()
assert len(validation)==len(y)==682
assert np.array_equal(validation['label'].to_numpy().astype(int),y)
assert np.isfinite(p).all()
loss=(p-y)**2
brier=float(loss.mean())
logloss=float((-(y*np.log(np.clip(p,1e-12,1-1e-12))+(1-y)*np.log1p(-np.clip(p,1e-12,1-1e-12)))).mean())
ratio=5
pred=(p>=1/(1+ratio)).astype(int)
cost=float((((pred==1)&(y==0))+ratio*((pred==0)&(y==1))).mean())
# Bootstrap unique-feature groups, preserving their duplicated examples.
ids,idx=np.unique(validation['feature_hash'].astype(str).to_numpy(),return_inverse=True)
size=np.bincount(idx)
summed=np.bincount(idx, weights=loss)
rng=np.random.default_rng(2026)
boot=np.empty(10_000)
for i in range(len(boot)):
    sampled=rng.integers(len(ids),size=len(ids))
    boot[i]=summed[sampled].sum()/size[sampled].sum()
ci=np.quantile(boot,[.025,.975])
print(json.dumps({'dataset':'Spambase','validation_rows':len(y),'unique_feature_groups':len(ids),
'brier':brier,'brier_95pct_group_bootstrap_10000':[float(x) for x in ci],
'logloss':logloss,'decision_cost_fn_fp_5':cost,'seed':2026},indent=2))
