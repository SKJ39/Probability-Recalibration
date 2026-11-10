"""Recalculate every saved draw from prediction arrays. No test evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.study import ROOT, PROTOCOL, group_hashes, load_data, probability_metrics, per_case_cost

def verify(dataset: str) -> dict:
    out = ROOT/'results'/'validation'/dataset
    meta = json.loads((out/'metadata.json').read_text())
    assert meta['test_scored'] is False
    assert meta['draws'] == 50
    assert meta['cv_folds'] if 'cv_folds' in meta else meta['tuning']['cv_folds'] == 5
    raw = pd.read_csv(out/'calibration_draws.csv')
    summary = pd.read_csv(out/'calibration_summary.csv')
    assert len(raw) == 2 * 2 * len(PROTOCOL.calibration_sizes) * 50
    assert set(raw['calibration_size']) == set(PROTOCOL.calibration_sizes)
    assert set(raw.method) == {'sigmoid','isotonic'}
    assert set(raw.model) == {'Logistic Regression', 'Random Forest'}
    X, y, _ = load_data(dataset)
    split = pd.read_csv(out/'split_manifest.csv')
    assert split.shape[0] == len(X)
    assert split.split.value_counts().sum() == len(y)
    assert len(split.feature_hash.astype(str).unique()) > 0
    group_set = {k: set(v.feature_hash.astype(str)) for k,v in split.groupby('split')}
    keys = list(group_set)
    assert all(not (group_set[keys[i]] & group_set[keys[j]])
               for i in range(len(keys)) for j in range(i+1,len(keys)))
    actual_hash = group_hashes(X)
    assert np.array_equal(split.feature_hash.to_numpy(dtype='uint64'), actual_hash)
    validations = 0
    for model in ('Logistic Regression', 'Random Forest'):
        for method in ('sigmoid', 'isotonic'):
            for size in PROTOCOL.calibration_sizes:
                file = out/'prediction_arrays'/f'{model.lower().replace(" ","_")}_{method}_{size}.npz'
                with np.load(file) as a:
                    y_valid = a['y_eval']; base=a['original']; cal=a['calibrated']
                assert cal.shape == (50, len(y_valid))
                assert np.isfinite(cal).all() and np.isfinite(base).all()
                assert (cal >= 0).all() and (cal <= 1).all()
                assert np.array_equal(y_valid, split.loc[split.split=='validation','label'].to_numpy())
                sel = raw[(raw.model==model)&(raw.method==method)&(raw.calibration_size==size)].sort_values('draw')
                assert len(sel)==50 and np.array_equal(sel.draw.to_numpy(),np.arange(50))
                orig = probability_metrics(y_valid,base)
                for i,p in enumerate(cal):
                    results=probability_metrics(y_valid,p)
                    row=sel.iloc[i]
                    for m in ('brier','logloss'):
                        assert np.isclose(results[m],row[m],rtol=2e-8,atol=2e-9)
                        assert np.isclose(results[m]-orig[m],row['delta_'+m],rtol=2e-8,atol=2e-9)
                    for ratio in PROTOCOL.fn_fp_ratios:
                        cost=per_case_cost(y_valid,p,ratio).mean()
                        base_cost=per_case_cost(y_valid,base,ratio).mean()
                        assert np.isclose(cost,row[f'cost_ratio_{ratio}'],rtol=2e-8,atol=2e-9)
                        assert np.isclose(cost-base_cost,row[f'delta_cost_ratio_{ratio}'],rtol=2e-8,atol=2e-9)
                vals=sel[f'delta_cost_ratio_{PROTOCOL.main_ratio}']
                this=summary[(summary.model==model)&(summary.method==method)&(summary.calibration_size==size)&
                             (summary.metric==f'cost_ratio_{PROTOCOL.main_ratio}')].iloc[0]
                assert np.isclose(vals.mean(),this.mean_delta,atol=2e-9)
                assert np.isclose(vals.std(ddof=1),this.sd_across_draws,atol=2e-9)
                validations+=len(sel)
    return {'dataset':dataset,'calibration_draws_verified':validations,
            'test_scored':meta['test_scored'], 'duplicate_group_leakage':'none found',
            'CV_folds':meta['tuning']['cv_folds']}

if __name__ == '__main__':
    result=[verify('spambase')]
    print(json.dumps(result, indent=2))
