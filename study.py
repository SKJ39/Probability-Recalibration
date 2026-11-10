"""Professor-aligned Topic 20. Development evaluation ONLY, never score test here.

Run: python -m src.study --dataset spambase --draws 50
Data: original official UCI ZIPs in data/raw/. Python >=3.10, sklearn >=1.6.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Protocol:
    seed: int = 2026
    train_fraction: float = 0.40
    calibration_fraction: float = 0.30
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    logistic_C_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    cv_folds: int = 5
    forest_trees: int = 160
    forest_min_leaf: int = 3
    forest_max_features: str = 'sqrt'
    bins: int = 10
    calibration_sizes: tuple[int, ...] = (100, 250, 500, 1000)
    fn_fp_ratios: tuple[int, ...] = (1, 5, 20)
    bootstrap_repetitions: int = 300
    main_ratio: int = 5

PROTOCOL = Protocol()

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(name: str) -> tuple[pd.DataFrame, pd.Series, Path]:
    archive = ROOT / 'data' / 'raw' / ('spambase.zip' if name == 'spambase' else 'shoppers.zip')
    if name not in ('spambase', 'shoppers'):
        raise ValueError('Dataset must be spambase or shoppers')
    if not archive.is_file():
        raise FileNotFoundError(f'Missing original UCI dataset: {archive}')
    with zipfile.ZipFile(archive) as zf:
        files = [f for f in zf.namelist() if f.lower().endswith('.data' if name == 'spambase' else '.csv')]
        if len(files) != 1:
            raise ValueError(f'Expected one data file in {archive}, found: {files}')
        with zf.open(files[0]) as file:
            frame = pd.read_csv(file, header=None if name == 'spambase' else 'infer')
    if name == 'spambase':
        assert frame.shape == (4601, 58), f'Unexpected Spambase shape {frame.shape}'
        X = frame.iloc[:, :57].copy()
        X.columns = [f'feature_{i:02d}' for i in range(57)]
        y = frame.iloc[:, 57].astype(int)
    else:
        assert frame.shape == (12330, 18), f'Unexpected Shoppers shape {frame.shape}'
        X = frame.drop(columns=['Revenue']).copy()
        y = frame['Revenue'].astype(int)
        for c in ('OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend'):
            X[c] = X[c].astype(str)
    if X.isna().any().any() or y.isna().any():
        raise ValueError('Unexpected missing data: inspect the original file')
    if set(y.unique()) != {0, 1}:
        raise ValueError('Expected binary target coded 0/1')
    return X.reset_index(drop=True), y.reset_index(drop=True), archive


def group_hashes(X: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(X, index=False).to_numpy(dtype='uint64')


def split_indices(X: pd.DataFrame, y: pd.Series, p: Protocol = PROTOCOL) -> dict[str, np.ndarray]:
    """Keep identical feature rows together. Stratify at unique-feature GROUP level.

    Percentages are approximate in row counts because groups cannot be broken up.
    Fixed indexes never change when calibration sample size changes.
    """
    gh = group_hashes(X)
    work = pd.DataFrame({'g': gh, 'y': np.asarray(y, dtype=int)})
    agg = work.groupby('g', sort=False)['y'].agg(['mean', 'size'])
    group_ids = agg.index.to_numpy(dtype='uint64')
    majority = (agg['mean'].to_numpy() >= .5).astype(int)
    def part(ids, labels, fraction, seed):
        return train_test_split(ids, test_size=fraction, stratify=labels, random_state=seed)
    kept, test = part(group_ids, majority, p.test_fraction, p.seed)
    labels = pd.Series(majority, index=group_ids)
    main, calval = part(kept, labels.loc[kept].to_numpy(),
                        (p.calibration_fraction + p.validation_fraction) / (1-p.test_fraction), p.seed + 1)
    cal, val = part(calval, labels.loc[calval].to_numpy(),
                    p.validation_fraction / (p.calibration_fraction + p.validation_fraction), p.seed + 2)
    out = {'train': np.flatnonzero(np.isin(gh, main)),
           'calibration_pool': np.flatnonzero(np.isin(gh, cal)),
           'validation': np.flatnonzero(np.isin(gh, val)),
           'reserved_test': np.flatnonzero(np.isin(gh, test))}
    assert len(np.unique(np.concatenate(list(out.values())))) == len(y)
    sets = [set(gh[idx]) for idx in out.values()]
    assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i+1,4))
    assert len(out['calibration_pool']) >= max(p.calibration_sizes), 'Calibration pool is too small'
    assert all(len(np.unique(y.iloc[idx])) == 2 for idx in out.values())
    return out


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric = [c for c in X.columns if c not in cat]
    steps = []
    if numeric:
        steps.append(('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())]), numeric))
    if cat:
        steps.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
    return ColumnTransformer(steps, remainder='drop', sparse_threshold=0)


def fit_models(X: pd.DataFrame, y: pd.Series, p: Protocol = PROTOCOL):
    groups = group_hashes(X)
    cv = StratifiedGroupKFold(n_splits=p.cv_folds, shuffle=True, random_state=p.seed)
    pipe = Pipeline([('features', preprocessor(X)),
                     ('classifier', LogisticRegression(solver='lbfgs', max_iter=1500, random_state=p.seed))])
    grid = GridSearchCV(pipe, {'classifier__C': list(p.logistic_C_grid)},
                        scoring='neg_log_loss', cv=cv, refit=True, n_jobs=1,
                        error_score='raise')
    grid.fit(X, y, groups=groups)
    rf = Pipeline([('features', preprocessor(X)),
                   ('classifier', RandomForestClassifier(n_estimators=p.forest_trees,
                     min_samples_leaf=p.forest_min_leaf, max_features=p.forest_max_features,
                     random_state=p.seed, n_jobs=1))])
    rf.fit(X, y)
    return {'Logistic Regression': grid.best_estimator_, 'Random Forest': rf}, {
        'logistic_C': float(grid.best_params_['classifier__C']),
        'training_CV_logloss': float(-grid.best_score_),
        'cv_folds': p.cv_folds,
        'forest_params': {'n_estimators': p.forest_trees,
                          'min_samples_leaf': p.forest_min_leaf,
                          'max_features': p.forest_max_features}}


def cost_threshold(fn_fp_ratio: int) -> float:
    return 1.0 / (1 + fn_fp_ratio)


def per_case_cost(y, prob, fn_fp_ratio: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    assert y.shape == prob.shape
    pred = prob >= cost_threshold(fn_fp_ratio)
    return (((pred == 1) & (y == 0)).astype(float) +
            fn_fp_ratio * ((pred == 0) & (y == 1)).astype(float))


def probability_metrics(y, prob) -> dict[str, float]:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, 1-1e-12)
    return {'brier': float(brier_score_loss(y, prob)),
            'logloss': float(log_loss(y, prob, labels=[0, 1]))}


def model_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Fixed base model scores. Platt uses decision_function if available."""
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float).reshape(-1)


def fit_score_calibrator(scores: np.ndarray, labels: np.ndarray, method: str):
    """Fit only a 1-D post-hoc mapping; base classifier remains untouched.

    Sigmoid is Platt-style logistic calibration on raw classifier scores,
    with fixed weak L2 regularization for numerical stability.
    Isotonic is non-decreasing least squares on the same raw scores.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if method == 'sigmoid':
        cal = LogisticRegression(C=100., max_iter=400, solver='lbfgs')
        cal.fit(scores.reshape(-1, 1), labels)
        return lambda sc: cal.predict_proba(np.asarray(sc).reshape(-1, 1))[:, 1]
    if method == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds='clip')
        cal.fit(scores, labels)
        return lambda sc: cal.predict(np.asarray(sc).reshape(-1))
    raise ValueError(method)


def calibration_indices(y_cal: pd.Series, size: int, seed: int) -> np.ndarray:
    selected, _ = train_test_split(np.arange(len(y_cal)), train_size=size,
                                  stratify=y_cal.to_numpy(), random_state=seed)
    return np.sort(selected)


def two_way_bootstrap(delta: np.ndarray, B: int, seed: int) -> tuple[float, float]:
    """Paired two-way bootstrap of calibration draws and shared validation cases.

    delta[r,i] is the per-case loss difference on one validation point for
    calibration draw r. Resampling rows and columns is approximate, conditional
    on fixed base model and one validation population; NOT an external CI.
    """
    rng = np.random.default_rng(seed)
    draws, cases = delta.shape
    stats = np.empty(B, dtype=float)
    for b in range(B):
        r = rng.integers(0, draws, size=draws)
        i = rng.integers(0, cases, size=cases)
        stats[b] = np.mean(delta[np.ix_(r, i)])
    return tuple(np.quantile(stats, [.025, .975]).astype(float))


def make_figures(raw: pd.DataFrame, summary: pd.DataFrame, folder: Path, dataset: str, p: Protocol, label: str = 'validation'):
    folder.mkdir(parents=True, exist_ok=True)
    # Nov 9 requirement: Logistic Regression; exactly 100 and 1000; all three prespecified ratios.
    part = raw[(raw.model == 'Logistic Regression') & raw.calibration_size.isin([100, 1000])]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), sharex=True)
    for ax, ratio in zip(axes, p.fn_fp_ratios):
        for method, color in (('sigmoid', '#1b78aa'), ('isotonic', '#d06d36')):
            sel = part[part.method == method]
            values = [sel.loc[sel.calibration_size == n, f'delta_cost_ratio_{ratio}'] for n in (100, 1000)]
            mean = [v.mean() for v in values]
            std = [v.std(ddof=1) for v in values]
            ax.errorbar([100, 1000], mean, yerr=std, fmt='o-', capsize=4, label=method, color=color)
        ax.axhline(0, color='black', ls='--', alpha=.6, lw=1)
        ax.set_title(f'False-negative : false-positive = {ratio}:1')
        ax.set_xlabel('Calibration observations')
        ax.set_xticks([100, 1000])
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Change in mean cost per validation record\n(calibrated - original; lower is better)')
    axes[-1].legend(loc='best', fontsize=8)
    fig.suptitle(f'{dataset}: Logistic Regression calibration ({label}; {int(part.draw.max())+1} draws; SD bars)', fontsize=12)
    fig.tight_layout()
    fig.savefig(folder/'NOV09_logistic_cost_vs_size.png', dpi=190)
    plt.close(fig)

    # Full exploratory study all budgets, both models, cost 1:5.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, ('Logistic Regression', 'Random Forest')):
        for method, color in (('sigmoid', '#1b78aa'), ('isotonic', '#d06d36')):
            block = summary[(summary.model==model) & (summary.method==method) &
                            (summary.metric == f'cost_ratio_{p.main_ratio}')].sort_values('calibration_size')
            ax.errorbar(block.calibration_size, block.mean_delta,
                        yerr=block.sd_across_draws, fmt='o-', label=method,
                        color=color, capsize=3)
        ax.axhline(0, color='black', ls='--', alpha=.6)
        ax.set_xscale('log'); ax.set_xticks(list(p.calibration_sizes))
        ax.set_xticklabels(list(p.calibration_sizes)); ax.grid(alpha=.2)
        ax.set_title(model)
        ax.set_xlabel('Calibration size (observations)')
    axes[0].set_ylabel('Change in cost per validation record\n(calibrated - original; SD bars)')
    axes[1].legend(loc='best')
    fig.suptitle(f'{dataset}: {label} decision-cost comparison (1:{p.main_ratio})')
    fig.tight_layout()
    fig.savefig(folder/'full_budget_cost_1_to_5.png', dpi=190)
    plt.close(fig)


def run_dataset(dataset: str, draws: int = 50, p: Protocol = PROTOCOL, output: Path | None = None, final_test: bool = False):
    if draws < 2:
        raise ValueError('At least two calibration draws are required')
    X, y, archive = load_data(dataset)
    splits = split_indices(X, y, p)
    out = output or ROOT/'results'/('final_test' if final_test else 'validation')/dataset
    if final_test and (out/'metadata.json').exists():
        raise RuntimeError(f'Final test already evaluated for {dataset}; refusing to overwrite. Verify saved predictions instead.')
    if final_test:
        locked_meta = ROOT/'results'/'validation'/dataset/'metadata.json'
        if not locked_meta.exists():
            raise RuntimeError('Missing completed validation protocol: refusing final test.')
        previous = json.loads(locked_meta.read_text())
        serialized_protocol = json.loads(json.dumps(asdict(p)))
        if previous['protocol'] != serialized_protocol or previous['draws'] != draws:
            raise RuntimeError('Final-test protocol does not match the frozen validation protocol.')
    out.mkdir(parents=True, exist_ok=True)
    group = group_hashes(X)
    manifest = pd.DataFrame({'source_row': np.arange(len(X)), 'feature_hash': group,
                             'label': y.to_numpy(), 'split': 'UNASSIGNED'})
    for part, idx in splits.items():
        manifest.loc[idx, 'split'] = part
    if not final_test:
        # Keep the validation-stage export from exposing reserved-test labels.
        manifest.loc[manifest['split'] == 'reserved_test', 'label'] = float('nan')
    manifest.to_csv(out/'split_manifest.csv', index=False)
    counts = {}
    for part, idx in splits.items():
        counts[part] = {'rows': int(len(idx)),
                        'unique_feature_vectors': int(len(np.unique(group[idx])))}
        if part != 'reserved_test' or final_test:
            counts[part]['positives'] = int(y.iloc[idx].sum())
            counts[part]['fraction_positive'] = float(y.iloc[idx].mean())
    # Reserved test is NEVER used to fit/calibrate/measure in this function.
    X_train = X.iloc[splits['train']]; y_train = y.iloc[splits['train']]
    X_cal = X.iloc[splits['calibration_pool']]; y_cal = y.iloc[splits['calibration_pool']]
    eval_part = 'reserved_test' if final_test else 'validation'
    X_val = X.iloc[splits[eval_part]]; y_val = y.iloc[splits[eval_part]].to_numpy()
    models, tuning = fit_models(X_train, y_train, p)
    metadata = {
        'dataset': dataset, 'archive_sha256': sha256(archive),
        'protocol': asdict(p), 'split_counts': counts,
        'source_rows': len(X), 'features': X.shape[1],
        'split_manifest_sha256': sha256(out/'split_manifest.csv'),
        'tuning': tuning, 'numpy_version': np.__version__,
        'pandas_version': pd.__version__, 'sklearn_version': sklearn.__version__,
        'python_version': platform.python_version(), 'draws': draws,
        'test_scored': bool(final_test), 'note': (
            'Final pre-specified test grid; research datasets were inspected in earlier exploratory work.'
            if final_test else 'Validation only. Previously examined data: exploratory.')}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'{dataset} split rows: {counts}; C={tuning["logistic_C"]}', flush=True)

    raw = []
    summaries = []
    pred_dir = out/'prediction_arrays'
    pred_dir.mkdir(parents=True, exist_ok=True)
    reference = []
    for model_i, (model_name, model) in enumerate(models.items()):
        p_orig = model.predict_proba(X_val)[:, 1]
        base_cal_scores = model_scores(model, X_cal)
        base_val_scores = model_scores(model, X_val)
        orig = probability_metrics(y_val, p_orig)
        for ratio in p.fn_fp_ratios:
            orig[f'cost_ratio_{ratio}'] = float(per_case_cost(y_val, p_orig, ratio).mean())
        reference.append({'model': model_name, **orig})
        for size in p.calibration_sizes:
            if size > len(X_cal):
                continue
            for method_i, method in enumerate(('sigmoid', 'isotonic')):
                preds = []
                for draw in range(draws):
                    # The same calibration rows are used for both methods and classifiers.
                    subset = calibration_indices(y_cal, size, seed=p.seed + 100000*size + draw)
                    fitted_cal = fit_score_calibrator(base_cal_scores[subset], y_cal.iloc[subset].to_numpy(), method)
                    p_new = fitted_cal(base_val_scores)
                    preds.append(p_new)
                    metrics = probability_metrics(y_val, p_new)
                    row = {'dataset': dataset, 'model': model_name, 'method': method,
                           'calibration_size': size, 'draw': draw, **metrics,
                           'delta_brier': metrics['brier']-orig['brier'],
                           'delta_logloss': metrics['logloss']-orig['logloss']}
                    for ratio in p.fn_fp_ratios:
                        cost = float(per_case_cost(y_val, p_new, ratio).mean())
                        row[f'cost_ratio_{ratio}'] = cost
                        row[f'delta_cost_ratio_{ratio}'] = cost-orig[f'cost_ratio_{ratio}']
                    raw.append(row)
                mat = np.stack(preds)
                file_key = f'{model_name.lower().replace(" ", "_")}_{method}_{size}'
                np.savez_compressed(pred_dir/f'{file_key}.npz',
                                    y_eval=y_val, original=p_orig, calibrated=mat)
                for metric in ('brier', 'logloss') + tuple(f'cost_ratio_{r}' for r in p.fn_fp_ratios):
                    if metric == 'brier':
                        difference = ((mat-y_val[None,:])**2 - (p_orig[None,:]-y_val[None,:])**2)
                    elif metric == 'logloss':
                        q = np.clip(mat, 1e-12, 1-1e-12)
                        base = np.clip(p_orig,1e-12,1-1e-12)
                        new_loss = -(y_val[None,:]*np.log(q)+(1-y_val[None,:])*np.log1p(-q))
                        old_loss = -(y_val*np.log(base)+(1-y_val)*np.log1p(-base))
                        difference = new_loss-old_loss[None,:]
                    else:
                        ratio = int(metric.split('_')[-1])
                        difference = (np.stack([per_case_cost(y_val, pp, ratio) for pp in mat]) -
                                      per_case_cost(y_val, p_orig, ratio)[None,:])
                    vals = difference.mean(axis=1)
                    ci_low, ci_high = two_way_bootstrap(difference, p.bootstrap_repetitions,
                        seed=202600 + 101*size + 5*model_i + 17*method_i + sum(ord(c) for c in metric))
                    summaries.append({'dataset': dataset, 'model': model_name, 'method': method,
                        'calibration_size': size, 'metric': metric, 'draws': draws,
                        'mean_delta': float(vals.mean()), 'sd_across_draws': float(vals.std(ddof=1)),
                        'ci_low': float(ci_low), 'ci_high': float(ci_high),
                        'conclusion': ('improvement' if ci_high < 0 else
                                       'worse' if ci_low > 0 else 'tie_or_inconclusive')})
            print(f'  {model_name}: calibration size {size} done', flush=True)
    raw_df = pd.DataFrame(raw)
    summary_df = pd.DataFrame(summaries)
    raw_df.to_csv(out/'calibration_draws.csv', index=False, float_format='%.12g')
    summary_df.to_csv(out/'calibration_summary.csv', index=False, float_format='%.12g')
    pd.DataFrame(reference).to_csv(out/'uncalibrated_validation.csv', index=False, float_format='%.12g')
    make_figures(raw_df, summary_df, out/'figures', dataset, p,
                 label=('final_test' if final_test else 'validation'))
    # Smallest *tested* budget with two-way-bootstrap CI wholly below zero.
    first_improvements = []
    for (model_name, method, metric), v in summary_df[summary_df.metric.str.startswith('cost_ratio_')].groupby(['model','method','metric']):
        c = v[v.ci_high < 0].sort_values('calibration_size')
        first_improvements.append({'model': model_name, 'method': method, 'cost_metric': metric,
                                   'smallest_tested_supported_improvement':
                                      int(c.calibration_size.iloc[0]) if len(c) else None})
    (out/'minimum_budget_findings.json').write_text(json.dumps(first_improvements, indent=2))
    (out/'CHECKPOINT_STATUS.txt').write_text(
        ('FINAL TEST EVALUATED UNDER LOCKED PROTOCOL.\n' if final_test else
         'Validation-only results. No final test outcomes scored.\n') +
        'Original dataset has been explored previously; inference is exploratory.\n'
        'SD bars describe 50 calibration draws, conditional on one fixed base model.\n'
        'Paired two-way bootstrap intervals resample validation rows and calibration draws.\n'
        'Do not call this independent external confirmation.\n', encoding='utf-8')
    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['spambase','shoppers','both'], default='spambase')
    parser.add_argument('--draws', type=int, default=50)
    parser.add_argument('--evaluate-final-test', action='store_true',
                        help='Run once only AFTER the professor feedback and all methods are frozen')
    parser.add_argument('--confirm-final', default='', help='Must be exactly LOCKED_AND_APPROVED')
    args = parser.parse_args()
    if args.evaluate_final_test and args.confirm_final != 'LOCKED_AND_APPROVED':
        parser.error('Final test blocked. Must explicitly confirm LOCKED_AND_APPROVED.')
    if not args.evaluate_final_test and args.confirm_final:
        parser.error('--confirm-final applies only to final-test evaluation')
    datasets = ['spambase','shoppers'] if args.dataset=='both' else [args.dataset]
    if args.evaluate_final_test:
        for ds in datasets:
            target = ROOT / 'results' / 'final_test' / ds / 'metadata.json'
            if target.exists():
                parser.error(f'Already evaluated {ds} final test: refusing repeat/overwrite.')
    for ds in datasets:
        run_dataset(ds, draws=args.draws, final_test=args.evaluate_final_test)

if __name__ == '__main__':
    main()
