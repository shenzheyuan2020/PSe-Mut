#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate results from outputs/experiments/exp1_three_split into a summary JSON
compatible with scripts/analysis/benchmark_visualization.py.

It extracts per-seed validation metrics for:
- Traditional baselines (best representatives):
  linear/elasticnet, svr/svr_rbf, tree/rf, mlp/mlp_sklearn, kernel_knn/kernel_ridge_rbf
- Siamese models: base, no_exchange, diff_only

Output path: outputs/benchmarks/all/summary_all_seeds.json
"""

import os
import json
import argparse
from sklearn.metrics import roc_auc_score


def read_json_safe(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def extract_metric(d: dict, *keys, metric: str = 'rmse'):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if isinstance(cur, dict) and metric in cur:
        return cur
    return None


def main():
    ap = argparse.ArgumentParser('build exp1 summary for visualization')
    ap.add_argument('--exp1_root', type=str, default=os.path.join('outputs', 'experiments', 'exp1_three_split'))
    ap.add_argument('--out_path', type=str, default=os.path.join('outputs', 'benchmarks', 'all', 'summary_all_seeds.json'))
    args = ap.parse_args()

    if not os.path.isdir(args.exp1_root):
        raise FileNotFoundError(f"exp1 directory not found: {args.exp1_root}")

    # discover seeds from baselines directories
    seeds = []
    for name in sorted(os.listdir(args.exp1_root)):
        if name.startswith('baselines_seed_'):
            try:
                sd = int(name.rsplit('_', 1)[-1])
            except Exception:
                continue
            seeds.append(sd)
    seeds = sorted(list(set(seeds)))
    if not seeds:
        # fallback to default
        seeds = [42, 123, 456]

    per_seed = {}
    per_seed_test = {}

    for sd in seeds:
        entry = {
            'linear': {}, 'svr': {}, 'tree': {}, 'mlp': {}, 'kernel_knn': {},
            'model': {}
        }
        entry_te = {
            'linear': {}, 'svr': {}, 'tree': {}, 'mlp': {}, 'kernel_knn': {},
            'model': {}
        }
        base_root = os.path.join(args.exp1_root, f'baselines_seed_{sd}')
        # baselines (val)
        linear = read_json_safe(os.path.join(base_root, 'linear', 'val_metrics.json')) or {}
        svr = read_json_safe(os.path.join(base_root, 'svr', 'val_metrics.json')) or {}
        tree = read_json_safe(os.path.join(base_root, 'tree', 'val_metrics.json')) or {}
        mlp = read_json_safe(os.path.join(base_root, 'mlp', 'val_metrics.json')) or {}
        kknn = read_json_safe(os.path.join(base_root, 'kernel_knn', 'val_metrics.json')) or {}
        # baselines (test)
        linear_te = read_json_safe(os.path.join(base_root, 'linear', 'test_metrics.json')) or {}
        svr_te = read_json_safe(os.path.join(base_root, 'svr', 'test_metrics.json')) or {}
        tree_te = read_json_safe(os.path.join(base_root, 'tree', 'test_metrics.json')) or {}
        mlp_te = read_json_safe(os.path.join(base_root, 'mlp', 'test_metrics.json')) or {}
        kknn_te = read_json_safe(os.path.join(base_root, 'kernel_knn', 'test_metrics.json')) or {}

        # select representatives
        # copy all available baseline methods (val)
        for k, v in linear.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry['linear'][k] = v
        for k, v in svr.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry['svr'][k] = v
        for k, v in tree.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry['tree'][k] = v
        for k, v in mlp.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry['mlp'][k] = v
        for k, v in kknn.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry['kernel_knn'][k] = v
        # copy all available baseline methods (test)
        for k, v in linear_te.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry_te['linear'][k] = v
        for k, v in svr_te.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry_te['svr'][k] = v
        for k, v in tree_te.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry_te['tree'][k] = v
        for k, v in mlp_te.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry_te['mlp'][k] = v
        for k, v in kknn_te.items():
            if isinstance(v, dict) and 'rmse' in v:
                entry_te['kernel_knn'][k] = v

        # attach AUC-ROC from raw files per baseline family to representative methods
        fam_method = {
            'linear': ('linear', 'elasticnet'),
            'svr': ('svr', 'svr_rbf'),
            'tree': ('tree', 'rf'),
            'mlp': ('mlp', 'mlp_sklearn'),
            'kernel_knn': ('kernel_knn', 'kernel_ridge_rbf'),
        }
        for fam, (dir_name, rep) in fam_method.items():
            # val
            raw_v = read_json_safe(os.path.join(base_root, dir_name, 'auc_raw_val.json'))
            if isinstance(raw_v, dict) and 'y_true_binary' in raw_v and 'scores' in raw_v:
                try:
                    auc_v = float(roc_auc_score(raw_v['y_true_binary'], raw_v['scores']))
                    entry.setdefault(fam, {}).setdefault(rep, {})['auc'] = auc_v
                except Exception:
                    pass
            # test
            raw_t = read_json_safe(os.path.join(base_root, dir_name, 'auc_raw_test.json'))
            if isinstance(raw_t, dict) and 'y_true_binary' in raw_t and 'scores' in raw_t:
                try:
                    auc_t = float(roc_auc_score(raw_t['y_true_binary'], raw_t['scores']))
                    entry_te.setdefault(fam, {}).setdefault(rep, {})['auc'] = auc_t
                except Exception:
                    pass

        # siamese models (use val_metrics.json inside each seed dir)
        for tag in ['model_base', 'model_no_exchange', 'model_diff_only', 'model_mut_only', 'model_wt_only']:
            tag_short = tag.replace('model_', '')
            seed_dir = os.path.join(args.exp1_root, tag, f'seed_{sd}')
            vm = read_json_safe(os.path.join(seed_dir, 'val_metrics.json'))
            if isinstance(vm, dict) and all(k in vm for k in ['rmse', 'mae', 'r2']):
                rec = {
                    'rmse': float(vm['rmse']),
                    'mae': float(vm['mae']),
                    'r2': float(vm['r2'])
                }
                # Optional metrics if present
                if 'pearson' in vm:
                    rec['pearson'] = float(vm['pearson'])
                if 'auc' in vm:
                    rec['auc'] = float(vm['auc'])
                entry['model'][tag_short] = rec
            # test metrics for siamese
            tm = read_json_safe(os.path.join(seed_dir, 'test_metrics.json'))
            if isinstance(tm, dict) and 'test_rmse' in tm:
                rec_te = {
                    'rmse': float(tm.get('test_rmse')), 'mae': float(tm.get('test_mae', float('nan'))),
                    'r2': float(tm.get('test_r2', float('nan')))
                }
                if 'test_pearson' in tm:
                    rec_te['pearson'] = float(tm['test_pearson'])
                if 'test_auc_roc' in tm:
                    rec_te['auc'] = float(tm['test_auc_roc'])
                entry_te['model'][tag_short] = rec_te

        per_seed[sd] = entry
        per_seed_test[sd] = entry_te

    summary = {
        'seeds': seeds,
        'per_seed': per_seed,
        'per_seed_test': per_seed_test,
        'exp1_root': args.exp1_root
    }

    out_dir = os.path.dirname(args.out_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('Saved:', args.out_path)


if __name__ == '__main__':
    main()


