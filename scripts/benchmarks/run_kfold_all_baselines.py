#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-fold run all baselines and original model ablation, and aggregate mean/std.

Example:
  python benchmarks/run_kfold_all_baselines.py \
    --data_dir siamese_dataset \
    --out_root bench_results_kfold/run_seed42 \
    --svd_dim 384 --seed 42 --k 3 --mode random \
    --use_xgb --use_lgb --use_gpu --tree_fast --tree_subsample 0.5 --svr_subsample 0.3

Optional: existing kfold file
  python tools/freeze_splits.py --data_dir siamese_dataset --mode kfold_random --k 3 --seed 42
  python benchmarks/run_kfold_all_baselines.py --data_dir siamese_dataset --out_root ... --svd_dim 384 --seed 42 --kfold_file siamese_dataset/kfold_splits.json
"""
import os
import sys
import json
import argparse
import subprocess
import statistics
from typing import Dict, List


def run(cmd: str):
    print("[RUN]", cmd)
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def safe_load_json(path: str) -> Dict:
    try:
        if os.path.exists(path):
            return json.load(open(path, 'r', encoding='utf-8'))
    except Exception:
        pass
    return {}


def aggregate_metrics(mdicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    # Input: a list of {rmse, mae, r2} dictionaries; Output: mean/std for each metric
    out: Dict[str, Dict[str, float]] = {}
    keys = set().union(*[d.keys() for d in mdicts if isinstance(d, dict)])
    for k in keys:
        vals = [float(d[k]) for d in mdicts if (isinstance(d, dict) and k in d and isinstance(d[k], (int,float)))]
        if len(vals) == 0:
            continue
        out[k] = {"mean": float(statistics.mean(vals)), "std": float(statistics.pstdev(vals))}
    return out


def main():
    ap = argparse.ArgumentParser("kfold all baselines & ablations")
    ap.add_argument("--data_dir", type=str, default="siamese_dataset")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--svd_dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mode", type=str, default="random", choices=["random","group"], help="Auto-generate if kfold_file not provided")
    ap.add_argument("--kfold_file", type=str, default=None, help="Optional: existing K-fold split file")
    ap.add_argument("--use_xgb", action='store_true')
    ap.add_argument("--use_lgb", action='store_true')
    ap.add_argument("--use_gpu", action='store_true')
    ap.add_argument("--tree_fast", action='store_true')
    ap.add_argument("--tree_subsample", type=float, default=1.0)
    ap.add_argument("--svr_subsample", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # 1) Get/generate K-fold splits
    kfold_path = args.kfold_file or os.path.join(args.data_dir, 'kfold_splits.json')
    if args.kfold_file is None:
        mode = 'kfold_random' if args.mode == 'random' else 'kfold_group'
        run(f"python tools/freeze_splits.py --data_dir {args.data_dir} --mode {mode} --k {args.k} --seed {args.seed}")
    kf = json.load(open(kfold_path, 'r', encoding='utf-8'))
    folds = kf.get('folds', [])
    if len(folds) == 0:
        raise RuntimeError("kfold_splits.json content is empty or no folds")
    folds = folds[:args.k]

    # 2) Run each fold
    summary_all: Dict[str, Dict] = {}
    # Collect metrics for each fold
    agg_linear, agg_svr, agg_tree, agg_mlp, agg_kernel = [], [], [], [], []
    agg_model_base, agg_model_noex, agg_model_diff = [], [], []

    for i, fv in enumerate(folds):
        fold_dir = os.path.join(args.out_root, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)
        # Save current fold's split_file
        split_file = os.path.join(fold_dir, 'split_info.json')
        json.dump({
            'mode': f'kfold_{args.mode}',
            'seed': int(args.seed),
            'train_idx': [int(x) for x in fv['train_idx']],
            'val_idx': [int(x) for x in fv['val_idx']],
        }, open(split_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

        # Baselines
        out_linear = os.path.join(fold_dir, 'linear')
        run(f"python benchmarks/run_linear_baselines.py --data_dir {args.data_dir} --split_file {split_file} --out_dir {out_linear} --svd_dim {args.svd_dim} --seed {args.seed}")
        agg_linear.append(safe_load_json(os.path.join(out_linear, 'val_metrics.json')))

        out_svr = os.path.join(fold_dir, 'svr')
        try:
            run(f"python benchmarks/run_svr_baselines.py --data_dir {args.data_dir} --split_file {split_file} --out_dir {out_svr} --svd_dim {args.svd_dim} --seed {args.seed} --subsample {args.svr_subsample}")
            agg_svr.append(safe_load_json(os.path.join(out_svr, 'val_metrics.json')))
        except Exception as _e:
            print(f"[WARN] fold {i} SVR skipped: {_e}")

        out_tree = os.path.join(fold_dir, 'tree')
        tree_cmd = f"python benchmarks/run_tree_baselines.py --data_dir {args.data_dir} --split_file {split_file} --out_dir {out_tree} --svd_dim {args.svd_dim} --seed {args.seed} --subsample {args.tree_subsample}"
        if args.tree_fast:
            tree_cmd += " --fast"
        if args.use_xgb:
            tree_cmd += " --use_xgb"
        if args.use_lgb:
            tree_cmd += " --use_lgb"
        run(tree_cmd)
        agg_tree.append(safe_load_json(os.path.join(out_tree, 'val_metrics.json')))

        out_mlp = os.path.join(fold_dir, 'mlp')
        run(f"python benchmarks/run_mlp_baseline.py --data_dir {args.data_dir} --split_file {split_file} --out_dir {out_mlp} --svd_dim {args.svd_dim} --seed {args.seed}")
        agg_mlp.append(safe_load_json(os.path.join(out_mlp, 'val_metrics.json')))

        out_kernel = os.path.join(fold_dir, 'kernel_knn')
        run(f"python benchmarks/run_kernel_knn_baselines.py --data_dir {args.data_dir} --split_file {split_file} --out_dir {out_kernel} --svd_dim {args.svd_dim} --seed {args.seed}")
        agg_kernel.append(safe_load_json(os.path.join(out_kernel, 'val_metrics.json')))

        # Model ablations (150 epochs)
        dev_flag = "--use_gpu" if args.use_gpu else ""
        out_model = os.path.join(fold_dir, 'model')
        os.makedirs(out_model, exist_ok=True)

        # base
        run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'base')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --split_file {split_file} {dev_flag}")
        m_base = safe_load_json(os.path.join(out_model, 'base', f'seed_{args.seed}', 'val_metrics.json'))
        if m_base: agg_model_base.append(m_base)

        # no_exchange
        run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'no_exchange')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --enable_multitask --svd_dim {args.svd_dim} --split_file {split_file} {dev_flag}")
        m_noex = safe_load_json(os.path.join(out_model, 'no_exchange', f'seed_{args.seed}', 'val_metrics.json'))
        if m_noex: agg_model_noex.append(m_noex)

        # diff_only
        run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'diff_only')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --use_diff_only --split_file {split_file} {dev_flag}")
        m_diff = safe_load_json(os.path.join(out_model, 'diff_only', f'seed_{args.seed}', 'val_metrics.json'))
        if m_diff: agg_model_diff.append(m_diff)

        # mut_only
        run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'mut_only')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --use_mut_only --split_file {split_file} {dev_flag}")
        m_mut = safe_load_json(os.path.join(out_model, 'mut_only', f'seed_{args.seed}', 'val_metrics.json'))
        if m_mut: agg_model_diff.append(m_mut)

        # wt_only
        run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'wt_only')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --use_wt_only --split_file {split_file} {dev_flag}")
        m_wt = safe_load_json(os.path.join(out_model, 'wt_only', f'seed_{args.seed}', 'val_metrics.json'))
        if m_wt: agg_model_diff.append(m_wt)

    # 3) Aggregate K-fold
    summary_k = {
        'linear': aggregate_metrics(agg_linear),
        'svr': aggregate_metrics(agg_svr) if agg_svr else {},
        'tree': aggregate_metrics(agg_tree),
        'mlp': aggregate_metrics(agg_mlp),
        'kernel_knn': aggregate_metrics(agg_kernel),
        'model': {
            'base': aggregate_metrics(agg_model_base),
            'no_exchange': aggregate_metrics(agg_model_noex),
            'diff_only_or_single_branch': aggregate_metrics(agg_model_diff),
        }
    }
    with open(os.path.join(args.out_root, 'summary_kfold.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_k, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(args.out_root, 'summary_kfold.json'))


if __name__ == "__main__":
    main()
