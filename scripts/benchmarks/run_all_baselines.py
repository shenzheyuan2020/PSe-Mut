#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click run: Linear/Ridge/ElasticNet, SVR, RF/XGB, MLP and original model ablation, and aggregate results.

Example:
  python benchmarks/run_all_baselines.py \
    --data_dir siamese_dataset \
    --split_file siamese_dataset/split_info.json \
    --out_root bench_results_all/run_seed42 \
    --svd_dim 384 --seed 42 --use_xgb --use_gpu

Description:
- Each subtask result is saved in out_root subdirectories; final aggregation in out_root/summary.json
- Original model ablation includes:
  - base (recommended configuration)
  - no_exchange (disable exchange consistency)
  - diff_only (only use diff branch for features; here as baseline, reuse three-branch with linear/ML models but zero out mut/wt)
"""
import os
import sys
import json
import argparse
import subprocess
from typing import Dict


def safe_run(cmd: str):
    print("[RUN]", cmd)
    try:
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            print(f"[WARN] Command failed (code={proc.returncode}): {cmd}")
    except Exception as e:
        print(f"[WARN] Command exception: {e}\nCMD: {cmd}")


def read_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, 'r', encoding='utf-8'))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser("run all baselines and ablations")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--split_file", type=str, default=os.path.join("data", "processed", "siamese", "split_info.json"))
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--svd_dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_xgb", action='store_true')
    ap.add_argument("--use_gpu", action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # 1) Traditional ML baselines
    out_linear = os.path.join(args.out_root, 'linear')
    safe_run(f"python benchmarks/run_linear_baselines.py --data_dir {args.data_dir} --split_file {args.split_file} --out_dir {out_linear} --svd_dim {args.svd_dim} --seed {args.seed}")

    # SVR optional (skip when slow)
    out_svr = os.path.join(args.out_root, 'svr')
    safe_run(f"python benchmarks/run_svr_baselines.py --data_dir {args.data_dir} --split_file {args.split_file} --out_dir {out_svr} --svd_dim {args.svd_dim} --seed {args.seed} --subsample 0.3")

    out_tree = os.path.join(args.out_root, 'tree')
    tree_cmd = f"python benchmarks/run_tree_baselines.py --data_dir {args.data_dir} --split_file {args.split_file} --out_dir {out_tree} --svd_dim {args.svd_dim} --seed {args.seed}"
    if args.use_xgb:
        tree_cmd += " --use_xgb"
    safe_run(tree_cmd)

    out_mlp = os.path.join(args.out_root, 'mlp')
    safe_run(f"python benchmarks/run_mlp_baseline.py --data_dir {args.data_dir} --split_file {args.split_file} --out_dir {out_mlp} --svd_dim {args.svd_dim} --seed {args.seed}")

    out_kernel = os.path.join(args.out_root, 'kernel_knn')
    safe_run(f"python benchmarks/run_kernel_knn_baselines.py --data_dir {args.data_dir} --split_file {args.split_file} --out_dir {out_kernel} --svd_dim {args.svd_dim} --seed {args.seed}")

    # 2) Original model ablation (using same split_file, training for specified epochs)
    out_model = os.path.join(args.out_root, 'model')
    os.makedirs(out_model, exist_ok=True)
    dev_flag = "--use_gpu" if args.use_gpu else ""

    # base (150 epochs)
    safe_run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'base')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --split_file {args.split_file} {dev_flag}")

    # no_exchange (remove exchange consistency, 150 epochs)
    safe_run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'no_exchange')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --enable_multitask --svd_dim {args.svd_dim} --split_file {args.split_file} {dev_flag}")

    # diff_only (only diff branch, 150 epochs)
    safe_run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'diff_only')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --use_diff_only --split_file {args.split_file} {dev_flag}")

    # mut_only (150 epochs)
    safe_run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'mut_only')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --use_mut_only --split_file {args.split_file} {dev_flag}")

    # wt_only (150 epochs)
    safe_run(f"python dl_trainers/train_siamese_exchange.py --data_dir {args.data_dir} --output_dir {os.path.join(out_model,'wt_only')} --epochs 150 --batch_size 256 --lr 1e-3 --dropout 0.2 --exchange_consistency --enable_multitask --svd_dim {args.svd_dim} --use_wt_only --split_file {args.split_file} {dev_flag}")

    # Optional: diff_only (only use diff branch as features -> replace with fast linear/ML)
    # Here reuse linear baseline but don't change code in common, directly let mut/wt branches drop to 0 and process by SVD+Scaler which may distort, so simplify to reference linear results.
    # If strict diff-only version of siamese tower is needed, can extend training scripts later to support branch selection.

    # 3) Aggregate all results (models use val_metrics.json from seed directories)
    model_base = read_json(os.path.join(out_model, 'base', f'seed_{args.seed}', 'val_metrics.json'))
    model_noex = read_json(os.path.join(out_model, 'no_exchange', f'seed_{args.seed}', 'val_metrics.json'))
    model_diff = read_json(os.path.join(out_model, 'diff_only', f'seed_{args.seed}', 'val_metrics.json'))
    model_mut = read_json(os.path.join(out_model, 'mut_only', f'seed_{args.seed}', 'val_metrics.json'))
    model_wt = read_json(os.path.join(out_model, 'wt_only', f'seed_{args.seed}', 'val_metrics.json'))

    summary = {
        'linear': read_json(os.path.join(out_linear, 'val_metrics.json')),
        'svr': read_json(os.path.join(out_svr, 'val_metrics.json')),
        'tree': read_json(os.path.join(out_tree, 'val_metrics.json')),
        'mlp': read_json(os.path.join(out_mlp, 'val_metrics.json')),
        'kernel_knn': read_json(os.path.join(out_kernel, 'val_metrics.json')),
        'model': {
            'base': model_base,
            'no_exchange': model_noex,
            'diff_only': model_diff,
            'mut_only': model_mut,
            'wt_only': model_wt,
        }
    }
    with open(os.path.join(args.out_root, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(args.out_root, 'summary.json'))

    # 4) Generate brief report (success/failure)
    def ok(d):
        return isinstance(d, dict) and len(d) > 0
    report_lines = []
    for k in ['linear','svr','tree','mlp','kernel_knn']:
        report_lines.append(f"{k}: {'OK' if ok(summary.get(k)) else 'FAIL'}")
    for k in ['base','no_exchange','diff_only','mut_only','wt_only']:
        report_lines.append(f"model/{k}: {'OK' if ok(summary['model'].get(k)) else 'FAIL'}")
    with open(os.path.join(args.out_root, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


if __name__ == "__main__":
    main()
