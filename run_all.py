#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click run all:
1) Generate/reuse random split (non-grouped)
2) Run all baselines and ablations (150 epochs) for three random seeds (42/123/456) sequentially
3) Generate summary.json for each seed and summary_all_seeds.json across seeds

Usage (recommended to run directly without parameters):
  python run_all.py

Optional parameters:
  python run_all.py --data_dir siamese_dataset --out_root bench_results_all \
    --svd_dim 384 --seeds 42,123,456 --val_ratio 0.2 --split_seed 42 --use_xgb --use_gpu
"""
import os
import sys
import json
import argparse
import subprocess


def safe_run(cmd: str):
    print("[RUN]", cmd)
    try:
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            print(f"[WARN] Command failed (code={proc.returncode}): {cmd}")
    except Exception as e:
        print(f"[WARN] Command exception: {e}\nCMD: {cmd}")


def read_json(path: str):
    try:
        if os.path.exists(path):
            return json.load(open(path, 'r', encoding='utf-8'))
    except Exception:
        pass
    return {}


def main():
    ap = argparse.ArgumentParser("one-click all seeds baselines & ablations")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--out_root", type=str, default=os.path.join("outputs", "benchmarks", "all"))
    ap.add_argument("--svd_dim", type=int, default=384)
    ap.add_argument("--seeds", type=str, default="42,123,456")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--split_mode", type=str, default="random", choices=["random"])  # Fixed to use non-grouped
    ap.add_argument("--use_xgb", action='store_true')
    # Auto-detect GPU; also allow manual override
    try:
        import torch  # noqa: F401
        gpu_default = True
    except Exception:
        gpu_default = False
    ap.add_argument("--use_gpu", action='store_true', default=gpu_default)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # 1) Generate/reuse fixed random split (non-grouped)
    split_file = os.path.join(args.data_dir, 'split_info.json')
    if not os.path.exists(split_file):
        safe_run(
            f"python tools/freeze_splits.py --data_dir {args.data_dir} --mode {args.split_mode} --val_ratio {args.val_ratio} --seed {args.split_seed}"
        )
    else:
        print(f"[INFO] Reuse existing split_file: {split_file}")

    # 2) Run full suite for each seed
    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    per_seed = {}
    for s in seeds:
        out_run = os.path.join(args.out_root, f"run_seed{s}")
        os.makedirs(out_run, exist_ok=True)
        cmd = (
            f"python benchmarks/run_all_baselines.py "
            f"--data_dir {args.data_dir} "
            f"--split_file {split_file} "
            f"--out_root {out_run} "
            f"--svd_dim {args.svd_dim} "
            f"--seed {s} "
            f"{'--use_xgb' if args.use_xgb else ''} "
            f"{'--use_gpu' if args.use_gpu else ''}"
        )
        safe_run(cmd)
        per_seed[s] = read_json(os.path.join(out_run, 'summary.json'))

    # 3) Aggregate across seeds
    summary_all = {
        'seeds': seeds,
        'per_seed': per_seed,
    }
    out_summary = os.path.join(args.out_root, 'summary_all_seeds.json')
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print('Saved:', out_summary)


if __name__ == "__main__":
    main()




