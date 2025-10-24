#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顺序执行实验流水：

实验1：
- 生成/确认三划分 split（group3: val=0.2, test=0.2, seed=42）
- 对每个种子（默认 42,123,456）顺序运行 Baseline：linear/svr/tree(xgb+fast)/mlp/kernel_knn
- 依次运行孪生模型(base, no_exchange, diff_only, mut_only, wt_only)，三划分(train/val/test)

实验2：
- 全部完成后，运行主模型（含多任务分类头）25 个随机种子（2000-2024），三划分；
  保存权重与 scaler 以便后续推理（支持回归与分类概率）

注意：严格顺序执行，不并行。
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import List


def run(cmd: List[str], env=None, cwd=None):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def ensure_three_split(data_dir: str, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42, env=None):
    split_file = os.path.join(data_dir, 'split_info.json')
    need_generate = True
    if os.path.exists(split_file):
        try:
            sp = json.load(open(split_file, 'r', encoding='utf-8'))
            need_generate = ('test_idx' not in sp) or (sp.get('test_idx') is None)
        except Exception:
            need_generate = True
    if need_generate:
        print("[INFO] 生成 random3 三划分 split_info.json …")
        run([sys.executable, os.path.join('scripts','analysis','freeze_splits.py'),
             '--data_dir', data_dir,
             '--mode', 'random3',
             '--val_ratio', str(val_ratio),
             '--test_ratio', str(test_ratio),
             '--seed', str(seed)], env=env)
    else:
        print("[INFO] 已存在含 test_idx 的 split_info.json，跳过生成。")
    return split_file


def run_baselines_seq(data_dir: str, split_file: str, out_root: str, seed: int, env=None):
    os.makedirs(out_root, exist_ok=True)
    # linear
    run([sys.executable, os.path.join('scripts','benchmarks','run_linear_baselines.py'),
         '--data_dir', data_dir, '--split_file', split_file, '--out_dir', os.path.join(out_root, 'linear'),
         '--svd_dim', '384', '--seed', str(seed)], env=env)
    # svr
    run([sys.executable, os.path.join('scripts','benchmarks','run_svr_baselines.py'),
         '--data_dir', data_dir, '--split_file', split_file, '--out_dir', os.path.join(out_root, 'svr'),
         '--svd_dim', '384', '--seed', str(seed), '--subsample', '0.3'], env=env)
    # tree（包含 XGBoost 且 fast）
    run([sys.executable, os.path.join('scripts','benchmarks','run_tree_baselines.py'),
         '--data_dir', data_dir, '--split_file', split_file, '--out_dir', os.path.join(out_root, 'tree'),
        '--svd_dim', '384', '--seed', str(seed), '--fast'], env=env)
    # mlp
    run([sys.executable, os.path.join('scripts','benchmarks','run_mlp_baseline.py'),
         '--data_dir', data_dir, '--split_file', split_file, '--out_dir', os.path.join(out_root, 'mlp'),
         '--svd_dim', '384', '--seed', str(seed)], env=env)
    # kernel & knn
    run([sys.executable, os.path.join('scripts','benchmarks','run_kernel_knn_baselines.py'),
         '--data_dir', data_dir, '--split_file', split_file, '--out_dir', os.path.join(out_root, 'kernel_knn'),
         '--svd_dim', '384', '--seed', str(seed)], env=env)


def run_siamese_three_split(out_dir: str, data_dir: str, split_file: str, seeds: List[int], mode: str, env=None,
                            exchange: bool = False, diff_only: bool = False, mut_only: bool = False, wt_only: bool = False,
                            enable_multitask: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join('dl_trainers','train_siamese_exchange.py'),
           '--data_dir', data_dir,
           '--output_dir', out_dir,
           '--use_three_split', '--split_file', split_file,
           '--label_standardize', '--svd_dim', '384', '--epochs', '150', '--batch_size', '256', '--lr', '1e-3']
    if exchange:
        cmd.append('--exchange_consistency')
    if diff_only:
        cmd.append('--use_diff_only')
    if mut_only:
        cmd.append('--use_mut_only')
    if wt_only:
        cmd.append('--use_wt_only')
    if enable_multitask:
        cmd.append('--enable_multitask')
    # 传入自定义 seeds 列表（训练脚本支持）
    cmd += ['--seeds', ','.join(str(s) for s in seeds)]
    # 如可用，启用 GPU
    cmd.append('--use_gpu')
    print(f"[INFO] 运行孪生模型 {mode}，seeds={seeds}")
    run(cmd, env=env)


def main():
    ap = argparse.ArgumentParser("sequential experiments runner")
    ap.add_argument('--data_dir', type=str, default=os.path.join('data','processed','siamese'))
    ap.add_argument('--exp1_seeds', type=str, default='42,123,456', help='实验1三种子')
    ap.add_argument('--exp2_seed_start', type=int, default=2000)
    ap.add_argument('--exp2_seed_count', type=int, default=25)
    ap.add_argument('--out_root', type=str, default=os.path.join('outputs','experiments'))
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env = os.environ.copy()
    # 确保脚本子进程可导入项目内模块
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env and env['PYTHONPATH'] else '')

    os.makedirs(args.out_root, exist_ok=True)

    # 准备 split（三划分）
    split_file = ensure_three_split(args.data_dir, env=env)

    # 实验1：Baselines + 我们模型（含消融），三划分，三个种子
    exp1_dir = os.path.join(args.out_root, 'exp1_three_split')
    os.makedirs(exp1_dir, exist_ok=True)
    exp1_seeds = [int(x.strip()) for x in args.exp1_seeds.split(',') if x.strip()]

    # Baselines（逐 seed 顺序跑）
    for sd in exp1_seeds:
        print(f"\n===== 实验1 Baselines: seed {sd} =====")
        out_b = os.path.join(exp1_dir, f'baselines_seed_{sd}')
        run_baselines_seq(args.data_dir, split_file, out_b, seed=sd, env=env)

    # 我们模型（顺序执行 base -> no_exchange -> diff_only -> mut_only -> wt_only）
    print("\n===== 实验1 Siamese（三划分）=====")
    run_siamese_three_split(out_dir=os.path.join(exp1_dir, 'model_base'), data_dir=args.data_dir, split_file=split_file,
                            seeds=exp1_seeds, mode='base', env=env, exchange=True, enable_multitask=True)
    run_siamese_three_split(out_dir=os.path.join(exp1_dir, 'model_no_exchange'), data_dir=args.data_dir, split_file=split_file,
                            seeds=exp1_seeds, mode='no_exchange', env=env, exchange=False, enable_multitask=True)
    run_siamese_three_split(out_dir=os.path.join(exp1_dir, 'model_diff_only'), data_dir=args.data_dir, split_file=split_file,
                            seeds=exp1_seeds, mode='diff_only', env=env, exchange=True, diff_only=True, enable_multitask=True)
    run_siamese_three_split(out_dir=os.path.join(exp1_dir, 'model_mut_only'), data_dir=args.data_dir, split_file=split_file,
                            seeds=exp1_seeds, mode='mut_only', env=env, exchange=True, mut_only=True, enable_multitask=True)
    run_siamese_three_split(out_dir=os.path.join(exp1_dir, 'model_wt_only'), data_dir=args.data_dir, split_file=split_file,
                            seeds=exp1_seeds, mode='wt_only', env=env, exchange=True, wt_only=True, enable_multitask=True)

    # 实验2：主模型（只关注 AUC，保存可用权重）
    print("\n===== 实验2 主模型 25 种子（2000-2024）=====")
    s0 = int(args.exp2_seed_start)
    n = int(args.exp2_seed_count)
    exp2_seeds = list(range(s0, s0 + n))
    exp2_dir = os.path.join(args.out_root, f'exp2_main_{s0}_{s0 + n - 1}')
    run_siamese_three_split(out_dir=exp2_dir, data_dir=args.data_dir, split_file=split_file,
                            seeds=exp2_seeds, mode='main_auc', env=env, exchange=True, enable_multitask=True)

    print("\n全部实验完成。产物目录：")
    print(" - 实验1:", exp1_dir)
    print(" - 实验2:", exp2_dir)


if __name__ == '__main__':
    main()


