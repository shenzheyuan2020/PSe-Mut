#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid/sweep runner for train_siamese_exchange.py with multiple option combinations.

Usage:
  python dl_trainers/run_siamese_exchange_sweep.py --data_dir siamese_dataset --base_out dl_results_sweep

It will run a set of reasonable configurations sequentially and write a summary table.
"""

import os
import json
import itertools
import subprocess
import argparse
from datetime import datetime


def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='siamese_dataset')
    ap.add_argument('--base_out', default='dl_results_sweep')
    ap.add_argument('--use_gpu', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.base_out, exist_ok=True)

    # second-stage fine sweep capped to 16 combos
    # Grid: cw(2) x svd(2) x multitask(2) x ranking(2) = 16
    losses = ['huber']      # fixed
    hubs = [1.8]            # fixed
    scheds = ['cosine']     # fixed
    drops = [0.2]           # fixed to best
    hiddens = ['1024,512']  # fixed to best
    exch_w = [0.08, 0.10]   # 2 levels
    svd_dims = [256, 384]   # 2 levels
    enable_multitask_opts = [False, True]  # 2 levels
    enable_ranking_opts = [False, True]    # 2 levels

    # fixed common settings
    seeds = 3
    patience = 30
    epochs = 150
    bs = 256

    runs = []
    for loss, delta, sch, dr, hd, cw, svd, mt, rk in itertools.product(
            losses, hubs, scheds, drops, hiddens, exch_w, svd_dims,
            enable_multitask_opts, enable_ranking_opts):
        tag = f"loss-{loss}_delta-{delta}_sch-{sch}_do-{dr}_hd-{hd.replace(',', '-')}_cons-{cw}_svd-{svd}_mt-{int(mt)}_rk-{int(rk)}"
        out_dir = os.path.join(args.base_out, tag)
        cmd = [
            'python', 'dl_trainers/train_siamese_exchange.py',
            '--data_dir', args.data_dir,
            '--output_dir', out_dir,
            '--num_seeds', str(seeds),
            '--epochs', str(epochs),
            '--batch_size', str(bs),
            '--dropout', str(dr),
            '--hidden', hd,
            '--loss', loss,
            '--huber_delta', str(delta),
            '--scheduler', sch,
            '--warmup_epochs', '8',
            '--exchange_consistency',
            '--consistency_weight', str(cw),
            '--svd_dim', str(svd),
            '--early_stopping_patience', str(patience),
        ]
        # label standardization on (regression)
        cmd += ['--label_standardize']
        if mt:
            cmd += ['--enable_multitask', '--cls_weight', '0.1']
        if rk:
            cmd += ['--enable_ranking', '--ranking_weight', '0.05', '--ranking_margin', '0.1']
        if args.use_gpu:
            cmd += ['--use_gpu']
        runs.append((tag, out_dir, cmd))

    summary_rows = []
    print(f"Planned runs: {len(runs)} (should be 16)")
    for tag, out_dir, cmd in runs:
        rc = run_cmd(cmd)
        recap = {"tag": tag, "out_dir": out_dir, "return_code": rc}
        # try read summary
        summ = os.path.join(out_dir, 'summary_metrics.json')
        if os.path.exists(summ):
            try:
                with open(summ, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ms = data.get('mean_std', {})
                def get_mean(d, k):
                    v = ms.get(k)
                    if isinstance(v, dict):
                        return v.get('mean')
                    # backward compat
                    if isinstance(v, (list, tuple)) and len(v) >= 1:
                        return v[0]
                    return None
                recap.update({
                    'rmse_mean': get_mean(ms, 'rmse'),
                    'mae_mean': get_mean(ms, 'mae'),
                    'r2_mean': get_mean(ms, 'r2'),
                    'pearson_mean': get_mean(ms, 'pearson'),
                })
                # also load per-seed best epochs
                per_seed = data.get('per_seed', {})
                recap['per_seed_best_epoch'] = {k: v.get('best_epoch') for k, v in per_seed.items()}
            except Exception as e:
                recap['error'] = str(e)
        # also copy last 10 lines of train.log for context
        tlog = os.path.join(out_dir, 'train.log')
        if os.path.exists(tlog):
            try:
                with open(tlog, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                recap['train_log_tail'] = ''.join(lines[-10:])
            except Exception:
                pass
        summary_rows.append(recap)

    # write sweep summary
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = os.path.join(args.base_out, f'sweep_summary_{stamp}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print('\nSaved sweep summary ->', out_json)


if __name__ == '__main__':
    main()


