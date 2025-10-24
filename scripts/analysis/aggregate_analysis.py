#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate dl_results_compare_mt/seed_*/analysis/analysis_summary.json and branch_ablation.csv,
Output:
  - CSV summary table (mean ± std)
  - LaTeX table (adapted for papers)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def load_seed_analysis(seed_dir: str) -> dict:
    ajs = os.path.join(seed_dir, 'analysis', 'analysis_summary.json')
    abl = os.path.join(seed_dir, 'analysis', 'branch_ablation.csv')
    if not os.path.exists(ajs):
        raise FileNotFoundError(f"Missing {ajs}")
    with open(ajs, 'r', encoding='utf-8') as f:
        summ = json.load(f)
    if os.path.exists(abl):
        df_ab = pd.read_csv(abl)
        summ['ablation_table'] = df_ab
    return summ


def mean_std_str(arr):
    arr = np.asarray(arr, dtype=float)
    return f"{np.nanmean(arr):.4f} \\pm {np.nanstd(arr):.4f}"


def main():
    p = argparse.ArgumentParser('Aggregate seed analyses to CSV + LaTeX table')
    p.add_argument('--root', type=str, default=os.path.join('outputs', 'dl', 'compare_mt'))
    p.add_argument('--out_prefix', type=str, default=os.path.join('outputs', 'figures', 'analysis_aggregate'))
    args = p.parse_args()

    # Backward-compatible path resolution: prefer organized outputs/ if legacy paths are missing
    if not os.path.exists(args.root):
        alt_root = os.path.join('outputs', 'dl', 'compare_mt')
        if os.path.exists(alt_root):
            args.root = alt_root

    # Resolve output prefix directory. If legacy 'figures/' is absent, write under 'outputs/figures/'
    out_dir = os.path.dirname(args.out_prefix) or '.'
    if not os.path.exists(out_dir):
        alt_fig_dir = os.path.join('outputs', 'figures')
        os.makedirs(alt_fig_dir, exist_ok=True)
        base_name = os.path.basename(args.out_prefix) or 'analysis_aggregate'
        args.out_prefix = os.path.join(alt_fig_dir, base_name)

    seed_dirs = []
    for name in sorted(os.listdir(args.root)):
        if name.startswith('seed_'):
            d = os.path.join(args.root, name)
            if os.path.isdir(d):
                seed_dirs.append(d)
    if not seed_dirs:
        raise RuntimeError('No seed_* directories found')

    rows = []
    head_auc, head_ap = [], []
    ex_pearson, ex_mae = [], []
    abl_head_mut, abl_head_wt, abl_head_diff = [], [], []
    abl_reg_mut, abl_reg_wt, abl_reg_diff = [], [], []

    for sd in seed_dirs:
        summ = load_seed_analysis(sd)
        head_auc.append(summ.get('head_auc', np.nan))
        head_ap.append(summ.get('head_ap', np.nan))
        ex_pearson.append(summ.get('exchange_pearson', np.nan))
        ex_mae.append(summ.get('exchange_mae_negflip', np.nan))
        ab = summ.get('ablation', {})
        hmap = ab.get('head_mean_abs_delta_prob', {})
        rmap = ab.get('reg_mean_abs_delta', {})
        abl_head_mut.append(hmap.get('mut', np.nan))
        abl_head_wt.append(hmap.get('wt', np.nan))
        abl_head_diff.append(hmap.get('diff', np.nan))
        abl_reg_mut.append(rmap.get('mut', np.nan))
        abl_reg_wt.append(rmap.get('wt', np.nan))
        abl_reg_diff.append(rmap.get('diff', np.nan))

    # 汇总表（CSV）
    agg = pd.DataFrame({
        'head_auc_mean': [np.nanmean(head_auc)], 'head_auc_std': [np.nanstd(head_auc)],
        'head_ap_mean': [np.nanmean(head_ap)], 'head_ap_std': [np.nanstd(head_ap)],
        'exchange_pearson_mean': [np.nanmean(ex_pearson)], 'exchange_pearson_std': [np.nanstd(ex_pearson)],
        'exchange_mae_mean': [np.nanmean(ex_mae)], 'exchange_mae_std': [np.nanstd(ex_mae)],
        'abl_head_mut_mean': [np.nanmean(abl_head_mut)], 'abl_head_mut_std': [np.nanstd(abl_head_mut)],
        'abl_head_wt_mean': [np.nanmean(abl_head_wt)], 'abl_head_wt_std': [np.nanstd(abl_head_wt)],
        'abl_head_diff_mean': [np.nanmean(abl_head_diff)], 'abl_head_diff_std': [np.nanstd(abl_head_diff)],
        'abl_reg_mut_mean': [np.nanmean(abl_reg_mut)], 'abl_reg_mut_std': [np.nanstd(abl_reg_mut)],
        'abl_reg_wt_mean': [np.nanmean(abl_reg_wt)], 'abl_reg_wt_std': [np.nanstd(abl_reg_wt)],
        'abl_reg_diff_mean': [np.nanmean(abl_reg_diff)], 'abl_reg_diff_std': [np.nanstd(abl_reg_diff)],
    })
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    # 覆盖旧文件
    for ext in ['_summary.csv', '_table.tex']:
        try:
            p = args.out_prefix + ext
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    agg.to_csv(args.out_prefix + '_summary.csv', index=False)

    # LaTeX 表格（更美观，蓝灰主题数值）
    latex = []
    latex.append('\\begin{table}[ht]')
    latex.append('\\centering')
    latex.append('\\small')
    latex.append('\\begin{tabular}{lcccc}')
    latex.append('\\toprule')
    latex.append('Metric & Mean $\\pm$ Std & & Metric & Mean $\\pm$ Std \\')
    latex.append('\\midrule')
    latex.append(f"Head ROC-AUC & {mean_std_str(head_auc)} & & Head PR-AP & {mean_std_str(head_ap)} \\")
    latex.append(f"Exchange Pearson & {mean_std_str(ex_pearson)} & & Exchange MAE (neg flip) & {mean_std_str(ex_mae)} \\")
    latex.append('\\midrule')
    latex.append(f"Head $|\\Delta p|$ (mut) & {mean_std_str(abl_head_mut)} & & Head $|\\Delta p|$ (wt) & {mean_std_str(abl_head_wt)} \\")
    latex.append(f"Head $|\\Delta p|$ (diff) & {mean_std_str(abl_head_diff)} & & Reg $|\\Delta y|$ (mut) & {mean_std_str(abl_reg_mut)} \\")
    latex.append(f"Reg $|\\Delta y|$ (wt) & {mean_std_str(abl_reg_wt)} & & Reg $|\\Delta y|$ (diff) & {mean_std_str(abl_reg_diff)} \\")
    latex.append('\\bottomrule')
    latex.append('\\caption{Directionality (head), exchange consistency and branch ablation (mean$\\pm$std across seeds).}')
    latex.append('\\label{tab:head_consistency_ablation}')
    latex.append('\\end{tabular}')
    latex.append('\\end{table}')

    with open(args.out_prefix + '_table.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex))
    print('Saved:', args.out_prefix + '_summary.csv', args.out_prefix + '_table.tex')


if __name__ == '__main__':
    main()


