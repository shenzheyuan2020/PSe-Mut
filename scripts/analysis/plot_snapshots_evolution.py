#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练期保存的 snapshots 目录生成“epoch × 层”的演化面板图。

输入目录结构（由训练脚本自动产生）：
  snapshots/
    epoch_001_layer1_diff.npy
    epoch_001_layermid_diff.npy
    epoch_001_layerlast_diff.npy
    epoch_001_head_hidden.npy        (可选)
    epoch_001_head_logits.npy        (可选)
    epoch_001_meta.csv               (含 label_bin/uniprot_id)
    ... epoch_XXX_*.npy / .csv

输出：
  - 面板图 PNG/PDF（蓝-灰风格，带透明）
  - 每个 epoch 的 AUC 列表 CSV（如存在 head_logits）

用法示例：
  python tools/plot_snapshots_evolution.py \
    --snap_dir dl_results_compare_mt/seed_42/analysis/snapshots \
    --out figures/snapshots_evolution_seed42 \
    --max_epochs 6 --method umap
"""

import os
import re
import argparse
from glob import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import roc_auc_score


def find_epochs(snap_dir: str) -> List[int]:
    metas = sorted(glob(os.path.join(snap_dir, 'epoch_*_meta.csv')))
    eps = []
    for p in metas:
        m = re.search(r'epoch_(\d+)_meta\.csv$', p)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def even_pick(values: List[int], k: int) -> List[int]:
    if k <= 0 or k >= len(values):
        return values
    idxs = np.linspace(0, len(values) - 1, num=k, dtype=int)
    return [values[i] for i in idxs]


def reduce_2d(X: np.ndarray, method: str, seed: int, n_neighbors: int, min_dist: float, perplexity: int) -> np.ndarray:
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2,
                            metric='euclidean', random_state=seed)
        return reducer.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=seed, n_iter=1000)
    return tsne.fit_transform(X)


def plot_panel(snap_dir: str, epochs: List[int], layers: List[str], method: str,
               out_prefix: str, seed: int, n_neighbors: int, min_dist: float,
               perplexity: int, max_points: int, branch: str) -> None:
    rows = len(epochs)
    cols = len(layers)
    # High-quality paper style
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.facecolor": "#ffffff",
    })
    plt.figure(figsize=(cols * 4.2, rows * 3.4))
    auc_rows = []

    for r, ep in enumerate(epochs):
        meta_path = os.path.join(snap_dir, f'epoch_{ep:03d}_meta.csv')
        meta = pd.read_csv(meta_path)
        # Label binary
        if 'label_bin' in meta.columns:
            y = meta['label_bin'].astype(int).values
        elif 'label_diff' in meta.columns:
            y = (meta['label_diff'].astype(float).values >= 0).astype(int)
        else:
            y = None

        # AUC (if head_logits exists)
        auc_val = None
        log_path = os.path.join(snap_dir, f'epoch_{ep:03d}_head_logits.npy')
        if os.path.exists(log_path) and y is not None:
            logits = np.load(log_path)
            try:
                probs = 1.0 / (1.0 + np.exp(-logits))
                auc_val = float(roc_auc_score(y, probs))
            except Exception:
                auc_val = None
        auc_rows.append({'epoch': ep, 'head_auc': auc_val})

        for c, layer in enumerate(layers):
            ax = plt.subplot(rows, cols, r * cols + c + 1)
            npy_path = os.path.join(snap_dir, f'epoch_{ep:03d}_{layer}_{branch}.npy')
            if not os.path.exists(npy_path):
                ax.text(0.5, 0.5, f'missing {layer}', ha='center', va='center')
                ax.axis('off')
                continue
            X = np.load(npy_path)
            # Align X and y lengths to avoid out of bounds
            n_x = X.shape[0]
            n_y = len(y) if y is not None else n_x
            n = min(n_x, n_y)
            X = X[:n]
            y_clip = y[:n] if y is not None else None
            if max_points > 0 and n > max_points:
                sel = np.random.RandomState(seed).choice(n, size=max_points, replace=False)
                X = X[sel]
                y_plot = y_clip[sel] if y_clip is not None else None
            else:
                y_plot = y_clip
            try:
                Z = reduce_2d(X, method, seed, n_neighbors, min_dist, perplexity)
            except Exception:
                Z = X[:, :2]
            # Blue-gray color scheme
            if y_plot is None:
                ax.scatter(Z[:, 0], Z[:, 1], s=7, alpha=0.7, c="#1f77b4")
            else:
                pos = (y_plot == 1)
                ax.scatter(Z[~pos, 0], Z[~pos, 1], s=6, alpha=0.5, c="#c7ccd4", label='neg')
                ax.scatter(Z[pos, 0], Z[pos, 1], s=7, alpha=0.8, c="#1f77b4", label='pos')
            ax.set_xticks([]); ax.set_yticks([])
            title = f"ep {ep} | {layer}"
            if c == 0 and auc_val is not None:
                title += f" | AUC {auc_val:.3f}"
            ax.set_title(title)
            if r == 0 and c == cols - 1 and y_plot is not None:
                leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    # Overwrite old images (delete if they exist)
    for ext in ['.png', '.pdf']:
        try:
            p = out_prefix + ext
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    plt.savefig(out_prefix + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(out_prefix + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # Save AUC for each epoch
    try:
        pd.DataFrame(auc_rows).to_csv(out_prefix + '_epoch_auc.csv', index=False)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser('Plot epoch×layer evolution from snapshots')
    p.add_argument('--snap_dir', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--layers', type=str, default='layer1,layermid,layerlast')
    p.add_argument('--max_epochs', type=int, default=6)
    p.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--umap_neighbors', type=int, default=15)
    p.add_argument('--umap_min_dist', type=float, default=0.1)
    p.add_argument('--tsne_perplexity', type=int, default=30)
    p.add_argument('--max_points', type=int, default=2000)
    p.add_argument('--branch', type=str, default='diff', choices=['mut','wt','diff'])
    args = p.parse_args()

    epochs = find_epochs(args.snap_dir)
    if not epochs:
        raise RuntimeError('No epoch_*_meta.csv found')
    use_epochs = even_pick(epochs, args.max_epochs) if args.max_epochs > 0 else epochs
    layers = [s.strip() for s in args.layers.split(',') if s.strip()]
    # Auto-detect branch (avoid missing layer1)
    chosen_branch = args.branch
    first_ep = use_epochs[0]
    if not os.path.exists(os.path.join(args.snap_dir, f'epoch_{first_ep:03d}_{layers[0]}_{chosen_branch}.npy')):
        for b in ['diff', 'mut', 'wt']:
            if os.path.exists(os.path.join(args.snap_dir, f'epoch_{first_ep:03d}_{layers[0]}_{b}.npy')):
                chosen_branch = b
                break
    plot_panel(args.snap_dir, use_epochs, layers, args.method, args.out, args.seed,
               args.umap_neighbors, args.umap_min_dist, args.tsne_perplexity, args.max_points, chosen_branch)


if __name__ == '__main__':
    main()


