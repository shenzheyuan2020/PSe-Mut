#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import pickle
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from matplotlib.ticker import FuncFormatter


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dl_trainers.train_siamese_exchange import (  # type: ignore
    SiameseRegressor, split_mut_wt_diff, load_dataset_arrays
)


def load_run_args(run_dir: str) -> Dict:
    with open(os.path.join(run_dir, "run_args.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_indices(split_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(split_file, "r", encoding="utf-8") as f:
        sp = json.load(f)
    tr = np.array(sp["train_idx"], dtype=int)
    va = np.array(sp["val_idx"], dtype=int)
    te = np.array(sp["test_idx"], dtype=int)
    return tr, va, te


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fit_branch_svd(m_tr: np.ndarray, w_tr: np.ndarray, d_tr: np.ndarray, target_dim: int, seed: int):
    svd_m = TruncatedSVD(n_components=target_dim, random_state=seed).fit(m_tr)
    svd_w = TruncatedSVD(n_components=target_dim, random_state=seed).fit(w_tr)
    svd_d = TruncatedSVD(n_components=target_dim, random_state=seed).fit(d_tr)
    return svd_m, svd_w, svd_d


def transform_branches(svd_m, svd_w, svd_d, m: np.ndarray, w: np.ndarray, d: np.ndarray):
    return svd_m.transform(m), svd_w.transform(w), svd_d.transform(d)


def build_model(in_dim: int, hidden: List[int], dropout: float, state_dict: Dict[str, torch.Tensor]) -> SiameseRegressor:
    has_cls = any(k.startswith("cls_head.") for k in state_dict.keys())
    fusion_dim = max(128, hidden[-1])
    model = SiameseRegressor(in_dim, hidden=hidden, fusion_dim=fusion_dim, dropout=dropout,
                             enable_multitask=has_cls).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_with_model(model: SiameseRegressor, scaler_m, scaler_w, scaler_d,
                       m: np.ndarray, w: np.ndarray, d: np.ndarray) -> np.ndarray:
    m_s = scaler_m.transform(m)
    w_s = scaler_w.transform(w)
    d_s = scaler_d.transform(d)
    device = next(model.parameters()).device
    mt = torch.from_numpy(m_s).float().to(device)
    wt = torch.from_numpy(w_s).float().to(device)
    dt = torch.from_numpy(d_s).float().to(device)
    with torch.no_grad():
        pred, _ = model(mt, wt, dt)
    return pred.squeeze(-1).detach().cpu().numpy().reshape(-1)


def mm_to_in(mm: float) -> float:
    return float(mm) / 25.4


def paper_style(font_family: str = "Arial", base_font_size: int = 8):
    sns.set_theme(context="paper", style="white")
    sns.set_style("white", {
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": False,
    })
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": base_font_size,
        "axes.titlesize": base_font_size,
        "axes.labelsize": max(6, base_font_size - 1),
        "legend.fontsize": max(6, base_font_size - 1),
        "xtick.labelsize": max(6, base_font_size - 2),
        "ytick.labelsize": max(6, base_font_size - 2),
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


def _blue_theme_cmap():
    # 更平滑的蓝色渐变：增加中间过渡色，避免突兀
    return LinearSegmentedColormap.from_list(
        'blue_theme', [
            '#e6f0fb', '#deebf7', '#d4e5f4', '#c6dbef', '#b3d0ea', '#9ecae1',
            '#82b9da', '#6baed6', '#4f9dcb', '#3182bd', '#1767ab', '#08519c'
        ], N=256
    )


def _green_theme_cmap():
    return LinearSegmentedColormap.from_list(
        'green_theme', ['#f7fcf5', '#e2f0d9', '#ccebc5', '#a1d99b', '#74c476', '#31a354', '#006d2c'], N=256
    )


def _theme_cmap(theme: str) -> LinearSegmentedColormap:
    if (theme or '').lower().strip() == 'green':
        return _green_theme_cmap()
    if (theme or '').lower().strip() == 'bluegreen':
        # 蓝→青→绿混合渐变（顶端更偏绿）
        return LinearSegmentedColormap.from_list(
            'blue_green_mix', [
                '#e6f0fb', '#deebf7', '#cfe7f3', '#bfe0ee', '#a7d4e7',
                '#8ec8df', '#6bb9d6', '#3aa6b8', '#2fc0a8', '#2dc67d'
            ], N=256
        )
    return _blue_theme_cmap()


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_png: str, out_svg: str,
                 theme: str = 'blue', fig_w_mm: float = 85.0, fig_h_mm: float = 70.0,
                 font_family: str = "Arial", base_font_size: int = 8,
                 color_p_lo: float = 20.0, color_p_hi: float = 98.0):
    paper_style(font_family=font_family, base_font_size=base_font_size)
    fig, ax = plt.subplots(figsize=(mm_to_in(fig_w_mm), mm_to_in(fig_h_mm)))

    vmin = float(min(np.min(y_true), np.min(y_pred)))
    vmax = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    lo, hi = vmin - pad, vmax + pad

    cmap = _theme_cmap(theme)
    # 线性 + 全范围：与坐标轴一致，不做裁剪与非线性
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 轻量背景纹理
    try:
        ax.hexbin(y_true, y_pred, gridsize=45, cmap='Greys', alpha=0.05, linewidths=0.0, mincnt=1, zorder=0)
    except Exception:
        pass

    ax.scatter(y_true, y_pred, s=15, c=y_true, cmap=cmap, norm=norm, alpha=0.80, edgecolors="none", zorder=2)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, c="#a8b2bf", zorder=1)

    try:
        coeffs = np.polyfit(y_true, y_pred, 1)
        xs = np.linspace(lo, hi, 100)
        ys = coeffs[0] * xs + coeffs[1]
        ax.plot(xs, ys, c=("#238b45" if theme == 'green' else "#2171b5"), lw=1.0, alpha=0.9, zorder=3)
    except Exception:
        pass

    r, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0.0, None)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    ax.set_title(title)
    # 统一刻度为一位小数
    fmt = FuncFormatter(lambda v, _: f"{v:.1f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    sns.despine(ax=ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.07)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('True Δy', rotation=90)
    # 颜色条刻度格式化为一位小数
    try:
        fmt_cb = FuncFormatter(lambda v, _: f"{v:.1f}")
        cb.ax.yaxis.set_major_formatter(fmt_cb)
        cb.ax.tick_params(labelsize=max(6, base_font_size - 2))
        try:
            cb.ax.yaxis.label.set_size(max(6, base_font_size - 1))
        except Exception:
            pass
    except Exception:
        pass

    # 底部信息面板：加大间距，文字两行排列并右移，避免遮挡
    bax = divider.append_axes("bottom", size="15%", pad=0.36)
    bax.axis('off')
    patch = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, transform=bax.transAxes,
                           boxstyle="round,pad=0.25", linewidth=0.6,
                           edgecolor="#e6e8eb", facecolor="#ffffff")
    bax.add_patch(patch)
    line1 = f"Pearson r = {r:.3f}    R$^2$ = {r2:.3f}"
    line2 = f"RMSE = {rmse:.3f}    MAE = {mae:.3f}"
    info_fs = max(6, base_font_size - 1)
    bax.text(0.12, 0.66, line1, va='center', ha='left', fontsize=info_fs)
    bax.text(0.12, 0.34, line2, va='center', ha='left', fontsize=info_fs)

    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_png: str, out_svg: str,
                   theme: str = 'blue', fig_w_mm: float = 85.0, fig_h_mm: float = 70.0,
                   font_family: str = "Arial", base_font_size: int = 10):
    paper_style(font_family=font_family, base_font_size=base_font_size)
    fig, ax = plt.subplots(figsize=(mm_to_in(fig_w_mm), mm_to_in(fig_h_mm)))

    residuals = y_pred - y_true
    vmin = float(min(np.min(y_pred), np.min(y_true)))
    vmax = float(max(np.max(y_pred), np.max(y_true)))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    lo, hi = vmin - pad, vmax + pad

    cmap = _theme_cmap(theme)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    ax.scatter(y_pred, residuals, s=14, c=y_true, cmap=cmap, norm=norm, alpha=0.65, edgecolors="none")
    ax.axhline(0.0, ls='--', lw=1.0, c="#7a869a")

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    ax.set_xlabel('Pred')
    ax.set_ylabel('Residual (Pred - True)')
    ax.set_title(title + ' — Residuals')
    sns.despine(ax=ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('True Δy', rotation=90)

    bax = divider.append_axes("bottom", size="18%", pad=0.40)
    bax.axis('off')
    patch = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, transform=bax.transAxes,
                           boxstyle="round,pad=0.3", linewidth=0.8,
                           edgecolor="#e5e7eb", facecolor="#ffffff")
    bax.add_patch(patch)
    line1 = f"R$^2$ = {r2:.3f}"
    line2 = f"RMSE = {rmse:.3f}    MAE = {mae:.3f}"
    bax.text(0.12, 0.66, line1, va='center', ha='left', fontsize=8)
    bax.text(0.12, 0.34, line2, va='center', ha='left', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_svg)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("plot regression scatter for one seed (pretty)")
    ap.add_argument('--run_dir', type=str, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--data_dir', type=str, default=os.path.join('data', 'processed', 'siamese'))
    args = ap.parse_args()

    run_args = load_run_args(args.run_dir)
    seed_dir = os.path.join(args.run_dir, f'seed_{args.seed}')
    if not os.path.exists(seed_dir):
        raise FileNotFoundError(seed_dir)

    X, df = load_dataset_arrays(args.data_dir)
    labels = df["label_diff"].values.astype(np.float32)
    split_file = os.path.normpath(run_args.get('split_file') or os.path.join(args.data_dir, 'split_info.json'))
    tr_idx, va_idx, te_idx = load_split_indices(split_file)

    mut, wt, diff = split_mut_wt_diff(X)
    svd_dim = int(run_args.get('svd_dim', 384))
    svd_m, svd_w, svd_d = fit_branch_svd(mut[tr_idx], wt[tr_idx], diff[tr_idx], svd_dim, seed=args.seed)
    m_va, w_va, d_va = transform_branches(svd_m, svd_w, svd_d, mut[va_idx], wt[va_idx], diff[va_idx])
    m_te, w_te, d_te = transform_branches(svd_m, svd_w, svd_d, mut[te_idx], wt[te_idx], diff[te_idx])

    with open(os.path.join(seed_dir, 'scaler_m.pkl'), 'rb') as f:
        scaler_m = pickle.load(f)
    with open(os.path.join(seed_dir, 'scaler_w.pkl'), 'rb') as f:
        scaler_w = pickle.load(f)
    with open(os.path.join(seed_dir, 'scaler_d.pkl'), 'rb') as f:
        scaler_d = pickle.load(f)

    state_dict = torch.load(os.path.join(seed_dir, 'model_siamese.pth'), map_location='cpu')
    hidden = [int(x) for x in str(run_args.get('hidden', '1024,512')).split(',') if str(x).strip()]
    dropout = float(run_args.get('dropout', 0.2))
    model = build_model(svd_dim, hidden, dropout, state_dict)

    y_va_true = labels[va_idx]
    y_te_true = labels[te_idx]
    y_va_pred = predict_with_model(model, scaler_m, scaler_w, scaler_d, m_va, w_va, d_va)
    y_te_pred = predict_with_model(model, scaler_m, scaler_w, scaler_d, m_te, w_te, d_te)

    out_dir = os.path.join(seed_dir, 'plots')
    ensure_dir(out_dir)

    pd.DataFrame({'y_true': y_va_true, 'y_pred': y_va_pred}).to_csv(os.path.join(out_dir, f'val_predictions_seed{args.seed}.csv'), index=False)
    pd.DataFrame({'y_true': y_te_true, 'y_pred': y_te_pred}).to_csv(os.path.join(out_dir, f'test_predictions_seed{args.seed}.csv'), index=False)

    plot_scatter(y_va_true, y_va_pred, 'Validation',
                 os.path.join(out_dir, f'val_pred_vs_true_seed{args.seed}.png'),
                 os.path.join(out_dir, f'val_pred_vs_true_seed{args.seed}.svg'),
                 theme='bluegreen')
    plot_scatter(y_te_true, y_te_pred, 'Test',
                 os.path.join(out_dir, f'test_pred_vs_true_seed{args.seed}.png'),
                 os.path.join(out_dir, f'test_pred_vs_true_seed{args.seed}.svg'),
                 theme='bluegreen')

    print(f"Saved plots and CSV to: {out_dir}")


if __name__ == '__main__':
    main()


