#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal UMAP visualization for pre-input vs learned z on Siamese Psemut.

Inputs:
  - data_dir: contains siamese_pairs.csv, siamese_fingerprints.npy
  - weights_root/seed_*: contains model_siamese.pth, scaler_*.pkl, optional svd_*.pkl, split_info.json

Outputs (PNG+SVG):
  out_dir/seed_<seed>/{pre_umap,post_umap,compare_pre_vs_post_umap}.{png,svg}
  out_dir/seed_<seed>/points_pre_umap.csv, points_post_umap.csv
"""
import os
import sys
import json
import argparse
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except Exception:
    from sklearn.decomposition import PCA
    HAS_UMAP = False

from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn


def load_json(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def try_load_pkl(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def branch_split(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert X.shape[1] % 3 == 0, 'Expected 3-branch concatenation'
    d = X.shape[1] // 3
    return X[:, :d], X[:, d:2*d], X[:, 2*d:3*d]


def apply_svd_scaler(arr: np.ndarray, svd, scaler) -> np.ndarray:
    X = arr
    if svd is not None:
        X = svd.transform(X)
    if scaler is not None:
        # If scaler feature dim mismatches due to version drift, adapt via SVD
        exp = getattr(scaler, 'n_features_in_', None)
        if exp is not None and X.shape[1] != exp:
            k = int(exp)
            k = max(1, min(k, X.shape[1]))
            X = TruncatedSVD(n_components=k, random_state=42).fit_transform(X)
        X = scaler.transform(X)
    return X


def reduce_umap(Z: np.ndarray, y: Optional[np.ndarray] = None, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=30, min_dist=0.15, metric='euclidean')
        if y is not None:
            try:
                return reducer.fit_transform(Z, y=y.reshape(-1, 1))
            except Exception:
                return reducer.fit_transform(Z)
        return reducer.fit_transform(Z)
    # Fallback to PCA
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=random_state).fit_transform(Z)


def save_scatter_2d(X2d: np.ndarray, color: np.ndarray, title: str, out_path: str, cmap: str = 'coolwarm') -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=color, cmap=cmap, s=6, alpha=0.8, linewidths=0)
    plt.colorbar(sc, label='label_diff')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    try:
        plt.savefig(os.path.splitext(out_path)[0] + '.svg', format='svg')
    except Exception:
        pass
    plt.close()


def main():
    ap = argparse.ArgumentParser('UMAP visualization (test-only optional)')
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--weights_root', type=str, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str, default=os.path.join('outputs', 'figures', 'embeddings_umap'))
    ap.add_argument('--subset', type=str, default='all', choices=['all', 'train', 'val', 'test'])
    ap.add_argument('--branch', type=str, default='concat', choices=['diff', 'concat'])
    ap.add_argument('--space', type=str, default='input', choices=['input', 'raw'])
    args = ap.parse_args()

    # Ensure project root on sys.path so that dl_trainers can be imported
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    seed_dir = os.path.join(args.weights_root, f'seed_{args.seed}')
    # Robustly read run_args.json (allow UTF-8 BOM)
    run_args = {}
    rp = os.path.join(seed_dir, 'run_args.json')
    if os.path.exists(rp):
        try:
            import codecs
            run_args = json.load(codecs.open(rp, 'r', 'utf-8-sig'))
        except Exception:
            run_args = load_json(rp)
    if not run_args:
        run_args = load_json(os.path.join(args.weights_root, 'run_args.json'))
    hidden = [int(x) for x in str(run_args.get('hidden', '1024,512')).split(',')]
    dropout = float(run_args.get('dropout', 0.2))

    # Load data
    pairs_csv = os.path.join(args.data_dir, 'siamese_pairs.csv')
    npy_path = os.path.join(args.data_dir, 'siamese_fingerprints.npy')
    df = pd.read_csv(pairs_csv)
    X = np.load(npy_path)
    y = df['label_diff'].values.astype(np.float32)

    # Subset based on seed-specific split if requested
    if args.subset != 'all':
        sp = load_json(os.path.join(seed_dir, 'split_info.json'))
        sel = []
        if args.subset == 'train':
            sel = sp.get('train_idx', [])
        elif args.subset == 'val':
            sel = sp.get('val_idx', [])
        else:
            sel = sp.get('test_idx', [])
        sel = np.array(sel, dtype=int)
        if sel.size > 0:
            mask = np.zeros(len(df), dtype=bool)
            mask[sel[(sel >= 0) & (sel < len(df))]] = True
            X = X[mask]
            y = y[mask]
            df = df.iloc[mask].reset_index(drop=True)

    # Branch split
    m_raw, w_raw, d_raw = branch_split(X)

    # Load scalers/SVDs
    scaler_m = try_load_pkl(os.path.join(seed_dir, 'scaler_m.pkl'))
    scaler_w = try_load_pkl(os.path.join(seed_dir, 'scaler_w.pkl'))
    scaler_d = try_load_pkl(os.path.join(seed_dir, 'scaler_d.pkl'))
    svd_m = try_load_pkl(os.path.join(seed_dir, 'svd_m.pkl'))
    svd_w = try_load_pkl(os.path.join(seed_dir, 'svd_w.pkl'))
    svd_d = try_load_pkl(os.path.join(seed_dir, 'svd_d.pkl'))

    # If SVDs are missing, attempt to fit from train indices for alignment
    if svd_m is None or svd_w is None or svd_d is None:
        sp = load_json(os.path.join(seed_dir, 'split_info.json'))
        tr = np.array(sp.get('train_idx', []), dtype=int)
        if tr.size >= 10:
            k = int(run_args.get('svd_dim', 384))
            tr = tr[(tr >= 0) & (tr < len(df))]
            if tr.size >= 10:
                svd_m = TruncatedSVD(n_components=min(k, m_raw.shape[1]-1), random_state=args.seed).fit(m_raw[tr])
                svd_w = TruncatedSVD(n_components=min(k, w_raw.shape[1]-1), random_state=args.seed).fit(w_raw[tr])
                svd_d = TruncatedSVD(n_components=min(k, d_raw.shape[1]-1), random_state=args.seed).fit(d_raw[tr])

    # Prepare pre-space input as the model sees
    if args.space == 'input':
        m_in = apply_svd_scaler(m_raw, svd_m, scaler_m)
        w_in = apply_svd_scaler(w_raw, svd_w, scaler_w)
        d_in = apply_svd_scaler(d_raw, svd_d, scaler_d)
    else:
        m_in, w_in, d_in = m_raw, w_raw, d_raw

    if args.branch == 'diff':
        pre_in = d_in
    else:
        pre_in = np.concatenate([m_in, w_in, d_in], axis=1)

    # Build model and compute z = concat(enc(mut), enc(wt), enc(diff))
    from dl_trainers.train_siamese_exchange import Tower, SiameseRegressor  # re-use definitions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = m_in.shape[1]
    fusion_dim = max(128, hidden[-1])
    model = SiameseRegressor(in_dim, hidden=hidden, fusion_dim=fusion_dim, dropout=dropout, enable_multitask=False).to(device)
    state = None
    model_path = os.path.join(seed_dir, 'model_siamese.pth')
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu')
    if state is not None:
        model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        bs = 2048
        Z = []
        for i in range(0, m_in.shape[0], bs):
            mm = torch.from_numpy(m_in[i:i+bs]).float().to(device)
            ww = torch.from_numpy(w_in[i:i+bs]).float().to(device)
            dd = torch.from_numpy(d_in[i:i+bs]).float().to(device)
            em = model.encoder(mm)
            ew = model.encoder(ww)
            ed = model.encoder(dd)
            z = torch.cat([em, ew, ed], dim=1)
            Z.append(z.detach().cpu().numpy())
        post_z = np.concatenate(Z, axis=0)

    # Harmonize dims for joint projection
    pre_dim = pre_in.shape[1]
    post_dim = post_z.shape[1]
    if pre_dim != post_dim:
        target = min(pre_dim, post_dim)
        if pre_dim != target:
            pre_in = TruncatedSVD(n_components=target, random_state=42).fit_transform(pre_in)
        if post_dim != target:
            post_z = TruncatedSVD(n_components=target, random_state=42).fit_transform(post_z)

    union = np.vstack([pre_in, post_z])
    U = reduce_umap(union, y)
    pre_embed = U[:len(pre_in)]
    post_embed = U[len(pre_in):]

    out_seed = os.path.join(args.out_dir, args.subset, f'seed_{args.seed}')
    os.makedirs(out_seed, exist_ok=True)
    save_scatter_2d(pre_embed, y, f'Pre-space ({args.branch}/{args.space})', os.path.join(out_seed, 'pre_umap.png'))
    save_scatter_2d(post_embed, y, 'Post-space (Psemut z)', os.path.join(out_seed, 'post_umap.png'))

    # Side-by-side
    plt.figure(figsize=(10, 4))
    cmap = 'coolwarm'
    plt.subplot(1, 2, 1)
    _ = plt.scatter(pre_embed[:, 0], pre_embed[:, 1], c=y, cmap=cmap, s=6, alpha=0.8, linewidths=0)
    plt.title(f'Pre ({args.branch}/{args.space})'); plt.xticks([]); plt.yticks([])
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(post_embed[:, 0], post_embed[:, 1], c=y, cmap=cmap, s=6, alpha=0.8, linewidths=0)
    plt.title('Post (Psemut z)'); plt.xticks([]); plt.yticks([])
    cax = plt.axes([0.92, 0.15, 0.015, 0.7])
    plt.colorbar(sc2, cax=cax, label='label_diff')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    combo = os.path.join(out_seed, 'compare_pre_vs_post_umap.png')
    plt.savefig(combo, dpi=300)
    try:
        plt.savefig(os.path.splitext(combo)[0] + '.svg', format='svg')
    except Exception:
        pass
    plt.close()

    # Export points
    def to_df(X2: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({'x': X2[:, 0].astype(float), 'y': X2[:, 1].astype(float), 'label': y.astype(float)})
    to_df(pre_embed).to_csv(os.path.join(out_seed, 'points_pre_umap.csv'), index=False, encoding='utf-8')
    to_df(post_embed).to_csv(os.path.join(out_seed, 'points_post_umap.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    main()


