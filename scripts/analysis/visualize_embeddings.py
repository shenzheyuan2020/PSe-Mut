#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize representation shift for Siamese Psemut model:
- Pre: model input space (per-branch SVD+Scaler transformed) or raw concatenation
- Post: learned concatenated hidden z = [enc(mut), enc(wt), enc(diff)]

We fit a single 2D reducer (UMAP if available, else PCA) on the UNION of
pre and post embeddings to ensure a comparable coordinate system, then
plot side-by-side figures with identical color mapping.

Outputs (PNG+SVG):
  <out_dir>/<seed>/pre_concat_<tag>.(png|svg)
  <out_dir>/<seed>/post_psemut_<tag>.(png|svg)
  <out_dir>/<seed>/compare_<tag>.(png|svg)

Notes about “endpoint” for Siamese:
  The regression head in Psemut consumes z = concat(enc(mut), enc(wt), enc(diff)).
  We therefore visualize z as the model’s learned joint representation (endpoint).
"""

import os
import json
import argparse
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns

try:
    import umap
    HAS_UMAP = True
except Exception:
    from sklearn.decomposition import PCA, TruncatedSVD
    HAS_UMAP = False

# Ensure TruncatedSVD is available even if UMAP is installed
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.manifold import trustworthiness
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_json(path: str) -> dict:
    try:
        return json.load(open(path, 'r', encoding='utf-8'))
    except Exception:
        return {}


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: list, dropout: float):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: list, fusion_dim: int, dropout: float, enable_multitask: bool = False):
        super().__init__()
        self.encoder = Tower(in_dim, hidden, dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden[-1] * 3, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )
        self.enable_multitask = enable_multitask
        if enable_multitask:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden[-1] * 3, max(64, fusion_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(64, fusion_dim // 2), 1),
            )

    def forward(self, mut: torch.Tensor, wt: torch.Tensor, diff: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        em = self.encoder(mut)
        ew = self.encoder(wt)
        ed = self.encoder(diff)
        z = torch.cat([em, ew, ed], dim=1)
        reg = self.regressor(z)
        if self.enable_multitask:
            logits = self.cls_head(z)
            return reg, logits, z
        return reg, None, z


def try_load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            return None


def load_seed_artifacts(seed_dir: str):
    """Load model args, weights, scalers/SVDs from seed directory."""
    run_args = load_json(os.path.join(seed_dir, 'run_args.json'))
    hidden = [int(x) for x in str(run_args.get('hidden', '1024,512')).split(',')]
    dropout = float(run_args.get('dropout', 0.2))
    enable_multitask = bool(run_args.get('enable_multitask', True))
    svd_dim = int(run_args.get('svd_dim', 384))

    # Load scalers (required)
    scaler_m = try_load(os.path.join(seed_dir, 'scaler_m.pkl'))
    scaler_w = try_load(os.path.join(seed_dir, 'scaler_w.pkl'))
    scaler_d = try_load(os.path.join(seed_dir, 'scaler_d.pkl'))
    # SVDs (optional)
    svd_m = try_load(os.path.join(seed_dir, 'svd_m.pkl'))
    svd_w = try_load(os.path.join(seed_dir, 'svd_w.pkl'))
    svd_d = try_load(os.path.join(seed_dir, 'svd_d.pkl'))

    # Build model stub to infer in_dim from transformed branch
    # We'll compute after we transform one mini-batch

    # Load weights
    model_path = os.path.join(seed_dir, 'model_siamese.pth')
    state = torch.load(model_path, map_location='cpu') if os.path.exists(model_path) else None

    return {
        'run_args': run_args,
        'hidden': hidden,
        'dropout': dropout,
        'enable_multitask': enable_multitask,
        'svd_dim': svd_dim,
        'scalers': (scaler_m, scaler_w, scaler_d),
        'svds': (svd_m, svd_w, svd_d),
        'state': state,
    }


def branch_split(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert X.shape[1] % 3 == 0, 'Expected 3-branch concatenation'
    d = X.shape[1] // 3
    return X[:, :d], X[:, d:2*d], X[:, 2*d:3*d]


def apply_svd_scaler(arr: np.ndarray, svd, scaler) -> np.ndarray:
    X = arr
    if svd is not None:
        X = svd.transform(X)
    if scaler is not None:
        exp = getattr(scaler, 'n_features_in_', None)
        if exp is not None and X.shape[1] != exp:
            k = int(exp)
            k = max(1, min(k, X.shape[1]))
            X = TruncatedSVD(n_components=k, random_state=42).fit_transform(X)
        X = scaler.transform(X)
    return X


def reduce_2d(Z: np.ndarray, random_state: int = 42):
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=30, min_dist=0.15, metric='euclidean')
        return reducer.fit_transform(Z)
    # PCA fallback
    return PCA(n_components=2, random_state=random_state).fit_transform(Z)


def save_fig_scatter(X2d: np.ndarray, color: np.ndarray, title: str, out_path: str, cmap: str = 'coolwarm'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=color, cmap=cmap, s=6, alpha=0.8, linewidths=0)
    plt.colorbar(sc, label='label_diff')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    try:
        svg_path = os.path.splitext(out_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg')
    except Exception:
        pass
    plt.close()


def main():
    ap = argparse.ArgumentParser('Visualize pre-concat vs learned z embeddings')
    ap.add_argument('--data_dir', type=str, default=os.path.join('data', 'processed', 'siamese'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--weights_root', type=str, default=os.path.join('outputs', 'dl', 'compare_mt'))
    ap.add_argument('--branch', type=str, default='diff', choices=['diff', 'concat'], help='Pre space: use diff only or full concat')
    ap.add_argument('--space', type=str, default='input', choices=['input', 'raw'], help='Pre space: model input (SVD+Scaler) or raw')
    ap.add_argument('--num', type=int, default=3000)
    ap.add_argument('--out_dir', type=str, default=os.path.join('outputs', 'figures', 'embeddings'))
    ap.add_argument('--subset', type=str, default='all', choices=['all', 'train', 'val', 'test'],
                    help='Restrict visualization to a specific split if split_info.json is available')
    ap.add_argument('--use_gpu', action='store_true')
    ap.add_argument('--dump_points', action='store_true', help='Export 2D points CSV and metrics JSON for pre/post views')
    ap.add_argument('--proj', type=str, default='umap', choices=['umap', 'pca', 'pls2', 'umap3d', 'pca3d', 'umap_sup', 'umap3d_sup'], help='2D/3D projection method')
    ap.add_argument('--post_endpoint', type=str, default='z', choices=['z', 'head'], help='Use z or regressor hidden as post endpoint')
    ap.add_argument('--dump_probes', action='store_true', help='Export K-fold probe metrics (linear/MLP) on pre_in/post_z/post_head')
    args = ap.parse_args()

    seed_dir = os.path.join(args.weights_root, f'seed_{args.seed}')
    arts = load_seed_artifacts(seed_dir)

    # Load data
    pairs_csv = os.path.join(args.data_dir, 'siamese_pairs.csv')
    npy_path = os.path.join(args.data_dir, 'siamese_fingerprints.npy')
    df = pd.read_csv(pairs_csv)
    X = np.load(npy_path)
    y = df['label_diff'].values.astype(np.float32)

    n = len(df)
    orig_idx = np.arange(n)

    # Optional: restrict to a specific subset based on split_info.json in seed_dir
    if args.subset != 'all':
        split_path_subset = os.path.join(seed_dir, 'split_info.json')
        if os.path.exists(split_path_subset):
            sp_sub = load_json(split_path_subset)
            if args.subset == 'train':
                tgt = np.array(sp_sub.get('train_idx', []), dtype=int)
            elif args.subset == 'val':
                tgt = np.array(sp_sub.get('val_idx', []), dtype=int)
            else:  # test
                tgt = np.array(sp_sub.get('test_idx', []), dtype=int)
            if tgt.size > 0:
                keep_mask = np.zeros(n, dtype=bool)
                keep_mask[tgt[(tgt >= 0) & (tgt < n)]] = True
                X = X[keep_mask]
                y = y[keep_mask]
                df = df.iloc[keep_mask].reset_index(drop=True)
                orig_idx = orig_idx[keep_mask]
                n = len(df)
    if args.num and args.num < n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=args.num, replace=False)
        X = X[idx]
        y = y[idx]
        df = df.iloc[idx].reset_index(drop=True)
        orig_idx = orig_idx[idx]

    m_raw, w_raw, d_raw = branch_split(X)

    # Transform branches as model sees
    scaler_m, scaler_w, scaler_d = arts['scalers']
    svd_m, svd_w, svd_d = arts['svds']

    # If SVDs are missing, reconstruct from training split indices for faithful mapping
    if svd_m is None or svd_w is None or svd_d is None:
        split_path = os.path.join(seed_dir, 'split_info.json')
        if os.path.exists(split_path):
            sp = load_json(split_path)
            tr_idx_full = np.array(sp.get('train_idx', []), dtype=int)
            # map original indices to current subset positions
            pos_map = {int(o): int(i) for i, o in enumerate(orig_idx.tolist())}
            tr_local = [pos_map[o] for o in tr_idx_full if o in pos_map]
            if len(tr_local) >= 10:
                tr_local = np.array(tr_local, dtype=int)
                k = int(arts['svd_dim']) if arts['svd_dim'] else min(m_raw.shape[1], 384)
                rng_seed = int(args.seed)
                svd_m = TruncatedSVD(n_components=min(k, m_raw.shape[1]-1), random_state=rng_seed).fit(m_raw[tr_local])
                svd_w = TruncatedSVD(n_components=min(k, w_raw.shape[1]-1), random_state=rng_seed).fit(w_raw[tr_local])
                svd_d = TruncatedSVD(n_components=min(k, d_raw.shape[1]-1), random_state=rng_seed).fit(d_raw[tr_local])

    m_in = apply_svd_scaler(m_raw, svd_m, scaler_m)
    w_in = apply_svd_scaler(w_raw, svd_w, scaler_w)
    d_in = apply_svd_scaler(d_raw, svd_d, scaler_d)

    # Pre-embedding selection
    if args.branch == 'diff':
        pre_in = d_in if args.space == 'input' else d_raw
        pre_tag = f'pre_branch-{args.branch}_{args.space}'
    else:  # concat
        if args.space == 'input':
            pre_in = np.concatenate([m_in, w_in, d_in], axis=1)
        else:
            pre_in = X
        pre_tag = f'pre_branch-concat_{args.space}'

    # Build model and compute learned endpoint
    device = torch.device('cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu')
    in_dim = m_in.shape[1]
    hidden = arts['hidden']
    fusion_dim = max(128, hidden[-1])
    model = SiameseRegressor(in_dim, hidden, fusion_dim, dropout=arts['dropout'], enable_multitask=arts['enable_multitask']).to(device)
    if arts['state'] is not None:
        # Some checkpoints may not contain optional cls_head weights; load trunk strictly and ignore missing heads
        model.load_state_dict(arts['state'], strict=False)
    model.eval()

    with torch.no_grad():
        bs = 2048
        Z_list = []
        H_list = []
        for i in range(0, m_in.shape[0], bs):
            mm = torch.from_numpy(m_in[i:i+bs]).float().to(device)
            ww = torch.from_numpy(w_in[i:i+bs]).float().to(device)
            dd = torch.from_numpy(d_in[i:i+bs]).float().to(device)
            _, _, z = model(mm, ww, dd)
            Z_list.append(z.detach().cpu().numpy())
            # regressor hidden (before final linear)
            h = model.regressor[0](z)
            h = model.regressor[1](h)
            h = model.regressor[2](h)
            h = model.regressor[3](h)
            H_list.append(h.detach().cpu().numpy())
        post_z = np.concatenate(Z_list, axis=0)
        post_h = np.concatenate(H_list, axis=0)

    # Project to 2D/3D according to --proj
    is_3d = args.proj in ('umap3d', 'pca3d', 'umap3d_sup')
    if args.proj in ('umap', 'pca', 'umap3d', 'pca3d', 'umap_sup', 'umap3d_sup'):
        # Harmonize dims
        pre_dim = pre_in.shape[1]
        post_dim = post_z.shape[1]
        if pre_dim != post_dim:
            target_dim = min(pre_dim, post_dim)
            if pre_dim != target_dim:
                pre_in = TruncatedSVD(n_components=target_dim, random_state=42).fit_transform(pre_in)
            if post_dim != target_dim:
                post_z = TruncatedSVD(n_components=target_dim, random_state=42).fit_transform(post_z)
        union = np.vstack([pre_in, post_z])
        y_union = np.concatenate([y, y])
        if args.proj in ('umap', 'umap3d', 'umap_sup', 'umap3d_sup'):
            if HAS_UMAP:
                n_comp = 3 if is_3d else 2
                supervised = args.proj in ('umap_sup', 'umap3d_sup')
                reducer = umap.UMAP(n_components=n_comp, random_state=42, n_neighbors=30, min_dist=0.15, metric='euclidean', target_weight=0.5 if supervised else 0.0, target_metric='l2')
                if supervised:
                    U = reducer.fit_transform(union, y=y_union.reshape(-1, 1))
                else:
                    U = reducer.fit_transform(union)
            else:
                from sklearn.decomposition import PCA as _PCA
                n_comp = 3 if is_3d else 2
                U = _PCA(n_components=n_comp, random_state=42).fit_transform(union)
            if args.proj in ('umap', 'umap3d'):
                proj_tag = '' if not is_3d else '_3d'
            else:
                proj_tag = '_sup' if not is_3d else '_3d_sup'
        else:
            from sklearn.decomposition import PCA as _PCA
            n_comp = 3 if is_3d else 2
            U = _PCA(n_components=n_comp, random_state=42).fit_transform(union)
            proj_tag = '_pca' if not is_3d else '_pca3d'
        pre_embed = U[:len(pre_in)]
        post_embed = U[len(pre_in):]
    else:
        # PLS-2 remains 2D
        pre_dim = pre_in.shape[1]
        post_dim = post_z.shape[1]
        if pre_dim != post_dim:
            target_dim = min(pre_dim, post_dim)
            if pre_dim != target_dim:
                pre_in = TruncatedSVD(n_components=target_dim, random_state=42).fit_transform(pre_in)
            if post_dim != target_dim:
                post_z = TruncatedSVD(n_components=target_dim, random_state=42).fit_transform(post_z)
        union = np.vstack([pre_in, post_z])
        y_union = np.concatenate([y, y]).reshape(-1, 1)
        pls = PLSRegression(n_components=2)
        pls.fit(union, y_union)
        pre_embed = pls.transform(pre_in)
        post_embed = pls.transform(post_z)
        proj_tag = '_pls2'

    # Save figures
    out_seed = os.path.join(args.out_dir, f'seed_{args.seed}')
    os.makedirs(out_seed, exist_ok=True)
    if not is_3d:
        save_fig_scatter(pre_embed, y, f'Pre-space ({args.branch}/{args.space})', os.path.join(out_seed, f'{pre_tag}{proj_tag}.png'))
        save_fig_scatter(post_embed, y, 'Post-space (Psemut z)', os.path.join(out_seed, f'post_psemut_z{proj_tag}.png'))
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
        combo = os.path.join(out_seed, f'compare_pre_vs_post{proj_tag}.png')
        plt.savefig(combo, dpi=300)
        try:
            plt.savefig(os.path.splitext(combo)[0] + '.svg', format='svg')
        except Exception:
            pass
        plt.close()
    else:
        # 3D scatter
        def _save_fig_scatter3d(X3: np.ndarray, color: np.ndarray, title: str, out_path: str, cmap: str = 'coolwarm'):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            p = ax.scatter(X3[:,0], X3[:,1], X3[:,2], c=color, cmap=cmap, s=6, alpha=0.85)
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            fig.colorbar(p, ax=ax, shrink=0.6, label='label_diff')
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            try:
                plt.savefig(os.path.splitext(out_path)[0] + '.svg', format='svg')
            except Exception:
                pass
            plt.close(fig)
        _save_fig_scatter3d(pre_embed, y, f'Pre-space 3D ({args.branch}/{args.space})', os.path.join(out_seed, f'{pre_tag}{proj_tag}.png'))
        _save_fig_scatter3d(post_embed, y, 'Post-space 3D (Psemut z)', os.path.join(out_seed, f'post_psemut_z{proj_tag}.png'))

    # Neighborhood label variance histograms (HD vs embedding)
    def _neigh_label_var(X: np.ndarray, y_arr: np.ndarray, k: int = 15) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors
        k = max(2, min(k, len(X)-1))
        nbr = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X)
        d, idx = nbr.kneighbors(X)
        # exclude self (first neighbor is self)
        idx = idx[:, 1:]
        vals = []
        for i in range(len(X)):
            nbr_y = y_arr[idx[i]]
            vals.append(np.var(nbr_y))
        return np.asarray(vals, dtype=float)

    def _save_hist(data: np.ndarray, title: str, out_path: str):
        plt.figure(figsize=(5, 3.2))
        sns.histplot(data, bins=40, kde=False, color='#56B4E9')
        plt.title(title)
        plt.xlabel('Neighborhood label variance')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        try:
            plt.savefig(os.path.splitext(out_path)[0] + '.svg', format='svg')
        except Exception:
            pass
        plt.close()

    # Compute and export
    hd_pre_var = _neigh_label_var(pre_in, y)
    hd_post_var = _neigh_label_var(post_z, y)
    emb_pre_var = _neigh_label_var(pre_embed, y)
    emb_post_var = _neigh_label_var(post_embed, y)

    _save_hist(hd_pre_var, 'HD pre-space neighborhood variance', os.path.join(out_seed, f'neighvar_hd_{pre_tag}.png'))
    _save_hist(hd_post_var, 'HD post-space neighborhood variance', os.path.join(out_seed, f'neighvar_hd_post_psemut_z.png'))
    _save_hist(emb_pre_var, 'Embed pre-space neighborhood variance', os.path.join(out_seed, f'neighvar_embed_{pre_tag}{proj_tag}.png'))
    _save_hist(emb_post_var, 'Embed post-space neighborhood variance', os.path.join(out_seed, f'neighvar_embed_post_psemut_z{proj_tag}.png'))

    # Optional: dump 2D points and simple metrics for interpretability
    if args.dump_points:
        # CSVs for pre and post
        pre_csv = os.path.join(out_seed, f'points_{pre_tag}{proj_tag}.csv')
        post_csv = os.path.join(out_seed, f'points_post_psemut_z{proj_tag}.csv')
        # Try to include an identifier if present
        id_col = None
        for cand in ['uniprot_id', 'UNIPROT_ID', 'pid', 'protein_id']:
            if cand in df.columns:
                id_col = cand
                break
        def _mk_df(X2: np.ndarray) -> pd.DataFrame:
            dct = {
                'x': X2[:, 0].astype(float),
                'y': X2[:, 1].astype(float),
                'label': y.astype(float),
                'row_index': np.arange(len(X2), dtype=int),
            }
            if id_col is not None:
                dct[id_col] = df[id_col].values
            return pd.DataFrame(dct)
        _mk_df(pre_embed).to_csv(pre_csv, index=False, encoding='utf-8')
        _mk_df(post_embed).to_csv(post_csv, index=False, encoding='utf-8')

        # Simple linear predictability metrics in 2D
        def _lin_r2(X2: np.ndarray) -> float:
            try:
                reg = LinearRegression().fit(X2, y)
                y_hat = reg.predict(X2)
                return float(r2_score(y, y_hat))
            except Exception:
                return float('nan')
        def _pearson(X2: np.ndarray) -> float:
            try:
                # Use radial coordinate as a simple 1D summary and correlate with label
                r = np.linalg.norm(X2, axis=1)
                c = np.corrcoef(r, y)[0, 1]
                return float(c)
            except Exception:
                return float('nan')
        def _knn_r2(X2: np.ndarray, k: int = 15) -> float:
            try:
                k = max(2, min(k, len(X2) - 1))
                knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                y_hat = knn.fit(X2, y).predict(X2)
                return float(r2_score(y, y_hat))
            except Exception:
                return float('nan')

        def _local_mse(X2: np.ndarray, k: int = 15) -> float:
            try:
                k = max(2, min(k, len(X2) - 1))
                knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                y_hat = knn.fit(X2, y).predict(X2)
                return float(np.mean((y_hat - y) ** 2))
            except Exception:
                return float('nan')

        def _spearman_dist_labeldiff(X2: np.ndarray, sample_n: int = 5000) -> float:
            try:
                n = len(X2)
                if n > sample_n:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(n, size=sample_n, replace=False)
                else:
                    idx = np.arange(n)
                # Pairwise on a subsample
                A = X2[idx]
                ya = y[idx]
                # distances vs label differences
                from scipy.spatial.distance import pdist
                d = pdist(A, metric='euclidean')
                dy = pdist(ya.reshape(-1, 1), metric='euclidean')
                s, _ = spearmanr(d, dy)
                return float(s)
            except Exception:
                return float('nan')

        def _quantile_auc(X2: np.ndarray, q: float = 0.1) -> float:
            try:
                r = np.linalg.norm(X2, axis=1)
                y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
                thr_low = np.quantile(y_norm, q)
                thr_high = np.quantile(y_norm, 1 - q)
                y_low = (y_norm <= thr_low).astype(int)
                y_high = (y_norm >= thr_high).astype(int)
                # Use r as score; AUC for extreme low vs high separately
                from sklearn.metrics import roc_auc_score
                def safe_auc(t, s):
                    try:
                        if len(np.unique(t)) < 2:
                            return float('nan')
                        return float(roc_auc_score(t, s))
                    except Exception:
                        return float('nan')
                auc_low = safe_auc(y_low, -r)  # smaller radius for low label?
                auc_high = safe_auc(y_high, r) # larger radius for high label?
                return float(np.nanmean([auc_low, auc_high]))
            except Exception:
                return float('nan')

        # High-dimensional metrics on pre_in and post_z
        def _cv_r2(model, X: np.ndarray, y_arr: np.ndarray, n_splits: int = 5) -> float:
            try:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = []
                for tr, va in kf.split(X):
                    m = model()
                    m.fit(X[tr], y_arr[tr])
                    scores.append(float(r2_score(y_arr[va], m.predict(X[va]))))
                return float(np.mean(scores)) if scores else float('nan')
            except Exception:
                return float('nan')

        def _spearman_dist_labeldiff_hd(X: np.ndarray, sample_n: int = 2000) -> float:
            try:
                n = len(X)
                if n > sample_n:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(n, size=sample_n, replace=False)
                else:
                    idx = np.arange(n)
                from scipy.spatial.distance import pdist
                d = pdist(X[idx], metric='euclidean')
                dy = pdist(y[idx].reshape(-1, 1), metric='euclidean')
                s, _ = spearmanr(d, dy)
                return float(s)
            except Exception:
                return float('nan')

        # Compose base metrics (use first 2 dims if 3D)
        def _maybe_2d(E: np.ndarray) -> np.ndarray:
            return E[:, :2] if E.shape[1] > 2 else E
        pre2 = _maybe_2d(pre_embed)
        post2 = _maybe_2d(post_embed)
        metrics = {
            'r2_linear_pre': _lin_r2(pre2),
            'r2_linear_post': _lin_r2(post2),
            'knn_r2_pre': _knn_r2(pre2),
            'knn_r2_post': _knn_r2(post2),
            'local_mse_pre': _local_mse(pre2),
            'local_mse_post': _local_mse(post2),
            'spearman_dist_labeldiff_pre': _spearman_dist_labeldiff(pre2),
            'spearman_dist_labeldiff_post': _spearman_dist_labeldiff(post2),
            'quantile_auc_pre': _quantile_auc(pre2),
            'quantile_auc_post': _quantile_auc(post2),
            'pearson_pre_radius_label': _pearson(pre2),
            'pearson_post_radius_label': _pearson(post2),
            'num_points': int(len(y)),
            'note': 'R2: linear/knn; local_mse from kNN; spearman on pairwise distance vs |Δlabel|; quantile_auc on extremes.'
        }

        # Add high-dimensional CV R^2 and structure metrics
        metrics.update({
            'hd_linear_cv_r2_pre': _cv_r2(lambda: LinearRegression(), pre_in, y),
            'hd_linear_cv_r2_post': _cv_r2(lambda: LinearRegression(), post_z, y),
            'hd_knn_cv_r2_pre': _cv_r2(lambda: KNeighborsRegressor(n_neighbors=15, weights='distance'), pre_in, y),
            'hd_knn_cv_r2_post': _cv_r2(lambda: KNeighborsRegressor(n_neighbors=15, weights='distance'), post_z, y),
            'hd_mlp_cv_r2_pre': _cv_r2(lambda: MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42), pre_in, y),
            'hd_mlp_cv_r2_post': _cv_r2(lambda: MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42), post_z, y),
            'hd_spearman_dist_labeldiff_pre': _spearman_dist_labeldiff_hd(pre_in),
            'hd_spearman_dist_labeldiff_post': _spearman_dist_labeldiff_hd(post_z),
            'trustworthiness_pre2_wrt_pre': float(trustworthiness(pre_in, pre2, n_neighbors=10)),
            'trustworthiness_post2_wrt_post': float(trustworthiness(post_z, post2, n_neighbors=10)),
        })

        if args.proj == 'pls2':
            # K-fold CV R^2 under PLS-2 mapping (fit within-fold per space)
            def _cv_pls2_r2(X: np.ndarray, y_arr: np.ndarray, n_splits: int = 5) -> float:
                try:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    scores = []
                    for tr, va in kf.split(X):
                        pls_cv = PLSRegression(n_components=2)
                        pls_cv.fit(X[tr], y_arr[tr].reshape(-1, 1))
                        Xtr2 = pls_cv.transform(X[tr])
                        Xva2 = pls_cv.transform(X[va])
                        r2_cv = LinearRegression().fit(Xtr2, y_arr[tr]).score(Xva2, y_arr[va])
                        scores.append(r2_cv)
                    return float(np.mean(scores))
                except Exception:
                    return float('nan')
            metrics['pls2_cv_r2_pre'] = _cv_pls2_r2(pre_in, y)
            metrics['pls2_cv_r2_post'] = _cv_pls2_r2(post_z, y)

        with open(os.path.join(out_seed, f'points_metrics{proj_tag}.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Optional: dump high-dimensional K-fold probe metrics
    if args.dump_probes:
        def _build_pipeline(model_name: str, params: dict):
            if model_name == 'linear':
                return make_pipeline(StandardScaler(with_mean=True), LinearRegression())
            if model_name == 'ridge':
                alpha = params.get('alpha', 1.0)
                return make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=alpha, random_state=42))
            if model_name == 'kr':
                alpha = params.get('alpha', 1.0)
                gamma = params.get('gamma', 0.1)
                return make_pipeline(StandardScaler(with_mean=True), KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma))
            if model_name == 'mlp':
                hls = params.get('hls', (128, 64))
                return make_pipeline(
                    StandardScaler(with_mean=True),
                    MLPRegressor(hidden_layer_sizes=hls, max_iter=1000, random_state=42, early_stopping=True, n_iter_no_change=15)
                )
            return make_pipeline(StandardScaler(with_mean=True), LinearRegression())

        def _cv_metrics(X: np.ndarray, y_arr: np.ndarray, model_name: str, n_splits: int = 5, indices: np.ndarray = None) -> dict:
            try:
                idx = np.arange(len(X)) if indices is None else indices
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                def eval_params(build_fn):
                    r2s, maes, rmses = [], [], []
                    for tr_local, va_local in kf.split(idx):
                        tr = idx[tr_local]; va = idx[va_local]
                        mdl = build_fn()
                        mdl.fit(X[tr], y_arr[tr])
                        y_hat = mdl.predict(X[va])
                        r2s.append(r2_score(y_arr[va], y_hat))
                        maes.append(mean_absolute_error(y_arr[va], y_hat))
                        rmses.append(np.sqrt(mean_squared_error(y_arr[va], y_hat)))
                    return float(np.mean(r2s)), float(np.mean(maes)), float(np.mean(rmses))

                best = {'r2_mean': -1e9, 'mae_mean': 1e9, 'rmse_mean': 1e9, 'params': {}}
                if model_name == 'linear':
                    def build():
                        return _build_pipeline('linear', {})
                    r2, mae, rmse = eval_params(build)
                    best = {'r2_mean': r2, 'mae_mean': mae, 'rmse_mean': rmse, 'params': {}}
                elif model_name == 'ridge':
                    for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                        def build_a(alpha=a):
                            return _build_pipeline('ridge', {'alpha': alpha})
                        r2, mae, rmse = eval_params(build_a)
                        if r2 > best['r2_mean']:
                            best = {'r2_mean': r2, 'mae_mean': mae, 'rmse_mean': rmse, 'params': {'alpha': a}}
                elif model_name == 'kr':
                    for a in [1e-3, 1e-2, 1e-1, 1.0]:
                        for g in [1e-3, 1e-2, 1e-1, 1.0]:
                            def build_ag(alpha=a, gamma=g):
                                return _build_pipeline('kr', {'alpha': alpha, 'gamma': gamma})
                            r2, mae, rmse = eval_params(build_ag)
                            if r2 > best['r2_mean']:
                                best = {'r2_mean': r2, 'mae_mean': mae, 'rmse_mean': rmse, 'params': {'alpha': a, 'gamma': g}}
                elif model_name == 'mlp':
                    for h in [(128,64), (256,128)]:
                        def build_h(hls=h):
                            return _build_pipeline('mlp', {'hls': hls})
                        r2, mae, rmse = eval_params(build_h)
                        if r2 > best['r2_mean']:
                            best = {'r2_mean': r2, 'mae_mean': mae, 'rmse_mean': rmse, 'params': {'hls': h}}
                else:
                    def build():
                        return _build_pipeline('linear', {})
                    r2, mae, rmse = eval_params(build)
                    best = {'r2_mean': r2, 'mae_mean': mae, 'rmse_mean': rmse, 'params': {}}

                return best
            except Exception:
                return {'r2_mean': float('nan'), 'mae_mean': float('nan'), 'rmse_mean': float('nan'), 'params': {}}

        def _test_metrics(X: np.ndarray, y_arr: np.ndarray, model_name: str, params: dict, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
            try:
                mdl = _build_pipeline(model_name, params)
                mdl.fit(X[train_idx], y_arr[train_idx])
                y_hat = mdl.predict(X[test_idx])
                return {
                    'r2': float(r2_score(y_arr[test_idx], y_hat)),
                    'mae': float(mean_absolute_error(y_arr[test_idx], y_hat)),
                    'rmse': float(np.sqrt(mean_squared_error(y_arr[test_idx], y_hat))),
                }
            except Exception:
                return {'r2': float('nan'), 'mae': float('nan'), 'rmse': float('nan')}

        # Determine split-aligned indices if available
        train_local = val_local = test_local = None
        split_path = os.path.join(seed_dir, 'split_info.json')
        if os.path.exists(split_path):
            sp = load_json(split_path)
            tr_idx_full = np.array(sp.get('train_idx', []), dtype=int)
            va_idx_full = np.array(sp.get('val_idx', []), dtype=int)
            te_idx_full = np.array(sp.get('test_idx', []), dtype=int)
            pos_map = {int(o): int(i) for i, o in enumerate(orig_idx.tolist())}
            train_local = np.array([pos_map[o] for o in tr_idx_full if o in pos_map], dtype=int)
            val_local = np.array([pos_map[o] for o in va_idx_full if o in pos_map], dtype=int)
            test_local = np.array([pos_map[o] for o in te_idx_full if o in pos_map], dtype=int)
            # Merge train+val as train set for probe selection
            if train_local.size and val_local.size:
                train_local = np.concatenate([train_local, val_local], axis=0)

        rows = []
        spaces = [('pre_in', pre_in), ('post_z', post_z), ('post_head', post_h)]
        models = ['linear', 'ridge', 'kr', 'mlp']
        for space_name, Xspace in spaces:
            for model_name in models:
                if train_local is not None and train_local.size >= 20 and test_local is not None and test_local.size >= 10:
                    # Train-aligned CV on train_local
                    cv_best = _cv_metrics(Xspace, y, model_name, n_splits=5, indices=train_local)
                    rows.append({'set': 'train_cv', 'space': space_name, 'model': model_name, **cv_best})
                    # Test-only with best params
                    test_res = _test_metrics(Xspace, y, model_name, cv_best.get('params', {}), train_local, test_local)
                    rows.append({'set': 'test', 'space': space_name, 'model': model_name, **test_res, 'params': cv_best.get('params', {})})
                else:
                    # Fallback to global KFold
                    cv_best = _cv_metrics(Xspace, y, model_name, n_splits=5, indices=None)
                    rows.append({'set': 'global_cv', 'space': space_name, 'model': model_name, **cv_best})

        df_probe = pd.DataFrame(rows)
        out_probe_csv = os.path.join(out_seed, f'probe_metrics_{args.branch}_{args.space}_split.csv')
        out_probe_json = os.path.join(out_seed, f'probe_metrics_{args.branch}_{args.space}_split.json')
        df_probe.to_csv(out_probe_csv, index=False, encoding='utf-8')
        with open(out_probe_json, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()


