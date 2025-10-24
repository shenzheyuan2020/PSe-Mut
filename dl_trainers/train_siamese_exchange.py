#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siamese-tower regressor with exchange-consistency on siamese_dataset.

Design:
  - Split combined features [mut, wt, diff] -> feed mut, wt, diff through a shared MLP encoder
  - Concatenate encoded {Emut, Ewt, Ediff} and predict ΔΔG
  - Exchange consistency: swap(mut, wt, diff)->(wt, mut, -diff) and enforce prediction ≈ -original

Usage:
  python dl_trainers/train_siamese_exchange.py --data_dir siamese_dataset --output_dir dl_results_siamese
"""

import os
import json
import math
import time
import random
import argparse
import pickle
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
import logging


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset_arrays(data_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    csv_file = os.path.join(data_dir, "siamese_pairs.csv")
    npy_file = os.path.join(data_dir, "siamese_fingerprints.npy")
    if not os.path.exists(csv_file) or not os.path.exists(npy_file):
        raise FileNotFoundError(f"Missing data files: {csv_file} or {npy_file}")
    df = pd.read_csv(csv_file)
    X = np.load(npy_file)
    if len(df) != X.shape[0]:
        raise ValueError(f"CSV row count ({len(df)}) does not match fingerprint count ({X.shape[0]})")
    return X, df


def split_mut_wt_diff(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[1] % 3 != 0:
        raise ValueError("Feature dimension is not a multiple of 3, cannot split mut/wt/diff")
    d = X.shape[1] // 3
    mut = X[:, :d]
    wt = X[:, d:2*d]
    diff = X[:, 2*d:3*d]
    return mut, wt, diff


class TrioDataset(Dataset):
    def __init__(self, mut: np.ndarray, wt: np.ndarray, diff: np.ndarray, labels: np.ndarray, uniprot_ids: List[str]):
        self.m = mut.astype(np.float32)
        self.w = wt.astype(np.float32)
        self.d = diff.astype(np.float32)
        self.y = labels.astype(np.float32)
        self.uniprot_ids = list(uniprot_ids)

    def __len__(self):
        return self.m.shape[0]

    def __getitem__(self, idx):
        return {
            "mut": torch.from_numpy(self.m[idx]),
            "wt": torch.from_numpy(self.w[idx]),
            "diff": torch.from_numpy(self.d[idx]),
            "y": torch.tensor([self.y[idx]], dtype=torch.float32),
            "uniprot": self.uniprot_ids[idx],
        }


def protein_group_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if "uniprot_id" in df.columns:
        pid_col = "uniprot_id"
    elif "UNIPROT_ID" in df.columns:
        pid_col = "UNIPROT_ID"
    else:
        raise KeyError("CSV missing uniprot_id column")
    groups = df.groupby(pid_col).indices
    all_pids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(all_pids)
    total = len(df)
    target_val = int(total * val_ratio)
    val_pids, val_count = [], 0
    for pid in all_pids:
        if val_count < target_val:
            val_pids.append(pid)
            val_count += len(groups[pid])
    val_indices = np.concatenate([groups[p] for p in val_pids]).astype(int)
    train_mask = np.ones(total, dtype=bool)
    train_mask[val_indices] = False
    train_indices = np.arange(total)[train_mask]
    return train_indices, val_indices


def random_split(total: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(total)
    rng.shuffle(idx)
    v = int(total * val_ratio)
    return idx[v:], idx[:v]


def three_way_split(total: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way split: train/validation/test"""
    rng = np.random.RandomState(seed)
    idx = np.arange(total)
    rng.shuffle(idx)
    
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size
    
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]
    
    return train_idx, val_idx, test_idx


def protein_group_three_split(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way split based on protein grouping"""
    if "uniprot_id" in df.columns:
        pid_col = "uniprot_id"
    elif "UNIPROT_ID" in df.columns:
        pid_col = "UNIPROT_ID"
    else:
        raise KeyError("CSV missing uniprot_id column")
    
    groups = df.groupby(pid_col).indices
    all_pids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(all_pids)
    
    total = len(df)
    target_test = int(total * test_ratio)
    target_val = int(total * val_ratio)
    
            # First separate test set
    test_pids, test_count = [], 0
    for pid in all_pids:
        if test_count < target_test:
            test_pids.append(pid)
            test_count += len(groups[pid])
    
            # Separate validation set from remaining proteins
    remaining_pids = [p for p in all_pids if p not in test_pids]
    val_pids, val_count = [], 0
    for pid in remaining_pids:
        if val_count < target_val:
            val_pids.append(pid)
            val_count += len(groups[pid])
    
            # Remaining as training set
    train_pids = [p for p in all_pids if p not in test_pids and p not in val_pids]
    
    test_indices = np.concatenate([groups[p] for p in test_pids]).astype(int)
    val_indices = np.concatenate([groups[p] for p in val_pids]).astype(int)
    train_indices = np.concatenate([groups[p] for p in train_pids]).astype(int)
    
    return train_indices, val_indices, test_indices


def get_kfold_splits(df: pd.DataFrame, n_splits: int, seed: int, restrict: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold cross-validation split"""
    if restrict:
        # K-fold based on protein grouping
        if "uniprot_id" in df.columns:
            pid_col = "uniprot_id"
        elif "UNIPROT_ID" in df.columns:
            pid_col = "UNIPROT_ID"
        else:
            raise KeyError("CSV missing uniprot_id column")
        
        groups = df.groupby(pid_col).indices
        all_pids = list(groups.keys())
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        
        for train_pid_idx, val_pid_idx in kf.split(all_pids):
            train_pids = [all_pids[i] for i in train_pid_idx]
            val_pids = [all_pids[i] for i in val_pid_idx]
            
            train_indices = np.concatenate([groups[p] for p in train_pids]).astype(int)
            val_indices = np.concatenate([groups[p] for p in val_pids]).astype(int)
            
            splits.append((train_indices, val_indices))
    else:
        # Regular K-fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        total = len(df)
        
        for train_idx, val_idx in kf.split(range(total)):
            splits.append((train_idx, val_idx))
    
    return splits


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], fusion_dim: int, dropout: float,
                 enable_multitask: bool = False):
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

    def forward(self, mut: torch.Tensor, wt: torch.Tensor, diff: torch.Tensor):
        em = self.encoder(mut)
        ew = self.encoder(wt)
        ed = self.encoder(diff)
        z = torch.cat([em, ew, ed], dim=1)
        reg = self.regressor(z)
        if self.enable_multitask:
            logits = self.cls_head(z)
            return reg, logits
        return reg, None


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0.0, None)
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": float(pr)}


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1, num_pairs: int = 512) -> torch.Tensor:
    """Simple pairwise hinge ranking loss: encourage ordering consistency."""
    n = pred.shape[0]
    if n < 2:
        return pred.new_tensor(0.0)
    i_idx = torch.randint(0, n, (num_pairs,), device=pred.device)
    j_idx = torch.randint(0, n, (num_pairs,), device=pred.device)
    mask = i_idx != j_idx
    if not mask.any():
        return pred.new_tensor(0.0)
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    y_i, y_j = target[i_idx], target[j_idx]
    p_i, p_j = pred[i_idx], pred[j_idx]
    s = torch.sign(y_j - y_i)
    valid = s != 0
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    s = s[valid]
    p_i, p_j = p_i[valid], p_j[valid]
    losses = torch.relu(margin - s * (p_j - p_i))
    return losses.mean()


def train_one_fold(args, seed: int, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray, 
                   X: np.ndarray, df: pd.DataFrame, labels: np.ndarray, logger: logging.Logger) -> Dict[str, dict]:
    """Train single fold"""
    set_global_seed(seed + fold_idx)  # Ensure different randomness for each fold
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    
    mut, wt, diff = split_mut_wt_diff(X)
    pid_series = df["uniprot_id"] if "uniprot_id" in df.columns else df["UNIPROT_ID"]
    
    m_tr, w_tr, d_tr, y_tr = mut[train_idx], wt[train_idx], diff[train_idx], labels[train_idx]
    m_va, w_va, d_va, y_va = mut[val_idx], wt[val_idx], diff[val_idx], labels[val_idx]
    
    # Optional SVD reduction (per-branch)
    svd_m = svd_w = svd_d = None
    if hasattr(args, 'svd_dim') and args.svd_dim and args.svd_dim > 0:
        svd_m = TruncatedSVD(n_components=args.svd_dim, random_state=seed).fit(m_tr)
        svd_w = TruncatedSVD(n_components=args.svd_dim, random_state=seed).fit(w_tr)
        svd_d = TruncatedSVD(n_components=args.svd_dim, random_state=seed).fit(d_tr)
        m_tr, w_tr, d_tr = svd_m.transform(m_tr), svd_w.transform(w_tr), svd_d.transform(d_tr)
        m_va, w_va, d_va = svd_m.transform(m_va), svd_w.transform(w_va), svd_d.transform(d_va)

    # Standardize with train only
    scaler_m = StandardScaler().fit(m_tr)
    scaler_w = StandardScaler().fit(w_tr)
    scaler_d = StandardScaler().fit(d_tr)
    m_tr, w_tr, d_tr = scaler_m.transform(m_tr), scaler_w.transform(w_tr), scaler_d.transform(d_tr)
    m_va, w_va, d_va = scaler_m.transform(m_va), scaler_w.transform(w_va), scaler_d.transform(d_va)

    # Diff-only ablation: zero-out mut/wt branches
    if getattr(args, 'use_diff_only', False):
        import numpy as _np
        m_tr = _np.zeros_like(m_tr)
        w_tr = _np.zeros_like(w_tr)
        m_va = _np.zeros_like(m_va)
        w_va = _np.zeros_like(w_va)
    if getattr(args, 'use_mut_only', False):
        import numpy as _np
        w_tr = _np.zeros_like(w_tr)
        d_tr = _np.zeros_like(d_tr)
        w_va = _np.zeros_like(w_va)
        d_va = _np.zeros_like(d_va)
    if getattr(args, 'use_wt_only', False):
        import numpy as _np
        m_tr = _np.zeros_like(m_tr)
        d_tr = _np.zeros_like(d_tr)
        m_va = _np.zeros_like(m_va)
        d_va = _np.zeros_like(d_va)

    # Label standardization stats (for regression)
    y_mean = float(np.mean(y_tr))
    y_std = float(np.std(y_tr) + 1e-8)
    y_abs_mean = float(np.mean(np.abs(y_tr)))
    y_abs_std = float(np.std(np.abs(y_tr)) + 1e-8)

    train_ds = TrioDataset(m_tr, w_tr, d_tr, y_tr, pid_series.iloc[train_idx].tolist())
    val_ds = TrioDataset(m_va, w_va, d_va, y_va, pid_series.iloc[val_idx].tolist())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    logger.info(f"[SEED {seed} FOLD {fold_idx}] Train size: {len(train_ds)}, Val size: {len(val_ds)}, in_dim: {m_tr.shape[1]}")

    in_dim = m_tr.shape[1]
    hidden_dims = [int(x) for x in args.hidden.split(',')]
    model = SiameseRegressor(in_dim, hidden=hidden_dims, fusion_dim=max(128, hidden_dims[-1]), dropout=args.dropout,
                             enable_multitask=args.enable_multitask).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Fixed to use Huber loss (best configuration)
    reg_criterion = nn.HuberLoss(delta=args.huber_delta)
    # Optional classification loss (predict positive/negative of label_diff)
    cls_criterion = None
    pos_weight_tensor = None
    if args.enable_multitask:
        # Calculate pos_weight based on positive/negative ratio of original labels in training set (avoid extreme imbalance)
        y_bin_tr = (y_tr >= 0).astype(np.float32)
        num_pos = float(y_bin_tr.sum())
        num_neg = float(len(y_bin_tr) - num_pos)
        if num_pos > 0:
            pos_weight = max(1.0, num_neg / num_pos)
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_rmse = float("inf")
    best_state = None
    patience = args.early_stopping_patience
    no_imp = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            m = batch["mut"].to(device)
            w = batch["wt"].to(device)
            d = batch["diff"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred, logits = model(m, w, d)
            loss = reg_criterion(pred, y)
            if args.enable_multitask and logits is not None and cls_criterion is not None:
                y_bin = (y >= 0).float()
                loss_cls = cls_criterion(logits, y_bin)
                loss = loss + args.cls_weight * loss_cls

            # Fixed to enable exchange consistency (best configuration)
            if args.exchange_consistency:
                # swap(mut, wt, diff)->(wt, mut, -diff); pred should be -original
                pred_swap, _ = model(w, m, -d)
                loss_cons = F.l1_loss(pred_swap, -pred.detach())
                loss = loss + args.consistency_weight * loss_cons

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * m.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        val_cls_logits, val_cls_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                m = batch["mut"].to(device)
                w = batch["wt"].to(device)
                d = batch["diff"].to(device)
                y = batch["y"].to(device)
                p, logits = model(m, w, d)
                val_preds.append(p.detach().cpu().numpy().reshape(-1))
                val_true.append(y.cpu().numpy().reshape(-1))
                if args.enable_multitask and logits is not None:
                    val_cls_logits.append(logits.detach().cpu().numpy().reshape(-1))
                    val_cls_true.append((y.cpu().numpy().reshape(-1) >= 0).astype(np.int32))

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        metrics = {
            "rmse": math.sqrt(mean_squared_error(val_true, val_preds)),
            "mae": mean_absolute_error(val_true, val_preds),
            "r2": r2_score(val_true, val_preds),
            "pearson": float(pearsonr(val_true, val_preds)[0]) if len(val_true) > 1 else 0.0,
        }
        if args.enable_multitask and len(val_cls_logits) > 0:
            try:
                logits_all = np.concatenate(val_cls_logits)
                ybin_all = np.concatenate(val_cls_true)
                probs_all = 1.0 / (1.0 + np.exp(-logits_all))
                auc = roc_auc_score(ybin_all, probs_all)
            except Exception:
                auc = float('nan')
            metrics["auc"] = float(auc)
        
        if epoch % 10 == 0 or epoch <= 5:  # Reduce log output
            auc_str = f" AUC {metrics['auc']:.4f}" if "auc" in metrics else ""
            msg = f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val RMSE {metrics['rmse']:.4f} R2 {metrics['r2']:.4f}{auc_str}"
            print(msg)
            logger.info(f"[SEED {seed} FOLD {fold_idx}] {msg}")

        # Snapshot: 保存层间表征（val 子集）以及 head 中间特征/概率（用于解释“为何能分开”）
        try:
            if getattr(args, 'snapshot_interval', 0) and args.snapshot_interval > 0 and (epoch % args.snapshot_interval == 0 or epoch == 1 or epoch == args.epochs):
                # Only take snapshot for one branch (default diff)
                branch = getattr(args, 'snapshot_branch', 'diff')
                branch_arr = {'mut': m_va, 'wt': w_va, 'diff': d_va}.get(branch, d_va)
                num = int(min(len(branch_arr), getattr(args, 'snapshot_max', 2000)))
                if num > 0:
                    # Target directory
                    out_snap = os.path.join(args.output_dir, f"seed_{seed}", 'analysis', 'snapshots')
                    os.makedirs(out_snap, exist_ok=True)
                    # Calculate required layer indices
                    num_blocks = sum(1 for m in model.encoder.net if isinstance(m, nn.Linear))
                    first_idx = 0
                    last_idx = max(0, num_blocks - 1)
                    mid_idx = (first_idx + last_idx) // 2
                    wanted = sorted(set([first_idx, mid_idx, last_idx]))
                    # Hook registration
                    buf: Dict[int, list] = {}
                    handles = []
                    for bi in wanted:
                        module_idx = bi * 4 + 3  # Dropout module
                        if module_idx >= len(model.encoder.net):
                            continue
                        def _make_hook(layer_id: int):
                            def _hook(_m, _i, out):
                                buf.setdefault(layer_id, []).append(out.detach().cpu())
                            return _hook
                        handles.append(model.encoder.net[module_idx].register_forward_hook(_make_hook(bi)))
                    # Forward pass
                    model.encoder.eval()
                    with torch.no_grad():
                        bs = 2048
                        for i in range(0, num, bs):
                            b = torch.from_numpy(branch_arr[i:i+bs]).float().to(device)
                            _ = model.encoder(b)
                    for h in handles:
                        h.remove()
                    # Save
                    layer_map = {first_idx: 'layer1', mid_idx: 'layermid', last_idx: 'layerlast'}
                    for li, name in layer_map.items():
                        if li in buf and len(buf[li]) > 0:
                            Z = torch.cat(buf[li], dim=0).cpu().numpy()
                            np.save(os.path.join(out_snap, f"epoch_{epoch:03d}_{name}_{branch}.npy"), Z)
                    # Head intermediate features and probabilities (using same subset alignment)
                    # Construct three-branch encoding to get z
                    with torch.no_grad():
                        mm = torch.from_numpy(m_va[:num]).float().to(device)
                        ww = torch.from_numpy(w_va[:num]).float().to(device)
                        dd = torch.from_numpy(d_va[:num]).float().to(device)
                        em = model.encoder(mm); ew = model.encoder(ww); ed = model.encoder(dd)
                        z = torch.cat([em, ew, ed], dim=1)
                        head_h = None; head_logits = None
                        if args.enable_multitask:
                            # Take cls_head previous layer activation as head hidden
                            h0 = model.cls_head[0](z)
                            h1 = model.cls_head[1](h0)
                            h2 = model.cls_head[2](h1)
                            head_h = h2.detach().cpu().numpy()
                            head_logits = model.cls_head[3](h2).detach().cpu().numpy().reshape(-1)
                            np.save(os.path.join(out_snap, f"epoch_{epoch:03d}_head_hidden.npy"), head_h)
                            np.save(os.path.join(out_snap, f"epoch_{epoch:03d}_head_logits.npy"), head_logits)
                    # Metadata
                    meta = pd.DataFrame({
                        'val_index': val_idx[:num],
                        'label_diff': y_va[:num],
                        'label_bin': (y_va[:num] >= 0).astype(np.int32),
                        'uniprot_id': (pid_series.iloc[val_idx].astype(str).values)[:num],
                        'branch': [branch]*num,
                    })
                    meta.to_csv(os.path.join(out_snap, f"epoch_{epoch:03d}_meta.csv"), index=False)
        except Exception as _e:
            logger.warning(f"[SEED {seed}] snapshot failed at epoch {epoch}: {_e}")

        if metrics["rmse"] + 1e-6 < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = {
                "epoch": epoch,
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "scaler_m": scaler_m,
                "scaler_w": scaler_w,
                "scaler_d": scaler_d,
                "svd_m": svd_m,
                "svd_w": svd_w,
                "svd_d": svd_d,
                "val_indices": val_idx,
                "metrics": {k: float(v) for k, v in metrics.items()},
            }
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                logger.info(f"[SEED {seed} FOLD {fold_idx}] 早停触发 at epoch {epoch}")
                break

    if best_state is None:
        best_state = {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "scaler_m": scaler_m,
            "scaler_w": scaler_w,
            "scaler_d": scaler_d,
            "svd_m": svd_m,
            "svd_w": svd_w,
            "svd_d": svd_d,
            "val_indices": val_idx,
            "metrics": {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "pearson": float("nan")},
        }
    
    return {"metrics": best_state.get("metrics", {}), "best_epoch": int(best_state.get("epoch", -1)), "best_state": best_state}


def _transform_with_svd_scaler(arr: np.ndarray, svd: object, scaler: object) -> np.ndarray:
    X = arr if (svd is None) else svd.transform(arr)
    return scaler.transform(X)


def _posthoc_analysis(out_dir: str, args, best_state: dict,
                      m_raw: np.ndarray, w_raw: np.ndarray, d_raw: np.ndarray,
                      y_raw: np.ndarray, hidden_dims: List[int], device: torch.device) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # 重建模型
    in_dim = (_transform_with_svd_scaler(m_raw, best_state.get('svd_m'), best_state['scaler_m']).shape[1])
    fusion_dim = max(128, hidden_dims[-1])
    model = SiameseRegressor(in_dim, hidden=hidden_dims, fusion_dim=fusion_dim,
                             dropout=args.dropout, enable_multitask=args.enable_multitask).to(device)
    model.load_state_dict(best_state["model"])
    model.eval()

    m_t = _transform_with_svd_scaler(m_raw, best_state.get('svd_m'), best_state['scaler_m'])
    w_t = _transform_with_svd_scaler(w_raw, best_state.get('svd_w'), best_state['scaler_w'])
    d_t = _transform_with_svd_scaler(d_raw, best_state.get('svd_d'), best_state['scaler_d'])

    # 交换一致性
    with torch.no_grad():
        pa_list, pb_list = [], []
        for i in range(0, m_t.shape[0], 2048):
            mm = torch.from_numpy(m_t[i:i+2048]).float().to(device)
            ww = torch.from_numpy(w_t[i:i+2048]).float().to(device)
            dd = torch.from_numpy(d_t[i:i+2048]).float().to(device)
            pa, _ = model(mm, ww, dd)
            pb, _ = model(ww, mm, -dd)
            pa_list.append(pa.detach().cpu().numpy().reshape(-1))
            pb_list.append(pb.detach().cpu().numpy().reshape(-1))
    pa = np.concatenate(pa_list); pb = np.concatenate(pb_list)
    fig = plt.figure(figsize=(5,5)); ax = plt.gca()
    ax.scatter(pa, -pb, s=6, alpha=0.5, c="#1f77b4")
    lim = float(max(np.percentile(np.abs(pa), 99), np.percentile(np.abs(pb), 99)))
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    ax.set_xlabel('pred(mut,wt,diff)'); ax.set_ylabel('-pred(wt,mut,-diff)')
    ax.set_title('Exchange consistency (regression)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'exchange_consistency.png'), dpi=300); plt.close()
    try:
        ex_pearson = float(np.corrcoef(pa, -pb)[0,1])
    except Exception:
        ex_pearson = float('nan')
    ex_mae = float(np.mean(np.abs(pa + pb)))

    # 分类头 AUC/AP 与概率分布
    head_auc = head_ap = float('nan')
    if args.enable_multitask:
        y_true = (y_raw >= 0).astype(np.int32)
        logits_all = []
        with torch.no_grad():
            for i in range(0, m_t.shape[0], 2048):
                mm = torch.from_numpy(m_t[i:i+2048]).float().to(device)
                ww = torch.from_numpy(w_t[i:i+2048]).float().to(device)
                dd = torch.from_numpy(d_t[i:i+2048]).float().to(device)
                _, logits = model(mm, ww, dd)
                if logits is not None:
                    logits_all.append(logits.detach().cpu().numpy().reshape(-1))
        if logits_all:
            logits_cat = np.concatenate(logits_all)
            probs = 1.0/(1.0+np.exp(-logits_cat))
            try:
                head_auc = float(roc_auc_score(y_true, probs))
            except Exception:
                head_auc = float('nan')
            try:
                head_ap = float(average_precision_score(y_true, probs))
            except Exception:
                head_ap = float('nan')
            fig = plt.figure(figsize=(6,4)); ax = plt.gca()
            ax.hist(probs[y_true==1], bins=50, alpha=0.6, label='pos', density=True, color="#1f77b4")
            ax.hist(probs[y_true==0], bins=50, alpha=0.6, label='neg', density=True, color="#c7ccd4")
            ax.set_title(f'Head prob distribution (AUC={head_auc:.3f})'); ax.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'head_prob_hist.png'), dpi=300); plt.close()
            # ROC & PR 曲线
            try:
                fpr, tpr, _ = roc_curve(y_true, probs)
                prec, rec, _ = precision_recall_curve(y_true, probs)
                fig = plt.figure(figsize=(10,4))
                ax1 = plt.subplot(1,2,1)
                ax1.plot(fpr, tpr, c="#1f77b4"); ax1.plot([0,1],[0,1], c="#c7ccd4", ls='--')
                ax1.set_title(f'ROC (AUC={head_auc:.3f})'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
                ax2 = plt.subplot(1,2,2)
                ax2.plot(rec, prec, c="#1f77b4"); ax2.set_title(f'PR (AP={head_ap:.3f})'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'head_roc_pr.png'), dpi=300); plt.close()
            except Exception:
                pass

    # 分支消融敏感度
    ablation = {}
    if args.enable_multitask:
        base_logits_list, base_reg_list = [], []
        zm_log, zw_log, zd_log = [], [], []
        zm_reg, zw_reg, zd_reg = [], [], []
        with torch.no_grad():
            for i in range(0, m_t.shape[0], 2048):
                mm = torch.from_numpy(m_t[i:i+2048]).float().to(device)
                ww = torch.from_numpy(w_t[i:i+2048]).float().to(device)
                dd = torch.from_numpy(d_t[i:i+2048]).float().to(device)
                em = model.encoder(mm); ew = model.encoder(ww); ed = model.encoder(dd)
                z = torch.cat([em, ew, ed], dim=1)
                # baseline
                if args.enable_multitask:
                    h = model.cls_head[2](model.cls_head[1](model.cls_head[0](z)))
                    base_logits_list.append(model.cls_head[3](h).detach().cpu().numpy().reshape(-1))
                base_reg_list.append(model.regressor(z).detach().cpu().numpy().reshape(-1))
                # zero variants
                z_m0 = torch.cat([torch.zeros_like(em), ew, ed], dim=1)
                z_w0 = torch.cat([em, torch.zeros_like(ew), ed], dim=1)
                z_d0 = torch.cat([em, ew, torch.zeros_like(ed)], dim=1)
                for zvar, Llist, Rlist in [
                    (z_m0, zm_log, zm_reg), (z_w0, zw_log, zw_reg), (z_d0, zd_log, zd_reg)]:
                    if args.enable_multitask:
                        hvar = model.cls_head[2](model.cls_head[1](model.cls_head[0](zvar)))
                        Llist.append(model.cls_head[3](hvar).detach().cpu().numpy().reshape(-1))
                    Rlist.append(model.regressor(zvar).detach().cpu().numpy().reshape(-1))
        if base_logits_list:
            base_prob = 1.0/(1.0+np.exp(-np.concatenate(base_logits_list)))
            zm_prob = 1.0/(1.0+np.exp(-np.concatenate(zm_log)))
            zw_prob = 1.0/(1.0+np.exp(-np.concatenate(zw_log)))
            zd_prob = 1.0/(1.0+np.exp(-np.concatenate(zd_log)))
            ablation['head_mean_abs_delta_prob'] = {
                'mut': float(np.mean(np.abs(zm_prob - base_prob))),
                'wt': float(np.mean(np.abs(zw_prob - base_prob))),
                'diff': float(np.mean(np.abs(zd_prob - base_prob))),
            }
        base_reg = np.concatenate(base_reg_list)
        ablation['reg_mean_abs_delta'] = {
            'mut': float(np.mean(np.abs(np.concatenate(zm_reg) - base_reg))),
            'wt': float(np.mean(np.abs(np.concatenate(zw_reg) - base_reg))),
            'diff': float(np.mean(np.abs(np.concatenate(zd_reg) - base_reg))),
        }
        # 可视化与CSV
        try:
            fig = plt.figure(figsize=(7,4)); ax = plt.gca()
            names = ['mut','wt','diff']
            if 'head_mean_abs_delta_prob' in ablation:
                ax.bar(np.arange(3)-0.2, [ablation['head_mean_abs_delta_prob'][k] for k in names], width=0.4, label='head Δprob', color="#1f77b4", alpha=0.8)
            ax.bar(np.arange(3)+0.2, [ablation['reg_mean_abs_delta'][k] for k in names], width=0.4, label='reg Δpred', color="#c7ccd4", alpha=0.8)
            ax.set_xticks(np.arange(3)); ax.set_xticklabels(names)
            ax.set_title('Branch ablation sensitivity'); ax.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'branch_ablation.png'), dpi=300); plt.close()
            rows = []
            for k in ['mut','wt','diff']:
                rows.append({'branch': k,
                             'head_delta_prob_mean_abs': ablation.get('head_mean_abs_delta_prob', {}).get(k, float('nan')),
                             'reg_delta_mean_abs': ablation['reg_mean_abs_delta'][k]})
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'branch_ablation.csv'), index=False)
        except Exception:
            pass

    # 汇总保存
    summary = {
        'exchange_pearson': ex_pearson,
        'exchange_mae_negflip': ex_mae,
        'head_auc': head_auc,
        'head_ap': head_ap,
        'ablation': ablation,
    }
    with open(os.path.join(out_dir, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def train_one_seed(args, seed: int, logger: logging.Logger) -> Dict[str, dict]:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    X, df = load_dataset_arrays(args.data_dir)
    labels = df["label_diff"].values.astype(np.float32)
    
    # 根据验证模式选择不同的训练策略
    if args.use_kfold:
        # K折交叉验证
        splits = get_kfold_splits(df, args.kfold_splits, seed, args.restrict)
        fold_results = []
        
        logger.info(f"[SEED {seed}] 开始 {args.kfold_splits} 折交叉验证")
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"[SEED {seed}] 训练第 {fold_idx + 1}/{args.kfold_splits} 折")
            fold_result = train_one_fold(args, seed, fold_idx, train_idx, val_idx, X, df, labels, logger)
            fold_results.append(fold_result)
            
            # 保存每折的模型
            fold_dir = os.path.join(args.output_dir, f"seed_{seed}", f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(fold_result["best_state"]["model"], os.path.join(fold_dir, "model_siamese.pth"))
            
            # 保存标准化器
            with open(os.path.join(fold_dir, "scaler_m.pkl"), "wb") as f:
                pickle.dump(fold_result["best_state"]["scaler_m"], f)
            with open(os.path.join(fold_dir, "scaler_w.pkl"), "wb") as f:
                pickle.dump(fold_result["best_state"]["scaler_w"], f)
            with open(os.path.join(fold_dir, "scaler_d.pkl"), "wb") as f:
                pickle.dump(fold_result["best_state"]["scaler_d"], f)
            
            with open(os.path.join(fold_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(fold_result["metrics"], f, indent=2, ensure_ascii=False)
        
        # 计算K折平均结果
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'r2', 'pearson']:
            values = [fold["metrics"][metric] for fold in fold_results if not np.isnan(fold["metrics"][metric])]
            if values:
                avg_metrics[metric] = float(np.mean(values))
                avg_metrics[f"{metric}_std"] = float(np.std(values))
            else:
                avg_metrics[metric] = float("nan")
                avg_metrics[f"{metric}_std"] = float("nan")
        
        logger.info(f"[SEED {seed}] K折平均结果: RMSE={avg_metrics.get('rmse', 'nan'):.4f}±{avg_metrics.get('rmse_std', 'nan'):.4f}")
        return {"metrics": avg_metrics, "fold_results": fold_results}
        
    elif args.use_three_split:
        # 三分割：训练-验证-测试
        split_loaded = False
        if getattr(args, 'split_file', None):
            try:
                with open(args.split_file, 'r', encoding='utf-8') as f:
                    sp = json.load(f)
                tr_idx = np.array(sp.get('train_idx'), dtype=int)
                va_idx = np.array(sp.get('val_idx'), dtype=int)
                te_idx = np.array(sp.get('test_idx'), dtype=int)
                assert tr_idx is not None and va_idx is not None and te_idx is not None
                split_loaded = True
                logger.info(f"[SEED {seed}] Loaded split from {args.split_file} (three-split)")
            except Exception as e:
                logger.warning(f"[SEED {seed}] Failed to load split_file: {e}. Falling back to on-the-fly split.")
        if not split_loaded:
            if args.restrict:
                tr_idx, va_idx, te_idx = protein_group_three_split(df, args.val_ratio, args.test_ratio, seed)
            else:
                tr_idx, va_idx, te_idx = three_way_split(len(df), args.val_ratio, args.test_ratio, seed)
        
        logger.info(f"[SEED {seed}] 三分割: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")

        # 保存实际使用的划分
        out_seed = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(out_seed, exist_ok=True)
        with open(os.path.join(out_seed, 'split_info.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'three_split_group' if args.restrict else 'three_split_random',
                'seed': int(seed),
                'val_ratio': float(args.val_ratio),
                'test_ratio': float(args.test_ratio),
                'restrict': bool(args.restrict),
                'train_idx': [int(x) for x in tr_idx.tolist()],
                'val_idx': [int(x) for x in va_idx.tolist()],
                'test_idx': [int(x) for x in te_idx.tolist()],
            }, f, indent=2, ensure_ascii=False)
        
        # 训练模型（使用训练+验证集）
        train_result = train_one_fold(args, seed, 0, tr_idx, va_idx, X, df, labels, logger)
        
        # 在验证/测试集上评估（并为ROC出图准备原始数据）
        mut, wt, diff = split_mut_wt_diff(X)
        m_va, w_va, d_va, y_va = mut[va_idx], wt[va_idx], diff[va_idx], labels[va_idx]
        m_te, w_te, d_te, y_te = mut[te_idx], wt[te_idx], diff[te_idx], labels[te_idx]
        
        # 使用训练好的标准化器和SVD变换器
        best_state = train_result["best_state"]
        
        # 如果训练时使用了SVD，需要先对验证/测试集应用相同的SVD变换
        if hasattr(args, 'svd_dim') and args.svd_dim and args.svd_dim > 0:
            # 需要重新获取训练时的SVD变换器
            # 重新训练SVD（基于训练集）
            m_tr_for_svd, w_tr_for_svd, d_tr_for_svd = mut[tr_idx], wt[tr_idx], diff[tr_idx]
            svd_m = TruncatedSVD(n_components=args.svd_dim, random_state=seed).fit(m_tr_for_svd)
            svd_w = TruncatedSVD(n_components=args.svd_dim, random_state=seed).fit(w_tr_for_svd)
            svd_d = TruncatedSVD(n_components=args.svd_dim, random_state=seed).fit(d_tr_for_svd)
            # 验证/测试SVD变换
            m_va = svd_m.transform(m_va); w_va = svd_w.transform(w_va); d_va = svd_d.transform(d_va)
            m_te = svd_m.transform(m_te); w_te = svd_w.transform(w_te); d_te = svd_d.transform(d_te)
        
        # 然后应用标准化（验证/测试）
        m_va = best_state["scaler_m"].transform(m_va)
        w_va = best_state["scaler_w"].transform(w_va)
        d_va = best_state["scaler_d"].transform(d_va)
        m_te = best_state["scaler_m"].transform(m_te)
        w_te = best_state["scaler_w"].transform(w_te)
        d_te = best_state["scaler_d"].transform(d_te)
        
        # 加载最佳模型进行测试
        in_dim = m_te.shape[1]
        hidden_dims = [int(x) for x in args.hidden.split(',')]
        # 若best_state包含分类头权重，则启用多任务头以避免load_state_dict键不匹配
        state_keys = list(best_state["model"].keys())
        has_cls_head = any(k.startswith("cls_head.") for k in state_keys)
        test_model = SiameseRegressor(in_dim, hidden=hidden_dims, fusion_dim=max(128, hidden_dims[-1]), dropout=args.dropout,
                                     enable_multitask=has_cls_head).to(device)
        test_model.load_state_dict(best_state["model"])
        test_model.eval()
        
        # 验证/测试数据集
        pid_series = df["uniprot_id"] if "uniprot_id" in df.columns else df["UNIPROT_ID"]
        val_ds = TrioDataset(m_va, w_va, d_va, y_va, pid_series.iloc[va_idx].tolist())
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_ds = TrioDataset(m_te, w_te, d_te, y_te, pid_series.iloc[te_idx].tolist())
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        test_preds, test_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                m = batch["mut"].to(device)
                w = batch["wt"].to(device)
                d = batch["diff"].to(device)
                y = batch["y"].to(device)
                p, _ = test_model(m, w, d)
                test_preds.append(p.detach().cpu().numpy().reshape(-1))
                test_true.append(y.cpu().numpy().reshape(-1))
        
        test_preds = np.concatenate(test_preds)
        test_true = np.concatenate(test_true)
        test_metrics = {
            "test_rmse": float(math.sqrt(mean_squared_error(test_true, test_preds))),
            "test_mae": float(mean_absolute_error(test_true, test_preds)),
            "test_r2": float(r2_score(test_true, test_preds)),
            "test_pearson": float(pearsonr(test_true, test_preds)[0]) if len(test_true) > 1 else 0.0,
        }
        # 若模型含分类头，计算验证/测试 AUC-ROC 并保存原始数据
        try:
            if has_cls_head:
                # 验证集 logits / 概率
                logits_list_v, y_bin_v = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        m = batch["mut"].to(device); w = batch["wt"].to(device); d = batch["diff"].to(device)
                        yb = batch["y"].to(device)
                        _, logits = test_model(m, w, d)
                        if logits is not None:
                            logits_list_v.append(logits.detach().cpu().numpy().reshape(-1))
                        y_bin_v.append((yb.detach().cpu().numpy().reshape(-1) > 0).astype(int))
                if logits_list_v:
                    logits_all_v = np.concatenate(logits_list_v); ybin_all_v = np.concatenate(y_bin_v)
                    probs_all_v = 1.0 / (1.0 + np.exp(-logits_all_v))
                    with open(os.path.join(out_seed, "auc_raw_val.json"), "w", encoding="utf-8") as f:
                        json.dump({"y_true_binary": ybin_all_v.astype(int).tolist(), "probs": probs_all_v.astype(float).tolist()}, f, indent=2, ensure_ascii=False)

                # 测试集 logits / 概率
                logits_list = []
                with torch.no_grad():
                    for batch in test_loader:
                        m = batch["mut"].to(device); w = batch["wt"].to(device); d = batch["diff"].to(device)
                        yb = batch["y"].to(device)
                        _, logits = test_model(m, w, d)
                        if logits is not None:
                            logits_list.append(logits.detach().cpu().numpy().reshape(-1))
                if logits_list:
                    logits_all = np.concatenate(logits_list)
                    ybin_all = (test_true > 0).astype(int)
                    probs_all = 1.0 / (1.0 + np.exp(-logits_all))
                    auc_te = float(roc_auc_score(ybin_all, probs_all))
                    test_metrics["test_auc_roc"] = auc_te
                    with open(os.path.join(out_seed, "auc_raw_test.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "y_true_binary": ybin_all.astype(int).tolist(),
                            "probs": probs_all.astype(float).tolist()
                        }, f, indent=2, ensure_ascii=False)
                    logger.info(f"[SEED {seed}] 测试集 AUC-ROC: {auc_te:.4f}")
        except Exception:
            pass
        
        # 合并验证和测试结果
        combined_metrics = {**train_result["metrics"], **test_metrics}
        logger.info(f"[SEED {seed}] 测试集结果: RMSE={test_metrics['test_rmse']:.4f}, R2={test_metrics['test_r2']:.4f}")
        
        # 保存模型和结果
        out_seed = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(out_seed, exist_ok=True)
        torch.save(best_state["model"], os.path.join(out_seed, "model_siamese.pth"))
        with open(os.path.join(out_seed, "scaler_m.pkl"), "wb") as f:
            pickle.dump(best_state["scaler_m"], f)
        with open(os.path.join(out_seed, "scaler_w.pkl"), "wb") as f:
            pickle.dump(best_state["scaler_w"], f)
        with open(os.path.join(out_seed, "scaler_d.pkl"), "wb") as f:
            pickle.dump(best_state["scaler_d"], f)
        with open(os.path.join(out_seed, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(train_result["metrics"], f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_seed, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        
        return {"metrics": combined_metrics, "val_metrics": train_result["metrics"], "test_metrics": test_metrics}
    
    else:
        # 原来的单次分割模式
        split_loaded = False
        if getattr(args, 'split_file', None):
            try:
                with open(args.split_file, 'r', encoding='utf-8') as f:
                    sp = json.load(f)
                tr_idx = np.array(sp.get('train_idx'), dtype=int)
                va_idx = np.array(sp.get('val_idx'), dtype=int)
                assert tr_idx is not None and va_idx is not None
                split_loaded = True
                logger.info(f"[SEED {seed}] Loaded split from {args.split_file} (train/val)")
            except Exception as e:
                logger.warning(f"[SEED {seed}] Failed to load split_file: {e}. Falling back to on-the-fly split.")
        if not split_loaded:
            if args.restrict:
                tr_idx, va_idx = protein_group_split(df, args.val_ratio, seed)
            else:
                tr_idx, va_idx = random_split(len(df), args.val_ratio, seed)
        
        # 使用原来的训练逻辑
        train_result = train_one_fold(args, seed, 0, tr_idx, va_idx, X, df, labels, logger)
        
        # 保存模型和结果（保持原来的格式）
        out_seed = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(out_seed, exist_ok=True)

        # 保存实际使用的划分（train/val）
        with open(os.path.join(out_seed, 'split_info.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'mode': 'group_split' if args.restrict else 'random_split',
                'seed': int(seed),
                'val_ratio': float(args.val_ratio),
                'restrict': bool(args.restrict),
                'train_idx': [int(x) for x in tr_idx.tolist()],
                'val_idx': [int(x) for x in va_idx.tolist()],
            }, f, indent=2, ensure_ascii=False)
        best_state = train_result["best_state"]
        torch.save(best_state["model"], os.path.join(out_seed, "model_siamese.pth"))
        with open(os.path.join(out_seed, "scaler_m.pkl"), "wb") as f:
            pickle.dump(best_state["scaler_m"], f)
        with open(os.path.join(out_seed, "scaler_w.pkl"), "wb") as f:
            pickle.dump(best_state["scaler_w"], f)
        with open(os.path.join(out_seed, "scaler_d.pkl"), "wb") as f:
            pickle.dump(best_state["scaler_d"], f)
        with open(os.path.join(out_seed, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(train_result["metrics"], f, indent=2, ensure_ascii=False)
        # Posthoc: 使用训练期 SVD/Scaler 与验证集索引，自动生成图与指标
        mut_all, wt_all, diff_all = split_mut_wt_diff(X)
        va_idx_local = train_result["best_state"].get("val_indices", None)
        if va_idx_local is not None:
            m_va_raw = mut_all[va_idx_local]
            w_va_raw = wt_all[va_idx_local]
            d_va_raw = diff_all[va_idx_local]
            y_va_raw = labels[va_idx_local]
            hidden_dims = [int(x) for x in args.hidden.split(',')]
            analysis_dir = os.path.join(out_seed, 'analysis')
            _posthoc_analysis(analysis_dir, args, train_result["best_state"],
                              m_va_raw, w_va_raw, d_va_raw, y_va_raw, hidden_dims,
                              device)
        
        logger.info(f"[SEED {seed}] Finished. Best epoch: {train_result.get('best_epoch')} metrics: {train_result.get('metrics', {})}")
        return {"metrics": train_result.get("metrics", {}), "best_epoch": int(train_result.get("best_epoch", -1))}


def aggregate_and_save(all_seeds: Dict[int, Dict[str, dict]], out_dir: str) -> None:
    def agg(key: str, prefix: str = "") -> Tuple[float, float]:
        vals = []
        for mwrap in all_seeds.values():
            met = mwrap.get("metrics", {})
            full_key = f"{prefix}{key}" if prefix else key
            if full_key in met:
                vals.append(met[full_key])
        vals = [v for v in vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    # 检测是否有K折或三分割结果
    has_kfold = any("fold_results" in seed_data for seed_data in all_seeds.values())
    has_test = any("test_metrics" in seed_data for seed_data in all_seeds.values())
    
    summary = {
        "seeds": list(all_seeds.keys()),
        "per_seed": {},
        "mean_std": {},
        "timestamp": int(time.time()),
    }
    
    # 处理每个种子的结果
    for k, wrap in all_seeds.items():
        seed_summary = {"metrics": {kk: float(vv) for kk, vv in wrap.get("metrics", {}).items()}}
        if "best_epoch" in wrap:
            seed_summary["best_epoch"] = int(wrap.get("best_epoch", -1))
        if "fold_results" in wrap:
            seed_summary["fold_results"] = wrap["fold_results"]
        if "val_metrics" in wrap:
            seed_summary["val_metrics"] = wrap["val_metrics"]
        if "test_metrics" in wrap:
            seed_summary["test_metrics"] = wrap["test_metrics"]
        summary["per_seed"][int(k)] = seed_summary
    
    # 计算汇总统计
    if has_test:
        # 三分割模式：分别汇总验证和测试结果
        summary["mean_std"]["validation"] = {
            "rmse": agg("rmse"),
            "mae": agg("mae"),
            "r2": agg("r2"),
            "pearson": agg("pearson"),
        }
        summary["mean_std"]["test"] = {
            "rmse": agg("rmse", "test_"),
            "mae": agg("mae", "test_"),
            "r2": agg("r2", "test_"),
            "pearson": agg("pearson", "test_"),
        }
    elif has_kfold:
        # K折模式：可能有标准差信息
        for metric in ["rmse", "mae", "r2", "pearson"]:
            summary["mean_std"][metric] = agg(metric)
            # 如果有K折标准差，也记录
            std_key = f"{metric}_std"
            if any(std_key in seed_data.get("metrics", {}) for seed_data in all_seeds.values()):
                summary["mean_std"][f"{metric}_kfold_std"] = agg(std_key)
    else:
        # 标准模式
        summary["mean_std"] = {
            "rmse": agg("rmse"),
            "mae": agg("mae"),
            "r2": agg("r2"),
            "pearson": agg("pearson"),
        }
    
    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser("Siamese tower training with exchange-consistency")
    
    # 基础配置
    p.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    p.add_argument("--output_dir", type=str, default=os.path.join("outputs", "dl", "siamese"))
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--restrict", action="store_true", help="基于蛋白质分组分割数据")
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--seeds", type=str, default=None, help="逗号分隔的随机种子列表，覆盖默认 [42,123,456]")
    p.add_argument("--use_gpu", action="store_true")
    
    # 训练超参数（保留最佳配置的默认值）
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden", type=str, default="1024,512", help="塔的隐藏层配置，如 '1024,512'")
    p.add_argument("--early_stopping_patience", type=int, default=30)
    
    # 最佳配置的固定参数（保留可调性但设为最佳默认值）
    p.add_argument("--huber_delta", type=float, default=1.8, help="Huber损失的delta参数")
    p.add_argument("--exchange_consistency", action="store_true", help="启用交换一致性约束")
    p.add_argument("--consistency_weight", type=float, default=0.1, help="交换一致性损失权重")
    p.add_argument("--label_standardize", action="store_true", help="启用标签标准化")
    p.add_argument("--svd_dim", type=int, default=384, help="SVD降维维度，0表示不降维")
    # 多任务分类（预测 label_diff 正/负）
    p.add_argument("--enable_multitask", action="store_true", help="启用多任务分类头（预测正/负）")
    p.add_argument("--cls_weight", type=float, default=0.2, help="分类损失权重（总损失=回归+cls_weight*分类）")
    
    # 调度器参数（最佳配置使用cosine）
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"], help="学习率调度器")
    p.add_argument("--warmup_epochs", type=int, default=8, help="预热轮数")
    
    # 验证模式（新增功能）
    p.add_argument("--use_kfold", action="store_true", help="使用K折交叉验证而非单次分割")
    p.add_argument("--kfold_splits", type=int, default=5, help="K折交叉验证的折数")
    p.add_argument("--use_three_split", action="store_true", help="使用训练-验证-测试三分割")
    p.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例（当使用三分割时）")
    # 固定划分
    p.add_argument("--split_file", type=str, default=None, help="可选：从JSON加载固定划分(train/val[/test])")
    # 快照配置（层间表征随训练演化）
    p.add_argument("--snapshot_interval", type=int, default=0, help="每多少个epoch保存一次层间表征快照（0禁用）")
    p.add_argument("--snapshot_branch", type=str, default="diff", choices=["mut","wt","diff"], help="快照分支")
    p.add_argument("--snapshot_max", type=int, default=2000, help="每次快照的最大样本数")
    # 分支消融
    p.add_argument("--use_diff_only", action="store_true", help="仅使用 diff 分支（将 mut/wt 分支置零）")
    p.add_argument("--use_mut_only", action="store_true", help="仅使用 mut 分支（将 wt/diff 分支置零）")
    p.add_argument("--use_wt_only", action="store_true", help="仅使用 wt 分支（将 mut/diff 分支置零）")
    
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # setup logger
    logger = logging.getLogger("siex")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(args.output_dir, "train.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # record args
    with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2, ensure_ascii=False)

    if getattr(args, "seeds", None):
        try:
            seeds = [int(x.strip()) for x in str(args.seeds).split(',') if str(x).strip() != ""]
        except Exception:
            raise ValueError("--seeds 需为以逗号分隔的整数列表，例如: 42,123,456")
    else:
        seeds = [42, 123, 456][: max(1, args.num_seeds)]
    
    # 验证模式描述
    mode_desc = "标准验证"
    if args.use_kfold:
        mode_desc = f"{args.kfold_splits}折交叉验证"
    elif args.use_three_split:
        mode_desc = "训练-验证-测试三分割"
    
    print("=== Siamese + Exchange 训练 ===")
    print(f"数据: {args.data_dir}")
    print(f"输出: {args.output_dir}")
    print(f"验证模式: {mode_desc}")
    print(f"种子: {seeds}")
    print(f"设备: {'CUDA' if (torch.cuda.is_available() and args.use_gpu) else 'CPU'}")
    
    logger.info(f"Start run. data={args.data_dir}, out={args.output_dir}, mode={mode_desc}, seeds={seeds}, device={'CUDA' if (torch.cuda.is_available() and args.use_gpu) else 'CPU'}")

    per_seed: Dict[int, Dict[str, dict]] = {}
    for sd in seeds:
        print(f"\n---- Seed {sd} ----")
        logger.info(f"==== SEED {sd} ====")
        m = train_one_seed(args, sd, logger)
        per_seed[sd] = m

    aggregate_and_save(per_seed, args.output_dir)
    print("\n完成。汇总已保存到 summary_metrics.json")
    logger.info("Summary saved to summary_metrics.json")


if __name__ == "__main__":
    main()


