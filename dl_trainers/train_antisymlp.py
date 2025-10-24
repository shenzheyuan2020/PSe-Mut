#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anti-symmetric MLP trainer on siamese_dataset.

Inputs:
  - siamese_dataset/siamese_pairs.csv
  - siamese_dataset/siamese_fingerprints.npy

Features:
  - Transform [mut, wt, diff] (3D) -> [s, a] where s=0.5(mut+wt), a=0.5(mut-wt)
  - Optional exchange augmentation (swap mut/wt => flip sign of a and label)
  - Protein-aware split (group by uniprot_id)
  - Multi-seed training with early stopping
  - Metrics: RMSE, MAE, R2, Pearson

Usage (examples):
  python dl_trainers/train_antisymlp.py --data_dir siamese_dataset --output_dir dl_results_antisymlp
  python dl_trainers/train_antisymlp.py --restrict --num_seeds 3 --exchange_augment
"""

import os
import json
import math
import time
import random
import argparse
import pickle
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


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
        raise FileNotFoundError(f"缺少数据文件: {csv_file} 或 {npy_file}")
    df = pd.read_csv(csv_file)
    X = np.load(npy_file)
    if len(df) != X.shape[0]:
        raise ValueError(f"CSV行数({len(df)})与指纹数量({X.shape[0]})不匹配")
    return X, df


def to_antisymlp_features(X: np.ndarray) -> np.ndarray:
    """[mut, wt, diff] -> [s, a] with s=0.5(mut+wt), a=0.5(mut-wt)."""
    if X.shape[1] % 3 != 0:
        raise ValueError("特征维度不是3的倍数，无法进行anti-sym转换")
    d = X.shape[1] // 3
    mut = X[:, :d]
    wt = X[:, d:2*d]
    # diff = X[:, 2*d:3*d]  # 不直接使用
    s = 0.5 * (mut + wt)
    a = 0.5 * (mut - wt)
    return np.concatenate([s, a], axis=1)


class AntiSymDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, uniprot_ids: List[str]):
        self.X = features.astype(np.float32)
        self.y = labels.astype(np.float32)
        self.uniprot_ids = list(uniprot_ids)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.X[idx]),
            "y": torch.tensor([self.y[idx]], dtype=torch.float32),
            "uniprot": self.uniprot_ids[idx],
        }


def protein_group_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Protein-aware split: ensure no leakage across train/val by uniprot groups."""
    if "uniprot_id" in df.columns:
        pid_col = "uniprot_id"
    elif "UNIPROT_ID" in df.columns:
        pid_col = "UNIPROT_ID"
    else:
        raise KeyError("CSV缺少uniprot_id列")

    groups = df.groupby(pid_col).indices  # dict: pid -> index list
    all_pids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(all_pids)

    total = len(df)
    target_val = int(total * val_ratio)
    val_pids = []
    val_count = 0
    for pid in all_pids:
        idxs = groups[pid]
        if val_count < target_val:
            val_pids.append(pid)
            val_count += len(idxs)

    val_indices = np.concatenate([groups[pid] for pid in val_pids]).astype(int)
    train_mask = np.ones(total, dtype=bool)
    train_mask[val_indices] = False
    train_indices = np.arange(total)[train_mask]
    return train_indices, val_indices


def random_split(total: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    val_size = int(total * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0.0, None)
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": float(pr)}


def train_one_seed(args, seed: int) -> Dict[str, float]:
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # 1) Load & transform
    X_raw, df = load_dataset_arrays(args.data_dir)
    X = to_antisymlp_features(X_raw)
    labels = df["label_diff"].values.astype(np.float32)
    pid_series = df["uniprot_id"] if "uniprot_id" in df.columns else df["UNIPROT_ID"]

    # 2) Split
    if args.restrict:
        tr_idx, va_idx = protein_group_split(df, args.val_ratio, seed)
    else:
        tr_idx, va_idx = random_split(len(df), args.val_ratio, seed)

    Xtr, ytr = X[tr_idx], labels[tr_idx]
    Xva, yva = X[va_idx], labels[va_idx]
    ptr_tr = pid_series.iloc[tr_idx].tolist()
    ptr_va = pid_series.iloc[va_idx].tolist()

    # 3) Standardize on train only
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    train_ds = AntiSymDataset(Xtr, ytr, ptr_tr)
    val_ds = AntiSymDataset(Xva, yva, ptr_va)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4) Model
    input_dim = X.shape[1]
    model = MLPRegressor(input_dim=input_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    def make_exchange(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Anti-sym features are [s, a]; swap mut/wt => a -> -a, s unchanged; y -> -y
        d2 = x.shape[1] // 2
        s, a = x[:, :d2], x[:, d2:]
        x_swap = torch.cat([s, -a], dim=1)
        y_swap = -y
        return x_swap, y_swap

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)

            if args.exchange_augment:
                xb_swap, yb_swap = make_exchange(xb, yb)
                xb = torch.cat([xb, xb_swap], dim=0)
                yb = torch.cat([yb, yb_swap], dim=0)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds) * (2.0 if args.exchange_augment else 1.0)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                xb = batch["x"].to(device)
                yb = batch["y"].to(device)
                pred = model(xb)
                val_preds.append(pred.cpu().numpy().reshape(-1))
                val_true.append(yb.cpu().numpy().reshape(-1))

        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        metrics = evaluate_metrics(val_true, val_preds)

        print(f"Epoch {epoch:03d} | TrainLoss {epoch_loss:.4f} | Val RMSE {metrics['rmse']:.4f} R2 {metrics['r2']:.4f}")

        # Early stopping on RMSE
        if metrics["rmse"] + 1e-6 < best_val:
            best_val = metrics["rmse"]
            # ensure pure python floats for JSON
            metrics_native = {k: float(v) for k, v in metrics.items()}
            best_state = {
                "epoch": epoch,
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "scaler": scaler,
                "metrics": metrics_native,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                print("早停触发")
                break

    # Save artifacts per seed
    out_seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(out_seed_dir, exist_ok=True)
    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "scaler": scaler,
            "metrics": {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "pearson": float("nan")},
        }
    torch.save(best_state["model"], os.path.join(out_seed_dir, "model_antisymlp.pth"))
    with open(os.path.join(out_seed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(best_state["scaler"], f)
    with open(os.path.join(out_seed_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in best_state.get("metrics", {}).items()}, f, indent=2, ensure_ascii=False)

    return best_state.get("metrics", {})


def aggregate_and_save(all_metrics: Dict[int, Dict[str, float]], out_dir: str) -> None:
    # Compute mean/std across seeds
    def agg(key: str) -> Tuple[float, float]:
        vals = [m.get(key, float("nan")) for m in all_metrics.values() if key in m]
        vals = [v for v in vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    # build mean/std dict with pure floats
    rmse_mean, rmse_std = agg("rmse")
    mae_mean, mae_std = agg("mae")
    r2_mean, r2_std = agg("r2")
    pearson_mean, pearson_std = agg("pearson")

    summary = {
        "seeds": list(all_metrics.keys()),
        "per_seed": {int(k): {kk: float(vv) for kk, vv in m.items()} for k, m in all_metrics.items()},
        "mean_std": {
            "rmse": {"mean": float(rmse_mean), "std": float(rmse_std)},
            "mae": {"mean": float(mae_mean), "std": float(mae_std)},
            "r2": {"mean": float(r2_mean), "std": float(r2_std)},
            "pearson": {"mean": float(pearson_mean), "std": float(pearson_std)},
        },
        "timestamp": int(time.time()),
    }
    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser("Anti-sym MLP training on siamese_dataset")
    p.add_argument("--data_dir", type=str, default="siamese_dataset")
    p.add_argument("--output_dir", type=str, default="dl_results_antisymlp")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--restrict", action="store_true", help="按蛋白分组划分")
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--early_stopping_patience", type=int, default=20)
    p.add_argument("--exchange_augment", action="store_true")
    p.add_argument("--use_gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = [42, 123, 456][: max(1, args.num_seeds)]
    print("=== Anti-sym MLP 训练 ===")
    print(f"数据: {args.data_dir} \n输出: {args.output_dir} \n种子: {seeds} \n设备: {'CUDA' if (torch.cuda.is_available() and args.use_gpu) else 'CPU'}")

    per_seed_metrics: Dict[int, Dict[str, float]] = {}
    for sd in seeds:
        print(f"\n---- Seed {sd} ----")
        m = train_one_seed(args, sd)
        per_seed_metrics[sd] = m

    aggregate_and_save(per_seed_metrics, args.output_dir)
    print("\n完成。汇总已保存到 summary_metrics.json")


if __name__ == "__main__":
    main()


