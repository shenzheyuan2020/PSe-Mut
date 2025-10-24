#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common baseline tools: load data/splits, three-branch SVD/Scaler transformation, metrics and saving."""
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_dataset(data_dir: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    csv_file = os.path.join(data_dir, "siamese_pairs.csv")
    npy_file = os.path.join(data_dir, "siamese_fingerprints.npy")
    if not (os.path.exists(csv_file) and os.path.exists(npy_file)):
        raise FileNotFoundError(f"Missing data: {csv_file} or {npy_file}")
    df = pd.read_csv(csv_file)
    X = np.load(npy_file)
    if len(df) != X.shape[0]:
        raise ValueError("CSV and NPY row counts are inconsistent")
    y = df["label_diff"].values.astype(np.float32)
    return X, df, y


def load_split(split_file: str) -> Dict[str, np.ndarray]:
    with open(split_file, 'r', encoding='utf-8') as f:
        sp = json.load(f)
    out = {}
    for k in ["train_idx", "val_idx", "test_idx"]:
        if k in sp and sp[k] is not None:
            out[k] = np.array(sp[k], dtype=int)
    return out


def split_triple(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[1] % 3 != 0:
        raise ValueError("Feature dimension is not a multiple of 3")
    d = X.shape[1] // 3
    return X[:, :d], X[:, d:2*d], X[:, 2*d:3*d]


def fit_branch_svd_scaler(Xm: np.ndarray, Xw: np.ndarray, Xd: np.ndarray, target_dim: int, seed: int):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    svd_m = svd_w = svd_d = None
    if target_dim and target_dim > 0:
        svd_m = TruncatedSVD(n_components=target_dim, random_state=seed).fit(Xm)
        svd_w = TruncatedSVD(n_components=target_dim, random_state=seed).fit(Xw)
        svd_d = TruncatedSVD(n_components=target_dim, random_state=seed).fit(Xd)
        Xm = svd_m.transform(Xm); Xw = svd_w.transform(Xw); Xd = svd_d.transform(Xd)
    sc_m = StandardScaler().fit(Xm)
    sc_w = StandardScaler().fit(Xw)
    sc_d = StandardScaler().fit(Xd)
    Xm = sc_m.transform(Xm); Xw = sc_w.transform(Xw); Xd = sc_d.transform(Xd)
    return (svd_m, svd_w, svd_d), (sc_m, sc_w, sc_d), (Xm, Xw, Xd)


def apply_branch_svd_scaler(Xm: np.ndarray, Xw: np.ndarray, Xd: np.ndarray, svds, scalers):
    svd_m, svd_w, svd_d = svds
    sc_m, sc_w, sc_d = scalers
    if svd_m is not None:
        Xm = svd_m.transform(Xm)
        Xw = svd_w.transform(Xw)
        Xd = svd_d.transform(Xd)
    Xm = sc_m.transform(Xm)
    Xw = sc_w.transform(Xw)
    Xd = sc_d.transform(Xd)
    return Xm, Xw, Xd


def concat_three(Xm: np.ndarray, Xw: np.ndarray, Xd: np.ndarray) -> np.ndarray:
    return np.concatenate([Xm, Xw, Xd], axis=1)


def metrics_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    import math
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



