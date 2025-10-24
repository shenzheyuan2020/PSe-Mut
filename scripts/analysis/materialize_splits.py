#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialize random splits into physical subsets: based on split_info.json, write siamese_pairs.csv / siamese_fingerprints.npy
by train/val(/test) to separate CSV and NPY files, ensuring consistent row order and indices with training.

Usage:
  python tools/materialize_splits.py \
    --data_dir siamese_dataset \
    --split_file siamese_dataset/split_info.json \
    --out_dir siamese_dataset_split_seed42
Output:
  out_dir/{train,val[,test]}/siamese_pairs.csv
  out_dir/{train,val[,test]}/siamese_fingerprints.npy
"""
import os
import json
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser("materialize splits")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--split_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    csv_file = os.path.join(args.data_dir, "siamese_pairs.csv")
    npy_file = os.path.join(args.data_dir, "siamese_fingerprints.npy")
    if not (os.path.exists(csv_file) and os.path.exists(npy_file)):
        raise FileNotFoundError("数据集文件缺失")
    df = pd.read_csv(csv_file)
    X = np.load(npy_file)
    if len(df) != X.shape[0]:
        raise ValueError("CSV 与 NPY 行数不一致")

    with open(args.split_file, 'r', encoding='utf-8') as f:
        sp = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    def dump(name: str, idx):
        if idx is None:
            return
        idx = np.array(idx, dtype=int)
        sub_dir = os.path.join(args.out_dir, name)
        os.makedirs(sub_dir, exist_ok=True)
        df_sub = df.iloc[idx].copy()
        X_sub = X[idx]
        df_sub.to_csv(os.path.join(sub_dir, 'siamese_pairs.csv'), index=False)
        np.save(os.path.join(sub_dir, 'siamese_fingerprints.npy'), X_sub)
        print(f"saved {name}: {df_sub.shape} -> {sub_dir}")

    dump('train', sp.get('train_idx'))
    dump('val', sp.get('val_idx'))
    dump('test', sp.get('test_idx'))


if __name__ == "__main__":
    main()



