#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成并冻结孪生数据集的划分索引，便于深度模型与传统ML共享相同的 train/val(/test) 或 K折。

用法示例：
  python tools/freeze_splits.py --data_dir siamese_dataset --mode group --val_ratio 0.2 --seed 42
  python tools/freeze_splits.py --data_dir siamese_dataset --mode group3 --val_ratio 0.2 --test_ratio 0.2 --seed 42
  python tools/freeze_splits.py --data_dir siamese_dataset --mode kfold_group --k 5 --seed 42
输出：在 data_dir 下保存 split_info.json 或 kfold_splits.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path（使得可从 tools/ 导入兄弟目录 dl_trainers/）
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CUR_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from dl_trainers.train_siamese_exchange import (
    protein_group_split,
    protein_group_three_split,
    get_kfold_splits,
)


def main():
    ap = argparse.ArgumentParser("freeze fixed splits for siamese dataset")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--mode", type=str, default="group", choices=[
        "random",        # train/val 随机划分
        "group",         # train/val 蛋白分组划分
        "random3",       # train/val/test 随机三分割
        "group3",        # train/val/test 分组三分割
        "kfold_random",  # K折随机
        "kfold_group",   # K折分组
    ])
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    csv_file = os.path.join(args.data_dir, "siamese_pairs.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} 不存在")
    df = pd.read_csv(csv_file)

    out_path = os.path.join(args.data_dir, "split_info.json")

    if args.mode in {"random", "group"}:
        if args.mode == "group":
            tr, va = protein_group_split(df, args.val_ratio, args.seed)
            mode_tag = "group_split"
        else:
            # 兼容：用 np.random 实现随机二分
            idx = np.arange(len(df))
            rng = np.random.RandomState(args.seed)
            rng.shuffle(idx)
            v = int(len(df) * args.val_ratio)
            va, tr = idx[:v], idx[v:]
            mode_tag = "random_split"
        obj = {
            "mode": mode_tag,
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "train_idx": [int(x) for x in tr.tolist()],
            "val_idx": [int(x) for x in va.tolist()],
        }
        json.dump(obj, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("saved", out_path)
        return

    if args.mode in {"random3", "group3"}:
        if args.mode == "group3":
            tr, va, te = protein_group_three_split(df, args.val_ratio, args.test_ratio, args.seed)
            mode_tag = "three_split_group"
        else:
            idx = np.arange(len(df))
            rng = np.random.RandomState(args.seed)
            rng.shuffle(idx)
            test_size = int(len(df) * args.test_ratio)
            val_size = int(len(df) * args.val_ratio)
            train_size = len(df) - test_size - val_size
            tr = idx[:train_size]
            va = idx[train_size:train_size+val_size]
            te = idx[train_size+val_size:]
            mode_tag = "three_split_random"
        obj = {
            "mode": mode_tag,
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "train_idx": [int(x) for x in tr.tolist()],
            "val_idx": [int(x) for x in va.tolist()],
            "test_idx": [int(x) for x in te.tolist()],
        }
        json.dump(obj, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("saved", out_path)
        return

    # K 折
    restrict = (args.mode == "kfold_group")
    splits = get_kfold_splits(df, args.k, args.seed, restrict)
    out_path = os.path.join(args.data_dir, "kfold_splits.json")
    payload = {
        "mode": "kfold_group" if restrict else "kfold_random",
        "k": int(args.k),
        "seed": int(args.seed),
        "folds": [
            {
                "train_idx": [int(x) for x in tr.tolist()],
                "val_idx": [int(x) for x in va.tolist()],
            }
            for (tr, va) in splits
        ],
    }
    json.dump(payload, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()
