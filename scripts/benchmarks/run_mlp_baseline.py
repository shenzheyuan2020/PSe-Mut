#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shallow MLP baseline (sklearn MLPRegressor), using fixed splits and three-branch transformation.

Test command:
  python benchmarks/run_mlp_baseline.py \
    --data_dir siamese_dataset \
    --split_file siamese_dataset/split_info.json \
    --out_dir bench_results/mlp \
    --svd_dim 384 --seed 42
"""
import argparse
import os
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score

_CUR = os.path.dirname(os.path.abspath(__file__))
import importlib.util
common_path = os.path.join(_CUR, 'common.py')
spec = importlib.util.spec_from_file_location('bench_common_local', common_path)
bench_common = importlib.util.module_from_spec(spec)  # type: ignore
assert spec is not None and spec.loader is not None
spec.loader.exec_module(bench_common)  # type: ignore


def main():
    ap = argparse.ArgumentParser("mlp baseline")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--split_file", type=str, default=os.path.join("data", "processed", "siamese", "split_info.json"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "benchmarks", "latest", "mlp"))
    ap.add_argument("--svd_dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, df, y = bench_common.load_dataset(args.data_dir)
    sp = bench_common.load_split(args.split_file)
    tr, va = sp["train_idx"], sp["val_idx"]
    te = sp.get("test_idx")

    Xm, Xw, Xd = bench_common.split_triple(X)
    (svds, scalers, (m_tr, w_tr, d_tr)) = bench_common.fit_branch_svd_scaler(Xm[tr], Xw[tr], Xd[tr], args.svd_dim, args.seed)
    m_va, w_va, d_va = bench_common.apply_branch_svd_scaler(Xm[va], Xw[va], Xd[va], svds, scalers)

    Xtr = bench_common.concat_three(m_tr, w_tr, d_tr)
    Xva = bench_common.concat_three(m_va, w_va, d_va)
    ytr, yva = y[tr], y[va]

    results = {}

    mlp = MLPRegressor(hidden_layer_sizes=(512, 256), activation='relu', solver='adam', alpha=1e-4,
                       learning_rate='adaptive', max_iter=300, random_state=args.seed)
    mlp.fit(Xtr, ytr)
    pred = mlp.predict(Xva)
    results['mlp_sklearn'] = bench_common.metrics_reg(yva, pred)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_mlp_sklearn.json'))
    except Exception:
        pass

    # AUC-ROC on validation (use MLP scores)
    try:
        ybin_va = (yva > 0).astype(int)
        mlp_scores_va = mlp.predict(Xva)
        auc_va = float(roc_auc_score(ybin_va, mlp_scores_va))
        results['auc_roc'] = auc_va
        bench_common.save_json({
            'y_true_binary': ybin_va.astype(int).tolist(),
            'scores': mlp_scores_va.astype(float).tolist()
        }, os.path.join(args.out_dir, 'auc_raw_val.json'))
        print(f"[VAL] AUC-ROC (MLP): {auc_va:.4f}")
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    bench_common.save_json(results, os.path.join(args.out_dir, 'val_metrics.json'))

    # Optional: test set evaluation
    if te is not None:
        m_te, w_te, d_te = bench_common.apply_branch_svd_scaler(Xm[te], Xw[te], Xd[te], svds, scalers)
        Xte = bench_common.concat_three(m_te, w_te, d_te)
        yte = y[te]
        try:
            pr = mlp.predict(Xte)
            tm = {'mlp_sklearn': bench_common.metrics_reg(yte, pr)}
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': pr.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_mlp_sklearn.json'))
            bench_common.save_json(tm, os.path.join(args.out_dir, 'test_metrics.json'))
        except Exception:
            pass
    print(results)

    # Visualization: MLP prediction vs true
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        mlp_pred = pred
        lim = float(max(_np.percentile(_np.abs(yva), 99), _np.percentile(_np.abs(mlp_pred), 99)))
        plt.figure(figsize=(5,5))
        plt.scatter(yva, mlp_pred, s=6, alpha=0.5)
        plt.plot([-lim, lim], [-lim, lim], 'r--')
        plt.xlabel('True'); plt.ylabel('Pred'); plt.title('MLP Pred vs True'); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'mlp_pred_vs_true.png'), dpi=200); plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
