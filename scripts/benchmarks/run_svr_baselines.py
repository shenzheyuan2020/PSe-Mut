#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR baselines (linear/RBF), using fixed splits and three-branch transformation.

Test command:
  python benchmarks/run_svr_baselines.py \
    --data_dir siamese_dataset \
    --split_file siamese_dataset/split_info.json \
    --out_dir bench_results/svr \
    --svd_dim 384 --seed 42
"""
import argparse
import os
import sys
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score

_CUR = os.path.dirname(os.path.abspath(__file__))
import importlib.util
common_path = os.path.join(_CUR, 'common.py')
spec = importlib.util.spec_from_file_location('bench_common_local', common_path)
bench_common = importlib.util.module_from_spec(spec)  # type: ignore
assert spec is not None and spec.loader is not None
spec.loader.exec_module(bench_common)  # type: ignore


def main():
    ap = argparse.ArgumentParser("svr baselines")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--split_file", type=str, default=os.path.join("data", "processed", "siamese", "split_info.json"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "benchmarks", "latest", "svr"))
    ap.add_argument("--svd_dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample", type=float, default=1.0, help="Training set subsampling ratio (0,1], accelerate SVR")
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

    # Subsampling acceleration
    if 0 < args.subsample < 1.0:
        import numpy as _np
        rng = _np.random.RandomState(args.seed)
        idx = _np.arange(Xtr.shape[0])
        rng.shuffle(idx)
        k = max(1000, int(len(idx) * args.subsample))
        idx = idx[:k]
        Xtr, ytr = Xtr[idx], ytr[idx]

    results = {}

    # SVR linear
    svr_lin = SVR(kernel='linear', C=1.0)
    svr_lin.fit(Xtr, ytr)
    pred = svr_lin.predict(Xva)
    results['svr_linear'] = bench_common.metrics_reg(yva, pred)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_svr_linear.json'))
    except Exception:
        pass

    # SVR RBF
    svr_rbf = SVR(kernel='rbf', C=10.0, gamma='scale')
    svr_rbf.fit(Xtr, ytr)
    pred = svr_rbf.predict(Xva)
    results['svr_rbf'] = bench_common.metrics_reg(yva, pred)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_svr_rbf.json'))
    except Exception:
        pass

    # AUC-ROC on validation (use RBF SVR scores)
    try:
        ybin_va = (yva > 0).astype(int)
        rbf_scores_va = svr_rbf.predict(Xva)
        auc_va = float(roc_auc_score(ybin_va, rbf_scores_va))
        results['auc_roc'] = auc_va
        bench_common.save_json({
            'y_true_binary': ybin_va.astype(int).tolist(),
            'scores': rbf_scores_va.astype(float).tolist()
        }, os.path.join(args.out_dir, 'auc_raw_val.json'))
        print(f"[VAL] AUC-ROC (SVR-RBF): {auc_va:.4f}")
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    bench_common.save_json(results, os.path.join(args.out_dir, 'val_metrics.json'))

    # Optional: test set evaluation
    if te is not None:
        m_te, w_te, d_te = bench_common.apply_branch_svd_scaler(Xm[te], Xw[te], Xd[te], svds, scalers)
        Xte = bench_common.concat_three(m_te, w_te, d_te)
        yte = y[te]
        test_results = {}
        try:
            pr = svr_lin.predict(Xte); test_results['svr_linear'] = bench_common.metrics_reg(yte, pr)
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': pr.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_svr_linear.json'))
        except Exception:
            pass
        try:
            pr = svr_rbf.predict(Xte); test_results['svr_rbf'] = bench_common.metrics_reg(yte, pr)
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': pr.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_svr_rbf.json'))
        except Exception:
            pass
        # AUC-ROC on test (RBF SVR)
        try:
            ybin_te = (yte > 0).astype(int)
            rbf_scores_te = svr_rbf.predict(Xte)
            auc_te = float(roc_auc_score(ybin_te, rbf_scores_te))
            test_results['auc_roc'] = auc_te
            bench_common.save_json({
                'y_true_binary': ybin_te.astype(int).tolist(),
                'scores': rbf_scores_te.astype(float).tolist()
            }, os.path.join(args.out_dir, 'auc_raw_test.json'))
            print(f"[TEST] AUC-ROC (SVR-RBF): {auc_te:.4f}")
        except Exception:
            pass
        bench_common.save_json(test_results, os.path.join(args.out_dir, 'test_metrics.json'))
    print(results)

    # Brief visualization: RBF prediction vs true
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        rbf_pred = pred
        lim = float(max(_np.percentile(_np.abs(yva), 99), _np.percentile(_np.abs(rbf_pred), 99)))
        plt.figure(figsize=(5,5))
        plt.scatter(yva, rbf_pred, s=6, alpha=0.5)
        plt.plot([-lim, lim], [-lim, lim], 'r--')
        plt.xlabel('True'); plt.ylabel('Pred'); plt.title('SVR RBF Pred vs True'); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'svr_rbf_pred_vs_true.png'), dpi=200); plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
