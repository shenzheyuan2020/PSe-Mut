#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear regression/Ridge regression/ElasticNet baselines (using fixed splits and three-branch transformation).

Test command (example):
  python benchmarks/run_linear_baselines.py \
    --data_dir siamese_dataset \
    --split_file siamese_dataset/split_info.json \
    --out_dir bench_results/linear \
    --svd_dim 384 --seed 42
"""
import argparse
import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, LassoCV, BayesianRidge, HuberRegressor, SGDRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score

# Force import benchmarks/common.py from local file to avoid being overridden by system packages with the same name
_CUR = os.path.dirname(os.path.abspath(__file__))
import importlib.util
common_path = os.path.join(_CUR, 'common.py')
spec = importlib.util.spec_from_file_location('bench_common_local', common_path)
bench_common = importlib.util.module_from_spec(spec)  # type: ignore
assert spec is not None and spec.loader is not None
spec.loader.exec_module(bench_common)  # type: ignore


def main():
    ap = argparse.ArgumentParser("linear baselines")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--split_file", type=str, default=os.path.join("data", "processed", "siamese", "split_info.json"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "benchmarks", "latest", "linear"))
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

    # Linear Regression
    lr = LinearRegression()
    lr.fit(Xtr, ytr)
    pred = lr.predict(Xva)
    results['linear'] = bench_common.metrics_reg(yva, pred)
    # AUC raw (val)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_linear.json'))
    except Exception:
        pass

    # Ridge (CV)
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 25), cv=5)
    ridge.fit(Xtr, ytr)
    pred = ridge.predict(Xva)
    results['ridge'] = bench_common.metrics_reg(yva, pred)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_ridge.json'))
    except Exception:
        pass

    # ElasticNet (CV)
    enet = ElasticNetCV(l1_ratio=[.1,.5,.7,.9,1.0], alphas=None, cv=5, random_state=args.seed, n_jobs=-1)
    enet.fit(Xtr, ytr)
    pred = enet.predict(Xva)
    results['elasticnet'] = bench_common.metrics_reg(yva, pred)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_elasticnet.json'))
    except Exception:
        pass

    # Lasso (CV)
    lasso = LassoCV(alphas=None, cv=5, random_state=args.seed, n_jobs=-1)
    lasso.fit(Xtr, ytr)
    pred_lasso = lasso.predict(Xva)
    results['lasso'] = bench_common.metrics_reg(yva, pred_lasso)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred_lasso.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_lasso.json'))
    except Exception:
        pass

    # Bayesian Ridge
    bayes = BayesianRidge()
    bayes.fit(Xtr, ytr)
    pred_bayes = bayes.predict(Xva)
    results['bayesian_ridge'] = bench_common.metrics_reg(yva, pred_bayes)
    try:
        ybin_va = (yva > 0).astype(int)
        bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred_bayes.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_bayesian_ridge.json'))
    except Exception:
        pass

    # HuberRegressor (robust)
    huber = HuberRegressor(alpha=1e-4)
    huber.fit(Xtr, ytr)
    pred_huber = huber.predict(Xva)
    results['huber'] = bench_common.metrics_reg(yva, pred_huber)

    # SGDRegressor (elastic net)
    sgd = SGDRegressor(loss='huber', penalty='elasticnet', l1_ratio=0.15, alpha=1e-4, random_state=args.seed, max_iter=2000)
    sgd.fit(Xtr, ytr)
    pred_sgd = sgd.predict(Xva)
    results['sgd_enet'] = bench_common.metrics_reg(yva, pred_sgd)

    # AUC-ROC on validation (using ElasticNet scores)
    try:
        ybin_va = (yva > 0).astype(int)
        enet_scores_va = enet.predict(Xva)
        auc_va = float(roc_auc_score(ybin_va, enet_scores_va))
        results['auc_roc'] = auc_va
        bench_common.save_json({
            'y_true_binary': ybin_va.astype(int).tolist(),
            'scores': enet_scores_va.astype(float).tolist()
        }, os.path.join(args.out_dir, 'auc_raw_val.json'))
        print(f"[VAL] AUC-ROC (ElasticNet): {auc_va:.4f}")
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    bench_common.save_json(results, os.path.join(args.out_dir, 'val_metrics.json'))

    # Optional: evaluate on test set if present
    if te is not None:
        m_te, w_te, d_te = bench_common.apply_branch_svd_scaler(Xm[te], Xw[te], Xd[te], svds, scalers)
        Xte = bench_common.concat_three(m_te, w_te, d_te)
        yte = y[te]

        test_results = {}
        try:
            test_results['linear'] = bench_common.metrics_reg(yte, lr.predict(Xte))
            # raw
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': lr.predict(Xte).astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_linear.json'))
        except Exception:
            pass
        try:
            test_results['ridge'] = bench_common.metrics_reg(yte, ridge.predict(Xte))
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': ridge.predict(Xte).astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_ridge.json'))
        except Exception:
            pass
        try:
            test_results['elasticnet'] = bench_common.metrics_reg(yte, enet.predict(Xte))
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': enet.predict(Xte).astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_elasticnet.json'))
        except Exception:
            pass
        try:
            test_results['lasso'] = bench_common.metrics_reg(yte, lasso.predict(Xte))
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': lasso.predict(Xte).astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_lasso.json'))
        except Exception:
            pass
        try:
            test_results['bayesian_ridge'] = bench_common.metrics_reg(yte, bayes.predict(Xte))
            ybin_te = (yte > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': bayes.predict(Xte).astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_bayesian_ridge.json'))
        except Exception:
            pass
        try:
            test_results['huber'] = bench_common.metrics_reg(yte, huber.predict(Xte))
        except Exception:
            pass
        try:
            test_results['sgd_enet'] = bench_common.metrics_reg(yte, sgd.predict(Xte))
        except Exception:
            pass
        # AUC-ROC on test (using ElasticNet scores)
        try:
            ybin_te = (yte > 0).astype(int)
            enet_scores_te = enet.predict(Xte)
            auc_te = float(roc_auc_score(ybin_te, enet_scores_te))
            test_results['auc_roc'] = auc_te
            bench_common.save_json({
                'y_true_binary': ybin_te.astype(int).tolist(),
                'scores': enet_scores_te.astype(float).tolist()
            }, os.path.join(args.out_dir, 'auc_raw_test.json'))
            print(f"[TEST] AUC-ROC (ElasticNet): {auc_te:.4f}")
        except Exception:
            pass
        bench_common.save_json(test_results, os.path.join(args.out_dir, 'test_metrics.json'))
    print(results)

    # Visualization: residual histogram + prediction-actual scatter plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Using ElasticNet as example
        enet_pred = pred
        resid = yva - enet_pred
        plt.figure(figsize=(5,4))
        plt.hist(resid, bins=50, color="#1f77b4", alpha=0.8)
        plt.title('ElasticNet Residuals'); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'elasticnet_residuals.png'), dpi=200); plt.close()

        import numpy as _np
        lim = float(max(_np.percentile(_np.abs(yva), 99), _np.percentile(_np.abs(enet_pred), 99)))
        plt.figure(figsize=(5,5))
        plt.scatter(yva, enet_pred, s=6, alpha=0.5)
        plt.plot([-lim, lim], [-lim, lim], 'r--')
        plt.xlabel('True'); plt.ylabel('Pred'); plt.title('ElasticNet Pred vs True'); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'elasticnet_pred_vs_true.png'), dpi=200); plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
