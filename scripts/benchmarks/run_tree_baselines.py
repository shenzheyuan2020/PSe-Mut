#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree model baselines (Random Forest, XGBoost), using fixed splits and three-branch transformation.

Test command:
  python benchmarks/run_tree_baselines.py \
    --data_dir siamese_dataset \
    --split_file siamese_dataset/split_info.json \
    --out_dir bench_results/tree \
    --svd_dim 384 --seed 42
"""
import argparse
import os
import sys
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

_CUR = os.path.dirname(os.path.abspath(__file__))
import importlib.util
common_path = os.path.join(_CUR, 'common.py')
spec = importlib.util.spec_from_file_location('bench_common_local', common_path)
bench_common = importlib.util.module_from_spec(spec)  # type: ignore
assert spec is not None and spec.loader is not None
spec.loader.exec_module(bench_common)  # type: ignore


def main():
    ap = argparse.ArgumentParser("tree baselines")
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed", "siamese"))
    ap.add_argument("--split_file", type=str, default=os.path.join("data", "processed", "siamese", "split_info.json"))
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "benchmarks", "latest", "tree"))
    ap.add_argument("--svd_dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_xgb", action='store_true', help='Enable XGBoost (requires xgboost installation)')
    ap.add_argument("--only", type=str, default=None, choices=["xgb", "rf", "extratrees"],
                    help='Run only a specific tree method (overrides others)')
    ap.add_argument("--use_lgb", action='store_true', help='Enable LightGBM (requires lightgbm installation)')
    ap.add_argument("--fast", action='store_true', help='Fast mode: fewer trees, shallower depth, subsampling')
    ap.add_argument("--rf_estimators", type=int, default=None, help='Override RF tree count')
    ap.add_argument("--rf_max_depth", type=int, default=None, help='Override RF max depth')
    ap.add_argument("--rf_min_samples_leaf", type=int, default=None, help='Override RF min leaf samples')
    ap.add_argument("--subsample", type=float, default=1.0, help="Training set subsampling ratio (0,1], accelerate tree models")
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
        k = max(2000, int(len(idx) * args.subsample))
        idx = idx[:k]
        Xtr, ytr = Xtr[idx], ytr[idx]

    results = {}

    # Determine which methods to run
    run_rf = (args.only is None or args.only == 'rf')
    run_et = (args.only is None or args.only == 'extratrees')
    run_xgb = (args.only == 'xgb') or (args.only is None and args.use_xgb)

    # Random Forest (with fast mode/controllable parameters)
    n_estimators = 1000
    max_depth = None
    min_leaf = 1
    if args.fast:
        n_estimators = 300
        max_depth = 16
        min_leaf = 5
    if args.rf_estimators is not None:
        n_estimators = args.rf_estimators
    if args.rf_max_depth is not None:
        max_depth = args.rf_max_depth
    if args.rf_min_samples_leaf is not None:
        min_leaf = args.rf_min_samples_leaf

    if run_rf:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_leaf=min_leaf, random_state=args.seed, n_jobs=-1,
                                   max_features='sqrt', bootstrap=True)
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xva)
        results['rf'] = bench_common.metrics_reg(yva, pred)
        try:
            ybin_va = (yva > 0).astype(int)
            bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_rf.json'))
        except Exception:
            pass

    # ExtraTrees (extremely randomized trees)
    if run_et:
        try:
            et = ExtraTreesRegressor(n_estimators=n_estimators if n_estimators < 800 else 800,
                                     max_depth=max_depth, min_samples_leaf=min_leaf,
                                     random_state=args.seed, n_jobs=-1, max_features='sqrt')
            et.fit(Xtr, ytr)
            pred_et = et.predict(Xva)
            results['extratrees'] = bench_common.metrics_reg(yva, pred_et)
            try:
                ybin_va = (yva > 0).astype(int)
                bench_common.save_json({'y_true_binary': ybin_va.astype(int).tolist(), 'scores': pred_et.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_val_extratrees.json'))
            except Exception:
                pass
        except Exception as e:
            results['extratrees'] = {"error": str(e)}

    # XGBoost (optional or only)
    if run_xgb:
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(n_estimators=1200 if not args.fast else 500,
                               learning_rate=0.05 if not args.fast else 0.08,
                               subsample=0.8, colsample_bytree=0.8,
                               max_depth=7 if not args.fast else 6,
                               reg_lambda=1.0, random_state=args.seed, n_jobs=-1,
                               tree_method='hist')
            xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = xgb.predict(Xva)
            results['xgb'] = bench_common.metrics_reg(yva, pred)
        except Exception as e:
            results['xgb'] = {"error": str(e)}

    # LightGBM (optional)
    if args.use_lgb:
        try:
            import lightgbm as lgb
            lgbm = lgb.LGBMRegressor(
                n_estimators=2000 if not args.fast else 800,
                learning_rate=0.05 if not args.fast else 0.08,
                num_leaves=64 if not args.fast else 48,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=args.seed, n_jobs=-1,
                device_type='cpu'
            )
            lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='rmse', verbose=False,
                     callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
            pred = lgbm.predict(Xva)
            results['lgbm'] = bench_common.metrics_reg(yva, pred)
        except Exception as e:
            results['lgbm'] = {"error": str(e)}

    # AUC-ROC on validation (prefer RF, else XGB, else ExtraTrees)
    try:
        ybin_va = (yva > 0).astype(int)
        ref_scores_va = None
        model_tag = None
        if run_rf:
            ref_scores_va = rf.predict(Xva); model_tag = 'RF'
        elif run_xgb:
            ref_scores_va = pred; model_tag = 'XGB'
        elif run_et:
            ref_scores_va = pred_et; model_tag = 'ExtraTrees'
        if ref_scores_va is not None:
            auc_va = float(roc_auc_score(ybin_va, ref_scores_va))
            results['auc_roc'] = auc_va
            bench_common.save_json({
                'y_true_binary': ybin_va.astype(int).tolist(),
                'scores': [float(x) for x in ref_scores_va]
            }, os.path.join(args.out_dir, 'auc_raw_val.json'))
            print(f"[VAL] AUC-ROC ({model_tag}): {auc_va:.4f}")
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
        if run_rf:
            try:
                pr = rf.predict(Xte); test_results['rf'] = bench_common.metrics_reg(yte, pr)
                ybin_te = (yte > 0).astype(int)
                bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': pr.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_rf.json'))
            except Exception:
                pass
        if run_et:
            try:
                pr = et.predict(Xte); test_results['extratrees'] = bench_common.metrics_reg(yte, pr)
                ybin_te = (yte > 0).astype(int)
                bench_common.save_json({'y_true_binary': ybin_te.astype(int).tolist(), 'scores': pr.astype(float).tolist()}, os.path.join(args.out_dir, 'auc_raw_test_extratrees.json'))
            except Exception:
                pass
        if run_xgb:
            try:
                test_results['xgb'] = bench_common.metrics_reg(yte, xgb.predict(Xte))
            except Exception:
                pass
        if args.use_lgb:
            try:
                test_results['lgbm'] = bench_common.metrics_reg(yte, lgbm.predict(Xte))
            except Exception:
                pass
        # AUC-ROC on test
        try:
            ybin_te = (yte > 0).astype(int)
            ref_scores_te = None
            model_tag = None
            if run_rf:
                ref_scores_te = rf.predict(Xte); model_tag='RF'
            elif run_xgb:
                ref_scores_te = xgb.predict(Xte); model_tag='XGB'
            elif run_et:
                ref_scores_te = et.predict(Xte); model_tag='ExtraTrees'
            if ref_scores_te is not None:
                auc_te = float(roc_auc_score(ybin_te, ref_scores_te))
                test_results['auc_roc'] = auc_te
                bench_common.save_json({
                    'y_true_binary': ybin_te.astype(int).tolist(),
                    'scores': [float(x) for x in ref_scores_te]
                }, os.path.join(args.out_dir, 'auc_raw_test.json'))
                print(f"[TEST] AUC-ROC ({model_tag}): {auc_te:.4f}")
        except Exception:
            pass
        bench_common.save_json(test_results, os.path.join(args.out_dir, 'test_metrics.json'))
    print(results)

    # Visualization: RF prediction vs true
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        rf_pred = pred
        lim = float(max(_np.percentile(_np.abs(yva), 99), _np.percentile(_np.abs(rf_pred), 99)))
        plt.figure(figsize=(5,5))
        plt.scatter(yva, rf_pred, s=6, alpha=0.5)
        plt.plot([-lim, lim], [-lim, lim], 'r--')
        plt.xlabel('True'); plt.ylabel('Pred'); plt.title('RF Pred vs True'); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'rf_pred_vs_true.png'), dpi=200); plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
