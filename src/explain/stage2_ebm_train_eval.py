"""Stage 2 Run B EBM -- generalization gate + predictive-level negative
control (sections 6/7). Grouped 5-fold CV (same folds as every other
Stage 2 method), holdout evaluation, LightGBM sanity-check comparison, and
a full grouped-CV re-run under a shuffled target.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from src.explain.stage2_ebm_common import EBM_CONFIG, fit_ebm, mae, r2, rmse
from src.explain.stage2_v2_common import R, load_locked_split
from src.models.trees import make_lightgbm

CACHE_PATH = f"{R}/_stage2_ebm_fold_models.joblib"
RANDOM_STATE = 42


def main():
    split = load_locked_split()
    X_train, y_train = split.X_train, split.y_train.to_numpy(dtype=np.float64)
    X_test, y_test = split.X_test, split.y_test.to_numpy(dtype=np.float64)
    groups_train = split.groups_train.to_numpy()
    n_train, n_inputs = X_train.shape
    n_lots = split.groups_train.nunique()

    print("=" * 70)
    print("STAGE 2 RUN B -- EBM CONFIGURATION & SAMPLE-SIZE ACCOUNTING")
    print("=" * 70)
    print(f"Predictors: {n_inputs}  |  Training rows: {n_train}  |  Unique lots: {n_lots}  |  Holdout: {len(y_test)}")
    print(f"Config: {EBM_CONFIG}")

    print("\nRunning grouped 5-fold CV (real target)...")
    cv_rows = []
    fold_models = []
    t0 = time.time()
    for fold_i, (tr, va) in enumerate(split.folds):
        ebm = fit_ebm(X_train.iloc[tr], y_train[tr], groups_train[tr], random_state=RANDOM_STATE)
        p_tr = ebm.predict(X_train.iloc[tr])
        p_va = ebm.predict(X_train.iloc[va])
        cv_rows.append({
            "fold": fold_i, "train_rmse": rmse(y_train[tr], p_tr), "val_rmse": rmse(y_train[va], p_va),
            "val_mae": mae(y_train[va], p_va), "val_r2": r2(y_train[va], p_va),
        })
        fold_models.append({"tr": tr, "va": va, "ebm": ebm})
        print(f"  fold {fold_i}: train_rmse={cv_rows[-1]['train_rmse']:.4f} val_rmse={cv_rows[-1]['val_rmse']:.4f} "
              f"val_r2={cv_rows[-1]['val_r2']:.4f} ({time.time()-t0:.0f}s elapsed)", flush=True)
    cv_df = pd.DataFrame(cv_rows)
    print(f"Real-target CV done in {time.time()-t0:.1f}s")

    cv_val_rmse_mean, cv_val_rmse_std = cv_df["val_rmse"].mean(), cv_df["val_rmse"].std()
    cv_val_mae_mean = cv_df["val_mae"].mean()
    cv_val_r2_mean, cv_val_r2_std = cv_df["val_r2"].mean(), cv_df["val_r2"].std()
    cv_train_rmse_mean = cv_df["train_rmse"].mean()
    train_val_gap = cv_val_rmse_mean - cv_train_rmse_mean

    print("\nFitting final model on the full training set...")
    t0 = time.time()
    final_ebm = fit_ebm(X_train, y_train, groups_train, random_state=RANDOM_STATE)
    train_pred_full = final_ebm.predict(X_train)
    train_rmse_full = rmse(y_train, train_pred_full)
    holdout_pred = final_ebm.predict(X_test)
    holdout_rmse, holdout_mae, holdout_r2 = rmse(y_test, holdout_pred), mae(y_test, holdout_pred), r2(y_test, holdout_pred)
    print(f"Final model fit in {time.time()-t0:.1f}s -- full-train RMSE={train_rmse_full:.4f}")
    print(f"Holdout: RMSE={holdout_rmse:.4f} MAE={holdout_mae:.4f} R^2={holdout_r2:.4f}")

    print("\nFitting LightGBM (same locked split) as a sanity-check comparison only...")
    gbm = make_lightgbm(random_state=RANDOM_STATE)
    gbm.fit(X_train, split.y_train)
    gbm_holdout_pred = gbm.predict(X_test)
    gbm_holdout_rmse, gbm_holdout_r2 = rmse(y_test, gbm_holdout_pred), r2(y_test, gbm_holdout_pred)
    print(f"LightGBM holdout: RMSE={gbm_holdout_rmse:.4f} R^2={gbm_holdout_r2:.4f}")

    overfit_ratio = train_val_gap / cv_train_rmse_mean if cv_train_rmse_mean > 0 else np.inf
    if cv_val_r2_mean < 0:
        overfit_verdict = "SEVERE (grouped-CV R^2 is negative)"
    elif overfit_ratio > 0.5:
        overfit_verdict = "SUBSTANTIAL"
    elif overfit_ratio > 0.2:
        overfit_verdict = "MODERATE"
    else:
        overfit_verdict = "MILD/ACCEPTABLE"
    print(f"\nOverfitting assessment: {overfit_verdict} (train-val gap={train_val_gap:.4f}, ratio={overfit_ratio:.3f})")

    print("\nRunning target-shuffle negative control (full grouped 5-fold CV under shuffle)...")
    rng = np.random.RandomState(RANDOM_STATE)
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)
    shuf_rows = []
    shuf_fold_models = []
    t0 = time.time()
    for fold_i, (tr, va) in enumerate(split.folds):
        ebm_s = fit_ebm(X_train.iloc[tr], y_shuf[tr], groups_train[tr], random_state=RANDOM_STATE)
        p_va = ebm_s.predict(X_train.iloc[va])
        shuf_rows.append({"fold": fold_i, "val_rmse": rmse(y_shuf[va], p_va), "val_r2": r2(y_shuf[va], p_va)})
        shuf_fold_models.append({"tr": tr, "va": va, "ebm": ebm_s})
        print(f"  shuffled fold {fold_i}: val_r2={shuf_rows[-1]['val_r2']:.4f} ({time.time()-t0:.0f}s elapsed)", flush=True)
    shuf_df = pd.DataFrame(shuf_rows)
    shuf_r2_mean = shuf_df["val_r2"].mean()
    print(f"Shuffled-target CV R^2: mean={shuf_r2_mean:.4f} (real={cv_val_r2_mean:.4f})")
    negctrl_pass = shuf_r2_mean < 0.05

    joblib.dump({"fold_models": fold_models, "final_ebm": final_ebm, "shuf_fold_models": shuf_fold_models},
                CACHE_PATH)
    print(f"\nCached fold models, final model, and shuffled fold models to {CACHE_PATH}")

    metrics_row = {
        "n_inputs": n_inputs, "n_train_rows": n_train, "n_unique_lots": n_lots, "n_holdout_rows": len(y_test),
        **{f"cfg_{k}": v for k, v in EBM_CONFIG.items()},
        "train_rmse_full": round(train_rmse_full, 4),
        "cv_train_rmse_mean": round(cv_train_rmse_mean, 4),
        "cv_val_rmse_mean": round(cv_val_rmse_mean, 4), "cv_val_rmse_std": round(cv_val_rmse_std, 4),
        "cv_val_mae_mean": round(cv_val_mae_mean, 4),
        "cv_val_r2_mean": round(cv_val_r2_mean, 4), "cv_val_r2_std": round(cv_val_r2_std, 4),
        "train_val_gap_rmse": round(train_val_gap, 4), "overfit_ratio": round(overfit_ratio, 4),
        "overfit_verdict": overfit_verdict,
        "holdout_rmse": round(holdout_rmse, 4), "holdout_mae": round(holdout_mae, 4), "holdout_r2": round(holdout_r2, 4),
        "lightgbm_holdout_rmse_sanity_check": round(gbm_holdout_rmse, 4),
        "lightgbm_holdout_r2_sanity_check": round(gbm_holdout_r2, 4),
        "shuffled_target_cv_r2_mean": round(shuf_r2_mean, 4),
        "negative_control_predictive_pass": negctrl_pass,
    }
    pd.DataFrame([metrics_row]).to_csv(f"{R}/stage2_runB_ebm_model_metrics.csv", index=False)
    print(f"\nSaved {R}/stage2_runB_ebm_model_metrics.csv")
    print(f"\nGATE: negative_control_predictive_pass = {negctrl_pass}, cv_val_r2_mean = {cv_val_r2_mean:.4f}")


if __name__ == "__main__":
    main()
