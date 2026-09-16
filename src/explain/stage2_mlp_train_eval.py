"""Stage 2 Run B -- MLP generalization gate (section 5/6 of the checkpoint).

Trains the fixed (16,8) MLP with lot-grouped manual early stopping, then
evaluates: train RMSE, grouped 5-fold CV RMSE/MAE/R^2 (fold stability),
holdout RMSE/MAE/R^2, train-vs-validation gap, and a LightGBM sanity-check
comparison. Also runs the target-shuffle negative control. This is the
MANDATORY gate before any relationship-discovery work (single-parameter,
2-way, 3-way) is attempted -- if the MLP overfits severely or the negative
control does not collapse, that is reported here and used to decide
whether to proceed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_mlp_common import (
    ALPHA, HIDDEN_LAYERS, LEARNING_RATE_INIT, MAX_EPOCHS, PATIENCE, RANDOM_STATE,
    count_trainable_params, fit_mlp_grouped_early_stopping, mae, predict_scaled, r2, rmse,
)
from src.explain.stage2_v2_common import R, load_locked_split
from src.models.trees import make_lightgbm


def main():
    split = load_locked_split()
    ds = split.ds
    X_train = split.X_train.to_numpy(dtype=np.float64)
    y_train = split.y_train.to_numpy(dtype=np.float64)
    X_test = split.X_test.to_numpy(dtype=np.float64)
    y_test = split.y_test.to_numpy(dtype=np.float64)
    groups_train = split.groups_train.to_numpy()
    n_inputs = X_train.shape[1]
    n_train = X_train.shape[0]
    n_lots = split.groups_train.nunique()
    n_params = count_trainable_params(HIDDEN_LAYERS, n_inputs)

    print("=" * 70)
    print("STAGE 2 RUN B -- MLP ARCHITECTURE & SAMPLE-SIZE ACCOUNTING")
    print("=" * 70)
    print(f"Predictors (input dim): {n_inputs} (1,405 numeric + 20 categorical dummies)")
    print(f"Training rows: {n_train}  |  Unique lots: {n_lots}  |  Holdout rows: {len(y_test)}")
    print(f"Hidden layers: {HIDDEN_LAYERS}  |  Trainable parameters: {n_params}")
    print(f"Sample-to-parameter ratio: {n_train / n_params:.4f} ({n_train} rows / {n_params} params)")
    print(f"alpha (L2)={ALPHA}, learning_rate_init={LEARNING_RATE_INIT}, "
          f"max_epochs={MAX_EPOCHS}, patience={PATIENCE}, random_state={RANDOM_STATE}")

    # ---- Grouped 5-fold CV (reusing the SAME folds as every other Stage 2 method) ----
    print("\nRunning grouped 5-fold CV...")
    cv_rows = []
    t0 = time.time()
    for fold_i, (tr, va) in enumerate(split.folds):
        fit = fit_mlp_grouped_early_stopping(X_train[tr], y_train[tr], groups_train[tr])
        p_tr = predict_scaled(fit, X_train[tr])
        p_va = predict_scaled(fit, X_train[va])
        cv_rows.append({
            "fold": fold_i, "best_epoch": fit.best_epoch, "n_epochs_run": fit.n_epochs_run,
            "train_rmse": rmse(y_train[tr], p_tr), "val_rmse": rmse(y_train[va], p_va),
            "val_mae": mae(y_train[va], p_va), "val_r2": r2(y_train[va], p_va),
        })
        print(f"  fold {fold_i}: best_epoch={fit.best_epoch} train_rmse={cv_rows[-1]['train_rmse']:.4f} "
              f"val_rmse={cv_rows[-1]['val_rmse']:.4f} val_r2={cv_rows[-1]['val_r2']:.4f}", flush=True)
    cv_df = pd.DataFrame(cv_rows)
    print(f"CV done in {time.time()-t0:.1f}s")

    cv_val_rmse_mean, cv_val_rmse_std = cv_df["val_rmse"].mean(), cv_df["val_rmse"].std()
    cv_val_mae_mean = cv_df["val_mae"].mean()
    cv_val_r2_mean, cv_val_r2_std = cv_df["val_r2"].mean(), cv_df["val_r2"].std()
    cv_train_rmse_mean = cv_df["train_rmse"].mean()
    train_val_gap = cv_val_rmse_mean - cv_train_rmse_mean

    # ---- Final model: fit on the FULL training set (own internal grouped early-stopping split) ----
    print("\nFitting final model on the full training set (internal grouped early-stopping split)...")
    final_fit = fit_mlp_grouped_early_stopping(X_train, y_train, groups_train)
    train_pred_full = predict_scaled(final_fit, X_train)
    train_rmse_full = rmse(y_train, train_pred_full)
    holdout_pred = predict_scaled(final_fit, X_test)
    holdout_rmse = rmse(y_test, holdout_pred)
    holdout_mae = mae(y_test, holdout_pred)
    holdout_r2 = r2(y_test, holdout_pred)
    print(f"Final model: best_epoch={final_fit.best_epoch}/{final_fit.n_epochs_run}, "
          f"full-train RMSE={train_rmse_full:.4f}")
    print(f"Holdout: RMSE={holdout_rmse:.4f} MAE={holdout_mae:.4f} R^2={holdout_r2:.4f}")

    # ---- LightGBM sanity-check comparison (same locked split) ----
    print("\nFitting LightGBM (same locked Run B split) as a sanity-check comparison only...")
    gbm = make_lightgbm(random_state=RANDOM_STATE)
    gbm.fit(split.X_train, split.y_train)
    gbm_holdout_pred = gbm.predict(split.X_test)
    gbm_holdout_rmse = rmse(y_test, gbm_holdout_pred)
    gbm_holdout_r2 = r2(y_test, gbm_holdout_pred)
    print(f"LightGBM holdout: RMSE={gbm_holdout_rmse:.4f} R^2={gbm_holdout_r2:.4f}")

    # ---- overfitting assessment ----
    overfit_ratio = train_val_gap / cv_train_rmse_mean if cv_train_rmse_mean > 0 else np.inf
    if cv_val_r2_mean < 0:
        overfit_verdict = "SEVERE (grouped-CV R^2 is negative -- worse than predicting the mean)"
    elif overfit_ratio > 0.5:
        overfit_verdict = "SUBSTANTIAL (train-to-validation RMSE gap exceeds 50% of train RMSE)"
    elif overfit_ratio > 0.2:
        overfit_verdict = "MODERATE"
    else:
        overfit_verdict = "MILD/ACCEPTABLE"
    print(f"\nOverfitting assessment: {overfit_verdict} (train-val gap={train_val_gap:.4f}, "
          f"ratio-to-train-rmse={overfit_ratio:.3f})")

    # ---- target-shuffle negative control (grouped CV only, cheaper than full re-run) ----
    print("\nRunning target-shuffle negative control (grouped 5-fold CV)...")
    rng = np.random.RandomState(RANDOM_STATE)
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)
    shuf_rows = []
    for fold_i, (tr, va) in enumerate(split.folds):
        fit = fit_mlp_grouped_early_stopping(X_train[tr], y_shuf[tr], groups_train[tr])
        p_va = predict_scaled(fit, X_train[va])
        shuf_rows.append({"fold": fold_i, "val_rmse": rmse(y_shuf[va], p_va), "val_r2": r2(y_shuf[va], p_va)})
    shuf_df = pd.DataFrame(shuf_rows)
    shuf_r2_mean = shuf_df["val_r2"].mean()
    print(f"Shuffled-target grouped-CV R^2: mean={shuf_r2_mean:.4f} (real={cv_val_r2_mean:.4f})")

    negctrl_pass = shuf_r2_mean < 0.05
    print(f"Negative control: {'PASS -- collapsed toward/below 0' if negctrl_pass else 'FAIL -- substantial residual predictive power under shuffle'}")

    # ---- save metrics ----
    metrics_row = {
        "n_inputs": n_inputs, "n_train_rows": n_train, "n_unique_lots": n_lots, "n_holdout_rows": len(y_test),
        "hidden_layers": str(HIDDEN_LAYERS), "trainable_params": n_params,
        "sample_to_param_ratio": round(n_train / n_params, 4),
        "alpha_l2": ALPHA, "learning_rate_init": LEARNING_RATE_INIT, "random_state": RANDOM_STATE,
        "final_best_epoch": final_fit.best_epoch, "final_epochs_run": final_fit.n_epochs_run,
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
        "negative_control_pass": negctrl_pass,
    }
    out_path = f"{R}/stage2_runB_mlp_model_metrics.csv"
    pd.DataFrame([metrics_row]).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    cv_df.to_csv(f"{R}/_stage2_mlp_cv_fold_detail.csv", index=False)

    with open(f"{R}/_stage2_mlp_gate_decision.txt", "w") as f:
        f.write(f"overfit_verdict={overfit_verdict}\n")
        f.write(f"cv_val_r2_mean={cv_val_r2_mean:.4f}\n")
        f.write(f"negative_control_pass={negctrl_pass}\n")
        proceed = negctrl_pass and cv_val_r2_mean > -0.2
        f.write(f"proceed_to_relationship_discovery={proceed}\n")
    print(f"\nGATE DECISION: proceed_to_relationship_discovery = {negctrl_pass and cv_val_r2_mean > -0.2}")


if __name__ == "__main__":
    main()
