"""Stage 2 Run B MLP -- section 7: model-agnostic global feature evidence.

Fits (and caches, for reuse by the later single-parameter/interaction
scripts in this checkpoint) the 5 grouped-CV fold models plus one
full-training-set model, all with the same lot-grouped manual early
stopping as the gate checkpoint. For every one of the 1,425 Run B
predictors, computes two model-agnostic importance signals on each fold's
own held-out validation rows (never training rows):
  - grouped permutation importance: shuffle the column within the fold's
    validation set, measure RMSE degradation
  - ablation effect: set the column to its TRAINING-set reference (mean)
    value for all validation rows, measure RMSE degradation
Both are averaged across the 5 folds; stability = how many of the 5 folds
show a genuine (positive) degradation when the feature is perturbed.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from src.explain.stage2_mlp_common import fit_mlp_grouped_early_stopping, predict_scaled, rmse
from src.explain.stage2_v2_common import R, load_locked_split

CACHE_PATH = f"{R}/_stage2_mlp_fold_models.joblib"
N_PERMUTATION_REPEATS = 2


def evidence_label(consistency: int, n_folds: int, mean_effect: float) -> str:
    if consistency >= 4 and mean_effect > 0:
        return "STRONG"
    if consistency == 3 and mean_effect > 0:
        return "MODERATE"
    if consistency in (1, 2):
        return "WEAK"
    return "INCONCLUSIVE"


def main():
    split = load_locked_split()
    ds = split.ds
    X_train = split.X_train.to_numpy(dtype=np.float64)
    y_train = split.y_train.to_numpy(dtype=np.float64)
    groups_train = split.groups_train.to_numpy()
    feature_cols = list(split.X_train.columns)
    n_features = len(feature_cols)
    orig_names = [ds.inverse_name_map.get(c, c) for c in feature_cols]
    train_means = X_train.mean(axis=0)

    print(f"Fitting/caching 5 fold models + 1 full-train model for MLP relationship discovery...")
    t0 = time.time()
    fold_models = []
    for fold_i, (tr, va) in enumerate(split.folds):
        fit = fit_mlp_grouped_early_stopping(X_train[tr], y_train[tr], groups_train[tr])
        fold_models.append({"tr": tr, "va": va, "fit": fit})
        print(f"  fold {fold_i} fitted (best_epoch={fit.best_epoch})", flush=True)
    full_fit = fit_mlp_grouped_early_stopping(X_train, y_train, groups_train)
    print(f"All models fit in {time.time()-t0:.1f}s")

    joblib.dump({"fold_models": fold_models, "full_fit": full_fit, "feature_cols": feature_cols,
                 "orig_names": orig_names, "train_means": train_means}, CACHE_PATH)
    print(f"Cached fitted models to {CACHE_PATH}")

    print(f"\nComputing grouped permutation importance + ablation for {n_features} features "
          f"across 5 folds ({N_PERMUTATION_REPEATS} permutation repeats each)...")
    rng = np.random.RandomState(42)

    # baseline RMSE per fold (unperturbed)
    baseline_rmse = []
    for fm in fold_models:
        p = predict_scaled(fm["fit"], X_train[fm["va"]])
        baseline_rmse.append(rmse(y_train[fm["va"]], p))

    perm_effects = np.zeros((5, n_features))
    ablation_effects = np.zeros((5, n_features))

    for fold_i, fm in enumerate(fold_models):
        va = fm["va"]
        X_va = X_train[va]
        y_va = y_train[va]
        fit = fm["fit"]
        for j in range(n_features):
            # permutation
            deltas = []
            for r in range(N_PERMUTATION_REPEATS):
                X_perm = X_va.copy()
                perm_idx = rng.permutation(len(va))
                X_perm[:, j] = X_perm[perm_idx, j]
                p_perm = predict_scaled(fit, X_perm)
                deltas.append(rmse(y_va, p_perm) - baseline_rmse[fold_i])
            perm_effects[fold_i, j] = np.mean(deltas)

            # ablation (set to training-set reference mean)
            X_abl = X_va.copy()
            X_abl[:, j] = train_means[j]
            p_abl = predict_scaled(fit, X_abl)
            ablation_effects[fold_i, j] = rmse(y_va, p_abl) - baseline_rmse[fold_i]
        print(f"  fold {fold_i} feature-evidence pass done ({time.time()-t0:.1f}s elapsed)...", flush=True)

    mean_perm = perm_effects.mean(axis=0)
    mean_abl = ablation_effects.mean(axis=0)
    consistency_perm = (perm_effects > 0).sum(axis=0)
    consistency_abl = (ablation_effects > 0).sum(axis=0)

    rows = []
    for j in range(n_features):
        label = evidence_label(int(consistency_perm[j]), 5, mean_perm[j])
        rows.append({
            "raw_csv_parameter_name": orig_names[j],
            "permutation_importance": round(float(mean_perm[j]), 5),
            "permutation_fold_consistency": f"{int(consistency_perm[j])}/5",
            "ablation_effect": round(float(mean_abl[j]), 5),
            "ablation_fold_consistency": f"{int(consistency_abl[j])}/5",
            "evidence_label": label,
        })
    df = pd.DataFrame(rows).sort_values("permutation_importance", ascending=False)
    out_path = f"{R}/stage2_runB_mlp_feature_evidence.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} predictors, 100% coverage)")
    print(df["evidence_label"].value_counts().to_string())
    print("\nTop 15 by permutation importance:")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
