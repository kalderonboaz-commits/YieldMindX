"""Stage 2 Run B v2, item 4 Stage B: expensive confirmation of the
broad-screen's top candidate pairs.

Takes the top-K pairs from stage2_runB_v2_pair_screen.csv (Stage A) and
confirms EACH SPECIFIC PAIR (not all pairs among the involved features)
using:
  - SHAP interaction values (extracted from one full interaction-matrix
    computation on the union of involved features -- a compact LightGBM
    model fit on exactly that union, matching the proven-safe pattern from
    earlier checkpoints: fitting a subset-column model avoids the
    dtype/scope segfault found previously)
  - Friedman's H-statistic, computed individually per candidate pair only
    (not a full pairwise matrix -- O(K), not O(features^2))
  - 5-fold GroupKFold out-of-fold reproducibility: for each candidate pair,
    is its SHAP-interaction value consistently in the top half of that
    fold's own interaction-value distribution?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.data.splits import make_group_kfold
from src.explain.interactions import compute_h_statistic
from src.explain.stage2_v2_common import R, load_locked_split
from src.models.trees import make_lightgbm

RANDOM_STATE = 42


class _Shim:
    def __init__(self, inv): self._inv = inv
    def to_original(self, c): return self._inv.get(c, c)


def main():
    split = load_locked_split()
    ds = split.ds
    shim = _Shim(ds.inverse_name_map)

    pool = pd.read_csv(f"{R}/stage2_runB_v2_pair_screen.csv")
    candidate_pairs = list(zip(pool["parameter_A"], pool["parameter_B"]))
    union_features = sorted(set(pool["parameter_A"]) | set(pool["parameter_B"]))
    print(f"Confirming {len(candidate_pairs)} candidate pairs from Stage A, "
          f"involving {len(union_features)} unique features")

    model_cols = [ds.name_map[f] for f in union_features]
    X_train_u = split.X_train[model_cols]
    X_test_u = split.X_test[model_cols]

    print("Fitting compact confirmation model on the union of candidate features (train portion)...")
    model = make_lightgbm(random_state=RANDOM_STATE)
    model.fit(X_train_u, split.y_train)

    print("Computing full SHAP interaction matrix on held-out test set (one computation, all pairs extracted)...")
    explainer = shap.TreeExplainer(model)
    X_sample = X_test_u.sample(min(300, len(X_test_u)), random_state=RANDOM_STATE)
    inter = explainer.shap_interaction_values(X_sample)
    mean_abs_inter = np.abs(inter).mean(axis=0)
    col_index = {c: i for i, c in enumerate(model_cols)}

    print("Computing H-statistic for each candidate pair individually (test set)...")
    X_bg = X_test_u.sample(min(60, len(X_test_u)), random_state=RANDOM_STATE).astype(np.float64)

    rows = []
    for a, b in candidate_pairs:
        ma, mb = ds.name_map[a], ds.name_map[b]
        shap_strength = float(mean_abs_inter[col_index[ma], col_index[mb]])
        try:
            h_strength = compute_h_statistic(model, X_bg, ma, mb, grid_resolution=6)
        except Exception:
            h_strength = np.nan
        rows.append({"parameter_A": a, "parameter_B": b, "broad_screen_score": None,
                      "shap_interaction_strength": shap_strength, "h_statistic_strength": h_strength})
    conf_df = pd.DataFrame(rows)
    conf_df = conf_df.merge(pool[["parameter_A", "parameter_B", "broad_screen_score"]],
                             on=["parameter_A", "parameter_B"], how="left", suffixes=("", "_orig"))
    conf_df["broad_screen_score"] = conf_df["broad_screen_score_orig"]
    conf_df = conf_df.drop(columns=["broad_screen_score_orig"])

    print("\nFitting 5 GroupKFold compact confirmation models (out-of-fold interaction reproducibility)...")
    _, folds = make_group_kfold(split.groups_train, n_splits=5)
    fold_above_median = {p: 0 for p in zip(conf_df["parameter_A"], conf_df["parameter_B"])}
    for fold_i, (tr, va) in enumerate(folds):
        m = make_lightgbm(random_state=RANDOM_STATE)
        m.fit(X_train_u.iloc[tr], split.y_train.iloc[tr])
        X_va = X_train_u.iloc[va].reset_index(drop=True)
        X_va_sample = X_va.sample(min(200, len(X_va)), random_state=RANDOM_STATE) if len(X_va) > 200 else X_va
        expl_f = shap.TreeExplainer(m)
        inter_f = expl_f.shap_interaction_values(X_va_sample)
        mean_abs_f = np.abs(inter_f).mean(axis=0)
        iu = np.triu_indices(mean_abs_f.shape[0], k=1)
        median_val = np.median(mean_abs_f[iu])
        for a, b in zip(conf_df["parameter_A"], conf_df["parameter_B"]):
            ma, mb = ds.name_map[a], ds.name_map[b]
            val = mean_abs_f[col_index[ma], col_index[mb]]
            if val > median_val:
                fold_above_median[(a, b)] += 1
        print(f"  fold {fold_i} done", flush=True)

    conf_df["fold_reproducibility"] = [f"{fold_above_median[(a,b)]}/5" for a, b in zip(conf_df["parameter_A"], conf_df["parameter_B"])]
    conf_df["fold_count"] = [fold_above_median[(a, b)] for a, b in zip(conf_df["parameter_A"], conf_df["parameter_B"])]

    def label_row(r):
        n_methods = int(pd.notna(r["shap_interaction_strength"]) and r["shap_interaction_strength"] > 0) + int(pd.notna(r["h_statistic_strength"]))
        fc = r["fold_count"]
        if fc >= 4 and n_methods >= 2:
            return "STRONG"
        if fc >= 3 or (fc >= 2 and n_methods >= 2):
            return "MODERATE"
        if fc >= 1:
            return "WEAK"
        return "INCONCLUSIVE"

    conf_df["evidence_label"] = conf_df.apply(label_row, axis=1)
    conf_df = conf_df.sort_values(
        by=["evidence_label", "shap_interaction_strength"], ascending=[True, False],
        key=lambda s: s.map({"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}) if s.name == "evidence_label" else s,
    )
    conf_df.to_csv(f"{R}/stage2_runB_v2_confirmed_interactions.csv", index=False)
    print(f"\nSaved {R}/stage2_runB_v2_confirmed_interactions.csv ({len(conf_df)} confirmed pairs)")
    print(conf_df["evidence_label"].value_counts().to_string())
    print("\nTop 15 STRONG confirmed interactions:")
    print(conf_df[conf_df["evidence_label"] == "STRONG"].head(15)[
        ["parameter_A", "parameter_B", "shap_interaction_strength", "h_statistic_strength", "fold_reproducibility"]
    ].to_string(index=False))

    with open(f"{R}/_stage2_v2_pair_screen_coverage.txt", "a") as f:
        f.write(f"union_features_in_confirmation={len(union_features)}\n")
        f.write(f"pairs_confirmed={len(conf_df)}\n")


if __name__ == "__main__":
    main()
