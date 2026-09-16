"""Stage 2 Run B MLP -- section 9: 2-way interaction evidence.

Candidate pool (generic, no ground-truth names): union of the top 35
features by MLP permutation importance (section 7) and the top 15
HIGH_EVIDENCE features from the existing frozen v4 feature-evidence graph
(cross-method corroboration channel, explicitly requested by the
instruction -- "existing frozen generic Stage 2 evidence graph").

Interaction test: reuses src/explain/interactions.py's two_d_partial_
dependence / one_d_partial_dependence machinery (already model-agnostic
via sklearn.inspection.partial_dependence) on the MLP wrapped as a
scaler+model Pipeline. For each pair: PD_ab(a,b) vs. PD_a(a)+PD_b(b)-ref
(the additive prediction) -- the SAME Tukey non-additivity decomposition
used by v3's H-statistic and by the generic binned broad-screen elsewhere
in this project, just evaluated through the neural network's own
predictions instead of LightGBM's.

Two-stage: Stage A computes the interaction score (H-statistic AND a raw
RMS non-additivity residual, in yield-point units) for ALL candidate pairs
using the FULL model. Stage B recomputes the score on each of the 5 cached
fold models (fold's own training portion) for ONLY the top 100 Stage-A
pairs, for fold-stability confidence labeling.
"""
from __future__ import annotations

import time
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.explain.interactions import one_d_partial_dependence, two_d_partial_dependence
from src.explain.stage2_mlp_feature_evidence import CACHE_PATH
from src.explain.stage2_v2_common import R, load_locked_split

TOP_MLP_FEATURES = 35
TOP_V4_HIGH_FEATURES = 15
STAGE_B_TOP_K = 100
BACKGROUND_SIZE_A = 60
BACKGROUND_SIZE_B = 40
GRID_RES_A = 6
GRID_RES_B = 5


def interaction_scores(pipe, X_bg, col_a, col_b, grid_resolution):
    grid_2d, pd_ab = two_d_partial_dependence(pipe, X_bg, col_a, col_b, grid_resolution=grid_resolution)
    grid_a, pd_a = one_d_partial_dependence(pipe, X_bg, col_a, grid_resolution=grid_resolution)
    grid_b, pd_b = one_d_partial_dependence(pipe, X_bg, col_b, grid_resolution=grid_resolution)
    if len(pd_a) != pd_ab.shape[0] or len(pd_b) != pd_ab.shape[1]:
        return None
    pd_ab_c = pd_ab - pd_ab.mean()
    pd_a_c = pd_a - pd_a.mean()
    pd_b_c = pd_b - pd_b.mean()
    residual = pd_ab_c - pd_a_c[:, None] - pd_b_c[None, :]
    h2 = np.sum(residual ** 2) / max(np.sum(pd_ab_c ** 2), 1e-12)
    h_stat = float(np.sqrt(max(h2, 0.0)))
    rms_residual = float(np.sqrt(np.mean(residual ** 2)))
    best_idx = np.unravel_index(np.argmax(pd_ab), pd_ab.shape)
    worst_idx = np.unravel_index(np.argmin(pd_ab), pd_ab.shape)
    high_region = f"{col_a}~{grid_2d[0][best_idx[0]]:.4g}, {col_b}~{grid_2d[1][best_idx[1]]:.4g}"
    low_region = f"{col_a}~{grid_2d[0][worst_idx[0]]:.4g}, {col_b}~{grid_2d[1][worst_idx[1]]:.4g}"
    return h_stat, rms_residual, high_region, low_region


def main():
    split = load_locked_split()
    cache = joblib.load(CACHE_PATH)
    fold_models = cache["fold_models"]
    full_fit = cache["full_fit"]
    name_to_col = dict(zip(cache["orig_names"], cache["feature_cols"]))
    full_pipe = Pipeline([("scaler", full_fit.scaler), ("mlp", full_fit.model)])
    fold_pipes = [Pipeline([("scaler", fm["fit"].scaler), ("mlp", fm["fit"].model)]) for fm in fold_models]

    fe = pd.read_csv(f"{R}/stage2_runB_mlp_feature_evidence.csv")
    top_mlp = fe.sort_values("permutation_importance", ascending=False).head(TOP_MLP_FEATURES)["raw_csv_parameter_name"].tolist()

    eg = pd.read_csv(f"{R}/stage2_runB_v4_feature_evidence_graph.csv")
    top_v4 = eg[eg["evidence_tier"] == "HIGH_EVIDENCE"].sort_values(
        "n_independent_methods_supporting", ascending=False).head(TOP_V4_HIGH_FEATURES)["raw_csv_parameter_name"].tolist()

    candidate_features = sorted(set(top_mlp) | set(top_v4))
    candidate_cols = [name_to_col[f] for f in candidate_features if f in name_to_col]
    print(f"Candidate features: {len(candidate_cols)} ({TOP_MLP_FEATURES} top-MLP-importance + {TOP_V4_HIGH_FEATURES} top-v4-HIGH_EVIDENCE, deduped)")
    pairs = list(combinations(candidate_cols, 2))
    print(f"Candidate pairs (Stage A, full model): {len(pairs)}")

    X_bg_a = split.X_train.sample(min(BACKGROUND_SIZE_A, len(split.X_train)), random_state=42).astype(np.float64)

    print("Stage A: computing interaction scores for all candidate pairs (full model)...")
    t0 = time.time()
    stage_a_rows = []
    for i, (ca, cb) in enumerate(pairs):
        res = interaction_scores(full_pipe, X_bg_a, ca, cb, GRID_RES_A)
        if res is None:
            continue
        h_stat, rms_resid, high_region, low_region = res
        stage_a_rows.append({
            "parameter_A": cache["orig_names"][cache["feature_cols"].index(ca)],
            "parameter_B": cache["orig_names"][cache["feature_cols"].index(cb)],
            "col_a": ca, "col_b": cb,
            "mlp_h_statistic": round(h_stat, 4), "mlp_rms_nonadditivity": round(rms_resid, 5),
            "high_yield_region": high_region, "low_yield_region": low_region,
        })
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(pairs)} ({time.time()-t0:.0f}s elapsed)...", flush=True)
    stage_a_df = pd.DataFrame(stage_a_rows).sort_values("mlp_h_statistic", ascending=False)
    print(f"Stage A done in {time.time()-t0:.1f}s")

    stage_b_pool = stage_a_df.head(STAGE_B_TOP_K)
    print(f"\nStage B: fold-stability check on top {len(stage_b_pool)} pairs (5 cached fold models)...")
    t0 = time.time()
    fold_h = {i: [] for i in stage_b_pool.index}
    for fm, pipe in zip(fold_models, fold_pipes):
        X_fold = split.X_train.iloc[fm["tr"]]
        X_bg_b = X_fold.sample(min(BACKGROUND_SIZE_B, len(X_fold)), random_state=42).astype(np.float64)
        for idx, row in stage_b_pool.iterrows():
            res = interaction_scores(pipe, X_bg_b, row["col_a"], row["col_b"], GRID_RES_B)
            fold_h[idx].append(res[0] if res is not None else np.nan)
    print(f"Stage B done in {time.time()-t0:.1f}s")

    median_h_all_folds = np.nanmedian([v for vals in fold_h.values() for v in vals])
    final_rows = []
    for idx, row in stage_b_pool.iterrows():
        h_vals = np.array(fold_h[idx])
        above_median = int(np.nansum(h_vals > median_h_all_folds))
        n_valid = int(np.sum(~np.isnan(h_vals)))
        fold_repro = f"{above_median}/{n_valid}" if n_valid else "0/0"
        if above_median >= 4:
            label = "STRONG"
        elif above_median == 3:
            label = "MODERATE"
        elif above_median <= 1:
            label = "INCONCLUSIVE"
        else:
            label = "WEAK"
        final_rows.append({
            "parameter_A": row["parameter_A"], "parameter_B": row["parameter_B"],
            "mlp_h_statistic": row["mlp_h_statistic"], "mlp_rms_nonadditivity": row["mlp_rms_nonadditivity"],
            "fold_reproducibility": fold_repro,
            "high_yield_region": row["high_yield_region"], "low_yield_region": row["low_yield_region"],
            "confidence": label,
        })
    final_df = pd.DataFrame(final_rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    final_df = final_df.sort_values(["confidence", "mlp_h_statistic"],
                                     key=lambda s: s.map(order) if s.name == "confidence" else s, ascending=[True, False])
    out_path = f"{R}/stage2_runB_mlp_2way_interactions.csv"
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(final_df)} confirmed pairs out of {len(pairs)} Stage-A candidates)")
    print(final_df["confidence"].value_counts().to_string())
    print("\nTop 10 STRONG:")
    print(final_df[final_df["confidence"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "mlp_h_statistic", "fold_reproducibility"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
