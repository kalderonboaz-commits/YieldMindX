"""Stage 2 Run B MLP -- section 8: single-parameter response analysis.

Reuses the EXISTING, already-validated model-agnostic PDP/shape-
classification machinery from src/explain/interactions.py
(one_d_partial_dependence, classify_pdp_shape) -- these functions only
call sklearn.inspection.partial_dependence on whatever estimator they are
given, so they work identically for the cached MLP models wrapped as a
scaler+model Pipeline, with no modification.

Scope: the top 100 predictors by MLP permutation importance (a disclosed,
generic cap -- "important MLP-supported predictors" per instruction,
not all 1,425, to keep this reporting-focused and tractable). 5-fold
stability via the 5 cached fold models (own training portion each).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.explain.interactions import classify_pdp_shape, one_d_partial_dependence
from src.explain.stage2_mlp_feature_evidence import CACHE_PATH
from src.explain.stage2_v2_common import R, load_locked_split

TOP_N_FEATURES = 100


def main():
    split = load_locked_split()
    cache = joblib.load(CACHE_PATH)
    fold_models = cache["fold_models"]
    full_fit = cache["full_fit"]
    feature_cols = cache["feature_cols"]
    orig_names = cache["orig_names"]
    name_to_col = dict(zip(orig_names, feature_cols))

    fe = pd.read_csv(f"{R}/stage2_runB_mlp_feature_evidence.csv")
    top_features = fe.sort_values("permutation_importance", ascending=False).head(TOP_N_FEATURES)["raw_csv_parameter_name"].tolist()
    print(f"Single-parameter PDP analysis for top {TOP_N_FEATURES} MLP-important features")

    full_pipeline = Pipeline([("scaler", full_fit.scaler), ("mlp", full_fit.model)])
    fold_pipelines = [Pipeline([("scaler", fm["fit"].scaler), ("mlp", fm["fit"].model)]) for fm in fold_models]

    rows = []
    for i, feat in enumerate(top_features):
        col = name_to_col.get(feat)
        if col is None or col not in split.X_train.columns:
            continue
        main_shape, main_dir = classify_pdp_shape(full_pipeline, split.X_train, col, grid_resolution=15)

        per_fold_shapes, per_fold_dirs = [], []
        for fm, pipe in zip(fold_models, fold_pipelines):
            X_fold = split.X_train.iloc[fm["tr"]]
            try:
                shp, drc = classify_pdp_shape(pipe, X_fold, col, grid_resolution=12)
            except Exception:
                shp, drc = "insufficient_data", "n/a"
            per_fold_shapes.append(shp)
            per_fold_dirs.append(drc)

        shape_consistency = sum(1 for s in per_fold_shapes if s == main_shape)
        dir_consistency = sum(1 for d in per_fold_dirs if d == main_dir)

        if main_shape == "flat_no_clear_effect":
            label = "INCONCLUSIVE"
        elif shape_consistency >= 4:
            label = "STRONG"
        elif shape_consistency == 3:
            label = "MODERATE"
        elif shape_consistency <= 1:
            label = "INCONCLUSIVE"
        else:
            label = "WEAK"

        grid, values = one_d_partial_dependence(full_pipeline, split.X_train, col, grid_resolution=15)
        effect_range = float(np.max(values) - np.min(values))

        rows.append({
            "raw_csv_parameter_name": feat,
            "relationship_class": main_shape,
            "direction": main_dir,
            "effect_range_yield_pct": round(effect_range, 4),
            "shape_fold_consistency": f"{shape_consistency}/5",
            "direction_fold_consistency": f"{dir_consistency}/5",
            "evidence_label": label,
            "observed_x_min": round(float(grid.min()), 6),
            "observed_x_max": round(float(grid.max()), 6),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(top_features)} done...", flush=True)

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label", "effect_range_yield_pct"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s, ascending=[True, False])
    out_path = f"{R}/stage2_runB_mlp_single_parameter_relationships.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} rows -- top {TOP_N_FEATURES} MLP-important features, not full-universe coverage)")
    print(df["evidence_label"].value_counts().to_string())
    print("\nRelationship class counts:")
    print(df["relationship_class"].value_counts().to_string())
    print("\nTop 10 STRONG by effect range:")
    print(df[df["evidence_label"] == "STRONG"].head(10)[
        ["raw_csv_parameter_name", "relationship_class", "direction", "effect_range_yield_pct"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
