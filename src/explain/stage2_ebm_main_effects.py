"""Stage 2 Run B EBM -- section 8: single-feature relationship extraction.

EBM directly stores a fitted shape function (bin edges + scores +
bagging-based standard deviations) for every one of the 1,425 main-effect
terms -- no expensive PDP recomputation needed, unlike the MLP checkpoint.
100% coverage is therefore cheap.

Shape classification REUSES the existing, already-validated
classify_curve_shape from src/explain/stage2_spline_common.py unmodified
-- it is fully generic (operates on any grid_x/curve_y/se_curve arrays),
so it works identically here with EBM's own bin centers/scores/standard
deviations in place of a spline-fitted curve.

Fold stability: the SAME shape classification is repeated on each of the
5 cached grouped-CV fold models' own main-effect term for that feature,
and compared to the final (full-training-set) model's shape.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.explain.stage2_ebm_common import classify_ebm_main_shape, real_bin_layout
from src.explain.stage2_ebm_train_eval import CACHE_PATH
from src.explain.stage2_v2_common import R, load_locked_split


def main():
    split = load_locked_split()
    cache = joblib.load(CACHE_PATH)
    final_ebm = cache["final_ebm"]
    fold_models = cache["fold_models"]

    feature_names = list(final_ebm.feature_names_in_)
    main_term_idx = {tuple(t): i for i, t in enumerate(final_ebm.term_features_) if len(t) == 1}
    print(f"Extracting {len(feature_names)} EBM main-effect terms (100% coverage)...")

    # EBM-appropriate flat-reference scale: the MEDIAN range across all
    # 1,425 main-effect terms (see classify_ebm_main_shape docstring for
    # why the target's own y_std is the wrong reference here).
    all_ranges = []
    for t_idx, t in enumerate(final_ebm.term_features_):
        if len(t) != 1:
            continue
        _, s, _ = real_bin_layout(final_ebm, t_idx)
        all_ranges.append(float(s.max() - s.min()))
    flat_ref_scale = float(np.median(all_ranges))
    print(f"Flat-reference scale (median main-effect term range across all {len(all_ranges)} terms): {flat_ref_scale:.6f}")

    # fold models' own main-term lookup by feature name (their own feature_names_in_
    # order can differ trivially in edge cases, so build per-model)
    fold_main_idx = []
    for fm in fold_models:
        ebm_f = fm["ebm"]
        idx_map = {tuple(t): i for i, t in enumerate(ebm_f.term_features_) if len(t) == 1}
        name_to_feat_idx = {n: i for i, n in enumerate(ebm_f.feature_names_in_)}
        fold_main_idx.append((ebm_f, idx_map, name_to_feat_idx))

    final_name_to_feat_idx = {n: i for i, n in enumerate(feature_names)}

    rows = []
    for k, fname in enumerate(feature_names):
        feat_idx = final_name_to_feat_idx[fname]
        term_idx = main_term_idx.get((feat_idx,))
        if term_idx is None:
            continue
        centers, scores, stds = real_bin_layout(final_ebm, term_idx)
        result = classify_ebm_main_shape(centers, scores, stds, flat_ref_scale)

        importance = float(final_ebm.term_importances()[term_idx])

        fold_shapes = []
        for ebm_f, idx_map, name_to_feat_idx in fold_main_idx:
            fi = name_to_feat_idx.get(fname)
            ti = idx_map.get((fi,)) if fi is not None else None
            if ti is None:
                fold_shapes.append(None)
                continue
            c_f, s_f, sd_f = real_bin_layout(ebm_f, ti)
            r_f = classify_ebm_main_shape(c_f, s_f, sd_f, flat_ref_scale)
            fold_shapes.append(r_f["shape"])

        n_valid = sum(1 for s in fold_shapes if s is not None)
        shape_consistency = sum(1 for s in fold_shapes if s == result["shape"])
        if result["shape"] in ("FLAT", "INCONCLUSIVE") or n_valid == 0:
            label = "INCONCLUSIVE"
        elif shape_consistency >= 4:
            label = "STRONG"
        elif shape_consistency == 3:
            label = "MODERATE"
        elif shape_consistency <= 1:
            label = "INCONCLUSIVE"
        else:
            label = "WEAK"

        tps = result["turning_points"]
        rows.append({
            "raw_csv_parameter_name": fname,
            "ebm_importance": round(importance, 6),
            "relationship_class": result["shape"], "direction": result["direction"],
            "turning_point_1": tps[0] if len(tps) > 0 else None,
            "turning_point_2": tps[1] if len(tps) > 1 else None,
            "beneficial_range": result["beneficial_range"], "harmful_range": result["harmful_range"],
            "effect_size_yield_pct": result["effect_range"],
            "shape_fold_consistency": f"{shape_consistency}/{n_valid}" if n_valid else "0/0",
            "evidence_label": label,
        })
        if (k + 1) % 300 == 0:
            print(f"  {k+1}/{len(feature_names)} done...", flush=True)

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label", "ebm_importance"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s, ascending=[True, False])
    out_path = f"{R}/stage2_runB_ebm_main_effects.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)}/{len(feature_names)} predictors, 100% coverage)")
    print(df["evidence_label"].value_counts().to_string())
    print("\nRelationship class counts:")
    print(df["relationship_class"].value_counts().to_string())
    print("\nTop 15 STRONG by EBM importance:")
    print(df[df["evidence_label"] == "STRONG"].sort_values("ebm_importance", ascending=False).head(15)[
        ["raw_csv_parameter_name", "relationship_class", "direction", "ebm_importance", "effect_size_yield_pct"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
