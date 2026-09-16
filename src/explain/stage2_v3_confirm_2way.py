"""Stage 2 Run B v3 -- expensive 2-way interaction confirmation.

Core, mandatory test (per project requirement): does the JOINT response
f(A,B) explain Yield beyond the ADDITIVE sum f(A) + f(B)? A pair is never
labeled a strong interaction from importance alone -- the confidence label
is gated on grouped-fold-reproducible incremental predictive improvement of
the joint 2D-binned model over the additive model (src/explain/
stage2_v3_common.py::additive_vs_joint_cv). SHAP-interaction and
H-statistic are also computed and reported (reusing the union-features
compact-model pattern established in v2, which avoids the earlier SHAP
segfault) as corroborating, not gating, evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.data.splits import make_group_kfold
from src.explain.interactions import compute_h_statistic
from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import (
    additive_vs_joint_cv, conditional_effect_text, fit_additive_joint,
)
from src.models.trees import make_lightgbm

RANDOM_STATE = 42


def label_row(cv: dict) -> tuple[str, str]:
    fc, n = cv["folds_joint_better"], cv["n_folds_used"]
    inc_r2 = cv["incremental_r2"]
    if cv["joint_beats_additive"] and fc == n:
        return "STRONG", f"Joint model beat additive in {fc}/{n} folds (unanimous), incremental R^2={inc_r2:.4f}."
    if cv["joint_beats_additive"] and fc >= n - 1:
        return "MODERATE", f"Joint model beat additive in {fc}/{n} folds, incremental R^2={inc_r2:.4f}."
    if fc >= max(3, int(round(0.6 * n))) and inc_r2 > 0.01:
        return "MODERATE", f"Joint model beat additive in {fc}/{n} folds with incremental R^2={inc_r2:.4f} -- majority but not unanimous."
    if inc_r2 > 0 or fc >= max(2, int(round(0.4 * n))):
        return "WEAK", f"Joint model beat additive in only {fc}/{n} folds, incremental R^2={inc_r2:.4f} -- inconsistent."
    return "INCONCLUSIVE", f"Joint model did not reproducibly beat additive ({fc}/{n} folds, incremental R^2={inc_r2:.4f})."


def region_label(fit: dict, edges_key: str, marg_key: str) -> None:
    pass


def describe_region(fit: dict, param_a: str, param_b: str, cell_idx: tuple, kind: str) -> str:
    ia, ib = cell_idx
    edges_a, edges_b = fit["edges_a"], fit["edges_b"]
    def bin_range(edges, i):
        lo = edges[i - 1] if i > 0 else -np.inf
        hi = edges[i] if i < len(edges) else np.inf
        return lo, hi
    lo_a, hi_a = bin_range(edges_a, ia)
    lo_b, hi_b = bin_range(edges_b, ib)
    fmt = lambda v: "-inf" if v == -np.inf else ("+inf" if v == np.inf else f"{v:.4g}")
    return f"{param_a} in [{fmt(lo_a)}, {fmt(hi_a)}] AND {param_b} in [{fmt(lo_b)}, {fmt(hi_b)}]"


def main():
    split = load_locked_split()
    ds = split.ds
    pool = pd.read_csv(f"{R}/stage2_runB_v3_pair_candidates.csv")
    candidate_pairs = list(zip(pool["parameter_A"], pool["parameter_B"]))
    union_features = sorted(set(pool["parameter_A"]) | set(pool["parameter_B"]))
    print(f"Confirming {len(candidate_pairs)} v3 candidate pairs, involving {len(union_features)} unique features")

    model_cols = [ds.name_map[f] for f in union_features]
    X_train_u = split.X_train[model_cols]
    X_test_u = split.X_test[model_cols]
    y_train_arr = split.y_train.to_numpy()
    _, folds = make_group_kfold(split.groups_train, n_splits=5)

    print("Fitting compact confirmation model on union features (train portion) for SHAP/H-statistic...")
    model = make_lightgbm(random_state=RANDOM_STATE)
    model.fit(X_train_u, split.y_train)

    print("Computing full SHAP interaction matrix on held-out test set...")
    explainer = shap.TreeExplainer(model)
    X_sample = X_test_u.sample(min(300, len(X_test_u)), random_state=RANDOM_STATE)
    inter = explainer.shap_interaction_values(X_sample)
    mean_abs_inter = np.abs(inter).mean(axis=0)
    col_index = {c: i for i, c in enumerate(model_cols)}

    X_bg = X_test_u.sample(min(60, len(X_test_u)), random_state=RANDOM_STATE).astype(np.float64)

    rows = []
    for k, (a, b) in enumerate(candidate_pairs):
        xa = split.X_train[ds.name_map[a]].to_numpy(dtype=np.float64)
        xb = split.X_train[ds.name_map[b]].to_numpy(dtype=np.float64)

        cv = additive_vs_joint_cv(xa, xb, y_train_arr, folds, n_bins=5)
        if cv is None:
            continue
        label, reason = label_row(cv)

        ma, mb = ds.name_map[a], ds.name_map[b]
        shap_strength = float(mean_abs_inter[col_index[ma], col_index[mb]])
        try:
            h_strength = compute_h_statistic(model, X_bg, ma, mb, grid_resolution=6)
        except Exception:
            h_strength = np.nan

        full_fit5 = fit_additive_joint(xa, xb, y_train_arr, n_bins=5)
        jg = full_fit5["joint_grid"]
        best_cell = np.unravel_index(np.argmax(jg), jg.shape)
        worst_cell = np.unravel_index(np.argmin(jg), jg.shape)
        high_region = describe_region(full_fit5, a, b, best_cell, "high")
        low_region = describe_region(full_fit5, a, b, worst_cell, "low")

        cond_text = ""
        if label in ("STRONG", "MODERATE"):
            full_fit3 = fit_additive_joint(xa, xb, y_train_arr, n_bins=3)
            cond_text = conditional_effect_text(full_fit3, a, b, n_bins_text=3)

        rows.append({
            "parameter_A": a, "parameter_B": b,
            "channels": pool.loc[(pool["parameter_A"] == a) & (pool["parameter_B"] == b), "channels"].values[0]
                        if ((pool["parameter_A"] == a) & (pool["parameter_B"] == b)).any() else "",
            "additive_cv_rmse": round(cv["additive_cv_rmse"], 4),
            "joint_cv_rmse": round(cv["joint_cv_rmse"], 4),
            "interaction_rmse_improvement_pct": cv["rmse_improvement_pct"],
            "additive_cv_r2": round(cv["additive_cv_r2"], 4),
            "joint_cv_r2": round(cv["joint_cv_r2"], 4),
            "interaction_incremental_r2": cv["incremental_r2"],
            "shap_interaction_strength": round(shap_strength, 5),
            "h_statistic_strength": round(h_strength, 4) if not np.isnan(h_strength) else None,
            "fold_reproducibility": f"{cv['folds_joint_better']}/{cv['n_folds_used']}",
            "high_yield_region": high_region,
            "low_yield_region": low_region,
            "conditional_effect_description": cond_text,
            "evidence_label": label,
            "reason": reason,
        })
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(candidate_pairs)} pairs confirmed...", flush=True)

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label", "interaction_incremental_r2"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s,
                         ascending=[True, False])
    out_path = f"{R}/stage2_runB_v3_confirmed_2way_interactions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} confirmed pairs)")
    print(df["evidence_label"].value_counts().to_string())

    print("\nTop 10 STRONG confirmed interactions:")
    print(df[df["evidence_label"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "interaction_incremental_r2", "interaction_rmse_improvement_pct", "fold_reproducibility"]
    ].to_string(index=False))

    print("\nSample conditional-effect descriptions (STRONG):")
    for _, r in df[df["evidence_label"] == "STRONG"].head(5).iterrows():
        print(f"  {r['parameter_A']} x {r['parameter_B']}: {r['conditional_effect_description']}")


if __name__ == "__main__":
    main()
