"""Relationship Validation Engine -- 2-way interaction validation.

For every pair PROMOTED by the broad screen (stage2_relval_pair_screen.py):
tests genuine non-additivity via the additive-vs-joint grouped-CV
comparison (pair_cv, extending v3_common's fit_additive_joint/
predict_additive/predict_joint machinery, reused unmodified, with MAE
added), with a permutation-based empirical p-value (pooled null from
B_PERM target shuffles across the SAME promoted candidate pool) and
Benjamini-Hochberg FDR as the primary acceptance gate, on top of grouped-
fold stability. Region/conditional-pattern description is extracted from
one full-training-set fit_additive_joint (reused unmodified), not
recomputed per CV fold.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.explain.stage2_relval_common import (
    CACHE_DIR, R, RANDOM_STATE, benjamini_hochberg, empirical_pvalue, group_block_shuffle, load_locked_split,
    pair_cv,
)
from src.explain.stage2_v3_common import fit_additive_joint

B_PERM = 10
ALPHA_FDR = 0.05
MIN_FOLD_FRACTION_STRONG = 1.0    # was: absolute count 5 (dormant bug -- see audit; fraction is correct for candidates with <5 valid folds)
MIN_FOLD_FRACTION_MODERATE = 0.8  # was: absolute count 4


def describe_region(fit: dict, param_a: str, param_b: str, cell_idx: tuple) -> str:
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
    y_train_arr = split.y_train.to_numpy()

    pool = pd.read_csv(f"{R}/stage2_runB_relationship_validation_pair_screen.csv")
    pool = pool[pool["promoted"]].reset_index(drop=True)
    candidate_pairs = list(zip(pool["parameter_A"], pool["parameter_B"]))
    channels_lookup = dict(zip(zip(pool["parameter_A"], pool["parameter_B"]), pool["promotion_channels"]))
    print(f"2-way validation: {len(candidate_pairs)} promoted candidate pairs")

    x_cache = {}
    def get_x(f):
        if f not in x_cache:
            x_cache[f] = split.X_train[ds.name_map[f]].to_numpy(dtype=np.float64)
        return x_cache[f]

    print("Real-target pass...")
    t0 = time.time()
    real_rows = []
    real_stat = {}
    for k, (a, b) in enumerate(candidate_pairs):
        xa, xb = get_x(a), get_x(b)
        cv = pair_cv(xa, xb, y_train_arr, split.folds, n_bins=5)
        if cv is None:
            continue
        full_fit = fit_additive_joint(xa, xb, y_train_arr, n_bins=5)
        jg = full_fit["joint_grid"]
        best_cell = np.unravel_index(np.argmax(jg), jg.shape)
        worst_cell = np.unravel_index(np.argmin(jg), jg.shape)
        high_region = describe_region(full_fit, a, b, best_cell)
        low_region = describe_region(full_fit, a, b, worst_cell)
        conditional_pattern = (
            f"Joint mean ranges {jg.min():.3f} to {jg.max():.3f} across the {jg.shape[0]}x{jg.shape[1]} bin grid "
            f"(vs. additive-only range {full_fit['additive_grid'].min():.3f} to {full_fit['additive_grid'].max():.3f})"
        )
        real_rows.append({
            "parameter_A": a, "parameter_B": b, "promotion_channels": channels_lookup.get((a, b), ""),
            "additive_cv_rmse": round(cv["additive_cv_rmse"], 4), "joint_cv_rmse": round(cv["joint_cv_rmse"], 4),
            "delta_rmse": round(cv["delta_rmse"], 4),
            "additive_cv_mae": round(cv["additive_cv_mae"], 4), "joint_cv_mae": round(cv["joint_cv_mae"], 4),
            "delta_mae": round(cv["delta_mae"], 4),
            "additive_cv_r2": round(cv["additive_cv_r2"], 4), "joint_cv_r2": round(cv["joint_cv_r2"], 4),
            "delta_r2": round(cv["delta_r2"], 6),
            "folds_improved": f"{cv['folds_improved']}/{cv['n_folds']}",
            "interaction_strength_delta_r2": round(cv["delta_r2"], 6),
            "high_yield_region": high_region, "low_yield_region": low_region,
            "conditional_pattern": conditional_pattern,
        })
        real_stat[(a, b)] = cv["delta_r2"]
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(candidate_pairs)} ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"Real-target pass done in {time.time()-t0:.1f}s ({len(real_rows)} pairs with valid CV)")

    print(f"\nRunning {B_PERM} shuffled-target permutations on the SAME {len(candidate_pairs)}-pair pool...")
    null_pool = []
    t0 = time.time()
    groups_train = split.groups_train.to_numpy()
    for perm_i in range(B_PERM):
        rng = np.random.RandomState(RANDOM_STATE + 2000 + perm_i)
        y_shuf = group_block_shuffle(y_train_arr, groups_train, rng)
        for a, b in candidate_pairs:
            cv = pair_cv(get_x(a), get_x(b), y_shuf, split.folds, n_bins=5)
            if cv is not None:
                null_pool.append(cv["delta_r2"])
        print(f"  permutation {perm_i+1}/{B_PERM} done ({time.time()-t0:.0f}s elapsed, pool size={len(null_pool)})", flush=True)
    null_pool = np.array(null_pool)
    print(f"Permutation pass done in {time.time()-t0:.1f}s -- pooled null size={len(null_pool)}")
    effect_size_95th = float(np.percentile(null_pool, 95))
    print(f"Empirical null 95th percentile of delta_r2: {effect_size_95th:.6f}")

    df = pd.DataFrame(real_rows)
    pvals = np.array([empirical_pvalue(real_stat[(r["parameter_A"], r["parameter_B"])], null_pool) for _, r in df.iterrows()])
    qvals, accept = benjamini_hochberg(pvals, alpha=ALPHA_FDR)
    df["empirical_p_value"] = np.round(pvals, 6)
    df["bh_adjusted_q_value"] = np.round(qvals, 6)
    df["survives_fdr"] = accept

    def label_row(r):
        fi, fn = (int(x) for x in r["folds_improved"].split("/"))
        meaningful = r["interaction_strength_delta_r2"] > effect_size_95th
        if not r["survives_fdr"]:
            return "INCONCLUSIVE", "fails FDR"
        if fn > 0 and (fi / fn) >= MIN_FOLD_FRACTION_STRONG and meaningful:
            return "STRONG", f"{fi}/{fn} folds improved, effect exceeds null 95th pctile, survives FDR"
        if fn > 0 and (fi / fn) >= MIN_FOLD_FRACTION_MODERATE and meaningful:
            return "MODERATE", f"{fi}/{fn} folds improved, effect exceeds null 95th pctile, survives FDR"
        if fi >= 2 or meaningful:
            return "WEAK", f"{fi}/{fn} folds improved, marginal effect"
        return "INCONCLUSIVE", "insufficient fold consistency or effect size"

    labels_reasons = df.apply(label_row, axis=1)
    df["confidence"] = [x[0] for x in labels_reasons]
    df["reason"] = [x[1] for x in labels_reasons]

    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["confidence", "interaction_strength_delta_r2"],
                         key=lambda s: s.map(order) if s.name == "confidence" else s, ascending=[True, False])
    out_path = f"{R}/stage2_runB_relationship_validation_2way.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} validated pairs)")
    print(df["confidence"].value_counts().to_string())
    print(f"Hypotheses tested: {len(df)} | Survive FDR: {int(df['survives_fdr'].sum())}")

    with open(f"{CACHE_DIR}/pair_null_summary.json", "w") as f:
        json.dump({
            "family": "2way", "n_hypotheses": int(len(df)), "b_perm": B_PERM,
            "null_pool_size": int(len(null_pool)),
            "null_delta_r2_mean": float(np.mean(null_pool)), "null_delta_r2_95th_pctile": effect_size_95th,
            "null_delta_r2_99th_pctile": float(np.percentile(null_pool, 99)),
            "real_strong_count": int((df["confidence"] == "STRONG").sum()),
            "real_moderate_count": int((df["confidence"] == "MODERATE").sum()),
            "real_survives_fdr_count": int(df["survives_fdr"].sum()),
        }, f, indent=2)
    np.save(f"{CACHE_DIR}/pair_null_pool.npy", null_pool)

    print("\nTop 10 STRONG by interaction strength:")
    print(df[df["confidence"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "interaction_strength_delta_r2", "folds_improved", "bh_adjusted_q_value"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
