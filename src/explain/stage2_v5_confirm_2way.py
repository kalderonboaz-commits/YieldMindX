"""Stage 2 Run B v5 -- confirmation of the extended pair pool. V4-confirmed
pairs are PRESERVED (reused directly, not recomputed) exactly as
instructed; only the newly-admitted MEDIUM x MEDIUM pairs are confirmed,
using the EXACT SAME unchanged v3/v4 additive-vs-joint grouped-CV logic
(literal function imports).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import additive_vs_joint_cv
from src.explain.stage2_v3_confirm_2way import label_row


def main():
    split = load_locked_split()
    ds = split.ds
    v5_pool = pd.read_csv(f"{R}/stage2_runB_v5_pair_candidates.csv")
    v4_conf = pd.read_csv(f"{R}/stage2_runB_v4_confirmed_2way_interactions.csv")
    v4_conf_lookup = {frozenset([a, b]): row for a, b, row in
                       zip(v4_conf["parameter_A"], v4_conf["parameter_B"], v4_conf.to_dict("records"))}

    y_train_arr = split.y_train.to_numpy()
    _, folds = make_group_kfold(split.groups_train, n_splits=5)
    x_cache = {}
    def get_x(f):
        if f not in x_cache:
            x_cache[f] = split.X_train[ds.name_map[f]].to_numpy(dtype=np.float64)
        return x_cache[f]

    preserved, newly_confirmed = 0, 0
    rows = []
    t0 = time.time()
    for k, r in v5_pool.iterrows():
        a, b = r["parameter_A"], r["parameter_B"]
        key = frozenset([a, b])
        if key in v4_conf_lookup:
            prev = v4_conf_lookup[key]
            rows.append({
                "parameter_A": a, "parameter_B": b, "channels": r["channels"],
                "evidence_tier_A": r["evidence_tier_A"], "evidence_tier_B": r["evidence_tier_B"],
                "source": "preserved_from_v4",
                "additive_cv_rmse": prev["additive_cv_rmse"], "joint_cv_rmse": prev["joint_cv_rmse"],
                "interaction_rmse_improvement_pct": prev["interaction_rmse_improvement_pct"],
                "additive_cv_r2": prev["additive_cv_r2"], "joint_cv_r2": prev["joint_cv_r2"],
                "interaction_incremental_r2": prev["interaction_incremental_r2"],
                "fold_reproducibility": prev["fold_reproducibility"],
                "evidence_label": prev["evidence_label"], "reason": prev["reason"],
            })
            preserved += 1
            continue
        cv = additive_vs_joint_cv(get_x(a), get_x(b), y_train_arr, folds, n_bins=5)
        if cv is None:
            continue
        label, reason = label_row(cv)
        rows.append({
            "parameter_A": a, "parameter_B": b, "channels": r["channels"],
            "evidence_tier_A": r["evidence_tier_A"], "evidence_tier_B": r["evidence_tier_B"],
            "source": "newly_confirmed_v5",
            "additive_cv_rmse": round(cv["additive_cv_rmse"], 4), "joint_cv_rmse": round(cv["joint_cv_rmse"], 4),
            "interaction_rmse_improvement_pct": cv["rmse_improvement_pct"],
            "additive_cv_r2": round(cv["additive_cv_r2"], 4), "joint_cv_r2": round(cv["joint_cv_r2"], 4),
            "interaction_incremental_r2": cv["incremental_r2"],
            "fold_reproducibility": f"{cv['folds_joint_better']}/{cv['n_folds_used']}",
            "evidence_label": label, "reason": reason,
        })
        newly_confirmed += 1
        if newly_confirmed % 5000 == 0:
            print(f"  {newly_confirmed} newly confirmed ({time.time()-t0:.0f}s elapsed)...", flush=True)

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label", "interaction_incremental_r2"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s,
                         ascending=[True, False])
    out_path = f"{R}/stage2_runB_v5_confirmed_2way_interactions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} rows: {preserved} preserved from v4, {newly_confirmed} newly confirmed)")
    print(df["evidence_label"].value_counts().to_string())
    print("\nNewly-confirmed-only breakdown:")
    print(df[df["source"] == "newly_confirmed_v5"]["evidence_label"].value_counts().to_string())

    print("\nTop 10 newly-confirmed STRONG (MEDIUM x MEDIUM) by incremental R^2:")
    newv5 = df[df["source"] == "newly_confirmed_v5"]
    print(newv5[newv5["evidence_label"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "channels", "interaction_incremental_r2", "fold_reproducibility"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
