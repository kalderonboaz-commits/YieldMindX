"""Stage 2 Run B v4 -- confirmation of evidence-graph-driven pair
candidates, using the EXISTING v3 additive-vs-joint grouped-CV confirmation
logic UNCHANGED (src/explain/stage2_v3_common.py::additive_vs_joint_cv and
src/explain/stage2_v3_confirm_2way.py::label_row, imported not
reimplemented). This checkpoint tests candidate-generation improvement,
not confirmation redesign -- SHAP-interaction/H-statistic corroboration
(v3's extra context columns) are intentionally omitted here given the much
larger candidate pool; the core additive-vs-joint test and its confidence
label are byte-for-byte the same function calls as v3.
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
    pool = pd.read_csv(f"{R}/stage2_runB_v4_pair_candidates.csv")
    candidate_pairs = list(zip(pool["parameter_A"], pool["parameter_B"]))
    channels_lookup = dict(zip(zip(pool["parameter_A"], pool["parameter_B"]), pool["channels"]))
    print(f"Confirming {len(candidate_pairs)} v4 candidate pairs using the unchanged v3 additive-vs-joint test")

    y_train_arr = split.y_train.to_numpy()
    _, folds = make_group_kfold(split.groups_train, n_splits=5)

    x_cache = {}
    def get_x(f):
        if f not in x_cache:
            x_cache[f] = split.X_train[ds.name_map[f]].to_numpy(dtype=np.float64)
        return x_cache[f]

    rows = []
    t0 = time.time()
    for k, (a, b) in enumerate(candidate_pairs):
        xa, xb = get_x(a), get_x(b)
        cv = additive_vs_joint_cv(xa, xb, y_train_arr, folds, n_bins=5)
        if cv is None:
            continue
        label, reason = label_row(cv)
        rows.append({
            "parameter_A": a, "parameter_B": b,
            "channels": channels_lookup.get((a, b), ""),
            "additive_cv_rmse": round(cv["additive_cv_rmse"], 4),
            "joint_cv_rmse": round(cv["joint_cv_rmse"], 4),
            "interaction_rmse_improvement_pct": cv["rmse_improvement_pct"],
            "additive_cv_r2": round(cv["additive_cv_r2"], 4),
            "joint_cv_r2": round(cv["joint_cv_r2"], 4),
            "interaction_incremental_r2": cv["incremental_r2"],
            "fold_reproducibility": f"{cv['folds_joint_better']}/{cv['n_folds_used']}",
            "evidence_label": label,
            "reason": reason,
        })
        if (k + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {k+1}/{len(candidate_pairs)} confirmed ({elapsed:.1f}s elapsed, "
                  f"{elapsed/(k+1)*1000:.2f} ms/pair, ETA {elapsed/(k+1)*(len(candidate_pairs)-k-1):.0f}s)...", flush=True)

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label", "interaction_incremental_r2"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s,
                         ascending=[True, False])
    out_path = f"{R}/stage2_runB_v4_confirmed_2way_interactions.csv"
    df.to_csv(out_path, index=False)
    total_time = time.time() - t0
    print(f"\nSaved {out_path} ({len(df)} confirmed pairs, total confirmation time {total_time:.1f}s)")
    print(df["evidence_label"].value_counts().to_string())

    print("\nTop 10 STRONG by incremental R^2:")
    print(df[df["evidence_label"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "channels", "interaction_incremental_r2", "fold_reproducibility"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
