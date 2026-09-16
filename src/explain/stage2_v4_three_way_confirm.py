"""Stage 2 Run B v4 -- confirmation of interaction-graph-driven 3-way
candidates, using the EXISTING v3 lower-order-vs-full-3-way grouped-CV
logic UNCHANGED (src/explain/stage2_v3_common.py::lower_vs_full_cv and
src/explain/stage2_v3_three_way.py::label_row, imported not
reimplemented).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import lower_vs_full_cv
from src.explain.stage2_v3_three_way import label_row


def main():
    split = load_locked_split()
    ds = split.ds
    cand = pd.read_csv(f"{R}/stage2_runB_v4_three_way_candidates.csv")
    triples = list(zip(cand["parameter_A"], cand["parameter_B"], cand["parameter_C"]))
    channels_lookup = dict(zip(zip(cand["parameter_A"], cand["parameter_B"], cand["parameter_C"]), cand["channels"]))
    print(f"Confirming {len(triples)} v4 3-way candidates using the unchanged v3 lower-order-vs-full test")

    y_train_arr = split.y_train.to_numpy()
    _, folds = make_group_kfold(split.groups_train, n_splits=5)

    x_cache = {}
    def get_x(f):
        if f not in x_cache:
            x_cache[f] = split.X_train[ds.name_map[f]].to_numpy(dtype=np.float64)
        return x_cache[f]

    rows = []
    t0 = time.time()
    for k, (a, b, c) in enumerate(triples):
        if a not in ds.name_map or b not in ds.name_map or c not in ds.name_map:
            continue
        xa, xb, xc = get_x(a), get_x(b), get_x(c)
        cv = lower_vs_full_cv(xa, xb, xc, y_train_arr, folds, n_bins=2)
        if cv is None:
            continue
        label, reason = label_row(cv)
        rows.append({
            "parameter_A": a, "parameter_B": b, "parameter_C": c,
            "channels": channels_lookup.get((a, b, c), ""),
            "lower_order_cv_rmse": round(cv["lower_cv_rmse"], 4),
            "full_3way_cv_rmse": round(cv["full_cv_rmse"], 4),
            "incremental_rmse_improvement_pct": cv["rmse_improvement_pct"],
            "lower_order_cv_r2": round(cv["lower_cv_r2"], 4),
            "full_3way_cv_r2": round(cv["full_cv_r2"], 4),
            "incremental_r2": cv["incremental_r2"],
            "fold_reproducibility": f"{cv['folds_full_better']}/{cv['n_folds_used']}",
            "evidence_label": label,
            "reason": reason,
        })
        if (k + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {k+1}/{len(triples)} confirmed ({elapsed:.1f}s elapsed, ETA "
                  f"{elapsed/(k+1)*(len(triples)-k-1):.0f}s)...", flush=True)

    df = pd.DataFrame(rows)
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3}
    df = df.sort_values(["evidence_label", "incremental_r2"],
                         key=lambda s: s.map(order) if s.name == "evidence_label" else s,
                         ascending=[True, False])
    out_path = f"{R}/stage2_runB_v4_three_way_interactions.csv"
    df.to_csv(out_path, index=False)
    total_time = time.time() - t0
    print(f"\nSaved {out_path} ({len(df)} confirmed triples, total confirmation time {total_time:.1f}s)")
    print(df["evidence_label"].value_counts().to_string())

    print("\nTop 10 STRONG by incremental R^2:")
    print(df[df["evidence_label"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "parameter_C", "channels", "incremental_r2", "fold_reproducibility"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
