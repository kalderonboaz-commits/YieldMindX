"""Stage 2 Run B v4 -- shuffle-based negative control for the expanded
(evidence-graph-driven) pair and 3-way candidate pools. ONE shuffled-target
permutation (same convention as v1/v2/spline/v3), reusing the EXACT same
v4 candidate pairs/triples and the EXACT same unchanged v3 confirmation
functions -- only the target values are permuted. Purpose: confirm that
expanding the candidate pool ~340x (2-way) / ~212x (3-way) over v3 did not
also expand the false-discovery rate.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import additive_vs_joint_cv, lower_vs_full_cv
from src.explain.stage2_v3_confirm_2way import label_row as label_2way
from src.explain.stage2_v3_three_way import label_row as label_3way

RANDOM_STATE = 42


def main():
    split = load_locked_split()
    ds = split.ds
    y_real = split.y_train.to_numpy()
    rng = np.random.RandomState(RANDOM_STATE)
    y_shuf = y_real.copy()
    rng.shuffle(y_shuf)

    _, folds = make_group_kfold(split.groups_train, n_splits=5)
    x_cache = {}
    def get_x(f):
        if f not in x_cache:
            x_cache[f] = split.X_train[ds.name_map[f]].to_numpy(dtype=np.float64)
        return x_cache[f]

    lines = ["STAGE 2 RUN B v4 -- NEGATIVE CONTROL (shuffled-target, expanded candidate pools)",
             "=" * 78, ""]

    # ---- 1. V4 pair candidates ----
    print("1/2: V4 pair candidates, shuffled target...")
    pool = pd.read_csv(f"{R}/stage2_runB_v4_pair_candidates.csv")
    pairs = list(zip(pool["parameter_A"], pool["parameter_B"]))
    t0 = time.time()
    labels2 = []
    for k, (a, b) in enumerate(pairs):
        cv = additive_vs_joint_cv(get_x(a), get_x(b), y_shuf, folds, n_bins=5)
        if cv is None:
            continue
        label, _ = label_2way(cv)
        labels2.append(label)
        if (k + 1) % 20000 == 0:
            print(f"  {k+1}/{len(pairs)} ({time.time()-t0:.0f}s elapsed)...", flush=True)
    s2 = pd.Series(labels2)
    c2 = s2.value_counts()
    t2 = len(s2)
    real_conf = pd.read_csv(f"{R}/stage2_runB_v4_confirmed_2way_interactions.csv")
    real_strong2 = int((real_conf["evidence_label"] == "STRONG").sum())
    real_mod2 = int((real_conf["evidence_label"] == "MODERATE").sum())
    real_total2 = len(real_conf)
    real_rate2 = 100 * (real_strong2 + real_mod2) / real_total2
    shuf_rate2 = 100 * (c2.get("STRONG", 0) + c2.get("MODERATE", 0)) / t2

    lines += [
        "1. V4 PAIR CANDIDATES (evidence-graph-driven, 74,365 pairs)",
        "-" * 78,
        f"Pairs re-tested under shuffled target: {t2} / {len(pairs)}",
        f"  STRONG:       {int(c2.get('STRONG', 0))} ({100*c2.get('STRONG', 0)/t2:.2f}%)",
        f"  MODERATE:     {int(c2.get('MODERATE', 0))} ({100*c2.get('MODERATE', 0)/t2:.2f}%)",
        f"  WEAK:         {int(c2.get('WEAK', 0))} ({100*c2.get('WEAK', 0)/t2:.2f}%)",
        f"  INCONCLUSIVE: {int(c2.get('INCONCLUSIVE', 0))} ({100*c2.get('INCONCLUSIVE', 0)/t2:.2f}%)",
        f"Real-target STRONG+MODERATE rate: {real_rate2:.1f}% ({real_strong2+real_mod2}/{real_total2})",
        f"Shuffled-target STRONG+MODERATE rate: {shuf_rate2:.2f}%",
    ]
    clean2 = shuf_rate2 <= 5.0
    lines.append(
        f"-> {'CLEAN SEPARATION' if clean2 else 'WARNING'}: shuffled rate is "
        f"{'a small fraction of' if clean2 else 'NOT clearly separated from'} the real rate "
        f"({shuf_rate2:.2f}% vs {real_rate2:.1f}%). "
        + ("Candidate-pool expansion did not increase the false-discovery floor." if clean2 else
           "Candidate-pool expansion may have increased false discoveries -- reported as-is, not recalibrated against ground truth.")
    )
    lines.append("")
    print(lines[-2])

    # ---- 2. V4 3-way candidates ----
    print("2/2: V4 3-way candidates, shuffled target...")
    cand3 = pd.read_csv(f"{R}/stage2_runB_v4_three_way_candidates.csv")
    triples = list(zip(cand3["parameter_A"], cand3["parameter_B"], cand3["parameter_C"]))
    t0 = time.time()
    labels3 = []
    for k, (a, b, c) in enumerate(triples):
        cv = lower_vs_full_cv(get_x(a), get_x(b), get_x(c), y_shuf, folds, n_bins=2)
        if cv is None:
            continue
        label, _ = label_3way(cv)
        labels3.append(label)
        if (k + 1) % 20000 == 0:
            print(f"  {k+1}/{len(triples)} ({time.time()-t0:.0f}s elapsed)...", flush=True)
    s3 = pd.Series(labels3)
    c3 = s3.value_counts()
    t3 = len(s3)
    real_three = pd.read_csv(f"{R}/stage2_runB_v4_three_way_interactions.csv")
    real_strong3 = int((real_three["evidence_label"] == "STRONG").sum())
    real_mod3 = int((real_three["evidence_label"] == "MODERATE").sum())
    real_total3 = len(real_three)
    real_rate3 = 100 * (real_strong3 + real_mod3) / real_total3
    shuf_rate3 = 100 * (c3.get("STRONG", 0) + c3.get("MODERATE", 0)) / t3

    lines += [
        "2. V4 3-WAY CANDIDATES (interaction-graph-driven, 69,606 triples)",
        "-" * 78,
        f"Triples re-tested under shuffled target: {t3} / {len(triples)}",
        f"  STRONG:       {int(c3.get('STRONG', 0))} ({100*c3.get('STRONG', 0)/t3:.2f}%)",
        f"  MODERATE:     {int(c3.get('MODERATE', 0))} ({100*c3.get('MODERATE', 0)/t3:.2f}%)",
        f"  WEAK:         {int(c3.get('WEAK', 0))} ({100*c3.get('WEAK', 0)/t3:.2f}%)",
        f"  INCONCLUSIVE: {int(c3.get('INCONCLUSIVE', 0))} ({100*c3.get('INCONCLUSIVE', 0)/t3:.2f}%)",
        f"Real-target STRONG+MODERATE rate: {real_rate3:.1f}% ({real_strong3+real_mod3}/{real_total3})",
        f"Shuffled-target STRONG+MODERATE rate: {shuf_rate3:.2f}%",
    ]
    clean3 = shuf_rate3 <= 5.0
    lines.append(
        f"-> {'CLEAN SEPARATION' if clean3 else 'WARNING'}: shuffled rate is "
        f"{'a small fraction of' if clean3 else 'NOT clearly separated from'} the real rate "
        f"({shuf_rate3:.2f}% vs {real_rate3:.1f}%). "
        + ("Candidate-pool expansion did not increase the false-discovery floor." if clean3 else
           "Candidate-pool expansion may have increased false discoveries -- reported as-is, not recalibrated against ground truth.")
    )
    lines.append("")
    print(lines[-2])

    out_path = f"{R}/stage2_runB_v4_negative_control.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
