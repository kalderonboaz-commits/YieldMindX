"""Stage 2 Run B v3 -- shuffle-based negative control for all three new
interaction-discovery stages: the full-coverage broad non-additivity
screen, the additive-vs-joint 2-way confirmation, and the lower-order-vs-
full 3-way confirmation. ONE shuffled-target permutation (same convention
as v1/v2/spline negative controls in this project), reusing the EXACT same
candidate pools/pairs/triples already generated on the real target -- only
the target values are permuted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import make_group_kfold
from src.explain.stage2_v2_common import R, load_locked_split
from src.explain.stage2_v3_common import additive_vs_joint_cv, compute_nonadditivity_screen, lower_vs_full_cv
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

    lines = ["STAGE 2 RUN B v3 -- INTERACTION-DISCOVERY NEGATIVE CONTROL (shuffled-target)",
             "=" * 74, ""]

    # ---- 1. Broad non-additivity screen (score_2), full coverage ----
    print("1/3: Broad non-additivity screen (score_2), full coverage, shuffled target...")
    numeric_features = ds.numeric_cols_original
    model_cols = [ds.name_map[f] for f in numeric_features]
    X_arr = split.X_train[model_cols].to_numpy(dtype=np.float64)
    score2_real = compute_nonadditivity_screen(X_arr, y_real, n_bins=4)
    score2_shuf = compute_nonadditivity_screen(X_arr, y_shuf, n_bins=4)
    n = len(numeric_features)
    iu = np.triu_indices(n, k=1)
    real_flat, shuf_flat = score2_real[iu], score2_shuf[iu]
    real_top300_cutoff = np.sort(real_flat)[-300]
    shuf_max = shuf_flat.max()
    n_shuf_exceeding_real_cutoff = int((shuf_flat >= real_top300_cutoff).sum())

    lines += [
        "1. BROAD 2-WAY NON-ADDITIVITY SCREEN (score_2, full 986,310-pair coverage)",
        "-" * 74,
        f"Real-target score needed to enter the top-300 shortlist: {real_top300_cutoff:.4f}",
        f"Shuffled-target maximum score (across all 986,310 pairs): {shuf_max:.4f}",
        f"Shuffled-target pairs exceeding the real-target top-300 cutoff: {n_shuf_exceeding_real_cutoff}",
        ("-> CLEAN SEPARATION: the real top-300 shortlist reflects real structure, not screen noise."
         if n_shuf_exceeding_real_cutoff == 0 else
         f"-> WARNING: {n_shuf_exceeding_real_cutoff} shuffled pairs would have entered the real top-300 shortlist -- recalibration needed."),
        "",
    ]
    print(lines[-2])

    # ---- 2. Confirmed 2-way (same candidate pool, shuffled target) ----
    print("2/3: Confirmed 2-way interaction test, same 219-pair candidate pool, shuffled target...")
    pool = pd.read_csv(f"{R}/stage2_runB_v3_pair_candidates.csv")
    candidate_pairs = list(zip(pool["parameter_A"], pool["parameter_B"]))
    _, folds = make_group_kfold(split.groups_train, n_splits=5)
    labels_2way = []
    for a, b in candidate_pairs:
        xa = split.X_train[ds.name_map[a]].to_numpy(dtype=np.float64)
        xb = split.X_train[ds.name_map[b]].to_numpy(dtype=np.float64)
        cv = additive_vs_joint_cv(xa, xb, y_shuf, folds, n_bins=5)
        if cv is None:
            continue
        label, _ = label_2way(cv)
        labels_2way.append(label)
    s2 = pd.Series(labels_2way)
    counts2 = s2.value_counts()
    total2 = len(s2)

    lines += [
        "2. CONFIRMED 2-WAY INTERACTIONS (same 219-pair candidate pool as the real run)",
        "-" * 74,
        f"Pairs re-tested under shuffled target: {total2} / 219",
        f"  STRONG:       {int(counts2.get('STRONG', 0))} ({100*counts2.get('STRONG', 0)/total2:.1f}%)",
        f"  MODERATE:     {int(counts2.get('MODERATE', 0))} ({100*counts2.get('MODERATE', 0)/total2:.1f}%)",
        f"  WEAK:         {int(counts2.get('WEAK', 0))} ({100*counts2.get('WEAK', 0)/total2:.1f}%)",
        f"  INCONCLUSIVE: {int(counts2.get('INCONCLUSIVE', 0))} ({100*counts2.get('INCONCLUSIVE', 0)/total2:.1f}%)",
        "Compare to the real-target result: 176/219 STRONG (80.4%), 30/219 MODERATE (13.7%).",
        ("-> CLEAN SEPARATION: 0 STRONG/MODERATE survive under shuffle -- the confirmation "
         "gate is not noise-driven; the high real-target hit rate reflects the candidate pool "
         "already being pre-filtered toward real non-additive structure by the broad screen, "
         "not a miscalibrated threshold."
         if counts2.get('STRONG', 0) == 0 and counts2.get('MODERATE', 0) == 0 else
         "-> WARNING: STRONG/MODERATE findings survive under shuffle -- recalibration needed."),
        "",
    ]
    print(lines[-2])

    # ---- 3. 3-way (same candidate triples, shuffled target) ----
    print("3/3: 3-way lower-order-vs-full test, same candidate triples, shuffled target...")
    three = pd.read_csv(f"{R}/stage2_runB_v3_three_way_interactions.csv")
    triples = list(zip(three["parameter_A"], three["parameter_B"], three["parameter_C"]))
    labels_3way = []
    for a, b, c in triples:
        if a not in ds.name_map or b not in ds.name_map or c not in ds.name_map:
            continue
        xa = split.X_train[ds.name_map[a]].to_numpy(dtype=np.float64)
        xb = split.X_train[ds.name_map[b]].to_numpy(dtype=np.float64)
        xc = split.X_train[ds.name_map[c]].to_numpy(dtype=np.float64)
        cv = lower_vs_full_cv(xa, xb, xc, y_shuf, folds, n_bins=2)
        if cv is None:
            continue
        label, _ = label_3way(cv)
        labels_3way.append(label)
    s3 = pd.Series(labels_3way)
    counts3 = s3.value_counts()
    total3 = len(s3)

    lines += [
        "3. 3-WAY INTERACTIONS (same candidate triples as the real run)",
        "-" * 74,
        f"Triples re-tested under shuffled target: {total3} / {len(triples)}",
        f"  STRONG:       {int(counts3.get('STRONG', 0))} ({100*counts3.get('STRONG', 0)/total3:.1f}%)",
        f"  MODERATE:     {int(counts3.get('MODERATE', 0))} ({100*counts3.get('MODERATE', 0)/total3:.1f}%)",
        f"  WEAK:         {int(counts3.get('WEAK', 0))} ({100*counts3.get('WEAK', 0)/total3:.1f}%)",
        f"  INCONCLUSIVE: {int(counts3.get('INCONCLUSIVE', 0))} ({100*counts3.get('INCONCLUSIVE', 0)/total3:.1f}%)",
        "Compare to the real-target result: 105/328 STRONG (32.0%), 88/328 MODERATE (26.8%).",
    ]
    shuf_rate3 = 100 * (counts3.get('STRONG', 0) + counts3.get('MODERATE', 0)) / total3
    real_rate3 = 100 * (105 + 88) / 328
    if shuf_rate3 <= 5.0 and real_rate3 >= 5 * max(shuf_rate3, 0.1):
        lines.append(f"-> CLEAN SEPARATION: shuffled STRONG+MODERATE rate ({shuf_rate3:.1f}%) is a small fraction "
                      f"of the real rate ({real_rate3:.1f}%), consistent with an expected, non-alarming false-"
                      f"discovery floor (not 'many' survivors) -- thresholds are not noise-driven, no recalibration performed.")
    else:
        lines.append("-> WARNING: shuffled STRONG/MODERATE rate is not a small fraction of the real rate -- recalibration needed.")
    lines.append("")
    print(lines[-2])

    out_path = f"{R}/stage2_runB_v3_interaction_negative_control.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
