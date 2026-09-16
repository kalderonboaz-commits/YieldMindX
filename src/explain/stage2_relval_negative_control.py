"""Relationship Validation Engine -- consolidated negative control +
calibration gate (section 7).

Reuses the pooled null distributions already computed (and saved) by each
family's own validation script (effect-size / delta_r2 distributions --
no recomputation) for the STRONG/MODERATE-count and effect-size-
distribution comparisons. Additionally runs ONE FRESH shuffled-target pass
per family (a fixed seed disjoint from every seed already used for the
per-family pooled nulls) through the SAME acceptance rule real candidates
were judged by (same fold-count thresholds, same effect-size-vs-null-95th-
percentile gate) to directly report shuffled-target STRONG/MODERATE counts
and the fold-stability (folds-improved) DISTRIBUTION under shuffle,
compared against the real-target distribution already in the frozen
per-family CSVs.

This is the explicit PASS/FAIL calibration gate: per instruction, if
shuffled-target discovery remains high or comparable to the real target,
this script reports FAIL and the engine must not be represented as
reliably calibrated (no further development occurs regardless, since
ground-truth validation is separately gated by the user's explicit
approval either way).
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.explain.stage2_relval_common import (
    CACHE_DIR, R, RANDOM_STATE, group_block_shuffle, load_locked_split, pair_cv, single_param_cv, triple_cv,
)

FRESH_SEED_OFFSET = 99000
MIN_FOLD_FRACTION_STRONG = 1.0
MIN_FOLD_FRACTION_MODERATE = 0.8
PASS_RATIO_THRESHOLD = 0.25  # shuffled STRONG+MODERATE rate must be well below (<25% of) the real rate to PASS


def fold_dist_from_column(series: pd.Series) -> dict:
    counts = Counter(int(s.split("/")[0]) for s in series)
    return {k: counts.get(k, 0) for k in range(6)}


def main():
    split = load_locked_split()
    ds = split.ds
    y_train_arr = split.y_train.to_numpy()
    groups_train = split.groups_train.to_numpy()

    lines = ["STAGE 2 RUN B -- RELATIONSHIP VALIDATION ENGINE: CONSOLIDATED NEGATIVE CONTROL",
             "=" * 78, ""]

    with open(f"{CACHE_DIR}/single_null_summary.json") as f:
        single_null = json.load(f)
    with open(f"{CACHE_DIR}/pair_null_summary.json") as f:
        pair_null = json.load(f)
    with open(f"{CACHE_DIR}/triple_null_summary.json") as f:
        triple_null = json.load(f)

    single_df = pd.read_csv(f"{R}/stage2_runB_relationship_validation_single.csv")
    pair_df = pd.read_csv(f"{R}/stage2_runB_relationship_validation_2way.csv")
    triple_df = pd.read_csv(f"{R}/stage2_runB_relationship_validation_3way.csv")

    overall_pass = True

    # ---- SINGLE-PARAMETER ----
    print("Fresh shuffled pass: single-parameter (1 permutation, same acceptance rule)...")
    t0 = time.time()
    rng = np.random.RandomState(RANDOM_STATE + FRESH_SEED_OFFSET)
    y_shuf = group_block_shuffle(y_train_arr, groups_train, rng)
    shuf_fi = []
    shuf_strong, shuf_moderate = 0, 0
    eff95 = single_null["null_delta_r2_95th_pctile"]
    for f in ds.numeric_cols_original:
        x = split.X_train[ds.name_map[f]].to_numpy(dtype=float)
        cv = single_param_cv(x, y_shuf, split.folds)
        if cv is None:
            continue
        shuf_fi.append(cv["folds_improved"])
        meaningful = cv["delta_r2"] > eff95
        if cv["n_folds"] > 0 and (cv["folds_improved"] / cv["n_folds"]) >= MIN_FOLD_FRACTION_STRONG and meaningful:
            shuf_strong += 1
        elif cv["n_folds"] > 0 and (cv["folds_improved"] / cv["n_folds"]) >= MIN_FOLD_FRACTION_MODERATE and meaningful:
            shuf_moderate += 1
    print(f"  done in {time.time()-t0:.0f}s")

    real_strong, real_moderate = single_null["real_strong_count"], single_null["real_moderate_count"]
    real_total = single_null["n_hypotheses"]
    real_rate = 100 * (real_strong + real_moderate) / real_total
    shuf_rate = 100 * (shuf_strong + shuf_moderate) / len(shuf_fi)
    single_pass = shuf_rate <= PASS_RATIO_THRESHOLD * real_rate
    overall_pass = overall_pass and single_pass
    real_fold_dist = fold_dist_from_column(single_df["folds_improved"])
    shuf_fold_dist = dict(Counter(shuf_fi))
    lines += [
        "1. SINGLE-PARAMETER FAMILY",
        "-" * 78,
        f"Real: STRONG={real_strong}, MODERATE={real_moderate} / {real_total} tested ({real_rate:.1f}% STRONG+MODERATE)",
        f"Real: {int(single_null['real_survives_fdr_count'])}/{real_total} survive FDR",
        f"Shuffled (1 fresh permutation, same acceptance rule): STRONG={shuf_strong}, MODERATE={shuf_moderate} "
        f"/ {len(shuf_fi)} tested ({shuf_rate:.2f}%)",
        f"Real pooled-null effect-size distribution (delta_r2): mean={single_null['null_delta_r2_mean']:.5f}, "
        f"95th pctile={single_null['null_delta_r2_95th_pctile']:.5f}, 99th pctile={single_null['null_delta_r2_99th_pctile']:.5f} "
        f"(from {single_null['b_perm']} pooled permutations, n={single_null['null_pool_size']})",
        f"Real-target fold-improved distribution (0-5 folds): {real_fold_dist}",
        f"Shuffled-target fold-improved distribution (this fresh permutation): {shuf_fold_dist}",
        f"-> {'PASS' if single_pass else 'FAIL'}: shuffled STRONG+MODERATE rate ({shuf_rate:.2f}%) is "
        f"{'well below' if single_pass else 'NOT clearly below'} {PASS_RATIO_THRESHOLD*100:.0f}% of the real rate ({real_rate:.1f}%).",
        "",
    ]
    print(lines[-2])

    # ---- 2-WAY ----
    print("Fresh shuffled pass: 2-way (1 permutation, same candidate pool, same acceptance rule)...")
    t0 = time.time()
    pool = pd.read_csv(f"{R}/stage2_runB_relationship_validation_pair_screen.csv")
    pool = pool[pool["promoted"]]
    pairs = list(zip(pool["parameter_A"], pool["parameter_B"]))
    rng = np.random.RandomState(RANDOM_STATE + FRESH_SEED_OFFSET + 1)
    y_shuf = group_block_shuffle(y_train_arr, groups_train, rng)
    shuf_fi = []
    shuf_strong, shuf_moderate = 0, 0
    eff95 = pair_null["null_delta_r2_95th_pctile"]
    x_cache = {}
    def get_x(f):
        if f not in x_cache:
            x_cache[f] = split.X_train[ds.name_map[f]].to_numpy(dtype=np.float64)
        return x_cache[f]
    for a, b in pairs:
        cv = pair_cv(get_x(a), get_x(b), y_shuf, split.folds, n_bins=5)
        if cv is None:
            continue
        shuf_fi.append(cv["folds_improved"])
        meaningful = cv["delta_r2"] > eff95
        if cv["n_folds"] > 0 and (cv["folds_improved"] / cv["n_folds"]) >= MIN_FOLD_FRACTION_STRONG and meaningful:
            shuf_strong += 1
        elif cv["n_folds"] > 0 and (cv["folds_improved"] / cv["n_folds"]) >= MIN_FOLD_FRACTION_MODERATE and meaningful:
            shuf_moderate += 1
    print(f"  done in {time.time()-t0:.0f}s")

    real_strong, real_moderate = pair_null["real_strong_count"], pair_null["real_moderate_count"]
    real_total = pair_null["n_hypotheses"]
    real_rate = 100 * (real_strong + real_moderate) / real_total
    shuf_rate = 100 * (shuf_strong + shuf_moderate) / len(shuf_fi)
    pair_pass = shuf_rate <= PASS_RATIO_THRESHOLD * real_rate
    overall_pass = overall_pass and pair_pass
    real_fold_dist = fold_dist_from_column(pair_df["folds_improved"])
    shuf_fold_dist = dict(Counter(shuf_fi))
    lines += [
        "2. 2-WAY FAMILY",
        "-" * 78,
        f"Real: STRONG={real_strong}, MODERATE={real_moderate} / {real_total} tested ({real_rate:.1f}% STRONG+MODERATE)",
        f"Real: {int(pair_null['real_survives_fdr_count'])}/{real_total} survive FDR",
        f"Shuffled (1 fresh permutation, same candidate pool, same acceptance rule): STRONG={shuf_strong}, "
        f"MODERATE={shuf_moderate} / {len(shuf_fi)} tested ({shuf_rate:.2f}%)",
        f"Real pooled-null effect-size distribution (delta_r2): mean={pair_null['null_delta_r2_mean']:.5f}, "
        f"95th pctile={pair_null['null_delta_r2_95th_pctile']:.5f}, 99th pctile={pair_null['null_delta_r2_99th_pctile']:.5f} "
        f"(from {pair_null['b_perm']} pooled permutations, n={pair_null['null_pool_size']})",
        f"Real-target fold-improved distribution: {real_fold_dist}",
        f"Shuffled-target fold-improved distribution (this fresh permutation): {shuf_fold_dist}",
        f"-> {'PASS' if pair_pass else 'FAIL'}: shuffled STRONG+MODERATE rate ({shuf_rate:.2f}%) is "
        f"{'well below' if pair_pass else 'NOT clearly below'} {PASS_RATIO_THRESHOLD*100:.0f}% of the real rate ({real_rate:.1f}%).",
        "",
    ]
    print(lines[-2])

    # ---- 3-WAY ----
    print("Fresh shuffled pass: 3-way (1 permutation, same candidate pool, same acceptance rule)...")
    t0 = time.time()
    cand = pd.read_csv(f"{R}/stage2_runB_relationship_validation_triple_screen.csv")
    triples = list(zip(cand["parameter_A"], cand["parameter_B"], cand["parameter_C"]))
    rng = np.random.RandomState(RANDOM_STATE + FRESH_SEED_OFFSET + 2)
    y_shuf = group_block_shuffle(y_train_arr, groups_train, rng)
    shuf_fi = []
    shuf_strong, shuf_moderate = 0, 0
    eff95 = triple_null["null_delta_r2_95th_pctile"]
    for a, b, c in triples:
        if a not in ds.name_map or b not in ds.name_map or c not in ds.name_map:
            continue
        cv = triple_cv(get_x(a), get_x(b), get_x(c), y_shuf, split.folds, n_bins=2)
        if cv is None:
            continue
        shuf_fi.append(cv["folds_improved"])
        meaningful = cv["delta_r2"] > eff95
        if cv["n_folds"] > 0 and (cv["folds_improved"] / cv["n_folds"]) >= MIN_FOLD_FRACTION_STRONG and meaningful:
            shuf_strong += 1
        elif cv["n_folds"] > 0 and (cv["folds_improved"] / cv["n_folds"]) >= MIN_FOLD_FRACTION_MODERATE and meaningful:
            shuf_moderate += 1
    print(f"  done in {time.time()-t0:.0f}s")

    real_strong, real_moderate = triple_null["real_strong_count"], triple_null["real_moderate_count"]
    real_total = triple_null["n_hypotheses"]
    real_rate = 100 * (real_strong + real_moderate) / real_total
    shuf_rate = 100 * (shuf_strong + shuf_moderate) / max(len(shuf_fi), 1)
    triple_pass = shuf_rate <= PASS_RATIO_THRESHOLD * real_rate
    overall_pass = overall_pass and triple_pass
    real_fold_dist = fold_dist_from_column(triple_df["folds_improved"])
    shuf_fold_dist = dict(Counter(shuf_fi))
    lines += [
        "3. 3-WAY FAMILY",
        "-" * 78,
        f"Real: STRONG={real_strong}, MODERATE={real_moderate} / {real_total} tested ({real_rate:.1f}% STRONG+MODERATE)",
        f"Real: {int(triple_null['real_survives_fdr_count'])}/{real_total} survive FDR",
        f"Shuffled (1 fresh permutation, same candidate pool, same acceptance rule): STRONG={shuf_strong}, "
        f"MODERATE={shuf_moderate} / {len(shuf_fi)} tested ({shuf_rate:.2f}%)",
        f"Real pooled-null effect-size distribution (delta_r2): mean={triple_null['null_delta_r2_mean']:.5f}, "
        f"95th pctile={triple_null['null_delta_r2_95th_pctile']:.5f}, 99th pctile={triple_null['null_delta_r2_99th_pctile']:.5f} "
        f"(from {triple_null['b_perm']} pooled permutations, n={triple_null['null_pool_size']})",
        f"Real-target fold-improved distribution: {real_fold_dist}",
        f"Shuffled-target fold-improved distribution (this fresh permutation): {shuf_fold_dist}",
        f"-> {'PASS' if triple_pass else 'FAIL'}: shuffled STRONG+MODERATE rate ({shuf_rate:.2f}%) is "
        f"{'well below' if triple_pass else 'NOT clearly below'} {PASS_RATIO_THRESHOLD*100:.0f}% of the real rate ({real_rate:.1f}%).",
        "",
    ]
    print(lines[-2])

    lines += [
        "=" * 78,
        f"OVERALL CALIBRATION GATE: {'PASS' if overall_pass else 'FAIL'}",
        "=" * 78,
    ]
    if overall_pass:
        lines.append("All three families show shuffled-target discovery rates far below their real-target "
                      "rates, with clean fold-stability separation. The engine is calibrated reliably enough "
                      "to report its findings as exploratory/statistically-screened evidence. Per instruction, "
                      "ground-truth validation is a SEPARATE, explicitly-gated future checkpoint -- this PASS "
                      "does not itself authorize it.")
    else:
        lines.append("At least one family did NOT show clean separation from its shuffled-target control. "
                      "Per instruction, this is reported as a calibration problem rather than concealed or "
                      "worked around, and no further development (including any ground-truth comparison) "
                      "should proceed on the affected family until recalibrated in a future checkpoint.")

    out_path = f"{R}/stage2_runB_relationship_validation_negative_control.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {out_path}")
    print(f"\nOVERALL GATE: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
