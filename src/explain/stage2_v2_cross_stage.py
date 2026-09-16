"""Stage 2 Run B v2, item 5: cross-stage relationship discovery.

Reuses the ALREADY-GENERIC module-prefix matching logic from
src/features/engineered.py's find_stage_groups() (built purely from the
raw column-naming schema -- module prefix before '_ET_'/'_CD_'/etc. --
never from any planted-relationship knowledge). This module does NOT
create or feed any STAGE_DELTA/STAGE_RANGE/STAGE_STD column into
LightGBM; it only computes a raw difference TRANSIENTLY, in-memory, purely
as a statistical analysis quantity for this report, exactly the same way a
Pearson correlation is computed on the fly without becoming a permanent
model feature.

For every pair of original columns sharing a stage-comparable suffix
(module A vs module B measuring "the same" electrical quantity), tests
whether the RAW DIFFERENCE (colA - colB, computed only for this analysis)
shows a stable relationship with yield -- i.e. whether one stage being
higher/lower than another associates with yield, using the same model-free
binned-shape classifier as the univariate screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.stage2_v2_common import (
    R, classify_binned_shape, evidence_label_from_consistency, load_locked_split,
)
from src.features.engineered import find_stage_groups


def main():
    split = load_locked_split()
    ds = split.ds

    stage_groups = find_stage_groups(ds.numeric_cols_original)
    print(f"Stage-comparable suffix groups found (generic, from column-naming structure only): {len(stage_groups.groups)}")

    from itertools import combinations
    pairs = []
    for suffix, modules in stage_groups.groups.items():
        for (modA, colA), (modB, colB) in combinations(sorted(modules.items()), 2):
            pairs.append((suffix, modA, colA, modB, colB))
    print(f"Total cross-stage column pairs to analyze: {len(pairs)}")

    y_train_arr = split.y_train.to_numpy()
    rows = []
    for suffix, modA, colA, modB, colB in pairs:
        mcolA, mcolB = ds.name_map[colA], ds.name_map[colB]
        diff_full = (split.X_train[mcolA] - split.X_train[mcolB]).to_numpy()  # transient, in-memory only
        main_shape, main_dir, _, means = classify_binned_shape(diff_full, y_train_arr, n_bins=6, min_bin_size=20)

        per_fold_shapes, per_fold_dirs = [], []
        for tr, va in split.folds:
            diff_fold = (split.X_train.iloc[va][mcolA] - split.X_train.iloc[va][mcolB]).to_numpy()
            y_fold = split.y_train.iloc[va].to_numpy()
            shp, drc, _, _ = classify_binned_shape(diff_fold, y_fold, n_bins=4, min_bin_size=10)
            per_fold_shapes.append(shp)
            per_fold_dirs.append(drc)

        shape_consistency = sum(1 for s in per_fold_shapes if s == main_shape)
        direction_consistency = sum(1 for d in per_fold_dirs if d == main_dir) if main_dir != "n/a" else shape_consistency
        label, reason = evidence_label_from_consistency(shape_consistency, direction_consistency, 5, main_shape)

        rng = float(means.max() - means.min()) if len(means) else 0.0
        rows.append({
            "parameter_A": colA, "module_A": modA,
            "parameter_B": colB, "module_B": modB,
            "shared_suffix": suffix,
            "stage_relationship_pattern": main_shape,
            "direction_higher_A_minus_B": main_dir,
            "effect_range_yield_pct": round(rng, 4),
            "shape_fold_consistency": f"{shape_consistency}/5",
            "direction_fold_consistency": f"{direction_consistency}/5",
            "evidence_label": label,
            "reason": reason,
        })

    df = pd.DataFrame(rows).sort_values(
        "evidence_label", key=lambda s: s.map({"STRONG": 0, "MODERATE": 1, "WEAK": 2, "INCONCLUSIVE": 3})
    )
    df.to_csv(f"{R}/stage2_runB_v2_cross_stage_relationships.csv", index=False)
    print(f"\nSaved {R}/stage2_runB_v2_cross_stage_relationships.csv ({len(df)} rows, "
          f"{len(df)}/{len(pairs)} coverage = 100% of generically-matched cross-stage pairs)")
    print("\nEvidence label counts:")
    print(df["evidence_label"].value_counts().to_string())
    print("\nTop 10 STRONG cross-stage findings:")
    print(df[df["evidence_label"] == "STRONG"].head(10)[
        ["parameter_A", "parameter_B", "stage_relationship_pattern", "effect_range_yield_pct"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
