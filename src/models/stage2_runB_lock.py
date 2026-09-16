"""Locks and re-verifies the Run B Stage 2 feature universe as the
baseline for all subsequent Stage 2 relationship-discovery work.

Explicit, hard-asserted checks (not assumed from prior checkpoints):
  - SortingYield (target) is not a predictor
  - Actual Sorting Yield (%), TotalYield, EstimatedSupplyChipsCount (confirmed leakage) excluded
  - LineYield, BackEndYield, VIYield, LTYield (review columns) excluded from Run B
  - metadata/ID/date/traceability columns excluded
  - MaskSetName_group (exact duplicate) excluded
  - zero engineered features (no STAGE_ prefix) present
  - the Stage 1 283-feature subset is NOT used as the Stage 2 universe (Stage 2 uses
    the full 1,425-column Run B set, independently derived from the Stage 2 audit)
"""
from __future__ import annotations

import pandas as pd

from src.features.stage2_matrix import REVIEW_COLUMNS, load_stage2_dataset

CONFIRMED_LEAKAGE = ["Actual Sorting Yield (%)", "TotalYield", "EstimatedSupplyChipsCount"]
METADATA_ID_DATE = ["Sorting Date", "FinishLineDate", "LotName", "WaferNum", "Process",
                     "FinishLineYear", "Exception", "Sort_WW", "Year_WW"]
EXACT_DUPLICATE = ["MaskSetName_group"]
TARGET = "SortingYield"


def lock_and_verify() -> dict:
    ds = load_stage2_dataset(include_review_columns=False)
    original_cols = set(ds.numeric_cols_original) | set(ds.name_map.keys())

    checks = {}

    checks["target_excluded"] = TARGET not in original_cols
    checks["confirmed_leakage_excluded"] = all(c not in original_cols for c in CONFIRMED_LEAKAGE)
    checks["review_columns_excluded"] = all(c not in original_cols for c in REVIEW_COLUMNS)
    checks["metadata_excluded"] = all(c not in original_cols for c in METADATA_ID_DATE)
    checks["duplicate_excluded"] = all(c not in original_cols for c in EXACT_DUPLICATE)
    checks["zero_engineered_features"] = not any(c.startswith("STAGE_") for c in original_cols)
    checks["predictor_count_is_1425"] = ds.X.shape[1] == 1425
    checks["no_missing_values"] = ds.X.isna().sum().sum() == 0

    for name, ok in checks.items():
        assert ok, f"Stage 2 Run B lock FAILED: {name}"

    return checks, ds


if __name__ == "__main__":
    checks, ds = lock_and_verify()
    print("=== STAGE 2 RUN B FEATURE UNIVERSE -- LOCKED AND VERIFIED ===")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nFinal Run B predictor matrix: {ds.X.shape[0]} rows x {ds.X.shape[1]} predictors")
    print(f"  Numeric process/ET predictors (category B): {len(ds.numeric_cols_original)}")
    print(f"  Categorical one-hot dummy predictors (category C): {len(ds.categorical_dummy_cols)}")
    print(f"  Review columns included: {len(ds.review_cols_original)} (must be 0 for Run B)")
    print("\nAll checks passed. Run B is now the locked Stage 2 baseline.")
