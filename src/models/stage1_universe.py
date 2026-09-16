"""Stage 1 (Yield Prediction Engine) predictor universe: ORIGINAL raw-CSV
predictors only.

Per this checkpoint's explicit instruction, all 307 engineered features
(STAGE_RANGE_*, STAGE_STD_*, STAGE_DELTA_*, produced by
src/features/engineered.py) are completely excluded from Stage 1. This is
enforced with a hard assertion, not just a convention -- any code path that
accidentally includes an engineered column in the Stage 1 predictor list
will raise immediately rather than silently producing a mixed feature set.

This exclusion is SCOPED TO STAGE 1 ONLY. The engineered features
themselves are not deleted from the project and remain fully available for
Stage 2 (Relationship Discovery), which is not touched by this module.
"""
from __future__ import annotations

from src.features.matrix import FeatureMatrix


def get_stage1_predictors(fm: FeatureMatrix) -> list[str]:
    """Original-CSV-only predictor list for Stage 1: raw numeric ET/CD
    columns + categorical one-hot dummy columns. No engineered feature is
    ever included."""
    predictors = list(fm.numeric_cols) + list(fm.categorical_cols)
    assert_no_engineered_features(predictors, fm)
    return predictors


def assert_no_engineered_features(cols: list[str], fm: FeatureMatrix) -> None:
    """Hard programmatic guard: raises if any engineered feature (by name
    match against fm.engineered_cols, or by the STAGE_ prefix convention
    used by every engineered feature) is present in `cols`."""
    engineered_set = set(fm.engineered_cols)
    violations = [c for c in cols if c in engineered_set or c.startswith("STAGE_")]
    assert not violations, (
        f"Engineered feature(s) leaked into the Stage 1 predictor universe: {violations[:10]}"
        f"{'...' if len(violations) > 10 else ''}. Stage 1 must use ONLY original raw-CSV predictors."
    )


if __name__ == "__main__":
    from src.data.load import load_dataset
    from src.features.matrix import build_feature_matrix

    ds = load_dataset()
    fm = build_feature_matrix(ds)
    stage1_cols = get_stage1_predictors(fm)

    print(f"Raw numeric predictors: {len(fm.numeric_cols)}")
    print(f"Categorical predictors (one-hot dummies): {len(fm.categorical_cols)}")
    print(f"Total Stage 1 original safe predictors (before reduction): {len(stage1_cols)}")
    print(f"Engineered features excluded: {len(fm.engineered_cols)}")

    assert_no_engineered_features(stage1_cols, fm)
    print("\nHard assertion PASSED: no engineered feature present in the Stage 1 predictor universe.")

    # negative-control test: prove the assertion actually catches a violation
    try:
        assert_no_engineered_features(stage1_cols + [fm.engineered_cols[0]], fm)
        print("ERROR: assertion did not fire on a deliberately injected engineered feature!")
    except AssertionError as e:
        print(f"Negative-control check PASSED (assertion correctly fired): {str(e)[:100]}...")
