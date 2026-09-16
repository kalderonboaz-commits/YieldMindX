"""Stage 2 (Relationship Discovery Engine) feature matrix builder.

Builds the model-input matrix directly from the raw CSV using the
APPROVED Stage 2 feature-eligibility audit
(outputs/reports/stage2_feature_eligibility.csv) as the single source of
truth for which columns are eligible -- not the Stage 1 data contract, and
not src/features/matrix.py's build_feature_matrix (which generates the 307
engineered features Stage 2 must not use).

Predictor set = category B (legitimate process/ET, numeric) + category C
(categorical process parameters, one-hot encoded) from the audit, plus
optionally the four category-G "review" columns (LineYield, BackEndYield,
VIYield, LTYield) when explicitly requested -- this is the only axis of
variation this module exposes, by design, so the two-run comparison
experiment cannot accidentally differ on anything else.
"""
from __future__ import annotations

import dataclasses
import re

import pandas as pd

RAW_CSV_PATH = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\data\correlation_data_table.csv"
AUDIT_CSV_PATH = r"D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table\outputs\reports\stage2_feature_eligibility.csv"

TARGET = "SortingYield"
GROUP_COL = "LotName"
REVIEW_COLUMNS = ["LineYield", "BackEndYield", "VIYield", "LTYield"]

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(name: str) -> str:
    return _UNSAFE_CHARS.sub("_", name)


@dataclasses.dataclass
class Stage2Dataset:
    X: pd.DataFrame            # sanitized column names, model-ready
    y: pd.Series
    groups: pd.Series
    numeric_cols_original: list[str]
    categorical_dummy_cols: list[str]
    review_cols_original: list[str]   # empty if include_review_columns=False
    name_map: dict[str, str]          # original -> sanitized
    inverse_name_map: dict[str, str]


def load_stage2_dataset(include_review_columns: bool) -> Stage2Dataset:
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]

    audit = pd.read_csv(AUDIT_CSV_PATH)
    numeric_cols = audit[audit["category"] == "B"]["column_name"].tolist()
    categorical_cols = audit[audit["category"] == "C"]["column_name"].tolist()

    assert TARGET not in numeric_cols and TARGET not in categorical_cols
    for c in REVIEW_COLUMNS:
        assert c not in numeric_cols and c not in categorical_cols, (
            f"{c} must come ONLY from the explicit include_review_columns flag, not the base B/C audit categories"
        )

    review_cols = list(REVIEW_COLUMNS) if include_review_columns else []
    all_numeric = numeric_cols + review_cols

    numeric_df = df[all_numeric].copy()
    cat_df = pd.get_dummies(df[categorical_cols].astype(str), prefix=categorical_cols, drop_first=False)

    X_raw = pd.concat([numeric_df, cat_df], axis=1)

    # hard guard: no engineered feature (STAGE_ prefix) can ever appear
    engineered_leak = [c for c in X_raw.columns if c.startswith("STAGE_")]
    assert not engineered_leak, f"Engineered feature(s) leaked into Stage 2 matrix: {engineered_leak}"

    name_map = {}
    seen = {}
    new_cols = []
    for col in X_raw.columns:
        safe = _sanitize(col)
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}__{seen[safe]}"
        else:
            seen[safe] = 0
        name_map[col] = safe
        new_cols.append(safe)
    X = X_raw.copy()
    X.columns = new_cols
    inverse_name_map = {v: k for k, v in name_map.items()}

    return Stage2Dataset(
        X=X, y=df[TARGET], groups=df[GROUP_COL],
        numeric_cols_original=numeric_cols, categorical_dummy_cols=list(cat_df.columns),
        review_cols_original=review_cols, name_map=name_map, inverse_name_map=inverse_name_map,
    )


if __name__ == "__main__":
    for include in (False, True):
        ds = load_stage2_dataset(include_review_columns=include)
        print(f"include_review_columns={include}: X shape={ds.X.shape} "
              f"(numeric_B={len(ds.numeric_cols_original)}, "
              f"review={len(ds.review_cols_original)}, categorical_dummies={len(ds.categorical_dummy_cols)})")
        assert ds.X.isna().sum().sum() == 0
    print("OK -- both Stage 2 matrix variants build cleanly with no engineered features.")
