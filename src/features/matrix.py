"""Builds the final modeling feature matrix: raw numeric + categorical
(one-hot) + generic cross-stage engineered features.

Also tracks, for every resulting model-input column, which "source" it came
from (raw / engineered) and which multicollinearity cluster it belongs to
(raw numeric columns only -- engineered features are lower-cardinality
derived signals and are not re-clustered). This mapping is what lets the
explainability layer report both individual-feature and cluster/family-level
importance later.
"""
from __future__ import annotations

import dataclasses
import re

import pandas as pd

from src.data.load import Dataset
from src.features.clustering import FeatureClustering, cluster_features
from src.features.engineered import build_stage_features, find_stage_groups

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_name(name: str) -> str:
    """LightGBM rejects special JSON characters ('.', '[', ']', etc.) in
    feature names. Model-facing column names are sanitized; original,
    human-readable names are preserved via FeatureMatrix.name_map for all
    reporting/plotting/explanation output."""
    return _UNSAFE_CHARS.sub("_", name)


def _sanitize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    seen: dict[str, int] = {}
    new_cols = []
    name_map: dict[str, str] = {}  # original -> sanitized
    for col in df.columns:
        safe = _sanitize_name(col)
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}__{seen[safe]}"
        else:
            seen[safe] = 0
        new_cols.append(safe)
        name_map[col] = safe
    out = df.copy()
    out.columns = new_cols
    return out, name_map


@dataclasses.dataclass
class FeatureMatrix:
    X: pd.DataFrame                  # model-facing matrix, SANITIZED column names
    numeric_cols: list[str]          # raw numeric ET/CD columns (original names)
    engineered_cols: list[str]       # STAGE_RANGE / STAGE_STD / STAGE_DELTA (original names)
    categorical_cols: list[str]      # one-hot encoded categorical dummies (original names)
    clustering: FeatureClustering    # clustering over raw numeric_cols (original names)
    name_map: dict[str, str]         # original column name -> sanitized column name (as used in X)
    inverse_name_map: dict[str, str] # sanitized -> original

    def to_model_cols(self, original_cols: list[str]) -> list[str]:
        """Translate a list of original-name columns into the sanitized
        names actually present in self.X."""
        return [self.name_map[c] for c in original_cols]

    def to_original(self, model_col: str) -> str:
        return self.inverse_name_map.get(model_col, model_col)


def build_feature_matrix(ds: Dataset) -> FeatureMatrix:
    numeric_df = ds.raw[ds.numeric_feature_cols].copy()

    stage_groups = find_stage_groups(ds.numeric_feature_cols)
    engineered_df = build_stage_features(ds.raw, stage_groups)

    cat_df = pd.get_dummies(
        ds.raw[ds.categorical_feature_cols].astype(str),
        prefix=ds.categorical_feature_cols,
        drop_first=False,
    )

    X_raw = pd.concat([numeric_df, engineered_df, cat_df], axis=1)
    X, name_map = _sanitize_columns(X_raw)
    inverse_name_map = {v: k for k, v in name_map.items()}

    clustering = cluster_features(numeric_df)

    return FeatureMatrix(
        X=X,
        numeric_cols=list(numeric_df.columns),
        engineered_cols=list(engineered_df.columns),
        categorical_cols=list(cat_df.columns),
        clustering=clustering,
        name_map=name_map,
        inverse_name_map=inverse_name_map,
    )


if __name__ == "__main__":
    from src.data.load import load_dataset

    ds = load_dataset()
    fm = build_feature_matrix(ds)
    print("Final X shape:", fm.X.shape)
    print("  raw numeric:", len(fm.numeric_cols))
    print("  engineered:", len(fm.engineered_cols))
    print("  categorical (one-hot):", len(fm.categorical_cols))
    print("Any NaNs in X:", fm.X.isna().sum().sum())
    print("Any NaNs from engineered (std of single non-null across 2 modules is fine, no NaN expected):",
          fm.X[fm.engineered_cols].isna().sum().sum())
