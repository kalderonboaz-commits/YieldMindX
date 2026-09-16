"""Single entrypoint for loading the correlation data table.

Every other module must load data through `load_dataset()` rather than
reading the CSV directly, so the leakage/id/target contract is applied
consistently everywhere.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "correlation_data_table.csv"
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "config" / "data_contract.yaml"


@dataclasses.dataclass
class DataContract:
    target: str
    leakage_columns: list[str]
    id_columns: list[str]
    group_column: str
    categorical_columns: list[str]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONTRACT_PATH) -> "DataContract":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            target=raw["target"],
            leakage_columns=raw["leakage_columns"],
            id_columns=raw["id_columns"],
            group_column=raw["group_column"],
            categorical_columns=raw["categorical_columns"],
        )


@dataclasses.dataclass
class Dataset:
    """Container separating target / groups / metadata / features cleanly."""

    raw: pd.DataFrame              # full frame, columns BOM-fixed, nothing dropped
    contract: DataContract
    y: pd.Series                   # target column
    groups: pd.Series              # group_column, for GroupKFold etc.
    numeric_feature_cols: list[str]
    categorical_feature_cols: list[str]

    @property
    def feature_cols(self) -> list[str]:
        return self.numeric_feature_cols + self.categorical_feature_cols

    @property
    def X(self) -> pd.DataFrame:
        return self.raw[self.feature_cols]


def _fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("﻿", "") for c in df.columns]
    return df


def load_dataset(
    csv_path: Path = DEFAULT_CSV_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> Dataset:
    contract = DataContract.load(contract_path)

    df = pd.read_csv(csv_path, low_memory=False)
    df = _fix_columns(df)

    assert contract.target in df.columns, f"Target '{contract.target}' not found in CSV"
    assert df[contract.target].isna().sum() == 0, "Target has missing values -- decide handling before proceeding"

    excluded = set(contract.leakage_columns) | set(contract.id_columns) | {contract.target}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_feature_cols = [
        c for c in numeric_cols
        if c not in excluded and c not in contract.categorical_columns
    ]
    categorical_feature_cols = [c for c in contract.categorical_columns if c in df.columns]

    # sanity: no leakage or id column may have leaked into the feature set
    overlap = (set(numeric_feature_cols) | set(categorical_feature_cols)) & excluded
    assert not overlap, f"Leakage/id columns leaked into feature set: {overlap}"

    return Dataset(
        raw=df,
        contract=contract,
        y=df[contract.target],
        groups=df[contract.group_column],
        numeric_feature_cols=numeric_feature_cols,
        categorical_feature_cols=categorical_feature_cols,
    )


if __name__ == "__main__":
    ds = load_dataset()
    print("Rows:", ds.raw.shape[0])
    print("Target:", ds.contract.target, "| mean:", ds.y.mean().round(3))
    print("Groups (unique lots):", ds.groups.nunique())
    print("Numeric feature cols:", len(ds.numeric_feature_cols))
    print("Categorical feature cols:", len(ds.categorical_feature_cols))
    print("Total feature cols:", len(ds.feature_cols))
