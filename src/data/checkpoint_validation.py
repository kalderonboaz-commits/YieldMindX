"""Schema/process/duplicate/target validation and per-checkpoint column
inventories for the four-checkpoint final-yield prediction pipeline.

Pure read/validate -- never fits, trains, or mutates the input file. Every
function here either returns a report object or raises/asserts loudly on a
genuine data problem; nothing here silently "fixes" a bad row.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from src.data.checkpoint_contract import CheckpointContract


@dataclasses.dataclass
class SchemaCheck:
    required_columns_present: bool
    missing_required_columns: list[str]
    process_column_present: bool


@dataclasses.dataclass
class ProcessFilterReport:
    rows_before: int
    rows_after: int
    rows_dropped: int
    lots_before: int
    lots_after: int
    other_process_values_seen: dict  # value -> row count, for everything NOT matching process_value


@dataclasses.dataclass
class DuplicateIdReport:
    duplicate_row_count: int
    duplicate_keys_sample: list[tuple]


@dataclasses.dataclass
class TargetCheck:
    column: str
    dtype: str
    n_missing: int
    n_rows: int
    min_value: float
    max_value: float
    mean_value: float
    plausible_percentage_range: bool  # True if min/max both fall within [0, 100]


@dataclasses.dataclass
class CheckpointColumnInventory:
    name: str
    order: int
    prefixes: list[str]
    n_predictor_columns: int
    n_leakage_columns_excluded: int
    leakage_columns_found_in_prefix_range: list[str]
    sample_columns: list[str]


def load_raw(contract: CheckpointContract) -> pd.DataFrame:
    df = pd.read_csv(contract.input_csv_path, low_memory=False)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    return df


def check_schema(df: pd.DataFrame, contract: CheckpointContract) -> SchemaCheck:
    required = {contract.target, contract.group_column, contract.secondary_id_column, contract.process_column}
    missing = sorted(required - set(df.columns))
    return SchemaCheck(
        required_columns_present=len(missing) == 0,
        missing_required_columns=missing,
        process_column_present=contract.process_column in df.columns,
    )


def filter_process(df: pd.DataFrame, contract: CheckpointContract) -> tuple[pd.DataFrame, ProcessFilterReport]:
    rows_before = len(df)
    lots_before = df[contract.group_column].nunique() if contract.group_column in df.columns else -1

    mask = df[contract.process_column].astype(str) == str(contract.process_value)
    other = df.loc[~mask, contract.process_column].astype(str).value_counts().to_dict()

    filtered = df.loc[mask].reset_index(drop=True)
    report = ProcessFilterReport(
        rows_before=rows_before,
        rows_after=len(filtered),
        rows_dropped=rows_before - len(filtered),
        lots_before=lots_before,
        lots_after=filtered[contract.group_column].nunique() if contract.group_column in filtered.columns else -1,
        other_process_values_seen=other,
    )
    return filtered, report


def check_duplicate_ids(df: pd.DataFrame, contract: CheckpointContract) -> DuplicateIdReport:
    key_cols = [contract.group_column, contract.secondary_id_column]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_rows = df.loc[dup_mask, key_cols]
    sample = list(dup_rows.drop_duplicates().itertuples(index=False, name=None))[:10]
    return DuplicateIdReport(duplicate_row_count=int(dup_mask.sum()), duplicate_keys_sample=sample)


def check_target(df: pd.DataFrame, contract: CheckpointContract) -> TargetCheck:
    col = contract.target
    s = df[col]
    mn, mx = float(s.min()), float(s.max())
    return TargetCheck(
        column=col,
        dtype=str(s.dtype),
        n_missing=int(s.isna().sum()),
        n_rows=len(s),
        min_value=mn,
        max_value=mx,
        mean_value=float(s.mean()),
        plausible_percentage_range=(0.0 <= mn <= 100.0 and 0.0 <= mx <= 100.0),
    )


def build_checkpoint_column_inventories(
    df: pd.DataFrame, contract: CheckpointContract
) -> list[CheckpointColumnInventory]:
    inventories = []
    for cp in contract.checkpoints:
        predictor_cols = [
            c for c in df.columns
            if any(c.startswith(p) for p in cp.prefixes)
        ]
        leakage_hits = [c for c in predictor_cols if c in contract.leakage_columns]
        clean_cols = [c for c in predictor_cols if c not in contract.leakage_columns]
        inventories.append(CheckpointColumnInventory(
            name=cp.name,
            order=cp.order,
            prefixes=cp.prefixes,
            n_predictor_columns=len(clean_cols),
            n_leakage_columns_excluded=len(leakage_hits),
            leakage_columns_found_in_prefix_range=leakage_hits,
            sample_columns=clean_cols[:5],
        ))
    return inventories


def get_checkpoint_predictor_columns(
    df: pd.DataFrame, contract: CheckpointContract, checkpoint_name: str
) -> list[str]:
    cp = next(c for c in contract.checkpoints if c.name == checkpoint_name)
    cols = [c for c in df.columns if any(c.startswith(p) for p in cp.prefixes)]
    return [c for c in cols if c not in contract.leakage_columns]


def run_all_checks(contract: CheckpointContract) -> dict:
    df = load_raw(contract)
    schema = check_schema(df, contract)
    assert schema.required_columns_present, f"Missing required columns: {schema.missing_required_columns}"

    filtered, process_report = filter_process(df, contract)
    dup_report = check_duplicate_ids(filtered, contract)
    target_report = check_target(filtered, contract)
    inventories = build_checkpoint_column_inventories(filtered, contract)

    return {
        "mode": contract.mode,
        "input_csv_path": str(contract.input_csv_path),
        "schema": schema,
        "process_filter": process_report,
        "duplicates": dup_report,
        "target": target_report,
        "checkpoint_inventories": inventories,
        "filtered_df": filtered,
    }


if __name__ == "__main__":
    contract = CheckpointContract.load()
    results = run_all_checks(contract)
    print(f"Mode: {results['mode']} | Input: {results['input_csv_path']}")
    print(f"Schema OK: {results['schema'].required_columns_present}")
    pf = results["process_filter"]
    print(f"Process filter: {pf.rows_before} -> {pf.rows_after} rows "
          f"({pf.lots_before} -> {pf.lots_after} lots); other values seen: {pf.other_process_values_seen}")
    print(f"Duplicate LotName+WaferNum rows: {results['duplicates'].duplicate_row_count}")
    t = results["target"]
    print(f"Target '{t.column}': n={t.n_rows} missing={t.n_missing} "
          f"range=[{t.min_value:.3f}, {t.max_value:.3f}] plausible_pct_range={t.plausible_percentage_range}")
    for inv in results["checkpoint_inventories"]:
        print(f"  Checkpoint {inv.order} '{inv.name}': {inv.n_predictor_columns} predictors "
              f"(leakage excluded: {inv.n_leakage_columns_excluded}) sample={inv.sample_columns}")
