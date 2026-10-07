"""The ONE shared LotName-grouped train/validation/test assignment reused
identically across all four checkpoints, plus an assessment (not a forced
switch) of whether a chronological split would also be reliable.

Reuses src/data/splits.py's make_holdout_split/make_group_kfold UNMODIFIED
-- this module only decides what grouping key/order to feed them, exactly
once, so every checkpoint is evaluated on the same lots in the same roles.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.data.splits import HoldoutSplit, make_group_kfold, make_holdout_split


@dataclasses.dataclass
class ChronologicalAssessment:
    candidate_column: str | None
    usable: bool
    reason: str
    n_unique: int | None = None
    n_missing: int | None = None


@dataclasses.dataclass
class SharedSplit:
    holdout: HoldoutSplit          # random, LotName-grouped (primary split used for training)
    folds: list                    # GroupKFold folds over the holdout.train_idx portion
    chronological: ChronologicalAssessment
    chronological_holdout: HoldoutSplit | None  # populated only if a chronological split was usable


def assess_chronological_split(df: pd.DataFrame, group_column: str, candidates: list[str]) -> ChronologicalAssessment:
    for col in candidates:
        if col not in df.columns:
            continue
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique())
        if n_missing > 0:
            continue  # try the next candidate rather than accept a column with gaps
        if n_unique < 2:
            continue  # constant column, e.g. FinishLineYear -- cannot order anything by it
        return ChronologicalAssessment(
            candidate_column=col, usable=True,
            reason=f"'{col}' has {n_unique} unique values and 0 missing -- usable for chronological ordering",
            n_unique=n_unique, n_missing=n_missing,
        )
    return ChronologicalAssessment(
        candidate_column=None, usable=False,
        reason="No candidate chronological column had both 0 missing values and >1 unique value",
    )


def _parse_order_key(df: pd.DataFrame, column: str) -> pd.Series:
    """Build a sortable key from the chosen chronological column without
    assuming a specific date format -- tries pandas date parsing first
    (handles 'Sorting Date' dd/mm/yyyy strings), falls back to the raw
    value (handles 'Year_WW'/'Sort_WW', which sort correctly as strings/ints
    already)."""
    try:
        parsed = pd.to_datetime(df[column], dayfirst=True, errors="raise")
        return parsed
    except Exception:
        return df[column]


def make_chronological_holdout(
    df: pd.DataFrame, group_column: str, order_column: str, test_size: float
) -> HoldoutSplit:
    """Assign the LAST test_size share of LOTS (by each lot's own earliest
    timestamp) to the test set -- simulates predicting future lots from
    past ones. A lot is never split across partitions: the grouping key is
    the lot's own earliest timestamp, and ALL of that lot's rows move
    together."""
    order_key = _parse_order_key(df, order_column)
    lot_order = (
        pd.DataFrame({group_column: df[group_column], "_order_key": order_key})
        .groupby(group_column)["_order_key"].min()
        .sort_values()
    )
    n_lots = len(lot_order)
    n_test_lots = max(1, round(n_lots * test_size))
    test_lots = set(lot_order.index[-n_test_lots:])

    test_mask = df[group_column].isin(test_lots)
    test_idx = np.where(test_mask.to_numpy())[0]
    train_idx = np.where(~test_mask.to_numpy())[0]
    return HoldoutSplit(train_idx=train_idx, test_idx=test_idx)


def build_shared_split(
    df: pd.DataFrame, group_column: str, chronological_candidates: list[str],
    test_size: float, random_state: int, n_cv_splits: int,
) -> SharedSplit:
    holdout = make_holdout_split(df[group_column], test_size=test_size, random_state=random_state)
    groups_train = df[group_column].iloc[holdout.train_idx].reset_index(drop=True)
    _, folds = make_group_kfold(groups_train, n_splits=n_cv_splits)

    chrono_assessment = assess_chronological_split(df, group_column, chronological_candidates)
    chrono_holdout = None
    if chrono_assessment.usable:
        chrono_holdout = make_chronological_holdout(
            df, group_column, chrono_assessment.candidate_column, test_size
        )

    return SharedSplit(
        holdout=holdout, folds=folds,
        chronological=chrono_assessment, chronological_holdout=chrono_holdout,
    )


def verify_no_lot_crosses_partitions(df: pd.DataFrame, group_column: str, split: HoldoutSplit) -> bool:
    train_lots = set(df[group_column].iloc[split.train_idx])
    test_lots = set(df[group_column].iloc[split.test_idx])
    return len(train_lots & test_lots) == 0


if __name__ == "__main__":
    from src.data.checkpoint_contract import CheckpointContract
    from src.data.checkpoint_validation import load_raw, filter_process

    contract = CheckpointContract.load()
    df = load_raw(contract)
    filtered, _ = filter_process(df, contract)

    shared = build_shared_split(
        filtered, contract.group_column, contract.chronological_candidates,
        contract.holdout_test_size, contract.random_state, contract.n_cv_splits,
    )
    print(f"Random grouped holdout: train={len(shared.holdout.train_idx)} rows "
          f"({filtered[contract.group_column].iloc[shared.holdout.train_idx].nunique()} lots), "
          f"test={len(shared.holdout.test_idx)} rows "
          f"({filtered[contract.group_column].iloc[shared.holdout.test_idx].nunique()} lots)")
    print("No-lot-crosses-partitions check:", verify_no_lot_crosses_partitions(filtered, contract.group_column, shared.holdout))
    print(f"CV folds: {len(shared.folds)}")

    print(f"\nChronological assessment: usable={shared.chronological.usable} -- {shared.chronological.reason}")
    if shared.chronological_holdout:
        ch = shared.chronological_holdout
        print(f"Chronological holdout: train={len(ch.train_idx)} rows "
              f"({filtered[contract.group_column].iloc[ch.train_idx].nunique()} lots), "
              f"test={len(ch.test_idx)} rows ({filtered[contract.group_column].iloc[ch.test_idx].nunique()} lots)")
        print("No-lot-crosses-partitions check:", verify_no_lot_crosses_partitions(filtered, contract.group_column, ch))
