"""Group-aware train/validation/test splitting.

Rows are wafers nested within lots (LotName x WaferNum). A naive random
row-level split would let wafers from the same lot appear in both train and
test, which risks optimistic bias if there is any residual lot-level
correlation not fully captured by the recorded features. All splitting in
this project therefore groups by LotName.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclasses.dataclass
class HoldoutSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_holdout_split(
    groups: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> HoldoutSplit:
    """One held-out test set of lots, untouched until final evaluation."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
    return HoldoutSplit(train_idx=train_idx, test_idx=test_idx)


def make_group_kfold(groups: pd.Series, n_splits: int = 5):
    """GroupKFold splitter keyed on LotName, for use on the training portion."""
    gkf = GroupKFold(n_splits=n_splits)
    return gkf, list(gkf.split(np.zeros(len(groups)), groups=groups))


if __name__ == "__main__":
    from src.data.load import load_dataset

    ds = load_dataset()
    holdout = make_holdout_split(ds.groups)
    print("Train rows:", len(holdout.train_idx), "| Test rows:", len(holdout.test_idx))
    print("Train lots:", ds.groups.iloc[holdout.train_idx].nunique(),
          "| Test lots:", ds.groups.iloc[holdout.test_idx].nunique())
    overlap = set(ds.groups.iloc[holdout.train_idx]) & set(ds.groups.iloc[holdout.test_idx])
    print("Lot overlap between train/test (must be empty):", overlap)

    gkf, folds = make_group_kfold(ds.groups.iloc[holdout.train_idx].reset_index(drop=True), n_splits=5)
    for i, (tr, va) in enumerate(folds):
        print(f"fold {i}: train={len(tr)} val={len(va)}")
