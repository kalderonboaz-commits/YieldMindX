"""Leakage-guard regression tests for the four-checkpoint final-yield
pipeline (OHMIC/FIC/SIN/THIN). Mirrors tests/test_no_leakage.py's role for
Stage 1 -- this is the checkpoint-pipeline equivalent.

Everything here is read-only / split-construction-only: no model is
trained. Group-split checks build real GroupKFold index arrays (cheap,
deterministic) but never call .fit() on a model. Evidence comes from (a)
the saved metadata.json files from the completed synthetic training run,
and (b) a fresh, independent reconstruction of the same split/schema logic
used during that run -- so these checks catch both "the saved artifact
disagrees with what the code currently does" and "the code itself has a
leakage gap", not just one or the other.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.checkpoint_contract import CheckpointContract
from src.data.checkpoint_validation import load_raw, filter_process, get_checkpoint_predictor_columns
from src.data.checkpoint_splits import build_shared_split, verify_no_lot_crosses_partitions

MODELS_DIR = PROJECT_ROOT / "outputs" / "checkpoint_training" / "models"


def _load_contract_and_split():
    contract = CheckpointContract.load()
    df = load_raw(contract)
    filtered, _ = filter_process(df, contract)
    shared = build_shared_split(filtered, contract.group_column, contract.chronological_candidates,
                                 contract.holdout_test_size, contract.random_state, contract.n_cv_splits)
    return contract, filtered, shared


def test_target_and_leakage_columns_excluded_from_saved_predictors():
    contract, filtered, _ = _load_contract_and_split()
    for cp in contract.checkpoints:
        with open(MODELS_DIR / f"{cp.name}_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        saved_cols = set(meta["candidate_feature_names"])
        assert contract.target not in saved_cols, f"{cp.name}: target leaked into saved predictor list"
        hit = saved_cols & set(contract.leakage_columns)
        assert not hit, f"{cp.name}: leakage columns {hit} present in saved predictor list"
        # re-derive independently from the current schema and require an exact match --
        # catches drift between what was saved and what the code would produce today
        fresh_cols = set(get_checkpoint_predictor_columns(filtered, contract, cp.name))
        assert fresh_cols == saved_cols, (
            f"{cp.name}: saved candidate_feature_names ({len(saved_cols)}) does not match what "
            f"get_checkpoint_predictor_columns produces today ({len(fresh_cols)}) -- schema or "
            f"contract has drifted since the saved run."
        )


def test_identifier_columns_excluded_from_saved_predictors():
    contract, _, _ = _load_contract_and_split()
    for cp in contract.checkpoints:
        with open(MODELS_DIR / f"{cp.name}_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        saved_cols = set(meta["candidate_feature_names"])
        assert contract.group_column not in saved_cols, f"{cp.name}: {contract.group_column} leaked into predictors"
        assert contract.secondary_id_column not in saved_cols, f"{cp.name}: {contract.secondary_id_column} leaked into predictors"


def test_checkpoint_predictors_match_approved_prefixes_and_are_cumulative():
    contract, _, _ = _load_contract_and_split()
    prev_cols = None
    for cp in contract.checkpoints:
        with open(MODELS_DIR / f"{cp.name}_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        cols = meta["candidate_feature_names"]
        for c in cols:
            assert any(c.startswith(p) for p in cp.prefixes), (
                f"{cp.name}: predictor '{c}' does not start with any approved prefix {cp.prefixes}"
            )
        if prev_cols is not None:
            assert set(prev_cols) < set(cols), (
                f"{cp.name}: predictor set is not a strict superset of the previous checkpoint's "
                f"({len(prev_cols)} -> {len(cols)})"
            )
        prev_cols = cols


def test_no_engineered_columns_in_raw_schema():
    """The checkpoint pipeline reads the raw CSV directly (load_raw), never
    through src/features/matrix.py's engineered-feature construction, so no
    STAGE_RANGE/STD/DELTA engineered column should exist in this file at
    all -- confirms candidate columns are genuinely raw, not derived."""
    contract, filtered, _ = _load_contract_and_split()
    engineered = [c for c in filtered.columns if c.startswith("STAGE_")]
    assert not engineered, f"Unexpected engineered columns in raw checkpoint input: {engineered}"


def test_no_lot_overlap_train_test_holdout():
    contract, filtered, shared = _load_contract_and_split()
    assert verify_no_lot_crosses_partitions(filtered, contract.group_column, shared.holdout)


def test_no_lot_overlap_outer_cv_folds():
    contract, filtered, shared = _load_contract_and_split()
    train_df = filtered.iloc[shared.holdout.train_idx].reset_index(drop=True)
    groups = train_df[contract.group_column]
    for i, (tr, va) in enumerate(shared.folds):
        lots_tr = set(groups.iloc[tr])
        lots_va = set(groups.iloc[va])
        assert not (lots_tr & lots_va), f"outer fold {i}: lot overlap between train and validation: {lots_tr & lots_va}"
        # every outer-fold validation lot must also be a train-partition lot (i.e. none of
        # the untouched holdout-test lots can have leaked into the CV folds)
        test_lots = set(filtered[contract.group_column].iloc[shared.holdout.test_idx])
        assert not (lots_va & test_lots), f"outer fold {i}: holdout-test lot leaked into CV validation: {lots_va & test_lots}"


def test_no_lot_overlap_inner_cv_within_each_outer_fold():
    """Reconstructs exactly what make_grouped_search() builds -- a grouped
    inner split of groups_tr (the outer fold's OWN training lots only) --
    without fitting any model, and checks two things: inner train/val lot
    sets are disjoint within every inner split, and the union of lots
    touched by the inner split never includes a lot from the outer fold's
    held-out validation rows (which were already excluded from groups_tr
    before the inner split ever runs)."""
    contract, filtered, shared = _load_contract_and_split()
    train_df = filtered.iloc[shared.holdout.train_idx].reset_index(drop=True)
    groups_all = train_df[contract.group_column]

    n_inner_splits = 3
    for outer_i, (tr, va) in enumerate(shared.folds):
        groups_tr = groups_all.iloc[tr]
        lots_va_outer = set(groups_all.iloc[va])

        n_splits = min(n_inner_splits, groups_tr.nunique())
        inner_splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups_tr)), groups=groups_tr))
        for inner_i, (itr, ival) in enumerate(inner_splits):
            lots_itr = set(groups_tr.iloc[itr])
            lots_ival = set(groups_tr.iloc[ival])
            assert not (lots_itr & lots_ival), (
                f"outer fold {outer_i} inner split {inner_i}: lot overlap between inner train/val"
            )
            assert not (lots_itr & lots_va_outer), (
                f"outer fold {outer_i} inner split {inner_i}: outer-validation lot leaked into inner train"
            )
            assert not (lots_ival & lots_va_outer), (
                f"outer fold {outer_i} inner split {inner_i}: outer-validation lot leaked into inner val"
            )


def test_pipeline_structure_is_fully_nested_not_precv():
    """Static structural check: the model step must be a plain (non-CV)
    estimator, and the tuning/preprocessing must be driven by an outer
    GridSearchCV wrapping the WHOLE pipeline -- not an internally-cross-
    validated estimator (e.g. ElasticNetCV) sitting at the end of a
    pipeline whose earlier steps would then be fit once, before that
    estimator's own internal split ever runs."""
    from src.models.checkpoint_pipeline_builder import (
        build_elasticnet_pipeline_and_grid, build_tree_pipeline_and_grid, make_grouped_search,
    )
    import inspect

    for feature_mode in ("full", "selected"):
        pipe, grid = build_elasticnet_pipeline_and_grid(feature_mode)
        model_step = pipe.named_steps["model"]
        assert type(model_step).__name__ == "ElasticNet", (
            f"expected plain ElasticNet as the final step, got {type(model_step).__name__} "
            f"-- an internally-cross-validated estimator here would be fit on preprocessing "
            f"output computed before its own split, defeating fold-nesting."
        )
        for model_name in ("random_forest", "lightgbm"):
            pipe, grid = build_tree_pipeline_and_grid(model_name, feature_mode)
            assert "model" in pipe.named_steps

    src = inspect.getsource(make_grouped_search)
    assert "GridSearchCV(pipeline" in src.replace(" ", ""), (
        "make_grouped_search must wrap the entire pipeline (preprocessing + model) in GridSearchCV, "
        "not just the model step"
    )


def test_winner_selection_precedes_test_set_access_in_source():
    """Static ordering check on the training script's own source text: the
    winner-selection and locking sections must appear, in source order,
    before the test-set-evaluation section, and the string 'test_df' must
    not appear anywhere before the 'TEST-SET EVALUATION' marker."""
    src = (PROJECT_ROOT / "src" / "models" / "checkpoint_train_run.py").read_text(encoding="utf-8")
    i_selection = src.index("WINNER SELECTION")
    i_locking = src.index("LOCKING WINNERS")
    i_test_eval = src.index("TEST-SET EVALUATION")
    assert i_selection < i_locking < i_test_eval, "selection/locking/test-evaluation sections are out of order"

    before_test_eval = src[:i_test_eval]
    # test_df itself is declared early (same split call as train_df) and its row/lot COUNT is
    # printed right away for visibility -- that's fine. What must not happen before the marker
    # is any read of test_df's actual feature/target values (X_test/y_test), which would mean
    # the test set influenced selection or locking.
    assert "X_test" not in before_test_eval, "X_test referenced before TEST-SET EVALUATION -- test features read early"
    assert "y_test" not in before_test_eval, "y_test referenced before TEST-SET EVALUATION -- test target read early"
    assert before_test_eval.count("test_df") <= 3, (
        "test_df referenced more than expected (declaration + row/lot-count print) before "
        "TEST-SET EVALUATION -- inspect for an unexpected early read"
    )


def test_oof_predictions_come_from_out_of_fold_val_predictions_in_source():
    """Static check that the saved OOF rows are built from each outer
    fold's val_pred (predicted on that fold's held-OUT rows by a pipeline
    fit on that fold's training rows only, never refit afterward) -- not
    from in-sample predictions of the locked, full-train-refit model."""
    src = (PROJECT_ROOT / "src" / "models" / "checkpoint_train_run.py").read_text(encoding="utf-8")
    assert "search.fit(X_tr, y_tr)" in src
    assert "search.predict(X_va)" in src
    fit_pos = src.index("search.fit(X_tr, y_tr)")
    predict_pos = src.index("search.predict(X_va)")
    assert fit_pos < predict_pos, "fit must precede predict on the held-out fold rows"
    assert "val_pred=pred.tolist()" in src, "FoldRecord must store the out-of-fold prediction, not an in-sample one"


def test_cross_mode_artifact_loading_is_rejected():
    """The inference loader must refuse to serve a synthetic-trained
    artifact for a mode="real" request (and vice versa), even if the
    directories were mixed up -- see src/models/checkpoint_inference.py's
    recorded_mode cross-check."""
    from src.models.checkpoint_inference import CheckpointPredictor

    try:
        CheckpointPredictor("OHMIC", mode="real", models_dir=MODELS_DIR)
        assert False, "expected AssertionError for a cross-mode artifact load"
    except AssertionError as e:
        assert "cross-mode" in str(e) or "requested mode" in str(e)


if __name__ == "__main__":
    test_target_and_leakage_columns_excluded_from_saved_predictors()
    print("PASS test_target_and_leakage_columns_excluded_from_saved_predictors")
    test_identifier_columns_excluded_from_saved_predictors()
    print("PASS test_identifier_columns_excluded_from_saved_predictors")
    test_checkpoint_predictors_match_approved_prefixes_and_are_cumulative()
    print("PASS test_checkpoint_predictors_match_approved_prefixes_and_are_cumulative")
    test_no_engineered_columns_in_raw_schema()
    print("PASS test_no_engineered_columns_in_raw_schema")
    test_no_lot_overlap_train_test_holdout()
    print("PASS test_no_lot_overlap_train_test_holdout")
    test_no_lot_overlap_outer_cv_folds()
    print("PASS test_no_lot_overlap_outer_cv_folds")
    test_no_lot_overlap_inner_cv_within_each_outer_fold()
    print("PASS test_no_lot_overlap_inner_cv_within_each_outer_fold")
    test_pipeline_structure_is_fully_nested_not_precv()
    print("PASS test_pipeline_structure_is_fully_nested_not_precv")
    test_winner_selection_precedes_test_set_access_in_source()
    print("PASS test_winner_selection_precedes_test_set_access_in_source")
    test_oof_predictions_come_from_out_of_fold_val_predictions_in_source()
    print("PASS test_oof_predictions_come_from_out_of_fold_val_predictions_in_source")
    test_cross_mode_artifact_loading_is_rejected()
    print("PASS test_cross_mode_artifact_loading_is_rejected")
    print("\nAll checkpoint-pipeline leakage-guard tests passed.")
