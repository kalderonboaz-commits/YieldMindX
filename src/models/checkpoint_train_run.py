"""Four-checkpoint model training/selection/evaluation run.

24 candidate configurations: {ElasticNet, RandomForest, LightGBM} x
{full, selected} x {OHMIC, FIC, SIN, THIN}. Uses the ONE shared
LotName-grouped outer split (checkpoint_splits.py) for every checkpoint,
and the fully fold-nested pipelines (checkpoint_pipeline_builder.py) for
every candidate -- imputation, constant-removal, redundancy selection,
scaling, and hyperparameter search all refit inside each inner training
fold, never once globally.

Protocol:
  1. For each of the 24 configs, run the 5 OUTER grouped folds. Within each
     outer fold, GridSearchCV with a grouped INNER cv drives hyperparameter
     selection on that fold's training rows only; the resulting
     best_estimator_ (refit on the full outer-fold training rows) predicts
     the outer fold's held-out validation rows. Metrics + raw OOF
     predictions are recorded per fold.
  2. Per checkpoint, aggregate the 6 (model x mode) candidates' CV RMSE and
     pick ONE winner via the predefined 1-SE rule + simplicity ordering
     (actual mean retained-feature count, then ElasticNet<RF<LightGBM,
     then lower RMSE std, then lower fit time, then best mean RMSE).
  3. Lock the winner: refit ONE final pipeline via grouped GridSearchCV on
     the FULL 80-lot train partition (not touching the test set).
  4. ONLY AFTER all four winners are locked, evaluate each on the
     untouched 20-lot test partition. No selection is revised afterward.
  5. Save fitted pipeline, metadata (hyperparameters, retained features,
     split provenance), and LotName/WaferNum-keyed predictions (grouped-OOF
     for train rows from step 1's winner-fold predictions, holdout for
     test rows from step 4) per checkpoint.

Mode (synthetic/real) is entirely config-driven via config/checkpoint_contract.yaml
(CheckpointContract.load() asserts mode is one of {"synthetic","real"} and that
the two modes' process_value differ, so a run can never silently relabel one
mode's rows as the other's). Output directory, saved label, and the printed/
logged report are all mode-aware; synthetic and real artifacts never share a
directory (see _out_dirs()). Existing outputs and Stage 2 are not touched --
everything here writes under outputs/checkpoint_training/ (synthetic) or
outputs/checkpoint_training_<mode>/ (any other mode).

Provenance (src/models/checkpoint_provenance.py): every run writes
<out>/provenance/ -- full tee'd log, timestamped event stream, exact
lot/wafer partition and inner/outer fold assignments actually used, data/
config/code hashes, package versions, and a winner-lock record (pipeline
files hashed and flushed to disk BEFORE the test partition is first read).
The run refuses to start if the output directory already holds pipelines,
so earlier artifacts are never silently overwritten.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.checkpoint_contract import CheckpointContract
from src.data.checkpoint_validation import load_raw, filter_process, get_checkpoint_predictor_columns, check_schema
from src.data.checkpoint_splits import build_shared_split, verify_no_lot_crosses_partitions
from src.models.checkpoint_pipeline_builder import (
    build_elasticnet_pipeline_and_grid, build_tree_pipeline_and_grid,
    make_grouped_search, retained_feature_count,
)
from src.models.checkpoint_provenance import (
    PROVENANCE_SCHEMA_VERSION, ProvenanceRecorder, TeeLog, build_inner_fold_table,
    build_partition_table, code_hashes, environment_info, file_record, sha256_file,
    splits_equal, utc_now,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _out_dirs(mode: str) -> tuple[Path, Path, Path]:
    """Synthetic and real runs must never share an output directory, so a
    real-data run can never silently overwrite the synthetic artifacts
    already delivered. Synthetic keeps its original path unchanged."""
    base = PROJECT_ROOT / "outputs" / ("checkpoint_training" if mode == "synthetic" else f"checkpoint_training_{mode}")
    return base, base / "models", base / "predictions"

MODEL_NAMES = ("elasticnet", "random_forest", "lightgbm")
FEATURE_MODES = ("full", "selected")
MODEL_SIMPLICITY_RANK = {"elasticnet": 0, "random_forest": 1, "lightgbm": 2}

N_INNER_SPLITS = 3


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_pipeline_and_grid(model_name: str, feature_mode: str):
    if model_name == "elasticnet":
        return build_elasticnet_pipeline_and_grid(feature_mode)
    return build_tree_pipeline_and_grid(model_name, feature_mode)


@dataclasses.dataclass
class FoldRecord:
    checkpoint: str
    model: str
    mode: str
    fold: int
    rmse: float
    mae: float
    r2: float
    n_features_retained: int
    best_params: dict
    fit_seconds: float
    val_lotname: list
    val_wafernum: list
    val_actual: list
    val_pred: list


def run_outer_cv_for_config(checkpoint_name, model_name, feature_mode,
                              train_df, cols, target, group_col, id_col, folds) -> tuple[list[FoldRecord], list]:
    """Returns (fold records, inner splits actually used per outer fold --
    read back from each fitted GridSearchCV's own cv attribute)."""
    records = []
    inner_used = []
    X_all = train_df[cols]
    y_all = train_df[target]
    groups_all = train_df[group_col]
    lot_all = train_df[group_col]
    wafer_all = train_df[id_col]

    for fold_i, (tr, va) in enumerate(folds):
        X_tr, X_va = X_all.iloc[tr], X_all.iloc[va]
        y_tr, y_va = y_all.iloc[tr], y_all.iloc[va]
        groups_tr = groups_all.iloc[tr]

        pipe, grid = _build_pipeline_and_grid(model_name, feature_mode)
        search = make_grouped_search(pipe, grid, groups_tr, n_inner_splits=N_INNER_SPLITS)

        t0 = time.time()
        search.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        pred = search.predict(X_va)
        inner_used.append(list(search.cv))
        m = _metrics(y_va, pred)
        n_feat = retained_feature_count(search.best_estimator_, feature_mode, len(cols))

        records.append(FoldRecord(
            checkpoint=checkpoint_name, model=model_name, mode=feature_mode, fold=fold_i,
            rmse=m["rmse"], mae=m["mae"], r2=m["r2"], n_features_retained=n_feat,
            best_params={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in search.best_params_.items()},
            fit_seconds=elapsed,
            val_lotname=lot_all.iloc[va].tolist(), val_wafernum=wafer_all.iloc[va].tolist(),
            val_actual=y_va.tolist(), val_pred=pred.tolist(),
        ))
        print(f"  [{checkpoint_name}/{model_name}/{feature_mode}] fold {fold_i}: "
              f"rmse={m['rmse']:.4f} n_feat={n_feat} ({elapsed:.1f}s)", flush=True)
    return records, inner_used


def aggregate(records: list[FoldRecord]) -> dict:
    rmses = [r.rmse for r in records]
    return {
        "mean_rmse": float(np.mean(rmses)), "std_rmse": float(np.std(rmses, ddof=1)),
        "mean_mae": float(np.mean([r.mae for r in records])),
        "mean_r2": float(np.mean([r.r2 for r in records])),
        "mean_n_features": float(np.mean([r.n_features_retained for r in records])),
        "mean_fit_seconds": float(np.mean([r.fit_seconds for r in records])),
        "n_folds": len(records),
    }


def select_winner(candidates: dict) -> tuple:
    """candidates: {(model,mode): agg_dict}. Returns the winning (model,mode) key.
    1-SE rule: SE_best = std_rmse_of_best / sqrt(n_folds). Acceptable set =
    candidates within mean_rmse_best + SE_best. Tie-break: fewer actual mean
    retained features -> simpler model family (elasticnet<rf<lgbm) -> lower
    std_rmse -> lower mean_fit_seconds -> best mean_rmse (deterministic)."""
    best_key = min(candidates, key=lambda k: candidates[k]["mean_rmse"])
    best = candidates[best_key]
    se_best = best["std_rmse"] / np.sqrt(best["n_folds"])
    threshold = best["mean_rmse"] + se_best

    acceptable = {k: v for k, v in candidates.items() if v["mean_rmse"] <= threshold}

    def sort_key(k):
        v = candidates[k]
        model_name, _mode = k
        return (
            v["mean_n_features"],
            MODEL_SIMPLICITY_RANK[model_name],
            v["std_rmse"],
            v["mean_fit_seconds"],
            v["mean_rmse"],
        )

    winner = min(acceptable, key=sort_key)
    return winner, se_best, threshold


def lock_final_model(checkpoint_name, model_name, feature_mode, train_df, cols, target, group_col):
    pipe, grid = _build_pipeline_and_grid(model_name, feature_mode)
    groups_full = train_df[group_col]
    search = make_grouped_search(pipe, grid, groups_full, n_inner_splits=N_INNER_SPLITS)
    t0 = time.time()
    search.fit(train_df[cols], train_df[target])
    elapsed = time.time() - t0
    n_feat = retained_feature_count(search.best_estimator_, feature_mode, len(cols))
    return search.best_estimator_, search.best_params_, n_feat, elapsed, list(search.cv)


def main():
    contract = CheckpointContract.load()  # already asserts mode in {"synthetic","real"} and that the two
                                           # modes' process_value differ -- see CheckpointContract.load()
    OUT_DIR, MODELS_DIR, PRED_DIR = _out_dirs(contract.mode)
    existing = sorted(MODELS_DIR.glob("*_final_pipeline.joblib")) if MODELS_DIR.exists() else []
    if existing or (OUT_DIR / "provenance").exists():
        sys.exit(
            f"Refusing to run: {OUT_DIR} already holds artifacts ({len(existing)} pipelines"
            f"{', provenance/' if (OUT_DIR / 'provenance').exists() else ''}). Move that directory "
            f"aside (e.g. rename it with a date suffix) before retraining -- existing run artifacts "
            f"are never overwritten in place."
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    run_id = utc_now().replace(":", "").replace("-", "").replace(".", "_")
    prov = ProvenanceRecorder(OUT_DIR / "provenance", run_id)
    with TeeLog(prov.dir / "run_log.txt"):
        _run(contract, OUT_DIR, MODELS_DIR, PRED_DIR, prov)


def _run(contract, OUT_DIR, MODELS_DIR, PRED_DIR, prov):
    overall_t0 = time.time()
    started_utc = prov.event("run_started", argv=sys.argv, mode=contract.mode)
    print(f"run_id={prov.run_id} started_utc={started_utc}", flush=True)
    # inputs/code/env are hashed BEFORE any data is read, so the manifest
    # describes what this run actually consumed
    inputs = {
        "data_csv": file_record(contract.input_csv_path),
        "checkpoint_contract_yaml": file_record(PROJECT_ROOT / "config" / "checkpoint_contract.yaml"),
    }
    code = code_hashes(PROJECT_ROOT)
    env = environment_info()

    df = load_raw(contract)
    schema = check_schema(df, contract)
    assert schema.required_columns_present, (
        f"mode={contract.mode!r}: input file {contract.input_csv_path} is missing required columns "
        f"{schema.missing_required_columns} -- refusing to proceed. Real data must use the same column "
        f"names as the synthetic dataset; if it doesn't, fix the column names or the contract, not this check."
    )
    filtered, process_report = filter_process(df, contract)
    assert process_report.rows_after > 0, (
        f"mode={contract.mode!r}: filtering {contract.process_column}=={contract.process_value!r} "
        f"left 0 rows (saw: {process_report.other_process_values_seen}) -- refusing to train on an empty set."
    )
    print(f"mode={contract.mode} | input={contract.input_csv_path} | "
          f"{contract.process_column}=={contract.process_value!r}: "
          f"{process_report.rows_before} -> {process_report.rows_after} rows, "
          f"{process_report.lots_before} -> {process_report.lots_after} lots", flush=True)

    shared = build_shared_split(filtered, contract.group_column, contract.chronological_candidates,
                                 contract.holdout_test_size, contract.random_state, contract.n_cv_splits)
    assert verify_no_lot_crosses_partitions(filtered, contract.group_column, shared.holdout), \
        "Lot crossed train/test partitions -- aborting."

    train_df = filtered.iloc[shared.holdout.train_idx].reset_index(drop=True)
    test_df = filtered.iloc[shared.holdout.test_idx].reset_index(drop=True)
    print(f"{contract.mode.upper()} DATA RUN. Train: {len(train_df)} rows / {train_df[contract.group_column].nunique()} lots. "
          f"Test: {len(test_df)} rows / {test_df[contract.group_column].nunique()} lots (untouched until final eval).",
          flush=True)

    partition = build_partition_table(filtered, shared.holdout.train_idx, shared.holdout.test_idx, shared.folds,
                                      contract.group_column, contract.secondary_id_column)
    partition_sha = prov.write_frame("partition_assignments.csv", partition)
    prov.event("split_built", partition_assignments_sha256=partition_sha,
               n_train_rows=int((partition.partition == "train").sum()),
               n_test_rows=int((partition.partition == "test").sum()))

    all_fold_records: list[FoldRecord] = []
    checkpoint_results = {}  # name -> {(model,mode): agg}
    checkpoint_records = {}  # name -> {(model,mode): [FoldRecord,...]}
    inner_by_outer = None    # inner splits actually used; must be identical for every config

    for cp in contract.checkpoints:
        cols = get_checkpoint_predictor_columns(filtered, contract, cp.name)
        print(f"\n{'='*70}\nCHECKPOINT {cp.order}: {cp.name} ({len(cols)} candidate features)\n{'='*70}", flush=True)
        cand_agg, cand_records = {}, {}
        for model_name in MODEL_NAMES:
            for mode in FEATURE_MODES:
                recs, inner_used = run_outer_cv_for_config(
                    cp.name, model_name, mode, train_df, cols, contract.target,
                    contract.group_column, contract.secondary_id_column, shared.folds,
                )
                if inner_by_outer is None:
                    inner_by_outer = inner_used
                assert all(splits_equal(a, b) for a, b in zip(inner_by_outer, inner_used)), \
                    f"{cp.name}/{model_name}/{mode}: inner CV splits differ from other configs"
                all_fold_records.extend(recs)
                cand_records[(model_name, mode)] = recs
                cand_agg[(model_name, mode)] = aggregate(recs)
        checkpoint_results[cp.name] = cand_agg
        checkpoint_records[cp.name] = cand_records
    prov.event("outer_cv_done", n_fold_records=len(all_fold_records))

    # ---- selection (CV only) ----
    winners = {}
    print(f"\n{'='*70}\nWINNER SELECTION (CV only, 1-SE rule)\n{'='*70}", flush=True)
    for cp in contract.checkpoints:
        agg = checkpoint_results[cp.name]
        winner_key, se_best, threshold = select_winner(agg)
        winners[cp.name] = winner_key
        print(f"{cp.name}: winner={winner_key}  mean_rmse={agg[winner_key]['mean_rmse']:.4f}  "
              f"(best_rmse+SE threshold={threshold:.4f}, SE={se_best:.4f})", flush=True)
        for k, v in sorted(agg.items(), key=lambda kv: kv[1]["mean_rmse"]):
            print(f"    {k}: mean_rmse={v['mean_rmse']:.4f} std={v['std_rmse']:.4f} "
                  f"n_feat={v['mean_n_features']:.1f} fit_s={v['mean_fit_seconds']:.2f}", flush=True)
    prov.event("winners_selected", winners={k: list(v) for k, v in winners.items()})

    # ---- lock winners (refit on full train partition, CV only, no test access) ----
    print(f"\n{'='*70}\nLOCKING WINNERS (refit on full 80-lot train partition)\n{'='*70}", flush=True)
    locked = {}
    lock_inner = None
    for cp in contract.checkpoints:
        model_name, mode = winners[cp.name]
        cols = get_checkpoint_predictor_columns(filtered, contract, cp.name)
        pipeline, best_params, n_feat, elapsed, lock_cv = lock_final_model(
            cp.name, model_name, mode, train_df, cols, contract.target, contract.group_column
        )
        lock_inner = lock_inner or lock_cv
        assert splits_equal(lock_inner, lock_cv), f"{cp.name}: lock-refit inner splits differ across checkpoints"
        locked[cp.name] = {
            "pipeline": pipeline, "model": model_name, "mode": mode,
            "best_params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in best_params.items()},
            "n_features_retained": n_feat, "lock_fit_seconds": elapsed, "cols": cols,
        }
        print(f"{cp.name}: locked {model_name}/{mode}, n_feat={n_feat}, params={best_params} ({elapsed:.1f}s)", flush=True)

    # pipelines are written and hashed now, and the lock record flushed to
    # disk, so the saved models provably predate the first test-set read
    lock_record = {"run_id": prov.run_id, "locked_utc": utc_now(), "checkpoints": {}}
    for cp in contract.checkpoints:
        info = locked[cp.name]
        p = MODELS_DIR / f"{cp.name}_final_pipeline.joblib"
        joblib.dump(info["pipeline"], p)
        lock_record["checkpoints"][cp.name] = {
            "winner": [info["model"], info["mode"]], "best_params": info["best_params"],
            "n_features_retained": info["n_features_retained"], "candidate_feature_names": info["cols"],
            "cv_summary_all_candidates": {f"{k[0]}|{k[1]}": v for k, v in checkpoint_results[cp.name].items()},
            "pipeline_sha256": sha256_file(p),
        }
    lock_sha = prov.write_json("winner_lock.json", lock_record)
    prov.event("winners_locked", winner_lock_sha256=lock_sha)

    level_splits = {f"outer_fold_{k}": (tr, inner_by_outer[k]) for k, (tr, _va) in enumerate(shared.folds)}
    level_splits["final_lock_refit"] = (np.arange(len(train_df)), lock_inner)
    inner_table = build_inner_fold_table(train_df, level_splits, contract.group_column, contract.secondary_id_column)
    inner_sha = prov.write_frame("inner_fold_assignments.csv", inner_table)

    # ---- evaluate locked winners on the UNTOUCHED test set ----
    print(f"\n{'='*70}\nTEST-SET EVALUATION (untouched 20-lot partition, locked models only)\n{'='*70}", flush=True)
    prov.event("test_set_first_access")
    test_results = {}
    test_predictions = {}
    for cp in contract.checkpoints:
        info = locked[cp.name]
        X_test = test_df[info["cols"]]
        y_test = test_df[contract.target]
        pred_test = info["pipeline"].predict(X_test)
        m = _metrics(y_test, pred_test)
        test_results[cp.name] = m
        test_predictions[cp.name] = pd.DataFrame({
            "LotName": test_df[contract.group_column].values,
            "WaferNum": test_df[contract.secondary_id_column].values,
            "Actual Sort Yield": y_test.values,
            "Predicted Yield": pred_test,
            "source": "holdout",
        })
        print(f"{cp.name}: TEST rmse={m['rmse']:.4f} mae={m['mae']:.4f} r2={m['r2']:.4f}", flush=True)
    prov.event("test_eval_done", test_metrics=test_results)

    # ---- save everything ----
    print(f"\n{'='*70}\nSAVING ARTIFACTS\n{'='*70}", flush=True)
    for cp in contract.checkpoints:
        info = locked[cp.name]
        model_name, mode = winners[cp.name]

        # pipeline file was written at lock time; confirm it was not altered since
        assert sha256_file(MODELS_DIR / f"{cp.name}_final_pipeline.joblib") == \
            lock_record["checkpoints"][cp.name]["pipeline_sha256"], f"{cp.name}: pipeline changed after lock"

        metadata = {
            "checkpoint": cp.name, "label": f"{contract.mode.upper()} DATA RESULT",
            "winning_model": model_name, "feature_mode": mode,
            "candidate_prefixes": cp.prefixes, "n_candidate_features": len(info["cols"]),
            "candidate_feature_names": info["cols"],
            "n_features_retained_final": info["n_features_retained"],
            "best_hyperparameters": info["best_params"],
            "lock_refit_seconds": info["lock_fit_seconds"],
            "cv_summary_all_candidates": {f"{k[0]}|{k[1]}": v for k, v in checkpoint_results[cp.name].items()},
            "test_metrics": test_results[cp.name],
            "split_provenance": {
                "random_state": contract.random_state, "holdout_test_size": contract.holdout_test_size,
                "n_cv_splits": contract.n_cv_splits, "n_inner_splits": N_INNER_SPLITS,
                "train_rows": len(train_df), "train_lots": int(train_df[contract.group_column].nunique()),
                "test_rows": len(test_df), "test_lots": int(test_df[contract.group_column].nunique()),
                "chronological_column_assessed": shared.chronological.candidate_column,
                "mode": contract.mode, "input_csv_path": str(contract.input_csv_path),
                "process_column": contract.process_column, "process_value": contract.process_value,
                "run_id": prov.run_id,
                "provenance_dir": "provenance/",
            },
        }
        with open(MODELS_DIR / f"{cp.name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # grouped-OOF predictions from the WINNER's own fold records (train rows)
        winner_recs = checkpoint_records[cp.name][winners[cp.name]]
        oof_rows = []
        for r in winner_recs:
            for lot, wafer, actual, pred in zip(r.val_lotname, r.val_wafernum, r.val_actual, r.val_pred):
                oof_rows.append({"LotName": lot, "WaferNum": wafer, "Actual Sort Yield": actual,
                                  "Predicted Yield": pred, "source": "grouped_oof_cv"})
        oof_df = pd.DataFrame(oof_rows)
        full_pred_df = pd.concat([oof_df, test_predictions[cp.name]], ignore_index=True)
        full_pred_df.to_csv(PRED_DIR / f"{cp.name}_predictions.csv", index=False)
        print(f"{cp.name}: saved pipeline, metadata, predictions ({len(full_pred_df)} rows: "
              f"{len(oof_df)} OOF train + {len(test_predictions[cp.name])} test)", flush=True)

    # ---- full fold-level audit CSV (all 24 configs x 5 folds) ----
    audit_rows = [{
        "checkpoint": r.checkpoint, "model": r.model, "mode": r.mode, "fold": r.fold,
        "rmse": r.rmse, "mae": r.mae, "r2": r.r2, "n_features_retained": r.n_features_retained,
        "fit_seconds": r.fit_seconds, "best_params": json.dumps(r.best_params),
    } for r in all_fold_records]
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "cv_fold_results_all_24_configs.csv", index=False)

    # ---- comparison report ----
    total_elapsed = time.time() - overall_t0
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    log("FOUR-CHECKPOINT MODEL TRAINING -- COMPARISON REPORT")
    log("ALL RESULTS ARE {}-DATA RESULTS (Process == '{}').".format(contract.mode.upper(), contract.process_value))
    log("="*100)
    log(f"Total run time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log(f"Shared split: {len(train_df)} train rows / {train_df[contract.group_column].nunique()} lots, "
        f"{len(test_df)} test rows / {test_df[contract.group_column].nunique()} lots. Test set untouched until final eval.")
    log("")
    log(f"{'Checkpoint':10} {'Model':14} {'Mode':9} {'CV RMSE':>9} {'CV std':>8} {'CV MAE':>8} {'CV R2':>7} "
        f"{'n_feat':>7} {'Test RMSE':>10} {'Test MAE':>9} {'Test R2':>8} {'Winner':>7}")
    log("-"*130)
    for cp in contract.checkpoints:
        for (model_name, mode), v in sorted(checkpoint_results[cp.name].items(), key=lambda kv: kv[1]["mean_rmse"]):
            is_winner = (model_name, mode) == winners[cp.name]
            test_m = test_results[cp.name] if is_winner else {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            log(f"{cp.name:10} {model_name:14} {mode:9} {v['mean_rmse']:9.4f} {v['std_rmse']:8.4f} "
                f"{v['mean_mae']:8.4f} {v['mean_r2']:7.4f} {v['mean_n_features']:7.1f} "
                f"{test_m['rmse']:10.4f} {test_m['mae']:9.4f} {test_m['r2']:8.4f} "
                f"{'<-- WINNER' if is_winner else '':>10}")
        log("")

    with open(OUT_DIR / "comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    prov.event("artifacts_saved")

    output_rels = (["comparison_report.txt", "cv_fold_results_all_24_configs.csv"]
                   + [f"models/{cp.name}_final_pipeline.joblib" for cp in contract.checkpoints]
                   + [f"models/{cp.name}_metadata.json" for cp in contract.checkpoints]
                   + [f"predictions/{cp.name}_predictions.csv" for cp in contract.checkpoints])
    manifest = {
        "schema_version": PROVENANCE_SCHEMA_VERSION, "retrospective": False,
        "run_id": prov.run_id, "started_utc": started_utc, "finished_utc": utc_now(),
        "mode": contract.mode, "group_column": contract.group_column,
        "secondary_id_column": contract.secondary_id_column, "target": contract.target,
        "split_parameters": {"random_state": contract.random_state, "holdout_test_size": contract.holdout_test_size,
                             "n_cv_splits": contract.n_cv_splits, "n_inner_splits": N_INNER_SPLITS},
        "inputs": inputs, "code_files": code, "environment": env,
        "code_files_unchanged_during_run": code == code_hashes(PROJECT_ROOT),
        "partition_assignments_sha256": partition_sha, "inner_fold_assignments_sha256": inner_sha,
        "winner_lock_sha256": lock_sha,
        "output_files": {rel: file_record(OUT_DIR / rel) for rel in output_rels},
    }
    prov.write_json("run_manifest.json", manifest)
    prov.event("run_completed", total_seconds=total_elapsed)

    print(f"\nDone. Total time: {total_elapsed:.1f}s. Artifacts under {OUT_DIR}")
    print(f"Provenance: {prov.dir} (validate with: python -m src.models.checkpoint_provenance {OUT_DIR})")


if __name__ == "__main__":
    main()
