"""Provenance tests for the four-checkpoint pipeline. NO real model is
trained: behavioral checks use sklearn's DummyRegressor (and a recording
transformer) inside the project's real split/search/orchestration code, and
every write goes to a temporary directory -- except nothing at all is
written under outputs/.

Run: .venv\\Scripts\\python.exe tests\\test_checkpoint_provenance.py
(pytest also works if installed.)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data.checkpoint_contract import CheckpointContract
from src.data.checkpoint_validation import load_raw, filter_process, get_checkpoint_predictor_columns
from src.data.checkpoint_splits import build_shared_split
from src.models.checkpoint_pipeline_builder import make_grouped_search
from src.models import checkpoint_provenance as cpv

OUT_DIR = PROJECT_ROOT / "outputs" / "checkpoint_training"
RETRO_DIR = OUT_DIR / "provenance_retrospective"
_CACHE = {}


def _real_split():
    if "split" not in _CACHE:
        c = CheckpointContract.load()
        f, _ = filter_process(load_raw(c), c)
        sh = build_shared_split(f, c.group_column, c.chronological_candidates,
                                c.holdout_test_size, c.random_state, c.n_cv_splits)
        _CACHE["split"] = (c, f, sh, f.iloc[sh.holdout.train_idx].reset_index(drop=True))
    return _CACHE["split"]


class _Recorder(BaseEstimator, TransformerMixin):
    log = []

    def __init__(self, tag=0):
        self.tag = tag

    def fit(self, X, y=None):
        _Recorder.log.append(frozenset(X.index))
        return self

    def transform(self, X):
        return X


# ----------------------------------------------------------------------
# partition / fold tables
# ----------------------------------------------------------------------

def test_partition_table_covers_every_row_once_and_rejects_lot_crossing():
    df = pd.DataFrame({"LotName": list("AAABBBCCCDDD"), "WaferNum": ["W1", "W2", "W3"] * 4})
    train_idx, test_idx = np.arange(9), np.arange(9, 12)
    folds = [(np.arange(3, 9), np.arange(0, 3)), (np.r_[0:3, 6:9], np.arange(3, 6)), (np.arange(0, 6), np.arange(6, 9))]
    t = cpv.build_partition_table(df, train_idx, test_idx, folds, "LotName", "WaferNum")
    assert len(t) == 12 and (t.partition == "test").sum() == 3
    assert (t.loc[t.partition == "test", "outer_fold"] == -1).all()
    assert sorted(t.loc[t.partition == "train", "outer_fold"].unique()) == [0, 1, 2]

    bad_test = np.array([8, 9, 10, 11])          # lot C split across train/test
    try:
        cpv.build_partition_table(df, np.arange(8), bad_test, [(np.arange(3, 8), np.arange(0, 3))], "LotName", "WaferNum")
        raise AssertionError("lot crossing train/test was not rejected")
    except AssertionError as e:
        assert "not rejected" not in str(e)


def test_real_partition_and_inner_folds_are_lot_disjoint():
    c, f, sh, train_df = _real_split()
    part = cpv.build_partition_table(f, sh.holdout.train_idx, sh.holdout.test_idx, sh.folds,
                                     c.group_column, c.secondary_id_column)
    levels = {f"outer_fold_{k}": (tr, list(make_grouped_search(None, {}, train_df[c.group_column].iloc[tr], 3).cv))
              for k, (tr, _va) in enumerate(sh.folds)}
    inner = cpv.build_inner_fold_table(train_df, levels, c.group_column, c.secondary_id_column)
    for k, (tr, va) in enumerate(sh.folds):
        sub = inner[inner.level == f"outer_fold_{k}"]
        assert sorted(sub.train_df_row) == sorted(tr), f"outer fold {k}: inner val rows must tile its training rows"
        assert not set(sub.train_df_row) & set(va), f"outer fold {k}: outer-val row inside inner CV"
        assert (sub.groupby(c.group_column).inner_fold.nunique() == 1).all()
    assert part.partition.value_counts().to_dict() == {"train": len(sh.holdout.train_idx), "test": len(sh.holdout.test_idx)}


def test_learned_steps_fit_only_on_training_rows_instrumented():
    """Behavioral (not source-text): the project's real make_grouped_search
    on the real outer folds; a recording transformer logs every row set any
    fit() sees. DummyRegressor -- no real model is trained."""
    c, f, sh, train_df = _real_split()
    cols = get_checkpoint_predictor_columns(f, c, "OHMIC")
    for k, (tr, va) in enumerate(sh.folds):
        _Recorder.log = []
        X_tr = train_df[cols].iloc[tr]
        s = make_grouped_search(Pipeline([("rec", _Recorder()), ("model", DummyRegressor())]),
                                {"rec__tag": [0, 1]}, train_df[c.group_column].iloc[tr], 3)
        s.n_jobs = 1  # keep the recorder in-process; split construction unchanged
        s.fit(X_tr, train_df[c.target].iloc[tr])
        allowed = [frozenset(X_tr.index[a]) for a, _ in s.cv] + [frozenset(X_tr.index)]
        assert len(_Recorder.log) == 2 * len(s.cv) + 1
        assert all(rows in allowed for rows in _Recorder.log), f"outer fold {k}: a fit saw rows outside an inner-train set"
        assert not any(rows & set(va) for rows in _Recorder.log), f"outer fold {k}: a fit saw outer-validation rows"


# ----------------------------------------------------------------------
# recorder / tee / validator
# ----------------------------------------------------------------------

def test_tee_log_captures_output_and_tracebacks():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "run_log.txt"
        try:
            with cpv.TeeLog(log):
                print("hello-provenance")
                raise RuntimeError("boom-provenance")
        except RuntimeError:
            pass
        text = log.read_text(encoding="utf-8")
        assert "hello-provenance" in text and "boom-provenance" in text


def _fake_run_dir(d: Path) -> Path:
    """Minimal synthetic provenance directory written with the real
    recorder helpers -- no model involved (pipelines are plain dicts)."""
    out = Path(d)
    (out / "models").mkdir()
    (out / "predictions").mkdir()
    data = out / "data.csv"
    data.write_text("x\n1\n", encoding="utf-8")
    rec = cpv.ProvenanceRecorder(out / "provenance", "RID")
    df = pd.DataFrame({"LotName": ["A", "A", "B", "C"], "WaferNum": ["W1", "W2", "W1", "W1"]})
    part = cpv.build_partition_table(df, np.array([0, 1, 2]), np.array([3]),
                                     [(np.array([2]), np.array([0, 1])), (np.array([0, 1]), np.array([2]))],
                                     "LotName", "WaferNum")
    psha = rec.write_frame("partition_assignments.csv", part)
    rec.write_frame("inner_fold_assignments.csv", pd.DataFrame({"level": ["x"]}))
    for e in cpv.EVENT_ORDER[:5]:
        rec.event(e)
    joblib.dump({"stub": 1}, out / "models" / "OHMIC_final_pipeline.joblib")
    lock = {"checkpoints": {"OHMIC": {"pipeline_sha256": cpv.sha256_file(out / "models" / "OHMIC_final_pipeline.joblib")}}}
    rec.write_json("winner_lock.json", lock)
    for e in cpv.EVENT_ORDER[5:]:
        rec.event(e)
    pd.DataFrame({"LotName": ["A", "A", "B", "C"], "WaferNum": ["W1", "W2", "W1", "W1"],
                  "Actual Sort Yield": [1, 2, 3, 4], "Predicted Yield": [1, 2, 3, 4],
                  "source": ["grouped_oof_cv"] * 3 + ["holdout"]}).to_csv(out / "predictions" / "OHMIC_predictions.csv", index=False)
    (out / "provenance" / "run_log.txt").write_text("...\nDone. Total time: 1.0s\n", encoding="utf-8")
    rec.write_json("run_manifest.json", {
        "schema_version": cpv.PROVENANCE_SCHEMA_VERSION, "retrospective": False, "run_id": "RID",
        "group_column": "LotName", "secondary_id_column": "WaferNum",
        "inputs": {"data_csv": cpv.file_record(data)}, "partition_assignments_sha256": psha,
        "output_files": {"predictions/OHMIC_predictions.csv": cpv.file_record(out / "predictions" / "OHMIC_predictions.csv")},
    })
    return out


def test_validator_accepts_consistent_run_and_rejects_tampering():
    with tempfile.TemporaryDirectory() as d:
        out = _fake_run_dir(Path(d))
        assert cpv.validate_run_provenance(out) == [], cpv.validate_run_provenance(out)

        # test access logged before the lock -> rejected
        ev = (out / "provenance" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        i_lock = next(i for i, l in enumerate(ev) if '"winners_locked"' in l)
        i_test = next(i for i, l in enumerate(ev) if '"test_set_first_access"' in l)
        ev[i_lock], ev[i_test] = ev[i_test], ev[i_lock]
        (out / "provenance" / "events.jsonl").write_text("\n".join(ev) + "\n", encoding="utf-8")
        assert any("order" in p for p in cpv.validate_run_provenance(out))

    with tempfile.TemporaryDirectory() as d:
        out = _fake_run_dir(Path(d))
        joblib.dump({"stub": 2}, out / "models" / "OHMIC_final_pipeline.joblib")   # pipeline swapped after lock
        assert any("lock time" in p for p in cpv.validate_run_provenance(out))

    with tempfile.TemporaryDirectory() as d:
        out = _fake_run_dir(Path(d))
        p = out / "predictions" / "OHMIC_predictions.csv"
        pr = pd.read_csv(p)
        pr.loc[3, "source"] = "grouped_oof_cv"                                     # test row relabeled as OOF
        pr.to_csv(p, index=False)
        probs = cpv.validate_run_provenance(out)
        assert any("changed since run" in x for x in probs) and any("partition" in x for x in probs)


def test_training_refuses_to_overwrite_existing_artifacts():
    """Calls the real entry point; it must exit before loading data when
    the output dir already holds pipelines. Skipped (not run) if that
    precondition does not hold, so this can never start a training run."""
    c = CheckpointContract.load()
    from src.models import checkpoint_train_run as ctr
    out, models, _ = ctr._out_dirs(c.mode)
    if not list(models.glob("*_final_pipeline.joblib")):
        print("  SKIP: no existing pipelines for this mode; refusing to call main()")
        return
    before = sorted((p.name, p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file())
    try:
        ctr.main()
        raise AssertionError("main() did not refuse")
    except SystemExit as e:
        assert "Refusing to run" in str(e)
    after = sorted((p.name, p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file())
    assert before == after and not (out / "provenance").exists()


def test_dry_run_with_dummy_estimator_writes_valid_provenance():
    """Executes the REAL _run() orchestration (split, outer CV, inner
    GridSearchCV, selection, lock, test eval, saving, provenance) with every
    candidate replaced by impute -> dropconst -> DummyRegressor, into a temp
    directory. No real model is trained; outputs/ is not touched."""
    from src.models import checkpoint_train_run as ctr
    c = CheckpointContract.load()

    def dummy_builder(model_name, feature_mode):
        return Pipeline([("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
                         ("dropconst", VarianceThreshold(1e-8).set_output(transform="pandas")),
                         ("model", DummyRegressor())]), {"model__strategy": ["mean", "median"]}

    orig = ctr._build_pipeline_and_grid
    ctr._build_pipeline_and_grid = dummy_builder
    try:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            models, preds = out / "models", out / "predictions"
            models.mkdir(); preds.mkdir()
            prov = cpv.ProvenanceRecorder(out / "provenance", "DRYRUN")
            with cpv.TeeLog(prov.dir / "run_log.txt"):
                ctr._run(c, out, models, preds, prov)
            problems = cpv.validate_run_provenance(out)
            assert problems == [], problems
            ev = [e["event"] for e in cpv.read_events(prov.dir)]
            assert ev == list(cpv.EVENT_ORDER), ev
            lock = json.loads((prov.dir / "winner_lock.json").read_text(encoding="utf-8"))
            assert set(lock["checkpoints"]) == {cp.name for cp in c.checkpoints}
            manifest = json.loads((prov.dir / "run_manifest.json").read_text(encoding="utf-8"))
            assert manifest["retrospective"] is False and manifest["code_files_unchanged_during_run"]
            assert manifest["environment"]["packages"]["scikit-learn"]
            inner = pd.read_csv(prov.dir / "inner_fold_assignments.csv")
            assert set(inner.level) == {f"outer_fold_{k}" for k in range(c.n_cv_splits)} | {"final_lock_refit"}
            meta = json.loads((models / "OHMIC_metadata.json").read_text(encoding="utf-8"))
            assert meta["split_provenance"]["run_id"] == "DRYRUN"
    finally:
        ctr._build_pipeline_and_grid = orig


# ----------------------------------------------------------------------
# existing (original) run + its retrospective provenance
# ----------------------------------------------------------------------

def test_existing_run_has_no_original_provenance_and_retrospective_is_labeled():
    assert not (OUT_DIR / "provenance").exists(), "the 2026-10-05 run never wrote original provenance"
    assert any("missing provenance" in p for p in cpv.validate_run_provenance(OUT_DIR))
    m = json.loads((RETRO_DIR / "retrospective_manifest.json").read_text(encoding="utf-8"))
    assert m["retrospective"] is True and "RETROSPECTIVE" in m["provenance_label"]
    for name in ("partition_assignments_RETROSPECTIVE.csv", "inner_fold_assignments_RETROSPECTIVE.csv"):
        assert set(pd.read_csv(RETRO_DIR / name).provenance_label) == {"RETROSPECTIVE"}
    assert "RETROSPECTIVE" in json.loads((RETRO_DIR / "reconstruction_verification_RETROSPECTIVE.json")
                                         .read_text(encoding="utf-8"))["provenance_label"]


def test_existing_artifacts_unchanged_since_retrospective_hashing():
    m = json.loads((RETRO_DIR / "retrospective_manifest.json").read_text(encoding="utf-8"))
    recorded = m["artifact_hashes_at_reconstruction_time"]
    assert recorded, "no artifact hashes recorded"
    for rel, rec in recorded.items():
        assert cpv.sha256_file(OUT_DIR / rel) == rec["sha256"], f"{rel} changed after retrospective hashing"
    data = m["inputs_at_reconstruction_time"]["data_csv"]
    assert cpv.sha256_file(Path(data["path"])) == data["sha256"], "input data changed"


def test_retrospective_reconstruction_reproduces_saved_artifacts():
    v = json.loads((RETRO_DIR / "reconstruction_verification_RETROSPECTIVE.json").read_text(encoding="utf-8"))
    for cp, ch in v["per_checkpoint"].items():
        assert ch["oof_rows_equal_reconstructed_train_rows"] and ch["holdout_rows_equal_reconstructed_test_rows"], cp
        assert ch["actual_values_equal_source_target"] and ch["pipeline_feature_names_equal_metadata"], cp
        assert ch["imputer_medians_equal_train_partition_medians"], cp
        assert ch["max_abs_diff_fold_rmse_recomputed_vs_saved_csv"] < 1e-9, cp
        assert ch["max_abs_diff_reloaded_pipeline_vs_saved_holdout_pred"] < 1e-9, cp
        for k in ("rmse", "mae", "r2"):
            assert abs(ch["test_metrics_recomputed"][k] - ch["test_metrics_saved"][k]) < 1e-12, (cp, k)
    # the reconstructed partition still matches what the code produces today
    c, f, sh, _ = _real_split()
    part = pd.read_csv(RETRO_DIR / "partition_assignments_RETROSPECTIVE.csv")
    fresh = cpv.build_partition_table(f, sh.holdout.train_idx, sh.holdout.test_idx, sh.folds,
                                      c.group_column, c.secondary_id_column)
    assert part.drop(columns="provenance_label").equals(fresh)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} provenance tests passed.")
