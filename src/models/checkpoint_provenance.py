"""Run-provenance recording and validation for the four-checkpoint pipeline.

Used by src/models/checkpoint_train_run.py to write, for every future run,
a self-contained provenance/ directory next to the run's artifacts:

  run_log.txt                    full stdout+stderr of the run (tee'd live)
  events.jsonl                   append-only, timestamped event stream
                                 (run_started -> split_built -> outer_cv_done
                                 -> winners_selected -> winners_locked ->
                                 test_set_first_access -> test_eval_done ->
                                 artifacts_saved -> run_completed)
  partition_assignments.csv      every row: LotName, WaferNum, train/test,
                                 outer CV fold
  inner_fold_assignments.csv     every inner GridSearchCV split actually used
                                 (per outer fold, and for the final lock refit)
  winner_lock.json               winners, CV aggregates, locked params, and
                                 sha256 of each pipeline file -- written and
                                 flushed BEFORE the test partition is read
  run_manifest.json              data/config/code hashes, package versions,
                                 timestamps, output-file hashes

Also provides validate_run_provenance(), a non-training check that a
provenance directory is internally consistent and still matches the
artifacts on disk. Nothing in this module fits a model.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import platform
import sys
from importlib import metadata as _importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_SCHEMA_VERSION = 1

# every source file whose content can change what a checkpoint run produces
CODE_FILES = (
    "src/data/checkpoint_contract.py",
    "src/data/checkpoint_validation.py",
    "src/data/checkpoint_splits.py",
    "src/data/splits.py",
    "src/features/clustering.py",
    "src/models/checkpoint_pipeline_builder.py",
    "src/models/checkpoint_train_run.py",
    "src/models/checkpoint_provenance.py",
    "src/models/checkpoint_inference.py",
)
PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "joblib", "PyYAML")

# required order of events in events.jsonl for a valid run
EVENT_ORDER = (
    "run_started", "split_built", "outer_cv_done", "winners_selected",
    "winners_locked", "test_set_first_access", "test_eval_done",
    "artifacts_saved", "run_completed",
)


# ----------------------------------------------------------------------
# hashing / environment
# ----------------------------------------------------------------------

def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Hash of a frame's canonical CSV serialization (index excluded)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def file_record(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path), "exists": True, "sha256": sha256_file(path), "bytes": st.st_size,
        "mtime_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(),
    }


def code_hashes(root: Path = PROJECT_ROOT) -> dict:
    return {rel: file_record(root / rel) for rel in CODE_FILES}


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = _importlib_metadata.version(name)
        except _importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_info() -> dict:
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": package_versions(),
    }


# ----------------------------------------------------------------------
# partition / fold tables
# ----------------------------------------------------------------------

def build_partition_table(filtered: pd.DataFrame, train_idx, test_idx, folds,
                          group_col: str, id_col: str) -> pd.DataFrame:
    """One row per filtered-input row. `folds` index into the TRAIN partition
    (train_idx order), exactly as produced by build_shared_split. Raises if
    any row is unassigned/double-assigned or any lot crosses a boundary."""
    n = len(filtered)
    partition = np.array([""] * n, dtype=object)
    partition[np.asarray(train_idx)] = "train"
    partition[np.asarray(test_idx)] = "test"
    assert (partition != "").all(), "some rows are in neither train nor test"
    assert len(train_idx) + len(test_idx) == n, "train/test indices overlap"

    outer = np.full(n, -1, dtype=int)
    train_idx = np.asarray(train_idx)
    for k, (_, va) in enumerate(folds):
        rows = train_idx[np.asarray(va)]
        assert (outer[rows] == -1).all(), f"outer fold {k}: row assigned to two validation folds"
        outer[rows] = k
    assert (outer[train_idx] >= 0).all(), "a train row is in no outer validation fold"

    table = pd.DataFrame({
        "row_index": np.arange(n),
        group_col: filtered[group_col].to_numpy(),
        id_col: filtered[id_col].to_numpy(),
        "partition": partition,
        "outer_fold": outer,
    })
    assert not table.duplicated([group_col, id_col]).any(), "duplicate (lot, wafer) identity"
    lots_per = table.groupby(group_col)["partition"].nunique()
    assert (lots_per == 1).all(), f"lots split across train/test: {list(lots_per[lots_per > 1].index)}"
    tr = table[table.partition == "train"]
    assert (tr.groupby(group_col)["outer_fold"].nunique() == 1).all(), "lot split across outer folds"
    return table


def build_inner_fold_table(train_df: pd.DataFrame, level_splits: dict, group_col: str, id_col: str) -> pd.DataFrame:
    """level_splits: {level_name: (row_positions_in_train_df, [(inner_tr, inner_va), ...])}
    where inner indices are positions within row_positions_in_train_df (i.e.
    exactly what GridSearchCV's cv= received). Emits one row per
    (level, inner_fold, validation row)."""
    parts = []
    for level, (positions, splits) in level_splits.items():
        positions = np.asarray(positions)
        for j, (itr, iva) in enumerate(splits):
            rows = positions[np.asarray(iva)]
            sub = train_df.iloc[rows]
            parts.append(pd.DataFrame({
                "level": level, "inner_fold": j, "train_df_row": rows,
                group_col: sub[group_col].to_numpy(), id_col: sub[id_col].to_numpy(),
                "n_inner_train_rows": len(itr),
            }))
            lots_tr = set(train_df[group_col].iloc[positions[np.asarray(itr)]])
            assert not (lots_tr & set(sub[group_col])), f"{level} inner fold {j}: lot in inner train and val"
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def splits_equal(a, b) -> bool:
    return len(a) == len(b) and all(
        np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b)
    )


# ----------------------------------------------------------------------
# live recording
# ----------------------------------------------------------------------

class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


class TeeLog:
    """Context manager: everything printed to stdout/stderr is also written
    (and flushed line-by-line) to `path`, including tracebacks."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._old = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._old[0], self._f)
        sys.stderr = _Tee(self._old[1], self._f)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            import traceback
            traceback.print_exception(exc_type, exc, tb)  # into the log too
        sys.stdout, sys.stderr = self._old
        self._f.close()
        return False


class ProvenanceRecorder:
    def __init__(self, prov_dir: Path, run_id: str):
        self.dir = Path(prov_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.events_path = self.dir / "events.jsonl"

    def event(self, name: str, **details) -> str:
        ts = utc_now()
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts_utc": ts, "run_id": self.run_id, "event": name, **details}, default=str) + "\n")
            f.flush()
        return ts

    def write_json(self, name: str, obj) -> str:
        path = self.dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
            f.flush()
        return sha256_file(path)

    def write_frame(self, name: str, df: pd.DataFrame) -> str:
        df.to_csv(self.dir / name, index=False, lineterminator="\n")
        return sha256_file(self.dir / name)


# ----------------------------------------------------------------------
# validation (non-training)
# ----------------------------------------------------------------------

def read_events(prov_dir: Path) -> list[dict]:
    with open(Path(prov_dir) / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_run_provenance(out_dir: Path, check_data_hash: bool = True) -> list[str]:
    """Returns a list of problems (empty == consistent). Checks event order
    (lock strictly before first test access), pipeline files unchanged since
    lock, recorded file hashes, and that saved prediction rows agree with the
    recorded partition. Never fits or refits anything."""
    out_dir = Path(out_dir)
    prov = out_dir / "provenance"
    problems = []
    for name in ("run_manifest.json", "events.jsonl", "winner_lock.json",
                 "partition_assignments.csv", "inner_fold_assignments.csv", "run_log.txt"):
        if not (prov / name).exists():
            problems.append(f"missing provenance/{name}")
    if problems:
        return problems

    manifest = json.loads((prov / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        problems.append(f"schema_version {manifest.get('schema_version')} != {PROVENANCE_SCHEMA_VERSION}")
    if manifest.get("retrospective"):
        problems.append("manifest is marked retrospective -- not original run evidence")

    events = read_events(prov)
    run_ids = {e["run_id"] for e in events}
    if run_ids != {manifest.get("run_id")}:
        problems.append(f"events.jsonl run_ids {run_ids} != manifest run_id {manifest.get('run_id')}")
    first = {}
    for e in events:
        first.setdefault(e["event"], e["ts_utc"])
    missing = [e for e in EVENT_ORDER if e not in first]
    if missing:
        problems.append(f"missing events: {missing}")
    else:
        times = [first[e] for e in EVENT_ORDER]
        if times != sorted(times):
            problems.append(f"events out of order: {list(zip(EVENT_ORDER, times))}")
        # the order in the append-only file itself must also respect the protocol
        pos = {e: next(i for i, ev in enumerate(events) if ev["event"] == e) for e in EVENT_ORDER}
        if [pos[e] for e in EVENT_ORDER] != sorted(pos[e] for e in EVENT_ORDER):
            problems.append("events.jsonl line order violates protocol")

    lock = json.loads((prov / "winner_lock.json").read_text(encoding="utf-8"))
    for cp, rec in lock["checkpoints"].items():
        p = out_dir / "models" / f"{cp}_final_pipeline.joblib"
        if not p.exists() or sha256_file(p) != rec["pipeline_sha256"]:
            problems.append(f"{cp}: pipeline file differs from the one hashed at lock time")

    for rel, rec in manifest.get("output_files", {}).items():
        p = out_dir / rel
        if not p.exists() or sha256_file(p) != rec["sha256"]:
            problems.append(f"output file changed since run: {rel}")

    part = pd.read_csv(prov / "partition_assignments.csv")
    if sha256_file(prov / "partition_assignments.csv") != manifest.get("partition_assignments_sha256"):
        problems.append("partition_assignments.csv hash differs from manifest")
    g, w = manifest["group_column"], manifest["secondary_id_column"]
    train_keys = set(zip(part.loc[part.partition == "train", g], part.loc[part.partition == "train", w]))
    test_keys = set(zip(part.loc[part.partition == "test", g], part.loc[part.partition == "test", w]))
    if train_keys & test_keys or set(part.loc[part.partition == "train", g]) & set(part.loc[part.partition == "test", g]):
        problems.append("partition table has lot/wafer overlap between train and test")
    fold_of = dict(zip(zip(part[g], part[w]), part["outer_fold"]))
    for cp in lock["checkpoints"]:
        pf = out_dir / "predictions" / f"{cp}_predictions.csv"
        if not pf.exists():
            problems.append(f"{cp}: predictions file missing")
            continue
        pr = pd.read_csv(pf)
        oof = pr[pr.source == "grouped_oof_cv"]
        ho = pr[pr.source == "holdout"]
        if set(zip(oof.LotName, oof.WaferNum)) != train_keys:
            problems.append(f"{cp}: OOF prediction rows != recorded train partition")
        if set(zip(ho.LotName, ho.WaferNum)) != test_keys:
            problems.append(f"{cp}: holdout prediction rows != recorded test partition")
        if pr.duplicated(["LotName", "WaferNum"]).any():
            problems.append(f"{cp}: duplicate identifiers in predictions")
        if any(fold_of.get(k, -1) < 0 for k in zip(oof.LotName, oof.WaferNum)):
            problems.append(f"{cp}: OOF row without an outer fold in partition table")

    if check_data_hash:
        data = manifest["inputs"]["data_csv"]
        if not Path(data["path"]).exists() or sha256_file(Path(data["path"])) != data["sha256"]:
            problems.append("input data file changed or missing since run")

    log = (prov / "run_log.txt").read_text(encoding="utf-8", errors="replace")
    if "Done. Total time" not in log:
        problems.append("run_log.txt does not contain the completion line")
    return problems


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate a checkpoint run's provenance directory (no training).")
    ap.add_argument("out_dir", nargs="?", default=str(PROJECT_ROOT / "outputs" / "checkpoint_training"))
    args = ap.parse_args()
    probs = validate_run_provenance(Path(args.out_dir))
    if probs:
        print("PROVENANCE PROBLEMS:")
        for p in probs:
            print("  -", p)
        sys.exit(1)
    print("provenance OK:", args.out_dir)
