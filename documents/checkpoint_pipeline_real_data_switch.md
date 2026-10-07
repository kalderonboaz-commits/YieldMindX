# Switching the four-checkpoint pipeline from synthetic to real data

Everything below assumes real data uses the **same column names** as the
synthetic dataset (the explicit assumption this pipeline was built under).
If real column names differ, `src/data/checkpoint_validation.py`'s schema
check will fail loudly and name the missing columns -- it will not guess.

All commands run from the project root with the project virtual environment:

```
cd D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table
.venv\Scripts\python.exe <script>
```

## 1. Place the real data file and flip the contract

1. Put the real CSV at `data/real_production_data.csv` (or edit the path below
   to point wherever it actually lives).
2. Edit `config/checkpoint_contract.yaml`:
   - `mode: "real"` (was `"synthetic"`)
   - `input_csv_path.real:` -- confirm/update the path
   - `process_value.real: "0.25 HP"` -- already set; do not change this to
     match the synthetic value. `CheckpointContract.load()` asserts the two
     mode values must differ and will refuse to load otherwise.

Nothing else in the contract needs to change if column names truly match.

## 2. Validate the real file before anything else touches it

```
.venv\Scripts\python.exe -m src.data.checkpoint_validation
```

Confirms: required columns present, `Process == "0.25 HP"` row/lot counts,
zero LotName+WaferNum duplicates, target plausibility, and the per-checkpoint
candidate column inventory (OHMIC/FIC/SIN/THIN counts). Fix any reported
schema gap before proceeding -- do not patch around it in code.

## 3. Rebuild the shared split and confirm no lot crosses partitions

```
.venv\Scripts\python.exe -m src.data.checkpoint_splits
```

This produces a brand-new 80/20-lot split from the real data's own lots
(`random_state=42`, same mechanism, but the actual lots and row counts will
differ from the synthetic run). `verify_no_lot_crosses_partitions` must pass.

## 4. Re-run EDA on the real TRAINING partition only, before any training

Re-run the same EDA modules used for synthetic data
(`src/data/checkpoint_eda_v2.py` and the Sweetviz generation pattern used
during the synthetic EDA phase) against the real training partition. Do not
reuse synthetic EDA conclusions or any globally-fitted preprocessing decision
-- every learned step still refits inside each CV fold during training
regardless of what EDA suggests. Generate heavy checkpoints (SIN, THIN) one
at a time in a fresh process if memory pressure recurs, exactly as done for
synthetic data. Never access the real test partition during this step.

## 5. Full refit -- all 24 candidate configurations

```
.venv\Scripts\python.exe -m src.models.checkpoint_train_run
```

`main()` is fully config-driven now -- it no longer hard-asserts
`mode == "synthetic"`. Setting `mode: "real"` in step 1 is enough; no code
change is needed to run real data. What actually gates the run is explicit,
not a mode check: `check_schema()` (required columns present) and
`filter_process()` (the process filter leaves at least one row) both assert
loudly and name the problem if the real file doesn't qualify. Output
directory, the saved `label` field, and the printed/logged report are all
derived from `contract.mode`, so a real run writes to
`outputs/checkpoint_training_real/` (synthetic artifacts at
`outputs/checkpoint_training/` are never touched) and every metadata.json
says `"label": "REAL DATA RESULT"` with `"split_provenance": {"mode": "real", ...}`.

This refits every model/mode/checkpoint combination from scratch -- selection,
locking, and test evaluation all repeat on the real data's own split. No
synthetic-trained artifact, hyperparameter, or feature list is reused.

**Before launching:**
- Run the non-training checks first (section 5b). Every test must pass.
- Make sure nothing is editing `src/`, `config/` or the data file while the
  run is in progress. The manifest records whether the code hashes changed
  during the run (`code_files_unchanged_during_run`).
- The run **refuses to start** if the output directory already contains
  `*_final_pipeline.joblib` or a `provenance/` folder. To retrain a mode,
  first rename the old directory (for example
  `outputs/checkpoint_training_real_2026-10-20/`). Artifacts are never
  overwritten in place.

**Do not patch artifacts after the run.** If a column name, label or
metadata field turns out to be wrong, fix the code and rerun, or write any
correction to a separate, clearly-labeled file. Editing saved metadata or
predictions breaks the hashes recorded in `provenance/run_manifest.json`,
and section 5b's validator will (correctly) report the run as altered. This
is what happened to the synthetic run at 18:22 on 2026-10-05; see
`outputs/checkpoint_training/provenance_retrospective/RETROSPECTIVE_NOTICE.md`.

## 5a. Provenance written automatically by every run

`outputs/checkpoint_training_real/provenance/`:

| File | Content |
|---|---|
| `run_log.txt` | Full stdout and stderr, tee'd live, including any traceback. |
| `events.jsonl` | Append-only, UTC-timestamped events, in order: `run_started`, `split_built`, `outer_cv_done`, `winners_selected`, `winners_locked`, `test_set_first_access`, `test_eval_done`, `artifacts_saved`, `run_completed`. |
| `partition_assignments.csv` | Every row's LotName, WaferNum, train/test assignment and outer CV fold. |
| `inner_fold_assignments.csv` | The inner GridSearchCV validation fold of every row, for each outer fold and for the final lock refit. Read back from the fitted searches; identical across all 24 configs (asserted). |
| `winner_lock.json` | Winners, CV aggregates, locked hyperparameters, feature lists and the **sha256 of each saved pipeline**. Written to disk before `test_set_first_access`. |
| `run_manifest.json` | sha256, size and mtime of the data CSV, contract YAML and every pipeline source file; Python and package versions; start and end times; hashes of all output files. |

Each `metadata.json` also carries `split_provenance.run_id`, linking it to
this record.

## 5b. Non-training checks (run before AND after a real training run)

```
.venv\Scripts\python.exe tests\test_checkpoint_pipeline_leakage.py
.venv\Scripts\python.exe tests\test_checkpoint_provenance.py
```

`test_checkpoint_provenance.py` trains no real model:
- It uses the real split and search code with a recording transformer and
  DummyRegressor.
- It dry-runs the full orchestration into a temporary directory and
  validates the provenance that dry run writes.
- It checks that the training entry point refuses to overwrite existing
  artifacts.
- It checks that the 2026-10-05 synthetic artifacts still match their
  retrospective hashes.

Note that `test_checkpoint_pipeline_leakage.py` currently reads the
synthetic `outputs/checkpoint_training/` artifacts, so it verifies the
synthetic run, not a real one. The validator below is what checks a
real run.

After a real run finishes, validate it:

```
.venv\Scripts\python.exe -m src.models.checkpoint_provenance outputs\checkpoint_training_real
```

The validator exits non-zero and lists every problem if:
- any required provenance file is missing;
- events are missing or out of order (in particular `winners_locked` must
  precede `test_set_first_access`);
- any pipeline differs from the one hashed at lock time;
- any output file changed after the run;
- prediction rows do not match the recorded train/test partition;
- the input data file has changed since the run.

Archive the whole output directory (artifacts plus `provenance/`) together.

### What the automatic provenance still cannot prove
- That no one looked at the test partition outside this run. The event
  log only covers what the run itself did.
- Who changed the code between runs. Hashes show *that* a file changed,
  not why.
- That `test_df` was never touched before the marker. The run builds
  `test_df` together with `train_df`, and prints its row/lot counts early.
  The first read of its feature or target values is logged as
  `test_set_first_access`.

## 6. Inference afterward

```
.venv\Scripts\python.exe -m src.models.checkpoint_inference OHMIC measurements.csv --mode real --out preds.csv
```

`CheckpointPredictor(checkpoint_name, mode="real")` reads from
`outputs/checkpoint_training_real/models/`, and has two independent guards
against ever serving a synthetic-trained artifact for a real-mode request
(or vice versa): the two modes' artifacts physically live in different
directories, AND the loader cross-checks the `mode` recorded in each
metadata.json against the mode you requested, raising `AssertionError`
immediately on any mismatch -- so even a manually copied/misplaced file
would be caught, not silently served.

## Existing synthetic run (2026-10-05): provenance is retrospective only

The synthetic run in `outputs/checkpoint_training/` predates automatic
provenance and has **no** `provenance/` directory. The validator reports it
as missing provenance, and that is correct. What was reconstructed or
recovered afterward is in
`outputs/checkpoint_training/provenance_retrospective/`. Every file there is
labeled RETROSPECTIVE, and `RETROSPECTIVE_NOTICE.md` explains:
- the post-run 18:22 metadata/predictions patch;
- the 18:22 and 18:37 code edits;
- what remains unverifiable.

It was generated by
`src/validation/checkpoint_retrospective_provenance.py`. Do not cite it as
original run evidence.

## What this pipeline still cannot verify automatically

- That any real column actually carries the same physical meaning as its
  synthetic namesake, beyond matching by name.
- Physical plausibility of real-data outliers, units, or sentinel values --
  rerun the EDA's data-quality checks and have an engineer confirm findings
  that were only hypotheses on synthetic data.
- Whether the chronological-split candidates (`Sorting Date`, `Year_WW`,
  `Sort_WW`) are populated/reliable in the real file -- `checkpoint_splits.py`
  reports this but does not force a chronological split either way.
