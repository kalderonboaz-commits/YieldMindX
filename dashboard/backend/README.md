# Dashboard backend

Connects the four locked checkpoint models (`outputs/checkpoint_training/models/`)
to the dashboard in `dashboard/output/`. Predicted `SortingYield` only --
no actual end-of-line yield is connected here.

## Run

One-click: double-click `dashboard\Launch_Dashboard.bat`.

Manual:
```
cd D:\MS.c\MS.c\Gal-El\correlation_data_table\correlation_data_table
.venv\Scripts\python.exe dashboard\backend\server.py
```
Then open `http://127.0.0.1:8765/output/YieldMinx_dashboard_rev5.html`.

Binds to `127.0.0.1` only. No external network calls, no global package
installs (Flask was already present in the project `.venv`).

## API

- `GET /api/health` -- which checkpoints loaded, model info (winning
  model/feature mode/mode/label/pipeline sha256), load errors if any.
- `POST /api/predict` -- form fields `checkpoint` (`OHMIC`/`FIC`/`SIN`/`THIN`,
  model-space name; `FIC` is the dashboard's "BFIC") and `file` (measurements
  CSV). Requires `LotName` + `WaferNum` plus exactly that checkpoint's own
  cumulative measurement columns (via `CheckpointPredictor` from
  `src/models/checkpoint_inference.py`, reused unmodified). Never requires
  `SortingYield`, `FinishLineDate`, or later-checkpoint columns. Returns the
  new prediction rows or a clear `{"error": "..."}` message (400).
- `GET /api/predictions/<checkpoint>` -- full persisted history for that
  checkpoint, oldest first. Used by the dashboard to populate each chart.

## Persistence

SQLite at `dashboard/backend/predictions.db` (created on first run). One row
per prediction, append-only -- repeated predictions for the same wafer are
never merged or overwritten, so history and repeats are both visible.
Columns: `lot_name`, `wafer_num`, `checkpoint`, `predicted_yield`,
`created_at_utc` (ISO-8601, UTC), `model_name`, `feature_mode`, `mode`,
`label`, `pipeline_sha256`, `source_filename`.

To start from a clean history, stop the server and delete
`dashboard/backend/predictions.db` -- it will be recreated empty on next
start. The locked model artifacts under `outputs/checkpoint_training/` are
never touched by this backend, deleting the prediction history does not
affect them.

## What this does NOT do

- No training, tuning, or model-selection logic -- strictly loads the
  already-locked pipelines and calls `.predict()`.
- No imputation of missing CSV *columns* -- a required measurement column
  that is absent from the upload is reported as a missing-column error, not
  silently filled in. (The pipeline's own internal median imputer only fills
  missing *cell values* within columns that ARE present -- that behavior is
  unchanged from the locked pipeline and is not part of this backend.)
- No "actual yield" data or claim of any kind -- every chart explicitly
  labels that series as pending IT integration.
- No mode other than `synthetic` today, because no `real`-mode artifacts
  exist yet (see `documents/checkpoint_pipeline_real_data_switch.md`).
