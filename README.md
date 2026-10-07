# YieldMindX — four-checkpoint yield pipeline + dashboard (update)

This update adds the four-checkpoint (OHMIC → FIC → SIN → THIN) final-yield
prediction pipeline, its dashboard integration, and a static offline preview,
on top of this branch's existing source tree. **All results referenced here
are synthetic-data results** (`Process == "GaAs_pHEMT_MMIC"`); nothing here
is a claim about real production accuracy.

## Quick links

| What | Where |
|---|---|
| Static offline preview (no install needed) | [`preview/index.html`](preview/index.html) — see below |
| Connected dashboard (needs Python + trained models) | [`dashboard/`](dashboard/) |
| Pipeline source | [`src/data/checkpoint_*.py`](src/data/), [`src/models/checkpoint_*.py`](src/models/) |
| Real-data switch instructions | [`documents/checkpoint_pipeline_real_data_switch.md`](documents/checkpoint_pipeline_real_data_switch.md) |
| Tests | [`tests/test_checkpoint_pipeline_leakage.py`](tests/test_checkpoint_pipeline_leakage.py), [`tests/test_checkpoint_provenance.py`](tests/test_checkpoint_provenance.py) |

## 1. Static preview — download and open, no install

`preview/index.html` is a single, self-contained HTML file. Its data (a small
`preview/synthetic_results_snapshot.json`, 20 wafers × 4 checkpoints, already
produced predicted yields plus already-recorded synthetic actual yields) is
embedded **inline** in the page, not fetched separately — so it works from a
plain double-click, with **no Python, no server, and no trained model files**,
including when opened directly from disk (`file://`).

**To use it:** download `preview/index.html` (that one file is enough — the
`.json` file next to it is the same data, kept separately only for
inspection/reuse) and open it in any browser.

It shows all four checkpoint charts (predicted yield dashed, actual yield
solid, LotName/WaferNum tooltips on every point) and is clearly labeled
**"SAVED SYNTHETIC RESULTS — NOT LIVE"**. Its upload/checkpoint controls are
visibly disabled with an explanation — this page cannot run new predictions.

**A GitHub file link to this page is source/download access, not a hosted
running page.** Opening `preview/index.html`'s GitHub blob URL in a browser
shows GitHub's own source viewer, not the rendered dashboard. Download the
raw file (or clone the repo) and open it locally to see it render.

## 2. Running the full pipeline later, with separately supplied data

Nothing here includes the raw dataset or trained model files (see
"Intentionally excluded" below). To run the pipeline yourself:

```
pip install -r requirements.txt
```

Place your CSV at the path configured in `config/checkpoint_contract.yaml`
(`data/correlation_data_table.csv` for synthetic mode by default), then:

```
python -m src.data.checkpoint_validation      # schema/process/target checks
python -m src.data.checkpoint_splits          # shared lot-grouped split
python -m src.models.checkpoint_train_run     # full 24-config nested-CV training
```

Full real-data switch procedure (mode flip, schema expectations, re-running
EDA, what must be refit): `documents/checkpoint_pipeline_real_data_switch.md`.

## 3. Running the connected dashboard, with separately supplied model artifacts

The connected dashboard (`dashboard/`) needs the four trained pipelines,
which are **not included** in this repository (see below). After running
`checkpoint_train_run.py` yourself (step 2 above), you'll have
`outputs/checkpoint_training/models/{OHMIC,FIC,SIN,THIN}_final_pipeline.joblib`
— place them at that same path, then:

```
dashboard\Launch_Dashboard.bat
```

This starts the local backend (`dashboard/backend/server.py`, binds to
`127.0.0.1` only) and opens the dashboard. **Missing model files are handled
gracefully**, not treated as fatal: the backend loads each checkpoint
independently at startup, and any checkpoint whose `.joblib` isn't present is
reported via `/api/health`'s `load_errors` instead of crashing the server —
you can run the dashboard with however many of the four models you actually
have.

Full backend/API documentation: `dashboard/backend/README.md`.

## Repository layout (this update's additions)

```
config/checkpoint_contract.yaml        Mode (synthetic/real), paths, cumulative checkpoint definitions
requirements.txt                        Pinned Python dependencies
src/data/checkpoint_contract.py         Contract loader/validator
src/data/checkpoint_validation.py       Schema, process-filter, target, column-inventory checks
src/data/checkpoint_splits.py           Shared lot-grouped train/test split + CV folds
src/data/checkpoint_eda_v2.py           Dual-method EDA (near-constant/outlier/redundancy)
src/models/checkpoint_pipeline_builder.py  Nested preprocessing/feature-selection/model pipelines
src/models/checkpoint_train_run.py      24-config nested-CV training/selection/evaluation orchestration
src/models/checkpoint_provenance.py     Run provenance recording + validation
src/models/checkpoint_inference.py      Loads a locked pipeline, predicts on new measurements
src/validation/checkpoint_overfitting_assessment.py  Train/CV/test gap assessment
tests/test_checkpoint_pipeline_leakage.py  Leakage/split/pipeline-structure regression checks
tests/test_checkpoint_provenance.py        Provenance-recording regression checks
dashboard/backend/server.py             Flask backend (local-only) connecting models to the dashboard
dashboard/output/                       The dashboard itself (wrapper + workspace + GVB data + Section 2-4 content)
dashboard/simulation_data/              IT-flow simulation demo CSVs + upload guide
dashboard/Launch_Dashboard.bat          One-click launcher
preview/                                Static offline preview (see above)
documents/checkpoint_pipeline_real_data_switch.md  Real-data switch procedure
outputs/checkpoint_training/            Metadata, predictions, CV results from the completed synthetic run
outputs/checkpoint_overfitting_assessment/  Train/CV/test gap results
outputs/eda_checkpoints/                EDA text summaries + plots (small files only, see below)
```

## Intentionally excluded (and why)

| Excluded | Reason |
|---|---|
| `outputs/checkpoint_training/models/*.joblib` (trained pipelines) | Explicitly excluded; supply your own after running the pipeline (section 3 above) |
| `outputs/eda_checkpoints/sweetviz_*.html` | Large generated reports (up to ~49 MB each); regenerate via Sweetviz against your own data, not needed to run the pipeline |
| `data/*.csv` (raw dataset) | Large raw dataset; supply your own |
| `dashboard/backend/predictions.db` | Local prediction history (SQLite); created fresh on first run |
| `.venv/`, caches, credentials | Local environment, never committed |
| Older Stage 1/Stage 2 exploratory source already on this branch (`src/data/load.py`, `src/explain/`, etc.) | Out of scope for this update — left exactly as already present, not modified |
| `archive/` (local project only) | Explicitly superseded/obsolete files, not pushed |

## A note on the existing `/output/` directory at repo root

This branch already had an `/output/` directory (an earlier dashboard
revision, predating this checkpoint-model integration) before this update.
This update does not modify, replace, or read from that directory — the
connected dashboard and preview added here live entirely under `dashboard/`
and `preview/`, a self-contained addition. Reconciling or retiring the older
`/output/` directory is left for a separate, explicit decision.

## Labeling

All results in `dashboard/`, `preview/`, and `outputs/` are synthetic-data
results from `Process == "GaAs_pHEMT_MMIC"`. No production-accuracy or
real-data-compatibility claim is made.
