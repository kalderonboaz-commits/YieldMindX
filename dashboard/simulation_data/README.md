# IT data-flow simulation — files and upload guide

**Everything in this directory is a synthetic demo of what an eventual real
IT data feed might look like.** It uses real synthetic-dataset measurement
values for 20 real holdout wafers, but the dates attached to them are
entirely invented for this demo. Nothing here is a recovered or estimated
real production timestamp.

## Files

| File | Rows | Columns | Purpose |
|---|---|---|---|
| `simulation_OHMIC.csv` | 20 | LotName, WaferNum, SimulationDate, 16 `OHMIC_*` columns | OHMIC checkpoint measurements |
| `simulation_FIC.csv` | 20 | LotName, WaferNum, SimulationDate, 109 `OHMIC_*`+`FIC_*` columns | FIC checkpoint measurements (dashboard label: BFIC) |
| `simulation_SIN.csv` | 20 | LotName, WaferNum, SimulationDate, 195 `OHMIC_*`+`FIC_*`+`SIN_*` columns | SIN checkpoint measurements |
| `simulation_THIN.csv` | 20 | LotName, WaferNum, SimulationDate, 317 `OHMIC_*`+`FIC_*`+`SIN_*`+`THIN_*` columns | THIN checkpoint measurements |
| `simulation_actual_results.csv` | 20 | LotName, WaferNum, SortingYield, ResultDate | Simulated actual-yield results, available after THIN |

`SortingYield` and every column listed in `config/checkpoint_contract.yaml`'s
`leakage_columns` are excluded from all four measurement files (verified
programmatically when generated — see `generate_simulation_csvs.py`'s
assertions, run from `C:\Users\halif\...\scratchpad\` during generation, not
stored in the project).

## Where the wafers and values come from

**The same 20 wafers, selected deterministically, in all five files.**
Selection rule: take the project's existing synthetic holdout test
partition (the untouched 300-row / 20-lot split already used for the four
checkpoint models' test evaluation), sort by `(LotName, WaferNum)`
ascending, take the first 20 rows. No randomness. The 20 wafers are lot
`L260001` (all 15 wafers) and lot `L260005` (wafers W01–W05) — a
consequence of alphabetical sort, not a deliberate lot selection; re-running
the generation script against the same data and contract reproduces the
exact same 20 wafers every time.

Every measurement value is copied as-is from the original synthetic
dataset's raw columns for that wafer — nothing is altered, resampled, or
re-derived. `SortingYield` in `simulation_actual_results.csv` is the real
synthetic-dataset target value for that wafer, not a fabricated number.

## Dates: invented, not recovered — documented explicitly

The project's dataset has no per-checkpoint date field (verified in a prior
read-only review: only `FinishLineDate` and `Sorting Date` exist, both
whole-wafer, non-checkpoint-specific, and always logged after all four
checkpoints complete). To demonstrate the dashboard's simulation mode, each
wafer here is assigned a **fictional** cumulative date sequence:

| Checkpoint | Month | Day-of-month |
|---|---|---|
| OHMIC | 2026-06 | `03 + wafer_index` (03–22) |
| FIC | 2026-07 | `03 + wafer_index` |
| SIN | 2026-08 | `03 + wafer_index` |
| THIN | 2026-09 | `03 + wafer_index` |
| Actual result (`ResultDate`) | 2026-10 | `03 + wafer_index` |

`wafer_index` is the wafer's 0-based position in the deterministic 20-wafer
selection above (0 for `L260001`/W01, …, 19 for `L260005`/W05). This
guarantees, for every wafer: `OHMIC < FIC < SIN < THIN < ResultDate`
(verified programmatically at generation time), with each checkpoint in a
different month and some day-of-month spread so points aren't all stacked
on one date. **These dates do not represent anything that happened in
reality — they exist only so the simulation has a four-month, then a
result-availability, timeline to plot.**

## Upload guide (dashboard, section 1, "IT data-flow simulation" panel)

1. Launch the dashboard (`dashboard\Launch_Dashboard.bat`).
2. In section 1, find the **"IT data-flow simulation"** panel (separate
   from the live "Measurements CSV" panel above it).
3. Upload in this order, selecting the matching checkpoint each time, and
   click **"Run simulation prediction"** after each:
   1. `simulation_OHMIC.csv` → checkpoint **OHMIC**
   2. `simulation_FIC.csv` → checkpoint **BFIC**
   3. `simulation_SIN.csv` → checkpoint **SIN**
   4. `simulation_THIN.csv` → checkpoint **THIN**
4. Upload `simulation_actual_results.csv` via the panel's **"Actual results
   CSV"** control and click **"Upload actual results"**.
5. Switch the chart-view toggle above the four charts to **"IT-flow
   simulation"** to see the simulated timeline (Jun–Sep 2026 predictions,
   actual yield joined into each point's tooltip once uploaded). Switch back
   to **"Live predictions"** at any time — that view is unaffected by any of
   this and keeps showing real prediction-creation timestamps as before.

Re-uploading the same simulation file again updates that wafer/checkpoint's
simulated point in place rather than adding a duplicate — safe to re-run.
