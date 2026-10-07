# Correction log

## 2026-10-06 — conflicting-value test left a wrong actual-yield displayed

**What happened:** during backend conflict-handling verification (previous
session), a deliberately modified file `conflicting_actual.csv` (a scratch
copy of `simulation_actual_results.csv` with `L260001`/`W01`'s
`SortingYield` changed by +1.5, from 88.95238128 to 90.45238128) was
uploaded to `/api/actual-results` to confirm the backend detects and
reports a conflicting re-upload. It did (`status: "updated"`), but this
left `90.45238128` as the *latest*, displayed value for that wafer --
diverging from the canonical `dashboard/simulation_data/simulation_actual_results.csv`.

**Detection:** compared every value currently returned by
`GET /api/actual-results` against the canonical CSV. Exactly one mismatch:
`(L260001, W01)`: displayed `90.45238128`, canonical `88.95238128`. All
other 19 wafers matched the canonical file exactly.

**Correction applied:** re-uploaded the unmodified
`simulation_actual_results.csv` through the normal
`POST /api/actual-results` endpoint (the same endpoint and flow a real user
would use -- no direct database edit, no special-case code path). The
backend's existing conflict logic recognized this as a 20th conflicting
upload for that one row (`summary: {"updated": 1, "duplicate": 19}`) and
inserted a new row with the canonical value, which is now the latest
(displayed) record.

**History preserved, nothing deleted.** `GET /api/actual-results?history=true`
for `(L260001, W01)` shows all four rows in order:
1. `88.95238128` (2026-10-06T04:55:09Z) -- original canonical upload
2. `88.95238128` (2026-10-06T04:55:09Z) -- identical repeat upload (duplicate)
3. `90.45238128` (2026-10-06T04:55:21Z, `conflicting_actual.csv`) -- the test contamination
4. `88.95238128` (2026-10-06T07:32:13Z, `simulation_actual_results.csv`) -- **this correction, now latest**

The displayed/latest value for every one of the 20 simulation wafers now
matches `simulation_actual_results.csv` exactly. `conflicting_actual.csv`
was a scratch test file, never part of the committed `simulation_data/`
deliverables, and was not retained.
