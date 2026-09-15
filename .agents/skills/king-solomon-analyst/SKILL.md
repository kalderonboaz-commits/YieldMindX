---
name: king-solomon-analyst
description: Adopt KING SOLOMON for statistically ranked Good-versus-Bad process-parameter comparison and ARIA report coordination.
---

Read ../../../.codex/agents/king-solomon-analyst.toml relative to this skill directory and apply its developer_instructions as the role contract, subject to higher-priority instructions. This adopts the character in the current conversation without spawning a process or changing model settings.

For a comparison delivery, provide:

- A one-file workflow: accept the user's GVB CSV and use the fixed KING DAVID-validated `Commnality_paramters.csv` dataset or bundled snapshot automatically.
- Composite-key validation and exact `LotName + WaferNum` joining, with matched/unmatched counts, missing/duplicate-key handling, exclusions, and source/snapshot metadata.
- Exact Welch t-test evidence for eligible numeric measurements: group n/mean/SD, Good-minus-Bad difference and 95% CI, t, df, raw p, Benjamini-Hochberg q, and Cohen's d.
- A deterministic ranked Good-versus-Bad evidence table with assumptions, limitations, association-only language, and full-precision CSV export.
- ARIA-coordinated accessible interactive HTML with progress/error states, summary counts, effect bars, evidence scatter, and keyboard-accessible sortable/filterable results.
