# Correlation Data Table

This project defines a reproducible analytical workflow for discovering planted relationships in a synthetic 1,500-wafer pHEMT/MMIC dataset. The primary response is `SortingYield`. The analysis must prevent target leakage and recover direct, nonlinear, threshold, interaction, and derived-stage effects.

The supplied source brief is [corelation_data_table.docx](corelation_data_table.docx). It is a ground-truth evaluation reference. Its numerical values are synthetic training landmarks, not production specifications or control limits.

## Start here

Read [AGENTS.md](AGENTS.md), the [project index](project-managment/00_index.md), the [PRD](project-managment/01_prd.md), and the [technical specification](project-managment/02_spec.md). Keep ground-truth labels and planted thresholds outside the discovery stage; use them only after discoveries are frozen for evaluation.

## Current status

- Requirements and analytical architecture are drafted from the Word brief.
- The actual 1,500-wafer dataset is not present in this scaffold.
- The complete schema, units, technology-specific Gate CD targets, and ET leakage-family membership remain unresolved.
- No analytical implementation or production deployment is claimed.
- `data_audit.py` and `eda.py` remain in the original source folder and have not been accepted as implementation inputs.

## Agent infrastructure

- [JANUS](.codex/agents/janus-cpto.toml): product direction, scope, coordination, and acceptance.
- [ATLAS](.codex/agents/atlas.toml): data quality, statistical analysis, modeling, visualization, and analytical evidence.
- [ARIA](.codex/agents/aria-uiux.toml): inspectable user experience and visual review when an interface is required.
- [KALI](.codex/agents/kali-dev.toml): implementation, testing, automation, and integration.
- [Runtime configuration](.codex/config.toml): agent enablement and concurrency.
- [Team policy](.codex/agent-policy.md): delegation and task-dependent model guidance.

Project skills are [JANUS](.agents/skills/janus-cpto/SKILL.md), [ATLAS](.agents/skills/atlas/SKILL.md), [ARIA](.agents/skills/aria-uiux/SKILL.md), and [KALI](.agents/skills/kali-dev/SKILL.md). Invoke them with `$janus-cpto`, `$atlas`, `$aria-uiux`, and `$kali-dev`. Skill invocation adopts a role in the current conversation; it does not by itself create a separate agent or change model settings.

## Planning and delivery

- [Product requirements](project-managment/01_prd.md)
- [Technical specification](project-managment/02_spec.md)
- [Software structure](project-managment/07_software_structure.md)
- [Execution and release](project-managment/08_deployment.md)
- [Engineering workflow](project-managment/09_engineering_workflow.md)
- [Sprint 01](project-managment/sprints/sprint_01/00_index.md): data contract, audit, and baseline
- [Sprint 02](project-managment/sprints/sprint_02/00_index.md): relationship discovery
- [Sprint 03](project-managment/sprints/sprint_03/00_index.md): evaluation, reporting, and hardening

The repository still contains generic placeholders for optional web, API, database, deployment, and shared-package layers. They are not selected architecture. Remove or repurpose them only after the intended execution interface and deployment boundary are approved.
