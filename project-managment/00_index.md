# Project Management Index

This folder hosts all project PRDs and top-level specifications.

## Documents

- [01_prd.md](01_prd.md) — Product requirements.
- [02_spec.md](02_spec.md) — Top-level technical specification.
- [03_game_research.md](03_game_research.md) — Comparable games, procedural-world research and product rationale.
- 04_conversation_design_review.md (not included in this scaffold) — Conversation GBU review, evidence gaps, and revision acceptance criteria.
- 05_agent_test_plan.md (not included in this scaffold) — Fresh-chat instructor exercise and student-kit sequence.
- 06_agent_evaluation.md (not included in this scaffold) — Remediation, observed evaluation results, and remaining client-test limits.

## Naming conventions

- All specification documents use `XX_name.md`, with a two-digit numeric prefix and a descriptive snake_case name.
- Keep `00_index.md`, `01_prd.md`, and `02_spec.md` as the core documents.
- Number additional top-level documents from `03` onward and link them here.
- Store sprint plans in `sprints/sprint_01/`, `sprints/sprint_02/`, and so on, starting at 1.
- Use the same `XX_name.md` convention for sprint documents.

## Sprints

- [Sprint 01](sprints/sprint_01/00_index.md) — PRD and UI/UX; current planning phase.
- [Sprint 02](sprints/sprint_02/00_index.md) — World creation and basic movement.
- [Sprint 03](sprints/sprint_03/00_index.md) — Real playability and validation.

## Software and delivery scaffold

- [07_software_structure.md](07_software_structure.md) - Frontend, backend, contracts, databases and repository layout.
- [08_deployment.md](08_deployment.md) - Environments, delivery, migrations and recovery.
- [09_engineering_workflow.md](09_engineering_workflow.md) - Git, testing and CI/CD conventions.
