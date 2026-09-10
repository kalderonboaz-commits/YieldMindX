# Project Management Index

This folder hosts project requirements, specifications, research, and sprint plans.

## Documents

Add `01_prd.md` for product requirements, `02_spec.md` for the technical specification, and subsequent numbered documents for research and design reviews.

## Naming conventions

- All specification documents use `XX_name.md`, with a two-digit numeric prefix and a descriptive snake_case name.
- Keep `00_index.md`, `01_prd.md`, and `02_spec.md` as the core documents.
- Number additional top-level documents from `03` onward and link them here.
- Store sprint plans in `sprints/sprint_01/`, `sprints/sprint_02/`, and so on, starting at 1.
- Use the same `XX_name.md` convention for sprint documents.

## Sprints

Add sprint plans under `sprints/sprint_01/`, `sprints/sprint_02/`, and later numbered folders as the project evolves.

## Software and delivery scaffold

- [07_software_structure.md](07_software_structure.md) - Frontend, backend, contracts, databases and repository layout.
- [08_deployment.md](08_deployment.md) - Environments, delivery, migrations and recovery.
- [09_engineering_workflow.md](09_engineering_workflow.md) - Git, testing and CI/CD conventions.
