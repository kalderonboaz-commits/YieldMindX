# Software structure

Status: reusable scaffold guidance, added 2026-09-10. This layout supports a frontend, backend and databases; it does not select a backend language, database engine or cloud. Record actual choices in 02_spec.md before implementation. There is no single universal folder tree.

## Repository map

```text
/
  AGENTS.md
  README.md
  CONTRIBUTING.md
  .codex/                   Agent roles, runtime config and team/model policy
  .agents/skills/            On-demand workflows
  .github/                  Optional GitHub PR template and workflow blueprint
  apps/
    web/
      src/
        app/                Composition, routing and providers
        features/           Product features; colocate their unit/component tests
        components/         Reusable frontend components
        lib/                Frontend adapters and utilities
      public/               Static public assets
      tests/                Frontend integration tests
    api/
      src/
        modules/            Business capabilities and their unit tests
        platform/           Runtime, persistence and service adapters
      tests/                API and database integration tests
  packages/
    contracts/              API/schema source of truth; generated clients derived here
    ui/                     Optional shared UI package for multiple consumers
  databases/
    migrations/             Ordered, version-controlled schema migrations
    seeds/                  Synthetic development/test fixtures only
  tests/
    e2e/                    Critical cross-system user journeys
    contract/               Consumer/provider compatibility checks
    performance/            Measured load and latency budgets
    fixtures/               Shared synthetic test inputs
  deployment/
    local/                  Local service composition
    containers/             Build and container definitions when selected
    infrastructure/         Infrastructure-as-code when provider selected
    environments/
      development/
      staging/
      production/
    runbooks/               Release, rollback, restore and incident procedures
  scripts/                  Repository automation added during implementation
  project-managment/        PRD, specs and numbered sprint documents
```

.gitkeep files preserve empty folders in Git. The directory names are project conventions, not framework requirements. A selected React framework may require a different routing layout. Use only needed packages; a single frontend can keep its UI in apps/web/src/components instead of creating a shared package.

## Boundaries

- Frontend calls the backend API; it never connects directly to a private database or contains server credentials.
- Backend owns authorization, business rules and database access. Begin with one modular backend; split services only for a demonstrated requirement.
- Share public contracts, not database entities or server-only libraries, with the frontend. Generate clients from one schema source where useful and validate inputs at runtime.
- Keep unit tests beside the behavior they exercise. Package-specific integration tests stay with the app; cross-system tests live at the root.
- Keep runtime data, backups, credentials, dependency folders and generated build output outside Git. Commit manifests, lockfiles, schemas and migrations.
- For multiple databases, add engine/service subdirectories beneath migrations and seeds. Add cache or object storage only when requirements call for them.

## Stack selection

For a React frontend, evaluate a supported framework when routing/server rendering/data-loading needs justify it. A client-only course or game can use React with a build tool such as Vite. Do not add a second backend if the selected full-stack framework already covers the required server boundary. The apps/api placeholder can be removed after that choice is recorded.

Select the backend based on team experience and runtime needs; this scaffold does not require Python. Start with one database that fits the data model. Choose supported tool/runtime versions at implementation time, pin them, and commit the relevant lockfiles. A mixed-language monorepo can keep each app's native tooling; do not force a JavaScript package manager on the backend.

## First implementation decisions

Record frontend approach, backend runtime, API contract format, database engine, package manager(s), test tools, hosting, environment configuration and named owners in 02_spec.md. Then add actual manifests, executable build/test commands and deployment definitions. This archive intentionally contains no application implementation or runnable delivery pipeline.

## Basis

Reviewed 2026-09-10: [React app creation guidance](https://react.dev/learn/creating-a-react-app) and [React from scratch guidance](https://react.dev/learn/build-a-react-app-from-scratch). The folder layout and modular monorepo recommendation are project design choices informed by those sources, not an official mandated architecture.
