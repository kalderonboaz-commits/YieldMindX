# Git, testing and delivery workflow

Status: reusable team defaults. Agent roles and model routing remain in .codex; this document describes repository engineering practice.

## Git

Use one repository and a stable main branch with short-lived topic branches and reviewed pull requests. Agent-created branches default to codex/<task>. Prefer small commits with clear intent. Link the relevant requirement/sprint and describe user-visible behavior, evidence and migration impact.

Protect main with required checks and review using the chosen Git host. Avoid force pushes on shared branches. Choose a consistent merge policy during setup. Tag accepted releases and record their commit/artifact identities.

Parallel contributors should use separate branches and Git worktrees when available. Git worktrees isolate working files but share repository objects; each needs its own dependency setup and local secrets. Agree file ownership and API contracts, and assign one integrator. Shared-workspace contributors must not overwrite one another's edits. Agent responsibility/model policy remains in ../.codex/agent-policy.md.

## Meaningful tests

| Layer | Location | Purpose |
|---|---|---|
| Unit | Beside feature/module code | Domain rules, boundaries and failure cases |
| Component | Beside frontend components | Interactions, states and accessibility behavior |
| Integration | apps/*/tests | API, persistence and adapter behavior with realistic dependencies |
| Contract | tests/contract | Detect incompatible API changes |
| End-to-end | tests/e2e | Critical user-visible workflows across services |
| Performance | tests/performance | Verify agreed latency/load budgets when relevant |

Use independent synthetic fixtures and disposable databases; never run destructive tests against shared production data. Test migration upgrades as well as initial setup. End-to-end tests should use user-facing locators and reliable waiting, not arbitrary sleeps. Retain failure traces/reports as CI artifacts, not source files. Flaky tests need an owner and fix, not repeated retries hiding failures.

Require checks that actually execute the relevant tests. Do not configure a scaffold-only check or an empty test command as evidence that the app works. Select tools after the stack is chosen; Playwright is an option for browser tests. Favor meaningful coverage of critical behavior over arbitrary percentage targets.

## Implementation setup

1. Choose and document the stack in 02_spec.md.
2. Populate app manifests, lockfiles and actual build/test commands.
3. Implement CI from .github/workflows/README.md or translate it to the chosen Git host.
4. Confirm failing tests block merge and successful checks have inspectable evidence.
5. Implement deployment and recovery from 08_deployment.md.

No repository is initialized or remote created by this ZIP. To start a new repository after extraction, inspect for an existing .git first. If absent, run git init -b main, inspect git status, stage the scaffold, inspect the staged diff and commit it. Add the intended remote only after selecting its destination. The included .gitignore excludes common generated and private files, but does not remove already-tracked files.

## Sources

Reviewed 2026-09-10: [Git worktrees](https://git-scm.com/docs/git-worktree), [Git ignore rules](https://git-scm.com/docs/gitignore) and [Playwright test practices](https://playwright.dev/docs/best-practices). Branch and review conventions above are proposed team choices.
