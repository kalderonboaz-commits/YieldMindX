# Snake3d scaffold

This archive contains the project structure, current project documentation, and full project-local Codex agent configuration from C:/SynaptixLabs/projects/snake3d, packaged on 2026-09-10. Application code, preview-server code, evaluation executables/screenshots, and Git history are omitted. The software scaffold adds Git-preserved empty directories for implementation.

## Start here

Extract the entire snake3d-scaffold folder, including the hidden .codex and .agents directories. Add the extracted folder as a Codex project source and start a task there. Read [working agreements](AGENTS.md) and [the project index](project-managment/00_index.md).

## Agent infrastructure

- [JANUS](.codex/agents/janus-cpto.toml): coordination, product decisions, and review; personal implementation requires an explicit request and approval of the concrete exception.
- [ARIA](.codex/agents/aria-uiux.toml): user experience, inspectable design artifacts, and visual review.
- [KALI](.codex/agents/kali-dev.toml): full-stack implementation and integration.
- [Runtime configuration](.codex/config.toml): agent enablement and concurrency.
- [Team and model policy](.codex/agent-policy.md): delegation and task-dependent model/effort selection.
- Project skills: [JANUS](.agents/skills/janus-cpto/SKILL.md), [ARIA](.agents/skills/aria-uiux/SKILL.md), and [KALI](.agents/skills/kali-dev/SKILL.md).

The current $janus-cpto, $aria-uiux, and $kali-dev skills still adopt roles in the current conversation. Role/skill separation was discussed but is not implemented in this snapshot. Invocation does not change the conversation model. Native agents require runtime support. Model availability, user permissions, and installed plugins are supplied by the receiving Codex environment; account credentials and global configuration are not bundled. JANUS's code boundary is a behavioral instruction, not a filesystem restriction.

## Project and evaluation documents

The PRD and specifications retain the current Snake3d requirements; they are not blank templates. Review them before adapting this scaffold to another project. Sprint folders start at sprint_01 and each includes 00_index.md. Review records describe past work and may reference code, screenshots, local URLs, or former document locations intentionally absent from this archive. They do not prove the extracted scaffold is a runnable application.

Instructor test plan (not included in this scaffold) describes an exercise to run after setup. Only the documents present in this archive are bundled. This is a project snapshot, not the finalized student course kit.
## Software and deployment scaffold

Start with [software structure](project-managment/07_software_structure.md), [deployment](project-managment/08_deployment.md), and [Git/testing workflow](project-managment/09_engineering_workflow.md). Follow [CONTRIBUTING.md](CONTRIBUTING.md) for changes.

The layout includes apps/web, apps/api, packages, databases, tests and deployment. Git defaults and an optional GitHub PR template are included. The CI/CD workflow blueprint still needs real build/test commands and a chosen hosting provider; no application code, executable pipelines, credentials or Git history are included. This is a proposed modern baseline, not a universal architecture standard. Backend language, database engine and hosting remain unselected.
