# Project scaffold

This repository contains a reusable project structure and project-local Codex agent configuration. Application code, evaluation artifacts, and Git history are not included. Empty implementation directories are preserved with `.gitkeep` files.

## Start here

Add this folder as a project source and start a task there. Read [working agreements](AGENTS.md) and [the project index](project-managment/00_index.md).

## Agent infrastructure

- [JANUS](.codex/agents/janus-cpto.toml): coordination, product decisions, and review; personal implementation requires an explicit request and approval of the concrete exception.
- [ARIA](.codex/agents/aria-uiux.toml): user experience, inspectable design artifacts, and visual review.
- [KALI](.codex/agents/kali-dev.toml): full-stack implementation and integration.
- [Runtime configuration](.codex/config.toml): agent enablement and concurrency.
- [Team and model policy](.codex/agent-policy.md): delegation and task-dependent model/effort selection.
- Project skills: [JANUS](.agents/skills/janus-cpto/SKILL.md), [ARIA](.agents/skills/aria-uiux/SKILL.md), and [KALI](.agents/skills/kali-dev/SKILL.md).

The current $janus-cpto, $aria-uiux, and $kali-dev skills adopt roles in the current conversation. Native agents require runtime support. Model availability, user permissions, and installed plugins are supplied by the receiving Codex environment; account credentials and global configuration are not bundled. Role boundaries are behavioral instructions, not filesystem restrictions.

## Project documents

The project-management folder contains reusable planning and delivery guidance. Add a project PRD, technical specification, research, and sprint records there after the new project is defined.
## Software and deployment scaffold

Start with [software structure](project-managment/07_software_structure.md), [deployment](project-managment/08_deployment.md), and [Git/testing workflow](project-managment/09_engineering_workflow.md). Follow [CONTRIBUTING.md](CONTRIBUTING.md) for changes.

The layout includes apps/web, apps/api, packages, databases, tests and deployment. Git defaults and an optional GitHub PR template are included. The CI/CD workflow blueprint still needs real build/test commands and a chosen hosting provider; no application code, executable pipelines, credentials or Git history are included. This is a proposed modern baseline, not a universal architecture standard. Backend language, database engine and hosting remain unselected.
