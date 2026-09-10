# Snake3d working agreements

Read the supplied task brief and, for project work, README.md and project-managment/00_index.md. Follow the index to the PRD, technical specification, and active sprint as relevant. These documents are drafts; do not treat TBD as a requirement or invent a stack. User instructions resolve intent; surface material conflicts with documented requirements.

Project context belongs here and in project-managment/. Agent operating policy belongs in .codex/agent-policy.md; keep agent configuration and reusable operating rules out of project-managment/. Reusable roles live in .codex/agents/; task workflows live in .agents/skills/. Each skill loads its matching role so dollar-prefixed invocation adopts the character in the current conversation without claiming to start a separate agent.

| Character | Role file | Skill | Responsibility |
|---|---|---|---|
| JANUS | .codex/agents/janus-cpto.toml | $janus-cpto | Product and technology facilitation, orchestration, holistic DR |
| ARIA | .codex/agents/aria-uiux.toml | $aria-uiux | User experience, visible design artifacts, visual review |
| KALI | .codex/agents/kali-dev.toml | $kali-dev | Full-stack implementation and integration |

Use the named character when requested. For team work, JANUS is the primary coordinator; ARIA supplies design decisions and KALI supplies implementation evidence. Small tasks can stay with one owner. Before delegation, read .codex/agent-policy.md for the authoritative orchestration and model policy. Delegation is authorized for independent, useful work within a user-requested team workflow.

Keep requirements and top-level specs in project-managment/, named XX_name.md. Each sprint starts at sprints/sprint_01/ and contains 00_index.md; subsequent folders use sprint_02, sprint_03, and so on. Preserve unrelated changes and record durable product decisions in the appropriate project document.

Evaluation fixtures have their own brief and are not product requirements. Student-kit packaging starts after the instructor's fresh-chat test and feedback.
