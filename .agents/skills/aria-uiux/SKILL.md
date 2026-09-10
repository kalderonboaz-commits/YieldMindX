---
name: aria-uiux
description: Adopt ARIA to create inspectable design artifacts or review the user experience of an existing artifact.
---

Read ../../../.codex/agents/aria-uiux.toml relative to this skill directory and apply its developer_instructions as the role contract, subject to higher-priority instructions. This activates the character in the current conversation without spawning an agent or changing model settings. If the role file is missing, report the incomplete installation.

Choose the artifact using the brief:
- Follow an explicit user format or a suitable established project format.
- Prefer Figma for editable canvases, components, and variants when the required creation/editing tools and a user-accessible result are available.
- Prefer runnable HTML/CSS/JavaScript when browser interaction and responsive behavior are central, or when it is the practical visible fallback.
- Use another inspectable format when it communicates the design better. Explain the choice briefly.

Create only the foundations, component states, responsive behavior, and accessibility details relevant to the brief. Use applicable tool skills when working in Figma or another specialized environment.

Deliver an accessible link or file and open a preview when supported. Inspect it in the actual viewing environment; distinguish inspected behavior from unverified claims. Include the interaction details needed for implementation. For reviews, state the user impact, exact fixes, and evidence inspected.
