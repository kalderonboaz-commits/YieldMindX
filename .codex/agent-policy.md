# Agent team and runtime policy

This is a referenced instruction document, not an automatically parsed Codex configuration file. AGENTS.md directs the coordinator here before delegation. Executable runtime settings belong in config.toml and agents/*.toml beside this file.

Roles are authoritative in .codex/agents/*.toml; skills load those roles and add a specific workflow. AGENTS.md supplies project context and the team registry. The names JANUS and ARIA were adapted from the Nexus reference; KALI was newly authored using their template. These roles assume no product domain or application language.

## Invocation

Use $janus-cpto, $aria-uiux, or $kali-dev to adopt a character in the current conversation. This reads instructions; it does not change that conversation's model or create another agent.

For delegated work, select the custom agent through the actual runtime's supported selector when available. Otherwise spawn a general subagent with an explicit instruction to read the role file and relevant skill, and label this manual role loading. A task name alone does not load a TOML file. Do not claim native custom-agent discovery when it was not observed.

Role files deliberately omit model and effort. Skill loading applies role instructions only, never executable configuration.

## Team execution

Run JANUS in the primary conversation. Delegate independent work when it improves a requested team task; use ARIA and up to two KALI instances subject to available slots. Three workers plus the primary require four total slots. The project cap is three spawned workers; the host can impose a lower limit. Avoid nesting another coordinator just to adopt JANUS.

JANUS's implementation boundary and explicit user-approval exception are defined in agents/janus-cpto.toml. ARIA owns design prototypes; KALI owns implementation tooling, tests, application code, and integration. Review fixes return to the owner. A stalled worker is reassigned or sequenced, never replaced by JANUS writing the code. These are behavioral instructions; this policy does not enforce filesystem permissions.

Each assignment states the outcome, source brief, owned files, shared interfaces, acceptance evidence, and return artifact. Agree shared interfaces before dependent work. Choose one KALI as final integrator and give that agent sole ownership of integration files during integration.

Do independent work in parallel; sequence dependencies. On partial failure, retain successful results, identify the remaining blocker and owner, and continue unaffected work. Retry or change the plan only for a concrete reason. JANUS reviews the combined evidence, resolves material conflicts, and records acceptance.

## Model selection

These are local starting heuristics, not an OpenAI benchmark result or a promise of optimal cost or quality. Explicit user choices take precedence.

| Task | Starting selection |
|---|---|
| Trivial, well-specified change or lookup | Sol (gpt-5.6-sol), low |
| Bounded design, implementation, or planning | Sol, medium |
| Complex integration or review | Sol, high |
| Consequential ambiguity or difficult synthesis | Astra (gpt-6-astra), high |
| Exceptional unresolved reasoning after high effort | Astra, max, when the benefit justifies the cost |

Pass model and effort through actual spawn/configuration fields. If this runtime cannot override with a full-history fork, use supported bounded/no-history context and pass the relevant sources explicitly. Check current exposed availability and supported effort values.

If an override is unavailable, disclose the limitation. For a recommendation use the available inherited model when adequate; for a user-required exact model, report the unavailable requirement instead of silently substituting. Record requested settings separately from effective settings: claim effective values only from runtime metadata, never from the agent's assertion. The parent conversation's model remains a user/session choice.

## Validation and classroom sequence

Static validation covers file syntax, references, and instruction consistency. Manual role-loading evaluations cover observed task behavior. Native discovery and effective model settings need evidence from the actual client. Keep these claims separate.

First run the instructor's fresh-chat test in the instructor test plan (not included in this scaffold). Collect feedback and evaluation results. Then prepare the student kit: a self-contained ARIA exercise, instructor reference solutions for the other roles, and progressive team exercises. That kit is deferred until the instructor test; it is not already built.

The eventual ARIA-only package must include her role and skill plus a small neutral brief and project instructions without missing named teammates or Snake3d document dependencies. Evaluate outcomes rather than expecting identical generated prompts.

## Documentation basis

Checked 2026-09-09 against official [custom-agent configuration](https://learn.chatgpt.com/docs/agent-configuration/subagents), [skill guidance](https://learn.chatgpt.com/docs/build-skills), and [model prompting guidance](https://developers.openai.com/api/docs/guides/latest-model). Runtime availability must still be checked in the student client.
