# Snake3d — Top-Level Technical Specification

Status: Sprint 01 architecture proposal · 2026-09-09
Related: [PRD](01_prd.md) · [Sprint 01](sprints/sprint_01/00_index.md)

## Decision status

Proceeding with the PRD's desktop-browser and finite-world defaults for the design prototype. This is not evidence of production performance or a completed Sprint 01 review. The design artifact uses local HTML/CSS/JavaScript with no external dependency; its renderer does not automatically select the production rendering stack. React is the user-required application framework.

Recommended first production spike: Canvas2D isometric rendering, with an engine-independent logical grid simulation. Compare the actual target scene against Q01 before locking the renderer. If depth sorting, art requirements or measured performance warrant WebGL, record that decision before Sprint 02. User decision (2026-09-09): React is required for the production application. Python is optional if a backend need is established. UI/UX review comes first; do not begin production migration before delivering the design for review. No game renderer is mandated.

## Components and responsibilities

- Simulation: fixed-step grid movement, body occupancy, turn queue, growth, collision and score; no DOM or camera dependency.
- World: finite seeded walkability graph, validated safe spawn/loops/connectivity, versioned seed contract and decorative metadata.
- Pickups: seeded selection from valid candidates, lifetime/effects in active simulation time.
- Presentation: fixed isometric projection, interpolated body, following camera, depth ordering, occlusion treatment and reduced-motion alternative.
- Interface: start/pause/outcome/settings and readable HUD; inputs become simulation commands.
- Persistence: best score and settings only; optional local storage with failure handling. No backend needed.

## State and contracts

Run state includes seed, generatorVersion, phase, tick, heading, queuedTurn, ordered body cells, food, score, active timers and elapsed active time. Cell identity uses integer x/y; elevation is decorative in MVP. Camera/animation state must not alter collision or seeded gameplay randomness.

Phases: ready, playing, paused, dead and complete. Simulation pauses on focus loss and long stalls; explicit resume required. Use separate random streams for terrain, gameplay pickups and visual decoration. Replaying the seed reproduces terrain/initial pickups; deterministic run reproduction additionally requires input timing and the same rules version.

Per tick: read queued legal turn; compute next cell; establish whether tail vacates; resolve terrain/body collision; if alive commit body and collection; update score/effects and replenish food; check completion. Define effect expiry relative to the tick clock and test boundary cases. Never use rendered pixels as collision geometry.

## Generation constraints

Build routes first and decorate afterward. Validate W01–W03 before exposing a world to the player. Avoid degree-one corridors, include cycles and plazas, and ensure bridge endpoints agree. Bound generation attempts; on failure report/retry safely before play rather than releasing invalid terrain. Pickups use finite candidate enumeration; no infinite random retry loops.

Exact map dimensions, tile scale and reference machine are Sprint 01 spike outputs. The world must be larger than the viewport and support six readable forward cells. Do not invent an endless-world requirement.

## Rendering and input

Project square-grid coordinates to a fixed isometric view. Offset the scene around an interpolated head position; do not rotate the camera. At finite-world edges retain head centering and draw backdrop. Use stable depth ordering plus prop fading/placement to preserve visibility. Reduced motion disables decorative motion and camera lag without stopping required snake movement.

Keyboard mapping and one-turn queue follow M02/M04. Menus use semantic buttons, visible focus and keyboard navigation. Prototype scene controls must be identified as preview controls rather than promised game mechanics.

## Validation strategy

- Sprint 01: visually inspect animated prototype, controls, terrain variants, pause/outcome and reduced motion. Record findings and distinguish manual checks from measured performance.
- Sprint 02: deterministic movement replay, 100-seed invariants, safe spawn and camera/occlusion checks (W01–W05, M01–M05).
- Sprint 03: tail-vacating versus retained collision, fatal reward precedence, pickup saturation, timer pause/expiry, completion and persistence failure (G01–G08); UX and quality gates U/Q.

## Remaining decisions

Production renderer, map dimensions, reference hardware, browser versions, audio assets and five-player usability results remain pending. A scripted design tour is not evidence that procedural generation or production gameplay exists. Resolve these in the Sprint 01 acceptance record before claiming completion.

