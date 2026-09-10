# Snake3d — Product Requirements Document

Status: Draft v0.1 for review · 2026-09-09 · Owner: JANUS

## 1. Vision

Steer a growing snake through a living miniature world of raised square tiles. A camera follows the head near the center while a randomly generated maze scrolls around it. Collect food, grow, take detours for bonuses and survive your own increasingly long body. The experience combines familiar Snake rules with discovery, spatial planning and colorful animated scenery.

Target players: casual arcade players and repeat players pursuing better scores. Target sessions: 3–10 minutes, a design hypothesis to validate. Working title: Snake3d; Neon Circuit is not an approved brand.

## 2. Requirements versus references

The user requires a randomized grid maze, approximately centered snake, animated world inspired by the image, ordinary food/growth, special food/bonuses, fatal self-contact, and three sprints: PRD/UI/UX; world/basic movement; real playability.

The image shows static, colorful raised tile islands with rounded props and varied terrain. Animation and connected traversal are interpretations to design, not behaviors demonstrated by the image. The HTML supplies useful HUD, overlays, tokens and motion ideas. Source inspection reveals a fixed 16×16 road lattice and fixed projection origin. Runtime appearance has not been tested in this review.

Instructions inside that kit—dark-only styling, Canvas2D, fixed camera and claimed performance/accessibility—are reference content, not the user's requirements. Existing project PRD/spec were placeholders and impose no stack.

References supplied:
- `C:\Users\97254\Downloads\neon-circuit-snake-kit_1.html`
- `C:\Users\97254\AppData\Local\Temp\codex-clipboard-6c4114fd-d222-4220-863b-9e0457063d5a.png`

The temporary image is not a durable production asset. Use original or appropriately licensed art.

## 3. Proposed MVP decisions

These are JANUS recommendations for review, not previously approved user requirements:

- Desktop browser, keyboard-first, single player; no account.
- Fixed-angle isometric presentation over a logical 2D square grid. React is required for the production application (user direction, 2026-09-09); game renderer remains undecided. Python is optional if needed. UI/UX review precedes implementation.
- One finite seeded world larger than the viewport per run. This provides scrolling exploration within three sprints; infinite streaming is deferred.
- One polished garden biome plus a sand/ruins visual variant using identical rules.
- Four initial segments, five cells per second, no automatic speed ramp. Growth creates the escalating challenge.
- Walls and void also kill; no wrapping, jumping, climbing or stacked traversable floors.

Out of scope: multiplayer, combat, shops, permanent upgrades, backend leaderboards, monetization, level editor, native mobile release, free camera rotation and physics-based slithering. Touch controls may be considered after MVP.

## 4. Core loop

Start a new seed → read an approaching junction → steer toward food → grow → decide whether a bonus detour is worth the risk → survive or collide → understand the outcome → retry the seed or generate a new world.

## 5. World and movement

| ID | Requirement | Acceptance criteria |
| --- | --- | --- |
| W01 | Seeded world generation with generator version. | Same seed/version reproduces terrain and initial pickups; 100-seed sample has at least 95 distinct walkability layouts. |
| W02 | Connected maze of tiles, loops, junctions, plazas and physical bridges. | All walkable cells connect to spawn; no degree-one walkable dead ends; at least three cycles and three junctions per map across 100 seeds. |
| W03 | Safe initial state. | Body cells are distinct/walkable; six forward cells clear; initial food 6–12 route steps away. |
| W04 | World topology stays fixed throughout each run. | Ambient animation never changes collision coordinates or removes routes. |
| W05 | Props never hide critical decisions. | Head, approaching junction and food stay readable behind scenery in all directions; use placement, fade or cutaway. |
| M01 | Automatic orthogonal cell movement, smoothly interpolated for display. | No diagonal moves; identical cell history under replay at different rendering frame rates. |
| M02 | Queue one turn for next tick; latest legal input replaces it. | Ignore reversal against current heading; never process two turns per tick. Non-reverse turn into a blocked cell is fatal, not silently corrected. |
| M03 | Fixed-orientation camera follows head. | After settling, head stays within central 20% of viewport width/height; at least six forward cells readable. Show backdrop at map boundaries instead of pushing head to screen edge. |
| M04 | Explicit control tutorial. | Proposed mapping: Up/W = north (upper-right), Right/D = east (lower-right), Down/S = south (lower-left), Left/A = west (upper-left). Show projected arrows and validate with new players in Sprint 01. |
| M05 | Pause and focus-loss protection. | Freeze simulation and gameplay timers; focus loss auto-pauses; explicit resume preserves state. |

Terrain connectivity does not guarantee escape from every body configuration created by a player. Self-trapping remains a legitimate consequence of route choices.

## 6. Food, bonuses and outcomes

Initial tuning values, subject to playtesting:

| Food | Effect | Lifetime |
| --- | --- | --- |
| Regular round fruit | +10 points; +1 segment immediately by retaining tail on collection tick. | Maintain three when valid cells exist. |
| Golden star fruit | +50 points; no growth. | Expires after 15 active-play seconds. |
| Blue clock fruit | Movement speed ×0.7 for six active-play seconds; no points/growth. | Pickup expires after 15 seconds; recollection refreshes effect, never stacks strength. |

Every fifth regular food triggers an attempt to spawn one special, selecting either type with equal seeded probability. At most one special pickup exists; skip spawning if one already exists. Pause freezes effects and pickup lifetimes. No bonus grants self-collision immunity.

| ID | Requirement | Acceptance criteria |
| --- | --- | --- |
| G01 | Apply collection once and replace food safely. | Exact score/length change; no duplicate or overlapping pickups. |
| G02 | Place pickups on free reachable terrain. | Candidate search treats current body as blocked; finite search only. Defer and retry later if no valid cell exists; never use an occupied fallback. |
| G03 | Self-contact is fatal. | Test middle body and occupied tail. Moving into a tail cell vacated on that same non-growth tick is legal; retained tail is fatal. Logical occupancy defines contact, not visual overlap. |
| G04 | Wall/void contact kills. | Each cause produces a clear outcome and highlighted collision cell. |
| G05 | Collision precedes rewards. | Fatal moves grant no pickup/score; game over triggers once. |
| G06 | Score is sum of food reward points. | Length/time displayed separately; best score stored locally where available; storage failure never blocks play. |
| G07 | Filling all walkable cells is World Complete. | Near-capacity test completes without spawn hang; temporary lack of reachable food is not victory. |
| G08 | Bonuses are legible and reset. | Icon/name/countdown in HUD; expiry restores speed without catch-up burst; retry clears all effects. |

## 7. UI/UX direction

World first: rounded raised tiles, clear seams, soft shadows, readable paths and restrained miniature foliage/buildings. Connect islands with visible traversable bridges. Keep the kit's compact HUD, distinct head and simple overlays; move toward warm terrain with selective luminous rewards rather than pervasive neon.

Animation: foliage sway, water shimmer, distant accents, cell-anchored food bob, brief pickup feedback and smooth body/camera travel. Do not bob the whole playable grid. Reduced motion removes ambient pulses/bob, camera lag and shake while retaining movement needed to play. Head, body, regular food and bonuses differ in silhouette as well as color.

| ID | UI requirement | Acceptance criteria |
| --- | --- | --- |
| U01 | Start with Play, short controls and settings. | One primary action starts; no registration. |
| U02 | HUD shows score, length, effect/time and pause. | Does not cover head/forward route at 1280×720 or 1920×1080. If all food is offscreen, subtle direction cue; it does not promise a safe route. |
| U03 | Pause/settings expose Resume, Restart, sound and reduced motion. | Keyboard focus visible; restart during live run confirms loss. |
| U04 | Outcome shows cause, score/best, length/time, Retry Seed and New World. | One-action retry from outcome resets all run state. |
| U05 | Sprint 01 includes a visible animated prototype. | Demonstrate camera follow, junction turns, two terrain looks, food/bonuses, pause/outcome and reduced-motion comparison. Static moodboard alone fails. |
| U06 | Accessible cues. | UI text contrast target ≥4.5:1, non-color distinctions, no rapid flashing; sound never sole feedback. |

## 8. Quality and success gates

- Q01: Target 60 FPS and p95 frame time ≤20 ms in a five-minute 1080p run with 200 segments. Select and record reference device/browser in Sprint 01; these are targets, not measured claims.
- Q02: Input feedback ≤100 ms; turns on next simulation tick. New run ready within two seconds after assets load. Long stalls pause instead of executing unseen moves.
- Q03: Desktop Chrome/Edge versions recorded at release; 1280×720 minimum viewport. Menus usable at 200% zoom; clear viewport prompt if play area becomes too small.
- Q04: 100-seed generation validation, deterministic gameplay edge-case checks, and 30-minute stability session with pause/retry/focus loss/storage failure. No known crash, hidden collision or invalid start at release.
- Q05: Five new-player study: four collect first food and make intended turns within 60 seconds without coaching; four explain their death correctly; three voluntarily replay. Directional product evidence, not representative retention statistics.

## 9. Exactly three sprints

| Sprint | Deliverable | Exit gate |
| --- | --- | --- |
| [01](sprints/sprint_01/00_index.md) | PRD/research, UI/UX, animated prototype and technical decisions. | Reviewed requirements, U05 demonstrated, control/readability findings recorded; spec ready for implementation. PRD creation alone does not complete this sprint. |
| [02](sprints/sprint_02/00_index.md) | Generated world and basic movement sandbox. | W01–W05 and M01–M05 demonstrated; 100-seed checks pass; camera/occlusion reviewed. |
| [03](sprints/sprint_03/00_index.md) | Real playability and release candidate. | G01–G08, U01–U04/U06 and Q01–Q05 evidenced or residual limitations explicitly reviewed. |

Calendar dates and sprint lengths were not supplied. Each exit gates the next. Discipline owners: JANUS product acceptance, ARIA design/visual evidence, KALI implementation/test evidence. No separate agents are implied by this ownership plan.

## 10. Risks and Sprint 01 decisions

Finite versus infinite world and desktop platform need product review; defaults above keep scope bounded. Dense art can hide routes, so approve art density only after long-body movement demonstration. Isometric controls require comprehension testing. Loops/plazas reduce unfair maze traps but cannot eliminate player mistakes. Choose renderer through a visual/performance spike, not the kit's embedded prescription. Keep two bonuses until balancing evidence supports more.

Research: [03_game_research.md](03_game_research.md). Visual prototype, implementation, performance and playtest results remain pending.

