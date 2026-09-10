# Sprint 01 — Prototype Review

Date: 2026-09-09 · Owner: JANUS

## Scope

Review the local animated design prototype against U05 and the visual direction in the PRD. This review does not certify the production game, procedural generation, the 100-seed suite or five-player study.

## Decisions carried forward

The user's instruction to proceed authorizes continuing the proposed desktop-browser, finite-world design exploration. Preserve a fixed camera angle, head following, square logical grid and decorative-only environmental motion. Keep the renderer decision provisional until measured against representative production assets.

## Evidence

PRD, supplied reference image and HTML, technical proposal and local preview workflow inspected. Browser observations will be recorded after the artifact is runnable.

## Remaining acceptance work

Five-player study, representative hardware/performance measurement and production renderer decision remain open. Sprint 01 is not complete merely because the prototype opens successfully.

## Browser review results

Observed in the local browser: animated garden route and follow camera, sand/cactus variant, round food score/growth feedback, golden reward, clock effect countdown, pause/resume, guided-tour outcome and restart, and reduced-motion switch state. Reviewed at 1280×720. Small viewport guard was also observed; subsequent compact-layout CSS was added but not exhaustively tested.

Corrections made after review: ground renders before actors, each segment has independent depth ordering, props receive supporting tiles, the head has a contrasting coral silhouette, collected pickups cannot reward repeatedly, only clock food slows movement, and the scripted tour outcome is explicitly illustrative. Hidden dialogs no longer remain exposed through opacity alone.

## Review disposition

APPROVE WITH CONDITIONS for an initial UI/UX direction review, not Sprint 01 completion. The artifact is available at `prototype/index.html`, served with `node preview-server.cjs` at http://127.0.0.1:4173.

GOOD: inspectable motion, clear head emphasis, lightweight HUD, two terrain palettes, reduced-motion control and understandable outcome presentation.

BAD: authored route rather than procedural world; simplified scenery; no five-player study or measured performance results. Steering preview is illustrative and not the final gameplay acceptance implementation. No complete start/settings flow, audio or production asset set yet.

UGLY → FIX: production self-collision/growth edge cases and full pickup spawning remain KALI's Sprint 03 work; renderer/asset feasibility and player control comprehension remain Sprint 01 acceptance work. ARIA owns further visual refinement after user feedback. Do not certify production playability from this tour.

User clarification: UI/UX first. React is required for the later application; Python only if needed. No production migration has started.
