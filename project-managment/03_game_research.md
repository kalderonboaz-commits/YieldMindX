# Comparable Games and Procedural World Research

Date: 2026-09-09 · Desk research by JANUS

## Scope and limits

Focused state-of-the-art scan of relevant released-game patterns and a procedural generation technique. This is not an exhaustive market survey or hands-on benchmark. Sources below are publisher/developer descriptions and the algorithm author's repository. Product recommendations are our interpretations; no source proves this exact combination is commercially validated.

## Comparables

| Reference | Sourced observation | Application to Snake3d (inference) |
| --- | --- | --- |
| [Snake Pass](https://store.steampowered.com/app/544330/Snake_Pass/) | Physics-based snake puzzle platforming, climbing and collectible exploration. | A snake can support an appealing world-exploration fantasy. Borrow environmental charm; avoid physics/climbing complexity within three sprints. |
| [SNKRX](https://store.steampowered.com/app/915310/SNKRX/) | Continuously moving snake of heroes, automatic attacks and class/build bonuses. | Temporary rewards can create run variation. Use two readable food bonuses; defer combat and build systems. |
| [Dorfromantik](https://store.steampowered.com/app/1455840/Dorfromantik/) | Tile placement builds growing landscapes with colorful biomes, procedural tile stacks and score objectives. | Modular scenery can make repeated tiles feel inviting. Snake3d needs square traversable routes; tile placement itself is not part of our loop. |
| [Unrailed!](https://store.steampowered.com/app/1016920/Unrailed/) | Track construction through procedural worlds, distinct biomes and an advancing train creates time pressure. | Readable routes and upcoming terrain matter when movement continues. Borrow forward visibility and biome variety; omit resource crafting and cooperation. |

## Procedural approach

[WaveFunctionCollapse, original repository](https://github.com/mxgmn/WaveFunctionCollapse) describes example-driven bitmap/tilemap generation. Local tile compatibility is useful for coherent scenery. It should not be treated as evidence that a generated Snake maze is connected, safe or survivable.

Recommendation: generate and validate a route graph first, including loops, plazas, safe spawn and matching bridges; then decorate it with modular terrain. A constrained tile solver is optional for decoration. Separate seeded gameplay randomness from ambient randomness so visual effects cannot change food or terrain outcomes. Keep seed plus generator version for reproducibility.

This recommendation is engineering judgment, not a selected library or stack. A finite map generated before play is the recommended MVP. Infinite streaming requires additional persistence and body-occupancy management and is deferred.

## Reference assessment

The uploaded HTML provides strong reusable component structure, modal flow, head emphasis and ambient motion. Code inspection shows static roads and a fixed camera origin; it does not fulfill procedural exploration. The image provides a richer miniature-world direction but is static, visually dense and shows separated islands. Connect islands, simplify route-adjacent scenery and design original animation.

## Product synthesis

The opportunity is familiar Snake tension inside an attractive scrolling tile diorama. Visual richness must preserve immediate collision readability. Bonuses should invite a detour without erasing the defining self-collision rule. Replayability should come first from seeds and body-management choices, before adding progression systems.

Sprint 01 must test camera readability and projected controls in an animated prototype. Sprint 02 must validate terrain invariants across seeds. Sprint 03 must test whether players understand deaths and choose to replay. These outcomes remain unverified.
