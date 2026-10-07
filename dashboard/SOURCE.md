# Dashboard source record

## Origin

| Field | Value |
|---|---|
| Repository | https://github.com/kalderonboaz-commits/YieldMindX |
| Branch | `linoy` |
| Commit | `c9065c5d9b14012e7762555f54ba158439a86612` |
| Commit message | "Add interactive Good/Bad parameter box plots" |
| Commit date | 2026-09-16T11:43:28Z |
| Commit author | kalderonboaz-commits |
| Source path in repo | `output/` (preserved here as `dashboard/output/`) |

Fetched directly from `raw.githubusercontent.com` **pinned to the commit SHA**
(not the mutable branch name), so these bytes cannot drift if `linoy` moves
later. Not cloned from the working-tree checkout, to avoid git's Windows
line-ending (CRLF) normalization changing file bytes.

## Why this commit

Verified against the live deployment at
`https://yieldmindx-revision-5.kalderonboaz.chatgpt.site/output/YieldMinx_dashboard_rev5_github_template_demo`
in a prior step of this engagement: the deployed iframe content
(`yieldmindx-five-section-workspace-chart-recovery.html`) is byte-identical
to this commit's version of that file, except for a Cloudflare-injected bot
script and one blank-line whitespace difference (both confirmed to be
hosting-layer artifacts, not source differences). The `boaz` branch's
version of the same file uses an older bar-chart/scatter-plot GVB
visualization and does NOT match the deployed page; `linoy`'s commit (this
one) added the box-plot visualization that the deployment actually serves.

## Files copied

| File | SHA-256 | Size (bytes) |
|---|---|---|
| `output/YieldMinx_dashboard_rev5.html` | `1aead6f5fe82593a1f4f1835b99645a6b19cf816e2be2372722a8b410eb4fc0e` | 4,080 |
| `output/yieldmindx-five-section-workspace-chart-recovery.html` | `056f1a67920cb0991c472d517b14a190a1e3201177891033e297ea182982fe81` | 41,275 |
| `output/yieldmindx-commonality-parameters.js` | `4cbba36e1e252c0cd1076c80024e6a8809650d7d46c69f8362101f200346382a` | 25,567,590 |

All three hashes were recomputed after copying into this project and matched
the pinned-commit fetch exactly (see verification below).

## Known difference vs. the live deployment (not fixed here, reported for awareness)

The live deployment's wrapper references `yieldmindx-rev5-team-results.js`,
a file that does not exist anywhere in this repository (confirmed by GitHub
code search across all branches). The live deployment's copy of
`yieldmindx-commonality-parameters.js` also returns HTTP 404, so the GVB
Autopilot section is likely non-functional on the live site. Neither of
these affects this local copy, which uses the committed, working files.

## Entry point

Open `dashboard/output/YieldMinx_dashboard_rev5.html` directly in a browser
(double-click, or `file://` URL). It embeds
`yieldmindx-five-section-workspace-chart-recovery.html` via an iframe, which
in turn loads `yieldmindx-commonality-parameters.js` for the GVB Autopilot
section (section 5) -- all three files must stay together in the same
`output/` folder for the relative paths to resolve.

## Not done, per instructions

No model was connected, retrained, or redesigned. Nothing was committed or
pushed -- this directory is a local, static copy only. Existing project
files outside `dashboard/` were not touched.
