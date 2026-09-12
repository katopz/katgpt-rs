# Issue 751 — the cross-repo citation sweep (18 repos katgpt-rs cannot see)

STATUS: OPEN — measured, instrument NOT yet trustworthy enough to pin

## Why

`scripts/issue_citation_gate.py` (Issue 749) gates **this repo only**. Every
other sweep in the family has a workstation-wide half —
`docs_drift_sweep.py`, `numbering_drift_sweep.py`,
`percentile_drift_sweep.py`, `trap_sentinel_drift_sweep.py` — because the
defect class is not repo-local. This one is *especially* not: a citation's
whole problem is that its referent lives somewhere else.

## Measured 2026-09-12 — and why the number is NOT usable yet

First run over 19 contract repos, 2,939 citations across AGENTS.md + HISTORY.md:

| repo | cites | flagged | ambiguous |
|---|---|---|---|
| riir-game-sdk | 287 | 122 | 99 |
| riir-mmorpg-examples | 536 | 60 | 441 |
| riir-clippy | 855 | 41 | 671 |
| riir-dapps | 174 | 21 | 124 |
| seal-remake | 66 | 17 | 45 |
| riir-viewbridge | 40 | 14 | 14 |
| riir-deployer | 36 | 7 | 20 |
| riir-auth | 14 | 6 | 1 |
| riir-chain / riir-dao | 109 / 43 | 5 / 5 | 100 / 29 |
| riir-train / riir-ai | 72 / 509 | 4 / 3 | 62 / 195 |
| katgpt-web / riir-kat / riir-neuron-db | 6 / 2 / 46 | 1 each | 4 / 0 / 33 |
| **katgpt-rs** (gated) | 140 | **0** | 107 |
| riir-esp32 / riir-shader / seal-game-editor | 4 / 0 / 0 | 0 | 4 / 0 / 0 |
| **TOTAL** | **2,939** | **308** | — |

⛔ **308 is a contaminated upper bound and must not be quoted as a finding
count.** Spot-checking the first four rows found **two clear false positives**
(repo named by its short form — "dapps Issue 027"). The alias repair landed in
749's addendum, but the post-repair number has **not** been re-measured or
re-spot-checked, and the remaining classes are unaudited:

- repo named by a **crate** rather than a repo (`riir-games-mmorpg::...`
  implies riir-ai) — seen, unresolved, arguably still a real finding
- **zero-padded** forms (`Issue 006`, `Proposal 010`) — parsed, but whether a
  padded number is the same allocator namespace is unverified
- the window's known false-negative (an unrelated repo name within 3 lines)
- repos whose numbering conventions differ from katgpt-rs's entirely

## Tasks

- [ ] T1 Re-measure post-alias-repair; **spot-check >= 20 rows** stratified
      across repos and record the measured FP rate. A classifier's bucket
      boundaries ARE the finding — this number is not reportable until its
      error rate is.
- [ ] T2 Audit the crate-name-implies-repo and zero-padded classes; decide
      per class whether it is a finding or a qualification form.
- [ ] T3 `scripts/citation_drift_sweep.py` in the sweep family — report,
      exit 0, population derived (BOUNDARY.md + `.git`), expectations in
      `scripts/citation_drift_floors.txt`. **Two floors, not one**: the
      finding ceiling needs the population that produced it AND the walk size
      under it (`min_repos`, `min_citations_scanned`).
- [ ] T4 Where a sweep re-states a quantity the per-push gate owns, ASSERT
      they agree rather than trusting them (the `trap_sentinel_drift_sweep`
      vs `trap_sentinel_gate.POPULATION_FLOOR` precedent).
- [ ] T5 Per-repo repair is the OWNING repo's call, not this one's — file
      per-repo issues from T1's verified rows, do not edit siblings' docs
      from here.

## Non-goal

Fixing 300-odd citations across 18 repos. The deliverable is a trustworthy
instrument plus per-repo issues; the prose repairs belong to whoever owns each
repo's AGENTS.md.
