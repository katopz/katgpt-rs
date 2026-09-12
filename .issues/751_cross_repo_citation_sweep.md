# Issue 751 — the cross-repo citation sweep (18 repos katgpt-rs cannot see)

STATUS: RESOLVED — sweep landed, FP rate MEASURED (7/43 = 16%), 4 per-repo issues filed — 2026-09-12

## Why

`scripts/issue_citation_gate.py` (Issue 749) gates **this repo only**. Every
other sweep in the family has a workstation-wide half —
`docs_drift_sweep.py`, `numbering_drift_sweep.py`,
`percentile_drift_sweep.py`, `trap_sentinel_drift_sweep.py` — because the
defect class is not repo-local. This one is *especially* not: a citation's
whole problem is that its referent lives somewhere else.

## Tasks

- [x] T1 Re-measure post-alias-repair; spot-check >= 20 rows stratified
      across repos and record the measured FP rate.
- [x] T2 Audit the crate-name-implies-repo and zero-padded classes; decide
      per class whether it is a finding or a qualification form.
- [x] T3 `scripts/citation_drift_sweep.py` + `scripts/citation_drift_floors.txt`.
- [x] T4 Assert the re-stated quantities against the per-push gate.
- [x] T5 File per-repo issues from T1's verified rows.

## T1 — the number, and the error rate that makes it reportable

Population: **19 contract repos** (derived, BOUNDARY.md + `.git` dir) ·
**35 documents** (AGENTS.md + HISTORY.md; 3 repos carry no HISTORY.md) ·
**2,939 citations** walked.

| bucket | count | standing |
|---|---|---|
| **CROSS** — not local, no repo named, a sibling owns it | **254** | the finding |
| **IN-LOCAL-RANGE** — at/under this repo's own top allocation | **46** | ⛔ UNDECIDED, never folded into CROSS |
| **ORPHAN** — nobody in the workspace owns it | **0** | its own repair, kept as its own column |
| **AMBIGUOUS** — local AND sibling | **905** | the gate's stated blind spot, reported not gated |
| ⛔MISLEADING crate hints | **4** | worse than bare — a plausible address that is wrong |

**Measured false-positive rate: 7/43 = 16%**, a stratified manual read of 43
rows across **13** repos (the surrounding prose read with `sed -n`, not the
row text). The predecessor figure — 308 — was quoted with no error rate at
all and was ~50% contaminated in its first four rows (749's addendum). A
count without its error rate is not a measurement, and the four FP classes
are written into the sweep's docstring rather than silently healed:

| FP class | example |
|---|---|
| alias/abbreviation not in the alias table | `ndb` = riir-neuron-db (riir-dapps Bench 486); `mmorpg` = riir-mmorpg-examples (riir-deployer Issue 094) |
| attribution just OUTSIDE the 3-line window | 4 lines up (seal-remake Plan 191), or 2 lines DOWN — the window is backward-only (riir-game-sdk Issue 074) |
| intra-document back-reference | "see Issue 092 above", where THIS doc carries a heading for the foreign number (seal-remake) |
| a local number one above the local max | riir-ai cites its own `Plan 590` with `.plans` topping out at 588 |

**The IN-LOCAL-RANGE split is the substantive re-measurement**, and it is
what took 300 to 254. riir-chain's HISTORY narrates its own work as "Closed
as a class (Issue 039, same day)" — riir-chain allocated 1-38 and 41-46,
never 39, and six siblings have one. Calling that a cross-repo citation is a
misattribution; calling it clean would be worse. It is its own bucket.

⛔ The corpus MOVES: riir-mmorpg-examples' HISTORY.md gained 46 lines between
two runs an hour apart during this measurement. Line numbers in any row list
are a snapshot.

## T2 — both classes decided, with evidence

**(a) A crate name is NOT a qualification form — the rows stay findings.**
The workspace crate->repo map is unique (**156** package names, **ZERO**
collisions across 19 repos), so `riir-games-mmorpg::…` *could* address
riir-ai mechanically. Two things refute it as a qualifier: the map lives in
no document the reader has, and — measured — of the **45** CROSS rows
carrying a crate name in the window, **4 resolve to a repo that does NOT own
the cited number**. riir-game-sdk's `Issue 097` sits next to
`riir-games-mmorpg::sync_facades::avatar_sync` while 097 belongs to
riir-mmorpg-examples / riir-chain / seal-game-editor. Accepting crate names
would certify those four as clean, and they are the worst rows in the corpus.
Kept as findings, sub-labelled `crate-hint` / `⛔MISLEADING` to ORDER the
repair, never to excuse it.

**(b) Zero-padding is the SAME allocator namespace — not a finding.**
`Issue 006` and `Issue 6` are one number: both sides already `int()` (the
citation regex `\d{2,4}`, `allocated()`'s `^(\d+)_`). Measured over 5,839
allocated file names: 5,758 3-digit + 81 2-digit, and the **78** numbers
carrying BOTH widths are width-normalisation RENAMES of one document
(`.research/07_Screening_Absolute_Relevance.md` -> `.research/007_…`, same
title; `.research/00_Neuro-Symbolic…` -> `000_…`). Corpus usage is genuinely
mixed — **1,713 padded vs 1,221 unpadded** citations — so treating the forms
as distinct namespaces would misread 58% of it. Recorded blind spot, measured
at zero cost today: `\d{2,4}` cannot see a single-digit `Issue 6`, and there
are **0** such citations workspace-wide.

## T3 + T4 — the instrument

`scripts/citation_drift_sweep.py` (report; exit 0 clean / 1 drift / **2
instrument untrustworthy**) + `scripts/citation_drift_floors.txt`.
Population derived; the citation regex, alias table, allocation walk and
document list are the GATE's, imported not re-derived. Two floors:
`min_citations` per repo (~60% of measured — a regex regression takes it to
0 and every ceiling passes) and `min_repos` (a collapsed population makes
every citation read local-or-orphan). Ceilings are a RATCHET at measured,
per `numbering_drift_sweep`'s stance: 254 rows exist in repos this session
does not own, and the repair is theirs.

T4, two assertions, both probed:

- `min_repos` is restated in the sweep's floors and **asserted equal** to
  `issue_citation_floors.txt`'s (the `trap_sentinel_drift_sweep` vs
  `trap_sentinel_gate.POPULATION_FLOOR` precedent). `documents` is NOT
  restated — it is read from the gate's pins, because a quantity with one
  home cannot drift.
- the katgpt-rs row must **equal the gate's own run**, which the sweep
  executes and parses. The sweep PARTITIONS the gate's finding set into
  CROSS/IN-LOCAL-RANGE/ORPHAN, so the totals cannot disagree.

### Verified by revert probe — every arm FIRES

| probe | result |
|---|---|
| plant a bare `Issue 912` in AGENTS.md (NOT locally allocated — a valid probe per 749's addendum §3) | exit **1**, `katgpt-rs cross=1 -> riir-ai`, gate cross-check still AGREE at 141/1; AGENTS.md restored byte-identical |
| `min_citations` floor breached (katgpt-rs 84 -> 500) | exit **1**, "walk FLOOR breached: 140 < 500" |
| citation regex blinded (`icg.citations` -> `[]`) | exit **2** — caught by the SELFTEST, before any ceiling could pass |
| `min_repos` population floor (walk -> 1 repo) | exit **2** |
| `min_repos` restated != the gate's pin (15 -> 16) | exit **1**, "pin drift … same quantity, two files" |
| gate cross-check desynced (`gate_says` -> 999) | exit **1**, CROSS-CHECK row |
| ceiling ratchet (riir-auth 6 -> 5) | exit **1** |
| IN-LOCAL-RANGE split DELETED from `audit()` | exit **2** — the selftest's own bucket-boundary arm |

## T5 — per-repo issues filed (VERIFIED rows only)

| repo | issue | rows | referent |
|---|---|---|---|
| riir-game-sdk | `.issues/030` | 108 CROSS / 287 cites — **38%**, the workspace's highest density; 50 distinct numbers | 79 rows are the `Issue 458-531` riir-ai GM-dashboard/extraction arc |
| riir-viewbridge | `.issues/008` | 12 / 40 | `Plan 532` ×9 -> riir-ai; `Issue 701` -> katgpt-rs; `Issue 037` -> riir-clippy (exact title match) |
| riir-auth | `.issues/004` | 6 / 14 — all six verified | `Plan 488` ×5 -> riir-ai; `Issue 734` -> katgpt-rs |
| riir-dao | `.issues/004` | 5 / 43 | `Proposal 008` + `Proposal 010` ×3 -> **riir-clippy** — not a repo any reader would guess for a governance proposal |

Each filed issue also names the rows that are **classifier false positives**
in that repo, so the owner does not "fix" prose that is already correct.

Not filed: seal-remake (3 rows read, 2 FP + 1 undecided, 0 verified TP),
riir-clippy / riir-dapps / riir-deployer / riir-chain / riir-train /
riir-mmorpg-examples / riir-ai / katgpt-web — rows verified but at low
density or few enough that the sweep's own output is the report. Per the
non-goal below, the prose repair is the owning repo's call and the sweep
names every row on demand.

## Non-goal

Fixing 250-odd citations across 18 repos. The deliverable is a trustworthy
instrument plus per-repo issues; the prose repairs belong to whoever owns
each repo's AGENTS.md.
