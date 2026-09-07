# Issue 738 — the wasm32 lanes compile what they NAME; nothing checks that what they name is the whole surface

**Status:** OPEN 2026-09-07 — instrument landed (`scripts/wasm32_surface_audit.py`), first measurement below: **9 NAMED · 15 UNRESOLVED · 0 UNCOVERED** over 17 repos / 196 files / 24 positive-cfg packages. Nothing is provably uncompiled; 15 packages need a per-package read.
**Owner:** this repo (the cross-repo audit family), with per-repo follow-ups.

## The axis

Issue 737's nine-repo audit closed six axes, and every one of them is about
*how* a lane compiles what it names — the triple, the `target_feature` arm,
the feature set, the arm reached via a dependent, the separate workspace,
`check`-vs-lints. seal-remake `.issues/010` T2 asked the next question up:

> is what the lane names **the whole surface**?

There, it was not. The root package carried a positive
`#[cfg(target_arch = "wasm32")]` block that no row built — and the block had
been **uncompilable since it was written**, because it called a function
declared `#[cfg(not(target_arch = "wasm32"))]`. A seventh axis, and the only
one that no amount of strengthening the existing rows can reach: a row cannot
notice a package it does not select.

## The instrument

`scripts/wasm32_surface_audit.py` — derived population (BOUNDARY.md + a `.git`
**directory**, so a throwaway worktree is not double-counted), tracked files
only, three buckets:

| bucket | meaning |
|---|---|
| **NAMED** | a wasm32 compile row selects the package by name (`-p`, or a literal `--manifest-path`), or is a bare row in a single-package repo |
| **UNRESOLVED** | a `--workspace` or **derived** row exists — whether it reaches this package is the separate-workspace axis (737 #5), undecidable statically |
| **UNCOVERED** | no row in the repo could reach it at all |

⛔ **The predicate is the POSITIVE cfg.** `not(target_arch = "wasm32")` is an
ordinary native-only guard and means the file has no browser code (riir-ai
`.issues/892` T4) — riir-game-sdk's 13 wasm32-mentioning files are *all*
negated, so its correct surface is the empty set. **Comment lines are excluded
too**: prose explaining a cfg is not a cfg, and in seal-remake the comment
documenting why a file has no wasm32 arm made that file read as browser code.

## First measurement (2026-09-07)

```
floors — 196 file(s) walked over 17 repo(s); 24 package(s) with positive wasm32 cfgs
9 NAMED · 15 UNRESOLVED · 0 UNCOVERED
```

**0 UNCOVERED is the honest headline** — every repo with any wasm32 lane has
at least one derived or wildcard row, so no package is provably uncompiled.
The 15 UNRESOLVED are the work: katgpt-rs (`katgpt-core`, `katgpt-types` —
78 positive sites), riir-ai (9 packages incl. a tracked `wgpu-hal` fork),
riir-dapps (`kat-service`), riir-deployer (`control-do`),
riir-mmorpg-examples (`riir-bevy`, `warm-tier-do`).

⛔ **UNRESOLVED is not clean** and must never be folded into either neighbour
— it is exactly where this class hides.

## Three instrument bugs, all caught by reading the output against known facts

Recorded because each produced a *confident* wrong number, and two of them
would have been filed as findings:

1. **A confident zero.** The first run walked **0 files** over 17 repos. The
   Python pattern `target_arch\s*=\s*"wasm32"` was handed to `git grep -E`,
   which is POSIX ERE and has no `\s` — it matched nothing. Caught only
   because the walk-size floor is printed next to the verdict; a bare
   "0 uncovered" would have read as a clean bill of health over nothing.
2. **17 false findings.** Everything not literally named was called
   UNCOVERED, including katgpt-rs — whose layer 2b *derives* its `-p` list
   on purpose, so a new wasm32-bearing crate joins by existing. The better
   design read as the worst result. Fixed by making DERIVED its own bucket.
3. **2 more false findings.** riir-dapps and riir-deployer select units with
   `--manifest-path "$unit/Cargo.toml"` over a derived list; matching only
   `-p $VAR` read those as bare rows, mis-credited the repo ROOT package, and
   reported their real Worker crates (`kat-service`, `control-do`) as
   UNCOVERED — the very crates whose lanes had just been landed by Issues 022
   and 004. One missing alternative in one regex.

The pattern across all three: **a classifier's bucket boundaries are the
finding**, and they are only testable against cases whose answer is already
known independently.

## Tasks

- [x] T0 — build the instrument; derive the population; classify into three
      buckets with the walk size printed underneath the verdict.
- [ ] T1 — resolve the 15 UNRESOLVED rows per package: for each, does the
      repo's derived/wildcard row actually select it? Where it does, the row
      is fine and the audit can be taught the mapping. Where it does not,
      either add an arm or record why the package cannot have one.
- [ ] T2 — decide whether the seal-remake / riir-game-sdk **membership pin**
      pattern (positive-cfg package set pinned by membership, walk size
      floored underneath) belongs in every repo, or whether this workspace-wide
      report supersedes seventeen local copies. One instrument beats seventeen
      hand-maintained pins if someone runs it; the pins fail closed and this
      does not. They are not exclusive — the pin is a gate, this is a report.
- [ ] T3 — `riir-ai`'s tracked `wgpu-hal` (11 positive sites) is a vendored
      fork. Confirm it is intended to be in the walk at all; a vendored
      upstream's wasm32 code is not this workspace's to gate.
