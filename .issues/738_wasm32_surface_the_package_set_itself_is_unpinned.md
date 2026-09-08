# Issue 738 — the wasm32 lanes compile what they NAME; nothing checks that what they name is the whole surface

**Status:** RESOLVED 2026-09-08 — T0–T3 landed. T1: 14 of 15 UNRESOLVED packages resolved on static evidence (two resolver shapes, four-way canaried; the 15th — riir-ai `riir-examples` — measured UNCOMPILABLE for wasm32 and filed as riir-ai `.issues/894`, which is a finding, not a folding). T1 also surfaced one real lane gap: riir-mmorpg-examples' `warm-tier-do` (a standalone wasm32 Worker workspace nothing compiled) — lane landed same day (`riir-mmorpg-examples` `b23dc52`, layer 2e, canaried). T3: vendored upstream excluded from the walk (riir-ai's tracked `wgpu-hal` fork lives under `vendor/`, which its own lane already excluded). Standing headline: **22 NAMED · 1 UNRESOLVED (riir-examples → 894) · 0 UNCOVERED** over 23 packages / 190 files / 17 repos.
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
- [x] T1 — **RESOLVED 14/15, and the 15th is itself the finding.** The
      resolver (`resolve_unresolved` in the audit) upgrades a derived-row
      package to ✓ derived only on static evidence from a ROW-BEARING lane
      file (a file that already carries a wasm32 cargo row — a DEPLOY path
      mentioning a unit can never upgrade a package; `build-*.sh` is not a
      gate is this family's founding lesson):

      - **Shape A** — the lane carries the workspace's standard derivation
        formula (`crates/([^/]*)/src/` → `-p <name>`, both sed spellings);
        any package with a positive site in that scope is selected by
        construction. Root-`src/` sites deliberately NOT extended (katgpt-rs
        appends the root, riir-ai/mmorpg do not, and katgpt-rs's root is
        already named by its bare row).
      - **Shape B** — a membership pin names a unit dir whose manifest IS
        the package (dapps `kat-service`, deployer `control-do`, mmorpg
        `warm-tier-do`). The manifest must EXIST: `package_of`'s walk probes
        `repo/Cargo.toml` as a fallback, so a phantom unit token resolved to
        the repo ROOT package — caught by canary 3, not by reading.

      Four canaries: both sed spellings detected + non-formula rejected;
      katgpt-rs resolves exactly {katgpt-core, katgpt-types} with
      already-named packages untouched; a phantom unit token upgrades
      nothing; vendored wgpu-hal excluded. An earlier draft had a THIRD rule
      (package name as a bare token on a non-comment lane line) — deleted:
      it upgraded riir-ai's `riir-examples` off a NATIVE `-p` row, the exact
      confident-wrong this issue's bug log warns about.

      **The 15th:** riir-ai `riir-examples` — no `src/` scope site, and the
      browser examples measured UNCOMPILABLE for wasm32 (`uuid` missing the
      `js` randomness feature; the same defect 892 fixed for `riir-games`,
      one crate over). Filed as **riir-ai `.issues/894`** — an unresolved
      package that cannot compile is a finding, and it stays ? UNRESOLVED
      here until 894 lands.

      **And one real lane gap fell out of the per-package reads:**
      riir-mmorpg-examples' `warm-tier-do` — a STANDALONE wasm32 Worker
      workspace whose 2c/2d lanes provably could not reach it — got its lane
      that day (`b23dc52`, guard layer 2e, derived + membership-pinned,
      canaried, measured clean 36.9s cold). The audit finding it as
      UNRESOLVED is what surfaced the gap; the resolved bucket is how the
      next one gets noticed.
- [x] T2 — **DECIDED: both, with the division the issue itself sketched.**
      The local membership pins stay where they landed (riir-ai 1.22,
      riir-game-sdk 027, riir-dapps 022, riir-deployer 004, seal-remake 010,
      riir-mmorpg-examples 2c/2e) — they are GATES: fail-closed, cheap,
      per-repo, and they red the moment their set changes. This audit stays
      a REPORT (exit 0): it is the drift net for the repos and surfaces the
      pins cannot see (standalone workspaces, new packages, new repos), run
      on demand from katgpt-rs. No seventeenth hand-maintained copy of
      either — a repo earns a pin when it gains a wasm32 surface, and this
      report is what notices.
- [x] T3 — **CONFIRMED: vendored upstream is out of the walk.** riir-ai's
      `wgpu-hal-30.0.0` (11 positive sites) is the tracked fork under
      `vendor/` (Plan 536 / Issue 663, exposed for the raw Metal buffer),
      and riir-ai's OWN lane already excludes `vendor/` from its derivation
      (`grep -v '^vendor/'`). The audit now excludes it too — a vendored
      upstream's wasm32 code is not this workspace's to gate, and counting
      it kept a permanent UNRESOLVED row alive for code nobody here will
      ever lint. Walk floor 196 → 190 files; package floor 24 → 23.
