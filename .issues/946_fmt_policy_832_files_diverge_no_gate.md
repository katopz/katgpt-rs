# Issue 946 — formatting policy: 832 tracked files disagree with bare rustfmt; no rustfmt.toml, no fmt gate anywhere (policy call)

**Status:** Open — owner policy call (2026-10-17, measured; nothing changed — detection only). **Audit landed same day (this section below the finding): the 832 was a file:line HEADER count, not files — true divergence 158 files (5.7% of tracked .rs); no config collapses it; option 1's cost ~5× lower than first costed. The slug carries the artifact number (kept — the number is the identity).**

## Audit (2026-10-17, session zed-m3-idle-1017b, M3) — the 832 is a header count; true divergence 158 files (5.7%)

Correction first: the "832 distinct files" was measured by `sort -u` over the full
`Diff in <file>:<line>:` headers — it counts unique file:line pairs (840 hunks → 832 unique
headers), not files. True count at the same `.rs` tree (byte-identical `8e164b339` →
`5fb1a210f`, verified `git diff --stat -- '*.rs'` empty): **158 distinct files of 2,770
tracked `.rs` (5.7%)**, 840 diff hunks, ~11.2k captured diff lines. Direct
`rustfmt --check` at defaults over all 2,770 tracked files agrees: 160 diverge
(±2 = edition-2024-for-all + module-recursion posture noise vs per-crate cargo fmt).

Class table over the 840 hunks (hunks / distinct files — files overlap across classes;
samples of the largest class are the SAME packing family: fn-param lists, struct literals
and let-bindings packed onto one line):

| class | hunks | files |
|---|---|---|
| other (samples = fn-params / struct-lits / let-joins packed one-line) | 267 | 93 |
| call-arg-stack (args exploded one-per-line; `fn_call_width` default 60) | 177 | 63 |
| macro-arg-stack (`println!`/`format!`/`assert!` family) | 100 | 44 |
| string-literal-shape | 91 | 55 |
| indent-whitespace only | 78 | 29 |
| rewrap-long-line (>100-col minus side) | 70 | 41 |
| import-layout | 41 | 33 |
| attr-placement | 15 | 9 |
| trailing-comma | 1 | 1 |

Whole-repo residual curve (unique diverging files / 2,770; `rustfmt --check`, edition 2024):

| config | diverging |
|---|---|
| defaults | 160 |
| `fn_call_width=100` | 2,044 |
| `use_small_heuristics=Max` | 2,469 |
| Max + `match_block_trailing_comma=true` | 2,485 |
| Max + `group_imports=StdExternalCrate` | 2,521 |

**The repo is rustfmt-default-styled.** 2,313 files diverge ONLY under the packed configs
(they are default-styled today); 156 diverge under both defaults and Max; 297 are clean
under both. There is no tuned `rustfmt.toml` that collapses the divergence: every setting
wide enough to keep the 160 drifted files untouched unpacks 2,000+ default-styled files.
The step-1 hope above — "if concentrated in 1–2 settings, a tuned rustfmt.toml may collapse
the divergence to near-zero" — is REFUTED by measurement.

What this does to the options:

- **Option 1's cost collapses ~5×**: the one-time fmt commit is 158 files / 840 hunks — a
  normal-sized commit, not a "whole-repo event". `.git-blame-ignore-revs` covers the cliff;
  the landing window is still required but is an hour of wall clock, not a coordinated event.
- **Option 2 stays viable** (fmt-on-touch reaches uniformity in ordinary churn time at 5.7%
  drifted), unchanged.
- **Options 3/4 unchanged**, but their main unknown is gone: the divergence is hand-drift
  (files edited without ever running fmt), not a house-style disagreement — the 2,612
  default-styled files are the style.

Posture: `cargo fmt --all --check` = the pinned 1.98.1 toolchain's rustfmt at defaults (no
rustfmt.toml exists); variants via direct `rustfmt --check --edition 2024`; read-only
detection, nothing reformatted, no gate, no config file written (non-goals honored).
Classifier + captures are one-off /tmp artifacts, not tracked; the section above is the
full reproducible method.

## The finding

`cargo fmt --all --check` exits 1 with **832 distinct files** carrying rustfmt diffs (measured
2026-10-17 on `develop` at `8e164b339`) — **[corrected by the audit below: 832 is the unique
file:line HEADER count; distinct files = 158 (5.7%)].** There is **no `rustfmt.toml` / `.rustfmt.toml`** at
the repo root and no fmt layer in any gate (`docs_gate.sh`, `full_gate.sh`, CI workflows all
format-blind). The earlier session note said "~30 files"; the real number is 832 — the
codebase style has drifted far enough from rustfmt defaults that a bare `cargo fmt` would be a
whole-repo event, not a cleanup.

## Why this matters

- **No gate = no invariant.** Any agent or contributor who DOES run `cargo fmt` (mechanical
  habit; most Rust repos gate on it) lands an 832-file diff that buries their actual change,
  poisons `git blame`, and collides with every concurrent session's in-flight hunks.
- **The divergence is load-bearing?** — unknown. 832 files of agreement with NON-default
  formatting suggests either (a) an old formatting pass under different defaults (the
  1.98.1 toolchain pin vs the edition/toolchain at authoring time), or (b) hand style.
  Nobody has audited WHICH classes differ; that audit is step 1 of any fix.
- Cross-repo note: sibling repos face the same question at smaller scales (this is the
  workspace's largest).

## Options (costs in blame/conflict/CI-minutes)

1. **Adopt + gate** — commit a `rustfmt.toml` (empty = defaults), run ONE dedicated
   whole-repo `cargo fmt` commit (`docs:`/`style:`), then add a fmt layer to `full_gate.sh`.
   Cost: one blame cliff (use `.git-blame-ignore-revs` — recommended companion), one
   coordinated landing window (no concurrent sessions mid-landing), zero recurring cost.
2. **Adopt per-touch** — gate fmt only on CHANGED files per PR/commit (fmt your hunks).
   Cost: no cliff, but per-commit noise and a slower drift to uniformity; the gate is
   per-diff-tooling, not one line in full_gate.
3. **Document no-fmt (status quo, explicit)** — write the policy down here + a `rustfmt.toml`
   that PINS the current style as closely as measured (requires the step-1 audit to pick
   settings), no gate. Cost: drift continues silently; every future audit re-finds this.
4. **Do nothing (implicit)** — the current state; this issue exists to make it a DECISION.

Recommendation: **option 1** (with `.git-blame-ignore-revs` + the landing-window discipline),
or option 2 if the blame cliff is unacceptable. Either way the audit of which rustfmt classes
the 832 files disagree on comes first — if the diff is concentrated in 1–2 settings (e.g.
`imports_granularity`, `match_block_trailing_comma`), a tuned `rustfmt.toml` may collapse the
divergence to near-zero without any mass reformat.

## Non-goals

- No reformat inside this issue — detection only (the finding/fil-first rule).
- No gate landing before the owner picks an option.
