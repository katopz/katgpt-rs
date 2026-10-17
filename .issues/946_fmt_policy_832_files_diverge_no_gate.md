# Issue 946 — formatting policy: 832 tracked files disagree with bare rustfmt; no rustfmt.toml, no fmt gate anywhere (policy call)

**Status:** Open — owner policy call (2026-10-17, measured; nothing changed — detection only)

## The finding

`cargo fmt --all --check` exits 1 with **832 distinct files** carrying rustfmt diffs (measured
2026-10-17 on `develop` at `8e164b339`). There is **no `rustfmt.toml` / `.rustfmt.toml`** at
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
