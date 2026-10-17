# Issue 947 — katgpt-core lib.rs is 3,894 lines, over the 3,200 ceiling (58% is per-module provenance comments)

**Status:** Open — refactor, deferred to a quiet window (measured 2026-10-17; nothing changed — filing only).

## Finding

`crates/katgpt-core/src/lib.rs` = **3,894 lines** (the global rule: keep `.rs` under 2,048
where possible, 3,200 the ceiling). Composition at `99a0d90ed`:

| kind | lines |
|---|---|
| comment (`//`, `///`, `//!`) | 2260 |
| `#[cfg]` attributes | 431 |
| `pub mod` / `mod` | 260 |
| `pub use` | 204 |
| blank | 251 |

The bulk is provenance prose ABOVE each `mod`/`pub use` line (plan/issue number, why the
module exists, spin-out history). It is the most-churned file in the crate — **14 commits
in the 7 days before filing** — because every new feature lands a `#[cfg] pub mod` row here.

Other tracked `.rs` over 3,200: none. Over 2,048: 20+ (`katgpt-speculative/src/weaver.rs`
3,009, test files ~2.1–2.9k) — not in scope here.

## Fix (mechanical, one rule)

Move each module's provenance comment into that module's own `//!` header (where rustdoc
renders it and where a reader of the module looks), leaving one line per `mod` here at most.
Spun-out re-exports (`pub use katgpt_hla as hla;` etc.) keep a ≤2-line note. Expected
result ≈ 1,500 lines, with no change to the public surface.

- [ ] T1 move the provenance comments to the module headers (merge with an existing `//!`,
  never duplicate it); one commit, docs-only.
- [ ] T2 verify: the full-gate clippy command, `cargo doc -p katgpt-core --all-features`
  builds, `scripts/docs_gate.sh` green, `scripts/fmt_gate.py` green.

## Why deferred

The cost is collisions, not the work: every session landing a feature edits this file, and
T1 touches up to ~260 module headers. Land it when `git log --since='24 hours ago' -- crates/katgpt-core/src/lib.rs`
is empty and no live claim names a katgpt-core feature landing. No perf or behaviour gain;
the value is readability, and keeping the next additions from pushing the file further over.
