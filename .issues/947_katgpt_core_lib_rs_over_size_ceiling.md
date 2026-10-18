# Issue 947 — katgpt-core lib.rs is 3,894 lines, over the 3,200 ceiling (58% is per-module provenance comments)

**Status:** DONE pending close — T1+T2 landed 2026-10-18 (see ticked notes below); owner closes.

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

- [x] T1 move the provenance comments to the module headers (merge with an existing `//!`,
  never duplicate it); one commit, docs-only. **LANDED 2026-10-18:** 252 blocks / 2,036 comment
  lines moved from lib.rs into 222 module files' `//!` headers (existing header first, provenance
  appended after a `//!` separator; every word preserved, only the `//`/`///` → `//!` marker
  changed). Spun-out re-exports keep ≤2-line notes (12 condensed: hla, assign, dec-adjacent
  simd/types/micro_belief/personality/sleep re-exports + the BoM G2 / Issue 578 / Issue 582
  blocks). Region labels kept in lib.rs (Phase 10/12 absorption headers; the Phase-10 block
  split — its Issue-741 alloc-gate story moved to alloc.rs). Inline `pub mod sense` block kept.
  lib.rs: **3,894 → 1,853 lines** (the ≈1,500 estimate assumed the fn doc comments also move;
  those stay — they document the 5 `pub fn`s, not a module). Two rustfmt reorder hazards
  handled: 62 blank-line barriers keep the original mod/pub use order under rustfmt's
  default-on `reorder_modules`/`reorder_imports`, and `dllm/mod.rs` + `exact_mass_admit.rs`
  kept their CRLF endings (insertion-only diffs). Zero public-surface change: the non-comment
  line sequence of lib.rs is byte-identical to HEAD (cfg 431=431, pub mod 260=260, pub use
  204=204, pub fn 5=5), and the multiset of moved comment words equals the multiset inserted
  module-side (script-verified).
- [x] T2 verify: the full-gate clippy command, `cargo doc -p katgpt-core --all-features`
  builds, `scripts/docs_gate.sh` green, `scripts/fmt_gate.py` green. **ALL GREEN 2026-10-18**
  (CARGO_TARGET_DIR=/tmp/katgpt947 — sibling agents' cargo was live): (1) full-gate clippy
  `--workspace --all-targets --all-features --keep-going -D …` **exit 0, 0 error lines**;
  (2) `cargo doc -p katgpt-core --all-features` — katgpt-core itself builds clean (`--no-deps`
  exit 0; 178 doc warnings vs **250 at HEAD** — the move removed 72, added none); the with-deps
  run exits 101 on a **pre-existing** rustdoc error in `katgpt-assign/src/types.rs:92`
  ("unresolved link to `pinned`" — math prose brackets under that crate's own deny), a file
  this change never touched (byte-identical at HEAD d8f80e8dc); (3) `docs_gate.sh` **PASSED
  35/35**; (4) `fmt_gate.py` **PASSED** (2,771 files, 0 pinned).

## Why deferred

The cost is collisions, not the work: every session landing a feature edits this file, and
T1 touches up to ~260 module headers. Land it when `git log --since='24 hours ago' -- crates/katgpt-core/src/lib.rs`
is empty and no live claim names a katgpt-core feature landing. No perf or behaviour gain;
the value is readability, and keeping the next additions from pushing the file further over.
