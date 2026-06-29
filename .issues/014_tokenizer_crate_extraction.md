# Issue 014 — Extract `src/tokenizer/` into standalone `katgpt-tokenizer` crate

**Date:** 2026-06-29
**Status:** Complete (committed katgpt-rs 8a70d2af + riir-ai d0144595)
**Severity:** DRY / decoupling (user rule: "DRY, Modular, Generic, Decouple")

## Problem

`katgpt-rs/src/tokenizer/` is a self-contained module (BPE + ToaST split-tree
+ ConvexTok LP optimizer + Double-Array Trie) gated by two features
(`toast_tokenizer`, `convex_tok`). It:

- Has **zero `crate::` deps** — only `super::` + `std` + `serde` (+ `good_lp`
  under `convex_tok`). It is already a clean leaf.
- Forces root `Cargo.toml` to retain `good_lp` solely for
  `tokenizer/convex_solver.rs` (comment says so explicitly).
- Forces `riir-ai/crates/riir-data` to depend on the **whole `katgpt-rs`**
  crate (100+ features, `wasmi`/`bevy_ecs`/`metal`/`reqwest` deps) when it
  only needs the ConvexTok training pipeline.

The repo already has 10 extracted leaf crates following this exact pattern
(`katgpt-types`, `katgpt-dec`, `katgpt-hla`, `katgpt-percepta`, …). The
tokenizer is the next natural candidate.

## Verdict

**Extract.** Strong case on all five GOAT dimensions:

- G1 correctness: pure move + re-export → bit-identical API surface.
- G2 perf: build-time win for `riir-data` (drops heavy deps); parallelizable
  workspace compile.
- G3 no-regression: default features untouched.
- G4 alloc-free: N/A (structural).
- G5 modelless: yes — tokenization is the modelless-est concern in the repo.

## Plan

### Move

`src/tokenizer/*.rs` → `crates/katgpt-tokenizer/src/*.rs` (13 files).

### New crate (`katgpt-tokenizer`)

- Leaf crate, no `katgpt-*` deps.
- `default = []`
- `toast_tokenizer = []`
- `convex_tok = ["dep:good_lp", "toast_tokenizer"]`
- `datrie_vocab = ["toast_tokenizer"]`
- Carries `good_lp` dep (moved out of root).

### Root `Cargo.toml`

- Add non-optional `katgpt-tokenizer = { path = "crates/katgpt-tokenizer" }`.
- Add to workspace members.
- Remove `good_lp` from root (used only by `convex_solver.rs`).
- Convert feature gates to forwarders:
  - `toast_tokenizer = ["katgpt-tokenizer/toast_tokenizer"]`
  - `convex_tok = ["katgpt-tokenizer/convex_tok"]`
  - `datrie_vocab = ["katgpt-tokenizer/datrie_vocab"]`

### Root `src/lib.rs`

- `pub mod tokenizer;` → `pub use katgpt_tokenizer as tokenizer;`
  (back-compat shim → zero churn for tests/examples/validator).

### `riir-ai/crates/riir-data/Cargo.toml`

- Swap `katgpt-rs` dep → `katgpt-tokenizer` for the convex training pipeline.

## Tasks

- [x] Write issue doc
- [x] Scaffold `crates/katgpt-tokenizer/` (Cargo.toml + lib.rs)
- [x] `git mv` 13 files `src/tokenizer/` → `crates/katgpt-tokenizer/src/`
- [x] Wire root Cargo.toml (members + dep + feature forwarders + drop good_lp)
- [x] Wire root src/lib.rs (re-export shim)
- [x] Update riir-data Cargo.toml
- [x] GOAT gate: `cargo check --all-features` (root) + `-p katgpt-tokenizer` + `--features convex_tok`
- [x] Commit on `develop`

## Impact

| Site | Change |
|------|--------|
| `katgpt-rs/tests/*.rs`, `examples/*.rs`, `src/validator/syn_pruner.rs` | None (re-export preserves `katgpt_rs::tokenizer::*` paths) |
| `riir-ai/crates/riir-data` | Lighter dep (`katgpt-tokenizer` vs whole `katgpt-rs`) |

## TL;DR

Extract `src/tokenizer/` → `crates/katgpt-tokenizer/`. Pure move + re-export
shim. Moves `good_lp` out of root, trims `riir-data`'s dep footprint, matches
the 10-crate extraction pattern. Near-zero churn.
