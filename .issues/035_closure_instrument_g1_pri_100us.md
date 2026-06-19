# Issue 035: Closure-Expansion Instrument G1 — PRI < 100µs (ahash + bit matrix)

**Status:** RESOLVED — G1 ✅ PASS. Bit-matrix + ahash + rolling-tag optimization
shipped; PRI measured at **19–26µs / 1K traces** across 4 back-to-back release
runs (2026-06-19). Was 4507µs. ~180× speedup, 4× safety margin under the 100µs
canonical target.

**Date opened:** 2026-06-19
**Date closed:** 2026-06-19 (commit `0e209c05` on develop)
**Origin:** README.md G1 row, Plan 290 GOAT gate (`.benchmarks/290_closure_instrument_goat.md`)
**Skill:** `.contexts/optimization.md` (Hot-path Rust optimization patterns)

## Problem

Plan 290 (Closure-Expansion Instrument) G1 gate **failed** the canonical target:

| Gate | Target | Measured (pre-fix) | Measured (post-fix) | Verdict |
|------|--------|--------------------|---------------------|---------|
| G1   | PRI < 100µs / 1K traces (hot-tier) | **4507µs** | **19–26µs** | ✅ PASS |

Root cause (per benchmark doc): `compute_pri` used
`std::collections::HashMap<PrimitiveKind, HashSet<u32>>` + a fresh
`HashSet<PrimitiveKind>` allocated per-PTG. std's SipHash costs ~200ns/insert;
for 1K traces × 8 nodes ≈ 8K hash inserts ≈ 4ms.

The data domain is **bounded** — `PrimitiveKind::to_u32()` maps the whole space
into `[0, 512)` (256 `UserDefined` + 256 `Composite`), and `task_family_id`
values are small in practice (5 in the bench corpus). We can replace hashing
with a dense bit matrix.

## Goal

Bring `compute_pri` under the canonical **100µs / 1K-trace** hot-tier target so
G1 turns green and the gate stops blocking promotion of `closure_instrument`.

## Plan of attack

Two complementary optimizations, both drawn from `.contexts/optimization.md`
("Data Structures" + "Don't: Allocate inside hot loops"):

1. **Bit matrix instead of nested HashMap.**
   Pre-pass: collect unique `task_family_id`s (≤ corpus-size) → assign each a
   dense bit index. Allocate one zero-init `Vec<u64>` of shape
   `512 primitives × ⌈n_families/64⌉ words`. Per-node hot path becomes:
   `bits[prim_idx * stride + word] |= 1u64 << bit` — one array write, no hash.
2. **Rolling-tag per-PTG dedup.**
   Replace per-PTG `HashSet<PrimitiveKind>` allocation with a stack
   `[u32; 512]` tag array + a wrapping generation counter. Touched entries are
   detected by `tag[i] == cur_gen`; the array is never cleared — amortized
   zero cost.
3. **ahash for the small outer maps.**
   The unique-family pre-pass and the final scores map use `ahash::AHashMap`
   (already a transitive dep via `hashbrown 0.14.5` — zero new top-level deps).
   SipHash → aHash knocks ~5× off the unavoidable hash work.

The `PriScores(pub HashMap<PrimitiveKind, f32>)` field type changes to
`PriScores(pub AHashMap<PrimitiveKind, f32>)`. Public consumers in the codebase
only call `.get()`, `.len()`, `.is_empty()` — all of which `AHashMap` provides.

## Tasks

- [x] Read existing code (`closure/metrics.rs`, `closure/mod.rs`, bench test).
- [x] Add `ahash` dep to `crates/katgpt-core/Cargo.toml` behind `closure_instrument`.
- [x] Rewrite `compute_pri` with bit-matrix + rolling tag.
- [x] Switch `PriScores` field to `AHashMap`.
- [x] Update benchmark doc `.benchmarks/290_closure_instrument_goat.md`.
- [x] Update README.md G1 row + Decision paragraph.
- [x] Run GOAT test — 9/9 PASS; G1 reports `✅ PASSED: PRI < 100µs canonical target`.

## Acceptance

- `cargo test --release --features closure_instrument --test bench_290_closure_instrument_goat`
  → **9/9 PASS** ✅ (measured 2026-06-19, 4 back-to-back release runs).
  G1 verdict reported as **✅ PASS** (19–26µs vs 100µs canonical target).
- All 9 GOAT tests pass (no correctness regression).
- All 9 `closure::metrics` unit tests pass (incl. 2 new edge-case tests for
  >64 families and composite-primitive indexing).
- All 6 wire-integration tests pass (API-compatible `PriScores` change).
- G1 number **measured and recorded**: 4507µs → 19–26µs (~180× speedup).

## What this does NOT fix

G4 (10K-trace snapshot 1.774MB vs 1MB target) remains the sole blocker for
promoting `closure_instrument` to default-on. G4 is structural — it stems from
the locked Phase 0 data model (`blake3_in: [u8; 32]` per node), not from any
implementation bug. Per Plan 290 T4.7 (G1–G4 must ALL pass), the feature stays
opt-in until G4 is also resolved.
