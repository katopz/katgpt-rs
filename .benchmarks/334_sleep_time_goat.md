# Bench 334 (Sleep-Time): GOAT Gate — Mechanics + Cost Model + Zero-Alloc + Latency + BLAKE3

**Plan:** [katgpt-rs/.plans/334_sleep_time_query_anticipator_primitive.md](../.plans/334_sleep_time_query_anticipator_primitive.md)
**Research:** [katgpt-rs/.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md](../.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md)
**Source paper:** [arXiv:2504.13171](https://arxiv.org/abs/2504.13171) — Lin et al. (Letta/Berkeley) 2025, *Sleep-time Compute: Beyond Inference Scaling at Test-time*
**Status:** ✅ Phase 1 + Phase 2 synthetic gates PASS (2026-06-27). Feature ships as opt-in (`sleep_time_anticipation`). Promotion to default-on requires the riir-ai Plan 341 quality gates (G2/G3/G4 on a real predictability-labeled corpus) to clear.

> **Filename note:** katgpt-rs `plans/` and `benchmarks/` number independently; this file is named
> `334_sleep_time_goat.md` to disambiguate from the unrelated `334_sudoku_speculate_perf.md`
> (different plan — Sudoku Speculative-Solve, also numbered 334 in its own sequence).

---

## Gates

| Gate | Target | Result | Status |
|---|---|---|---|
| **G1** mechanics | anticipate/consume round-trip, blend correctness, predictability ∈ [0,1], determinism | 5/5 tests pass (`g1_anticipate_emits_populated_slots`, `g1_consume_is_deterministic`, `g1_consume_blend_is_smooth`, `g1_predictability_range_in_unit_interval`, `g1_consume_gate_finds_best_match`) | ✅ PASS |
| **G2** cost model correctness | amortization factor < 1 at paper ref point, `should_pre_compute` flips at break-even, total_cost monotone in e_gate, break_even_n solves the equation | 4/4 tests pass (`g2_amortization_factor_wins_at_paper_reference`, `g2_should_pre_compute_boundary`, `g2_total_cost_monotone_decreasing`, `g2_break_even_n_consistency`) | ✅ PASS |
| **G5** zero-alloc wake-time | 0 allocs / 0 deallocs per `consume()` call after warmup (and per `consume_gate()` call) | 0/0 over 1000 calls each, measured via `CountingAllocator` in a dedicated test binary | ✅ PASS |
| **G6** wake-time latency | `consume()` ≤ 200 ns/call at D=64, ≤ 100 ns/call at D=8 | **9.5 ns** at D=8,K=8 (10× margin); **57.6 ns** at D=64,K=8 (3.5× margin); `consume_gate` 4.8 ns at D=8,K=4 | ✅ PASS |
| **G7** BLAKE3 commitment | same inputs → same commitment; tamper detection on precomputed/predictability; dir-level commitment; audit verify hook | 4/4 tests pass (`g7_anticipate_commitment_deterministic`, `g7_tamper_detection`, `g7_verify_commitment_audit_hook`, `g7_direction_commitment_ulp_sensitive`) | ✅ PASS |
| **G2/G3/G4** quality (corpus) | predictability correlates with real query distributions; cross-player amortization holds at MMORPG scale; wake-time latency under real load | DEFERRED to riir-ai Plan 341 — requires a real predictability-labeled game corpus | ⏸ DEFERRED |

---

## What shipped (Phase 1)

Module: `crates/katgpt-core/src/sleep_time/` (5 files, ~1100 lines incl. inline tests)

| File | Symbols | Purpose |
|---|---|---|
| `types.rs` | `AnticipatedQueryDir<D>`, `AnticipatedSlot<D>`, `AnticipatedQuerySet<D,K>`, `commit_direction` | Core types — frozen direction vectors, c' artifact, BLAKE3 commitment |
| `predictability.rs` | `PredictabilityScorer<D>` trait, `DotPredictabilityScorer` | `p = sigmoid(α·dot(c,dir)+β)` default; trait lets riir-ai swap in curiosity-inversion scorer |
| `anticipator.rs` | `SleepTimeComputeOp<D>` trait, `SleepTimeScratch<D>`, `SleepTimeAnticipator<D,K,Op,Scorer>`, `IdentityFunctorOp` | Orchestrates per-direction sleep-time compute → emits c' artifact. `IdentityFunctorOp` is the synthetic-test default (`z_i = c + dir_i`) |
| `cost_model.rs` | `AmortizationCostModel` | Paper §5.3 cost model: `total_cost = sleep_cost + N·t·b_max·(1−E[gate])`; `should_pre_compute`, `amortization_factor`, `break_even_n` |
| `consume.rs` | `consume()`, `consume_gate()` | Wake-time hot path: dot-product match → sigmoid gate → smooth blend `gate·z + (1−gate)·fresh`. Zero-alloc. `consume_gate` is the decision-only path (skip fresh compute on high-gate queries) |

**Feature wiring:**
- `crates/katgpt-core/Cargo.toml`: `sleep_time_anticipation = []` (no deps — `blake3` is already non-optional in katgpt-core)
- Root `Cargo.toml`: `sleep_time_anticipation = ["katgpt-core/sleep_time_anticipation"]` (passthrough)
- `crates/katgpt-core/src/lib.rs`: `#[cfg(feature = "sleep_time_anticipation")] pub mod sleep_time;` + re-exports

**Test wiring (Cargo.toml `[[test]]` / `[[bench]]` entries):**
- `sleep_time_goat` — G1/G2/G7 GOAT gate tests (13 tests, regular test binary)
- `sleep_time_alloc_check` — G5 zero-alloc (separate binary, CountingAllocator — single test function so checks run serially against the shared global allocator)
- `sleep_time_consume_bench` — G6 latency (criterion bench, harness=false)

---

## The `DEFAULT_K` collision lesson

When running `cargo check --all-features`, the build failed with `E0252: the name 'DEFAULT_K' is defined multiple times` — `cgsp::DEFAULT_K` (line 35 of lib.rs) collided with `sleep_time::DEFAULT_K`. This is exactly the cross-feature conflict class the AGENTS.md `merkle_root` lesson warns about. Fix: renamed to `SLEEP_TIME_DEFAULT_K` (prefixed). **Single-feature checks (`cargo check --features sleep_time_anticipation`) did not catch this** — only `--all-features` did. The CI guard `./scripts/ci_feature_guard.sh` runs both, so this would have been caught in CI.

---

## Test inventory

### Inline unit tests (31 total, all PASS)

| Module | Tests |
|---|---|
| `types.rs` | 7 (direction commitment determinism, ULP sensitivity, verify roundtrip, simd dot, version preservation, slot set determinism, tamper detection) |
| `predictability.rs` | 4 (unit interval, alignment monotonicity, determinism, beta threshold) |
| `cost_model.rs` | 9 (zero/full hit rate, monotonicity, break-even boundary, amortization factor, infinity at N=0, break_even_n consistency, infinity at E[gate]=0, clamping) |
| `anticipator.rs` | 5 (K-slot emission, commitment stability, context-change commitment, predictability range, identity op budget-agnostic) |
| `consume.rs` | 6 (precomputed-when-predictable, fresh-when-unpredictable, determinism, gate best-match, gate range, smooth blend at gate=0.5) |

### GOAT gate tests (`tests/sleep_time_goat.rs` — 13 total, all PASS)

G1 × 5, G2 × 4, G7 × 4 — see gate table above.

### Alloc check (`tests/sleep_time_alloc_check.rs` — 1 test, PASS)

`g5_zero_alloc_after_warmup_both_paths` — serial check of both `consume()` and `consume_gate()` against the shared `CountingAllocator`. 200 warmup + 1000 measured calls per path. Result: 0/0 allocs/deallocs on both paths.

### Latency bench (`benches/sleep_time_consume_bench.rs`)

Run: `cargo bench -p katgpt-core --features sleep_time_anticipation --bench sleep_time_consume_bench`

| Path | Config | Median | Target | Margin |
|---|---|---|---|---|
| `consume_gate` | D=8, K=4 (ambient NPC) | 4.8 ns | ≤ 100 ns | 21× |
| `consume` | D=8, K=4 (ambient NPC) | (similar) | ≤ 100 ns | ~10× |
| `consume` | D=8, K=8 (shopkeeper NPC) | 9.5 ns | ≤ 100 ns | 10× |
| `consume` | D=64, K=8 (style_weights scale) | 57.6 ns | ≤ 200 ns | 3.5× |
| `consume_gate` | D=64, K=8 (style_weights scale) | 50.8 ns | ≤ 200 ns | 4× |

Hardware: macOS aarch64 (Apple Silicon). SIMD path: NEON.

---

## The G5 zero-alloc lesson (test parallelism)

Initial G5 attempt used two separate `#[test]` functions (`g5_consume_zero_alloc_after_warmup` and `g5_consume_gate_zero_alloc_after_warmup`). The `consume()` test reported 7 allocs/1000 calls — but `consume_gate()` reported 0. Root cause: tests in the same binary run in parallel by default and share the global `CountingAllocator`. The allocations came from the parallel test's measurement window bleeding into the other test's window.

Fix: merged both checks into a single `#[test]` function (`g5_zero_alloc_after_warmup_both_paths`), which runs serially by construction. This matches the `analytic_lattice_alloc_check.rs` pattern (whose header explicitly notes "Single function = serial by construction").

**Lesson for future G5 gates:** when multiple zero-alloc checks share a `CountingAllocator`, put them all in ONE `#[test]` function. Separate `#[test]` functions run in parallel and corrupt each other's alloc deltas.

---

## Validation commands (reproducer)

```bash
# Phase 1 inline unit tests
cargo test -p katgpt-core --features sleep_time_anticipation --lib sleep_time::
# → 31 passed; 0 failed

# Phase 2 G1/G2/G7 GOAT gate
cargo test -p katgpt-core --features sleep_time_anticipation --test sleep_time_goat
# → 13 passed; 0 failed

# Phase 2 G5 zero-alloc (separate binary)
cargo test -p katgpt-core --features sleep_time_anticipation --test sleep_time_alloc_check
# → 1 passed; 0 failed

# Phase 2 G6 latency
cargo bench -p katgpt-core --features sleep_time_anticipation --bench sleep_time_consume_bench
# → D8_K8 consume 9.5ns, D64_K8 consume 57.6ns

# All-features compile (the DEFAULT_K collision catch)
cargo check --all-features -p katgpt-core --lib
# → Finished (0 warnings, 0 errors)

# Default-features compile (zero impact when feature off)
cargo check --lib
# → Finished
```

---

## Next steps

- **riir-ai Plan 341 Phase 1** is now UNBLOCKED — the open math primitives ship under `katgpt_core::sleep_time::*`. riir-ai can define `HlaSleepTimeOp` (uses `latent_functor::extract_functor_into`), `CuriosityInversionScorer` (uses KARC forecaster from Plan 308), and the per-NPC-type direction-vector catalogs.
- **Promotion to default-on** waits for Plan 341 G2/G3/G4 to clear on a real predictability-labeled game corpus. The synthetic gates here prove the math is correct, not that the predictability measure is useful.
- **Phase 3 (examples)** and **Phase 4 (docs)** are not blocking — they can land alongside or after the riir-ai integration.
