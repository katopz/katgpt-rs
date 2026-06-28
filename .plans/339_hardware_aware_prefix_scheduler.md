# Plan 339: Hardware-Aware Prefix Scheduler — Multi-Request Verification Budget Allocator

**Status:** Phase 1–6 COMPLETE — feature ships opt-in (`hardware_aware_scheduler`). GOAT gate (G1–G5) PASS on synthetic multi-request workload; stays opt-in per AGENTS.md until a real multi-request caller exercises it (Issue 003 risk: katgpt-rs default is single-request, so the gate is vacuous without a multi-request batch caller).
**Date:** 2026-06-28
**Research:** `.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md`
**Source paper:** DSpark (DeepSeek-AI, 2026) §3.2.2, Algorithm 1, Appendix A
**Target:** `src/speculative/prefix_scheduler.rs` (new module) + Cargo feature `hardware_aware_scheduler`
**Issue:** #003

---

## Motivation

katgpt-rs has per-request verification budget selectors (`caddtree_budget.rs`, `budget.rs`) but no
**multi-request global** verification budget allocator. When multiple spec-decode requests share a
target model forward pass (batch serving, or crowd-scale NPC cognition in riir-ai), static
per-request block lengths waste target compute on low-survival suffix tokens while starving
high-survival tokens in other requests. DSpark §3.2.2 formulates this as global throughput
maximization `Θ = τ · SPS(B)` solved greedily with a non-anticipating early-stop that preserves
the lossless distribution guarantee (Appendix A correctness proof).

## Architecture

```mermaid
graph TD
    R[R active requests] --> Probs[Per-position survival probs a_r_j]
    Probs --> Sort[Global sort by a_r_j desc]
    Curve[SPS curve LUT] --> Lookup[O(1) SPS lookup]
    Sort --> Greedy[Greedy admit]
    Lookup --> Greedy
    Greedy --> Stop{Θ ≤ Θ_best?}
    Stop -->|no| Greedy
    Stop -->|yes - non-anticipating| Alloc[Per-request prefix lengths ℓ*_1..ℓ*_R]
```

## Scope

**IS:**
- The generic scheduler primitive (global sort + greedy + cost-curve lookup + non-anticipating early-stop).
- A profiled `SpsCurve` abstraction (load once at init, linear-interpolation LUT).
- A single-request-isolated correctness test proving R=1 matches LeviathanVerifier semantics (no
  selection bias from the early-stop).

**IS NOT:**
- A multi-request batch execution engine (katgpt-rs is single-request by default).
- The confidence head producing `c_k` (reuse `AcceptanceForecast`, Bebop Plan 243).
- The semi-autoregressive drafter (training → riir-train).
- Sequential Temperature Scaling (separate concern).

## Tasks

### Phase 1 — Core scheduler primitive

- [x] T1 `src/speculative/prefix_scheduler.rs`: `HardwareAwarePrefixScheduler` struct +
  `schedule(&self, survival_probs: &[&[f32]]) -> Vec<usize>`. The schedule output is per-request
  prefix length ℓ*_r. Caller-owned output (`Vec<usize>`); the scheduler is otherwise zero-alloc.
- [x] T2 Helper `cumprod` (cumulative product) on `&[f32]` for `a_{r,j} = Π_{i≤j} c_{r,i}` when
  callers pass raw `c_k`. Mirrors `cumprodsum_scalar(a, x=0, ...)` from `src/cumprodsum.rs`.
- [x] T3 Candidate materialization: for each `(r, j)`, build `(a_{r,j}, r, j)` tuples into a
  caller-supplied scratch buffer (`&mut [(f32, usize, usize); total_tokens]` — generic over total
  token count, no fixed cap). Sorted descending by `a_{r,j}` via `sort_unstable_by`.

### Phase 2 — SPS curve abstraction

- [x] T4 `SpsCurve` struct: `from_profile(samples: &[(usize, f32)]) -> Self`, validating the
  profile is non-empty and monotone-validated (clamped, not strictly monotonic — DSpark §5.2 notes
  real hardware has jagged step-wise curves).
- [x] T5 `steps_per_second(batch_size: usize) -> f32`: O(log n) binary search for the bracketing
  samples + linear interpolation; clamp at ends (extrapolation forbidden).
- [x] T6 Profile storage as `Box<[(usize, f32)]>` (one heap allocation at construction, zero
  thereafter — the LUT is read-only).

### Phase 3 — Non-anticipating early-stop (correctness theorem)

- [x] T7 Implement the DSpark Appendix A early-stop: break the greedy loop when
  `Θ = τ · SPS(B)` first drops to/below `Θ_best`. Document the Appendix A counterexample in a
  doc-comment (vocab {A,B}, p_t=(0.7,0.3), p_d=(0.5,0.5) → output must be (0.7,0.3), not (0.85,0.15)).
- [x] T8 The early-stop logic is **a correctness theorem, not a heuristic** — removing it would
  leak future-token info into the current-token admission decision, breaking distribution
  preservation. Doc-comment cites the proof; no feature flag may disable it.

### Phase 4 — Feature flag wiring

- [x] T9 `Cargo.toml`: add `hardware_aware_scheduler = []` (empty, default-OFF). The scheduler
  depends on no other feature — it is a pure-Rust algorithmic primitive over `f32` survival probs.
- [x] T10 `src/speculative/mod.rs`: add `#[cfg(feature = "hardware_aware_scheduler")] pub mod
  prefix_scheduler;` + re-export `HardwareAwarePrefixScheduler`, `SpsCurve`.

### Phase 5 — Tests

- [x] T11 Single-request correctness (R=1): `schedule` output must reduce to "verify the full
  block" or "skip" depending on SPS shape — never bias the accepted distribution. Synthetic SPS
  curve. Hard-asserts the Appendix A counterexample: a deterministic transform of a uniform-drafter
  signal yields the target distribution bit-identically.
- [x] T12 Multi-request throughput (R=4): synthetic SPS curve (monotone-decreasing with a cliff).
  Verify the scheduler allocates longer prefixes to high-survival requests, shorter to low-survival,
  and that `Θ_scheduler ≥ Θ_uniform`.
- [x] T13 Empty/degenerate inputs: empty `survival_probs`, single-request single-position,
  all-zero survival probs, single-sample SPS curve — all must return without panic.

### Phase 6 — GOAT gate (`tests/prefix_scheduler_goat.rs`)

- [x] T14 G1 (single-request correctness): R=1 scheduler output preserves the LeviathanVerifier
  accepted distribution (no selection bias).
- [x] T15 G2 (multi-request throughput): R=4, `Θ_scheduler ≥ Θ_uniform * 1.05` (≥5% throughput
  gain) on the cliff SPS curve.
- [x] T16 G3 (no regression): `cargo test -p katgpt-core --lib` passes (no other feature affected
  — `hardware_aware_scheduler` has no feature deps).
- [x] T17 G4 (zero-alloc on hot path): the `schedule` method itself performs no heap allocations
  after warm-up (the `Vec<usize>` output is the documented caller-owned return — verified by
  `#[track_allocations]`-style manual counting via a custom allocator in the test).
- [x] T18 G5 (sigmoid only, no softmax): the early-stop uses raw `τ · SPS(B)` (no normalization
  step); a `grep -r "softmax\|Softmax"` over the new file returns zero hits.

### Phase 7 — Promotion (DEFERRED)

- [-] T19 If G1+G2 pass on a **real multi-request batch caller** (riir-ai crowd-NPC cognition or a
  katgpt-rs batch server), promote `hardware_aware_scheduler` to default-on. **DEFERRED**: katgpt-rs
  default is single-request; the synthetic G2 gate is vacuous without a multi-request caller.
  Stays opt-in until that landing. This matches the issue's own risk #1.

## Risks (carried from Issue 003)

- **Single-request default.** The scheduler only helps when the caller batches multiple spec-decode
  requests into one target forward pass. The synthetic G2 gate is necessary but not sufficient for
  promotion.
- **SPS curve shape.** DSpark §5.2 notes real hardware has jagged step-wise curves. The early-stop
  assumes unimodality for global optimality; the production DSpark variant uses async 2-step-prior
  prediction + removing the early-stop. Our CPU/SIMD/wgpu stack must be re-profiled before
  promotion; we ship the synchronous non-anticipating form because it is provably correct.
- **Non-anticipating early-stop is a correctness theorem.** Do NOT remove it without porting the
  async-ZOS causality proof from DSpark §5.2.

## File change summary

| File | Change |
|------|--------|
| `src/speculative/prefix_scheduler.rs` | New: scheduler + `SpsCurve` + non-anticipating early-stop |
| `Cargo.toml` | New `hardware_aware_scheduler = []` feature (default-OFF) |
| `src/speculative/mod.rs` | New `#[cfg(feature)] pub mod` + re-exports |
| `tests/prefix_scheduler_goat.rs` | New: G1–G5 GOAT gate |

## Cross-references

- Issue: `.issues/003_hardware_aware_prefix_scheduler.md`
- Research: `.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md`
- Per-request analog: `src/speculative/caddtree_budget.rs`, `src/speculative/budget.rs`
- Survival-prob producer: `src/speculative/acceptance_forecast.rs` (`AcceptanceForecast`, Bebop Plan 243)
- SIMD Π c_i: `src/cumprodsum.rs::cumprodsum_scalar`
- DSpark paper §3.2.2 (Algorithm 1), §5.2 (production async variant), Appendix A (correctness proof)

## TL;DR

Plan 339 ships a generic, modelless, zero-allocation `HardwareAwarePrefixScheduler` behind
`hardware_aware_scheduler` (default-OFF). Given R requests each with monotone-non-increasing
per-position survival probabilities `a_{r,j}` and a profiled `SPS(B)` engine cost curve, produces
per-request prefix lengths `ℓ*_r` maximizing `Θ = τ · SPS(B)` via global sort + greedy admission +
**non-anticipating early-stop** (the Appendix A correctness theorem — removing it breaks the
lossless distribution guarantee). The scheduler does NOT remove the early-stop for throughput (the
production DSpark async-ZOS variant requires a separate causality proof we do not port). Promotion
to default-on is deferred: katgpt-rs is single-request by default, so the synthetic G2 gate is
necessary but not sufficient — a real multi-request batch caller (riir-ai crowd cognition or a
katgpt-rs batch server) must exercise it first.
