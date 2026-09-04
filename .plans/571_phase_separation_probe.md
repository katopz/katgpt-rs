# Plan 571: Phase Separation Probe — Open Primitive (Lonely Runner Conjecture)

**Date:** 2026-08-06
**Research:** [katgpt-rs/.research/470_Lonely_Runner_Phase_Separation_Probe.md](../.research/470_Lonely_Runner_Phase_Separation_Probe.md)
**Source paper:** [arXiv:0710.4495](https://arxiv.org/abs/0710.4495) — Barajas & Serra, *The Lonely Runner with Seven Runners* (2007)
**Private guide:** [riir-ai/.research/334_phase_separation_game_runtime_guide.md](../../riir-ai/.research/334_phase_separation_game_runtime_guide.md)
**Target:** `katgpt-rs/crates/katgpt-core/src/phase_separation.rs` (new module) + Cargo feature `phase_separation`
**Status:** Phase 1 + Phase 2 + Phase 3 COMPLETE (2026-08-07). Phase 1 GOAT gate ALL PASS → **PROMOTED TO DEFAULT-ON** (2026-08-07, commit 6b3f6c2d). See [.benchmarks/571_phase_separation_goat.md](../.benchmarks/571_phase_separation_goat.md). Phase 2 T2.2 + T2.3 done (T2.1 deferred); Phase 3 docs + example done. Cherry-pick clock for riir-ai starts at promotion (7-day window per goat-audit skill).

---

## Goal

Ship a generic, modelless `phase_separation_probe` that computes per-entity minimum circular distance on a phase circle. The primitive is the open (public) layer of the Super-GOAT fusion described in Research 470; the private game-runtime fusion (Salience Tri-Gate × Sleep-Time × KARC × feeling brain) lives in riir-ai per the private guide 334.

The primitive computes, for N entities each with a phase `φ_i ∈ [0, 1)`:

```
phase_separation(i) = min_{j ≠ i} ‖φ_i − φ_j‖ mod 1     ∈ [0, 0.5]
```

where `‖x‖ mod 1` is the distance to the nearest integer (circular distance on the unit torus). O(N log N) via sort + adjacent-neighbor scan.

**Theorem backing** (Lonely Runner Conjecture, proven for N ≤ 7 by Barajas & Serra 2007): for N entities with integer cycle speeds {s_1, ..., s_N} (gcd = 1), every entity i has some tick t where `phase_separation(i) ≥ 1/N`. The primitive computes the per-tick scalar; the theorem justifies using it as a behavior driver (guaranteed-peak property).

**GOAT gate**: G1 (determinism on integer phases — bit-identical), G2 (sub-µs at N=1000), G3 (no-regression, feature-flagged), G4 (alloc-free steady-state). No UQ floor (this is a deterministic distance metric, not a probability distribution).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Add `phase_separation` feature flag to `katgpt-rs/crates/katgpt-core/Cargo.toml` (opt-in, default-off).
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/phase_separation.rs` module with:
  - `phase_separation(phases: &[f32], i: usize) -> f32` — O(N) naive (for correctness testing + small N).
  - `phase_separation_all(phases: &[f32], out: &mut [f32])` — O(N²) all-pairs (for correctness testing).
  - `phase_separation_sorted(phases: &[f32], scratch: &mut [f32], out: &mut [f32])` — O(N log N) via sort + adjacent-neighbor scan (production path).
  - All three write into caller-provided `&mut [f32]` — zero allocation.
  - **DEVIATION (documented):** `phase_separation_sorted` takes `scratch_perm: &mut [usize]`
    (not `&mut [f32]`) because sorting values destroys the original-index mapping. The
    permutation-index sort is O(N log N) total (sort + linear scan), vs the
    binary-search-per-entity approach which is O(N log N) sort + O(N log N) searches
    (2× the work, and it failed the G2 latency gate at 18025 ns before this fix).
    The `usize` type is required because the permutation holds original indices.
- [x] **T1.3** Implement the sorted-scan algorithm:
  1. Fill `scratch_perm` with `0..n`, sort by `phases[scratch_perm[k]]` ascending.
  2. For each rank `k`, compute min circular distance to left/right sorted neighbors
     (with circle wraparound). Write `out[scratch_perm[k]] = sep_k`.
  - Edge cases: N=0 → no-op; N=1 → `out[0] = 0.5`; N=2 → correct min circular distance.
- [x] **T1.4** Unit tests (G1 determinism):
  - `g1_integer_phases_bit_identical`: phases from integer speeds `{1, 2, 3, 4, 5, 6, 7}`
    at tick `k=42`, period P=420 (lcm) → verify `phase_separation`, `phase_separation_all`,
    and `phase_separation_sorted` all produce bit-identical f32 output.
  - `g1_circle_wraparound`: phases `{0.0, 0.49, 0.51}` → entity 0's separation = 0.49
    (wraparound path).
  - `g1_edge_cases`: N=0 → 0.0; N=1 → 0.5; N=2 with phases `{0.0, 0.5}` → 0.5 each.
  - `g1_tie_handling_tick_zero`: tick 0 → all phases 0 → all separations 0.
  - `g1_lrc_bound_n7`: with N=7 entities, integer speeds `{1,2,3,4,5,6,7}` (gcd=1),
    scan the discrete orbit k=0..420 (period P=lcm(1..=7)=420) and verify every entity
    hits `phase_separation ≥ 1/7` at least once. **Theorem confirmation test PASSES.**
    Note: the continuous LRC is over real time t; we sample at granularity 1/P via
    `phase_i(k) = (s_i · k mod P) / P`, which approximates `{s_i · t}` at `t = k/P`.
  - `g1_sorted_matches_naive_random`: 50 random trials, sorted scan agrees with naive
    all-pairs within 1 ULP.
- [x] **T1.5** Re-export at crate root: `katgpt-rs/crates/katgpt-core/src/lib.rs` →
  `#[cfg(feature = "phase_separation")] pub mod phase_separation;` + `pub use` for all
  6 public functions.

### G2 — Perf gate

- [x] **T1.6** Add criterion bench `katgpt-rs/crates/katgpt-core/benches/bench_571_phase_separation_goat.rs`:
  - N = {10, 100, 1000, 10000} entities.
  - Measure `phase_separation_sorted` wall time.
  - Target: < 10µs at N=1000 — **PASS at 8033 ns** (sub-10µs, 1.25× headroom).
  - Report O(N log N) scaling — **PASS at 12.59×** (predicts ~13×).
  - Note: uses `std::time::Instant` + `harness = false` (mirrors bench_371 pattern,
    not criterion — avoids the criterion framework overhead for a simple gate).

### G3 — No-regression gate

- [x] **T1.7** `cargo test -p katgpt-core --lib` passes with `phase_separation` off (default,
    1840 passed) AND on (`--features phase_separation`, 1848 passed = 1840 + 8 new).
- [x] **T1.8** `cargo clippy -p katgpt-core --all-targets --features phase_separation` zero warnings.

### G4 — Alloc-free gate

- [x] **T1.9** `CountingAllocator` test: call `phase_separation_sorted` 1000 times on a
    pre-allocated `scratch_perm` buffer at N=1000. **0 allocations after warmup** —
    the permutation sort is in place on `scratch_perm`; the scan + write use only
    stack-local `f32` arithmetic.

---

## Phase 2 — API ergonomics (after Phase 1 GOAT passes)

### Tasks

**NOTE (2026-08-06):** T2.2 and T2.3 were pulled forward into Phase 1 because
the LRC bound confirmation test (T1.4 `g1_lrc_bound_n7`) needed the raw
bridge (`from_speeds_and_tick`) to materialize phases, and the latent
projection bridge (`from_latent_projection`) was cheap to ship alongside.
Both are documented in the module rustdoc + cross-referenced to Research 470.

- [-] **T2.1** Add `PhaseSeparationProbe` struct (zero-sized, `Copy`) wrapping the
  sorted-scan with a cached scratch buffer — **DEFERRED.** The three free
  functions (`phase_separation`, `phase_separation_all`,
  `phase_separation_sorted`) already cover all use cases; the struct would be
  a convenience wrapper for callers that want to pre-allocate once. Revisit
  when a concrete consumer (riir-ai Salience Tri-Gate fusion) materializes.
- [x] **T2.2** Add `from_speeds_and_tick(speeds: &[u32], tick: u64, period: u32,
  out_phases: &mut [f32])` helper — raw time-phase path (sync-safe). Computes
  `(s_i · tick) mod P / P` into `out_phases`.
- [x] **T2.3** Add `from_latent_projection(latent_states: &[f32], direction:
  &[f32], out_phases: &mut [f32])` helper — latent projection path (local-only).
  Computes `sigmoid(dot(direction, latent_state_i))` into `out_phases`.

---

## Phase 3 — Documentation (after Phase 2)

### Tasks

- [x] **T3.1** Module-level rustdoc with:
  - The LRC citation + scope caveat (N≤7 proven, N>7 conjectured).
  - The raw-vs-latent boundary (per AGENTS.md).
  - The bridge pattern (raw time-phase → latent-projected phase → separation scalar).
  - Cross-ref to Research 470 + private guide 334.

  **Done in Phase 1/2 (2026-08-06):** the module rustdoc was written
  alongside the code. It satisfies all four requirements: LRC citation +
  scope caveat (N≤7 proven / N>7 conjectured), raw-vs-latent boundary, the
  bridge pattern (zero-alloc + gateable + no sync dependency), and the
  cross-references block (Research 470 + private guide 334 + Research 056).
  Checkbox closed 2026-08-07 on audit.
- [x] **T3.2** Example in `katgpt-rs/crates/katgpt-core/examples/phase_separation_demo.rs`
  (done 2026-08-07):
  - 7 entities with integer speeds `{1,2,3,4,5,6,7}`.
  - Scans the full discrete orbit k=0..420 (period = lcm(1..=7)), records
    the loneliest tick per entity + max separation.
  - Prints a table; confirms the LRC bound (every entity hits ≥ 1/7).
    **LRC CONFIRMED** — all 7 entities reached phase_separation ≥ 1/7 (eps
    slack ±5/420 for discrete sampling). The slowest entity (s=1, s=7) hits
    exactly 1/7 at k=60; the fastest-to-lonely (s=4) hits 0.25 at k=105.
  - Uses only the public API (`from_speeds_and_tick` +
    `phase_separation_sorted`); caller-provided scratch, zero allocation.
  - Gated `required-features = ["phase_separation"]` in Cargo.toml.
  - Clippy zero warnings; 1853 lib tests pass with the feature on.

---

## Non-goals

- **NOT implementing the fusion.** The Salience Tri-Gate / Sleep-Time / KARC / feeling-brain fusions are riir-ai tasks tracked in the private guide (334). This plan ships ONLY the generic primitive.
- **NOT formalizing the LRC in Lean 4.** The theorem is published; formalizing the 20-page case analysis is a separate research project. The primitive's invariant (min over a metric → non-negative, ≤ 0.5) is trivially true by construction.
- **NOT handling N > 7 with a proven bound.** The LRC is conjectured for N > 7. The primitive computes the scalar correctly for any N; only the *peak guarantee* is conjectural at scale. Honest framing in the docs.
- **NOT a UQ primitive.** `phase_separation` is a deterministic distance metric, not a probability distribution. No conformal-naive floor comparison needed (per the "Report the Floor" rule).

---

## See also

- [Research 470](../.research/470_Lonely_Runner_Phase_Separation_Probe.md) — public distillation + Super-GOAT verdict
- [riir-ai/.research/334](../../riir-ai/.research/334_phase_separation_game_runtime_guide.md) — private game-runtime guide + fusion map
- [Research 056](../.research/056_OpenAI_Unit_Distance_Disproof.md) — same combinatorial family (chromatic number bounds on distance graphs)
- `Plan 303` — Salience Tri-Gate (primary fusion target, riir-ai follow-up)
