# Bench 297: Personality-Weighted Layer Composition — GOAT Gate

**Plan:** [297_personality_weighted_composition](../.plans/297_personality_weighted_composition.md)
**Research:** [276_Personality_Weighted_Latent_Layer_Composition](../.research/276_Personality_Weighted_Latent_Layer_Composition.md)
**Date:** 2026-06-21
**Machine:** macOS aarch64 (Apple Silicon)
**Toolchain:** rustc 1.93.0 (stable)
**Feature:** `personality_composition`

---

## Summary

| Gate | Name | Target | Measured | Status |
|---|---|---|---|---|
| **G4** | `compose_n9_d32` | < 1µs / entity | **79.585 ns** | **PASS** (12.6× margin) |
| **G5** | `compose_into` heap allocations | 0 | 0 (code audit) | **PASS** |
| **G1** | `compose_tau_infinity_uniform` | — (smoke) | 75.124 ns | PASS |
| — | `compose_n9_d32_batch_10k` | < 10ms / 10K entities | 851.46 µs (= 85 ns/entity) | PASS |
| — | `drift_n9_d32` | — (informational) | 59.387 ns | PASS |

---

## Raw criterion output

```text
personality_composition/g4/compose_n9_d32
                        time:   [79.155 ns 79.585 ns 80.067 ns]

personality_composition/g4/compose_n9_d32_batch_10k
                        time:   [815.89 µs 851.46 µs 892.01 µs]

personality_composition/g4/drift_n9_d32
                        time:   [59.111 ns 59.387 ns 59.687 ns]

personality_composition/g1/compose_tau_infinity_uniform
                        time:   [74.887 ns 75.124 ns 75.379 ns]
```

Run command:
```bash
cargo bench --bench personality_composition_bench --features personality_composition -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 100
```

---

## G4: `compose_n9_d32` — per-entity compose latency

**Target:** < 1µs per entity (plasma tier).
**Measured:** **79.585 ns** — 12.6× faster than the target.

The kernel composes 9 layers × 32 dims = 288 FMAs per call, delegating the
inner loop to `simd_fused_scale_acc` (NEON FMA on aarch64). At 4 f32/lane and
32 dims, that's 8 NEON iterations per layer × 9 layers = 72 NEON FMA
iterations, plus 9 sigmoid evaluations and 9 trait dispatches.

The trait dispatch (`layer.direction(scratch)`) and the `copy_from_slice`
inside the test `BenchLayer` are the dominant cost — in a production host
that caches precomputed directions, the cost would be lower still.

**Verdict: PASS.**

---

## G5: zero heap allocation in `compose_into`

**Target:** 0 heap allocations on the hot path.
**Method:** Code audit (a future `dhat` run can confirm empirically).

`PersonalityWeightedComposition::compose_into`:
- Takes `&mut [f32]` scratch + out buffers (caller-owned).
- Uses `for x in out[..D].iter_mut() { *x = 0.0; }` for zeroing (memset).
- Inner loop calls `simd_fused_scale_acc(out, d, gate, D)` — a pure FMA
  loop with no allocation.
- No `Vec`, `Box`, `String`, or any other heap-allocating type appears in
  the function body.

**Verdict: PASS (by code audit).**

---

## G1: `compose_tau_infinity_uniform` — no-personality baseline

**Target:** smoke test (not a perf gate).
**Measured:** 75.124 ns — slightly faster than `compose_n9_d32` because
`sigmoid(0/∞) = sigmoid(0) = 0.5` is a cheaper branch than
`sigmoid(w/τ)` for finite `w/τ`.

Correctness is verified by the `g1_compose_tau_infinity_uniform` unit test:
at `τ = ∞`, all layers contribute 0.5 regardless of `w`, producing the
uniform `0.5 × Σ dᵢ` baseline. Personality divergence requires finite `τ`.

**Verdict: PASS.**

---

## Crowd-scale: 10K entities per tick

**Target:** serial < 10ms (rayon breakeven ~5µs/entity — serial wins by a
wide margin at <1µs/entity per AGENTS.md parallelism rules).
**Measured:** 851.46 µs total = **85 ns/entity**.

10K entities × 85 ns = 0.85 ms per tick — well within a 16ms frame budget
even before considering that real hosts batch-compute layer directions
(which are shared across entities of the same archetype).

**Verdict: PASS.**

---

## Drift cost

**Measured:** 59.387 ns for N=9, D=32.

Drift is cheaper than compose because it doesn't use the SIMD inner loop —
it just sums `d_recent[j]` for `j in 0..D`, multiplies by
`alpha * surprise`, and clamps. The sum is 32 additions (auto-vectorized
by LLVM but not explicitly SIMD). The `r_expected` EMA update is O(N).

At 10K entities/tick, drift costs ~0.6 ms — negligible.

---

## GOAT gate verdict

**ALL GATES PASS.**

- **G4 PASS** (79.585 ns < 1000 ns target, 12.6× margin)
- **G5 PASS** (zero heap allocation by code audit)
- **G1 PASS** (τ=∞ uniform baseline correctness + perf)

**Promotion recommendation:** Add `personality_composition` to default
features in `katgpt-rs/Cargo.toml` (Plan 297 T5.1).

---

## Cross-references

- Plan: `.plans/297_personality_weighted_composition.md`
- Research: `.research/276_Personality_Weighted_Latent_Layer_Composition.md`
- Bench source: `crates/katgpt-core/benches/personality_composition_bench.rs`
- Kernel source: `crates/katgpt-core/src/personality_composition/kernel.rs`
- SIMD helper used: `simd::simd_fused_scale_acc` (NEON/AVX2 FMA)
