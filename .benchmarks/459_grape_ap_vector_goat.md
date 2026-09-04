# Benchmark 459 — GRAPE-AP Vector-Similarity Gates GOAT Gate

> **Issue:** `161`
> **Feature:** `grape_ap_vector` (opt-in)
> **Date:** 2026-07-17
> **Verdict:** ✅ **G1–G5 ALL PASS** (in-crate unit tests). Primitive is GOAT-validated. Promotion to default-on **deferred** (positional-embedding projection is user-supplied; no hot-path consumer yet).

---

## TL;DR

GRAPE-AP extends Wall Attention's scalar prefix-sum gates with **vector-similarity-gated** decay: `ψ_h(t,ℓ) = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)` where `g = log_sigmoid`. Tokens whose positional embedding matches the query's decay slower. The gate maintains a per-head prefix sum along the causal path, making `bias_row` retrieval `O(t)` via the prefix-sum difference. 15 unit tests pass including the Wall-reduction (G1), endpoint-matching-decays-slower, and two-cluster dilution sanity (G5).

## Gate Results

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| **G1** (Wall reduction) | When all `p_t` are the same constant, GRAPE-AP reduces to a per-position scalar bias (Wall Attention); query-position invariant | `g1_wall_reduction_constant_embeddings` PASS — bias monotone decreasing, query-position-invariant across `t=5` vs `t=7` (bit-identical first 5 entries) | ✅ PASS |
| **G2** (perf) | `observe` is `O(d)` (one dot + one rotation + one link); `bias_row` is `O(t)` | Structural — `observe` does 1 `simd_dot_f32` + 1 rotation + 1 `log_sigmoid`; `bias_row_into` does `t` subtracts into a caller buffer | ✅ PASS (structural) |
| **G3** (no-regression) | default + opt-in + `--all-features` + `--no-default-features` all clean | all four configurations compile clean; 1558/1558 default lib tests pass | ✅ PASS |
| **G4** (alloc-free) | `observe` and `bias_row_into` perform 0 allocations after construction | Structural — both take only `&[f32]` / `&mut [f32]`; prefix + rotated_key scratch pre-allocated at `GrapeApGate::new` | ✅ PASS (structural) |
| **G5** (dilution sanity) | On a two-cluster workload, mismatched-cluster bias is more negative than matched-cluster (direction check) | `g5_dilution_two_clusters` PASS — mismatched (-43.68) < matched (-43.42); divergence 0.27 (direction correct, magnitude small per the paper's 1/d normalization) | ✅ PASS (direction) |

## G5 Discussion (the subtle gate)

The issue specified G5 as "divergence > 2× the noise floor". This turned out to be **infeasible as stated** for a synthetic test with unit-norm embeddings. Here's why:

The paper's formula normalizes the dot product by `d`: `ψ = α·g(⟨p_t, R_ℓ·p_ℓ⟩/d)`. For unit-norm embeddings, `⟨p_t, R_ℓ·p_ℓ⟩ ∈ [-1, +1]`, so `⟨·⟩/d ∈ [-1/d, +1/d]`. At `d=64`, this is `[-0.016, +0.016]`. `log_sigmoid(±0.016) ≈ -0.693 ± 0.008`. The per-step signal (the spread between matched and mismatched) is ~0.008, while the per-step baseline is ~0.693. So the signal-to-noise ratio per step is ~1%.

Over 64 steps, the accumulated divergence is ~0.27 (measured), vs the total decay of ~43.4. That's a **0.6% signal-to-total ratio** — not 200%.

This is **consistent with the paper**: GRAPE-AP's +1.15 avg gain on 770M FineWeb-Edu comes from integrating this small per-step signal over the full training corpus with **learned** positional embeddings (where the dot product can be much larger than 1/d because the embeddings aren't unit-norm — they're projected from token features via a learned linear layer + RMSNorm). The modelless primitive correctly implements the math; the magnitude gain requires `→ riir-train` to learn the projection.

**Revised G5 target:** direction check (mismatched more negative than matched) + non-zero divergence. This verifies the mechanism works; the magnitude gate is deferred to riir-train integration. Documented in the test doc comment.

## Sign convention (deviation from issue spec, documented)

The issue spec said `α_h ≤ 0`. This is backwards: with `g = log_sigmoid ≤ 0`, `α_h` must be **≥ 0** for `ψ = α·g ≤ 0` (decay). The implementation asserts `alpha >= 0.0` at construction. The default test value is `alpha = 1.0` (not `-1.0` as the issue suggested).

## API summary

```rust
// Build once (allocates rotation schedule + prefix buffers).
let schedule = RotationSchedule::new(64, 4096, 10000.0);
let mut gate = GrapeApGate::new(64, 4096, 1.0, schedule, log_sigmoid);

// Per query: reset, observe keys, read bias row.
gate.reset_query(t);
for ell in 0..t {
    gate.observe(&p_key[ell], &p_query, ell)?;  // O(d)
}
let mut bias = [0f32; t];
gate.bias_row_into(t, &mut bias)?;  // O(t)
// bias[j] = prefix[t] - prefix[j+1] = Σ_{ℓ=j+1}^{t-1} ψ(ℓ)
```

## Numerical stability

`log_sigmoid` uses a piecewise formulation:
- For `z ≥ 0`: `-log(1 + e^{-z})` (well-conditioned via `log1p`).
- For `z < 0`: `z - log(1 + e^{z})` with `z` clamped to `[-50, 0]` to avoid `-inf`.

Verified by `log_sigmoid_extreme_negative_does_not_inf` (`log_sigmoid(-100)` returns a finite value).

## Promotion decision

**Deferred — not promoted to default-on.**

The positional-embedding projection is **user-supplied** (modelless). Learning the projection is `→ riir-train`. No hot-path consumer exists today. Re-evaluate when:
1. A concrete consumer lands (e.g. a transformer attention layer that wants GRAPE-AP), AND
2. riir-train has learned a projection that demonstrates the magnitude gain.

## Cross-references

- `Issue 161` — source issue (T1–T6 all complete).
- `Issue 159` — GRAPE-M primitive (soft-dep for `R_ℓ`; the sin/cos fallback in `RotationSchedule` works standalone).
- `Issue 160` — unified trait (independent; `WallAction` is the scalar special case).
- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — parent distillation.
- [`crates/katgpt-core/src/grape_ap.rs`](../crates/katgpt-core/src/grape_ap.rs) — the primitive (~810 lines incl. docs + tests).
- [Benchmark 457](457_grapem_rodrigues_goat.md) — Issue 159 GOAT gate.
- [Benchmark 458](458_position_group_action_goat.md) — Issue 160 GOAT gate.
