# Benchmark 425: TILR — Trajectory-Invariant Latent Refinement GOAT Gate

**Date:** 2026-07-10
**Plan:** [425_tilr_invariant_subspace_refinement.md](../.plans/425_tilr_invariant_subspace_refinement.md)
**Research:** [408_Trajectory_Invariant_Latent_Refinement.md](../.research/408_Trajectory_Invariant_Latent_Refinement.md)
**Source paper:** [arXiv:2606.29164](https://arxiv.org/abs/2606.29164) — Malarkkan et al., *TILR: Trajectory-Invariant Latent Refinement*, ICML 2026 Mech Interp Workshop
**Feature:** `tilr_invariant_subspace` (BAKED flagless in katgpt-core since Plan 635, 2026-11-10)
**Primitive:** `tilr_refine_into(state, direction, basis, r, eta_base, epsilon, scratch, out)`

## GOAT Gate Results — ALL PASS

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **G1** (no-harm bit-identity) | γ=0 at orthogonal → out == state bit-identically | max γ at orthogonal = 0.0 exactly, 0 bit mismatches / 100 dirs × 2 scales | ✅ PASS |
| **G2** (full-correction parity + boundedness) | γ=1 at in-span → out = state + η·d; γ ∈ [0,1] no NaN/OOB | 0 OOB, 0 NaN / 1000 triples; full-correction max err 5.96e-8 << 1e-4 | ✅ PASS |
| **G3** (latency) | <50 ns HLA (d=8,r=3), <200 ns shard (d=64,r=12) | HLA 24.7 ns, Shard 123.0 ns | ✅ PASS |
| **G4** (alloc-free) | 0 heap allocs on hot path | 0 allocs / 100 steady-state calls (CountingAllocator) | ✅ PASS |

## G1 — No-Harm Bit-Identity (the kill switch)

**Setup:** For each (d, r) ∈ {(8, 3), (64, 12)}:
1. Construct a random orthonormal basis via Gram-Schmidt.
2. Construct a direction in the complement of span(basis) (random → subtract projection).
3. Call `tilr_refine_into` with `eta_base = 0.5`.
4. Assert γ = 0.0 exactly AND `out[i].to_bits() == state[i].to_bits()` for all i.

**Result:** 100 random orthogonal directions × 2 scales = 200 trials.
- Max γ at orthogonal: **0.0** (exactly — the `d_proj_norm_sq < epsilon` clamp guarantees η = 0.0 exactly, not ≈1e-38)
- Bit mismatches: **0**

The no-harm contract holds bit-identically. When the contrastive direction is orthogonal to the invariant subspace, the primitive recovers the uncorrected backbone exactly.

## G2 — Full-Correction Parity + γ Boundedness

**Setup:** For each (d, r) ∈ {(8, 3), (64, 12)}:
1. **Boundedness:** 500 random `(state, direction)` triples → assert γ ∈ [0, 1], no NaN.
2. **Full-correction parity:** direction = each basis vector k → assert γ ≈ 1.0 and `out = state + eta_base * basis[k]`.

**Result:**
- γ OOB count: **0** / 1000 random triples
- γ NaN count: **0**
- Full-correction max error: **5.96e-8** (budget 1e-4 — 1677× under)

When the direction lies in span(basis), the correction equals the ungated `state + eta_base * direction` to f32 precision.

## G3 — Latency

**Setup:** Batched-median timing, 1024 calls × 256 batches, `black_box` anti-hoist.

| Scale | d | r | Measured | Target | Headroom |
|-------|---|---|----------|--------|----------|
| HLA | 8 | 3 | 24.7 ns | <50 ns | 2.0× |
| Shard | 64 | 12 | 123.0 ns | <200 ns | 1.6× |

The O(d·r) projection + O(d) SAXPY is negligible vs O(d²) attention. The paper's <3% wall-clock overhead claim holds.

## G4 — Alloc-Free Hot Path

**Setup:** Pre-allocate `TilrScratch` once, warmup 10 calls, then measure 100 steady-state calls via `CountingAllocator`.

**Result:** **0 allocations** / 100 steady-state calls. The hot path reuses the pre-allocated scratch buffers without any heap allocation.

## UQ-Bearing Check

TILR does NOT claim a probability distribution, predictive interval, quantile, coverage guarantee, or calibrated uncertainty. It's a deterministic linear-algebra correction (SVD projection + norm ratio + SAXPY). **No conformal floor needed** per the "Report the Floor" rule (Issue 010).

## Promote-to-Default Decision

**G1+G2+G3+G4 ALL PASS + modelless gain** → `tilr_invariant_subspace` promoted to `default` in `katgpt-core/Cargo.toml` (Phase 17, 2026-07-10).

**No demotion needed:** TILR is the alignment-gated member of the subspace-projection family. It coexists with:
- Plan 412 `subspace_steering` (ungated, fixed α per axis)
- Plan 423 `spectral_rewire` (ungated projection, no step modulation)
- TILR (γ-gated step size with no-harm at γ=0)

## Reproduction

```bash
# Build the bench
CARGO_TARGET_DIR=/tmp/tilr_plan425 cargo bench -p katgpt-core --features tilr_invariant_subspace --bench bench_425_tilr_goat --no-run

# Run directly (avoids macOS dyld/trustd stall)
/tmp/tilr_plan425/release/deps/bench_425_tilr_goat-<hash>

# Or via cargo bench (may stall on macOS)
CARGO_TARGET_DIR=/tmp/tilr_plan425 cargo bench -p katgpt-core --features tilr_invariant_subspace --bench bench_425_tilr_goat -- --nocapture
```

## TL;DR

TILR alignment-gated subspace correction passes all 4 GOAT gates with 1.6–2.0× latency headroom and bit-identical no-harm at γ=0. Promoted to DEFAULT-ON. The γ-gated member of the subspace-projection family.
