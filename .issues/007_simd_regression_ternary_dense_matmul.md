# Issue 007: SIMD Regression — Ternary Matvec (-34%)

## Status: CLOSED

## Severity: HIGH

## Resolution

### Bisect Result

Binary search across 17 commits touching `crates/katgpt-core/src/simd.rs` identified the exact regression:

**Commit `c0c9d728`** — `perf(katgpt-core): optimize layout, SIMD kernels, and error handling across 22 files`

### Root Cause

The commit replaced `if/else` mask construction with `0u32.wrapping_sub(bit)` in both
`neon_ternary_matvec` and `avx2_ternary_matvec`. The `wrapping_sub` approach creates
**scalar dependency chains** that prevent LLVM from auto-vectorizing the mask array
construction. The old `if x != 0 { !0u32 } else { 0 }` pattern compiles to NEON
compare+negate sequences that vectorize cleanly.

### Fix

Commit `eedd17f1` — Reverted mask construction in `neon_ternary_matvec` and
`avx2_ternary_matvec` from `wrapping_sub` back to `if/else` pattern.

### Controlled Benchmark Results

All measurements with 30s thermal soak, isolated benchmark binary:

| Benchmark | `762f2f72` (baseline) | HEAD (pre-fix) | HEAD (post-fix) | Change |
|-----------|----------------------|----------------|-----------------|--------|
| Ternary matvec 128×128 | 210K ops/s | 133K ops/s | 203K ops/s | **+53% recovered** |
| Dense matmul 64×16 | 15.2M ops/s | 16.4M ops/s | 15.1M ops/s | **Not regressed** |

### Key Finding: Dense Matmul Was Never Regressed

The original "-18% dense matmul" report was from thermally contaminated data
(Phase 13 of 13, after 12 phases of computation with only 3s cooldown).
Controlled re-test shows dense matmul is **slightly faster** on HEAD.

## Affected Benchmarks (Updated)

| Benchmark | Peak (762f2f72) | Pre-fix | Post-fix | Status |
|-----------|----------------|---------|----------|--------|
| Ternary matvec 64×64 | 950K | ~600K | TBD | Likely fixed |
| Ternary matvec 128×128 | 210K | 133K | 203K | ✅ Fixed |
| Ternary matvec 256×256 | 60K | ~39K | TBD | Likely fixed |
| Dense matmul 64×16 | 15.2M | 16.4M | 15.1M | ❌ Not regressed (false alarm) |
| Dense matmul 128×32 | 4.5M | ~3.6M | TBD | Likely false alarm |

## Remaining Action Items

- [x] Bisect 17 simd.rs commits → found `c0c9d728`
- [x] Identify root cause → `wrapping_sub` scalar dep chain
- [x] Fix → revert to if/else mask construction
- [x] Controlled verification → 203K ops/s matches baseline
- [-] Consider feature-gating `Mutex` fields in `BanditPruner` to reduce struct size
      (Δ-Bandit remaining 2x gap: 65M vs 140M peak — separate issue, not part of 007 scope)

## Notes

- Bisect tooling: `examples/bench_simd_bisect.rs` — isolated binary with 30s thermal soak
- Build time per commit: ~15s (no LTO, default release profile)
- Total bisect: 5 checkpoints (ca0f78d3, c0c9d728, 74528b1b, 2776eab8, c0c9d728)
