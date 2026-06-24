# Issue 003: Geometric Product SiLU `exp()` Perf Unblock

**Date:** 2026-06-25
**Parent:** Plan 319 (Clifford Geometric Product), Research 299
**Type:** Optimization
**Priority:** Medium (blocks default-on promotion of `geometric_product`)
**Status:** Open

---

## Problem

The channel-wise geometric product primitive (`katgpt-rs/crates/katgpt-core/src/linalg/geometric_product.rs`)
is a **quality GOAT** — the wedge carries non-redundant information proven on two
independent criteria (G1 non-redundancy +17.6pp, G2 rotational recovery r=0.96).
See `.benchmarks/319_geometric_product_goat.md`.

However, the **absolute latency** misses the plasma-tier targets:

| Config | Current | Target | Gap |
|--------|---------|--------|-----|
| D=8, \|S\|=4 (HLA) | 152 ns | < 50 ns | 3× over |
| D=64, \|S\|=7 (shard) | 1071 ns | < 200 ns | 5× over |

The bottleneck is **SiLU's `exp()` call** — `x / (1 + e^{-x})`. At D=8 there are
`8×4=32` SiLU evaluations; at D=64 there are `64×7=448`. Even at 2 ns per `exp()`,
that's 64–896 ns minimum — the targets are below the `exp()` floor.

The **algorithmic speedup** (9.33× vs O(D²) naive at D=64) is correct. The wedge
arithmetic (Hadamard + subtract) is cheap; `exp()` dominates.

## Proposed Solutions (any one unblocks; combination preferred)

### Option A — Polynomial Sigmoid Approximation (cheapest, ~5–10× speedup)

Replace `silu(x) = x / (1 + e^{-x})` with a degree-3 polynomial approximation.
Common choices:

- **Padé approximant** `[1, 0, 0] / [1, 0, 1]` — rational, stable, ~1e-3 error.
- **Chebyshev fit** on `[-6, 6]` — degree-3, ~1e-4 max error in the active range.
- **Piecewise linear + quadratic** — branch on `|x|`, use cheap forms in each region.

**Tradeoff:** sacrifices ~1e-3 to 1e-4 numerical exactness. Must verify the G1/G2
quality gates still pass with the approximation (they have margin: G1 is +17.6pp,
G2 is r=0.96 — both well above their 0.90 / 0.75 thresholds).

**Expected latency:** D=8 ~30 ns, D=64 ~200 ns.

### Option B — Batch SIMD `exp()` (restructure inner loop)

The codebase already ships `simd::simd_sigmoid_inplace` (NEON/AVX2). Currently the
SiLU is applied per-element inside the shift loop. Restructure to:
1. Accumulate raw Hadamard products into `dot_out` across all shifts (no SiLU).
2. Apply `simd_sigmoid_inplace` + element-wise multiply ONCE on the accumulated buffer.

**Tradeoff:** adds a second pass over `dot_out` (D elements), but replaces
`D×|S|` scalar `exp()` calls with `D/4` SIMD `exp()` calls. At D=64, that's 64/4=16
SIMD `exp()` vs 448 scalar `exp()` — ~28× fewer `exp()` calls.

**Expected latency:** D=8 ~80 ns, D=64 ~300 ns (better but may still miss targets).

### Option C — `geometric_product_wedge_only` Variant (skip SiLU entirely)

For callers that only need the wedge (structural divergence) and don't need the
coherence gate (e.g. shard retrieval, CGSP curiosity), provide a variant that
skips the dot/SiLU path entirely:

```rust
pub fn geometric_product_wedge_into(u, v, dim, shifts, wedge_out, scratch_u, scratch_v)
```

**Tradeoff:** no coherence signal. Some callers (HLA complementarity) need both
dot and wedge; this variant is only for wedge-only use cases.

**Expected latency:** D=8 ~15 ns, D=64 ~120 ns (no `exp()` at all — wedge is just
Hadamard + subtract).

## Recommendation

Implement **Option A (polynomial sigmoid) + Option C (wedge-only variant)**. Option
A unblocks the full primitive for hot-path use; Option C provides an ultra-fast
path for cold-path shard retrieval. Option B is a fallback if A's numerical error
turns out to break G1/G2.

## Acceptance Criteria

- [ ] D=8 latency < 50 ns/call (release, Apple Silicon)
- [ ] D=64 latency < 200 ns/call (release, Apple Silicon)
- [ ] G1 non-redundancy still ≥ +10pp (wedge-only vs dot-only on A-vs-B)
- [ ] G2 rotational recovery still r ≥ 0.85 (some slack from 0.90 for approximation error)
- [ ] G3 zero alloc maintained
- [ ] GOAT gate bench re-run with updated numbers in `.benchmarks/319_geometric_product_goat.md`
- [ ] If all pass → promote `geometric_product` to default-on

## References

- Primitive: `katgpt-rs/crates/katgpt-core/src/linalg/geometric_product.rs`
- GOAT gate: `katgpt-rs/crates/katgpt-core/benches/bench_319_geometric_product_goat.rs`
- Results: `katgpt-rs/.benchmarks/319_geometric_product_goat.md`
- SIMD sigmoid: `katgpt-rs/crates/katgpt-core/src/simd/activations.rs::fast_sigmoid`
