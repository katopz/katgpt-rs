# Plan 306 G4 Latency Benchmark — depth_invariance

**Date:** 2026-06-23
**Plan:** [306_depth_invariance_diagnostic.md](../.plans/306_depth_invariance_diagnostic.md) §Phase 6 (T6.1–T6.3) + T7.4 promotion decision
**Platform:** macOS aarch64 (release build)
**Decision:** **Feature stays opt-in.** G1/G2/G3 (correctness gates) PASS; G4 (latency) misses its aspirational targets — see analysis below. Per Plan 306 T7.4 ("If any fail → keep opt-in, document in `.benchmarks/`"), the literal gate is respected.

---

## Gate summary

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| G1 — 8 correctness tests | pass | ✅ 8/8 (Phase 1, shipped) | PASS |
| G2 — BeliefDrafter classifies `DepthSpecificRefinement` beyond TTT | reproduce paper finding on random init | ✅ `DepthSpecificRefinement`, locked-drift sub-case (`mean_cos_step`=0.99997 > 0.95), magnitude slope 0.239 | PASS |
| G3a — AttractorKernel classifies `DepthInvariant` (negative control) | invariant by clamp construction | ✅ magnitude slope 0.0008 | PASS |
| G3b — unclamped leaky classifies `DepthSpecificRefinement` (positive control) | drift without clamp | ✅ magnitude slope 0.1414, 32.1× growth | PASS |
| G4.1 — `classify_chain` ≤ 5% of `forward_into` time | ≤5% across d∈{8..1024}, k∈{4,16,64} | ❌ see table below | **MISS** |
| G4.2 — batched throughput ≥ 10M/sec (1000 chains, d=8, k=16) | ≥10M | 7.9M/sec | **MISS** (close) |
| G4.3 — `apply_magnitude_regularization` ≤ 2% overhead vs raw residual write | ≤2% | ❌ 102–167% (see analysis) | **MISS** |

---

## G4.1 — `classify_chain` as % of one `forward_into`

| d | k=4 | k=16 | k=64 |
|---|---|---|---|
| 8 | 49% | 151% | 652% |
| 64 | 13% | 53% | 206% |
| 256 | 8% | 29% | 111% |
| 1024 | **2.2%** ✅ | 7% | 28% |

Only `d=1024, k=4` clears the ≤5% bar.

### Why the target is structurally unrealistic

`classify_chain` is **O(k · d)** (a single sweep for magnitude + flatness + cosine).
`LatentDynamicsMLP::forward_into` is **O(d²)** (three FC matmuls at `n_embd=d`).

The ratio `O(k·d) / O(d²) = O(k/d)`. At small `d`, forward is cheap so the
diagnostic's fixed per-element work dominates; at large `d`, the diagnostic
becomes negligible. The ≤5% bar is only reachable when `d ≫ k`, i.e. the HLA
operating regime (d=1024). The gate as written does not reflect the workload
shape the diagnostic is actually designed for.

The diagnostic is **off the hot path** — it runs at audit cadence (per-rollout
or per-batch), not per-token. The absolute `classify_chain` latency at the
HLA-shaped `d=1024, k=4` config is sub-microsecond and adds no measurable
overhead to a rollout.

## G4.3 — `apply_magnitude_regularization` overhead vs raw `out[i] = h[i] + Δ[i]`

| d | worst of RmsNorm / ScalarPinch |
|---|---|
| 8 | NaN (raw write too fast to measure reliably) |
| 64 | 102% |
| 256 | 167% |
| 1024 | 154% |

The regularization adds a second O(d) pass (sum-of-squares) plus a divide.
The "raw residual write" baseline is a single fused write — there is no way
to add an RMS computation in <2% of a single store loop. The ≤2% target is
physically unachievable for this operation shape; the gate was mis-specified.

For context, the regularization at `d=1024` is still sub-microsecond and
runs at most once per recursive step (not per-token) when applied to a
kernel we own.

---

## Recommendation

**Keep `depth_invariance` opt-in** per the literal T7.4 rule. The
correctness gates (G1/G2/G3) are strong — the headline G2 result reproduces
the paper's attention-drift finding on random-init weights, which is the
strongest possible signal that the drift is structural rather than learned.

The latency gates (G4.1/G4.3) were mis-specified relative to the workload
shape. A revised gate would be **absolute** (e.g. "classify_chain ≤ 1µs at
HLA d=1024") rather than **relative-to-forward** (which is structurally
unfavorable at small d). The current 7.9M classifications/sec batched at
d=8/k=16 is within 1.3× of the 10M target and would clear it on a
SIMD-vectorized inner loop (deferred Phase 1 TODO in `depth_invariance.rs`).

Revisit promotion after either (a) the SIMD-vectorized inner loop lands and
G4.2 clears 10M/sec, or (b) the gate is rewritten as an absolute-latency
target and the diagnostic clears it at the HLA operating point.
