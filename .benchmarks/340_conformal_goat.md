# Plan 340 GOAT Gate — Conformal Predictive Intervals Primitive

**Date:** 2026-06-30
**Plan:** [`.plans/340_conformal_predictive_intervals_primitive.md`](../.plans/340_conformal_predictive_intervals_primitive.md) Phase 1
**Issue:** 010 (T1 — RESOLVED; tracker removed; floor shipped here)
**Feature flag:** `conformal_predictive_intervals` (opt-in)
**Modelless:** ✅ Yes — no training, no learned parameters, no gradient descent. Pure empirical-quantile calibration over a residual reservoir.

---

## TL;DR

**GOAT gate: ✅ PASS (G1, G2, G3, G4 all clear).** The conformal predictive
intervals primitive ships behind `conformal_predictive_intervals` (opt-in).
Promotion to default-on is **deferred** per Plan 340 T1.14: the open primitive's
gates prove the math; the runtime gates (Plan 342 riir-ai) prove the utility.
The primitive IS the canonical conformal-naive floor instance — Issue 010's
retroactive UQ-floor comparison harness is now unblocked.

**Perf note (the G2 win):** the initial implementation called `weighted_quantile`
twice per `interval_into` (once for q_{α/2}, once for q_{1−α/2}), recomputing
the full O(n) `exp()` weight scan on each call — **4n `exp()` calls per
interval**. This put H=1 at 1.15µs, 15% over the 1µs budget. The fix:
`weighted_quantile_pair` computes the weights once into a 4KB stack buffer
(`WEIGHTS_BUF_LEN = 1024`) and reuses them for both quantile lookups — **n
`exp()` calls per interval**, a 4× reduction. Result: H=1 dropped to **642ns**
(44% faster, 36% headroom under budget). This is the "Don't recompute
unchanged values" optimization rule applied at the micro-level.

---

## G1 — Coverage ✅ PASS

**Gate:** On stationary seasonal synthetic data `y_t = sin(2πt/m) + ε_t`,
`ε ~ N(0, σ)`, empirical coverage at α=0.05 over 10,000 ticks ∈ [0.93, 0.97]
for ALL `m ∈ {12, 24, 48}` AND on `m=1` HStep mode.

**Test:** `tests/conformal_coverage.rs` (3 tests)

### Results (α=0.05, target coverage 0.95)

| m | σ | Coverage | In [0.93, 0.97]? |
|---|---|---|---|
| 12 | 0.1 | 0.9447 | ✅ |
| 12 | 0.5 | 0.9459 | ✅ |
| 12 | 1.0 | 0.9467 | ✅ |
| 24 | 0.1 | 0.9464 | ✅ |
| 24 | 0.5 | 0.9426 / 0.9463 | ✅ |
| 24 | 1.0 | 0.9454 | ✅ |
| 48 | 0.1 | 0.9493 | ✅ |
| 48 | 0.5 | 0.9476 | ✅ |
| 48 | 1.0 | 0.9461 | ✅ |
| 1 (HStep) | 0.1 | 0.9457 | ✅ [0.90, 0.99] |
| 1 (HStep) | 0.5 | 0.9447 | ✅ [0.90, 0.99] |
| 1 (HStep) | 1.0 | 0.9445 | ✅ [0.90, 0.99] |

### Alpha sweep (m=24, σ=0.5)

| α | Target | Coverage | In tolerance? |
|---|---|---|---|
| 0.01 | 0.99 | 0.9842 | ✅ |
| 0.05 | 0.95 | 0.9463 | ✅ |
| 0.10 | 0.90 | 0.8966 | ✅ |
| 0.20 | 0.80 | 0.7954 | ✅ |

**Verdict:** The conformal calibration math is correct. Coverage tracks the
nominal level across all seasonal periods, noise levels, and alpha values.

---

## G2 — Latency ✅ PASS

**Gate:** `interval_into` ≤ 1µs at H=1, ≤ 100µs at H=8×8 channels.

**Bench:** `benches/conformal_interval_bench.rs` (criterion, 30 samples,
0.5s measurement, release build, Apple M-series).

### Results (median)

| Config | Before optim | **After optim** | Target | Verdict |
|---|---|---|---|---|
| `interval_into` H=1 (1ch) | 1.15µs | **642ns** | ≤ 1µs | ✅ PASS (36% headroom) |
| `interval_into` H=8 (1ch) | 9.25µs | **5.04µs** | — | ✅ (45% faster) |
| `interval_into` H=8×8 | 75.3µs | **40.3µs** | ≤ 100µs | ✅ PASS (60% headroom) |
| `update_residual` H=1 | 233ns | 233ns | — | (unchanged) |
| `update_residual` H=8×8 | 15.4µs | ~15µs | — | (unchanged) |

**Verdict:** All latency targets met with comfortable margin after the
weights-compute-once optimization.

---

## G3 — Zero-alloc ✅ PASS

**Gate:** `update_residual` and `interval_into` perform zero allocations after
warmup.

**Test:** `tests/conformal_alloc_check.rs` (CountingAllocator pattern, 1000
calls × 8 channels after warmup).

| Method | Allocs (1000 × 8 calls) | Verdict |
|---|---|---|
| `interval_into` | 0 | ✅ PASS |
| `update_residual` | 0 | ✅ PASS |

**Warmup note:** the first few `interval_into` calls trigger lazy allocations
from the libm `exp()` implementation's first-use init. The warmup sweeps all
`(alpha, h, channel)` combinations (4 alphas × 50 reps × 8 ch × 8 h = 12,800
calls) before measurement to settle these. This matches the `karc_alloc_check`
and `analytic_lattice_alloc_check` pattern.

**Verdict:** Zero allocations on the read and write hot paths after warmup.

---

## G4 — Bit-reproducibility ✅ PASS

**Gate:** Two `ConformalIntervalCalibrator` instances with identical
`(residual_pool, m, alpha, h, decay_config, orientation)` produce byte-
identical `PredictiveInterval` bounds (verified via `f32::to_bits`).

**Test:** `tests/conformal_reproducibility.rs` (3 tests)

| Check | Variations | Verdict |
|---|---|---|
| Identical configs → identical bounds | α ∈ {0.01, 0.05, 0.1, 0.2} × h ∈ {1, 8, 24} | ✅ PASS |
| Read idempotence (10× reads vs 1×) | α=0.05, h=1 | ✅ PASS |
| `sample_predictive_distribution` deterministic w/ fixed seed | 50 samples | ✅ PASS |

**Verdict:** Bit-identical reproducibility holds. The LatCal sync-boundary
story (two quorum nodes with the same residuals produce the same intervals) is
intact.

---

## AirPassengers CRPS — "Report the Floor" reference ✅

**Example:** `examples/conformal_airpassengers.rs` (synthetic proxy, 144
monthly observations, multiplicative seasonality m=12, log-linear trend).

| Metric | Conformal Overlay | Seasonal-Naive ±2σ | Winner |
|---|---|---|---|
| Empirical coverage (α=0.05) | 0.9167 | 1.0000 | Conformal (closer to 0.95) |
| Mean interval CRPS | **115.06** | 468.75 | Conformal (4× sharper) |
| Mean Winkler score | **126.87** | 468.75 | Conformal |
| Point-forecast RMSE | 63.26 | 63.26 | tie (same point forecaster) |

**Verdict:** Conformal overlay CRPS (115.06) is within 2× of the baseline
(gate holds), and is in fact **4× sharper** than the Gaussian ±2σ baseline.
The ±2σ baseline over-covers (1.0000) because the residuals are
non-Gaussian (multiplicative), making the Gaussian assumption conservative;
the conformal overlay correctly adapts to the empirical residual distribution.

This IS the canonical "Report the Floor" reference. Future UQ-bearing
primitives must beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`
with `m=1` on CRPS / coverage / Winkler at their GOAT gate.

---

## What ships

| File | Role |
|---|---|
| `src/conformal/mod.rs` | `ConformalIntervalCalibrator<F>`, `PointForecaster` trait, `PredictiveInterval`, `ResidualMode`, `DecayUnit` |
| `src/conformal/ring.rs` | `SortedRing`, `ResidualRingBuffer`, `RingBuffer`, `RingView` |
| `src/conformal/seasonal.rs` | `SeasonalPoolForecaster`, `SeasonalNaiveForecaster` (type alias), `seasonal_naive_floor()` |
| `src/conformal/metrics.rs` | `crps`, `crps_interval`, `winkler_score`, `empirical_coverage`, `mean_crps_interval`, `mean_winkler` |
| `tests/conformal_coverage.rs` | G1 gate (3 tests) |
| `tests/conformal_reproducibility.rs` | G4 gate (3 tests) |
| `tests/conformal_alloc_check.rs` | G3 gate (2 tests) |
| `benches/conformal_interval_bench.rs` | G2 gate (5 configs) |
| `examples/conformal_airpassengers.rs` | CRPS reproduction / "Report the Floor" reference |

**Total:** 24 unit tests + 8 integration tests = 32 tests, all passing.

---

## Promotion decision

**Opt-in (NOT default-on).** Per Plan 340 T1.14:

> Promotion deferred until the riir-ai runtime integration (Plan 342) confirms
> the curiosity false-positive win (G3 in the private guide) — the open
> primitive's gates prove the math; the runtime gates prove the utility.

The open primitive's GOAT gate (this document) is PASS. The runtime promotion
gate is a separate concern tracked in riir-ai Plan 342. This matches the
KARC pattern (Plan 308 shipped opt-in at Phase 1, promotion is a separate
decision).

---

## Unblocks

- **Issue 010 T1** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`
  with `m=1` is now shipped. The retroactive UQ-floor comparison harness
  (Issue 010 T2–T7) is now actionable.
- **Plan 340 Phase 2** — KARC adapter can now be built on the validated
  `PointForecaster` trait + `ConformalIntervalCalibrator` substrate.
- **Plan 342 (riir-ai)** — runtime integration (HLA overlay, curiosity,
  sleep-time, MCTS collapse) can consume the open primitive.

---

## Phase 2 update — KARC adapter + Lorenz-63 coverage (2026-06-30)

Phase 2 shipped the `KarcChannelForecaster` adapter (T2.1), the Lorenz-63
coverage demonstration (T2.2), and the no-regression gate (T2.3). All
Phase 1 gates remain GREEN; the trait signature change (`PointForecaster::
forecast_into` `&self` → `&mut self`) has zero perf impact.

### Trait signature change (zero perf impact)

`PointForecaster::forecast_into`: `&self` → `&mut self`. Cascading:
`ConformalIntervalCalibrator::{interval_into, coverage_violation,
sample_predictive_distribution}` → `&mut self`. The mutation is only to the
wrapped forecaster's scratch (impl detail); the residual pool (observable
state) is untouched on reads.

### G2 re-verification (post trait change)

| Config | Phase 1 | Phase 2 | Target | Verdict |
|---|---|---|---|---|
| `interval_into` H=1 | 642 ns | **640 ns** | ≤ 1µs | ✅ unchanged |
| `interval_into` H=8×1ch | — | 5.12 µs | ≤ 10µs | ✅ |
| `interval_into` H=8×8ch | 40.3 µs | **40.9 µs** | ≤ 100µs | ✅ unchanged |
| `update_residual` H=1 | — | 208 ns | — | ✅ |

(Within criterion noise — the `&mut self` change is a borrow-checker
annotation, not a runtime code change.)

### Phase 2 coverage gate — Lorenz-63 (chaotic)

KARC `D=3, M=8, K=4, λ=1e-3` fitted on 4000 samples of Lorenz-63 (RK4,
dt=0.02, normalized to [-1,1] for Chebyshev stability). Conformal overlay
on 2000 test ticks at α=0.05:

| Channel | Coverage | Mean CRPS | Mean half-width | Point RMSE |
|---|---|---|---|---|
| x | **0.9425** | 0.0002 | 0.0001 | 0.0001 |
| y | **0.9520** | 0.0018 | 0.0009 | 0.0005 |
| z | **0.9485** | 0.0002 | 0.0001 | 0.0001 |

Target: [0.90, 1.00] (chaotic regime, widened from Phase 1's [0.93, 0.97]
because KARC residuals are heavier-tailed on chaotic attractors). Nominal
0.95. ✅ All 3 channels pass — the conformal overlay is well-calibrated on
top of a chaotic KARC forecast.

### No-regression gate (T2.3)

3 active tests, all pass:
1. KARC forecast bit-identical across repeated calls (no hidden state
   perturbation from the conformal feature being compiled in).
2. `wout` matrix unchanged after 100 forecast calls (scratch reuse doesn't
   leak into the readout).
3. FourierBasis KARC also produces finite output (Chebyshev isn't special).

Plus 1 `#[ignore]`'d latency sanity test (authoritative gate is the
criterion bench, which is unchanged — see G2 table above).

### Tests shipped

- **4 adapter unit tests** (`conformal::karc_adapter::tests::*`): channel
  extraction, `observe_and_update` write path, channel-out-of-range panic,
  empty-delay-state panic in debug.
- **3 no-regression integration tests** (`tests/conformal_karc_no_regression.rs`):
  bit-identical forecasts, `wout` stability, FourierBasis smoke.
- **1 example** (`examples/conformal_karc_overlay.rs`): Lorenz-63 coverage.
- **All 24 Phase 1 tests still pass** (G1/G3/G4 gates unchanged).

### Total test count

- Phase 1: 24 unit + 8 integration = 32
- Phase 2: 4 unit + 3 integration = 7
- **Grand total: 39 tests, all GREEN.**

---

## References

- **CSP paper:** [arXiv:2605.03789](https://arxiv.org/abs/2605.03789)
- **"Report the Floor":** [arXiv:2606.09473](https://arxiv.org/abs/2606.09473)
- **Plan 340:** [`.plans/340_conformal_predictive_intervals_primitive.md`](../.plans/340_conformal_predictive_intervals_primitive.md)
- **Issue 010:** RESOLVED; tracker removed. See `.benchmarks/010_report_the_floor_consolidated.md` for the lasting audit record.
- **Research 322:** `.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md`
