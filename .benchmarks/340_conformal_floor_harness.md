# Plan 340 Phase 2.5 GOAT Gate — "Report the Floor" Comparison Harness

**Date:** 2026-06-30
**Plan:** [`.plans/340_conformal_predictive_intervals_primitive.md`](../.plans/340_conformal_predictive_intervals_primitive.md) Phase 2.5
**Issue:** 010 (T2 — RESOLVED; tracker removed; this benchmark + `.benchmarks/010_report_the_floor_consolidated.md` are the lasting record)
**Feature flag:** `conformal_predictive_intervals` (opt-in, same as Phase 1+2)
**Modelless:** ✅ Yes — the harness only orchestrates and scores. No training, no learned params.

---

## TL;DR

**Issue 010 T2: ✅ COMPLETE.** The floor-comparison harness ships as
`crates/katgpt-core/src/conformal/floor_harness.rs`. It is the enforcement
substrate for the "Report the Floor" policy: any UQ-bearing primitive can now
be evaluated against the canonical conformal-naive floor
(`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` m=1) via a single
`run_floor_comparison` call.

**Test count:** 13 unit + 10 integration = **23 new tests, all GREEN**. Total
conformal test surface (Phase 1 + 2 + 2.5): **62 tests**.

**No regressions:** all Phase 1 (32) and Phase 2 (7) conformal tests still
pass; KARC no-regression gate still passes.

---

## G1 — Harness correctness ✅ PASS

**Gate:** The harness correctly identifies BeatsFloor / TiesFloor /
LosesToFloor / Mixed / NotApplicable across 5 canonical scenarios on standard
corpora.

**Tests:** `tests/conformal_floor_harness.rs` (10 integration tests) +
`src/conformal/floor_harness.rs::tests` (13 unit tests).

### Reference results (α=0.05, deterministic seeds)

#### Scenario 1: True oracle (peek at next value) on stationary seasonal

```
=== Floor Comparison: true-oracle ===
Corpus: stationary_seasonal_m12_sigma0.5_n200 (n_scored=152, α=0.05)
Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict
Mean CRPS          |     0.0000 |     3.4171 |             0.0000 | WIN
Mean Winkler       |     0.0000 |     4.0295 |             0.0000 | WIN
Coverage (nom=0.95) |     1.0000 |     0.9605 | err 0.0500 vs 0.0105 | LOSE
Overall: ✅ BEATS FLOOR — primitive adds UQ value
```

**Note:** coverage shows LOSE (oracle over-covers at 1.0000 vs nominal 0.95),
but the verdict is correctly `BeatsFloor` because over-coverage is acceptable
per the harness's coverage policy — the extra width is already penalized via
CRPS (here, CRPS is vanishingly small because the oracle's interval is ±1e-6).
This validates the policy: only **under-coverage** fails the gate.

#### Scenario 2: Over-wide (±10) on stationary seasonal

```
=== Floor Comparison: over-wide (±10) ===
Corpus: stationary_seasonal_m12_sigma0.5_n200 (n_scored=152, α=0.05)
Metric             | Primitive  | Floor      | Ratio (prim/floor) | Verdict
Mean CRPS          |    20.0000 |     3.4171 |             5.8530 | LOSE
Mean Winkler       |    20.0000 |     4.0295 |             4.9635 | LOSE
Coverage (nom=0.95) |     1.0000 |     0.9605 | err 0.0500 vs 0.0105 | LOSE
Overall: ❌ LOSES TO FLOOR — primitive does not add UQ value
```

#### Scenario 3: Empty primitive (no output)

```
=== Floor Comparison: empty ===
Corpus: stationary_seasonal_m12_sigma0.5_n200 (n_scored=0, n_unscorable=190, α=0.05)
Overall: ⚪ N/A — primitive 'empty' produced no scorable output on corpus 'stationary_seasonal_m12_sigma0.5_n200'
```

#### Scenario 4: Mean-tracker on white noise (floor's worst case)

On i.i.d. white noise, the optimal forecast is the MEAN (not the last value).
The floor (seasonal-naive = last value) is worst-case here; the mean-tracker
decisively beats it.

**Gate assertion:** `crps_ratio < 0.9` ✅ (the mean-tracker wins)

#### Scenario 5: Mean-tracker on seasonal data (floor wins)

On seasonal data, the mean-tracker misses the seasonal structure entirely
(predicts the global mean for every step). The floor captures the structure
and wins.

**Gate assertion:** `crps_ratio > 2.0` ✅ (the floor wins)

This pair (Scenarios 4 + 5) is the **policy-critical validation**: the same
primitive can BeatsFloor on one corpus and LosesToFloor on another. The
"Report the Floor" evaluation protocol requires a multi-corpus sweep, not a
single-corpus verdict.

---

## G2 — Determinism ✅ PASS

**Gate:** Identical `(primitive, corpus, alpha, warmup)` → identical report.

**Tests:** `corpus_stationary_seasonal_is_deterministic`,
`corpus_white_noise_is_deterministic`.

The corpus constructors use a deterministic SplitMix64 RNG (same algorithm as
`examples/conformal_airpassengers.rs`), so corpora are bit-reproducible across
runs. Given a fixed primitive, the report is deterministic.

---

## G3 — Feature matrix ✅ PASS

| Build config | Result |
|---|---|
| `cargo check -p katgpt-core` (default features) | ✅ clean (harness is opt-in, not compiled) |
| `cargo check -p katgpt-core --all-features` | ✅ clean |
| `cargo check -p katgpt-core --no-default-features --features conformal_predictive_intervals` | ✅ clean |
| `cargo check -p katgpt-core --features conformal_predictive_intervals` | ✅ clean |

Zero-overhead when the feature is off: the module is `#[cfg(feature =
"conformal_predictive_intervals")] mod floor_harness;` and not even compiled
into the default build.

---

## G4 — No regression ✅ PASS

| Existing gate | Before Phase 2.5 | After Phase 2.5 |
|---|---|---|
| Phase 1 unit tests (`conformal::*`) | 24 pass | 24 pass ✅ |
| Phase 1 integration tests (`conformal_coverage/reproducibility/alloc_check`) | 8 pass | 8 pass ✅ |
| Phase 2 unit tests (`karc_adapter`) | 4 pass | 4 pass ✅ |
| Phase 2 integration tests (`conformal_karc_no_regression`) | 3 pass + 1 ignored | 3 pass + 1 ignored ✅ |
| KARC lib tests (`karc::*`) | 23 pass | 23 pass ✅ |

No existing test broke. The harness is a pure addition — it consumes the
floor and metrics modules without modifying them.

---

## The harness API

### `UqPrimitiveUnderTest` trait

```rust
pub trait UqPrimitiveUnderTest {
    fn name(&self) -> &str;
    fn predict_next(&mut self) -> PredictiveOutput;  // BEFORE observe
    fn observe(&mut self, y: f32);                   // AFTER predict_next
}
```

### `run_floor_comparison` entry point

```rust
pub fn run_floor_comparison<P: UqPrimitiveUnderTest>(
    primitive: &mut P,
    corpus: &[f32],
    alpha: f32,
    warmup: usize,
    corpus_name: &str,
) -> FloorComparisonReport
```

### `PredictiveOutput` — samples, interval, or both

```rust
pub struct PredictiveOutput {
    pub samples: Option<Vec<f32>>,
    pub interval: Option<PredictiveInterval>,
}
```

The harness normalizes samples → interval via `empirical_quantile_interval`
(type-7 quantile, R default) so both primitive and floor are scored on the
same interval metrics. This is how a samples-only primitive like BoMSampler
will be scored in T3.

### `OverallVerdict` — the policy substrate

```rust
pub enum OverallVerdict {
    BeatsFloor,                           // >5% better on ≥1 metric, no loss
    TiesFloor,                            // within ±5% on all metrics
    LosesToFloor,                         // >5% worse on ≥1 metric, no win
    Mixed,                                // better on some, worse on others
    NotApplicable { reason: String },     // no scorable output
}
```

**Coverage policy (the load-bearing design decision):** over-coverage is
ACCEPTABLE. A primitive that covers more than nominal (e.g., the true oracle
at coverage=1.0) is NOT penalized — the extra width is already captured by
CRPS. Only **under-coverage** (false confidence) fails the gate. This is why
`coverage_ok = primitive_coverage >= floor_coverage − COVERAGE_TOL` (one-sided
check), not a two-sided `|primitive_cov_err| <= |floor_cov_err| + tol`.

The verdict is a **hint, not a judgment**. The harness exposes the raw metrics
+ ratios so a human (or the T3–T7 adapter author) can overrule it — e.g., a
primitive that ties on CRPS but is 10× faster is still valuable (the policy's
"reframing" escape hatch per Issue 010 §"Failure mode").

---

## Standard corpora

| Constructor | Formula | Floor's expected behavior | Recommended use |
|---|---|---|---|
| `stationary_seasonal(m, σ, n, seed)` | `y_t = sin(2πt/m) + N(0,σ)` | Floor (m=1) is suboptimal — it ignores the seasonality. A primitive that captures m should beat it. | Default corpus for primitives with seasonal structure. |
| `white_noise(σ, n, seed)` | `y_t ~ N(0,σ)` i.i.d. | Floor is WORST-CASE (forecast=last value; optimal=mean). A mean-tracking primitive should beat it. | Stress test — separates "captures the mean" primitives from "follows the noise" primitives. |
| `from_slice(name, values, warmup)` | arbitrary | depends on data | Lorenz-63, real trajectories, multi-channel projections. |

All constructors use a deterministic SplitMix64 RNG for bit-reproducible
corpora across runs.

---

## What ships

| File | Role | Lines |
|---|---|---|
| `src/conformal/floor_harness.rs` | Harness module: trait, FloorAdapter, report types, `run_floor_comparison`, corpora, 13 unit tests | ~650 |
| `tests/conformal_floor_harness.rs` | 10 integration tests + canonical adapter-pattern examples for T3–T7 authors | ~375 |
| `src/conformal/mod.rs` | Wire `mod floor_harness` + re-export | +9 lines |
| `src/lib.rs` | Re-export harness types at crate root | +7 lines |
| `Cargo.toml` | `[[test]] conformal_floor_harness` entry | +8 lines |

---

## Unblocks

- **Issue 010 T3** — BoMSampler floor comparison. Adapter implements
  `UqPrimitiveUnderTest`, wraps `BoMSampler::sample_k_states` output as
  `PredictiveOutput::from_samples`, calls `run_floor_comparison`.
- **Issue 010 T4** — Sleep-Time Anticipator floor comparison. Adapter wraps
  the predictability scorer + anticipate/consume lifecycle.
- **Issue 010 T5** — Best-Belief Beta Selector floor comparison. Adapter wraps
  `best_belief_score` as an inverse-CDF read, produces an interval from the
  Beta ε-quantile.
- **Issue 010 T6** — Alien Sampler decision (UQ-bearing or not?) + comparison
  if applicable.
- **Issue 010 T7** — Document results across all primitives in this folder.

Each T3–T7 task now reduces to ~50 lines of adapter code + one
`run_floor_comparison` call.

---

## References

- **"Report the Floor" policy:** `katgpt-rs/AGENTS.md` Feature Flag Discipline, Issue 010.
- **Companion paper:** [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) — *Report the Floor*.
- **Plan 340:** [`.plans/340_conformal_predictive_intervals_primitive.md`](../.plans/340_conformal_predictive_intervals_primitive.md) Phase 2.5.
- **Phase 1 GOAT:** [`.benchmarks/340_conformal_goat.md`](340_conformal_goat.md).
- **Phase 2 GOAT:** [`.benchmarks/340_conformal_goat.md`](340_conformal_goat.md) §"Phase 2 update".
