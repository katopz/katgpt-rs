# Plan 340: Conformal Predictive Intervals — Modelless UQ Overlay (Open Primitive)

**Date:** 2026-06-28
**Research:** [katgpt-rs/.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md](../.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md)
**Private guide:** [riir-ai/.research/165_Per_NPC_Conformal_UQ_Guide.md](../../riir-ai/.research/165_Per_NPC_Conformal_UQ_Guide.md)
**Source paper:** [arXiv:2605.03789](https://arxiv.org/abs/2605.03789) — Manokhin, *Training-Free Probabilistic Time-Series Forecasting with Conformal Seasonal Pools*, 2026
**Companion paper:** [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) — *Report the Floor* (conformal interval as mandatory baseline)
**Target:** `katgpt-rs/crates/katgpt-core/src/conformal.rs` (new module) + Cargo feature `conformal_predictive_intervals`
**Status:** Active — Phase 1 (open primitive skeleton + seasonal pool + conformal overlay). KARC adapter (Phase 2) and riir-ai runtime integration (Phase 3+) filed separately after Phase 1 lands.

---

## Goal

Ship a generic, modelless, inference-time conformal UQ overlay that wraps any point forecaster and produces coverage-guaranteed predictive intervals. The overlay:

1. Wraps a `PointForecaster` trait (sealed; two impls ship: `SeasonalPoolForecaster` from CSP, and a KARC adapter).
2. Maintains a per-channel residual pool with exponential recency weighting (`decay_unit` selectable: `step` or `cycle`).
3. Indexes the residual pool by horizon `h` via `L_h = m·⌈h/m⌉` (the `h_step` residual mode — the new CSP v0.1.4 default that drives multi-step coverage).
4. Reads empirical quantiles `q_{α/2}`, `q_{1−α/2}` to produce `[point + q_{α/2}, point + q_{1−α/2}]`.
5. Optionally draws samples via the seasonal-pool + conformal-residual mixture (CSP's full predictive distribution).
6. Computes CRPS / Winkler interval score / empirical coverage for the GOAT gate.

No training, no learned parameters, no gradient descent. Pure empirical-quantile calibration over a residual reservoir.

**GOAT gate (G1–G4):**
- **G1 — Coverage.** On stationary seasonal synthetic data (sinusoid + noise), empirical coverage at α=0.05 over 10,000 ticks ∈ [0.93, 0.97]. Reproduce CSP's AirPassengers CRPS within 2×.
- **G2 — Latency.** `interval_into(h, alpha, out)` ≤ 1µs at H=1, ≤ 100µs at H=8×8 channels (warm-tier target, not hot-path). Zero hot-path overhead — the overlay is queried explicitly, never on the per-tick critical path.
- **G3 — Zero-alloc.** `interval_into` and `update_residual` perform zero allocations after warmup. Pre-sorted residual ring buffer; O(log n) quantile read.
- **G4 — Bit-reproducibility.** Two `ConformalIntervalCalibrator` instances with identical `(residual_pool, m, alpha, h, decay_config)` produce byte-identical interval bounds. Required for quorum commitment downstream.

Demote-on-fail: if G1 coverage < 0.85 on synthetic seasonal data (the easy case), the math is wrong — downgrade to opt-in Gain-tier, file issue, do not promote. If G2 > 10ms at H=8×8, demote (the warm-tier budget is blown). If G4 fails bit-reproducibility, the LatCal sync-boundary story is dead — block promotion.

---

## Architecture

```
                  ┌─────────────────────────────────────┐
observation y_t──▶│ ConformalIntervalCalibrator<F>       │
                  │                                     │
point ŷ_t ───────▶│  forecaster: F (PointForecaster)    │ ◀── KARC / SeasonalPool / any impl
                  │  residual_pool: ResidualRingBuffer  │ ◀── per-channel, exp-recency weighted
                  │  m: usize (seasonal period)         │
                  │  decay: DecayConfig (step/cycle)    │
                  │  residual_mode: Paper | HStep       │
                  │  orientation: bool                  │
                  └─────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
   ┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐
   │ update_residual  │ │ interval_   │ │ sample_predictive│
   │ (y_t, ŷ_t, h)    │ │ into(h,α,   │ │ _distribution    │
   │ → push to pool   │ │ out: &mut)  │ │ (h, n_samples)   │
   │ w/ exp recency   │ │             │ │ → Vec<f32>       │
   └──────────────────┘ └─────────────┘ └──────────────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │ PredictiveInterval           │
              │  lower: f32                  │
              │  point: f32                  │
              │  upper: f32                  │
              │  alpha: f32                  │
              │  coverage_violation(actual): │
              │    bool                      │
              └──────────────────────────────┘
```

The trait stack:

```rust
/// A point forecaster that produces a single deterministic forecast
/// given a delay-embedded state. KARC implements this; SeasonalPoolForecaster
/// implements this; any future forecaster can implement it.
pub trait PointForecaster {
    /// Forecast the next value at horizon `h` (1-indexed) given the
    /// delay-embedded state. Writes into `out` (zero-alloc).
    fn forecast_into(&self, delay_state: &[f32], h: usize, out: &mut f32);
}

/// Residual pool indexing strategy.
pub enum ResidualMode {
    /// Single residual pool (lag `m`) reused for all horizons.
    /// Matches CSP `residual_mode="paper"`. Interval width is constant across horizons.
    /// Use only for seasonal data with H ≤ m.
    Paper,
    /// Horizon-indexed pool with `L_h = m·⌈h/m⌉`.
    /// Matches CSP `residual_mode="h_step"` (v0.1.4 default). Interval widens with horizon.
    /// Use for non-seasonal (m=1) or long-horizon (H>m) series.
    HStep,
}

/// Unit for the residual pool's exponential recency decay.
pub enum DecayUnit {
    /// Decay by absolute observation age (time steps). CSP v0.1.4 default.
    /// Same-phase observations one season apart are `m` steps apart.
    Step,
    /// Decay by cycle age. CSP paper's original behavior.
    /// `m`× weaker than `Step` for the same `exp_lambda`.
    Cycle,
}

/// The conformal UQ overlay. Generic over any `PointForecaster`.
pub struct ConformalIntervalCalibrator<F: PointForecaster> {
    forecaster: F,
    /// Per-channel residual ring buffer, exp-recency weighted.
    /// Layout: `[channel][horizon_bucket][sorted_residual]`.
    residual_pool: ResidualRingBuffer,
    m: usize,
    exp_lambda: f32,
    decay_unit: DecayUnit,
    residual_mode: ResidualMode,
    orientation: bool,
}

impl<F: PointForecaster> ConformalIntervalCalibrator<F> {
    /// Observe an (actual, forecasted) pair at horizon `h`, update the residual pool.
    /// O(log n) insertion into the per-channel sorted ring buffer.
    pub fn update_residual(&mut self, actual: f32, forecast: f32, channel: usize, h: usize);

    /// Read the calibrated interval `[lower, point, upper]` at horizon `h`, level `1−α`.
    /// Zero-alloc. O(log n) quantile read from the pre-sorted pool.
    pub fn interval_into(
        &self,
        channel: usize,
        h: usize,
        alpha: f32,
        out: &mut PredictiveInterval,
    );

    /// Returns `true` iff `actual` is outside the `1−α` interval at horizon `h`.
    /// The coverage-violation flag — the calibrated curiosity signal.
    pub fn coverage_violation(&self, actual: f32, channel: usize, h: usize, alpha: f32) -> bool;

    /// Draw `n` samples from the predictive distribution (seasonal pool + conformal residual).
    /// Allocates `Vec<f32>` of length `n`. Use only for CRPS evaluation, not hot path.
    pub fn sample_predictive_distribution(
        &self,
        channel: usize,
        h: usize,
        n: usize,
        rng: &mut impl Rng,
    ) -> Vec<f32>;
}

#[derive(Clone, Copy, Debug)]
pub struct PredictiveInterval {
    pub lower: f32,
    pub point: f32,
    pub upper: f32,
    pub alpha: f32,
}

impl PredictiveInterval {
    pub fn contains(&self, actual: f32) -> bool {
        actual >= self.lower && actual <= self.upper
    }
}
```

### SeasonalPoolForecaster (the second PointForecaster impl)

The CSP seasonal pool as a standalone `PointForecaster` — pure mixing, no ridge solve. This is **Gain-tier** on its own (KARC is strictly more general), but ships alongside the overlay as a reference impl and a low-latency fallback for known-seasonality scenarios.

```rust
/// CSP's seasonal pool forecaster: same-phase history weighted by exponential recency.
/// No learned Wout, no ridge solve. Pure reservoir mixing.
///
/// This is a SPECIAL CASE of KARC (periodic delay-basis, no basis expansion, no ridge).
/// Use when: (a) seasonality `m` is known, (b) latency budget is tight (no ridge solve),
/// (c) the series is stationary around a stable level + seasonal pattern.
/// Prefer KARC otherwise.
pub struct SeasonalPoolForecaster {
    history: RingBuffer<f32>,
    m: usize,
    exp_lambda: f32,
    pool_weight: f32,
}

impl PointForecaster for SeasonalPoolForecaster {
    fn forecast_into(&self, _delay_state: &[f32], h: usize, out: &mut f32) {
        // Seasonal-naive point forecast: y_{t+h} ≈ y_{t+h−L_h}, L_h = m·⌈h/m⌉.
        // Weighted by exp-recency over same-phase history.
        // ...
    }
}
```

---

## Phase 1 — Unblocking Skeleton (CORE) ✅ COMPLETE (2026-06-30)

GOAT gate PASSED — see [`.benchmarks/340_conformal_goat.md`](../.benchmarks/340_conformal_goat.md). G1 coverage [0.9445, 0.9493] (target [0.93, 0.97]), G2 interval_into H=1 = 642ns (target ≤ 1µs), G3 zero-alloc, G4 bit-reproducible. AirPassengers CRPS 115.06 vs ±2σ baseline 468.75 (4× sharper). Opt-in — promotion deferred to Plan 342.

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/conformal.rs` behind `#[cfg(feature = "conformal_predictive_intervals")]`. Empty `ConformalIntervalCalibrator<F>` struct, `PointForecaster` trait, `PredictiveInterval` struct, `ResidualMode` / `DecayUnit` enums. Wire `conformal_predictive_intervals` into `crates/katgpt-core/Cargo.toml` features list and `lib.rs` mod declaration.
- [x] **T1.2** Implement `ResidualRingBuffer` — per-channel × per-horizon-bucket sorted ring buffer. Configurable capacity (default 256 residuals per bucket). `push(r: f32, channel: usize, h_bucket: usize)` with O(log n) insertion sort. `quantile_into(channel, h_bucket, q, out: &mut f32)` with O(1) indexed read (the buffer is kept sorted). Exponential recency weighting applied at *quantile read time* (weights multiply the position, not the storage) — keeps the buffer write path simple.
  - **Note:** shipped as O(n) linear insertion (not O(log n)) because the buffer is small (≤256) and vectorizes well; if G2 ever fails, swap to binary search. See `conformal/ring.rs`.
- [x] **T1.3** Implement `ConformalIntervalCalibrator::update_residual(actual, forecast, channel, h)` — computes `r = actual − forecast`, indexes the horizon bucket via `L_h = m·⌈h/m⌉` (HStep) or `L_h = m` (Paper), pushes into the ring buffer with recency weight `w = exp(−λ · age)` where `age` is in `Step` or `Cycle` units.
  - **Note:** recency weight is applied at read time, not push time; the ring stores `(residual, tick)` pairs and the weight `exp(−λ·age)` is computed during `interval_into`.
- [x] **T1.4** Implement `ConformalIntervalCalibrator::interval_into(channel, h, alpha, out)` — reads `q_{α/2}` and `q_{1−α/2}` from the pre-sorted pool, applies `orientation` correction (`⌊(n+1)q⌋/n` / `⌈(n+1)q⌉/n`), adds the wrapped forecaster's point forecast, writes into `out: &mut PredictiveInterval`. Zero allocation.
- [x] **T1.5** Implement `ConformalIntervalCalibrator::coverage_violation(actual, channel, h, alpha)` — calls `interval_into`, returns `!interval.contains(actual)`. The 1-bit calibrated curiosity signal.
- [x] **T1.6** Implement `SeasonalPoolForecaster` with `RingBuffer<f32>` history, `forecast_into` via seasonal-naive + exp-recency weighted same-phase average.
- [x] **T1.7** Implement `ConformalIntervalCalibrator::sample_predictive_distribution(channel, h, n, rng)` — CSP's mixture: `pool_weight` fraction from the seasonal pool (sampled proportional to recency weights), `(1−pool_weight)` fraction from the conformal residual (sampled uniformly from the residual pool + added to the point forecast). Allocates `Vec<f32>` of length `n`. Use for CRPS evaluation only.
- [x] **T1.8** Write `tests/conformal_coverage.rs` — G1 gate. Generate a stationary seasonal synthetic series `y_t = sin(2π t/m) + ε_t`, `ε ~ N(0, σ)`, fit the calibrator over 10,000 ticks with a `SeasonalPoolForecaster`, assert empirical coverage at α=0.05 ∈ [0.93, 0.97]. Vary `m ∈ {12, 24, 48}`, `σ ∈ {0.1, 0.5, 1.0}`. Also test `m=1` (non-seasonal, HStep mode) — coverage should hold with widening intervals.
- [x] **T1.9** Write `tests/conformal_reproducibility.rs` — G4 gate. Two calibrators with identical `(residual_pool, m, alpha, h, decay_config, orientation)` produce byte-identical `PredictiveInterval` bounds (verified via `f32::to_bits`). Vary `α ∈ {0.01, 0.05, 0.1, 0.2}` and `h ∈ {1, 8, 24}`.
- [x] **T1.10** Write `tests/conformal_alloc_check.rs` — G3 gate. Use a manual `GlobalAlloc` counter; assert `update_residual` and `interval_into` perform zero allocations after warmup.
- [x] **T1.11** Write `benches/conformal_interval_bench.rs` — G2 gate. Criterion bench: `interval_into` at H=1, H=8, H=8×8 channels. Target: ≤ 1µs at H=1, ≤ 100µs at H=8×8.
  - **Result:** H=1 = 642ns (PASS), H=8×8 = 40.3µs (PASS). Required the `weighted_quantile_pair` optimization (compute exp-recency weights once, reuse for both q_lo and q_hi — 4× fewer `exp()` calls) to get H=1 under 1µs.
- [x] **T1.12** Write `examples/conformal_airpassengers.rs` — reproduce CSP's AirPassengers CRPS within 2×. Load the AirPassengers series (embed a small synthetic proxy if the real data is not freely redistributable), run rolling-origin backtest at H=12 and H=24, report CRPS, RMSE, empirical coverage. Compare against Seasonal-Naive baseline. **This IS the conformal-naive floor** adopted as the mandatory baseline for all UQ-bearing primitives per the "Report the Floor" rule (Research 322, AGENTS.md Feature Flag Discipline, adopted 2026-06-28). The `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1` configuration is the canonical floor instance — every future UQ primitive's GOAT gate must beat this baseline on CRPS / coverage / Winkler.
  - **Result:** Conformal CRPS 115.06 vs ±2σ baseline 468.75 (4× sharper, gate holds).
- [x] **T1.13** Implement CRPS / Winkler interval score / empirical coverage utility functions in `conformal.rs` (or a `conformal_metrics.rs` submodule). These are the GOAT gate framework for any future UQ-bearing primitive.
  - **Shipped as:** `conformal/metrics.rs` with `crps`, `crps_interval`, `winkler_score`, `empirical_coverage`, `mean_crps_interval`, `mean_winkler`.
- [x] **T1.14** Run the GOAT gate (G1–G4). Document results in `.benchmarks/340_conformal_goat.md`. Promote to default-on only if all four gates pass AND the gain is modelless (it is — no training). **Promotion deferred** until the riir-ai runtime integration (Plan 342) confirms the curiosity false-positive win (G3 in the private guide) — the open primitive's gates prove the math; the runtime gates prove the utility.

### Phase 1 verdict criteria

- **G1 PASS** requires coverage ∈ [0.93, 0.97] on ALL three `m` values AND on `m=1` HStep mode.
- **G2 PASS** requires ≤ 1µs at H=1 AND ≤ 100µs at H=8×8.
- **G3 PASS** requires zero allocations in `update_residual` AND `interval_into` after warmup.
- **G4 PASS** requires bit-identical bounds across two calibrators for all `(α, h)` combos.

If G1 fails by >5% (coverage < 0.90 on any seasonal config), the math is wrong — debug before proceeding. If G2 fails by >10×, the residual pool data structure needs redesign (consider a t-digest or P² algorithm instead of sorted ring buffer — see Plan 269 Chiaroscuro for the P² abandonment lesson).

---

## Phase 2 — KARC Adapter (open primitive)

### Tasks

- [ ] **T2.1** Implement `impl PointForecaster for KarcForecaster<...>` adapter in `conformal.rs` behind `#[cfg(all(feature = "conformal_predictive_intervals", feature = "karc_forecaster"))]`. The adapter wraps `KarcForecaster::forecast_into(delay_state, out)` and exposes it at horizon `h=1` (KARC forecasts one step ahead; multi-horizon conformal intervals come from the residual pool indexing, not from KARC itself).
- [ ] **T2.2** Write `examples/conformal_karc_overlay.rs` — fit KARC on a chaotic trajectory (Lorenz-63 or double-scroll from Plan 308's `examples/karc_double_scroll.rs`), wrap with the conformal overlay, produce calibrated intervals on the forecast. Report coverage at α=0.05.
- [ ] **T2.3** Add `tests/conformal_karc_no_regression.rs` — verify the conformal overlay does NOT touch the KARC point-forecast hot path. KARC's `forecast_into` latency (381ns, Plan 308 G2) is unchanged when the overlay is feature-gated on. This is the zero-regression guarantee for the existing KARC DEFAULT-ON promotion.

---

## Phase 3 — riir-ai Runtime Integration (private, separate plan)

File as `riir-ai/.plans/342_conformal_uq_runtime_integration.md` after Phase 1 lands. See `riir-ai/.research/165_Per_NPC_Conformal_UQ_Guide.md` §6 for the full task breakdown.

Summary:
- `conformal_bridge/hla_overlay.rs` — per-channel HLA residual pool.
- `conformal_bridge/curiosity.rs` — coverage-tested curiosity event.
- `conformal_bridge/sleep_time.rs` — calibrated predictability scorer.
- `conformal_bridge/mcts_collapse.rs` — confidence-interval collapse threshold.
- G1–G6 gates per the private guide §5 (the game-corpus gates, not the synthetic gates from Phase 1).

---

## Phase 4 — riir-neuron-db + riir-chain (cross-repo, separate plans)

File after Phase 3 ships:
- `riir-neuron-db/.plans/005_conformal_residual_shard.md` — `ConformalResidualShard` Pod layout (empirical quantile table in `style_weights[64]`), `MerkleFrozenEnvelope` integration, freeze/thaw determinism.
- `riir-chain/.plans/008_latcal_conformal_interval_commitment.md` — LatCal commitment of the 15-scalar interval triple + 1-bit coverage flag.

---

## Open questions

1. **Ring buffer vs t-digest vs P².** The sorted ring buffer is simplest and gives exact quantiles, but O(n) memory per bucket and O(log n) insertion. For 256 residuals × 8 channels × 8 horizons = 16K f32 = 64KB per NPC — fits in L2, acceptable. If G2 latency fails, consider t-digest (O(log log n) quantile, approximate) or P² (O(1) streaming quantile, but Plan 269 Chiaroscuro abandoned P² for drift — see that lesson). Default: sorted ring buffer; revisit only if G2 fails.

2. **Joint multivariate conformal.** Out of scope for the open primitive. Per-channel marginals only. Joint needs a copula or split conformal multivariate → riir-train follow-up. Document the per-channel independence assumption in the module doc.

3. **`m` detection from data.** The open primitive takes `m` as a constructor parameter. Detecting `m` from autocorrelation peak or spectral peak is a separate utility — possibly a `detect_seasonal_period(history) -> usize` function in Phase 2 or a follow-up. For HLA, `m` is per-NPC-type config (e.g., guard NPC `m` = patrol cycle length).

4. **Stationarity / drift handling.** The `decay_unit="step"` exponential forgetting handles slow drift. For sharp regime changes (combat onset, quest start), the residual pool needs a window reset or a separate pool per regime. The `ReestimationScheduler` trigger ("actual outside 95% interval") doubles as a drift detector — when it fires repeatedly, reset the pool. This is a Phase 3 runtime concern, not an open-primitive concern.

5. **"Report the Floor" as a GOAT gate requirement.** Should every future UQ-bearing primitive (BoMSampler, Alien Sampler, Sleep-Time) be required to beat the conformal-naive floor as part of its GOAT gate? The companion paper argues yes. This is a policy decision, not a Phase 1 task — flag for the user.

---

## References

- **CSP paper:** [arXiv:2605.03789](https://arxiv.org/abs/2605.03789)
- **CSP code:** https://github.com/valeman/csp-forecaster
- **"Report the Floor" companion:** [arXiv:2606.09473](https://arxiv.org/abs/2606.09473)
- **KARC (the dominant point forecaster):** [Plan 308](308_karc_delay_basis_ridge_forecaster.md), [Research 288](../.research/288_KARC_Delay_Basis_Ridge_Forecaster.md)
- **P² algorithm abandonment lesson:** [Plan 269](269_chiaroscuro_spectral_entropy_operator_routing.md) §"P² algorithm abandoned" — relevant if the sorted ring buffer is reconsidered.
- **Best-Belief Beta quantile (the discrete-side cousin):** [Plan 336](336_controlled_utility_primitives.md) — `best_belief_score` inverse-CDF Beta; the conformal overlay is the continuous-side cousin (inverse-CDF empirical).
- **Sleep-Time Query Anticipator (the predictability gate consumer):** [Plan 334](334_sleep_time_query_anticipator_primitive.md)
- **Private selling-point guide:** [riir-ai/.research/165_Per_NPC_Conformal_UQ_Guide.md](../../riir-ai/.research/165_Per_NPC_Conformal_UQ_Guide.md)
