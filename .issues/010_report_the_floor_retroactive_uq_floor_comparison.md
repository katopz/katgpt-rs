# Issue 010: Retroactive "Report the Floor" — conformal-naive floor comparison for existing UQ-bearing primitives

**Filed:** 2026-06-28
**Last updated:** 2026-07-01 (T4 shipped — Sleep-Time Anticipator EXCLUDED; T5 shipped — Best-Belief BEATS floor; T6 shipped — Alien Sampler EXCLUDED)
**Policy source:** Research 322 (`.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md`), Plan 340 (`.plans/340_conformal_predictive_intervals_primitive.md`), `katgpt-rs/AGENTS.md` Feature Flag Discipline, research skill `SKILL.md` §Workflow 2.
**Companion paper:** *Report the Floor* (arXiv:2606.09473) — argues a training-free conformal interval is a mandatory baseline for any probabilistic forecaster.
**Blocking dependency:** ✅ RESOLVED 2026-06-30 — Plan 340 Phase 1 shipped `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (the floor instance) behind `conformal_predictive_intervals`. GOAT gate PASSED (see `.benchmarks/340_conformal_goat.md`). Plan 340 Phase 2 additionally shipped the `KarcChannelForecaster` adapter + the Lorenz-63 coverage demonstration (x=0.9425, y=0.9520, z=0.9485 at α=0.05). The retroactive comparison work (T2–T7) is now ACTIONABLE.

---

## Context

On 2026-06-28, the "Report the Floor" policy was adopted as a GOAT gate extension for all UQ-bearing primitives: any primitive claiming a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty must beat the conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1`) on CRPS / coverage / Winkler score. If it can't beat the floor, it's noise, not UQ.

The policy applies **prospectively** to all future UQ primitives (enforced from initial GOAT gate) and **retroactively** to existing UQ-bearing primitives (grandfathered at current promotion state, but must include the floor at their next re-gate or feature-touch). This issue tracks the retroactive work.

## UQ-bearing primitives requiring retroactive floor comparison

| Primitive | Plan | Current state | Floor comparison needed at |
|---|---|---|---|
| **BoMSampler** | 281 | shipped | **DONE 2026-06-30 — EXCLUDED** (see T3). The comparison was run; BoM's hypothesis spread is exploration noise (σ-controlled), not calibrated UQ. False-confidence signature: wins CRPS (0.87/0.31 ratio) but covers only 5–15% vs nominal 95%, Winkler 4–14× the floor. Excluded per T3 escape hatch; `bom_sampling` stays DEFAULT-ON (its GOAT gate is planning quality, not UQ). See `.benchmarks/010_bom_floor_comparison.md`. |
| **Sleep-Time Query Anticipator** | 334 (open) / 341 (riir-ai runtime) | shipped | **DONE 2026-07-01 — EXCLUDED** (see T4). Same false-confidence signature as BoM: predictability-derived intervals win CRPS (0.55–0.63 ratio) but lose coverage (37–54% vs nominal 95%) and Winkler (2.5–3.4× the floor). T4-specific difficulty-correlation test shows near-zero per-step correlation for BOTH primitives (the floor achieves coverage by marginal calibration, not per-step tracking). The anticipator's value is amortized compute gating, not calibrated UQ — `sleep_time_anticipation` stays OPT-IN (unchanged). See `.benchmarks/010_sleep_time_floor_comparison.md`. |
| **Best-Belief Beta Selector** | 336 | shipped (DEFAULT-ON, Plan 336 Phase 2) | **DONE 2026-07-01 — BEATS FLOOR** (see T5). The first primitive to genuinely beat its floor. Selection-regret comparison: at uniform n, Beta TIES MLE (monotonicity theorem — `BB_ε(S,n−S)` is monotone in S/n, same argmax). At variable n (heteroscedastic — the real-world case), Beta WINS by 15–30% regret reduction (and 61–77% on the low-data false-positive stress test, n_lo=2). `best_belief` stays DEFAULT-ON. See `.benchmarks/010_best_belief_floor_comparison.md`. |
| **KARC + conformal overlay** | 308 + 340 | KARC shipped (DEFAULT-ON); overlay in Plan 340 | Plan 340 Phase 2 (KARC adapter) — the overlay itself defines the floor, so this is the reference, not a comparison target |
| **Alien Sampler** | 311 | shipped (GOAT 1/4 FAIL, opt-in) | **DONE 2026-07-01 — EXCLUDED** (see T6). Decision: NOT UQ-bearing. It produces a within-pool z-scored ranking (relative selection signal), not a probability distribution / interval / quantile. Its GOAT gate measures motif-collapse reduction (diversity), not coverage/CRPS/Winkler. Same structural exclusion as BoM (planning) and Sleep-Time (gating). |

## Borderline / excluded primitives (for clarity)

These are **not** UQ-bearing under the policy definition (they don't claim a distribution / interval / quantile / confidence):

- KARC point forecast alone (Plan 308) — single point, no uncertainty claim. Becomes UQ-bearing only when wrapped by the conformal overlay (Plan 340 Phase 2).
- Constraint pruners, bandits, DDTree, speculative decode — these claim validity/relevance/reward, not calibrated uncertainty.
- Salience Tri-Gate (Plan 303) — discrete Speak/Silent/Delegate decision, not a distribution. (Though it *consumes* UQ from the conformal overlay.)
- CGSP runtime curiosity — currently a magnitude (not calibrated). **After** Plan 340 integration, curiosity becomes coverage-tested (a calibrated event) and would fall under the policy.

## Tasks

- [x] **T1** Wait for Plan 340 Phase 1 to ship `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (the floor instance).
  - **DONE 2026-06-30.** Plan 340 Phase 1 shipped behind `conformal_predictive_intervals`. GOAT gate PASSED: G1 coverage [0.9445, 0.9493], G2 interval_into H=1 = 642ns, G3 zero-alloc, G4 bit-reproducible. AirPassengers CRPS 115.06 (4× sharper than ±2σ baseline). See `.benchmarks/340_conformal_goat.md`.
- [x] **T2** Define the floor-comparison harness: a reusable benchmark fixture that wraps any UQ-bearing primitive, runs it on a standard trajectory corpus, and compares CRPS / coverage / Winkler against the floor. File as a follow-up plan or as an addition to Plan 340 Phase 2.
  - **DONE 2026-06-30.** Shipped as `crates/katgpt-core/src/conformal/floor_harness.rs` (gated on `conformal_predictive_intervals`). The harness exposes:
    - `UqPrimitiveUnderTest` trait (`name`, `predict_next`, `observe`) — adapters implement this for each primitive.
    - `FloorAdapter` — wraps the canonical `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1, capacity 256, no recency decay) as a `UqPrimitiveUnderTest`.
    - `PredictiveOutput` — holds samples and/or interval; `into_interval` normalizes samples → interval via empirical quantile.
    - `run_floor_comparison(&mut primitive, corpus, α, warmup, name) -> FloorComparisonReport` — the one-call entry point.
    - `TrajectoryCorpus` — standard fixtures (`stationary_seasonal`, `white_noise`, `from_slice`) with deterministic SplitMix64 RNG.
    - `OverallVerdict` — `BeatsFloor` / `TiesFloor` / `LosesToFloor` / `Mixed` / `NotApplicable`. Coverage policy: over-coverage is acceptable (penalized via CRPS width); only under-coverage fails the gate.
    - 13 unit tests + 10 integration tests, all green. See `.benchmarks/340_conformal_floor_harness.md`.
  - T3–T7 adapters each reduce to: implement `UqPrimitiveUnderTest` for primitive X, then call `run_floor_comparison`.
- [x] **T3** Run the floor comparison on BoMSampler (Plan 281). The comparison angle: BoMSampler produces a discrete hypothesis distribution; the floor produces a continuous interval. Reconcile by evaluating both on a task where the ground truth is a continuous value that both must predict (e.g., next HLA channel value). If BoMSampler can't be evaluated on a continuous metric, document why and exclude it from the policy (it's a discrete selector, not a continuous UQ primitive).
  - **DONE 2026-06-30 — EXCLUDED.** Adapter + 5 tests shipped at `crates/katgpt-core/tests/conformal_floor_bom.rs` (gated `conformal_predictive_intervals` + `bom_sampling`). The comparison WAS run (BoM can be evaluated on a continuous metric via `from_samples` → empirical quantile); the result is the evidence for exclusion.
  - **False-confidence finding:** BoM *wins on CRPS* (seasonal ratio 0.866, white-noise ratio 0.306) because its σ-bound intervals are narrow and CRPS rewards narrowness. But it *loses catastrophically on coverage* (5.5% / 15.1% vs nominal 95%) — the textbook false-confidence failure mode. Winkler (penalty 2/α = 40 per miss) exposes it: 13.79× / 4.11× the floor.
  - **Structural smoking gun (width-vs-volatility test):** BoM's interval width ratio across a 15× volatility change is **0.990** (≈1.0, not 15.0). Its width tracks σ (the hyperparameter), not the data's residual stream. This is why no σ gives both competitive CRPS AND nominal coverage (σ-sweep: even σ=0.5 reaches only 0.254 coverage).
  - **Verdict:** BoM is a belief-space exploration sampler, not a calibrated forecaster. Its GOAT gate (Plan 281 G2) measures *planning* win rate (+31.49pp on the riir-ai arena, Plan 314), NOT calibrated UQ. Excluding it from the UQ policy does NOT demote it — `bom_sampling` stays DEFAULT-ON. See `.benchmarks/010_bom_floor_comparison.md` for the full report.
- [x] **T4** Run the floor comparison on Sleep-Time Query Anticipator (Plan 334/341). The comparison angle: predictability scores from the anticipator vs interval-width from the floor. Both should correlate with actual forecast difficulty; the one with higher correlation wins.
  - **DONE 2026-07-01 — EXCLUDED.** Adapter + 6 tests shipped at `crates/katgpt-core/tests/conformal_floor_sleep_time.rs` (gated `conformal_predictive_intervals` + `sleep_time_anticipation`). Two evaluation angles:
    - **Calibration** (via `run_floor_comparison`): all 3 corpora (seasonal, white noise, regime-switching) show the BoM false-confidence signature — WIN CRPS (0.55–0.63 ratio, narrower intervals), LOSE coverage (37–54% vs 95%), LOSE Winkler (2.5–3.4× the floor). The anticipator's `p_best` sigmoid-saturates high (~0.73 mean), so `(1−p_best)·scale` produces narrower intervals than the conformal quantile.
    - **Difficulty correlation** (T4's specific ask): Pearson r of each primitive's half-width with `|Δy|` is near-zero for BOTH (anticipator |r| < 0.08; floor |r| < 0.03). The floor achieves nominal coverage by *marginal* calibration (residual quantile is roughly constant on stationary data), not per-step difficulty tracking. The regime-switching corpus (designed for genuine difficulty variation) still shows r ≈ 0.03 for both — the floor's 256-capacity residual pool mixes both regimes.
  - **Verdict:** the anticipator's predictability is a gate heuristic, not a calibrated UQ signal. Its GOAT gate (Plan 334 G1) measures gate mechanics + amortization, NOT UQ. Excluded via the reframing escape hatch; `sleep_time_anticipation` stays OPT-IN (unchanged). See `.benchmarks/010_sleep_time_floor_comparison.md`.
- [x] **T5** Run the floor comparison on Best-Belief Beta Selector (Plan 336). The comparison angle: conservative candidate selection via Beta ε-quantile vs via empirical ε-quantile (the floor). Both are inverse-CDF reads; the question is whether the Beta prior (discrete, parametric) beats the empirical prior (continuous, nonparametric) on selection quality.
  - **DONE 2026-07-01 — BEATS FLOOR.** Adapter + 6 tests shipped at `crates/katgpt-core/tests/conformal_floor_best_belief.rs` (gated `conformal_predictive_intervals` + `best_belief`). NOT an interval-calibration test — it's a **selection-regret** test (`θ_best − θ_selected`). The floor is the empirical MLE (`S/(S+F)`, pure exploitation, no regularization).
    - **Foundational finding (uniform n):** at fixed observation count n per candidate, Beta and MLE produce IDENTICAL selections. This is a theorem: `BB_ε(S, n−S)` is monotone in S/n → same argmax. The Beta conservatism shifts absolute scores (useful for thresholding/gating) but not the within-pool ordering.
    - **Real-world finding (variable n):** when candidates have different evidence weights (heteroscedastic — frozen snapshots with different deployment durations), Beta WINS by 15–30% regret reduction across n_mean ∈ [4, 128]. The improvement is stable (25–30% mid-range), confirming structural value.
    - **Stress test (one low-data candidate, n_lo=2):** Beta WINS by 61–77%. A 2/2 lucky streak has MLE=1.0 (picked by MLE over a genuinely better 60/80 candidate) but BB_0.05(2,0)≈0.025 (correctly discounted). The improvement grows with n_mean as the contrast sharpens.
  - **Verdict:** the first primitive to genuinely beat its floor (unlike BoM T3 and Sleep-Time T4, which were EXCLUDED via reframing). `best_belief` stays DEFAULT-ON (Plan 336 Phase 2 promotion confirmed). See `.benchmarks/010_best_belief_floor_comparison.md`.
- [x] **T6** Decide on Alien Sampler (Plan 311): UQ-bearing or not? If yes, run floor comparison.
  - **DONE 2026-07-01 — EXCLUDED (not UQ-bearing).** The Alien Sampler produces a within-pool z-scored ranking: `score = (1−β)·z_coh + β·(−z_avail)`. This is a *relative selection signal* (which candidate is more diverse in this pool), not a calibrated uncertainty estimate. It claims no probability distribution, predictive interval, quantile, coverage guarantee, or confidence score. Its GOAT gate (Plan 311, 1/4 PASS — demoted to opt-in) measures **motif-collapse reduction** (a diversity metric: concentration ratio), NOT coverage/CRPS/Winkler. Same structural exclusion as BoM (planning quality) and Sleep-Time (compute gating). No floor comparison needed.
- [x] **T7** Document results in `.benchmarks/` and update each primitive's plan with the floor-comparison row in its GOAT gate table.
  - **DONE 2026-07-01.** Benchmark docs: `.benchmarks/010_bom_floor_comparison.md` (T3, prior session), `010_sleep_time_floor_comparison.md` (T4), `010_best_belief_floor_comparison.md` (T5). Plan updates: Plan 281 `G-UQ` row (prior session), Plan 334 "Report the Floor" subsection (T4), Plan 336 "Report the Floor" subsection (T5), Plan 311 status-line note (T6). All four UQ-bearing candidates now have a documented floor-comparison verdict in both the issue tracker and their primitive plan.

## Failure mode

If any primitive fails to beat the floor, the policy requires either:
1. **Demotion** (if currently default-on) back to opt-in, with a note that the primitive is not adding UQ value over the conformal-naive baseline.
2. **Reframing** — the primitive may be valuable for a non-UQ reason (latency, interpretability, composition) even if it doesn't beat the floor on raw UQ quality. In that case, drop the UQ claim from the selling point and re-position the primitive.

The floor comparison is a quality bar, not a deletion trigger. A primitive that ties the floor but is 10× faster is still valuable — it just can't claim "better UQ than the baseline".

## References

- Policy: `katgpt-rs/AGENTS.md` Feature Flag Discipline, research skill `SKILL.md` §Workflow 2.
- Research: `.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md`.
- Floor implementation: `.plans/340_conformal_predictive_intervals_primitive.md` Phase 1.
- Companion paper: [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) — *Report the Floor*.
