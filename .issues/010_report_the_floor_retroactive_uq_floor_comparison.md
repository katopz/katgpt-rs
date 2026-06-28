# Issue 010: Retroactive "Report the Floor" — conformal-naive floor comparison for existing UQ-bearing primitives

**Filed:** 2026-06-28
**Policy source:** Research 322 (`.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md`), Plan 340 (`.plans/340_conformal_predictive_intervals_primitive.md`), `katgpt-rs/AGENTS.md` Feature Flag Discipline, research skill `SKILL.md` §Workflow 2.
**Companion paper:** *Report the Floor* (arXiv:2606.09473) — argues a training-free conformal interval is a mandatory baseline for any probabilistic forecaster.
**Blocking dependency:** Plan 340 Phase 1 must ship the `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` floor instance before any retroactive comparison can run. Until then, this issue is recorded but not actionable.

---

## Context

On 2026-06-28, the "Report the Floor" policy was adopted as a GOAT gate extension for all UQ-bearing primitives: any primitive claiming a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty must beat the conformal-naive floor (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1`) on CRPS / coverage / Winkler score. If it can't beat the floor, it's noise, not UQ.

The policy applies **prospectively** to all future UQ primitives (enforced from initial GOAT gate) and **retroactively** to existing UQ-bearing primitives (grandfathered at current promotion state, but must include the floor at their next re-gate or feature-touch). This issue tracks the retroactive work.

## UQ-bearing primitives requiring retroactive floor comparison

| Primitive | Plan | Current state | Floor comparison needed at |
|---|---|---|---|
| **BoMSampler** | 281 | shipped | next re-gate or feature-touch |
| **Sleep-Time Query Anticipator** | 334 (open) / 341 (riir-ai runtime) | shipped | next re-gate or feature-touch |
| **Best-Belief Beta Selector** | 336 | shipped (G2 FAIL, LUT unblock in progress) | next re-gate or feature-touch |
| **KARC + conformal overlay** | 308 + 340 | KARC shipped (DEFAULT-ON); overlay in Plan 340 | Plan 340 Phase 2 (KARC adapter) — the overlay itself defines the floor, so this is the reference, not a comparison target |
| **Alien Sampler** | 311 | shipped | borderline — it's a selection/ranking mechanism, not a calibrated distribution. **Decision needed:** does "coherence × availability frontier ranking" count as UQ? If yes, add floor comparison; if no, exclude. |

## Borderline / excluded primitives (for clarity)

These are **not** UQ-bearing under the policy definition (they don't claim a distribution / interval / quantile / confidence):

- KARC point forecast alone (Plan 308) — single point, no uncertainty claim. Becomes UQ-bearing only when wrapped by the conformal overlay (Plan 340 Phase 2).
- Constraint pruners, bandits, DDTree, speculative decode — these claim validity/relevance/reward, not calibrated uncertainty.
- Salience Tri-Gate (Plan 303) — discrete Speak/Silent/Delegate decision, not a distribution. (Though it *consumes* UQ from the conformal overlay.)
- CGSP runtime curiosity — currently a magnitude (not calibrated). **After** Plan 340 integration, curiosity becomes coverage-tested (a calibrated event) and would fall under the policy.

## Tasks

- [ ] **T1** Wait for Plan 340 Phase 1 to ship `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (the floor instance).
- [ ] **T2** Define the floor-comparison harness: a reusable benchmark fixture that wraps any UQ-bearing primitive, runs it on a standard trajectory corpus, and compares CRPS / coverage / Winkler against the floor. File as a follow-up plan or as an addition to Plan 340 Phase 2.
- [ ] **T3** Run the floor comparison on BoMSampler (Plan 281). The comparison angle: BoMSampler produces a discrete hypothesis distribution; the floor produces a continuous interval. Reconcile by evaluating both on a task where the ground truth is a continuous value that both must predict (e.g., next HLA channel value). If BoMSampler can't be evaluated on a continuous metric, document why and exclude it from the policy (it's a discrete selector, not a continuous UQ primitive).
- [ ] **T4** Run the floor comparison on Sleep-Time Query Anticipator (Plan 334/341). The comparison angle: predictability scores from the anticipator vs interval-width from the floor. Both should correlate with actual forecast difficulty; the one with higher correlation wins.
- [ ] **T5** Run the floor comparison on Best-Belief Beta Selector (Plan 336). The comparison angle: conservative candidate selection via Beta ε-quantile vs via empirical ε-quantile (the floor). Both are inverse-CDF reads; the question is whether the Beta prior (discrete, parametric) beats the empirical prior (continuous, nonparametric) on selection quality.
- [ ] **T6** Decide on Alien Sampler (Plan 311): UQ-bearing or not? If yes, run floor comparison.
- [ ] **T7** Document results in `.benchmarks/` and update each primitive's plan with the floor-comparison row in its GOAT gate table.

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
