# Benchmark 010 (Consolidated): "Report the Floor" — All Grandfathered UQ Primitives

**Date:** 2026-07-02 (consolidation; individual comparisons run 2026-06-30 → 2026-07-01)
**Task:** Issue 010 T7 — cross-primitive documentation of the "Report the Floor" verdicts.
**Policy:** `katgpt-rs/AGENTS.md` → Feature Flag Discipline → "UQ-bearing primitive GOAT gate extension".
**Companion paper:** [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) — *Report the Floor*.
**Floor:** `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 Phase 1, `m=1`, plain split conformal).
**Harness:** `crates/katgpt-core/src/conformal/floor_harness.rs` (`run_floor_comparison`, Issue 010 T2).

---

## TL;DR

Four UQ-bearing primitives were grandfathered when the "Report the Floor" rule was adopted (2026-06-28). This doc consolidates their retroactive floor comparisons. **One genuinely beats the floor; three are EXCLUDED via the reframing escape hatch; one (KARC) is the floor's own overlay example.**

| # | Primitive | T# | Verdict | Why |
|---|---|---|---|---|
| 1 | BoMSampler (Plan 281) | T3 | 🟠 **EXCLUDED** | Belief-space exploration, not calibrated UQ. False-confidence signature (5–15% coverage). |
| 2 | Sleep-Time Anticipator (Plan 334) | T4 | 🟠 **EXCLUDED** | Compute-gating heuristic, not calibrated UQ. Same false-confidence signature (37–54% coverage). |
| 3 | Best-Belief Beta Selector (Plan 336) | T5 | 🟢 **BEATS FLOOR** | 15–30% selection-regret reduction in the heteroscedastic regime (the real-world case). Genuine UQ gain. |
| 4 | KARC + conformal overlay (Plan 308 + 340 Ph.2) | — | 🟢 **IS THE OVERLAY** | The conformal overlay *is* the floor applied to KARC's point forecast. Covered by the `conformal_karc_overlay` example. |
| 5 | Alien Sampler (Plan 311) | T6 | 🟠 **EXCLUDED** | Relative ranking signal (which candidate is most alien), not calibrated UQ. No distribution to feed the harness. Additionally exiled to `katgpt-deprecated` (GOAT 2/4 PASS — initially 1/4, G3 closed via Rayon). |

**Net outcome:** the policy is enforceable and exercised. Of the four grandfathered primitives that actually make a UQ-adjacent claim, **only Best-Belief (T5) genuinely beats the floor on its native metric** — and it does so decisively in the regime that matters (heteroscedastic evidence). The other three are correctly excluded: their value propositions (planning diversity, compute gating, population diversity) are orthogonal to calibrated predictive intervals, and the floor comparison exposes any attempt to reframe them as UQ.

---

## The common false-confidence signature (T3, T4)

BoM (T3) and Sleep-Time (T4) exhibit the **same structural failure mode** when forced into the UQ harness: they produce intervals that are **too narrow** because their width is driven by an **exogenous signal** (a hyperparameter or a modelless heuristic), not by the **endogenous residual stream** that split conformal calibrates against.

| Property | Conformal UQ (the floor) | BoM (T3) | Sleep-Time (T4) |
|---|---|---|---|
| Width driver | Data residuals (endogenous) | σ hyperparameter (exogenous) | Context-direction alignment (modelless heuristic) |
| Adapts to volatility? | Yes (residual quantile) | No (width-volatility test: ratio ≈ 1.0 across 15× volatility change) | No (sigmoid-saturated, ~constant) |
| Coverage guarantee | Yes (split conformal, nominal ± tol) | No (5–15% at nominal 95%) | No (37–54% at nominal 95%) |
| CRPS result | baseline | **WINS** (rewards narrowness) | **WINS** (rewards narrowness) |
| Winkler result | baseline | **LOSES** 4–14× (penalizes misses) | **LOSES** 2.5–3.4× (penalizes misses) |
| Verdict | (is the floor) | EXCLUDED (planning, not UQ) | EXCLUDED (gating, not UQ) |

**The lesson:** CRPS alone is a misleading UQ metric because it rewards narrow intervals. The policy's coverage + Winkler requirements are load-bearing — they expose false confidence that CRPS hides. This is exactly the failure mode the "Report the Floor" paper warns against, and the harness catches it deterministically.

---

## Per-primitive summaries

### T3 — BoMSampler: EXCLUDED

BoM's K diverse next-belief-states are **exploration noise** (controlled by σ), not calibrated predictive intervals. Projecting the K hypotheses to a scalar and converting to an empirical-quantile interval produces the false-confidence signature: CRPS wins (narrow σ-bound intervals), coverage collapses (5–15%), Winkler explodes (4–14×). The σ-sweep shows no σ fixes this — σ=0.5 lifts coverage to only 0.254. BoM's GOAT gate (Plan 281 G2) measures **planning** win rate (+31.49pp on the riir-ai arena, Plan 314), not calibrated UQ. `bom_sampling` stays DEFAULT-ON; the exclusion means BoM cannot claim "calibrated UQ" as a selling point, which it never did.

- **Full benchmark:** [`010_bom_floor_comparison.md`](010_bom_floor_comparison.md)
- **Adapter + tests:** `crates/katgpt-core/tests/conformal_floor_bom.rs` (5 tests)

### T4 — Sleep-Time Query Anticipator: EXCLUDED

The anticipator's predictability score `p = sigmoid(α·dot(c, dir) + β)` is a **compute-gate heuristic** (should we pre-compute this query?), not a calibrated UQ signal. The false-confidence signature appears (CRPS wins, coverage 37–54%, Winkler 2.5–3.4×). The T4-specific difficulty-correlation test shows near-zero per-step correlation (|r| < 0.08) between the anticipator's width signal and actual innovation magnitude — but the floor's width has equally weak per-step correlation (coverage comes from marginal calibration, not per-step difficulty tracking). The anticipator's GOAT gate (Plan 334 G1) measures **mechanics + amortization**, not calibrated UQ. `sleep_time_anticipation` stays OPT-IN; the exclusion is consistent with its actual selling point.

- **Full benchmark:** [`010_sleep_time_floor_comparison.md`](010_sleep_time_floor_comparison.md)
- **Adapter + tests:** `crates/katgpt-core/tests/conformal_floor_sleep_time.rs` (6 tests)

### T5 — Best-Belief Beta Selector: BEATS FLOOR ✅

The only primitive that genuinely beats its floor. `BB_ε(S, F) = I⁻¹_ε(1+S, 1+F)` (the ε-quantile of the Beta posterior) beats the empirical MLE `S/(S+F)` on **selection regret** `θ_best − θ_selected`:

- **Uniform n:** TIE (monotonicity theorem — `BB_ε(S, n−S)` is monotone in `S/n`, so argmax is preserved).
- **Variable n (real-world):** Beta WINS by **15–30%** regret reduction (stable across n_mean).
- **Low-data stress (n_lo=2):** Beta WINS by **61–77%** (discounts 2/2 false positives that MLE scores as 1.0).

This is a genuine UQ gain, not a reframing. The Beta prior earns its keep exactly where the MLE fails: heteroscedastic evidence weights (frozen snapshots / archetype shards with different deployment durations → different observation counts). `best_belief` stays DEFAULT-ON (promoted in Plan 336 Phase 2); the floor comparison confirms the promotion.

- **Full benchmark:** [`010_best_belief_floor_comparison.md`](010_best_belief_floor_comparison.md)
- **Adapter + tests:** `crates/katgpt-core/tests/conformal_floor_best_belief.rs` (6 tests)

### KARC + conformal overlay: IS THE OVERLAY

KARC (Plan 308) is a delay-basis ridge forecaster — a **point forecast** primitive. The conformal overlay (Plan 340 Phase 2) *is* the floor applied to KARC's point forecast: `ConformalIntervalCalibrator<KarcForecaster>`. There is no separate "KARC vs floor" comparison because KARC + overlay **is** the floor pattern composed with a better point forecaster. The composite is covered by the `conformal_karc_overlay` example in Plan 340 Phase 2. The GOAT gate for the composite is in [`.benchmarks/340_conformal_goat.md`](340_conformal_goat.md).

### T6 — Alien Sampler: EXCLUDED

The Alien Sampler produces a within-pool z-scored ranking `score = (1−β)·z_coh + β·(−z_avail)` — a **relative selection signal** (which candidate is more diverse in this pool), not a calibrated uncertainty estimate. It claims no probability distribution, predictive interval, quantile, coverage guarantee, or confidence score. Its GOAT gate (Plan 311 Phase 3) measures motif-collapse reduction (population diversity). **Unlike BoM (which at least produces K samples projectable to an interval), Alien Sampler's output is a single ranking scalar per candidate — there is no spread to feed the harness.**

Structurally the same exclusion class as T3/T4: a primitive whose value proposition (population diversity) is orthogonal to calibrated predictive intervals. Additionally, the primitive has since been **exiled to `katgpt-deprecated`** (Proposal 003 Phase 3a) after its own GOAT gate failed (2/4 PASS: G1 borderline 0.5010, G2 fail 0.6722, G3 closed via Rayon ~4.5×, G4 pass — see [`.benchmarks/311_alien_sampler_goat.md`](311_alien_sampler_goat.md)). The UQ-policy question is therefore doubly moot.

**No adapter test was written** — there is nothing UQ-shaped to adapt. The exclusion is structural (not UQ-bearing by design), not empirical (no comparison was run).

- **Plan verdict:** [`311_alien_sampler_primitive.md`](../.plans/311_alien_sampler_primitive.md) line 9 (EXCLUDED rationale).
- **GOAT benchmark (separate question):** [`311_alien_sampler_goat.md`](311_alien_sampler_goat.md).

---

## Master verdict table (the audit surface)

| # | Primitive | Feature | UQ claim? | Verdict | Floor metric | Key number | Promotion impact |
|---|---|---|---|---|---|---|---|
| T3 | BoMSampler | `bom_sampling` (DEFAULT-ON) | No (planning) | EXCLUDED | CRPS/coverage/Winkler | 5–15% coverage | None — stays DEFAULT-ON |
| T4 | Sleep-Time | `sleep_time_anticipation` (OPT-IN) | No (gating) | EXCLUDED | CRPS/coverage/Winkler + difficulty r | 37–54% coverage | None — stays OPT-IN |
| T5 | Best-Belief | `best_belief` (DEFAULT-ON) | Yes (conservative selection) | **BEATS FLOOR** | Selection regret | 15–30% regret ↓ (variable n) | Confirms DEFAULT-ON |
| — | KARC + overlay | `karc_forecaster` + `conformal_predictive_intervals` | Yes (calibrated intervals via overlay) | **IS THE OVERLAY** | (is the floor) | n/a | n/a |
| T6 | Alien Sampler | `alien_sampler` (exiled → `katgpt-deprecated`) | No (ranking) | EXCLUDED | n/a (structural) | n/a | Already demoted on GOAT |

---

## Reproducibility

All four runnable comparisons (T3, T4, T5, + the KARC overlay example) are bit-reproducible via deterministic SplitMix64 + Box-Muller RNGs (same constants as the harness's private RNG) and pinned configs. See the individual benchmark files for per-primitive `cargo test` invocations.

```bash
# T3 — BoM
cargo test -p katgpt-core --test conformal_floor_bom \
  --features conformal_predictive_intervals,bom_sampling -- --nocapture

# T4 — Sleep-Time
cargo test -p katgpt-core --test conformal_floor_sleep_time \
  --features conformal_predictive_intervals,sleep_time_anticipation -- --nocapture

# T5 — Best-Belief
cargo test -p katgpt-core --test conformal_floor_best_belief \
  --features conformal_predictive_intervals,best_belief -- --nocapture

# KARC overlay — see Plan 340 Phase 2 example
# T6 — Alien Sampler: no test (structural exclusion)
```

---

## References

- **Policy:** `katgpt-rs/AGENTS.md` → Feature Flag Discipline → "UQ-bearing primitive GOAT gate extension".
- **Consolidated audit:** this document (replaces the resolved Issue 010 tracker).
- **Plan 340 (the floor):** [`.plans/340_conformal_predictive_intervals_primitive.md`](../.plans/340_conformal_predictive_intervals_primitive.md).
- **Harness GOAT:** [`.benchmarks/340_conformal_floor_harness.md`](340_conformal_floor_harness.md).
- **Companion paper:** [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) — *Report the Floor*.
- **Individual benchmarks:** [`010_bom_floor_comparison.md`](010_bom_floor_comparison.md), [`010_sleep_time_floor_comparison.md`](010_sleep_time_floor_comparison.md), [`010_best_belief_floor_comparison.md`](010_best_belief_floor_comparison.md), [`311_alien_sampler_goat.md`](311_alien_sampler_goat.md), [`340_conformal_goat.md`](340_conformal_goat.md).

---

## TL;DR

The "Report the Floor" policy is enforceable and exercised. Of the five grandfathered UQ-adjacent primitives, **only Best-Belief (T5) genuinely beats the floor** — and it does so in the heteroscedastic regime that characterizes its actual use case (15–30% selection-regret reduction, up to 77% on low-data false positives). The other four are correctly excluded: BoM and Sleep-Time produce the textbook false-confidence signature (narrow intervals, collapsed coverage) and are reframed as planning/gating primitives; KARC + overlay *is* the floor composed with a better point forecast; Alien Sampler has no UQ-shaped output to evaluate. The policy's coverage + Winkler requirements are load-bearing — they catch false confidence that CRPS alone hides.
