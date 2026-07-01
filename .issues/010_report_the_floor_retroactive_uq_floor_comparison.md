# Issue 010: "Report the Floor" — Retroactive UQ Floor Comparison for Grandfathered Primitives

**Date:** 2026-06-28 (filed retroactively 2026-07-02; infrastructure shipped 2026-06-30)
**Severity:** 🟡 POLICY / TRACKING (enforcement substrate, not a bug)
**Status:** ✅ **ALL TASKS (T1–T7) COMPLETE.** The rule is **enforceable** since Plan 340 Phase 1 shipped the floor. See `.benchmarks/010_report_the_floor_consolidated.md` for the cross-primitive audit doc (T7).
**Plan:** [340_conformal_predictive_intervals_primitive.md](../.plans/340_conformal_predictive_intervals_primitive.md) (Phases 1, 2, 2.5)
**Benchmarks:** [340_conformal_floor_harness.md](../.benchmarks/340_conformal_floor_harness.md)
**Rule source:** `katgpt-rs/AGENTS.md` → Feature Flag Discipline → "UQ-bearing primitive GOAT gate extension"

---

## Summary

The "Report the Floor" rule (adopted 2026-06-28 per Research 322 / Plan 340, companion paper [arXiv:2606.09473](https://arxiv.org/abs/2606.09473)) requires that **any primitive claiming a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty** (collectively: UQ-bearing) MUST benchmark against the **conformal-naive floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 with `m=1`, plain split conformal) — on CRPS / coverage / Winkler score. If the primitive cannot beat the floor, the GOAT gate FAILS.

This issue tracks:
1. The floor itself (T1).
2. The comparison harness (T2).
3. Retroactive floor comparison for the **four grandfathered** UQ-bearing primitives (T3–T6).
4. Cross-primitive documentation (T7).

**Future UQ primitives must include the floor at their initial GOAT gate** — they are NOT grandfathered.

---

## The grandfathered primitives

Per `AGENTS.md`, four existing UQ-bearing primitives are grandfathered (shipped before the floor existed) but must include the floor at their next re-gate:

| # | Primitive | Plan | Feature flag | T# | Status |
|---|---|---|---|---|---|
| 1 | BoMSampler | [281](../.plans/281_bom_single_pass_diverse_sampling.md) | `bom_sampling` | T3 | ✅ DONE — **EXCLUDED** (see verdict) |
| 2 | Sleep-Time Query Anticipator | [334](../.plans/334_sleep_time_query_anticipator_primitive.md) | `sleep_time_anticipation` | T4 | ✅ DONE |
| 3 | Best-Belief Beta Selector | [336](../.plans/336_controlled_utility_primitives.md) | `best_belief` | T5 | ✅ DONE |
| 4 | KARC + conformal overlay | [308](../.plans/308_karc_delay_basis_ridge_forecaster.md) + [340](../.plans/340_conformal_predictive_intervals_primitive.md) Ph.2 | `karc_forecaster` + `conformal_predictive_intervals` | (covered by `conformal_karc_overlay` example) | ✅ DONE |
| (5) | Alien Sampler | [311](../.plans/311_alien_sampler_primitive.md) | `alien_sampler` | T6 | ✅ DONE — **EXCLUDED** (not UQ-bearing; see verdict) |

---

## Task breakdown

### T1 — Ship the conformal-naive floor ✅ COMPLETE

- [x] `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1` (plain split conformal).
- [x] Ships behind `conformal_predictive_intervals` (opt-in).
- [x] Plan 340 Phase 1. GOAT gate: `.benchmarks/340_conformal_goat.md`.

The floor is the canonical baseline. Every UQ primitive's GOAT gate MUST beat it on CRPS / coverage / Winkler. Until Plan 340 Phase 1 shipped (2026-06-30), the rule was recorded in `AGENTS.md` but **not enforceable**; it is now enforceable.

### T2 — Ship the "Report the Floor" comparison harness ✅ COMPLETE

- [x] `run_floor_comparison` entry point + `UqPrimitiveUnderTest` trait + `FloorAdapter` + `OverallVerdict` enum.
- [x] Standard corpora: `stationary_seasonal`, `white_noise`, `from_slice`.
- [x] Plan 340 Phase 2.5. GOAT gate: `.benchmarks/340_conformal_floor_harness.md` (23 tests green).
- [x] Tests: `tests/conformal_floor_harness.rs` (10 integration) + `src/conformal/floor_harness.rs::tests` (13 unit).

**Coverage policy (load-bearing design decision):** over-coverage is ACCEPTABLE (extra width already penalized via CRPS). Only **under-coverage** (false confidence) fails the gate. This is why BoM's narrow σ-bound intervals fail despite flattering CRPS.

### T3 — BoMSampler (Plan 281) floor comparison ✅ DONE — EXCLUDED

- [x] `tests/conformal_floor_bom.rs` (5 tests, all green).
- [x] Verdict recorded in test header + assertions.

**Verdict: EXCLUDED from the "Report the Floor" policy.** BoM's K-hypothesis spread is **exploration noise** (controlled by σ), not calibrated predictive uncertainty. The empirical evidence:

| Corpus | CRPS ratio | Winkler ratio | Coverage (nom 0.95) | Verdict |
|---|---|---|---|---|
| seasonal | 0.866 | 13.79 | 0.055 | Mixed |
| white noise | 0.306 | 4.11 | 0.151 | Mixed |

BoM *wins on CRPS* because its intervals are narrow (σ-bound), and CRPS rewards narrowness. But it *loses catastrophically on coverage* (5–15% vs nominal 95%) — the textbook **false-confidence** failure mode. The Winkler score (penalizes misses by 2/α = 40) exposes the under-coverage: 4–14× the floor. No value of σ fixes this (σ=0.5 lifts coverage to only 0.254, still a third of nominal).

**Implication:** BoM's GOAT gate (Plan 281 G2) measures *planning* win rate (+31.49pp on diverse sampling), NOT calibrated UQ. BoM is a diversity-for-exploration primitive, not a UQ primitive. It is correctly excluded from the floor policy — its value proposition is orthogonal to calibrated prediction.

### T4 — Sleep-Time Query Anticipator (Plan 334) floor comparison ✅ DONE

- [x] `tests/conformal_floor_sleep_time.rs` (green).
- [x] Two evaluation angles: (1) calibration via `run_floor_comparison` (CRPS/coverage/Winkler on a derived interval `width = z·scale·(1−p_best)`), (2) correlation of `(1−p_best)`-derived width with actual `|Δy|` difficulty vs the floor's width.

**Question answered:** is the anticipator's modelless predictability score (`DotPredictabilityScorer`, `p = sigmoid(α·dot(c, dir) + β)`) a calibrated UQ signal, or an uncalibrated gate heuristic? See the test file for the canonical verdict — the comparison isolates the interval-calibration question by holding the point forecast identical (last observation) to the floor's seasonal-naive.

### T5 — Best-Belief Beta Selector (Plan 336) floor comparison ✅ DONE

- [x] `tests/conformal_floor_best_belief.rs` (green).
- [x] Comparison angle: **selection-quality** (selection regret `θ_best − θ_selected`), NOT interval-calibration.

**Why this is a different comparison:** the Best-Belief `BB_ε(S, F) = I⁻¹_ε(1+S, 1+F)` is an inverse-CDF read (the ε-quantile of the Beta(1+S, 1+F) posterior). The honest floor is the **empirical MLE** `S/(S+F)` (maximum-likelihood point estimate, no regularization). The question: does the Beta prior (regularizes low-data candidates toward 0.5) + ε-quantile conservatism beat raw MLE on selection quality? Expected: Best-Belief WINS at low observation counts (MLE over-fits noise — 2/2 looks perfect under MLE but mediocre under Beta), TIES at high counts.

### T6 — Alien Sampler (Plan 311) floor comparison ✅ DONE — EXCLUDED

- [x] Decision: is Alien Sampler UQ-bearing? **NO — EXCLUDED.**
- [x] ~~If UQ-bearing: adapter + `run_floor_comparison`.~~ N/A — see verdict.

**Verdict: EXCLUDED from the "Report the Floor" policy.** The Alien Sampler produces a within-pool z-scored ranking (`score = (1−β)·z_coh + β·(−z_avail)`), which is a *relative selection signal* (which candidate is more diverse in this pool), not a calibrated uncertainty estimate. It claims no probability distribution, predictive interval, quantile, coverage guarantee, or confidence score. Its GOAT gate (Plan 311 Phase 3) measures motif-collapse reduction (population diversity), not coverage/CRPS/Winkler.

**Structural exclusion — same class as BoM (T3, planning), Sleep-Time (T4, compute gating).** Unlike BoM (which at least produces K samples that can be projected to an empirical-quantile interval), Alien Sampler's output is a single ranking scalar per candidate — there is no spread, distribution, or interval to feed the harness. Running `run_floor_comparison` would require inventing a UQ interpretation the primitive does not make.

**Additional context (post-T6 discovery):** Alien Sampler has since been **exiled to `katgpt-deprecated`** (Proposal 003 Phase 3a) following its GOAT gate failure (2/4 PASS: G1 borderline-fail 0.5010, G2 fail 0.6722, G3 closed via Rayon ~4.5×, G4 pass — see `.benchmarks/311_alien_sampler_goat.md`). The primitive is already demoted on its own GOAT merits, so the UQ-policy question is doubly moot.

**Citation note (FIXED 2026-07-02):** Proposal 003 and `Cargo.toml` previously cited "`issues/010` T6" as the source of the "1/4 PASS — demoted to opt-in" verdict. That was a citation error — the GOAT verdict comes from **Plan 311 Phase 3** + `.benchmarks/311_alien_sampler_goat.md`, not from this issue's T6 (which was always about the UQ floor comparison, a separate question). The citations have been corrected to point at Plan 311 Phase 3 / the bench file, and the stale "1/4 PASS" has been updated to "2/4 PASS" (G3 was closed via Rayon on 2026-06-24; the demotion verdict is unchanged because G1+G2 are the real drivers). The demotion itself was always correct; only the citation chain was wrong.

### T7 — Cross-primitive documentation ✅ DONE

- [x] Consolidate T3–T6 verdicts into a single `.benchmarks/` doc.

**Deliverable:** [`.benchmarks/010_report_the_floor_consolidated.md`](../.benchmarks/010_report_the_floor_consolidated.md). Consolidates all five grandfathered-primitive verdicts (BoM T3, Sleep-Time T4, Best-Belief T5, KARC+overlay, Alien Sampler T6) into one auditability doc with a master verdict table, the common false-confidence signature, and per-primitive one-line summaries with links to the full benchmark files.

---

## Failure mode (the policy's escape hatch)

The verdict is a **hint, not a judgment**. The harness exposes raw metrics + ratios so a human can overrule it. A primitive that ties on CRPS but is 10× faster is still valuable — the "reframing" escape hatch. Conversely, a primitive that "wins" CRPS by being dangerously narrow (BoM) is correctly flagged as false-confidence.

The policy's purpose is not to ban primitives that lose to the floor — it is to **force the comparison to be reported** so that a loss is a conscious, documented decision, not a silent omission.

---

## References

- **Rule:** `katgpt-rs/AGENTS.md` → Feature Flag Discipline → "UQ-bearing primitive GOAT gate extension".
- **Companion paper:** [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) — *Report the Floor*.
- **Plan 340:** [`.plans/340_conformal_predictive_intervals_primitive.md`](../.plans/340_conformal_predictive_intervals_primitive.md).
- **Harness GOAT:** [`.benchmarks/340_conformal_floor_harness.md`](../.benchmarks/340_conformal_floor_harness.md).
- **Research:** [`.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md`](../.research/322_Conformal_Seasonal_Pools_Calibrated_UQ_Overlay.md).
