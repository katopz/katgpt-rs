# Benchmark 010: Sleep-Time Query Anticipator Floor Comparison (Issue 010 T4)

**Date:** 2026-07-01
**Task:** Issue 010 T4 — "Report the Floor" comparison for Sleep-Time Query Anticipator (Plan 334/341).
**Verdict:** **EXCLUDED from the "Report the Floor" policy** (reframing escape hatch — the anticipator's predictability is a gate heuristic, not a calibrated UQ signal).
**Harness:** `crates/katgpt-core/src/conformal/floor_harness.rs` (Issue 010 T2).
**Adapter + tests:** `crates/katgpt-core/tests/conformal_floor_sleep_time.rs`.

---

## TL;DR

The Sleep-Time Query Anticipator (Plan 334) produces a per-direction
**predictability score** `p_i ∈ [0,1]` via the modelless `DotPredictabilityScorer`
`p = sigmoid(α·dot(c, dir) + β)`. T4 asks: is this score a calibrated UQ signal,
or an uncalibrated gate heuristic?

The answer, with evidence: **it is an uncalibrated gate heuristic.** The same
false-confidence signature as BoM (T3) appears — the anticipator's
predictability-derived intervals WIN on CRPS (narrow) but LOSE catastrophically
on coverage (37–54% vs nominal 95%) and Winkler (2.5–3.4× the floor). And the
T4-specific **difficulty-correlation test** shows near-zero correlation
(|r| < 0.08) between the anticipator's width signal and actual one-step
innovation magnitude — the floor's width has equally weak per-step correlation
(coverage comes from marginal calibration, not per-step difficulty tracking).

The anticipator's GOAT gate (Plan 334 G1) measures **gate mechanics +
amortization** (BLAKE3 commitment, consume() latency, E[gate] hit rate), NOT
calibrated UQ — so this exclusion is consistent with the anticipator's actual
selling point. Excluding it does NOT demote it: `sleep_time_anticipation` stays
OPT-IN (unchanged).

---

## Method

The `SleepTimeAnticipatorAdapter` (`tests/conformal_floor_sleep_time.rs`) wraps
`SleepTimeAnticipator<D=4, K=4, IdentityFunctorOp, DotPredictabilityScorer>` as
a `UqPrimitiveUnderTest`:

- **Context embedding**: `c = [y_{t−1}, y_{t−2}, y_{t−3}, y_{t−4}]` — the last
  D observations (delay embedding). Generic over the time series domain;
  no game IP.
- **Direction set** (K=4, fixed): common query classes on a delay embedding:
  - `[+1, 0, 0, 0]` — "recent level"
  - `[+1, +1, 0, 0]` — "two-back level"
  - `[+1, −1, 0, 0]` — "recent trend" (first difference)
  - `[0, 0, +1, −1]` — "older trend"
- **Point forecast**: last observation (matches the floor's seasonal-naive m=1,
  so both points are identical — the comparison isolates the interval-
  calibration question, not the point-forecast question).
- **Interval width**: `z_{α/2} · residual_scale · (1 − p_best + ε)`, where
  `p_best` = max predictability across K directions, `residual_scale` =
  empirical std of one-step residuals over warmup (the SAME information the
  floor's residual pool sees), `z_{α/2}` = 1.95996 (normal quantile at α=0.05),
  `ε = 0.05` floor to avoid zero-width.
- **State advance**: on `observe(y)`, shift the delay window left and append
  `y` at position 0.
- **Scorer config**: paper default `α=1.0, β=0.0` (no per-corpus tuning — the
  honest baseline).

Three corpora (same fixtures as T3 where applicable, plus a regime-switching
corpus for the correlation angle):

- **seasonal_m12**: `sin(2πt/12) + N(0, 0.1)` — the floor's home turf.
- **white_noise**: `N(0, 0.5)` — the floor's worst-case point forecast.
- **regime_switching**: alternating 40-step blocks of seasonal vs random-walk
  — genuine variation in forecast difficulty (the key test for the correlation
  angle; the anticipator *should* show high p in seasonal blocks, low p in
  random-walk blocks, IF its predictability score tracks difficulty).

---

## Canonical run (α = 0.05, scorer α=1.0 β=0.0, D=4, K=4)

### Calibration (CRPS / coverage / Winkler via `run_floor_comparison`)

| Corpus | CRPS ratio | Winkler ratio | Coverage (nom 0.95) | Verdict |
|---|---|---|---|---|
| seasonal_m12 | 0.6294 (WIN) | 3.3889 (LOSE) | 0.4257 | Mixed |
| white_noise | 0.5508 (WIN) | 2.5472 (LOSE) | 0.5429 | Mixed |
| regime_switching | 0.5544 (WIN) | 2.9247 (LOSE) | 0.3764 | Mixed |

All three corpora show the identical false-confidence signature: the
anticipator wins CRPS (its predictability-modulated intervals are narrower
than the conformal floor's) but loses coverage (37–54% vs 95%) and Winkler
(2.5–3.4× the floor). This is structurally the same failure mode as BoM (T3):
a modelless width signal that is not calibrated to the residual distribution
produces overconfident intervals.

### Difficulty correlation (T4's specific angle)

T4 asks: "predictability scores from the anticipator vs interval-width from the
floor. Both should correlate with actual forecast difficulty; the one with
higher correlation wins." Pearson r of each primitive's half-width with
`|y_t − y_{t−1}|` (the one-step innovation magnitude = actual difficulty):

| Corpus | Anticipator r | Floor r |
|---|---|---|
| seasonal_m12 | −0.0433 | −0.0240 |
| white_noise | −0.0720 | +0.0222 |
| regime_switching | +0.0308 | +0.0278 |

**Both signals have near-zero per-step correlation with difficulty.** This is
the honest, slightly counterintuitive finding: the floor achieves nominal
coverage NOT by per-step difficulty tracking (its width is set by the marginal
residual quantile, which is roughly constant on stationary data), but by
*marginal* calibration. The anticipator's predictability score also doesn't
track per-step difficulty — it tracks context-direction alignment, which is a
different signal.

The regime-switching corpus was designed to have *genuine* difficulty variation
(seasonal blocks are predictable, random-walk blocks are not). Yet neither
signal achieves meaningful correlation (r ≈ 0.03). The reason: on a 40-step
block scale, the floor's residual pool mixes residuals from both regimes
(capacity 256 > block size 40), so its width reflects the average volatility,
not the current regime. The anticipator's `p_best` is similarly smoothed by the
delay embedding.

---

## Why the anticipator wins CRPS but loses coverage (the false-confidence trap, revisited)

Same mechanism as T3 (BoM): CRPS for a uniform interval `[l, u]` rewards
narrowness. The anticipator's `(1 − p_best + ε) · z · scale` produces intervals
narrower than the conformal quantile (because `p_best` is usually high — the
sigmoid saturates near 1.0 for aligned contexts), so CRPS wins. But narrow
intervals miss more actuals, collapsing coverage to 37–54% and exploding
Winkler (2.5–3.4× the floor's 2/α = 40 per-miss penalty).

The anticipator's `p_best` is high on average because the sigmoid saturates:
with `α=1.0` and a 4-dim context of bounded values, `dot(c, dir)` is typically
O(1), and `sigmoid(1.0)` ≈ 0.73. Most steps get `p_best` > 0.7, so width ≈
`1.96 · scale · 0.35` ≈ `0.69 · scale` — narrower than the floor's
`1.96 · scale` half-width. This is the overconfidence.

---

## Verdict

**EXCLUDED from the "Report the Floor" policy** (reframing escape hatch).

Per Issue 010's failure-mode clause:
> Reframing — the primitive may be valuable for a non-UQ reason (latency,
> interpretability, composition) even if it doesn't beat the floor on raw UQ
> quality. In that case, drop the UQ claim from the selling point and
> re-position the primitive.

The anticipator's value is in **amortized compute gating** (sleep-time
pre-compute vs wake-time fresh compute, per Plan 334's `AmortizationCostModel`),
NOT in calibrated forecasting. Its predictability score answers "should we
pre-compute this query?" — a routing/gate decision — not "what's the calibrated
predictive interval for the next observation?". The floor comparison measures
the latter; the anticipator's GOAT gate measures the former.

| Property | Conformal UQ (the floor) | Anticipator predictability |
|---|---|---|
| Width driver | Data residuals (endogenous) | Context-direction alignment (modelless heuristic) |
| Adapts to volatility? | Yes (residual quantile) | No (sigmoid-saturated, ~constant) |
| Coverage guarantee | Yes (split conformal, nominal ± tol) | No (37–54% at nominal 95%) |
| Point forecast | Seasonal-naive (last value) | Last value (matched) |
| Selling point | Calibrated intervals | Amortized compute gating |
| GOAT gate | (is the floor) | Plan 334 G1: mechanics + amortization |

Excluding the anticipator from the UQ policy does NOT demote it —
`sleep_time_anticipation` stays OPT-IN (its promotion path runs through riir-ai
Plan 341's G1–G5 on a real game corpus, not through UQ calibration). It just
means the anticipator cannot claim "calibrated UQ" as a selling point, which it
never did.

---

## Reproducibility

```bash
cargo test -p katgpt-core --test conformal_floor_sleep_time \
  --features conformal_predictive_intervals,sleep_time_anticipation -- --nocapture
```

All 6 tests green. Numbers are bit-reproducible (deterministic SplitMix64 with
the same constants as the harness's private RNG; deterministic corpus seeds
0xA1/0xB2/0xC3; deterministic anticipator scorer α=1.0 β=0.0).

Config pinned: `D=4`, `K=4`, scorer `α=1.0 β=0.0`, `z_{0.025}=1.959964`,
`ε=0.05`, warmup = 48 (seasonal, 4·12) / 64 (white noise) / 80 (regime
switching, 2 blocks).

---

## References

- Policy: `katgpt-rs/AGENTS.md` Feature Flag Discipline (UQ-bearing primitive
  GOAT gate extension), Issue 010.
- Harness: `crates/katgpt-core/src/conformal/floor_harness.rs` (Issue 010 T2).
- Adapter + tests: `crates/katgpt-core/tests/conformal_floor_sleep_time.rs`.
- Sleep-Time Anticipator: `crates/katgpt-sleep/`, Plan 334 (open math);
  riir-ai Plan 341 (private runtime).
- Anticipator GOAT gate (G1 mechanics): `.benchmarks/334_sleep_time_goat.md`.
- Companion paper: [arXiv:2606.09473](https://arxiv.org/abs/2606.09473) —
  *Report the Floor*.
- T3 precedent (BoM, same false-confidence signature):
  `.benchmarks/010_bom_floor_comparison.md`.
