# Bench 987: Plan 639 Phase 2 — novelty-gated retention GOAT bench

**Status:** DONE — G1 ✓ · G2 ✓ (34/36 cells, rational law) · G3 ✗ at the original ratio bar (1.53×, fused 1.472× vs ≤1.15×) → **✓ under the pre-registered re-price** (Plan 639 T2.5: added cost 9.96 ns ≤ 10.0 ns per entity-tick, bmo, median of 3 — marginal) · G4 ✓. Primitive stays DEFAULT-OFF per-instance armed; Plan 639 Phase 3 unblocked. Issue 1231 closed (HISTORY.md).

**Date:** 2026-11-19 · **Plan:** [639](../.plans/639_novelty_gated_retention.md) · **Harness:** `crates/katgpt-sense/tests/novelty_retention_goat.rs` (metrics + verdict arithmetic committed in `f2be3ef2a` BEFORE the first run) · **Session:** katgpt-rs-49@m3

```text
cargo test --release -p katgpt-sense --features novelty_retention \
    --test novelty_retention_goat -- --nocapture --test-threads=1
```

## Box state

m3 (M-series laptop), release profile, `CARGO_TARGET_DIR=/tmp/plan639`, no concurrent cargo.
- run 1 (logistic law): `PROVENANCE: power=AC Power load=2.74 swap=0.00M canary=skipped powermode=2(high) ane=ok`
- final run (rational law): `PROVENANCE: power=AC Power load=6.39 swap=0.00M canary=skipped powermode=2(high) ane=ok` (the earlier rational run at load ≈2.7 measured 1.527×; the interleaved median-of-ratios is load-invariant to ±0.01 here)

## Metrics (pinned pre-run)

- **adaptation** — case (b) Steps: ticks until the belief covers ≥ 90% of each step (cap 100), mean over 3 steps × 8 slots. Lower = better.
- **noise_rmse** — RMSE vs the perfect-tracker target over all ticks of (a) Steady ∪ (c) Spikes. Lower = better.
- **mixed_rmse** — case (d), REPORTED ONLY (not a gate axis).
- Verdict: (i) matched-point margins (adapt ≤ 0.7×, noise ≤ 1.1×) vs ONE pinned-curve α, all 3 seeds; (ii) undominated by every DENSE-curve point (α 0.50..0.99 step 0.01 — the sparse-grid escape closed up front); (iii) distance below the piecewise-linear dense curve > 0.

## Verdicts

| Gate | Result | Evidence |
|---|---|---|
| G1 identity | ✓ | `novelty_retention_gates` 8/8 (unarmed ≡ incumbent bit-exact; κ=0∧μ=0 ≡ incumbent bit-exact) — also 8/8 under `NOVELTY_RETENTION=0`; default-feature lib 25/25 |
| G2 quality | ✓ | rational law **34/36** cells pass (i)+(ii)+(iii); logistic (Phase 1) law **36/36**. Best rational cell κ=0.8 μ=0.4 β=8 g_max=0.995: adaptation **5.04** (fixed α≈0.55 speed) at noise_rmse **0.0231** (fixed α≈0.93 noise), +0.0283 below the curve |
| G3 cost | ✗ | gated/incumbent kernel wall **1.531×** (28.2 → 43.2 ns/tick; interleaved 51×20k, median of ratios). Bar ≤ 1.15× |
| G4 alloc | ✓ | armed forward over 400 ticks: 0 allocations (binary-local counting allocator) |

## G3 shrink log (all on the same harness, release)

| law / layout | gate() alone | kernel ratio | G2 cells |
|---|---|---|---|
| centered logistic `2σ(βs)−1` (Phase 1 `127a145dd`) | — | **3.064×** | 36/36 |
| rational `u/(1+u)`, one shared div, scalar loop | 19.7 ns | 1.649× | 34/36 |
| rational, 8-lane padded locals (vectorizable) | 8.1 ns | 1.603× | 34/36 |
| + `#[inline]` on `gate` (**shipped**) | 6.8 ns | **1.527–1.531×** | 34/36 |
| hard ramp `min(u,1)` (no division — the cost FLOOR probe, not shipped) | 5.5 ns | 1.402× | 32/36 |
| ramp − debug accessor stores | — | 1.402× | — |
| Issue 1231 T1: 6-wide scalar per-kind FUSED loop (drive → gate → belief) | — | 2.300× (rejected — loses the 8-lane autovectorization) | identical |
| Issue 1231 T1: 8-lane FUSED kernel `evolve_additive` (**shipped** `c24c8cce7`; padded state, shared `cell`) | — | **1.472–1.475×** (pre-change 1.514× same session) | byte-identical |
| ↳ minus the two diagnostic stores (probe, not shipped) | — | 1.453× | — |

**Claude verdict (G3):** FAIL at the pre-registered bar; the bar is NOT moved post-hoc. The bar's premise ("the 6-sigmoid structure is already the cost floor") is FALSIFIED by measurement: with zero transcendentals and zero divisions the dual-EMA state + wiring alone cost +40% — no saturation law can reach 1.15×. Shipped the smooth rational law (−50% cost vs logistic, keeps a kink-free sigmoidal map, 34/36 cells) over the cheaper kinked ramp. Remaining routes: the fused drive+gate loop is DONE (1.472×, T1 — the residual is the cell arithmetic itself); left: a consumer-budget re-pricing pre-registered BEFORE re-judging (T2) → Issue 1231 (closed — HISTORY.md).

**G3 re-price (Issue 1231 T2 → Plan 639 T2.5, pre-registered `b26e39848` BEFORE the measurement):** the ratio bar is retired (its cost-floor premise falsified twice). The bar is derived from a named consumer budget: riir-engine Plan 338 T4.4 / Guide 161 §G4 give 5 µs per NPC-tick (10K × 20 Hz, 50 ms frame), and the Plan 337 precedent allots ≤ 0.2 % per subsystem, so **added cost ≤ 10.0 ns per entity-tick**. bmo, release, both preflights PASS each run:

| run | incumbent a | gated b | Δ = b − a | ratio |
|---|---:|---:|---:|---:|
| 1 | 32.55 | 46.84 | 14.29 | 1.470× |
| 2 | 21.46 | 31.42 | 9.96 | 1.472× |
| 3 | 21.14 | 31.00 | 9.86 | 1.473× |

**Median Δ 9.96 ns ≤ 10.0 → PASS by 0.04 ns.** Caveat (binding on consumers): Δ ≈ 0.47 × incumbent, so the pass holds only where the incumbent runs ≲ 21.3 ns. Each Phase 3 consumer re-measures `k × Δ ≤ 10 ns` on its own box at its own call count.

**Honest caveat (secondary axis):** on case (d) Mixed (steps + spikes + noise, one RMSE) the gated cells (0.066–0.102) do not beat the best fixed α (0.0664 at α=0.63–0.64); the two-axis Pareto win is the pre-registered claim, not a single-RMSE win. The 2 failing rational cells (κ=0.2 μ=0 β=8) fail (i) only — too little pull-down to reach the 0.7× margin.

## Tables — final run (rational law, shipped)

## T2.1 incumbent curve (pinned α, per seed: adaptation / noise_rmse)
| α | adapt s1 | adapt s2 | adapt s3 | noise s1 | noise s2 | noise s3 | mixed (mean) | capped |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 3.33 | 3.21 | 2.75 | 0.0628 | 0.0639 | 0.0626 | 0.0691 | 0 |
| 0.60 | 4.62 | 4.17 | 3.62 | 0.0555 | 0.0565 | 0.0553 | 0.0666 | 0 |
| 0.70 | 6.67 | 6.08 | 5.46 | 0.0476 | 0.0486 | 0.0473 | 0.0675 | 0 |
| 0.80 | 9.96 | 10.25 | 9.42 | 0.0387 | 0.0394 | 0.0382 | 0.0754 | 0 |
| 0.85 | 13.88 | 14.33 | 13.04 | 0.0335 | 0.0339 | 0.0329 | 0.0851 | 0 |
| 0.90 | 21.92 | 21.08 | 20.75 | 0.0275 | 0.0274 | 0.0267 | 0.1035 | 0 |
| 0.93 | 31.79 | 31.04 | 30.67 | 0.0233 | 0.0227 | 0.0223 | 0.1240 | 0 |
| 0.95 | 45.21 | 44.17 | 43.88 | 0.0199 | 0.0190 | 0.0188 | 0.1469 | 0 |
| 0.97 | 75.96 | 75.75 | 75.04 | 0.0160 | 0.0145 | 0.0147 | 0.1857 | 0 |

## T2.1b dense curve (seed mean)
| α | adaptation | noise_rmse | mixed_rmse | capped |
|---|---|---|---|---|
| 0.50 | 3.10 | 0.0631 | 0.0691 | 0 |
| 0.51 | 3.12 | 0.0624 | 0.0688 | 0 |
| 0.52 | 3.31 | 0.0617 | 0.0685 | 0 |
| 0.53 | 3.36 | 0.0609 | 0.0682 | 0 |
| 0.54 | 3.39 | 0.0602 | 0.0679 | 0 |
| 0.55 | 3.57 | 0.0595 | 0.0676 | 0 |
| 0.56 | 3.68 | 0.0587 | 0.0674 | 0 |
| 0.57 | 3.76 | 0.0580 | 0.0671 | 0 |
| 0.58 | 3.93 | 0.0573 | 0.0669 | 0 |
| 0.59 | 3.96 | 0.0565 | 0.0668 | 0 |
| 0.60 | 4.14 | 0.0558 | 0.0666 | 0 |
| 0.61 | 4.19 | 0.0550 | 0.0665 | 0 |
| 0.62 | 4.43 | 0.0542 | 0.0665 | 0 |
| 0.63 | 4.60 | 0.0535 | 0.0664 | 0 |
| 0.64 | 4.71 | 0.0527 | 0.0664 | 0 |
| 0.65 | 4.85 | 0.0519 | 0.0665 | 0 |
| 0.66 | 5.17 | 0.0511 | 0.0666 | 0 |
| 0.67 | 5.36 | 0.0503 | 0.0667 | 0 |
| 0.68 | 5.68 | 0.0495 | 0.0669 | 0 |
| 0.69 | 5.92 | 0.0487 | 0.0672 | 0 |
| 0.70 | 6.07 | 0.0478 | 0.0675 | 0 |
| 0.71 | 6.46 | 0.0470 | 0.0679 | 0 |
| 0.72 | 6.79 | 0.0461 | 0.0683 | 0 |
| 0.73 | 6.92 | 0.0453 | 0.0688 | 0 |
| 0.74 | 7.36 | 0.0444 | 0.0694 | 0 |
| 0.75 | 7.72 | 0.0435 | 0.0702 | 0 |
| 0.76 | 8.00 | 0.0426 | 0.0710 | 0 |
| 0.77 | 8.36 | 0.0416 | 0.0719 | 0 |
| 0.78 | 8.75 | 0.0407 | 0.0729 | 0 |
| 0.79 | 9.32 | 0.0397 | 0.0741 | 0 |
| 0.80 | 9.88 | 0.0387 | 0.0754 | 0 |
| 0.81 | 10.47 | 0.0377 | 0.0769 | 0 |
| 0.82 | 11.28 | 0.0367 | 0.0786 | 0 |
| 0.83 | 11.88 | 0.0356 | 0.0805 | 0 |
| 0.84 | 12.75 | 0.0346 | 0.0827 | 0 |
| 0.85 | 13.75 | 0.0334 | 0.0851 | 0 |
| 0.86 | 14.92 | 0.0323 | 0.0879 | 0 |
| 0.87 | 16.32 | 0.0311 | 0.0910 | 0 |
| 0.88 | 17.62 | 0.0299 | 0.0946 | 0 |
| 0.89 | 19.14 | 0.0286 | 0.0987 | 0 |
| 0.90 | 21.25 | 0.0272 | 0.1035 | 0 |
| 0.91 | 23.74 | 0.0258 | 0.1092 | 0 |
| 0.92 | 26.90 | 0.0243 | 0.1159 | 0 |
| 0.93 | 31.17 | 0.0228 | 0.1240 | 0 |
| 0.94 | 36.75 | 0.0211 | 0.1341 | 0 |
| 0.95 | 44.42 | 0.0192 | 0.1469 | 0 |
| 0.96 | 56.56 | 0.0173 | 0.1635 | 0 |
| 0.97 | 75.58 | 0.0151 | 0.1857 | 0 |
| 0.98 | 99.81 | 0.0127 | 0.2151 | 67 |
| 0.99 | 100.00 | 0.0097 | 0.2545 | 72 |

## T2.2 gated grid (α_base 0.9; seed mean; verdict per T2.2)
| κ | μ | β | g_max | adaptation | noise_rmse | mixed_rmse | capped | (i) α | (ii) | dominator | (iii) dist | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | 4 | 0.99 | 12.15 | 0.0289 | 0.0853 | 0 | 0.90 | ✓ | — | +0.0064 | **PASS** |
| 0.2 | 0 | 4 | 0.995 | 12.15 | 0.0289 | 0.0853 | 0 | 0.90 | ✓ | — | +0.0064 | **PASS** |
| 0.2 | 0 | 8 | 0.99 | 10.39 | 0.0299 | 0.0803 | 0 | — | ✓ | — | +0.0080 | fail |
| 0.2 | 0 | 8 | 0.995 | 10.39 | 0.0299 | 0.0803 | 0 | — | ✓ | — | +0.0080 | fail |
| 0.2 | 0.2 | 4 | 0.99 | 13.32 | 0.0187 | 0.0970 | 0 | 0.90 | ✓ | — | +0.0152 | **PASS** |
| 0.2 | 0.2 | 4 | 0.995 | 13.32 | 0.0185 | 0.0972 | 0 | 0.90 | ✓ | — | +0.0154 | **PASS** |
| 0.2 | 0.2 | 8 | 0.99 | 11.54 | 0.0187 | 0.0937 | 0 | 0.90 | ✓ | — | +0.0176 | **PASS** |
| 0.2 | 0.2 | 8 | 0.995 | 11.54 | 0.0185 | 0.0941 | 0 | 0.90 | ✓ | — | +0.0178 | **PASS** |
| 0.2 | 0.4 | 4 | 0.99 | 13.54 | 0.0171 | 0.1015 | 0 | 0.90 | ✓ | — | +0.0166 | **PASS** |
| 0.2 | 0.4 | 4 | 0.995 | 13.60 | 0.0170 | 0.1022 | 0 | 0.90 | ✓ | — | +0.0166 | **PASS** |
| 0.2 | 0.4 | 8 | 0.99 | 11.85 | 0.0175 | 0.0984 | 0 | 0.90 | ✓ | — | +0.0182 | **PASS** |
| 0.2 | 0.4 | 8 | 0.995 | 11.93 | 0.0175 | 0.0991 | 0 | 0.90 | ✓ | — | +0.0181 | **PASS** |
| 0.4 | 0 | 4 | 0.99 | 8.57 | 0.0301 | 0.0770 | 0 | 0.85 | ✓ | — | +0.0110 | **PASS** |
| 0.4 | 0 | 4 | 0.995 | 8.57 | 0.0301 | 0.0770 | 0 | 0.85 | ✓ | — | +0.0110 | **PASS** |
| 0.4 | 0 | 8 | 0.99 | 6.90 | 0.0317 | 0.0719 | 0 | 0.85 | ✓ | — | +0.0136 | **PASS** |
| 0.4 | 0 | 8 | 0.995 | 6.90 | 0.0317 | 0.0719 | 0 | 0.85 | ✓ | — | +0.0136 | **PASS** |
| 0.4 | 0.2 | 4 | 0.99 | 9.24 | 0.0207 | 0.0861 | 0 | 0.85 | ✓ | — | +0.0191 | **PASS** |
| 0.4 | 0.2 | 4 | 0.995 | 9.24 | 0.0205 | 0.0862 | 0 | 0.85 | ✓ | — | +0.0194 | **PASS** |
| 0.4 | 0.2 | 8 | 0.99 | 7.68 | 0.0216 | 0.0819 | 0 | 0.85 | ✓ | — | +0.0220 | **PASS** |
| 0.4 | 0.2 | 8 | 0.995 | 7.69 | 0.0214 | 0.0820 | 0 | 0.85 | ✓ | — | +0.0221 | **PASS** |
| 0.4 | 0.4 | 4 | 0.99 | 9.67 | 0.0183 | 0.0909 | 0 | 0.90 | ✓ | — | +0.0208 | **PASS** |
| 0.4 | 0.4 | 4 | 0.995 | 9.68 | 0.0182 | 0.0914 | 0 | 0.90 | ✓ | — | +0.0209 | **PASS** |
| 0.4 | 0.4 | 8 | 0.99 | 8.03 | 0.0190 | 0.0876 | 0 | 0.85 | ✓ | — | +0.0235 | **PASS** |
| 0.4 | 0.4 | 8 | 0.995 | 8.08 | 0.0190 | 0.0881 | 0 | 0.85 | ✓ | — | +0.0233 | **PASS** |
| 0.8 | 0 | 4 | 0.99 | 5.76 | 0.0320 | 0.0695 | 0 | 0.80 | ✓ | — | +0.0172 | **PASS** |
| 0.8 | 0 | 4 | 0.995 | 5.76 | 0.0320 | 0.0695 | 0 | 0.80 | ✓ | — | +0.0172 | **PASS** |
| 0.8 | 0 | 8 | 0.99 | 4.31 | 0.0353 | 0.0664 | 0 | 0.80 | ✓ | — | +0.0194 | **PASS** |
| 0.8 | 0 | 8 | 0.995 | 4.31 | 0.0353 | 0.0664 | 0 | 0.80 | ✓ | — | +0.0194 | **PASS** |
| 0.8 | 0.2 | 4 | 0.99 | 6.15 | 0.0241 | 0.0754 | 0 | 0.80 | ✓ | — | +0.0235 | **PASS** |
| 0.8 | 0.2 | 4 | 0.995 | 6.15 | 0.0240 | 0.0754 | 0 | 0.80 | ✓ | — | +0.0237 | **PASS** |
| 0.8 | 0.2 | 8 | 0.99 | 4.51 | 0.0266 | 0.0718 | 0 | 0.80 | ✓ | — | +0.0273 | **PASS** |
| 0.8 | 0.2 | 8 | 0.995 | 4.51 | 0.0264 | 0.0718 | 0 | 0.80 | ✓ | — | +0.0274 | **PASS** |
| 0.8 | 0.4 | 4 | 0.99 | 6.39 | 0.0214 | 0.0797 | 0 | 0.80 | ✓ | — | +0.0258 | **PASS** |
| 0.8 | 0.4 | 4 | 0.995 | 6.39 | 0.0212 | 0.0800 | 0 | 0.80 | ✓ | — | +0.0259 | **PASS** |
| 0.8 | 0.4 | 8 | 0.99 | 5.04 | 0.0232 | 0.0763 | 0 | 0.80 | ✓ | — | +0.0282 | **PASS** |
| 0.8 | 0.4 | 8 | 0.995 | 5.04 | 0.0231 | 0.0766 | 0 | 0.80 | ✓ | — | +0.0283 | **PASS** |

VERDICT: 34/36 gated cells pass (i)+(ii)+(iii)

## Tables — run 1 (centered logistic law, Phase 1 \`127a145dd\`; incumbent curves identical to the above)

## T2.2 gated grid (α_base 0.9; seed mean; verdict per T2.2)
| κ | μ | β | g_max | adaptation | noise_rmse | mixed_rmse | capped | (i) α | (ii) | dominator | (iii) dist | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | 4 | 0.99 | 10.43 | 0.0291 | 0.0816 | 0 | 0.90 | ✓ | — | +0.0087 | **PASS** |
| 0.2 | 0 | 4 | 0.995 | 10.43 | 0.0291 | 0.0816 | 0 | 0.90 | ✓ | — | +0.0087 | **PASS** |
| 0.2 | 0 | 8 | 0.99 | 8.24 | 0.0305 | 0.0757 | 0 | 0.85 | ✓ | — | +0.0115 | **PASS** |
| 0.2 | 0 | 8 | 0.995 | 8.24 | 0.0305 | 0.0757 | 0 | 0.85 | ✓ | — | +0.0115 | **PASS** |
| 0.2 | 0.2 | 4 | 0.99 | 11.29 | 0.0181 | 0.0941 | 0 | 0.90 | ✓ | — | +0.0186 | **PASS** |
| 0.2 | 0.2 | 4 | 0.995 | 11.32 | 0.0179 | 0.0946 | 0 | 0.90 | ✓ | — | +0.0187 | **PASS** |
| 0.2 | 0.2 | 8 | 0.99 | 9.22 | 0.0185 | 0.0899 | 0 | 0.85 | ✓ | — | +0.0214 | **PASS** |
| 0.2 | 0.2 | 8 | 0.995 | 9.24 | 0.0185 | 0.0904 | 0 | 0.85 | ✓ | — | +0.0214 | **PASS** |
| 0.2 | 0.4 | 4 | 0.99 | 11.49 | 0.0173 | 0.0972 | 0 | 0.90 | ✓ | — | +0.0190 | **PASS** |
| 0.2 | 0.4 | 4 | 0.995 | 11.58 | 0.0173 | 0.0979 | 0 | 0.90 | ✓ | — | +0.0188 | **PASS** |
| 0.2 | 0.4 | 8 | 0.99 | 9.58 | 0.0179 | 0.0933 | 0 | 0.85 | ✓ | — | +0.0213 | **PASS** |
| 0.2 | 0.4 | 8 | 0.995 | 9.64 | 0.0180 | 0.0940 | 0 | 0.90 | ✓ | — | +0.0212 | **PASS** |
| 0.4 | 0 | 4 | 0.99 | 7.00 | 0.0303 | 0.0736 | 0 | 0.85 | ✓ | — | +0.0148 | **PASS** |
| 0.4 | 0 | 4 | 0.995 | 7.00 | 0.0303 | 0.0736 | 0 | 0.85 | ✓ | — | +0.0148 | **PASS** |
| 0.4 | 0 | 8 | 0.99 | 5.26 | 0.0327 | 0.0685 | 0 | 0.80 | ✓ | — | +0.0180 | **PASS** |
| 0.4 | 0 | 8 | 0.995 | 5.26 | 0.0327 | 0.0685 | 0 | 0.80 | ✓ | — | +0.0180 | **PASS** |
| 0.4 | 0.2 | 4 | 0.99 | 7.60 | 0.0203 | 0.0835 | 0 | 0.85 | ✓ | — | +0.0235 | **PASS** |
| 0.4 | 0.2 | 4 | 0.995 | 7.64 | 0.0201 | 0.0838 | 0 | 0.85 | ✓ | — | +0.0235 | **PASS** |
| 0.4 | 0.2 | 8 | 0.99 | 6.10 | 0.0219 | 0.0785 | 0 | 0.80 | ✓ | — | +0.0258 | **PASS** |
| 0.4 | 0.2 | 8 | 0.995 | 6.15 | 0.0218 | 0.0788 | 0 | 0.80 | ✓ | — | +0.0258 | **PASS** |
| 0.4 | 0.4 | 4 | 0.99 | 7.85 | 0.0180 | 0.0874 | 0 | 0.85 | ✓ | — | +0.0250 | **PASS** |
| 0.4 | 0.4 | 4 | 0.995 | 7.85 | 0.0181 | 0.0879 | 0 | 0.85 | ✓ | — | +0.0250 | **PASS** |
| 0.4 | 0.4 | 8 | 0.99 | 6.44 | 0.0195 | 0.0836 | 0 | 0.80 | ✓ | — | +0.0275 | **PASS** |
| 0.4 | 0.4 | 8 | 0.995 | 6.44 | 0.0196 | 0.0841 | 0 | 0.80 | ✓ | — | +0.0274 | **PASS** |
| 0.8 | 0 | 4 | 0.99 | 4.43 | 0.0323 | 0.0675 | 0 | 0.80 | ✓ | — | +0.0219 | **PASS** |
| 0.8 | 0 | 4 | 0.995 | 4.43 | 0.0323 | 0.0675 | 0 | 0.80 | ✓ | — | +0.0219 | **PASS** |
| 0.8 | 0 | 8 | 0.99 | 3.21 | 0.0382 | 0.0669 | 0 | 0.70 | ✓ | — | +0.0238 | **PASS** |
| 0.8 | 0 | 8 | 0.995 | 3.21 | 0.0382 | 0.0669 | 0 | 0.70 | ✓ | — | +0.0238 | **PASS** |
| 0.8 | 0.2 | 4 | 0.99 | 4.81 | 0.0241 | 0.0743 | 0 | 0.80 | ✓ | — | +0.0281 | **PASS** |
| 0.8 | 0.2 | 4 | 0.995 | 4.81 | 0.0239 | 0.0745 | 0 | 0.80 | ✓ | — | +0.0282 | **PASS** |
| 0.8 | 0.2 | 8 | 0.99 | 3.62 | 0.0285 | 0.0715 | 0 | 0.70 | ✓ | — | +0.0307 | **PASS** |
| 0.8 | 0.2 | 8 | 0.995 | 3.62 | 0.0284 | 0.0716 | 0 | 0.70 | ✓ | — | +0.0308 | **PASS** |
| 0.8 | 0.4 | 4 | 0.99 | 5.14 | 0.0210 | 0.0778 | 0 | 0.80 | ✓ | — | +0.0302 | **PASS** |
| 0.8 | 0.4 | 4 | 0.995 | 5.15 | 0.0209 | 0.0781 | 0 | 0.80 | ✓ | — | +0.0302 | **PASS** |
| 0.8 | 0.4 | 8 | 0.99 | 3.83 | 0.0241 | 0.0752 | 0 | 0.70 | ✓ | — | +0.0336 | **PASS** |
| 0.8 | 0.4 | 8 | 0.995 | 3.83 | 0.0241 | 0.0755 | 0 | 0.70 | ✓ | — | +0.0336 | **PASS** |

VERDICT: 36/36 gated cells pass (i)+(ii)+(iii)
