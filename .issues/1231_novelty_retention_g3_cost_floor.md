# Issue 1231: novelty_retention G3 cost floor — fused kernel or a pre-registered re-price

**Status:** open — opt/perf task; blocks Plan 639 Phase 3 (consumer wiring). T1 claimed.
**Claimed:** katgpt-rs-49@amo, 1795076101, expires 2026-11-20T08:15:01Z

**Origin:** [Bench 987](../.benchmarks/987_novelty_retention_goat.md) (Plan 639 Phase 2). G2 quality PASSES (34/36 cells,
best: fixed-α≈0.55 adaptation speed at fixed-α≈0.93 noise), G1/G4 pass, **G3 FAILS: 1.53× kernel wall vs bar ≤ 1.15×**.
The bar's premise ("the 6-sigmoid structure is already the cost floor") is measured false: a division-free,
transcendental-free hard-ramp probe still costs 1.40× — the per-kind dual-EMA state + wiring alone is ~+40%.
`gate()` alone measures 6.8 ns; the kernel delta is ~15 ns, so ~8 ns is integration (separate drive pass → padded
copies → gate → belief loop).

## Tasks

- [ ] **T1** Fused kernel: compute drive, the dual-EMA update and `g` in ONE per-kind loop inside
  `evolve_belief_additive` (armed arm only; unarmed incumbent code path untouched — G1 is the code path). Keep all ops
  elementwise (Issue 940). Re-run `novelty_retention_goat` — G2 cells must be bit-identical to Bench 987 (same law),
  report the new G3 ratio.
- [ ] **T2** If T1 still > 1.15×: Claude verdict on a re-price **pre-registered before re-judging** — derive the bar from a
  named consumer's per-tick budget (riir-engine goal_salience / belief kernel entities × tick rate), not from the
  observed ratio. Record the verdict in Plan 639; never move the bar to fit the measurement.
- [ ] **T3** On a G3 pass under the (old or re-priced) bar: tick Plan 639 G3, unblock Phase 3 (T3.1–T3.4 file their own
  issues). On a fail: Plan 639 closes with G2-pass/G3-fail recorded, primitive stays DEFAULT-OFF opt-in.

**Not in scope:** changing the saturation law again (the kinked ramp is 1.40× / 32 of 36 cells — measured, rejected for
the smooth rational law; Bench 987 shrink log).
