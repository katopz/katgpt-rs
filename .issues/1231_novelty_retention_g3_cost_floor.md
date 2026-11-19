# Issue 1231: novelty_retention G3 cost floor — fused kernel or a pre-registered re-price

**Status:** open — T1 LANDED `c24c8cce7` (1.514× → 1.472×, still > 1.15×) → **T2 in flight (claimed)**: the pre-registered re-price verdict. Blocks Plan 639 Phase 3.
**Claimed:** katgpt-rs-49@amo, 1795076856, expires 2026-11-20T08:27:36Z
**Lease closed:** katgpt-rs-49@amo, 1795076101, expires 2026-11-20T08:15:01Z — T1 landed `c24c8cce7`; T2 left unclaimed

**Origin:** [Bench 987](../.benchmarks/987_novelty_retention_goat.md) (Plan 639 Phase 2). G2 quality PASSES (34/36 cells,
best: fixed-α≈0.55 adaptation speed at fixed-α≈0.93 noise), G1/G4 pass, **G3 FAILS: 1.53× kernel wall vs bar ≤ 1.15×**.
The bar's premise ("the 6-sigmoid structure is already the cost floor") is measured false: a division-free,
transcendental-free hard-ramp probe still costs 1.40× — the per-kind dual-EMA state + wiring alone is ~+40%.
`gate()` alone measures 6.8 ns; the kernel delta is ~15 ns, so ~8 ns is integration (separate drive pass → padded
copies → gate → belief loop).

## Tasks

- [x] **T1** Fused kernel: compute drive, the dual-EMA update and `g` in ONE per-kind loop inside
  `evolve_belief_additive` (armed arm only; unarmed incumbent code path untouched — G1 is the code path). Keep all ops
  elementwise (Issue 940). Re-run `novelty_retention_goat` — G2 cells must be bit-identical to Bench 987 (same law),
  report the new G3 ratio.
  **LANDED `c24c8cce7` (2026-11-19, katgpt-rs-49@amo): G3 1.514× → 1.472× — still FAILS ≤ 1.15×.** Fused in the 8-lane domain (`NoveltyRetention::evolve_additive`: padded state, one shared per-element `cell`, KIND_MAP shuffle once). A 6-wide scalar per-kind fusion measured 2.300× (lost the 8-lane autovectorization; rejected). Dropping the diagnostic stores saves ~1% — the residual ~+47% (~+7.5–12 ns over a 16–26 ns incumbent) is the dual-EMA cell arithmetic itself, so no further fusion closes the gap. G2 sweep table byte-identical to the pre-change run; pin `fused_kernel_is_bit_identical_to_gate`; G1/G4 green. PROVENANCE: `power=AC Power load=5.20 swap=3961.19M canary=skipped powermode=0(auto) ane=ok` (amo, release, 51 interleaved rounds, median).
- [ ] **T2** If T1 still > 1.15×: Claude verdict on a re-price **pre-registered before re-judging** — derive the bar from a
  named consumer's per-tick budget (riir-engine goal_salience / belief kernel entities × tick rate), not from the
  observed ratio. Record the verdict in Plan 639; never move the bar to fit the measurement.
- [ ] **T3** On a G3 pass under the (old or re-priced) bar: tick Plan 639 G3, unblock Phase 3 (T3.1–T3.4 file their own
  issues). On a fail: Plan 639 closes with G2-pass/G3-fail recorded, primitive stays DEFAULT-OFF opt-in.

**Not in scope:** changing the saturation law again (the kinked ramp is 1.40× / 32 of 36 cells — measured, rejected for
the smooth rational law; Bench 987 shrink log).
