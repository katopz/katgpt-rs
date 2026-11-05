# Bench 978: probe-authored weight-edit derivation PoC — fn-grammar + params-vocab (Research 649)

**Date:** 2026-11-04 (verdict round 2 revised: probe-derived grids, matched blind baseline, held-out configs, failure-rate acceptance; round 3 wording fixes: floor-vs-roundup, cap semantics, blind-sweep arithmetic)
**Feature:** `probe_circuit_grammar` (katgpt-core, implies `weight_edit_overlay`), opt-in, no default consumer
**Research:** [`.research/649_Probe_Authored_Circuit_Grammar.md`](../.research/649_Probe_Authored_Circuit_Grammar.md) — the user-idea fusion: activation probes as the Research-648 lane's author (peek at what activates → fn-grammar parse → params-vocab emission)
**Carrier:** the shipped Issue-980 `WeightEditOverlay` (GOAT G1–G4 PASS, Bench 976) — consumed, not forked
**Test:** `crates/katgpt-core/tests/probe_circuit_grammar_poc.rs` (P0–P3 + P2b, 5 tests, `--features probe_circuit_grammar`)
**Verdict: PoC PASS on FOUR pinned configs — the probe-blind parse recovers every injected constant on all four, and the behavior-verified emission emits the EXACT k₀ on all four.** Stays opt-in (no-default-consumer rule); promotion consumer-gated like 980.

> **Fold note (2026-11-06):** the lane's own pre-registered demotion law has FIRED — the first
> consumer PoC (the twinarm fixture, riir-instinct record 0064) lost its G1 value claim to fair
> continue-train. The lane merged into the carrier: `weight_edit_overlay::probe_circuit_grammar`
> (no separate feature; katgpt-rs `abbb22d57`). This record's PoC-PASS verdict stands as the
> identification/repair record only — never as a value claim.

## The claim this proves

Research 648's weight-edit lane had a reasoning model (or a hand that knew the fixture) as its author. This lane replaces that author with **activation probes**: peek at which neurons activate during failing episodes, parse the failing circuit into a **fn-grammar** record (drive → wire → gap with the band gate), emit the repair as a **params-vocab** sentence over the shipped `EditForm` words — the strength GRID derived from the probe, the WORD chosen by behavior verification. No LLM, no gradient, no fixture knowledge at runtime — the injected constants exist per-config as ground-truth assertions only.

## The pinned config set (constants pinned before the run)

`FixtureParams { carrier_col, gap_row, k0, band }` — design constraints from the fixture's analytic shape (never tuned per run): carrier ∈ {3,4,5}, gap ∈ {1,3,5}, carrier ≠ gap (off-diagonal injections; twin value 0). Qualification (verdict round 2): the CONSTANTS were pinned before the run; the fixture INPUTS for gap rows 3/5 (the smooth y/8 diag channel) were repaired after the first held-out run — see the design-stability note below.

| config | carrier | gap | band | k₀ |
|---|---|---|---|---|
| BASE (the 980 original) | 4 | 1 | 4 | 8 |
| HELD_B | 3 | 5 | 3 | 5 |
| HELD_C | 5 | 3 | 5 | 10 |
| HELD_D | 3 | 1 | 3 | 6 |

## P1 — probe-blind circuit identification, all four configs (the parse)

The parse consumes ONLY rollout taps `(y, h1, y2_pre)` + the readout difference row `W3[down] − W3[up]`. Statistics byte-identical across configs (gate scan form, band grid {2..6}, z-bar 1.0, slope window ±3, grid margin 3 / cap 12):

| config | drive col | gap row | band | gate_z | slope (true k₀) | candidates | digest stable |
|---|---|---|---|---|---|---|---|
| BASE | **4** ✓ | **1** ✓ | **4** ✓ (right grid cell) | 1.378 | −7.5238 (−8) ✓ window | exactly 1 | ✓ |
| HELD_B | **3** ✓ | **5** ✓ | **3** ✓ | 1.399 | −4.6030 (−5) ✓ | exactly 1 | ✓ |
| HELD_C | **5** ✓ | **3** ✓ | **5** ✓ | 1.378 | −9.4471 (−10) ✓ | exactly 1 | ✓ |
| HELD_D | **3** ✓ | **1** ✓ | **3** ✓ | 1.399 | −5.6029 (−6) ✓ | exactly 1 | ✓ |

Twin telemetry refuses at the SHIPPED arming precondition before any parse, on all four configs (the obstacle-vs-ability falsifier).

## P2 — behavior-verified emission (grammar proposes, behavior disposes)

**Emission grid DERIVED from the probe** (`derive_strength_grid(ŝ, margin=3, cap=12)` — contains the true k₀ on all four, asserted) and **acceptance = the FAILURE rate** (fraction of authoring episodes never reaching the goal): an edit that turns oscillation into drift-failure scores 1.0 and is REJECTED — the obstacle-vs-ability law measured directly.

| config | derived grid | emitted k | verdict |
|---|---|---|---|
| BASE | {4..11} | **8** (5 authoring evals) | exact k₀ — the slope FLOOR (k=7) provably fails 8/8 (recorded negative) |
| HELD_B | {1..8} | **5** | exact k₀ |
| HELD_C | {6..12} | **10** | exact k₀ (the window top was silently capped: the un-truncated window is {6..13}) |
| HELD_D | {2..9} | **6** | exact k₀ |

**The slope-bias fact (verdict round 3, stated plainly):** ⌈|ŝ|⌉ = k₀ in ALL FOUR configs — the single-regressor slope UNDERESTIMATES by 0.40–0.55 every time (−7.52→8, −4.60→5, −9.45→10, −5.60→6). So floor fails 4/4 and nearest-rounding fails 1/4 (HELD_C: 9.45 → 9 ≠ 10), while round-UP would have hit 4/4 — but only with the bias direction known in advance, which a real lane never has. **The sweep's job is verification: it picks the rounding without prior knowledge of which way the bias runs**, and its dose arms double as the dose-response evidence. The BASE k=7 word is the FLOOR negative (recorded: oscillates 8/8).

BASE full battery (all PASS): full-eval obstacle 0.0 AND failure 0.0 (16×12 seeded starts) · success parity with the twin · dose-response `{1,1,1,0}` monotone (oscillation statistic, the 980 G1a arm) · neighbor-column isolation null (rate ≥ 0.95) · out-of-region forwards bit-identical + action prefixes identical · provenance digest (the fn-grammar record) round-trips the freeze envelope.

P2b held-outs (all PASS): authoring failure 0 → full-eval obstacle 0 + failure 0 + success parity, with the SAME design constants — no per-config tuning.

## P3 — peek vs blind search (verdict round 2 framing)

- **Reachability control:** 16 norm-matched random-direction arms at the RIGHT column + 16 blind column×direction arms, each dose-sweeping the same grid with the same acceptance: NONE repairs (mean failure rate 1.0); probe 0.0; paired **LCB95(best random − probe) = +1.0 > 0** (every per-seed delta +1). A random 64-dim unit direction reaches `|d[gap]| ≥ 0.875` with probability ≈ 0.
- **Matched blind baseline (the strongest blind searcher):** ±e_gap words at every column, same grid, same acceptance, gap row given free: first-found repair after **69 authoring evals vs the probe's 5 — 13.8×** for parse-and-verify over enumeration (≈ **6 vs 69 counting the probe's one capture rollout**). **The 69 is arithmetic of the search order, not a measurement**: ascending columns, sign − first, 8 strengths → `16·carrier_col + 5` (= 64 failures at cols 0–3 + 5 at col 4). 13.8× is therefore an ORDER-DEPENDENT LOWER BOUND measured on BASE only; under a uniformly random column order the expected cost is ≈ 16·31.5 + 5 ≈ 510 (≈ 100×), worst case ≈ 1,029. The blind first-found lands at the TRUE column (4, −, 8): the failure-rate acceptance killed the plausible compensating edits before they cleared, so the locus difference did not materialize — the honest claim is the eval saving + attribution (the probe names the anomalous column from ONE capture; the sweep tested 69 words to find it).

## Design-stability note (the fixture iteration, recorded honestly)

The first HELD_B/HELD_C run exposed two fixture-design flaws: (a) the HELD_B twin could not reach the goal — the gap row's diag channel carried a constant input whose tiny vertical bias beat the horizontal pull near x=5; (b) HELD_C never ARMED — its oscillation swung active↔zero (−9.9 ↔ 0), which never pushes the consecutive-state cosine negative, so the shipped halter saw no oscillation. Both were FIXED IN THE FIXTURE (gap rows 3/5 got the smooth `y/8` input their diag channel needs — the same input row 1 always had), required by the fixture's own analytic contract (twin centering + halter-visible sign-flip). **The parse statistics were never retuned** — byte-identical across all runs and all four configs. The halter-visibility bound is now recorded as caveat 7 in the research note.

## Posture

- Feature `probe_circuit_grammar = ["weight_edit_overlay"]`, opt-in; no default consumer (metabolic-gate precedent, same posture as 980). Demotion law pre-registered in the Cargo.toml comment: the battery failing identification/repair records the negative and the lane folds into `weight_edit_overlay`.
- Determinism: sequential folds, first-index tie rules, BLAKE3-seeded arms; repeated captures produce byte-identical bindings (asserted per config).
- Box state: macOS arm64 (M3), debug-profile test run, isolated target dir `/tmp/r649_tgt`; full PoC wall ~43 s in debug. Module tests: 9 (incl. the derived-grid window + cap-refusal arms); full katgpt-core lib: 2231 passed; clippy clean at default + feature postures.
