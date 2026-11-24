# Bench 993 — Issue 1239 GOAT cell: `karc_conditioned_readout` (conditioned-KARC correction bank)

**Status:** G1/G2/G3/G4 PASS (as pre-registered); attribution caveat recorded — the keyed σ-gate win TIES the plain keyed-bias control; the promotion-bearing fixed-gate arm A3 won its own gate.
**Verdict:** A4 WIN vs A1 (r̄ −1.894 ± 0.175, δ 1.000) AND A5 wide not-WIN vs A4 (r̄ −0.236 ± 0.241) — the branch claim survives the honest-wide control (1238-G3's shape did not fire). **Attribution: A4−A1c r̄ +0.747 → the keyed win carries NO margin beyond the plain keyed-bias control — credited to the CONDITION KEYING only via the A3 fixed-gate result, never to the σ-gating.** Promotion-bearing A3 (keyed-mean, fixed gate) vs A1: **−3.443 pp WIN**. Stays opt-in under `estimators` per the issue's promotion ceiling (runtime wiring is the follow-up row); any default promotion needs the Feature-Flag-Discipline proposal — never auto-promoted from this cell.

## Provenance (box state)

- Rig: **amo** (Apple silicon builder), SHARED lane, release profile. ⚠ **No bench_preflight PROVENANCE line in this suite's output** (the first corrected suite printed none; disclosed rather than invented). Box-state mitigation, both directions:
  1. the G2 wall cell is the interleaved (a,b) chunk-pair median design (the sequential-A/B box-noise law, 200 pairs × 128 calls/arm) — the sanctioned defense against exactly this;
  2. the walls agree with the independent m3 dev-loop cell measured earlier the same day (corpus 1.41× vs 1.420×, HLA-ish 1.14× vs 1.139×) — two postures, same verdict, each comfortably under the 1.5× bar.
- **Post-hoc PROVENANCE capture (same quiet window, minutes after the cell, amo idle):** `PROVENANCE: power=AC Power load=2.56 swap=3974.69M canary=skipped powermode=0(auto) ane=ok` (`../riir-reflex/scripts/bench_preflight.sh` on amo, run via ssh immediately after the job drained — the same AC/auto/idle posture the cell ran under; quoted as corroboration, never as a substitute for an in-suite line).
- Commit: `297f9412a` (develop, clean checkout — the run's first line pins it).
- Job: amo `56176ff0` (log `01a20e8c-c7c7-7312-ba73-861956176ff0.log`); the FIRST corrected-suite attempt failed for a mundane reason — the original suite named a test target that does not exist (cargo's available-targets dump was its whole log); no gate code ever failed.

## The cell (as pre-registered in `.issues/1239_*`, PINNED block)

Corpora: C1 regime-switching double-scroll + exogenous condition drive (blocks of 200 steps, c ∈ {0,1}; 5 observation-noise seeds σ = 0.01), C2 pure double-scroll (c ≡ 0 — corrections must abstain), C3 3-channel multi-sine (period 20). Splits fit 0–60 % / gate-calibration 60–70 % / test 70–100 %; one-step NRMSE; train NRMSE reported (gap check). D=3, K=4 delays, M=8 Chebyshev, d_h=96, ridge λ = 1e-4. Arms at matched bytes (base 288 f32; 2 corrections × 99 = 198 → 486):

| arm | f32 | NRMSE mean | r̄ vs A1 (pp) | train NRMSE |
|---|---|---|---|---|
| A1 base | 288 | 0.0378 | 0.000 ±0.000 | 0.0353 |
| A2 mux (SELECT control) | 288 | 0.0417 | **+10.203** ±0.196 | — |
| A3 keyed-mean (fixed gate) | 486 | 0.0365 | **−3.443** ±0.100 | — |
| A3b keyed Σ/√2 (Bench-992 scale control) | 486 | 0.0365 | −3.351 ±0.052 | — |
| A4 σ-gated (the conditioned arm) | 486 | 0.0371 | −1.894 ±0.175 | — |
| A5 single wide M=13 (honest-wide control) | 468 | 0.0370 | −2.127 ±0.180 | 0.0337 |
| A6 private M=7 ×2 (Table-5 control) | 504 | 0.0381 | +0.842 ±0.286 | — |
| A1c base + per-condition bias (attribution control) | 294 | 0.0368 | −2.641 ±0.034 | — |
| A1k per-condition full readouts | 576 | 0.0366 | −3.280 ±0.121 | 0.0330 |
| SeasonalNaive floor | 0 | 0.1164 | +207.836 ±0.854 | — |

## Gates

- **G1 (quality)**: A4 WIN vs A1 on C1 (−1.894 ± 0.175, δ 1.000) **PASS**; A5 not-WIN vs A4 (−0.236 ± 0.241) **PASS** — the wide fit does NOT beat the bank at matched bytes; no demote-to-null-record.
- **G2 (wall)**: interleaved median wall ratio A4/A1 ≤ 1.5×, gates ARMED: **corpus D=3 d_h=96: 1.420×; HLA-ish D=8 d_h=192: 1.139×** — PASS both shapes.
- **G3 (no-regression)**: C2 A4 r −0.071 pp (abstain near-exact, A4 == A1 to 4 decimals) **PASS**; C3 A4 r +0.000 pp **PASS** — corrections abstain where the base is already good. SeasonalNaive floor reported beside C3 (0.0404 vs A4 0.0000 — G5 floor preserved; Report-the-Floor carried by the harness printout).
- **G4 (alloc)**: `karc_conditioned_g4_alloc` 2/2 PASS (apply/observe/apply_fixed zero-alloc; CountingAllocator).
- A2 note: the SELECT control LOST (+10.2 pp) — routing between KARC and SeasonalNaive is the wrong axis on this corpus; the correction-bank composition beat both the mux and the private-cache A6 (+0.84 pp), the MBT Table-5 leg.

## Reading (what this cell does and does not claim)

- The branch-composition claim survives: a rank-1 correction bank over the incumbent base fit beats the base at matched bytes, and the honest-wide control does not beat the bank.
- The σ-gating does NOT earn its complexity on this corpus: the fixed-gate A3 beats A4 (−3.443 vs −1.894) AND beats A1c (−3.443 vs −2.641), so the FIXED keyed-mean bank is the best simple arm and the promotion-bearing result. The gate's abstain-first shape still buys the G3 exactness (it abstains perfectly on C2/C3), but on C1 it only throttles a bank that would have helped fully open.
- A3b (Σ/√2) ≈ A3 (−3.351 vs −3.443): the Bench-992 scale-control lesson does not invert here (train/held-out gap check clean — A3's train NRMSE not reported by the harness; A1k's gap 0.0330→0.0366 is small).
- NO GOAT promotion claim: opt-in under `estimators` per the issue's promotion ceiling; the runtime consumer (riir-engine `karc_bridge` condition-keyed corrections beside `npc_directions`) is the follow-up row, feature `karc_conditioned_runtime` opt-in, unarmed boot byte-identical.

## What remains (from the issue)

- [ ] Promotion proposal (separate filing): only A3's fixed-gate result carries; the proposal must price the A1c attribution caveat (a per-condition bias capture alone took 2.641 of the 3.443 pp).
- [ ] Follow-up row: riir-engine `karc_bridge` consumer wiring (separate claim).
