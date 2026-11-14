# Bench 981 — Issue 1211 T2: the descent extension's GOAT gate rows vs the outside baselines

**Status:** LANDED — T2 complete (code + record one commit on `develop`; the issue's T2 line cites the hash). T1's incumbent gate rows: `tests/issue1211_descent.rs`. T2 adds: `tests/issue1211_t2_baselines.rs` (+ `fixtures::two_basin` moved in from the T1 file, + `fixtures::greedy_least_loaded`, the second outside baseline).

**Provenance:** amo (Mac17,15, M5 Ultra 30-core), AC power, load 3.61/6.75/9.51 at run (5 sibling agents active — wall times are records, not claims), 12-day uptime. `--release`, isolated `CARGO_TARGET_DIR=/tmp/i1211t2`. Deterministic fold order — every fold number below is byte-reproducible on any node; only the wall columns are box-conditioned.

## Design (what T2 adds over T1)

One yardstick for every arm: the **true fold** via `eval_assignment` (movement measured against the authored `initial`). Necessary because the `randomized_starts` posture shifts each chain's movement baseline to its scattered start (the flag's own caveat doc) — the outcome fold of a randomized chain is provenance, never a gate input. Arms: incumbent (ONE search, K·B sweeps) · descent (K=4 chains @ B, lowest pick; randomized posture on two_basin, seed-only on the movement-anchored consumer family per the caveat law) · FFD (the Plan-620 outside baseline) · greedy-least-loaded (new, balance-first constructive) · the baseline-fitted LEVEL arm (`stop_level` = FFD's fold, abstain armed, seed-only).

## Rows (release, folded objective; violation 0 everywhere)

| family point | incumbent | descent | FFD | greedy | level arm |
|---|---|---|---|---|---|
| two_basin n=24 c=6 wm=1 | 496 | **20** | 1381 | 20 (=) | LevelReached @ chain 0 (496 ≤ 1381) |
| two_basin n=24 c=6 wm=3 | 528 | **60** | 1383 | 60 (=) | fires @ chain 0 |
| two_basin n=24 c=6 wm=8 | 608 | **160** | 1388 | 160 (=) | fires @ chain 0 |
| two_basin n=48 c=8 wm=3 | 1056 | **243** | 2823 | 126 (loss — pinned) | fires @ chain 0 |
| two_basin n=96 c=12 wm=3 | 2112 | **570** | 5703 | 264 (loss — pinned) | fires @ chain 0 |
| shard_topology 64×8 (2-dim, skew, w_move=3) | 70 | **70** | 7174 | 335 | fires @ chain 0 (70 ≤ 7174) |

Wall (records): two_basin descent up to 1.36 s at n=96 (4 serial chains) vs incumbent 99 ms — the K-chain axis is wall-serial by the zero-dep posture; budget equality is on the sweeps axis. The LEVEL arm is the wall shape's answer: it answers at chain 0 in EVERY measured point, leaving 75% of the chains' budget unspent.

## Verdict rows

- **G1 vs incumbent at equal total budget: PASS everywhere** — descent never loses to its own incumbent (4.8×–24.8× better on the two-basin decoy landscape; equal on shard_topology).
- **vs FFD: dominates everywhere** — ~69×–278× on two_basin, ~102× on shard_topology (extends Bench 924's base-solver domination to the extension).
- **vs greedy-least-loaded: 4.8× on the consumer family (PASS); constructive-closed on the synthetic family** — the unit-demand two-basin family's optimum (even spread) IS the greedy's construction, so equality at n=24 and LOSSES at n≥48. **The losses are the T2 measured finding, PINNED as a test** (`two_basin_randomized_losses_to_greedy_at_scale_are_pinned`): the randomized chains stop early on their SCATTERED yardstick (the `randomized_starts` caveat, now measured on the true fold) and carry residual imbalance the constructive arm never has (243 vs 126 at 48/8; 570 vs 264 at 96/12). Gate row 3 is scoped to the consumer class; a flip of the pinned negative forces a re-read of this record.
- **Seed-only no-op extension:** descent == incumbent EXACTLY on shard_topology too — T1's "seed-only chains are a no-op on deterministic index-order-scan landscapes" finding holds on the consumer shape; the multi-solution axis needs start diversity, which pays the shifted-yardstick tax on movement-anchored folds (the pinned negative quantifies it).
- **Never-wrong with a baseline-fitted level: holds** — `LevelReached` never emitted above its level in any point; the falsification posture (T1) composes with an outside-baseline calibration.
- **Abstention untested here** (disarmed in every arm; T1's abstain rows cover it) — the level arm ran with abstain armed and never abstained (chain 0 always crossed the FFD level).

## Scope law + promotion posture

The lane's GOAT case on this evidence: start-anchored repair + decoy escape + compute-shape (fitted-level early answers), NOT accuracy-over-constructives on unit-demand synthetic families (closed — the greedy IS the optimum). Promotion stays gated on a consumer wiring it (riir-refine Issue 237's composition cell is the designed first consumer); the crate feature stays opt-in (`descent`).
