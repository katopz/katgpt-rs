# Issue 720 — Convergence-cadence outcome gate: churn plateau = escalate (damp/deliberate/restart), decay = commit; not just halt

**Status:** OPEN — filed 2026-09-03, poc/proof task. **T1 + T2 + T4 DONE 2026-09-03.** Remaining: T3 (collision-blocked — riir-mmorpg-examples is the Issue-092 sibling's active zone) + T5 (deferred by design).

> **Source:** Rodrigues & Kang, "Dissecting Hierarchical Reasoning Models: A Mechanistic Study" (ICML 2026 Mech Interp Workshop; alphaxiv `2609.dissecting-hierarchical-reasoning-models`) — distilled in `.research/529_HRM_Mechanistic_Dissection.md`. **Consumers:** `GainCostLoopHalter` (Plan 304 / Research 282 — halt semantics extension), `forward_looped` deep runs (Issue 717 T1/T2 harness — the detector 717's damping knob lacks), per-NPC belief loops (`katgpt-sense` `evolve_belief` — deliberation/settle consumers in riir-mmorpg-examples Issue 054 L2).

## Why

The paper's Finding 4: on HRM's recurrent refinement, the windowed update-magnitude trajectory classifies outcome long before the final step — solved runs' ‖Δz_H‖ decays (0.30 by step 7-8, cos → 0.998, state norm GROWS to 9.4) while failed runs plateau HIGH (1.46, ~4.9×, cos stalls 0.97, norm stalls 6.8). n=93/107. Signal-diff (Research 529 §table):

- `GainCostLoopHalter` consumes step-size ‖Δh‖ but only for **HALT** (decay = concavity stop; growth = expansion stop). It has no OUTCOME read and no escalation arm. A run that plateaus at high ‖Δh‖ eventually halts, but the caller cannot distinguish "nothing left to gain" from "stuck churning — abort and try something else".
- Issue 717 ships (T3/T4, when landed) damping + tangential knobs with the rule "don't damp unless inference already degrades" — but has no degradation DETECTOR. Cadence is the detector.
- `surprise_norm` / `DerivativeCuriosity` read churn as *novelty* (explore), never as *failure* (abstain/escalate).
- riir-neuron-db `can_freeze` gates consolidation on output convergence (validated by this paper's finding) but is measure-time, not a live predictor.

The gap in one sentence: **"plateau-high" is diagnosable from deltas we already compute, and it warrants different ACTIONS per consumer — damp (lt2), deliberate (NPC), restart-with-new-conjecture (CGSP) — none of which a halt-only signal can express.**

Constraints carried from the source + cousins:
- ABSOLUTE update magnitude, never relative (Issue 717 T6 growing-denominator trap).
- Windowed trajectory shape (decay vs plateau), not a single-step threshold.
- Near-orthogonal successive updates (cos_updates ≈ 0 in the paper) → the churn is rotational; pair with 717-T4 tangential decomposition before choosing radial damping.
- Classification ≠ anti-cheat: this is a think-brain signal, never a sync/raw surface (AGENTS.md domain rules).

## Tasks

- [x] T1: `ConvergenceCadence` probe (katgpt-core, feature-gated `cadence_gate`): zero-alloc ring of last-K update norms (‖Δh‖ or ‖Δbelief‖, caller-fed — the halter and `evolve_belief` both already have the delta in hand), emits `Settled { mag } | Churning { mag, plateau_len }` from decay-ratio + plateau detection. G1: bit-identical when feature off; G4 alloc-free. *(LANDED 2026-09-03, katgpt-rs `99920de2` — 12/12 feature-on tests incl. paper-shape fixtures + G4 0-alloc hot path + shuffled non-vacuity; default 1992/0 bit-unchanged; clippy 0 both states.)*
- [x] T2: Falsifiable A/B on a controlled loop (defend-wrong, riir-poc): three arms on `forward_looped` T=64 — (a) no gate, (b) halt-only (shipped halter), (c) halt + cadence-escalation (on plateau-high: apply 717 damping / restart from perturbed state). Metric: accuracy-at-equal-or-less compute + abort-precision (cadence verdict vs ground-truth solved/failed on a suite with known outcomes). Non-vacuity: gate must FAIL when fed shuffled cadences. *(DONE 2026-09-03, riir-ai `b18b7b2bb` — `crates/riir-poc/tests/cadence_gate_poc.rs` behind the opt-in `cadence_gate_poc` feature, 6/6 gates PASS. Controlled d=16 world, budget 64, three dynamics-generated families: contraction-solvable r∈[0.85,0.95] (upper bound DERIVED from the classifier's K=32 resolution r < 0.958), rotational-churn (8-plane radius-restoring rotation, plateau ‖Δ‖ = 2R·sin(θ/2) constant-high, cos_updates = cos θ), reversal-churn (θ=180°, the halter's home turf — honest complementarity control). Measured: halt-only NEVER fires on rotational churn — burns all 64 steps and commits err≈R with no warning (the Finding-4 gap, pinned per-instance); cadence aborts every rotational run at t=32-33 = **1.96× compute savings** with **precision 1.000 / recall 1.000**; solvable+reversal bit-identical to halt-only (no-regression); shuffled control still fires (46 flags) but precision collapses **1.000 → 0.696** (gap 0.304 ≥ 0.2 bar) — the probe reads windowed SHAPE, not magnitude; damp arm honest-negative (0 compute saved, 0 flags, err 5.75 → 4.64 = 19% partial radius shrink, no recovery) — supports law 3. Defend-wrong paid off twice en route: the first harness's single-plane rotation had a 14-dim fixed-point axis + sub-floor plateaus the gates REFUSED to pass over (world fixed to the 8-plane direct sum); θ=90° measured and EXCLUDED — the halter's patience-2 detector fires on f32 noise at true cos 0 (spurious Oscillation halts at t≈4-5), a documented kernel boundary finding, suite keeps cos θ ≥ 0.259. Honest scope: synthetic closed-form dynamics — mechanism characterization, no production `forward_looped` claim; escalation arms abstain (a failed run's dynamics are uncoupled from the answer — nothing to repair into).)*
- [ ] T3: NPC consumer sketch (riir-mmorpg-examples, Issue 054 L2 deliberation): belief-churn over the think window as an ALTERNATIVE/ADDITIONAL stuck trigger (indecision detection, generalizes position-stuck), + settled-belief early-commit (skip think cycle when no new evidence and cadence settled). Gated, default-off; measure think-tick savings + deliberation precision on the 1000-NPC harness.
- [x] T4: Doc pins: (a) GainCostLoopHalter docs gain the outcome-semantics note (halt ≠ classify); (b) Research 529's three-law combo (absolute Δ / windowed shape / tangential-first) recorded at the probe site; (c) cross-link Issue 717 (its detector) and riir-neuron-db `can_freeze` (its consolidation-side sibling). *(DONE 2026-09-03: (b) landed with T1 `99920de2`; (a) `4cff9830` — "Halt ≠ classification" block on the halter struct incl. the measured theta=90° noise edge; (c) riir-neuron-db `9e45d42` — the live-prediction-sibling note on `can_freeze` (freeze-time gate vs in-loop outcome read, same finding, two cadences).)*
- [ ] T5: `- [-]` deferred unless T2 lands positive: CGSP restart-with-new-conjecture arm (DerivativeCuriosity owns the explore axis; only add the *abandon* axis if T2 shows explore-alone is insufficient).

## References

- Research 529 (this paper's distill), Issue 717 (damping/tangential/f32 + residual trap — T1 here feeds its T3/T4), Plan 304/Research 282 (halter), Plan 277 (surprise/derivative curiosity), Plan 108 (forward_looped), riir-neuron-db `can_freeze`, riir-mmorpg-examples Issue 054 (L2 deliberation)

## Summary

**(1) Original task:** file the convergence-cadence extraction from the HRM dissection paper.
**(2) Accomplished:** issue filed with signal-diff against halter/surprise/can_freeze/717, constraints (absolute Δ, windowed, tangential-first), T1-T5. T1 landed (`99920de2`); T2 landed (riir-poc `b18b7b2bb` — all 6 gates PASS, 1.96× rotational compute savings at precision/recall 1.000, shuffled non-vacuity gap 0.304, damp honest-negative); T4 module-level pins carried by T1's commit.
**(3) What remains:** T3 (riir-mmorpg-examples Issue 054 L2 consumer — belief-churn as an additional stuck trigger + settled early-commit; the T2 evidence unblocks it; **collision-blocked today** — the Issue-092 slice-2 session is actively editing riir-games-mmorpg + the repo's lib-count pins), T5 deferred unless a consumer asks for the abandon axis.
**(4) Active plan state:** this issue (OPEN — T1+T2+T4 DONE; T3 blocked, T5 deferred); Research 529 (RECORD); Issue 717 (its detector — landed).

## T2 verdict record (2026-09-03, riir-poc `b18b7b2bb`)

| gate | result |
|---|---|
| G1 determinism | PASS — all 5 arms bit-identical double runs |
| G2 compute + no-regression | PASS — halt-only 2048 vs cadence 1044 steps on rotational churn (**1.96×**); halt-only pinned never-fire/never-flag (the gap); solvable+reversal bit-identical |
| G3 abort precision/recall | PASS — **1.000 / 1.000** (32/32 rotational flagged, 0 false aborts) |
| G4 shuffled non-vacuity | PASS — control fires 46 flags, precision **1.000 → 0.696**, gap 0.304 (shape, not magnitude) |
| G5 damp honest-negative | PASS — 0 compute saved, 0 flags, err 5.75 → 4.64 (19% radius shrink, no recovery) |

En-route findings recorded in the harness header: (1) single-plane rotation = degenerate world (14-dim fixed-point axis; sub-floor plateaus in [0.5, 1.0) invisible to BOTH signals) — caught by the pre-registered gates, fixed to the 8-plane direct sum; (2) θ=90°: the halter's oscillation detector fires on f32 noise at true cos 0 — a real `GainCostLoopHalter` boundary characteristic, excluded from the suite (cos θ ≥ 0.259), documented for the halter lane.

Run: `CARGO_TARGET_DIR=/tmp/riirpoc_720 cargo test -p riir-poc --features cadence_gate_poc --test cadence_gate_poc -- --nocapture`
