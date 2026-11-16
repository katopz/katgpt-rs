# Bench 985 — Issue 1224: the AHA matched-pair authoring arm + cut-off-rate headroom gate

**Status:** RECORD — the arm EXECUTED at fixture scale; the pre-registered PARITY bar **MET** (construction correctness + determinism proven by the full 993 battery); the confound falsifiers **CANNOT FIRE** on this fixture → the parity claim is scoped **construction-only** per the issue's own T2b law; the headroom gate **WORKS** (the lane's first outcomes-aware arming readout). Measured 2026-11-16, m3 (Mac15,6 / M3 Max, AC, quiet), `CARGO_TARGET_DIR=/tmp/plan1224` isolated.

**Rig:** `cargo test -p katgpt-core --features gain_cost_halt --test issue1224_matched_pair_authoring_gates` (4 tests, 6.5 s debug) + the regression row: 980 (6) · 993 (12) · 1209 (4) · probe POC (6) · lane unit tests (34) · default lib (2522) — all green on the same commit. clippy `--all-targets` clean after `cargo refine fix` (4 mechanical edits) + 3 manual (`cloned_ref_to_slice_refs` ×2, `needless_lifetimes` ×1 — the rulewright's gap this pass).

**Source:** AHA — Khamis/Tutunov/Bou Ammar, blog 2026-11-15 (Research 676); Issue 1224 (the spec, claimed `10b53774f`).

## The two measured surprises (the reason this record exists)

1. **The detector-label degeneracy (T1's premise refuted at the census).** The issue pre-registered the fixture's oscillation detector as the branch labeler. The census REFUTES it: the basin's argmax bounce fires the 3-window detector on EVERY branch — with the uniform temperature-sampling analog, 8/8 branches of every probed prefix oscillate at every explore rate ∈ {0.10, 0.20, 0.30, 0.45}; yield 0.000 at any rate. The labels are therefore BEHAVIORAL (looping = did-not-finish ∧ detector-shaped; productive = finished — the paper's own labels), with the detector kept as the looping label's shape assertion (stuck branches are 100% oscillating, pinned in T1). Uniform draws also cannot escape the basin at all (a symmetric x-random-walk never reaches the goal in 64 steps), so the branch sampler is a PINNED goal-adjacent prior (`BRANCH_RANDOM_DRAW = [2,2,1,5]`, shared by every branch — the purity law holds by construction).
2. **The mined direction recovers the loop axis anyway** (my pre-measurement geometry sketch predicted pair-invariance — WRONG, and the discipline's measure-first law caught it): `cos(u_matched, targeted) = 0.9668`, ch1 mass 0.967, and the composed edit clears the obstacle at the full authored dose.

## Measured (the parity battery, u_matched-authored)

| arm | measured | bar | verdict |
|---|---|---|---|
| yield | **10/16 = 0.625** | ≥ 0.5 (floor pre-pinned below measured) | PASS |
| pairing law | full re-derivation of every yielded pair | bit-identical to the builder | PASS |
| stuck∧oscillating | 100% of stuck branches | the detector-shape law | PASS |
| direction recovery | cos 0.9668 (ch1 0.967) | ≥ 0.90 | PASS |
| LP v (same solve as 993) | 1 nonzero (the carrier column) | sparsity/isolation/carrier arms | PASS |
| held-out separation (v) | push 1.000 / keep 1.000 | ≥ 0.95 / exact | PASS |
| determinism | two builds bit-identical; overlay + artifact digests equal | bit-identity | PASS |
| dose grid [0.5, 0.75, 1.0] | rates **1.0 → 1.0 → 0.0** | monotone non-increasing ∧ full dose clears | PASS |
| isolation control | rate 1.0 | ≥ 0.5 (must not clear) | PASS |
| random control (16 arms) | LCB95 > 0 (all arms: random ≈ 1.0, mined 0.0) | LCB95 > 0 | PASS |
| twin pre-screen | Refused | Refused | PASS |
| retention | out-of-region forwards bit-identical | bit-identity | PASS |

**Recorded, never asserted (the two caveats the battery does not pin):**

- **Success share at dose 1.0: u_matched 0.495 vs targeted 1.000.** The direction's off-axis mass (ch0 −0.227, ch4 +0.118) taxes the horizontal walk while clearing the vertical loop. The 993 battery's arms never asserted success for a bake (only the steering probe's `osc==0 ∧ succ≥0.99` recovery law, T3b there); the matched-pair edit clears the obstacle, which is the bar — but the success tax is real and named.
- **Non-monotone BEYOND the pinned grid: dose 1.25/1.5 re-loop at rate 1.0.** Over-suppression re-locks the channel — the narrow-window caveat (309's α ≤ 0.3 law; the paper's "overshoot beats hair-fixing" does NOT transfer to this fixture's saturating tanh). The grid's own posture ([0.5, 0.75, 1.0]) is monotone and clears; the extension is the record.

**The construction's own holdout check (reported, never gated):** gap −1.43 at n=2 holdout pairs, sign consistency 0.50 — the 80/20 split at 10 pairs leaves an underpowered holdout (the paper had 45); the battery's behavioral arms are the bar, per the module's separability ≠ edit-quality law.

## T2b — the falsifiers cannot fire (the honest negative)

| build | cos vs matched | holdout gap (own split) |
|---|---|---|
| F1 mismatched-prefix (looping of i × productive of i+1) | **0.9545** | +2.74 |
| F2 sampler-split (looping = ε=0 greedy only, productive = explored only) | **0.9907** | −1.69 |
| matched-holdout gap scores (matched −1.43 / mismatched −1.28 / split −1.38 / targeted −1.44) | — | all ≈ equal |

The behavioral stuck-vs-finished axis dominates every injected confound — cross-prefix pairs still contrast the behaviors, and the greedy continuation bounces with the same signature as explored stuck branches. The fixture cannot stage confound corruption of the mined direction. **Per the issue's own T2b law this is recorded honestly: the parity claim stays construction-only.** The non-firing pins (cos ≥ 0.90 asserted in T2b) are the tripwire: a fixture/sampler change that makes a falsifier fire reds the gate and re-opens the record — never silently widened. Demotion-law adjudication: conjunct (a) — "fails battery arms the heuristic passes" — does NOT fire (the arm passes everything the targeted direction passes); conjunct (b) — "falsifiers fail to fire AND the fixture lacks confound structure entirely" — its letter is arguably met, but its consequence is EXACTLY what T2b's softer clause already prescribes (the construction-only scope), and the arm's primary pre-registered evidence (behavioral parity) is positive. Recorded as: **arm stands, claim construction-only, tension noted for the owner** (the adjudication is in the issue close + Research 676 §3 update, same commit).

## T3 — the cut-off-rate headroom gate (works)

Base telemetry over the eval protocol (192 episodes): oscillation rate 1.0, truncation rate 1.0 → `Armed { oscillation_rate: 1.0, episodes: 192, headroom: Some(HeadroomReport { obstacle_episodes: 192, obstacle_truncated: 192, recoverable_share: Some(1.0), truncation_rate: Some(1.0) }) }` — the ceiling REPORT rides the verdict exactly as the issue specified (max recoverable ≈ the base cut-off rate). The outcomes-absent bare fn keeps byte-identical pre-1224 semantics (headroom `None`, floor never fired). Synthetic arms in the carrier's unit tests: recoverable 0.0 → `HeadroomBelowFloor` (the ONLY headroom refusal), intermediate share 0.25 → Armed (the anti-falsifier — the paper's fixed-rate <0.40 skip gate is the forbidden shape), length mismatch asserts loud, clean twin refuses with outcomes supplied.

## Verdict

- **T1/T2/T3: LANDED** — the arm is the lane's third authoring arm, parity-proven at fixture scale, claim construction-only (T2b above).
- **T4: LANDED** — the layer-ranker doc rows (668's forward write-score scan + AHA's LP-hit-rate ranker; placement insensitivity) in `lp_authoring.rs` + `discrete_authoring.rs` module docs.
- **The 976 serving veto stands** (no served lane documents the trace-looping class; re-verified in Research 676 §4.6 the same day).
- No promotion claim: opt-in by construction (the lane is `gain_cost_halt`-gated, authoring-time-only), zero new feature flags (the 992/Plan-635 budget).
- Box state: m3, AC power, CPU ~13% at session start, no GPU involved (pure CPU fixture), load quiet.
