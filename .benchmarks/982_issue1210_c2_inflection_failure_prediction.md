# Bench 982 — Issue 1210 C2 inflection-count failure prediction: NO VERDICT (label-census degenerate at the micro posture)

**Status:** RECORD — the C2 gate cell EXECUTED and honestly undecidable at fixture scale; the rows primitive (the real substrate delta) is separately LANDED `fd3f451e2`. The cell re-opens only in a decode regime with a real failure mass (the scale-up condition below). Measured 2026-11-15, m3 (M3 Max, AC, powermode=high, load 2.89), release, byte-identical across two runs.

**Rig:** `../riir-ai/crates/riir-poc/benches/inflection_failure_prediction.rs` — `cargo bench -p riir-poc --bench inflection_failure_prediction` (~3 min incl. training; deterministic: SEED 1210, TRAIN_SEED 42 — the Bench-802 training posture exactly: micro_dllm bumped to 2 layers / n_embd 32 / mlp 128, mask ratio 0.85, 1500 epochs, vocab 27). Counts read ONLY through the new provenance-row seam (`StabilityTracker::observe_row` → `StabilityReason::ArgmaxFlip`); λ from the Bench-802 probe rule (forward-1 confidence p75); label = ground truth (any committed token ≠ target), deliberately NOT Bench 802's self-consistency label (the Noise class demonstrates why).

## Measured

| read | value |
|---|---|
| λ_ref | 0.311 (p75 over 560 probe rows — first-8-of-eval form) |
| failure mass | **315 fail / 320 requests (98.4%)** — the model solves almost nothing at block decode |
| pooled AUC (count → failure) | 0.7206 — **NOT adjudicable**: with 5 successes in 320 requests any near-constant score reproduces ≈0.5, and a real separation is UNDECIDABLE at 5 positives, never "passing" |
| falsifier | UNDECIDABLE — the permutation null needs both classes to permute; the 30/30 census floor refuses |
| within-family AUC | Alt 0.771 (59/64 fail — the only family with both classes); PairSwap/Fib/Cycle4/Noise 0.500 by census (64/64 fail — AUC undefined, reported 0.5) |
| flip means per family | Alt 3.72 · PairSwap 2.80 · Cycle4 2.23 · Noise 2.25 · Fib 2.23 |

**Verdict: NO VERDICT — the C2 premise is untestable at this posture.** The honest reading is the LABEL CENSUS, not the AUC number: a micro model that fails 98.4% of requests cannot furnish the failure/success separation the predictor question needs. The bench now refuses this posture itself (the 30/30 census floor prints UNDECIDABLE and exits NO VERDICT — the green-zero rule applied to a statistic).

## What the cell still measured (disclosed, not gated)

- **Flip counts ARE structured, not noise:** Alt (the learned family) carries ~66% more flips than the unlearnable families (3.72 vs 2.2–2.8) and is the only family whose failures separate by count (within-AUC 0.771 on a 59/5 split). Directionally consistent with C2's premise (harder requests → more unsettled decodes), but with ONE family holding both label classes, that single read is weak evidence, not a gate pass.
- **Determinism:** two full runs byte-identical (seeded streams end to end).
- **The λ discrepancy note:** the first draft used a dedicated probe draw and read λ=0.394 vs the 802 rig's 0.451 — different stream consumption changes the confidence mixture (the p75 of 560 heavily-bimodal values is mixture-sensitive). Fixed to the matched-protocol form (probe = first 8 eval seqs, the rig's `take(8)` shape); the landed bench reads 0.311 on ITS OWN stream, which is the correct self-consistent posture — cross-bench λ equality is NOT a requirement (each bench derives λ from its own data).

## Scale-up condition (the re-open trigger)

A decode regime with **≥ 30 real failures AND ≥ 30 real successes** at the same binary + policy: a stronger micro model (more layers/epochs — the failure mass must drop BELOW ~90%), or a real consumer capture where both classes exist. The cell's own census floor then arms the permutation null and the gate adjudicates for real. Until then, C2 stays staged-unclaimed as a predictor claim; the provenance ROWS remain the landed deliverable (observability value independent of the predictor question).

## Substrate-first adjudication (recorded with the cell, per the issue's mandate)

- **COVERED by the landed substrate:** the divergence datum (`topk_drift` via `jsd_topk_sets`, boundary-lawed), the cross-step state machine (item-5 laws), the point AUC estimator (`DiffusionSampler::evaluate_auc`), the labeled capture shape (`collect_trajectories`).
- **The real delta — BUILT (`fd3f451e2`):** `observe` returned a bare `(u8, f32)` and discarded which branch fired; `observe_row` + `StabilityReason` enumerate the branches (FirstObservation / ArgmaxHeld / ArgmaxFlip / Disarmed / RemaskReset) — the fleet's typed-reason-at-the-observation-seam pattern applied to the tracker. No new feature (rides the `dllm+tri_mode` module gating, the 993/1209 un-gated-inside-parent precedent); the tuple form delegates so the two projections cannot disagree (superset test over a scripted mixed sequence).
- **Fixture-side by ungoat discipline:** the inflection-count fold (a caller-side sum over rows) and this AUC cell — no substrate operator, no GOAT claim.
- **Routed to consumers:** the dynamics rider (escalation-rate decay after corpus folding) — katgpt-rs holds no mutated decision state (the stationary-surface law); the folding lives in riir-refine/reflex.
