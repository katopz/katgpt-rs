# Bench 983 — Issue 1210 C3: the supervision-signal position census (drift vs rollout position)

**Status:** RECORD — the C3 diagnostic EXECUTED at the Bench-982 posture; REPORT gate (the issue's own semantics: "an inverted ratio is a finding, not a failure" — no pass/fail exists here). Measured 2026-11-15, amo (Mac17,15 / M5 Ultra, AC power, load 3.21/3.24/3.00), `CARGO_TARGET_DIR=/tmp/i1210c3` isolated; census figures byte-identical across two full runs (the only inter-run delta was a corrected verdict-label wording, below).

**Rig:** `../riir-ai/crates/riir-poc/benches/supervision_signal_position_census.rs` — `cargo bench -p riir-poc --bench supervision_signal_position_census` (~4 min incl. training; deterministic: SEED 1210, TRAIN_SEED 42 — Bench 982's posture byte-for-byte, so the decode dynamics are comparable across the two C-cells). λ_ref = **0.311** — exactly Bench 982's λ, confirming posture identity.

**Statistic:** `StabilityObservation::topk_drift` (the inter-step restricted JSD between successive armed steps' top-3 sets at a position, `jsd_topk_sets`, ln-2 bounded), read ONLY through the landed `observe_row` seam (`fd3f451e2`), on armed rows (`ArgmaxHeld | ArgmaxFlip`) — the branches the item-5 laws measure. **Mapping (stated, never claimed equivalent):** the paper reads teacher/student divergence per AR rollout position; the local analog reads the drift field per block-diffusion sequence position (low j = near the conditioning prompt, high j = deep in the rollout). The denoise-STEP axis is a different decomposition and is unmeasured here (the follow-up if the position axis shows concentration).

**Capture:** 320 requests × 5 families (64 eval each, 982's corpora exactly) → 51,284 provenance rows, **46,804 armed drift rows** (support floor 200 — cleared ×234).

## Measured

Per-position census (EARLY = positions 0–6, LATE = 7–13):

| pos | armed | flips | flip % | mean-drift | n(non-0) | mean(non-0) |
|---|---|---|---|---|---|---|
| 0 | 3608 | 84 | 2.33 | 0.010309 | 136 | 0.273503 |
| 1 | 3461 | 45 | 1.30 | 0.010521 | 133 | 0.273777 |
| 2 | 3644 | 88 | 2.41 | 0.012460 | 140 | 0.324322 |
| 3 | 2782 | 33 | 1.19 | 0.005187 | 62 | 0.232768 |
| 4 | 3437 | 69 | 2.01 | 0.010269 | 125 | 0.282367 |
| 5 | 3473 | 48 | 1.38 | 0.010858 | 129 | 0.292330 |
| 6 | 3633 | 77 | 2.12 | 0.011503 | 145 | 0.288218 |
| 7 | 2714 | 24 | 0.88 | 0.005279 | 58 | 0.247034 |
| 8 | 3354 | 84 | 2.50 | 0.011139 | 122 | 0.306219 |
| 9 | 3677 | 67 | 1.82 | 0.011910 | 141 | 0.310601 |
| 10 | 3753 | 88 | 2.34 | 0.012276 | 153 | 0.301127 |
| 11 | 2623 | 21 | 0.80 | 0.005210 | 47 | 0.290740 |
| 12 | 3178 | 69 | 2.17 | 0.010402 | 106 | 0.311877 |
| 13 | 3467 | 50 | 1.44 | 0.010278 | 123 | 0.289711 |

| read | ratio | 95% CI (request bootstrap, 10k) |
|---|---|---|
| **primary** (all armed rows) | **1.0694** | **[1.0153, 1.1270]** — excludes 1.0, hi-tail support 250 |
| secondary (nonzero drifts only) | 0.9562 | [0.9121, 1.0039] — straddles 1.0, not resolved |

Per-family point estimates: Alt 1.0873 · PairSwap 1.0165 · Fib 1.1818 · Cycle4 1.1023 · **Noise 0.9744**.

## Verdict: REPORT — direction consistent, magnitude absent at this scale

- **Direction agrees with the paper:** the primary ratio is > 1 with a CI excluding 1.0 — the drift field leans EARLY (decays along the rollout, the paper's washout direction), significant at this scale.
- **Magnitude does NOT transfer:** the concentration is **~7%**, nothing like the paper's 3.3×/17× LM-scale decay ratios. The honest headline: the local field is near-FLAT along the rollout with a small significant early lean — "never transfer on authority" vindicated in the direction, refuted in the magnitude.
- **The spike mass is flat-to-slightly-late:** restricting to nonzero drifts (the spike mass without stall dilution) flips the lean (0.9562, CI straddling 1.0) — the early lean lives in the zero-drift stall rows' POSITION DISTRIBUTION, not in the spikes. Disclosed, not adjudicated.
- **Learnability decomposition (disclosed):** structured families lean early (Fib strongest 1.18), pure Noise leans late (0.9744) — directionally "learnable structure → early supervision concentration". But the ordering does NOT track measured learnability cleanly (Fib > Alt despite Alt being the only family the model actually learns per Bench 802/982) — the read is a posture observation, never a claim.
- **The period-4 dip (the census's most structured observation):** positions 3/7/11 carry conspicuously thinner fields (armed ~2.6–2.8k vs ~3.1–3.8k elsewhere; flip % 0.8–1.2 vs 1.3–2.5; mean-drift ~half) — j ≡ 3 (mod 4) is the predictable phase of the periodic families (Alt/Cycle4/PairSwap are 2/4-periodic), so the model's uncertainty collapses earliest exactly there. Disclosed, not gated.

## En-route: the wrong-verdict class caught by the cell's own output

The first run printed "the inversion is significant" for a ratio of 1.0694 with a CI entirely ABOVE 1.0 — the significance-label branches carried each other's names (`lo > 1.0` was labeled as the inversion). The output's own contradiction (direction label vs significance label) caught it before landing; fixed, re-run, census figures byte-identical. Recorded because it is exactly the defend-wrong class this issue exists for: a wrong VERDICT on correct numbers, caught only because the labels were forced to agree with the arithmetic they describe.

## Substrate-first adjudication (per the issue's mandate)

Consumes ONLY the landed Issue-802/1210 seams (`observe_row`, `SamplerFeatures::from_logits_into`, `jsd_topk_sets` through the tracker). No substrate operator added; the census, request-level bootstrap, and percentile helpers are fixture-local by ungoat discipline (the C2 precedent — no CI primitive is built substrate-side). Bench-local decode stops at the capture (no labels → no forced batch — the engine-shape delta vs 982 is named in the source).

## What C3 leaves open

- **C4 (post-hoc τ-sweep / escalation-band instrument)** rides the SAME capture shape — the per-position (and, next, per-step) drift log this bench now emits is the frozen capture C4's sweep consumes. Staged-unclaimed.
- **C5** stays trigger-gated (the plan-623 cross-host activation channel).
- **C7** (verify-cache posture) staged-unclaimed.
- The denoise-STEP axis of this census (the features already carry `step_norm`) — the natural follow-up if a consumer needs the temporal decomposition.
