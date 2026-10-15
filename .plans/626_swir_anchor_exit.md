# Plan 626: SwiR Anchor-Exit — Boundary-Argmax Span Termination (Modelless)

**Date:** 2026-10-15
> **Status:** Active — Phase 0 (probe / kill gate)
> **Claimed:** katgpt-rs-idle-626, 1792102437, expires 2026-10-18T22:14:00Z
**Research:** [katgpt-rs/.research/622_HybridAgent_Hybrid_Latent_Text_Agents.md](../.research/622_HybridAgent_Hybrid_Latent_Text_Agents.md)
**Source paper:** [arXiv:2610.18861](https://arxiv.org/abs/2610.18861) HybridAgent (anchor exit, trained form) + trained-form prior arXiv:2610.17189 (implicit-readout exit); modelless adaptation unclaimed
**Target:** `crates/katgpt-transformer/src/swir/` (controller extension) + Cargo feature `swir_anchor_exit` (implies `swir_switch_thinking`)
**Depends On:** Plan 275 (SwiR controller + soft embedding), Plan 194 (thinking_cot host), Plan 313 (real-model harness path)
**Cross-informs:** riir-train Plan 461 (E1 publishes the trained upper bound of the same signal)

---

## Goal

Add a **boundary-identity exit signal** beside Plan 275's entropy-trend law: terminate a latent/soft-embedding span when the implicit argmax readout lands in a derived boundary-token set after a minimum dwell, hard-capped at max dwell — composed as `exit = anchor ∨ entropy ∨ cap`, first match wins, with a per-span close-reason tag. The anchor signal is deterministic, membership-check O(1), threshold-free at decision time (candidate boundary tokens are disqualified upstream by T0.2's measured frequency cut), and immune to entropy-plateau chatter; on frozen models it is a *completeness proxy* (controller vocabulary, not model vocabulary), so **Phase 0 is a designed kill gate with TWO measured failure modes**: under-firing (the proxy is empty — decline is a correct answer) and over-firing (the boundary set is so frequent-in-domain that the anchor arm collapses into a fixed min-dwell cap).

**Stack slot:** thinking-mode control (beside `swir`). Feature `swir_anchor_exit`, opt-in until GOAT; the demotable unit is the anchor ARM (0% fire ⇒ demote inert; ~100% fire ⇒ narrow the set; both with measured receipts), never the entropy fallback.

**Controller API constraint (Plan 275 is DEFAULT-ON):** `SwiRController::step(entropy, step_index)` keeps its signature UNCHANGED; the anchor lane enters through a NEW entry point (`step_with_readout`, taking the argmax alongside entropy) so flag-off behavior is byte-identical BY CONSTRUCTION. The argmax is O(V) but computed in the SAME pass as `shannon_entropy` (one walk over the probs computing sum, sum-of-logs, argmax) — effectively free; only the boundary-set membership check is on the hot path (sorted small array + binary search, |B| ≤ ~50).

## Phase 0 — Anchor-Agreement Probe (KILL GATE, two failure modes)

### Tasks

- [ ] **T0.1** `BoundarySet` derivation. DEFAULT arm is MINIMAL: `{</think>, EOS, <end_of_turn>}` only — newline/period-class tokens are a SEPARATE arm (they are frequent argmaxes in math reasoning and would collapse the anchor arm into a fixed min-dwell cap). Config override; sorted small array + binary-search membership; unit tests on construction + membership.
- [ ] **T0.2** Over-firing telemetry: measure how often boundary-set tokens are the argmax in an ordinary explicit-CoT stream on the served model (per boundary token). A token whose ambient argmax frequency is high is disqualified from the set with a number, not a guess.
- [ ] **T0.3** Intent-agreement probe with a NON-SELF-ANSWERING oracle: the oracle greedy-decodes starting at step exit+1 (NOT at the exit step — its first token there IS the anchor's argmax by construction and any ±δ window including step 0 auto-agrees); agreement = oracle emits a `B` member within the window; chance baseline via a shuffled-exit control. Report per-arm (minimal set / punctuation arm).
- [ ] **T0.4** Outcome-based kill arm (the decisive half): on paired items, force exit at the anchor step vs at the entropy step vs at matched-step forced exits; compare answer correctness and tokens-to-answer. This half cannot auto-agree and directly measures whether the anchor fires at BETTER-than-entropy moments.
- [ ] **T0.5** KILL GATE: pre-register BOTH bars in the bench record BEFORE the run. PASS → Phase 1. FAIL (under-firing at chance, or over-firing per T0.2, or outcome arm shows anchor ≤ entropy exits) → record the negative in `.benchmarks/`, set plan status CLOSED-NEGATIVE, entropy-exit remains the only exit law.

## Phase 1 — Controller Extension

### Tasks

- [ ] **T1.1** `ExitSignal` enum `{EntropyTrend, AnchorBoundary, Composite}` + `AnchorExitConfig { boundary_set, min_dwell, max_steps, hysteresis_k }` with defaults (min_dwell 4, hysteresis_k 1, max_steps from existing `max_steps`, boundary_set = the T0.1 minimal arm).
- [ ] **T1.2** `AnchorBoundary` predicate: exit iff `argmax ∈ B ∧ dwell ≥ min_dwell ∧ k-consecutive stability`; membership via sorted-array binary search, zero-alloc (counters only).
- [ ] **T1.3** New entry point `step_with_readout(readout, entropy, step)` (readout = argmax + optional top-2 margin); the existing `step(entropy, step_index)` signature is UNCHANGED and delegates with readout = None — flag-off byte-identity by construction (G5 posture). Single-pass argmax+entropy computation documented for hosts.
- [ ] **T1.4** Composite precedence in `SwiRController::step_with_readout`: anchor ∨ entropy ∨ cap, first match wins; `SpanCloseReason { Anchor, Entropy, Forced }` recorded into `SwiRStats` (per-span reason histogram — the calibration instrument; Anchor at 0% ⇒ arm demotes inert, Anchor ≈ 100% ⇒ set narrows per T0.2 telemetry).
- [ ] **T1.5** Wire through the strategy adapter (`StepDirective` translation); byte-identity test: `step()` (legacy signature) and `step_with_readout` with an EMPTY boundary set are bit-identical to current entropy-only behavior (the G5 posture, by construction and by test).
- [ ] **T1.6** Tests: dwell gating, hysteresis stability, precedence determinism under argmax flicker (oscillation test — no mode chatter), reason-tag accounting, forced-exit materialization (cap feeds the boundary token via the existing `ControlToken` enqueue path), over-firing guard (a set containing a high-ambient-frequency token trips the T0.2-style telemetry in stats).
- [ ] **T1.7** Feature gate `swir_anchor_exit = ["swir_switch_thinking"]`; flag-OFF compiles to nothing (zero-code posture verified).

## Phase 2 — Bench + GOAT Gate

### Tasks

- [ ] **T2.1** Extend the Plan 275 bench harness (`ProblemSource`/`DecodeBackend` traits): arms = explicit-CoT / swir-entropy / swir-anchor-composite; metrics: tokens-to-answer, wall-clock, accuracy, per-span reason histogram, forced-exit rate.
- [ ] **T2.2** G2 efficiency gate ≥1.05× vs entropy-only at equal accuracy (the anchor's value claim: fewer forced exits + fewer entropy false-exits), `--release` + `scripts/bench_preflight.sh` PROVENANCE line. Same honest posture as Plan 275: the accuracy axis was capability-blocked on Gemma 2 2B — efficiency is the primary claim.
- [ ] **T2.3** G3 non-inferiority: paired accuracy vs entropy-only AND vs no-latent (existing stats utilities, LB95).
- [ ] **T2.4** G4 alloc: predicate zero-alloc (counting-allocator canary).
- [ ] **T2.5** G5: flag-off byte-identity across the full `thinking_cot` suite.
- [ ] **T2.6** Record `.benchmarks/NNN`; promotion verdict: if G2 passes AND G3 non-inferior, the anchor ARM joins the default composite (entropy stays as fallback); anchor-arm demote rule per T1.3 receipt.
- [-] **T2.7** (deferred) Margin-gated sharpening `σ(β·(top1−top2 margin) − θ)` with thresholds from measured pairs (sigmoid calibration utilities); any UQ claim rides the conformal-naive floor.
- [-] **T2.8** (deferred) Latent arm in the thinking_cot direct-vs-CoT bandit (one arm addition; only after the probe proves the signal).

## Risks

- **The proxy may be empty on frozen models** — the paper's anchor is meaningful because format-activation SFT trained the projection; our Phase 0 exists to measure exactly this, and the demote path is first-class.
- **The proxy may be EVERYTHING on frozen models** — a boundary set carrying frequent-in-domain tokens (newline/period) collapses the anchor arm into a fixed min-dwell cap; the T0.1 minimal default and the T0.2 over-firing telemetry exist for exactly this, and the set only grows with per-token disqualification numbers behind it.
- **Entropy exit is also deterministic** — the honest deltas are threshold-free operation, flicker resistance, and O(1)-membership; do not claim a determinism win over Plan 275.
- **Probe self-answering** — an oracle window that includes the exit step agrees by construction; T0.3 starts at exit+1 and T0.4 is the outcome-based half that cannot auto-agree.
