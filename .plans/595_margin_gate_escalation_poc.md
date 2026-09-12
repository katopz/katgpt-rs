# Plan 595 — Margin-gated verification escalation PoC (Issue 745, TriSpec distill)

**Status:** COMPLETE (2026-09-11) — Bench 711 ALL 9 GATES PASS; split verdict (cascade REFUTED at ε, accept mapping VIABLE); margin_gate stays opt-in

**Date:** 2026-09-11
**Issue:** `.issues/745_margin_gate_escalation_poc.md`
**Research:** [`.research/548_TriSpec_Margin_Gated_Verification_Escalation.md`](../.research/548_TriSpec_Margin_Gated_Verification_Escalation.md)
**Bench target:** `.benchmarks/711_margin_gate_escalation_poc.md`

## Substrate reality check (correction to Research 548 §2)

Research 548 claimed "the margin operator is ABSENT workspace-wide" (grepped
`top1_share`/`top2`/`rank_gap`). That grep missed the shipped substrate by
vocabulary: `SamplerFeatures::from_logits`
(`crates/katgpt-forward/src/diffusion_sampler.rs`) has carried `margin`
(top1_prob − top2_prob) since the Plan 399 relocation era (verified in git:
present at `5cc3918d^`, 2026-07-05). What is genuinely ABSENT is the
**threshold gate predicate + the cascade fusion** — the escalation rule
itself. This plan lands the operator in `katgpt-core` (the cascade's crate;
forward→core is the dep direction, so `diffusion_sampler.rs` cannot be
consumed from `pruners/`), reuses `sample_residual_distribution_into`
(`katgpt-core/src/speculative/sampling.rs` — the exact substrate
`d2f_verifier.rs` + `step.rs` consume) as the escalation re-decide, and does
NOT duplicate the d2f softmax-margin (T1's "extend SamplerFeatures" is
satisfied by consuming the existing field; adding a redundant `top2` field to
a 6-feature learned-sampler struct would be a DRY violation — recorded, not
done).

## Design

- Feature `margin_gate = ["indicator_cascade"]` (opt-in, katgpt-core).
- `pruners/margin_gate.rs`:
  - `MarginSplit { top1, top2, gap, argmax }` + `margin_split(&[f32])` —
    zero-alloc two-running-maxima pass (no sort, no alloc).
  - `MarginPolarity { DecisiveTrusted, CoherentTrusted }` — the trust sign is
    a workload-geometry property: TriSpec's token-acceptance domain trusts a
    decisive top verdict (`gap ≥ λ`); the cascade's correlated-evidence
    geometry (Zhou et al. cluster premise, Bench 320) trusts coherent
    multi-indicator evidence (`gap ≤ λ`). Both arms ship; the bench measures
    both (defend-wrong).
  - `MarginGate { lambda, polarity }` → `trusted(gap) -> bool`.
  - `MarginDecision<L> { Clean, Trusted(L), Escalated(L) }` +
    `MarginGatedCascade` — flagged-only semantics preserved (unflagged
    candidates never reach the verifier); `Trusted` skips the verifier; 
    `Escalated` re-decides through the expensive path (stage-2 verifier in
    the cascade mapping; `sample_residual_distribution_into` residual
    substitution in the d2f draft-accept mapping — proven by test).
- Bench `benches/bench_711_margin_gate_escalation_poc.rs` (harness=false,
  required-features): three competitors (probe-bank-only baseline = Bench
  320, margin-gated both polarities, always-verify upper bound) × two toy
  worlds (cluster geometry / decisive geometry); invocation-rate vs
  FPR trade curve over λ; G1–G4 + T4 bandit-λ (UCB1, reward mapped from
  meta_router `acceptance_rate × latency_improvement`).

## Tasks

- [x] T0 Plan + substrate correction recorded (this file) — Research 548's
      "margin operator ABSENT" claim was a vocabulary miss
      (`SamplerFeatures.margin` shipped 2026-07-05, Plan 399 era); recorded in
      the bench doc + issue.
- [x] T1 Margin operator `margin_split` + property tests (uniform → gap 0;
      one-hot → gap = 1 − p₂; sort-reference cross-check incl. ties; single/
      empty-slice edges) — 12/12 unit tests pass. "Extend SamplerFeatures"
      satisfied by CONSUMING the existing field (adding a redundant `top2`
      would be a DRY violation) — recorded, not done.
- [x] T2 `MarginGatedCascade` stage-1 firing predicate (feature `margin_gate`,
      default-off); flagged-only preserved (Clean never reaches the verifier —
      unit-tested); escalation seam proven against
      `sample_residual_distribution_into` (unit test, determinism + residual
      support).
- [x] T3 PoC bench: 3 competitors × 3 worlds (A cluster / B decisive / C
      TriSpec-faithful accept), invocation-vs-regression trade curves, ε-axis
      explicit (ε=0.5%), tail-sensitivity control. Defend-wrong outcome: the
      CASCADE mapping is REFUTED at ε on both A and B (best cells: 77.5% cut
      at 3.3% regression; 22.4% at 8.0%); the ACCEPT mapping is VIABLE (50.5%
      cut at 0.25% regression, λ=0.5) and the tail=2% control fires.
- [x] T4 Bandit-λ arm (T3 split → both arms run): world B paired UCB1 passes
      (−0.74%); world C masked UCB1 passes (−0.19%) AFTER two measured
      findings: (a) the mapped meta-router reward is BLIND to the lossy tail
      (λ=0.3 ties λ=0.5 at 39976.1 raw reward despite 18.9% regression); (b) a
      point-estimate feasibility mask starves a good arm (2 wrongs in 196
      pulls = 1.02% > ε → permanent mask → 42% of rounds on the suboptimal
      arm); the z=3 confidence-bound mask fixes both. Novelty-vs-MARS: the
      λ-mask + reward-blindness finding is the delta recorded; the paper-class
      margin predicate itself is published (per Research 548 Q1).
- [x] T5 GOAT verdict + feature-flag decision: G1–G4 + T4 ALL PASS as a
      MECHANISM, but NOT promoted — opt-in pending a live consumer
      (similarity_inference demotion precedent); the candidate consumer is the
      d2f draft-accept loop (engine-lane owner call). Recorded in the Cargo.toml
      feature comment + bench doc 711.
- [-] T6 Healer follow-up (riir-clippy) — NOT in scope (per issue).
- [x] T7 Docs: bench doc 711, issue 745 status + checkboxes, highwaters
      (.plans 594→595, .benchmarks 710→711), clippy clean (feature on
      `--all-targets` + `--no-default-features`), default-lib regression check
      (1987 passed / 0 failed), commit + push.
