# 745: Margin-gated verification escalation — PoC (TriSpec distill)

**Status:** Resolved 2026-09-11 (Plan 595 + Bench 711 — ALL 9 GATES PASS; split verdict: cascade mapping REFUTED at ε, accept mapping VIABLE at tail ≤ 0.5%; `margin_gate` shipped opt-in, promotion blocked on a live consumer)

**Date:** 2026-09-11
**Research:** [katgpt-rs/.research/548_TriSpec_Margin_Gated_Verification_Escalation.md](../.research/548_TriSpec_Margin_Gated_Verification_Escalation.md)
**Plan:** [`.plans/595_margin_gate_escalation_poc.md`](../.plans/595_margin_gate_escalation_poc.md)
**Bench:** [`.benchmarks/711_margin_gate_escalation_poc.md`](../.benchmarks/711_margin_gate_escalation_poc.md)
**Source:** [arXiv:2601.23180](https://arxiv.org/abs/2601.23180) — TriSpec (Qwen Team, Feb 2026)
**Verdict:** Gain (fusion idea, novelty TBD) — class is published (MARS / VIA-SD / TriSpec); the deltas to prove are substrate fusion + bandit-λ. **Measured (Bench 711): substrate-fusion delta REFUTED for the cascade mapping, PROVEN for the accept mapping; bandit-λ delta landed two standalone findings (reward-blindness + mask starvation).**

## Goal

Prove (or refute) that a zero-training top1−top2 margin gate on already-materialized logits can (a) replace/augment `IndicatorCascade`'s stage-1 boolean probe bank with a graded trigger and (b) cut expensive-verifier invocations on the draft-accept path — without the lossy tail TriSpec accepts (≤1% accuracy) exceeding our G1 bar.

## Substrate consumed (all ships — do not re-implement)

- `katgpt-forward/src/d2f/mod.rs` — `SamplerFeatures::from_logits` (add `top2`/`rank_gap`), accept loop, `sample_residual_distribution_into`
- `katgpt-core/src/pruners/indicator_cascade.rs` — stage-1/stage-2 cascade, `IndicatorVerifier` trait
- `katgpt-core/src/speculative/` — TreeNode/DraftResult/Leviathan residual sampling
- `katgpt-attn/src/dash_attn/meta_router.rs` — UCB1 arms, reward `acceptance_rate × latency_improvement`

## Substrate reality (correction to the research + this issue's premise)

The margin operator was NOT absent: `SamplerFeatures::from_logits`
(`katgpt-forward/src/diffusion_sampler.rs`) has carried `margin`
(top1 − top2) since the Plan 399 era — Research 548's grep missed it by
vocabulary (`top2`/`rank_gap` instead of `margin`). What was absent — and
what landed — is the gate predicate + cascade fusion (below). The canonical
vocabulary-translation lesson, again.

## Tasks

- [x] **T1** Margin operator: `margin_split` zero-alloc two-running-maxima
      pass → `(top1, top2, gap, argmax)` in `katgpt-core/pruners/margin_gate.rs`
      (branchless, stable tie-break, 5–7 ns @ N=8); property tests landed
      (gap=0 on uniform, gap=1−p₂ on one-hot, sort-reference cross-check).
      `SamplerFeatures` NOT extended — it already carries `margin`; consuming
      it instead of duplicating `top2` is the DRY-correct reading.
- [x] **T2** Wired as a graded stage-1 firing predicate: `MarginGatedCascade`
      (feature `margin_gate`, default-off, implies `indicator_cascade`);
      flagged-only semantics preserved (Clean never reaches the verifier);
      escalation re-uses `speculative::sample_residual_distribution_into`
      (the exact d2f residual substitution) — seam proven by unit test.
- [x] **T3** PoC bench (defend-wrong, 3 competitors × 3 worlds): the
      invocation-rate vs regression trade curves are in Bench 711. G1 ε=0.5%:
      the cascade mapping REFUTED on both geometries (A: best 77.5% cut at
      3.3% regression; B: best 22.4% at 8.0% — score-domain saturation +
      noise-floor gap overlap); the TriSpec-faithful ACCEPT mapping VIABLE
      (50.5% target-invocation cut at 0.25% regression, λ=0.5) and the
      tail=2% control fires (genuinely ε-sensitive).
- [x] **T4** Bandit-λ arm (T3 split → ran on both mappings): UCB1 ≈ best
      fixed λ on both (paired −0.74% / −0.19%). Two standalone findings: the
      mapped `meta_router::compute_reward` reward is BLIND to the lossy tail
      (trusted-wrong and escalated-wrong both score 0 → the ε-violating λ
      ties the best arm), and a point-estimate feasibility mask starves a
      good arm on sampling noise (2 wrongs in 196 pulls → permanent mask →
      −8.8% reward); the z=3 confidence-bound mask is the fix. Novelty delta
      vs MARS = the masked-constrained-λ formulation + these two measured
      failure modes; the bare margin predicate is published class (per
      Research 548 Q1) — no novelty claim beyond that.
- [x] **T5** GOAT gate: feature flag `margin_gate` (opt-in) + bench; G1–G4 +
      T4 ALL PASS as a mechanism. NOT promoted to default — no live consumer
      (similarity_inference demotion precedent); the candidate consumer is
      the d2f draft-accept loop, an engine-lane owner call (perf-league axis
      change, Research 548 §5). Loser slot: the cascade-mapping polarity
      assumption recorded as refuted rather than demoted (it never shipped
      default).
- [-] **T6** Healer follow-up (separate repo): file riir-clippy issue for the
      draft→pre-verify→clippy_verify tier when that repo's sibling claim
      clears. NOT in this issue's scope (unchanged).

## Non-goals

- No same-family proxy training (riir-train defer — Research 548 §4).
- No GGUF-runner integration this PoC (toy domain first; league lane is owner-gated).
