# 745: Margin-gated verification escalation — PoC (TriSpec distill)

**Status:** Open — PoC not started (load-gated this session; box saturated by sibling compute lanes)

**Date:** 2026-09-11
**Research:** [katgpt-rs/.research/548_TriSpec_Margin_Gated_Verification_Escalation.md](../.research/548_TriSpec_Margin_Gated_Verification_Escalation.md)
**Source:** [arXiv:2601.23180](https://arxiv.org/abs/2601.23180) — TriSpec (Qwen Team, Feb 2026)
**Verdict:** Gain (fusion idea, novelty TBD) — class is published (MARS / VIA-SD / TriSpec); the deltas to prove are substrate fusion + bandit-λ.

## Goal

Prove (or refute) that a zero-training top1−top2 margin gate on already-materialized logits can (a) replace/augment `IndicatorCascade`'s stage-1 boolean probe bank with a graded trigger and (b) cut expensive-verifier invocations on the draft-accept path — without the lossy tail TriSpec accepts (≤1% accuracy) exceeding our G1 bar.

## Substrate consumed (all ships — do not re-implement)

- `katgpt-forward/src/d2f/mod.rs` — `SamplerFeatures::from_logits` (add `top2`/`rank_gap`), accept loop, `sample_residual_distribution_into`
- `katgpt-core/src/pruners/indicator_cascade.rs` — stage-1/stage-2 cascade, `IndicatorVerifier` trait
- `katgpt-core/src/speculative/` — TreeNode/DraftResult/Leviathan residual sampling
- `katgpt-attn/src/dash_attn/meta_router.rs` — UCB1 arms, reward `acceptance_rate × latency_improvement`

## Tasks

- [ ] **T1** Margin operator: zero-alloc two-running-maxima pass over logits → `(top1, top2, gap)`; extend `SamplerFeatures`; property test (gap = 0 on uniform, gap = 1−p₂ on one-hot).
- [ ] **T2** Wire as `IndicatorCascade` stage-1 firing predicate (feature `margin_gate`, default-off): flagged-only semantics preserved; escalation path reuses d2f residual substitution.
- [ ] **T3** PoC bench (defend-wrong): three competitors — margin-gated cascade, probe-bank-only cascade (Bench 320 baseline), always-verify upper bound — on a controlled toy scoring task; report invocation-rate vs FPR trade curve. G1: flagged-case accuracy must not regress beyond a stated ε (the lossy axis made explicit).
- [ ] **T4** Bandit-λ arm (only if T3 wins): λ as a UCB1 arm under the existing meta-router reward; compare vs fixed λ∈{0.3, 0.5, 0.7}. Novelty-vs-MARS check (web) before claiming the delta.
- [ ] **T5** GOAT gate: feature flag + bench; promote to default only on G1–G4 pass; demote loser slot if outrun. UQ n/a (no distributional claim — scalar gate).
- [ ] **T6** Healer follow-up (separate repo): file riir-clippy issue for the draft→pre-verify→clippy_verify tier when that repo's sibling claim clears. NOT in this issue's scope.

## Non-goals

- No same-family proxy training (riir-train defer — Research 548 §4).
- No GGUF-runner integration this PoC (toy domain first; league lane is owner-gated).
