# Issue 023: Adaptive γ (draft_lookahead) from Entropy-Linear Acceptance Forecast

> **Research:** [katgpt-rs/.research/243_Bebop_Entropy_Bounded_MTP_Acceptance_Adaptive_Gamma.md](../.research/243_Bebop_Entropy_Bounded_MTP_Acceptance_Adaptive_Gamma.md)
> **Source paper:** [arXiv:2606.12370](https://arxiv.org/abs/2606.12370) — Bebop (Qwen Team), §3 (entropy–acceptance bound), §7.6 (adaptive-γ future work)
> **Date:** 2026-06-16
> **Status:** Open — optimization task (Gain tier, GOAT gate required)
> **Type:** Optimization (per AGENTS.md: optimization tasks → issues, not plans)

---

## Problem

`Config::draft_lookahead` (the speculative-decode draft length γ) is a **static** field. It is set once at config time (`micro`=8, `small_target`=5, `bpe`=8, `gemma2_2b`=0, `game`=0) and never adapted at runtime. This wastes compute when the target model's entropy rises mid-generation (acceptance rate drops, but we still draft γ tokens), and leaves throughput on the table when entropy is low (we could safely draft more).

Bebop (arXiv:2606.12370 §3) proves the acceptance rate is **linearly bounded by target entropy**:

```
α ≈ a − b · H(p)
```

with `H(p) = −Σ p_v ln p_v` computable in one pass over the next-token logits, and `a, b` fitted once from warmup. The bound is remarkably stable across model sizes, tasks, and training stages. The paper explicitly flags adaptive-γ from this forecast as future work (§7.6) **without proof** — so this issue is a GOAT-gated experiment, not a guaranteed win.

## Current State (verified)

- `llmexec_guard` (`src/llmexec_guard.rs`) already maps entropy → verification tier, but via **ad-hoc sigmoid** `sigmoid(-steepness·(entropy−0.5) + depth_bonus)`, not a calibrated α forecast. It gates *which verifier runs*, not *how many tokens to draft*.
- `AdaptiveTraceCompactor::observe_entropy` (`src/attn_match/adaptive_cot.rs:159`) already computes EMA entropy per trace — reusable for free.
- `freq_bandit` / `fold_bandit` / `meta_router` already use `acceptance_rate` as bandit reward — the forecast can serve as a Bayesian prior.
- `LeviathanVerifier` already does p/q rejection sampling (the paper's core inference mechanism) — no change needed there.

## Proposed Optimization

1. Add `AcceptanceForecast { a, b, ema_entropy, ema_alpha }` primitive (two-parameter linear model, O(1) forecast after O(vocab) entropy computation, zero-allocation, fits in L1). See Research 243 §4 for the sketch.
2. Feature flag: `adaptive_gamma_forecast` (default-OFF until GOAT gate passes).
3. Wire the forecast into `draft_lookahead` adaptation: `γ_t = clamp(ceil(L_target / α_forecast), γ_min, γ_max)`.
4. Optional: skip speculative decode entirely when `α_forecast < α_breakeven` (single-token path is cheaper than draft+verify cycle).
5. Optional: feed `α_forecast` as a prior into `freq_bandit` to speed convergence.

## GOAT Gate

Benchmark `accepted_tokens/sec` and `μs/step` with feature ON vs OFF, on a workload with **varying entropy** (long CoT reasoning traces where entropy rises mid-generation, per paper Fig. 12b; agentic multi-turn per paper §7.7). 

- **Promote to default** if ≥5% throughput gain with no quality regression (output distribution unchanged — RS preserves unbiased sampling regardless of γ).
- **Demote `llmexec_guard`'s ad-hoc sigmoid** if the calibrated forecast strictly dominates it on the same tier-routing benchmark.
- **Close as won't-fix** if gain < 5% or if the EMA entropy overhead negates the γ-adaptation benefit.

## Risks

- **Unproven in the source paper.** §7.6 is a one-sentence suggestion. The linear bound is proven; the adaptive-γ throughput gain is not. This issue exists to produce that proof (or disproof) for our stack.
- **EMA lag.** Entropy can spike faster than the EMA tracks (α=0.1 → ~10-token lag). May need a min(raw_h, ema_h) guard for spike protection (we already have `RejectionReason::EntropySpike` in TRD for this).
- **Fit cost.** `a, b` must be fitted from warmup data. If the fit is unstable across workloads, the forecast is useless. Paper Fig. 1a suggests it is stable, but that's for Qwen3.5/3.6/3.7 — we need to verify on our models.
- **Interaction with `belief_drafter`.** `belief_drafter_entropy_threshold` is a static 2.0. The forecast could replace it, but the interaction needs testing.

## Tasks

- [ ] **T1** Implement `AcceptanceForecast` primitive in `crates/katgpt-core/src/` (or `src/speculative/`), with `observe_and_forecast` + `adaptive_gamma` methods.
- [ ] **T2** Add feature flag `adaptive_gamma_forecast` (default-OFF).
- [ ] **T3** Fit `a, b` from warmup: run N steps of standard spec-decode, record `(H(p_t), actual_accept_rate_t)` pairs, linear-fit. Expose fit via a `fit_from_warmup` constructor.
- [ ] **T4** Wire `γ_t = adaptive_gamma(...)` into the speculative decode loop, replacing static `draft_lookahead` when feature is ON.
- [ ] **T5** Benchmark on entropy-varying workload (long CoT, agentic multi-turn). Record `accepted_tokens/sec`, `μs/step`, output-distribution KL vs feature-OFF baseline.
- [ ] **T6** GOAT decision: promote / demote / close.

## References

- Research 243 (this issue's distillation)
- Bebop paper §3 (entropy–acceptance bound), §7.6 (adaptive-γ future work)
- `src/llmexec_guard.rs` (existing ad-hoc entropy gate — candidate for demotion)
- `src/attn_match/adaptive_cot.rs:159` (EMA entropy helper to reuse)
- `src/freq_bandit.rs:315` (acceptance-rate bandit reward — forecast becomes prior)
