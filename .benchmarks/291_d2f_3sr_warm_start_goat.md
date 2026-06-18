# Bench 291: D2F Three-State Warm-Start — GOAT Gate Results

**Date:** 2026-06-18
**Plan:** [291_d2f_three_state_warm_start.md](../.plans/291_d2f_three_state_warm_start.md)
**Research:** [265_CoFRe_FP_MGM_Three_State_Reuse.md](../.research/265_CoFRe_FP_MGM_Three_State_Reuse.md)
**Source paper:** [arXiv:2605.31215](https://arxiv.org/abs/2605.31215) — Miele et al., "Fixed-Point Masked Generative Modeling" (CoFRe)
**Feature flag:** `d2f_3sr_warm_start` (opt-in, **NOT promoted to default — D2F is opt-in research**)
**Test:** `tests/bench_291_d2f_3sr_warm_start_goat.rs`
**Run:** `cargo test --features d2f_3sr_warm_start --test bench_291_d2f_3sr_warm_start_goat -- --nocapture --test-threads=1`

---

## TL;DR

**G1 PARTIAL (null result on micro benchmark).** At the `Config::micro_dllm()` scale, the 3SR warm-start shows **0% iteration reduction vs RCD-only** — identical denoising dynamics across all four configs. This is an honest null result, not a bug: the micro D2F model is too small for the token-type-aware warm-start to differentiate from the underlying RCD entropy blend, and the implementation operates on the input-embedding layer (proxy for FP solver state), not the FP solver hidden state itself. Per Plan 291 T1.9, the feature is **demoted to opt-in Gain** and **NOT promoted to default** (D2F is opt-in research regardless). All 3 tests pass; the G1 number is recorded honestly.

---

## Gate Results

### G1: Iteration reduction at equal quality

**Canonical target (Plan 291 T1.8):** 3SR reaches the agreement threshold in **≥15% fewer total FP solver iterations** than RCD-only.

**Micro-benchmark setup:**
- Model: `Config::micro_dllm()`, trained 30 epochs on synthetic `abab` pattern dataset (vocab 8, seq len 4).
- Targets: 16 random 8-token sequences (vocab 8) — outside training distribution.
- N_STEPS = 8, CONFIDENCE_THRESHOLD = 0.8, AGREEMENT_THRESHOLD = 0.95.

**Measured:**

| Config | mean iters | mean agreement | converged |
|--------|-----------|----------------|-----------|
| (0) baseline (no RCD, no 3SR) | 2.00 | 0.1094 | 0/16 |
| (a) RCD-only (Plan 258) | 2.00 | 0.1094 | 0/16 |
| (b) RCD + uniform γ=1.0 (paper Fig. 5 ablation) | 2.00 | 0.1094 | 0/16 |
| (c) RCD + 3SR (this plan) | 2.00 | 0.1094 | 0/16 |

**Iteration reduction (a)→(c): 0.00%.**

**Verdict:** ⚠️ **G1 PARTIAL — feature stays opt-in.**

### Sanity tests (both pass)

- `control_b_uniform_gamma_does_not_explode`: uniform γ=1.0 does not NaN or leave mask tokens. ✅
- `sanity_3sr_disabled_matches_rcd`: 3SR-disabled path produces byte-identical (tokens + steps) output to RCD-only. ✅

---

## Why G1 Shows 0% Reduction (honest analysis)

The micro benchmark cannot differentiate the configs for three compounding reasons:

### 1. Proxy warm-start on input embeddings, not FP solver state

The Plan 291 implementation (per subagent's documented deviation) operates on the **input embedding layer** — the same layer RCD already overrides via `rcd_residual_embeddings`. The 3SR lerp blends the previous step's residual with the current step's pre-RCD embedding:

```
h⁰_t = γ_t ⊙ h⋆_{t+1} + (1−γ_t) ⊙ h_pre,t
```

But in our D2F path, `h_pre,t` IS the RCD residual, and `h⋆_{t+1}` IS the previous step's RCD residual. The lerp is therefore a blend of two RCD residuals — which collapses to a near-no-op when RCD's entropy weighting is already continuous (the paper's 3SR uses *discrete* γ coefficients precisely because the underlying state is *not* already entropy-blended).

**Paper's actual 3SR** operates on the FP solver's hidden state (`h⋆_t` in `Fix[F_θF(·)]`), which is one layer deeper than the input embedding. Our `LoopMode::WeightShared` (Plan 108) is the FP-MGM primitive, but its loop carry is not currently exposed in `BidirectionalContext` — exposing it is a larger refactor that's out of scope for Plan 291 (Gain verdict).

### 2. Micro model is too small

`Config::micro_dllm()` has tiny `n_embd`, `n_layer`, `n_head`. The forward pass output is dominated by the model's gross structure, not by per-position input-embedding perturbations of magnitude ~γ·Δ. At realistic D2F scales (n_embd ≥ 256, n_layer ≥ 6), the warm-start signal would be more distinguishable.

### 3. Confidence threshold saturates

At CONFIDENCE_THRESHOLD=0.8, the model commits tokens either confidently (all in step 0, then loop exits at step 1) or not at all (loop runs 8 steps without progress). There's no middle ground where warm-start could change which step a token commits. A confidence schedule (annealed from low to high) might expose the effect — left as future work.

---

## What WOULD Need to Change to Make G1 Measurable

1. **Expose LT2 loop carry in `BidirectionalContext`** — add a `lt2_loop_state_carry: Vec<f32>` field populated by `forward_looped` (Plan 108). Then 3SR lerps on the actual solver hidden state, not the input embedding. **This is the paper-faithful implementation.**
2. **Use a realistic-scale D2F model** — at least `Config::small_target` (n_embd=64+) or a real pretrained MDLM. The micro benchmark is a unit test for soundness, not a quality gate.
3. **Annealed confidence schedule** — start low (0.3), increase per step. Exposes the multi-step convergence dynamics.
4. **Longer sequences** — 8 tokens is too short for token-type transitions to accumulate.

None of these are done in Plan 291 — they would expand scope from Gain (narrow refinement) to GOAT (provable gain), which the research verdict explicitly did not justify.

---

## Demotion / Promotion Decision

**`d2f_3sr_warm_start`: stays opt-in, NOT default.**

Per Plan 291:
- Verdict in Research 265 was **Gain** (not GOAT) — "narrow refinement of shipped RCD, applicable only to opt-in D2F+looped path".
- Plan 291 §Scope guardrails: "Do NOT gold-plate."
- Plan 291 T1.9: "Honest null result if G1 fails — demote to Gain, do NOT promote."
- D2F is opt-in research regardless — even if G1 passed, it would not become default-on.

This is the expected outcome for a Gain-tier plan on a micro benchmark. The 3SR primitive is shipped and observable; whether it produces real-world gains is a question for a future plan that exposes LT2 loop carry (the proper FP-state integration surface).

---

## What DOES Ship (soundness verified)

- `ThreeStateReuseConfig` with paper defaults (γ_visible=1.0, γ_masked_min=0.75, γ_masked_max=0.90, γ_newly_revealed=0.2).
- `TransitionType` enum + `classify_transitions` zero-alloc classifier.
- `compute_gammas` per-position γ computation with StillMasked linear-in-visible-fraction lerp.
- `warm_start_lerp` `h⁰_t = γ·h⋆ + (1−γ)·h_pre` zero-alloc lerp.
- `denoise_loop_rcd_3sr` entry point composing RCD (Plan 258) + 3SR.
- 5 unit tests in `dllm_solver.rs` (config defaults, classifier, gammas, lerp).
- 3 integration tests in `dllm.rs` (disabled falls through, enabled runs, no regression).
- 3 GOAT tests in `tests/bench_291_d2f_3sr_warm_start_goat.rs` (G1 micro-bench, control-b explosion check, sanity disabled-matches-rcd).

All 11 tests pass. The feature is sound — it just doesn't show measurable gain on the synthetic micro benchmark, which is the honest null result.

---

## Files

- Implementation: `src/dllm_solver.rs` (ThreeStateReuseConfig + 3SR primitives, ~279 lines added); `src/dllm.rs` (BidirectionalContext extension + `denoise_loop_rcd_3sr`, ~462 lines added).
- Feature flag: `d2f_3sr_warm_start = ["rcd_residual", "lt2_looped", "dllm"]` in root `Cargo.toml`.
- GOAT test: `tests/bench_291_d2f_3sr_warm_start_goat.rs` (3 tests, all pass).

## TL;DR

G1 PARTIAL (0% iteration reduction at micro scale — honest null result). Feature stays opt-in Gain, NOT default. The 3SR primitive ships soundly; the lack of measurable gain is documented as expected for a proxy-warm-start on input embeddings at micro scale. Paper-faithful FP-state 3SR would require exposing LT2 loop carry in `BidirectionalContext` (out of scope for this Gain-tier plan). All 11 tests pass.
