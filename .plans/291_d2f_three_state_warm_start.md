# Plan 291: D2F Three-State Warm-Start (3SR × LT2-Looped × RCD Fusion)

**Date:** 2026-06-18
**Research:** [katgpt-rs/.research/265_CoFRe_FP_MGM_Three_State_Reuse.md](../.research/265_CoFRe_FP_MGM_Three_State_Reuse.md)
**Source paper:** [arXiv:2605.31215](https://arxiv.org/abs/2605.31215) — Miele et al., "Fixed-Point Masked Generative Modeling" (CoFRe)
**Target:** `katgpt-rs/src/dllm.rs` (extend `denoise_loop_rcd` family) + Cargo feature `d2f_3sr_warm_start` (depends on `rcd_residual` + `lt2_looped` + `dllm`)
**Status:** Active — Phase 1 not started

---

## Goal

Add the CoFRe **three-state reuse (3SR)** rule as an opt-in refinement to the LT2-looped-inside-D2F denoising path. When `LoopMode::WeightShared` runs inside a D2F denoising loop, the loop-state carry across denoising steps becomes **token-type-aware**: unchanged-visible positions inherit the previous step's solved loop state fully (γ=1.0), still-masked positions partially (γ∈[0.75,0.9], linear in visible fraction), newly-revealed positions weakly (γ=0.2). This composes the paper's 3SR with our shipped LT2 loop (Plan 108) and our shipped D2F mask tracking (Plan 066).

**GOAT gate:** at equal sampling quality (gen-PPL proxy or token-agreement rate on a micro-D2F benchmark), 3SR-warm-start must achieve the same quality in **≥15% fewer total FP solver iterations** than the RCD-only baseline (Plan 258). If it wins, it may stay as an opt-in refinement (NOT default — D2F is not a hot path). If it loses or ties, demote to Gain and document the null result.

**Scope guardrails:**
- D2F is opt-in (`dllm` feature) and NOT in any arena/game hot path. This plan does not touch AR inference.
- This is a **Gain** verdict per Research 265 — narrow refinement of shipped RCD (Plan 258). Do NOT gold-plate.
- Training-side CoFRe contributions (L_CONS, SJFB, distillation) are out of scope → riir-train.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Add feature flag `d2f_3sr_warm_start` to `Cargo.toml`, depending on `rcd_residual`, `lt2_looped`, `dllm`. Gated module: `src/dllm_3sr.rs` (or extend `dllm.rs` behind `#[cfg(feature = "d2f_3sr_warm_start")]`).
- [ ] **T1.2** Define `ThreeStateReuseConfig` struct:
  ```rust
  pub struct ThreeStateReuseConfig {
      pub gamma_visible: f32,       // default 1.0
      pub gamma_masked_min: f32,    // default 0.75
      pub gamma_masked_max: f32,    // default 0.90
      pub gamma_newly_revealed: f32,// default 0.2
      pub enabled: bool,
  }
  ```
  Impl `Default` matching paper defaults. Zero-cost when `enabled=false`.
- [ ] **T1.3** Implement transition-type classifier: given `z_prev` and `z_t` token slices + `mask_token`, classify each position into `{UnchangedVisible, StillMasked, NewlyRevealed}`. Reuse the existing "still masked" tracking already in `d2f.rs` (`masked: Vec<bool>`) — do not duplicate.
- [ ] **T1.4** Implement per-position γ computation:
  - `UnchangedVisible` → `gamma_visible`
  - `StillMasked` → `gamma_masked_min + (gamma_masked_max − gamma_masked_min) * v_t` where `v_t = visible_fraction`
  - `NewlyRevealed` → `gamma_newly_revealed`
- [ ] **T1.5** Implement the warm-start lerp `h⁰_t = γ_t ⊙ h⋆_{t+1} + (1−γ_t) ⊙ h_pre,t` operating on the LT2 loop carry buffer. Reuse the existing AHLA state carry buffer from `forward_looped` — extend it to be per-position-γ-weighted when the feature is on.
- [ ] **T1.6** Wire `denoise_loop_rcd_3sr` entry point that combines RCD (Plan 258, input-embedding carry) with 3SR (solver-state carry). Both carry mechanisms are orthogonal and compose at different layers; neither subsumes the other.

### Phase 1 GOAT gate (G1 — does it converge faster?)

- [ ] **T1.7** Micro-benchmark: micro_dllm config, 8 denoising steps, measure **total FP solver iterations to reach a fixed token-agreement threshold** (vs ground-truth tokens) for three configs:
  - (a) RCD-only baseline (Plan 258 as shipped)
  - (b) RCD + uniform-γ warm-start (γ=1.0 everywhere — the "full reuse" ablation from paper Fig. 5)
  - (c) RCD + 3SR warm-start (this plan)
- [ ] **T1.8** **G1 PASS condition:** (c) reaches the agreement threshold in ≥15% fewer total FP iterations than (a). (b) is the degenerate full-reuse control — paper shows it can be unstable at high budgets, so (b) underperforming (a) on some configs is expected and not a failure of (c).
- [ ] **T1.9** Record results in `.benchmarks/NNN_d2f_3sr_warm_start_goat.md`. Honest null result if G1 fails — demote to Gain, do NOT promote.

## Phase 2 — Robustness Sweep (only if G1 passes)

### Tasks

- [ ] **T2.1** γ-coefficient grid search on the micro benchmark: sweep `gamma_masked_min ∈ {0.6, 0.75, 0.85}`, `gamma_masked_max ∈ {0.9, 0.95}`, `gamma_newly_revealed ∈ {0.0, 0.1, 0.2, 0.3}`. Confirm paper's finding that performance is robust within a moderate range (paper Tables 4–5).
- [ ] **T2.2** Budget-allocation sanity check: confirm decreasing FP-iteration schedule (more solver steps early) is at least as good as fixed allocation on the micro benchmark. Our `TrainingFreeLoopConfig` already supports window/count tuning — reuse, do not reinvent.
- [ ] **T2.3** **G2 PASS condition:** chosen γ-config is within 5% of the grid-best, and the grid-best itself passes G1. Document the chosen config in `ThreeStateReuseConfig::default()`.

## Phase 3 — Wire + Docs (only if G2 passes)

### Tasks

- [ ] **T3.1** Expose `ThreeStateReuseConfig` via `InferenceOverrides` (optional override, like the existing `depth_tier`).
- [ ] **T3.2** Add `examples/d2f_3sr_demo.rs` showing before/after iteration count at fixed agreement threshold.
- [ ] **T3.3** Update `.docs/01_overview.md` feature-flag table — mark `d2f_3sr_warm_start` as **opt-in**, reference Research 265 + this plan. Do NOT add to default feature list.
- [ ] **T3.4** Cross-link Research 265 §2.3 (fusion) and Plan 258 (RCD) from the module doc comment.

## Out of Scope (→ riir-train)

- L_CONS cross-step consistency loss (training-side, needs backprop through hidden states).
- SJFB (Stochastic Jacobian-Free Backpropagation) for implicit-layer training.
- Pretrained MDLM → FP-MDLM distillation (CKA-guided layer mapping, 40k-step KL adaptation).
- Over-sharpening early-stop rule for L_CONS post-training.

These are real training contributions; route to `riir-train/.research/` if iterated-solver distillation becomes a priority. Not filed from this workflow per the 3-repo discipline.

## References

- Research: `.research/265_CoFRe_FP_MGM_Three_State_Reuse.md`
- Closest shipped cousin: Plan 258 (RCD) — `.plans/258_rcd_residual_context_diffusion.md`, `src/dllm.rs::denoise_loop_rcd`
- FP-MGM primitive (shipped): Plan 108 (LT2 Looped) — `LoopMode::WeightShared`, `forward_looped`
- Masked diffusion (shipped): Plan 066 (D2F), Plan 109 (DMax SPD)
- Adaptive depth (shipped): Plans 165, 284
- Source paper: [arXiv:2605.31215](https://arxiv.org/abs/2605.31215), code: https://github.com/andreamiele/fp-mgm (`fp-mdlm`, `fp-maskgit` branches)

## TL;DR

Add CoFRe's 3SR token-type-aware warm-start as an opt-in refinement (`d2f_3sr_warm_start`) to the LT2-looped-inside-D2F path. Composes 3SR × RCD (Plan 258) × LT2 (Plan 108). GOAT gate G1: ≥15% fewer FP iterations at equal quality vs RCD-only. NOT default — D2F is opt-in research, not a hot path. Training-side CoFRe (L_CONS, SJFB, distillation) → riir-train, out of scope.
