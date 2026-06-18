# Research 265: CoFRe / FP-MGM — Fixed-Point Masked Generative Modeling

> **Source:** "Fixed-Point Masked Generative Modeling" — Miele, Qin, Carballo-Castro, Deschenaux, Frossard (EPFL). [arXiv:2605.31215](https://arxiv.org/abs/2605.31215). May 2026.
> **Reference impl:** https://github.com/andreamiele/fp-mgm (`fp-mdlm`, `fp-maskgit` branches)
> **Date:** 2026-06-18
> **Status:** Done
> **Related Research:** 035 (Attractor / DEQ comparison), 073 (LT2 Looped), 097 (Training-Free Looped), 148 (Hydra adaptive depth), 228 (TwinProp dendritic adaptive compute)
> **Related Plans:** 066 (D2F), 108 (LT2 Looped — ships `LoopMode::WeightShared`), 109 (DMax SPD), 136 (TF-Loop), 165 (Hydra adaptive budget), 258 (RCD — closest cousin to 3SR), 284 (Adaptive depth tier), **291 (this paper's plan)**
> **Classification:** Public

---

## TL;DR

CoFRe introduces **Fixed-Point Masked Generative Models (FP-MGMs)**: replace the middle stack of a masked-diffusion denoiser (MDLM/MaskGIT) with a weight-sharing fixed-point block iterated N times, giving adaptive effective depth without extra parameters. The paper's three inference-time contributions are (a) the FP weight-shared block itself, (b) **three-state reuse (3SR)** — a token-aware warm-start of the FP solver across denoising steps with per-transition-type coefficients (unchanged-visible γ=1.0, still-masked γ∈[0.75,0.9], newly-revealed γ=0.2), and (c) decreasing budget schedules. The training-side contributions (SJFB, L_CONS cross-step consistency loss, pretrained→FP KL distillation) **→ riir-train** (out of scope here).

**Distilled for katgpt-rs (modelless, inference-time):** ~80% of the paper's inference surface is **already shipped**. `LoopMode::WeightShared { loop_count }` (Plan 108, GOAT 8/8) *is* the FP-MGM weight-sharing primitive; `LoopMode::TrainingFree` (Plan 136, GOAT 4/4) is a strictly more sophisticated ODE-motivated variant; Plan 165 + Plan 284 ship adaptive depth gating; Plan 066 D2F + Plan 109 DMax SPD ship the masked-diffusion denoiser; **Plan 258 RCD** in `dllm.rs::denoise_loop_rcd` already ships the closest cousin to 3SR — entropy-weighted residual embedding carry for "still masked" positions across D2F steps (`ẽ_i = (1−α_i)·E_mask + α_i·Δ_i`, `α_i = H(p_i)/log V`). The only delta is a small refinement: apply the token-type-aware carry to the **FP solver hidden state** with three discrete coefficients instead of to the input embedding with a continuous entropy blend.

**Verdict: Gain** (leaning Pass). The 3SR rule is a genuinely transferable primitive but it is a narrow variation on shipped prior art, applicable only to the opt-in D2F + looped path. Plan only, behind feature flag, NOT default.

---

## 1. Paper Core Findings

### 1.1 FP-MGM architecture
Decompose a masked-diffusion denoiser into four parts:
```
h_pre  = P_θP(z_t, t)                       # explicit preprocessing stack
h̃_t   = G_θG(h_pre)                          # input-conditioning projection
h⋆_t  = Fix[F_θF(·; h̃_t, t)]                 # IMPLICIT fixed-point block (iterated N times)
ℓ_θ   = H_θH(h⋆_t, t)                        # explicit postprocessing stack → logits
```
The fixed-point block `F_θF` is a single shared transformer block iterated `N` times; `N` controls effective depth at inference without adding parameters. Per-step cost = `K_pre + N·K_fp + K_post` forward passes; parameter count = `K_pre + K_fp + K_post` only. Applied to MDLM → FP-MDLM (text), MaskGIT → FP-MaskGIT (images).

### 1.2 Three-state reuse (3SR) — the inference-time contribution
Empirical observation (paper Fig. 3): token-wise movement of the solved FP state `‖h⋆_{t+1}(i) − h⋆_t(i)‖₂` differs sharply by transition type — newly-revealed tokens move most, unchanged-visible move least, still-masked decay as context stabilises. 3SR initialises the solver:
```
h⁰_t = γ_t ⊙ h⋆_{t+1} + (1 − γ_t) ⊙ h_pre,t
γ_it = 1.0                  if position i is unchanged visible
      = γ_mask(v_t) ∈[0.75,0.90]   if still masked   (linear in visible fraction v_t)
      = 0.2                  if newly revealed
```
Grid-searched coefficients (paper Tables 4–5); robust within a moderate range.

### 1.3 Cross-step consistency loss (L_CONS) — TRAINING → riir-train
MSE on hidden states between noisier-student and cleaner-teacher from correlated nested masks: `L = L_MDLM + λ·‖h_s − sg(h_c)‖²₂`. Drives most of the low-budget gain (paper Table 9: budget-96 gen-PPL 375.6 → 104.2). Behaves like cross-time self-distillation. **Training-side — redirect.**

### 1.4 Budget allocation
Decreasing FP-iteration schedule (more solver steps early when sequence is most corrupted) beats fixed/increasing/cosine/front-loaded (paper Table 22). Not interchangeable with denoising-step count (paper Fig. 13 heatmap).

### 1.5 Pretrained → FP conversion — TRAINING → riir-train
Map pretrained MDLM layers 1→pre, 6→FP block, 12→post (motivated by CKA analysis showing layers 6–12 are near-identical, within-block CKA 0.998). Short 40k-step KL distillation adapts. **Training-side — redirect.**

### 1.6 Headline numbers (vs MDLM/MaskGIT baselines)
- FP-MDLM: −38.8% params (104M vs 170M), −11.5% train time, −16.9% VRAM. Budget-96 gen-PPL 830.8 → 375.6 (FP alone) → 101.8 (CoFRe = FP+L_CONS+3SR).
- FP-MaskGIT: −48.6% train time, −50.7% VRAM, FID improved at all budgets.
- Latency: CoFRe is 1.12–1.45× slower than MDLM+SDTT at equal budget but reaches much lower gen-PPL — a quality/latency win, not a raw speed win.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| FP weight-shared iterated block | `LoopMode::WeightShared { loop_count }` — default-on, GOAT 8/8 | Plan 108, `crates/katgpt-core/src/types.rs:314`, `forward_looped` |
| ODE-motivated iterated block (more general than FP) | `LoopMode::TrainingFree` + `TrainingFreeLoopConfig` (K-stage RK β=0.5, window, cache strategy) — default-on, GOAT 4/4 | Plan 136, `tf_loop` |
| Adaptive depth (stop when converged) | Hydra cumulative-DE convergence gate; runtime depth-tier cap | Plans 165, 284; `InferenceOverrides.depth_tier` |
| Masked diffusion denoiser (MDLM-like) | D2F block-parallel denoising, bidirectional positions, mask token | Plan 066 (`dllm` feature), `src/dllm.rs`, `src/speculative/d2f.rs` |
| Masked diffusion (MaskGIT-like) | DMax Soft Parallel Decode — hybrid token/mask embeddings | Plan 109 (`dmax_spd`), default-on |
| **3SR closest cousin** — token-state-aware carry across D2F steps | **RCD Residual Context Diffusion** — entropy-weighted residual embedding for "still masked" positions | **Plan 258** (`rcd_residual`), `dllm.rs::denoise_loop_rcd:2184` |
| Cross-step warm-start / carry | HRM-Text learned init for recurrent carry; TF-Loop sub-stepping; LT2 AHLA state carry across calls | Plans 082b, 136, 108 |
| Fixed-point residual scoring (different sense — convergence) | `deep_manifold` L2/KL residual; Newton-Schulz cubic fixed-point | Plans 085, 152 |
| DEQ comparison / attractor fixed-point basins | Attractor Models — iterations decrease as backbone learns to warm-start | Research 035 |

### 2.2 The 3SR delta vs RCD (Plan 258)

RCD and 3SR solve the same problem — *carry information about not-yet-committed positions across denoising steps* — but at different layers and with different blending rules:

| Aspect | RCD (Plan 258, shipped) | 3SR (CoFRe, this paper) |
|---|---|---|
| Operates on | Input embedding `ẽ_i` | FP solver hidden state `h⁰_t` |
| Token-type awareness | Binary: "still masked" gets residual blend, others get standard embedding | Three-way: unchanged-visible / still-masked / newly-revealed |
| Blend coefficient | Continuous: `α_i = H(p_i)/log V` (entropy-normalised) | Discrete schedule: γ∈{1.0, [0.75,0.9], 0.2} (grid-searched, linear in visible fraction for masked) |
| Cost | One entropy-weighted residual per masked position per step | One lerp per position per FP solver initialisation |
| Hot path? | Opt-in (`rcd_residual` feature) on D2F path | Would be opt-in on D2F + looped path |

**Honest read:** RCD already ships the conceptual primitive (token-state-aware carry across D2F steps). 3SR is a refinement that (a) operates one layer deeper (solver state vs input embedding) and (b) uses a discrete three-state rule. Both choices are reasonable; neither is a new capability class. The paper's empirical justification (Fig. 3 — token movement differs by transition type) is the genuinely useful insight and is consistent with RCD's entropy-weighted design (high-entropy = uncertain = "still masked-ish" = less reuse).

### 2.3 Fusion

**3SR × RCD (Plan 258) × LT2-Looped (Plan 108):** when `LoopMode::WeightShared` runs inside a D2F denoising loop, currently the AHLA/loop state carry is uniform across positions. The fusion: make that carry **token-type-aware** using the 3SR three-coefficient rule — unchanged-visible tokens inherit previous step's solved loop state fully (γ=1.0), still-masked tokens partially (γ∈[0.75,0.9]), newly-revealed tokens weakly (γ=0.2). This composes the paper's 3SR with our shipped LT2 loop and our shipped D2F mask tracking.

This fusion is **novel** (no note/code combines these three), but **narrow**: it only applies when LT2 looping is enabled inside D2F denoising — an opt-in research path, not a hot path. Expected gain: fewer LT2 iterations needed per D2F step for the same quality (paper's residual analysis, Fig. 16: full-reuse starts ~1984× closer to equilibrium). Not a new capability class — same outputs, less compute, on a non-hot path.

### 2.4 → riir-train (out of scope, noted for completeness)
- L_CONS cross-step consistency loss (the paper's largest quality driver) — training-side, needs backprop through hidden states.
- SJFB (Stochastic Jacobian-Free Backpropagation) for implicit-layer training.
- Pretrained MDLM → FP-MDLM conversion via KL distillation (40k-step teacher-student adaptation, CKA-guided layer mapping).
- Over-sharpening / entropy-collapse early-stopping rule for L_CONS post-training (validation-PPL first-crossing of 1.15× pre-L_CONS value).

These are real training contributions and belong in `riir-train/.research/` if/when distillation-of-iterated-solvers becomes a priority. Not filed from this session per workflow rules.

---

## 3. Verdict

**Gain** (leaning Pass — heavy prior art).

One-line reasoning: the paper's inference-time surface is ~80% already shipped (`LoopMode::WeightShared` = FP-MGM; Plan 258 RCD = closest cousin to 3SR; Plans 066/109 = masked diffusion; Plans 165/284 = adaptive depth); the only delta is the 3SR token-type-aware warm-start rule, which is a narrow refinement of RCD applicable only to the opt-in D2F+looped path. Training-side (L_CONS, SJFB, distillation) → riir-train.

**Why not Super-GOAT:** fails novelty gate Q1 (mechanism already shipped as `LoopMode::WeightShared` + RCD), Q2 (no new capability class — same outputs, less compute, on a non-hot path), Q3 (no product selling point — D2F is opt-in research, not in any arena/game hot path), Q4 (force multiplier is weak — only multiplies D2F, which itself is opt-in).

**Why not GOAT:** no provable latency/quality gain over the existing shipped approach (RCD) on a hot path. The paper's gains are measured against MDLM/MaskGIT baselines that we don't run in production; against RCD the delta is small and on an opt-in path.

**Why not Pass:** the 3SR rule IS a genuinely transferable inference-time primitive (token-state-aware warm-start for iterated solvers with discrete per-transition-type coefficients), and the paper's empirical justification (Fig. 3 token-movement-by-transition-type) is a useful design insight worth recording. A small plan to add it as an opt-in refinement on the D2F+looped path is justified — but it must NOT be default and must clear a GOAT gate before promotion.

| Tier | Criteria | Routing |
|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + force multiplier | — |
| GOAT | Provable gain over existing approach, new default candidate | — |
| **Gain** ← | Incremental, useful, not headline-worthy | **Plan 291 only, behind feature flag, NOT default** |
| Pass | Not relevant / training-only | — |

---

## 4. Action

- **Plan 291** (`katgpt-rs/.plans/291_d2f_three_state_warm_start.md`) — add 3SR token-type-aware warm-start as an opt-in refinement to the LT2-looped-inside-D2F path. Feature flag `d2f_3sr_warm_start`. GOAT gate: equal-quality fewer-FP-iterations vs RCD-only baseline on a micro-D2F benchmark. Default-off; promote only if it wins on a real (non-toy) D2F workload.
- **riir-train note (NOT filed this session):** L_CONS, SJFB, pretrained→FP distillation belong in `riir-train/.research/` if iterated-solver distillation becomes a priority.

## TL;DR

CoFRe's FP-MGM = our shipped `LoopMode::WeightShared`; CoFRe's 3SR = a refinement of our shipped Plan 258 RCD (token-type-aware warm-start, three discrete coefficients, applied to solver hidden state); CoFRe's training-side (L_CONS, SJFB, distillation) → riir-train. Verdict **Gain** — Plan 291 only, opt-in feature flag, NOT default, must clear GOAT gate. No Super-GOAT guide (fails novelty gate Q1–Q4).
