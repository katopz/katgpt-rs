# Research Verdict 41: RePlaid — Continuous Diffusion Scales Competitively with Discrete Diffusion for Language

**Paper:** Continuous Diffusion Scales Competitively with Discrete Diffusion for Language (arXiv:2605.18530)
**Authors:** Zhihan Yang, Wei Guo, Shuibai Zhang, Subham Sekhar Sahoo, Yongxin Chen, Arash Vahdat, Morteza Mardani, John Thickstun (NVIDIA, Cornell, Georgia Tech, UW-Madison, MBZUAI-IFM)
**Date:** May 19, 2026
**Status:** 🟡 SELECTIVE ADOPTION — Theoretical framework adopted (variance-minimized schedules, ELBO regularization). Full continuous diffusion mechanism rejected (incompatible with DDTree speculative decoding, same verdict as Research 10 ColaDLM).

---

## 1. Core Premise

RePlaid revisits **Plaid** (Gulrajani & Hashimoto, NeurIPS 2023) — a likelihood-based continuous diffusion language model — and modernizes it to match the architecture of discrete DLMs (MDLM, Duo). Under this unified setting:

1. **Continuous diffusion scales on par with discrete diffusion** — 20× compute gap vs AR (vs MDLM's 14×, Duo's 22×)
2. **PPL 22.1 on OpenWebText** — SOTA among continuous DLMs, beating MDLM low-var (23.1), Duo (25.2), LangFlow (32.2)
3. **1.8× fewer parameters** than MDLM/Duo at compute-optimal frontier
4. **Over-training advantage** — beats MDLM when trained past compute-optimal (critical for small inference models)

The key enablers are: (a) ELBO training with learnable noise schedule, (b) learnable low-dim embeddings (d_e=16), (c) self-conditioning, (d) output prior logits.

---

## 2. Key Findings

### 2.1 Scaling Laws (Unified Protocol)

| Method | Compute Gap vs AR | PPL (OWT 1M) | Parameters vs MDLM |
|--------|-------------------|---------------|---------------------|
| AR | 1× | 17.5 | — |
| MDLM (low var) | 14× | 23.1 | baseline |
| Duo | 22× | 25.2 | ≈1× MDLM |
| **RePlaid (s.c.)** | **20×** | **22.1** | **1.8× fewer** |
| RePlaid (no s.c.) | 27× | 23.6 | 1.8× fewer |

First unified scaling comparison between continuous and discrete DLMs using Sahoo et al. protocol (SlimPajama, Llama-2 tokenizer, identical DiT architecture).

### 2.2 Ablation Impact (Table 2)

| Component Removed | PPL | Δ |
|---|---|---|
| Full RePlaid (s.c.) | 22.1 | — |
| w/o output prior | 22.5 | +0.4 |
| w/o self-conditioning | 23.6 | +1.5 |
| w/o learnable noise schedule | 24.4 | +2.3 |
| **w/o learnable embeddings** | **39.4** | **+17.3** |

**Learnable embeddings are the dominant factor** — without them, RePlaid becomes the worst DLM. This is the single most important finding for our stack.

### 2.3 Theoretical Results

**Proposition 1 (Constant Per-Timestep Diffusion Loss):** There exists a unique noise schedule γ* such that the per-timestep diffusion loss is constant across time. Training the noise schedule to minimize Monte-Carlo variance converges to this schedule.

**Lemma 3 (Linear Information Decay):** Under Bayes-optimal denoiser with optimal noise schedule, mutual information I(e; z_t) decays linearly in t: I(e; z_t) = I(e; z_0) - κt.

**Proposition 2 (Near-Linear CE):** Per-timestep cross-entropy decomposes into a linear-in-t trend plus a non-negative residual (conditional total correlation that increases in t). Empirically, CE is near-linear when the noise schedule is learned.

### 2.4 Embedding Geometry Discovery

ELBO-trained embeddings at d_e=16 show:
- **POS-structured clustering** (Fig 5a) — discovered without any POS labels
- **Low-rank structure** — 90% variance in 6 principal components (Fig 5b)
- Adding cross-entropy loss **destroys** this structure (Fig 5c) — PCA flattens, PPL degrades from 22.1→26.1

### 2.5 Sampling Results

- **DPM-Solver++(2M)** beats DDPM at low NFEs (< 64 steps)
- **DDPM** beats deterministic solvers at high NFEs (≥ 64 steps)
- RePlaid (no s.c.) outperforms Duo at every T on MAUVE
- Self-conditioning compounds confidence → lower entropy at high T

### 2.6 The ELBO vs Cross-Entropy Incompatibility

This is the most architecturally significant finding for our distillation stack:

**Adding CE to ELBO makes things WORSE** (Tab 2 + Sec 5.1):
- RePlaid with pure ELBO: PPL 22.1
- RePlaid with ELBO + auxiliary CE: PPL 26.1 (+4.0 degradation)
- Root cause: CE gradients are **separative** — they push embedding vectors apart, destroying the low-rank POS-clustered structure ELBO creates
- PCA scree flattens from 6 PCs (90% variance) to 13 PCs when CE is added (Fig 5b vs 5c)

**Implication for our pipeline:**
- Our ROPD rubric distillation (Plan 071/072) uses pointwise scoring — functionally CE-like
- Our SDAR sigmoid gating (Plan 072/073) uses teacher-student gap with learned β — closer to ELBO-like variance minimization
- **SDAR's gating mechanism is more compatible with embedding structure preservation than ROPD's pointwise scoring**
- If combining both, **gate the ROPD signal through SDAR** (don't add them independently)

### 2.7 Over-Training Advantage Validates Modelless Approach

RePlaid beats MDLM when trained past compute-optimal (Fig 1c):
- A 66M RePlaid trained for 3.1× optimal compute matches MDLM trained for 6.9× optimal
- The crossover occurs because ELBO regularization prevents overfitting in small models

**Why this matters for modelless:** Our modelless pruners (GFlowNet, ROPD, SDAR) are essentially "over-training" the pruner on observation data without a teacher model. RePlaid's result provides theoretical support:
- Small, over-trained pruners can beat larger, compute-optimal ones
- Self-supervised regularization (variance minimization) is the key enabler
- This validates our modelless-first approach (Plan 049 Phase 1 T1–T5) where we train pruners on self-play data without a model

---

## 3. Relationship to Our Previous Research

### 3.1 vs Research 10 (ColaDLM) — This Paper is Stronger

| Aspect | ColaDLM (Research 10) | RePlaid (This Paper) |
|--------|----------------------|---------------------|
| Architecture | VAE + DiT (2B params) | Modernized Plaid (0.1B) |
| Latent space | d=16-128 VAE latents | d_e=16 learnable embeddings |
| Training | Flow matching (not likelihood) | ELBO (true likelihood bound) |
| PPL | Not reported (generative metrics only) | 22.1 (SOTA continuous) |
| Our verdict | Mechanism rejected | Theoretical framework adopted |
| Actionable insight | Block-causal generation | Variance-minimized schedules |

**Key difference:** ColaDLM required a VAE+DiT rewrite that was fundamentally incompatible with our stack. RePlaid's insights decompose into modular, adoptable pieces (schedules, self-conditioning, embedding geometry) that work within our existing architecture.

### 3.2 vs Research 34 (D2F Discrete Diffusion Forcing)

| Aspect | D2F (Research 34) | RePlaid |
|--------|-------------------|---------|
| Diffusion type | Discrete (mask tokens) | Continuous (Gaussian noise on embeddings) |
| Noise schedule | Fixed monotonic (min→max ratio) | Learned (variance-minimized) |
| Solver | Iterative DDPM-style (step-by-step unmask) | DDPM + DDIM + DPM-Solver++(2M) + Heun |
| Our implementation | ✅ Plan 066 complete | ❌ Not implementing full pipeline |
| What to adapt | **Variance-minimized schedule + higher-order solvers for D2F** | — |

**New finding (Sec 4.2, Appendix D):** RePlaid shows DPM-Solver++(2M) — a second-order multistep ODE solver — significantly outperforms first-order DDIM/DDPM at low NFEs. This is directly transferable to our D2F pipeline: replacing iterative step-by-step unmasking with a higher-order denoising schedule could reduce steps from 16→4 while maintaining quality. The solver caches the previous prediction and linearly extrapolates (Eq 16-17), requiring only 1 NFE per step.

### 3.3 vs Research 38 (SDAR Sigmoid Gating)

| Aspect | SDAR (Research 38) | RePlaid |
|--------|-------------------|---------|
| Gating mechanism | Sigmoid gate σ(β·Δt), β=5.0 fixed | Learned noise schedule γ̃(t) |
| Self-supervised signal | Teacher-student gap | ELBO variance |
| Our modelless result | ELO 954 Bomber, draws 100% FFT | — |
| What to adapt | **Learnable β via variance minimization** | — |

---

## 4. What We Distill (Adopted)

### 4.1 Variance-Minimized Noise Schedules — ADOPT

**Insight:** Training the noise schedule to minimize Monte-Carlo variance of the per-timestep loss yields a constant-difficulty schedule (Prop 1). This is self-supervised — no teacher needed.

**Adaptation:**
- **D2F noise schedule** (`src/dllm.rs` `NoiseSchedule`): Replace fixed monotonic ratios with variance-minimized ratios that equalize per-step reconstruction difficulty.
- **Bandit exploration schedule** (`src/pruners/bandit.rs`): Make exploration rate adaptive based on per-episode reward variance tracking.
- **SDAR β learning** (`src/pruners/sdar_gate.rs`): Replace fixed β=5.0 with variance-minimized β that equalizes gated signal intensity across episodes.

**Priority:** High. Self-contained, no model needed, directly improves existing infrastructure.

### 4.2 Self-Conditioning Draft-Refine Loop — ADOPT (Model Path)

**Insight:** For 25% of training, run an initial gradient-free forward pass to estimate clean data, then feed it back as conditioning. At inference, always self-condition. Contributes 1.5 PPL improvement.

**Adaptation:**
- In MTP drafter (Plan 055) and speculative decoding, implement a draft-refine loop where the first pass generates a coarse prediction and the second pass refines it conditioned on the draft.
- Compatible with existing `GpuForwardPass` — just run forward twice.

**Priority:** Medium. Requires wgpu kernel changes but compatible with existing MTP infrastructure.

### 4.3 Over-Training Small Models with ELBO Auxiliary — CONSIDER

**Insight:** RePlaid beats MDLM in the over-trained regime (3.1× vs 6.9× past optimal). ELBO regularization prevents overfitting in small models.

**Adaptation:**
- Add an ELBO-style auxiliary loss to wgpu LoRA training (Plan 008) that regularizes embedding geometry alongside cross-entropy.
- Relevant for QLoRA + IA3 PEFT (Plan 071) targeting small, efficient models.

**Priority:** Medium. Requires new wgpu loss kernels but the scaling law justification is strong.

### 4.4 Low-Rank Game State Embeddings — CONSIDER

**Insight:** ELBO training discovers low-rank, linguistically structured embeddings at d_e=16 without supervision. Adding CE loss disrupts this structure.

**Adaptation:**
- Apply variance-minimized schedule optimization to game state representations (`GoState`, `BomberState`) to discover latent game structure without labeled data.
- Relevant to SpectralQuant eigenbasis calibration (Plan 077) — same principle of letting variance minimization discover structure.

**Priority:** Low-Medium. Interesting but would require new infrastructure.

### 4.5 Over-Training Advantage Validates Modelless — ADOPT (Conceptual)

**Insight:** RePlaid's over-training crossover (Fig 1c) proves small, aggressively-trained models with proper regularization can beat larger compute-optimal models.

**Implication for modelless distillation:**
- Our GFlowNet modelless (Plan 052) and SDAR modelless (Plan 072) train pruners on observed game trajectories — effectively "over-training" on limited data
- RePlaid shows this is a valid strategy: small models trained past compute-optimal with self-supervised regularization beat larger models
- Our `.benchmarks/008_sdar_gated_modelless.md` results (negative for arena, positive for components) align with RePlaid's finding that not all over-training regimes help — the regularization method matters

**Priority:** Conceptual. Validates existing modelless strategy, no new code needed.

### 4.6 Higher-Order D2F Denoising — CONSIDER

**Insight:** RePlaid Sec 4.2 shows DPM-Solver++(2M) beats DDPM at low NFEs. The solver uses linear extrapolation from cached predictions (Eq 16-17), reducing steps by 4× with maintained quality.

**Adaptation for D2F:**
- Our `d2f_decode_block()` uses iterative denoising (DDPM-style, step-by-step)
- A second-order variant could cache the previous step's logits and extrapolate
- Potential: reduce `D2fDecodeConfig::quality()` from 16 steps to 4 steps
- Implementation: WGSL kernel for multistep logit extrapolation (cache `x_θ^(i-1)`, `x_θ^(i-2)`)
- This is discrete-adapted: we cache logit vectors, not continuous latent states

**Priority:** Medium. Would require new WGSL kernels but could 4× D2F throughput.

---

## 5. What We Do NOT Distill (Rejected)

### 5.1 Full Continuous Diffusion Pipeline — REJECTED

**Reason:** Same verdict as Research 10 (ColaDLM). Our DDTree speculative decoding branches on discrete token indices. Continuous diffusion trajectories branch on latent vectors. These are fundamentally incompatible.

Our D2F (Plan 066) is discrete diffusion and works well with our stack. Switching to continuous would require rewriting the entire DDTree + ConstraintPruner + ScreeningPruner pipeline.

### 5.2 Low-dim Embeddings (d_e=16) Everywhere — REJECTED

**Reason:** RePlaid's 50× FLOP savings at d_e=16 vs d_e=768 is specific to the embedding corruption step in continuous diffusion. Our game states and validator embeddings serve different purposes (classification, reward, constraint checking, not generative modeling).

### 5.3 ELBO as Sole Training Objective — REJECTED (with nuance)

**Reason:** Our SDAR (Plan 072) and ROPD (Plan 071) results show sigmoid gating and rubric vectors work well modellessly. Adding ELBO would be over-engineering for the modelless path. The model path uses cross-entropy + SDAR loss, which our benchmarks show is effective.

**Nuance (Sec 2.6):** If we ever combine ROPD and SDAR losses, we must gate ROPD through SDAR — never add them independently. RePlaid's ELBO+CE experiment (22.1→26.1) shows that mixing objectives with different embedding geometry properties is destructive. SDAR's sigmoid gating preserves structure; ROPD's pointwise scoring disperses it.

### 5.4 ODE Solver Distillation — PARTIALLY ADOPTED

**Original stance:** Continuous diffusion enables trajectory distillation via PFODE solvers, unavailable to discrete diffusion. Long-term research direction only.

**Updated stance (Sec 4.6):** The **solver acceleration** insight transfers. RePlaid shows DPM-Solver++(2M) achieves comparable quality in 4× fewer steps via multistep extrapolation. Our D2F pipeline can adopt the same principle: cache previous denoising predictions and extrapolate. This doesn't require continuous diffusion — just smarter step scheduling. Promoted to Section 4.6.

---

## 6. Honest Corrections and Caveats

### 6.1 The 20× Gap is Still Large

RePlaid closes the gap from 64× (original Plaid) to 20×, but 20× more compute than AR is still impractical for production at our scale. The theoretical insights (variance minimization, linear info decay) are more valuable than the mechanism itself.

### 6.2 RePlaid's Gains Are Architecture-Specific

The PPL improvements come from aligning continuous diffusion with discrete DLM architecture + adding Plaid-specific components. The variance-minimized schedule works for any process with per-step losses, but the specific PPL numbers are not transferable.

### 6.3 Self-Conditioning Has Entropy Cost

Self-conditioning improves PPL but reduces output entropy at high sampling steps (Fig 3c, 4a). For our game arenas, lower entropy means less diversity in exploration. This is a trade-off we must monitor.

### 6.4 Theoretical Results Assume Bayes-Optimal Denoiser

Prop 2 (near-linear CE) assumes a Bayes-optimal denoiser (Def 1). Our D2F models are far from Bayes-optimal. The practical benefit of the linear schedule may be smaller than the theory predicts.

---

## 7. Actionable Mappings

### 7.1 Immediate (Modelless — microgpt-rs)

| Code Location | Current | Change | Insight Source |
|---|---|---|---|
| `src/dllm.rs` `NoiseSchedule` | Fixed monotonic min→max | Variance-minimized ratios (track per-step loss, equalize difficulty) | Prop 1 |
| `src/pruners/bandit.rs` `BanditStrategy` | Fixed ε, UCB1 | Add `VarianceEpsilon` that adapts ε to equalize per-episode reward variance | Prop 1 analog |
| `src/pruners/sdar_gate.rs` `SDAR_BETA` | Fixed 5.0 | Add `learned_beta()` that minimizes gated-signal variance across episodes | Sec 5.2 |
| `src/pruners/sdar_gate.rs` + `ropd.rs` | Independent losses | Gate ROPD through SDAR (never add independently) | Sec 2.6, Fig 5c |

### 7.2 Medium-term (Model — riir-ai)

| Code Location | Current | Change | Insight Source |
|---|---|---|---|
| `riir-gpu/src/backward.rs` | Pure CE loss | Add ELBO auxiliary loss (reconstruction + diffusion terms) | Sec 2, Eq 4 |
| `riir-gpu/src/forward.rs` `GpuForwardPass::forward()` | Single forward pass | Add self-conditioning path (forward → estimate → forward with conditioning) | Sec 2 |
| `riir-gpu/src/training_loop.rs` `TrainingConfig` | Fixed LR schedule | Add per-component LR (embeddings: 1e-2, backbone: 2e-4) from Tab 7 | Tab 7, 9 |

### 7.3 Medium-term (D2F Acceleration)

| Code Location | Current | Change | Insight Source |
|---|---|---|---|
| `src/speculative/d2f.rs` `d2f_decode_block()` | Iterative DDPM-style (16 steps) | Higher-order multistep solver with cached prediction extrapolation (4 steps) | Sec 4.2, Eq 16-17 |
| `src/dllm.rs` `D2fContext` | Single-step logit buffer | Add `prev_logits` cache for DPM-Solver++(2M) extrapolation | Appendix D |

### 7.4 Long-term Research

- PFODE distillation path for continuous embeddings (if we ever adopt them)
- Low-rank game state embeddings via variance-minimized optimization
- Linear information decay for MCTS simulation budget allocation

---

## 8. Final Verdict

**Overall: 🟡 SELECTIVE ADOPTION**

RePlaid is significantly stronger than ColaDLM (Research 10) because its insights decompose into modular, adoptable pieces that work within our existing architecture. The paper does not change our fundamental decision to use discrete diffusion (D2F) over continuous diffusion.

**The most valuable contribution is the theoretical framework:**
1. **Prop 1** — Variance minimization yields optimal schedules (self-supervised, no teacher needed)
2. **Lemma 3** — Linear information decay under optimal conditions (principles for budget allocation)
3. **Embedding geometry** — ELBO creates structured low-rank spaces without supervision

**The mechanism (continuous diffusion itself) is rejected** for the same reasons as Research 10: incompatible with DDTree speculative decoding.

**Recommended next steps:**
- Plan 078 (microgpt-rs modelless): Variance-minimized schedules for D2F noise + bandit exploration + SDAR β
- Plan 079 (riir-ai model-based): Self-conditioning draft-refine loop + ELBO auxiliary loss for wgpu LoRA

---

## 9. Paper Metadata

| Field | Value |
|---|---|
| Parameter Scale | 19M–2121M (scaling law), 0.1B (benchmark) |
| Training Data | SlimPajama (scaling law), OpenWebText (benchmark) |
| Tokenizer | Llama-2 (V=32,001) |
| Sequence Length | 2048 (scaling law), 1024 (benchmark) |
| Embedding Dimension | d_e=16 (low-dim, learnable, unit-norm) |
| Key Baselines | AR, MDLM, Duo, FLM, LangFlow |
| Compute | 8× GB200 (scaling), 8× H200 (benchmark) |
| Key Ablation | Learnable embeddings: +17.3 PPL (Table 2) |
| Samplers | DDPM, DDIM, DPM-Solver++(2M), Heun |
| Feature Flags | Self-conditioning (25% train / 100% inference) |