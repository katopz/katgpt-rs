# Research 73: LT2 — Linear-Time Looped Transformers

> **Paper:** [LT2: Linear-Time Looped Transformers](https://arxiv.org/abs/2605.20670) — Deng, Zhang, Zhu, Xu, Liu, Ng, Chen (Rice/Apple/UCSC/CMU), May 2026
> **Code:** https://github.com/facebookresearch/lingua (forked, apps/LT2)
> **Date:** 2026-05, distilled 2025-07
> **Related Research:** 28 (HLA), 70 (Gated DeltaNet-2), 71 (DashAttention), 55 (Nemotron TriMode), 58 (GRAM), 057 (Higher-order LA), 097 (Delta Attention Residuals)
> **Related Plans:** 108 (LT2 Looped Inference Pipeline)
> **PASS-Redirects (synthesis):** Loopie [arXiv:2607.16051 "Loop the Loopies!"] — layer-loop swap `for layer { for tau }` **conflicts** with our LT2 rank-T state-upgrade math which requires model-loop ordering `for tau { for layer }` (shipped as `forward_looped`); paper's training-only contributions (Recipe/SPT/GSPO+DAPO RL) → riir-train. Loopie's §8 admits "we have not yet conducted systematic studies of inference-time computation".
> **PASS-Redirects (synthesis):** MoR — Mixture-of-Recursions [arXiv:2507.10524 "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"] (Bae et al., KAIST/Google, Jul 2025) — pre-training framework that trains a router end-to-end to assign per-token recursion depth inside a weight-shared recursive transformer (Middle-Cycle sharing + expert-choice/token-choice routing + recursion-wise KV caching). §3.5 Path 0 fails: the router score `g = sigmoid(θ^T h)` is modelless math, but θ is an irreducibly trained artifact ("which tokens need more depth" is learned from the data distribution during pretraining) — not a closed-form decomposition like Flow Sampling's conditional drift. Paths 1-3 all fail (no frozen snapshot, no deterministic LoRA construction, no latent projection substitutes for a learned router). The architecture (weight-shared looping) is shipped as LT2 (Plan 108); per-token compute allocation is shipped as Self-Advantage Gate (Plan 283) + PathwayTracker (Plan 231) + Collapse-Aware (Plan 212); block-level routing is shipped as dMoE (Research 161). Router training + MoR-from-scratch pretraining → **riir-train**.
> **PASS-Redirects (modelless-proxy follow-up, 2026-07-31):** The prior PASS verdict's optional follow-up ("can a deterministic modelless proxy match MoR's trained router allocation?") was investigated and closed. MoR §5.2 says the router learns "contextual predictability of the subsequent token" — which is exactly an entropy / max-logit signal. That signal is **already shipped modellessly** in three forms: (1) `DendriticGate` (Plan 260, `katgpt-core::dendritic_gate`) — `low_entropy_closes_gate` / `high_entropy_high_coincidence_opens_gate` tests confirm high-entropy tokens get more compute, low-entropy tokens exit; (2) `entropy_from_logits` shipped in `katgpt-transformer/swir/entropy.rs` + `attn_match_adaptive_cot.rs` + `riir-engine/llmexec_guard_bridge.rs`; (3) `early_exit_patience` + `early_exit_gap` (dd_tree) for absolute-gap confidence. Self-Advantage Gate (Plan 283) is the IMPROVEMENT-based variant of the same functional goal (per-token compute allocation via confidence). Whether these modelless proxies MATCH MoR's trained router allocation is empirically unprovable without a MoR-pretrained model (riir-train) — and the paper provides no head-to-head comparison. **Conclusion: no new primitive needed; the functional goal is covered; the only MoR-specific delta is the trained θ, which is irreducibly a riir-train concern.**
> **PASS-Redirects (synthesis):** "Towards Looped Models Done Right" [notion:ifm-research/Towards-Looped-Models-Done-Right, Huang/Shi/Chen/Wen/Liu/Xing/Ma, Aug 2026] — systematic controlled ablation of Ouro→Huginn design space (sandwich envelope, input injection, latent-state org) trained from scratch under matched params/depth/tokens. Findings validate our shipped choices: (1) sandwich/prelude-loop-coda envelope wins on math/reasoning — consistent with LT2 windowed mode + Research 097 depth-fraction rule (0.45–0.60 mid-stack); (2) shared-module H/L hierarchy has no consistent benefit — confirms HRM-Text rejection (Research 048); (3) random recurrent-state init is mixed (direct init better on 6/10 benchmarks) — consistent with Plan 276 MicroRecurrentBeliefState null result at random init. Input-injection finding (helps context/code but HURTS math) is a cautionary flag for any fusion adding persistent injection to belief kernels; no shipped config contradicts it. MoE iteration-specific-expert-selection finding applies to TRAINED looped MoEs (riir-train territory); our Training-Free Loop layer-mode (note 097) remains correct for frozen MoE checkpoints. Training-time architecture decisions → riir-train.
> **PASS-Redirects (synthesis):** Ouro [arXiv:2510.25741 "Scaling Latent Reasoning via Looped Language Models"] (Zhu/Wang/Hua/Zhang/Li/Que/Wei/Wen/Yin/Xing/Li/Shi/Ma/Li/Kergan/Smith/Qu/Hui/Wu/Min/Huang/Zhou/Ye/Liu/Yang/Shi/Lin/Zhao/Cai/Zhang/Huang/Bengio/Eshraghian, v5 Jul 2026) — open-sourced family of PRE-TRAINED Looped Language Models (LoopLM, 1.4B + 2.6B) with (i) iterative latent computation, (ii) entropy-regularized objective for learned depth allocation, (iii) scaling to 7.7T tokens; matches up to 12B SOTA LLMs, claims advantage stems from knowledge MANIPULATION (not capacity) + reasoning traces more aligned with outputs than explicit CoT. §3.5 Path 0 fails: the depth allocator IS the entropy-regularized pre-training objective (not closed-form math like Flow Sampling's conditional drift — the policy is a learned artifact of training-from-scratch on 7.7T tokens). Paths 1-3 fail (gain is structural — base weights trained to reason via iteration, not a snapshot/LoRA/latent-projection-correctable systematic bias). Pre-training recipe + 7.7T-token scaling → **riir-train**. Runtime mechanism (looped latent compute + depth halting) covered by shipped prior art: LT2 (Plan 108) = the architecture; Training-Free Loop (Plan 136 / Research 097) = zero-training wrapper for frozen checkpoints; GainCostLoopHalter (Plan 304 / Research 282) = per-loop gain/cost halting (the runtime analog of Ouro's trained depth allocator — modelless via gain/cost economics, not entropy-regularized training); Fully Looped Transformer (Research 414) = parameter-free loop stability. "Knowledge manipulation > capacity" + "traces more aligned than CoT" = scaling validations of the latent-reasoning thesis, no actionable config change for our stack. Ouro is the PARENT paper of "Towards Looped Models Done Right" (same Bengio/Eshraghian lab; "Done Right" is the controlled ablation of Ouro→Huginn design space) — both verdicted PASS on the same grounds (training-time architecture decisions → riir-train). Sibling verdicts: Research 343 (System-1.5 — same pattern: training-only depth+step routing → PASS), Research 273 (ELT — Any-Time framing, ILSD training → riir-train).
> **PASS-Redirects (synthesis):** Wang et al. [arXiv:2608.08888 "Full-bandwidth transformer"] (Microsoft/JHU/Princeton, Aug 2026) — latent feedback decoding fuses the previous top-layer hidden state with the sampled token embedding via a GLU (`W_U h_{t-1} ⊙ σ(W_G e_t)`, state on value pathway, token as gate) and feeds it back as the next input, widening the inter-step feedback channel from 1 token to D dimensions. Negligible inference cost (2 D×D matmuls/token), vLLM-compatible, 1.5× data efficiency at 1B/400B-token scale. §3.5 Path 0 fails: GLU fusion is modelless math, but the paper explicitly states "a pretrained model has never seen hidden states in its input, so latent feedback cannot simply be switched on at inference" — the model's inability to process hidden-state-space inputs is a missing capability, not a correctable systematic bias. Paths 1–3 all fail. Multi-pass scheduled pretraining (progressive schedule + prefix mixin + jitter noise + depth scaling + weight tying) is load-bearing → **riir-train**, but at a scale (1B+ model, 100B+ tokens) beyond our current training scope (C13/C14 0.4B from-scratch, Gemma-2 rank-16 LoRA). Runtime mechanism already shipped modellessly: SwiR soft-embedding feedback `ẽ_t = Σ_v p_t[v]·e(v)` (Research 241, Plan 275, DEFAULT-ON — the training-free analog that projects hidden state back through the vocab embedding matrix, staying in token space the model can already process); LT2 looped weight-sharing (this note, Plan 108); Training-Free Loop stack looping (Research 097, Plan 136). NPC cognition already has full-bandwidth feedback at the belief level — HLA `evolve_belief` carries the full 8-dim affective state forward across ticks, so there is no narrow token-level channel to widen on the game-AI hot path. Same PASS pattern as Ouro / System-1.5 / ELT / MoR / "Towards Looped Models Done Right". No new primitive, no plan, no guide.
> **PASS-Redirects (synthesis):** SMELT [arXiv:2609.01343 "SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers"] (Wang & Zhang et al., Tsinghua/ByteDance Seed/M-A-P/TokenWave, Sep 2026) — first looped study to match ALL THREE budgets at once (per-token FLOPs + total non-embedding params + KV cache, residual mismatch ≤4%) across a 4-scale × 4-sparsity ladder to 54B non-embedding params, fitting a SEPARATE Chinchilla surface per architecture: L = E + A(1−S)^b/F^a + K/D^c with compute-equivalent sparsity S from FLOPs ratios. Recipe (SMELT = loop middle-50% of layers exactly 2×, narrow H to pay for the extra visit, recover params via expert count, retune head/GQA for KV parity, scale looped-span residuals by 1/r): frontier exponent γ 0.250 vs baseline 0.237 → 6.8–18% training-FLOP savings at equal loss (CE-Gain inversion), concentrated on Code (20.4% CE), long samples (1.52× the short-bucket gain; matched parameter/expert-addition controls stay flat 0.98/0.88), and ICL (gap widens with demonstrations; Dyck 29.8 vs 26.4 at k=32). Mechanism: visit-2 amplifies ALIGNED residual writes (1.2–3.5× norm ratio, cross-visit cosine 0.56 vs 0.16 off-diagonal), reuses Q/K retrieval coordinates (cos 0.89–0.93) while V diverges (0.65–0.74), reuses a core expert subset (2–3/8 shared at S≈97%, far above random), and cuts attention-sink mass (BOS 0.60→0.02 on the Dyck head) against the Baseline's sink-grows-with-depth trend. §3.5 Path 0: the win is a FROM-SCRATCH PRETRAINING effect — looping pays only when weights are TRAINED shared (WSD 215B tokens; 1/r scaling + routing divergence across visits are load-bearing) — the same learned-artifact failure class as Ouro/MoR; paths 1–3 n/a; full pretraining out of scope per the fusion ladder → **riir-train**. Corroborations: r=2 confirms Loopie [Gao et al.]; middle-50% span + larger effective depth-to-width confirm "Done Right"'s sandwich envelope; the 1/r looped-residual scaling is the TRAINED twin of Research 097's damped Euler 1/K sub-stepping (same form, opposite regimes). Modelless residue: (a) sink-is-parking corroboration for Research 487/Issue-716 — but frozen NON-looped checkpoints keep persistent sinks, so the sink-sidecar policy stands unchanged; (b) GUARD: do NOT try mid-layer re-execution on frozen Bonsai/qwen38 as a quality lever — SMELT's gains exist only under looped training; the Training-Free Loop (Research 097 / Plan 136) remains the only sanctioned frozen-checkpoint loop; (c) CE-Gain (compute saved at equal loss) is the loss-side mirror of what the league already measures (tok/s at pinned bit-identity); a live consumer would be a future lossy-surface league cell with a loss metric — none today. (d) MODELLESS-MOE AUDIT (user-challenge discharge, 2026-09-02): the stack DOES carry modelless MoE — FAME R302 (per-entity frozen blend π, sampling-invariance is the design goal — deliberately NO visit divergence), dMoE R161 (block-coreset routing; its own MoR redirect already sent loop×router θ to riir-train), MoA R126, ZEDA R107, healer KernelExpertRouter, + the SERVED kimi_k3 latent-MoE FFN (R447, Issues 693/694). Signal-diff: SMELT consumes trained-router dynamics across repeated weight-tied FFN executions (divergence recovers tying-lost expressivity; 2–3/8 reuse) — none of our modelless MoEs has a visit axis, weight tying to recover, or a trainable router; kimi_k3's MoE is a served checkpoint (pin/observe only — R097 layer-mode pinning is the shipped frozen-MoE answer, and SMELT adds zero frozen-checkpoint evidence). If LT2 ever loops a kimi KDA span crossing MoE FFNs, R097's pinning applies unchanged. No new primitive, no plan, no guide.

> **PASS-Redirects (synthesis):** Sotaku [github.com/chenglou/sotaku "Sotaku v2 — 99.12% of a 25K-puzzle Sudoku benchmark with an 800K-parameter looped transformer"] (Cheng Lou, Sep 2026, MIT, pinned `6cdb9a9b`) — NOT an LM: a 797K-param 4-layer shared-weight supervised solver (2D RoPE, prediction-feedback recurrence, 16 train iterations) whose value is the **late-state training recipe** (20% of batches: no-grad burn-in to horizon ~ U{32…512}, detached, then the ordinary 16-iteration averaged CE from the detached state — trajectories reach iteration 528 at O(16) memory, no fixed-point assumption) plus the measured **16→1024→4096 peak-and-decline iteration-scaling curve** (99.12% @1024 FP32; BF16 @4096 = 43.7% — precision amplifies with loop depth, the mirror of Bench 802's attention-dilutes finding) and a **delayed-damping runtime rescue** (`h ← (1−α)h + αF(h)` after burn-in B; 5.64→95.66% @1024 on a collapsed checkpoint; tangential-only ×0.25 also rescues, radial-only often worsens; NOT-a-fixed-point evidence: state RMS 24→706 while residual plateaus 0.63, root-solvers diverge — which kills IFT/one-step backward for this class and complements Research 035). No new architecture class (UT lineage); training recipe → **riir-train** (Plan 373 + Research 440 there); modelless runtime residue (damping knob, tangential scaling, f32-state contract, relative-residual halting trap) → katgpt-rs **Issue 717**; `forward_looped` (Plan 108, T=4) is the shipped inference-side cousin — deep-T behavior currently unmeasured, which Issue 717 owns. Corroborations: SMELT/Loopie looped-scaling verdicts unchanged (sotaku is supervised-solver, not LM-scale pretraining); LDT (2605.08605, 800K, 100%-with-abstention, lattice supervision) remains the tiny-model SOTA claim — sotaku is rule-agnostic SOTA-adjacent on a 2.7M-puzzle pool (not data-controlled vs HRM/TRM's 1K protocol). [Superseded 2026-10-01: deep-T IS measured — Bench 699 T1/T2, `forward_looped` flat to T=1024 on the stable fixture; damping + radial/tangential axis toolbox shipped there.]
> **PASS-Redirects (synthesis):** Looped-DiT [arXiv:2609.40305 "Looped Diffusion Transformer"] (Chng/Chen et al., SenseTime/LeapLab-Tsinghua/NTU, Sep 2026) — the T2I member of the training-time looped-scaling stack (first systematic looped T2I study, MiniT2I MMDiT [6,5,6] split, N=4): naive looping ERODES linearly-decodable info (ridge position-probe R² 0.865→0.562 over 8 loops), fixed by Deep Supervision (final+mean (1/3,1/3,1/3,1) beats exponential; intermediates decoded through the SHARED post-loop stage — training-only ~1.54× FLOPs, zero inference cost; stabilizes early exits + train@4→infer@8) and Self-Modulating Attention — XSA G = I − v̂v̂ᵀ on softmax head outputs, looped blocks only, loop-SPECIFIC (+1.6 looped vs −0.1 non-looped); XSA is the attention-write member of the 414/LoopStabilityMode stability taxonomy and a DIFFERENT mechanism from our measured radial-magnitude axis (Bench 699 T4) — fit to our failure mode unmeasured. Adaptive halter trained on realized max-future-gain (Eq. 7) = the trained sibling of Plan 304's modelless halter (282's line). Loop depth > more denoising steps at matched inference FLOPs = diffusion-side datapoint for the 343/593 allocation axis, recorded not claimed as a law. Training recipes → riir-train Plan 421 Phase 6 riders; modelless residue (probe-R² + update-norm diagnostics) → Bench 699 T8 row.
> **Verdict: HIGH VALUE — LT2's looped weight-sharing is a natural fit for our parameter-constrained CPU inference. The rank-T state upgrade from looping directly amplifies our existing HLA/AHLA recurrent states. Hybrid (Full+GDN) with 1:4 ratio is the flagship recipe. SDPA output gate is a free lunch (+0.3–0.5 avg points). Feature-gate as `lt2_looped`. Priority: looped AHLA (our existing linear attention) first, then hybrid with windowed SDPA.**

---

## TL;DR

LT2 replaces the quadratic attention in Looped Transformers with subquadratic token mixers (linear, sparse, or hybrid). The key finding: **looping synergizes uniquely with subquadratic attention** — T loops turn rank-1 DPLR state updates into rank-T updates (enabling state tracking), and turn window-w sparse attention into effective receptive field T·w (enabling long-context).

Two flagship hybrid variants:
1. **LT2-hybrid (Full+GDN)** — 1:4 full-to-linear ratio. Best quality: +2.1 avg points over standard looped transformer at 1.3B, ~2.7× decode speedup.
2. **LT2-hybrid (GDN+DSA)** — Fully linear-time. Matches full-attention looped transformer quality with ~5.7× decode speedup.

Distillation pathway: Pre-trained Ouro-1.4B → LT2-hybrid with ~1B tokens training, competitive with industry 4B models.

**For our stack:** We already have AHLA (O(1) memory linear attention). Looping AHLA T=4 times gives rank-4 state updates for free — same weights, 4× effective depth. The SDPA output gate eliminates attention-sink compounding. The hybrid pattern (1:4 full+linear) maps directly to our existing SDPA + AHLA dispatch.

---

## Core Innovation: Loop × Subquadratic Synergy

### 1. Rank-T State Upgrade (Loop × DPLR Linear Attention)

A single DPLR block (GDN, KDA, DeltaNet) applies a rank-1 perturbation to recurrent state:

```
S_t = A_t · S_{t-1} + β_t · k_t · v_tᵀ
A_t = Diag(α_t)(I - β_t · k_t · k_tᵀ)  // rank-1 + diagonal
```

Looping T times composes T such operators:

```
A_eff = ∏_{τ=1}^{T} Diag(α_t^(τ))(I - β_t^(τ) · k_t^(τ) · k_t^(τ)ᵀ)
```

**Key result:** When loop-specific keys are approximately orthogonal (expected in high-dim spaces), the effective perturbation rank is T, not 1. By Cartan-Dieudonné theorem, T ≥ d_k loops suffice to realize any orthogonal transformation in O(d_k).

| Loops | Rank | State Tracking |
|-------|------|---------------|
| T=1 | 1 | Cannot solve S_n (n≥3) |
| T=2 | 2 | Reflections + rotations in 2D |
| T=4 | ≤4 | Solves prefix products for S_5 |
| T=d_k | ≤d_k | Universal orthogonal representation |

**Connection to our HLA (Research 28):** Our AHLA maintains (PKV, mK, E, n) state in O(d·dv). Looping AHLA T times means T independent key projections acting on this state — the same rank-T upgrade applies. Our second-order SK accumulator would benefit even more: T loops produce T rank-1 key-direction corrections to SK, yielding a rank-T update to the key second-moment matrix.

### 2. Receptive Field Expansion (Loop × Sparse Attention)

Window-w sparse attention, looped T times:

```
ℐ_t^(T) ⊇ {max(1, t - T·w + 1), ..., t},  |ℐ_t^(T)| = O(T·w)
```

T=4 loops of window-2048 → effective receptive field of 8192. This matches 4 stacked layers of window-2048 attention but with 4× fewer parameters.

**Important caveat (from paper Appendix B.2.2):** Residual connections cap the *effective* receptive field. With residual skip connections (α ≈ 0.95), influence decays as `(1-α)^{⌈d/w⌉}`, yielding an effective horizon of ~1.5w regardless of T. The combinatorial reach is O(Tw), but the signal quality at distance d > 2w is exponentially attenuated.

**Practical implication:** Looping sparse attention helps mostly in the 1–2w band. For truly long-range recall, you still need either (a) some full attention layers, or (b) linear attention with recurrent state carry.

### 3. SDPA Output Gate (Attention Sink Suppression)

In looped transformers, the attention sink (first-token mass concentration) compounds across loops — a sawtooth pattern that intensifies each iteration. Fix: a head-specific sigmoid gate after SDPA:

```python
gate = sigmoid(x @ W_gate.T)  # zero-init → starts at 0.5
output = sdpa_output * gate    # before output projection
```

Results at 1.3B:
| Model | Gate | PPL | Avg |
|-------|------|-----|-----|
| Looped Transformer | — | 9.87 | 59.27 |
| Looped Transformer | ✓ | 9.39 | 60.70 |
| LT2-Hybrid (Full+GDN) | — | 9.31 | 61.39 |
| LT2-Hybrid (Full+GDN) | ✓ | **9.03** | **62.33** |

---

## Hybrid Architecture: The Pareto Frontier

### Depth-Level Hybrid (1:4 Full+GDN Interleave)

The winning pattern: every 5th layer uses full attention, the other 4 use linear attention.

```
[GDN, GDN, GDN, GDN, Full, GDN, GDN, GDN, GDN, Full, ...]
```

Ablation results (1.3B, T=4):
| Full:GDN Ratio | PPL | Avg |
|----------------|-----|-----|
| 1:0 (full only) | 9.87 | 59.27 |
| 1:1 | 9.41 | 60.92 |
| **1:4 (optimal)** | **9.03** | **62.33** |
| 1:6 | 9.36 | 61.07 |
| 1:12 | 9.74 | 59.51 |
| 0:1 (GDN only) | 10.02 | 58.42 |

Pattern placement matters: bookend > interleave > front-loaded > back-loaded. Spreading full attention layers is critical.

### Loop-Level Hybrid (Coarse→Fine)

Less effective than depth-level. The paper tried:
- Full → SWA-512 → SWA-256 → SWA-128 (coarse→fine)
- SWA-128 → SWA-256 → SWA-512 → Full (fine→coarse)

Both underperform fixed depth-level interleave. The coarse→fine wins on PPL but loses on downstream (overfits local statistics in final loops).

---

## Key Experimental Results

### Language Modeling (100B tokens FineWeb-Edu, T=4)

| Model (1.3B) | PPL | ARC-E | ARC-C | HellaS | PIQA | Avg |
|---|---|---|---|---|---|---|
| Transformer | 10.65 | 67.52 | 33.84 | 52.47 | 71.03 | 56.04 |
| Looped Transformer | 9.87 | 70.83 | 37.54 | 57.06 | 72.43 | 59.27 |
| Looped GDN | 9.75 | 71.28 | 38.33 | 57.73 | 73.37 | 59.92 |
| Looped KDA | 9.68 | 71.57 | 38.62 | 57.99 | 73.53 | **60.14** |
| Looped DSA | 9.97 | 69.93 | 36.93 | 56.38 | 71.94 | 58.54 |
| **Hybrid (Full+GDN)** | **9.03** | **74.82** | **41.63** | **61.04** | **75.93** | **62.33** |
| Hybrid (GDN+DSA) | 9.50 | 72.44 | 39.33 | 58.84 | 73.98 | 60.73 |

### Efficiency at Long Context (1.3B, H100)

| Variant | Decode@8K (t/s) | Decode@32K (t/s) | OOM Frontier (bs=8) |
|---------|-----------------|------------------|---------------------|
| Looped Transformer | 125 | 22 | 8K |
| Looped GDN | 135 | 120 | >32K |
| Hybrid (Full+GDN) | 130 | 105 | >32K |
| Hybrid (GDN+DSA) | 128 | 115 | 16K |

Linear-time variants hold flat decode throughput across the entire range. Looped Transformer loses 82% of throughput between 4K and 32K.

### Distillation: Ouro → LT2-hybrid

3-stage recipe:
1. **Linear pre-alignment** (100M tokens, len=512): MSE loss aligning GDN blocks to teacher attention outputs
2. **Hybrid logit distillation** (600M tokens, len=4096): KL-div with per-loop supervision schedule
3. **Long-context continuation** (600M tokens, len=32768): extend with OpenThoughts reasoning data

Result: Ouro-Hybrid-1.4B matches industry 1B models, approaches 4B models, with ~1B tokens total training.

---

## Training Stability Findings

Critical for implementation. The paper found clear stability tiers:

| Tier | Mixers | Behavior |
|------|--------|----------|
| **Most stable** | GDN (gating + delta rule) | Smoothest loss, smallest gradient norms |
| **Stable** | Mamba2 (gating, no delta rule), DeltaNet (delta rule, weaker gating) | Occasional spikes |
| **Unstable** | RetNet (no gating, no delta rule) | **Diverges** |

**Takeaway for us:** Our looped implementation MUST include data-dependent gating (α_t) and preferably a delta-rule update (β_t · k_t · v_tᵀ). Our AHLA already has channel-wise decay — this maps to α_t. The delta rule is additive — easy to layer on.

---

## Mapping to Our Architecture

### What We Already Have

| LT2 Component | Our Equivalent | Status |
|---|---|---|
| GDN linear attention | AHLA (asymmetric HLA) | ✅ Implemented (Plan 057) |
| Sliding window attention | SDPA with window | ✅ In transformer.rs |
| SDPA output gate | — | ❌ Not yet |
| Loop weight sharing | — | ❌ Not yet |
| Per-loop residual gate ρ_τ | — | ❌ Not yet |
| DSA sparse attention | DashAttention α-entmax | 🔬 Research 71 |
| GDN2 channel-wise erase | — | 🔬 Research 70 |

### Natural Fit Points

1. **Looped AHLA** — Our AHLA already maintains O(d·dv) constant state. Looping T=4 times gives:
   - 4× effective depth with same parameter count
   - Rank-4 state updates (up from rank-1)
   - ~95% of SDPA throughput maintained (from our benchmarks)
   - No KV cache growth per loop iteration

2. **Hybrid SDPA+AHLA** — Our `forward()` already dispatches on `HlaMode`. Adding a loop with depth-level hybrid:
   - Every 5th "layer" uses standard SDPA (exact recall)
   - Other 4 use AHLA (constant memory, streaming)
   - Only 1/5 of layers pay quadratic cost

3. **SDPA Output Gate** — Zero-init sigmoid gate after attention, before Wo projection. ~`n_heads × head_dim` extra parameters. Free +0.3–0.5 avg points.

### What We Don't Need

- **RetNet, HGRN2, DeltaNet, Mamba2** — Our AHLA is our linear attention. The paper confirms: gating + delta rule matters, specific mixer family matters less.
- **NSA** — DashAttention (Research 71) covers this with α-entmax, which is strictly better than top-k routing.
- **Loop-level hybrid** — Underperforms depth-level. Skip.
- **ACT (adaptive computation time)** — Paper tried it, found unstable at scale. We use fixed T.

---

## Implementation Strategy

### Phase 1: Looped Forward Pass (Feature: `lt2_looped`)

Core change to `transformer.rs`:

```rust
/// Looped transformer configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoopMode {
    /// Standard single-pass (no looping).
    #[default]
    None,
    /// Weight-shared looping: same layers applied T times.
    /// Effective depth = n_layer × loop_count.
    WeightShared {
        loop_count: usize,     // T (paper default: 4)
    },
}

/// Hybrid attention pattern for looped inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HybridPattern {
    /// All layers use the same attention mode.
    #[default]
    Uniform,
    /// Depth-level interleave: every Nth layer uses full SDPA.
    /// e.g., Interleave { full_ratio: 5 } = every 5th layer is full.
    Interleave { full_ratio: usize },
    /// Bookend: first and last layers are full, middle is linear.
    Bookend,
}
```

Forward pass change:
```rust
// Before: single pass through all layers
for layer in 0..config.n_layer {
    forward_layer(layer, ...)
}

// After: looped with weight sharing
for tau in 0..loop_count {
    for layer in 0..config.n_layer {
        let is_full = match hybrid_pattern {
            HybridPattern::Uniform => false,
            HybridPattern::Interleave { full_ratio } => {
                (layer % full_ratio) == full_ratio - 1
            }
            HybridPattern::Bookend => {
                layer == 0 || layer == config.n_layer - 1
            }
        };
        forward_layer(layer, is_full, ...);
    }
    // Per-loop residual gate: h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
    apply_residual_gate(tau, &mut hidden_state, &residual_gates);
}
```

### Phase 2: SDPA Output Gate (Feature: `lt2_looped`)

```rust
/// Head-specific sigmoid gate after SDPA, before Wo.
/// Zero-initialized → starts at sigmoid(0) = 0.5 (neutral).
pub struct SdpaOutputGate {
    pub w_gate: Vec<f32>,  // [n_heads * head_dim, dim]
}
```

### Phase 3: Looped AHLA State Carry (Feature: `lt2_looped`)

The key: AHLA state (PKV, mK, E, n) carries across loop iterations, accumulating rank-T updates.

```rust
// Per-layer AHLA state persists across loops within a single sequence
let mut ahla_states: Vec<AhlaState> = vec![AhlaState::new(config); n_layer];

for tau in 0..loop_count {
    for layer in 0..n_layer {
        if is_linear_layer(layer) {
            forward_ahla_layer(layer, &mut ahla_states[layer], ...);
        } else {
            forward_sdpa_layer(layer, &mut kv_cache[layer], ...);
        }
    }
}
```

---

## Feature Gates

| Gate | Scope | Description |
|------|-------|-------------|
| `lt2_looped` | katgpt-rs | Looped forward pass with weight sharing + hybrid dispatch + SDPA output gate |
| `lt2_looped` | katgpt-core | `LoopMode`, `HybridPattern` enums, `SdpaOutputGate` struct |

Dependencies: `lt2_looped` requires `hla_attention` (for AHLA linear layers in hybrid mode).

---

## Benchmarking Strategy

Before implementation, benchmark our existing forward pass to establish baselines:
1. **Single-layer SDPA** (current) — tokens/second, µs/step
2. **Single-layer AHLA** (current) — tokens/second, µs/step
3. **4× looped SDPA** (naive) — expected 4× slowdown, KV cache ×4
4. **4× looped AHLA** — expected ~4× compute, constant memory
5. **Hybrid 1:4 (SDPA+AHLA)** — expected ~1.5× compute, 1/4 KV cache of full loop

Target: hybrid (1:4 SDPA+AHLA) at ≥60% of single-pass SDPA throughput with 75% KV cache reduction at long contexts.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Loop instability (gradient explosion) | High | Data-dependent gating (α_t) + delta rule (β_t) in AHLA |
| Per-loop residual gate adds params | Low | Only `loop_count × dim` scalars, zero-init |
| No training loop yet | Medium | Focus on inference first; training in riir-ai |
| Effective receptive field limited by residuals | Medium | Hybrid with full attention recovers recall |
| KV cache still needed for full-attention layers | Low | Only 1/5 of layers in hybrid pattern |

---

## Open Questions

1. **How does looped AHLA quality compare to looped GDN?** Our AHLA is asymmetric (different inductive bias than GDN). The rank-T upgrade applies to both, but quality may differ. Needs empirical validation.

2. **Optimal T for CPU inference?** Paper uses T=4. On CPU, each loop iteration is pure compute (no GPU parallelism). T=2 or T=3 may be optimal for throughput/quality tradeoff.

3. **Cross-loop state sharing?** Paper explicitly notes this as future work: "principled state-sharing across loops may further improve long-context modeling." Our AHLA states naturally carry across loops — this is a potential advantage.

4. **Distillation from pre-trained models?** The Ouro→LT2 pathway requires a pre-trained teacher. We'd need to either (a) train from scratch with looped config, or (b) adapt from an existing model. Option (a) is more aligned with our modelless-first philosophy.

---

## References

- LT2 paper: https://arxiv.org/abs/2605.20670
- LT2 codebase (reference): `.raw/LT2/`
- Gated DeltaNet (GDN): https://arxiv.org/abs/2412.06464
- Gated DeltaNet-2 (Research 70): our distilled research on channel-wise erase/write
- DashAttention (Research 71): our distilled research on α-entmax sparse attention
- HLA (Research 28): our implemented second-order linear attention
- Ouro (looped LM at scale): https://arxiv.org/abs/2502.09556
> **PASS-Redirects (synthesis):** Huang, Dwivedi, Ma, Moreno, Liu [alphaXiv 2611.latent-feedback-transformers-scaling-law "Latent Feedback Is a Compute and Data Amplifier: Scaling Law for Latent Feedback Transformers on Repeated Data"] (IFM/USC/CMU, 2026-11-11; code github.com/ifm-ai/LoRM, Apache-2.0) — the budget-fair redo this family lacked: latent feedback (top-layer state via learned W into the next token's input) priced against a Transformer spending the same FLOPs on more EPOCHS over repeated data — the exact axis the 2608.08888 redirect above flagged as missing ("none repeats data"). kappa up to 1.91x compute (400 UTPP, 2ep), mu up to 1.69x data; savings grow with UTPP and budget, reverse below break-even (75/33/20/12 UTPP at budgets 1-4); one coupled law (feedback = Transformer x1.14 params but /c epochs, c = 1.20-1.27; held-out 778M losses <= 0.29% err). Training track routed: riir-train Research 497 + Issue 690 (cell on the Plan-344 micro-scale rig). Modelless: still PASS — SwiR soft-embedding feedback (Research 241, Plan 275, DEFAULT-ON) remains the shipped training-free runtime analog; a learned-W runtime requires training (the model must learn to read hidden-state inputs), the same Path-0 fail recorded above.
