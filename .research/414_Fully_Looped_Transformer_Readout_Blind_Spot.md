# Research 414: Fully Looped Transformer + Readout Blind Spot — Parameter-Free Loop Stability

> **Source:** [Simply Stabilizing the Loop via Fully Looped Transformer](https://arxiv.org/abs/2605.18797) — Fu et al. (HKBU/JLU), May 2026
> **Source:** [Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models](https://arxiv.org/abs/2606.24898) — Sharma & Vu, Jun 2026
> **Date:** 2026-07-12
> **Status:** Active
> **Related Research:** 073 (LT2 — architecture we ship), 097 (Training-Free Loop), 273 (ELT elastic any-time), 282 (LoopCoder-v2 gain/cost halting)
> **Related Plans:** 108 (LT2 — shipped, GOAT 8/8), 428 (this plan — loop-stability PoC + implementation)
> **PASS-Redirects (synthesis):** Loopie [arXiv:2607.16051 "Loop the Loopies!"] — training-time compute-matched sizing; our loop-stability work targets parameter-free inference-time fixes (inter-loop RMSNorm, FLA, Attention Injection) instead. Loopie's layer-loop variant conflicts with our model-loop LT2.
> **PASS-Redirects (synthesis):** "Towards Looped Models Done Right" [notion:ifm-research/Towards-Looped-Models-Done-Right, Huang/Shi/Chen/Wen/Liu/Xing/Ma, Aug 2026] — controlled ablation isolating the three Ouro→Huginn design axes (sandwich envelope, input injection, latent-state org) under matched from-scratch pretraining. The sandwich-envelope finding (prelude–loop–coda beats full-stack on math/reasoning) is the training-time analog of our FLA + windowed-loop topology; the input-injection finding (persistent write of fixed prelude rep helps context/code but HURTS math) is a cautionary flag for any fusion adding injection to our loop-stability fixes — combining FLA/Attention-Injection with persistent input injection may interfere with the iterative state refinement that math reasoning requires. Training-time topology study → riir-train; our parameter-free inference-time fixes (inter-loop RMSNorm, FLA, AI) remain the modelless path.
> **PASS-Redirects (synthesis):** Godey & Artzi [arXiv:2603.10145 "Lost in Backpropagation: The LM Head is a Gradient Bottleneck"] — contrast class for this note's fix taxonomy: 414's pathologies (residual explosion, gradient oscillation) admitted parameter-free §3.5-path-2 fixes, but the LM-head gradient-suppression pathology admits NO known fix (all three mitigation attempts slowed convergence) — training-time head-width budgeting only, captured in riir-train Research 416.
> **Cross-ref (riir-ai):** Proposal 018 §5.1 (T-pass: both §3.5 paths 2 and 3 are UNTRIED)
> **Classification:** Public

---

## TL;DR

Two papers diagnose and fix the **structural instability** of looped transformers (our T-pass / LT2 runtime). The fixes are **parameter-free architectural modifications** — §3.5 path 2 candidates that can be applied at inference time without gradient descent. This directly addresses Proposal 018's P0 gap (T-pass loop training): the gap may be closeable **modellessly**, not via riir-train.

**Paper 1 (Fully Looped Transformer, 2605.18797)** identifies two instability sources:
1. **Residual explosion** — hidden-state norm grows rapidly across loop iterations.
2. **Gradient oscillation** — gradients accumulate through shared looped blocks during training.

Fixes (both parameter-free, zero new learnable params):
- **Fully Looped Architecture (FLA)**: feed the previous loop's output `h_L^(t-1)` to EVERY layer (not just the first). Currently our `forward_looped` only uses `prev_h` for the post-loop residual gate (line 505). FLA makes it available to all layers.
- **Attention Injection (AI)**: use `h_L^(t-1)` as the **Query** in cross-attention (K, V from current layer). This routes the recurrent signal through softmax, bounding its magnitude. Reuses existing WQ, WK, WV — zero new params.

**Paper 2 (Readout Blind Spot, 2606.24898)** identifies the root cause of residual explosion:
- Scale-invariant readouts (RMSNorm, LayerNorm) **hide radial scale** from cross-entropy.
- Per-loop CE drives hidden-state norms into thousands/tens-of-thousands.
- Fixes: (a) **scale-visible readouts** [architectural], (b) **explicit norm penalties** [training], (c) **scale-removing recurrence** (inter-loop normalization) [architectural].

**Distilled for katgpt-rs (modelless, inference-time):** The transferable primitives are three parameter-free architectural modifications to `forward_looped`:
1. **Inter-loop RMSNorm** — normalize `ctx.x` between loop iterations (scale-removing recurrence).
2. **Fully Looped Architecture** — feed `ctx.prev_h` into every layer's computation, not just the post-loop gate.
3. **Attention Injection** — route `ctx.prev_h` as cross-attention Query within each layer.

All three are §3.5 path 2 (deterministic architectural modifications, no gradient descent). They can be feature-gated behind `loop_stability_fix` and applied to the existing T-pass runtime.

**Cautionary tale:** CART (2606.01495) tried a *learned* LTI gate for loop stability — the ablation showed it was "individually vestigial" and CART lost to dense baselines by ~10%. This is evidence **against** the "learned gate" approach and **for** the "parameter-free architectural fix" approach.

---

## 1. Paper Core Findings

### 1.1 Fully Looped Transformer (2605.18797)

**Diagnosis (Section 3):** The paper monitors three quantities during early training (first 2000 steps): training loss, residual-state norm, and gradient L2 norm. Key observations:

- 12-loop LT **collapses**: loss plateaus, residual norm grows unbounded.
- 9-loop LT doesn't collapse but maintains higher loss than 6-loop LT.
- 12-loop FLT (the fix) remains stable: smooth loss reduction, small residual norms, stable gradients.

**Fix 1 — Fully Looped Architecture (FLA):**
In vanilla LT, the previous loop output `h_L^(t-1)` is passed only to the first layer of the next iteration. FLA makes it available to every layer:

```
h_l^(t) = f_θ(h_{l-1}^(t), h_L^(t-1)),  l = 1, ...,, L
```

This is a "shortcut connection in the recurrent dimension" — analogous to ResNet's residual connections but across loop iterations. It reduces the effective recurrent depth.

**Fix 2 — Attention Injection (AI):**
Instead of directly adding `h_L^(t-1)` to the residual flow, AI uses it as the **Query** in a cross-attention operation:

```
a_l^(t) = Attention(Q = WQ · h_L^(t-1), K = WK · z_l^(t), V = WV · z_l^(t))
```

where `z_l^(t)` is the current layer's pre-attention output. Key properties:
- The previous loop state determines **what to retrieve**, but the injected signal is constructed from the **current** value vectors.
- Softmax normalization bounds the injected signal magnitude.
- Reuses existing WQ, WK, WV projections — **zero new parameters**.
- Preserves KV-cache compatibility (K, V streams stay in standard form).

**Results (Table 1):**
- FLT trains stably up to 12 loops; vanilla LT collapses at 9 loops (base size).
- FLT improves average downstream performance by up to 13.2% at 6 loops (base size).
- FLT is the only variant whose downstream average consistently increases with loop count.
- Compatible with GQA, MLA, SWA, and Full Attention.

### 1.2 Readout Blind Spot (2606.24898)

**Diagnosis:** The paper asks: which state variables does cross-entropy actually control? Answer: only the variables exposed by the readout, not every variable active in the recurrent transition.

**The blind spot:** Scale-invariant readouts (RMSNorm, LayerNorm) hide radial scale:
- RMSNorm normalizes `x / ||x||_rms`, so the output is scale-invariant.
- Cross-entropy through RMSNorm cannot see the hidden-state norm.
- But pre-norm residual recurrence continues to carry and update that same scale.
- Result: per-loop loss makes early exits usable **without controlling recurrent scale**.

**Empirical evidence:** In 44M and 129M looped transformers without inter-loop normalization, per-loop cross-entropy through RMSNorm readouts drives final hidden-state norms into the **thousands or tens of thousands**.

**Fixes:**
1. **Scale-visible readouts** (architectural): use a readout where scale is visible to the loss.
2. **Explicit norm penalties** (training): add a norm penalty to the loss.
3. **Scale-removing recurrence** (architectural): normalize the hidden state between loop iterations — inter-loop normalization.

**Result:** Scale-controlled variants achieve **lower perplexity** at matched inference-depth operating points.

### 1.3 CART Cautionary Tale (2606.01495)

CART uses a "learned Linear Time-Invariant (LTI) gate" for loop stability, with spectral radius settling in [0.79, 0.83]. However:
- The ablation shows the LTI gate is "individually vestigial" — zero contribution.
- CART loses to parameter-matched dense baseline by 1-2% (stored-param) to ~10% (effective-param).
- Variable-R inference degrades on both sides of the trained R.

**Lesson:** The learned-gate approach doesn't work. Parameter-free architectural fixes (FLT, inter-loop norm) are the right path.

---

## 2. Distillation

### 2.1 What ships in our T-pass (forward_looped)

Current `forward_looped` (katgpt-rs/crates/katgpt-percepta/src/transformer.rs:268):
- Outer loop: `for tau in 0..loop_count`
- Saves `ctx.prev_h` at start of each loop (line 379)
- Per-layer: RMSNorm → QKV → attention → output proj + residual → MLP + residual
- Per-loop residual gate (lines 499-513): `h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)` (zero-init, first loop is identity)
- `prev_h` is ONLY used for the post-loop residual gate — NOT distributed to each layer

### 2.2 The gap (what's missing)

| Fix | Paper | Current status | §3.5 path |
|---|---|---|---|
| Inter-loop RMSNorm | Readout Blind Spot | ❌ NOT shipped — no normalization between loop iterations | Path 2 (deterministic) |
| Fully Looped Architecture | FLT 2605.18797 | ❌ NOT shipped — `prev_h` only used for post-loop gate, not fed to each layer | Path 2 (deterministic) |
| Attention Injection | FLT 2605.18797 | ❌ NOT shipped — no cross-attention with `prev_h` as Query | Path 2 (deterministic) |
| Scale-visible readout | Readout Blind Spot | ❌ NOT shipped — RMSNorm is scale-invariant | Path 2 (deterministic) |
| Per-loop residual gate | Already ours | ✅ Shipped (lines 499-513), zero-init | N/A |

### 2.3 Fusion — what novel combination produces?

**Fusion A (FLT × Readout Blind Spot):** Inter-loop RMSNorm + FLA + AI together address both instability sources:
- Inter-loop RMSNorm → controls residual explosion (scale-removing recurrence)
- FLA → distributes recurrent signal, reduces effective depth
- AI → bounds injected signal magnitude via softmax, suppresses oscillation

**Fusion B (FLT × existing residual gate):** Our per-loop residual gate (zero-init) can be combined with FLA. The gate provides a learned blend factor; FLA provides the structural connectivity. The gate's `ρ_τ` can be **deterministically set** to a decay factor (e.g., 0.8, matching CART's spectral radius target) instead of zero-init — a §3.5 path 2 improvement to the existing model-based path.

**Fusion C (FLT × Training-Free Loop 097):** The Training-Free Loop's damped Euler sub-stepping (`x_{k+1} = (1-1/K)·x_k + (1/K)·g(x_k)`) already controls residual growth by damping. FLT's inter-loop RMSNorm is a complementary mechanism — damping controls the update magnitude, RMSNorm controls the state magnitude. Combining both may be synergistic.

### 2.4 Latent-space reframing

The loop-stability problem has a latent-space interpretation:
- **Residual explosion** = the latent state's radial scale diverges. In HLA terms, the per-NPC latent state's norm grows unbounded — the "emotional intensity" explodes.
- **Inter-loop RMSNorm** = projecting the latent state back onto the unit sphere after each loop — a deterministic projection.
- **Attention Injection** = routing the previous loop's latent state through a sigmoid-bounded (softmax) gate — the injected signal is a convex combination of current value vectors, bounded by construction.
- **FLA** = making the recurrent latent state available to all computational stages, not just the first — reduces the "depth" of the latent state's propagation path.

This maps cleanly to the latent-to-latent preference: all operations stay in latent space, no decoding to tokens, no weight training.

---

## 3. Verdict

**Tier: GOAT** (provable gain over existing approach, but not a new class of capability — it's a stability improvement to an existing primitive)

**One-line reasoning:** Three parameter-free architectural fixes address the T-pass loop instability that Proposal 018 identifies as the P0 gap, closeable via §3.5 path 2 (deterministic construction) without riir-train.

**MOAT gate (katgpt-rs):** In-scope — fundamental inference primitive (looped transformer stability). The T-pass is a katgpt-rs runtime primitive (Plan 108). The fixes are generic architectural modifications applicable to any looped transformer.

**§3.5 modelless unblock check:**
1. **Freeze/thaw** — N/A (no weights to freeze; the fixes are architectural, not weight-based).
2. **Raw/lora reader-writer hot-swap** — N/A (no weight correction needed; the fixes are parameter-free).
3. **Latent-space correction** — ✅ YES. Inter-loop RMSNorm is a deterministic projection (normalize latent state between loops). Attention Injection is a sigmoid-bounded routing of the recurrent latent signal. Both are latent-space corrections.

All three fixes are modelless. No riir-train dependency.

**Defend-wrong PoC requirement (§3.6):** The claim "these fixes improve loop stability" is a **quality claim** that requires a PoC. The PoC must show:
1. Baseline (vanilla looped) exhibits residual explosion (norm growth).
2. Each fix reduces norm growth.
3. The fixes don't degrade output quality (logit divergence stays bounded).
4. Latency overhead is negligible (< 5% per loop iteration).

PoC → Plan 428.

---

## 4. Implementation Notes

### 4.1 Inter-loop RMSNorm (simplest fix)

Add after line 497 (end of inner layer loop, before residual gate):
```rust
// Inter-loop RMSNorm: normalize hidden state between loop iterations
// to control residual explosion (Readout Blind Spot, 2606.24898)
if tau > 0 {
    crate::types::rmsnorm(&mut ctx.x);
}
```

This is 1 line, zero allocation, parameter-free. The `rmsnorm` function already exists.

### 4.2 Fully Looped Architecture (medium fix)

Modify the inner layer loop to accept `prev_h` as a second input:
```rust
// FLA: add prev_h to each layer's residual (simplest implementation)
if tau > 0 {
    katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.prev_h[..n]);
}
```

This is the `FLT_res` variant (direct residual addition). The paper shows this alone doesn't fully prevent collapse but significantly reduces residual explosion. The full FLT uses Attention Injection instead.

### 4.3 Attention Injection (full fix)

This requires modifying the attention computation to use `prev_h` as Query when `tau > 0`:
```rust
if tau > 0 {
    // Cross-attention: Q = WQ · prev_h, K = WK · x, V = WV · x
    crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.prev_h, n, n);
} else {
    // Standard self-attention: Q = WQ · x
    crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
}
```

K and V stay the same (computed from current layer's `ctx.x`). Only Q changes. This is ~3 lines of change per layer, zero new parameters, reuses existing WQ projection.

### 4.4 Feature gate

All fixes behind `loop_stability_fix` feature flag (opt-in). When disabled, `forward_looped` is byte-identical to current behavior.

### 4.5 Model-based improvement (for riir-train, noted not implemented)

The "model-based" path improvements (noted for riir-train):
1. **Train with FLT fixes** — weights adapted to the looped architecture (the paper trains from scratch with FLT).
2. **Explicit norm penalty** — add `λ · ||h||^2` to the training loss (Readout Blind Spot's training fix).
3. **Stochastic loop count during training** — randomize loop count to improve OOD robustness (2606.29983).

> **Update 2026-08-30 (arXiv:2608.15062 GRT):** item 3 now has direct measured evidence + a simpler form — GRT's *uniform* depth sampling `r ~ U{1..R}` yields emergent early exit with ZERO auxiliary losses (92% accuracy at half depth), beating MoR/RRT/Ouro/heavy-tail-Poisson; also supplies the fixed-anchor evidence this note's FLT_res variant lacked (frozen prelude anchor beats the drifting `h(r-1)` anchor by 0.70 nats, trained-gate ablation) and a convex (norm-bounded) gate form vs our additive `ResidualGate`. Filed: `riir-train/.plans/364` (training recipe), `.issues/698` (runtime corollaries); full distillation `.research/519`.

These are → riir-train. The PoC tests only the modelless (inference-time) fixes.

---

## TL;DR

Two papers (2605.18797 + 2606.24898) diagnose looped transformer instability as **residual explosion** + **gradient oscillation**, caused by scale-invariant readouts hiding radial scale. Three **parameter-free architectural fixes** address this: (1) inter-loop RMSNorm, (2) Fully Looped Architecture (distribute prev loop state to all layers), (3) Attention Injection (route prev loop state through cross-attention Q). All are §3.5 path 2 (deterministic, modelless, no gradient descent). CART (2606.01495) is a cautionary tale — the learned-gate approach is vestigial. PoC (Plan 428) tests baseline vs each fix vs combined, measuring norm growth + output stability + latency. Model-based improvements (train with FLT, norm penalty) noted for riir-train.

> **Follow-up (synthesis):** Chen, Vegesna, Dahal, Wilson [arXiv:2609.19107 "How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents"] — the full boundary operator `BO(h,e) = Norm(h) + α·e` extends this note's inter-loop RMSNorm with input re-injection AND a before-readout (coda) application worth ~2×10⁻³ (their Table 4); the re-injection half is untested in our stack and is hazard-classed with Issue 568 until a BO-trained checkpoint exists (riir-train Plan 421 P4). Their exponent-vs-constant framing + KL-effective-depth instrumentation distill to `.research/592` + `.issues/898`.
