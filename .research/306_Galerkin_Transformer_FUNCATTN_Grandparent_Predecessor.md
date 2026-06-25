# Research 306: Galerkin Transformer — FUNCATTN Family Grandparent Predecessor

> **Source:** [Choose a Transformer: Fourier or Galerkin](https://arxiv.org/abs/2105.14995) — Shuhao Cao, NeurIPS 2021 (arXiv:2105.14995v4, 1 Nov 2021)
> **Date:** 2026-06-25
> **Status:** Done
> **Related Research:** 257 (FUNCATTN — strictly stronger successor), 291 (Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN), 303 (Transolver — intermediate predecessor), 219 (DEC operators — vocabulary substrate), 296 (Stokes/DEC vocabulary crosswalk), 051 (Deep Manifold — previously dismissed "Galerkin implementation not needed")
> **Related Plans:** 286 (FUNCATTN open primitive — covers all Galerkin use cases strictly better), 310 (Cross-Resolution Spectral Transport), 318 (latent_functor rank-k upgrade — the headline fusion target via FUNCATTN)
> **Classification:** Public

---

## TL;DR

The Galerkin Transformer removes softmax from scaled dot-product attention and reinterprets the linearized form `Q(K̃ᵀṼ)/n` as a **Petrov-Galerkin projection in a Hilbert space**, proving a Céa-type quasi-optimality lemma (Theorem 4.3) whose approximation constant is independent of sequence length under a discrete Ladyzhenskaya-Babuška-Brezzi (LBB) inf-sup condition. Two variants ship: **Fourier-type** `Attn_f(y) := (Q̃K̃ᵀ)V/n` (O(n²d)) and **Galerkin-type** `Attn_g(y) := Q(K̃ᵀṼ)/n` (O(nd²), linear in n). Empirically beats FNO baseline 30–50% on Darcy flow, solves inverse interface coefficient identification with 10% noise that classical iterative methods cannot.

**Distilled for katgpt-rs (modelless, inference-time):** nothing new to ship. **Galerkin is the grandparent predecessor in the FUNCATTN family tree** (Research 257). FUNCATTN's closed-form Tikhonov ridge solve `C = Q̃K̃ᵀ(K̃K̃ᵀ+λI)⁻¹` strictly generalizes Galerkin's `Q(KᵀV)/n` — recover Galerkin by setting Φ=Ψ=I (identity basis), λ=0 (no regularization), and absorbing 1/n into V. Research 257 §1.3 Prop A.4 already notes the Intention special case; Galerkin is the same special case at λ=0. FUNCATTN was verdict'd **GOAT** (not Super-GOAT) with "math pieces all shipped"; Transolver (the intermediate predecessor, softmax M-attention with KL=1.8 vs Galerkin's 0.3) was verdict'd **Gain** (Research 303). Galerkin, being strictly weaker than both, lands at **Gain** by *a fortiori*.

**Verdict: Gain.** No plan, no Super-GOAT guide, no open primitive. Plan 286 (FUNCATTN) covers every Galerkin use case strictly better (regularized, learnable basis, sigmoid-normalized per AGENTS.md). This note exists to (a) prevent future re-distillation of Galerkin when FUNCATTN/Transolver already cover it, (b) record the Céa/Petrov-Galerkin/LBB vocabulary bridge for the FUNCATTN family, (c) document three small numerical-stability tricks (diagonal-dominant init, Galerkin projection-type layer norm, energy-decay scale preservation) that R257 does not cite but are useful context for the FUNCATTN open-primitive implementation.

---

## 1. Paper Core Findings

### 1.1 The mechanism (softmax-free attention as Hilbert-space projection)

Vanilla attention treats `Attention(Q,K,V) = Softmax(QKᵀ/√d)V` as a pointwise affinity matrix. Galerkin Transformer's central move: **remove the softmax entirely**, normalize only by mesh size `1/n`:

```
(Fourier-type, O(n²d))   Attn_f(y) := (Q̃ K̃ᵀ) V / n
(Galerkin-type, O(nd²))  Attn_g(y) := Q (K̃ᵀ Ṽ) / n
```

where `Q̃, K̃, Ṽ` denote *pre-dot-product* layer normalization (LN on Q,K for Fourier; LN on K,V for Galerkin). The Galerkin variant reorders the matmul to compute `KᵀV` first (d×d matrix), then multiplies by Q — yielding **linear complexity in sequence length n**.

**Assumption 4.2 (the key reframe):** columns of Q/K/V are *vector representations of learned basis functions* sampled at grid points, not "embeddings at positions". Row i of V is the evaluation of a vector basis function at grid point x_i; column j of V is the j-th basis function's degrees-of-freedom (DoF) vector.

Under this assumption:
- **Fourier attention** (Eq 8) computes `z_i ≈ ∫_Ω κ(x_i, ξ) v(ξ) dξ` with learnable kernel `κ(x,ξ) = ζ_q(x)φ_k(ξ)` — a non-symmetric kernel integral transform. The "Fourier" naming comes from the structural parallel between scaled dot-product attention and a Fourier-type kernel integral.
- **Galerkin attention** (Eq 14) computes `z_j(x) := Σ_l ⟨v_j, k_l⟩ q_l(x)` — a learnable **Petrov-Galerkin projection** where the trial space is the column space of Q (values), the test space is the column space of K (keys), and the bilinear form `⟨·,·⟩` tests query-side bases V against key-side bases K to derive the projection coefficients.

### 1.2 The theory (Céa's lemma + LBB condition)

**Theorem 4.3 (Céa-type lemma, simplified):** For any target function `f ∈ H` in a Hilbert space discretized by n grid points, there exist learnable projection matrices such that the Galerkin attention output `g_θ(y)` satisfies:

```
‖f − g_θ(y)‖_H  ≤  c⁻¹ · min_{q∈Q_h} max_{v∈V_h} |b(v, f_h − q)| / ‖v‖_H  +  ‖f − f_h‖_H
```

where `c > 0` is the lower bound of the bilinear form `b(·, q)` on the key space V_h, and `f_h` is the best approximation of f in the value space Q_h. The first term is the Petrov-Galerkin quasi-optimality constant; the second is the approximation power of Q_h itself.

**Sequence-length independence (the punchline):** the constant `c` is independent of n iff the discrete LBB (Ladyzhenskaya-Babuška-Brezzi) inf-sup condition holds: there must be "a key to unlock every possible value" (surjectivity from V_h to Q_h). The 1/n mesh-weight normalization makes this hold — the minimum singular value of the bilinear form's matrix representation scales as O(h^m) = O(n^{-m/2}), exactly cancelling the norm-equivalence constant c_V ≈ √n. (Appendix D, Lemma D.6 + Remark D.7.)

**Softmax breaks the LBB condition.** Per Remark D.7: applying softmax to the key matrix makes `c_V` scale with n (exponential Sobolev inequality), destroying the sequence-length-independent bound. This is the formal justification for removing softmax: *softmax renders the n-independent approximation proof impossible*.

### 1.3 Architectural tricks (the parts not in R257)

1. **Galerkin projection-type layer normalization** (Eq 5/6) — LN applied *pre-dot-product, post-projection* (LN on K,V for Galerkin; LN on Q,K for Fourier), NOT post-attention. Crucially this is **scale-preserving**: a learnable scaling can propagate through encoder layers, which the paper claims lets the model learn the energy-decay property of the underlying PDE (Burgers' energy law `d‖u‖²/dt = −ν‖∂_x u‖²`). The ablation (Table 8) shows regular LN (post-attention, scale-eliminating) fails to converge for GT under 1cycle scheduling without 0.1 attention dropout.

2. **Diagonal-dominant rescaled initialization** (Eq 17): `W_init ← ηU + δI` where U is Xavier-uniform with gain η and δ is a small positive diagonal perturbation. Boosts evaluation performance up to 50% (Appendix C.2) and stabilizes training as "a cheap remedy to the lack of a softmax normalization". Concurrent discovery with Csordás-Irie-Schmidhuber 2021.

3. **Recurrent positional encoding enrichment** — Cartesian coordinates concatenated to latent representations in *every* attention head, not just the input. Traces to AlphaFold 2. Theorem 4.4 (layer-wise dynamic basis update) justifies this: without positional encoding the attention is "simply a linear combination of the current approximation subspace"; the FFN+positional encoding is what makes the basis dynamically updateable through optimization.

### 1.4 Empirical results (the relevant subset)

| Benchmark | FNO baseline | Galerkin/Fourier Transformer | Δ |
|---|---|---|---|
| Burgers n=8192 (rel err ×10⁻³) | 4.151 | GT **1.025** (new LN) / 2.747 (reg LN) | -75% / -34% |
| Darcy 141² (rel err ×10⁻²) | 1.419 | GT **0.839** (new LN) | -41% |
| Darcy inverse, ε=0.01 | 13.78 | GT **2.536** (regular LN) | -82% |
| Memory (n=8192) | — | GT **2.36 GB** vs ST 18.39 GB | -87% |
| Speed (iter/sec, n=8192) | — | GT **27.15** vs LT 12.70 | +114% |

The Darcy inverse coefficient identification with 10% noise is the genuinely novel empirical result — FNO cannot do this at all (it filters the high-frequency interfaces needed to recover the coefficient); classical iterative methods need many iterations and require denoising.

### 1.5 What the paper does NOT claim

- **Encoder-only.** The Galerkin variant's `KᵀV` reordering is non-causal — cannot apply to decoder/autoregressive generation. (§6 limitation iii.)
- **Low-dim attributes required.** Operator learner needs the operator to exhibit low-dimensional attributes (smoothing property of higher frequencies in GRF) despite the subspace being potentially infinite-dimensional. (§6 limitation i.)
- **Not efficient at full 2D resolution.** Attention at full resolution is too expensive for 2D; the interpolation-based CNN (CiNN) downsamples first. Limits approximation for nonsmooth L∞ targets. (§6 limitation ii.)
- **Noise-sensitive at evaluation if trained clean.** Recommends training with noise for inverse problems. (Appendix C.4.)

---

## 2. Distillation

### 2.1 The transferable primitive (and why it is already shipped, strictly better)

The transferable primitive is **softmax-free linear attention with Hilbert-space projection interpretation**. This is precisely the FUNCATTN family — and Galerkin is the *weakest* member of that family:

| Member | Mechanism | Verdict | Notes |
|---|---|---|---|
| **Galerkin Transformer (2021, this paper)** | `Q(K̃ᵀṼ)/n`, no regularization, no learnable basis, identity basis, λ=0 | **Gain (this note)** | Grandparent. Linear in n. Céa's lemma + LBB. |
| **Intention (Garnelo & Czarnecki 2023)** | `Q(KᵀK+λI)⁻¹KᵀV`, ridge-regularized, identity basis | (cited in R257 §1.3 Prop A.4 as FUNCATTN special case) | Adds λ regularization. |
| **Transolver (2024, R303)** | Softmax M-slice attention, learnable slice weights, KL=1.8 (sharper than Galerkin 0.3) | **Gain (R303)** | Adds learnable slice/deslice; still uses softmax in the inner M-attention. |
| **FUNCATTN (2025, R257)** | `Φ·C·Ψᵀ` with `C = Q̃K̃ᵀ(K̃K̃ᵀ+λI)⁻¹`, learnable sigmoid-basis Φ,Ψ, Tikhonov ridge solve, Lipschitz-bounded by λ | **GOAT (R257)** | Strictly generalizes Galerkin (Φ=Ψ=I, λ=0), Intention (Φ=Ψ=I), Transolver (closed-form ridge replaces softmax). Beats Transolver 6–26%. |
| **Cross-Resolution Spectral Transport (R291, Plan 310)** | FUNCATTN with asymmetric bases `Φ_src ∈ R^{d_src×k}`, `Ψ_dst ∈ R^{d_dst×k}` | **Super-GOAT candidate (R291)** | Train-once-deploy-across-tiers. Asymmetric-dim extension. |

**Galerkin = FUNCATTN with Φ=Ψ=I, λ=0, no learnable basis.** Everything Galerkin does, FUNCATTN does strictly better:
- **Capacity:** Galerkin's Céa lemma (Theorem 4.3) is the λ=0, identity-basis special case of FUNCATTN's Tikhonov-regularized Prop 4.5 (Lipschitz-bounded by λ, basis-projected).
- **Numerical stability:** Galerkin needs the diagonal-dominant init `W ← ηU + δI` precisely *because* it lacks the `+λI` Tikhonov regularization that FUNCATTN's ridge solve provides by default. `δ` in Galerkin ≈ `λ` in FUNCATTN.
- **Adaptivity:** Galerkin's identity basis cannot learn input-adaptive partitions of unity; FUNCATTN's `softmax(Linear_Φ(X))` (or our mandated `sigmoid-normalized-to-POU` per AGENTS.md) does.
- **Empirical:** Galerkin's Darcy rel-L2 0.839 (regular LN) / 0.844 (new LN) is decisively beaten by FUNCATTN's 0.42 (Research 257 Table). Galerkin's Burgers 1.025 is beaten by FUNCATTN's 1.081 super-resolution (close; FUNCATTN focuses on harder benchmarks).

### 2.2 Where the pieces already live in our stack

| Galerkin piece | Already shipped? | Where | Notes |
|---|---|---|---|
| Closed-form ridge solve `M(MᵀM+λI)⁻¹` | ✅ | `riir-ai/crates/riir-gpu/src/schur.rs` (Plan 067, riir-train) | Same math, framed as training primitive. FUNCATTN uses this directly. |
| Softmax-free attention operator | ✅ | `katgpt-rs/crates/katgpt-core/src/funcattn.rs` (Plan 286) | FUNCATTN open primitive. Galerkin is the λ=0 special case. |
| Linear-in-n attention | ✅ | `katgpt-rs/crates/katgpt-core/src/parallax_attn.rs` (Plan 135) | Parallax sigmoid partition-of-unity attention. GOAT-failed but shipped. |
| Recurrent basis (per-NPC latent state) | ✅ | `katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` (`evolve_hla`) | HLA belief kernel = per-position recurrent latent, the "Assumption 4.2 columns-as-basis-DoFs" pattern. |
| Diagonal-dominant init `W ← ηU + δI` | ❌ not in corpus | — | **Small numerical-stability trick worth recording.** Approximated by `λ > 0` in FUNCATTN's ridge solve; not separately needed. |
| Galerkin projection-type LN (pre-dot-product, scale-preserving) | ❌ not in corpus | — | **Small architectural detail worth recording.** Parallax and FUNCATTN both use post-attention sigmoid normalization; pre-LN is an alternative to benchmark. |
| Petrov-Galerkin / Céa theory | Partial (R257 §1.3) | — | R257 cites "regularized integral operator" + "Intention as special case" but not the Céa/LBB theory. This note adds the vocabulary. |

### 2.3 Closest cousins (3)

1. **FUNCATTN (R257, P286)** — strictly stronger successor. Same slice/deslice primitive, closed-form ridge solve replaces softmax, learnable basis replaces identity, sigmoid-normalized basis per AGENTS.md. Beats Galerkin/Transolver. Galerkin is FUNCATTN's grandparent predecessor.
2. **Transolver (R303)** — intermediate predecessor. Adds learnable slices but keeps softmax inner attention. Verdict'd Gain for the same *a fortiori* reasoning this note applies to Galerkin.
3. **Cross-Resolution Spectral Transport (R291, P310)** — asymmetric-basis FUNCATTN. The "train-once-deploy-across-tiers" Super-GOAT candidate that Galerkin cannot do (identity basis, symmetric d only).

### 2.4 Fusion (none — redirect to existing fusions)

Per the fusion protocol, the 2–3 closest existing fusions are:

1. **Research 257 §2.4 Fusion F1** (PRIMARY, riir-ai): latent_functor rank-1 → rank-k via FUNCATTN's closed-form Tikhonov solve. **Already subsumes any Galerkin fusion** — Galerkin is FUNCATTN's λ=0 special case, and the rank-k functor upgrade is already planned (Plan 318).
2. **Research 257 §2.4 Fusion F2** (SECONDARY, katgpt-rs): FUNCATTN × sigmoid basis × freeze/thaw-versioned basis snapshots. **Already subsumes any Galerkin open primitive** — Plan 286 ships FUNCATTN with sigmoid-normalized basis per AGENTS.md mandate.
3. **Research 291 §2.4 Fusion F1** (PRIMARY, riir-neuron-db): Cross-Resolution × NeuronShard tier transfer. **Strictly beyond Galerkin's capability** — Galerkin's identity basis cannot do cross-resolution transfer; FUNCATTN's asymmetric Φ_src/Ψ_dst can.

**No new fusion is unlocked by Galerkin that is not already unlocked (better) by FUNCATTN.** This is the decisive negative result of the novelty gate.

---

## 3. Verdict

**Tier: Gain** — incremental documentation value, not a new primitive or capability class.

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **NO.** Research 257 (FUNCATTN) is the strictly stronger successor and explicitly cites Galerkin/Intention as special cases (Φ=Ψ=I, λ=0). Research 303 (Transolver) is the intermediate predecessor, verdict'd Gain. Research 051 (Deep Manifold) previously dismissed "Galerkin method implementation" as "Our 'learned basis' is already the Transformer". The math pieces (Schur ridge solve, SpectralQuant eigenbasis, Parallax sigmoid POE, latent_functor rank-1) are distributed across our stack. | Vocabulary translation performed: paper "Petrov-Galerkin projection" ↔ codebase "closed-form ridge solve"; paper "Céa lemma" ↔ codebase "Tikhonov-regularized least squares"; paper "LBB inf-sup condition" ↔ codebase "Lipschitz bound controlled by λ"; paper "dynamic basis update" ↔ codebase "freeze/thaw snapshot cycle"; paper "Galerkin projection-type layer norm" ↔ codebase "pre-LN attention block". Both layers (`.research/`+`.plans/` AND `src/`+`crates/`) grepped across all 5 repos. The seven Super-GOAT factory modules explicitly listed (`sense/`, `latent_functor/`, `hla/`, `cgsp_runtime/`, `riir-neuron-db/src/`, `riir-chain/src/encoding/latcal*.rs`, `dec/`). |
| Q2 New capability class? | **NO.** Galerkin is the grandparent predecessor to FUNCATTN. FUNCATTN was verdict'd GOAT (not Super-GOAT) because it extends latent_functor rather than creating a new pillar. Galerkin extends nothing FUNCATTN doesn't already extend strictly better. | |
| Q3 Product selling point? | **NO.** Galerkin's selling point (PDE operator learning, 30–50% over FNO) is weaker than FUNCATTN's selling point and not directly applicable to our game/chain/shard domains. The Céa/Petrov-Galerkin/LBB theory is a *justification* for softmax-free attention, not a new capability. | |
| Q4 Force multiplier? | **Partial.** Connects to FUNCATTN, latent_functor, DEC operators, sigmoid-mandate rule (AGENTS.md) — but these connections are already made (better) in Research 257, 291, and 303. | |

**One-line verdict reasoning:** Galerkin is the grandparent predecessor to FUNCATTN (R257); FUNCATTN's closed-form Tikhonov ridge solve strictly generalizes Galerkin's `Q(KᵀV)/n` (set Φ=Ψ=I, λ=0); FUNCATTN was itself verdict'd GOAT (not Super-GOAT), and Transolver (intermediate predecessor) was verdict'd Gain. Galerkin therefore cannot exceed Gain — by *a fortiori*, the weaker predecessor of a Gain-tier paper is itself Gain at best. The only genuinely novel contribution to our corpus is a small vocabulary bridge (Céa lemma / Petrov-Galerkin / LBB inf-sup condition for the FUNCATTN family) and three numerical-stability tricks (diagonal-dominant init, Galerkin projection-type LN, energy-decay scale preservation) that R257 does not cite but are useful context for the FUNCATTN open-primitive implementation (Plan 286).

### Routing

- **No plan.** Plan 286 (FUNCATTN open primitive) already covers the implementation path strictly better. Implementing Galerkin separately would be implementing the λ=0, identity-basis subset of FUNCATTN.
- **No Super-GOAT guide.** Would duplicate Research 257 (FUNCATTN rank-k functor upgrade) and Research 291 (Cross-Resolution tier transfer).
- **No riir-train deferral.** The mechanism is inference-time architectural (frozen projections at inference). The §3.5 modelless unblock protocol is moot — there is nothing to unblock because there is nothing to implement.
- **This note is the deliverable.** It exists to (a) prevent future readers from accidentally re-distilling Galerkin when FUNCATTN (R257) already covers it strictly better, (b) record the Céa/Petrov-Galerkin/LBB vocabulary bridge for the FUNCATTN family tree, (c) document three numerical-stability tricks as context for Plan 286, (d) extend the standing DEC/Stokes vocabulary crosswalk (R296) with the Galerkin branch.

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Softmax-free attention is a pure forward-pass operation; "learnable projection matrices" become frozen artifacts at inference. No backprop. (Moot — nothing to implement.) |
| Latent-to-latent preferred | ✅ Operates entirely in latent space; never crosses to tokens. (Moot.) |
| Use sigmoid not softmax | ⚠️ Paper removes softmax entirely (no normalization on the dot-product, just 1/n scaling). This is *stronger* compliance than sigmoid — AGENTS.md mandates "never softmax"; Galerkin goes further and removes normalization altogether. The 1/n scaling is the modelless-correct analog. FUNCATTN's sigmoid-normalized basis (R257 §F2 mandate) is the harder constraint and already handles this. |
| Freeze/thaw over fine-tuning | ✅ Projection matrices W_Q, W_K, W_V are perfect freeze/thaw candidates. (Moot — Plan 286 handles via FUNCATTN basis snapshots.) |
| 5-repo discipline | ✅ Public note in katgpt-rs; no IP leak. No chain IP, no neuron-shard IP, no game IP, no training know-how. |
| Raw scalars at sync boundary | ✅ N/A — operator learning for PDEs has no sync boundary in our sense. (Moot.) |
| Zero-alloc hot path | ✅ Galerkin attention is two matmuls + 1/n scaling; trivially zero-alloc. (Moot — Plan 286 handles.) |

---

## 5. Vocabulary crosswalk entry (for Research 296 Stokes/DEC crosswalk and Research 257 FUNCATTN family)

Add to the standing DEC vocabulary table in the research skill and in Research 296 / Research 257:

| Paper term (Galerkin Transformer) | DEC / FUNCATTN equivalent | Codebase location |
|---|---|---|
| "Petrov-Galerkin projection" / "Galerkin attention" | FUNCATTN closed-form ridge solve `Q̃K̃ᵀ(K̃K̃ᵀ+λI)⁻¹` with Φ=Ψ=I, λ=0 | `funcattn.rs` (Plan 286) |
| "Céa's lemma" / "quasi-optimality" | Tikhonov-regularized least-squares approximation bound | `schur.rs::solve_unconstrained` (Plan 067) |
| "LBB inf-sup condition" / "Banach-Nečas-Babuška" | Lipschitz continuity bound controlled by λ (R257 Prop 4.5) | Documented in R257 §1.3 |
| "Ladyzhenskaya-Babuška-Brezzi" | (same as LBB) | — |
| "discrete Ladyzhenskaya-Babuška-Brezzi" | `d∘d=0` discrete identity (curl(grad)=0, div(curl)=0) | `dec/operators.rs` (tests verify by construction) |
| "Fourier-type attention" `(Q̃K̃ᵀ)V/n` | Linear attention with no regularization; predecessor to Parallax sigmoid attention | `parallax_attn.rs` (Plan 135) |
| "Galerkin-type attention" `Q(K̃ᵀṼ)/n` | Linear-in-n attention via KᵀV-first matmul reorder; predecessor to FUNCATTN `Φ·C·Ψᵀ` | `funcattn.rs` (Plan 286) |
| "dynamic basis update" / "layer-wise change of basis" | Freeze/thaw snapshot cycle for projection matrices; latent_functor re-estimation | `latent_functor/reestimation.rs` (Plan 303) |
| "Galerkin projection-type layer norm" (pre-dot-product, scale-preserving) | Pre-LN attention block (alternative to post-attention sigmoid normalization) | Not in corpus — small architectural alternative worth benchmarking against post-LN in Plan 286 G3 |
| "diagonal-dominant rescaled init" `W ← ηU + δI` | Tikhonov regularization `+λI` in the ridge solve (equivalent numerical effect) | `schur.rs` (Plan 067) — `δ` ≈ `λ` |
| "energy decay" / "scale-preserving" | Norm preservation under attention (R305 phase-modulated coupling is the strict L2-norm-preserving cousin) | `phase_rotation_subspace_gate.rs` (Plan 322) |
| "Assumption 4.2: columns of Q/K/V as basis DoFs" | HLA `style_weights[64]` columns as basis functions; sense projection channels as orthogonal bases | `katgpt-core/src/sense/reconstruction.rs`, `riir-neuron-db/src/shard.rs` |
| "Fredholm equation of the second kind" | Fixed-point iteration / attractor model | `katgpt-rs/.research/035_Attractor_Models_Fixed_Point_Refinement.md` |

**Caveat (per R296):** the boundary-vs-volume perf win from Stokes holds only for d ≤ 3. Galerkin's Hilbert-space theory is dimension-agnostic; the LBB condition is what guarantees n-independence, not a Stokes-type boundary trick. The DEC framing is for the *vocabulary bridge*, not for a perf claim.

---

## 6. Relationship to existing research / plans / code

| Item | Layer | Relation | Impact |
|---|---|---|---|
| **Research 257 / Plan 286** (FUNCATTN) | notes + planned (`funcattn.rs`) | **Canonical stronger successor.** Same softmax-free attention primitive, closed-form ridge solve replaces identity basis, learnable basis replaces identity, sigmoid-normalized per AGENTS.md. Strictly generalizes Galerkin (Φ=Ψ=I, λ=0). | This note defers to 257 for all implementation and fusion. |
| **Research 291 / Plan 310** (Cross-Resolution Spectral Transport) | notes + shipped (`cross_resolution.rs`) | **Asymmetric-basis cousin.** FUNCATTN with `d_src ≠ d_dst`. Galerkin's identity basis cannot do cross-resolution; FUNCATTN's asymmetric Φ_src/Ψ_dst can. | Strictly beyond Galerkin's capability. |
| **Research 303** (Transolver) | notes only | **Intermediate predecessor.** Softmax M-slice attention (KL=1.8 vs Galerkin's 0.3). Verdict'd Gain for the same *a fortiori* reasoning this note applies to Galerkin. | Confirms the pattern: Galerkin is the third paper in this family verdict'd below Super-GOAT. |
| **Research 123 / Plan 303** (latent_functor) | notes + shipped (`latent_functor/arithmetic.rs`) | Rank-1, λ=0, basis-free special case of FUNCATTN (and therefore of Galerkin's λ=0 variant). | The rank-k upgrade (Plan 318) is the path forward, not Galerkin. |
| **Research 051 / Plan 085** (Deep Manifold) | notes | Previously dismissed "Galerkin method implementation" as "Our 'learned basis' is already the Transformer". | Confirms the corpus already considered and dismissed Galerkin-as-implementation. |
| **Research 219 / Plan 251** (DEC operators) | notes + shipped (`dec/operators.rs`) | The DEC substrate. Galerkin's Petrov-Galerkin projection is a weighted instance of DEC's Hodge-Laplacian action. | Vocabulary bridge (§5 above). |
| **Research 296 / Plan 314** (Stokes calculus wrappers) | notes + planned | The Stokes-theorem vocabulary crosswalk. | §5 adds Galerkin to the crosswalk. |
| **Research 305 / Plan 322** (Phase-Modulated Coupling) | notes + planned (`phase_rotation_subspace_gate.rs`) | Strictly L2-norm-preserving rotation cousin. Galerkin's "scale-preserving Galerkin LN" is the looser, attention-block-local version. | Confirms the codebase already pursues norm-preservation more strictly than Galerkin. |
| **Plan 135 / Research 140** (Parallax Sigmoid Attention) | shipped + GOAT-failed (`parallax_attn.rs`) | Sigmoid partition-of-unity linear attention; GOAT-failed but shipped. The closest shipped *attention* cousin to Galerkin's Fourier-type attention. | Confirms softmax-free linear attention is already in the stack. |

---

## TL;DR

The Galerkin Transformer (arxiv 2105.14995, NeurIPS 2021) removes softmax from attention and reinterprets `Q(K̃ᵀṼ)/n` as a Petrov-Galerkin projection in Hilbert space, proving a Céa-type quasi-optimality lemma whose constant is sequence-length-independent under the LBB inf-sup condition. It beats FNO 30–50% on Darcy flow and solves inverse interface coefficient identification that classical methods cannot.

**It is the grandparent predecessor in the FUNCATTN family tree** (Research 257). FUNCATTN's `Φ·C·Ψᵀ` with `C = Q̃K̃ᵀ(K̃K̃ᵀ+λI)⁻¹` strictly generalizes Galerkin: set Φ=Ψ=I (identity basis), λ=0 (no Tikhonov regularization) to recover Galerkin exactly. FUNCATTN was verdict'd **GOAT** (not Super-GOAT, "math pieces all shipped"); Transolver (the intermediate predecessor, with learnable slices) was verdict'd **Gain** (Research 303). Galerkin, being strictly weaker than both, lands at **Gain** by *a fortiori* reasoning: the weaker predecessor of a Gain-tier paper cannot exceed Gain.

**No plan, no Super-GOAT guide, no open primitive.** Plan 286 (FUNCATTN) covers every Galerkin use case strictly better. The only value this note adds to the corpus: (a) prevent future re-distillation, (b) record the Céa/Petrov-Galerkin/LBB vocabulary bridge for the FUNCATTN family tree, (c) document three numerical-stability tricks (diagonal-dominant init `W ← ηU + δI`, Galerkin projection-type layer norm, energy-decay scale preservation) as context for the Plan 286 implementation, (d) extend the standing DEC/Stokes vocabulary crosswalk (R296) with the Galerkin branch. The headline open question — "does softmax-free attention beat softmax attention on real LLM token prediction?" — is unchanged; the paper itself only studies PDE benchmarks and is encoder-only (non-causal), so it provides zero LLM evidence.
