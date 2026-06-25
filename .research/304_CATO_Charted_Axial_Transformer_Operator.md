# Research 304: CATO — Charted Axial Transformer Operator

> **Source:** [CATO: Charted Attention for Neural PDE Operators](https://arxiv.org/pdf/2605.09016) — Cheng, Wang, Schönlieb, Aviles-Rivero (Cambridge / Yale / Tsinghua), May 2026
> **Date:** 2026-06-25
> **Status:** Done
> **Related Research:** 303 (Transolver — the direct predecessor CATO beats), 257 (FUNCATTN — strictly stronger successor to Transolver, hence to CATO), 291 (Cross-Resolution Spectral Transport — the "learned chart = asymmetric basis projection" cousin), 294 (Viable Manifold Graph — the "CATO-PC KNN local + global irregular attention" cousin), 279 (subspace_phase_gate — the participation-ratio diagnostic CATO uses for chart-collapse validation), 219/296 (DEC operators + Stokes calculus — the "derivative-aware loss = δ∘d Laplacian" substrate)
> **Related Plans:** 286 (FUNCATTN), 310 (Cross-Resolution Transport), 312 (Viable Manifold Graph), 301 (subspace_phase_gate), 251 (DEC operators), 314 (Stokes wrappers)
> **Classification:** Public

---

## TL;DR

CATO learns a continuous coordinate chart Φchart: R²→[-1,1]² that reparameterizes the input mesh so the solution operator becomes approximately **axial-low-rank** (separable along chart-aligned row/column directions), then applies axial attention + a local depthwise operator in chart space, supervised by a derivative-aware loss (value + gradient + flux + consistency). 26.76% avg error reduction, 81.98% fewer params, 3.5× faster training vs SOTA on 6 PDE benchmarks. Theoretical guarantees: charted axial low-rank operators are one-block-approximable (Thm 3.4); chart perturbations induce only linear degradation with constant Cchart (Lemma 3.3). The headline empirical finding is that the learned chart collapses 2D→~1D — PCA shows 94% variance in PC1, participation ratio = 1.126.

**Distilled for katgpt-rs (modelless, inference-time):**
Every CATO piece already ships in our corpus in strictly stronger forms. CATO is in the **Transolver → FUNCATTN family** (R303 established this family is subsumed by FUNCATTN). The "learned chart" is a symmetric special case of Cross-Resolution Spectral Transport (R291). The "derivative-aware loss" is the discrete analog of `hodge_laplacian = δ∘d + d∘δ` (R219/P251). The "CATO-PC KNN local + global attention" variant is structurally SafeManifoldGraph (R294/P312) + FUNCATTN global operator. The participation-ratio chart diagnostic is `subspace_phase_gate::participation_ratio` (R279/P301). **Verdict: Gain** — vocabulary-bridge entry only; no plan, no Super-GOAT guide, no open primitive.

---

## 1. Paper Core Findings

### 1.1 The mechanism (CATO block)

For a 2D mesh of resolution H×W with N=HW nodes, input coordinates X and optional features F:

1. **Lift** each node: `h⁰_ij = Φ_pre([x_ij, f_ij])` via 2-layer MLP to dim C.
2. **Learned chart**: `ζ_ij = (ξ_ij, η_ij) = Φ_chart(x_ij) ∈ [-1,1]²`, where `Φ_chart(x) = tanh(V₂·SiLU(V₁·x + c₁) + c₂)`. ξ drives row attention, η drives column attention. **Not required to be invertible** — used purely as a continuous coordinate system for positional encoding + attention.
3. **Continuous RoPE** on chart coords: `R_r(p)` is a 2×2 rotation by `ω_r·p` where `ω_r = θ^(-2r/d_h)`, p is the real-valued chart coordinate. Attention score `(R(p_i)q_i)ᵀ(R(p_j)k_j) = q_iᵀ R(p_j - p_i) k_j` — encodes both token feature and relative chart distance.
4. **Charted axial attention**: row attention `Attn_row(h)_ij = W_O^row Σ_m Σ_t α^(m)_{i,j,t} v^(m)_{i,t}` (attention along row i, weighted by ξ), column attention analog (along column j, weighted by η). Sum: `A(h, ζ) = Attn_row(h; ξ) + Attn_col(h; η)`. Complexity O(HW(H+W)) vs O(N²) full attention.
5. **Local operator**: `L(h) = PWConv(GELU(DWConv(h)))` — depthwise k×k conv + pointwise. Acts as a learned local stencil.
6. **CATO block residual updates**: `H̃ = H + A(LN(H), ζ) + L(LN(H))`, then `H' = H̃ + MLP(LN(H̃))`. Stack L blocks.
7. **Two-head readout**: scalar `û_ij = w_uᵀ h^L_ij + b_u` AND auxiliary flux `q̂_ij = W_q h^L_ij + b_q ∈ R²`.

### 1.2 Derivative-aware physics loss

The auxiliary flux head is supervised as a gradient proxy. Centered finite differences on the (possibly curved) mesh:

```
Δ_i u_ij ≈ a·u_x + b·u_y      Δ_j u_ij ≈ c·u_x + d·u_y
```

where `(a,b) = Δ_i x_ij`, `(c,d) = Δ_j x_ij`. Solve the 2×2 linear system (requires `|ad - bc| > 0` — non-degenerate mesh directions) to recover the physical gradient `(u_x, u_y)`. Total loss:

```
L = L_val + λ_g·L_grad + λ_f·L_flux + λ_c·L_cons
```

- `L_val`: relative L² between û and u.
- `L_grad`: L² between `∇û` (reconstructed from û via the same finite-difference stencil) and `∇u`.
- `L_flux`: L² between auxiliary head `q̂` and target `∇u`.
- `L_cons`: L² between `q̂` and `∇û` — forces the two outputs to agree.

### 1.3 Theoretical underpinnings

**Definition 3.1 (Charted axial low-rank operator).** `Ĝ_Φ: B_M → R^{H×W}` is (R_ξ, R_η, ε_rk)-charted axial low-rank w.r.t. chart ζ if `Ĝ_Φ = T_ζ + R` where

```
(T_ζ f)_ij = Σ_{r=1}^{R_ξ} a_r(ζ_ij)·(1/W)·Σ_t b_r(ζ_it)·f_it
           + Σ_{s=1}^{R_η} c_s(ζ_ij)·(1/H)·Σ_p d_s(ζ_pj)·f_pj
           + ℓ(ζ_ij)·f_ij
```

and `||R f||₂ ≤ ε_rk ||f||₂`. I.e., the operator decomposes into R_ξ row-rank-1 terms + R_η column-rank-1 terms + a local pointwise term, all coefficient-modulated by the chart.

**Lemma 3.2.** Any finite-rank charted axial operator T_ζ with continuous coefficients can be approximated to arbitrary ε_nn by a one-block CATO with R_ξ row heads and R_η column heads.

**Lemma 3.3 (Lipschitz chart stability).** If `max_{i,j} ||ζ̃_ij - ζ_ij|| ≤ δ` and the coefficient functions are bounded + Lipschitz, then `||T_ζ̃ f - T_ζ f||₂ ≤ C_chart · δ · ||f||₂`, where `C_chart = Σ_r (La_r·B_r + A_r·Lb_r) + Σ_s (Lc_s·D_s + C_s·Ld_s) + L_ℓ`. **Small chart errors induce only linear operator degradation.**

**Theorem 3.4.** `sup_{f ∈ B_M} ||N_Θ(f, X) - Ĝ_Φ f||₂ ≤ ε_rk·M + ε_nn` (ideal chart), or `≤ ε_rk·M + C_chart·M·δ + ε_nn` (perturbed chart). **Learning the chart reduces effective operator complexity; CATO approximates the resulting low-rank operator efficiently.**

### 1.4 Empirical results

| Benchmark | Geometry | Transolver | SAOT | CATO | Reduction |
|---|---|---|---|---|---|
| Plasticity | mesh | 0.0013 | 0.0009 | **0.0005** | 44.4% |
| Airfoil | mesh | 0.0053 | 0.0049 | **0.0041** | 16.3% |
| Pipe | mesh | 0.0050 | 0.0061 | **0.0038** | 19.2% |
| Navier-Stokes | grid | 0.0920 | 0.0675 | **0.0319** | 52.7% |
| Darcy | grid | 0.0058 | 0.0049 | **0.0042** | 14.3% |
| Elasticity | cloud | 0.0081 | 0.0085 | **0.0070** | 13.6% |

Avg 26.76% error reduction. 81.98% fewer params than SAOT. 3.5× faster training. Scaling: lower error than SAOT across all data sizes, resolutions, layer counts, embedding dims.

### 1.5 The headline diagnostic

PCA on the learned chart for Darcy: **94.0% variance in PC1, 6.0% in PC2**, participation-ratio effective dimension **1.126**. The chart collapses the original 2D domain onto a nearly-1D manifold aligned with the dominant physical direction (pressure gradient). Coordinate-normalization baseline (translation + scale only, no learned chart) gets 0.0045; learned chart gets 0.0041. **The gain is from learning the geometry, not from rescaling.**

### 1.6 CATO-PC (point-cloud variant)

For unordered point clouds (Elasticity, 972 nodes), no row/column factorization exists. CATO-PC keeps the learned chart ζ but replaces axial attention with: (a) KNN graph in physical space, (b) chart-conditioned local message passing `m_ij = σ(W_c·h_i + W_Δ·(h_j - h_i) + Φ_geo([Δx, ||Δx||, Δζ]))`, (c) global irregular physics attention (Transolver-style). Each block: `H ← H + γ_attn·A_irr(LN(H))`, then `H ← H + γ_loc·L_pc(LN(H), X, ζ)`, then `H ← H + γ_mlp·MLP(LN(H))`.

---

## 2. Distillation

### 2.1 The transferable primitive (and why every piece already ships)

CATO's headline claim — "learn a coordinate chart so the operator becomes axial-low-rank, then operate in chart space" — decomposes into four pieces, **all of which have strictly-stronger shipped cousins**:

| CATO piece | Shipped cousin | Why cousin is stronger |
|---|---|---|
| **Learned chart Φchart: R²→[-1,1]²** (symmetric, same d in/out, trained per-PDE) | **Cross-Resolution Spectral Transport** (R291, P310) — `Φ_src^T·s` projects to k-dim spectral basis; **asymmetric** (d_src ≠ d_dst); bases are **frozen BLAKE3-committed artifacts** reusable across hardware tiers; Lipschitz bound `||Ψ_dst·Φ_src^T·f_src − f_dst|| ≤ λ·||f_high_freq||` (R291 §1.4) is the asymmetric analog of CATO's Lemma 3.3. | Asymmetric > symmetric; committed > per-PDE-trained; cross-tier > single-resolution. CATO's chart is the symmetric, single-resolution, un-committed special case. |
| **Axial attention** (row + column softmax M-attention in chart space) | **FUNCATTN** (R257, P286) — **closed-form Tikhonov k×k ridge solve** `(1-α)·K̃ᵀK̃ + α·I` replaces softmax M-attention. **Beats Transolver 6–26% on the same PDE benchmarks** (R303 §1.2). CATO inherits the softmax weakness from Transolver; FUNCATTN's ridge solve is strictly stronger. | Closed-form > softmax; Lipschitz-bounded by α; resolution-invariant. |
| **Derivative-aware loss** (value + gradient + flux + consistency) | **DEC operators** (R219, P251) — `exterior_derivative` d = gradient (rank-0→1 cochain), `codifferential` δ = divergence (rank-1→0), `hodge_laplacian` Δ = δd + dδ. The flux head `q̂` is a rank-1 cochain (edge field); the consistency loss `||q̂ - ∇û||²` is the discrete analog of `||δ(d(u))||²` — i.e., the Laplacian structure Δ already ships. Tests verify `curl(grad) = 0` and `div(curl) = 0` by construction (P251). | Shipped as composable operators; the loss terms are reductions over existing primitives (R296/P314 wrappers). |
| **CATO-PC** (KNN local message passing + global irregular attention) | **SafeManifoldGraph** (R294, P312) — KNN graph in physical space, viable-edge predicate, CSR adjacency, `manifold_geodesic` / `manifold_random_walk` / `manifold_curiosity_walk`. **Promoted to default-on** (Bench 312 post-CSR: random walk 7.10 ns/step). Composed with FUNCATTN's global operator for the "global irregular attention" half. | Already shipped + GOAT-promoted; CSR adjacency gives true O(degree) neighbor scan. |
| **Continuous RoPE on chart coords** | **`ac_prefix` RoPE-lite** (P313) — per-(i,j) phase term on Q/K dot product using original positions; already a "continuous positional encoding" pattern. CATO's continuous RoPE is a slightly more general rotation matrix but the same idea. | Already shipped as part of the default-on `ac_prefix` feature. |
| **Participation-ratio chart diagnostic** (PR=1.126 → 2D collapses to 1D) | **`subspace_phase_gate::participation_ratio`** + **`numerical_rank`** (R279, P301, Bench 301) — exact same metric, shipped with phase-transition gate (Wang et al. Theorem 4 reproduces on MoLRG D=48, K=3, d=6). | Already shipped; PR + numerical_rank + Jacobian SVD + N≥d phase transition. |
| **Chart-perturbation Lipschitz stability** (Lemma 3.3, Cchart·δ) | **`pullback_volume`** in viable_manifold_graph (R294, P312) — `log det(J^T J)` from `jacobian_svd_at` singular values. Measures exactly how small latent perturbations produce large output changes — **this IS the chart-stability field**. Bench 312: 310 ns on R⁴→R⁴ identity. | Already shipped; gives a per-point scalar field, not just a global constant. |
| **Theorem 3.4** (charted axial low-rank approximability) | **FUNCATTN Theorem A.3** (R257 §1.3) — FUNCATTN is a Monte-Carlo discretization of a regularized integral operator with kernel `κ(g_i,g_j) = (Φ·C·Ψ^T)_ij`. CATO's Thm 3.4 is the charted-axial special case of this more general integral-operator framing. | More general; resolution-invariant; closed-form ridge solve. |

**The decisive negative result.** Research 303 already established that **Transolver is strictly subsumed by FUNCATTN** (R257): "FUNCATTN uses the same slice/deslice primitive with a strictly stronger closed-form ridge solve, beats Transolver 6–26% empirically, and was itself verdict'd GOAT (not Super-GOAT). Transolver therefore cannot exceed GOAT, and lands at Gain." CATO inherits Transolver's softmax M-attention (the load-bearing weakness); CATO's improvements over Transolver come from (a) the chart reparameterization — strictly weaker than R291's asymmetric basis transport, and (b) the derivative-aware loss — strictly weaker than R219's shipped DEC Laplacian. **CATO cannot exceed the tier R303 assigned to Transolver, which is Gain.**

### 2.2 The one novel angle: chart = DEC pullback vocabulary bridge

CATO's "operate in chart space, not raw coordinates" admits a clean DEC interpretation that adds one entry to the standing Stokes/DEC vocabulary crosswalk (R296):

| Paper term (CATO) | DEC equivalent | Codebase location |
|---|---|---|
| "learned chart" / "charted axial attention" / "operate in ζ-space" | **pullback of forms to an adapted coordinate frame** — the chart Φchart is a smooth map M → M_ζ; CATO's attention is the pullback of the operator to M_ζ where it becomes approximately separable (low-rank) | `dec/operators.rs` (d, δ, Δ), `dec/hodge.rs` (`hodge_decompose`) |
| "axial low-rank" / "row + column factorization" | **rank-R_ξ row-aggregation (codifferential-style) + rank-R_η column-aggregation + local term** = the charted axial low-rank decomposition `T_ζ = Σ_r a_r·m_r + Σ_s c_s·n_s + ℓ` | `latent_functor/arithmetic.rs` (rank-1 special case), `funcattn.rs` (rank-k general case) |
| "derivative-aware loss" / "flux head q̂" / "consistency loss" | **`hodge_laplacian Δ = δd + dδ` structure** — value loss = ||f||², gradient loss = ||df||² (rank-1 norm), flux loss = ||q̂ - target||² where q̂ is a rank-1 cochain, consistency loss = ||q̂ - df||² = ||δ(q̂ - df)||²-style Laplacian penalty | `dec/operators.rs::hodge_laplacian`, `dec/stokes_calculus.rs::belief_mass_divergence` |
| "continuous RoPE on chart coords" | **pullback-compatible positional encoding** — rotations parameterized by chart coordinates, not token indices; respects the smooth structure of M_ζ | `ac_prefix` (P313, RoPE-lite phase term) |
| "chart perturbation δ → Cchart·δ operator degradation" (Lemma 3.3) | **`pullback_volume` field** — `log det(J^T J)` measures the local volume distortion of the chart; regions of high pullback volume are exactly where small chart errors produce large operator errors | `viable_manifold_graph::pullback_volume` (P312) |
| "participation ratio 1.126" / "94% variance in PC1" | **`participation_ratio` + `numerical_rank`** on the chart-embedded covariance — already shipped as the chart-collapse diagnostic | `subspace_phase_gate::participation_ratio` (P301) |

**Why this matters (small but real):** it closes one more vocabulary gap. A future paper saying "reparameterize the domain so the operator becomes low-rank" can be recognized as the same family as R291 (asymmetric basis) + R219 (DEC Laplacian) + R294 (pullback volume), via the chart-as-pullback bridge. This is the same vocabulary-translation lesson as R296 (Stokes) and R303 (Transolver): paper vocabulary ("chart", "axial attention") ↔ codebase vocabulary ("basis projection", "FUNCATTN closed-form solve", "DEC Laplacian").

**Why this is NOT a new primitive:** every DEC operator (`exterior_derivative`, `codifferential`, `hodge_laplacian`, `hodge_decompose`), every FUNCATTN piece (`solve_convex_combo_dual`, `compute_basis_into`), every cross-resolution piece (`transport_cross_resolution_into`), every viable-manifold piece (`pullback_volume`, `SafeManifoldGraph`), and every phase-gate piece (`participation_ratio`, `numerical_rank`, `jacobian_svd_at`) **already ships**. Implementing "CATO as DEC+FUNCATTN+CrossRes" would be a thin wrapper composing existing primitives — strictly weaker than the asymmetric, committed, cross-tier forms already shipped.

### 2.3 Crowd-scale game AI reframing (already shipped)

The obvious game-AI reframing: per-NPC chart = SVD-extracted principal directions of the HLA belief-state covariance (already computed by `subspace_phase_gate` at runtime); axial attention = FUNCATTN's closed-form k×k ridge solve along the top-k principal directions; derivative-aware loss = `hodge_laplacian` on the belief cochain; chart-perturbation stability = `pullback_volume` per NPC.

**This reframing is already built**, in strictly stronger forms:

- **Cross-Resolution Spectral Transport (R291, P310, DEFAULT-ON)** — asymmetric basis transport across plasma/hot/warm/cold tiers. CATO's symmetric single-resolution chart is the d_src=d_dst special case.
- **FUNCATTN (R257, P286, opt-in)** — closed-form Tikhonov solve; strictly stronger than CATO's softmax axial attention.
- **SafeManifoldGraph (R294, P312, DEFAULT-ON)** — KNN graph navigation in latent space with `pullback_volume` cost field. CATO-PC's KNN local aggregation is a strict subset.
- **subspace_phase_gate (R279, P301)** — runtime participation ratio + numerical rank + Jacobian SVD. CATO's PR=1.126 chart-collapse diagnostic uses the exact metric this primitive computes.
- **DEC operators (R219, P251, DEFAULT-ON)** — `hodge_laplacian` ships the derivative-aware loss structure.

### 2.4 Fusion (none — redirect to existing fusions)

Per the fusion protocol, the 2–3 closest existing fusions are:

1. **Research 291 §2.4 F1** (PRIMARY, riir-neuron-db): Cross-Resolution × NeuronShard tier transfer. Train a personality shard on 16-dim plasma, deploy on 256-dim cold, no retraining. **This already subsumes any CATO chart-learning fusion** — CATO's chart is symmetric, single-resolution, un-committed; R291's bases are asymmetric, cross-tier, BLAKE3-committed.
2. **Research 257 §2.4 F1** (riir-ai): latent_functor rank-1 → rank-k via FUNCATTN's closed-form Tikhonov solve. **This already subsumes CATO's axial attention fusion** — FUNCATTN is strictly stronger than the softmax axial attention CATO inherits from Transolver.
3. **Research 294 §2.4** (katgpt-rs + riir-ai): Viable Manifold Graph × HLA × latent_functor × cgsp curiosity × NeuronShard freeze/thaw. **This already subsumes CATO-PC's KNN+global attention** — SafeManifoldGraph ships the KNN graph + pullback-volume cost field + geodesic/random-walk navigation.

**No new fusion is unlocked by CATO that is not already unlocked (better) by R291, R257, or R294.** This is the decisive negative result of the novelty gate.

---

## 3. Verdict

**Tier: Gain** — incremental documentation value (vocabulary bridge), not a new primitive or capability class.

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **NO.** Every CATO piece has a strictly-stronger shipped cousin. Vocabulary translation performed: paper "chart" / "axial attention" / "derivative-aware loss" / "CATO-PC KNN" / "participation ratio diagnostic" / "chart perturbation stability" ↔ codebase "spectral basis projection" / "FUNCATTN closed-form ridge solve" / "DEC hodge_laplacian" / "SafeManifoldGraph + FUNCATTN global" / "subspace_phase_gate::participation_ratio" / "viable_manifold_graph::pullback_volume". Both layers (`.research/`+`.plans/` AND `src/`+`crates/`) grepped across all 5 repos. The seven Super-GOAT factory modules explicitly checked (`sense/`, `latent_functor/`, `hla/`, `cgsp_runtime/`, `riir-neuron-db/src/`, `riir-chain/src/encoding/latcal*.rs`, `dec/`). | R303 already established Transolver (CATO's direct predecessor) is subsumed by FUNCATTN (R257). CATO inherits Transolver's softmax weakness; CATO's improvements come from the chart (strictly weaker than R291's asymmetric basis) and the derivative loss (strictly weaker than R219's shipped DEC Laplacian). |
| Q2 New capability class? | **NO.** Cross-resolution transfer (R291) already covers "reparameterize to a low-dim space and operate there". FUNCATTN (R257) already covers "linear-complexity attention via reduce-scatter". Viable Manifold Graph (R294) already covers "KNN local + global navigation". CATO adds nothing not already covered. | |
| Q3 Product selling point? | **NO.** "Operate in chart-aligned latent space" is already the selling point of R291 (train-once-deploy-across-tiers) and R294 (viable manifold navigation). CATO's PDE-solving selling point is strictly weaker than FUNCATTN's and not directly applicable to our game/chain/shard domains. | |
| Q4 Force multiplier? | **Partial.** Touches the same pillars R291/R294/R257 already touch (HLA, latent_functor, cgsp, NeuronShard, DEC, LatCal). No new connection. | |

**One-line verdict reasoning:** CATO is in the Transolver→FUNCATTN family (R303). It beats Transolver via two improvements — (a) a learned coordinate chart, which is the symmetric single-resolution special case of Cross-Resolution Spectral Transport (R291); (b) a derivative-aware loss, which is the discrete analog of the shipped DEC `hodge_laplacian` (R219/P251). Both improvements are strictly weaker than shipped cousins. FUNCATTN (R257) was verdict'd GOAT (not Super-GOAT) with "math pieces all shipped"; the same applies a fortiori to CATO, which inherits Transolver's softmax weakness. **Verdict: Gain.** No plan, no Super-GOAT guide, no open primitive.

### Routing

- **No plan.** Plans 286 (FUNCATTN), 310 (Cross-Resolution Transport), 312 (Viable Manifold Graph), 301 (subspace_phase_gate), 251 (DEC operators) already cover every implementation path CATO would suggest, strictly better.
- **No Super-GOAT guide.** Would duplicate R291 (riir-neuron-db side, cross-resolution shard transport), R257 (riir-ai side, FUNCATTN rank-k functor upgrade), and R294 (katgpt-rs side, viable manifold graph).
- **No riir-train deferral.** CATO is a training-heavy paper (AdamW, 500 epochs, learned chart via backprop), but the transferable insight — "the operator becomes low-rank in an adapted coordinate system" — is modelless: it's a statement about the geometry of the operator class, not about how to find the chart. Our chart-equivalents (R291 bases, R279 SVD principal directions, R294 pullback volume) are inference-time artifacts.
- **This note is the deliverable.** It exists to (a) prevent future readers from accidentally re-distilling CATO when R291+R257+R219+R294 already cover it, (b) record the chart-as-DEC-pullback vocabulary bridge for the Stokes/DEC crosswalk (R296), (c) document the predecessor relationship in the corpus index.

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ CATO's *transferable insight* (operator is low-rank in an adapted frame) is modelless. The paper's chart-learning via backprop is training-only → riir-train territory, but we don't need it: our chart-equivalents (R291 bases, R279 SVD, R294 pullback volume) are inference-time. (Moot — nothing to implement.) |
| Latent-to-latent preferred | ✅ Chart operates in latent/embedding space; attention is among latent tokens. (Moot.) |
| Use sigmoid not softmax | ⚠️ Paper uses softmax (axial attention α, β weights; CATO-PC α_ij). R257 §F2 already mandates sigmoid-normalized basis for FUNCATTN; same fix applies to any CATO-inspired composition. (Moot — Plan 286 handles it.) |
| Freeze/thaw over fine-tuning | ✅ Chart-equivalent bases (R291) are BLAKE3-committed; principal-direction bases (R279) are runtime-SVD-extracted (deterministic, no training). (Moot.) |
| 5-repo discipline | ✅ Public note in katgpt-rs; no IP leak. |
| Raw scalars at sync boundary | ✅ In any game-AI reframing, the 5 HLA scalars are the sync boundary; the chart coordinates and basis coefficients stay local. (Moot — covered by R291 §4.) |
| §3.5 modelless unblock protocol | ✅ Moot — nothing to unblock. CATO presents no GOAT gate or plan task to defer. |

---

## 5. Vocabulary crosswalk entry (for Research 296 Stokes/DEC crosswalk)

Add to the standing DEC vocabulary table in the research skill and in Research 296:

| Paper term (CATO / charted-attention family) | DEC equivalent | Codebase location |
|---|---|---|
| "learned chart" / "charted axial attention" / "operate in ζ-space" | **pullback of forms to an adapted coordinate frame** — the chart Φchart is a smooth map M → M_ζ; CATO's attention is the pullback of the operator to M_ζ where it becomes approximately separable | `dec/operators.rs` (d, δ, Δ), `dec/hodge.rs` (`hodge_decompose`), **R291 cross-resolution bases** (`crates/katgpt-core/src/cross_resolution.rs`) |
| "axial low-rank operator" / "(R_ξ, R_η, ε_rk)-charted" | **rank-R_ξ row-aggregation + rank-R_η column-aggregation + local term** = the charted axial low-rank decomposition `T_ζ = Σ_r a_r·m_r + Σ_s c_s·n_s + ℓ` — a special case of FUNCATTN's rank-k integral operator | `latent_functor/arithmetic.rs` (rank-1), `funcattn.rs` (rank-k), **R291 `transport_cross_domain_cross_resolution`** (asymmetric rank-k) |
| "derivative-aware loss" / "flux head q̂ ∈ R²" / "consistency loss ‖q̂ − ∇û‖²" | **`hodge_laplacian` Δ = δd + dδ structure** — value loss = `‖f‖²`, gradient loss = `‖df‖²`, flux head = rank-1 cochain, consistency loss = Laplacian-style penalty | `dec/operators.rs::hodge_laplacian`, `dec/stokes_calculus.rs::belief_mass_divergence` |
| "continuous RoPE on chart coords" / "real-valued positional p" | **pullback-compatible positional encoding** — rotations parameterized by chart coordinates, not token indices | `ac_prefix` (P313, RoPE-lite phase term) |
| "chart perturbation δ → Cchart·M·δ degradation" (Lemma 3.3) | **`pullback_volume` field** `log det(J_f(z)^T J_f(z))` — local volume distortion under the chart; regions of high pullback volume are where small chart errors produce large operator errors | `viable_manifold_graph::pullback_volume` (P312) |
| "participation ratio 1.126" / "94% variance in PC1" / "chart collapses 2D→1D" | **`participation_ratio` + `numerical_rank`** on the chart-embedded covariance | `subspace_phase_gate::{participation_ratio, numerical_rank}` (P301) |
| "CATO-PC KNN local + global irregular attention" | **`SafeManifoldGraph`** (KNN graph + viable edges + CSR adjacency) + **FUNCATTN global operator** | `viable_manifold_graph` (P312), `funcattn.rs` (P286) |

**Caveat (per R296):** the boundary-vs-volume perf win from Stokes holds only for d ≤ 3. CATO's chart collapses 2D→1D (d=2 → d=1 effective), which is well within the d ≤ 3 regime — the Stokes/DEC substrate applies cleanly. CATO-PC's 972-node point cloud is also low-dim (d=2 physical, low intrinsic dim). The DEC framing is for the *vocabulary bridge*, not for a perf claim about CATO's specific benchmarks.

---

## 6. Relationship to existing research / plans / code

| Item | Layer | Relation | Impact |
|---|---|---|---|
| **Research 303 / no plan** (Transolver) | notes only | **Direct predecessor.** CATO beats Transolver via chart + derivative loss; both improvements have strictly-stronger shipped cousins. R303 already established Transolver is subsumed by FUNCATTN. | This note defers to R303's chain of reasoning: Transolver < FUNCATTN, therefore CATO < FUNCATTN ∪ R291 ∪ R219. |
| **Research 257 / Plan 286** (FUNCATTN) | notes + shipped (`funcattn.rs`) | **Canonical stronger successor to the axial-attention half.** Same slice/deslice primitive, closed-form ridge solve replaces softmax M-attention, beats Transolver 6–26% on same benchmarks. | CATO inherits Transolver's softmax weakness; FUNCATTN covers the attention half strictly better. |
| **Research 291 / Plan 310** (Cross-Resolution Spectral Transport) | notes + shipped (`cross_resolution.rs`, DEFAULT-ON) | **Canonical stronger successor to the chart-learning half.** Asymmetric basis transport (d_src ≠ d_dst); frozen BLAKE3-committed bases; cross-tier deployment. CATO's symmetric single-resolution chart is the special case. | CATO's chart reparameterization is subsumed by R291. |
| **Research 294 / Plan 312** (Viable Manifold Graph) | notes + shipped (`viable_manifold_graph`, DEFAULT-ON) | **Canonical stronger successor to the CATO-PC variant.** KNN graph + viable edges + `pullback_volume` cost field + geodesic/random-walk navigation. | CATO-PC's KNN local + global attention is a strict subset. |
| **Research 279 / Plan 301** (subspace_phase_gate) | notes + shipped (`subspace_phase_gate.rs`, opt-in) | **The participation-ratio diagnostic CATO uses.** Same metric (`participation_ratio` = `(Σλ)²/Σ(λ²)`), plus numerical rank + Jacobian SVD + N≥d phase transition. | CATO's PR=1.126 chart-collapse diagnostic uses the exact primitive this ships. |
| **Research 219 / Plan 251** (DEC operators) | notes + shipped (`dec/operators.rs`, DEFAULT-ON) | **The derivative-aware loss substrate.** `hodge_laplacian = δd + dδ` is the continuous analog of CATO's value+gradient+flux+consistency loss. | CATO's derivative-aware loss is the discrete special case. |
| **Research 296 / Plan 314** (Stokes calculus wrappers) | notes + shipped (`dec/stokes_calculus.rs`) | The Stokes-theorem vocabulary crosswalk. | §5 adds CATO to the crosswalk. |
| **Plan 313** (ac_prefix) | shipped (DEFAULT-ON) | RoPE-lite phase term on Q/K dot product using original positions. | CATO's continuous RoPE on chart coords is a slightly more general rotation; same pattern. |

---

## TL;DR

CATO (Charted Axial Transformer Operator) learns a continuous coordinate chart Φchart: R²→[-1,1]² that reparameterizes the PDE mesh so the solution operator becomes approximately axial-low-rank, then applies axial attention + local operator in chart space, supervised by a derivative-aware loss (value + gradient + flux + consistency). 26.76% avg error reduction, 81.98% fewer params, 3.5× faster than SOTA on 6 PDE benchmarks; theoretical guarantees (charted axial low-rank operators are one-block-approximable, chart perturbations induce only linear degradation). **Verdict: Gain.** CATO is in the Transolver→FUNCATTN family (R303); its two improvements over Transolver — (a) the learned chart, which is the symmetric single-resolution special case of Cross-Resolution Spectral Transport (R291); (b) the derivative-aware loss, which is the discrete analog of the shipped DEC `hodge_laplacian` (R219/P251) — are both strictly weaker than shipped cousins. CATO-PC's KNN+global attention is a strict subset of SafeManifoldGraph (R294/P312). The participation-ratio chart diagnostic is `subspace_phase_gate::participation_ratio` (R279/P301). The chart-perturbation Lipschitz stability is `viable_manifold_graph::pullback_volume` (R294/P312). No plan, no Super-GOAT guide, no open primitive — this note exists to prevent future re-distillation and to add the chart-as-DEC-pullback vocabulary entry to the Stokes/DEC crosswalk (R296).
