# Research 303: Transolver — Physics-Attention as the Predecessor to FUNCATTN

> **Source:** [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366) — Wu, Luo, Wang, Wang, Long (Tsinghua / BNRist), ICML 2024
> **Date:** 2026-06-25
> **Status:** Done
> **Related Research:** 257 (FUNCATTN — the canonical, strictly stronger successor), 302 (FAME — per-entity MoE blend cousin), 246 (Manifold Power Iteration MoE Router — sibling GOAT), 123 (latent_functor rank-1 special case), 219/296 (DEC operators + Stokes calculus)
> **Related Plans:** 286 (FUNCATTN open primitive — Gain, ships the math), 318 (latent_functor rank-k upgrade — GOAT, the riir-ai primary value), 321 (FAME CommittedFieldBlend — Super-GOAT, today)
> **Classification:** Public

---

## TL;DR

Transolver's **Physics-Attention** reduces quadratic attention over N mesh points to M ≪ N **learned physics-aware slice tokens** via a three-step reduce-scatter: (1) `w = Softmax(Project(x))` per-point slice weights, (2) `z_j = Σ w_{i,j} x_i / Σ w_{i,j}` aggregate to M centroid tokens, (3) attention among M tokens, (4) `x'_i = Σ_j w_{i,j} z'_j` deslice back to N points. Linear complexity O(NMC + M²C), M constant.

**This paper is the *predecessor* to FUNCATTN (Research 257)** — same slice/deslice primitive, but Transolver uses softmax M-attention where FUNCATTN uses a closed-form Tikhonov k×k ridge solve. FUNCATTN beats Transolver 6–26% on the *same* PDE benchmarks (Elasticity 0.50 vs 0.64, Darcy 0.42 vs 0.57, AirfRANS OOD CL 23.4 vs 32.2). Research 257 was verdict'd **GOAT** (not Super-GOAT) with reasoning "math pieces all shipped" — the same reasoning applies a fortiori to Transolver.

**Distilled for katgpt-rs (modelless, inference-time):** the slice/deslice primitive is **already conceptually covered** by three stronger cousins:
- **Research 257 (FUNCATTN)** — strictly stronger successor. Plan 286 ships the open primitive; Plan 318 ships the riir-ai rank-k functor upgrade. Transolver is a strict subset.
- **Research 302 (FAME)** — per-entity fixed MoE blend, verdict'd Super-GOAT today. Covers the "M slices = K archetype fields, weights computed once and frozen" angle as a stronger commitment-tier primitive.
- **latent_functor rank-1 (Research 123 / Plan 303)** — the rank-1, basis-free special case already shipped in `riir-engine/src/latent_functor/arithmetic.rs`.

**The only genuinely novel angle Transolver adds to the corpus** (not in Research 257/302/123) is a small reframing: **slice = DEC codifferential δ (rank-1 → rank-0 aggregation), deslice = DEC exterior_derivative d (rank-0 → rank-1 coboundary broadcast)**. The DEC operators (`codifferential`, `exterior_derivative`, `hodge_decompose`) already ship in `crates/katgpt-core/src/dec/operators.rs` (Plan 251). This is a vocabulary bridge, not a new primitive.

**Verdict: Gain.** No plan, no Super-GOAT guide, no open primitive. Document the predecessor relationship + the DEC reframing so future readers don't accidentally re-distill Transolver when FUNCATTN (257) already covers it strictly better.

---

## 1. Paper Core Findings

### 1.1 The mechanism (Physics-Attention)

Given N mesh points with features `x ∈ R^{N×C}`:

1. **Slice weights** (Eq. 1): `w_i = Softmax(Project(x_i)) ∈ R^{1×M}` per point. `Project` is a point-wise linear layer (or 3×3 conv for structured grids). Softmax along the M-dim makes slice assignment low-entropy.
2. **Physics-aware tokens** (Eq. 2): `z_j = (Σ_i w_{i,j} · x_i) / (Σ_i w_{i,j}) ∈ R^{1×C}` — normalized weighted aggregation. M tokens total.
3. **Attention among tokens** (Eq. 3): `q,k,v = Linear(z)`, `z' = Softmax(qk^T/√C) · v`. Standard multi-head attention on M tokens.
4. **Deslice** (Eq. 4): `x'_i = Σ_j w_{i,j} · z'_j` — broadcast token updates back to mesh points via the same slice weights.

**Complexity:** O(NMC + M²C). M constant (32–256), M ≪ N (N up to 168,921). Linear in N.

**Theorem 3.4:** Physics-Attention is equivalent to a learnable integral operator `G(u)(g*) = ∫_Ω κ(g*,ξ) u(ξ) dξ` on the slice domain Ω_s (diffeomorphic to Ω). The slice/deslice is the change-of-variables; the M-attention is the discrete kernel on Ω_s.

### 1.2 Empirical results (the relevant subset)

State-of-the-art on 6 PDE benchmarks + 2 industrial design tasks (Shape-Net Car, AirfRANS). 22% average relative gain over previous SOTA. Notable: Transolver **beats geo-FNO badly** on unstructured meshes (Shape-Net Car rel-L2 0.021 vs 0.167) because the learnable slices adapt to geometry where Fourier bases assume periodic boundaries.

### 1.3 What Transolver does NOT claim

- **Mesh points are not the primitive unit.** The whole point is that N mesh points are a finite sampling of an underlying continuous physics space; M slices capture the intrinsic physical states.
- **Slices are not spatial partitions.** Remark 3.2 explicitly contrasts with FEM computation areas: slices can group spatially-distant-but-physically-similar points (e.g., windshield + license plate + headlight all in the "front drag" slice).
- **Softmax is load-bearing.** Ablation Table 4 shows M=1 (single global pool) collapses; the slice weights must be low-entropy to produce informative tokens. KL-divergence of learned attention vs uniform: Galerkin (linear attention on mesh points) ≈ 0.3, Transolver ≈ 1.8 (Table 5) — sharper, more informative.

---

## 2. Distillation

### 2.1 The transferable primitive (and why it is already shipped)

The transferable primitive is **soft-cluster reduce-scatter attention**: project N items to M soft-cluster centroids via learned sigmoid/softmax weights, attend among M centroids, scatter updates back via the same weights. Linear in N.

**This primitive is already in our stack, in three strictly-stronger forms:**

| Cousin | File / Plan | What it has that Transolver doesn't |
|---|---|---|
| **FUNCATTN (R257)** | `crates/katgpt-core/src/funcattn.rs` (Plan 286, Gain-tier pending) | Closed-form Tikhonov k×k ridge solve `(1-α)·K̃ᵀK̃ + α·I` replacing softmax M-attention. Beats Transolver 6–26% on the same benchmarks. Lipschitz-bounded by α. Resolution-invariant. |
| **FAME CommittedFieldBlend (R302)** | `crates/katgpt-core/src/committed_field_blend.rs` (Plan 321, Super-GOAT, today) | Per-ENTITY FIXED MoE blend computed ONCE from trajectory summary then frozen. M slices = K archetype operator fields, weights π committed via BLAKE3, sampling-invariant. |
| **latent_functor rank-1 (R123)** | `riir-engine/src/latent_functor/arithmetic.rs` (Plan 303, shipped) | The k=1, λ=0, basis-free special case. `extract_functor: f = mean_k(target_k - source_k)`, apply via `out = source + f`. Rank-1 operator between two latent spaces. |

Research 257 §2.2 explicitly notes: *"Math pieces all shipped (Schur ridge solve, SpectralQuant eigenbasis, Parallax sigmoid partition-of-unity, latent_functor rank-1 operator). The combination as an attention operator + as a rank-k functor upgrade is novel."* The same is true for Transolver — except Transolver is **strictly weaker** than FUNCATTN (softmax M-attention vs closed-form ridge solve), so there is no combination Transolver enables that FUNCATTN doesn't already enable better.

### 2.2 The one novel angle: DEC codifferential/exterior_derivative reframing

Research 257 frames the slice/deslice as functional-map basis projection (ΦᵀQ, Φ·C·Ṽ). Research 302 frames it as archetype blend (Σ_k π_k · f_k). **Neither frames it as a DEC operator pair.** Transolver's mechanism admits a clean DEC interpretation that is worth recording:

- **Slice** (Eq. 2: `z_j = Σ_i w_{i,j} x_i / Σ_i w_{i,j}`) is a **codifferential-style aggregation** — it takes a 0-cochain (per-mesh-point field, rank 0) and produces M scalar values per channel (a rank-(-1)-like "super-cell" cochain). In DEC terms, `codifferential` (`dec/operators.rs:126`) maps rank k → rank k−1 by summing boundary contributions with signs. Transolver's slice is the same shape: rank 0 (points) → aggregate to M meta-cells, weighted by `w_{i,j}` instead of ±1 boundary signs.
- **Deslice** (Eq. 4: `x'_i = Σ_j w_{i,j} z'_j`) is an **exterior-derivative-style coboundary broadcast** — rank 0 meta-cells → rank 0 points, broadcasting each meta-cell's value to its member points. `exterior_derivative` (`dec/operators.rs:48`) maps rank k → rank k+1 by accumulating coboundary entries.
- **Slice + deslice round-trip with M-attention in between** is structurally a **δ-then-d composition with an M-dim inner operator** — i.e., a low-rank approximation of the Hodge-Laplacian's action, where the M-attention plays the role of the harmonic projector on the slice domain.

**Why this matters (small but real):** it gives a vocabulary bridge from Transolver/FUNCATTN to the DEC substrate (Plan 251, Research 219/296). A future Stokes-calculus paper that says "discretization-invariant integral operator via boundary flux" can be recognized as the same family as Transolver's Theorem 3.4, via the DEC identity `d∘d=0`. This is the same vocabulary-translation lesson as Research 296 (Stokes): paper vocabulary ("slice", "physics-aware token") ↔ codebase vocabulary ("codifferential aggregation", "coboundary broadcast"). Adding Transolver to the standing vocabulary crosswalk closes one more gap.

**Why this is NOT a new primitive:** the DEC operators already ship (`codifferential_into`, `exterior_derivative_into`, `hodge_decompose`). The Transolver mechanism is a *specific weighted instance* of these operators with `w_{i,j}` replacing the ±1 boundary signs. Implementing "Transolver as DEC" would be a thin wrapper calling existing operators with learned weight matrices — Plan 286 (FUNCATTN) already covers the implementation path, and FUNCATTN's closed-form ridge solve is strictly better than Transolver's softmax M-attention.

### 2.3 Crowd-scale game AI reframing (already partially shipped)

The obvious game-AI reframing: M = emotional-role slices per zone (fleeing / stalking / curious / panicked / ...), N NPCs softly assigned via sigmoid projection of HLA state, M role-centroids attend among themselves, updates scatter back. O(N·M) crowd-coherence instead of O(N²).

**This reframing is already partially built:**
- `crates/katgpt-core/src/latent_steering.rs::apply_field_to_crowd` (Plan 290, shipped) — applies a direction-vector field to a crowd of latent states in a single zero-alloc SAXPY sweep.
- `crates/riir-games/src/crowd_mcgs/` (Plan 298, promoted to default 2026-06-17) — crowd-scale MCGS retrieval with `RetrievalScratch` dense-vec path.
- `crates/riir-games/src/crowd_coherence_bench/` (Plan 331, 2026-06-23) — drives N NPCs through T ticks of HLA leaky integrator / latent_functor chain.
- FAME (R302, today) — per-entity committed archetype blend is the **commitment-tier** version of "M role slices per NPC"; the crowd version would be `apply_field_to_crowd` over a library of K archetype fields.

Transolver's specific contribution (soft-cluster slice attention among M role centroids) would be one more operator in this already-crowded stack. FAME's per-entity committed blend is structurally stronger because it commits the weights once (sampling-invariant under fog-of-war), where Transolver's softmax weights are recomputed per forward pass (not commitment-tier).

### 2.4 Fusion (none — redirect to existing fusions)

Per the fusion protocol, the 2–3 closest existing fusions are:

1. **Research 257 §2.4 Fusion F1** (PRIMARY, riir-ai): latent_functor rank-1 → rank-k via FUNCATTN's closed-form Tikhonov solve. **This already subsumes any Transolver fusion** — FUNCATTN is strictly stronger and the fusion is already planned (Plan 318).
2. **Research 302 §2.3 Fusion table** (Super-GOAT, today): per-NPC committed archetype blend × KARC × PersonalityWeightedComposition × NeuronShard × LatCal × DEC. **This already subsumes the crowd-scale Transolver reframing** — FAME's per-entity MoE blend IS the commitment-tier version of Transolver's per-forward-pass slice blend.
3. **Research 296 / Plan 314** (Stokes calculus wrappers): `belief_mass_divergence`, `boundary_flux_mass`, `line_integral`. **The DEC reframing in §2.2 above is a small additional vocabulary entry** in this crosswalk, not a new fusion.

**No new fusion is unlocked by Transolver that is not already unlocked (better) by FUNCATTN or FAME.** This is the decisive negative result of the novelty gate.

---

## 3. Verdict

**Tier: Gain** — incremental documentation value, not a new primitive or capability class.

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **NO.** Research 257 (FUNCATTN) is the strictly stronger successor and explicitly beats Transolver on the same benchmarks. Research 302 (FAME) covers the per-entity MoE blend angle. latent_functor rank-1 (R123) covers the basis-free special case. The math is distributed across Schur/SpectralQuant/Parallax/latent_functor/DEC. | Vocabulary translation performed: paper "Physics-Attention" / "slice" / "physics-aware token" / "deslice" ↔ codebase "soft-cluster attention" / "codifferential aggregation" / "centroid token" / "coboundary broadcast". Both layers (`.research/` + `.plans/` AND `src/`/`crates/`) grepped across all 5 repos. The seven Super-GOAT factory modules explicitly listed (`sense/`, `latent_functor/`, `hla/`, `cgsp_runtime/`, `riir-neuron-db/src/`, `riir-chain/src/encoding/latcal*.rs`, `dec/`). |
| Q2 New capability class? | **NO.** Transolver is the predecessor to FUNCATTN. FUNCATTN was verdict'd GOAT (not Super-GOAT) because it extends latent_functor rather than creating a new pillar. Transolver extends nothing FUNCATTN doesn't already extend better. | |
| Q3 Product selling point? | **NO.** Transolver's selling point (PDE solving on general geometries) is weaker than FUNCATTN's and not directly applicable to our game/chain/shard domains. The crowd-scale game AI reframing is already covered by FAME + `apply_field_to_crowd` + `crowd_mcgs`. | |
| Q4 Force multiplier? | **Partial.** Connects to latent_functor, DEC, crowd-scale game AI — but these connections are already made (better) in Research 257 and 302. | |

**One-line verdict reasoning:** Transolver is the predecessor to FUNCATTN (Research 257); FUNCATTN uses the same slice/deslice primitive with a strictly stronger closed-form ridge solve, beats Transolver 6–26% empirically, and was itself verdict'd GOAT (not Super-GOAT). Transolver therefore cannot exceed GOAT, and lands at Gain because the only genuinely novel contribution to our corpus is a small DEC vocabulary-bridge entry (slice = codifferential, deslice = exterior_derivative) — not a new primitive, plan, or guide.

### Routing

- **No plan.** Plan 286 (FUNCATTN open primitive) already covers the implementation path. Implementing Transolver separately would be implementing a strictly weaker subset.
- **No Super-GOAT guide.** Would duplicate Research 257 (riir-ai side, FUNCATTN rank-k functor upgrade) and Research 302 (riir-ai side, FAME per-entity committed archetype blend).
- **No riir-train deferral.** The mechanism is inference-time architectural (frozen slice-weight projections at inference). The §3.5 modelless unblock protocol is moot — there is nothing to unblock because there is nothing to implement.
- **This note is the deliverable.** It exists to (a) prevent future readers from accidentally re-distilling Transolver when FUNCATTN already covers it, (b) record the DEC vocabulary bridge for the Stokes crosswalk (Research 296), (c) document the predecessor relationship in the corpus index.

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Physics-Attention is a pure forward-pass operation; "learnable slices" become frozen projection matrices at inference. No backprop. (Moot — nothing to implement.) |
| Latent-to-latent preferred | ✅ Slice weights are dot-product projections; tokens are latent centroids; deslice is latent broadcast. (Moot.) |
| Use sigmoid not softmax | ⚠️ Paper uses softmax (Eq. 1). Research 257 §F2 already mandates sigmoid-normalized basis for FUNCATTN; same fix applies to Transolver. (Moot — Plan 286 handles it.) |
| Freeze/thaw over fine-tuning | ✅ Slice projection matrices are perfect freeze/thaw candidates. (Moot.) |
| 5-repo discipline | ✅ Public note in katgpt-rs; no IP leak. |
| Raw scalars at sync boundary | ✅ In the game-AI reframing, the 5 HLA scalars (valence/arousal/desperation/calm/fear) are the sync boundary; the slice weights and M-centroid tokens stay local. (Moot — covered by FAME R302 §2.4(d).) |

---

## 5. Vocabulary crosswalk entry (for Research 296 Stokes/DEC crosswalk)

Add to the standing DEC vocabulary table in the research skill and in Research 296:

| Paper term (Transolver / FUNCATTN family) | DEC equivalent | Codebase location |
|---|---|---|
| "slice" / "aggregate to M tokens" / "physics-aware token" | `codifferential` (δ, rank k → k−1 aggregation) | `dec/operators.rs:126` |
| "deslice" / "broadcast back to N points" | `exterior_derivative` (d, rank k → k+1 coboundary) | `dec/operators.rs:48` |
| "slice + deslice round-trip with M-attention" | low-rank δ-then-d with M-dim inner operator (harmonic-projector-shaped) | `dec/hodge.rs:302 hodge_decompose` |
| "Physics-Attention = learnable integral on Ω_s" (Thm 3.4) | DEC identity `d∘d=0` on the slice complex (discretization-invariant operator) | `dec/operators.rs` (tests verify `curl(grad)=0`, `div(curl)=0`) |

**Caveat (per R296):** the boundary-vs-volume perf win from Stokes holds only for d ≤ 3. Transolver's M-slice domain is small (M=32–256) but the underlying mesh can be high-dim (3D Shape-Net Car, 32,186 points). The DEC framing is for the *vocabulary bridge*, not for a perf claim.

---

## 6. Relationship to existing research / plans / code

| Item | Layer | Relation | Impact |
|---|---|---|---|
| **Research 257 / Plan 286** (FUNCATTN) | notes + planned (`funcattn.rs`) | **Canonical stronger successor.** Same slice/deslice primitive, closed-form ridge solve replaces softmax M-attention, beats Transolver 6–26% on same benchmarks. | This note defers to 257 for all implementation and fusion. |
| **Research 302 / Plan 321** (FAME CommittedFieldBlend) | notes + planned (`committed_field_blend.rs`, today) | **Per-entity commitment-tier cousin.** Per-entity FIXED MoE blend computed once then frozen — the commitment-tier version of Transolver's per-forward-pass slice blend. | Crowd-scale Transolver reframing subsumed by FAME. |
| **Research 246 / Plan 279** (Manifold Power Iteration MoE Router) | notes + shipped (`manifold_power_iter_router.rs`) | Sibling GOAT. Power iteration on router rows; same "shipped math, novel application" pattern. | Confirms the pattern: Transolver is the third paper in this family verdict'd below Super-GOAT. |
| **Research 123 / Plan 303** (latent_functor) | notes + shipped (`latent_functor/arithmetic.rs`) | Rank-1, λ=0, basis-free special case of FUNCATTN (and therefore of Transolver). | The rank-k upgrade (Plan 318) is the path forward, not Transolver. |
| **Research 219 / Plan 251** (DEC operators) | notes + shipped (`dec/operators.rs`) | The DEC substrate. Slice/deslice = codifferential/exterior_derivative. | Vocabulary bridge (§5 above). |
| **Research 296 / Plan 314** (Stokes calculus wrappers) | notes + planned | The Stokes-theorem vocabulary crosswalk. | §5 adds Transolver to the crosswalk. |
| **Plan 290** (Latent Field Steering, `apply_field_to_crowd`) | shipped (`latent_steering.rs`) | Crowd-scale field application primitive. | The crowd-scale Transolver reframing would compose with this, not replace it. |
| **Plan 298** (Crowd MCGS) | shipped + promoted default (`crowd_mcgs/`) | Crowd-scale retrieval infrastructure. | Crowd-scale game AI stack already exists; Transolver is not unblocking. |

---

## TL;DR

Transolver's Physics-Attention (slice N points → M tokens via softmax projection, attend among M, deslice back via same weights, O(NMC+M²C)) is **the predecessor to FUNCATTN (Research 257)** — same primitive, strictly weaker (softmax M-attention vs closed-form Tikhonov ridge solve), beaten 6–26% on the same PDE benchmarks. FUNCATTN was verdict'd GOAT (not Super-GOAT) with "math pieces all shipped"; the same applies a fortiori to Transolver. **Verdict: Gain.** No plan, no guide, no open primitive — Plan 286 (FUNCATTN) already covers the implementation path strictly better. The only genuinely novel contribution to our corpus is a small DEC vocabulary-bridge entry: Transolver's slice = DEC `codifferential` (rank-aggregation), deslice = DEC `exterior_derivative` (coboundary broadcast). This note exists to prevent future re-distillation and to add the vocabulary entry to the Stokes/DEC crosswalk (Research 296).
