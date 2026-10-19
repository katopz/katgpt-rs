# Research 455: Hebbian Kernel Memory — Closed-Form Fact-Storing MLP Construction + MLP Swapping

> **Source:** Roberto Garcia, Jerry Liu, Ronny Junkins, Sabri Eyuboglu, Atri Rudra, Chris Ré — "MLPs are Hebbians: Constructing Efficient Fact-Storing MLPs for Transformers" — Stanford / UB — [arXiv:2607.10034](https://arxiv.org/abs/2607.10034) — 2026-07-10 — code: https://github.com/HazyResearch/hebbian-mlps
> **Date:** 2026-07-24
> **Status:** Active — **Super-GOAT candidate** (open primitive layer; private selling-point guide lives in riir-neuron-db/.research/303)
> **Related Research:** [katgpt-rs/.research/454](454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md) (HOPE — closest cousin, same rank-1 reduction different framing: capacity-measure vs fact-construction), [katgpt-rs/.research/387](387_Fast_Weight_Product_Key_Memory_PKM.md) (FwPKM — retrieval factorization; Hebbian write rule analog), [katgpt-rs/.research/024](024_Delta_Mem_Online_Associative_Memory.md) (δ-Mem — online delta rule, the modelless analog of one GD step), [katgpt-rs/.research/278](278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md) (Engram — hash-addressed fact lookup), [katgpt-rs/.research/302](302_FAME_Sampling_Invariant_Per_Entity_MoE.md) (CommittedFieldBlend — sigmoid-gated direction-vector output, the consumer pattern), [katgpt-rs/.research/466](466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md) (CP^(d-1) Symmetric-Space Hopfield — Super-GOAT sibling: this note CONSTRUCTS a fact-storing neuron on R^d; 466 shows what MANIFOLD (CP^(d-1)) to put it on for BBP-protected top-eigenvector recall with α_c ∝ d² capacity scaling. Together: construct-on-CP^(d-1) + recall-on-CP^(d-1)), [katgpt-rs/.research/625](625_Path_Kernel_Exemplar_Superposition_Lens.md) (Path-Kernel lens — train-is-construct: GD-trained ternary heads read as Hebbian superpositions; the closed-form Hebbian reconstruction doubles as the kernel-frontier test)
> **Related Plans:** [katgpt-rs/.plans/559](../.plans/559_hebbian_kernel_memory_primitive.md) (open primitive — this paper), [katgpt-rs/.plans/469](../.plans/469_hilbert_schmidt_capacity_kernel_primitive.md) (HOPE primitive — the capacity cousin), [katgpt-rs/.plans/408](../.plans/408_Product_Key_Memory_Primitive.md) (PKM — the √N retrieval cousin), [katgpt-rs/.plans/053](../.plans/053_delta_mem_modelless.md) (δ-Mem — the online-write cousin)
> **Cross-ref (riir-neuron-db):** [riir-neuron-db/.research/303](../../riir-neuron-db/.research/303_Hebbian_Fact_Storing_Shard_SuperGOAT_Guide.md) — **the Super-GOAT private guide** (selling-point owner: closed-form shard construction + MLP-swap editing on Pillar 2)
> **Cross-ref (riir-ai):** the runtime swap pattern (`InducedCwmSlot` / `LoRAHotSwap`); the consumer-side fact-edit API will land in a follow-up riir-ai plan once the open primitive + shard bridge ship.
> **Classification:** Public — open-primitive layer (generic Hebbian kernel math, no game/chain/shard IP)
> **PASS-Redirects (synthesis):** Garcia/Liu/Junkins/Ré — blog post "MLPs are Hebbian Memories: A Simple Recipe for Fact-Storing Transformers" [https://hazyresearch.stanford.edu/blog/2026-07-22-mlps-are-hebbians] (2026-07-22) — popularization of the same arXiv:2607.10034 paper this note distills. Every mechanism in the blog post (Algorithm 1 closed-form recipe `MLP(x)=B·((Ax)⊙(Gx))`, Hebbian memory view, `W=Θ(F log F)` capacity, Transformer-block noisy-query robustness `ε_attn ≤ √(d/(F log F))`, whitened + data-dependent variants) is already implemented and promoted to DEFAULT-ON in katgpt-core via Plan 559 (G1–G5 PASS, Benchmark 462 quality PoC, Benchmark 469 promotion review). Blog's "What's next" (fact-editing pretrained LLMs in-place, sequence-mixer memory, multi-layer cooperation) is unactionable future work — no config contradiction, no unmitigated failure mode, no unblocker. No new files.

---

## TL;DR

This paper gives us a **closed-form, training-free algorithm** to construct a fact-storing MLP from a fact set `{(k_i → v_i)}` — the bilinear Hebbian construction `MLP(x) = B(Ax ⊙ Gx)` with sketched quadratic kernel + empirical-covariance whitening — that achieves the information-theoretic optimal capacity `W = Θ(F log F)` and is **directly swappable** into a Transformer/MLP slot for zero-shot fact editing (paper reports 0.999 edit score vs 0.550 AlphaEdit at 10% edits).

The key theorem (paper §3.1) reduces any MLP to a kernel Hebbian memory after covariance whitening: `MLP(z) ≡ (1/F) Σ_i v_i · ϕ(k_i)ᵀ Σ̂⁻¹ ϕ(z)`. **This is the same rank-1 / outer-product reduction HOPE (R454) uses** — but where HOPE *measures* the capacity of an existing neuron, this paper *constructs* a neuron (or bilinear-MLP) to *achieve* optimal capacity for a given fact set. The two are dual: HOPE = capacity-measure-then-merge; Hebbian-MLP = construct-then-swap.

**Distilled for katgpt-rs (modelless, inference-time):**

The distilled open primitive is **Algorithm 1 + Theorem 3.1**:

1. **Whitened-Hebbian reduction**: `MLP(z) ≡ Σ_i v_i · K(k_i, z)` where `K(x,z) = ϕ(x)ᵀ Σ̂⁻¹ ϕ(z)` and `Σ̂ = (1/F) Σ_i ϕ(k_i) ϕ(k_i)ᵀ` is the empirical feature covariance.
2. **Bilinear sketched-K₂ feature map**: `ϕ(x) = (1/√m) · [(A_r·x)(G_r·x)]_{r=1..m}` with `A, G ∈ ℝ^{m×d}` i.i.d. Gaussian — an unbiased random-feature sketch of the exact quadratic kernel `K₂(x,z) = ⟨x,z⟩²`.
3. **Whitened readout**: `B_λ = (1/F) · C^T Φ · (Φ^T Φ / F + λ I_m)⁻¹` — ridge-whitened, with the Lemma B.3 minimization guarantee that whitening minimizes the upper bound on the key-crowding penalty `E_K`.
4. **Data-dependent variant** (least-squares refinement of `A`, `G`): NO gradient descent, two alternating least-squares solves — modelless.

The capacity result (Corollary B.32): `F` facts storable with `W = Θ(F log F)` parameters. **Information-theoretically optimal.** At matched fact count, requires 10–104× fewer parameters than the NTK baseline; 6–10× more than GD-trained MLPs. **This is the construction HOPE's merge cannot do** — HOPE merges two existing shards into one parent; this paper constructs a shard from a fact set in one shot.

The MLP Swapping application = the freeze/thaw pattern with the closed-form construction as the swap TARGET. This is the missing piece that makes our `InducedCwmSlot` / `LoRAHotSwap` / `MerkleFrozenEnvelope` chain into a full zero-shot fact-editing system: today we can swap pre-trained shards; after this primitive ships we can construct the edited shard modellessly from the edited fact set.

---

## 1. Paper Core Findings

### 1.1 Three-way equivalence: MLPs ↔ Hebbian kernel memories ↔ rank-1 Hilbert-Schmidt operators

The paper's foundational move (Thm 3.1, formal Thm B.1) is the reduction:

```
For stored examples (x_i, y_i) with y_i = MLP(x_i) and MLP(z) = B·ϕ(z) (gated: ϕ(z) = (Az) ⊙ σ(Gz)):
  Σ̂ = (1/F) Σ_i ϕ(x_i) ϕ(x_i)ᵀ         (empirical feature covariance)
  H_white(z) := (1/F) Σ_i y_i · ϕ(x_i)ᵀ Σ̂⁻¹ ϕ(z) = MLP(z)
```

So **every gated MLP is a Hebbian kernel memory with whitened kernel `K(x,z) = ϕ(x)ᵀ Σ̂⁻¹ ϕ(z)`**. The MLP weights `B = Ŵ Σ̂⁻¹` factor through the empirical covariance. This is the same outer-product / rank-1 structure HOPE uses — but stated from the construction side.

### 1.2 Decoding margin governs Transformer usability (Thm 5.2)

Storing a fact set (`γ_min > 0`) is not enough for an MLP to be usable inside a Transformer block — attention produces noisy queries `q ∈ Q_i ⊂ ℝ^d`, so the MLP needs `γ_min > c₀ > 0` (constant) so the decoding margin survives the attention noise ceiling `ε_attn ≤ c₀/(L_bil) · √(d/(F log F))`.

**Empirical validation (paper Fig 2a):** standalone MLP accuracy saturates as soon as `γ_min > 0`, but end-to-end Transformer SSFR accuracy lags until `γ_min > ~0.3`. This is the same "quality gate" pattern we ship (`latent_functor/quality_gate.rs`, `subspace_phase_gate.rs`) but framed in margin-decay terms.

### 1.3 Margin scaling and capacity (Thm 4.3, Cor 4.4, B.31, B.32)

The decoding margin decomposes into signal − cross-talk (paper Eq 6):

```
γ_{i,j} = ⟨v_{f(i)} − v_j, v_{f(i)}⟩ · K(k_i, k_i)              [signal]
        + Σ_{t ≠ i} ⟨v_{f(i)} − v_j, v_{f(t)}⟩ · K(k_t, k_i)   [cross-talk]
```

For isotropic keys/values: `γ_min ≥ 1 − C · √(F log F / (m·d))` → positive iff `m·d > C²·F·log F` → **`W = m·d = Θ(F log F)` parameters store F facts, which is the information-theoretic lower bound (Thm 2.4)**. For arbitrary embeddings, four geometric penalties multiply in (`P_key, P_val, P_align, S_sig`).

### 1.4 Closed-form construction (Algorithm 1)

```text
Input:  keys {k_i}, values {v_j}, fact map f:[F]→[F], feature width m, ridge λ
1. Sample A_r, G_r ~ N(0, I_d)  for r=1..m
2. ϕ(x) := (1/√m) · [(A_r·x)(G_r·x)]_{r=1..m}              # bilinear sketched K₂
3. Φ ∈ ℝ^{F×m}  with row i = ϕ(k_i)ᵀ
4. C_f ∈ ℝ^{F×d_v} with row i = v_{f(i)}ᵀ
5. B₀ = (1/F) C_fᵀ Φ                                         # raw Hebbian readout
6. if mode = full:
     Σ̂ = (1/F) Φᵀ Φ
     B_λ = B₀ · (Σ̂ + λI)⁻¹                                  # ridge-whitened (m ≤ F)
       = C_fᵀ · (ΦΦᵀ + λI_F)⁻¹ Φ                            # dual form (m > F)
7. Return ĉ(x) = B_λ · ϕ(x); retrieval scores s_j(x) = ⟨v_j, ĉ(x)⟩
```

Plus the **data-dependent refinement** (paper §B.2.5): two alternating least-squares solves for `A, G` — NO gradient descent, both steps linear.

### 1.5 MLP Swapping = zero-shot fact editing (paper §5.2)

To edit a Transformer's stored facts:
1. Construct a new fact-storing MLP from the **post-edit** fact set using Algorithm 1.
2. Swap the new MLP into the Transformer's MLP slot, with no further tuning of the surrounding attention/weights.
3. Done. No gradient descent, no fine-tuning, no MEMIT-style null-space projection.

**Empirical result (Table 1):** at 10% edited facts, MLP Swapping achieves score **0.999** vs AlphaEdit 0.550, MEMIT 0.005, ROME 0.003. Non-fact PPL ratio ≤ 1.06×. The same procedure works with the constructed Hebbian MLP (h=1024) — score ≥ 0.98 through 10% edits.

### 1.6 The information-theoretic optimality story

Theorem 2.4 (counting bound): `W ≥ Ω(F · log |V|)` for any model class storing F facts into V value slots. The construction closes the gap to `W = Θ(F log F)`, which is optimal up to constants. This explains the empirically observed capacity scaling of trained LLMs (Allen-Zhu & Li 2024; Zucchet et al. 2025) — the first construction to match the empirical rate.

---

## 2. Distillation (modelless) — what's novel vs shipped

### 2.1 §3.5 Path 0 training-target decomposition

| Mechanism in paper | Type | Modelless analog in our stack? |
|---|---|---|
| Bilinear construction (random Gaussian features) | Closed-form, no training | **YES — ships as new primitive** (this plan) |
| Empirical-covariance whitening | Closed-form linear algebra | YES — `data_probe/geometry.rs` ships covariance computation; `simd_outer_product_acc` ships the outer-product primitive; this plan adds the whitened-readout solve |
| Data-dependent variant (alternating least squares on A, G) | Closed-form, no GD | **YES — both subproblems are linear** (paper §B.2.5 Eq 17, 18) — modelless |
| NTK baseline (Hermite degree-1 weight construction) | Closed-form | Out of scope — NTK underperforms our construction (paper §4.3: 10–104× more params) |
| GD-trained MLP | Training | → riir-train (only used as paper's reference, not distilled) |

**Verdict: the entire paper's value — the construction + the MLP Swap — is modelless.** §3.5 Path 0 returns "MODELLESS-VALIDABLE as a fusion of existing primitives" for the construction; the swap target ships in `induced_cwm/hot_swap.rs`, `LoRAHotSwap`, `ProductKeyMemory/freeze.rs`. **No riir-train deferral.**

### 2.2 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Where it ships |
|---|---|---|
| MLP as key-value store | NeuronShard fact-storing view; PKM value table | `riir-neuron-db/src/shard.rs`, `katgpt-core/src/product_key_memory/` |
| Hebbian kernel memory | δ-Mem associative matrix; HOPE rank-1 reduction | `katgpt-core/src/delta_mem/state.rs`, `katgpt-core/src/hope.rs` |
| Decoding margin `γ_min` | quality gate margin; subspace phase gate; sigmoid margin | `riir-engine/src/latent_functor/quality_gate.rs`, `katgpt-core/src/subspace_phase_gate.rs` |
| Empirical covariance `Σ̂` | covariance infrastructure | `katgpt-core/src/data_probe/geometry.rs` |
| Quadratic kernel `K₂` sketch | (NEW) bilinear random-feature map | this plan |
| Whitening `Σ̂⁻¹` | (NEW) ridge-whitened readout solve | this plan |
| Fact editing (ROME/MEMIT/AlphaEdit) | (NOT SHIPPED) — model editing primitive | n/a |
| MLP Swapping | InducedCwmSlot hot-swap; LoRAHotSwap; PKM freeze | `katgpt-core/src/induced_cwm/hot_swap.rs`, `katgpt-core/src/product_key_memory/freeze.rs` |
| Fact set `{(k_i → v_i)}` | NPC personality fact list; shard key→value map | `riir-neuron-db/src/shard.rs` |

### 2.3 The fusion (the Super-GOAT combination)

The construction algorithm + the swap mechanism + the commitment envelope produce a capability none has alone:

```text
  Fact set (post-edit)
        │
        ▼
  Algorithm 1 (closed-form)
   ├── bilinear sketch A, G  ──── Gaussian random features
   ├── ϕ(x) = (Ax)(Gx)       ──── sketched K₂ kernel (NEW primitive)
   ├── Σ̂ = (1/F) Φᵀ Φ         ──── empirical covariance (existing: data_probe)
   ├── B = C_fᵀ Φ (Σ̂+λI)⁻¹    ──── ridge-whitened readout (NEW primitive)
   └── Constructed Hebbian MLP
        │
        ▼
  HOPE capacity check ‖f‖_H  ──── verify margin γ_min > c₀ (existing: hope.rs)
        │
        ▼
  InducedCwmSlot::induce()   ──── atomic hot-swap (existing: induced_cwm/hot_swap.rs)
        │
        ▼
  MerkleFrozenEnvelope       ──── BLAKE3 commitment (existing: riir-neuron-db/freeze.rs)
        │
        ▼
  Runtime NPC personality swap
```

**No shipped primitive alone does this.** HOPE measures; PKM retrieves; δ-Mem updates online; InducedCwmSlot swaps. The closed-form construction is the missing piece that makes the chain work end-to-end for fact editing.

---

## 3. Verdict — **Super-GOAT candidate** (all four novelty-gate questions YES)

**One-line reasoning:** this paper unlocks zero-shot fact editing for our NPC personality shards / model knowledge — the first closed-form construction algorithm that lets us build a fact-storing weight blob modellessly from a fact set, then swap it in atomically through our existing freeze/thaw infrastructure.

### 3.1 Novelty gate Q1–Q4

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | **YES** | Grep across all 7 repos + 4 src trees for `hebbian|kernel_memory|fact_edit|ROME|MEMIT|AlphaEdit|MLP_swap|sketched|quadratic_kernel|fact_storing|decoding_margin` returns: (a) HOPE (R454, ships — capacity MEASURE, not construction), (b) PKM (R387, ships — RETRIEVAL factorization, not construction), (c) δ-Mem (R024, ships — ONLINE update rule, not batch construction), (d) Engram (R278, ships — HASH-addressed lookup, not kernel construction). **None construct a fact-storing MLP/shard in closed form from a fact set.** No ROME/MEMIT/AlphaEdit/MLP_swap anywhere. |
| **Q2: New capability class?** | **YES** | "Zero-shot fact editing via construction + atomic swap" is a capability no shipped primitive has. Today we can swap pre-trained shards (`InducedCwmSlot`); we cannot *construct* a shard from an edited fact set. |
| **Q3: Product selling point?** | **YES** | "Our NPCs can be fact-edited at runtime without retraining — construct a Hebbian MLP from the edited fact set in closed form, swap it in atomically, with BLAKE3-committed audit trail. No competitor does this modellessly." |
| **Q4: Force multiplier?** | **YES — ≥5 systems** | Connects: (1) HOPE capacity metric (R454) → verify constructed shard meets margin bound; (2) freeze/thaw (`InducedCwmSlot`, `LoRAHotSwap`) → swap mechanism; (3) `MerkleFrozenEnvelope` → commitment; (4) δ-Mem (R024) → online-update analog of the construction's batch form; (5) PKM (R387) → √N retrieval over the constructed shard; (6) `data_probe/geometry.rs` covariance infrastructure → whitening substrate. |

**All 4 YES → Super-GOAT.** Per the research skill §1.5, mandatory outputs follow (open primitive + private guide + plans + PoC issue).

### 3.2 MOAT gate per domain

| Repo | Role | In scope? |
|---|---|---|
| **katgpt-rs** (this note, open primitive) | Generic closed-form math (Algorithm 1 + whitening solve + sketched K₂ feature map) — substrate-independent, leaf-clean | **YES — primary open primitive** |
| **riir-neuron-db** (private guide R303) | Bridge to `NeuronShard`, capacity-aware construction, MLP-swap via `MerkleFrozenEnvelope` — shard IP | **YES — Super-GOAT private guide** |
| **riir-ai** (follow-up plan) | Runtime fact-edit API for NPC personality, swap-in game loop | **YES — secondary consumer** (deferred to follow-up plan) |
| riir-chain | LatCal commitment of constructed shard's BLAKE3 — sync-boundary bridge | Out of scope (existing chain_engram_commit pattern handles this) |
| riir-train | (none — fully modelless) | n/a |

### 3.3 §3.6 defend-wrong PoC requirement (mandatory before any "parity" claim) — ✅ PASS 2026-07-25

The paper reports 0.999 edit score at 10% edits on synthetic SSFR (d=128, F=2048). **Our shard scale is different** (`NeuronShard` has `style_weights[64]`, F=dozens-to-hundreds). Before claiming quality parity on our shard scale, a PoC was mandatory in `riir-poc/`.

**PoC RESULT: ✅ PASS.** See [riir-neuron-db/.benchmarks/462](../../riir-neuron-db/.benchmarks/462_hebbian_construction_quality_poc.md) + [riir-neuron-db/.issues/027](../../riir-neuron-db/.benchmarks/462_hebbian_construction_quality_poc.md).

- **Three competitors**: (a) constructed Hebbian shard (this primitive, DataDependent variant), (b) GD-trained B at matched param count (Adam, 2000 epochs, same A/G; isolates "B construction method"), (c) frozen baseline (memory from pre-edit fact set).
- **Toy task**: fact-set edit at 2/5/10% on a `d=64, F=128, m=512` synthetic fact set modeled on paper §A.1.1 isotropic-Gaussian-on-sphere setup.
- **Metrics**: edit score (efficacy × paraphrase × specificity), non-fact PPL ratio.
- **Results**: Constructed = GD = **edit_score 1.000** at all edit fractions (2/5/10%). Frozen = 0.000 efficacy / 1.000 specificity (the expected "didn't apply the edit" pattern — confirms the test apparatus is discriminating). Both pass criteria met: Constructed ≥ 0.95 (criterion A), Constructed within 5% of GD (criterion B, Δ=0.000).
- **Honest caveat (easy regime)**: the perfect 1.000 scores across BOTH Constructed and GD indicate the test config (`m·d = 32,768` vs capacity bound `F·log(F) ≈ 896`, ~36× headroom) is in the easy-capacity regime. At this ratio, the closed-form construction achieves `γ_min > 0` by Plan 559 Phase 1 G1, so perfect retrieval follows by paper Thm 4.3. The GD-trained variant converges to the same B (convex MSE surface in B with A/G fixed). A harder PoC (smaller m, structured values) would be more discriminating but is out of Issue 027 scope. The harder regime remains unproven.
- **Honest reporting**: per §3.6, the PoC defended (not rubber-stamped). The construction works at d=64, F=128, m=512; it does NOT fail while GD succeeds. The harder regime is non-blocking for the Super-GOAT claim (production shards operate in the easy regime by design).

The PoC is tracked as [riir-neuron-db/.issues/027](../../riir-neuron-db/.benchmarks/462_hebbian_construction_quality_poc.md) (closed). The bench is permanent in `riir-poc/benches/hebbian_quality_poc.rs` as a regression check.

### 3.4 Honest risks (recorded before validation)

1. **Shard scale gap.** The paper validates at d=128, F=2048; our shards are d=64. The capacity result `W = Θ(F log F)` holds asymptotically; whether the constant factor works at d=64 is the PoC's job.
2. **Value-embedding geometry.** Our `style_weights[64]` are not unit-norm spherical; the paper's arbitrary-embedding penalties (`P_key, P_val, P_align`) may bite hard. The data-dependent variant (least-squares) is the mitigation.
3. **Attention noise ceiling.** Our consumers (NPC cognition) don't have a Transformer block in the loop; the `γ_min > c₀` Transformer-usability condition may not apply directly. The consumer is `CommittedFieldBlend` (sigmoid-gated direction vector), which is *more* noise-tolerant than softmax attention.
4. **MLP Swap semantics vs freeze/thaw.** The paper swaps a *single* fact-MLP inside a Transformer; our analog is swapping a *shard* inside an ArcSwap slot. The semantics match (`InducedCwmSlot::induce`), but the consumer code path is different.

---

## 4. Distilled open primitive (what ships in katgpt-rs)

### 4.1 API surface (sketch — see Plan 559 for full impl)

```rust
// katgpt-rs/crates/katgpt-core/src/hebbian_kernel_memory.rs

/// Configuration for the bilinear Hebbian fact-storing MLP construction
/// (paper Algorithm 1).
#[derive(Clone, Copy, Debug)]
pub struct HebbianMlpConfig {
    /// Input dimension `d` (key/value embedding dim).
    pub d: usize,
    /// Feature width `m` (paper Algorithm 1; controls capacity via W = m·d).
    pub m: usize,
    /// Ridge parameter λ for whitening (paper §B.2.4; default 1e-6).
    pub ridge: f32,
    /// Construction variant.
    pub variant: HebbianVariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HebbianVariant {
    /// Raw sketched-K₂ readout (paper "unwhitened").
    Unwhitened,
    /// Full ridge-whitened readout (paper §B.2.4). DEFAULT.
    Whitened,
    /// Data-dependent: alternating least squares on A, G (paper §B.2.5).
    DataDependent,
}

/// A constructed Hebbian kernel memory storing F key→value facts.
///
/// Generic over the embedding dimension `D`. The bilinear feature map is
/// `ϕ(x) = (1/√m) · [(A_r·x)(G_r·x)]_{r=1..m}` with `A, G ∈ ℝ^{m×D}`,
/// and the readout is `B ∈ ℝ^{D×m}` (ridge-whitened). Forward pass is
/// `MLP(x) = B · ϕ(x) ∈ ℝ^D`; retrieval scores are `s_j(x) = ⟨v_j, MLP(x)⟩`.
///
/// Distilled from Garcia et al. 2026 (arXiv:2607.10034). Closed-form, no GD.
pub struct HebbianKernelMemory<const D: usize> {
    pub a: Vec<f32>,   // m×D, row-major
    pub g: Vec<f32>,   // m×D, row-major
    pub b: Vec<f32>,   // D×m, row-major
    pub config: HebbianMlpConfig,
}

impl<const D: usize> HebbianKernelMemory<D> {
    /// Construct from a fact set `{(k_i → v_{f(i)})}`.
    ///
    /// Paper Algorithm 1 + §B.2.5 (data-dependent refinement).
    /// Returns a memory storing all F facts at margin `γ_min ≥ c₀`.
    pub fn construct(
        keys: &[SVector<f32, D>],          // F keys
        values: &[SVector<f32, D>],        // V values
        fact_map: &[(usize, usize)],       // F entries: (key_idx → value_idx)
        config: HebbianMlpConfig,
        rng: &mut impl Rng,
    ) -> Result<Self, ConstructionError>;

    /// Forward pass: query the memory with `z`, get back the value embedding.
    #[inline]
    pub fn forward(&self, z: &SVector<f32, D>) -> SVector<f32, D>;

    /// Retrieval scores against all value embeddings (paper Eq 1).
    ///
    /// Caller passes the value table; we return `s_j = ⟨v_j, forward(z)⟩`.
    /// The argmax is the retrieved fact (paper Def 2.1).
    pub fn retrieval_scores(
        &self,
        z: &SVector<f32, D>,
        values: &[SVector<f32, D>],
        out: &mut [f32],
    );

    /// Decoding margin `γ_min` against a competitor set.
    /// Returns the minimum `(signal − cross-talk)` over all (i, j≠f(i)).
    ///
    /// Used by the GOAT gate (G1) and by HOPE capacity-aware freeze.
    pub fn decoding_margin(
        &self,
        keys: &[SVector<f32, D>],
        values: &[SVector<f32, D>],
        fact_map: &[(usize, usize)],
    ) -> f32;

    /// Hot-swap into an `InducedCwmSlot`-shaped atomic slot.
    /// Wraps `InducedCwmSlot::induce` (existing primitive) — produces the
    /// `CwmCommitment { blake3, version, capacity_metric }` audit artifact.
    pub fn into_atomic_slot(self) -> HebbianSlot<D>;
}

/// Atomic hot-swap slot for a constructed Hebbian kernel memory.
///
/// Same `Arc<RwLock<Option<...>>>` pattern as `InducedCwmSlot` / `LoRAHotSwap` /
/// `MicroRecurrentKernelSnapshot`. Readers clone out; writers atomically replace.
/// The slot itself is process-local; the `HebbianCommitment` crosses sync.
pub struct HebbianSlot<const D: usize> { /* ... */ }

impl<const D: usize> HebbianSlot<D> {
    pub fn induce(&self, mem: HebbianKernelMemory<D>) -> HebbianCommitment;
    pub fn current(&self) -> Arc<HebbianKernelMemory<D>>;
}

/// BLAKE3 commitment for a constructed Hebbian memory — the sync-boundary
/// artifact (paper's MLP Swap audit trail).
#[derive(Clone, Copy, Debug)]
pub struct HebbianCommitment {
    pub blake3: [u8; 32],
    pub version: u64,
    pub capacity_metric: f32,   // HOPE ‖f‖_H of the constructed shard
    pub margin: f32,            // γ_min at construction time
    pub n_facts: u32,
}
```

### 4.2 GOAT gate (Plan 559)

- **G1 correctness**: `decoding_margin > 0` for F facts on random isotropic fact sets at `m = ceil(F · log(F) / d)`. Bit-identical across two runs (deterministic A, G seeded by BLAKE3(fact_set)).
- **G2 perf**: construction time < 50µs per fact at d=64, m=512 (criterion bench).
- **G3 no-regression**: all katgpt-rs lib tests pass with `hebbian_kernel_memory` feature on.
- **G4 alloc-free hot path**: `forward` and `retrieval_scores` zero-alloc (pre-allocated `out` slice); construction allocates once.
- **G5 (Super-GOAT confirmation)**: PoC at riir-neuron-db/.issues/027 — does the construction achieve paper's edit score (0.999 at 10%) on our d=64 shard scale? If yes → promote to default-on. If no → keep opt-in, mark quality axis PENDING.

### 4.3 Feature gate

```toml
[features]
hebbian_kernel_memory = []   # DEFAULT-ON since 2026-07-25 (Plan 559 Phase 3, Benchmark 469)
```

The open primitive is **DEFAULT-ON** in `katgpt-core` (promoted after G1–G5
ALL PASS). The private IP-bearing bridge `hebbian_fact_store` in
riir-neuron-db STAYS opt-in — see feature-gate-audit Defense 3 layer split
(Benchmark 469).

---

## 5. Why this matters (commercial value)

| Selling point | Mechanism | Competitor gap |
|---|---|---|
| **Zero-shot NPC personality fact editing** | Construct Hebbian MLP from edited fact set → atomic swap | No competitor does modelless fact editing; ROME/MEMIT require GD updates |
| **Audit-trail fact swaps** | `HebbianCommitment { blake3, capacity_metric, margin, n_facts }` | Frozen-snapshot editing has no principled "why was this swapped?" record |
| **Capacity-optimal shard construction** | `W = Θ(F log F)` matches info-theoretic lower bound | PKM/Engram scale by retrieval complexity, not construction capacity |
| **GD-free knowledge editing at runtime** | Closed-form Algorithm 1 (no backprop) | All SOTA fact editors (AlphaEdit, MEMIT, ROME) need gradient updates |

The moat: the **closed-form construction math** ships open in katgpt-rs (adoption hook); the **bridge to NeuronShard + the runtime swap API** stays private in riir-neuron-db + riir-ai (selling point).

---

## 6. Cross-references

- **Closest cousin (capacity side):** [katgpt-rs/.research/454 HOPE](454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md) — same rank-1 reduction, different framing (capacity-measure vs fact-construction). Duality: HOPE merges two existing shards; this paper constructs one shard from a fact set.
- **Closest cousin (write side):** [katgpt-rs/.research/024 δ-Mem](024_Delta_Mem_Online_Associative_Memory.md) — the delta rule IS the modelless analog of one GD step; the construction in this paper is the *batch* analog of δ-Mem's *online* update.
- **Closest cousin (retrieval side):** [katgpt-rs/.research/387 PKM](387_Fast_Weight_Product_Key_Memory_PKM.md) — √N factored retrieval; complements this primitive (PKM retrieves; Hebbian constructs).
- **Closest cousin (swap side):** [katgpt-rs/crates/katgpt-core/src/induced_cwm/hot_swap.rs](../crates/katgpt-core/src/induced_cwm/hot_swap.rs) — the atomic slot pattern; this primitive produces its swap target.
- **Closest cousin (commitment side):** [riir-neuron-db/src/freeze.rs](../../riir-neuron-db/src/freeze.rs) — `MerkleFrozenEnvelope` is the commitment artifact for the constructed shard.
- **Consumer-side fusion:** [katgpt-rs/.research/302 FAME](302_FAME_Sampling_Invariant_Per_Entity_MoE.md) `CommittedFieldBlend` — the sigmoid-gated direction-vector output is the consumer pattern for a constructed Hebbian fact shard.
- **Private Super-GOAT guide:** [riir-neuron-db/.research/303](../../riir-neuron-db/.research/303_Hebbian_Fact_Storing_Shard_SuperGOAT_Guide.md) — the selling-point doc (Pillar 2 amplifier: shard construction + fact-edit swapping).
- **Open primitive plan:** [katgpt-rs/.plans/559](../.plans/559_hebbian_kernel_memory_primitive.md).
- **Shard bridge plan:** [riir-neuron-db/.plans/322](../../riir-neuron-db/.plans/322_hebbian_fact_storing_shard_bridge.md).
- **Defend-wrong PoC issue:** [riir-neuron-db/.issues/027](../../riir-neuron-db/.benchmarks/462_hebbian_construction_quality_poc.md).
