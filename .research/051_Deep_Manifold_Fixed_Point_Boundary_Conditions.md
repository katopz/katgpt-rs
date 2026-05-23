# Research 51: Deep Manifold Part 2 — Fixed-Point Boundary Conditions for Model-Based/Modelless Architecture

> **Paper:** [Deep Manifold Part 2: Neural Network Mathematics](https://arxiv.org/pdf/2512.06563) — Max Y. Ma & Gen-Hua Shi, Dec 2025 (81 pages)
> **Date:** 2025-12, distilled 2025-12
> **Related Research:** 35 (Attractor Fixed-Point), 37 (REAP Model-Based/Modelless), 38 (SDAR Gated), 49 (PTRM Recursive), 50 (LDT Lattice Deduction)
> **Related Plans:** Plan 085 (microgpt-rs, Deep Manifold Boundary Conditions)

---

## TL;DR

Deep Manifold Part 2 formalizes neural networks as **boundary-conditioned fixed-point iterations on stacked piecewise manifolds**. Three key distillations for our stack:

1. **Three-Stage Boundary Condition Theory** (§2.6.1) maps directly to our modelless→model-based pipeline: Pre-training = weak boundary, SFT/Rubric = intended boundary, RL/GRPO = perturbed boundary. Our `ROPD` + `SDAR` + `G-Zero` stack is already the correct decomposition.

2. **Manifold Federation + Model CAP Theorem** (§7.2–7.3) validates our `BanditPruner<P>` + `PromptRouter` + `ExpertBundle` architecture. One monolithic model cannot maximize Coverage, Accuracy, and Performance simultaneously — distributed small elastic models with bandit routing is the structural solution.

3. **Symmetric Boundary Conditions** (§2.6.2) explains why `Bradley-Terry` pairwise ranking (Plan 079) works better than pointwise scoring. BT contrastive pairs produce symmetric attraction/repulsion boundaries — the mathematically optimal convergence corridor for unknown fixed points.

**Verdict: STRONG THEORETICAL VALIDATION OF EXISTING DESIGN. The paper provides the mathematical foundation for WHY our three-layer trait stack (`ConstraintPruner` → `ScreeningPruner` → `BanditPruner`) works. Two actionable distillations: (1) Manifold Residual Scoring trait for explicit fixed-point residual tracking, (2) Federated Boundary Alignment trait for cross-model KL coupling. Both are feature-gated additions.**

---

## 1. Paper Core Ideas (Mapped to Our Stack)

### 1.1 Stacked Piecewise Manifolds → Our DDTree + Layer Stack

The paper's central geometric claim: neural networks are **stacked collections of smooth "pancake" manifolds** (Eq. 1):

```
Mk = ⋃ᵢ Mk,i    (each layer = union of smooth pieces)
```

Each piece is locally low-order, but the global composition captures high-order nonlinearity. This is literally our `DDTree` branch structure:

| Paper Concept | Our Implementation | Location |
|---------------|-------------------|----------|
| Piecewise manifold Mk,i | DDTree branch (per-draft candidate) | `speculative/ddtree.rs` |
| Manifold evolution Φk,i→j | Layer transformation f(hk) | `transformer.rs` |
| Node cover Uk,n(t) | Active token positions per layer | `types.rs` SlotState |
| Data-transit manifold γ(x) | Forward pass trajectory h₀...hL | inference pipeline |
| Intrinsic pathway (§4.2) | Selected DDTree branch after bandit pruning | `BanditPruner<P>` |

**Insight:** Our `DDTree` with `draft_lookahead` already decomposes high-order token prediction into locally tractable pieces. The paper formalizes WHY this works — each branch covers a local manifold slice, and the union recovers global structure.

### 1.2 Fixed-Point Residual → Our Hint-δ Signal

The paper's primitive equation (Eq. 23):

```
f(x) - x = e(x),   minimize e(x)
```

This is structurally identical to our `HintDelta` from G-Zero (Plan 049):

```
δ = (1/T) Σ [log πG(at|q,h,a<t) - log πG(at|q,a<t)]
```

| Paper | Our Stack | Signal |
|-------|-----------|--------|
| Fixed-point residual ‖fθ(x) - x‖² | Hint-δ log-prob shift | Distance from equilibrium |
| Lagrangian energy E(θ) | GRPO advantage + DPO preference | Constrained residual |
| Boundary condition ∂Ω(p) | Prompt + domain constraints | Convergence corridor |
| Saddle-point stationarity ∇θL = 0 | LoRA gradient convergence | Fixed-point reached |

The δ signal IS the fixed-point residual. When δ is large, the generator is far from its fixed point (hint moves it significantly). When δ ≈ 0, the generator is already at equilibrium for that query.

### 1.3 Three-Stage Boundary Conditions → Our Distillation Pipeline

The paper's unified boundary formula (Eq. 71):

```
θ* ∈ argmin_θ E_p [α·KL(pdata‖qθ) + β·C(Φθ(p)) + (1-α-β)·ℓ(qθ,y)]
```

| Stage | Boundary Type | Paper Name | Our Implementation |
|-------|---------------|------------|-------------------|
| **Stage 0** | Weak/implicit | Pre-training | Base model (frozen weights) |
| **Stage 1** | Intended/structured | SFT | ROPD Rubric (Plan 071) — multi-criteria scoring |
| **Stage 2** | Perturbed/discrete | RL | SDAR Gate (Plan 072) + GRPO (Plan 059) |

**Key insight:** Our three distillation paths (A/B/C in G-Zero) map exactly to the paper's three boundary types:

- **Path A (modelless HL):** Stage 0 → Stage 1 via heuristic rules = weak + intended boundaries
- **Path B (δ-modelless):** Stage 1 → Stage 2 via δ-gated absorb = intended + perturbed boundaries
- **Path C (model-based DPO):** Full Stage 2 via gradient updates = perturbed boundary iteration

The paper's claim that "Stage 0 remains indispensable" validates our design: the frozen base model provides the statistical foundation, and all distillation builds on top.

### 1.4 Symmetric Boundary Conditions → Why BT Ranking Works

Paper §2.6.2: When fixed-point location is unknown, **symmetric boundaries** (contrasting positive + negative examples) produce the narrowest convergence corridor.

This explains our empirical result from Plan 079 (Bradley-Terry):

```
Lcon = -E[log(exp(-d(h,h+)) / (exp(-d(h,h+)) + Σⱼ exp(-d(h,h⁻ⱼ))))]
```

BT pairwise ranking IS symmetric boundary condition application:
- Positive pair (chosen) = attraction boundary
- Negative pair (rejected) = repulsion boundary
- Together = two-sided symmetric constraint → fastest convergence to unknown fixed point

The paper formalizes: contrastive learning is the "exact symmetric specialization" of Eq. 71. Our BT ranking for DDTree selection produces the same symmetric constraint at inference time.

---

## 2. Model CAP Theorem → Our Bandit Architecture

### 2.1 The CAP Constraint (§7.2)

A single model cannot simultaneously maximize:
- **Coverage** — breadth of real-world manifold representation
- **Accuracy** — local geometric fidelity (curvature, fixed-point stability)
- **Performance** — numerical efficiency (latency, throughput)

This is structural, not implementational. Pushing Coverage + Accuracy → geometric stiffness → Performance collapses.

### 2.2 Our Existing CAP Solution

Our architecture already distributes across the CAP axes:

| CAP Axis | Our Component | Feature Gate |
|----------|---------------|-------------|
| **Coverage** | `PromptRouter` + `ExpertBundle` (multiple domain LoRAs) | `embedding_router` |
| **Accuracy** | `SpectralQuant` eigenbasis + `MaxSim` late-interaction | `spectral_quant`, `maxsim` |
| **Performance** | `BanditPruner<P>` + `DDTreeBranchCache` + SIMD hot path | `bandit` |

The `BanditPruner<P: ScreeningPruner>` trait is the CAP balancer:
- Modelless mode (cheap Q-values) → optimizes Performance
- Model-based mode (δ signal) → optimizes Accuracy
- Router/domain switching → expands Coverage

### 2.3 Validation: Bandit Results

From our existing benchmarks (Plan 025, riir-ai):
- Bandit-only: 49% accuracy, fastest
- Domain + bandit: 54% accuracy, near-fastest
- Domain only: 51% accuracy, slowest

The bandit is solving the CAP tradeoff dynamically per-query. The paper's §7.2 proves this is the only viable approach under learning complexity.

---

## 3. Manifold Federation → Our Expert/Router Design

### 3.1 Mosaic of Small Elastic Models (§7.3)

The paper argues: "millions of small models, each capturing a local piece of the manifold, aligned with the nonlinearity of its data slice."

This IS our `riir-ai/crates/riir-router` architecture:

| Paper Concept | Our Implementation | Status |
|---------------|-------------------|--------|
| Small elastic model | Domain LoRA adapter (QLoRA/IA3) | ✅ Plan 071 |
| Local manifold piece | Expert domain (bomber, go, fft) | ✅ Plan 023 |
| Mosaic alignment | KL coupling between experts | ◼️ New trait |
| Foundation prior | Frozen base model weights | ✅ Plan 059 |

### 3.2 Federated Learning as Distributed Manifold Alignment (§7.6)

Paper Eq. 163-164: Cross-model KL coupling replaces gradient exchange:

```
q₋ᵢ(·|x) = Σⱼ≠ᵢ αᵢⱼ pθⱼ(·|x)     (ensemble of other models as reward)
θ*ᵢ = argmin [ℓ(θᵢ) + λ·KL(pθᵢ ‖ q₋ᵢ)]  (local manifold aligns to neighbors)
```

This maps to our architecture:

| Paper | Our Stack | Feasibility |
|-------|-----------|-------------|
| Local manifold Mi | Domain LoRA weights per expert | ✅ Already have |
| KL coupling KL(pθᵢ ‖ q₋ᵢ) | Cross-domain consistency scoring | ◼️ New trait |
| Foundation prior θ₀ | Shared frozen base model | ✅ Already have |
| Fisher metric Gi(θ) | Curvature proxy from SpectralQuant | ✅ Already have |
| Boundary sampling μb | Bandit arm selection distribution | ✅ Already have |

**Actionable distillation:** We can implement KL coupling between our domain LoRAs without data exchange. Each expert's output distribution is aligned to the ensemble of other experts' outputs. This is privacy-preserving manifold federation.

---

## 4. Propertyless Activations → Our Validator Trait

### 4.1 The Propertyless Principle (§2.2.4)

Paper: "Activations need not encode intrinsic semantics — they can serve as a universal medium of representation."

This is exactly our `Validator` trait design:

```rust
pub trait Validator: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn is_valid(&self, token: usize, context: &[u8]) -> bool;
    fn validate_string(&self, output: &str) -> bool { /* default */ }
    fn relevance(&self, token: usize, context: &[u8]) -> f32 { /* default */ }
}
```

The `Validator` treats tokens as propertyless — it doesn't care about semantics, only about structural validity. A Sudoku validator, a Bomber move validator, and a Go move validator all implement the same trait because the underlying representation is propertyless.

### 4.2 Semantic-Symbolic Pairing (§4.5)

Paper Eq. 95: Symbolic ↔ Semantic is a near-isomorphic manifold mapping:

```
Φ: S → M    (symbolic → semantic)
Ψ: M → S    (semantic → symbolic)
Ψ ∘ Φ ≈ Id_S,  Φ ∘ Ψ ≈ Id_M
```

Our `PromptRouter` does exactly this:
- Symbolic input (API call, domain tag) → semantic embedding → domain routing
- Semantic output (LoRA logits) → symbolic action (token selection, game move)

---

## 5. Learning Triangle → Our Trait Stack

### 5.1 The Triangle (§5.5)

Learning emerges only when three forces align:

```
Composite Operator: h_{k+1} = Φ_arch ∘ ∂Ω_train ∘ M_data(h_k)
```

| Triangle Side | Paper | Our Stack |
|---------------|-------|-----------|
| **Data** (manifold geometry) | M_data integral | Training corpus + domain config |
| **Training** (boundary conditions) | ∂Ω_train(k) | Loss functions (GRPO, DPO, ROPD, SDAR) |
| **Architecture** (mapping efficiency) | Φ_arch(h_k) | Transformer + LoRA + Pruner traits |

### 5.2 Our Trait Stack IS the Learning Triangle

```
ConstraintPruner    → Data boundary (what's valid in the domain)
    ↓
ScreeningPruner     → Architecture boundary (what's relevant)
    ↓
BanditPruner<P>     → Training boundary (what works, updated online)
    ↓
SpeculativeVerifier → Full architecture (forward pass verification)
```

Each trait layer corresponds to a triangle side:

- `ConstraintPruner::is_valid()` = **Data** constraints (domain-specific validity)
- `ScreeningPruner::relevance()` = **Architecture** scoring (how well does this piece fit)
- `BanditPruner` Q-update = **Training** dynamics (online learning from rewards)

---

## 6. Actionable Distillations

### 6.1 ✅ Already Distilled (No New Code)

| Paper Concept | Our Existing Implementation | Evidence |
|---------------|---------------------------|----------|
| Fixed-point residual | HintDelta (Plan 049) | δ = log-prob shift = residual |
| Three-stage boundaries | ROPD + SDAR + GRPO stack | Plans 071/072/059 |
| Symmetric boundaries | BT pairwise ranking | Plan 079, .benchmarks/011 |
| Node cover dynamics | DDTree branch activation | speculative/ddtree.rs |
| Model CAP distribution | BanditPruner trait composition | Plan 030, .benchmarks |
| Propertyless activations | Validator trait | riir-validator-sdk |
| Iterated integral forward pass | Transformer forward + speculative decode | transformer.rs |
| Stochastic fixed points | SDE noise injection (ELF) | Plan 079, elf_sde feature |
| Small elastic models | Domain LoRA adapters | riir-gpu wgpu training |
| Intrinsic pathways | DDTree + BanditPruner branch selection | speculative/ |

### 6.2 🔧 New Distillations (Feature-Gated)

#### Distillation A: Manifold Residual Scoring

**Paper basis:** §2.3.1 — Fixed-point residual as primitive equation
**What:** Explicit residual tracking for DDTree branch quality.

```rust
// New trait under `deep_manifold` feature gate
pub trait ManifoldResidual: Send + Sync {
    /// Compute fixed-point residual ‖f(x) - x‖ for a candidate
    fn residual(&self, candidate_logits: &[f32], base_logits: &[f32]) -> f32;
    
    /// Check if residual is below convergence threshold
    fn is_converged(&self, residual: f32, tolerance: f32) -> bool {
        residual < tolerance
    }
}
```

**Why:** Currently our `ScreeningPruner::relevance()` returns a scalar without distinguishing "how far from equilibrium" vs "how relevant." Explicit residual tracking lets us separate convergence quality from domain relevance.

**Feature gate:** `deep_manifold` (off by default, research)

#### Distillation B: Federated Boundary Alignment

**Paper basis:** §7.6 — Cross-model reward coupling via KL divergence
**What:** KL alignment between domain experts without data exchange.

```rust
// New trait under `federation` feature gate
pub trait BoundaryAlignment: Send + Sync {
    /// Compute KL divergence between this expert's output distribution and ensemble
    fn kl_alignment(&self, local_logits: &[f32], ensemble_logits: &[f32]) -> f32;
    
    /// Compute alignment weight for boundary coupling
    fn coupling_weight(&self, domain: &str, neighbor_domains: &[&str]) -> f32;
}
```

**Why:** Our domain experts currently train independently. The paper shows KL coupling between experts produces coherent global manifold without centralized aggregation. This would let bomber/go/fft experts inform each other.

**Feature gate:** `federation` (off by default, depends on `bandit`)

#### Distillation C: Symmetric Boundary Gate

**Paper basis:** §2.6.2 — Symmetric boundaries are optimal for unknown fixed points
**What:** Explicit positive/negative boundary pair for BT ranking enhancement.

```rust
// Enhancement to existing bt_rank feature
pub struct SymmetricBoundaryPair {
    pub attraction: f32,  // positive example boundary
    pub repulsion: f32,   // negative example boundary
}

impl SymmetricBoundaryPair {
    /// Paper Eq. 73: symmetric contrastive boundary
    pub fn boundary_strength(&self) -> f32 {
        (self.attraction - self.repulsion).abs() / (self.attraction + self.repulsion + 1e-8)
    }
}
```

**Why:** Our BT ranking already uses pairwise contrast, but doesn't explicitly track the symmetry. Adding explicit positive/negative boundary tracking enables adaptive β scaling based on boundary symmetry quality.

**Feature gate:** enhance existing `bt_rank` feature

---

## 7. What We Don't Need

| Paper Concept | Why Not Needed |
|---------------|----------------|
| Numerical Manifold Method (NMM) solver | Our inference is token-level, not PDE-level |
| Galerkin method implementation | Our "learned basis" is already the Transformer |
| Second-order optimizer (Hessian) | We use AdamW via candle/wgpu, not custom optimizer |
| Physics-Informed constraints (PINN) | We're not solving PDEs, we're doing inference |
| Lagrangian dual variable λ | Our loss functions already handle this implicitly |
| Constitutive modeling | Not applicable to our token prediction domain |
| Formal fixed-point existence proof | Empirical GOAT proof is sufficient for our needs |

---

## 8. Theoretical Validation Summary

The Deep Manifold framework provides the **mathematical why** for several empirical observations in our codebase:

| Our Empirical Observation | Deep Manifold Explanation |
|--------------------------|--------------------------|
| BT ranking > pointwise scoring | Symmetric boundaries are optimal for unknown fixed points (§2.6.2) |
| Bandit routing > static routing | CAP theorem forces dynamic tradeoff (§7.2) |
| δ-modelless works without gradients | Weak boundary conditions still guide iteration (§2.6.3) |
| SDAR gate stabilizes distillation | Perturbed boundaries must be low-magnitude (§2.6.1 Stage 2) |
| ROPD rubric > token-level imitation | Intended boundaries > weak boundaries (§2.6.1 Stage 1) |
| ELF SDE noise improves diversity | Stochastic perturbation prevents basin collapse (§2.5.1 Eq. 41) |
| Width > depth in DDTree | More manifold pieces > deeper single piece (§4.2 intrinsic pathway) |
| LoRA transfer learning works | Early layers encode global curvature, redundant across covers (§5.6) |

---

## 9. Verdict

**STRONG THEORETICAL VALIDATION.** The paper doesn't change our architecture — it explains why our architecture is correct. Our three-layer trait stack (`ConstraintPruner` → `ScreeningPruner` → `BanditPruner`) is the operational form of the paper's Data-Architecture-Training learning triangle. Our distillation pipeline (ROPD → SDAR → GRPO) implements the three-stage boundary condition theory. Our BT ranking instantiates symmetric boundary conditions.

**Two new feature-gated additions** (Manifold Residual Scoring, Federated Boundary Alignment) would deepen the theoretical alignment and enable explicit fixed-point residual tracking. Both are additive, non-breaking, and gated behind new features.

**The paper's most important contribution to our work:** It proves that **modelless distillation is not a hack** — it's the mathematically correct approach when fixed-point locations are unknown (§2.6.3: "weak and discrete boundaries are the only effective choice when symmetric boundaries are not available"). Our Hint-δ → AbsorbCompress → BanditPruner pipeline is the paper's weak boundary iteration instantiated in Rust.

---

## References

- Paper: https://arxiv.org/pdf/2512.06563
- Part 1: https://arxiv.org/abs/2409.17592 (Deep Manifold Part 1: Anatomy of Neural Network Manifolds)
- Our Research 35: Attractor Models — fixed-point iterative refinement
- Our Research 37: REAP — model-based/modelless duality mapping
- Our Research 38: SDAR — sigmoid-gated distillation (perturbed boundary)
- Our Research 49: PTRM — recursive tiny models with noise (stochastic fixed points)
- Our Research 50: LDT — lattice deduction (constraint pruning formalized)
