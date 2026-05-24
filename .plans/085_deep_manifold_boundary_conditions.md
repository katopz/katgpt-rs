# Plan 085: Deep Manifold Boundary Conditions — Feature-Gated Additions

> **Source:** Research 51 — Deep Manifold Part 2 Fixed-Point Boundary Conditions
> **Date:** 2025-12
> **Feature Gates:** `deep_manifold`, `federation`
> **Status:** ✅ Complete

## Tasks

- [x] T1: Add `deep_manifold` feature gate to `Cargo.toml`
- [x] T2: Implement `ManifoldResidual` trait + `L2ResidualScorer` in `src/pruners/manifold_residual.rs`
- [x] T3: Add benchmark `bench_manifold_residual` — residual vs relevance scoring comparison
- [x] T4: Implement `BoundaryAlignment` trait + `KlBoundaryAligner` in `src/pruners/boundary_alignment.rs`
- [x] T5: Add `federation` feature gate (depends on `bandit`)
- [x] T6: Add benchmark `bench_boundary_alignment` — KL coupling between domain experts
- [x] T7: Enhance `bt_rank` with `SymmetricBoundaryPair` tracking
- [x] T8: GOAT proof test — deep_manifold residual convergence on bomber arena
- [x] T9: Update README.md — Deep Manifold section + feature flags table
- [x] T10: Update Research 51 with benchmark results

---

## Context

Deep Manifold Part 2 (arXiv:2512.06563) provides the mathematical foundation for WHY our three-layer trait stack works. The paper formalizes:

1. **Fixed-point residual** `f(x) - x = e(x)` as the primitive neural network equation
2. **Three-stage boundary conditions** — weak → intended → perturbed (our ROPD→SDAR→GRPO pipeline)
3. **Model CAP Theorem** — Coverage/Accuracy/Performance tradeoff (our BanditPruner solves this)
4. **Manifold Federation** — KL coupling between local experts without data exchange

Our architecture already implements these concepts empirically. This plan adds **explicit trait-level support** for two paper concepts that don't yet have first-class representations:

- **Manifold Residual Scoring** — explicit `‖f(x)-x‖` tracking (currently implicit in `relevance()`)
- **Federated Boundary Alignment** — cross-expert KL coupling (currently experts train independently)

Both are **additive, non-breaking, feature-gated** behind new flags. Default build is unaffected.

---

## T1: Feature Gate — `deep_manifold`

**File:** `Cargo.toml`

```toml
deep_manifold = []  # Deep Manifold fixed-point residual scoring (Research 51, Plan 085)
```

No dependencies. Off by default (research feature).

---

## T2: `ManifoldResidual` Trait

**File:** `src/pruners/manifold_residual.rs` (new, gated by `deep_manifold`)

```rust
//! Deep Manifold Part 2 — Fixed-Point Residual Scoring (Research 51)
//!
//! Paper Eq. 23: f(x) - x = e(x), minimize e(x)
//! Our HintDelta already computes this as log-prob shift.
//! This trait makes residual tracking explicit and composable.

/// Fixed-point residual scorer for candidate evaluation.
///
/// In the Deep Manifold framework, inference is boundary-conditioned
/// fixed-point iteration on stacked piecewise manifolds. The residual
/// ‖f(x) - x‖ measures distance from equilibrium — how far a candidate
/// is from its stable fixed point.
///
/// Our HintDelta (G-Zero Plan 049) instantiates this:
///   δ = (1/T) Σ [log πG(at|q,h,a<t) - log πG(at|q,a<t)]
/// where δ ≈ 0 means the generator is at equilibrium.
pub trait ManifoldResidual: Send + Sync {
    /// Compute fixed-point residual between candidate and base logits.
    ///
    /// Returns ‖candidate - base‖ — the distance from the base
    /// distribution's equilibrium. Lower = closer to fixed point.
    fn residual(&self, candidate: &[f32], base: &[f32]) -> f32;

    /// Check if residual is below convergence threshold.
    ///
    /// Paper §2.5.1 Eq. 50: convergence when ρ(J_fi) < 1 and sup‖δt‖ < ∞.
    /// We approximate with L2 residual < tolerance.
    fn is_converged(&self, residual: f32, tolerance: f32) -> bool {
        residual < tolerance
    }

    /// Compute per-position residuals for fine-grained analysis.
    ///
    /// Useful for identifying which tokens are far from equilibrium
    /// vs which have already converged (intrinsic pathway analysis).
    fn per_position_residual(&self, candidate: &[f32], base: &[f32]) -> Vec<f32> {
        candidate.iter().zip(base.iter())
            .map(|(c, b)| (c - b).powi(2))
            .collect()
    }
}

/// L2 norm residual scorer — standard Euclidean distance.
///
/// Paper §2.3.2: Lagrangian energy E(θ) = ∫ ‖fθ(x) - x‖² dμ
pub struct L2ResidualScorer {
    /// Convergence tolerance (default: 1e-4, matching Attractor paper ε)
    pub tolerance: f32,
}

impl Default for L2ResidualScorer {
    fn default() -> Self {
        Self { tolerance: 1e-4 }
    }
}

impl ManifoldResidual for L2ResidualScorer {
    fn residual(&self, candidate: &[f32], base: &[f32]) -> f32 {
        let sum_sq: f32 = candidate.iter().zip(base.iter())
            .map(|(c, b)| (c - b).powi(2))
            .sum();
        sum_sq.sqrt()
    }

    fn is_converged(&self, residual: f32, tolerance: f32) -> bool {
        residual < tolerance
    }
}

/// KL-divergence residual scorer — distributional distance.
///
/// For probability distributions (after softmax), KL divergence
/// measures how much information is lost when using candidate
/// to approximate base. This is the paper's §7.6 KL coupling.
pub struct KlResidualScorer {
    pub tolerance: f32,
}

impl Default for KlResidualScorer {
    fn default() -> Self {
        Self { tolerance: 0.01 }
    }
}

impl ManifoldResidual for KlResidualScorer {
    fn residual(&self, candidate: &[f32], base: &[f32]) -> f32 {
        candidate.iter().zip(base.iter())
            .filter(|(_, b)| **b > 1e-10)
            .map(|(c, b)| {
                let c_safe = c.max(1e-10);
                c_safe * (c_safe / b).ln()
            })
            .sum()
    }
}

/// Composite scorer combining residual with relevance.
///
/// Paper §5.5 Learning Triangle:
///   Composite = Φ_arch ∘ ∂Ω_train ∘ M_data
///
/// This combines manifold residual (architecture quality)
/// with ScreeningPruner relevance (domain fitness).
pub struct ResidualRelevanceScorer<R: ManifoldResidual> {
    pub residual_scorer: R,
    /// Weight for residual vs relevance (0.0 = pure relevance, 1.0 = pure residual)
    pub residual_weight: f32,
}

impl<R: ManifoldResidual> ResidualRelevanceScorer<R> {
    pub fn blended_score(&self, residual: f32, relevance: f32) -> f32 {
        let normalized_residual = 1.0 / (1.0 + residual); // invert: low residual = high score
        let w = self.residual_weight;
        w * normalized_residual + (1.0 - w) * relevance
    }
}
```

**Update `src/pruners/mod.rs`:**
```rust
#[cfg(feature = "deep_manifold")]
pub mod manifold_residual;
```

---

## T3: Benchmark — Residual vs Relevance

**File:** `tests/bench_manifold_residual.rs` (new)

Compare residual-based branch selection vs relevance-based selection on DDTree candidates.

**Metric:** Win rate in bomber arena (1000 rounds) using residual-weighted branch selection vs baseline.

---

## T4: `BoundaryAlignment` Trait

**File:** `src/pruners/boundary_alignment.rs` (new, gated by `federation`)

```rust
//! Deep Manifold Part 2 — Federated Boundary Alignment (Research 51, §7.6)
//!
//! Paper Eq. 163-164: Cross-model KL coupling replaces gradient exchange.
//!   q₋ᵢ(·|x) = Σⱼ≠ᵢ αᵢⱼ pθⱼ(·|x)
//!   θ*ᵢ = argmin [ℓ(θᵢ) + λ·KL(pθᵢ ‖ q₋ᵢ)]
//!
//! Each local expert aligns to the ensemble of other experts,
//! producing coherent global manifold without centralized aggregation.

/// Federated boundary alignment between domain experts.
///
/// In the Deep Manifold framework, each domain expert is a local
/// manifold piece. Boundary alignment ensures these pieces form
/// a coherent global structure through KL coupling — no data exchange,
/// no privacy concern.
pub trait BoundaryAlignment: Send + Sync {
    /// Compute KL divergence between local expert and ensemble.
    ///
    /// Paper §7.6: This is the boundary misalignment measure.
    /// Lower KL = better aligned to global manifold.
    fn kl_divergence(&self, local: &[f32], ensemble: &[f32]) -> f32;

    /// Compute coupling weight for a domain relative to neighbors.
    ///
    /// Domains with higher coupling weight should prioritize alignment.
    /// Weight can be derived from bandit Q-values (high-uncertainty domains
    /// need more alignment) or domain similarity.
    fn coupling_weight(&self, domain: &str, neighbors: &[&str]) -> f32;

    /// Compute the federated boundary penalty for training.
    ///
    /// Paper Eq. 164: L_total = L_base + λ·KL(pθᵢ ‖ q₋ᵢ)
    /// This returns the λ·KL term to add to the base loss.
    fn boundary_penalty(&self, local: &[f32], ensemble: &[f32], lambda: f32) -> f32 {
        lambda * self.kl_divergence(local, ensemble)
    }
}

/// Simple KL-based boundary aligner using symmetric KL.
pub struct KlBoundaryAligner {
    /// Regularization for KL computation (prevents log(0))
    pub epsilon: f32,
}

impl Default for KlBoundaryAligner {
    fn default() -> Self {
        Self { epsilon: 1e-10 }
    }
}

impl BoundaryAlignment for KlBoundaryAligner {
    fn kl_divergence(&self, local: &[f32], ensemble: &[f32]) -> f32 {
        let kl_forward: f32 = local.iter().zip(ensemble.iter())
            .map(|(l, e)| {
                let l_safe = l.max(self.epsilon);
                let e_safe = e.max(self.epsilon);
                l_safe * (l_safe / e_safe).ln()
            })
            .sum();

        let kl_reverse: f32 = ensemble.iter().zip(local.iter())
            .map(|(e, l)| {
                let e_safe = e.max(self.epsilon);
                let l_safe = l.max(self.epsilon);
                e_safe * (e_safe / l_safe).ln()
            })
            .sum();

        // Symmetric KL (Jensen-Shannon proxy)
        (kl_forward + kl_reverse) / 2.0
    }

    fn coupling_weight(&self, domain: &str, _neighbors: &[&str]) -> f32 {
        // Default: uniform coupling. Domain-specific weights can be
        // learned from bandit Q-values in a real deployment.
        let _ = domain;
        1.0
    }
}
```

**Update `src/pruners/mod.rs`:**
```rust
#[cfg(feature = "federation")]
pub mod boundary_alignment;
```

---

## T5: `federation` Feature Gate

**File:** `Cargo.toml`

```toml
federation = ["bandit"]  # Deep Manifold federated boundary alignment — KL coupling (Research 51, Plan 085)
```

Depends on `bandit` because coupling weights are derived from bandit Q-values.

Add to `full` feature:
```toml
full = [..., "federation", "deep_manifold"]
```

---

## T6: Benchmark — Boundary Alignment

**File:** `tests/bench_boundary_alignment.rs` (new)

Test KL coupling between bomber and go domain experts:
1. Generate logits from each domain's LoRA
2. Compute KL divergence with/without alignment
3. Measure coherence improvement (cross-domain task performance)

---

## T7: Symmetric Boundary Pair Enhancement

**File:** Enhance existing `bt_rank` implementation.

Add `SymmetricBoundaryPair` to existing `src/rerank.rs`:

```rust
/// Deep Manifold §2.6.2: Symmetric boundary pair for BT ranking.
///
/// When fixed-point location is unknown, symmetric boundaries
/// (positive attraction + negative repulsion) produce the narrowest
/// convergence corridor. BT pairwise ranking IS symmetric boundary
/// condition application.
#[cfg(feature = "bt_rank")]
pub struct SymmetricBoundaryPair {
    /// Positive (chosen) boundary strength
    pub attraction: f32,
    /// Negative (rejected) boundary strength
    pub repulsion: f32,
}

#[cfg(feature = "bt_rank")]
impl SymmetricBoundaryPair {
    /// Paper Eq. 73: symmetric contrastive boundary strength.
    ///
    /// Higher = more symmetric = better convergence corridor.
    pub fn symmetry(&self) -> f32 {
        let sum = self.attraction + self.repulsion;
        if sum < 1e-8 { return 0.0; }
        1.0 - (self.attraction - self.repulsion).abs() / sum
    }

    /// Adaptive β based on boundary quality.
    ///
    /// More symmetric pairs → higher β → stronger boundary enforcement.
    pub fn adaptive_beta(&self, base_beta: f32) -> f32 {
        base_beta * (0.5 + 0.5 * self.symmetry())
    }
}
```

---

## T8: GOAT Proof Test

**File:** `tests/goat_deep_manifold.rs` (new, gated by `deep_manifold,bomber`)

Prove that residual-weighted branch selection improves bomber arena win rate:

1. Run bomber arena (1000 rounds) with standard `BanditPruner`
2. Run bomber arena (1000 rounds) with `ResidualRelevanceScorer` blended selection
3. Assert: residual-weighted selection >= baseline (within statistical variance)

This proves the Deep Manifold fixed-point residual is a useful signal for branch quality, not just theoretical.

---

## T9: README Update

Add Deep Manifold section to README.md under the existing research stack:

```markdown
## 🧮 Deep Manifold: Fixed-Point Boundary Conditions (Research 51)

Mathematical foundation from [Deep Manifold Part 2](https://arxiv.org/pdf/2512.06563) explaining WHY our trait stack works:

| Paper Concept | Our Implementation | Feature Gate |
|---------------|-------------------|-------------|
| Fixed-point residual ‖f(x)-x‖ | HintDelta + ManifoldResidual trait | `deep_manifold` |
| Three-stage boundaries | ROPD→SDAR→GRPO pipeline | `ropd_rubric`, `sdar_gate` |
| Symmetric boundaries | BT pairwise ranking | `bt_rank` |
| Model CAP tradeoff | BanditPruner dynamic routing | `bandit` |
| Manifold federation | BoundaryAlignment KL coupling | `federation` |
```

---

## T10: Update Research 51

Add benchmark results section to `katgpt-rs/.research/051_Deep_Manifold_Fixed_Point_Boundary_Conditions.md`:

- Residual scoring benchmark results
- Federation alignment benchmark results
- GOAT proof results

---

## Module Structure

```
src/pruners/
├── mod.rs                        # +2 conditional mods
├── manifold_residual.rs          # NEW (deep_manifold gate)
├── boundary_alignment.rs         # NEW (federation gate)
├── bandit.rs                     # existing
├── constraint.rs                 # existing
├── screening.rs                  # existing
└── ...
```

## File Changes Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `Cargo.toml` | Edit (+2 features, +2 to full) | ~5 |
| `src/pruners/mod.rs` | Edit (+2 conditional mods) | ~4 |
| `src/pruners/manifold_residual.rs` | **NEW** | ~120 |
| `src/pruners/boundary_alignment.rs` | **NEW** | ~80 |
| `src/rerank.rs` | Edit (+SymmetricBoundaryPair) | ~35 |
| `tests/bench_manifold_residual.rs` | **NEW** | ~100 |
| `tests/bench_boundary_alignment.rs` | **NEW** | ~80 |
| `tests/goat_deep_manifold.rs` | **NEW** | ~80 |
| `README.md` | Edit (+Deep Manifold section) | ~20 |

**Total new code:** ~500 lines (all feature-gated, default build unaffected)

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Residual scoring adds latency | Feature-gated, O(n) SIMD-able, only active when `deep_manifold` on |
| KL coupling requires multi-expert setup | Feature-gated, testable with synthetic logits |
| BT symmetry enhancement changes ranking | Additive only, doesn't modify existing BT logic |
| Feature flag proliferation | Both flags are research-grade, not in default |

---

## Dependencies

- T1 → T2 (feature gate must exist first)
- T2 → T3 (trait must exist for benchmark)
- T5 → T4 (feature gate must exist first)
- T4 → T6 (trait must exist for benchmark)
- T2 + T7 → T8 (GOAT proof needs residual + symmetry)
- T3 + T6 + T8 → T9 + T10 (benchmarks before docs)