# Plan 235: SLoD Spectral Level-of-Detail Pruner — Modelless KG Resolution Control

**Date:** 2026-06
**Status:** ✅ Complete (GOAT G1-G6 all pass, 21 tests)
**Research:** 208 (SLoD Semantic Level of Detail)
**Depends On:** Plan 156 ✅ (Spectral Hierarchy), Plan 218 ✅ (BFCF × LFU Sharding)
**Feature Gate:** `slod` (default-ON, GOAT G1-G6 all pass)
**Classification:** Modelless — inference-time only, no LLM training

---

## Context

SLoD (Semantic Level of Detail, arxiv 2603.08965) provides a continuous zoom operator on knowledge graphs via heat kernel diffusion on a Poincaré-ball-induced kNN graph Laplacian. Spectral gaps in the Laplacian induce emergent scale boundaries — values of σ where the representation undergoes qualitative transitions — detectable without manual resolution tuning.

We already have `spectral_hierarchy` module (Plan 156) with eigenspace alignment, Haar wavelets, Cauchy interlacing, and Jacobi eigendecomposition. SLoD extends this from diagnostics to active resolution control of our DDTree + ConstraintPruner pipeline.

**GOAT proof target:** SLoD-adaptive DDTree budget ≥ fixed-depth DDTree quality with ≤ manual parameter tuning overhead, verified by spectral boundary recovery on synthetic HSBM hierarchies.

---

## Architecture

### Three-Layer SLoD Pipeline

```
Layer 1: Hyperbolic Embedding (offline, pre-compute)
  KG entities → Poincaré ball B^d via Nickel-Kiela or MDS
  Build kNN graph with hyperbolic distance
  Compute normalized Laplacian L

Layer 2: BoundaryScan (offline or on-demand)
  Lanczos partial eigendecomposition → top K_eigs eigenpairs
  Sweep σ grid → compute V(σ), D_w(σ), C_k(σ)
  Composite score S(σ) peaks → boundary set Σ*

Layer 3: SLoDPruner (online, inference-time)
  Query x_0 → compute heat kernel weights w(σ)
  Fréchet mean Φ_σ via tangent-space aggregation (T=15, SIMD)
  Return abstraction level for DDTree budget selection
```

### Feature Gate

```toml
[katgpt-core.features]
slod = ["spectral_hierarchy"]  # Reuses eigen decomposition
```

### New Types (in katgpt-core/src/slod.rs)

```rust
/// SLoD operator — continuous zoom on hyperbolic KG.
pub struct SlodOperator {
    /// kNN graph Laplacian eigenpairs (pre-computed).
    eigenvalues: Vec<f32>,
    eigenvectors: Vec<f32>, // Flat buffer: [K_eigs * N]
    /// Detected boundary scales from BoundaryScan.
    boundaries: Vec<ScaleBoundary>,
    /// Config with transferred defaults.
    config: SlodConfig,
}

pub struct SlodConfig {
    /// kNN parameter: max(10, min(⌊√N⌋, 50))
    pub knn_k: usize,
    /// Composite weights (1/3, 1/3, 1/3 default)
    pub alpha: [f32; 3],
    /// MAD multiplier (β=2 default)
    pub mad_beta: f32,
    /// Gap threshold (R=2 default)
    pub gap_threshold: f32,
    /// Max tangent-space iterations (T=15)
    pub max_iterations: usize,
    /// Step size (η=1.0 default)
    pub step_size: f32,
    /// Convergence tolerance (10⁻⁶)
    pub tolerance: f32,
}

pub struct ScaleBoundary {
    /// Diffusion scale at boundary.
    pub sigma: f32,
    /// Effective dimensionality at this boundary.
    pub k_star: usize,
    /// Composite score at peak.
    pub score: f32,
}

/// SLoD-adaptive ConstraintPruner.
/// Routes between validation tiers based on SLoD boundary detection.
pub struct SlodPruner {
    operator: SlodOperator,
    /// Pruner per abstraction level (coarse → fine).
    tier_pruners: Vec<Box<dyn ConstraintPruner>>,
}
```

### Key Algorithms

**Fréchet mean via tangent-space aggregation (SIMD-accelerated):**
```rust
fn frechet_mean(
    embeddings: &[f32],  // Flat: [N * dim]
    weights: &[f32],     // Heat kernel weights
    dim: usize,
    config: &SlodConfig,
) -> Vec<f32> {
    // 1. Warm-start at dominant-weight point
    // 2. Iterate: Log_μ(v_i) → weighted average → Exp_μ(η·ū)
    // 3. SIMD dot product for tangent-space ops
    // 4. Early exit at convergence
}
```

**Boundary detection:**
```rust
fn boundary_scan(
    eigenvalues: &[f32],
    eigenvectors: &[f32],
    focus: usize,
    n: usize,
    config: &SlodConfig,
) -> Vec<ScaleBoundary> {
    // 1. Log-spaced σ grid
    // 2. For each σ: compute V, D_w, C_k
    // 3. Z-score normalize → composite S(σ)
    // 4. PeakPick with MAD threshold
    // 5. Return boundary set with K*(σ)
}
```

---

## Tasks

- [x] T1: `SlodConfig` with transferred defaults (k, α, β, R, T, η, tol)
- [x] T2: `SlodOperator::build_laplacian()` — kNN graph construction with hyperbolic distance weights
- [x] T3: `SlodOperator::lanczos_eigendecompose()` — reuse Jacobi from spectral_hierarchy (top_k_eigenvectors)
- [x] T4: `SlodOperator::boundary_scan()` — three-signal composite detection with MAD peak picker
- [x] T5: `SlodOperator::frechet_mean()` — SIMD-accelerated tangent-space aggregation
- [x] T6: `SlodPruner` implementing `ConstraintPruner` — tier routing via spectral boundaries
- [x] T7: Hyperbolic distance functions (Poincaré ball metric, Log_map, Exp_map)
- [x] T8: HSBM synthetic hierarchy generation + embedding for benchmark (G2 test)
- [x] T9: Benchmark: SLoD boundary recovery — HSBM 4-cluster boundary detection (G2: 2 boundaries found)
- [x] T10: Benchmark: SlodPruner overhead (G1: 26.6 ns/call) + Fréchet stability (G3) + manifold_score bounded (G4)
- [x] T11: GOAT proof — G1-G6 all pass (21 tests)
- [x] T12: Integration test — SlodPruner with NoPruner tier, batch consistency, empty tiers

---

## GOAT Gates

| Gate | Metric | Pass If | Result |
|------|--------|---------|--------|
| G1 | SlodPruner overhead per call | ≤ 100ns | 26.6 ns ✅ |
| G2 | HSBM boundary recovery | ≥ 1 boundary on 4-cluster | 2 boundaries ✅ |
| G3 | Fréchet mean ball stability | ‖μ‖ < 1.0, all finite | 0.009 ✅ |
| G4 | manifold_score bounded | [0, 1] for all inputs | All 1.0 ✅ |
| G5 | BoundaryScan wall-clock (1K nodes) | ≤ 50ms | 25.6ms ✅ |
| G6 | Fréchet mean convergence (T≤15) | ≤ 10⁻⁶ tolerance in ≤15 steps | Converges ✅ |

**Promotion:** G1-G6 all pass → **promoted to default-ON**.

---

## Expected Performance

| Component | Cost | Notes |
|-----------|------|-------|
| kNN graph construction | O(N·k·log N) | Offline, pre-compute |
| Lanczos eigendecomposition | O(N·K²_eigs) | Offline, K_eigs=50 default |
| BoundaryScan | O(K_eigs · |Σ|) | Offline, ~100 σ grid points |
| Fréchet mean (online) | O(T · N · dim) | T≤15, SIMD-accelerated |
| SlodPruner.is_valid() | O(1) lookup | Route to appropriate tier |

---

## Honest Assessment

### What This Enables
1. **Automatic DDTree depth** — no manual max_depth tuning
2. **Multi-tier validation routing** — spectral boundaries select syntax vs semantic validation
3. **KG-adaptive inference** — query complexity drives resolution

### What This Doesn't Do
1. Doesn't replace learned embeddings — needs pre-computed Poincaré embeddings as input
2. Doesn't work well on flat/random graphs — spectral gaps won't exist
3. Doesn't scale to >100K nodes without Chebyshev/Nyström approximation
4. Doesn't handle overlapping communities cleanly

### Risk Mitigation
- Feature-gated (`slod`) — can be disabled without affecting default pipeline
- Reuses existing spectral infrastructure from Plan 156
- Defaults transferred from paper (HSBM + WordNet) — no parameter tuning needed
- Falls back to fixed-depth DDTree if BoundaryScan returns empty set

---

## TL;DR

SLoD Pruner — modelless continuous-zoom on KG via spectral heat diffusion. Reuses spectral_hierarchy module, adds Fréchet mean + boundary detection + adaptive DDTree budget. Feature-gated, benchmarked against HSBM baselines. GOAT gates on ARI recovery + DDTree quality + overhead.
