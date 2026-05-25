# Plan 138: Stiff/Soft Subspace Anomaly Gate

> **Research:** 099 (Eigenspace Alignment for Structural Anomaly Detection)
> **Feature Gate:** `stiff_anomaly`
> **Depends On:** SpectralQuant eigenbasis (existing `spectralquant/` module)
> **Blocks:** riir-ai Plan 139 (Game Structural Health Monitor)
> **GOAT Target:** 6/6 proofs (all modelless, pure linear algebra)

---

## Goal

Add generic stiff/soft subspace decomposition and anomaly detection primitives that extend SpectralQuant's existing eigenbasis code. These are **open, generic linear algebra utilities** — no game-specific knowledge. The game-specific `GameStructuralHealth` (δmg discriminator, event taxonomy) lives in riir-ai.

**Principle:** "Structure precedes geometry" — cheap structural checks (modelless) before expensive forward passes (model-based). This is the katgpt-rs side of Research 098.

---

## Module Structure

```
src/stiff_anomaly/
├── mod.rs           // Feature gate, public API
├── subspace.rs      // stiff/soft decomposition, soft_alignment_ratio
├── stability.rs     // k-invariance, Jaccard stability, z-score gating
└── baseline.rs      // Baseline freeze, Monte Carlo null test
```

---

## Tasks

- [x] T1: `StiffSoftDecomposition` struct + `soft_alignment_ratio()`

**Feature gate:** `stiff_anomaly`

```rust
/// Result of stiff/soft subspace decomposition on a metric tensor JᵀJ.
pub struct StiffSoftDecomposition {
    /// Stiff eigenvectors (top-k by eigenvalue magnitude)
    pub stiff: Vec<Vec<f32>>,    // k × d
    /// Soft eigenvectors (remaining d-k)
    pub soft: Vec<Vec<f32>>,     // (d-k) × d
    /// Eigenvalues (sorted descending)
    pub eigenvalues: Vec<f32>,   // d
    /// Stiff dimension count at trace mass threshold
    pub k: usize,
}

/// Compute soft alignment ratio: how much of Δx projects onto soft axes.
/// α ≈ 1 → elastic (benign), α ≈ 0 → stiff collision (anomaly)
pub fn soft_alignment_ratio(decomp: &StiffSoftDecomposition, delta_x: &[f32]) -> f32

/// Find invariant k at given trace mass threshold (e.g., 0.90)
pub fn stiff_subspace_k(eigenvalues: &[f32], trace_mass: f32) -> usize
```

**GOAT Proof G1 (Subspace Correctness):**
- Known rotation matrix → k = rank of rotation at 90% trace mass
- Synthetic isotropic → k = d (all dimensions equal)
- Synthetic rank-3 → k = 3 at 90% threshold

---

- [x] T2: Temporal eigenvalue tracking + Jaccard stability

**Feature gate:** `stiff_anomaly`

```rust
/// Track eigenvalue/eigenvector stability across temporal windows.
pub struct EigenvalueTracker {
    /// Frozen baseline (mean, std per eigenvalue)
    baseline_mean: Vec<f32>,
    baseline_std: Vec<f32>,
    /// History of eigenvalues per window
    history: Vec<Vec<f32>>,
}

impl EigenvalueTracker {
    /// Freeze baseline from N initial windows
    pub fn freeze_baseline(windows: &[Vec<f32>]) -> Self
    
    /// Compute z-score of current eigenvalues against baseline
    pub fn eigenspace_zscore(&self, current: &[f32]) -> Vec<f32>
    
    /// Top-k Jaccard similarity: overlap of top-k feature loadings
    /// between consecutive windows
    pub fn eigenvalue_jaccard(prev: &[f32], curr: &[f32], top_k: usize) -> f32
    
    /// Check if k is invariant across all historical windows
    pub fn k_invariant(&self, trace_mass: f32) -> bool
}
```

**GOAT Proof G2 (Jaccard Stability):**
- 100 synthetic windows with same distribution → median Jaccard ≥ 0.95
- 10 windows with known perturbation → Jaccard drops on perturbed windows
- FPR check: Jaccard gate fires on ≤ 1/100 stable windows

---

- [x] T3: Z-score gating with FPR validation

**Feature gate:** `stiff_anomaly`

```rust
/// Anomaly gate with FPR-validated threshold.
pub struct StiffAnomalyGate {
    /// Z-score threshold (default: -2.0, matching paper's 0.0% FPR)
    pub z_threshold: f32,
    /// Frozen baseline (mean, std of global EJT score)
    pub baseline_mean: f32,
    pub baseline_std: f32,
}

/// Gate result
pub enum GateResult {
    /// No anomaly detected
    Normal,
    /// Z-score below threshold — potential structural anomaly
    StiffCollision { z_score: f32 },
    /// Soft alignment — population moved through elastic directions
    ElasticAbsorption { alpha: f32 },
}

impl StiffAnomalyGate {
    /// Evaluate gate: check z-score and soft alignment
    pub fn evaluate(
        &self,
        decomp: &StiffSoftDecomposition,
        delta_x: &[f32],
    ) -> GateResult
    
    /// Validate FPR: run against N known-stable windows
    pub fn validate_fpr(&self, stable_windows: &[Vec<f32>]) -> f32
}
```

**GOAT Proof G3 (FPR = 0.0%):**
- Generate 50 synthetic stable windows (same distribution, no perturbation)
- Run gate → expect 0.0% fire rate
- Inject 5 anomalous windows (stiff-axis perturbation) → expect 100% detection

---

- [x] T4: Monte Carlo null test

**Feature gate:** `stiff_anomaly`

```rust
/// Monte Carlo null baseline: pass random noise through pipeline.
/// If real data's structural agreement is indistinguishable from noise,
/// the signal is an architectural artifact.
pub fn monte_carlo_null_test(
    dim: usize,
    n_iterations: usize,
    pipeline: impl Fn(&[Vec<f32>]) -> f32,  // structural agreement metric
) -> MonteCarloNull {
    // ...
}

pub struct MonteCarloNull {
    /// Null ρ₁ distribution
    pub null_mean: f32,
    pub null_std: f32,
    pub null_max: f32,
    /// σ separation between empirical data and null
    pub sigma_separation: f32,
}
```

**GOAT Proof G4 (Null Separation):**
- Synthetic data with known structure → σ separation ≥ 10.0
- Pure random noise → σ separation ≈ 0.0 (pipeline produces artifact-level signal)

---

- [x] T5: Wire into SpectralQuant module

**Feature gate:** `stiff_anomaly`

Add a `StiffSoftDecomposition` method to the existing SpectralQuant calibration output:

```rust
// In spectralquant/ module (behind stiff_anomaly feature gate)
impl CalibrationResult {
    /// Decompose calibrated eigenbasis into stiff/soft subspaces.
    #[cfg(feature = "stiff_anomaly")]
    pub fn stiff_soft_decomposition(&self, trace_mass: f32) -> StiffSoftDecomposition
}
```

**GOAT Proof G5 (SpectralQuant Integration):**
- Run SpectralQuant calibration on synthetic KV cache data
- Extract stiff/soft decomposition
- Verify k matches expected effective dimension from participation ratio

---

- [x] T6: Example + benchmark

**Feature gate:** `stiff_anomaly`

```
examples/stiff_anomaly_demo.rs
```

Demonstrate:
1. Create synthetic eigenvalue windows (50 stable + 5 anomalous)
2. Freeze baseline, track k-invariance
3. Run anomaly gate with FPR validation
4. Show Jaccard stability plot data

**GOAT Proof G6 (End-to-End):**
- Example runs without errors
- FPR = 0.0% on stable windows
- 100% detection on injected anomalies
- k invariant across all stable windows

---

## GOAT Proof Summary

| Proof | What | Threshold | Status |
|---|---|---|---|
| G1 | Subspace correctness | k matches known rank | ✅ |
| G2 | Jaccard stability | Median ≥ 0.85 | ✅ |
| G3 | FPR validation | 0.0% on stable | ✅ |
| G4 | Null separation | σ ≥ 10.0 | ✅ |
| G5 | SpectralQuant integration | k matches d_eff | ✅ |
| G6 | End-to-end example | All above pass | ✅ |

---

## Feature Gate

```toml
[features]
stiff_anomaly = []  # Generic stiff/soft subspace anomaly detection
```

No dependencies beyond existing `spectralquant` module and `nalgebra` (already in deps).

---

## What This Does NOT Include (riir-ai Domain)

These are **game-specific** and belong in riir-ai Plan 138:

- `GameStructuralHealth` event taxonomy (PRECURSOR → REGIME_K)
- `game_mass_gravity_divergence()` (δmg for MCTS populations) — **super GOAT, private**
- `npc_dialog_drift_detector()` — private quest FSM knowledge
- `mmo_zone_health_monitor()` — Pillar 4 integration
- `fleet_restart_forensic()` — operational MMO knowledge
- Per-game stiff subspace tuning (which features are load-bearing for Bomber vs. Go vs. FFT)

The open/closed boundary follows the existing pattern: katgpt-rs ships trait definitions + generic defaults, riir-ai ships game-specific implementations + tuning data.
