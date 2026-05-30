# Plan 152: Newton-Schulz Orthogonalization + River-Valley Diagnostics

**Date:** 2026-05-26
**Status:** ✅ COMPLETE
**Source:** Research 114 (AMUSE — Anytime Muon with Stable Gradient Evaluation)
**Related:** Plan 149 (riir-ai — AMUSE game LoRA optimizer), Research 113 (NITP — representation geometry)
**Feature Gates:** `newton_schulz` (default-on), `river_valley` (default-on)
**GOAT Target:** Newton-Schulz convergence ≤5 iters, river-valley diagnostics on D2F mini training
**GOAT Result:** ✅ PROVED 25/25 (Bench 050, 2026-05-30). Both features promoted to **default-on** — GOAT proved, no perf regression, zero-cost until APIs are called.

---

## Task Index

- [x] T1: Newton-Schulz Orthogonalization (feature: `newton_schulz`)
- [x] T2: River-Valley Diagnostic Metrics (feature: `river_valley`)
- [x] T3: Muon Momentum Buffer (feature: `newton_schulz`)
- [x] T4: Wire into D2F Training (feature: `newton_schulz`)

## Goal

Extract two infrastructure components from AMUSE (Research 114) into katgpt-rs:

1. **Newton-Schulz orthogonalization** — a standalone matrix operation that converts any matrix to its nearest orthogonal factor via 5 fixed-point iterations. Generic building block usable by any optimizer or matrix decomposition.

2. **River-valley diagnostic metrics** — dominant/bulk subspace ratio computation, effective rank, and cosine similarity. Modelless diagnostics that reveal why training is (or isn't) converging. No training code changes needed — post-hoc analysis.

These are **infrastructure only** — no AMUSE optimizer. The full AMUSE optimizer (with time-varying β, Schedule-Free averaging) goes in riir-ai Plan 149.

---

## Why katgpt-rs (Open)

Newton-Schulz is a standard matrix operation. River-valley diagnostics are analysis tools. Neither encodes game-specific knowledge. Shipping them in the open engine:
- Provides building blocks for anyone implementing Muon-family optimizers
- Gives diagnostic tools for understanding training dynamics
- Does NOT expose game-specific AMUSE tuning (β₁, ρ, warmup schedules per game domain)

---

## Task Breakdown

- [x] ### T1: Newton-Schulz Orthogonalization (feature: `newton_schulz`)

**File:** `src/newton_schulz.rs`

Implement the 5-iteration Newton-Schulz algorithm for computing the nearest orthogonal matrix:

```rust
/// Newton-Schulz 5-iteration orthogonalization.
/// Converts matrix G to its nearest orthogonal factor X via:
///   X = G / ||G||_F
///   for 5 iters: A = X @ X^T; X = a*X + (b*A + c*A@A) @ X
/// where a=3.4445, b=-4.7750, c=2.0315
pub fn newton_schulz5(g: &[f32], rows: usize, cols: usize, out: &mut [f32])
```

**GOAT proof:**
- Test: Random 64×64 matrix → output is orthogonal (X@X^T ≈ I, max error < 1e-4)
- Test: Convergence in ≤5 iterations for matrices with condition number < 100
- Test: Handle non-square (transpose if rows > cols)
- Bench: 64×64 matmul throughput

**Implementation notes:**
- SIMD-friendly: each iteration is matmul + matmul + element-wise scaling
- No dynamic allocation: work in pre-allocated buffers
- Constants a=3.4445, b=-4.7750, c=2.0315 from the paper (converges for singular values in [0, 1])

- [x] ### T2: River-Valley Diagnostic Metrics (feature: `river_valley`)

**File:** `src/river_valley.rs`

```rust
/// Compute dominant/bulk subspace alignment ratios.
/// Given gradient vector g and top-k Hessian eigenvectors U_k:
///   r_dom = ||U_k^T @ g|| / ||g||
///   r_bulk = sqrt(1 - r_dom^2)
pub fn subspace_ratios(
    gradient: &[f32],
    dominant_eigvecs: &[Vec<f32>],  // top-k eigenvectors
) -> (f32, f32)  // (r_dom, r_bulk)

/// Effective rank of a matrix (entropy of normalized singular values).
pub fn effective_rank(matrix: &[f32], rows: usize, cols: usize) -> f32

/// Average cosine similarity between consecutive updates.
/// AMUSE's key stability metric: high cos(Δx_t, Δx_{t+1}) = smooth trajectory.
pub fn update_cosine_similarity(updates: &[[f32; D]]) -> f32
```

**GOAT proof:**
- Test: Known dominant subspace (random 10-dim, k=3) → r_dom + r_bulk = 1.0
- Test: Effective rank of identity matrix = full rank
- Test: Update cosine similarity of constant direction = 1.0

- [x] ### T3: Muon Momentum Buffer (feature: `newton_schulz`)

**File:** `src/newton_schulz.rs` (extend T1)

```rust
/// Muon-style momentum + orthogonalization step.
/// m = β*m + grad
/// update = newton_schulz5(m) * scaling
pub fn muon_update(
    grad: &[f32],
    momentum: &mut [f32],
    beta: f32,
    rows: usize,
    cols: usize,
    out: &mut [f32],
)
```

**GOAT proof:**
- Test: Muon update on 64×64 matrix produces orthogonal output
- Test: Momentum accumulation: 3 steps with same gradient → increasing magnitude

- [x] ### T4: Wire into D2F Training (feature: `newton_schulz`)

**File:** `src/dllm.rs`

Replace the `sgd_update` in D2F mini training with a Muon-style update for matrix parameters, keeping SGD for scalar parameters. This validates that Newton-Schulz works in our training loop.

**GOAT proof:**
- D2F training with Muon reaches same accuracy in fewer epochs than SGD
- No NaN/Inf in training loop

---

## Feature Gates

```toml
[features]
newton_schulz = []      # Newton-Schulz orthogonalization + Muon momentum — default-ON (GOAT 25/25)
river_valley = []       # River-valley diagnostic metrics — default-ON (GOAT 25/25)
```

Both are default-on after GOAT proof passed (Bench 050, 25/25). Zero-cost until APIs are called.

---

## GOAT Summary

| Test | Criterion | Target |
|------|-----------|--------|
| T1.1 | Orthogonality | X@X^T ≈ I (max error < 1e-4) |
| T1.2 | Convergence | ≤5 iterations |
| T1.3 | Non-square | Correct transpose handling |
| T2.1 | Subspace ratios | r_dom + r_bulk = 1.0 |
| T2.2 | Effective rank | Known matrix → correct rank |
| T2.3 | Cosine similarity | Constant direction = 1.0 |
| T3.1 | Muon output | Orthogonal |
| T3.2 | Momentum | Accumulating |
| T4.1 | D2F convergence | Same accuracy, fewer epochs |

**Estimated effort:** 2-3 days for T1-T3. T4 is 0.5 day wiring.

---

## What This Does NOT Do

- No AMUSE optimizer (that's riir-ai Plan 149)
- No Schedule-Free averaging (that's riir-ai Plan 149)
- No time-varying β (that's riir-ai Plan 149)
- No game-specific tuning (that's riir-ai Plan 149)
- No GPU compute kernel (future: riir-gpu WGSL kernel for Newton-Schulz)

This plan ships the **open building blocks**. The **secret sauce** (how to combine them into AMUSE with game-specific tuning) stays in riir-ai.
