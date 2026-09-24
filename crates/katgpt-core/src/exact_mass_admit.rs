//! Exact-mass sigmoid admission — the modelless "sigmoid top-k" operator
//! distilled from MAttr (arXiv:2609.25518 "Matryoshka attribution",
//! Research 584 / Issue 879).
//!
//! Given scores `S`, a mass budget `k`, and a temperature `T`, bisect the
//! threshold `τ` over `[min(S) − 10T, max(S) + 10T]` until
//! `Σᵢ σ((sᵢ − τ)/T) ≈ k`, then emit the soft admission mask
//! `mᵢ = σ((sᵢ − τ)/T) ∈ (0, 1)`. The mask is the *unique* sigmoid family
//! whose total mass is exactly the budget — the differentiable analogue of
//! a hard top-k cut, and the operator MAttr uses as its supervision target
//! for randomized-budget (matryoshka) attribution.
//!
//! # Naming (load-bearing)
//!
//! [`katgpt-spectral`'s `gate_sigmoid_topk_into`] ships **per-expert
//! independent sigmoids plus a selection-sort hard cut** — uncalibrated
//! mass: `Σσ` is whatever it is. This module is the *opposite* semantics
//! one token apart: calibrated mass, `Σm = k` by construction. The two
//! must never share a name; every consumer of exact mass says
//! `exact_mass_admit`, every consumer of hard cut says `gate_sigmoid_topk`.
//!
//! # Properties (pinned in `tests/exact_mass_admit_g1.rs`)
//!
//! - **Sum-to-k**: `|Σmᵢ − k|` is bounded by the bisection tolerance
//!   (≤ `1e-9·n` early-exit + per-element f32 cast, ~`1.2e-7·n` worst case).
//!   The bracket reach bounds the extreme tail: for `k` within
//!   `σ(−10)·n ≈ 4.5e-5·n` of `n` (or of 0), the root sits outside the
//!   bracket and the converged mass is the bracket-end mass — documented,
//!   not clamped.
//! - **Shift-invariance**: `S + c` (power-of-two `c`, dyadic scores)
//!   produces a bit-identical mask and `τ + c` exactly — only relative
//!   scores matter.
//! - **Nestedness in k**: `mask(k+1) ≥ mask(k)` elementwise (τ is strictly
//!   decreasing in k) — the matryoshka property that makes budgets
//!   nestable.
//! - **Zero-alloc**, `O(MAX_BISECT_ITERS · n)` scalar work.
//!
//! # Posture
//!
//! Offline/calibration-tier operator (KV `DensityBudget` ladder boundaries,
//! `thermal_lod` attention_k tier elbows, log-frontier probe scoring) — NOT
//! a hot-path replacement for a hard top-k cut; Bench 884 records the cost
//! honestly. Sigmoid substrate: [`crate::simd::exact_sigmoid_f64`] (the
//! bit-stable libm variant — a calibration primitive wants exactness, not
//! the ~1.2e-7-drift Cephes kernel).
//!
//! Distilled from the paper's published math only (arXiv:2609.25518); the
//! upstream repo `aryamanarora/matryoshka-attribution` carries no license
//! and no code was copied.
//!
//! (Sibling for contrast: katgpt-spectral's `gate_sigmoid_topk_into` — the
//! hard-cut, uncalibrated-mass operator this module must never be confused
//! with.)

use crate::simd::exact_sigmoid_f64;

/// Maximum bisection iterations. The paper uses a fixed 50; f64 resolution
/// or the mass tolerance ends the loop earlier for well-separated scores.
pub const MAX_BISECT_ITERS: usize = 50;

/// Bracket half-width in temperature units: `τ ∈ [min − 10T, max + 10T]`.
/// At the low end every element contributes ≥ σ(10) ≈ 0.99995; at the high
/// end every element contributes ≤ σ(−10) ≈ 4.5e-5.
const BRACKET_T_MULTIPLE: f64 = 10.0;

/// Early-exit mass tolerance, relative to the population: the loop stops
/// once `|Σm − k| ≤ MASS_TOL_REL · n`. At the flattest slope
/// (`dmass/dτ ≈ −n/(4T)`) this pins τ to ~`4·MASS_TOL_REL·T`.
const MASS_TOL_REL: f64 = 1e-9;

/// Exact-mass sigmoid admission, zero-alloc (out-param) form.
///
/// Writes `mᵢ = σ((sᵢ − τ)/T)` into `out` and returns `τ`.
///
/// - `k ≤ 0` → all-zero mask, returns `+∞`.
/// - `k ≥ n` → all-one mask, returns `−∞`.
/// - `temperature ≤ 0` is clamped to `f32::EPSILON` (debug-asserted
///   finite; scores must be finite).
///
/// `out.len()` must equal `scores.len()`.
pub fn exact_mass_admit_into(scores: &[f32], k: f32, temperature: f32, out: &mut [f32]) -> f32 {
    debug_assert_eq!(out.len(), scores.len(), "out must match scores length");
    debug_assert!(k.is_finite(), "k must be finite");
    debug_assert!(
        temperature.is_finite(),
        "temperature must be finite (clamped to EPSILON only when non-positive)"
    );
    debug_assert!(
        scores.iter().all(|s| s.is_finite()),
        "scores must be finite — NaN/inf inputs are a caller bug"
    );

    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    if k <= 0.0 {
        out.fill(0.0);
        return f32::INFINITY;
    }
    if k >= n as f32 {
        out.fill(1.0);
        return f32::NEG_INFINITY;
    }

    let t = temperature.max(f32::EPSILON) as f64;
    let target = k as f64;
    let mass_tol = MASS_TOL_REL * n as f64;

    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &s in scores {
        if s < mn {
            mn = s;
        }
        if s > mx {
            mx = s;
        }
    }
    let mut lo = mn as f64 - BRACKET_T_MULTIPLE * t;
    let mut hi = mx as f64 + BRACKET_T_MULTIPLE * t;

    // Bisection on the strictly-decreasing mass(τ). Branch-free interval
    // updates (arithmetic select): `need_lower == 1` ⟺ mass(mid) < k ⟺ the
    // root sits below mid ⟹ hi = mid; else lo = mid.
    let mut tau = 0.5 * (lo + hi);
    for _ in 0..MAX_BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        tau = mid;
        if !(mid > lo && mid < hi) {
            break; // f64 bisection resolution exhausted
        }
        let mut mass = 0.0f64;
        for &s in scores {
            mass += exact_sigmoid_f64((s as f64 - mid) / t);
        }
        if (mass - target).abs() <= mass_tol {
            break;
        }
        // Branch-free interval select (bool → 0/1 f64 via u32; the machine
        // form is a cmov/select, not a branch): need_lower == 1 ⟺
        // mass(mid) < k ⟺ the root sits below mid ⟹ hi = mid; else
        // lo = mid.
        let need_lower = (mass < target) as u32 as f64;
        lo += (mid - lo) * (1.0 - need_lower);
        hi += (mid - hi) * need_lower;
    }

    for (m, &s) in out.iter_mut().zip(scores) {
        *m = exact_sigmoid_f64((s as f64 - tau) / t) as f32;
    }
    tau as f32
}

/// Allocating convenience wrapper over [`exact_mass_admit_into`] — returns
/// `(mask, τ)` with a fresh `Vec<f32>` (the `gate_sigmoid_topk` /
/// `gate_sigmoid_topk_into` pair shape). The zero-alloc contract belongs to
/// the `_into` form; this one is for setup/calibration code.
pub fn exact_mass_admit(scores: &[f32], k: f32, temperature: f32) -> (Vec<f32>, f32) {
    let mut out = vec![0.0f32; scores.len()];
    let tau = exact_mass_admit_into(scores, k, temperature, &mut out);
    (out, tau)
}
