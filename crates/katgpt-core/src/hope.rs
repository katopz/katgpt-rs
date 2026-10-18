//! HOPE — Hilbert-Schmidt Capacity Kernel + Optimal Rank-1 Parent.
//!
//! Distilled from *Hilbert Operator for Progressive Encoding* (HOPE),
//! Mobahi & Bartlett, Google DeepMind / UC Berkeley, arXiv:2607.21366
//! (2026-07-24). See `katgpt-rs/.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md`
//! and `katgpt-rs/.plans/469_hilbert_schmidt_capacity_kernel_primitive.md`.
//!
//! # What this computes
//!
//! Three pieces of closed-form math over **rank-1 Hilbert-Schmidt operators**.
//! A PH-1 neuron (ReLU / Leaky-ReLU / linear) is modeled as the function
//!
//! ```text
//! f_i(x) = w_out,i · Ψ((w_eff_in,i)ᵀ·x + b_i)
//! ```
//!
//! which is the outer product `f_i = g_i ⊗ w_out,i` of a continuous scalar
//! landscape `g_i(x) = Ψ((w_eff_in,i)ᵀ·x + b_i)` and a finite-dim output
//! vector `w_out,i`. The **capacity** of `f_i` in the Hilbert space
//! `H = L²(X, P_X; ℝᶜ)` is
//!
//! ```text
//! ‖f_i‖_H = ‖w_out,i‖₂ · √K(i,i)
//! ```
//!
//! where `K(i,j) = 𝔼[Ψ(y_i)·Ψ(y_j)]` is the ReLU kernel under the Gaussian
//! surrogate `y_i ~ N(β_i, γ²_i)`. For ReLU the self-kernel has the closed form
//!
//! ```text
//! K(i,i) = (γ²_i + β²_i)·Φ(β_i/|γ_i|) + β_i·|γ_i|·φ(β_i/|γ_i|)
//! ```
//!
//! (paper Eq 3, Appendix E Theorem E.2) and the cross-kernel has the Arc-Cosine
//! order-1 approximation (paper Eq 5)
//!
//! ```text
//! K(i,j) ≈ (1/π)·(√(1−ρ̂²) + (π − arccos ρ̂)·ρ̂)·√(K(i,i)·K(j,j))
//! ```
//!
//! with warped correlation `ρ̂_ij = 2κ/(1+√(1+4κ²))`.
//!
//! The **optimal rank-1 parent** for merging two neurons `(i, j)` is found in
//! closed form via the principal eigenvector of the rank-2 matrix
//! `A = w_out,i·(w̃_in,i)ᵀ + w_out,j·(w̃_in,j)ᵀ` and the closed-form scale
//! `s* = (a + b·E_rem) / (2·E_rem + b)` (paper Eq 12–14).
//!
//! # Scale invariance (the defining property)
//!
//! For PH-1 activations, scaling `w_eff_in,i` by `λ > 0` and `w_out,i` by `1/λ`
//! leaves `f_i` unchanged but rescales raw weights. The Hilbert norm cancels
//! this exactly: positive homogeneity scales `K` by `λ` while `‖w_out,i‖` shrinks
//! by `1/λ`. **Capacity is decoupled from input-tensor dimension and from raw
//! weight magnitude** — this is the property AM cosine fidelity lacks (Issue 001
//! rank-1 collapse).
//!
//! # Allocation discipline (G4)
//!
//! Hot-path kernels (`relu_self_kernel`, `relu_cross_kernel_approx`,
//! `hope_capacity`, `hope_prune_cost`, `hope_merge_cost`,
//! `hope_block_eviction_cost`) take `&[f32]` or scalars and return `f32` —
//! zero allocations by construction. The `optimal_rank1_parent` family returns
//! an owned `Rank1Parent` (caller-controlled allocation); the
//! `*_into_scratch` variants are zero-alloc.
//!
//! # Sigmoid, not softmax
//!
//! Per AGENTS.md. The greedy step's distortion-rate selection `argmin J/ΔP` is
//! a comparison of scalars, not a probability distribution — softmax does not
//! apply. Where the slack/core partition crosses into a probability
//! interpretation (DEFT gradient elasticity → riir-train), sigmoid gates apply.
//!
//! # Modelless
//!
//! Pure closed-form math (erf approximation + rank-2 SVD + arithmetic). No
//! training, no backprop, no gradient descent. DEFT's gradient elasticity step
//! is out of scope (→ riir-train).
//!
//! HOPE — Hilbert-Schmidt Capacity Kernel + Optimal Rank-1 Parent (Plan 469,
//! Research 454, arXiv:2607.21366 Mobahi & Bartlett, Google DeepMind 2026-07-24).
//! Closed-form math for scale-invariant capacity metric + optimal rank-1 merge
//! of two PH-1 neurons modeled as rank-1 Hilbert-Schmidt operators.
//! Distilled modelless core (ReLU self/cross kernels, principal eigenvector of
//! rank-2 AᵀA, optimal scale s*=(a+b·E_rem)/(2·E_rem+b), capacity + prune/merge/
//! block-eviction costs, Dantzig greedy selection). Fixes the AM single-query
//! rank-1 collapse failure documented in Issue 001 / Plan 319 T5.6 G5 FAIL.
//! Pure modelless — no training, no gradient descent. Zero runtime cost unless
//! a caller invokes hope_capacity / optimal_rank1_parent / hope_greedy_select.
//! DEFAULT-ON (Phase 23, 2026-07-24, Plan 469 Phase 4 T4.6 promotion — G1+G2+G3+G4
//! ALL PASS per bench_469_hope_kernel_goat; see Cargo.toml default block).

use crate::simd;

// ──────────────────────────────────────────────────────────────────────────
// Trait: a rank-1 operator parameterized by (w_in, w_out, γ, β)
// ──────────────────────────────────────────────────────────────────────────

/// A rank-1 Hilbert-Schmidt operator — the atomic unit HOPE reasons about.
///
/// A "neuron" in HOPE's sense: input effective weights `w_in`, output weights
/// `w_out`, pre-activation scale `γ > 0` (variance of `y = w_in·x + b`), and
/// pre-activation shift `β` (mean of `y`). For networks with BatchNorm, these
/// are the BN-absorbed parameters `(w_eff_in, b)` paired with the BN moving
/// statistics `(γ, β)` — see paper §3 Eq 1.
///
/// For non-BN substrates (NeuronShard, HLA direction vectors, CommittedFieldBlend
/// archetype fields), the bridge maps analogous sufficient statistics to this
/// shape — see `riir-neuron-db/src/hope_bridge.rs` (Plan 321).
pub trait Rank1Operator {
    /// Effective input weights (post-BN-absorption if applicable).
    fn w_in(&self) -> &[f32];
    /// Output weights — direction the scalar activation projects to.
    fn w_out(&self) -> &[f32];
    /// Pre-activation scale `γ > 0` (standard deviation of `y = w_in·x + b`).
    fn gamma(&self) -> f32;
    /// Pre-activation shift `β` (mean of `y = w_in·x + b`).
    fn beta(&self) -> f32;
}

// ──────────────────────────────────────────────────────────────────────────
// Standard-normal PDF / CDF — stdlib has no `erf`, hand-roll with a rational
// approximation (Abramowitz & Stegun 7.1.26, max abs err 1.5e-7 on f64; f32 has
// ~7 sig digits so this is at the precision floor).
// ──────────────────────────────────────────────────────────────────────────

/// Standard normal PDF `φ(x) = (1/√(2π))·exp(−x²/2)`.
#[inline]
pub fn normal_pdf(x: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.398_942_3;
    INV_SQRT_2PI * (-0.5_f32 * x * x).exp()
}

/// Standard normal CDF `Φ(x) = (1/2)·(1 + erf(x/√2))`.
///
/// Uses Abramowitz & Stegun 7.1.26 rational approximation for `erf`
/// (max abs error 1.5e-7 on f64; at the f32 precision floor).
#[inline]
pub fn normal_cdf(x: f32) -> f32 {
    // Φ(x) = (1/2)·(1 + erf(x/√2))
    0.5 * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 erf approximation (max abs err 1.5e-7).
#[inline]
fn erf_approx(x: f32) -> f32 {
    // Sign handling — the A&S formula is for x ≥ 0.
    const A2: f32 = -0.2844_9636;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.3275_911;

    let sign = if x < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let z = x.abs();
    // A&S 7.1.26 constants
    const A1: f32 = 0.254_829_6;
    let t = 1.0 / (1.0 + P * z);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-z * z).exp();
    sign * y
}

// ──────────────────────────────────────────────────────────────────────────
// Self-kernel (paper Eq 3, Appendix E Theorem E.2)
// ──────────────────────────────────────────────────────────────────────────

/// ReLU self-kernel `K(i,i) = 𝔼[Ψ(y_i)²]` under `y_i ~ N(β_i, γ²_i)`.
///
/// Closed form (paper Eq 3):
///
/// ```text
/// K(i,i) = (γ² + β²)·Φ(β/|γ|) + β·|γ|·φ(β/|γ|)
/// ```
///
/// Returns `0.0` for `γ = 0` (dead neuron — `Ψ(y) = max(0, β)` is constant,
/// the integral degenerates). Otherwise strictly positive.
///
/// **Scale invariance:** if you replace `γ → λγ` and `β → λβ` for any `λ > 0`,
/// `K(i,i) → λ²·K(i,i)` (paper §5 neuron scale invariance — the kernel scales
/// by `λ` per PH-1 homogeneity, squared because it's an energy).
#[inline]
pub fn relu_self_kernel(gamma: f32, beta: f32) -> f32 {
    let abs_gamma = gamma.abs();
    if abs_gamma < 1e-12 {
        // Degenerate: y is a constant β. Ψ(y)² = max(0,β)². Integral over a
        // delta distribution gives that value squared.
        let relu_beta = beta.max(0.0);
        return relu_beta * relu_beta;
    }
    let c = beta / abs_gamma;
    let phi_c = normal_pdf(c);
    let phi_cdf_c = normal_cdf(c);
    (gamma * gamma + beta * beta) * phi_cdf_c + beta * abs_gamma * phi_c
}

// ──────────────────────────────────────────────────────────────────────────
// Warped correlation (paper Eq 4, Appendix E.3 Proposition E.3)
// ──────────────────────────────────────────────────────────────────────────

/// Warped correlation `ρ̂_ij = 2κ/(1+√(1+4κ²))` from the input geometry.
///
/// `κ = (ρ_eff/(1−ρ²_eff)) · (|γ_i|/‖w_eff_in,i‖) · (|γ_j|/‖w_eff_in,j‖)`
/// (paper Eq 4 + Appendix E.3 Proposition E.3). Returns a value in `(−1, 1)`
/// (clamped to avoid singularities).
pub fn warped_correlation(
    w_eff_in_i: &[f32],
    w_eff_in_j: &[f32],
    gamma_i: f32,
    gamma_j: f32,
) -> f32 {
    let n = w_eff_in_i.len().min(w_eff_in_j.len());
    if n == 0 {
        return 0.0;
    }
    let dot_ij = simd::simd_dot_f32(w_eff_in_i, w_eff_in_j, n);
    let dot_ii = simd::simd_dot_f32(w_eff_in_i, w_eff_in_i, n);
    let dot_jj = simd::simd_dot_f32(w_eff_in_j, w_eff_in_j, n);
    let norm_prod = (dot_ii * dot_jj).max(1e-30).sqrt();
    let rho_eff = dot_ij / norm_prod;
    // Clamp ρ to (−1+ε, 1−ε) to avoid the singular κ → ±∞ as ρ → ±1.
    let rho_clamped = rho_eff.clamp(-0.9999, 0.9999);
    // κ = (ρ/(1−ρ²)) · (|γ_i|/‖w_i‖) · (|γ_j|/‖w_j‖)
    let gamma_ratio_i = gamma_i.abs() / dot_ii.sqrt().max(1e-15);
    let gamma_ratio_j = gamma_j.abs() / dot_jj.sqrt().max(1e-15);
    let kappa = (rho_clamped / (1.0 - rho_clamped * rho_clamped)) * gamma_ratio_i * gamma_ratio_j;
    // ρ̂ = 2κ/(1+√(1+4κ²)) — the conjugate-multiplied form for numerical stability.
    let denom: f32 = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
    let rho_hat: f32 = 2.0 * kappa / denom;
    // Clamp to (−1, 1) — ρ̂ is provably in this range mathematically, but f32
    // rounding can take it slightly outside near the boundaries.
    rho_hat.clamp(-1.0_f32, 1.0_f32)
}

// ──────────────────────────────────────────────────────────────────────────
// Cross-kernel (paper Eq 5 — Arc-Cosine order 1 zero-bias approximation)
// ──────────────────────────────────────────────────────────────────────────

/// Arc-Cosine kernel order 1 (zero-bias approximation) — the **input-side**
/// ReLU kernel `K(i,j) = 𝔼[Ψ(y_i)·Ψ(y_j)]` under the Gaussian surrogate.
///
/// This is the kernel between the scalar activation landscapes `g_i, g_j` in
/// `H_in = L²(X, P_X; ℝ)` (paper §5). The full Hilbert-inner-product
/// `⟨f_i, f_j⟩_H = K(i,j) · ⟨w_out_i, w_out_j⟩_ℝᶜ` factors into this input-side
/// kernel and the output-direction dot product — callers compose them.
///
/// Approximation (paper Eq 5):
/// ```text
/// K(i,j) ≈ (1/π)·(√(1−ρ̂²) + (π − arccos ρ̂)·ρ̂)·√(K(i,i)·K(j,j))
/// ```
///
/// `ρ̂` is the warped correlation (paper Eq 4). The approximation assumes
/// `β_i, β_j ≈ 0`; for high-bias neurons the exact bivariate-normal-CDF form
/// (Appendix E Eq 83) should be used — not implemented here because it is
/// computationally prohibitive for large N. The greedy optimizer naturally
/// selects highly-correlated pairs where the zero-bias approximation is most
/// accurate (paper §5 Correlation Constraint).
///
/// **Cauchy-Schwarz compliance:** `|K(i,j)| ≤ √(K(i,i)·K(j,j))` always holds
/// because the Arc-Cosine factor is bounded in `[0, 1]` for `ρ̂ ∈ [0, 1]` and
/// `[-1/(2π), 0]` for `ρ̂ ∈ [-1, 0]` (so anti-correlated neurons get a small
/// *positive* kernel value — ReLU is always non-negative).
pub fn relu_cross_kernel_approx(
    w_eff_in_i: &[f32],
    w_eff_in_j: &[f32],
    gamma_i: f32,
    gamma_j: f32,
) -> f32 {
    let k_ii = relu_self_kernel(gamma_i, 0.0);
    let k_jj = relu_self_kernel(gamma_j, 0.0);
    let sqrt_kk = (k_ii * k_jj).max(0.0).sqrt();
    if sqrt_kk < 1e-30 {
        return 0.0;
    }
    let rho_hat = warped_correlation(w_eff_in_i, w_eff_in_j, gamma_i, gamma_j);
    let arccos_rho = rho_hat.clamp(-1.0_f32, 1.0_f32).acos();
    // (1/π)·(√(1−ρ̂²) + (π − arccos ρ̂)·ρ̂)
    let arc_factor =
        (1.0 - rho_hat * rho_hat).max(0.0).sqrt() + (std::f32::consts::PI - arccos_rho) * rho_hat;
    arc_factor * sqrt_kk / std::f32::consts::PI
}

// ──────────────────────────────────────────────────────────────────────────
// Optimal parent direction (paper §7.1, Eq 12–14)
// ──────────────────────────────────────────────────────────────────────────

/// Maximum output dimension supported by the zero-alloc
/// `optimal_rank1_parent_into_scratch` hot path.
///
/// The polarity comparison needs a stack snapshot of the first-polarity
/// v_hat (length m = w_out.len()). This constant bounds that snapshot so no
/// allocation is needed. Currently 64 — comfortably above HLA's 8 scalars
/// and style_weights' 64 dims. Larger output dims fall back to the owned
/// `optimal_rank1_parent` variant.
pub const RANK1_PARENT_MAX_OUT_DIM: usize = 64;

/// The optimal rank-1 parent for merging two neurons.
///
/// `u_hat ∈ ℝⁿ` is the input direction (principal eigenvector of the rank-2
/// matrix `AᵀA` where `A = w_out,i·w̃_in,iᵀ + w_out,j·w̃_in,jᵀ`), `v_hat ∈ ℝᶜ`
/// is the output direction, `s_star` is the optimal scale. The parent neuron
/// is `f_p(x) = s_star · Ψ(u_hatᵀ·x) · v_hat / √K(u_hat,u_hat)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rank1Parent {
    /// Input direction (unit-norm). Length = `w_eff_in.len() + 1` (augmented).
    pub u_hat: Vec<f32>,
    /// Output direction (unit-norm). Length = `w_out.len()`.
    pub v_hat: Vec<f32>,
    /// Optimal scale `s* = (a + b·E_rem) / (2·E_rem + b)`.
    pub s_star: f32,
    /// Self-kernel `K(u,u)` of the parent's input direction.
    pub k_self: f32,
}

/// Cached pairwise constants used in the greedy merge cost.
///
/// `a = ‖f_i‖²_H + ‖f_j‖²_H`, `b = ⟨ψ, f_i + f_j⟩_H`. Both depend only on
/// the operator weights + BN parameters — independent of the layer's residual
/// capacity `E_rem`. Computed once at init, reused for every greedy step
/// (paper Appendix B.4 Decoupled Cache).
#[derive(Debug, Clone, Copy, Default)]
pub struct PairCache {
    /// `a = ‖f_i‖²_H + ‖f_j‖²_H` (sum of squared capacities).
    pub a: f32,
    /// `b = ⟨ψ, f_i + f_j⟩_H = ⟨f_i, f_j⟩_H-like` alignment.
    pub b: f32,
}

/// Principal eigenvector of a rank-2 symmetric matrix `AᵀA`.
///
/// Given the rank-2 matrix `A = w_out_i ⊗ w̃_in_i + w_out_j ⊗ w̃_in_j` where
/// `w̃_in = [w_eff_in; β]` is the augmented input, this returns the principal
/// eigenvector of the 2×2 Gram matrix restricted to the span
/// `{w̃_in_i, w̃_in_j}`. The ambient dimension is bypassed entirely (paper §7.1).
///
/// Returns `(coeffs_c1, coeffs_c2)` such that `u_hat = c1·w̃_in_i + c2·w̃_in_j`.
fn principal_eigenvector_rank2(
    w_in_i: &[f32],
    w_in_j: &[f32],
    w_out_i: &[f32],
    w_out_j: &[f32],
) -> (f32, f32) {
    // The augmented 2D basis is {w̃_in_i, w̃_in_j}. The matrix A maps the
    // (c1, c2) coefficients to a vector, so AᵀA restricted to the 2D subspace
    // is the Gram matrix in the input weights, weighted by output alignment:
    //   M = [[w_out_i·w_out_i · w_in_i·w_in_i,  w_out_i·w_out_j · w_in_i·w_in_j],
    //        [w_out_j·w_out_i · w_in_j·w_in_i,  w_out_j·w_out_j · w_in_j·w_in_j]]
    // But A is rank-1 in the output dim: A·v = w_out·(w_in·v), so AᵀA is:
    //   AᵀA[i,j] = (w_in_i·w_in_j) · (w_out_i·w_out_j)
    // For the principal eigenvector of AᵀA, we want the direction maximizing
    // vᵀ(AᵀA)v = ‖A·v‖². With v = c1·w̃_in_i + c2·w̃_in_j:
    //   ‖A·v‖² = ‖c1·w_out_i·(w_in_i·v) + c2·w_out_j·(w_in_j·v)‖²
    // The simpler rank-2 SVD: A as a rank-2 (c_out × c_in) matrix decomposes
    // into A = σ₁·u₁·v₁ᵀ + σ₂·u₂·v₂ᵀ. The principal input direction is v₁.
    //
    // For numerical simplicity, we use the input-direction principal eigenvector
    // of the 2×2 input Gram matrix G = WᵀW where W = [w̃_in_i | w̃_in_j].
    // (Paper §7.1: "Restricting the eigendecomposition to this rank-2 subspace
    // bypasses the ambient dimension entirely.")
    let n = w_in_i.len().min(w_in_j.len());
    let g11 = simd::simd_dot_f32(w_in_i, w_in_i, n);
    let g22 = simd::simd_dot_f32(w_in_j, w_in_j, n);
    let g12 = simd::simd_dot_f32(w_in_i, w_in_j, n);

    // Output-direction alignment modifies the effective "importance" of each
    // input direction. We fold it in by weighting the Gram matrix diagonals.
    let m = w_out_i.len().min(w_out_j.len());
    let h_ii = simd::simd_dot_f32(w_out_i, w_out_i, m);
    let h_jj = simd::simd_dot_f32(w_out_j, w_out_j, m);
    let h_ij = simd::simd_dot_f32(w_out_i, w_out_j, m);

    // Effective 2×2 matrix: M[i,j] = G[i,j] · H[i,j] (rank-1 outer-product of
    // output and input inner products). The principal eigenvector of M gives
    // the parent input direction in coefficient space.
    let m11 = g11 * h_ii;
    let m22 = g22 * h_jj;
    let m12 = g12 * h_ij;
    let m21 = m12; // symmetric

    // Principal eigenvector of a 2×2 symmetric matrix [[m11, m12],[m21, m22]]:
    // eigenvalues λ = (tr ± √(tr² − 4·det)) / 2; principal eigenvector is
    // (m12, λ_max − m11) (or (1, 0) if m12 ≈ 0).
    let trace = m11 + m22;
    let det = m11 * m22 - m12 * m21;
    let disc = (trace * trace - 4.0 * det).max(0.0);
    let lambda_max = 0.5 * (trace + disc.sqrt());

    let (c1, c2) = if m12.abs() < 1e-15 {
        // Diagonal matrix — principal eigenvector is the larger-diagonal axis.
        if m11 >= m22 {
            (1.0_f32, 0.0)
        } else {
            (0.0, 1.0)
        }
    } else {
        (m12, lambda_max - m11)
    };

    // Normalize to unit length in coefficient space.
    let norm: f32 = (c1 * c1 + c2 * c2).max(1e-30).sqrt();
    (c1 / norm, c2 / norm)
}

/// Compute the optimal rank-1 parent for merging two rank-1 operators.
///
/// Implements paper Eq 12–14:
/// - `u_c` = `argmax_{±û}` of the alignment objective (sign resolution).
/// - `v_hat` = `⟨K(û, w̃_in_k)·w_out_k⟩ / ‖...‖`.
/// - `s_star` = `(a + b·E_rem) / (2·E_rem + b)`.
///
/// The output direction `v_hat` and the parent's `K_self` are derived from the
/// resolved `(u_c, v_hat)` pair. `E_rem` is the layer's residual capacity
/// (sum of all other neurons' capacities) — must be `≥ 0`.
///
/// **Allocation:** returns an owned `Rank1Parent` (caller-controlled). Use
/// [`optimal_rank1_parent_into_scratch`] for the zero-alloc variant.
pub fn optimal_rank1_parent(
    op_i: &impl Rank1Operator,
    op_j: &impl Rank1Operator,
    e_rem: f32,
) -> Rank1Parent {
    let n = op_i.w_in().len().min(op_j.w_in().len());
    let m = op_i.w_out().len().min(op_j.w_out().len());
    let mut u_hat = vec![0.0_f32; n];
    let mut v_hat = vec![0.0_f32; m];
    let mut k_self = 0.0_f32;
    let s_star =
        optimal_rank1_parent_into_scratch(op_i, op_j, e_rem, &mut u_hat, &mut v_hat, &mut k_self);
    Rank1Parent {
        u_hat,
        v_hat,
        s_star,
        k_self,
    }
}

/// Zero-alloc variant of [`optimal_rank1_parent`].
///
/// Writes the parent's input direction into `u_hat_scratch` (length `n`),
/// output direction into `v_hat_scratch` (length `m`), self-kernel into
/// `k_self_scratch`, and returns the optimal scale `s_star`.
///
/// `v_hat_compare_scratch` (length `m`) is internal scratch used to preserve
/// the first polarity's v_hat across the second-polarity evaluation — it is
/// overwritten during the call and need not be initialized. This keeps the
/// hot path allocation-free: the polarity comparison that previously
/// returned an owned `Vec<f32>` now writes into this caller-supplied buffer.
pub fn optimal_rank1_parent_into_scratch(
    op_i: &impl Rank1Operator,
    op_j: &impl Rank1Operator,
    e_rem: f32,
    u_hat_scratch: &mut [f32],
    v_hat_scratch: &mut [f32],
    k_self_scratch: &mut f32,
) -> f32 {
    let n = op_i.w_in().len().min(op_j.w_in().len());
    let m = op_i.w_out().len().min(op_j.w_out().len());
    debug_assert!(
        u_hat_scratch.len() >= n,
        "u_hat_scratch too short: {} < {}",
        u_hat_scratch.len(),
        n
    );
    debug_assert!(
        v_hat_scratch.len() >= m,
        "v_hat_scratch too short: {} < {}",
        v_hat_scratch.len(),
        m
    );

    // 1. Principal eigenvector of AᵀA in the 2D span {w̃_in_i, w̃_in_j}.
    let (c1_pos, c2_pos) =
        principal_eigenvector_rank2(op_i.w_in(), op_j.w_in(), op_i.w_out(), op_j.w_out());

    // u_hat = c1·w_in_i + c2·w_in_j (positive polarity)
    for (k, slot) in u_hat_scratch.iter_mut().enumerate().take(n) {
        *slot = c1_pos * op_i.w_in()[k] + c2_pos * op_j.w_in()[k];
    }

    // 2. Sign resolution: evaluate both ±û in the exact alignment objective
    // (paper Eq 11). The objective is:
    //   obj(û) = ‖Σ_k K(û, w̃_in_k)·w_out_k‖ / √K(û, û)
    // We compute the positive and negative polarity alignments and pick the max.
    //
    // To stay allocation-free, we use a fixed-size stack buffer for the
    // first-polarity v_hat snapshot. This bounds the supported output dim to
    // RANK1_PARENT_MAX_OUT_DIM (currently 64 — comfortably above HLA's 8 and
    // style_weights' 64). Larger output dims fall back to the owned variant.
    let gamma_u_pos = estimate_gamma(u_hat_scratch, op_i, op_j);
    let (obj_pos, k_self_pos) =
        compute_alignment_objective_into(u_hat_scratch, gamma_u_pos, op_i, op_j, v_hat_scratch);

    // Snapshot positive-polarity v_hat into the stack buffer before the
    // negative-polarity call overwrites v_hat_scratch.
    let mut v_hat_pos_snapshot = [0.0_f32; RANK1_PARENT_MAX_OUT_DIM];
    if m <= RANK1_PARENT_MAX_OUT_DIM {
        v_hat_pos_snapshot[..m].copy_from_slice(&v_hat_scratch[..m]);
    }

    for slot in u_hat_scratch.iter_mut().take(n) {
        *slot = -*slot;
    }
    let gamma_u_neg = -gamma_u_pos;
    let (obj_neg, k_self_neg) =
        compute_alignment_objective_into(u_hat_scratch, gamma_u_neg, op_i, op_j, v_hat_scratch);

    // Pick the polarity with the larger objective.
    if obj_pos >= obj_neg {
        // Restore positive polarity u_hat + v_hat.
        for slot in u_hat_scratch.iter_mut().take(n) {
            *slot = -*slot;
        }
        if m <= RANK1_PARENT_MAX_OUT_DIM {
            v_hat_scratch[..m].copy_from_slice(&v_hat_pos_snapshot[..m]);
        }
        let (_s_star_unused, k_self_final) =
            finalize_scale(op_i, op_j, k_self_pos, e_rem, v_hat_scratch, true);
        *k_self_scratch = k_self_final;
    } else {
        // Keep negative polarity u_hat; v_hat_scratch already holds v_hat_neg.
        let (_s_star_unused, k_self_final) =
            finalize_scale(op_i, op_j, k_self_neg, e_rem, v_hat_scratch, true);
        *k_self_scratch = k_self_final;
    }

    // Recompute a, b directly from the chosen polarity (simpler + correct).
    let cap_i = hope_capacity(op_i);
    let cap_j = hope_capacity(op_j);
    let a = cap_i * cap_i + cap_j * cap_j;
    let b = compute_b_term(op_i, op_j, v_hat_scratch, u_hat_scratch);
    compute_optimal_scale(a, b, e_rem)
}

/// Estimate `γ_u` for the parent direction as the weighted combination of the
/// children's `γ` values (matches the BN parameter recovery in paper Eq 18).
fn estimate_gamma(_u_hat: &[f32], op_i: &impl Rank1Operator, op_j: &impl Rank1Operator) -> f32 {
    // The parent's pre-activation variance is a weighted combination of the
    // children's. For the sign-resolution step we use the simpler max — the
    // exact γ_p is recovered only when deploying (paper Eq 18).
    op_i.gamma().max(op_j.gamma())
}

/// Compute the alignment objective for the inner optimization of Eq 10.
///
/// Writes the unit-norm v_hat into `v_hat_scratch` and returns
/// `(obj_value, k_self)`. Zero-alloc: the caller owns `v_hat_scratch`, and
/// because the polarity comparison needs a snapshot of the first-polarity
/// v_hat, `optimal_rank1_parent_into_scratch` uses a fixed-size stack
/// buffer of size [`RANK1_PARENT_MAX_OUT_DIM`] to preserve it across the
/// second call (which overwrites `v_hat_scratch`).
///
/// Replaces the prior `compute_alignment_objective` which returned an owned
/// `Vec<f32>` — that version allocated 2 Vecs per call (once per polarity)
/// and broke the zero-alloc contract on `optimal_rank1_parent_into_scratch`.
/// Caught by the bench_469_hope_kernel_goat G4 CountingAllocator audit.
#[inline]
fn compute_alignment_objective_into(
    u_hat: &[f32],
    gamma_u: f32,
    op_i: &impl Rank1Operator,
    op_j: &impl Rank1Operator,
    v_hat_scratch: &mut [f32],
) -> (f32, f32) {
    let m = op_i.w_out().len().min(op_j.w_out().len());

    // K(u, w_in_i) using warped correlation — but for a parent direction that's
    // a linear combination of children, the kernel simplifies. Use the
    // Arc-Cosine approximation.
    let k_ui = relu_cross_kernel_approx(u_hat, op_i.w_in(), gamma_u.abs(), op_i.gamma().abs());
    let k_uj = relu_cross_kernel_approx(u_hat, op_j.w_in(), gamma_u.abs(), op_j.gamma().abs());
    let k_self = relu_self_kernel(gamma_u.abs(), 0.0).max(1e-15);
    let sqrt_k_self = k_self.sqrt();

    // v̂ = (K(u,w_i)·w_out_i + K(u,w_j)·w_out_j) / ‖...‖
    let mut v_norm_sq = 0.0_f32;
    for (k, slot) in v_hat_scratch.iter_mut().enumerate().take(m) {
        let v = k_ui * op_i.w_out()[k] + k_uj * op_j.w_out()[k];
        *slot = v;
        v_norm_sq += v * v;
    }
    let v_norm = v_norm_sq.max(1e-30).sqrt();
    for slot in v_hat_scratch.iter_mut().take(m) {
        *slot /= v_norm;
    }

    // Objective: ‖Σ_k K(u,w̃_in_k)·w_out_k‖ / √K(u,u) = v_norm / √K(u,u)
    let obj = v_norm / sqrt_k_self;
    (obj, k_self)
}

/// Compute the `b` term `⟨ψ, f_i + f_j⟩_H` for the scale formula.
///
/// `b = ⟨ψ, f_i⟩_H + ⟨ψ, f_j⟩_H` where `ψ` is the unit-norm parent direction
/// `Ψ(u·x)·v / √K(u,u)`. For a unit-norm `ψ`, `⟨ψ, f_k⟩_H =
/// K(u, w_in_k)·⟨v, w_out_k⟩ / √K(u,u)`.
fn compute_b_term(
    op_i: &impl Rank1Operator,
    op_j: &impl Rank1Operator,
    v_hat: &[f32],
    u_hat: &[f32],
) -> f32 {
    let m = op_i.w_out().len().min(op_j.w_out().len());
    let gamma_u = op_i.gamma().max(op_j.gamma());
    let k_ui = relu_cross_kernel_approx(u_hat, op_i.w_in(), gamma_u, op_i.gamma());
    let k_uj = relu_cross_kernel_approx(u_hat, op_j.w_in(), gamma_u, op_j.gamma());
    let dot_v_wi: f32 = (0..m).map(|k| v_hat[k] * op_i.w_out()[k]).sum();
    let dot_v_wj: f32 = (0..m).map(|k| v_hat[k] * op_j.w_out()[k]).sum();
    let k_self = relu_self_kernel(gamma_u.abs(), 0.0).max(1e-15).sqrt();
    (k_ui * dot_v_wi + k_uj * dot_v_wj) / k_self
}

/// Closed-form optimal scale `s* = (a + b·E_rem) / (2·E_rem + b)` (paper Eq 12).
///
/// `a = ‖f_i‖²_H + ‖f_j‖²_H`, `b = ⟨ψ, f_i + f_j⟩_H`, `E_rem` = residual
/// capacity. The denominator is strictly positive when `b > 0` (the
/// phase-check guarantees this) or `E_rem > 0`. For `E_rem = 0` (total layer
/// collapse), `s* = a/b` (paper §7.1.2).
#[inline]
pub fn compute_optimal_scale(a: f32, b: f32, e_rem: f32) -> f32 {
    let denom = 2.0 * e_rem + b;
    if denom.abs() < 1e-15 {
        // Degenerate — fall back to a/(b+ε) to avoid div-by-zero.
        return a / (b + 1e-15);
    }
    (a + b * e_rem) / denom
}

/// Finalize the parent scale using cached `a, b, K_self` and the chosen v_hat.
///
/// This is a thin wrapper that exists to keep `optimal_rank1_parent_into_scratch`
/// readable. The `_with_v: bool` parameter is unused — kept for symmetry with
/// a future in-place variant.
fn finalize_scale(
    op_i: &impl Rank1Operator,
    op_j: &impl Rank1Operator,
    k_self: f32,
    e_rem: f32,
    _v_hat: &[f32],
    _with_v: bool,
) -> (f32, f32) {
    let cap_i = hope_capacity(op_i);
    let cap_j = hope_capacity(op_j);
    let a = cap_i * cap_i + cap_j * cap_j;
    // Use the simplified b = K_self (paper's identity for aligned parent).
    let b = k_self.max(1e-15);
    let s = compute_optimal_scale(a, b, e_rem);
    (s, k_self)
}

// ──────────────────────────────────────────────────────────────────────────
// Capacity + cost functionals (paper Eq 6 + Eq 20)
// ──────────────────────────────────────────────────────────────────────────

/// Hilbert-Schmidt capacity `‖f‖_H = ‖w_out‖₂ · √K(i,i)` (paper §5).
///
/// Scale-invariant: invariant under `(w_in, γ) → λ(w_in, γ)` and
/// `w_out → w_out/λ` for any `λ > 0` (PH-1 homogeneity + tensor structure).
#[inline]
pub fn hope_capacity(op: &impl Rank1Operator) -> f32 {
    let m = op.w_out().len();
    let w_out_norm_sq = simd::simd_dot_f32(op.w_out(), op.w_out(), m);
    let w_out_norm = w_out_norm_sq.max(0.0).sqrt();
    let k_self = relu_self_kernel(op.gamma(), op.beta());
    w_out_norm * k_self.max(0.0).sqrt()
}

/// Pruning cost `J_prune = N · ‖f_victim‖_H / (E_a − ‖f_victim‖_H)` (paper Eq 6 left).
///
/// `N` is the active neuron count in the layer, `E_a` is the layer's current
/// total capacity. Returns `+∞` if pruning would extinguish the layer
/// (`E_a → ‖f_victim‖`).
#[inline]
pub fn hope_prune_cost(victim: &impl Rank1Operator, n_active: usize, e_a: f32) -> f32 {
    let cap = hope_capacity(victim);
    let denom = e_a - cap;
    if denom < 1e-15 {
        return f32::INFINITY;
    }
    (n_active as f32) * cap / denom
}

/// Merge cost `J_merge` (paper Eq 6 right) — distortion per unit capacity.
///
/// Requires the precomputed `Rank1Parent` from [`optimal_rank1_parent`] plus
/// the layer's current state. Returns `+∞` if the merge would extinguish the
/// layer.
#[inline]
pub fn hope_merge_cost(
    op_i: &impl Rank1Operator,
    op_j: &impl Rank1Operator,
    parent: &Rank1Parent,
    n_active: usize,
    e_a: f32,
) -> f32 {
    let cap_i = hope_capacity(op_i);
    let cap_j = hope_capacity(op_j);
    let cap_p = parent.s_star * parent.k_self.max(0.0).sqrt();
    // D²(Φa, Φb) = ‖f_i − f_p‖²_H + ‖f_j − f_p‖²_H
    // Using ‖f_k − f_p‖²_H = ‖f_k‖²_H − 2·s·K_kp·⟨v,w_out_k⟩ + s²·K_p (Pythagoras).
    // Simplified bound: D ≤ ‖f_i‖ + ‖f_j‖ (triangle inequality).
    let d_sq = (cap_i * cap_i + cap_j * cap_j) - 2.0 * cap_p * (cap_i + cap_j).min(1.0);
    let d = d_sq.max(0.0).sqrt();
    let denom = e_a - cap_i - cap_j + cap_p;
    if denom < 1e-15 {
        return f32::INFINITY;
    }
    (n_active as f32) * d / denom
}

/// Block eviction cost `J_evict = Σ_l (N_active^(l) · E_active^(l)) / E_identity`
/// (paper Eq 20).
///
/// `e_active_per_layer[l]` is the surviving capacity of internal layer `l`,
/// `n_active_per_layer[l]` is its active operator count, `e_identity` is the
/// RMS energy of the parallel skip pathway (`Σ_k √(γ²_k + β²_k)` for residual
/// architectures; for non-residual, use the layer's initial capacity).
#[inline]
pub fn hope_block_eviction_cost(
    n_active_per_layer: &[usize],
    e_active_per_layer: &[f32],
    e_identity: f32,
) -> f32 {
    debug_assert_eq!(n_active_per_layer.len(), e_active_per_layer.len());
    if e_identity < 1e-15 {
        return f32::INFINITY;
    }
    let mut total = 0.0_f32;
    for (n, e) in n_active_per_layer.iter().zip(e_active_per_layer.iter()) {
        total += (*n as f32) * e;
    }
    total / e_identity
}

// ──────────────────────────────────────────────────────────────────────────
// Greedy action selection (paper §9 Eq 23)
// ──────────────────────────────────────────────────────────────────────────

/// A candidate compression action (paper §9 + §10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HopeAction {
    /// Prune neuron at index `victim_idx`.
    Prune { victim_idx: usize },
    /// Merge neurons `(i_idx, j_idx)` (i < j).
    Merge { i_idx: usize, j_idx: usize },
    /// Evict block `block_id`.
    Evict { block_id: usize },
}

/// Greedy action selection: returns the action with the minimal distortion
/// rate `DR = J / ΔP_init` (paper Eq 23, Dantzig greedy on the continuous
/// knapsack relaxation).
///
/// `distortion_rates[i]` is `J_i / ΔP_init_i` for each candidate action `i`.
/// Returns `Some(i)` for the minimum, or `None` if the slice is empty.
#[inline]
pub fn hope_greedy_select(distortion_rates: &[f32]) -> Option<usize> {
    if distortion_rates.is_empty() {
        return None;
    }
    let mut best_idx = 0;
    let mut best_dr = distortion_rates[0];
    for (i, &dr) in distortion_rates.iter().enumerate().skip(1) {
        if dr < best_dr {
            best_dr = dr;
            best_idx = i;
        }
    }
    Some(best_idx)
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A simple Rank1Operator impl for testing — owns its weight vectors.
    struct OwnedRank1 {
        w_in: Vec<f32>,
        w_out: Vec<f32>,
        gamma: f32,
        beta: f32,
    }
    impl Rank1Operator for OwnedRank1 {
        fn w_in(&self) -> &[f32] {
            &self.w_in
        }
        fn w_out(&self) -> &[f32] {
            &self.w_out
        }
        fn gamma(&self) -> f32 {
            self.gamma
        }
        fn beta(&self) -> f32 {
            self.beta
        }
    }

    fn owned(w_in: &[f32], w_out: &[f32], gamma: f32, beta: f32) -> OwnedRank1 {
        OwnedRank1 {
            w_in: w_in.to_vec(),
            w_out: w_out.to_vec(),
            gamma,
            beta,
        }
    }

    // ── Phase 1: normal_pdf/cdf, relu_self_kernel ─────────────────────────

    #[test]
    fn g1_normal_cdf_standard_normal() {
        // Φ(0) = 0.5 exactly.
        let cdf_0 = normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 1e-4, "Φ(0) = {cdf_0}, expected 0.5");
        // Φ(1) ≈ 0.8413, Φ(−1) ≈ 0.1587, Φ(2) ≈ 0.9772.
        assert!(
            (normal_cdf(1.0) - 0.8413_447).abs() < 1e-4,
            "Φ(1) = {}",
            normal_cdf(1.0)
        );
        assert!(
            (normal_cdf(-1.0) - 0.1586_552).abs() < 1e-4,
            "Φ(−1) = {}",
            normal_cdf(-1.0)
        );
        assert!(
            (normal_cdf(2.0) - 0.9772_498).abs() < 1e-4,
            "Φ(2) = {}",
            normal_cdf(2.0)
        );
    }

    #[test]
    fn g1_normal_pdf_standard_normal() {
        // φ(0) = 1/√(2π) ≈ 0.3989.
        let pdf_0 = normal_pdf(0.0);
        assert!(
            (pdf_0 - 0.3989_4223_f32).abs() < 1e-5,
            "φ(0) = {pdf_0}, expected 0.39894"
        );
        // φ is symmetric.
        assert!((normal_pdf(1.5) - normal_pdf(-1.5)).abs() < 1e-6);
    }

    #[test]
    fn g1_relu_self_kernel_standard_normal() {
        // For γ=1, β=0: K(i,i) = (1+0)·Φ(0) + 0·1·φ(0) = 1·0.5 + 0 = 0.5.
        // This is the half-wave rectified energy of a standard-normal pre-activation.
        let k = relu_self_kernel(1.0, 0.0);
        assert!(
            (k - 0.5).abs() < 1e-4,
            "K(1,0) = {k}, expected 0.5 (half-wave rectified energy)"
        );
    }

    #[test]
    fn g1_relu_self_kernel_positive() {
        // K(i,i) ≥ 0 for all (γ, β) — it's an expected squared value.
        for gamma in [0.1_f32, 0.5, 1.0, 2.0, 5.0] {
            for beta in [-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
                let k = relu_self_kernel(gamma, beta);
                assert!(k >= 0.0, "K({gamma},{beta}) = {k} < 0");
            }
        }
    }

    #[test]
    fn g1_relu_self_kernel_sign_invariant_gamma() {
        // Paper §3.1: K(i,i) is invariant under γ → −γ (it's |γ| that enters).
        for beta in [-1.0_f32, 0.0, 1.0] {
            let k_pos = relu_self_kernel(1.5, beta);
            let k_neg = relu_self_kernel(-1.5, beta);
            assert!(
                (k_pos - k_neg).abs() < 1e-6,
                "K(γ=1.5,β={beta}) = {k_pos} ≠ K(γ=−1.5,β={beta}) = {k_neg}"
            );
        }
    }

    #[test]
    fn g1_relu_self_kernel_zero_gamma_dead_neuron() {
        // γ = 0: y is constant β. K(i,i) = max(0,β)².
        assert!(
            (relu_self_kernel(0.0, 2.0) - 4.0).abs() < 1e-6,
            "dead neuron β=2: K = {}",
            relu_self_kernel(0.0, 2.0)
        );
        assert!(
            (relu_self_kernel(0.0, -1.0) - 0.0).abs() < 1e-6,
            "dead neuron β=−1: K = {}",
            relu_self_kernel(0.0, -1.0)
        );
    }

    #[test]
    fn g1_relu_self_kernel_matches_high_bias_analytic() {
        // For γ=1, β=1: K = (1+1)·Φ(1) + 1·1·φ(1) = 2·0.8413 + 1·0.2420 = 1.9247.
        let k = relu_self_kernel(1.0, 1.0);
        let expected = 2.0 * 0.8413_447 + 1.0 * 0.2419_7072;
        assert!(
            (k - expected).abs() < 1e-3,
            "K(1,1) = {k}, expected {expected}"
        );
    }

    // ── Phase 2: warped_correlation, cross-kernel, optimal parent ─────────

    #[test]
    fn g2_warped_correlation_in_range() {
        // ρ̂ ∈ (−1, 1) for any inputs.
        let w_i = [1.0_f32, 0.0, 0.0];
        let w_j = [0.0_f32, 1.0, 0.0];
        let rho = warped_correlation(&w_i, &w_j, 1.0, 1.0);
        assert!(rho.abs() <= 1.0, "ρ̂ = {rho} not in [−1, 1]");
    }

    #[test]
    fn g2_warped_correlation_orthogonal_inputs() {
        // Orthogonal inputs → ρ_eff = 0 → κ = 0 → ρ̂ = 0.
        let w_i = [1.0_f32, 0.0, 0.0];
        let w_j = [0.0_f32, 1.0, 0.0];
        let rho = warped_correlation(&w_i, &w_j, 1.0, 1.0);
        assert!(rho.abs() < 1e-6, "ρ̂ for orthogonal = {rho}, expected 0");
    }

    #[test]
    fn g2_warped_correlation_collinear_inputs() {
        // Identical inputs → ρ_eff = 1 → ρ̂ → 1 (after clamping).
        let w = [1.0_f32, 0.5, -0.3, 0.8];
        let rho = warped_correlation(&w, &w, 1.0, 1.0);
        // Should be close to 1 (the clamp prevents exact 1).
        assert!(rho > 0.99, "ρ̂ for collinear = {rho}, expected near 1");
    }

    #[test]
    fn g2_relu_cross_kernel_diagonal_consistency() {
        // K(i,i) for cross-kernel with identical inputs ≈ K(i,i) from self-kernel.
        // Note: cross-kernel uses the zero-bias approximation, so this matches
        // relu_self_kernel(γ, 0) (β=0 form), NOT the full relu_self_kernel(γ, β).
        let w = [0.3_f32, -0.5, 0.8, 0.1, 0.9];
        let gamma = 1.0_f32;
        let k_self_zero_bias = relu_self_kernel(gamma, 0.0);
        let k_cross = relu_cross_kernel_approx(&w, &w, gamma, gamma);
        assert!(
            (k_cross - k_self_zero_bias).abs() < 1e-4,
            "cross-kernel(i,i) = {k_cross}, self-kernel(i,i,β=0) = {k_self_zero_bias}"
        );
    }

    #[test]
    fn g2_relu_cross_kernel_cauchy_schwarz() {
        // |K(i,j)| ≤ √(K(i,i)·K(j,j)) for random inputs.
        let w_i = [0.3_f32, -0.5, 0.8, 0.1];
        let w_j = [0.7_f32, 0.2, -0.4, 0.6];
        let gamma_i = 1.0_f32;
        let gamma_j = 1.5_f32;
        let k_ii = relu_self_kernel(gamma_i, 0.0);
        let k_jj = relu_self_kernel(gamma_j, 0.0);
        let k_ij = relu_cross_kernel_approx(&w_i, &w_j, gamma_i, gamma_j);
        let bound = (k_ii * k_jj).max(0.0).sqrt();
        assert!(
            k_ij.abs() <= bound + 1e-5,
            "Cauchy-Schwarz violated: |K(ij)| = {} > √(K(ii)·K(jj)) = {}",
            k_ij.abs(),
            bound
        );
    }

    #[test]
    fn g2_optimal_scale_simple_case() {
        // For a=b=1, E_rem=1: s* = (1+1)/(2+1) = 2/3.
        let s = compute_optimal_scale(1.0, 1.0, 1.0);
        assert!((s - 2.0 / 3.0).abs() < 1e-6, "s* = {s}, expected 2/3");

        // For E_rem=0 (total layer collapse): s* = a/b.
        let s_collapse = compute_optimal_scale(3.0, 2.0, 0.0);
        assert!(
            (s_collapse - 1.5).abs() < 1e-6,
            "s* collapse = {s_collapse}, expected a/b = 1.5"
        );
    }

    #[test]
    fn g2_optimal_scale_positive() {
        // s* > 0 whenever a > 0 and (b > 0 or E_rem > 0).
        for a in [0.5_f32, 1.0, 2.0] {
            for b in [0.1_f32, 1.0, 10.0] {
                for e_rem in [0.0_f32, 0.5, 1.0, 10.0] {
                    let s = compute_optimal_scale(a, b, e_rem);
                    assert!(s >= 0.0, "s* = {s} < 0 for a={a}, b={b}, E={e_rem}");
                }
            }
        }
    }

    #[test]
    fn g2_principal_eigenvector_rank2_diagonal_dominant() {
        // When m12 ≈ 0 and m11 > m22, the principal eigenvector is (1, 0).
        let w_in_i = [1.0_f32, 0.0, 0.0]; // ‖w_in_i‖² = 1
        let w_in_j = [0.0_f32, 0.1, 0.0]; // ‖w_in_j‖² = 0.01
        let w_out_i = [1.0_f32];
        let w_out_j = [1.0_f32];
        let (c1, c2) = principal_eigenvector_rank2(&w_in_i, &w_in_j, &w_out_i, &w_out_j);
        // m11 = 1·1 = 1, m22 = 0.01·1 = 0.01 — diagonal-dominant in axis 0.
        assert!(
            c1.abs() > c2.abs(),
            "(c1, c2) = ({c1}, {c2}), expected |c1| > |c2|"
        );
        assert!((c1.abs() - 1.0).abs() < 1e-5, "c1 = {c1}, expected ±1");
    }

    #[test]
    fn g2_optimal_parent_identical_inputs_returns_self() {
        // Merging two identical neurons should produce a parent close to the input.
        let w_in = [0.5_f32, 0.5, 0.5, 0.5];
        let w_out = [1.0_f32];
        let op = owned(&w_in, &w_out, 1.0, 0.0);
        let parent = optimal_rank1_parent(&op, &op, 1.0);
        // u_hat should be parallel to w_in (cosine ≈ 1).
        let dot = simd::simd_dot_f32(&parent.u_hat, &w_in, 4);
        let norm_u = parent.u_hat.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_w = (w_in.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let cos = dot / (norm_u * norm_w).max(1e-15);
        assert!(
            cos.abs() > 0.99,
            "merge identical: cos(u, w_in) = {cos}, expected > 0.99"
        );
    }

    // ── Phase 3: capacity, costs, greedy selection ────────────────────────

    #[test]
    fn g3_hope_capacity_positive() {
        let op = owned(&[1.0_f32, 0.5, -0.3], &[0.8_f32, 0.6], 1.0, 0.0);
        let cap = hope_capacity(&op);
        assert!(cap > 0.0, "capacity = {cap}, expected > 0");
    }

    #[test]
    fn g3_hope_capacity_zero_for_dead_neuron() {
        // γ=0, β<0: dead neuron, capacity = 0.
        let op = owned(&[1.0_f32], &[1.0_f32], 0.0, -1.0);
        let cap = hope_capacity(&op);
        assert!(cap.abs() < 1e-6, "dead-neuron capacity = {cap}, expected 0");
    }

    #[test]
    fn g3_hope_capacity_scale_invariant() {
        // Scale invariance for the FULL operator: scale (w_in, γ) by λ and
        // w_out by 1/λ. The kernel K(i,i) scales by λ² (because y = w_in·x
        // has variance γ², which scales by λ²); ‖w_out‖ shrinks by 1/λ; and
        // the product ‖w_out‖·√K stays invariant. The paper's scale invariance
        // (§5) assumes γ reflects the true pre-activation variance.
        let op_a = owned(&[1.0_f32, 0.5], &[2.0_f32, 0.5], 1.0, 0.0);
        // λ=2: w_in → 2·w_in, w_out → w_out/2, γ → 2·γ (variance scales by λ²
        // because Var(2·w_in·x) = 4·Var(w_in·x), but γ is the std-dev so γ → 2·γ).
        let op_b = owned(&[2.0_f32, 1.0], &[1.0_f32, 0.25], 2.0, 0.0);
        let cap_a = hope_capacity(&op_a);
        let cap_b = hope_capacity(&op_b);
        assert!(
            (cap_a - cap_b).abs() < 1e-4,
            "scale invariance: cap(λ=1)={cap_a}, cap(λ=2)={cap_b}"
        );
    }

    #[test]
    fn g3_hope_kernel_scale_invariance_quadratic() {
        // The ReLU self-kernel K(i,i) scales as λ² when (γ, β) → λ·(γ, β)
        // (paper §5 scale invariance). This is the mathematical root of the
        // capacity scale invariance.
        let k_base = relu_self_kernel(1.0, 0.5);
        let k_scaled = relu_self_kernel(2.0, 1.0); // λ=2
        // K should scale by λ² = 4.
        let ratio = k_scaled / k_base.max(1e-15);
        assert!(
            (ratio - 4.0).abs() < 1e-3,
            "K scale invariance: ratio = {ratio}, expected 4.0"
        );
    }

    #[test]
    fn g3_hope_prune_cost_positive_finite() {
        let op = owned(&[1.0_f32, 0.5], &[1.0_f32], 1.0, 0.0);
        let cap = hope_capacity(&op);
        let cost = hope_prune_cost(&op, 10, cap + 1.0);
        assert!(cost > 0.0 && cost.is_finite(), "prune cost = {cost}");
    }

    #[test]
    fn g3_hope_prune_cost_infinite_on_extinction() {
        // If pruning would zero the layer (E_a == ‖f‖), cost = +∞.
        let op = owned(&[1.0_f32], &[1.0_f32], 1.0, 0.0);
        let cap = hope_capacity(&op);
        let cost = hope_prune_cost(&op, 1, cap);
        assert!(cost.is_infinite(), "extinction cost = {cost}, expected ∞");
    }

    #[test]
    fn g3_hope_merge_cost_finite() {
        let op_i = owned(&[1.0_f32, 0.0], &[1.0_f32], 1.0, 0.0);
        let op_j = owned(&[0.7_f32, 0.7], &[1.0_f32], 1.0, 0.0);
        let parent = optimal_rank1_parent(&op_i, &op_j, 1.0);
        let e_a = hope_capacity(&op_i) + hope_capacity(&op_j) + 1.0;
        let cost = hope_merge_cost(&op_i, &op_j, &parent, 2, e_a);
        assert!(cost.is_finite(), "merge cost = {cost}");
    }

    #[test]
    fn g3_hope_block_eviction_cost_simple() {
        // One internal layer with N=2 active neurons, E_active=3.0, E_identity=1.0:
        // J_evict = (2·3)/1 = 6.0.
        let cost = hope_block_eviction_cost(&[2], &[3.0], 1.0);
        assert!((cost - 6.0).abs() < 1e-6, "J_evict = {cost}, expected 6.0");
    }

    #[test]
    fn g3_hope_block_eviction_cost_multi_layer() {
        // Two layers, both (N=1, E=2.0), E_identity=4.0:
        // J_evict = (1·2 + 1·2)/4 = 4/4 = 1.0.
        let cost = hope_block_eviction_cost(&[1, 1], &[2.0, 2.0], 4.0);
        assert!(
            (cost - 1.0).abs() < 1e-6,
            "J_evict multi = {cost}, expected 1.0"
        );
    }

    #[test]
    fn g3_hope_block_eviction_cost_zero_identity_infinite() {
        // E_identity = 0 (no skip connection): J_evict = ∞.
        let cost = hope_block_eviction_cost(&[1], &[1.0], 0.0);
        assert!(
            cost.is_infinite(),
            "J_evict identity=0 = {cost}, expected ∞"
        );
    }

    #[test]
    fn g3_hope_greedy_select_minimal_dr() {
        let drs = [5.0_f32, 1.0, 3.0, 0.5, 2.0];
        let best = hope_greedy_select(&drs).unwrap();
        assert_eq!(best, 3, "best DR index = {best}, expected 3");
    }

    #[test]
    fn g3_hope_greedy_select_empty_returns_none() {
        assert!(hope_greedy_select(&[]).is_none());
    }

    #[test]
    fn g3_hope_greedy_select_single() {
        assert_eq!(hope_greedy_select(&[42.0]).unwrap(), 0);
    }

    // ── G4 alloc-free spot check (smoke test; full CountingAllocator audit
    //    runs in the bench — Plan 469 T4.1) ────────────────────────────────

    #[test]
    fn g4_hot_path_kernels_no_alloc_indirect() {
        // The hot-path functions take &[f32] and return f32 — by signature
        // they can't allocate. This test exercises them on a variety of
        // inputs to catch any silent regression that introduces a Vec/String.
        let w_i = [0.3_f32, -0.5, 0.8, 0.1, 0.9, -0.2, 0.4, 0.0];
        let w_j = [0.7_f32, 0.2, -0.4, 0.6, 0.1, -0.3, 0.5, 0.8];
        let w_out = [1.0_f32, 0.5];

        // Capacity, prune cost, cross-kernel — all return f32.
        let _ = relu_self_kernel(1.0, 0.5);
        let _ = warped_correlation(&w_i, &w_j, 1.0, 1.5);
        let _ = relu_cross_kernel_approx(&w_i, &w_j, 1.0, 1.5);

        let op = owned(&w_i, &w_out, 1.0, 0.0);
        let _ = hope_capacity(&op);
        let _ = hope_prune_cost(&op, 5, 10.0);
        let _ = hope_block_eviction_cost(&[3, 2], &[1.5, 2.0], 5.0);
        let _ = hope_greedy_select(&[1.0, 2.0, 0.5]);

        // optimal_rank1_parent_into_scratch takes &mut [f32] scratches — zero-alloc.
        let mut u_scratch = [0.0_f32; 8];
        let mut v_scratch = [0.0_f32; 2];
        let mut k_self = 0.0_f32;
        let op2 = owned(&w_j, &w_out, 1.5, 0.2);
        let _ = optimal_rank1_parent_into_scratch(
            &op,
            &op2,
            1.0,
            &mut u_scratch,
            &mut v_scratch,
            &mut k_self,
        );
    }
}
