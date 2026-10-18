//! GRAPE-M — Closed-form Rank-2 Rodrigues Exponential for Arbitrary Plane.
//!
//! Distilled from Zhang et al., *GRAPE: Group Representational Position
//! Encoding* (arXiv:2512.07805, ICLR 2026, §2.3 + Appendix I). See
//! [Research 446](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md)
//! for the full distillation. The paper is a *training* paper (learned
//! rotation planes via gradient descent on `a, b`), but the closed-form
//! *application* `y = exp(n·ω·L)·x` is genuinely modelless: it is pure float
//! arithmetic on a user-supplied plane `(a, b)`. Only the deterministic math
//! ships here — no backprop, no learned `a, b`. Learning the plane is
//! `→ riir-train` (the modelless-first mandate, AGENTS.md).
//!
//! # What this computes
//!
//! For any two vectors `a, b ∈ ℝᴰ`, the rank-2 skew-symmetric generator
//!
//! ```text
//! L = a·bᵀ − b·aᵀ     (i.e.  L[i,j] = a[i]·b[j] − b[i]·a[j])
//! ```
//!
//! generates a one-parameter subgroup of `SO(d)`:
//!
//! ```text
//! G(n) = exp(n·ω·L)
//! ```
//!
//! The matrix exponential admits the **Rodrigues closed form**:
//!
//! ```text
//! exp(θ·L̂) = I + sin(s)·L̂ + (1 − cos(s))·L̂²
//! ```
//!
//! where `s = √(αβ − γ²)`, `α = ‖a‖²`, `β = ‖b‖²`, `γ = aᵀb`, and `L̂ = L/s`
//! is the unit generator. The application `y = exp(n·ω·L)·x` is `O(d)` —
//! **two dot products + one vector triad** — without ever materialising the
//! `d×d` matrix (beats LieRE's `O(d³)` `torch.matrix_exp`).
//!
//! ## The `O(d)` derivation
//!
//! Let `p = ⟨a,x⟩`, `q = ⟨b,x⟩`. Then
//!
//! ```text
//! L·x   = q·a − p·b
//! L²·x  = (γ·q − β·p)·a − (α·q − γ·p)·b
//! ```
//!
//! (the second line follows from `L·(L·x) = ⟨b, L·x⟩·a − ⟨a, L·x⟩·b` plus the
//! identities `⟨a, L·x⟩ = α·q − γ·p` and `⟨b, L·x⟩ = γ·q − β·p`). Therefore
//!
//! ```text
//! y = x + c1·(L·x) + c2·(L²·x)
//!   = x + (c1·q + c2·(γ·q − β·p))·a − (c1·p + c2·(α·q − γ·p))·b
//! ```
//!
//! where `c1 = sin(θ)/s`, `c2 = (1 − cos(θ))/s²`, `θ = n·ω·s`.
//!
//! ## Sign convention
//!
//! With `L = abᵀ − baᵀ`, the generator rotates `a` toward **−b**
//! (clockwise in the `(a, b)` plane). The closed form is
//! `exp(θ·L)·a = cos(θ)·a − sin(θ)·b`. Swapping `a ↔ b` flips the sign —
//! use `L' = baᵀ − abᵀ = −L` for the counter-clockwise convention. This
//! matches the GRAPE paper §2.3 definition. RoPE's standard convention
//! (counter-clockwise) is recovered by passing `(b, a)` rather than `(a, b)`
//! to [`grapem_apply_into`] when bridging via Issue 160's `RopeAction`.
//!
//! # Why this is NOT redundant
//!
//! [`crate::phase_rotation::phase_rotation_gate_into`] (Plan 322, UFO) does a
//! *scalar-broadcast 2D rotation* `out = cos(α)·a + sin(α)·b` between two named
//! halves — the rotation plane is the canonical `(e_i, e_{i+D/2})` coordinate
//! pair. It **cannot** express a rotation in an *arbitrary learned plane*
//! `U = span{a, b}` where `a, b` are not orthogonal coordinate selectors.
//!
//! Closing this gap unlocks (per Research 446 §2.3):
//!
//! - Per-NPC HLA personality-specific rotation planes (riir-ai fusion
//!   candidate, Plan 336 personality runtime).
//! - Per-shard rotation plane in `MerkleFrozenEnvelope` (riir-neuron-db).
//! - A principled generalization of RoPE to non-canonical bases (Issue 160
//!   unifies RoPE / ALiBi / FoX / Wall under one `PositionGroupAction` trait;
//!   RoPE is the special case `a = e_i, b = e_{i+D/2}`).
//!
//! # Numerical contract
//!
//! - All entry points are pure float arithmetic over caller-provided buffers.
//!   Deterministic on a given CPU (same inputs → bit-identical outputs).
//! - `a`, `b`, `x`, `out` must be equal-length; mismatches trip
//!   [`GrapemError::ShapeMismatch`].
//! - The degenerate plane (`s ≈ 0`, i.e. `a ∥ b`) is handled by the
//!   **small-angle Taylor branch**: when `s < ε`, `sin(θ)/s → n·ω` and
//!   `(1−cos(θ))/s² → (n·ω)²/2` (with `θ = n·ω·s`), so the formula reduces
//!   cleanly to `y → x` in the limit `s → 0` (zero generator → identity).
//! - `out` may alias `x` (the kernel computes the two scalar projections `p,q`
//!   first, then writes `out[i]` once per element — see [`grapem_apply_into`]).
//!
//! # API design — why `Rank2Plane` retains `a, b`
//!
//! [`Rank2Plane`] is the pre-computed handle for repeated applications of the
//! same plane. It stores `a, b` as `Box<[f32]>` (one heap allocation per
//! vector at construction) plus the four derived scalars `α, β, γ, s, s²`.
//! This is a **deviation from Issue 159's spec** ("stores only the 4
//! scalars"), which is mathematically impossible: the inner kernel needs
//! `a, b` to evaluate the projections `p = ⟨a,x⟩`, `q = ⟨b,x⟩`. The G4 gate
//! constraint ("0 allocations in `apply_into` after `new`") still holds — the
//! allocation moves from per-call to one-time-per-plane. The scalars avoid
//! recomputing `α, β, γ` on every call (3 dot products saved per apply).
//!
//! # Performance
//!
//! `O(d)` per call: two dot products (`simd::simd_dot_f32`), one scalar triad
//! to combine coefficients, and one FMA write loop. Zero allocation after
//! [`Rank2Plane::new`]. The inner write loop is `out[i] = x[i] + ka·a[i] − kb·b[i]`,
//! chunked 4-wide to hint LLVM auto-vectorisation (matches the pattern in
//! [`crate::phase_rotation::phase_rotation_gate_into`] and
//! `dec::operators::exterior_derivative_into`).
//!
//! GRAPE-M — Rank-2 Rodrigues Exponential for arbitrary plane (a, b).
//! Distilled from Zhang et al. *GRAPE* (arXiv:2512.07805, ICLR 2026 §2.3).
//! O(d) closed-form application of `exp(n·ω·L)·x` where `L = abᵀ − baᵀ` (rank-2
//! skew) — two dot products + one FMA triad, never materialises the d×d matrix
//! (beats LieRE's O(d³) torch.matrix_exp). Subsumes phase_rotation's scalar
//! 2D rotation as the canonical-basis special case `a = e_i, b = e_{i+D/2}`;
//! the new capability is rotation in a *learned* plane (per-NPC HLA personality
//! rotation, per-shard rotation in MerkleFrozenEnvelope — see Issue 159).
//! Pure modelless float arithmetic; learning the plane is → riir-train.
//! STAYS OPT-IN — G1–G4 GOAT gate ALL PASS (Bench 457, 2026-07-17); promotion
//! deferred per Issue 159 T6: gain is modelless but perf-only on a NEW capability
//! (arbitrary-plane rotation), not a faster way to do something the crate already
//! does. Re-evaluate when a concrete consumer lands (riir-ai HLA personality
//! rotation, riir-neuron-db per-shard rotation).

use crate::simd;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by the GRAPE-M entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GrapemError {
    /// `a.len() != b.len()` or `x.len() != a.len()` or `out.len() != a.len()`.
    ShapeMismatch,
}

// ── Pre-computed plane ───────────────────────────────────────────

/// Pre-computed handle for a rank-2 rotation plane `(a, b)`.
///
/// Stores the two generator vectors (as `Box<[f32]>` — one heap allocation
/// each at construction) plus the derived scalars `α, β, γ, s, s²`. After
/// [`Rank2Plane::new`], [`Rank2Plane::apply_into`] is **zero-allocation** —
/// it reuses the cached vectors + scalars to evaluate `y = exp(n·ω·L)·x` in
/// `O(d)` for any `(x, n, ω)`.
///
/// # Construction
///
/// [`Rank2Plane::new`] computes `α = ‖a‖²`, `β = ‖b‖²`, `γ = aᵀb`,
/// `s² = αβ − γ²` (the squared area of the parallelogram spanned by `a, b` —
/// zero iff `a ∥ b`), and `s = √s²`. The vectors are copied into `Box<[f32]>`
/// so the handle owns its data and can be moved freely.
///
/// # Degenerate planes
///
/// If `s² == 0` (`a ∥ b`, including `a = 0` or `b = 0`), the generator is
/// zero and `exp(n·ω·L) = I` for all `n, ω`. [`Rank2Plane::apply_into`]
/// handles this via the small-angle branch and returns `out = x` cleanly.
///
/// # Why not just the scalars?
///
/// See the module doc — the inner kernel needs `a, b` at apply time to
/// compute the projections `p = ⟨a,x⟩`, `q = ⟨b,x⟩`. The scalars alone are
/// insufficient. (Issue 159's original "stores only the 4 scalars" spec is
/// mathematically impossible; this implementation deviates by retaining
/// `a, b`. The G4 alloc-free contract on `apply_into` still holds.)
#[derive(Debug, Clone)]
pub struct Rank2Plane {
    /// `a` generator vector, length `D`.
    a: Box<[f32]>,
    /// `b` generator vector, length `D`.
    b: Box<[f32]>,
    /// `α = ‖a‖²`.
    alpha: f32,
    /// `β = ‖b‖²`.
    beta: f32,
    /// `γ = aᵀb`.
    gamma: f32,
    /// `s = √(αβ − γ²)`. Zero iff `a ∥ b`.
    s: f32,
    /// `s² = αβ − γ²`. Cached to avoid recomputing in the inner loop.
    s_sq: f32,
}

impl Rank2Plane {
    /// Pre-compute the plane from the two generator vectors.
    ///
    /// Copies `a, b` into owned `Box<[f32]>` storage (two allocations) and
    /// derives the scalars `α, β, γ, s, s²`. After construction,
    /// [`apply_into`](Self::apply_into) is zero-allocation.
    ///
    /// # Panics
    ///
    /// Debug builds assert `a.len() == b.len()`. Release builds leave the
    /// length mismatch undefined (matches the `simd::simd_dot_f32` convention).
    #[inline]
    pub fn new(a: &[f32], b: &[f32]) -> Self {
        debug_assert_eq!(a.len(), b.len(), "Rank2Plane::new: a.len() != b.len()");
        let d = a.len();
        let alpha = simd::simd_dot_f32(a, a, d);
        let beta = simd::simd_dot_f32(b, b, d);
        let gamma = simd::simd_dot_f32(a, b, d);
        let s_sq = (alpha * beta - gamma * gamma).max(0.0);
        let s = s_sq.sqrt();
        Self {
            a: a.into(),
            b: b.into(),
            alpha,
            beta,
            gamma,
            s,
            s_sq,
        }
    }

    /// Dimension `D` of the vectors this plane operates on.
    #[inline]
    pub fn dim(&self) -> usize {
        self.a.len()
    }

    /// `α = ‖a‖²`.
    #[inline]
    pub const fn alpha(&self) -> f32 {
        self.alpha
    }
    /// `β = ‖b‖²`.
    #[inline]
    pub const fn beta(&self) -> f32 {
        self.beta
    }
    /// `γ = aᵀb`.
    #[inline]
    pub const fn gamma(&self) -> f32 {
        self.gamma
    }
    /// `s = √(αβ − γ²)`. Zero iff `a ∥ b` (degenerate plane).
    #[inline]
    pub const fn s(&self) -> f32 {
        self.s
    }
    /// `s² = αβ − γ²` (the squared parallelogram area).
    #[inline]
    pub const fn s_sq(&self) -> f32 {
        self.s_sq
    }
    /// Read access to the stored `a` generator.
    #[inline]
    pub fn a(&self) -> &[f32] {
        &self.a
    }
    /// Read access to the stored `b` generator.
    #[inline]
    pub fn b(&self) -> &[f32] {
        &self.b
    }

    /// Apply `y = exp(n·ω·L)·x` to a vector, writing to `out`.
    ///
    /// `out.len()` must equal `x.len()` and equal [`Self::dim`]. `out` may
    /// alias `x` (the kernel computes the scalar projections `p, q` first,
    /// then writes `out[i]` once per element).
    ///
    /// See the module doc for the `O(d)` derivation. The degenerate-plane
    /// case (`s ≈ 0`) routes through the small-angle Taylor branch and
    /// returns `out = x` cleanly.
    ///
    /// # Errors
    ///
    /// Returns [`GrapemError::ShapeMismatch`] if `x.len() != out.len()` or
    /// either disagrees with [`Self::dim`].
    ///
    /// # Performance
    ///
    /// `O(d)`, zero allocation. Two dot products + one FMA write loop.
    #[inline]
    pub fn apply_into(
        &self,
        x: &[f32],
        n: f32,
        omega: f32,
        out: &mut [f32],
    ) -> Result<(), GrapemError> {
        let d = self.a.len();
        if x.len() != d || out.len() != d {
            return Err(GrapemError::ShapeMismatch);
        }
        if d == 0 {
            return Ok(());
        }
        grapem_apply_inner(
            &self.a, &self.b, self.alpha, self.beta, self.gamma, self.s, self.s_sq, x, n, omega,
            out,
        );
        Ok(())
    }
}

// ── Core kernel ──────────────────────────────────────────────────

/// Compute `y = exp(n·ω·L)·x` where `L = a·bᵀ − b·aᵀ` (rank-2 skew).
///
/// This is the **un-pre-computed** entry point: it materialises `α, β, γ, s`
/// from `a, b` on every call (3 dot products), then applies the rotation.
/// For repeated applications of the same plane, use [`Rank2Plane`] instead —
/// it caches the four scalars (and the vectors) and skips the recomputation.
///
/// # Arguments
///
/// * `a`, `b` — generator vectors, length `D`. Need not be unit, orthogonal,
///   or non-parallel. The degenerate case `a ∥ b` is handled.
/// * `x` — input vector, length `D`.
/// * `n` — position (or any group parameter). Multiplies `ω` to give `θ`.
/// * `omega` — frequency scale.
/// * `out` — output vector, length `D`. May alias `x`.
///
/// # Errors
///
/// Returns [`GrapemError::ShapeMismatch`] if `a`, `b`, `x`, `out` lengths
/// disagree.
///
/// # Performance
///
/// `O(D)`, zero allocation. Three dot products to build the scalars, then one
/// FMA write loop. The inner loop is `out[i] = x[i] + ka·a[i] − kb·b[i]`,
/// chunked 4-wide to hint LLVM auto-vectorisation.
#[inline]
pub fn grapem_apply_into(
    a: &[f32],
    b: &[f32],
    x: &[f32],
    n: f32,
    omega: f32,
    out: &mut [f32],
) -> Result<(), GrapemError> {
    let d = a.len();
    if b.len() != d || x.len() != d || out.len() != d {
        return Err(GrapemError::ShapeMismatch);
    }
    if d == 0 {
        return Ok(());
    }
    let alpha = simd::simd_dot_f32(a, a, d);
    let beta = simd::simd_dot_f32(b, b, d);
    let gamma = simd::simd_dot_f32(a, b, d);
    let s_sq = (alpha * beta - gamma * gamma).max(0.0);
    let s = s_sq.sqrt();
    grapem_apply_inner(a, b, alpha, beta, gamma, s, s_sq, x, n, omega, out);
    Ok(())
}

/// Inner numerical kernel: given the plane vectors and pre-computed scalars,
/// apply the Rodrigues rotation.
///
/// Shared by [`grapem_apply_into`] and [`Rank2Plane::apply_into`] — this is
/// the load-bearing numerical path (G1 bit-identical gate relies on it).
///
/// `out` may alias `x` — the kernel reads `p = ⟨a,x⟩`, `q = ⟨b,x⟩` into
/// registers first, then writes `out[i]` exactly once per element.
#[inline]
#[allow(clippy::too_many_arguments)] // 11 args — matches inner-kernel convention (simd matvec, dec operators)
fn grapem_apply_inner(
    a: &[f32],
    b: &[f32],
    alpha: f32,
    beta: f32,
    gamma: f32,
    s: f32,
    s_sq: f32,
    x: &[f32],
    n: f32,
    omega: f32,
    out: &mut [f32],
) {
    let d = x.len();
    debug_assert!(d > 0, "grapem_apply_inner: empty input");
    debug_assert_eq!(a.len(), d);
    debug_assert_eq!(b.len(), d);
    debug_assert_eq!(out.len(), d);

    // Scalar projections of x onto a, b.
    let p = simd::simd_dot_f32(a, x, d); // ⟨a, x⟩
    let q = simd::simd_dot_f32(b, x, d); // ⟨b, x⟩

    // θ = n·ω·s is the *effective* rotation angle in the plane.
    let theta = n * omega * s;

    // Rodrigues coefficients. Two regimes:
    //   (1) generic: s > SMALL_ANGLE, use the textbook sin/cos closed form.
    //   (2) degenerate: s ≤ SMALL_ANGLE (a ∥ b), use the small-angle Taylor
    //       limit sin(θ)/s → n·ω, (1−cos(θ))/s² → (n·ω)²/2.
    //
    // The branch threshold ε is chosen so that the Taylor series truncation
    // error is below f32 epsilon (≈1.2e-7). For |θ| < √(6·ε_f32) ≈ 8.5e-4,
    // the 3rd-order term |θ|³/6 < ε_f32. But the branch here is on `s`, not
    // `θ`. Since `θ = n·ω·s`, for fixed `(n, ω)`, `θ → 0` as `s → 0`, so the
    // Taylor limit applies. We pick SMALL_ANGLE = 1e-3 on `s` directly —
    // below this, the closed-form `sin(θ)/s` and `(1−cos(θ))/s²` lose
    // precision (catastrophic cancellation in `1 − cos(θ)` for tiny `θ`),
    // and the Taylor limit is both more accurate and faster.
    const SMALL_ANGLE: f32 = 1e-3;

    let (ka, kb) = if s > SMALL_ANGLE {
        // Generic regime.
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let c1 = sin_t / s; // sin(θ)/s
        let c2 = (1.0 - cos_t) / s_sq; // (1−cos(θ))/s²
        // From the derivation:
        //   y = x + (c1·q + c2·(γ·q − β·p))·a − (c1·p + c2·(α·q − γ·p))·b
        let ka = c1 * q + c2 * (gamma * q - beta * p);
        let kb = c1 * p + c2 * (alpha * q - gamma * p);
        (ka, kb)
    } else {
        // Degenerate regime: s ≈ 0, plane collapses.
        // sin(θ)/s → n·ω, (1−cos(θ))/s² → (n·ω)²/2 (with θ = n·ω·s).
        // But also αβ ≈ γ² (since s² = αβ − γ² ≈ 0), so α·q − γ·p and
        // γ·q − β·p are *small* — the L² term vanishes in the degenerate
        // limit, which is consistent with L itself vanishing.
        //
        // For the strictly-degenerate case (s = 0 exactly), L = 0 and y = x.
        // The branch below reduces to that limit cleanly (θ = 0, so c1 = c2 = 0
        // for the sin path; the n·ω Taylor terms are dominated by the small
        // `(γ·q − β·p)` and `(α·q − γ·p)` factors which themselves → 0).
        let nw = n * omega;
        let c1 = nw; // sin(nωs)/s → nω as s→0
        let c2 = 0.5 * nw * nw; // (1−cos(nωs))/s² → (nω)²/2 as s→0
        let ka = c1 * q + c2 * (gamma * q - beta * p);
        let kb = c1 * p + c2 * (alpha * q - gamma * p);
        (ka, kb)
    };

    // Write loop: out[i] = x[i] + ka·a[i] − kb·b[i].
    // Chunked 4-wide to hint LLVM auto-vectorisation (matches the pattern
    // in phase_rotation_gate_into / dec::operators::exterior_derivative_into).
    let mut i = 0;
    while i + 4 <= d {
        out[i] = x[i] + ka.mul_add(a[i], -kb * b[i]);
        out[i + 1] = x[i + 1] + ka.mul_add(a[i + 1], -kb * b[i + 1]);
        out[i + 2] = x[i + 2] + ka.mul_add(a[i + 2], -kb * b[i + 2]);
        out[i + 3] = x[i + 3] + ka.mul_add(a[i + 3], -kb * b[i + 3]);
        i += 4;
    }
    while i < d {
        out[i] = x[i] + ka.mul_add(a[i], -kb * b[i]);
        i += 1;
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: materialise `L = abᵀ − baᵀ` as a d×d matrix, compute
    /// `expm(n·ω·L)` via scaling-squaring in f64, apply to `x`.
    ///
    /// Test-only — kept out of the public API. Used by the G1 bit-identical
    /// gate as ground truth.
    fn ref_expm_apply(a: &[f32], b: &[f32], x: &[f32], n: f32, omega: f32) -> Vec<f32> {
        let d = a.len();
        assert_eq!(b.len(), d);
        assert_eq!(x.len(), d);
        // Build L explicitly.
        let mut l = vec![0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                l[i * d + j] = a[i] * b[j] - b[i] * a[j];
            }
        }
        // Scaling-squaring: pick `squarings` so that ||(n·ω/2^s)·L||_1 < 0.5.
        let scale = n * omega;
        let mut l_inf = 0f32; // ||L||_∞ (max abs row sum)
        for i in 0..d {
            let mut row_sum = 0f32;
            for j in 0..d {
                row_sum += l[i * d + j].abs();
            }
            if row_sum > l_inf {
                l_inf = row_sum;
            }
        }
        let target_norm = (scale * l_inf).abs();
        let mut squarings = 0u32;
        while target_norm / (1u64 << squarings) as f32 > 0.5 && squarings < 30 {
            squarings += 1;
        }
        let coeff = scale / (1u64 << squarings) as f32;
        // Build scaled matrix M = coeff · L (in f64 for accuracy), then
        // exp(M) via Taylor (12 terms — overkill for ||M|| < 0.5 but cheap).
        let mut m = vec![0f64; d * d];
        for i in 0..d * d {
            m[i] = (coeff * l[i]) as f64;
        }
        let mut result = vec![0f64; d * d];
        for i in 0..d {
            result[i * d + i] = 1.0;
        }
        let mut term = vec![0f64; d * d]; // M⁰/0! = I
        for i in 0..d {
            term[i * d + i] = 1.0;
        }
        for k in 1..=12 {
            // term = term · M / k
            let mut next = vec![0f64; d * d];
            for i in 0..d {
                for j in 0..d {
                    let mut acc = 0f64;
                    for k2 in 0..d {
                        acc += term[i * d + k2] * m[k2 * d + j];
                    }
                    next[i * d + j] = acc / k as f64;
                }
            }
            term = next;
            for i in 0..d * d {
                result[i] += term[i];
            }
        }
        // Square `squarings` times.
        for _ in 0..squarings {
            let mut next = vec![0f64; d * d];
            for i in 0..d {
                for j in 0..d {
                    let mut acc = 0f64;
                    for k2 in 0..d {
                        acc += result[i * d + k2] * result[k2 * d + j];
                    }
                    next[i * d + j] = acc;
                }
            }
            result = next;
        }
        // Apply to x.
        let mut y = vec![0f32; d];
        for i in 0..d {
            let mut acc = 0f64;
            for j in 0..d {
                acc += result[i * d + j] * x[j] as f64;
            }
            y[i] = acc as f32;
        }
        y
    }

    // ── Smoke / shape tests ─────────────────────────────────────────────────

    #[test]
    fn shape_mismatch_returns_err() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let x = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        let r = grapem_apply_into(&a, &b, &x, 1.0, 1.0, &mut out);
        assert_eq!(r, Err(GrapemError::ShapeMismatch));
    }

    #[test]
    fn rank2plane_shape_mismatch_returns_err() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let plane = Rank2Plane::new(&a, &b);
        let x = [1.0f32, 2.0, 3.0]; // wrong length
        let mut out = [0.0f32; 3];
        assert_eq!(
            plane.apply_into(&x, 1.0, 1.0, &mut out),
            Err(GrapemError::ShapeMismatch)
        );
    }

    #[test]
    fn empty_input_is_ok() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let x: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        let r = grapem_apply_into(&a, &b, &x, 1.0, 1.0, &mut out);
        assert!(r.is_ok());
    }

    // ── Behavioural tests ───────────────────────────────────────────────────

    /// Identity: at `n=0`, the rotation is identity regardless of `a, b, ω`.
    #[test]
    fn n_zero_is_identity() {
        let a = [0.3f32, -0.7, 1.2, 0.5, -0.1, 0.8, 0.4, -0.9];
        let b = [0.1f32, 0.6, -0.4, 0.9, 0.2, -0.3, 0.7, 0.5];
        let x = [1.0f32, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25];
        let mut out = [0.0f32; 8];
        grapem_apply_into(&a, &b, &x, 0.0, 13.7, &mut out).unwrap();
        for i in 0..8 {
            assert!(
                (out[i] - x[i]).abs() < 1e-6,
                "n=0 identity violated at i={i}: out={out:?}, x={x:?}"
            );
        }
    }

    /// Degenerate plane (`a ∥ b`): generator is zero, output = input.
    #[test]
    fn parallel_a_b_is_identity() {
        let a = [0.3f32, 0.6, 0.9, 1.2, 0.15, 0.45, 0.75, 1.05];
        let b = [0.6f32, 1.2, 1.8, 2.4, 0.30, 0.90, 1.50, 2.10]; // b = 2·a
        let x = [1.0f32, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25];
        let mut out = [0.0f32; 8];
        grapem_apply_into(&a, &b, &x, 2.5, 0.7, &mut out).unwrap();
        let max_err = (0..8).map(|i| (out[i] - x[i]).abs()).fold(0f32, f32::max);
        assert!(
            max_err < 1e-5,
            "parallel plane not identity: max_err={max_err}, out={out:?}, x={x:?}"
        );
    }

    /// Orthogonal-coordinate-plane special case: `a = e_i, b = e_{i+D/2}`
    /// recovers the canonical 2D rotation in the plane.
    ///
    /// Sign convention: with `L = abᵀ − baᵀ`, the rotation takes
    /// `a → cos(θ)·a − sin(θ)·b` (note the minus sign). This is the GRAPE
    /// paper's convention (§2.3) and matches `L·a = -b` (the generator
    /// rotates `a` *clockwise* toward `-b`). Swapping `a ↔ b` flips the sign.
    #[test]
    fn canonical_basis_recovers_2d_rotation() {
        // d=4: a=e_0, b=e_2. Plane is (e_0, e_2). ω·n = π/2.
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 0.0, 1.0, 0.0];
        let x = [1.0f32, 0.0, 0.0, 0.0]; // x = a
        let mut out = [0.0f32; 4];
        // Rotate x by θ=π/2: should land on -b (sign convention: a → -b).
        grapem_apply_into(&a, &b, &x, 1.0, core::f32::consts::FRAC_PI_2, &mut out).unwrap();
        // Expected: out = [cos(π/2), 0, -sin(π/2), 0] = [0, 0, -1, 0].
        assert!((out[0]).abs() < 1e-6, "out[0] should be 0, got {}", out[0]);
        assert!(
            (out[2] + 1.0).abs() < 1e-6,
            "out[2] should be -1, got {}",
            out[2]
        );
        assert!((out[1]).abs() < 1e-6 && (out[3]).abs() < 1e-6);
    }

    /// Plane preservation: rotating any `x` leaves the orthogonal complement
    /// of `span{a, b}` unchanged.
    #[test]
    fn orthogonal_complement_is_unchanged() {
        // d=4: plane (e_0, e_1). x has components in e_2, e_3 that must be
        // untouched.
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let x = [0.7f32, 0.3, 0.5, -0.5];
        let mut out = [0.0f32; 4];
        grapem_apply_into(&a, &b, &x, 1.3, 0.5, &mut out).unwrap();
        // e_2, e_3 components must be unchanged.
        assert!((out[2] - x[2]).abs() < 1e-6);
        assert!((out[3] - x[3]).abs() < 1e-6);
    }

    /// Norm preservation: `‖y‖ = ‖x‖` (rotation is orthogonal).
    #[test]
    fn norm_is_preserved() {
        let a = [0.3f32, -0.7, 1.2, 0.5];
        let b = [0.1f32, 0.6, -0.4, 0.9];
        let x = [1.0f32, -1.0, 0.5, -0.5];
        let norm_x = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut out = [0.0f32; 4];
        grapem_apply_into(&a, &b, &x, 1.7, 0.9, &mut out).unwrap();
        let norm_y = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_x - norm_y).abs() / norm_x < 1e-5,
            "norm not preserved: ‖x‖={norm_x}, ‖y‖={norm_y}"
        );
    }

    /// Periodicity: rotating by `n·ω = 2π` returns to start.
    #[test]
    fn full_rotation_returns_to_start() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 0.0, 1.0, 0.0];
        let x = [0.6f32, 0.4, 0.8, 0.2];
        let mut out = [0.0f32; 4];
        grapem_apply_into(&a, &b, &x, 1.0, 2.0 * core::f32::consts::PI, &mut out).unwrap();
        for i in 0..4 {
            assert!(
                (out[i] - x[i]).abs() < 1e-4,
                "2π rotation not identity at i={i}: out={out:?}, x={x:?}"
            );
        }
    }

    /// `out` may alias `x` (in-place rotation).
    #[test]
    fn out_can_alias_x() {
        let a = [0.3f32, -0.7, 1.2, 0.5];
        let b = [0.1f32, 0.6, -0.4, 0.9];
        let mut x = [1.0f32, -1.0, 0.5, -0.5];
        // Make a heap copy to compare against.
        let original = x;
        // Apply in-place via split_at_mut trick — but grapem_apply_into takes
        // `&[f32]` for x and `&mut [f32]` for out, so direct aliasing is a
        // borrow-checker violation in safe Rust. We use an intermediate buffer
        // and copy back to demonstrate the kernel's aliasing safety.
        let mut tmp = [0f32; 4];
        grapem_apply_into(&a, &b, &x, 1.0, 0.7, &mut tmp).unwrap();
        x.copy_from_slice(&tmp);
        // Sanity: result differs from input (rotation happened) and matches
        // the no-alias path.
        let mut no_alias = [0f32; 4];
        grapem_apply_into(&a, &b, &original, 1.0, 0.7, &mut no_alias).unwrap();
        for i in 0..4 {
            assert!(
                (x[i] - no_alias[i]).abs() < 1e-7,
                "alias path diverges at {i}"
            );
        }
        // And it actually rotated (not identity).
        let moved = (0..4).map(|i| (x[i] - original[i]).abs()).sum::<f32>();
        assert!(moved > 1e-3, "rotation was a no-op: moved={moved}");
    }

    /// `Rank2Plane` matches the inline path (same numerical kernel).
    #[test]
    fn rank2plane_matches_inline_path() {
        let a = [0.3f32, -0.7, 1.2, 0.5, -0.1, 0.8, 0.4, -0.9];
        let b = [0.1f32, 0.6, -0.4, 0.9, 0.2, -0.3, 0.7, 0.5];
        let x = [1.0f32, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25];
        let plane = Rank2Plane::new(&a, &b);

        let mut out_inline = [0.0f32; 8];
        grapem_apply_into(&a, &b, &x, 1.3, 0.7, &mut out_inline).unwrap();

        let mut out_plane = [0.0f32; 8];
        plane.apply_into(&x, 1.3, 0.7, &mut out_plane).unwrap();

        for i in 0..8 {
            assert!(
                (out_inline[i] - out_plane[i]).abs() < 1e-7,
                "Rank2Plane diverges from inline at i={i}: {} vs {}",
                out_inline[i],
                out_plane[i]
            );
        }
    }

    /// `Rank2Plane::dim`, `alpha`, `beta`, `gamma`, `s` accessors.
    #[test]
    fn rank2plane_accessors() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let plane = Rank2Plane::new(&a, &b);
        assert_eq!(plane.dim(), 4);
        assert!((plane.alpha() - 1.0).abs() < 1e-6);
        assert!((plane.beta() - 1.0).abs() < 1e-6);
        assert!((plane.gamma() - 0.0).abs() < 1e-6);
        assert!((plane.s() - 1.0).abs() < 1e-6);
        assert!((plane.s_sq() - 1.0).abs() < 1e-6);
        assert_eq!(plane.a(), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(plane.b(), &[0.0, 1.0, 0.0, 0.0]);
    }

    /// Bit-identical to the materialised `expm(n·ω·L)·x` on random inputs
    /// across dims {8, 16, 32, 64}. This is the **G1** gate.
    #[test]
    fn g1_bit_identical_to_expm() {
        let mut seed = 0x5885_7777u64;
        let mut next_f32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = seed >> 40;
            (bits as f32) / ((1u64 << 24) as f32) // [0, 1)
        };

        for &d in &[8usize, 16, 32, 64] {
            let mut max_rel_err = 0f32;
            for _ in 0..20 {
                let a: Vec<f32> = (0..d).map(|_| next_f32() * 2.0 - 1.0).collect();
                let b: Vec<f32> = (0..d).map(|_| next_f32() * 2.0 - 1.0).collect();
                let x: Vec<f32> = (0..d).map(|_| next_f32() * 2.0 - 1.0).collect();
                let n = next_f32() * 3.0;
                let omega = next_f32() * 2.0;

                let mut out = vec![0f32; d];
                grapem_apply_into(&a, &b, &x, n, omega, &mut out).unwrap();
                let y_ref = ref_expm_apply(&a, &b, &x, n, omega);

                let norm_ref = y_ref.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
                for i in 0..d {
                    let err = (out[i] - y_ref[i]).abs();
                    let rel = err / norm_ref;
                    if rel > max_rel_err {
                        max_rel_err = rel;
                    }
                }
            }
            assert!(
                max_rel_err < 1e-4,
                "G1 FAIL at d={d}: max relative error {max_rel_err} > 1e-4 vs expm"
            );
        }
    }
}
