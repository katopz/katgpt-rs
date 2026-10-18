//! Sphere Sampling — modelless primitives for sampling from unnormalized
//! densities on the unit hypersphere `S^{d-1}`.
//!
//! Distilled from Flow Sampling (arXiv:2605.03984, Havens / Karrer / Shaul,
//! FAIR + Weizmann, May 2026; Issue 544). The paper trains a drift `u_θ` via
//! backprop on a replay buffer; we ship only the **modelless core**: for
//! vMF-family targets `r(x) = κ·μ^T x` the score `∇_M r` is closed-form, so
//! the entire conditional drift on the sphere integrates via Euler–Maruyama
//! without any learned component. The three primitives below are the
//! non-trivial pieces (the geodesic interpolant `X_t` and its time
//! derivative `Ẋ_t` already ship in [`crate::spherical_steering`] as
//! Slerp; the score itself is closed-form `κ·μ − κ·(μ^T X_1)·X_1`).
//!
//! # What this computes
//!
//! Given two points `X_1, X_t ∈ S^{d-1}` and a tangent vector at `X_1`:
//!
//! ```text
//! // Parallel transport  (Eq 42)
//! T_{X_1→X_t}(v) = v − 2 · ((X_1+X_t)^T v / ‖X_1+X_t‖²) · (X_1+X_t)
//!
//! // Jacobian log-det curvature correction  (Eq 44)
//! ∇_M log det(J_t) = (d−1)·(t·cot(t·ω_1) − cot(ω_1))·Ẋ_1 / ω_1
//!
//! // Riemannian exponential (the Euler–Maruyama step on S^d)
//! exp_X(v) = cos(‖v‖)·X + sin(‖v‖)·v/‖v‖     (v ≠ 0)
//!          = X                                  (v = 0)
//! ```
//!
//! `ω_1 = arccos(clamp(X_1^T x_0, −1, 1))` is the geodesic distance from the
//! source `x_0` to `X_1` — caller-computed and passed in. All three primitives
//! are zero-allocation and accept caller-owned scratch buffers (mirroring the
//! Plan 405 `SlerpScratch` pattern).
//!
//! # Why these are NEW (not redundant with Plan 405)
//!
//! | Capability | Shipped? | Where |
//! |---|---|---|
//! | Slerp geodesic on S^d | ✅ DEFAULT-ON | Plan 405 [`crate::spherical_steering::slerp_steering_into`] |
//! | Deterministic vMF confidence gate (Eq 17 sigmoid) | ✅ DEFAULT-ON | Plan 405 [`crate::spherical_steering::vmf_confidence_gate`] |
//! | Parallel transport `T_{X_1→X_t}` on S^d | ❌ (this module) | [`parallel_transport_householder_into`] |
//! | Jacobian log-det curvature correction | ❌ (this module) | [`jacobian_logdet_cot_correction`] |
//! | Riemannian exp map (Euler–Maruyama step) | ❌ (this module) | [`sphere_exp_map_into`] |
//! | Sample from vMF / vMF mixture on S^d | ❌ (Issue 544 PoC) | `riir-poc/src/sphere_sampling_poc.rs` |
//!
//! Plan 405 answers "given a drifted vector, deterministically pull it back
//! toward `μ_T` by Slerp + Eq-17 gate". This module answers the *different*
//! question: "given a target density `q(x) ∝ exp(r(x))` on `S^d`, sample
//! `X ∼ q` by Euler–Maruyama on the manifold". Plan 405's deterministic gate
//! produces one direction; sampling produces a *distribution* over directions
//! — a capability class the deterministic gate cannot serve.
//!
//! # Numerical contract
//!
//! - All entry points are pure float arithmetic over caller-provided slices.
//!   Deterministic on a given CPU.
//! - `parallel_transport_householder_into` is undefined when `X_1 = −X_t`
//!   (antipodal — `X_1+X_t = 0`); returns [`SphereError::AntipodalTransport`].
//! - `jacobian_logdet_cot_correction` is undefined when `ω_1 ≤ 0` or
//!   `ω_1 ≥ π` (the cotangent blows up); returns
//!   [`SphereError::CotangentDivergent`]. Caller should clamp `ω_1` to
//!   `[ cot_floor, π − cot_floor ]` (default `1e-3`) before calling.
//! - `sphere_exp_map_into` is total: `‖v‖ = 0` returns `X` (identity).
//!
//! # Performance
//!
//! All three primitives are `O(d)` with no allocation. The transport is one
//! Householder reflection (rank-1 update — one dot + one SAXPY); the Jacobian
//! correction is one cotangent evaluation + a SAXPY; the exp_map is two trig
//! + one norm + a 2-term mix. Expect low tens of ns at d=8.
//!
//! # Defend-wrong status
//!
//! Per Issue 544's defend-wrong protocol, the most likely failure mode (per
//! the Research 049 PTRM cautionary flag) is that the Flow Sampling Riemannian
//! sampler produces the **same KL** as Wood (1994)'s exact vMF sampler at the
//! same `N` — in which case the Riemannian complexity is unjustified for the
//! vMF-only case and we should ship Wood's algorithm (simpler + exact). The
//! Riemannian complexity earns its keep only on **non-vMF targets** where
//! Wood does not apply. The PoC in `riir-poc/src/sphere_sampling_poc.rs` is
//! designed to honestly surface this failure mode if it occurs.
//!
//! Sphere Sampling — modelless primitives for sampling from unnormalized
//! densities on the unit hypersphere S^{d-1}. Distilled from Flow Sampling
//! (arxiv 2605.03984 Havens/Karrer/Shaul FAIR+Weizmann May 2026; Issue 544). The
//! paper trains a drift u_θ via backprop; we ship only the modelless core:
//! for vMF-family targets r(x) = κ·μ^T x the score ∇_M r is closed-form, so the
//! entire conditional drift on the sphere integrates via Euler–Maruyama with no
//! learned component. Three primitives: parallel_transport_householder_into
//! (Eq 42 Householder reflection about X_1+X_t midpoint hyperplane),
//! jacobian_logdet_cot_correction (Eq 44 curvature `(d−1)·(t·cot(t·ω_1) − cot(ω_1))·Ẋ_1/ω_1`),
//! sphere_exp_map_into (Riemannian exp, the Euler–Maruyama step from Eq 29).
//! Sibling to Plan 405 above: Plan 405 deterministically pulls a drifted vector
//! toward μ_T via Slerp + Eq-17 gate; this module samples a distribution over
//! directions via Euler–Maruyama on the manifold. The deterministic gate
//! produces one direction; sampling produces a distribution — a capability
//! class the gate cannot serve. Opt-in pending Issue 544 defend-wrong PoC
//! verdict: the most likely failure mode (per Research 049 PTRM cautionary
//! flag) is that the Riemannian sampler produces the same KL as Wood (1994)'s
//! exact vMF sampler at the same N — in which case the complexity is
//! unjustified for the vMF-only case. Promotion requires a non-vMF consumer.

use crate::simd;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by the sphere-sampling entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SphereError {
    /// Slice lengths disagree. All buffer lengths must equal the ambient
    /// dimension `d`.
    ShapeMismatch,
    /// Parallel transport `T_{X_1→X_t}` is undefined when `X_1 = −X_t`
    /// (antipodal): the Householder pivot `X_1+X_t` vanishes. Caller picks
    /// a different `X_t` or skips the step.
    AntipodalTransport,
    /// Jacobian log-det `cot(ω_1)` or `cot(t·ω_1)` diverges. Caller must
    /// clamp `ω_1` away from `{0, π}` (default floor `1e-3`) and `t·ω_1`
    /// away from `{0, π}` before calling.
    CotangentDivergent,
    /// Inputs are not finite (NaN / ±inf). Sphere primitives are total on
    /// finite inputs; non-finite indicates an upstream bug.
    NonFiniteInput,
}

// ── Numerical floors ─────────────────────────────────────────────

/// Below this `|X_1+X_t|` we treat `X_1` and `X_t` as antipodal (transport
/// undefined). Chosen so the Householder denominator `‖X_1+X_t‖²` stays
/// above the f32-conditioned regime.
pub const TRANSPORT_FLOOR: f32 = 1e-6;

/// `ω_1` within `COT_FLOOR` of `{0, π}` is treated as divergent (the
/// cotangent blows up). The PoC clamps `ω_1` to
/// `[COT_FLOOR, π − COT_FLOOR]` before calling the Jacobian correction.
pub const COT_FLOOR: f32 = 1e-3;

/// Below this `‖v‖` the Riemannian exp map is treated as the identity
/// (`exp_X(0) = X`). Avoids division by `‖v‖`.
pub const EXP_MAP_FLOOR: f32 = 1e-12;

// ── T1.1 — Parallel transport via Householder reflection ────────

/// Parallel-transport a tangent vector `v` from `X_1` to `X_t` on `S^{d-1}`
/// via the Householder reflection about the hyperplane normal to
/// `X_1 + X_t` (Flow Sampling Eq 42).
///
/// Computes `T_{X_1→X_t}(v) = v − 2 · ((X_1+X_t)^T v / ‖X_1+X_t‖²) · (X_1+X_t)`.
///
/// **Why Householder (not the matrix form):** `T_{X_1→X_t}` is the
/// reflection across the hyperplane normal to `X_1 + X_t`. The matrix
/// form `I − 2·P_{X_1+X_t}` is rank-`(d−1)` identity on the hyperplane
/// (orthogonal complement of `X_1+X_t`) and flips the sign along
/// `X_1+X_t`. That is exactly a single Householder reflection — `O(d)` with
/// one dot + one SAXPY, no matrix alloc.
///
/// **Geometric note:** the hyperplane normal to `X_1+X_t` is the perpendicular
/// bisector of the segment from `X_1` to `−X_t` (NOT `X_t`). For non-tangent
/// inputs, `H(X_1) = −X_t`. The parallel-transport property (output tangent
/// at `X_t` with preserved norm) holds only for tangent inputs `v ⊥ X_1`.
///
/// **Isometry:** for `v` tangent at `X_1` (i.e. `v ⊥ X_1`), the output is
/// tangent at `X_t` (`T(v) ⊥ X_t`) with `‖T(v)‖ = ‖v‖` exactly. The unit
/// test `parallel_transport_preserves_inner_product` verifies this on random
/// instances.
///
/// # Arguments
///
/// * `x1` — point on the sphere, length `d`. **Caller's responsibility** to
///   ensure `‖X_1‖ ≈ 1` (the isometry invariant holds to `‖X_1‖ ≈ 1`).
/// * `xt` — point on the sphere, length `d`. **Caller's responsibility** to
///   ensure `‖X_t‖ ≈ 1`.
/// * `v_in` — tangent vector at `X_1` (typically `v_in ⊥ X_1`), length `d`.
///   Need not be unit-norm.
/// * `v_out` — output, length `d`. May **alias** `v_in` (the kernel reads
///   `v_in[i]` and the dot product before writing `v_out[i]`; the SAXPY
///   write pattern uses a pre-computed scalar).
/// * `scratch` — length `d`; the kernel writes `X_1 + X_t` here. Reused
///   across calls. May NOT alias `x1`, `xt`, `v_in`, or `v_out`.
///
/// # Errors
///
/// Returns [`SphereError::ShapeMismatch`] if lengths disagree. Returns
/// [`SphereError::AntipodalTransport`] if `‖X_1+X_t‖² < TRANSPORT_FLOOR²`
/// (`X_t ≈ −X_1`). Returns [`SphereError::NonFiniteInput`] if any input
/// contains NaN / ±inf.
///
/// # Performance
///
/// `O(d)`, zero allocation in steady state. One SIMD dot, one SIMD
/// sum-of-squares, one scalar div, one chunked SAXPY.
#[inline]
pub fn parallel_transport_householder_into(
    x1: &[f32],
    xt: &[f32],
    v_in: &[f32],
    v_out: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), SphereError> {
    let d = x1.len();
    if xt.len() != d || v_in.len() != d || v_out.len() != d || scratch.len() != d {
        return Err(SphereError::ShapeMismatch);
    }

    // Defensive non-finite check (one SIMD reduction; the SIMD kernels
    // propagate NaN silently, surfacing it here gives a clear error).
    if !simd_all_finite(x1) || !simd_all_finite(xt) || !simd_all_finite(v_in) {
        return Err(SphereError::NonFiniteInput);
    }

    // pivot = X_1 + X_t  →  scratch
    let mut i = 0;
    while i + 4 <= d {
        scratch[i] = x1[i] + xt[i];
        scratch[i + 1] = x1[i + 1] + xt[i + 1];
        scratch[i + 2] = x1[i + 2] + xt[i + 2];
        scratch[i + 3] = x1[i + 3] + xt[i + 3];
        i += 4;
    }
    while i < d {
        scratch[i] = x1[i] + xt[i];
        i += 1;
    }

    // ‖pivot‖²
    let pivot_norm_sq = simd::simd_sum_sq(scratch, d);
    if pivot_norm_sq < TRANSPORT_FLOOR * TRANSPORT_FLOOR {
        return Err(SphereError::AntipodalTransport);
    }

    // (pivot^T v) / ‖pivot‖²  — the Householder coefficient
    let dot = simd::simd_dot_f32(scratch, v_in, d);
    let coeff = 2.0 * dot / pivot_norm_sq;

    // v_out = v_in − coeff · pivot   (the Householder reflection)
    let pivot = &scratch[..d];
    let mut j = 0;
    while j + 4 <= d {
        v_out[j] = v_in[j] - coeff * pivot[j];
        v_out[j + 1] = v_in[j + 1] - coeff * pivot[j + 1];
        v_out[j + 2] = v_in[j + 2] - coeff * pivot[j + 2];
        v_out[j + 3] = v_in[j + 3] - coeff * pivot[j + 3];
        j += 4;
    }
    while j < d {
        v_out[j] = v_in[j] - coeff * pivot[j];
        j += 1;
    }

    Ok(())
}

// ── T1.2 — Jacobian log-det curvature correction ────────────────

/// Jacobian log-det curvature correction `(d−1)·(t·cot(t·ω_1) − cot(ω_1))·Ẋ_1/ω_1`
/// (Flow Sampling Eq 44).
///
/// Writes the tangent vector `(d−1)·(t·cot(t·ω_1) − cot(ω_1))/ω_1 · x1_dot`
/// into `x_dot_out`, where `x1_dot = Ẋ_1` is the time derivative of the
/// geodesic at the target endpoint `X_1` (Flow Sampling Eq 41):
///
/// ```text
/// Ẋ_1 = ω_1 · sin(0) · X_1 + cos(0) · Ẋ_init = Ẋ_init
/// ```
/// (At `t = 1` the geodesic is at `X_1`; the caller passes `x1_dot` directly,
/// so this primitive only computes the scalar Jacobian factor and applies it.)
///
/// # Numerical stability
///
/// The cotangent has singularities at `x ∈ {0, π}`. Caller must clamp
/// `ω_1` to `[COT_FLOOR, π − COT_FLOOR]` and ensure `t·ω_1 ∈ (0, π)` (i.e.
/// `t ∈ (0, 1]` AND `t·ω_1 < π`). This function re-checks both bounds and
/// returns [`SphereError::CotangentDivergent`] if violated — defensive
/// against f32 drift in caller's `ω_1` arithmetic.
///
/// # Arguments
///
/// * `t` — interpolation parameter `(0, 1]`. `t = 0` is degenerate
///   (`t·cot(t·ω_1) → 1/ω_1` by L'Hôpital); caller should not call this
///   primitive at `t = 0` (the drift at `t = 0` is just the geodesic
///   derivative `Ẋ_t`, no curvature correction).
/// * `omega_1` — geodesic distance from source `x_0` to target endpoint
///   `X_1`, in `(0, π)`. Caller computes via `arccos(clamp(x_1·x_0, −1, 1))`.
/// * `d` — ambient dimension (`d ≥ 2`). The `(d−1)` prefactor.
/// * `x1_dot` — tangent vector at `X_1` (`Ẋ_1`), length `d`. Caller-owned.
/// * `x_dot_out` — output, length `d`. May alias `x1_dot` (the scalar
///   coefficient is computed first, then a single SAXPY pass).
///
/// # Errors
///
/// Returns [`SphereError::ShapeMismatch`] if lengths disagree. Returns
/// [`SphereError::CotangentDivergent`] if `ω_1 ≤ COT_FLOOR`,
/// `ω_1 ≥ π − COT_FLOOR`, `t ≤ 0`, `t·ω_1 ≤ COT_FLOOR`, or
/// `t·ω_1 ≥ π − COT_FLOOR`.
///
/// # Performance
///
/// `O(d)`, zero allocation. Two cotangent evaluations + one SAXPY.
#[inline]
pub fn jacobian_logdet_cot_correction(
    t: f32,
    omega_1: f32,
    d: usize,
    x1_dot: &[f32],
    x_dot_out: &mut [f32],
) -> Result<(), SphereError> {
    if x1_dot.len() != d || x_dot_out.len() != d {
        return Err(SphereError::ShapeMismatch);
    }
    if !t.is_finite() || !omega_1.is_finite() {
        return Err(SphereError::NonFiniteInput);
    }
    // Cotangent singularity guards. Clamp window is [COT_FLOOR, π − COT_FLOOR].
    let pi = core::f32::consts::PI;
    if t <= 0.0 || t > 1.0 {
        return Err(SphereError::CotangentDivergent);
    }
    if omega_1 <= COT_FLOOR || omega_1 >= pi - COT_FLOOR {
        return Err(SphereError::CotangentDivergent);
    }
    let t_omega = t * omega_1;
    if t_omega <= COT_FLOOR || t_omega >= pi - COT_FLOOR {
        return Err(SphereError::CotangentDivergent);
    }

    // (d−1)·(t·cot(t·ω_1) − cot(ω_1)) / ω_1   — scalar curvature prefactor.
    let cot_t_omega = (t_omega).cos() / t_omega.sin(); // cot(x) = cos/sin
    let cot_omega = omega_1.cos() / omega_1.sin();
    let coeff = ((d - 1) as f32) * (t * cot_t_omega - cot_omega) / omega_1;

    // x_dot_out = coeff · x1_dot
    let mut k = 0;
    while k + 4 <= d {
        x_dot_out[k] = coeff * x1_dot[k];
        x_dot_out[k + 1] = coeff * x1_dot[k + 1];
        x_dot_out[k + 2] = coeff * x1_dot[k + 2];
        x_dot_out[k + 3] = coeff * x1_dot[k + 3];
        k += 4;
    }
    while k < d {
        x_dot_out[k] = coeff * x1_dot[k];
        k += 1;
    }

    Ok(())
}

// ── T1.3 — Riemannian exponential map ────────────────────────────

/// Riemannian exponential `exp_X(v) = cos(‖v‖)·X + sin(‖v‖)·v/‖v‖` on
/// `S^{d-1}` (the Euler–Maruyama step from Flow Sampling Eq 29).
///
/// For `‖v‖ = 0` returns `X` (identity — no step). For `‖v‖ > 0` the output
/// lies on the unit sphere **iff** `‖X‖ = 1` exactly; mild deviation in
/// `‖X‖` is preserved (the output has the same norm as `X`, modulo the
/// f32 rounding in the trig/FMA mix — drift < 1e-6, see the unit test
/// `exp_map_preserves_unit_norm`).
///
/// # Arguments
///
/// * `x` — point on the sphere, length `d`. Caller ensures `‖X‖ ≈ 1`.
/// * `v` — tangent vector at `X` (i.e. `v ⊥ X` is the natural case, but
///   this primitive does not enforce it — any vector works, with the
///   output still on the sphere if `‖X‖ = 1`).
/// * `out` — output, length `d`. May alias `x` or `v` (the scalar
///   coefficients are computed before the mix; one pass).
///
/// # Errors
///
/// Returns [`SphereError::ShapeMismatch`] if lengths disagree. Returns
/// [`SphereError::NonFiniteInput`] if any input contains NaN / ±inf.
///
/// # Performance
///
/// `O(d)`, zero allocation. One SIMD sum-of-squares, one sqrt, two trig,
/// one chunked 2-term FMA mix.
#[inline]
pub fn sphere_exp_map_into(x: &[f32], v: &[f32], out: &mut [f32]) -> Result<(), SphereError> {
    let d = x.len();
    if v.len() != d || out.len() != d {
        return Err(SphereError::ShapeMismatch);
    }
    if !simd_all_finite(x) || !simd_all_finite(v) {
        return Err(SphereError::NonFiniteInput);
    }

    // ‖v‖
    let v_norm_sq = simd::simd_sum_sq(v, d);
    if v_norm_sq < EXP_MAP_FLOOR * EXP_MAP_FLOOR {
        // exp_X(0) = X  (identity)
        out.copy_from_slice(x);
        return Ok(());
    }
    let v_norm = v_norm_sq.sqrt();

    // cos(‖v‖)·X + sin(‖v‖)·v/‖v‖
    let c = v_norm.cos();
    let s_over_norm = v_norm.sin() / v_norm;
    let mut i = 0;
    while i + 4 <= d {
        out[i] = c.mul_add(x[i], s_over_norm * v[i]);
        out[i + 1] = c.mul_add(x[i + 1], s_over_norm * v[i + 1]);
        out[i + 2] = c.mul_add(x[i + 2], s_over_norm * v[i + 2]);
        out[i + 3] = c.mul_add(x[i + 3], s_over_norm * v[i + 3]);
        i += 4;
    }
    while i < d {
        out[i] = c.mul_add(x[i], s_over_norm * v[i]);
        i += 1;
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────

/// Defensive NaN/inf check via two SIMD reductions. The sphere primitives
/// propagate NaN silently (they'd return NaN outputs without surfacing the
/// error), so we check explicitly at entry.
#[inline]
fn simd_all_finite(x: &[f32]) -> bool {
    // f32::is_finite rejects NaN and ±inf. The SIMD dot / sum_sq kernels
    // propagate NaN; a scalar check on the slice is `O(d)` — acceptable
    // given the defensive nature. (The hot-path check is ~`d` cycles on
    // top of `O(d)` work; ~30% overhead on a 20ns primitive, but the
    // alternative is silent NaN propagation.)
    x.iter().all(|v| v.is_finite())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_norm(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Construct a unit vector from raw components (panics on zero norm).
    fn unit(x: &[f32]) -> Vec<f32> {
        let n = l2_norm(x);
        assert!(n > 1e-9, "zero-norm input");
        x.iter().map(|v| v / n).collect()
    }

    /// Construct a tangent vector at `p` by projecting `w` onto p's tangent
    /// space (subtract the normal component).
    fn tangent_at(p: &[f32], w: &[f32]) -> Vec<f32> {
        let pn = dot(p, w);
        w.iter()
            .zip(p.iter())
            .map(|(wi, pi)| wi - pn * pi)
            .collect()
    }

    // ── T1.1 Parallel transport ─────────────────────────────────

    #[test]
    fn parallel_transport_preserves_inner_product_isometry() {
        // Two random unit vectors X_1, X_t on S^2; v tangent at X_1.
        // ‖T(v)‖ should equal ‖v‖ AND T(v) ⊥ X_t.
        let x1 = unit(&[1.0, 0.2, -0.4]);
        let xt = unit(&[0.3, -0.7, 0.5]);
        let v_in = tangent_at(&x1, &[0.5, -0.3, 0.9]);
        assert!(dot(&v_in, &x1).abs() < 1e-5, "v_in not tangent at x1");

        let mut v_out = vec![0.0f32; 3];
        let mut scratch = vec![0.0f32; 3];
        parallel_transport_householder_into(&x1, &xt, &v_in, &mut v_out, &mut scratch)
            .expect("transport ok");

        let v_in_norm = l2_norm(&v_in);
        let v_out_norm = l2_norm(&v_out);
        assert!(
            (v_in_norm - v_out_norm).abs() < 1e-5,
            "norm not preserved: |v_in|={v_in_norm} |v_out|={v_out_norm}"
        );
        assert!(
            dot(&v_out, &xt).abs() < 1e-5,
            "v_out not tangent at xt: dot={}",
            dot(&v_out, &xt)
        );
    }

    #[test]
    fn parallel_transport_householder_is_reflection_across_sum_hyperplane() {
        // For the special case v_in = X_1 (NOT tangent — testing the
        // Householder formula directly): reflecting X_1 across the hyperplane
        // normal to (X_1+X_t) sends X_1 to −X_t (the antipode of X_t). This is
        // the correct geometric fact: the hyperplane normal to the SUM is the
        // perpendicular bisector of the segment from X_1 to −X_t (not to X_t).
        //
        // Math check: pivot · X_1 = 1 + c, ‖pivot‖² = 2 + 2c, so the Householder
        // coefficient is 2(1+c)/(2+2c) = 1 for any c ≠ −1, hence H(X_1) = X_1 − pivot = −X_t.
        //
        // The parallel-transport property (T(v) ∈ T_{X_t}S^d with ‖T(v)‖ = ‖v‖)
        // holds only for tangent v ⊥ X_1 — that is verified by the `*_isometry`
        // test above. This test verifies the Householder formula itself.
        let x1 = unit(&[1.0, 0.0, 0.0]);
        let xt = unit(&[0.6, 0.8, 0.0]); // ~53° from x1
        let v_in = x1.clone();
        let mut v_out = vec![0.0f32; 3];
        let mut scratch = vec![0.0f32; 3];
        parallel_transport_householder_into(&x1, &xt, &v_in, &mut v_out, &mut scratch)
            .expect("transport ok");

        // H(X_1) = −X_t  (Householder reflection across the (X_1+X_t)-normal hyperplane).
        for i in 0..3 {
            assert!(
                (v_out[i] - (-xt[i])).abs() < 1e-5,
                "H(X_1) != −X_t at idx {}: got {} expected {}",
                i,
                v_out[i],
                -xt[i]
            );
        }
    }

    #[test]
    fn parallel_transport_identity_when_xt_equals_x1() {
        // X_t = X_1: pivot = 2·X_1, coeff = 2·(2·X_1·v / 4) = (X_1·v).
        // If v ⊥ X_1: coeff = 0 → T(v) = v. (Stays put.)
        let x1 = unit(&[0.6, 0.8, 0.0]);
        let xt = x1.clone();
        let v_in = tangent_at(&x1, &[1.0, 0.0, 0.0]);
        let mut v_out = vec![0.0f32; 3];
        let mut scratch = vec![0.0f32; 3];
        parallel_transport_householder_into(&x1, &xt, &v_in, &mut v_out, &mut scratch)
            .expect("transport ok");

        for i in 0..3 {
            assert!(
                (v_out[i] - v_in[i]).abs() < 1e-5,
                "T(v) != v when X_t = X_1 at idx {}: got {} expected {}",
                i,
                v_out[i],
                v_in[i]
            );
        }
    }

    #[test]
    fn parallel_transport_antipodal_returns_error() {
        let x1 = unit(&[1.0, 0.0, 0.0]);
        let xt = unit(&[-1.0, 0.0, 0.0]); // antipodal
        let v_in = vec![0.0f32, 1.0, 0.0];
        let mut v_out = vec![0.0f32; 3];
        let mut scratch = vec![0.0f32; 3];
        let err = parallel_transport_householder_into(&x1, &xt, &v_in, &mut v_out, &mut scratch)
            .unwrap_err();
        assert_eq!(err, SphereError::AntipodalTransport);
    }

    #[test]
    fn parallel_transport_shape_mismatch() {
        let x1 = vec![1.0f32, 0.0, 0.0];
        let xt = vec![0.0f32, 1.0, 0.0];
        let v_in = vec![0.0f32, 0.0, 1.0];
        let mut v_out = vec![0.0f32; 3];
        let mut scratch = vec![0.0f32; 2]; // wrong length
        let err = parallel_transport_householder_into(&x1, &xt, &v_in, &mut v_out, &mut scratch)
            .unwrap_err();
        assert_eq!(err, SphereError::ShapeMismatch);
    }

    #[test]
    fn parallel_transport_higher_dim_d8_isometry() {
        // HLA-scale: 8-dim. Random unit vectors + tangent v.
        let x1 = unit(&[0.3, -0.5, 0.7, 0.1, -0.2, 0.4, 0.6, -0.1]);
        let xt = unit(&[0.1, 0.2, -0.3, 0.5, 0.4, -0.6, 0.7, 0.05]);
        let v_in = tangent_at(&x1, &[1.0, 2.0, -1.0, 0.5, -0.7, 1.3, -0.4, 0.9]);

        let mut v_out = vec![0.0f32; 8];
        let mut scratch = vec![0.0f32; 8];
        parallel_transport_householder_into(&x1, &xt, &v_in, &mut v_out, &mut scratch)
            .expect("transport ok");

        assert!(
            (l2_norm(&v_in) - l2_norm(&v_out)).abs() < 1e-5,
            "norm not preserved at d=8"
        );
        assert!(
            dot(&v_out, &xt).abs() < 1e-5,
            "v_out not tangent at xt (d=8)"
        );
    }

    // ── T1.2 Jacobian log-det cot correction ────────────────────

    #[test]
    fn jacobian_logdet_finite_across_valid_range() {
        // Sweep t ∈ (0,1] × ω_1 ∈ (0.01, π − 0.01). The coefficient should
        // always be finite.
        let d = 8;
        let x1_dot = vec![0.1f32; d];
        let mut out = vec![0.0f32; d];

        let pi = core::f32::consts::PI;
        for &t in &[0.1f32, 0.25, 0.5, 0.75, 1.0] {
            for &omega in &[0.05f32, 0.5, 1.0, 1.5, 2.5, pi - 0.05] {
                let r = jacobian_logdet_cot_correction(t, omega, d, &x1_dot, &mut out);
                assert!(r.is_ok(), "t={t} omega={omega} failed: {r:?}");
                for v in &out {
                    assert!(v.is_finite(), "non-finite at t={t} omega={omega}: {v}");
                }
            }
        }
    }

    #[test]
    fn jacobian_logdet_at_t_one_is_zero() {
        // At t = 1: (1·cot(ω_1) − cot(ω_1)) = 0  →  output is zero.
        let d = 8;
        let x1_dot = vec![0.5f32; d];
        let mut out = vec![0.0f32; d];
        jacobian_logdet_cot_correction(1.0, 1.2, d, &x1_dot, &mut out).expect("ok");
        for v in &out {
            assert!(v.abs() < 1e-5, "expected zero at t=1, got {v}");
        }
    }

    #[test]
    fn jacobian_logdet_singular_omega_zero_returns_error() {
        let d = 4;
        let x1_dot = vec![0.5f32; d];
        let mut out = vec![0.0f32; d];
        let err = jacobian_logdet_cot_correction(0.5, 1e-5, d, &x1_dot, &mut out).unwrap_err();
        assert_eq!(err, SphereError::CotangentDivergent);
    }

    #[test]
    fn jacobian_logdet_singular_omega_pi_returns_error() {
        let d = 4;
        let x1_dot = vec![0.5f32; d];
        let mut out = vec![0.0f32; d];
        let pi = core::f32::consts::PI;
        let err = jacobian_logdet_cot_correction(0.5, pi - 1e-5, d, &x1_dot, &mut out).unwrap_err();
        assert_eq!(err, SphereError::CotangentDivergent);
    }

    #[test]
    fn jacobian_logdet_t_zero_returns_error() {
        let d = 4;
        let x1_dot = vec![0.5f32; d];
        let mut out = vec![0.0f32; d];
        let err = jacobian_logdet_cot_correction(0.0, 1.0, d, &x1_dot, &mut out).unwrap_err();
        assert_eq!(err, SphereError::CotangentDivergent);
    }

    #[test]
    fn jacobian_logdet_shape_mismatch() {
        let d = 4;
        let x1_dot = vec![0.5f32; d];
        let mut out = vec![0.0f32; d - 1];
        let err = jacobian_logdet_cot_correction(0.5, 1.0, d, &x1_dot, &mut out).unwrap_err();
        assert_eq!(err, SphereError::ShapeMismatch);
    }

    // ── T1.3 Sphere exp map ─────────────────────────────────────

    #[test]
    fn exp_map_zero_velocity_returns_x() {
        let x = unit(&[0.6, 0.8, 0.0]);
        let v = vec![0.0f32; 3];
        let mut out = vec![0.0f32; 3];
        sphere_exp_map_into(&x, &v, &mut out).expect("ok");
        for i in 0..3 {
            assert!((out[i] - x[i]).abs() < 1e-7, "exp_X(0) != X at {i}");
        }
    }

    #[test]
    fn exp_map_preserves_unit_norm() {
        // For unit ‖X‖ and any tangent v, exp_X(v) should have ‖·‖ = 1.
        let x = unit(&[0.3, -0.5, 0.7, 0.1, -0.2, 0.4, 0.6, -0.1]);
        let v = tangent_at(&x, &[1.0, 2.0, -1.0, 0.5, -0.7, 1.3, -0.4, 0.9]);
        // Scale v to a few different magnitudes — small enough to stay well-
        // conditioned (cos/sin), large enough to be non-trivial.
        for &scale in &[0.01f32, 0.1, 0.5, 1.0, 2.0] {
            let v_scaled: Vec<f32> = v.iter().map(|vi| vi * scale).collect();
            let mut out = vec![0.0f32; 8];
            sphere_exp_map_into(&x, &v_scaled, &mut out).expect("ok");
            let out_norm = l2_norm(&out);
            assert!(
                (out_norm - 1.0).abs() < 1e-5,
                "exp_map broke unit norm at scale={scale}: {out_norm}"
            );
        }
    }

    #[test]
    fn exp_map_small_velocity_matches_first_order() {
        // For small ‖v‖: exp_X(v) ≈ X + v (first-order Taylor).
        // (cos ε ≈ 1, sin ε ≈ ε, so exp ≈ X + ε·v/ε = X + v.)
        let x = unit(&[0.6, 0.8, 0.0]);
        let v = tangent_at(&x, &[0.0, 0.0, 1.0]); // ⊥ x by construction
        let scale = 1e-4f32;
        let v_scaled: Vec<f32> = v.iter().map(|vi| vi * scale).collect();
        let mut out = vec![0.0f32; 3];
        sphere_exp_map_into(&x, &v_scaled, &mut out).expect("ok");
        // out ≈ X + v_scaled up to O(scale²) curvature.
        for i in 0..3 {
            let expected = x[i] + v_scaled[i];
            assert!(
                (out[i] - expected).abs() < 1e-7,
                "first-order mismatch at {}: got {} expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn exp_map_pi_rotation_flips_x() {
        // v perpendicular to X with ‖v‖ = π: exp_X(v) = cos(π)·X + sin(π)·v/π
        //                                       = −X + 0 = −X.
        // (The classic antipodal-via-tangent-π case.)
        let x = unit(&[1.0, 0.0, 0.0]);
        let v = vec![0.0f32, core::f32::consts::PI, 0.0]; // ⊥ x, ‖v‖ = π
        let mut out = vec![0.0f32; 3];
        sphere_exp_map_into(&x, &v, &mut out).expect("ok");
        for i in 0..3 {
            assert!(
                (out[i] - (-x[i])).abs() < 1e-4,
                "exp_X(π·v_perp) != −X at {}: got {} expected {}",
                i,
                out[i],
                -x[i]
            );
        }
    }

    #[test]
    fn exp_map_shape_mismatch() {
        let x = vec![1.0f32, 0.0, 0.0];
        let v = vec![0.0f32, 1.0, 0.0];
        let mut out = vec![0.0f32; 2];
        let err = sphere_exp_map_into(&x, &v, &mut out).unwrap_err();
        assert_eq!(err, SphereError::ShapeMismatch);
    }

    // ── Integration: small-step Euler–Maruyama preserves unit norm ─

    #[test]
    fn euler_maruyama_step_preserves_unit_norm() {
        // Compose the three primitives for one Euler–Maruyama step at a
        // random vMF drift target. The output should still be on the sphere.
        // This is the core correctness invariant of the PoC sampler.
        let d = 8;
        let x0 = unit(&[0.3, -0.5, 0.7, 0.1, -0.2, 0.4, 0.6, -0.1]); // source
        let mu = unit(&[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]); // vMF mean
        let kappa = 5.0f32;

        // ω_1 = arccos(X_1 · x_0) — between mu (taken as X_1) and x_0.
        let s = dot(&mu, &x0).clamp(-1.0, 1.0);
        let omega_1 = (1.0f32 - s * s).max(0.0).sqrt().atan2(s);
        let omega_1 = omega_1.clamp(COT_FLOOR, core::f32::consts::PI - COT_FLOOR);

        // Ẋ_1 = ω_1·sin(0)·X_1 + cos(0)·Ẋ_init = Ẋ_init, where Ẋ_init is
        // the initial unit-tangent (perp to x_0). Use the projection of mu
        // onto x_0's tangent space, scaled by omega_1 to match the geodesic
        // parametrization.
        let x_init_tan = tangent_at(&x0, &mu);
        let x_init_tan_norm = l2_norm(&x_init_tan).max(1e-9);
        let x1_dot: Vec<f32> = x_init_tan
            .iter()
            .map(|v| v / x_init_tan_norm * omega_1)
            .collect();

        // Jacobian correction at t = 0.5.
        let t = 0.5f32;
        let mut jac_out = vec![0.0f32; d];
        jacobian_logdet_cot_correction(t, omega_1, d, &x1_dot, &mut jac_out).expect("jac ok");

        // vMF score at X_1 (= mu): ∇_M r(X_1) = κ·μ − κ·(μ·X_1)·X_1 = κ·(μ − μ) = 0.
        // (At X_1 = mu, the score vanishes — the gradient is zero at the mode.)
        // Use a different X_1 to get a non-trivial score: take X_1 = midpoint
        // of the geodesic, computed via Slerp.
        let c0 = ((1.0f32 - t) * omega_1).sin() / omega_1.sin();
        let c1 = (t * omega_1).sin() / omega_1.sin();
        let x1: Vec<f32> = (0..d).map(|i| c0 * x0[i] + c1 * mu[i]).collect();
        let x1 = unit(&x1); // renormalize against f32 drift

        // Score at X_1: κ·(μ − (μ·X_1)·X_1).
        let mu_dot_x1 = dot(&mu, &x1);
        let score: Vec<f32> = (0..d)
            .map(|i| kappa * (mu[i] - mu_dot_x1 * x1[i]))
            .collect();

        // Drift = score − Jacobian correction (the vMF closed-form drift).
        // (γ_t absorbed into the step size h below.)
        let drift: Vec<f32> = (0..d).map(|i| score[i] - jac_out[i]).collect();

        // Noise v_perp ~ N(0, I), projected onto x1's tangent space.
        // Deterministic test vector (no RNG) — just an arbitrary tangent.
        let noise_raw = tangent_at(&x1, &[0.7, -0.3, 0.4, 0.2, -0.5, 0.1, 0.6, -0.2]);
        let h = 0.01f32;
        let gamma_t = 1.0f32;
        let v_step: Vec<f32> = (0..d)
            .map(|i| h * drift[i] + (2.0 * gamma_t * h).sqrt() * noise_raw[i])
            .collect();

        // Project the step onto x1's tangent space (Euler–Maruyama on the
        // manifold requires tangent steps; exp_map will then renormalize).
        let v_step_tan = tangent_at(&x1, &v_step);

        let mut out = vec![0.0f32; d];
        sphere_exp_map_into(&x1, &v_step_tan, &mut out).expect("exp ok");
        let out_norm = l2_norm(&out);
        assert!(
            (out_norm - 1.0).abs() < 1e-5,
            "Euler–Maruyama step broke unit norm: {out_norm}"
        );
    }
}
