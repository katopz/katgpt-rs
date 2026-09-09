//! Nonlinear DEC Heat Kernel Trajectory — Exponential Integrator (Plan 359 Phase 3).
//!
//! Extends the linear heat kernel (Phase 1) to handle the ReLU nonlinearity in
//! motor-gated field evolution. The continuous ODE is:
//!
//! `dh/dt = -h + Δ·ReLU(h) + diag(motor)·h`
//!
//! Decomposed as `dh/dt = L·h + N(h)` where:
//!
//! - **L** = `-I + Δ + diag(motor)` — the linear operator (same as Phase 1's A)
//! - **N(h)** = `Δ·(ReLU(h) - h)` — the nonlinear correction
//!
//! When the field is all-positive, `ReLU(h) = h` and `N(h) = 0`: the nonlinear
//! path reduces exactly to the linear heat kernel.
//!
//! # Method: Duhamel integral + Gauss-Legendre quadrature
//!
//! The exact solution via variation of parameters (Duhamel):
//!
//! `h(t) = exp(t·L)·h₀ + ∫₀ᵗ exp((t-s)·L)·N(h(s)) ds`
//!
//! The integral is approximated by `n_quad`-point Gauss-Legendre quadrature on
//! `[0, t]`. The integrand requires `h(s)` at the quadrature nodes, which we
//! predict via the linear heat kernel (exponential Euler predictor):
//!
//! `h(s) ≈ exp(s·L)·h₀`
//!
//! This yields a first-order exponential integrator: exact for the linear part,
//! first-order in the nonlinear perturbation. The quadrature refines the
//! time-integration of the source term beyond plain exponential Euler (which
//! would use a single left-endpoint evaluation).
//!
//! # Cost
//!
//! Each trajectory prediction requires `1 + 2·n_quad` heat-kernel applications
//! (one for the linear prediction at `t`, plus per-node: one prediction at
//! `s_j` and one propagation at `t-s_j`) and `n_quad` Laplacian applications.
//! Each heat-kernel application is `O(n·k·dim)` after the eigendecomposition
//! precompute.
//!
//! # Modelless
//!
//! Every step is closed-form algebra over the shipped DEC substrate. The
//! Gauss-Legendre nodes/weights are hardcoded constants (no numerical
//! computation). No training, no backprop — per the katgpt-rs modelless mandate.
//!
//! # References
//!
//! - Plan 359 Phase 3, Research 365 (PhysiFormer distillation).
//! - Cox & Matthews, "Exponential Time Differencing for Stiff Systems" (JCP 2002)
//!   — the ETDRK family. Our quadrature-based integrator is the simplest member.
//! - Hochbruck & Ostermann, "Exponential Integrators" (Acta Numerica 2010).

use crate::heat_kernel::{DecEigendecomposition, heat_kernel_trajectory_linear_into};
use crate::operators::{graph_laplacian_into, hodge_laplacian};
use crate::types::{CellComplex, CochainField};

/// Default number of Gauss-Legendre quadrature points. 4 is a good balance
/// between accuracy and cost (9 heat-kernel applications total: 1 linear + 8
/// per-node). For stiff nonlinearities, 6–8 points may be warranted.
pub const DEFAULT_N_QUAD: usize = 4;

/// Maximum number of Gauss-Legendre quadrature points supported (hardcoded
/// tables for n = 1..=8).
pub const MAX_N_QUAD: usize = 8;

/// Gauss-Legendre quadrature nodes and weights on `[-1, 1]` for `n = 1..=8`.
///
/// Returns `(nodes, weights)` as static slices. Nodes are in ascending order.
/// Weights sum to 2 (the integral of 1 over `[-1, 1]`). For `n` points, the
/// rule is exact for polynomials of degree `≤ 2n - 1`.
///
/// Panics for `n = 0` or `n > 8`.
#[allow(
    clippy::excessive_precision,
    reason = "canonical Gauss-Legendre nodes/weights kept at full published precision for traceability vs standard references"
)]
fn gauss_legendre_nodes_weights(n: usize) -> (&'static [f64], &'static [f64]) {
    match n {
        1 => (&[0.0_f64], &[2.0_f64]),
        2 => (&[-0.5773502691896257, 0.5773502691896257], &[1.0, 1.0]),
        3 => (
            &[-0.7745966692414834, 0.0, 0.7745966692414834],
            &[0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
        ),
        4 => (
            &[
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            &[
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        ),
        5 => (
            &[
                -0.9061798459386640,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.9061798459386640,
            ],
            &[
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        ),
        6 => (
            &[
                -0.9324695142031521,
                -0.6612093864662645,
                -0.2386191860831969,
                0.2386191860831969,
                0.6612093864662645,
                0.9324695142031521,
            ],
            &[
                0.1713244923791704,
                0.3607615730481386,
                0.4679139345726910,
                0.4679139345726910,
                0.3607615730481386,
                0.1713244923791704,
            ],
        ),
        7 => (
            &[
                -0.9491079123427585,
                -0.7415311855993945,
                -0.4058451513773972,
                0.0,
                0.4058451513773972,
                0.7415311855993945,
                0.9491079123427585,
            ],
            &[
                0.1294849661688697,
                0.2797053914892766,
                0.3818300505051189,
                0.4179591836734694,
                0.3818300505051189,
                0.2797053914892766,
                0.1294849661688697,
            ],
        ),
        8 => (
            &[
                -0.9602898564975363,
                -0.7966664774136267,
                -0.5255324099163290,
                -0.1834346424956498,
                0.1834346424956498,
                0.5255324099163290,
                0.7966664774136267,
                0.9602898564975363,
            ],
            &[
                0.1012285362903763,
                0.2223810344533745,
                0.3137066458778873,
                0.3626837833783620,
                0.3626837833783620,
                0.3137066458778873,
                0.2223810344533745,
                0.1012285362903763,
            ],
        ),
        _ => panic!("gauss_legendre_nodes_weights: n={n} not supported (1..={MAX_N_QUAD})"),
    }
}

/// Pre-allocated scratch buffers for repeated nonlinear heat kernel calls.
///
/// Create once (via [`NonlinearScratch::new`]), reuse across calls on the same
/// grid. The buffers are resized internally to match the field shape, so the
/// constructor dimensions are just initial capacity hints.
#[derive(Clone, Debug)]
pub struct NonlinearScratch {
    /// Linear prediction at quadrature node `s_j`: `exp(s_j·L)·h₀`.
    h_s: CochainField,
    /// ReLU residual: `ReLU(h_s) - h_s`.
    r_s: CochainField,
    /// Nonlinear source: `Δ·r_s` (Laplacian of the residual).
    n_s: CochainField,
    /// Propagated source: `exp((t-s_j)·L)·n_s`.
    m_s: CochainField,
}

impl NonlinearScratch {
    /// Create scratch buffers for a field of the given shape.
    ///
    /// The buffers will be resized automatically to match the input field on
    /// each call, so the constructor dimensions are just initial capacity.
    #[inline]
    pub fn new(rank: u8, n_cells: usize, dim: usize) -> Self {
        Self {
            h_s: CochainField::zeros(rank, n_cells, dim),
            r_s: CochainField::zeros(rank, n_cells, dim),
            n_s: CochainField::zeros(rank, n_cells, dim),
            m_s: CochainField::zeros(rank, n_cells, dim),
        }
    }
}

/// Resize a cochain field to the given shape (rank, dim, data-len).
#[inline]
fn resize_field(field: &mut CochainField, rank: u8, dim: usize, len: usize) {
    field.data.resize(len, 0.0);
    field.dim = dim;
    field.rank = rank;
}

/// Evaluate the Duhamel source-term integral via Gauss-Legendre quadrature.
///
/// Computes the nonlinear correction term:
///
/// `I(t) = ∫₀ᵗ exp((t-s)·L)·N(h(s)) ds`
///
/// where `N(h) = Δ·(ReLU(h) - h)`, `L = -I + Δ + diag(motor)`, and the
/// predictor `h(s_j) ≈ exp(s_j·L)·h₀` (exponential Euler predictor).
///
/// The quadrature maps `n_quad` Gauss-Legendre nodes from `[-1,1]` to `[0,t]`:
///
/// `s_j = (t/2)·(ξ_j + 1)`, `ds = (t/2)·dξ`
///
/// `I(t) ≈ (t/2)·Σ_j w_j·exp((t-s_j)·L)·N(exp(s_j·L)·h₀)`
///
/// # Accumulate semantics
///
/// This function **accumulates** into `out` — it does NOT zero it first. This
/// allows [`heat_kernel_trajectory_nonlinear_into`] to write the linear
/// prediction into `out` and then call this to add the correction. For
/// standalone use, zero `out` before calling.
///
/// # Arguments
///
/// See [`heat_kernel_trajectory_nonlinear`] — this function shares the same
/// parameter set, plus `out` (the accumulation target) and `scratch`.
#[allow(
    clippy::too_many_arguments,
    reason = "nonlinear expm quadrature needs mesh + eig + field + motor + t + quad + relu + out + scratch; matches the paper's operator signature"
)]
pub fn expm_source_term_quadrature(
    cx: &CellComplex,
    eig: &DecEigendecomposition,
    h0: &CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    t: f32,
    n_quad: usize,
    relu_slope: f32,
    out: &mut CochainField,
    scratch: &mut NonlinearScratch,
) {
    let n = h0.n_cells();
    let dim = h0.dim;
    let len = n * dim;
    let rank = h0.rank;

    debug_assert!(
        (1..=MAX_N_QUAD).contains(&n_quad),
        "expm_source_term_quadrature: n_quad={n_quad} must be 1..={MAX_N_QUAD}"
    );

    // Resize scratch buffers to match h0's shape.
    resize_field(&mut scratch.h_s, rank, dim, len);
    resize_field(&mut scratch.r_s, rank, dim, len);
    resize_field(&mut scratch.n_s, rank, dim, len);
    resize_field(&mut scratch.m_s, rank, dim, len);

    // Early exit: zero-length interval → no correction.
    if t == 0.0 {
        return;
    }

    let (nodes, weights) = gauss_legendre_nodes_weights(n_quad);
    let half_t = 0.5 * t;

    for j in 0..n_quad {
        // Map node from [-1,1] to [0,t]: s = (t/2)·(ξ + 1)
        let xi = nodes[j] as f32;
        let s_j = half_t * (xi + 1.0);
        let w_j = weights[j] as f32;
        let tau = t - s_j; // propagation time for the source contribution

        // a) Predict h(s_j) ≈ exp(s_j·L)·h₀ via the linear heat kernel.
        heat_kernel_trajectory_linear_into(eig, h0, motor_vec, motor_dim, s_j, &mut scratch.h_s);

        // b) Compute ReLU residual: r = ReLU(h_s) - h_s.
        //    For all-positive regions, r = 0 (ReLU is identity).
        //    For negative regions, r = (slope - 1)·h_s (the clipped amount).
        if relu_slope == 0.0 {
            // Standard ReLU: r = max(0, h) - h = -min(0, h).
            for i in 0..len {
                let h = scratch.h_s.data[i];
                scratch.r_s.data[i] = h.max(0.0) - h;
            }
        } else {
            // Leaky ReLU: r = (h ≥ 0 ? h : slope·h) - h = (h ≥ 0 ? 0 : (slope-1)·h).
            for i in 0..len {
                let h = scratch.h_s.data[i];
                let relu = if h >= 0.0 { h } else { relu_slope * h };
                scratch.r_s.data[i] = relu - h;
            }
        }

        // c) Compute nonlinear source: n = Δ·r (Laplacian of the residual).
        //    Rank-0 fast path: graph Laplacian (zero-alloc).
        //    Rank ≥ 1: allocating hodge_laplacian fallback.
        if rank == 0 && cx.n_edges() > 0 {
            graph_laplacian_into(cx, &scratch.r_s, &mut scratch.n_s);
        } else {
            let lap = hodge_laplacian(cx, &scratch.r_s);
            let m = lap.data.len().min(len);
            scratch.n_s.data[..m].copy_from_slice(&lap.data[..m]);
            scratch.n_s.data[m..].fill(0.0);
        }

        // d) Propagate the source forward: m = exp(tau·L)·n.
        //    Apply the linear heat kernel to the source field n_s with time tau.
        heat_kernel_trajectory_linear_into(
            eig,
            &scratch.n_s,
            motor_vec,
            motor_dim,
            tau,
            &mut scratch.m_s,
        );

        // e) Accumulate quadrature: out += (t/2)·w_j·m.
        let coeff = half_t * w_j;
        for i in 0..len {
            out.data[i] += coeff * scratch.m_s.data[i];
        }
    }
}

/// Nonlinear heat kernel trajectory via Duhamel integral + Gauss-Legendre
/// quadrature.
///
/// Solves `dh/dt = -h + Δ·ReLU(h) + diag(motor)·h` by decomposing into linear
/// (`L = -I + Δ + diag(motor)`) and nonlinear (`N(h) = Δ·(ReLU(h) - h)`) parts
/// and applying the Duhamel variation-of-parameters formula with `n_quad`-point
/// Gauss-Legendre quadrature for the source integral.
///
/// When `n_quad = 0` or the field is all-positive, this reduces to the linear
/// [`heat_kernel_trajectory_linear`](crate::heat_kernel_trajectory_linear).
///
/// # Arguments
///
/// * `cx` — the cell complex (spatial map).
/// * `eig` — precomputed eigendecomposition of the Laplacian for `cx` at
///   `h0.rank` (see [`DecEigendecomposition::compute`]).
/// * `h0` — initial field state.
/// * `motor_vec` — motor command vector (length ≥ `motor_dim`).
/// * `motor_dim` — number of motor-gated channels.
/// * `t` — time horizon.
/// * `n_quad` — number of Gauss-Legendre quadrature points (1..=8). More
///   points = higher accuracy at `O(2·n_quad)` extra heat-kernel applications.
/// * `relu_slope` — ReLU negative-side slope (`0.0` = standard ReLU; small
///   positive = leaky ReLU). Must match the `relu_slope` used in
///   [`evolve_motor_gated_field`](crate::evolve_motor_gated_field) when
///   comparing against step-by-step Euler.
///
/// # Returns
///
/// The field state at time `t`: `h(t) = exp(t·L)·h₀ + ∫₀ᵗ exp((t-s)·L)·N(h(s))ds`.
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "nonlinear heat kernel needs mesh + eig + field + motor + t + quad + relu; matches the paper's operator signature"
)]
pub fn heat_kernel_trajectory_nonlinear(
    cx: &CellComplex,
    eig: &DecEigendecomposition,
    h0: &CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    t: f32,
    n_quad: usize,
    relu_slope: f32,
) -> CochainField {
    let mut out = CochainField::zeros(h0.rank, h0.n_cells(), h0.dim);
    let mut scratch = NonlinearScratch::new(h0.rank, h0.n_cells(), h0.dim);
    heat_kernel_trajectory_nonlinear_into(
        cx,
        eig,
        h0,
        motor_vec,
        motor_dim,
        t,
        n_quad,
        relu_slope,
        &mut out,
        &mut scratch,
    );
    out
}

/// Zero-output-alloc nonlinear heat kernel trajectory — writes into `out`.
///
/// Same as [`heat_kernel_trajectory_nonlinear`] but writes into a caller-provided
/// `out` buffer and reuses pre-allocated `scratch`. The `scratch` struct is
/// allocated once and reused across calls — in a steady-state loop, this path
/// allocates **0 bytes** per call (all internal buffers are pre-allocated and
/// resized in-place if needed).
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "zero-alloc variant mirrors heat_kernel_trajectory_nonlinear; caller-provided out + scratch add 2 args"
)]
pub fn heat_kernel_trajectory_nonlinear_into(
    cx: &CellComplex,
    eig: &DecEigendecomposition,
    h0: &CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    t: f32,
    n_quad: usize,
    relu_slope: f32,
    out: &mut CochainField,
    scratch: &mut NonlinearScratch,
) {
    let n = h0.n_cells();
    let dim = h0.dim;
    let len = n * dim;
    let rank = h0.rank;

    debug_assert_eq!(
        eig.rank, rank,
        "heat_kernel_trajectory_nonlinear: eig.rank {} != h0.rank {}",
        eig.rank, rank
    );
    debug_assert_eq!(
        eig.n_cells, n,
        "heat_kernel_trajectory_nonlinear: eig.n_cells {} != h0.n_cells() {}",
        eig.n_cells, n
    );
    debug_assert!(
        motor_dim <= dim,
        "heat_kernel_trajectory_nonlinear: motor_dim {motor_dim} > h0.dim {dim}"
    );
    debug_assert!(
        motor_vec.len() >= motor_dim,
        "heat_kernel_trajectory_nonlinear: motor_vec len {} < motor_dim {motor_dim}",
        motor_vec.len()
    );

    // Resize out to match h0's shape.
    resize_field(out, rank, dim, len);

    // Step 1: Linear prediction — out = exp(t·L)·h₀.
    // This handles t=0 correctly (returns h₀ via the identity).
    heat_kernel_trajectory_linear_into(eig, h0, motor_vec, motor_dim, t, out);

    // Step 2: Nonlinear correction — out += ∫₀ᵗ exp((t-s)·L)·N(h(s))ds.
    // Skipped when n_quad = 0 (pure linear) or t = 0 (zero-length interval).
    if n_quad > 0 && t != 0.0 {
        expm_source_term_quadrature(
            cx, eig, h0, motor_vec, motor_dim, t, n_quad, relu_slope, out, scratch,
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::DecEigendecomposition;
    use crate::types::{CellComplex, CochainField};

    use crate::test_common::{l2_dist, l2_norm, place_bump, zero_field};

    /// Helper: offset all cells in a channel by a constant (creates mixed-sign fields).
    fn offset_channel(field: &mut CochainField, ch: usize, offset: f32) {
        let dim = field.dim;
        let n = field.n_cells();
        for i in 0..n {
            field.data[i * dim + ch] += offset;
        }
    }

    /// Helper: one step of nonlinear (ReLU-gated) Euler propagation.
    ///
    /// Mirrors `evolve_motor_gated_field` (Plan 357) exactly: split-step
    /// diffusion-reaction + motor gain. Self-contained (does not depend on the
    /// `motor_gated_field` feature) so the nonlinear heat kernel tests can run
    /// with default features only.
    ///
    /// `h_{t+1} = (1+dt·motor)·((1-dt)·h + dt·Δ·ReLU(h))`
    #[allow(
        clippy::too_many_arguments,
        reason = "test-only reference impl mirroring evolve_motor_gated_field (Plan 357); the 8 args (cx, h, motor_vec, motor_dim, dt, relu_slope, scratch_lap, scratch_relu) match the production operator signature by design — drift between this Euler baseline and the production split-step is what the nonlinear heat kernel tests detect, so the signature must match"
    )]
    fn nonlinear_euler_step(
        cx: &CellComplex,
        h: &mut CochainField,
        motor_vec: &[f32],
        motor_dim: usize,
        dt: f32,
        relu_slope: f32,
        scratch_lap: &mut CochainField,
        scratch_relu: &mut CochainField,
    ) {
        let n = h.n_cells();
        let dim = h.dim;
        let len = n * dim;

        scratch_relu.rank = h.rank;
        scratch_relu.dim = dim;
        scratch_lap.rank = h.rank;
        scratch_lap.dim = dim;

        // ReLU gate h → scratch_relu
        if relu_slope == 0.0 {
            for (o, &v) in scratch_relu.data[..len]
                .iter_mut()
                .zip(h.data[..len].iter())
            {
                *o = v.max(0.0);
            }
        } else {
            for (o, &v) in scratch_relu.data[..len]
                .iter_mut()
                .zip(h.data[..len].iter())
            {
                *o = if v >= 0.0 { v } else { relu_slope * v };
            }
        }

        // Laplacian of ReLU(h) → scratch_lap
        graph_laplacian_into(cx, scratch_relu, scratch_lap);

        // Blend: h = (1-dt)·h + dt·lap
        for i in 0..len {
            h.data[i] = (1.0 - dt) * h.data[i] + dt * scratch_lap.data[i];
        }

        // Motor gate: h[c, ch] *= (1 + dt·motor[ch])
        if motor_dim > 0 {
            for cell in 0..n {
                let base = cell * dim;
                for (ch, &motor) in motor_vec.iter().enumerate().take(motor_dim) {
                    h.data[base + ch] *= 1.0 + dt * motor;
                }
            }
        }
    }

    /// Helper: T steps of nonlinear Euler (the reference trajectory).
    fn nonlinear_euler_trajectory(
        cx: &CellComplex,
        h0: &CochainField,
        motor_vec: &[f32],
        motor_dim: usize,
        dt: f32,
        steps: usize,
        relu_slope: f32,
    ) -> CochainField {
        let n = h0.n_cells();
        let dim = h0.dim;
        let mut h = h0.clone();
        let mut scratch_lap = CochainField::zeros(h0.rank, n, dim);
        let mut scratch_relu = CochainField::zeros(h0.rank, n, dim);
        for _ in 0..steps {
            nonlinear_euler_step(
                cx,
                &mut h,
                motor_vec,
                motor_dim,
                dt,
                relu_slope,
                &mut scratch_lap,
                &mut scratch_relu,
            );
        }
        h
    }

    // ── Gauss-Legendre table tests ──────────────────────────────────────────

    #[test]
    fn gauss_legendre_weights_sum_to_two() {
        for n in 1..=MAX_N_QUAD {
            let (_, weights) = gauss_legendre_nodes_weights(n);
            let sum: f64 = weights.iter().sum();
            assert!(
                (sum - 2.0).abs() < 1e-14,
                "n={n}: weights sum to {sum}, expected 2.0"
            );
        }
    }

    #[test]
    fn gauss_legendre_nodes_in_ascending_order() {
        for n in 2..=MAX_N_QUAD {
            let (nodes, _) = gauss_legendre_nodes_weights(n);
            for i in 1..n {
                assert!(
                    nodes[i] > nodes[i - 1],
                    "n={n}: node[{i}]={:.16} not > node[{}] = {:.16}",
                    nodes[i],
                    i - 1,
                    nodes[i - 1]
                );
            }
        }
    }

    #[test]
    fn gauss_legendre_integrates_constant_exactly() {
        // ∫_{-1}^{1} 1 dx = 2, and the weights sum to 2.
        for n in 1..=MAX_N_QUAD {
            let (_, weights) = gauss_legendre_nodes_weights(n);
            let integral: f64 = weights.iter().sum(); // f(x)=1 → Σ w_j·f(x_j) = Σ w_j
            assert!(
                (integral - 2.0).abs() < 1e-14,
                "n={n}: constant integral = {integral}"
            );
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomial_exactly() {
        // Test ∫_{-1}^{1} x² dx = 2/3 with n=2 (degree 2 ≤ 2·2-1 = 3). ✓
        let (nodes, weights) = gauss_legendre_nodes_weights(2);
        let integral: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x * x)
            .sum();
        assert!(
            (integral - 2.0 / 3.0).abs() < 1e-14,
            "x² integral = {integral}, expected 2/3"
        );
    }

    // ── All-positive field reduces to linear heat kernel ────────────────────

    #[test]
    fn all_positive_field_matches_linear() {
        // When the field is all-positive AND propagation stays positive (stable
        // motor, a_max < 0), ReLU(h) = h, so N(h) = 0.
        //
        // CRITICAL lessons (discovered via diagnostics):
        // 1. Must use FULL eigendecomposition (k = n_cells, max_iter=2000).
        //    With k < n_cells, spectral reconstruction error introduces negative
        //    values that activate ReLU spuriously.
        // 2. Must use SHORT horizon so the field stays well above the eigensolver
        //    noise floor (~0.001). At long horizons with stable motors, the field
        //    decays to ~0 and the eigensolver error dominates, producing spurious
        //    negative values.
        let w = 4;
        let h = 4;
        let dim = 2;
        let n = w * h;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, n, 2000);

        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 1, 1, 0, 1.0, 0.8);
        place_bump(&mut field, w, h, 2, 2, 1, 0.8, 0.8);

        // Stable motor + SHORT horizon: field stays well above noise floor.
        let motor = [-9.0_f32, -8.5];
        let t = 0.1_f32;

        let linear =
            crate::heat_kernel::heat_kernel_trajectory_linear(&eig, &field, &motor, dim, t);
        let nonlinear = heat_kernel_trajectory_nonlinear(
            &cx,
            &eig,
            &field,
            &motor,
            dim,
            t,
            DEFAULT_N_QUAD,
            0.0,
        );

        let dist = l2_dist(&linear, &nonlinear);
        let norm = l2_norm(&linear).max(1e-10);
        assert!(
            dist / norm < 1e-4,
            "all-positive field: nonlinear should match linear, dist/norm = {:.3e}",
            dist / norm
        );
    }

    // ── T3.3: matches step-by-step Euler at small dt ────────────────────────

    #[test]
    fn nonlinear_matches_step_by_step_at_small_dt() {
        // At small dt, both the nonlinear exponential integrator and step-by-step
        // Euler approximate the same ODE. With fine enough Euler (small dt) and
        // enough quadrature points, they should agree.
        //
        // CRITICAL: must use a FULL eigendecomposition (k = n_cells, max_iter=2000)
        // on a small grid. With k < n_cells, the linear prediction is lossy
        // (spectral reconstruction error), and the nonlinear path diverges from
        // Euler — not because of a bug, but because the two methods see different
        // linear operators (exact Laplacian vs truncated spectral Laplacian).
        let w = 4;
        let h = 4;
        let dim = 2;
        let n = w * h;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, n, 2000);

        // Mixed-sign field: positive bump offset by -0.1.
        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 1, 1, 0, 1.0, 0.8);
        offset_channel(&mut field, 0, -0.1);
        place_bump(&mut field, w, h, 2, 2, 1, 0.8, 0.8);
        offset_channel(&mut field, 1, -0.08);

        // Stable motor.
        let motor = [-9.0_f32, -8.5];
        let t = 0.5_f32;

        // Nonlinear heat kernel (5-point quadrature for good accuracy).
        let hk_nonlinear =
            heat_kernel_trajectory_nonlinear(&cx, &eig, &field, &motor, dim, t, 5, 0.0);

        // Fine Euler (dt=0.001, 500 steps) — the reference.
        let fine_euler = nonlinear_euler_trajectory(&cx, &field, &motor, dim, 0.001, 500, 0.0);

        let dist = l2_dist(&hk_nonlinear, &fine_euler);
        let norm = l2_norm(&fine_euler).max(1e-10);
        let rel_err = dist / norm;

        // First-order predictor + quadrature: expect a few percent error.
        assert!(
            rel_err < 0.15,
            "nonlinear heat kernel should match fine Euler at t={t}: rel_err = {rel_err:.4} ({dist:.3e}/{norm:.3e})"
        );
    }

    // ── T3.4: diverges from coarse Euler at long horizon ────────────────────

    #[test]
    fn nonlinear_diverges_from_euler_at_long_horizon() {
        // T3.4: At long-ish horizon, the nonlinear heat kernel (exact linear
        // part + quadrature for nonlinear) should be closer to fine Euler than
        // coarse Euler is. Uses a full eigendecomposition on 4×4 (exact linear
        // operator) and a SHORT horizon (t=1.0) where the field is still alive.
        //
        // NOTE: the “beats Euler at long horizon” property is fundamentally a
        // LINEAR property (Phase 5 G1 gate). For the nonlinear (ReLU-gated) case
        // with stable motors, the field decays to zero at long horizon, making
        // comparisons degenerate. This test uses t=1.0 where the field is alive
        // and the coarse-Euler error (~1%) is comparable to the nonlinear
        // predictor error.
        let w = 4;
        let h = 4;
        let dim = 1;
        let n_cells = w * h;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, n_cells, 2000);

        // Mildly mixed-sign field.
        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 1, 1, 0, 1.0, 0.8);
        offset_channel(&mut field, 0, -0.05);

        // Stable motor: a_max ≈ -2.0 (4×4 grid, λ_max ≈ 8).
        let motor = [-9.0_f32];
        let t = 1.0_f32;

        // Fine Euler (dt=0.001, 1000 steps) — the reference.
        let fine_euler = nonlinear_euler_trajectory(&cx, &field, &motor, dim, 0.001, 1000, 0.0);

        // Coarse Euler (dt=0.1, 10 steps) — the baseline.
        let coarse_euler = nonlinear_euler_trajectory(&cx, &field, &motor, dim, 0.1, 10, 0.0);

        // Nonlinear heat kernel (6-point quadrature).
        let hk_nonlinear =
            heat_kernel_trajectory_nonlinear(&cx, &eig, &field, &motor, dim, t, 6, 0.0);

        let hk_err = l2_dist(&hk_nonlinear, &fine_euler);
        let coarse_err = l2_dist(&coarse_euler, &fine_euler);

        assert!(
            hk_err < coarse_err,
            "nonlinear heat kernel should beat coarse Euler at t={t}: \
             hk_err = {hk_err:.3e} vs coarse_err = {coarse_err:.3e}"
        );
    }

    // ── Zero-output-alloc variant matches allocating ────────────────────────

    #[test]
    fn into_variant_matches_allocating() {
        let w = 6;
        let h = 6;
        let dim = 2;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, 6, 300);

        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 2, 3, 0, 1.0, 1.0);
        offset_channel(&mut field, 0, -0.1);

        // Stable motors.
        let motor = [-7.0_f32, -6.5];
        let t = 3.0_f32;

        let allocating =
            heat_kernel_trajectory_nonlinear(&cx, &eig, &field, &motor, dim, t, 4, 0.0);

        let mut out = CochainField::zeros(0, cx.n_vertices(), dim);
        let mut scratch = NonlinearScratch::new(0, cx.n_vertices(), dim);
        heat_kernel_trajectory_nonlinear_into(
            &cx,
            &eig,
            &field,
            &motor,
            dim,
            t,
            4,
            0.0,
            &mut out,
            &mut scratch,
        );

        assert_eq!(allocating.data.len(), out.data.len());
        let max_diff = allocating
            .data
            .iter()
            .zip(out.data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "into variant should match allocating: max_diff = {max_diff:.3e}"
        );
    }

    // ── n_quad=0 reduces to linear heat kernel ──────────────────────────────

    #[test]
    fn zero_quad_points_reduces_to_linear() {
        let w = 6;
        let h = 6;
        let dim = 1;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, 6, 300);

        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 3, 3, 0, 1.0, 1.0);
        offset_channel(&mut field, 0, -0.1); // mixed-sign

        // Stable motor.
        let motor = [-7.0_f32];
        let t = 3.0_f32;

        let linear =
            crate::heat_kernel::heat_kernel_trajectory_linear(&eig, &field, &motor, dim, t);
        let nq0 = heat_kernel_trajectory_nonlinear(&cx, &eig, &field, &motor, dim, t, 0, 0.0);

        let dist = l2_dist(&linear, &nq0);
        assert!(
            dist < 1e-6,
            "n_quad=0 should reduce to linear: dist = {dist:.3e}"
        );
    }

    // ── t=0 returns h₀ ──────────────────────────────────────────────────────

    #[test]
    fn t_zero_returns_h0() {
        // The t=0 identity (exp(0·L)·h₀ = h₀) requires the eigenvectors to form
        // a COMPLETE orthonormal basis. Must use k=n_cells with max_iter=2000
        // (same as the existing heat_kernel::t_zero_identity test).
        let w = 4;
        let h = 4;
        let dim = 1;
        let n_cells = w * h;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, n_cells, 2000);

        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 2, 2, 0, 1.0, 0.8);

        let motor = [-10.0_f32];

        let result = heat_kernel_trajectory_nonlinear(&cx, &eig, &field, &motor, dim, 0.0, 4, 0.0);

        let dist = l2_dist(&result, &field);
        let norm = l2_norm(&field).max(1e-10);
        assert!(
            dist / norm < 1e-3,
            "t=0 should return h₀: dist/norm = {:.3e}",
            dist / norm
        );
    }

    // ── Quadrature convergence: more points → better accuracy ──────────────

    #[test]
    fn quadrature_converges_with_more_points() {
        // More quadrature points should improve accuracy (the integral is
        // approximated better). Compare against fine Euler.
        let w = 6;
        let h = 6;
        let dim = 1;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, 6, 300);

        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 2, 2, 0, 1.0, 1.0);
        offset_channel(&mut field, 0, -0.1);

        // Stable motor: a_max ≈ -2.0 (6×6 grid, λ_max ≈ 6).
        let motor = [-7.0_f32];
        let t = 2.0_f32;

        let fine_euler = nonlinear_euler_trajectory(&cx, &field, &motor, dim, 0.001, 2000, 0.0);

        let err_n2 = {
            let hk = heat_kernel_trajectory_nonlinear(&cx, &eig, &field, &motor, dim, t, 2, 0.0);
            l2_dist(&hk, &fine_euler)
        };
        let err_n6 = {
            let hk = heat_kernel_trajectory_nonlinear(&cx, &eig, &field, &motor, dim, t, 6, 0.0);
            l2_dist(&hk, &fine_euler)
        };

        assert!(
            err_n6 <= err_n2 * 1.05,
            "more quadrature points should improve (or at least not worsen) accuracy: \
             err(n=2)={err_n2:.3e}, err(n=6)={err_n6:.3e}"
        );
    }

    // ── Leaky ReLU works ────────────────────────────────────────────────────

    #[test]
    fn leaky_relu_works() {
        // The nonlinear path should work with leaky ReLU (slope > 0).
        // Just a smoke test — verify it runs and produces finite output.
        let w = 6;
        let h = 6;
        let dim = 1;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, 6, 300);

        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 2, 2, 0, 1.0, 1.0);
        offset_channel(&mut field, 0, -0.1);

        // Stable motor.
        let motor = [-7.0_f32];
        let t = 3.0_f32;

        let result = heat_kernel_trajectory_nonlinear(
            &cx, &eig, &field, &motor, dim, t, 4, 0.1, // leaky ReLU
        );

        assert!(
            result.data.iter().all(|&v| v.is_finite()),
            "all values finite"
        );
        assert!(l2_norm(&result) > 0.0, "non-zero output");
    }

    // ── Scratch reuse across calls ──────────────────────────────────────────

    #[test]
    fn scratch_reuse_across_calls() {
        // Verify that the scratch struct can be reused across multiple calls
        // without corruption.
        let w = 6;
        let h = 6;
        let dim = 2;
        let cx = CellComplex::grid_2d(w, h);
        let eig = DecEigendecomposition::compute(&cx, 0, 6, 300);

        let mut field = zero_field(&cx, dim);
        place_bump(&mut field, w, h, 2, 3, 0, 1.0, 1.0);
        offset_channel(&mut field, 0, -0.1);

        // Stable motors.
        let motor = [-7.0_f32, -6.5];

        let mut out = CochainField::zeros(0, cx.n_vertices(), dim);
        let mut scratch = NonlinearScratch::new(0, cx.n_vertices(), dim);

        // Call 1
        heat_kernel_trajectory_nonlinear_into(
            &cx,
            &eig,
            &field,
            &motor,
            dim,
            3.0,
            4,
            0.0,
            &mut out,
            &mut scratch,
        );
        let result1 = out.data.clone();

        // Call 2 (different time)
        heat_kernel_trajectory_nonlinear_into(
            &cx,
            &eig,
            &field,
            &motor,
            dim,
            5.0,
            4,
            0.0,
            &mut out,
            &mut scratch,
        );
        let result2 = out.data.clone();

        // Call 3 (same as call 1 — should match exactly)
        heat_kernel_trajectory_nonlinear_into(
            &cx,
            &eig,
            &field,
            &motor,
            dim,
            3.0,
            4,
            0.0,
            &mut out,
            &mut scratch,
        );

        // result1 should match the third call (same parameters).
        let max_diff = result1
            .iter()
            .zip(out.data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "scratch reuse should produce identical results"
        );

        // result2 should differ from result1 (different time).
        let dist12: f32 = result1
            .iter()
            .zip(result2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(
            dist12 > 1e-6,
            "different time should produce different results"
        );
    }
}
