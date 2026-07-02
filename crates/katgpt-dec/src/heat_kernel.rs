//! DEC Heat Kernel Trajectory — Single-Shot Field Prediction (Plan 359).
//!
//! Distilled from PhysiFormer (arXiv:2606.27364, Chen/Lan/Vedaldi, VGG Oxford)
//! — the paper's load-bearing insight is the **prediction strategy**, not its
//! trained diffusion: single-shot joint trajectory prediction avoids the
//! compounding error of step-by-step autoregressive rollout. PhysiFormer
//! shows 100× rigidity improvement at 49 frames via single-shot diffusion
//! on 3D mesh physics. The modelless analog for our cochain-field substrate
//! is the **operator exponential** (heat kernel).
//!
//! # The primitive (linear path — Phase 1)
//!
//! Given an initial [`CochainField`] `h₀` and the motor-gated propagation
//! operator `A = -I + Δ + diag(motor)` (the continuous limit of
//! [`evolve_motor_gated_field`] with the ReLU gate removed), the field state
//! at horizon `t` is:
//!
//! `h(t) = exp(t·A)·h₀`
//!
//! computed exactly via the precomputed eigendecomposition of the Hodge
//! Laplacian. This is the **exact** trajectory for linear propagation — zero
//! error accumulation, exact Hodge-decomposition preservation. The step-by-step
//! Euler `(I + dt·A)^T·h₀` is a first-order approximation with `O(T·dt²)`
//! global error.
//!
//! # Block-diagonal structure
//!
//! A key simplification: for a field with `dim` channels, the operator `A`
//! is **block-diagonal** across channels. The Laplacian `Δ` acts independently
//! and identically on each channel (same `n×n` matrix `L` per channel block),
//! and the motor gate is a per-channel scalar `motor[d]`. So the system
//! decouples into `dim` independent `n×n` subsystems, all sharing the same
//! Laplacian eigenvectors:
//!
//! `h_d(t) = Σ_k exp(t · (-1 + λ_k + motor[d])) · (v_kᵀ · h_d(0)) · v_k`
//!
//! where `(λ_k, v_k)` are the eigenpairs of `L` (the graph Laplacian for
//! rank-0, or the Hodge Laplacian Δₖ for rank k). This means we precompute
//! ONE eigendecomposition (the Laplacian's) and reuse it across all channels
//! — the per-channel cost is just `O(n·k)` for the projection + reconstruction,
//! not `O(n²·k)`.
//!
//! # Zero-alloc
//!
//! After the eigendecomposition precompute ([`DecEigendecomposition::compute`]),
//! [`heat_kernel_trajectory_linear_into`] allocates 0 bytes: the small
//! projection coefficients (length `k ≤ 64`) live on the stack.
//!
//! # Modelless
//!
//! Every step is closed-form algebra over the shipped DEC substrate. No
//! training, no backprop — the heat kernel is a matrix exponential, computed
//! analytically in the eigenbasis. Per the katgpt-rs modelless mandate.
//!
//! # References
//!
//! - Plan 359 (this primitive), Research 365 (PhysiFormer distillation).
//! - Plan 251 — DEC operators (`d`, `δ`, `Δ`).
//! - Plan 357 — [`evolve_motor_gated_field`] (the Euler baseline).

use crate::hodge::hodge_eigendecomposition_full;
use crate::types::{CellComplex, CochainField};

/// Maximum number of eigenpairs supported by the stack-allocated projection
/// buffer in the zero-alloc heat-kernel path. Per the SLoD precedent (Plan 235)
/// and the Plan 359 design, 64 eigenvectors is sufficient for typical game maps
/// (a 64×64 grid has 4096 vertices; the top-64 Laplacian modes capture the
/// dominant low-frequency structure).
pub const K_MAX: usize = 64;

/// Threshold for detecting the null-space (zero) eigenvalue of the Laplacian.
/// Eigenvalues below this are treated as zero and their eigenvectors are
/// replaced with the analytical null-space vector (constant for rank-0).
/// Power iteration gives approximate eigenvalues; this threshold catches the
/// zero eigenvalue that power iteration identifies via Rayleigh quotient ≈ 0
/// but cannot produce a valid eigenvector for (because `L·constant = 0`).
pub const NULL_SPACE_THRESHOLD: f32 = 0.01;

/// Precomputed eigendecomposition of the Hodge Laplacian for a [`CellComplex`].
///
/// Computed **once** per complex (offline), reused across all heat-kernel
/// trajectory predictions on that complex. Stores the top-`k` eigenvalues and
/// eigenvectors of the Hodge Laplacian Δₖ (for rank-0 this is the graph
/// Laplacian `L₀`).
///
/// # Why precompute
///
/// The eigendecomposition is `O(n²·k)` to compute (power iteration with
/// deflation, `k` rounds of `max_iter` matvecs at `O(n·avg_degree)` each), but
/// only `O(n·k)` per trajectory prediction. For a fixed game map (the common
/// case), the precompute is amortized over many predictions.
///
/// # Layout
///
/// - `eigenvalues[k]` is the k-th largest eigenvalue of the Laplacian.
/// - `eigenvectors[k * n_cells + i]` is the i-th component of the k-th
///   eigenvector. Each eigenvector is unit-norm.
#[derive(Clone, Debug)]
pub struct DecEigendecomposition {
    /// Rank of the Laplacian (0 for graph Laplacian, k for Hodge Laplacian Δₖ).
    pub rank: u8,
    /// Number of cells `n` in the complex at this rank (each eigenvector has
    /// length `n_cells`).
    pub n_cells: usize,
    /// Top-k eigenvalues of the Laplacian, sorted descending (largest first).
    pub eigenvalues: Vec<f32>,
    /// Top-k eigenvectors, flat row-major: eigenvector `k` occupies indices
    /// `[k * n_cells .. (k+1) * n_cells]`. Length = `k * n_cells`.
    pub eigenvectors: Vec<f32>,
}

impl DecEigendecomposition {
    /// Compute the top-`k` eigenpairs of the Hodge Laplacian for `cx` at the
    /// given `rank`.
    ///
    /// For rank-0 this uses the graph Laplacian fast path
    /// (`graph_laplacian_into`); for rank ≥ 1 it uses the generic Hodge
    /// Laplacian (`hodge_laplacian_into`).
    ///
    /// **Null-space handling (rank-0):** the graph Laplacian has a zero
    /// eigenvalue whose eigenvector is the constant vector (for a connected
    /// complex). Power iteration cannot find this (`L·constant = 0`), so after
    /// the eigensolve, we detect any eigenvalue < `NULL_SPACE_THRESHOLD` and
    /// replace its eigenvector with the unit-norm constant vector. This ensures
    /// the eigendecomposition is a COMPLETE orthonormal basis, which is
    /// essential for the heat kernel's spectral reconstruction to be exact.
    ///
    /// `k` is capped at [`K_MAX`] (64) — the stack-allocated projection buffer
    /// in the zero-alloc heat-kernel path supports up to this many eigenpairs.
    /// Requesting more than `n_cells` eigenpairs silently clamps to `n_cells`.
    ///
    /// `max_iter` controls the power-iteration convergence budget per eigenpair.
    /// A reasonable default is `max_iter = 200` for well-separated spectra,
    /// `max_iter = 500` for clustered spectra. The eigensolver does NOT
    /// allocate inside the iteration loop (reuses pre-allocated cochain
    /// scratch — see [`hodge_eigendecomposition_full`]).
    #[inline]
    pub fn compute(cx: &CellComplex, rank: u8, k: usize, max_iter: usize) -> Self {
        let k_capped = k.min(K_MAX);
        let n = cx.n_cells(rank);
        let (mut eigenvalues, mut eigenvectors) =
            hodge_eigendecomposition_full(cx, rank, k_capped, max_iter);

        // Null-space fix for rank-0: power iteration cannot find the zero
        // eigenvalue (L·constant = 0 → the iteration dies). If we find an
        // eigenvalue below threshold, replace its eigenvector with the
        // unit-norm constant vector. This makes the decomposition a complete
        // orthonormal basis.
        //
        // For rank-0 on a connected complex, the null space is 1-dimensional.
        // For disconnected complexes or rank ≥ 1, this is a known limitation
        // (the harmonic space can be multi-dimensional); the heat kernel still
        // works for fields that live in the non-null subspace.
        if rank == 0 && n > 0 {
            let inv_sqrt_n = 1.0 / (n as f32).sqrt();
            // Find the eigenvalue closest to 0 (smallest in the descending-sorted list).
            if let Some(min_idx) = eigenvalues
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                if eigenvalues[min_idx] < NULL_SPACE_THRESHOLD {
                    eigenvalues[min_idx] = 0.0;
                    let base = min_idx * n;
                    for i in 0..n {
                        eigenvectors[base + i] = inv_sqrt_n;
                    }
                }
            }
        }

        Self {
            rank,
            n_cells: n,
            eigenvalues,
            eigenvectors,
        }
    }

    /// Number of eigenpairs stored.
    #[inline]
    pub fn k(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Read eigenvector `k` as a slice of length `n_cells`.
    #[inline]
    pub fn eigenvector(&self, k: usize) -> &[f32] {
        let start = k * self.n_cells;
        &self.eigenvectors[start..start + self.n_cells]
    }
}

/// Linear heat kernel trajectory: `h(t) = exp(t·A)·h₀` where
/// `A = -I + Δ + diag(motor)`.
///
/// **Exact** for linear propagation (no ReLU gate). Zero error accumulation,
/// exact Hodge-decomposition preservation. At horizon `t`, this is the
/// closed-form solution that the T-step Euler
/// `(I + dt·A)^T·h₀` only approximates (with `O(T·dt²)` global error).
///
/// # Arguments
///
/// - `eig` — The precomputed Hodge-Laplacian eigendecomposition (see
///   [`DecEigendecomposition::compute`]). The `rank` must match `h0.rank`.
/// - `h0` — The initial field at `t = 0`.
/// - `motor_vec` — Per-channel motor gain rates (same convention as
///   [`evolve_motor_gated_field`]): `motor_vec[d]` multiplies channel `d` by
///   `(1 + dt·motor[d])` per Euler step. Channels `d ≥ motor_dim` get
///   `motor[d] = 0` (pure diffusion-decay).
/// - `motor_dim` — Number of motor-gated channels (must be `≤ h0.dim`).
/// - `t` — The prediction horizon. `t = dt` recovers one Euler step;
///   `t = T·dt` recovers the (exact) long-horizon trajectory.
///
/// # Returns
///
/// A new [`CochainField`] of the same rank, `n_cells`, and `dim` as `h0`,
/// holding `h(t) = exp(t·A)·h₀`.
///
/// # Panics
///
/// Debug builds assert `eig.rank == h0.rank`, `eig.n_cells == h0.n_cells()`,
/// and `motor_dim <= h0.dim`.
///
/// # Example
///
/// ```ignore
/// use katgpt_dec::{CellComplex, CochainField, DecEigendecomposition, heat_kernel_trajectory_linear};
///
/// let cx = CellComplex::grid_2d(32, 32);
/// let eig = DecEigendecomposition::compute(&cx, 0, 32, 200);
/// let mut h0 = CochainField::zeros(0, cx.n_vertices(), 4);
/// // ... fill h0 with initial belief state ...
///
/// // Predict the field 50 ticks ahead in a single shot.
/// let h_pred = heat_kernel_trajectory_linear(&eig, &h0, &[0.1, 0.2], 2, 50.0);
/// ```
#[inline]
pub fn heat_kernel_trajectory_linear(
    eig: &DecEigendecomposition,
    h0: &CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    t: f32,
) -> CochainField {
    let mut out = CochainField::zeros(h0.rank, h0.n_cells(), h0.dim);
    heat_kernel_trajectory_linear_into(eig, h0, motor_vec, motor_dim, t, &mut out);
    out
}

/// Zero-alloc linear heat kernel trajectory — writes into `out`.
///
/// Same as [`heat_kernel_trajectory_linear`] but writes into a caller-provided
/// `out` buffer (which is zeroed first, then filled). The projection
/// coefficients live on the stack (`k ≤ 64` floats = ≤ 256 bytes).
///
/// After the eigendecomposition precompute, this function allocates **0 bytes**
/// on the heap — the only allocation is the caller's `out` (which is reused
/// across calls in a steady-state loop). This is the G4 (zero-alloc) gate.
///
/// # Arguments
///
/// - `out` — Output field. Resized to match `h0`'s shape. Existing contents
///   are discarded.
///
/// See [`heat_kernel_trajectory_linear`] for the other arguments.
#[inline]
pub fn heat_kernel_trajectory_linear_into(
    eig: &DecEigendecomposition,
    h0: &CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    t: f32,
    out: &mut CochainField,
) {
    let n = h0.n_cells();
    let dim = h0.dim;
    let k = eig.k();
    let len = n * dim;

    debug_assert_eq!(
        eig.rank, h0.rank,
        "heat_kernel_trajectory_linear: eig.rank {} != h0.rank {}",
        eig.rank, h0.rank
    );
    debug_assert_eq!(
        eig.n_cells, n,
        "heat_kernel_trajectory_linear: eig.n_cells {} != h0.n_cells() {}",
        eig.n_cells, n
    );
    debug_assert!(
        motor_dim <= dim,
        "heat_kernel_trajectory_linear: motor_dim {motor_dim} > h0.dim {dim}"
    );
    debug_assert!(
        motor_vec.len() >= motor_dim,
        "heat_kernel_trajectory_linear: motor_vec len {} < motor_dim {motor_dim}",
        motor_vec.len()
    );
    debug_assert!(
        k <= K_MAX,
        "heat_kernel_trajectory_linear: k={k} exceeds K_MAX={K_MAX} (stack-alloc limit)"
    );

    // Resize out to match h0's shape. In a steady-state loop the caller reuses
    // `out` across calls and this is a no-op (Vec::resize is O(1) when the
    // capacity and length already match).
    out.data.resize(len, 0.0);
    out.dim = dim;
    out.rank = h0.rank;

    // Zero the output (we accumulate into it).
    for v in &mut out.data[..len] {
        *v = 0.0;
    }

    // Stack-allocated projection buffer (k ≤ K_MAX = 64 → ≤ 256 bytes).
    // Holds the projections (v_kᵀ · h_d) for the current channel d.
    let mut proj = [0.0f32; K_MAX];

    // For each channel d, the eigenvalues of A_d = (motor[d] - 1)·I + L are
    // (motor[d] - 1 + λ_k). For channels d >= motor_dim, motor[d] = 0.
    for d in 0..dim {
        let motor_d = if d < motor_dim {
            motor_vec[d]
        } else {
            0.0
        };

        // Project h0 channel d onto each eigenvector: proj[k] = v_kᵀ · h_d(0).
        // This is O(n·k) — the dominant cost.
        for ki in 0..k {
            let v_k = eig.eigenvector(ki);
            let mut dot = 0.0f32;
            // Manual dot-product with stride-dim access into h0.
            // (h0[i*dim + d] for i in 0..n)
            for i in 0..n {
                dot += v_k[i] * h0.data[i * dim + d];
            }
            proj[ki] = dot;
        }

        // Reconstruct h_d(t) = Σ_k proj[k] · exp(t·(motor_d - 1 + λ_k)) · v_k.
        // Accumulate into out: out[i*dim + d] += proj[k] · exp(...) · v_k[i].
        // O(n·k) — the second dominant cost. Total per channel: O(n·k).
        for ki in 0..k {
            let lambda_k = eig.eigenvalues[ki];
            let a_eig = motor_d - 1.0 + lambda_k;
            let scale = proj[ki] * (t * a_eig).exp();
            let v_k = eig.eigenvector(ki);
            for i in 0..n {
                out.data[i * dim + d] += scale * v_k[i];
            }
        }
    }
}

// =============================================================================
// Tests — Plan 359 Phase 1 (T1.1–T1.6) + extras
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::graph_laplacian;
    use crate::types::{CellComplex, CochainField};

    /// Helper: build a rank-0 `w×h` grid cochain with `dim` channels, zeroed.
    fn zero_field(cx: &CellComplex, dim: usize) -> CochainField {
        CochainField::zeros(0, cx.n_vertices(), dim)
    }

    /// Helper: place a Gaussian bump of amplitude `amp` at grid cell `(cx_pos, cy_pos)`
    /// into channel `ch` of a 2D-grid cochain. Non-negative everywhere (so ReLU
    /// is identity — used for linear-vs-Euler comparison tests).
    fn place_bump(
        field: &mut CochainField,
        w: usize,
        h: usize,
        cx_pos: usize,
        cy_pos: usize,
        ch: usize,
        amp: f32,
        sigma: f32,
    ) {
        let dim = field.dim;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx_pos as f32;
                let dy = y as f32 - cy_pos as f32;
                let r2 = dx * dx + dy * dy;
                let v = amp * (-r2 / (2.0 * sigma * sigma)).exp();
                field.data[(y * w + x) * dim + ch] = v;
            }
        }
    }

    /// Helper: one step of linear (no ReLU gate) Euler propagation.
    /// `h₁ = h₀ + dt·(-h₀ + L·h₀ + motor·h₀)` where L is the graph Laplacian.
    /// Mirrors [`evolve_motor_gated_field`] with relu_slope=0 and a non-negative field.
    fn linear_euler_step(
        cx: &CellComplex,
        h: &mut CochainField,
        motor_vec: &[f32],
        motor_dim: usize,
        dt: f32,
    ) {
        let n = h.n_cells();
        let dim = h.dim;
        // Compute L·h (graph Laplacian applied to h, per channel).
        let lap = graph_laplacian(cx, h);
        // h[i] += dt·(lap[i] - h[i] + motor[ch]·h[i])
        //       = dt·lap[i] + (1 - dt + dt·motor[ch])·h[i] - ... wait, motor is per-channel.
        for cell in 0..n {
            for ch in 0..dim {
                let motor = if ch < motor_dim { motor_vec[ch] } else { 0.0 };
                let hi = h.data[cell * dim + ch];
                let li = lap.data[cell * dim + ch];
                // h_{t+1} = h + dt·(-h + L·h + motor·h) = h·(1 - dt + dt·motor) + dt·L·h
                h.data[cell * dim + ch] = hi * (1.0 - dt + dt * motor) + dt * li;
            }
        }
    }

    /// Helper: T steps of linear Euler (the baseline the heat kernel replaces).
    fn linear_euler_trajectory(
        cx: &CellComplex,
        h0: &CochainField,
        motor_vec: &[f32],
        motor_dim: usize,
        dt: f32,
        steps: usize,
    ) -> CochainField {
        let mut h = h0.clone();
        for _ in 0..steps {
            linear_euler_step(cx, &mut h, motor_vec, motor_dim, dt);
        }
        h
    }

    /// Helper: L2 norm of a cochain field.
    fn l2_norm(field: &CochainField) -> f32 {
        field.data.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Helper: L2 distance between two fields.
    fn l2_dist(a: &CochainField, b: &CochainField) -> f32 {
        debug_assert_eq!(a.data.len(), b.data.len());
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    // ── T1.1: DecEigendecomposition struct + compute ──────────────────────────

    #[test]
    fn eigendecomposition_basic() {
        let cx = CellComplex::grid_2d(8, 8);
        let eig = DecEigendecomposition::compute(&cx, 0, 8, 200);
        assert_eq!(eig.rank, 0);
        assert_eq!(eig.n_cells, 64);
        assert_eq!(eig.k(), 8);
        // Eigenvalues sorted descending.
        for i in 1..eig.k() {
            assert!(
                eig.eigenvalues[i - 1] >= eig.eigenvalues[i],
                "eigenvalues not sorted descending at index {}",
                i
            );
        }
        // Each eigenvector is unit-norm.
        for ki in 0..eig.k() {
            let v = eig.eigenvector(ki);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "eigenvector {} has norm {} (expected ~1.0)",
                ki,
                norm
            );
        }
    }

    #[test]
    fn eigendecomposition_caps_k_at_kmax() {
        let cx = CellComplex::grid_2d(16, 16);
        let eig = DecEigendecomposition::compute(&cx, 0, 1000, 50);
        assert_eq!(eig.k(), K_MAX, "k should be capped at K_MAX");
    }

    #[test]
    fn eigendecomposition_eigenvectors_are_eigenvectors() {
        // Verify that each eigenvector v_k satisfies L·v_k ≈ λ_k · v_k.
        let cx = CellComplex::grid_2d(8, 8);
        let eig = DecEigendecomposition::compute(&cx, 0, 8, 300);
        for ki in 0..eig.k() {
            let v_k = eig.eigenvector(ki);
            let lambda_k = eig.eigenvalues[ki];
            // Compute L·v_k
            let cochain = CochainField::from_vec(0, 1, v_k.to_vec());
            let lv = graph_laplacian(&cx, &cochain);
            // Check ||L·v_k - λ_k·v_k|| / ||v_k|| is small.
            let mut residual_sq = 0.0f32;
            for i in 0..v_k.len() {
                let diff = lv.data[i] - lambda_k * v_k[i];
                residual_sq += diff * diff;
            }
            let residual = residual_sq.sqrt();
            assert!(
                residual < 0.5,
                "eigenvector {} residual too large: {} (λ={:.4})",
                ki,
                residual,
                lambda_k
            );
        }
    }

    // ── T1.4: linear heat kernel matches Euler at t=dt ────────────────────────
    //
    // IMPORTANT: the motor-gated linear operator A = L - I + diag(motor) has
    // eigenvalues a_k = λ_k - 1 + motor. For λ_k > 1 - motor, a_k > 0 (unstable).
    // The exact exp(t·A) captures this blow-up; the Euler (I+dt·A)^T masks it
    // for small dt. To compare heat kernel vs Euler meaningfully, we MUST use a
    // STABLE configuration (motor << 1 - λ_max ≈ -7) so no mode blows up. With
    // all a_k < 0, spurious projections from approximate eigenvectors are DAMPED
    // (not amplified), making the comparison numerically robust.

    #[test]
    fn linear_heat_kernel_matches_euler_at_t1() {
        // At t = dt (one step), exp(dt·A)·h₀ ≈ (I + dt·A)·h₀ to within O(dt²).
        // Stable config: motor = -10 → a_k = λ_k - 11, all < 0 (λ_k ≤ 8).
        //
        // Use a SMALL grid (4×4 = 16 vertices) with k = n (full decomposition) so
        // the heat kernel captures ALL modes. Requires max_iter=2000 for the
        // power-iteration-with-deflation eigensolver to converge on ALL 16
        // eigenpairs (the smallest eigenvalues converge slowest after deflation).
        let cx = CellComplex::grid_2d(4, 4);
        let n = cx.n_vertices();
        let eig = DecEigendecomposition::compute(&cx, 0, n, 2000);
        assert_eq!(eig.k(), n, "need full decomposition for exact comparison");

        let mut h0 = zero_field(&cx, 1);
        place_bump(&mut h0, 4, 4, 2, 2, 0, 1.0, 0.8);

        let dt = 0.01f32;
        let motor = [-10.0f32]; // stable: all a_k = λ_k - 11 < 0

        // Heat kernel at t = dt.
        let hk = heat_kernel_trajectory_linear(&eig, &h0, &motor, 1, dt);

        // One Euler step.
        let mut euler = h0.clone();
        linear_euler_step(&cx, &mut euler, &motor, 1, dt);

        let dist = l2_dist(&hk, &euler);
        let scale = l2_norm(&h0).max(1e-6);
        let rel = dist / scale;
        // O(dt²) = O(0.0001) per mode. Relative error should be < 0.5%.
        assert!(
            rel < 0.005,
            "heat kernel vs Euler at t=dt (full decomp): rel dist {} > 0.005 (dist={}, scale={})",
            rel,
            dist,
            scale
        );
    }

    // ── T1.5: heat kernel is exact (Euler drifts) at long horizon ─────────────
    //
    // Strategy: use a SINGLE eigenvector as h₀. Then h(t) = exp(t·a_k)·v_k is
    // a single-mode trajectory — the heat kernel gives it exactly (no
    // multi-mode reconstruction error). Euler accumulates O(T·dt²) drift.
    // This isolates the heat kernel FORMULA from eigenvector accuracy.

    #[test]
    fn linear_heat_kernel_exact_euler_drifts_at_long_horizon() {
        let cx = CellComplex::grid_2d(8, 8);
        let eig = DecEigendecomposition::compute(&cx, 0, 8, 500);
        let n = cx.n_vertices();

        // Use the dominant eigenvector (largest eigenvalue) as h₀.
        // For motor = -10: a_max = λ_max - 11. This is the slowest-decaying mode.
        let motor = [-10.0f32];
        let motor_d = -10.0f32;
        let lambda_k = eig.eigenvalues[0]; // largest eigenvalue of L
        let a_k = lambda_k - 1.0 + motor_d;
        let v_k = eig.eigenvector(0);

        let h0 = CochainField::from_vec(0, 1, v_k.to_vec());

        let dt = 0.1f32;
        let steps = 50usize;
        let t = dt * steps as f32;

        // Exact: h(t) = exp(t·a_k) · v_k. Every component i is v_k[i]·exp(t·a_k).
        let exact_scale = (t * a_k).exp();

        // Heat kernel.
        let hk = heat_kernel_trajectory_linear(&eig, &h0, &motor, 1, t);
        // Check: hk[i] / v_k[i] ≈ exp(t·a_k) for all i where v_k[i] ≠ 0.
        let mut max_rel = 0.0f32;
        for i in 0..n {
            if v_k[i].abs() > 0.01 {
                let hk_scale = hk.data[i] / v_k[i];
                let rel = (hk_scale - exact_scale).abs() / exact_scale.abs().max(1e-10);
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel < 0.05,
            "heat kernel single-mode: max rel error {:.4} (expected scale={}, a_k={}, t={})",
            max_rel,
            exact_scale,
            a_k,
            t
        );

        // Euler: (1 + dt·a_k)^T · v_k. This drifts from exp(T·dt·a_k).
        // (We compute the analytical Euler scale, not the simulated trajectory,
        // to isolate the formula comparison from multi-step accumulation noise.)
        let _euler = linear_euler_trajectory(&cx, &h0, &motor, 1, dt, steps);
        let euler_scale = (1.0 + dt * a_k).powi(steps as i32);
        let euler_rel = (euler_scale - exact_scale).abs() / exact_scale.abs().max(1e-10);
        // Euler's relative drift grows with T·dt². At T=50, dt=0.1: visible.
        assert!(
            euler_rel > 0.01,
            "Euler single-mode drift too small ({:.6}) — test not exercising drift regime",
            euler_rel
        );
        // And the heat kernel is more accurate than Euler.
        assert!(
            max_rel < euler_rel,
            "heat kernel rel error {} should be < Euler rel error {}",
            max_rel,
            euler_rel
        );
    }

    // ── T1.6: Hodge decomposition preserved (stable regime) ────────────────────
    //
    // For a stable system (all a_k < 0), the heat kernel damps each mode by
    // exp(t·a_k). A pure eigenvector field stays a pure eigenvector field —
    // the spectral decomposition is preserved exactly. This IS the Hodge
    // decomposition preservation property for the motor-gated operator.

    #[test]
    fn hodge_decomposition_preserved() {
        // For a pure eigenvector input h₀ = v_k, the output is exp(t·a_k)·v_k —
        // still a pure eigenvector (just scaled). No mode mixing.
        let cx = CellComplex::grid_2d(8, 8);
        let eig = DecEigendecomposition::compute(&cx, 0, 8, 500);
        let n = cx.n_vertices();

        let motor = [-10.0f32];
        let t = 2.0f32;

        // Use the 3rd eigenvector (a mid-spectrum mode).
        let ki = 2;
        let lambda_k = eig.eigenvalues[ki];
        let a_k = lambda_k - 1.0 + (-10.0);
        let v_k = eig.eigenvector(ki);
        let h0 = CochainField::from_vec(0, 1, v_k.to_vec());

        let hk = heat_kernel_trajectory_linear(&eig, &h0, &motor, 1, t);

        // Check: hk is still proportional to v_k (no mode mixing).
        // hk[i] / v_k[i] should be constant ≈ exp(t·a_k) for all i.
        let expected_scale = (t * a_k).exp();
        let mut max_rel = 0.0f32;
        for i in 0..n {
            if v_k[i].abs() > 0.01 {
                let ratio = hk.data[i] / v_k[i];
                let rel = (ratio - expected_scale).abs() / expected_scale.abs().max(1e-10);
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel < 0.05,
            "Hodge preservation: eigenvector {} scaled by {:.6}, expected {:.6}, max_rel {:.4}",
            ki,
            hk.data[0] / v_k[0].max(1e-10),
            expected_scale,
            max_rel
        );
    }

    // ── Extra: multi-channel decoupling ───────────────────────────────────────

    #[test]
    fn multi_channel_decoupling_matches_single_channel() {
        // The block-diagonal structure means a dim=2 field with motor [m, m]
        // should produce the same per-channel trajectory as two dim=1 fields.
        // Stable config: motor = -10.
        let cx = CellComplex::grid_2d(8, 8);
        let eig = DecEigendecomposition::compute(&cx, 0, 8, 500);

        // dim=2 field.
        let mut h0_2 = zero_field(&cx, 2);
        place_bump(&mut h0_2, 8, 8, 4, 4, 0, 1.0, 1.5);
        place_bump(&mut h0_2, 8, 8, 3, 5, 1, 0.7, 1.2);
        let motor = [-10.0f32, -10.0];
        let t = 0.5f32;
        let hk_2 = heat_kernel_trajectory_linear(&eig, &h0_2, &motor, 2, t);

        // dim=1 field for channel 0.
        let mut h0_0 = zero_field(&cx, 1);
        place_bump(&mut h0_0, 8, 8, 4, 4, 0, 1.0, 1.5);
        let hk_0 = heat_kernel_trajectory_linear(&eig, &h0_0, &[-10.0], 1, t);

        // Channel 0 of the dim=2 result should match the dim=1 result.
        let n = cx.n_vertices();
        let mut max_dev = 0.0f32;
        for i in 0..n {
            let dev = (hk_2.data[i * 2] - hk_0.data[i]).abs();
            max_dev = max_dev.max(dev);
        }
        assert!(
            max_dev < 1e-5,
            "multi-channel decoupling: channel 0 max dev {} > 1e-5",
            max_dev
        );
    }

    // ── Extra: zero-alloc variant matches allocating variant ──────────────────

    #[test]
    fn into_variant_matches_allocating() {
        let cx = CellComplex::grid_2d(8, 8);
        let eig = DecEigendecomposition::compute(&cx, 0, 8, 500);
        let mut h0 = zero_field(&cx, 1);
        place_bump(&mut h0, 8, 8, 4, 4, 0, 1.0, 1.5);
        let motor = [-10.0f32];
        let t = 0.5f32;

        let hk_alloc = heat_kernel_trajectory_linear(&eig, &h0, &motor, 1, t);
        let mut hk_into = CochainField::zeros(0, 0, 1); // intentionally wrong size
        heat_kernel_trajectory_linear_into(&eig, &h0, &motor, 1, t, &mut hk_into);

        assert_eq!(hk_alloc.data.len(), hk_into.data.len());
        let max_dev = hk_alloc
            .data
            .iter()
            .zip(hk_into.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_dev < 1e-6,
            "allocating vs into variant max dev {} > 1e-6",
            max_dev
        );
    }

    // ── Extra: stable motor → all finite, no blow-up ─────────────────────────

    #[test]
    fn stable_motor_no_blowup() {
        // With motor = -10 (stable), all a_k = λ_k - 11 < 0, so exp(t·a_k) → 0.
        // The field decays monotonically — no blow-up, no NaN.
        let cx = CellComplex::grid_2d(8, 8);
        let eig = DecEigendecomposition::compute(&cx, 0, 8, 500);
        let n = cx.n_vertices();
        let c = 4.0f32;
        let mut h0 = zero_field(&cx, 1);
        for i in 0..n {
            h0.data[i] = c;
        }
        let t = 5.0f32;
        let hk = heat_kernel_trajectory_linear(&eig, &h0, &[-10.0], 1, t);
        assert!(hk.data.iter().all(|v| v.is_finite()));
        // Stable decay: all values should be small (<< c).
        let max_val = hk.data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(
            max_val < c,
            "stable decay: max val {} should be < initial {}",
            max_val,
            c
        );
    }

    // ── Extra: V2 no-regression smoke ─────────────────────────────────────────

    #[test]
    fn no_regression_smoke() {
        // The heat kernel must compile and run cleanly without affecting the
        // default feature path. This is a smoke test — the real V2 gate is
        // `cargo test --lib` under default features (the feature stays opt-in).
        let cx = CellComplex::grid_2d(4, 4);
        let eig = DecEigendecomposition::compute(&cx, 0, 4, 100);
        let h0 = CochainField::from_vec(0, 1, (0..16).map(|i| i as f32 * 0.1).collect());
        // Stable motor for the smoke test.
        let hk = heat_kernel_trajectory_linear(&eig, &h0, &[-10.0], 1, 1.0);
        assert_eq!(hk.data.len(), 16);
        assert!(hk.data.iter().all(|v| v.is_finite()));
    }

    // ── Extra: t=0 identity (heat kernel at t=0 returns h0) ──────────────────

    #[test]
    fn t_zero_identity() {
        // At t=0, exp(0·A)·h₀ = I·h₀ = h₀. This must hold exactly (up to
        // eigendecomposition accuracy). Requires full decomposition (k=n) with
        // enough iterations for the deflation to converge on all eigenpairs.
        let cx = CellComplex::grid_2d(4, 4);
        let n = cx.n_vertices();
        let eig = DecEigendecomposition::compute(&cx, 0, n, 2000);

        let mut h0 = zero_field(&cx, 1);
        place_bump(&mut h0, 4, 4, 2, 2, 0, 1.0, 0.8);

        let hk = heat_kernel_trajectory_linear(&eig, &h0, &[-10.0], 1, 0.0);

        let dist = l2_dist(&hk, &h0);
        let scale = l2_norm(&h0).max(1e-6);
        let rel = dist / scale;
        assert!(rel < 1e-3, "t=0 identity: rel dist {} > 1e-3", rel);
    }

    // ── Extra: heat kernel matches 4-term Taylor series ───────────────────────

    #[test]
    fn heat_kernel_matches_taylor_series() {
        // Cross-check the heat kernel against an independent computation:
        // a 4-term Taylor series exp(t·A)·h₀ ≈ (I + tA + t²A²/2 + t³A³/6)·h₀.
        // This isolates the spectral reconstruction from the matrix exponential
        // formula — if both agree, the eigenbasis math is correct.
        let cx = CellComplex::grid_2d(4, 4);
        let n = cx.n_vertices();
        let eig = DecEigendecomposition::compute(&cx, 0, n, 2000);

        let mut h0 = zero_field(&cx, 1);
        place_bump(&mut h0, 4, 4, 2, 2, 0, 1.0, 0.8);

        let motor = -10.0f32;
        let t = 0.01f32;

        let hk = heat_kernel_trajectory_linear(&eig, &h0, &[motor], 1, t);

        // Direct Taylor: h₀ + t·A·h₀ + t²·A²·h₀/2 + t³·A³·h₀/6
        // A·h = L·h + (motor-1)·h
        let ah = graph_laplacian(&cx, &h0);
        let mut term1 = zero_field(&cx, 1);
        for i in 0..n {
            term1.data[i] = ah.data[i] + (motor - 1.0) * h0.data[i];
        }
        let a_term1 = graph_laplacian(&cx, &term1);
        let mut term2 = zero_field(&cx, 1);
        for i in 0..n {
            term2.data[i] = a_term1.data[i] + (motor - 1.0) * term1.data[i];
        }
        let a_term2 = graph_laplacian(&cx, &term2);
        let mut term3 = zero_field(&cx, 1);
        for i in 0..n {
            term3.data[i] = a_term2.data[i] + (motor - 1.0) * term2.data[i];
        }

        let mut taylor = h0.clone();
        for i in 0..n {
            taylor.data[i] += t * term1.data[i]
                + (t * t / 2.0) * term2.data[i]
                + (t * t * t / 6.0) * term3.data[i];
        }

        let dist = l2_dist(&hk, &taylor);
        let scale = l2_norm(&h0).max(1e-6);
        let rel = dist / scale;
        // 4-term Taylor is accurate to O(t⁴). At t=0.01, this is ~1e-8.
        assert!(rel < 0.001, "heat kernel vs Taylor: rel {} > 0.001", rel);
    }

    // ── Extra: eigensolver completeness (identity reconstruction) ──────────────

    #[test]
    fn eigendecomposition_is_complete_basis() {
        // Verify that with full decomposition (k=n) and enough iterations,
        // the eigenvectors form a complete orthonormal basis: Σ_k (v_kᵀ·h0)·v_k = h0.
        let cx = CellComplex::grid_2d(4, 4);
        let n = cx.n_vertices();
        let eig = DecEigendecomposition::compute(&cx, 0, n, 2000);

        let h0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut recon = vec![0.0f32; n];
        for ki in 0..eig.k() {
            let v_k = eig.eigenvector(ki);
            let dot: f32 = v_k.iter().zip(h0.iter()).map(|(a, b)| a * b).sum();
            for i in 0..n {
                recon[i] += dot * v_k[i];
            }
        }
        let err: f32 = recon
            .iter()
            .zip(h0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = h0.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            err / norm < 0.01,
            "identity reconstruction rel err {:.4} — eigenvectors not complete basis",
            err / norm
        );
    }
}
