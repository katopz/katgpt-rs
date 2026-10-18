//! PC-ALM dual accumulator + closed-form rate laws (Issue 775, Research 554).
//!
//! Distilled from "Deep Learning without Backpropagation via Augmented
//! Lagrangian Predictive Coding" (PC-ALM, arXiv:2605.31022) — the **dual**
//! half of the interleaved primal-dual pair. Every item here is modelless
//! arithmetic: no training, no backprop, no autodiff graph. The one weight
//! mutation is the dual accumulator itself — latent state, never base
//! weights (the raw/latent sync discipline: at convergence the raw plane
//! `h` returns to forward-pass values; all correction lives in `λ`).
//!
//! # The primitive
//!
//! Constrained form of a layered map: `min f(h) s.t. r_i = h_i − W_i h_{i−1}
//! = 0`, relaxed on the augmented Lagrangian `L_ρ = f + λᵀr + (ρ/2)‖r‖²`.
//! One dual step interleaved per primal step:
//!
//! ```text
//! primal:  h ← h − η_h ∇_h L_ρ        (layer-local: h_{i±1}, λ_{i±1} only)
//! dual:    λ ← λ + α · r(h^new)        (sees the POST-primal residual)
//! ```
//!
//! `α = 0` recovers pure quadratic-penalty relaxation (the incumbent
//! diffusion-mode path — bit-identical by construction: the dual stays
//! zero and the primal reduces to the penalty step). `α = ρ` with an exact
//! inner solve recovers method-of-multipliers.
//!
//! # The identities (paper Eq 8–11, LeCun 1988 App. A)
//!
//! - **Completing the square:** `λᵀr + (ρ/2)‖r‖² = (ρ/2)‖h − (ŷ − λ/ρ)‖² −
//!   ‖λ‖²/(2ρ)` — the augmented Lagrangian IS the penalty energy with every
//!   prediction target shifted by `−λ/ρ` ([`target_shift_into`]).
//! - **Exact adjoint at KKT:** at any feasible KKT point `λ_i = −δ_i`, the
//!   reverse-mode adjoints — so the dual accumulator doubles as a
//!   modelless sensitivity readout on frozen maps
//!   ([`adjoint_readout_into`]).
//! - **Composite credit:** `e = λ + ρ·r` (integral history + current
//!   residual) — the layer signal ([`composite_credit_into`]).
//!
//! # The dynamics laws (paper C.6 — why this is not "add an integrator")
//!
//! Per singular mode `σ` of the constraint operator (for a chain,
//! first-difference/coboundary operators have `σ(k) = 2 sin(k/2)`, so
//! `λ_max ≈ 4`), the interleaved pair has the 2×2 per-mode matrix
//!
//! ```text
//! M = [ 1 − ηρσ²          , −ησ          ]
//!     [ ασ(1 − ηρσ²)      , 1 − αησ²     ]
//! ```
//!
//! with `trace τ = 2 − ησ²(ρ + α)` and `det δ = 1 − ηρσ²` — the determinant
//! is **α-independent** (the annulus law: `|μ±| = √(1 − ηρσ²)`, α slides
//! phase only, never magnitude). The 2×2 Jury/Schur conditions reduce to:
//!
//! - `η ρ σ² < 2` (from `|det| < 1`; at `α = 0` the whole bound)
//! - `η σ² (2ρ + α) < 4` (from `trace > −(1 + det)`) — the headline bound,
//!   recovering PC's `ηρσ² < 2` at `α = 0`.
//!
//! Propagation is **ballistic**: dispersion `μ±(k) ≈ exp(−½ρη_h k² ±
//! i√(αη_h)k)` — a damped wave with group velocity `√(αη_h)`, reach
//! `T√(αη_h)`, arrival `t_infl ≈ L/√(αη_h)` — versus the incumbent
//! diffusion mode's `O(√T)` reach (see [`katgpt_dec::wave_kernel`], the
//! DEC twin of this module, gated `dual_wave` there).
//!
//! # Vocabulary note
//!
//! The dual accumulator here has the same shape as `sheaf_admm`'s scaled
//! dual `u ← u + x − z` (R438/Plan 407) but a different guarantee: no
//! dynamics laws, no adjoint identity there. Grep for "dual" sees both;
//! they compose (the sheaf z-update can become wave-mode via the DEC twin).
//!
//! # References
//!
//! - Issue 775 (this extraction), Research 554 (distillation + prior art).
//! - R438 / Plan 407 — `sheaf_admm` (the diffusion-mode sibling).
//! - Plan 359 — `heat_kernel` (the parabolic family this is the hyperbolic
//!   twin of).
//!
//! PC-ALM dual accumulator + closed-form rate laws (Issue 775, Research 554).
//! The dual half of the interleaved primal-dual pair: ballistic (wave-mode)
//! credit propagation with Jury-certified rates + the exact-adjoint-at-KKT
//! readout. α=0 bit-identical to the incumbent penalty path. The DEC twin
//! (wave-mode interleave on CochainField pairs) is katgpt-dec's
//! `wave_kernel`, behind the same-named flag there. Opt-in.

// ---------------------------------------------------------------------------
// T1 — the accumulator arithmetic (zero-alloc, slice-based)
// ---------------------------------------------------------------------------

/// Dual accumulator step: `λ ← λ + α·r`, elementwise.
///
/// `α = 0` leaves `λ` bit-identical (the pure-penalty/incumbent path).
/// `residual.len()` must equal `lambda.len()` (panics otherwise — slice
/// index).
#[inline]
pub fn dual_accumulate_into(lambda: &mut [f32], residual: &[f32], alpha: f32) {
    debug_assert_eq!(
        lambda.len(),
        residual.len(),
        "dual_accumulate_into: len mismatch ({} vs {})",
        lambda.len(),
        residual.len()
    );
    for (l, &r) in lambda.iter_mut().zip(residual.iter()) {
        *l += alpha * r;
    }
}

/// Completing-the-square target shift: `out = target − λ/ρ`.
///
/// The augmented Lagrangian equals the penalty energy with every prediction
/// target shifted by `−λ/ρ` (paper Eq 8–9). At convergence the raw plane
/// returns to forward-pass values — the shift is read by the primal step,
/// never synced as state.
#[inline]
pub fn target_shift_into(target: &[f32], lambda: &[f32], rho: f32, out: &mut [f32]) {
    debug_assert_eq!(target.len(), lambda.len());
    debug_assert_eq!(target.len(), out.len());
    let inv_rho = 1.0 / rho;
    for k in 0..target.len() {
        out[k] = target[k] - inv_rho * lambda[k];
    }
}

/// Composite credit: `out = λ + ρ·r` — integral history + current residual
/// (paper Eq 10–11). This is the layer signal pulled back through frozen
/// operators downstream; a fixed linear map, no autodiff graph needed.
#[inline]
pub fn composite_credit_into(lambda: &[f32], residual: &[f32], rho: f32, out: &mut [f32]) {
    debug_assert_eq!(lambda.len(), residual.len());
    debug_assert_eq!(lambda.len(), out.len());
    for k in 0..lambda.len() {
        out[k] = lambda[k] + rho * residual[k];
    }
}

/// Dual energy term: `‖λ‖² / (2ρ)` — the constant the completing-the-square
/// identity peels off the augmented Lagrangian. Also the natural magnitude
/// readout for "how much unmet constraint has this node absorbed".
#[inline]
pub fn dual_energy(lambda: &[f32], rho: f32) -> f32 {
    let mut sq = 0.0f32;
    for &l in lambda {
        sq += l * l;
    }
    sq / (2.0 * rho)
}

// ---------------------------------------------------------------------------
// T2 — closed-form rate laws (paper C.6)
// ---------------------------------------------------------------------------

/// Largest stable primal rate for a mode with squared singular value
/// `sigma_sq`: `η_max = 4 / (σ̂² (2ρ + α))` (the Jury bound solved for η).
///
/// At `α = 0` this is `2/(ρσ̂²)` — PC's classical bound.
#[inline]
pub fn jury_eta_max(sigma_sq: f32, rho: f32, alpha: f32) -> f32 {
    4.0 / (sigma_sq * (2.0 * rho + alpha))
}

/// Largest stable dual rate for a mode with squared singular value
/// `sigma_sq`: `α_max = 4/(ησ̂²) − 2ρ` (the same Jury bound solved for α).
#[inline]
pub fn jury_alpha_max(eta: f32, sigma_sq: f32, rho: f32) -> f32 {
    4.0 / (eta * sigma_sq) - 2.0 * rho
}

/// Dynamic regime of one singular mode under the interleaved pair.
///
/// Derived from the per-mode 2×2 matrix (see module docs): Jury/Schur
/// conditions on `trace = 2 − ησ²(ρ+α)`, `det = 1 − ηρσ²`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DualRegime {
    /// Real eigenvalues inside the unit circle — overdamped relaxation,
    /// no phase. Every `α = 0` configuration is here (pure heat).
    Monotone = 0,
    /// Complex conjugate pair on the annulus `|μ±| = √(1 − ηρσ²)` — damped
    /// oscillation, the ballistic regime (group velocity `√(αη)`).
    DampedOscillatory = 1,
    /// A Jury condition is violated — at least one eigenvalue outside the
    /// unit circle. Never ship a configuration that classifies here.
    Unstable = 2,
}

/// Per-mode dynamics summary: the 2×2 invariants plus the closed-form
/// annulus quantities (paper C.6).
#[derive(Clone, Copy, Debug)]
pub struct ModeDynamics {
    /// Trace `τ = 2 − ησ²(ρ + α)` of the per-mode matrix.
    pub trace: f32,
    /// Determinant `δ = 1 − ηρσ²` — α-independent (the annulus law).
    pub det: f32,
    /// Annulus radius `√δ = √(1 − ηρσ²)` (meaningful when `regime` is
    /// `DampedOscillatory`; the magnitude of both eigenvalues).
    pub radius: f32,
    /// Phase cosine `τ / (2√δ)` (ditto; `cos θ` of the eigenvalue angle).
    pub cos_theta: f32,
    /// The regime verdict.
    pub regime: DualRegime,
}

/// Classify one singular mode `(η, σ², ρ, α)` — the closed-form Jury /
/// annulus / phase readout.
///
/// Jury stability (2×2 Schur): `|det| < 1` and `|trace| < 1 + det`, which
/// reduce to `ηρσ² < 2` and `ησ²(2ρ + α) < 4` (see module docs). Complex
/// eigenvalues — the damped-oscillatory regime — iff `(τ/2)² < δ`.
pub fn mode_dynamics(eta: f32, sigma_sq: f32, rho: f32, alpha: f32) -> ModeDynamics {
    let trace = 2.0 - eta * sigma_sq * (rho + alpha);
    let det = 1.0 - eta * rho * sigma_sq;
    let radius = det.max(0.0).sqrt();
    let cos_theta = if radius > 0.0 {
        trace / (2.0 * radius)
    } else {
        1.0
    };

    // Jury stability, reduced form: |det| < 1 ⟺ ηρσ² < 2 (det < 1 always
    // holds for ηρσ² > 0), and trace > −(1+det) ⟺ ησ²(2ρ+α) < 4 (the
    // α > 0 leg of trace < 1+det holds by construction). det ∈ (−1, 0) is
    // LEGAL — real negative eigenvalues, |μ| < 1 (period-2 relaxation).
    let stable = eta * rho * sigma_sq < 2.0 && eta * sigma_sq * (2.0 * rho + alpha) < 4.0;
    let regime = if !stable {
        DualRegime::Unstable
    } else if (0.5 * trace) * (0.5 * trace) < det {
        DualRegime::DampedOscillatory
    } else {
        DualRegime::Monotone
    };
    ModeDynamics {
        trace,
        det,
        radius,
        cos_theta,
        regime,
    }
}

/// Fast predicate: every Jury condition holds for this mode.
#[inline]
pub fn is_jury_stable(eta: f32, sigma_sq: f32, rho: f32, alpha: f32) -> bool {
    mode_dynamics(eta, sigma_sq, rho, alpha).regime != DualRegime::Unstable
}

// ---------------------------------------------------------------------------
// T3 — arrival-time laws (paper Eq 24)
// ---------------------------------------------------------------------------

/// Largest eigenvalue of the path/difference operator family
/// (`σ(k) = 2 sin(k/2)` ⇒ `λ_max → 4` as the chain grows). The
/// self-calibration constant behind [`alpha_reach`].
pub const PATH_OPERATOR_LAMBDA_MAX: f32 = 4.0;

/// Arrival / inflection time: `t_infl ≈ L / √(αη)` — ticks before the
/// ballistic front (group velocity `√(αη)`) reaches depth `L`.
#[inline]
pub fn t_infl(depth: usize, alpha: f32, eta: f32) -> f32 {
    depth as f32 / (alpha * eta).sqrt()
}

/// The `α` that makes the front arrive exactly at `budget_ticks`:
/// `α_reach = L² / (η T²)`. At the self-calibrated budget `T = 2L` with
/// `η = 1/λ_max = 1/4` this is `≈ 1` (the paper's default constant).
#[inline]
pub fn alpha_reach(depth: usize, budget_ticks: usize, eta: f32) -> f32 {
    let l = depth as f32;
    let t = budget_ticks.max(1) as f32;
    l * l / (eta * t * t)
}

/// The self-calibrating tick budget: `T = 2L` (paper Eq 24 — matches BP at
/// this budget across the paper's full (N, L) grid).
#[inline]
pub fn budget_ticks(depth: usize) -> usize {
    2 * depth
}

// ---------------------------------------------------------------------------
// T8 — exact-adjoint readout ("backprop without backprop")
// ---------------------------------------------------------------------------

/// Estimated squared spectral norm of a row-major `d_out × d_in` matrix via
/// a few deterministic power iterations on `WᵀW` (ones-initialized — no
/// RNG dependency). Returns `‖W‖²_F / d_in` as the degenerate fallback for
/// zero matrices.
fn estimate_sigma_sq(w: &[f32], d_in: usize, d_out: usize) -> f32 {
    let mut v = vec![1.0f32; d_in];
    let mut sigma_sq = 0.0f32;
    for _ in 0..8 {
        // u = W v  (d_out)
        let mut u = vec![0.0f32; d_out];
        for r in 0..d_out {
            let mut acc = 0.0f32;
            for c in 0..d_in {
                acc += w[r * d_in + c] * v[c];
            }
            u[r] = acc;
        }
        // v = Wᵀ u  (d_in)
        let mut v_next = vec![0.0f32; d_in];
        for r in 0..d_out {
            let ur = u[r];
            if ur == 0.0 {
                continue;
            }
            for c in 0..d_in {
                v_next[c] += w[r * d_in + c] * ur;
            }
        }
        let norm_sq: f32 = v_next.iter().map(|x| x * x).sum();
        if norm_sq <= 0.0 {
            break;
        }
        sigma_sq = norm_sq; // ‖Wᵀu‖² with ‖u‖≈‖Wv‖… normalized below
        let inv = 1.0 / norm_sq.sqrt();
        for x in v_next.iter_mut() {
            *x *= inv;
        }
        v = v_next;
    }
    if sigma_sq <= 0.0 {
        // Zero matrix (or exhausted): Frobenius fallback.
        let fro: f32 = w.iter().map(|x| x * x).sum();
        return if d_in > 0 { fro / d_in as f32 } else { 0.0 };
    }
    // After normalization ‖v‖ = 1, the Rayleigh quotient vᵀWᵀWv = ‖Wv‖² is
    // the running estimate; recompute it cleanly on the final vector.
    let mut u = vec![0.0f32; d_out];
    for r in 0..d_out {
        let mut acc = 0.0f32;
        for c in 0..d_in {
            acc += w[r * d_in + c] * v[c];
        }
        u[r] = acc;
    }
    u.iter().map(|x| x * x).sum()
}

/// Estimate the squared spectral norm of the STACKED chain constraint
/// operator `A` (rows `r_i = h_i − W_i h_{i−1}`, with `h_0` fixed — the
/// linearized operator sees it as zero) by deterministic power iteration
/// on `AᵀA`, using the scratch activations as the stacked workspace.
/// Accurate where the Gershgorin bound `(1+σ̂_max)²` is loose — and the
/// rate it feeds is the difference between converging at T=2L and
/// converging at all (measured: the bound overshrank η by ~2×).
fn estimate_chain_sigma_sq(
    weights: &[&[f32]],
    dims: &[usize],
    scratch: &mut AdjointScratch,
) -> f32 {
    let n_layers = weights.len();
    scratch.h[0].fill(0.0); // the linearized A reads h_0 as zero
    // Deterministic seed with per-coordinate variation (near-identity
    // chains still excite the top mode).
    for lv in 0..n_layers {
        for (c, x) in scratch.h[lv + 1].iter_mut().enumerate() {
            *x = 1.0 + (c % 7) as f32 * 0.1;
        }
    }
    normalize_stacked(&mut scratch.h);

    let mut sigma_sq = 0.0f32;
    for _ in 0..8 {
        // u = A v  (u_i = v_i − W_i v_{i−1}) — reads v from scratch.h,
        // writes u to scratch.credit.
        for lv in 0..n_layers {
            let (d_out, d_in) = (dims[lv + 1], dims[lv]);
            let w = weights[lv];
            for r in 0..d_out {
                let mut pred = 0.0f32;
                for c in 0..d_in {
                    pred += w[r * d_in + c] * scratch.h[lv][c];
                }
                scratch.credit[lv + 1][r] = scratch.h[lv + 1][r] - pred;
            }
        }
        // z = Aᵀ u  (z_i = u_i − W_{i+1}ᵀ u_{i+1}; z_L = u_L) — writes into
        // scratch.pullback[1..=L] (init-assign, then in-place subtract).
        for lv in 0..n_layers {
            let i = lv + 1;
            let d = dims[i];
            for c in 0..d {
                scratch.pullback[i][c] = scratch.credit[i][c];
            }
            if i < n_layers {
                let w_next = weights[i];
                let d_out = dims[i + 1];
                for r in 0..d_out {
                    let ur = scratch.credit[i + 1][r];
                    if ur == 0.0 {
                        continue;
                    }
                    for c in 0..d {
                        scratch.pullback[i][c] -= w_next[r * d + c] * ur;
                    }
                }
            }
        }
        // σ̂² = ‖AᵀAv‖ (with ‖v‖ = 1, this converges to λ_max(AᵀA) =
        // σ²_max(A) along the top direction) — note norm_sq is ‖z‖², so the
        // estimate is its SQUARE ROOT.
        let mut norm_sq = 0.0f32;
        for lv in 0..n_layers {
            for x in scratch.pullback[lv + 1].iter() {
                norm_sq += x * x;
            }
        }
        if norm_sq <= 0.0 {
            break;
        }
        sigma_sq = norm_sq.sqrt();
        let inv = 1.0 / norm_sq.sqrt();
        for lv in 0..n_layers {
            for c in 0..dims[lv + 1] {
                scratch.h[lv + 1][c] = scratch.pullback[lv + 1][c] * inv;
            }
        }
    }
    if sigma_sq <= 0.0 {
        // Degenerate (zero chain): the Gershgorin bound still works.
        let mut sigma_hat = 0.0f32;
        for lv in 0..n_layers {
            let s = estimate_sigma_sq(weights[lv], dims[lv], dims[lv + 1]).sqrt();
            sigma_hat = sigma_hat.max(s);
        }
        return (1.0 + sigma_hat) * (1.0 + sigma_hat);
    }
    sigma_sq
}

/// Normalize the stacked vector `h[1..=L]` to unit norm (in place).
fn normalize_stacked(h: &mut [Vec<f32>]) {
    let stacked = &mut h[1..];
    let mut norm_sq = 0.0f32;
    for layer in stacked.iter() {
        for x in layer.iter() {
            norm_sq += x * x;
        }
    }
    if norm_sq <= 0.0 {
        return;
    }
    let inv = 1.0 / norm_sq.sqrt();
    for layer in stacked.iter_mut() {
        for x in layer.iter_mut() {
            *x *= inv;
        }
    }
}

/// Normalize a row-major `d_out × d_in` matrix to UNIT spectral norm in
/// place (scale by `1/σ̂`), returning the estimated `σ̂` it had. This is the
/// paper's Fig-5 regime construction: layers with `σ(W) ≈ 1` make the
/// stacked constraint operator a first-difference operator with
/// `λ_max ≈ 4` DEPTH-INDEPENDENT — the regime where `budget_ticks(L) = 2L`
/// is exactly the arrival law `t_infl = L/√(αη)`. Layers with `σ(W) ≠ 1`
/// still converge — but their budget comes from the measured arrival law,
/// not the 2L shortcut (recorded as the depth caveat in Bench 763).
/// Returns 0.0 for the zero matrix (left unscaled).
pub fn normalize_spectral_into(w: &mut [f32], d_in: usize, d_out: usize) -> f32 {
    let sigma_sq = estimate_sigma_sq(w, d_in, d_out);
    if sigma_sq <= 0.0 {
        return 0.0;
    }
    let sigma = sigma_sq.sqrt();
    let inv = 1.0 / sigma;
    for x in w.iter_mut() {
        *x *= inv;
    }
    sigma
}

/// Scratch for [`adjoint_readout_into`] — allocate once per map shape and
/// reuse across readouts. Layer `i` occupies index `i` (`0..=L`).
#[derive(Clone, Debug, Default)]
pub struct AdjointScratch {
    /// Activations `h_0..h_L` (`h_0` is the frozen input).
    pub h: Vec<Vec<f32>>,
    /// Duals `λ_1..λ_L` for the constraints `r_i = h_i − W_i h_{i−1}`
    /// (index 0 unused, kept for layer-index alignment).
    pub lambda: Vec<Vec<f32>>,
    /// Composite credit `e_i = λ_i + ρ r_i` per layer.
    pub credit: Vec<Vec<f32>>,
    /// Pullback buffer `W_{i+1}ᵀ e_{i+1}` per layer.
    pub pullback: Vec<Vec<f32>>,
    /// Per-layer primal rates (recomputed each call — power-iteration σ̂²
    /// × Jury bound; sized here so the call is allocation-free beyond this
    /// buffer's own reuse).
    pub etas: Vec<f32>,
}

impl AdjointScratch {
    /// Size the scratch for a chain with `dims.len() == L + 1` layer widths
    /// (`dims[0]` = input, `dims[L]` = output).
    pub fn new(dims: &[usize]) -> Self {
        let mut h = Vec::with_capacity(dims.len());
        let mut lambda = Vec::with_capacity(dims.len());
        let mut credit = Vec::with_capacity(dims.len());
        let mut pullback = Vec::with_capacity(dims.len());
        for (i, &d) in dims.iter().enumerate() {
            h.push(vec![0.0; d]);
            // λ_0 has no constraint; keep the slot for index alignment.
            lambda.push(if i == 0 { Vec::new() } else { vec![0.0; d] });
            credit.push(if i == 0 { Vec::new() } else { vec![0.0; d] });
            pullback.push(if i == 0 { Vec::new() } else { vec![0.0; d] });
        }
        Self {
            h,
            lambda,
            credit,
            pullback,
            etas: vec![0.0; dims.len() - 1],
        }
    }
}

/// Run the interleaved primal-dual iteration on a **frozen linear chain**
/// `h_i = W_i h_{i−1}` with terminal objective `J = ½‖h_L − y‖², and leave
/// the converged duals in `scratch.lambda` — the exact reverse-mode
/// adjoints at the KKT point (`λ_i → −δ_i`, LeCun 1988 App. A / paper
/// §2.3), by layer-local updates only. No autodiff graph, no weight
/// mutation. One-shot form: [`adjoint_readout_init_into`] + `ticks` ×
/// [`adjoint_readout_tick_into`].
///
/// Rates are set **from the closed-form laws** (T2 consumed in anger): the
/// chain-level squared spectral norm `σ̂²_chain` is estimated by
/// deterministic power iteration on the stacked constraint operator's
/// normal equation (`AᵀA`, block tridiagonal), and
/// `η = safety · jury_eta_max(σ̂²_chain, ρ, α)` with `safety = 0.8`. This
/// is a modelless heuristic for the coupled chain (the Jury classifier is
/// exact per singular mode); convergence at `ticks = budget_ticks(L) = 2L`
/// is verified empirically in the GOAT bench (Bench 763) rather than
/// claimed from the scalar theory.
///
/// `weights[i−1]` is `W_i` as a row-major `dims[i] × dims[i−1]` slice;
/// `dims.len() == weights.len() + 1`.
///
/// # Panics
/// If `dims`/`weights`/`input`/`target` disagree in length.
#[allow(clippy::too_many_arguments)] // solver invocation — all 8 params are genuinely needed
pub fn adjoint_readout_into(
    weights: &[&[f32]],
    dims: &[usize],
    input: &[f32],
    target: &[f32],
    rho: f32,
    alpha: f32,
    ticks: usize,
    scratch: &mut AdjointScratch,
) {
    adjoint_readout_init_into(weights, dims, input, target, rho, alpha, scratch);
    for _ in 0..ticks {
        adjoint_readout_tick_into(weights, dims, target, rho, alpha, scratch);
    }
}

/// Setup pass of [`adjoint_readout_into`]: chain-level Jury rate (σ̂² from
/// power iteration on the stacked AᵀA) + forward init (`h_i = W_i h_{i−1}`,
/// `λ = 0`). Call once, then [`adjoint_readout_tick_into`] as many times as
/// needed — e.g. until the feasibility residual converges (the honest
/// readout protocol; the settling budget is chain-dependent: arrival is
/// ballistic ~L/√(αη), low-mode settling up to ~L²).
///
/// # Panics
/// If `dims`/`weights`/`input` disagree in length.
pub fn adjoint_readout_init_into(
    weights: &[&[f32]],
    dims: &[usize],
    input: &[f32],
    target: &[f32],
    rho: f32,
    alpha: f32,
    scratch: &mut AdjointScratch,
) {
    let n_layers = weights.len();
    assert_eq!(
        dims.len(),
        n_layers + 1,
        "adjoint_readout_init_into: dims.len() {} != weights.len() + 1",
        dims.len()
    );
    assert_eq!(input.len(), dims[0]);
    assert_eq!(target.len(), dims[n_layers]);
    assert_eq!(
        scratch.h.len(),
        dims.len(),
        "scratch shaped for a different chain"
    );

    // ---- chain-level Jury-set primal rate (BEFORE the forward init — the
    //      estimate uses the scratch activations as workspace) -------------
    // The Jury bound applies per singular mode of the STACKED constraint
    // operator A (block-tridiagonal AᵀA). σ̂²_chain from power iteration
    // on AᵀA directly (accurate where the Gershgorin bound is loose — the
    // bound overshrinks η ~2× and with it all convergence; measured).
    let sigma_sq_chain = estimate_chain_sigma_sq(weights, dims, scratch);
    let eta = 0.8 * jury_eta_max(sigma_sq_chain, rho, alpha);
    scratch.etas.iter_mut().for_each(|e| *e = eta);

    // ---- forward init: h_i = W_i h_{i−1}, λ = 0 --------------------------
    scratch.h[0].copy_from_slice(input);
    for lv in 0..n_layers {
        let (d_out, d_in) = (dims[lv + 1], dims[lv]);
        let w = weights[lv];
        assert_eq!(w.len(), d_out * d_in, "W_{} shape mismatch", lv + 1);
        for r in 0..d_out {
            let mut acc = 0.0f32;
            for c in 0..d_in {
                acc += w[r * d_in + c] * scratch.h[lv][c];
            }
            scratch.h[lv + 1][r] = acc;
        }
        scratch.lambda[lv + 1].fill(0.0);
    }
}

/// One interleaved primal-dual tick (Jacobi reads, dual on the POST-primal
/// residual — the interleave that makes propagation ballistic). See the
/// module docs for the update order. Requires a scratch initialized by
/// [`adjoint_readout_init_into`].
///
/// # Panics
/// If `dims`/`weights`/`target` disagree in length.
pub fn adjoint_readout_tick_into(
    weights: &[&[f32]],
    dims: &[usize],
    target: &[f32],
    rho: f32,
    alpha: f32,
    scratch: &mut AdjointScratch,
) {
    let n_layers = weights.len();
    assert_eq!(dims.len(), n_layers + 1);
    assert_eq!(target.len(), dims[n_layers]);

    // 1. Residuals + composite credits from the current snapshot
    //    (e_i = λ_i + ρ r_i).
    for lv in 0..n_layers {
        let (d_out, d_in) = (dims[lv + 1], dims[lv]);
        let w = weights[lv];
        for r in 0..d_out {
            let mut pred = 0.0f32;
            for c in 0..d_in {
                pred += w[r * d_in + c] * scratch.h[lv][c];
            }
            let res = scratch.h[lv + 1][r] - pred;
            scratch.credit[lv + 1][r] = scratch.lambda[lv + 1][r] + rho * res;
        }
    }
    // 2. Pullbacks from those (pre-update) credits — Jacobi semantics.
    refresh_pullbacks(weights, dims, scratch);
    // 3. Primal step: g_i = e_i − W_{i+1}ᵀ e_{i+1} (interior); the
    //    terminal layer adds the objective gradient (h_L − y).
    let interior = scratch.h[1..n_layers]
        .iter_mut()
        .zip(scratch.credit[1..n_layers].iter())
        .zip(scratch.pullback[1..n_layers].iter())
        .zip(scratch.etas[..n_layers - 1].iter());
    for (((h_i, credit_i), pullback_i), &eta) in interior {
        for (hk, (&ck, &pk)) in h_i.iter_mut().zip(credit_i.iter().zip(pullback_i.iter())) {
            let g = ck - pk;
            *hk -= eta * g;
        }
    }
    let eta_last = scratch.etas[n_layers - 1];
    for (hk, (&ck, &yk)) in scratch.h[n_layers]
        .iter_mut()
        .zip(scratch.credit[n_layers].iter().zip(target.iter()))
    {
        let g = ck + (*hk - yk);
        *hk -= eta_last * g;
    }
    // 4. Dual step on the POST-primal residual.
    for lv in 0..n_layers {
        let (d_out, d_in) = (dims[lv + 1], dims[lv]);
        let w = weights[lv];
        for r in 0..d_out {
            let mut pred = 0.0f32;
            for c in 0..d_in {
                pred += w[r * d_in + c] * scratch.h[lv][c];
            }
            let res = scratch.h[lv + 1][r] - pred;
            scratch.lambda[lv + 1][r] += alpha * res;
        }
    }
}

/// Recompute `pullback[i] = W_{i+1}ᵀ credit[i+1]` for interior layers, from
/// the credit buffer as it stands.
fn refresh_pullbacks(weights: &[&[f32]], dims: &[usize], scratch: &mut AdjointScratch) {
    let n_layers = weights.len();
    for i in 1..n_layers {
        // pullback for layer i uses W_{i+1} = weights[i]
        let w = weights[i];
        let d_in = dims[i];
        let d_out = dims[i + 1];
        for k in 0..d_in {
            let mut acc = 0.0f32;
            for r in 0..d_out {
                acc += w[r * d_in + k] * scratch.credit[i + 1][r];
            }
            scratch.pullback[i][k] = acc;
        }
    }
    // Terminal layer has no pullback (objective gradient is local).
    if n_layers > 0 {
        let last = dims[n_layers];
        for k in 0..last {
            scratch.pullback[n_layers][k] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic SplitMix64 (same shape as the dec benches — no rand dep).
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let bits = z >> 40;
            (bits as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    #[test]
    fn dual_accumulate_zero_alpha_is_bit_identical() {
        let mut lambda = vec![0.25f32, -0.5, 1.5, 0.0];
        let before = lambda.clone();
        let residual = vec![1.0f32, 2.0, -3.0, 0.5];
        dual_accumulate_into(&mut lambda, &residual, 0.0);
        assert_eq!(lambda, before); // α=0: the incumbent path, untouched
    }

    #[test]
    fn dual_accumulate_matches_manual() {
        let mut lambda = vec![0.25f32, -0.5, 1.5];
        let residual = vec![1.0f32, 2.0, -3.0];
        let alpha = 0.7f32;
        let expected: Vec<f32> = lambda
            .iter()
            .zip(&residual)
            .map(|(&l, &r)| l + alpha * r)
            .collect();
        dual_accumulate_into(&mut lambda, &residual, alpha);
        assert_eq!(lambda, expected);
    }

    #[test]
    fn completing_the_square_identity_holds() {
        // λᵀr + (ρ/2)‖r‖² == (ρ/2)‖r′‖² − ‖λ‖²/(2ρ) with r′ = r + λ/ρ
        // (the target-shifted residual: h − (ŷ − λ/ρ) = r + λ/ρ).
        let mut rng = SplitMix64(0x775_C0FFEE);
        let n = 32usize;
        let h: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let target: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let lambda: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let rho = 1.3f32;

        let r: Vec<f32> = h.iter().zip(&target).map(|(&a, &b)| a - b).collect();

        let mut shifted_target = vec![0.0f32; n];
        target_shift_into(&target, &lambda, rho, &mut shifted_target);
        let r_prime: Vec<f32> = h
            .iter()
            .zip(&shifted_target)
            .map(|(&a, &b)| a - b)
            .collect();

        let lhs: f32 = lambda.iter().zip(&r).map(|(&l, &x)| l * x).sum::<f32>()
            + 0.5 * rho * r.iter().map(|x| x * x).sum::<f32>();
        let rhs: f32 =
            0.5 * rho * r_prime.iter().map(|x| x * x).sum::<f32>() - dual_energy(&lambda, rho);
        let rel = ((lhs - rhs).abs() / lhs.abs().max(rhs.abs().max(1e-30))) as f64;
        assert!(
            rel < 1e-4,
            "identity drifted: lhs={lhs} rhs={rhs} rel={rel}"
        );
    }

    #[test]
    fn composite_credit_matches_manual() {
        let lambda = vec![0.3f32, -0.2, 0.9];
        let residual = vec![-1.0f32, 0.4, 2.0];
        let rho = 0.8f32;
        let mut out = vec![0.0f32; 3];
        composite_credit_into(&lambda, &residual, rho, &mut out);
        for k in 0..3 {
            assert_eq!(out[k], lambda[k] + rho * residual[k]);
        }
    }

    #[test]
    fn jury_bounds_recover_pc_at_alpha_zero() {
        // At α=0: η_max = 2/(ρσ²) — PC's classical bound.
        let (sigma_sq, rho) = (2.5f32, 1.7f32);
        let eta0 = jury_eta_max(sigma_sq, rho, 0.0);
        assert!((eta0 - 2.0 / (rho * sigma_sq)).abs() < 1e-6);
        // Just inside: stable. Just outside: unstable.
        assert!(is_jury_stable(eta0 * 0.99, sigma_sq, rho, 0.0));
        assert!(!is_jury_stable(eta0 * 1.01, sigma_sq, rho, 0.0));
        // α loosens the η budget: η_max decreases as α grows.
        let eta_a = jury_eta_max(sigma_sq, rho, 0.5);
        assert!(eta_a < eta0);
        // And the solved-for-α form agrees at the boundary:
        let alpha_max = jury_alpha_max(eta0, sigma_sq, rho);
        assert!((alpha_max).abs() < 1e-5); // η = η_max(α=0) ⇒ α_max = 0
    }

    #[test]
    fn classifier_matches_quadratic_oracle() {
        // Independent oracle: brute-force eigenvalues of the per-mode 2×2
        // via the quadratic formula; compare verdicts over a parameter grid.
        let mut mismatches = 0usize;
        for &eta in &[0.05f32, 0.15, 0.25, 0.4, 0.6] {
            for &sigma_sq in &[0.5f32, 1.0, 2.0, 3.5, 4.0] {
                for &rho in &[0.5f32, 1.0, 2.0] {
                    for &alpha in &[0.0f32, 0.3, 1.0, 2.0, 3.5] {
                        let m = mode_dynamics(eta, sigma_sq, rho, alpha);
                        // Skip cells within f32 noise of a Jury boundary:
                        // the classifier's strict inequality reads the exact
                        // boundary (μ = −1, marginal) as Unstable — the safe
                        // engineering call — while the oracle reads |μ| ≈ 1.
                        let b1 = (eta * sigma_sq * (2.0 * rho + alpha) - 4.0).abs();
                        let b2 = (eta * rho * sigma_sq - 2.0).abs();
                        if b1 < 1e-4 || b2 < 1e-4 {
                            continue;
                        }
                        // Oracle in f64 — the f32 quadratic formula loses
                        // the α=0 triangular case (true eigenvalue exactly
                        // 1, computed as 1+ε ⇒ phantom Unstable).
                        let a = 1.0 - (eta * rho * sigma_sq) as f64;
                        let sig = (sigma_sq as f64).sqrt();
                        let b = -(eta as f64) * sig;
                        let c = (alpha as f64) * sig * a;
                        let d = 1.0 - (alpha as f64) * (eta as f64) * (sigma_sq as f64);
                        let tr = a + d;
                        let det = a * d - b * c;
                        let disc = tr * tr - 4.0 * det;
                        // Skip degenerate cells (|disc| ≈ 0 — the exact
                        // complex/real transition, decided by rounding).
                        if disc.abs() < 1e-4 {
                            continue;
                        }
                        let (mu1, mu2, complex) = if disc < 0.0 {
                            (det.sqrt(), -det.sqrt(), true)
                        } else {
                            let s = disc.sqrt();
                            ((tr + s) / 2.0, (tr - s) / 2.0, false)
                        };
                        let spectral = mu1.abs().max(mu2.abs());
                        // Marginal (|μ| == 1 to f32 noise — the α=0 dual
                        // direction is pinned, never growing) is NOT
                        // unstable: that is the incumbent bit-identity path.
                        let oracle = if spectral > 1.0 + 1e-6 {
                            DualRegime::Unstable
                        } else if complex {
                            DualRegime::DampedOscillatory
                        } else {
                            DualRegime::Monotone
                        };
                        if oracle != m.regime {
                            mismatches += 1;
                            eprintln!(
                                "mismatch at eta={eta} s2={sigma_sq} rho={rho} a={alpha}: \
                                 oracle={oracle:?} classifier={:?}",
                                m.regime
                            );
                        }
                        // Annulus law: |det classifier| matches oracle det.
                        assert!((m.det - det as f32).abs() < 1e-4);
                        if complex && m.regime == DualRegime::DampedOscillatory {
                            assert!((m.radius - spectral as f32).abs() < 1e-4);
                        }
                    }
                }
            }
        }
        assert_eq!(mismatches, 0, "classifier diverged from the oracle");
    }

    #[test]
    fn annulus_radius_is_alpha_independent() {
        let (eta, sigma_sq, rho) = (0.25f32, 2.0f32, 1.0f32);
        let m1 = mode_dynamics(eta, sigma_sq, rho, 0.5);
        let m2 = mode_dynamics(eta, sigma_sq, rho, 2.0);
        assert_eq!(m1.det, m2.det);
        assert_eq!(m1.radius, m2.radius);
        // α moves the phase only:
        assert!(m2.cos_theta < m1.cos_theta);
    }

    #[test]
    fn alpha_zero_is_always_monotone() {
        // (τ/2)² < δ requires η²ρ²σ⁴/4 < 0 at α=0 — never. Pure heat does
        // not oscillate.
        for &eta in &[0.05f32, 0.25, 0.7] {
            for &sigma_sq in &[0.5f32, 2.0, 4.0] {
                for &rho in &[0.5f32, 1.0, 2.0] {
                    let m = mode_dynamics(eta, sigma_sq, rho, 0.0);
                    assert_ne!(m.regime, DualRegime::DampedOscillatory);
                }
            }
        }
    }

    #[test]
    fn arrival_self_calibration() {
        for &l in &[4usize, 16, 64, 128] {
            assert_eq!(budget_ticks(l), 2 * l);
            // T = 2L, η = 1/λ_max = 1/4 ⇒ α_reach ≈ 1 (the paper default).
            let eta = 1.0 / PATH_OPERATOR_LAMBDA_MAX;
            let alpha = alpha_reach(l, budget_ticks(l), eta);
            assert!((alpha - 1.0).abs() < 1e-4, "alpha={alpha}");
            // And the arrival law is consistent with that α:
            let t = t_infl(l, alpha, eta);
            assert!((t - 2.0 * l as f32).abs() < 1e-3, "t_infl={t}");
        }
    }

    // -------------------------------------------------------------------
    // T8 — adjoint readout
    // -------------------------------------------------------------------

    fn random_chain(l: usize, d: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut rng = SplitMix64(seed);
        let dims = vec![d; l + 1];
        let mut weights = Vec::with_capacity(l);
        for _ in 0..l {
            let mut w = Vec::with_capacity(d * d);
            for _ in 0..d * d {
                w.push(rng.next_f32() / (d as f32).sqrt());
            }
            weights.push(w);
        }
        (weights, dims)
    }

    /// Explicit reverse-mode adjoints at the forward point:
    /// δ_L = h_L − y, δ_i = W_{i+1}ᵀ δ_{i+1}.
    fn reverse_mode(
        weights: &[Vec<f32>],
        dims: &[usize],
        input: &[f32],
        target: &[f32],
    ) -> Vec<Vec<f32>> {
        let n_layers = weights.len();
        // Forward.
        let mut h: Vec<Vec<f32>> = vec![vec![0.0; dims[0]]; dims.len()];
        h[0].copy_from_slice(input);
        for lv in 0..n_layers {
            let (d_out, d_in) = (dims[lv + 1], dims[lv]);
            for r in 0..d_out {
                let mut acc = 0.0f32;
                for c in 0..d_in {
                    acc += weights[lv][r * d_in + c] * h[lv][c];
                }
                h[lv + 1][r] = acc;
            }
        }
        // Reverse.
        let mut delta: Vec<Vec<f32>> = vec![Vec::new(); dims.len()];
        let last = n_layers;
        delta[last] = h[last].iter().zip(target).map(|(&a, &b)| a - b).collect();
        for lv in (0..n_layers).rev() {
            let i = lv + 1;
            let (d_out, d_in) = (dims[i], dims[i - 1]);
            let mut d_prev = vec![0.0f32; d_in];
            for r in 0..d_out {
                let dr = delta[i][r];
                for c in 0..d_in {
                    d_prev[c] += weights[lv][r * d_in + c] * dr;
                }
            }
            delta[lv] = d_prev;
        }
        delta
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for k in 0..a.len() {
            dot += a[k] * b[k];
            na += a[k] * a[k];
            nb += b[k] * b[k];
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn adjoint_readout_matches_reverse_mode_linear_chain() {
        // λ_i → −δ_i at the KKT point (LeCun 1988). The GOAT gate (cos ≥ 0.9
        // at T=2L) runs in Bench 763 on the same construction; this unit
        // test pins convergence on one fixed chain at a generous budget.
        let (l, d) = (4usize, 8usize);
        let (weights, dims) = random_chain(l, d, 0x775_A0D0);
        let wrefs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
        let mut rng = SplitMix64(0x775_A0D1);
        let input: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let target: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();

        let delta = reverse_mode(&weights, &dims, &input, &target);

        let mut scratch = AdjointScratch::new(&dims);
        adjoint_readout_into(
            &wrefs,
            &dims,
            &input,
            &target,
            1.0,
            1.0,
            8 * l,
            &mut scratch,
        );

        for (i, d_i) in delta.iter().enumerate().take(l + 1).skip(1) {
            let neg_delta: Vec<f32> = d_i.iter().map(|x| -x).collect();
            let cos = cosine(&scratch.lambda[i], &neg_delta);
            assert!(
                cos > 0.99,
                "layer {i}: cos(λ, −δ) = {cos} (λ={:?} −δ={:?})",
                scratch.lambda[i],
                neg_delta
            );
        }
        // Feasibility: the raw plane returns to forward-pass values.
        // (Residual magnitudes shrink to ~0 — spot-check the last layer.)
        let mut pred = vec![0.0f32; d];
        for r in 0..d {
            for c in 0..d {
                pred[r] += weights[l - 1][r * d + c] * scratch.h[l - 1][c];
            }
        }
        let res_norm: f32 = (0..d).map(|r| (scratch.h[l][r] - pred[r]).powi(2)).sum();
        assert!(res_norm < 1e-3, "terminal residual norm² = {res_norm}");
    }

    #[test]
    fn adjoint_readout_matches_finite_differences() {
        // δ_i = dJ/dh_i through the downstream chain only — the honest FD
        // of the frozen map (fix h_i, vary one coordinate, run layers
        // i+1..L). Spot-check layer 1 and 2 coordinates on a 3-layer chain.
        // 16·L ticks — the FD spot-check is a per-COORDINATE magnitude
        // match (stricter than the directional cosine gate); it runs past
        // the transient so the comparison isolates the identity λ = −δ.
        let (l, d) = (3usize, 6usize);
        let (weights, dims) = random_chain(l, d, 0x0007_75FD);
        let wrefs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
        let mut rng = SplitMix64(0x775_FD2);
        let input: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let target: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();

        let mut scratch = AdjointScratch::new(&dims);
        adjoint_readout_into(
            &wrefs,
            &dims,
            &input,
            &target,
            1.0,
            1.0,
            16 * l,
            &mut scratch,
        );

        // FD on h_1 (layer index 1): run layers 2..L from a perturbed h_1.
        let eps = 1e-3f32;
        for probe_layer in [1usize, 2usize] {
            // h at the probe layer, post-convergence (≈ forward values).
            let h_probe = scratch.h[probe_layer].clone();
            for c in [0usize, d / 2, d - 1] {
                let j = |dir: f32| -> f32 {
                    let mut hp = h_probe.clone();
                    hp[c] += eps * dir;
                    let mut cur = hp;
                    for w in weights.iter().take(l).skip(probe_layer) {
                        let mut next = vec![0.0f32; d];
                        for r in 0..d {
                            for cc in 0..d {
                                next[r] += w[r * d + cc] * cur[cc];
                            }
                        }
                        cur = next;
                    }
                    0.5 * cur
                        .iter()
                        .zip(&target)
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<f32>()
                };
                let fd = (j(1.0) - j(-1.0)) / (2.0 * eps);
                let dual_fd = -scratch.lambda[probe_layer][c]; // λ = −δ
                let denom = fd.abs().max(dual_fd.abs()).max(1e-3);
                let rel = ((fd - dual_fd).abs() / denom) as f64;
                assert!(
                    rel < 5e-2,
                    "layer {probe_layer} coord {c}: FD={fd} λ={dual_fd} rel={rel}"
                );
            }
        }
    }
}
