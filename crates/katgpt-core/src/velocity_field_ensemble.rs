//! Velocity-Field Ensemble — Algebraic Combination of Pre-Trained Models
//! (Plan 376, Research 375).
//!
//! A modelless, inference-time primitive: combine **P frozen pre-trained
//! velocity fields** (any forward model: LLM drafter, HLA forecaster, LinOSS
//! drafter, KARC forecaster, archetype operator field) into a single
//! regression-optimal combined drift
//!
//! ```text
//! b̂(x) = Σ_i η_i · b_i(x)
//! ```
//!
//! where `η ∈ R^P` is **solved once from N data pairs** via the existing
//! [`crate::linalg::ridge_solve::ridge_solve_direct_f32`] P×P Cholesky path.
//!
//! # The math (one paragraph)
//!
//! Given N data pairs `(I_t^n, İ_t^n)` — interpolant samples and their
//! time-derivatives — we solve the normal equations of the least-squares
//! problem `min_η Σ_n ‖Σ_i η_i b_i(I_t^n) − İ_t^n‖²`:
//!
//! ```text
//! K_t η_t = r_t
//! K_t[i,j] = (1/N) Σ_n b_i(I_t^n) · b_j(I_t^n)     (P×P Gram)
//! r_t[i]   = (1/N) Σ_n b_i(I_t^n) · İ_t^n           (P-dim RHS)
//! ```
//!
//! This is exactly KARC's ridge solve (`linalg::ridge_solve::ridge_solve_direct_f32`)
//! applied to a basis of P frozen model outputs instead of delay-embedded
//! features. The output `b̂` is the least-squares-best linear combination of
//! the P fields for fitting the observed `(I_t, İ_t)` pairs. Solved once;
//! reused for any number of evals. No backprop, no gradient descent, no
//! softmax. η CAN be negative — this is a signed linear combination, not a
//! probabilistic mixture (see AGENTS.md: sigmoid, not softmax — but η is
//! regression-solved, not sigmoid-normalized either).
//!
//! # Distilled from
//!
//! Coeurdoux et al., *Generative Modeling via Kernelized Stochastic
//! Interpolants*, ICML 2026 SPIGM Workshop (arXiv:2602.20070). Paper
//! Proposition 2.1: the paper's "feature gradient" basis becomes "frozen
//! model forward outputs"; the paper's `K_t η_t = r_t` system becomes our
//! `ridge_solve_direct_f32`. The combination is regression-optimal for the
//! target distribution (data pairs), valid across heterogeneous architectures
//! (paper §2.5 + Appendix E cross-domain composition).
//!
//! # Why this is NOT a duplicate of KARC
//!
//! The P×P ridge solve IS KARC's math — we reuse [`ridge_solve_direct_f32`]
//! directly, never re-implement. The contribution is the **basis
//! construction**: P velocity-field outputs as features, not delay-embedded
//! basis-expanded observations. KARC's basis is `Ψ(delay_embed(x))`; our
//! basis is `[b_1(x), b_2(x), …, b_P(x)]` — the forward passes of P frozen
//! models. Anyone reviewing this module should grep `ridge_solve_direct_f32`
//! and confirm KARC's `fit_direct` is the same linear-algebra operation.
//!
//! # Zero-allocation contract
//!
//! All hot-path operations write into caller-provided scratch:
//! - [`VelocityFieldEnsemble::fit_into`] takes `&mut EnsembleFitScratch<P, D>`.
//! - [`VelocityFieldEnsemble::eval_into`] takes `&mut [f32; D]` per-field scratch.
//! - [`VelocityFieldEnsemble::eval_batch_into`] reuses the same scratch across the batch.
//! - [`stochastic_interpolant_step_into`] writes into `x_out` and takes the
//!   drift slice directly.
//!
//! # Feature gate
//!
//! Gated behind the `velocity_field_ensemble` Cargo feature. **DEFAULT-ON**
//! since Plan 376 Phase 3 (2026-07-04) — G1 (9/9 unit tests, η recovery <1e-4),
//! G2 (3/3 cross-domain metrics, 3.5× MSE reduction), G3 (0 allocs + clean),
//! G4 (fit_into 6.27µs<50µs + eval_into 21ns<200ns) ALL PASS. See
//! `katgpt-rs/.plans/376_velocity_field_ensemble_primitive.md`.
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/376_velocity_field_ensemble_primitive.md`
//! - **Research:** `katgpt-rs/.research/375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md`
//! - **Source paper:** arXiv:2602.20070 — Coeurdoux et al., ICML 2026 SPIGM
//! - **Sibling ridge path:** `crates/katgpt-core/src/karc.rs` (Plan 308) — same
//!   `(K + λI)^{-1} r` math via `linalg::ridge_solve`.
//! - **Sibling composition path:** `crates/katgpt-core/src/committed_field_blend.rs`
//!   (Plan 321) — sigmoid-projected weights (voting); this module is the
//!   regression-optimal counterpart (least-squares).
//!
//! Velocity-Field Ensemble — Algebraic Combination of Pre-Trained Models
//! (Plan 376, Research 375, arXiv:2602.20070 Coeurdoux et al. ICML 2026 SPIGM).
//! Combine P frozen pre-trained velocity fields (any forward model: LLM
//! drafter, HLA forecaster, KARC forecaster, archetype operator field) into a
//! single regression-optimal combined drift b̂(x) = Σ_i η_i · b_i(x), where η
//! is solved once from N data pairs via the existing linalg::ridge_solve
//! P×P Cholesky path (the SAME math KARC uses — KARC's basis is delay-embedded
//! features; this primitive's basis is P frozen model forward outputs).
//!
//! The contribution is the *basis construction*, NOT the ridge solve — anyone
//! reviewing should grep `ridge_solve_direct_f32` and confirm KARC's `fit_direct`
//! is the same linear-algebra operation. No duplicate math; pure DRY reuse.
//!
//! η CAN be negative (signed combination, not probabilistic mixture). No
//! softmax anywhere; no sigmoid on η either (η is regression-solved, not
//! projected). The sigmoid-not-softmax rule applies to *gating*, not to
//! regression-optimal weights.
//!
//! Includes the optimal-diffusion SDE integrator (paper Algorithm 1, eq. 14
//! with D*_t = α_t γ_t / β_t) as a decoupled utility — composes with any drift
//! source, not just the ensemble.
//!
//! DEFAULT-ON (Plan 376 Phase 3, 2026-07-04): G1–G4 ALL PASS. G2 (cross-domain
//! quality) is the make-or-break gate — the paper proves cross-domain
//! composition for image generation only; Phase 2 PoC is mandatory before any
//! quality-parity claim for game AI.

use crate::linalg::ridge_solve::ridge_solve_direct_f32;
use crate::simd::simd_dot_f32;

// Heterogeneous-D support (Phase 4) calls the lower-level project/reconstruct
// functions directly to avoid CrossResScratch's resize-on-k-change behavior
// (which would allocate when fields have different k values).
#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
use crate::cross_resolution::{
    CrossResolutionBases, project_to_spectral_into, reconstruct_from_spectral_into,
};

// ── Trait ─────────────────────────────────────────────────────────────────

/// A frozen forward model whose output is a velocity/drift vector in `R^D`.
///
/// Implementors: LLM drafters, HLA forecasters, KARC forecasters, LinOSS
/// ModalSpec drafters, archetype operator fields — any pre-trained model
/// whose forward pass produces a `D`-dim direction. The ensemble treats
/// these as the regression basis (paper §2.5).
///
/// All `P` ensemble members must agree on `D`. Use Cross-Resolution transport
/// (Plan 310) to project heterogeneous-`d` members to a common `D` before
/// ensemble fit (deferred — Phase 4).
///
/// Const-generic `D` follows KARC's `KarcBasis<const M>` pattern: the output
/// dimension is a type-level guarantee, not a runtime check. For closures
/// that can't directly implement this trait, wrap them in [`ClosureField`].
pub trait VelocityField<const D: usize> {
    /// Evaluate the velocity field at state `x`, writing into `out`.
    ///
    /// Zero-allocation contract: implementor MUST NOT allocate; `out` is
    /// caller-provided scratch of length `D`. `x` length is implementor-defined
    /// (some fields take raw state, others take delay-embedded state, etc).
    fn eval_into(&self, x: &[f32], out: &mut [f32; D]);

    /// Identifier for BLAKE3 commitment of this field's frozen weights.
    /// Two ensemble members with the same `field_id` are duplicates (the
    /// Gram becomes singular; caller should add ridge regularization `λ > 0`).
    fn field_id(&self) -> u64;
}

/// Wrapper to adapt any closure `Fn(&[f32], &mut [f32; D])` into a
/// [`VelocityField<D>`].
///
/// Usage:
/// ```no_run
/// # use katgpt_core::velocity_field_ensemble::{ClosureField, VelocityField};
/// let field = ClosureField::<4, _>::new(
///     0x1234_5678u64,
///     |x: &[f32], out: &mut [f32; 4]| {
///         out[0] = x[0] * 2.0;
///         out[1] = x[1] + 1.0;
///         out[2] = x[2] - x[3];
///         out[3] = x[3] * 0.5;
///     },
/// );
/// ```
///
/// This avoids the awkward blanket-impl-on-closure problem with const-generic
/// `D` (a blanket `impl<F: Fn(&[f32], &mut [f32; D])> VelocityField<D> for F`
/// would conflict with `D` appearing on the impl, not the closure type).
#[derive(Clone, Copy)]
pub struct ClosureField<const D: usize, F>
where
    F: Fn(&[f32], &mut [f32; D]),
{
    id: u64,
    f: F,
}

impl<const D: usize, F> ClosureField<D, F>
where
    F: Fn(&[f32], &mut [f32; D]),
{
    /// Construct a `VelocityField<D>` from a closure and a `field_id`
    /// (typically the BLAKE3 hash of the frozen weights, truncated to `u64`).
    #[inline]
    pub const fn new(id: u64, f: F) -> Self {
        Self { id, f }
    }
}

impl<const D: usize, F> VelocityField<D> for ClosureField<D, F>
where
    F: Fn(&[f32], &mut [f32; D]),
{
    #[inline]
    fn eval_into(&self, x: &[f32], out: &mut [f32; D]) {
        (self.f)(x, out);
    }

    #[inline]
    fn field_id(&self) -> u64 {
        self.id
    }
}

// ── Core ensemble ─────────────────────────────────────────────────────────

/// An algebraic ensemble of `P` frozen velocity fields, combined via
/// regression-optimal weights `η ∈ R^P` solved from data pairs.
///
/// The combined drift is `b̂(x) = Σ_i η_i · b_i(x)`, evaluated zero-alloc via
/// [`VelocityFieldEnsemble::eval_into`]. The weights `η` are solved once via
/// [`VelocityFieldEnsemble::fit_into`] and frozen for the lifetime of the
/// target (NPC, game episode, generation run).
///
/// All `P` fields must be the same Rust type `F`. For heterogeneous field
/// types, wrap them in an enum that implements [`VelocityField<D>`].
///
/// `P` is the ensemble size (number of velocity fields); `D` is the output
/// dimension (shared across all fields).
#[derive(Clone)]
pub struct VelocityFieldEnsemble<F, const P: usize, const D: usize>
where
    F: VelocityField<D>,
{
    /// The `P` frozen velocity fields. Immutable after construction (the
    /// "freeze" half of freeze/thaw — see AGENTS.md).
    pub fields: [F; P],
    /// Solved combination weights (regression-optimal for the fit data).
    /// Initialized to uniform `1/P` by [`VelocityFieldEnsemble::new`];
    /// overwritten by [`VelocityFieldEnsemble::fit_into`].
    pub eta: [f32; P],
}

impl<F, const P: usize, const D: usize> VelocityFieldEnsemble<F, P, D>
where
    F: VelocityField<D>,
{
    /// Construct an ensemble from `P` frozen fields. Weights `η` start at
    /// uniform `1/P` (equal-weight average) — call [`Self::fit_into`] to
    /// solve the regression-optimal weights from data pairs.
    pub fn new(fields: [F; P]) -> Self {
        // Equal-weight init: η_i = 1/P. This is the "no-fit" fallback (the
        // ensemble reduces to a plain average). It is NOT regression-optimal;
        // callers who want the paper's gain MUST call `fit_into`.
        // Splat directly instead of zero-init-then-overwrite.
        let eta = [1.0 / (P as f32); P];
        Self { fields, eta }
    }

    /// Returns the solved weight vector `η`. Length `P`.
    #[inline]
    pub fn eta(&self) -> &[f32; P] {
        &self.eta
    }
}

// ── Fit scratch ───────────────────────────────────────────────────────────

/// Zero-allocation scratch buffer for [`VelocityFieldEnsemble::fit_into`].
///
/// Allocate once (via [`Self::new``]), reuse across fits. The `P×P` buffers
/// (`gram`, `gram_reg`, `chol`) are ``Vec<f32>`` because stable Rust does not
/// allow `P * P` in array types when `P` is a const-generic parameter (would
/// require the `generic_const_exprs` nightly feature). The Vec is allocated
/// exactly once at construction; the hot path (`fit_into`, `eval_into`)
/// performs zero allocations.
///
/// The `P`-dim and `D`-dim buffers (`rhs`, `z_solve`, `b_out_i`, `b_out_j`)
/// are fixed-size `[f32; N]` arrays — single const generics are allowed.
pub struct EnsembleFitScratch<const P: usize, const D: usize> {
    /// `K_t[i,j] = E[b_i(I_t)·b_j(I_t)]` — the `P×P` Gram matrix (row-major).
    pub gram: Vec<f32>,
    /// `r_t[i] = E[b_i(I_t)·İ_t]` — the `P`-dim RHS vector.
    pub rhs: [f32; P],
    /// `K + λI` (Gram + ridge diagonal). Input to `ridge_solve_direct_f32`.
    pub gram_reg: Vec<f32>,
    /// Cholesky `L` factor scratch (`P×P`). Overwritten by the solver.
    pub chol: Vec<f32>,
    /// Back-substitution scratch (`z` in `L z = r`, `Lᵀ η = z`). Length `P`.
    pub z_solve: [f32; P],
    /// Per-field eval scratch for `b_i(I_t)`. Length `D`.
    pub b_out_i: [f32; D],
    /// Per-field eval scratch for `b_j(I_t)` (the `j`-loop inner field).
    /// Length `D`.
    pub b_out_j: [f32; D],
}

impl<const P: usize, const D: usize> EnsembleFitScratch<P, D> {
    /// Construct a zero-initialized scratch buffer. Allocates the three `P×P`
    /// buffers once; subsequent `fit_into` calls reuse them without allocation.
    pub fn new() -> Self {
        Self {
            gram: vec![0.0; P * P],
            rhs: [0.0; P],
            gram_reg: vec![0.0; P * P],
            chol: vec![0.0; P * P],
            z_solve: [0.0; P],
            b_out_i: [0.0; D],
            b_out_j: [0.0; D],
        }
    }

    /// Zero all accumulation buffers (Gram, RHS). Does not touch solver
    /// scratch (`gram_reg`, `chol`, `z_solve`) — those are overwritten by
    /// `fit_into` before use.
    #[inline]
    pub fn clear(&mut self) {
        self.gram.fill(0.0);
        self.rhs.fill(0.0);
    }
}

impl<const P: usize, const D: usize> Default for EnsembleFitScratch<P, D> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Accumulate one pair ───────────────────────────────────────────────────

/// Accumulate one data pair `(I_t, İ_t)` into the Gram and RHS scratch.
///
/// For each `(i, j)` with `i ≤ j`, accumulates `b_i(I_t)·b_j(I_t)` into
/// `scratch.gram[i*P + j]` (and the symmetric mirror `[j*P + i]`).
/// For each `i`, accumulates `b_i(I_t)·İ_t` into `scratch.rhs[i]`.
///
/// `i_t` is the interpolant sample (length `D` — passed as `x` to each
/// field). `dot_i_t` is the interpolant time-derivative `İ_t` (length `D`).
///
/// Reuses `scratch.b_out_i` and `scratch.b_out_j` to avoid per-iteration
/// allocation. The `i == j` shortcut avoids re-evaluating `b_i`.
pub fn accumulate_pair_into<F, const P: usize, const D: usize>(
    fields: &[F; P],
    i_t: &[f32],
    dot_i_t: &[f32],
    scratch: &mut EnsembleFitScratch<P, D>,
) where
    F: VelocityField<D>,
{
    for (i, field_i) in fields.iter().enumerate().take(P) {
        field_i.eval_into(i_t, &mut scratch.b_out_i);
        let row_i = i * P;
        for (j, field_j) in fields.iter().enumerate().take(P).skip(i) {
            // Upper triangle (i ≤ j).
            let dot = if i == j {
                // b_i already in b_out_i; dot with itself (no re-eval).
                simd_dot_f32(&scratch.b_out_i, &scratch.b_out_i, D)
            } else {
                // Evaluate b_j into b_out_j; dot with b_out_i.
                field_j.eval_into(i_t, &mut scratch.b_out_j);
                simd_dot_f32(&scratch.b_out_i, &scratch.b_out_j, D)
            };
            scratch.gram[row_i + j] += dot;
            if i != j {
                // Mirror to lower triangle — K is symmetric.
                scratch.gram[j * P + i] += dot;
            }
        }
        // RHS: r[i] += b_i(I_t) · İ_t.
        scratch.rhs[i] += simd_dot_f32(&scratch.b_out_i, dot_i_t, D);
    }
}

// ── Shared post-accumulate solve (Issue 044 Finding A) ──────────────────────

/// Normalize Gram + RHS by `N`, add ridge `λI`, solve `(K + λI) η = r` via
/// Cholesky. Writes the `P`-dim solution into `eta`.
///
/// Shared by [`VelocityFieldEnsemble::fit_into`] and
/// [`HeterogeneousEnsemble::fit_into`] — the per-pair accumulation differs
/// (the heterogeneous variant transports each field to `D` first), but the
/// solve-after-accumulate math is byte-for-byte identical. Extracted as a free
/// function so future numerical-methods improvements (adaptive `λ`, LOO-CV,
/// rank-deficient fallback, Tikhonov `Γ ≠ I`) land in one place.
///
/// # Allocation discipline
///
/// Zero. Reads/writes caller-provided slices in place. `gram_reg` is scratch
/// for the regularized copy; `chol` and `z_solve` are solver scratch.
///
/// # Panics
///
/// Panics (via `ridge_solve_direct_f32`) if `gram_reg.len() < P*P`,
/// `chol.len() < P*P`, `rhs.len() < P`, or `n == 0`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn solve_ridge_eta_into<const P: usize>(
    eta: &mut [f32; P],
    gram: &mut [f32],
    rhs: &mut [f32; P],
    gram_reg: &mut [f32],
    chol: &mut [f32],
    z_solve: &mut [f32; P],
    n: usize,
    lambda: f32,
) {
    let inv_n = 1.0 / (n as f32);
    for g in gram.iter_mut() {
        *g *= inv_n;
    }
    for r in rhs.iter_mut() {
        *r *= inv_n;
    }

    gram_reg[..P * P].copy_from_slice(&gram[..P * P]);
    for i in 0..P {
        gram_reg[i * P + i] += lambda;
    }

    ridge_solve_direct_f32(eta, chol, z_solve, gram_reg, rhs, P, 1);
}

// ── Fit ───────────────────────────────────────────────────────────────────

impl<F, const P: usize, const D: usize> VelocityFieldEnsemble<F, P, D>
where
    F: VelocityField<D>,
{
    /// Solve the regression-optimal weights `η` from `N` data pairs.
    ///
    /// Accumulates the Gram and RHS from all pairs, normalizes by `N`, adds
    /// ridge regularization `λI`, and solves `(K + λI) η = r` via
    /// [`ridge_solve_direct_f32`] (the same Cholesky path KARC uses).
    ///
    /// # Arguments
    ///
    /// - `i_t_samples`: `N` interpolant samples, each length `D`. These are
    ///   the `I_t^n` values.
    /// - `dot_i_t_samples`: `N` interpolant time-derivatives, each length
    ///   `D`. These are the `İ_t^n` targets. Must be the same length as
    ///   `i_t_samples`.
    /// - `lambda`: ridge regularization `λ > 0`. Required for positive-
    ///   definiteness when fields are collinear or duplicated; also stabilizes
    ///   the Cholesky at small matrix scales. Start at `1e-4` and tune.
    /// - `scratch`: caller-provided fit scratch (allocate once, reuse).
    ///
    /// # Panics
    ///
    /// Panics if `i_t_samples.len() != dot_i_t_samples.len()`, if any sample
    /// is not length `D`, or if `lambda <= 0.0`.
    pub fn fit_into(
        &mut self,
        i_t_samples: &[&[f32]],
        dot_i_t_samples: &[&[f32]],
        lambda: f32,
        scratch: &mut EnsembleFitScratch<P, D>,
    ) {
        assert!(
            i_t_samples.len() == dot_i_t_samples.len(),
            "i_t_samples and dot_i_t_samples must have the same length"
        );
        assert!(lambda > 0.0, "lambda must be > 0 for ridge regularization");
        let n = i_t_samples.len();
        assert!(n > 0, "need at least one data pair to fit");

        // Reset accumulation buffers.
        scratch.clear();

        // Accumulate all pairs into gram and rhs.
        for k in 0..n {
            let i_t = i_t_samples[k];
            let dot_i_t = dot_i_t_samples[k];
            assert_eq!(
                i_t.len(),
                D,
                "i_t_samples[{}] length {} != D = {}",
                k,
                i_t.len(),
                D
            );
            assert_eq!(
                dot_i_t.len(),
                D,
                "dot_i_t_samples[{}] length {} != D = {}",
                k,
                dot_i_t.len(),
                D
            );
            accumulate_pair_into(&self.fields, i_t, dot_i_t, scratch);
        }

        // Normalize by N, build K + λI, solve (K + λI) η = r via Cholesky.
        // Shared with HeterogeneousEnsemble::fit_into via solve_ridge_eta_into
        // (Issue 044 Finding A). The accumulation differs; the solve does not.
        solve_ridge_eta_into(
            &mut self.eta,
            &mut scratch.gram,
            &mut scratch.rhs,
            &mut scratch.gram_reg,
            &mut scratch.chol,
            &mut scratch.z_solve,
            n,
            lambda,
        );
    }
}

// ── Eval ──────────────────────────────────────────────────────────────────

impl<F, const P: usize, const D: usize> VelocityFieldEnsemble<F, P, D>
where
    F: VelocityField<D>,
{
    /// Evaluate the combined drift `b̂(x) = Σ_i η_i · b_i(x)` at state `x`.
    ///
    /// Writes the `D`-dim result into `out`. `scratch_b` is caller-provided
    /// per-field eval scratch (length `D`) — pass `&mut fit_scratch.b_out_i`
    /// or a dedicated buffer. Zero allocation.
    ///
    /// The caller is responsible for having called [`Self::fit_into`] first;
    /// otherwise `η` is the uniform `1/P` average (the no-fit fallback).
    #[inline]
    pub fn eval_into(&self, x: &[f32], out: &mut [f32; D], scratch_b: &mut [f32; D]) {
        // out = 0, then out += η_i · b_i(x) for each i.
        // `fill` lowers to `memset`; this is a documented ~21 ns/call path.
        out.fill(0.0);
        for i in 0..P {
            self.fields[i].eval_into(x, scratch_b);
            let eta_i = self.eta[i];
            for k in 0..D {
                out[k] += eta_i * scratch_b[k];
            }
        }
    }

    /// Evaluate the combined drift for `N` states in a tight loop.
    ///
    /// `x_batch` is `N` input states (each length `D`); `out_batch` is `N`
    /// output buffers (each a `&mut [f32; D]`). `scratch_b` is reused across
    /// all `N` evals (zero allocation across the batch).
    ///
    /// Used for hot-path inference (e.g., 1000 ticks of an NPC's HLA update).
    /// The caller owns all allocation; this function does not allocate.
    pub fn eval_batch_into(
        &self,
        x_batch: &[&[f32]],
        out_batch: &mut [&mut [f32; D]],
        scratch_b: &mut [f32; D],
    ) {
        assert_eq!(
            x_batch.len(),
            out_batch.len(),
            "x_batch and out_batch must have the same length"
        );
        for n in 0..x_batch.len() {
            self.eval_into(x_batch[n], out_batch[n], scratch_b);
        }
    }
}

// ── Schedule + optimal-diffusion integrator ───────────────────────────────

/// Interpolant schedule for the optimal-diffusion SDE integrator
/// (paper §2.4, Proposition 2.2).
///
/// The schedule defines `α_t` and `β_t` such that the interpolant
/// `I_t = α_t · x_0 + β_t · x_1` (where `x_0` is Gaussian noise, `x_1` is
/// the data). The derived quantity `γ_t = α_t · β̇_t − α̇_t · β_t` enters
/// the optimal diffusion coefficient `D*_t = α_t · γ_t / β_t`.
///
/// Both shipped schedules have **constant** `γ_t` (a nice analytical
/// property — see the derivation in each variant's doc).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Schedule {
    /// Linear schedule: `α_t = 1 − t`, `β_t = t`. Derivative `γ_t = 1`
    /// (constant). This is the simplest schedule; `D*_t = (1 − t) / t`.
    Linear,

    /// Trigonometric schedule: `α_t = cos(πt/2)`, `β_t = sin(πt/2)`.
    /// Derivative `γ_t = π/2` (constant). Smoother endpoint behavior than
    /// linear; `D*_t = (π/2) · cot(πt/2)`.
    Trigonometric,
}

impl Schedule {
    /// Evaluate `(α_t, β_t)` at time `t ∈ [0, 1]`.
    ///
    /// At `t = 0`: `(α, β) = (1, 0)` (pure noise). At `t = 1`: `(α, β) = (0, 1)`
    /// (pure data). The caller must handle the singular `β = 0` endpoint
    /// (see [`stochastic_interpolant_step_into`] — it takes `β_t` and
    /// `β_{t+h}` directly, so the caller can clip away from `t = 0`).
    #[inline]
    pub fn alpha_beta(&self, t: f32) -> (f32, f32) {
        match self {
            Self::Linear => (1.0 - t, t),
            Self::Trigonometric => {
                let theta = std::f32::consts::FRAC_PI_2 * t;
                (theta.cos(), theta.sin())
            }
        }
    }

    /// Evaluate `γ_t = α_t · β̇_t − α̇_t · β_t` at time `t ∈ [0, 1]`.
    ///
    /// For both shipped schedules this is a positive constant (`Linear → 1`,
    /// `Trigonometric → π/2`). The `t` parameter is unused for the shipped
    /// schedules but kept in the signature for future variable-γ schedules.
    #[inline]
    pub fn gamma(&self, _t: f32) -> f32 {
        match self {
            Self::Linear => 1.0,
            Self::Trigonometric => std::f32::consts::FRAC_PI_2,
        }
    }

    /// Evaluate the optimal diffusion coefficient `D*_t = α_t · γ_t / β_t`.
    ///
    /// Returns `+∞` when `β_t = 0` (the `t = 0` endpoint) — the caller must
    /// clip `t` away from 0 before calling, or handle the infinity. This
    /// matches the paper: `D*_t → ∞` as `t → 0` (full Gaussian noise regime),
    /// `D*_t → 0` as `t → 1` (pure ODE transport).
    #[inline]
    pub fn optimal_diffusion(&self, t: f32) -> f32 {
        let (alpha, beta) = self.alpha_beta(t);
        let gamma = self.gamma(t);
        alpha * gamma / beta
    }
}

/// Step the optimal-diffusion SDE from `x_t` to `x_{t+h}` using a precomputed
/// drift `b̂_t(x_t)`.
///
/// Implements paper eq. 14 (Algorithm 1) with `D*_t = α_t γ_t / β_t`:
///
/// ```text
/// X_{t+h} = (β_t / β_{t+h}) · X_t
///         + h · (1 + β_t / β_{t+h}) · b̂_t(X_t)
///         + sqrt(h · (α_t β_t γ_t + α_{t+h} β_{t+h} γ_{t+h}) / β_{t+h}) · g_t
/// ```
///
/// where `g_t ~ N(0, I_D)` is a standard-normal D-dim increment.
///
/// **Not coupled to [`VelocityFieldEnsemble`]** — takes any precomputed drift
/// slice. This lets future non-ensemble primitives (e.g., KARC + `D*_t`) also
/// use the integrator.
///
/// # Arguments
///
/// - `x_t`: current state, length `D`.
/// - `x_out`: next state `x_{t+h}`, length `D`. May alias `x_t`.
/// - `schedule`: the interpolant schedule (defines α, β, γ).
/// - `t`: current time `∈ [0, 1)`.
/// - `h`: step size `> 0` (must satisfy `t + h ≤ 1`).
/// - `drift_at_t`: `b̂_t(x_t)`, precomputed by the caller via
///   [`VelocityFieldEnsemble::eval_into`] or any other drift source.
/// - `sample_normal`: a closure `FnMut() -> f32` returning i.i.d. standard-
///   normal samples. Called `D` times per step (once per coordinate). The
///   caller controls the RNG (deterministic seeds, reproducibility).
///
/// # Panics
///
/// Panics if `h <= 0.0`, if `t < 0.0`, or if `x_t.len() != x_out.len()` (or
/// either differs from `drift_at_t.len()`).
pub fn stochastic_interpolant_step_into<R>(
    x_t: &[f32],
    x_out: &mut [f32],
    schedule: Schedule,
    t: f32,
    h: f32,
    drift_at_t: &[f32],
    sample_normal: &mut R,
) where
    R: FnMut() -> f32,
{
    assert!(h > 0.0, "step size h must be > 0");
    assert!(t >= 0.0, "time t must be >= 0");
    let d = x_t.len();
    assert_eq!(x_out.len(), d, "x_out length must match x_t");
    assert_eq!(drift_at_t.len(), d, "drift_at_t length must match x_t");

    let t_plus_h = t + h;
    let (alpha_t, beta_t) = schedule.alpha_beta(t);
    let (alpha_t_plus_h, beta_t_plus_h) = schedule.alpha_beta(t_plus_h);
    let gamma_t = schedule.gamma(t);
    let gamma_t_plus_h = schedule.gamma(t_plus_h);

    let beta_ratio = beta_t / beta_t_plus_h;
    let drift_coeff = h * (1.0 + beta_ratio);
    // Noise variance: h · (α_t β_t γ_t + α_{t+h} β_{t+h} γ_{t+h}) / β_{t+h}.
    // Under the optimal D*_t schedule (Proposition 2.2), this is the path-KL-
    // optimal diffusion contribution. sqrt for the std-dev of the Brownian.
    let noise_variance = h
        * (alpha_t * beta_t * gamma_t + alpha_t_plus_h * beta_t_plus_h * gamma_t_plus_h)
        / beta_t_plus_h;
    let noise_coeff = noise_variance.max(0.0).sqrt();

    for k in 0..d {
        let g = sample_normal();
        x_out[k] = beta_ratio * x_t[k] + drift_coeff * drift_at_t[k] + noise_coeff * g;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Three constant velocity fields (drift does not depend on `x`).
    /// Used by `test_fit_recovers_known_eta` and `test_eval_is_linear_combination`.
    ///
    /// Each field writes a fixed vector into `out`, ignoring `x`.
    fn make_constant_field<const D: usize>(
        id: u64,
        constant: [f32; D],
    ) -> ClosureField<D, impl Fn(&[f32], &mut [f32; D])> {
        ClosureField::new(id, move |_x: &[f32], out: &mut [f32; D]| {
            *out = constant;
        })
    }

    #[test]
    fn test_fit_recovers_known_eta() {
        // 3 constant velocity fields in R^4. η* = [0.5, 0.3, 0.2].
        // The fit should recover η* exactly (no noise).
        const P: usize = 3;
        const D: usize = 4;

        let b1 = make_constant_field(1, [1.0, 0.0, 0.0, 0.0]);
        let b2 = make_constant_field(2, [0.0, 1.0, 0.0, 0.0]);
        let b3 = make_constant_field(3, [0.0, 0.0, 1.0, 0.0]);

        let mut ensemble = VelocityFieldEnsemble::<_, P, D>::new([b1, b2, b3]);
        let mut scratch = EnsembleFitScratch::<P, D>::new();

        // True η*. The combined drift is İ_t = 0.5·b1 + 0.3·b2 + 0.2·b3.
        let eta_star = [0.5f32, 0.3, 0.2];
        // İ_t is the same for all pairs (constant fields → constant target).
        let dot_i_t = [eta_star[0] * 1.0, eta_star[1] * 1.0, eta_star[2] * 1.0, 0.0];
        // I_t can be anything (fields ignore it); use zeros.
        let i_t = [0.0f32; D];

        // N = 50 pairs (all identical — the math still works; normalization
        // by N cancels for constant fields).
        let n = 50;
        let i_t_samples: Vec<&[f32]> = (0..n).map(|_| &i_t[..]).collect();
        let dot_i_t_samples: Vec<&[f32]> = (0..n).map(|_| &dot_i_t[..]).collect();

        ensemble.fit_into(&i_t_samples, &dot_i_t_samples, 1e-6, &mut scratch);

        let eta = ensemble.eta();
        let max_err = eta
            .iter()
            .zip(eta_star.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "η recovery failed: η = {eta:?}, η* = {eta_star:?}, max_err = {max_err}"
        );
    }

    #[test]
    fn test_eval_is_linear_combination() {
        // With known η, eval_into must produce exactly Σ η_i b_i(x).
        const P: usize = 2;
        const D: usize = 3;

        // b_1(x) = [x[0], 0, x[2]]   (linear in x)
        // b_2(x) = [0, x[1], x[1]]   (linear in x)
        //
        // Use named `fn` items (not closures) so both fields share the same
        // `fn(&[f32], &mut [f32; D])` type. Inline closures each get a unique
        // anonymous type, which breaks the `[F; P]` array requirement.
        fn b1_fn(x: &[f32], out: &mut [f32; 3]) {
            out[0] = x[0];
            out[1] = 0.0;
            out[2] = x[2];
        }
        fn b2_fn(x: &[f32], out: &mut [f32; 3]) {
            out[0] = 0.0;
            out[1] = x[1];
            out[2] = x[1];
        }

        type FieldFn = fn(&[f32], &mut [f32; D]);
        let b1: ClosureField<D, FieldFn> = ClosureField::new(1, b1_fn);
        let b2: ClosureField<D, FieldFn> = ClosureField::new(2, b2_fn);

        let mut ensemble = VelocityFieldEnsemble::<_, P, D>::new([b1, b2]);
        // Manually set η = [0.7, -0.4] (note: negative weight is allowed —
        // this is a signed combination, not a probabilistic mixture).
        ensemble.eta = [0.7, -0.4];

        let x = [1.5f32, 2.0, 0.5];
        let mut out = [0.0f32; D];
        let mut scratch_b = [0.0f32; D];
        ensemble.eval_into(&x, &mut out, &mut scratch_b);

        // Expected: 0.7·b_1(x) + (-0.4)·b_2(x)
        let expected = [
            0.7 * 1.5,
            -0.4 * 2.0,
            0.7 * 0.5 + (-0.4) * 2.0, // 0.35 - 0.8 = -0.45
        ];
        for k in 0..D {
            assert!(
                (out[k] - expected[k]).abs() < 1e-6,
                "eval_into[{}] = {} != expected {}",
                k,
                out[k],
                expected[k]
            );
        }
    }

    #[test]
    fn test_gram_symmetric() {
        // After accumulate_pair_into, gram[i*P+j] == gram[j*P+i].
        const P: usize = 3;
        const D: usize = 4;

        let b1 = make_constant_field(1, [1.0, 2.0, 0.0, -1.0]);
        let b2 = make_constant_field(2, [0.5, -1.0, 2.0, 0.0]);
        let b3 = make_constant_field(3, [0.0, 1.0, 1.0, 1.0]);

        let fields = [b1, b2, b3];
        let mut scratch = EnsembleFitScratch::<P, D>::new();

        let i_t = [0.5f32, -0.3, 1.2, 0.8]; // fields ignore it, but pass it anyway
        let dot_i_t = [0.1f32, 0.2, 0.3, 0.4];

        accumulate_pair_into(&fields, &i_t, &dot_i_t, &mut scratch);

        for i in 0..P {
            for j in 0..P {
                let ij = scratch.gram[i * P + j];
                let ji = scratch.gram[j * P + i];
                assert!(
                    (ij - ji).abs() < 1e-6,
                    "gram asymmetric: gram[{}*{}+{}={}] = {} != gram[{}*{}+{}={}] = {}",
                    i,
                    P,
                    j,
                    i * P + j,
                    ij,
                    j,
                    P,
                    i,
                    j * P + i,
                    ji
                );
            }
        }
    }

    #[test]
    fn test_chosen_lambda_stabilizes_ill_conditioned_gram() {
        // Duplicate velocity fields → singular Gram. With λ > 0, the solve
        // must produce finite (non-NaN, non-inf) η.
        const P: usize = 2;
        const D: usize = 2;

        // Two identical fields (field_id differs but the drift is identical).
        let b1 = make_constant_field(1, [1.0, 0.0]);
        let b2 = make_constant_field(2, [1.0, 0.0]); // duplicate drift

        let mut ensemble = VelocityFieldEnsemble::<_, P, D>::new([b1, b2]);
        let mut scratch = EnsembleFitScratch::<P, D>::new();

        let i_t = [0.0f32; D];
        let dot_i_t = [1.0f32, 0.0]; // target along x[0]
        let i_t_samples: Vec<&[f32]> = vec![&i_t[..]];
        let dot_i_t_samples: Vec<&[f32]> = vec![&dot_i_t[..]];

        // λ = 1e-3 should stabilize the singular Gram.
        ensemble.fit_into(&i_t_samples, &dot_i_t_samples, 1e-3, &mut scratch);

        for (i, &eta_i) in ensemble.eta().iter().enumerate() {
            assert!(
                eta_i.is_finite(),
                "η[{i}] = {eta_i} is not finite (λ failed to stabilize)"
            );
        }
        // The ridge solution splits the target equally between the duplicates
        // (symmetric). Check approximate equality.
        let (e0, e1) = (ensemble.eta()[0], ensemble.eta()[1]);
        assert!(
            (e0 - e1).abs() < 1e-3,
            "duplicate fields should get equal weights: η = [{e0}, {e1}]"
        );
    }

    #[test]
    fn test_eval_batch_reuses_scratch() {
        // eval_batch_into must produce the same output as N individual
        // eval_into calls. Verifies the scratch is correctly reset per batch
        // element (out is zeroed at the start of each eval_into).
        const P: usize = 2;
        const D: usize = 3;

        let b1 = make_constant_field(1, [1.0, 0.0, 0.0]);
        let b2 = make_constant_field(2, [0.0, 1.0, 0.0]);

        let mut ensemble = VelocityFieldEnsemble::<_, P, D>::new([b1, b2]);
        ensemble.eta = [0.5, 0.5]; // equal-weight average

        // 5 states (different x values, though fields ignore them here).
        let x_vals: [[f32; D]; 5] = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
        ];
        let x_batch_refs: Vec<&[f32]> = x_vals.iter().map(|v| &v[..]).collect();
        let mut out_batch_arr: [[f32; D]; 5] = [[0.0; D]; 5];
        let mut out_batch_refs: Vec<&mut [f32; D]> = out_batch_arr.iter_mut().collect();
        let mut scratch_b = [0.0f32; D];

        ensemble.eval_batch_into(&x_batch_refs, &mut out_batch_refs, &mut scratch_b);

        // Each output should be 0.5·b_1 + 0.5·b_2 = [0.5, 0.5, 0.0].
        for (n, out) in out_batch_arr.iter().enumerate() {
            assert!(
                (out[0] - 0.5).abs() < 1e-6,
                "batch[{}][0] = {} != 0.5",
                n,
                out[0]
            );
            assert!(
                (out[1] - 0.5).abs() < 1e-6,
                "batch[{}][1] = {} != 0.5",
                n,
                out[1]
            );
            assert!(out[2].abs() < 1e-6, "batch[{}][2] = {} != 0.0", n, out[2]);
        }
    }

    #[test]
    fn test_schedule_linear() {
        let s = Schedule::Linear;
        // t=0: (α, β) = (1, 0).
        let (a0, b0) = s.alpha_beta(0.0);
        assert!((a0 - 1.0).abs() < 1e-6);
        assert!(b0.abs() < 1e-6);
        // t=1: (α, β) = (0, 1).
        let (a1, b1) = s.alpha_beta(1.0);
        assert!(a1.abs() < 1e-6);
        assert!((b1 - 1.0).abs() < 1e-6);
        // t=0.5: (α, β) = (0.5, 0.5).
        let (a, b) = s.alpha_beta(0.5);
        assert!((a - 0.5).abs() < 1e-6);
        assert!((b - 0.5).abs() < 1e-6);
        // γ = 1 (constant).
        assert!((s.gamma(0.0) - 1.0).abs() < 1e-6);
        assert!((s.gamma(0.5) - 1.0).abs() < 1e-6);
        assert!((s.gamma(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_trigonometric() {
        let s = Schedule::Trigonometric;
        // t=0: (α, β) = (1, 0).
        let (a0, b0) = s.alpha_beta(0.0);
        assert!((a0 - 1.0).abs() < 1e-6);
        assert!(b0.abs() < 1e-6);
        // t=1: (α, β) = (0, 1).
        let (a1, b1) = s.alpha_beta(1.0);
        assert!(a1.abs() < 1e-6);
        assert!((b1 - 1.0).abs() < 1e-6);
        // γ = π/2 (constant).
        let pi_2 = std::f32::consts::FRAC_PI_2;
        assert!((s.gamma(0.5) - pi_2).abs() < 1e-6);
    }

    #[test]
    fn test_stochastic_interpolant_step_no_drift_no_noise() {
        // With zero drift and zero noise (sample_normal returns 0), the step
        // is a pure transport: x_{t+h} = (β_t / β_{t+h}) · x_t.
        const D: usize = 3;
        let x_t = [1.0f32, 2.0, 3.0];
        let mut x_out = [0.0f32; D];
        let drift = [0.0f32; D];
        let schedule = Schedule::Linear;
        let t = 0.5;
        let h = 0.1;

        stochastic_interpolant_step_into(
            &x_t,
            &mut x_out,
            schedule,
            t,
            h,
            &drift,
            &mut || 0.0, // zero noise
        );

        let (_, beta_t) = schedule.alpha_beta(t);
        let (_, beta_t_plus_h) = schedule.alpha_beta(t + h);
        let beta_ratio = beta_t / beta_t_plus_h;
        for k in 0..D {
            assert!(
                (x_out[k] - beta_ratio * x_t[k]).abs() < 1e-6,
                "x_out[{}] = {} != {} (pure transport)",
                k,
                x_out[k],
                beta_ratio * x_t[k]
            );
        }
    }

    #[test]
    fn test_stochastic_interpolant_step_with_drift() {
        // With non-zero drift and zero noise, verify the drift contribution.
        const D: usize = 2;
        let x_t = [1.0f32, 1.0];
        let mut x_out = [0.0f32; D];
        let drift = [0.5f32, -0.5];
        let schedule = Schedule::Linear;
        let t = 0.3;
        let h = 0.2;

        stochastic_interpolant_step_into(
            &x_t,
            &mut x_out,
            schedule,
            t,
            h,
            &drift,
            &mut || 0.0, // zero noise
        );

        let (_, beta_t) = schedule.alpha_beta(t);
        let (_, beta_t_plus_h) = schedule.alpha_beta(t + h);
        let beta_ratio = beta_t / beta_t_plus_h;
        let drift_coeff = h * (1.0 + beta_ratio);
        for k in 0..D {
            let expected = beta_ratio * x_t[k] + drift_coeff * drift[k];
            assert!(
                (x_out[k] - expected).abs() < 1e-6,
                "x_out[{}] = {} != {} (with drift)",
                k,
                x_out[k],
                expected
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// PHASE 4: Heterogeneous-D velocity fields via Cross-Resolution transport.
// Each field has its own native dim `d_i`; all are projected to a common `D`
// via `CrossResolutionBases` (Plan 310), then ensemble-combined using the
// same regression-optimal ridge solve as the homogeneous path.
// ─────────────────────────────────────────────────────────────────────────

/// A frozen forward model whose native output dim is runtime-known (not
/// const-generic).
///
/// This is the heterogeneous-D counterpart to [`VelocityField<D>`]. Use it
/// when the P ensemble members have different output dims `d_i` — e.g., a
/// 16-dim plasma-tier shard alongside a 64-dim cold-tier shard. Each field is
/// paired with a [`CrossResolutionBases`] that transports its native output
/// to the ensemble's common `D`.
///
/// Implementors: wrap any closure-based field in [`HeterogeneousClosureField`]
/// or implement this trait directly for concrete field types.
///
/// # Object safety
///
/// This trait is object-safe. [`HeterogeneousEntry`] stores
/// `Box<dyn HeterogeneousVelocityField>` so the P entries can have different
/// concrete types / native dims. Users who want static dispatch should
/// implement the trait for an enum wrapping their concrete field types.
#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
pub trait HeterogeneousVelocityField: Send + Sync {
    /// Evaluate the field at state `x`, writing the native-dim output into
    /// `out_native`. Caller-provided; length must equal `self.native_dim()`.
    /// Zero-allocation contract: implementor MUST NOT allocate.
    fn eval_native_into(&self, x: &[f32], out_native: &mut [f32]);

    /// Native output dimension `d_i`. Must match the `d_src` of the paired
    /// [`CrossResolutionBases`].
    fn native_dim(&self) -> usize;

    /// Identifier for BLAKE3 commitment of this field's frozen weights
    /// (mirrors [`VelocityField::field_id`]).
    fn field_id(&self) -> u64;
}

/// Closure wrapper for [`HeterogeneousVelocityField`] — the heterogeneous
/// analog of [`ClosureField`].
///
/// Stores the closure inline (no `Box` around the closure itself — but the
/// field is typically stored as `Box<dyn HeterogeneousVelocityField>` inside
/// [`HeterogeneousEntry`], so there's still one heap allocation per entry for
/// the trait object).
#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
pub struct HeterogeneousClosureField<F>
where
    F: Fn(&[f32], &mut [f32]) + Send + Sync,
{
    id: u64,
    native_dim: usize,
    f: F,
}

#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
impl<F> HeterogeneousClosureField<F>
where
    F: Fn(&[f32], &mut [f32]) + Send + Sync,
{
    /// Construct from a closure, a `field_id`, and the native output dim.
    #[inline]
    pub fn new(id: u64, native_dim: usize, f: F) -> Self {
        Self { id, native_dim, f }
    }
}

#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
impl<F> HeterogeneousVelocityField for HeterogeneousClosureField<F>
where
    F: Fn(&[f32], &mut [f32]) + Send + Sync,
{
    #[inline]
    fn eval_native_into(&self, x: &[f32], out_native: &mut [f32]) {
        (self.f)(x, out_native);
    }

    #[inline]
    fn native_dim(&self) -> usize {
        self.native_dim
    }

    #[inline]
    fn field_id(&self) -> u64 {
        self.id
    }
}

/// One heterogeneous field entry: the field plus its native→D transport bases.
///
/// This is the "field-library format" extension (Plan 376 T4.2): each entry
/// is now a `(field, transport)` pair rather than just a field. The transport
/// bases are frozen BLAKE3-committed artifacts (Plan 310), so the entry is
/// fully content-addressable.
///
/// `bases.d_src` must equal `field.native_dim()`; `bases.d_dst` must equal
/// the ensemble's `D`. Verified at construction.
#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
pub struct HeterogeneousEntry {
    /// The heterogeneous field (boxed for trait-object dispatch).
    pub field: Box<dyn HeterogeneousVelocityField>,
    /// Cross-resolution bases transporting `d_src = field.native_dim()` to
    /// `d_dst = D` (the ensemble's common dim).
    pub bases: CrossResolutionBases,
}

#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
impl HeterogeneousEntry {
    /// Construct an entry, verifying that `bases.d_src == field.native_dim()`.
    ///
    /// `bases.d_dst` is NOT checked here — it's checked by
    /// [`HeterogeneousEnsemble::new`] against the ensemble's `D`.
    pub fn new(field: Box<dyn HeterogeneousVelocityField>, bases: CrossResolutionBases) -> Self {
        assert_eq!(
            bases.d_src,
            field.native_dim(),
            "HeterogeneousEntry: bases.d_src ({}) != field.native_dim () ({})",
            bases.d_src,
            field.native_dim()
        );
        Self { field, bases }
    }
}

/// Ensemble of `P` heterogeneous velocity fields, all transported to common
/// dim `D`, then combined via regression-optimal weights `η ∈ R^P`.
///
/// Mirrors [`VelocityFieldEnsemble`] but with per-field native dims. The fit
/// and eval logic is the same regression-optimal ridge solve; the only
/// difference is the per-field transport step (native eval → cross-resolution
/// transport → D-dim output) before accumulation into the Gram.
///
/// # Zero-allocation contract
///
/// Allocation happens exactly once at construction (in
/// [`HeterogeneousFitScratch::new`]). The hot path (`fit_into`, `eval_into`)
/// performs zero allocations — verified by `test_heterogeneous_zero_alloc`.
#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
pub struct HeterogeneousEnsemble<const P: usize, const D: usize> {
    /// The P heterogeneous entries (field + transport bases).
    pub entries: [HeterogeneousEntry; P],
    /// Solved combination weights. Initialized to uniform `1/P`; overwritten
    /// by [`HeterogeneousEnsemble::fit_into`].
    pub eta: [f32; P],
}

#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
impl<const P: usize, const D: usize> HeterogeneousEnsemble<P, D> {
    /// Construct from `P` entries. All entries' `bases.d_dst` must equal `D`.
    pub fn new(entries: [HeterogeneousEntry; P]) -> Self {
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(
                entry.bases.d_dst, D,
                "HeterogeneousEnsemble::new: entries[{}].bases.d_dst ({}) != D ({})",
                i, entry.bases.d_dst, D
            );
        }
        // Splat directly instead of zero-init-then-overwrite.
        let eta = [1.0 / (P as f32); P];
        Self { entries, eta }
    }

    /// Returns the solved weight vector `η`. Length `P`.
    #[inline]
    pub fn eta(&self) -> &[f32; P] {
        &self.eta
    }

    /// Solve regression-optimal `η` from `N` data pairs.
    ///
    /// Same math as [`VelocityFieldEnsemble::fit_into`] — accumulate the P×P
    /// Gram and P-dim RHS, normalize by N, add ridge `λI`, solve via
    /// [`ridge_solve_direct_f32`]. The only difference: each field's output is
    /// transported to D before the dot products.
    ///
    /// See [`VelocityFieldEnsemble::fit_into`] for argument semantics — they
    /// are identical (the heterogeneous path just adds transport).
    pub fn fit_into(
        &mut self,
        i_t_samples: &[&[f32]],
        dot_i_t_samples: &[&[f32]],
        lambda: f32,
        scratch: &mut HeterogeneousFitScratch<P, D>,
    ) {
        assert!(
            i_t_samples.len() == dot_i_t_samples.len(),
            "i_t_samples and dot_i_t_samples must have the same length"
        );
        assert!(lambda > 0.0, "lambda must be > 0 for ridge regularization");
        let n = i_t_samples.len();
        assert!(n > 0, "need at least one data pair to fit");

        scratch.clear();

        for k in 0..n {
            let i_t = i_t_samples[k];
            let dot_i_t = dot_i_t_samples[k];
            assert_eq!(
                i_t.len(),
                D,
                "i_t_samples[{}] length {} != D = {}",
                k,
                i_t.len(),
                D
            );
            assert_eq!(
                dot_i_t.len(),
                D,
                "dot_i_t_samples[{}] length {} != D = {}",
                k,
                dot_i_t.len(),
                D
            );
            self.accumulate_pair_heterogeneous_into(i_t, dot_i_t, scratch);
        }

        // Normalize by N, build K + λI, solve (K + λI) η = r via Cholesky.
        // Shared with VelocityFieldEnsemble::fit_into via solve_ridge_eta_into
        // (Issue 044 Finding A). The accumulation differs (transport step);
        // the solve does not.
        solve_ridge_eta_into(
            &mut self.eta,
            &mut scratch.gram,
            &mut scratch.rhs,
            &mut scratch.gram_reg,
            &mut scratch.chol,
            &mut scratch.z_solve,
            n,
            lambda,
        );
    }

    /// Evaluate the combined drift `b̂(x) = Σ_i η_i · transport(b_i(x))` at `x`.
    ///
    /// Each field is evaluated at its native dim, transported to D, then
    /// scaled by `η_i` and summed. Zero-allocation given caller-provided
    /// scratch.
    ///
    /// `scratch_b_at_d` is a D-dim buffer reused across fields (per-field
    /// output at D, before scaling into `out`).
    #[inline]
    pub fn eval_into(
        &self,
        x: &[f32],
        out: &mut [f32; D],
        scratch: &mut HeterogeneousFitScratch<P, D>,
    ) {
        out.fill(0.0);
        for i in 0..P {
            // Hoist the entry lookup: it was re-indexed four times per field.
            let entry = &self.entries[i];
            // Size the native buffer to this field's native_dim.
            let d_i = entry.field.native_dim();
            let k_i = entry.bases.k;
            let native_buf_i = &mut scratch.native_buf_i[..d_i];
            let spectral = &mut scratch.spectral_buf[..k_i];
            entry.field.eval_native_into(x, native_buf_i);
            // Inline transport: project to spectral, reconstruct at D.
            project_to_spectral_into(native_buf_i, &entry.bases, spectral);
            reconstruct_from_spectral_into(spectral, &entry.bases, &mut scratch.b_at_d_i);
            let eta_i = self.eta[i];
            // `out` and `b_at_d_i` are both `[f32; D]`, so the `.take(D)` that
            // used to be here was a no-op that hid the length from LLVM.
            for (out_k, b_k) in out.iter_mut().zip(scratch.b_at_d_i.iter()) {
                *out_k += eta_i * b_k;
            }
        }
    }

    /// Accumulate one pair into Gram + RHS, with per-field transport.
    ///
    /// For each `(i, j)` with `i ≤ j`: transport `b_i(I_t)` and `b_j(I_t)` to
    /// D, then dot. For each `i`: transport `b_i(I_t)` to D, dot with `İ_t`.
    #[inline]
    fn accumulate_pair_heterogeneous_into(
        &self,
        i_t: &[f32],
        dot_i_t: &[f32],
        scratch: &mut HeterogeneousFitScratch<P, D>,
    ) {
        for i in 0..P {
            // Size the native buffer to this field's native_dim. The scratch
            // buffer is max-sized across entries; we borrow the prefix.
            // Hoist the entry lookup out of the O(P²) loop nest.
            let entry_i = &self.entries[i];
            let d_i = entry_i.field.native_dim();
            let k_i = entry_i.bases.k;
            let native_buf_i = &mut scratch.native_buf_i[..d_i];
            let spectral = &mut scratch.spectral_buf[..k_i];
            entry_i.field.eval_native_into(i_t, native_buf_i);
            // Inline transport: project to spectral, reconstruct at D.
            project_to_spectral_into(native_buf_i, &entry_i.bases, spectral);
            reconstruct_from_spectral_into(spectral, &entry_i.bases, &mut scratch.b_at_d_i);
            for j in i..P {
                let dot = if i == j {
                    simd_dot_f32(&scratch.b_at_d_i, &scratch.b_at_d_i, D)
                } else {
                    let entry_j = &self.entries[j];
                    let d_j = entry_j.field.native_dim();
                    let k_j = entry_j.bases.k;
                    let native_buf_j = &mut scratch.native_buf_j[..d_j];
                    let spectral_j = &mut scratch.spectral_buf[..k_j];
                    entry_j.field.eval_native_into(i_t, native_buf_j);
                    project_to_spectral_into(native_buf_j, &entry_j.bases, spectral_j);
                    reconstruct_from_spectral_into(
                        spectral_j,
                        &entry_j.bases,
                        &mut scratch.b_at_d_j,
                    );
                    simd_dot_f32(&scratch.b_at_d_i, &scratch.b_at_d_j, D)
                };
                scratch.gram[i * P + j] += dot;
                if i != j {
                    scratch.gram[j * P + i] += dot;
                }
            }
            scratch.rhs[i] += simd_dot_f32(&scratch.b_at_d_i, dot_i_t, D);
        }
    }
}

/// Zero-allocation scratch for [`HeterogeneousEnsemble::fit_into`] /
/// [`HeterogeneousEnsemble::eval_into`].
///
/// Mirrors [`EnsembleFitScratch`] with two additions:
/// - `transport_scratch`: shared `CrossResScratch` for the per-field transport.
/// - `native_buf_i` / `native_buf_j`: per-field native-dim buffers (sized to
///   the max native dim across entries).
///
/// Construct once via [`Self::new`] (passing the entries so native buffer
/// sizes can be queried); reuse across fits/evals.
#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
pub struct HeterogeneousFitScratch<const P: usize, const D: usize> {
    /// `P×P` Gram matrix (row-major). `Vec<f32>` because stable Rust cannot
    /// express `[f32; P*P]` with const-generic `P`.
    pub gram: Vec<f32>,
    /// P-dim RHS.
    pub rhs: [f32; P],
    /// `K + λI` — Gram + ridge diagonal.
    pub gram_reg: Vec<f32>,
    /// Cholesky `L` factor scratch (`P×P`).
    pub chol: Vec<f32>,
    /// Back-substitution z scratch (`P`).
    pub z_solve: [f32; P],
    /// Field i output projected to D (post-transport).
    pub b_at_d_i: [f32; D],
    /// Field j output projected to D (post-transport).
    pub b_at_d_j: [f32; D],
    /// Spectral coefficient buffer (k-dim, sliced per field). Sized to max k
    /// across entries; borrowed as `&mut [..bases.k]` per transport call.
    /// Avoids CrossResScratch's resize-on-k-change (which would allocate).
    pub spectral_buf: Vec<f32>,
    /// Native-dim buffer for field i (sized to max native dim across entries).
    pub native_buf_i: Vec<f32>,
    /// Native-dim buffer for field j (sized to max native dim across entries).
    pub native_buf_j: Vec<f32>,
}

#[cfg(feature = "velocity_field_ensemble_heterogeneous")]
impl<const P: usize, const D: usize> HeterogeneousFitScratch<P, D> {
    /// Construct scratch sized for the given entries.
    ///
    /// Queries each entry's `native_dim()` and `bases.k` to size buffers.
    pub fn new(entries: &[HeterogeneousEntry; P]) -> Self {
        let max_native = entries
            .iter()
            .map(|e| e.field.native_dim())
            .max()
            .unwrap_or(1);
        let max_k = entries.iter().map(|e| e.bases.k).max().unwrap_or(1);
        Self {
            gram: vec![0.0; P * P],
            rhs: [0.0; P],
            gram_reg: vec![0.0; P * P],
            chol: vec![0.0; P * P],
            z_solve: [0.0; P],
            b_at_d_i: [0.0; D],
            b_at_d_j: [0.0; D],
            spectral_buf: vec![0.0; max_k],
            native_buf_i: vec![0.0; max_native],
            native_buf_j: vec![0.0; max_native],
        }
    }

    /// Zero the Gram and RHS (does not touch solver scratch).
    #[inline]
    pub fn clear(&mut self) {
        self.gram.fill(0.0);
        self.rhs.fill(0.0);
    }
}

#[cfg(all(test, feature = "velocity_field_ensemble_heterogeneous"))]
mod heterogeneous_tests {
    use super::*;
    use crate::cross_resolution::CrossResolutionBases;

    /// Build "pad-to-D" bases: identity `phi_src` (k = d_src) + `psi_dst`
    /// that places the first `d_src` coords of spectral into the first
    /// `d_src` coords of `d_dst`, zero-padding the rest. NOT orthonormal at
    /// `d_dst > d_src` but fine for testing the transport path.
    fn pad_bases(d_src: usize, d_dst: usize) -> CrossResolutionBases {
        assert!(d_src <= d_dst, "pad_bases requires d_src <= d_dst");
        let k = d_src;
        // phi_src: d_src × k identity (row-major).
        let mut phi_src = vec![0.0f32; d_src * k];
        for r in 0..d_src {
            phi_src[r * k + r] = 1.0;
        }
        // psi_dst: d_dst × k. First d_src rows = identity columns; rest = 0.
        let mut psi_dst = vec![0.0f32; d_dst * k];
        for r in 0..d_src {
            psi_dst[r * k + r] = 1.0;
        }
        CrossResolutionBases::new(phi_src, psi_dst, d_src, d_dst, k).unwrap()
    }

    /// Linear native field: `b(x) = W · x` where W is `native_dim × input_dim`.
    /// Stored row-major. Used as a controlled-basis test fixture.
    struct LinearNativeField {
        w: Vec<f32>, // native_dim × input_dim, row-major
        input_dim: usize,
        native_dim: usize,
        id: u64,
    }

    impl HeterogeneousVelocityField for LinearNativeField {
        fn eval_native_into(&self, x: &[f32], out_native: &mut [f32]) {
            let nd = self.native_dim;
            let id = self.input_dim;
            debug_assert_eq!(x.len(), id);
            debug_assert_eq!(out_native.len(), nd);
            for (r, out_slot) in out_native.iter_mut().enumerate().take(nd) {
                let row = &self.w[r * id..(r + 1) * id];
                *out_slot = simd_dot_f32(row, x, id);
            }
        }
        #[inline]
        fn native_dim(&self) -> usize {
            self.native_dim
        }
        #[inline]
        fn field_id(&self) -> u64 {
            self.id
        }
    }

    /// G1 mechanics: with identity-like (pad) transport, the heterogeneous
    /// ensemble is equivalent to a homogeneous ensemble on the padded fields.
    /// Solve a known-η recovery problem and verify `|η - η*|_∞ < 1e-4`.
    #[test]
    fn test_heterogeneous_fit_recovers_known_eta() {
        // D = 4. Field 0: native 2. Field 1: native 3. Field 2: native 4 (= D).
        const P: usize = 3;
        const D: usize = 4;
        const INPUT_DIM: usize = 4;

        // Construct 3 fields with distinct W matrices.
        let w0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2×4
        let w1 = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
        ]; // 3×4
        let w2 = vec![
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5,
        ]; // 4×4 (scaled identity)

        let f0 = LinearNativeField {
            w: w0,
            input_dim: INPUT_DIM,
            native_dim: 2,
            id: 100,
        };
        let f1 = LinearNativeField {
            w: w1,
            input_dim: INPUT_DIM,
            native_dim: 3,
            id: 101,
        };
        let f2 = LinearNativeField {
            w: w2,
            input_dim: INPUT_DIM,
            native_dim: 4,
            id: 102,
        };

        let b0 = pad_bases(2, D);
        let b1 = pad_bases(3, D);
        let b2 = pad_bases(4, D);

        let entries = [
            HeterogeneousEntry::new(Box::new(f0), b0),
            HeterogeneousEntry::new(Box::new(f1), b1),
            HeterogeneousEntry::new(Box::new(f2), b2),
        ];

        // Construct ensemble + scratch.
        let mut ens = HeterogeneousEnsemble::<P, D>::new(entries);
        let mut scratch = HeterogeneousFitScratch::<P, D>::new(&ens.entries);

        // Build N data pairs: for each x_n, the target is a known combination
        // of the transported fields. Use η* = [0.5, 0.3, 0.2].
        // transported b_i(x) at D: b0 → [x0, x1, 0, 0]; b1 → [x0, x1, x2, 0]; b2 → [0.5x0, 0.5x1, 0.5x2, 0.5x3].
        // target = η*[0]*b0_T + η*[1]*b1_T + η*[2]*b2_T.
        let eta_star = [0.5f32, 0.3, 0.2];
        let n = 50;
        let mut i_t_samples: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut dot_i_t_samples: Vec<Vec<f32>> = Vec::with_capacity(n);
        // Deterministic LCG for reproducibility (no external rand dep).
        let mut seed = 0x1234_5678u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0 // [-1, 1]
        };
        for _ in 0..n {
            let x = vec![lcg(), lcg(), lcg(), lcg()];
            let mut target = [0.0f32; D];
            // b0_T = [x0, x1, 0, 0]
            target[0] += eta_star[0] * x[0];
            target[1] += eta_star[0] * x[1];
            // b1_T = [x0, x1, x2, 0]
            target[0] += eta_star[1] * x[0];
            target[1] += eta_star[1] * x[1];
            target[2] += eta_star[1] * x[2];
            // b2_T = [0.5x0, 0.5x1, 0.5x2, 0.5x3]
            target[0] += eta_star[2] * 0.5 * x[0];
            target[1] += eta_star[2] * 0.5 * x[1];
            target[2] += eta_star[2] * 0.5 * x[2];
            target[3] += eta_star[2] * 0.5 * x[3];
            i_t_samples.push(x);
            dot_i_t_samples.push(target.to_vec());
        }
        let i_t_refs: Vec<&[f32]> = i_t_samples.iter().map(|v| v.as_slice()).collect();
        let dot_refs: Vec<&[f32]> = dot_i_t_samples.iter().map(|v| v.as_slice()).collect();

        ens.fit_into(&i_t_refs, &dot_refs, 1e-6, &mut scratch);

        // Tolerance is 5e-4 (not 1e-4) because:
        //   (a) the fields after pad-transport are correlated (b0_T's support
        //       is a subset of b1_T's), so the Gram is non-diagonal and the
        //       ridge bias λ·K^{-1}·η* is non-trivial even at λ=1e-6;
        //   (b) f32 precision limits the Gram accumulation accuracy at N=50.
        // 5e-4 is still 3 orders of magnitude below the smallest η* component
        // (0.2), so the recovery is unambiguous.
        for (i, (eta_i, eta_star_i)) in ens.eta.iter().zip(eta_star.iter()).enumerate().take(P) {
            assert!(
                (eta_i - eta_star_i).abs() < 5e-4,
                "η[{}] = {} != η*[{}] = {} (diff {})",
                i,
                eta_i,
                i,
                eta_star_i,
                eta_i - eta_star_i
            );
        }
    }

    /// Eval produces the correct D-dim output — verify against manual transport.
    #[test]
    fn test_heterogeneous_eval_matches_manual() {
        const P: usize = 2;
        const D: usize = 4;
        const INPUT_DIM: usize = 2;

        // Field 0: native dim 2, identity: b(x) = x.
        let w0 = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let f0 = LinearNativeField {
            w: w0,
            input_dim: INPUT_DIM,
            native_dim: 2,
            id: 300,
        };
        // Field 1: native dim 2, scaled identity ×2.
        let w1 = vec![2.0, 0.0, 0.0, 2.0];
        let f1 = LinearNativeField {
            w: w1,
            input_dim: INPUT_DIM,
            native_dim: 2,
            id: 301,
        };

        let b = pad_bases(2, D);
        let entries = [
            HeterogeneousEntry::new(Box::new(f0), CrossResolutionBases::clone(&b)),
            HeterogeneousEntry::new(Box::new(f1), b),
        ];
        let mut ens = HeterogeneousEnsemble::<P, D>::new(entries);

        // Force η = [0.5, 0.5] directly (skip fit).
        ens.eta = [0.5, 0.5];
        let mut scratch = HeterogeneousFitScratch::<P, D>::new(&ens.entries);

        let x = vec![1.0f32, 2.0];
        let mut out = [0.0f32; D];
        ens.eval_into(&x, &mut out, &mut scratch);

        // b0(x) = [1, 2], transport → [1, 2, 0, 0].
        // b1(x) = [2, 4], transport → [2, 4, 0, 0].
        // 0.5 * b0 + 0.5 * b1 = [1.5, 3.0, 0, 0].
        let expected = [1.5f32, 3.0, 0.0, 0.0];
        for k in 0..D {
            assert!(
                (out[k] - expected[k]).abs() < 1e-6,
                "out[{}] = {} != {}",
                k,
                out[k],
                expected[k]
            );
        }
    }
}
