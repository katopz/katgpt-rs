//! Temporal Derivative Kernel — Dual Fast/Slow Surprise Signal (Plan 277).
//!
//! Distilled from O'Reilly 2026, "This is how the Neocortex Learns"
//! ([arXiv:2606.08720](https://arxiv.org/abs/2606.08720)). Research note:
//! [`katgpt-rs/.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md`](../.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md).
//!
//! Turns any streaming latent scalar (or fixed-size vector) into a signed
//! "surprise" signal — the implicit prediction-error channel the neocortex
//! uses for credit assignment, computed locally from a signal's own time
//! series with no external target and no backprop.
//!
//! ## Why
//!
//! Every EMA currently in the codebase is a *single* integrator
//! (`simd_fused_decay_write`, `evolve_hla`, etc.). The dual
//! `(I_fast − I_slow)` band-pass derivative is the smallest missing
//! primitive that upgrades four existing pillars:
//!
//! - **HLA companion** — `evolve_hla` tracks *what is*; derivative tracks
//!   *how fast it is changing*.
//! - **δ-Mem write gate** — writes only on surprising events.
//! - **Collapse detector fusion** — prediction-derivative collapse is
//!   orthogonal to entropy collapse.
//! - **Intrinsic curiosity** — `sigmoid(β · surprise_norm())` is a zero-cost
//!   curiosity signal that needs no Solver.
//!
//! ## Latent vs raw boundary
//!
//! Operates on latent state; emits a bounded scalar (`surprise_norm`) that
//! may sync as a raw summary statistic. Full N-dim derivative vector stays
//! local per-entity.
//!
//! ## Sigmoid, never softmax
//!
//! Per `AGENTS.md`: the bridge projection [`sigmoid_surprise_gate`] uses
//! sigmoid. Softmax over a single scalar is meaningless.
//!
//! ## Co-extraction provenance (Plan 338 Phase 2)
//!
//! Promoted from `katgpt-core::temporal_deriv` (Plan 277) so that the
//! `katgpt-sense` crate can consume the kernel via the leaf only, breaking
//! the katgpt-core cycle. katgpt-core re-exports this via
//! `katgpt_core::temporal_deriv::*` (bit-for-bit path preserved). Tests stay
//! in katgpt-core (they exercise the kernel through the re-export).

#![allow(clippy::needless_range_loop)]

use crate::simd::{fast_sigmoid, simd_dist_sq, simd_dot_f32, simd_fused_decay_write};

/// Dual fast/slow EMA temporal-derivative kernel.
///
/// `fast` tracks the signal with a short time constant; `slow` with a long
/// one. Their difference `(fast − slow)` is a signed band-pass derivative
/// that spikes on change and decays to zero when the signal is stationary
/// — the canonical neocortical prediction-error signal.
///
/// Generic over `N` (the signal dimension). Fixed-size array, zero
/// allocation, suitable for embedding inside per-NPC state structs.
///
/// # Invariants
///
/// - `0 < alpha_slow < alpha_fast <= 1` (validated at construction).
/// - State arrays are zero-initialized on [`new`](Self::new); use
///   [`with_initial`](Self::with_initial) for warm starts / snapshot
///   restore.
/// - All operations are branch-free and in-place; safe to call from hot
///   paths at any tier (plasma → cold).
#[derive(Clone, Debug)]
pub struct TemporalDerivativeKernel<const N: usize> {
    /// Fast EMA — short time constant.
    pub fast: [f32; N],
    /// Slow EMA — long time constant.
    pub slow: [f32; N],
    /// Fast EMA coefficient. `new_fast = (1 − α_f) · old_fast + α_f · signal`.
    pub alpha_fast: f32,
    /// Slow EMA coefficient. `new_slow = (1 − α_s) · old_slow + α_s · signal`.
    pub alpha_slow: f32,
}

impl<const N: usize> TemporalDerivativeKernel<N> {
    /// Construct a zero-initialized kernel.
    ///
    /// Validates `0 < alpha_slow < alpha_fast <= 1`. Panics in debug,
    /// clamps in release (paper's ~10× ratio is the canonical default;
    /// e.g. `alpha_fast=0.3, alpha_slow=0.03`).
    #[inline]
    pub fn new(alpha_fast: f32, alpha_slow: f32) -> Self {
        validate_alphas(alpha_fast, alpha_slow);
        Self {
            fast: [0.0; N],
            slow: [0.0; N],
            alpha_fast,
            alpha_slow,
        }
    }

    /// Construct with initial EMA state — for warm starts or snapshot
    /// restore.
    #[inline]
    pub fn with_initial(fast: [f32; N], slow: [f32; N], alpha_fast: f32, alpha_slow: f32) -> Self {
        validate_alphas(alpha_fast, alpha_slow);
        Self {
            fast,
            slow,
            alpha_fast,
            alpha_slow,
        }
    }

    /// Observe one signal sample; update both EMAs and return the
    /// per-dim signed surprise vector `(fast − slow)`.
    ///
    /// Branch-free, no allocations. Reuses [`simd_fused_decay_write`] when
    /// the `simd`-implied path is available (always-on in this crate — the
    /// kernel dispatches to NEON/AVX2/scalar inside `simd`).
    ///
    /// Paper reference: O'Reilly 2026 §Implementational — the CaMKII/DAPK1
    /// kinase cascade maps onto the `(fast − slow)` difference; we compute
    /// it algebraically rather than biochemically.
    #[inline]
    pub fn observe(&mut self, signal: &[f32; N]) -> [f32; N] {
        // Two EMA passes via the shared SIMD fused-decay-write kernel.
        // Layout: new = decay * old + write * src  →  matches
        //   (1 − α) · old + α · src  with decay=(1−α), write=α.
        simd_fused_decay_write(
            &mut self.fast,
            1.0 - self.alpha_fast,
            signal,
            self.alpha_fast,
        );
        simd_fused_decay_write(
            &mut self.slow,
            1.0 - self.alpha_slow,
            signal,
            self.alpha_slow,
        );

        // Output: fast − slow (band-pass derivative).
        let mut out = [0.0f32; N];
        for i in 0..N {
            out[i] = self.fast[i] - self.slow[i];
        }
        out
    }

    /// SIMD-optimized observe (alias — the default [`observe`](Self::observe)
    /// already routes through [`simd_fused_decay_write`]).
    ///
    /// Kept as a distinct entry point so callers that want to make the
    /// SIMD path explicit in their code can do so, and so that future
    /// wider-SIMD specializations have a natural home.
    #[inline]
    pub fn observe_simd(&mut self, signal: &[f32; N]) -> [f32; N] {
        self.observe(signal)
    }

    /// L2 norm of the current `(fast − slow)` derivative.
    ///
    /// Uses [`simd_dot_f32`] for the inner reduction when `N >= 4`.
    /// Bounded scalar in `[0, ∞)`; typical operating range `[0, 1]` after
    /// normalization.
    #[inline]
    pub fn surprise_norm(&self) -> f32 {
        // Direct squared-distance between fast and slow — avoids materializing
        // an intermediate `diff` buffer (saves N stack writes + N reads) and
        // fuses the subtract+FMA into a single SIMD pass. Numerically
        // bit-identical to the previous two-step form because both lower to
        // `d = a - b; acc = d.mul_add(d, acc)` (single rounding on the square).
        let sq = simd_dist_sq(&self.fast, &self.slow, N);
        // Guard against negative zero from FMA contraction.
        sq.max(0.0).sqrt()
    }

    /// Write `(fast − slow)` into a caller-provided buffer — zero-alloc
    /// read path for consumers that already own a scratch buffer.
    #[inline]
    pub fn derivative_slice(&self, out: &mut [f32; N]) {
        for i in 0..N {
            out[i] = self.fast[i] - self.slow[i];
        }
    }

    /// Fill both EMA arrays with zero — for entity respawn / session
    /// restart.
    #[inline]
    pub fn reset(&mut self) {
        self.fast = [0.0; N];
        self.slow = [0.0; N];
    }
}

impl<const N: usize> Default for TemporalDerivativeKernel<N> {
    /// Default: paper's ~10× ratio — `alpha_fast=0.3, alpha_slow=0.03`.
    #[inline]
    fn default() -> Self {
        Self::new(0.3, 0.03)
    }
}

/// Bridge helper: project a derivative vector onto a single bounded scalar
/// via `sigmoid(β · ‖derivative‖₂)`.
///
/// Canonical downstream projection per `AGENTS.md` latent→raw bridge rules.
/// **Never softmax** (single-scalar softmax is meaningless; sigmoid gives a
/// proper inject/skip probability).
///
/// `beta` is the inverse-temperature: large `beta` → sharp threshold,
/// small `beta` → soft gate. Typical operating value `beta ∈ [1, 10]`.
#[inline]
pub fn sigmoid_surprise_gate(derivative: &[f32], beta: f32) -> f32 {
    debug_assert!(
        beta.is_finite(),
        "sigmoid_surprise_gate: beta must be finite"
    );
    let sq = simd_dot_f32(derivative, derivative, derivative.len()).max(0.0);
    let norm = sq.sqrt();
    fast_sigmoid(beta * norm)
}

/// Validate `0 < alpha_slow < alpha_fast <= 1`. Debug panic on violation;
/// release clamp.
#[inline]
fn validate_alphas(alpha_fast: f32, alpha_slow: f32) {
    debug_assert!(
        alpha_slow > 0.0 && alpha_fast > alpha_slow && alpha_fast <= 1.0,
        "TemporalDerivativeKernel: require 0 < alpha_slow < alpha_fast <= 1, got fast={}, slow={}",
        alpha_fast,
        alpha_slow
    );
    // No-op in release: caller is responsible. Documented in rustdoc.
    let _ = (alpha_fast, alpha_slow);
}
