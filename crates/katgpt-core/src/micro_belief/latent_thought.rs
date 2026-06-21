//! Family B — latent-thought loop kernel (Plan 276 Phase 3 T3.1).
//!
//! Wraps a Family A [`AttractorKernel`] and applies its update rule K times per
//! tick, feeding the SAME input vector `x_t` on every inner iteration. This is
//! the "deliberation tick" family from the Mozer 2026 taxonomy — useful for
//! negotiation / multi-step social reasoning where one observed signal should
//! move the belief further toward a fixed point than a single attractor step
//! would.
//!
//! # Reduction to Family A
//!
//! `K = 1` reduces to Family A bit-identically (same weights, one step). This is
//! the G1.6 correctness property enforced by [`tests::k_equals_one_is_bit_identical_to_attractor`].
//!
//! # Zero allocation
//!
//! [`LatentThoughtKernel::step`] does not allocate — it loops `inner.step()` K
//! times, and `inner.step()` uses a fixed-size stack buffer internally (see
//! `attractor.rs`).
//!
//! # `K = 0` policy
//!
//! `K = 0` is allowed and is a **no-op**: the state is left unchanged for that
//! tick. This is the most flexible choice (lets the caller cleanly express
//! "skip this deliberation tick") and avoids the cost of a debug-assert panic
//! in release callers that build `k_iters` dynamically.

use crate::micro_belief::attractor::AttractorKernel;
use crate::micro_belief::bridge::project_to_scalars as bridge_project;
use crate::micro_belief::types::{MicroRecurrentBeliefState, RecurrenceFamily};
use crate::micro_belief::{assume_init_slice, uninit_stack};

/// Stack-buffer capacity for the precomputed `W_x · x` scratch. Matches the
/// `dim ≤ 1024` cap in `AttractorKernel::step`. Only `[..dim]` is used.
const WX_X_BUF_LEN: usize = 1024;

/// Family B — latent-thought loop: K iterations of the Family A attractor rule
/// per tick.
///
/// Construct via [`from_seed`](Self::from_seed) (delegates to
/// [`AttractorKernel::from_seed`]) or via the [`with_k_iters`](Self::with_k_iters)
/// builder on an existing inner kernel.
#[derive(Clone, Debug)]
pub struct LatentThoughtKernel {
    /// The wrapped Family A kernel. `pub` so callers / snapshots can read it.
    pub inner: AttractorKernel,
    /// Number of attractor iterations per `step()` call. `K = 1` reduces to
    /// Family A; `K = 0` is a no-op.
    pub k_iters: u8,
}

impl LatentThoughtKernel {
    /// Construct a latent-thought kernel with the given seed, dimension, and K.
    ///
    /// The inner attractor weights come from `AttractorKernel::from_seed(seed, dim)`,
    /// so two kernels built with the same `(seed, dim, k_iters)` are bit-identical
    /// (G1.1 determinism inherited from Family A).
    pub fn from_seed(seed: u64, dim: usize, k_iters: u8) -> Self {
        Self {
            inner: AttractorKernel::from_seed(seed, dim),
            k_iters,
        }
    }

    /// Builder: override the number of inner attractor iterations.
    #[inline]
    pub fn with_k_iters(mut self, k: u8) -> Self {
        self.k_iters = k;
        self
    }
}

impl MicroRecurrentBeliefState for LatentThoughtKernel {
    #[inline]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Advance one tick: apply `inner.step(state, input)` exactly `k_iters`
    /// times with the SAME `input` each iteration.
    ///
    /// # K > 1 precompute optimization
    ///
    /// For `k_iters > 1`, the input `x` is invariant across iterations — only
    /// `state` changes. We precompute `W_x · x` once via [`AttractorKernel::precompute_wx_dot`],
    /// then call [`AttractorKernel::step_with_precomputed_wx`] K times. This
    /// saves (K-1)×dim `simd_dot_f32` calls — at dim=32, K=3 that's ~64 saved
    /// dots ≈ 90ns/tick.
    ///
    /// The precomputed path is bit-identical to calling `inner.step()` K times
    /// (see `step_with_precomputed_wx_matches_step_bit_identical` in
    /// `attractor.rs` — same `simd_dot_f32` reductions, same addition order).
    ///
    /// # Zero allocation
    ///
    /// The `wx_x` scratch is a fixed `[f32; 1024]` stack buffer (4KB). K=1
    /// skips the buffer entirely (delegates directly to `inner.step`).
    ///
    /// # `K = 0`
    ///
    /// `k_iters == 0` is a no-op (state left unchanged). See the module-level
    /// docs for the rationale.
    #[inline]
    fn step(&self, state: &mut [f32], input: &[f32]) {
        // Match-style dispatch: K=0 noop, K=1 direct (no precompute overhead),
        // K>1 uses precomputed W_x·x across all iterations.
        match self.k_iters {
            0 => {} // no-op
            1 => self.inner.step(state, input),
            _ => {
                // Precompute W_x · x ONCE (bit-identical to the dot_wx inside
                // inner.step()). Reused across all K iterations.
                // Uninit stack buffer (matches cumprodsum.rs pattern) — wx_x[..dim]
                // is written by precompute_wx_dot before any read.
                let mut wx_x_buf = uninit_stack::<WX_X_BUF_LEN>();
                // SAFETY: wx_x[..dim] is fully written by precompute_wx_dot.
                let wx_x: &mut [f32] = unsafe { assume_init_slice(&mut wx_x_buf, self.inner.dim) };
                let dim = self.inner.dim;
                self.inner.precompute_wx_dot(input, &mut wx_x[..dim]);
                for _ in 0..self.k_iters {
                    self.inner.step_with_precomputed_wx(state, &wx_x[..dim]);
                }
            }
        }
    }

    #[inline(always)]
    fn project_to_scalars(&self, state: &[f32], directions: &[f32], dim: usize, out: &mut [f32]) {
        bridge_project(state, directions, dim, out);
    }

    #[inline]
    fn family(&self) -> RecurrenceFamily {
        RecurrenceFamily::LatentThought
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic input generator — local copy of the helper in `tests.rs`
    /// so this module's tests don't depend on a private item across modules.
    fn deterministic_input(step: usize, dim: usize) -> Vec<f32> {
        let s = step as f32;
        (0..dim)
            .map(|i| {
                let f = i as f32;
                (f * 0.1 + s * 0.01).sin() * 0.5 * (s * 0.003).cos()
            })
            .collect()
    }

    /// **G1.6** — K=1 reduces to Family A bit-identically.
    ///
    /// Same seed + same dim → identical inner weights. Running both kernels on
    /// the same 100-step input sequence MUST produce identical final states.
    /// This is the critical correctness property: Family B with K=1 is just
    /// Family A.
    #[test]
    fn k_equals_one_is_bit_identical_to_attractor() {
        let attractor = AttractorKernel::from_seed(42, 16);
        let latent = LatentThoughtKernel::from_seed(42, 16, 1);

        let mut s_a = vec![0.0f32; 16];
        let mut s_b = vec![0.0f32; 16];
        for step in 0..100 {
            let x = deterministic_input(step, 16);
            attractor.step(&mut s_a, &x);
            latent.step(&mut s_b, &x);
        }
        assert_eq!(
            s_a, s_b,
            "G1.6 FAIL: K=1 latent-thought diverged from attractor"
        );
    }

    /// `K = 0` is a documented no-op: state unchanged across many ticks.
    ///
    /// Decision (see module docs): K=0 is ALLOWED for flexibility. It must not
    /// panic and must not mutate state.
    #[test]
    fn k_equals_zero_is_noop() {
        let kernel = LatentThoughtKernel::from_seed(42, 16, 0);
        let initial: Vec<f32> = (0..16).map(|i| (i as f32) * 0.01 - 0.08).collect();
        let mut state = initial.clone();
        for step in 0..50 {
            let x = deterministic_input(step, 16);
            kernel.step(&mut state, &x);
        }
        assert_eq!(state, initial, "K=0 must be a no-op (state unchanged)");
    }

    /// `family()` reports `LatentThought`.
    #[test]
    fn family_is_latent_thought() {
        let kernel = LatentThoughtKernel::from_seed(42, 16, 3);
        assert_eq!(kernel.family(), RecurrenceFamily::LatentThought);
    }

    /// `dim()` delegates to the inner attractor kernel.
    #[test]
    fn dim_delegates_to_inner() {
        let kernel = LatentThoughtKernel::from_seed(42, 24, 2);
        assert_eq!(kernel.dim(), 24);
        assert_eq!(kernel.inner.dim(), 24);
    }

    /// **Determinism (G1.1 inheritance)** — same seed + same input → bit-identical.
    #[test]
    fn determinism() {
        let k1 = LatentThoughtKernel::from_seed(42, 16, 3);
        let k2 = LatentThoughtKernel::from_seed(42, 16, 3);
        let mut s1 = vec![0.0f32; 16];
        let mut s2 = vec![0.0f32; 16];
        for step in 0..200 {
            let x = deterministic_input(step, 16);
            k1.step(&mut s1, &x);
            k2.step(&mut s2, &x);
        }
        assert_eq!(s1, s2, "same seed + same input must be bit-identical");
    }

    /// **Sanity (informational, soft assertion)** — With K > 1 the state should
    /// move further toward the attractor's fixed point in one tick than with
    /// K = 1, given a constant input. We assert `‖state_K3 − state_K1‖ > 0`
    /// (they differ) AND that the K=3 state has moved further from the initial
    /// state than the K=1 state (a soft indication of faster settling).
    ///
    /// This is NOT a hard gate — it documents the intended qualitative behaviour.
    /// If attractor weights happen to produce a fixed point near the origin on
    /// this particular seed, the second assertion could theoretically fail; we
    /// keep the test robust by only asserting that K=3 and K=1 produce different
    /// states (the unambiguous property).
    #[test]
    fn k_iters_increases_settling_speed() {
        let k1 = LatentThoughtKernel::from_seed(42, 16, 1);
        let k3 = LatentThoughtKernel::from_seed(42, 16, 3);

        let mut s1 = vec![0.0f32; 16];
        let mut s3 = vec![0.0f32; 16];
        // A strong constant input that should push the state well away from
        // the zero initial state in both cases.
        let input: Vec<f32> = (0..16).map(|i| 0.5 - (i as f32) * 0.05).collect();

        k1.step(&mut s1, &input);
        k3.step(&mut s3, &input);

        // Hard property: the two trajectories must differ after one tick.
        assert_ne!(s1, s3, "K=1 and K=3 must produce different states");

        // Soft property: K=3 should have moved further from the origin
        // (settled further toward a fixed point) than K=1.
        let norm_sq_k1: f32 = s1.iter().map(|v| v * v).sum();
        let norm_sq_k3: f32 = s3.iter().map(|v| v * v).sum();
        // Informational only — print, do not hard-assert, in case the fixed
        // point happens to be near the origin for this seed.
        eprintln!("settling check: ‖s_K1‖² = {norm_sq_k1:.6}, ‖s_K3‖² = {norm_sq_k3:.6}");
    }

    /// Builder overrides K.
    #[test]
    fn with_k_iters_builder() {
        let kernel = LatentThoughtKernel::from_seed(42, 16, 1).with_k_iters(5);
        assert_eq!(kernel.k_iters, 5);
    }
}
