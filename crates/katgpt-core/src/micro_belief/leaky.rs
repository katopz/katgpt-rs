//! Family C — leaky integrator / delta-rule SSM kernel.
//!
//! This module exposes the Family C implementation of
//! [`MicroRecurrentBeliefState`] for Plan 276. It reuses the **shared**
//! leaky-integrator update math that lives in [`crate::leaky_core`] — the
//! same primitive that [`ReconstructionState::evolve_hla`] delegates to.
//!
//! # History / scope
//!
//! Previously this file carried a standalone **mirror** of the `evolve_hla`
//! math, with a note that refactoring `evolve_hla` itself to delegate here was
//! "OUT OF SCOPE / locked". Plan 276 Phase 2 T2.1 has now landed: the math
//! was lifted into [`crate::leaky_core`] (ungated, so `sense` can depend on it
//! without pulling in the `micro_belief` feature). Both callers now share one
//! update body:
//!
//! - [`LeakyIntegrator::step`] computes `total = Σ input[0..dim]` and calls
//!   [`crate::leaky_core::leaky_step`].
//! - [`ReconstructionState::evolve_hla`] computes `total = Σ kind_activations[0..6]`
//!   and calls [`crate::leaky_core::leaky_step`] with the `KIND_MAP`-gathered
//!   8-element input.
//!
//! # Why the two callers pass different `total`s
//!
//! `evolve_hla`'s normalization mass is the 6 distinct SenseKind activations,
//! but its per-element update loop runs over 8 gathered inputs (dims 6,7 reuse
//! kinds 0,1). The generic kernel here has no such wrap, so it sums all `dim`
//! inputs. Both are correct for their respective call sites; the shared core
//! takes `total` as a parameter precisely so neither quirk leaks into the
//! primitive. See [`crate::leaky_core`] for the exact formula and rationale.
//!
//! # Stable public API (G2.1 benchmark depends on it)
//!
//! [`LeakyIntegrator`] exposes `new`, `hla_default`, `step`,
//! `project_to_scalars`, `family`. These are NOT changed by T2.1.
//!
//! Properties inherited from the shared core:
//! - Always stable: output clamped to `[-1, 1]`.
//! - Zero allocation: operates on the `&mut [f32]` slice.
//! - No softmax: pure additive update with sigmoid-style bounds.

use crate::micro_belief::bridge::project_to_scalars as bridge_project;
use crate::micro_belief::types::{MicroRecurrentBeliefState, RecurrenceFamily};

/// Family C leaky-integrator kernel — mirrors `ReconstructionState::evolve_hla`.
///
/// Construct with [`new`](Self::new). The kernel is stateless aside from its
/// config (`lr`, `max_delta`, `dim`) — the belief vector lives in the caller's
/// `&mut [f32]`.
#[derive(Clone, Debug)]
pub struct LeakyIntegrator {
    /// Learning rate (`config.hla_learning_rate` in `ReconstructionState`).
    pub lr: f32,
    /// Maximum per-tick delta (`config.max_hla_delta` in `ReconstructionState`).
    pub max_delta: f32,
    /// Belief-vector dimension.
    pub dim: usize,
}

impl LeakyIntegrator {
    /// Construct a new leaky integrator.
    ///
    /// Defaults that match `ReconstructionConfig::default()`:
    /// - `lr = 0.1`
    /// - `max_delta = 0.2`
    pub fn new(lr: f32, max_delta: f32, dim: usize) -> Self {
        Self { lr, max_delta, dim }
    }

    /// Construct with HLA-default config (`lr=0.1`, `max_delta=0.2`).
    pub fn hla_default(dim: usize) -> Self {
        Self::new(0.1, 0.2, dim)
    }

    /// Recursively advance the kernel for `inputs.len()` ticks and classify the
    /// resulting belief-vector chain with [`crate::classify_chain`].
    ///
    /// The chain `s_0, s_1, …, s_k` (where `s_0 = initial_state` and
    /// `k = inputs.len()`) is captured into a flattened buffer and classified.
    /// Each input `inputs[t]` drives one [`MicroRecurrentBeliefState::step`]
    /// invocation producing `s_{t+1}`.
    ///
    /// **Zero per-step allocation** — double-buffered `s_a` / `s_b` plus a
    /// single up-front `Vec::with_capacity` for the chain. The depth-invariance
    /// `Scratch` is allocated inside this call; tight-loop callers should reuse
    /// one via the raw [`crate::classify_chain`] primitive.
    ///
    /// # Plan 306 Phase 4 (G3 — T4.3 caveat)
    ///
    /// `LeakyIntegrator::step` clamps the state to `[-1, 1]` on every tick
    /// (per-element `.clamp(-1.0, 1.0)` in [`crate::leaky_core::leaky_step`]).
    /// It therefore **also classifies as `DepthInvariant`** by construction,
    /// like the attractor. The T4.3 negative control builds an *unclamped*
    /// leaky update inline in the test (no kernel-level support needed) so the
    /// diagnostic can demonstrate its discriminating power on a real drift
    /// kernel — see `tests/depth_invariance_micro_belief.rs::leaky_*_depth_specific`.
    #[cfg(feature = "depth_invariance")]
    pub fn audit_depth_invariance(
        &self,
        initial_state: &[f32],
        inputs: &[&[f32]],
        cfg: &crate::DepthInvarianceConfig,
    ) -> crate::DepthInvarianceDiagnostic {
        let dim = self.dim;
        assert_eq!(initial_state.len(), dim, "initial_state must have length dim");
        for (i, inp) in inputs.iter().enumerate() {
            assert_eq!(inp.len(), dim, "inputs[{i}] must have length dim");
        }

        let k = inputs.len();
        let k_plus_1 = k + 1;

        let mut chain: Vec<f32> = Vec::with_capacity(k_plus_1 * dim);
        chain.extend_from_slice(initial_state);

        // Double-buffered state — see AttractorKernel::audit_depth_invariance
        // for the aliasing rationale.
        let mut s_a: Vec<f32> = initial_state.to_vec();
        let mut s_b: Vec<f32> = initial_state.to_vec();

        for inp in inputs {
            s_b.copy_from_slice(&s_a);
            MicroRecurrentBeliefState::step(self, &mut s_b, inp);
            chain.extend_from_slice(&s_b);
            std::mem::swap(&mut s_a, &mut s_b);
        }

        let mut scratch = crate::Scratch::with_capacity(k_plus_1, dim);
        crate::classify_chain(&chain, dim, cfg, &mut scratch)
    }
}

impl MicroRecurrentBeliefState for LeakyIntegrator {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Advance one tick using the leaky-integrator update.
    ///
    /// Delegates to the shared [`crate::leaky_core::leaky_step`] primitive —
    /// the same body used by `ReconstructionState::evolve_hla`. Here `total` is
    /// `Σ input[0..dim]` (no KIND_MAP wrap); see the module docs for why
    /// `evolve_hla` passes a different `total`.
    #[inline]
    fn step(&self, state: &mut [f32], input: &[f32]) {
        debug_assert_eq!(state.len(), self.dim, "state/dim mismatch");
        debug_assert_eq!(input.len(), self.dim, "input/dim mismatch");

        // Generic Family C kernel: normalize over the full dim-length input.
        let total: f32 = input.iter().copied().sum();
        crate::leaky_core::leaky_step(state, input, total, self.lr, self.max_delta);
    }

    #[inline(always)]
    fn project_to_scalars(
        &self,
        state: &[f32],
        directions: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        bridge_project(state, directions, dim, out);
    }

    #[inline]
    fn family(&self) -> RecurrenceFamily {
        RecurrenceFamily::DeltaRule
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_input_is_noop() {
        // Matches evolve_hla: total < 1e-8 → early return, state unchanged.
        let k = LeakyIntegrator::hla_default(8);
        let mut state = [0.5f32; 8];
        let input = [0.0f32; 8];
        k.step(&mut state, &input);
        assert_eq!(state, [0.5f32; 8], "zero input must be a no-op");
    }

    #[test]
    fn state_stays_in_minus_one_to_one() {
        // Mirror of G1.2 for Family C — must always pass (linear, always stable).
        let k = LeakyIntegrator::new(0.5, 1.0, 8);
        let mut state = [0.0f32; 8];
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..10_000 {
            let input: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, rng.f32(), rng.f32(), rng.f32()];
            k.step(&mut state, &input);
            for &v in &state {
                assert!((-1.0..=1.0).contains(&v), "Family C diverged: {v}");
            }
        }
    }

    #[test]
    fn family_is_delta_rule() {
        let k = LeakyIntegrator::hla_default(8);
        assert_eq!(k.family(), RecurrenceFamily::DeltaRule);
    }

    #[test]
    fn matches_evolve_hla_math_reference() {
        // Reference implementation of the evolve_hla math, computed directly.
        // The kernel MUST produce identical output for the same input.
        let lr = 0.1f32;
        let max_delta = 0.2f32;
        let dim = 8usize;
        let k = LeakyIntegrator::new(lr, max_delta, dim);

        let input: [f32; 8] = [0.1, 0.2, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0];
        let mut state_actual = [0.0f32; 8];
        let mut state_ref = [0.0f32; 8];

        // Reference: verbatim evolve_hla body (without KIND_MAP — direct input).
        let total: f32 = input.iter().sum();
        assert!(total >= 1e-8);
        let t_min = total.min(1.0);
        let scale = lr * t_min / total;
        let half_total = 0.5 * total;
        for i in 0..dim {
            let normalized = input[i];
            let delta = scale * (normalized - half_total);
            let clamped_delta = delta.clamp(-max_delta, max_delta);
            state_ref[i] = (state_ref[i] + clamped_delta).clamp(-1.0, 1.0);
        }

        // Actual: through the kernel.
        k.step(&mut state_actual, &input);

        assert_eq!(state_actual, state_ref, "kernel must match evolve_hla math");
    }

    #[test]
    fn hla_default_matches_reconstruction_defaults() {
        // The defaults here must match ReconstructionConfig's HLA defaults so
        // the future T2.1 refactor is a true zero-behavior-change delegate.
        let k = LeakyIntegrator::hla_default(8);
        assert_eq!(k.lr, 0.1);
        assert_eq!(k.max_delta, 0.2);
        assert_eq!(k.dim, 8);
    }
}
