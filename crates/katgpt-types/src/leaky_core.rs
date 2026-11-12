//! Shared leaky-integrator / delta-rule step primitive (Plan 276 Phase 2, T2.1).
//!
//! This is the **single source of truth** for the belief leaky-integrator update
//! math. It is compiled UNGATED (no `micro_belief` feature required) so that
//! `sense::reconstruction::ReconstructionState::evolve_belief` can delegate to it
//! without forcing the opt-in `micro_belief` feature on whenever `sense`
//! compiles — preserving the G1.4 latency gate decision (Issue 024, resolved +
//! removed).
//!
//! Two callers, one math:
//! - `crate::sense::reconstruction::ReconstructionState::evolve_belief` passes
//!   `total = Σ kind_activations[0..6]` (the 6 source SenseKind activations)
//!   and an 8-element `input` gathered via `KIND_MAP = [0,1,2,3,4,5,0,1]`.
//! - `crate::micro_belief::leaky::LeakyIntegrator::step` passes
//!   `total = Σ input[0..dim]` and a flat `dim`-element `input`.
//!
//! # Why `total` is a parameter, not computed inside
//!
//! The two callers aggregate activation mass over **different** element counts
//! (6 vs `dim`). Baking either count into this primitive would leak a
//! caller-specific detail. Instead each caller computes its own `total` and
//! hands it in; this fn owns everything downstream (early-return guard, scale,
//! half-total, per-element delta/clamp/apply). The body is identical bytes for
//! both callers given the same `(state, input, total, lr, max_delta)`.
//!
//! # Math (verbatim from the original `evolve_belief` body)
//!
//! ```text
//! if total < 1e-8: return                     // div-by-zero guard, no update
//! t_min      = total.min(1.0)
//! scale      = lr * t_min / total
//! half_total = 0.5 * total
//! for i in 0..state.len():
//!     delta         = scale * (input[i] - half_total)
//!     clamped_delta = delta.clamp(-max_delta, max_delta)
//!     state[i]      = (state[i] + clamped_delta).clamp(-1.0, 1.0)
//! ```
//!
//! Properties:
//! - Always stable: output clamped to `[-1, 1]`.
//! - Zero allocation: operates purely on the borrowed `&mut [f32]` / `&[f32]`.
//! - No softmax: pure additive update with sigmoid-style bounds.
//!
//! # Not a contraction (Research 667 §4.1 / Issue 1000 — a distinction, not a
//! defect)
//!
//! This kernel's update is STATE-INDEPENDENT (`z ↦ clamp(z + c(e))`): two
//! trajectories differing by a constant interior offset never converge, and
//! its fading is clamp-boundary-driven, not flow contraction. It is therefore
//! NOT a contraction in the arXiv:2611.09217 sense — recorded as a measured
//! distinction, never a defect: the divisive kernel is GOAT-adopted for
//! superposition-scaling reasons (ratio 5.17 additive-vs-divisive axes are
//! orthogonal, `modality_superposition_bench`). Do not "fix" the math. If a
//! consumer needs a CERTIFIED belief filter, the modality-additive kernel
//! ([`crate::belief_contract`], `evolve_belief_additive` in katgpt-sense) is
//! the one that carries the certificate — `ρ = α_m` exact, `κ = −ln α_m`.
//!
//! `state.len()` determines the iteration count; `input` must be at least that
//! long. Callers ensure `state.len() == input.len()`.

/// Advance `state` by one leaky-integrator tick driven by `input`.
///
/// `total` is the precomputed activation mass used for normalization (see module
/// docs for why it is a parameter). `lr` is the learning rate, `max_delta`
/// bounds the per-tick step. Zero-allocation, always stable to `[-1, 1]`.
///
/// Debug-asserts `state.len() == input.len()`; in release the loop runs over
/// `state.len()` and indexes `input`, so the caller must guarantee equal length.
#[inline]
pub fn leaky_step(state: &mut [f32], input: &[f32], total: f32, lr: f32, max_delta: f32) {
    debug_assert_eq!(
        state.len(),
        input.len(),
        "leaky_step: state/input length mismatch"
    );

    // Div-by-zero / degenerate-input guard — verbatim from evolve_belief.
    if total < 1e-8 {
        return;
    }

    let t_min = total.min(1.0);
    let scale = lr * t_min / total;
    let half_total = 0.5 * total;

    // Direct indexing (not iterator) over the fixed-length loop lets LLVM
    // unroll; for the 8-element belief this fully unrolls.
    let n = state.len();
    for i in 0..n {
        let normalized = input[i];
        let delta = scale * (normalized - half_total);
        let clamped_delta = delta.clamp(-max_delta, max_delta);
        state[i] = (state[i] + clamped_delta).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero total → early return, state untouched (mirrors evolve_belief guard).
    #[test]
    fn zero_total_is_noop() {
        let mut state = [0.5f32; 8];
        let input = [0.0f32; 8];
        leaky_step(&mut state, &input, 0.0, 0.1, 0.2);
        assert_eq!(state, [0.5f32; 8], "zero total must be a no-op");
    }

    /// Output never leaves `[-1, 1]` regardless of input magnitude.
    #[test]
    fn state_stays_bounded() {
        let mut state = [0.0f32; 8];
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..10_000 {
            let input: [f32; 8] = std::array::from_fn(|_| rng.f32());
            let total: f32 = input.iter().copied().sum();
            leaky_step(&mut state, &input, total, 0.5, 1.0);
            for &v in &state {
                assert!((-1.0..=1.0).contains(&v), "state diverged: {v}");
            }
        }
    }

    /// Reference math (hand-rolled) must match the primitive for a fixed input.
    #[test]
    fn matches_reference_math() {
        let lr = 0.1f32;
        let max_delta = 0.2f32;
        let input: [f32; 8] = [0.1, 0.2, 0.3, 0.0, 0.5, 0.0, 0.4, 0.4];
        let total: f32 = input.iter().copied().sum();

        let mut state_actual = [0.0f32; 8];
        leaky_step(&mut state_actual, &input, total, lr, max_delta);

        // Reference: verbatim formula.
        let t_min = total.min(1.0);
        let scale = lr * t_min / total;
        let half_total = 0.5 * total;
        let mut state_ref = [0.0f32; 8];
        for i in 0..8 {
            let delta = scale * (input[i] - half_total);
            let clamped = delta.clamp(-max_delta, max_delta);
            state_ref[i] = (state_ref[i] + clamped).clamp(-1.0, 1.0);
        }

        assert_eq!(
            state_actual, state_ref,
            "primitive must match reference math"
        );
    }

    /// Evolve_belief's sum-over-6 quirk: passing `total = Σ[0..6]` while looping
    /// over 8 gathered inputs must produce a different (and correct) result
    /// than summing all 8. This pins the contract callers rely on.
    #[test]
    fn total_is_caller_controlled() {
        let lr = 0.1f32;
        let max_delta = 0.2f32;
        // Gathered input with non-zero wrap positions (k0,k1 repeated at 6,7).
        let input: [f32; 8] = [0.5, 0.3, 0.2, 0.1, 0.4, 0.0, 0.5, 0.3];
        let total_six: f32 = input[..6].iter().copied().sum();
        let total_eight: f32 = input.iter().copied().sum();
        assert_ne!(total_six, total_eight);

        let mut s_six = [0.0f32; 8];
        let mut s_eight = [0.0f32; 8];
        leaky_step(&mut s_six, &input, total_six, lr, max_delta);
        leaky_step(&mut s_eight, &input, total_eight, lr, max_delta);
        assert_ne!(
            s_six, s_eight,
            "different totals must yield different states"
        );
    }
}
