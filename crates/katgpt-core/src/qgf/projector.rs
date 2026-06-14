//! `FirstOrderProjector` (Plan 268 F2) — one-step Euler projection of a
//! partial generation chain to its likely final output.
//!
//! # QGF Paper Eq 7
//!
//! ```text
//! â_1 = a_t + (1 - t) · v_θ(s, a_t, t)
//! ```
//!
//! Continuous form: integrate the velocity field `v_θ` for the remaining
//! fraction `(1 - t)` in a single Euler step.
//!
//! # Discrete Analogue
//!
//! For our discrete `SpeculativeGenerator`, the "remaining fraction" is
//! collapsed into a **single generator call**. We invoke the generator once
//! with the current prefix and take its output as the projection `â_1`.
//! This is the discrete "big Euler step":
//!
//! - **Cheaper** than full chain generation (one call, not many)
//! - **Better** than full chain generation, because it allows mode selection
//!   rather than strict adherence to the reference policy's exact distribution
//!   (QGF paper §5: "QGF is better at mode selection")
//! - **No BPTT**: the projection is a forward call, not differentiated
//!
//! The cost is exactly one `SpeculativeGenerator::generate()` call per
//! projection. Batch via `generate_batch` amortizes this further.
//!
//! # Why This Is Load-Bearing
//!
//! QGF's key insight is that the critic gradient should be evaluated at the
//! *projected* clean output, not at the intermediate noisy state. This is
//! what avoids both:
//! - The OOD bias of `∇_{a_t} Q(s, a_t)` (critic untrained on intermediates)
//! - The high variance / cost of `∇_{a_t} Q(s, ODE(a_t))` (BPTT through time)
//!
//! By exposing projection as a reusable primitive, every downstream consumer
//! (NFCoT FlowScore, ThoughtFold, ECHO, TRD, QGuidedDrafter) can query
//! "what would this chain likely end up as?" in O(1) generator calls.

use crate::traits::SpeculativeGenerator;

/// Project a partial generation chain to its likely final output via a
/// single generator call.
///
/// This is the discrete analogue of QGF's `â_1 = a_t + (1-t)·v_θ(s, a_t, t)`.
///
/// The generator is invoked **once** with the given condition (prefix). The
/// returned `Output` is the projection `â_1` — the generator's best guess
/// at the final output given the prefix.
///
/// # Cost
///
/// Exactly one `generate()` call. No backprop, no chain-rule, no BPTT.
/// For batch amortization, use [`project_batch`].
///
/// # QGF Properties
///
/// - **First-order**: collapses remaining generation budget into one step
/// - **Mode-selecting**: allows deviation from exact reference distribution
///   (this is *why* it beats full-chain projection — paper §5)
/// - **Jacobian-dropped**: the caller does not differentiate through this
///
/// # Panics
///
/// Panics if the generator returns zero candidates. `SpeculativeGenerator`
/// contract requires at least one output per call — this is the documented
/// precondition that QGF relies on.
#[inline]
pub fn project_one_step<G>(
    generator: &mut G,
    condition: &G::Condition,
    rng: &mut fastrand::Rng,
) -> Result<G::Output, G::Error>
where
    G: SpeculativeGenerator,
{
    // Single forward call. The generator's `generate()` returns a Vec<Output>;
    // we take the first (or only) candidate as the projection.
    //
    // Rationale: `generate()` is designed to produce *candidate* outputs.
    // For the projection, we want the single most likely one. We take [0]
    // as a convention — generators that produce multiple candidates should
    // rank them by likelihood (most generators already do).
    let mut candidates = generator.generate(condition, rng)?;
    if candidates.is_empty() {
        panic!(
            "QGF project_one_step: generator returned zero candidates. \
             SpeculativeGenerator::generate() must produce at least one output."
        );
    }
    // Take the first (highest-ranked) candidate. Generators that produce
    // multiple candidates are expected to rank them by likelihood, best first.
    //
    // `swap_remove(0)` is O(1) (single element copy + length decrement) vs
    // `remove(0)` which is O(n) (shifts all remaining elements). Since we
    // immediately drop the rest of `candidates`, the reorder is irrelevant.
    Ok(candidates.swap_remove(0))
}

/// Batch variant — projects multiple prefixes in a single
/// `generate_batch()` call for amortization.
///
/// Use this when projecting many prefixes at once (e.g., in a beam search
/// or when evaluating guidance for many candidates). Amortizes the generator
/// dispatch overhead.
///
/// # Cost
///
/// Exactly one `generate_batch()` call (which internally may call `generate()`
/// per condition by default, or batch on GPU if implemented).
///
/// # Returns
///
/// A `Vec<G::Output>` of projections, one per condition. Order is preserved.
///
/// # Panics
///
/// Panics if the generator returns zero candidates for any condition.
#[inline]
pub fn project_batch<G>(
    generator: &mut G,
    conditions: &[G::Condition],
    rng: &mut fastrand::Rng,
) -> Result<Vec<G::Output>, G::Error>
where
    G: SpeculativeGenerator,
{
    let batches = generator.generate_batch(conditions, rng)?;
    // Pre-allocate the output Vec to avoid reallocation during `collect()`.
    // Each batch entry contributes exactly one projection (the first candidate).
    let mut projections = Vec::with_capacity(batches.len());
    for mut candidates in batches {
        if candidates.is_empty() {
            panic!(
                "QGF project_batch: generator returned zero candidates for a condition. \
                 SpeculativeGenerator::generate_batch() must produce at least one output per condition."
            );
        }
        // Take the first (highest-ranked) candidate.
        // `swap_remove(0)` is O(1) — see `project_one_step` for rationale.
        projections.push(candidates.swap_remove(0));
    }
    Ok(projections)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock generator that returns a deterministic candidate based on the condition.
    struct MockGen;

    impl SpeculativeGenerator for MockGen {
        type Condition = u32;
        type Output = u32;
        type Error = ();

        fn generate(
            &mut self,
            condition: &Self::Condition,
            _rng: &mut fastrand::Rng,
        ) -> Result<Vec<Self::Output>, Self::Error> {
            // Projection: return condition + 1 (simulates one-step lookahead).
            Ok(vec![condition + 1])
        }
    }

    #[test]
    fn test_project_one_step_deterministic() {
        let mut generator = MockGen;
        let mut rng = fastrand::Rng::new();
        let projection = project_one_step(&mut generator, &5, &mut rng).unwrap();
        assert_eq!(projection, 6);
    }

    #[test]
    fn test_project_batch_preserves_order() {
        let mut generator = MockGen;
        let mut rng = fastrand::Rng::new();
        let conditions = vec![1, 2, 3];
        let projections = project_batch(&mut generator, &conditions, &mut rng).unwrap();
        assert_eq!(projections, vec![2, 3, 4]);
    }

    #[test]
    fn test_project_one_step_cost_is_one_call() {
        // The projection should make exactly one generate() call.
        // We verify by using a generator that panics on the second call.
        struct OneShotGen {
            calls: u32,
        }
        impl SpeculativeGenerator for OneShotGen {
            type Condition = ();
            type Output = u32;
            type Error = ();
            fn generate(
                &mut self,
                _condition: &Self::Condition,
                _rng: &mut fastrand::Rng,
            ) -> Result<Vec<Self::Output>, Self::Error> {
                self.calls += 1;
                if self.calls > 1 {
                    panic!("project_one_step should only call generate() once");
                }
                Ok(vec![42])
            }
        }

        let mut generator = OneShotGen { calls: 0 };
        let mut rng = fastrand::Rng::new();
        let projection = project_one_step(&mut generator, &(), &mut rng).unwrap();
        assert_eq!(projection, 42);
        assert_eq!(generator.calls, 1, "project_one_step must call generate() exactly once");
    }
}
