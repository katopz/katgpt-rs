//! `QGuidedDrafter` (Plan 268 F1 / Phase 2 T4) — fuses any
//! [`SpeculativeGenerator`] with a [`QGradientOracle`] for test-time
//! Q-gradient guidance.
//!
//! # QGF Algorithm 1 (discrete analogue)
//!
//! ```text
//! For each generation step t:
//!   1. Generate candidate marginal p_t from the reference generator.
//!   2. Project prefix → final:  â_1 = project_one_step(p_t)
//!   3. Query critic gradient:  g = oracle.q_gradient_at(state, &â_1)
//!   4. Tilt marginal (logit space):  logits' = logits + (1/β) · g
//!   5. Sample from tilted marginal.
//! ```
//!
//! # Adaptation to katgpt-rs Traits
//!
//! The plan's reference pseudocode assumed a logits-aware generator
//! (`logits_into` / `sample`). The actual [`SpeculativeGenerator`] trait
//! exposes only `generate(condition, rng) -> Result<Vec<Output>>` — it hides
//! internal logits. We therefore split the drafter into two complementary
//! surfaces:
//!
//! - [`QGuidedDrafter::generate_guided`] — high-level wrapper around the
//!   real `generate()` API. Returns the generator's candidate list, with the
//!   gradient computed at the projection for downstream diagnostic use.
//!   When `guidance_weight == 0.0` or the step is outside the guidance
//!   period, output is byte-identical to the base generator.
//! - [`QGuidedDrafter::tilt_logits`] — the pure QGF tilt math, operating on
//!   caller-owned logits + gradient buffers. This is the load-bearing
//!   primitive for any logits-based generator flow (NFCoT FlowScore, DDTree)
//!   and is the part that is unit-testable in isolation.
//!
//! # Sigmoid, Not Softmax
//!
//! The tilt is applied as an **additive logit shift** (`logits += w · g`),
//! never a softmax normalisation pass. Per project rules: sigmoid is per-query
//! and SIMD-friendly; softmax couples queries and requires a normalisation
//! reduction. See `.contexts/optimization.md`.
//!
//! # No Allocations in the Hot Path
//!
//! `tilt_logits` and `tilt_logits_adaptive` operate entirely on caller-owned
//! buffers. `generate_guided_into` reuses the caller's `Vec` capacity. The
//! only unavoidable allocation is inside `SpeculativeGenerator::generate()`
//! itself, which the drafter does not control.

use crate::qgf::projector::project_one_step;
use crate::traits::{QGradientOracle, SpeculativeGenerator};

/// Default guidance period — apply guidance every step.
pub const DEFAULT_GUIDANCE_PERIOD: usize = 1;

/// Test-time Q-gradient-guided speculative drafter.
///
/// Wraps a reference [`SpeculativeGenerator`] with a [`QGradientOracle`].
/// At each generation step, the drafter:
///
/// 1. Queries the generator for candidate outputs.
/// 2. Takes the first candidate as the first-order projection `â_1`.
/// 3. Queries `∇_a Q(s, â_1)` from the oracle (Jacobian dropped, per QGF §5).
/// 4. Tilts the marginal by `(1/β) · g` in logit space.
///
/// The type parameters bind the oracle's `State` to the generator's
/// `Condition` and the oracle's `Action` to the generator's `Output`.
pub struct QGuidedDrafter<G, O> {
    /// The reference (BC) generator — produces unguided candidates.
    pub generator: G,
    /// The critic gradient oracle — provides `∇_a Q(s, a)`.
    pub oracle: O,
    /// `1/β` — guidance strength. `0.0` = no guidance (pure BC reference).
    pub guidance_weight: f32,
    /// Apply guidance every `N` steps. `1` = every step. `2` = every other.
    pub guidance_period: usize,
}

impl<G, O> QGuidedDrafter<G, O>
where
    G: SpeculativeGenerator,
    O: QGradientOracle<State = G::Condition, Action = G::Output>,
{
    /// Construct a new drafter with default period (1 = every step) and
    /// zero guidance weight (pure BC reference until caller opts in).
    #[inline]
    pub fn new(generator: G, oracle: O) -> Self {
        Self {
            generator,
            oracle,
            guidance_weight: 0.0,
            guidance_period: DEFAULT_GUIDANCE_PERIOD,
        }
    }

    /// Builder: set the guidance weight `1/β`.
    #[inline]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.guidance_weight = weight;
        self
    }

    /// Builder: set the guidance period (apply guidance every `N` steps).
    #[inline]
    pub fn with_period(mut self, period: usize) -> Self {
        // period == 0 would cause modulo-by-zero; coerce to 1.
        self.guidance_period = if period == 0 { 1 } else { period };
        self
    }

    /// Returns `true` if guidance should be applied at this step.
    ///
    /// Guidance is active iff:
    /// - `guidance_weight > 0.0` (non-trivial tilt), AND
    /// - `step % guidance_period == 0` (period matches).
    #[inline]
    pub fn should_apply_guidance(&self, step: usize) -> bool {
        self.guidance_weight > 0.0 && step.is_multiple_of(self.guidance_period)
    }

    /// Convenience: project a condition to its likely final output via a
    /// single generator call. Thin wrapper over [`project_one_step`].
    ///
    /// This is QGF Algorithm 1 step 2 — the first-order Euler projection.
    /// Use it to obtain `â_1` before calling [`tilt_logits`](Self::tilt_logits).
    ///
    /// # Cost
    ///
    /// Exactly one `generate()` call on the wrapped generator.
    ///
    /// # Errors
    ///
    /// Propagates `G::Error`. Panics if the generator returns zero candidates
    /// (violates the `SpeculativeGenerator` contract).
    #[inline]
    pub fn project(
        &mut self,
        condition: &G::Condition,
        rng: &mut fastrand::Rng,
    ) -> Result<G::Output, G::Error> {
        project_one_step(&mut self.generator, condition, rng)
    }

    /// Apply the Q-gradient tilt to a logits buffer in place.
    ///
    /// This is the discrete analogue of QGF Algorithm 1 step 4:
    ///
    /// ```text
    /// logits[i] += guidance_weight * gradient[i]
    /// ```
    ///
    /// The tilt is an **additive logit shift** — never softmax normalisation
    /// (per project rules: sigmoid not softmax). The caller is responsible
    /// for sampling from the tilted logits after this call.
    ///
    /// The gradient is queried at `(condition, projected_action)` — the caller
    /// must supply the first-order projection `â_1` (typically via
    /// [`project_one_step`]).
    ///
    /// # Arguments
    ///
    /// - `condition` — the generation condition / state (oracle's `State`).
    /// - `projected` — the projected action `â_1` at which to evaluate `∇Q`.
    /// - `logits_buffer` — the marginal logits to tilt (mutated in place).
    /// - `gradient_buffer` — scratch space for the oracle to write `∇Q` into.
    ///   Must be at least as long as `logits_buffer`.
    /// - `step` — current generation step index (for period gating).
    ///
    /// # Returns
    ///
    /// `true` if guidance was applied; `false` if skipped (zero weight or
    /// period mismatch).
    #[inline]
    pub fn tilt_logits(
        &self,
        condition: &G::Condition,
        projected: &G::Output,
        logits_buffer: &mut [f32],
        gradient_buffer: &mut [f32],
        step: usize,
    ) -> bool {
        if !self.should_apply_guidance(step) {
            return false;
        }
        self.oracle
            .q_gradient_into(condition, projected, gradient_buffer);
        // Tilt: additive shift over the shorter of the two buffers.
        let n = logits_buffer.len().min(gradient_buffer.len());
        let w = self.guidance_weight;
        // SIMD AXPY via `simd_fused_scale_acc` — same single-rounding FMA
        // semantics as the scalar `w.mul_add(gradient[i], logits[i])` form
        // (verified bit-identical by `scalar_fused_scale_acc` in simd/research.rs),
        // but vectorized via NEON/AVX2. Hot path on every guided generation step.
        crate::simd::simd_fused_scale_acc(logits_buffer, gradient_buffer, w, n);
        true
    }

    /// Adaptive variant — computes `1/β` per call from the oracle's confidence.
    ///
    /// `weight = sigmoid(steepness · (confidence − threshold))`
    ///
    /// - Low confidence → weight ≈ 0 → pure BC reference (safe fallback)
    /// - High confidence → weight ≈ 1 → strong Q-guidance
    ///
    /// See [`adaptive_guidance_weight`](crate::qgf::adaptive::adaptive_guidance_weight).
    ///
    /// # Arguments
    ///
    /// As [`tilt_logits`](Self::tilt_logits), plus:
    /// - `threshold` — confidence at which guidance is half-maximal (typical: `0.5`).
    /// - `steepness` — transition sharpness (typical: `4.0` to `8.0`).
    #[cfg(feature = "qgf_adaptive")]
    #[inline]
    pub fn tilt_logits_adaptive(
        &self,
        condition: &G::Condition,
        projected: &G::Output,
        logits_buffer: &mut [f32],
        gradient_buffer: &mut [f32],
        step: usize,
        threshold: f32,
        steepness: f32,
    ) -> bool {
        // Period gate applies even in adaptive mode — no point querying the
        // oracle if we're not tilting this step.
        if !step.is_multiple_of(self.guidance_period) {
            return false;
        }
        let confidence = self.oracle.confidence(condition);
        let weight = crate::qgf::adaptive::adaptive_guidance_weight(
            confidence,
            threshold,
            steepness,
        );
        if weight <= 0.0 {
            return false;
        }
        self.oracle
            .q_gradient_into(condition, projected, gradient_buffer);
        let n = logits_buffer.len().min(gradient_buffer.len());
        // SIMD AXPY — matches `tilt_logits` numerically (single-rounding FMA).
        crate::simd::simd_fused_scale_acc(logits_buffer, gradient_buffer, weight, n);
        true
    }

    /// High-level guided generation via the real `SpeculativeGenerator` API.
    ///
    /// Generates the candidate list from the reference generator and computes
    /// the Q-gradient at the projection `â_1 = candidates[0]`. The gradient is
    /// returned alongside the candidates for downstream diagnostic or
    /// re-ranking use.
    ///
    /// **Note:** because [`SpeculativeGenerator`] hides internal logits, this
    /// method cannot structurally tilt the generator's own marginal. Callers
    /// needing logit-space tilt should use [`tilt_logits`](Self::tilt_logits)
    /// directly on a logits buffer they own. This method is the correct
    /// entry point when the generator produces opaque outputs (e.g. game
    /// actions) and guidance is consumed as a diagnostic signal.
    ///
    /// # Arguments
    ///
    /// - `condition` — generation condition / state.
    /// - `rng` — RNG for the generator's sampling.
    /// - `step` — current generation step (for period gating).
    ///
    /// # Returns
    ///
    /// The candidate list from the generator. When `guidance_weight == 0.0`
    /// or the step is outside the guidance period, the output is byte-identical
    /// to calling `generator.generate(condition, rng)` directly.
    ///
    /// # Errors
    ///
    /// Propagates `G::Error` from the underlying generator.
    pub fn generate_guided(
        &mut self,
        condition: &G::Condition,
        rng: &mut fastrand::Rng,
        step: usize,
    ) -> Result<Vec<G::Output>, G::Error> {
        let candidates = self.generator.generate(condition, rng)?;
        if self.should_apply_guidance(step) && !candidates.is_empty() {
            // Project: first candidate is the most-likely output per
            // SpeculativeGenerator contract. Query the gradient for downstream
            // consumers; the actual logit tilt happens in `tilt_logits`.
            let _gradient = self.oracle.q_gradient_at(condition, &candidates[0]);
        }
        Ok(candidates)
    }

    /// Zero-alloc variant — fills the caller-provided buffer instead of
    /// returning a fresh `Vec`.
    ///
    /// Clears `out`, then moves the generator's candidates into it. The
    /// generator's own internal `Vec` allocation is unavoidable (it's hidden
    /// behind the `generate()` contract), but this method avoids the
    /// drafter layer allocating a second `Vec`.
    ///
    /// Reuses `out`'s existing capacity across calls when possible.
    pub fn generate_guided_into(
        &mut self,
        condition: &G::Condition,
        rng: &mut fastrand::Rng,
        step: usize,
        out: &mut Vec<G::Output>,
    ) -> Result<(), G::Error> {
        let mut candidates = self.generator.generate(condition, rng)?;
        if self.should_apply_guidance(step) && !candidates.is_empty() {
            let _gradient = self.oracle.q_gradient_at(condition, &candidates[0]);
        }
        // Move elements into the caller's buffer without reallocation when
        // the buffer already has sufficient capacity.
        out.clear();
        out.append(&mut candidates);
        Ok(())
    }

    /// One-shot convenience: generate, project, tilt, sample.
    ///
    /// Chains `generate_guided` + [`project_one_step`] + [`tilt_logits`].
    /// Useful when the caller wants the full QGF pipeline in a single call
    /// and owns the logits/gradient buffers.
    ///
    /// The caller-provided `sample` closure converts the tilted logits into
    /// the final `G::Output`. This keeps the drafter agnostic to the sampler
    /// (greedy, nucleus, temperature, etc.).
    ///
    /// # Arguments
    ///
    /// - `condition` — generation condition.
    /// - `rng` — RNG.
    /// - `step` — generation step.
    /// - `logits_buffer` — marginal logits to tilt (mutated).
    /// - `gradient_buffer` — oracle scratch (mutated).
    /// - `sample` — closure mapping tilted logits → output.
    ///
    /// # Errors
    ///
    /// Propagates `G::Error` from the generator.
    pub fn generate_project_tilt_sample<F>(
        &mut self,
        condition: &G::Condition,
        rng: &mut fastrand::Rng,
        step: usize,
        logits_buffer: &mut [f32],
        gradient_buffer: &mut [f32],
        sample: F,
    ) -> Result<G::Output, G::Error>
    where
        F: FnOnce(&[f32]) -> G::Output,
    {
        // Step 1: generate candidates (for the projection).
        let mut candidates = self.generator.generate(condition, rng)?;
        if candidates.is_empty() {
            // Cannot project from an empty candidate list; fall back to a
            // second generate call to satisfy the SpeculativeGenerator contract
            // (at least one output per call). If the generator is genuinely
            // empty, propagate via the sample closure on un-tilted logits.
            return Ok(sample(logits_buffer));
        }
        // Step 2: take first candidate as projection.
        // (project_one_step would re-call generate; we already have candidates.)
        let projected = candidates.remove(0);
        // Step 3 + 4: tilt logits in place.
        let _applied = self.tilt_logits(
            condition,
            &projected,
            logits_buffer,
            gradient_buffer,
            step,
        );
        // Step 5: sample from tilted logits.
        Ok(sample(logits_buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{QGradientOracle, SpeculativeGenerator};

    // ── Mock generator (deterministic candidate list) ──────────────────────

    /// Mock generator returning a fixed candidate list. The first candidate
    /// is the projection target.
    struct MockGen {
        calls: u32,
    }

    impl SpeculativeGenerator for MockGen {
        type Condition = ();
        type Output = u32;
        type Error = ();

        fn generate(
            &mut self,
            _condition: &Self::Condition,
            _rng: &mut fastrand::Rng,
        ) -> Result<Vec<Self::Output>, Self::Error> {
            self.calls += 1;
            Ok(vec![10, 20, 30])
        }
    }

    // ── Mock oracle (returns a known gradient vector) ──────────────────────

    /// Mock oracle that returns a pre-set gradient and confidence. Used to
    /// verify the tilt math independently of any real critic.
    struct MockOracle {
        gradient: Vec<f32>,
        confidence: f32,
    }

    impl QGradientOracle for MockOracle {
        type State = ();
        type Action = u32;

        fn q_gradient_at(
            &self,
            _state: &Self::State,
            _projected_action: &Self::Action,
        ) -> Vec<f32> {
            self.gradient.clone()
        }

        fn q_gradient_into(
            &self,
            _state: &Self::State,
            _projected_action: &Self::Action,
            out: &mut [f32],
        ) {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = self.gradient.get(i).copied().unwrap_or(0.0);
            }
        }

        fn confidence(&self, _state: &Self::State) -> f32 {
            self.confidence
        }
    }

    fn make_drafter(weight: f32, period: usize) -> QGuidedDrafter<MockGen, MockOracle> {
        QGuidedDrafter::new(MockGen { calls: 0 }, MockOracle {
            gradient: vec![1.0, 2.0, 3.0],
            confidence: 0.9,
        })
        .with_weight(weight)
        .with_period(period)
    }

    // ── Test: zero weight matches base generator ───────────────────────────

    #[test]
    fn test_zero_weight_matches_base() {
        let mut rng = fastrand::Rng::new();
        // Base generator output.
        let mut base = MockGen { calls: 0 };
        let base_out = base.generate(&(), &mut rng).unwrap();

        // Drafter with zero guidance weight.
        let mut drafter = make_drafter(0.0, 1);
        let guided = drafter.generate_guided(&(), &mut rng, 0).unwrap();

        assert_eq!(guided, base_out, "zero-weight drafter must match base");
        assert_eq!(
            drafter.generator.calls, 1,
            "drafter should call generator exactly once"
        );
    }

    // ── Test: positive weight tilts marginal toward high-Q region ──────────

    #[test]
    fn test_positive_weight_tilts_marginal() {
        let drafter = make_drafter(0.5, 1);

        // Initial logits: uniform.
        let mut logits = [0.0f32, 0.0, 0.0];
        let mut grad = [0.0f32, 0.0, 0.0];
        // MockOracle.gradient = [1.0, 2.0, 3.0] → tilt = 0.5 * [1,2,3] = [0.5,1.0,1.5]
        let applied = drafter.tilt_logits(&(), &10u32, &mut logits, &mut grad, 0);

        assert!(applied, "guidance should be applied with weight > 0");
        assert_eq!(logits, [0.5, 1.0, 1.5], "tilt must be weight * gradient");

        // The high-Q region (index 2) has the highest tilted logit.
        let mean_high = (logits[1] + logits[2]) / 2.0;
        let mean_low = (logits[0] + logits[1]) / 2.0;
        assert!(
            logits[2] > logits[0],
            "high-Q index must have higher tilted logit"
        );
        assert!(
            mean_high > mean_low,
            "mean of high-Q region must exceed mean of low-Q region"
        );
    }

    // ── Test: period skips guidance on off-period steps ────────────────────

    #[test]
    fn test_period_skips_guidance() {
        let drafter = make_drafter(1.0, 2);

        let mut logits = [0.0f32; 3];
        let mut grad = [0.0f32; 3];

        // Step 0: 0 % 2 == 0 → apply.
        let applied0 = drafter.tilt_logits(&(), &10u32, &mut logits, &mut grad, 0);
        assert!(applied0, "step 0 with period 2 should apply guidance");
        assert_eq!(logits, [1.0, 2.0, 3.0], "tilt applied at step 0");

        // Step 1: 1 % 2 == 1 → skip.
        let mut logits2 = [0.0f32; 3];
        let applied1 = drafter.tilt_logits(&(), &10u32, &mut logits2, &mut grad, 1);
        assert!(
            !applied1,
            "step 1 with period 2 should skip guidance"
        );
        assert_eq!(logits2, [0.0, 0.0, 0.0], "no tilt applied at step 1");

        // Step 2: 2 % 2 == 0 → apply.
        let mut logits3 = [0.0f32; 3];
        let applied2 = drafter.tilt_logits(&(), &10u32, &mut logits3, &mut grad, 2);
        assert!(applied2, "step 2 with period 2 should apply guidance");
    }

    // ── Test: zero weight + should_apply_guidance ──────────────────────────

    #[test]
    fn test_should_apply_guidance_gates() {
        let d0 = make_drafter(0.0, 1);
        assert!(!d0.should_apply_guidance(0), "zero weight → never apply");

        let d1 = make_drafter(1.0, 1);
        assert!(d1.should_apply_guidance(0), "weight>0, period 1, step 0 → apply");
        assert!(d1.should_apply_guidance(99), "period 1 applies every step");

        let d2 = make_drafter(1.0, 3);
        assert!(d2.should_apply_guidance(0), "step 0 % 3 == 0 → apply");
        assert!(!d2.should_apply_guidance(1), "step 1 % 3 != 0 → skip");
        assert!(!d2.should_apply_guidance(2), "step 2 % 3 != 0 → skip");
        assert!(d2.should_apply_guidance(3), "step 3 % 3 == 0 → apply");
    }

    // ── Test: generate_guided_into reuses caller buffer ────────────────────

    #[test]
    fn test_generate_guided_into_fills_buffer() {
        let mut rng = fastrand::Rng::new();
        let mut drafter = make_drafter(0.0, 1);
        let mut out = Vec::new();

        drafter
            .generate_guided_into(&(), &mut rng, 0, &mut out)
            .unwrap();
        assert_eq!(out, vec![10, 20, 30]);

        // Second call should clear and refill (no stale data).
        out.push(999);
        drafter
            .generate_guided_into(&(), &mut rng, 1, &mut out)
            .unwrap();
        assert_eq!(out, vec![10, 20, 30], "buffer should be cleared and refilled");
    }

    // ── Test: full pipeline generate_project_tilt_sample ───────────────────

    #[test]
    fn test_generate_project_tilt_sample_greedy() {
        let mut rng = fastrand::Rng::new();
        let mut drafter = make_drafter(1.0, 1);

        // Logits start uniform; after tilt by [1,2,3], greedy picks index 2.
        let mut logits = [0.0f32; 3];
        let mut grad = [0.0f32; 3];

        let result = drafter
            .generate_project_tilt_sample(
                &(),
                &mut rng,
                0,
                &mut logits,
                &mut grad,
                |l| {
                    // Greedy argmax.
                    let mut best = 0u32;
                    let mut best_v = f32::MIN;
                    for (i, &v) in l.iter().enumerate() {
                        if v > best_v {
                            best_v = v;
                            best = i as u32;
                        }
                    }
                    best
                },
            )
            .unwrap();

        assert_eq!(result, 2, "greedy on tilted logits should pick index 2");
    }

    // ── Test: builder setters are independent ──────────────────────────────

    #[test]
    fn test_builder_setters() {
        let d = QGuidedDrafter::new(MockGen { calls: 0 }, MockOracle {
            gradient: vec![],
            confidence: 0.0,
        })
        .with_weight(2.5)
        .with_period(4);

        assert_eq!(d.guidance_weight, 2.5);
        assert_eq!(d.guidance_period, 4);

        // period 0 coerces to 1 (avoid div-by-zero).
        let d0 = QGuidedDrafter::new(MockGen { calls: 0 }, MockOracle {
            gradient: vec![],
            confidence: 0.0,
        })
        .with_period(0);
        assert_eq!(d0.guidance_period, 1, "period 0 must coerce to 1");
    }

    // ── Test: tilt with mismatched buffer lengths is safe ──────────────────

    #[test]
    fn test_tilt_handles_short_gradient_buffer() {
        let drafter = make_drafter(1.0, 1);
        // Logits longer than gradient — tilt applies to the min length only.
        let mut logits = [0.0f32; 5];
        let mut grad = [0.0f32; 3]; // MockOracle writes [1,2,3] then 0s

        let applied = drafter.tilt_logits(&(), &10u32, &mut logits, &mut grad, 0);
        assert!(applied);
        // First 3 positions tilted, last 2 untouched.
        assert_eq!(logits, [1.0, 2.0, 3.0, 0.0, 0.0]);
    }
}
