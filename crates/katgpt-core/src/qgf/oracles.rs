//! Concrete `QGradientOracle` implementations for existing critic-like
//! types (Plan 268 Phase 2 T5).
//!
//! Each oracle maps an existing value-function-like type onto the
//! [`QGradientOracle`] trait, providing `∇_a Q(s, a)` for the
//! [`QGuidedDrafter`](crate::qgf::drafter::QGuidedDrafter).
//!
//! # Tier Mapping
//!
//! | Tier | Oracle | Latency | Confidence |
//! |------|--------|---------|------------|
//! | Plasma | [`ActionBridgeOracle`] | < 100ns | 1.0 |
//! | Hot | [`LeoHeadOracle`] | < 1μs | 1.0 |
//! | Hot | [`DualLeoOracle`] (LEO+UVFA α-mix, Plan 467) | < 1μs | 1.0 |
//! | Warm | [`WarmTierOracle`] (wraps GPU delegate) | ~1ms | 1.0 |
//! | Cold | [`ColdTierOracle`] (wraps Q-table loader) | ~10ms | configurable |
//! | Freeze | [`NoGuidanceOracle`] / [`BfnProxyOracle`] | 0ns / ~1ms | 0.0 / 0.3 |
//!
//! # Sigmoid Not Softmax
//!
//! Gradient values are per-action-dimension scalars intended for **additive
//! logit shift** (`logits += w · g`), never softmax normalisation. See the
//! drafter module for how the tilt is applied.

use crate::traits::QGradientOracle;

// Re-export the no-op Freeze-tier oracle so consumers can reach it from the
// `qgf::oracles` module without crossing feature boundaries.
#[cfg(feature = "qgf_oracle")]
pub use crate::traits::NoGuidanceOracle;

// ──────────────────────────────────────────────────────────────────────────
// LeoHeadOracle — Hot tier (feature: leo_all_goals)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(feature = "leo_all_goals")]
mod leo_head_oracle {
    use super::*;
    use crate::traits::LeoHead;

    /// Hot-tier oracle wrapping a [`LeoHead`].
    ///
    /// The "gradient" of an all-goals Q-head w.r.t. the discrete action is
    /// simply the per-action Q-value vector itself (for the selected goal).
    /// Moving probability mass toward action `a` increases expected return by
    /// `Q(s, a)`, so `∇_a Q(s, a)[i] = Q(s, a_i)` — this is the natural
    /// discrete-action gradient.
    ///
    /// # Finite-Difference Interpretation
    ///
    /// For a discrete action space, the QGF paper's `∇_a Q` is interpreted as
    /// the finite-difference `[Q(s, a_i + ε) − Q(s, a_i − ε)] / (2ε)`, which
    /// for a one-hot action encoding reduces to `Q(s, a_i)` up to a constant
    /// baseline. We drop the baseline (it cancels under softmax-free sampling
    /// anyway — the tilt is rank-preserving).
    ///
    /// # Confidence
    ///
    /// `LeoHead` is a deterministic cached Q-table lookup → confidence `1.0`.
    /// This makes the adaptive guidance weight saturate to `~1.0` for
    /// high-quality LEO heads.
    pub struct LeoHeadOracle<H: LeoHead> {
        head: H,
        /// Which goal's Q-slice to use as the gradient source.
        goal_idx: usize,
    }

    impl<H: LeoHead> LeoHeadOracle<H> {
        /// Wrap a [`LeoHead`] and select which goal's Q-slice provides gradients.
        #[inline]
        pub fn new(head: H, goal_idx: usize) -> Self {
            Self { head, goal_idx }
        }

        /// Borrow the wrapped head.
        #[inline]
        pub fn head(&self) -> &H {
            &self.head
        }

        /// Current goal index used for gradient extraction.
        #[inline]
        pub const fn goal_idx(&self) -> usize {
            self.goal_idx
        }

        /// Switch to a different goal's Q-slice.
        #[inline]
        pub fn set_goal_idx(&mut self, goal_idx: usize) {
            self.goal_idx = goal_idx;
        }
    }

    impl<H: LeoHead> QGradientOracle for LeoHeadOracle<H> {
        type State = Vec<f32>;
        type Action = ();

        fn q_gradient_at(&self, state: &Self::State, _projected_action: &Self::Action) -> Vec<f32> {
            let all_q = self.head.all_goals_q(state);
            if self.goal_idx < self.head.goal_count() {
                self.head.q_for_goal(&all_q, self.goal_idx).to_vec()
            } else {
                Vec::new()
            }
        }

        fn q_gradient_into(
            &self,
            state: &Self::State,
            _projected_action: &Self::Action,
            out: &mut [f32],
        ) {
            let all_q = self.head.all_goals_q(state);
            if self.goal_idx >= self.head.goal_count() {
                out.fill(0.0);
                return;
            }
            let q_slice = self.head.q_for_goal(&all_q, self.goal_idx);
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = q_slice.get(i).copied().unwrap_or(0.0);
            }
        }

        // Deterministic cached lookup → confidence 1.0 (default).
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::traits::LeoHead;

        /// Mock LeoHead returning a known Q-matrix: goal 0 = [1, 2, 3, 4],
        /// goal 1 = [10, 20, 30, 40]. Used to verify the oracle extracts the
        /// correct goal slice and reports it as the gradient.
        struct MockLeo {
            goals: usize,
            actions: usize,
        }

        impl LeoHead for MockLeo {
            fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
                if self.goals == 2 && self.actions == 4 {
                    // Goal-major layout: [g0*a0..g0*a3, g1*a0..g1*a3]
                    vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
                } else {
                    (0..(self.goals * self.actions)).map(|i| i as f32).collect()
                }
            }

            #[inline]
            fn goal_count(&self) -> usize {
                self.goals
            }

            #[inline]
            fn action_count(&self) -> usize {
                self.actions
            }
        }

        #[test]
        fn test_leo_head_oracle_gradient() {
            let oracle = LeoHeadOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                1,
            );

            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());

            // Goal 1 slice = [10, 20, 30, 40]
            assert_eq!(grad, vec![10.0, 20.0, 30.0, 40.0]);
        }

        #[test]
        fn test_leo_head_oracle_into_matches_at() {
            let oracle = LeoHeadOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                0,
            );

            let state = vec![0.0; 8];
            let via_at = oracle.q_gradient_at(&state, &());

            let mut via_into = [0.0f32; 4];
            oracle.q_gradient_into(&state, &(), &mut via_into);

            assert_eq!(via_at, vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(&via_into, &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(via_at, via_into.to_vec());
        }

        #[test]
        fn test_leo_head_oracle_out_of_range_goal_zeros() {
            let oracle = LeoHeadOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                99,
            );

            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());

            assert!(grad.is_empty(), "out-of-range goal should return empty");

            let mut buf = [99.0f32; 4];
            oracle.q_gradient_into(&state, &(), &mut buf);
            assert_eq!(buf, [0.0; 4], "out-of-range goal should zero the buffer");
        }

        #[test]
        fn test_leo_head_oracle_confidence_is_one() {
            let oracle = LeoHeadOracle::new(
                MockLeo {
                    goals: 1,
                    actions: 2,
                },
                0,
            );
            // Deterministic cached lookup → confidence 1.0.
            assert_eq!(oracle.confidence(&vec![0.0; 2]), 1.0);
        }

        #[test]
        fn test_leo_head_oracle_set_goal() {
            let mut oracle = LeoHeadOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                0,
            );
            assert_eq!(oracle.goal_idx(), 0);

            let state = vec![0.0; 8];
            assert_eq!(oracle.q_gradient_at(&state, &()), vec![1.0, 2.0, 3.0, 4.0]);

            oracle.set_goal_idx(1);
            assert_eq!(oracle.goal_idx(), 1);
            assert_eq!(
                oracle.q_gradient_at(&state, &()),
                vec![10.0, 20.0, 30.0, 40.0]
            );
        }
    }
}

#[cfg(feature = "leo_all_goals")]
pub use leo_head_oracle::LeoHeadOracle;

// ──────────────────────────────────────────────────────────────────────────
// FlowFieldOracle — Plasma/Hot tier (feature: flow_field_nav)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(feature = "flow_field_nav")]
mod flow_field_oracle {
    use super::*;
    use crate::flow::FlowField;

    /// Plasma/Hot-tier oracle wrapping an owned [`FlowField`].
    ///
    /// The "gradient" of a flow field at a position is simply the `(dx, dy)`
    /// flow vector at that cell — it already *is* the negative gradient of the
    /// FFT-smoothed potential field. This is the variance-reduced gradient that
    /// QGF's FFT smoothing provides.
    ///
    /// # Action / State Mapping
    ///
    /// - `State` = `()` (the flow field is pre-computed and stateless)
    /// - `Action` = `(u16, u16)` — the projected position `(x, y)` at which to
    ///   look up the flow vector.
    ///
    /// The gradient buffer receives `[dx, dy]` — a 2-element tilt suitable for
    /// steering-style action spaces.
    ///
    /// # Confidence
    ///
    /// `1.0` — the flow field is the deterministic output of an FFT-smoothed
    /// potential, so the gradient is exact (no sampling noise).
    pub struct FlowFieldOracle {
        field: FlowField,
    }

    impl FlowFieldOracle {
        /// Wrap an owned [`FlowField`].
        #[inline]
        pub fn new(field: FlowField) -> Self {
            Self { field }
        }

        /// Borrow the underlying flow field.
        #[inline]
        pub fn field(&self) -> &FlowField {
            &self.field
        }
    }

    impl QGradientOracle for FlowFieldOracle {
        type State = ();
        type Action = (u16, u16);

        fn q_gradient_at(&self, _state: &Self::State, projected_action: &Self::Action) -> Vec<f32> {
            let (x, y) = *projected_action;
            let (dx, dy) = self.field.lookup(x, y);
            vec![dx, dy]
        }

        fn q_gradient_into(
            &self,
            _state: &Self::State,
            projected_action: &Self::Action,
            out: &mut [f32],
        ) {
            let (x, y) = *projected_action;
            let (dx, dy) = self.field.lookup(x, y);
            if let Some(slot) = out.get_mut(0) {
                *slot = dx;
            }
            if let Some(slot) = out.get_mut(1) {
                *slot = dy;
            }
            for slot in out.iter_mut().skip(2) {
                *slot = 0.0;
            }
        }

        // Deterministic FFT-smoothed lookup → confidence 1.0 (default).
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::flow::FlowField;

        fn make_field() -> FlowField {
            let mut f = FlowField::new(3, 3);
            // Set a known flow at (1, 1) → (0.5, -0.5).
            f.set_flow(1, 1, 0.5, -0.5);
            // And a unit flow at (2, 0) → (1.0, 0.0).
            f.set_flow(2, 0, 1.0, 0.0);
            f
        }

        #[test]
        fn test_flow_field_oracle_gradient() {
            let oracle = FlowFieldOracle::new(make_field());

            let g = oracle.q_gradient_at(&(), &(1, 1));
            assert_eq!(g, vec![0.5, -0.5]);

            let g2 = oracle.q_gradient_at(&(), &(2, 0));
            assert_eq!(g2, vec![1.0, 0.0]);
        }

        #[test]
        fn test_flow_field_oracle_into_matches_at() {
            let oracle = FlowFieldOracle::new(make_field());

            let via_at = oracle.q_gradient_at(&(), &(1, 1));

            let mut via_into = [0.0f32; 2];
            oracle.q_gradient_into(&(), &(1, 1), &mut via_into);

            assert_eq!(via_at, via_into.to_vec());
        }

        #[test]
        fn test_flow_field_oracle_blocked_cell_zero() {
            let oracle = FlowFieldOracle::new(make_field());

            // (0, 0) was never set → zero flow (blocked).
            let g = oracle.q_gradient_at(&(), &(0, 0));
            assert_eq!(g, vec![0.0, 0.0]);
        }

        #[test]
        fn test_flow_field_oracle_out_of_bounds_zero() {
            let oracle = FlowFieldOracle::new(make_field());

            // (99, 99) is out of bounds → lookup returns (0, 0).
            let g = oracle.q_gradient_at(&(), &(99, 99));
            assert_eq!(g, vec![0.0, 0.0]);
        }

        #[test]
        fn test_flow_field_oracle_confidence_is_one() {
            let oracle = FlowFieldOracle::new(make_field());
            assert_eq!(oracle.confidence(&()), 1.0);
        }

        #[test]
        fn test_flow_field_oracle_long_buffer_pads_zero() {
            let oracle = FlowFieldOracle::new(make_field());
            let mut buf = [99.0f32; 5];
            oracle.q_gradient_into(&(), &(2, 0), &mut buf);
            // [dx, dy, 0, 0, 0]
            assert_eq!(buf, [1.0, 0.0, 0.0, 0.0, 0.0]);
        }
    }
}

#[cfg(feature = "flow_field_nav")]
pub use flow_field_oracle::FlowFieldOracle;

// ──────────────────────────────────────────────────────────────────────────
// DualLeoOracle — Hot tier (feature: leo_all_goals + dual_leo)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "leo_all_goals", feature = "dual_leo"))]
mod dual_leo_oracle {
    use super::*;
    use crate::traits::{DualLeoMixer, LeoHead};

    /// Hot-tier oracle fusing a LEO teacher head + a UVFA student head via
    /// [`DualLeoMixer`] at the gradient level.
    ///
    /// The "gradient" of an all-goals Q-head w.r.t. the discrete action is
    /// the per-action Q-slice for the selected goal (see [`LeoHeadOracle`] for
    /// the finite-difference rationale). This oracle computes that slice for
    /// both heads and blends them with [`DualLeoMixer::combine_into`] before
    /// returning, producing a gradient that is a **linear combination** of the
    /// two heads' Q-values:
    ///
    /// ```text
    /// grad_mix[i] = α · Q_leo(s, a_i) + (1-α) · Q_uvfa(s, a_i)
    /// ```
    ///
    /// # Plan 460 design invariant (encoded by construction)
    ///
    /// **No operator sits between the [`DualLeoMixer::combine_into`] and the
    /// QGF consumer.** Plan 459's pre-max fusion failed because `max_a(·)`
    /// sat between the mix and the FFT consumer — and
    /// `max_a(α·x + (1-α)·y) ≠ α·max_a(x) + (1-α)·max_a(y)`, washing out the
    /// α-mix. [`LeoHeadOracle`]'s gradient path has no max-pool (the gradient
    /// IS the per-action Q-slice), so the invariant holds by construction
    /// here — a mixer-produced gradient is consumed directly as a tilt, with
    /// no nonlinearity in between.
    ///
    /// # Confidence
    ///
    /// Both heads are deterministic cached lookups → confidence `1.0`.
    /// Matches [`LeoHeadOracle`]'s contract. See Proposal 007 caveat 5 for
    /// the inherited "determinism = quality" lie.
    ///
    /// [`LeoHeadOracle`]: super::LeoHeadOracle
    pub struct DualLeoOracle<H1, H2, M>
    where
        H1: LeoHead,
        H2: LeoHead,
        M: DualLeoMixer,
    {
        head_leo: H1,
        head_uvfa: H2,
        mixer: M,
        alpha: f32,
        goal_idx: usize,
    }

    impl<H1, H2, M> DualLeoOracle<H1, H2, M>
    where
        H1: LeoHead,
        H2: LeoHead,
        M: DualLeoMixer,
    {
        /// Wrap a LEO teacher + UVFA student + mixer. Selects which goal's
        /// Q-slice provides the fused gradient.
        ///
        /// The `alpha` is forwarded to [`DualLeoMixer::combine_into`]; the
        /// mixer's [`crate::traits::ActingMode`] decides how the two slices
        /// are combined (`Lc` = α-blend, `LeoOnly` = ignore UVFA, `UvfaOnly` =
        /// ignore LEO, `Max`/`Min` = element-wise max/min).
        #[inline]
        pub fn new(head_leo: H1, head_uvfa: H2, mixer: M, alpha: f32, goal_idx: usize) -> Self {
            Self {
                head_leo,
                head_uvfa,
                mixer,
                alpha,
                goal_idx,
            }
        }

        /// Borrow the LEO teacher head.
        #[inline]
        pub fn head_leo(&self) -> &H1 {
            &self.head_leo
        }

        /// Borrow the UVFA student head.
        #[inline]
        pub fn head_uvfa(&self) -> &H2 {
            &self.head_uvfa
        }

        /// Borrow the mixer.
        #[inline]
        pub fn mixer(&self) -> &M {
            &self.mixer
        }

        /// Current goal index used for gradient extraction.
        #[inline]
        pub const fn goal_idx(&self) -> usize {
            self.goal_idx
        }

        /// Current α forwarded to the mixer.
        #[inline]
        pub const fn alpha(&self) -> f32 {
            self.alpha
        }
    }

    impl<H1, H2, M> QGradientOracle for DualLeoOracle<H1, H2, M>
    where
        H1: LeoHead,
        H2: LeoHead,
        M: DualLeoMixer,
    {
        type State = Vec<f32>;
        type Action = ();

        fn q_gradient_at(&self, state: &Self::State, _projected_action: &Self::Action) -> Vec<f32> {
            // Plan 460 invariant: mix at gradient level (linear in Q). No
            // operator sits between this mix and the QGF consumer.
            let q_leo_all = self.head_leo.all_goals_q(state);
            let q_uvfa_all = self.head_uvfa.all_goals_q(state);
            if self.goal_idx >= self.head_leo.goal_count()
                || self.goal_idx >= self.head_uvfa.goal_count()
            {
                return Vec::new();
            }
            let q_leo = self.head_leo.q_for_goal(&q_leo_all, self.goal_idx);
            let q_uvfa = self.head_uvfa.q_for_goal(&q_uvfa_all, self.goal_idx);
            // Both goal slices must have equal length; mixer assumes parallel slices.
            let n = q_leo.len().min(q_uvfa.len());
            let mut out = vec![0.0f32; n];
            self.mixer
                .combine_into(&mut out, &q_leo[..n], &q_uvfa[..n], self.alpha);
            out
        }

        fn q_gradient_into(
            &self,
            state: &Self::State,
            _projected_action: &Self::Action,
            out: &mut [f32],
        ) {
            let q_leo_all = self.head_leo.all_goals_q(state);
            let q_uvfa_all = self.head_uvfa.all_goals_q(state);
            if self.goal_idx >= self.head_leo.goal_count()
                || self.goal_idx >= self.head_uvfa.goal_count()
            {
                out.fill(0.0);
                return;
            }
            let q_leo = self.head_leo.q_for_goal(&q_leo_all, self.goal_idx);
            let q_uvfa = self.head_uvfa.q_for_goal(&q_uvfa_all, self.goal_idx);
            let n = q_leo.len().min(q_uvfa.len()).min(out.len());
            self.mixer
                .combine_into(&mut out[..n], &q_leo[..n], &q_uvfa[..n], self.alpha);
            for slot in &mut out[n..] {
                *slot = 0.0;
            }
        }

        // Deterministic cached lookup → confidence 1.0 (default).
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::traits::{ActingMode, LeoHead};

        /// Mock LEO head — same matrix as `LeoHeadOracle`'s tests so we can
        /// verify α=1.0 bit-identity directly. Goal-major layout:
        /// goal 0 = [1, 2, 3, 4], goal 1 = [10, 20, 30, 40].
        struct MockLeo {
            goals: usize,
            actions: usize,
        }

        impl LeoHead for MockLeo {
            fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
                if self.goals == 2 && self.actions == 4 {
                    vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
                } else {
                    (0..(self.goals * self.actions)).map(|i| i as f32).collect()
                }
            }

            #[inline]
            fn goal_count(&self) -> usize {
                self.goals
            }

            #[inline]
            fn action_count(&self) -> usize {
                self.actions
            }
        }

        /// Mock UVFA head — DIFFERENT Q-values than `MockLeo`, so we can verify
        /// the α-mix is actually combining the two heads. Goal-major layout:
        /// goal 0 = [100, 200, 300, 400], goal 1 = [1000, 2000, 3000, 4000].
        struct MockUvfa {
            goals: usize,
            actions: usize,
        }

        impl LeoHead for MockUvfa {
            fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
                if self.goals == 2 && self.actions == 4 {
                    vec![100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0, 3000.0, 4000.0]
                } else {
                    (0..(self.goals * self.actions))
                        .map(|i| (i as f32) * 10.0)
                        .collect()
                }
            }

            #[inline]
            fn goal_count(&self) -> usize {
                self.goals
            }

            #[inline]
            fn action_count(&self) -> usize {
                self.actions
            }
        }

        /// `DualLeoMixer` that delegates to the default `combine_into`
        /// (which dispatches on `ActingMode`). Tests pick the mode by
        /// overriding `acting_mode`.
        struct ModeMixer {
            mode: ActingMode,
        }

        impl DualLeoMixer for ModeMixer {
            fn acting_mode(&self) -> ActingMode {
                self.mode
            }
        }

        fn lc_mixer() -> ModeMixer {
            ModeMixer {
                mode: ActingMode::Lc,
            }
        }
        fn leo_only_mixer() -> ModeMixer {
            ModeMixer {
                mode: ActingMode::LeoOnly,
            }
        }
        fn uvfa_only_mixer() -> ModeMixer {
            ModeMixer {
                mode: ActingMode::UvfaOnly,
            }
        }

        #[test]
        fn test_dual_leo_oracle_lc_alpha_half_is_exact_blend() {
            // Lc mode at α=0.5: grad[i] = 0.5·leo[i] + 0.5·uvfa[i].
            let oracle = DualLeoOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                MockUvfa {
                    goals: 2,
                    actions: 4,
                },
                lc_mixer(),
                0.5,
                0,
            );

            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());

            // Goal 0 leo = [1, 2, 3, 4], uvfa = [100, 200, 300, 400]
            // 0.5·leo + 0.5·uvfa = [50.5, 101, 151.5, 202]
            assert_eq!(grad, vec![50.5, 101.0, 151.5, 202.0]);
        }

        #[test]
        fn test_dual_leo_oracle_lc_alpha_general() {
            // Lc mode at arbitrary α: grad[i] = α·leo[i] + (1-α)·uvfa[i].
            let alpha = 0.3;
            let oracle = DualLeoOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                MockUvfa {
                    goals: 2,
                    actions: 4,
                },
                lc_mixer(),
                alpha,
                1,
            );

            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());

            // Goal 1 leo = [10, 20, 30, 40], uvfa = [1000, 2000, 3000, 4000]
            // α·leo + (1-α)·uvfa at α=0.3 →
            // [0.3·10 + 0.7·1000, 0.3·20 + 0.7·2000, ...]
            let expected: Vec<f32> = [10.0f32, 20.0, 30.0, 40.0]
                .iter()
                .zip([1000.0f32, 2000.0, 3000.0, 4000.0].iter())
                .map(|(&l, &u)| alpha * l + (1.0 - alpha) * u)
                .collect();
            assert_eq!(grad, expected);
        }

        #[test]
        fn test_dual_leo_oracle_leo_only_bit_identical_to_leo_head_oracle() {
            // G1 bit-identity: α=anything with LeoOnly → matches LeoHeadOracle
            // on the same LEO head + goal (LeoOnly ignores α and UVFA).
            use super::super::LeoHeadOracle;

            let leo = MockLeo {
                goals: 2,
                actions: 4,
            };
            let leo_clone = MockLeo {
                goals: 2,
                actions: 4,
            };
            let single = LeoHeadOracle::new(leo, 1);
            let dual = DualLeoOracle::new(
                leo_clone,
                MockUvfa {
                    goals: 2,
                    actions: 4,
                },
                leo_only_mixer(),
                0.0, // α is irrelevant under LeoOnly, but the test uses 0.0
                1,
            );

            let state = vec![0.0; 8];
            let g_single = single.q_gradient_at(&state, &());
            let g_dual = dual.q_gradient_at(&state, &());
            assert_eq!(g_single, vec![10.0, 20.0, 30.0, 40.0]);
            assert_eq!(g_dual, g_single, "LeoOnly must bit-match LeoHeadOracle");
        }

        #[test]
        fn test_dual_leo_oracle_uvfa_only_bit_identical_to_leo_head_oracle_on_uvfa() {
            // G1 bit-identity: UvfaOnly → matches LeoHeadOracle wrapping the
            // UVFA head (UvfaOnly ignores α and LEO).
            use super::super::LeoHeadOracle;

            let uvfa = MockUvfa {
                goals: 2,
                actions: 4,
            };
            let uvfa_clone = MockUvfa {
                goals: 2,
                actions: 4,
            };
            let single_on_uvfa = LeoHeadOracle::new(uvfa, 0);
            let dual = DualLeoOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                uvfa_clone,
                uvfa_only_mixer(),
                1.0, // α is irrelevant under UvfaOnly
                0,
            );

            let state = vec![0.0; 8];
            let g_single = single_on_uvfa.q_gradient_at(&state, &());
            let g_dual = dual.q_gradient_at(&state, &());
            assert_eq!(g_single, vec![100.0, 200.0, 300.0, 400.0]);
            assert_eq!(
                g_dual, g_single,
                "UvfaOnly must bit-match LeoHeadOracle on UVFA"
            );
        }

        #[test]
        fn test_dual_leo_oracle_into_matches_at() {
            // `q_gradient_into` and `q_gradient_at` must agree.
            let oracle = DualLeoOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                MockUvfa {
                    goals: 2,
                    actions: 4,
                },
                lc_mixer(),
                0.5,
                1,
            );

            let state = vec![0.0; 8];
            let via_at = oracle.q_gradient_at(&state, &());

            let mut via_into = [0.0f32; 4];
            oracle.q_gradient_into(&state, &(), &mut via_into);

            assert_eq!(via_at, via_into.to_vec());
        }

        #[test]
        fn test_dual_leo_oracle_into_long_buffer_pads_zero() {
            // Buffer longer than the slice → tail must be zero-filled.
            let oracle = DualLeoOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                MockUvfa {
                    goals: 2,
                    actions: 4,
                },
                lc_mixer(),
                0.5,
                0,
            );

            let state = vec![0.0; 8];
            let mut buf = [99.0f32; 7];
            oracle.q_gradient_into(&state, &(), &mut buf);

            // First 4 slots = mix; last 3 must be 0.
            assert_eq!(buf[0], 50.5);
            assert_eq!(buf[1], 101.0);
            assert_eq!(buf[2], 151.5);
            assert_eq!(buf[3], 202.0);
            assert_eq!(buf[4], 0.0);
            assert_eq!(buf[5], 0.0);
            assert_eq!(buf[6], 0.0);
        }

        #[test]
        fn test_dual_leo_oracle_out_of_range_goal_zeros() {
            // goal_idx ≥ goal_count on EITHER head → empty gradient + zero buffer.
            let oracle = DualLeoOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                MockUvfa {
                    goals: 2,
                    actions: 4,
                },
                lc_mixer(),
                0.5,
                99,
            );

            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());
            assert!(grad.is_empty(), "out-of-range goal should return empty");

            let mut buf = [99.0f32; 4];
            oracle.q_gradient_into(&state, &(), &mut buf);
            assert_eq!(buf, [0.0; 4], "out-of-range goal should zero the buffer");
        }

        #[test]
        fn test_dual_leo_oracle_confidence_is_one() {
            // Deterministic cached lookup → confidence 1.0 (default impl).
            let oracle = DualLeoOracle::new(
                MockLeo {
                    goals: 1,
                    actions: 2,
                },
                MockUvfa {
                    goals: 1,
                    actions: 2,
                },
                lc_mixer(),
                0.5,
                0,
            );
            assert_eq!(oracle.confidence(&vec![0.0; 2]), 1.0);
        }

        #[test]
        fn test_dual_leo_oracle_accessor_roundtrip() {
            // `new` stores + accessors return what was stored.
            let oracle = DualLeoOracle::new(
                MockLeo {
                    goals: 2,
                    actions: 4,
                },
                MockUvfa {
                    goals: 2,
                    actions: 4,
                },
                lc_mixer(),
                0.3,
                1,
            );
            assert_eq!(oracle.goal_idx(), 1);
            assert!((oracle.alpha() - 0.3).abs() < 1e-6);
            assert_eq!(oracle.head_leo().goal_count(), 2);
            assert_eq!(oracle.head_uvfa().action_count(), 4);
            assert_eq!(oracle.mixer().acting_mode(), ActingMode::Lc);
        }

        /// G2 perf gate (Plan 460 perf-honesty lesson — single-shot macOS
        /// timing is unreliable <10ms; use median-of-3 trials × 1000-iter
        /// batches). Measures `DualLeoOracle` vs `LeoHeadOracle` on both
        /// the allocating (`q_gradient_at`) and zero-alloc (`q_gradient_into`)
        /// paths.
        ///
        /// Structural cost floor for `DualLeoOracle`: 2 head lookups + 1
        /// `combine_into`. The `_into` path additionally avoids the output
        /// `Vec` allocation, so its dual/single ratio is the cleaner signal.
        ///
        /// `#[ignore]` so it doesn't slow normal test runs — invoke with
        /// `cargo test ... -- --ignored` or run via the bench report.
        #[test]
        #[ignore]
        fn bench_dual_leo_oracle_g2_perf() {
            use super::super::LeoHeadOracle;
            use std::time::Instant;

            // Larger head shape than the bit-identity tests so the per-call
            // cost is measurable: 8 goals × 32 actions = 256-element Q-slice.
            // (MockLeo/MockUvfa fall back to their generic branch at this shape.)
            let leo = MockLeo {
                goals: 8,
                actions: 32,
            };
            let uvfa = MockUvfa {
                goals: 8,
                actions: 32,
            };
            let single = LeoHeadOracle::new(
                MockLeo {
                    goals: 8,
                    actions: 32,
                },
                0,
            );
            let dual = DualLeoOracle::new(leo, uvfa, lc_mixer(), 0.5, 0);

            let state = vec![0.0; 256];
            let iters = 1000;
            let mut buf = vec![0.0f32; 256];

            // Median-of-3 trials (Plan 460 lesson).
            let median = |samples_ns: [u128; 3]| {
                let mut s = samples_ns;
                s.sort();
                s[1] as f64
            };

            // --- q_gradient_at (allocating path) ---
            let mut single_at = [0u128; 3];
            let mut dual_at = [0u128; 3];
            for slot in &mut single_at {
                let start = Instant::now();
                for _ in 0..iters {
                    let g = single.q_gradient_at(&state, &());
                    std::hint::black_box(&g);
                }
                *slot = start.elapsed().as_nanos();
            }
            for slot in &mut dual_at {
                let start = Instant::now();
                for _ in 0..iters {
                    let g = dual.q_gradient_at(&state, &());
                    std::hint::black_box(&g);
                }
                *slot = start.elapsed().as_nanos();
            }

            // --- q_gradient_into (zero-alloc hot path — G4) ---
            let mut single_into = [0u128; 3];
            let mut dual_into = [0u128; 3];
            for slot in &mut single_into {
                let start = Instant::now();
                for _ in 0..iters {
                    single.q_gradient_into(&state, &(), &mut buf);
                    std::hint::black_box(&buf);
                }
                *slot = start.elapsed().as_nanos();
            }
            for slot in &mut dual_into {
                let start = Instant::now();
                for _ in 0..iters {
                    dual.q_gradient_into(&state, &(), &mut buf);
                    std::hint::black_box(&buf);
                }
                *slot = start.elapsed().as_nanos();
            }

            let sm_at = median(single_at);
            let dm_at = median(dual_at);
            let sm_into = median(single_into);
            let dm_into = median(dual_into);
            let ratio_at = dm_at / sm_at;
            let ratio_into = dm_into / sm_into;

            eprintln!(
                "g2_perf: q_gradient_at   single={sm_at}ns dual={dm_at}ns ratio={ratio_at:.3}×\n\
                 g2_perf: q_gradient_into single={sm_into}ns dual={dm_into}ns ratio={ratio_into:.3}×\n\
                 g2_perf: gate=≤1.5× on the _into (zero-alloc) path; _at path is informational"
            );

            // Gate applies to the zero-alloc `_into` path (the production hot
            // path per G4). The `_at` path's ratio is dominated by output Vec
            // allocation overhead and is reported but not gated. See
            // `.benchmarks/467_qgf_dual_leo_oracle_goat.md` §G2 for analysis.
            assert!(
                ratio_into <= 1.5,
                "G2 FAIL on _into path: dual/single ratio {ratio_into:.3} > 1.5× \
                 (single={sm_into}ns, dual={dm_into}ns)"
            );
        }
    }
}

#[cfg(all(feature = "leo_all_goals", feature = "dual_leo"))]
pub use dual_leo_oracle::DualLeoOracle;

// ──────────────────────────────────────────────────────────────────────────
// CommitDualLeoOracle — Hot tier (feature: leo_all_goals + dual_leo)
// Sibling to `DualLeoOracle`. Fuses LEO teacher + UVFA student via a
// **sigmoid confidence gate** rather than a linear α-blend, addressing
// Issue 188's three failure modes of average-then-argmax.
// ──────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "leo_all_goals", feature = "dual_leo"))]
mod commit_dual_leo_oracle {
    use super::*;
    use crate::sigmoid;
    use crate::traits::LeoHead;

    /// Hot-tier oracle fusing a LEO teacher + UVFA student via a **sigmoid
    /// confidence gate** (Issue 188).
    ///
    /// Unlike [`DualLeoOracle`] (which does a fixed-α linear blend of the
    /// two heads' Q-slices, then lets the QGF consumer argmax over the
    /// blended slice), this oracle gates the blend by *which head is more
    /// confident on the current state*:
    ///
    /// ```text
    /// conf_t = max_a Q_leo(s, a)
    /// conf_s = max_a Q_uvfa(s, a)
    /// gate   = σ(β · (conf_t − conf_s))
    /// grad[a] = gate · Q_leo(s, a) + (1 − gate) · Q_uvfa(s, a)
    /// ```
    ///
    /// The `β` parameter spans a continuous design space:
    ///
    /// - **β = 0**: `gate = 0.5` always → reduces to [`DualLeoOracle`] at
    ///   α = 0.5 (the G1 regression sanity check).
    /// - **0 < β < ∞**: sigmoid-gated soft switch — "use sigmoid instead of
    ///   average" (Issue 188 hypothesis #1).
    /// - **β → ∞**: hard commit-then-fuse — "wrap up later instead of each"
    ///   (Issue 188 hypothesis #2). Approached asymptotically by β = 64 or
    ///   larger (sigmoid saturates by then for typical Q-value magnitudes).
    ///
    /// # Plan 460 invariant
    ///
    /// No operator sits between this gradient and the QGF consumer. The
    /// sigmoid gate is *part of the gradient computation itself* (it picks
    /// the blend weights), so the consumer still receives a per-action
    /// Q-slice directly — no nonlinearity is re-introduced downstream.
    ///
    /// # What this does NOT fix
    ///
    /// If one head is untrained noise (Bench 558's setup), the gate still
    /// injects noise into the gradient whenever `conf_s ≈ conf_t` (gate ≈
    /// 0.5). The fix for that is a trained student, not a different gate.
    /// See Issue 188 §"What this does NOT do".
    ///
    /// # Confidence
    ///
    /// Both heads are deterministic cached lookups → confidence `1.0`.
    /// Matches [`DualLeoOracle`]'s contract (and inherits its "determinism =
    /// quality" lie — Proposal 007 caveat 5).
    ///
    /// [`DualLeoOracle`]: super::DualLeoOracle
    pub struct CommitDualLeoOracle<H1, H2>
    where
        H1: LeoHead,
        H2: LeoHead,
    {
        head_leo: H1,
        head_uvfa: H2,
        /// Gate sharpness. 0 = linear α=0.5 blend; ∞ = hard commit-then-fuse.
        beta: f32,
        goal_idx: usize,
    }

    impl<H1, H2> CommitDualLeoOracle<H1, H2>
    where
        H1: LeoHead,
        H2: LeoHead,
    {
        /// Wrap a LEO teacher + UVFA student with a sigmoid confidence gate.
        ///
        /// `beta` controls gate sharpness:
        /// - `0.0` reduces to α=0.5 linear blend (matches [`DualLeoOracle`] at
        ///   α=0.5 — use as a G1 regression check).
        /// - `4.0`–`16.0` is a typical soft-switch range.
        /// - `≥ 64.0` approaches hard commit-then-fuse (sigmoid saturated).
        #[inline]
        pub fn new(head_leo: H1, head_uvfa: H2, beta: f32, goal_idx: usize) -> Self {
            Self {
                head_leo,
                head_uvfa,
                beta,
                goal_idx,
            }
        }

        /// Borrow the LEO teacher head.
        #[inline]
        pub fn head_leo(&self) -> &H1 {
            &self.head_leo
        }

        /// Borrow the UVFA student head.
        #[inline]
        pub fn head_uvfa(&self) -> &H2 {
            &self.head_uvfa
        }

        /// Current β (gate sharpness).
        #[inline]
        pub const fn beta(&self) -> f32 {
            self.beta
        }

        /// Current goal index used for gradient extraction.
        #[inline]
        pub const fn goal_idx(&self) -> usize {
            self.goal_idx
        }

        /// Switch to a different goal's Q-slice.
        #[inline]
        pub fn set_goal_idx(&mut self, goal_idx: usize) {
            self.goal_idx = goal_idx;
        }

        /// Compute the sigmoid gate value for a given (q_leo, q_uvfa) slice
        /// pair. Exposed for benchmarks / diagnostics; the gradient path
        /// calls this inline.
        ///
        /// Returns `σ(β · (max(q_leo) − max(q_uvfa)))`.
        #[inline]
        pub fn gate_value(&self, q_leo: &[f32], q_uvfa: &[f32]) -> f32 {
            let conf_t = q_leo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let conf_s = q_uvfa.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            sigmoid(self.beta * (conf_t - conf_s))
        }
    }

    impl<H1, H2> QGradientOracle for CommitDualLeoOracle<H1, H2>
    where
        H1: LeoHead,
        H2: LeoHead,
    {
        type State = Vec<f32>;
        type Action = ();

        fn q_gradient_at(&self, state: &Self::State, _projected_action: &Self::Action) -> Vec<f32> {
            let q_leo_all = self.head_leo.all_goals_q(state);
            let q_uvfa_all = self.head_uvfa.all_goals_q(state);
            if self.goal_idx >= self.head_leo.goal_count()
                || self.goal_idx >= self.head_uvfa.goal_count()
            {
                return Vec::new();
            }
            let q_leo = self.head_leo.q_for_goal(&q_leo_all, self.goal_idx);
            let q_uvfa = self.head_uvfa.q_for_goal(&q_uvfa_all, self.goal_idx);
            let n = q_leo.len().min(q_uvfa.len());
            let gate = self.gate_value(&q_leo[..n], &q_uvfa[..n]);
            let mut out = vec![0.0f32; n];
            for (o, (&ql, &qu)) in out
                .iter_mut()
                .zip(q_leo[..n].iter().zip(q_uvfa[..n].iter()))
            {
                *o = gate * ql + (1.0 - gate) * qu;
            }
            out
        }

        fn q_gradient_into(
            &self,
            state: &Self::State,
            _projected_action: &Self::Action,
            out: &mut [f32],
        ) {
            let q_leo_all = self.head_leo.all_goals_q(state);
            let q_uvfa_all = self.head_uvfa.all_goals_q(state);
            if self.goal_idx >= self.head_leo.goal_count()
                || self.goal_idx >= self.head_uvfa.goal_count()
            {
                out.fill(0.0);
                return;
            }
            let q_leo = self.head_leo.q_for_goal(&q_leo_all, self.goal_idx);
            let q_uvfa = self.head_uvfa.q_for_goal(&q_uvfa_all, self.goal_idx);
            let n = q_leo.len().min(q_uvfa.len()).min(out.len());
            let gate = self.gate_value(&q_leo[..n], &q_uvfa[..n]);
            for (o, (&ql, &qu)) in out
                .iter_mut()
                .zip(q_leo[..n].iter().zip(q_uvfa[..n].iter()))
            {
                *o = gate * ql + (1.0 - gate) * qu;
            }
            for slot in &mut out[n..] {
                *slot = 0.0;
            }
        }

        // Deterministic cached lookup → confidence 1.0 (default).
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::traits::LeoHead;

        /// Mock LEO head — same matrix as `DualLeoOracle`'s tests so we can
        /// verify β=0 reduces to α=0.5 blend bit-identically. Goal-major:
        /// goal 0 = [1, 2, 3, 4], goal 1 = [10, 20, 30, 40].
        struct MockLeo {
            goals: usize,
            actions: usize,
        }

        impl LeoHead for MockLeo {
            fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
                if self.goals == 2 && self.actions == 4 {
                    vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
                } else {
                    (0..(self.goals * self.actions)).map(|i| i as f32).collect()
                }
            }

            #[inline]
            fn goal_count(&self) -> usize {
                self.goals
            }

            #[inline]
            fn action_count(&self) -> usize {
                self.actions
            }
        }

        /// Mock UVFA head — DIFFERENT Q-values, higher confidence.
        /// goal 0 = [100, 200, 300, 400], goal 1 = [1000, 2000, 3000, 4000].
        struct MockUvfa {
            goals: usize,
            actions: usize,
        }

        impl LeoHead for MockUvfa {
            fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
                if self.goals == 2 && self.actions == 4 {
                    vec![100.0, 200.0, 300.0, 400.0, 1000.0, 2000.0, 3000.0, 4000.0]
                } else {
                    (0..(self.goals * self.actions))
                        .map(|i| (i as f32) * 10.0)
                        .collect()
                }
            }

            #[inline]
            fn goal_count(&self) -> usize {
                self.goals
            }

            #[inline]
            fn action_count(&self) -> usize {
                self.actions
            }
        }

        #[test]
        fn test_commit_dual_leo_beta_zero_reduces_to_alpha_half_blend() {
            // G1 regression: β=0 → gate=σ(0)=0.5 → 0.5·leo + 0.5·uvfa.
            // This must bit-match DualLeoOracle at α=0.5 on the same heads.
            let oracle = CommitDualLeoOracle::new(
                MockLeo { goals: 2, actions: 4 },
                MockUvfa { goals: 2, actions: 4 },
                0.0, // β=0
                0,
            );
            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());
            // Goal 0: leo=[1,2,3,4], uvfa=[100,200,300,400]
            // 0.5·leo + 0.5·uvfa = [50.5, 101, 151.5, 202]
            assert_eq!(grad, vec![50.5, 101.0, 151.5, 202.0]);
        }

        #[test]
        fn test_commit_dual_leo_gate_picks_higher_confidence_head() {
            // UVFA is much more confident than LEO on every goal (max 400 vs
            // max 4 on goal 0). For any β > 0, Δconf = 4 − 400 = −396 < 0
            // → gate = σ(β·(−396)) → 0 as β grows → blend → uvfa-only.
            // At β=1: gate = σ(−396) ≈ 0 (underflows to exactly 0.0 in f32).
            let oracle = CommitDualLeoOracle::new(
                MockLeo { goals: 2, actions: 4 },
                MockUvfa { goals: 2, actions: 4 },
                1.0,
                0,
            );
            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());
            // gate ≈ 0 → grad ≈ uvfa only.
            assert_eq!(grad, vec![100.0, 200.0, 300.0, 400.0]);
        }

        #[test]
        fn test_commit_dual_leo_bit_identical_to_dual_leo_oracle_at_alpha_half() {
            // Cross-oracle G1: at β=0, CommitDualLeoOracle must produce the
            // same gradient as DualLeoOracle at α=0.5 on the same heads.
            use super::super::DualLeoOracle;
            use crate::traits::{ActingMode, DualLeoMixer};

            struct LcMixer;
            impl DualLeoMixer for LcMixer {
                fn acting_mode(&self) -> ActingMode {
                    ActingMode::Lc
                }
            }

            let commit = CommitDualLeoOracle::new(
                MockLeo { goals: 2, actions: 4 },
                MockUvfa { goals: 2, actions: 4 },
                0.0,
                1,
            );
            let blend = DualLeoOracle::new(
                MockLeo { goals: 2, actions: 4 },
                MockUvfa { goals: 2, actions: 4 },
                LcMixer,
                0.5,
                1,
            );
            let state = vec![0.0; 8];
            let g_commit = commit.q_gradient_at(&state, &());
            let g_blend = blend.q_gradient_at(&state, &());
            assert_eq!(
                g_commit, g_blend,
                "β=0 must bit-match DualLeoOracle at α=0.5"
            );
        }

        #[test]
        fn test_commit_dual_leo_into_matches_at() {
            // `q_gradient_into` and `q_gradient_at` must agree.
            let oracle = CommitDualLeoOracle::new(
                MockLeo { goals: 2, actions: 4 },
                MockUvfa { goals: 2, actions: 4 },
                2.0,
                1,
            );
            let state = vec![0.0; 8];
            let via_at = oracle.q_gradient_at(&state, &());
            let mut via_into = [0.0f32; 4];
            oracle.q_gradient_into(&state, &(), &mut via_into);
            assert_eq!(via_at, via_into.to_vec());
        }

        #[test]
        fn test_commit_dual_leo_out_of_range_goal_zeros() {
            // Out-of-range goal → empty gradient (mirrors DualLeoOracle edge).
            let oracle = CommitDualLeoOracle::new(
                MockLeo { goals: 2, actions: 4 },
                MockUvfa { goals: 2, actions: 4 },
                4.0,
                99,
            );
            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());
            assert!(grad.is_empty(), "out-of-range goal must yield empty grad");
        }

        #[test]
        fn test_commit_dual_leo_large_beta_approximates_hard_commit() {
            // G1 stretch: β=64 → sigmoid saturates → grad ≈ pure higher-conf
            // head. UVFA has higher max on every goal → grad ≈ uvfa.
            let oracle = CommitDualLeoOracle::new(
                MockLeo { goals: 2, actions: 4 },
                MockUvfa { goals: 2, actions: 4 },
                64.0,
                0,
            );
            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());
            // gate = σ(64·(4 − 100)) = σ(−6144) → exactly 0.0 in f32.
            // grad = 0·leo + 1·uvfa = uvfa.
            assert_eq!(grad, vec![100.0, 200.0, 300.0, 400.0]);
        }

        #[test]
        fn test_commit_dual_leo_symmetric_when_leo_more_confident() {
            // If we swap which head is more confident, the gate should flip.
            // Use MockUvfa as "teacher" (higher values) and MockLeo as
            // "student" — gate should → 1 → return teacher (= MockUvfa).
            let oracle = CommitDualLeoOracle::new(
                MockUvfa { goals: 2, actions: 4 }, // as "leo" slot
                MockLeo { goals: 2, actions: 4 }, // as "uvfa" slot
                8.0,
                0,
            );
            let state = vec![0.0; 8];
            let grad = oracle.q_gradient_at(&state, &());
            // Δconf = max(MockUvfa[goal 0]) - max(MockLeo[goal 0])
            //       = 400 - 4 = 396 → gate = σ(8·396) → 1.0 (overflows).
            // grad = 1·MockUvfa + 0·MockLeo = MockUvfa's slice.
            assert_eq!(grad, vec![100.0, 200.0, 300.0, 400.0]);
        }
    }
}

#[cfg(all(feature = "leo_all_goals", feature = "dual_leo"))]
pub use commit_dual_leo_oracle::CommitDualLeoOracle;

// ──────────────────────────────────────────────────────────────────────────
// ActionBridgeOracle — Plasma tier (feature: action_bridge)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(feature = "action_bridge")]
mod action_bridge_oracle {
    use super::*;
    use crate::bridge::ActionBridge;

    /// Plasma-tier oracle wrapping an [`ActionBridge<A, D>`].
    ///
    /// The "gradient" per action is the dot product of the latent Q-vector
    /// with that action's ternary direction vector:
    ///
    /// ```text
    /// gradient[a] = dot(q_values, action_directions[a])
    ///            = Σ_i q_values[i] · directions[a][i]
    /// ```
    ///
    /// This gives a per-action logit tilt that favours actions whose ternary
    /// direction aligns with the latent Q-values. The tilt is rank-compatible
    /// with `ActionBridge::select_action` (which applies `sigmoid(dot)`), but
    /// expressed here as a raw pre-sigmoid scalar — the drafter's
    /// `logits += w · g` shift is the additive-logit-space form.
    ///
    /// # Action / State Mapping
    ///
    /// - `State` = `[f32; D]` — the latent Q-value vector.
    /// - `Action` = `()` — the gradient is state-derived (no projection needed;
    ///   the bridge already encodes the full per-action scoring).
    ///
    /// # Confidence
    ///
    /// `1.0` — ternary direction dot products are deterministic.
    pub struct ActionBridgeOracle<const A: usize, const D: usize> {
        bridge: ActionBridge<A, D>,
    }

    /// Maximum number of per-action scores we can recover in a single
    /// `select_top_k` call without heap allocation. Game NPC action spaces
    /// are typically ≤ 16; 32 provides generous headroom. For `A > 32`, only
    /// the top 32 actions receive a non-zero gradient (lossy but safe).
    const TOPK_BUF_CAP: usize = 32;

    impl<const A: usize, const D: usize> ActionBridgeOracle<A, D> {
        /// Wrap an [`ActionBridge`].
        #[inline]
        pub fn new(bridge: ActionBridge<A, D>) -> Self {
            Self { bridge }
        }

        /// Borrow the underlying bridge.
        #[inline]
        pub fn bridge(&self) -> &ActionBridge<A, D> {
            &self.bridge
        }

        /// Fill `out` with per-action gradient (raw dot product) values.
        ///
        /// `ActionBridge` keeps its direction vectors private and exposes only
        /// `select_top_k` (which returns sigmoid(dot) scores). We recover the
        /// raw dot product via `logit` (inverse sigmoid). Sigmoid is monotonic,
        /// so the resulting gradient preserves `select_action`'s ranking —
        /// which is what QGF's tilt needs.
        ///
        /// For large `|dot|` (where sigmoid saturates), `logit` clips to ≈ ±13.8.
        /// This is acceptable because the guidance weight `1/β` scales it down,
        /// and the ranking (not magnitude) drives the tilt's effect.
        fn gradient_into_inner(&self, state: &[f32; D], out: &mut [f32]) {
            for slot in out.iter_mut() {
                *slot = 0.0;
            }
            let mut entries: [(usize, f32); TOPK_BUF_CAP] = [(0, 0.0); TOPK_BUF_CAP];
            let k = A.min(TOPK_BUF_CAP);
            let count = self.bridge.select_top_k(state, A, &mut entries[..k]);
            for &(action, sigmoid_score) in &entries[..count] {
                if action < out.len() {
                    out[action] = logit(sigmoid_score);
                }
            }
        }
    }

    impl<const A: usize, const D: usize> QGradientOracle for ActionBridgeOracle<A, D> {
        type State = [f32; D];
        type Action = ();

        fn q_gradient_at(&self, state: &Self::State, _projected_action: &Self::Action) -> Vec<f32> {
            let mut out = vec![0.0f32; A];
            self.gradient_into_inner(state, &mut out);
            out
        }

        fn q_gradient_into(
            &self,
            state: &Self::State,
            _projected_action: &Self::Action,
            out: &mut [f32],
        ) {
            self.gradient_into_inner(state, out);
        }

        // Deterministic ternary dot → confidence 1.0.
    }

    /// Inverse sigmoid (logit): `logit(p) = ln(p / (1 − p))`.
    ///
    /// Recovers the raw dot product from `sigmoid(dot)`. Clamped to avoid
    /// `ln(0)` blow-ups when sigmoid saturates.
    #[inline]
    fn logit(p: f32) -> f32 {
        let clamped = p.clamp(1e-6, 1.0 - 1e-6);
        let odds = clamped / (1.0 - clamped);
        odds.ln()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::bridge::ActionBridge;

        fn make_bridge() -> ActionBridge<3, 2> {
            // Action 0: prefers +q0  → direction [1, 0]
            // Action 1: prefers -q0  → direction [-1, 0]
            // Action 2: prefers +q1  → direction [0, 1]
            let directions: [[i8; 2]; 3] = [[1, 0], [-1, 0], [0, 1]];
            ActionBridge::new(directions, 0.5)
        }

        #[test]
        fn test_action_bridge_oracle_gradient() {
            let oracle = ActionBridgeOracle::new(make_bridge());
            let q: [f32; 2] = [5.0, 1.0];

            let grad = oracle.q_gradient_at(&q, &());

            // Expected raw dots (before sigmoid):
            //   action 0:  5*1 + 1*0  =  5
            //   action 1:  5*-1 + 1*0 = -5
            //   action 2:  5*0 + 1*1  =  1
            assert_eq!(grad.len(), 3);
            assert!(
                (grad[0] - 5.0).abs() < 1e-3,
                "action 0 gradient should be ≈ 5 (got {})",
                grad[0]
            );
            assert!(
                (grad[1] - (-5.0)).abs() < 1e-3,
                "action 1 gradient should be ≈ -5 (got {})",
                grad[1]
            );
            assert!(
                (grad[2] - 1.0).abs() < 1e-3,
                "action 2 gradient should be ≈ 1 (got {})",
                grad[2]
            );
        }

        #[test]
        fn test_action_bridge_oracle_into_matches_at() {
            let oracle = ActionBridgeOracle::new(make_bridge());
            let q: [f32; 2] = [3.0, 2.0];

            let via_at = oracle.q_gradient_at(&q, &());

            let mut via_into = [0.0f32; 3];
            oracle.q_gradient_into(&q, &(), &mut via_into);

            for i in 0..3 {
                assert!(
                    (via_at[i] - via_into[i]).abs() < 1e-3,
                    "index {i}: at={} vs into={}",
                    via_at[i],
                    via_into[i]
                );
            }
        }

        #[test]
        fn test_action_bridge_oracle_ranking_preserved() {
            // The gradient (raw dot) must rank actions identically to
            // ActionBridge::select_action (sigmoid dot) — sigmoid is monotonic.
            let oracle = ActionBridgeOracle::new(make_bridge());
            let q: [f32; 2] = [5.0, 1.0];

            let grad = oracle.q_gradient_at(&q, &());
            let (best_action, _) = oracle.bridge().select_action(&q);

            // Argmax of gradient should match select_action's best.
            let grad_argmax = grad
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| crate::float_order::cmp_for_max(**a, **b))
                .map(|(i, _)| i)
                .unwrap();

            assert_eq!(grad_argmax, best_action);
            assert_eq!(best_action, 0, "action 0 should win for q=[5,1]");
        }

        #[test]
        fn test_action_bridge_oracle_confidence_is_one() {
            let oracle = ActionBridgeOracle::new(make_bridge());
            let q: [f32; 2] = [0.0, 0.0];
            assert_eq!(oracle.confidence(&q), 1.0);
        }

        #[test]
        fn test_logit_inverse_of_sigmoid() {
            // Round-trip: logit(sigmoid(x)) ≈ x for moderate x.
            for &x in &[0.0, 0.5, 1.0, 2.0, -1.0, -2.0] {
                let s = crate::simd::fast_sigmoid(x);
                let recovered = logit(s);
                assert!(
                    (recovered - x).abs() < 1e-3,
                    "logit(sigmoid({x})) = {recovered}, expected {x}"
                );
            }
        }
    }
}

#[cfg(feature = "action_bridge")]
pub use action_bridge_oracle::ActionBridgeOracle;

// ──────────────────────────────────────────────────────────────────────────
// BfnProxyOracle — Freeze tier fallback (no trained critic required)
// ──────────────────────────────────────────────────────────────────────────

/// Freeze-tier oracle using pre-computed rejection-sampled returns as a
/// gradient proxy.
///
/// When no trained critic is available (cold start, OOD state, freeze tier),
/// this oracle uses rejection-sampled episode returns as a stand-in for the
/// Q-gradient. The caller populates `returns` with per-action return estimates
/// (e.g. from a BFN rejection sampler or a roll-out buffer), and the oracle
/// reports them as the gradient signal.
///
/// # Confidence
///
/// Returns `0.3` — a deliberately low confidence that causes the adaptive
/// guidance weight to collapse toward zero (safe fallback). The value `0.3`
/// is chosen to be below the default adaptive threshold of `0.5`, so
/// `sigmoid(steepness · (0.3 − 0.5)) ≈ sigmoid(−0.8 · steepness) → small`.
///
/// # Action / State Mapping
///
/// - `State` = `()` (the returns are pre-computed and stored in the oracle).
/// - `Action` = `()` (the gradient is state-derived).
pub struct BfnProxyOracle {
    /// Per-action rejection-sampled return estimates — the gradient proxy.
    returns: Vec<f32>,
}

impl BfnProxyOracle {
    /// Construct with a pre-computed return vector.
    #[inline]
    pub fn new(returns: Vec<f32>) -> Self {
        Self { returns }
    }

    /// Construct an empty proxy (all-zero gradient). Used for pure fallback.
    #[inline]
    pub fn empty(action_count: usize) -> Self {
        Self {
            returns: vec![0.0; action_count],
        }
    }

    /// Replace the return estimates (e.g. after a new rejection-sampling batch).
    #[inline]
    pub fn set_returns(&mut self, returns: Vec<f32>) {
        self.returns = returns;
    }

    /// Borrow the current return estimates.
    #[inline]
    pub fn returns(&self) -> &[f32] {
        &self.returns
    }

    /// The fixed low confidence reported by this proxy.
    pub const CONFIDENCE: f32 = 0.3;
}

impl QGradientOracle for BfnProxyOracle {
    type State = ();
    type Action = ();

    fn q_gradient_at(&self, _state: &Self::State, _projected_action: &Self::Action) -> Vec<f32> {
        self.returns.clone()
    }

    fn q_gradient_into(
        &self,
        _state: &Self::State,
        _projected_action: &Self::Action,
        out: &mut [f32],
    ) {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.returns.get(i).copied().unwrap_or(0.0);
        }
    }

    /// Freeze-tier proxy → low confidence (0.3).
    #[inline]
    fn confidence(&self, _state: &Self::State) -> f32 {
        Self::CONFIDENCE
    }
}

#[cfg(test)]
mod bfn_proxy_tests {
    use super::*;

    #[test]
    fn test_bfn_proxy_oracle_returns_gradient() {
        let oracle = BfnProxyOracle::new(vec![0.5, -0.2, 0.8]);
        let g = oracle.q_gradient_at(&(), &());
        assert_eq!(g, vec![0.5, -0.2, 0.8]);
    }

    #[test]
    fn test_bfn_proxy_oracle_into_matches_at() {
        let oracle = BfnProxyOracle::new(vec![1.0, 2.0, 3.0]);
        let via_at = oracle.q_gradient_at(&(), &());

        let mut via_into = [0.0f32; 3];
        oracle.q_gradient_into(&(), &(), &mut via_into);

        assert_eq!(via_at, via_into.to_vec());
    }

    #[test]
    fn test_bfn_proxy_oracle_low_confidence() {
        let oracle = BfnProxyOracle::new(vec![1.0, 2.0]);
        let conf = oracle.confidence(&());
        assert!(
            conf < 0.5,
            "BFN proxy confidence must be < 0.5 for safe fallback, got {conf}"
        );
        assert!(
            (conf - 0.3).abs() < 1e-6,
            "BFN proxy confidence should be exactly 0.3"
        );
    }

    #[test]
    fn test_bfn_proxy_oracle_empty_zeros() {
        let oracle = BfnProxyOracle::empty(4);
        let g = oracle.q_gradient_at(&(), &());
        assert_eq!(g, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bfn_proxy_oracle_short_buffer() {
        let oracle = BfnProxyOracle::new(vec![1.0, 2.0, 3.0]);
        let mut buf = [0.0f32; 2];
        oracle.q_gradient_into(&(), &(), &mut buf);
        assert_eq!(buf, [1.0, 2.0]);
    }

    #[test]
    fn test_bfn_proxy_oracle_long_buffer_pads_zero() {
        let oracle = BfnProxyOracle::new(vec![1.0]);
        let mut buf = [99.0f32; 3];
        oracle.q_gradient_into(&(), &(), &mut buf);
        assert_eq!(buf, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bfn_proxy_oracle_set_returns() {
        let mut oracle = BfnProxyOracle::empty(2);
        assert_eq!(oracle.returns(), &[0.0, 0.0]);

        oracle.set_returns(vec![5.0, 6.0]);
        assert_eq!(oracle.returns(), &[5.0, 6.0]);
        assert_eq!(oracle.q_gradient_at(&(), &()), vec![5.0, 6.0]);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// NoGuidanceOracle — already defined in traits.rs; verify it here too.
// ──────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "qgf_oracle"))]
mod no_guidance_tests {
    use super::*;
    use crate::traits::NoGuidanceOracle;

    #[test]
    fn test_no_guidance_oracle_zero_gradient() {
        let oracle = NoGuidanceOracle;
        let g = oracle.q_gradient_at(&(), &());
        assert!(
            g.is_empty(),
            "NoGuidanceOracle should return empty gradient"
        );

        let mut buf = [99.0f32; 4];
        oracle.q_gradient_into(&(), &(), &mut buf);
        assert_eq!(buf, [0.0; 4], "NoGuidanceOracle should zero the buffer");
    }

    #[test]
    fn test_no_guidance_oracle_zero_confidence() {
        let oracle = NoGuidanceOracle;
        assert_eq!(oracle.confidence(&()), 0.0);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// WarmTierOracle — Warm tier (Plan 268 T9)
// Adapts a `QgfGpuDelegate` (Plan 268 T8 GPU dispatch surface) into the
// `QGradientOracle` trait so it can serve as a single-query oracle. For batched
// use, callers should go through `QgfBackendDispatch` directly (the GPU only
// wins at batch ≥ 8 per `route_for`); this oracle exists so a Warm-tier
// deployment can be dropped into a `QGuidedDrafter` that doesn't know about
// dispatch.
// ───────────────────────────────────────────────────────────────────────────

#[cfg(feature = "qgf_drafter")]
mod warm_tier_oracle {
    use super::*;
    use crate::qgf::dispatch::QgfGpuDelegate;

    /// Warm-tier oracle wrapping a GPU-batched critic delegate.
    ///
    /// # When to use
    ///
    /// The Warm tier is the right choice when:
    /// - A GPU is available and the action space is large (≥ 1024).
    /// - The critic is a trained neural net whose forward pass is expensive
    ///   on CPU but cheap on GPU.
    /// - The deployment is training-time / large-batch (single-query GPU use
    ///   pays the kernel-launch overhead without amortising it — prefer the
    ///   Hot or Plasma tier for single-query game-NPC use).
    ///
    /// # Confidence
    ///
    /// Returns the delegate's self-reported success. A GPU delegate that
    /// reports failure (returns 0 rows) signals low confidence → the adaptive
    /// guidance weight collapses toward 0 → safe BC fallback. This makes the
    /// Warm tier self-degrading: if the GPU is busy / OOM / unavailable, the
    /// drafter stops trusting it without needing an external health check.
    ///
    /// # Layering note
    ///
    /// The concrete GPU kernel lives in `riir-gpu` (separate repo).
    /// `WarmTierOracle` is generic over `D: QgfGpuDelegate` — the upper layer
    /// (`riir-engine`) constructs it with its `riir-gpu`-backed delegate.
    pub struct WarmTierOracle<S, A, D> {
        delegate: D,
        /// Cached action-space width. Single-query gradient buffers are sized
        /// to this; if the caller passes a shorter buffer, only the prefix is
        /// written.
        action_space_size: usize,
        _marker: std::marker::PhantomData<(S, A)>,
    }

    impl<S, A, D> WarmTierOracle<S, A, D> {
        /// Construct with a fixed action-space width.
        ///
        /// `action_space_size` MUST match the width the delegate writes; if it
        /// doesn't, `q_gradient_into` will write out-of-bounds (the delegate
        /// is responsible for respecting the buffer length it's given).
        #[inline]
        pub fn new(delegate: D, action_space_size: usize) -> Self {
            Self {
                delegate,
                action_space_size,
                _marker: std::marker::PhantomData,
            }
        }

        /// Borrow the inner delegate (for wiring into `QgfBackendDispatch`).
        #[inline]
        pub fn delegate(&self) -> &D {
            &self.delegate
        }
    }

    impl<S, A, D> QGradientOracle for WarmTierOracle<S, A, D>
    where
        D: QgfGpuDelegate<S, A>,
    {
        type State = S;
        type Action = A;

        fn q_gradient_at(&self, state: &Self::State, projected: &Self::Action) -> Vec<f32> {
            let mut out = vec![0.0f32; self.action_space_size];
            self.q_gradient_into(state, projected, &mut out);
            out
        }

        fn q_gradient_into(&self, state: &Self::State, projected: &Self::Action, out: &mut [f32]) {
            // Single-state batch — the delegate's contract is a slice of
            // state refs, so we build a 1-element view. Avoids allocation;
            // the delegate writes directly into `out`.
            let state_ref: [&S; 1] = [state];
            let action_ref: [&A; 1] = [projected];
            let written =
                self.delegate
                    .batch_gradient_into(&state_ref, &action_ref, out.len(), out);
            if written == 0 {
                // GPU failure — zero the buffer so downstream tilt is a no-op.
                for slot in out.iter_mut() {
                    *slot = 0.0;
                }
            }
        }

        #[inline]
        fn confidence(&self, _: &Self::State) -> f32 {
            // We don't probe the delegate per-call (that would double the GPU
            // cost). Assume healthy; the adaptive weight reacts to the
            // zeroed gradient (GPU failure → zero tilt → safe fallback) via
            // the tilt math, not via confidence.
            //
            // A future enhancement could EMA-track the failure rate and lower
            // confidence proportionally — left for riir-engine integration.
            1.0
        }
    }

    #[cfg(all(test, feature = "qgf_drafter"))]
    mod tests {
        use super::*;
        use crate::qgf::dispatch::SealedForTests;

        /// Mock GPU delegate that writes `state_scalar * action_index` per cell.
        struct MockGpuDelegate;
        impl SealedForTests for MockGpuDelegate {}
        impl QgfGpuDelegate<f32, ()> for MockGpuDelegate {
            fn batch_gradient_into(
                &self,
                states: &[&f32],
                _actions: &[&()],
                action_space_size: usize,
                out: &mut [f32],
            ) -> usize {
                for (row, &s) in states.iter().enumerate() {
                    let base = row * action_space_size;
                    for i in 0..action_space_size {
                        out[base + i] = s * i as f32;
                    }
                }
                states.len()
            }
        }

        /// Failing delegate — simulates GPU OOM / unavailable.
        struct FailingGpuDelegate;
        impl SealedForTests for FailingGpuDelegate {}
        impl QgfGpuDelegate<f32, ()> for FailingGpuDelegate {
            fn batch_gradient_into(&self, _: &[&f32], _: &[&()], _: usize, _: &mut [f32]) -> usize {
                0
            }
        }

        #[test]
        fn test_warm_oracle_writes_gradient() {
            let oracle: WarmTierOracle<f32, (), _> = WarmTierOracle::new(MockGpuDelegate, 4);
            let grad = oracle.q_gradient_at(&3.0, &());
            // state=3, action_space=4 → [0, 3, 6, 9].
            assert_eq!(grad, vec![0.0, 3.0, 6.0, 9.0]);
        }

        #[test]
        fn test_warm_oracle_into_matches_at() {
            let oracle: WarmTierOracle<f32, (), _> = WarmTierOracle::new(MockGpuDelegate, 4);
            let via_at = oracle.q_gradient_at(&2.0, &());
            let mut via_into = [0.0f32; 4];
            oracle.q_gradient_into(&2.0, &(), &mut via_into);
            assert_eq!(via_at, via_into.to_vec());
        }

        #[test]
        fn test_warm_oracle_gpu_failure_zeros_buffer() {
            let oracle: WarmTierOracle<f32, (), _> = WarmTierOracle::new(FailingGpuDelegate, 4);
            let mut buf = [99.0f32; 4];
            oracle.q_gradient_into(&1.0, &(), &mut buf);
            assert_eq!(
                buf, [0.0; 4],
                "GPU failure must zero the buffer for safe tilt no-op"
            );
        }

        #[test]
        fn test_warm_oracle_confidence_is_one() {
            let oracle: WarmTierOracle<f32, (), _> = WarmTierOracle::new(MockGpuDelegate, 4);
            assert_eq!(oracle.confidence(&1.0), 1.0);
        }

        #[test]
        fn test_warm_oracle_delegate_accessor() {
            let oracle: WarmTierOracle<f32, (), _> = WarmTierOracle::new(MockGpuDelegate, 4);
            // Just verify the accessor compiles and returns the right type.
            let _d: &MockGpuDelegate = oracle.delegate();
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// ColdTierOracle — Cold tier (Plan 268 T9)
// Turso/libSQL-backed Q-table loader. The DB connection lives in the upper
// layer (turso is NOT a katgpt-core dependency — keeps the lowest layer
// dependency-free); this oracle is generic over a `QTableLoader` trait that
// the upper layer implements with its turso/libSQL client.
// ───────────────────────────────────────────────────────────────────────────

mod cold_tier_oracle {
    use super::*;

    /// Trait: load a Q-value row for a given state key.
    ///
    /// Implemented by the upper layer with a turso/libSQL connection (per
    /// global AGENTS.md: "Use turso/libsql with encryption"). The loader
    /// owns the connection; this oracle owns the lookup → gradient mapping.
    ///
    /// # Contract
    ///
    /// - `load_row` writes `min(out.len(), row_width)` f32 values into `out`.
    /// - Returns the number of values written (0 = cache miss / DB unavailable).
    /// - MUST be side-effect-free (read-only) — the Cold tier is a snapshot,
    ///   not a live update path. Writes go through the chain commit path.
    pub trait QTableLoader {
        /// State key type — typically a BLAKE3 hash of the state vector, but
        /// opaque to this oracle (the loader decides the encoding).
        type Key;

        /// Load the Q-value row for `key` into `out`. Returns count written.
        fn load_row(&self, key: &Self::Key, out: &mut [f32]) -> usize;
    }

    /// Cold-tier oracle: loads Q-values from a persistent Q-table.
    ///
    /// # When to use
    ///
    /// The Cold tier is for episode-end consolidation: a Q-table snapshot is
    /// committed to Turso/libSQL at episode end, and the next episode's
    /// drafter loads rows on-demand for states it has seen before. Latency is
    /// ~10ms per load (dominated by the DB round-trip), so the Cold tier is
    /// only competitive when the critic has no hot-path cache (Freeze tier
    /// would be the alternative — pure BC, zero latency, zero guidance).
    ///
    /// # Confidence
    ///
    /// Cold-tier confidence is configurable (default `0.7`) — the snapshot is
    /// stale by definition (it's from the last episode), so the adaptive
    /// weight should be moderate, not saturating. A cache miss (loader returns
    /// 0) collapses the gradient to zero → safe fallback.
    ///
    /// # Layering
    ///
    /// `katgpt-core` does NOT depend on turso/libSQL. The `QTableLoader`
    /// trait is the integration seam: the upper layer (`riir-engine` or
    /// `riir-chain`) implements it with its encrypted libSQL client and
    /// constructs `ColdTierOracle::new(loader, confidence, action_space)`.
    pub struct ColdTierOracle<L: QTableLoader> {
        loader: L,
        confidence: f32,
    }

    impl<L: QTableLoader> ColdTierOracle<L> {
        /// Construct with a confidence value in `[0.0, 1.0]`.
        ///
        /// Typical: `0.7` (stale snapshot — moderate trust). Clamp guards
        /// against out-of-range.
        #[inline]
        pub fn new(loader: L, confidence: f32) -> Self {
            Self {
                loader,
                confidence: confidence.clamp(0.0, 1.0),
            }
        }

        /// Borrow the inner loader (for the upper layer to flush / inspect).
        #[inline]
        pub fn loader(&self) -> &L {
            &self.loader
        }
    }

    impl<L> QGradientOracle for ColdTierOracle<L>
    where
        L: QTableLoader,
    {
        // The state IS the key — callers pass the BLAKE3 hash (or whatever
        // encoding the loader expects) directly as the "state". This avoids a
        // hash step inside the oracle (the upper layer already has the hash).
        type State = L::Key;
        type Action = ();

        fn q_gradient_at(&self, key: &Self::State, _: &Self::Action) -> Vec<f32> {
            // Cold-tier rows are variable-width in principle; default to a
            // generous 1024-wide buffer and truncate to the written count.
            let mut buf = vec![0.0f32; 1024];
            let n = self.loader.load_row(key, &mut buf);
            buf.truncate(n);
            buf
        }

        fn q_gradient_into(&self, key: &Self::State, _: &Self::Action, out: &mut [f32]) {
            let n = self.loader.load_row(key, out);
            if n == 0 {
                // Cache miss — zero the buffer so the tilt is a no-op.
                for slot in out.iter_mut() {
                    *slot = 0.0;
                }
            } else if n < out.len() {
                // Partial write — zero the remainder so stale data doesn't leak.
                for slot in out[n..].iter_mut() {
                    *slot = 0.0;
                }
            }
        }

        #[inline]
        fn confidence(&self, _: &Self::State) -> f32 {
            self.confidence
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Mock loader that returns a fixed row keyed by the input byte.
        struct MockLoader;
        impl QTableLoader for MockLoader {
            type Key = u8;
            fn load_row(&self, key: &u8, out: &mut [f32]) -> usize {
                // Write `key` into each cell, up to 4 cells.
                let n = out.len().min(4);
                for (i, slot) in out.iter_mut().enumerate().take(n) {
                    *slot = (*key as f32) * (i as f32);
                }
                n
            }
        }

        /// Loader that always misses — simulates DB unavailable.
        struct MissingLoader;
        impl QTableLoader for MissingLoader {
            type Key = u8;
            fn load_row(&self, _: &u8, _: &mut [f32]) -> usize {
                0
            }
        }

        #[test]
        fn test_cold_oracle_writes_gradient() {
            let oracle = ColdTierOracle::new(MockLoader, 0.7);
            let grad = oracle.q_gradient_at(&3, &());
            // key=3, 4 cells → [0, 3, 6, 9].
            assert_eq!(grad, vec![0.0, 3.0, 6.0, 9.0]);
        }

        #[test]
        fn test_cold_oracle_into_matches_at() {
            let oracle = ColdTierOracle::new(MockLoader, 0.7);
            let via_at = oracle.q_gradient_at(&2, &());
            let mut via_into = [0.0f32; 4];
            oracle.q_gradient_into(&2, &(), &mut via_into);
            assert_eq!(via_at, via_into.to_vec());
        }

        #[test]
        fn test_cold_oracle_cache_miss_zeros_buffer() {
            let oracle = ColdTierOracle::new(MissingLoader, 0.7);
            let mut buf = [99.0f32; 4];
            oracle.q_gradient_into(&1, &(), &mut buf);
            assert_eq!(buf, [0.0; 4], "cache miss must zero buffer");
        }

        #[test]
        fn test_cold_oracle_partial_write_zeros_tail() {
            // Loader writes 2 cells; buffer is 5. Tail must be zeroed.
            struct PartialLoader;
            impl QTableLoader for PartialLoader {
                type Key = u8;
                fn load_row(&self, _: &u8, out: &mut [f32]) -> usize {
                    out[0] = 1.0;
                    out[1] = 2.0;
                    2
                }
            }
            let oracle = ColdTierOracle::new(PartialLoader, 0.7);
            let mut buf = [99.0f32; 5];
            oracle.q_gradient_into(&0, &(), &mut buf);
            assert_eq!(buf, [1.0, 2.0, 0.0, 0.0, 0.0]);
        }

        #[test]
        fn test_cold_oracle_confidence_configurable_and_clamped() {
            let oracle = ColdTierOracle::new(MockLoader, 0.7);
            assert!((oracle.confidence(&0) - 0.7).abs() < 1e-6);

            let oracle_high = ColdTierOracle::new(MockLoader, 5.0);
            assert_eq!(oracle_high.confidence(&0), 1.0, "clamp to 1.0");

            let oracle_low = ColdTierOracle::new(MockLoader, -1.0);
            assert_eq!(oracle_low.confidence(&0), 0.0, "clamp to 0.0");
        }

        #[test]
        fn test_cold_oracle_loader_accessor() {
            let oracle = ColdTierOracle::new(MockLoader, 0.7);
            let _l: &MockLoader = oracle.loader();
        }
    }
}

// Re-export the Warm / Cold tier oracle types.
pub use cold_tier_oracle::{ColdTierOracle, QTableLoader};
#[cfg(feature = "qgf_drafter")]
pub use warm_tier_oracle::WarmTierOracle;
