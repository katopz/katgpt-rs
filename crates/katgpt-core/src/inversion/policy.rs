//! Candidate-enumeration policies for SipIt inversion.
//!
//! The [`InversionPolicy`] enum selects how the driver enumerates the
//! vocabulary `V` at each position. The default [`RandomPolicy`] is
//! uniform-without-replacement: worst case `|V|` trials per position
//! (amortized `|V|/2`), but the random ordering makes the worst case
//! astronomically unlikely on real transformers (paper §E.1 reports
//! gradient-guided finds the token in <0.25% of |V|; uniform-random does
//! noticeably worse but is still correct in the limit).
//!
//! Phase 2 adds [`GradientGuidedPolicy`] behind the `grad_policy` sub-feature:
//! ranks candidates by L∞ distance of `F(v; π, t)` from `h̆_t`, using the
//! caller-supplied gradient hook (paper Alg 3). The Phase 1 driver never
//! reaches this branch.

use fastrand::Rng;

/// Policy selector. See module docs.
///
/// Phase 2 (behind `grad_policy`): [`Self::GradientGuided`] ranks candidates
/// by gradient descent on a continuous proxy embedding (paper Alg 3). The
/// driver calls [`GradientGuidedPolicy`] which owns the proxy buffer + a
/// [`RandomPolicy`] fallback.
#[derive(Clone, Copy, Debug, Default)]
pub enum InversionPolicy {
    /// Uniform-without-replacement. Worst case `T · |V|` trials.
    #[default]
    Random,
    /// Gradient-guided ranking (paper Alg 3). Caller must supply
    /// `crate::inversion::InversionGradient` via
    /// [`crate::inversion::invert_sequence_grad_into`].
    ///
    /// `max_grad_steps` bounds the inner gradient-descent loop;
    /// `projection_period` is the number of gradient steps between
    /// nearest-vocab-token projection + acceptance tests (paper §E.1
    /// defaults: K=50). On exhaustion, falls back to uniform-without-
    /// replacement enumeration via the embedded [`RandomPolicy`].
    #[cfg(feature = "grad_policy")]
    GradientGuided {
        step_size: f32,
        grad_clip: f32,
        max_grad_steps: usize,
        projection_period: usize,
    },
}

#[cfg(feature = "grad_policy")]
impl InversionPolicy {
    /// Paper §E.1 defaults: step γ = 0.1, clip norm 1, 200 gradient steps,
    /// projection every K = 50 steps. Tuned for `|V|` in the 32K–128K range;
    /// callers on tiny vocabs may want smaller `max_grad_steps`.
    pub fn gradient_guided_default() -> Self {
        Self::GradientGuided {
            step_size: 0.1,
            grad_clip: 1.0,
            max_grad_steps: 200,
            projection_period: 50,
        }
    }
}

/// Uniform-without-replacement vocabulary enumeration.
///
/// Generated lazily via a Fisher-Yates shuffle on a `Vec<u32>` of length
/// `|V|`; the shuffle is allocation-free on subsequent positions (the
/// permutation buffer is reused). The Rng is held by the policy so that
/// two consecutive calls within the same driver run produce different
/// orderings.
pub struct RandomPolicy {
    rng: Rng,
    permutation: Vec<u32>,
    cursor: usize,
}

impl RandomPolicy {
    /// Construct for a vocabulary of size `vocab_size`. Allocates one
    /// `Vec<u32>` of length `vocab_size`; this is a one-time setup cost,
    /// not a per-position allocation.
    pub fn new(vocab_size: u32, seed: u64) -> Self {
        let permutation: Vec<u32> = (0..vocab_size).collect();
        Self {
            rng: Rng::with_seed(seed),
            permutation,
            cursor: 0,
        }
    }

    /// Reset for the next position: re-shuffle the remaining-vocabulary
    /// permutation and reset the cursor. Allocation-free.
    ///
    /// Per AGENTS.md hot-loop rules, we use a Fisher-Yates shuffle in-place
    /// on the existing buffer.
    pub fn reset(&mut self) {
        let n = self.permutation.len();
        for i in (1..n).rev() {
            let j = self.rng.usize(0..=i);
            self.permutation.swap(i, j);
        }
        self.cursor = 0;
    }

    /// Return the next candidate token, or `None` if the vocabulary is
    /// exhausted. Allocation-free.
    #[inline]
    pub fn next_candidate(&mut self) -> Option<u32> {
        if self.cursor < self.permutation.len() {
            let v = self.permutation[self.cursor];
            self.cursor += 1;
            Some(v)
        } else {
            None
        }
    }

    /// Number of candidates already returned by `next_candidate` since the
    /// last `reset`.
    #[inline]
    pub fn candidates_tried(&self) -> usize {
        self.cursor
    }
}

/// Gradient-guided vocabulary enumeration (paper Alg 3).
///
/// Maintains a continuous proxy embedding `e ∈ R^d` that is refined by
/// gradient descent on the loss `L(e) = ½·‖h̆_t − F(e; π, t)‖²`. Every
/// `projection_period` steps the proxy is projected to the nearest
/// vocabulary token (`argmin_v ‖e − embedding[v]‖²`) and that token is
/// acceptance-tested via the local verifier.
///
/// The policy owns two reusable buffers (`proxy`, `grad_scratch`) plus an
/// embedded [`RandomPolicy`] for the post-exhaustion fallback path. Per
/// AGENTS.md hot-loop rules: zero allocation after `new`.
///
/// # Algorithm
///
/// For each position `t`:
/// 1. Zero the proxy (`e ← 0`).
/// 2. For `step` in `0..max_grad_steps`:
///    a. `grad ← grad_hidden_at_into(prefix, observed_state, proxy, t)`.
///    b. Clip `grad` to L2 norm ≤ `grad_clip`.
///    c. `proxy ← proxy − step_size · grad`.
///    d. If `step+1` is divisible by `projection_period`, OR this is the
///    final step: `v ← nearest_token(proxy)`, then acceptance-test.
///    e. On acceptance: return `Accepted(v)`.
/// 3. On exhaustion of gradient budget: delegate to [`RandomPolicy`] for the
///    remaining `|V| − projected_tokens` candidates.
#[cfg(feature = "grad_policy")]
pub struct GradientGuidedPolicy {
    step_size: f32,
    grad_clip: f32,
    max_grad_steps: usize,
    projection_period: usize,
    pub(crate) proxy: Vec<f32>,
    pub(crate) grad_scratch: Vec<f32>,
    pub(crate) random_fallback: RandomPolicy,
    /// Tokens already projection-tested this position (so the random fallback
    /// doesn't re-test them). Reset per position.
    pub(crate) projected: Vec<bool>,
}

#[cfg(feature = "grad_policy")]
impl GradientGuidedPolicy {
    /// Construct for a vocabulary of size `vocab_size` + hidden dimension `d`.
    /// Allocates the proxy + grad scratch (each `d` floats), the random
    /// fallback permutation (`vocab_size` `u32`s), and the projected-set
    /// bitmap (`vocab_size` bits). These are one-time setup costs.
    pub fn new(vocab_size: u32, d_len: usize, seed: u64, policy: &InversionPolicy) -> Self {
        let (step_size, grad_clip, max_grad_steps, projection_period) = match *policy {
            InversionPolicy::GradientGuided {
                step_size,
                grad_clip,
                max_grad_steps,
                projection_period,
            } => (step_size, grad_clip, max_grad_steps, projection_period),
            InversionPolicy::Random => {
                // Caller passed Random to the grad policy; use defaults so the
                // algorithm still runs if the user reaches this path. The
                // driver normally dispatches Random to RandomPolicy directly.
                (0.1, 1.0, 200, 50)
            }
        };
        Self {
            step_size,
            grad_clip,
            max_grad_steps,
            projection_period: projection_period.max(1),
            proxy: vec![0.0; d_len],
            grad_scratch: vec![0.0; d_len],
            random_fallback: RandomPolicy::new(vocab_size, seed),
            projected: vec![false; vocab_size as usize],
        }
    }

    /// Reset per-position state: zero the proxy, clear the projected bitmap,
    /// and reset the random fallback cursor. Allocation-free.
    pub fn reset(&mut self) {
        self.proxy.fill(0.0);
        self.projected.fill(false);
        // Random fallback is reset lazily on first use.
    }

    /// Returns the count of acceptance tests performed so far this position
    /// (projections + random fallback trials).
    #[inline]
    pub fn candidates_tried(&self) -> usize {
        self.projected.iter().filter(|&&b| b).count() + self.random_fallback.candidates_tried()
    }

    // ── pub(crate) accessors for the driver ─────────────────────────────

    #[inline]
    pub(crate) fn step_size(&self) -> f32 {
        self.step_size
    }

    #[inline]
    pub(crate) fn grad_clip(&self) -> f32 {
        self.grad_clip
    }

    #[inline]
    pub(crate) fn max_grad_steps(&self) -> usize {
        self.max_grad_steps
    }

    #[inline]
    pub(crate) fn projection_period(&self) -> usize {
        self.projection_period
    }

    /// L2 norm of a slice.
    #[inline]
    pub(crate) fn l2_norm(v: &[f32]) -> f32 {
        let mut s = 0.0_f32;
        for x in v {
            s += x * x;
        }
        s.sqrt()
    }
}

#[cfg(feature = "grad_policy")]
#[cfg(test)]
mod grad_policy_tests {
    use super::*;

    #[test]
    fn gradient_guided_policy_default_constants_match_paper() {
        match InversionPolicy::gradient_guided_default() {
            InversionPolicy::GradientGuided {
                step_size,
                grad_clip,
                max_grad_steps,
                projection_period,
            } => {
                assert_eq!(step_size, 0.1);
                assert_eq!(grad_clip, 1.0);
                assert_eq!(max_grad_steps, 200);
                assert_eq!(projection_period, 50);
            }
            InversionPolicy::Random => panic!("gradient_guided_default returned Random"),
        }
    }

    #[test]
    fn gradient_guided_policy_reset_zeros_proxy() {
        let p = InversionPolicy::gradient_guided_default();
        let mut gp = GradientGuidedPolicy::new(8, 4, 0, &p);
        // Pollute the proxy + projected bitmap.
        gp.proxy.fill(1.0);
        gp.projected[3] = true;
        gp.reset();
        assert!(gp.proxy.iter().all(|&x| x == 0.0));
        assert!(gp.projected.iter().all(|&b| !b));
    }

    #[test]
    fn gradient_guided_policy_candidates_tried_includes_random() {
        let p = InversionPolicy::gradient_guided_default();
        let mut gp = GradientGuidedPolicy::new(8, 4, 0, &p);
        gp.reset();
        // Before any projection + before any fallback trial.
        assert_eq!(gp.candidates_tried(), 0);
        // Mark two projected.
        gp.projected[2] = true;
        gp.projected[5] = true;
        assert_eq!(gp.candidates_tried(), 2);
        // Add one random fallback trial.
        gp.random_fallback.reset();
        let _ = gp.random_fallback.next_candidate();
        assert_eq!(gp.candidates_tried(), 3);
    }

    #[test]
    fn l2_norm_basic() {
        assert!((GradientGuidedPolicy::l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-5);
        assert_eq!(GradientGuidedPolicy::l2_norm(&[0.0; 8]), 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn random_policy_visits_each_token_once_per_reset() {
        let mut p = RandomPolicy::new(8, 0);
        p.reset();
        let mut seen: HashSet<u32> = HashSet::new();
        let mut count = 0;
        while let Some(v) = p.next_candidate() {
            assert!(seen.insert(v), "token {v} returned twice in one pass");
            count += 1;
        }
        assert_eq!(count, 8);
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn random_policy_two_passes_can_differ() {
        // With seed 0 on vocab 8, the two passes should differ in ordering
        // at least once (probability of identical orderings is 1/8! ≈ 2.5e-5).
        let mut p = RandomPolicy::new(8, 0);
        p.reset();
        let first: Vec<u32> = std::iter::from_fn(|| p.next_candidate()).collect();
        p.reset();
        let second: Vec<u32> = std::iter::from_fn(|| p.next_candidate()).collect();
        assert_eq!(first.len(), 8);
        assert_eq!(second.len(), 8);
        // Same set, possibly different order.
        let s1: HashSet<u32> = first.iter().copied().collect();
        let s2: HashSet<u32> = second.iter().copied().collect();
        assert_eq!(s1, s2);
    }

    #[test]
    fn random_policy_candidates_tried_advances() {
        let mut p = RandomPolicy::new(4, 1);
        p.reset();
        assert_eq!(p.candidates_tried(), 0);
        let _ = p.next_candidate();
        assert_eq!(p.candidates_tried(), 1);
        let _ = p.next_candidate();
        let _ = p.next_candidate();
        assert_eq!(p.candidates_tried(), 3);
    }

    #[test]
    fn random_policy_returns_none_after_exhaustion() {
        let mut p = RandomPolicy::new(2, 0);
        p.reset();
        let _ = p.next_candidate();
        let _ = p.next_candidate();
        assert!(p.next_candidate().is_none());
    }
}
