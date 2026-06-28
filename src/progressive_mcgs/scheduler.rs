//! Entropy-gated exploration→exploitation scheduler.
//!
//! Paper §3.2.2 "Selection with Progressive Exploration Scheduling".
//!
//! # The Trick
//!
//! At each step, the system chooses between UCT-based exploration
//! (higher entropy) and Elite-Guided exploitation (lower entropy)
//! according to a time-dependent weight:
//!
//! ```text
//! P(S_t = UCT)   = w(t)
//! P(S_t = Elite) = 1 − w(t)
//! ```
//!
//! where `w(t)` gradually decreases from 1.0 to a minimum threshold
//! `w_min` as search time progresses. The schedule is designed so that
//! the empirical branch-selection entropy `H(π_t)` progressively decreases
//! over time, concentrating computation on promising branches.
//!
//! Empirically (paper Figure 3): `exp(H(π_t))` drops from 4.8 → 2.8
//! active branches under the schedule; vanilla MCTS stays flat at ≈4.3.
//!
//! # Note on `1/rank` vs Sigmoid
//!
//! AGENTS.md prefers sigmoid over softmax for *latent projections onto
//! direction vectors*. The Elite sampler's `1/rank` weighting (paper Eq. 5)
//! is **not** a latent projection — it's a discrete rank-based sampling
//! weight over already-evaluated nodes. The sigmoid rule does not apply.
//! If a sigmoid variant is desired for uniformity, use
//! `sigmoid(a − b·rank)` which is monotonic in rank and sigmoid-shaped.

use crate::progressive_mcgs::types::NodeId;

/// Selection mode chosen by [`EntropyGatedScheduler::pick_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SelectMode {
    /// UCT-based exploration — descend the `E_T` tree backbone per paper Eq. 3.
    Uct = 0,
    /// Elite-Guided exploitation — sample from top-K globally best nodes
    /// weighted by `1/rank` (paper Eq. 5).
    Elite = 1,
}

/// Entropy-gated scheduler — the load-bearing primitive for the
/// exploration→exploitation transition.
///
/// Hyperparameters default to paper Table 4 values; override for tick-budget
/// contexts via [`EntropyGatedScheduler::with_config`] or field mutation.
#[derive(Debug, Clone)]
pub struct EntropyGatedScheduler {
    /// Floor on UCT-selection probability. Paper default 0.2.
    pub w_min: f32,
    /// Normalized progress at which decay begins. Paper default 0.5.
    pub switch_start: f32,
    /// Normalized progress at which decay saturates at `w_min`. Paper default 0.7.
    pub switch_end: f32,
    /// Top-K nodes considered in Elite mode. Paper default 3.
    pub elite_topk: usize,
}

impl Default for EntropyGatedScheduler {
    #[inline]
    fn default() -> Self {
        Self {
            w_min: 0.2,
            switch_start: 0.5,
            switch_end: 0.7,
            elite_topk: 3,
        }
    }
}

impl EntropyGatedScheduler {
    /// Construct with explicit hyperparameters.
    #[must_use]
    pub const fn with_config(w_min: f32, switch_start: f32, switch_end: f32, elite_topk: usize) -> Self {
        Self {
            w_min,
            switch_start,
            switch_end,
            elite_topk,
        }
    }

    /// Compute the UCT-selection probability `w(t_norm)` (paper Eq. 4).
    ///
    /// Piecewise-linear schedule:
    /// - `t_norm < switch_start` → `1.0` (pure UCT)
    /// - `t_norm > switch_end`   → `w_min` (Elite dominates)
    /// - otherwise               → linear interpolation 1.0 → `w_min`
    ///
    /// This is allocation-free and branch-light. For a smoother schedule,
    /// consumers may swap this for `sigmoid(-λ·(t_norm − midpoint))`; the
    /// empirical entropy decay behavior is essentially equivalent.
    #[inline]
    #[must_use]
    pub fn w(&self, t_norm: f32) -> f32 {
        if t_norm <= self.switch_start {
            1.0
        } else if t_norm >= self.switch_end {
            self.w_min
        } else {
            // Linear interpolation in the transition window.
            let span = self.switch_end - self.switch_start;
            let s = (t_norm - self.switch_start) / span;
            1.0 + s * (self.w_min - 1.0)
        }
    }

    /// Soft-switch between [`SelectMode::Uct`] and [`SelectMode::Elite`].
    ///
    /// Returns `Uct` with probability `w(t_norm)`, else `Elite`.
    /// RNG is supplied by the caller for deterministic replay.
    ///
    /// Per AGENTS.md RNG rule (Plan 270 fair-game RNG split), the caller
    /// should seed the RNG per-zone per-cycle from a blake3-committed seed.
    #[inline]
    #[must_use]
    pub fn pick_mode(&self, t_norm: f32, rng: &mut impl RngLite) -> SelectMode {
        let w = self.w(t_norm);
        if rng.next_f32() < w {
            SelectMode::Uct
        } else {
            SelectMode::Elite
        }
    }

    /// Sample one node from the Elite set, weighted by `1/rank` (paper Eq. 5).
    ///
    /// `ranked_nodes` MUST be sorted by descending Q-value (rank 0 = best).
    /// Only the first `min(self.elite_topk, ranked_nodes.len())` nodes are
    /// considered.
    ///
    /// Returns `None` if `ranked_nodes` is empty.
    ///
    /// # Algorithm
    ///
    /// Weight of node at rank `i` (1-indexed): `1/i`.
    /// Normalized: `P(v_i) = (1/i) / Σ_{j=1..K} (1/j)`.
    /// Sample via cumulative-sum traversal — O(K) where K ≤ `elite_topk`.
    #[inline]
    #[must_use]
    pub fn elite_sample<'a>(
        &self,
        ranked_nodes: &'a [NodeId],
        rng: &mut impl RngLite,
    ) -> Option<&'a NodeId> {
        let k = ranked_nodes.len().min(self.elite_topk);
        if k == 0 {
            return None;
        }
        if k == 1 {
            return Some(&ranked_nodes[0]);
        }

        // Cumulative weights: w_i = 1/(i+1). Compute total, sample u ∈ [0, total).
        // Optimization: harmonic sum H_K = Σ 1/(i+1) for i in 0..k.
        // For small K (≤ elite_topk, default 3), this is cheap.
        let mut cum = [0.0f32; 32]; // stack array, supports up to K=32
        debug_assert!(k <= cum.len(), "elite_topk exceeds stack buffer");
        let k_eff = k.min(cum.len());
        let mut total = 0.0f32;
        for (i, slot) in cum.iter_mut().enumerate().take(k_eff) {
            total += 1.0 / (i as f32 + 1.0);
            *slot = total;
        }
        let u = rng.next_f32() * total;
        // Linear scan to find slot. For K=3, faster than binary search.
        for i in 0..k_eff {
            if u <= cum[i] {
                return Some(&ranked_nodes[i]);
            }
        }
        // Floating-point fallback — return last.
        Some(&ranked_nodes[k_eff - 1])
    }

    /// Compute the Shannon entropy of the branch-selection distribution.
    ///
    /// `selection_counts[i]` = number of times branch `i` was selected.
    /// Returns `H(π_t) = −Σ_i p_i · log(p_i)` in nats.
    ///
    /// This is a **diagnostic metric**, not an objective. Log it, don't
    /// gradient through it. The schedule [`Self::w`] is rule-based —
    /// the entropy decay is an emergent property of the soft switch,
    /// not a quantity being optimized.
    ///
    /// Returns `0.0` if `total == 0` (no selections yet).
    #[must_use]
    pub fn branch_selection_entropy(selection_counts: &[u32]) -> f32 {
        let total: u64 = selection_counts.iter().map(|&c| c as u64).sum();
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f32;
        let mut h = 0.0f32;
        for &c in selection_counts {
            if c == 0 {
                continue;
            }
            let p = c as f32 / total_f;
            h -= p * p.ln();
        }
        h
    }

    /// Effective number of active branches: `exp(H(π_t))`.
    ///
    /// Paper Figure 3 plots this directly. A uniform distribution over N
    /// branches yields `exp(H) = N`; a degenerate distribution yields 1.
    /// The schedule should drive this from ~N_lineages down to ~1-2 over
    /// the search.
    #[inline]
    #[must_use]
    pub fn effective_branch_count(selection_counts: &[u32]) -> f32 {
        Self::branch_selection_entropy(selection_counts).exp()
    }
}

/// Minimal RNG trait — abstracts over `fastrand::Rng` and custom RNGs.
///
/// This decoupling lets consumers supply their own deterministic RNG
/// (e.g., blake3-seeded per AGENTS.md Plan 270) without forcing a dependency
/// on any specific crate.
pub trait RngLite {
    /// Return a uniformly-distributed `f32` in `[0, 1)`.
    fn next_f32(&mut self) -> f32;
}

/// Adapter — `fastrand::Rng` implements [`RngLite`].
impl RngLite for fastrand::Rng {
    #[inline]
    fn next_f32(&mut self) -> f32 {
        Self::f32(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_pure_uct_before_switch_start() {
        let s = EntropyGatedScheduler::default();
        assert_eq!(s.w(0.0), 1.0);
        assert_eq!(s.w(0.3), 1.0);
        assert_eq!(s.w(0.5), 1.0); // boundary — inclusive
    }

    #[test]
    fn w_at_w_min_after_switch_end() {
        let s = EntropyGatedScheduler::default();
        assert!((s.w(0.7) - 0.2).abs() < 1e-6, "switch_end boundary");
        assert_eq!(s.w(1.0), 0.2);
        assert_eq!(s.w(2.0), 0.2); // clamped past 1.0
    }

    #[test]
    fn w_monotonic_non_increasing() {
        let s = EntropyGatedScheduler::default();
        let mut prev = s.w(0.0);
        for i in 1..=100 {
            let t = i as f32 / 100.0;
            let w = s.w(t);
            assert!(
                w <= prev + 1e-6,
                "w(t) not monotonic non-increasing at t={t}: prev={prev}, w={w}"
            );
            prev = w;
        }
    }

    #[test]
    fn w_interpolates_in_window() {
        let s = EntropyGatedScheduler::default();
        let w_mid = s.w(0.6); // midpoint of [0.5, 0.7]
        let expected = 1.0 + 0.5 * (0.2 - 1.0); // 0.6
        assert!((w_mid - expected).abs() < 1e-6, "midpoint interpolation");
    }

    #[test]
    fn elite_sample_single_node() {
        let s = EntropyGatedScheduler::default();
        let mut rng = fastrand::Rng::with_seed(42);
        let nodes = [NodeId(7)];
        let picked = s.elite_sample(&nodes, &mut rng);
        assert_eq!(picked, Some(&NodeId(7)));
    }

    #[test]
    fn elite_sample_empty() {
        let s = EntropyGatedScheduler::default();
        let mut rng = fastrand::Rng::with_seed(42);
        let nodes: [NodeId; 0] = [];
        assert!(s.elite_sample(&nodes, &mut rng).is_none());
    }

    #[test]
    fn elite_sample_distribution_skews_top_rank() {
        // With K=3 and weights [1, 1/2, 1/3], top node should be picked
        // ≈54.5% of the time (1 / (1 + 0.5 + 0.333)).
        let s = EntropyGatedScheduler::default();
        let mut rng = fastrand::Rng::with_seed(42);
        let nodes = [NodeId(0), NodeId(1), NodeId(2)];
        let mut top_count = 0u32;
        let trials = 10_000;
        for _ in 0..trials {
            if let Some(&NodeId(0)) = s.elite_sample(&nodes, &mut rng) {
                top_count += 1;
            }
        }
        let ratio = top_count as f32 / trials as f32;
        let expected = 1.0 / (1.0 + 0.5 + 1.0 / 3.0); // ≈ 0.545
        assert!(
            (ratio - expected).abs() < 0.02,
            "top-rank ratio {ratio:.4}, expected ≈{expected:.4}"
        );
    }

    #[test]
    fn entropy_uniform_max() {
        // Uniform distribution over 4 branches → H = log(4).
        let counts = [1u32, 1, 1, 1];
        let h = EntropyGatedScheduler::branch_selection_entropy(&counts);
        assert!((h - 4.0f32.ln()).abs() < 1e-5, "uniform entropy = log(N)");
        // exp(H) ≈ 4
        assert!((EntropyGatedScheduler::effective_branch_count(&counts) - 4.0).abs() < 1e-3);
    }

    #[test]
    fn entropy_degenerate_zero() {
        let counts = [4u32, 0, 0, 0];
        let h = EntropyGatedScheduler::branch_selection_entropy(&counts);
        assert!(h.abs() < 1e-5, "degenerate entropy = 0");
        assert!((EntropyGatedScheduler::effective_branch_count(&counts) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn entropy_empty_zero() {
        let counts: [u32; 0] = [];
        assert_eq!(EntropyGatedScheduler::branch_selection_entropy(&counts), 0.0);
    }
}
