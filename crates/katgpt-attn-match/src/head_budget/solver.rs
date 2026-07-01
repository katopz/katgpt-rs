//! Greedy-swap head budget solver (Plan 271 Phase 3, Algorithm 4).
//!
//! Given per-head sensitivity curves and a target overall ratio, find a
//! per-head share allocation that minimizes total quality loss. The greedy
//! swap algorithm is a coordinate-descent variant: repeatedly move a small
//! budget `η` from the head with the smallest marginal loss to the head with
//! the largest marginal gain, until no improving swap exists.
//!
//! # Algorithm (informal)
//! 1. Initialize each head with `target_ratio` (uniform per-head ratio).
//! 2. Repeat:
//!    - Find the head `g` with the largest `marginal_gain(r_g, η)`.
//!    - Find the head `l` with the largest `marginal_gain(r_l - η, η)` (i.e.,
//!      the head whose loss from giving up η is smallest — equivalently, the
//!      one whose curve is flattest at its current ratio).
//!    - If `gain_g > loss_l` (net improvement), transfer η from `l` to `g`.
//!    - Else stop.
//! 3. Return the final per-head ratios.
//!
//! # Convergence
//! Each improving swap strictly decreases total quality loss, and total loss
//! is bounded below by 0, so the algorithm terminates. Worst-case iteration
//! count is `O(num_heads / η)` but in practice far fewer.

#![allow(clippy::needless_range_loop)]

/// Convergence threshold for the greedy swap: if the net quality gain from
/// the best swap is below this, stop. We use a larger threshold than
/// `STABILITY_EPS` (1e-12) because f32 interpolation rounding can produce
/// tiny spurious gains (~1e-7) when curves are symmetric.
const SWAP_EPS: f32 = 1e-6;

use super::curve::HeadSensitivityCurve;
use crate::STABILITY_EPS;

/// Default step size η for the greedy swap. Smaller η → finer-grained but
/// more iterations. 0.05 gives ~20 steps per head at full ratio range, which
/// is fine-grained enough for typical sensitivity curves.
pub const DEFAULT_STEP_SIZE: f32 = 0.05;

/// Maximum number of greedy-swap iterations before bailing out. Acts as a
/// safety net — convergence should happen well before this. With 32 heads
/// and η=0.05, the theoretical max is ~640 swaps.
pub const MAX_ITERATIONS: usize = 10_000;

/// Per-head budget solver.
pub struct HeadBudgetSolver {
    curves: Vec<HeadSensitivityCurve>,
    num_layers: usize,
    num_heads: usize,
    step_size: f32,
}

impl HeadBudgetSolver {
    /// Construct a solver.
    ///
    /// # Arguments
    /// * `curves` - One curve per (layer, head). Length must be
    ///   `num_layers * num_heads`. Curve `i` describes head `i %
    ///   num_heads` in layer `i / num_heads`.
    /// * `num_layers` - Number of transformer layers.
    /// * `num_heads` - Number of attention heads per layer.
    ///
    /// # Panics
    /// Panics if `curves.len() != num_layers * num_heads`.
    pub fn new(curves: Vec<HeadSensitivityCurve>, num_layers: usize, num_heads: usize) -> Self {
        let expected = num_layers * num_heads;
        assert_eq!(
            curves.len(),
            expected,
            "curves.len()={} but num_layers*num_heads={}",
            curves.len(),
            expected
        );
        assert!(num_heads > 0, "num_heads must be > 0");
        assert!(num_layers > 0, "num_layers must be > 0");
        Self {
            curves,
            num_layers,
            num_heads,
            step_size: DEFAULT_STEP_SIZE,
        }
    }

    /// Override the default step size η.
    #[inline]
    pub fn with_step_size(mut self, step: f32) -> Self {
        assert!(step > 0.0, "step_size must be > 0");
        self.step_size = step;
        self
    }

    /// Step size η used by the solver.
    #[inline]
    pub fn step_size(&self) -> f32 {
        self.step_size
    }

    /// Number of curves (one per (layer, head) pair).
    #[inline]
    pub fn num_curves(&self) -> usize {
        self.curves.len()
    }

    /// Number of transformer layers this solver covers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of attention heads per layer.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Solve for the per-head budget that minimizes total quality loss,
    /// subject to the per-head ratios averaging to `target_ratio`.
    ///
    /// Returns a flat `Vec<f32>` of length `num_layers * num_heads`. Each
    /// entry is the ratio (in `[0, 1]`) allocated to that head. The arithmetic
    /// mean of the returned ratios equals `target_ratio` (up to floating-point
    /// rounding from the discrete step size).
    ///
    /// # Algorithm
    /// Greedy swap (paper Algorithm 4):
    /// 1. Start with uniform `target_ratio` for every head.
    /// 2. Repeatedly find the head whose marginal gain from +η is highest
    ///    and the head whose marginal loss from -η is lowest. If net gain,
    ///    swap η between them.
    /// 3. Stop when no improving swap exists (or `MAX_ITERATIONS` reached).
    pub fn solve(&self, target_ratio: f32) -> Vec<f32> {
        let n = self.curves.len();
        assert!(n > 0);
        assert!(
            (0.0..=1.0).contains(&target_ratio),
            "target_ratio must be in [0, 1], got {}",
            target_ratio
        );

        // Initialize uniform allocation. Each head starts at target_ratio.
        let mut ratios = vec![target_ratio; n];

        // We must keep the average at target_ratio: Σ ratios / n = target_ratio.
        // A swap moves η from head l to head g: r_l -= η, r_g += η. This
        // preserves the sum (and hence the average) by construction.
        let step = self.step_size;

        // Cached gain[i] = marginal_gain(ratios[i], step)
        // and loss[i] = marginal_gain(ratios[i] - step, step) = quality cost of donating η.
        // After each swap only 2 entries change, so we keep these in sync
        // incrementally and scan the cache (O(n) reads, no curve interpolation)
        // instead of recomputing all 2n interpolations per iteration.
        //
        // Sentinel encoding eliminates the `can_recv` / `can_donate` bool arrays:
        // gains[i] = NEG_INFINITY  ⟺ head i cannot receive (+η would exceed 1.0)
        // losses[i] = INFINITY     ⟺ head i cannot donate (r < η)
        // Both sentinels are never selected by the argmax/argmin scans, so the
        // per-iteration inner loops are branch-free (no flag lookup).
        let mut gains = vec![0.0f32; n];
        let mut losses = vec![0.0f32; n];
        for (i, g) in gains.iter_mut().enumerate() {
            let r = ratios[i];
            *g = if r + step <= 1.0 + STABILITY_EPS {
                self.curves[i].marginal_gain(r, step)
            } else {
                f32::NEG_INFINITY
            };
        }
        for (i, l) in losses.iter_mut().enumerate() {
            let r = ratios[i];
            *l = if r >= step {
                self.curves[i].marginal_gain(r - step, step)
            } else {
                f32::INFINITY
            };
        }

        // Safety: bound iterations to prevent pathological loops.
        let mut iter = 0usize;
        while iter < MAX_ITERATIONS {
            iter += 1;

            // Pass 1: find head `g` with the largest cached gain. Heads that
            // cannot receive carry NEG_INFINITY, so they never win the argmax —
            // no separate flag scan needed.
            let mut best_gain_idx: Option<usize> = None;
            let mut best_gain = f32::NEG_INFINITY;
            for i in 0..n {
                if gains[i] > best_gain {
                    best_gain = gains[i];
                    best_gain_idx = Some(i);
                }
            }
            let Some(gain_idx) = best_gain_idx else {
                break; // No head can receive → done.
            };

            // Pass 2: find head `l` (≠ g) with the smallest cached loss. Heads
            // that cannot donate carry INFINITY, so they never win the argmin —
            // no separate flag scan needed.
            let mut best_loss_idx: Option<usize> = None;
            let mut best_loss = f32::INFINITY;
            for i in 0..n {
                if i == gain_idx {
                    continue;
                }
                if losses[i] < best_loss {
                    best_loss = losses[i];
                    best_loss_idx = Some(i);
                }
            }
            let Some(loss_idx) = best_loss_idx else {
                break; // No head can donate → done.
            };

            // Net quality change from swap = gain - loss. If positive, swap.
            if !best_gain.is_finite() || !best_loss.is_finite() {
                break;
            }
            let net = best_gain - best_loss;
            if net <= SWAP_EPS {
                break;
            }

            // Apply the swap and incrementally update only the 2 touched heads.
            ratios[gain_idx] += step;
            ratios[loss_idx] -= step;
            // Refresh sentinel-encoded cached curves for both heads.
            for &idx in &[gain_idx, loss_idx] {
                let r = ratios[idx];
                gains[idx] = if r + step <= 1.0 + STABILITY_EPS {
                    self.curves[idx].marginal_gain(r, step)
                } else {
                    f32::NEG_INFINITY
                };
                losses[idx] = if r >= step {
                    self.curves[idx].marginal_gain(r - step, step)
                } else {
                    f32::INFINITY
                };
            }
        }

        ratios
    }

    /// Compute the total quality loss for a given per-head ratio allocation,
    /// evaluated against the stored curves. Useful for verifying convergence
    /// (a converged solution should have no improving neighbor).
    pub fn total_quality_loss(&self, ratios: &[f32]) -> f32 {
        assert_eq!(ratios.len(), self.curves.len());
        let mut total = 0.0f32;
        for (r, c) in ratios.iter().zip(self.curves.iter()) {
            total += c.interpolate(*r);
        }
        total
    }

    /// Verify that no single-swap neighbor improves the solution. Used by the
    /// convergence test. Returns `true` if the solution is locally optimal
    /// (no improving single swap).
    pub fn is_locally_optimal(&self, ratios: &[f32]) -> bool {
        let n = self.curves.len();
        let step = self.step_size;
        for g in 0..n {
            if ratios[g] + step > 1.0 + STABILITY_EPS {
                continue;
            }
            let gain = self.curves[g].marginal_gain(ratios[g], step);
            for l in 0..n {
                if l == g {
                    continue;
                }
                if ratios[l] < step {
                    continue;
                }
                let loss = self.curves[l].marginal_gain(ratios[l] - step, step);
                if gain - loss > SWAP_EPS {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a flat curve (same delta for all ratios → insensitive).
    fn flat_curve(head_id: usize, delta: f32) -> HeadSensitivityCurve {
        HeadSensitivityCurve::new(head_id, vec![0.1, 1.0], vec![delta, delta])
    }

    /// Helper: build a steep curve (sensitive — quality drops fast as r drops).
    fn steep_curve(head_id: usize) -> HeadSensitivityCurve {
        // delta grows quickly as r shrinks: at r=0.1 → 0.9, at r=1.0 → 0.
        HeadSensitivityCurve::new(head_id, vec![0.1, 0.5, 1.0], vec![0.9, 0.4, 0.0])
    }

    #[test]
    fn test_uniform_allocation_equal_shares() {
        // All heads have identical curves → uniform solution.
        let curves = vec![
            flat_curve(0, 0.5),
            flat_curve(1, 0.5),
            flat_curve(2, 0.5),
            flat_curve(3, 0.5),
        ];
        let solver = HeadBudgetSolver::new(curves, 1, 4);
        let shares = solver.solve(0.5);
        assert_eq!(shares.len(), 4);
        // Every head gets target_ratio exactly.
        for &s in &shares {
            assert!((s - 0.5).abs() < 1e-6, "expected uniform 0.5, got {}", s);
        }
        let avg: f32 = shares.iter().sum::<f32>() / shares.len() as f32;
        assert!((avg - 0.5).abs() < 1e-6, "average must equal target_ratio");
    }

    #[test]
    fn test_greedy_swap_converges() {
        // Mixed curves: heads 0,2 steep; heads 1,3 flat. Steep heads should
        // receive more budget.
        let curves = vec![
            steep_curve(0),
            flat_curve(1, 0.1),
            steep_curve(2),
            flat_curve(3, 0.1),
        ];
        let solver = HeadBudgetSolver::new(curves, 1, 4).with_step_size(0.05);
        let shares = solver.solve(0.3);
        // The solution must be locally optimal: no improving single swap.
        assert!(
            solver.is_locally_optimal(&shares),
            "solver did not converge to local optimum: shares={:?}",
            shares
        );
    }

    #[test]
    fn test_solver_handles_sensitive_heads() {
        // 4 heads, one is much steeper than the others.
        let curves = vec![
            steep_curve(0),      // very sensitive
            flat_curve(1, 0.05), // barely matters
            flat_curve(2, 0.05),
            flat_curve(3, 0.05),
        ];
        let solver = HeadBudgetSolver::new(curves, 1, 4).with_step_size(0.05);
        let shares = solver.solve(0.5);
        // The steep head should get more than the flat ones.
        assert!(
            shares[0] > shares[1],
            "steep head should get more budget: shares={:?}",
            shares
        );
        assert!(
            shares[0] > shares[2],
            "steep head should get more budget: shares={:?}",
            shares
        );
        assert!(
            shares[0] > shares[3],
            "steep head should get more budget: shares={:?}",
            shares
        );
        // Note: the greedy swap drains flat heads sequentially (lowest index
        // first when losses tie), so flat heads are NOT necessarily equal.
        // We only assert each flat head is below the steep head and above 0.
        for &s in &shares[1..] {
            assert!(s >= 0.0, "flat head share {} should be non-negative", s);
            assert!(s < shares[0], "flat head should have less than steep head");
        }
        // Conservation: average must equal target.
        let avg: f32 = shares.iter().sum::<f32>() / shares.len() as f32;
        assert!(
            (avg - 0.5).abs() < 1e-3,
            "average ratio {} should equal target 0.5",
            avg
        );
        // Solution must be locally optimal.
        assert!(
            solver.is_locally_optimal(&shares),
            "solver should converge to local optimum: shares={:?}",
            shares
        );
    }

    #[test]
    fn test_solver_preserves_total_budget() {
        let curves = vec![
            HeadSensitivityCurve::new(0, vec![0.1, 0.3, 0.7, 1.0], vec![0.8, 0.5, 0.2, 0.0]),
            HeadSensitivityCurve::new(1, vec![0.1, 0.3, 0.7, 1.0], vec![0.6, 0.4, 0.1, 0.0]),
            HeadSensitivityCurve::new(2, vec![0.1, 0.3, 0.7, 1.0], vec![0.7, 0.3, 0.15, 0.0]),
        ];
        let solver = HeadBudgetSolver::new(curves, 1, 3).with_step_size(0.02);
        for &target in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let shares = solver.solve(target);
            let avg: f32 = shares.iter().sum::<f32>() / shares.len() as f32;
            assert!(
                (avg - target).abs() < 1e-2,
                "target {}: avg ratio {} leaked budget",
                target,
                avg
            );
        }
    }

    #[test]
    fn test_solver_multi_layer() {
        // 2 layers × 3 heads = 6 curves.
        let curves: Vec<HeadSensitivityCurve> = (0..6)
            .map(|i| HeadSensitivityCurve::new(i, vec![0.1, 0.5, 1.0], vec![0.8, 0.3, 0.0]))
            .collect();
        let solver = HeadBudgetSolver::new(curves, 2, 3);
        let shares = solver.solve(0.4);
        assert_eq!(shares.len(), 6);
        // Identical curves → uniform.
        for &s in &shares {
            assert!((s - 0.4).abs() < 1e-6, "uniform expected, got {}", s);
        }
    }

    #[test]
    #[should_panic(expected = "curves.len()")]
    fn test_solver_rejects_curve_count_mismatch() {
        let curves = vec![flat_curve(0, 0.5)];
        let _ = HeadBudgetSolver::new(curves, 2, 3); // expected 6, got 1
    }

    #[test]
    #[should_panic(expected = "target_ratio must be in")]
    fn test_solver_rejects_invalid_target() {
        let curves = vec![flat_curve(0, 0.5), flat_curve(1, 0.5)];
        let solver = HeadBudgetSolver::new(curves, 1, 2);
        let _ = solver.solve(2.0);
    }
}
