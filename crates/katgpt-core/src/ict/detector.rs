//! ICT distributional branching-point detector.
//!
//! Plan 294, Research 270 §2.4. The runtime structure that consumes K
//! candidate trajectories per step and emits a per-trajectory branching
//! mask + per-step β + per-trajectory JS-uniqueness.
//!
//! ## Algorithm (R270 §2.4)
//!
//! ```text
//! For each tick t:
//!   1. Receive K candidate action distributions {π_k(·|t)}.
//!   2. P̄(·|t) = mean over k of π_k.
//!   3. u_k(t) = D_JS(π_k ‖ P̄)                 # uniqueness per trajectory
//!   4. β(t)   = collision_purity(P̄)            # population concentration
//!   5. mask[k] = top-k_percent(u_k)             # critical trajectories
//!   6. (optional) EMA-update ema_alpha/ema_beta for downstream Bebop/Curiosity
//! ```
//!
//! ## Zero-allocation hot path
//!
//! All scratch buffers are pre-allocated in [`BranchingDetector::new`]. The
//! only allocation in [`BranchingDetector::observe_and_detect`] is the
//! returned [`BranchingReport`] (which is a snapshot — callers in a tight
//! loop should use [`BranchingDetector::observe_and_detect_into`] to write
//! into a pre-allocated report instead). G5 (`bench_294_ict_g5.rs`) verifies
//! this.

use crate::ict::branching::branching_point_mask_into;
use crate::ict::math::{collision_purity, js_divergence};
use crate::ict::types::BranchingReport;

/// Population-level ICT branching-point detector.
///
/// Construct once with `(k_trajectories, action_dim, k_percent, eta)`, then
/// call [`observe_and_detect`] each tick with the K candidate action
/// distributions. The detector returns a [`BranchingReport`] containing the
/// per-trajectory branching mask, the population-mean β, and the per-trajectory
/// uniqueness scores.
///
/// All scratch is pre-allocated in [`new`](Self::new); the hot path is
/// zero-alloc via [`observe_and_detect_into`](Self::observe_and_detect_into).
///
/// # Example
///
/// ```
/// # use katgpt_core::ict::detector::BranchingDetector;
/// // 4 trajectories, action_dim 3, top 25%, η = 0.05.
/// let mut det = BranchingDetector::new(4, 3, 0.25, 0.05);
/// let t0 = [0.5_f32, 0.3, 0.2];
/// let t1 = [0.1_f32, 0.8, 0.1];
/// let t2 = [0.34_f32, 0.33, 0.33];
/// let t3 = [0.6_f32, 0.2, 0.2];
/// let trajs: Vec<&[f32]> = vec![&t0, &t1, &t2, &t3];
/// let report = det.observe_and_detect(&trajs);
/// assert_eq!(report.mask.len(), 4);
/// assert_eq!(report.uniqueness_scores.len(), 4);
/// ```
///
/// [`observe_and_detect`]: Self::observe_and_detect
pub struct BranchingDetector {
    /// Number of candidate trajectories per tick. Fixed at construction.
    pub k_trajectories: usize,
    /// Action distribution dimension (vocab size, action count, etc.).
    pub action_dim: usize,
    /// Top-k% selector sparsity. ICT §A.4.1 reports 0.10 for LLM tokens.
    pub k_percent: f32,
    /// Critical-branching tolerance `|π(a*) − β| < η`.
    pub eta: f32,

    // ── Pre-allocated scratch (zero-alloc hot path) ──
    /// Population mean: `(1/K) Σ_k π_k`. Length `action_dim`.
    scratch_p_avg: Vec<f32>,
    /// Mid-point `(π_k + P̄) / 2` for the per-trajectory JS call. Length
    /// `action_dim`. Reused across k.
    scratch_m: Vec<f32>,
    /// Per-trajectory uniqueness scores `u_k = JS(π_k, P̄)`. Length `k_trajectories`.
    scratch_u: Vec<f32>,
    /// Per-trajectory branching mask. Length `k_trajectories`.
    scratch_mask: Vec<bool>,
    /// Scratch buffer for sorting uniqueness scores to find the top-k
    /// threshold. Length `k_trajectories`. Kept as a field so the hot path
    /// `observe_and_detect_into` is zero-alloc (G5 contract).
    scratch_sorted: Vec<f32>,

    /// EMA of β across calls. `ema_beta = (1 - ema_decay) · ema_beta + ema_decay · β(t)`.
    /// Read by downstream consumers (Bebop, Curiosity Pulse) as a smoothed
    /// concentration signal. Updated every [`observe_and_detect`] call.
    pub ema_beta: f32,
    /// EMA of population-mean action probability (max over actions). Read by
    /// downstream consumers as a smoothed "decisiveness" signal.
    pub ema_alpha: f32,
    /// EMA decay factor in `[0, 1]`. Default 0.9 (half-life ~6.6 ticks).
    pub ema_decay: f32,
}

impl BranchingDetector {
    /// Construct a detector with pre-allocated scratch.
    ///
    /// Panics in debug if `k_trajectories == 0` or `action_dim == 0`. The
    /// `ema_decay` defaults to `0.9`.
    pub fn new(k_trajectories: usize, action_dim: usize, k_percent: f32, eta: f32) -> Self {
        debug_assert!(k_trajectories > 0, "k_trajectories must be > 0");
        debug_assert!(action_dim > 0, "action_dim must be > 0");
        Self {
            k_trajectories,
            action_dim,
            k_percent,
            eta,
            scratch_p_avg: vec![0.0; action_dim],
            scratch_m: vec![0.0; action_dim],
            scratch_u: vec![0.0; k_trajectories],
            scratch_mask: vec![false; k_trajectories],
            scratch_sorted: vec![0.0; k_trajectories],
            ema_beta: 0.0,
            ema_alpha: 0.0,
            ema_decay: 0.9,
        }
    }

    /// Override the default EMA decay (0.9). Must be in `[0, 1]`; values
    /// outside are clamped.
    pub fn with_ema_decay(mut self, decay: f32) -> Self {
        self.ema_decay = if decay < 0.0 {
            0.0
        } else if decay > 1.0 {
            1.0
        } else {
            decay
        };
        self
    }

    /// Snapshot accessor for the pre-allocated branching mask (for callers
    /// that want to read the most recent result without cloning the report).
    ///
    /// Length is `k_trajectories`. Valid until the next `observe_and_detect*`
    /// call overwrites it.
    #[inline]
    pub fn last_mask(&self) -> &[bool] {
        &self.scratch_mask
    }

    /// Snapshot accessor for the most recent per-trajectory uniqueness scores.
    #[inline]
    pub fn last_uniqueness_scores(&self) -> &[f32] {
        &self.scratch_u
    }

    /// Snapshot accessor for the most recent population-mean distribution.
    #[inline]
    pub fn last_population_mean(&self) -> &[f32] {
        &self.scratch_p_avg
    }

    /// Run one detection step on K candidate trajectories.
    ///
    /// `trajectories.len()` must equal `self.k_trajectories` and each
    /// trajectory must have length `self.action_dim`; mismatches are silently
    /// ignored (the report comes back zeroed). The returned [`BranchingReport`]
    /// is a fresh allocation — for the zero-alloc path use
    /// [`observe_and_detect_into`](Self::observe_and_detect_into).
    pub fn observe_and_detect(&mut self, trajectories: &[&[f32]]) -> BranchingReport {
        let mut report = BranchingReport {
            mask: vec![false; self.k_trajectories],
            beta_per_step: vec![0.0; self.k_trajectories],
            uniqueness_scores: vec![0.0; self.k_trajectories],
        };
        self.observe_and_detect_into(trajectories, &mut report);
        report
    }

    /// Zero-alloc variant — writes the result into the caller-provided
    /// `report`. `report.mask`, `report.uniqueness_scores`, and
    /// `report.beta_per_step` must each have length `>= k_trajectories`; the
    /// detector will only touch the first `k_trajectories` slots.
    ///
    /// This is the hot path that G5 (`bench_294_ict_g5.rs`) verifies
    /// allocates zero bytes after warmup.
    pub fn observe_and_detect_into(&mut self, trajectories: &[&[f32]], report: &mut BranchingReport) {
        let k = self.k_trajectories;
        let n = self.action_dim;

        // ── Guard: shape mismatches → write zeros, return. ──
        if trajectories.len() != k
            || report.mask.len() < k
            || report.uniqueness_scores.len() < k
            || report.beta_per_step.len() < k
        {
            for i in 0..report.mask.len().min(k) {
                report.mask[i] = false;
                report.uniqueness_scores[i] = 0.0;
                report.beta_per_step[i] = 0.0;
            }
            return;
        }
        for t in trajectories {
            if t.len() != n {
                for i in 0..k {
                    report.mask[i] = false;
                    report.uniqueness_scores[i] = 0.0;
                    report.beta_per_step[i] = 0.0;
                }
                return;
            }
        }

        // ── Step 2: P̄ = (1/K) Σ_k π_k, written into scratch_p_avg. ──
        // Chunked-4 accumulation helps autovectorization per AGENTS.md.
        for slot in self.scratch_p_avg[..n].iter_mut() {
            *slot = 0.0;
        }
        for traj in trajectories {
            let mut a = 0;
            while a + 4 <= n {
                self.scratch_p_avg[a] += traj[a];
                self.scratch_p_avg[a + 1] += traj[a + 1];
                self.scratch_p_avg[a + 2] += traj[a + 2];
                self.scratch_p_avg[a + 3] += traj[a + 3];
                a += 4;
            }
            while a < n {
                self.scratch_p_avg[a] += traj[a];
                a += 1;
            }
        }
        let inv_k = 1.0_f32 / (k as f32);
        for slot in self.scratch_p_avg[..n].iter_mut() {
            *slot *= inv_k;
        }

        // ── Step 4: β(t) = collision_purity(P̄). ──
        let beta = collision_purity(&self.scratch_p_avg);

        // ── Step 3: u_k = JS(π_k, P̄). Reuse scratch_m as the (π_k + P̄)/2 buffer. ──
        for (i, traj) in trajectories.iter().enumerate() {
            self.scratch_u[i] = js_divergence(traj, &self.scratch_p_avg, &mut self.scratch_m);
        }

        // ── Step 5: top-k_percent selection into scratch_mask. ──
        // Sort a copy of the scores into scratch_sorted to find the k-th
        // largest (order-statistic threshold), then write via
        // branching_point_mask_into. For K ≤ 32 a full sort is cheaper than
        // quickselect and deterministic. scratch_sorted is a pre-allocated
        // field so this path is zero-alloc (G5 contract).
        let k_select = ((self.k_percent * k as f32).ceil() as usize).max(1).min(k);
        // Sort into scratch_sorted (we keep one as a field for zero-alloc).
        self.scratch_sorted[..k].copy_from_slice(&self.scratch_u[..k]);
        // Could use sort_unstable_by for speed; sort_by is stable and fine
        // for K ≤ 32.
        self.scratch_sorted[..k].sort_by(|a, b| {
            b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal)
        });
        let threshold = self.scratch_sorted[k_select - 1];

        branching_point_mask_into(&self.scratch_u, threshold, &mut self.scratch_mask);
        // Enforce exact k_select count, ties broken by lower index.
        let mut count = 0;
        for i in 0..k {
            if self.scratch_mask[i] {
                count += 1;
                if count > k_select {
                    self.scratch_mask[i] = false;
                }
            }
        }

        // ── Emit the report. ──
        for i in 0..k {
            report.mask[i] = self.scratch_mask[i];
            report.uniqueness_scores[i] = self.scratch_u[i];
            // beta_per_step carries the population β for every column —
            // callers wanting per-trajectory β should recompute
            // collision_purity(π_k) themselves.
            report.beta_per_step[i] = beta;
        }

        // ── Step 6: EMA update for downstream consumers. ──
        let d = self.ema_decay;
        self.ema_beta = (1.0 - d) * self.ema_beta + d * beta;
        // ema_alpha tracks the max action probability in the population mean
        // (a smoothed "decisiveness" signal).
        let mut max_p = 0.0_f32;
        for &p in &self.scratch_p_avg[..n] {
            if p > max_p {
                max_p = p;
            }
        }
        self.ema_alpha = (1.0 - d) * self.ema_alpha + d * max_p;
    }

    /// Clear EMA state. Scratch buffers are NOT cleared (they're overwritten
    /// on the next call anyway).
    pub fn reset(&mut self) {
        self.ema_beta = 0.0;
        self.ema_alpha = 0.0;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests — Plan 294 Phase 1 T1.6 (10 tests for detector.rs)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> BranchingDetector {
        BranchingDetector::new(4, 3, 0.25, 0.05)
    }

    #[test]
    fn new_preallocates_scratch() {
        let d = make_detector();
        assert_eq!(d.k_trajectories, 4);
        assert_eq!(d.action_dim, 3);
        assert_eq!(d.k_percent, 0.25);
        assert_eq!(d.eta, 0.05);
        assert_eq!(d.last_mask().len(), 4);
        assert_eq!(d.last_uniqueness_scores().len(), 4);
        assert_eq!(d.last_population_mean().len(), 3);
    }

    #[test]
    fn observe_returns_correct_lengths() {
        let mut d = make_detector();
        let t0 = [0.5_f32, 0.3, 0.2];
        let t1 = [0.1_f32, 0.8, 0.1];
        let t2 = [0.34_f32, 0.33, 0.33];
        let t3 = [0.6_f32, 0.2, 0.2];
        let trajs: Vec<&[f32]> = vec![&t0, &t1, &t2, &t3];
        let r = d.observe_and_detect(&trajs);
        assert_eq!(r.mask.len(), 4);
        assert_eq!(r.uniqueness_scores.len(), 4);
        assert_eq!(r.beta_per_step.len(), 4);
    }

    #[test]
    fn identical_trajectories_have_zero_uniqueness() {
        // All K trajectories identical → mean is the same → JS = 0 for all.
        let mut d = BranchingDetector::new(3, 4, 0.33, 0.05);
        let t = [0.25_f32, 0.25, 0.25, 0.25];
        let trajs: Vec<&[f32]> = vec![&t, &t, &t];
        let r = d.observe_and_detect(&trajs);
        for (i, &u) in r.uniqueness_scores.iter().enumerate() {
            assert!(u.abs() < 1e-5, "identical traj {i}: u={u}, expected ~0");
        }
    }

    #[test]
    fn disjoint_trajectory_has_high_uniqueness() {
        // One trajectory disjoint from others → high JS-to-mean.
        let mut d = BranchingDetector::new(3, 4, 0.34, 0.05);
        let t_normal = [0.25_f32, 0.25, 0.25, 0.25];
        let t_normal2 = [0.25_f32, 0.25, 0.25, 0.25];
        let t_disjoint = [0.0_f32, 0.0, 0.5, 0.5];
        let trajs: Vec<&[f32]> = vec![&t_normal, &t_normal2, &t_disjoint];
        let r = d.observe_and_detect(&trajs);
        // The disjoint trajectory should have the highest uniqueness.
        let u_disjoint = r.uniqueness_scores[2];
        let u_normal_max = r.uniqueness_scores[0].max(r.uniqueness_scores[1]);
        assert!(
            u_disjoint > u_normal_max,
            "disjoint traj should have highest u: disjoint={u_disjoint}, normal_max={u_normal_max}"
        );
    }

    #[test]
    fn top_k_percent_flags_highest_uniqueness() {
        // 4 trajectories, top 25% = 1. The most divergent should be flagged.
        let mut d = BranchingDetector::new(4, 3, 0.25, 0.05);
        let t0 = [0.4_f32, 0.3, 0.3];
        let t1 = [0.35_f32, 0.35, 0.3];
        let t2 = [0.33_f32, 0.34, 0.33];
        let t3 = [0.9_f32, 0.05, 0.05]; // very different
        let trajs: Vec<&[f32]> = vec![&t0, &t1, &t2, &t3];
        let r = d.observe_and_detect(&trajs);
        let count = r.branching_count();
        assert_eq!(count, 1, "top 25% of 4 should flag 1, got {count}");
        // t3 is most divergent; verify it has the highest u.
        let u_t3 = r.uniqueness_scores[3];
        let u_max = r.uniqueness_scores.iter().cloned().fold(0.0_f32, f32::max);
        assert!((u_t3 - u_max).abs() < 1e-5, "t3 should have max u");
        assert!(r.mask[3], "t3 should be flagged, mask={:?}", r.mask);
    }

    #[test]
    fn beta_per_step_matches_population_mean_collision_purity() {
        let mut d = BranchingDetector::new(2, 3, 0.5, 0.05);
        let t0 = [0.5_f32, 0.3, 0.2];
        let t1 = [0.1_f32, 0.8, 0.1];
        let trajs: Vec<&[f32]> = vec![&t0, &t1];
        let r = d.observe_and_detect(&trajs);
        // Manual: mean = [0.3, 0.55, 0.15], β = 0.09 + 0.3025 + 0.0225 = 0.415
        let expected_beta = 0.3 * 0.3 + 0.55 * 0.55 + 0.15 * 0.15;
        for &b in &r.beta_per_step {
            assert!((b - expected_beta).abs() < 1e-4, "β={b}, expected {expected_beta}");
        }
    }

    #[test]
    fn ema_updates_after_observe() {
        let mut d = BranchingDetector::new(2, 2, 0.5, 0.05).with_ema_decay(0.5);
        // Initially zero.
        assert!(d.ema_beta.abs() < 1e-6);
        assert!(d.ema_alpha.abs() < 1e-6);
        let t0 = [0.5_f32, 0.5];
        let t1 = [0.5_f32, 0.5];
        let trajs: Vec<&[f32]> = vec![&t0, &t1];
        let r = d.observe_and_detect(&trajs);
        // β = 0.25 + 0.25 = 0.5, max_p = 0.5, decay = 0.5 → EMA = 0.5 · 0.5 = 0.25
        assert!((d.ema_beta - 0.25).abs() < 1e-4, "ema_beta={}", d.ema_beta);
        assert!((d.ema_alpha - 0.25).abs() < 1e-4, "ema_alpha={}", d.ema_alpha);
        // beta_per_step should equal β = 0.5.
        assert!((r.beta_per_step[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn reset_clears_ema() {
        let mut d = make_detector();
        let t = [0.5_f32, 0.3, 0.2];
        let trajs: Vec<&[f32]> = vec![&t, &t, &t, &t];
        let _ = d.observe_and_detect(&trajs);
        assert!(d.ema_beta.abs() > 1e-6 || d.ema_alpha.abs() > 1e-6);
        d.reset();
        assert!(d.ema_beta.abs() < 1e-6);
        assert!(d.ema_alpha.abs() < 1e-6);
    }

    #[test]
    fn shape_mismatch_yields_zero_report() {
        let mut d = BranchingDetector::new(4, 3, 0.25, 0.05);
        // Wrong number of trajectories.
        let t0 = [0.5_f32, 0.3, 0.2];
        let trajs: Vec<&[f32]> = vec![&t0]; // only 1, expected 4
        let r = d.observe_and_detect(&trajs);
        for &u in &r.uniqueness_scores {
            assert!(u.abs() < 1e-6, "shape mismatch should give u=0, got {u}");
        }
        for &m in &r.mask {
            assert!(!m);
        }
    }

    #[test]
    fn wrong_action_dim_yields_zero_report() {
        let mut d = BranchingDetector::new(2, 3, 0.5, 0.05);
        let t0 = [0.5_f32, 0.5]; // wrong dim, expected 3
        let t1 = [0.3_f32, 0.7];
        let trajs: Vec<&[f32]> = vec![&t0, &t1];
        let r = d.observe_and_detect(&trajs);
        for &u in &r.uniqueness_scores {
            assert!(u.abs() < 1e-6, "wrong action_dim should give u=0, got {u}");
        }
    }

    #[test]
    fn observe_into_reuses_report_allocation() {
        let mut d = make_detector();
        let mut report = BranchingReport {
            mask: vec![false; 4],
            beta_per_step: vec![0.0; 4],
            uniqueness_scores: vec![0.0; 4],
        };
        let t0 = [0.4_f32, 0.3, 0.3];
        let t1 = [0.35_f32, 0.35, 0.3];
        let t2 = [0.33_f32, 0.34, 0.33];
        let t3 = [0.9_f32, 0.05, 0.05];
        let trajs: Vec<&[f32]> = vec![&t0, &t1, &t2, &t3];
        let ptr_before = report.mask.as_ptr();
        d.observe_and_detect_into(&trajs, &mut report);
        let ptr_after = report.mask.as_ptr();
        assert_eq!(
            ptr_before, ptr_after,
            "observe_into must reuse the report's allocation"
        );
        assert_eq!(report.branching_count(), 1);
    }
}
