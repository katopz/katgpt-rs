//! # CGSP — Curiosity-Guided Self-Play (Plan 274)
//!
//! Generic, modelless, zero-allocation triad that fuses the SGS paper's
//! Solver / Conjecturer / Guide architecture with existing katgpt-rs
//! primitives (Hint-δ bandit from Plan 049, collapse_aware_thinking from
//! Plan 212, data_gate from Plan 111, breakeven_complexity from Plan 250).
//!
//! **No game semantics here.** Game IP lives in `riir-ai` Plan 299.
//!
//! ## Feature gate
//!
//! `cgsp` — opt-in initially. Promoted to default-on only after the GOAT
//! gate (G1–G6) in Plan 274 Phase 3 passes.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use katgpt::cgsp::{
//!     CgspConfig, CgspLoop, ColinearityBatchGate, HlaProjectionGuide,
//!     PoolConjecturer, BreakevenDifficultyFilter, ScratchBuffers, Target,
//! };
//!
//! let pool = vec![/* direction vectors */];
//! let conjecturer = PoolConjecturer::new(pool.clone(), 42);
//! let guide = HlaProjectionGuide::new(2.0, 1.0, Default::default());
//! // solver + bandit provided by caller
//! let mut loop_ = CgspLoop::new(conjecturer, guide, solver, bandit, CgspConfig::default())
//!     .with_difficulty_filter(BreakevenDifficultyFilter::default())
//!     .with_batch_gate(ColinearityBatchGate::default());
//! let mut scratch = ScratchBuffers::new(4, pool.len());
//! let target = Target::new(pool[0].clone());
//! let result = loop_.cycle(&target, &mut scratch);
//! ```

pub mod conjecturer;
#[cfg(feature = "temporal_deriv")]
pub mod derivative_curiosity;
pub mod filters;
pub mod guide;
pub mod loop_;
pub mod traits;
pub mod types;

// Dual-pool reachable memory router (Plan 282, Research 249 — DecentMem).
// Opt-in until G1–G5 GOAT gate passes.
#[cfg(feature = "cgsp_dual_pool")]
pub mod dual_pool;

// Convenience re-exports — flat namespace for callers.
#[cfg(feature = "temporal_deriv")]
pub use derivative_curiosity::DerivativeCuriosity;
pub use conjecturer::PoolConjecturer;
pub use filters::{BreakevenDifficultyFilter, ColinearityBatchGate};
pub use guide::{structural_complexity, ComplexityWeights, HlaProjectionGuide};
pub use loop_::{CgspConfig, CgspLoop, EntropyCollapse};
pub use traits::{
    BatchQualityGate, CollapseSignal, CuriosityConjecturer, DifficultyFilter, HintDeltaBandit,
    NoOpBatchGate, NoOpDifficultyFilter, QualityGuide, Solver,
};
pub use types::{
    entropy_nats, sigmoid, Candidate, CuriosityPrioritySnapshot, CycleResult, CycleStats,
    Direction, Priority, ScratchBuffers, SolveRate, Target, DEFAULT_HLA_DIM, DEFAULT_K,
    DEFAULT_POOL_SIZE,
};

// Dual-pool re-exports (Plan 282, Research 249).
#[cfg(feature = "cgsp_dual_pool")]
pub use dual_pool::{DualPoolBandit, DualPoolConfig, PoolId, ReachableDualPoolRouter};

// ── Integration tests (T1.7) ──────────────────────────────────────────────

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cgsp::traits::{HintDeltaBandit, Solver};

    /// Test bandit: priority table backed by a Vec<f32>, with simple additive
    /// absorb and renormalize-on-read.
    pub(crate) struct VecBandit {
        prios: Vec<f32>,
    }
    impl VecBandit {
        pub(crate) fn uniform(n: usize) -> Self {
            Self {
                prios: vec![1.0 / n as f32; n],
            }
        }
    }
    impl HintDeltaBandit for VecBandit {
        fn absorb(&mut self, arm: usize, reward: f32) {
            if let Some(p) = self.prios.get_mut(arm) {
                *p += reward.max(0.0);
            }
        }
        fn priority(&self, arm: usize) -> Priority {
            self.prios.get(arm).copied().unwrap_or(0.0)
        }
        fn priorities(&self) -> &[Priority] {
            &self.prios
        }
        fn priorities_mut(&mut self) -> &mut [Priority] {
            &mut self.prios
        }
    }

    /// Solver that returns a solve-rate proportional to dot-product with the
    /// target via sigmoid. Reproducible and deterministic.
    pub(crate) struct DotSolver {
        pub sharpness: f32,
    }
    impl Solver for DotSolver {
        fn attempt(&mut self, target: &Target, candidate: &Candidate) -> f32 {
            let d = candidate.direction.dot(&target.direction);
            sigmoid(self.sharpness * d)
        }
    }

    fn make_8_direction_pool() -> Vec<Direction> {
        // 8 orthonormal-ish directions in 8-D.
        (0..8)
            .map(|i| {
                let mut coords = vec![0.0f32; 8];
                coords[i] = 1.0;
                if i >= 1 {
                    coords[(i + 1) % 8] = 0.1;
                }
                let norm: f32 = coords.iter().map(|c| c * c).sum::<f32>().sqrt();
                for c in &mut coords {
                    *c /= norm.max(1e-9);
                }
                Direction { coords }
            })
            .collect()
    }

    #[test]
    fn integration_full_cycle_no_panic_no_nan() {
        let pool = make_8_direction_pool();
        let conj = PoolConjecturer::new(pool.clone(), 12345);
        let guide = HlaProjectionGuide::new(3.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 2.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default())
            .with_difficulty_filter(BreakevenDifficultyFilter::default())
            .with_batch_gate(ColinearityBatchGate::default());
        let target = Target::new(pool[0].clone()).with_priority_hint(0.9);
        let mut scratch = ScratchBuffers::new(8, 8);

        for cycle in 0..100 {
            let r = lp.cycle(&target, &mut scratch);
            // No NaN / infinity anywhere observable.
            assert!(r.stats.priority_entropy.is_finite(), "cycle {cycle}: entropy NaN");
            assert!(
                r.stats.mean_guide_score.is_finite(),
                "cycle {cycle}: guide score NaN"
            );
            assert!(
                r.stats.mean_r_synth.is_finite(),
                "cycle {cycle}: r_synth NaN"
            );
            for &p in lp.bandit().priorities() {
                assert!(p.is_finite(), "cycle {cycle}: priority NaN");
                assert!(p >= 0.0, "cycle {cycle}: priority negative");
            }
        }
    }

    #[test]
    fn integration_priorities_converge_toward_target() {
        // After 100 cycles targeting arm 0, arm 0 should have higher priority
        // than the median arm.
        let pool = make_8_direction_pool();
        let conj = PoolConjecturer::new(pool.clone(), 7);
        let guide = HlaProjectionGuide::new(4.0, 0.5, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 2.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        for _ in 0..100 {
            let _ = lp.cycle(&target, &mut scratch);
        }

        let prios = lp.bandit().priorities();
        let target_p = prios[0];
        let mut sorted: Vec<f32> = prios.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        assert!(
            target_p >= median,
            "target-aligned arm ({target_p}) should beat median ({median})"
        );
    }

    #[test]
    fn integration_snapshot_persists_priorities() {
        let pool = make_8_direction_pool();
        let conj = PoolConjecturer::new(pool.clone(), 99);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
        let target = Target::new(pool[3].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        for _ in 0..20 {
            let _ = lp.cycle(&target, &mut scratch);
        }
        let snap = lp.snapshot();
        // BLAKE3 commitment is well-formed.
        let h = snap.blake3_hash();
        assert!(h.iter().any(|&b| b != 0), "BLAKE3 should not be all zeros");

        // Round-trip priorities through encode/decode.
        let mut buf = Vec::new();
        snap.encode_to(&mut buf);
        let back = CuriosityPrioritySnapshot::decode(&buf).expect("decode");
        assert_eq!(back.priorities, snap.priorities);
    }

    #[test]
    fn integration_collapse_recovery() {
        // Force collapse, verify the loop recovers (entropy rises above τ_low)
        // within a handful of cycles. The collapse_aware path injects
        // exploration after detecting low entropy, so the NEXT cycle's
        // entropy should be measurably higher.
        let pool = make_8_direction_pool();
        let conj = PoolConjecturer::new(pool.clone(), 5);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        // Force one-hot collapse.
        for (i, p) in lp.bandit_mut().priorities_mut().iter_mut().enumerate() {
            *p = if i == 3 { 1.0 } else { 0.0 };
        }
        let h_collapsed = entropy_nats(lp.bandit().priorities());
        assert!(h_collapsed < 0.30, "collapsed entropy should be low: {h_collapsed}");

        // Run cycles — the collapse_aware path should inject exploration,
        // so within a few cycles entropy should rise above the collapsed
        // baseline. We don't break early so we can observe the recovery.
        let mut max_h = h_collapsed;
        let mut triggered = false;
        for _ in 0..10 {
            let r = lp.cycle(&target, &mut scratch);
            if r.collapse_triggered {
                triggered = true;
            }
            // After cycle returns, the priority table has been updated —
            // sample its entropy directly to see post-injection state.
            let live_h = entropy_nats(lp.bandit().priorities());
            if live_h > max_h {
                max_h = live_h;
            }
        }
        assert!(triggered, "collapse should have triggered at least once");
        assert!(
            max_h > h_collapsed,
            "entropy should rise after recovery: {h_collapsed} -> {max_h}"
        );
    }
}
