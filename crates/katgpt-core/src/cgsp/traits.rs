//! CGSP traits — generic modelless contracts (Plan 274).
//!
//! These traits are deliberately narrow and game-agnostic. They let the
//! `CgspLoop` fuse the SGS triad (Solver / Conjecturer / Guide) with existing
//! katgpt-rs primitives without leaking any game IP into the public engine.
//!
//! # Trait surface
//!
//! - [`CuriosityConjecturer`] — proposes candidate direction vectors.
//! - [`QualityGuide`]         — scores candidates for relevance × elegance.
//! - [`Solver`]               — attempts a candidate and reports a solve-rate.
//! - [`HintDeltaBandit`]      — priority-table update driven by Hint-δ.
//! - [`BatchQualityGate`]     — degenerate-batch detector (`data_gate` analogue).
//! - [`CollapseSignal`]       — entropy-collapse detector / exploration injector.
//!
//! All trait methods are zero-allocation by contract — they either return
//! scalars or write into caller-provided scratch buffers.

use crate::cgsp::types::{Candidate, CycleResult, Direction, Priority, Target};

// ── CuriosityConjecturer ──────────────────────────────────────────────────

/// Frozen conjecturer that proposes candidate subgoal direction vectors.
///
/// Modelless counterpart of the SGS paper's `g_ϕ` (which is trained).
/// Implementations sample from a fixed direction pool weighted by the current
/// `priorities` table — no weights are mutated, only the priority table.
///
/// # Contract
///
/// - Writes exactly `out.len()` candidates into `out`.
/// - Does **not** allocate (caller-provided `out` slice).
/// - Sampling distribution reflects `priorities` (higher priority → more likely).
pub trait CuriosityConjecturer {
    /// Sample `out.len()` candidate directions weighted by `priorities`.
    ///
    /// `priorities` length must equal the conjecturer pool size. The
    /// conjecturer is free to perturb directions or sample with replacement.
    ///
    /// Takes `&mut self` because most non-trivial conjecturers carry
    /// internal RNG state that advances on each call. The direction pool
    /// itself stays frozen.
    ///
    /// `cdf_scratch` is caller-provided scratch for the priority-weighted
    /// CDF — kept as a separate arg so the conjecturer doesn't take a
    /// mutable borrow on the whole `ScratchBuffers` struct.
    fn sample_candidates(
        &mut self,
        target: &Target,
        priorities: &[Priority],
        out: &mut [Candidate],
        cdf_scratch: &mut Vec<f32>,
    );

    /// Pool size (number of distinct arms the conjecturer can sample from).
    fn pool_size(&self) -> usize;

    /// Reference to the underlying pool so the loop can build snapshots.
    fn pool_directions(&self) -> &[Direction];
}

// ── QualityGuide ──────────────────────────────────────────────────────────

/// Frozen guide that scores candidate directions for relevance × elegance.
///
/// Modelless counterpart of the SGS paper's `ρ`. Score ∈ `[0, 1]`.
/// Combines a relevance projection (`dot-product + sigmoid` onto the target)
/// with an elegance penalty (`sigmoid(−α · complexity)`).
///
/// # Contract
///
/// - Returns a finite `f32` in `[0, 1]`.
/// - Never uses softmax — sigmoid only.
pub trait QualityGuide {
    /// Score ∈ `[0, 1]`. Higher = more relevant × more elegant.
    fn score(&self, target: &Target, candidate: &Direction) -> f32;
}

// ── Solver ────────────────────────────────────────────────────────────────

/// Frozen inference brain that attempts a candidate against the target.
///
/// Returns a raw solve-rate scalar in `[0, 1]` that may cross the sync
/// boundary. The latent `Direction` it was produced from stays local.
///
/// Modelless counterpart of the SGS paper's `π_θ` (which is trained).
/// No weight mutation — only reads.
pub trait Solver {
    /// Attempt `candidate` against `target`.
    ///
    /// Returns the empirical solve-rate in `[0, 1]`. A return of `1.0`
    /// means the solver trivially succeeded (filtered out by the
    /// difficulty router), `0.0` means it failed outright (also filtered
    /// out — intermediate-difficulty band only).
    fn attempt(&mut self, target: &Target, candidate: &Candidate) -> f32;
}

// ── HintDeltaBandit ───────────────────────────────────────────────────────

/// Priority-table update driven by Hint-δ absorb-compress (Plan 049).
///
/// Alias of the contract satisfied by `DeltaGatedAbsorbCompress` /
/// `DeltaBanditPruner` from `crate::pruners::g_zero`. Promoted to a CGSP-local
/// trait so the loop stays generic over the bandit backend.
///
/// # Contract
///
/// - `absorb(arm, reward)` — incremental update, no allocation.
/// - `priority(arm)`       — returns the current priority ∈ `[0, 1]`.
/// - `priorities_mut()`    — exposes the priority table for collapse checks
///                           and snapshotting.
pub trait HintDeltaBandit {
    /// Absorb a synthetic reward `(1 − solve_rate) · guide_score` for `arm`.
    fn absorb(&mut self, arm: usize, reward: f32);

    /// Current priority weight for `arm`. Higher = more sampling mass.
    fn priority(&self, arm: usize) -> Priority;

    /// Borrow the full priority table (read-only).
    fn priorities(&self) -> &[Priority];

    /// Mutably borrow the priority table (for entropy checks, renormalization,
    /// exploration injection, and snapshotting).
    fn priorities_mut(&mut self) -> &mut [Priority];

    /// Number of arms in the bandit.
    fn num_arms(&self) -> usize {
        self.priorities().len()
    }

    /// Push a new arm with the given priority. Returns the new arm index.
    ///
    /// Default implementation is a **no-op** (returns current arm count) —
    /// only growing bandit backends override this. Non-growing bandits
    /// (fixed-size) inherit the default and silently ignore the push.
    ///
    /// Used by `DualPoolBandit` Phase 4 consolidation (DecentMem Eq. 8):
    /// rewarded X-pool arms are promoted into the E-pool as new arms.
    fn push_arm(&mut self, _priority: Priority) -> usize {
        self.num_arms()
    }

    /// Whether this bandit supports dynamic arm growth.
    ///
    /// Default: `false`. Growing backends (e.g. `Vec`-backed bandits) override
    /// to `true`. `DualPoolBandit::consolidate()` checks this to decide whether
    /// to do priority-blend (Phase 1) or arm growth (Phase 4).
    fn is_growing(&self) -> bool {
        false
    }
}

// ── BatchQualityGate ──────────────────────────────────────────────────────

/// Degenerate-batch detector — `data_gate` analogue (Plan 111).
///
/// Called before the bandit update inside `cycle()`. If the candidate batch
/// is structurally degenerate (all-same direction, all-rejected by the
/// difficulty filter, etc.), the loop skips the bandit update and forces
/// exploration injection via [`CollapseSignal::inject_exploration`].
///
/// Game-agnostic: implementations decide what "degenerate" means in their
/// domain (e.g. all candidates colinear with each other, all candidates
/// pointing at the same zone, etc.).
pub trait BatchQualityGate {
    /// Returns `true` if the current candidate batch is degenerate and the
    /// bandit update should be skipped.
    ///
    /// Inputs are the per-cycle buffers already populated by `cycle()`:
    /// - `candidates` — conjecturer output
    /// - `admitted`   — difficulty-filter decisions
    /// - `guide_scores` — guide quality scores
    fn is_degenerate(
        &self,
        candidates: &[Candidate],
        admitted: &[bool],
        guide_scores: &[f32],
    ) -> bool;
}

// ── CollapseSignal ────────────────────────────────────────────────────────

/// Entropy-collapse detector + exploration injector (Plan 212 analogue).
///
/// The loop calls [`check_collapse`](Self::check_collapse) at the end of each
/// cycle. If the priority table entropy has dropped below the configured
/// threshold, the loop calls [`inject_exploration`](Self::inject_exploration)
/// to raise the conjecturer's sampling temperature for the next cycle.
///
/// This trait fuses two SGS anti-collapse mechanisms (Solver entropy
/// preservation + Conjecturer drift detection) into a single modelless
/// contract.
pub trait CollapseSignal {
    /// Inspect the current priority table and decide whether collapse
    /// has occurred. Mutates internal detector state (e.g. EMA threshold).
    ///
    /// Returns `true` if exploration should be injected next cycle.
    fn check_collapse(&mut self, priorities: &[Priority], cycle_stats: &CycleResult) -> bool;

    /// Raise the sampling temperature by `magnitude ∈ [0, 1]`.
    ///
    /// `magnitude = 0.0` is a no-op; `magnitude = 1.0` flattens the priority
    /// table to uniform. Idempotent within a cycle.
    fn inject_exploration(&mut self, priorities: &mut [Priority], magnitude: f32);
}

// ── DifficultyFilter ──────────────────────────────────────────────────────

/// Intermediate-difficulty admission router — `breakeven_complexity` analogue
/// (Plan 250). Drops candidates whose estimated solve-rate is `0` (too easy /
/// already known) or `1` (too hard / Solver has no chance).
///
/// The loop fills `admitted[i] = true` for candidates that survive.
pub trait DifficultyFilter {
    /// Decide whether `candidate` with `guide_score` and `estimated_solve_rate`
    /// should be admitted to the solver attempt stage.
    fn admit(&self, guide_score: f32, estimated_solve_rate: f32) -> bool;
}

// ── Default no-op implementations ─────────────────────────────────────────

/// A `BatchQualityGate` that never fires — useful for the G3 (g_zero-only)
/// baseline benchmark.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOpBatchGate;

impl BatchQualityGate for NoOpBatchGate {
    #[inline]
    fn is_degenerate(
        &self,
        _candidates: &[Candidate],
        _admitted: &[bool],
        _guide_scores: &[f32],
    ) -> bool {
        false
    }
}

/// A `DifficultyFilter` that admits everything (no breakeven filter).
/// Used by the G1 (g_zero-only) baseline.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoOpDifficultyFilter;

impl DifficultyFilter for NoOpDifficultyFilter {
    #[inline]
    fn admit(&self, _guide_score: f32, _estimated_solve_rate: f32) -> bool {
        true
    }
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    /// Sanity: a stub bandit satisfies the trait.
    struct StubBandit {
        prios: Vec<f32>,
    }
    impl HintDeltaBandit for StubBandit {
        fn absorb(&mut self, arm: usize, reward: f32) {
            if let Some(p) = self.prios.get_mut(arm) {
                *p = (*p + reward).min(1.0);
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

    #[test]
    fn stub_bandit_satisfies_trait() {
        let mut b = StubBandit { prios: vec![0.1; 4] };
        b.absorb(0, 0.5);
        assert!((b.priority(0) - 0.6).abs() < 1e-6);
        assert_eq!(b.num_arms(), 4);
        b.priorities_mut()[1] = 0.9;
        assert!((b.priority(1) - 0.9).abs() < 1e-6);
        // Default push_arm is a no-op for non-growing bandits.
        assert!(!b.is_growing());
        let idx = b.push_arm(0.5);
        assert_eq!(idx, 4, "no-op push_arm returns current count");
        assert_eq!(b.num_arms(), 4, "non-growing bandit size unchanged");
    }
}
