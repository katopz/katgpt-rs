//! `CgspLoop` — the zero-allocation main loop (Plan 274 §2.3, Phase 1 T1.4).
//!
//! Fuses the SGS triad (Solver / Conjecturer / Guide) with existing katgpt-rs
//! primitives. One `cycle()` call:
//!
//! 1. Conjecturer samples k candidates into `scratch.candidates`.
//! 2. Guide scores each into `scratch.guide_scores`.
//! 3. Difficulty filter marks admit/reject in `scratch.admitted`.
//! 4. Solver attempts admitted candidates, writes solve rates into
//!    `scratch.solve_rates`.
//! 5. Computes `r_synth[i] = (1 − solve_rates[i]) · guide_scores[i]` for
//!    admitted candidates.
//! 6. `BatchQualityGate` checks for degeneracy; if degenerate, skips the
//!    bandit update and forces exploration injection.
//! 7. Bandit absorbs `r_synth` for each admitted candidate.
//! 8. `CollapseSignal::check_collapse` decides whether to inject exploration.
//!
//! All hot-path writes go into the caller-provided `ScratchBuffers` — zero
//! allocation in steady state.

use crate::cgsp::traits::{
    BatchQualityGate, CollapseSignal, CuriosityConjecturer, DifficultyFilter, HintDeltaBandit,
    NoOpBatchGate, NoOpDifficultyFilter, QualityGuide, Solver,
};
use crate::cgsp::types::{
    entropy_nats, CuriosityPrioritySnapshot, CycleResult, CycleStats, Direction, Priority,
    ScratchBuffers, Target,
};

// ── CgspConfig ────────────────────────────────────────────────────────────

/// Tunable parameters for the loop. All defaults chosen to give a reasonable
/// plasma-tier budget (≤ 1µs per cycle for k ≤ 4) on Apple Silicon NEON.
#[derive(Clone, Debug)]
pub struct CgspConfig {
    /// Number of candidates to sample per cycle (k).
    pub k: usize,
    /// τ_low: priority-table entropy threshold below which collapse is triggered.
    pub tau_low: f32,
    /// Exploration injection magnitude when collapse triggers (in `[0, 1]`).
    pub exploration_magnitude: f32,
    /// Estimated solve-rate floor — drop candidates below this as "trivially
    /// solved" (already-known) per the breakeven router.
    pub solve_rate_floor: f32,
    /// Estimated solve-rate ceiling — drop candidates above this as "too hard".
    pub solve_rate_ceiling: f32,
}

impl Default for CgspConfig {
    fn default() -> Self {
        Self {
            k: 4,
            tau_low: 0.30,
            exploration_magnitude: 0.35,
            solve_rate_floor: 0.05,
            solve_rate_ceiling: 0.95,
        }
    }
}

// ── CgspLoop ──────────────────────────────────────────────────────────────

/// The full CGSP triad loop. Generic over its five pluggable components.
///
/// Lifecycle:
/// - Construct once via [`CgspLoop::new`].
/// - Reuse across cycles: pass a pre-allocated [`ScratchBuffers`] to `cycle()`.
/// - Snapshot / restore for the freeze/thaw cycle (Phase 2).
pub struct CgspLoop<C, G, S, B, Col = EntropyCollapse, Df = NoOpDifficultyFilter, Qg = NoOpBatchGate>
{
    pub(crate) conjecturer: C,
    pub(crate) guide: G,
    pub(crate) solver: S,
    pub(crate) bandit: B,
    pub(crate) collapse: Col,
    pub(crate) difficulty_filter: Df,
    pub(crate) batch_gate: Qg,
    pub(crate) config: CgspConfig,
    /// Internal RNG state for the priority-weighted sampler.
    pub(crate) rng_state: u64,
}

impl<C, G, S, B> CgspLoop<C, G, S, B>
where
    C: CuriosityConjecturer,
    G: QualityGuide,
    S: Solver,
    B: HintDeltaBandit,
{
    /// Build a new loop with default collapse detector, difficulty filter,
    /// and batch gate.
    pub fn new(conjecturer: C, guide: G, solver: S, bandit: B, config: CgspConfig) -> Self {
        Self {
            conjecturer,
            guide,
            solver,
            bandit,
            collapse: EntropyCollapse::default(),
            difficulty_filter: NoOpDifficultyFilter,
            batch_gate: NoOpBatchGate,
            config,
            rng_state: 0x9E37_79B9_7F4A_7C15, // splittable64 seed
        }
    }
}

impl<C, G, S, B, Col, Df, Qg> CgspLoop<C, G, S, B, Col, Df, Qg>
where
    C: CuriosityConjecturer,
    G: QualityGuide,
    S: Solver,
    B: HintDeltaBandit,
    Col: CollapseSignal,
    Df: DifficultyFilter,
    Qg: BatchQualityGate,
{
    /// Replace the collapse detector.
    pub fn with_collapse<Col2: CollapseSignal>(self, collapse: Col2) -> CgspLoop<C, G, S, B, Col2, Df, Qg> {
        CgspLoop {
            conjecturer: self.conjecturer,
            guide: self.guide,
            solver: self.solver,
            bandit: self.bandit,
            collapse,
            difficulty_filter: self.difficulty_filter,
            batch_gate: self.batch_gate,
            config: self.config,
            rng_state: self.rng_state,
        }
    }

    /// Replace the difficulty filter.
    pub fn with_difficulty_filter<Df2: DifficultyFilter>(
        self,
        difficulty_filter: Df2,
    ) -> CgspLoop<C, G, S, B, Col, Df2, Qg> {
        CgspLoop {
            conjecturer: self.conjecturer,
            guide: self.guide,
            solver: self.solver,
            bandit: self.bandit,
            collapse: self.collapse,
            difficulty_filter,
            batch_gate: self.batch_gate,
            config: self.config,
            rng_state: self.rng_state,
        }
    }

    /// Replace the batch quality gate.
    pub fn with_batch_gate<Qg2: BatchQualityGate>(
        self,
        batch_gate: Qg2,
    ) -> CgspLoop<C, G, S, B, Col, Df, Qg2> {
        CgspLoop {
            conjecturer: self.conjecturer,
            guide: self.guide,
            solver: self.solver,
            bandit: self.bandit,
            collapse: self.collapse,
            difficulty_filter: self.difficulty_filter,
            batch_gate,
            config: self.config,
            rng_state: self.rng_state,
        }
    }

    /// Borrow the conjecturer.
    pub fn conjecturer(&self) -> &C {
        &self.conjecturer
    }

    /// Borrow the guide.
    pub fn guide(&self) -> &G {
        &self.guide
    }

    /// Borrow the solver.
    pub fn solver(&self) -> &S {
        &self.solver
    }

    /// Borrow the bandit.
    pub fn bandit(&self) -> &B {
        &self.bandit
    }

    /// Borrow the bandit mutably.
    pub fn bandit_mut(&mut self) -> &mut B {
        &mut self.bandit
    }

    /// Borrow the current config.
    pub fn config(&self) -> &CgspConfig {
        &self.config
    }

    /// Mutably borrow the current config.
    pub fn config_mut(&mut self) -> &mut CgspConfig {
        &mut self.config
    }

    /// Run one CGSP cycle. Writes into `scratch`, returns raw observables.
    ///
    /// Zero-allocation in steady state — every write goes into `scratch`.
    ///
    /// See issue 021 for the two allocation sites that historically broke
    /// this invariant (now fixed by Option A + Option B):
    /// - Site 1 (clear+resize on `scratch.candidates`) → replaced by
    ///   [`ScratchBuffers::ensure_len`], which materialises slots once.
    /// - Site 2 (`candidates[i].clone()` to dodge a borrow conflict) →
    ///   removed by changing [`Solver::attempt`] to take `&Direction`.
    pub fn cycle(&mut self, target: &Target, scratch: &mut ScratchBuffers) -> CycleResult {
        scratch.reset();
        let k = self.config.k;
        // Materialise exactly k reusable slots once, then overwrite in place
        // every cycle. Alloc-free in steady state (issue 021 Site 1).
        scratch.ensure_len(k, target.dim());

        // Split the mutable borrow so we can hand disjoint fields to the
        // conjecturer (`candidates` + `cdf_scratch`) while keeping the rest
        // for ourselves. Rust's borrow checker is fine with this when the
        // field accesses happen at the same expression level.
        let ScratchBuffers {
            candidates,
            guide_scores,
            admitted,
            solve_rates,
            r_synth,
            cdf_scratch,
        } = scratch;

        // ── Step 1: Conjecturer samples k candidates ─────────────────────
        self.conjecturer.sample_candidates(
            target,
            self.bandit.priorities(),
            candidates,
            cdf_scratch,
        );

        // ── Step 2: Guide scores each candidate ──────────────────────────
        for (i, c) in candidates.iter().enumerate() {
            guide_scores[i] = self.guide.score(target, &c.direction);
        }

        // ── Step 3: Difficulty filter (breakeven-style admission) ────────
        // First, get an *estimate* of solve rate from the solver's perspective
        // by probing once on a tiny budget. To stay zero-allocation we instead
        // use the guide score as the proxy: very high guide ≈ aligned with
        // target → easy; very low guide ≈ orthogonal → hard. This is a coarse
        // but allocation-free stand-in for the true breakeven_complexity
        // router; production callers should swap in their own DifficultyFilter.
        for i in 0..k {
            let gs = guide_scores[i];
            let est = gs; // proxy; real impl replaces via with_difficulty_filter.
            admitted[i] = self.difficulty_filter.admit(gs, est);
        }

        // ── Step 4: Solver attempts admitted candidates ──────────────────
        let mut admitted_count = 0u32;
        let mut solved_count = 0u32;
        let mut guide_sum = 0.0f32;
        let mut r_synth_sum = 0.0f32;
        for i in 0..k {
            if !admitted[i] {
                solve_rates[i] = 0.0;
                continue;
            }
            admitted_count += 1;
            // Borrow the direction + pool_index directly from the scratch
            // slot — no `Candidate` clone (issue 021, Site 2). The solver
            // trait takes `&Direction` + `pool_index` precisely so this call
            // site can pass disjoint fields without aliasing `solver`.
            let pool_index = candidates[i].pool_index;
            let rate = self
                .solver
                .attempt(target, &candidates[i].direction, pool_index);
            let rate_clamped = rate.clamp(0.0, 1.0);
            solve_rates[i] = rate_clamped;
            if rate_clamped > 0.5 {
                solved_count += 1;
            }
            guide_sum += guide_scores[i];
            // ── Step 5: Synthetic reward (1 − solve_rate) · guide_score ──
            let rs = (1.0 - rate_clamped) * guide_scores[i];
            r_synth[i] = rs;
            r_synth_sum += rs;
        }

        // ── Step 6: Batch quality gate — skip update if degenerate ───────
        let degenerate = self
            .batch_gate
            .is_degenerate(candidates, admitted, guide_scores);

        // ── Step 7: Bandit absorb-compress (skip if degenerate) ─────────
        if !degenerate {
            for i in 0..k {
                if !admitted[i] {
                    continue;
                }
                let arm = candidates[i].pool_index;
                if arm != usize::MAX {
                    self.bandit.absorb(arm, r_synth[i]);
                }
            }
        }

        // Clamp + renormalize priorities to keep them in a stable range.
        renormalize_priorities(self.bandit.priorities_mut());

        let priority_entropy = entropy_nats(self.bandit.priorities());

        let stats = CycleStats {
            candidates_sampled: k as u32,
            candidates_admitted: admitted_count,
            candidates_solved: solved_count,
            mean_guide_score: if admitted_count > 0 {
                guide_sum / admitted_count as f32
            } else {
                0.0
            },
            mean_r_synth: if admitted_count > 0 {
                r_synth_sum / admitted_count as f32
            } else {
                0.0
            },
            priority_entropy,
        };

        let mut result = CycleResult {
            collapse_triggered: false,
            batch_degenerate: degenerate,
            stats,
        };

        // ── Step 8: Collapse check + exploration injection ───────────────
        let collapsed = self
            .collapse
            .check_collapse(self.bandit.priorities(), &result);
        if collapsed || degenerate {
            let magnitude = if degenerate {
                self.config.exploration_magnitude.max(0.5)
            } else {
                self.config.exploration_magnitude
            };
            self.collapse
                .inject_exploration(self.bandit.priorities_mut(), magnitude);
            renormalize_priorities(self.bandit.priorities_mut());
            result.collapse_triggered = true;
        }

        result
    }

    /// Manually inject exploration (collapse_aware_thinking entry point, T1.5).
    pub fn inject_exploration(&mut self, magnitude: f32) {
        self.collapse
            .inject_exploration(self.bandit.priorities_mut(), magnitude.clamp(0.0, 1.0));
        renormalize_priorities(self.bandit.priorities_mut());
    }

    /// Capture an atomic snapshot of the priority table + direction pool (T2.2).
    ///
    /// The snapshot format pairs one direction with one priority (paired
    /// encoding, BLAKE3-committed). For the common single-pool case the
    /// conjecturer's pool size equals the bandit's arm count, so this is a
    /// 1:1 copy. For the dual-pool growth case (`DualPoolBandit` with
    /// `growth_enabled`, Plan 282/312) the E-pool bandit can grow beyond the
    /// conjecturer's frozen basis — those extra arms have no associated
    /// direction (they are phantom priority slots that are never sampled,
    /// since `PoolConjecturer::sample_candidates` clamps the sampled index
    /// to `pool.len() - 1`). We pad the directions vec with zero vectors so
    /// the paired format stays valid; `restore()` only copies priorities, so
    /// the padding has no downstream effect.
    pub fn snapshot(&self) -> CuriosityPrioritySnapshot {
        let pool_dirs = self.conjecturer.pool_directions();
        let priorities: Vec<f32> = self.bandit.priorities().to_vec();
        let mut directions: Vec<Direction> = pool_dirs.iter().cloned().collect();
        // Dual-pool growth: bandit arms may exceed the conjecturer's frozen
        // basis. Pad with zero vectors so directions.len() == priorities.len()
        // (required by the paired snapshot format). Zero = "no associated
        // direction" — honest for phantom arms.
        if directions.len() < priorities.len() {
            let dim = directions.first().map(|d| d.dim()).unwrap_or(0);
            let pad_count = priorities.len() - directions.len();
            directions.reserve(pad_count);
            for _ in 0..pad_count {
                directions.push(Direction::zeros(dim));
            }
        }
        CuriosityPrioritySnapshot::new(directions, priorities)
    }

    /// Restore internal state from a snapshot atomically (T2.2).
    ///
    /// Replaces the bandit priority table in-place. The conjecturer pool is
    /// not mutated (it is frozen by design — only priorities change).
    /// Returns `Err` if the snapshot's pool size doesn't match the bandit.
    pub fn restore(&mut self, snapshot: &CuriosityPrioritySnapshot) -> Result<(), String> {
        let bandit_arms = self.bandit.num_arms();
        if snapshot.pool_size() != bandit_arms {
            return Err(format!(
                "cgsp restore: snapshot pool {} != bandit arms {}",
                snapshot.pool_size(),
                bandit_arms
            ));
        }
        let dst = self.bandit.priorities_mut();
        if dst.len() != snapshot.priorities.len() {
            return Err(format!(
                "cgsp restore: priority length {} != {}",
                dst.len(),
                snapshot.priorities.len()
            ));
        }
        dst.copy_from_slice(&snapshot.priorities);
        Ok(())
    }

    /// Run `n` cycles, snapshotting into `sink` every `every_n` cycles (T2.3).
    ///
    /// The snapshot at the end of the run is always emitted regardless of
    /// `every_n`. Used by the riir-ai runtime to persist personality
    /// checkpoints.
    pub fn run_with_snapshotting<F>(
        &mut self,
        target: &Target,
        scratch: &mut ScratchBuffers,
        n: usize,
        every_n: usize,
        mut sink: F,
    ) -> Vec<CuriosityPrioritySnapshot>
    where
        F: FnMut(&CuriosityPrioritySnapshot),
    {
        let mut emitted = Vec::new();
        let every_n = every_n.max(1);
        for i in 0..n {
            let _ = self.cycle(target, scratch);
            let is_checkpoint = (i + 1) % every_n == 0;
            let is_final = i + 1 == n;
            if is_checkpoint || is_final {
                let snap = self.snapshot();
                sink(&snap);
                emitted.push(snap);
            }
        }
        emitted
    }
}

// ── EntropyCollapse: default CollapseSignal impl ──────────────────────────

/// Default collapse detector — uses Shannon entropy of the priority table.
///
/// When entropy drops below `tau_low`, exploration is injected by mixing the
/// current priorities with the uniform distribution.
#[derive(Clone, Debug)]
pub struct EntropyCollapse {
    pub tau_low: f32,
    pub last_entropy: f32,
}

impl Default for EntropyCollapse {
    fn default() -> Self {
        Self {
            tau_low: 0.30,
            last_entropy: f32::MAX,
        }
    }
}

impl EntropyCollapse {
    /// Build with a custom τ_low threshold.
    pub fn new(tau_low: f32) -> Self {
        Self {
            tau_low,
            last_entropy: f32::MAX,
        }
    }
}

impl CollapseSignal for EntropyCollapse {
    fn check_collapse(&mut self, _priorities: &[Priority], cycle_stats: &CycleResult) -> bool {
        let h = cycle_stats.stats.priority_entropy;
        self.last_entropy = h;
        h < self.tau_low
    }

    fn inject_exploration(&mut self, priorities: &mut [Priority], magnitude: f32) {
        if priorities.is_empty() {
            return;
        }
        let m = magnitude.clamp(0.0, 1.0);
        let uniform = 1.0 / priorities.len() as f32;
        // Mix: p' = (1 − m) · p + m · uniform
        for p in priorities.iter_mut() {
            *p = (1.0 - m) * *p + m * uniform;
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Clamp priorities to `[0, ∞)` and rescale so the max is `1.0` if it exceeds.
///
/// Keeps the priority table numerically stable across many cycles.
pub fn renormalize_priorities(p: &mut [Priority]) {
    if p.is_empty() {
        return;
    }
    let mut max = 0.0f32;
    for &v in p.iter() {
        // Sanitize: NaN / negative -> 0.
        let v = if v.is_finite() && v >= 0.0 { v } else { 0.0 };
        if v > max {
            max = v;
        }
    }
    if max == 0.0 || !max.is_finite() {
        // All-zero table — restore uniform.
        let u = 1.0 / p.len() as f32;
        for v in p.iter_mut() {
            *v = u;
        }
        return;
    }
    // Normalize to maximum = 1.0 (preserves relative ordering; sigmoid-friendly).
    for v in p.iter_mut() {
        let sanitized = if v.is_finite() && *v >= 0.0 { *v } else { 0.0 };
        *v = sanitized / max;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgsp::conjecturer::PoolConjecturer;
    use crate::cgsp::filters::BreakevenDifficultyFilter;
    use crate::cgsp::guide::{ComplexityWeights, HlaProjectionGuide};
    use crate::cgsp::traits::{HintDeltaBandit, Solver};

    /// Simple bandit backed by a `Vec<f32>`. For unit tests only.
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
    /// target — admits intermediate-difficulty candidates.
    pub(crate) struct DotSolver {
        pub sharpness: f32,
    }
    impl Solver for DotSolver {
        fn attempt(
            &mut self,
            target: &Target,
            candidate_direction: &Direction,
            _pool_index: usize,
        ) -> f32 {
            let d = candidate_direction.dot(&target.direction);
            // Map dot in [-1, 1] to solve-rate in [0, 1] via sigmoid.
            crate::cgsp::types::sigmoid(self.sharpness * d)
        }
    }

    fn make_orthonormal_pool(dim: usize, n: usize) -> Vec<Direction> {
        // Build n strictly orthonormal directions (canonical basis vectors).
        // No cross-term so dot products are exactly 0 or 1, giving the guide
        // a clean signal to favor the target-aligned arm.
        assert!(n <= dim, "pool size {n} must be ≤ dim {dim} for orthonormal pool");
        (0..n)
            .map(|i| {
                let mut coords = vec![0.0f32; dim];
                coords[i] = 1.0;
                Direction { coords }
            })
            .collect()
    }

    #[test]
    fn cycle_produces_finite_priorities() {
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 12345);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        for _ in 0..50 {
            let r = lp.cycle(&target, &mut scratch);
            assert!(r.stats.priority_entropy.is_finite(), "entropy NaN");
            for &p in lp.bandit().priorities() {
                assert!(p.is_finite(), "priority NaN");
                assert!(p >= 0.0, "priority negative");
            }
        }
    }

    #[test]
    fn cycle_priority_monotone_in_reward() {
        // Higher synthetic reward (lower solve_rate, higher guide_score)
        // should drive the priority of the target-aligned direction up.
        //
        // Tuning: with sharpness=0.5, target solve_rate ≈ sigmoid(0.5*1) ≈ 0.62
        // which sits in the intermediate-difficulty band (so r_synth > 0).
        // Non-target arms get lower guide_scores so they accrue less reward.
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 42);
        let guide = HlaProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 0.5 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default())
            .with_difficulty_filter(BreakevenDifficultyFilter::new(0.0, 1.0));
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        let before = lp.bandit().priority(0);
        for _ in 0..100 {
            let _ = lp.cycle(&target, &mut scratch);
        }
        let after = lp.bandit().priority(0);
        assert!(
            after >= before,
            "target-aligned arm should grow: before={before}, after={after}"
        );
    }

    #[test]
    fn snapshot_restore_roundtrip() {
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 7);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
        let target = Target::new(pool[3].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        // Run a few cycles to drift the priorities.
        for _ in 0..10 {
            let _ = lp.cycle(&target, &mut scratch);
        }
        let snap = lp.snapshot();
        let prios_before = lp.bandit().priorities().to_vec();

        // Drift more.
        for _ in 0..10 {
            let _ = lp.cycle(&target, &mut scratch);
        }
        // Restore.
        lp.restore(&snap).expect("restore");
        let prios_after = lp.bandit().priorities().to_vec();
        assert_eq!(prios_before, prios_after, "restore must be exact");
    }

    #[test]
    fn snapshot_pads_directions_when_bandit_grows_beyond_pool() {
        // Regression test for the dual-pool growth case (Plan 282/312):
        // when the bandit has more arms than the conjecturer's frozen
        // direction pool, `snapshot()` must pad directions with zero vectors
        // so the paired snapshot format stays valid (directions.len() ==
        // priorities.len()). Previously this tripped a `debug_assert_eq!` in
        // `CuriosityPrioritySnapshot::new` (see `g5_epool_persistence` in
        // riir-ai's `dual_pool_bridge`, documented as pre-existing failure
        // across Plans 312/341/008).
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 7);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        // Build a bandit with MORE arms than the 8-direction pool, simulating
        // the post-consolidation E-pool state in dual-pool growth mode.
        let bandit = VecBandit {
            prios: vec![0.5; 16],
        };
        let lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());

        // Snapshot must not panic and must produce a paired-format snapshot.
        let snap = lp.snapshot();
        assert_eq!(
            snap.priorities.len(),
            16,
            "priorities should reflect bandit arm count"
        );
        assert_eq!(
            snap.directions.len(),
            16,
            "directions must be padded to match priorities length"
        );
        // The first 8 directions are the real pool; the last 8 are zero pads.
        for (i, d) in snap.directions.iter().take(8).enumerate() {
            assert_eq!(
                d.dim(),
                pool[i].dim(),
                "real direction {i} dim mismatch"
            );
            assert_ne!(
                d.norm_sq(),
                0.0,
                "real direction {i} should be nonzero (orthonormal pool)"
            );
        }
        for (i, d) in snap.directions.iter().enumerate().skip(8) {
            assert_eq!(d.dim(), 8, "padded direction {i} dim should match pool dim");
            assert_eq!(
                d.norm_sq(),
                0.0,
                "padded direction {i} should be a zero vector"
            );
        }

        // Roundtrip: restore into a fresh loop with the same bandit size.
        // `restore()` only copies priorities; the directions padding has no
        // downstream effect.
        let conj2 = PoolConjecturer::new(pool.clone(), 99);
        let guide2 = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver2 = DotSolver { sharpness: 1.0 };
        let bandit2 = VecBandit {
            prios: vec![0.0; 16],
        };
        let mut lp2 = CgspLoop::new(conj2, guide2, solver2, bandit2, CgspConfig::default());
        lp2.restore(&snap).expect("restore should succeed with matching arm count");
        let prios_after = lp2.bandit().priorities().to_vec();
        assert_eq!(prios_after, snap.priorities, "priorities must roundtrip exactly");
    }

    #[test]
    fn run_with_snapshotting_emits_at_correct_intervals() {
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 99);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        let mut sink_calls = 0usize;
        let emitted = lp.run_with_snapshotting(&target, &mut scratch, 30, 10, |_| {
            sink_calls += 1;
        });
        // 30 / 10 = 3 checkpoints + final = 3 (final is also a checkpoint).
        assert_eq!(emitted.len(), 3);
        assert_eq!(sink_calls, 3);
    }

    #[test]
    fn inject_exploration_raises_entropy() {
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 5);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());

        // Force one-hot priorities to simulate collapse.
        for (i, p) in lp.bandit_mut().priorities_mut().iter_mut().enumerate() {
            *p = if i == 0 { 1.0 } else { 0.0 };
        }
        let h_before = entropy_nats(lp.bandit().priorities());
        lp.inject_exploration(0.5);
        let h_after = entropy_nats(lp.bandit().priorities());
        assert!(
            h_after > h_before,
            "exploration must raise entropy: {h_before} -> {h_after}"
        );
    }
}
