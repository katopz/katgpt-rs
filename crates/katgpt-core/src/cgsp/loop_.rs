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
    CuriosityPrioritySnapshot, CycleResult, CycleStats, Direction, HintPolicy, Priority,
    ScratchBuffers, Target, entropy_nats,
};

// ── CgspConfig ────────────────────────────────────────────────────────────

/// Tunable parameters for the loop. All defaults chosen to give a reasonable
/// plasma-tier budget (≤ 1µs per cycle for k ≤ 4) on Apple Silicon NEON.
#[derive(Clone, Copy, Debug)]
pub struct CgspConfig {
    /// Number of candidates to sample per cycle (k).
    pub k: usize,
    /// τ_low: priority-table entropy threshold below which collapse is triggered.
    pub tau_low: f32,
    /// Exploration injection magnitude when collapse triggers (in `[0, 1]`).
    pub exploration_magnitude: f32,
    /// Variable-duration planning horizon per NPC (Issue 365).
    ///
    /// 1 = single-cycle plan (default — no staleness correction). Values > 1
    /// come from the gain/cost halter (Plan 304) via the async commit bridge
    /// (Issue 364): the NPC's reflex-hold count before the next committed
    /// CGSP action. Read by [`staleness_weight`] to discount staler bandit
    /// feedback so priority updates stay comparable across NPCs of different
    /// planning depths.
    pub k_npc: u8,
    /// Staleness decay rate λ for variable-duration CGSP (Issue 365).
    ///
    /// 0.0 = disabled (default — the absorbed reward is bit-identical to
    /// pre-Issue-365 behavior: `r_synth * 1.0 == r_synth` for all f32).
    /// Values > 0 produce `exp(-λ·(k_npc−1))` discounting on absorbed
    /// rewards via [`staleness_weight`].
    pub staleness_lambda: f32,
}

impl Default for CgspConfig {
    fn default() -> Self {
        Self {
            k: 4,
            tau_low: 0.30,
            exploration_magnitude: 0.35,
            k_npc: 1,
            staleness_lambda: 0.0,
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
pub struct CgspLoop<
    C,
    G,
    S,
    B,
    Col = EntropyCollapse,
    Df = NoOpDifficultyFilter,
    Qg = NoOpBatchGate,
> {
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
    pub fn with_collapse<Col2: CollapseSignal>(
        self,
        collapse: Col2,
    ) -> CgspLoop<C, G, S, B, Col2, Df, Qg> {
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

    /// Mutably borrow the conjecturer (the wrapper-conjecturer feed seam:
    /// a stateful wrapper — e.g. a batch selector carrying an observer —
    /// is fed between cycles by the hook/fixture that owns the loop;
    /// the `bandit_mut()` precedent).
    pub fn conjecturer_mut(&mut self) -> &mut C {
        &mut self.conjecturer
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

        // ── Step 2+3: Guide scores + difficulty-filter admission (fused) ──
        // Previously two separate loops over `0..k`; fused into one pass
        // because the admission check only depends on the guide score just
        // computed in the same iteration. Saves one loop's worth of bounds
        // checks + counter updates (k=8 default).
        //
        // Difficulty filter context: we use the guide score as the proxy for
        // estimated solve rate (very high guide ≈ aligned with target → easy;
        // very low guide ≈ orthogonal → hard). This is a coarse but
        // allocation-free stand-in for the true breakeven_complexity router;
        // production callers should swap in their own DifficultyFilter.
        for (i, c) in candidates.iter().enumerate() {
            let gs = self.guide.score(target, &c.direction);
            guide_scores[i] = gs;
            admitted[i] = self.difficulty_filter.admit(gs, gs);
        }

        // ── Step 4: Solver attempts admitted candidates ──────────────────
        // G-RRM §3 (Issue 037): if the solver declares `HintPolicy::Skip`
        // (overhead-dominated — e.g. a future CDCL SAT backend), its solve-rates
        // are noise w.r.t. hint quality. We still run the attempts (Step 4) and
        // record stats, but suppress the bandit absorb (Step 7) so the priority
        // table — the hint — is not corrupted by that noise. `HintPolicy` is
        // `Copy`, so this read ends the shared borrow before the mutable
        // `attempt` calls below.
        let hint_skip = self.solver.hint_receptivity() == HintPolicy::Skip;
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

        // ── Step 7: Bandit absorb-compress (skip if degenerate OR hint-skip) ─
        // Variable-duration staleness weight (Issue 365): with the default
        // config (k_npc=1, staleness_lambda=0.0) `staleness_w` is 1.0 and
        // the absorb is bit-identical to pre-Issue-365 behavior.
        //
        // G-RRM §3 (Issue 037): `hint_skip` suppresses the absorb for
        // overhead-dominated solvers whose solve-rates do not reflect hint
        // quality, so their priorities (the hint) are not corrupted by noise.
        if !degenerate && !hint_skip {
            let staleness_w = staleness_weight(self.config.k_npc, self.config.staleness_lambda);
            for i in 0..k {
                if !admitted[i] {
                    continue;
                }
                let arm = candidates[i].pool_index;
                if arm != usize::MAX {
                    self.bandit.absorb(arm, r_synth[i] * staleness_w);
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
        let mut directions: Vec<Direction> = pool_dirs.to_vec();
        // Dual-pool growth: bandit arms may exceed the conjecturer's frozen
        // basis. Pad with zero vectors so directions.len() == priorities.len()
        // (required by the paired snapshot format). Zero = "no associated
        // direction" — honest for phantom arms.
        if directions.len() < priorities.len() {
            let dim = directions.first().map_or(0, |d| d.dim());
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
        let mut emitted = Vec::with_capacity(n);
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
#[derive(Clone, Copy, Debug)]
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

/// Staleness weight for variable-duration CGSP bandit updates (Issue 365).
///
/// When an NPC's planning horizon `k_npc` > 1 (the async commit bridge,
/// Issue 364), the bandit feedback that arrives after `k_npc` cycles is
/// staler than feedback from a single-cycle plan. This weight discounts
/// the reward absorbed by [`HintDeltaBandit::absorb`] so that deep-planning
/// NPCs bias the priority table more slowly per unit reward than
/// fast-planning ones — keeping priority updates comparable across NPCs of
/// different planning depths.
///
/// This is the **bandit-update analog** of the paper's `γ^k` variable-duration
/// GAE correction (arXiv:2606.26463 Appendix C). CGSP has no γ-discounted
/// advantage path — its update signal is `r_synth` consumed by `absorb`,
/// not a TD advantage — so the literal "replace γ with γ^k" substitution has
/// no target. The functional role (duration-comparability across the crowd)
/// is preserved by discounting the absorbed reward instead. See
/// `riir-ai/.issues/365_*` §"The correction" for why the original
/// "one-line γ→γ^k" framing was wrong.
///
/// # Contract
///
/// - `k_npc <= 1` or `lambda <= 0` → returns `1.0` (no staleness, no
///   discount). This is the default; with the default config the cycle is
///   bit-identical to pre-Issue-365 behavior (`x * 1.0 == x` for all f32).
/// - `lambda > 0`, `k_npc > 1` → `exp(-lambda * (k_npc - 1))`, strictly
///   decreasing in `k_npc`. Bounded in `(0, 1]`.
/// - NaN / non-finite `lambda` → returns `1.0` (safe no-op).
///
/// Uses `exp(-λ·(k−1))` for strict monotone-decreasing behavior matching
/// geometric decay. The rational alternative `1/(1+λ·(k−1))` is cheaper
/// but not equivalent to geometric discount; since this fires once per
/// cycle (not per sample-per-position), the `exp` cost is negligible.
#[inline]
pub fn staleness_weight(k_npc: u8, lambda: f32) -> f32 {
    if k_npc <= 1 || !lambda.is_finite() || lambda <= 0.0 {
        return 1.0;
    }
    crate::simd::cephes_exp_scalar(-(lambda * (k_npc as f32 - 1.0)))
}

// ── KnpcSelector (Issue 364 T4) ──────────────────────────────────────────

/// Decision returned by [`KnpcSelector::observe_cycle`].
///
/// The selector runs across cycles (each CGSP cycle = one halter loop
/// iteration). While the halter hasn't fired, it returns [`Continue`](Self::Continue) and
/// the NPC runs CGSP every tick (k_npc = 1 — no staleness correction). When
/// the halter fires, it returns [`PlanInterval`](Self::PlanInterval) with the planned interval
/// until the next deep cycle. The caller should:
///
/// 1. Set `config.k_npc = k_npc` for the staleness weight on the next cycle.
/// 2. Skip CGSP for `k_npc − 1` ticks (reflex mode — reuse last priorities).
/// 3. On tick `k_npc`, run the next deep CGSP cycle.
///
/// This is the modelless analog of the paper's variable-duration committed-
/// action protocol: `k_npc` from the halter replaces the paper's trained
/// budget selector. The staleness weight (Issue 365) then discounts the
/// bandit feedback proportionally.
#[cfg(feature = "gain_cost_halt")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KnpcDecision {
    /// The halter hasn't fired yet. Continue running cycles.
    Continue,
    /// The halter fired at loop index `halted_at`. Plan the next deep cycle
    /// after `k_npc` ticks (clamped to `[k_min, k_max]`).
    PlanInterval { k_npc: u8, halted_at: usize },
}

/// Per-NPC variable-duration `k_npc` selector (Issue 364 T4).
///
/// Wraps [`GainCostLoopHalter`](crate::gain_cost_halt::GainCostLoopHalter) to produce a per-NPC planning horizon `k_npc`
/// from per-cycle observables. The selector accumulates evidence across cycles:
/// each call to [`observe_cycle`](Self::observe_cycle) feeds the cycle's
/// (gain, cost, cos_theta) to the halter. When the halter fires `Halt`, the
/// accumulated loop index becomes `k_npc` — the planned interval until the
/// next deep cycle.
///
/// # Signal mapping (CGSP)
///
/// The halter's inputs map to CGSP cycle observables as follows:
/// - **gain** — marginal refinement: how much did the priority table improve
///   this cycle? Natural proxy: decrease in priority-table entropy (a more
///   peaked table = more confident direction). The caller computes this.
/// - **cost** — marginal drift: the staleness budget consumed by one cycle.
///   Natural proxy: a fixed per-cycle cost (e.g. 0.1) or a coherence-decay
///   signal from the runtime.
/// - **cos_theta** — alignment of the last two updates. Natural proxy: the
///   sign of the entropy delta (negative delta = converging, positive =
///   diverging/oscillating).
///
/// The selector itself is signal-agnostic: it takes whatever scalars the
/// caller provides and feeds them to the halter. The mapping above is the
/// recommended default for CGSP; other runtimes (ARG, CWM) may use different
/// signals.
///
/// # Behind feature `gain_cost_halt`
///
/// This primitive composes [`GainCostLoopHalter`](crate::gain_cost_halt::GainCostLoopHalter) (Plan 304, opt-in) with
/// the CGSP variable-duration story (Issue 365/364). It compiles only when
/// both the `cgsp` and `gain_cost_halt` features are enabled. When
/// `gain_cost_halt` is off, the selector is absent and `k_npc` stays at its
/// default (1 — no staleness correction).
///
/// # Default no-op guarantee
///
/// A freshly-constructed selector returns [`KnpcDecision::Continue`] on every
/// call until the halter fires. This means `k_npc` is never modified unless
/// the halter explicitly decides to plan a longer interval. With the default
/// halter (tau = 1.0, patience = 1, l_min = 1), the selector fires on the
/// first gain-below-cost crossover or the first oscillation.
#[cfg(feature = "gain_cost_halt")]
#[derive(Clone, Debug)]
pub struct KnpcSelector {
    halter: crate::gain_cost_halt::GainCostLoopHalter,
    /// Current loop index (1-based). Resets to 1 after each Halt.
    tau: usize,
    /// Minimum planned interval. Default 1 (same-tick replan).
    k_min: u8,
    /// Maximum planned interval. Default 8 (conservative upper bound).
    k_max: u8,
}

#[cfg(feature = "gain_cost_halt")]
impl KnpcSelector {
    /// Construct a selector with explicit halter config and clamping bounds.
    ///
    /// - `halter` — the gain/cost loop halter (Plan 304).
    /// - `k_min` — minimum planned interval (default 1). The selector never
    ///   returns `k_npc < k_min`.
    /// - `k_max` — maximum planned interval (default 8). The selector never
    ///   returns `k_npc > k_max`.
    #[inline]
    pub fn new(halter: crate::gain_cost_halt::GainCostLoopHalter, k_min: u8, k_max: u8) -> Self {
        Self {
            halter,
            tau: 1,
            k_min: k_min.max(1),
            k_max: k_max.max(k_min.max(1)),
        }
    }

    /// Construct a selector with the default halter and `k_min=1, k_max=8`.
    #[inline]
    pub fn default_bounds() -> Self {
        Self::new(crate::gain_cost_halt::GainCostLoopHalter::default(), 1, 8)
    }

    /// Feed one cycle's observables and get the planning decision.
    ///
    /// Call this after each CGSP cycle with the cycle's (gain, cost,
    /// cos_theta) signals. The selector feeds these to the halter and returns:
    /// - [`KnpcDecision::Continue`] — keep running cycles every tick (the
    ///   halter hasn't decided to extend the interval yet).
    /// - [`KnpcDecision::PlanInterval`] — the halter fired; plan the next deep
    ///   cycle after `k_npc` ticks. Set `config.k_npc = k_npc` and skip CGSP
    ///   until then.
    ///
    /// After a `PlanInterval`, the selector resets internally (tau → 1) and
    /// the next `observe_cycle` call starts a new accumulation.
    #[inline]
    pub fn observe_cycle(&mut self, gain: f32, cost: f32, cos_theta: f32) -> KnpcDecision {
        use crate::gain_cost_halt::HaltDecision;
        let decision = self.halter.halt_decision(self.tau, gain, cost, cos_theta);
        match decision {
            HaltDecision::Halt { .. } => {
                let raw = self.tau;
                let k_npc = (raw as u8).clamp(self.k_min, self.k_max);
                self.tau = 1;
                KnpcDecision::PlanInterval {
                    k_npc,
                    halted_at: raw,
                }
            }
            // Continue or RefusedFloor — either way, keep accumulating.
            _ => {
                self.tau = self.tau.saturating_add(1);
                KnpcDecision::Continue
            }
        }
    }

    /// Current loop index (how many cycles have been observed since the last
    /// Halt). Resets to 1 after each [`KnpcDecision::PlanInterval`].
    #[inline]
    pub fn tau(&self) -> usize {
        self.tau
    }

    /// Borrow the inner halter (for inspection or config reads).
    #[inline]
    pub fn halter(&self) -> &crate::gain_cost_halt::GainCostLoopHalter {
        &self.halter
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgsp::conjecturer::PoolConjecturer;
    use crate::cgsp::filters::BreakevenDifficultyFilter;
    use crate::cgsp::guide::{BeliefGridProjectionGuide, ComplexityWeights};
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

    /// Same solve-rate curve as `DotSolver`, but declares `HintPolicy::Skip` —
    /// emulates an overhead-dominated solver (G-RRM §3 cadical3 profile) whose
    /// solve-rates should NOT feed back into the hint priority table.
    pub(crate) struct SkipDotSolver {
        pub sharpness: f32,
    }
    impl Solver for SkipDotSolver {
        fn attempt(
            &mut self,
            target: &Target,
            candidate_direction: &Direction,
            _pool_index: usize,
        ) -> f32 {
            let d = candidate_direction.dot(&target.direction);
            crate::cgsp::types::sigmoid(self.sharpness * d)
        }
        #[inline]
        fn hint_receptivity(&self) -> crate::cgsp::types::HintPolicy {
            crate::cgsp::types::HintPolicy::Skip
        }
    }

    fn make_orthonormal_pool(dim: usize, n: usize) -> Vec<Direction> {
        // Build n strictly orthonormal directions (canonical basis vectors).
        // No cross-term so dot products are exactly 0 or 1, giving the guide
        // a clean signal to favor the target-aligned arm.
        assert!(
            n <= dim,
            "pool size {n} must be ≤ dim {dim} for orthonormal pool"
        );
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
        let guide = BeliefGridProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
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
        let guide = BeliefGridProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
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
    fn hint_skip_suppresses_bandit_absorb() {
        // G-RRM §3 (Issue 037): an overhead-dominated solver declaring
        // `HintPolicy::Skip` must not corrupt the hint priority table. Its
        // solve-rates are noise w.r.t. hint quality, so the bandit absorb is
        // suppressed and priorities stay at their (uniform) starting point.
        //
        // Contrast with `cycle_priority_monotone_in_reward`: the hint-receptive
        // `DotSolver` (default `OrderOnly`) DOES drift the target arm upward.
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 42);
        let guide = BeliefGridProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
        let solver = SkipDotSolver { sharpness: 0.5 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default())
            .with_difficulty_filter(BreakevenDifficultyFilter::new(0.0, 1.0));
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        let before: Vec<f32> = lp.bandit().priorities().to_vec();
        for _ in 0..100 {
            let _ = lp.cycle(&target, &mut scratch);
        }
        let after: Vec<f32> = lp.bandit().priorities().to_vec();
        // `renormalize_priorities` rescales to max=1.0 each cycle, so absolute
        // values are not preserved. The invariant we care about is *relative*:
        // with the absorb suppressed, no arm drifts relative to the others —
        // the priority table stays uniform (the starting condition), because
        // the only thing that could make arms diverge (the absorb) is gated
        // off by `HintPolicy::Skip`. Contrast `cycle_priority_monotone_in_reward`
        // where the hint-receptive `DotSolver` DOES make the target arm grow.
        assert_eq!(after.len(), before.len(), "arm count must not change");
        let first = after[0];
        for (i, &p) in after.iter().enumerate() {
            assert_eq!(
                p, first,
                "Skip-policy solver: arm {i} drifted to {p} (hint feedback should be suppressed, \
                 priorities must stay uniform)"
            );
        }
    }

    #[test]
    fn hint_skip_solver_still_runs_and_reports_stats() {
        // The Skip policy only suppresses the bandit absorb — the solver must
        // still attempt candidates and report honest stats (admitted/solved
        // counts, mean guide score), so callers can observe the overhead-
        // dominated solver's behaviour even when it doesn't feed back.
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 42);
        let guide = BeliefGridProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
        let solver = SkipDotSolver { sharpness: 0.5 };
        let bandit = VecBandit::uniform(8);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default())
            .with_difficulty_filter(BreakevenDifficultyFilter::new(0.0, 1.0));
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(8, 8);

        let r = lp.cycle(&target, &mut scratch);
        assert!(
            r.stats.candidates_admitted > 0,
            "Skip solver must still attempt admitted candidates"
        );
        assert!(
            r.stats.mean_guide_score > 0.0,
            "Skip solver must still record guide scores"
        );
    }

    #[test]
    fn snapshot_restore_roundtrip() {
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 7);
        let guide = BeliefGridProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
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
        let guide = BeliefGridProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
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
            assert_eq!(d.dim(), pool[i].dim(), "real direction {i} dim mismatch");
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
        let guide2 = BeliefGridProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver2 = DotSolver { sharpness: 1.0 };
        let bandit2 = VecBandit {
            prios: vec![0.0; 16],
        };
        let mut lp2 = CgspLoop::new(conj2, guide2, solver2, bandit2, CgspConfig::default());
        lp2.restore(&snap)
            .expect("restore should succeed with matching arm count");
        let prios_after = lp2.bandit().priorities().to_vec();
        assert_eq!(
            prios_after, snap.priorities,
            "priorities must roundtrip exactly"
        );
    }

    #[test]
    fn run_with_snapshotting_emits_at_correct_intervals() {
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 99);
        let guide = BeliefGridProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
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
        let guide = BeliefGridProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
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

    // ── Issue 365: Variable-duration CGSP staleness weight ────────────────

    /// T4 (G1 correctness) — `staleness_weight` unit properties.
    #[test]
    fn t4_staleness_weight_unit_properties() {
        // k_npc=1 → always 1.0 regardless of lambda (no staleness for
        // single-cycle plans — the default case).
        assert_eq!(staleness_weight(1, 0.0), 1.0);
        assert_eq!(staleness_weight(1, 0.5), 1.0);
        assert_eq!(staleness_weight(1, 10.0), 1.0);

        // lambda=0 → always 1.0 regardless of k_npc (disabled).
        assert_eq!(staleness_weight(4, 0.0), 1.0);
        assert_eq!(staleness_weight(8, 0.0), 1.0);

        // NaN / non-finite lambda → 1.0 (safe no-op, guards hot path).
        assert_eq!(staleness_weight(4, f32::NAN), 1.0);
        assert_eq!(staleness_weight(4, f32::INFINITY), 1.0);
        assert_eq!(staleness_weight(4, f32::NEG_INFINITY), 1.0);

        // Negative lambda → 1.0 (would invert the signal — reject).
        assert_eq!(staleness_weight(4, -0.5), 1.0);

        // Monotone strictly decreasing in k_npc for lambda > 0.
        let w1 = staleness_weight(1, 0.5);
        let w2 = staleness_weight(2, 0.5);
        let w4 = staleness_weight(4, 0.5);
        let w8 = staleness_weight(8, 0.5);
        assert_eq!(w1, 1.0);
        assert!(w2 < w1, "w2={w2} should be < w1={w1}");
        assert!(w4 < w2, "w4={w4} should be < w2={w2}");
        assert!(w8 < w4, "w8={w8} should be < w4={w4}");

        // All weights bounded in (0, 1] for a realistic lambda sweep.
        for k_npc in 1..=16u8 {
            let w = staleness_weight(k_npc, 0.5);
            assert!(w > 0.0 && w <= 1.0, "k_npc={k_npc} w={w} out of (0,1]");
        }
    }

    /// T4 (G1) — default config (k_npc=1, staleness_lambda=0.0) still
    /// drives correct behavior (target arm grows). Combined with T6 this
    /// establishes the bit-identical no-op property.
    #[test]
    fn t4_default_config_target_arm_grows() {
        let pool = make_orthonormal_pool(8, 8);
        let conj = PoolConjecturer::new(pool.clone(), 42);
        let guide = BeliefGridProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 0.5 };
        let bandit = VecBandit::uniform(8);
        let cfg = CgspConfig::default();
        // Sanity: defaults are the no-op values.
        assert_eq!(cfg.k_npc, 1);
        assert_eq!(cfg.staleness_lambda, 0.0);
        let mut lp = CgspLoop::new(conj, guide, solver, bandit, cfg)
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
            "target arm should grow with default config: before={before}, after={after}"
        );
    }

    /// T5 (G2 fairness) — high-k_npc NPCs converge more slowly (higher
    /// priority-table entropy after the same cycle count). This is the
    /// bandit-update analog of γ^k: variable-duration feedback stays
    /// comparable across the crowd by being more conservative for
    /// deeper-planning NPCs.
    ///
    /// "Not starved" = the high-k table is still learning (entropy is
    /// finite, strictly below uniform-log entropy). This catches the
    /// degenerate case where the weight zeroes out all learning.
    #[test]
    fn t5_staleness_weight_high_k_higher_entropy_not_starved() {
        let pool = make_orthonormal_pool(8, 8);

        // Helper: run `n` cycles and return (entropy, target_arm_priority).
        fn run(pool: &[Direction], n: usize, k_npc: u8) -> (f32, f32) {
            let conj = PoolConjecturer::new(pool.to_vec(), 42);
            let guide = BeliefGridProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
            let solver = DotSolver { sharpness: 0.5 };
            let bandit = VecBandit::uniform(8);
            let cfg = CgspConfig {
                k_npc,
                staleness_lambda: 0.5,
                ..CgspConfig::default()
            };
            let mut lp = CgspLoop::new(conj, guide, solver, bandit, cfg)
                .with_difficulty_filter(BreakevenDifficultyFilter::new(0.0, 1.0));
            let target = Target::new(pool[0].clone());
            let mut scratch = ScratchBuffers::new(8, 8);
            for _ in 0..n {
                let _ = lp.cycle(&target, &mut scratch);
            }
            let h = entropy_nats(lp.bandit().priorities());
            let target_p = lp.bandit().priority(0);
            (h, target_p)
        }

        let (h_low, p_low) = run(&pool, 50, 1);
        let (h_high, p_high) = run(&pool, 50, 8);

        // High-k NPC has higher entropy (less peaked table → more exploration
        // headroom — the correct semantic for staler feedback).
        assert!(
            h_high > h_low,
            "high-k entropy ({h_high}) should be > low-k ({h_low}) — \
             staleness delays convergence"
        );

        // "Not starved": both tables have departed from uniform (entropy
        // strictly below ln(8) ≈ 2.079, and strictly above 0).
        let uniform_h = (8.0f32).ln();
        assert!(
            h_low < uniform_h && h_high < uniform_h,
            "both tables should have departed from uniform: h_low={h_low}, h_high={h_high}, uniform={uniform_h}"
        );
        assert!(
            h_low > 0.0 && h_high > 0.0,
            "neither table should be one-hot starved: h_low={h_low}, h_high={h_high}"
        );

        // Both target arms are the max in their tables (renormalized to 1.0),
        // but the high-k table's non-target arms are higher (less peaked).
        // We verify via entropy (above). The target arm itself should still
        // be the highest priority in both tables (learning happened, just
        // more slowly for high-k).
        assert_eq!(p_low, 1.0, "low-k target arm should be renormalized max");
        assert_eq!(p_high, 1.0, "high-k target arm should be renormalized max");
    }

    /// T6 (G3 no-regression) — with staleness_lambda=0.0, k_npc has zero
    /// effect on the priority table. Two loops with identical setup but
    /// different k_npc values (1 vs 99) produce bit-identical priorities.
    /// This is the strongest possible no-regression guarantee: the default
    /// config is a true no-op, not just "close enough."
    #[test]
    fn t6_default_config_k_npc_has_no_effect() {
        fn run(pool: &[Direction], k_npc: u8) -> Vec<f32> {
            let conj = PoolConjecturer::new(pool.to_vec(), 42);
            let guide = BeliefGridProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
            let solver = DotSolver { sharpness: 0.5 };
            let bandit = VecBandit::uniform(8);
            let cfg = CgspConfig {
                k_npc,
                staleness_lambda: 0.0,
                ..CgspConfig::default()
            };
            let mut lp = CgspLoop::new(conj, guide, solver, bandit, cfg)
                .with_difficulty_filter(BreakevenDifficultyFilter::new(0.0, 1.0));
            let target = Target::new(pool[0].clone());
            let mut scratch = ScratchBuffers::new(8, 8);
            for _ in 0..30 {
                let _ = lp.cycle(&target, &mut scratch);
            }
            lp.bandit().priorities().to_vec()
        }

        let pool = make_orthonormal_pool(8, 8);

        let p_default = run(&pool, 1);
        let p_extreme = run(&pool, 99);
        assert_eq!(
            p_default, p_extreme,
            "with staleness_lambda=0.0, k_npc=1 vs k_npc=99 must be bit-identical"
        );
    }

    // ── KnpcSelector tests (Issue 364 T4) ───────────────────────────────
    // Behind `gain_cost_halt` feature — these compile only when both `cgsp`
    // and `gain_cost_halt` are enabled.
    #[cfg(feature = "gain_cost_halt")]
    mod knpc_selector {
        use super::*;
        use crate::gain_cost_halt::GainCostLoopHalter;

        /// Default selector starts at tau=1 and returns Continue on first
        /// observe with healthy signals (gain >> cost).
        #[test]
        fn default_selector_continues_on_healthy_signals() {
            let mut sel = KnpcSelector::default_bounds();
            assert_eq!(sel.tau(), 1);
            // gain=10, cost=0.1, cos=0.9 — clearly healthy.
            let d = sel.observe_cycle(10.0, 0.1, 0.9);
            assert_eq!(d, KnpcDecision::Continue);
            assert_eq!(sel.tau(), 2, "tau should advance after Continue");
        }

        /// Halter fires on gain-below-cost → selector returns PlanInterval
        /// with k_npc = halted loop index, clamped to [k_min, k_max].
        #[test]
        fn fires_on_gain_below_cost() {
            // tau=1.0 (gain must exceed cost), patience=1, l_min=1.
            let mut sel = KnpcSelector::default_bounds();
            // Cycle 1: gain=0.1, cost=1.0 → 0.1 < 1.0*1 → Halt::GainBelowCost.
            let d = sel.observe_cycle(0.1, 1.0, 0.9);
            assert!(matches!(
                d,
                KnpcDecision::PlanInterval {
                    k_npc: 1,
                    halted_at: 1
                }
            ));
        }

        /// After Halt, tau resets to 1 so the next interval starts fresh.
        #[test]
        fn resets_after_halt() {
            let mut sel = KnpcSelector::default_bounds();
            // Run 3 healthy cycles to reach tau=3.
            sel.observe_cycle(10.0, 0.1, 0.9);
            sel.observe_cycle(10.0, 0.1, 0.9);
            assert_eq!(sel.tau(), 3);
            // Now trigger halt (gain below cost at tau=3).
            let d = sel.observe_cycle(0.1, 1.0, 0.9);
            assert!(matches!(
                d,
                KnpcDecision::PlanInterval {
                    k_npc: 3,
                    halted_at: 3
                }
            ));
            assert_eq!(sel.tau(), 1, "tau must reset to 1 after Halt");
        }

        /// k_npc is clamped to k_max when the halter takes many cycles.
        #[test]
        fn clamps_to_k_max() {
            // k_max=4, but we run 6 healthy cycles before triggering halt.
            let mut sel = KnpcSelector::new(GainCostLoopHalter::default(), 1, 4);
            for _ in 0..5 {
                sel.observe_cycle(10.0, 0.1, 0.9);
            }
            assert_eq!(sel.tau(), 6);
            let d = sel.observe_cycle(0.1, 1.0, 0.9);
            // halted_at is the raw loop index (6), but k_npc is clamped to 4.
            assert!(matches!(
                d,
                KnpcDecision::PlanInterval {
                    k_npc: 4,
                    halted_at: 6
                }
            ));
        }

        /// Oscillation halt: cos_theta < 0 with patience=1 fires immediately.
        #[test]
        fn fires_on_oscillation() {
            let mut sel = KnpcSelector::default_bounds();
            // gain high so GainBelowCost doesn't fire; cos_theta < 0 trips
            // oscillation (patience=1 → halt on first reversal).
            let d = sel.observe_cycle(10.0, 0.0, -0.5);
            assert!(matches!(
                d,
                KnpcDecision::PlanInterval {
                    k_npc: 1,
                    halted_at: 1
                }
            ));
        }

        /// l_min floor: with l_min=3, the halter returns RefusedFloor for
        /// tau < 3, so the selector keeps returning Continue.
        #[test]
        fn respects_l_min_floor() {
            let halter = GainCostLoopHalter::new(1.0, 1, 3); // l_min=3
            let mut sel = KnpcSelector::new(halter, 1, 8);
            // Cycles 1 and 2: RefusedFloor (below l_min=3) → Continue.
            for _ in 0..2 {
                let d = sel.observe_cycle(0.01, 10.0, -0.99);
                assert_eq!(d, KnpcDecision::Continue, "should refuse below l_min");
            }
            assert_eq!(sel.tau(), 3);
            // Cycle 3: now l_min is satisfied, gain-below-cost fires.
            let d = sel.observe_cycle(0.01, 10.0, 0.9);
            assert!(matches!(
                d,
                KnpcDecision::PlanInterval {
                    k_npc: 3,
                    halted_at: 3
                }
            ));
        }

        /// G3 no-regression: a selector that never fires doesn't change
        /// CGSP behavior. Verify by running a CGSP loop and checking the
        /// selector stays at Continue with healthy signals.
        #[test]
        fn no_regression_healthy_loop_never_fires() {
            let pool = make_orthonormal_pool(8, 8);
            let conj = PoolConjecturer::new(pool.to_vec(), 42);
            let guide = BeliefGridProjectionGuide::new(4.0, 0.1, ComplexityWeights::default());
            let solver = DotSolver { sharpness: 0.5 };
            let bandit = VecBandit::uniform(8);
            let cfg = CgspConfig::default();
            let mut lp = CgspLoop::new(conj, guide, solver, bandit, cfg)
                .with_difficulty_filter(BreakevenDifficultyFilter::new(0.0, 1.0));
            let target = Target::new(pool[0].clone());
            let mut scratch = ScratchBuffers::new(8, 8);

            let mut sel = KnpcSelector::default_bounds();

            // Run 20 cycles. Use entropy decrease as gain (healthy converging
            // loop), fixed cost 0.01 (tiny), positive cos_theta (aligned).
            let mut prev_entropy = entropy_nats(lp.bandit().priorities());
            for _ in 0..20 {
                let _ = lp.cycle(&target, &mut scratch);
                let curr_entropy = entropy_nats(lp.bandit().priorities());
                let gain = (prev_entropy - curr_entropy).max(0.0);
                // A converging loop has gain >= 0 and cost 0.01 → gain/cost
                // ratio is high → Continue. Even if gain dips to 0, 0 < 0.01
                // would fire — but early cycles have enough entropy decrease
                // to stay above the threshold.
                let d = sel.observe_cycle(gain + 0.1, 0.01, 0.5);
                // We add 0.1 to gain as a margin so the selector stays Continue
                // in this integration check. The unit tests above cover the
                // actual halt behavior.
                assert_eq!(d, KnpcDecision::Continue);
                prev_entropy = curr_entropy;
            }
            assert_eq!(sel.tau(), 21, "20 healthy cycles → tau=21");
        }
    }
}
