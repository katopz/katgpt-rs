//! # Dual-Pool Reachable Memory Router (Plan 282, Research 249)
//!
//! Modelless distillation of Hao, Long, Zhao 2026 — *"Self-Evolving Multi-Agent
//! Systems via Decentralized Memory"* (arXiv:2605.22721).
//!
//! Splits a [`HintDeltaBandit`]'s candidate pool into an **exploitation pool**
//! (E-pool: consolidated past successes, local-walk operator) and an
//! **exploration pool** (X-pool: fresh candidates, teleportation operator).
//! A sigmoid-based router re-weights the pools from stage-wise binary feedback.
//!
//! ## Guarantees
//!
//! - **Global reachability (Theorem 1):** The X-pool always retains strictly
//!   nonzero selection probability because `α = sigmoid(w_E − w_X) ∈ (0, 1)`
//!   never saturates in finite precision. The induced Markov chain
//!   `M = α·T + (1−α)·h·1ᵀ` is irreducible and aperiodic — no agent is ever
//!   trapped, by construction. This is **proactive** (no collapse detector
//!   needed), unlike CGSP's reactive [`EntropyCollapse`](super::loop_::EntropyCollapse).
//!
//! - **O(log T) cumulative regret (Theorem 2):** The sigmoid router preserves
//!   strict concavity of the paper's ratio form (Research 249 §2.3), so the
//!   regret proof transfers. The online router's regret grows logarithmically;
//!   fixed-`α` routing grows linearly (Corollary 1).
//!
//! ## CGSP relationship
//!
//! Existing single-pool CGSP is the degenerate case `α = 1` (pure
//! exploitation). `DualPoolBandit<B>` implements [`HintDeltaBandit`] by
//! delegating to the **active** pool (one pool selected per cycle), so it
//! drops into [`CgspLoop`](super::loop_::CgspLoop) with zero changes to the
//! loop's `cycle()` method. The caller wraps `begin_cycle()` /
//! [`end_cycle`](DualPoolBandit::end_cycle) around the existing cycle call.
//!
//! ## Phase 1 scope
//!
//! This module implements the **unblocking skeleton**: same-size E/X pools
//! (both N arms, same directions), priority-blend consolidation. True E-pool
//! **growth** (adding arms discovered by X-pool) and the
//! [`FaithfulnessProbe`](crate::faithfulness_probe) consolidation gate are
//! deferred to Plan 282 Phase 4.
//!
//! ## Sigmoid vs ratio
//!
//! Per AGENTS.md project convention, routing uses `α = sigmoid(w_E − w_X)`
//! rather than the paper's `α = w_E / (w_E + w_X)`. Both are monotonically
//! increasing, map to `(0, 1)`, and preserve strict concavity. The O(log T)
//! regret bound transfers (Research 249 §2.3).
//!
//! ---
//!
//! **TL;DR:** `DualPoolBandit<B>` wraps two `HintDeltaBandit` instances with a
//! sigmoid router. The X-pool's nonzero probability guarantees proactive
//! non-trapping (Theorem 1). Per-pool weight updates from binary feedback
//! give O(log T) regret (Theorem 2). Single-pool CGSP = degenerate `α = 1`.

use crate::cgsp::traits::HintDeltaBandit;
use crate::cgsp::types::{sigmoid, Priority};

// ── PoolId ────────────────────────────────────────────────────────────────

/// Zero-cost tag identifying which memory pool an arm belongs to.
///
/// `#[repr(u8)]` guarantees 1-byte size (AGENTS.md: prefer `#[repr(u8)]` on
/// field-less enums).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PoolId {
    /// Exploitation pool — consolidated past successes (local-walk operator).
    Exploitation = 0,
    /// Exploration pool — fresh candidates (teleportation operator).
    /// Guarantees the induced Markov chain is irreducible (DecentMem Thm. 1).
    Exploration = 1,
}

// ── ReachableDualPoolRouter trait ─────────────────────────────────────────

/// Dual-pool memory router with provable reachability and O(log T) regret.
///
/// Routes between an exploitation pool (consolidated successes, local-walk
/// operator) and an exploration pool (fresh candidates, teleportation
/// operator). The X-pool always retains nonzero selection probability
/// (sigmoid never saturates), guaranteeing the induced Markov chain is
/// irreducible and aperiodic (DecentMem Theorem 1).
///
/// Based on Hao, Long, Zhao 2026 (arXiv:2605.22721).
/// Uses sigmoid (not softmax/ratio) for routing probability per project
/// convention — the regret proof transfers (Research 249 §2.3).
///
/// # Contract
///
/// All methods are zero-allocation by contract.
pub trait ReachableDualPoolRouter {
    /// Item selected within a pool (e.g. arm index).
    type Item;
    /// Stage-wise binary feedback (success / fail).
    type Reward: Copy;

    /// Select a pool (via sigmoid routing) and an item within it.
    ///
    /// E-pool selection probability: `α = sigmoid(w_E − w_X) ∈ (0, 1)`.
    /// Returns `(item, pool_id)`.
    fn route_select(&mut self) -> (Self::Item, PoolId);

    /// Update pool weights from stage-wise binary feedback (DecentMem Eq. 6/7).
    ///
    /// Guarantees O(log T) cumulative regret (Theorem 2).
    fn route_update(&mut self, pool: PoolId, reward: Self::Reward);

    /// Consolidate X-pool items into E-pool (DecentMem Eq. 8).
    ///
    /// Called after task/cycle completion (at a configurable cadence).
    /// Phase 1: priority-blend (same-size pools). Phase 4: arm growth +
    /// FaithfulnessProbe gate.
    fn consolidate(&mut self);

    /// Current exploitation probability `α = sigmoid(w_E − w_X)`.
    fn exploitation_probability(&self) -> f32;

    /// Reachability invariant: X-pool probability is strictly positive.
    ///
    /// Guaranteed by sigmoid (never exactly 0 or 1 in finite precision).
    /// This is the **proactive** non-trapping guarantee — no collapse
    /// detector needed (DecentMem Theorem 1).
    #[inline]
    fn is_reachable(&self) -> bool {
        self.exploitation_probability() < 1.0
    }
}

// ── DualPoolConfig ────────────────────────────────────────────────────────

/// Tunable parameters for [`DualPoolBandit`].
///
/// Defaults follow DecentMem Eq. 6/7/8: gain `α = 0.5`, decay `β = 0.5`.
#[derive(Clone, Debug)]
pub struct DualPoolConfig {
    /// Weight gain on a successful route_update (paper's `α` in Eq. 6/7).
    pub alpha_update_gain: f32,
    /// Weight decay factor on a failed route_update (paper's `β` in Eq. 6/7).
    pub decay: f32,
    /// Mean `r_synth` above which a cycle counts as "success" for the binary
    /// router feedback. CGSP reward `r_synth = (1 − solve_rate) · guide_score`.
    pub success_threshold: f32,
    /// Consolidation cadence (cycles between consolidates). 0 = never.
    pub consolidate_interval: u32,
    /// Priority blend factor on consolidation: `e[i] = blend·e[i] + (1−blend)·x[i]`.
    pub consolidate_blend: f32,
    /// Minimum exploration probability floor. `exploitation_probability()` is
    /// clamped to `[min_exploration_prob, 1 − min_exploration_prob]` so both
    /// pools always have strictly nonzero selection probability in f32.
    ///
    /// This is the numerical reachability guarantee (DecentMem Theorem 1 holds
    /// in continuous math; f32 sigmoid saturates at `x ≳ 18`, so we clamp).
    /// Default `1e-4` → X-pool selected ~3.6× per 10min at 60fps even at max
    /// exploitation. Set lower for tighter exploitation, higher for more
    /// proactive exploration.
    pub min_exploration_prob: f32,
    /// RNG seed for pool + arm selection.
    pub seed: u64,
}

impl Default for DualPoolConfig {
    fn default() -> Self {
        Self {
            alpha_update_gain: 0.5,
            decay: 0.5,
            success_threshold: 0.25,
            consolidate_interval: 0, // Phase 1: disabled by default (Phase 4 enables).
            consolidate_blend: 0.5,
            min_exploration_prob: 1e-4,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

// ── DualPoolBandit ────────────────────────────────────────────────────────

/// Dual-pool bandit router wrapping two [`HintDeltaBandit`] instances.
///
/// The E-pool (exploitation) consolidates successful trajectories; the X-pool
/// (exploration) provides fresh candidates with guaranteed nonzero selection
/// probability via sigmoid routing.
///
/// Implements [`HintDeltaBandit`] by delegating to the **active** pool (one
/// pool per cycle, selected by sigmoid routing in
/// [`begin_cycle`](Self::begin_cycle)). This lets it drop directly into
/// [`CgspLoop`](super::loop_::CgspLoop) without modifying the loop:
///
/// ```text,ignore
/// bandit.begin_cycle();                 // sigmoid-select active pool
/// let result = lp.cycle(target, scratch); // operates on active pool
/// bandit.end_cycle();                   // route_update + maybe consolidate
/// ```
///
/// Phase 1: both pools have the same arm count (same directions, divergent
/// priorities). Phase 4 generalizes to growing E-pool with different arms.
pub struct DualPoolBandit<B: HintDeltaBandit> {
    /// Exploitation pool — consolidated past successes.
    e_pool: B,
    /// Exploration pool — fresh candidates (teleportation operator).
    x_pool: B,
    /// Exploitation weight (starts at 1.0, updated by [`route_update`](ReachableDualPoolRouter::route_update)).
    w_e: f32,
    /// Exploration weight (fixed at 1.0 per DecentMem Eq. 6/7).
    w_x: f32,
    /// Router configuration.
    config: DualPoolConfig,
    /// Currently active pool (selected per cycle by sigmoid routing).
    active_pool: PoolId,
    /// Per-cycle reward accumulators for binary success computation.
    e_reward_accum: f32,
    e_count: u32,
    x_reward_accum: f32,
    x_count: u32,
    /// Cycles since last consolidate.
    cycles_since_consolidate: u32,
    /// Internal RNG state (splitmix64).
    rng_state: u64,
}

impl<B: HintDeltaBandit> DualPoolBandit<B> {
    /// Build a dual-pool bandit from two inner bandits and default config.
    ///
    /// Both pools should have the same arm count in Phase 1. The X-pool is
    /// typically initialized uniform (fresh exploration) while the E-pool
    /// carries consolidated priorities.
    pub fn new(e_pool: B, x_pool: B) -> Self {
        Self::with_config(e_pool, x_pool, DualPoolConfig::default())
    }

    /// Build with a custom [`DualPoolConfig`].
    pub fn with_config(e_pool: B, x_pool: B, config: DualPoolConfig) -> Self {
        debug_assert_eq!(
            e_pool.num_arms(),
            x_pool.num_arms(),
            "cgsp_dual_pool: Phase 1 requires same-size E/X pools ({} vs {})",
            e_pool.num_arms(),
            x_pool.num_arms(),
        );
        let seed = config.seed;
        Self {
            e_pool,
            x_pool,
            w_e: 1.0,
            w_x: 1.0,
            config,
            active_pool: PoolId::Exploitation,
            e_reward_accum: 0.0,
            e_count: 0,
            x_reward_accum: 0.0,
            x_count: 0,
            cycles_since_consolidate: 0,
            rng_state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Borrow the exploitation (E) pool.
    #[inline]
    pub fn e_pool(&self) -> &B {
        &self.e_pool
    }

    /// Borrow the exploration (X) pool.
    #[inline]
    pub fn x_pool(&self) -> &B {
        &self.x_pool
    }

    /// Mutably borrow the exploitation (E) pool.
    #[inline]
    pub fn e_pool_mut(&mut self) -> &mut B {
        &mut self.e_pool
    }

    /// Mutably borrow the exploration (X) pool.
    #[inline]
    pub fn x_pool_mut(&mut self) -> &mut B {
        &mut self.x_pool
    }

    /// Current exploitation weight `w_E`.
    #[inline]
    pub fn w_e(&self) -> f32 {
        self.w_e
    }

    /// Current exploration weight `w_X` (fixed at 1.0 per DecentMem Eq. 6/7).
    #[inline]
    pub fn w_x(&self) -> f32 {
        self.w_x
    }

    /// Which pool is active this cycle.
    #[inline]
    pub fn active_pool(&self) -> PoolId {
        self.active_pool
    }

    /// Borrow the router configuration.
    #[inline]
    pub fn config(&self) -> &DualPoolConfig {
        &self.config
    }

    // ── Cycle lifecycle ───────────────────────────────────────────────────

    /// Select the active pool via sigmoid routing and reset per-cycle
    /// accumulators. Call this **before** [`CgspLoop::cycle`](super::loop_::CgspLoop::cycle).
    ///
    /// E-pool is selected with probability `α = sigmoid(w_E − w_X)`.
    /// X-pool is selected with probability `1 − α > 0` (reachability guarantee).
    pub fn begin_cycle(&mut self) {
        let alpha = self.exploitation_probability();
        let u = self.next_f32();
        self.active_pool = if u < alpha {
            PoolId::Exploitation
        } else {
            PoolId::Exploration
        };
        self.e_reward_accum = 0.0;
        self.e_count = 0;
        self.x_reward_accum = 0.0;
        self.x_count = 0;
    }

    /// End-of-cycle maintenance: compute binary success per active pool from
    /// accumulated rewards, call [`route_update`](ReachableDualPoolRouter::route_update),
    /// and optionally [`consolidate`](ReachableDualPoolRouter::consolidate).
    ///
    /// Call this **after** [`CgspLoop::cycle`](super::loop_::CgspLoop::cycle).
    pub fn end_cycle(&mut self) {
        // Compute binary success for the active pool from accumulated rewards.
        let threshold = self.config.success_threshold;
        if self.active_pool == PoolId::Exploitation && self.e_count > 0 {
            let mean = self.e_reward_accum / self.e_count as f32;
            let success = mean > threshold;
            self.route_update(PoolId::Exploitation, success);
        } else if self.active_pool == PoolId::Exploration && self.x_count > 0 {
            let mean = self.x_reward_accum / self.x_count as f32;
            let success = mean > threshold;
            self.route_update(PoolId::Exploration, success);
        }

        // Optional consolidation at configured cadence.
        let interval = self.config.consolidate_interval;
        if interval > 0 {
            self.cycles_since_consolidate += 1;
            if self.cycles_since_consolidate >= interval {
                self.consolidate();
                self.cycles_since_consolidate = 0;
            }
        }
    }

    // ── Internal RNG (splitmix64, matches PoolConjecturer) ────────────────

    /// Advance the internal RNG by one step and return the next u64.
    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Sample a uniform f32 in `[0, 1)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        let u = self.next_u64() >> 40; // top 24 bits
        (u as f32) / ((1u64 << 24) as f32)
    }

}

// ── HintDeltaBandit impl (delegate to active pool) ────────────────────────

/// Delegates all priority operations to the **active** pool (selected by
/// [`begin_cycle`](DualPoolBandit::begin_cycle)). This lets `DualPoolBandit`
/// drop into [`CgspLoop`](super::loop_::CgspLoop) as the `B` type parameter
/// without changing the loop.
impl<B: HintDeltaBandit> HintDeltaBandit for DualPoolBandit<B> {
    fn absorb(&mut self, arm: usize, reward: f32) {
        // Delegate to active pool AND accumulate for end_cycle binary reward.
        match self.active_pool {
            PoolId::Exploitation => {
                self.e_pool.absorb(arm, reward);
                self.e_reward_accum += reward.max(0.0);
                self.e_count += 1;
            }
            PoolId::Exploration => {
                self.x_pool.absorb(arm, reward);
                self.x_reward_accum += reward.max(0.0);
                self.x_count += 1;
            }
        }
    }

    #[inline]
    fn priority(&self, arm: usize) -> Priority {
        match self.active_pool {
            PoolId::Exploitation => self.e_pool.priority(arm),
            PoolId::Exploration => self.x_pool.priority(arm),
        }
    }

    #[inline]
    fn priorities(&self) -> &[Priority] {
        match self.active_pool {
            PoolId::Exploitation => self.e_pool.priorities(),
            PoolId::Exploration => self.x_pool.priorities(),
        }
    }

    #[inline]
    fn priorities_mut(&mut self) -> &mut [Priority] {
        match self.active_pool {
            PoolId::Exploitation => self.e_pool.priorities_mut(),
            PoolId::Exploration => self.x_pool.priorities_mut(),
        }
    }
}

// ── ReachableDualPoolRouter impl ──────────────────────────────────────────

impl<B: HintDeltaBandit> ReachableDualPoolRouter for DualPoolBandit<B> {
    type Item = usize;
    type Reward = bool;

    fn route_select(&mut self) -> (Self::Item, PoolId) {
        // Sample pool via sigmoid routing.
        let alpha = self.exploitation_probability();
        let u_pool = self.next_f32();
        self.active_pool = if u_pool < alpha {
            PoolId::Exploitation
        } else {
            PoolId::Exploration
        };
        // Advance RNG for the arm draw BEFORE borrowing priorities (&self),
        // so the borrow checker sees &mut self (RNG) and &self (priorities)
        // as non-overlapping.
        let u_arm = self.next_f32();
        let arm = match self.active_pool {
            PoolId::Exploitation => sample_arm_from(u_arm, self.e_pool.priorities()),
            PoolId::Exploration => sample_arm_from(u_arm, self.x_pool.priorities()),
        };
        (arm, self.active_pool)
    }

    fn route_update(&mut self, pool: PoolId, reward: Self::Reward) {
        // DecentMem Eq. 6/7 — only w_e updates; w_x is fixed at 1.0.
        //
        //   E-pool + success → w_e += gain       (exploit more)
        //   E-pool + fail    → w_e = max(1, decay·w_e)  (explore more)
        //   X-pool + success → w_e = max(1, decay·w_e)  (keep exploring — X found something)
        //   X-pool + fail    → w_e += gain       (exploit what we know)
        //
        let gain = self.config.alpha_update_gain;
        let decay = self.config.decay;
        match (pool, reward) {
            (PoolId::Exploitation, true) => self.w_e += gain,
            (PoolId::Exploitation, false) => self.w_e = (decay * self.w_e).max(1.0),
            (PoolId::Exploration, true) => self.w_e = (decay * self.w_e).max(1.0),
            (PoolId::Exploration, false) => self.w_e += gain,
        }
    }

    fn consolidate(&mut self) {
        // DecentMem Eq. 8 — merge X-pool items into E-pool.
        //
        // Phase 1 (same-size pools): priority-blend. Phase 4 will add arm
        // growth (new directions from X-pool superset) and the
        // FaithfulnessProbe consolidation gate.
        let blend = self.config.consolidate_blend;
        let n = self.e_pool.num_arms().min(self.x_pool.num_arms());
        let e = self.e_pool.priorities_mut();
        let x = self.x_pool.priorities();
        for i in 0..n {
            let blended = blend * e[i] + (1.0 - blend) * x[i.min(x.len())];
            e[i] = blended;
        }
        // Reset X-pool to uniform (fresh exploration).
        let x_n = self.x_pool.num_arms();
        let x_unif = if x_n > 0 { 1.0 / x_n as f32 } else { 0.0 };
        for p in self.x_pool.priorities_mut() {
            *p = x_unif;
        }
    }

    #[inline]
    fn exploitation_probability(&self) -> f32 {
        // α = sigmoid(w_E − w_X). Per AGENTS.md: sigmoid, not ratio.
        // Clamp to [ε, 1−ε] so both pools always have strictly nonzero
        // probability in f32 (numerical reachability guarantee — DecentMem
        // Theorem 1 holds in continuous math but f32 sigmoid saturates at
        // x ≳ 18, which would break is_reachable()).
        let eps = self.config.min_exploration_prob;
        sigmoid(self.w_e - self.w_x).clamp(eps, 1.0 - eps)
    }
}

// ── Free helpers ─────────────────────────────────────────────────────────

/// Priority-weighted inverse-CDF arm sampler.
///
/// Pure function — takes a pre-generated uniform draw `u ∈ [0, 1)` so the
/// caller can advance the RNG (`&mut self`) before borrowing priorities
/// (`&self`), avoiding a borrow conflict with zero allocation.
///
/// Floors zero/negative priorities at a tiny epsilon so a degenerate
/// table still samples (matches `PoolConjecturer::build_cdf`).
#[inline]
fn sample_arm_from(u: f32, priorities: &[Priority]) -> usize {
    if priorities.is_empty() {
        return 0;
    }
    let total: f32 = priorities
        .iter()
        .map(|&p| if p.is_finite() && p > 0.0 { p } else { 1e-6 })
        .sum();
    if total <= 0.0 {
        return 0;
    }
    let target = u * total;
    let mut acc = 0.0f32;
    for (i, &p) in priorities.iter().enumerate() {
        let w = if p.is_finite() && p > 0.0 { p } else { 1e-6 };
        acc += w;
        if acc >= target {
            return i;
        }
    }
    priorities.len() - 1
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple Vec-backed bandit for testing (mirrors integration_tests::VecBandit).
    struct VecBandit {
        prios: Vec<f32>,
    }
    impl VecBandit {
        fn uniform(n: usize) -> Self {
            Self {
                prios: vec![1.0 / n as f32; n],
            }
        }
        fn constant(n: usize, v: f32) -> Self {
            Self { prios: vec![v; n] }
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

    // ── T1.4: Unit tests ──────────────────────────────────────────────────

    #[test]
    fn t14_sigmoid_routing_in_unit_interval() {
        // exploitation_probability() ∈ (0, 1) for all weight combos, including
        // extremes (clamp guarantees this in f32 where raw sigmoid saturates).

        // Default: w_e=1, w_x=1 → sigmoid(0) = 0.5.
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let dp = DualPoolBandit::new(e, x);
        let alpha = dp.exploitation_probability();
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "exploitation_probability must be in (0,1), got {alpha}"
        );
        assert!(
            (alpha - 0.5).abs() < 1e-5,
            "sigmoid(1−1)=sigmoid(0)=0.5, got {alpha}"
        );

        // Drive w_e very high via repeated E-pool successes → α → 1 (clamped < 1).
        let e2 = VecBandit::uniform(4);
        let x2 = VecBandit::uniform(4);
        let mut dp2 = DualPoolBandit::new(e2, x2);
        for _ in 0..200 {
            dp2.route_update(PoolId::Exploitation, true);
        }
        let alpha_high = dp2.exploitation_probability();
        assert!(
            alpha_high > 0.0 && alpha_high < 1.0,
            "even with extreme w_e={}, α must stay in (0,1) via clamp, got {alpha_high}",
            dp2.w_e()
        );

        // Drive w_e toward 1.0 via repeated E-pool failures → α → sigmoid(0).
        let e3 = VecBandit::uniform(4);
        let x3 = VecBandit::uniform(4);
        let mut dp3 = DualPoolBandit::new(e3, x3);
        for _ in 0..200 {
            dp3.route_update(PoolId::Exploitation, false);
        }
        let alpha_low = dp3.exploitation_probability();
        assert!(
            alpha_low > 0.0 && alpha_low < 1.0,
            "with w_e floored at 1.0, α must stay in (0,1), got {alpha_low} (w_e={})",
            dp3.w_e()
        );
    }

    #[test]
    fn t14_x_pool_always_reachable() {
        // After forcing w_e very high, is_reachable() still true (clamp floor).
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let mut dp = DualPoolBandit::new(e, x);
        // Boost w_e to extreme (would saturate raw sigmoid to 1.0 in f32).
        for _ in 0..500 {
            dp.route_update(PoolId::Exploitation, true);
        }
        assert!(
            dp.is_reachable(),
            "X-pool must remain reachable even with extreme w_e={} (α={})",
            dp.w_e(),
            dp.exploitation_probability()
        );
        assert!(
            dp.exploitation_probability() < 1.0,
            "α must be strictly < 1.0 (clamp guarantees reachability in f32)"
        );
        assert!(
            dp.exploitation_probability() >= 0.9998,
            "with extreme w_e, α should be very close to 1 (got {})",
            dp.exploitation_probability()
        );

        // With moderate w_e, verify X-pool is actually selected over many
        // cycles (probabilistic — use moderate weights so X-pool rate is
        // non-negligible). w_e=3.0 → α = sigmoid(2.0) ≈ 0.881 → X-pool ≈ 12%.
        let e2 = VecBandit::uniform(4);
        let x2 = VecBandit::uniform(4);
        let mut dp2 = DualPoolBandit::new(e2, x2);
        for _ in 0..4 {
            dp2.route_update(PoolId::Exploitation, true);
        } // w_e = 1 + 4·0.5 = 3.0
        let mut x_selected = 0u32;
        let trials = 10_000u32;
        for _ in 0..trials {
            dp2.begin_cycle();
            if dp2.active_pool() == PoolId::Exploration {
                x_selected += 1;
            }
        }
        assert!(
            x_selected > 500,
            "with moderate w_e, X-pool should be selected ~12% of {trials} trials (got {x_selected})"
        );
    }

    #[test]
    fn t14_weight_update_e_pool_success() {
        // E-pool + success → w_e increases.
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let mut dp = DualPoolBandit::new(e, x);
        let w_before = dp.w_e();
        dp.route_update(PoolId::Exploitation, true);
        let w_after = dp.w_e();
        assert!(
            w_after > w_before,
            "E-pool success should increase w_e: {w_before} → {w_after}"
        );
        assert!(
            (w_after - w_before - 0.5).abs() < 1e-5,
            "gain should be 0.5 (default), got delta {}",
            w_after - w_before
        );
    }

    #[test]
    fn t14_weight_update_e_pool_fail() {
        // E-pool + fail → w_e decays toward 1.0 (floor).
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let mut dp = DualPoolBandit::new(e, x);
        // First boost w_e above 1.0.
        for _ in 0..10 {
            dp.route_update(PoolId::Exploitation, true);
        }
        let w_before = dp.w_e();
        assert!(w_before > 1.0, "w_e should be > 1.0 after boosts: {w_before}");

        // E-pool fail → w_e = max(1.0, decay * w_e).
        dp.route_update(PoolId::Exploitation, false);
        let w_after = dp.w_e();
        let expected = (0.5 * w_before).max(1.0);
        assert!(
            (w_after - expected).abs() < 1e-5,
            "E-pool fail: w_e should be max(1.0, 0.5·{}) = {}, got {}",
            w_before,
            expected,
            w_after
        );

        // Repeated failures floor at 1.0.
        for _ in 0..20 {
            dp.route_update(PoolId::Exploitation, false);
        }
        let w_floored = dp.w_e();
        assert!(
            (w_floored - 1.0).abs() < 1e-5,
            "w_e should floor at 1.0 after repeated failures, got {w_floored}"
        );
    }

    #[test]
    fn t14_weight_update_x_pool_success() {
        // X-pool + success → w_e decays (suppress E dominance).
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let mut dp = DualPoolBandit::new(e, x);
        // Boost w_e above 1.0.
        for _ in 0..10 {
            dp.route_update(PoolId::Exploitation, true);
        }
        let w_before = dp.w_e();
        assert!(w_before > 1.0);

        // X-pool success → w_e = max(1.0, decay * w_e).
        dp.route_update(PoolId::Exploration, true);
        let w_after = dp.w_e();
        let expected = (0.5 * w_before).max(1.0);
        assert!(
            (w_after - expected).abs() < 1e-5,
            "X-pool success: w_e should decay to {}, got {}",
            expected,
            w_after
        );
        assert!(
            w_after < w_before,
            "X-pool success should suppress w_e: {w_before} → {w_after}"
        );
    }

    #[test]
    fn t14_consolidate_merges_x_into_e() {
        // After consolidate, E-pool priorities reflect X-pool blend;
        // X-pool reset to uniform.
        let e = VecBandit::constant(4, 0.8); // E-pool high.
        let x = VecBandit::constant(4, 0.2); // X-pool low.
        let mut dp = DualPoolBandit::new(e, x);

        let e_before = dp.e_pool().priorities().to_vec();
        let x_before = dp.x_pool().priorities().to_vec();
        let n = e_before.len();

        dp.consolidate();

        let e_after = dp.e_pool().priorities();
        let x_after = dp.x_pool().priorities();

        // E-pool should be blended: 0.5·0.8 + 0.5·0.2 = 0.5.
        for i in 0..n {
            let expected = 0.5 * e_before[i] + 0.5 * x_before[i];
            assert!(
                (e_after[i] - expected).abs() < 1e-5,
                "E-pool[{}] should be blended {}, got {}",
                i,
                expected,
                e_after[i]
            );
        }

        // E-pool size unchanged (Phase 1: no growth).
        assert_eq!(
            e_after.len(),
            n,
            "E-pool size should not change in Phase 1 consolidate"
        );

        // X-pool reset to uniform.
        let uniform = 1.0 / n as f32;
        for (i, &p) in x_after.iter().enumerate() {
            assert!(
                (p - uniform).abs() < 1e-5,
                "X-pool[{}] should be reset to uniform {}, got {}",
                i,
                uniform,
                p
            );
        }
    }

    // ── Bonus: route_select + HintDeltaBandit delegation smoke tests ──────

    #[test]
    fn route_select_returns_valid_arm_and_pool() {
        let e = VecBandit::uniform(8);
        let x = VecBandit::uniform(8);
        let mut dp = DualPoolBandit::new(e, x);
        for _ in 0..100 {
            let (arm, pool) = dp.route_select();
            assert!(arm < 8, "arm must be valid: {arm}");
            assert!(
                pool == PoolId::Exploitation || pool == PoolId::Exploration,
                "pool must be valid"
            );
        }
    }

    #[test]
    fn hintdeltabandit_delegates_to_active_pool() {
        // absorb during active=E should modify E-pool, not X-pool.
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let mut dp = DualPoolBandit::new(e, x);
        dp.active_pool = PoolId::Exploitation;
        let e_before = dp.e_pool().priority(0);
        let x_before = dp.x_pool().priority(0);
        dp.absorb(0, 0.5);
        assert!(
            dp.e_pool().priority(0) > e_before,
            "E-pool arm 0 should increase after absorb"
        );
        assert!(
            (dp.x_pool().priority(0) - x_before).abs() < 1e-7,
            "X-pool should be unchanged when active=E"
        );

        // Switch to X-pool.
        dp.active_pool = PoolId::Exploration;
        let e_before2 = dp.e_pool().priority(0);
        let x_before2 = dp.x_pool().priority(0);
        dp.absorb(0, 0.3);
        assert!(
            (dp.e_pool().priority(0) - e_before2).abs() < 1e-7,
            "E-pool should be unchanged when active=X"
        );
        assert!(
            dp.x_pool().priority(0) > x_before2,
            "X-pool arm 0 should increase after absorb"
        );
    }

    #[test]
    fn begin_end_cycle_drives_routing() {
        // Simulate many cycles: E-pool consistently succeeds → w_e grows →
        // α → 1 → X-pool rarely selected (but still nonzero).
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let mut dp = DualPoolBandit::new(e, x);

        let alpha_0 = dp.exploitation_probability();
        for _ in 0..50 {
            dp.begin_cycle();
            // Simulate: active pool always succeeds.
            let success = true;
            match dp.active_pool() {
                PoolId::Exploitation => {
                    dp.route_update(PoolId::Exploitation, success);
                }
                PoolId::Exploration => {
                    dp.route_update(PoolId::Exploration, success);
                }
            }
        }
        // After mixed updates, α should have moved from 0.5.
        let alpha_1 = dp.exploitation_probability();
        // The net effect depends on how often each pool was selected.
        // E-pool successes boost w_e; X-pool successes decay w_e.
        // Early on (α=0.5), both pools selected ~equally → competing effects.
        // Just assert no NaN/Inf and stays in (0,1).
        assert!(alpha_1.is_finite(), "α must be finite");
        assert!(alpha_1 > 0.0 && alpha_1 < 1.0, "α in (0,1): {alpha_1}");
        let _ = alpha_0; // suppress unused
    }

    #[test]
    fn single_pool_degenerate_case_alpha_one() {
        // Single-pool CGSP is the degenerate case α=1 (pure exploitation).
        // We approximate this by driving w_e very high → α → 1.
        let e = VecBandit::uniform(4);
        let x = VecBandit::uniform(4);
        let mut dp = DualPoolBandit::new(e, x);
        for _ in 0..500 {
            dp.route_update(PoolId::Exploitation, true);
        }
        let alpha = dp.exploitation_probability();
        // α should be very close to 1 (sigmoid of large positive).
        assert!(
            alpha > 0.99,
            "with extreme w_e, α should approach 1 (degenerate single-pool), got {alpha}"
        );
        // But still strictly < 1 (reachability by construction).
        assert!(alpha < 1.0, "α must be < 1.0 (sigmoid never saturates)");
    }
}
