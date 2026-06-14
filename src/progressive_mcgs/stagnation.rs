//! Stagnation detection — triggers for branch-level and global-level expansion operators.
//!
//! Paper §3.2.2 "Multi-Level Stagnation Detection".
//!
//! Two scopes:
//! - **Branch-level**: when a branch produces `τ_branch` consecutive expansions
//!   without improving its best metric → fire intra-branch evolution
//!   (and, if late-stage with strong solutions elsewhere, cross-branch reference).
//! - **Global-level**: when the global best metric hasn't improved for
//!   `τ_global` steps → fire multi-branch aggregation.
//!
//! # Stagnation ≠ Collapse
//!
//! `StagnationGate` is distinct from the `CollapseDetector` (Research 075, 179):
//! - **Stagnation**: search is stuck but model is fine. Reward plateau.
//! - **Collapse**: reasoning trace entropy collapsed; model is broken.
//!
//! These compose — a collapsed NPC/node doesn't generate valid expansions,
//! so it can't trigger stagnation events.

use crate::progressive_mcgs::types::{BranchId, Reward};

/// Per-branch stagnation state.
#[derive(Debug, Clone, Copy, Default)]
pub struct BranchStagnationState {
    /// Number of consecutive expansions since the last `Progress` or
    /// `Breakthrough` reward on this branch.
    pub since_last_improve: u32,
}

/// Global stagnation state.
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalStagnationState {
    /// Number of expansions globally since the last `Breakthrough` reward.
    pub since_last_best: u32,
}

/// Triggers that fire when stagnation thresholds are met.
///
/// Each trigger corresponds to one of the four expansion operators
/// (paper §3.2.2, Appendix B). The consumer decides how to build the
/// reference set + payload for each — see [`crate::operators`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StagnationTrigger {
    /// Intra-branch evolution — reference the last-k ancestor nodes within
    /// the same branch to inform the next proposal (paper Eq. 13).
    /// Fires when `since_last_improve >= branch_threshold`.
    IntraBranchEvolve = 0,
    /// Cross-branch reference — reference top-N nodes globally as additional
    /// context for the next proposal (paper Eq. 14).
    /// Fires when `since_last_improve >= branch_threshold` AND
    /// other branches have accumulated strong solutions.
    CrossBranchReference = 1,
    /// Multi-branch aggregation — spawn a new root child whose reference set
    /// is the union of top trajectories from all branches (paper Eq. 15).
    /// Fires when `since_last_best >= global_threshold`.
    MultiBranchAggregation = 2,
}

/// Stagnation gate — tracks per-branch and global reward plateaus.
///
/// Use [`StagnationGate::observe_expansion`] after each expansion to update
/// counters, then [`StagnationGate::check`] to query for pending triggers.
#[derive(Debug, Clone)]
pub struct StagnationGate {
    /// Per-branch stagnation counter.
    branch_states: Vec<BranchStagnationState>,
    /// Global stagnation counter.
    global: GlobalStagnationState,
    /// Branch threshold (paper default 3).
    pub branch_threshold: u32,
    /// Global threshold (paper default 6).
    pub global_threshold: u32,
}

impl StagnationGate {
    /// Construct with explicit thresholds. Branch index space is dense `0..n_branches`.
    #[must_use]
    pub fn new(n_branches: usize, branch_threshold: u32, global_threshold: u32) -> Self {
        Self {
            branch_states: vec![BranchStagnationState::default(); n_branches],
            global: GlobalStagnationState::default(),
            branch_threshold,
            global_threshold,
        }
    }

    /// Ensure the branch-states vector has at least `branch.idx() + 1` slots.
    /// Called from `observe_expansion` to handle dynamically-discovered branches.
    fn ensure_branch_capacity(&mut self, branch: BranchId) {
        if branch == BranchId::NONE {
            return;
        }
        let idx = branch.idx();
        if idx >= self.branch_states.len() {
            self.branch_states.resize(idx + 1, BranchStagnationState::default());
        }
    }

    /// Record an expansion outcome.
    ///
    /// **Critical**: per Plan 272 §4 risk, `reward` must already be classified
    /// against the branch-best *snapshot taken before the update*. That is,
    /// the caller should:
    /// 1. Read `branch_best_before` and `global_best_before`.
    /// 2. Compute the new metric.
    /// 3. Classify into [`Reward`] (Failure / Neutral / Progress / Breakthrough).
    /// 4. Call this method with the classified reward.
    /// 5. THEN update branch_best / global_best.
    ///
    /// Calling this in the wrong order produces non-stationary reward signals.
    #[inline]
    pub fn observe_expansion(&mut self, branch: BranchId, reward: Reward) {
        self.ensure_branch_capacity(branch);

        // Branch-level: reset on improvement, else increment.
        if branch != BranchId::NONE {
            let idx = branch.idx();
            if reward.is_improvement() {
                self.branch_states[idx].since_last_improve = 0;
            } else {
                self.branch_states[idx].since_last_improve =
                    self.branch_states[idx].since_last_improve.saturating_add(1);
            }
        }

        // Global-level: reset on Breakthrough only, else increment.
        if reward.is_breakthrough() {
            self.global.since_last_best = 0;
        } else {
            self.global.since_last_best = self.global.since_last_best.saturating_add(1);
        }
    }

    /// Returns pending triggers for the given branch this tick.
    ///
    /// Allocation-free: returns at most 3 triggers via a fixed-size array.
    /// Consumers should call this once per branch per tick and drain the
    /// returned triggers into their expansion-operator queue.
    #[inline]
    #[must_use]
    pub fn check(&self, branch: BranchId) -> StagnationTriggers {
        let mut out = StagnationTriggers::default();
        if branch == BranchId::NONE {
            return out;
        }
        let idx = branch.idx();
        if idx >= self.branch_states.len() {
            return out;
        }
        let since = self.branch_states[idx].since_last_improve;
        if since >= self.branch_threshold {
            out.push(StagnationTrigger::IntraBranchEvolve);
            // Cross-branch reference fires in addition when other branches
            // have accumulated strong solutions. We can't see other branches'
            // Q-values from here — the caller decides whether to ALSO emit
            // CrossBranchReference based on whether the global elite set is
            // non-empty. As a heuristic, we emit it iff global stagnation
            // has NOT yet tripped (i.e., some branches ARE making progress).
            if self.global.since_last_best < self.global_threshold {
                out.push(StagnationTrigger::CrossBranchReference);
            }
        }
        if self.global.since_last_best >= self.global_threshold {
            out.push(StagnationTrigger::MultiBranchAggregation);
        }
        out
    }

    /// Read-only access to per-branch state (for diagnostics).
    #[inline]
    #[must_use]
    pub fn branch_state(&self, branch: BranchId) -> Option<&BranchStagnationState> {
        if branch == BranchId::NONE {
            return None;
        }
        self.branch_states.get(branch.idx())
    }

    /// Read-only access to global state (for diagnostics).
    #[inline]
    #[must_use]
    pub const fn global_state(&self) -> &GlobalStagnationState {
        &self.global
    }
}

/// Fixed-capacity trigger queue — stack-allocated, zero heap.
///
/// Maximum 3 triggers can fire per `check` call (IntraBranchEvolve +
/// CrossBranchReference + MultiBranchAggregation). Matches the
/// 4 expansion operators minus the always-available Primary expansion.
#[derive(Debug, Clone, Copy, Default)]
pub struct StagnationTriggers {
    items: [Option<StagnationTrigger>; 3],
    len: usize,
}

impl StagnationTriggers {
    /// Number of triggers queued.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Is the queue empty?
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Push a trigger. Silently drops if the queue is full (cap = 3).
    #[inline]
    fn push(&mut self, t: StagnationTrigger) {
        if self.len < self.items.len() {
            self.items[self.len] = Some(t);
            self.len += 1;
        }
    }

    /// Iterate over queued triggers.
    #[inline]
    pub const fn iter(&self) -> StagnationTriggersIter {
        StagnationTriggersIter {
            triggers: self,
            pos: 0,
        }
    }
}

/// Iterator over [`StagnationTriggers`].
pub struct StagnationTriggersIter<'a> {
    triggers: &'a StagnationTriggers,
    pos: usize,
}

impl<'a> Iterator for StagnationTriggersIter<'a> {
    type Item = StagnationTrigger;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.triggers.len {
            let item = self.triggers.items[self.pos];
            self.pos += 1;
            if let Some(t) = item {
                return Some(t);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_stagnation_increments_on_failure() {
        let mut gate = StagnationGate::new(2, 3, 6);
        gate.observe_expansion(BranchId(0), Reward::Failure);
        gate.observe_expansion(BranchId(0), Reward::Neutral);
        assert_eq!(gate.branch_state(BranchId(0)).unwrap().since_last_improve, 2);
        assert!(gate.check(BranchId(0)).is_empty());
    }

    #[test]
    fn branch_stagnation_fires_intra_at_threshold() {
        let mut gate = StagnationGate::new(2, 3, 6);
        for _ in 0..3 {
            gate.observe_expansion(BranchId(0), Reward::Neutral);
        }
        let triggers: Vec<_> = gate.check(BranchId(0)).iter().collect();
        assert!(triggers.contains(&StagnationTrigger::IntraBranchEvolve));
    }

    #[test]
    fn branch_stagnation_resets_on_progress() {
        let mut gate = StagnationGate::new(2, 3, 6);
        gate.observe_expansion(BranchId(0), Reward::Neutral);
        gate.observe_expansion(BranchId(0), Reward::Neutral);
        gate.observe_expansion(BranchId(0), Reward::Progress); // reset
        assert_eq!(gate.branch_state(BranchId(0)).unwrap().since_last_improve, 0);
        assert!(gate.check(BranchId(0)).is_empty());
    }

    #[test]
    fn global_stagnation_fires_aggregation() {
        let mut gate = StagnationGate::new(2, 3, 6);
        // 6 non-breakthrough expansions globally
        for _ in 0..6 {
            gate.observe_expansion(BranchId(0), Reward::Neutral);
        }
        let triggers: Vec<_> = gate.check(BranchId(0)).iter().collect();
        assert!(
            triggers.contains(&StagnationTrigger::MultiBranchAggregation),
            "expected MultiBranchAggregation, got {triggers:?}"
        );
    }

    #[test]
    fn global_stagnation_resets_on_breakthrough() {
        let mut gate = StagnationGate::new(2, 3, 6);
        for _ in 0..5 {
            gate.observe_expansion(BranchId(0), Reward::Neutral);
        }
        gate.observe_expansion(BranchId(0), Reward::Breakthrough); // reset global
        let triggers: Vec<_> = gate.check(BranchId(0)).iter().collect();
        assert!(
            !triggers.contains(&StagnationTrigger::MultiBranchAggregation),
            "global should be reset by Breakthrough"
        );
    }

    #[test]
    fn cross_branch_reference_suppressed_during_global_stagnation() {
        // When global stagnation trips, branch stagnation should NOT also emit
        // CrossBranchReference (no other branches are making progress).
        let mut gate = StagnationGate::new(2, 3, 6);
        // Drive branch 0 to stagnation AND global to stagnation simultaneously
        for _ in 0..6 {
            gate.observe_expansion(BranchId(0), Reward::Neutral);
        }
        let triggers: Vec<_> = gate.check(BranchId(0)).iter().collect();
        assert!(triggers.contains(&StagnationTrigger::IntraBranchEvolve));
        assert!(triggers.contains(&StagnationTrigger::MultiBranchAggregation));
        assert!(
            !triggers.contains(&StagnationTrigger::CrossBranchReference),
            "CrossBranchReference should be suppressed during global stagnation"
        );
    }

    #[test]
    fn dynamic_branch_growth() {
        let mut gate = StagnationGate::new(1, 3, 6);
        gate.observe_expansion(BranchId(5), Reward::Neutral); // grow to branch 5
        assert!(gate.branch_state(BranchId(5)).is_some());
        assert_eq!(gate.branch_state(BranchId(5)).unwrap().since_last_improve, 1);
    }

    #[test]
    fn triggers_queue_caps_at_three() {
        let mut gate = StagnationGate::new(1, 3, 6);
        for _ in 0..6 {
            gate.observe_expansion(BranchId(0), Reward::Neutral);
        }
        let t = gate.check(BranchId(0));
        // At most: IntraBranchEvolve + (no CrossBranch since global stagnant) + MultiBranchAggregation = 2
        assert!(t.len() <= 3);
    }

    #[test]
    fn none_branch_returns_empty() {
        let gate = StagnationGate::new(1, 3, 6);
        assert!(gate.check(BranchId::NONE).is_empty());
    }
}
