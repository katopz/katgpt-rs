//! `BranchBank` — bounded bank of persistent cognitive branches with
//! spawn / merge / prune lifecycle (Plan 329 T1.3).
//!
//! The bank pre-allocates a `Vec` of capacity `max_branches` and maintains a
//! free-list of slot indices for O(1) reuse on prune. The hot path
//! (`active_branches()` iterator consumed by the router) is a single linear
//! scan over the slot array with a branch-predictable lifecycle filter.
//!
//! # Allocation discipline
//!
//! - `new(max_branches)` allocates the `branches` Vec **once** with
//!   `with_capacity(max_branches)` and an empty free-list.
//! - `spawn` either reuses a free slot (no alloc) or pushes a new branch
//!   (allocates the branch's internal Vecs — unavoidable for a fresh branch).
//! - `prune` clears the branch's memory Vecs (`clear()` preserves capacity for
//!   reuse) and pushes the slot index onto the free-list.
//! - `merge` drains one branch into another (`append` reuses capacity) and
//!   prunes the source slot.
//!
//! The hot read-path (`active_branches`) allocates nothing.

use crate::branching::types::{
    BranchId, BranchLifecycle, CognitiveBranch,
};

/// Default maximum active branches. Matches the G2 perf target (router < 1µs
/// at ≤64 branches). Callers may override via [`BranchBank::new`].
pub const DEFAULT_MAX_BRANCHES: usize = 64;

/// Bounded bank of persistent [`CognitiveBranch`]es.
///
/// Each branch occupies a stable slot indexed by its [`BranchId`]. Pruned
/// branches keep their slot (lifecycle = `Removed`) and the slot is added to
/// the free-list for reuse on the next `spawn`. The invariant
/// `free_slots.len() == branches.len() - n_active` is maintained across all
/// operations.
///
/// Branch count is bounded by `max_branches` (the slot capacity). Once all
/// slots are active, `spawn` returns `None` (the router reports `Frozen`).
pub struct BranchBank<E: Clone> {
    /// Dense slot array; index == `BranchId.0 as usize`. Pruned slots stay in
    /// place with `lifecycle = Removed` until reused.
    branches: Vec<CognitiveBranch<E>>,
    /// Stack of slot indices available for reuse (LIFO).
    free_slots: Vec<u32>,
    /// Maximum number of slots (active + removed).
    max_branches: usize,
    /// Current count of branches with `lifecycle.is_routable()`.
    n_active: usize,
}

impl<E: Clone> BranchBank<E> {
    /// Construct an empty bank with the given slot capacity.
    ///
    /// Pre-allocates `branches` with `with_capacity(max_branches)` so `spawn`
    /// never reallocates the slot array. The free-list starts empty.
    #[inline]
    #[must_use]
    pub fn new(max_branches: usize) -> Self {
        Self {
            branches: Vec::with_capacity(max_branches),
            free_slots: Vec::with_capacity(max_branches),
            max_branches,
            n_active: 0,
        }
    }

    /// Construct with [`DEFAULT_MAX_BRANCHES`].
    #[inline]
    #[must_use]
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_MAX_BRANCHES)
    }

    /// Maximum number of slots (active + removed).
    #[inline]
    #[must_use]
    pub const fn max_branches(&self) -> usize {
        self.max_branches
    }

    /// Current number of active (routable) branches.
    #[inline]
    #[must_use]
    pub const fn n_active(&self) -> usize {
        self.n_active
    }

    /// Total number of allocated slots (active + removed, not counting unused
    /// capacity). `slots() - n_active()` == number of removed/reusable slots.
    #[inline]
    #[must_use]
    pub fn slots(&self) -> usize {
        self.branches.len()
    }

    /// True if a new branch can be spawned (active count below capacity).
    #[inline]
    #[must_use]
    pub fn can_spawn(&self) -> bool {
        self.n_active < self.max_branches
    }

    /// Get a reference to the branch at `id` (any lifecycle state).
    #[inline]
    #[must_use]
    pub fn get(&self, id: BranchId) -> Option<&CognitiveBranch<E>> {
        self.branches.get(id.0 as usize)
    }

    /// Get a mutable reference to the branch at `id` (any lifecycle state).
    #[inline]
    pub fn get_mut(&mut self, id: BranchId) -> Option<&mut CognitiveBranch<E>> {
        self.branches.get_mut(id.0 as usize)
    }

    /// Iterator over active (routable) branches. The router consumes this.
    #[inline]
    pub fn active_branches(&self) -> impl Iterator<Item = &CognitiveBranch<E>> {
        self.branches.iter().filter(|b| b.lifecycle.is_routable())
    }

    /// Spawn a new branch anchored on `spawn_anchor`. Returns `None` if the
    /// bank is at capacity (all slots active).
    ///
    /// Reuses a freed slot if available; otherwise appends a new slot.
    pub fn spawn(&mut self, spawn_anchor: Vec<f32>) -> Option<BranchId> {
        if !self.can_spawn() {
            return None;
        }

        self.n_active += 1;

        // Reuse a freed slot if available (O(1), no slot-array alloc).
        if let Some(slot) = self.free_slots.pop() {
            let fresh = CognitiveBranch::new(BranchId(slot), spawn_anchor);
            self.branches[slot as usize] = fresh;
            return Some(BranchId(slot));
        }

        // Otherwise append a new slot. The branches Vec was pre-allocated with
        // max_branches capacity, so push does not realloc.
        let slot = self.branches.len() as u32;
        debug_assert!(
            (slot as usize) < self.max_branches,
            "spawn invariants: n_active < max_branches but no free slot and branches.len() >= max_branches"
        );
        self.branches.push(CognitiveBranch::new(BranchId(slot), spawn_anchor));
        Some(BranchId(slot))
    }

    /// Mark the branch at `id` as `Removed`, clear its memory stores (preserving
    /// capacity for reuse), and push its slot onto the free-list.
    ///
    /// Returns `true` if the branch was active and is now removed; `false` if
    /// the id is out of range or the branch was already non-active.
    pub fn prune(&mut self, id: BranchId) -> bool {
        let Some(branch) = self.branches.get_mut(id.0 as usize) else {
            return false;
        };
        if !branch.lifecycle.is_routable() {
            return false;
        }

        branch.lifecycle = BranchLifecycle::Removed;
        branch.episodic.clear();
        branch.procedural.clear();
        branch.failures.clear();
        branch.token_signature.clear();
        branch.spawn_anchor.clear();
        self.n_active -= 1;
        self.free_slots.push(id.0);
        true
    }

    /// Merge branch `source` into branch `target`: concatenate episodic /
    /// procedural / failures, sum stats, compute a normalized merged
    /// spawn-anchor, union token signatures, then prune `source`.
    ///
    /// Returns `Some(target_id)` on success; `None` if either id is out of
    /// range, the two ids are identical, or either branch is non-active.
    ///
    /// **Anchor merge**: element-wise sum of the two anchor vectors, then
    /// L2-normalized. If the two anchors have different dimensionality, the
    /// shorter is zero-padded. This is the "ors anchors into normalized sum"
    /// from Plan 329 T1.3.
    ///
    /// **Stats merge**: `n_writes` and `n_reads` are summed; `avg_reward` is
    /// the write-count-weighted average; `last_touch_tick` is the max.
    ///
    /// **Allocation**: this method allocates a fresh `Vec` for the merged
    /// anchor and merged token signature. Merge is a cold-path lifecycle
    /// operation (periodic sweep), not a hot-path read/write.
    pub fn merge(&mut self, target: BranchId, source: BranchId) -> Option<BranchId> {
        let ti = target.0 as usize;
        let si = source.0 as usize;

        // Validate ids.
        if ti >= self.branches.len() || si >= self.branches.len() || ti == si {
            return None;
        }

        // Validate both are active (check before split to avoid aliasing issues).
        if !self.branches[ti].lifecycle.is_routable()
            || !self.branches[si].lifecycle.is_routable()
        {
            return None;
        }

        // Compute merged anchor from both (read-only), then write to target.
        let dim = self.branches[ti]
            .spawn_anchor
            .len()
            .max(self.branches[si].spawn_anchor.len());
        let mut merged_anchor = Vec::with_capacity(dim);
        let mut norm_sq = 0.0f32;
        for i in 0..dim {
            let a = self.branches[ti].spawn_anchor.get(i).copied().unwrap_or(0.0);
            let b = self.branches[si].spawn_anchor.get(i).copied().unwrap_or(0.0);
            let v = a + b;
            norm_sq += v * v;
            merged_anchor.push(v);
        }
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for v in &mut merged_anchor {
                *v *= inv_norm;
            }
        }

        // Compute merged token signature (sorted union) from both.
        let merged_tokens = sorted_union(
            &self.branches[ti].token_signature,
            &self.branches[si].token_signature,
        );

        // Read source stats for weighted-average computation.
        let source_n_writes = self.branches[si].stats.n_writes;
        let source_avg_reward = self.branches[si].stats.avg_reward;
        let source_n_reads = self.branches[si].stats.n_reads;
        let source_last_touch = self.branches[si].stats.last_touch_tick;

        // Get mutable access to both slots via split_at_mut.
        let (tgt, src) = if ti < si {
            let (left, right) = self.branches.split_at_mut(si);
            (&mut left[ti], &mut right[0])
        } else {
            // ti > si (equal is rejected above)
            let (left, right) = self.branches.split_at_mut(ti);
            (&mut right[0], &mut left[si])
        };

        // Move memory from source into target (append reuses target capacity).
        tgt.episodic.append(&mut src.episodic);
        tgt.procedural.append(&mut src.procedural);
        tgt.failures.append(&mut src.failures);

        // Merge stats: sum counters, write-weighted avg reward, max tick.
        let total_writes = tgt.stats.n_writes as f32 + source_n_writes as f32;
        if total_writes > 0.0 {
            tgt.stats.avg_reward = (tgt.stats.avg_reward * tgt.stats.n_writes as f32
                + source_avg_reward * source_n_writes as f32)
                / total_writes;
        }
        tgt.stats.n_writes = tgt.stats.n_writes.saturating_add(source_n_writes);
        tgt.stats.n_reads = tgt.stats.n_reads.saturating_add(source_n_reads);
        tgt.stats.last_touch_tick = tgt.stats.last_touch_tick.max(source_last_touch);

        // Apply merged anchor + tokens.
        tgt.spawn_anchor = merged_anchor;
        tgt.token_signature = merged_tokens;

        // Drop the &mut refs before calling prune (which takes &mut self).
        // The refs are no longer used after this point; the explicit
        // reassignment to _ is a no-op hint to the reader that the borrows
        // end here. The actual borrow scope ends at the call to prune below
        // because Rust NLL ends the borrow at last-use.
        let _ = (tgt, src);

        // Prune the source slot.
        self.prune(source);
        Some(target)
    }

    /// Run a merge sweep: for every pair of active branches whose
    /// `spawn_anchor` cosine similarity exceeds `cos_threshold`, merge the
    /// lower-utility one into the higher-utility one.
    ///
    /// Utility = `stats.n_writes + stats.n_reads`. Returns the number of
    /// merges performed. This is the cold-path lifecycle sweep that collapses
    /// redundant branches (RIZZ §"branch lifecycle").
    ///
    /// O(n_active²) — only call on a periodic sweep cadence, not per-tick.
    pub fn merge_sweep(&mut self, cos_threshold: f32) -> usize {
        let active_ids: Vec<BranchId> = self
            .branches
            .iter()
            .filter(|b| b.lifecycle.is_routable())
            .map(|b| b.id)
            .collect();

        let mut n_merges = 0;
        let mut merged: Vec<BranchId> = Vec::new();

        for &i in &active_ids {
            if merged.contains(&i) {
                continue;
            }
            let i_util = self
                .get(i)
                .map(|b| b.stats.n_writes as u64 + b.stats.n_reads as u64)
                .unwrap_or(0);

            for &j in &active_ids {
                if i == j || merged.contains(&j) {
                    continue;
                }
                // Both must still be active (a prior merge may have pruned one).
                let (Some(bi), Some(bj)) = (self.get(i), self.get(j)) else {
                    continue;
                };
                if !bi.lifecycle.is_routable() || !bj.lifecycle.is_routable() {
                    continue;
                }

                let cos = dot(&bi.spawn_anchor, &bj.spawn_anchor);
                if cos < cos_threshold {
                    continue;
                }

                let j_util = bj.stats.n_writes as u64 + bj.stats.n_reads as u64;
                let (target, source) = if i_util >= j_util { (i, j) } else { (j, i) };

                if self.merge(target, source).is_some() {
                    n_merges += 1;
                    merged.push(source);
                }
                break; // i has merged once; move to the next i
            }
        }

        n_merges
    }

    /// Run a prune sweep: prune every active branch whose stats are stale
    /// (no touch within `stale_window` ticks of `now`) AND whose memory is
    /// below `min_examples` entries. Returns the number of prunes.
    ///
    /// This is the RIZZ §"branch lifecycle" prune half: stale low-volume
    /// branches are reclaimed so their slots can be reused for novel inputs.
    pub fn prune_sweep(&mut self, now: u64, stale_window: u64, min_examples: usize) -> usize {
        let stale_ids: Vec<BranchId> = self
            .branches
            .iter()
            .filter(|b| {
                b.lifecycle.is_routable()
                    && b.stats.is_stale(now, stale_window)
                    && b.len() < min_examples
            })
            .map(|b| b.id)
            .collect();

        let n = stale_ids.len();
        for id in stale_ids {
            self.prune(id);
        }
        n
    }
}

impl<E: Clone + Default> BranchBank<E> {
    /// Record a verifier-approved episodic write into branch `id`.
    ///
    /// Updates the branch's episodic store, stats, and last-touch tick.
    /// Returns `true` on success; `false` if the branch is missing or inactive.
    pub fn write_episodic(
        &mut self,
        id: BranchId,
        embedding: Vec<f32>,
        payload: E,
        reward: f32,
        scope: Option<u64>,
        tick: u64,
    ) -> bool {
        let Some(branch) = self.branches.get_mut(id.0 as usize) else {
            return false;
        };
        if !branch.lifecycle.is_routable() {
            return false;
        }
        branch.episodic.push(crate::branching::types::EpisodicEntry {
            embedding,
            payload,
            reward,
            scope,
            tick,
        });
        branch.stats.record_write(reward, tick);
        true
    }
}

impl<E: Clone> core::fmt::Debug for BranchBank<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BranchBank")
            .field("max_branches", &self.max_branches)
            .field("n_active", &self.n_active)
            .field("slots", &self.branches.len())
            .field("free_slots", &self.free_slots.len())
            .finish()
    }
}

// Manual `Clone` (not derived) because the struct has an `E: Clone` bound on
// the type parameter, which `#[derive(Clone)]` would express as
// `where E: Clone` (correct here) but the manual form is more explicit and
// matches the existing manual `Debug` impl style. Required by downstream
// consumers (riir-ai Plan 338 `NpcCognitiveBranches::clone`) that need to
// deep-copy an entire NPC's cognitive state for A/B comparison or
// forked-counterfactual evaluation.
impl<E: Clone> Clone for BranchBank<E> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            branches: self.branches.clone(),
            free_slots: self.free_slots.clone(),
            max_branches: self.max_branches,
            n_active: self.n_active,
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Dot product of two f32 slices. Zero-alloc, auto-vectorizable inner loop.
/// Mismatched lengths use the shorter (caller should pre-normalize dimensions).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Sorted-union of two sorted, deduplicated `u64` slices. O(|a| + |b|).
/// Returns a new sorted, deduplicated Vec.
fn sorted_union(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
        }
    }
    while i < a.len() {
        result.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        result.push(b[j]);
        j += 1;
    }
    result
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bank_is_empty() {
        let bank: BranchBank<()> = BranchBank::new(8);
        assert_eq!(bank.max_branches(), 8);
        assert_eq!(bank.n_active(), 0);
        assert_eq!(bank.slots(), 0);
        assert!(bank.can_spawn());
        assert_eq!(bank.active_branches().count(), 0);
    }

    #[test]
    fn spawn_returns_distinct_ids() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        let a = bank.spawn(vec![1.0, 0.0]).unwrap();
        let b = bank.spawn(vec![0.0, 1.0]).unwrap();
        let c = bank.spawn(vec![0.0, 0.0]).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        assert_eq!(bank.n_active(), 3);
        assert_eq!(bank.slots(), 3);
    }

    #[test]
    fn spawn_returns_none_at_capacity() {
        let mut bank: BranchBank<()> = BranchBank::new(2);
        assert!(bank.spawn(vec![1.0]).is_some());
        assert!(bank.spawn(vec![1.0]).is_some());
        assert!(bank.spawn(vec![1.0]).is_none()); // at capacity
        assert_eq!(bank.n_active(), 2);
        assert!(!bank.can_spawn());
    }

    #[test]
    fn get_returns_branch_by_id() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        let id = bank.spawn(vec![1.0, 0.0, 0.0]).unwrap();
        let branch = bank.get(id).unwrap();
        assert_eq!(branch.id, id);
        assert_eq!(branch.spawn_anchor, vec![1.0, 0.0, 0.0]);
        assert!(branch.lifecycle.is_routable());
    }

    #[test]
    fn get_returns_none_for_invalid_id() {
        let bank: BranchBank<()> = BranchBank::new(4);
        assert!(bank.get(BranchId::new(0)).is_none());
        assert!(bank.get(BranchId::new(99)).is_none());
    }

    #[test]
    fn prune_frees_slot_for_reuse() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        let a = bank.spawn(vec![1.0]).unwrap();
        let _b = bank.spawn(vec![2.0]).unwrap();
        assert_eq!(bank.n_active(), 2);

        assert!(bank.prune(a));
        assert_eq!(bank.n_active(), 1);
        assert!(!bank.get(a).unwrap().lifecycle.is_routable());

        // Spawning should reuse slot a.
        let c = bank.spawn(vec![3.0]).unwrap();
        assert_eq!(c, a); // slot reused
        assert_eq!(bank.n_active(), 2);
        assert_eq!(bank.slots(), 2); // no new slot allocated
        assert_eq!(bank.get(c).unwrap().spawn_anchor, vec![3.0]);
    }

    #[test]
    fn prune_twice_is_noop() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        let a = bank.spawn(vec![1.0]).unwrap();
        assert!(bank.prune(a));
        assert!(!bank.prune(a)); // already removed
        assert_eq!(bank.n_active(), 0);
    }

    #[test]
    fn prune_invalid_id_is_noop() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        bank.spawn(vec![1.0]);
        assert!(!bank.prune(BranchId::new(99)));
        assert_eq!(bank.n_active(), 1);
    }

    #[test]
    fn merge_concatenates_memory_and_unions_anchor() {
        let mut bank: BranchBank<&'static str> = BranchBank::new(8);

        // Two branches with orthogonal anchors.
        let a = bank.spawn(vec![1.0, 0.0]).unwrap();
        let b = bank.spawn(vec![0.0, 1.0]).unwrap();
        bank.write_episodic(a, vec![1.0], "a1", 0.8, None, 10);
        bank.write_episodic(b, vec![0.0], "b1", 0.6, None, 20);
        assert_eq!(bank.n_active(), 2);

        // Merge b into a.
        let merged = bank.merge(a, b).unwrap();
        assert_eq!(merged, a);
        assert_eq!(bank.n_active(), 1);

        // Target should have both episodic entries.
        let target = bank.get(a).unwrap();
        assert_eq!(target.episodic.len(), 2);
        assert_eq!(target.episodic[0].payload, "a1");
        assert_eq!(target.episodic[1].payload, "b1");

        // Stats: summed writes, write-weighted avg reward.
        assert_eq!(target.stats.n_writes, 2);
        // avg = (0.8*1 + 0.6*1) / 2 = 0.7
        assert!((target.stats.avg_reward - 0.7).abs() < 1e-6);

        // Anchor: normalized sum of [1,0] + [0,1] = [1,1] / sqrt(2).
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        assert!((target.spawn_anchor[0] - inv_sqrt2).abs() < 1e-6);
        assert!((target.spawn_anchor[1] - inv_sqrt2).abs() < 1e-6);

        // Source should be pruned.
        assert!(bank.get(b).is_some()); // slot still exists
        assert!(!bank.get(b).unwrap().lifecycle.is_routable());
    }

    #[test]
    fn merge_rejects_identical_ids() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        let a = bank.spawn(vec![1.0]).unwrap();
        assert!(bank.merge(a, a).is_none());
    }

    #[test]
    fn merge_rejects_inactive_branch() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        let a = bank.spawn(vec![1.0]).unwrap();
        let b = bank.spawn(vec![1.0]).unwrap();
        bank.prune(b);
        assert!(bank.merge(a, b).is_none());
    }

    #[test]
    fn merge_unions_token_signatures() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let a = bank.spawn(vec![1.0]).unwrap();
        let b = bank.spawn(vec![1.0]).unwrap();
        bank.get_mut(a).unwrap().push_token(10);
        bank.get_mut(a).unwrap().push_token(30);
        bank.get_mut(b).unwrap().push_token(20);
        bank.get_mut(b).unwrap().push_token(30); // dup with a

        bank.merge(a, b).unwrap();

        let target = bank.get(a).unwrap();
        assert_eq!(target.token_signature, vec![10, 20, 30]);
    }

    #[test]
    fn active_branches_skips_removed() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let a = bank.spawn(vec![1.0]).unwrap();
        let _b = bank.spawn(vec![2.0]).unwrap();
        let c = bank.spawn(vec![3.0]).unwrap();
        bank.prune(_b);

        let active: Vec<BranchId> = bank.active_branches().map(|b| b.id).collect();
        assert_eq!(active, vec![a, c]);
    }

    #[test]
    fn write_episodic_updates_stats() {
        let mut bank: BranchBank<&'static str> = BranchBank::new(4);
        let id = bank.spawn(vec![1.0]).unwrap();
        bank.write_episodic(id, vec![1.0], "x", 0.5, Some(7), 100);
        bank.write_episodic(id, vec![1.0], "y", 1.0, None, 200);

        let branch = bank.get(id).unwrap();
        assert_eq!(branch.episodic.len(), 2);
        assert_eq!(branch.stats.n_writes, 2);
        assert_eq!(branch.stats.last_touch_tick, 200);
        assert!((branch.stats.avg_reward - 0.75).abs() < 1e-6);
    }

    #[test]
    fn write_episodic_rejects_inactive_branch() {
        let mut bank: BranchBank<&'static str> = BranchBank::new(4);
        let id = bank.spawn(vec![1.0]).unwrap();
        bank.prune(id);
        assert!(!bank.write_episodic(id, vec![], "x", 0.5, None, 1));
    }

    #[test]
    fn prune_sweep_removes_stale_low_volume_branches() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let fresh = bank.spawn(vec![1.0]).unwrap();
        let stale = bank.spawn(vec![1.0]).unwrap();
        // Touch the fresh branch recently (within the stale window).
        bank.get_mut(fresh).unwrap().stats.last_touch_tick = 180;
        // Leave the stale branch at tick 0.
        // At now=200, window=50: fresh was touched at 180 → 20 ticks ago (not stale);
        // stale was touched at 0 → 200 ticks ago (stale). min_examples=5.
        let n = bank.prune_sweep(200, 50, 5);
        assert_eq!(n, 1);
        assert!(bank.get(fresh).unwrap().lifecycle.is_routable());
        assert!(!bank.get(stale).unwrap().lifecycle.is_routable());
    }

    #[test]
    fn merge_sweep_collapses_redundant_branches() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        // Two near-identical anchors → cosine ≈ 1.0.
        let a = bank.spawn(vec![1.0, 0.01]).unwrap();
        let _b = bank.spawn(vec![1.0, 0.0]).unwrap();
        let c = bank.spawn(vec![0.0, 1.0]).unwrap(); // orthogonal to a, b

        // Give a more writes so it wins the utility tiebreak.
        bank.get_mut(a).unwrap().stats.n_writes = 10;

        let n = bank.merge_sweep(0.99);
        assert_eq!(n, 1); // a + b merged; c untouched (orthogonal)
        assert!(bank.get(a).unwrap().lifecycle.is_routable());
        assert!(bank.get(c).unwrap().lifecycle.is_routable());
        // Exactly one of a/b is still active (a won the utility tiebreak).
        assert_eq!(bank.n_active(), 2);
    }

    #[test]
    fn sorted_union_deduplicates() {
        // Inputs MUST be sorted + deduplicated (function contract).
        assert_eq!(sorted_union(&[1, 3, 5], &[2, 3, 4]), vec![1, 2, 3, 4, 5]);
        assert_eq!(sorted_union(&[], &[1, 2]), vec![1, 2]);
        assert_eq!(sorted_union(&[1, 2], &[]), vec![1, 2]);
        assert!(sorted_union(&[], &[]).is_empty());
        // Overlap in inputs produces dedup in output.
        assert_eq!(sorted_union(&[1, 2, 3], &[2, 3, 4]), vec![1, 2, 3, 4]);
    }

    #[test]
    fn debug_format() {
        let mut bank: BranchBank<()> = BranchBank::new(4);
        bank.spawn(vec![1.0]);
        let s = format!("{:?}", bank);
        assert!(s.contains("max_branches: 4"));
        assert!(s.contains("n_active: 1"));
    }
}
