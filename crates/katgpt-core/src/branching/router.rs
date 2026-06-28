//! `BranchRouter` — dot-product snap routing with Jaccard fallback
//! (Plan 329 T1.4).
//!
//! Routes a query embedding to the best-matching active branch by:
//! 1. **Dot-product snap** (primary): `s(b) = dot(query, b.spawn_anchor)`. If
//!    the max score ≥ `tau_snap`, snap to the best branch (`Reuse`).
//! 2. **Jaccard fallback** (secondary): if no dot-product snap and the caller
//!    supplied query tokens, compute Jaccard similarity against each branch's
//!    `token_signature`. If any branch's Jaccard ≥ `tau_jaccard`, snap to the
//!    best-Jaccard branch (`Reuse`, lower confidence).
//! 3. **Spawn** (fallback): if no snap and capacity remains → `Spawn`.
//! 4. **Frozen**: no snap and no capacity → `Frozen` (write rejected).
//!
//! # Hot path
//!
//! The dot-product scan is a branch-free max-reduction over the active branch
//! array. Zero allocation. The `RouteResult` is a 2-word stack struct.
//!
//! # Latent reframing (Research 310 §2.2)
//!
//! RIZZ uses an LLM judge to propose `(function, application)` labels, then
//! snaps by cosine on label embeddings. Our reframing: the "label" IS the
//! branch's `spawn_anchor` direction in latent space — no LLM judge needed.
//! The caller pre-normalizes embeddings (their responsibility); the router
//! only does dot products.

use crate::branching::bank::BranchBank;
use crate::branching::types::BranchId;

/// Default dot-product snap threshold (RIZZ §"hierarchical routing" uses
/// cosine 0.92; with pre-normalized embeddings dot == cosine).
pub const DEFAULT_TAU_SNAP: f32 = 0.92;

/// Default Jaccard fallback threshold (RIZZ §"hierarchical routing" uses 0.40).
pub const DEFAULT_TAU_JACCARD: f32 = 0.40;

/// Default spawn threshold: spawn when the max dot-product score is strictly
/// below this. `0.0` means "spawn only when no positive dot-product match".
pub const DEFAULT_TAU_SPAWN: f32 = 0.0;

/// How the router resolved a route query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RouteMode {
    /// Snapped to an existing branch (dot-product or Jaccard match).
    Reuse = 0,
    /// No match; a new branch should be spawned (capacity available).
    Spawn = 1,
    /// No match and no capacity; the write is rejected.
    Frozen = 2,
}

/// Result of a route query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteResult {
    /// The matched branch id, if `mode == Reuse`. `None` for `Spawn` / `Frozen`.
    pub branch: Option<BranchId>,
    /// How the route was resolved.
    pub mode: RouteMode,
}

impl RouteResult {
    /// Construct a `Reuse` result.
    #[inline]
    #[must_use]
    pub const fn reuse(branch: BranchId) -> Self {
        Self {
            branch: Some(branch),
            mode: RouteMode::Reuse,
        }
    }

    /// Construct a `Spawn` result.
    #[inline]
    #[must_use]
    pub const fn spawn() -> Self {
        Self {
            branch: None,
            mode: RouteMode::Spawn,
        }
    }

    /// Construct a `Frozen` result.
    #[inline]
    #[must_use]
    pub const fn frozen() -> Self {
        Self {
            branch: None,
            mode: RouteMode::Frozen,
        }
    }
}

/// Dot-product snap router with optional Jaccard fallback.
///
/// Construct with [`BranchRouter::default`] for RIZZ-paper-aligned thresholds
/// (τ_snap=0.92, τ_jaccard=0.40, τ_spawn=0.0), or with [`BranchRouter::new`]
/// for custom thresholds.
#[derive(Clone, Copy, Debug)]
pub struct BranchRouter {
    /// Dot-product score ≥ this → snap to best branch.
    pub tau_snap: f32,
    /// Jaccard similarity ≥ this → snap to best-Jaccard branch (fallback).
    pub tau_jaccard: f32,
    /// Max dot-product score < this → consider spawn (if no snap).
    pub tau_spawn: f32,
}

impl Default for BranchRouter {
    #[inline]
    fn default() -> Self {
        Self {
            tau_snap: DEFAULT_TAU_SNAP,
            tau_jaccard: DEFAULT_TAU_JACCARD,
            tau_spawn: DEFAULT_TAU_SPAWN,
        }
    }
}

impl BranchRouter {
    /// Construct with custom thresholds.
    #[inline]
    #[must_use]
    pub const fn new(tau_snap: f32, tau_jaccard: f32, tau_spawn: f32) -> Self {
        Self {
            tau_snap,
            tau_jaccard,
            tau_spawn,
        }
    }

    /// Route a query embedding to a branch (dot-product snap only).
    ///
    /// This is the primary hot-path entry point. The caller pre-normalizes the
    /// query embedding so dot-product == cosine similarity. If the max
    /// dot-product ≥ `tau_snap`, snaps to the best branch; otherwise falls
    /// through to spawn/frozen.
    ///
    /// Zero allocation on the hot path.
    #[inline]
    pub fn route<E: Clone>(
        &self,
        query_embedding: &[f32],
        bank: &BranchBank<E>,
    ) -> RouteResult {
        // Primary: dot-product snap.
        if let Some(id) = self.snap_dot(query_embedding, bank) {
            return RouteResult::reuse(id);
        }

        // No snap. Spawn if capacity remains, else frozen.
        if bank.can_spawn() {
            RouteResult::spawn()
        } else {
            RouteResult::frozen()
        }
    }

    /// Route a query embedding AND query tokens (dot-product snap + Jaccard
    /// fallback).
    ///
    /// Use this when the caller can supply hash tokens (e.g., from an Engram
    /// query). The Jaccard fallback runs only when the primary dot-product snap
    /// fails AND both the query and at least one branch have non-empty token
    /// signatures.
    ///
    /// `query_tokens` MUST be sorted + deduplicated by the caller (same contract
    /// as `CognitiveBranch::token_signature`).
    ///
    /// Zero allocation on the hot path (the Jaccard merge-walk is scratch-free).
    #[inline]
    pub fn route_with_tokens<E: Clone>(
        &self,
        query_embedding: &[f32],
        query_tokens: &[u64],
        bank: &BranchBank<E>,
    ) -> RouteResult {
        // Primary: dot-product snap.
        if let Some(id) = self.snap_dot(query_embedding, bank) {
            return RouteResult::reuse(id);
        }

        // Secondary: Jaccard fallback (only if query has tokens).
        if !query_tokens.is_empty()
            && let Some(id) = self.snap_jaccard(query_tokens, bank) {
                return RouteResult::reuse(id);
            }

        // No snap. Spawn if capacity remains, else frozen.
        if bank.can_spawn() {
            RouteResult::spawn()
        } else {
            RouteResult::frozen()
        }
    }

    /// Dot-product snap: find the active branch with the highest
    /// `dot(query, spawn_anchor)`. Returns `Some(best_id)` if max ≥ `tau_snap`.
    ///
    /// Branch-free max-reduction over the active branch iterator. Zero alloc.
    #[inline]
    fn snap_dot<E: Clone>(
        &self,
        query_embedding: &[f32],
        bank: &BranchBank<E>,
    ) -> Option<BranchId> {
        let mut best_id = None;
        let mut best_score = f32::NEG_INFINITY;

        for branch in bank.active_branches() {
            let score = dot(query_embedding, &branch.spawn_anchor);
            if score > best_score {
                best_score = score;
                best_id = Some(branch.id);
            }
        }

        // Snap only if the best score clears tau_snap.
        if best_score >= self.tau_snap {
            best_id
        } else {
            None
        }
    }

    /// Jaccard fallback: find the active branch with the highest token-overlap
    /// Jaccard similarity. Returns `Some(best_id)` if max ≥ `tau_jaccard`.
    ///
    /// Skips branches with empty `token_signature`. Zero alloc.
    #[inline]
    fn snap_jaccard<E: Clone>(
        &self,
        query_tokens: &[u64],
        bank: &BranchBank<E>,
    ) -> Option<BranchId> {
        let mut best_id = None;
        let mut best_jaccard = self.tau_jaccard; // start at threshold

        for branch in bank.active_branches() {
            if branch.token_signature.is_empty() {
                continue;
            }
            let j = jaccard(query_tokens, &branch.token_signature);
            if j > best_jaccard {
                best_jaccard = j;
                best_id = Some(branch.id);
            }
        }

        best_id
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Dot product of two f32 slices. Zero-alloc, auto-vectorizable inner loop.
/// Uses the shorter length when dimensions mismatch (caller should
/// pre-normalize dimensions).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Jaccard similarity of two sorted, deduplicated `u64` slices.
/// Returns `0.0` if either slice is empty. O(|a| + |b|) merge-walk, zero alloc.
#[inline]
fn jaccard(a: &[u64], b: &[u64]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let mut intersection = 0u32;
    let mut union = 0u32;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Equal => {
                intersection += 1;
                union += 1;
                i += 1;
                j += 1;
            }
            core::cmp::Ordering::Less => {
                union += 1;
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                union += 1;
                j += 1;
            }
        }
    }
    // Remaining tail elements all go to the union.
    union += (a.len() - i) as u32;
    union += (b.len() - j) as u32;

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bank_with_branches(anchors: &[Vec<f32>]) -> BranchBank<()> {
        let mut bank = BranchBank::new(anchors.len() + 2);
        for a in anchors {
            bank.spawn(a.clone()).unwrap();
        }
        bank
    }

    #[test]
    fn route_empty_bank_returns_spawn() {
        let bank: BranchBank<()> = BranchBank::new(4);
        let router = BranchRouter::default();
        let result = router.route(&[1.0, 0.0], &bank);
        assert_eq!(result.mode, RouteMode::Spawn);
        assert!(result.branch.is_none());
    }

    #[test]
    fn route_returns_frozen_at_capacity() {
        let mut bank: BranchBank<()> = BranchBank::new(1);
        bank.spawn(vec![1.0, 0.0]).unwrap();
        // Query is orthogonal → no snap. Bank full → Frozen.
        let router = BranchRouter::default();
        let result = router.route(&[0.0, 1.0], &bank);
        assert_eq!(result.mode, RouteMode::Frozen);
    }

    #[test]
    fn route_snaps_on_high_dot_product() {
        let bank = make_bank_with_branches(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let router = BranchRouter::default();
        // Query matches branch 0 almost perfectly.
        let result = router.route(&[0.99, 0.01], &bank);
        assert_eq!(result.mode, RouteMode::Reuse);
        assert_eq!(result.branch, Some(BranchId::new(0)));
    }

    #[test]
    fn route_picks_best_match() {
        let bank = make_bank_with_branches(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let router = BranchRouter::default();
        // Query matches branch 1 better.
        let result = router.route(&[0.01, 0.99], &bank);
        assert_eq!(result.mode, RouteMode::Reuse);
        assert_eq!(result.branch, Some(BranchId::new(1)));
    }

    #[test]
    fn route_returns_spawn_when_below_snap_threshold() {
        let bank = make_bank_with_branches(&[vec![1.0, 0.0]]);
        let router = BranchRouter::default();
        // Query is at 45 degrees → cosine ≈ 0.707 < 0.92 tau_snap.
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        let result = router.route(&[inv_sqrt2, inv_sqrt2], &bank);
        assert_eq!(result.mode, RouteMode::Spawn);
    }

    #[test]
    fn route_with_tokens_uses_jaccard_fallback() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![1.0, 0.0]).unwrap();
        bank.get_mut(id).unwrap().token_signature = vec![10, 20, 30];

        let router = BranchRouter::default();
        // Query embedding doesn't snap (orthogonal), but query tokens overlap.
        let result = router.route_with_tokens(&[0.0, 1.0], &[10, 20, 40], &bank);
        // Jaccard = |{10,20}| / |{10,20,30,40}| = 2/4 = 0.5 ≥ 0.40 → snap.
        assert_eq!(result.mode, RouteMode::Reuse);
        assert_eq!(result.branch, Some(id));
    }

    #[test]
    fn route_with_tokens_skips_jaccard_when_dot_snaps() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![1.0, 0.0]).unwrap();
        bank.get_mut(id).unwrap().token_signature = vec![999];

        let router = BranchRouter::default();
        // Query snaps via dot-product (no need for Jaccard).
        let result = router.route_with_tokens(&[0.99, 0.01], &[1, 2, 3], &bank);
        assert_eq!(result.mode, RouteMode::Reuse);
        assert_eq!(result.branch, Some(id));
    }

    #[test]
    fn route_with_tokens_empty_query_tokens_skips_jaccard() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![1.0, 0.0]).unwrap();
        bank.get_mut(id).unwrap().token_signature = vec![10, 20];

        let router = BranchRouter::default();
        // No query tokens → Jaccard skipped. No dot snap → spawn.
        let result = router.route_with_tokens(&[0.0, 1.0], &[], &bank);
        assert_eq!(result.mode, RouteMode::Spawn);
    }

    #[test]
    fn jaccard_identical_sets_is_one() {
        assert!((jaccard(&[1, 2, 3], &[1, 2, 3]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_disjoint_sets_is_zero() {
        assert!((jaccard(&[1, 2], &[3, 4])).abs() < 1e-6);
    }

    #[test]
    fn jaccard_half_overlap() {
        // {1,2} ∩ {2,3,4} = {2} (size 1); union = {1,2,3,4} (size 4).
        // jaccard = 1/4 = 0.25.
        assert!((jaccard(&[1, 2], &[2, 3, 4]) - 1.0 / 4.0).abs() < 1e-6);
        // {1,2,3} ∩ {2,3,4,5} = {2,3} (size 2); union = {1,2,3,4,5} (size 5).
        // jaccard = 2/5 = 0.4.
        assert!((jaccard(&[1, 2, 3], &[2, 3, 4, 5]) - 2.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_empty_returns_zero() {
        assert!((jaccard(&[], &[1, 2])).abs() < 1e-6);
        assert!((jaccard(&[1, 2], &[])).abs() < 1e-6);
        assert!((jaccard(&[], &[])).abs() < 1e-6);
    }

    #[test]
    fn route_result_constructors() {
        let r = RouteResult::reuse(BranchId::new(3));
        assert_eq!(r.mode, RouteMode::Reuse);
        assert_eq!(r.branch, Some(BranchId::new(3)));

        let s = RouteResult::spawn();
        assert_eq!(s.mode, RouteMode::Spawn);
        assert!(s.branch.is_none());

        let f = RouteResult::frozen();
        assert_eq!(f.mode, RouteMode::Frozen);
        assert!(f.branch.is_none());
    }

    #[test]
    fn router_custom_thresholds() {
        let router = BranchRouter::new(0.5, 0.3, 0.0);
        let bank = make_bank_with_branches(&[vec![1.0, 0.0]]);
        // cosine 0.707 ≥ 0.5 (custom tau_snap) → snap.
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        let result = router.route(&[inv_sqrt2, inv_sqrt2], &bank);
        assert_eq!(result.mode, RouteMode::Reuse);
    }
}
