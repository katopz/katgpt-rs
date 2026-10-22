//! Speculative-decoding types for katgpt-rs root (Plan 008 Step 5 shim).
//!
//! The pure-substrate half of this module moved to `katgpt_core::speculative`
//! (data types, configs, algorithms, trait implementations). This file is now
//! a thin re-export shim that preserves all existing import paths
//! (`crate::speculative::types::TreeNode`, `...::DraftResult`, etc.) plus the
//! root-only composition types that depend on `katgpt-transformer` or other
//! root-only modules:
//!
//! - `SpeculativeContext` — re-exported from `katgpt_forward` (Plan 393, moved
//!   there because it composes `ForwardContext`)
//! - [`DDTreeBranchCache`] — needs `PagedKVCache` + `forward_paged`
//!   (from `katgpt-transformer`)
//! - `TesConfig` — needs `BanditStrategy` (from `crate::pruners::bandit`)
//! - `SelfSpecConfig` — needs `D2fDecodeConfig` + `DiffusionSampler`
//!   (from `crate::speculative::{d2f, diffusion_sampler}`)
//!
//! Everything else re-exports from `katgpt_core::speculative::types`.

use crate::transformer::{ForwardContext, PagedKVCache, TransformerWeights, forward_paged};
use crate::types::Config;

// ── Re-exported from katgpt-core (Plan 107 Phase 0 + Plan 008 Step 5) ──
// The substrate traits (ConstraintPruner, ScreeningPruner, DominoPruner,
// NoPruner, NoScreeningPruner, BinaryScreeningPruner) have lived in
// `katgpt_core::traits` since Plan 107 Phase 0. The substrate data types,
// configs, and algorithms joined them in Plan 008 Step 5
// (`katgpt_core::speculative::types`).
//
// (BanditStrategy / TesConfig moved to katgpt_pruners; no import needed here.)

pub use katgpt_core::traits::{
    BinaryScreeningPruner, CompletionHorizon, ConstraintPruner, DominoPruner, NoPruner,
    NoScreeningPruner, ScreeningPruner,
};

// Always-available speculative substrate types (no feature gate in katgpt-core).
pub use katgpt_core::speculative::types::{
    BlockScores, BudgetAdaptation, DecodeStrategy, DraftEvent, DraftResult, FlashPrefillConfig,
    PrefillMode, RejectionReason, ScoreReduction, SdeConfig, TreeNode, TreePath,
};

// Feature-gated re-exports — MUST mirror katgpt-core's gates.
//
// `EarlyStopGate` / `SpecCostSnapshot` / `StabilitySnapshot` are gated in
// katgpt-core behind `elf_sde` / `spec_cost_model` / `stability_metrics`
// respectively. `RoutingOverlapSnapshot` is gated behind `domain_latent`
// (which katgpt-core turns on transitively via `octree_ctc → sense_composition`,
// so it's almost always available — but we still gate the re-export on root's
// `domain_latent` so root's API surface matches root's feature flags).
//
// See Issue 016: the prior unconditional re-export broke
// `cargo check --no-default-features` because the gated names don't exist in
// katgpt-core when their features are off.
#[cfg(feature = "elf_sde")]
pub use katgpt_core::speculative::types::EarlyStopGate;
#[cfg(feature = "domain_latent")]
pub use katgpt_core::speculative::types::RoutingOverlapSnapshot;
#[cfg(feature = "spec_cost_model")]
pub use katgpt_core::speculative::types::SpecCostSnapshot;
#[cfg(feature = "stability_metrics")]
pub use katgpt_core::speculative::types::StabilitySnapshot;

// NB: `katgpt_core::speculative::types` also exports feature-gated substrate
// types (`MarginalFusionConfig`, `KvRoutingConfig`, `PositionWeightedBudget`,
// `LdtPruneConfig`, `LDT_THETA_ELIM`, `ConflictDetector`, `EntropyConflictDetector`,
// `TesNode`, `TrajectoryCredit`). We don't re-export those here because root
// already re-exports some of them by other paths (e.g. root's `speculative::alpha`
// owns the LDT α-operator surface) and adding blanket re-exports would create
// duplicate-path ambiguity. Consumers that need them can `use katgpt_core::speculative::types::*`.
// The blanket `pub use katgpt_core::speculative::types::*` is intentionally omitted.

#[cfg(feature = "dflare_kv_routing")]
pub use katgpt_core::speculative::types::KvRoutingConfig;
#[cfg(feature = "dflare_fusion")]
pub use katgpt_core::speculative::types::MarginalFusionConfig;
#[cfg(feature = "dflare_progressive_budget")]
pub use katgpt_core::speculative::types::PositionWeightedBudget;
#[cfg(feature = "lattice_deduction")]
pub use katgpt_core::speculative::types::{
    ConflictDetector, EntropyConflictDetector, LDT_THETA_ELIM, LdtPruneConfig,
};
#[cfg(feature = "tes_loop")]
pub use katgpt_core::speculative::types::{TesNode, TrajectoryCredit};

// ── Pre-allocated Speculative Context (moved to katgpt-forward, Plan 393) ──
// `SpeculativeContext` lives in `katgpt_forward::speculative_context` because it
// composes `ForwardContext` (also in katgpt-forward). It cannot live in
// katgpt-speculative (the leaf below katgpt-forward) without creating a cycle.
// Re-exported here so all `crate::speculative::types::SpeculativeContext` and
// `crate::speculative::SpeculativeContext` import paths continue to resolve.
pub use katgpt_forward::SpeculativeContext;

// ── DDTree Branch Cache (composition — needs katgpt-transformer) ──

/// Paged KV cache wrapper for DDTree branch exploration.
/// Forks share prefix pages, only new pages allocate after fork point.
///
/// This enables best-first search in DDTree where multiple token branches
/// are explored from a shared prefix — copy-on-write avoids duplicating
/// the entire KV cache for each branch.
pub struct DDTreeBranchCache {
    paged: PagedKVCache,
    branch_count: usize,
    max_branches: usize,
}

impl DDTreeBranchCache {
    /// Create a new branch cache.
    /// `max_branches` determines how many concurrent sequences the paged cache supports.
    pub fn new(config: &Config, max_branches: usize) -> Self {
        Self {
            paged: PagedKVCache::new(config, max_branches),
            branch_count: 1, // sequence 0 = trunk
            max_branches,
        }
    }

    /// Fork from an existing sequence at the given position.
    /// Returns the new sequence index.
    /// Shared prefix pages are NOT copied — copy-on-write semantics.
    pub fn fork_branch(&mut self, from: usize, at_pos: usize) -> usize {
        if self.branch_count >= self.max_branches {
            return from; // budget exhausted, reuse source
        }
        let new_seq = self.paged.fork(from, at_pos);
        self.branch_count += 1;
        new_seq
    }

    /// Forward pass on a specific branch sequence.
    /// Returns logits slice via `forward_paged`.
    pub fn forward_branch<'a>(
        &mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        seq_idx: usize,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        forward_paged(ctx, weights, &mut self.paged, seq_idx, token, pos, config)
    }

    /// Rollback a branch to a given position, freeing exclusive pages.
    ///
    /// Keeps pages covering positions `[0..at_pos)` and truncates the rest.
    /// Pages exclusively owned by this branch (not shared with other branches)
    /// are returned to the free list. Shared prefix pages are preserved.
    pub fn rollback_branch(&mut self, seq_idx: usize, at_pos: usize) {
        self.paged.rollback(seq_idx, at_pos);
    }

    /// Fully discard a branch, freeing all its exclusive pages.
    ///
    /// Rolls back to position 0 and decrements `branch_count` if the branch
    /// is not the trunk (seq 0). The trunk cannot be discarded — use `reset()`
    /// to clear everything.
    pub fn discard_branch(&mut self, seq_idx: usize) {
        self.paged.rollback(seq_idx, 0);
        if seq_idx > 0 && self.branch_count > 1 {
            self.branch_count -= 1;
        }
    }

    /// Reset all branches, freeing pages back to pool.
    pub fn reset(&mut self) {
        self.paged.reset();
        self.branch_count = 1;
    }
}

// ── Self-Speculation Config (composition — needs root-only D2F types) ──

/// Config for D2F-drafter self-speculation mode.
/// Wraps D2F decode config + draft width for speculative step.
#[cfg(feature = "tri_mode")]
#[derive(Debug, Clone)]
pub struct SelfSpecConfig {
    /// Number of tokens per D2F draft block (default: 8).
    pub draft_width: usize,
    /// D2F decode parameters.
    pub d2f_config: crate::speculative::d2f::D2fDecodeConfig,
    /// Acceptance policy (Issue 587 / FLARE Eq 8/21/22 taxonomy).
    /// Default: `SoftmaxArgmax` — exact + cheapest. See
    /// `crate::speculative::d2f_verifier::DraftAcceptPolicy`.
    pub accept_policy: crate::speculative::d2f_verifier::DraftAcceptPolicy,
    /// Optional trained sampler for adaptive confidence (Plan 116 T3).
    /// When `Some`, uses per-position features to decide accept/reject instead
    /// of the fixed `tau_conf` threshold in the D2F denoising loop.
    pub sampler: Option<crate::speculative::diffusion_sampler::DiffusionSampler>,
}

#[cfg(feature = "tri_mode")]
impl Default for SelfSpecConfig {
    fn default() -> Self {
        Self {
            draft_width: 8,
            d2f_config: crate::speculative::d2f::D2fDecodeConfig::default(),
            accept_policy: crate::speculative::d2f_verifier::DraftAcceptPolicy::default(),
            sampler: None,
        }
    }
}

// ── SimpleTES Config (composition — needs root-only BanditStrategy) ──
// NB: `TesNode` + `TrajectoryCredit` (the pure-data / pure-algorithm half of
// SimpleTES) live in `katgpt_core::speculative::types`. `TesConfig` moved to
// `katgpt_pruners::tes_loop` (Plan 005) because its `bandit_strategy` field is
// typed as `BanditStrategy` from `katgpt_pruners::bandit`. Re-exported here so
// existing `crate::speculative::types::TesConfig` import paths still resolve.
#[cfg(feature = "tes_loop")]
pub use katgpt_pruners::tes_loop::TesConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::Rng;

    // DDTreeBranchCache tests stay here — they need `ForwardContext` +
    // `TransformerWeights` + `Rng` + `Config` (all root-only or
    // katgpt-transformer composition). The substrate tests for the moved
    // types (DraftEvent, RejectionReason, DecodeStrategy, EarlyStopGate,
    // dflare_*) now live in `katgpt_core::speculative::types::tests`.

    #[test]
    fn test_branch_cache_fork_branch() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        // Run trunk forward at pos 0..3
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        for pos in 0..4 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Fork at pos 2 — should return a new sequence index (not source)
        let branch = cache.fork_branch(0, 2);
        assert_ne!(
            branch, 0,
            "first fork should return different seq_idx than source"
        );

        // Fork again — should return another new unique sequence index
        let branch2 = cache.fork_branch(0, 2);
        assert_ne!(
            branch2, 0,
            "second fork should return different seq_idx than source"
        );
        assert_ne!(branch2, branch, "each fork should return unique seq_idx");
    }

    #[test]
    fn test_branch_cache_fork_branch_budget_exhausted() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 2); // max 2 branches

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk at pos 0
        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // First fork succeeds — returns a new seq_idx (not source)
        let b1 = cache.fork_branch(0, 0);
        assert_ne!(b1, 0, "fork should return new seq_idx");

        // Budget exhausted (branch_count == max_branches), returns source
        let b2 = cache.fork_branch(0, 0);
        assert_eq!(b2, 0, "budget exhausted should return source seq_idx");
    }

    #[test]
    fn test_branch_cache_forward_branch_logits() {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut cache = DDTreeBranchCache::new(&config, 4);
        let mut ctx = ForwardContext::new(&config);

        let logits = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);

        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} not finite: {l}");
        }
    }

    #[test]
    fn test_branch_cache_fork_shared_prefix_forward() {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut cache = DDTreeBranchCache::new(&config, 4);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2
        for pos in 0..3 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Fork at pos 2 — returns a new seq_idx (not source)
        let branch = cache.fork_branch(0, 2);
        assert_ne!(branch, 0, "fork should return new seq_idx");

        // Continue branch from pos 2 with different token
        let logits = cache.forward_branch(&mut ctx, &weights, branch, 5, 2, &config);
        assert_eq!(logits.len(), config.vocab_size);

        // All logits should be finite
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "branch logit {i} not finite: {l}");
        }
    }

    #[test]
    fn test_branch_cache_reset() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);
        let _ = cache.fork_branch(0, 0);

        assert_eq!(cache.branch_count, 2);

        cache.reset();

        assert_eq!(
            cache.branch_count, 1,
            "reset should restore branch_count to 1"
        );
    }

    #[test]
    fn test_branch_cache_rollback_branch_allows_forward_after() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2, 3
        for pos in 0..4 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Rollback trunk to pos 2
        cache.rollback_branch(0, 2);

        // Forward should still work after rollback — pages for pos 0..2 are intact
        let logits = cache.forward_branch(&mut ctx, &weights, 0, 5, 2, &config);
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "logit {i} after rollback should be finite: {l}"
            );
        }
    }

    #[test]
    fn test_branch_cache_discard_branch_decrements_count() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 8);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk at pos 0
        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // Fork a branch
        let branch = cache.fork_branch(0, 0);
        assert_ne!(branch, 0, "fork should return new seq_idx");
        assert_eq!(cache.branch_count, 2);

        // Forward on branch
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 5, 1, &config);

        // Discard branch
        cache.discard_branch(branch);
        assert_eq!(
            cache.branch_count, 1,
            "discard should decrement branch_count"
        );
    }

    #[test]
    fn test_branch_cache_discard_trunk_does_not_decrement() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        let _ = cache.forward_branch(&mut ctx, &weights, 0, 0, 0, &config);

        // Discard trunk (seq 0) should not decrement branch_count
        cache.discard_branch(0);
        assert_eq!(
            cache.branch_count, 1,
            "discarding trunk should not decrement branch_count"
        );
    }

    #[test]
    fn test_branch_cache_rollback_shared_pages_preserved() {
        let config = Config::draft();
        let mut cache = DDTreeBranchCache::new(&config, 4);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);

        // Trunk: pos 0, 1, 2
        for pos in 0..3 {
            let _ = cache.forward_branch(&mut ctx, &weights, 0, pos, pos, &config);
        }

        // Capture trunk state at pos 2 before branching
        let _trunk_logits = cache
            .forward_branch(&mut ctx, &weights, 0, 0, 3, &config)
            .to_vec();

        // Fork at pos 1
        let branch = cache.fork_branch(0, 1);
        assert_ne!(branch, 0, "fork should return new seq_idx");

        // Forward on branch to allocate exclusive pages
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 5, 1, &config);
        let _ = cache.forward_branch(&mut ctx, &weights, branch, 7, 2, &config);

        // Rollback branch to pos 1 — shared prefix pages should be preserved
        cache.rollback_branch(branch, 1);

        // Trunk should still work: shared pages (pos 0) were not freed
        let trunk_logits_after = cache
            .forward_branch(&mut ctx, &weights, 0, 1, 3, &config)
            .to_vec();
        for (i, &l) in trunk_logits_after.iter().enumerate() {
            assert!(
                l.is_finite(),
                "trunk logit {i} should be finite after branch rollback: {l}"
            );
        }

        // Forking again from trunk should succeed — shared pages are intact
        let branch2 = cache.fork_branch(0, 1);
        assert_ne!(branch2, 0, "fork after rollback should return new seq_idx");
        let logits2 = cache.forward_branch(&mut ctx, &weights, branch2, 3, 1, &config);
        assert_eq!(logits2.len(), config.vocab_size);
        for (i, &l) in logits2.iter().enumerate() {
            assert!(l.is_finite(), "branch2 logit {i} should be finite: {l}");
        }
    }
}
