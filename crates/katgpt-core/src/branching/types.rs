//! Type definitions for non-interference memory branches (Plan 329 T1.2).
//!
//! All types are `Clone + Debug`. Pod-compatible types (no `Vec`, no generic
//! payload) are `#[repr(C)]` for sync-friendly layout. The owning containers
//! (`Vec`-backed stores) allocate only at construction / write time, never on
//! the read-side hot path.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `BranchId` | **Raw** | YES (deterministic dense index) |
//! | `BranchLifecycle` | **Raw** | YES (deterministic enum) |
//! | `BranchStats` | **Raw** | YES (deterministic counters) |
//! | `ProceduralRule` (counters) | **Raw** | YES |
//! | `ProceduralRule.direction` | **Latent** | NO (projection vector) |
//! | `EpisodicEntry.embedding` | **Latent** | NO |
//! | `EpisodicEntry.payload` | **Caller-defined** | Caller decides |
//! | `EpisodicEntry.reward` | **Raw** | YES |
//! | `spawn_anchor` | **Latent** | NO (direction vector) |
//! | `token_signature` | **Raw** | YES (deterministic hashes) |

use core::fmt;

// ─── Branch identifier ────────────────────────────────────────────────────

/// Dense index into a [`crate::branching::bank::BranchBank`].
///
/// `#[repr(transparent)]` over `u32` so an array of `BranchId` is byte-compatible
/// with `&[u32]`. Stable for the lifetime of the slot: a pruned branch keeps its
/// `BranchId`; a reused slot inherits the slot's `BranchId`. Callers that need
/// tamper-evident continuity across prune-reuse MUST consult the ARG
/// `RedirectTable` (when `arg_protocol` is enabled) — `BranchId` alone does not
/// distinguish "old branch at this slot" from "reused branch at this slot".
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BranchId(pub u32);

impl BranchId {
    /// Construct from a raw `u32`.
    #[inline]
    #[must_use]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    /// Sentinel for "no branch". Used as a non-match marker by the router;
    /// never a valid index into a `BranchBank`.
    pub const SENTINEL: Self = Self(u32::MAX);

    /// True if this is the sentinel value.
    #[inline]
    pub const fn is_sentinel(self) -> bool {
        self.0 == u32::MAX
    }
}

impl From<u32> for BranchId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<BranchId> for u32 {
    #[inline]
    fn from(id: BranchId) -> u32 {
        id.0
    }
}

impl fmt::Display for BranchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_sentinel() {
            write!(f, "branch:SENTINEL")
        } else {
            write!(f, "branch:{}", self.0)
        }
    }
}

// ─── Branch lifecycle ─────────────────────────────────────────────────────

/// Lifecycle state for a cognitive branch.
///
/// When the `arg_protocol` feature is on, this is a type alias for
/// [`crate::arg::LifecycleState`] — the same type used by the ARG protocol's
/// ontology lifecycle (Step E). When `arg_protocol` is off, a local enum with
/// identical discriminants and semantics is provided so this module compiles
/// standalone.
///
/// Progression is monotonic: `Active → Deprecated → Removed`. `Shadow` is the
/// pre-promotion staging state (reduces blast radius during early adoption).
#[cfg(feature = "arg_protocol")]
pub type BranchLifecycle = crate::arg::LifecycleState;

#[cfg(not(feature = "arg_protocol"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BranchLifecycle {
    /// Visible and routable. The default for newly spawned branches.
    #[default]
    Active = 0,
    /// Pre-promotion staging: suggested but with limited routing weight.
    Shadow = 1,
    /// Superseded by a merged replacement; still resolvable via redirect.
    Deprecated = 2,
    /// Permanently gone. The slot may be reused; the branch id does not
    /// survive reuse. Lookups against a `Removed` branch MUST consult the
    /// `RedirectTable` (when `arg_protocol` is enabled).
    Removed = 3,
}

#[cfg(not(feature = "arg_protocol"))]
impl BranchLifecycle {
    /// Returns `true` when the branch is routable online (router may snap to it).
    #[inline]
    pub fn is_routable(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Returns `true` when lookups MUST consult a redirect table before
    /// returning (continuity requirement mirroring ARG §3.5).
    #[inline]
    pub fn requires_redirect(self) -> bool {
        matches!(self, Self::Deprecated | Self::Removed)
    }
}

// ─── Episodic entry ───────────────────────────────────────────────────────

/// A verifier-approved episodic example stored in a branch.
///
/// `embedding` is the latent vector at write time (used for centroid / quarantine
/// checks). `payload` is the caller-defined content (e.g., an Engram shard ref,
/// a closure motif, a game-state snapshot). `reward` is the verifier score that
/// admitted the write. `scope` is an optional caller-defined scope tag
/// (e.g., task family id). `tick` is the deterministic write tick.
#[derive(Clone, Debug)]
pub struct EpisodicEntry<E> {
    /// Latent embedding at write time (caller-normalized).
    pub embedding: Vec<f32>,
    /// Caller-defined payload (Engram ref, motif, snapshot, ...).
    pub payload: E,
    /// Verifier reward `r ∈ [0,1]` that admitted this write.
    pub reward: f32,
    /// Optional caller-defined scope tag (e.g., task family id).
    pub scope: Option<u64>,
    /// Deterministic write tick (raw, syncable).
    pub tick: u64,
}

// ─── Procedural rule ──────────────────────────────────────────────────────

/// An IF-THEN procedural rule with helpful / harmful counters.
///
/// Distilled from RIZZ §"procedural rules" `(u_j, α_j, β_j, H_j, A_j)`:
/// - `direction` is the latent direction this rule fires on (dot-product gate).
/// - `antecedent` is a BLAKE3 commitment of the rule's precondition (the "IF").
/// - `strategy` is a BLAKE3 commitment of the rule's action (the "THEN").
/// - `helpful` counts how often firing this rule improved the outcome.
/// - `harmful` counts how often firing this rule worsened the outcome.
///
/// The net credit is `helpful - harmful`; the rule is pruned when net credit
/// falls below zero for a sustained window. This is the procedural analogue of
/// the CLR reward gate.
#[derive(Clone, Debug)]
pub struct ProceduralRule {
    /// Latent direction this rule fires on (caller-normalized).
    pub direction: Vec<f32>,
    /// BLAKE3 commitment of the precondition (the "IF" side).
    pub antecedent: [u8; 32],
    /// BLAKE3 commitment of the action (the "THEN" side).
    pub strategy: [u8; 32],
    /// Count of outcome-improving firings.
    pub helpful: u32,
    /// Count of outcome-worsening firings.
    pub harmful: u32,
}

impl ProceduralRule {
    /// Net credit (`helpful - harmful`). Positive = keep; negative = prune candidate.
    #[inline]
    #[must_use]
    pub fn net_credit(&self) -> i64 {
        self.helpful as i64 - self.harmful as i64
    }

    /// Increment the helpful counter (rule fired and outcome improved).
    #[inline]
    pub fn record_helpful(&mut self) {
        self.helpful = self.helpful.saturating_add(1);
    }

    /// Increment the harmful counter (rule fired and outcome worsened).
    #[inline]
    pub fn record_harmful(&mut self) {
        self.harmful = self.harmful.saturating_add(1);
    }
}

// ─── Failure entry ────────────────────────────────────────────────────────

/// A substantive failure (anti-pattern) stored in a branch.
///
/// RIZZ §"branch-local memory": failures are concrete anti-patterns — things
/// that demonstrably did not work. Unlike episodic entries (positive examples)
/// and procedural rules (IF-THEN with credit), failures are stored as
/// "do not do this near this branch" anchors. They stay branch-local: a failure
/// in the combat branch never contaminates the crafting branch.
#[derive(Clone, Debug)]
pub struct FailureEntry<E> {
    /// Latent embedding of the failed input.
    pub embedding: Vec<f32>,
    /// Caller-defined payload describing the failure.
    pub payload: E,
    /// Deterministic write tick (raw, syncable).
    pub tick: u64,
}

// ─── Branch statistics ────────────────────────────────────────────────────

/// Per-branch statistics tracked for lifecycle decisions (merge / prune).
///
/// `#[repr(C)]` + all fields are `Copy` → Pod-compatible, sync-friendly,
/// zero-copy mmap-able when the consumer persists a branch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct BranchStats {
    /// Number of verifier-approved writes to this branch.
    pub n_writes: u32,
    /// Number of route-side reads (snap-to-this-branch events).
    pub n_reads: u32,
    /// Running average reward of admitted writes (incremental update).
    pub avg_reward: f32,
    /// Last tick this branch was touched (read or write).
    pub last_touch_tick: u64,
}

impl BranchStats {
    /// Record a write and update the incremental average reward.
    #[inline]
    pub fn record_write(&mut self, reward: f32, tick: u64) {
        self.n_writes = self.n_writes.saturating_add(1);
        let n = self.n_writes as f32;
        self.avg_reward += (reward - self.avg_reward) / n;
        self.last_touch_tick = tick;
    }

    /// Record a read (snap-to-this-branch event).
    #[inline]
    pub fn record_read(&mut self, tick: u64) {
        self.n_reads = self.n_reads.saturating_add(1);
        self.last_touch_tick = tick;
    }

    /// True if this branch is stale (no touch within `stale_window` ticks of `now`).
    #[inline]
    #[must_use]
    pub fn is_stale(&self, now: u64, stale_window: u64) -> bool {
        now.saturating_sub(self.last_touch_tick) > stale_window
    }
}

// ─── Cognitive branch ─────────────────────────────────────────────────────

/// A persistent cognitive branch — a "zero-interference zone" (RIZZ §"memory
/// branch").
///
/// Each branch accumulates verifier-approved episodic examples, procedural
/// rules (with helpful/harmful counters), and failure anti-patterns. The
/// `spawn_anchor` is the latent direction this branch represents; the router
/// snaps query embeddings to branches by dot-product against `spawn_anchor`.
/// `token_signature` is an optional sorted set of hash tokens enabling the
/// Jaccard fallback path in the router.
///
/// Non-interference is structural: two branches `b_i`, `b_j` are non-interfering
/// iff their anchor directions are orthogonal (`dot(g_{b_i}, g_{b_j}) ≈ 0`).
/// Writes projected onto one branch's direction have zero component along any
/// orthogonal sibling's direction (Phase 2 `NonInterferenceProjection`).
#[derive(Clone, Debug)]
pub struct CognitiveBranch<E> {
    /// Dense slot index (matches the slot this branch lives in).
    pub id: BranchId,
    /// Latent direction this branch represents (caller-normalized).
    pub spawn_anchor: Vec<f32>,
    /// Sorted, deduplicated hash tokens for Jaccard fallback (empty = disabled).
    pub token_signature: Vec<u64>,
    /// Verifier-approved episodic examples (positive memory).
    pub episodic: Vec<EpisodicEntry<E>>,
    /// Procedural rules with helpful/harmful credit counters.
    pub procedural: Vec<ProceduralRule>,
    /// Failure anti-patterns (branch-local negative memory).
    pub failures: Vec<FailureEntry<E>>,
    /// Optional caller-defined scope context tag.
    pub scope_ctx: Option<u64>,
    /// Per-branch statistics for lifecycle decisions.
    pub stats: BranchStats,
    /// Lifecycle state (Active / Shadow / Deprecated / Removed).
    pub lifecycle: BranchLifecycle,
}

impl<E> CognitiveBranch<E> {
    /// Construct a fresh active branch with the given spawn anchor.
    ///
    /// All memory stores start empty; `token_signature` starts empty (Jaccard
    /// fallback disabled until the caller populates it).
    #[inline]
    #[must_use]
    pub fn new(id: BranchId, spawn_anchor: Vec<f32>) -> Self {
        Self {
            id,
            spawn_anchor,
            token_signature: Vec::new(),
            episodic: Vec::new(),
            procedural: Vec::new(),
            failures: Vec::new(),
            scope_ctx: None,
            stats: BranchStats::default(),
            lifecycle: BranchLifecycle::default(),
        }
    }

    /// Number of memory entries (episodic + procedural + failures).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.episodic.len() + self.procedural.len() + self.failures.len()
    }

    /// True if all memory stores are empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a sorted token onto the signature (maintains sort + dedup).
    /// Call this after writing an episodic entry to enable Jaccard fallback.
    pub fn push_token(&mut self, token: u64) {
        let pos = self.token_signature.binary_search(&token).unwrap_or_else(|p| p);
        if pos == self.token_signature.len() || self.token_signature[pos] != token {
            self.token_signature.insert(pos, token);
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_id_round_trip() {
        let id = BranchId::new(42);
        assert_eq!(u32::from(id), 42);
        assert_eq!(BranchId::from(42u32), id);
        assert!(!id.is_sentinel());
        assert!(BranchId::SENTINEL.is_sentinel());
        assert_eq!(format!("{id}"), "branch:42");
        assert_eq!(format!("{}", BranchId::SENTINEL), "branch:SENTINEL");
    }

    #[test]
    fn branch_id_default_is_zero() {
        assert_eq!(BranchId::default(), BranchId::new(0));
    }

    #[test]
    fn lifecycle_active_is_routable() {
        assert!(BranchLifecycle::default().is_routable());
        assert!(BranchLifecycle::Active.is_routable());
        assert!(!BranchLifecycle::Shadow.is_routable());
        assert!(!BranchLifecycle::Deprecated.is_routable());
        assert!(!BranchLifecycle::Removed.is_routable());
    }

    #[test]
    fn lifecycle_redirect_requirements() {
        assert!(!BranchLifecycle::Active.requires_redirect());
        assert!(!BranchLifecycle::Shadow.requires_redirect());
        assert!(BranchLifecycle::Deprecated.requires_redirect());
        assert!(BranchLifecycle::Removed.requires_redirect());
    }

    #[test]
    fn procedural_rule_credit_arithmetic() {
        let mut rule = ProceduralRule {
            direction: vec![1.0, 0.0],
            antecedent: [0u8; 32],
            strategy: [1u8; 32],
            helpful: 5,
            harmful: 2,
        };
        assert_eq!(rule.net_credit(), 3);
        rule.record_helpful();
        assert_eq!(rule.net_credit(), 4);
        rule.record_harmful();
        rule.record_harmful();
        assert_eq!(rule.net_credit(), 2);
    }

    #[test]
    fn procedural_rule_saturating_counters() {
        let mut rule = ProceduralRule {
            direction: vec![],
            antecedent: [0u8; 32],
            strategy: [0u8; 32],
            helpful: u32::MAX,
            harmful: 0,
        };
        rule.record_helpful();
        assert_eq!(rule.helpful, u32::MAX); // saturated
        assert_eq!(rule.net_credit(), u32::MAX as i64);
    }

    #[test]
    fn branch_stats_incremental_average() {
        let mut stats = BranchStats::default();
        stats.record_write(0.5, 1);
        assert!((stats.avg_reward - 0.5).abs() < 1e-6);
        stats.record_write(1.0, 2);
        assert!((stats.avg_reward - 0.75).abs() < 1e-6);
        stats.record_write(0.25, 3);
        // (0.5 + 1.0 + 0.25) / 3 = 0.583...
        assert!((stats.avg_reward - 0.58333).abs() < 1e-3);
        assert_eq!(stats.n_writes, 3);
        assert_eq!(stats.last_touch_tick, 3);
    }

    #[test]
    fn branch_stats_staleness() {
        let stats = BranchStats {
            last_touch_tick: 100,
            ..Default::default()
        };
        assert!(!stats.is_stale(150, 100)); // 50 ticks since touch, window 100
        assert!(stats.is_stale(250, 100)); // 150 ticks since touch, window 100
        assert!(!stats.is_stale(50, 100)); // now < touch (saturating_sub → 0)
    }

    #[test]
    fn cognitive_branch_new_is_empty_active() {
        let branch = CognitiveBranch::<()>::new(BranchId::new(0), vec![1.0, 0.0, 0.0]);
        assert!(branch.is_empty());
        assert_eq!(branch.len(), 0);
        assert!(branch.lifecycle.is_routable());
        assert!(branch.token_signature.is_empty());
        assert!(branch.episodic.is_empty());
        assert!(branch.procedural.is_empty());
        assert!(branch.failures.is_empty());
        assert_eq!(branch.spawn_anchor, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn cognitive_branch_push_token_maintains_sorted_dedup() {
        let mut branch = CognitiveBranch::<()>::new(BranchId::new(0), vec![]);
        branch.push_token(30);
        branch.push_token(10);
        branch.push_token(20);
        branch.push_token(10); // duplicate
        branch.push_token(30); // duplicate
        assert_eq!(branch.token_signature, vec![10, 20, 30]);
    }

    #[test]
    fn cognitive_branch_clone_preserves_all_fields() {
        let mut branch = CognitiveBranch::<&'static str>::new(BranchId::new(3), vec![0.0, 1.0]);
        branch.episodic.push(EpisodicEntry {
            embedding: vec![1.0],
            payload: "hello",
            reward: 0.8,
            scope: Some(7),
            tick: 42,
        });
        branch.stats.n_writes = 5;
        branch.push_token(99);

        let cloned = branch.clone();
        assert_eq!(cloned.id, branch.id);
        assert_eq!(cloned.spawn_anchor, branch.spawn_anchor);
        assert_eq!(cloned.token_signature, branch.token_signature);
        assert_eq!(cloned.episodic.len(), 1);
        assert_eq!(cloned.episodic[0].payload, "hello");
        assert_eq!(cloned.stats.n_writes, 5);
    }
}
