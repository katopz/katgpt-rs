//! Proof sketch types — core data structures for Elo-rated population search (Plan 128, T2).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "Sketch entries store proof state, pending goals, and lessons learned
//! from each Ralph loop iteration, rated by Plackett-Luce → Elo."
//!
//! # Types
//!
//! - [`SketchId`] — unique 16-byte identifier (UUID-layout compatible)
//! - [`ProofState`] — canonical game/proof state with blake3 hash
//! - [`Goal`] — unresolved subgoal with label and canonical bytes
//! - [`SketchEntry`] — population entry: state + goals + lessons + Elo + visits
//! - [`DiversityStrategy`] — structured exploration injection (T5)
//! - [`DiversityHint`] — concrete hint produced by a diversity strategy
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::goal_cache::GoalHash;

// ── SketchId ───────────────────────────────────────────────────

/// Unique identifier for a proof sketch entry.
///
/// 16-byte identifier with the same layout as UUID, generated via
/// blake3 hash for determinism. Compatible with `Uuid::now_v7()`
/// if the `uuid` crate is added later.
///
/// # Generation
///
/// Uses a global counter + blake3 to produce unique IDs without
/// requiring external dependencies. IDs are not time-ordered by
/// default — use [`SketchEntry::created_at`] for temporal ordering.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SketchId(#[serde(with = "id_bytes")] pub(crate) [u8; 16]);

mod id_bytes {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8; 16], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(data)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 16], D::Error> {
        let bytes: Vec<u8> = serde::de::Deserialize::deserialize(deserializer)?;
        let mut arr = [0u8; 16];
        let len = bytes.len().min(16);
        arr[..len].copy_from_slice(&bytes[..len]);
        Ok(arr)
    }
}

/// Global counter for unique ID generation.
static SKETCH_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl SketchId {
    /// Generate a new unique sketch ID.
    ///
    /// Uses atomic counter + blake3 for uniqueness without collisions.
    pub fn new() -> Self {
        let counter = SKETCH_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let hash = blake3::hash(&counter.to_le_bytes());
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        Self(id)
    }

    /// Create from raw bytes (e.g., deserialized from disk).
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Raw bytes of the ID.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Create a nil (all-zero) ID for sentinel use.
    pub const fn nil() -> Self {
        Self([0u8; 16])
    }

    /// Is this a nil ID?
    pub fn is_nil(&self) -> bool {
        self.0 == [0u8; 16]
    }
}

impl Default for SketchId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SketchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SketchId(")?;
        for &b in &self.0[..8] {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for SketchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for &b in &self.0[..4] {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

// ── ProofState ─────────────────────────────────────────────────

/// Canonical proof/game state with blake3 hash for fast comparison.
///
/// Stores the serialized state as bytes and a pre-computed blake3 hash.
/// Two states are equal iff their hashes are equal (collision probability
/// negligible for cache/key purposes).
///
/// # Canonical Encoding
///
/// The caller must ensure the byte encoding is deterministic:
/// - Same logical state → same bytes
/// - Different logical states → different bytes (with high probability)
///
/// For game domains, this is typically a serialization of board position +
/// player state + turn number. For proof domains, it's the Lean proof state.
#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ProofState {
    /// Canonical byte representation of the state.
    canonical: Vec<u8>,
    /// Pre-computed blake3 hash for fast comparison.
    #[serde(skip)]
    hash: GoalHash,
}

/// Manual Deserialize: recomputes blake3 hash from canonical bytes.
/// Avoids requiring `Default` on `GoalHash` and ensures hash integrity.
impl<'de> serde::Deserialize<'de> for ProofState {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct ProofStateData {
            canonical: Vec<u8>,
        }
        let data = ProofStateData::deserialize(deserializer)?;
        Ok(Self::new(data.canonical))
    }
}

impl ProofState {
    /// Create a new proof state from canonical bytes.
    ///
    /// Hashes the bytes with blake3 for fast comparison.
    pub fn new(canonical: Vec<u8>) -> Self {
        let hash = GoalHash::from_canonical(&canonical);
        Self { canonical, hash }
    }

    /// Canonical bytes of the state.
    pub fn canonical(&self) -> &[u8] {
        &self.canonical
    }

    /// Pre-computed blake3 hash.
    pub fn hash(&self) -> &GoalHash {
        &self.hash
    }

    /// Size of the canonical bytes.
    pub fn len(&self) -> usize {
        self.canonical.len()
    }

    /// Is the state empty (no bytes)?
    pub fn is_empty(&self) -> bool {
        self.canonical.is_empty()
    }
}

impl fmt::Debug for ProofState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.canonical.len() {
            0 => write!(f, "ProofState(<empty>)"),
            n if n <= 32 => write!(f, "ProofState({:02x?})", self.canonical),
            _ => {
                write!(f, "ProofState(")?;
                for &b in &self.canonical[..16] {
                    write!(f, "{b:02x}")?;
                }
                write!(f, "... ({} bytes))", self.canonical.len())
            }
        }
    }
}

// ── Goal ───────────────────────────────────────────────────────

/// An unresolved subgoal within a proof sketch.
///
/// Represents a proof obligation or strategic objective that needs
/// to be resolved. Each goal has a human-readable label and canonical
/// bytes for hashing/comparison.
///
/// # Examples
///
/// - Proof domain: "Prove that n is even", "Show f(x) > 0 for all x ∈ S"
/// - Game domain: "Control center territory", "Prevent enemy flanking"
/// - Bomber domain: "Reach safe position within 3 ticks"
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Goal {
    /// Human-readable goal description.
    label: String,
    /// Canonical bytes for hashing and comparison.
    canonical: Vec<u8>,
}

impl Goal {
    /// Create a new goal with label and canonical bytes.
    pub fn new(label: impl Into<String>, canonical: Vec<u8>) -> Self {
        Self {
            label: label.into(),
            canonical,
        }
    }

    /// Create a goal with only a label (canonical = label bytes).
    pub fn from_label(label: impl Into<String>) -> Self {
        let label = label.into();
        let canonical = label.as_bytes().to_vec();
        Self { label, canonical }
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Canonical bytes.
    pub fn canonical(&self) -> &[u8] {
        &self.canonical
    }

    /// blake3 hash of the canonical bytes.
    pub fn hash(&self) -> GoalHash {
        GoalHash::from_canonical(&self.canonical)
    }
}

impl fmt::Display for Goal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label)
    }
}

// ── DiversityStrategy ──────────────────────────────────────────

/// Structured exploration strategy for diversity injection (Plan 128, T5).
///
/// Prevents population collapse by injecting structured exploration hints
/// during the explore arm of P-UCB sampling. Distilled from AlphaProof Nexus
/// Supplementary Insight 7: "Cheap structured exploration strategies
/// (Decompose/Combine/NovelApproach) prevent population collapse."
///
/// Each variant maps to a domain-specific exploration pattern:
///
/// | Variant | Proof Domain | Game Domain |
/// |---------|-------------|-------------|
/// | `Decompose` | Split lemma into sub-lemmas | Split territory fight |
/// | `Combine` | Merge prior proof attempts | Team tactic merge |
/// | `NovelApproach` | Try new proof technique | Switch opening/heuristic |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiversityStrategy {
    /// Split complex goals into sub-goals.
    ///
    /// Proof: "Decompose goal into simpler sub-lemmas"
    /// Game: "Split territory into independent fights"
    Decompose,
    /// Merge ideas from prior attempts.
    ///
    /// Proof: "Combine successful proof fragments"
    /// Game: "Merge team tactics from previous rounds"
    Combine,
    /// Try completely new strategy.
    ///
    /// Proof: "Apply novel proof technique"
    /// Game: "Switch opening/heuristic entirely"
    NovelApproach,
}

impl DiversityStrategy {
    /// All strategy variants for iteration/sampling.
    pub const ALL: [DiversityStrategy; 3] = [
        DiversityStrategy::Decompose,
        DiversityStrategy::Combine,
        DiversityStrategy::NovelApproach,
    ];

    /// Index for use in RNG-based selection.
    pub fn index(self) -> usize {
        match self {
            Self::Decompose => 0,
            Self::Combine => 1,
            Self::NovelApproach => 2,
        }
    }

    /// Get strategy from index (wraps around for indices > 2).
    pub fn from_index(idx: usize) -> Self {
        match idx % 3 {
            0 => Self::Decompose,
            1 => Self::Combine,
            _ => Self::NovelApproach,
        }
    }

    /// Human-readable description of the strategy's intent.
    pub fn description(self) -> &'static str {
        match self {
            Self::Decompose => "Split complex goals into sub-goals",
            Self::Combine => "Merge ideas from prior attempts",
            Self::NovelApproach => "Try completely new strategy",
        }
    }
}

impl fmt::Display for DiversityStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decompose => write!(f, "Decompose"),
            Self::Combine => write!(f, "Combine"),
            Self::NovelApproach => write!(f, "NovelApproach"),
        }
    }
}

// ── DiversityHint ──────────────────────────────────────────────

/// Concrete exploration hint produced by a diversity strategy.
///
/// Combines the strategy type with optional context for the agent.
/// Generated during the explore arm of P-UCB sampling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiversityHint {
    /// The strategy that produced this hint.
    pub strategy: DiversityStrategy,
    /// Optional domain-specific context (e.g., "focus on left flank").
    pub context: Option<String>,
}

impl DiversityHint {
    /// Create a hint with no additional context.
    pub fn new(strategy: DiversityStrategy) -> Self {
        Self {
            strategy,
            context: None,
        }
    }

    /// Create a hint with context.
    pub fn with_context(strategy: DiversityStrategy, context: impl Into<String>) -> Self {
        Self {
            strategy,
            context: Some(context.into()),
        }
    }
}

impl fmt::Display for DiversityHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.context {
            Some(ctx) => write!(f, "{}: {ctx}", self.strategy),
            None => write!(f, "{}", self.strategy),
        }
    }
}

// ── SketchEntry ────────────────────────────────────────────────

/// A proof sketch entry in the Elo-rated population database (Plan 128, T3).
///
/// Represents a single proof/game strategy with its current state,
/// unresolved goals, learned lessons, and Elo rating. Entries are
/// stored in a top-K population and sampled via P-UCB.
///
/// # Paper Mapping
///
/// From AlphaProof Nexus:
/// - `proof_state` ↔ Lean proof state (serialized)
/// - `pending_goals` ↔ Unresolved "sorry" placeholders
/// - `lessons` ↔ Ralph loop episode summaries
/// - `elo_rating` ↔ Plackett-Luce aggregated Elo
/// - `visits` ↔ P-UCB visit count
///
/// # Lifecycle
///
/// 1. Created with initial state + goals → Elo 1200 (paper default)
/// 2. Evaluated via LLM raters or game outcomes
/// 3. Elo updated via Plackett-Luce rating (T4)
/// 4. Sampled via P-UCB for next iteration (T5)
/// 5. Evicted if outside top-K by Elo (T3)
#[derive(Clone, Debug, Serialize)]
pub struct SketchEntry {
    /// Unique identifier.
    pub id: SketchId,
    /// Current proof/game state.
    pub proof_state: ProofState,
    /// Unresolved subgoals.
    pub pending_goals: Vec<Goal>,
    /// Lessons learned from episodes (Ralph loop summaries).
    pub lessons: VecDeque<String>,
    /// Elo rating from Plackett-Luce aggregation.
    pub elo_rating: f64,
    /// P-UCB visit count.
    pub visits: usize,
    /// Creation timestamp.
    #[serde(skip)]
    pub created_at: Instant,
}

/// Manual Deserialize: uses `Instant::now()` for the skipped timestamp field.
/// Avoids requiring `Default` on `Instant` (which is intentionally not implemented).
impl<'de> serde::Deserialize<'de> for SketchEntry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct SketchEntryData {
            id: SketchId,
            proof_state: ProofState,
            pending_goals: Vec<Goal>,
            lessons: VecDeque<String>,
            elo_rating: f64,
            visits: usize,
        }
        let data = SketchEntryData::deserialize(deserializer)?;
        Ok(Self {
            id: data.id,
            proof_state: data.proof_state,
            pending_goals: data.pending_goals,
            lessons: data.lessons,
            elo_rating: data.elo_rating,
            visits: data.visits,
            created_at: Instant::now(),
        })
    }
}

/// Default Elo rating for new entries (paper default: 1200).
pub const DEFAULT_ELO: f64 = 1200.0;

/// Elo scale factor (paper default: 400, standard chess Elo).
pub const ELO_SCALE: f64 = 400.0;

/// Maximum lessons stored per entry (prevents unbounded growth).
pub const MAX_LESSONS: usize = 16;

/// Maximum pending goals stored per entry.
pub const MAX_PENDING_GOALS: usize = 32;

impl SketchEntry {
    /// Create a new sketch entry with default Elo (1200).
    ///
    /// # Arguments
    ///
    /// * `proof_state` — the current game/proof state
    /// * `pending_goals` — unresolved subgoals
    pub fn new(proof_state: ProofState, pending_goals: Vec<Goal>) -> Self {
        Self {
            id: SketchId::new(),
            proof_state,
            pending_goals: pending_goals.into_iter().take(MAX_PENDING_GOALS).collect(),
            lessons: VecDeque::new(),
            elo_rating: DEFAULT_ELO,
            visits: 0,
            created_at: Instant::now(),
        }
    }

    /// Create with initial Elo rating.
    pub fn with_elo(proof_state: ProofState, pending_goals: Vec<Goal>, elo: f64) -> Self {
        Self {
            elo_rating: elo,
            ..Self::new(proof_state, pending_goals)
        }
    }

    /// Add a lesson from an episode.
    ///
    /// Keeps only the last `MAX_LESSONS` entries (FIFO eviction).
    pub fn add_lesson(&mut self, lesson: String) {
        if self.lessons.len() >= MAX_LESSONS {
            self.lessons.pop_front();
        }
        self.lessons.push_back(lesson);
    }

    /// Record a visit (increment visit count).
    pub fn record_visit(&mut self) {
        self.visits += 1;
    }

    /// Update Elo rating.
    pub fn update_elo(&mut self, new_elo: f64) {
        self.elo_rating = new_elo;
    }

    /// Number of pending (unresolved) goals.
    pub fn pending_goal_count(&self) -> usize {
        self.pending_goals.len()
    }

    /// Number of lessons learned.
    pub fn lesson_count(&self) -> usize {
        self.lessons.len()
    }

    /// Has this entry been visited at least once?
    pub fn is_explored(&self) -> bool {
        self.visits > 0
    }

    /// Age of this entry as elapsed time since creation.
    ///
    /// Returns `None` if the entry was deserialized (no Instant).
    /// Use `serde(skip)` note: deserialized entries have `Instant::now()`.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// P-UCB score for this entry.
    ///
    /// `score = q + c * sqrt(total_visits / (visits + 1))`
    ///
    /// where `q` is the normalized Elo rating in [0, 1].
    ///
    /// # Arguments
    ///
    /// * `total_visits` — sum of visits across all entries in population
    /// * `min_elo` — minimum Elo in population (for normalization)
    /// * `max_elo` — maximum Elo in population (for normalization)
    /// * `c` — exploration constant (paper uses 0.2)
    pub fn p_ucb_score(&self, total_visits: usize, min_elo: f64, max_elo: f64, c: f64) -> f64 {
        let q = normalize_to_01(self.elo_rating, min_elo, max_elo);
        let exploration = c * (total_visits as f64 / (self.visits + 1) as f64).sqrt();
        q + exploration
    }
}

/// Normalize a value to [0.0, 1.0] range given min and max bounds.
///
/// Returns 0.5 if min == max (degenerate case).
fn normalize_to_01(value: f64, min: f64, max: f64) -> f64 {
    match max > min {
        true => (value - min) / (max - min),
        false => 0.5,
    }
}

impl fmt::Display for SketchEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SketchEntry(id={}, elo={:.0}, visits={}, goals={}, lessons={})",
            self.id,
            self.elo_rating,
            self.visits,
            self.pending_goals.len(),
            self.lessons.len(),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SketchId Tests ────────────────────────────────────────

    #[test]
    fn sketch_id_unique() {
        let id1 = SketchId::new();
        let id2 = SketchId::new();
        assert_ne!(id1, id2, "consecutive IDs must be unique");
    }

    #[test]
    fn sketch_id_nil() {
        let nil = SketchId::nil();
        assert!(nil.is_nil());
        assert_eq!(nil.as_bytes(), &[0u8; 16]);
    }

    #[test]
    fn sketch_id_non_nil() {
        let id = SketchId::new();
        assert!(!id.is_nil());
    }

    #[test]
    fn sketch_id_from_bytes_roundtrip() {
        let id = SketchId::new();
        let bytes = *id.as_bytes();
        let restored = SketchId::from_bytes(bytes);
        assert_eq!(id, restored);
    }

    #[test]
    fn sketch_id_display_short() {
        let id = SketchId::new();
        let display = format!("{id}");
        assert_eq!(display.len(), 8, "display shows first 8 hex chars");
    }

    #[test]
    fn sketch_id_debug_format() {
        let id = SketchId::new();
        let debug = format!("{id:?}");
        assert!(debug.starts_with("SketchId("));
    }

    // ── ProofState Tests ──────────────────────────────────────

    #[test]
    fn proof_state_hash_deterministic() {
        let s1 = ProofState::new(b"state_A".to_vec());
        let s2 = ProofState::new(b"state_A".to_vec());
        assert_eq!(s1.hash(), s2.hash());
        assert_eq!(s1, s2);
    }

    #[test]
    fn proof_state_different_states() {
        let s1 = ProofState::new(b"state_A".to_vec());
        let s2 = ProofState::new(b"state_B".to_vec());
        assert_ne!(s1, s2);
        assert_ne!(s1.hash(), s2.hash());
    }

    #[test]
    fn proof_state_len() {
        let s = ProofState::new(b"12345".to_vec());
        assert_eq!(s.len(), 5);
        assert!(!s.is_empty());
    }

    #[test]
    fn proof_state_empty() {
        let s = ProofState::new(vec![]);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn proof_state_debug_truncates_long() {
        let long = vec![0xAB_u8; 100];
        let s = ProofState::new(long);
        let debug = format!("{s:?}");
        assert!(debug.contains("..."));
        assert!(debug.contains("100 bytes"));
    }

    // ── Goal Tests ────────────────────────────────────────────

    #[test]
    fn goal_from_label() {
        let g = Goal::from_label("Prove n is even");
        assert_eq!(g.label(), "Prove n is even");
        assert_eq!(g.canonical(), b"Prove n is even");
    }

    #[test]
    fn goal_with_custom_canonical() {
        let g = Goal::new("Goal A", b"\x01\x02\x03".to_vec());
        assert_eq!(g.label(), "Goal A");
        assert_eq!(g.canonical(), &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn goal_hash_matches_canonical() {
        let g = Goal::from_label("test");
        let expected = GoalHash::from_canonical(b"test");
        assert_eq!(g.hash(), expected);
    }

    #[test]
    fn goal_display_shows_label() {
        let g = Goal::from_label("Control center");
        assert_eq!(format!("{g}"), "Control center");
    }

    #[test]
    fn goal_equality() {
        let g1 = Goal::new("A", b"\x01".to_vec());
        let g2 = Goal::new("A", b"\x01".to_vec());
        let g3 = Goal::new("B", b"\x02".to_vec());
        assert_eq!(g1, g2);
        assert_ne!(g1, g3);
    }

    // ── DiversityStrategy Tests ───────────────────────────────

    #[test]
    fn diversity_strategy_all_three() {
        assert_eq!(DiversityStrategy::ALL.len(), 3);
    }

    #[test]
    fn diversity_strategy_index_roundtrip() {
        for strategy in DiversityStrategy::ALL {
            assert_eq!(DiversityStrategy::from_index(strategy.index()), strategy);
        }
    }

    #[test]
    fn diversity_strategy_from_index_wraps() {
        assert_eq!(
            DiversityStrategy::from_index(3),
            DiversityStrategy::Decompose
        );
        assert_eq!(DiversityStrategy::from_index(4), DiversityStrategy::Combine);
        assert_eq!(
            DiversityStrategy::from_index(5),
            DiversityStrategy::NovelApproach
        );
        assert_eq!(
            DiversityStrategy::from_index(100),
            DiversityStrategy::from_index(100 % 3)
        );
    }

    #[test]
    fn diversity_strategy_descriptions() {
        assert!(!DiversityStrategy::Decompose.description().is_empty());
        assert!(!DiversityStrategy::Combine.description().is_empty());
        assert!(!DiversityStrategy::NovelApproach.description().is_empty());
    }

    #[test]
    fn diversity_strategy_display() {
        assert_eq!(format!("{}", DiversityStrategy::Decompose), "Decompose");
        assert_eq!(format!("{}", DiversityStrategy::Combine), "Combine");
        assert_eq!(
            format!("{}", DiversityStrategy::NovelApproach),
            "NovelApproach"
        );
    }

    // ── DiversityHint Tests ───────────────────────────────────

    #[test]
    fn diversity_hint_no_context() {
        let hint = DiversityHint::new(DiversityStrategy::Decompose);
        assert_eq!(hint.strategy, DiversityStrategy::Decompose);
        assert!(hint.context.is_none());
    }

    #[test]
    fn diversity_hint_with_context() {
        let hint = DiversityHint::with_context(DiversityStrategy::Combine, "left flank");
        assert_eq!(hint.context, Some("left flank".to_string()));
    }

    #[test]
    fn diversity_hint_display_no_context() {
        let hint = DiversityHint::new(DiversityStrategy::NovelApproach);
        assert_eq!(format!("{hint}"), "NovelApproach");
    }

    #[test]
    fn diversity_hint_display_with_context() {
        let hint = DiversityHint::with_context(DiversityStrategy::Decompose, "center");
        assert_eq!(format!("{hint}"), "Decompose: center");
    }

    // ── SketchEntry Tests ─────────────────────────────────────

    #[test]
    fn sketch_entry_new_defaults() {
        let state = ProofState::new(b"initial".to_vec());
        let entry = SketchEntry::new(state, vec![Goal::from_label("goal_1")]);

        assert!(!entry.id.is_nil());
        assert_eq!(entry.elo_rating, DEFAULT_ELO);
        assert_eq!(entry.visits, 0);
        assert!(!entry.is_explored());
        assert_eq!(entry.pending_goal_count(), 1);
        assert_eq!(entry.lesson_count(), 0);
        assert!(entry.lessons.is_empty());
    }

    #[test]
    fn sketch_entry_with_custom_elo() {
        let state = ProofState::new(b"s".to_vec());
        let entry = SketchEntry::with_elo(state, vec![], 1500.0);
        assert_eq!(entry.elo_rating, 1500.0);
    }

    #[test]
    fn sketch_entry_add_lesson() {
        let state = ProofState::new(b"s".to_vec());
        let mut entry = SketchEntry::new(state, vec![]);

        entry.add_lesson("First lesson".to_string());
        assert_eq!(entry.lesson_count(), 1);
        assert_eq!(entry.lessons[0], "First lesson");

        entry.add_lesson("Second lesson".to_string());
        assert_eq!(entry.lesson_count(), 2);
    }

    #[test]
    fn sketch_entry_lesson_fifo_eviction() {
        let state = ProofState::new(b"s".to_vec());
        let mut entry = SketchEntry::new(state, vec![]);

        // Fill beyond MAX_LESSONS
        for i in 0..=MAX_LESSONS {
            entry.add_lesson(format!("Lesson {i}"));
        }

        assert_eq!(entry.lesson_count(), MAX_LESSONS);
        // First lesson should be evicted
        assert_eq!(entry.lessons[0], "Lesson 1");
    }

    #[test]
    fn sketch_entry_record_visit() {
        let state = ProofState::new(b"s".to_vec());
        let mut entry = SketchEntry::new(state, vec![]);

        assert_eq!(entry.visits, 0);
        assert!(!entry.is_explored());

        entry.record_visit();
        assert_eq!(entry.visits, 1);
        assert!(entry.is_explored());

        entry.record_visit();
        assert_eq!(entry.visits, 2);
    }

    #[test]
    fn sketch_entry_update_elo() {
        let state = ProofState::new(b"s".to_vec());
        let mut entry = SketchEntry::new(state, vec![]);

        assert_eq!(entry.elo_rating, DEFAULT_ELO);
        entry.update_elo(1350.0);
        assert_eq!(entry.elo_rating, 1350.0);
    }

    #[test]
    fn sketch_entry_pending_goals_cap() {
        let state = ProofState::new(b"s".to_vec());
        let too_many_goals: Vec<Goal> =
            (0..50).map(|i| Goal::from_label(format!("g{i}"))).collect();
        let entry = SketchEntry::new(state, too_many_goals);
        assert_eq!(entry.pending_goal_count(), MAX_PENDING_GOALS);
    }

    #[test]
    fn sketch_entry_p_ucb_score() {
        let state = ProofState::new(b"s".to_vec());
        let mut entry = SketchEntry::new(state, vec![]);
        entry.update_elo(1400.0);

        // Unvisited: high exploration bonus
        let score_unvisited = entry.p_ucb_score(100, 1200.0, 1600.0, 0.2);
        assert!(score_unvisited > 0.0);

        // After visits: lower exploration bonus
        entry.record_visit();
        entry.record_visit();
        let score_visited = entry.p_ucb_score(100, 1200.0, 1600.0, 0.2);
        assert!(score_visited < score_unvisited);
    }

    #[test]
    fn sketch_entry_p_ucb_score_degenerate_range() {
        let state = ProofState::new(b"s".to_vec());
        let entry = SketchEntry::new(state, vec![]);

        // min == max: normalize returns 0.5
        let score = entry.p_ucb_score(0, 1200.0, 1200.0, 0.0);
        assert!((score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn sketch_entry_display() {
        let state = ProofState::new(b"s".to_vec());
        let entry = SketchEntry::new(state, vec![Goal::from_label("g1")]);
        let display = format!("{entry}");
        assert!(display.contains("elo=1200"));
        assert!(display.contains("visits=0"));
        assert!(display.contains("goals=1"));
    }

    // ── Normalize Tests ───────────────────────────────────────

    #[test]
    fn normalize_midrange() {
        let v = normalize_to_01(1400.0, 1200.0, 1600.0);
        assert!((v - 0.5).abs() < 1e-9);
    }

    #[test]
    fn normalize_min() {
        let v = normalize_to_01(1200.0, 1200.0, 1600.0);
        assert!((v - 0.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_max() {
        let v = normalize_to_01(1600.0, 1200.0, 1600.0);
        assert!((v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_degenerate() {
        let v = normalize_to_01(100.0, 100.0, 100.0);
        assert!((v - 0.5).abs() < 1e-9);
    }

    // ── Serialization Tests ───────────────────────────────────

    #[test]
    fn sketch_id_serde_roundtrip() {
        let id = SketchId::new();
        let json = serde_json::to_string(&id).unwrap();
        let restored: SketchId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn goal_serde_roundtrip() {
        let goal = Goal::new("Test Goal", b"\x01\x02".to_vec());
        let json = serde_json::to_string(&goal).unwrap();
        let restored: Goal = serde_json::from_str(&json).unwrap();
        assert_eq!(goal, restored);
    }

    #[test]
    fn diversity_strategy_serde_roundtrip() {
        for strategy in DiversityStrategy::ALL {
            let json = serde_json::to_string(&strategy).unwrap();
            let restored: DiversityStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, restored);
        }
    }

    #[test]
    fn sketch_entry_serde_roundtrip() {
        let state = ProofState::new(b"serde_test".to_vec());
        let mut entry = SketchEntry::new(state, vec![Goal::from_label("g1")]);
        entry.add_lesson("learned something".to_string());
        entry.update_elo(1350.0);
        entry.record_visit();

        let json = serde_json::to_string(&entry).unwrap();
        let restored: SketchEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(entry.id, restored.id);
        assert_eq!(entry.elo_rating, restored.elo_rating);
        assert_eq!(entry.visits, restored.visits);
        assert_eq!(entry.lessons, restored.lessons);
        assert_eq!(entry.pending_goals, restored.pending_goals);
    }
}
