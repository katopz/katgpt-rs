//! Episode-Guided Constraint Synthesis (EGCS) — Plan 206.
//!
//! Wraps any inner `ConstraintPruner` and adds episode-guided constraint synthesis.
//! When a reference solution exists in an episode DB, diffs the candidate against
//! the reference and synthesizes structural constraints.
//!
//! # Architecture
//!
//! ```text
//! DDTree → EpisodePruner → inner.is_valid()
//!                    ↓
//!              EpisodeLookup → reference?
//!                    ↓ yes
//!              ConstraintSynthesizer → structural diff → constraints
//!                    ↓
//!              synthesized.is_valid() AND inner.is_valid()
//! ```
//!
//! Zero cost on miss path (no episode found → inner pruner only).
//! Feature-gated behind `egcs`.

use crate::speculative::types::ConstraintPruner;

// ── Synthesized Constraint ────────────────────────────────────────

/// Synthesized constraint from structural diff.
///
/// Constrains token choices at specific position ranges based on
/// how the candidate diverges from a verified reference solution.
#[cfg(feature = "egcs")]
#[derive(Clone, Debug)]
pub struct SynthesizedConstraint {
    /// Position range where constraint applies [start, end).
    pub position_range: (usize, usize),
    /// Token indices allowed at these positions (empty = no restriction).
    pub allowed_tokens: Vec<usize>,
    /// Token indices disallowed at these positions.
    pub disallowed_tokens: Vec<usize>,
}

// ── ConstraintSynthesizer Trait ───────────────────────────────────

/// Trait for synthesizing constraints from structural diffs.
///
/// Given a candidate token sequence and a reference sequence, produces
/// position-level constraints that guide future candidates toward the
/// reference where they diverge.
#[cfg(feature = "egcs")]
pub trait ConstraintSynthesizer: Send + Sync {
    /// Compare candidate against reference, produce constraints.
    fn synthesize(&self, candidate: &[usize], reference: &[usize]) -> Vec<SynthesizedConstraint>;
}

// ── Episode Types ─────────────────────────────────────────────────

/// A reference solution from the episode database.
#[cfg(feature = "egcs")]
#[derive(Clone, Debug)]
pub struct Episode {
    /// Hash of the prompt that produced this episode.
    pub prompt_hash: u64,
    /// Reference token sequence.
    pub reference_tokens: Vec<usize>,
    /// Optional metadata (success rate, etc.).
    pub metadata: EpisodeMetadata,
}

/// Metadata about an episode's verification history.
#[cfg(feature = "egcs")]
#[derive(Clone, Debug, Default)]
pub struct EpisodeMetadata {
    /// How many times this episode was verified correct.
    pub verification_count: usize,
    /// Average acceptance rate when using this episode as reference.
    pub avg_acceptance: f32,
}

// ── EpisodeLookup Trait ───────────────────────────────────────────

/// Trait for looking up reference solutions.
///
/// Abstracts over any backend (in-memory, SQLite, vector DB).
/// The default `lookup_similar` returns empty — override for embedding-based retrieval.
#[cfg(feature = "egcs")]
pub trait EpisodeLookup: Send + Sync {
    /// Exact lookup by prompt hash.
    fn lookup(&self, prompt_hash: u64) -> Option<Episode>;

    /// Find similar episodes by embedding proximity (optional, default: empty).
    fn lookup_similar(&self, _prompt_embedding: &[f32], _k: usize) -> Vec<Episode> {
        Vec::new()
    }
}

// ── StructuralDiffSynthesizer ─────────────────────────────────────

/// Default `ConstraintSynthesizer`: token-level structural diff.
///
/// Compares candidate vs reference position-by-position:
/// - **Agree** (same token at same position) → no constraint needed
/// - **Disagree** (different token) → constrain to reference's token
/// - **Candidate longer** (len > ref) → no constraint (can't guide beyond reference)
/// - **Reference longer** (ref len > candidate) → no constraint for future positions
#[cfg(feature = "egcs")]
#[derive(Clone, Debug, Default)]
pub struct StructuralDiffSynthesizer;

#[cfg(feature = "egcs")]
impl ConstraintSynthesizer for StructuralDiffSynthesizer {
    fn synthesize(&self, candidate: &[usize], reference: &[usize]) -> Vec<SynthesizedConstraint> {
        let min_len = candidate.len().min(reference.len());
        let mut constraints = Vec::new();

        for pos in 0..min_len {
            match candidate[pos] == reference[pos] {
                true => continue, // agree — no constraint
                false => {
                    // disagree — constrain to reference's token
                    constraints.push(SynthesizedConstraint {
                        position_range: (pos, pos + 1),
                        allowed_tokens: vec![reference[pos]],
                        disallowed_tokens: vec![candidate[pos]],
                    });
                }
            }
        }

        constraints
    }
}

// ── EpisodePruner ─────────────────────────────────────────────────

/// Episode-guided constraint pruner — wraps inner pruner with reference-based constraints.
///
/// When a reference solution exists in the episode DB, synthesizes position-level
/// constraints from structural diff against the reference. Falls back to inner
/// pruner alone when no episode is found (zero-cost miss path).
///
/// Feature-gated behind `egcs`.
#[cfg(feature = "egcs")]
pub struct EpisodePruner<P: ConstraintPruner, L: EpisodeLookup, S: ConstraintSynthesizer> {
    /// Inner pruner (base structural validity).
    inner: P,
    /// Episode lookup backend.
    _lookup: L,
    /// Constraint synthesizer (diff → constraints).
    _synthesizer: S,
    /// Cached synthesized constraints, keyed by prompt hash.
    constraint_cache: Vec<(u64, Vec<SynthesizedConstraint>)>,
    /// Maximum cache entries before eviction (LRU-style, default 64).
    max_cache: usize,
    /// Current prompt hash being processed.
    current_prompt_hash: u64,
}

#[cfg(feature = "egcs")]
impl<P: ConstraintPruner, L: EpisodeLookup, S: ConstraintSynthesizer> EpisodePruner<P, L, S> {
    /// Create a new `EpisodePruner` wrapping `inner` with episode lookup and synthesizer.
    pub fn new(inner: P, lookup: L, synthesizer: S) -> Self {
        Self {
            inner,
            _lookup: lookup,
            _synthesizer: synthesizer,
            constraint_cache: Vec::new(),
            max_cache: 64,
            current_prompt_hash: 0,
        }
    }

    /// Set maximum cache entries (evicts oldest when exceeded).
    pub fn with_max_cache(mut self, max_cache: usize) -> Self {
        self.max_cache = max_cache.max(1);
        self
    }

    /// Set the current prompt hash for episode lookup.
    ///
    /// Call this before each generation pass. The hash is used to look up
    /// reference solutions in the episode DB.
    pub fn set_prompt(&mut self, hash: u64) {
        self.current_prompt_hash = hash;
    }

    /// Cache-aware constraint synthesis.
    ///
    /// Returns synthesized constraints for the current prompt hash, using the
    /// cache if available or synthesizing from the episode DB reference.
    /// Where candidate and reference agree, no constraints are emitted.
    #[allow(dead_code)]
    pub(crate) fn get_or_synthesize(&mut self, candidate: &[usize]) -> &[SynthesizedConstraint] {
        let hash = self.current_prompt_hash;

        // Check cache first
        if let Some(idx) = self.constraint_cache.iter().position(|(h, _)| *h == hash) {
            return &self.constraint_cache[idx].1;
        }

        // Lookup episode and synthesize
        let constraints = match self._lookup.lookup(hash) {
            Some(episode) => self
                ._synthesizer
                .synthesize(candidate, &episode.reference_tokens),
            None => Vec::new(),
        };

        // Evict oldest if at capacity
        if self.constraint_cache.len() >= self.max_cache {
            self.constraint_cache.remove(0);
        }

        self.constraint_cache.push((hash, constraints));

        // Return the last entry (just pushed)
        &self.constraint_cache.last().expect("just pushed").1
    }

    /// Check if a token at a given depth is disallowed by any synthesized constraint.
    fn is_disallowed_by_synthesis(
        constraints: &[SynthesizedConstraint],
        depth: usize,
        token_idx: usize,
    ) -> bool {
        for c in constraints {
            match depth >= c.position_range.0 && depth < c.position_range.1 {
                false => continue,
                true => {
                    // If allowed_tokens is non-empty, token must be in it
                    if !c.allowed_tokens.is_empty() && !c.allowed_tokens.contains(&token_idx) {
                        return true;
                    }
                    // If token is explicitly disallowed, reject
                    if c.disallowed_tokens.contains(&token_idx) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(feature = "egcs")]
impl<P: ConstraintPruner, L: EpisodeLookup, S: ConstraintSynthesizer> ConstraintPruner
    for EpisodePruner<P, L, S>
{
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Inner pruner is the base gate
        if !self.inner.is_valid(depth, token_idx, parent_tokens) {
            return false;
        }

        // Check cached constraints (read-only borrow)
        for (_, constraints) in &self.constraint_cache {
            if Self::is_disallowed_by_synthesis(constraints, depth, token_idx) {
                return false;
            }
        }

        true
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        // Delegate to inner batch first
        self.inner
            .batch_is_valid(depth, candidates, parent_tokens, results);

        // Apply synthesized constraints in batch
        let len = candidates.len().min(results.len());
        for (_, constraints) in &self.constraint_cache {
            for i in 0..len {
                match results[i] {
                    false => continue, // already rejected by inner
                    true => {
                        if Self::is_disallowed_by_synthesis(constraints, depth, candidates[i]) {
                            results[i] = false;
                        }
                    }
                }
            }
        }
    }
}

// ── MemoryEpisodeLookup ───────────────────────────────────────────

/// Simple in-memory episode store for testing and demos.
///
/// Stores episodes in a `Vec` with linear lookup by prompt hash.
/// Not suitable for production (use a HashMap or SQLite backend).
#[cfg(feature = "egcs")]
pub struct MemoryEpisodeLookup {
    episodes: Vec<Episode>,
}

#[cfg(feature = "egcs")]
impl MemoryEpisodeLookup {
    /// Create an empty episode store.
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
        }
    }

    /// Add an episode to the store.
    pub fn insert(&mut self, episode: Episode) {
        self.episodes.push(episode);
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

#[cfg(feature = "egcs")]
impl Default for MemoryEpisodeLookup {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "egcs")]
impl EpisodeLookup for MemoryEpisodeLookup {
    fn lookup(&self, prompt_hash: u64) -> Option<Episode> {
        for ep in &self.episodes {
            match ep.prompt_hash == prompt_hash {
                true => return Some(ep.clone()),
                false => continue,
            }
        }
        None
    }

    fn lookup_similar(&self, _prompt_embedding: &[f32], _k: usize) -> Vec<Episode> {
        // Linear scan not suitable for embedding search — override with proper backend
        Vec::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Pruner that accepts everything (for testing episode layer in isolation).
    struct AcceptAllPruner;

    impl ConstraintPruner for AcceptAllPruner {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            true
        }
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_episode_pruner_no_episode_fallback() {
        // No episode in DB → behaves exactly like inner pruner (accept all)
        let lookup = MemoryEpisodeLookup::new();
        let synthesizer = StructuralDiffSynthesizer;
        let pruner = EpisodePruner::new(AcceptAllPruner, lookup, synthesizer);

        // No prompt set → no constraints → accept everything
        assert!(pruner.is_valid(0, 42, &[]));
        assert!(pruner.is_valid(5, 0, &[1, 2, 3, 4, 5]));
        assert!(pruner.is_valid(100, 999, &[]));
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_episode_pruner_with_reference() {
        // Insert an episode with reference tokens [10, 20, 30]
        let mut lookup = MemoryEpisodeLookup::new();
        lookup.insert(Episode {
            prompt_hash: 123,
            reference_tokens: vec![10, 20, 30],
            metadata: EpisodeMetadata::default(),
        });

        let synthesizer = StructuralDiffSynthesizer;
        let mut pruner = EpisodePruner::new(AcceptAllPruner, lookup, synthesizer);
        pruner.set_prompt(123);

        // Synthesize constraints with candidate [10, 25, 30]
        // Position 1 disagrees: candidate=25, reference=20 → allowed=[20], disallowed=[25]
        let constraints = pruner.get_or_synthesize(&[10, 25, 30]);
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].position_range, (1, 2));
        assert_eq!(constraints[0].allowed_tokens, vec![20]);
        assert_eq!(constraints[0].disallowed_tokens, vec![25]);

        // Inner accepts everything, but synthesized constraint restricts position 1
        assert!(pruner.is_valid(0, 10, &[])); // position 0: no constraint
        assert!(!pruner.is_valid(1, 25, &[10])); // position 1: disallowed=25
        assert!(pruner.is_valid(1, 20, &[10])); // position 1: allowed=20
        assert!(pruner.is_valid(2, 30, &[10, 20])); // position 2: no constraint
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_constraint_synthesizer_basic() {
        let synth = StructuralDiffSynthesizer;

        // candidate: [1, 2, 3, 4]
        // reference: [1, 9, 3, 5]
        // Position 0: agree → no constraint
        // Position 1: disagree (2 vs 9) → allowed=[9], disallowed=[2]
        // Position 2: agree → no constraint
        // Position 3: disagree (4 vs 5) → allowed=[5], disallowed=[4]
        let constraints = synth.synthesize(&[1, 2, 3, 4], &[1, 9, 3, 5]);

        assert_eq!(constraints.len(), 2);

        assert_eq!(constraints[0].position_range, (1, 2));
        assert_eq!(constraints[0].allowed_tokens, vec![9]);
        assert_eq!(constraints[0].disallowed_tokens, vec![2]);

        assert_eq!(constraints[1].position_range, (3, 4));
        assert_eq!(constraints[1].allowed_tokens, vec![5]);
        assert_eq!(constraints[1].disallowed_tokens, vec![4]);
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_structural_diff_identical() {
        let synth = StructuralDiffSynthesizer;

        // Identical sequences → no constraints
        let constraints = synth.synthesize(&[1, 2, 3], &[1, 2, 3]);
        assert!(constraints.is_empty());

        // Empty sequences → no constraints
        let constraints = synth.synthesize(&[], &[]);
        assert!(constraints.is_empty());
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_structural_diff_disjoint() {
        let synth = StructuralDiffSynthesizer;

        // Completely different → constraints at all positions
        let constraints = synth.synthesize(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(constraints.len(), 3);

        assert_eq!(constraints[0].allowed_tokens, vec![4]);
        assert_eq!(constraints[1].allowed_tokens, vec![5]);
        assert_eq!(constraints[2].allowed_tokens, vec![6]);
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_cache_reuse() {
        let mut lookup = MemoryEpisodeLookup::new();
        lookup.insert(Episode {
            prompt_hash: 42,
            reference_tokens: vec![1, 2, 3],
            metadata: EpisodeMetadata::default(),
        });

        let mut pruner = EpisodePruner::new(AcceptAllPruner, lookup, StructuralDiffSynthesizer);

        // First synthesis — cache miss → synthesizes
        pruner.set_prompt(42);
        let c1 = pruner.get_or_synthesize(&[1, 5, 3]).to_vec();
        assert_eq!(c1.len(), 1); // position 1 disagrees

        // Second synthesis — same hash → cache hit (same result)
        let c2 = pruner.get_or_synthesize(&[1, 99, 3]);
        assert_eq!(c1.len(), c2.len());
        assert_eq!(c1[0].position_range, c2[0].position_range);

        // Different hash → no episode → empty constraints
        pruner.set_prompt(999);
        let c3 = pruner.get_or_synthesize(&[1, 2, 3]);
        assert!(c3.is_empty());
    }
}
