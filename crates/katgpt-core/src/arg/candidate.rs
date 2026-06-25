//! TypedOfflineCandidate + CandidateIntent — ARG Step C (Collection) primitive.
//!
//! Distilled from ARG §3.2 (Typed Candidates). The offline loop's Collection
//! step produces *typed* candidates — never free-form. Each candidate carries:
//!
//! - a `CandidateKind` (Split / Merge / Edge / Taxonomy / NewNode / RegistryDedup)
//! - a `target_label` (the ontology leaf being operated on)
//! - `before` / `after` `LabelSet`s (the structural delta)
//! - `evidence_refs` — IDs of the episodic records that motivated the candidate
//!
//! The candidate itself is *unscored* until [`super::scorer::OfflineCandidateScorer`]
//! evaluates it against resolved evidence. The `score` field is the cache slot
//! for the post-scoring value (`None` until scored).
//!
//! All slices are caller-owned (zero-alloc hot path). The `LabelSet`s are inline
//! bounded (cap 32) — see [`super::taxonomy::LabelSet`].

use super::taxonomy::{LabelId, LabelSet};

/// Stable evidence identifier — references an episodic record. Distinct newtype
/// so it cannot be confused with `LabelId` or array indices.
pub type EvidenceId = u64;

/// ARG §3.2 candidate kinds. The offline loop emits exactly one of these per
/// proposed ontology mutation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CandidateKind {
    /// Split an over-broad leaf into N narrower leaves.
    Split = 0,
    /// Merge N near-duplicate leaves into one (RegistryDedup is the
    /// retrieval-level precursor; Merge is the ontology-level commit).
    Merge = 1,
    /// Add/remove/rewire an edge between two existing leaves.
    Edge = 2,
    /// Taxonomy-level refactor (parent reassignment, kind change).
    #[default]
    Taxonomy = 3,
    /// Mint a brand-new leaf (cold-start or genuine new intent).
    NewNode = 4,
    /// Retrieval-level dedup precursor — two `InfoKey`s resolve to one canonical.
    RegistryDedup = 5,
}

impl CandidateKind {
    /// Returns `true` for kinds that mint new label ids (require id allocation).
    #[inline]
    pub fn mints_new_label(self) -> bool {
        matches!(self, CandidateKind::Split | CandidateKind::NewNode)
    }

    /// Returns `true` for kinds that retire label ids (require redirect entries).
    #[inline]
    pub fn retires_label(self) -> bool {
        matches!(
            self,
            CandidateKind::Merge | CandidateKind::RegistryDedup
        )
    }
}

/// The structural delta + provenance of an offline candidate.
///
/// `before` / `after` are the `LabelSet`s *as the candidate sees them* — for a
/// `Split`, `before` is the single over-broad leaf and `after` is the N narrower
/// leaves; for a `Merge`, the reverse. The exact encoding is caller-defined; the
/// validator (Step D) enforces structural coherence.
///
/// `evidence_refs` is a caller-owned slice of [`EvidenceId`]s — the scorer does
/// NOT resolve these; the caller resolves them to [`super::scorer::Evidence`]
/// before invoking the scorer.
#[derive(Clone, Debug)]
pub struct CandidateIntent<'a> {
    pub kind: CandidateKind,
    pub target_label: LabelId,
    pub before: LabelSet,
    pub after: LabelSet,
    pub evidence_refs: &'a [EvidenceId],
}

impl<'a> CandidateIntent<'a> {
    /// Number of evidence references backing this candidate.
    #[inline]
    pub fn evidence_count(&self) -> usize {
        self.evidence_refs.len()
    }
}

/// A typed offline candidate — intent + an optional cached score.
///
/// The `score` slot is `None` until the candidate passes through
/// [`super::scorer::OfflineCandidateScorer::score`]; the caller fills it in
/// after scoring if they want to persist the cached value. The scorer itself
/// is pure and does not mutate the candidate.
#[derive(Clone, Debug)]
pub struct TypedOfflineCandidate<'a> {
    pub intent: CandidateIntent<'a>,
    /// Cached score — `None` until scored. Set by the caller post-scoring.
    pub score: Option<f32>,
}

impl<'a> TypedOfflineCandidate<'a> {
    /// Construct an unscored candidate.
    #[inline]
    pub fn new(intent: CandidateIntent<'a>) -> Self {
        TypedOfflineCandidate {
            intent,
            score: None,
        }
    }

    /// Construct an unscored candidate with the minimal fields.
    #[inline]
    pub fn bare(
        kind: CandidateKind,
        target_label: LabelId,
        evidence_refs: &'a [EvidenceId],
    ) -> Self {
        TypedOfflineCandidate {
            intent: CandidateIntent {
                kind,
                target_label,
                before: LabelSet::new(),
                after: LabelSet::new(),
                evidence_refs,
            },
            score: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lbl(n: u32) -> LabelId {
        LabelId::new(n)
    }

    #[test]
    fn candidate_kind_mints_and_retires_partitions_all_variants() {
        // Every variant must be classified by exactly one of the two predicates
        // (or neither) — sanity check that we didn't misclassify one.
        let all = [
            CandidateKind::Split,
            CandidateKind::Merge,
            CandidateKind::Edge,
            CandidateKind::Taxonomy,
            CandidateKind::NewNode,
            CandidateKind::RegistryDedup,
        ];
        for k in all {
            let mints = k.mints_new_label();
            let retires = k.retires_label();
            // Mints and retires are disjoint by construction (a candidate that
            // both mints and retires would be two candidates).
            assert!(
                !(mints && retires),
                "kind {:?} cannot both mint and retire",
                k
            );
        }
        // Spot-check the specific classifications.
        assert!(CandidateKind::Split.mints_new_label());
        assert!(CandidateKind::NewNode.mints_new_label());
        assert!(!CandidateKind::Merge.mints_new_label());
        assert!(CandidateKind::Merge.retires_label());
        assert!(CandidateKind::RegistryDedup.retires_label());
        assert!(!CandidateKind::Edge.retires_label());
        assert!(!CandidateKind::Taxonomy.mints_new_label());
        assert!(!CandidateKind::Taxonomy.retires_label());
    }

    #[test]
    fn default_kind_is_taxonomy() {
        // The #[default] attribute makes Taxonomy the zero value — important for
        // `MaybeUninit` / Pod-style layouts and for `Default::default()`.
        assert_eq!(CandidateKind::default(), CandidateKind::Taxonomy);
        assert_eq!(CandidateKind::Taxonomy as u8, 3u8);
    }

    #[test]
    fn bare_candidate_is_unscored_with_empty_label_sets() {
        let refs = [EvidenceId::from(1u64), EvidenceId::from(2u64)];
        let c = TypedOfflineCandidate::bare(CandidateKind::Edge, lbl(7), &refs);
        assert_eq!(c.intent.kind, CandidateKind::Edge);
        assert_eq!(c.intent.target_label, lbl(7));
        assert!(c.intent.before.is_empty());
        assert!(c.intent.after.is_empty());
        assert_eq!(c.intent.evidence_count(), 2);
        assert!(c.score.is_none());
    }

    #[test]
    fn candidate_with_before_after_label_sets() {
        let mut before = LabelSet::new();
        before.insert(lbl(1));
        let mut after = LabelSet::new();
        after.insert(lbl(10));
        after.insert(lbl(11));
        let refs: &[EvidenceId] = &[];
        let intent = CandidateIntent {
            kind: CandidateKind::Split,
            target_label: lbl(1),
            before,
            after,
            evidence_refs: refs,
        };
        let c = TypedOfflineCandidate::new(intent);
        assert_eq!(c.intent.kind, CandidateKind::Split);
        assert_eq!(c.intent.before.len(), 1);
        assert_eq!(c.intent.after.len(), 2);
        assert_eq!(c.intent.evidence_count(), 0);
        assert!(c.score.is_none());
        // Split mints new labels (the after-set leaves).
        assert!(c.intent.kind.mints_new_label());
    }
}
