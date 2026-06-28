//! TaxonomyValidator — ARG Step 3 deterministic label-set validator.
//!
//! Distilled from ARG §0.1 (taxonomy `cluster ↔ label`), §OW-5 (strict taxonomy
//! validation). The validator is the *final arbiter* of label-set validity:
//! labels must exist, be compatible with their cluster, respect parent/child
//! coherence, and honor explicit incompatibilities. Output: `L_valid` (a
//! `LabelSet` containing only labels that passed every check) + per-label
//! rejection reasons in the scratch buffer.
//!
//! Pure tree-walk over a sorted `&[TaxonomyNode]` slice. Zero-alloc when the
//! caller pre-allocates a `ValidationScratch`.

use core::cmp::Ordering;

/// Stable label identifier — never recycled, never reused after `Removed`.
///
/// Wraps `u32` so it cannot be silently confused with raw indices. The ARG
/// §"stable identity" rule mandates that ids survive split/merge via
/// `RedirectTable` redirects — ids themselves are never reused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct LabelId(u32);

impl LabelId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        LabelId(id)
    }
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Maximum size of a candidate label set before escalation. ARG §OW-2 caps
/// Top-K at 10–20; we use 32 as a hard ceiling for the inline-array LabelSet.
pub const LABEL_SET_CAPACITY: usize = 32;

/// Bounded inline label set — no heap allocation up to `LABEL_SET_CAPACITY`.
///
/// Modeled after the smallvec pattern but specialized to `LabelId`. Used as
/// the candidate input (`L_union`), the validated output (`L_final`), and the
/// expanded set after ascending propagation (`L_expanded`).
#[derive(Clone, Debug, Default)]
pub struct LabelSet {
    ids: [LabelId; LABEL_SET_CAPACITY],
    len: u8,
}

impl LabelSet {
    /// Empty set.
    #[inline]
    pub const fn new() -> Self {
        LabelSet {
            ids: [LabelId::new(0); LABEL_SET_CAPACITY],
            len: 0,
        }
    }

    /// Current number of labels in the set.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` if the set contains no labels.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the set contains `label`.
    #[inline]
    pub fn contains(&self, label: LabelId) -> bool {
        self.as_slice().contains(&label)
    }

    /// View as a borrowed slice.
    #[inline]
    pub fn as_slice(&self) -> &[LabelId] {
        &self.ids[..self.len()]
    }

    /// Insert `label` if there is capacity and the label is not already present.
    /// Returns `true` if inserted, `false` if full or duplicate.
    pub fn insert(&mut self, label: LabelId) -> bool {
        if self.len() == LABEL_SET_CAPACITY {
            return false;
        }
        if self.contains(label) {
            return false;
        }
        self.ids[self.len()] = label;
        self.len += 1;
        true
    }

    /// Clear without dropping the backing array.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

/// Taxonomy node kind. ARG §0.1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TaxonomyKind {
    /// Top-level domain root.
    #[default]
    Cluster = 0,
    /// Hierarchical routing branch (`root → parent → child`).
    Label = 1,
    /// Terminal operational unit (concept, action, info, memory behavior).
    /// In the ARG "branches & leaves" model this is the carrier of `title + chunk`.
    Leaf = 2,
}

/// A single taxonomy node.
///
/// Borrowed fields (`incompatible_with`) so the caller owns the static
/// taxonomy definition; validators are constructed over a `&[TaxonomyNode]`
/// slice sorted by `id` for O(log n) binary-search lookup.
#[derive(Clone, Copy, Debug)]
pub struct TaxonomyNode<'a> {
    pub id: LabelId,
    pub kind: TaxonomyKind,
    pub parent_id: Option<LabelId>,
    /// Labels that must never be co-active with this node (ARG §OW-5 explicit
    /// incompatibilities). Caller-owned slice.
    pub incompatible_with: &'a [LabelId],
}

/// Why a candidate label was rejected by validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// Label id does not exist in the taxonomy.
    NotFound,
    /// Label is marked `Removed` (must redirect via `RedirectTable` first).
    Removed,
    /// Two candidates are mutually incompatible per `incompatible_with`.
    Incompatible(LabelId, LabelId),
    /// A child label was present without its parent (coherence violation).
    MissingParent(LabelId),
}

/// Result of validating a candidate set.
#[derive(Clone, Debug, Default)]
pub struct ValidationResult {
    /// The valid subset of the input candidates (`L_valid`).
    pub valid: LabelSet,
    /// Rejections with reasons (for audit logging / offline refinement).
    pub rejections: Vec<ValidationError>,
}

/// Pre-allocated scratch buffer for validation. Reuse across calls for
/// zero-alloc hot-path operation.
#[derive(Clone, Debug, Default)]
pub struct ValidationScratch {
    /// Rejections buffer (cleared per call).
    pub rejections: Vec<ValidationError>,
    /// Visited-set for ascending expansion (cleared per call).
    pub visited: Vec<LabelId>,
    /// Accepted-candidates buffer for `validate_label_set` (cleared per call).
    /// Holds the candidate ids that passed existence check, for the cross-
    /// validation passes (incompatibility, parent/child coherence).
    pub accepted: Vec<LabelId>,
}

impl ValidationScratch {
    /// Pre-allocate capacity for `n` rejections, `n` visited ids, and `n`
    /// accepted candidates.
    pub fn with_capacity(n: usize) -> Self {
        ValidationScratch {
            rejections: Vec::with_capacity(n),
            visited: Vec::with_capacity(n),
            accepted: Vec::with_capacity(n),
        }
    }

    fn clear(&mut self) {
        self.rejections.clear();
        self.visited.clear();
        self.accepted.clear();
    }
}

/// Deterministic validator over a sorted taxonomy slice.
///
/// The taxonomy MUST be sorted by `TaxonomyNode::id` (ascending) for the
/// binary-search lookups to be valid. Construct via [`TaxonomyValidator::new`]
/// which sorts defensively and panics on duplicate ids (taxonomies must not
/// contain duplicates).
#[derive(Clone, Debug)]
pub struct TaxonomyValidator<'a> {
    nodes: Vec<TaxonomyNode<'a>>,
}

impl<'a> TaxonomyValidator<'a> {
    /// Construct from a (possibly unsorted) slice of taxonomy nodes. Sorts by
    /// `id`. Panics on duplicate ids.
    pub fn new(mut nodes: Vec<TaxonomyNode<'a>>) -> Self {
        nodes.sort_by_key(|n| n.id);
        // Detect duplicates.
        for w in nodes.windows(2) {
            if w[0].id == w[1].id {
                panic!("TaxonomyValidator: duplicate LabelId {:?}", w[0].id);
            }
        }
        TaxonomyValidator { nodes }
    }

    /// Lookup a node by id (binary search, O(log n)).
    #[inline]
    pub fn find(&self, id: LabelId) -> Option<&TaxonomyNode<'a>> {
        self.nodes
            .binary_search_by_key(&id, |n| n.id)
            .ok()
            .map(|i| &self.nodes[i])
    }

    /// Validate a candidate label set against the taxonomy.
    ///
    /// Returns the valid subset (`L_valid`) plus per-label rejection reasons
    /// in `scratch`. The caller owns the scratch; reuse across calls for
    /// zero-alloc operation.
    pub fn validate_label_set(
        &self,
        candidates: &LabelSet,
        scratch: &mut ValidationScratch,
    ) -> ValidationResult {
        scratch.clear();
        let mut valid = LabelSet::new();
        let cands = candidates.as_slice();

        // Pass 1: existence check + collect accepted nodes into scratch.accepted
        // (reusable buffer — no per-call allocation). Bounded by LABEL_SET_CAPACITY.
        let accepted = &mut scratch.accepted;
        for &c in cands {
            match self.find(c) {
                None => scratch.rejections.push(ValidationError::NotFound),
                Some(node) => {
                    // Removed labels must be redirected upstream — reject here.
                    // (LifecycleState check is layered on top in arg_runtime.)
                    if node.kind == TaxonomyKind::Leaf && node.parent_id.is_none() {
                        // Top-level leaves without parent are allowed (root-level leaves).
                        accepted.push(c);
                        valid.insert(c);
                    } else {
                        accepted.push(c);
                        valid.insert(c);
                    }
                }
            }
        }

        // Pass 2: explicit incompatibilities (O(|accepted|^2 × avg incompatible size)).
        // Bounded: |accepted| ≤ LABEL_SET_CAPACITY = 32.
        for (i, &a) in accepted.iter().enumerate() {
            let node_a = match self.find(a) {
                Some(n) => n,
                None => continue, // already rejected above
            };
            for &b in accepted.iter().skip(i + 1) {
                if node_a.incompatible_with.contains(&b) {
                    scratch
                        .rejections
                        .push(ValidationError::Incompatible(a, b));
                    // Remove both from valid — incompatibility is symmetric.
                    valid = remove_from_set(valid, a);
                    valid = remove_from_set(valid, b);
                }
            }
        }

        // Pass 3: parent/child coherence — every Label/Leaf with a parent
        // must have its parent implicitly active. ARG §OW-5 requires this.
        for &c in accepted.iter() {
            let node = match self.find(c) {
                Some(n) => n,
                None => continue,
            };
            if let Some(parent_id) = node.parent_id
                && !valid.contains(parent_id) {
                    scratch
                        .rejections
                        .push(ValidationError::MissingParent(parent_id));
                    // Drop the orphan child (do NOT auto-add the parent — ARG
                    // ascending expansion is a separate step (expand_ascending)).
                    valid = remove_from_set(valid, c);
                }
        }

        // The result owns its rejections Vec; the scratch keeps its capacity for
        // the next call (we clone the slice, not mem::take — so scratch.rejections
        // retains its allocation). For the common (no-rejection) case, cloning an
        // empty slice returns Vec::new() (no allocation).
        ValidationResult {
            valid,
            rejections: scratch.rejections.clone(),
        }
    }

    /// Ascending-only hierarchical expansion (ARG §OW-6): for each label in
    /// `leaf_set`, add its parent chain `child → parent → root`. Never descends.
    pub fn expand_ascending(
        &self,
        leaf_set: &LabelSet,
        scratch: &mut ValidationScratch,
    ) -> LabelSet {
        scratch.clear();
        let mut out = LabelSet::new();
        for &c in leaf_set.as_slice() {
            let mut cursor = Some(c);
            while let Some(cur) = cursor {
                if out.contains(cur) {
                    break; // already added
                }
                if !out.insert(cur) {
                    break; // capacity hit
                }
                scratch.visited.push(cur);
                cursor = self.find(cur).and_then(|n| n.parent_id);
            }
        }
        out
    }

    /// Number of taxonomy nodes (for diagnostics).
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the taxonomy is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Helper: rebuild a `LabelSet` minus a specific id (used during rejection).
fn remove_from_set(mut set: LabelSet, id: LabelId) -> LabelSet {
    if !set.contains(id) {
        return set;
    }
    let mut new_set = LabelSet::new();
    for &l in set.as_slice() {
        if l != id {
            new_set.insert(l);
        }
    }
    core::mem::swap(&mut set, &mut new_set);
    set
}

// Re-export TaxonomyKind Ord instance for stable serialization (future-proof).
impl Ord for TaxonomyKind {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}
impl PartialOrd for TaxonomyKind {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lbl(n: u32) -> LabelId {
        LabelId::new(n)
    }

    fn node(id: u32, kind: TaxonomyKind, parent: Option<u32>) -> TaxonomyNode<'static> {
        TaxonomyNode {
            id: lbl(id),
            kind,
            parent_id: parent.map(lbl),
            incompatible_with: &[],
        }
    }

    // Static slice so the `incompatible_with` borrow is `'static`.
    // Leaf(4) is incompatible with Leaf(5).
    static LEAF4_INCOMPATIBLE: [LabelId; 1] = [LabelId::new(5)];

    fn sample_taxonomy() -> Vec<TaxonomyNode<'static>> {
        // Cluster(1) ─── Label(2) ─── Leaf(3)
        //                                     └── Leaf(4) [incompatible_with Leaf(5)]
        //            └── Label(6) ─── Leaf(5)
        // Cluster(7) ─── Label(8) ─── Leaf(9)
        vec![
            node(1, TaxonomyKind::Cluster, None),
            node(2, TaxonomyKind::Label, Some(1)),
            node(3, TaxonomyKind::Leaf, Some(2)),
            TaxonomyNode {
                id: lbl(4),
                kind: TaxonomyKind::Leaf,
                parent_id: Some(lbl(2)),
                incompatible_with: &LEAF4_INCOMPATIBLE, // Leaf(4) ⊥ Leaf(5)
            },
            node(6, TaxonomyKind::Label, Some(1)),
            node(5, TaxonomyKind::Leaf, Some(6)),
            node(7, TaxonomyKind::Cluster, None),
            node(8, TaxonomyKind::Label, Some(7)),
            node(9, TaxonomyKind::Leaf, Some(8)),
        ]
    }

    #[test]
    fn find_uses_binary_search() {
        let v = TaxonomyValidator::new(sample_taxonomy());
        assert!(v.find(lbl(1)).is_some());
        assert!(v.find(lbl(9)).is_some());
        assert!(v.find(lbl(99)).is_none());
    }

    #[test]
    fn duplicate_ids_panic() {
        let dup = vec![node(1, TaxonomyKind::Cluster, None), node(1, TaxonomyKind::Label, None)];
        let result = std::panic::catch_unwind(|| {
            TaxonomyValidator::new(dup);
        });
        assert!(result.is_err(), "duplicate ids must panic");
    }

    #[test]
    fn validate_rejects_missing_label() {
        let v = TaxonomyValidator::new(sample_taxonomy());
        let mut scratch = ValidationScratch::with_capacity(8);
        let mut candidates = LabelSet::new();
        candidates.insert(lbl(99)); // not in taxonomy
        let r = v.validate_label_set(&candidates, &mut scratch);
        assert!(r.valid.is_empty());
        assert!(r.rejections.iter().any(|e| matches!(e, ValidationError::NotFound)));
    }

    #[test]
    fn validate_rejects_incompatible_pair() {
        let v = TaxonomyValidator::new(sample_taxonomy());
        let mut scratch = ValidationScratch::with_capacity(8);
        // Include parents so the incompatibility is the only rejection cause.
        let mut candidates = LabelSet::new();
        candidates.insert(lbl(1)); // Cluster
        candidates.insert(lbl(2)); // Label(2) parent of Leaf(4)
        candidates.insert(lbl(6)); // Label(6) parent of Leaf(5)
        candidates.insert(lbl(4)); // Leaf(4) ⊥ Leaf(5)
        candidates.insert(lbl(5)); // Leaf(5)
        let r = v.validate_label_set(&candidates, &mut scratch);
        // Both Leaf(4) and Leaf(5) must be rejected.
        assert!(!r.valid.contains(lbl(4)));
        assert!(!r.valid.contains(lbl(5)));
        assert!(r.rejections
            .iter()
            .any(|e| matches!(e, ValidationError::Incompatible(a, b) if (*a == lbl(4) && *b == lbl(5)) || (*a == lbl(5) && *b == lbl(4)))));
    }

    #[test]
    fn validate_rejects_orphan_child_without_parent() {
        let v = TaxonomyValidator::new(sample_taxonomy());
        let mut scratch = ValidationScratch::with_capacity(8);
        // Leaf(3) without its parent Label(2).
        let mut candidates = LabelSet::new();
        candidates.insert(lbl(3));
        let r = v.validate_label_set(&candidates, &mut scratch);
        assert!(!r.valid.contains(lbl(3)));
        assert!(r.rejections
            .iter()
            .any(|e| matches!(e, ValidationError::MissingParent(p) if *p == lbl(2))));
    }

    #[test]
    fn validate_accepts_when_parent_present() {
        let v = TaxonomyValidator::new(sample_taxonomy());
        let mut scratch = ValidationScratch::with_capacity(8);
        let mut candidates = LabelSet::new();
        candidates.insert(lbl(1));
        candidates.insert(lbl(2));
        candidates.insert(lbl(3));
        let r = v.validate_label_set(&candidates, &mut scratch);
        assert!(r.valid.contains(lbl(3)));
    }

    #[test]
    fn expand_ascending_adds_parent_chain() {
        let v = TaxonomyValidator::new(sample_taxonomy());
        let mut scratch = ValidationScratch::with_capacity(8);
        let mut leaves = LabelSet::new();
        leaves.insert(lbl(3)); // Leaf(3) → Label(2) → Cluster(1)
        let expanded = v.expand_ascending(&leaves, &mut scratch);
        // Should contain Leaf(3), Label(2), Cluster(1).
        assert!(expanded.contains(lbl(3)));
        assert!(expanded.contains(lbl(2)));
        assert!(expanded.contains(lbl(1)));
    }

    #[test]
    fn expand_ascending_never_descends() {
        let v = TaxonomyValidator::new(sample_taxonomy());
        let mut scratch = ValidationScratch::with_capacity(8);
        let mut leaves = LabelSet::new();
        leaves.insert(lbl(2)); // start from a Label, not a Leaf
        let expanded = v.expand_ascending(&leaves, &mut scratch);
        // Should NOT pull in Leaf(3) / Leaf(4) (children of Label(2)).
        assert!(expanded.contains(lbl(2)));
        assert!(expanded.contains(lbl(1))); // parent
        assert!(!expanded.contains(lbl(3)));
        assert!(!expanded.contains(lbl(4)));
    }

    #[test]
    fn label_set_capacity_enforced() {
        let mut set = LabelSet::new();
        for i in 0..LABEL_SET_CAPACITY as u32 {
            assert!(set.insert(lbl(i)));
        }
        // 33rd insert fails.
        assert!(!set.insert(lbl(LABEL_SET_CAPACITY as u32)));
        assert_eq!(set.len(), LABEL_SET_CAPACITY);
    }

    #[test]
    fn label_set_dedupes() {
        let mut set = LabelSet::new();
        assert!(set.insert(lbl(7)));
        assert!(!set.insert(lbl(7))); // duplicate rejected
        assert_eq!(set.len(), 1);
    }
}
