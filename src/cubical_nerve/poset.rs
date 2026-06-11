//! Distributive meet-semilattice trait and zone poset implementation.
//!
//! The cubical nerve functor ⊞ (arXiv:2503.13663) sends a distributive
//! meet-semilattice L to a cubical set ⊞[L] whose geometric realization
//! is a CAT(0) cube complex. This module provides the `ZonePoset` struct
//! (a partially ordered set of game zones) and the
//! `DistributiveMeetSemilattice` trait it implements.
//!
//! Plan 252 Phase 3 (T16), Research 220.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ZoneId — newtype wrapper around usize for zone identification
// ---------------------------------------------------------------------------

/// Opaque zone identifier.
///
/// Wraps a `usize` with `#[repr(transparent)]` so FFI/layout is identical
/// to `usize` while providing type safety in API signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ZoneId(pub usize);

impl ZoneId {
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for ZoneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Zone({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// ZonePoset — partially ordered set of zones
// ---------------------------------------------------------------------------

/// A partially ordered set of zones representing connectivity / topology.
///
/// The partial order `a ≤ b` means "zone a is reachable from zone b" (or
/// equivalently, zone a is a sub-zone of b). Internally we store the
/// explicit order pairs and build a lookup table on construction for O(1)
/// `leq` queries.
///
/// For `meet`, we pre-compute the transitive closure of the covering
/// relations so that finding the greatest common ancestor is a set
/// intersection over the ancestor sets of each zone.
pub struct ZonePoset {
    zones: Vec<ZoneId>,
    /// Map from ZoneId → set of ZoneIds that are ≤ it (ancestors including self).
    /// `ancestors[a]` = { x | x ≤ a }.
    ancestors: HashMap<ZoneId, Vec<ZoneId>>,
}

impl ZonePoset {
    /// Construct a `ZonePoset` from an explicit list of zones and order pairs.
    ///
    /// `order_pairs` contains pairs `(a, b)` meaning `a ≤ b`.
    /// Reflexive closure is added automatically (every zone is ≤ itself).
    ///
    /// Internally computes the transitive closure of the order relation
    /// so that `leq` and `meet` are O(1) / O(n) respectively.
    pub fn from_edges(zones: Vec<ZoneId>, order_pairs: Vec<(ZoneId, ZoneId)>) -> Self {
        let n = zones.len();

        // Index mapping: ZoneId → position in `zones` (for stable ordering)
        let index: HashMap<ZoneId, usize> =
            zones.iter().enumerate().map(|(i, &z)| (z, i)).collect();

        // Adjacency: lower → set of elements directly above it (covering relation).
        // We'll compute transitive closure via BFS/DFS.
        // Instead, for correctness and simplicity with small zone counts,
        // we build the reachability matrix using Warshall-style propagation.

        // `leq_matrix[i][j]` = true iff zones[i] ≤ zones[j]
        let mut leq_matrix = vec![vec![false; n]; n];

        // Reflexive closure: every element is ≤ itself
        for i in 0..n {
            leq_matrix[i][i] = true;
        }

        // Add explicit order pairs
        for (a, b) in &order_pairs {
            match (index.get(a), index.get(b)) {
                (Some(&ia), Some(&ib)) => {
                    leq_matrix[ia][ib] = true;
                }
                _ => {
                    // Order pair references unknown zone — skip silently.
                    // In debug builds this might indicate a data error, but we
                    // don't panic in production.
                }
            }
        }

        // Transitive closure (Warshall's algorithm): O(n³) but zone counts
        // are small (typically < 100 zones per map).
        for k in 0..n {
            for i in 0..n {
                if leq_matrix[i][k] {
                    for j in 0..n {
                        if leq_matrix[k][j] {
                            leq_matrix[i][j] = true;
                        }
                    }
                }
            }
        }

        // Build ancestor lookup: for each zone j, collect all i where i ≤ j.
        // Sorted for deterministic iteration order.
        let mut ancestors: HashMap<ZoneId, Vec<ZoneId>> = HashMap::with_capacity(n);
        for j in 0..n {
            let mut anc: Vec<ZoneId> = Vec::with_capacity(n);
            for i in 0..n {
                if leq_matrix[i][j] {
                    anc.push(zones[i]);
                }
            }
            anc.sort();
            ancestors.insert(zones[j], anc);
        }

        Self { zones, ancestors }
    }

    /// Return the zones in this poset.
    #[inline]
    pub fn zones(&self) -> &[ZoneId] {
        &self.zones
    }

    /// Check if `a ≤ b` in the partial order (reflexive, transitive).
    ///
    /// O(1) amortized — uses pre-computed ancestor set lookup.
    pub fn leq(&self, a: ZoneId, b: ZoneId) -> bool {
        match self.ancestors.get(&b) {
            Some(anc) => anc.binary_search(&a).is_ok(),
            None => false,
        }
    }

    /// Compute the meet (greatest lower bound) of two zones.
    ///
    /// The meet of `a` and `b` is the greatest element `c` such that
    /// `c ≤ a` AND `c ≤ b`. For a zone poset, this is the greatest
    /// common ancestor (GCA) in the zone hierarchy.
    ///
    /// Returns `None` if no common lower bound exists.
    ///
    /// Implementation: intersect the ancestor sets of `a` and `b`,
    /// then pick the greatest element (by the partial order, i.e. the
    /// one that all others are ≤).
    pub fn meet(&self, a: ZoneId, b: ZoneId) -> Option<ZoneId> {
        let anc_a = self.ancestors.get(&a)?;
        let anc_b = self.ancestors.get(&b)?;

        // Both sets are sorted. Intersect them.
        let mut common: Vec<ZoneId> = Vec::with_capacity(anc_a.len().min(anc_b.len()));
        let mut i = 0usize;
        let mut j = 0usize;
        while i < anc_a.len() && j < anc_b.len() {
            match anc_a[i].cmp(&anc_b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Equal => {
                    common.push(anc_a[i]);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        if common.is_empty() {
            return None;
        }

        // The meet is the greatest common lower bound — i.e. the element
        // that is ≥ all others in the common set.
        // Since `common` is sorted (ascending by ZoneId's Ord), we need
        // the element c such that for all d in common, d ≤ c.
        // That's the element with the largest ancestor set among common elements.
        let mut best = common[0];
        let mut best_anc_len = self.ancestors.get(&best).map_or(0usize, |v| v.len());

        for &candidate in &common[1..] {
            let candidate_anc_len = self.ancestors.get(&candidate).map_or(0usize, |v| v.len());

            // candidate ≥ best iff ancestors[best] ⊆ ancestors[candidate],
            // which (since both are sorted) means
            // ancestors[candidate] is a superset → longer or equal.
            // The element with the most ancestors is the greatest.
            if candidate_anc_len > best_anc_len {
                best = candidate;
                best_anc_len = candidate_anc_len;
            }
        }

        Some(best)
    }
}

// ---------------------------------------------------------------------------
// DistributiveMeetSemilattice trait
// ---------------------------------------------------------------------------

/// A distributive meet-semilattice.
///
/// A meet-semilattice is a partially ordered set where any two elements
/// have a greatest lower bound (meet). "Distributive" means:
///
/// ```text
/// a ∧ (b ∧ c) = (a ∧ b) ∧ (a ∧ c)    // not needed for meet-only,
///                                       // but required by the cubical nerve
/// ```
///
/// The cubical nerve functor ⊞ sends a distributive meet-semilattice L
/// to a cubical set ⊞[L] whose geometric realization is a CAT(0) cube
/// complex, guaranteeing unique geodesics for navigation.
pub trait DistributiveMeetSemilattice {
    type Elem;

    /// Compute the meet (greatest lower bound) of two elements.
    ///
    /// Returns `None` if no lower bound exists.
    fn meet(&self, a: &Self::Elem, b: &Self::Elem) -> Option<Self::Elem>;

    /// Check if `a ≤ b` in the partial order.
    fn leq(&self, a: &Self::Elem, b: &Self::Elem) -> bool;

    /// Iterate all elements of the semilattice.
    fn elements(&self) -> Vec<Self::Elem>;
}

// ---------------------------------------------------------------------------
// impl DistributiveMeetSemilattice for ZonePoset
// ---------------------------------------------------------------------------

impl DistributiveMeetSemilattice for ZonePoset {
    type Elem = ZoneId;

    fn meet(&self, a: &Self::Elem, b: &Self::Elem) -> Option<Self::Elem> {
        ZonePoset::meet(self, *a, *b)
    }

    fn leq(&self, a: &Self::Elem, b: &Self::Elem) -> bool {
        ZonePoset::leq(self, *a, *b)
    }

    fn elements(&self) -> Vec<Self::Elem> {
        self.zones.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn za(id: usize) -> ZoneId {
        ZoneId::new(id)
    }

    #[test]
    fn test_leq_reflexive() {
        let poset = ZonePoset::from_edges(vec![za(0), za(1)], vec![]);
        assert!(poset.leq(za(0), za(0)));
        assert!(poset.leq(za(1), za(1)));
        assert!(!poset.leq(za(0), za(1)));
    }

    #[test]
    fn test_leq_transitive() {
        // 0 ≤ 1 ≤ 2 (chain)
        let poset = ZonePoset::from_edges(
            vec![za(0), za(1), za(2)],
            vec![(za(0), za(1)), (za(1), za(2))],
        );
        assert!(poset.leq(za(0), za(2))); // transitive
        assert!(poset.leq(za(0), za(1)));
        assert!(poset.leq(za(1), za(2)));
        assert!(!poset.leq(za(2), za(0)));
    }

    #[test]
    fn test_meet_chain() {
        // 0 ≤ 1 ≤ 2 (chain)
        // meet(0, 2) = 0 (greatest element ≤ both)
        // meet(1, 2) = 1
        let poset = ZonePoset::from_edges(
            vec![za(0), za(1), za(2)],
            vec![(za(0), za(1)), (za(1), za(2))],
        );
        assert_eq!(poset.meet(za(0), za(2)), Some(za(0)));
        assert_eq!(poset.meet(za(1), za(2)), Some(za(1)));
        assert_eq!(poset.meet(za(0), za(0)), Some(za(0)));
    }

    #[test]
    fn test_meet_disjoint() {
        // Two unrelated zones with no common lower bound except themselves.
        // If 0 and 1 share no order relation, their only common lower bound
        // is themselves IF they are each ≤ themselves... but they need a
        // common element. With no relation, common = {} unless they share
        // an ancestor.
        let poset = ZonePoset::from_edges(
            vec![za(0), za(1)],
            vec![], // no relations
        );
        // meet(0, 1): ancestors of 0 = {0}, ancestors of 1 = {1}
        // intersection = {} → None
        assert_eq!(poset.meet(za(0), za(1)), None);
    }

    #[test]
    fn test_meet_diamond() {
        // Diamond: 0 ≤ 1, 0 ≤ 2, 1 ≤ 3, 2 ≤ 3
        // meet(1, 2) = 0 (greatest common lower bound)
        let poset = ZonePoset::from_edges(
            vec![za(0), za(1), za(2), za(3)],
            vec![
                (za(0), za(1)),
                (za(0), za(2)),
                (za(1), za(3)),
                (za(2), za(3)),
            ],
        );
        assert_eq!(poset.meet(za(1), za(2)), Some(za(0)));
        assert_eq!(poset.meet(za(1), za(3)), Some(za(1)));
        assert_eq!(poset.meet(za(2), za(3)), Some(za(2)));
        assert_eq!(poset.meet(za(3), za(3)), Some(za(3)));
    }

    #[test]
    fn test_trait_impl() {
        let poset = ZonePoset::from_edges(
            vec![za(0), za(1), za(2)],
            vec![(za(0), za(1)), (za(1), za(2))],
        );

        // Test trait methods (inherent methods take ZoneId by value)
        assert!(poset.leq(za(0), za(2)));
        assert_eq!(poset.meet(za(0), za(2)), Some(za(0)));
        assert_eq!(poset.elements().len(), 3);
    }

    #[test]
    fn test_zone_id_repr_transparent() {
        assert_eq!(std::mem::size_of::<ZoneId>(), std::mem::size_of::<usize>());
    }
}
