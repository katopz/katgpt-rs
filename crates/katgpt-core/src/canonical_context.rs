//! Canonical context assembly — a deterministic ordering of retrieved /
//! context items so an order-sensitive consumer sees the SAME context for
//! every permutation of the same item set (Issue 882 P4 rider (a),
//! Research 586, Bench 897).
//!
//! # Why
//!
//! An oracle-anchored eval that feeds retrieved items into an order-sensitive
//! scorer (an LLM prompt, a position-weighted reranker, a primacy/recency
//! biased judge) carries a **permutation spread**: the same retrieved SET,
//! delivered in a different order by a nondeterministic retriever (ties,
//! parallel fan-in, hash-map iteration), scores differently. That spread is
//! noise in the eval, not signal about the system. Sorting the items by a
//! stable content key before assembly makes the spread **0 by construction**
//! — the assembled context is a function of the item multiset alone.
//!
//! ⚠ It does NOT remove position bias; it makes it **constant**. The scorer
//! still weighs position, but it weighs the same assignment every run, so the
//! eval's variance collapses and a real delta is no longer drowned in order
//! noise. Measuring the bias itself is a different instrument.
//!
//! # Two orders, one rule
//!
//! - [`canonical_order_into`] — pure content order: ascending by an `Ord` key
//!   (a BLAKE3 digest via [`content_key`], or a caller key), tie-break by the
//!   input index. Relevance order is discarded.
//! - [`canonical_order_by_score_into`] — relevance order kept: score
//!   descending (NaN LAST, `float_order::desc`), ties broken by the content
//!   key, then by index. Retrieval ties are exactly where retriever
//!   nondeterminism enters, so this is the minimal fix when the order itself
//!   carries meaning.
//!
//! Both are **total orders** over distinct indices, so `sort_unstable_by`
//! (in-place, zero allocation — the Bench 894 finding that the stable sort
//! heap-allocates its merge scratch) yields one unique sequence.
//!
//! # Permutation invariance — the exact condition
//!
//! The assembled context is permutation-invariant iff every pair of items
//! with EQUAL keys also has EQUAL content. With BLAKE3 content keys that is
//! the collision-resistance assumption (equal digest ⇒ equal bytes, and equal
//! bytes assemble identically whichever index wins the tie). With CALLER keys
//! it is the caller's obligation, and both functions return the number of
//! adjacent key ties so the caller can assert it (`ties == 0` ⇒ invariance
//! holds with no content assumption at all).
//!
//! # Idempotence
//!
//! Applying the order and re-canonicalizing yields the identity permutation:
//! the ordered keys are already sorted, and the index tie-break over a sorted
//! run preserves position. Pinned in the unit tests and in Bench 897 G1a.

use crate::float_order;

/// A 32-byte content key — the BLAKE3 digest of an item's bytes.
pub type ContentKey = [u8; 32];

/// BLAKE3 digest of an item's bytes — the default stable content key.
///
/// Compute it once at ingest and keep it beside the item; hashing on every
/// assembly is the dominant cost, the sort is not.
#[inline]
pub fn content_key(bytes: &[u8]) -> ContentKey {
    *blake3::hash(bytes).as_bytes()
}

/// Fill `order` with the canonical permutation of `keys`: ascending by key,
/// ties broken by input index. Returns the number of adjacent key ties.
///
/// `order.len()` must equal `keys.len()` and `keys.len() <= u32::MAX`.
/// Zero allocation (`sort_unstable_by` is in-place; the comparator is a
/// total order over distinct indices, so the result is unique).
pub fn canonical_order_into<K: Ord>(keys: &[K], order: &mut [u32]) -> usize {
    assert_eq!(
        order.len(),
        keys.len(),
        "canonical_order_into: order length"
    );
    assert!(
        keys.len() <= u32::MAX as usize,
        "canonical_order_into: too many items"
    );
    fill_identity(order);
    order.sort_unstable_by(|&a, &b| {
        keys[a as usize]
            .cmp(&keys[b as usize])
            .then_with(|| a.cmp(&b))
    });
    count_ties(order, |a, b| keys[a as usize] == keys[b as usize])
}

/// Fill `order` with the relevance-preserving canonical permutation: score
/// DESCENDING (NaN last, `-0.0 ≡ +0.0` — `float_order::desc`), ties broken by
/// `keys` ascending, then by input index. Returns the number of adjacent
/// (score, key) ties.
///
/// `scores`, `keys` and `order` must share one length `<= u32::MAX`.
/// Zero allocation.
pub fn canonical_order_by_score_into<K: Ord>(
    scores: &[f32],
    keys: &[K],
    order: &mut [u32],
) -> usize {
    assert_eq!(
        scores.len(),
        keys.len(),
        "canonical_order_by_score_into: keys length"
    );
    assert_eq!(
        order.len(),
        keys.len(),
        "canonical_order_by_score_into: order length"
    );
    assert!(
        keys.len() <= u32::MAX as usize,
        "canonical_order_by_score_into: too many items"
    );
    fill_identity(order);
    order.sort_unstable_by(|&a, &b| {
        let (ia, ib) = (a as usize, b as usize);
        float_order::desc(scores[ia], scores[ib])
            .then_with(|| keys[ia].cmp(&keys[ib]))
            .then_with(|| a.cmp(&b))
    });
    count_ties(order, |a, b| {
        let (ia, ib) = (a as usize, b as usize);
        float_order::desc(scores[ia], scores[ib]).is_eq() && keys[ia] == keys[ib]
    })
}

/// Assemble `items` in `order` into `out`, writing `sep` between items.
///
/// `out` is cleared first. Zero allocation when `out.capacity()` already
/// covers the assembled length (caller-owned buffer, reused across calls).
pub fn assemble_into(items: &[&[u8]], order: &[u32], sep: &[u8], out: &mut Vec<u8>) {
    out.clear();
    for (pos, &i) in order.iter().enumerate() {
        if pos > 0 {
            out.extend_from_slice(sep);
        }
        out.extend_from_slice(items[i as usize]);
    }
}

/// Bytes [`assemble_into`] will write — size the caller's buffer once.
pub fn assembled_len(items: &[&[u8]], sep: &[u8]) -> usize {
    let body: usize = items.iter().map(|b| b.len()).sum();
    body + sep.len() * items.len().saturating_sub(1)
}

#[inline]
fn fill_identity(order: &mut [u32]) {
    for (i, o) in order.iter_mut().enumerate() {
        *o = i as u32;
    }
}

#[inline]
fn count_ties(order: &[u32], eq: impl Fn(u32, u32) -> bool) -> usize {
    order.windows(2).filter(|w| eq(w[0], w[1])).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply<T: Copy>(xs: &[T], order: &[u32]) -> Vec<T> {
        order.iter().map(|&i| xs[i as usize]).collect()
    }

    #[test]
    fn content_order_is_permutation_invariant() {
        let items: Vec<Vec<u8>> = (0..17u8).map(|i| vec![i; 3 + i as usize % 5]).collect();
        let keys: Vec<ContentKey> = items.iter().map(|b| content_key(b)).collect();
        let mut order = vec![0u32; items.len()];
        let ties = canonical_order_into(&keys, &mut order);
        assert_eq!(ties, 0);
        let refs: Vec<&[u8]> = items.iter().map(|b| b.as_slice()).collect();
        let mut canon = Vec::new();
        assemble_into(&refs, &order, b"\n", &mut canon);
        assert_eq!(canon.len(), assembled_len(&refs, b"\n"));

        // Reverse and rotate the input; the assembled bytes must not move.
        for rot in 0..items.len() {
            let mut perm: Vec<usize> = (0..items.len()).rev().collect();
            perm.rotate_left(rot);
            let p_items: Vec<&[u8]> = perm.iter().map(|&i| refs[i]).collect();
            let p_keys: Vec<ContentKey> = perm.iter().map(|&i| keys[i]).collect();
            let mut p_order = vec![0u32; items.len()];
            canonical_order_into(&p_keys, &mut p_order);
            let mut got = Vec::new();
            assemble_into(&p_items, &p_order, b"\n", &mut got);
            assert_eq!(got, canon, "rotation {rot}");
        }
    }

    #[test]
    fn canonicalization_is_idempotent() {
        let keys = [5u32, 1, 9, 1, 3, 7, 7, 0];
        let mut order = [0u32; 8];
        let ties = canonical_order_into(&keys, &mut order);
        assert_eq!(ties, 2, "1,1 and 7,7");
        let sorted = apply(&keys, &order);
        let mut again = [0u32; 8];
        canonical_order_into(&sorted, &mut again);
        assert_eq!(again, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn score_order_keeps_relevance_and_breaks_ties_by_key() {
        let scores = [0.5f32, 0.9, f32::NAN, 0.5, -0.0, 0.0];
        let keys = [3u8, 0, 0, 1, 9, 2];
        let mut order = [0u32; 6];
        let ties = canonical_order_by_score_into(&scores, &keys, &mut order);
        // 0.9 first; the two 0.5s by key (1 before 3); ±0 by key (2 before 9); NaN last.
        assert_eq!(order, [1, 3, 0, 5, 4, 2]);
        assert_eq!(ties, 0);
    }

    #[test]
    fn empty_and_single() {
        let mut order: [u32; 0] = [];
        assert_eq!(canonical_order_into::<u8>(&[], &mut order), 0);
        let mut one = [7u32];
        assert_eq!(canonical_order_into(&[42u8], &mut one), 0);
        assert_eq!(one, [0]);
        let mut out = Vec::new();
        assemble_into(&[], &[], b"|", &mut out);
        assert!(out.is_empty());
    }
}
