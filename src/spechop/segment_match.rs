//! Sub-prompt matching for SpecHop using rolling hash segment indexing.
//!
//! Wraps `cache_prune::RollingHash` to maintain a segment pool of previously
//! seen hop observations and enables O(n) sub-sequence matching across hops.
//! When a new observation arrives, the index checks whether any sub-segment
//! matches a previously seen segment — enabling reused-pattern detection across
//! the speculation pipeline.
//!
//! **Feature gate:** `spechop` + `cache_prune`

use std::collections::VecDeque;

use crate::cache_prune::RollingHash;

// ---------------------------------------------------------------------------
// IndexedSegment
// ---------------------------------------------------------------------------

/// A previously indexed hop observation segment.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IndexedSegment {
    /// Which hop this segment came from.
    pub hop_idx: usize,
    /// Hashes of the token sequence.
    pub token_hashes: Vec<u32>,
    /// Pre-computed rolling hash (full sequence).
    pub rolling_hash: u64,
    /// blake3 hash for verification.
    pub full_hash: [u8; 32],
}

// ---------------------------------------------------------------------------
// SegmentMatch
// ---------------------------------------------------------------------------

/// Result of a segment match query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentMatch {
    /// Which previous hop matched.
    pub source_hop_idx: usize,
    /// Start position in the source segment's token sequence.
    pub match_start: usize,
    /// Length of the match.
    pub match_length: usize,
    /// blake3 verification hash of the matched sub-sequence.
    pub full_hash: [u8; 32],
}

// ---------------------------------------------------------------------------
// HopSegmentIndex
// ---------------------------------------------------------------------------

/// Rolling-hash segment index for hop observations.
///
/// Maintains a pool of previously seen hop token sequences and supports
/// prefix matching and substring matching via the Mersenne-prime rolling
/// hash from `cache_prune`.
pub struct HopSegmentIndex {
    roller: RollingHash,
    pool: VecDeque<IndexedSegment>,
    max_segments: usize,
}

impl HopSegmentIndex {
    /// Create a new index with the given maximum segment length for
    /// rolling-hash power pre-computation.
    ///
    /// `max_segment_length` determines how many powers of the hash base are
    /// pre-computed. Segments longer than this cannot be indexed.
    pub fn new(max_segment_length: usize) -> Self {
        Self {
            roller: RollingHash::new(max_segment_length),
            pool: VecDeque::new(),
            max_segments: 1024,
        }
    }

    /// Set the maximum number of segments retained in the pool.
    /// When the pool exceeds this limit, the oldest entry is evicted.
    pub fn with_max_segments(mut self, max: usize) -> Self {
        self.max_segments = max;
        self
    }

    /// Add a hop's tokens to the index.
    ///
    /// Computes the rolling hash and blake3 digest, then stores the segment
    /// for future matching queries.
    pub fn index_observation(&mut self, hop_idx: usize, tokens: &[u32]) {
        let prefixes = self.roller.prefix_hashes(tokens);
        let rolling_hash = self.roller.substring_hash(&prefixes, 0, tokens.len());

        let mut hasher = blake3::Hasher::new();
        for &t in tokens {
            hasher.update(&t.to_le_bytes());
        }
        let full_hash: [u8; 32] = hasher.finalize().into();

        self.pool.push_back(IndexedSegment {
            hop_idx,
            token_hashes: tokens.to_vec(),
            rolling_hash,
            full_hash,
        });

        // Evict oldest if over capacity — O(1) with VecDeque pop_front.
        if self.pool.len() > self.max_segments {
            self.pool.pop_front();
        }
    }

    /// Find the longest matching prefix across all indexed segments.
    ///
    /// Compares the query `tokens` against every indexed segment, looking for
    /// the longest common prefix. Uses rolling hash for O(1) per-length checks
    /// and blake3 for final verification.
    pub fn find_matching_prefix(&self, tokens: &[u32]) -> Option<SegmentMatch> {
        if tokens.is_empty() || self.pool.is_empty() {
            return None;
        }

        let query_prefixes = self.roller.prefix_hashes(tokens);
        let mut best: Option<SegmentMatch> = None;
        let mut best_len = 0;

        for seg in &self.pool {
            let max_check = tokens.len().min(seg.token_hashes.len());
            if max_check <= best_len {
                // Cannot beat current best — skip.
                continue;
            }

            let seg_prefixes = self.roller.prefix_hashes(&seg.token_hashes);

            // Binary search for the longest matching prefix using rolling hash.
            let mut lo = best_len + 1;
            let mut hi = max_check;
            let mut found_len = 0;

            while lo <= hi {
                let mid = lo + (hi - lo) / 2;
                let q_hash = self.roller.substring_hash(&query_prefixes, 0, mid);
                let s_hash = self.roller.substring_hash(&seg_prefixes, 0, mid);

                if q_hash == s_hash {
                    found_len = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }

            if found_len > best_len {
                // Verify with blake3 before accepting.
                let mut hasher = blake3::Hasher::new();
                for &t in &tokens[..found_len] {
                    hasher.update(&t.to_le_bytes());
                }
                let verify_hash: [u8; 32] = hasher.finalize().into();

                // Cross-verify against source segment's same prefix.
                let mut src_hasher = blake3::Hasher::new();
                for &t in &seg.token_hashes[..found_len] {
                    src_hasher.update(&t.to_le_bytes());
                }
                let src_verify: [u8; 32] = src_hasher.finalize().into();

                if verify_hash == src_verify {
                    best_len = found_len;
                    best = Some(SegmentMatch {
                        source_hop_idx: seg.hop_idx,
                        match_start: 0,
                        match_length: found_len,
                        full_hash: verify_hash,
                    });
                }
            }
        }

        best
    }

    /// Find a matching sub-sequence (not necessarily starting at position 0)
    /// of at least `min_length` tokens across all indexed segments.
    ///
    /// For each indexed segment, slides a window across the query tokens
    /// looking for any window that matches a contiguous sub-range of the
    /// segment. Returns the first match found (by segment order, longest
    /// match preferred).
    pub fn find_substring_match(&self, tokens: &[u32], min_length: usize) -> Option<SegmentMatch> {
        if tokens.len() < min_length || self.pool.is_empty() {
            return None;
        }

        let query_prefixes = self.roller.prefix_hashes(tokens);
        let mut best: Option<SegmentMatch> = None;
        let mut best_len = min_length - 1; // Must beat this.

        for seg in &self.pool {
            let seg_prefixes = self.roller.prefix_hashes(&seg.token_hashes);
            let seg_len = seg.token_hashes.len();

            // Try every starting position in the query, looking for a match
            // against any position in the segment.
            for q_start in 0..tokens.len() {
                let remaining = tokens.len() - q_start;
                if remaining <= best_len {
                    break; // No chance to beat best from this position.
                }

                // For each query start, try to find a matching sub-range in
                // the segment using binary search on length.
                let max_possible = remaining.min(seg_len);
                if max_possible <= best_len {
                    continue;
                }

                for s_start in 0..seg_len {
                    let seg_remaining = seg_len - s_start;
                    let max_check = remaining.min(seg_remaining);
                    if max_check <= best_len {
                        continue;
                    }

                    // Binary search for the longest match starting at
                    // (q_start, s_start).
                    let mut lo = best_len + 1;
                    let mut hi = max_check;
                    let mut found_len = 0;

                    while lo <= hi {
                        let mid = lo + (hi - lo) / 2;
                        let q_hash =
                            self.roller
                                .substring_hash(&query_prefixes, q_start, q_start + mid);
                        let s_hash =
                            self.roller
                                .substring_hash(&seg_prefixes, s_start, s_start + mid);

                        if q_hash == s_hash {
                            found_len = mid;
                            lo = mid + 1;
                        } else {
                            hi = mid - 1;
                        }
                    }

                    if found_len > best_len {
                        // blake3 verification.
                        let mut hasher = blake3::Hasher::new();
                        for &t in &tokens[q_start..q_start + found_len] {
                            hasher.update(&t.to_le_bytes());
                        }
                        let verify_hash: [u8; 32] = hasher.finalize().into();

                        let mut src_hasher = blake3::Hasher::new();
                        for &t in &seg.token_hashes[s_start..s_start + found_len] {
                            src_hasher.update(&t.to_le_bytes());
                        }
                        let src_verify: [u8; 32] = src_hasher.finalize().into();

                        if verify_hash == src_verify {
                            best_len = found_len;
                            best = Some(SegmentMatch {
                                source_hop_idx: seg.hop_idx,
                                match_start: s_start,
                                match_length: found_len,
                                full_hash: verify_hash,
                            });
                        }
                    }
                }
            }
        }

        best
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_match_after_indexing_two_observations() {
        let mut idx = HopSegmentIndex::new(256);

        // Hop 0: tokens [1, 2, 3, 4, 5, 6]
        let hop0: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        idx.index_observation(0, &hop0);

        // Hop 1: tokens [10, 20, 30]
        let hop1: Vec<u32> = vec![10, 20, 30];
        idx.index_observation(1, &hop1);

        // Query shares a prefix of length 4 with hop 0.
        let query: Vec<u32> = vec![1, 2, 3, 4, 99, 100];
        let m = idx
            .find_matching_prefix(&query)
            .expect("should find prefix match");

        assert_eq!(m.source_hop_idx, 0);
        assert_eq!(m.match_start, 0);
        assert_eq!(m.match_length, 4);
    }

    #[test]
    fn test_no_false_match_on_disjoint_observations() {
        let mut idx = HopSegmentIndex::new(256);

        let hop0: Vec<u32> = vec![100, 200, 300, 400, 500];
        idx.index_observation(0, &hop0);

        let hop1: Vec<u32> = vec![600, 700, 800, 900, 1000];
        idx.index_observation(1, &hop1);

        // Query is completely disjoint.
        let query: Vec<u32> = vec![1, 2, 3, 4, 5];
        assert!(
            idx.find_matching_prefix(&query).is_none(),
            "should not match disjoint sequences"
        );
        assert!(
            idx.find_substring_match(&query, 2).is_none(),
            "should not substring-match disjoint sequences"
        );
    }

    #[test]
    fn test_substring_match_across_overlapping_segments() {
        let mut idx = HopSegmentIndex::new(256);

        // Hop 0: [10, 20, 30, 40, 50, 60]
        let hop0: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
        idx.index_observation(0, &hop0);

        // Hop 1: [100, 200, 300]
        let hop1: Vec<u32> = vec![100, 200, 300];
        idx.index_observation(1, &hop1);

        // Query contains [30, 40, 50] which appears inside hop 0 starting at
        // position 2.
        let query: Vec<u32> = vec![999, 888, 777, 30, 40, 50, 666];
        let m = idx
            .find_substring_match(&query, 3)
            .expect("should find substring match");

        assert_eq!(m.source_hop_idx, 0);
        assert_eq!(m.match_start, 2); // position in hop0
        assert_eq!(m.match_length, 3);
    }

    #[test]
    fn test_exact_full_match() {
        let mut idx = HopSegmentIndex::new(256);

        let tokens: Vec<u32> = (0..32).collect();
        idx.index_observation(0, &tokens);

        // Query identical to the indexed segment.
        let m = idx
            .find_matching_prefix(&tokens)
            .expect("should match full prefix");

        assert_eq!(m.source_hop_idx, 0);
        assert_eq!(m.match_length, 32);
    }

    #[test]
    fn test_empty_query_returns_none() {
        let mut idx = HopSegmentIndex::new(256);
        idx.index_observation(0, &[1, 2, 3]);

        assert!(idx.find_matching_prefix(&[]).is_none());
        assert!(idx.find_substring_match(&[], 1).is_none());
    }

    #[test]
    fn test_empty_pool_returns_none() {
        let idx = HopSegmentIndex::new(256);
        assert!(idx.find_matching_prefix(&[1, 2, 3]).is_none());
        assert!(idx.find_substring_match(&[1, 2, 3], 1).is_none());
    }

    #[test]
    fn test_max_segments_eviction() {
        let mut idx = HopSegmentIndex::new(256).with_max_segments(2);

        idx.index_observation(0, &[1, 2, 3]);
        idx.index_observation(1, &[4, 5, 6]);
        idx.index_observation(2, &[7, 8, 9]);

        // Hop 0 should have been evicted. Query for its prefix should not
        // match (it matches hop 0 which is gone).
        assert!(idx.find_matching_prefix(&[1, 2, 3]).is_none());

        // But hop 1 and 2 are still there.
        let m = idx
            .find_matching_prefix(&[4, 5, 6])
            .expect("hop 1 still present");
        assert_eq!(m.source_hop_idx, 1);
    }

    #[test]
    fn test_indexed_segment_serde_roundtrip() {
        let seg = IndexedSegment {
            hop_idx: 42,
            token_hashes: vec![1, 2, 3, 4],
            rolling_hash: 0xDEADBEEF,
            full_hash: [0xAA; 32],
        };

        let json = serde_json::to_string(&seg).unwrap();
        let back: IndexedSegment = serde_json::from_str(&json).unwrap();

        assert_eq!(back.hop_idx, 42);
        assert_eq!(back.token_hashes, vec![1, 2, 3, 4]);
        assert_eq!(back.rolling_hash, 0xDEADBEEF);
        assert_eq!(back.full_hash, [0xAA; 32]);
    }
}
