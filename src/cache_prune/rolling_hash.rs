//! Polynomial rolling hash for O(n) variable-length segment matching.
//!
//! Implements a Mersenne-prime rolling hash and a KV segment pool for
//! fast prefix-filter + full-hash verification of cached prompt segments.
//!
//! Reference: Plan 140 — CachePrune (arXiv:2605.23640).

#![allow(clippy::too_many_lines)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RollingHash
// ---------------------------------------------------------------------------

/// Polynomial rolling hash using Mersenne prime 2^61 - 1.
///
/// Provides O(1) substring hashing and O(1) sliding-window updates for
/// variable-length segment matching against cached KV-pool entries.
pub struct RollingHash {
    base: u64,
    modulus: u64, // 2^61 - 1
    powers: Vec<u64>,
}

impl RollingHash {
    /// Initialise with `base = 127` and modulus `2^61 - 1`, pre-computing
    /// `base^i mod modulus` for `i` in `0..=max_length`.
    pub fn new(max_length: usize) -> Self {
        const BASE: u64 = 127;
        const MODULUS: u64 = (1u64 << 61) - 1;

        let mut powers = Vec::with_capacity(max_length + 1);
        powers.push(1);
        for i in 1..=max_length {
            powers.push(mersenne_mul(powers[i - 1], BASE, MODULUS));
        }

        Self {
            base: BASE,
            modulus: MODULUS,
            powers,
        }
    }

    /// Compute prefix hash array for `tokens`.
    ///
    /// `hash[i] = hash[i-1] * base + token[i]  (mod modulus)`,
    /// with `hash[0] = 0`.  Returns an array of length `tokens.len() + 1`.
    pub fn prefix_hashes(&self, tokens: &[u32]) -> Vec<u64> {
        let n = tokens.len();
        let mut h = Vec::with_capacity(n + 1);
        h.push(0);
        for &t in tokens {
            let prev = *h.last().unwrap();
            // prev * base + t  (mod Mersenne prime)
            let val = mersenne_add(
                mersenne_mul(prev, self.base, self.modulus),
                t as u64,
                self.modulus,
            );
            h.push(val);
        }
        h
    }

    /// Hash of the half-open substring `[l..r)` from a precomputed prefix
    /// hash array.  O(1).
    ///
    /// Returns `prefix[r] - prefix[l] * powers[r-l]  (mod modulus)`.
    pub fn substring_hash(&self, prefixes: &[u64], l: usize, r: usize) -> u64 {
        debug_assert!(l <= r, "substring_hash: l must be <= r");
        debug_assert!(
            r < prefixes.len(),
            "substring_hash: r out of bounds in prefix array"
        );
        let len = r - l;
        let lhs = mersenne_mul(prefixes[l], self.powers[len], self.modulus);
        mersenne_sub(prefixes[r], lhs, self.modulus)
    }

    /// Slide a fixed-size window hash by removing `old_token` and adding
    /// `new_token`.  O(1).
    ///
    /// `new_hash = old_hash - old_token * base^(window-1)  (mod modulus)`,
    /// then `new_hash = new_hash * base + new_token  (mod modulus)`.
    pub fn slide(&self, old_hash: u64, old_token: u32, new_token: u32, window_size: usize) -> u64 {
        let old_contrib =
            mersenne_mul(old_token as u64, self.powers[window_size - 1], self.modulus);
        let removed = mersenne_sub(old_hash, old_contrib, self.modulus);
        let shifted = mersenne_mul(removed, self.base, self.modulus);
        mersenne_add(shifted, new_token as u64, self.modulus)
    }

    /// Direct computation of the hash of `tokens[0..len]`.  Used only in
    /// tests for consistency checking.
    pub fn direct_hash(&self, tokens: &[u32]) -> u64 {
        let mut h: u64 = 0;
        for &t in tokens {
            h = mersenne_add(
                mersenne_mul(h, self.base, self.modulus),
                t as u64,
                self.modulus,
            );
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Mersenne-prime arithmetic helpers  (modulus = 2^61 - 1)
// ---------------------------------------------------------------------------

/// Multiply followed by Mersenne reduction.
fn mersenne_mul(a: u64, b: u64, m: u64) -> u64 {
    // For 2^61 - 1 we can use the simple identity:
    //   a * b mod (2^61 - 1)  =  lo + hi  (then reduce once more if >= m)
    // where a * b is split into the low 61 and high bits.
    //
    // Since `a, b < m < 2^61`, their product fits in 122 bits, well within
    // u128.  A plain `% m` on the u128 product compiles to a single
    // multiplication + reduction on most architectures, so this is fine.
    ((a as u128) * (b as u128) % (m as u128)) as u64
}

/// Modular addition.
fn mersenne_add(a: u64, b: u64, m: u64) -> u64 {
    let sum = a + b;
    if sum >= m { sum - m } else { sum }
}

/// Modular subtraction.
fn mersenne_sub(a: u64, b: u64, m: u64) -> u64 {
    if a >= b { a - b } else { m - b + a }
}

// ---------------------------------------------------------------------------
// CachedSegment / KvSegmentPool / MatchResult
// ---------------------------------------------------------------------------

/// A cached KV-pool segment.
pub struct CachedSegment {
    /// Token ids that make up the segment.
    pub token_hashes: Vec<u32>,
    /// Rolling hash of the first `min(128, len)` tokens.
    pub prefix_hash: u64,
    /// blake3 digest of the full token sequence.
    pub full_hash: [u8; 32],
    /// Start position in the *original* prompt that produced this segment.
    pub start: usize,
    /// End position (exclusive) in the original prompt.
    pub end: usize,
}

/// Pool of cached segments indexed by prefix rolling hash.
pub struct KvSegmentPool {
    segments: Vec<CachedSegment>,
    /// prefix rolling hash → indices into `segments`.
    prefix_index: HashMap<u64, Vec<usize>>,
}

/// Result of a segment match query.
pub struct MatchResult {
    /// Index into the segment pool.
    pub segment_idx: usize,
    /// Start offset within the request tokens.
    pub start: usize,
    /// End offset (exclusive) within the request tokens.
    pub end: usize,
    /// Whether the candidate survived blake3 full-hash verification.
    pub verified: bool,
}

/// Number of tokens used for the fast prefix filter.
const PREFIX_WINDOW: usize = 128;

impl KvSegmentPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            prefix_index: HashMap::new(),
        }
    }

    /// Add a segment to the pool.
    ///
    /// Computes a rolling-hash prefix (first `min(128, len)` tokens) and a
    /// blake3 digest of the full token sequence, then indexes by prefix hash.
    pub fn add_segment(&mut self, tokens: &[u32], roller: &RollingHash, start: usize, end: usize) {
        let prefix_len = PREFIX_WINDOW.min(tokens.len());
        let prefixes = roller.prefix_hashes(tokens);
        let prefix_hash = roller.substring_hash(&prefixes, 0, prefix_len);

        // blake3 of the full token sequence — hash the little-endian u32 bytes.
        let mut hasher = blake3::Hasher::new();
        for &t in tokens {
            hasher.update(&t.to_le_bytes());
        }
        let full_hash: [u8; 32] = hasher.finalize().into();

        let idx = self.segments.len();
        self.segments.push(CachedSegment {
            token_hashes: tokens.to_vec(),
            prefix_hash,
            full_hash,
            start,
            end,
        });

        self.prefix_index.entry(prefix_hash).or_default().push(idx);
    }

    /// Two-phase match: prefix-filter via rolling hash, then blake3
    /// verification.
    ///
    /// For every position in `request_tokens`, slide a window equal to each
    /// candidate segment's length and check whether the prefix rolling hash
    /// matches any pool entry.  On a hit, verify with blake3.
    pub fn find_matches(&self, request_tokens: &[u32], roller: &RollingHash) -> Vec<MatchResult> {
        let request_prefixes = roller.prefix_hashes(request_tokens);
        let n = request_tokens.len();
        let mut results = Vec::new();

        // Group segments by length so we can reuse the sliding logic per
        // length class.
        let mut by_len: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, seg) in self.segments.iter().enumerate() {
            by_len.entry(seg.token_hashes.len()).or_default().push(i);
        }

        for seg_len in by_len.keys() {
            if *seg_len == 0 || *seg_len > n {
                continue;
            }

            // How many tokens to use for the prefix filter.
            let prefix_len = PREFIX_WINDOW.min(*seg_len);

            // Slide across the request tokens.
            for pos in 0..=(n - seg_len) {
                let req_prefix_hash =
                    roller.substring_hash(&request_prefixes, pos, pos + prefix_len);

                if let Some(candidates) = self.prefix_index.get(&req_prefix_hash) {
                    // Phase 2: blake3 verification.
                    let window = &request_tokens[pos..pos + seg_len];
                    let mut hasher = blake3::Hasher::new();
                    for &t in window {
                        hasher.update(&t.to_le_bytes());
                    }
                    let window_hash: [u8; 32] = hasher.finalize().into();

                    for &seg_idx in candidates {
                        // Only compare against segments of the right length.
                        let seg = &self.segments[seg_idx];
                        if seg.token_hashes.len() != *seg_len {
                            continue;
                        }
                        let verified = seg.full_hash == window_hash;
                        results.push(MatchResult {
                            segment_idx: seg_idx,
                            start: pos,
                            end: pos + seg_len,
                            verified,
                        });
                    }
                }
            }
        }

        results
    }
}

impl Default for KvSegmentPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: `prefix_hashes` + `substring_hash` matches direct computation.
    #[test]
    fn rolling_hash_consistency() {
        let roller = RollingHash::new(256);
        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];

        let prefixes = roller.prefix_hashes(&tokens);

        // Check every possible substring.
        for l in 0..tokens.len() {
            for r in (l + 1)..=tokens.len() {
                let sub_hash = roller.substring_hash(&prefixes, l, r);
                let direct = roller.direct_hash(&tokens[l..r]);
                assert_eq!(
                    sub_hash, direct,
                    "Mismatch for [{l}..{r}): rolling={sub_hash}, direct={direct}"
                );
            }
        }
    }

    /// Test 2: `slide` matches window recomputation.
    #[test]
    fn slide_matches_window_recomputation() {
        let window_size = 4;
        let roller = RollingHash::new(window_size + 1);
        let tokens: Vec<u32> = vec![100, 200, 300, 400, 500, 600, 700];

        // Initial window [0..4).
        let prefixes = roller.prefix_hashes(&tokens);
        let mut current_hash = roller.substring_hash(&prefixes, 0, window_size);

        // Slide one position at a time and compare.
        for i in 0..(tokens.len() - window_size) {
            let expected = roller.substring_hash(&prefixes, i, i + window_size);
            assert_eq!(current_hash, expected, "Slide mismatch at position {i}");

            if i + window_size < tokens.len() {
                current_hash = roller.slide(
                    current_hash,
                    tokens[i],
                    tokens[i + window_size],
                    window_size,
                );
            }
        }
    }

    /// Test 3: Segment pool exact match is found and verified.
    #[test]
    fn segment_pool_exact_match() {
        let roller = RollingHash::new(256);
        let mut pool = KvSegmentPool::new();

        let segment_tokens: Vec<u32> = (0..64).collect();
        pool.add_segment(&segment_tokens, &roller, 0, 64);

        // Query with the same tokens — should match.
        let matches = pool.find_matches(&segment_tokens, &roller);
        assert!(!matches.is_empty(), "Expected at least one match");
        assert!(matches[0].verified, "Match should be blake3-verified");
        assert_eq!(matches[0].segment_idx, 0);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[0].end, 64);
    }

    /// Test 4: No match returns empty.
    #[test]
    fn segment_pool_no_match() {
        let roller = RollingHash::new(256);
        let mut pool = KvSegmentPool::new();

        let segment_tokens: Vec<u32> = (0..64).collect();
        pool.add_segment(&segment_tokens, &roller, 0, 64);

        // Completely different tokens.
        let request: Vec<u32> = (1000..1064).collect();
        let matches = pool.find_matches(&request, &roller);
        assert!(
            matches.is_empty(),
            "Expected no matches for unrelated tokens"
        );
    }

    /// Test 5: Collision resistance — different sequences produce different
    /// hashes with high probability.
    #[test]
    fn collision_resistance() {
        let roller = RollingHash::new(256);
        let mut seen_hashes: HashMap<u64, Vec<u32>> = HashMap::new();

        // Generate many distinct short sequences and check for collisions.
        for seed in 0u32..200 {
            let tokens: Vec<u32> = vec![seed, seed.wrapping_mul(31), seed.wrapping_add(7)];
            let prefixes = roller.prefix_hashes(&tokens);
            let h = roller.substring_hash(&prefixes, 0, tokens.len());

            if let Some(prev) = seen_hashes.get(&h) {
                assert_eq!(
                    prev, &tokens,
                    "Collision detected between different sequences (seed {seed})"
                );
            } else {
                seen_hashes.insert(h, tokens);
            }
        }

        assert_eq!(seen_hashes.len(), 200, "All 200 hashes should be unique");
    }
}
