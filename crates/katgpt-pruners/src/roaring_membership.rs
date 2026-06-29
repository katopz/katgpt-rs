//! Roaring bitmap-backed region membership (Plan 220 Phase 3).
//!
//! Custom lightweight Roaring-like bitmap (`CompactBitmap`) that mimics the
//! Roaring two-level structure without adding an external dependency:
//!
//! - **Sparse chunks** (<4096 set bits) → sorted `Vec<u16>` array container
//! - **Dense chunks**  (≥4096 set bits) → `Box<[u64; 1024]>` bit container
//!
//! Typical savings vs `Vec<bool>` for a 128K vocab:
//! - Accept regions (~5% fill):  array containers  → ~640 B vs 128 KB
//! - Reject regions (~50% fill): bit containers     →   8 KB vs 128 KB
//! - Maybe regions  (~10% fill): array containers  → ~1.3 KB vs 128 KB

// ── BitmapContainer ────────────────────────────────────────────

/// Container for one 65536-key chunk.
#[derive(Clone, Debug)]
enum BitmapContainer {
    /// Sorted array of set-bit positions within [0, 65536).
    Array(Vec<u16>),
    /// 1024 × 64 = 65 536 bits.
    Bits(Box<[u64; 1024]>, u64),
}

/// Threshold for switching between array and bit container (Roaring spec).
const ARRAY_MAX_CARDINALITY: usize = 4096;

/// Enum-based iterator over a `BitmapContainer` — avoids the `Box<dyn Iterator>`
/// heap allocation that the previous `iter()` implementation paid per container
/// (Issue 001 H-13). One enum variant per container kind; the `Iterator` impl
/// dispatches on the variant.
pub enum BitmapContainerIter<'a> {
    /// Sorted-array iteration.
    Array(std::slice::Iter<'a, u16>),
    /// Bit-container iteration: `(word_index, remaining_words)`.
    Bits {
        words: std::slice::Iter<'a, u64>,
        word_index: usize,
        current_word: u64,
    },
}

impl Iterator for BitmapContainerIter<'_> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        match self {
            BitmapContainerIter::Array(it) => it.next().copied(),
            BitmapContainerIter::Bits {
                words,
                word_index,
                current_word,
            } => loop {
                if *current_word != 0 {
                    // trailing_zeros gives the bit position of the lowest set bit.
                    let bit = current_word.trailing_zeros() as u16;
                    *current_word &= *current_word - 1; // clear lowest set bit
                    return Some((*word_index as u16) * 64 + bit);
                }
                match words.next() {
                    Some(&w) => {
                        *current_word = w;
                        *word_index += 1;
                    }
                    None => return None,
                }
            },
        }
    }
}

impl BitmapContainer {
    fn new() -> Self {
        BitmapContainer::Array(Vec::new())
    }

    #[inline]
    fn len(&self) -> u64 {
        match self {
            BitmapContainer::Array(a) => a.len() as u64,
            BitmapContainer::Bits(_, count) => *count,
        }
    }

    #[inline]
    fn contains(&self, lo: u16) -> bool {
        match self {
            BitmapContainer::Array(a) => a.binary_search(&lo).is_ok(),
            BitmapContainer::Bits(b, _) => {
                let word = lo as usize / 64;
                let bit = lo as usize % 64;
                (b[word] >> bit) & 1 == 1
            }
        }
    }

    fn insert(&mut self, lo: u16) {
        match self {
            BitmapContainer::Array(a) => {
                if let Err(pos) = a.binary_search(&lo) {
                    a.insert(pos, lo);
                    self.maybe_promote();
                }
            }
            BitmapContainer::Bits(b, count) => {
                let word = lo as usize / 64;
                let bit = lo as usize % 64;
                let old = b[word];
                let mask = 1u64 << bit;
                if old & mask == 0 {
                    *count += 1;
                }
                b[word] = old | mask;
            }
        }
    }

    fn iter(&self) -> BitmapContainerIter<'_> {
        match self {
            BitmapContainer::Array(a) => BitmapContainerIter::Array(a.iter()),
            // Start with current_word=0 and let the first `next()` call pull the
            // first word from `words`. word_index starts at 0 and increments on
            // each pull, so the emitted bit index is word_index * 64 + bit.
            BitmapContainer::Bits(b, _) => BitmapContainerIter::Bits {
                words: b.iter(),
                word_index: 0,
                current_word: 0,
            },
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            BitmapContainer::Array(a) => {
                a.capacity() * std::mem::size_of::<u16>() + std::mem::size_of::<Vec<u16>>()
            }
            BitmapContainer::Bits(b, _) => std::mem::size_of_val(b.as_ref()),
        }
    }

    /// Promote array → bits if cardinality exceeds threshold.
    fn maybe_promote(&mut self) {
        if let BitmapContainer::Array(a) = self
            && a.len() > ARRAY_MAX_CARDINALITY
        {
            let mut bits = Box::new([0u64; 1024]);
            for &lo in a.iter() {
                bits[lo as usize / 64] |= 1u64 << (lo as usize % 64);
            }
            *self = BitmapContainer::Bits(bits, a.len() as u64);
        }
    }

    /// Build a container from a slice of bools within one 65536-key chunk.
    fn from_bool_chunk(chunk: &[bool]) -> Self {
        let set_indices: Vec<u16> = chunk
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v)
            .map(|(i, _)| i as u16)
            .collect();

        if set_indices.len() >= ARRAY_MAX_CARDINALITY {
            let mut bits = Box::new([0u64; 1024]);
            for lo in set_indices {
                bits[lo as usize / 64] |= 1u64 << (lo as usize % 64);
            }
            let count = bits.iter().map(|w| w.count_ones() as u64).sum();
            BitmapContainer::Bits(bits, count)
        } else {
            // Already sorted by construction.
            BitmapContainer::Array(set_indices)
        }
    }
}

// ── CompactBitmap ──────────────────────────────────────────────

/// Lightweight Roaring-like bitmap backed by containers keyed on high-16 bits.
///
/// Each container covers a range of 65536 consecutive indices.
/// Sparse ranges use sorted `Vec<u16>` (array container), dense ranges use
/// `Box<[u64; 1024]>` (bit container). This gives the classic Roaring
/// space-time trade-off without any external crate.
#[derive(Clone, Debug)]
pub struct CompactBitmap {
    /// Sorted by container index (high 16 bits of u32 key).
    containers: Vec<(u16, BitmapContainer)>,
}

impl CompactBitmap {
    /// Empty bitmap.
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
        }
    }

    /// Convert a `Vec<bool>` (indexed by token position) into a `CompactBitmap`.
    pub fn from_bool_vec(bools: &[bool]) -> Self {
        let mut containers: Vec<(u16, BitmapContainer)> = Vec::new();
        let chunk_size = 65536usize;

        for (chunk_idx, start) in (0..bools.len()).step_by(chunk_size).enumerate() {
            let end = (start + chunk_size).min(bools.len());
            let chunk = &bools[start..end];
            let container = BitmapContainer::from_bool_chunk(chunk);
            if container.len() > 0 {
                containers.push((chunk_idx as u16, container));
            }
        }

        Self { containers }
    }

    /// Set bit at `index`.
    pub fn insert(&mut self, index: u32) {
        let hi = (index >> 16) as u16;
        let lo = (index & 0xFFFF) as u16;

        match self.containers.binary_search_by_key(&hi, |(k, _)| *k) {
            Ok(pos) => self.containers[pos].1.insert(lo),
            Err(pos) => {
                let mut c = BitmapContainer::new();
                c.insert(lo);
                self.containers.insert(pos, (hi, c));
            }
        }
    }

    /// Test bit at `index`.
    #[inline]
    pub fn contains(&self, index: u32) -> bool {
        let hi = (index >> 16) as u16;
        let lo = (index & 0xFFFF) as u16;

        self.containers
            .binary_search_by_key(&hi, |(k, _)| *k)
            .map(|pos| self.containers[pos].1.contains(lo))
            .unwrap_or(false)
    }

    /// Total number of set bits. O(containers).
    pub fn len(&self) -> u64 {
        self.containers.iter().map(|(_, c)| c.len()).sum()
    }

    /// True when no bits are set.
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// Iterate over all set bits in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.containers.iter().flat_map(|&(hi, ref c)| {
            let base = (hi as u32) << 16;
            c.iter().map(move |lo| base | lo as u32)
        })
    }

    /// Set difference: `self - other` (bits in self that are NOT in other).
    pub fn difference(&self, other: &CompactBitmap) -> CompactBitmap {
        let mut result = CompactBitmap::new();

        for &(hi, ref container) in &self.containers {
            let other_container = other
                .containers
                .binary_search_by_key(&hi, |(k, _)| *k)
                .ok()
                .map(|pos| &other.containers[pos].1);

            match (container, other_container) {
                // Same container in both → compute diff.
                (this, Some(other_c)) => {
                    for idx in this.iter() {
                        if !other_c.contains(idx) {
                            result.insert(((hi as u32) << 16) | idx as u32);
                        }
                    }
                }
                // Only in self → copy all.
                (this, None) => {
                    for idx in this.iter() {
                        result.insert(((hi as u32) << 16) | idx as u32);
                    }
                }
            }
        }

        result
    }

    /// In-place union: merge all bits from `other` into `self`.
    pub fn union_with(&mut self, other: &CompactBitmap) {
        for &(hi, ref container) in &other.containers {
            let base = (hi as u32) << 16;
            for lo in container.iter() {
                self.insert(base | lo as u32);
            }
        }
    }

    /// Total heap memory used by this bitmap in bytes.
    pub fn memory_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>()
            + self.containers.capacity() * std::mem::size_of::<(u16, BitmapContainer)>();
        let containers: usize = self.containers.iter().map(|(_, c)| c.memory_bytes()).sum();
        base + containers
    }
}

impl Default for CompactBitmap {
    fn default() -> Self {
        Self::new()
    }
}

// ── RoaringBatching trait ──────────────────────────────────────

/// Extension trait for Roaring bitmap batch operations.
/// Extend — don't modify the existing `RegionBatching` trait (SOLID).
pub trait RoaringBatching: Send + Sync {
    /// Sum total set bits across all provided bitmaps (reject count).
    fn roaring_reject_count(&self, bitmaps: &[CompactBitmap]) -> u64;
    /// Collect up to `max_tokens` set bits from all bitmaps (accept tokens).
    fn roaring_accept_tokens(&self, bitmaps: &[CompactBitmap], max_tokens: usize) -> Vec<u32>;
    /// Set difference between cached and current membership (refinement).
    fn roaring_refine_diff(&self, cached: &CompactBitmap, current: &CompactBitmap)
    -> CompactBitmap;
}

// ── RoaringMembership ─────────────────────────────────────────

/// Per-region Roaring bitmap membership store.
pub struct RoaringMembership {
    /// One bitmap per region.
    pub bitmaps: Vec<CompactBitmap>,
}

impl RoaringMembership {
    pub fn new() -> Self {
        Self {
            bitmaps: Vec::new(),
        }
    }

    pub fn from_bool_vecs(regions: &[Vec<bool>]) -> Self {
        Self {
            bitmaps: regions
                .iter()
                .map(|v| CompactBitmap::from_bool_vec(v))
                .collect(),
        }
    }

    /// Create from pre-built bitmaps.
    pub fn from_bitmaps(bitmaps: Vec<CompactBitmap>) -> Self {
        Self { bitmaps }
    }

    /// Access the underlying bitmaps.
    pub fn bitmaps(&self) -> &[CompactBitmap] {
        &self.bitmaps
    }
}

impl Default for RoaringMembership {
    fn default() -> Self {
        Self::new()
    }
}

impl RoaringBatching for RoaringMembership {
    /// O(bitmaps) — each `len()` is O(containers), not O(vocab).
    fn roaring_reject_count(&self, bitmaps: &[CompactBitmap]) -> u64 {
        bitmaps.iter().map(|b| b.len()).sum()
    }

    /// Gather accept tokens up to cap. Iteration is sorted ascending.
    fn roaring_accept_tokens(&self, bitmaps: &[CompactBitmap], max_tokens: usize) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(max_tokens.min(1024));
        for bm in bitmaps {
            for idx in bm.iter() {
                if tokens.len() >= max_tokens {
                    return tokens;
                }
                tokens.push(idx);
            }
        }
        tokens
    }

    /// Tokens in `current` that aren't in `cached` (newly changed region).
    fn roaring_refine_diff(
        &self,
        cached: &CompactBitmap,
        current: &CompactBitmap,
    ) -> CompactBitmap {
        current.difference(cached)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_contains() {
        let mut bm = CompactBitmap::new();
        bm.insert(42);
        bm.insert(100_000);
        bm.insert(0);
        assert!(bm.contains(42));
        assert!(bm.contains(100_000));
        assert!(bm.contains(0));
    }

    #[test]
    fn test_contains_absent() {
        let mut bm = CompactBitmap::new();
        bm.insert(10);
        bm.insert(20);
        assert!(!bm.contains(15));
        assert!(!bm.contains(0));
        assert!(!bm.contains(30));
    }

    #[test]
    fn test_from_bool_vec() {
        let mut bools = vec![false; 256];
        bools[0] = true;
        bools[10] = true;
        bools[255] = true;

        let bm = CompactBitmap::from_bool_vec(&bools);
        assert!(bm.contains(0));
        assert!(bm.contains(10));
        assert!(bm.contains(255));
        assert!(!bm.contains(1));
        assert!(!bm.contains(11));
        assert!(!bm.contains(254));
        assert_eq!(bm.len(), 3);
    }

    #[test]
    fn test_len_count() {
        let mut bm = CompactBitmap::new();
        let indices: Vec<u32> = vec![5, 100, 65_535, 65_536, 200_000];
        for &i in &indices {
            bm.insert(i);
        }
        assert_eq!(bm.len(), indices.len() as u64);
    }

    #[test]
    fn test_iter_sorted() {
        let mut bm = CompactBitmap::new();
        bm.insert(200);
        bm.insert(5);
        bm.insert(65_540);
        bm.insert(65_537);
        bm.insert(10);

        let collected: Vec<u32> = bm.iter().collect();
        assert_eq!(collected, vec![5, 10, 200, 65_537, 65_540]);
    }

    #[test]
    fn test_difference() {
        let mut a = CompactBitmap::new();
        for i in [1u32, 2, 3, 4, 5] {
            a.insert(i);
        }

        let mut b = CompactBitmap::new();
        for i in [2u32, 4] {
            b.insert(i);
        }

        let diff = a.difference(&b);
        let collected: Vec<u32> = diff.iter().collect();
        assert_eq!(collected, vec![1, 3, 5]);
    }

    #[test]
    fn test_union() {
        let mut a = CompactBitmap::new();
        for i in [1u32, 2, 3] {
            a.insert(i);
        }

        let mut b = CompactBitmap::new();
        for i in [3u32, 4, 5] {
            b.insert(i);
        }

        a.union_with(&b);
        let collected: Vec<u32> = a.iter().collect();
        assert_eq!(collected, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_empty_difference() {
        let mut a = CompactBitmap::new();
        for i in [1u32, 2, 3] {
            a.insert(i);
        }

        let mut b = CompactBitmap::new();
        for i in [1u32, 2, 3] {
            b.insert(i);
        }

        let diff = a.difference(&b);
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn test_memory_less_than_bool_vec() {
        // 128K vocab, ~30% fill (38400 true values spread across the range).
        let vocab_size = 131_072;
        let mut bools = vec![false; vocab_size];
        for (i, b) in bools.iter_mut().enumerate() {
            if i % 3 == 0 || i % 7 == 0 {
                *b = true;
            }
        }

        let bm = CompactBitmap::from_bool_vec(&bools);
        let bool_mem = bools.len(); // 1 byte per bool in Vec<bool>
        let bm_mem = bm.memory_bytes();

        assert!(
            bm_mem < bool_mem,
            "CompactBitmap ({} B) should be smaller than Vec<bool> ({} B)",
            bm_mem,
            bool_mem
        );
        // Expect at least 2× reduction for 30% fill.
        assert!(
            bm_mem * 3 < bool_mem,
            "Expected ≥3× memory reduction, got {} B vs {} B ({:.1}×)",
            bm_mem,
            bool_mem,
            bool_mem as f64 / bm_mem as f64
        );
    }

    #[test]
    fn test_reject_count_matches_sum() {
        let mut rm = RoaringMembership::new();
        let mut b1 = CompactBitmap::new();
        for i in 0u32..100 {
            b1.insert(i);
        }
        let mut b2 = CompactBitmap::new();
        for i in 200u32..350 {
            b2.insert(i);
        }
        let mut b3 = CompactBitmap::new();
        for i in 500u32..501 {
            b3.insert(i);
        }

        rm.bitmaps = vec![b1, b2, b3];
        let count = rm.roaring_reject_count(&rm.bitmaps);
        // b1: 100, b2: 150, b3: 1 → total 251
        assert_eq!(count, 251);
    }

    #[test]
    fn test_accept_tokens_capped() {
        let rm = RoaringMembership::new();

        let mut bm = CompactBitmap::new();
        for i in 0u32..1000 {
            bm.insert(i);
        }

        let tokens = rm.roaring_accept_tokens(&[bm], 10);
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_large_vocab_performance() {
        use std::time::Instant;

        let vocab_size: usize = 131_072; // 128K
        let num_regions = 50;

        // Build 50 regions with varying fill rates (pre-constructed, not timed).
        let bitmaps: Vec<CompactBitmap> = (0..num_regions)
            .map(|r| {
                let fill = if r % 3 == 0 {
                    0.50
                } else if r % 3 == 1 {
                    0.05
                } else {
                    0.10
                };
                let mut bm = CompactBitmap::new();
                for i in 0u32..vocab_size as u32 {
                    let hash = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (r as u64);
                    let val = (hash >> 33) as f64 / (1u64 << 31) as f64;
                    if val < fill {
                        bm.insert(i);
                    }
                }
                bm
            })
            .collect();

        // --- Timed section: only batch operations, not construction ---
        let start = Instant::now();

        // Batch reject count.
        let rm = RoaringMembership {
            bitmaps: bitmaps.clone(),
        };
        let reject_count = rm.roaring_reject_count(&bitmaps);

        // Accept tokens capped.
        let accept_tokens = rm.roaring_accept_tokens(&bitmaps, 256);

        // Refine diff between two adjacent regions.
        let _diff = rm.roaring_refine_diff(&bitmaps[0], &bitmaps[1]);

        let elapsed = start.elapsed();
        assert!(
            elapsed.as_micros() < 1000,
            "128K × 50 regions batch ops took {:?}, expected < 1ms",
            elapsed
        );

        // Sanity: reject count should be non-trivial.
        assert!(reject_count > 0, "reject_count should be > 0");
        assert!(
            !accept_tokens.is_empty(),
            "accept_tokens should not be empty"
        );
    }

    #[test]
    fn test_from_bool_vec_cross_container() {
        // Indices spanning two containers (65536 boundary).
        let mut bools = vec![false; 131_072];
        bools[100] = true;
        bools[65_636] = true; // second container, index 100 within it

        let bm = CompactBitmap::from_bool_vec(&bools);
        assert!(bm.contains(100));
        assert!(bm.contains(65_636));
        assert_eq!(bm.len(), 2);

        let collected: Vec<u32> = bm.iter().collect();
        assert_eq!(collected, vec![100, 65_636]);
    }

    #[test]
    fn test_container_promotion() {
        let mut bm = CompactBitmap::new();
        // Insert > 4096 elements into the first container to trigger promotion.
        for i in 0u32..5000 {
            bm.insert(i);
        }
        assert_eq!(bm.len(), 5000);
        // Verify all inserted values are still present.
        for i in 0u32..5000 {
            assert!(bm.contains(i), "should contain {}", i);
        }
        assert!(!bm.contains(5001));
    }

    #[test]
    fn test_union_across_containers() {
        let mut a = CompactBitmap::new();
        a.insert(100); // container 0
        a.insert(65_540); // container 1

        let mut b = CompactBitmap::new();
        b.insert(200); // container 0
        b.insert(131_100); // container 2

        a.union_with(&b);
        assert!(a.contains(100));
        assert!(a.contains(200));
        assert!(a.contains(65_540));
        assert!(a.contains(131_100));
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn test_refine_diff_returns_new_bits() {
        let rm = RoaringMembership::new();

        let mut cached = CompactBitmap::new();
        for i in [1u32, 2, 3, 4, 5] {
            cached.insert(i);
        }

        let mut current = CompactBitmap::new();
        for i in [1u32, 2, 3, 6, 7] {
            current.insert(i);
        }

        let diff = rm.roaring_refine_diff(&cached, &current);
        let collected: Vec<u32> = diff.iter().collect();
        assert_eq!(collected, vec![6, 7]);
    }
}
