//! Count-Min Sketch for O(1) frequency estimation — Plan 220 Phase 2.
//!
//! 4-row × 256-col (2KB) CMS with saturating u16 counters. Provides one-sided
//! overestimate of access frequency, safe for LFU eviction: we never underestimate
//! hot entries. Decay is O(1024) constant time. Input keys are already BLAKE3 hashes
//! so hash derivation uses cheap XOR-fold + seed multiplication.

use super::bfcp_region_cache::FreqTier;

// ── Constants ──────────────────────────────────────────────────

const ROWS: usize = 4;
const COLS: usize = 256;

// Pre-computed seeds: BLAKE3("cms_row_0")..BLAKE3("cms_row_3"), truncated to u64.
const SEEDS: [u64; ROWS] = [
    0x4A_2B_E7_CF_91_D3_58_A6,
    0x7F_03_E1_A8_C4_B2_6D_19,
    0xD5_8C_F2_47_0E_A3_6B_91,
    0x31_F7_BA_94_D0_65_2C_E8,
];

// ── CountMinSketch ─────────────────────────────────────────────

/// Count-Min Sketch: 4 × 256 u16 counters (2KB) for O(1) frequency estimation.
///
/// One-sided error: `estimate(key) >= true_count(key)`. Safe for LFU eviction
/// because we never underestimate hot entries.
pub struct CountMinSketch {
    counters: Box<[[u16; COLS]; ROWS]>,
    total_updates: u64,
}

impl CountMinSketch {
    /// Initialize all counters to zero.
    pub fn new() -> Self {
        Self {
            counters: Box::new([[0u16; COLS]; ROWS]),
            total_updates: 0,
        }
    }

    /// Derive column index for row `i` from a 32-byte key.
    ///
    /// The key is already a BLAKE3 hash, so we XOR-fold 4×8-byte chunks to get
    /// a single u64, then multiply by the row seed and take modulo 256.
    #[inline]
    fn col_index(key: &[u8; 32], row: usize) -> usize {
        let k0 = u64::from_le_bytes(key[0..8].try_into().unwrap());
        let k1 = u64::from_le_bytes(key[8..16].try_into().unwrap());
        let k2 = u64::from_le_bytes(key[16..24].try_into().unwrap());
        let k3 = u64::from_le_bytes(key[24..32].try_into().unwrap());
        let folded = k0 ^ k1 ^ k2 ^ k3;
        let mixed = folded.wrapping_mul(SEEDS[row]);
        (mixed as usize) % COLS
    }

    /// Increment counters for `key`. Saturates at `u16::MAX` per cell.
    pub fn update(&mut self, key: &[u8; 32]) {
        for row in 0..ROWS {
            let col = Self::col_index(key, row);
            self.counters[row][col] = self.counters[row][col].saturating_add(1);
        }
        self.total_updates = self.total_updates.saturating_add(1);
    }

    /// Minimum counter across all 4 rows → upper-bound estimate of true frequency.
    ///
    /// Returns `u32` to avoid overflow when callers compare against `u32` thresholds.
    #[inline]
    pub fn estimate(&self, key: &[u8; 32]) -> u32 {
        let mut min = u16::MAX;
        for row in 0..ROWS {
            let col = Self::col_index(key, row);
            min = min.min(self.counters[row][col]);
        }
        min as u32
    }

    /// Multiply all 1024 counters by `lambda`, truncating to u16.
    ///
    /// O(1024) constant time — no per-key iteration. Typical lambda ∈ [0.5, 1.0]
    /// for gradual aging.
    ///
    /// Uses Q16 fixed-point integer math to avoid f32 arithmetic on hot path.
    /// `lambda_fixed = round(lambda * 65536)`, then `new_v = (v * lambda_fixed) >> 16`.
    pub fn decay(&mut self, lambda: f32) {
        let lambda_fixed = (lambda * 65536.0) as u32;
        for row in 0..ROWS {
            for col in 0..COLS {
                let v = self.counters[row][col] as u32;
                self.counters[row][col] = ((v * lambda_fixed) >> 16) as u16;
            }
        }
    }

    /// Zero all counters and reset update count.
    pub fn reset(&mut self) {
        *self.counters = [[0u16; COLS]; ROWS];
        self.total_updates = 0;
    }

    /// Total number of `update()` calls since creation or last reset.
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Number of counter cells (always 1024 = 4 × 256).
    pub fn cell_count() -> usize {
        ROWS * COLS
    }
}

impl Default for CountMinSketch {
    fn default() -> Self {
        Self::new()
    }
}

// ── SketchFrequency Trait ──────────────────────────────────────

/// Frequency estimation via Count-Min Sketch — O(1) amortized per key.
#[cfg(feature = "bfcf_lsh_cms")]
pub trait SketchFrequency: Send + Sync {
    /// Estimate frequency of a BLAKE3-hashed key.
    fn estimate_freq(&self, hash: &[u8; 32]) -> u32;
    /// Decay all counters by `lambda` ∈ [0, 1].
    fn sketch_decay(&mut self, lambda: f32);
    /// Classify into Hot/Warm/Cold based on CMS estimate.
    fn freq_tier_sketch(
        &self,
        hash: &[u8; 32],
        hot_threshold: u32,
        warm_threshold: u32,
    ) -> FreqTier;
}

impl SketchFrequency for CountMinSketch {
    #[inline]
    fn estimate_freq(&self, hash: &[u8; 32]) -> u32 {
        self.estimate(hash)
    }

    fn sketch_decay(&mut self, lambda: f32) {
        self.decay(lambda);
    }

    #[inline]
    fn freq_tier_sketch(
        &self,
        hash: &[u8; 32],
        hot_threshold: u32,
        warm_threshold: u32,
    ) -> FreqTier {
        let est = self.estimate(hash);
        if est > hot_threshold {
            FreqTier::Hot
        } else if est > warm_threshold {
            FreqTier::Warm
        } else {
            FreqTier::Cold
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(seed: u8) -> [u8; 32] {
        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        key
    }

    #[test]
    fn test_cms_estimate_zero_for_unseen() {
        let cms = CountMinSketch::new();
        let key = test_key(42);
        assert_eq!(cms.estimate(&key), 0);
    }

    #[test]
    fn test_cms_update_increases_estimate() {
        let mut cms = CountMinSketch::new();
        let key = test_key(7);
        cms.update(&key);
        assert!(cms.estimate(&key) >= 1);
    }

    #[test]
    fn test_cms_overestimate_property() {
        let mut cms = CountMinSketch::new();
        let key = test_key(99);
        let n: u32 = 50;
        for _ in 0..n {
            cms.update(&key);
        }
        let est = cms.estimate(&key);
        assert!(est >= n, "CMS estimate {est} should be >= true count {n}");
    }

    #[test]
    fn test_cms_different_keys_different_estimates() {
        let mut cms = CountMinSketch::new();
        let key_a = test_key(1);
        let key_b = test_key(200);

        for _ in 0..100 {
            cms.update(&key_a);
        }
        for _ in 0..5 {
            cms.update(&key_b);
        }

        let est_a = cms.estimate(&key_a);
        let est_b = cms.estimate(&key_b);
        assert!(
            est_a > est_b,
            "key_a (100 updates, est={est_a}) should dominate key_b (5 updates, est={est_b})"
        );
    }

    #[test]
    fn test_cms_decay_reduces_estimates() {
        let mut cms = CountMinSketch::new();
        let key = test_key(55);

        for _ in 0..100 {
            cms.update(&key);
        }
        cms.decay(0.5);
        let est = cms.estimate(&key);

        let expected = 50.0f32;
        let tolerance = expected * 0.10;
        assert!(
            (est as f32 - expected).abs() <= tolerance,
            "After decay(0.5), estimate {est} should be ≈{expected} (±10%)"
        );
    }

    #[test]
    fn test_cms_saturating_counters() {
        let mut cms = CountMinSketch::new();
        let key = test_key(13);

        // Hammer the same key beyond u16::MAX — must not panic.
        for _ in 0..(u16::MAX as u64 + 100) {
            cms.update(&key);
        }
        let est = cms.estimate(&key);
        assert_eq!(est, u16::MAX as u32, "Saturated counter should be u16::MAX");
    }

    #[test]
    fn test_cms_freq_tier_hot() {
        let mut cms = CountMinSketch::new();
        let key = test_key(10);
        for _ in 0..200 {
            cms.update(&key);
        }
        let tier = cms.freq_tier_sketch(&key, 100, 10);
        assert_eq!(tier, FreqTier::Hot);
    }

    #[test]
    fn test_cms_freq_tier_warm() {
        let mut cms = CountMinSketch::new();
        let key = test_key(20);
        for _ in 0..50 {
            cms.update(&key);
        }
        let tier = cms.freq_tier_sketch(&key, 100, 10);
        assert_eq!(tier, FreqTier::Warm);
    }

    #[test]
    fn test_cms_freq_tier_cold() {
        let mut cms = CountMinSketch::new();
        let key = test_key(30);
        cms.update(&key);
        let tier = cms.freq_tier_sketch(&key, 100, 10);
        assert_eq!(tier, FreqTier::Cold);
    }

    #[test]
    fn test_cms_decay_constant_time() {
        let mut cms = CountMinSketch::new();

        // Pre-fill with non-zero values to verify decay touches all cells.
        for row in 0..ROWS {
            for col in 0..COLS {
                cms.counters[row][col] = 100;
            }
        }

        cms.decay(0.5);

        // Every cell should now be 50 (100 * 0.5 = 50).
        for row in 0..ROWS {
            for col in 0..COLS {
                assert_eq!(
                    cms.counters[row][col], 50,
                    "Cell [{row}][{col}] should be 50 after decay"
                );
            }
        }
        assert_eq!(CountMinSketch::cell_count(), 1024);
    }
}
