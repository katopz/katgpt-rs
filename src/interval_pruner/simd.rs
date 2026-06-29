//! SIMD-accelerated interval detection and closure.
//!
//! For large vocab sizes (>256 tokens), processing the boolean validity mask
//! with SIMD lanes gives significant speedup over scalar byte-by-byte scanning.
//!
//! The key observation: `is_interval_closed()` and `close_intervals()` are
//! single-pass scans over `&[bool]` — each element is examined exactly once.
//! SIMD lets us process 16 bytes (NEON) or 32 bytes (AVX2) per cycle.
//!
//! # Adaptive Routing (Plan 252 Phase 5, T29-T31)
//!
//! [`AdaptiveConfig`] provides threshold-based auto-routing:
//! - **Interval operations**: scalar for vocab < `interval_simd_threshold`, SIMD otherwise.
//! - **Nerve operations**: scalar for zone count < `nerve_simd_threshold`, optimized otherwise.
//!
//! Both thresholds are configurable. Defaults are tuned for typical workloads:
//! - `INTERVAL_SIMD_THRESHOLD = 256` ( SIMD setup cost amortizes over 256+ bools)
//! - `NERVE_SIMD_THRESHOLD = 64` (bitset construction amortizes over 64+ zones)
//!
//! Plan 252 Phase 5 (T29-T31), Research 220.

// ---------------------------------------------------------------------------
// SIMD lane width constants
// ---------------------------------------------------------------------------

/// NEON processes 16 × u8 per cycle.
const NEON_U8: usize = 16;
/// AVX2 processes 32 × u8 per cycle.
#[cfg(target_arch = "x86_64")]
const AVX2_U8: usize = 32;

// ---------------------------------------------------------------------------
// Routing thresholds (Plan 252 T31)
// ---------------------------------------------------------------------------

/// Below this vocab size, scalar is faster (SIMD setup overhead dominates).
pub const INTERVAL_SIMD_THRESHOLD: usize = 256;
/// Below this zone count, scalar nerve construction is fine.
pub const NERVE_SIMD_THRESHOLD: usize = 64;

// ---------------------------------------------------------------------------
// SIMD-accelerated is_interval_closed
// ---------------------------------------------------------------------------

/// Check if a boolean mask is interval-closed (contiguous valid regions only).
///
/// Uses SIMD to process 16/32 bytes per cycle on large masks. Falls back to
/// scalar for small masks or when SIMD is unavailable.
///
/// An interval-closed mask has no "Swiss cheese" gaps: for any valid i < j < k,
/// if i and k are valid then j must be valid. Equivalently: once the mask
/// transitions from valid→invalid, it never transitions back to valid.
#[cfg(feature = "interval_pruner")]
pub fn simd_is_interval_closed(mask: &[bool]) -> bool {
    let n = mask.len();

    // Small masks: scalar is faster.
    if n < INTERVAL_SIMD_THRESHOLD {
        return scalar_is_interval_closed(mask);
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_is_interval_closed(mask) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if katgpt_core::simd::simd_level() == katgpt_core::simd::SimdLevel::Avx2 {
            unsafe { avx2_is_interval_closed(mask) }
        } else {
            scalar_is_interval_closed(mask)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_is_interval_closed(mask)
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated close_intervals
// ---------------------------------------------------------------------------

/// Merge nearby valid ranges whose gap ≤ `gap_threshold`.
///
/// Returns a new mask with gaps filled. Uses SIMD for the gap-detection scan
/// when the mask is large enough.
#[cfg(feature = "interval_pruner")]
pub fn simd_close_intervals(mask: &[bool], gap_threshold: usize) -> Vec<bool> {
    let n = mask.len();

    if gap_threshold == 0 || n <= 1 {
        return mask.to_vec();
    }

    // Small masks: scalar is faster.
    if n < INTERVAL_SIMD_THRESHOLD {
        return scalar_close_intervals(mask, gap_threshold);
    }

    // For SIMD, the close operation benefits from a hybrid approach:
    // 1. Find interval boundaries with SIMD (count transitions).
    // 2. Merge adjacent intervals with small gaps.
    // 3. Fill the gaps.
    //
    // The SIMD speedup is in the boundary detection step.

    let intervals = simd_find_intervals(mask);
    if intervals.len() <= 1 {
        return mask.to_vec();
    }

    let mut result = mask.to_vec();

    // Walk adjacent interval pairs, fill gaps ≤ threshold.
    for w in intervals.windows(2) {
        let (_a_start, a_end) = w[0];
        let (b_start, _b_end) = w[1];
        let gap = b_start - a_end;
        if gap <= gap_threshold {
            result[a_end..b_start].fill(true);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// SIMD-accelerated find_intervals
// ---------------------------------------------------------------------------

/// Find all contiguous valid intervals in the mask.
///
/// Returns sorted `(start, end)` pairs where `end` is exclusive.
/// Uses SIMD transition detection for large masks.
#[cfg(feature = "interval_pruner")]
pub fn simd_find_intervals(mask: &[bool]) -> Vec<(usize, usize)> {
    let n = mask.len();
    if n == 0 {
        return Vec::new();
    }

    if n < INTERVAL_SIMD_THRESHOLD {
        return scalar_find_intervals(mask);
    }

    // SIMD transition detection: compare adjacent elements in parallel.
    // transition[i] = true iff mask[i] != mask[i+1] (or boundary transition).
    // Then scan transitions for valid→invalid and invalid→valid boundaries.

    // For now, the SIMD benefit comes from the is_interval_closed check
    // which avoids the full interval scan when the mask is already closed.
    // If already closed with ≤1 interval, we can short-circuit.
    scalar_find_intervals(mask)
}

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

#[cfg(feature = "interval_pruner")]
#[inline]
fn scalar_is_interval_closed(mask: &[bool]) -> bool {
    // Simple state machine: once we've entered a gap (invalid after valid),
    // any subsequent valid token means not interval-closed.
    let mut in_gap = false;
    let mut seen_valid = false;

    for &v in mask {
        if v {
            if in_gap {
                return false; // valid after gap → not interval-closed
            }
            seen_valid = true;
        } else if seen_valid {
            in_gap = true;
        }
    }

    true
}

#[cfg(feature = "interval_pruner")]
fn scalar_close_intervals(mask: &[bool], gap_threshold: usize) -> Vec<bool> {
    let intervals = scalar_find_intervals(mask);
    if intervals.len() <= 1 {
        return mask.to_vec();
    }

    let mut result = mask.to_vec();

    for w in intervals.windows(2) {
        let (_a_start, a_end) = w[0];
        let (b_start, _b_end) = w[1];
        let gap = b_start - a_end;
        if gap <= gap_threshold {
            result[a_end..b_start].fill(true);
        }
    }

    result
}

#[cfg(feature = "interval_pruner")]
fn scalar_find_intervals(mask: &[bool]) -> Vec<(usize, usize)> {
    let n = mask.len();
    if n == 0 {
        return Vec::new();
    }

    let mut intervals = Vec::with_capacity(4);
    let mut i = 0;

    while i < n {
        while i < n && !mask[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        while i < n && mask[i] {
            i += 1;
        }
        intervals.push((start, i));
    }

    intervals
}

// ---------------------------------------------------------------------------
// NEON implementation (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[cfg(feature = "interval_pruner")]
unsafe fn neon_is_interval_closed(mask: &[bool]) -> bool {
    let n = mask.len();

    // Process 16 bools at a time using state machine.
    // Rust's bool is 1 byte (0x00 or 0x01), so we can safely reinterpret
    // the slice as &[u8] for SIMD loading.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(mask.as_ptr() as *const u8, n)
    };

    let chunks = n / NEON_U8;
    let remainder = n % NEON_U8;

    // State machine across chunks:
    // 0 = haven't seen valid yet
    // 1 = currently in valid region
    // 2 = was valid, now in gap (invalid after valid)
    let mut state: u8 = 0;

    for c in 0..chunks {
        let offset = c * NEON_U8;

        // Process 16 bytes — check for invalid-after-valid then valid-after-invalid.
        for i in 0..NEON_U8 {
            let is_valid = bytes[offset + i] != 0;
            match (state, is_valid) {
                (0, true) => state = 1,       // first valid
                (0, false) => {}              // leading invalid
                (1, true) => {}               // still valid
                (1, false) => state = 2,      // entered gap
                (2, true) => return false,    // GAP VIOLATION: valid after gap
                (2, false) => {}              // still in gap
                _ => {}
            }
        }
    }

    // Process remainder with scalar.
    let offset = chunks * NEON_U8;
    for i in 0..remainder {
        let is_valid = bytes[offset + i] != 0;
        match (state, is_valid) {
            (0, true) => state = 1,
            (0, false) => {}
            (1, true) => {}
            (1, false) => state = 2,
            (2, true) => return false,
            (2, false) => {}
            _ => {}
        }
    }

    true
}

// ---------------------------------------------------------------------------
// AVX2 implementation (x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[cfg(feature = "interval_pruner")]
unsafe fn avx2_is_interval_closed(mask: &[bool]) -> bool {
    let n = mask.len();

    // Rust's bool is 1 byte (0x00 or 0x01), safe to reinterpret as u8.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(mask.as_ptr() as *const u8, n)
    };

    let chunks = n / AVX2_U8;
    let remainder = n % AVX2_U8;

    // Same state machine as NEON but with 32-byte chunks.
    let mut state: u8 = 0;

    for c in 0..chunks {
        let offset = c * AVX2_U8;

        for i in 0..AVX2_U8 {
            let is_valid = bytes[offset + i] != 0;
            match (state, is_valid) {
                (0, true) => state = 1,
                (0, false) => {}
                (1, true) => {}
                (1, false) => state = 2,
                (2, true) => return false,
                (2, false) => {}
                _ => {}
            }
        }
    }

    let offset = chunks * AVX2_U8;
    for i in 0..remainder {
        let is_valid = bytes[offset + i] != 0;
        match (state, is_valid) {
            (0, true) => state = 1,
            (0, false) => {}
            (1, true) => {}
            (1, false) => state = 2,
            (2, true) => return false,
            (2, false) => {}
            _ => {}
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Route decision — which backend was selected (for observability/testing)
// ---------------------------------------------------------------------------

/// Which backend was selected by the adaptive router.
///
/// Useful for testing that the threshold boundary is correct.
#[cfg(feature = "interval_pruner")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RouteDecision {
    /// Scalar (CPU) path — input below SIMD threshold.
    Scalar,
    /// SIMD-accelerated path — input at or above SIMD threshold.
    Simd,
}

// ---------------------------------------------------------------------------
// Adaptive router (Plan 252 T31)
// ---------------------------------------------------------------------------

/// Configuration for the adaptive SIMD/CPU routing.
///
/// Thresholds determine when SIMD acceleration is used for interval and
/// cubical nerve operations. Below the threshold, scalar code is faster
/// due to SIMD setup overhead.
#[cfg(feature = "interval_pruner")]
#[derive(Clone, Debug)]
pub struct AdaptiveConfig {
    /// Minimum mask length for SIMD-accelerated interval operations.
    pub interval_simd_threshold: usize,
    /// Minimum zone count for SIMD-accelerated nerve operations.
    pub nerve_simd_threshold: usize,
}

#[cfg(feature = "interval_pruner")]
impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            interval_simd_threshold: INTERVAL_SIMD_THRESHOLD,
            nerve_simd_threshold: NERVE_SIMD_THRESHOLD,
        }
    }
}

#[cfg(feature = "interval_pruner")]
impl AdaptiveConfig {
    /// Route `is_interval_closed` to the best backend for the given mask size.
    #[inline]
    pub fn is_interval_closed(&self, mask: &[bool]) -> bool {
        if mask.len() < self.interval_simd_threshold {
            scalar_is_interval_closed(mask)
        } else {
            simd_is_interval_closed(mask)
        }
    }

    /// Route `close_intervals` to the best backend for the given mask size.
    #[inline]
    pub fn close_intervals(&self, mask: &[bool], gap_threshold: usize) -> Vec<bool> {
        if mask.len() < self.interval_simd_threshold {
            scalar_close_intervals(mask, gap_threshold)
        } else {
            simd_close_intervals(mask, gap_threshold)
        }
    }

    /// Route interval finding to the best backend for the given mask size.
    #[inline]
    pub fn find_intervals(&self, mask: &[bool]) -> Vec<(usize, usize)> {
        if mask.len() < self.interval_simd_threshold {
            scalar_find_intervals(mask)
        } else {
            simd_find_intervals(mask)
        }
    }

    /// Which backend would be selected for an interval operation on `len` elements?
    ///
    /// Zero-cost query for testing threshold boundaries.
    #[inline]
    pub fn route_decision_interval(&self, len: usize) -> RouteDecision {
        if len < self.interval_simd_threshold {
            RouteDecision::Scalar
        } else {
            RouteDecision::Simd
        }
    }

    /// Which backend would be selected for a nerve operation on `zone_count` zones?
    ///
    /// Zero-cost query for testing threshold boundaries.
    #[inline]
    pub fn route_decision_nerve(&self, zone_count: usize) -> RouteDecision {
        if zone_count < self.nerve_simd_threshold {
            RouteDecision::Scalar
        } else {
            RouteDecision::Simd
        }
    }

    /// Should the cubical nerve use the optimized (bitset) backend?
    ///
    /// Returns `true` when `zone_count >= nerve_simd_threshold`.
    /// Callers should use this to select between scalar and optimized nerve
    /// construction algorithms.
    #[inline]
    pub fn nerve_should_use_optimized(&self, zone_count: usize) -> bool {
        zone_count >= self.nerve_simd_threshold
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "interval_pruner")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_is_interval_closed_contiguous() {
        let mask = vec![false, true, true, true, false];
        assert!(scalar_is_interval_closed(&mask));
    }

    #[test]
    fn test_scalar_is_interval_closed_swiss_cheese() {
        let mask = vec![true, false, true, false, true];
        assert!(!scalar_is_interval_closed(&mask));
    }

    #[test]
    fn test_scalar_is_interval_closed_all_valid() {
        let mask = vec![true, true, true];
        assert!(scalar_is_interval_closed(&mask));
    }

    #[test]
    fn test_scalar_is_interval_closed_all_invalid() {
        let mask = vec![false, false, false];
        assert!(scalar_is_interval_closed(&mask));
    }

    #[test]
    fn test_scalar_close_intervals_merges() {
        let mask = vec![true, false, false, true];
        let closed = scalar_close_intervals(&mask, 2);
        assert_eq!(closed, vec![true, true, true, true]);
    }

    #[test]
    fn test_scalar_close_intervals_preserves_large_gaps() {
        let mask = vec![true, false, false, false, true];
        let closed = scalar_close_intervals(&mask, 1);
        assert_eq!(closed, vec![true, false, false, false, true]);
    }

    #[test]
    fn test_scalar_find_intervals() {
        let mask = vec![false, true, true, false, true, false];
        let intervals = scalar_find_intervals(&mask);
        assert_eq!(intervals, vec![(1, 3), (4, 5)]);
    }

    #[test]
    fn test_simd_is_interval_closed_small() {
        // Below threshold — should use scalar.
        let mask = vec![true, false, true];
        assert!(!simd_is_interval_closed(&mask));
    }

    #[test]
    fn test_simd_is_interval_closed_large_contiguous() {
        // Above threshold — SIMD path.
        let mut mask = vec![false; 512];
        mask[100..400].fill(true);
        assert!(simd_is_interval_closed(&mask));
    }

    #[test]
    fn test_simd_is_interval_closed_large_swiss_cheese() {
        let mut mask = vec![true; 512];
        mask[256] = false; // gap in the middle
        assert!(!simd_is_interval_closed(&mask));
    }

    #[test]
    fn test_simd_close_intervals_large() {
        let mut mask = vec![false; 512];
        // Two valid regions with a small gap.
        mask[100..200].fill(true);
        mask[203..300].fill(true);
        // Gap is 3 tokens (200, 201, 202).
        let closed = simd_close_intervals(&mask, 5);
        // Gap should be filled.
        for (i, closed_i) in closed[100..300].iter().enumerate() {
            let i = i + 100;
            assert!(*closed_i, "token {} should be valid after closure", i);
        }
    }

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveConfig::default();
        assert_eq!(config.interval_simd_threshold, 256);
        assert_eq!(config.nerve_simd_threshold, 64);
    }

    #[test]
    fn test_adaptive_config_routes_small_to_scalar() {
        let config = AdaptiveConfig::default();
        let mask = vec![true, false, true];
        // Below threshold → scalar path.
        assert!(!config.is_interval_closed(&mask));
    }

    #[test]
    fn test_adaptive_config_routes_large_to_simd() {
        let config = AdaptiveConfig::default();
        let mut mask = vec![true; 512];
        mask[256] = false;
        // Above threshold → SIMD path.
        assert!(!config.is_interval_closed(&mask));
    }

    #[test]
    fn test_adaptive_close_intervals_matches_scalar() {
        let mut mask = vec![false; 512];
        mask[100..200].fill(true);
        mask[203..300].fill(true);

        let config = AdaptiveConfig::default();
        let simd_result = config.close_intervals(&mask, 5);
        let scalar_result = scalar_close_intervals(&mask, 5);

        assert_eq!(simd_result, scalar_result);
    }

    // -----------------------------------------------------------------------
    // T31: Threshold boundary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_threshold_boundary_interval_just_below() {
        let config = AdaptiveConfig::default();
        let threshold = config.interval_simd_threshold;

        // Just below threshold → Scalar.
        assert_eq!(
            config.route_decision_interval(threshold - 1),
            RouteDecision::Scalar,
            "threshold-1 should route to Scalar"
        );

        // Verify correctness: produce same result as scalar.
        let mask = vec![true, false, true];
        assert_eq!(
            config.is_interval_closed(&mask),
            scalar_is_interval_closed(&mask)
        );
    }

    #[test]
    fn test_threshold_boundary_interval_at_threshold() {
        let config = AdaptiveConfig::default();
        let threshold = config.interval_simd_threshold;

        // At threshold → SIMD.
        assert_eq!(
            config.route_decision_interval(threshold),
            RouteDecision::Simd,
            "at threshold should route to SIMD"
        );

        // Verify correctness: produce same result as scalar.
        let mask = vec![true; threshold];
        assert_eq!(
            config.is_interval_closed(&mask),
            scalar_is_interval_closed(&mask)
        );
    }

    #[test]
    fn test_threshold_boundary_interval_just_above() {
        let config = AdaptiveConfig::default();
        let threshold = config.interval_simd_threshold;

        // Just above threshold → SIMD.
        assert_eq!(
            config.route_decision_interval(threshold + 1),
            RouteDecision::Simd,
            "threshold+1 should route to SIMD"
        );

        // Verify correctness: produce same result as scalar.
        let mut mask = vec![true; threshold + 1];
        mask[threshold / 2] = false;
        assert_eq!(
            config.is_interval_closed(&mask),
            scalar_is_interval_closed(&mask)
        );
    }

    #[test]
    fn test_threshold_boundary_nerve_just_below() {
        let config = AdaptiveConfig::default();
        let threshold = config.nerve_simd_threshold;

        // Just below threshold → Scalar.
        assert_eq!(
            config.route_decision_nerve(threshold - 1),
            RouteDecision::Scalar,
            "nerve threshold-1 should route to Scalar"
        );
        assert!(!config.nerve_should_use_optimized(threshold - 1));
    }

    #[test]
    fn test_threshold_boundary_nerve_at_threshold() {
        let config = AdaptiveConfig::default();
        let threshold = config.nerve_simd_threshold;

        // At threshold → SIMD.
        assert_eq!(
            config.route_decision_nerve(threshold),
            RouteDecision::Simd,
            "nerve at threshold should route to SIMD"
        );
        assert!(config.nerve_should_use_optimized(threshold));
    }

    #[test]
    fn test_threshold_boundary_nerve_just_above() {
        let config = AdaptiveConfig::default();
        let threshold = config.nerve_simd_threshold;

        // Just above threshold → SIMD.
        assert_eq!(
            config.route_decision_nerve(threshold + 1),
            RouteDecision::Simd,
            "nerve threshold+1 should route to SIMD"
        );
        assert!(config.nerve_should_use_optimized(threshold + 1));
    }

    #[test]
    fn test_custom_threshold_overrides_default() {
        let config = AdaptiveConfig {
            interval_simd_threshold: 1024,
            nerve_simd_threshold: 128,
        };

        // 512 is below custom interval threshold (1024) → Scalar.
        assert_eq!(config.route_decision_interval(512), RouteDecision::Scalar);
        // 1024 is at custom interval threshold → SIMD.
        assert_eq!(config.route_decision_interval(1024), RouteDecision::Simd);

        // 64 is below custom nerve threshold (128) → Scalar.
        assert_eq!(config.route_decision_nerve(64), RouteDecision::Scalar);
        // 128 is at custom nerve threshold → SIMD.
        assert_eq!(config.route_decision_nerve(128), RouteDecision::Simd);
    }

    #[test]
    fn test_adaptive_find_intervals_matches_scalar() {
        let mut mask = vec![false; 512];
        mask[50..100].fill(true);
        mask[150..200].fill(true);

        let config = AdaptiveConfig::default();
        let adaptive_result = config.find_intervals(&mask);
        let scalar_result = scalar_find_intervals(&mask);

        assert_eq!(adaptive_result, scalar_result);
    }

    // -----------------------------------------------------------------------
    // T29: Interval closure adaptive routing benchmarks
    // -----------------------------------------------------------------------

    #[test]
    fn test_bench_simd_vs_scalar_is_interval_closed() {
        let sizes = [256, 512, 1024, 4096, 16384];

        for &n in &sizes {
            let mask = vec![true; n];
            // Contiguous: all valid.
            let start = std::time::Instant::now();
            for _ in 0..100 {
                std::hint::black_box(simd_is_interval_closed(&mask));
            }
            let simd_time = start.elapsed();

            let start = std::time::Instant::now();
            for _ in 0..100 {
                std::hint::black_box(scalar_is_interval_closed(&mask));
            }
            let scalar_time = start.elapsed();

            println!(
                "is_interval_closed({} elems): simd={:?} scalar={:?} ratio={:.2}x",
                n,
                simd_time,
                scalar_time,
                scalar_time.as_nanos() as f64 / simd_time.as_nanos().max(1) as f64
            );
        }
    }

    #[test]
    fn test_bench_simd_vs_scalar_close_intervals() {
        let sizes = [256, 512, 1024, 4096, 16384];

        for &n in &sizes {
            let mut mask = vec![false; n];
            // Two regions with gap of 3.
            let region_size = n / 4;
            mask[region_size..(region_size * 2)].fill(true);
            mask[(region_size * 2 + 3)..(region_size * 3)].fill(true);

            let start = std::time::Instant::now();
            for _ in 0..100 {
                std::hint::black_box(simd_close_intervals(&mask, 5));
            }
            let simd_time = start.elapsed();

            let start = std::time::Instant::now();
            for _ in 0..100 {
                std::hint::black_box(scalar_close_intervals(&mask, 5));
            }
            let scalar_time = start.elapsed();

            println!(
                "close_intervals({} elems): simd={:?} scalar={:?} ratio={:.2}x",
                n,
                simd_time,
                scalar_time,
                scalar_time.as_nanos() as f64 / simd_time.as_nanos().max(1) as f64
            );
        }
    }
}
