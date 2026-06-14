//! Self-Learning Selectivity Router — Adaptive CoT (Plan 204).
//!
//! Routes between direct and Chain-of-Thought inference modes based on
//! per-position excess kurtosis tracked via EMA. As the model serves requests,
//! routing automatically improves — zero training required.
//!
//! **Key insight** (Research 180):
//! - High kurtosis (selective/monosemantic) → model confident → direct mode
//! - Low kurtosis (polysemantic) → model uncertain → CoT mode
//!
//! Feature-gated behind `selectivity_router`. Default OFF until GOAT proof.

// ── Errors ──────────────────────────────────────────────────────────

/// Errors for profile serialization/deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum ProfileError {
    /// Magic bytes do not match expected `b"SLR4"`.
    InvalidMagic,
    /// Serialized version is not supported.
    VersionMismatch,
    /// Data is too short to contain a valid profile.
    TruncatedData,
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::InvalidMagic => write!(f, "invalid magic bytes"),
            ProfileError::VersionMismatch => write!(f, "version mismatch"),
            ProfileError::TruncatedData => write!(f, "truncated data"),
        }
    }
}

impl std::error::Error for ProfileError {}

// ── Compute Route ───────────────────────────────────────────────────

/// Route recommendation based on position selectivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComputeRoute {
    /// High kurtosis → predictable → CPU speculative decode.
    CpuSpeculative,
    /// Low kurtosis → complex → GPU autoregressive decode.
    GpuAutoregressive,
}

// ── Selectivity Router ──────────────────────────────────────────────

/// Per-position selectivity router using the polarization effect.
///
/// High kurtosis (selective/monosemantic) → direct mode (no thinking).
/// Low kurtosis (polysemantic) → CoT mode (thinking needed).
///
/// Self-learning: observes kurtosis at each position across inference
/// requests. As the model (or domain) changes, the routing adapts.
#[derive(Debug)]
pub struct SelectivityRouter {
    /// Per-position EMA of excess kurtosis.
    /// Grows dynamically, pre-allocate with `with_capacity()`.
    position_kurtosis: Vec<f32>,
    /// Threshold for direct vs CoT routing.
    /// kurtosis >= threshold → direct mode.
    /// kurtosis < threshold → CoT mode.
    kurtosis_threshold: f32,
    /// EMA decay factor. Lower = slower adaptation.
    alpha: f32,
}

/// Wire format constants for profile serialization.
const MAGIC: [u8; 4] = *b"SLR4";
const VERSION: u32 = 1;
/// Header: 4 (magic) + 4 (version) + 4 (len) = 12 bytes.
const HEADER_SIZE: usize = 12;

impl SelectivityRouter {
    /// Create a new router with default thresholds.
    ///
    /// Defaults: `kurtosis_threshold = 1.0`, `alpha = 0.1`.
    pub fn new() -> Self {
        Self {
            position_kurtosis: Vec::new(),
            kurtosis_threshold: 1.0,
            alpha: 0.1,
        }
    }

    /// Create with pre-allocated capacity for `max_positions` positions.
    pub fn with_capacity(max_positions: usize) -> Self {
        Self {
            position_kurtosis: Vec::with_capacity(max_positions),
            kurtosis_threshold: 1.0,
            alpha: 0.1,
        }
    }

    /// Should this position use CoT (thinking) mode?
    ///
    /// - Returns `true` if kurtosis is LOW → polysemantic → needs thinking.
    /// - Returns `false` if kurtosis is HIGH → monosemantic → direct answer.
    /// - Returns `false` if no data yet (optimistic direct mode).
    ///
    /// O(1) — single array lookup + comparison.
    pub fn should_think(&self, position: usize) -> bool {
        match self.position_kurtosis.get(position) {
            Some(&k) => k < self.kurtosis_threshold,
            None => false, // No data → optimistic direct mode.
        }
    }

    /// Observe kurtosis at a given position. Updates EMA.
    ///
    /// Call after each speculative decode step with the computed kurtosis.
    /// O(1) amortized — Vec resize only when new positions encountered.
    pub fn observe(&mut self, position: usize, kurtosis: f32) {
        if position >= self.position_kurtosis.len() {
            self.position_kurtosis.resize(position + 1, 0.0);
        }
        let prev = self.position_kurtosis[position];
        self.position_kurtosis[position] = self.alpha * kurtosis + (1.0 - self.alpha) * prev;
    }

    /// Get the current EMA kurtosis for a position.
    ///
    /// Returns `None` if position has never been observed.
    pub fn kurtosis_at(&self, position: usize) -> Option<f32> {
        self.position_kurtosis.get(position).copied()
    }

    /// Reset all tracking state. Use when switching domains or sessions.
    pub fn reset(&mut self) {
        self.position_kurtosis.clear();
    }

    /// Recommend compute route for a position.
    ///
    /// - High kurtosis → predictable → `CpuSpeculative`
    /// - Low kurtosis → complex → `GpuAutoregressive`
    /// - No data → `CpuSpeculative` (optimistic)
    pub fn recommend_route(&self, position: usize) -> ComputeRoute {
        match self.should_think(position) {
            true => ComputeRoute::GpuAutoregressive,
            false => ComputeRoute::CpuSpeculative,
        }
    }

    /// Serialize kurtosis profile to bytes.
    ///
    /// Format: `[magic:4][version:4][len:4][f32_slice:len*4]`
    pub fn serialize(&self) -> Vec<u8> {
        let len = self.position_kurtosis.len() as u32;
        let mut buf = Vec::with_capacity(HEADER_SIZE + len as usize * 4);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&len.to_le_bytes());
        let bytes = bytemuck::cast_slice::<f32, u8>(&self.position_kurtosis);
        buf.extend_from_slice(bytes);
        buf
    }

    /// Deserialize kurtosis profile from bytes (cold start recovery).
    pub fn deserialize(data: &[u8]) -> Result<Self, ProfileError> {
        if data.len() < HEADER_SIZE {
            return Err(ProfileError::TruncatedData);
        }

        // Length already validated above; the `try_into` calls below cannot
        // fail because each slice is exactly 4 bytes. Use `?` with a defensive
        // error mapping rather than `unwrap()` to keep the fallible surface
        // explicit and panic-free.
        let magic: [u8; 4] = data[0..4]
            .try_into()
            .map_err(|_| ProfileError::TruncatedData)?;
        if magic != MAGIC {
            return Err(ProfileError::InvalidMagic);
        }

        let version = u32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| ProfileError::TruncatedData)?,
        );
        if version != VERSION {
            return Err(ProfileError::VersionMismatch);
        }

        let len = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| ProfileError::TruncatedData)?,
        ) as usize;
        let expected = HEADER_SIZE + len * 4;
        if data.len() < expected {
            return Err(ProfileError::TruncatedData);
        }

        let f32_bytes = &data[HEADER_SIZE..expected];
        // bytemuck::cast_slice is infallible and zero-copy when the input is
        // properly aligned; otherwise it returns the original input. We use
        // `to_vec()` to materialize the owned Vec<f32>.
        let position_kurtosis = bytemuck::cast_slice::<u8, f32>(f32_bytes).to_vec();

        Ok(Self {
            position_kurtosis,
            kurtosis_threshold: 1.0,
            alpha: 0.1,
        })
    }

    /// Current number of tracked positions.
    pub fn len(&self) -> usize {
        self.position_kurtosis.len()
    }

    /// Whether any positions are tracked.
    pub fn is_empty(&self) -> bool {
        self.position_kurtosis.is_empty()
    }
}

impl Default for SelectivityRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_router_direct_mode() {
        let router = SelectivityRouter::new();
        // No observations → should_think returns false (optimistic direct mode).
        assert!(!router.should_think(0));
        assert!(!router.should_think(100));
        assert!(!router.should_think(9999));
    }

    #[test]
    fn test_high_kurtosis_direct_mode() {
        let mut router = SelectivityRouter::new();
        // High kurtosis (3.0+) → monosemantic → direct mode → should_think = false.
        for _ in 0..20 {
            router.observe(0, 3.5);
        }
        assert!(!router.should_think(0));
    }

    #[test]
    fn test_low_kurtosis_cot_mode() {
        let mut router = SelectivityRouter::new();
        // Low kurtosis (0.0) → polysemantic → CoT mode → should_think = true.
        for _ in 0..20 {
            router.observe(0, 0.0);
        }
        assert!(router.should_think(0));
    }

    #[test]
    fn test_ema_convergence_recent_dominates() {
        let mut router = SelectivityRouter::new();

        // Observe high kurtosis many times.
        for _ in 0..50 {
            router.observe(0, 5.0);
        }
        // Should be firmly direct.
        assert!(!router.should_think(0));

        // Now observe low kurtosis many times — recent should dominate.
        for _ in 0..50 {
            router.observe(0, 0.0);
        }
        // Should now be CoT mode.
        assert!(router.should_think(0));
    }

    #[test]
    fn test_router_converges_after_n_observations() {
        let mut router = SelectivityRouter::new();

        // Observe consistently low kurtosis at position 0 for 100 iterations.
        for _ in 0..100 {
            router.observe(0, 0.2);
        }
        assert!(router.should_think(0));

        // Observe consistently high kurtosis at position 1 for 100 iterations.
        for _ in 0..100 {
            router.observe(1, 3.0);
        }
        assert!(!router.should_think(1));
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut router = SelectivityRouter::new();
        router.observe(0, 0.5);
        router.observe(1, 2.5);
        router.observe(5, 1.5);

        let data = router.serialize();
        let restored = SelectivityRouter::deserialize(&data).unwrap();

        // Verify identical routing decisions.
        assert_eq!(router.should_think(0), restored.should_think(0));
        assert_eq!(router.should_think(1), restored.should_think(1));
        assert_eq!(router.should_think(5), restored.should_think(5));

        // Verify kurtosis values match.
        assert_eq!(router.kurtosis_at(0), restored.kurtosis_at(0));
        assert_eq!(router.kurtosis_at(1), restored.kurtosis_at(1));
        assert_eq!(router.kurtosis_at(5), restored.kurtosis_at(5));
    }

    #[test]
    fn test_recommend_route_mapping() {
        let mut router = SelectivityRouter::new();

        // No data → CpuSpeculative (optimistic direct).
        assert_eq!(router.recommend_route(0), ComputeRoute::CpuSpeculative);

        // Low kurtosis → GpuAutoregressive (needs CoT → GPU).
        for _ in 0..20 {
            router.observe(0, 0.0);
        }
        assert_eq!(router.recommend_route(0), ComputeRoute::GpuAutoregressive);

        // High kurtosis → CpuSpeculative (direct → CPU).
        for _ in 0..20 {
            router.observe(1, 5.0);
        }
        assert_eq!(router.recommend_route(1), ComputeRoute::CpuSpeculative);
    }

    #[test]
    fn test_with_capacity_preallocates() {
        let router = SelectivityRouter::with_capacity(1000);
        assert!(router.is_empty());
        assert!(router.position_kurtosis.capacity() >= 1000);
    }

    #[test]
    fn test_kurtosis_at_returns_none_for_unobserved() {
        let router = SelectivityRouter::new();
        assert!(router.kurtosis_at(0).is_none());
        assert!(router.kurtosis_at(999).is_none());
    }

    #[test]
    fn test_kurtosis_at_returns_some_after_observe() {
        let mut router = SelectivityRouter::new();
        router.observe(5, 2.0);
        let k = router.kurtosis_at(5);
        assert!(k.is_some());
        // After one observation: alpha * 2.0 + (1-alpha) * 0.0 = 0.2
        let expected = 0.1 * 2.0 + 0.9 * 0.0;
        assert!((k.unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_reset_clears_tracking() {
        let mut router = SelectivityRouter::new();
        router.observe(0, 2.0);
        router.observe(1, 0.5);
        assert!(!router.is_empty());

        router.reset();
        assert!(router.is_empty());
        assert!(!router.should_think(0));
        assert!(router.kurtosis_at(0).is_none());
    }

    #[test]
    fn test_deserialize_invalid_magic() {
        let data = [0xFFu8; HEADER_SIZE];
        let result = SelectivityRouter::deserialize(&data);
        assert_eq!(result.unwrap_err(), ProfileError::InvalidMagic);
    }

    #[test]
    fn test_deserialize_version_mismatch() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(&MAGIC);
        data[4..8].copy_from_slice(&99u32.to_le_bytes()); // Bad version.
        let result = SelectivityRouter::deserialize(&data);
        assert_eq!(result.unwrap_err(), ProfileError::VersionMismatch);
    }

    #[test]
    fn test_deserialize_truncated_data() {
        // Too short for header.
        assert_eq!(
            SelectivityRouter::deserialize(&[0u8; 4]).unwrap_err(),
            ProfileError::TruncatedData
        );

        // Header OK, but f32 payload truncated.
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(&MAGIC);
        data[4..8].copy_from_slice(&VERSION.to_le_bytes());
        data[8..12].copy_from_slice(&100u32.to_le_bytes()); // Claims 100 positions.
        assert_eq!(
            SelectivityRouter::deserialize(&data).unwrap_err(),
            ProfileError::TruncatedData
        );
    }

    #[test]
    fn test_profile_error_display() {
        assert_eq!(
            ProfileError::InvalidMagic.to_string(),
            "invalid magic bytes"
        );
        assert_eq!(
            ProfileError::VersionMismatch.to_string(),
            "version mismatch"
        );
        assert_eq!(ProfileError::TruncatedData.to_string(), "truncated data");
    }

    #[test]
    fn test_serialize_deserialize_large_profile() {
        let mut router = SelectivityRouter::new();
        for i in 0..1000 {
            router.observe(i, (i as f32 % 5.0) - 1.0);
        }
        let data = router.serialize();
        let restored = SelectivityRouter::deserialize(&data).unwrap();

        for i in 0..1000 {
            assert_eq!(router.kurtosis_at(i), restored.kurtosis_at(i));
            assert_eq!(router.should_think(i), restored.should_think(i));
        }
    }

    #[test]
    fn test_ema_formula() {
        let mut router = SelectivityRouter::new();
        // alpha=0.1, first observe(0, 10.0): ema = 0.1*10 + 0.9*0 = 1.0
        router.observe(0, 10.0);
        assert!((router.kurtosis_at(0).unwrap() - 1.0).abs() < 1e-6);

        // Second observe(0, 10.0): ema = 0.1*10 + 0.9*1.0 = 1.9
        router.observe(0, 10.0);
        assert!((router.kurtosis_at(0).unwrap() - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_default_impl() {
        let router = SelectivityRouter::default();
        assert!(router.is_empty());
        assert!(!router.should_think(0));
    }
}

// ── Benchmarks ──────────────────────────────────────────────────────

#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_should_think_under_100ns() {
        let mut router = SelectivityRouter::with_capacity(10_000);
        for i in 0..10_000 {
            router.observe(i, (i as f32 % 5.0) - 1.0);
        }

        let iterations: u32 = 100_000;
        let start = Instant::now();
        for i in 0..iterations {
            std::hint::black_box(router.should_think((i as usize) % 10_000));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;
        assert!(
            per_call.as_nanos() < 100,
            "should_think took {:?} per call, expected < 100ns",
            per_call
        );
    }

    #[test]
    fn bench_observe_under_100ns() {
        let mut router = SelectivityRouter::with_capacity(10_000);

        let iterations: u32 = 100_000;
        let start = Instant::now();
        for i in 0..iterations {
            router.observe((i as usize) % 10_000, (i as f32).sin());
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;
        assert!(
            per_call.as_nanos() < 200,
            "observe took {:?} per call, expected < 200ns",
            per_call
        );
    }

    #[test]
    fn bench_serialize_1k_positions() {
        let mut router = SelectivityRouter::new();
        for i in 0..1_000 {
            router.observe(i, (i as f32).sin());
        }
        let start = Instant::now();
        let data = std::hint::black_box(router.serialize());
        let elapsed = start.elapsed();
        assert!(!data.is_empty(), "serialized data should be non-empty");
        eprintln!("serialize 1K positions: {:?}", elapsed);
    }

    #[test]
    fn bench_serialize_10k_positions() {
        let mut router = SelectivityRouter::new();
        for i in 0..10_000 {
            router.observe(i, (i as f32).sin());
        }
        let start = Instant::now();
        let data = std::hint::black_box(router.serialize());
        let elapsed = start.elapsed();
        assert!(!data.is_empty());
        eprintln!("serialize 10K positions: {:?}", elapsed);
    }

    #[test]
    fn bench_serialize_100k_positions() {
        let mut router = SelectivityRouter::new();
        for i in 0..100_000 {
            router.observe(i, (i as f32).sin());
        }
        let start = Instant::now();
        let data = std::hint::black_box(router.serialize());
        let elapsed = start.elapsed();
        assert!(!data.is_empty());
        eprintln!("serialize 100K positions: {:?}", elapsed);
    }

    #[test]
    fn bench_deserialize_1k_positions() {
        let mut router = SelectivityRouter::new();
        for i in 0..1_000 {
            router.observe(i, (i as f32).sin());
        }
        let data = router.serialize();
        let start = Instant::now();
        let _restored = std::hint::black_box(SelectivityRouter::deserialize(&data).unwrap());
        let elapsed = start.elapsed();
        eprintln!("deserialize 1K positions: {:?}", elapsed);
    }

    #[test]
    fn bench_deserialize_10k_positions() {
        let mut router = SelectivityRouter::new();
        for i in 0..10_000 {
            router.observe(i, (i as f32).sin());
        }
        let data = router.serialize();
        let start = Instant::now();
        let _restored = std::hint::black_box(SelectivityRouter::deserialize(&data).unwrap());
        let elapsed = start.elapsed();
        eprintln!("deserialize 10K positions: {:?}", elapsed);
    }

    #[test]
    fn bench_deserialize_100k_positions() {
        let mut router = SelectivityRouter::new();
        for i in 0..100_000 {
            router.observe(i, (i as f32).sin());
        }
        let data = router.serialize();
        let start = Instant::now();
        let _restored = std::hint::black_box(SelectivityRouter::deserialize(&data).unwrap());
        let elapsed = start.elapsed();
        eprintln!("deserialize 100K positions: {:?}", elapsed);
    }
}

// TL;DR: Self-learning adaptive CoT router. Per-position EMA kurtosis → routes direct vs CoT.
// High kurtosis = confident = direct. Low kurtosis = uncertain = CoT. Feature-gated behind `selectivity_router`.
