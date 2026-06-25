//! Misalignment Indicator Probe Bank — multi-direction OR-fused cascade (Plan 320, Research 301).
//!
//! Structured bank of N pre-computed direction vectors, each tagged with a
//! fine-grained indicator label, projected via dot-product + sigmoid, and
//! OR-fused into a single firing label. Generalizes the single-direction
//! primitives we already ship (`FutureBehaviorProbe` Plan 292,
//! `EmotionDirections::project` Plan 162, `ClaimVerifier` Plan 284) into a
//! structured multi-direction set with inspectable similarity structure.
//!
//! The bank is generic over `L: IndicatorLabel` (the domain label type) and
//! `const D: usize` (the state-space dimensionality). It carries zero game
//! semantics — the private selling-point moat (NPC behavioral-trait directions,
//! bidirectional cognitive monitoring, KG-triple audit trail) lives in
//! `riir-ai/.research/157_*.md` and downstream riir-ai plans.
//!
//! ## Modelless-first contract
//!
//! The bank holds direction vectors loaded at init from a frozen, BLAKE3-committed
//! artifact (freeze/thaw-compatible). They are NEVER updated at runtime — runtime
//! updates go through the latent-state kernel, not through the bank. The only
//! weight mutations allowed are freeze/thaw (snapshot swap) per AGENTS.md.

use core::marker::PhantomData;

use crate::simd::{fast_sigmoid, simd_dot_f32};

/// Magic bytes for the frozen bank wire format.
pub const BANK_MAGIC: [u8; 4] = [b'I', b'P', b'B', b'K']; // Indicator Probe BanK
/// Current wire-format version.
pub const BANK_WIRE_VERSION: u64 = 1;

// -----------------------------------------------------------------------------
// Trait: IndicatorLabel
// -----------------------------------------------------------------------------

/// Generic indicator label. Domain-specific instantiations (e.g., NPC
/// behavioral traits) impl this trait with their own label enum.
///
/// The trait is monomorphized per instantiation (`L` is a type parameter, not
/// `dyn`) — zero-cost. `as_u8` / `from_u8` give a stable wire discriminant.
pub trait IndicatorLabel: Copy + Eq + core::hash::Hash + Send + Sync + 'static {
    /// Stable u8 discriminant for serialization / sync.
    fn as_u8(&self) -> u8;
    /// Recover from u8. Returns `None` if out of range.
    fn from_u8(d: u8) -> Option<Self>;
    /// Number of distinct labels in this instantiation.
    const COUNT: usize;
}

/// Canonical example label for tests + docs. Real instantiations live in
/// consumer crates (riir-ai supplies the 18-indicator NPC taxonomy).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DemoIndicatorLabel {
    A = 0,
    B = 1,
    C = 2,
}

impl IndicatorLabel for DemoIndicatorLabel {
    #[inline]
    fn as_u8(&self) -> u8 {
        *self as u8
    }
    #[inline]
    fn from_u8(d: u8) -> Option<Self> {
        match d {
            0 => Some(Self::A),
            1 => Some(Self::B),
            2 => Some(Self::C),
            _ => None,
        }
    }
    const COUNT: usize = 3;
}

// -----------------------------------------------------------------------------
// Bank
// -----------------------------------------------------------------------------

/// Structured bank of N pre-computed direction vectors, each tagged with an
/// `IndicatorLabel`. The bank is the open primitive from Research 301:
/// N directions, sigmoid-gated, OR-fused into a single flag.
///
/// Direction vectors are loaded at init from a frozen, BLAKE3-committed
/// artifact (freeze/thaw-compatible). They are NEVER updated at runtime.
pub struct IndicatorProbeBank<L: IndicatorLabel, const D: usize> {
    /// `directions[i]` is the direction vector for label `L::from_u8(i as u8)`.
    /// Shape: `[N][D]` flattened for SIMD-friendly iteration.
    directions: Vec<f32>,
    /// `thresholds[i]` is the sigmoid-input threshold above which label i fires.
    thresholds: Vec<f32>,
    /// Per-bank BLAKE3 manifest hash of the directions + thresholds.
    /// Computed at load time; embedded in the bank for freeze/thaw.
    blake3: [u8; 32],
    /// Freeze/thaw version (monotonic).
    version: u64,
    /// Marker for the label type.
    _marker: PhantomData<L>,
}

/// Error returned by [`IndicatorProbeBank::from_frozen_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BankLoadError {
    /// Input buffer too short to hold a header.
    TooShort,
    /// Magic bytes do not match [`BANK_MAGIC`].
    BadMagic,
    /// Wire-format version is not understood by this loader.
    UnsupportedVersion(u64),
    /// Header `n` or `d` does not match the expected `L::COUNT` / `D`.
    DimMismatch { n: u16, d: u16 },
    /// Trailer length does not match `n * d * 4 + n * 4` bytes.
    Truncated { expected: usize, got: usize },
    /// Recomputed BLAKE3 does not match the embedded hash (tamper-evident).
    HashMismatch,
}

impl<L: IndicatorLabel, const D: usize> IndicatorProbeBank<L, D> {
    /// Construct a bank from caller-owned direction + threshold slices.
    ///
    /// Computes the BLAKE3 commitment over `directions ++ thresholds` and
    /// assigns `version = 1`. This is the test/builder path; production callers
    /// load via [`Self::from_frozen_bytes`] from a committed artifact.
    ///
    /// Returns `None` if the slices don't have the expected shape
    /// (`directions.len() == L::COUNT * D`, `thresholds.len() == L::COUNT`).
    pub fn new(directions: Vec<f32>, thresholds: Vec<f32>) -> Option<Self> {
        if directions.len() != L::COUNT * D || thresholds.len() != L::COUNT {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        // Reinterpret f32 slices as bytes without allocating a Vec<u8>:
        // both inputs are already validated-aligned; `update` accepts &[u8].
        hasher.update(to_bytes(&directions));
        hasher.update(to_bytes(&thresholds));
        let blake3 = *hasher.finalize().as_bytes();
        Some(Self {
            directions,
            thresholds,
            blake3,
            version: BANK_WIRE_VERSION,
            _marker: PhantomData,
        })
    }

    /// Number of indicator directions in the bank (`L::COUNT`).
    #[inline]
    pub fn n(&self) -> usize {
        L::COUNT
    }

    /// State-space dimensionality (`D`).
    #[inline]
    pub fn d(&self) -> usize {
        D
    }

    /// BLAKE3 commitment over `directions ++ thresholds` (tamper-evident).
    #[inline]
    pub fn blake3(&self) -> [u8; 32] {
        self.blake3
    }

    /// Freeze/thaw version (monotonic).
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Read-only view of label `i`'s direction vector (length `D`).
    #[inline]
    pub fn direction(&self, i: usize) -> &[f32] {
        &self.directions[i * D..(i + 1) * D]
    }

    /// Read-only view of label `i`'s threshold.
    #[inline]
    pub fn threshold(&self, i: usize) -> f32 {
        self.thresholds[i]
    }

    /// Read-only view of the flattened `[N][D]` direction buffer.
    #[inline]
    pub fn directions_flat(&self) -> &[f32] {
        &self.directions
    }

    /// Read-only view of the `[N]` threshold buffer.
    #[inline]
    pub fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Project `state` onto direction `label`, return sigmoid score in (0, 1).
    ///
    /// Zero-allocation: reuses `simd_dot_f32`. Matches Plan 292
    /// `FutureBehaviorProbe::forecast` latency target (<200ns at D≤2048).
    #[inline]
    pub fn project(&self, state: &[f32; D], label: L) -> f32 {
        let idx = label.as_u8() as usize;
        debug_assert!(
            idx < L::COUNT,
            "IndicatorLabel::as_u8 returned out-of-range index"
        );
        let dir = &self.directions[idx * D..(idx + 1) * D];
        let raw = simd_dot_f32(dir, state, D);
        fast_sigmoid(raw - self.thresholds[idx])
    }

    /// Project `state` onto every direction, write sigmoid scores into
    /// `out_scores` (caller-allocated scratch, length `L::COUNT`).
    ///
    /// Zero-allocation. This is the per-NPC-per-tick hot path.
    #[inline]
    pub fn project_all_into(&self, state: &[f32; D], out_scores: &mut [f32]) {
        debug_assert_eq!(
            out_scores.len(),
            L::COUNT,
            "out_scores must have length L::COUNT"
        );
        for i in 0..L::COUNT {
            let dir = &self.directions[i * D..(i + 1) * D];
            let raw = simd_dot_f32(dir, state, D);
            out_scores[i] = fast_sigmoid(raw - self.thresholds[i]);
        }
    }

    /// After [`Self::project_all_into`], return the firing label with the
    /// highest score if any score strictly exceeds `tau_fire`, else `None`.
    ///
    /// Paper §2.2 end + §2.3 OR-fusion: "a turn is flagged if any indicator
    /// probe exceeds its threshold on any sentence". We collapse to one fire
    /// per state (the argmax). Ties break by lowest index (stable).
    #[inline]
    pub fn or_fused_fire(&self, scores: &[f32], tau_fire: f32) -> Option<L> {
        debug_assert_eq!(
            scores.len(),
            L::COUNT,
            "scores must have length L::COUNT"
        );
        let mut best_label: Option<L> = None;
        let mut best_score: f32 = tau_fire; // must strictly exceed
        for (i, &s) in scores.iter().enumerate() {
            // Strict `>` (not `>=`) gives lowest-index tie-break: a later equal
            // score cannot displace the earlier label.
            if s > best_score {
                best_score = s;
                best_label = L::from_u8(i as u8);
            }
        }
        best_label
    }

    /// Serialize the bank to its frozen wire format.
    ///
    /// Layout: header `[magic(4) | version(8) | n(2) | d(2) | blake3(32)]`
    /// followed by `directions (n*d*4 bytes)` then `thresholds (n*4 bytes)`.
    /// The blake3 in the header commits over `directions ++ thresholds` only.
    pub fn to_frozen_bytes(&self) -> Vec<u8> {
        let n = L::COUNT;
        let header_len = 4 + 8 + 2 + 2 + 32; // = 48
        let body_len = n * D * 4 + n * 4;
        let mut out = Vec::with_capacity(header_len + body_len);
        out.extend_from_slice(&BANK_MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(D as u16).to_le_bytes());
        out.extend_from_slice(&self.blake3);
        out.extend_from_slice(to_bytes(&self.directions));
        out.extend_from_slice(to_bytes(&self.thresholds));
        out
    }

    /// Load a bank from its frozen wire format. Verifies the embedded BLAKE3
    /// hash; returns `BankLoadError::HashMismatch` on tamper.
    ///
    /// The wire format is the canonical layout produced by
    /// [`Self::to_frozen_bytes`]. The bank is loaded ONCE at init; runtime code
    /// holds an `Arc<IndicatorProbeBank>`.
    pub fn from_frozen_bytes(bytes: &[u8]) -> Result<Self, BankLoadError> {
        let header_len = 4 + 8 + 2 + 2 + 32; // 48
        if bytes.len() < header_len {
            return Err(BankLoadError::TooShort);
        }
        if bytes[..4] != BANK_MAGIC {
            return Err(BankLoadError::BadMagic);
        }
        let version = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        if version != BANK_WIRE_VERSION {
            return Err(BankLoadError::UnsupportedVersion(version));
        }
        let n = u16::from_le_bytes(bytes[12..14].try_into().unwrap()) as usize;
        let d = u16::from_le_bytes(bytes[14..16].try_into().unwrap()) as usize;
        if n != L::COUNT || d != D {
            return Err(BankLoadError::DimMismatch {
                n: n as u16,
                d: d as u16,
            });
        }
        let mut blake3 = [0u8; 32];
        blake3.copy_from_slice(&bytes[16..48]);

        let body_len = n * d * 4 + n * 4;
        let total_expected = header_len + body_len;
        if bytes.len() != total_expected {
            return Err(BankLoadError::Truncated {
                expected: total_expected,
                got: bytes.len(),
            });
        }

        // Verify BLAKE3 over directions ++ thresholds (bytes[header_len..]).
        let computed = *blake3::hash(&bytes[header_len..]).as_bytes();
        if computed != blake3 {
            return Err(BankLoadError::HashMismatch);
        }

        // Reconstruct direction + threshold slices.
        let dir_bytes = header_len + n * d * 4;
        let directions = from_bytes::<f32>(&bytes[header_len..dir_bytes]).to_vec();
        let thresholds = from_bytes::<f32>(&bytes[dir_bytes..total_expected]).to_vec();

        Ok(Self {
            directions,
            thresholds,
            blake3,
            version,
            _marker: PhantomData,
        })
    }
}

impl<L: IndicatorLabel, const D: usize> PartialEq for IndicatorProbeBank<L, D> {
    fn eq(&self, other: &Self) -> bool {
        self.directions == other.directions
            && self.thresholds == other.thresholds
            && self.blake3 == other.blake3
            && self.version == other.version
    }
}

impl<L: IndicatorLabel, const D: usize> core::fmt::Debug for IndicatorProbeBank<L, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IndicatorProbeBank")
            .field("n", &L::COUNT)
            .field("d", &D)
            .field("blake3", &hex_prefix(&self.blake3))
            .field("version", &self.version)
            .finish()
    }
}

// -----------------------------------------------------------------------------
// Wire header view (zero-copy parse; used by similarity matrix + external tools)
// -----------------------------------------------------------------------------

/// Parsed view over a frozen bank wire header (48 bytes). Does NOT verify the
/// BLAKE3; callers wanting a loaded, verified bank should use
/// [`IndicatorProbeBank::from_frozen_bytes`].
#[derive(Clone, Copy)]
#[repr(C)]
pub struct IndicatorBankWireHeader {
    pub magic: [u8; 4],
    pub version: u64,
    pub n: u16,
    pub d: u16,
    pub blake3: [u8; 32],
}

impl IndicatorBankWireHeader {
    /// Parse a header from the first 48 bytes of a frozen wire blob.
    ///
    /// Returns `None` if too short, magic mismatched, or version unsupported.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        const HEADER_LEN: usize = 4 + 8 + 2 + 2 + 32;
        if bytes.len() < HEADER_LEN {
            return None;
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[..4]);
        if magic != BANK_MAGIC {
            return None;
        }
        let version = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
        if version != BANK_WIRE_VERSION {
            return None;
        }
        let n = u16::from_le_bytes(bytes[12..14].try_into().ok()?);
        let d = u16::from_le_bytes(bytes[14..16].try_into().ok()?);
        let mut blake3 = [0u8; 32];
        blake3.copy_from_slice(&bytes[16..48]);
        Some(Self {
            magic,
            version,
            n,
            d,
            blake3,
        })
    }
}

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

/// Reinterpret an `&[f32]` as `&[u8]` for hashing / serialization.
/// Relies on `f32` being 4-byte aligned (always true for Vec<f32> / &mut [f32]).
#[inline]
fn to_bytes(xs: &[f32]) -> &[u8] {
    // SAFETY: f32 is transparent over [u8; 4]; the slice is well-aligned and
    // sized. This is the standard bytemuck-free reinterp; alignment is
    // guaranteed by Rust's slice allocator.
    let (prefix, mid, suffix) = unsafe { xs.align_to::<u8>() };
    debug_assert!(prefix.is_empty() && suffix.is_empty());
    mid
}

/// Reinterpret an `&[u8]` body (a whole number of f32s) as `&[f32]`.
#[inline]
fn from_bytes<T: bytemuck::Pod>(b: &[u8]) -> &[T] {
    bytemuck::cast_slice(b)
}

/// 8-byte hex prefix of a BLAKE3 for debug printing (first 4 bytes shown).
#[inline]
fn hex_prefix(h: &[u8; 32]) -> String {
    use core::fmt::Write;
    let mut s = String::with_capacity(11);
    for b in &h[..4] {
        let _ = write!(s, "{:02x}", b);
    }
    s.push_str("..");
    s
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    type Bank = IndicatorProbeBank<DemoIndicatorLabel, 4>;

    /// Build a demo bank with known direction vectors so tests are deterministic.
    fn demo_bank() -> Bank {
        // 3 labels × D=4: orthogonal-ish unit directions + small thresholds.
        let directions: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // A
            0.0, 1.0, 0.0, 0.0, // B
            0.0, 0.0, 1.0, 0.0, // C
        ];
        let thresholds: Vec<f32> = vec![0.0, 0.0, 0.0];
        Bank::new(directions, thresholds).expect("demo bank shape ok")
    }

    #[test]
    fn test_project_returns_sigmoid_in_unit_interval() {
        let bank = demo_bank();
        // Zero state → raw dot = 0, threshold = 0 → sigmoid(0) = 0.5.
        let zero = [0.0f32; 4];
        for l in [DemoIndicatorLabel::A, DemoIndicatorLabel::B, DemoIndicatorLabel::C] {
            let s = bank.project(&zero, l);
            assert!(
                (s - 0.5).abs() < 1e-6,
                "zero-state project for {:?} should be 0.5, got {}",
                l,
                s
            );
            assert!((0.0..=1.0).contains(&s), "score must be in (0,1): {}", s);
        }
        // State == direction → raw dot = ‖d‖² = 1, threshold = 0 → sigmoid(1).
        let state_a = [1.0f32, 0.0, 0.0, 0.0];
        let s = bank.project(&state_a, DemoIndicatorLabel::A);
        let expected = fast_sigmoid(1.0 - 0.0);
        assert!((s - expected).abs() < 1e-6, "direction-self project: {} vs {}", s, expected);
    }

    #[test]
    fn test_project_all_into_writes_all_n_scores() {
        let bank = demo_bank();
        let state = [1.0f32, 2.0, 3.0, 0.0];
        let mut scores = [0.0f32; DemoIndicatorLabel::COUNT];
        bank.project_all_into(&state, &mut scores);
        assert_eq!(scores.len(), DemoIndicatorLabel::COUNT);
        // Cross-check against single project.
        let all_labels = [
            DemoIndicatorLabel::A,
            DemoIndicatorLabel::B,
            DemoIndicatorLabel::C,
        ];
        for (i, l) in all_labels.iter().enumerate() {
            let single = bank.project(&state, *l);
            assert!(
                (single - scores[i]).abs() < 1e-6,
                "project_all_into[{}] = {} != project = {}",
                i,
                scores[i],
                single
            );
        }
    }

    #[test]
    fn test_or_fused_fire_none_below_tau() {
        let bank = demo_bank();
        // All scores below tau → None.
        let scores = [0.4f32, 0.3, 0.2];
        assert_eq!(bank.or_fused_fire(&scores, 0.5), None);
        // tau exactly equal to max score → None (strict >).
        let scores2 = [0.5f32, 0.3, 0.2];
        assert_eq!(bank.or_fused_fire(&scores2, 0.5), None);
    }

    #[test]
    fn test_or_fused_fire_argmax_above_tau() {
        let bank = demo_bank();
        let scores = [0.4f32, 0.9, 0.2];
        assert_eq!(
            bank.or_fused_fire(&scores, 0.5),
            Some(DemoIndicatorLabel::B),
            "B has the highest score above tau"
        );
    }

    #[test]
    fn test_or_fused_fire_tie_breaks_by_lowest_index() {
        let bank = demo_bank();
        // B and C both at 0.9, above tau 0.5 → lower index (B) wins.
        let scores = [0.1f32, 0.9, 0.9];
        assert_eq!(
            bank.or_fused_fire(&scores, 0.5),
            Some(DemoIndicatorLabel::B),
            "ties must break by lowest index"
        );
        // A and B tied → A (index 0).
        let scores2 = [0.9f32, 0.9, 0.1];
        assert_eq!(bank.or_fused_fire(&scores2, 0.5), Some(DemoIndicatorLabel::A));
    }

    #[test]
    fn test_indicator_label_u8_round_trip() {
        let all = [
            DemoIndicatorLabel::A,
            DemoIndicatorLabel::B,
            DemoIndicatorLabel::C,
        ];
        for l in all {
            let d = l.as_u8();
            let back = DemoIndicatorLabel::from_u8(d).expect("valid discriminant");
            assert_eq!(l, back, "round-trip failed for {:?}", l);
        }
        // Out-of-range → None.
        assert!(DemoIndicatorLabel::from_u8(3).is_none());
        assert!(DemoIndicatorLabel::from_u8(255).is_none());
        assert_eq!(DemoIndicatorLabel::COUNT, 3);
    }

    #[test]
    fn test_from_frozen_bytes_round_trip() {
        let bank = demo_bank();
        let bytes = bank.to_frozen_bytes();
        let reloaded = Bank::from_frozen_bytes(&bytes).expect("round-trip loads");
        assert_eq!(bank, reloaded, "round-trip must produce equal bank");
        assert_eq!(reloaded.blake3(), bank.blake3());
        assert_eq!(reloaded.version(), bank.version());
        assert_eq!(reloaded.n(), DemoIndicatorLabel::COUNT);
        assert_eq!(reloaded.d(), 4);
    }

    #[test]
    fn test_from_frozen_bytes_rejects_tampered_hash() {
        let bank = demo_bank();
        let mut bytes = bank.to_frozen_bytes();
        // Flip one byte in the directions body (after the 48-byte header).
        let header_len = 48;
        bytes[header_len] ^= 0xFF;
        let err = Bank::from_frozen_bytes(&bytes).unwrap_err();
        assert_eq!(err, BankLoadError::HashMismatch, "tamper must be detected");
    }

    #[test]
    fn test_from_frozen_bytes_rejects_bad_magic() {
        let bank = demo_bank();
        let mut bytes = bank.to_frozen_bytes();
        bytes[0] = b'X';
        assert_eq!(
            Bank::from_frozen_bytes(&bytes).unwrap_err(),
            BankLoadError::BadMagic
        );
    }

    #[test]
    fn test_from_frozen_bytes_rejects_too_short() {
        let short = [0u8; 10];
        assert_eq!(
            Bank::from_frozen_bytes(&short).unwrap_err(),
            BankLoadError::TooShort
        );
    }

    #[test]
    fn test_from_frozen_bytes_rejects_truncated_body() {
        let bank = demo_bank();
        let mut bytes = bank.to_frozen_bytes();
        // Truncate the body by one byte.
        let truncated_len = bytes.len() - 1;
        bytes.truncate(truncated_len);
        match Bank::from_frozen_bytes(&bytes).unwrap_err() {
            BankLoadError::Truncated { expected, got } => {
                assert_eq!(got, truncated_len);
                assert_ne!(expected, truncated_len);
            }
            other => panic!("expected Truncated, got {:?}", other),
        }
    }

    #[test]
    fn test_new_rejects_mismatched_shapes() {
        // directions too short.
        assert!(Bank::new(vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]).is_none());
        // thresholds too short.
        assert!(
            Bank::new(
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0]
            )
            .is_none()
        );
    }

    #[test]
    fn test_wire_header_parse_round_trip() {
        let bank = demo_bank();
        let bytes = bank.to_frozen_bytes();
        let hdr = IndicatorBankWireHeader::parse(&bytes).expect("parse header");
        assert_eq!(hdr.magic, BANK_MAGIC);
        assert_eq!(hdr.version, BANK_WIRE_VERSION);
        assert_eq!(hdr.n as usize, DemoIndicatorLabel::COUNT);
        assert_eq!(hdr.d as usize, 4);
        assert_eq!(hdr.blake3, bank.blake3());
    }
}
