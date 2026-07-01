//! BLAKE3 commitment for BabelCodec compressed payloads (Plan 331 Phase 4).
//!
//! [`BabelCommitment`] is a `[u8; 32]` BLAKE3 digest newtype over the
//! compressed bytes. It is the load-bearing piece for the future LatCal
//! chain-commitment bridge (`.issues/002_deterministic_babeltele_chain_commitment.md`):
//! because the BT-P8 codec is deterministic, two independent parties
//! compressing the same input produce byte-identical compressed bytes and thus
//! identical commitments — enabling trust-minimized commitment of semantic KG
//! triples at lower byte cost than the uncompressed form.
//!
//! # Determinism
//!
//! BLAKE3 is a deterministic, cross-platform hash: same input bytes → same
//! `[u8; 32]` on every architecture (ARM64 / x86_64 / wasm32). The
//! cross-architecture check (G5) is therefore trivially satisfied for any
//! deterministic codec — what is non-trivial is the codec itself producing
//! byte-identical compressed bytes across architectures, which the fixed-rule
//! mapping guarantees by construction (no float math in the text codec path).
//!
//! # Why BLAKE3 (per AGENTS.md)
//!
//! AGENTS.md mandates BLAKE3 over SHA1/SHA256 for all commitment paths in this
//! codebase. `blake3` is already a non-optional dependency of `katgpt-core`,
//! so this module adds zero new deps.

use core::fmt;

/// BLAKE3 `[u8; 32]` commitment of a BabelCodec compressed payload.
///
/// Construct via [`BabelCommitment::of`] (hash raw bytes) or
/// [`BabelCommitment::from_bytes`] (wrap a precomputed digest). Compare with
/// [`BabelCommitment::matches`].
///
/// `Copy` + `Eq` + `Hash` (via the inner array) so it can be used as a key in
/// content-addressed stores and dedup maps.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct BabelCommitment([u8; 32]);

impl BabelCommitment {
    /// Compute the BLAKE3 digest of `bytes` and wrap it.
    ///
    /// Deterministic across architectures (BLAKE3 is portable, no float math).
    #[inline]
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    /// Wrap a precomputed 32-byte digest (no rehash).
    ///
    /// Use when the digest was already computed elsewhere and you want to
    /// avoid the rehash. The caller is responsible for ensuring the digest
    /// was computed with the same hashing convention ([`Self::of`]).
    #[inline]
    pub const fn from_bytes(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    /// Access the raw 32-byte digest.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Recompute the commitment of `bytes` and compare in constant time
    /// (single short-circuit on first mismatch is acceptable for a 32-byte
    /// comparison; the goal is functional equality, not side-channel defense).
    #[inline]
    pub fn matches(&self, bytes: &[u8]) -> bool {
        // Recompute and compare the 32-byte digests.
        let recomputed = blake3::hash(bytes);
        &self.0 == recomputed.as_bytes()
    }

    /// All-zero commitment — the canonical "no payload" / empty-input digest.
    ///
    /// Note: this is NOT the BLAKE3 of the empty string (that is a specific
    /// non-zero digest). This is the all-zero placeholder, useful as a sentinel
    /// "no commitment computed yet" value.
    #[inline]
    pub const fn zero() -> Self {
        Self([0u8; 32])
    }

    /// True if this is the all-zero sentinel ([`Self::zero`]).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

impl fmt::Debug for BabelCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BabelCommitment(")?;
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        write!(f, ")")
    }
}

impl From<[u8; 32]> for BabelCommitment {
    #[inline]
    fn from(digest: [u8; 32]) -> Self {
        Self(digest)
    }
}

impl From<BabelCommitment> for [u8; 32] {
    #[inline]
    fn from(c: BabelCommitment) -> Self {
        c.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_of_is_deterministic() {
        // Same input → same digest.
        let c1 = BabelCommitment::of(b"hello world");
        let c2 = BabelCommitment::of(b"hello world");
        assert_eq!(c1, c2, "BLAKE3 must be deterministic");
    }

    #[test]
    fn commitment_differs_for_different_input() {
        let c1 = BabelCommitment::of(b"hello world");
        let c2 = BabelCommitment::of(b"hello world!");
        assert_ne!(c1, c2, "different inputs must produce different commitments");
    }

    #[test]
    fn commitment_matches_returns_true_for_same_input() {
        let payload = b"some compressed bytes";
        let c = BabelCommitment::of(payload);
        assert!(c.matches(payload), "matches must return true for the original payload");
    }

    #[test]
    fn commitment_matches_returns_false_for_tampered_input() {
        let payload = b"some compressed bytes";
        let tampered = b"some compressed bytes "; // appended space
        let c = BabelCommitment::of(payload);
        assert!(
            !c.matches(tampered),
            "matches must return false for a tampered payload (tamper detection)"
        );
    }

    #[test]
    fn commitment_of_empty_input_is_well_defined_and_nonzero() {
        // BLAKE3 of the empty string is a specific known digest.
        let c = BabelCommitment::of(b"");
        assert!(!c.is_zero(), "empty-input commitment must be a real BLAKE3 digest, not the zero sentinel");
        // Cross-checked against a fresh computation.
        let expected: [u8; 32] = *blake3::hash(b"").as_bytes();
        assert_eq!(c.as_bytes(), &expected);
    }

    #[test]
    fn commitment_zero_sentinel_is_all_zero() {
        let z = BabelCommitment::zero();
        assert!(z.is_zero(), "zero() sentinel must be all zeros");
        // The empty-input BLAKE3 is NOT the zero sentinel.
        assert!(!BabelCommitment::of(b"").is_zero());
    }

    #[test]
    fn commitment_debug_formats_as_hex() {
        let c = BabelCommitment::of(b"abc");
        let s = format!("{c:?}");
        assert!(s.starts_with("BabelCommitment("), "debug must start with BabelCommitment(: {s}");
        assert!(s.ends_with(')'), "debug must end with ): {s}");
        // 64 hex chars + 17 prefix + 1 suffix = 82.
        assert_eq!(s.len(), "BabelCommitment(".len() + 64 + 1);
        // Hex chars only in the middle.
        let hex = &s["BabelCommitment(".len()..s.len() - 1];
        for b in hex.bytes() {
            assert!(b.is_ascii_hexdigit(), "non-hex char in commitment debug: {hex}");
        }
    }

    #[test]
    fn commitment_round_trips_through_array_conversion() {
        let c = BabelCommitment::of(b"roundtrip");
        let arr: [u8; 32] = c.into();
        let c2 = BabelCommitment::from(arr);
        assert_eq!(c, c2);
    }

    #[test]
    fn commitment_of_compressed_bytes_is_cross_arch_deterministic_documentation() {
        // This is the load-bearing property for issue #002 (LatCal chain
        // commitment). We cannot test cross-architecture here — that is the G5
        // bench's job. What we CAN test is that two calls produce the same
        // digest, which is a necessary (but not sufficient) condition.
        let payload = b"*(entity):key=value";
        let c1 = BabelCommitment::of(payload);
        let c2 = BabelCommitment::of(payload);
        assert_eq!(c1, c2);
        // Cross-arch determinism is a property of BLAKE3 (portable C reference
        // impl, no float math). The text codec produces ASCII bytes with no
        // float math either, so the cross-arch guarantee holds by construction.
        // G5 bench verifies this on ARM64 + x86_64 (+ wasm32 if feasible).
    }

    #[test]
    fn commitment_from_bytes_does_not_rehash() {
        // Construct a digest directly; verify it round-trips.
        let raw = [0x42u8; 32];
        let c = BabelCommitment::from_bytes(raw);
        assert_eq!(c.as_bytes(), &raw);
        // And matches() on the bytes whose hash equals `raw`... we don't know
        // such input here, so we only check the accessor.
    }

    #[test]
    fn commitment_copy_clone_eq_hash() {
        let c1 = BabelCommitment::of(b"clone me");
        let c2 = c1; // Copy
        let c3 = c1;
        assert_eq!(c1, c2);
        assert_eq!(c1, c3);
        // Hash: collect into a HashSet to exercise Hash.
        let set = core::iter::once(c1).chain(core::iter::once(c2)).collect::<std::collections::HashSet<_>>();
        assert_eq!(set.len(), 1, "Copy + Hash must dedup identical commitments");
    }
}
