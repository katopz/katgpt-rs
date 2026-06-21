//! [`PersonalitySnapshot`] — the freeze/thaw artifact for a
//! [`PersonalityWeightedComposition`] (Plan 297 Phase 3 T3.1).
//!
//! A snapshot is a versioned, BLAKE3-committed blob of the personality weights
//! `w` plus an archetype label. It is the *personality artifact* of an entity:
//! two same-type NPCs can diverge by holding different snapshot versions.
//!
//! # Sync boundary
//!
//! - The **weights blob** (`w`) is latent, local, never synced. Syncing it
//!   would destroy per-entity personality divergence.
//! - The **commitment** (BLAKE3 hash + version + archetype) IS synced as an
//!   audit event when a hot-swap occurs.
//!
//! # Commitment scheme
//!
//! BLAKE3 over the streaming input `archetype_bytes || w_bytes_le`:
//!
//! ```text
//! hasher.update(archetype.as_bytes());   // 16 bytes
//! for &wi in &w {
//!     hasher.update(&wi.to_le_bytes());  // 4 bytes each
//! }
//! self.blake3 = *hasher.finalize().as_bytes();
//! ```
//!
//! This matches the streaming-`Hasher` pattern used by
//! [`MicroRecurrentKernelSnapshot`](crate::micro_belief::snapshot::MicroRecurrentKernelSnapshot)
//! (R242) and is independent of struct layout / padding.
//!
//! # `version`
//!
//! Monotonic per-entity counter, incremented on each hot-swap. Combined with
//! `blake3`, it forms the audit key for personality provenance. Per R242
//! precedent, `version` is NOT part of the BLAKE3 input — two snapshots with
//! identical `w` but different versions are the same personality at different
//! points in time.

use crate::personality_composition::kernel::PersonalityWeightedComposition;
use crate::personality_composition::types::ArchetypeLabel;

/// Snapshot version format. Bump if `w` layout or hashing scheme changes.
///
/// Currently `1` — corresponds to the `[f32; N]` LE-bytes hashing scheme
/// established in Plan 297 Phase 3 T3.1.
pub const SNAPSHOT_VERSION: u64 = 1;

/// A versioned, BLAKE3-committed snapshot of a
/// [`PersonalityWeightedComposition`]'s weights.
///
/// Construct via [`from_composition`](Self::from_composition) (computes the
/// hash) or [`from_parts`](Self::from_parts) (raw fields, for deserialisation
/// paths where the hash is already known).
///
/// # Const generic `N`
///
/// The snapshot carries the full `[f32; N]` weight array directly (unlike
/// [`MicroRecurrentKernelSnapshot`] which uses a `Vec<u8>` blob, because
/// attractor weights are variable-size `dim × dim`). Personality weights are
/// fixed-size `N` so a fixed array is simpler and allocation-free.
///
/// # Serialization
///
/// serde's derive does not support `[f32; N]` for arbitrary `N` (only up to
/// 32 on stable). We intentionally skip `serde::Serialize`/`Deserialize`
/// derives here and rely on BLAKE3 + raw bytes for integrity. Callers that
/// need serde persistence should serialize via [`to_bytes`](Self::to_bytes)
/// (which is layout-stable) and verify via [`verify_blake3`](Self::verify_blake3).
#[derive(Clone, Debug)]
pub struct PersonalitySnapshot<const N: usize> {
    /// Personality weights at snapshot time. Hashed into `blake3`.
    pub w: [f32; N],

    /// Archetype label. Hashed into `blake3` (so different archetypes with
    /// identical weights produce different commitments).
    pub archetype: ArchetypeLabel,

    /// BLAKE3 commitment over `(archetype, w)`. Filled by
    /// [`commit`](Self::commit); zeroed during hashing so the commitment
    /// doesn't feed back into itself.
    pub blake3: [u8; 32],

    /// Monotonic version counter (caller-managed). NOT part of the BLAKE3
    /// input — two snapshots with identical `w` + `archetype` but different
    /// versions are the same personality at different points in time.
    pub version: u64,
}

impl<const N: usize, const D: usize> PersonalityWeightedComposition<N, D> {
    /// Build a snapshot from the composition's current weights.
    ///
    /// Copies `w` out, records the archetype, and computes the BLAKE3
    /// commitment. `version` is caller-managed — typically incremented by the
    /// hot-swap layer on each swap.
    ///
    /// This is NOT on the hot path — snapshots are rare (per-entity
    /// personality version events, not per-tick).
    pub fn snapshot(&self, archetype: ArchetypeLabel, version: u64) -> PersonalitySnapshot<N> {
        PersonalitySnapshot::from_parts_with_commit(self.w, archetype, version)
    }
}

impl<const N: usize> PersonalitySnapshot<N> {
    /// Build a snapshot from a composition's current weights.
    ///
    /// This is a convenience constructor — prefer
    /// [`PersonalityWeightedComposition::snapshot`] at call sites (it resolves
    /// without needing to re-state `N`).
    ///
    /// Copies `w` out of the composition, records the archetype, and computes
    /// the BLAKE3 commitment. `version` is caller-managed — typically
    /// incremented by the hot-swap layer on each swap.
    ///
    /// This is NOT on the hot path — snapshots are rare (per-entity
    /// personality version events, not per-tick).
    pub fn from_composition<const D: usize>(
        composition: &PersonalityWeightedComposition<N, D>,
        archetype: ArchetypeLabel,
        version: u64,
    ) -> Self {
        Self::from_parts_with_commit(composition.w, archetype, version)
    }
    /// Build a snapshot from raw parts, computing the BLAKE3 commitment fresh.
    ///
    /// Internal helper shared by [`from_composition`](Self::from_composition)
    /// and
    /// [`PersonalityWeightedComposition::snapshot`].
    fn from_parts_with_commit(w: [f32; N], archetype: ArchetypeLabel, version: u64) -> Self {
        let mut snap = Self {
            w,
            archetype,
            blake3: [0u8; 32],
            version,
        };
        snap.commit();
        snap
    }

    /// Build a snapshot from raw parts WITHOUT computing the commitment.
    ///
    /// Useful for deserialisation paths where the commitment is already known
    /// (e.g. loading from disk). Call [`commit`](Self::commit) afterwards if
    /// you need to recompute, or [`verify_blake3`](Self::verify_blake3) to
    /// check integrity.
    pub fn from_parts(
        w: [f32; N],
        archetype: ArchetypeLabel,
        blake3: [u8; 32],
        version: u64,
    ) -> Self {
        Self {
            w,
            archetype,
            blake3,
            version,
        }
    }

    /// Compute (or recompute) the BLAKE3 commitment over `(archetype, w)`.
    ///
    /// Idempotent: calling twice produces the same hash. The existing
    /// `self.blake3` is zeroed internally before hashing so the commitment
    /// never feeds back into itself.
    pub fn commit(&mut self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.archetype.as_bytes());
        for &wi in &self.w {
            hasher.update(&wi.to_le_bytes());
        }
        let hash = *hasher.finalize().as_bytes();
        self.blake3 = hash;
        hash
    }

    /// Recompute the commitment and compare with the stored `self.blake3`.
    ///
    /// Returns `true` iff the stored weights produce the stored hash. A
    /// `false` result indicates tampering or corruption.
    pub fn verify_blake3(&self) -> bool {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.archetype.as_bytes());
        for &wi in &self.w {
            hasher.update(&wi.to_le_bytes());
        }
        let recomputed = *hasher.finalize().as_bytes();
        recomputed == self.blake3
    }

    /// Serialize the snapshot to a flat byte buffer (LE f32 weights + archetype
    /// + blake3 + version).
    ///
    /// Layout: `archetype (16) || w (N*4 LE) || blake3 (32) || version (8 LE)`.
    /// Total: `56 + N*4` bytes.
    ///
    /// This is the persistence format for cold-tier storage. The BLAKE3 in
    /// the buffer covers `(archetype, w)` only — `version` is metadata.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.w.len() * 4 + 32 + 8);
        buf.extend_from_slice(self.archetype.as_bytes());
        for &wi in &self.w {
            buf.extend_from_slice(&wi.to_le_bytes());
        }
        buf.extend_from_slice(&self.blake3);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf
    }

    /// Deserialize from a flat byte buffer produced by [`to_bytes`](Self::to_bytes).
    ///
    /// Returns `None` if the buffer is malformed (wrong length). Does NOT
    /// re-verify the BLAKE3 — call [`verify_blake3`](Self::verify_blake3)
    /// afterwards to check integrity.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        let expected_len = 16 + N * 4 + 32 + 8;
        if buf.len() != expected_len {
            return None;
        }
        let archetype_bytes: [u8; 16] = buf[0..16].try_into().ok()?;
        let archetype = ArchetypeLabel::new(archetype_bytes);
        let mut w = [0.0f32; N];
        for (i, slot) in w.iter_mut().enumerate() {
            let off = 16 + i * 4;
            *slot = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        }
        let blake3: [u8; 32] = buf[16 + N * 4..16 + N * 4 + 32].try_into().ok()?;
        let version =
            u64::from_le_bytes(buf[16 + N * 4 + 32..16 + N * 4 + 32 + 8].try_into().ok()?);
        Some(Self {
            w,
            archetype,
            blake3,
            version,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::personality_composition::kernel::PersonalityWeightedComposition;
    use crate::personality_composition::types::PersonalityConfig;

    #[test]
    fn snapshot_roundtrips_commit() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let snap = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::empty(), 1);
        assert!(
            snap.verify_blake3(),
            "freshly-committed snapshot must verify"
        );
        assert_eq!(snap.w, [0.1, -0.2, 0.3]);
        assert_eq!(snap.version, 1);
        assert_ne!(
            snap.blake3, [0u8; 32],
            "blake3 must be non-zero after commit"
        );
    }

    #[test]
    fn commit_is_idempotent() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let mut snap = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::empty(), 1);
        let h1 = snap.blake3;
        let h2 = snap.commit();
        assert_eq!(h1, h2, "commit must be idempotent");
    }

    #[test]
    fn tampered_w_fails_verify() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let mut snap = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::empty(), 1);
        assert!(snap.verify_blake3());
        // Mutate a weight — must break the commitment.
        snap.w[0] += 0.001;
        assert!(!snap.verify_blake3(), "tampered w must fail verify");
    }

    #[test]
    fn different_weights_produce_different_commitment() {
        let k1 = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let k2 = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.4],
        );
        let s1 = PersonalitySnapshot::from_composition(&k1, ArchetypeLabel::empty(), 1);
        let s2 = PersonalitySnapshot::from_composition(&k2, ArchetypeLabel::empty(), 1);
        assert_ne!(s1.blake3, s2.blake3, "different w → different hash");
    }

    #[test]
    fn different_archetype_produces_different_commitment() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let s1 = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::from_str("predator"), 1);
        let s2 = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::from_str("prey"), 1);
        assert_ne!(s1.blake3, s2.blake3, "different archetype → different hash");
    }

    #[test]
    fn version_does_not_affect_commitment() {
        // Two snapshots with identical w + archetype but different versions
        // should have the SAME blake3 — version is metadata, not contents.
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let s1 = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::empty(), 1);
        let s2 = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::empty(), 999);
        assert_eq!(s1.blake3, s2.blake3, "version must not affect blake3");
        assert_ne!(s1.version, s2.version);
    }

    #[test]
    fn bytes_roundtrip_preserves_all_fields() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let snap =
            PersonalitySnapshot::from_composition(&k, ArchetypeLabel::from_str("predator"), 7);
        let bytes = snap.to_bytes();
        let back = PersonalitySnapshot::<3>::from_bytes(&bytes).expect("deserialize");
        assert_eq!(back.w, snap.w);
        assert_eq!(back.archetype, snap.archetype);
        assert_eq!(back.blake3, snap.blake3);
        assert_eq!(back.version, snap.version);
        assert!(
            back.verify_blake3(),
            "deserialised snapshot must still verify"
        );
    }

    #[test]
    fn from_bytes_rejects_wrong_length() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let snap = PersonalitySnapshot::from_composition(&k, ArchetypeLabel::empty(), 1);
        let bytes = snap.to_bytes();
        // Truncate by one byte — must fail.
        let truncated = &bytes[..bytes.len() - 1];
        assert!(PersonalitySnapshot::<3>::from_bytes(truncated).is_none());
        // Expand by one byte — must fail.
        let mut expanded = bytes.clone();
        expanded.push(0);
        assert!(PersonalitySnapshot::<3>::from_bytes(&expanded).is_none());
    }

    #[test]
    fn restore_roundtrip_via_from_parts() {
        // from_parts is the "I already know the hash" constructor.
        let w = [0.5f32, -0.5, 0.0, 0.25];
        let archetype = ArchetypeLabel::from_str("test");
        let k = PersonalityWeightedComposition::<4, 32>::new(PersonalityConfig::default(), w);
        let original = PersonalitySnapshot::from_composition(&k, archetype, 3);

        // Pretend we serialised + deserialised, preserving blake3.
        let restored = PersonalitySnapshot::from_parts(w, archetype, original.blake3, 3);
        assert!(restored.verify_blake3());
        assert_eq!(restored.blake3, original.blake3);
    }

    /// G6: build → mutate → mismatch → restore → match.
    #[test]
    fn g6_build_mutate_mismatch_restore_match() {
        let mut k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let archetype = ArchetypeLabel::from_str("predator");

        // 1. Build snapshot from initial state.
        let snap = PersonalitySnapshot::from_composition(&k, archetype, 1);
        assert!(snap.verify_blake3());

        // 2. Mutate w in-place (simulated drift).
        k.w[0] += 1.0;
        k.w[1] -= 0.5;

        // 3. The old snapshot must NOT match the mutated state.
        let mutated_snap = PersonalitySnapshot::from_composition(&k, archetype, 2);
        assert_ne!(
            snap.blake3, mutated_snap.blake3,
            "mutated w must produce a different commitment"
        );

        // 4. Restore from the snapshot.
        k.restore_w(snap.w);

        // 5. Re-snapshot must now match the original.
        let restored_snap = PersonalitySnapshot::from_composition(&k, archetype, 1);
        assert_eq!(
            snap.blake3, restored_snap.blake3,
            "restored w must produce the original commitment"
        );
    }
}
