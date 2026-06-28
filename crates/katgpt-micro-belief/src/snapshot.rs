//! `MicroRecurrentKernelSnapshot` — the freeze/thaw artifact for a
//! `MicroRecurrentBeliefState`.
//!
//! A snapshot is a versioned, BLAKE3-committed blob of kernel weights. It is
//! the *personality artifact* of an entity: two same-type NPCs can diverge by
//! holding different snapshot versions (Plan 276 / Research 242 §2.1).
//!
//! # Sync boundary
//!
//! - The **weights blob** is latent, local, never synced. Syncing it would
//!   destroy per-entity personality divergence and waste bandwidth.
//! - The **commitment** (BLAKE3 hash + version) IS synced as an audit event
//!   when a hot-swap occurs — clients can verify that the entity they observe
//!   matches a committed personality, without learning the weights themselves.
//!
//! # Commitment scheme
//!
//! BLAKE3 over the streaming input `family_byte || dim_le_bytes || weights_blob`:
//!
//! ```text
//! hasher.update(&[self.family as u8]);
//! hasher.update(&(self.dim as u64).to_le_bytes());
//! hasher.update(&self.weights_blob);
//! self.blake3 = *hasher.finalize().as_bytes();
//! ```
//!
//! This matches the streaming-`Hasher` pattern used by `GpartAdapter::commitment`
//! (`katgpt-rs/crates/katgpt-core/src/types.rs` L3233–3245) and is independent
//! of struct layout / padding — only the logical fields contribute to the hash.
//!
//! # `version`
//!
//! Monotonically-increasing per-entity counter, incremented on each hot-swap.
//! Combined with `blake3`, it forms the audit key for personality provenance.
//! Per AGENTS.md we'd normally use `Uuid::now_v7()` for IDs, but this struct
//! is the *contents* of a personality version, not the ID of an event — the
//! caller (riir-ai's `KernelHotSwap`) is free to tag the swap event itself
//! with a v7 UUID.

use crate::types::{MicroRecurrentBeliefState, RecurrenceFamily};

/// Snapshot version format. Bump if `weights_blob` layout changes.
///
/// Currently `1` — corresponds to the `AttractorKernel::to_snapshot_blob` layout
/// `ws (dim*dim*4) || wx (dim*dim*4) || b (dim*4)` and the Family C layout
/// (future T2.1 will define its own blob layout and bump this if needed).
pub const SNAPSHOT_VERSION: u64 = 1;

/// A versioned, BLAKE3-committed snapshot of a `MicroRecurrentBeliefState`'s
/// weights.
///
/// Construct via [`from_kernel`](Self::from_kernel) (caller provides the
/// serialised weights blob) or [`from_parts`](Self::from_parts) (raw fields).
/// Verify integrity via [`verify`](Self::verify).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MicroRecurrentKernelSnapshot {
    /// Recurrence family the weights belong to.
    pub family: RecurrenceFamily,
    /// Belief-vector dimension the kernel was constructed with.
    pub dim: usize,
    /// Serialised kernel weights. Layout depends on `family`:
    /// - `Attractor`: `ws (dim*dim*4) || wx (dim*dim*4) || b (dim*4)` LE f32
    ///   (see `AttractorKernel::to_snapshot_blob`).
    /// - `DeltaRule`: TBD by future T2.1 (likely `(lr, max_delta)` as 8 bytes).
    /// - `LatentThought`: TBD by Phase 3 T3.1.
    pub weights_blob: Vec<u8>,
    /// BLAKE3 commitment over `(family, dim, weights_blob)`. Filled by
    /// [`commit`](Self::commit); zeroed during hashing (so the commitment does
    /// not contribute to its own hash).
    pub blake3: [u8; 32],
    /// Monotonic version counter (caller-managed). NOT part of the BLAKE3 input
    /// — two snapshots with identical weights but different versions are the
    /// *same* personality at different points in time.
    pub version: u64,
}

impl MicroRecurrentKernelSnapshot {
    /// Build a snapshot from a kernel and a caller-provided weights blob.
    ///
    /// The caller is responsible for serialising the kernel's weights into
    /// `weights_blob` (e.g. via `AttractorKernel::to_snapshot_blob`). This
    /// function records `family` + `dim` from the kernel, stores the blob, and
    /// computes the BLAKE3 commitment.
    ///
    /// `version` is caller-managed — typically incremented by the hot-swap
    /// layer on each swap.
    pub fn from_kernel<K: MicroRecurrentBeliefState>(
        kernel: &K,
        weights_blob: Vec<u8>,
        version: u64,
    ) -> Self {
        let mut snap = Self {
            family: kernel.family(),
            dim: kernel.dim(),
            weights_blob,
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
    /// you need to recompute, or [`verify`](Self::verify) to check integrity.
    pub fn from_parts(
        family: RecurrenceFamily,
        dim: usize,
        weights_blob: Vec<u8>,
        blake3: [u8; 32],
        version: u64,
    ) -> Self {
        Self { family, dim, weights_blob, blake3, version }
    }

    /// Compute (or recompute) the BLAKE3 commitment over
    /// `(family, dim, weights_blob)`.
    ///
    /// Idempotent: calling twice produces the same hash. The existing
    /// `self.blake3` is zeroed internally before hashing so the commitment
    /// never feeds back into itself.
    pub fn commit(&mut self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[self.family as u8]);
        hasher.update(&(self.dim as u64).to_le_bytes());
        hasher.update(&self.weights_blob);
        let hash = *hasher.finalize().as_bytes();
        self.blake3 = hash;
        hash
    }

    /// Recompute the commitment and compare with the stored `self.blake3`.
    ///
    /// Returns `true` iff the stored weights produce the stored hash. A `false`
    /// result indicates tampering or corruption.
    pub fn verify(&self) -> bool {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[self.family as u8]);
        hasher.update(&(self.dim as u64).to_le_bytes());
        hasher.update(&self.weights_blob);
        let recomputed = *hasher.finalize().as_bytes();
        recomputed == self.blake3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attractor::AttractorKernel;

    #[test]
    fn snapshot_roundtrips_commit() {
        let k = AttractorKernel::from_seed(42, 32);
        let blob = k.to_snapshot_blob();
        let snap = MicroRecurrentKernelSnapshot::from_kernel(&k, blob.clone(), 1);
        assert!(snap.verify(), "freshly-committed snapshot must verify");
        assert_eq!(snap.family, RecurrenceFamily::Attractor);
        assert_eq!(snap.dim, 32);
        assert_eq!(snap.weights_blob, blob);
        assert_eq!(snap.version, 1);
        assert_ne!(snap.blake3, [0u8; 32], "blake3 must be non-zero after commit");
    }

    #[test]
    fn commit_is_idempotent() {
        let k = AttractorKernel::from_seed(42, 32);
        let blob = k.to_snapshot_blob();
        let mut snap = MicroRecurrentKernelSnapshot::from_kernel(&k, blob, 1);
        let h1 = snap.blake3;
        let h2 = snap.commit();
        assert_eq!(h1, h2, "commit must be idempotent");
    }

    #[test]
    fn tampered_blob_fails_verify() {
        let k = AttractorKernel::from_seed(42, 32);
        let blob = k.to_snapshot_blob();
        let mut snap = MicroRecurrentKernelSnapshot::from_kernel(&k, blob, 1);
        assert!(snap.verify());
        // Flip a byte in the weights blob — must break the commitment.
        snap.weights_blob[0] ^= 0xFF;
        assert!(!snap.verify(), "tampered blob must fail verify");
    }

    #[test]
    fn different_seed_produces_different_commitment() {
        let k1 = AttractorKernel::from_seed(1, 32);
        let k2 = AttractorKernel::from_seed(2, 32);
        let s1 = MicroRecurrentKernelSnapshot::from_kernel(&k1, k1.to_snapshot_blob(), 1);
        let s2 = MicroRecurrentKernelSnapshot::from_kernel(&k2, k2.to_snapshot_blob(), 1);
        assert_ne!(s1.blake3, s2.blake3, "different weights → different hash");
    }

    #[test]
    fn version_does_not_affect_commitment() {
        // Two snapshots with identical weights but different versions should
        // have the SAME blake3 — version is metadata, not personality contents.
        let k = AttractorKernel::from_seed(42, 32);
        let blob = k.to_snapshot_blob();
        let s1 = MicroRecurrentKernelSnapshot::from_kernel(&k, blob.clone(), 1);
        let s2 = MicroRecurrentKernelSnapshot::from_kernel(&k, blob, 999);
        assert_eq!(s1.blake3, s2.blake3, "version must not affect blake3");
        assert_ne!(s1.version, s2.version);
    }

    #[test]
    fn serde_roundtrip_preserves_all_fields() {
        let k = AttractorKernel::from_seed(42, 16);
        let snap = MicroRecurrentKernelSnapshot::from_kernel(&k, k.to_snapshot_blob(), 7);
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: MicroRecurrentKernelSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.family, snap.family);
        assert_eq!(back.dim, snap.dim);
        assert_eq!(back.weights_blob, snap.weights_blob);
        assert_eq!(back.blake3, snap.blake3);
        assert_eq!(back.version, snap.version);
        assert!(back.verify(), "deserialised snapshot must still verify");
    }

    #[test]
    fn from_parts_does_not_recommit() {
        // from_parts is the "I already know the hash" constructor.
        let k = AttractorKernel::from_seed(42, 32);
        let blob = k.to_snapshot_blob();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[RecurrenceFamily::Attractor as u8]);
        hasher.update(&(32u64).to_le_bytes());
        hasher.update(&blob);
        let expected_hash = *hasher.finalize().as_bytes();

        let snap = MicroRecurrentKernelSnapshot::from_parts(
            RecurrenceFamily::Attractor,
            32,
            blob,
            expected_hash,
            1,
        );
        assert!(snap.verify());
        assert_eq!(snap.blake3, expected_hash);
    }
}
