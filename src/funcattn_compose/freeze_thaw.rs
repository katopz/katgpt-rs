//! T5.3 — Freeze/thaw for FUNCATTN basis + QKV weight snapshots.
//!
//! Plan 286 T5.3: version basis snapshots `W_Φ, W_Ψ` (here: the full FUNCATTN
//! personality — basis `w_basis` + `w_q`/`w_k`/`w_v`) as atomic, BLAKE3-
//! committed, versioned artifacts swappable at runtime. This is the freeze/thaw
//! bridge to riir-ai Plan 318 (per-domain basis hot-swap).
//!
//! # Design
//!
//! Mirrors the established `micro_belief::snapshot::MicroRecurrentKernelSnapshot`
//! contract (katgpt-core):
//! - [`FuncAttnWeightsSnapshot`] holds the weight matrices + a BLAKE3
//!   commitment + a monotonic `version`.
//! - [`FuncAttnSnapshotStore`] wraps `RwLock<Arc<...>>` for atomic hot-swap.
//!   Readers clone the `Arc` (cheap, lock-held briefly); writers swap under a
//!   write lock. This matches the `future_probe` pattern (`RwLock<Arc<...>>`).
//!
//! # Sync boundary
//!
//! The **weights are latent and never synced** — syncing them would destroy
//! per-domain basis divergence and waste bandwidth. Only the **commitment**
//! (BLAKE3 + version) is emitted as an audit event by a runtime consumer, so
//! observers can verify an entity is running a committed personality without
//! learning the weights (same contract as `micro_belief::snapshot`).
//!
//! # `version`
//!
//! A monotonic per-entity counter, incremented on each swap — *not* a UUID.
//! Per AGENTS.md we use `Uuid::now_v7()` for event IDs, but `version` here is
//! the *contents ordinal* of a personality (matching the `micro_belief`
//! precedent: "version is the contents of a personality version, not the ID of
//! an event"). A swap *event* may be tagged with a v7 UUID by the caller.
//!
//! # Commitment scheme
//!
//! BLAKE3 over the streaming input
//! `fmt_byte || d_le || k_le || basis_byte || alpha_le || temp_le || w_basis_le || w_q_le || w_k_le || w_v_le`.
//! Layout-independent (only logical fields contribute); the stored `blake3` is
//! zeroed internally before hashing so the commitment never feeds back into
//! itself. `version` is deliberately excluded — two snapshots with identical
//! weights but different versions are the *same* personality at different
//! points in time.

use std::sync::{Arc, RwLock};

use katgpt_core::funcattn::FuncAttnBasis;

/// Snapshot format version. Bump if the hashed field set / encoding changes.
pub const FUNCATTN_SNAPSHOT_FMT: u8 = 1;

/// A versioned, BLAKE3-committed snapshot of a FUNCATTN head's weights.
///
/// Construct via [`FuncAttnWeightsSnapshot::from_weights`] (computes the
/// commitment) or [`FuncAttnWeightsSnapshot::from_parts`] (commitment already
/// known, e.g. deserialised from disk). Verify integrity via [`Self::verify`].
///
/// The stored weight vectors are directly usable by
/// [`katgpt_core::funcattn::funcattn_forward`] — pass `&snap.w_basis[..]` etc.
/// No reallocation on the read path.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FuncAttnWeightsSnapshot {
    /// Head/feature dim `d`.
    pub d: usize,
    /// Basis dim `k`.
    pub k: usize,
    /// Basis activation scheme (so a thawed snapshot is self-describing).
    pub basis: FuncAttnBasis,
    /// Convex-combo regularization α.
    pub alpha: f32,
    /// Basis temperature τ.
    pub temperature: f32,
    /// Basis projection weights `W_Φ`, `(k, d)` row-major.
    pub w_basis: Vec<f32>,
    /// `to_q` weights, `(d, d)` row-major.
    pub w_q: Vec<f32>,
    /// `to_k` weights, `(d, d)` row-major.
    pub w_k: Vec<f32>,
    /// `to_v` weights, `(d, d)` row-major.
    pub w_v: Vec<f32>,
    /// BLAKE3 commitment over the logical fields. Filled by [`Self::commit`];
    /// zeroed during hashing.
    pub blake3: [u8; 32],
    /// Monotonic version counter (caller-managed, NOT part of the hash input).
    pub version: u64,
}

impl FuncAttnWeightsSnapshot {
    /// Build a snapshot from raw weights + config, computing the BLAKE3 commitment.
    ///
    /// `version` is caller-managed (typically incremented by the store on each
    /// swap). The weights are moved into the snapshot (no copy).
    pub fn from_weights(
        d: usize,
        k: usize,
        basis: FuncAttnBasis,
        alpha: f32,
        temperature: f32,
        w_basis: Vec<f32>,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        version: u64,
    ) -> Self {
        debug_assert_eq!(w_basis.len(), k * d, "w_basis must be (k, d)");
        debug_assert_eq!(w_q.len(), d * d, "w_q must be (d, d)");
        debug_assert_eq!(w_k.len(), d * d, "w_k must be (d, d)");
        debug_assert_eq!(w_v.len(), d * d, "w_v must be (d, d)");
        let mut snap = Self {
            d, k, basis, alpha, temperature,
            w_basis, w_q, w_k, w_v,
            blake3: [0u8; 32],
            version,
        };
        snap.commit();
        snap
    }

    /// Build a snapshot from raw parts WITHOUT recomputing the commitment.
    ///
    /// For deserialisation paths where the hash is already known (e.g. loading
    /// from disk). Call [`Self::commit`] to recompute, or [`Self::verify`] to
    /// check integrity.
    pub fn from_parts(
        d: usize,
        k: usize,
        basis: FuncAttnBasis,
        alpha: f32,
        temperature: f32,
        w_basis: Vec<f32>,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        blake3: [u8; 32],
        version: u64,
    ) -> Self {
        Self { d, k, basis, alpha, temperature, w_basis, w_q, w_k, w_v, blake3, version }
    }

    /// Streaming hash helper. Pushes every logical field (except `blake3` and
    /// `version`) into the hasher in a fixed, layout-independent order.
    fn hash_into(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&[FUNCATTN_SNAPSHOT_FMT]);
        hasher.update(&(self.d as u64).to_le_bytes());
        hasher.update(&(self.k as u64).to_le_bytes());
        hasher.update(&[self.basis as u8]);
        hasher.update(&self.alpha.to_le_bytes());
        hasher.update(&self.temperature.to_le_bytes());
        hash_f32_slice(hasher, &self.w_basis);
        hash_f32_slice(hasher, &self.w_q);
        hash_f32_slice(hasher, &self.w_k);
        hash_f32_slice(hasher, &self.w_v);
    }

    /// Compute (or recompute) the BLAKE3 commitment. Idempotent.
    pub fn commit(&mut self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        // Hash the logical fields with blake3 zeroed (so it doesn't feed back).
        let saved = self.blake3;
        self.blake3 = [0u8; 32];
        self.hash_into(&mut hasher);
        self.blake3 = saved;
        let hash = *hasher.finalize().as_bytes();
        self.blake3 = hash;
        hash
    }

    /// Recompute the commitment and compare with the stored `self.blake3`.
    ///
    /// Returns `true` iff the stored weights produce the stored hash. `false`
    /// indicates tampering or corruption.
    pub fn verify(&self) -> bool {
        let mut hasher = blake3::Hasher::new();
        // verify must ignore self.blake3 (treat as zero) regardless of stored value.
        let mut probe = self.clone();
        probe.blake3 = [0u8; 32];
        probe.hash_into(&mut hasher);
        let recomputed = *hasher.finalize().as_bytes();
        recomputed == self.blake3
    }
}

/// Hash a `&[f32]` slice by streaming each element's little-endian bytes.
///
/// Length-prefixing (u64 LE) prevents collision between e.g. `[1.0]` and
/// `[1.0, 0.0]`-prefix-of-longer.
#[inline]
fn hash_f32_slice(hasher: &mut blake3::Hasher, xs: &[f32]) {
    hasher.update(&(xs.len() as u64).to_le_bytes());
    for x in xs {
        hasher.update(&x.to_le_bytes());
    }
}

/// Atomic hot-swap store for FUNCATTN weight snapshots.
///
/// Wraps `RwLock<Arc<FuncAttnWeightsSnapshot>>`. Readers obtain a cheap `Arc`
/// clone under a brief read lock; writers install a new snapshot under a write
/// lock and receive the displaced snapshot. Readers holding an `Arc` keep the
/// old snapshot alive (no torn reads) — this is the freeze/thaw invariant
/// (AGENTS.md: "the only weight mutation allowed at runtime is swapping a
/// frozen snapshot (atomic, versioned, BLAKE3-checked)").
///
/// # Poison
///
/// Uses `std::sync::RwLock`. A panicking writer poisons the lock; [`Self::swap`]
/// and [`Self::current`] propagate the poison via `Result`. Callers that cannot
/// tolerate poison should run the swap in a `catch_unwind`.
pub struct FuncAttnSnapshotStore {
    inner: RwLock<Arc<FuncAttnWeightsSnapshot>>,
}

impl FuncAttnSnapshotStore {
    /// Create a store seeded with `initial` (typically version 0).
    pub fn new(initial: FuncAttnWeightsSnapshot) -> Self {
        Self { inner: RwLock::new(Arc::new(initial)) }
    }

    /// Get a cheap `Arc` handle to the current snapshot.
    ///
    /// The lock is held only for the duration of the `Arc` clone. The returned
    /// `Arc` keeps the snapshot valid for as long as the caller holds it, even
    /// across a subsequent [`Self::swap`].
    pub fn current(&self) -> Result<Arc<FuncAttnWeightsSnapshot>, FuncAttnSnapshotStoreError> {
        let guard = self.inner.read().map_err(|_| FuncAttnSnapshotStoreError::Poisoned)?;
        Ok(Arc::clone(&guard))
    }

    /// Atomically install `next` as the current snapshot.
    ///
    /// Returns the displaced snapshot (the previous `current`). The new
    /// snapshot's `version` should be `displaced.version + 1` (caller-managed —
    /// the store does not auto-increment, so callers can assign their own
    /// versioning scheme, e.g. monotonic-per-domain).
    pub fn swap(
        &self,
        next: FuncAttnWeightsSnapshot,
    ) -> Result<Arc<FuncAttnWeightsSnapshot>, FuncAttnSnapshotStoreError> {
        // Verify before installing — a corrupted snapshot must never become live.
        if !next.verify() {
            return Err(FuncAttnSnapshotStoreError::CommitmentMismatch);
        }
        let mut guard = self.inner.write().map_err(|_| FuncAttnSnapshotStoreError::Poisoned)?;
        let new_arc = Arc::new(next);
        let old = std::mem::replace(&mut *guard, new_arc);
        Ok(old)
    }

    /// Current version counter (convenience for deciding the next version).
    pub fn current_version(&self) -> Result<u64, FuncAttnSnapshotStoreError> {
        Ok(self.current()?.version)
    }
}

/// Errors from [`FuncAttnSnapshotStore`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuncAttnSnapshotStoreError {
    /// The inner `RwLock` is poisoned (a writer panicked).
    Poisoned,
    /// A snapshot offered to [`FuncAttnSnapshotStore::swap`] failed `verify()`.
    CommitmentMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_weights(d: usize, k: usize, seed: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let w_basis: Vec<f32> = (0..k * d).map(|i| seed + i as f32 * 0.001).collect();
        let w_q: Vec<f32> = (0..d * d).map(|i| seed + 0.1 + i as f32 * 0.001).collect();
        let w_k: Vec<f32> = (0..d * d).map(|i| seed + 0.2 + i as f32 * 0.001).collect();
        let w_v: Vec<f32> = (0..d * d).map(|i| seed + 0.3 + i as f32 * 0.001).collect();
        (w_basis, w_q, w_k, w_v)
    }

    #[test]
    fn from_weights_commits_and_verifies() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 1.0);
        let snap = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb, wq, wk, wv, 1,
        );
        assert!(snap.verify(), "freshly-committed snapshot must verify");
        assert_ne!(snap.blake3, [0u8; 32], "blake3 must be non-zero");
        assert_eq!(snap.version, 1);
    }

    #[test]
    fn commit_is_idempotent() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 2.0);
        let mut snap = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb, wq, wk, wv, 1,
        );
        let h1 = snap.blake3;
        let h2 = snap.commit();
        assert_eq!(h1, h2, "commit must be idempotent");
    }

    #[test]
    fn tampered_weights_fail_verify() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 3.0);
        let mut snap = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb, wq, wk, wv, 1,
        );
        assert!(snap.verify());
        snap.w_basis[0] += 1.0; // tamper
        assert!(!snap.verify(), "tampered weights must fail verify");
    }

    #[test]
    fn different_weights_different_commitment() {
        let d = 8;
        let k = 4;
        let (wb1, wq1, wk1, wv1) = make_weights(d, k, 1.0);
        let (wb2, wq2, wk2, wv2) = make_weights(d, k, 2.0);
        let s1 = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb1, wq1, wk1, wv1, 1,
        );
        let s2 = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb2, wq2, wk2, wv2, 1,
        );
        assert_ne!(s1.blake3, s2.blake3, "different weights → different hash");
    }

    #[test]
    fn version_does_not_affect_commitment() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 5.0);
        let s1 = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb.clone(), wq.clone(), wk.clone(), wv.clone(), 1,
        );
        let s2 = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb, wq, wk, wv, 999,
        );
        assert_eq!(s1.blake3, s2.blake3, "version must not affect blake3");
    }

    #[test]
    fn different_basis_different_commitment() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 5.0);
        let s_sig = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb.clone(), wq.clone(), wk.clone(), wv.clone(), 1,
        );
        let s_sft = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Softmax, 0.5, 0.1, wb, wq, wk, wv, 1,
        );
        assert_ne!(s_sig.blake3, s_sft.blake3, "different basis → different hash");
    }

    #[test]
    fn store_swap_is_atomic_and_readers_keep_old() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 1.0);
        let snap0 = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb, wq, wk, wv, 0,
        );
        let store = FuncAttnSnapshotStore::new(snap0);

        // Reader grabs v0 before the swap.
        let r0 = store.current().unwrap();
        assert_eq!(r0.version, 0);

        // Swap in v1.
        let (wb1, wq1, wk1, wv1) = make_weights(d, k, 2.0);
        let snap1 = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb1, wq1, wk1, wv1, 1,
        );
        let displaced = store.swap(snap1).unwrap();
        assert_eq!(displaced.version, 0, "displaced = previous current");

        // New readers see v1.
        let r1 = store.current().unwrap();
        assert_eq!(r1.version, 1);

        // Old reader still holds v0 (no torn read).
        assert_eq!(r0.version, 0, "old Arc keeps old snapshot alive");
    }

    #[test]
    fn store_rejects_tampered_snapshot() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 1.0);
        let snap0 = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb, wq, wk, wv, 0,
        );
        let store = FuncAttnSnapshotStore::new(snap0);

        // Build a snapshot then corrupt its weights without recomitting.
        let (wb1, wq1, wk1, wv1) = make_weights(d, k, 2.0);
        let mut bad = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb1, wq1, wk1, wv1, 1,
        );
        bad.w_basis[0] += 100.0; // tamper after commit
        let err = store.swap(bad).unwrap_err();
        assert_eq!(err, FuncAttnSnapshotStoreError::CommitmentMismatch);

        // Current is still the original v0.
        assert_eq!(store.current_version().unwrap(), 0);
    }

    #[test]
    fn serde_roundtrip_preserves_commitment() {
        let d = 8;
        let k = 4;
        let (wb, wq, wk, wv) = make_weights(d, k, 7.0);
        let snap = FuncAttnWeightsSnapshot::from_weights(
            d, k, FuncAttnBasis::Sigmoid, 0.5, 0.1, wb, wq, wk, wv, 3,
        );
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: FuncAttnWeightsSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.blake3, snap.blake3);
        assert_eq!(back.version, snap.version);
        assert!(back.verify(), "deserialised snapshot must still verify");
    }
}
