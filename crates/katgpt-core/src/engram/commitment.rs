//! Engram table commitment — content-addressed identity.
//!
//! Plan 299 Phase 5 T5.5–T5.6 (partial). Provides:
//! - [`EngramTableId`] — a 32-byte content-addressed identity for any
//!   [`EngramTable`](super::EngramTable). Two tables with the same slot
//!   contents produce the same ID. This is the artifact that crosses the
//!   sync boundary (latent slots stay latent; the commitment is raw).
//! - [`build_merkle_root`] — binary Merkle tree of per-slot BLAKE3 hashes.
//!   Leaves = `BLAKE3(slot_bytes)`; internal = `BLAKE3(left || right)`;
//!   root = table identity.
//!
//! # Streaming BLAKE3 pattern
//!
//! Matches `micro_belief/snapshot.rs` and `types.rs::GpartAdapter::commitment`:
//! ```text
//! let mut hasher = blake3::Hasher::new();
//! hasher.update(&bytes_a);
//! hasher.update(&bytes_b);
//! let hash = *hasher.finalize().as_bytes();
//! ```
//!
//! # Hot-path contract
//!
//! [`build_merkle_root`] is **not** on the inference hot path — it's called
//! once at build time and cached by [`InMemoryEngramTable::commitment`](super::InMemoryEngramTable).
//! Allocates a `Vec<[u8; 32]>` for the working layer; acceptable here since
//! table builds are infrequent. The lookup path itself is zero-alloc.
//!
//! # TODO (Phase 5 follow-on, deferred)
//!
//! - T5.1–T5.4 `EngramHotSwap` — AtomicPtr<Box<dyn EngramTable>> + reader
//!   closure. Mirror `sense/hotswap.rs`. Deferred — file when first
//!   consumer needs runtime table replacement.
//! - T5.7–T5.8 unit tests for hot-swap atomicity + G5 concurrent reader /
//!   writer gate. Deferred with T5.1–T5.4.

use super::EngramTable;

/// 32-byte content-addressed identity for an [`EngramTable`].
///
/// Computed as the BLAKE3 Merkle root of per-slot hashes — see
/// [`build_merkle_root`]. Two tables with identical slot contents produce
/// the same [`EngramTableId`], regardless of head configuration. This is
/// the audit artifact that crosses the sync boundary (the latent slot
/// contents themselves never sync).
///
/// # Verification
///
/// [`EngramTableId::verify`] recomputes the Merkle root from a live table
/// and compares. A `false` result indicates tampering or corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngramTableId(pub [u8; 32]);

impl EngramTableId {
    /// Compute the table's identity from its current contents.
    ///
    /// Pulls `commitment()` from the table, which is the cached Merkle root
    /// (or computes + caches it on first call). The returned
    /// [`EngramTableId`] can be serialized, synced, and later verified
    /// against a re-loaded table.
    #[inline]
    pub fn from_table(table: &dyn EngramTable) -> Self {
        EngramTableId(table.commitment())
    }

    /// Recompute the table's identity and compare with `self`.
    ///
    /// Returns `true` iff the table's current contents hash to `self`. A
    /// `false` result indicates tampering or corruption — the table's slots
    /// no longer match the contents that produced this ID.
    #[inline]
    pub fn verify(&self, table: &dyn EngramTable) -> bool {
        self.0 == table.commitment()
    }
}

/// Binary Merkle root over per-slot BLAKE3 hashes.
///
/// Each row of `slots` (a slice of `d` f32s at offset `i*d`) is hashed as
/// its raw little-endian bytes to form a leaf. Leaves are paired up
/// breadth-first; each internal node is `BLAKE3(left || right)`. The final
/// root (or the lone leaf, if there's only one slot) is the table identity.
///
/// # Edge cases
///
/// - `slots.is_empty()` or `d == 0` → returns `BLAKE3()` of empty input
///   (the all-zero-slot sentinel root). Callers should treat this as the
///   "empty table" identity.
/// - One slot → root is that slot's leaf hash (no internal nodes).
/// - Non-power-of-two slot count → the final unpaired leaf is hashed
///   alone at each layer (`BLAKE3(leaf || [0u8;32])`). This is the
///   standard "padding leaf = zero hash" Merkle convention.
///
/// # Allocation
///
/// Allocates a `Vec<[u8; 32]>` for each working layer. Acceptable for
/// build-time use; the inference hot path uses the cached result from
/// [`InMemoryEngramTable::commitment`](super::InMemoryEngramTable).
///
/// # Determinism
///
/// Same `slots` contents → same root, always, regardless of head config or
/// table metadata. This is the key contract for content-addressed sync.
pub fn build_merkle_root(slots: &[f32], d: usize) -> [u8; 32] {
    // Empty / zero-dim guard: hash empty input → BLAKE3's well-known
    // empty-input digest. Two empty tables thus share the same identity.
    if d == 0 || slots.is_empty() {
        // Empty input: no `update()` calls, so no `mut` needed.
        // `finalize()` takes `&self`.
        let h = blake3::Hasher::new();
        return *h.finalize().as_bytes();
    }

    let n_slots = slots.len() / d;
    debug_assert_eq!(
        slots.len(),
        n_slots * d,
        "build_merkle_root: slots.len() must be a multiple of d"
    );

    // Leaf layer: hash each slot's raw bytes.
    // We cast &[f32] to &[u8] via to_le_bytes per-element to avoid host
    // endianness ambiguity in the commitment. f32 → [u8; 4] little-endian.
    let mut layer: Vec<[u8; 32]> = Vec::with_capacity(n_slots.max(1));
    for i in 0..n_slots {
        let row = &slots[i * d..(i + 1) * d];
        let mut hasher = blake3::Hasher::new();
        for &f in row {
            hasher.update(&f.to_le_bytes());
        }
        layer.push(*hasher.finalize().as_bytes());
    }

    // Reduce: pair up, hash (left || right), until one root remains.
    // Padding convention: an unpaired leaf at any layer is hashed as
    // BLAKE3(leaf || [0u8; 32]).
    let zero: [u8; 32] = [0u8; 32];
    while layer.len() > 1 {
        let mut next: Vec<[u8; 32]> = Vec::with_capacity((layer.len() + 1) / 2);
        let mut i = 0;
        while i < layer.len() {
            let left = layer[i];
            let right = if i + 1 < layer.len() {
                layer[i + 1]
            } else {
                zero
            };
            let mut hasher = blake3::Hasher::new();
            hasher.update(&left);
            hasher.update(&right);
            next.push(*hasher.finalize().as_bytes());
            i += 2;
        }
        layer = next;
    }

    // layer now has exactly one element — the root.
    layer[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{EngramHash, EngramTableBuilder, K_MAX};

    #[test]
    fn empty_slots_zero_root() {
        // Empty slots → BLAKE3 of empty input (the well-known digest).
        let root = build_merkle_root(&[], 4);
        // Compute the expected empty-input BLAKE3 directly.
        let h = blake3::Hasher::new();
        let expected = *h.finalize().as_bytes();
        assert_eq!(root, expected);
    }

    #[test]
    fn same_slots_same_root() {
        // Determinism: same slots contents → same Merkle root.
        let slots_a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let slots_b = slots_a.clone();
        let ra = build_merkle_root(&slots_a, 4);
        let rb = build_merkle_root(&slots_b, 4);
        assert_eq!(ra, rb, "identical slot contents → identical roots");
    }

    #[test]
    fn one_slot_changed_different_root() {
        // Sensitivity: changing one slot changes the root.
        let mut slots_a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let slots_b = slots_a.clone();
        let ra = build_merkle_root(&slots_a, 4);
        // Flip a byte in slot 0 (element 0).
        slots_a[0] = 999.0;
        let ra_mut = build_merkle_root(&slots_a, 4);
        let rb = build_merkle_root(&slots_b, 4);
        assert_ne!(ra_mut, ra, "mutated slots must change root");
        assert_ne!(ra_mut, rb, "mutated slots differ from original");
        assert_eq!(ra, rb, "originals must still match");
    }

    #[test]
    fn single_slot_root_is_leaf_hash() {
        // Edge case: one slot → root = that slot's leaf hash (no internal nodes).
        let slots: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let root = build_merkle_root(&slots, 4);

        // Compute the expected leaf hash directly.
        let mut h = blake3::Hasher::new();
        for f in &slots {
            h.update(&f.to_le_bytes());
        }
        let expected = *h.finalize().as_bytes();
        assert_eq!(root, expected);
    }

    #[test]
    fn engram_table_id_roundtrip() {
        // End-to-end: EngramTableId::from_table + verify.
        let mut b = EngramTableBuilder::new(64, 4);
        for i in 0..8u64 {
            let pat = [i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
            b.add_pattern(EngramHash(i), &pat);
        }
        let table = b.build();
        let id = EngramTableId::from_table(&table);
        assert!(id.verify(&table), "freshly-built table must verify");
    }

    #[test]
    fn engram_table_id_distinguishes_tables() {
        // Two tables with different contents → different IDs.
        let mut b1 = EngramTableBuilder::new(16, 4);
        let mut b2 = EngramTableBuilder::new(16, 4);
        b1.add_pattern(EngramHash(0), &[1.0, 2.0, 3.0, 4.0]);
        b2.add_pattern(EngramHash(0), &[9.0, 9.0, 9.0, 9.0]);
        let t1 = b1.build();
        let t2 = b2.build();
        let id1 = EngramTableId::from_table(&t1);
        let id2 = EngramTableId::from_table(&t2);
        assert_ne!(id1, id2, "different contents → different IDs");
        assert!(id1.verify(&t1));
        assert!(id2.verify(&t2));
        assert!(!id1.verify(&t2), "id1 must not verify t2");
        assert!(!id2.verify(&t1), "id2 must not verify t1");
    }

    #[test]
    fn merkle_root_handles_non_power_of_two() {
        // 3 slots (non-power-of-two) → must still produce a deterministic
        // root using the padding-leaf convention.
        let slots: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let root_a = build_merkle_root(&slots, 2); // 3 slots of dim 2
        let root_b = build_merkle_root(&slots, 2);
        assert_eq!(
            root_a, root_b,
            "non-power-of-two must still be deterministic"
        );
    }

    #[test]
    fn k_max_imported_for_doc_continuity() {
        // Tiny sanity check that K_MAX is visible from this module — keeps
        // the `use` alive as documentation that this module is part of the
        // K_MAX-headed retrieval family.
        assert_eq!(K_MAX, 16);
    }
}
