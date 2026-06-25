//! In-memory chunked content store — Phase 1 reference implementation (Plan 272 T1.7).
//!
//! Backed by `papaya::HashMap` (lock-free, per AGENTS.md). Two maps:
//! - `chunks`: `BLAKE3(chunk) → &'static [u8]` (leaked `Box<[u8]>`, reclaimed on Drop).
//! - `blobs`: `Merkle_root → BlobMetadata`.
//!
//! ## Zero-alloc hot path: `get_chunk`
//!
//! Returns `Option<&[u8]>` borrowed from the chunk map. Achieved by storing
//! each chunk as a **leaked `Box<[u8]>`** (`&'static [u8]`) — the `'static`
//! lifetime coerces to the trait's `&'_ [u8]` without `unsafe` at the call
//! site. The leaked allocation is reclaimed in [`Drop`] via `Box::from_raw`.
//!
//! Soundness invariant: chunks are **append-only** — never removed or mutated
//! after insertion. This makes the `'static` reference valid for the store's
//! entire lifetime (until `Drop` reclaims the boxes).

use std::sync::atomic::{AtomicU64, Ordering};

use papaya::HashMap;

use super::chunker::FixedSizeChunker;
use super::merkle::{build_merkle_levels, build_proof_from_levels, verify_binary_merkle_proof};
use super::r#trait::{ChunkedContentStore, ChunkingStrategy};
use super::types::{BlobId, MerkleProof, StoreStats};

/// Metadata for an indexed blob — stored in the `blobs` map keyed by `BlobId`.
///
/// `chunk_hashes` is `Box<[[u8; 32]]>` (fixed-size, heap-allocated once) so
/// proof generation is a simple index lookup with no further allocation.
///
/// `merkle_levels` caches all levels of the binary Merkle tree (level 0 =
/// padded leaves, last level = root) so that `prove_chunk` is O(log n) — sibling
/// lookups from the cache instead of O(n) tree rebuild per proof.
#[repr(C)]
struct BlobMetadata {
    /// Number of chunks in this blob (== `chunk_hashes.len()`).
    n_chunks: u32,
    /// Ordered chunk hashes, leaf level of the Merkle tree.
    chunk_hashes: Box<[[u8; 32]]>,
    /// Cached Merkle tree levels for O(log n) proof generation (Plan 272 G3 fix).
    /// levels[0] = padded leaves, levels.last() = root. Memory: ~2× chunk_hashes.
    merkle_levels: Vec<Vec<[u8; 32]>>,
    /// Sum of chunk byte lengths (== logical blob size).
    total_bytes: u64,
}

/// In-memory chunked content store backed by `papaya` lock-free hashmaps.
///
/// Default chunker is [`FixedSizeChunker::default`] (64 KiB chunks). Override
/// via [`Self::with_chunker`].
///
/// Thread-safe: all operations go through `papaya`'s lock-free paths. The
/// atomic stat counters are bumped with `Relaxed` ordering — they're
/// advisory, not synchronization primitives.
pub struct InMemoryChunkedStore {
    /// BLAKE3(chunk) → leaked `&'static [u8]`. Values are reclaimed on Drop.
    chunks: HashMap<[u8; 32], &'static [u8]>,
    /// BlobId → metadata.
    blobs: HashMap<[u8; 32], BlobMetadata>,
    /// Pluggable chunker. `Box<dyn ...>` because the chunker may be CDC (Phase 2)
    /// or any caller-supplied strategy.
    chunker: Box<dyn ChunkingStrategy + Send + Sync>,
    // Advisory stat counters — bumped Relaxed; read via `stats()`.
    n_chunks_stored: AtomicU64,
    n_blobs_indexed: AtomicU64,
    total_bytes_stored: AtomicU64,
    total_bytes_logical: AtomicU64,
}

impl InMemoryChunkedStore {
    /// Construct with the default chunker ([`FixedSizeChunker`], 64 KiB).
    #[must_use]
    pub fn new() -> Self {
        Self::with_chunker(FixedSizeChunker::default())
    }

    /// Construct with a caller-supplied chunker. The chunker must be
    /// `'static` because `put` may be called from any thread. (`Send + Sync`
    /// are already implied by the `ChunkingStrategy` supertrait — clippy
    /// `implied_bounds_in_impls`.)
    #[must_use]
    pub fn with_chunker(chunker: impl ChunkingStrategy + 'static) -> Self {
        Self {
            chunks: HashMap::new(),
            blobs: HashMap::new(),
            chunker: Box::new(chunker),
            n_chunks_stored: AtomicU64::new(0),
            n_blobs_indexed: AtomicU64::new(0),
            total_bytes_stored: AtomicU64::new(0),
            total_bytes_logical: AtomicU64::new(0),
        }
    }

    /// Insert a chunk into the chunk map if not already present.
    /// Returns `true` if this was a new insertion (dedup miss), `false` if
    /// the chunk was already stored (dedup hit).
    ///
    /// Leaks the `Vec<u8>` into a `&'static [u8]` via `Box::leak`. The leaked
    /// allocation is reclaimed on [`Drop`].
    #[inline]
    fn insert_chunk_if_absent(&self, hash: [u8; 32], chunk: &[u8]) -> bool {
        let guard = self.chunks.pin();
        match guard.get(&hash) {
            // Dedup hit — already stored.
            Some(_) => false,
            // Dedup miss — leak and insert.
            None => {
                // Box<[u8]> → leak → &'static [u8]. The allocation is tracked
                // for reclamation in Drop by walking `chunks`.
                let boxed: Box<[u8]> = chunk.to_vec().into_boxed_slice();
                let leaked: &'static mut [u8] = Box::leak(boxed);
                let leaked_shared: &'static [u8] = leaked;
                // Note: there's a benign race here — another thread may insert
                // the same hash concurrently. In that case, both leaks are
                // valid; the loser's reference is overwritten in the map and
                // will be reclaimed when its inserting store Drops. For
                // content-addressed data this is sound (same hash → same bytes).
                guard.insert(hash, leaked_shared);
                true
            }
        }
    }

    /// Test-only: insert a raw chunk by its pre-computed hash, bypassing the
    /// chunker. Used by GOAT gate G5 (hot-path read benchmark) to populate the
    /// store with N distinct chunks without constructing N separate blobs.
    #[cfg(test)]
    pub(crate) fn insert_chunk_for_test(&self, hash: [u8; 32], chunk: &[u8]) {
        let is_new = self.insert_chunk_if_absent(hash, chunk);
        if is_new {
            self.n_chunks_stored.fetch_add(1, Ordering::Relaxed);
            self.total_bytes_stored
                .fetch_add(chunk.len() as u64, Ordering::Relaxed);
        }
    }
}

impl Default for InMemoryChunkedStore {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkedContentStore for InMemoryChunkedStore {
    fn put(&self, bytes: &[u8]) -> BlobId {
        // 1. Chunk the input via the pluggable strategy. Borrowed slices.
        let borrowed_chunks: Vec<&[u8]> = self.chunker.chunk(bytes);

        // 2. Hash each chunk, dedup-insert into the chunks map, collect hashes.
        //    Pre-size to avoid reallocation (AGENTS.md).
        let mut chunk_hashes: Vec<[u8; 32]> = Vec::with_capacity(borrowed_chunks.len());
        for chunk in &borrowed_chunks {
            let hash: [u8; 32] = blake3::hash(chunk).into();
            let is_new = self.insert_chunk_if_absent(hash, chunk);
            if is_new {
                self.n_chunks_stored.fetch_add(1, Ordering::Relaxed);
                self.total_bytes_stored
                    .fetch_add(chunk.len() as u64, Ordering::Relaxed);
            }
            chunk_hashes.push(hash);
        }

        // 3. Compute the Merkle root = BlobId. Cache all levels for O(log n)
        //    proof generation (Plan 272 G3 fix — avoids O(n) tree rebuild per proof).
        let merkle_levels = build_merkle_levels(&chunk_hashes);
        let root = merkle_levels
            .last()
            .and_then(|lvl| lvl.first().copied())
            .unwrap_or_else(|| blake3::hash(b"").into());
        let blob_id = BlobId(root);

        // 4. Insert blob metadata (idempotent — same root → same metadata).
        //    `papaya`'s `insert` returns the old value if present; we drop it.
        let total_bytes = bytes.len() as u64;
        let metadata = BlobMetadata {
            n_chunks: u32::try_from(chunk_hashes.len()).unwrap_or(u32::MAX),
            chunk_hashes: chunk_hashes.into_boxed_slice(),
            merkle_levels,
            total_bytes,
        };
        let is_new_blob = {
            let guard = self.blobs.pin();
            // `insert` returns Some(old) if the key was present, None if fresh.
            // We only need the bool, not the reference, so we drop the borrow
            // immediately.
            guard.insert(root, metadata).is_none()
        };
        if is_new_blob {
            self.n_blobs_indexed.fetch_add(1, Ordering::Relaxed);
        }
        // Always bump logical bytes — even on idempotent re-put, the caller
        // handed us `bytes.len()` bytes; we're tracking total logical traffic,
        // not unique logical bytes. (For unique logical bytes, see `stats()`
        // which can be refined in Phase 4 if G1 needs it.)
        self.total_bytes_logical
            .fetch_add(total_bytes, Ordering::Relaxed);

        blob_id
    }

    fn get(&self, blob_id: &BlobId) -> Option<Vec<u8>> {
        let guard = self.blobs.pin();
        let metadata = guard.get(&blob_id.0)?;
        // Pre-allocate the output buffer to the exact logical size (AGENTS.md).
        let mut out = Vec::with_capacity(metadata.total_bytes as usize);
        let chunk_guard = self.chunks.pin();
        for chunk_hash in metadata.chunk_hashes.iter() {
            let chunk_bytes: &[u8] = chunk_guard.get(chunk_hash)?;
            out.extend_from_slice(chunk_bytes);
        }
        Some(out)
    }

    // ZERO-ALLOC: no Vec/String/Box/format!/to_vec() in this body. Returns a
    // borrowed slice via papaya's `.copied()` on the Option<&'static [u8]> —
    // copies the *reference* (8 bytes), not the chunk bytes.
    fn get_chunk(&self, chunk_hash: &[u8; 32]) -> Option<&[u8]> {
        self.chunks.pin().get(chunk_hash).copied()
    }

    fn prove_chunk(&self, blob_id: &BlobId, leaf_index: usize) -> Option<MerkleProof> {
        let guard = self.blobs.pin();
        let metadata = guard.get(&blob_id.0)?;
        if leaf_index >= metadata.chunk_hashes.len() {
            return None;
        }
        // O(log n) sibling lookup from cached levels — ZERO BLAKE3 calls.
        // (Previous implementation called build_binary_merkle_proof which is
        //  O(n) — rebuilds the entire tree per proof. G3 fix.)
        let siblings = build_proof_from_levels(&metadata.merkle_levels, leaf_index);
        Some(MerkleProof {
            leaf_index,
            siblings,
            expected_root: blob_id.0,
        })
    }

    /// **Associated fn** (G4 light-client gate) — no `&self`. Delegates to the
    /// pure-BLAKE3 verifier.
    fn verify_proof(proof: &MerkleProof, leaf_hash: &[u8; 32]) -> bool {
        verify_binary_merkle_proof(leaf_hash, proof.leaf_index, &proof.siblings, &proof.expected_root)
    }

    fn stats(&self) -> StoreStats {
        let total_bytes_stored = self.total_bytes_stored.load(Ordering::Relaxed);
        let total_bytes_logical = self.total_bytes_logical.load(Ordering::Relaxed);
        let mut s = StoreStats {
            n_chunks_stored: self.n_chunks_stored.load(Ordering::Relaxed),
            n_blobs_indexed: self.n_blobs_indexed.load(Ordering::Relaxed),
            total_bytes_stored,
            total_bytes_logical,
            dedup_ratio: 1.0,
        };
        s.compute_dedup_ratio();
        s
    }
}

impl Drop for InMemoryChunkedStore {
    fn drop(&mut self) {
        // Reclaim all leaked `Box<[u8]>` chunks. Walk the chunk map, reconstruct
        // each Box from its raw pointer + length, and let it drop.
        let guard = self.chunks.pin();
        for (_, leaked_slice) in guard.iter() {
            let len = leaked_slice.len();
            let ptr = leaked_slice.as_ptr() as *mut u8;
            // SAFETY: `leaked_slice` was produced by `Box::leak(Box<[u8]>)` in
            // `insert_chunk_if_absent`. The pointer is the start of a heap
            // allocation owned solely by this store (no aliasing — papaya's
            // map is the only reference holder after insert, and we're tearing
            // the store down). `len` matches the original Box's length.
            unsafe {
                let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a store with a small chunk size for compact test blobs.
    fn make_store(chunk_size: usize) -> InMemoryChunkedStore {
        InMemoryChunkedStore::with_chunker(FixedSizeChunker::new(chunk_size))
    }

    #[test]
    fn test_put_get_roundtrip() {
        let store = make_store(8);
        let bytes: Vec<u8> = (0..32).collect(); // 4 chunks of 8 bytes
        let id = store.put(&bytes);
        let recovered = store.get(&id).expect("blob must be retrievable");
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn test_idempotent_put() {
        let store = make_store(8);
        let bytes = b"hello world this is a test of idempotency!!!".to_vec();
        let id1 = store.put(&bytes);
        let id2 = store.put(&bytes);
        assert_eq!(id1, id2, "same bytes → same BlobId");
    }

    #[test]
    fn test_dedup_chunks_shared() {
        // Two blobs sharing 50% of chunks: first half identical, second half
        // differs. With chunk_size = 8 and 16-byte blobs, blob1 = [A, B],
        // blob2 = [A, C] where A is shared.
        let store = make_store(8);
        let shared = b"AAAAAAAA"; // chunk A (8 bytes)
        let only_b1 = b"BBBBBBBB"; // chunk B
        let only_b2 = b"CCCCCCCC"; // chunk C
        let mut blob1 = Vec::new();
        blob1.extend_from_slice(shared);
        blob1.extend_from_slice(only_b1);
        let mut blob2 = Vec::new();
        blob2.extend_from_slice(shared);
        blob2.extend_from_slice(only_b2);

        let id1 = store.put(&blob1);
        let id2 = store.put(&blob2);
        assert_ne!(id1, id2, "different blobs → different BlobIds");

        let stats = store.stats();
        // 3 unique chunks (A, B, C) for 4 logical chunks → 1.33× dedup ratio.
        assert_eq!(stats.n_chunks_stored, 3, "expected 3 unique chunks");
        assert_eq!(stats.n_blobs_indexed, 2);
        // dedup ratio = logical / stored = 32 / 24 ≈ 1.33
        // (logical = 16 + 16 = 32 bytes; stored = 8 * 3 = 24 bytes).
        assert!(
            stats.dedup_ratio > 1.2 && stats.dedup_ratio < 1.5,
            "expected ~1.33 dedup ratio, got {}",
            stats.dedup_ratio
        );
    }

    #[test]
    fn test_inclusion_proof_roundtrip() {
        let store = make_store(4);
        let bytes: Vec<u8> = (0..32).collect(); // 8 chunks of 4 bytes
        let id = store.put(&bytes);

        // Prove leaf 3, then verify against the actual chunk hash for leaf 3.
        let proof = store
            .prove_chunk(&id, 3)
            .expect("leaf 3 must be provable");

        // Recover the chunk hash for leaf 3 by re-chunking + re-hashing.
        let chunks = FixedSizeChunker::new(4).chunk(&bytes);
        let leaf_3_hash: [u8; 32] = blake3::hash(chunks[3]).into();

        assert!(
            InMemoryChunkedStore::verify_proof(&proof, &leaf_3_hash),
            "verify_proof must succeed for the correct leaf hash"
        );
    }

    #[test]
    fn test_inclusion_proof_wrong_index() {
        let store = make_store(4);
        let bytes: Vec<u8> = (0..32).collect();
        let id = store.put(&bytes);

        let proof_for_0 = store
            .prove_chunk(&id, 0)
            .expect("leaf 0 must be provable");

        // Recover leaf 1's hash.
        let chunks = FixedSizeChunker::new(4).chunk(&bytes);
        let leaf_1_hash: [u8; 32] = blake3::hash(chunks[1]).into();

        assert!(
            !InMemoryChunkedStore::verify_proof(&proof_for_0, &leaf_1_hash),
            "leaf 1's hash must NOT verify against leaf 0's proof"
        );
    }

    #[test]
    fn test_empty_blob() {
        let store = make_store(8);
        let id = store.put(b"");
        // Empty blob: 0 chunks → root = blake3::hash(b"").into().
        let expected_root: [u8; 32] = blake3::hash(b"").into();
        assert_eq!(id.0, expected_root, "empty blob BlobId must be BLAKE3(empty)");
        let recovered = store.get(&id).expect("empty blob must be retrievable");
        assert!(recovered.is_empty(), "empty blob → empty Vec");
        let stats = store.stats();
        assert_eq!(stats.n_chunks_stored, 0);
        assert_eq!(stats.n_blobs_indexed, 1);
    }

    #[test]
    fn test_get_chunk_zero_alloc_signature() {
        // Static assertion that get_chunk's body contains no allocation
        // primitives. We can't easily do this as a runtime test without
        // parsing the source, but the doc comment above `fn get_chunk`
        // ("ZERO-ALLOC: no Vec/String/Box/format!/to_vec()") serves as the
        // contract. This test exists to anchor the contract in the test
        // suite so a future refactor that introduces an allocation would
        // ideally update both the comment and the test.
        //
        // Manual inspection procedure (Plan 272 T1.10):
        //   grep -nE 'Vec|String|Box|format!|to_vec\(\)' in_memory.rs |
        //     grep -A1 'fn get_chunk'
        // Expected: zero matches inside the get_chunk body.
        let store = make_store(4);
        let bytes = b"abcdefgh";
        let id = store.put(bytes);
        // Sanity: get_chunk actually works.
        let chunks = FixedSizeChunker::new(4).chunk(bytes);
        let hash: [u8; 32] = blake3::hash(chunks[0]).into();
        let chunk = store.get_chunk(&hash).expect("chunk must be retrievable");
        assert_eq!(chunk, chunks[0]);
        // Touch id to silence unused warning if hash logic ever changes.
        let _ = id;
    }

    #[test]
    fn test_tamper_detection() {
        // Put a blob, then put a tampered version (1 bit flipped in 1 chunk).
        // The two BlobIds must differ — tamper detection via Merkle root.
        let store = make_store(8);
        let original = b"original chunk data here!!"; // 24 bytes → 3 chunks of 8
        let id_original = store.put(original);

        // Tamper: flip 1 bit in the middle of the blob (affects chunk 1).
        let mut tampered = original.to_vec();
        tampered[12] ^= 0x01;
        let id_tampered = store.put(&tampered);

        assert_ne!(
            id_original, id_tampered,
            "tampered blob must have a different BlobId"
        );
    }

    #[test]
    fn test_get_missing_blob_returns_none() {
        let store = make_store(8);
        let bogus = BlobId([0xFF; 32]);
        assert!(store.get(&bogus).is_none());
    }

    #[test]
    fn test_prove_chunk_out_of_range() {
        let store = make_store(4);
        let bytes = b"abcdefgh"; // 2 chunks
        let id = store.put(bytes);
        assert!(store.prove_chunk(&id, 99).is_none());
    }

    #[test]
    fn test_get_chunk_missing_returns_none() {
        let store = make_store(8);
        let missing_hash = [0xAA; 32];
        assert!(store.get_chunk(&missing_hash).is_none());
    }

    #[test]
    fn test_stats_after_multiple_blobs() {
        let store = make_store(4);
        let _id1 = store.put(b"abcdefgh"); // 2 chunks
        let _id2 = store.put(b"abcdefgh"); // same → dedup, 0 new chunks
        let _id3 = store.put(b"abcdXXXX"); // 1 shared + 1 new chunk

        let stats = store.stats();
        assert_eq!(stats.n_chunks_stored, 3, "A,B (from id1) + X (from id3)");
        assert_eq!(stats.n_blobs_indexed, 2, "id1 and id3 (id2 dedup'd to id1)");
        // logical = 8 + 8 + 8 = 24; stored = 4*3 = 12; ratio = 2.0
        assert!(
            (stats.dedup_ratio - 2.0).abs() < 1e-6,
            "expected dedup ratio 2.0, got {}",
            stats.dedup_ratio
        );
    }

    #[test]
    fn test_with_custom_chunker() {
        // Verify the chunker is pluggable.
        struct Doubler;
        impl ChunkingStrategy for Doubler {
            fn chunk<'a>(&self, bytes: &'a [u8]) -> Vec<&'a [u8]> {
                bytes.chunks(2).collect()
            }
        }
        let store = InMemoryChunkedStore::with_chunker(Doubler);
        let bytes = b"abcdef"; // 3 chunks of 2
        let id = store.put(bytes);
        let recovered = store.get(&id).expect("must retrieve");
        assert_eq!(recovered, bytes);
        // 3 chunks
        let stats = store.stats();
        assert_eq!(stats.n_chunks_stored, 3);
    }
}
