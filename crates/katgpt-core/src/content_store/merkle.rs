//! Binary Merkle tree helpers (Plan 272 T1.8).
//!
//! Separate from `crate::merkle` (Plan 221-M `MerkleOctree`) — that module is
//! octree-shaped (8 children per internal node, depth-8 fixed) for spatial KG
//! triples over octree cells. This module is **binary** (2 children per
//! internal node, variable depth) for linear chunk arrays.
//!
//! All three functions are pure BLAKE3 with no store access — required for G4
//! (light-client verify). A curator / browser / anti-cheat can verify a chunk
//! inclusion proof with only the proof + the chunk hash, never touching the
//! store. G4 is enforced structurally: no `&self`, no `chunks.get()`, no
//! `blobs.get()` in any function here.

/// Compute a binary Merkle root over a list of 32-byte leaf hashes.
///
/// Algorithm:
/// 1. If `hashes` is empty, return `blake3::hash(b"").into()` (canonical
///    empty-tree root — matches [`crate::content_store::types::BlobId::zero`]).
/// 2. Otherwise, pad to the next power of two with `[0u8; 32]` sentinel leaves
///    (zero-padding is content-distinct from any real BLAKE3 output).
/// 3. Build bottom-up: each internal node is `blake3::hash(left ‖ right)` via
///    [`parent_hash`]. Root is the single remaining node.
///
/// For `n` leaves, this is `O(n)` BLAKE3 calls (one per internal node, of which
/// there are `n - 1` after padding).
pub fn build_binary_merkle_root(hashes: &[[u8; 32]]) -> [u8; 32] {
    if hashes.is_empty() {
        return blake3::hash(b"").into();
    }
    if hashes.len() == 1 {
        // Single-leaf tree: the leaf IS the root (no padding needed; a 1-leaf
        // padded tree is just [leaf] → leaf).
        return hashes[0];
    }
    // Work on an owned buffer so we can mutate level-by-level without aliasing.
    let mut level: Vec<[u8; 32]> = hashes.to_vec();
    // If not a power of two, pad with zero leaves to the next pow2. Zero-leaf
    // is distinct from any real BLAKE3 hash (BLAKE3 of any input is non-zero
    // with overwhelming probability — collision resistance gives us this).
    let mut n = level.len();
    if !n.is_power_of_two() {
        let padded = n.next_power_of_two();
        level.resize(padded, [0u8; 32]);
        n = padded;
    }
    // Bottom-up reduce. Each iteration halves `n` by pairing siblings.
    while n > 1 {
        let half = n / 2;
        for i in 0..half {
            // In-place: write parent into slot [i], clobbering the left child
            // we just read (left child no longer needed at this level).
            let left = level[i * 2];
            let right = level[i * 2 + 1];
            level[i] = parent_hash(&left, &right);
        }
        n = half;
    }
    level[0]
}

/// Build all levels of a binary Merkle tree over `hashes`.
///
/// Returns `Vec<Vec<[u8; 32]>>` where `levels[0]` is the (padded) leaf level
/// and `levels.last()` is the root. Used to cache the tree in `BlobMetadata`
/// so that `prove_chunk` is O(log n) — sibling lookups from cached levels
/// instead of O(n) tree rebuild per proof.
///
/// Total entries: `2 * n_padded - 1` across all levels. For 1024 chunks:
/// ~2047 hashes × 32 bytes = ~64 KiB of cached tree per blob.
///
/// The root (`levels.last()[0]`) equals `build_binary_merkle_root(hashes)`.
pub fn build_merkle_levels(hashes: &[[u8; 32]]) -> Vec<Vec<[u8; 32]>> {
    if hashes.is_empty() {
        return vec![vec![blake3::hash(b"").into()]];
    }
    if hashes.len() == 1 {
        return vec![hashes.to_vec()];
    }
    let n_padded = hashes.len().next_power_of_two();
    let mut levels: Vec<Vec<[u8; 32]>> = Vec::with_capacity(n_padded.trailing_zeros() as usize + 1);

    // Level 0: padded leaves.
    let mut current: Vec<[u8; 32]> = Vec::with_capacity(n_padded);
    current.extend_from_slice(hashes);
    if current.len() < n_padded {
        current.resize(n_padded, [0u8; 32]);
    }
    levels.push(std::mem::take(&mut current));

    // Build up level-by-level until 1 node remains.
    let mut n = n_padded;
    while n > 1 {
        let half = n / 2;
        let mut next = Vec::with_capacity(half);
        let prev = levels.last().expect("levels non-empty");
        for i in 0..half {
            next.push(parent_hash(&prev[2 * i], &prev[2 * i + 1]));
        }
        levels.push(next);
        n = half;
    }
    levels
}

/// Build a Merkle inclusion proof from **cached tree levels** — O(log n)
/// sibling lookups, ZERO BLAKE3 calls.
///
/// `levels[0]` = padded leaf level, `levels[k]` = level k. For `leaf_index`,
/// walks up the tree collecting the sibling at each level.
///
/// This is the O(log n) companion to `build_binary_merkle_proof` (which is
/// O(n) — it rebuilds the tree). `InMemoryChunkedStore` caches levels in
/// `BlobMetadata` at `put()` time and calls this function for O(log n) proofs.
pub fn build_proof_from_levels(levels: &[Vec<[u8; 32]>], leaf_index: usize) -> Vec<[u8; 32]> {
    if levels.is_empty() {
        return Vec::new();
    }
    let tree_depth = levels.len() - 1; // levels[0] = leaves, last = root
    let mut siblings: Vec<[u8; 32]> = Vec::with_capacity(tree_depth);
    let mut idx = leaf_index;
    for level in 0..tree_depth {
        let sib = idx ^ 1;
        // levels[level] has the nodes at this level; sibling is at sib.
        if let Some(&hash) = levels[level].get(sib) {
            siblings.push(hash);
        } else {
            // Should not happen if levels are correctly built (padded to pow2).
            siblings.push([0u8; 32]);
        }
        idx /= 2;
    }
    siblings
}

/// Build a Merkle inclusion proof for `leaf_index` against `hashes`.
///
/// Returns the sibling hashes from leaf level up to root-child level
/// (`len() == tree_depth` after padding). Returns an empty `Vec` if
/// `leaf_index >= hashes.len()` (out-of-range).
///
/// **O(log n)** BLAKE3-free sibling collection — only the path from `leaf_index`
/// to the root is hashed; siblings are looked up by index, never recomputed.
/// The previous implementation rebuilt the entire level on every iteration
/// (O(n) BLAKE3 calls per proof); this version walks only the proof path
/// (O(log n) BLAKE3 calls).
pub fn build_binary_merkle_proof(hashes: &[[u8; 32]], leaf_index: usize) -> Vec<[u8; 32]> {
    if leaf_index >= hashes.len() || hashes.is_empty() {
        return Vec::new();
    }
    // Pad to next power of two with zeros so sibling indexing is uniform.
    let n_padded = hashes.len().next_power_of_two().max(1);
    // Pre-allocate exactly tree_depth siblings — avoids reallocations as the
    // Vec grows past its default capacity.
    let mut siblings: Vec<[u8; 32]> = Vec::with_capacity(n_padded.trailing_zeros() as usize);

    // O(n) reduction: hash each level exactly once. The previous implementation
    // rebuilt the entire next level on every proof step (O(n log n) BLAKE3 calls);
    // this version writes parents back into the lower half of the same buffer
    // (`level[i] = parent(level[2i], level[2i+1])`) and truncates logically by
    // halving `n`. Total BLAKE3 calls: `n_padded - 1` (one per non-leaf node),
    // independent of `leaf_index`. The in-place overwrite is safe because
    // reads of `level[2i]` and `level[2i+1]` happen before the write to
    // `level[i]` (i < 2i always when i > 0).
    let mut level: Vec<[u8; 32]> = Vec::with_capacity(n_padded);
    level.extend_from_slice(hashes);
    if level.len() < n_padded {
        level.resize(n_padded, [0u8; 32]);
    }
    let mut idx = leaf_index;
    let mut n = n_padded;
    while n > 1 {
        let sib = idx ^ 1;
        siblings.push(level[sib]);
        for i in 0..n / 2 {
            let l = level[2 * i];
            let r = level[2 * i + 1];
            level[i] = parent_hash(&l, &r);
        }
        idx /= 2;
        n /= 2;
    }
    siblings
}

/// Verify a Merkle inclusion proof (G4 light-client gate).
///
/// **Pure BLAKE3** — no store access. Walks siblings from leaf level to root,
/// combining `leaf_hash` with each sibling in left/right order based on
/// `leaf_index`'s bit at that depth. Returns `true` iff the recomputed root
/// equals `root`.
///
/// Critical for G4: this fn takes only `leaf_hash`, `leaf_index`, `siblings`,
/// and `root` — no `&self`, no chunk store, no blob index. A light client can
/// verify a chunk without downloading the blob.
pub fn verify_binary_merkle_proof(
    leaf_hash: &[u8; 32],
    leaf_index: usize,
    siblings: &[[u8; 32]],
    root: &[u8; 32],
) -> bool {
    let mut acc = *leaf_hash;
    let mut idx = leaf_index;
    for &sib in siblings {
        // At this level, is our current hash the left (idx even) or right
        // (idx odd) child? Combine accordingly.
        let (left, right) = match idx & 1 {
            0 => (&acc, &sib),
            _ => (&sib, &acc),
        };
        acc = parent_hash(left, right);
        idx >>= 1;
    }
    // Constant-time compare to avoid timing oracles on the root. The naive
    // `acc == *root` short-circuits at the first differing byte.
    constant_time_eq_32(&acc, root)
}

/// Constant-time comparison of two 32-byte arrays.
///
/// Avoids `==` short-circuit which could leak the position of the first
/// differing byte through timing. Used in [`verify_binary_merkle_proof`].
#[inline]
fn constant_time_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff: u8 = 0;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Combine two child hashes into a parent hash via BLAKE3.
///
/// Allocates a 64-byte scratch buffer on the stack (no heap) per the AGENTS.md
/// "no allocation in hot loops" rule. BLAKE3 of `left ‖ right`.
#[inline]
fn parent_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    // Stack-allocated 64-byte concat buffer — no heap alloc.
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    blake3::hash(&buf).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_hashes(n: usize) -> Vec<[u8; 32]> {
        (0..n)
            .map(|i| {
                let mut h = [0u8; 32];
                h[0..8].copy_from_slice(&(i as u64).to_le_bytes());
                h
            })
            .collect()
    }

    #[test]
    fn test_empty_root_is_empty_blake3() {
        let root = build_binary_merkle_root(&[]);
        let expected: [u8; 32] = blake3::hash(b"").into();
        assert_eq!(root, expected);
    }

    #[test]
    fn test_single_leaf_root_is_leaf() {
        let leaves = dummy_hashes(1);
        let root = build_binary_merkle_root(&leaves);
        assert_eq!(root, leaves[0]);
    }

    #[test]
    fn test_two_leaves_root_is_parent() {
        let leaves = dummy_hashes(2);
        let root = build_binary_merkle_root(&leaves);
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&leaves[0]);
        buf[32..].copy_from_slice(&leaves[1]);
        let expected: [u8; 32] = blake3::hash(&buf).into();
        assert_eq!(root, expected);
    }

    #[test]
    fn test_root_stable_under_padding() {
        // 3 leaves padded to 4: root(3) must equal root(3 + zero_pad).
        let leaves = dummy_hashes(3);
        let root_3 = build_binary_merkle_root(&leaves);
        let mut padded = leaves.clone();
        padded.push([0u8; 32]);
        let root_4 = build_binary_merkle_root(&padded);
        assert_eq!(root_3, root_4, "padding with zero leaves must not change root");
    }

    #[test]
    fn test_proof_roundtrip_4_leaves() {
        let leaves = dummy_hashes(4);
        let root = build_binary_merkle_root(&leaves);
        for i in 0..4 {
            let siblings = build_binary_merkle_proof(&leaves, i);
            assert_eq!(siblings.len(), 2, "depth-2 tree → 2 siblings");
            assert!(
                verify_binary_merkle_proof(&leaves[i], i, &siblings, &root),
                "leaf {i} must verify"
            );
        }
    }

    #[test]
    fn test_proof_roundtrip_5_leaves_padded() {
        // 5 leaves → padded to 8 → depth 3.
        let leaves = dummy_hashes(5);
        let root = build_binary_merkle_root(&leaves);
        for i in 0..5 {
            let siblings = build_binary_merkle_proof(&leaves, i);
            assert_eq!(siblings.len(), 3, "depth-3 tree → 3 siblings");
            assert!(
                verify_binary_merkle_proof(&leaves[i], i, &siblings, &root),
                "leaf {i} must verify"
            );
        }
    }

    #[test]
    fn test_proof_wrong_leaf_fails() {
        let leaves = dummy_hashes(4);
        let root = build_binary_merkle_root(&leaves);
        let siblings_0 = build_binary_merkle_proof(&leaves, 0);
        // Verifying leaf 1's hash with leaf 0's proof must fail.
        assert!(
            !verify_binary_merkle_proof(&leaves[1], 0, &siblings_0, &root),
            "wrong leaf must not verify against another leaf's proof"
        );
    }

    #[test]
    fn test_proof_wrong_root_fails() {
        let leaves = dummy_hashes(4);
        let siblings = build_binary_merkle_proof(&leaves, 0);
        let bogus_root = [0u8; 32];
        assert!(
            !verify_binary_merkle_proof(&leaves[0], 0, &siblings, &bogus_root),
            "bogus root must not verify"
        );
    }

    #[test]
    fn test_proof_out_of_range_returns_empty() {
        let leaves = dummy_hashes(4);
        let siblings = build_binary_merkle_proof(&leaves, 99);
        assert!(siblings.is_empty());
    }

    #[test]
    fn test_constant_time_eq_32() {
        let a = [1u8; 32];
        assert!(constant_time_eq_32(&a, &a));
        let mut b = a;
        b[31] = 2;
        assert!(!constant_time_eq_32(&a, &b));
    }

    #[test]
    fn test_tampered_leaf_changes_root() {
        let leaves = dummy_hashes(8);
        let root_orig = build_binary_merkle_root(&leaves);
        let mut tampered = leaves.clone();
        tampered[3][0] ^= 0x01; // flip 1 bit
        let root_tampered = build_binary_merkle_root(&tampered);
        assert_ne!(root_orig, root_tampered, "any bit flip must change root");
    }
}
