//! Merkle octree — hierarchical BLAKE3 commitment for KG latent octree nodes.
//!
//! Depth-3 octree = 73 nodes: 1 root + 8 internal + 64 leaves.
//! Each node stores a `[u8; 32]` BLAKE3 hash.
//! Feature-gated behind `merkle_octree`.
//!
//! Layout (index-based, bottom-up construction):
//! ```text
//! Index 0:         root
//! Index 1..=8:     internal nodes (8)
//! Index 9..=72:    leaf nodes (64)
//! ```
//!
//! For node at index `i`:
//! - Children start at `i * 8 + 1` (if within bounds)
//! - Parent is `(i - 1) / 8`
//!
//! Plan 221-M Phase 1: T1 (MerkleOctree), T3 (MerkleProof).

/// Depth-3 octree node count: 1 root + 8 internal + 64 leaves.
pub const MERKLE_OCTREE_NODES: usize = 73;
/// Number of leaves in a depth-3 octree.
pub const MERKLE_OCTREE_LEAVES: usize = 64;
/// Number of children per internal node (octree = 8).
pub const MERKLE_OCTREE_BRANCHING: usize = 8;
/// BLAKE3 hash size.
pub const HASH_SIZE: usize = 32;
/// Number of internal (non-leaf, non-root) nodes.
pub const MERKLE_OCTREE_INTERNAL: usize = 8;
/// Depth of the Merkle octree.
pub const MERKLE_OCTREE_DEPTH: u8 = 3;

/// Merkle octree: 73-node fixed array with per-node BLAKE3 hashes.
///
/// Build bottom-up: hash leaves from data, then internal nodes from children,
/// then root from top-level children.
#[derive(Clone, Debug)]
pub struct MerkleOctree {
    /// Per-node BLAKE3 hashes. 73 x 32 = 2336 bytes.
    pub hashes: [[u8; HASH_SIZE]; MERKLE_OCTREE_NODES],
}

impl MerkleOctree {
    /// All-zero Merkle tree (default state).
    pub const ZERO: Self = Self {
        hashes: [[0u8; HASH_SIZE]; MERKLE_OCTREE_NODES],
    };

    /// Build from 64 pre-hashed leaves. Each leaf already has its BLAKE3 hash.
    /// Internal nodes computed bottom-up from children.
    ///
    /// GOAT target: < 5µs for full build.
    pub fn build_from_leaves(leaf_hashes: &[[u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES]) -> Self {
        let mut tree = Self::ZERO;

        // 1. Copy leaf hashes (index 9..=72)
        //    Leaf i → node index = MERKLE_OCTREE_INTERNAL + 1 + i
        //    Layout: root=0, internal=1..=8, leaves=9..=72.
        for (i, leaf_hash) in leaf_hashes.iter().enumerate() {
            tree.hashes[MERKLE_OCTREE_INTERNAL + 1 + i] = *leaf_hash;
        }

        // 2. Build internal nodes (index 1..=8) from their 8 children
        for i in (1..=MERKLE_OCTREE_INTERNAL).rev() {
            let child_start = i * MERKLE_OCTREE_BRANCHING + 1;
            tree.hashes[i] = Self::hash_children(
                &tree.hashes[child_start..child_start + MERKLE_OCTREE_BRANCHING],
            );
        }

        // 3. Build root (index 0) from internal nodes (index 1..=8)
        tree.hashes[0] = Self::hash_children(&tree.hashes[1..=MERKLE_OCTREE_INTERNAL]);

        tree
    }

    /// Build from raw leaf data by hashing each leaf first.
    ///
    /// Each `&[u8]` slice is independently BLAKE3-hashed to produce a leaf hash.
    /// Slices beyond 64 are ignored.
    pub fn build_from_raw_leaves(leaf_data_blocks: &[&[u8]]) -> Self {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for (i, block) in leaf_data_blocks.iter().enumerate() {
            if i >= MERKLE_OCTREE_LEAVES {
                break;
            }
            leaf_hashes[i] = *blake3::hash(block).as_bytes();
        }
        Self::build_from_leaves(&leaf_hashes)
    }

    /// Get the Merkle root hash (top of the tree).
    #[inline(always)]
    pub fn root(&self) -> &[u8; HASH_SIZE] {
        &self.hashes[0]
    }

    /// Get a specific leaf hash by index (0..64).
    #[inline(always)]
    pub fn leaf_hash(&self, leaf_index: usize) -> Option<&[u8; HASH_SIZE]> {
        if leaf_index < MERKLE_OCTREE_LEAVES {
            Some(&self.hashes[MERKLE_OCTREE_INTERNAL + 1 + leaf_index])
        } else {
            None
        }
    }

    /// Get a specific internal node hash by index (0..7).
    #[inline(always)]
    pub fn internal_hash(&self, internal_index: usize) -> Option<&[u8; HASH_SIZE]> {
        if internal_index < MERKLE_OCTREE_INTERNAL {
            Some(&self.hashes[1 + internal_index])
        } else {
            None
        }
    }

    /// Hash 8 children into a parent node.
    #[inline(always)]
    fn hash_children(children: &[[u8; HASH_SIZE]]) -> [u8; HASH_SIZE] {
        let mut hasher = blake3::Hasher::new();
        for child in children {
            hasher.update(child);
        }
        *hasher.finalize().as_bytes()
    }
}

/// Merkle inclusion proof for a leaf in the octree.
///
/// For a depth-3 octree with layout root(0)→internal(1-8)→leaves(9-72),
/// a proof requires 2 levels of sibling hashes (leaf→internal→root).
/// Each level has 7 siblings (the other 7 children of the same parent).
///
/// Total proof size: 2 x 7 x 32 = 448 bytes + 1 byte leaf index + 32 bytes leaf hash.
/// GOAT target: generate < 1µs, verify < 1µs.
#[derive(Clone, Debug)]
pub struct MerkleProof {
    /// Leaf index being proven (0..63).
    pub leaf_index: u8,
    /// Leaf hash being proven.
    pub leaf_hash: [u8; HASH_SIZE],
    /// Sibling hashes at each level (2 levels, 7 siblings each).
    /// Level 0 = leaf's siblings (other leaves under same internal node),
    /// Level 1 = internal node's siblings (other internal nodes under root).
    pub siblings: [[u8; HASH_SIZE]; Self::SIBLING_LEVELS * Self::SIBLINGS_PER_LEVEL],
}

impl MerkleProof {
    /// Number of siblings per level (8 children - 1 self).
    const SIBLINGS_PER_LEVEL: usize = MERKLE_OCTREE_BRANCHING - 1;
    /// Number of sibling levels (leaf→internal, internal→root).
    const SIBLING_LEVELS: usize = MERKLE_OCTREE_DEPTH as usize - 1;

    /// Generate a Merkle proof for the given leaf index.
    ///
    /// Returns `None` if `leaf_index >= 64`.
    pub fn generate(tree: &MerkleOctree, leaf_index: u8) -> Option<Self> {
        if leaf_index as usize >= MERKLE_OCTREE_LEAVES {
            return None;
        }

        let leaf_hash = tree.hashes[MERKLE_OCTREE_INTERNAL + 1 + leaf_index as usize];

        // Collect siblings at each level (flat array: 2 levels x 7 siblings)
        let mut siblings = [[0u8; HASH_SIZE]; Self::SIBLING_LEVELS * Self::SIBLINGS_PER_LEVEL];
        let mut current_idx = MERKLE_OCTREE_INTERNAL + 1 + leaf_index as usize;

        for level in 0..Self::SIBLING_LEVELS {
            let parent_idx = (current_idx - 1) / MERKLE_OCTREE_BRANCHING;
            let child_start = parent_idx * MERKLE_OCTREE_BRANCHING + 1;

            let mut sib_idx = 0;
            for c in 0..MERKLE_OCTREE_BRANCHING {
                let child_idx = child_start + c;
                if child_idx != current_idx {
                    siblings[level * Self::SIBLINGS_PER_LEVEL + sib_idx] = tree.hashes[child_idx];
                    sib_idx += 1;
                }
            }

            current_idx = parent_idx;
        }

        Some(Self {
            leaf_index,
            leaf_hash,
            siblings,
        })
    }

    /// Verify this proof against an expected Merkle root.
    ///
    /// Recomputes the root from the leaf hash + siblings, then compares.
    pub fn verify(&self, expected_root: &[u8; HASH_SIZE]) -> bool {
        let mut current_hash = self.leaf_hash;
        let mut current_idx = MERKLE_OCTREE_INTERNAL + 1 + self.leaf_index as usize;

        for level in 0..Self::SIBLING_LEVELS {
            let parent_idx = (current_idx - 1) / MERKLE_OCTREE_BRANCHING;
            let child_start = parent_idx * MERKLE_OCTREE_BRANCHING + 1;

            // Position of current node among its siblings
            let position_in_parent = current_idx - child_start;

            // Reconstruct the 8 children in order, inserting current hash at correct position
            let mut hasher = blake3::Hasher::new();
            let mut sib_idx = 0;

            for c in 0..MERKLE_OCTREE_BRANCHING {
                if c == position_in_parent {
                    hasher.update(&current_hash);
                } else {
                    hasher.update(&self.siblings[level * Self::SIBLINGS_PER_LEVEL + sib_idx]);
                    sib_idx += 1;
                }
            }

            current_hash = *hasher.finalize().as_bytes();
            current_idx = parent_idx;
        }

        current_hash == *expected_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_octree_build_all_zero_leaves() {
        let leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);

        // All-zero leaves → deterministic root
        let root = tree.root();
        assert_ne!(
            root, &[0u8; HASH_SIZE],
            "root should be non-zero (hash of 8 zero children)"
        );

        // All leaves should be zero
        for i in 0..MERKLE_OCTREE_LEAVES {
            assert_eq!(tree.leaf_hash(i).unwrap(), &[0u8; HASH_SIZE]);
        }

        // Deterministic: same input → same root
        let tree2 = MerkleOctree::build_from_leaves(&leaf_hashes);
        assert_eq!(tree.root(), tree2.root());
    }

    #[test]
    fn test_merkle_octree_build_nontrivial() {
        let mut leaf_hashes_a = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        let mut leaf_hashes_b = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];

        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0] = i as u8;
            leaf_hashes_a[i] = *blake3::hash(&buf).as_bytes();

            buf[1] = 0xFF;
            leaf_hashes_b[i] = *blake3::hash(&buf).as_bytes();
        }

        let tree_a = MerkleOctree::build_from_leaves(&leaf_hashes_a);
        let tree_b = MerkleOctree::build_from_leaves(&leaf_hashes_b);

        // Different leaf data → different root
        assert_ne!(tree_a.root(), tree_b.root());

        // Each leaf should be retrievable
        for i in 0..MERKLE_OCTREE_LEAVES {
            assert_eq!(tree_a.leaf_hash(i).unwrap(), &leaf_hashes_a[i]);
            assert_eq!(tree_b.leaf_hash(i).unwrap(), &leaf_hashes_b[i]);
        }
    }

    #[test]
    fn test_merkle_proof_generate_and_verify() {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf_hashes[i] = *blake3::hash(&buf).as_bytes();
        }

        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);

        // Generate proof for leaf 0 and verify
        let proof = MerkleProof::generate(&tree, 0).expect("proof for leaf 0");
        assert_eq!(proof.leaf_index, 0);
        assert_eq!(proof.leaf_hash, leaf_hashes[0]);
        assert!(proof.verify(tree.root()), "valid proof should verify");

        // Generate proof for leaf 42 and verify
        let proof42 = MerkleProof::generate(&tree, 42).expect("proof for leaf 42");
        assert_eq!(proof42.leaf_index, 42);
        assert_eq!(proof42.leaf_hash, leaf_hashes[42]);
        assert!(
            proof42.verify(tree.root()),
            "valid proof for leaf 42 should verify"
        );
    }

    #[test]
    fn test_merkle_proof_tampered_leaf_fails() {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf_hashes[i] = *blake3::hash(&buf).as_bytes();
        }

        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);
        let mut proof = MerkleProof::generate(&tree, 7).expect("proof for leaf 7");
        assert!(proof.verify(tree.root()));

        // Tamper with leaf hash
        proof.leaf_hash[0] ^= 0xFF;
        assert!(
            !proof.verify(tree.root()),
            "tampered leaf should fail verification"
        );

        // Tamper with a sibling
        proof.leaf_hash = leaf_hashes[7]; // restore
        proof.siblings[0][0] ^= 0xFF;
        assert!(
            !proof.verify(tree.root()),
            "tampered sibling should fail verification"
        );
    }

    #[test]
    fn test_merkle_proof_all_leaves() {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf_hashes[i] = *blake3::hash(&buf).as_bytes();
        }

        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);

        // Verify proofs for all 64 leaves
        for i in 0..MERKLE_OCTREE_LEAVES {
            let proof = MerkleProof::generate(&tree, i as u8)
                .unwrap_or_else(|| panic!("proof generation failed for leaf {i}"));
            assert!(
                proof.verify(tree.root()),
                "proof for leaf {i} should verify"
            );
        }
    }

    #[test]
    fn test_merkle_proof_invalid_leaf_index() {
        let leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);

        assert!(MerkleProof::generate(&tree, 64).is_none());
        assert!(MerkleProof::generate(&tree, 255).is_none());
    }

    #[test]
    fn test_leaf_hash_out_of_bounds() {
        let leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);

        assert!(tree.leaf_hash(63).is_some());
        assert!(tree.leaf_hash(64).is_none());
        assert!(tree.leaf_hash(1000).is_none());
    }

    #[test]
    fn test_internal_hash_access() {
        let leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);

        // All 8 internal nodes should be accessible
        for i in 0..8 {
            assert!(tree.internal_hash(i).is_some());
        }
        assert!(tree.internal_hash(8).is_none());
    }

    #[test]
    fn test_build_from_raw_leaves() {
        static mut BUFS: [[u8; 8]; 64] = [[0u8; 8]; 64];
        // SAFETY: test-only, single-threaded
        unsafe {
            for i in 0..64u64 {
                BUFS[i as usize][0..8].copy_from_slice(&i.to_le_bytes());
            }
        }
        let raw_data: Vec<&[u8]> = (0..64).map(|i| unsafe { BUFS[i].as_slice() }).collect();

        let tree = MerkleOctree::build_from_raw_leaves(&raw_data);
        // Root should be non-zero since leaves have non-trivial data
        assert_ne!(tree.root(), &[0u8; HASH_SIZE]);
    }

    #[test]
    fn test_single_leaf_change_changes_root() {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0] = i as u8;
            leaf_hashes[i] = *blake3::hash(&buf).as_bytes();
        }

        let tree_a = MerkleOctree::build_from_leaves(&leaf_hashes);

        // Change a single leaf
        leaf_hashes[0][0] ^= 0xFF;
        let tree_b = MerkleOctree::build_from_leaves(&leaf_hashes);

        assert_ne!(
            tree_a.root(),
            tree_b.root(),
            "changing one leaf must change root"
        );
    }

    #[test]
    fn test_wrong_root_fails() {
        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        for i in 0..MERKLE_OCTREE_LEAVES {
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf_hashes[i] = *blake3::hash(&buf).as_bytes();
        }

        let tree = MerkleOctree::build_from_leaves(&leaf_hashes);
        let proof = MerkleProof::generate(&tree, 0).unwrap();

        let mut wrong_root = *tree.root();
        wrong_root[0] ^= 0xFF;
        assert!(!proof.verify(&wrong_root), "wrong root should fail");
    }
}
