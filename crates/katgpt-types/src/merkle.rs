//! Merkle octree — hierarchical BLAKE3 commitment for KG latent octree nodes.
//!
//! Depth-3 octree = 73 nodes: 1 root + 8 internal + 64 leaves.
//! Each node stores a `[u8; 32]` BLAKE3 hash.
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
//!
//! ## Co-extraction provenance (Plan 338 Phase 2.5)
//!
//! Promoted from `katgpt-core::merkle` (Plan 221-M) so that the promoted
//! `katgpt-sense` crate (which uses `build_with_merkle` in `octree.rs` behind
//! the `merkle_octree` feature) can depend on the leaf only. katgpt-core
//! re-exports this via `katgpt_core::merkle::*` (bit-for-bit path preserved,
//! feature-gate `merkle_octree` preserved). Tests stay in katgpt-core.

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
        // Bulk copy: all 64 leaf hashes are contiguous in both source and destination.
        tree.hashes[MERKLE_OCTREE_INTERNAL + 1..=MERKLE_OCTREE_INTERNAL + MERKLE_OCTREE_LEAVES]
            .copy_from_slice(leaf_hashes);

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
