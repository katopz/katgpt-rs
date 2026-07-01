//! Merkle octree — hierarchical BLAKE3 commitment for KG latent octree nodes.
//!
//! Depth-3 octree = 73 nodes: 1 root + 8 internal + 64 leaves.
//! Each node stores a `[u8; 32]` BLAKE3 hash.
//! Feature-gated behind `merkle_octree`.
//!
//! ## Co-extraction provenance (Plan 338 Phase 2.5)
//!
//! The Merkle octree surface (constants + `MerkleOctree` + `MerkleProof` +
//! impls) was promoted to `katgpt_types::merkle` so the promoted
//! `katgpt-sense` crate (which uses `build_with_merkle` in `octree.rs`)
//! can depend on the leaf only, breaking the katgpt-core cycle. This file is
//! now a thin re-export shim — `katgpt_core::merkle::*` paths are preserved
//! bit-for-bit. Tests stay here and exercise the surface through the
//! re-export.

pub use katgpt_types::merkle::{
    HASH_SIZE, MERKLE_OCTREE_BRANCHING, MERKLE_OCTREE_DEPTH, MERKLE_OCTREE_INTERNAL,
    MERKLE_OCTREE_LEAVES, MERKLE_OCTREE_NODES, MerkleOctree, MerkleProof,
};

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
