//! Sense octree builder — KG embeddings → bit-plane octree.

use crate::types::SenseKind;
use crate::types::SenseModule;
use crate::types::TernaryDir;

/// Lightweight KG embedding for octree construction.
#[derive(Clone, Debug)]
pub struct KgEmbedding {
    pub entity_hash: u64,
    pub relation_hash: u64,
    pub embedding: [f32; 8],
    pub confidence: f32,
    pub sign: bool,
}

/// Builds octree bit-planes from KG embeddings.
pub struct SenseOctreeBuilder {
    max_depth: u8,
}

impl SenseOctreeBuilder {
    pub fn new(max_depth: u8) -> Self {
        Self {
            max_depth: max_depth.min(3),
        }
    }

    /// Build a SenseModule from KG embeddings.
    ///
    /// Module confidence = mean of embedding confidences (KG weight bridge).
    /// Falls back to 0.5 when embeddings exist but have no confidence signal.
    pub fn build(&self, kind: SenseKind, embeddings: &[KgEmbedding]) -> SenseModule {
        let module_confidence = if embeddings.is_empty() {
            0.0
        } else {
            let mean: f32 =
                embeddings.iter().map(|e| e.confidence).sum::<f32>() / embeddings.len() as f32;
            if mean <= 0.0 { 0.5 } else { mean }
        };

        let mut module = SenseModule {
            kind,
            version: 1,
            octree_depth: self.max_depth,
            n_directions: 0,
            _reserved: 0,
            octree_bits: [0; 4],
            directions: [TernaryDir::zero(); 8],
            confidence: module_confidence,
            commitment: [0u8; 32],
        };

        if embeddings.is_empty() {
            module.commit();
            return module;
        }

        // Build octree occupancy from embeddings
        for emb in embeddings {
            self.insert_embedding(&mut module.octree_bits, &emb.embedding, emb.sign);
        }

        // Extract direction vectors from embeddings
        let n_dirs = embeddings.len().min(8);
        module.n_directions = n_dirs as u8;
        for (i, emb) in embeddings.iter().take(n_dirs).enumerate() {
            module.directions[i] = Self::embedding_to_ternary(&emb.embedding);
        }

        module.commit();
        module
    }

    fn insert_embedding(&self, bits: &mut [u64; 4], embedding: &[f32; 8], _sign: bool) {
        // dim 0..7 → word=0, bit=dim (8 dims fit in one u64)
        for (dim, &val) in embedding.iter().enumerate() {
            let mask = 1u64 << dim;
            // Branchless: set bit when |val| > 0.1
            bits[0] |= mask & ((val.abs() > 0.1) as u64).wrapping_neg();
        }
    }

    /// Build a SenseModule seeded from schema class centroids.
    ///
    /// Uses centroid-based embedding initialization (Plan 237) instead of
    /// raw embeddings. The centroid is quantized to ternary direction vectors.
    /// Falls back to random directions if no centroid is available.
    #[cfg(feature = "schema_centroid")]
    pub fn build_from_centroid(
        &self,
        kind: SenseKind,
        class_hashes: &[u64],
        cache: &super::schema_centroid::SchemaCentroidCache,
        rng: &mut fastrand::Rng,
    ) -> SenseModule {
        use super::schema_centroid::schema_init_entity;

        let embedding = schema_init_entity(class_hashes, cache, 0.3, rng);
        let module_confidence = 0.5; // Default for centroid-seeded modules

        let mut module = SenseModule {
            kind,
            version: 1,
            octree_depth: self.max_depth,
            n_directions: 1,
            _reserved: 0,
            octree_bits: [0; 4],
            directions: [TernaryDir::zero(); 8],
            confidence: module_confidence,
            commitment: [0u8; 32],
        };

        // Use centroid-derived embedding as the primary direction
        module.directions[0] = Self::embedding_to_ternary(&embedding);

        // Mark occupancy from the centroid embedding
        self.insert_embedding(&mut module.octree_bits, &embedding, true);

        module.commit();
        module
    }

    fn embedding_to_ternary(embedding: &[f32; 8]) -> TernaryDir {
        let mut pos_bits = 0u64;
        let mut neg_bits = 0u64;
        let mut scale_sum = 0.0f32;

        // Branchless: use bool-as-u64 to conditionally apply masks
        for (i, &val) in embedding.iter().enumerate() {
            let mask = 1u64 << i;
            let is_pos = (val > 0.01) as u64;
            let is_neg = (val < -0.01) as u64;
            pos_bits |= mask & is_pos.wrapping_neg();
            neg_bits |= mask & is_neg.wrapping_neg();
            scale_sum += val.abs() * (is_pos | is_neg) as u8 as f32;
        }

        let row_scale = if scale_sum > 0.0 {
            scale_sum / (pos_bits.count_ones() + neg_bits.count_ones()).max(1) as f32
        } else {
            0.0
        };

        TernaryDir {
            pos_bits,
            neg_bits,
            row_scale,
        }
    }

    /// Build a SenseModule with Merkle octree commitment.
    ///
    /// Replaces the flat BLAKE3 commitment with a hierarchical Merkle root.
    /// Each KG embedding is hashed into a leaf, then the full depth-3 Merkle
    /// tree is built bottom-up. The root hash replaces `commitment`.
    ///
    /// GOAT target: overhead < 2µs on top of existing `build()`.
    #[cfg(feature = "merkle_octree")]
    pub fn build_with_merkle(
        &self,
        kind: SenseKind,
        embeddings: &[KgEmbedding],
    ) -> (SenseModule, crate::merkle::MerkleOctree) {
        let mut module = self.build(kind, embeddings);
        let tree = Self::build_merkle_only(embeddings);

        // Replace flat BLAKE3 commitment with Merkle root
        module.commitment = *tree.root();

        (module, tree)
    }

    /// Build only the Merkle octree from KG embeddings (no SenseModule).
    ///
    /// Each embedding is serialized to bytes and BLAKE3-hashed as a leaf.
    /// Leaves beyond 64 are ignored. Unused leaves get zero hashes.
    ///
    /// Leaf data layout per embedding: entity_hash(8) || relation_hash(8) || embedding(32) || confidence(4) || sign(1) = 53 bytes.
    #[cfg(feature = "merkle_octree")]
    pub fn build_merkle_only(embeddings: &[KgEmbedding]) -> crate::merkle::MerkleOctree {
        use crate::merkle::{HASH_SIZE, MERKLE_OCTREE_LEAVES};

        let mut leaf_hashes = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
        let mut scratch = [0u8; 53]; // entity_hash(8) + relation_hash(8) + embedding(32) + confidence(4) + sign(1)

        for (i, emb) in embeddings.iter().enumerate() {
            if i >= MERKLE_OCTREE_LEAVES {
                break;
            }
            scratch[0..8].copy_from_slice(&emb.entity_hash.to_le_bytes());
            scratch[8..16].copy_from_slice(&emb.relation_hash.to_le_bytes());
            // embedding: 8 x f32 = 32 bytes
            for (j, val) in emb.embedding.iter().enumerate() {
                scratch[16 + j * 4..20 + j * 4].copy_from_slice(&val.to_le_bytes());
            }
            scratch[48..52].copy_from_slice(&emb.confidence.to_le_bytes());
            scratch[52] = emb.sign as u8;
            leaf_hashes[i] = *blake3::hash(&scratch).as_bytes();
        }

        crate::merkle::MerkleOctree::build_from_leaves(&leaf_hashes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input_all_zeros() {
        let builder = SenseOctreeBuilder::new(3);
        let module = builder.build(SenseKind::SpatialSense, &[]);
        assert_eq!(module.octree_bits, [0; 4]);
        assert_eq!(module.n_directions, 0);
        assert!(module.verify());
    }

    #[test]
    fn test_single_triple() {
        let builder = SenseOctreeBuilder::new(3);
        let emb = KgEmbedding {
            entity_hash: 1,
            relation_hash: 2,
            embedding: [0.5, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
            confidence: 1.0,
        };
        let module = builder.build(SenseKind::SpatialSense, &[emb]);
        assert_eq!(module.n_directions, 1);
        assert!(module.verify());
    }

    #[cfg(feature = "schema_centroid")]
    #[test]
    fn test_build_from_centroid() {
        use super::super::schema_centroid::SchemaCentroidCache;

        let cache = SchemaCentroidCache::new();
        let class_hash = 42u64;
        let embs: Vec<KgEmbedding> = (0..5)
            .map(|i| KgEmbedding {
                entity_hash: i as u64,
                relation_hash: 0,
                embedding: [1.0, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                sign: true,
                confidence: 1.0,
            })
            .collect();
        cache.compute_and_insert(class_hash, &embs);

        let builder = SenseOctreeBuilder::new(3);
        let mut rng = fastrand::Rng::with_seed(42);
        let module =
            builder.build_from_centroid(SenseKind::FighterSense, &[class_hash], &cache, &mut rng);

        assert_eq!(module.n_directions, 1);
        assert_eq!(module.kind, SenseKind::FighterSense);
        assert!(module.verify());
    }

    #[cfg(feature = "schema_centroid")]
    #[test]
    fn test_build_from_centroid_fallback() {
        use super::super::schema_centroid::SchemaCentroidCache;

        let cache = SchemaCentroidCache::new();
        // No centroids cached → should fall back to random init

        let builder = SenseOctreeBuilder::new(3);
        let mut rng = fastrand::Rng::with_seed(99);
        let module = builder.build_from_centroid(
            SenseKind::SpatialSense,
            &[404u64], // unknown class
            &cache,
            &mut rng,
        );

        assert_eq!(module.n_directions, 1);
        assert!(module.verify());
    }

    #[test]
    fn test_many_triples() {
        let builder = SenseOctreeBuilder::new(3);
        let embeddings: Vec<KgEmbedding> = (0..10)
            .map(|i| KgEmbedding {
                entity_hash: i as u64,
                relation_hash: i as u64 * 2,
                embedding: [i as f32 * 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                sign: i % 2 == 0,
                confidence: 1.0,
            })
            .collect();
        let module = builder.build(SenseKind::FighterSense, &embeddings);
        assert_eq!(module.n_directions, 8); // capped at 8
        assert!(module.verify());
    }

    #[cfg(feature = "merkle_octree")]
    #[test]
    fn test_build_with_merkle() {
        use crate::merkle::MerkleProof;

        let builder = SenseOctreeBuilder::new(3);
        let embeddings: Vec<KgEmbedding> = (0..8)
            .map(|i| KgEmbedding {
                entity_hash: i as u64,
                relation_hash: i as u64 * 3,
                embedding: [(i as f32) * 0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                sign: i % 2 == 0,
                confidence: 0.8 + i as f32 * 0.02,
            })
            .collect();

        let (module, tree) = builder.build_with_merkle(SenseKind::SpatialSense, &embeddings);

        // Module should have correct metadata
        assert_eq!(module.n_directions, 8);
        assert_eq!(module.kind, SenseKind::SpatialSense);

        // Commitment should equal Merkle root
        assert_eq!(module.commitment, *tree.root());

        // Merkle proofs should verify for all embedded leaves
        for i in 0..embeddings.len().min(64) {
            let proof = MerkleProof::generate(&tree, i as u8)
                .unwrap_or_else(|| panic!("proof generation failed for leaf {i}"));
            assert!(
                proof.verify(tree.root()),
                "proof for leaf {i} should verify against Merkle root"
            );
        }
    }

    #[cfg(feature = "merkle_octree")]
    #[test]
    fn test_build_with_merkle_empty() {
        let builder = SenseOctreeBuilder::new(3);
        let (module, tree) = builder.build_with_merkle(SenseKind::SpatialSense, &[]);

        assert_eq!(module.n_directions, 0);
        assert_eq!(module.commitment, *tree.root());
        // All leaves are zero, root should be hash-of-zeros
        assert_ne!(tree.root(), &[0u8; 32]);
    }

    #[cfg(feature = "merkle_octree")]
    #[test]
    fn test_build_merkle_only_deterministic() {
        let embeddings: Vec<KgEmbedding> = (0..5)
            .map(|i| KgEmbedding {
                entity_hash: i as u64,
                relation_hash: 0,
                embedding: [1.0, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                sign: true,
                confidence: 1.0,
            })
            .collect();

        let tree_a = SenseOctreeBuilder::build_merkle_only(&embeddings);
        let tree_b = SenseOctreeBuilder::build_merkle_only(&embeddings);

        // Deterministic: same embeddings → same root
        assert_eq!(tree_a.root(), tree_b.root());
    }
}
