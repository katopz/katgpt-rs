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
    pub sign: bool,
    /// KG triple confidence from extraction pipeline.
    /// Flows into SenseModule via builder, used to weight projection.
    /// Default 1.0 = no attenuation (backward compatible).
    pub confidence: f32,
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
        // Simple spatial partition: use first 2 dims for quadtree-like indexing
        // Mark nodes as occupied based on embedding magnitude
        for (dim, &val) in embedding.iter().enumerate() {
            if val.abs() > 0.1 {
                let word = dim / 64;
                let bit = dim % 64;
                if word < bits.len() {
                    bits[word] |= 1u64 << bit;
                }
            }
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

        for (i, &val) in embedding.iter().enumerate() {
            let mask = 1u64 << i;
            if val > 0.01 {
                pos_bits |= mask;
                scale_sum += val;
            } else if val < -0.01 {
                neg_bits |= mask;
                scale_sum += val.abs();
            }
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
}
