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
    pub fn build(&self, kind: SenseKind, embeddings: &[KgEmbedding]) -> SenseModule {
        let mut module = SenseModule {
            kind,
            version: 1,
            octree_depth: self.max_depth,
            n_directions: 0,
            _reserved: 0,
            octree_bits: [0; 4],
            directions: [TernaryDir::zero(); 8],
            confidence: if embeddings.is_empty() { 0.0 } else { 0.5 },
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
        for dim in 0..8.min(embedding.len()) {
            if embedding[dim].abs() > 0.1 {
                let word = dim / 64;
                let bit = dim % 64;
                if word < bits.len() {
                    bits[word] |= 1u64 << bit;
                }
            }
        }
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
        };
        let module = builder.build(SenseKind::SpatialSense, &[emb]);
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
            })
            .collect();
        let module = builder.build(SenseKind::FighterSense, &embeddings);
        assert_eq!(module.n_directions, 8); // capped at 8
        assert!(module.verify());
    }
}
