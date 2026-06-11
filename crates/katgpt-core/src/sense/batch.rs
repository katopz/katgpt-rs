//! Batch sense projection for multiple NPCs.

use crate::sense::brain::NpcBrain;

#[cfg(feature = "sense_lod")]
use crate::sense::lod::SenseLodRouter;

/// Batch-project all brains. Pinned NPCs skip computation.
pub fn batch_project_all(brains: &[NpcBrain], results: &mut [Vec<f32>]) {
    assert_eq!(brains.len(), results.len());
    for (brain, result) in brains.iter().zip(results.iter_mut()) {
        brain.project_all_into(result);
    }
}

/// Batch-project with rayon parallelism when N > 64.
#[cfg(feature = "plasma_path")]
pub fn batch_project_all_par(brains: &[NpcBrain], results: &mut [Vec<f32>]) {
    use rayon::prelude::*;
    assert_eq!(brains.len(), results.len());
    if brains.len() > 64 {
        brains
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(brain, result)| {
                brain.project_all_into(result);
            });
    } else {
        batch_project_all(brains, results);
    }
}

/// Assign LOD levels to brains based on distances via router.
/// Uses `set_lod` to keep cached mask in sync.
/// Zero-allocation: routes per-element inline instead of collecting into a Vec.
#[cfg(feature = "sense_lod")]
pub fn assign_lods_to_brains(brains: &mut [NpcBrain], router: &SenseLodRouter, distances: &[f32]) {
    assert_eq!(brains.len(), distances.len());
    for (brain, &dist) in brains.iter_mut().zip(distances.iter()) {
        brain.set_lod(router.route(dist));
    }
}

/// Reset all brains to Full LOD. Used as fallback when no boundaries available.
#[cfg(feature = "sense_lod")]
pub fn reset_lods_to_full(brains: &mut [NpcBrain]) {
    use crate::sense::lod::SenseLodLevel;
    for brain in brains.iter_mut() {
        brain.set_lod(SenseLodLevel::Full);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
    use crate::types::SenseKind;

    #[test]
    fn test_batch_matches_individual() {
        let builder = SenseOctreeBuilder::new(3);
        let module = builder.build(
            SenseKind::SpatialSense,
            &[KgEmbedding {
                entity_hash: 1,
                relation_hash: 1,
                embedding: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                sign: true,
                confidence: 1.0,
            }],
        );

        let brains = vec![NpcBrain::compose(vec![module.clone()]); 5];
        let mut results = vec![vec![]; 5];
        batch_project_all(&brains, &mut results);

        let individual: Vec<f32> = brains[0].project_all();
        for result in &results {
            assert_eq!(result.len(), individual.len());
        }
    }
}

#[cfg(test)]
#[cfg(feature = "sense_lod")]
mod lod_tests {
    use super::*;
    use crate::sense::lod::{SenseLodLevel, SenseLodRouter};
    use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
    use crate::slod::ScaleBoundary;
    use crate::types::SenseKind;

    fn make_brains(n: usize) -> Vec<NpcBrain> {
        let builder = SenseOctreeBuilder::new(3);
        let module = builder.build(
            SenseKind::SpatialSense,
            &[KgEmbedding {
                entity_hash: 1,
                relation_hash: 1,
                embedding: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                sign: true,
                confidence: 1.0,
            }],
        );
        vec![NpcBrain::compose(vec![module.clone()]); n]
    }

    #[test]
    fn test_assign_lods_batch() {
        let boundaries = vec![
            ScaleBoundary {
                sigma: 1.0,
                k_star: 3,
                score: 0.8,
            },
            ScaleBoundary {
                sigma: 3.0,
                k_star: 1,
                score: 0.4,
            },
        ];
        let router = SenseLodRouter::from_boundaries(&boundaries).unwrap();
        let mut brains = make_brains(3);
        let distances = [0.5, 2.0, 5.0];
        assign_lods_to_brains(&mut brains, &router, &distances);
        assert_eq!(brains[0].active_lod, SenseLodLevel::Full);
        assert_eq!(brains[1].active_lod, SenseLodLevel::Compressed);
        assert_eq!(brains[2].active_lod, SenseLodLevel::Minimal);
    }

    #[test]
    fn test_reset_lods_to_full() {
        let mut brains = make_brains(2);
        brains[0].active_lod = SenseLodLevel::Minimal;
        brains[1].active_lod = SenseLodLevel::Compressed;
        reset_lods_to_full(&mut brains);
        assert_eq!(brains[0].active_lod, SenseLodLevel::Full);
        assert_eq!(brains[1].active_lod, SenseLodLevel::Full);
    }
}
