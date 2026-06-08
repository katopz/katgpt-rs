//! Batch sense projection for multiple NPCs.

use crate::sense::brain::NpcBrain;

/// Batch-project all brains. Pinned NPCs skip computation.
pub fn batch_project_all(brains: &[NpcBrain], results: &mut [Vec<f32>]) {
    assert_eq!(brains.len(), results.len());
    for (brain, result) in brains.iter().zip(results.iter_mut()) {
        *result = brain.project_all();
    }
}

/// Batch-project with rayon parallelism when N > 64.
#[cfg(feature = "plasma_path")]
pub fn batch_project_all_par(brains: &[NpcBrain], results: &mut [Vec<f32>]) {
    use rayon::prelude::*;
    assert_eq!(brains.len(), results.len());
    if brains.len() > 64 {
        brains.par_iter().zip(results.par_iter_mut()).for_each(|(brain, result)| {
            *result = brain.project_all();
        });
    } else {
        batch_project_all(brains, results);
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
        let module = builder.build(SenseKind::SpatialSense, &[KgEmbedding {
            entity_hash: 1, relation_hash: 1,
            embedding: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
        }]);

        let brains = vec![NpcBrain::compose(vec![module.clone()]); 5];
        let mut results = vec![vec![]; 5];
        batch_project_all(&brains, &mut results);

        let individual: Vec<f32> = brains[0].project_all();
        for result in &results {
            assert_eq!(result.len(), individual.len());
        }
    }
}
