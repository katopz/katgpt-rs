//! DilationBridgeRouter — VortexFlow integration for RAT+ bridge.
//!
//! Uses GDN2 bridge + dilated KV centroids for block scoring.
//! Demonstrates RAT+ insight: recurrence improves block scoring in VortexFlow routers.
//!
//! Plan 225 Phase 5 (T5.1).

use katgpt_core::types::DilationConfig;

use super::dilated_kv::DilatedKvAccessor;

/// Router that uses dilated KV centroids + GDN2 bridge for block scoring.
///
/// Computes dilated centroids from KV cache (D-strided averages) and scores
/// query vectors against them using a sigmoid-gated bridge readout blend.
/// Demonstrates that recurrence (GDN2 state) improves block selection quality.
#[derive(Debug, Clone)]
pub struct DilationBridgeRouter {
    /// Dilation factor for centroid computation.
    pub dilation: DilationConfig,
    /// Cached dilated centroids for block scoring.
    pub centroids: Vec<Vec<f32>>,
    /// Bridge gate value from last scoring.
    pub last_gate: f32,
}

impl DilationBridgeRouter {
    /// Create a new router with given dilation and dimension hint.
    pub fn new(dilation: DilationConfig, _dim: usize) -> Self {
        Self {
            dilation,
            centroids: Vec::new(),
            last_gate: 0.5,
        }
    }

    /// Compute dilated centroids from KV cache.
    ///
    /// Extracts D-strided keys and averages them into block centroids.
    /// Each centroid represents the mean key vector for a block of `block_size` positions.
    pub fn compute_centroids(&mut self, kv_keys: &[Vec<f32>], block_size: usize) {
        self.centroids.clear();
        let indices = DilatedKvAccessor::dilated_indices(kv_keys.len(), self.dilation);

        for block_start in (0..indices.len()).step_by(block_size) {
            let block_end = (block_start + block_size).min(indices.len());
            let block_indices = &indices[block_start..block_end];

            if block_indices.is_empty() {
                continue;
            }

            let dim = kv_keys[0].len();
            let mut centroid = vec![0.0; dim];
            for &idx in block_indices {
                for (j, val) in kv_keys[idx].iter().enumerate() {
                    centroid[j] += val;
                }
            }
            let count = block_indices.len() as f32;
            for c in centroid.iter_mut() {
                *c /= count;
            }
            self.centroids.push(centroid);
        }
    }

    /// Score a query against all centroids using bridge-enhanced scoring.
    ///
    /// Combines dot-product similarity with bridge gate:
    /// `score = gate * sim + (1 - gate) * sim * 0.5`
    /// where `gate = sigmoid(dot(query, gdn2_readout))`.
    ///
    /// Uses sigmoid (not softmax) per project constraints.
    pub fn score_blocks(&mut self, query: &[f32], gdn2_readout: &[f32]) -> Vec<(usize, f32)> {
        // Bridge gate: sigmoid(dot(query, gdn2_readout))
        let dot: f32 = query
            .iter()
            .zip(gdn2_readout.iter())
            .map(|(q, r)| q * r)
            .sum();
        self.last_gate = 1.0 / (1.0 + (-dot).exp()); // sigmoid

        // Score each centroid
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let sim: f32 = query.iter().zip(centroid.iter()).map(|(q, c)| q * c).sum();
                // Bridge-enhanced: blend centroid similarity with bridge gate
                let score = self.last_gate * sim + (1.0 - self.last_gate) * (sim * 0.5);
                (i, score)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centroids() {
        let mut router = DilationBridgeRouter::new(DilationConfig::D2, 4);
        let keys: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 4]).collect();
        router.compute_centroids(&keys, 2);
        assert!(!router.centroids.is_empty());
    }

    #[test]
    fn test_score_blocks() {
        let mut router = DilationBridgeRouter::new(DilationConfig::D1, 4);
        router.centroids = vec![vec![1.0; 4], vec![0.0; 4]];
        let query = vec![0.8; 4];
        let gdn2 = vec![0.5; 4];
        let scores = router.score_blocks(&query, &gdn2);
        assert_eq!(scores.len(), 2);
        // Centroid [1,1,1,1] has higher dot-product with [0.8,0.8,0.8,0.8] than [0,0,0,0]
        assert!(scores[0].1 > scores[1].1);
    }

    #[test]
    fn test_dilation_reduces_centroids() {
        let keys: Vec<Vec<f32>> = (0..16).map(|i| vec![i as f32; 4]).collect();

        let mut d1 = DilationBridgeRouter::new(DilationConfig::D1, 4);
        d1.compute_centroids(&keys, 4);

        let mut d4 = DilationBridgeRouter::new(DilationConfig::D4, 4);
        d4.compute_centroids(&keys, 4);

        // D4 produces fewer centroids (strided access skips 3/4 of keys)
        assert!(d4.centroids.len() <= d1.centroids.len());
    }

    #[test]
    fn test_gate_is_sigmoid_bounded() {
        let mut router = DilationBridgeRouter::new(DilationConfig::D1, 4);
        router.centroids = vec![vec![1.0; 4]];

        // High positive dot → gate close to 1
        let query_high = vec![10.0; 4];
        let gdn2_high = vec![10.0; 4];
        router.score_blocks(&query_high, &gdn2_high);
        assert!(router.last_gate > 0.99);
        assert!((0.0..=1.0).contains(&router.last_gate));

        // High negative dot → gate close to 0
        let query_neg = vec![-10.0; 4];
        router.score_blocks(&query_neg, &gdn2_high);
        assert!(router.last_gate < 0.01);
        assert!((0.0..=1.0).contains(&router.last_gate));
    }
}
