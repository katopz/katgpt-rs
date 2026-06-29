//! Activation-based influence proxy + cosine-similarity scoring.

use super::catalyst::catalyst_score;
use super::types::{InfluenceConfig, MechInfluenceScore};

/// Proxy that maps activation vectors to influence scores via similarity to
/// stored catalyst activation patterns.
#[derive(Debug, Clone)]
pub struct ActivationInfluenceProxy {
    /// Stored activation patterns from known good outcomes.
    catalyst_patterns: Vec<Vec<f32>>,
    /// Dimensionality of activation vectors.
    dim: usize,
}

impl ActivationInfluenceProxy {
    /// Create a new proxy expecting activation vectors of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            catalyst_patterns: Vec::new(),
            dim,
        }
    }

    /// Record a catalyst activation vector with its outcome quality.
    /// High-quality outcomes get stronger storage (multiple copies).
    pub fn record_catalyst_activation(&mut self, activations: &[f32], outcome_quality: f32) {
        assert_eq!(
            activations.len(),
            self.dim,
            "activation dimension mismatch: expected {}, got {}",
            self.dim,
            activations.len()
        );
        // Store copies proportional to quality (1-3 copies)
        let copies = (outcome_quality * 3.0).ceil() as usize;
        let copies = copies.clamp(1, 3);
        for _ in 0..copies {
            self.catalyst_patterns.push(activations.to_vec());
        }
    }

    /// Compute influence score for a new activation vector.
    /// Returns max cosine similarity to any stored catalyst pattern.
    pub fn influence_score(&self, activations: &[f32]) -> f32 {
        assert_eq!(
            activations.len(),
            self.dim,
            "activation dimension mismatch: expected {}, got {}",
            self.dim,
            activations.len()
        );

        if self.catalyst_patterns.is_empty() {
            return 0.0;
        }

        let mut best_sim = f32::NEG_INFINITY;
        for pattern in &self.catalyst_patterns {
            let sim = cosine_similarity(activations, pattern);
            best_sim = best_sim.max(sim);
        }

        // Normalize from [-1, 1] to [0, 1]
        (best_sim + 1.0) / 2.0
    }

    /// Number of stored catalyst patterns.
    pub fn stored_count(&self) -> usize {
        self.catalyst_patterns.len()
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Rank a batch of text samples by influence score.
///
/// Returns a sorted list of `(index, MechInfluenceScore)` in descending order
/// of `catalyst_overlap`. The top-K fraction is marked `is_high_influence`.
pub fn batch_influence_rank(
    samples: &[&str],
    proxy: &ActivationInfluenceProxy,
    config: &InfluenceConfig,
) -> Vec<(usize, MechInfluenceScore)> {
    let mut results: Vec<(usize, MechInfluenceScore)> = samples
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let mut score = catalyst_score(text, config);
            // Combine catalyst overlap with activation similarity
            let activation_sim = if proxy.stored_count() > 0 {
                // Use a deterministic activation proxy from text hash
                let mut hasher = blake3::Hasher::new();
                hasher.update(text.as_bytes());
                let hash = hasher.finalize();
                let hash_bytes = hash.as_bytes();
                let dim = 8; // use first 32 bytes as 8 f32s
                let mut activations = [0.0f32; 8];
                for j in 0..dim {
                    let bytes = &hash_bytes[j * 4..(j + 1) * 4];
                    activations[j] = f32::from_bits(u32::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                    ]));
                }
                proxy.influence_score(&activations)
            } else {
                0.0
            };
            score.catalyst_overlap = score.catalyst_overlap * 0.7 + activation_sim * 0.3;
            (i, score)
        })
        .collect();

    // Sort descending by catalyst_overlap
    results.sort_by(|a, b| {
        b.1.catalyst_overlap
            .partial_cmp(&a.1.catalyst_overlap)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Mark top-K as high influence
    let top_k = ((samples.len() as f32) * config.top_k_fraction).ceil() as usize;
    let top_k = top_k.max(1).min(samples.len());
    for (rank, (_, score)) in results.iter_mut().enumerate() {
        score.is_high_influence = rank < top_k;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = [1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "identical vectors should have sim ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-5,
            "orthogonal vectors should have sim ~0.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v = [1.0, 2.0, 3.0];
        let opp = [-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&v, &opp);
        assert!(
            (sim + 1.0).abs() < 1e-5,
            "opposite vectors should have sim ~-1.0, got {sim}"
        );
    }

    #[test]
    fn test_proxy_influence_empty() {
        let proxy = ActivationInfluenceProxy::new(4);
        assert_eq!(proxy.influence_score(&[1.0, 0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_proxy_record_and_score() {
        let mut proxy = ActivationInfluenceProxy::new(3);
        proxy.record_catalyst_activation(&[1.0, 0.0, 0.0], 1.0);
        proxy.record_catalyst_activation(&[0.0, 1.0, 0.0], 0.5);

        let score = proxy.influence_score(&[1.0, 0.0, 0.0]);
        // Should be near 1.0 since we stored a copy of this exact pattern
        assert!(score > 0.9, "expected high similarity, got {score}");
        assert_eq!(proxy.stored_count(), 5); // 3 copies for quality=1.0 + 2 copies for quality=0.5
    }

    #[test]
    fn test_batch_influence_rank() {
        let config = InfluenceConfig {
            top_k_fraction: 0.5,
            catalyst_threshold: 0.0,
            ..Default::default()
        };
        let proxy = ActivationInfluenceProxy::new(8);

        let samples = [
            "<root><a>1</a><b>2</b></root>",
            "normal text here",
            "fn foo() { let x = 1; }",
        ];

        let ranked = batch_influence_rank(&samples, &proxy, &config);
        assert_eq!(ranked.len(), 3);

        // Top-K (top 50% = 2 samples) should be marked high influence
        let high_count = ranked.iter().filter(|(_, s)| s.is_high_influence).count();
        assert_eq!(high_count, 2, "top 50% of 3 = 2 high-influence samples");
    }
}
