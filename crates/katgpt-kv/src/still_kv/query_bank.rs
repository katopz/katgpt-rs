//! Query bank: generates latent queries for cross-attention compaction.
//!
//! Each compaction strategy produces different initial queries that bias
//! the perceiver toward capturing strategy-relevant information.
//!
//! KV cache layout: flat `[seq_len * kv_dim]` where each row of `kv_dim`
//! floats represents one token. `latent_dim` is the query dimension — if
//! it differs from `kv_dim`, we truncate or zero-pad to bridge.

use crate::still_kv::compact_cache::CompactionStrategy;

/// Trait for generating latent queries for perceiver cross-attention.
pub trait QueryBank: Send + Sync {
    /// Generate latent queries for the given KV cache and token budget.
    ///
    /// # Arguments
    /// * `kv_cache` - Flat f32 KV cache buffer, shape `[seq_len * kv_dim]`
    /// * `budget` - Number of compact tokens to produce
    ///
    /// # Returns
    /// Flat f32 buffer of latent queries, shape `[budget * latent_dim]`.
    fn generate_queries(&self, kv_cache: &[f32], budget: usize) -> Vec<f32>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract one token row from flat kv_cache, truncating or zero-padding
/// to `latent_dim`.
fn token_to_latent(
    kv_cache: &[f32],
    token_idx: usize,
    kv_dim: usize,
    latent_dim: usize,
) -> Vec<f32> {
    let start = token_idx * kv_dim;
    let end = (start + kv_dim).min(kv_cache.len());
    let mut out = vec![0.0f32; latent_dim];
    let copy_len = (end - start).min(latent_dim);
    out[..copy_len].copy_from_slice(&kv_cache[start..start + copy_len]);
    out
}

/// Borrow one token row from flat kv_cache as a slice.
///
/// Fast path for the common case `kv_dim == latent_dim` (true for every built-in
/// `QueryBank` impl in this file, which sets `kv_dim = self.latent_dim`).
/// Avoids the per-call `Vec` allocation that `token_to_latent` does — this matters
/// because the k-means loops call this O(seq_len × iters × centroids) times.
///
/// Caller must guarantee `token_idx * kv_dim + kv_dim <= kv_cache.len()`, which
/// holds when `seq_len = kv_cache.len() / kv_dim` is used to bound `token_idx`.
#[inline]
fn token_slice(kv_cache: &[f32], token_idx: usize, kv_dim: usize) -> &[f32] {
    let start = token_idx * kv_dim;
    &kv_cache[start..start + kv_dim]
}

/// Squared Euclidean distance between two slices.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Sum of squared values.
fn sq_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

// ---------------------------------------------------------------------------
// T6: ClusterQueryBank — Mini-batch k-means++ centroids
// ---------------------------------------------------------------------------

/// Cluster centroid query bank — initializes queries as k-means++ centroids.
#[derive(Debug, Clone)]
pub struct ClusterQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for ClusterQueryBank {
    fn generate_queries(&self, kv_cache: &[f32], budget: usize) -> Vec<f32> {
        let kv_dim = self.latent_dim;
        let seq_len = if kv_dim == 0 {
            0
        } else {
            kv_cache.len() / kv_dim
        };

        if seq_len == 0 || budget == 0 || kv_dim == 0 {
            return vec![0.0f32; budget * self.latent_dim];
        }

        let n = budget.min(seq_len);

        // --- k-means++ initialization ---
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n);

        // First centroid: token 0
        centroids.push(token_to_latent(kv_cache, 0, kv_dim, self.latent_dim));

        // For deterministic results without RNG, use a simple linear congruential
        // generator seeded from the data.
        let mut seed: u64 = 42;
        for _ in 1..n {
            // Compute squared distance from each point to its nearest centroid.
            // Use borrowed slices (no per-token Vec allocation) — this is the hot
            // k-means++ init loop.
            let min_dists: Vec<f32> = (0..seq_len)
                .map(|t| {
                    let point = token_slice(kv_cache, t, kv_dim);
                    centroids
                        .iter()
                        .map(|c| sq_dist(point, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            // Convert to cumulative distribution
            let total: f32 = min_dists.iter().sum();
            if total < 1e-12 {
                // All points coincident — pick next token linearly
                centroids.push(token_to_latent(
                    kv_cache,
                    centroids.len().min(seq_len - 1),
                    kv_dim,
                    self.latent_dim,
                ));
                continue;
            }

            // Simple LCG for deterministic "random" selection
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let r = (seed as f32 / u64::MAX as f32) * total;

            let mut cumsum = 0.0f32;
            let mut chosen = seq_len - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                cumsum += d;
                if cumsum >= r {
                    chosen = i;
                    break;
                }
            }
            centroids.push(token_to_latent(kv_cache, chosen, kv_dim, self.latent_dim));
        }

        // --- k-means iterations (max 10) ---
        let max_iters = 10;
        let mut assignments = vec![0usize; seq_len];
        // Reusable scratch buffers — allocated once, zeroed each iteration.
        // Avoids `vec![vec![0.0; latent_dim]; n]` allocation every iteration.
        let mut sums = vec![0.0f32; n * self.latent_dim];
        let mut counts = vec![0usize; n];

        for _ in 0..max_iters {
            // Assign each point to nearest centroid (borrowed slices, no alloc)
            let mut changed = false;
            for (t, slot) in assignments.iter_mut().enumerate() {
                let point = token_slice(kv_cache, t, kv_dim);
                let (best_idx, _) = centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, sq_dist(point, c)))
                    .fold(
                        (0, f32::INFINITY),
                        |acc, (i, d)| if d < acc.1 { (i, d) } else { acc },
                    );
                if *slot != best_idx {
                    changed = true;
                    *slot = best_idx;
                }
            }

            if !changed {
                break;
            }

            // Update centroids as mean of assigned points
            sums.fill(0.0);
            counts.fill(0);
            for (t, &c) in assignments.iter().enumerate() {
                let point = token_slice(kv_cache, t, kv_dim);
                let sum_row = &mut sums[c * self.latent_dim..(c + 1) * self.latent_dim];
                for (s, p) in sum_row.iter_mut().zip(point.iter()) {
                    *s += p;
                }
                counts[c] += 1;
            }
            for (c, cnt) in counts.iter().enumerate() {
                if *cnt > 0 {
                    let inv = 1.0f32 / *cnt as f32;
                    let sum_row = &sums[c * self.latent_dim..(c + 1) * self.latent_dim];
                    for (cent, sum_val) in centroids[c].iter_mut().zip(sum_row.iter()) {
                        *cent = sum_val * inv;
                    }
                }
            }
        }

        // Flatten centroids into output
        let mut out = Vec::with_capacity(budget * self.latent_dim);
        for c in &centroids {
            out.extend_from_slice(c);
        }
        // Pad if budget > n (shouldn't happen but be safe)
        while out.len() < budget * self.latent_dim {
            out.extend_from_slice(&centroids[out.len() / self.latent_dim % n]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// T7: AttentionQueryBank — Magnitude-weighted importance sampling
// ---------------------------------------------------------------------------

/// Attention-weighted query bank — places queries at high-attention positions.
#[derive(Debug, Clone)]
pub struct AttentionQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for AttentionQueryBank {
    fn generate_queries(&self, kv_cache: &[f32], budget: usize) -> Vec<f32> {
        let kv_dim = self.latent_dim;
        let seq_len = if kv_dim == 0 {
            0
        } else {
            kv_cache.len() / kv_dim
        };

        if seq_len == 0 || budget == 0 || kv_dim == 0 {
            return vec![0.0f32; budget * self.latent_dim];
        }

        // Compute importance = sum of absolute values per token
        let mut importance: Vec<f32> = (0..seq_len)
            .map(|t| {
                let start = t * kv_dim;
                let end = (start + kv_dim).min(kv_cache.len());
                kv_cache[start..end].iter().map(|x| x.abs()).sum()
            })
            .collect();

        // Normalize to probability distribution
        let total: f32 = importance.iter().sum();
        if total < 1e-12 {
            // Uniform sampling fallback
            return sample_uniform(kv_cache, budget, seq_len, kv_dim, self.latent_dim);
        }

        let inv_total = 1.0 / total;
        for w in &mut importance {
            *w *= inv_total;
        }

        // Sample `budget` tokens proportional to importance using deterministic LCG.
        // `remaining_importance` is never mutated (importance stays — see comment below),
        // so `rem_total` is constant across iterations and can be hoisted out of the loop.
        // This also lets us drop the `importance.clone()` allocation.
        let rem_total: f32 = importance.iter().sum();
        if rem_total < 1e-12 {
            // Uniform sampling fallback
            return sample_uniform(kv_cache, budget, seq_len, kv_dim, self.latent_dim);
        }

        let mut seed: u64 = 12345;
        let mut chosen = Vec::with_capacity(budget);

        for _ in 0..budget {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let r = (seed as f32 / u64::MAX as f32) * rem_total;

            let mut cumsum = 0.0f32;
            let mut idx = seq_len - 1;
            for (i, &w) in importance.iter().enumerate() {
                cumsum += w;
                if cumsum >= r {
                    idx = i;
                    break;
                }
            }
            chosen.push(idx);
            // Don't remove — allow re-sampling for simplicity (importance stays)
        }

        // Pad if we didn't get enough (edge case)
        while chosen.len() < budget {
            chosen.push(chosen.len() % seq_len);
        }

        let mut out = Vec::with_capacity(budget * self.latent_dim);
        for &idx in &chosen[..budget] {
            out.extend_from_slice(&token_to_latent(kv_cache, idx, kv_dim, self.latent_dim));
        }
        out
    }
}

/// Uniform sampling fallback when importance is degenerate.
fn sample_uniform(
    kv_cache: &[f32],
    budget: usize,
    seq_len: usize,
    kv_dim: usize,
    latent_dim: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(budget * latent_dim);
    for i in 0..budget {
        let idx = i % seq_len;
        out.extend_from_slice(&token_to_latent(kv_cache, idx, kv_dim, latent_dim));
    }
    out
}

// ---------------------------------------------------------------------------
// T8: SpectralQueryBank — Top-k tokens by variance
// ---------------------------------------------------------------------------

/// Spectral projection query bank — top eigenvectors as initial queries.
///
/// Simplified first implementation: pick top-k tokens by per-token variance
/// across the kv_dim axis. These high-variance tokens approximate the most
/// information-rich directions in the data.
#[derive(Debug, Clone)]
pub struct SpectralQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for SpectralQueryBank {
    fn generate_queries(&self, kv_cache: &[f32], budget: usize) -> Vec<f32> {
        let kv_dim = self.latent_dim;
        let seq_len = if kv_dim == 0 {
            0
        } else {
            kv_cache.len() / kv_dim
        };

        if seq_len == 0 || budget == 0 || kv_dim == 0 {
            return vec![0.0f32; budget * self.latent_dim];
        }

        // Compute per-token variance across kv_dim
        let mut variances: Vec<(usize, f32)> = (0..seq_len)
            .map(|t| {
                let start = t * kv_dim;
                let end = (start + kv_dim).min(kv_cache.len());
                let slice = &kv_cache[start..end];
                let n = slice.len() as f32;
                if n < 1.0 {
                    (t, 0.0)
                } else {
                    let mean: f32 = slice.iter().sum::<f32>() / n;
                    let var: f32 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
                    (t, var)
                }
            })
            .collect();

        // Sort descending by variance
        variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top `budget` tokens and normalize each to unit length
        let mut out = Vec::with_capacity(budget * self.latent_dim);
        for i in 0..budget {
            let token_idx = variances[i % seq_len].0;
            let mut vec = token_to_latent(kv_cache, token_idx, kv_dim, self.latent_dim);
            let norm = sq_norm(&vec).sqrt().max(1e-8);
            let inv_norm = 1.0 / norm;
            for v in &mut vec {
                *v *= inv_norm;
            }
            out.extend_from_slice(&vec);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// T9: BfcfQueryBank — Region centroids
// ---------------------------------------------------------------------------

/// BFCF region-blend query bank — region-weighted initialization.
///
/// Divides the sequence into equal-sized regions and computes the centroid
/// (mean) of each region. Returns `budget` region centroids.
#[derive(Debug, Clone)]
pub struct BfcfQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for BfcfQueryBank {
    fn generate_queries(&self, kv_cache: &[f32], budget: usize) -> Vec<f32> {
        let kv_dim = self.latent_dim;
        let seq_len = if kv_dim == 0 {
            0
        } else {
            kv_cache.len() / kv_dim
        };

        if seq_len == 0 || budget == 0 || kv_dim == 0 {
            return vec![0.0f32; budget * self.latent_dim];
        }

        let num_regions = budget.min(seq_len);
        let chunk_size = seq_len.div_ceil(num_regions); // ceil div

        // Compute centroid for each region
        let mut regions: Vec<Vec<f32>> = Vec::with_capacity(num_regions);
        for r in 0..num_regions {
            let start_tok = r * chunk_size;
            let end_tok = (start_tok + chunk_size).min(seq_len);
            if start_tok >= seq_len {
                break;
            }
            let count = (end_tok - start_tok) as f32;
            let mut centroid = vec![0.0f32; self.latent_dim];
            for t in start_tok..end_tok {
                let token = token_slice(kv_cache, t, kv_dim);
                for (c, v) in centroid.iter_mut().zip(token.iter()) {
                    *c += v;
                }
            }
            if count > 0.0 {
                let inv = 1.0 / count;
                for c in &mut centroid {
                    *c *= inv;
                }
            }
            regions.push(centroid);
        }

        // Build output: if fewer regions than budget, cycle; if more, take top by magnitude
        let mut out = Vec::with_capacity(budget * self.latent_dim);

        if regions.len() >= budget {
            // Sort regions by magnitude (L2 norm) descending, take top budget
            let mut indexed: Vec<(usize, f32)> = regions
                .iter()
                .enumerate()
                .map(|(i, r)| (i, sq_norm(r)))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(idx, _) in &indexed[..budget] {
                out.extend_from_slice(&regions[idx]);
            }
        } else {
            // Cycle through available regions to fill budget
            for i in 0..budget {
                out.extend_from_slice(&regions[i % regions.len()]);
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// T10: MuxQueryBank — Hadamard-like quasi-orthogonal superposition
// ---------------------------------------------------------------------------

/// Mux superposition query bank — multiplexed encoding initialization.
///
/// Generates budget quasi-orthogonal vectors using a deterministic
/// Hadamard-like pattern. For query `i`, element `j` is set to
/// `±1/sqrt(latent_dim)` with the sign alternating based on `(i + j) % 2`
/// with an offset from `i` to decorrelate different queries.
#[derive(Debug, Clone)]
pub struct MuxQueryBank {
    /// Latent dimension per query.
    pub latent_dim: usize,
}

impl QueryBank for MuxQueryBank {
    fn generate_queries(&self, _kv_cache: &[f32], budget: usize) -> Vec<f32> {
        if budget == 0 || self.latent_dim == 0 {
            return vec![0.0f32; budget * self.latent_dim];
        }

        let scale = 1.0 / (self.latent_dim as f32).sqrt();
        let mut out = Vec::with_capacity(budget * self.latent_dim);

        for i in 0..budget {
            for j in 0..self.latent_dim {
                // Deterministic sign pattern using FNV-like mixing:
                // hash = i ^ (i >> 3) ^ (j << 5) ^ j
                // Then use bit parity for the sign.
                // This decorrelates different (i, j) pairs more effectively
                // than simple multiplication.
                let mut h: u64 = (i as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(j as u64)
                    .wrapping_mul(0xC6EF43637C9E1B5A);
                h ^= h >> 33;
                h = h.wrapping_mul(0xFF51AFD7ED558CCD);
                h ^= h >> 33;
                let sign: f32 = if h.count_ones().is_multiple_of(2) {
                    scale
                } else {
                    -scale
                };
                out.push(sign);
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a query bank for the given strategy.
pub fn create_query_bank(strategy: CompactionStrategy, latent_dim: usize) -> Box<dyn QueryBank> {
    match strategy {
        CompactionStrategy::ClusterCentroids => Box::new(ClusterQueryBank { latent_dim }),
        CompactionStrategy::AttentionWeighted => Box::new(AttentionQueryBank { latent_dim }),
        CompactionStrategy::SpectralProjection => Box::new(SpectralQueryBank { latent_dim }),
        CompactionStrategy::BfcfRegionBlend => Box::new(BfcfQueryBank { latent_dim }),
        CompactionStrategy::MuxSuperposition => Box::new(MuxQueryBank { latent_dim }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple kv_cache with `seq_len` tokens of dimension `kv_dim`.
    fn make_kv_cache(seq_len: usize, kv_dim: usize) -> Vec<f32> {
        (0..seq_len * kv_dim)
            .map(|i| i as f32 * 0.1 + 1.0)
            .collect()
    }

    #[test]
    fn test_create_query_bank_all_strategies() {
        let strategies = [
            CompactionStrategy::ClusterCentroids,
            CompactionStrategy::AttentionWeighted,
            CompactionStrategy::SpectralProjection,
            CompactionStrategy::BfcfRegionBlend,
            CompactionStrategy::MuxSuperposition,
        ];
        for strategy in strategies {
            let bank = create_query_bank(strategy, 32);
            let queries = bank.generate_queries(&[1.0f32; 64], 4);
            assert_eq!(queries.len(), 4 * 32);
        }
    }

    // ----- T6: ClusterQueryBank -----

    #[test]
    fn test_cluster_query_bank_nonzero() {
        let bank = ClusterQueryBank { latent_dim: 8 };
        let kv = make_kv_cache(20, 8);
        let budget = 4;
        let queries = bank.generate_queries(&kv, budget);

        assert_eq!(queries.len(), budget * 8);
        // Must not be all zeros
        let all_zero = queries.iter().all(|&v| v == 0.0);
        assert!(!all_zero, "ClusterQueryBank should return non-zero queries");
    }

    #[test]
    fn test_cluster_query_bank_converges() {
        // Create 3 well-separated clusters
        let mut kv = Vec::with_capacity(30 * 4);
        // Cluster A: around [10, 10, 10, 10]
        for _ in 0..10 {
            kv.extend_from_slice(&[10.0, 10.0, 10.0, 10.0]);
        }
        // Cluster B: around [0, 0, 0, 0]
        for _ in 0..10 {
            kv.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }
        // Cluster C: around [-10, -10, -10, -10]
        for _ in 0..10 {
            kv.extend_from_slice(&[-10.0, -10.0, -10.0, -10.0]);
        }

        let bank = ClusterQueryBank { latent_dim: 4 };
        let queries = bank.generate_queries(&kv, 3);

        assert_eq!(queries.len(), 3 * 4);
        // At least some centroids should be near cluster centers
        let has_positive = queries.iter().any(|&v| v > 5.0);
        let has_negative = queries.iter().any(|&v| v < -5.0);
        assert!(
            has_positive || has_negative,
            "Should find at least one real cluster center"
        );
    }

    #[test]
    fn test_cluster_query_bank_empty_input() {
        let bank = ClusterQueryBank { latent_dim: 4 };
        let queries = bank.generate_queries(&[], 3);
        assert_eq!(queries.len(), 3 * 4);
    }

    // ----- T7: AttentionQueryBank -----

    #[test]
    fn test_attention_query_bank_samples() {
        let bank = AttentionQueryBank { latent_dim: 8 };
        let kv = make_kv_cache(20, 8);
        let budget = 5;
        let queries = bank.generate_queries(&kv, budget);

        assert_eq!(queries.len(), budget * 8);
        // Should return non-zero vectors (input has non-zero values)
        let all_zero = queries.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "AttentionQueryBank should return non-zero queries"
        );

        // Each sampled vector should appear in the original kv_cache
        for q in 0..budget {
            let q_vec = &queries[q * 8..(q + 1) * 8];
            let found = (0..20).any(|t| {
                let start = t * 8;
                kv[start..start + 8] == *q_vec
            });
            assert!(found, "Sampled query should exist in kv_cache");
        }
    }

    #[test]
    fn test_attention_query_bank_prefers_high_magnitude() {
        // Create data where one token has much higher magnitude
        let mut kv = vec![0.01f32; 10 * 4]; // 10 tokens, dim 4, low magnitude
        // Token 5 has high magnitude
        kv[20..24].copy_from_slice(&[100.0, 100.0, 100.0, 100.0]);

        let bank = AttentionQueryBank { latent_dim: 4 };
        // Sample 10 queries — token 5 should appear frequently
        let queries = bank.generate_queries(&kv, 10);

        let high_mag_count = (0..10)
            .filter(|&q| {
                let v = &queries[q * 4..(q + 1) * 4];
                v[0] > 50.0
            })
            .count();

        // With importance sampling, high-magnitude token should appear multiple times
        assert!(
            high_mag_count >= 3,
            "High-magnitude token should be sampled frequently, got {high_mag_count}/10"
        );
    }

    // ----- T8: SpectralQueryBank -----

    #[test]
    fn test_spectral_query_bank_normalized() {
        let bank = SpectralQueryBank { latent_dim: 8 };
        let kv = make_kv_cache(20, 8);
        let budget = 4;
        let queries = bank.generate_queries(&kv, budget);

        assert_eq!(queries.len(), budget * 8);

        // Each query should be roughly unit-length
        for q in 0..budget {
            let v = &queries[q * 8..(q + 1) * 8];
            let norm = sq_norm(v).sqrt();
            assert!(
                (norm - 1.0).abs() < 0.1,
                "Query {q} should be unit-length, got norm={norm}"
            );
        }
    }

    #[test]
    fn test_spectral_query_bank_picks_high_variance() {
        // Create data where token 3 has highest variance
        let mut kv = vec![0.0f32; 5 * 4]; // 5 tokens, dim 4
        kv[0..4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]); // token 0: constant
        kv[4..8].copy_from_slice(&[2.0, 2.0, 2.0, 2.0]); // token 1: constant
        kv[8..12].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]); // token 2: constant
        kv[12..16].copy_from_slice(&[-10.0, 10.0, -10.0, 10.0]); // token 3: HIGH variance
        kv[16..20].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]); // token 4: constant

        let bank = SpectralQueryBank { latent_dim: 4 };
        let queries = bank.generate_queries(&kv, 2);

        // First query should be from token 3 (highest variance)
        // Token 3 is [-10, 10, -10, 10], normalized to unit length → each element ≈ ±0.5
        let first = &queries[0..4];
        let is_high_var = (first[0] - first[1]).abs() > 0.5;
        assert!(
            is_high_var,
            "SpectralQueryBank should prefer high-variance token, got {:?}",
            first
        );
    }

    // ----- T9: BfcfQueryBank -----

    #[test]
    fn test_bfcf_query_bank_regions() {
        let bank = BfcfQueryBank { latent_dim: 8 };
        let kv = make_kv_cache(40, 8);
        let budget = 4;
        let queries = bank.generate_queries(&kv, budget);

        assert_eq!(
            queries.len(),
            budget * 8,
            "Should return budget * latent_dim values"
        );

        // Should return non-zero centroids
        let all_zero = queries.iter().all(|&v| v == 0.0);
        assert!(
            !all_zero,
            "BfcfQueryBank should return non-zero region centroids"
        );
    }

    #[test]
    fn test_bfcf_query_bank_equal_regions() {
        // 20 tokens, dim 4, budget 4 → 4 regions of 5 tokens each
        let kv = make_kv_cache(20, 4);
        let bank = BfcfQueryBank { latent_dim: 4 };
        let queries = bank.generate_queries(&kv, 4);

        assert_eq!(queries.len(), 16);
        // Each region centroid should be the mean of its tokens
        // Region 0: tokens 0-4, each token t has values [0.1*t+1, 0.1*t+1.1, ...]
        // Just check non-zero and distinct
        let r0 = &queries[0..4];
        let r1 = &queries[4..8];
        assert_ne!(r0, r1, "Adjacent region centroids should differ");
    }

    // ----- T10: MuxQueryBank -----

    #[test]
    fn test_mux_query_bank_orthogonal() {
        let bank = MuxQueryBank { latent_dim: 64 };
        let kv = make_kv_cache(10, 64);
        let budget = 8;
        let queries = bank.generate_queries(&kv, budget);

        assert_eq!(queries.len(), budget * 64);

        // All queries should be non-zero
        let all_zero = queries.iter().all(|&v| v == 0.0);
        assert!(!all_zero);

        // Check that queries are diverse: no two should be identical
        for i in 0..budget {
            for j in (i + 1)..budget {
                let vi = &queries[i * 64..(i + 1) * 64];
                let vj = &queries[j * 64..(j + 1) * 64];
                assert_ne!(vi, vj, "MuxQueryBank queries {i} and {j} should differ");
            }
        }
    }

    #[test]
    fn test_mux_query_bank_unit_scale() {
        let latent_dim = 16;
        let bank = MuxQueryBank { latent_dim };
        let queries = bank.generate_queries(&[], 1);

        let expected_scale = 1.0 / (latent_dim as f32).sqrt();
        for &v in &queries {
            assert!(
                (v.abs() - expected_scale).abs() < 1e-6,
                "MuxQueryBank values should be ±1/sqrt(latent_dim)"
            );
        }
    }

    // ----- Edge cases -----

    #[test]
    fn test_zero_budget_returns_empty() {
        let bank = ClusterQueryBank { latent_dim: 4 };
        let kv = make_kv_cache(10, 4);
        let queries = bank.generate_queries(&kv, 0);
        assert!(queries.is_empty());
    }

    #[test]
    fn test_budget_exceeds_seq_len() {
        // budget > seq_len: should still produce budget queries (by repetition)
        let bank = ClusterQueryBank { latent_dim: 4 };
        let kv = make_kv_cache(3, 4); // only 3 tokens
        let queries = bank.generate_queries(&kv, 10);
        assert_eq!(queries.len(), 10 * 4);
    }
}
