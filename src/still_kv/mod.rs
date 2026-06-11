//! StillKV: Perceiver-based KV cache compaction — modelless (Plan 245).
//!
//! Compacts KV caches via cross-attention synthesis without model-specific training.
//! Architecture mirrors STILL (Baseten, 2026): per-layer Perceiver compactor with
//! 2d latent dimension ([K_free; V] concat), β additive attention bias,
//! internal RoPE on cross-attention queries/keys, no final RMSNorm.
//!
//! β strategies:
//! - **β-A (MassMatching)**: log(T/t) uniform scalar baseline
//! - **β-D (VortexFlow)**: attention-concentration-weighted per-latent bias
//!
//! Query bank strategies:
//! - **ClusterCentroids**: k-means-style cluster representatives
//! - **AttentionWeighted**: attention-score-weighted importance sampling
//! - **SpectralProjection**: PCA/SVD low-rank projection
//! - **BfcfRegionBlend**: BFCF region-weighted blending
//! - **MuxSuperposition**: multiplexed superposition encoding

pub mod beta_bias;
pub mod compact_cache;
pub mod iterative;
pub mod perceiver;
pub mod position_free;
pub mod query_bank;

pub use beta_bias::{BetaBias, BetaStrategy, compute_beta_mass_matching, compute_beta_vortex_flow};
pub use compact_cache::{CompactKVCache, CompactionMeta, CompactionStrategy};
pub use iterative::{IterativeChunkCompactor, KVChunk};
pub use perceiver::{StillPerceiver, StillPerceiverConfig};
pub use position_free::PositionFreeCompactor;
pub use query_bank::QueryBank;

/// Compute cosine similarity between two flat f32 vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in (0..a.len()).step_by(4) {
        let end = (i + 4).min(a.len());
        for j in i..end {
            dot += a[j] * b[j];
            norm_a += a[j] * a[j];
            norm_b += b[j] * b[j];
        }
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use half::f16;

    /// T19: Position-free compaction round-trip.
    /// un-rotate → compact → re-rotate should approximately preserve semantics.
    #[test]
    fn test_position_free_compaction_roundtrip() {
        let head_dim = 16;
        let num_heads = 2;
        let seq_len = 32;
        let rope_theta = 10000.0;

        let compactor = PositionFreeCompactor::new(rope_theta, head_dim);

        // Create synthetic keys at position 0
        let original_f32: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let original_f16: Vec<f16> = original_f32.iter().map(|&v| f16::from_f32(v)).collect();

        // Un-rotate at pos 0 (should be identity since angle=0)
        let position_free = compactor.un_rotate_keys(&original_f16, 0);

        // Compact: just take first 16 tokens (simple truncation budget)
        let budget = 16;
        let compact_f32 = position_free[..budget * head_dim].to_vec();

        // Re-rotate at new position
        let new_start_pos = 0;
        let re_rotated_f16 = compactor.re_rotate_keys(&compact_f32, new_start_pos);

        // Verify: at pos 0, rotation is identity, so f16 round-trip should be exact
        for i in 0..budget * head_dim {
            let original = f16::from_f32(original_f32[i]);
            assert_eq!(re_rotated_f16[i], original, "Mismatch at index {}", i);
        }
    }

    /// T19b: Non-trivial position round-trip.
    /// un-rotate at pos 100, re-rotate at pos 100 should recover original.
    #[test]
    fn test_position_free_compaction_roundtrip_nontrivial_pos() {
        let head_dim = 16;
        let seq_len = 8;
        let rope_theta = 10000.0;
        let start_pos = 100;

        let compactor = PositionFreeCompactor::new(rope_theta, head_dim);

        let original_f32: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let original_f16: Vec<f16> = original_f32.iter().map(|&v| f16::from_f32(v)).collect();

        // Un-rotate at start_pos
        let position_free = compactor.un_rotate_keys(&original_f16, start_pos);

        // Re-rotate at same position
        let recovered_f16 = compactor.re_rotate_keys(&position_free, start_pos);

        // Should recover original within f16 precision
        for i in 0..seq_len * head_dim {
            let diff = (recovered_f16[i].to_f32() - original_f16[i].to_f32()).abs();
            assert!(
                diff < 0.01,
                "Round-trip error too large at index {}: {}",
                i,
                diff
            );
        }
    }

    /// T20: compact_into produces correct budget size via IterativeChunkCompactor.
    #[test]
    fn test_compact_into_correct_budget() {
        let chunk_size = 16;
        let num_heads = 2;
        let head_dim = 8;
        let compression_ratio = 4;
        let rope_theta = 10000.0;
        let tokens_per_elem = num_heads * head_dim;

        let compactor = IterativeChunkCompactor::new(
            chunk_size,
            0,
            num_heads,
            head_dim,
            CompactionStrategy::ClusterCentroids,
            rope_theta,
            compression_ratio,
        );

        // Create 32 tokens of data (2 chunks)
        let total_tokens = 32;
        let keys = vec![f16::from_f32(1.0); total_tokens * tokens_per_elem];
        let values = vec![f16::from_f32(2.0); total_tokens * tokens_per_elem];

        let chunks = compactor.split_into_chunks(&keys, &values, 0);
        assert_eq!(chunks.len(), 2);

        let budget = compactor.compact_budget();
        assert_eq!(budget, chunk_size / compression_ratio);

        // Compact first chunk
        let compacted = compactor.compact_chunk(&chunks[0], None, budget);
        assert_eq!(compacted.len, budget);
    }

    /// T21: Iterative compaction produces linear growth at rate 1/c.
    #[test]
    fn test_iterative_linear_growth() {
        let chunk_size = 16;
        let num_heads = 2;
        let head_dim = 8;
        let compression_ratio = 4;
        let rope_theta = 10000.0;
        let tokens_per_elem = num_heads * head_dim;

        let compactor = IterativeChunkCompactor::new(
            chunk_size,
            0,
            num_heads,
            head_dim,
            CompactionStrategy::ClusterCentroids,
            rope_theta,
            compression_ratio,
        );

        // Create 64 tokens (4 chunks)
        let total_tokens = 64;
        let keys: Vec<f16> = (0..total_tokens * tokens_per_elem)
            .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
            .collect();
        let values: Vec<f16> = (0..total_tokens * tokens_per_elem)
            .map(|i| f16::from_f32((i as f32 * 0.2).cos()))
            .collect();

        let chunks = compactor.split_into_chunks(&keys, &values, 0);
        assert_eq!(chunks.len(), 4);

        let stream_result = compactor.compact_stream(chunks);

        // Each compacted chunk should have budget = 16/4 = 4 tokens
        let budget = chunk_size / compression_ratio;
        for (i, chunk) in stream_result.iter().enumerate() {
            assert_eq!(
                chunk.len, budget,
                "Chunk {} has {} tokens, expected {}",
                i, chunk.len, budget
            );
        }

        // Total compact tokens = 4 chunks × 4 tokens = 16
        let total_compact: usize = stream_result.iter().map(|c| c.len).sum();
        assert_eq!(total_compact, 16);

        // Compression ratio: 64 original → 16 compact = 4x
        assert_eq!(total_tokens / total_compact, compression_ratio);
    }

    /// End-to-end: full pipeline with all strategies.
    #[test]
    fn test_full_pipeline_all_strategies() {
        let strategies = [
            CompactionStrategy::ClusterCentroids,
            CompactionStrategy::AttentionWeighted,
            CompactionStrategy::SpectralProjection,
            CompactionStrategy::BfcfRegionBlend,
            CompactionStrategy::MuxSuperposition,
        ];

        for strategy in strategies {
            let compactor = IterativeChunkCompactor::new(16, 0, 2, 8, strategy, 10000.0, 4);

            let keys = vec![f16::from_f32(1.0); 16 * 16];
            let values = vec![f16::from_f32(2.0); 16 * 16];
            let chunks = compactor.split_into_chunks(&keys, &values, 0);

            let budget = compactor.compact_budget();
            let compacted = compactor.compact_chunk(&chunks[0], None, budget);

            assert_eq!(
                compacted.len, budget,
                "Strategy {:?} produced wrong budget",
                strategy
            );
            assert!(!compacted.keys.is_empty());
            assert!(!compacted.values.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // T22-T24: Benchmarks & GOAT gate
    // -----------------------------------------------------------------------

    /// Generate synthetic KV data: sine waves at different frequencies per head.
    ///
    /// Returns (keys_f16, values_f16) each of shape `[seq_len * num_heads * head_dim]`.
    fn generate_synthetic_kv(
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> (Vec<f16>, Vec<f16>) {
        let total = seq_len * num_heads * head_dim;
        let kv_dim = num_heads * head_dim;

        // Keys: sine wave with frequency varying by head
        let keys: Vec<f16> = (0..total)
            .map(|i| {
                let token = i / kv_dim;
                let elem = i % kv_dim;
                let head = elem / head_dim;
                let freq = (head as f32 + 1.0) * 0.03; // different freq per head
                let val = (token as f32 * freq + elem as f32 * 0.01).sin();
                f16::from_f32(val)
            })
            .collect();

        // Values: cosine wave with different frequencies
        let values: Vec<f16> = (0..total)
            .map(|i| {
                let token = i / kv_dim;
                let elem = i % kv_dim;
                let head = elem / head_dim;
                let freq = (head as f32 + 1.0) * 0.05;
                let val = (token as f32 * freq + elem as f32 * 0.02).cos();
                f16::from_f32(val)
            })
            .collect();

        (keys, values)
    }

    /// Compute MSE between two f32 slices.
    fn compute_mse(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        if a.is_empty() {
            return 0.0;
        }
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        sum / a.len() as f32
    }

    /// Compute mean vector across tokens from a flat `[seq_len * dim]` buffer.
    fn mean_across_tokens(data: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
        let mut mean = vec![0.0f32; dim];
        for t in 0..seq_len {
            let base = t * dim;
            for d in 0..dim {
                mean[d] += data[base + d];
            }
        }
        let inv = 1.0 / seq_len as f32;
        for m in mean.iter_mut() {
            *m *= inv;
        }
        mean
    }

    /// Run compaction on synthetic data, return (compact_keys_f32, compact_values_f32, elapsed).
    fn run_compaction(
        keys_f16: &[f16],
        values_f16: &[f16],
        num_heads: usize,
        head_dim: usize,
        compression_ratio: usize,
        chunk_size: usize,
        strategy: CompactionStrategy,
    ) -> (Vec<f32>, Vec<f32>, std::time::Duration) {
        let _kv_dim = num_heads * head_dim;
        let compactor = IterativeChunkCompactor::new(
            chunk_size,
            0,
            num_heads,
            head_dim,
            strategy,
            10000.0,
            compression_ratio,
        );

        let chunks = compactor.split_into_chunks(keys_f16, values_f16, 0);
        let budget = compactor.compact_budget();

        let start = std::time::Instant::now();
        let compacted = compactor.compact_chunk(&chunks[0], chunks.get(1), budget);
        let elapsed = start.elapsed();

        // Convert compact output to f32 for quality measurement
        let compact_keys_f32: Vec<f32> = compacted.keys.iter().map(|v| v.to_f32()).collect();
        let compact_values_f32: Vec<f32> = compacted.values.iter().map(|v| v.to_f32()).collect();

        (compact_keys_f32, compact_values_f32, elapsed)
    }

    /// T22: Benchmark — StillKV compression quality at 8x, 16x, 32x.
    ///
    /// Measures cosine similarity and MSE between original and compacted KV
    /// at three compression ratios. Prints results as a table.
    #[test]
    fn bench_t22_compression_quality() {
        let seq_len = 1024;
        let num_heads = 8;
        let head_dim = 64;
        let kv_dim = num_heads * head_dim;
        let chunk_size = 256; // must be divisible by all compression ratios

        let (keys_f16, values_f16) = generate_synthetic_kv(seq_len, num_heads, head_dim);

        // Convert first chunk to f32 for comparison
        let first_chunk_tokens = chunk_size;
        let keys_f32: Vec<f32> = keys_f16[..first_chunk_tokens * kv_dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let _values_f32: Vec<f32> = values_f16[..first_chunk_tokens * kv_dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();

        let strategies = [
            CompactionStrategy::ClusterCentroids,
            CompactionStrategy::AttentionWeighted,
            CompactionStrategy::SpectralProjection,
            CompactionStrategy::BfcfRegionBlend,
            CompactionStrategy::MuxSuperposition,
        ];
        let ratios = [8usize, 16, 32];

        println!(
            "\n=== T22: StillKV Compression Quality ({} tokens × {} heads × {} dim) ===",
            seq_len, num_heads, head_dim
        );
        println!(
            "{:>25} | {:>4}x | {:>10} | {:>10} | {:>10}",
            "Strategy", "Rx", "CosSim(K)", "MSE(K)", "Time(ms)"
        );
        println!("{}", "-".repeat(80));

        for &strategy in &strategies {
            for &ratio in &ratios {
                let (ck, _cv, elapsed) = run_compaction(
                    &keys_f16,
                    &values_f16,
                    num_heads,
                    head_dim,
                    ratio,
                    chunk_size,
                    strategy,
                );

                let compact_tokens = ck.len() / kv_dim;

                // Quality metric 1: cosine similarity of mean-pooled tokens
                let orig_mean = mean_across_tokens(&keys_f32, first_chunk_tokens, kv_dim);
                let compact_mean = mean_across_tokens(&ck, compact_tokens, kv_dim);
                let cos_sim = cosine_similarity(&orig_mean, &compact_mean);

                // Quality metric 2: MSE between original first compact_tokens and compact
                let orig_prefix: Vec<f32> = keys_f32[..compact_tokens * kv_dim].to_vec();
                let mse = compute_mse(&orig_prefix, &ck);

                println!(
                    "{:>25} | {:>4}x | {:>10.4} | {:>10.6} | {:>10.2}",
                    format!("{:?}", strategy),
                    ratio,
                    cos_sim,
                    mse,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
    }

    /// H2O-style selection baseline: pick top-k tokens by L2 norm.
    ///
    /// Returns selected keys and values in f32, each `[budget * kv_dim]`.
    fn select_topk_by_norm(
        keys_f16: &[f16],
        values_f16: &[f16],
        num_tokens: usize,
        kv_dim: usize,
        budget: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // Compute L2 norm per token
        let mut norms: Vec<(usize, f32)> = (0..num_tokens)
            .map(|t| {
                let base = t * kv_dim;
                let norm: f32 = (base..base + kv_dim)
                    .map(|i| {
                        let v = keys_f16[i].to_f32();
                        v * v
                    })
                    .sum();
                (t, norm)
            })
            .collect();

        // Sort descending by norm, take top budget
        norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut sel_keys = Vec::with_capacity(budget * kv_dim);
        let mut sel_values = Vec::with_capacity(budget * kv_dim);
        for i in 0..budget.min(norms.len()) {
            let t = norms[i].0;
            let base = t * kv_dim;
            for d in 0..kv_dim {
                sel_keys.push(keys_f16[base + d].to_f32());
                sel_values.push(values_f16[base + d].to_f32());
            }
        }

        (sel_keys, sel_values)
    }

    /// T23: Benchmark — StillKV synthesis vs selection (H2O-style) quality.
    ///
    /// Compares perceiver-based synthesis with top-k selection by L2 norm.
    #[test]
    fn bench_t23_synthesis_vs_selection() {
        let seq_len = 1024;
        let num_heads = 8;
        let head_dim = 64;
        let kv_dim = num_heads * head_dim;
        let chunk_size = 256;

        let (keys_f16, values_f16) = generate_synthetic_kv(seq_len, num_heads, head_dim);

        // Original first chunk as f32
        let keys_f32: Vec<f32> = keys_f16[..chunk_size * kv_dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();

        let ratios = [8usize, 16, 32];

        println!(
            "\n=== T23: Synthesis vs Selection ({} tokens × {} heads × {} dim) ===",
            seq_len, num_heads, head_dim
        );
        println!(
            "{:>10} | {:>12} | {:>10} | {:>10}",
            "Ratio", "Method", "CosSim(K)", "MSE(K)"
        );
        println!("{}", "-".repeat(60));

        for &ratio in &ratios {
            let budget = chunk_size / ratio;

            // --- Selection baseline ---
            let (sel_keys, _sel_values) =
                select_topk_by_norm(&keys_f16, &values_f16, chunk_size, kv_dim, budget);

            let sel_tokens = sel_keys.len() / kv_dim;
            let orig_mean = mean_across_tokens(&keys_f32, chunk_size, kv_dim);
            let sel_mean = mean_across_tokens(&sel_keys, sel_tokens, kv_dim);
            let sel_cos = cosine_similarity(&orig_mean, &sel_mean);
            let orig_prefix: Vec<f32> = keys_f32[..sel_tokens * kv_dim].to_vec();
            let sel_mse = compute_mse(&orig_prefix, &sel_keys);

            println!(
                "{:>10} | {:>12} | {:>10.4} | {:>10.6}",
                format!("{}x", ratio),
                "Selection",
                sel_cos,
                sel_mse
            );

            // --- Synthesis (StillKV) ---
            // Use ClusterCentroids as representative strategy
            let (syn_keys, _syn_values, elapsed) = run_compaction(
                &keys_f16,
                &values_f16,
                num_heads,
                head_dim,
                ratio,
                chunk_size,
                CompactionStrategy::ClusterCentroids,
            );

            let syn_tokens = syn_keys.len() / kv_dim;
            let syn_mean = mean_across_tokens(&syn_keys, syn_tokens, kv_dim);
            let syn_cos = cosine_similarity(&orig_mean, &syn_mean);
            let syn_prefix: Vec<f32> = keys_f32[..syn_tokens * kv_dim].to_vec();
            let syn_mse = compute_mse(&syn_prefix, &syn_keys);

            println!(
                "{:>10} | {:>12} | {:>10.4} | {:>10.6} | {:.2}ms",
                format!("{}x", ratio),
                "Synthesis",
                syn_cos,
                syn_mse,
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    /// T24: GOAT gate — measure compact-cache quality at 8x, 16x, 32x.
    ///
    /// Asserts minimum quality thresholds. If thresholds are not met with
    /// current heuristic initialization, the test documents actual results
    /// and the GOAT gate stays BLOCKED.
    #[test]
    fn goat_t24_compact_cache_quality() {
        let seq_len = 1024;
        let num_heads = 8;
        let head_dim = 64;
        let kv_dim = num_heads * head_dim;
        let chunk_size = 256;

        let (keys_f16, values_f16) = generate_synthetic_kv(seq_len, num_heads, head_dim);

        let keys_f32: Vec<f32> = keys_f16[..chunk_size * kv_dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let orig_mean = mean_across_tokens(&keys_f32, chunk_size, kv_dim);

        // GOAT thresholds: (compression_ratio, min_cosine_sim)
        let thresholds = [(8usize, 0.7f32), (16usize, 0.5f32), (32usize, 0.3f32)];

        println!(
            "\n=== T24: GOAT Gate — Compact Cache Quality ({} tokens × {} heads × {} dim) ===",
            seq_len, num_heads, head_dim
        );
        println!(
            "{:>4}x | {:>10} | {:>10} | {:>6}",
            "Ratio", "CosSim", "Threshold", "Status"
        );
        println!("{}", "-".repeat(50));

        // Use ClusterCentroids as the primary strategy for GOAT
        let strategy = CompactionStrategy::ClusterCentroids;
        let mut all_pass = true;

        for &(ratio, min_cos) in &thresholds {
            let (ck, _cv, elapsed) = run_compaction(
                &keys_f16,
                &values_f16,
                num_heads,
                head_dim,
                ratio,
                chunk_size,
                strategy,
            );

            let compact_tokens = ck.len() / kv_dim;
            let compact_mean = mean_across_tokens(&ck, compact_tokens, kv_dim);
            let cos_sim = cosine_similarity(&orig_mean, &compact_mean);
            let mse = compute_mse(&keys_f32[..compact_tokens * kv_dim], &ck);

            let pass = cos_sim >= min_cos;
            let status = if pass { "PASS" } else { "FAIL" };
            if !pass {
                all_pass = false;
            }

            println!(
                "{:>4}x | {:>10.4} | {:>10.4} | {:>6} | MSE={:.6} | {:.2}ms",
                ratio,
                cos_sim,
                min_cos,
                status,
                mse,
                elapsed.as_secs_f64() * 1000.0
            );
        }

        // Also test all strategies at 8x to find the GOAT
        println!("\n--- All strategies at 8x compression ---");
        println!("{:>25} | {:>10} | {:>10}", "Strategy", "CosSim", "MSE");
        println!("{}", "-".repeat(55));

        let all_strategies = [
            CompactionStrategy::ClusterCentroids,
            CompactionStrategy::AttentionWeighted,
            CompactionStrategy::SpectralProjection,
            CompactionStrategy::BfcfRegionBlend,
            CompactionStrategy::MuxSuperposition,
        ];

        let mut best_cos = f32::NEG_INFINITY;
        let mut best_strategy = all_strategies[0];

        for &s in &all_strategies {
            let (ck, _cv, elapsed) = run_compaction(
                &keys_f16,
                &values_f16,
                num_heads,
                head_dim,
                8,
                chunk_size,
                s,
            );

            let compact_tokens = ck.len() / kv_dim;
            let compact_mean = mean_across_tokens(&ck, compact_tokens, kv_dim);
            let cos_sim = cosine_similarity(&orig_mean, &compact_mean);
            let mse = compute_mse(&keys_f32[..compact_tokens * kv_dim], &ck);

            println!(
                "{:>25} | {:>10.4} | {:>10.6} | {:.2}ms",
                format!("{:?}", s),
                cos_sim,
                mse,
                elapsed.as_secs_f64() * 1000.0
            );

            if cos_sim > best_cos {
                best_cos = cos_sim;
                best_strategy = s;
            }
        }

        println!(
            "\nGOAT strategy at 8x: {:?} (cos_sim={:.4})",
            best_strategy, best_cos
        );

        // Final GOAT assertion
        assert!(
            all_pass,
            "GOAT gate BLOCKED: StillKV quality does not meet minimum thresholds. \
             Check output above for actual values. \
             To promote: improve query bank initialization or lower thresholds."
        );

        if all_pass {
            println!("\nGOAT gate PASSED: All quality thresholds met.");
        }
    }
}
