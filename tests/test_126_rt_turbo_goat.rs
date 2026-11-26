#![cfg(feature = "rt_turbo")]

//! GOAT proofs for RTPurbo — Retrieval Head Sparse Decode (Plan 126 Phase 5).
//!
//! 6 proofs required for default-on promotion:
//!
//! | Proof | Test | Claim |
//! |-------|------|-------|
//! | T21 | Calibration stability | Partition identical across 1-seq vs 10-seq calibration |
//! | T22 | Top-p vs top-k | top-p ≥ 90% mass with fewer tokens than top-k=4096 |
//! | T23 | Low-dim recall | 16-dim projection ≥ 85% overlap with full-dim top-256 |
//! | T24 | Decode routing efficiency | Retrieval heads attend to < seq_len tokens; total FLOPs < uniform |
//! | T25 | Accuracy preservation | Sparse attention cosine similarity > 0.99 vs dense |
//! | T26 | Compatibility | No panics / NaN across edge configurations |

use std::collections::HashSet;

use katgpt_rs::types::RtTurboConfig;
use katgpt_speculative::rt_turbo::*;

// ── Deterministic PRNG (no `rand` dependency) ──────────────────

/// Xorshift64 PRNG for reproducible synthetic data.
struct SeedRng {
    state: u64,
}

impl SeedRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns f32 in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        let bits = ((self.next_u64() >> 41) as u32) | 0x3F80_0000;
        f32::from_bits(bits) - 1.0
    }

    /// Returns f32 in [lo, hi).
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ── Attention Matrix Helpers ───────────────────────────────────

/// Create seq_len × seq_len attention where post-needle rows attend strongly
/// to pre-needle columns (retrieval behavior).
fn make_retrieval_attn(seq_len: usize, needle_len: usize, strength: f32) -> Vec<f32> {
    let mut attn = vec![1.0f32 / seq_len as f32; seq_len * seq_len];
    let post_start = seq_len - needle_len;
    for t in post_start..seq_len {
        let row_off = t * seq_len;
        let pre_mass = strength / needle_len as f32;
        let remain_mass = (1.0 - strength) / (seq_len - needle_len) as f32;
        for j in 0..needle_len {
            attn[row_off + j] = pre_mass;
        }
        for j in needle_len..seq_len {
            attn[row_off + j] = remain_mass;
        }
    }
    attn
}

/// Create seq_len × seq_len attention with local window (diagonal-dominant).
fn make_local_attn(seq_len: usize, window: usize) -> Vec<f32> {
    let mut attn = vec![0.0f32; seq_len * seq_len];
    for t in 0..seq_len {
        let row_off = t * seq_len;
        let start = t.saturating_sub(window);
        let count = t - start + 1;
        let val = 1.0 / count as f32;
        for j in start..=t {
            attn[row_off + j] = val;
        }
    }
    attn
}

// ── Calibration Helper ─────────────────────────────────────────

/// Create calibration where first `n_retrieval` heads are classified retrieval.
///
/// Assigns decreasing scores so heads 0..n_retrieval always win.
fn make_calibration(n_heads: usize, n_retrieval: usize, config: &RtTurboConfig) -> HeadCalibration {
    let mut scores = vec![0.0f32; n_heads];
    for (i, s) in scores.iter_mut().enumerate().take(n_retrieval.min(n_heads)) {
        *s = 1.0 - i as f32 * 0.01;
    }
    calibrate_from_scores(&scores, config)
}

// ── Math Helpers ───────────────────────────────────────────────

/// Numerically stable softmax.
fn inline_softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![0.0; scores.len()]
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Select top-K indices by score value (descending).
fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Compute softmax mass captured by given indices (relative to all scores).
fn softmax_mass_at(scores: &[f32], indices: &[usize]) -> f32 {
    if scores.is_empty() || indices.is_empty() {
        return 0.0;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let all_exp: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
    let total: f32 = all_exp.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let selected: f32 = indices.iter().map(|&i| all_exp[i]).sum();
    selected / total
}

/// Assert all values in slice are finite (no NaN, no Inf).
fn assert_no_nan(slice: &[f32], label: &str) {
    for (i, &v) in slice.iter().enumerate() {
        assert!(v.is_finite(), "{label}[{i}] is not finite: {v}");
    }
}

// ════════════════════════════════════════════════════════════════
// T21: Proof 1 — Calibration Stability
// ════════════════════════════════════════════════════════════════

/// Prove head partition is input-agnostic: single-sequence calibration vs
/// 10-sequence averaged calibration produces identical partition (±0 heads).
///
/// Uses 32 heads with 5 truly-retrieval heads (15% ratio). Three seeds
/// produce different attention patterns, but the ranking is stable because
/// retrieval heads consistently score 5–10× higher than local heads.
#[test]
fn test_goat_1_calibration_stability() {
    let n_heads = 32;
    let seq_len = 128;
    let needle_len = 16;
    let needle_start = 0;
    let needle_end = needle_len;
    let post_needle_start = seq_len - needle_len;
    let post_needle_end = seq_len;

    let config = RtTurboConfig {
        retrieval_head_ratio: 0.15, // → ceil(32 × 0.15) = 5 retrieval heads
        ..RtTurboConfig::default()
    };

    // First 5 heads are "truly retrieval" with high strength
    let n_truly_retrieval = 5;

    for seed in [42u64, 12345, 99999] {
        // ── Single-sequence calibration ────────────────────────
        let mut rng1 = SeedRng::new(seed);
        let single_scores: Vec<f32> = (0..n_heads)
            .map(|h| {
                let attn = if h < n_truly_retrieval {
                    let strength = 0.7 + 0.04 * h as f32 + rng1.next_f32_range(-0.02, 0.02);
                    make_retrieval_attn(seq_len, needle_len, strength)
                } else {
                    make_local_attn(seq_len, 4)
                };
                compute_retrieval_score(
                    &attn,
                    seq_len,
                    needle_start,
                    needle_end,
                    post_needle_start,
                    post_needle_end,
                )
            })
            .collect();

        let single_cal = calibrate_from_scores(&single_scores, &config);

        // ── 10-sequence calibration (averaged) ─────────────────
        let mut rng2 = SeedRng::new(seed.wrapping_add(1000));
        let multi_scores: Vec<f32> = (0..n_heads)
            .map(|h| {
                let mut total = 0.0f32;
                for _ in 0..10 {
                    let attn = if h < n_truly_retrieval {
                        let strength = 0.7 + 0.04 * h as f32 + rng2.next_f32_range(-0.02, 0.02);
                        make_retrieval_attn(seq_len, needle_len, strength)
                    } else {
                        make_local_attn(seq_len, 4)
                    };
                    total += compute_retrieval_score(
                        &attn,
                        seq_len,
                        needle_start,
                        needle_end,
                        post_needle_start,
                        post_needle_end,
                    );
                }
                total / 10.0
            })
            .collect();

        let multi_cal = calibrate_from_scores(&multi_scores, &config);

        // ── Assert identical partition (±0 difference) ─────────
        assert_eq!(
            single_cal.retrieval_set, multi_cal.retrieval_set,
            "Seed {seed}: retrieval_set mismatch — single={:?} multi={:?}",
            single_cal.retrieval_set, multi_cal.retrieval_set,
        );
        assert_eq!(
            single_cal.local_set, multi_cal.local_set,
            "Seed {seed}: local_set mismatch",
        );
    }
}

// ════════════════════════════════════════════════════════════════
// T22: Proof 2 — Top-P vs Top-K
// ════════════════════════════════════════════════════════════════

/// Prove dynamic top-p achieves ≥90% attention mass recall while selecting
/// fewer tokens than a naive top-k=4096 baseline.
///
/// Uses power-law synthetic scores: 100 "spike" positions with high values,
/// 9900 low-value positions. This mimics real long-context attention where
/// a handful of positions carry most of the mass.
#[test]
fn test_goat_2_top_p_better_mass_recall() {
    let seq_len = 10_000;

    // Generate power-law scores: 100 spikes + 9900 low values
    let mut scores = vec![0.0f32; seq_len];
    let mut rng = SeedRng::new(42);

    // 100 spike positions spread deterministically across the sequence
    for i in 0..100 {
        let pos = (i * 97 + 13) % seq_len;
        scores[pos] = 5.0 + rng.next_f32() * 2.0; // [5.0, 7.0]
    }
    // Fill remaining with low scores
    for s in scores.iter_mut() {
        if *s == 0.0 {
            *s = rng.next_f32_range(-0.5, 0.5);
        }
    }

    // ── Top-p selection (p=0.9) ────────────────────────────────
    let top_p_result = select_top_p(&scores, 0.9);

    // ── Top-k selection (k=4096) ───────────────────────────────
    let top_k_idx = top_k_indices(&scores, 4096);
    let top_k_mass = softmax_mass_at(&scores, &top_k_idx);

    // ── Assertions ─────────────────────────────────────────────
    assert!(
        top_p_result.cumulative_mass >= 0.9,
        "top_p mass {:.4} < 0.9",
        top_p_result.cumulative_mass,
    );

    assert!(
        top_p_result.selected_indices.len() < top_k_idx.len(),
        "top_p selected {} >= top_k selected {} — should use fewer",
        top_p_result.selected_indices.len(),
        top_k_idx.len(),
    );

    // Sanity: top_p mass should be competitive (within 10% of top_k)
    assert!(
        top_p_result.cumulative_mass >= top_k_mass - 0.10,
        "top_p mass {:.4} much less than top_k mass {:.4}",
        top_p_result.cumulative_mass,
        top_k_mass,
    );
}

// ════════════════════════════════════════════════════════════════
// T23: Proof 3 — Low-Dim Recall
// ════════════════════════════════════════════════════════════════

/// Prove 16-dim identity projection preserves ≥85% overlap with full-dim
/// top-256 token indices across 100 random query/key pairs.
///
/// Generates vectors where the first 16 dimensions are dominant (10× larger
/// magnitude range than remaining 48 dimensions), matching the RTPurbo
/// finding that low-frequency pre-RoPE components drive retrieval.
#[test]
fn test_goat_3_low_dim_recall() {
    let head_dim = 64;
    let low_dim = 16;
    let n_keys = 1000;
    let top_k = 256;
    let n_trials = 100;

    let projection = RetrievalProjection::identity(1, head_dim, low_dim);

    let mut overlaps = Vec::with_capacity(n_trials);
    let mut rng = SeedRng::new(42);

    for _ in 0..n_trials {
        // Query: first 16 dims dominant
        let query: Vec<f32> = (0..head_dim)
            .map(|d| {
                if d < low_dim {
                    rng.next_f32_range(-2.0, 2.0)
                } else {
                    rng.next_f32_range(-0.2, 0.2)
                }
            })
            .collect();

        // Key matrix: first 16 dims dominant
        let mut key_matrix = vec![0.0f32; n_keys * head_dim];
        for pos in 0..n_keys {
            for d in 0..head_dim {
                let val = if d < low_dim {
                    rng.next_f32_range(-2.0, 2.0)
                } else {
                    rng.next_f32_range(-0.2, 0.2)
                };
                key_matrix[pos * head_dim + d] = val;
            }
        }

        // Full-dim scores: dot product of all dims
        let full_scores: Vec<f32> = (0..n_keys)
            .map(|pos| {
                let off = pos * head_dim;
                (0..head_dim).map(|d| query[d] * key_matrix[off + d]).sum()
            })
            .collect();

        // Low-dim scores via identity projection
        let low_scores = projection.batch_project_scores(0, &query, &key_matrix);

        // Top-256 from each
        let full_top: HashSet<usize> = top_k_indices(&full_scores, top_k).into_iter().collect();
        let low_top: HashSet<usize> = top_k_indices(&low_scores, top_k).into_iter().collect();

        let overlap = full_top.intersection(&low_top).count();
        overlaps.push(overlap as f32 / top_k as f32);
    }

    let mean_overlap: f32 = overlaps.iter().sum::<f32>() / overlaps.len() as f32;

    assert!(
        mean_overlap >= 0.85,
        "Mean low-dim recall {mean_overlap:.4} < 0.85 (85%)",
    );
}

// ════════════════════════════════════════════════════════════════
// T24: Proof 4 — Decode Routing Efficiency (Structural)
// ════════════════════════════════════════════════════════════════

/// Prove decode-phase routing structurally reduces attended tokens:
/// - Local heads: sliding_window + sink_tokens only
/// - Retrieval heads: top-p selected tokens (should be << seq_len)
/// - Total FLOPs (proportional to attended tokens) < uniform decode
///
/// Uses 8 heads, 2 retrieval, 6 local at seq_len=16384 with peaked
/// score distributions for retrieval heads.
#[test]
fn test_goat_4_decode_routing_efficiency() {
    let n_heads = 8;
    let head_dim = 64;
    let seq_len = 16384;

    let config = RtTurboConfig {
        retrieval_head_ratio: 0.25, // → ceil(8 × 0.25) = 2 retrieval heads
        low_dim: 16,
        top_p: 0.9,
        sliding_window: 8192,
        sink_tokens: 4,
        block_size: 64,
        ..RtTurboConfig::default()
    };

    let calibration = make_calibration(n_heads, 2, &config);
    assert_eq!(calibration.n_retrieval(), 2, "Expected 2 retrieval heads");
    assert_eq!(calibration.n_local(), 6, "Expected 6 local heads");

    let projection = RetrievalProjection::identity(2, head_dim, config.low_dim);

    // ── Peaked key_pre_rope: spikes every 1024 positions for retrieval heads ──
    let total_dim = n_heads * head_dim;
    let mut key_pre_rope = vec![vec![0.01f32; total_dim]; seq_len];
    for &global_head in calibration.retrieval_set[..calibration.n_retrieval()].iter() {
        let offset = global_head * head_dim;
        // Create 16 spike positions (every 1024 steps)
        for spike_idx in 0..16 {
            let pos = spike_idx * 1024;
            for d in 0..config.low_dim {
                key_pre_rope[pos][offset + d] = 1.0;
            }
        }
    }

    // ── Query: retrieval heads have strong first-16 dims ───────
    let query: Vec<Vec<f32>> = (0..n_heads)
        .map(|h| {
            if calibration.is_retrieval(h) {
                let mut q = vec![0.0f32; head_dim];
                for v in q.iter_mut().take(config.low_dim) {
                    *v = 1.0;
                }
                q
            } else {
                vec![0.5; head_dim]
            }
        })
        .collect();

    // ── KV cache (seq_len determines positions) ────────────────
    let kv_dim = 2 * n_heads * head_dim;
    let kv_cache: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![0.0; kv_dim]).collect();

    let result = forward_rt_turbo_decode(
        &calibration,
        &projection,
        &config,
        &kv_cache,
        &query,
        &key_pre_rope,
    );

    // ── Verify local window ────────────────────────────────────
    let local_window_size = result.local_window.1 - result.local_window.0;
    assert_eq!(
        local_window_size, config.sliding_window,
        "Local window should be {}",
        config.sliding_window,
    );

    // ── Verify retrieval heads are sparsified ──────────────────
    for (r, indices) in result.selected_indices.iter().enumerate() {
        assert!(
            indices.len() < seq_len,
            "Retrieval head {r} selected {} >= {seq_len} — not sparsified",
            indices.len(),
        );
    }

    // ── Verify total FLOPs reduction ───────────────────────────
    let uniform_flops = n_heads * seq_len;
    let local_per_head = local_window_size + result.sink_indices.len();
    let retrieval_total: usize = result.selected_indices.iter().map(|v| v.len()).sum();
    let rt_flops = calibration.n_local() * local_per_head + retrieval_total;

    assert!(
        rt_flops < uniform_flops,
        "RTPurbo FLOPs ({rt_flops}) >= uniform ({uniform_flops}) — no improvement",
    );

    // Verify sinks are included in retrieval head selections
    for (r, indices) in result.selected_indices.iter().enumerate() {
        for &sink in &result.sink_indices {
            assert!(
                indices.contains(&sink),
                "Retrieval head {r} missing sink {sink}",
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════
// T25: Proof 5 — Accuracy Preservation (Cosine Proxy)
// ════════════════════════════════════════════════════════════════

/// Prove sparse attention (via top-p selection) preserves output quality:
/// cosine_similarity(dense_output, sparse_output) > 0.99 at per-head level.
///
/// Uses peaked synthetic data where 10 "hot" key positions carry most of
/// the attention mass. select_top_p captures these positions, making the
/// sparse output nearly identical to dense.
#[test]
fn test_goat_5_accuracy_preservation() {
    let head_dim = 64;
    let seq_len = 256;
    let top_p = 0.9;

    // Deterministic peaked pattern: hot positions have large keys
    let hot_positions: [usize; 10] = [10, 30, 50, 80, 120, 140, 170, 200, 220, 240];

    let mut rng = SeedRng::new(42);

    // Query: all-positive for clear signal direction
    let query: Vec<f32> = (0..head_dim)
        .map(|_| rng.next_f32_range(0.5, 1.5))
        .collect();

    // Keys: hot positions have large values, others small
    let keys: Vec<Vec<f32>> = (0..seq_len)
        .map(|pos| {
            if hot_positions.contains(&pos) {
                (0..head_dim)
                    .map(|_| rng.next_f32_range(1.0, 3.0))
                    .collect()
            } else {
                (0..head_dim)
                    .map(|_| rng.next_f32_range(-0.05, 0.05))
                    .collect()
            }
        })
        .collect();

    // Values: hot positions have larger signal
    let values: Vec<Vec<f32>> = (0..seq_len)
        .map(|pos| {
            if hot_positions.contains(&pos) {
                (0..head_dim)
                    .map(|_| rng.next_f32_range(-2.0, 2.0))
                    .collect()
            } else {
                (0..head_dim)
                    .map(|_| rng.next_f32_range(-0.1, 0.1))
                    .collect()
            }
        })
        .collect();

    // ── Compute attention scores ───────────────────────────────
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores: Vec<f32> = keys
        .iter()
        .map(|k| {
            query
                .iter()
                .zip(k.iter())
                .map(|(q, ki)| q * ki)
                .sum::<f32>()
                * scale
        })
        .collect();

    // ── Dense attention output ─────────────────────────────────
    let probs = inline_softmax(&scores);
    let dense_output: Vec<f32> = (0..head_dim)
        .map(|d| {
            probs
                .iter()
                .enumerate()
                .map(|(j, &p)| p * values[j][d])
                .sum()
        })
        .collect();

    // ── Sparse attention output (top-p selected) ───────────────
    let top_p_result = select_top_p(&scores, top_p);

    let sparse_output: Vec<f32> = (0..head_dim)
        .map(|d| {
            top_p_result
                .selected_indices
                .iter()
                .zip(top_p_result.selected_probs.iter())
                .map(|(&idx, &prob)| prob * values[idx][d])
                .sum()
        })
        .collect();

    // ── Assert cosine similarity > 0.99 ───────────────────────
    let cos_sim = cosine_similarity(&dense_output, &sparse_output);

    assert!(
        cos_sim > 0.99,
        "Cosine similarity {cos_sim:.6} <= 0.99 — accuracy not preserved",
    );

    // Verify top-p captured the hot positions
    let hot_set: HashSet<usize> = hot_positions.iter().copied().collect();
    let selected_set: HashSet<usize> = top_p_result.selected_indices.iter().copied().collect();
    let hot_captured = hot_set.intersection(&selected_set).count();
    assert!(
        hot_captured >= 8,
        "Only {hot_captured}/10 hot positions captured by top-p",
    );
}

// ════════════════════════════════════════════════════════════════
// T26: Proof 6 — Compatibility (No Panics, No NaN)
// ════════════════════════════════════════════════════════════════

/// Prove the RTPurbo pipeline handles edge configurations without panics
/// or NaN: extreme head counts, short/long sequences, boundary top-p values,
/// and varied block sizes.
#[test]
fn test_goat_6_compatibility_no_panics() {
    // ── Sub-test A: 1 head, 1 retrieval ────────────────────────
    {
        let config = RtTurboConfig {
            retrieval_head_ratio: 1.0,
            low_dim: 4,
            ..RtTurboConfig::default()
        };
        let cal = make_calibration(1, 1, &config);
        let proj = RetrievalProjection::identity(1, 16, config.low_dim);
        let kv: Vec<Vec<f32>> = (0..32).map(|_| vec![0.0; 32]).collect();
        let query = vec![vec![0.5f32; 16]; 1];
        let kpr: Vec<Vec<f32>> = (0..32).map(|p| vec![p as f32 * 0.01; 16]).collect();

        let result = forward_rt_turbo_decode(&cal, &proj, &config, &kv, &query, &kpr);
        assert_eq!(result.n_retrieval_heads, 1);
        assert_eq!(result.n_local_heads, 0);
        for indices in &result.selected_indices {
            for &idx in indices {
                assert!(idx < 32, "Index {idx} out of bounds");
            }
        }
    }

    // ── Sub-test B: 4 heads, 0 retrieval (all_local) ───────────
    {
        let config = RtTurboConfig::default();
        let cal = HeadCalibration::all_local(4, &config);
        assert_eq!(cal.n_retrieval(), 0);
        let proj = RetrievalProjection::identity(0, 16, config.low_dim);
        let kv: Vec<Vec<f32>> = (0..32).map(|_| vec![0.0; 128]).collect();
        let query = vec![vec![0.5f32; 16]; 4];
        let kpr: Vec<Vec<f32>> = (0..32).map(|p| vec![p as f32 * 0.01; 64]).collect();

        let result = forward_rt_turbo_decode(&cal, &proj, &config, &kv, &query, &kpr);
        assert_eq!(result.n_retrieval_heads, 0);
        assert!(result.selected_indices.is_empty());
    }

    // ── Sub-test C: 128 heads ──────────────────────────────────
    {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.15, // → ceil(128 × 0.15) = 20
            low_dim: 4,
            ..RtTurboConfig::default()
        };
        let cal = make_calibration(128, 20, &config);
        assert_eq!(cal.n_retrieval(), 20);
        let proj = RetrievalProjection::identity(20, 16, config.low_dim);
        let kv: Vec<Vec<f32>> = (0..64).map(|_| vec![0.0; 2 * 128 * 16]).collect();
        let query = vec![vec![0.5f32; 16]; 128];
        let kpr: Vec<Vec<f32>> = (0..64).map(|p| vec![p as f32 * 0.001; 128 * 16]).collect();

        let result = forward_rt_turbo_decode(&cal, &proj, &config, &kv, &query, &kpr);
        assert_eq!(result.n_retrieval_heads, 20);
        assert_eq!(result.n_local_heads, 108);
    }

    // ── Sub-test D: Very short seq_len (1, 2, 4) ───────────────
    for seq_len in [1_usize, 2, 4] {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.5,
            low_dim: 4,
            sliding_window: 8,
            sink_tokens: 2,
            block_size: 2,
            ..RtTurboConfig::default()
        };
        let n_heads = 4;
        let head_dim = 16;
        let n_retrieval = 2;
        let cal = make_calibration(n_heads, n_retrieval, &config);
        let proj = RetrievalProjection::identity(n_retrieval, head_dim, config.low_dim);

        let kv: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| vec![0.0; 2 * n_heads * head_dim])
            .collect();
        let query = vec![vec![0.5f32; head_dim]; n_heads];
        let kpr: Vec<Vec<f32>> = (0..seq_len)
            .map(|p| vec![p as f32 * 0.01; n_heads * head_dim])
            .collect();

        let result = forward_rt_turbo_decode(&cal, &proj, &config, &kv, &query, &kpr);

        // Verify shapes
        assert_eq!(result.selected_indices.len(), n_retrieval);
        assert_eq!(result.sink_indices.len(), config.sink_tokens.min(seq_len));
        for indices in &result.selected_indices {
            for &idx in indices {
                assert!(
                    idx < seq_len,
                    "seq_len={seq_len}: index {idx} out of bounds"
                );
            }
        }
    }

    // ── Sub-test E: select_top_p with long input (100000 scores) ──
    {
        let mut rng = SeedRng::new(77);
        let scores: Vec<f32> = (0..100_000)
            .map(|i| {
                if i % 1000 == 0 {
                    5.0
                } else {
                    rng.next_f32_range(-0.5, 0.5)
                }
            })
            .collect();

        let result = select_top_p(&scores, 0.9);
        assert!(result.cumulative_mass >= 0.9, "Long input mass check");
        assert!(
            result.selected_indices.len() < 100_000,
            "Should select fewer than all tokens",
        );
        assert_no_nan(&result.selected_probs, "long_top_p_probs");
    }

    // ── Sub-test F: top_p = 0.0 (minimum selection) ────────────
    {
        let scores: Vec<f32> = vec![3.0, 1.0, 0.5, 0.2, 0.1];
        let result = select_top_p(&scores, 0.0);
        assert!(
            !result.selected_indices.is_empty(),
            "top_p=0.0 should select at least 1 token",
        );
        assert_no_nan(&result.selected_probs, "top_p_zero_probs");
    }

    // ── Sub-test G: top_p = 1.0 (maximum selection) ────────────
    {
        let scores: Vec<f32> = vec![3.0, 1.0, 0.5, 0.2, 0.1];
        let result = select_top_p(&scores, 1.0);
        assert_eq!(
            result.selected_indices.len(),
            scores.len(),
            "top_p=1.0 should select all tokens",
        );
        assert_no_nan(&result.selected_probs, "top_p_one_probs");
    }

    // ── Sub-test H: Various block sizes ────────────────────────
    {
        let mut rng = SeedRng::new(88);
        let scores: Vec<f32> = (0..1024)
            .map(|i| {
                if i % 64 == 0 {
                    5.0
                } else {
                    rng.next_f32() * 0.5
                }
            })
            .collect();

        for block_size in [1_usize, 32, 64, 128, 256] {
            let result = select_top_p_blockwise(&scores, 0.9, block_size);
            assert!(
                result.cumulative_mass > 0.0,
                "block_size={block_size}: zero mass",
            );
            assert_no_nan(
                &result.selected_probs,
                &format!("block_size_{block_size}_probs"),
            );
            for &idx in &result.selected_indices {
                assert!(
                    idx < 1024,
                    "block_size={block_size}: index {idx} out of bounds"
                );
            }
        }
    }

    // ── Sub-test I: block_size >= seq_len (falls back to fine-grained) ──
    {
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let result_fine = select_top_p(&scores, 0.9);
        let result_block = select_top_p_blockwise(&scores, 0.9, 1000);
        // Block size >= seq_len should behave like fine-grained
        assert_eq!(
            result_fine.selected_indices.len(),
            result_block.selected_indices.len(),
            "Large block_size should fall back to fine-grained",
        );
    }

    // ── Sub-test J: Prefill compatibility ──────────────────────
    {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.5,
            low_dim: 4,
            ..RtTurboConfig::default()
        };
        let cal = make_calibration(8, 4, &config);
        let proj = RetrievalProjection::identity(4, 16, config.low_dim);

        let result = forward_rt_turbo_prefill(&cal, &proj);
        assert_eq!(result.n_total_heads, 8);
        assert_eq!(result.dense_heads.len(), 8);
    }
}
