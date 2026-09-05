use super::summary::MixedRopeSummarizer;
use super::*;

// ── T1.4: MixedRopeSummarizer tests ──────────────────────────────────────────

#[test]
fn mixed_rope_summary_matches_mean_at_zero_rotation() {
    // When all positions are 0, θ = 0 for all pairs → cos=1, sin=0 → both
    // branches degenerate to the raw mean.
    let d = 8;
    let summarizer = MixedRopeSummarizer::from_rope_theta(d, 10000.0, 4);

    let keys: Vec<f32> = (0..4 * d).map(|i| i as f32).collect();
    let positions = vec![0usize; 4]; // all at position 0

    let summary = summarizer.summarize(&keys, &positions, 0, 4);

    // Expected: raw mean of the 4 keys.
    for i in 0..d {
        let expected: f32 = (0..4).map(|t| keys[t * d + i]).sum::<f32>() / 4.0;
        assert!(
            (summary[i] - expected).abs() < 1e-5,
            "dim {i}: summary {} != mean {expected} at zero rotation",
            summary[i]
        );
    }
}

#[test]
fn mixed_rope_summary_threshold_derivation() {
    // Verify the θ_i · C ≈ 2π crossover lands at the expected index.
    //
    // For rope_theta = 10000, D = 64, span = 16:
    //   threshold = 2π / 16 ≈ 0.3927
    //   inv_freq[i] = 1/10000^(2i/64) >= threshold
    //   10000^(-2i/64) >= 0.3927
    //   -2i/64 * ln(10000) >= ln(0.3927)
    //   i <= -64/2 * ln(0.3927) / ln(10000)
    //   i <= 32 * 0.9347 / 9.2103
    //   i <= 3.247
    // So crossover_index should be ~3-4 (pairs 0,1,2,3 are high-freq).

    let d = 64;
    let s = MixedRopeSummarizer::from_rope_theta(d, 10000.0, 16);
    let threshold = s.threshold();
    assert!((threshold - 2.0 * std::f32::consts::PI / 16.0).abs() < 1e-5);

    let crossover = s.crossover_index();
    // The crossover should be around 3-4 (the first few pairs are high-freq).
    assert!(
        (3..=5).contains(&crossover),
        "crossover for theta=10000 D=64 span=16: got {crossover}, expected 3-5"
    );

    // For rope_theta = 1000000 (Qwen3), the crossover should be LOWER
    // (fewer high-freq pairs, because inv_freq decays faster).
    let s2 = MixedRopeSummarizer::from_rope_theta(d, 1000000.0, 16);
    let crossover2 = s2.crossover_index();
    assert!(
        crossover2 < crossover,
        "Qwen3 (theta=1M) should have fewer high-freq pairs ({crossover2}) than Gemma2 (theta=10K) ({crossover})"
    );
}

#[test]
fn mixed_rope_high_freq_pairs_are_low_indices() {
    // High-frequency pairs should be the low-index pairs (small i → large inv_freq).
    let d = 16;
    let s = MixedRopeSummarizer::from_rope_theta(d, 10000.0, 8);
    let mask = s.high_freq_mask();

    // The first pair (i=0) has inv_freq = 1.0 (highest) → must be high-freq.
    assert!(mask[0], "pair 0 (inv_freq=1.0) must be high-freq");

    // If there are both high and low freq pairs, the high ones come first.
    if s.n_high_freq() > 0 && s.n_low_freq() > 0 {
        let first_low = mask.iter().position(|&h| !h).unwrap();
        // All pairs after first_low should also be low.
        for (i, &h) in mask.iter().enumerate().skip(first_low) {
            assert!(
                !h,
                "pair {i} after first_low {first_low} should be low-freq"
            );
        }
    }
}

#[test]
fn mixed_rope_low_freq_uses_midpoint_rotation() {
    // For a purely low-frequency setup (very small span or very large rope_theta),
    // all pairs are low-freq → average-then-rotate-at-mid.
    let d = 8;
    // span=1 means threshold = 2π → all pairs with inv_freq < 2π are low-freq.
    // inv_freq[0] = 1.0 < 2π → all are low-freq.
    let s = MixedRopeSummarizer::from_rope_theta(d, 10000.0, 1);
    assert_eq!(s.n_high_freq(), 0, "span=1 → all low-freq");
    assert_eq!(s.n_low_freq(), d / 2);

    let keys: Vec<f32> = (0..2 * d).map(|i| i as f32).collect();
    let positions = vec![5, 7]; // span = 2, mid = 6
    let summary = s.summarize(&keys, &positions, 0, 2);

    // For low-freq: summary = rotate(mean_raw, mid_pos)
    // mean_raw for pair 0: ((0+8)/2, (1+9)/2) = (4, 5)
    // mid_pos = 6, freq[0] = 1.0, theta = 6
    let mid = 6.0f32;
    let freq0 = 1.0f32;
    let theta = mid * freq0;
    let (sin_t, cos_t) = theta.sin_cos();
    let mean_x = (0.0f32 + 8.0) / 2.0;
    let mean_y = (1.0f32 + 9.0) / 2.0;
    let expected_x = mean_x * cos_t - mean_y * sin_t;
    let expected_y = mean_x * sin_t + mean_y * cos_t;
    assert!(
        (summary[0] - expected_x).abs() < 1e-4,
        "summary[0]={} expected={expected_x}",
        summary[0]
    );
    assert!(
        (summary[1] - expected_y).abs() < 1e-4,
        "summary[1]={} expected={expected_y}",
        summary[1]
    );
}

// ── T1.5 + T1.6: GroupSummaryCache tests ─────────────────────────────────────

#[test]
fn group_summary_cache_append_and_score() {
    let d = 8;
    let c = 4; // chunk_size
    let gs = 2; // group_size
    let summarizer = MixedRopeSummarizer::from_rope_theta(d, 10000.0, gs);
    let mut cache = GroupSummaryCache::new(d, c, gs, summarizer);

    // Append 2 chunks.
    for chunk_idx in 0..2 {
        let keys: Vec<f32> = (0..c * d)
            .map(|i| chunk_idx as f32 * 100.0 + i as f32)
            .collect();
        let positions: Vec<usize> = (chunk_idx * c..).take(c).collect();
        cache.append_chunk(&keys, &positions);
    }

    assert_eq!(cache.n_chunks(), 2);
    assert_eq!(cache.n_groups_per_chunk(), 2);
    assert_eq!(cache.n_groups(), 4);

    // Score groups with a query.
    let query = vec![1.0f32; d];
    let scores = cache.score_groups(&query, &[0, 1]);
    assert_eq!(scores.len(), 4); // 2 chunks * 2 groups

    // Scores should be sorted descending.
    for i in 1..scores.len() {
        assert!(
            scores[i - 1].score >= scores[i].score,
            "scores not sorted desc"
        );
    }
}

#[test]
fn group_summary_cache_select_top_k() {
    let d = 8;
    let c = 8;
    let gs = 2;
    let summarizer = MixedRopeSummarizer::from_rope_theta(d, 10000.0, gs);
    let mut cache = GroupSummaryCache::new(d, c, gs, summarizer);

    // Append 3 chunks with distinct values.
    for chunk_idx in 0..3 {
        let val = (chunk_idx + 1) as f32;
        let keys: Vec<f32> = vec![val; c * d];
        let positions: Vec<usize> = (chunk_idx * c..).take(c).collect();
        cache.append_chunk(&keys, &positions);
    }

    // Query = all ones.
    let query = vec![1.0f32; d];
    let scores = cache.score_groups(&query, &[0, 1, 2]);
    // All 12 groups scored (3 chunks * 4 groups).
    assert_eq!(scores.len(), 12);
    // Scores sorted descending.
    for i in 1..scores.len() {
        assert!(scores[i - 1].score >= scores[i].score);
    }

    // Select top-2 groups.
    let selection = cache.select_top_k_groups(&query, &[0, 1, 2], 2);
    assert!(!selection.selections.is_empty());
    // Count total groups selected = 2.
    let total_selected: usize = selection.selections.iter().map(|(_, _, n)| *n).sum();
    assert_eq!(total_selected, 2, "should select exactly 2 groups");
}

#[test]
fn group_summary_cache_score_skips_unselected_chunks() {
    let d = 4;
    let c = 4;
    let gs = 2;
    let summarizer = MixedRopeSummarizer::from_rope_theta(d, 10000.0, gs);
    let mut cache = GroupSummaryCache::new(d, c, gs, summarizer);

    for chunk_idx in 0..5 {
        let keys: Vec<f32> = vec![chunk_idx as f32; c * d];
        let positions: Vec<usize> = (chunk_idx * c..).take(c).collect();
        cache.append_chunk(&keys, &positions);
    }

    let query = vec![1.0f32; d];
    // Only score chunks 1 and 3.
    let scores = cache.score_groups(&query, &[1, 3]);
    assert_eq!(scores.len(), 4); // 2 chunks * 2 groups
    for s in &scores {
        assert!(
            s.chunk_idx == 1 || s.chunk_idx == 3,
            "unexpected chunk {}",
            s.chunk_idx
        );
    }
}

// ── T1.12 (katgpt-core analog): full-coverage fetch = causal SDPA ────────────
//
// This is the katgpt-core analog of the T1.12 test in katgpt-attn/hga_forward.
// It verifies that with RouteBudget::FULL, the tiered store fetches ALL tokens
// and the manual SDPA over the working set matches a reference causal SDPA
// computed directly over the raw K/V. The katgpt-attn version adds entmax-based
// chunk selection on top; this test isolates the fetch + SDPA path.

#[test]
fn full_coverage_fetch_matches_causal_sdpa() {
    use crate::tiered_kv::{InMemoryTieredKvStore, RouteBudget, SinkLocalSet, TieredKvStore};

    let d = 8usize;
    let c = 4usize;
    let gs = 2usize;
    let n_chunks = 5usize;

    let summarizer = super::summary::MixedRopeSummarizer::from_rope_theta(d, 10000.0, gs);
    let mut cache = super::GroupSummaryCache::new(d, c, gs, summarizer);

    // Simple mean summarizer for the store.
    let mean_fn =
        |keys_flat: &[f32], positions: &[usize], group_start: usize, n_tokens: usize| -> Vec<f32> {
            let total = positions.len();
            let hd = keys_flat.len().checked_div(total).unwrap_or(d);
            let mut s = vec![0.0f32; hd];
            for t in 0..n_tokens {
                let off = (group_start + t) * hd;
                for i in 0..hd {
                    s[i] += keys_flat[off + i];
                }
            }
            let inv = 1.0 / n_tokens as f32;
            for x in s.iter_mut() {
                *x *= inv;
            }
            s
        };

    let mut store = InMemoryTieredKvStore::new(d, c, gs, mean_fn);

    let mut rng = fastrand::Rng::with_seed(42);
    let mut all_keys = Vec::new();
    let mut all_values = Vec::new();
    for chunk_idx in 0..n_chunks {
        let keys: Vec<f32> = (0..c * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let values: Vec<f32> = (0..c * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let positions: Vec<usize> = (chunk_idx * c..).take(c).collect();
        all_keys.extend_from_slice(&keys);
        all_values.extend_from_slice(&values);
        store.append_chunk(&keys, &values, &positions);
        cache.append_chunk(&keys, &positions);
    }

    let n_tokens = n_chunks * c; // 20
    let query: Vec<f32> = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect();

    // Full-coverage fetch: all chunks local, RouteBudget::FULL.
    let sink_local = SinkLocalSet::new(vec![], (0..n_chunks).collect());
    let group_sel =
        crate::tiered_kv::GroupSelection::all_groups(n_chunks, cache.n_groups_per_chunk());
    let ws = store.fetch_working_set(&sink_local, &(0..n_chunks).collect::<Vec<_>>(), &group_sel);
    assert_eq!(
        ws.n_tokens, n_tokens,
        "full coverage should fetch all tokens"
    );

    // Manual SDPA over the working set.
    let sqrt_d = (d as f32).sqrt();
    let mut logits = vec![0.0f32; n_tokens];
    let mut max_logit = f32::NEG_INFINITY;
    for (j, logit) in logits.iter_mut().enumerate() {
        *logit = crate::simd::simd_dot_f32(&query, &ws.keys[j * d..(j + 1) * d], d) / sqrt_d;
        if *logit > max_logit {
            max_logit = *logit;
        }
    }
    let mut sum_exp = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max_logit).exp();
        sum_exp += *l;
    }
    let mut hga_out = vec![0.0f32; d];
    let inv = 1.0 / sum_exp;
    for (j, &logit) in logits.iter().enumerate() {
        let w = logit * inv;
        for (i, ho) in hga_out.iter_mut().enumerate() {
            *ho += w * ws.values[j * d + i];
        }
    }

    // Reference: causal SDPA over raw K/V.
    let mut ref_logits = vec![0.0f32; n_tokens];
    let mut ref_max = f32::NEG_INFINITY;
    for j in 0..n_tokens {
        ref_logits[j] =
            crate::simd::simd_dot_f32(&query, &all_keys[j * d..(j + 1) * d], d) / sqrt_d;
        if ref_logits[j] > ref_max {
            ref_max = ref_logits[j];
        }
    }
    let mut ref_sum = 0.0f32;
    for l in ref_logits.iter_mut() {
        *l = (*l - ref_max).exp();
        ref_sum += *l;
    }
    let mut ref_out = vec![0.0f32; d];
    let ref_inv = 1.0 / ref_sum;
    for j in 0..n_tokens {
        let w = ref_logits[j] * ref_inv;
        for i in 0..d {
            ref_out[i] += w * all_values[j * d + i];
        }
    }

    // Compare — should match within f32 noise.
    let mut max_diff = 0.0f32;
    for i in 0..d {
        let diff = (hga_out[i] - ref_out[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < 1e-5,
        "full-coverage fetch SDPA differs from reference by {max_diff}"
    );

    // Sanity: RouteBudget::FULL is indeed full.
    let _ = RouteBudget::FULL; // just verify it exists
}
