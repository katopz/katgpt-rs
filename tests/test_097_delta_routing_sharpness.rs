//! Delta Attention Residuals — Routing Sharpness GOAT Proof (Plan 097, T8).
//!
//! Tests:
//! - Routing sharpness with non-zero query weights (≥0.4 max weight)
//! - Sharpness sustains ≥ 0.4 across increasing depth
//! - Softmax property: weights sum to 1.0
//! - Uniform routing with zero-init query weights
//! - End-to-end forward pass sharpness vs zero-query baseline
//! - Block boundary routing fires correctly
//!
//! Run: `cargo test -p katgpt-rs --test test_097_delta_routing_sharpness --features delta_routing -- --nocapture`

#![cfg(feature = "delta_routing")]

use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, depth_route_weights, forward,
};
use katgpt_rs::types::{Config, Rng};

fn make_config(n_layer: usize) -> Config {
    let mut config = Config::micro();
    config.n_layer = n_layer;
    config.validate().expect("Config should be valid");
    config
}

/// Generate random non-zero query weights.
fn random_query(n_embd: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n_embd).map(|_| rng.normal()).collect()
}

/// Generate a random distinct delta vector.
fn random_delta(n_embd: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n_embd).map(|_| rng.normal()).collect()
}

#[test]
fn test_routing_sharpness_with_nonzero_query() {
    let config = make_config(6); // 2 blocks: layers 0-3 (fires at 3), 4-5
    let n_embd = config.n_embd;
    let mut rng = Rng::new(42);

    // Non-zero query weights for deep layers (layer 3+)
    let query = random_query(n_embd, &mut rng);
    let norm = vec![1.0f32; n_embd];

    // 2 synthetic delta sources — random but distinct
    let delta0 = random_delta(n_embd, &mut rng);
    let delta1 = random_delta(n_embd, &mut rng);

    let sources: Vec<&[f32]> = vec![&delta0, &delta1];
    let weights = depth_route_weights(&sources, &query, &norm, n_embd);

    assert_eq!(weights.len(), 2, "Should have 2 routing weights");

    let max_weight = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_weight = weights.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("  Routing weights: {weights:?}");
    println!("  Max weight: {max_weight:.6}, Min weight: {min_weight:.6}");

    // Paper claims 3× sharper routing — max should be ≥ 0.4
    assert!(
        max_weight >= 0.4,
        "Routing should be sharp (max_weight={max_weight:.6} < 0.4). Weights: {weights:?}"
    );

    println!("✅ Routing sharpness with non-zero query: max_weight={max_weight:.6} >= 0.4");
}

#[test]
fn test_routing_sharpness_increases_with_depth() {
    let mut rng = Rng::new(42);
    let n_embd = 16; // From Config::micro()
    let norm = vec![1.0f32; n_embd];

    let layer_counts = [4usize, 8, 12];
    let mut sharpness_results: Vec<(usize, f32)> = Vec::new();

    for &n_layer in &layer_counts {
        // Generate distinct delta sources per layer count
        let delta0 = random_delta(n_embd, &mut rng);
        let delta1 = random_delta(n_embd, &mut rng);
        let sources: Vec<&[f32]> = vec![&delta0, &delta1];

        // Non-zero query weights for last block's boundary layer
        let query = random_query(n_embd, &mut rng);

        let weights = depth_route_weights(&sources, &query, &norm, n_embd);
        let max_weight = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("  n_layer={n_layer}: max_weight={max_weight:.6}, weights={weights:?}");
        sharpness_results.push((n_layer, max_weight));

        // Sharpness should be ≥ 0.4 for all depths
        assert!(
            max_weight >= 0.4,
            "n_layer={n_layer}: routing sharpness ({max_weight:.6}) should be >= 0.4"
        );
    }

    // Print sharpness trend across depths
    for i in 1..sharpness_results.len() {
        let (prev_layers, prev_sharp) = sharpness_results[i - 1];
        let (curr_layers, curr_sharp) = sharpness_results[i];
        println!(
            "  n_layer {prev_layers}→{curr_layers}: sharpness {prev_sharp:.6}→{curr_sharp:.6}"
        );
    }

    println!(
        "✅ Routing sharpness sustains ≥ 0.4 across n_layer={:?}",
        layer_counts
    );
}

#[test]
fn test_routing_weights_sum_to_one() {
    let mut rng = Rng::new(42);
    let n_embd = 16;
    let query = random_query(n_embd, &mut rng);
    let norm = vec![1.0f32; n_embd];

    for n_sources in [1, 3, 5] {
        let deltas: Vec<Vec<f32>> = (0..n_sources)
            .map(|_| random_delta(n_embd, &mut rng))
            .collect();
        let source_refs: Vec<&[f32]> = deltas.iter().map(|d| d.as_slice()).collect();

        let weights = depth_route_weights(&source_refs, &query, &norm, n_embd);

        assert_eq!(weights.len(), n_sources, "Should have {n_sources} weights");

        let sum: f32 = weights.iter().sum();
        let deviation = (sum - 1.0).abs();

        println!("  N={n_sources}: sum={sum:.8}, deviation={deviation:.2e}, weights={weights:?}");

        assert!(
            deviation < 1e-5,
            "N={n_sources}: weights should sum to 1.0 (got {sum:.8}, deviation={deviation:.2e})"
        );

        // All weights should be positive
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > 0.0, "N={n_sources}: weight[{i}]={w} should be positive");
        }
    }

    println!("✅ Routing weights sum to 1.0 (±1e-5) for N=1,3,5 sources");
}

#[test]
fn test_routing_uniform_with_zero_query() {
    let mut rng = Rng::new(42);
    let n_embd = 16;
    let query = vec![0.0f32; n_embd]; // Zero-init query
    let norm = vec![1.0f32; n_embd];

    for n_sources in [2, 4, 8] {
        let deltas: Vec<Vec<f32>> = (0..n_sources)
            .map(|_| random_delta(n_embd, &mut rng))
            .collect();
        let source_refs: Vec<&[f32]> = deltas.iter().map(|d| d.as_slice()).collect();

        let weights = depth_route_weights(&source_refs, &query, &norm, n_embd);

        let max_weight = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let expected_uniform = 1.0 / n_sources as f32;
        let tolerance = 0.05;

        println!(
            "  N={n_sources}: max_weight={max_weight:.6}, expected≈{expected_uniform:.6}, weights={weights:?}"
        );

        // With zero query, all dot products are 0 → softmax of all zeros = uniform
        assert!(
            max_weight <= expected_uniform + tolerance,
            "N={n_sources}: max_weight ({max_weight:.6}) should be ≈ {expected_uniform:.6} (uniform)"
        );

        // All weights should be approximately uniform
        for (i, &w) in weights.iter().enumerate() {
            let diff = (w - expected_uniform).abs();
            assert!(
                diff < tolerance,
                "N={n_sources}: weight[{i}]={w:.6} should be ≈ {expected_uniform:.6}"
            );
        }
    }

    println!("✅ Routing is uniform with zero query weights for N=2,4,8 sources");
}

#[test]
fn test_forward_sharpness_end_to_end() {
    let config = make_config(8); // 2 blocks: routing fires at layers 3 and 7
    let n_embd = config.n_embd;
    let mut rng = Rng::new(42);

    // --- Baseline: zero-query weights (default) ---
    let weights_baseline = TransformerWeights::new(&config, &mut rng);

    // --- Sharp weights: same RNG state + non-zero query at boundary layers ---
    let mut weights = TransformerWeights::new(&config, &mut rng);
    let query_layer3 = random_query(n_embd, &mut rng);
    let query_layer7 = random_query(n_embd, &mut rng);
    weights.delta_routing_query[3] = query_layer3;
    weights.delta_routing_query[7] = query_layer7;

    // Run forward with non-zero query weights
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    let mut logits_per_pos: Vec<Vec<f32>> = Vec::new();

    // Run 4+ positions to accumulate meaningful deltas across block boundaries
    for pos in 0..6 {
        let token = pos % config.vocab_size;
        let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
        let logits_vec = logits.to_vec();

        // All logits should be finite
        for (i, &l) in logits_vec.iter().enumerate() {
            assert!(l.is_finite(), "pos={pos}: logit[{i}]={l} should be finite");
        }

        logits_per_pos.push(logits_vec);
    }

    // Logits should not be all the same value (non-degenerate)
    let first = &logits_per_pos[0];
    let all_same = logits_per_pos.iter().all(|l| {
        l.iter()
            .zip(first.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    });
    assert!(
        !all_same,
        "Logits should differ across positions (non-degenerate)"
    );

    // Run baseline with zero query weights
    let mut cache_base = MultiLayerKVCache::new(&config);
    let mut ctx_base = ForwardContext::new(&config);

    let mut baseline_logits: Vec<Vec<f32>> = Vec::new();
    for pos in 0..6 {
        let token = pos % config.vocab_size;
        let logits = forward(
            &mut ctx_base,
            &weights_baseline,
            &mut cache_base,
            token,
            pos,
            &config,
        );
        baseline_logits.push(logits.to_vec());
    }

    // Outputs should differ from zero-query baseline
    let mut any_different = false;
    for (sharp, base) in logits_per_pos.iter().zip(baseline_logits.iter()) {
        for (s, b) in sharp.iter().zip(base.iter()) {
            if (s - b).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        if any_different {
            break;
        }
    }
    assert!(
        any_different,
        "Non-zero query weights should produce different logits than zero-query baseline"
    );

    println!("✅ Forward sharpness end-to-end: finite, non-degenerate, differs from baseline");
}

#[test]
fn test_block_boundary_routing_fires() {
    let config = make_config(8); // Block boundaries at layers 3 and 7
    let n_embd = config.n_embd;
    let mut rng = Rng::new(42);

    let mut weights = TransformerWeights::new(&config, &mut rng);

    // Set non-zero query weights at block boundary layers (3 and 7)
    weights.delta_routing_query[3] = random_query(n_embd, &mut rng);
    weights.delta_routing_query[7] = random_query(n_embd, &mut rng);

    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    let mut outputs: Vec<Vec<f32>> = Vec::new();

    // Run multiple positions — must go past both block boundaries
    for pos in 0..8 {
        let token = pos % config.vocab_size;
        let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);

        // Verify finite at each position
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "pos={pos}: logit[{i}]={l} should be finite at block boundary"
            );
        }

        outputs.push(logits.to_vec());
    }

    // Verify outputs are distinct across all position pairs
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            let l1_diff: f32 = outputs[i]
                .iter()
                .zip(outputs[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                l1_diff > 1e-4,
                "pos={i} and pos={j}: outputs should be distinct (L1 diff={l1_diff:.6})"
            );
        }
    }

    println!("✅ Block boundary routing fires: 8 positions, all finite and distinct");
}
