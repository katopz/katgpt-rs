//! Delta Attention Residuals GOAT Proof (Plan 097).
//!
//! Tests:
//! - Zero-cost when feature is OFF (guaranteed by #[cfg])
//! - Routing produces valid outputs
//! - Deterministic results
//! - Multiple layer counts
//! - Weight initialization correctness
//!
//! Run: `cargo test -p katgpt-rs --test test_delta_routing --features delta_routing -- --nocapture`

#![cfg(feature = "delta_routing")]

use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use katgpt_rs::types::{Config, Rng};

fn make_multilayer_config(n_layer: usize) -> Config {
    let mut config = Config::micro();
    config.n_layer = n_layer;
    config.validate().expect("Config should be valid");
    config
}

#[test]
fn test_delta_routing_produces_valid_output() {
    let config = make_multilayer_config(6);
    let mut rng = Rng::new(42);

    let weights = TransformerWeights::new(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    assert_eq!(
        logits.len(),
        config.vocab_size,
        "Logits length should match vocab_size"
    );
    assert!(
        logits.iter().all(|&l| l.is_finite()),
        "All logits should be finite"
    );

    println!("✅ Delta routing produces valid output at n_layer=6");
}

#[test]
fn test_delta_routing_deterministic() {
    let config = make_multilayer_config(6);

    let mut rng1 = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng1);

    // Run 1
    let mut cache1 = MultiLayerKVCache::new(&config);
    let mut ctx1 = ForwardContext::new(&config);
    let logits1 = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();

    // Run 2 (same weights, fresh context)
    let mut cache2 = MultiLayerKVCache::new(&config);
    let mut ctx2 = ForwardContext::new(&config);
    let logits2 = forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config).to_vec();

    for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "Logit {i} differs: {a} vs {b}");
    }

    println!("✅ Delta routing is deterministic");
}

#[test]
fn test_delta_routing_multiple_layers() {
    for n_layer in [1, 2, 4, 6, 8] {
        let config = make_multilayer_config(n_layer);
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut ctx = ForwardContext::new(&config);

        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

        assert_eq!(logits.len(), config.vocab_size);
        assert!(
            logits.iter().all(|&l| l.is_finite()),
            "n_layer={n_layer}: all logits should be finite"
        );

        println!("  ✅ n_layer={n_layer} passed");
    }

    println!("✅ Delta routing works across multiple layer counts");
}

#[test]
fn test_delta_routing_weights_zero_init() {
    let config = make_multilayer_config(6);
    let mut rng = Rng::new(42);

    let weights = TransformerWeights::new(&config, &mut rng);

    // Delta routing query weights should be zero-initialized (safe additive start)
    for (layer_idx, query) in weights.delta_routing_query.iter().enumerate() {
        for (d, &val) in query.iter().enumerate() {
            assert!(
                val == 0.0,
                "Layer {layer_idx} dim {d}: query should be zero-init, got {val}"
            );
        }
    }

    // Delta routing norm weights should be one-initialized (identity RMSNorm)
    for (layer_idx, norm) in weights.delta_routing_norm.iter().enumerate() {
        for (d, &val) in norm.iter().enumerate() {
            assert!(
                val == 1.0,
                "Layer {layer_idx} dim {d}: norm should be one-init, got {val}"
            );
        }
    }

    println!("✅ Delta routing weights correctly initialized (query=0, norm=1)");
}

#[test]
fn test_delta_routing_block_boundaries() {
    // With n_layer=8 and block_size=4, routing fires at layers 3 and 7
    // Verify no panic and valid output at exact block boundary
    let config = make_multilayer_config(8);
    let mut rng = Rng::new(42);

    let weights = TransformerWeights::new(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    // Run multi-position forward to exercise block boundaries
    for pos in 0..4 {
        let token = pos % config.vocab_size;
        let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
        assert!(
            logits.iter().all(|&l| l.is_finite()),
            "pos={pos}: all logits should be finite"
        );
    }

    println!("✅ Delta routing block boundaries work at n_layer=8, 4 positions");
}

#[test]
fn test_delta_routing_non_block_aligned() {
    // n_layer=5 with block_size=4: only layer 3 fires routing, layer 4 is incomplete block
    let config = make_multilayer_config(5);
    let mut rng = Rng::new(42);

    let weights = TransformerWeights::new(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|&l| l.is_finite()),
        "All logits should be finite for non-block-aligned n_layer"
    );

    println!("✅ Delta routing handles non-block-aligned n_layer=5");
}
