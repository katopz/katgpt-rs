//! SP-KV + Quantized KV Cache Fusion tests.
//! Plan 070 Phase 3, Task T12 — forward_sp_kv_quant validation.
//!
//! Tests:
//! 1. Fusion produces finite outputs across 8 decode steps
//! 2. Density tracking: open gates → ~100% retention
//! 3. Reset clears all state correctly
//! 4. Hard gate bias correctly masks non-retained positions
//! 5. Soft gating produces finite outputs (training mode)
//! 6. Multi-step consistency: logits stay in reasonable range
//!
//! Run with: cargo test --features "sp_kv turboquant" bench_sp_kv_quant -- --nocapture

#![cfg(all(feature = "sp_kv", feature = "turboquant"))]

use std::time::Instant;

use katgpt_rs::sp_kv::forward::forward_sp_kv_quant;
use katgpt_rs::sp_kv::{
    GateBiasBuffer, SpKvConfig, SpKvForwardContext, SpKvGateMode, SpKvPredictors, SpKvQuantCache,
};
use katgpt_rs::transformer::{ForwardContext, TransformerWeights};
use katgpt_rs::turboquant::TurboQuantKVCache;
use katgpt_rs::types::{Config, Rng, kv_dim};

/// Number of iterations for timing benchmarks.
const BENCH_ITERS: usize = 50;

/// Macro to abstract over `#[cfg(feature = "domain_latent")]` parameter.
///
/// The fused forward pass has an extra `domain_latent: Option<...>` parameter
/// when the `domain_latent` feature is enabled. This macro dispatches to the
/// correct call signature.
macro_rules! call_fused {
    ($ctx:expr, $weights:expr, $cache:expr, $predictors:expr, $sp_ctx:expr,
     $token:expr, $pos:expr, $config:expr, $lora:expr, $gate_mode:expr $(,)?) => {{
        #[cfg(feature = "domain_latent")]
        {
            forward_sp_kv_quant(
                $ctx,
                $weights,
                $cache,
                $predictors,
                $sp_ctx,
                $token,
                $pos,
                $config,
                $lora,
                $gate_mode,
                None,
            )
        }
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_sp_kv_quant(
                $ctx,
                $weights,
                $cache,
                $predictors,
                $sp_ctx,
                $token,
                $pos,
                $config,
                $lora,
                $gate_mode,
            )
        }
    }};
}

/// Create hybrid SP-KV + TurboQuant cache, predictors, and forward context.
fn setup(
    config: &Config,
) -> (
    SpKvQuantCache<TurboQuantKVCache>,
    SpKvPredictors,
    SpKvForwardContext,
) {
    let mut sp_config = SpKvConfig::default();
    sp_config.resolve_hidden(config.n_embd);

    let _kvd = kv_dim(config);
    let n_layers = config.n_layer;
    let block_size = config.block_size;

    let tq_cache = TurboQuantKVCache::new(config, 3, 3); // 3-bit key+value (default)
    let cache = SpKvQuantCache::new(sp_config.clone(), tq_cache, n_layers, block_size);

    let predictors = SpKvPredictors::new(
        n_layers,
        config.n_embd,
        sp_config.predictor_hidden,
        config.n_kv_head,
        sp_config.predictor_init_bias,
    );

    let sp_ctx = SpKvForwardContext::new(config, &sp_config);

    (cache, predictors, sp_ctx)
}

// ── T1: Fusion produces finite outputs ───────────────────────────

#[test]
fn test_fusion_produces_finite() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);

    let (mut cache, predictors, mut sp_ctx) = setup(&config);

    for pos in 0..8 {
        let token = (pos * 7 + 3) % config.vocab_size;
        let logits = call_fused!(
            &mut ctx,
            &weights,
            &mut cache,
            &predictors,
            &mut sp_ctx,
            token,
            pos,
            &config,
            None,
            SpKvGateMode::Hard,
        );

        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite at pos={pos}");
        }
    }
}

// ── T2: Density tracking — open gates → ~100% retention ──────────

#[test]
fn test_fusion_density_open_gates() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);

    let (mut cache, predictors, mut sp_ctx) = setup(&config);

    // With init_bias=5.0 (σ(5)≈0.993), gates start nearly fully open.
    // Utility predictor hasn't been trained to discriminate, so all
    // positions should be retained at default threshold=0.5.
    for pos in 0..16 {
        let token = pos % config.vocab_size;
        let _ = call_fused!(
            &mut ctx,
            &weights,
            &mut cache,
            &predictors,
            &mut sp_ctx,
            token,
            pos,
            &config,
            None,
            SpKvGateMode::Hard,
        );
    }

    let density = cache.avg_density(16);
    assert!(
        density > 0.9,
        "Expected density > 0.9 with open gates (init_bias=5.0), got {density}"
    );

    // Per-layer check
    for (layer, meta) in cache.meta.iter().enumerate() {
        let layer_density = meta.density(16);
        assert!(
            layer_density > 0.9,
            "Layer {layer}: density = {layer_density}, expected > 0.9"
        );
    }
}

// ── T3: Reset clears all state ───────────────────────────────────

#[test]
fn test_fusion_reset() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);

    let (mut cache, predictors, mut sp_ctx) = setup(&config);

    // Run some positions to populate state
    for pos in 0..4 {
        let token = pos % config.vocab_size;
        let _ = call_fused!(
            &mut ctx,
            &weights,
            &mut cache,
            &predictors,
            &mut sp_ctx,
            token,
            pos,
            &config,
            None,
            SpKvGateMode::Hard,
        );
    }

    let density_before = cache.avg_density(4);
    assert!(
        density_before > 0.0,
        "Should have some density before reset"
    );

    // Reset
    cache.reset();

    // All layers should have zero retained
    for (layer, meta) in cache.meta.iter().enumerate() {
        assert_eq!(
            meta.retained_count, 0,
            "Layer {layer} should have 0 retained after reset"
        );
        for (pos, &retained) in meta.retained.iter().enumerate() {
            assert!(!retained, "Layer {layer} pos {pos} should not be retained");
        }
    }
}

// ── T4: Hard gate bias correctly masks non-retained positions ────

#[test]
fn test_fusion_hard_gate_masks_pruned() {
    let config = Config::micro();

    let mut sp_config = SpKvConfig::default();
    sp_config.resolve_hidden(config.n_embd);
    // Very high threshold → aggressive pruning
    sp_config.threshold = 0.999;
    // Use tiny window so positions beyond it are truly "outside window"
    // Default window=128 means all positions within 128 of current pos are retained.
    // With window=1, only the immediate neighbor is in-window.
    sp_config.window = 1;

    let n_layers = config.n_layer;
    let block_size = config.block_size;
    let tq_cache = TurboQuantKVCache::new(&config, 3, 3); // 3-bit key+value (default)
    let mut cache = SpKvQuantCache::new(sp_config.clone(), tq_cache, n_layers, block_size);

    // Manually set utilities and retained for layer 0
    // Simulate being at pos=10: positions 9..=10 are in window (window=1).
    // Position 5 is outside window and has low utility → should be pruned.
    let layer = 0;
    let pos = 10;
    cache.meta[layer].utilities[5] = 0.1; // low → pruned (below 0.999), outside window
    cache.meta[layer].utilities[9] = 0.1; // low utility but in window → retained by window
    cache.meta[layer].utilities[10] = 1.0; // high → retained (current pos, always in window)
    cache.meta[layer].retained[9] = true;
    cache.meta[layer].retained[10] = true;
    cache.meta[layer].retained_count = 2;

    // Build hard gate biases
    let mut gate_bias = GateBiasBuffer::new(block_size);
    gate_bias.build_hard(
        &cache.meta[layer].utilities,
        &cache.meta[layer].retained,
        pos,
        sp_config.window,
        sp_config.threshold,
    );

    // Retained positions (in window or high utility): bias = 0.0
    assert_eq!(
        gate_bias.bias[10], 0.0,
        "Current position 10 should have bias 0.0 (always in window)"
    );
    assert_eq!(
        gate_bias.bias[9], 0.0,
        "Position 9 (in window) should have bias 0.0"
    );

    // Pruned position (outside window, low utility): bias = -inf
    assert!(
        gate_bias.bias[5].is_infinite() && gate_bias.bias[5].is_sign_negative(),
        "Position 5 (outside window, low utility) should have bias -inf, got {}",
        gate_bias.bias[5]
    );
}

// ── T5: Soft gating produces finite outputs ──────────────────────

#[test]
fn test_fusion_soft_gating_finite() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);

    let (mut cache, predictors, mut sp_ctx) = setup(&config);

    // Soft gating (training mode): bias = log(u + ε), differentiable
    for pos in 0..8 {
        let token = pos % config.vocab_size;
        let logits = call_fused!(
            &mut ctx,
            &weights,
            &mut cache,
            &predictors,
            &mut sp_ctx,
            token,
            pos,
            &config,
            None,
            SpKvGateMode::Soft,
        );

        for (i, &v) in logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Soft gating: logits[{i}] = {v} not finite at pos={pos}"
            );
        }
    }
}

// ── T6: Multi-step consistency ────────────────────────────────────

#[test]
fn test_fusion_multi_step_consistency() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);

    let (mut cache, predictors, mut sp_ctx) = setup(&config);

    let mut max_logit = f32::NEG_INFINITY;
    let mut min_logit = f32::INFINITY;

    for pos in 0..16 {
        let token = (pos * 13 + 7) % config.vocab_size;
        let logits = call_fused!(
            &mut ctx,
            &weights,
            &mut cache,
            &predictors,
            &mut sp_ctx,
            token,
            pos,
            &config,
            None,
            SpKvGateMode::Hard,
        );

        for &v in logits.iter() {
            max_logit = max_logit.max(v);
            min_logit = min_logit.min(v);
        }
    }

    // Logits should be finite and in reasonable range
    assert!(max_logit.is_finite(), "Max logit not finite: {max_logit}");
    assert!(min_logit.is_finite(), "Min logit not finite: {min_logit}");
    assert!(
        max_logit < 100.0,
        "Max logit suspiciously large: {max_logit}"
    );
    assert!(
        min_logit > -100.0,
        "Min logit suspiciously small: {min_logit}"
    );
}

// ── T7: Unconditional write always retains ───────────────────────

#[test]
fn test_fusion_unconditional_write() {
    let config = Config::micro();
    let kvd = kv_dim(&config);

    let mut sp_config = SpKvConfig::default();
    sp_config.resolve_hidden(config.n_embd);

    let n_layers = config.n_layer;
    let block_size = config.block_size;
    let tq_cache = TurboQuantKVCache::new(&config, 3, 3); // 3-bit key+value (default)
    let mut cache = SpKvQuantCache::new(sp_config, tq_cache, n_layers, block_size);

    let k: Vec<f32> = (0..kvd).map(|i| (i as f32 * 0.1).sin()).collect();
    let v: Vec<f32> = (0..kvd).map(|i| (i as f32 * 0.1).cos()).collect();

    // Unconditional writes should always retain
    for pos in 0..4 {
        cache.write_unconditional(0, &k, &v, pos);
        assert!(
            cache.is_retained(0, pos),
            "Position {pos} should be retained after unconditional write"
        );
    }

    assert_eq!(cache.layer_density(0, 4), 1.0);
}

// ── Benchmark: fusion decode throughput ──────────────────────────

#[test]
fn bench_sp_kv_quant_decode() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let (mut cache, predictors, mut sp_ctx) = setup(&config);

    // Warmup: 4 positions
    for pos in 0..4 {
        let token = pos % config.vocab_size;
        let _ = call_fused!(
            &mut ctx,
            &weights,
            &mut cache,
            &predictors,
            &mut sp_ctx,
            token,
            pos,
            &config,
            None,
            SpKvGateMode::Hard,
        );
    }

    // Reset for benchmark
    cache.reset();
    ctx.reset_dequant();

    let seq_len = config.block_size; // Respect block_size limits (micro=16)
    let start = Instant::now();

    for _ in 0..BENCH_ITERS {
        cache.reset();
        ctx.reset_dequant();
        for pos in 0..seq_len {
            let token = pos % config.vocab_size;
            let _ = call_fused!(
                &mut ctx,
                &weights,
                &mut cache,
                &predictors,
                &mut sp_ctx,
                token,
                pos,
                &config,
                None,
                SpKvGateMode::Hard,
            );
        }
    }

    let elapsed = start.elapsed();
    let total_tokens = BENCH_ITERS * seq_len;
    let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
    let avg_density = cache.avg_density(seq_len);

    println!(
        "SP-KV + TQ Fusion: {total_tokens} tokens in {elapsed:.2?} = {tok_per_sec:.0} tok/s (density={avg_density:.2})"
    );

    // Sanity: should produce at least some tokens per second
    assert!(tok_per_sec > 0.0, "Throughput should be positive");
}
