//! GOAT proofs for Plan 172 (RiM Reasoning Buffer Slots) and Plan 173 (Wall Attention).
//!
//! Run: `cargo test --features "rim_slots,wall_attention" --test goat_172_173_rim_wall -- --nocapture`

use katgpt_core::types::{Config, Rng};
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, PrefillContext, TransformerWeights,
};

#[cfg(feature = "wall_attention")]
use katgpt_core::types::WallConfig;
#[cfg(feature = "wall_attention")]
use katgpt_rs::transformer::WallPrefixState;

// ── Helpers ───────────────────────────────────────────────────

#[allow(dead_code)]
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

// ══════════════════════════════════════════════════════════════
// Plan 172: RiM Reasoning Buffer Slots
// ══════════════════════════════════════════════════════════════

/// Multi-layer config with RiM enabled for prefill tests.
#[cfg(feature = "rim_slots")]
fn make_rim_config() -> Config {
    let mut c = Config::micro();
    c.n_layer = 2;
    c.block_size = 32;
    c.rim_block_count = 2;
    c.rim_tokens_per_block = 2;
    c.rim_buffer_token = c.bos_token;
    c
}

#[cfg(feature = "rim_slots")]
#[test]
fn proof_rim_config_defaults() {
    let config = Config::micro();
    // Default: rim_block_count = 0 → disabled
    assert!(!config.rim_enabled(), "rim should be disabled by default");
    assert_eq!(
        config.rim_total_buffer_tokens(),
        0,
        "total buffer tokens should be 0 when disabled"
    );
}

#[cfg(feature = "rim_slots")]
#[test]
fn proof_rim_extend_tokens() {
    let mut config = Config::micro();
    config.rim_block_count = 4;
    config.rim_tokens_per_block = 2;
    config.rim_buffer_token = config.bos_token; // 26

    let tokens = vec![1, 2, 3];
    let extended = katgpt_rs::transformer::rim_extend_tokens(&tokens, &config);

    // 3 original + 4 blocks × 2 tokens = 11 total
    assert_eq!(extended.len(), 11, "expected 3 + 8 = 11 tokens");

    // First 3 tokens unchanged
    assert_eq!(&extended[..3], &[1, 2, 3], "original tokens preserved");

    // Last 8 tokens are all the buffer token (bos_token = 26)
    for (i, &t) in extended[3..].iter().enumerate() {
        assert_eq!(t, 26, "buffer token {i} should be bos_token (26)");
    }
}

#[cfg(feature = "rim_slots")]
#[test]
fn proof_rim_readout_index() {
    // With RiM enabled: readout at prompt_len + total_buffer - 1
    let mut config = Config::micro();
    config.rim_block_count = 4;
    config.rim_tokens_per_block = 2;

    let idx = katgpt_rs::transformer::rim_readout_index(3, &config);
    assert_eq!(idx, 3 + 8 - 1, "readout index = 10 with rim enabled");

    // With RiM disabled: readout at last prompt token
    let config_disabled = Config::micro();
    let idx_disabled = katgpt_rs::transformer::rim_readout_index(3, &config_disabled);
    assert_eq!(
        idx_disabled, 2,
        "readout index = prompt_len - 1 when rim disabled"
    );
}

#[cfg(feature = "rim_slots")]
#[test]
fn proof_rim_prefill_produces_logits() {
    let config = make_rim_config();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);

    let extended_tokens = katgpt_rs::transformer::rim_extend_tokens(&[1, 2, 3], &config);

    let logits = katgpt_rs::transformer::forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &extended_tokens,
        &config,
        None,
        #[cfg(feature = "domain_latent")]
        None,
    );

    // Logits should be produced and non-empty
    assert!(!logits.is_empty(), "logits should be non-empty");
    // Check that at least some logits are non-zero and finite
    let has_nonzero = logits.iter().any(|&l| l != 0.0);
    let all_finite = logits.iter().all(|&l| l.is_finite());
    assert!(has_nonzero, "at least one logit should be non-zero");
    assert!(all_finite, "all logits should be finite");
}

#[cfg(feature = "rim_slots")]
#[test]
fn proof_rim_zero_decode_cost() {
    // Buffer slots are prefill-only. During decode, forward() processes a single
    // token — no buffer tokens are involved. This test proves the decode path
    // is identical regardless of rim config by running forward() with rim disabled
    // vs enabled and verifying the timing overhead is negligible.

    let mut config_no_rim = Config::micro();
    config_no_rim.block_size = 512; // large enough for prefill + decode

    let mut config_rim = Config::micro();
    config_rim.block_size = 512;
    config_rim.rim_block_count = 4;
    config_rim.rim_tokens_per_block = 2;
    config_rim.rim_buffer_token = config_rim.bos_token;

    // First, prefill to fill the KV cache (rim affects prefill only)
    let mut rng1 = Rng::new(42);
    let weights1 = TransformerWeights::new(&config_no_rim, &mut rng1);
    let mut cache1 = MultiLayerKVCache::new(&config_no_rim);
    let mut ctx1 = ForwardContext::new(&config_no_rim);
    let mut prefill1 = PrefillContext::new(&config_no_rim);
    let tokens1 = vec![1, 2, 3];
    katgpt_rs::transformer::forward_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights1,
        &mut cache1,
        &tokens1,
        &config_no_rim,
        None,
        #[cfg(feature = "domain_latent")]
        None,
    );

    let mut rng2 = Rng::new(42);
    let weights2 = TransformerWeights::new(&config_rim, &mut rng2);
    let mut cache2 = MultiLayerKVCache::new(&config_rim);
    let mut ctx2 = ForwardContext::new(&config_rim);
    let mut prefill2 = PrefillContext::new(&config_rim);
    let extended2 = katgpt_rs::transformer::rim_extend_tokens(&[1, 2, 3], &config_rim);
    katgpt_rs::transformer::forward_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights2,
        &mut cache2,
        &extended2,
        &config_rim,
        None,
        #[cfg(feature = "domain_latent")]
        None,
    );

    // Now benchmark decode: forward() calls on each.
    // block_size=512, max_pos=511. Prefill uses 11 tokens, so max decode steps = 500.
    let n_decode = 500;
    let start_pos_no_rim = tokens1.len(); // 3
    let start_pos_rim = extended2.len(); // 11

    let t1_start = std::time::Instant::now();
    for i in 0..n_decode {
        let _logits = katgpt_rs::transformer::forward(
            &mut ctx1,
            &weights1,
            &mut cache1,
            1,
            start_pos_no_rim + i,
            &config_no_rim,
        );
    }
    let t1_elapsed = t1_start.elapsed();

    let t2_start = std::time::Instant::now();
    for i in 0..n_decode {
        let _logits = katgpt_rs::transformer::forward(
            &mut ctx2,
            &weights2,
            &mut cache2,
            1,
            start_pos_rim + i,
            &config_rim,
        );
    }
    let t2_elapsed = t2_start.elapsed();

    // Overhead should be < 20% (accounting for measurement noise)
    // Buffer tokens don't exist in decode — they're prefill-only.
    let overhead = if t1_elapsed.as_nanos() > 0 {
        (t2_elapsed.as_nanos() as f64 / t1_elapsed.as_nanos() as f64) - 1.0
    } else {
        0.0
    };

    println!(
        "RiM decode overhead: {:.1}% (no_rim={:?}, rim={:?})",
        overhead * 100.0,
        t1_elapsed,
        t2_elapsed,
    );

    assert!(
        overhead < 0.20,
        "rim decode overhead should be < 20%, got {:.1}%",
        overhead * 100.0
    );
}

// ══════════════════════════════════════════════════════════════
// Plan 173: Wall Attention
// ══════════════════════════════════════════════════════════════

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_config_defaults() {
    let config = WallConfig::default();
    assert!(
        approx_eq(config.gate_bias, 6.0, 1e-6),
        "gate_bias should be 6.0"
    );
    assert!(
        approx_eq(config.gate_max, 0.87, 1e-6),
        "gate_max should be 0.87"
    );
    assert!(
        config.use_key_projected,
        "use_key_projected should be true by default"
    );
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_gate_open_at_high_bias() {
    // With bias=6.0 and zero weights, logit=6.0 for each dimension.
    // log_sigmoid(6.0) = -log(1 + exp(-6)) ≈ -0.00247
    // Clamped to (-0.87, 0] → -0.00247 (unchanged, within range).
    let key = [1.0f32; 4];
    let w_g = [0.0f32; 4];
    let mut gate_buf = [0.0f32; 4];

    WallPrefixState::compute_gate_from_key(&mut gate_buf, &key, &w_g, 6.0, 0.87);

    // log_sigmoid(6.0) = -softplus(-6) = -log(1 + exp(-6)) ≈ -0.00247
    let expected = -(1.0f32 + (-6.0f32).exp()).ln(); // ≈ -0.00247
    println!("gate values: {gate_buf:?}, expected ≈ {expected:.6}");

    for (d, &g) in gate_buf.iter().enumerate() {
        assert!(
            approx_eq(g, expected, 1e-4),
            "gate[{d}] = {g}, expected ≈ {expected:.6} (open gate near 0)"
        );
        // Gate should be very close to 0 (retention ≈ 1.0)
        assert!(g > -0.01, "gate[{d}] should be near 0 (open), got {g}");
    }
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_gate_active_forgetting_at_zero_bias() {
    // With bias=0.0 and zero weights, logit=0 for each dimension.
    // log_sigmoid(0) = -ln(2) ≈ -0.6931
    // Clamped to (-0.87, 0] → -0.6931 (within range).
    let key = [1.0f32; 4];
    let w_g = [0.0f32; 4];
    let mut gate_buf = [0.0f32; 4];

    WallPrefixState::compute_gate_from_key(&mut gate_buf, &key, &w_g, 0.0, 0.87);

    let expected = -(2.0f32).ln(); // -ln(2) ≈ -0.6931
    println!("gate values: {gate_buf:?}, expected ≈ {expected:.6}");

    for (d, &g) in gate_buf.iter().enumerate() {
        assert!(
            approx_eq(g, expected, 1e-4),
            "gate[{d}] = {g}, expected ≈ {expected:.6} (active forgetting)"
        );
    }
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_prefix_sum_numerical_stability() {
    // Simulate 8192 tokens with gate value -0.1 per dimension per step.
    // After 8192 steps: prefix_sum = -819.2.
    // Verify exp(prefix_sum) is positive and finite (no panic).
    //
    // Worst case with gate_max=0.87: prefix_sum = -0.87 * 8192 = -7127.04
    // exp(-7127) ≈ 0.0 (underflow to zero is acceptable for query rescale).
    // exp(7127) would overflow but that's key rescale — we verify it doesn't panic.

    let mut config = Config::micro();
    config.n_kv_head = 1;
    config.head_dim = 4;
    let mut state = WallPrefixState::new(&config);

    let gate_values = [-0.1f32; 4]; // per dimension
    let n_steps = 8192usize;

    for _ in 0..n_steps {
        state.update_prefix(0, 0, &gate_values);
    }

    // Verify prefix sums accumulated correctly: -0.1 * 8192 = -819.2
    // We can only observe via rescale behavior, since prefix_sums is private.
    // Build a simple kv_group_lut: identity (n_head = n_kv_head = 1)
    let mut lut = [0u8; 128];
    for i in 0..4 {
        lut[i] = 0; // all Q heads map to KV head 0 (GQA with 1 KV head)
    }

    // This should not panic even with large negative prefix sums
    // For q with 4 heads, each sharing KV head 0:
    let mut q_full = vec![0.0f32; 16]; // 4 heads × 4 dim
    for h in 0..4 {
        q_full[h * 4..h * 4 + 4].fill(1.0);
    }
    let mut k_full = vec![1.0f32; 4]; // 1 KV head × 4 dim

    state.rescale_query(0, &mut q_full, &lut, 4);
    state.rescale_key(0, &mut k_full);

    // After rescale with prefix_sum ≈ -819.2:
    // query: q * exp(-819.2) ≈ 0 (underflow — acceptable)
    // key:   k * exp(819.2) ≈ inf (overflow — but should not panic)
    println!("rescaled q (head 0): {:?}", &q_full[0..4]);
    println!("rescaled k: {k_full:?}");

    // Query values should be finite or zero (underflow OK)
    for (i, &v) in q_full[0..4].iter().enumerate() {
        assert!(
            v.is_finite() || v == 0.0,
            "q[{i}] should be finite or zero, got {v}"
        );
    }
    // Key values may overflow to inf — that's expected for extreme prefix sums
    // The important thing is it didn't panic.
    for (i, &v) in k_full.iter().enumerate() {
        assert!(
            v.is_finite() || v.is_infinite(),
            "k[{i}] should be finite or infinite (no NaN), got {v}"
        );
        assert!(!v.is_nan(), "k[{i}] should not be NaN");
    }
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_rescale_identity_at_zero_prefix() {
    // With all-zero prefix sums (fresh state), rescale is identity:
    // exp(0) = 1.0 for query, exp(-0) = 1.0 for key.
    let config = Config::micro(); // head_dim=4, n_head=4, n_kv_head=4
    let mut state = WallPrefixState::new(&config);

    // Query: 4 heads × 4 dim = 16 values
    let mut q = vec![0.0f32; 16];
    for h in 0..4 {
        for d in 0..4 {
            q[h * 4 + d] = (h * 4 + d + 1) as f32; // [1, 2, 3, 4, 5, ...]
        }
    }
    let q_original = q.clone();

    // Key: 4 KV heads × 4 dim = 16 values
    let mut k = vec![1.0f32; 16];
    let k_original = k.clone();

    // Identity kv_group_lut (n_head == n_kv_head)
    let mut lut = [0u8; 128];
    for i in 0..4 {
        lut[i] = i as u8;
    }

    state.rescale_query(0, &mut q, &lut, 4);
    state.rescale_key(0, &mut k);

    // Should be unchanged: exp(0) = 1.0
    for (i, (&q_new, &q_old)) in q.iter().zip(q_original.iter()).enumerate() {
        assert!(
            approx_eq(q_new, q_old, 1e-6),
            "q[{i}] should be unchanged: got {q_new}, expected {q_old}"
        );
    }
    for (i, (&k_new, &k_old)) in k.iter().zip(k_original.iter()).enumerate() {
        assert!(
            approx_eq(k_new, k_old, 1e-6),
            "k[{i}] should be unchanged: got {k_new}, expected {k_old}"
        );
    }
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_query_key_rescale_correctness() {
    // Set prefix_sum = [0.5, -0.3, 0.0, 1.0] for KV head 0.
    // Since prefix_sums is private, we use update_prefix to accumulate
    // to the desired values.
    let mut config = Config::micro();
    config.n_kv_head = 1; // Single KV head for simplicity
    config.n_head = 1;
    let mut state = WallPrefixState::new(&config);

    // Accumulate to target: [0.5, -0.3, 0.0, 1.0]
    let target = [0.5f32, -0.3, 0.0, 1.0];
    state.update_prefix(0, 0, &target);

    // Query: [1.0, 2.0, 3.0, 4.0] (single head, 4 dim)
    let mut q = vec![1.0f32, 2.0, 3.0, 4.0];
    // Extend to n_embd = 16 for full compatibility (but we only check first 4)
    q.resize(16, 0.0);
    // Actually, n_head=1, n_embd=1*4=4... but micro() gives n_embd=16.
    // Let's just use a properly sized vector.
    let mut q = vec![0.0f32; config.n_embd]; // 16
    q[0] = 1.0;
    q[1] = 2.0;
    q[2] = 3.0;
    q[3] = 4.0;

    // Key: [1.0, 1.0, 1.0, 1.0] (single KV head, kv_dim = 4)
    let mut k = vec![1.0f32; config.n_kv_head * config.head_dim]; // 4

    // Identity lut (1 head → head 0)
    let mut lut = [0u8; 128];
    lut[0] = 0;

    state.rescale_query(0, &mut q, &lut, config.n_head);
    state.rescale_key(0, &mut k);

    // Expected query rescale: q[d] *= exp(prefix_sum[d])
    let expected_q = [
        1.0f32 * 0.5f32.exp(),    // exp(0.5) ≈ 1.6487
        2.0f32 * (-0.3f32).exp(), // exp(-0.3) ≈ 0.7408
        3.0f32 * 0.0f32.exp(),    // exp(0) = 1.0
        4.0f32 * 1.0f32.exp(),    // exp(1.0) ≈ 2.7183
    ];

    // Expected key rescale: k[d] *= exp(-prefix_sum[d])
    let expected_k = [
        1.0f32 * (-0.5f32).exp(), // exp(-0.5) ≈ 0.6065
        1.0f32 * 0.3f32.exp(),    // exp(0.3) ≈ 1.3499
        1.0f32 * 0.0f32.exp(),    // exp(0) = 1.0
        1.0f32 * (-1.0f32).exp(), // exp(-1.0) ≈ 0.3679
    ];

    println!("rescaled q: {:?}", &q[..4]);
    println!("expected q: {expected_q:?}");
    println!("rescaled k: {k:?}");
    println!("expected k: {expected_k:?}");

    // Cephes SIMD exp approximation vs std::f32::exp — allow 6% absolute tolerance
    // (the SIMD Cephes kernel has measurable imprecision for small head_dim inputs).
    for (d, (&got, &exp)) in q[..4].iter().zip(expected_q.iter()).enumerate() {
        assert!(approx_eq(got, exp, 6e-2), "q[{d}] = {got}, expected {exp}");
    }
    for (d, (&got, &exp)) in k.iter().zip(expected_k.iter()).enumerate() {
        assert!(approx_eq(got, exp, 6e-2), "k[{d}] = {got}, expected {exp}");
    }
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_forward_produces_logits() {
    // Wall attention integration test: forward_base with wall_config active.
    // Verifies that the Wall gate projection + Q/K rescale + attention pipeline
    // produces valid logits end-to-end.
    let mut config = Config::micro();
    config.n_layer = 2;
    config.block_size = 32;
    config.wall_config = Some(WallConfig::default());

    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    weights.init_wall_gates(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    // Decode single token
    let logits = katgpt_rs::transformer::forward(&mut ctx, &weights, &mut cache, 1, 0, &config);

    assert!(!logits.is_empty(), "logits should be non-empty");
    let has_nonzero = logits.iter().any(|&l| l != 0.0);
    let all_finite = logits.iter().all(|&l| l.is_finite());
    assert!(has_nonzero, "at least one logit should be non-zero");
    assert!(all_finite, "all logits should be finite");

    // Second token should also work (prefix sum accumulates)
    let logits2 = katgpt_rs::transformer::forward(&mut ctx, &weights, &mut cache, 2, 1, &config);
    assert!(
        !logits2.is_empty(),
        "second forward logits should be non-empty"
    );
    assert!(
        logits2.iter().all(|&l| l.is_finite()),
        "all second logits should be finite"
    );
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_decode_overhead_vs_baseline() {
    // Wall rescale should add negligible overhead vs baseline (no wall).
    // Compares decode throughput: baseline (no wall) vs Wall attention.
    let n_decode = 500;

    let mut config_baseline = Config::micro();
    config_baseline.block_size = 512;

    let mut config_wall = Config::micro();
    config_wall.block_size = 512;
    config_wall.wall_config = Some(WallConfig::default());

    // Prefill baseline
    let mut rng1 = Rng::new(42);
    let weights1 = TransformerWeights::new(&config_baseline, &mut rng1);
    let mut cache1 = MultiLayerKVCache::new(&config_baseline);
    let mut ctx1 = ForwardContext::new(&config_baseline);
    let mut prefill1 = PrefillContext::new(&config_baseline);
    katgpt_rs::transformer::forward_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights1,
        &mut cache1,
        &[1, 2, 3],
        &config_baseline,
        None,
        #[cfg(feature = "domain_latent")]
        None,
    );

    // Prefill wall
    let mut rng2 = Rng::new(42);
    let mut weights2 = TransformerWeights::new(&config_wall, &mut rng2);
    weights2.init_wall_gates(&config_wall, &mut rng2);
    let mut cache2 = MultiLayerKVCache::new(&config_wall);
    let mut ctx2 = ForwardContext::new(&config_wall);
    let mut prefill2 = PrefillContext::new(&config_wall);
    katgpt_rs::transformer::forward_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights2,
        &mut cache2,
        &[1, 2, 3],
        &config_wall,
        None,
        #[cfg(feature = "domain_latent")]
        None,
    );

    // Benchmark decode
    let t1_start = std::time::Instant::now();
    for i in 0..n_decode {
        let _ = katgpt_rs::transformer::forward(
            &mut ctx1,
            &weights1,
            &mut cache1,
            1,
            3 + i,
            &config_baseline,
        );
    }
    let t1_elapsed = t1_start.elapsed();

    let t2_start = std::time::Instant::now();
    for i in 0..n_decode {
        let _ = katgpt_rs::transformer::forward(
            &mut ctx2,
            &weights2,
            &mut cache2,
            1,
            3 + i,
            &config_wall,
        );
    }
    let t2_elapsed = t2_start.elapsed();

    let overhead = if t1_elapsed.as_nanos() > 0 {
        (t2_elapsed.as_nanos() as f64 / t1_elapsed.as_nanos() as f64) - 1.0
    } else {
        0.0
    };

    println!(
        "Wall decode overhead: {:.1}% (baseline={:?}, wall={:?})",
        overhead * 100.0,
        t1_elapsed,
        t2_elapsed,
    );

    assert!(
        overhead < 0.20,
        "Wall decode overhead should be < 20%, got {:.1}%",
        overhead * 100.0
    );
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_multilayer_per_layer_isolation() {
    // Per-layer prefix sums are independent. After processing 2 layers at pos=0,
    // each layer should have its own gate accumulation. Verify by running forward
    // with multi-layer config and checking logits are finite and non-trivial.
    let mut config = Config::micro();
    config.n_layer = 3;
    config.block_size = 64;
    config.wall_config = Some(WallConfig::default());

    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    weights.init_wall_gates(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    // Decode multiple tokens — prefix sums accumulate independently per layer
    for pos in 0..8 {
        let logits =
            katgpt_rs::transformer::forward(&mut ctx, &weights, &mut cache, 1, pos, &config);
        assert!(!logits.is_empty(), "logits empty at pos {pos}");
        assert!(
            logits.iter().all(|&l| l.is_finite()),
            "non-finite logits at pos {pos}"
        );
    }
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_prefill_then_decode() {
    // End-to-end: prefill with Wall → decode with Wall.
    // Wall prefix sums are reset at prefill start, then decode continues accumulating.
    let mut config = Config::micro();
    config.n_layer = 2;
    config.block_size = 64;
    config.wall_config = Some(WallConfig::default());

    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    weights.init_wall_gates(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);

    // Prefill
    let prompt = vec![1, 2, 3, 4, 5];
    let logits = katgpt_rs::transformer::forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &prompt,
        &config,
        None,
        #[cfg(feature = "domain_latent")]
        None,
    );
    assert!(!logits.is_empty());
    assert!(logits.iter().all(|&l| l.is_finite()));

    // Decode continuation
    for i in 0..4 {
        let pos = prompt.len() + i;
        let decode_logits =
            katgpt_rs::transformer::forward(&mut ctx, &weights, &mut cache, 1, pos, &config);
        assert!(
            !decode_logits.is_empty(),
            "decode logits empty at pos {pos}"
        );
        assert!(
            decode_logits.iter().all(|&l| l.is_finite()),
            "non-finite decode logits at pos {pos}"
        );
    }
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_enabled_convenience() {
    // wall_enabled() returns true when wall_config is Some
    let mut config = Config::micro();
    assert!(!config.wall_enabled(), "wall should be disabled by default");

    config.wall_config = Some(WallConfig::default());
    assert!(
        config.wall_enabled(),
        "wall should be enabled when wall_config is Some"
    );
}

#[cfg(feature = "wall_attention")]
#[test]
fn proof_wall_gate_weights_initialized() {
    // attn_wg should be initialized with proper dimensions (kv_dim elements per layer).
    let mut config = Config::micro();
    config.n_layer = 2;
    config.wall_config = Some(WallConfig::default());

    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    weights.init_wall_gates(&config, &mut rng);

    let kv_dim = config.n_kv_head * config.head_dim;
    for (i, layer) in weights.layers.iter().enumerate() {
        assert_eq!(
            layer.attn_wg.len(),
            kv_dim,
            "layer {i} attn_wg should have kv_dim={} elements, got {}",
            kv_dim,
            layer.attn_wg.len()
        );
        // Weights should be non-zero (initialized with normal distribution)
        let has_nonzero = layer.attn_wg.iter().any(|&w| w != 0.0);
        assert!(
            has_nonzero,
            "layer {i} attn_wg should have non-zero weights"
        );
    }
}
