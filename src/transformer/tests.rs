use super::*;
use crate::types;

#[test]
fn test_forward_output_size() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_logits_finite() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit {i} is not finite: {l}");
    }
}

#[test]
fn test_forward_cache_populated() {
    let config = Config::micro();
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    let key_sum: f32 = cache.layers[0].key[..kvd].iter().sum();
    let val_sum: f32 = cache.layers[0].value[..kvd].iter().sum();
    assert!(key_sum != 0.0, "K cache at pos 0 should be populated");
    assert!(val_sum != 0.0, "V cache at pos 0 should be populated");
}

#[test]
fn test_forward_positions_differ() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let logits_0 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
    let logits_1 = forward(&mut ctx, &weights, &mut cache, 0, 1, &config);
    let different = logits_0.iter().zip(logits_1).any(|(&a, b)| a != *b);
    assert!(different, "logits at different positions should differ");
}

#[test]
fn test_generate_deterministic() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut rng1 = Rng::new(100);
    let t1 = generate(&weights, &config, &mut rng1, 16);

    let mut rng2 = Rng::new(100);
    let t2 = generate(&weights, &config, &mut rng2, 16);

    assert_eq!(t1, t2, "Same seed must produce same tokens");
}

#[test]
fn test_generate_valid_tokens() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let tokens = generate(&weights, &config, &mut rng, 32);
    assert_eq!(tokens.len(), 32);
    for &t in &tokens {
        assert!(t < config.vocab_size, "Token {t} out of range");
    }
}

#[test]
fn test_tokens_to_string() {
    let tokens = vec![0, 1, 2, 25, 26];
    let s = tokens_to_string(&tokens);
    assert_eq!(s, "abcz_");
}

#[test]
fn test_forward_context_reuse() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // Multiple forward passes with same context should give same results
    let _l1 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
    let l2 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    // Note: results differ because cache accumulates, but buffers should not leak
    for &v in l2.iter() {
        assert!(v.is_finite(), "reused context produced non-finite: {v}");
    }
}

// ── Multi-layer tests ─────────────────────────────────────────

#[test]
fn test_forward_output_size_nlayer2() {
    let mut config = Config::micro();
    config.n_layer = 2;
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    assert_eq!(weights.layers.len(), 2);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    assert_eq!(cache.layers.len(), 2);
    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_logits_finite_nlayer4() {
    let mut config = Config::micro();
    config.n_layer = 4;
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit {i} is not finite with n_layer=4: {l}");
    }
}

#[test]
fn test_n_layer_1_matches_current() {
    // n_layer=1 must produce identical deterministic output to old single-layer code
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut rng1 = Rng::new(100);
    let t1 = generate(&weights, &config, &mut rng1, 16);

    let mut rng2 = Rng::new(100);
    let t2 = generate(&weights, &config, &mut rng2, 16);

    assert_eq!(t1, t2, "n_layer=1 should be deterministic");
    assert_eq!(config.n_layer, 1, "micro config should have n_layer=1");
}

#[test]
fn test_multi_layer_cache_populated() {
    let mut config = Config::micro();
    config.n_layer = 3;
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    // Every layer's cache should be populated
    for (layer_idx, layer_cache) in cache.layers.iter().enumerate() {
        let key_sum: f32 = layer_cache.key[..kvd].iter().sum();
        let val_sum: f32 = layer_cache.value[..kvd].iter().sum();
        assert!(
            key_sum != 0.0,
            "layer {layer_idx} K cache at pos 0 should be populated"
        );
        assert!(
            val_sum != 0.0,
            "layer {layer_idx} V cache at pos 0 should be populated"
        );
    }
}

#[test]
fn test_hidden_state_populated() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    let sum: f32 = ctx.hidden_state.iter().sum();
    assert!(
        sum != 0.0,
        "hidden_state should be populated after forward pass"
    );
    for (i, &v) in ctx.hidden_state.iter().enumerate() {
        assert!(v.is_finite(), "hidden_state[{i}] should be finite: {v}");
    }
}

#[test]
fn test_multi_layer_generate_valid() {
    let mut config = Config::micro();
    config.n_layer = 4;
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let tokens = generate(&weights, &config, &mut rng, 16);
    assert_eq!(tokens.len(), 16);
    for &t in &tokens {
        assert!(t < config.vocab_size, "Token {t} out of range");
    }
}

// ── GQA tests ───────────────────────────────────────────────

#[test]
fn test_gqa_produces_valid_logits() {
    let config = Config::gqa_draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    for pos in 0..4 {
        let logits = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "gqa_draft logit {i} at pos {pos} not finite: {l}"
            );
        }
    }
}

#[test]
fn test_gqa_mha_backward_compat() {
    // When n_kv_head == n_head, GQA produces identical results to standard MHA.
    // Micro config has n_kv_head=4, n_head=4 → pure MHA.
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut rng1 = Rng::new(100);
    let t1 = generate(&weights, &config, &mut rng1, 16);

    let mut rng2 = Rng::new(100);
    let t2 = generate(&weights, &config, &mut rng2, 16);

    assert_eq!(
        t1, t2,
        "MHA backward compat: same seed must produce same tokens"
    );
    assert_eq!(
        config.n_kv_head, config.n_head,
        "micro config should have n_kv_head == n_head"
    );
}

#[test]
fn test_gqa_kv_cache_smaller() {
    // GQA config should have smaller KV cache than equivalent MHA config
    let gqa = Config::gqa_draft();
    let kvd = crate::types::kv_dim(&gqa);
    assert_eq!(
        kvd,
        gqa.n_kv_head * gqa.head_dim,
        "kv_dim should be n_kv_head * head_dim"
    );
    assert!(
        kvd < gqa.n_embd,
        "GQA kv_dim ({kvd}) should be < n_embd ({})",
        gqa.n_embd
    );

    // Verify cache is correctly sized
    let cache = KVCache::new(&gqa);
    assert_eq!(
        cache.key.len(),
        gqa.block_size * kvd,
        "GQA key cache should use kv_dim"
    );
    assert_eq!(
        cache.value.len(),
        gqa.block_size * kvd,
        "GQA value cache should use kv_dim"
    );
}

#[test]
fn test_gqa_generate_valid_tokens() {
    let config = Config::gqa_draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let tokens = generate(&weights, &config, &mut rng, 8);
    assert_eq!(tokens.len(), 8);
    for &t in &tokens {
        assert!(t < config.vocab_size, "GQA token {t} out of range");
    }
}

#[test]
fn test_config_validate_gqa() {
    // Valid configs should pass validation
    assert!(Config::micro().validate().is_ok());
    assert!(Config::draft().validate().is_ok());
    assert!(Config::small_target().validate().is_ok());
    assert!(Config::gqa_draft().validate().is_ok());

    // Invalid: n_head not divisible by n_kv_head
    let mut bad = Config::micro();
    bad.n_kv_head = 3; // n_head=4, not divisible by 3
    assert!(bad.validate().is_err());

    // Invalid: n_head * head_dim != n_embd
    let mut bad2 = Config::micro();
    bad2.head_dim = 5; // 4*5=20 != 16
    assert!(bad2.validate().is_err());
}

// ── Paged KV cache tests ────────────────────────────────────

#[test]
fn test_paged_cache_write_read_roundtrip() {
    let config = Config::micro();
    let mut paged = PagedKVCache::new(&config, 1);
    let kvd = crate::types::kv_dim(&config);

    // Ensure pages for position 0
    paged.ensure_pages(0, 0);

    // Write some K/V data
    let k_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.1).collect();
    let v_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.2).collect();
    paged.write_kv(0, 0, 0, &k_data, &v_data);

    // Read back
    let mut k_out = vec![0.0f32; kvd];
    let mut v_out = vec![0.0f32; kvd];
    paged.read_kv(0, 0, 0, &mut k_out, &mut v_out);

    assert_eq!(k_out, k_data, "K data roundtrip mismatch");
    assert_eq!(v_out, v_data, "V data roundtrip mismatch");
}

#[test]
fn test_paged_cache_linear_matches_flat() {
    // Paged cache should produce same results as flat cache for a linear sequence
    let config = Config::micro();
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Run with flat cache
    let mut ctx = ForwardContext::new(&config);
    let mut flat_cache = MultiLayerKVCache::new(&config);
    let _flat_logits = forward(&mut ctx, &weights, &mut flat_cache, 0, 0, &config).to_vec();

    // Manually copy flat cache data to paged cache
    let mut paged = PagedKVCache::new(&config, 1);
    paged.ensure_pages(0, 0);

    for (layer_idx, layer_cache) in flat_cache.layers.iter().enumerate() {
        let k_data = &layer_cache.key[..kvd];
        let v_data = &layer_cache.value[..kvd];
        paged.write_kv(layer_idx, 0, 0, k_data, v_data);
    }

    // Read back and compare
    for (layer_idx, layer_cache) in flat_cache.layers[..config.n_layer].iter().enumerate() {
        let mut k_out = vec![0.0f32; kvd];
        let mut v_out = vec![0.0f32; kvd];
        paged.read_kv(layer_idx, 0, 0, &mut k_out, &mut v_out);

        let flat_k = &layer_cache.key[..kvd];
        let flat_v = &layer_cache.value[..kvd];
        assert_eq!(k_out, flat_k, "layer {layer_idx} K mismatch: paged vs flat");
        assert_eq!(v_out, flat_v, "layer {layer_idx} V mismatch: paged vs flat");
    }
}

#[test]
fn test_paged_cache_fork_no_corruption() {
    let config = Config::micro();
    let kvd = crate::types::kv_dim(&config);
    let mut paged = PagedKVCache::new(&config, 1);

    // Write data to seq 0 at position 0
    paged.ensure_pages(0, 0);
    let k_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 1.0).collect();
    let v_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 2.0).collect();
    paged.write_kv(0, 0, 0, &k_orig, &v_orig);

    // Fork at position 0 (share nothing — fork_page = 0/16 = 0)
    let fork_seq = paged.fork(0, 0);

    // Write different data to forked seq
    paged.ensure_pages(fork_seq, 0);
    let k_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 99.0).collect();
    let v_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 100.0).collect();
    paged.write_kv(0, fork_seq, 0, &k_fork, &v_fork);

    // Original seq should be unchanged
    let mut k_check = vec![0.0f32; kvd];
    let mut v_check = vec![0.0f32; kvd];
    paged.read_kv(0, 0, 0, &mut k_check, &mut v_check);
    assert_eq!(k_check, k_orig, "original K corrupted after fork write");
    assert_eq!(v_check, v_orig, "original V corrupted after fork write");
}

#[test]
fn test_paged_cache_fork_shares_prefix() {
    let config = Config::micro();
    let kvd = crate::types::kv_dim(&config);
    let mut paged = PagedKVCache::new(&config, 1);

    // Write data at positions 0..PAGE_SIZE (fills one page)
    paged.ensure_pages(0, PAGE_SIZE - 1);
    for pos in 0..PAGE_SIZE {
        let k: Vec<f32> = vec![pos as f32; kvd];
        let v: Vec<f32> = vec![pos as f32 * 2.0; kvd];
        paged.write_kv(0, 0, pos, &k, &v);
    }

    // Fork at position 8 (still within page 0)
    let fork_seq = paged.fork(0, 8);

    // Ensure forked seq has its own pages from fork point
    paged.ensure_pages(fork_seq, PAGE_SIZE);

    // The forked seq should share page 0 (prefix) but have its own page 1+
    // Verify shared prefix data is accessible
    let mut k_out = vec![0.0f32; kvd];
    let mut v_out = vec![0.0f32; kvd];
    paged.read_kv(0, fork_seq, 0, &mut k_out, &mut v_out);
    assert_eq!(k_out[0], 0.0, "forked seq should see original pos 0 data");
}

#[test]
fn test_paged_cache_reset_frees_pages() {
    let config = Config::micro();
    let mut paged = PagedKVCache::new(&config, 2);

    // Allocate pages for two sequences
    paged.ensure_pages(0, 31); // 2 pages (0..15 and 16..31)
    paged.ensure_pages(1, 15); // 1 page

    let total_before = paged.total_pages;
    assert!(total_before > 0, "should have allocated some pages");

    // Reset should free all pages
    paged.reset();

    // Free list should contain the freed pages
    // (exact count depends on implementation, but should be > 0)
    // After reset, we can allocate again and reuse freed pages
    paged.ensure_pages(0, 0);
    // If reuse works, total_pages shouldn't grow
    assert_eq!(paged.total_pages, total_before, "should reuse freed pages");
}

#[test]
fn test_snapshot_restore_roundtrip() {
    // Forward some tokens, snapshot, modify, restore, verify same logits
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // Fill cache with tokens at positions 0..4
    for pos in 0..4 {
        let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
    }

    // Snapshot at position 4
    let snapshot = cache.snapshot(4, &config);

    // Fill more positions
    for pos in 4..8 {
        let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
    }

    // Now restore
    cache.restore(&snapshot, &config);

    // Verify restored: forward at position 4 should give same result as fresh cache at pos 4
    let mut fresh_cache = MultiLayerKVCache::new(&config);
    let mut fresh_ctx = ForwardContext::new(&config);
    for pos in 0..4 {
        let _ = forward(
            &mut fresh_ctx,
            &weights,
            &mut fresh_cache,
            pos,
            pos,
            &config,
        );
    }

    let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
    let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

    for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "restored logits should match fresh: {a} vs {b}"
        );
    }
}

#[test]
fn test_snapshot_correct_size() {
    let config = Config::micro();
    let kd = types::kv_dim(&config);
    let cache = MultiLayerKVCache::new(&config);
    let snapshot = cache.snapshot(5, &config);

    assert_eq!(snapshot.pos, 5);
    assert_eq!(snapshot.layers.len(), config.n_layer);
    for layer in &snapshot.layers {
        assert_eq!(layer.key.len(), 5 * kd);
        assert_eq!(layer.value.len(), 5 * kd);
    }
}

#[test]
fn test_restore_preserves_snapshot_data() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // Fill cache
    for pos in 0..8 {
        let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
    }

    // Snapshot at position 3
    let snapshot = cache.snapshot(3, &config);

    // Restore
    cache.restore(&snapshot, &config);

    // Verify snapshot data is correctly restored (Issue 097: no zeroing beyond snapshot)
    let kd = types::kv_dim(&config);
    for (layer, snap_layer) in cache.layers.iter().zip(snapshot.layers.iter()) {
        assert_eq!(
            &layer.key[..3 * kd],
            &snap_layer.key,
            "key snapshot data mismatch"
        );
        assert_eq!(
            &layer.value[..3 * kd],
            &snap_layer.value,
            "value snapshot data mismatch"
        );
    }
}

#[test]
fn test_snapshot_restore_multi_layer() {
    // Test with n_layer > 1 (small_target config)
    let config = Config::small_target();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // Fill cache
    for pos in 0..4 {
        let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
    }

    let snapshot = cache.snapshot(4, &config);
    assert_eq!(snapshot.layers.len(), 4, "should have 4 layer snapshots");

    // Modify and restore
    for pos in 4..8 {
        let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
    }
    cache.restore(&snapshot, &config);

    // Verify restored correctly by checking logits match fresh cache
    let mut fresh_cache = MultiLayerKVCache::new(&config);
    let mut fresh_ctx = ForwardContext::new(&config);
    for pos in 0..4 {
        let _ = forward(
            &mut fresh_ctx,
            &weights,
            &mut fresh_cache,
            pos,
            pos,
            &config,
        );
    }

    let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
    let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

    for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "multi-layer restore should match fresh"
        );
    }
}

#[test]
fn test_snapshot_restore_gqa() {
    // Test with GQA config (kv_dim < n_embd)
    let config = Config::gqa_draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    for pos in 0..4 {
        let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
    }

    let snapshot = cache.snapshot(4, &config);
    let kd = types::kv_dim(&config);

    // Verify snapshot uses GQA kv_dim (smaller than n_embd)
    assert_eq!(kd, config.n_kv_head * config.head_dim);
    assert!(kd < config.n_embd, "GQA kv_dim should be < n_embd");
    for layer in &snapshot.layers {
        assert_eq!(layer.key.len(), 4 * kd);
    }

    // Restore and verify
    for pos in 4..8 {
        let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
    }
    cache.restore(&snapshot, &config);

    let mut fresh_cache = MultiLayerKVCache::new(&config);
    let mut fresh_ctx = ForwardContext::new(&config);
    for pos in 0..4 {
        let _ = forward(
            &mut fresh_ctx,
            &weights,
            &mut fresh_cache,
            pos,
            pos,
            &config,
        );
    }

    let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
    let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

    for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
        assert!((a - b).abs() < 1e-3, "GQA restore should match fresh");
    }
}

// ── forward_paged tests ──────────────────────────────────────

#[test]
fn test_forward_paged_logits_match_forward() {
    // forward_paged is a separate implementation that mirrors forward_base.
    // Under coda_fusion, forward() dispatches to forward_coda (fused kernels
    // with different float reassociation). Skip when the base path is altered.
    if !katgpt_forward::CPU_FORWARD_USES_DEVICE_BASE_PATH {
        return;
    }
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Flat cache forward
    let mut ctx_flat = ForwardContext::new(&config);
    let mut cache_flat = MultiLayerKVCache::new(&config);
    let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

    // Paged cache forward
    let mut ctx_paged = ForwardContext::new(&config);
    let mut cache_paged = PagedKVCache::new(&config, 1);
    let logits_paged = forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

    assert_eq!(logits_flat.len(), logits_paged.len());
    for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "forward_paged logit {i} differs: {a} vs {b}"
        );
    }
}

#[test]
fn test_forward_paged_logits_match_forward_multi_pos() {
    if !katgpt_forward::CPU_FORWARD_USES_DEVICE_BASE_PATH {
        return;
    }
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut ctx_flat = ForwardContext::new(&config);
    let mut cache_flat = MultiLayerKVCache::new(&config);

    let mut ctx_paged = ForwardContext::new(&config);
    let mut cache_paged = PagedKVCache::new(&config, 1);

    for pos in 0..4 {
        let token = pos; // simple: use pos as token
        let logits_flat = forward(
            &mut ctx_flat,
            &weights,
            &mut cache_flat,
            token,
            pos,
            &config,
        );
        let logits_paged = forward_paged(
            &mut ctx_paged,
            &weights,
            &mut cache_paged,
            0,
            token,
            pos,
            &config,
        );

        for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "pos {pos} logit {i} differs: {a} vs {b}"
            );
        }
    }
}

#[test]
fn test_forward_paged_gqa_logits_match() {
    if !katgpt_forward::CPU_FORWARD_USES_DEVICE_BASE_PATH {
        return;
    }
    let config = Config::gqa_draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut ctx_flat = ForwardContext::new(&config);
    let mut cache_flat = MultiLayerKVCache::new(&config);
    let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

    let mut ctx_paged = ForwardContext::new(&config);
    let mut cache_paged = PagedKVCache::new(&config, 1);
    let logits_paged = forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

    assert_eq!(logits_flat.len(), logits_paged.len());
    for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
        // Threshold accounts for FP accumulation-order differences between
        // the flat and paged matmul reductions (different tiling → different
        // rounding). 2e-3 is tight enough to catch real layout bugs while
        // tolerating weight-init-dependent reduction variance.
        assert!(
            (a - b).abs() < 2e-3,
            "GQA forward_paged logit {i} differs: {a} vs {b}"
        );
    }
}

#[test]
fn test_forward_paged_output_size() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = PagedKVCache::new(&config, 1);
    let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_paged_logits_finite() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = PagedKVCache::new(&config, 1);
    let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit {i} is not finite: {l}");
    }
}

// ── Rollback tests ─────────────────────────────────────────────

#[test]
fn test_paged_rollback_frees_exclusive_pages() {
    let config = Config::micro();
    let mut paged = PagedKVCache::new(&config, 2);

    // Allocate pages for seq 0 up to pos 31 (2 pages: 0..15, 16..31)
    paged.ensure_pages(0, 31);
    let seq0_pages_len = paged.layer_page_tables[0][0].len();
    assert!(seq0_pages_len >= 2, "seq 0 should have at least 2 pages");

    // Rollback seq 0 to pos 0 — all pages are exclusive (no other seq)
    paged.rollback(0, 0);

    // Page table should be truncated
    assert!(
        paged.layer_page_tables[0][0].is_empty(),
        "seq 0 page table should be empty after rollback to pos 0"
    );
    // All pages should be freed (they were exclusive)
    assert!(
        !paged.free_pages.is_empty(),
        "exclusive pages should be returned to free list"
    );
}

#[test]
fn test_paged_rollback_preserves_shared_pages() {
    let config = Config::micro();
    let mut paged = PagedKVCache::new(&config, 4);

    // Allocate pages for seq 0 up to pos 31
    paged.ensure_pages(0, 31);
    let _initial_pages_len = paged.layer_page_tables[0][0].len();

    // Fork a new sequence from seq 0 at pos 16 — shares first page
    // (fork returns layer_page_tables[0].len(), which may be > 1 if max_sequences > 1)
    let seq1 = paged.fork(0, 16);
    assert_ne!(seq1, 0, "fork should return a new sequence index");

    // Allocate exclusive pages for seq 0 beyond fork point
    paged.ensure_pages(0, 47); // extra pages after pos 31

    let free_before = paged.free_pages.len();
    let pages_before_rollback = paged.layer_page_tables[0][0].len();

    // Rollback seq 0 to pos 16 — keeps shared page, frees exclusive ones
    paged.rollback(0, 16);

    // Page table should be truncated to 1 page (covers 0..15)
    assert_eq!(
        paged.layer_page_tables[0][0].len(),
        1,
        "seq 0 should have 1 page after rollback to pos 16 (page covers 0..15)"
    );

    // Some pages should have been freed (the exclusive ones beyond page 0)
    let freed = paged.free_pages.len() - free_before;
    assert!(
        freed > 0,
        "exclusive pages beyond rollback point should be freed"
    );

    // But NOT more than what was removed from page table
    let removed = pages_before_rollback - 1;
    assert!(
        freed <= removed,
        "freed pages ({freed}) should not exceed removed pages ({removed})"
    );
}

#[test]
fn test_paged_rollback_shared_page_not_freed() {
    let config = Config::micro();
    let mut paged = PagedKVCache::new(&config, 4);

    // Allocate pages for seq 0
    paged.ensure_pages(0, 31);

    // Fork seq 1 at pos 0 — shares nothing initially (fork_page = 0)
    let seq1 = paged.fork(0, 0);

    // Allocate different pages for seq 1
    paged.ensure_pages(seq1, 31);

    // Now fork seq 2 from seq 0 at pos 16 — shares first page with seq 0
    let seq2 = paged.fork(0, 16);
    let shared_page_idx = paged.layer_page_tables[0][0][0];

    // Rollback seq 2 to pos 0 — the shared page should NOT be freed
    let _free_before = paged.free_pages.len();
    paged.rollback(seq2, 0);

    // Shared page should still be in seq 0's page table
    assert!(
        paged.layer_page_tables[0][0].contains(&shared_page_idx),
        "shared page should still be referenced by seq 0"
    );
    // Shared page should NOT be in free list
    assert!(
        !paged.free_pages.contains(&shared_page_idx),
        "shared page should not be freed"
    );
}

#[test]
fn test_paged_rollback_truncates_page_table() {
    let config = Config::micro();
    let mut paged = PagedKVCache::new(&config, 1);

    // Allocate 4 pages worth of positions
    paged.ensure_pages(0, 63);
    assert!(
        paged.layer_page_tables[0][0].len() >= 4,
        "should have at least 4 pages for pos 0..63"
    );

    // Rollback to pos 32 — should keep 2 pages (0..15, 16..31)
    paged.rollback(0, 32);
    assert_eq!(
        paged.layer_page_tables[0][0].len(),
        2,
        "should have exactly 2 pages after rollback to pos 32"
    );

    // Rollback to pos 16 — should keep 1 page (0..15)
    paged.rollback(0, 16);
    assert_eq!(
        paged.layer_page_tables[0][0].len(),
        1,
        "should have exactly 1 page after rollback to pos 16"
    );
}

#[test]
fn test_paged_rollback_all_layers_consistent() {
    let mut config = Config::micro();
    config.n_layer = 4;
    let mut paged = PagedKVCache::new(&config, 1);

    // Allocate pages for all layers
    paged.ensure_pages(0, 31);

    // Rollback to pos 16
    paged.rollback(0, 16);

    // All layers should have the same page table length
    let expected = 1; // 1 page covers 0..15
    for (layer_idx, lt) in paged.layer_page_tables.iter().enumerate() {
        assert_eq!(
            lt[0].len(),
            expected,
            "layer {layer_idx} should have {expected} pages after rollback"
        );
    }
}

// ======================================================================
// Sparse MLP tests (Plan 022: TwELL-inspired)
// ======================================================================

/// Sparse matmul produces identical output to dense at 0% sparsity (all alive).
#[cfg(feature = "sparse_mlp")]
#[test]
fn test_sparse_matmul_0_percent_sparsity() {
    let rows = 16;
    let cols = 64;
    let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let mut dense_out = vec![0.0f32; rows];
    let mut sparse_out = vec![0.0f32; rows];
    let mut indices = vec![0usize; cols];
    let mut values = vec![0.0f32; cols];

    crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
    crate::types::sparse_matmul(
        &mut sparse_out,
        &weight,
        &input,
        rows,
        cols,
        &mut indices,
        &mut values,
    );

    for i in 0..rows {
        assert!(
            (dense_out[i] - sparse_out[i]).abs() < 1e-3,
            "Mismatch at {i}: dense={}, sparse={}",
            dense_out[i],
            sparse_out[i]
        );
    }
}

/// Sparse matmul produces identical output at 95% sparsity.
#[cfg(feature = "sparse_mlp")]
#[test]
fn test_sparse_matmul_95_percent_sparsity() {
    let rows = 16;
    let cols = 64;
    let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
    let mut input = vec![0.0f32; cols];
    // 5% alive
    for i in (0..cols).step_by(20) {
        input[i] = 1.0;
    }
    let mut dense_out = vec![0.0f32; rows];
    let mut sparse_out = vec![0.0f32; rows];
    let mut indices = vec![0usize; cols];
    let mut values = vec![0.0f32; cols];

    crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
    crate::types::sparse_matmul(
        &mut sparse_out,
        &weight,
        &input,
        rows,
        cols,
        &mut indices,
        &mut values,
    );

    for i in 0..rows {
        assert!(
            (dense_out[i] - sparse_out[i]).abs() < 1e-4,
            "Mismatch at {i}: dense={}, sparse={}",
            dense_out[i],
            sparse_out[i]
        );
    }
}

/// Sparse matmul with 100% sparsity (all zeros) produces all-zero output.
#[cfg(feature = "sparse_mlp")]
#[test]
fn test_sparse_matmul_100_percent_sparsity() {
    let rows = 16;
    let cols = 64;
    let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
    let input = vec![0.0f32; cols];
    let mut sparse_out = vec![0.0f32; rows];
    let mut indices = vec![0usize; cols];
    let mut values = vec![0.0f32; cols];

    let alive = crate::types::sparse_matmul(
        &mut sparse_out,
        &weight,
        &input,
        rows,
        cols,
        &mut indices,
        &mut values,
    );

    assert_eq!(alive, 0, "Expected 0 alive neurons");
    for (i, &val) in sparse_out.iter().take(rows).enumerate() {
        assert_eq!(val, 0.0, "Expected zero output at {i}");
    }
}

/// ForwardContext buffers are correctly sized when sparse_mlp is enabled.
#[cfg(feature = "sparse_mlp")]
#[test]
fn test_forward_context_sparse_buffers() {
    let config = crate::types::Config::micro();
    let ctx = super::ForwardContext::new(&config);
    assert_eq!(ctx.active_indices.len(), config.mlp_hidden);
    assert_eq!(ctx.active_values.len(), config.mlp_hidden);
}

/// Forward pass works correctly with sparse_mlp enabled.
#[cfg(feature = "sparse_mlp")]
#[test]
fn test_forward_with_sparse_mlp() {
    let config = crate::types::Config::micro();
    let mut rng = crate::types::Rng::new(42);
    let weights = crate::transformer::TransformerWeights::new(&config, &mut rng);
    let mut ctx = crate::transformer::ForwardContext::new(&config);
    let mut cache = crate::transformer::MultiLayerKVCache::new(&config);

    let logits = crate::transformer::forward(&mut ctx, &weights, &mut cache, 26, 0, &config);

    // Verify logits are finite
    for l in logits {
        assert!(l.is_finite(), "Logit is not finite: {l}");
    }
}

/// Sparse matmul with negative values (should be treated as dead by ReLU context).
#[cfg(feature = "sparse_mlp")]
#[test]
fn test_sparse_matmul_negative_input() {
    let rows = 8;
    let cols = 32;
    let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
    let mut input = vec![0.0f32; cols];
    // Mix of positive, negative, zero
    input[0] = 1.0;
    input[1] = -1.0; // Should be ignored (not > 0)
    input[2] = 0.5;
    input[3] = -0.5; // Should be ignored
    // Rest are 0.0

    let mut dense_out = vec![0.0f32; rows];
    let mut sparse_out = vec![0.0f32; rows];
    let mut indices = vec![0usize; cols];
    let mut values = vec![0.0f32; cols];

    crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
    crate::types::sparse_matmul(
        &mut sparse_out,
        &weight,
        &input,
        rows,
        cols,
        &mut indices,
        &mut values,
    );

    // Both should match since matmul doesn't skip negatives but sparse_matmul skips input[c] <= 0
    // So we need to compare against a modified dense that also skips negatives
    for r in 0..rows {
        let mut expected = 0.0f32;
        for c in 0..cols {
            if input[c] > 0.0 {
                expected += weight[r * cols + c] * input[c];
            }
        }
        assert!(
            (sparse_out[r] - expected).abs() < 1e-4,
            "Mismatch at {r}: sparse={}, expected={}",
            sparse_out[r],
            expected
        );
    }
}

// -----------------------------------------------------------------------
// Plan 025: Bidirectional Prefill + Modality LoRA Switching
// -----------------------------------------------------------------------

#[test]
fn test_forward_prefill_logits_finite() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens: Vec<usize> = (0..8).collect();
    #[cfg(not(feature = "domain_latent"))]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    assert_eq!(logits.len(), config.vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "prefill logit {i} is not finite: {l}");
    }
}

#[test]
fn test_forward_prefill_populates_cache() {
    let config = Config::micro();
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens: Vec<usize> = (0..5).collect();
    #[cfg(not(feature = "domain_latent"))]
    forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    // All 5 positions should have K/V in cache
    for p in 0..5 {
        let off = p * kvd;
        let key_sum: f32 = cache.layers[0].key[off..off + kvd].iter().sum();
        let val_sum: f32 = cache.layers[0].value[off..off + kvd].iter().sum();
        assert!(key_sum != 0.0, "K cache at pos {p} should be populated");
        assert!(val_sum != 0.0, "V cache at pos {p} should be populated");
    }
}

#[test]
fn test_forward_prefill_logits_shape() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens: Vec<usize> = vec![0, 1, 2];
    #[cfg(not(feature = "domain_latent"))]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_forward_prefill_single_token() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens = vec![5];
    #[cfg(not(feature = "domain_latent"))]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    assert_eq!(logits.len(), config.vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(
            l.is_finite(),
            "single-token prefill logit {i} not finite: {l}"
        );
    }
}

#[test]
fn test_prefill_then_decode_shared_cache() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // Prefill with 4 tokens
    let prompt: Vec<usize> = (0..4).collect();
    #[cfg(not(feature = "domain_latent"))]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &prompt,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &prompt,
        &config,
        None,
        None,
    );
    assert_eq!(logits.len(), config.vocab_size);

    // Decode from position 4 (should use same cache)
    let logits2 = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
    assert_eq!(logits2.len(), config.vocab_size);
    for (i, &l) in logits2.iter().enumerate() {
        assert!(
            l.is_finite(),
            "decode after prefill logit {i} not finite: {l}"
        );
    }
}

#[test]
fn test_no_lora_matches_existing_forward() {
    // Under coda_fusion, forward() dispatches to forward_coda, not forward_base.
    // The two implementations differ by float reassociation (~1e-6).
    // Skip when the base path is altered.
    if !katgpt_forward::CPU_FORWARD_USES_DEVICE_BASE_PATH {
        return;
    }
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Existing forward (no LoRA)
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config);

    // New forward_base with None (should be identical)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    #[cfg(not(feature = "domain_latent"))]
    let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None);
    #[cfg(feature = "domain_latent")]
    let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None, None);

    for i in 0..config.vocab_size {
        let diff = (logits1[i] - logits2[i]).abs();
        assert!(
            diff < 5e-6,
            "forward and forward_base(None) differ at {i}: {diff}"
        );
    }
}

#[test]
fn test_generate_with_prefill_produces_tokens() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let prompt: Vec<usize> = (0..4).collect();
    let generated = {
        #[cfg(not(feature = "domain_latent"))]
        {
            generate_with_prefill(
                &mut ctx,
                &mut prefill,
                &weights,
                &mut cache,
                &config,
                &mut rng,
                &prompt,
                10,
                &crate::types::LoraPair::none(),
            )
        }
        #[cfg(feature = "domain_latent")]
        {
            generate_with_prefill(
                &mut ctx,
                &mut prefill,
                &weights,
                &mut cache,
                &config,
                &mut rng,
                &prompt,
                10,
                &crate::types::LoraPair::none(),
                None,
            )
        }
    };

    assert!(!generated.is_empty(), "should generate at least one token");
    assert!(generated.len() <= 10, "should not exceed max_gen_tokens");
    for (i, &t) in generated.iter().enumerate() {
        assert!(t < config.vocab_size, "token {i} out of range: {t}");
    }
}

// -----------------------------------------------------------------------
// Multi-layer prefill tests
// -----------------------------------------------------------------------

#[cfg(feature = "domain_latent")]
#[test]
fn test_generate_with_prefill_domain_latent() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);

    // Create a non-zero domain latent
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);

    let prompt: Vec<usize> = (0..4).collect();

    // Generate without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut prefill1 = PrefillContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let mut rng1 = Rng::new(42);
    let generated1 = generate_with_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights,
        &mut cache1,
        &config,
        &mut rng1,
        &prompt,
        10,
        &crate::types::LoraPair::none(),
        None,
    );

    // Generate with domain latent (same seed)
    let mut ctx2 = ForwardContext::new(&config);
    let mut prefill2 = PrefillContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let mut rng2 = Rng::new(42);
    let generated2 = generate_with_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights,
        &mut cache2,
        &config,
        &mut rng2,
        &prompt,
        10,
        &crate::types::LoraPair::none(),
        Some(&dl),
    );

    // Outputs should differ — domain latent modulates K/V at mid-layer
    assert_ne!(
        generated1, generated2,
        "domain latent should change generation output"
    );
}

fn small_target_2layer() -> Config {
    let mut c = Config::small_target();
    c.n_layer = 2;
    c
}

#[test]
fn test_forward_prefill_multilayer_logits_finite() {
    let config = small_target_2layer();
    config.validate().unwrap();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens: Vec<usize> = (0..8).collect();
    #[cfg(not(feature = "domain_latent"))]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    let logits = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    assert_eq!(logits.len(), config.vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(
            l.is_finite(),
            "multilayer prefill logit {i} not finite: {l}"
        );
    }
}

#[test]
fn test_forward_prefill_multilayer_cache_populated() {
    let config = small_target_2layer();
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let tokens: Vec<usize> = (0..4).collect();
    #[cfg(not(feature = "domain_latent"))]
    forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
    );
    #[cfg(feature = "domain_latent")]
    forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &tokens,
        &config,
        None,
        None,
    );
    // Both layers should have K/V populated
    for (layer, layer_cache) in cache.layers[..2].iter().enumerate() {
        for p in 0..4 {
            let off = p * kvd;
            let key_sum: f32 = layer_cache.key[off..off + kvd].iter().sum();
            let val_sum: f32 = layer_cache.value[off..off + kvd].iter().sum();
            assert!(
                key_sum != 0.0,
                "layer {layer} K cache at pos {p} should be populated"
            );
            assert!(
                val_sum != 0.0,
                "layer {layer} V cache at pos {p} should be populated"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Domain Latent injection (Plan 038)
// -----------------------------------------------------------------------

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_changes_logits() {
    let config = small_target_2layer(); // 2 layers, mid-layer = layer 1
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);

    // Without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

    // With domain latent (non-zero embedding)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        None,
        Some(&dl),
    );

    // Logits should differ — domain latent modulates K/V at mid-layer
    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff, "domain latent should change logits");
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_zero_embedding_same_logits() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);

    // Without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

    // With zero domain latent — should be identical
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let dl = crate::types::DomainLatent::zeros(kvd);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        None,
        Some(&dl),
    );

    for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "zero domain latent should not change logits, diff at {i}: {diff}"
        );
    }
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_prefill_changes_logits() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let tokens: Vec<usize> = (0..4).collect();

    // Without domain latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut prefill1 = PrefillContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights,
        &mut cache1,
        &tokens,
        &config,
        None,
        None,
    );

    // With domain latent
    let mut ctx2 = ForwardContext::new(&config);
    let mut prefill2 = PrefillContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.3; kvd]);
    let logits2 = forward_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights,
        &mut cache2,
        &tokens,
        &config,
        None,
        Some(&dl),
    );

    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff, "domain latent in prefill should change logits");
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_prefill_then_decode() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.2; kvd]);

    // Prefill with domain latent
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let prompt: Vec<usize> = (0..3).collect();
    let logits_prefill = forward_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &prompt,
        &config,
        None,
        Some(&dl),
    );
    assert_eq!(logits_prefill.len(), config.vocab_size);
    for (i, &l) in logits_prefill.iter().enumerate() {
        assert!(
            l.is_finite(),
            "prefill with domain_latent logit {i} not finite: {l}"
        );
    }

    // Decode with domain latent (position 3)
    let logits_decode = forward_base(
        &mut ctx,
        &weights,
        &mut cache,
        0,
        3,
        &config,
        None,
        Some(&dl),
    );
    assert_eq!(logits_decode.len(), config.vocab_size);
    for (i, &l) in logits_decode.iter().enumerate() {
        assert!(
            l.is_finite(),
            "decode after prefill with domain_latent logit {i} not finite: {l}"
        );
    }
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_forward_with_domain_latent_wrapper() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let dl = crate::types::DomainLatent::from_vec(vec![0.1; kvd]);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let logits = forward_with_domain_latent(
        &mut ctx,
        &weights,
        &mut cache,
        0,
        0,
        &config,
        None,
        Some(&dl),
    );
    assert_eq!(logits.len(), config.vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "logit {i} not finite: {l}");
    }
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_with_lora_changes_logits() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let rank = 4;
    let in_dim = config.n_embd;
    let out_dim = config.n_embd;

    let lora = crate::types::LoraAdapter {
        a: vec![0.1f32; rank * in_dim],
        b: vec![0.1f32; out_dim * rank],
        rank,
        alpha: 8.0,
        in_dim,
        out_dim,
    };
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);

    // With both lora + domain_latent
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(
        &mut ctx1,
        &weights,
        &mut cache1,
        0,
        0,
        &config,
        Some(&lora),
        Some(&dl),
    );

    // With lora only (no domain_latent)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        Some(&lora),
        None,
    );

    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(
        any_diff,
        "domain_latent + lora should differ from lora-only"
    );
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_with_lora_prefill_pipeline() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let rank = 4;
    let in_dim = config.n_embd;
    let out_dim = config.n_embd;

    let lora = crate::types::LoraAdapter {
        a: vec![0.1f32; rank * in_dim],
        b: vec![0.1f32; out_dim * rank],
        rank,
        alpha: 8.0,
        in_dim,
        out_dim,
    };
    let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
    let tokens: Vec<usize> = (0..3).collect();

    // Pipeline 1: prefill + decode with both lora + dl
    let mut ctx1 = ForwardContext::new(&config);
    let mut prefill1 = PrefillContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let _ = forward_prefill(
        &mut ctx1,
        &mut prefill1,
        &weights,
        &mut cache1,
        &tokens,
        &config,
        Some(&lora),
        Some(&dl),
    );
    let logits1 = forward_base(
        &mut ctx1,
        &weights,
        &mut cache1,
        0,
        tokens.len(),
        &config,
        Some(&lora),
        Some(&dl),
    );

    // Pipeline 2: prefill + decode with lora only
    let mut ctx2 = ForwardContext::new(&config);
    let mut prefill2 = PrefillContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let _ = forward_prefill(
        &mut ctx2,
        &mut prefill2,
        &weights,
        &mut cache2,
        &tokens,
        &config,
        Some(&lora),
        None,
    );
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        tokens.len(),
        &config,
        Some(&lora),
        None,
    );

    let mut any_diff = false;
    for (&a, &b) in logits1.iter().zip(logits2.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(
        any_diff,
        "prefill+decode with lora+dl should differ from lora-only pipeline"
    );
}

#[cfg(feature = "domain_latent")]
#[test]
fn test_domain_latent_zero_with_lora_same_as_lora_only() {
    let config = small_target_2layer();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let kvd = crate::types::kv_dim(&config);
    let rank = 4;
    let in_dim = config.n_embd;
    let out_dim = config.n_embd;

    let lora = crate::types::LoraAdapter {
        a: vec![0.1f32; rank * in_dim],
        b: vec![0.1f32; out_dim * rank],
        rank,
        alpha: 8.0,
        in_dim,
        out_dim,
    };
    let dl_zero = crate::types::DomainLatent::zeros(kvd);

    // With zero domain_latent + lora
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward_base(
        &mut ctx1,
        &weights,
        &mut cache1,
        0,
        0,
        &config,
        Some(&lora),
        Some(&dl_zero),
    );

    // With lora only (no domain_latent)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let logits2 = forward_base(
        &mut ctx2,
        &weights,
        &mut cache2,
        0,
        0,
        &config,
        Some(&lora),
        None,
    );

    for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "zero domain_latent + lora should match lora-only, diff at {i}: {diff}"
        );
    }
}

// ── Shared KV Cache (Phase 3, Plan 055) ─────────────────────

#[test]
fn test_preload_kv_cache_dimension_mismatch() {
    // bpe: n_kv_head=4, head_dim=8 → kv_dim=32
    // bpe_draft: n_kv_head=2, head_dim=8 → kv_dim=16
    let target_config = Config::bpe();
    let draft_config = Config::bpe_draft();

    let target_cache = MultiLayerKVCache::new(&target_config);
    let mut draft_cache = MultiLayerKVCache::new(&draft_config);

    // Preload should silently skip (kv_dim mismatch)
    preload_kv_cache(
        &mut draft_cache,
        &target_cache,
        1,
        &target_config,
        &draft_config,
    );

    // Draft cache should remain all zeros
    for layer in &draft_cache.layers {
        assert!(
            layer.key.iter().all(|&v| v == 0.0),
            "draft cache key should remain zero on dim mismatch"
        );
        assert!(
            layer.value.iter().all(|&v| v == 0.0),
            "draft cache value should remain zero on dim mismatch"
        );
    }
}

#[test]
fn test_preload_kv_cache_matching_dims() {
    // Same config for both → kv_dim matches
    let config = Config::small_target();
    let kvd = crate::types::kv_dim(&config);

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Populate target cache at pos 0 and pos 1
    let mut target_cache = MultiLayerKVCache::new(&config);
    let mut target_ctx = ForwardContext::new(&config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 1, 1, &config);

    // Create empty draft cache
    let mut draft_cache = MultiLayerKVCache::new(&config);

    // Preload positions [0..2) from target
    preload_kv_cache(&mut draft_cache, &target_cache, 2, &config, &config);

    // Verify draft cache has target's KV for positions 0 and 1
    for (layer_idx, draft_layer) in draft_cache.layers.iter().enumerate() {
        let target_layer = &target_cache.layers[layer_idx];
        let copy_len = 2 * kvd;
        for i in 0..copy_len {
            assert_eq!(
                draft_layer.key[i], target_layer.key[i],
                "draft key mismatch at layer {layer_idx}, idx {i}"
            );
            assert_eq!(
                draft_layer.value[i], target_layer.value[i],
                "draft value mismatch at layer {layer_idx}, idx {i}"
            );
        }
    }
}

#[test]
fn test_preload_kv_cache_zero_pos() {
    let config = Config::small_target();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut target_cache = MultiLayerKVCache::new(&config);
    let mut target_ctx = ForwardContext::new(&config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);

    let mut draft_cache = MultiLayerKVCache::new(&config);

    // Preload with pos=0 copies nothing (no positions to share)
    preload_kv_cache(&mut draft_cache, &target_cache, 0, &config, &config);

    // Draft cache should remain all zeros
    for layer in &draft_cache.layers {
        assert!(
            layer.key.iter().all(|&v| v == 0.0),
            "draft cache should remain zero with pos=0"
        );
    }
}

#[test]
fn test_preload_kv_cache_fewer_draft_layers() {
    // Target: 2 layers, Draft: 1 layer — only layer 0 shared
    let target_config = Config {
        n_layer: 2,
        ..Config::small_target()
    };
    let draft_config = Config {
        n_layer: 1,
        ..Config::small_target()
    };

    let kvd = crate::types::kv_dim(&target_config);
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&target_config, &mut rng);

    let mut target_cache = MultiLayerKVCache::new(&target_config);
    let mut target_ctx = ForwardContext::new(&target_config);
    let _ = forward(
        &mut target_ctx,
        &target_weights,
        &mut target_cache,
        0,
        0,
        &target_config,
    );

    let mut draft_cache = MultiLayerKVCache::new(&draft_config);

    preload_kv_cache(
        &mut draft_cache,
        &target_cache,
        1,
        &target_config,
        &draft_config,
    );

    // Draft has 1 layer, only layer 0 should be copied
    assert_eq!(draft_cache.layers.len(), 1);
    let draft_layer = &draft_cache.layers[0];
    let target_layer = &target_cache.layers[0];
    for i in 0..kvd {
        assert_eq!(
            draft_layer.key[i], target_layer.key[i],
            "layer 0 key should be copied"
        );
        assert_eq!(
            draft_layer.value[i], target_layer.value[i],
            "layer 0 value should be copied"
        );
    }
}

/// T14: Verify hybrid behavior — drafter forwards with preloaded target KV.
/// Past positions [0..pos) read from preloaded target KV,
/// new position [pos] computed by drafter and written to its own cache.
#[test]
fn test_preload_kv_cache_hybrid_forward() {
    let config = Config::small_target();
    let kvd = crate::types::kv_dim(&config);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Build target KV cache for positions 0 and 1
    let mut target_cache = MultiLayerKVCache::new(&config);
    let mut target_ctx = ForwardContext::new(&config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);
    let _ = forward(&mut target_ctx, &weights, &mut target_cache, 1, 1, &config);

    // Preload target KV [0..2) into draft cache
    let mut draft_cache = MultiLayerKVCache::new(&config);
    preload_kv_cache(&mut draft_cache, &target_cache, 2, &config, &config);

    // Drafter forwards at pos=2 with preloaded KV — should produce valid logits
    let mut draft_ctx = ForwardContext::new(&config);
    let logits = forward(&mut draft_ctx, &weights, &mut draft_cache, 2, 2, &config);

    // Logits must be finite (no NaN/Inf from garbage KV)
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] not finite: {v}");
    }

    // Draft cache now has: [0..2) from target, [2] from drafter
    for layer in &draft_cache.layers {
        // Position 2 should have non-zero KV (written by drafter)
        let pos2_off = 2 * kvd;
        let has_nonzero = layer.key[pos2_off..pos2_off + kvd]
            .iter()
            .any(|&v| v != 0.0);
        assert!(has_nonzero, "drafter should have written KV at pos 2");
    }
}

// --- T15–T19: Clustered LM Head Tests ---

#[test]
fn test_cluster_map_round_robin() {
    // 10 tokens, cluster_size=3 → 4 clusters: [0,1,2], [3,4,5], [6,7,8], [9]
    let map = cluster_map_round_robin(10, 3);
    assert_eq!(map.len(), 4);
    assert_eq!(map[0], vec![0, 1, 2]);
    assert_eq!(map[1], vec![3, 4, 5]);
    assert_eq!(map[2], vec![6, 7, 8]);
    assert_eq!(map[3], vec![9]);
}

#[test]
fn test_cluster_map_round_robin_exact_division() {
    // 8 tokens, cluster_size=4 → 2 clusters
    let map = cluster_map_round_robin(8, 4);
    assert_eq!(map.len(), 2);
    assert_eq!(map[0], vec![0, 1, 2, 3]);
    assert_eq!(map[1], vec![4, 5, 6, 7]);
}

#[test]
fn test_standard_lm_head_matches_matmul() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let n = config.n_embd;

    let mut logits_matmul = vec![0.0f32; config.vocab_size];
    let mut logits_standard = vec![0.0f32; config.vocab_size];
    let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    matmul(
        &mut logits_matmul,
        &weights.lm_head,
        &hidden,
        config.vocab_size,
        n,
    );
    standard_lm_head(
        &mut logits_standard,
        &hidden,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    for i in 0..config.vocab_size {
        let diff = (logits_matmul[i] - logits_standard[i]).abs();
        assert!(diff < 1e-6, "standard_lm_head differs at {i}: {diff}");
    }
}

#[test]
fn test_clustered_lm_head_only_cluster_tokens_finite() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    let n = config.n_embd;
    let cluster_size = 16;

    let cluster_map = cluster_map_round_robin(config.vocab_size, cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * n).map(|_| rng.normal()).collect();

    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map.clone());

    let mut logits = vec![0.0f32; config.vocab_size];
    let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    clustered_lm_head(
        &mut logits,
        &hidden,
        &weights.lm_head,
        weights.mtp_cluster_classifier.as_ref().unwrap(),
        weights.mtp_cluster_map.as_ref().unwrap(),
        config.vocab_size,
        n,
        1, // topk=1: backward compat (single cluster selection)
        &mut vec![0.0f32; config.vocab_size],
        &mut vec![(0usize, 0.0f32); config.vocab_size],
        &mut Vec::new(),
    );

    // Find winning cluster (the one with finite logits)
    let winning = cluster_map
        .iter()
        .find(|tokens| tokens.iter().all(|&t| logits[t].is_finite()))
        .expect("one cluster should have finite logits");

    // Cluster tokens: finite. Others: -inf
    let cluster_set: std::collections::HashSet<usize> = winning.iter().copied().collect();
    for (i, &logit) in logits.iter().enumerate() {
        if cluster_set.contains(&i) {
            assert!(logit.is_finite(), "token {i} in cluster should be finite");
        } else {
            assert_eq!(logit, f32::NEG_INFINITY, "token {i} should be -inf");
        }
    }
}

#[test]
fn test_clustered_lm_head_logits_match_standard() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    let n = config.n_embd;
    let cluster_size = 16;

    let cluster_map = cluster_map_round_robin(config.vocab_size, cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * n).map(|_| rng.normal()).collect();

    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map.clone());

    let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Standard logits
    let mut logits_std = vec![0.0f32; config.vocab_size];
    standard_lm_head(
        &mut logits_std,
        &hidden,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    // Clustered logits
    let mut logits_clust = vec![0.0f32; config.vocab_size];
    clustered_lm_head(
        &mut logits_clust,
        &hidden,
        &weights.lm_head,
        weights.mtp_cluster_classifier.as_ref().unwrap(),
        weights.mtp_cluster_map.as_ref().unwrap(),
        config.vocab_size,
        n,
        1, // topk=1: backward compat (single cluster selection)
        &mut vec![0.0f32; config.vocab_size],
        &mut vec![(0usize, 0.0f32); config.vocab_size],
        &mut Vec::new(),
    );

    // Find winning cluster
    let winning = cluster_map
        .iter()
        .find(|tokens| tokens.iter().all(|&t| logits_clust[t].is_finite()))
        .expect("one cluster should win");

    // Clustered logits for winning tokens should match standard exactly
    for &t in winning {
        let diff = (logits_clust[t] - logits_std[t]).abs();
        assert!(diff < 1e-5, "logit[{t}] mismatch: diff={diff}");
    }
}

#[test]
fn test_forward_base_clustered_dispatch() {
    // Config::bpe() has vocab=4096, threshold=4096 → 4096 >= 4096 activates
    // Use topk=1 so only 1 cluster is selected (produces -inf for non-cluster tokens)
    let mut config = Config::bpe();
    config.mtp_cluster_topk = 1;
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);

    let cluster_map = cluster_map_round_robin(config.vocab_size, config.mtp_cluster_size);
    let num_clusters = cluster_map.len();
    let classifier: Vec<f32> = (0..num_clusters * config.n_embd)
        .map(|_| rng.normal())
        .collect();
    weights.mtp_cluster_classifier = Some(classifier);
    weights.mtp_cluster_map = Some(cluster_map);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    // Clustered path active: some -inf, some finite
    let inf_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
    let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
    assert!(inf_count > 0, "should have -inf logits (clustered path)");
    assert!(
        finite_count > 0,
        "should have finite logits (cluster tokens)"
    );
    assert_eq!(inf_count + finite_count, config.vocab_size);
}

#[test]
fn test_forward_base_standard_fallback_no_weights() {
    // Config::micro() has threshold=usize::MAX → never activates clustered path
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

    // Standard path: all finite, no -inf
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] should be finite: {v}");
    }
}

/// Plan 574 changed `cluster_map_from_embeddings` from a round-robin stub into
/// real k-means, and this test originally asserted that all-identical rows
/// collapse into one mega-cluster.
///
/// `5528c2d9` reverted that outcome deliberately: a single cluster holding the
/// whole vocabulary is *degenerate*, not safe — `clustered_lm_head` then selects
/// it for any `topk >= 1` and computes the full vocabulary, so the pruning path
/// silently becomes a slower no-op. Constant projected geometry now falls back
/// to the baseline map instead, which is what k-means reduces to when there is
/// no geometry to exploit.
///
/// That commit missed this copy of the assertion (Issue 700 R2). Two other
/// tests pin the same contract — `cluster_build::tests::
/// constant_geometry_falls_back_to_round_robin` in the owning crate and
/// riir-ai's `test_cluster_map_from_embeddings_fallback` — so this one is
/// written against `cluster_map_round_robin` too, keeping all three in step.
#[test]
fn test_cluster_map_from_embeddings_falls_back_on_identical_rows() {
    let lm_head = vec![0.0f32; 100 * 32];
    let map = cluster_map_from_embeddings(&lm_head, 100, 32, 25);

    assert_eq!(
        map,
        cluster_map_round_robin(100, 25),
        "constant geometry must fall back to the baseline map, not one mega-cluster"
    );
    // Coverage is the invariant the old assertion was really guarding.
    let mut seen: Vec<usize> = map.iter().flatten().copied().collect();
    seen.sort_unstable();
    assert!(
        seen.iter().copied().eq(0..100),
        "every token must still be covered"
    );
}

/// A short weight buffer must fall back rather than panic.
#[test]
fn test_cluster_map_from_embeddings_short_buffer_falls_back() {
    let truncated = vec![0.0f32; 4];
    let map = cluster_map_from_embeddings(&truncated, 100, 32, 25);
    assert_eq!(map, cluster_map_round_robin(100, 25));
}

// ── Delta routing stability tests (Plan 134 T2) ─────────────

/// GOAT proof: verifies that `depth_route` norm stability holds empirically
/// across 36 simulated layers. See `depth_route` doc comment for the
/// theoretical argument (Plan 134 T1/T3, MGR §3.2).
#[test]
#[cfg(feature = "delta_routing")]
fn proof_depth_route_norm_stability() {
    let n_embd = 32;
    let n_sources = 4;

    // Create initial residual (simulating embedding output)
    let mut residual: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.1).sin()).collect();
    let initial_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Create synthetic sources (layer deltas), query weights, norm weights
    let sources: Vec<Vec<f32>> = (0..n_sources)
        .map(|s| {
            (0..n_embd)
                .map(|i| ((i + s * 7) as f32 * 0.05).cos() * 0.1)
                .collect()
        })
        .collect();
    let source_refs: Vec<&[f32]> = sources.iter().map(|s| s.as_slice()).collect();
    let query_weight: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.1).sin() * 0.01).collect();
    let norm_weight: Vec<f32> = vec![1.0; n_embd];
    let mut logits_buf = vec![0.0f32; n_sources];
    let mut scaled_buf = vec![0.0f32; n_embd];

    // Simulate 36 layers of additive routing
    for _ in 0..36 {
        depth_route(
            &mut residual,
            &source_refs,
            &query_weight,
            &norm_weight,
            &mut logits_buf,
            &mut scaled_buf,
            n_embd,
        );
    }

    let final_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        final_norm <= 10.0 * initial_norm,
        "Norm grew beyond 10x: initial={}, final={}, ratio={}",
        initial_norm,
        final_norm,
        final_norm / initial_norm,
    );
}

// ── Kog CPU fusion GOAT proofs (Plan 160) ───────────────────

/// GOAT proof (T5): folded gamma weights produce identical forward pass output.
///
/// Strategy: create weights with non-trivial gamma, run forward with unfolded gamma,
/// then fold gamma and run forward again — assert bit-identical output.
///
/// Only MLP gamma is folded (attention gamma is kept at runtime due to residual pattern).
#[test]
fn proof_gamma_folding_forward_base() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);

    // Set non-trivial gamma (not all 1.0)
    for layer in &mut weights.layers {
        for (i, g) in layer.attn_norm_gamma.iter_mut().enumerate() {
            *g = 0.5 + (i as f32 * 0.1).sin() * 0.8;
        }
        for (i, g) in layer.mlp_norm_gamma.iter_mut().enumerate() {
            *g = 0.5 + (i as f32 * 0.15).cos() * 0.6;
        }
    }

    // Capture gamma values before folding
    let attn_gammas: Vec<Vec<f32>> = weights
        .layers
        .iter()
        .map(|l| l.attn_norm_gamma.clone())
        .collect();
    let mlp_gammas: Vec<Vec<f32>> = weights
        .layers
        .iter()
        .map(|l| l.mlp_norm_gamma.clone())
        .collect();

    let n = config.n_embd;
    let kvd = types::kv_dim(&config);

    // ── Baseline: forward with unfolded gamma ──
    let mut ctx1 = ForwardContext::new(&config);
    let _cache1 = MultiLayerKVCache::new(&config);

    let tok_off = 0;
    let pos_off_emb = 0;
    katgpt_core::simd::simd_add_into(
        &mut ctx1.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    for (li, layer_weights) in weights.layers.iter().enumerate() {
        // Attention: rmsnorm_with_gamma → save residual → QKV
        types::rmsnorm_with_gamma(&mut ctx1.x[..n], &attn_gammas[li]);
        ctx1.xr[..n].copy_from_slice(&ctx1.x[..n]);
        types::matmul(&mut ctx1.q, &layer_weights.attn_wq, &ctx1.x, n, n);
        types::matmul(&mut ctx1.k, &layer_weights.attn_wk, &ctx1.x, kvd, n);
        types::matmul(&mut ctx1.v, &layer_weights.attn_wv, &ctx1.x, kvd, n);
        // Output projection + residual
        types::matmul(&mut ctx1.x, &layer_weights.attn_wo, &ctx1.attn_out, n, n);
        katgpt_core::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr[..n]);
        // MLP: save pre-norm residual → rmsnorm_with_gamma → MLP
        ctx1.xr2[..n].copy_from_slice(&ctx1.x[..n]);
        types::rmsnorm_with_gamma(&mut ctx1.x[..n], &mlp_gammas[li]);
        types::matmul_relu(
            &mut ctx1.hidden,
            &layer_weights.mlp_w1,
            &ctx1.x,
            config.mlp_hidden,
            n,
        );
        types::matmul(
            &mut ctx1.x,
            &layer_weights.mlp_w2,
            &ctx1.hidden,
            n,
            config.mlp_hidden,
        );
        katgpt_core::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr2[..n]);
    }

    let baseline_hidden: Vec<f32> = ctx1.x[..n].to_vec();

    // ── Fold MLP gamma into mlp_w1 ──
    weights.fold_gamma(&config);

    // ── Folded: forward with attn gamma at runtime, mlp gamma folded ──
    let mut ctx2 = ForwardContext::new(&config);
    let _cache2 = MultiLayerKVCache::new(&config);

    katgpt_core::simd::simd_add_into(
        &mut ctx2.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    for (li, layer_weights) in weights.layers.iter().enumerate() {
        // Attention: still uses rmsnorm_with_gamma (gamma not folded)
        types::rmsnorm_with_gamma(&mut ctx2.x[..n], &attn_gammas[li]);
        ctx2.xr[..n].copy_from_slice(&ctx2.x[..n]);
        types::matmul(&mut ctx2.q, &layer_weights.attn_wq, &ctx2.x, n, n);
        types::matmul(&mut ctx2.k, &layer_weights.attn_wk, &ctx2.x, kvd, n);
        types::matmul(&mut ctx2.v, &layer_weights.attn_wv, &ctx2.x, kvd, n);
        types::matmul(&mut ctx2.x, &layer_weights.attn_wo, &ctx2.attn_out, n, n);
        katgpt_core::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr[..n]);
        // MLP: gamma folded into w1, so plain rmsnorm (gamma is now identity)
        ctx2.xr2[..n].copy_from_slice(&ctx2.x[..n]);
        rmsnorm(&mut ctx2.x);
        types::matmul_relu(
            &mut ctx2.hidden,
            &layer_weights.mlp_w1,
            &ctx2.x,
            config.mlp_hidden,
            n,
        );
        types::matmul(
            &mut ctx2.x,
            &layer_weights.mlp_w2,
            &ctx2.hidden,
            n,
            config.mlp_hidden,
        );
        katgpt_core::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr2[..n]);
    }

    let folded_hidden: Vec<f32> = ctx2.x[..n].to_vec();

    // GOAT assertion: bit-identical (within FP tolerance)
    for i in 0..n {
        let diff = (baseline_hidden[i] - folded_hidden[i]).abs();
        assert!(
            diff < 1e-5,
            "GOAT FAIL: gamma fold mismatch at [{i}]: baseline={}, folded={}, diff={}",
            baseline_hidden[i],
            folded_hidden[i],
            diff
        );
    }
}

/// GOAT proof (T10): QKV interleaving produces identical attention output.
#[test]
fn proof_qkv_interleave_forward() {
    // This test manually reimplements the forward pass with fused QKV slices
    // and compares against forward(). Under kog_cpu_fusion or coda_fusion,
    // forward() uses a different norm/matmul path than the manual impl.
    // Skip when the base path is altered.
    if !katgpt_forward::CPU_FORWARD_USES_DEVICE_BASE_PATH {
        return;
    }
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);

    let n = config.n_embd;
    let kvd = types::kv_dim(&config);

    // Run forward with separate Q/K/V
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let logits1 = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();

    // Interleave QKV
    weights.interleave_qkv(&config);

    // Run forward with fused QKV (feature-gated path, but we test the fused weight slicing)
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);

    // Manual forward using fused weight slices
    katgpt_core::simd::simd_add_into(&mut ctx2.x[..n], &weights.wte[0..n], &weights.wpe[0..n]);

    for layer_weights in &weights.layers {
        rmsnorm(&mut ctx2.x);
        ctx2.xr[..n].copy_from_slice(&ctx2.x[..n]);

        let fused = layer_weights
            .attn_qkv_fused
            .as_ref()
            .expect("fused should be populated");
        // Q slice
        types::matmul(&mut ctx2.q, &fused[..n * n], &ctx2.x, n, n);
        // K slice
        types::matmul(&mut ctx2.k, &fused[n * n..(n + kvd) * n], &ctx2.x, kvd, n);
        // V slice
        types::matmul(&mut ctx2.v, &fused[(n + kvd) * n..], &ctx2.x, kvd, n);

        // Store K,V
        unsafe {
            std::ptr::copy_nonoverlapping(ctx2.k.as_ptr(), cache2.layers[0].key.as_mut_ptr(), kvd);
            std::ptr::copy_nonoverlapping(
                ctx2.v.as_ptr(),
                cache2.layers[0].value.as_mut_ptr(),
                kvd,
            );
        }

        // Attention
        let scale = ctx2.attn_scale;
        for h in 0..config.n_head {
            let kv_group = ctx2.kv_group_lut[h] as usize;
            unsafe {
                attention_head(
                    &ctx2.q,
                    &cache2.layers[0].key,
                    &cache2.layers[0].value,
                    &mut ctx2.attn_out,
                    &mut ctx2.scores,
                    h * config.head_dim,
                    kv_group * config.head_dim,
                    kvd,
                    config.head_dim,
                    1,
                    scale,
                );
            }
        }
        types::matmul(&mut ctx2.x, &layer_weights.attn_wo, &ctx2.attn_out, n, n);
        katgpt_core::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr[..n]);
        ctx2.xr2[..n].copy_from_slice(&ctx2.x[..n]);
        rmsnorm(&mut ctx2.x);
        types::matmul_relu(
            &mut ctx2.hidden,
            &layer_weights.mlp_w1,
            &ctx2.x,
            config.mlp_hidden,
            n,
        );
        types::matmul(
            &mut ctx2.x,
            &layer_weights.mlp_w2,
            &ctx2.hidden,
            n,
            config.mlp_hidden,
        );
        katgpt_core::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr2[..n]);
    }

    standard_lm_head(
        &mut ctx2.logits,
        &ctx2.x,
        &weights.lm_head,
        config.vocab_size,
        n,
    );
    let logits2 = ctx2.logits.to_vec();

    // GOAT assertion
    for i in 0..config.vocab_size {
        let diff = (logits1[i] - logits2[i]).abs();
        assert!(
            diff < 1e-4,
            "GOAT FAIL: QKV interleave mismatch at logit[{i}]: sep={}, fused={}, diff={}",
            logits1[i],
            logits2[i],
            diff
        );
    }
}

/// GOAT proof (T11): MLP gamma folding produces identical single-layer output.
/// Tests the safe folding path: MLP gamma folded into w1, attention gamma kept at runtime.
#[test]
fn proof_gamma_folding_single_layer() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    let _ = rng;

    let n = config.n_embd;
    let kvd = types::kv_dim(&config);

    // Set non-trivial gamma
    for layer in &mut weights.layers {
        for (i, g) in layer.attn_norm_gamma.iter_mut().enumerate() {
            *g = 0.5 + (i as f32 * 0.1).sin() * 0.8;
        }
        for (i, g) in layer.mlp_norm_gamma.iter_mut().enumerate() {
            *g = 0.5 + (i as f32 * 0.15).cos() * 0.6;
        }
    }

    // Capture gammas
    let attn_gamma = weights.layers[0].attn_norm_gamma.clone();
    let mlp_gamma = weights.layers[0].mlp_norm_gamma.clone();

    // ── Baseline: single layer with gamma ──
    let mut ctx1 = ForwardContext::new(&config);
    let _cache1 = MultiLayerKVCache::new(&config);

    // Embed
    katgpt_core::simd::simd_add_into(&mut ctx1.x[..n], &weights.wte[0..n], &weights.wpe[0..n]);

    // Attention with gamma
    types::rmsnorm_with_gamma(&mut ctx1.x[..n], &attn_gamma);
    ctx1.xr[..n].copy_from_slice(&ctx1.x[..n]);
    types::matmul(&mut ctx1.q, &weights.layers[0].attn_wq, &ctx1.x, n, n);
    types::matmul(&mut ctx1.k, &weights.layers[0].attn_wk, &ctx1.x, kvd, n);
    types::matmul(&mut ctx1.v, &weights.layers[0].attn_wv, &ctx1.x, kvd, n);
    types::matmul(
        &mut ctx1.x,
        &weights.layers[0].attn_wo,
        &ctx1.attn_out,
        n,
        n,
    );
    katgpt_core::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr[..n]);
    // MLP with gamma
    ctx1.xr2[..n].copy_from_slice(&ctx1.x[..n]);
    types::rmsnorm_with_gamma(&mut ctx1.x[..n], &mlp_gamma);
    types::matmul_relu(
        &mut ctx1.hidden,
        &weights.layers[0].mlp_w1,
        &ctx1.x,
        config.mlp_hidden,
        n,
    );
    types::matmul(
        &mut ctx1.x,
        &weights.layers[0].mlp_w2,
        &ctx1.hidden,
        n,
        config.mlp_hidden,
    );
    katgpt_core::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr2[..n]);

    let baseline_hidden: Vec<f32> = ctx1.x[..n].to_vec();

    // ── Fold gamma ──
    weights.fold_gamma(&config);

    // ── Folded path: attn gamma at runtime, mlp gamma folded ──
    let mut ctx2 = ForwardContext::new(&config);
    let _cache2 = MultiLayerKVCache::new(&config);

    katgpt_core::simd::simd_add_into(&mut ctx2.x[..n], &weights.wte[0..n], &weights.wpe[0..n]);

    // Attention with gamma (kept)
    types::rmsnorm_with_gamma(&mut ctx2.x[..n], &attn_gamma);
    ctx2.xr[..n].copy_from_slice(&ctx2.x[..n]);
    types::matmul(&mut ctx2.q, &weights.layers[0].attn_wq, &ctx2.x, n, n);
    types::matmul(&mut ctx2.k, &weights.layers[0].attn_wk, &ctx2.x, kvd, n);
    types::matmul(&mut ctx2.v, &weights.layers[0].attn_wv, &ctx2.x, kvd, n);
    types::matmul(
        &mut ctx2.x,
        &weights.layers[0].attn_wo,
        &ctx2.attn_out,
        n,
        n,
    );
    katgpt_core::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr[..n]);
    // MLP: gamma folded, so plain rmsnorm
    ctx2.xr2[..n].copy_from_slice(&ctx2.x[..n]);
    rmsnorm(&mut ctx2.x);
    types::matmul_relu(
        &mut ctx2.hidden,
        &weights.layers[0].mlp_w1,
        &ctx2.x,
        config.mlp_hidden,
        n,
    );
    types::matmul(
        &mut ctx2.x,
        &weights.layers[0].mlp_w2,
        &ctx2.hidden,
        n,
        config.mlp_hidden,
    );
    katgpt_core::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr2[..n]);

    let folded_hidden: Vec<f32> = ctx2.x[..n].to_vec();

    // GOAT assertion
    for i in 0..n {
        let diff = (baseline_hidden[i] - folded_hidden[i]).abs();
        assert!(
            diff < 1e-5,
            "GOAT FAIL: gamma fold mismatch at [{i}]: baseline={}, folded={}, diff={}",
            baseline_hidden[i],
            folded_hidden[i],
            diff
        );
    }
}
