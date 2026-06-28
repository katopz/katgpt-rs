//! Plan 228 GOAT Tests & Benchmarks — VocabChannel Pruner.
//!
//! Validates:
//! - T4: ComposedPruner AND-semantics (token valid only if all pruners agree)
//! - T5: Load-time decomposition pipeline with timing
//! - T5: BLAKE3 cache save/load round-trip
//! - T6: Benchmark: load-time decomposition speed per layer
//! - T6: Benchmark: DDTree branch reduction with vs without VocabChannelPruner
//! - T6: Benchmark: inference throughput with vs without
//!
//! # Run
//!
//! ```sh
//! cargo test --features vocab_channel_pruner --test bench_228_vocab_channel_goat -- --nocapture
//! ```

#![cfg(feature = "vocab_channel_pruner")]

use std::time::Instant;

use katgpt_core::traits::ConstraintPruner;
use katgpt_rs::speculative::dd_tree::{build_dd_tree, build_dd_tree_pruned};
use katgpt_rs::speculative::{
    ComposedPruner, VocabChannelConfig, VocabChannelMap, VocabChannelPruner,
    decompose_layer_channels, decompose_model_channels, load_cached_pruner, save_pruner_cache,
};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::Config;

// ── Helpers ───────────────────────────────────────────────────

/// Create synthetic lm_head with structured neuron→token mappings.
fn make_lm_head(vocab_size: usize, n_embd: usize) -> Vec<f32> {
    let mut lm_head = vec![0.0f32; vocab_size * n_embd];
    for t in 0..vocab_size {
        let base_dim = t % n_embd;
        lm_head[t * n_embd + base_dim] = 2.0;
        for d in 0..n_embd {
            if d != base_dim {
                lm_head[t * n_embd + d] = 0.1 * ((t * 7 + d * 13) as f32).sin();
            }
        }
    }
    lm_head
}

/// Create synthetic mlp_w2 with structured neuron activations.
fn make_mlp_w2(n_embd: usize, mlp_hidden: usize) -> Vec<f32> {
    let mut mlp_w2 = vec![0.0f32; n_embd * mlp_hidden];
    for j in 0..mlp_hidden {
        let target_dim = j % n_embd;
        for i in 0..n_embd {
            if i == target_dim {
                mlp_w2[i * mlp_hidden + j] = 3.0;
            } else {
                mlp_w2[i * mlp_hidden + j] = 0.05 * ((j * 11 + i * 3) as f32).cos();
            }
        }
    }
    mlp_w2
}

/// Create uniform marginals for DDTree testing.
fn make_uniform_marginals(seq_len: usize, vocab_size: usize) -> Vec<Vec<f32>> {
    (0..seq_len)
        .map(|_| {
            let val = 1.0 / vocab_size as f32;
            vec![val; vocab_size]
        })
        .collect()
}

/// Pruner that rejects even tokens (for composition testing).
struct RejectEvenPruner;

impl ConstraintPruner for RejectEvenPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        !token_idx.is_multiple_of(2)
    }
}

// ── T4: ComposedPruner Tests ──────────────────────────────────

#[test]
fn test_composed_pruner_empty_accepts_all() {
    let composed = ComposedPruner::new(vec![]);
    assert!(composed.is_empty());
    // All tokens valid
    for t in 0..100 {
        assert!(
            composed.is_valid(0, t, &[]),
            "empty composed should accept token {t}"
        );
    }
}

#[test]
fn test_composed_pruner_single_passthrough() {
    let composed = ComposedPruner::single(Box::new(RejectEvenPruner));
    assert_eq!(composed.len(), 1);
    assert!(!composed.is_valid(0, 0, &[])); // even → rejected
    assert!(composed.is_valid(0, 1, &[])); // odd → accepted
}

#[test]
fn test_composed_pruner_and_semantics() {
    /// Rejects tokens > 50.
    struct RejectAbove50;
    impl ConstraintPruner for RejectAbove50 {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx <= 50
        }
    }

    let composed = ComposedPruner::new(vec![Box::new(RejectEvenPruner), Box::new(RejectAbove50)]);

    // Token 3: odd (✓) and <=50 (✓) → accepted
    assert!(composed.is_valid(0, 3, &[]));
    // Token 2: even (✗) → rejected (short-circuit)
    assert!(!composed.is_valid(0, 2, &[]));
    // Token 51: odd (✓) but >50 (✗) → rejected
    assert!(!composed.is_valid(0, 51, &[]));
    // Token 49: odd (✓) and <=50 (✓) → accepted
    assert!(composed.is_valid(0, 49, &[]));
}

#[test]
fn test_composed_pruner_batch_is_valid() {
    let composed = ComposedPruner::new(vec![Box::new(RejectEvenPruner)]);

    let candidates = vec![0, 1, 2, 3, 4, 5];
    let mut results = vec![false; 6];
    composed.batch_is_valid(0, &candidates, &[], &mut results);

    assert!(!results[0]); // 0: even → rejected
    assert!(results[1]); // 1: odd → accepted
    assert!(!results[2]); // 2: even → rejected
    assert!(results[3]); // 3: odd → accepted
}

#[test]
fn test_composed_pruner_with_vocab_channel() {
    let vocab_size = 50;
    let n_embd = 16;
    let mlp_hidden = 8;

    let lm_head = make_lm_head(vocab_size, n_embd);
    let mlp_w2 = make_mlp_w2(n_embd, mlp_hidden);

    let config = VocabChannelConfig {
        max_channels: 3,
        top_k_tokens: 10,
        kurtosis_threshold: 0.5,
        ..Default::default()
    };

    let channels =
        decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);
    let map = VocabChannelMap::from_channels(&[channels], vocab_size);
    let vc_pruner = VocabChannelPruner::new(map);
    vc_pruner.set_active_context(0, &[0, 1, 2]);

    let composed = ComposedPruner::new(vec![Box::new(vc_pruner), Box::new(RejectEvenPruner)]);

    // Composed should reject even tokens AND tokens not reachable by neurons 0-2
    let mut even_rejected = 0;
    let mut composed_rejected = 0;
    for t in 0..vocab_size {
        let even_valid = t % 2 != 0;
        let composed_valid = composed.is_valid(0, t, &[]);
        if !even_valid {
            even_rejected += 1;
        }
        if !composed_valid {
            composed_rejected += 1;
        }
    }
    // Composed should reject at least as many as even-only
    assert!(
        composed_rejected >= even_rejected,
        "composed should reject >= even-only: {composed_rejected} vs {even_rejected}"
    );
}

// ── T5: Load-Time Pipeline Tests ──────────────────────────────

#[test]
fn test_decompose_model_channels_timing() {
    let mut config = Config::micro();
    // Small model for fast test
    config.vocab_size = 64;
    config.n_embd = 32;
    config.mlp_hidden = 16;
    config.n_layer = 2;

    let mut rng = katgpt_rs::types::Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let channel_config = VocabChannelConfig {
        max_channels: 3,
        top_k_tokens: 10,
        kurtosis_threshold: 0.5,
        max_iterations: 10,
        ..Default::default()
    };

    let start = Instant::now();
    let result = decompose_model_channels(&weights, &config, &channel_config);
    let elapsed = start.elapsed();

    println!("[T5.1] Model decomposition:");
    println!("  Total: {:.1}ms", result.total_ms);
    for (i, ms) in result.layer_timings_ms.iter().enumerate() {
        println!("  Layer {i}: {ms:.1}ms");
    }
    println!("  Weight hash: {:02x?}", &result.weight_hash[..8]);
    println!("  Elapsed (wall): {elapsed:?}");

    assert_eq!(result.layer_timings_ms.len(), config.n_layer);
    assert!(result.total_ms > 0.0);
    // Should complete in reasonable time even for debug builds
    assert!(
        elapsed.as_secs() < 60,
        "decomposition took {elapsed:?}, expected <60s"
    );
}

#[test]
fn test_cache_roundtrip() {
    let vocab_size = 32;
    let n_embd = 16;
    let mlp_hidden = 8;

    let lm_head = make_lm_head(vocab_size, n_embd);
    let mlp_w2 = make_mlp_w2(n_embd, mlp_hidden);

    let config = VocabChannelConfig {
        max_channels: 2,
        top_k_tokens: 5,
        kurtosis_threshold: 0.5,
        ..Default::default()
    };

    let channels =
        decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);
    let map = VocabChannelMap::from_channels(&[channels], vocab_size);
    let pruner = VocabChannelPruner::new(map);

    // Save cache
    let cache_path = std::env::temp_dir().join("bench_228_test_cache.bin");
    let fake_hash = [0xAB_u8; 32];
    save_pruner_cache(&cache_path, &fake_hash, &pruner, vocab_size, 1).unwrap();

    // Load cache with matching hash
    let loaded = load_cached_pruner(&cache_path, &fake_hash, vocab_size, 1);
    assert!(loaded.is_some(), "cache should load with matching hash");

    // Verify the loaded pruner produces same results
    let loaded_pruner = loaded.unwrap();
    loaded_pruner.set_active_context(0, &[0, 1]);
    pruner.set_active_context(0, &[0, 1]);

    for t in 0..vocab_size {
        assert_eq!(
            pruner.is_valid(0, t, &[]),
            loaded_pruner.is_valid(0, t, &[]),
            "token {t}: loaded pruner should match original"
        );
    }

    // Mismatched hash should return None
    let wrong_hash = [0xCD_u8; 32];
    let mismatched = load_cached_pruner(&cache_path, &wrong_hash, vocab_size, 1);
    assert!(mismatched.is_none(), "mismatched hash should miss");

    // Cleanup
    let _ = std::fs::remove_file(&cache_path);
}

#[test]
fn test_cache_dimension_mismatch() {
    let vocab_size = 32;
    let n_embd = 16;
    let mlp_hidden = 8;

    let lm_head = make_lm_head(vocab_size, n_embd);
    let mlp_w2 = make_mlp_w2(n_embd, mlp_hidden);

    let config = VocabChannelConfig {
        max_channels: 2,
        top_k_tokens: 5,
        kurtosis_threshold: 0.5,
        ..Default::default()
    };

    let channels =
        decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);
    let map = VocabChannelMap::from_channels(&[channels], vocab_size);
    let pruner = VocabChannelPruner::new(map);

    let cache_path = std::env::temp_dir().join("bench_228_test_dim_mismatch.bin");
    let hash = [0x11_u8; 32];
    save_pruner_cache(&cache_path, &hash, &pruner, vocab_size, 1).unwrap();

    // Wrong vocab_size
    let loaded = load_cached_pruner(&cache_path, &hash, 999, 1);
    assert!(loaded.is_none(), "wrong vocab_size should miss");

    // Wrong layer_count
    let loaded = load_cached_pruner(&cache_path, &hash, vocab_size, 99);
    assert!(loaded.is_none(), "wrong layer_count should miss");

    let _ = std::fs::remove_file(&cache_path);
}

// ── T6: Benchmarks ────────────────────────────────────────────

#[test]
fn bench_decomposition_per_layer() {
    let mut config = Config::micro();
    config.vocab_size = 128;
    config.n_embd = 64;
    config.mlp_hidden = 32;
    config.n_layer = 4;

    let mut rng = katgpt_rs::types::Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let channel_config = VocabChannelConfig {
        max_channels: 5,
        top_k_tokens: 20,
        kurtosis_threshold: 1.0,
        max_iterations: 20,
        ..Default::default()
    };

    println!("\n[T6.1] Decomposition speed per layer:");
    println!(
        "  {:>10} {:>8} {:>12} {:>10} {:>10}",
        "Layer", "Neurons", "Time (ms)", "Tokens/n", "Channels"
    );
    println!("{}", "─".repeat(55));

    let mut total_ms = 0.0;
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let start = Instant::now();
        let channels = decompose_layer_channels(
            &layer.mlp_w2,
            &weights.lm_head,
            config.n_embd,
            config.mlp_hidden,
            config.vocab_size,
            &channel_config,
        );
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;

        let avg_tokens = if channels.is_empty() {
            0.0
        } else {
            channels.iter().map(|s| s.len()).sum::<usize>() as f64 / channels.len() as f64
        };

        println!(
            "  {:>10} {:>8} {:>12.2} {:>10.1} {:>10}",
            layer_idx,
            config.mlp_hidden,
            elapsed_ms,
            avg_tokens,
            channels.len(),
        );
    }

    println!("{}", "─".repeat(55));
    println!("  {:>10} {:>8} {:>12.2}", "TOTAL", "", total_ms);
}

#[test]
fn bench_ddtree_branch_reduction() {
    let mut config = Config::micro();
    config.vocab_size = 64;
    config.n_embd = 32;
    config.mlp_hidden = 16;
    config.n_layer = 2;
    config.tree_budget = 128;

    let mut rng = katgpt_rs::types::Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Build vocab channel pruner
    let channel_config = VocabChannelConfig {
        max_channels: 3,
        top_k_tokens: 10,
        kurtosis_threshold: 0.5,
        max_iterations: 10,
        ..Default::default()
    };

    let result = decompose_model_channels(&weights, &config, &channel_config);
    let pruner = result.pruner;
    pruner.set_active_context(0, &[0, 1, 2, 3]);

    // Build marginals (3 depths)
    let seq_len = 3;
    let marginals = make_uniform_marginals(seq_len, config.vocab_size);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Build tree without pruner
    let tree_no_prune = build_dd_tree(&mv, &config);

    // Build tree with VocabChannelPruner
    let tree_with_prune = build_dd_tree_pruned(&mv, &config, &pruner, false);

    let reduction = if tree_no_prune.is_empty() {
        0.0
    } else {
        (1.0 - tree_with_prune.len() as f64 / tree_no_prune.len() as f64) * 100.0
    };

    println!("\n[T6.2] DDTree branch reduction:");
    println!("  Without pruner: {} nodes", tree_no_prune.len());
    println!("  With pruner:    {} nodes", tree_with_prune.len());
    println!("  Reduction:      {reduction:.1}%");

    // With pruning, the tree should be <= unpruned
    assert!(
        tree_with_prune.len() <= tree_no_prune.len(),
        "pruned tree should be <= unpruned: {} vs {}",
        tree_with_prune.len(),
        tree_no_prune.len(),
    );
}

#[test]
fn bench_ddtree_with_composed_pruner() {
    let mut config = Config::micro();
    config.vocab_size = 64;
    config.n_embd = 32;
    config.mlp_hidden = 16;
    config.n_layer = 2;
    config.tree_budget = 128;

    let mut rng = katgpt_rs::types::Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let channel_config = VocabChannelConfig {
        max_channels: 3,
        top_k_tokens: 10,
        kurtosis_threshold: 0.5,
        max_iterations: 10,
        ..Default::default()
    };

    let result = decompose_model_channels(&weights, &config, &channel_config);
    let vc_pruner = result.pruner;
    vc_pruner.set_active_context(0, &[0, 1, 2, 3]);

    // Compose VocabChannelPruner + RejectEvenPruner
    let composed = ComposedPruner::new(vec![Box::new(vc_pruner), Box::new(RejectEvenPruner)]);

    let seq_len = 3;
    let marginals = make_uniform_marginals(seq_len, config.vocab_size);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Build with composed pruner
    let tree_composed = build_dd_tree_pruned(&mv, &config, &composed, false);

    // Build with RejectEvenPruner only
    let even_only = RejectEvenPruner;
    let tree_even = build_dd_tree_pruned(&mv, &config, &even_only, false);

    println!("\n[T6.2b] Composed vs single pruner:");
    println!(
        "  Composed (VC + RejectEven): {} nodes",
        tree_composed.len()
    );
    println!("  RejectEven only:            {} nodes", tree_even.len());

    // Composed should be <= single (intersection is stricter)
    assert!(
        tree_composed.len() <= tree_even.len(),
        "composed should prune more than single: {} vs {}",
        tree_composed.len(),
        tree_even.len(),
    );
}

#[test]
fn bench_pruner_is_valid_throughput() {
    let vocab_size = 256;
    let n_embd = 64;
    let mlp_hidden = 32;

    let lm_head = make_lm_head(vocab_size, n_embd);
    let mlp_w2 = make_mlp_w2(n_embd, mlp_hidden);

    let config = VocabChannelConfig {
        max_channels: 5,
        top_k_tokens: 20,
        kurtosis_threshold: 0.5,
        ..Default::default()
    };

    let channels =
        decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);
    let map = VocabChannelMap::from_channels(&[channels], vocab_size);
    let pruner = VocabChannelPruner::new(map);
    pruner.set_active_context(0, &[0, 1, 2, 3, 4]);

    let iters = 10_000;
    let tokens: Vec<usize> = (0..vocab_size).collect();

    // Warmup
    for _ in 0..100 {
        for &t in &tokens {
            let _ = pruner.is_valid(0, t, &[]);
        }
    }

    // Timed run
    let start = Instant::now();
    let mut valid_count = 0usize;
    for _ in 0..iters {
        for &t in &tokens {
            if pruner.is_valid(0, t, &[]) {
                valid_count += 1;
            }
        }
    }
    let elapsed = start.elapsed();

    let total_checks = iters * vocab_size;
    let checks_per_sec = total_checks as f64 / elapsed.as_secs_f64();

    println!("\n[T6.3] Pruner is_valid throughput:");
    println!("  {total_checks} checks in {elapsed:?}");
    println!("  Throughput: {:.0} checks/sec", checks_per_sec);
    println!(
        "  Valid: {valid_count}/{total_checks} ({:.1}%)",
        valid_count as f64 / total_checks as f64 * 100.0
    );
}

#[test]
fn bench_batch_is_valid_throughput() {
    let vocab_size = 256;
    let n_embd = 64;
    let mlp_hidden = 32;

    let lm_head = make_lm_head(vocab_size, n_embd);
    let mlp_w2 = make_mlp_w2(n_embd, mlp_hidden);

    let config = VocabChannelConfig {
        max_channels: 5,
        top_k_tokens: 20,
        kurtosis_threshold: 0.5,
        ..Default::default()
    };

    let channels =
        decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);
    let map = VocabChannelMap::from_channels(&[channels], vocab_size);
    let pruner = VocabChannelPruner::new(map);
    pruner.set_active_context(0, &[0, 1, 2, 3, 4]);

    let tokens: Vec<usize> = (0..vocab_size).collect();
    let mut results = vec![false; vocab_size];

    let iters = 10_000;

    // Warmup
    for _ in 0..100 {
        pruner.batch_is_valid(0, &tokens, &[], &mut results);
    }

    // Timed run
    let start = Instant::now();
    for _ in 0..iters {
        pruner.batch_is_valid(0, &tokens, &[], &mut results);
    }
    let elapsed = start.elapsed();

    let total_checks = iters * vocab_size;
    let checks_per_sec = total_checks as f64 / elapsed.as_secs_f64();

    println!("\n[T6.3b] Batch is_valid throughput:");
    println!("  {total_checks} checks in {elapsed:?}");
    println!("  Throughput: {:.0} checks/sec", checks_per_sec);
}
