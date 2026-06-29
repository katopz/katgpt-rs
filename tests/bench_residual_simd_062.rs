//! SIMD Residual + Attention component benchmarks for Issue 062.
//!
//! Measures throughput for new SIMD primitives and their integration into
//! forward pass hot paths:
//! 1. simd_add_inplace (residual add, 4× per layer)
//! 2. simd_add_into (embedding add, 1× per forward)
//! 3. simd_max_f32 (softmax max-finding)
//! 4. simd_fused_decay_write (raven KV update)
//! 5. simd_dot_f32 (raven readout Q·K)
//! 6. E2E forward_base decode (exercises all SIMD paths)
//!
//! Run with: cargo test bench_residual_simd_062 -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::simd::{
    simd_add_inplace, simd_add_into, simd_dot_f32, simd_fused_decay_write, simd_max_f32,
    simd_scale_inplace,
};
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use katgpt_rs::types::{Config, Rng};

const WARMUP: usize = 100;
const ITERS: usize = 50000;

#[test]
fn bench_residual_simd_062() {
    let config = Config::micro();
    let n = config.n_embd;
    let kv_dim = config.n_kv_head * config.head_dim;
    let num_slots = 16;

    // ── Print SIMD level ──
    let level = katgpt_core::simd::simd_level();
    println!("=== bench_residual_simd_062 ===");
    println!("SIMD level: {level:?}");
    println!("n_embd={n}, kv_dim={kv_dim}, num_slots={num_slots}");
    println!();

    // ── Allocate standalone buffers ──
    let mut dst_buf = vec![0.0f32; n];
    let mut src_buf = vec![0.0f32; n];
    let a_buf = vec![1.0f32; n];
    let b_buf = vec![2.0f32; n];
    let mut max_buf: Vec<f32> = (0..config.vocab_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let mut kv_buf = vec![0.5f32; kv_dim * num_slots];
    let new_key = vec![0.3f32; kv_dim];
    let _new_value = vec![0.4f32; kv_dim];
    let query = vec![0.1f32; kv_dim];
    let keys = vec![0.2f32; kv_dim * num_slots];

    // ── Component A: simd_add_inplace (residual add) ──
    // Called 2× per layer (attn residual + MLP residual) per forward pass
    for _ in 0..WARMUP {
        dst_buf.fill(1.0);
        src_buf.fill(0.5);
        simd_add_inplace(&mut dst_buf, &src_buf);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        dst_buf.fill(1.0);
        src_buf.fill(0.5);
        simd_add_inplace(&mut dst_buf, &src_buf);
    }
    let t_add_inplace = start.elapsed();
    let add_inplace_ns = t_add_inplace.as_nanos() as f64 / ITERS as f64;

    // ── Component B: simd_add_into (embedding add) ──
    // Called 1× per forward pass (wte + wpe)
    for _ in 0..WARMUP {
        simd_add_into(&mut dst_buf, &a_buf, &b_buf);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        simd_add_into(&mut dst_buf, &a_buf, &b_buf);
    }
    let t_add_into = start.elapsed();
    let add_into_ns = t_add_into.as_nanos() as f64 / ITERS as f64;

    // ── Component C: simd_max_f32 (softmax max) ──
    // Called 1× per softmax (vocab_size elements)
    for _ in 0..WARMUP {
        black_box(simd_max_f32(&max_buf));
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        black_box(simd_max_f32(&max_buf));
    }
    let t_max = start.elapsed();
    let max_ns = t_max.as_nanos() as f64 / ITERS as f64;

    // ── Component D: simd_fused_decay_write (raven update) ──
    // Called per-slot per-layer in raven mode
    let decay = 0.9f32;
    let write = 0.1f32;
    for _ in 0..WARMUP {
        simd_fused_decay_write(&mut kv_buf[..kv_dim], decay, &new_key, write);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        simd_fused_decay_write(&mut kv_buf[..kv_dim], decay, &new_key, write);
    }
    let t_fused = start.elapsed();
    let fused_ns = t_fused.as_nanos() as f64 / ITERS as f64;

    // ── Component E: simd_dot_f32 (raven readout Q·K) ──
    // Called per-slot per-head per-layer in raven mode
    for _ in 0..WARMUP {
        black_box(simd_dot_f32(&query, &keys[..kv_dim], kv_dim));
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        black_box(simd_dot_f32(&query, &keys[..kv_dim], kv_dim));
    }
    let t_dot = start.elapsed();
    let dot_ns = t_dot.as_nanos() as f64 / ITERS as f64;

    // ── Component F: simd_scale_inplace (softmax normalize / residual normalize) ──
    // Already benchmarked in 057, included here for completeness
    let scale = 0.04f32; // ~1/vocab_size
    for _ in 0..WARMUP {
        max_buf.iter_mut().for_each(|v| *v = 0.5);
        simd_scale_inplace(&mut max_buf, scale);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        max_buf.iter_mut().for_each(|v| *v = 0.5);
        simd_scale_inplace(&mut max_buf, scale);
    }
    let t_scale = start.elapsed();
    let scale_ns = t_scale.as_nanos() as f64 / ITERS as f64;

    // ── Component G: E2E forward_base decode ──
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // Warmup
    for pos in 0..8 {
        let token = pos % config.vocab_size;
        forward(&mut ctx, &weights, &mut cache, token, pos, &config);
    }
    let forward_iters = 1000;
    let start = Instant::now();
    for pos in 0..forward_iters {
        let token = pos % config.vocab_size;
        forward(
            &mut ctx,
            &weights,
            &mut cache,
            token,
            pos % config.block_size,
            &config,
        );
    }
    let t_forward = start.elapsed();
    let forward_us = t_forward.as_micros() as f64 / forward_iters as f64;

    // ── Results ──
    println!("Component                          | Time (ns/iter) | Elements");
    println!("-----------------------------------|----------------|--------");
    println!("simd_add_inplace (residual add)    | {add_inplace_ns:>12.1} ns | n={n}");
    println!("simd_add_into (embedding add)      | {add_into_ns:>12.1} ns | n={n}");
    println!(
        "simd_max_f32 (softmax max)         | {max_ns:>12.1} ns | vocab={}",
        config.vocab_size
    );
    println!("simd_fused_decay_write (raven)     | {fused_ns:>12.1} ns | kv_dim={kv_dim}");
    println!("simd_dot_f32 (raven Q·K)           | {dot_ns:>12.1} ns | kv_dim={kv_dim}");
    println!(
        "simd_scale_inplace (normalize)     | {scale_ns:>12.1} ns | vocab={}",
        config.vocab_size
    );
    println!("-----------------------------------|----------------|--------");
    println!(
        "E2E forward_base decode            | {forward_us:>10.1} µs | n_layer={}",
        config.n_layer
    );
    println!();

    // ── Per-layer cost estimate ──
    // Each layer has: 2× residual add + 1× embedding (only first layer)
    let layer_residual_ns = 2.0 * add_inplace_ns;
    println!("Per-layer residual cost: {layer_residual_ns:.1} ns (2× simd_add_inplace)");
    println!(
        "Per-token residual cost: {:.1} ns ({} layers × 2 adds)",
        layer_residual_ns * config.n_layer as f64,
        config.n_layer
    );

    // Sanity checks — results should be non-trivial
    assert!(
        add_inplace_ns > 0.0,
        "add_inplace should take non-zero time"
    );
    assert!(add_into_ns > 0.0, "add_into should take non-zero time");
    assert!(max_ns > 0.0, "max should take non-zero time");
    assert!(fused_ns > 0.0, "fused should take non-zero time");
    assert!(dot_ns > 0.0, "dot should take non-zero time");
    assert!(forward_us > 0.0, "forward should take non-zero time");

    println!("\nAll benchmarks completed successfully.");
}
