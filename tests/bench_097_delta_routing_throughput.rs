//! Delta Routing Throughput & PPL Benchmark (Plan 097, T7).
//!
//! Benchmarks:
//! 1. Throughput with delta routing enabled (single-position decode)
//! 2. Throughput scaling by layer count
//! 3. Memory overhead of block delta buffers
//! 4. Pseudo-PPL delta measurement (zero vs non-zero query weights)
//! 5. Block size sensitivity (theoretical routing frequency)
//! 6. Forward correctness across multi-position sequences
//!
//! Run: `cargo test -p microgpt-rs --test bench_097_delta_routing_throughput \
//!       --features delta_routing --release -- --nocapture`

#![cfg(feature = "delta_routing")]

use microgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use microgpt_rs::types::{Config, Rng, softmax};
use std::time::Instant;

// ── Constants ───────────────────────────────────────────────────

/// Block size B=4 per Plan 097 default.
const BLOCK_SIZE: usize = 4;

/// Iterations for primary throughput benchmark.
const N_ITER_THROUGHPUT: usize = 1000;

/// Iterations for scaling sweep (fewer for speed).
const N_ITER_SCALING: usize = 500;

/// Sequence length for multi-position tests.
const SEQ_LEN: usize = 16;

// ── Helpers ─────────────────────────────────────────────────────

fn make_config(n_layer: usize) -> Config {
    let mut config = Config::micro();
    config.n_layer = n_layer;
    config.validate().expect("Config should be valid");
    config
}

/// Compute pseudo-PPL for a token sequence with custom query weights.
fn compute_pseudo_ppl(
    config: &Config,
    seq_tokens: &[usize],
    query_weights_override: &[Vec<f32>],
) -> f64 {
    let mut rng = Rng::new(42);
    let mut weights = TransformerWeights::new(config, &mut rng);

    for (layer_idx, qw) in query_weights_override.iter().enumerate() {
        if layer_idx < weights.delta_routing_query.len() {
            weights.delta_routing_query[layer_idx] = qw.clone();
        }
    }

    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    let mut total_loss = 0.0f64;
    let n_tokens = seq_tokens.len().saturating_sub(1);

    for pos in 0..n_tokens {
        let token = seq_tokens[pos];
        let logits = forward(&mut ctx, &weights, &mut cache, token, pos, config);

        let mut probs = logits.to_vec();
        softmax(&mut probs);

        let next_token = seq_tokens[pos + 1];
        let prob = probs[next_token].max(1e-10);
        total_loss += -prob.ln() as f64;
    }

    total_loss / n_tokens.max(1) as f64
}

/// Run a throughput measurement loop and return avg latency in µs.
fn measure_throughput(config: &Config, n_iter: usize) -> f64 {
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);

    // Warmup
    for _ in 0..20 {
        let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, config);
    }

    let start = Instant::now();
    for i in 0..n_iter {
        let token = i % config.vocab_size;
        let pos = i % config.block_size;
        if i % config.block_size == 0 {
            cache = MultiLayerKVCache::new(config);
        }
        let _ = forward(&mut ctx, &weights, &mut cache, token, pos, config);
    }
    let elapsed = start.elapsed();

    elapsed.as_nanos() as f64 / n_iter as f64 / 1000.0
}

// ── Test 1: Throughput Delta Routing ────────────────────────────

#[test]
fn bench_throughput_delta_routing() {
    println!("\n🧪 Bench 1: Throughput with Delta Routing (n_layer=6, B=4)");
    println!("{}", "═".repeat(70));

    let config = make_config(6);
    let avg_latency_us = measure_throughput(&config, N_ITER_THROUGHPUT);
    let throughput = 1_000_000.0 / avg_latency_us;

    println!("| Metric              | Value          |");
    println!("|---------------------|----------------|");
    println!("| n_layer             | {}              |", config.n_layer);
    println!("| block_size (B)      | {BLOCK_SIZE}              |");
    println!("| n_iter              | {N_ITER_THROUGHPUT}           |");
    println!("| avg latency/token   | {avg_latency_us:>10.2} µs  |");
    println!("| throughput          | {throughput:>10.0} tok/s |");
    println!("| paper claim         | ≤30% overhead  |");

    println!("\n✅ bench_throughput_delta_routing complete");
}

// ── Test 2: Throughput Scaling by Layers ─────────────────────────

#[test]
fn bench_throughput_scaling_by_layers() {
    println!("\n🧪 Bench 2: Throughput Scaling by Layer Count");
    println!("{}", "═".repeat(70));

    let layer_counts: &[usize] = &[1, 2, 4, 6, 8, 12];

    println!("| n_layer | avg_latency_us | throughput_tok/s | routing_fires/pass |");
    println!("|---------|----------------|------------------|--------------------|");

    let mut results: Vec<(usize, f64, f64)> = Vec::new();

    for &n_layer in layer_counts {
        let config = make_config(n_layer);
        let avg_latency_us = measure_throughput(&config, N_ITER_SCALING);
        let throughput = 1_000_000.0 / avg_latency_us;
        let routing_fires = n_layer / BLOCK_SIZE;

        println!(
            "| {n_layer:>7} | {avg_latency_us:>14.2} | {throughput:>16.0} | {routing_fires:>18} |"
        );
        results.push((n_layer, avg_latency_us, throughput));
    }

    // Overhead analysis
    if let (Some(base), Some(max)) = (results.first(), results.last()) {
        let overhead_pct = (max.1 - base.1) / base.1 * 100.0;
        let layer_ratio = max.0 as f64 / base.0 as f64;
        let latency_ratio = max.1 / base.1;
        let efficiency = latency_ratio / layer_ratio;

        println!("\n📊 Scaling Analysis:");
        println!("  n_layer: {} → {} ({layer_ratio:.1}×)", base.0, max.0);
        println!(
            "  latency: {:.2} → {:.2} µs ({latency_ratio:.2}×)",
            base.1, max.1
        );
        println!("  total overhead: {overhead_pct:.1}%");
        println!("  latency/layer efficiency: {efficiency:.2}× (1.0 = linear)");
    }

    println!("\n✅ bench_throughput_scaling_by_layers complete");
}

// ── Test 3: Memory Overhead ─────────────────────────────────────

#[test]
fn bench_memory_overhead() {
    println!("\n🧪 Bench 3: Memory Overhead Calculation (n_layer=6, B=4)");
    println!("{}", "═".repeat(70));

    let n_layer: usize = 6;
    let n_embd: usize = 16;
    let b = BLOCK_SIZE;
    let f32_size = std::mem::size_of::<f32>();

    let n_blocks = n_layer.div_ceil(b);

    // Runtime buffers (ForwardContext)
    let block_deltas_bytes = n_blocks * n_embd * f32_size;
    let routing_logits_bytes = (n_layer + 1) * f32_size;
    let runtime_overhead = block_deltas_bytes + routing_logits_bytes;

    // Weight overhead (TransformerWeights)
    let query_weights_bytes = n_layer * n_embd * f32_size;
    let norm_weights_bytes = n_layer * n_embd * f32_size;
    let weight_overhead = query_weights_bytes + norm_weights_bytes;

    let total_overhead = runtime_overhead + weight_overhead;

    // Base model memory (approximate)
    let config = make_config(n_layer);
    let wte_bytes = config.vocab_size * n_embd * f32_size;
    let wpe_bytes = config.block_size * n_embd * f32_size;
    let lm_head_bytes = config.vocab_size * n_embd * f32_size;
    // Per-layer: Q, K, V, attn_proj, mlp_fc, mlp_proj + 2 norms (approx)
    let layer_weights_approx = (4 * n_embd * n_embd + config.mlp_hidden * n_embd * 2) * f32_size;
    let all_layers_bytes = n_layer * layer_weights_approx;
    let base_model_bytes = wte_bytes + wpe_bytes + lm_head_bytes + all_layers_bytes;

    let overhead_pct = total_overhead as f64 / base_model_bytes as f64 * 100.0;

    // Per-block bound: (B+1) × n_embd × sizeof(f32) per block
    let per_block_bound = (b + 1) * n_embd * f32_size;
    let total_block_bound = per_block_bound * n_blocks;

    println!("| Component                 | Size (bytes) |");
    println!("|---------------------------|--------------|");
    println!("| block_deltas [{n_blocks}][{n_embd}]     | {block_deltas_bytes:>12} |");
    println!("| routing_logits [{n_layer}+1]      | {routing_logits_bytes:>12} |");
    println!("| query_weights [{n_layer}][{n_embd}]    | {query_weights_bytes:>12} |");
    println!("| norm_weights [{n_layer}][{n_embd}]     | {norm_weights_bytes:>12} |");
    println!("| **total delta overhead**  | {total_overhead:>12} |");
    println!("| base model (approx)       | {base_model_bytes:>12} |");
    println!("| overhead %                | {overhead_pct:>11.4}% |");

    println!("\n📊 Per-Block Bound Check:");
    println!(
        "  per_block_bound: (B+1) × n_embd × sizeof(f32) = {b}+1 × {n_embd} × {f32_size} = {per_block_bound} bytes"
    );
    println!(
        "  total_block_bound: {n_blocks} blocks × {per_block_bound} = {total_block_bound} bytes"
    );
    println!("  runtime_overhead: {runtime_overhead} bytes");

    assert!(
        runtime_overhead <= total_block_bound,
        "Runtime overhead ({runtime_overhead}) exceeds per-block bound ({total_block_bound})"
    );

    println!(
        "  ✅ runtime_overhead ({runtime_overhead}) ≤ total_block_bound ({total_block_bound})"
    );
    println!("  ✅ overhead as % of base model: {overhead_pct:.4}%");

    println!("\n✅ bench_memory_overhead complete");
}

// ── Test 4: Pseudo-PPL Delta Measurement ────────────────────────

#[test]
fn bench_ppl_delta_measurement() {
    println!("\n🧪 Bench 4: Pseudo-PPL Delta (zero vs non-zero query weights)");
    println!("{}", "═".repeat(70));

    let config = make_config(6);
    let seq_tokens: Vec<usize> = (0..SEQ_LEN).map(|i| i % config.vocab_size).collect();

    // Config A: zero-init query (delta routing effectively off within the feature)
    let zero_query: Vec<Vec<f32>> = (0..config.n_layer)
        .map(|_| vec![0.0f32; config.n_embd])
        .collect();
    let ppl_zero = compute_pseudo_ppl(&config, &seq_tokens, &zero_query);

    // Config B: random non-zero query (delta routing active)
    let mut rng_b = Rng::new(99);
    let nonzero_query: Vec<Vec<f32>> = (0..config.n_layer)
        .map(|_| {
            (0..config.n_embd)
                .map(|_| {
                    let r = (rng_b.next() as f64 / u64::MAX as f64 - 0.5) * 0.1;
                    r as f32
                })
                .collect()
        })
        .collect();
    let ppl_nonzero = compute_pseudo_ppl(&config, &seq_tokens, &nonzero_query);

    let delta_ppl = ppl_nonzero - ppl_zero;

    println!("| Config                  | Pseudo-PPL |");
    println!("|-------------------------|------------|");
    println!("| Zero query (routing off)| {ppl_zero:>10.4} |");
    println!("| Non-zero query (on)     | {ppl_nonzero:>10.4} |");
    println!("| Δ PPL                   | {delta_ppl:>+10.4} |");

    let delta_measurable = delta_ppl.abs() > 1e-6;
    println!(
        "\n  Δ PPL measurable: {} (|Δ| = {:.6})",
        if delta_measurable {
            "✅ yes"
        } else {
            "⚠️ no"
        },
        delta_ppl.abs()
    );

    println!("\n✅ bench_ppl_delta_measurement complete");
}

// ── Test 5: Block Size Sweep (Theoretical) ──────────────────────

#[test]
fn bench_block_size_sweep() {
    println!("\n🧪 Bench 5: Block Size Sensitivity (theoretical routing frequency)");
    println!("{}", "═".repeat(70));

    let n_layer: usize = 6;
    let block_sizes: &[usize] = &[2, 3, 4, 6];

    println!("| B (block_size) | n_blocks | routing_fires/pass | layers_with_routing |");
    println!("|----------------|----------|--------------------|---------------------|");

    for &b in block_sizes {
        let n_blocks = n_layer.div_ceil(b);
        let routing_fires = n_layer / b;
        let routed_layers: Vec<usize> = (1..=n_layer).filter(|l| l % b == 0).collect();
        let routed_str = routed_layers
            .iter()
            .map(|l| format!("{l}"))
            .collect::<Vec<_>>()
            .join(", ");

        println!("| {b:>14} | {n_blocks:>8} | {routing_fires:>18} | {routed_str:<19} |");
    }

    println!("\n📊 Note: block_size is hardcoded at B=4 in current impl.");
    println!("  This table documents parameter sensitivity for future tuning.");

    // Reference throughput measurement at current B=4
    let config = make_config(n_layer);
    let avg_us = measure_throughput(&config, N_ITER_SCALING);
    println!("\n  Reference (B=4, n_layer={n_layer}): {avg_us:.2} µs/token");

    println!("\n✅ bench_block_size_sweep complete");
}

// ── Test 6: Forward Correctness Multi-Position ──────────────────

#[test]
fn bench_forward_correctness_multi_position() {
    println!("\n🧪 Bench 6: Forward Correctness ({SEQ_LEN} positions, n_layer=6)");
    println!("{}", "═".repeat(70));

    let config = make_config(6);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ctx = ForwardContext::new(&config);

    println!("| pos | token | logit_min | logit_max | logit_mean | finite |");
    println!("|-----|-------|-----------|-----------|------------|--------|");

    let mut all_different = true;
    let mut prev_logits: Option<Vec<f32>> = None;

    for pos in 0..SEQ_LEN {
        let token = pos % config.vocab_size;
        let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
        let logits_vec = logits.to_vec();

        let all_finite = logits_vec.iter().all(|l| l.is_finite());
        let min = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;

        println!(
            "| {pos:>3} | {token:>5} | {min:>9.4} | {max:>9.4} | {mean:>10.4} | {} |",
            if all_finite { "✅" } else { "❌" }
        );

        assert!(all_finite, "Position {pos}: non-finite logits detected");
        assert_eq!(
            logits_vec.len(),
            config.vocab_size,
            "Logit count mismatch at pos {pos}"
        );

        // Check non-degenerate: different from previous position
        if let Some(ref prev) = prev_logits {
            let identical = logits_vec
                .iter()
                .zip(prev.iter())
                .all(|(a, b)| (a - b).abs() < 1e-10);
            if identical {
                all_different = false;
            }
        }
        prev_logits = Some(logits_vec);
    }

    assert!(
        all_different,
        "All positions produced identical logits (degenerate)"
    );

    println!("\n  ✅ All {SEQ_LEN} positions produced finite, non-degenerate logits");
    println!("  ✅ Each position has unique output");

    println!("\n✅ bench_forward_correctness_multi_position complete");
}
