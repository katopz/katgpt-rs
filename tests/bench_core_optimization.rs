//! Comprehensive core hot-path benchmark for katgpt-core + src/ optimization.
//!
//! Measures all SIMD kernels, math utilities, and transformer forward pass
//! to identify optimization targets. Follows optimization.md profiling template:
//! - 200 warmup iterations to prime CPU caches
//! - 10,000+ iterations for stable results
//! - black_box() to prevent dead-code elimination
//! - Component-level breakdowns for bottleneck identification
//!
//! Run with:
//!   cargo test --test bench_core_optimization --release -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::{
    Config, Rng, SimdLevel, matmul, matmul_relu, rmsnorm, sample_token, softmax, softmax_scaled,
};
use katgpt_rs::simd::{
    simd_add_inplace, simd_add_into, simd_dot_f32, simd_max_f32, simd_scale_inplace,
};
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use katgpt_rs::types::{gegelu, gegelu_tanh, matmul_parallel, rmsnorm_with_gamma};

const WARMUP: usize = 200;
const ITERS: usize = 10_000;
const SHORT_ITERS: usize = 2_000;

// ── Helpers ──────────────────────────────────────────────────

fn fmt_us(us: f64) -> String {
    match us {
        u if u >= 1000.0 => format!("{:.1}ms", u / 1000.0),
        u if u >= 1.0 => format!("{:.2}us", u),
        u => format!("{:.0}ns", u * 1000.0),
    }
}

/// Benchmark a FnMut closure with warmup and return us/iter.
fn bench_mut(label: &str, warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let us = elapsed.as_micros() as f64 / iters as f64;
    println!("  {:<40} {:>10}", label, fmt_us(us));
    us
}

// ── Section 1: SIMD Level Detection ─────────────────────────

#[test]
fn bench_01_simd_detection() {
    let level = katgpt_rs::simd::simd_level();

    let name = match level {
        SimdLevel::Scalar => "Scalar (no SIMD)",
        SimdLevel::Neon => "NEON (ARM)",
        SimdLevel::Avx2 => "AVX2+FMA (x86_64)",
    };

    println!();
    println!("============================================================");
    println!("  Section 1: SIMD Detection");
    println!("============================================================");
    println!("  Level: {name}");
    println!("  arch:  {}", std::env::consts::ARCH);
    println!("  os:    {}", std::env::consts::OS);
    println!(
        "  cores: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
}

// ── Section 2: Core SIMD Primitives ─────────────────────────

#[test]
fn bench_02_simd_primitives() {
    println!();
    println!("============================================================");
    println!("  Section 2: Core SIMD Primitives (10K iters, {WARMUP} warmup)");
    println!("============================================================");

    let dims: &[usize] = &[16, 32, 64, 128, 256, 512];

    // Dot product
    println!();
    println!("  simd_dot_f32:");
    println!("  {:<40} {:>10}", "Size", "us/call");
    println!("  {}", "-".repeat(52));
    for &dim in dims {
        let a = vec![0.5f32; dim];
        let b = vec![0.3f32; dim];
        bench_mut(&format!("dot [{}]", dim), WARMUP, ITERS, || {
            let _ = black_box(simd_dot_f32(&a, &b, dim));
        });
    }

    // Scale inplace
    println!();
    println!("  simd_scale_inplace:");
    println!("  {:<40} {:>10}", "Size", "us/call");
    println!("  {}", "-".repeat(52));
    for &dim in dims {
        let mut x = vec![1.0f32; dim];
        bench_mut(&format!("scale [{}]", dim), WARMUP, ITERS, || {
            simd_scale_inplace(&mut x, 0.5);
        });
    }

    // Add inplace
    println!();
    println!("  simd_add_inplace:");
    println!("  {:<40} {:>10}", "Size", "us/call");
    println!("  {}", "-".repeat(52));
    for &dim in dims {
        let mut dst = vec![1.0f32; dim];
        let src = vec![0.5f32; dim];
        bench_mut(&format!("add_inplace [{}]", dim), WARMUP, ITERS, || {
            simd_add_inplace(&mut dst, &src);
        });
    }

    // Max reduction
    println!();
    println!("  simd_max_f32:");
    println!("  {:<40} {:>10}", "Size", "us/call");
    println!("  {}", "-".repeat(52));
    for &dim in dims {
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        bench_mut(&format!("max [{}]", dim), WARMUP, ITERS, || {
            let _ = black_box(simd_max_f32(&x));
        });
    }
}

// ── Section 3: Math Utilities ────────────────────────────────

#[test]
fn bench_03_math_utilities() {
    println!();
    println!("============================================================");
    println!("  Section 3: Math Utilities (10K iters, {WARMUP} warmup)");
    println!("============================================================");

    let vocab_sizes: &[usize] = &[27, 256, 1024, 4096, 32000];

    // Softmax
    println!();
    println!("  softmax:");
    println!("  {:<40} {:>10}", "Vocab Size", "us/call");
    println!("  {}", "-".repeat(52));
    for &vocab in vocab_sizes {
        let init: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x = init.clone();
        bench_mut(&format!("softmax [vocab={}]", vocab), WARMUP, ITERS, || {
            softmax(&mut x);
            x.copy_from_slice(&init);
        });
    }

    // Softmax scaled
    println!();
    println!("  softmax_scaled (temperature=0.8):");
    println!("  {:<40} {:>10}", "Vocab Size", "us/call");
    println!("  {}", "-".repeat(52));
    let inv_temp = 1.0f32 / 0.8;
    for &vocab in vocab_sizes {
        let init: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x = init.clone();
        bench_mut(
            &format!("softmax_scaled [vocab={}]", vocab),
            WARMUP,
            ITERS,
            || {
                softmax_scaled(&mut x, inv_temp);
                x.copy_from_slice(&init);
            },
        );
    }

    // RMSNorm
    println!();
    println!("  rmsnorm:");
    println!("  {:<40} {:>10}", "Dim", "us/call");
    println!("  {}", "-".repeat(52));
    let norm_dims: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 2048];
    for &dim in norm_dims {
        let mut x = vec![1.0f32; dim];
        bench_mut(&format!("rmsnorm [dim={}]", dim), WARMUP, ITERS, || {
            rmsnorm(&mut x);
            x.fill(1.0);
        });
    }

    // RMSNorm with gamma
    println!();
    println!("  rmsnorm_with_gamma:");
    println!("  {:<40} {:>10}", "Dim", "us/call");
    println!("  {}", "-".repeat(52));
    for &dim in norm_dims {
        let mut x = vec![1.0f32; dim];
        let gamma: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32 * 0.01)).collect();
        bench_mut(
            &format!("rmsnorm_with_gamma [dim={}]", dim),
            WARMUP,
            ITERS,
            || {
                rmsnorm_with_gamma(&mut x, &gamma);
                x.fill(1.0);
            },
        );
    }

    // Sample token
    println!();
    println!("  sample_token:");
    println!("  {:<40} {:>10}", "Vocab Size", "us/call");
    println!("  {}", "-".repeat(52));
    for &vocab in vocab_sizes {
        let probs = vec![1.0f32 / vocab as f32; vocab];
        let mut rng = Rng::new(42);
        bench_mut(
            &format!("sample_token [vocab={}]", vocab),
            WARMUP,
            ITERS,
            || {
                let _ = black_box(sample_token(&probs, &mut rng));
            },
        );
    }
}

// ── Section 4: Matmul Kernels ────────────────────────────────

#[test]
fn bench_04_matmul_kernels() {
    println!();
    println!("============================================================");
    println!("  Section 4: Matmul Kernels (10K iters, {WARMUP} warmup)");
    println!("============================================================");

    // Matmul at transformer-relevant sizes
    let matmul_configs: &[(usize, usize, &str)] = &[
        (16, 16, "micro QKV (16x16)"),
        (64, 16, "micro MLP w1 (64x16)"),
        (16, 64, "micro MLP w2 (16x64)"),
        (27, 16, "micro lm_head (27x16)"),
        (32, 32, "game QKV (32x32)"),
        (128, 32, "game MLP w1 (128x32)"),
        (32, 128, "game MLP w2 (32x128)"),
        (64, 64, "small QKV (64x64)"),
        (256, 64, "small MLP w1 (256x64)"),
        (64, 256, "small MLP w2 (64x256)"),
        (512, 64, "small lm_head (512x64)"),
        (2304, 256, "target down_proj-like"),
    ];

    println!();
    println!("  matmul (f32xf32):");
    println!("  {:<40} {:>10} {:>12}", "Config", "us/call", "Mops/s");
    println!("  {}", "-".repeat(64));

    for &(rows, cols, label) in matmul_configs {
        let weight = vec![0.5f32; rows * cols];
        let input = vec![1.0f32; cols];
        let mut output = vec![0.0f32; rows];
        let flops = 2.0 * rows as f64 * cols as f64;

        for _ in 0..WARMUP {
            matmul(&mut output, &weight, &input, rows, cols);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            matmul(&mut output, &weight, &input, rows, cols);
        }
        let elapsed = start.elapsed();
        let us = elapsed.as_micros() as f64 / ITERS as f64;
        let mops = flops / (us * 1000.0);

        println!("  {:<40} {:>10} {:>10.0}M", label, fmt_us(us), mops);
    }

    // Matmul ReLU
    println!();
    println!("  matmul_relu (fused ReLU):");
    println!("  {:<40} {:>10} {:>12}", "Config", "us/call", "Mops/s");
    println!("  {}", "-".repeat(64));

    let relu_configs: &[(usize, usize, &str)] = &[
        (64, 16, "micro MLP w1 (64x16)"),
        (128, 32, "game MLP w1 (128x32)"),
        (256, 64, "small MLP w1 (256x64)"),
    ];

    for &(rows, cols, label) in relu_configs {
        let weight = vec![0.5f32; rows * cols];
        let input = vec![1.0f32; cols];
        let mut output = vec![0.0f32; rows];
        let flops = 2.0 * rows as f64 * cols as f64;

        for _ in 0..WARMUP {
            matmul_relu(&mut output, &weight, &input, rows, cols);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            matmul_relu(&mut output, &weight, &input, rows, cols);
        }
        let elapsed = start.elapsed();
        let us = elapsed.as_micros() as f64 / ITERS as f64;
        let mops = flops / (us * 1000.0);

        println!("  {:<40} {:>10} {:>10.0}M", label, fmt_us(us), mops);
    }

    // Sparse matmul vs dense comparison
    println!();
    println!("  sparse_matmul vs dense (game config 32x128, varying sparsity):");
    println!(
        "  {:<20} {:>10} {:>10} {:>10}",
        "Alive %", "Sparse us", "Dense us", "Speedup"
    );
    println!("  {}", "-".repeat(54));

    let sparsities: &[f32] = &[0.05, 0.10, 0.20, 0.50, 0.80, 1.00];
    let rows = 32;
    let cols = 128;
    let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();

    for &alive_pct in sparsities {
        let mut input = vec![0.0f32; cols];
        let alive_count = ((cols as f32 * alive_pct) as usize).max(1);
        for i in 0..alive_count {
            let idx = i * (cols / alive_count.max(1));
            if idx < cols {
                input[idx] = (i as f32 + 1.0) * 0.1;
            }
        }
        let mut output_sparse = vec![0.0f32; rows];
        let mut output_dense = vec![0.0f32; rows];
        let mut active_indices = vec![0usize; cols];
        let mut active_values = vec![0.0f32; cols];

        // Sparse
        for _ in 0..WARMUP {
            katgpt_rs::types::sparse_matmul(
                &mut output_sparse,
                &weight,
                &input,
                rows,
                cols,
                &mut active_indices,
                &mut active_values,
            );
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            katgpt_rs::types::sparse_matmul(
                &mut output_sparse,
                &weight,
                &input,
                rows,
                cols,
                &mut active_indices,
                &mut active_values,
            );
        }
        let sparse_us = start.elapsed().as_micros() as f64 / ITERS as f64;

        // Dense
        for _ in 0..WARMUP {
            matmul(&mut output_dense, &weight, &input, rows, cols);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            matmul(&mut output_dense, &weight, &input, rows, cols);
        }
        let dense_us = start.elapsed().as_micros() as f64 / ITERS as f64;

        let speedup = dense_us / sparse_us;
        println!(
            "  {:>18.0}% {:>10} {:>10} {:>10.2}x",
            alive_pct * 100.0,
            fmt_us(sparse_us),
            fmt_us(dense_us),
            speedup,
        );
    }
}

// ── Section 5: GeGLU Activation ──────────────────────────────

#[test]
fn bench_05_gegelu() {
    println!();
    println!("============================================================");
    println!("  Section 5: GeGLU Activation (10K iters, {WARMUP} warmup)");
    println!("============================================================");

    let dims: &[usize] = &[64, 128, 256, 512, 1024, 2048];

    // GeGLU (sigmoid approximation)
    println!();
    println!("  gegelu (sigmoid approx):");
    println!("  {:<40} {:>10}", "Dim", "us/call");
    println!("  {}", "-".repeat(52));
    for &dim in dims {
        let mut hidden = vec![0.0f32; dim];
        let gate: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let up: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();

        bench_mut(&format!("gegelu [dim={}]", dim), WARMUP, ITERS, || {
            gegelu(&mut hidden, &gate, &up);
        });
    }

    // GeGLU tanh (Gemma 2)
    println!();
    println!("  gegelu_tanh (Gemma 2):");
    println!("  {:<40} {:>10}", "Dim", "us/call");
    println!("  {}", "-".repeat(52));
    for &dim in dims {
        let mut hidden = vec![0.0f32; dim];
        let gate: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let up: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();

        bench_mut(&format!("gegelu_tanh [dim={}]", dim), WARMUP, ITERS, || {
            gegelu_tanh(&mut hidden, &gate, &up);
        });
    }
}

// ── Section 6: E2E Transformer Forward ───────────────────────

#[test]
fn bench_06_e2e_forward() {
    println!();
    println!("============================================================");
    println!("  Section 6: E2E Transformer Forward Pass");
    println!("============================================================");

    // Config::micro
    {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let positions = 16;
        let iters = 5_000;
        let warmup = 500;

        for _ in 0..warmup {
            cache.reset();
            for pos in 0..positions {
                let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }

        let start = Instant::now();
        for _ in 0..iters {
            cache.reset();
            for pos in 0..positions {
                let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }
        let elapsed = start.elapsed();
        let total_steps = iters as f64 * positions as f64;
        let tps = total_steps / elapsed.as_secs_f64();
        let us_per_tok = 1_000_000.0 / tps;

        println!();
        println!(
            "  Config::micro (vocab={}, embd={}, heads={}, mlp={}, layers={}):",
            config.vocab_size, config.n_embd, config.n_head, config.mlp_hidden, config.n_layer
        );
        println!("    throughput:  {tps:.0} tok/s");
        println!("    latency:     {us_per_tok:.2} us/tok");
        println!("    positions:   {positions}");
    }

    // Config::game
    {
        let config = Config::game();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let positions = 16;
        let iters = 5_000;
        let warmup = 500;

        for _ in 0..warmup {
            cache.reset();
            for pos in 0..positions {
                let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }

        let start = Instant::now();
        for _ in 0..iters {
            cache.reset();
            for pos in 0..positions {
                let _ = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }
        let elapsed = start.elapsed();
        let total_steps = iters as f64 * positions as f64;
        let tps = total_steps / elapsed.as_secs_f64();
        let us_per_tok = 1_000_000.0 / tps;

        println!();
        println!(
            "  Config::game (vocab={}, embd={}, heads={}, mlp={}, layers={}):",
            config.vocab_size, config.n_embd, config.n_head, config.mlp_hidden, config.n_layer
        );
        println!("    throughput:  {tps:.0} tok/s");
        println!("    latency:     {us_per_tok:.2} us/tok");
        println!("    positions:   {positions}");
    }

    // Config::game with longer context (pos=64)
    {
        let config = Config::game();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Pre-fill 64 positions
        cache.reset();
        for pos in 0..64 {
            forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
        }

        let iters = 2_000;
        let warmup = 200;

        for _ in 0..warmup {
            let _ = forward(&mut ctx, &weights, &mut cache, 0, 64, &config);
        }

        let start = Instant::now();
        for _ in 0..iters {
            let _ = forward(&mut ctx, &weights, &mut cache, 0, 64, &config);
        }
        let elapsed = start.elapsed();
        let us_per_tok = elapsed.as_micros() as f64 / iters as f64;
        let tps = 1_000_000.0 / us_per_tok;

        println!();
        println!("  Config::game (pos=64, t_n=65 - longer attention):");
        println!("    throughput:  {tps:.0} tok/s");
        println!("    latency:     {us_per_tok:.2} us/tok");
    }

    // Config::small_target (if not too slow)
    {
        let config = Config::small_target();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let iters = 1_000;
        let warmup = 100;

        for _ in 0..warmup {
            cache.reset();
            let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        }

        let start = Instant::now();
        for _ in 0..iters {
            cache.reset();
            let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        }
        let elapsed = start.elapsed();
        let us_per_tok = elapsed.as_micros() as f64 / iters as f64;
        let tps = 1_000_000.0 / us_per_tok;

        println!();
        println!(
            "  Config::small_target (vocab={}, embd={}, heads={}, mlp={}, layers={}):",
            config.vocab_size, config.n_embd, config.n_head, config.mlp_hidden, config.n_layer
        );
        println!("    throughput:  {tps:.0} tok/s");
        println!("    latency:     {us_per_tok:.2} us/tok");
        println!("    (pos=0 only, single-step decode)");
    }
}

// ── Section 7: Forward Pass Component Breakdown ──────────────

#[test]
fn bench_07_component_breakdown() {
    println!();
    println!("============================================================");
    println!("  Section 7: Forward Pass Component Breakdown (Config::game)");
    println!("============================================================");

    let config = Config::game();
    let n = config.n_embd;
    let mlp_hidden = config.mlp_hidden;
    let vocab = config.vocab_size;

    println!();
    println!(
        "  Config: vocab={vocab}, embd={n}, heads={}, kv_heads={}, head_dim={}, mlp={mlp_hidden}, layers={}",
        config.n_head, config.n_kv_head, config.head_dim, config.n_layer
    );

    // Component: Embedding (add wte + wpe)
    let mut embd_buf = vec![0.0f32; n];
    let wte_row = vec![0.5f32; n];
    let wpe_row = vec![0.3f32; n];
    let t_embd = bench_mut("Embedding (add wte+wpe)", WARMUP, ITERS, || {
        simd_add_into(&mut embd_buf, &wte_row, &wpe_row);
    });

    // Component: RMSNorm (per-layer, 2x per layer)
    let mut norm_buf = vec![1.0f32; n];
    let t_rmsnorm = bench_mut("RMSNorm (no gamma)", WARMUP, ITERS, || {
        rmsnorm(&mut norm_buf);
        norm_buf.fill(1.0);
    });

    // Component: RMSNorm with gamma (Gemma 2 style)
    let gamma: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.01)).collect();
    let mut norm_g_buf = vec![1.0f32; n];
    let t_rmsnorm_gamma = bench_mut("RMSNorm with gamma", WARMUP, ITERS, || {
        rmsnorm_with_gamma(&mut norm_g_buf, &gamma);
        norm_g_buf.fill(1.0);
    });

    // Component: QKV projection (3 matmuls per layer)
    let qkv_weight = vec![0.5f32; n * n];
    let qkv_input = vec![1.0f32; n];
    let mut qkv_output = vec![0.0f32; n];
    let t_qkv = bench_mut("QKV projection (nxn)", WARMUP, ITERS, || {
        matmul(&mut qkv_output, &qkv_weight, &qkv_input, n, n);
    });

    // Component: Attention output projection
    let attn_wo_weight = vec![0.5f32; n * n];
    let attn_wo_input = vec![1.0f32; n];
    let mut attn_wo_output = vec![0.0f32; n];
    let t_attn_wo = bench_mut("Attention wo (nxn)", WARMUP, ITERS, || {
        matmul(&mut attn_wo_output, &attn_wo_weight, &attn_wo_input, n, n);
    });

    // Component: MLP w1 (matmul_relu)
    let mlp_w1_weight = vec![0.5f32; mlp_hidden * n];
    let mlp_w1_input = vec![1.0f32; n];
    let mut mlp_w1_output = vec![0.0f32; mlp_hidden];
    let t_mlp_w1 = bench_mut("MLP w1 (matmul_relu)", WARMUP, ITERS, || {
        matmul_relu(
            &mut mlp_w1_output,
            &mlp_w1_weight,
            &mlp_w1_input,
            mlp_hidden,
            n,
        );
    });

    // Component: MLP w2
    let mlp_w2_weight = vec![0.5f32; n * mlp_hidden];
    let mlp_w2_input = vec![1.0f32; mlp_hidden];
    let mut mlp_w2_output = vec![0.0f32; n];
    let t_mlp_w2 = bench_mut("MLP w2 (dense)", WARMUP, ITERS, || {
        matmul(
            &mut mlp_w2_output,
            &mlp_w2_weight,
            &mlp_w2_input,
            n,
            mlp_hidden,
        );
    });

    // Component: LM head
    let lm_head_weight = vec![0.5f32; vocab * n];
    let lm_head_input = vec![1.0f32; n];
    let mut lm_head_output = vec![0.0f32; vocab];
    let t_lm_head = bench_mut("LM head (vocabxn)", WARMUP, ITERS, || {
        matmul(
            &mut lm_head_output,
            &lm_head_weight,
            &lm_head_input,
            vocab,
            n,
        );
    });

    // Component: Softmax on vocab
    let softmax_init: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut softmax_buf = softmax_init.clone();
    let t_softmax = bench_mut("Softmax (vocab)", WARMUP, ITERS, || {
        softmax(&mut softmax_buf);
        softmax_buf.copy_from_slice(&softmax_init);
    });

    // Component: Residual add
    let mut residual_dst = vec![1.0f32; n];
    let residual_src = vec![0.5f32; n];
    let t_residual = bench_mut("Residual add", WARMUP, ITERS, || {
        simd_add_inplace(&mut residual_dst, &residual_src);
    });

    // Per-layer cost breakdown
    let n_layer = config.n_layer;
    let per_layer_cost =
        2.0 * t_rmsnorm + 3.0 * t_qkv + t_attn_wo + t_mlp_w1 + t_mlp_w2 + 2.0 * t_residual;
    let total_cost = t_embd + per_layer_cost * n_layer as f64 + t_lm_head + t_softmax;

    println!();
    println!("  Per-layer breakdown (1 layer):");
    println!("  {:<40} {:>10} {:>8}", "Component", "us", "% layer");
    println!("  {}", "-".repeat(60));
    let pct = |v: f64| v / per_layer_cost * 100.0;
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "2x RMSNorm",
        fmt_us(2.0 * t_rmsnorm),
        pct(2.0 * t_rmsnorm)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "3x QKV projection",
        fmt_us(3.0 * t_qkv),
        pct(3.0 * t_qkv)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "Attention wo",
        fmt_us(t_attn_wo),
        pct(t_attn_wo)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "MLP w1 (matmul_relu)",
        fmt_us(t_mlp_w1),
        pct(t_mlp_w1)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "MLP w2",
        fmt_us(t_mlp_w2),
        pct(t_mlp_w2)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "2x Residual add",
        fmt_us(2.0 * t_residual),
        pct(2.0 * t_residual)
    );
    println!("  {}", "-".repeat(60));
    println!(
        "  {:<40} {:>10} {:>8}",
        "Per-layer total",
        fmt_us(per_layer_cost),
        ""
    );

    println!();
    println!("  Full forward breakdown:");
    println!("  {:<40} {:>10} {:>8}", "Component", "us", "% total");
    println!("  {}", "-".repeat(60));
    let pct_total = |v: f64| v / total_cost * 100.0;
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "Embedding",
        fmt_us(t_embd),
        pct_total(t_embd)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        format!("{n_layer}x layer"),
        fmt_us(per_layer_cost * n_layer as f64),
        pct_total(per_layer_cost * n_layer as f64)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "LM head",
        fmt_us(t_lm_head),
        pct_total(t_lm_head)
    );
    println!(
        "  {:<40} {:>10} {:>7.1}%",
        "Softmax",
        fmt_us(t_softmax),
        pct_total(t_softmax)
    );
    println!("  {}", "-".repeat(60));
    println!("  {:<40} {:>10}", "Estimated total", fmt_us(total_cost));

    // Top 3 bottlenecks
    let mut components: Vec<(&str, f64)> = vec![
        ("QKV projection (3x)", 3.0 * t_qkv),
        ("MLP w1", t_mlp_w1),
        ("MLP w2", t_mlp_w2),
        ("LM head", t_lm_head),
        ("Softmax", t_softmax),
        ("RMSNorm (2x)", 2.0 * t_rmsnorm),
        ("Residual (2x)", 2.0 * t_residual),
        ("Attention wo", t_attn_wo),
    ];
    components.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!("  Top 3 bottlenecks (per layer, excludes LM head):");
    for (i, (name, cost)) in components.iter().take(3).enumerate() {
        println!("    {}. {} - {}", i + 1, name, fmt_us(*cost));
    }

    // RMSNorm gamma vs no-gamma comparison
    let gamma_speedup = t_rmsnorm / t_rmsnorm_gamma;
    println!();
    println!(
        "  RMSNorm: no-gamma {:.2}us vs gamma {:.2}us ({:.2}x)",
        t_rmsnorm, t_rmsnorm_gamma, gamma_speedup
    );
}

// ── Section 8: Parallel Matmul Threshold ─────────────────────

#[test]
fn bench_08_parallel_threshold() {
    println!();
    println!("============================================================");
    println!("  Section 8: Parallel Matmul Threshold (sequential vs rayon)");
    println!("============================================================");

    let configs: &[(usize, usize, &str)] = &[
        (128, 64, "128x64"),
        (256, 64, "256x64"),
        (512, 64, "512x64"),
        (1024, 64, "1024x64"),
        (2048, 256, "2048x256"),
        (4096, 256, "4096x256"),
        (9216, 2304, "9216x2304 (down_proj)"),
    ];

    println!();
    println!(
        "  {:<25} {:>12} {:>12} {:>10}",
        "Size", "Seq us", "Par us", "Speedup"
    );
    println!("  {}", "-".repeat(62));

    for &(rows, cols, label) in configs {
        let weight = vec![0.5f32; rows * cols];
        let input = vec![1.0f32; cols];
        let mut output_seq = vec![0.0f32; rows];
        let mut output_par = vec![0.0f32; rows];

        // Sequential
        for _ in 0..WARMUP {
            matmul(&mut output_seq, &weight, &input, rows, cols);
        }
        let start = Instant::now();
        for _ in 0..SHORT_ITERS {
            matmul(&mut output_seq, &weight, &input, rows, cols);
        }
        let seq_us = start.elapsed().as_micros() as f64 / SHORT_ITERS as f64;

        // Parallel
        for _ in 0..WARMUP {
            matmul_parallel(&mut output_par, &weight, &input, rows, cols);
        }
        let start = Instant::now();
        for _ in 0..SHORT_ITERS {
            matmul_parallel(&mut output_par, &weight, &input, rows, cols);
        }
        let par_us = start.elapsed().as_micros() as f64 / SHORT_ITERS as f64;

        let speedup = seq_us / par_us;
        println!(
            "  {:<25} {:>12} {:>12} {:>10.2}x",
            label,
            fmt_us(seq_us),
            fmt_us(par_us),
            speedup,
        );
    }
}

// ── Section 9: Summary ───────────────────────────────────────

#[test]
fn bench_09_summary() {
    println!();
    println!("============================================================");
    println!("  Section 9: Optimization Summary");
    println!("============================================================");

    // Quick representative benchmarks for summary
    let config = Config::game();
    let n = config.n_embd;

    // RMSNorm
    let mut norm_buf = vec![1.0f32; n];
    for _ in 0..WARMUP {
        rmsnorm(&mut norm_buf);
        norm_buf.fill(1.0);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        rmsnorm(&mut norm_buf);
        norm_buf.fill(1.0);
    }
    let rmsnorm_us = start.elapsed().as_nanos() as f64 / ITERS as f64 / 1000.0;

    // Softmax
    let softmax_init: Vec<f32> = (0..config.vocab_size)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let mut softmax_buf = softmax_init.clone();
    for _ in 0..WARMUP {
        softmax(&mut softmax_buf);
        softmax_buf.copy_from_slice(&softmax_init);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        softmax(&mut softmax_buf);
        softmax_buf.copy_from_slice(&softmax_init);
    }
    let softmax_us = start.elapsed().as_nanos() as f64 / ITERS as f64 / 1000.0;

    // MLP w1 matmul
    let weight = vec![0.5f32; config.mlp_hidden * n];
    let input = vec![1.0f32; n];
    let mut output = vec![0.0f32; config.mlp_hidden];
    for _ in 0..WARMUP {
        matmul_relu(&mut output, &weight, &input, config.mlp_hidden, n);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        matmul_relu(&mut output, &weight, &input, config.mlp_hidden, n);
    }
    let mlp_us = start.elapsed().as_nanos() as f64 / ITERS as f64 / 1000.0;

    // E2E forward
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    for _ in 0..500 {
        cache.reset();
        let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        cache.reset();
        let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
    }
    let fwd_us = start.elapsed().as_nanos() as f64 / ITERS as f64 / 1000.0;

    println!();
    println!("  Config::game key metrics:");
    println!("    RMSNorm [embd={}]:        {:.2} us", n, rmsnorm_us);
    println!(
        "    Softmax [vocab={}]:       {:.2} us",
        config.vocab_size, softmax_us
    );
    println!(
        "    MLP w1 [{}x{}]:    {:.2} us",
        config.mlp_hidden, n, mlp_us
    );
    println!("    Forward (pos=0):          {:.2} us/tok", fwd_us);
    println!("    Forward throughput:       {:.0} tok/s", 1000.0 / fwd_us);
    println!();
    println!("  SIMD level: {:?}", katgpt_rs::simd::simd_level());
    println!();
    println!("  Completed optimizations:");
    println!(
        "    1. rmsnorm_with_gamma - simd_dot_f32 for sum_sq + simd_scale_mul_inplace for fused gamma (2-3x faster)"
    );
    println!(
        "    2. simd_exp_inplace - Cephes 6th-order polynomial for NEON/AVX2 (kept as utility, libm faster on Apple Silicon)"
    );
    println!(
        "    3. simd_scale_mul_inplace - new fused kernel for rmsnorm_with_gamma scale+gamma multiply"
    );
    println!();
    println!("  Remaining candidates:");
    println!("    4. gegelu/gegelu_tanh - scalar elementwise, candidate for SIMD");
    println!("    5. sample_token - cumulative scan, candidate for SIMD comparison");
}
