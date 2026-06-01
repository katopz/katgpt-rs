//! SIMD Matmul + HLA Kernel benchmarks.
//! Plan 060 Task T12 — collect actual throughput numbers for research doc update.
//!
//! Measures:
//! 1. SIMD detection level (NEON/AVX2/Scalar)
//! 2. matmul throughput for [16×16], [32×32] (n_embd for micro/game configs)
//! 3. HLA state update throughput for hd=4, hd=8
//! 4. AHLA step throughput for hd=4, hd=8
//! 5. E2E forward_hla / forward_ahla throughput (Config::micro)
//!
//! Run with: cargo test --features hla_attention bench_simd -- --nocapture

#![cfg(feature = "hla_attention")]

use katgpt_rs::hla::{
    AhlaQHeadState, HlaQHeadState, MultiLayerAhlaCache, MultiLayerHlaCache, ahla_step,
    forward_ahla, forward_hla, hla_state_update,
};
use katgpt_rs::simd::{self, SimdLevel};
use katgpt_rs::transformer::{ForwardContext, TransformerWeights, forward};
use katgpt_rs::types::{Config, Rng};
use std::time::Instant;

// ── Helpers ──────────────────────────────────────────────────

/// Format throughput as human-readable string.
fn fmt_tps(tps: f64) -> String {
    match tps {
        t if t >= 1_000_000.0 => format!("{:.1}M/s", t / 1_000_000.0),
        t if t >= 1_000.0 => format!("{:.0}K/s", t / 1_000.0),
        t => format!("{:.0}/s", t),
    }
}

/// Format microseconds per op.
fn fmt_us(us: f64) -> String {
    match us {
        u if u >= 1000.0 => format!("{:.1}ms", u / 1000.0),
        u => format!("{:.2}µs", u),
    }
}

// ── SIMD Level Detection ────────────────────────────────────

#[test]
fn bench_simd_level() {
    let level = simd::simd_level();
    let name = match level {
        SimdLevel::Scalar => "Scalar (no SIMD detected)",
        SimdLevel::Neon => "NEON (ARM)",
        SimdLevel::Avx2 => "AVX2 (x86_64)",
    };
    println!("\n━━ SIMD Detection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Level: {name}");
    println!("  arch:  {}", std::env::consts::ARCH);
    println!("  os:    {}", std::env::consts::OS);
    assert!(matches!(
        level,
        SimdLevel::Neon | SimdLevel::Avx2 | SimdLevel::Scalar
    ));
}

// ── Matmul Benchmarks ───────────────────────────────────────

#[test]
fn bench_simd_matmul() {
    let iters = 50_000;
    let warmup = 1_000;
    let dims: [(usize, &str); 3] = [(16, "16×16"), (32, "32×32"), (64, "64×64")];

    println!("\n━━ SIMD Matmul Throughput ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:<10} {:>12} {:>12}", "Size", "ops/s", "µs/op");
    println!("  {}", "─".repeat(36));

    for (dim, label) in &dims {
        let weight = vec![0.5f32; dim * dim];
        let input = vec![1.0f32; *dim];
        let mut output = vec![0.0f32; *dim];

        // Warmup
        for _ in 0..warmup {
            katgpt_rs::types::matmul(&mut output, &weight, &input, *dim, *dim);
        }

        let start = Instant::now();
        for _ in 0..iters {
            katgpt_rs::types::matmul(&mut output, &weight, &input, *dim, *dim);
        }
        let elapsed = start.elapsed();
        let tps = iters as f64 / elapsed.as_secs_f64();
        let us = 1_000_000.0 / tps;

        println!("  {:<10} {:>12} {:>12}", label, fmt_tps(tps), fmt_us(us));
    }
}

// ── Matmul ReLU Benchmarks ──────────────────────────────────

#[test]
fn bench_simd_matmul_relu() {
    let iters = 50_000;
    let warmup = 1_000;
    let dims: [(usize, &str); 2] = [(32, "32×32"), (128, "128×32")];

    println!("\n━━ SIMD Matmul-ReLU Throughput ━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:<12} {:>12} {:>12}", "Size", "ops/s", "µs/op");
    println!("  {}", "─".repeat(38));

    for (dim, label) in &dims {
        let rows = *dim;
        let cols = if rows == 128 { 32 } else { *dim };
        let weight = vec![0.5f32; rows * cols];
        let input = vec![1.0f32; cols];
        let mut output = vec![0.0f32; rows];

        for _ in 0..warmup {
            katgpt_rs::types::matmul_relu(&mut output, &weight, &input, rows, cols);
        }

        let start = Instant::now();
        for _ in 0..iters {
            katgpt_rs::types::matmul_relu(&mut output, &weight, &input, rows, cols);
        }
        let elapsed = start.elapsed();
        let tps = iters as f64 / elapsed.as_secs_f64();
        let us = 1_000_000.0 / tps;

        println!("  {:<12} {:>12} {:>12}", label, fmt_tps(tps), fmt_us(us));
    }
}

// ── Sparse Matmul Benchmarks ────────────────────────────────

#[test]
fn bench_simd_sparse_matmul() {
    let iters = 50_000;
    let warmup = 1_000;

    println!("\n━━ SIMD Sparse Matmul Throughput ━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  {:<18} {:>6} {:>12} {:>12}",
        "Config", "alive", "ops/s", "µs/op"
    );
    println!("  {}", "─".repeat(50));

    // Test configs: (rows, cols, alive_pct, label)
    let configs: [(usize, usize, f32, &str); 4] = [
        (16, 64, 0.20, "micro 16×64"),
        (32, 128, 0.20, "game 32×128"),
        (64, 256, 0.20, "small 64×256"),
        (32, 128, 0.10, "game 10% alive"),
    ];

    for (rows, cols, alive_pct, label) in &configs {
        let weight: Vec<f32> = (0..*rows * *cols)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();
        let mut input = vec![0.0f32; *cols];
        let alive_count = (*cols as f32 * alive_pct) as usize;
        // Place alive values evenly spaced
        for i in 0..alive_count {
            let idx = i * (*cols / alive_count.max(1));
            if idx < *cols {
                input[idx] = (i as f32 + 1.0) * 0.1;
            }
        }
        let mut output = vec![0.0f32; *rows];
        let mut active_indices = vec![0usize; *cols];
        let mut active_values = vec![0.0f32; *cols];

        for _ in 0..warmup {
            katgpt_rs::types::sparse_matmul(
                &mut output,
                &weight,
                &input,
                *rows,
                *cols,
                &mut active_indices,
                &mut active_values,
            );
        }

        let start = Instant::now();
        for _ in 0..iters {
            katgpt_rs::types::sparse_matmul(
                &mut output,
                &weight,
                &input,
                *rows,
                *cols,
                &mut active_indices,
                &mut active_values,
            );
        }
        let elapsed = start.elapsed();
        let tps = iters as f64 / elapsed.as_secs_f64();
        let us = 1_000_000.0 / tps;

        println!(
            "  {:<18} {:>6} {:>12} {:>12}",
            label,
            alive_count,
            fmt_tps(tps),
            fmt_us(us),
        );
    }

    // Compare sparse vs dense for game config when sparsity is high
    println!("\n  Sparse vs Dense (game config, 20% alive):");
    let rows = 32;
    let cols = 128;
    let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
    let mut input = vec![0.0f32; cols];
    let alive_count = (cols as f32 * 0.20) as usize;
    for i in 0..alive_count {
        let idx = i * (cols / alive_count);
        if idx < cols {
            input[idx] = (i as f32 + 1.0) * 0.1;
        }
    }
    let mut output_sparse = vec![0.0f32; rows];
    let mut output_dense = vec![0.0f32; rows];
    let mut active_indices = vec![0usize; cols];
    let mut active_values = vec![0.0f32; cols];

    // Sparse
    for _ in 0..warmup {
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
    for _ in 0..iters {
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
    let sparse_tps = iters as f64 / start.elapsed().as_secs_f64();

    // Dense
    for _ in 0..warmup {
        katgpt_rs::types::matmul(&mut output_dense, &weight, &input, rows, cols);
    }
    let start = Instant::now();
    for _ in 0..iters {
        katgpt_rs::types::matmul(&mut output_dense, &weight, &input, rows, cols);
    }
    let dense_tps = iters as f64 / start.elapsed().as_secs_f64();

    let speedup = sparse_tps / dense_tps;
    println!(
        "    Sparse (SIMD): {} ({:.1}× vs dense)",
        fmt_tps(sparse_tps),
        speedup,
    );
    println!("    Dense  (SIMD): {}", fmt_tps(dense_tps));
}

// ── HLA Kernel Benchmarks ───────────────────────────────────

#[test]
fn bench_simd_hla_kernels() {
    let iters = 100_000;
    let warmup = 5_000;
    let hd_configs: [usize; 2] = [4, 8];

    println!("\n━━ SIMD HLA Kernel Throughput ━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {:<20} {:>12} {:>12}", "Operation", "ops/s", "µs/op");
    println!("  {}", "─".repeat(46));

    for &hd in &hd_configs {
        // HLA state update
        {
            let mut sk = vec![0.0f32; hd * hd];
            let mut q_head = HlaQHeadState::new(hd);
            let q = vec![0.5f32; hd];
            let k = vec![0.3f32; hd];
            let v = vec![0.7f32; hd];
            let mut tmp_k_cqv = vec![0.0f32; hd];

            for _ in 0..warmup {
                hla_state_update(
                    &mut sk,
                    &mut q_head,
                    &q,
                    &k,
                    &v,
                    hd,
                    1.0,
                    &mut tmp_k_cqv,
                );
            }

            let start = Instant::now();
            for _ in 0..iters {
                hla_state_update(
                    &mut sk,
                    &mut q_head,
                    &q,
                    &k,
                    &v,
                    hd,
                    1.0,
                    &mut tmp_k_cqv,
                );
            }
            let elapsed = start.elapsed();
            let tps = iters as f64 / elapsed.as_secs_f64();
            let us = 1_000_000.0 / tps;

            println!(
                "  {:<20} {:>12} {:>12}",
                format!("hla_update hd={hd}"),
                fmt_tps(tps),
                fmt_us(us),
            );
        }

        // AHLA step
        {
            let mut pkv = vec![0.0f32; hd * hd];
            let mut mk = vec![0.0f32; hd];
            let mut q_head = AhlaQHeadState::new(hd);
            let q = vec![0.5f32; hd];
            let k = vec![0.3f32; hd];
            let v = vec![0.7f32; hd];
            let mut out = vec![0.0f32; hd];
            let mut tmp_r = vec![0.0f32; hd];

            for _ in 0..warmup {
                ahla_step(
                    &mut pkv,
                    &mut mk,
                    &mut q_head,
                    &q,
                    &k,
                    &v,
                    hd,
                    1.0,
                    &mut out,
                    &mut tmp_r,
                );
            }

            let start = Instant::now();
            for _ in 0..iters {
                ahla_step(
                    &mut pkv,
                    &mut mk,
                    &mut q_head,
                    &q,
                    &k,
                    &v,
                    hd,
                    1.0,
                    &mut out,
                    &mut tmp_r,
                );
            }
            let elapsed = start.elapsed();
            let tps = iters as f64 / elapsed.as_secs_f64();
            let us = 1_000_000.0 / tps;

            println!(
                "  {:<20} {:>12} {:>12}",
                format!("ahla_step hd={hd}"),
                fmt_tps(tps),
                fmt_us(us),
            );
        }
    }
}

// ── E2E Forward Benchmarks ──────────────────────────────────

#[test]
fn bench_simd_e2e_forward() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let iters = 5_000;
    let warmup = 500;
    let positions = 8;

    println!("\n━━ SIMD E2E Forward Throughput (Config::micro, {positions} pos) ━━━━━━━━━━");
    println!(
        "  {:<20} {:>12} {:>12} {:>10}",
        "Variant", "tok/s", "µs/tok", "pos/s"
    );
    println!("  {}", "─".repeat(56));

    // SDPA (baseline)
    {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = katgpt_rs::transformer::MultiLayerKVCache::new(&config);

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
        let us = 1_000_000.0 / tps;

        println!(
            "  {:<20} {:>12} {:>12} {:>10}",
            "forward (SDPA)",
            fmt_tps(tps),
            fmt_us(us),
            fmt_tps(tps),
        );
    }

    // HLA
    {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerHlaCache::new(&config);

        for _ in 0..warmup {
            cache.reset();
            for pos in 0..positions {
                let _ = forward_hla(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }

        let start = Instant::now();
        for _ in 0..iters {
            cache.reset();
            for pos in 0..positions {
                let _ = forward_hla(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }
        let elapsed = start.elapsed();
        let total_steps = iters as f64 * positions as f64;
        let tps = total_steps / elapsed.as_secs_f64();
        let us = 1_000_000.0 / tps;

        println!(
            "  {:<20} {:>12} {:>12} {:>10}",
            "forward_hla",
            fmt_tps(tps),
            fmt_us(us),
            fmt_tps(tps),
        );
    }

    // AHLA
    {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerAhlaCache::new(&config);

        for _ in 0..warmup {
            cache.reset();
            for pos in 0..positions {
                let _ = forward_ahla(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }

        let start = Instant::now();
        for _ in 0..iters {
            cache.reset();
            for pos in 0..positions {
                let _ = forward_ahla(&mut ctx, &weights, &mut cache, 0, pos, &config);
            }
        }
        let elapsed = start.elapsed();
        let total_steps = iters as f64 * positions as f64;
        let tps = total_steps / elapsed.as_secs_f64();
        let us = 1_000_000.0 / tps;

        println!(
            "  {:<20} {:>12} {:>12} {:>10}",
            "forward_ahla",
            fmt_tps(tps),
            fmt_us(us),
            fmt_tps(tps),
        );
    }
}

// ── 30K CCU Feasibility Summary ─────────────────────────────

#[test]
fn bench_simd_feasibility_summary() {
    let level = simd::simd_level();
    let level_name = match level {
        SimdLevel::Scalar => "Scalar",
        SimdLevel::Neon => "NEON",
        SimdLevel::Avx2 => "AVX2",
    };

    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let iters = 5_000;
    let positions = 8;

    // Measure HLA throughput for feasibility calc
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerHlaCache::new(&config);
    for _ in 0..500 {
        cache.reset();
        for pos in 0..positions {
            let _ = forward_hla(&mut ctx, &weights, &mut cache, 0, pos, &config);
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        cache.reset();
        for pos in 0..positions {
            let _ = forward_hla(&mut ctx, &weights, &mut cache, 0, pos, &config);
        }
    }
    let elapsed = start.elapsed();
    let total_steps = iters as f64 * positions as f64;
    let single_core_tps = total_steps / elapsed.as_secs_f64();

    // 30K CCU @ 20Hz = 600K tok/s required
    let required_tps = 600_000.0;
    let cores_needed = (required_tps / single_core_tps).ceil() as usize;
    let cores_available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let headroom_8c = (single_core_tps * 8.0) / required_tps;
    let headroom_full = (single_core_tps * cores_available as f64) / required_tps;

    println!("\n━━ 30K CCU @ 20Hz Feasibility ({level_name}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Config::micro (hd=4, n_embd=16, vocab=27)");
    println!("  Single-core HLA:  {}", fmt_tps(single_core_tps));
    println!("  Required:          600K tok/s (30K × 20Hz)");
    println!("  Cores needed:      {cores_needed}");
    println!("  Cores available:   {cores_available}");
    println!("  8-core headroom:   {headroom_8c:.1}×");
    println!("  Full-node headroom:{headroom_full:.1}×");
    println!();

    if single_core_tps >= required_tps {
        println!("  ✅ Single core handles 30K CCU @ 20Hz");
    } else if headroom_8c >= 5.0 {
        println!("  ✅ 8-core node handles 30K CCU @ 20Hz with ≥5× headroom");
    } else if headroom_8c >= 1.0 {
        println!("  ⚠️  8-core node handles 30K CCU @ 20Hz but tight ({headroom_8c:.1}× headroom)");
    } else {
        println!("  ❌ 8-core node insufficient for 30K CCU @ 20Hz");
    }
}
