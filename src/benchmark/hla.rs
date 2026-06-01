use super::{BenchCategory, BenchResult};
use crate::hla::{MultiLayerAhlaCache, MultiLayerHlaCache, forward_ahla, forward_hla};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::kv_dim;
use crate::types::{Config, Rng};
use std::time::Instant;

#[cfg(feature = "hla_attention")]
pub fn bench_hla_vs_flat_cache(_config: &Config) -> BenchResult {
    let bench_config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&bench_config, &mut rng);
    let iters = 200;
    let positions = 8;

    // Warm up all three paths
    {
        let mut ctx = ForwardContext::new(&bench_config);
        let mut cache = MultiLayerKVCache::new(&bench_config);
        let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, &bench_config);
    }
    {
        let mut ctx = ForwardContext::new(&bench_config);
        let mut cache = MultiLayerHlaCache::new(&bench_config);
        let _ = forward_hla(&mut ctx, &weights, &mut cache, 0, 0, &bench_config);
    }
    {
        let mut ctx = ForwardContext::new(&bench_config);
        let mut cache = MultiLayerAhlaCache::new(&bench_config);
        let _ = forward_ahla(&mut ctx, &weights, &mut cache, 0, 0, &bench_config);
    }

    // Benchmark flat cache (growing O(N) attention)
    let mut ctx_flat = ForwardContext::new(&bench_config);
    let mut cache_flat = MultiLayerKVCache::new(&bench_config);
    let start_flat = Instant::now();
    for _ in 0..iters {
        cache_flat.reset();
        for pos in 0..positions {
            let _ = forward(
                &mut ctx_flat,
                &weights,
                &mut cache_flat,
                0,
                pos,
                &bench_config,
            );
        }
    }
    let elapsed_flat = start_flat.elapsed();

    // Benchmark HLA (symmetric, O(1) per step)
    let mut ctx_hla = ForwardContext::new(&bench_config);
    let mut cache_hla = MultiLayerHlaCache::new(&bench_config);
    let start_hla = Instant::now();
    for _ in 0..iters {
        cache_hla.reset();
        for pos in 0..positions {
            let _ = forward_hla(
                &mut ctx_hla,
                &weights,
                &mut cache_hla,
                0,
                pos,
                &bench_config,
            );
        }
    }
    let elapsed_hla = start_hla.elapsed();

    // Benchmark AHLA (asymmetric, O(1) per step, smaller state)
    let mut ctx_ahla = ForwardContext::new(&bench_config);
    let mut cache_ahla = MultiLayerAhlaCache::new(&bench_config);
    let start_ahla = Instant::now();
    for _ in 0..iters {
        cache_ahla.reset();
        for pos in 0..positions {
            let _ = forward_ahla(
                &mut ctx_ahla,
                &weights,
                &mut cache_ahla,
                0,
                pos,
                &bench_config,
            );
        }
    }
    let elapsed_ahla = start_ahla.elapsed();

    let steps = iters as f64 * positions as f64;
    let flat_tps = steps / elapsed_flat.as_secs_f64();
    let hla_tps = steps / elapsed_hla.as_secs_f64();
    let ahla_tps = steps / elapsed_ahla.as_secs_f64();
    let flat_us = elapsed_flat.as_micros() as f64 / steps;
    let hla_us = elapsed_hla.as_micros() as f64 / steps;
    let ahla_us = elapsed_ahla.as_micros() as f64 / steps;

    // Memory per layer
    let kvd = kv_dim(&bench_config);
    let flat_mem = bench_config.block_size * kvd * 2 * 4; // key + value, f32
    let hla_mem = MultiLayerHlaCache::new(&bench_config).memory_bytes() / bench_config.n_layer;
    let ahla_mem = MultiLayerAhlaCache::new(&bench_config).memory_bytes() / bench_config.n_layer;

    println!(
        "\n\u{250c}\u{2500} HLA vs Flat Cache (micro, {iters}\u{00d7}{positions} pos) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}"
    );
    println!(
        "\u{2502} {:<22} {:>10} {:>12} {:>14} \u{2502}",
        "Method", "tok/s", "\u{00b5}s/step", "mem/layer (B)"
    );
    println!("\u{2502} {} \u{2502}", "-".repeat(60));
    println!(
        "\u{2502} {:<22} {:>10.1} {:>12.2} {:>14} \u{2502}",
        "Forward (flat KV)", flat_tps, flat_us, flat_mem
    );
    println!(
        "\u{2502} {:<22} {:>10.1} {:>12.2} {:>14} \u{2502}",
        "Forward HLA (sym)", hla_tps, hla_us, hla_mem
    );
    println!(
        "\u{2502} {:<22} {:>10.1} {:>12.2} {:>14} \u{2502}",
        "Forward AHLA (asym)", ahla_tps, ahla_us, ahla_mem
    );
    println!(
        "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}"
    );

    BenchResult {
        label: "forward (flat HLA bench)".into(),
        throughput: flat_tps,
        time_per_step_us: flat_us,
        avg_acceptance_len: 0.0,
        color: (100, 149, 237),
        category: BenchCategory::Infrastructure,
        feature_dim: "Attn".into(),
    }
}

#[cfg(feature = "hla_attention")]
pub fn bench_hla_memory(_config: &Config) -> BenchResult {
    let configs: [(&str, Config); 4] = [
        ("micro", Config::micro()),
        ("game", Config::game()),
        ("bpe", Config::bpe()),
        ("gqa_draft", Config::gqa_draft()),
    ];

    println!(
        "\n\u{250c}\u{2500} HLA Memory Usage by Config \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}"
    );
    println!(
        "\u{2502} {:<12} {:>10} {:>10} {:>12} {:>8} \u{2502}",
        "Config", "Flat KV", "HLA (sym)", "AHLA (asym)", "Savings"
    );
    println!("\u{2502} {} \u{2502}", "-".repeat(54));

    let mut total_flat: usize = 0;
    let mut total_ahla: usize = 0;

    for (name, cfg) in &configs {
        let kvd = kv_dim(cfg);
        let flat_bytes = cfg.block_size * kvd * 2 * 4;
        let hla_bytes = MultiLayerHlaCache::new(cfg).memory_bytes();
        let ahla_bytes = MultiLayerAhlaCache::new(cfg).memory_bytes();
        let savings = (1.0 - ahla_bytes as f64 / flat_bytes as f64) * 100.0;

        println!(
            "\u{2502} {:<12} {:>7} B {:>7} B {:>9} B {:>6.1}% \u{2502}",
            name, flat_bytes, hla_bytes, ahla_bytes, savings
        );

        total_flat += flat_bytes;
        total_ahla += ahla_bytes;
    }

    let avg_savings = (1.0 - total_ahla as f64 / total_flat as f64) * 100.0;
    println!(
        "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}"
    );
    println!("   \u{2192} Average AHLA savings vs flat KV: {avg_savings:.1}%");

    BenchResult {
        label: format!("hla_memory (avg {avg_savings:.1}% savings)"),
        throughput: 0.0,
        time_per_step_us: 0.0,
        avg_acceptance_len: avg_savings,
        color: (60, 179, 113),
        category: BenchCategory::Infrastructure,
        feature_dim: "Attn".into(),
    }
}

#[cfg(feature = "hla_attention")]
pub fn bench_hla_quality(_config: &Config) -> BenchResult {
    let bench_config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&bench_config, &mut rng);
    let n_tokens = 16;

    // Generate logits with flat SDPA
    let mut ctx = ForwardContext::new(&bench_config);
    let mut cache = MultiLayerKVCache::new(&bench_config);
    let mut sdpa_logits: Vec<Vec<f32>> = Vec::new();
    for pos in 0..n_tokens {
        let logits = forward(&mut ctx, &weights, &mut cache, 0, pos, &bench_config);
        sdpa_logits.push(logits.to_vec());
    }

    // Generate logits with HLA (symmetric)
    let mut ctx_hla = ForwardContext::new(&bench_config);
    let mut cache_hla = MultiLayerHlaCache::new(&bench_config);
    let mut hla_logits: Vec<Vec<f32>> = Vec::new();
    for pos in 0..n_tokens {
        let logits = forward_hla(
            &mut ctx_hla,
            &weights,
            &mut cache_hla,
            0,
            pos,
            &bench_config,
        );
        hla_logits.push(logits.to_vec());
    }

    // Generate logits with AHLA (asymmetric)
    let mut ctx_ahla = ForwardContext::new(&bench_config);
    let mut cache_ahla = MultiLayerAhlaCache::new(&bench_config);
    let mut ahla_logits: Vec<Vec<f32>> = Vec::new();
    for pos in 0..n_tokens {
        let logits = forward_ahla(
            &mut ctx_ahla,
            &weights,
            &mut cache_ahla,
            0,
            pos,
            &bench_config,
        );
        ahla_logits.push(logits.to_vec());
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-8 { 0.0 } else { dot / denom }
    }

    let mut hla_sims = Vec::with_capacity(n_tokens);
    let mut ahla_sims = Vec::with_capacity(n_tokens);
    for pos in 0..n_tokens {
        let sdpa = &sdpa_logits[pos];
        let hla_sim = cosine_sim(sdpa, &hla_logits[pos]);
        let ahla_sim = cosine_sim(sdpa, &ahla_logits[pos]);
        assert!(
            hla_sim.is_finite(),
            "HLA sim at pos {pos} not finite: {hla_sim}"
        );
        assert!(
            ahla_sim.is_finite(),
            "AHLA sim at pos {pos} not finite: {ahla_sim}"
        );
        hla_sims.push(hla_sim);
        ahla_sims.push(ahla_sim);
    }

    let hla_avg = hla_sims.iter().sum::<f32>() / n_tokens as f32;
    let hla_min = hla_sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let ahla_avg = ahla_sims.iter().sum::<f32>() / n_tokens as f32;
    let ahla_min = ahla_sims.iter().cloned().fold(f32::INFINITY, f32::min);

    println!(
        "\n\u{250c}\u{2500} HLA Quality Check (micro, {n_tokens} tokens, random weights) \u{2500}\u{2500}\u{2500}\u{2500}\u{2510}"
    );
    println!(
        "\u{2502} {:<22} {:>12} {:>12} \u{2502}",
        "Method", "avg cos-sim", "min cos-sim"
    );
    println!("\u{2502} {} \u{2502}", "-".repeat(48));
    println!(
        "\u{2502} {:<22} {:>12.4} {:>12.4} \u{2502}",
        "HLA (sym) vs SDPA", hla_avg, hla_min
    );
    println!(
        "\u{2502} {:<22} {:>12.4} {:>12.4} \u{2502}",
        "AHLA (asym) vs SDPA", ahla_avg, ahla_min
    );
    println!(
        "\u{2502} Note: random weights \u{2192} expect low sim (different functions)     \u{2502}"
    );
    println!(
        "\u{2502} Verified: all logits finite, non-NaN \u{2713}                          \u{2502}"
    );
    println!("└──────────────────────────────────────────────────────────────────────┘");

    BenchResult {
        label: format!("hla_quality (HLA={hla_avg:.3}, AHLA={ahla_avg:.3})"),
        throughput: 0.0,
        time_per_step_us: 0.0,
        avg_acceptance_len: ((hla_avg + ahla_avg) / 2.0) as f64,
        color: (255, 165, 0),
        category: BenchCategory::Infrastructure,
        feature_dim: "Attn".into(),
    }
}

/// SIMD micro-benchmark: matmul, HLA kernels, and end-to-end forward (Plan 060).
///
/// Measures throughput of SIMD-accelerated operations:
/// - `matmul` [32×32]×[32] (game config n_embd)
/// - `matmul` [16×16]×[16] (micro config n_embd)
/// - HLA state update hd=4, hd=8
/// - AHLA step hd=4, hd=8
/// - End-to-end `forward_hla()` and `forward_ahla()` with micro config
#[cfg(feature = "hla_attention")]
pub fn bench_simd(_config: &Config) -> BenchResult {
    use crate::simd::{self, SimdLevel};

    let level = simd::simd_level();
    let level_name = match level {
        SimdLevel::Scalar => "Scalar",
        SimdLevel::Neon => "NEON",
        SimdLevel::Avx2 => "AVX2",
    };

    let iters = 10_000;

    // ── Matmul benchmarks ──
    let matmul_configs: [(&str, usize); 2] = [("16×16", 16), ("32×32", 32)];
    let mut matmul_results: Vec<(&str, f64)> = Vec::new();

    for &(label, dim) in &matmul_configs {
        let weight = vec![0.5f32; dim * dim];
        let input = vec![1.0f32; dim];
        let mut output = vec![0.0f32; dim];

        // Warmup
        for _ in 0..100 {
            crate::types::matmul(&mut output, &weight, &input, dim, dim);
        }

        let start = Instant::now();
        for _ in 0..iters {
            crate::types::matmul(&mut output, &weight, &input, dim, dim);
        }
        let elapsed = start.elapsed();
        let tps = iters as f64 / elapsed.as_secs_f64();
        matmul_results.push((label, tps));
    }

    // ── HLA kernel benchmarks ──
    let hd_configs: [usize; 2] = [4, 8];
    let mut hla_update_tps: Vec<(usize, f64)> = Vec::new();
    let mut ahla_step_tps: Vec<(usize, f64)> = Vec::new();

    for &hd in &hd_configs {
        // HLA state update
        {
            let mut sk = vec![0.0f32; hd * hd];
            let mut q_head = crate::hla::HlaQHeadState::new(hd);
            let q = vec![0.5f32; hd];
            let k = vec![0.3f32; hd];
            let v = vec![0.7f32; hd];
            let mut tmp_k_cqv = vec![0.0f32; hd];

            let start = Instant::now();
            for _ in 0..iters {
                crate::hla::hla_state_update(
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
            hla_update_tps.push((hd, tps));
        }

        // AHLA step
        {
            let mut pkv = vec![0.0f32; hd * hd];
            let mut mk = vec![0.0f32; hd];
            let mut q_head = crate::hla::AhlaQHeadState::new(hd);
            let q = vec![0.5f32; hd];
            let k = vec![0.3f32; hd];
            let v = vec![0.7f32; hd];
            let mut out = vec![0.0f32; hd];
            let mut tmp_r = vec![0.0f32; hd];

            let start = Instant::now();
            for _ in 0..iters {
                crate::hla::ahla_step(
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
            ahla_step_tps.push((hd, tps));
        }
    }

    // ── End-to-end forward benchmarks ──
    let bench_config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&bench_config, &mut rng);
    let forward_iters = 2_000;
    let positions = 8;

    // Forward HLA
    let mut ctx_hla = ForwardContext::new(&bench_config);
    let mut cache_hla = MultiLayerHlaCache::new(&bench_config);
    // Warmup
    for pos in 0..positions {
        let _ = forward_hla(
            &mut ctx_hla,
            &weights,
            &mut cache_hla,
            0,
            pos,
            &bench_config,
        );
    }
    let start_hla = Instant::now();
    for _ in 0..forward_iters {
        cache_hla.reset();
        for pos in 0..positions {
            let _ = forward_hla(
                &mut ctx_hla,
                &weights,
                &mut cache_hla,
                0,
                pos,
                &bench_config,
            );
        }
    }
    let elapsed_hla = start_hla.elapsed();
    let hla_steps = forward_iters as f64 * positions as f64;
    let hla_tps = hla_steps / elapsed_hla.as_secs_f64();
    let hla_us = elapsed_hla.as_micros() as f64 / hla_steps;

    // Forward AHLA
    let mut ctx_ahla = ForwardContext::new(&bench_config);
    let mut cache_ahla = MultiLayerAhlaCache::new(&bench_config);
    // Warmup
    for pos in 0..positions {
        let _ = forward_ahla(
            &mut ctx_ahla,
            &weights,
            &mut cache_ahla,
            0,
            pos,
            &bench_config,
        );
    }
    let start_ahla = Instant::now();
    for _ in 0..forward_iters {
        cache_ahla.reset();
        for pos in 0..positions {
            let _ = forward_ahla(
                &mut ctx_ahla,
                &weights,
                &mut cache_ahla,
                0,
                pos,
                &bench_config,
            );
        }
    }
    let elapsed_ahla = start_ahla.elapsed();
    let ahla_steps = forward_iters as f64 * positions as f64;
    let ahla_tps = ahla_steps / elapsed_ahla.as_secs_f64();
    let ahla_us = elapsed_ahla.as_micros() as f64 / ahla_steps;

    // ── Print results ──
    println!("\n┌── SIMD Benchmark ({level_name}, {iters} iters) ──────────────────────────────┐");
    println!("│ {:<20} {:>14} {:>14} │", "Operation", "ops/s", "µs/op");
    println!("│ {} │", "─".repeat(50));

    for (label, tps) in &matmul_results {
        let us = 1_000_000.0 / tps;
        println!(
            "│ {:<20} {:>14.0} {:>14.2} │",
            format!("matmul [{label}]"),
            tps,
            us
        );
    }
    for &(hd, tps) in &hla_update_tps {
        let us = 1_000_000.0 / tps;
        println!(
            "│ {:<20} {:>14.0} {:>14.2} │",
            format!("hla_update hd={hd}"),
            tps,
            us
        );
    }
    for &(hd, tps) in &ahla_step_tps {
        let us = 1_000_000.0 / tps;
        println!(
            "│ {:<20} {:>14.0} {:>14.2} │",
            format!("ahla_step hd={hd}"),
            tps,
            us
        );
    }
    println!("│ {} │", "─".repeat(50));
    println!(
        "│ {:<20} {:>14.0} {:>14.2} │",
        "forward_hla (micro)", hla_tps, hla_us
    );
    println!(
        "│ {:<20} {:>14.0} {:>14.2} │",
        "forward_ahla (micro)", ahla_tps, ahla_us
    );
    println!("└──────────────────────────────────────────────────────────┘");

    BenchResult {
        label: format!("simd ({level_name}, hla={hla_tps:.0} tps)"),
        throughput: hla_tps,
        time_per_step_us: hla_us,
        avg_acceptance_len: ahla_tps,
        color: (0, 200, 150),
        category: BenchCategory::Infrastructure,
        feature_dim: "Attn".into(),
    }
}
