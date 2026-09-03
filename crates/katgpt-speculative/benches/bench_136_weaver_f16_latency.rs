//! Issue 136 — Weaver f16 vs f32 parallel forward latency benchmark.
//!
//! Measures the G2 (latency gain) GOAT gate: f16 parallel forward should be
//! ≥1.5× faster than f32 parallel forward on the real config size
//! (hidden=2304, d_ff=5824, seq_len=5, K=32).
//!
//! The f16 path halves weight-read bandwidth. The f16→f32 conversion happens
//! inside `simd_fused_scale_acc_f16` at 1 cycle per 4 elements on aarch64 NEON.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/136_weaver_f16 \
//!   cargo run -p katgpt-speculative --features weaver_runtime \
//!     --bench bench_136_weaver_f16_latency --release -- --nocapture
//! ```

use katgpt_speculative::weaver::{
    WeaverConfig, WeaverCorrector, WeaverCorrectorF16, WeaverInput, WeaverScratch, WeaverWeights,
    weaver_forward_parallel,
};

/// Real-data config (matches the trained checkpoint: Gemma2-2B hidden=2304).
fn real_config() -> WeaverConfig {
    WeaverConfig {
        hidden_dim: 2304,
        n_heads: 16,
        k_candidates: 32,
        n_layer: 1,
        d_ff: 5824,
        rms_eps: 1e-6,
        max_depth: 4,
    }
}

/// Build deterministic non-zero weights at the real config scale.
/// Uses a fixed seed for reproducibility.
fn real_weights(cfg: &WeaverConfig) -> WeaverWeights {
    let mut w = WeaverWeights::zeros(cfg.clone());
    // Unit norm scales.
    for s in &mut w.norm_cond {
        *s = 1.0;
    }
    for s in &mut w.norm_attn {
        *s = 1.0;
    }
    for s in &mut w.norm_mlp {
        *s = 1.0;
    }
    // Identity W_c (preserves the hidden state through conditioning).
    for i in 0..cfg.hidden_dim {
        w.w_c[i * cfg.hidden_dim + i] = 1.0;
    }
    // Deterministic small weights for other matrices.
    let mut seed = 42u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0 - 0.5
    };
    for w_val in &mut w.w_q {
        *w_val = rng() * 0.02;
    }
    for w_val in &mut w.w_k {
        *w_val = rng() * 0.02;
    }
    for w_val in &mut w.w_v {
        *w_val = rng() * 0.02;
    }
    for w_val in &mut w.w_o {
        *w_val = rng() * 0.02;
    }
    for w_val in &mut w.w_gate {
        *w_val = rng() * 0.02;
    }
    for w_val in &mut w.w_up {
        *w_val = rng() * 0.02;
    }
    for w_val in &mut w.w_down {
        *w_val = rng() * 0.02;
    }
    for w_val in &mut w.pos_emb {
        *w_val = rng() * 0.01;
    }
    w
}

/// Build a realistic WeaverInput (leaked to get 'static lifetime).
fn real_input(cfg: &WeaverConfig) -> WeaverInput<'static> {
    let h = cfg.hidden_dim;
    let k = cfg.k_candidates;
    let d = cfg.max_depth;
    let vocab = 256; // small vocab for the benchmark (embedding is not the bottleneck)

    let h_verifier: &'static [f32] = Box::leak(vec![0.5f32; h].into_boxed_slice());
    let mut h_dflash: Vec<&'static [f32]> = Vec::with_capacity(d);
    let mut topk_ids: Vec<&'static [u32]> = Vec::with_capacity(d);
    let mut dflash_logits: Vec<&'static [f32]> = Vec::with_capacity(d);
    for di in 0..d {
        h_dflash.push(Box::leak(
            (0..h)
                .map(|i| 0.3 + 0.001 * (di + i) as f32)
                .collect::<Vec<f32>>()
                .into_boxed_slice(),
        ));
        topk_ids.push(Box::leak(
            (0..k)
                .map(|i| (i as u32) % vocab as u32)
                .collect::<Vec<u32>>()
                .into_boxed_slice(),
        ));
        dflash_logits.push(Box::leak(
            (0..k)
                .map(|i| (i as f32) * 0.1)
                .collect::<Vec<f32>>()
                .into_boxed_slice(),
        ));
    }
    let emb: &'static [f32] = Box::leak(vec![0.1f32; vocab * h].into_boxed_slice());

    WeaverInput {
        h_verifier,
        h_dflash: Box::leak(h_dflash.into_boxed_slice()),
        topk_ids: Box::leak(topk_ids.into_boxed_slice()),
        dflash_logits: Box::leak(dflash_logits.into_boxed_slice()),
        embedding: emb,
        vocab_size: vocab,
    }
}

fn main() {
    let cfg = real_config();
    let weights_f32 = real_weights(&cfg);
    let corrector_f16 =
        WeaverCorrectorF16::from_f32(&WeaverCorrector::from_weights(weights_f32.clone()));
    let input = real_input(&cfg);

    // Warmup: 5 iterations on each path to stabilize caches / thread pool.
    {
        let mut scratch = WeaverScratch::new(&cfg);
        for _ in 0..5 {
            weaver_forward_parallel(&weights_f32, &input, &mut scratch);
        }
    }
    {
        let mut scratch = WeaverScratch::new(&cfg);
        for _ in 0..5 {
            corrector_f16.correct_parallel(&input, &mut scratch);
        }
    }

    // Measure: 50 iterations, report median + p99.
    const N: usize = 50;
    let mut times_f32 = Vec::with_capacity(N);
    let mut times_f16 = Vec::with_capacity(N);

    {
        let mut scratch = WeaverScratch::new(&cfg);
        for _ in 0..N {
            let start = std::time::Instant::now();
            weaver_forward_parallel(&weights_f32, &input, &mut scratch);
            times_f32.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    {
        let mut scratch = WeaverScratch::new(&cfg);
        for _ in 0..N {
            let start = std::time::Instant::now();
            corrector_f16.correct_parallel(&input, &mut scratch);
            times_f16.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    times_f32.sort_by(|a, b| a.total_cmp(b));
    times_f16.sort_by(|a, b| a.total_cmp(b));

    let median = |v: &[f64]| v[v.len() / 2];
    // N = 50, so `floor(50 * 0.99) = 49 = n - 1`: the old closure returned the
    // MAXIMUM under a p99 label, and printed it in the table below beside a
    // speedup ratio. `floor(n * p) == n - 1` for every n <= 1/(1-p), i.e.
    // n <= 100 at p99. Nearest rank (`ceil(p*n) - 1`, integer so 0.99's
    // inexact binary form cannot round back onto the max) is the correct form
    // -- but at N = 50 it is STILL the max, because a 99th percentile does not
    // exist in 50 samples: no rank below the maximum holds 99% of the mass.
    // So the honest fix is the label, not only the arithmetic -- the tail
    // support is printed with the number so the table cannot be read as a
    // quantile. (.issues/722; scripts/percentile_index_audit.py.)
    let p99_idx = |len: usize| (len * 99).div_ceil(100).saturating_sub(1).min(len - 1);
    let p99 = |v: &[f64]| v[p99_idx(v.len())];
    let p99_support = |v: &[f64]| v.len() - p99_idx(v.len());

    let med_f32 = median(&times_f32);
    let med_f16 = median(&times_f16);
    let speedup = med_f32 / med_f16;

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│ Issue 136 — Weaver f16 vs f32 Latency Benchmark     │");
    println!("├──────────────────┬──────────┬──────────┬────────────┤");
    println!("│ Path             │ Median   │ P99      │ Speedup    │");
    println!("├──────────────────┼──────────┼──────────┼────────────┤");
    println!(
        "│ f32 parallel     │ {:>6.2}ms │ {:>6.2}ms │ baseline   │",
        med_f32,
        p99(&times_f32)
    );
    println!(
        "│ f16 parallel     │ {:>6.2}ms │ {:>6.2}ms │ {:>.2}×      │",
        med_f16,
        p99(&times_f16),
        speedup
    );
    println!("└──────────────────┴──────────┴──────────┴────────────┘");
    println!(
        "  P99 tail support: {} of {} samples (f32) / {} of {} (f16) — \
         at N={N} a 99th percentile does not exist in the sample; \
         read the P99 column as the worst observation.",
        p99_support(&times_f32),
        times_f32.len(),
        p99_support(&times_f16),
        times_f16.len(),
    );
    println!();
    println!(
        "Config: hidden={}, d_ff={}, seq_len={}, K={}",
        cfg.hidden_dim,
        cfg.d_ff,
        cfg.max_depth + 1,
        cfg.k_candidates
    );
    println!("Iterations: {N}");
    println!();

    if speedup >= 1.5 {
        println!("✅ G2 PASS: f16 is {speedup:.2}× faster (≥1.5× target)");
    } else if speedup >= 1.2 {
        println!("⚠️  G2 MARGINAL: f16 is {speedup:.2}× faster (≥1.2× but <1.5× target)");
    } else {
        println!(
            "❌ G2 FAIL: f16 is {speedup:.2}× faster (<1.2× — f16 conversion overhead exceeds bandwidth savings)"
        );
    }

    // Also measure the f32→f16 conversion cost (one-time at load time).
    let conv_start = std::time::Instant::now();
    let _f16_weights =
        WeaverCorrectorF16::from_f32(&WeaverCorrector::from_weights(weights_f32.clone()));
    let conv_ms = conv_start.elapsed().as_secs_f64() * 1000.0;
    println!();
    println!("f32→f16 conversion (one-time load cost): {conv_ms:.1} ms");
}
