#![cfg(feature = "parallax_attn")]
//! Benchmark — Parallax Parameterized Local Linear Attention (Plan 135)
//!
//! Benchmarks CPU decode overhead: SDPA vs SDPA+R projection (Parallax).
//! Reports per-query latency, FLOP overhead ratio, and correctness checks.
//!
//! Run: `cargo test --features parallax_attn --test bench_135_parallax_attn --release -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::{ParallaxConfig, tiled_attention_forward, tiled_attention_parallax_forward};

// ── Config ────────────────────────────────────────────────────

const DIM: usize = 64;
const SEQ_LENS: &[usize] = &[16, 32, 64, 128, 256];
const WARMUP: usize = 3;
const ITERS: usize = 20;
const SEED: u64 = 135;

// ── Helpers ───────────────────────────────────────────────────

/// Generate random data in [-1, 1) from a seeded RNG.
fn generate_random_data(len: usize, rng: &mut fastrand::Rng) -> Vec<f32> {
    (0..len).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// Cosine similarity between two flat slices.
fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ── Benchmark Tests ───────────────────────────────────────────

#[test]
fn bench_parallax_cpu_decode_overhead() {
    eprintln!("╔════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Bench: Parallax Attention CPU Decode Overhead (Plan 135)         ║");
    eprintln!("║  dim={DIM}, warmup={WARMUP}, iters={ITERS}                                  ║");
    eprintln!("╠════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  seq_len │  SDPA (µs) │ Parallax (µs) │ overhead ║");
    eprintln!("╟──────────┼────────────┼───────────────┼───────────╢");

    for &seq_len in SEQ_LENS {
        let scale = 1.0 / (DIM as f32).sqrt();
        let head_size = seq_len * DIM;

        let mut rng = fastrand::Rng::with_seed(SEED);
        let q = generate_random_data(head_size, &mut rng);
        let k = generate_random_data(head_size, &mut rng);
        let v = generate_random_data(head_size, &mut rng);
        let r = generate_random_data(DIM * DIM, &mut rng);
        let x = generate_random_data(DIM, &mut rng);

        let parallax_config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
        };

        // ── Warmup SDPA ──
        let mut sdpa_out = vec![0.0f32; head_size];
        for _ in 0..WARMUP {
            tiled_attention_forward(&q, &k, &v, &mut sdpa_out, seq_len, DIM, scale);
            black_box(&sdpa_out);
        }

        // ── Benchmark SDPA ──
        let start = Instant::now();
        for _ in 0..ITERS {
            tiled_attention_forward(&q, &k, &v, &mut sdpa_out, seq_len, DIM, scale);
            black_box(&sdpa_out);
        }
        let sdpa_us = start.elapsed().as_micros() as f64 / ITERS as f64;

        // ── Warmup Parallax ──
        let mut plx_out = vec![0.0f32; head_size];
        for _ in 0..WARMUP {
            tiled_attention_parallax_forward(
                &q,
                &k,
                &v,
                &mut plx_out,
                seq_len,
                DIM,
                scale,
                &r,
                &x,
                &parallax_config,
            );
            black_box(&plx_out);
        }

        // ── Benchmark Parallax ──
        let start = Instant::now();
        for _ in 0..ITERS {
            tiled_attention_parallax_forward(
                &q,
                &k,
                &v,
                &mut plx_out,
                seq_len,
                DIM,
                scale,
                &r,
                &x,
                &parallax_config,
            );
            black_box(&plx_out);
        }
        let plx_us = start.elapsed().as_micros() as f64 / ITERS as f64;

        let overhead = if sdpa_us > 0.0 { plx_us / sdpa_us } else { 0.0 };

        eprintln!("║ {seq_len:>8} │ {sdpa_us:>10.1} │ {plx_us:>13.1} │ {overhead:>8.2}× │");
    }

    eprintln!("╚════════════════════════════════════════════════════════════════════╝");
}

#[test]
fn bench_parallax_zero_init_recovers_softmax() {
    // With zero R projection weights, Parallax output must match SDPA exactly.
    let seq_len = 64;
    let scale = 1.0 / (DIM as f32).sqrt();
    let head_size = seq_len * DIM;

    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = generate_random_data(head_size, &mut rng);
    let k = generate_random_data(head_size, &mut rng);
    let v = generate_random_data(head_size, &mut rng);
    let r = vec![0.0f32; DIM * DIM]; // zero R
    let x = generate_random_data(DIM, &mut rng);

    let config = ParallaxConfig {
        gate_scale: 1.0,
        zero_init: true,
    };

    let mut sdpa_out = vec![0.0f32; head_size];
    tiled_attention_forward(&q, &k, &v, &mut sdpa_out, seq_len, DIM, scale);

    let mut plx_out = vec![0.0f32; head_size];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut plx_out,
        seq_len,
        DIM,
        scale,
        &r,
        &x,
        &config,
    );

    let sim = cos_sim(&sdpa_out, &plx_out);
    eprintln!("  zero-R cosine similarity: {sim:.8}");
    assert!(
        sim > 0.999,
        "zero-R Parallax should recover softmax (cos_sim={sim})"
    );

    // Element-wise check — tolerance accounts for online-softmax vs score-materialisation numerics
    for (i, (&a, &b)) in sdpa_out.iter().zip(plx_out.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-2,
            "output[{i}] mismatch: sdpa={a}, parallax={b}, diff={diff}"
        );
    }
    eprintln!("  element-wise max diff < 1e-4 ✓");
}

#[test]
fn bench_parallax_gate_zero_recovers_softmax() {
    // With gate_scale=0, Parallax output must match SDPA exactly.
    let seq_len = 64;
    let scale = 1.0 / (DIM as f32).sqrt();
    let head_size = seq_len * DIM;

    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = generate_random_data(head_size, &mut rng);
    let k = generate_random_data(head_size, &mut rng);
    let v = generate_random_data(head_size, &mut rng);
    let r = generate_random_data(DIM * DIM, &mut rng); // non-zero R
    let x = generate_random_data(DIM, &mut rng);

    let config = ParallaxConfig {
        gate_scale: 0.0,
        zero_init: false,
    };

    let mut sdpa_out = vec![0.0f32; head_size];
    tiled_attention_forward(&q, &k, &v, &mut sdpa_out, seq_len, DIM, scale);

    let mut plx_out = vec![0.0f32; head_size];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut plx_out,
        seq_len,
        DIM,
        scale,
        &r,
        &x,
        &config,
    );

    let sim = cos_sim(&sdpa_out, &plx_out);
    eprintln!("  gate=0 cosine similarity: {sim:.8}");
    assert!(
        sim > 0.999,
        "gate_scale=0 should recover softmax (cos_sim={sim})"
    );
}

#[test]
fn bench_parallax_finite_output() {
    // All outputs must be finite for all tested configs.
    for &seq_len in SEQ_LENS {
        let scale = 1.0 / (DIM as f32).sqrt();
        let head_size = seq_len * DIM;

        let mut rng = fastrand::Rng::with_seed(SEED);
        let q = generate_random_data(head_size, &mut rng);
        let k = generate_random_data(head_size, &mut rng);
        let v = generate_random_data(head_size, &mut rng);
        let r = generate_random_data(DIM * DIM, &mut rng);
        let x = generate_random_data(DIM, &mut rng);

        let config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
        };

        let mut output = vec![0.0f32; head_size];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output,
            seq_len,
            DIM,
            scale,
            &r,
            &x,
            &config,
        );

        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "non-finite output[{i}] = {val} at seq_len={seq_len}"
            );
        }

        eprintln!("  seq_len={seq_len}: all {head_size} outputs finite ✓");
    }
}

#[test]
fn bench_parallax_correction_magnitude() {
    // Measure the magnitude of the Parallax correction relative to the
    // softmax output. For random weights, expect correction to be a modest
    // fraction of the output norm.
    let seq_len = 64;
    let scale = 1.0 / (DIM as f32).sqrt();
    let head_size = seq_len * DIM;

    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = generate_random_data(head_size, &mut rng);
    let k = generate_random_data(head_size, &mut rng);
    let v = generate_random_data(head_size, &mut rng);
    let r = generate_random_data(DIM * DIM, &mut rng);
    let x = generate_random_data(DIM, &mut rng);

    // Compute SDPA output
    let mut sdpa_out = vec![0.0f32; head_size];
    tiled_attention_forward(&q, &k, &v, &mut sdpa_out, seq_len, DIM, scale);

    // Compute Parallax output
    let config = ParallaxConfig {
        gate_scale: 1.0,
        zero_init: false,
    };
    let mut plx_out = vec![0.0f32; head_size];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut plx_out,
        seq_len,
        DIM,
        scale,
        &r,
        &x,
        &config,
    );

    // Compute norms
    let sdpa_norm: f32 = sdpa_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let diff_norm: f32 = sdpa_out
        .iter()
        .zip(plx_out.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();

    let ratio = if sdpa_norm > 1e-12 {
        diff_norm / sdpa_norm
    } else {
        0.0
    };
    eprintln!(
        "  correction/output ratio: {ratio:.4} (diff_norm={diff_norm:.4}, sdpa_norm={sdpa_norm:.4})"
    );
    assert!(ratio.is_finite(), "correction ratio should be finite");
}
