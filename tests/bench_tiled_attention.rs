#![cfg(feature = "tiled_attention")]
//! Benchmark — Tiled Online-Softmax Attention (Plan 115)
//!
//! Compares tiled attention vs full-materialization reference across
//! seq_len = {64, 128, 256, 512, 1024, 2048}.
//!
//! Metrics: throughput (μs), cosine similarity (accuracy), no NaN/Inf.
//!
//! Run: `cargo test --features tiled_attention --test bench_tiled_attention -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::tiled_attention_forward;

// ── Config ────────────────────────────────────────────────────

const HEADS: usize = 8;
const DIM: usize = 64;
const SEQ_LENS: &[usize] = &[64, 128, 256, 512, 1024, 2048];
const WARMUP: usize = 3;
const ITERS: usize = 10;
const SEED: u64 = 42;

// ── Helpers ───────────────────────────────────────────────────

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
    match denom < 1e-12 {
        true => 0.0,
        false => dot / denom,
    }
}

/// Reference attention: full score matrix materialization with row-wise softmax.
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    if seq_len == 0 {
        return;
    }

    // 1. scores = Q @ K.T  (seq_len × seq_len)
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot;
        }
    }

    // 2. Row-wise scaled softmax
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row = &mut scores[row_start..row_start + seq_len];

        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for val in row.iter_mut() {
            *val = ((*val - max_val) * scale).exp();
            sum += *val;
        }

        let inv_sum = 1.0 / sum;
        for val in row.iter_mut() {
            *val *= inv_sum;
        }
    }

    // 3. output = scores @ V
    for i in 0..seq_len {
        let scores_off = i * seq_len;
        let out_off = i * head_dim;
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                sum += scores[scores_off + j] * v[j * head_dim + d];
            }
            output[out_off + d] = sum;
        }
    }
}

/// Generate random data in [-1, 1) from a seeded RNG.
fn generate_random_data(len: usize, rng: &mut fastrand::Rng) -> Vec<f32> {
    (0..len).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// Benchmark a single (seq_len, head_dim) configuration.
/// Returns (reference_us, tiled_us, cosine_similarity).
fn bench_single(seq_len: usize, head_dim: usize) -> (f64, f64, f32) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_size = seq_len * head_dim;

    // Generate data with deterministic seed
    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = generate_random_data(head_size, &mut rng);
    let k = generate_random_data(head_size, &mut rng);
    let v = generate_random_data(head_size, &mut rng);

    // ── Warmup reference ──
    let mut ref_output = vec![0.0f32; head_size];
    for _ in 0..WARMUP {
        reference_attention(&q, &k, &v, &mut ref_output, seq_len, head_dim, scale);
        black_box(&ref_output);
    }

    // ── Benchmark reference ──
    let start_ref = Instant::now();
    for _ in 0..ITERS {
        reference_attention(&q, &k, &v, &mut ref_output, seq_len, head_dim, scale);
        black_box(&ref_output);
    }
    let ref_us = start_ref.elapsed().as_micros() as f64 / ITERS as f64;

    // ── Warmup tiled ──
    let mut tiled_output = vec![0.0f32; head_size];
    for _ in 0..WARMUP {
        tiled_attention_forward(&q, &k, &v, &mut tiled_output, seq_len, head_dim, scale);
        black_box(&tiled_output);
    }

    // ── Benchmark tiled ──
    let start_tiled = Instant::now();
    for _ in 0..ITERS {
        tiled_attention_forward(&q, &k, &v, &mut tiled_output, seq_len, head_dim, scale);
        black_box(&tiled_output);
    }
    let tiled_us = start_tiled.elapsed().as_micros() as f64 / ITERS as f64;

    // ── Cosine similarity ──
    let similarity = cos_sim(&ref_output, &tiled_output);

    (ref_us, tiled_us, similarity)
}

// ── Benchmark Tests ───────────────────────────────────────────

#[test]
fn bench_tiled_attention_throughput() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Bench: Tiled Online-Softmax Attention (Plan 115)           ║");
    eprintln!(
        "║  heads={HEADS}, dim={DIM}, warmup={WARMUP}, iters={ITERS}                       ║"
    );
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║ {{'seq_len':>8}} │ {{'ref (μs)':>10}} │ {{'tiled (μs)':>11}} │ {{'ratio':>8}} │ {{'cos_sim':>8}} │"
    );
    eprintln!("╟──────────┼────────────┼─────────────┼──────────┼──────────╢");

    let mut all_pass = true;

    for &seq_len in SEQ_LENS {
        let (ref_us, tiled_us, similarity) = bench_single(seq_len, DIM);

        let ratio = match ref_us > 0.0 {
            true => tiled_us / ref_us,
            false => 0.0,
        };

        // Verify no NaN/Inf in similarity
        assert!(
            similarity.is_finite(),
            "cosine similarity is not finite at seq_len={seq_len}"
        );

        // Verify similarity > 0.999
        let sim_pass = similarity > 0.999;
        if !sim_pass {
            all_pass = false;
        }

        let status = match sim_pass {
            true => "✓",
            false => "✗",
        };

        eprintln!(
            "║ {seq_len:>8} │ {ref_us:>10.1} │ {tiled_us:>11.1} │ {ratio:>7.2}x │ {similarity:>7.5}{status} │"
        );
    }

    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    assert!(
        all_pass,
        "one or more cosine similarity checks failed (threshold=0.999)"
    );
}

#[test]
fn bench_tiled_attention_finite_output() {
    // Verify no NaN/Inf at all configs
    for &seq_len in SEQ_LENS {
        let scale = 1.0 / (DIM as f32).sqrt();
        let head_size = seq_len * DIM;

        let mut rng = fastrand::Rng::with_seed(SEED);
        let q = generate_random_data(head_size, &mut rng);
        let k = generate_random_data(head_size, &mut rng);
        let v = generate_random_data(head_size, &mut rng);

        let mut output = vec![0.0f32; head_size];
        tiled_attention_forward(&q, &k, &v, &mut output, seq_len, DIM, scale);

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
fn bench_tiled_attention_peak_memory_estimate() {
    // Estimate peak memory per head for each path
    eprintln!("╔════════════════════════════════════════════════════════╗");
    eprintln!("║  Memory Estimate per Head (Plan 115)                  ║");
    eprintln!("╠════════════════════════════════════════════════════════╣");
    eprintln!(
        "║ {{'seq_len':>8}} │ {{'full (KB)':>10}} │ {{'tiled (KB)':>11}} │ {{'savings':>8}} │"
    );
    eprintln!("╟──────────┼────────────┼─────────────┼──────────╢");

    for &seq_len in SEQ_LENS {
        // Full materialization: N² + N*D + N*D + N*D (scores + Q + K + V + output)
        let full_bytes = seq_len * seq_len * 4 // score matrix
            + seq_len * DIM * 4   // Q
            + seq_len * DIM * 4   // K
            + seq_len * DIM * 4   // V
            + seq_len * DIM * 4; // output
        let full_kb = full_bytes as f64 / 1024.0;

        // Tiled: BR*BC*4 (score tile) + BR*D*4 (o_tile) + BR*4*3 (max+norm+max_new)
        //        + N*D*4 (output) + N*D*3 (Q,K,V read-only, no allocation)
        let tiled_bytes = 8 * 128 * 4    // score tile (BR × BC)
            + 8 * DIM * 4     // o_tile (BR × head_dim)
            + 8 * 4 * 3       // max_tile + norm_tile + max_new
            + seq_len * DIM * 4; // output
        let tiled_kb = tiled_bytes as f64 / 1024.0;

        let savings = match full_kb > 0.0 {
            true => full_kb / tiled_kb,
            false => 1.0,
        };

        eprintln!("║ {seq_len:>8} │ {full_kb:>9.1} │ {tiled_kb:>10.1} │ {savings:>6.1}×  │");
    }

    eprintln!("╚════════════════════════════════════════════════════════╝");
}
