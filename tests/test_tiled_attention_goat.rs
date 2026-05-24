#![cfg(feature = "tiled_attention")]
//! GOAT Proof Test — Tiled Online-Softmax Attention (Plan 115)
//!
//! Proves tiled attention output matches full-materialization output
//! to > 0.999 cosine similarity across multiple configurations.
//!
//! Configurations tested:
//! - (8 heads, 64 dim, seq=64)  — below threshold, uses fallback path
//! - (8 heads, 64 dim, seq=128) — at threshold boundary
//! - (8 heads, 64 dim, seq=256) — tiled path
//! - (8 heads, 64 dim, seq=512) — tiled path, larger
//!
//! Run: `cargo test --features tiled_attention --test test_tiled_attention_goat -- --nocapture`

use katgpt_core::tiled_attention_forward;

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

    // 2. Row-wise scaled softmax: same algorithm as types::softmax_scaled
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row = &mut scores[row_start..row_start + seq_len];

        // Pass 1: max
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Pass 2: exp + sum
        let mut sum = 0.0f32;
        for val in row.iter_mut() {
            *val = ((*val - max_val) * scale).exp();
            sum += *val;
        }

        // Pass 3: normalize
        let inv_sum = 1.0 / sum;
        for val in row.iter_mut() {
            *val *= inv_sum;
        }
    }

    // 3. output = scores @ V  (seq_len × head_dim)
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

// ── GOAT Proof ────────────────────────────────────────────────
//
// For each configuration (heads, dim, seq):
// 1. Generate random Q, K, V with seed 42
// 2. Compute reference output via full materialization
// 3. Compute tiled output via tiled_attention_forward
// 4. Assert cosine_similarity(reference, tiled) > 0.999
//
// The tiled path uses exp2() instead of exp() for the softmax,
// which may introduce tiny numerical differences. The 0.999
// threshold is generous — typical cosine similarity is > 0.9999.

#[test]
fn tiled_attention_cosine_similarity_goat() {
    let configs: &[(usize, usize, usize)] = &[
        (8, 64, 64),  // below threshold → fallback path (same algorithm)
        (8, 64, 128), // at threshold → fallback path (N < 128 is fallback, N=128 is tiled)
        (8, 64, 256), // tiled path
        (8, 64, 512), // tiled path, larger
    ];

    let threshold = 0.999;

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  GOAT Proof: Tiled Online-Softmax Attention (Plan 115)  ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");

    for &(heads, dim, seq) in configs {
        let scale = 1.0 / (dim as f32).sqrt();
        let head_size = seq * dim;

        eprintln!("\n── Config: heads={heads}, dim={dim}, seq={seq} ──");

        for head in 0..heads {
            // Deterministic per-head seeds from master seed 42
            let mut rng = fastrand::Rng::with_seed(42 + head as u64);
            let q = generate_random_data(head_size, &mut rng);
            let k = generate_random_data(head_size, &mut rng);
            let v = generate_random_data(head_size, &mut rng);

            // Reference output (full materialization)
            let mut ref_output = vec![0.0f32; head_size];
            reference_attention(&q, &k, &v, &mut ref_output, seq, dim, scale);

            // Tiled output
            let mut tiled_output = vec![0.0f32; head_size];
            tiled_attention_forward(&q, &k, &v, &mut tiled_output, seq, dim, scale);

            // Verify no NaN/Inf
            for (i, &val) in ref_output.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "ref_output[{i}] = {val} (NaN/Inf) at heads={heads}, dim={dim}, seq={seq}, head={head}"
                );
            }
            for (i, &val) in tiled_output.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "tiled_output[{i}] = {val} (NaN/Inf) at heads={heads}, dim={dim}, seq={seq}, head={head}"
                );
            }

            // Cosine similarity check
            let similarity = cos_sim(&ref_output, &tiled_output);
            assert!(
                similarity > threshold,
                "cosine similarity too low: heads={heads}, dim={dim}, seq={seq}, head={head}, cos_sim={similarity:.6} (threshold={threshold})"
            );

            eprintln!("  head {head:2}: cosine_similarity = {similarity:.7} ✓");
        }
    }

    eprintln!("\n✓ All GOAT proofs passed: cosine_similarity > {threshold}");
}
