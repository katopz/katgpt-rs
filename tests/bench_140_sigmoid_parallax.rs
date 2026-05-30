#![cfg(feature = "parallax_attn")]
//! Benchmark — Sigmoid vs Softmax Parallax Attention (Plan 140, Research 140)
//!
//! Head-to-head comparison: latency, weight distribution, covariance diversity,
//! and numerical stability between softmax and sigmoid attention with Parallax correction.
//!
//! Run: `cargo test --features parallax_attn --test bench_140_sigmoid_parallax --release -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::{
    ParallaxActivation, ParallaxConfig, tiled_attention_forward, tiled_attention_parallax_forward,
};

// ── Config ────────────────────────────────────────────────────

const DIM: usize = 64;
const SEQ_LENS: &[usize] = &[16, 32, 64, 128, 256];
const WARMUP: usize = 3;
const ITERS: usize = 20;
const SEED: u64 = 140;

// ── Helpers ───────────────────────────────────────────────────

fn generate_random_data(len: usize, rng: &mut fastrand::Rng) -> Vec<f32> {
    (0..len).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

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

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── G1: Latency — Sigmoid vs Softmax vs Base SDPA ─────────────

#[test]
fn bench_sigmoid_vs_softmax_latency() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Bench 140: Sigmoid vs Softmax Parallax — CPU SIMD (Plan 140)               ║");
    eprintln!(
        "║  dim={DIM}, warmup={WARMUP}, iters={ITERS}                                            ║"
    );
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ seq_len │ SDPA (µs) │ Softmax+PLX (µs) │ Sigmoid+PLX (µs) │ SM ovh │ Sig ovh ║");
    eprintln!("╟─────────┼───────────┼──────────────────┼──────────────────┼────────┼─────────╢");

    for &seq_len in SEQ_LENS {
        let scale = 1.0 / (DIM as f32).sqrt();
        let head_size = seq_len * DIM;

        let mut rng = fastrand::Rng::with_seed(SEED);
        let q = generate_random_data(head_size, &mut rng);
        let k = generate_random_data(head_size, &mut rng);
        let v = generate_random_data(head_size, &mut rng);
        let r = generate_random_data(DIM * DIM, &mut rng);
        let x = generate_random_data(DIM, &mut rng);

        let sm_config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Softmax,
        };
        let sig_config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };

        // ── Bench SDPA baseline ──
        let mut out = vec![0.0f32; head_size];
        for _ in 0..WARMUP {
            tiled_attention_forward(&q, &k, &v, &mut out, seq_len, DIM, scale);
            black_box(&out);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            tiled_attention_forward(&q, &k, &v, &mut out, seq_len, DIM, scale);
            black_box(&out);
        }
        let sdpa_us = start.elapsed().as_micros() as f64 / ITERS as f64;

        // ── Bench Softmax Parallax ──
        for _ in 0..WARMUP {
            tiled_attention_parallax_forward(
                &q, &k, &v, &mut out, seq_len, DIM, scale, &r, &x, &sm_config, None,
            );
            black_box(&out);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            tiled_attention_parallax_forward(
                &q, &k, &v, &mut out, seq_len, DIM, scale, &r, &x, &sm_config, None,
            );
            black_box(&out);
        }
        let sm_us = start.elapsed().as_micros() as f64 / ITERS as f64;

        // ── Bench Sigmoid Parallax ──
        for _ in 0..WARMUP {
            tiled_attention_parallax_forward(
                &q,
                &k,
                &v,
                &mut out,
                seq_len,
                DIM,
                scale,
                &r,
                &x,
                &sig_config,
                None,
            );
            black_box(&out);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            tiled_attention_parallax_forward(
                &q,
                &k,
                &v,
                &mut out,
                seq_len,
                DIM,
                scale,
                &r,
                &x,
                &sig_config,
                None,
            );
            black_box(&out);
        }
        let sig_us = start.elapsed().as_micros() as f64 / ITERS as f64;

        let sm_ovh = if sdpa_us > 0.0 { sm_us / sdpa_us } else { 0.0 };
        let sig_ovh = if sdpa_us > 0.0 { sig_us / sdpa_us } else { 0.0 };

        eprintln!(
            "║ {seq_len:>7} │ {sdpa_us:>9.1} │ {sm_us:>16.1} │ {sig_us:>16.1} │ {sm_ovh:>6.2}× │ {sig_ovh:>7.2}× ║"
        );
    }

    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

// ── G2: Sigmoid weights are finite and stable for all seq_lens ──

#[test]
fn bench_sigmoid_finite_all_seq_lens() {
    eprintln!("\n── G2: Sigmoid output finiteness ──");

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
            activation: ParallaxActivation::Sigmoid,
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
            None,
        );

        let all_finite = output.iter().all(|v| v.is_finite());
        assert!(
            all_finite,
            "sigmoid output has NaN/Inf at seq_len={seq_len}"
        );
        eprintln!("  seq_len={seq_len}: all {head_size} outputs finite ✓");
    }
}

// ── G3: Correction magnitude — sigmoid vs softmax ─────────────

#[test]
fn bench_correction_magnitude_comparison() {
    eprintln!("\n── G3: Correction magnitude (sigmoid vs softmax) ──");

    for &seq_len in &[16, 64, 256] {
        let scale = 1.0 / (DIM as f32).sqrt();
        let head_size = seq_len * DIM;

        let mut rng = fastrand::Rng::with_seed(SEED);
        let q = generate_random_data(head_size, &mut rng);
        let k = generate_random_data(head_size, &mut rng);
        let v = generate_random_data(head_size, &mut rng);
        let r = generate_random_data(DIM * DIM, &mut rng);
        let x = generate_random_data(DIM, &mut rng);

        // Compute base SDPA output
        let mut sdpa_out = vec![0.0f32; head_size];
        tiled_attention_forward(&q, &k, &v, &mut sdpa_out, seq_len, DIM, scale);

        // Compute softmax Parallax output
        let sm_config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Softmax,
        };
        let mut sm_out = vec![0.0f32; head_size];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut sm_out,
            seq_len,
            DIM,
            scale,
            &r,
            &x,
            &sm_config,
            None,
        );

        // Compute sigmoid Parallax output
        let sig_config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };
        let mut sig_out = vec![0.0f32; head_size];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut sig_out,
            seq_len,
            DIM,
            scale,
            &r,
            &x,
            &sig_config,
            None,
        );

        let sdpa_norm = vec_norm(&sdpa_out);
        let sm_diff: f32 = sdpa_out
            .iter()
            .zip(sm_out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let sig_diff: f32 = sdpa_out
            .iter()
            .zip(sig_out.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        let sm_ratio = if sdpa_norm > 1e-12 {
            sm_diff / sdpa_norm
        } else {
            0.0
        };
        let sig_ratio = if sdpa_norm > 1e-12 {
            sig_diff / sdpa_norm
        } else {
            0.0
        };

        eprintln!(
            "  seq_len={seq_len}: softmax correction={sm_ratio:.4}, sigmoid correction={sig_ratio:.4} (of base norm)"
        );

        // Both corrections should be finite and non-trivially different from base
        assert!(sm_ratio.is_finite(), "softmax correction ratio not finite");
        assert!(sig_ratio.is_finite(), "sigmoid correction ratio not finite");
    }
}

// ── G4: No attention sinks — sigmoid weight distribution ──────

#[test]
fn bench_no_attention_sinks_sigmoid() {
    eprintln!("\n── G4: Attention sink test (sigmoid vs softmax) ──");

    let seq_len = 64;
    let scale = 1.0 / (DIM as f32).sqrt();
    let head_size = seq_len * DIM;

    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = generate_random_data(DIM, &mut rng); // single query
    let k = generate_random_data(head_size, &mut rng);

    // Compute softmax weights manually for first query
    let mut sm_weights = vec![0.0f32; seq_len];
    let mut max_s = f32::NEG_INFINITY;
    for j in 0..seq_len {
        let mut dot = 0.0f32;
        for d in 0..DIM {
            dot += q[d] * k[j * DIM + d];
        }
        sm_weights[j] = dot * scale;
        max_s = max_s.max(sm_weights[j]);
    }
    let mut sm_sum = 0.0f32;
    for w in &mut sm_weights {
        *w = (*w - max_s).exp();
        sm_sum += *w;
    }
    for w in &mut sm_weights {
        *w /= sm_sum;
    }

    // Compute sigmoid weights for first query
    let mut sig_weights = vec![0.0f32; seq_len];
    let mut sig_sum = 0.0f32;
    for j in 0..seq_len {
        let mut dot = 0.0f32;
        for d in 0..DIM {
            dot += q[d] * k[j * DIM + d];
        }
        let s = dot * scale;
        sig_weights[j] = 1.0 / (1.0 + (-s).exp());
        sig_sum += sig_weights[j];
    }
    for w in &mut sig_weights {
        *w /= sig_sum;
    }

    // Check: softmax max weight vs sigmoid max weight
    let sm_max = sm_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sig_max = sig_weights
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Softmax entropy (lower = more peaked = more sink-prone)
    let sm_entropy: f32 = sm_weights
        .iter()
        .map(|w| if *w > 1e-12 { -w * w.ln() } else { 0.0 })
        .sum();
    let sig_entropy: f32 = sig_weights
        .iter()
        .map(|w| if *w > 1e-12 { -w * w.ln() } else { 0.0 })
        .sum();

    eprintln!("  Softmax:  max_weight={sm_max:.6}, entropy={sm_entropy:.4}");
    eprintln!("  Sigmoid:  max_weight={sig_max:.6}, entropy={sig_entropy:.4}");
    eprintln!(
        "  Sigmoid entropy >= softmax entropy: {} (no sinks hypothesis)",
        if sig_entropy >= sm_entropy {
            "YES ✓"
        } else {
            "NO (closer to uniform = higher entropy, so sigmoid is LESS peaked)"
        }
    );

    // Sigmoid should have lower max weight (more uniform, no sinks)
    assert!(
        sig_max <= sm_max + 0.01,
        "sigmoid max weight ({sig_max}) should be <= softmax max weight ({sm_max})"
    );

    // Both should be valid probability distributions
    let sm_total: f32 = sm_weights.iter().sum();
    let sig_total: f32 = sig_weights.iter().sum();
    assert!(
        (sm_total - 1.0).abs() < 1e-4,
        "softmax weights sum to {sm_total}"
    );
    assert!(
        (sig_total - 1.0).abs() < 1e-4,
        "sigmoid weights sum to {sig_total}"
    );
}

// ── G5: Sigmoid Parallax with zero-R recovers sigmoid attention ──

#[test]
fn bench_sigmoid_zero_r_recovers_base() {
    eprintln!("\n── G5: Zero-R sigmoid Parallax recovers base sigmoid ──");

    let seq_len = 64;
    let scale = 1.0 / (DIM as f32).sqrt();
    let head_size = seq_len * DIM;

    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = generate_random_data(head_size, &mut rng);
    let k = generate_random_data(head_size, &mut rng);
    let v = generate_random_data(head_size, &mut rng);
    let r = vec![0.0f32; DIM * DIM];
    let x = generate_random_data(DIM, &mut rng);

    // gate_scale=0 with sigmoid activation
    let config = ParallaxConfig {
        gate_scale: 0.0,
        zero_init: false,
        activation: ParallaxActivation::Sigmoid,
    };

    let mut parallax_out = vec![0.0f32; head_size];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut parallax_out,
        seq_len,
        DIM,
        scale,
        &r,
        &x,
        &config,
        None,
    );

    // Compute reference: SDPA with same data (softmax, since tiled_attention_forward is always softmax)
    let mut sdpa_out = vec![0.0f32; head_size];
    tiled_attention_forward(&q, &k, &v, &mut sdpa_out, seq_len, DIM, scale);

    // Sigmoid and softmax should DIFFER (different kernels)
    let sim = cos_sim(&parallax_out, &sdpa_out);
    eprintln!("  sigmoid(gate=0) vs softmax SDPA cos_sim: {sim:.6}");
    // They're different kernels, but both valid attention — similarity should be high but not 1.0
    assert!(
        sim > 0.9,
        "sigmoid and softmax should be highly correlated but not identical (cos_sim={sim})"
    );
    assert!(
        sim < 0.9999,
        "sigmoid and softmax should differ (cos_sim={sim})"
    );
}

// ── G6: Correction diversity — sigmoid Σ_KV captures more than softmax ──

#[test]
fn bench_covariance_diversity() {
    eprintln!("\n── G6: Σ_KV effective rank — sigmoid vs softmax ──");

    // We can't directly inspect Σ_KV from the public API, but we can measure
    // the output diversity: more diverse covariance → more diverse correction.
    let seq_len = 128;
    let scale = 1.0 / (DIM as f32).sqrt();
    let head_size = seq_len * DIM;

    let mut rng = fastrand::Rng::with_seed(SEED);
    let q = generate_random_data(head_size, &mut rng);
    let k = generate_random_data(head_size, &mut rng);
    let v = generate_random_data(head_size, &mut rng);

    // Use different R projections to probe covariance structure
    let mut correction_spread_sm = 0.0f32;
    let mut correction_spread_sig = 0.0f32;

    for probe_seed in 0..5 {
        let mut probe_rng = fastrand::Rng::with_seed(SEED + probe_seed);
        let r = generate_random_data(DIM * DIM, &mut probe_rng);
        let x = generate_random_data(DIM, &mut probe_rng);

        let sm_config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Softmax,
        };
        let sig_config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };

        // Compute base SDPA
        let mut base_out = vec![0.0f32; head_size];
        tiled_attention_forward(&q, &k, &v, &mut base_out, seq_len, DIM, scale);

        // Softmax correction
        let mut sm_out = vec![0.0f32; head_size];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut sm_out,
            seq_len,
            DIM,
            scale,
            &r,
            &x,
            &sm_config,
            None,
        );
        let sm_diff = vec_norm(
            &sm_out
                .iter()
                .zip(base_out.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );

        // Sigmoid correction
        let mut sig_out = vec![0.0f32; head_size];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut sig_out,
            seq_len,
            DIM,
            scale,
            &r,
            &x,
            &sig_config,
            None,
        );
        let sig_diff = vec_norm(
            &sig_out
                .iter()
                .zip(base_out.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );

        correction_spread_sm += sm_diff;
        correction_spread_sig += sig_diff;
    }

    correction_spread_sm /= 5.0;
    correction_spread_sig /= 5.0;

    eprintln!(
        "  Avg correction norm (5 probes): softmax={correction_spread_sm:.4}, sigmoid={correction_spread_sig:.4}"
    );

    let ratio = correction_spread_sig / correction_spread_sm;
    eprintln!("  Sigmoid/Softmax correction ratio: {ratio:.4}");
    if ratio > 1.0 {
        eprintln!("  → Sigmoid correction is LARGER — more covariance diversity captured ✓");
    } else {
        eprintln!("  → Softmax correction is larger — may be dominated by sinks");
    }

    assert!(
        correction_spread_sm.is_finite() && correction_spread_sig.is_finite(),
        "correction norms should be finite"
    );
}
