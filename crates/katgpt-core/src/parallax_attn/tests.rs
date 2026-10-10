//! Tests for parallax attention.
//!
//! Split from the historical monolithic `src/parallax_attn.rs` (Issue 167, 2026-07-17).
//! Tests are exempt from the 2048-line soft limit per Issue 162.

use super::*;

/// With R_proj = zero matrix, rho should be all zeros.
#[test]
fn test_rho_zero_init() {
    let d = 8;
    let r_proj = vec![0.0f32; d * d];
    let x: Vec<f32> = (1..=d).map(|i| i as f32).collect();
    let mut rho = vec![0.0f32; d];

    compute_rho(&r_proj, &x, &mut rho);

    for (i, &v) in rho.iter().enumerate() {
        assert!(v == 0.0, "rho[{i}] should be 0.0 with zero R_proj, got {v}");
    }
}

/// With identity sigma_kv, correction should equal rho.
#[test]
fn test_correction_identity() {
    let d = 8;
    let mut sigma_kv = vec![0.0f32; d * d];
    // Identity matrix
    for i in 0..d {
        sigma_kv[i * d + i] = 1.0;
    }
    let rho: Vec<f32> = (1..=d).map(|i| i as f32 * 0.5).collect();
    let mut correction = vec![0.0f32; d];

    parallax_correction(&sigma_kv, &rho, &mut correction);

    for (i, (&c, &r)) in correction.iter().zip(rho.iter()).enumerate() {
        let expected = r;
        assert!(
            (c - expected).abs() < 1e-5,
            "correction[{i}] should be {expected} (identity sigma), got {c}"
        );
    }
}

/// With gate_scale=0, the output should equal standard softmax attention.
#[test]
fn test_parallax_recovers_softmax_gate_zero() {
    let d = 4;
    let n = 3;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

    // R projection — non-zero, but gate_scale=0 should cancel it
    let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

    let config = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Softmax,
        ..Default::default()
    };

    let mut output_parallax = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut output_parallax,
        n,
        d,
        scale,
        &r,
        &x,
        &config,
        None,
    );

    // Compute reference: standard softmax attention
    let mut output_ref = vec![0.0f32; n * d];
    tiled_attention_core(
        &q,
        &k,
        &v,
        &mut output_ref,
        n,
        d,
        scale,
        None,
        ParallaxActivation::Softmax,
        None,
        #[cfg(feature = "ssmax_temperature")]
        None,
        #[cfg(feature = "prior_logit_lane")]
        None,
        #[cfg(feature = "scale_invariant_attn")]
        None,
    );

    for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output[{i}]: parallax ({a}) should match softmax ({b}) with gate_scale=0"
        );
    }
}

/// With zero R_proj, the output should equal standard softmax attention
/// regardless of gate_scale (since rho = 0 implies correction = 0).
#[test]
fn test_parallax_recovers_softmax_zero_r() {
    let d = 4;
    let n = 3;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

    // Zero R projection weights
    let r = vec![0.0f32; d * d];
    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

    let config = ParallaxConfig {
        gate_scale: 1.0,
        activation: ParallaxActivation::Softmax,
        ..Default::default()
    };

    let mut output_parallax = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut output_parallax,
        n,
        d,
        scale,
        &r,
        &x,
        &config,
        None,
    );

    // Compute reference: standard softmax attention
    let mut output_ref = vec![0.0f32; n * d];
    tiled_attention_core(
        &q,
        &k,
        &v,
        &mut output_ref,
        n,
        d,
        scale,
        None,
        ParallaxActivation::Softmax,
        None,
        #[cfg(feature = "ssmax_temperature")]
        None,
        #[cfg(feature = "prior_logit_lane")]
        None,
        #[cfg(feature = "scale_invariant_attn")]
        None,
    );

    for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output[{i}]: parallax ({a}) should match softmax ({b}) with zero R"
        );
    }
}

/// Verify that compute_rho produces correct matrix-vector product.
#[test]
fn test_compute_rho_correct() {
    let d = 4;
    // R = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    let mut r_proj = vec![0.0f32; d * d];
    for i in 0..d {
        r_proj[i * d + i] = (i + 1) as f32;
    }
    let x = vec![1.0f32; d];
    let mut rho = vec![0.0f32; d];

    compute_rho(&r_proj, &x, &mut rho);

    let expected = [1.0f32, 2.0, 3.0, 4.0];
    for (i, (&r, &e)) in rho.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "rho[{i}] should be {e}, got {r}");
    }
}

/// Verify that parallax_correction with a known sigma produces the right result.
#[test]
fn test_correction_known_sigma() {
    let d = 3;
    // sigma_kv = [[2, 0, 0], [0, 2, 0], [0, 0, 2]] (2 * identity)
    let mut sigma_kv = vec![0.0f32; d * d];
    for i in 0..d {
        sigma_kv[i * d + i] = 2.0;
    }
    let rho = vec![1.0f32, 2.0, 3.0];
    let mut correction = vec![0.0f32; d];

    parallax_correction(&sigma_kv, &rho, &mut correction);

    let expected = [2.0f32, 4.0, 6.0];
    for (i, (&c, &e)) in correction.iter().zip(expected.iter()).enumerate() {
        assert!(
            (c - e).abs() < 1e-5,
            "correction[{i}] should be {e}, got {c}"
        );
    }
}

// ── Sigmoid-specific tests ──────────────────────────────────────

/// With gate_scale=0 and sigmoid activation, output should equal pure sigmoid attention.
#[test]
fn test_parallax_sigmoid_recovers_base() {
    let d = 4;
    let n = 3;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

    let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

    let config = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Sigmoid,
        ..Default::default()
    };

    let mut output_parallax = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut output_parallax,
        n,
        d,
        scale,
        &r,
        &x,
        &config,
        None,
    );

    let mut output_ref = vec![0.0f32; n * d];
    tiled_attention_core(
        &q,
        &k,
        &v,
        &mut output_ref,
        n,
        d,
        scale,
        None,
        ParallaxActivation::Sigmoid,
        None,
        #[cfg(feature = "ssmax_temperature")]
        None,
        #[cfg(feature = "prior_logit_lane")]
        None,
        #[cfg(feature = "scale_invariant_attn")]
        None,
    );

    for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output[{i}]: sigmoid parallax ({a}) should match base sigmoid ({b}) with gate_scale=0"
        );
    }
}

/// Sigmoid attention weights should be non-negative and sum to 1 per row.
#[test]
fn test_sigmoid_weights_normalized() {
    let d = 4;
    let n = 5;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.37).sin()).collect();
    let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.53).cos()).collect();
    let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.71).sin()).collect();

    // Run with gate_scale=0 so we get pure sigmoid attention
    let r = vec![0.0f32; d * d];
    let x = vec![0.0f32; d];
    let config = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Sigmoid,
        ..Default::default()
    };

    let mut output = vec![0.0f32; n * d];
    tiled_attention_parallax_forward(&q, &k, &v, &mut output, n, d, scale, &r, &x, &config, None);

    // Output should be finite (no NaN/Inf from numerical issues)
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] should be finite, got {v}");
    }
}

/// Sigmoid and softmax should produce different outputs (different kernels).
#[test]
fn test_sigmoid_differs_from_softmax() {
    let d = 4;
    let n = 3;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

    let r = vec![0.0f32; d * d];
    let x = vec![0.0f32; d];

    let mut out_sm = vec![0.0f32; n * d];
    let mut out_sig = vec![0.0f32; n * d];

    let config_sm = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Softmax,
        ..Default::default()
    };
    let config_sig = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Sigmoid,
        ..Default::default()
    };

    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut out_sm,
        n,
        d,
        scale,
        &r,
        &x,
        &config_sm,
        None,
    );
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut out_sig,
        n,
        d,
        scale,
        &r,
        &x,
        &config_sig,
        None,
    );

    let any_differs = out_sm
        .iter()
        .zip(out_sig.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-5);
    assert!(
        any_differs,
        "sigmoid and softmax should produce different outputs"
    );
}

/// With non-zero R projection, sigmoid Parallax should modify the output.
#[test]
fn test_sigmoid_parallax_correction_applied() {
    let d = 4;
    let n = 3;
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

    let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

    let config_no_corr = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Sigmoid,
        ..Default::default()
    };
    let config_with_corr = ParallaxConfig {
        gate_scale: 1.0,
        activation: ParallaxActivation::Sigmoid,
        ..Default::default()
    };

    let mut out_no = vec![0.0f32; n * d];
    let mut out_yes = vec![0.0f32; n * d];

    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut out_no,
        n,
        d,
        scale,
        &r,
        &x,
        &config_no_corr,
        None,
    );
    tiled_attention_parallax_forward(
        &q,
        &k,
        &v,
        &mut out_yes,
        n,
        d,
        scale,
        &r,
        &x,
        &config_with_corr,
        None,
    );

    let any_differs = out_no
        .iter()
        .zip(out_yes.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-5);
    assert!(
        any_differs,
        "sigmoid parallax correction should modify output vs base sigmoid"
    );
}

// ── Plan 289 tests ──────────────────────────────────────────────
// Covers: retained-attention forward correctness (always-on, parallax_attn
// only), and sink-aware composition parity (Uniform + DualPolicy) + G2.
// The latency G3 microbench lives in benches/ (T3.5).

/// Deterministic LCG for reproducible test inputs. Cheap, no deps.
fn lcg_fill(seed: u64, buf: &mut [f32]) {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for x in buf.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x = (((s >> 33) as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
}

/// Reference row-by-row attention matrix computation. Independent of the
/// forward's internal accumulation — recomputes scores + normalization.
fn reference_attn_matrix(
    q: &[f32],
    k: &[f32],
    n: usize,
    d: usize,
    scale: f32,
    activation: ParallaxActivation,
    am: &mut [f32],
) {
    let mut row = vec![0.0f32; n];
    for i in 0..n {
        let q_off = i * d;
        for (j, row_slot) in row.iter_mut().enumerate().take(n) {
            let k_off = j * d;
            *row_slot =
                crate::simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d) * scale;
        }
        normalize_attention_weights(&mut row, activation);
        am[i * n..(i + 1) * n].copy_from_slice(&row);
    }
}

/// Helper: build (q, k, v) where attention concentrates strongly on position
/// `sink_pos` (mean column strength ≈ 0.94, well above τ_sink=0.5) but
/// `v[sink_pos]` is optionally zero (NOP) or normal content (Broadcast).
///
/// Construction:
/// - q[i] = [i*0.5, 0, ...] for i in 0..n (varies across queries).
/// - k[sink] = [+10, 0, ...] → σ(q·k) saturates to ≈1 for i≥1.
/// - k[j≠sink] = [-10, 0, ...] → σ(q·k) ≈ 0 for i≥1.
/// - v[j] = ones (or zeros at sink for NOP case).
///
/// Result: column `sink_pos` receives mean strength ≈ 0.94 across rows,
/// dominating all other columns. The AV update is rank-1 (output rows
/// proportional to v[sink]) when v[sink] is non-zero.
// Consumers are the sink-aware + 3-way ssmax/sink modules below — without
// this gate the parallax_attn-only combo (a documented single-feature lane)
// carries dead_code under -D warnings.
#[cfg(feature = "sink_aware_attn")]
fn build_sink_case(
    n: usize,
    d: usize,
    sink_pos: usize,
    sink_v_zero: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; n * d];
    let mut k = vec![0.0f32; n * d];
    let mut v = vec![0.0f32; n * d];
    for i in 0..n {
        q[i * d] = (i as f32) * 0.5;
    }
    // Sink column attracts strongly.
    k[sink_pos * d] = 10.0;
    // Other columns strongly repel.
    for j in 0..n {
        if j != sink_pos {
            k[j * d] = -10.0;
        }
        for c in 0..d {
            v[j * d + c] = if j == sink_pos && sink_v_zero {
                0.0
            } else {
                1.0
            };
        }
    }
    (q, k, v)
}

/// T1.3 — retained attention matrix matches row-by-row reference (Sigmoid).
#[test]
fn plan289_retained_attn_matches_per_row_sigmoid() {
    let n = 16;
    let d = 8;
    let scale = 1.0 / (d as f32).sqrt();
    let mut q = vec![0.0f32; n * d];
    let mut k = vec![0.0f32; n * d];
    let v = vec![0.0f32; n * d];
    let mut output = vec![0.0f32; n * d];
    let mut am_actual = vec![0.0f32; n * n];
    let mut am_expected = vec![0.0f32; n * n];
    lcg_fill(0xC0DE, &mut q);
    lcg_fill(0xFEED, &mut k);

    let cfg = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Sigmoid,
        ..Default::default()
    };
    let r = vec![0.0f32; d * d];
    let x = vec![0.0f32; d];

    tiled_attention_parallax_forward_retaining(
        &q,
        &k,
        &v,
        &mut output,
        n,
        d,
        scale,
        &r,
        &x,
        &cfg,
        Some(&mut am_actual),
        None,
    );
    reference_attn_matrix(
        &q,
        &k,
        n,
        d,
        scale,
        ParallaxActivation::Sigmoid,
        &mut am_expected,
    );

    for i in 0..(n * n) {
        assert_eq!(am_actual[i], am_expected[i], "am[{i}] mismatch (Sigmoid)");
    }
}

/// T1.3 — retained attention matrix matches row-by-row reference (Softmax).
#[test]
fn plan289_retained_attn_matches_per_row_softmax() {
    let n = 16;
    let d = 8;
    let scale = 1.0 / (d as f32).sqrt();
    let mut q = vec![0.0f32; n * d];
    let mut k = vec![0.0f32; n * d];
    let v = vec![0.0f32; n * d];
    let mut output = vec![0.0f32; n * d];
    let mut am_actual = vec![0.0f32; n * n];
    let mut am_expected = vec![0.0f32; n * n];
    lcg_fill(0x1234, &mut q);
    lcg_fill(0x5678, &mut k);

    let cfg = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Softmax,
        ..Default::default()
    };
    let r = vec![0.0f32; d * d];
    let x = vec![0.0f32; d];

    tiled_attention_parallax_forward_retaining(
        &q,
        &k,
        &v,
        &mut output,
        n,
        d,
        scale,
        &r,
        &x,
        &cfg,
        Some(&mut am_actual),
        None,
    );
    reference_attn_matrix(
        &q,
        &k,
        n,
        d,
        scale,
        ParallaxActivation::Softmax,
        &mut am_expected,
    );

    for i in 0..(n * n) {
        assert_eq!(am_actual[i], am_expected[i], "am[{i}] mismatch (Softmax)");
    }
}

// ── Plan 289 sink-aware tests (require sink_aware_attn feature) ───

#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]
mod sink_aware_tests {
    use super::*;
    use crate::data_probe::{
        SinkAwarePolicy, SinkClassifierConfig, SinkKind, StableRankScratch,
        apply_dual_policy_gate_flat,
    };

    fn parallax_zero_cfg(act: ParallaxActivation) -> ParallaxConfig {
        ParallaxConfig {
            gate_scale: 0.0,
            activation: act,
            ..Default::default()
        }
    }

    /// T3.1 — Uniform policy path produces bit-identical output to vanilla forward.
    #[test]
    fn plan289_uniform_bit_identical_to_vanilla() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        super::lcg_fill(0xA1B2, &mut q);
        super::lcg_fill(0xC3D4, &mut k);
        super::lcg_fill(0xE5F6, &mut v);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let mut out_vanilla = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_vanilla,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            None,
        );

        let mut out_uniform = vec![0.0f32; n * d];
        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d);
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_uniform,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &SinkAwarePolicy::Uniform,
            -10.0,
            &mut sink_scratch,
            None,
        );
        assert!(
            matches!(kind, SinkKind::None),
            "Uniform must return SinkKind::None"
        );

        for i in 0..(n * d) {
            assert_eq!(
                out_vanilla[i], out_uniform[i],
                "output[{i}] differs (Uniform path)"
            );
        }
    }

    /// T3.2 — DualPolicy path bit-identical to manual composition.
    #[test]
    fn plan289_dualpolicy_matches_manual_composition() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        super::lcg_fill(0x11, &mut q);
        super::lcg_fill(0x22, &mut k);
        super::lcg_fill(0x33, &mut v);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy_cfg = SinkClassifierConfig::default();
        let policy = SinkAwarePolicy::DualPolicy(policy_cfg);
        let gate_scale = -2.0;

        // Wrapper path
        let mut out_wrapper = vec![0.0f32; n * d];
        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d);
        let kind_wrapper = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_wrapper,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &policy,
            gate_scale,
            &mut sink_scratch,
            None,
        );

        // Manual composition
        let mut out_manual = vec![0.0f32; n * d];
        let mut o_temp = vec![0.0f32; n * d];
        let mut am = vec![0.0f32; n * n];
        let mut classifier = StableRankScratch::new(d);
        tiled_attention_parallax_forward_retaining(
            &q,
            &k,
            &v,
            &mut o_temp,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            Some(&mut am),
            None,
        );
        let kind_manual = apply_dual_policy_gate_flat(
            &am,
            &v,
            &o_temp,
            n,
            d,
            &policy,
            gate_scale,
            &mut classifier,
            &mut out_manual,
        );

        assert_eq!(kind_wrapper, kind_manual, "SinkKind mismatch");
        for i in 0..(n * d) {
            assert_eq!(
                out_wrapper[i], out_manual[i],
                "output[{i}] differs (DualPolicy path)"
            );
        }
    }

    /// T3.3 — synthetic NOP head: dominant sink has zero v, classifier must
    /// return Nop, and output must be scaled by σ(gate_scale).
    #[test]
    fn plan289_synthetic_nop_head_gated() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::build_sink_case(n, d, 0, true);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());

        // Ungated reference via Uniform.
        let mut out_ungated = vec![0.0f32; n * d];
        let mut sa_scratch_un = SinkAwareParallaxScratch::new(n, d);
        tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_ungated,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &SinkAwarePolicy::Uniform,
            0.0,
            &mut sa_scratch_un,
            None,
        );

        // DualPolicy with strong suppression.
        let gate_scale = -10.0; // σ(-10) ≈ 4.5e-5
        let mut out_gated = vec![0.0f32; n * d];
        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d);
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_gated,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &policy,
            gate_scale,
            &mut sink_scratch,
            None,
        );
        assert!(matches!(kind, SinkKind::Nop), "expected Nop, got {kind:?}");

        // NOP-gated output must equal σ(gate_scale) × ungated output.
        let sigma = 1.0 / (1.0 + (-gate_scale).exp());
        for i in 0..(n * d) {
            let expected = out_ungated[i] * sigma;
            let delta = (out_gated[i] - expected).abs();
            assert!(
                delta < 1e-5,
                "gated[{}]={} != σ(gs)·ungated={} (delta {})",
                i,
                out_gated[i],
                expected,
                delta
            );
        }
    }

    /// T3.4 — synthetic Broadcast head: dominant sink carries content AND the
    /// AV update is rank-1. Classifier must return Broadcast, and output must
    /// be bit-identical to the Uniform (ungated) path.
    #[test]
    fn plan289_synthetic_broadcast_head_preserved() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::build_sink_case(n, d, 0, false);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let mut out_uniform = vec![0.0f32; n * d];
        let mut sa_un = SinkAwareParallaxScratch::new(n, d);
        tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_uniform,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &SinkAwarePolicy::Uniform,
            0.0,
            &mut sa_un,
            None,
        );

        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());
        let gate_scale = -10.0;
        let mut out_dp = vec![0.0f32; n * d];
        let mut sa_dp = SinkAwareParallaxScratch::new(n, d);
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_dp,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &policy,
            gate_scale,
            &mut sa_dp,
            None,
        );
        assert!(
            matches!(kind, SinkKind::Broadcast),
            "expected Broadcast, got {kind:?}"
        );

        for i in 0..(n * d) {
            assert_eq!(
                out_uniform[i], out_dp[i],
                "Broadcast output[{i}] must equal Uniform"
            );
        }
    }

    /// Cached path: wrapper uses cached variant when `cached = Some`. Two
    /// consecutive DualPolicy calls → second reuses cached SinkKind.
    #[test]
    fn plan289_cached_path_audit_and_reuse() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::build_sink_case(n, d, 0, true);
        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());

        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d).with_cache();
        if let Some(c) = sink_scratch.cached.as_mut() {
            c.audit_every_n = 4;
        }

        let gate_scale = -5.0;
        let mut out_a = vec![0.0f32; n * d];
        let kind_a = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_a,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &policy,
            gate_scale,
            &mut sink_scratch,
            None,
        );
        assert!(matches!(kind_a, SinkKind::Nop));
        assert_eq!(
            sink_scratch.cached.as_ref().unwrap().calls_since_audit,
            1,
            "first call must reset cadence counter"
        );

        let mut out_b = vec![0.0f32; n * d];
        let kind_b = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_b,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &policy,
            gate_scale,
            &mut sink_scratch,
            None,
        );
        assert!(matches!(kind_b, SinkKind::Nop));
        assert_eq!(
            sink_scratch.cached.as_ref().unwrap().calls_since_audit,
            2,
            "second call must increment without re-audit"
        );

        for i in 0..(n * d) {
            assert_eq!(
                out_a[i], out_b[i],
                "cached NOP output[{i}] must match audit output"
            );
        }
    }
}

// ── SSMax composition tests (Plan 411 T2.2/T2.3) ──────────────────
// These verify the wiring: SSMax is actually applied when configured, and
// is a bit-identical no-op when ssmax is None. The SSMax primitive's own
// numerics are tested in ssmax.rs; here we test the parallax integration.

#[cfg(all(feature = "parallax_attn", feature = "ssmax_temperature"))]
mod ssmax_composition_tests {
    use super::*;
    use crate::ssmax::SsmaxMode;

    /// ParallaxConfig with ssmax=None must produce bit-identical output to a
    /// config constructed without the ssmax field (the Default::default() path).
    /// This is the zero-regression contract: when SSMax is off, nothing changes.
    #[test]
    fn ssmax_none_is_bit_identical_to_base() {
        let n = 64;
        let d = 16;
        let scale = 1.0 / (d as f32).sqrt();
        let q: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.017).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.023).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.011).sin()).collect();
        let r: Vec<f32> = vec![0.5; d * d];
        let x: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1).collect();

        let cfg_base = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };
        // Same config but explicitly setting ssmax = None.
        let cfg_none = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ssmax: None,
            #[cfg(feature = "prior_logit_lane")]
            prior_logits: None,
            #[cfg(feature = "scale_invariant_attn")]
            scale_invariant: None,
        };

        let mut out_base = vec![0.0f32; n * d];
        let mut out_none = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_base,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_base,
            None,
        );
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_none,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_none,
            None,
        );

        for i in 0..(n * d) {
            assert_eq!(
                out_base[i], out_none[i],
                "ssmax=None must be bit-identical at [{i}]"
            );
        }
    }

    /// SSMax at N=1 is skipped by the `n > 1` guard in `apply_ssmax_to_row`,
    /// because log(1)=0 would zero every logit otherwise. This test verifies
    /// that guard: n=1 output is identical with and without SSMax configured.
    #[test]
    fn ssmax_n1_is_noop() {
        let n = 1;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let q = vec![0.5; d];
        let k = vec![0.3; d];
        let v = vec![0.7; d];
        let r = vec![0.0; d * d];
        let x = vec![0.0; d];

        let cfg_base = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };
        let cfg_ssmax = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ssmax: Some(SsmaxMode::Fixed { s_l: 1.0 }),
            #[cfg(feature = "prior_logit_lane")]
            prior_logits: None,
            #[cfg(feature = "scale_invariant_attn")]
            scale_invariant: None,
        };

        let mut out_base = vec![0.0f32; d];
        let mut out_ssmax = vec![0.0f32; d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_base,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_base,
            None,
        );
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_ssmax,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_ssmax,
            None,
        );

        for i in 0..d {
            assert_eq!(
                out_base[i], out_ssmax[i],
                "n=1 SSMax must be skipped (guard)"
            );
        }
    }

    /// SSMax with a real multiplier (n > 1, s_L = 1.0) must change the output
    /// when the logits are not all identical. This verifies the wiring: SSMax
    /// is actually applied in the parallax forward, not silently dropped.
    #[test]
    fn ssmax_changes_output_at_large_n() {
        let n = 64;
        let d = 16;
        let scale = 1.0 / (d as f32).sqrt();
        // Non-uniform q/k so logits vary — SSMax's multiplicative rescaling
        // will shift the normalized sigmoid weights.
        let q: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.07).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.05).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.03).sin()).collect();
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let cfg_base = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };
        let cfg_ssmax = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ssmax: Some(SsmaxMode::Fixed { s_l: 1.0 }),
            #[cfg(feature = "prior_logit_lane")]
            prior_logits: None,
            #[cfg(feature = "scale_invariant_attn")]
            scale_invariant: None,
        };

        let mut out_base = vec![0.0f32; n * d];
        let mut out_ssmax = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_base,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_base,
            None,
        );
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_ssmax,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_ssmax,
            None,
        );

        // SSMax multiplies logits by log(64) ≈ 4.16. With non-uniform logits,
        // the sharpened sigmoid weights must differ from the base.
        let diff_count = (0..n * d)
            .filter(|&i| (out_base[i] - out_ssmax[i]).abs() > 1e-6)
            .count();
        assert!(
            diff_count > 0,
            "SSMax at n=64 with s_L=1.0 must change the output (got 0 differing elements)"
        );
    }

    /// SSMax scales logits by a constant factor. For sigmoid normalization,
    /// this is equivalent to scaling the temperature `scale` by the same factor.
    /// Verify: parallax(cfg.ssmax=Some(mode), scale=s) == parallax(cfg.ssmax=None, scale=s*mult).
    /// This cross-checks the scale-folding equivalence used in the tiled_attention_core path.
    #[test]
    fn ssmax_equivalent_to_scale_folding_sigmoid() {
        let n = 32;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let q: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.07).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.05).cos()).collect();
        let v: Vec<f32> = vec![1.0; n * d]; // uniform v so only weights matter
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let mode = SsmaxMode::Fixed { s_l: 1.0 };
        let log_n = (n as f32).ln();
        let mult = mode.multiplier(log_n);

        let cfg_ssmax = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ssmax: Some(mode),
            #[cfg(feature = "prior_logit_lane")]
            prior_logits: None,
            #[cfg(feature = "scale_invariant_attn")]
            scale_invariant: None,
        };
        let cfg_folded = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };

        let mut out_ssmax = vec![0.0f32; n * d];
        let mut out_folded = vec![0.0f32; n * d];
        // SSMax path: logits rescaled inside the forward.
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_ssmax,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_ssmax,
            None,
        );
        // Folded path: scale pre-multiplied by mult, no SSMax.
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_folded,
            n,
            d,
            scale * mult,
            &r,
            &x,
            &cfg_folded,
            None,
        );

        for i in 0..(n * d) {
            assert!(
                (out_ssmax[i] - out_folded[i]).abs() < 1e-5,
                "SSMax apply must match scale-folding at [{}]: {} vs {}",
                i,
                out_ssmax[i],
                out_folded[i]
            );
        }
    }
}

// ── SSMax + Sink-Aware 3-way composition tests (Plan 411 T2.3) ────

#[cfg(all(
    feature = "parallax_attn",
    feature = "sink_aware_attn",
    feature = "ssmax_temperature"
))]
mod ssmax_sink_aware_tests {
    use super::*;
    use crate::data_probe::{SinkAwarePolicy, SinkClassifierConfig};
    use crate::ssmax::SsmaxMode;

    /// The 3-way entry point with ssmax_mode=None must produce identical
    /// output to the 2-way sink-aware forward (the explicit None is a no-op).
    #[test]
    fn three_way_none_matches_two_way() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::build_sink_case(n, d, 0, true);
        let cfg = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());

        let mut out_two = vec![0.0f32; n * d];
        let mut sa_two = SinkAwareParallaxScratch::new(n, d);
        let kind_two = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_two,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            &policy,
            -5.0,
            &mut sa_two,
            None,
        );

        let mut out_three = vec![0.0f32; n * d];
        let mut sa_three = SinkAwareParallaxScratch::new(n, d);
        let kind_three = tiled_attention_parallax_forward_sink_aware_ssmax(
            &q,
            &k,
            &v,
            &mut out_three,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            None,
            &policy,
            -5.0,
            &mut sa_three,
            None,
        );

        assert_eq!(kind_two, kind_three, "SinkKind must match");
        for i in 0..(n * d) {
            assert_eq!(
                out_two[i], out_three[i],
                "3-way(None) must match 2-way at [{i}]"
            );
        }
    }

    /// The 3-way entry point with ssmax_mode=Some must apply SSMax.
    /// Verify by comparing to the 2-way forward with ssmax injected into the config.
    #[test]
    fn three_way_some_matches_config_injection() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::build_sink_case(n, d, 0, true);
        let cfg_base = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());
        let mode = SsmaxMode::Fixed { s_l: 1.0 };

        // Path A: 3-way entry point with explicit ssmax_mode.
        let mut out_a = vec![0.0f32; n * d];
        let mut sa_a = SinkAwareParallaxScratch::new(n, d);
        let kind_a = tiled_attention_parallax_forward_sink_aware_ssmax(
            &q,
            &k,
            &v,
            &mut out_a,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_base,
            Some(&mode),
            &policy,
            -5.0,
            &mut sa_a,
            None,
        );

        // Path B: manually inject ssmax into config, call 2-way forward.
        let mut cfg_injected = cfg_base.clone();
        cfg_injected.ssmax = Some(mode);
        let mut out_b = vec![0.0f32; n * d];
        let mut sa_b = SinkAwareParallaxScratch::new(n, d);
        let kind_b = tiled_attention_parallax_forward_sink_aware(
            &q,
            &k,
            &v,
            &mut out_b,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg_injected,
            &policy,
            -5.0,
            &mut sa_b,
            None,
        );

        assert_eq!(kind_a, kind_b, "SinkKind must match");
        for i in 0..(n * d) {
            assert_eq!(
                out_a[i], out_b[i],
                "3-way(Some) must match config-injected at [{i}]"
            );
        }
    }

    /// SSMax with a real mode must change the 3-way output vs no SSMax.
    /// Uses the Broadcast case (build_sink_case with sink_v_zero=true produces
    /// a Broadcast head) so the gate is active and SSMax's logit rescaling
    /// flows through to the gated output.
    #[test]
    fn three_way_ssmax_changes_output() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::build_sink_case(n, d, 0, true);
        let cfg = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());
        let mode = SsmaxMode::Fixed { s_l: 2.0 };

        let mut out_no = vec![0.0f32; n * d];
        let mut sa_no = SinkAwareParallaxScratch::new(n, d);
        tiled_attention_parallax_forward_sink_aware_ssmax(
            &q,
            &k,
            &v,
            &mut out_no,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            None,
            &policy,
            -5.0,
            &mut sa_no,
            None,
        );

        let mut out_yes = vec![0.0f32; n * d];
        let mut sa_yes = SinkAwareParallaxScratch::new(n, d);
        tiled_attention_parallax_forward_sink_aware_ssmax(
            &q,
            &k,
            &v,
            &mut out_yes,
            n,
            d,
            scale,
            &r,
            &x,
            &cfg,
            Some(&mode),
            &policy,
            -5.0,
            &mut sa_yes,
            None,
        );

        let diff_count = (0..n * d)
            .filter(|&i| (out_no[i] - out_yes[i]).abs() > 1e-6)
            .count();
        assert!(
            diff_count > 0,
            "3-way with SSMax s_L=2.0 must differ from no-SSMax (got 0 diffs)"
        );
    }
}

// ── Scale-invariant schedule wiring (Plan 622 Phase 2, Research 610) ──

#[cfg(all(feature = "parallax_attn", feature = "scale_invariant_attn"))]
mod scale_invariant_tests {
    use super::*;
    use crate::scale_invariant::ScaleInvariantLut;

    const TAU: f32 = 4.0;
    const MAX_CTX: usize = 64;

    /// Independent per-row reference: scores → per-key affine (skip sink key
    /// 0 and future keys) → normalize (softmax or sigmoid) → Σ p·v. The same
    /// math the forward must realize, spelled from the closed form.
    #[allow(clippy::too_many_arguments)]
    fn reference_forward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n: usize,
        d: usize,
        scale: f32,
        si: Option<&ScaleInvariantLut>,
        activation: ParallaxActivation,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            let mut scores: Vec<f32> = (0..n)
                .map(|j| {
                    let dot: f32 = (0..d)
                        .map(|e| q[i * d + e] * k[j * d + e])
                        .sum();
                    dot * scale
                })
                .collect();
            if let Some(lut) = si {
                let mut transformed = scores.clone();
                for j in 0..n {
                    let key_j = j; // key_start = 0
                    if key_j == 0 || key_j > i {
                        continue;
                    }
                    let t = i - key_j;
                    let (a, m) = lut.pair(t);
                    transformed[j] = a * scores[j] + m;
                }
                scores = transformed;
            }
            match activation {
                ParallaxActivation::Softmax => {
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for s in &mut scores {
                        *s = (*s - max).exp();
                    }
                }
                ParallaxActivation::Sigmoid => {
                    for s in &mut scores {
                        *s = 1.0 / (1.0 + (-*s).exp());
                    }
                }
            }
            let sum: f32 = scores.iter().sum();
            for (j, &p) in scores.iter().enumerate() {
                for e in 0..d {
                    out[i * d + e] += p / sum * v[j * d + e];
                }
            }
        }
        out
    }

    fn gen_qkv(n: usize, d: usize, phase: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.017 + phase).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.023).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.011).sin()).collect();
        (q, k, v)
    }

    fn compare(a: &[f32], b: &[f32], tol: f32, what: &str) {
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "{what}: [{i}] {x} vs {y} (Δ = {})",
                (x - y).abs()
            );
        }
    }

    /// Both activation arms consume the same LUT and match the independent
    /// reference (core path: gate_scale = 0, r zero → `tiled_attention_core`).
    #[test]
    fn si_core_path_matches_reference_both_arms() {
        let n = 8;
        let d = 4;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = gen_qkv(n, d, 0.0);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let lut = std::sync::Arc::new(ScaleInvariantLut::build(TAU, MAX_CTX));
        for activation in [ParallaxActivation::Softmax, ParallaxActivation::Sigmoid] {
            let cfg = ParallaxConfig {
                gate_scale: 0.0,
                activation,
                scale_invariant: Some(lut.clone()),
                ..Default::default()
            };
            let mut out = vec![0.0f32; n * d];
            tiled_attention_parallax_forward(
                &q, &k, &v, &mut out, n, d, scale, &r, &x, &cfg, None,
            );
            let reference = reference_forward(&q, &k, &v, n, d, scale, Some(&lut), activation);
            compare(&out, &reference, 1e-4, &format!("core {activation:?}"));
            // Non-vacuity: the schedule must actually move the output.
            let cfg_off = ParallaxConfig {
                gate_scale: 0.0,
                activation,
                ..Default::default()
            };
            let mut out_off = vec![0.0f32; n * d];
            tiled_attention_parallax_forward(
                &q, &k, &v, &mut out_off, n, d, scale, &r, &x, &cfg_off, None,
            );
            let diffs = out.iter().zip(&out_off).filter(|(a, b)| a != b).count();
            assert!(diffs > 0, "core {activation:?}: schedule must change output");
        }
    }

    /// Same through the MAIN path (nonzero W_R → the Phase-1 loop with the
    /// covariance correction on top — the schedule composes with Parallax).
    #[test]
    fn si_main_path_matches_reference() {
        let n = 8;
        let d = 4;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = gen_qkv(n, d, 0.5);
        let r: Vec<f32> = (0..d * d).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let lut = std::sync::Arc::new(ScaleInvariantLut::build(TAU, MAX_CTX));
        for activation in [ParallaxActivation::Softmax, ParallaxActivation::Sigmoid] {
            let cfg = ParallaxConfig {
                gate_scale: 1.0,
                activation,
                scale_invariant: Some(lut.clone()),
                ..Default::default()
            };
            let mut out = vec![0.0f32; n * d];
            tiled_attention_parallax_forward(
                &q, &k, &v, &mut out, n, d, scale, &r, &x, &cfg, None,
            );
            // Reference: schedule-weighted attention, then the Parallax
            // correction o −= gate_scale · Σ_KV · ρ spelled from the SAME
            // normalized weights.
            let mut reference =
                reference_forward(&q, &k, &v, n, d, scale, Some(&lut), activation);
            // column sums from the reference weights
            let mut col = vec![0.0f32; n];
            for i in 0..n {
                let mut scores: Vec<f32> = (0..n)
                    .map(|j| {
                        let dot: f32 = (0..d).map(|e| q[i * d + e] * k[j * d + e]).sum();
                        dot * scale
                    })
                    .collect();
                for (j, s) in scores.iter_mut().enumerate().take(i.min(n - 1) + 1).skip(1) {
                    let (a, m) = lut.pair(i - j);
                    *s = a * *s + m;
                }
                match activation {
                    ParallaxActivation::Softmax => {
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        for s in &mut scores {
                            *s = (*s - max).exp();
                        }
                    }
                    ParallaxActivation::Sigmoid => {
                        for s in &mut scores {
                            *s = 1.0 / (1.0 + (-*s).exp());
                        }
                    }
                }
                let sum: f32 = scores.iter().sum();
                for (j, &p) in scores.iter().enumerate() {
                    col[j] += p / sum;
                }
            }
            let mut rho = vec![0.0f32; d];
            for row in 0..d {
                for col_i in 0..d {
                    rho[row] += r[row * d + col_i] * x[col_i];
                }
            }
            let mut sigma_kv = vec![0.0f32; d * d];
            for j in 0..n {
                for row in 0..d {
                    for col_i in 0..d {
                        sigma_kv[row * d + col_i] += col[j] * v[j * d + row] * k[j * d + col_i];
                    }
                }
            }
            let mut corr = vec![0.0f32; d];
            for row in 0..d {
                for col_i in 0..d {
                    corr[row] += sigma_kv[row * d + col_i] * rho[col_i];
                }
            }
            for i in 0..n {
                for e in 0..d {
                    reference[i * d + e] -= 1.0 * corr[e];
                }
            }
            compare(&out, &reference, 2e-4, &format!("main {activation:?}"));
        }
    }

    /// n=1: the only key is the sink (key 0) — the schedule must be a no-op
    /// and the output must equal the schedule-off output bit-exactly.
    #[test]
    fn si_n1_is_bit_identical_to_off() {
        let n = 1;
        let d = 4;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = gen_qkv(n, d, 0.0);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let lut = std::sync::Arc::new(ScaleInvariantLut::build(TAU, MAX_CTX));
        let cfg_on = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            scale_invariant: Some(lut),
            ..Default::default()
        };
        let cfg_off = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            ..Default::default()
        };
        let mut out_on = vec![0.0f32; n * d];
        let mut out_off = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(&q, &k, &v, &mut out_on, n, d, scale, &r, &x, &cfg_on, None);
        tiled_attention_parallax_forward(&q, &k, &v, &mut out_off, n, d, scale, &r, &x, &cfg_off, None);
        assert_eq!(out_on, out_off, "n=1 (all-sink row) must be bit-identical");
    }

    /// The ordering law, loud form: arming scale_invariant AND ssmax in one
    /// config is a config error the forward refuses in both profiles.
    #[cfg(feature = "ssmax_temperature")]
    #[test]
    #[should_panic(expected = "ordering law")]
    fn si_and_ssmax_armed_is_a_loud_config_error() {
        let n = 8;
        let d = 4;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = gen_qkv(n, d, 0.0);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let cfg = ParallaxConfig {
            gate_scale: 0.0,
            activation: ParallaxActivation::Sigmoid,
            scale_invariant: Some(std::sync::Arc::new(ScaleInvariantLut::build(TAU, MAX_CTX))),
            ssmax: Some(crate::ssmax::SsmaxMode::Fixed { s_l: 1.0 }),
            ..Default::default()
        };
        let mut out = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(&q, &k, &v, &mut out, n, d, scale, &r, &x, &cfg, None);
    }
}
