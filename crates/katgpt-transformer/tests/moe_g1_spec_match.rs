//! G1 spec-match tests for the MoE forward (Proposal 032 Phase 3).
//!
//! Strategy (mirrors Research 327 MLA G1 + Research 328 §7.6):
//!
//! 1. Write an INDEPENDENT f64 reference implementation of the MoE forward
//!    matching the actual `modeling_kimi_k3_linear.py` (Research 330 §3):
//!    latent MoE wrapper + SiTU expert activation. This reference does NOT
//!    reuse the f32 code — it's written from scratch from the model source.
//! 2. Run both impls on the same random inputs.
//! 3. Assert the f32 impl matches the f64 reference within tolerance.
//!
//! The CRITICAL test is `g1_bias_does_not_leak_into_renormalization` — it
//! implements a "buggy" reference variant that uses `biased[topk_idx]` in the
//! renormalization, runs it, asserts its output DIFFERS from the correct
//! reference, then asserts the f32 impl matches the CORRECT reference. This
//! catches the §2.2 misreading bit-identically.

#![cfg(feature = "transformer_moe")]

use katgpt_transformer::moe::{MoeConfig, MoeForwardScratch, MoeWeights, moe_forward_token};

// ─── Independent f64 reference implementation ───────────────────────────────

/// f64 reference SiTU activation (matches `SituAndMul` from the actual model):
/// ```text
/// situ_a = beta * tanh(gate / beta) * sigmoid(gate)
/// up_t   = linear_beta * tanh(up / linear_beta)   (when linear_beta is Some)
/// output = situ_a * up_t
/// ```
#[inline]
fn ref_situ_f64(gate: f64, up: f64, beta: f64, linear_beta: Option<f64>) -> f64 {
    let situ_a = beta * (gate / beta).tanh() * (1.0 / (1.0 + (-gate).exp()));
    let up_t = match linear_beta {
        Some(lb) => lb * (up / lb).tanh(),
        None => up,
    };
    situ_a * up_t
}

/// f64 reference of a single SiTU expert FFN forward.
#[allow(clippy::too_many_arguments)]
fn ref_situ_expert_f64(
    gate_proj: &[f64],
    up_proj: &[f64],
    down_proj: &[f64],
    hidden_in: &[f64],
    d_in: usize,
    d_ffn: usize,
    beta: f64,
    linear_beta: Option<f64>,
    out: &mut [f64],
) {
    let mut intermediate = vec![0.0f64; d_ffn];
    let mut up = vec![0.0f64; d_ffn];
    // gate · h
    for o in 0..d_ffn {
        let mut acc = 0.0;
        for i in 0..d_in {
            acc += gate_proj[o * d_in + i] * hidden_in[i];
        }
        intermediate[o] = acc;
    }
    // up · h
    for o in 0..d_ffn {
        let mut acc = 0.0;
        for i in 0..d_in {
            acc += up_proj[o * d_in + i] * hidden_in[i];
        }
        up[o] = acc;
    }
    // SiTU activation
    for o in 0..d_ffn {
        intermediate[o] = ref_situ_f64(intermediate[o], up[o], beta, linear_beta);
    }
    // down · intermediate
    for o in 0..d_in {
        let mut acc = 0.0;
        for i in 0..d_ffn {
            acc += down_proj[o * d_ffn + i] * intermediate[i];
        }
        out[o] = acc;
    }
}

/// f64 RMSNorm with gamma.
fn ref_rmsnorm_f64(x: &mut [f64], gamma: &[f64], eps: f64) {
    let n = x.len();
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f64 + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

/// f64 reference sigmoid (plain exp, no approximations).
#[inline]
fn ref_sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// f64 reference of the FULL MoE forward (equations 12–16 from Research 328).
///
/// `buggy_bias_in_renorm`: when true, uses `biased[topk_idx]` in the
/// renormalization (the §2.2 misreading). When false, uses RAW sigmoid scores
/// (the correct behavior). Used by `g1_bias_does_not_leak_into_renormalization`.
fn ref_moe_forward_f64(
    weights: &MoeWeights,
    config: &MoeConfig,
    hidden_in: &[f32],
    out: &mut [f64],
    buggy_bias_in_renorm: bool,
) {
    let n_r = config.n_routed();
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();
    let d_moe = config.d_moe();
    let d_ffn_shared = config.d_ffn_shared();
    let use_latent_moe = config.routed_expert_hidden_size.is_some();
    let beta = config.situ_beta as f64;
    let linear_beta = config.situ_linear_beta.map(|v| v as f64);

    let hidden_f64: Vec<f64> = hidden_in.iter().map(|&v| v as f64).collect();

    // 1. Router logits + sigmoid scores
    let mut logits = vec![0.0f64; n_r];
    let mut scores = vec![0.0f64; n_r];
    for e in 0..n_r {
        let row = &weights.router_weight[e * d..e * d + d];
        let mut acc = 0.0;
        for (rw, h) in row.iter().zip(hidden_f64.iter()).take(d) {
            acc += *rw as f64 * h;
        }
        logits[e] = acc;
        scores[e] = ref_sigmoid_f64(logits[e]);
    }

    // 2. noaux_tc biased scores + top-K selection
    let mut biased: Vec<(usize, f64)> = (0..n_r)
        .map(|e| (e, scores[e] + weights.e_score_correction_bias[e] as f64))
        .collect();
    // Sort descending by biased score, take top-K.
    biased.sort_by(|a, b| b.1.total_cmp(&a.1));
    let topk_idx: Vec<usize> = biased.iter().take(k_r).map(|(i, _)| *i).collect();

    // 3. Renormalize
    let topk_s: Vec<f64> = if buggy_bias_in_renorm {
        // BUGGY: use biased score in renormalization
        topk_idx
            .iter()
            .map(|&i| scores[i] + weights.e_score_correction_bias[i] as f64)
            .collect()
    } else {
        // CORRECT: use raw sigmoid score
        topk_idx.iter().map(|&i| scores[i]).collect()
    };
    let sum: f64 = topk_s.iter().sum();
    let g: Vec<f64> = if config.renormalize {
        topk_s.iter().map(|&s| s / sum).collect()
    } else {
        topk_s
    };

    // 4. Shared expert (always on, full hidden dim) → base of output
    let shared = &weights.shared_experts[0];
    ref_situ_expert_f64(
        &cast_f32_slice(&shared.gate_proj),
        &cast_f32_slice(&shared.up_proj),
        &cast_f32_slice(&shared.down_proj),
        &hidden_f64,
        d,
        d_ffn_shared,
        beta,
        linear_beta,
        out,
    );
    // If N_s > 1, accumulate remaining shared experts.
    let mut expert_out = vec![0.0f64; d];
    for s in 1..weights.shared_experts.len() {
        let shared = &weights.shared_experts[s];
        ref_situ_expert_f64(
            &cast_f32_slice(&shared.gate_proj),
            &cast_f32_slice(&shared.up_proj),
            &cast_f32_slice(&shared.down_proj),
            &hidden_f64,
            d,
            d_ffn_shared,
            beta,
            linear_beta,
            &mut expert_out,
        );
        for i in 0..d {
            out[i] += expert_out[i];
        }
    }

    // 5. Routed experts
    if use_latent_moe {
        // Latent MoE path: down-project → experts on d_moe → norm → up-project
        let down_proj = weights.routed_expert_down_proj.as_ref().unwrap();
        let up_proj = weights.routed_expert_up_proj.as_ref().unwrap();

        // h_latent = down_proj · h
        let mut h_latent = vec![0.0f64; d_moe];
        for o in 0..d_moe {
            let mut acc = 0.0;
            for i in 0..d {
                acc += down_proj[o * d + i] as f64 * hidden_f64[i];
            }
            h_latent[o] = acc;
        }

        // Accumulate weighted expert outputs
        let mut latent_out = vec![0.0f64; d_moe];
        let mut single_expert_out = vec![0.0f64; d_moe];
        for k in 0..k_r {
            let idx = topk_idx[k];
            let w = g[k];
            let expert = &weights.experts[idx];
            ref_situ_expert_f64(
                &cast_f32_slice(&expert.gate_proj),
                &cast_f32_slice(&expert.up_proj),
                &cast_f32_slice(&expert.down_proj),
                &h_latent,
                d_moe,
                d_ffn,
                beta,
                linear_beta,
                &mut single_expert_out,
            );
            for i in 0..d_moe {
                latent_out[i] += w * single_expert_out[i];
            }
        }

        // Optional norm
        if config.latent_moe_use_norm
            && let Some(ref norm_w) = weights.routed_expert_norm_weight
        {
            let norm_f64: Vec<f64> = norm_w.iter().map(|&v| v as f64).collect();
            ref_rmsnorm_f64(&mut latent_out, &norm_f64, config.rms_norm_eps as f64);
        }

        // Up-project + add to output
        for o in 0..d {
            let mut acc = 0.0;
            for i in 0..d_moe {
                acc += up_proj[o * d_moe + i] as f64 * latent_out[i];
            }
            out[o] += acc;
        }
    } else {
        // Non-latent path: routed experts operate directly on hidden dim
        for k in 0..k_r {
            let idx = topk_idx[k];
            let w = g[k];
            let expert = &weights.experts[idx];
            ref_situ_expert_f64(
                &cast_f32_slice(&expert.gate_proj),
                &cast_f32_slice(&expert.up_proj),
                &cast_f32_slice(&expert.down_proj),
                &hidden_f64,
                d,
                d_ffn,
                beta,
                linear_beta,
                &mut expert_out,
            );
            for i in 0..d {
                out[i] += w * expert_out[i];
            }
        }
    }
}

/// Cast `&[f32]` to `Vec<f64>` for the reference impl.
fn cast_f32_slice(s: &[f32]) -> Vec<f64> {
    s.iter().map(|&v| v as f64).collect()
}

/// Max abs diff between two slices.
fn max_abs_diff(a: &[f32], b: &[f64]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y as f32).abs())
        .fold(0.0f32, f32::max)
}

// ─── G1 tests ───────────────────────────────────────────────────────────────

/// Tiny config: 4 experts, 1 shared, K=2, d=8, d_ffn=16, latent MoE d_moe=6.
fn tiny_config() -> MoeConfig {
    MoeConfig {
        num_experts: 4,
        num_shared_experts: 1,
        num_experts_per_token: 2,
        moe_intermediate_size: 16,
        hidden_size: 8,
        use_sigmoid_router: true,
        renormalize: true,
        routed_expert_hidden_size: Some(6),
        latent_moe_use_norm: true,
        rms_norm_eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
    }
}

#[test]
fn g1_zero_bias_matches_reference() {
    // Bias = 0 → pure sigmoid router, no noaux_tc influence on selection.
    let config = tiny_config();
    let mut weights = MoeWeights::random(&config, 42);
    // Zero out the bias.
    weights.e_score_correction_bias.fill(0.0);
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "zero-bias f32 vs f64 max diff = {diff} (tol 1e-4)"
    );
}

#[test]
fn g1_nonzero_bias_matches_reference() {
    // Bias ≠ 0 → noaux_tc influences selection. f32 must still match f64.
    let config = tiny_config();
    let weights = MoeWeights::random(&config, 99);
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.15 - 0.6).collect();
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "nonzero-bias f32 vs f64 max diff = {diff} (tol 1e-4)"
    );
}

#[test]
fn g1_bias_changes_selection() {
    // Prove the bias actually changes WHICH experts are picked.
    let config = tiny_config();
    let mut weights_zero_bias = MoeWeights::random(&config, 7);
    weights_zero_bias.e_score_correction_bias.fill(0.0);
    let mut weights_with_bias = weights_zero_bias.clone();
    // Set a strong bias that flips selection: boost expert 0 + 2, suppress 1 + 3.
    weights_with_bias.e_score_correction_bias = vec![0.9, -0.9, 0.9, -0.9];

    let hidden_in: Vec<f32> = vec![0.3; config.d()];

    let mut scratch = MoeForwardScratch::new(&config);
    let mut out = vec![0.0; config.d()];
    moe_forward_token(&weights_zero_bias, &config, &hidden_in, &mut out, &mut scratch);
    let idx_zero = scratch.topk_indices.clone();

    moe_forward_token(&weights_with_bias, &config, &hidden_in, &mut out, &mut scratch);
    let idx_biased = scratch.topk_indices.clone();

    // With the strong bias, experts {0, 2} should be selected (vs whatever the
    // raw sigmoid router picked under zero bias). We don't assert the exact
    // set under zero bias (depends on RNG), but we DO assert the sets differ.
    assert_ne!(
        idx_zero, idx_biased,
        "bias must change top-K selection — otherwise noaux_tc is a no-op"
    );

    // And under bias, experts 0 + 2 should win (they got +0.9).
    let mut sorted = idx_biased.clone();
    sorted.sort();
    assert_eq!(
        sorted, vec![0, 2],
        "with bias [+0.9, -0.9, +0.9, -0.9], experts {{0, 2}} must be selected, got {idx_biased:?}"
    );
}

#[test]
fn g1_bias_does_not_leak_into_renormalization() {
    // THE LOAD-BEARING TEST (Research 328 §7.6).
    //
    // The bias must participate in top-K SELECTION but NOT in renormalization.
    // We implement a "buggy" reference that uses `biased[topk_idx]` in the
    // renorm, run it, assert its output DIFFERS from the correct reference,
    // then assert the f32 impl matches the CORRECT reference.
    //
    // This catches the §2.2 misreading bit-identically: if the f32 impl used
    // biased scores in renorm, it would match the buggy reference, not the
    // correct one.
    let config = tiny_config();
    let weights = MoeWeights::random(&config, 314);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.2 - 0.7).collect();

    // f32 impl
    let mut scratch = MoeForwardScratch::new(&config);
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    // Correct f64 reference (raw-score renorm)
    let mut ref_correct = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_correct, false);

    // Buggy f64 reference (biased-score renorm)
    let mut ref_buggy = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_buggy, true);

    // (a) The two references must DIFFER — otherwise the test can't discriminate.
    let ref_diff: f64 = ref_correct
        .iter()
        .zip(ref_buggy.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        ref_diff > 1e-5,
        "correct vs buggy reference must differ when bias ≠ 0 (got max diff {ref_diff})"
    );

    // (b) The f32 impl must match the CORRECT reference, not the buggy one.
    let diff_correct = max_abs_diff(&f32_out, &ref_correct);
    let diff_buggy = max_abs_diff(&f32_out, &ref_buggy);
    assert!(
        diff_correct < 1e-4,
        "f32 must match CORRECT reference (raw-score renorm); diff = {diff_correct} (tol 1e-4)"
    );
    assert!(
        diff_buggy > 1e-5,
        "f32 must NOT match buggy reference (biased-score renorm); diff = {diff_buggy} (should be > 1e-5)"
    );
}

#[test]
fn g1_shared_expert_always_on() {
    // Disable all routed experts (zero their weights); the output must still
    // equal the shared-expert forward (proving the shared expert is ungated).
    let config = tiny_config();
    let mut weights = MoeWeights::random(&config, 55);
    for expert in &mut weights.experts {
        expert.gate_proj.fill(0.0);
        expert.up_proj.fill(0.0);
        expert.down_proj.fill(0.0);
    }
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.1 - 0.3).collect();

    let mut scratch = MoeForwardScratch::new(&config);
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "shared-expert-only f32 vs f64 max diff = {diff} (tol 1e-4)"
    );

    // And the output must be non-zero (the shared expert actually fires).
    let magnitude: f32 = f32_out.iter().map(|v| v.abs()).sum();
    assert!(
        magnitude > 1e-3,
        "shared expert output must be non-zero, got sum |v| = {magnitude}"
    );
}

#[test]
fn g1_kimi_k3_0_40b_dims_match_reference() {
    // Full Kimi-K3-0.40B dims: 8 experts, 1 shared, K=2, d=1024, d_moe=512, d_ffn=256.
    // Tolerance 1e-3 (larger dims accumulate more f32 error).
    let config = MoeConfig::kimi_k3_0_40b();
    let weights = MoeWeights::random(&config, 2718);
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in = vec![0.05; config.d()];
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-3,
        "kimi_k3_0_40b dims f32 vs f64 max diff = {diff} (tol 1e-3)"
    );
}

#[test]
fn g1_renormalization_disabled_matches_reference() {
    // When moe_renormalize=false, the f32 impl must use raw sigmoid scores
    // as weights (not renormalized). Verify against the reference.
    let mut config = tiny_config();
    config.renormalize = false;
    let weights = MoeWeights::random(&config, 161);
    let mut scratch = MoeForwardScratch::new(&config);
    let hidden_in: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.12).collect();
    let mut f32_out = vec![0.0; config.d()];
    moe_forward_token(&weights, &config, &hidden_in, &mut f32_out, &mut scratch);

    let mut ref_out = vec![0.0f64; config.d()];
    ref_moe_forward_f64(&weights, &config, &hidden_in, &mut ref_out, false);

    let diff = max_abs_diff(&f32_out, &ref_out);
    assert!(
        diff < 1e-4,
        "renormalize=false f32 vs f64 max diff = {diff} (tol 1e-4)"
    );
}
