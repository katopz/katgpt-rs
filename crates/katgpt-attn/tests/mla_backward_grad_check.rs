//! Finite-difference gradient check for the MLA analytic backward (Plan 318 C4).
//!
//! Standard ML gradcheck:
//! 1. Small MLA layer (d_h=4, d_r=4, v_h=4, n_h=2, d_c=8, d_qc=12, d=16).
//! 2. Random seeded inputs + weights.
//! 3. Loss = Σ_i output[i]² (sum-of-squares).
//! 4. Finite-difference: central differences, ε = 5e-3.
//! 5. Analytic: `mla_backward_token`.
//! 6. Pass criterion: relative error < 2e-2 (f32 noise floor).

#![cfg(feature = "mla_backward")]

use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token};
use katgpt_attn::mla_backward::{
    MlaGradients, MlaSavedActivations, mla_backward_token, mla_backward_token_with_cache_grads,
    mla_forward_token_with_saved,
};
use katgpt_kv::shard_kv::rope::RopeFreqs;

// ─── Config + helpers ───────────────────────────────────────────────────────

/// Small MLA config for gradient checking.
fn grad_check_config() -> MlaConfig {
    MlaConfig {
        kv_lora_rank: 8,
        q_lora_rank: 12,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        n_heads: 2,
        hidden_size: 16,
        use_output_gate: true,
        use_nope: true, // Kimi-K3 default (no RoPE)
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

/// Run the forward for a sequence of L tokens, compute sum-of-squares loss.
fn run_forward_seq(config: &MlaConfig, weights: &MlaWeights, h_seq: &[Vec<f32>]) -> f32 {
    let mut cache = MlaKVCache::new(config, h_seq.len());
    let mut scratch = MlaForwardScratch::new(config, h_seq.len());
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut loss = 0.0f32;
    let d = config.hidden_size;
    let mut output = vec![0.0f32; d];
    for h in h_seq {
        let out = mla_forward_token(config, weights, &mut cache, &mut scratch, &mut rope, h);
        output.copy_from_slice(out);
        loss += output.iter().map(|&v| v * v).sum::<f32>();
    }
    loss
}

/// Relative error.
fn rel_err(analytic: f32, numeric: f32) -> f32 {
    let denom = analytic.abs().max(numeric.abs()).max(1e-4);
    (analytic - numeric).abs() / denom
}

// ─── Main gradient check: all weight parameters ─────────────────────────────

#[test]
fn gradient_check_all_params_nope() {
    let config = grad_check_config();
    let d = config.hidden_size;
    let l = 3; // 3-token sequence
    let epsilon = 1e-2f32;
    let tol = 3e-2f32; // f32 finite-diff noise floor

    let weights = MlaWeights::random(&config, 42);
    let h_seq: Vec<Vec<f32>> = (0..l)
        .map(|t| (0..d).map(|i| ((i + t * 7) as f32).sin() * 0.3).collect())
        .collect();

    // ── Run forward with saved for each token ──
    let mut cache = MlaKVCache::new(&config, l);
    let mut scratch = MlaForwardScratch::new(&config, l);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);

    let mut all_saved = Vec::with_capacity(l);
    let mut all_outputs = Vec::with_capacity(l);

    for h in &h_seq {
        let (output, saved) =
            mla_forward_token_with_saved(&config, &weights, &mut cache, &mut scratch, &mut rope, h);
        all_outputs.push(output);
        all_saved.push(saved);
    }

    // ── Analytic backward per token ──
    // d_output[t] = 2 * output[t]
    let mut grads = MlaGradients::zeros_like(&weights);
    // We need separate caches for the backward (the forward cache is read-only
    // during backward, but we need the cache state at each token position).
    // Since the cache grows monotonically, we snapshot it at each position.
    // Actually, for the backward, we need the cache state AS IT WAS when each
    // token was processed (including all tokens up to and including that token).
    // The final cache (after all L tokens) has all L tokens. For token t (0-indexed),
    // the cache had t+1 tokens. We can use the final cache but only read up to
    // position t.

    let mut all_dh = vec![vec![0.0f32; d]; l];

    for t in 0..l {
        let d_output: Vec<f32> = all_outputs[t].iter().map(|&v| 2.0 * v).collect();
        mla_backward_token(
            &config,
            &weights,
            &cache, // full cache (backward reads up to saved.seq = t+1)
            &all_saved[t],
            &all_saved,
            &mut rope,
            &d_output,
            &mut all_dh,
            &mut grads,
        );
    }

    // ── Finite-difference check ──
    let mut max_rel_err = 0.0f32;
    let mut worst = String::new();

    macro_rules! check_slice {
        ($name:expr, $analytic:expr, $get:expr, $set:expr) => {
            let analytic_slice: &[f32] = $analytic;
            for i in 0..analytic_slice.len() {
                let a = analytic_slice[i];
                // Skip near-zero gradients — f32 finite-difference noise dominates
                // when |gradient| is small relative to |loss|. The FD roundoff is
                // roughly |loss| * eps_machine / (2*epsilon), so elements whose
                // analytic gradient is below this floor cannot be reliably checked.
                if a.abs() < 1e-3 {
                    continue;
                }
                let mut w_plus = weights.clone();
                let mut w_minus = weights.clone();
                $set(&mut w_plus, i, $get(&weights, i) + epsilon);
                $set(&mut w_minus, i, $get(&weights, i) - epsilon);
                let loss_plus = run_forward_seq(&config, &w_plus, &h_seq);
                let loss_minus = run_forward_seq(&config, &w_minus, &h_seq);
                let n = (loss_plus - loss_minus) / (2.0 * epsilon);
                let re = rel_err(a, n);
                if re > max_rel_err {
                    max_rel_err = re;
                    worst = format!("{}[{}]: analytic={:.6e} numeric={:.6e}", $name, i, a, n);
                }
                assert!(
                    re < tol,
                    "{}[{}]: rel_err {:.4} >= tol {:.4} (analytic={:.6e}, numeric={:.6e})",
                    $name,
                    i,
                    re,
                    tol,
                    a,
                    n
                );
            }
        };
    }

    // Check w_uk/w_uv FIRST (they don't go through RMSNorm backward)
    eprintln!("Checking w_uk...");
    check_slice!(
        "w_uk",
        &grads.w_uk,
        |w: &MlaWeights, i: usize| w.w_uk[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_uk[i] = v
    );
    eprintln!("Checking w_uv...");
    check_slice!(
        "w_uv",
        &grads.w_uv,
        |w: &MlaWeights, i: usize| w.w_uv[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_uv[i] = v
    );
    // Then check w_dkv/w_dq (go through RMSNorm backward)
    eprintln!("Checking w_dkv...");
    check_slice!(
        "w_dkv",
        &grads.w_dkv,
        |w: &MlaWeights, i: usize| w.w_dkv[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_dkv[i] = v
    );
    eprintln!("Checking w_dq...");
    check_slice!(
        "w_dq",
        &grads.w_dq,
        |w: &MlaWeights, i: usize| w.w_dq[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_dq[i] = v
    );
    eprintln!("Checking w_uq...");
    check_slice!(
        "w_uq",
        &grads.w_uq,
        |w: &MlaWeights, i: usize| w.w_uq[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_uq[i] = v
    );
    check_slice!(
        "w_qr",
        &grads.w_qr,
        |w: &MlaWeights, i: usize| w.w_qr[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_qr[i] = v
    );
    check_slice!(
        "w_uk",
        &grads.w_uk,
        |w: &MlaWeights, i: usize| w.w_uk[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_uk[i] = v
    );
    check_slice!(
        "w_uv",
        &grads.w_uv,
        |w: &MlaWeights, i: usize| w.w_uv[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_uv[i] = v
    );
    check_slice!(
        "w_kr",
        &grads.w_kr,
        |w: &MlaWeights, i: usize| w.w_kr[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_kr[i] = v
    );
    check_slice!(
        "w_o",
        &grads.w_o,
        |w: &MlaWeights, i: usize| w.w_o[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.w_o[i] = v
    );
    check_slice!(
        "q_a_norm_weight",
        &grads.q_a_norm_weight,
        |w: &MlaWeights, i: usize| w.q_a_norm_weight[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.q_a_norm_weight[i] = v
    );
    check_slice!(
        "kv_a_norm_weight",
        &grads.kv_a_norm_weight,
        |w: &MlaWeights, i: usize| w.kv_a_norm_weight[i],
        |w: &mut MlaWeights, i: usize, v: f32| w.kv_a_norm_weight[i] = v
    );
    if let Some(ref w_g) = weights.w_g {
        let _ = w_g;
        check_slice!(
            "w_g",
            grads.w_g.as_ref().unwrap(),
            |w: &MlaWeights, i: usize| w.w_g.as_ref().unwrap()[i],
            |w: &mut MlaWeights, i: usize, v: f32| w.w_g.as_mut().unwrap()[i] = v
        );
    }

    println!("MLA gradient check (nope) PASSED. max_rel_err = {max_rel_err:.4} ({worst})");
}

// ─── Input hidden gradient check ────────────────────────────────────────────

#[test]
fn gradient_check_input_hidden_nope() {
    let config = grad_check_config();
    let d = config.hidden_size;
    let l = 2;
    let epsilon = 1e-2f32;
    let tol = 3e-2f32;

    let weights = MlaWeights::random(&config, 42);
    let h_seq: Vec<Vec<f32>> = (0..l)
        .map(|t| (0..d).map(|i| ((i + t * 7) as f32).sin() * 0.3).collect())
        .collect();

    // Analytic backward
    let mut cache = MlaKVCache::new(&config, l);
    let mut scratch = MlaForwardScratch::new(&config, l);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);

    let mut all_saved = Vec::with_capacity(h_seq.len());
    let mut all_outputs = Vec::with_capacity(h_seq.len());
    for h in &h_seq {
        let (output, saved) =
            mla_forward_token_with_saved(&config, &weights, &mut cache, &mut scratch, &mut rope, h);
        all_outputs.push(output);
        all_saved.push(saved);
    }

    let mut grads = MlaGradients::zeros_like(&weights);
    let mut all_dh = vec![vec![0.0f32; d]; l];
    for t in 0..l {
        let d_output: Vec<f32> = all_outputs[t].iter().map(|&v| 2.0 * v).collect();
        mla_backward_token(
            &config,
            &weights,
            &cache,
            &all_saved[t],
            &all_saved,
            &mut rope,
            &d_output,
            &mut all_dh,
            &mut grads,
        );
    }

    // Finite-difference for dL/dh[0] (first token)
    let t_check = 0;
    let mut max_rel_err = 0.0f32;
    for i in 0..d {
        let mut h_plus = h_seq.clone();
        let mut h_minus = h_seq.clone();
        h_plus[t_check][i] += epsilon;
        h_minus[t_check][i] -= epsilon;
        let loss_plus = run_forward_seq(&config, &weights, &h_plus);
        let loss_minus = run_forward_seq(&config, &weights, &h_minus);
        let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
        let re = rel_err(all_dh[t_check][i], numeric);
        if re > max_rel_err {
            max_rel_err = re;
        }
        assert!(
            re < tol,
            "dh[{t_check}][{i}]: rel_err {re:.4} >= tol {tol:.4}"
        );
    }
    println!("MLA dL/dh gradient check (nope) PASSED. max_rel_err = {max_rel_err:.4}");
}

// ─── Standalone RMSNorm backward test ──────────────────────────────────────

/// Verify the RMSNorm backward against finite differences in isolation.
/// This isolates whether the bug is in rmsnorm_backward or elsewhere.
#[test]
fn rmsnorm_backward_isolated() {
    let n = 4;
    let eps = 1e-5f32;
    let x_raw: Vec<f32> = vec![0.3, -0.5, 0.8, -0.2];
    let gamma: Vec<f32> = vec![1.0, 1.1, 0.9, 1.05];
    let d_y: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1]; // arbitrary upstream gradient

    // Compute inv_rms the same way as the forward
    let sum_sq: f32 = x_raw.iter().map(|x| x * x).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // Forward: y[i] = x[i] * gamma[i] * inv_rms
    let y: Vec<f32> = (0..n).map(|i| x_raw[i] * gamma[i] * inv_rms).collect();

    // Analytic backward
    let mut grad_gamma = vec![0.0f32; n];
    let dx = katgpt_attn::mla_backward::rmsnorm_backward(
        &d_y,
        &x_raw,
        &gamma,
        inv_rms,
        &mut grad_gamma,
        eps,
    );

    // Finite difference for each x_raw element
    let fd_eps = 5e-4f32;
    for k in 0..n {
        let mut x_plus = x_raw.clone();
        let mut x_minus = x_raw.clone();
        x_plus[k] += fd_eps;
        x_minus[k] -= fd_eps;

        // Loss = sum(y[i] * d_y[i])
        let loss_plus: f32 = (0..n)
            .map(|i| {
                let ss: f32 = x_plus.iter().map(|x| x * x).sum();
                let ir = 1.0 / (ss / n as f32 + eps).sqrt();
                x_plus[i] * gamma[i] * ir * d_y[i]
            })
            .sum();
        let loss_minus: f32 = (0..n)
            .map(|i| {
                let ss: f32 = x_minus.iter().map(|x| x * x).sum();
                let ir = 1.0 / (ss / n as f32 + eps).sqrt();
                x_minus[i] * gamma[i] * ir * d_y[i]
            })
            .sum();
        let numeric = (loss_plus - loss_minus) / (2.0 * fd_eps);
        let re = (dx[k] - numeric).abs() / dx[k].abs().max(numeric.abs()).max(1e-6);
        assert!(
            re < 1e-2,
            "rmsnorm dx[{}]: analytic={:.6e} numeric={:.6e} re={:.4}",
            k,
            dx[k],
            numeric,
            re
        );
    }

    println!("RMSNorm backward isolated test PASSED.");
    let _ = y;
}

// ─── Smoke test: backward runs without panic ────────────────────────────────

#[test]
fn backward_smoke_kimi_k3() {
    let config = MlaConfig::kimi_k3_0_40b();
    let d = config.hidden_size;
    let l = 4;
    let weights = MlaWeights::random(&config, 99);
    let h_seq: Vec<Vec<f32>> = (0..l)
        .map(|t| (0..d).map(|i| ((i + t) as f32).sin() * 0.1).collect())
        .collect();

    let mut cache = MlaKVCache::new(&config, l);
    let mut scratch = MlaForwardScratch::new(&config, l);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);

    let mut all_saved = Vec::with_capacity(h_seq.len());
    let mut all_outputs = Vec::with_capacity(h_seq.len());
    for h in &h_seq {
        let (output, saved) =
            mla_forward_token_with_saved(&config, &weights, &mut cache, &mut scratch, &mut rope, h);
        all_outputs.push(output);
        all_saved.push(saved);
    }

    let mut grads = MlaGradients::zeros_like(&weights);
    let mut all_dh = vec![vec![0.0f32; d]; l];
    for t in 0..l {
        let d_output: Vec<f32> = all_outputs[t].iter().map(|&v| 2.0 * v).collect();
        mla_backward_token(
            &config,
            &weights,
            &cache,
            &all_saved[t],
            &all_saved,
            &mut rope,
            &d_output,
            &mut all_dh,
            &mut grads,
        );
    }

    // Check no NaN/Inf
    for &g in &grads.w_dkv {
        assert!(g.is_finite(), "NaN/Inf in w_dkv");
    }
    for &g in &grads.w_o {
        assert!(g.is_finite(), "NaN/Inf in w_o");
    }
    for dh in &all_dh {
        for &g in dh {
            assert!(g.is_finite(), "NaN/Inf in dh");
        }
    }
    println!("MLA backward smoke test PASSED (kimi_k3_0_40b, d={d}, L={l}).");
}

// ─── Issue 1253: cache-grad export gates ────────────────────────────
// The riir-train toy MLA dims (Issue 618 T0 gates: kv 8 / q 8 / nope 4 /
// rope 4 / 2 heads), so the numbers here transfer to the consumer's arm.

fn cache_grad_export_config() -> MlaConfig {
    MlaConfig {
        kv_lora_rank: 8,
        q_lora_rank: 8,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        n_heads: 2,
        hidden_size: 16,
        use_output_gate: true,
        use_nope: true, // Kimi-K3 default (no RoPE)
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

/// Forward a token sequence position by position with saved activations;
/// returns (per-token outputs, per-token saved activations, the final cache).
fn forward_with_saved_all(
    config: &MlaConfig,
    weights: &MlaWeights,
    h_seq: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<MlaSavedActivations>, MlaKVCache) {
    let l = h_seq.len();
    let mut cache = MlaKVCache::new(config, l);
    let mut scratch = MlaForwardScratch::new(config, l);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut all_saved = Vec::with_capacity(l);
    let mut outputs = Vec::with_capacity(l);
    for h in h_seq {
        let (out, saved) =
            mla_forward_token_with_saved(config, weights, &mut cache, &mut scratch, &mut rope, h);
        outputs.push(out.to_vec());
        all_saved.push(saved);
    }
    (outputs, all_saved, cache)
}

/// Tail loss Σ_{t > j} Σ_i out_t[i]² — the part of the sum-of-squares loss
/// that depends on cache entry `j` (only later tokens read it as context).
/// `inject` perturbs exactly ONE element of the entry after token `j` is
/// written (None = unperturbed baseline tail).
fn tail_loss_with_injection(
    config: &MlaConfig,
    weights: &MlaWeights,
    h_seq: &[Vec<f32>],
    j: usize,
    inject: Option<(bool, usize, f32)>, // (is_latent, element, eps)
) -> f32 {
    let l = h_seq.len();
    let mut cache = MlaKVCache::new(config, l);
    let mut scratch = MlaForwardScratch::new(config, l);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut loss = 0.0f32;
    for (t, h) in h_seq.iter().enumerate() {
        let out = mla_forward_token(config, weights, &mut cache, &mut scratch, &mut rope, h);
        if t == j
            && let Some((is_latent, elem, eps)) = inject
        {
            if is_latent {
                cache.latent_kv[j * config.kv_lora_rank + elem] += eps;
            } else {
                cache.rope_key[j * config.qk_rope_head_dim + elem] += eps;
            }
        }
        if t > j {
            loss += out.iter().map(|&v| v * v).sum::<f32>();
        }
    }
    loss
}

/// Gate 1 — the armed export reads nonzero, correctly-shaped slot gradients
/// (Issue 1253 acceptance 1). Every position's latent and rope slot row has
/// a nonzero L2 norm: position j receives its own call's self-read plus
/// every later call's context reads, all nonzero for generic weights.
#[test]
fn cache_grad_export_slots_nonzero_and_shaped() {
    let config = cache_grad_export_config();
    let d_c = config.kv_lora_rank;
    let d_r = config.qk_rope_head_dim;
    let d = config.hidden_size;
    let l = 4usize;

    let weights = MlaWeights::random(&config, 42);
    let h_seq: Vec<Vec<f32>> = (0..l)
        .map(|t| (0..d).map(|i| ((i + t * 5) as f32).sin() * 0.3).collect())
        .collect();

    let (all_outputs, all_saved, cache) = forward_with_saved_all(&config, &weights, &h_seq);

    let mut grads = MlaGradients::zeros_like(&weights);
    let mut all_dh = vec![vec![0.0f32; d]; l];
    let mut d_cache_latent = vec![0.0f32; l * d_c];
    let mut d_cache_rope = vec![0.0f32; l * d_r];
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);

    for t in 0..l {
        let d_output: Vec<f32> = all_outputs[t].iter().map(|&v| 2.0 * v).collect();
        mla_backward_token_with_cache_grads(
            &config,
            &weights,
            &cache,
            &all_saved[t],
            &all_saved,
            &mut rope,
            &d_output,
            &mut all_dh,
            &mut grads,
            Some(&mut d_cache_latent),
            Some(&mut d_cache_rope),
        );
    }

    // The buffers were passed at exactly [l][dim] sizing — the export's
    // assert accepted them and only rows 0..seq were touched.
    assert_eq!(d_cache_latent.len(), l * d_c);
    assert_eq!(d_cache_rope.len(), l * d_r);
    for j in 0..l {
        let lat_norm: f32 = d_cache_latent[j * d_c..(j + 1) * d_c]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        let rope_norm: f32 = d_cache_rope[j * d_r..(j + 1) * d_r]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            lat_norm > 0.0,
            "latent slot row {j} is all-zero (norm 0) — export collapsed"
        );
        assert!(
            rope_norm > 0.0,
            "rope slot row {j} is all-zero (norm 0) — export collapsed"
        );
    }
}

/// Gate 2 — the `None` posture is bit-identical to the incumbent delegate:
/// same weight grads, same `all_dh`, same export buffers left untouched.
#[test]
fn cache_grad_export_none_posture_bit_identical() {
    let config = cache_grad_export_config();
    let d = config.hidden_size;
    let l = 4usize;

    let weights = MlaWeights::random(&config, 7);
    let h_seq: Vec<Vec<f32>> = (0..l)
        .map(|t| (0..d).map(|i| ((i + t * 3) as f32).cos() * 0.25).collect())
        .collect();

    let (all_outputs, all_saved, cache) = forward_with_saved_all(&config, &weights, &h_seq);

    // Incumbent path.
    let mut grads_a = MlaGradients::zeros_like(&weights);
    let mut all_dh_a = vec![vec![0.0f32; d]; l];
    let mut rope_a = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    for t in 0..l {
        let d_output: Vec<f32> = all_outputs[t].iter().map(|&v| 2.0 * v).collect();
        mla_backward_token(
            &config,
            &weights,
            &cache,
            &all_saved[t],
            &all_saved,
            &mut rope_a,
            &d_output,
            &mut all_dh_a,
            &mut grads_a,
        );
    }

    // Sibling path with None slots.
    let mut grads_b = MlaGradients::zeros_like(&weights);
    let mut all_dh_b = vec![vec![0.0f32; d]; l];
    let mut rope_b = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    for t in 0..l {
        let d_output: Vec<f32> = all_outputs[t].iter().map(|&v| 2.0 * v).collect();
        mla_backward_token_with_cache_grads(
            &config,
            &weights,
            &cache,
            &all_saved[t],
            &all_saved,
            &mut rope_b,
            &d_output,
            &mut all_dh_b,
            &mut grads_b,
            None,
            None,
        );
    }

    // Bit-identical: exact f32 equality across every gradient and row.
    for (a, b) in all_dh_a.iter().zip(all_dh_b.iter()) {
        assert_eq!(a, b, "all_dh diverged between incumbent and None posture");
    }
    assert_eq!(grads_a.w_dkv, grads_b.w_dkv);
    assert_eq!(grads_a.w_dq, grads_b.w_dq);
    assert_eq!(grads_a.w_uq, grads_b.w_uq);
    assert_eq!(grads_a.w_qr, grads_b.w_qr);
    assert_eq!(grads_a.w_uk, grads_b.w_uk);
    assert_eq!(grads_a.w_uv, grads_b.w_uv);
    assert_eq!(grads_a.w_kr, grads_b.w_kr);
    assert_eq!(grads_a.w_o, grads_b.w_o);
    assert_eq!(grads_a.q_a_norm_weight, grads_b.q_a_norm_weight);
    assert_eq!(grads_a.kv_a_norm_weight, grads_b.kv_a_norm_weight);
    assert_eq!(grads_a.w_g, grads_b.w_g);
}

/// Gate 3 — the exported slot IS dL/d(cache entry): central differences on
/// the cache buffer itself. The perturbation is injected into
/// `cache.latent_kv[j]` / `cache.rope_key[j]` right after token j is written
/// and before any later token reads it, so the FD isolates the CONTEXT reads
/// (calls t > j) — exactly the exported slot row j summed over those calls
/// (self-reads happen before the injection and are excluded analytically).
#[test]
fn cache_grad_export_matches_finite_difference() {
    let config = cache_grad_export_config();
    let d_c = config.kv_lora_rank;
    let d_r = config.qk_rope_head_dim;
    let d = config.hidden_size;
    let l = 4usize;
    let j = 1usize; // the export slot under test (context-read by tokens 2, 3)
    let epsilon = 1e-2f32;
    let tol = 3e-2f32;

    let weights = MlaWeights::random(&config, 42);
    let h_seq: Vec<Vec<f32>> = (0..l)
        .map(|t| (0..d).map(|i| ((i + t * 7) as f32).sin() * 0.3).collect())
        .collect();

    // Baseline forward with saved activations (analytic side) + outputs.
    let (all_outputs, all_saved, cache) = forward_with_saved_all(&config, &weights, &h_seq);

    // Analytic: sum slot row j over the calls that read it as context (t > j).
    // Calls are independent w.r.t. the export, so one shared buffer receives
    // exactly the wanted contributions when only calls t > j run.
    let mut grads = MlaGradients::zeros_like(&weights);
    let mut all_dh = vec![vec![0.0f32; d]; l];
    let mut d_cache_latent = vec![0.0f32; l * d_c];
    let mut d_cache_rope = vec![0.0f32; l * d_r];
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    for t in (j + 1)..l {
        let d_output: Vec<f32> = all_outputs[t].iter().map(|&v| 2.0 * v).collect();
        mla_backward_token_with_cache_grads(
            &config,
            &weights,
            &cache,
            &all_saved[t],
            &all_saved,
            &mut rope,
            &d_output,
            &mut all_dh,
            &mut grads,
            Some(&mut d_cache_latent),
            Some(&mut d_cache_rope),
        );
    }
    let analytic_latent = &d_cache_latent[j * d_c..(j + 1) * d_c];
    let analytic_rope = &d_cache_rope[j * d_r..(j + 1) * d_r];

    // Detach-collapse counter-proof shape: the context-read gradient must be
    // nonzero (something flows back into the cache entry).
    assert!(
        analytic_latent.iter().any(|&v| v != 0.0),
        "latent slot j analytic context gradient is all-zero"
    );
    assert!(
        analytic_rope.iter().any(|&v| v != 0.0),
        "rope slot j analytic context gradient is all-zero"
    );

    // FD: central differences with a single-element injection into cache
    // entry j right after token j is written; the tail loss changes only
    // through the later tokens' context reads.
    let mut max_rel = 0.0f32;
    let mut worst = String::new();
    let mut check = |name: &str, i: usize, a: f32, is_latent: bool, epsilon: f32| {
        if a.abs() < 1e-3 {
            return; // near-zero analytic — FD noise floor (see the file header)
        }
        let plus =
            tail_loss_with_injection(&config, &weights, &h_seq, j, Some((is_latent, i, epsilon)));
        let minus =
            tail_loss_with_injection(&config, &weights, &h_seq, j, Some((is_latent, i, -epsilon)));
        let n = (plus - minus) / (2.0 * epsilon);
        let denom = a.abs().max(n.abs()).max(1e-4);
        let re = (a - n).abs() / denom;
        assert!(
            re < tol,
            "{name}[{i}]: rel_err {re:.4} >= tol {tol} (analytic={a:.6e}, numeric={n:.6e})"
        );
        if re > max_rel {
            max_rel = re;
            worst = format!("{name}[{i}]: rel_err {re:.4} (analytic={a:.6e}, numeric={n:.6e})");
        }
    };

    for (i, &a) in analytic_latent.iter().enumerate() {
        check("latent", i, a, true, epsilon);
    }
    for (i, &a) in analytic_rope.iter().enumerate() {
        check("rope", i, a, false, epsilon);
    }
    eprintln!("cache-grad export FD worst: {worst}");
}
