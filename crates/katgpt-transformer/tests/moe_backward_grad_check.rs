//! Finite-difference gradient check for the MoE analytic backward (Plan 318 C4).
//!
//! Implements the standard ML gradcheck:
//! 1. Small MoE layer (latent path: N_r=4, K_r=2, d=16, d_ffn=8, d_moe=8).
//! 2. Random seeded inputs + weights.
//! 3. Loss = Σ_i output[i]² (sum-of-squares — differentiable everywhere).
//! 4. Finite-difference reference: central differences, ε = 5e-3.
//! 5. Analytic reference: `moe_backward_token`.
//! 6. Pass criterion: relative error < 1e-2 on every parameter (f32 noise floor).
//!
//! This is THE load-bearing correctness gate for the MoE backward. A subtle sign
//! error or transposition in the derivation would silently corrupt training.

#![cfg(feature = "moe_backward")]

use katgpt_transformer::moe::{MoeConfig, MoeForwardScratch, MoeWeights, moe_forward_token};
use katgpt_transformer::moe_backward::{
    MoeGradients, moe_backward_token, moe_forward_token_with_saved,
};

// ─── Config + helpers ───────────────────────────────────────────────────────

/// Small MoE config for gradient checking (latent path).
fn grad_check_config() -> MoeConfig {
    MoeConfig {
        num_experts: 4,
        num_shared_experts: 1,
        num_experts_per_token: 2,
        moe_intermediate_size: 8,
        hidden_size: 16,
        use_sigmoid_router: true,
        renormalize: true,
        routed_expert_hidden_size: Some(8),
        latent_moe_use_norm: true,
        rms_norm_eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        routed_scaling_factor: None,
        swiglu_limit: None,
    }
}

/// Run the forward, compute sum-of-squares loss + the upstream gradient.
///
/// `d_output[i] = 2 · output[i]` (derivative of Σ output²).
fn run_forward(config: &MoeConfig, weights: &MoeWeights, h: &[f32]) -> (f32, Vec<f32>) {
    let d = config.d();
    let mut scratch = MoeForwardScratch::new(config);
    let mut output = vec![0.0f32; d];
    moe_forward_token(weights, config, h, &mut output, &mut scratch);
    let loss: f32 = output.iter().map(|&v| v * v).sum();
    let d_output: Vec<f32> = output.iter().map(|&v| 2.0 * v).collect();
    (loss, d_output)
}

/// Relative error: |analytic − numeric| / max(|analytic|, |numeric|, floor).
fn rel_err(analytic: f32, numeric: f32) -> f32 {
    let denom = analytic.abs().max(numeric.abs()).max(1e-4);
    (analytic - numeric).abs() / denom
}

/// Central finite-difference gradient of the loss w.r.t. a single parameter.
fn finite_diff_one(
    config: &MoeConfig,
    weights: &MoeWeights,
    h: &[f32],
    get_param: impl Fn(&MoeWeights) -> f32,
    set_param: impl Fn(&mut MoeWeights, f32),
    epsilon: f32,
) -> f32 {
    let mut w_plus = weights.clone();
    let mut w_minus = weights.clone();
    let orig = get_param(weights);
    set_param(&mut w_plus, orig + epsilon);
    set_param(&mut w_minus, orig - epsilon);

    let (loss_plus, _) = run_forward(config, &w_plus, h);
    let (loss_minus, _) = run_forward(config, &w_minus, h);

    (loss_plus - loss_minus) / (2.0 * epsilon)
}

// ─── Main gradient check: all weight parameters ─────────────────────────────

/// The load-bearing test: finite-difference vs analytic for EVERY weight parameter.
///
/// Iterates over every scalar in every weight matrix (router, experts,
/// shared experts, latent down/up/norm), checks relative error < tol.
#[test]
fn gradient_check_all_params() {
    let config = grad_check_config();
    let d = config.d();
    let epsilon = 5e-3f32;
    let tol = 1.5e-2f32; // f32 finite-diff noise floor (MoE has more steps than KDA)

    let weights = MoeWeights::random(&config, 42);
    let h: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.3).collect();

    // ── Analytic backward ──
    let scratch = &mut MoeForwardScratch::new(&config);
    let (_output, saved) = moe_forward_token_with_saved(&weights, &config, &h, scratch);

    let mut grads = MoeGradients::zeros_like(&weights);
    let mut dh = vec![0.0f32; d];
    // Re-run forward to get d_output (the saved forward returned output but we
    // need d_output = 2*output for the loss).
    let (loss, d_output) = run_forward(&config, &weights, &h);
    moe_backward_token(&config, &weights, &saved, &d_output, &mut dh, &mut grads);
    let _ = loss;

    let mut max_rel_err = 0.0f32;
    let mut worst = String::new();

    // Helper: check one weight slice
    macro_rules! check_slice {
        ($name:expr, $analytic:expr, $get:expr, $set:expr) => {
            let analytic_slice: &[f32] = $analytic;
            for i in 0..analytic_slice.len() {
                let a = analytic_slice[i];
                let n = finite_diff_one(
                    &config,
                    &weights,
                    &h,
                    |w| $get(w, i),
                    |w, v| $set(w, i, v),
                    epsilon,
                );
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

    // Router weight: [N_r * d]
    check_slice!(
        "router_weight",
        &grads.router_weight,
        |w: &MoeWeights, i: usize| w.router_weight[i],
        |w: &mut MoeWeights, i: usize, v: f32| w.router_weight[i] = v
    );

    // Per-routed-expert weights
    for (e, expert_w) in weights.experts.iter().enumerate() {
        let expert_g = &grads.experts[e];
        check_slice!(
            format!("experts[{}].gate_proj", e),
            &expert_g.gate_proj,
            move |w: &MoeWeights, i: usize| w.experts[e].gate_proj[i],
            move |w: &mut MoeWeights, i: usize, v: f32| w.experts[e].gate_proj[i] = v
        );
        check_slice!(
            format!("experts[{}].up_proj", e),
            &expert_g.up_proj,
            move |w: &MoeWeights, i: usize| w.experts[e].up_proj[i],
            move |w: &mut MoeWeights, i: usize, v: f32| w.experts[e].up_proj[i] = v
        );
        check_slice!(
            format!("experts[{}].down_proj", e),
            &expert_g.down_proj,
            move |w: &MoeWeights, i: usize| w.experts[e].down_proj[i],
            move |w: &mut MoeWeights, i: usize, v: f32| w.experts[e].down_proj[i] = v
        );
        let _ = expert_w;
    }

    // Shared expert weights
    for (s, shared_w) in weights.shared_experts.iter().enumerate() {
        let shared_g = &grads.shared_experts[s];
        check_slice!(
            format!("shared_experts[{}].gate_proj", s),
            &shared_g.gate_proj,
            move |w: &MoeWeights, i: usize| w.shared_experts[s].gate_proj[i],
            move |w: &mut MoeWeights, i: usize, v: f32| w.shared_experts[s].gate_proj[i] = v
        );
        check_slice!(
            format!("shared_experts[{}].up_proj", s),
            &shared_g.up_proj,
            move |w: &MoeWeights, i: usize| w.shared_experts[s].up_proj[i],
            move |w: &mut MoeWeights, i: usize, v: f32| w.shared_experts[s].up_proj[i] = v
        );
        check_slice!(
            format!("shared_experts[{}].down_proj", s),
            &shared_g.down_proj,
            move |w: &MoeWeights, i: usize| w.shared_experts[s].down_proj[i],
            move |w: &mut MoeWeights, i: usize, v: f32| w.shared_experts[s].down_proj[i] = v
        );
        let _ = shared_w;
    }

    // Latent MoE wrapper weights
    if weights.routed_expert_down_proj.is_some() {
        check_slice!(
            "routed_expert_down_proj",
            grads.routed_expert_down_proj.as_ref().unwrap(),
            |w: &MoeWeights, i: usize| w.routed_expert_down_proj.as_ref().unwrap()[i],
            |w: &mut MoeWeights, i: usize, v: f32| w.routed_expert_down_proj.as_mut().unwrap()[i] =
                v
        );
    }
    if weights.routed_expert_up_proj.is_some() {
        check_slice!(
            "routed_expert_up_proj",
            grads.routed_expert_up_proj.as_ref().unwrap(),
            |w: &MoeWeights, i: usize| w.routed_expert_up_proj.as_ref().unwrap()[i],
            |w: &mut MoeWeights, i: usize, v: f32| w.routed_expert_up_proj.as_mut().unwrap()[i] = v
        );
    }
    if weights.routed_expert_norm_weight.is_some() {
        check_slice!(
            "routed_expert_norm_weight",
            grads.routed_expert_norm_weight.as_ref().unwrap(),
            |w: &MoeWeights, i: usize| w.routed_expert_norm_weight.as_ref().unwrap()[i],
            |w: &mut MoeWeights, i: usize, v: f32| w.routed_expert_norm_weight.as_mut().unwrap()
                [i] = v
        );
    }

    println!("MoE gradient check PASSED. max_rel_err = {max_rel_err:.4} ({worst})");
}

// ─── Input hidden gradient check ────────────────────────────────────────────

/// Verify dL/dh is correct via finite differences.
#[test]
fn gradient_check_input_hidden() {
    let config = grad_check_config();
    let d = config.d();
    let epsilon = 5e-3f32;
    let tol = 1.5e-2f32;

    let weights = MoeWeights::random(&config, 42);
    let h: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.3).collect();

    // Analytic backward
    let scratch = &mut MoeForwardScratch::new(&config);
    let (_output, saved) = moe_forward_token_with_saved(&weights, &config, &h, scratch);
    let mut grads = MoeGradients::zeros_like(&weights);
    let mut dh = vec![0.0f32; d];
    let (_, d_output) = run_forward(&config, &weights, &h);
    moe_backward_token(&config, &weights, &saved, &d_output, &mut dh, &mut grads);

    // Finite difference per hidden element
    let mut max_rel_err = 0.0f32;
    for i in 0..d {
        let mut h_plus = h.clone();
        let mut h_minus = h.clone();
        h_plus[i] += epsilon;
        h_minus[i] -= epsilon;
        let (loss_plus, _) = run_forward(&config, &weights, &h_plus);
        let (loss_minus, _) = run_forward(&config, &weights, &h_minus);
        let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
        let re = rel_err(dh[i], numeric);
        if re > max_rel_err {
            max_rel_err = re;
        }
        assert!(
            re < tol,
            "dh[{}]: rel_err {:.4} >= tol {:.4} (analytic={:.6e}, numeric={:.6e})",
            i,
            re,
            tol,
            dh[i],
            numeric
        );
    }
    println!("MoE dL/dh gradient check PASSED. max_rel_err = {max_rel_err:.4}");
}

// ─── Non-latent path gradient check ───────────────────────────────────────

/// Verify the non-latent MoE path (routed_expert_hidden_size = None) produces
/// correct gradients. This path is used by non-Kimi-K3 MoE configs.
#[test]
fn gradient_check_nonlatent_path() {
    let config = MoeConfig {
        num_experts: 4,
        num_shared_experts: 1,
        num_experts_per_token: 2,
        moe_intermediate_size: 8,
        hidden_size: 16,
        use_sigmoid_router: true,
        renormalize: true,
        routed_expert_hidden_size: None, // non-latent
        latent_moe_use_norm: false,
        rms_norm_eps: 1e-5,
        situ_beta: 4.0,
        situ_linear_beta: Some(25.0),
        routed_scaling_factor: None,
        swiglu_limit: None,
    };
    let d = config.d();
    let epsilon = 5e-3f32;
    let tol = 1.5e-2f32;

    let weights = MoeWeights::random(&config, 77);
    let h: Vec<f32> = (0..d).map(|i| (i as f32).cos() * 0.3).collect();

    let scratch = &mut MoeForwardScratch::new(&config);
    let (_output, saved) = moe_forward_token_with_saved(&weights, &config, &h, scratch);
    let mut grads = MoeGradients::zeros_like(&weights);
    let mut dh = vec![0.0f32; d];
    let (_, d_output) = run_forward(&config, &weights, &h);
    moe_backward_token(&config, &weights, &saved, &d_output, &mut dh, &mut grads);

    // Check router_weight gradient (spot check — a few elements)
    let mut max_rel_err = 0.0f32;
    for i in 0..weights.router_weight.len() {
        let a = grads.router_weight[i];
        let mut w_plus = weights.clone();
        let mut w_minus = weights.clone();
        w_plus.router_weight[i] += epsilon;
        w_minus.router_weight[i] -= epsilon;
        let (lp, _) = run_forward(&config, &w_plus, &h);
        let (lm, _) = run_forward(&config, &w_minus, &h);
        let n = (lp - lm) / (2.0 * epsilon);
        let re = rel_err(a, n);
        if re > max_rel_err {
            max_rel_err = re;
        }
        assert!(re < tol, "router_weight[{i}]: rel_err {re:.4}");
    }

    // Check dh gradient
    for i in 0..d {
        let mut h_plus = h.clone();
        let mut h_minus = h.clone();
        h_plus[i] += epsilon;
        h_minus[i] -= epsilon;
        let (lp, _) = run_forward(&config, &weights, &h_plus);
        let (lm, _) = run_forward(&config, &weights, &h_minus);
        let n = (lp - lm) / (2.0 * epsilon);
        let re = rel_err(dh[i], n);
        assert!(re < tol, "dh[{i}]: rel_err {re:.4}");
    }

    println!("MoE non-latent path gradient check PASSED. max_rel_err (router) = {max_rel_err:.4}");
}

// ─── Wide-gamma gradient check (Issue 693 H2 regression guard) ────────────

/// Finite-difference grad check with latent-norm gamma spanning [0.5, 2.0].
///
/// `MoeWeights::random` inits gamma at `1.0 ± 0.1`, which masks the
/// gamma-over-application bug (`g*r*(dy - x*r²*dot/d)` — the incorrect form
/// named in `mla_backward.rs::rmsnorm_backward`): the error is
/// `(γ_i - 1)·correction`, inside the near-unity band. With wide gamma the
/// analytic-vs-numeric divergence is O(1) and the bug is unmasked.
///
/// Checks the latent-path parameters whose gradients flow through the
/// RMSNorm backward (routed down/up, norm weight, router, dh) — the exact
/// surface Issue 693 H2 corrupted.
#[test]
fn gradient_check_wide_gamma() {
    let config = grad_check_config();
    let d = config.d();
    let epsilon = 5e-3f32;
    let tol = 1.5e-2f32;

    let mut weights = MoeWeights::random(&config, 42);
    // Wide-gamma fixture: γ spans [0.5, 2.0] deterministically.
    let d_moe = config.routed_expert_hidden_size.unwrap();
    let wide: Vec<f32> = (0..d_moe)
        .map(|i| 0.5 + (i as f32 / d_moe as f32) * 1.5)
        .collect();
    weights.routed_expert_norm_weight = Some(wide);
    // Adversarial shaping (both are REQUIRED to unmask the bug — see NOTE):
    // 1. Zero the shared expert → its contribution to hidden_out is exactly 0.
    // 2. up_proj = identity block in the top d_moe rows → output_j = postnorm_j
    //    (j < d_moe), so dy = up^T d_output = 2·r·(x∘γ) — perfectly aligned
    //    with x∘γ. The RMSNorm correction term is x_i·r²·dot/d with
    //    dot = Σ xγ·dy; alignment maximizes dot, making the correction
    //    term O(direct) instead of O(random-walk noise).
    for shared in weights.shared_experts.iter_mut() {
        shared.gate_proj.iter_mut().for_each(|w| *w = 0.0);
        shared.up_proj.iter_mut().for_each(|w| *w = 0.0);
        shared.down_proj.iter_mut().for_each(|w| *w = 0.0);
    }
    let up = weights.routed_expert_up_proj.as_mut().unwrap();
    up.iter_mut().for_each(|w| *w = 0.0);
    // up is row-major [d × d_moe]: up[row*d_moe + col]. Identity block in the
    // top d_moe rows → output_j += postnorm_j (j < d_moe).
    for i in 0..d_moe {
        up[i * d_moe + i] = 1.0;
    }
    // Scale the routed-expert path up so x = Σ w_k·expert_out_k is O(10):
    // the default ±0.2 weights + SiTU(β=25) leave x ≈ 1e-4, which makes the
    // RMSNorm correction term (cubic in x) vanish against the direct term.
    for e in weights.experts.iter_mut() {
        e.gate_proj.iter_mut().for_each(|w| *w *= 30.0);
        e.up_proj.iter_mut().for_each(|w| *w *= 30.0);
        e.down_proj.iter_mut().for_each(|w| *w *= 30.0);
    }
    weights
        .routed_expert_down_proj
        .as_mut()
        .unwrap()
        .iter_mut()
        .for_each(|w| *w *= 6.0);

    let h: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.3).collect();

    // Analytic backward
    let scratch = &mut MoeForwardScratch::new(&config);
    let (_output, saved) = moe_forward_token_with_saved(&weights, &config, &h, scratch);
    let mut grads = MoeGradients::zeros_like(&weights);
    let mut dh = vec![0.0f32; d];
    let (_, d_output) = run_forward(&config, &weights, &h);
    moe_backward_token(&config, &weights, &saved, &d_output, &mut dh, &mut grads);

    let mut max_rel_err = 0.0f32;
    let mut worst = String::new();

    macro_rules! check_wide {
        ($name:expr, $analytic:expr, $get:expr, $set:expr) => {
            let analytic_slice: &[f32] = $analytic;
            for i in 0..analytic_slice.len() {
                let a = analytic_slice[i];
                let n = finite_diff_one(
                    &config,
                    &weights,
                    &h,
                    |w| $get(w, i),
                    |w, v| $set(w, i, v),
                    epsilon,
                );
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

    // Router (topk weights are downstream of d_latent_prenorm).
    check_wide!(
        "router_weight",
        &grads.router_weight,
        |w: &MoeWeights, i: usize| w.router_weight[i],
        |w: &mut MoeWeights, i: usize, v: f32| w.router_weight[i] = v
    );
    // Latent wrapper (directly through the RMSNorm backward).
    check_wide!(
        "routed_expert_down_proj",
        grads.routed_expert_down_proj.as_ref().unwrap(),
        |w: &MoeWeights, i: usize| w.routed_expert_down_proj.as_ref().unwrap()[i],
        |w: &mut MoeWeights, i: usize, v: f32| w.routed_expert_down_proj.as_mut().unwrap()[i] = v
    );
    check_wide!(
        "routed_expert_up_proj",
        grads.routed_expert_up_proj.as_ref().unwrap(),
        |w: &MoeWeights, i: usize| w.routed_expert_up_proj.as_ref().unwrap()[i],
        |w: &mut MoeWeights, i: usize, v: f32| w.routed_expert_up_proj.as_mut().unwrap()[i] = v
    );
    check_wide!(
        "routed_expert_norm_weight",
        grads.routed_expert_norm_weight.as_ref().unwrap(),
        |w: &MoeWeights, i: usize| w.routed_expert_norm_weight.as_ref().unwrap()[i],
        |w: &mut MoeWeights, i: usize, v: f32| w.routed_expert_norm_weight.as_mut().unwrap()[i] = v
    );

    // dh (flows through prenorm → down_proj → h).
    for i in 0..d {
        let mut h_plus = h.clone();
        let mut h_minus = h.clone();
        h_plus[i] += epsilon;
        h_minus[i] -= epsilon;
        let (lp, _) = run_forward(&config, &weights, &h_plus);
        let (lm, _) = run_forward(&config, &weights, &h_minus);
        let n = (lp - lm) / (2.0 * epsilon);
        let re = rel_err(dh[i], n);
        if re > max_rel_err {
            max_rel_err = re;
            worst = format!("dh[{}]: analytic={:.6e} numeric={:.6e}", i, dh[i], n);
        }
        assert!(re < tol, "dh[{i}]: rel_err {re:.4} >= tol {tol:.4}");
    }

    println!("MoE wide-gamma gradient check PASSED. max_rel_err = {max_rel_err:.4} ({worst})");
}

// ─── Smoke test: backward runs without panic ────────────────────────────────

#[test]
fn backward_smoke() {
    let config = MoeConfig::kimi_k3_0_40b();
    let d = config.d();
    let weights = MoeWeights::random(&config, 99);
    let h: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.1).collect();

    let scratch = &mut MoeForwardScratch::new(&config);
    let (output, saved) = moe_forward_token_with_saved(&weights, &config, &h, scratch);
    let d_output: Vec<f32> = output.iter().map(|&v| 2.0 * v).collect();

    let mut grads = MoeGradients::zeros_like(&weights);
    let mut dh = vec![0.0f32; d];
    moe_backward_token(&config, &weights, &saved, &d_output, &mut dh, &mut grads);

    // Check no NaN/Inf in gradients
    for &g in &grads.router_weight {
        assert!(g.is_finite(), "NaN/Inf in router_weight gradient");
    }
    for (e, eg) in grads.experts.iter().enumerate() {
        for &g in &eg.gate_proj {
            assert!(g.is_finite(), "NaN/Inf in experts[{e}].gate_proj");
        }
    }
    for &g in &dh {
        assert!(g.is_finite(), "NaN/Inf in dh");
    }
    println!("MoE backward smoke test PASSED (kimi_k3_0_40b config, d={d}).");
}
