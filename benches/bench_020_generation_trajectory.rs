//! Bench 020 — Autoregressive Generation Trajectory Discrimination
//!
//! Proposal 011 Phase 5 T5.6g follow-up to bench_018/019.
//!
//! ## Question
//!
//! Bench 018 proved the SEQUENCE trajectory (final hidden states across a
//! prompt's tokens) achieves 100% per-prompt discrimination at σ≥0.1. But
//! bench_018 used prompt PROCESSING — the model reads a fixed token sequence.
//!
//! The actual SWE-bench use case is GENERATION — the model WRITES tokens
//! (a patch). Does the generation trajectory (hidden states during greedy
//! decoding) discriminate as well as the processing trajectory?
//!
//! ## Method
//!
//! - 32 prompts (16 train + 16 test per model)
//! - For each prompt:
//!   1. Process a 16-token prefix (prime the KV cache)
//!   2. Greedily generate 48 tokens (argmax over logits)
//!   3. Capture `runtime.hidden` at each generation step
//! - The generation trajectory [h1, ..., h48] is encoded via the substrate
//!   `StateMagnitudeEncoder` (bench_019)
//! - Compare to the processing trajectory (bench_018's method on the same
//!   64 tokens)
//! - Classify: Euclidean nearest-centroid + Bayes-optimal ceiling Φ(d_M/2)
//!
//! ## Why this matters
//!
//! If generation trajectories are as discriminative as processing trajectories,
//! the substrate is validated for the full production use case (SWE-bench
//! patch generation). If not, the discrimination may be specific to the
//! processing regime.

#![cfg(all(feature = "kimi_k3_loader", feature = "swe_trajectory_freeze"))]
#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_core::swe_trajectory_freeze::StateMagnitudeEncoder;
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnWeights,
};
use katgpt_rs::kimi_k3::loader::{load_kimi_k3, KimiK3ModelWeights};
use katgpt_rs::kimi_k3::model::{
    kimi_k3_forward_token, kimi_k3_forward_token_traced, KimiK3ModelConfig, KimiK3Runtime,
};
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Encoder output dimension (substrate StateMagnitudeEncoder writes 8 features).
const D_ENC: usize = 8;

/// Number of classes (Model A vs Model B(σ)).
const N_CLASSES: usize = 2;

/// Number of prompts (each is one sample for classification).
const N_PROMPTS: usize = 32;

/// Training split (per model).
const N_TRAIN: usize = 16;

/// Prompt prefix length (tokens processed to prime KV cache before generation).
const PREFIX_LEN: usize = 16;

/// Number of tokens to generate greedily.
const N_GEN: usize = 48;

/// Total sequence length for the processing-trajectory comparison (matches bench_018).
const PROC_LEN: usize = PREFIX_LEN + N_GEN; // 64

/// Truncated vocab for token ID generation.
const BENCH_VOCAB: usize = 512;

/// Perturbation σ levels.
const SIGMA_LEVELS: &[f32] = &[0.0, 0.01, 0.05, 0.1, 0.5];

// ─── Deterministic LCG + weight perturbation (copied from bench_018) ───────

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

#[inline]
fn perturb_vec(v: &mut [f32], rng: &mut Lcg, sigma: f32) {
    if sigma == 0.0 {
        return;
    }
    for w in v.iter_mut() {
        let noise = rng.next_f32();
        *w *= 1.0 + sigma * noise;
    }
}

fn perturb_attn_res(w: &mut AttnResWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.norm_weight, rng, sigma);
    perturb_vec(&mut w.proj_weight, rng, sigma);
}

fn perturb_mla(w: &mut MlaWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.w_dkv, rng, sigma);
    perturb_vec(&mut w.w_dq, rng, sigma);
    perturb_vec(&mut w.w_uq, rng, sigma);
    perturb_vec(&mut w.w_qr, rng, sigma);
    perturb_vec(&mut w.w_uk, rng, sigma);
    perturb_vec(&mut w.w_uv, rng, sigma);
    perturb_vec(&mut w.w_kr, rng, sigma);
    perturb_vec(&mut w.w_o, rng, sigma);
    perturb_vec(&mut w.q_a_norm_weight, rng, sigma);
    perturb_vec(&mut w.kv_a_norm_weight, rng, sigma);
    if let Some(w_g) = w.w_g.as_mut() {
        perturb_vec(w_g, rng, sigma);
    }
}

fn perturb_kda(w: &mut KdaWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.q_proj, rng, sigma);
    perturb_vec(&mut w.k_proj, rng, sigma);
    perturb_vec(&mut w.v_proj, rng, sigma);
    perturb_vec(&mut w.q_conv_weight, rng, sigma);
    perturb_vec(&mut w.k_conv_weight, rng, sigma);
    perturb_vec(&mut w.v_conv_weight, rng, sigma);
    perturb_vec(&mut w.a_log, rng, sigma);
    perturb_vec(&mut w.f_a_proj, rng, sigma);
    perturb_vec(&mut w.f_b_proj, rng, sigma);
    perturb_vec(&mut w.dt_bias, rng, sigma);
    perturb_vec(&mut w.beta_proj, rng, sigma);
    perturb_vec(&mut w.g_proj, rng, sigma);
    perturb_vec(&mut w.o_norm_weight, rng, sigma);
    perturb_vec(&mut w.o_proj, rng, sigma);
}

fn perturb_swiglu(w: &mut SwiGluExpertWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.gate_proj, rng, sigma);
    perturb_vec(&mut w.up_proj, rng, sigma);
    perturb_vec(&mut w.down_proj, rng, sigma);
}

fn perturb_moe(w: &mut MoeWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.router_weight, rng, sigma);
    perturb_vec(&mut w.e_score_correction_bias, rng, sigma);
    for expert in w.experts.iter_mut() {
        perturb_swiglu(expert, rng, sigma);
    }
    for expert in w.shared_experts.iter_mut() {
        perturb_swiglu(expert, rng, sigma);
    }
    if let Some(p) = w.routed_expert_down_proj.as_mut() {
        perturb_vec(p, rng, sigma);
    }
    if let Some(p) = w.routed_expert_up_proj.as_mut() {
        perturb_vec(p, rng, sigma);
    }
    if let Some(p) = w.routed_expert_norm_weight.as_mut() {
        perturb_vec(p, rng, sigma);
    }
}

fn perturb_layer(w: &mut KimiDecoderLayerWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.input_layernorm_weight, rng, sigma);
    perturb_vec(&mut w.post_attention_layernorm_weight, rng, sigma);
    match &mut w.attention {
        KimiAttentionWeights::Mla(m) => perturb_mla(m, rng, sigma),
        KimiAttentionWeights::Kda(k) => perturb_kda(k, rng, sigma),
    }
    match &mut w.ffn {
        KimiFfnWeights::Dense(s) => perturb_swiglu(s, rng, sigma),
        KimiFfnWeights::Moe(m) => perturb_moe(m, rng, sigma),
    }
    perturb_attn_res(&mut w.self_attn_res, rng, sigma);
    perturb_attn_res(&mut w.mlp_attn_res, rng, sigma);
}

fn perturb_model(w: &mut KimiK3ModelWeights, sigma: f32) {
    let seed = (sigma * 1_000_000.0) as u64 | 0xA15E_0000;
    let mut rng = Lcg::new(seed);
    perturb_vec(&mut w.embed_weight, &mut rng, sigma);
    for layer in w.layers.iter_mut() {
        perturb_layer(layer, &mut rng, sigma);
    }
    perturb_vec(&mut w.final_norm_weight, &mut rng, sigma);
    if !w.lm_head_weight.is_empty() {
        perturb_vec(&mut w.lm_head_weight, &mut rng, sigma);
    }
    perturb_attn_res(&mut w.output_attn_res, &mut rng, sigma);
}

// ─── Trajectory extraction ─────────────────────────────────────────────────

/// Extract the GENERATION trajectory: process a prefix, then greedily
/// generate N tokens, capturing `runtime.hidden` at each generation step.
///
/// This differs from bench_018's processing trajectory in two ways:
/// 1. The generated tokens are MODEL-DEPENDENT (argmax over logits), not fixed
/// 2. The KV cache grows based on what the model generates, not what it reads
///
/// Returns the generation trajectory [h1, ..., hN_GEN].
fn extract_generation_trajectory(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    prefix_tokens: &[u32],
    n_gen: usize,
    gen_states: &mut Vec<Vec<f32>>,
    _depth_traj: &mut Vec<Vec<f32>>,
) -> Vec<u32> {
    runtime.reset();
    gen_states.clear();

    // Phase 1: Process the prefix (prime KV cache). Use forward_token (returns
    // logits) so we can start generation from the last prefix token's logits.
    let mut last_logits: &[f32] = &[];
    for &tok in prefix_tokens {
        last_logits = kimi_k3_forward_token(config, weights, runtime, tok);
    }
    let _ = last_logits; // used below

    // Phase 2: Greedily generate n_gen tokens.
    let mut generated = Vec::with_capacity(n_gen);
    let mut current_logits = last_logits.to_vec(); // copy (borrow ends)

    for _ in 0..n_gen {
        // Argmax over logits → next token.
        let next_tok = current_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b)).map_or(0, |(idx, _)| idx as u32);

        generated.push(next_tok);

        // Forward pass — use traced variant to get hidden state, then do
        // the LM head projection manually for the next iteration's logits.
        // Actually, kimi_k3_forward_token (non-traced) fills BOTH runtime.hidden
        // AND runtime.logits. So we use it and capture hidden after the call.
        kimi_k3_forward_token(config, weights, runtime, next_tok);
        gen_states.push(runtime.hidden.clone());

        // Update logits for next iteration.
        current_logits.clear();
        current_logits.extend_from_slice(&runtime.logits);
    }

    generated
}

/// Extract the PROCESSING trajectory (bench_018's method): process all tokens
/// sequentially, capturing the final hidden state at each step.
fn extract_processing_trajectory(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    tokens: &[u32],
    proc_states: &mut Vec<Vec<f32>>,
    depth_traj: &mut Vec<Vec<f32>>,
) {
    runtime.reset();
    proc_states.clear();

    for &tok in tokens {
        depth_traj.clear();
        let final_hidden = kimi_k3_forward_token_traced(config, weights, runtime, tok, depth_traj);
        proc_states.push(final_hidden.to_vec());
    }
}

// ─── Encoding (substrate StateMagnitudeEncoder) ────────────────────────────

fn encode_state_stats(states: &[Vec<f32>], encoder: &StateMagnitudeEncoder, out: &mut [f32; D_ENC]) {
    let refs: Vec<&[f32]> = states.iter().map(|v| v.as_slice()).collect();
    encoder.encode_into(&refs, out);
}

// ─── Classifier (Euclidean + Bayes-optimal) ────────────────────────────────

struct TrajResult {
    sigma: f32,
    regime: &'static str,
    euclidean_acc: f32,
    d_euclid: f32,
    bayes_optimal: f32,
}

/// Compute Euclidean nearest-centroid accuracy + centroid distance + Bayes-optimal.
fn evaluate(
    summaries: &[[[f32; D_ENC]; N_PROMPTS]; N_CLASSES],
    sigma: f32,
    regime: &'static str,
) -> TrajResult {
    // Train split: first N_TRAIN per class.
    let mut means = [[0.0_f32; D_ENC]; N_CLASSES];
    for k in 0..N_CLASSES {
        for j in 0..D_ENC {
            let mut sum = 0.0;
            for i in 0..N_TRAIN {
                sum += summaries[k][i][j];
            }
            means[k][j] = sum / N_TRAIN as f32;
        }
    }

    // Global centroid.
    let mut global = [0.0_f32; D_ENC];
    for j in 0..D_ENC {
        global[j] = (means[0][j] + means[1][j]) / 2.0;
    }

    // Directions (nearest-centroid).
    let mut directions = [[0.0_f32; D_ENC]; N_CLASSES];
    for k in 0..N_CLASSES {
        let mut norm_sq = 0.0_f32;
        for j in 0..D_ENC {
            directions[k][j] = means[k][j] - global[j];
            norm_sq += directions[k][j] * directions[k][j];
        }
        let norm = norm_sq.sqrt().max(1e-9);
        for j in 0..D_ENC {
            directions[k][j] /= norm;
        }
    }

    // Test split: N_TRAIN..N_PROMPTS per class.
    let mut n_correct = 0usize;
    let total = N_CLASSES * (N_PROMPTS - N_TRAIN);
    for k in 0..N_CLASSES {
        for i in N_TRAIN..N_PROMPTS {
            let x = &summaries[k][i];
            let mut centered = [0.0_f32; D_ENC];
            for j in 0..D_ENC {
                centered[j] = x[j] - global[j];
            }
            let mut best = 0usize;
            let mut best_dot = f32::NEG_INFINITY;
            for kk in 0..N_CLASSES {
                let mut dot = 0.0;
                for j in 0..D_ENC {
                    dot += centered[j] * directions[kk][j];
                }
                if dot > best_dot {
                    best_dot = dot;
                    best = kk;
                }
            }
            if best == k {
                n_correct += 1;
            }
        }
    }
    let euclidean_acc = n_correct as f32 / total as f32;

    // Centroid distance (Euclidean) for Bayes-optimal.
    let mut diff_sq = 0.0_f32;
    for j in 0..D_ENC {
        let d = means[0][j] - means[1][j];
        diff_sq += d * d;
    }
    let d_euclid = diff_sq.sqrt();

    // Within-class scatter (average variance across features).
    let mut var_sum = 0.0_f32;
    for k in 0..N_CLASSES {
        for j in 0..D_ENC {
            let mut sum_sq_dev = 0.0;
            for i in 0..N_TRAIN {
                let dev = summaries[k][i][j] - means[k][j];
                sum_sq_dev += dev * dev;
            }
            var_sum += sum_sq_dev / N_TRAIN as f32;
        }
    }
    let avg_var = var_sum / (N_CLASSES * D_ENC) as f32;
    let avg_std = avg_var.sqrt().max(1e-9);

    // Bayes-optimal for 2-class Gaussian with equal isotropic covariance:
    // Φ(d_M / 2) where d_M = ||μ_A - μ_B|| / σ.
    let d_mahalanobis = d_euclid / avg_std;
    let bayes_optimal = gaussian_cdf(d_mahalanobis / 2.0);

    TrajResult {
        sigma,
        regime,
        euclidean_acc,
        d_euclid,
        bayes_optimal,
    }
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn gaussian_cdf(x: f32) -> f32 {
    // Handle x <= 0 (including -0.0) by symmetry. Must use strict < to avoid
    // infinite recursion on -0.0 (where -(-0.0) == 0.0 == -0.0 in IEEE 754).
    if x < 0.0 {
        return 1.0 - gaussian_cdf(-x);
    }
    // x >= 0 (including +0.0 and -0.0).
    if x == 0.0 {
        return 0.5;
    }
    let k = 1.0 / (1.0 + 0.2316419 * x);
    let k2 = k * k;
    let k3 = k2 * k;
    let k4 = k3 * k;
    let poly = 0.31938153 * k - 0.35656378 * k2 + 1.781_477_9 * k3 - 1.821_255_9 * k4 + 1.330_274_5 * k4 * k;
    let pdf = (-0.5 * x * x).exp() / (2.0 * core::f32::consts::PI).sqrt();
    1.0 - pdf * poly
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    // Run in a thread with a large stack (the model forward pass + trajectory
    // extraction uses deep call stacks that overflow the default 8MB main stack).
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024) // 64MB
        .spawn(run_bench)
        .unwrap()
        .join()
        .unwrap();
}

fn run_bench() {
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    println!("Config: D_model={}, layers={}", config.hidden_size, config.num_layers);

    // Locate model.safetensors (same as bench_012-018).
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: this experiment requires real model.safetensors at {model_path}");
        std::process::exit(1);
    }

    print!("Loading real model.safetensors ... ");
    let t0 = std::time::Instant::now();
    let weights_a = load_kimi_k3(&model_path).unwrap_or_else(|e| {
        eprintln!("\n  load failed: {e}");
        std::process::exit(1);
    });
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());
    println!();

    // ── Generate prompt token sequences ───────────────────────────────────
    // Each prompt is a deterministic sequence of PROC_LEN token IDs.
    // The first PREFIX_LEN tokens are the prefix; the rest would be the
    // "processing" comparison.
    let prompts: Vec<Vec<u32>> = (0..N_PROMPTS)
        .map(|p| {
            (0..PROC_LEN)
                .map(|i| {
                    ((p as u32).wrapping_mul(31)
                        .wrapping_add((i as u32).wrapping_mul(7))
                        .wrapping_add(3))
                        % (BENCH_VOCAB as u32)
                })
                .collect()
        })
        .collect();

    // ── Shared setup ──────────────────────────────────────────────────────
    let max_seq_len = PROC_LEN;
    let mut runtime_a = KimiK3Runtime::new(&config, max_seq_len);
    let mut runtime_b = KimiK3Runtime::new(&config, max_seq_len);
    let encoder = StateMagnitudeEncoder::new();
    let mut gen_states: Vec<Vec<f32>> = Vec::new();
    let mut proc_states: Vec<Vec<f32>> = Vec::new();
    let mut depth_traj: Vec<Vec<f32>> = Vec::new();

    // ── Cache Model A trajectories ────────────────────────────────────────
    println!(
        "Extracting Model A trajectories ({N_PROMPTS} prompts: {PREFIX_LEN}-tok prefix + {N_GEN}-tok generation) ..."
    );
    let t0 = std::time::Instant::now();

    // Generation trajectories.
    let mut gen_a: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
    let mut gen_tokens_a: Vec<Vec<u32>> = Vec::with_capacity(N_PROMPTS);
    for prompt in &prompts {
        let toks = extract_generation_trajectory(
            &config,
            &weights_a,
            &mut runtime_a,
            &prompt[..PREFIX_LEN],
            N_GEN,
            &mut gen_states,
            &mut depth_traj,
        );
        gen_tokens_a.push(toks);
        gen_a.push(gen_states.clone());
    }

    // Processing trajectories (bench_018 method, for comparison).
    let mut proc_a: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
    for prompt in &prompts {
        extract_processing_trajectory(
            &config,
            &weights_a,
            &mut runtime_a,
            prompt,
            &mut proc_states,
            &mut depth_traj,
        );
        proc_a.push(proc_states.clone());
    }
    println!("  done ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── Run the sweep ─────────────────────────────────────────────────────
    let mut all_results: Vec<TrajResult> = Vec::new();

    for &sigma in SIGMA_LEVELS {
        let mut weights_b = weights_a.clone();
        perturb_model(&mut weights_b, sigma);

        print!(
            "Extracting Model B trajectories (σ={sigma}) ... "
        );
        let t0 = std::time::Instant::now();

        // Generation trajectories for Model B.
        // IMPORTANT: Model B generates from ITS OWN argmax (different tokens
        // than Model A if σ > 0). This is the realistic scenario.
        let mut gen_b: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
        for prompt in prompts.iter() {
            let _toks_b = extract_generation_trajectory(
                &config,
                &weights_b,
                &mut runtime_b,
                &prompt[..PREFIX_LEN],
                N_GEN,
                &mut gen_states,
                &mut depth_traj,
            );
            gen_b.push(gen_states.clone());
        }

        // Processing trajectories for Model B (same tokens as Model A).
        let mut proc_b: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
        for prompt in &prompts {
            extract_processing_trajectory(
                &config,
                &weights_b,
                &mut runtime_b,
                prompt,
                &mut proc_states,
                &mut depth_traj,
            );
            proc_b.push(proc_states.clone());
        }
        println!("done ({:.1}s)", t0.elapsed().as_secs_f64());

        // Encode + classify.
        let mut gen_summaries = [[[0.0_f32; D_ENC]; N_PROMPTS]; N_CLASSES];
        let mut proc_summaries = [[[0.0_f32; D_ENC]; N_PROMPTS]; N_CLASSES];

        for i in 0..N_PROMPTS {
            encode_state_stats(&gen_a[i], &encoder, &mut gen_summaries[0][i]);
            encode_state_stats(&gen_b[i], &encoder, &mut gen_summaries[1][i]);
            encode_state_stats(&proc_a[i], &encoder, &mut proc_summaries[0][i]);
            encode_state_stats(&proc_b[i], &encoder, &mut proc_summaries[1][i]);
        }

        let gen_result = evaluate(&gen_summaries, sigma, "generation");
        let proc_result = evaluate(&proc_summaries, sigma, "processing");

        all_results.push(gen_result);
        all_results.push(proc_result);
    }

    // ── Print results ─────────────────────────────────────────────────────
    println!();
    println!("═══ Results: Generation vs Processing Trajectory ═══");
    println!();
    println!(
        "  {:>6}  {:>12}  {:>9}  {:>9}  {:>9}",
        "σ", "regime", "Euclidean", "d_Euclid", "BayesOpt"
    );
    println!("  {}", "-".repeat(60));

    for r in &all_results {
        println!(
            "  {:>6.2}  {:>12}  {:>8.1}%  {:>9.3}  {:>8.1}%",
            r.sigma, r.regime, r.euclidean_acc * 100.0, r.d_euclid, r.bayes_optimal * 100.0
        );
    }

    // ── Summary ───────────────────────────────────────────────────────────
    println!();
    println!("═══ Summary ═══");
    println!();
    println!("Bench 018 (processing, bench-local encoder): 100% at σ≥0.1, d_M=14.526");
    println!();
    println!("This bench tests the GENERATION trajectory (model writes tokens via");
    println!("greedy argmax) vs the PROCESSING trajectory (model reads fixed tokens).");
    println!("Both use the substrate StateMagnitudeEncoder (bench_019).");
    println!();

    // Check if generation works as well as processing.
    let gen_at_01 = all_results
        .iter()
        .find(|r| r.sigma == 0.1 && r.regime == "generation").map_or(0.0, |r| r.euclidean_acc);
    let proc_at_01 = all_results
        .iter()
        .find(|r| r.sigma == 0.1 && r.regime == "processing").map_or(0.0, |r| r.euclidean_acc);

    println!("At σ=0.1: generation={:.1}% vs processing={:.1}%", gen_at_01 * 100.0, proc_at_01 * 100.0);
    if gen_at_01 >= 0.8 {
        println!();
        println!("VERDICT: POSITIVE — generation trajectory discriminates at ≥80%.");
        println!("The substrate is validated for the full SWE-bench use case (patch generation).");
    } else if gen_at_01 > proc_at_01 * 0.7 {
        println!();
        println!("VERDICT: PARTIAL — generation trajectory discriminates but weaker than processing.");
    } else {
        println!();
        println!("VERDICT: NEGATIVE — generation trajectory does NOT discriminate well.");
        println!("The substrate may need a different encoder for generation trajectories.");
    }
}
