//! Transformer-level FFN Mixture-of-Experts (MoE) — DeepSeek-V3 §3.3.
//!
//! Implements the auxiliary-loss-free load balancing from DeepSeek-V3
//! (`noaux_tc` top-K method) with the Kimi-K3 sigmoid router + SiTU expert
//! activation + latent MoE wrapper. See Research 328 for the routing
//! distillation, Research 330 §3 for the actual-model-code divergence
//! analysis, and Proposal 032 for the architectural context.
//!
//! # The mechanism (single-token decode)
//!
//! Matches the actual `modeling_kimi_k3_linear.py` forward path (Research 330
//! §3). The routed experts operate on a LATENT dimension
//! (`routed_expert_hidden_size`, 512) wrapped by down/up projections, and use
//! SiTU activation (not SwiGLU). The shared expert operates on the full hidden
//! dimension (1024).
//!
//! ```text
//! 1. Router affinity (sigmoid — independent per expert, NEVER softmax):
//!    logits[e] = dot(e_e, h)         for each routed expert e ∈ [0, N_r)
//!    s[e]      = sigmoid(logits[e])  ∈ (0, 1)
//!
//! 2. noaux_tc bias + top-K selection (THE LOAD-BEARING DETAIL):
//!    biased[e] = s[e] + b_e           (b_e = e_score_correction_bias[e])
//!    topk_idx  = argtopK(biased, K_r) (the K experts with highest BIASED score)
//!    topk_s    = s[topk_idx]          ← RAW s values (NOT biased!)
//!
//! 3. Renormalization (uses RAW scores — bias does NOT participate):
//!    g[k]      = topk_s[k] / sum(topk_s)
//!
//! 4. Latent MoE down-projection (routed experts only):
//!    h_latent  = routed_expert_down_proj · h    [d_moe]  (1024→512)
//!
//! 5. Routed experts (weighted by renormalized g, operating on h_latent):
//!    y_latent  = Σ_k g[k] * situ_ffn(expert[topk_idx[k]], h_latent)
//!
//! 6. Latent MoE norm + up-projection:
//!    if latent_moe_use_norm: y_latent = rmsnorm(y_latent, norm_weight, eps)
//!    y_routed  = routed_expert_up_proj · y_latent    [d]  (512→1024)
//!
//! 7. Shared expert (always on, no gating, operating on full hidden):
//!    out       = y_routed + situ_ffn(shared_weights, h)
//!
//! 8. Residual add is CALLER's responsibility (this fn writes the FFN output
//!    into `hidden_out`; the caller adds it to the residual stream).
//! ```
//!
//! # Why the bias is selection-only (Research 328 §3.3)
//!
//! DeepSeek-V3's design choice is deliberate: bias affects WHICH experts fire
//! (load balancing) but NOT how much each fires (signal strength). If the bias
//! leaked into renormalization, the model would conflate "picked for load
//! balancing" with "confident about this token" — degrading quality. The G1
//! test `g1_bias_does_not_leak_into_renormalization` enforces this bit-identically.
//!
//! # Latent MoE + SiTU (Research 330 §3)
//!
//! The original Phase 3 implementation used SwiGLU experts on the full hidden
//! dim. The actual model uses SiTU experts on a latent dim (512) wrapped by
//! down/up projections. This revision corrects the divergence.

use katgpt_core::sigmoid;
use katgpt_core::simd::simd_matmul_rows;
use katgpt_core::types::math::rmsnorm_with_gamma_eps;

// ─── Config ─────────────────────────────────────────────────────────────────

/// MoE configuration parameters.
///
/// Mirrors the Kimi-K3 / DeepSeek-V3 config fields. See Research 328 §4 for the
/// 0.40B-specific values.
#[derive(Clone, Debug)]
pub struct MoeConfig {
    /// Number of routed experts (`N_r`). Kimi-K3-0.40B: 8.
    pub num_experts: usize,
    /// Number of shared experts (`N_s`), always on. Kimi-K3-0.40B: 1.
    pub num_shared_experts: usize,
    /// Top-K routed experts per token (`K_r`). Kimi-K3-0.40B: 2.
    pub num_experts_per_token: usize,
    /// Expert FFN intermediate dim (`d_ffn`). Actual `config.json`: **256**.
    /// Loaded from `moe_intermediate_size`.
    pub moe_intermediate_size: usize,
    /// Hidden dim (`d`). Kimi-K3-0.40B: 1024.
    pub hidden_size: usize,
    /// When true (Kimi-K3), the router uses sigmoid (per-expert independent);
    /// when false (DeepSeek-V3 paper default), softmax over experts.
    /// This impl assumes sigmoid — the AGENTS.md global rule forbids softmax.
    pub use_sigmoid_router: bool,
    /// Renormalize selected experts' raw scores to sum to 1. Kimi-K3: true.
    pub renormalize: bool,
    /// Latent MoE dim (`routed_expert_hidden_size`). When `Some`, routed
    /// experts operate on this dim (wrapped by down/up projections). Actual
    /// `config.json`: **512**. When `None`, routed experts operate on
    /// `hidden_size` directly (no latent wrapper).
    pub routed_expert_hidden_size: Option<usize>,
    /// Whether to apply RMSNorm to the accumulated routed-expert output before
    /// up-projection (actual model: `latent_moe_use_norm`). Default: true.
    pub latent_moe_use_norm: bool,
    /// RMSNorm epsilon. Actual `config.json`: **1e-5**.
    pub rms_norm_eps: f32,
    /// SiTU activation beta (`activation_situ_beta`). Actual `config.json`:
    /// **4.0**.
    pub situ_beta: f32,
    /// SiTU activation linear_beta (`activation_situ_linear_beta`). Actual
    /// `config.json`: **25.0**.
    pub situ_linear_beta: Option<f32>,
}

impl MoeConfig {
    /// Kimi-K3-0.40B MoE configuration.
    ///
    /// Values match the actual `config.json` (Research 330 §7):
    /// - `moe_intermediate_size`: 256 (was placeholder 1024)
    /// - `routed_expert_hidden_size`: 512 (latent MoE)
    /// - `latent_moe_use_norm`: true
    /// - `situ_beta`: 4.0, `situ_linear_beta`: 25.0
    pub fn kimi_k3_0_40b() -> Self {
        Self {
            num_experts: 8,
            num_shared_experts: 1,
            num_experts_per_token: 2,
            moe_intermediate_size: 256,
            hidden_size: 1024,
            use_sigmoid_router: true,
            renormalize: true,
            routed_expert_hidden_size: Some(512),
            latent_moe_use_norm: true,
            rms_norm_eps: 1e-5,
            situ_beta: 4.0,
            situ_linear_beta: Some(25.0),
        }
    }

    /// `N_r` — number of routed experts.
    #[inline]
    pub fn n_routed(&self) -> usize {
        self.num_experts
    }

    /// `K_r` — top-K experts per token.
    #[inline]
    pub fn k_routed(&self) -> usize {
        self.num_experts_per_token
    }

    /// `d_ffn` — expert FFN intermediate dim.
    #[inline]
    pub fn d_ffn(&self) -> usize {
        self.moe_intermediate_size
    }

    /// `d` — hidden dim.
    #[inline]
    pub fn d(&self) -> usize {
        self.hidden_size
    }

    /// `d_moe` — latent MoE dim (routed expert input dim). Falls back to
    /// `hidden_size` when `routed_expert_hidden_size` is `None` (no latent MoE).
    #[inline]
    pub fn d_moe(&self) -> usize {
        self.routed_expert_hidden_size.unwrap_or(self.hidden_size)
    }

    /// Shared expert FFN intermediate dim: `moe_intermediate_size * num_shared_experts`.
    #[inline]
    pub fn d_ffn_shared(&self) -> usize {
        self.moe_intermediate_size * self.num_shared_experts
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

/// Per-expert FFN weights (gate + up + down).
///
/// Layouts (all row-major `Vec<f32>`):
/// - `gate_proj`: `[d_ffn, d_in]` — `out[o] = dot(row_o, hidden)`
/// - `up_proj`:   `[d_ffn, d_in]`
/// - `down_proj`: `[d_in, d_ffn]` — projects the activation output back to input dim
///
/// For routed experts: `d_in = routed_expert_hidden_size` (512), `d_ffn = moe_intermediate_size` (256).
/// For shared experts: `d_in = hidden_size` (1024), `d_ffn = moe_intermediate_size * num_shared_experts` (256).
///
/// Despite the name "SwiGlu", the activation is SiTU for Kimi-K3. The weight
/// layout is structurally identical; only the activation function differs.
#[derive(Clone, Debug)]
pub struct SwiGluExpertWeights {
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

/// MoE layer weight matrices.
///
/// Naming follows DeepSeek-V3 / Kimi-K3 conventions. See Research 328 §6 for
/// the safetensors tensor-name mapping (Phase 5 loader concern).
#[derive(Clone, Debug)]
pub struct MoeWeights {
    /// Router centroid matrix `[N_r, d]` — row `e` is expert `e`'s centroid
    /// `e_e`. The router logit for expert `e` is `dot(e_e, h)`.
    pub router_weight: Vec<f32>, // [N_r, d]
    /// noaux_tc per-expert bias `b_e` `[N_r]`. Added to the sigmoid score for
    /// top-K SELECTION ONLY — does NOT participate in renormalization.
    /// (Research 328 §3.3 — the load-bearing detail.)
    pub e_score_correction_bias: Vec<f32>, // [N_r]
    /// Routed expert FFN weights, length `N_r`.
    /// Operate on `routed_expert_hidden_size` (latent MoE dim) when present.
    pub experts: Vec<SwiGluExpertWeights>, // [N_r]
    /// Shared expert FFN weights, length `N_s` (always on, no gating).
    /// Operate on `hidden_size` (full dim).
    pub shared_experts: Vec<SwiGluExpertWeights>, // [N_s]
    /// Latent MoE down-projection `[d_moe, d]` (1024→512). Only present when
    /// `routed_expert_hidden_size` is `Some`. Actual model: `routed_expert_down_proj`.
    pub routed_expert_down_proj: Option<Vec<f32>>,
    /// Latent MoE up-projection `[d, d_moe]` (512→1024). Only present when
    /// `routed_expert_hidden_size` is `Some`. Actual model: `routed_expert_up_proj`.
    pub routed_expert_up_proj: Option<Vec<f32>>,
    /// Latent MoE RMSNorm gamma `[d_moe]`. Only present when
    /// `latent_moe_use_norm` is true. Actual model: `routed_expert_norm`.
    pub routed_expert_norm_weight: Option<Vec<f32>>,
}

impl MoeWeights {
    /// Test-only constructor: fills all weights with deterministic pseudo-
    /// random values from `seed`. Mirrors `MlaWeights::random` in katgpt-attn.
    ///
    /// The RNG is a simple xorshift — not cryptographic, just deterministic
    /// for reproducible G1 tests.
    pub fn random(config: &MoeConfig, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let d = config.d();
        let d_ffn = config.d_ffn();
        let n_r = config.n_routed();
        let n_s = config.num_shared_experts;
        let d_moe = config.d_moe();
        let d_ffn_shared = config.d_ffn_shared();

        let router_weight = (0..n_r * d).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        // Bias range: small ±0.5 — large enough to flip top-K selection in tests.
        let e_score_correction_bias =
            (0..n_r).map(|_| rng.next_f32() * 1.0 - 0.5).collect();
        // Routed experts operate on d_moe (latent dim when latent MoE is active)
        let experts = (0..n_r)
            .map(|_| SwiGluExpertWeights::random(&mut rng, d_moe, d_ffn))
            .collect();
        // Shared experts operate on full hidden dim, with expanded FFN
        let shared_experts = (0..n_s)
            .map(|_| SwiGluExpertWeights::random(&mut rng, d, d_ffn_shared))
            .collect();

        // Latent MoE wrapper weights (only when routed_expert_hidden_size is Some)
        let (routed_expert_down_proj, routed_expert_up_proj, routed_expert_norm_weight) =
            if config.routed_expert_hidden_size.is_some() {
                let down = (0..d_moe * d).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
                let up = (0..d * d_moe).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
                let norm = if config.latent_moe_use_norm {
                    Some((0..d_moe).map(|_| 1.0 + rng.next_f32() * 0.2 - 0.1).collect::<Vec<_>>())
                } else {
                    None
                };
                (Some(down), Some(up), norm)
            } else {
                (None, None, None)
            };

        Self {
            router_weight,
            e_score_correction_bias,
            experts,
            shared_experts,
            routed_expert_down_proj,
            routed_expert_up_proj,
            routed_expert_norm_weight,
        }
    }
}

impl SwiGluExpertWeights {
    /// Test-only random fill (same RNG as `MoeWeights::random`).
    pub fn random(rng: &mut Rng, d: usize, d_ffn: usize) -> Self {
        let gate_proj = (0..d_ffn * d).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        let up_proj = (0..d_ffn * d).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        let down_proj = (0..d * d_ffn).map(|_| rng.next_f32() * 0.4 - 0.2).collect();
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for `moe_forward_token`.
///
/// Allocated once at startup; reused across tokens. All hot-path state lives
/// here — `moe_forward_token` itself performs zero allocations (G4 gate).
pub struct MoeForwardScratch {
    /// Router logits `[N_r]` — `dot(e_e, h)` per expert.
    pub router_logits: Vec<f32>,
    /// Sigmoid scores `[N_r]` — `s[e] = sigmoid(logits[e])`.
    pub sigmoid_scores: Vec<f32>,
    /// Biased scores `[N_r]` — `s[e] + b_e` for top-K selection.
    pub biased_scores: Vec<f32>,
    /// Top-K selected expert indices `[K_r]`.
    pub topk_indices: Vec<usize>,
    /// Renormalized gating weights `[K_r]` (sum to 1 when `renormalize=true`).
    pub topk_weights: Vec<f32>,
    /// Expert FFN intermediate buffer `[max(d_ffn, d_ffn_shared)]` — gate proj.
    /// Reused across experts (one expert at a time).
    pub expert_intermediate: Vec<f32>,
    /// Expert FFN up-projection buffer `[max(d_ffn, d_ffn_shared)]`.
    pub expert_up: Vec<f32>,
    /// Expert output buffer `[max(d, d_moe)]` — single-expert output before
    /// weighted accumulation.
    pub expert_output: Vec<f32>,
    /// Latent MoE hidden buffer `[d_moe]` — down-projected input to routed experts.
    pub latent_hidden: Vec<f32>,
    /// Latent MoE output accumulator `[d_moe]` — sum of weighted expert outputs.
    pub latent_output: Vec<f32>,
}

impl MoeForwardScratch {
    /// Allocate scratch sized for `config`. Call once at startup.
    pub fn new(config: &MoeConfig) -> Self {
        let n_r = config.n_routed();
        let k_r = config.k_routed();
        let d = config.d();
        let d_ffn = config.d_ffn();
        let d_moe = config.d_moe();
        let d_ffn_shared = config.d_ffn_shared();
        // Expert scratch needs to fit both routed (d_ffn) and shared (d_ffn_shared)
        let max_d_ffn = d_ffn.max(d_ffn_shared);
        // Expert output needs to fit both routed (d_moe) and shared (d) dims
        let max_d_out = d.max(d_moe);
        Self {
            router_logits: vec![0.0; n_r],
            sigmoid_scores: vec![0.0; n_r],
            biased_scores: vec![0.0; n_r],
            topk_indices: vec![0; k_r],
            topk_weights: vec![0.0; k_r],
            expert_intermediate: vec![0.0; max_d_ffn],
            expert_up: vec![0.0; max_d_ffn],
            expert_output: vec![0.0; max_d_out],
            latent_hidden: vec![0.0; d_moe],
            latent_output: vec![0.0; d_moe],
        }
    }
}

// ─── Forward ────────────────────────────────────────────────────────────────

/// Single-token MoE decode forward.
///
/// Writes the FFN output (shared expert + weighted sum of top-K routed experts)
/// into `hidden_out`. The caller is responsible for the residual add
/// (`h' = h + hidden_out`) — this function does NOT add the residual.
///
/// # Arguments
/// * `weights` — MoE layer weights (router + experts + shared + bias)
/// * `config` — MoE configuration
/// * `hidden_in` — input hidden state `[d]`
/// * `hidden_out` — output buffer `[d]`, overwritten with the FFN output
/// * `scratch` — pre-allocated scratch (see `MoeForwardScratch`)
///
/// # Allocation discipline (G4)
///
/// Zero allocations in this function. All intermediate state lives in
/// `scratch`, which is allocated once at startup.
pub fn moe_forward_token(
    weights: &MoeWeights,
    config: &MoeConfig,
    hidden_in: &[f32],
    hidden_out: &mut [f32],
    scratch: &mut MoeForwardScratch,
) {
    let n_r = config.n_routed();
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();
    let d_moe = config.d_moe();
    let d_ffn_shared = config.d_ffn_shared();
    let use_latent_moe = config.routed_expert_hidden_size.is_some();

    debug_assert_eq!(hidden_in.len(), d);
    debug_assert_eq!(hidden_out.len(), d);
    debug_assert_eq!(weights.router_weight.len(), n_r * d);
    debug_assert_eq!(weights.e_score_correction_bias.len(), n_r);
    debug_assert_eq!(weights.experts.len(), n_r);
    debug_assert!(!weights.shared_experts.is_empty());
    debug_assert_eq!(scratch.router_logits.len(), n_r);
    debug_assert_eq!(scratch.sigmoid_scores.len(), n_r);
    debug_assert_eq!(scratch.biased_scores.len(), n_r);
    debug_assert_eq!(scratch.topk_indices.len(), k_r);
    debug_assert_eq!(scratch.topk_weights.len(), k_r);
    debug_assert!(scratch.expert_intermediate.len() >= d_ffn);
    debug_assert!(scratch.expert_intermediate.len() >= d_ffn_shared);
    debug_assert!(scratch.expert_up.len() >= d_ffn);
    debug_assert!(scratch.expert_up.len() >= d_ffn_shared);
    debug_assert!(scratch.expert_output.len() >= d);
    debug_assert!(scratch.expert_output.len() >= d_moe);
    debug_assert_eq!(scratch.latent_hidden.len(), d_moe);
    debug_assert_eq!(scratch.latent_output.len(), d_moe);

    // ── 1. Router logits: `logits[e] = dot(e_e, h)` per expert ───────────
    simd_matmul_rows(
        &mut scratch.router_logits[..n_r],
        &weights.router_weight,
        hidden_in,
        n_r,
        d,
    );

    // ── 2/3. Sigmoid scores (independent per expert, NEVER softmax) and the
    //    noaux_tc biased scores `biased[e] = s[e] + b_e`. Fused into one pass
    //    over the experts — the sigmoid result feeds the bias add directly
    //    instead of being written out and re-read.
    for e in 0..n_r {
        let s = sigmoid(scratch.router_logits[e]);
        scratch.sigmoid_scores[e] = s;
        scratch.biased_scores[e] = s + weights.e_score_correction_bias[e];
    }

    // ── 4. Top-K selection by BIASED score ───────────────────────────────
    select_topk_indices(
        &scratch.biased_scores[..n_r],
        k_r,
        &mut scratch.topk_indices[..k_r],
    );

    // ── 5. Renormalize selected experts' RAW scores (NOT biased!) ────────
    //    Per Research 328 §3.3: the bias is selection-only.
    let mut sum = 0.0f32;
    for k in 0..k_r {
        let idx = scratch.topk_indices[k];
        sum += scratch.sigmoid_scores[idx];
    }
    // Numerical guard (Research 328 §7.3): floor at f32::MIN_POSITIVE;
    // if absurdly small, fall back to uniform 1/K_r.
    if sum < 1.0e-20 {
        let uniform = 1.0 / (k_r as f32);
        for k in 0..k_r {
            scratch.topk_weights[k] = uniform;
        }
    } else if config.renormalize {
        let inv = 1.0 / sum;
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx] * inv;
        }
    } else {
        // moe_renormalize=false: use raw sigmoid scores as weights.
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx];
        }
    }

    // ── 6. Shared expert (always on, operating on full hidden dim) ───────
    //    Writes directly into hidden_out; routed experts axpy on top.
    let shared = &weights.shared_experts[0];
    situ_expert_forward(
        shared,
        hidden_in,
        &mut scratch.expert_intermediate,
        &mut scratch.expert_up,
        hidden_out,
        d,
        d_ffn_shared,
        config.situ_beta,
        config.situ_linear_beta,
    );
    // If N_s > 1, accumulate remaining shared experts (Kimi-K3-0.40B has N_s=1).
    for s in 1..weights.shared_experts.len() {
        let shared = &weights.shared_experts[s];
        situ_expert_forward(
            shared,
            hidden_in,
            &mut scratch.expert_intermediate,
            &mut scratch.expert_up,
            &mut scratch.expert_output,
            d,
            d_ffn_shared,
            config.situ_beta,
            config.situ_linear_beta,
        );
        // hidden_out += expert_output
        for (ho, eo) in hidden_out.iter_mut().zip(scratch.expert_output.iter()).take(d) {
            *ho += *eo;
        }
    }

    // ── 7. Routed experts (weighted by renormalized g) ───────────────────
    if use_latent_moe {
        // Latent MoE path: down-project → experts on latent dim → norm → up-project
        // h_latent = routed_expert_down_proj · h   [d_moe]
        simd_matmul_rows(
            &mut scratch.latent_hidden,
            weights.routed_expert_down_proj.as_ref().unwrap(),
            hidden_in,
            d_moe,
            d,
        );

        // Accumulate weighted expert outputs into latent_output
        scratch.latent_output[..d_moe].fill(0.0);
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            let w = scratch.topk_weights[k];
            let expert = &weights.experts[idx];
            situ_expert_forward(
                expert,
                &scratch.latent_hidden,
                &mut scratch.expert_intermediate,
                &mut scratch.expert_up,
                &mut scratch.expert_output,
                d_moe,
                d_ffn,
                config.situ_beta,
                config.situ_linear_beta,
            );
            // latent_output += w * expert_output
            for (lo, eo) in scratch.latent_output
                .iter_mut()
                .zip(scratch.expert_output.iter())
                .take(d_moe)
            {
                *lo += w * *eo;
            }
        }

        // Optional norm on accumulated latent output
        if config.latent_moe_use_norm
            && let Some(ref norm_w) = weights.routed_expert_norm_weight
        {
            rmsnorm_with_gamma_eps(
                &mut scratch.latent_output,
                norm_w,
                config.rms_norm_eps as f64,
            );
        }

        // Up-project: hidden_out += routed_expert_up_proj · latent_output   [d]
        simd_matmul_rows(
            &mut scratch.expert_output[..d],
            weights.routed_expert_up_proj.as_ref().unwrap(),
            &scratch.latent_output,
            d,
            d_moe,
        );
        for (ho, eo) in hidden_out.iter_mut().zip(scratch.expert_output.iter()).take(d) {
            *ho += *eo;
        }
    } else {
        // Non-latent path: routed experts operate directly on hidden dim
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            let w = scratch.topk_weights[k];
            let expert = &weights.experts[idx];
            situ_expert_forward(
                expert,
                hidden_in,
                &mut scratch.expert_intermediate,
                &mut scratch.expert_up,
                &mut scratch.expert_output,
                d,
                d_ffn,
                config.situ_beta,
                config.situ_linear_beta,
            );
            // hidden_out += w * expert_output
            for (ho, eo) in hidden_out.iter_mut().zip(scratch.expert_output.iter()).take(d) {
                *ho += w * *eo;
            }
        }
    }
}

/// Per-expert SiTU-gated FFN forward.
///
/// Computes `out = down_proj · SiTU(gate_proj · h, up_proj · h)` where SiTU is
/// the Kimi-K3 activation (Phase 1, confirmed correct in Research 330 §1):
/// ```text
/// situ_a = beta * tanh(gate / beta) * sigmoid(gate)
/// up_t   = linear_beta * tanh(up / linear_beta)   (when linear_beta is Some)
/// hidden = situ_a * up_t
/// ```
///
/// The gate projection lands in `intermediate` (caller-allocated scratch
/// `[d_ffn]`); the up projection lands in `up_buf` (caller-allocated scratch
/// `[d_ffn]`); the SiTU result overwrites `intermediate` in-place (safe — each
/// element is read+written independently); the down projection lands in `out`
/// (caller-allocated `[d_in]`).
///
/// Uses a local `situ_inplace` variant (katgpt-types' `situ` can't alias
/// hidden == gate due to borrow checker). Allocation-free.
#[inline]
#[allow(clippy::too_many_arguments)]
fn situ_expert_forward(
    expert: &SwiGluExpertWeights,
    hidden_in: &[f32],
    intermediate: &mut [f32],
    up_buf: &mut [f32],
    out: &mut [f32],
    d_in: usize,
    d_ffn: usize,
    beta: f32,
    linear_beta: Option<f32>,
) {
    debug_assert_eq!(expert.gate_proj.len(), d_ffn * d_in);
    debug_assert_eq!(expert.up_proj.len(), d_ffn * d_in);
    debug_assert_eq!(expert.down_proj.len(), d_in * d_ffn);
    debug_assert!(intermediate.len() >= d_ffn);
    debug_assert!(up_buf.len() >= d_ffn);
    debug_assert!(out.len() >= d_in);

    let intermediate = &mut intermediate[..d_ffn];
    let up_buf = &mut up_buf[..d_ffn];
    let out = &mut out[..d_in];

    // gate_proj · h → intermediate
    simd_matmul_rows(intermediate, &expert.gate_proj, hidden_in, d_ffn, d_in);

    // up_proj · h → up_buf
    simd_matmul_rows(up_buf, &expert.up_proj, hidden_in, d_ffn, d_in);

    // intermediate = SiTU(gate=intermediate, up=up_buf) — in-place variant
    // (can't use `situ(intermediate, intermediate, up_buf, ...)` due to borrow
    // checker; the computation is element-wise safe)
    situ_inplace(intermediate, up_buf, beta, linear_beta);

    // down_proj · intermediate → out
    simd_matmul_rows(out, &expert.down_proj, intermediate, d_in, d_ffn);
}

/// In-place SiTU: `gate[i]` is read + overwritten element-wise. Equivalent to
/// `situ(gate, gate, up, beta, linear_beta)` but avoids the aliasing borrow.
#[inline]
pub(crate) fn situ_inplace(gate: &mut [f32], up: &[f32], beta: f32, linear_beta: Option<f32>) {
    debug_assert!(beta > 0.0, "situ beta must be positive");
    let inv_beta = 1.0 / beta;
    // Slice `up` to `gate.len()` once (same panic-on-short-`up` behaviour as
    // the old `up[i]` indexing) so the element loops zip without per-element
    // bounds checks.
    let up = &up[..gate.len()];
    if let Some(lb) = linear_beta {
        debug_assert!(lb > 0.0, "situ linear_beta must be positive");
        let inv_lb = 1.0 / lb;
        for (g_slot, &u) in gate.iter_mut().zip(up.iter()) {
            let g = *g_slot;
            let gate_sigmoid = 1.0 / (1.0 + (-g).exp());
            let gate_tanh = (g * inv_beta).tanh();
            let up_t = lb * (u * inv_lb).tanh();
            *g_slot = beta * gate_tanh * gate_sigmoid * up_t;
        }
    } else {
        for (g_slot, &u) in gate.iter_mut().zip(up.iter()) {
            let g = *g_slot;
            let gate_sigmoid = 1.0 / (1.0 + (-g).exp());
            let gate_tanh = (g * inv_beta).tanh();
            *g_slot = beta * gate_tanh * gate_sigmoid * u;
        }
    }
}

// ─── Top-K selection ────────────────────────────────────────────────────────

/// Partial-selection top-K: picks the indices of the K largest values in
/// `scores`, in DESCENDING order of score.
///
/// Writes the selected indices into `out_idx[..k]`. Alloc-free.
///
/// Implementation: insertion-sort the first K elements, then scan the rest
/// replacing the current minimum when a larger value is found. O(n·k) — for
/// Kimi-K3-0.40B (n=8, k=2) this is 16 comparisons, faster than a full sort.
/// For larger n the `katgpt-attn::dash_attn::block_topk::argtopk_with_scratch`
/// SIMD primitive would be preferred, but the dep isn't worth pulling for
/// this small-n case.
pub(crate) fn select_topk_indices(scores: &[f32], k: usize, out_idx: &mut [usize]) {
    debug_assert!(k <= scores.len());
    debug_assert_eq!(out_idx.len(), k);

    if k == 0 {
        return;
    }

    // Seed with the first k indices, sorted descending by score.
    out_idx[0] = 0;
    for i in 1..k {
        let idx = i;
        let val = scores[idx];
        // Insertion-sort idx into out_idx[0..i] (which is sorted desc).
        let mut j = i;
        while j > 0 && scores[out_idx[j - 1]] < val {
            out_idx[j] = out_idx[j - 1];
            j -= 1;
        }
        out_idx[j] = idx;
    }

    // Scan the rest; replace the current minimum (last slot) when larger.
    for i in k..scores.len() {
        let val = scores[i];
        // The minimum of the current top-K is at out_idx[k-1] (descending sort).
        if val > scores[out_idx[k - 1]] {
            // Replace the minimum + bubble it up to its sorted position.
            out_idx[k - 1] = i;
            let mut j = k - 1;
            while j > 0 && scores[out_idx[j - 1]] < scores[out_idx[j]] {
                out_idx.swap(j - 1, j);
                j -= 1;
            }
        }
    }
}

// ─── Test RNG (deterministic xorshift) ──────────────────────────────────────

/// Deterministic xorshift RNG for G1 tests. Mirrors `MlaWeights::random`.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the degenerate all-zero state.
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64 — deterministic, fast, good enough for G1 test inputs.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f32 in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for the mantissa (f32 has 23 explicit bits + implicit 1).
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny config for fast tests: 4 experts, 1 shared, K=2, d=8, d_ffn=16.
    /// Uses latent MoE (d_moe=6) to exercise the down/up wrapper path.
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
    fn test_select_topk_descending_order() {
        let scores = [0.5, 0.9, 0.1, 0.7, 0.3];
        let mut idx = [0usize; 3];
        select_topk_indices(&scores, 3, &mut idx);
        // Top-3 by score: 0.9 (idx 1), 0.7 (idx 3), 0.5 (idx 0).
        assert_eq!(idx, [1, 3, 0]);
    }

    #[test]
    fn test_select_topk_handles_ties() {
        // Ties resolved by insertion order (stable).
        let scores = [0.5, 0.5, 0.5, 0.5];
        let mut idx = [0usize; 2];
        select_topk_indices(&scores, 2, &mut idx);
        // First two slots win on ties (>= comparison keeps earlier idx).
        assert_eq!(idx, [0, 1]);
    }

    #[test]
    fn test_moe_forward_runs_without_panic() {
        let config = tiny_config();
        let weights = MoeWeights::random(&config, 42);
        let mut scratch = MoeForwardScratch::new(&config);
        let hidden_in = vec![0.1; config.d()];
        let mut hidden_out = vec![0.0; config.d()];
        moe_forward_token(&weights, &config, &hidden_in, &mut hidden_out, &mut scratch);
        // Sanity: output is finite + non-zero (SiTU of random weights is non-zero).
        assert!(hidden_out.iter().all(|v| v.is_finite()));
        assert!(hidden_out.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn test_moe_forward_kimi_k3_dims() {
        // Smoke test: full Kimi-K3-0.40B dims run without panic.
        let config = MoeConfig::kimi_k3_0_40b();
        let weights = MoeWeights::random(&config, 123);
        let mut scratch = MoeForwardScratch::new(&config);
        let hidden_in = vec![0.05; config.d()];
        let mut hidden_out = vec![0.0; config.d()];
        moe_forward_token(&weights, &config, &hidden_in, &mut hidden_out, &mut scratch);
        assert!(hidden_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_renormalization_weights_sum_to_one() {
        let config = tiny_config();
        let mut weights = MoeWeights::random(&config, 7);
        // Zero out shared expert so it doesn't influence the weight check.
        for se in &mut weights.shared_experts {
            se.gate_proj.fill(0.0);
            se.up_proj.fill(0.0);
            se.down_proj.fill(0.0);
        }
        let mut scratch = MoeForwardScratch::new(&config);
        let hidden_in = vec![0.2; config.d()];
        let mut hidden_out = vec![0.0; config.d()];
        moe_forward_token(&weights, &config, &hidden_in, &mut hidden_out, &mut scratch);
        // After forward, scratch.topk_weights holds the renormalized g values.
        let sum: f32 = scratch.topk_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "renormalized weights must sum to 1, got {sum}"
        );
    }
}
