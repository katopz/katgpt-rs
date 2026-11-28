//! KDA (Kimi Delta Attention) forward layer.
//!
//! Implements the per-token decode forward for Kimi Linear's KDA linear attention
//! variant (arxiv 2510.26692). Used in 6 of 8 layers of Kimi-K3-0.40B (layers
//! 1, 2, 3, 5, 6, 7); layers 0 + 4 use MLA (Phase 2).
//!
//! # The mechanism (single-token decode, revised Research 330 §4)
//!
//! Mirrors the actual `KimiDeltaAttention` class from
//! `modeling_kimi_k3_linear.py` + the `fla.ops.kda.fused_recurrent_kda` Triton
//! kernel (decoded from the fla-org/flash-linear-attention repo).
//!
//! ```text
//! 1. Separate q/k/v projections (NOT shared):
//!    z_q = W^q · h      [head_dim * n_heads]
//!    z_k = W^k · h      [head_dim * n_heads]
//!    z_v = W^v · h      [head_dim * n_heads]
//!
//! 2. Separate ShortConv + SiLU for q, k, v (NOT shared):
//!    z_q_conv = SiLU(ShortConv1D(z_q))
//!    z_k_conv = SiLU(ShortConv1D(z_k))
//!    z_v_conv = SiLU(ShortConv1D(z_v))
//!
//! 3. Gate into kernel (f_a + f_b — low-rank):
//!    g = W^{f_b} · (W^{f_a} · h)   [head_dim * n_heads]
//!    β = W^β · h                    [n_heads]  (pre-sigmoid)
//!
//! 4. Per-head recurrent step (the fla.ops.kda kernel):
//!    L2Norm q, k inside the step; q *= scale (1/sqrt(head_dim)).
//!    Decay gate (per-key-channel, using dt_bias):
//!      gk[k] = -exp(A_log[h]) * softplus(g[k] + dt_bias[k])
//!    S' = S · diag(exp(gk))                    (decay each key channel)
//!    v' = sigmoid(β) · (v − S'·k)              (erase + scale)
//!    S = S' + v' · k^T                         (write)
//!    o = S · q                                  (readout — uses updated S)
//!
//! 5. Output gate (FULL rank) + FusedRMSNormGated:
//!    g_out = W^g · h              [head_dim * n_heads]
//!    o_norm_h = (RMSNorm(o_h, γ, eps)) * sigmoid(g_out_h)
//!
//! 6. Final projection:
//!    output = W_o · concat(o_norm_0, ..., o_norm_{H-1})
//! ```
//!
//! # gdn2 kernel equivalence (proven)
//!
//! The gdn2 kernel's `Kda` gate config computes the same recurrence as the
//! Triton `fused_recurrent_kda` kernel when called with:
//! - `alpha[k] = exp(gk[k])` — per-key-channel decay (gdn2 decays rows; the
//!   Triton kernel with `state_v_first=True` decays columns; these are
//!   transpose-dual and produce identical results)
//! - `b = sigmoid(β)·ones` — erase gate (β-broadcast)
//! - `w_val = sigmoid(β)` — write weight
//!
//! See `gdn2_recurrent_step` in `kernel.rs` for the implementation. The math:
//! - gdn2: `r = S'ᵀ·(b⊙k) = sigmoid(β)·S'ᵀ·k`; `delta = w_val·v − r = sigmoid(β)·(v − S'ᵀ·k)`
//! - Triton: `v' = sigmoid(β)·(v − S'·k)`
//!
//! These are identical (modulo the transposed state convention).

use crate::gdn2::kernel::gdn2_recurrent_step;
use crate::gdn2::short_conv::ShortConv1D;
use crate::gdn2::types::{Gdn2GateConfig, Gdn2HeadState};
use katgpt_core::simd::{simd_matmul_rows, simd_sum_sq};

// ─── Config ─────────────────────────────────────────────────────────────────

/// KDA configuration parameters.
///
/// Mirrors the Kimi-K3-0.40B `linear_attn_config` fields. See Research 330 §4
/// + §7 for the 0.40B-specific values.
#[derive(Clone, Debug)]
pub struct KdaConfig {
    /// Per-head key/query/value dim (`d_k = d_v`). Kimi-K3-0.40B: **32**
    /// (from `linear_attn_config.head_dim`, NOT 128).
    pub head_dim: usize,
    /// Number of attention heads (`n_h`). Kimi-K3-0.40B: 8.
    pub n_heads: usize,
    /// Hidden dim (`d`). Kimi-K3-0.40B: 1024.
    pub hidden_size: usize,
    /// ShortConv kernel size. Kimi-K3-0.40B: 4.
    pub conv_kernel_size: usize,
    /// Numerical floor for the decay `exp(gk)` (avoid denormals). Default 1e-5.
    pub alpha_eps: f32,
    /// RMSNorm epsilon for FusedRMSNormGated. Default 1e-5 (from config).
    pub rms_eps: f32,
}

impl KdaConfig {
    /// Kimi-K3-0.40B KDA configuration (text path, `kimi_linear` model type).
    ///
    /// Values from the actual `config.json` + `modeling_kimi_k3_linear.py`
    /// (Research 330 §7):
    /// - `head_dim = 32` (from `linear_attn_config.head_dim`)
    /// - `rms_norm_eps = 1e-5`
    pub fn kimi_k3_0_40b() -> Self {
        Self {
            head_dim: 32,
            n_heads: 8,
            hidden_size: 1024,
            conv_kernel_size: 4,
            alpha_eps: 1e-5,
            rms_eps: 1e-5,
        }
    }

    /// Total projection width: `head_dim * n_heads`.
    #[inline]
    pub fn proj_dim(&self) -> usize {
        self.head_dim * self.n_heads
    }

    /// Scale factor for q after L2Norm: `1 / sqrt(head_dim)`.
    #[inline]
    pub fn q_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

/// KDA layer weight matrices (row-major `Vec<f32>`).
///
/// Mirrors the `KimiDeltaAttention.__init__` weight layout from
/// `modeling_kimi_k3_linear.py` (Research 330 §4).
#[derive(Clone, Debug)]
pub struct KdaWeights {
    // ── Separate q/k/v projections (NOT shared) ──
    /// Query projection `W^q`. Shape `[proj_dim, hidden_size]`.
    pub q_proj: Vec<f32>,
    /// Key projection `W^k`. Shape `[proj_dim, hidden_size]`.
    pub k_proj: Vec<f32>,
    /// Value projection `W^v`. Shape `[proj_dim, hidden_size]`.
    pub v_proj: Vec<f32>,

    // ── Separate ShortConv depthwise filters ──
    /// ShortConv filter for q. Shape `[proj_dim, conv_kernel_size]`.
    pub q_conv_weight: Vec<f32>,
    /// ShortConv filter for k. Shape `[proj_dim, conv_kernel_size]`.
    pub k_conv_weight: Vec<f32>,
    /// ShortConv filter for v. Shape `[proj_dim, conv_kernel_size]`.
    pub v_conv_weight: Vec<f32>,

    // ── Per-head decay (A_log, log-space) ──
    /// Per-head A_log: `alpha = exp(A_log[h])`. Shape `[n_heads]`.
    /// Used in the decay gate: `gk = -alpha * softplus(g + dt_bias)`.
    pub a_log: Vec<f32>,

    // ── Gate into kernel (f_a + f_b — low-rank) ──
    /// Gate down-projection `W^{f_a}`. Shape `[head_dim, hidden_size]`.
    pub f_a_proj: Vec<f32>,
    /// Gate up-projection `W^{f_b}`. Shape `[proj_dim, head_dim]`.
    pub f_b_proj: Vec<f32>,

    // ── Per-channel dt_bias ──
    /// Bias added to the gate before softplus. Shape `[proj_dim]`.
    pub dt_bias: Vec<f32>,

    // ── Beta projection (per-head, pre-sigmoid) ──
    /// β projection `W^β`. Shape `[n_heads, hidden_size]`.
    pub beta_proj: Vec<f32>,

    // ── Full-rank output gate ──
    /// Output gate `W^g` (full-rank since `use_full_rank_gate=true`).
    /// Shape `[proj_dim, hidden_size]`.
    pub g_proj: Vec<f32>,

    // ── FusedRMSNormGated weight ──
    /// Output norm weight γ (shared across heads). Shape `[head_dim]`.
    pub o_norm_weight: Vec<f32>,

    // ── Output projection ──
    /// Output projection `W_o`. Shape `[hidden_size, proj_dim]`.
    pub o_proj: Vec<f32>,
}

impl KdaWeights {
    /// Construct random weights from a seeded RNG (for G1 testing).
    ///
    /// Uses a simple LCG to avoid pulling a new RNG crate dep. Weights are drawn
    /// from `[-1/sqrt(in_dim), 1/sqrt(in_dim)]` (Xavier-ish initialization for
    /// stable forward passes on random weights).
    pub fn random(config: &KdaConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let d = config.hidden_size;
        let dk = config.head_dim;
        let n_h = config.n_heads;
        let proj = dk * n_h;
        let ks = config.conv_kernel_size;

        // A_log: initialized uniform(1, 16) then log'd (mirrors the PyTorch init).
        let a_log: Vec<f32> = (0..n_h)
            .map(|_| {
                let u = rng.next_f32_pos(); // [0, 1)
                let val = 1.0 + u * 15.0; // [1, 16)
                val.ln() // log-space
            })
            .collect();

        // dt_bias: initialized near-zero (bias-like).
        let dt_bias: Vec<f32> = (0..proj).map(|_| rng.next_f32(0.1)).collect();

        Self {
            q_proj: random_matrix(&mut rng, proj, d),
            k_proj: random_matrix(&mut rng, proj, d),
            v_proj: random_matrix(&mut rng, proj, d),
            q_conv_weight: random_matrix(&mut rng, proj, ks),
            k_conv_weight: random_matrix(&mut rng, proj, ks),
            v_conv_weight: random_matrix(&mut rng, proj, ks),
            a_log,
            f_a_proj: random_matrix(&mut rng, dk, d),
            f_b_proj: random_matrix(&mut rng, proj, dk),
            dt_bias,
            beta_proj: random_matrix(&mut rng, n_h, d),
            g_proj: random_matrix(&mut rng, proj, d),
            o_norm_weight: random_matrix(&mut rng, dk, 1),
            o_proj: random_matrix(&mut rng, d, proj),
        }
    }
}

/// Tiny deterministic LCG RNG (mirrors the MLA pattern in `mla.rs`).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // xorshift64* — fast + deterministic.
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    fn next_f32_pos(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    /// Uniform in `[-scale, scale]`.
    #[inline]
    fn next_f32(&mut self, scale: f32) -> f32 {
        let u = self.next_u32();
        let signed = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
        signed * scale
    }
}

/// Allocate a `[rows, cols]` row-major matrix with Xavier-ish init.
fn random_matrix(rng: &mut SimpleRng, rows: usize, cols: usize) -> Vec<f32> {
    if cols == 0 {
        return Vec::new();
    }
    let scale = 1.0 / (cols as f32).sqrt();
    (0..rows * cols).map(|_| rng.next_f32(scale)).collect()
}

// ─── State cache ────────────────────────────────────────────────────────────

/// KDA recurrent state + conv ring buffers for a single layer.
///
/// The recurrent state matrix S per head lives in `Gdn2HeadState` (reused from
/// the gdn2 substrate). The ShortConv ring buffers are KDA-specific.
#[derive(Clone)]
pub struct KdaLayerCache {
    /// Per-head recurrent state matrices (head_dim × head_dim each).
    pub heads: Vec<Gdn2HeadState>,
    /// ShortConv for q (separate from k and v).
    pub q_conv: ShortConv1D,
    /// ShortConv for k (separate from q and v).
    pub k_conv: ShortConv1D,
    /// ShortConv for v (separate from q and k).
    pub v_conv: ShortConv1D,
}

impl KdaLayerCache {
    /// Allocate zeroed cache for one KDA layer.
    pub fn new(config: &KdaConfig) -> Self {
        let dk = config.head_dim;
        let dv = config.head_dim;
        let n_h = config.n_heads;
        let proj = dk * n_h;
        Self {
            heads: (0..n_h).map(|_| Gdn2HeadState::new(dk, dv)).collect(),
            q_conv: ShortConv1D::new(proj, config.conv_kernel_size),
            k_conv: ShortConv1D::new(proj, config.conv_kernel_size),
            v_conv: ShortConv1D::new(proj, config.conv_kernel_size),
        }
    }

    /// Reset state matrices + conv ring buffers to zeros (reuse allocations).
    pub fn reset(&mut self) {
        for h in &mut self.heads {
            h.reset();
        }
        self.q_conv.reset();
        self.k_conv.reset();
        self.v_conv.reset();
    }
}

/// Multi-layer KDA cache (for models with multiple KDA layers — Kimi-K3-0.40B has 6).
#[derive(Clone)]
pub struct KdaCache {
    /// Per-layer state + conv ring buffers.
    pub layers: Vec<KdaLayerCache>,
}

impl KdaCache {
    /// Allocate zeroed cache for `n_layers` KDA layers.
    pub fn new(config: &KdaConfig, n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| KdaLayerCache::new(config)).collect(),
        }
    }

    /// Reset all layers (reuse allocations).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

// ─── Forward scratch ────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for `kda_forward_token` (zero-alloc hot path).
///
/// All buffers are sized from the config at construction time; the forward path
/// only mutates them in place.
pub struct KdaForwardScratch {
    // ── Pre-conv projections (separate q/k/v) ──
    /// q projection output. Shape `[proj_dim]`.
    pub z_q: Vec<f32>,
    /// k projection output. Shape `[proj_dim]`.
    pub z_k: Vec<f32>,
    /// v projection output. Shape `[proj_dim]`.
    pub z_v: Vec<f32>,
    // ── Post-conv (separate, SiLU-activated) ──
    /// q post-conv. Shape `[proj_dim]`.
    pub z_q_conv: Vec<f32>,
    /// k post-conv. Shape `[proj_dim]`.
    pub z_k_conv: Vec<f32>,
    /// v post-conv. Shape `[proj_dim]`.
    pub z_v_conv: Vec<f32>,
    // ── Gate into kernel (f_a + f_b) ──
    /// f_a intermediate. Shape `[head_dim]`.
    pub f_a_hidden: Vec<f32>,
    /// f_b output (the raw gate, before dt_bias). Shape `[proj_dim]`.
    pub g_raw: Vec<f32>,
    // ── Beta projection ──
    /// β projection pre-sigmoid. Shape `[n_heads]`.
    pub beta_pre: Vec<f32>,
    /// β per head (post-sigmoid). Shape `[n_heads]`.
    pub beta: Vec<f32>,
    // ── Per-head recurrent scratch ──
    /// Per-head β-broadcast erase buffer (= sigmoid(β) · ones). Shape `[head_dim]`.
    pub erase_b_h: Vec<f32>,
    /// Per-head decay alpha (= exp(gk)). Shape `[head_dim]`.
    pub alpha_h: Vec<f32>,
    /// Per-head output buffer (readout result, pre-norm). Shape `[head_dim]`.
    pub o_h: Vec<f32>,
    /// Per-head normed output. Shape `[head_dim]`.
    pub o_h_norm: Vec<f32>,
    // ── Full-rank output gate ──
    /// Output gate (g_proj · h, pre-sigmoid). Shape `[proj_dim]`.
    pub g_out: Vec<f32>,
    // ── Concatenated + final output ──
    /// Concatenated per-head outputs (post-norm + gated). Shape `[proj_dim]`.
    pub o_concat: Vec<f32>,
    /// Final output (post o_proj). Shape `[hidden_size]`.
    pub output: Vec<f32>,
    // ── gdn2_recurrent_step inner scratch (passed per call) ──
    /// gdn2 temp buffer. Shape `[head_dim]`.
    pub gdn2_temp: Vec<f32>,
    /// gdn2 delta buffer. Shape `[head_dim]`.
    pub gdn2_delta: Vec<f32>,
    /// gdn2 write_w_channel buffer (unused for Kda mode but required by signature).
    /// Shape `[head_dim]`, filled with 1.0.
    pub gdn2_write_w: Vec<f32>,
}

impl KdaForwardScratch {
    /// Allocate scratch for the given config.
    pub fn new(config: &KdaConfig) -> Self {
        let dk = config.head_dim;
        let n_h = config.n_heads;
        let d = config.hidden_size;
        let proj = dk * n_h;
        Self {
            z_q: vec![0.0; proj],
            z_k: vec![0.0; proj],
            z_v: vec![0.0; proj],
            z_q_conv: vec![0.0; proj],
            z_k_conv: vec![0.0; proj],
            z_v_conv: vec![0.0; proj],
            f_a_hidden: vec![0.0; dk],
            g_raw: vec![0.0; proj],
            beta_pre: vec![0.0; n_h],
            beta: vec![0.0; n_h],
            erase_b_h: vec![0.0; dk],
            alpha_h: vec![0.0; dk],
            o_h: vec![0.0; dk],
            o_h_norm: vec![0.0; dk],
            g_out: vec![0.0; proj],
            o_concat: vec![0.0; proj],
            output: vec![0.0; d],
            gdn2_temp: vec![0.0; dk],
            gdn2_delta: vec![0.0; dk],
            gdn2_write_w: vec![1.0; dk],
        }
    }
}

// ─── Forward kernel ─────────────────────────────────────────────────────────

/// Per-token KDA forward pass.
///
/// Updates the layer's recurrent state in-place and returns a slice into
/// `scratch.output` of length `config.hidden_size`.
///
/// # Arguments
/// * `config` — KDA configuration
/// * `weights` — KDA weight matrices
/// * `cache` — per-layer recurrent state + conv ring buffers
/// * `scratch` — pre-allocated scratch (reused across calls)
/// * `h` — input hidden state `[hidden_size]`
///
/// # Returns
/// A mutable slice into `scratch.output[..hidden_size]`.
/// The forget-gate form. The incumbent (kimi-k3) computes the log-decay as
/// `-alpha · softplus(g + dt_bias)`; GLM-5.3-Flash (Plan 627 T4a; the
/// transformers `linear_lower_bound` / `safe_gate_lower_bound` branch) computes
/// `bound · sigmoid(alpha · g)` with `bound = -5.0` — a log-decay bounded in
/// `[bound, 0]`, so the decay factor stays in `[exp(bound), 1]` and never
/// reaches the `alpha_eps` floor.
#[derive(Clone, Copy, Debug)]
enum KdaGate {
    Softplus,
    SigmoidLowerBound(f32),
}

pub fn kda_forward_token<'s>(
    config: &KdaConfig,
    weights: &KdaWeights,
    cache: &mut KdaLayerCache,
    scratch: &'s mut KdaForwardScratch,
    h: &[f32],
) -> &'s mut [f32] {
    kda_forward_token_inner(config, weights, cache, scratch, h, KdaGate::Softplus)
}

/// GLM-5.3-Flash's safe lower-bound forget gate (Plan 627 T4a; the
/// transformers `Glm5NextTextForgetGate` branch taken when
/// `linear_lower_bound` is set): `gk = bound · sigmoid(exp(A_log) · (f_b(f_a(h))
/// + dt_bias))` — identical projections and recurrence to
/// [`kda_forward_token`], only the log-decay formula differs. `bound` comes
/// from the model config (`linear_lower_bound`, measured -5.0); the caller
/// passes it because [`KdaConfig`] deliberately stays gate-form-free (its
/// struct literals are pinned across two repos).
pub fn kda_forward_token_lower_bound_gate<'s>(
    config: &KdaConfig,
    weights: &KdaWeights,
    cache: &mut KdaLayerCache,
    scratch: &'s mut KdaForwardScratch,
    h: &[f32],
    bound: f32,
) -> &'s mut [f32] {
    kda_forward_token_inner(
        config,
        weights,
        cache,
        scratch,
        h,
        KdaGate::SigmoidLowerBound(bound),
    )
}

fn kda_forward_token_inner<'s>(
    config: &KdaConfig,
    weights: &KdaWeights,
    cache: &mut KdaLayerCache,
    scratch: &'s mut KdaForwardScratch,
    h: &[f32],
    gate: KdaGate,
) -> &'s mut [f32] {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;
    let scale = config.q_scale();
    debug_assert_eq!(h.len(), d, "hidden state dim mismatch");

    // ── Step 1: Separate q/k/v projections ──────────────────────────────────
    simd_matmul_rows(&mut scratch.z_q, &weights.q_proj, h, proj, d);
    simd_matmul_rows(&mut scratch.z_k, &weights.k_proj, h, proj, d);
    simd_matmul_rows(&mut scratch.z_v, &weights.v_proj, h, proj, d);

    // Gate into kernel: f_a_hidden = W^{f_a} · h  [head_dim]
    simd_matmul_rows(&mut scratch.f_a_hidden, &weights.f_a_proj, h, dk, d);
    // g_raw = W^{f_b} · f_a_hidden  [proj_dim]
    simd_matmul_rows(
        &mut scratch.g_raw,
        &weights.f_b_proj,
        &scratch.f_a_hidden,
        proj,
        dk,
    );

    // β pre-sigmoid: beta_pre = W^β · h  [n_heads]
    simd_matmul_rows(&mut scratch.beta_pre, &weights.beta_proj, h, n_h, d);

    // Full-rank output gate: g_out = W^g · h  [proj_dim]
    simd_matmul_rows(&mut scratch.g_out, &weights.g_proj, h, proj, d);

    // ── Step 2: Separate ShortConv + SiLU ──────────────────────────────────
    // Override identity-init weights with real weights before forward.
    cache.q_conv.weight.copy_from_slice(&weights.q_conv_weight);
    cache.k_conv.weight.copy_from_slice(&weights.k_conv_weight);
    cache.v_conv.weight.copy_from_slice(&weights.v_conv_weight);
    cache.q_conv.forward(&scratch.z_q, &mut scratch.z_q_conv);
    cache.k_conv.forward(&scratch.z_k, &mut scratch.z_k_conv);
    cache.v_conv.forward(&scratch.z_v, &mut scratch.z_v_conv);

    // SiLU (swish) activation on all three conv outputs.
    silu_inplace(&mut scratch.z_q_conv);
    silu_inplace(&mut scratch.z_k_conv);
    silu_inplace(&mut scratch.z_v_conv);

    // β = sigmoid(beta_pre)
    for i in 0..n_h {
        scratch.beta[i] = sigmoid(scratch.beta_pre[i]);
    }

    // ── Step 3: Per-head recurrent step (fla.ops.kda kernel) ───────────────
    // The gdn2_recurrent_step with Kda gate config computes the same recurrence
    // as the Triton kernel (see module-level proof). We apply L2Norm + scale
    // on q/k OUTSIDE the gdn2 call (the Triton kernel does it inside, but the
    // math is identical since gdn2 doesn't touch q/k normalization).
    for head in 0..n_h {
        let off = head * dk;

        // q_h: L2Norm then scale by 1/sqrt(head_dim).
        // Copy into o_h (reused as q buffer — the gdn2 kernel reads q after
        // writing out, so we need a separate q slice).
        scratch
            .o_h
            .copy_from_slice(&scratch.z_q_conv[off..off + dk]);
        l2_normalize_eps_kda(&mut scratch.o_h);
        for v in scratch.o_h[..dk].iter_mut() {
            *v *= scale;
        }
        let q_h = &scratch.o_h;

        // k_h: L2Norm (no scale). Use alpha_h as scratch (it's written below
        // before the gdn2 call, but we need k for the gdn2 call — borrow order
        // matters).
        scratch
            .alpha_h
            .copy_from_slice(&scratch.z_k_conv[off..off + dk]);
        l2_normalize_eps_kda(&mut scratch.alpha_h);
        let k_h = &scratch.alpha_h;

        // v_h = z_v_conv[h slice] (no norm).
        let v_h = &scratch.z_v_conv[off..off + dk];

        // Compute per-key-channel decay: gk[k] = -exp(A_log[h]) * softplus(g[k] + dt_bias[k])
        // alpha_h[k] = exp(gk[k])
        // NOTE: alpha_h is currently holding k_h (normed). We need to compute
        // the decay in a DIFFERENT buffer. Use erase_b_h as the decay buffer
        // (it's filled with sigmoid(beta) below before the gdn2 call).
        let alpha_head = weights.a_log[head].exp();
        for (i, slot) in scratch.erase_b_h[..dk].iter_mut().enumerate() {
            let g_plus_bias = scratch.g_raw[off + i] + weights.dt_bias[off + i];
            let gk = match gate {
                KdaGate::Softplus => -alpha_head * softplus(g_plus_bias),
                KdaGate::SigmoidLowerBound(bound) => bound * sigmoid(alpha_head * g_plus_bias),
            };
            let a = gk.exp();
            *slot = if a < config.alpha_eps {
                config.alpha_eps
            } else {
                a
            };
        }
        // Now erase_b_h holds alpha (decay). But we ALSO need erase_b = sigmoid(beta)*ones.
        // The gdn2 kernel takes alpha separately from erase_b. We need a separate
        // buffer for alpha. Let's restructure: use o_h_norm as the alpha buffer
        // (it's written after the gdn2 call).
        //
        // Actually wait — let me re-examine the borrow situation. We need:
        //   alpha: &[dk] (decay per channel) = scratch.erase_b_h currently
        //   erase_b: &[dk] (β-broadcast)     = needs a buffer
        //   k_h: &[dk]                       = scratch.alpha_h currently
        //
        // The issue: erase_b_h is holding alpha (decay), but we also need it to
        // hold the erase gate for the gdn2 call. We can't use it for both.
        // Solution: move decay into o_h_norm, then fill erase_b_h with β-broadcast.
        scratch.o_h_norm.copy_from_slice(&scratch.erase_b_h);
        let alpha_ref = &scratch.o_h_norm;

        // Fill erase_b_h with sigmoid(β) broadcast.
        let beta_h = scratch.beta[head];
        for slot in scratch.erase_b_h[..dk].iter_mut() {
            *slot = beta_h;
        }

        // Call gdn2_recurrent_step.
        // k = k_h (normed), v = v_h, q = q_h.
        // alpha = alpha_ref (per-channel decay = exp(gk)).
        // erase_b = erase_b_h (β-broadcast).
        // w_val = β (scalar write weight).
        let s = &mut cache.heads[head].s;
        // Need a scratch for the output — o_h is currently q_h. We need a fresh
        // buffer for the output. Use gdn2_temp... no, that's for gdn2 internals.
        // The gdn2 call writes into `out` which must be separate from q/k/v.
        // o_h holds q_h. We can't reuse it for output during the call.
        // BUT: the gdn2 kernel reads q LAST (in the readout step), after
        // modifying S. So if we write output into the same buffer as q, the
        // readout would read corrupted data.
        //
        // We need a separate output buffer. Use erase_b_h? No, it's the erase gate.
        // We have: o_h (q), alpha_h (k), z_v_conv[h] (v), erase_b_h (β-broadcast),
        // o_h_norm (alpha/decay). The only remaining per-head buffer is... none.
        //
        // Wait — gdn2 writes `out` AFTER reading q. The readout loop reads q[i]
        // and accumulates into out[j]. If out == q (same buffer), then writing
        // out[j] would corrupt q for subsequent i iterations. So they MUST be
        // separate.
        //
        // Let's allocate the output into a dedicated section. The gdn2_temp and
        // gdn2_delta buffers are used internally. We have o_concat which is the
        // final concat — we can write directly into the head's slice of o_concat.
        let out_slice = &mut scratch.o_concat[off..off + dk];

        gdn2_recurrent_step(
            k_h,                     // k
            v_h,                     // v
            q_h,                     // q
            s,                       // state matrix (dk × dv), updated in-place
            alpha_ref,               // per-channel decay
            &scratch.erase_b_h,      // β-broadcast erase gate
            beta_h,                  // w_val = sigmoid(β)
            &scratch.gdn2_write_w,   // write_w_channel (unused for Kda)
            out_slice,               // output [dv]
            &mut scratch.gdn2_temp,  // temp buffer [dv]
            &mut scratch.gdn2_delta, // delta buffer [dv]
            dk,
            dk,
            Gdn2GateConfig::Kda,
        );

        // ── FusedRMSNormGated on the output ────────────────────────────────
        // o_norm = (RMSNorm(o_h, γ, eps)) * sigmoid(g_out_h)
        // where g_out_h = g_proj(h)[head slice].
        // RMSNorm normalizes over head_dim, applies weight γ, then multiplies
        // by sigmoid(g_out) elementwise.
        //
        // out_slice currently holds the raw gdn2 output. We norm in-place into
        // the same concat position... but we need the raw values for the norm
        // denominator. Let's compute into o_h (which is now free since q was
        // consumed by the gdn2 readout).
        //
        // Wait — o_h held q_h which was read by gdn2. After gdn2 returns, o_h
        // is no longer needed. But we're about to overwrite it. That's fine —
        // q_h is consumed.
        //
        // Actually, out_slice IS the o_concat slice. We need to:
        // 1. Read out_slice (raw output)
        // 2. Compute RMSNorm(raw) * sigmoid(g_out)
        // 3. Write result back to out_slice
        //
        // We can do this in-place if we compute the RMS first (full pass to get
        // the denominator), then scale + gate in a second pass.

        // Pass 1: compute RMS denominator.
        let sum_sq = simd_sum_sq(out_slice, dk);
        let mean_sq = sum_sq / dk as f32;
        let inv_rms = 1.0 / (mean_sq + config.rms_eps).sqrt();

        // Pass 2: apply norm weight + sigmoid gate.
        for (i, out_slot) in out_slice.iter_mut().enumerate().take(dk) {
            let normed = *out_slot * inv_rms * weights.o_norm_weight[i];
            let gated = sigmoid(scratch.g_out[off + i]);
            *out_slot = normed * gated;
        }
    }

    // ── Step 4: Output projection ──────────────────────────────────────────
    // output = W_o · o_concat  [hidden_size]
    simd_matmul_rows(
        &mut scratch.output,
        &weights.o_proj,
        &scratch.o_concat,
        d,
        proj,
    );

    &mut scratch.output[..d]
}

// ─── Local math helpers ─────────────────────────────────────────────────────

/// SiLU (swish) activation in-place: `x * sigmoid(x)`.
fn silu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = *x * sigmoid(*x);
    }
}

/// Sigmoid function (delegates to the gdn2 kernel's sigmoid for consistency).
#[inline]
fn sigmoid(x: f32) -> f32 {
    crate::gdn2::kernel::sigmoid(x)
}

/// Numerically stable softplus: `log(1 + exp(x))`.
///
/// For `x >= 0`: `x + log1p(exp(-x))`.
/// For `x < 0`: `log1p(exp(x))`.
/// Matches the fla Triton kernel's `softplus` (from `fla.ops.utils.softplus`).
#[inline]
fn softplus(x: f32) -> f32 {
    if x >= 20.0 {
        // Beyond 20, exp(-x) underflows to 0 in f32; softplus ≈ x.
        x
    } else if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// L2 normalize matching the fla kernel's convention: `x / sqrt(sum(x^2) + 1e-6)`.
///
/// The shared `l2_normalize` in kernel.rs uses `1e-8` eps; the KDA Triton kernel
/// uses `1e-6`. This local variant matches the kernel exactly.
#[inline]
fn l2_normalize_eps_kda(x: &mut [f32]) {
    let norm_sq = simd_sum_sq(x, x.len());
    let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-6);
    katgpt_core::simd::simd_scale_inplace(x, inv_norm);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> KdaConfig {
        KdaConfig {
            head_dim: 8,
            n_heads: 2,
            hidden_size: 16,
            conv_kernel_size: 4,
            alpha_eps: 1e-5,
            rms_eps: 1e-5,
        }
    }

    // ── Plan 627 T4a: the GLM sigmoid lower-bound gate form ────────────

    #[test]
    fn lower_bound_gate_diverges_from_softplus_and_stays_deterministic() {
        // Same weights + tokens through both gate forms: the outputs differ
        // (different log-decay formulas), and each form is deterministic
        // across resets. The softplus arm's numeric correctness is pinned by
        // tests/kda_g1_spec_match.rs (the fla reference); this test pins the
        // BRANCH DIVERGENCE + the lower-bound form's own determinism.
        let config = small_config();
        let weights = KdaWeights::random(&config, 777);
        let tokens: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                (0..config.hidden_size)
                    .map(|k| ((i * 7 + k * 3) % 9) as f32 * 0.15 - 0.6)
                    .collect()
            })
            .collect();

        let run = |bound: Option<f32>| -> Vec<Vec<f32>> {
            let mut cache = KdaLayerCache::new(&config);
            let mut scratch = KdaForwardScratch::new(&config);
            tokens
                .iter()
                .map(|h| {
                    let out = match bound {
                        None => kda_forward_token(&config, &weights, &mut cache, &mut scratch, h),
                        Some(b) => kda_forward_token_lower_bound_gate(
                            &config,
                            &weights,
                            &mut cache,
                            &mut scratch,
                            h,
                            b,
                        ),
                    };
                    out.to_vec()
                })
                .collect()
        };

        let softplus = run(None);
        let lower_bound = run(Some(-5.0));
        let any_diff = softplus
            .iter()
            .zip(lower_bound.iter())
            .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
        assert!(any_diff, "the two gate forms produced identical outputs");
        // Determinism: a second lower-bound run reproduces the first.
        assert_eq!(run(Some(-5.0)), lower_bound);
    }

    #[test]
    fn smoke_kimi_k3_0_40b_dims_finite() {
        let config = KdaConfig::kimi_k3_0_40b();
        let weights = KdaWeights::random(&config, 42);
        let mut cache = KdaLayerCache::new(&config);
        let mut scratch = KdaForwardScratch::new(&config);

        let h: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32).sin() * 0.1)
            .collect();
        let out = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h);

        assert_eq!(out.len(), config.hidden_size);
        for &v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn smoke_small_config_finite_3_tokens() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 7);
        let mut cache = KdaLayerCache::new(&config);
        let mut scratch = KdaForwardScratch::new(&config);

        for t in 0..3 {
            let h: Vec<f32> = (0..config.hidden_size)
                .map(|i| ((i + t * 13) as f32).sin() * 0.1)
                .collect();
            let out = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h);
            assert_eq!(out.len(), config.hidden_size);
            for &v in out.iter() {
                assert!(v.is_finite(), "token {t}: non-finite output: {v}");
            }
        }
    }

    #[test]
    fn cache_reset_zeros_state() {
        let config = small_config();
        let mut cache = KdaLayerCache::new(&config);

        // Run a token to populate state.
        let weights = KdaWeights::random(&config, 1);
        let mut scratch = KdaForwardScratch::new(&config);
        let h = vec![0.1; config.hidden_size];
        let _ = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h);

        // Verify state is non-zero.
        let any_nonzero = cache.heads[0].s.iter().any(|&x| x != 0.0);
        assert!(any_nonzero, "state should be non-zero after a forward pass");

        // Reset.
        cache.reset();
        for head in &cache.heads {
            for &v in &head.s {
                assert_eq!(v, 0.0, "state should be zero after reset");
            }
        }
        assert_eq!(cache.q_conv.buf[0], 0.0);
    }

    #[test]
    fn determinism_same_inputs_same_output() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 99);

        let tokens: Vec<Vec<f32>> = (0..3)
            .map(|t| {
                (0..config.hidden_size)
                    .map(|i| ((i + t * 7) as f32).sin() * 0.1)
                    .collect()
            })
            .collect();

        // Run 1.
        let mut cache1 = KdaLayerCache::new(&config);
        let mut scratch1 = KdaForwardScratch::new(&config);
        let mut out1 = Vec::new();
        for h in &tokens {
            let o = kda_forward_token(&config, &weights, &mut cache1, &mut scratch1, h);
            out1.extend_from_slice(o);
        }

        // Run 2 (same inputs).
        let mut cache2 = KdaLayerCache::new(&config);
        let mut scratch2 = KdaForwardScratch::new(&config);
        let mut out2 = Vec::new();
        for h in &tokens {
            let o = kda_forward_token(&config, &weights, &mut cache2, &mut scratch2, h);
            out2.extend_from_slice(o);
        }

        assert_eq!(out1, out2, "same inputs must produce identical outputs");
    }

    #[test]
    fn state_grows_across_tokens() {
        // Verify the recurrent state evolves: output at t=1 differs from t=0.
        let config = small_config();
        let weights = KdaWeights::random(&config, 55);
        let mut cache = KdaLayerCache::new(&config);
        let mut scratch = KdaForwardScratch::new(&config);

        let h0: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32).sin() * 0.1)
            .collect();
        let h1: Vec<f32> = (0..config.hidden_size)
            .map(|i| ((i + 13) as f32).sin() * 0.1)
            .collect();

        let out0 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h0).to_vec();
        let out1 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h1).to_vec();

        // Outputs should differ (state accumulated from t0 affects t1).
        let max_diff = out0
            .iter()
            .zip(out1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "outputs at t=0 and t=1 should differ (max_diff={max_diff:.2e})"
        );
    }

    #[test]
    fn kimi_k3_0_40b_config_values() {
        let config = KdaConfig::kimi_k3_0_40b();
        assert_eq!(config.head_dim, 32, "head_dim must be 32 (not 128)");
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.conv_kernel_size, 4);
        assert!(
            (config.rms_eps - 1e-5).abs() < 1e-10,
            "rms_eps must be 1e-5"
        );
        assert_eq!(config.proj_dim(), 256); // 32 * 8
    }

    #[test]
    fn q_scale_is_inv_sqrt_head_dim() {
        let config = KdaConfig::kimi_k3_0_40b();
        let expected = 1.0 / (32.0_f32).sqrt();
        assert!(
            (config.q_scale() - expected).abs() < 1e-7,
            "q_scale mismatch"
        );
    }
}
