//! D2F Discrete Diffusion Forcing — Phase 0 Proof Tasks (Plan 066)
//!
//! Implements mini dLLM training infrastructure for researching whether
//! Discrete Diffusion Forcing is viable for our system.
//!
//! # Phase 0 Tasks
//!
//! - **Task 0.1**: Bidirectional attention on CPU
//! - **Task 0.2**: Mask token + noise schedule + corruption
//! - **Task 0.3**: Mini dLLM training loop with SGD backprop
//! - **Task 0.4**: Block-causal vs bidirectional A/B comparison
//! - **Task 0.5**: Constraint pruner during denoising
//!
//! # Module layout
//!
//! Split from the historical monolithic `src/dllm.rs` (Issue 166, 2026-07-17).
//! The implementation stays in `mod.rs`; the test suite moved to
//! [`tests`] (tests are exempt from the 2048-line soft limit per Issue 162).

use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng, kv_dim, matmul, matmul_relu, rmsnorm};

pub mod text_corpus;

#[cfg(feature = "replaid_schedules")]
use crate::pruners::variance_minimizer::{VarianceMinimizer, VarianceMinimizerConfig};

// ═══════════════════════════════════════════════════════════════
// Loss Averaging Strategy
// ═══════════════════════════════════════════════════════════════

/// Loss averaging strategy for masked positions in D2F training.
/// How to average the loss across masked positions.
///
/// `#[repr(u8)]` ensures 1-byte size for compact storage in hot-path structs.
/// Nemotron validates +2.12% accuracy with global averaging over per-sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum LossAveraging {
    /// Average loss across all masked positions in the batch (global).
    /// `L = (1/(N*L_masked)) * Σ_n Σ_i ℓ_{n,i}`
    /// Default — validated by Nemotron to improve accuracy.
    #[default]
    Global,
    /// Average per-sequence, then average across sequences.
    /// `L = (1/N) * Σ_n (1/L_n) * Σ_i ℓ_{n,i}`
    PerSequence,
}

// ═══════════════════════════════════════════════════════════════
// Task 0.2: Noise Schedule + Corruption
// ═══════════════════════════════════════════════════════════════

/// Noise schedule for discrete diffusion.
/// Produces monotonically increasing mask ratios for block-based corruption.
///
/// Field order: usize (8-byte) before f32 (4-byte) to eliminate padding.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    pub n_blocks: usize,
    pub max_ratio: f32,
    pub min_ratio: f32,
}

impl NoiseSchedule {
    pub fn new(min_ratio: f32, max_ratio: f32, n_blocks: usize) -> Self {
        Self {
            n_blocks,
            min_ratio,
            max_ratio,
        }
    }

    /// Returns mask ratios per block, monotonically increasing from min to max.
    pub fn monotonic_ratios(&self) -> Vec<f32> {
        match self.n_blocks {
            0 => Vec::new(),
            1 => vec![(self.min_ratio + self.max_ratio) / 2.0],
            n => {
                let step = (self.max_ratio - self.min_ratio) / (n - 1) as f32;
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(self.min_ratio + i as f32 * step);
                }
                out
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Plan 078: Adaptive Noise Schedule (RePlaid Variance-Minimized)
// ═══════════════════════════════════════════════════════════════

/// Adaptive noise schedule that equalizes per-step denoising difficulty.
///
/// RePlaid Prop 1: "there exists a unique noise schedule γ* such that
/// `ℓ_θ,γ*(t)` ≡ κ for all t, and consequently `Var_t[ℓ]` = 0."
///
/// We adapt this to discrete D2F: track per-step reconstruction accuracy,
/// then adjust mask ratios so each step contributes equal difficulty.
/// Steps that are too easy (high accuracy) get harder masks.
/// Steps that are too hard (low accuracy) get easier masks.
///
/// Field order: grouped by alignment (Vec/usize then f32) to minimize padding.
#[cfg(feature = "replaid_schedules")]
#[derive(Debug, Clone)]
pub struct AdaptiveNoiseSchedule {
    /// Per-step loss tracker (one VarianceMinimizer per block).
    step_trackers: Vec<VarianceMinimizer>,
    /// Current adapted ratios.
    current_ratios: Vec<f32>,
    /// Base schedule parameters.
    n_blocks: usize,
    /// Number of adaptation steps performed.
    adaptations: u32,
    max_ratio: f32,
    min_ratio: f32,
}

#[cfg(feature = "replaid_schedules")]
impl AdaptiveNoiseSchedule {
    /// Create a new adaptive schedule starting from monotonic ratios.
    pub fn new(min_ratio: f32, max_ratio: f32, n_blocks: usize) -> Self {
        let schedule = NoiseSchedule::new(min_ratio, max_ratio, n_blocks);
        let current_ratios = schedule.monotonic_ratios();

        let config = VarianceMinimizerConfig {
            mean_decay: 0.95,
            var_decay: 0.95,
            lr: 0.05,
            min_param: min_ratio,
            max_param: max_ratio,
        };

        let step_trackers = current_ratios
            .iter()
            .map(|&ratio| VarianceMinimizer::with_param(config, ratio))
            .collect();

        Self {
            step_trackers,
            current_ratios,
            n_blocks,
            adaptations: 0,
            min_ratio,
            max_ratio,
        }
    }

    /// Convenience constructor from an existing `NoiseSchedule`.
    pub fn from_schedule(schedule: &NoiseSchedule) -> Self {
        Self::new(schedule.min_ratio, schedule.max_ratio, schedule.n_blocks)
    }

    /// Record per-step reconstruction loss during training.
    ///
    /// Called after each denoising step to feed the adaptive tracker.
    /// Block index is clamped to valid range.
    pub fn record_step_loss(&mut self, block_idx: usize, loss: f32) {
        if self.step_trackers.is_empty() {
            return;
        }
        let idx = block_idx.min(self.step_trackers.len() - 1);
        self.step_trackers[idx].observe(loss);
    }

    /// Adapt ratios to flatten per-step loss variance.
    ///
    /// Each tracker independently adjusts its ratio, then we sort
    /// to maintain monotonicity (RePlaid requires ordered schedules).
    /// Returns `&self.current_ratios` after adaptation (avoids clone).
    pub fn adapt_ratios(&mut self) -> &[f32] {
        for (i, tracker) in self.step_trackers.iter_mut().enumerate() {
            self.current_ratios[i] = tracker.adapt();
        }
        // Sort to maintain monotonicity (min to max).
        // total_cmp avoids the partial_cmp + unwrap_or overhead and handles NaN.
        self.current_ratios.sort_by(f32::total_cmp);
        self.adaptations += 1;
        &self.current_ratios
    }

    /// Current ratios (monotonic before first adaptation).
    pub fn ratios(&self) -> &[f32] {
        &self.current_ratios
    }

    /// Reset all trackers and restore monotonic fallback ratios.
    pub fn reset(&mut self) {
        let schedule = NoiseSchedule::new(self.min_ratio, self.max_ratio, self.n_blocks);
        self.current_ratios = schedule.monotonic_ratios();

        let config = VarianceMinimizerConfig {
            mean_decay: 0.95,
            var_decay: 0.95,
            lr: 0.05,
            min_param: self.min_ratio,
            max_param: self.max_ratio,
        };

        self.step_trackers = self
            .current_ratios
            .iter()
            .map(|&ratio| VarianceMinimizer::with_param(config, ratio))
            .collect();

        self.adaptations = 0;
    }

    /// Number of adaptation steps performed so far.
    pub fn adaptations(&self) -> u32 {
        self.adaptations
    }
}

/// Corrupt a block of tokens by replacing some with the mask token (zero-alloc variant).
///
/// Writes into pre-allocated buffers to avoid per-call heap allocation.
/// `corrupted` and `is_masked` are cleared and refilled; `positions` is used as scratch for Fisher-Yates.
pub fn corrupt_block_into(
    tokens: &[usize],
    mask_ratio: f32,
    mask_token: usize,
    rng: &mut Rng,
    corrupted: &mut Vec<usize>,
    is_masked: &mut Vec<bool>,
    positions: &mut Vec<usize>,
) -> usize {
    let len = tokens.len();
    let n_mask = ((len as f32 * mask_ratio).ceil() as usize).min(len);

    // Reuse buffers: clear and refill
    corrupted.clear();
    corrupted.extend_from_slice(tokens);
    is_masked.clear();
    is_masked.resize(len, false);
    positions.clear();
    positions.extend(0..len);

    // Fisher-Yates shuffle to pick random positions
    for i in (1..positions.len()).rev() {
        let j = (rng.next() as usize) % (i + 1);
        positions.swap(i, j);
    }

    for &pos in &positions[..n_mask] {
        corrupted[pos] = mask_token;
        is_masked[pos] = true;
    }

    n_mask
}

/// Corrupt a block of tokens by replacing some with the mask token.
/// Returns (corrupted_tokens, is_masked indicators).
///
/// **Note:** This allocates buffers internally. For training loops, prefer
/// [`corrupt_block_into`] to avoid per-call heap allocation.
pub fn corrupt_block(
    tokens: &[usize],
    mask_ratio: f32,
    mask_token: usize,
    rng: &mut Rng,
) -> (Vec<usize>, Vec<bool>) {
    let mut corrupted = Vec::with_capacity(tokens.len());
    let mut is_masked = Vec::with_capacity(tokens.len());
    let mut positions = Vec::with_capacity(tokens.len());
    let _n_mask = corrupt_block_into(
        tokens,
        mask_ratio,
        mask_token,
        rng,
        &mut corrupted,
        &mut is_masked,
        &mut positions,
    );
    (corrupted, is_masked)
}

// ═══════════════════════════════════════════════════════════════
// Forward-Positions Cluster — re-export from katgpt-forward
// ═══════════════════════════════════════════════════════════════
//
// Plan 402 (2026-07-06): `BidirectionalContext`, `forward_bidirectional_positions`,
// `forward_bidirectional_positions_into`, `attention_forward_safe` (allocating
// wrapper), and `forward_block_causal_positions` moved to
// `katgpt_forward::forward_positions`. This block re-exports them so every
// historical `crate::dllm::*` import path (notably the `denoise_loop*` family,
// `evaluate_accuracy` training code, and `forward_save`) continues to resolve.
//
// The struct's fields are `pub` in katgpt-forward because root's
// `denoise_loop_rcd` / `denoise_loop_rcd_3sr` write directly to the
// cfg-gated `rcd_residual_embeddings` / `tsr_warm_start_embeddings` buffers
// (and the `rcd_active` / `tsr_active` flags) after each commitment phase.
// This mirrors the standard "move type, re-export, leave consumers in root"
// pattern (same as `forward_set_causal_positions` in Plan 401).
#[cfg(feature = "dllm")]
pub use katgpt_forward::forward_positions::{
    BidirectionalContext, attention_forward_safe, forward_bidirectional_positions,
    forward_bidirectional_positions_into,
};
// `forward_block_causal_positions` is re-exported separately near its original
// location (below, after the training code) for source-history continuity.

/// Safe bidirectional attention for one query position.
/// Returns (attn_output[n_embd], attn_weights[n_head * seq_len]).
///
/// Plan 398 (2026-07-05): the zero-alloc `_into` variant moved to
/// `katgpt_forward::d2f_context::attention_forward_safe_into` and is
/// re-exported here. Single source of truth across the root callers that
/// remain (`forward_save`, `forward_save_set_causal`).
/// Plan 402 (2026-07-06): the allocating wrapper + the bidirectional/block-causal
/// position forwards also moved to katgpt-forward; this `_into` re-export stays
/// because `forward_save` (training activations, root-resident) still calls it.
pub(crate) use katgpt_forward::attention_forward_safe_into;

// (End of forward-positions cluster re-export block — see Plan 402.)

// ═══════════════════════════════════════════════════════════════
// Task 0.3: Training Infrastructure
// ═══════════════════════════════════════════════════════════════

/// Saved activations from forward pass, needed for backward.
///
/// Borrows from `ForwardSaveContext` to avoid cloning all activations (Issue 110).
///
/// Field order: all references (8-byte) grouped, then usize to eliminate padding.
/// Saved activations for one training forward (Issue 869 T5: per-layer
/// planes).
///
/// Plane strides are **`block_size`-capacity** (`bs * <dim>`), not
/// `seq_len`-trimmed: only `[0..seq_len)` of each plane is active, exactly as
/// `ForwardSaveContext` lays them out. At `n_layer == 1` every per-layer
/// field's plane 0 is the pre-869 single-plane buffer, so the layer loop in
/// `forward_save`/`backward` degenerates to the historical op sequence
/// bit-for-bit.
struct ForwardActivations<'a> {
    embeddings: &'a [f32],     // [seq_len * n]
    after_norm1: &'a [f32],    // [seq_len * n] — residual stream h_0 (rmsnorm(embeddings))
    after_norm2: &'a [f32], // [seq_len * n] — layer 0's QKV input (rmsnorm(h_0): double-norm quirk)
    x_norm_rest: &'a [f32], // [(L-1) * bs * n] — layer l≥1 QKV inputs, rmsnorm(h_l), plane l-1
    q: &'a [f32],           // [L * bs * n]
    k: &'a [f32],           // [L * bs * kvd]
    v: &'a [f32],           // [L * bs * kvd]
    attn_weights: &'a [f32], // [L * bs * n_head * bs]
    attn_out: &'a [f32],    // [L * bs * n]
    after_attn_res: &'a [f32], // [L * bs * n]
    after_mlp_norm: &'a [f32], // [L * bs * n]
    mlp_hidden: &'a [f32],  // [L * bs * mlp_hidden]
    hidden_all: &'a [f32],  // [L * bs * n] — h_{l+1} planes; the last plane feeds lm_head
    logits: &'a [f32],      // [seq_len * vocab_size]
    seq_len: usize,
    n_layer: usize,
}

/// Pre-allocated context for `forward_save`, avoiding per-call allocations.
///
/// Field order: all Vec<f32> (24-byte, 8-byte aligned) before usize fields
/// to eliminate inter-field padding.
///
/// Issue 869 T5: the per-layer activation buffers carry `n_layer` planes of
/// `block_size` capacity each — plane `l` lives at `l * bs * <dim>`, mirroring
/// the decode kernel's KV-plane layout. `x_norm_rest` holds the layer-l≥1
/// QKV-input planes; `hidden_all` the per-layer output streams (`hidden_all`'s
/// last plane is the old `hidden_final`).
struct ForwardSaveContext {
    // Vec fields grouped first (all 24 bytes, 8-byte aligned)
    embeddings: Vec<f32>,
    after_norm1: Vec<f32>,
    after_norm2: Vec<f32>,
    x_norm_rest: Vec<f32>,
    q_all: Vec<f32>,
    k_all: Vec<f32>,
    v_all: Vec<f32>,
    attn_weights_all: Vec<f32>,
    attn_out_all: Vec<f32>,
    after_attn_res: Vec<f32>,
    after_mlp_norm: Vec<f32>,
    mlp_hidden_all: Vec<f32>,
    hidden_all: Vec<f32>,
    logits_all: Vec<f32>,
    // Per-position scratch (reused each iteration)
    x_buf: Vec<f32>,
    x_proj_buf: Vec<f32>,
    x_mlp_buf: Vec<f32>,
    // Attention scratch (reused across positions, avoids per-position allocation)
    attn_scratch_out: Vec<f32>,
    attn_scratch_weights: Vec<f32>,
    attn_scratch_scores: Vec<f32>,
    // usize fields last (8-byte aligned)
    // Dimension constants cached from config
    n: usize,
    kvd: usize,
    vocab_size: usize,
    mlp_hidden: usize,
    n_head: usize,
    n_layer: usize,
    seq_len: usize,
}

impl ForwardSaveContext {
    fn new(config: &Config) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let bs = config.block_size;
        let l = config.n_layer;
        assert!(l >= 1, "n_layer must be at least 1, got {l}");
        Self {
            embeddings: vec![0.0f32; bs * n],
            after_norm1: vec![0.0f32; bs * n],
            after_norm2: vec![0.0f32; bs * n],
            x_norm_rest: vec![0.0f32; (l - 1) * bs * n],
            q_all: vec![0.0f32; l * bs * n],
            k_all: vec![0.0f32; l * bs * kvd],
            v_all: vec![0.0f32; l * bs * kvd],
            attn_weights_all: vec![0.0f32; l * bs * config.n_head * bs],
            attn_out_all: vec![0.0f32; l * bs * n],
            after_attn_res: vec![0.0f32; l * bs * n],
            after_mlp_norm: vec![0.0f32; l * bs * n],
            mlp_hidden_all: vec![0.0f32; l * bs * config.mlp_hidden],
            hidden_all: vec![0.0f32; l * bs * n],
            logits_all: vec![0.0f32; bs * config.vocab_size],
            x_buf: vec![0.0f32; n],
            x_proj_buf: vec![0.0f32; n],
            x_mlp_buf: vec![0.0f32; n],
            attn_scratch_out: vec![0.0f32; n],
            attn_scratch_weights: vec![0.0f32; config.n_head * bs],
            attn_scratch_scores: vec![0.0f32; bs],
            n,
            kvd,
            vocab_size: config.vocab_size,
            mlp_hidden: config.mlp_hidden,
            n_head: config.n_head,
            n_layer: l,
            seq_len: 0,
        }
    }

    fn reset(&mut self, seq_len: usize) {
        self.seq_len = seq_len;
        let n = self.n;
        let kvd = self.kvd;
        let nh = self.n_head;
        let mlp_h = self.mlp_hidden;
        let vocab = self.vocab_size;

        // Single-plane buffers: trim to the active prefix.
        for buf in [
            &mut self.embeddings,
            &mut self.after_norm1,
            &mut self.after_norm2,
        ] {
            buf[..seq_len * n].fill(0.0);
        }
        self.logits_all[..seq_len * vocab].fill(0.0);

        // Multi-plane buffers: zero each plane's active prefix. Plane strides
        // are `block_size`-capacity — recover the capacity from the q planes
        // (`q_all.len() == n_layer * bs * n`).
        let bs_cap = self.q_all.len() / self.n_layer / n;
        for l in 0..self.n_layer {
            let q_base = l * bs_cap * n;
            self.q_all[q_base..q_base + seq_len * n].fill(0.0);
            self.attn_out_all[q_base..q_base + seq_len * n].fill(0.0);
            self.after_attn_res[q_base..q_base + seq_len * n].fill(0.0);
            self.after_mlp_norm[q_base..q_base + seq_len * n].fill(0.0);
            self.hidden_all[q_base..q_base + seq_len * n].fill(0.0);
            if l > 0 {
                let xn_base = (l - 1) * bs_cap * n;
                self.x_norm_rest[xn_base..xn_base + seq_len * n].fill(0.0);
            }
            let kv_base = l * bs_cap * kvd;
            self.k_all[kv_base..kv_base + seq_len * kvd].fill(0.0);
            self.v_all[kv_base..kv_base + seq_len * kvd].fill(0.0);
            let aw_base = l * bs_cap * nh * bs_cap;
            self.attn_weights_all[aw_base..aw_base + seq_len * nh * seq_len].fill(0.0);
            let mh_base = l * bs_cap * mlp_h;
            self.mlp_hidden_all[mh_base..mh_base + seq_len * mlp_h].fill(0.0);
        }
    }
}

/// Gradient storage mirroring TransformerWeights layout.
///
/// Issue 869 T5: the six per-layer matrices carry `n_layer` planes, plane `l`
/// at `l * <matrix len>` — `sgd_update` slices them per layer using the
/// weights' own layer lengths (identical across layers by construction).
struct TrainingGradients {
    wte: Vec<f32>,
    wpe: Vec<f32>,
    lm_head: Vec<f32>,
    attn_wq: Vec<f32>, // [L * n * n]
    attn_wk: Vec<f32>, // [L * kvd * n]
    attn_wv: Vec<f32>, // [L * kvd * n]
    attn_wo: Vec<f32>, // [L * n * n]
    mlp_w1: Vec<f32>,  // [L * mlp_hidden * n]
    mlp_w2: Vec<f32>,  // [L * n * mlp_hidden]
}

impl TrainingGradients {
    fn zeros(config: &Config) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let l = config.n_layer;
        Self {
            wte: vec![0.0; config.vocab_size * n],
            wpe: vec![0.0; config.block_size * n],
            lm_head: vec![0.0; config.vocab_size * n],
            attn_wq: vec![0.0; l * n * n],
            attn_wk: vec![0.0; l * kvd * n],
            attn_wv: vec![0.0; l * kvd * n],
            attn_wo: vec![0.0; l * n * n],
            mlp_w1: vec![0.0; l * config.mlp_hidden * n],
            mlp_w2: vec![0.0; l * n * config.mlp_hidden],
        }
    }
}

/// Pre-allocated context for `backward`, avoiding per-call allocations.
///
/// Issue 869 T5: the per-layer gradient buffers carry `n_layer`
/// `block_size`-capacity planes (same layout as `ForwardSaveContext`).
/// `d_x_norm` replaces `d_after_norm2` (the layer-l QKV-input gradient,
/// plane 0 being layer 0's after_norm2 gradient); `d_aar_saved` replaces
/// `d_after_attn_res_saved`; `d_h_next` is the residual-stream gradient
/// handoff from layer l+1's Phase 3 to layer l's Phase 1. The never-read
/// `d_after_norm1_final` buffer is gone.
struct BackwardContext {
    d_logits: Vec<f32>,
    d_hf: Vec<f32>,
    d_mh: Vec<f32>,
    d_amn: Vec<f32>,
    d_raw: Vec<f32>,
    d_an2: Vec<f32>,
    d_an1: Vec<f32>,
    d_h_next: Vec<f32>,
    /// Scratch buffer for rmsnorm_backward (avoids per-call allocation)
    d_rmsnorm_buf: Vec<f32>,
    /// Scratch buffer for softmax_backward (avoids per-call allocation)
    d_softmax_buf: Vec<f32>,
    /// Pre-allocated intermediate gradient buffers (Issue 109; Issue 869 T5 planes)
    d_attn_out: Vec<f32>, // [L * bs * n]
    d_q: Vec<f32>,         // [L * bs * n]
    d_k: Vec<f32>,         // [L * bs * kvd]
    d_v: Vec<f32>,         // [L * bs * kvd]
    d_x_norm: Vec<f32>,    // [L * bs * n]
    d_aar_saved: Vec<f32>, // [L * bs * n]
    /// Pre-allocated gradient accumulator — cleared + reused per backward call
    /// instead of allocating 9 Vecs each time.
    grads: TrainingGradients,
    /// Scratch buffer for masked_loss exp computation (avoids per-call allocation)
    loss_exp_buf: Vec<f32>,
}

impl BackwardContext {
    fn new(config: &Config) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let bs = config.block_size;
        let l = config.n_layer;
        Self {
            d_logits: vec![0.0f32; config.vocab_size],
            d_hf: vec![0.0f32; n],
            d_mh: vec![0.0f32; config.mlp_hidden],
            d_amn: vec![0.0f32; n],
            d_raw: vec![0.0f32; config.block_size],
            d_an2: vec![0.0f32; n],
            d_an1: vec![0.0f32; n],
            d_h_next: vec![0.0f32; bs * n],
            d_rmsnorm_buf: vec![0.0f32; n],
            d_softmax_buf: vec![0.0f32; config.block_size],
            d_attn_out: vec![0.0f32; l * bs * n],
            d_q: vec![0.0f32; l * bs * n],
            d_k: vec![0.0f32; l * bs * kvd],
            d_v: vec![0.0f32; l * bs * kvd],
            d_x_norm: vec![0.0f32; l * bs * n],
            d_aar_saved: vec![0.0f32; l * bs * n],
            grads: TrainingGradients::zeros(config),
            loss_exp_buf: vec![0.0f32; config.vocab_size],
        }
    }
}

/// Forward pass saving all activations for training.
///
/// Issue 869 T5: honors `config.n_layer` — the layer loop chains the residual
/// stream exactly as the decode kernel does (`h_0 = rmsnorm(embedding)` — the
/// lane's double-norm quirk; `h_{l+1} = h_l + MLP_l(rmsnorm(aar_l))` where
/// `aar_l = h_l + Attn_l(rmsnorm_l(h_l))`), each layer's Q/K/V/attention/MLP
/// activations saved into their own `block_size`-capacity plane. At
/// `n_layer == 1` the op sequence is the pre-869 single-layer one, bit for
/// bit (the layer loop degenerates; plane 0 is the old buffer).
fn forward_save<'a>(
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
    ctx: &'a mut ForwardSaveContext,
) -> ForwardActivations<'a> {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let seq_len = tokens.len().min(config.block_size);
    let scale = 1.0 / (hd as f32).sqrt();
    let l_total = config.n_layer;
    let bs = config.block_size;
    let nh = config.n_head;
    let mlp_h = config.mlp_hidden;

    ctx.reset(seq_len);

    // Plane bases (block_size-capacity strides, mirroring the KV planes of
    // the decode kernel).
    let q_stride = bs * n;
    let kv_stride = bs * kvd;
    let aw_stride = bs * nh * bs;
    let mh_stride = bs * mlp_h;

    // Phase A0: Embeddings + the layer-0 double norm (the lane's quirk —
    // identical to the pre-869 Phase A embedding part).
    for (p, &token) in tokens.iter().enumerate().take(seq_len) {
        katgpt_core::simd::simd_add_into(
            &mut ctx.embeddings[p * n..(p + 1) * n],
            &weights.wte[token * n..(token + 1) * n],
            &weights.wpe[p * n..(p + 1) * n],
        );
        ctx.x_buf[..n].copy_from_slice(&ctx.embeddings[p * n..(p + 1) * n]);
        rmsnorm(&mut ctx.x_buf);
        ctx.after_norm1[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf[..n]);
        rmsnorm(&mut ctx.x_buf);
        ctx.after_norm2[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf[..n]);
    }

    for (l, layer) in weights.layers.iter().take(l_total).enumerate() {
        let q_base = l * q_stride;
        let kv_base = l * kv_stride;
        let aw_base = l * aw_stride;
        let mh_base = l * mh_stride;
        let last = l + 1 == l_total;

        // Phase A(l): Q/K/V for all positions from this layer's input norm.
        // Layer 0 reads after_norm2 (the double-norm quirk); layer l ≥ 1
        // norms the residual stream h_l (= layer l-1's output plane).
        for p in 0..seq_len {
            if l == 0 {
                ctx.x_buf[..n].copy_from_slice(&ctx.after_norm2[p * n..(p + 1) * n]);
            } else {
                let h_prev = (l - 1) * q_stride;
                ctx.x_buf[..n]
                    .copy_from_slice(&ctx.hidden_all[h_prev + p * n..h_prev + (p + 1) * n]);
                rmsnorm(&mut ctx.x_buf);
                let xn = (l - 1) * q_stride;
                ctx.x_norm_rest[xn + p * n..xn + (p + 1) * n].copy_from_slice(&ctx.x_buf[..n]);
            }
            matmul(
                &mut ctx.q_all[q_base + p * n..q_base + (p + 1) * n],
                &layer.attn_wq,
                &ctx.x_buf,
                n,
                n,
            );
            matmul(
                &mut ctx.k_all[kv_base + p * kvd..kv_base + (p + 1) * kvd],
                &layer.attn_wk,
                &ctx.x_buf,
                kvd,
                n,
            );
            matmul(
                &mut ctx.v_all[kv_base + p * kvd..kv_base + (p + 1) * kvd],
                &layer.attn_wv,
                &ctx.x_buf,
                kvd,
                n,
            );
        }

        // Phase B(l): Bidirectional attention (zero-alloc, pre-allocated scratch)
        for p in 0..seq_len {
            attention_forward_safe_into(
                &ctx.q_all[q_base + p * n..q_base + (p + 1) * n],
                &ctx.k_all[kv_base..(l + 1) * kv_stride],
                &ctx.v_all[kv_base..(l + 1) * kv_stride],
                nh,
                config.n_kv_head,
                hd,
                kvd,
                seq_len,
                scale,
                &mut ctx.attn_scratch_out,
                &mut ctx.attn_scratch_weights,
                &mut ctx.attn_scratch_scores,
            );
            ctx.attn_out_all[q_base + p * n..q_base + (p + 1) * n]
                .copy_from_slice(&ctx.attn_scratch_out);
            ctx.attn_weights_all[aw_base + p * nh * seq_len..aw_base + (p + 1) * nh * seq_len]
                .copy_from_slice(&ctx.attn_scratch_weights[..nh * seq_len]);
        }

        // Phase C(l): Output projection + residual + MLP
        // Uses x_buf for xr2 temporary, x_proj_buf for rmsnorm I/O, x_mlp_buf for mlp output.
        // The residual base is h_l: after_norm1 for layer 0 (the lane's quirk),
        // layer l-1's output plane otherwise.
        let h_base = if l == 0 { 0 } else { (l - 1) * q_stride };
        for p in 0..seq_len {
            // x_proj = wo @ attn_out
            matmul(
                &mut ctx.x_proj_buf,
                &layer.attn_wo,
                &ctx.attn_out_all[q_base + p * n..q_base + (p + 1) * n],
                n,
                n,
            );
            // Add residual: x_proj += h_l (per-statement borrow — h_l's plane is
            // written by this loop at l ≥ 1, so no borrow may outlive the add)
            if l == 0 {
                katgpt_core::simd::simd_add_inplace(
                    &mut ctx.x_proj_buf,
                    &ctx.after_norm1[p * n..(p + 1) * n],
                );
            } else {
                katgpt_core::simd::simd_add_inplace(
                    &mut ctx.x_proj_buf,
                    &ctx.hidden_all[h_base + p * n..h_base + (p + 1) * n],
                );
            }
            // after_attn_res = x_proj (the residual output)
            ctx.after_attn_res[q_base + p * n..q_base + (p + 1) * n]
                .copy_from_slice(&ctx.x_proj_buf[..n]);

            // xr2 = x_proj (copy for later residual addition)
            ctx.x_buf[..n].copy_from_slice(&ctx.x_proj_buf[..n]);
            // rmsnorm(x_proj) in place
            rmsnorm(&mut ctx.x_proj_buf);
            ctx.after_mlp_norm[q_base + p * n..q_base + (p + 1) * n]
                .copy_from_slice(&ctx.x_proj_buf);
            matmul_relu(
                &mut ctx.mlp_hidden_all[mh_base + p * mlp_h..mh_base + (p + 1) * mlp_h],
                &layer.mlp_w1,
                &ctx.x_proj_buf,
                mlp_h,
                n,
            );
            matmul(
                &mut ctx.x_mlp_buf,
                &layer.mlp_w2,
                &ctx.mlp_hidden_all[mh_base + p * mlp_h..mh_base + (p + 1) * mlp_h],
                n,
                mlp_h,
            );
            // Add xr2 residual (stored in x_buf)
            katgpt_core::simd::simd_add_inplace(&mut ctx.x_mlp_buf, &ctx.x_buf[..n]);
            ctx.hidden_all[q_base + p * n..q_base + (p + 1) * n].copy_from_slice(&ctx.x_mlp_buf);
            if last {
                // Logits from the FINAL layer's stream only (the pre-869 position).
                matmul(
                    &mut ctx.logits_all[p * config.vocab_size..(p + 1) * config.vocab_size],
                    &weights.lm_head,
                    &ctx.x_mlp_buf,
                    config.vocab_size,
                    n,
                );
            }
        }
    }

    ForwardActivations {
        embeddings: &ctx.embeddings[..seq_len * n],
        after_norm1: &ctx.after_norm1[..seq_len * n],
        after_norm2: &ctx.after_norm2[..seq_len * n],
        x_norm_rest: &ctx.x_norm_rest,
        q: &ctx.q_all,
        k: &ctx.k_all,
        v: &ctx.v_all,
        attn_weights: &ctx.attn_weights_all,
        attn_out: &ctx.attn_out_all,
        after_attn_res: &ctx.after_attn_res,
        after_mlp_norm: &ctx.after_mlp_norm,
        mlp_hidden: &ctx.mlp_hidden_all,
        hidden_all: &ctx.hidden_all,
        logits: &ctx.logits_all[..seq_len * config.vocab_size],
        seq_len,
        n_layer: l_total,
    }
}

/// Set-causal forward pass with activation saving (for SW-SetDLM training).
///
/// Identical to [`forward_save`] except Phase B applies a set-causal attention
/// mask: position `q` attends only to positions `t` where
/// `gen_steps[t] <= gen_steps[q]`. Ineligible positions get zero attention
/// weight (never enter the softmax denominator). This is the training-time
/// companion of [`forward_set_causal_positions`] — it produces the same
/// logits/attention pattern but saves all intermediate activations into
/// `ctx` so that [`backward`] can compute gradients.
///
/// # The backward compatibility invariant
///
/// [`backward`] computes the softmax Jacobian-vector product via
/// [`softmax_backward_into`], whose formula is `d_scores[i] = w[i] * (dy[i] - dot(w, dy))`.
/// When `w[i] == 0.0` (ineligible position), `d_scores[i]` is identically zero,
/// so no gradient flows through masked attention paths. This means the
/// existing [`backward`] works correctly for set-causal WITHOUT modification —
/// the mask is encoded in the attention weights, not in the backward logic.
///
/// # Arguments
/// - `gen_steps`: generation step per position, length `seq_len`. Position `q`
///   attends to `t` iff `gen_steps[t] <= gen_steps[q]`. Use
///   [`crate::speculative::set_diffusion::order_to_gen_steps`] to convert a
///   sampled ordering to this buffer.
#[cfg(feature = "set_diffusion")]
fn forward_save_set_causal<'a>(
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
    gen_steps: &[u32],
    ctx: &'a mut ForwardSaveContext,
) -> ForwardActivations<'a> {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let seq_len = tokens.len().min(config.block_size);
    let scale = 1.0 / (hd as f32).sqrt();
    let l_total = config.n_layer;
    let bs = config.block_size;
    let nh = config.n_head;
    let mlp_h = config.mlp_hidden;

    debug_assert_eq!(gen_steps.len(), seq_len, "gen_steps length mismatch");

    ctx.reset(seq_len);

    let q_stride = bs * n;
    let kv_stride = bs * kvd;
    let aw_stride = bs * nh * bs;
    let mh_stride = bs * mlp_h;

    // Phase A0: Embeddings + the layer-0 double norm (identical to forward_save).
    for (p, &token) in tokens.iter().enumerate().take(seq_len) {
        katgpt_core::simd::simd_add_into(
            &mut ctx.embeddings[p * n..(p + 1) * n],
            &weights.wte[token * n..(token + 1) * n],
            &weights.wpe[p * n..(p + 1) * n],
        );
        ctx.x_buf[..n].copy_from_slice(&ctx.embeddings[p * n..(p + 1) * n]);
        rmsnorm(&mut ctx.x_buf);
        ctx.after_norm1[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf[..n]);
        rmsnorm(&mut ctx.x_buf);
        ctx.after_norm2[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf[..n]);
    }

    for (l, layer) in weights.layers.iter().take(l_total).enumerate() {
        let q_base = l * q_stride;
        let kv_base = l * kv_stride;
        let aw_base = l * aw_stride;
        let mh_base = l * mh_stride;
        let last = l + 1 == l_total;

        // Phase A(l): Q/K/V (mask-independent — same as forward_save).
        for p in 0..seq_len {
            if l == 0 {
                ctx.x_buf[..n].copy_from_slice(&ctx.after_norm2[p * n..(p + 1) * n]);
            } else {
                let h_prev = (l - 1) * q_stride;
                ctx.x_buf[..n]
                    .copy_from_slice(&ctx.hidden_all[h_prev + p * n..h_prev + (p + 1) * n]);
                rmsnorm(&mut ctx.x_buf);
                let xn = (l - 1) * q_stride;
                ctx.x_norm_rest[xn + p * n..xn + (p + 1) * n].copy_from_slice(&ctx.x_buf[..n]);
            }
            matmul(
                &mut ctx.q_all[q_base + p * n..q_base + (p + 1) * n],
                &layer.attn_wq,
                &ctx.x_buf,
                n,
                n,
            );
            matmul(
                &mut ctx.k_all[kv_base + p * kvd..kv_base + (p + 1) * kvd],
                &layer.attn_wk,
                &ctx.x_buf,
                kvd,
                n,
            );
            matmul(
                &mut ctx.v_all[kv_base + p * kvd..kv_base + (p + 1) * kvd],
                &layer.attn_wv,
                &ctx.x_buf,
                kvd,
                n,
            );
        }

        // Phase B(l): Set-causal attention with masked softmax.
        //
        // Mirrors `forward_set_causal_positions` Phase B: for each query q, compute
        // scores only for eligible positions (gen_steps[t] <= gen_steps[q]), apply
        // masked softmax (zero for ineligible), and accumulate the weighted value
        // sum. Saves into this layer's attn_out/attn_weights planes so backward()
        // sees the same layout as the bidirectional case (with zeros on masked
        // positions, which the softmax Jacobian handles correctly).
        for q in 0..seq_len {
            let q_gen_step = gen_steps[q];
            ctx.attn_scratch_out[..n].fill(0.0);

            for h in 0..nh {
                let kv_group = h * config.n_kv_head / nh;
                let q_off = h * hd;
                let kv_off = kv_group * hd;

                // Pass 1: scores for eligible positions only, track max for stability.
                let mut max_score = f32::NEG_INFINITY;
                for t in 0..seq_len {
                    if gen_steps[t] <= q_gen_step {
                        let dot = katgpt_core::simd::simd_dot_f32(
                            &ctx.q_all[q_base + q * n + q_off..q_base + q * n + q_off + hd],
                            &ctx.k_all[kv_base + t * kvd + kv_off..kv_base + t * kvd + kv_off + hd],
                            hd,
                        );
                        ctx.attn_scratch_scores[t] = dot * scale;
                        if ctx.attn_scratch_scores[t] > max_score {
                            max_score = ctx.attn_scratch_scores[t];
                        }
                    } else {
                        ctx.attn_scratch_scores[t] = 0.0;
                    }
                }

                // Pass 2: exp(score - max) for eligible positions, 0 for ineligible.
                let mut sum_exp = 0.0f32;
                for t in 0..seq_len {
                    if gen_steps[t] <= q_gen_step {
                        let e = (ctx.attn_scratch_scores[t] - max_score).exp();
                        ctx.attn_scratch_scores[t] = e;
                        sum_exp += e;
                    } else {
                        ctx.attn_scratch_scores[t] = 0.0;
                    }
                }

                // Normalize over eligible positions only.
                let inv_sum = 1.0 / sum_exp;
                for t in 0..seq_len {
                    if gen_steps[t] <= q_gen_step {
                        ctx.attn_scratch_scores[t] *= inv_sum;
                    }
                }

                // Persist attention weights (ineligible positions stay 0.0).
                ctx.attn_weights_all[aw_base + q * nh * seq_len + h * seq_len
                    ..aw_base + q * nh * seq_len + (h + 1) * seq_len]
                    .copy_from_slice(&ctx.attn_scratch_scores[..seq_len]);

                // Weighted value sum over eligible positions only.
                for t in 0..seq_len {
                    let s = ctx.attn_scratch_scores[t];
                    if s > 0.0 {
                        katgpt_core::simd::simd_fused_scale_acc(
                            &mut ctx.attn_scratch_out[q_off..q_off + hd],
                            &ctx.v_all[kv_base + t * kvd + kv_off..kv_base + t * kvd + kv_off + hd],
                            s,
                            hd,
                        );
                    }
                }
            }

            ctx.attn_out_all[q_base + q * n..q_base + (q + 1) * n]
                .copy_from_slice(&ctx.attn_scratch_out[..n]);
        }

        // Phase C(l): Output projection + residual + MLP + logits (identical to forward_save).
        let h_base = if l == 0 { 0 } else { (l - 1) * q_stride };
        for p in 0..seq_len {
            matmul(
                &mut ctx.x_proj_buf,
                &layer.attn_wo,
                &ctx.attn_out_all[q_base + p * n..q_base + (p + 1) * n],
                n,
                n,
            );
            if l == 0 {
                katgpt_core::simd::simd_add_inplace(
                    &mut ctx.x_proj_buf,
                    &ctx.after_norm1[p * n..(p + 1) * n],
                );
            } else {
                katgpt_core::simd::simd_add_inplace(
                    &mut ctx.x_proj_buf,
                    &ctx.hidden_all[h_base + p * n..h_base + (p + 1) * n],
                );
            }
            ctx.after_attn_res[q_base + p * n..q_base + (p + 1) * n]
                .copy_from_slice(&ctx.x_proj_buf[..n]);

            ctx.x_buf[..n].copy_from_slice(&ctx.x_proj_buf[..n]);
            rmsnorm(&mut ctx.x_proj_buf);
            ctx.after_mlp_norm[q_base + p * n..q_base + (p + 1) * n]
                .copy_from_slice(&ctx.x_proj_buf);
            matmul_relu(
                &mut ctx.mlp_hidden_all[mh_base + p * mlp_h..mh_base + (p + 1) * mlp_h],
                &layer.mlp_w1,
                &ctx.x_proj_buf,
                mlp_h,
                n,
            );
            matmul(
                &mut ctx.x_mlp_buf,
                &layer.mlp_w2,
                &ctx.mlp_hidden_all[mh_base + p * mlp_h..mh_base + (p + 1) * mlp_h],
                n,
                mlp_h,
            );
            katgpt_core::simd::simd_add_inplace(&mut ctx.x_mlp_buf[..n], &ctx.x_buf[..n]);
            ctx.hidden_all[q_base + p * n..q_base + (p + 1) * n].copy_from_slice(&ctx.x_mlp_buf);
            if last {
                matmul(
                    &mut ctx.logits_all[p * config.vocab_size..(p + 1) * config.vocab_size],
                    &weights.lm_head,
                    &ctx.x_mlp_buf,
                    config.vocab_size,
                    n,
                );
            }
        }
    }

    ForwardActivations {
        embeddings: &ctx.embeddings[..seq_len * n],
        after_norm1: &ctx.after_norm1[..seq_len * n],
        after_norm2: &ctx.after_norm2[..seq_len * n],
        x_norm_rest: &ctx.x_norm_rest,
        q: &ctx.q_all,
        k: &ctx.k_all,
        v: &ctx.v_all,
        attn_weights: &ctx.attn_weights_all,
        attn_out: &ctx.attn_out_all,
        after_attn_res: &ctx.after_attn_res,
        after_mlp_norm: &ctx.after_mlp_norm,
        mlp_hidden: &ctx.mlp_hidden_all,
        hidden_all: &ctx.hidden_all,
        logits: &ctx.logits_all[..seq_len * config.vocab_size],
        seq_len,
        n_layer: l_total,
    }
}

// ── Backward Helpers ──

/// RMSNorm backward: dx = (dy - y * mean(dy * y)) / rms
///
/// Allocating wrapper. Prefer [`rmsnorm_backward_into`] in hot paths.
#[allow(dead_code)]
#[inline]
fn rmsnorm_backward(x_input: &[f32], y_output: &[f32], dy: &[f32]) -> Vec<f32> {
    let n = x_input.len();
    let mut out = vec![0.0f32; n];
    rmsnorm_backward_into(x_input, y_output, dy, &mut out);
    out
}

/// Zero-alloc variant of [`rmsnorm_backward`] that writes into a pre-allocated buffer.
#[inline]
fn rmsnorm_backward_into(x_input: &[f32], y_output: &[f32], dy: &[f32], out: &mut [f32]) {
    let n = x_input.len();
    debug_assert!(out.len() >= n);
    let sum_sq = katgpt_core::simd::simd_sum_sq(x_input, n);
    let rms = (sum_sq / n as f32 + 1e-5).sqrt();
    let dot_dy_y = katgpt_core::simd::simd_dot_f32(dy, y_output, n);
    let mean_dy_y = dot_dy_y / n as f32;
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        out[i] = (dy[i] - y_output[i] * mean_dy_y) * inv_rms;
    }
}

/// Softmax backward: dx = y * (dy - dot(dy, y))
///
/// Allocating wrapper. Prefer [`softmax_backward_into`] in hot paths.
#[allow(dead_code)]
#[inline]
fn softmax_backward(weights: &[f32], dy: &[f32]) -> Vec<f32> {
    let n = weights.len();
    let mut out = vec![0.0f32; n];
    softmax_backward_into(weights, dy, &mut out);
    out
}

/// Zero-alloc variant of [`softmax_backward`] that writes into a pre-allocated buffer.
#[inline]
fn softmax_backward_into(weights: &[f32], dy: &[f32], out: &mut [f32]) {
    let n = weights.len();
    debug_assert!(out.len() >= n);
    let dot = katgpt_core::simd::simd_dot_f32(weights, dy, n);
    for i in 0..n {
        out[i] = weights[i] * (dy[i] - dot);
    }
}

/// Backward pass: compute gradients from saved activations.
///
/// Issue 869 T5: runs the three-phase backward once per layer, LAST layer
/// first. The residual-stream gradient `d_h` flows between layers through
/// `bctx.d_h_next`: layer l+1's Phase 3 hands `d_h_{l+1}` to layer l's
/// Phase 1 (where it plays the role `d_hf` had for the single-layer lane).
/// At `n_layer == 1` the layer loop runs once and every plane index is 0 —
/// the pre-869 op sequence, bit for bit.
fn backward(
    act: &ForwardActivations<'_>,
    weights: &TransformerWeights,
    tokens: &[usize],
    is_masked: &[bool],
    config: &Config,
    bctx: &mut BackwardContext,
) {
    let seq_len = act.seq_len;
    let l_total = act.n_layer;
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let vocab = config.vocab_size;
    let mlp_h = config.mlp_hidden;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let scale = 1.0 / (hd as f32).sqrt();
    let bs = config.block_size;

    // Activation-plane strides (block_size-capacity — matches ForwardSaveContext).
    let q_stride = bs * n;
    let kv_stride = bs * kvd;
    let aw_stride = bs * n_head * bs;
    let mh_stride = bs * mlp_h;
    // Weight-gradient plane strides (per-layer matrix lengths).
    let wq_stride = n * n;
    let wk_stride = kvd * n;
    let wo_stride = n * n;
    let w1_stride = mlp_h * n;
    let w2_stride = n * mlp_h;

    // Reuse pre-allocated gradient accumulator — clear instead of allocating 9 Vecs
    let grads = &mut bctx.grads;
    grads.wte.fill(0.0);
    grads.wpe.fill(0.0);
    grads.lm_head.fill(0.0);
    grads.attn_wq.fill(0.0);
    grads.attn_wk.fill(0.0);
    grads.attn_wv.fill(0.0);
    grads.attn_wo.fill(0.0);
    grads.mlp_w1.fill(0.0);
    grads.mlp_w2.fill(0.0);

    // Reuse pre-allocated intermediate gradient buffers (Issue 109; T5 planes)
    for l in 0..l_total {
        let q_base = l * q_stride;
        let kv_base = l * kv_stride;
        bctx.d_attn_out[q_base..q_base + seq_len * n].fill(0.0);
        bctx.d_q[q_base..q_base + seq_len * n].fill(0.0);
        bctx.d_k[kv_base..kv_base + seq_len * kvd].fill(0.0);
        bctx.d_v[kv_base..kv_base + seq_len * kvd].fill(0.0);
        bctx.d_x_norm[q_base..q_base + seq_len * n].fill(0.0);
        bctx.d_aar_saved[q_base..q_base + seq_len * n].fill(0.0);
    }
    bctx.d_h_next[..seq_len * n].fill(0.0);

    let any_masked = is_masked.iter().any(|&m| m);

    for l in (0..l_total).rev() {
        let layer = &weights.layers[l];
        let q_base = l * q_stride;
        let kv_base = l * kv_stride;
        let aw_base = l * aw_stride;
        let mh_base = l * mh_stride;
        let last = l + 1 == l_total;
        // Weight-gradient plane bases.
        let wq = l * wq_stride;
        let wk = l * wk_stride;
        let wv = l * wk_stride;
        let wo = l * wo_stride;
        let w1 = l * w1_stride;
        let w2 = l * w2_stride;

        // ── Phase 1(l): LM head (last layer only) → MLP → attention output projection ──
        // Issue 869 T5: the unmasked-position skip is valid ONLY at the readout
        // (last) layer, where an unmasked position's stream feeds its own
        // logits (no loss term) and nothing else. At inner layers, h_{l+1}[t]
        // for an UNMASKED t still carries gradient — downstream attention
        // reads the k/v derived from it at masked queries — so every position
        // is processed. At `n_layer == 1` the layer is always last and the
        // skip is exactly the pre-869 semantics, bit for bit.
        for p in 0..seq_len {
            if last && !is_masked[p] {
                continue;
            }

            if last {
                // Cross-entropy backward: d_logit[i] = softmax(logit)[i] - (1 if i==target else 0)
                let logits_p = &act.logits[p * vocab..(p + 1) * vocab];
                let target = tokens[p];
                let max_l = katgpt_core::simd::simd_max_f32(logits_p);
                // Compute exp(logits - max) once into d_logits using SIMD, then reuse for sum and gradient
                bctx.d_logits[..vocab].copy_from_slice(logits_p);
                katgpt_core::simd::simd_add_scalar_inplace(&mut bctx.d_logits[..vocab], -max_l);
                katgpt_core::simd::simd_exp_inplace(&mut bctx.d_logits[..vocab]);
                let sum_exp = katgpt_core::simd::simd_sum_f32(&bctx.d_logits[..vocab]);
                let inv_sum = 1.0 / sum_exp;
                katgpt_core::simd::simd_scale_inplace(&mut bctx.d_logits[..vocab], inv_sum);
                bctx.d_logits[target] -= 1.0;

                // LM Head: d_lm_head += outer(d_logits, hidden_final)
                let hf_base = (l_total - 1) * q_stride;
                let hf = &act.hidden_all[hf_base + p * n..hf_base + (p + 1) * n];
                katgpt_core::simd::simd_outer_product_acc(
                    &mut grads.lm_head,
                    &bctx.d_logits[..vocab],
                    hf,
                    vocab,
                    n,
                );

                // d_hidden_final = lm_head^T @ d_logits (row-wise dot products)
                bctx.d_hf[..n].fill(0.0);
                for i in 0..vocab {
                    let grad = bctx.d_logits[i];
                    katgpt_core::simd::simd_fused_scale_acc(
                        &mut bctx.d_hf[..n],
                        &weights.lm_head[i * n..(i + 1) * n],
                        grad,
                        n,
                    );
                }
            } else {
                // Inner layer: the stream gradient comes from layer l+1's Phase 3.
                bctx.d_hf[..n].copy_from_slice(&bctx.d_h_next[p * n..(p + 1) * n]);
            }

            // Residual: hidden_final = after_mlp + after_attn_res
            // d_after_mlp = d_hf, d_after_attn_res starts as d_hf
            bctx.d_an1[..n].copy_from_slice(&bctx.d_hf[..n]); // reuse d_an1 as d_after_attn_res temporarily

            // MLP w2: d_w2 += outer(d_after_mlp, mlp_hidden)
            let mh = &act.mlp_hidden[mh_base + p * mlp_h..mh_base + (p + 1) * mlp_h];
            katgpt_core::simd::simd_outer_product_acc(
                &mut grads.mlp_w2[w2..w2 + w2_stride],
                &bctx.d_hf[..n],
                mh,
                n,
                mlp_h,
            );
            // d_mlp_hidden = w2^T @ d_after_mlp, then ReLU backward
            bctx.d_mh[..mlp_h].fill(0.0);
            for i in 0..n {
                let grad = bctx.d_hf[i];
                katgpt_core::simd::simd_fused_scale_acc(
                    &mut bctx.d_mh[..mlp_h],
                    &layer.mlp_w2[i * mlp_h..(i + 1) * mlp_h],
                    grad,
                    mlp_h,
                );
            }
            // ReLU backward (branch-free: mask grad to zero when pre-activation ≤ 0)
            for j in 0..mlp_h {
                bctx.d_mh[j] *= (mh[j] > 0.0) as usize as f32;
            }

            // MLP w1: d_w1 += outer(d_mh, after_mlp_norm)
            let amn = &act.after_mlp_norm[q_base + p * n..q_base + (p + 1) * n];
            katgpt_core::simd::simd_outer_product_acc(
                &mut grads.mlp_w1[w1..w1 + w1_stride],
                &bctx.d_mh[..mlp_h],
                amn,
                mlp_h,
                n,
            );
            // d_after_mlp_norm = w1^T @ d_mh
            bctx.d_amn[..n].fill(0.0);
            for i in 0..mlp_h {
                let grad = bctx.d_mh[i];
                katgpt_core::simd::simd_fused_scale_acc(
                    &mut bctx.d_amn[..n],
                    &layer.mlp_w1[i * n..(i + 1) * n],
                    grad,
                    n,
                );
            }

            // RMSNorm backward (after_attn_res → after_mlp_norm)
            let aar = &act.after_attn_res[q_base + p * n..q_base + (p + 1) * n];
            rmsnorm_backward_into(aar, amn, &bctx.d_amn, &mut bctx.d_rmsnorm_buf);
            katgpt_core::simd::simd_add_inplace(&mut bctx.d_an1[..n], &bctx.d_rmsnorm_buf[..n]); // d_after_attn_res = d_hf + d_aar_from_mlp

            // Save d_after_attn_res for Phase 3
            bctx.d_aar_saved[q_base + p * n..q_base + (p + 1) * n]
                .copy_from_slice(&bctx.d_an1[..n]);

            // Attention output projection: d_wo += outer(d_after_attn_res, attn_out)
            let ao = &act.attn_out[q_base + p * n..q_base + (p + 1) * n];
            katgpt_core::simd::simd_outer_product_acc(
                &mut grads.attn_wo[wo..wo + wo_stride],
                &bctx.d_an1[..n],
                ao,
                n,
                n,
            );
            // d_attn_out = wo^T @ d_after_attn_res
            for i in 0..n {
                let grad = bctx.d_an1[i];
                katgpt_core::simd::simd_fused_scale_acc(
                    &mut bctx.d_attn_out[q_base + p * n..q_base + (p + 1) * n],
                    &layer.attn_wo[i * n..(i + 1) * n],
                    grad,
                    n,
                );
            }
        }

        // ── Phase 2(l): Attention backward ──
        // (Same T5 rule as Phase 1: unmasked positions participate at inner
        // layers — their q/attn_out carry gradient via the downstream stream.)
        for p in 0..seq_len {
            if last && !is_masked[p] {
                continue;
            }
            let d_ao = &bctx.d_attn_out[q_base + p * n..q_base + (p + 1) * n];
            let aw = &act.attn_weights
                [aw_base + p * n_head * seq_len..aw_base + (p + 1) * n_head * seq_len];

            for h in 0..n_head {
                let kv_group = h * n_kv / n_head;
                let q_off = h * hd;
                let kv_off = kv_group * hd;

                // `d_attn_out[p, h]` is invariant across `t` — slice it once instead of
                // re-slicing (and re-bounds-checking) it inside all three `t` loops.
                let d_ao_h = &d_ao[q_off..q_off + hd];

                // d_raw_weights[t] = dot(d_attn_out[h], v[t,h])
                // No fill needed: d_raw[t] is assigned (not accumulated) below.
                for t in 0..seq_len {
                    bctx.d_raw[t] = katgpt_core::simd::simd_dot_f32(
                        d_ao_h,
                        &act.v[kv_base + t * kvd + kv_off..kv_base + t * kvd + kv_off + hd],
                        hd,
                    );
                }

                // Softmax backward
                let w_h = &aw[h * seq_len..(h + 1) * seq_len];
                softmax_backward_into(
                    w_h,
                    &bctx.d_raw[..seq_len],
                    &mut bctx.d_softmax_buf[..seq_len],
                );
                let d_scores = &bctx.d_softmax_buf[..seq_len];

                // Fused d_v / d_q / d_k accumulation — one pass over `t` instead of three.
                //
                // GRADIENT SAFETY (bit-identical): `d_q[p, h]` is the only accumulator
                // whose float addition order matters here (all `t` write the same row),
                // and it still sees `t` strictly ascending — exactly as in the old
                // standalone loop. `d_v[t, h]` and `d_k[t, h]` each target a distinct
                // row per `t`, so interleaving them with the `d_q` reduction cannot
                // reassociate anything. `d_scores[t] * scale` is now computed once per
                // `t` instead of twice (same expression, same value).
                let q_ph = &act.q[q_base + p * n + q_off..q_base + p * n + q_off + hd];
                let d_q_h = &mut bctx.d_q[q_base + p * n + q_off..q_base + p * n + q_off + hd];
                for t in 0..seq_len {
                    let ds_scaled = d_scores[t] * scale;
                    // d_v[t] += weights[t] * d_attn_out[h]
                    katgpt_core::simd::simd_fused_scale_acc(
                        &mut bctx.d_v[kv_base + t * kvd + kv_off..kv_base + t * kvd + kv_off + hd],
                        d_ao_h,
                        w_h[t],
                        hd,
                    );
                    // d_q[h] += d_scores[t] * k[t,h] * scale
                    katgpt_core::simd::simd_fused_scale_acc(
                        d_q_h,
                        &act.k[kv_base + t * kvd + kv_off..kv_base + t * kvd + kv_off + hd],
                        ds_scaled,
                        hd,
                    );
                    // d_k[t,h] += d_scores[t] * q[p,h] * scale
                    katgpt_core::simd::simd_fused_scale_acc(
                        &mut bctx.d_k[kv_base + t * kvd + kv_off..kv_base + t * kvd + kv_off + hd],
                        q_ph,
                        ds_scaled,
                        hd,
                    );
                }
            }
        }

        // ── Phase 3(l): QKV projections → input norm → stream handoff ──
        // When any position was masked, bidirectional attention propagated non-zero
        // d_k/d_v to ALL positions in Phase 2 (every masked query attends to every
        // key). When none were masked, Phase 2 was a no-op and Phase 3 should be too.
        if any_masked {
            for p in 0..seq_len {
                // d_wq, d_wk, d_wv (this layer's planes)
                let xn = if l == 0 {
                    &act.after_norm2[p * n..(p + 1) * n]
                } else {
                    &act.x_norm_rest[q_base - q_stride + p * n..q_base - q_stride + (p + 1) * n]
                };
                katgpt_core::simd::simd_outer_product_acc(
                    &mut grads.attn_wq[wq..wq + wq_stride],
                    &bctx.d_q[q_base + p * n..q_base + p * n + n],
                    xn,
                    n,
                    n,
                );
                katgpt_core::simd::simd_outer_product_acc(
                    &mut grads.attn_wk[wk..wk + wk_stride],
                    &bctx.d_k[kv_base + p * kvd..kv_base + p * kvd + kvd],
                    xn,
                    kvd,
                    n,
                );
                katgpt_core::simd::simd_outer_product_acc(
                    &mut grads.attn_wv[wv..wv + wk_stride],
                    &bctx.d_v[kv_base + p * kvd..kv_base + p * kvd + kvd],
                    xn,
                    kvd,
                    n,
                );

                // d_x_norm = wq^T @ d_q + wk^T @ d_k + wv^T @ d_v
                bctx.d_an2[..n].fill(0.0);
                for i in 0..n {
                    let grad = bctx.d_q[q_base + p * n + i];
                    katgpt_core::simd::simd_fused_scale_acc(
                        &mut bctx.d_an2[..n],
                        &layer.attn_wq[i * n..(i + 1) * n],
                        grad,
                        n,
                    );
                }
                for i in 0..kvd {
                    let gk = bctx.d_k[kv_base + p * kvd + i];
                    let gv = bctx.d_v[kv_base + p * kvd + i];
                    katgpt_core::simd::simd_fused_scale_acc(
                        &mut bctx.d_an2[..n],
                        &layer.attn_wk[i * n..(i + 1) * n],
                        gk,
                        n,
                    );
                    katgpt_core::simd::simd_fused_scale_acc(
                        &mut bctx.d_an2[..n],
                        &layer.attn_wv[i * n..(i + 1) * n],
                        gv,
                        n,
                    );
                }
                bctx.d_x_norm[q_base + p * n..q_base + (p + 1) * n]
                    .copy_from_slice(&bctx.d_an2[..n]);
            }
        }

        // Stream handoff + (layer 0) embeddings backward: d_h_l = d_aar_l +
        // rmsnorm_backward(h_l, x_norm_l, d_x_norm_l). For l == 0, h_0 =
        // after_norm1 and x_norm_0 = after_norm2 (the double-norm quirk), and
        // d_h_0 continues into the embedding backward (wte/wpe).
        for p in 0..seq_len {
            bctx.d_an1[..n].fill(0.0);

            // From the layer-input norm backward.
            // rmsnorm_backward on all-zero dy produces all-zero output (mean_dy_y=0,
            // out[i]=dy[i]*inv_rms=0), so the zero-check is unnecessary overhead
            // in the hot path where an2_grad is non-zero.
            let xn_grad = &bctx.d_x_norm[q_base + p * n..q_base + (p + 1) * n];
            if l == 0 {
                let an1 = &act.after_norm1[p * n..(p + 1) * n];
                let an2 = &act.after_norm2[p * n..(p + 1) * n];
                rmsnorm_backward_into(an1, an2, xn_grad, &mut bctx.d_rmsnorm_buf);
            } else {
                let h_prev = q_base - q_stride;
                let h_l = &act.hidden_all[h_prev + p * n..h_prev + (p + 1) * n];
                let xnl = &act.x_norm_rest[h_prev + p * n..h_prev + (p + 1) * n];
                rmsnorm_backward_into(h_l, xnl, xn_grad, &mut bctx.d_rmsnorm_buf);
            }
            katgpt_core::simd::simd_add_inplace(&mut bctx.d_an1[..n], &bctx.d_rmsnorm_buf[..n]);

            // From residual: after_attn_res = wo @ attn_out + h_l
            // d_h_l += d_after_attn_res (saved from Phase 1)
            katgpt_core::simd::simd_add_inplace(
                &mut bctx.d_an1[..n],
                &bctx.d_aar_saved[q_base + p * n..q_base + p * n + n],
            );

            if l == 0 {
                // RMSNorm backward (embeddings → after_norm1)
                let emb = &act.embeddings[p * n..(p + 1) * n];
                let an1 = &act.after_norm1[p * n..(p + 1) * n];
                rmsnorm_backward_into(emb, an1, &bctx.d_an1, &mut bctx.d_rmsnorm_buf);

                // d_wte[token] += d_emb, d_wpe[p] += d_emb
                let token = tokens[p];
                katgpt_core::simd::simd_add_inplace(
                    &mut grads.wte[token * n..token * n + n],
                    &bctx.d_rmsnorm_buf[..n],
                );
                katgpt_core::simd::simd_add_inplace(
                    &mut grads.wpe[p * n..p * n + n],
                    &bctx.d_rmsnorm_buf[..n],
                );
            } else {
                bctx.d_h_next[p * n..(p + 1) * n].copy_from_slice(&bctx.d_an1[..n]);
            }
        }
    }
}

/// SGD update: w -= lr * grad
///
/// Issue 869 T5: updates EVERY layer's matrices — the gradient planes are
/// sliced by the weights' own per-layer lengths (identical across layers by
/// construction).
#[inline]
fn sgd_update(weights: &mut TransformerWeights, grads: &TrainingGradients, lr: f32) {
    // SIMD-fused: w[i] = 1.0*w[i] + (-lr)*g[i] = w[i] - lr*g[i]
    let neg_lr = -lr;
    katgpt_core::simd::simd_fused_decay_write(&mut weights.wte, 1.0, &grads.wte, neg_lr);
    katgpt_core::simd::simd_fused_decay_write(&mut weights.wpe, 1.0, &grads.wpe, neg_lr);
    katgpt_core::simd::simd_fused_decay_write(&mut weights.lm_head, 1.0, &grads.lm_head, neg_lr);
    for (l, layer) in weights.layers.iter_mut().enumerate() {
        let n2 = layer.attn_wq.len(); // n * n
        let kvn = layer.attn_wk.len(); // kvd * n
        let w1 = layer.mlp_w1.len(); // mlp_hidden * n
        let w2 = layer.mlp_w2.len(); // n * mlp_hidden
        katgpt_core::simd::simd_fused_decay_write(
            &mut layer.attn_wq,
            1.0,
            &grads.attn_wq[l * n2..(l + 1) * n2],
            neg_lr,
        );
        katgpt_core::simd::simd_fused_decay_write(
            &mut layer.attn_wk,
            1.0,
            &grads.attn_wk[l * kvn..(l + 1) * kvn],
            neg_lr,
        );
        katgpt_core::simd::simd_fused_decay_write(
            &mut layer.attn_wv,
            1.0,
            &grads.attn_wv[l * kvn..(l + 1) * kvn],
            neg_lr,
        );
        katgpt_core::simd::simd_fused_decay_write(
            &mut layer.attn_wo,
            1.0,
            &grads.attn_wo[l * n2..(l + 1) * n2],
            neg_lr,
        );
        katgpt_core::simd::simd_fused_decay_write(
            &mut layer.mlp_w1,
            1.0,
            &grads.mlp_w1[l * w1..(l + 1) * w1],
            neg_lr,
        );
        katgpt_core::simd::simd_fused_decay_write(
            &mut layer.mlp_w2,
            1.0,
            &grads.mlp_w2[l * w2..(l + 1) * w2],
            neg_lr,
        );
    }
}

/// Compute cross-entropy loss on masked positions.
/// Uses pre-allocated scratch buffer from `bctx.loss_exp_buf` to avoid per-call allocation.
#[inline]
fn masked_loss_into(
    logits: &[f32],
    targets: &[usize],
    is_masked: &[bool],
    vocab: usize,
    _averaging: LossAveraging,
    exp_buf: &mut [f32],
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for (p, &masked) in is_masked.iter().enumerate() {
        if !masked {
            continue;
        }
        let l = &logits[p * vocab..(p + 1) * vocab];
        // Log-softmax: log_softmax[i] = x[i] - max - ln(Σ exp(x - max))
        let max_l = katgpt_core::simd::simd_max_f32(l);
        exp_buf[..vocab].copy_from_slice(l);
        katgpt_core::simd::simd_add_scalar_inplace(&mut exp_buf[..vocab], -max_l);
        katgpt_core::simd::simd_exp_inplace(&mut exp_buf[..vocab]);
        let sum_exp = katgpt_core::simd::simd_sum_f32(&exp_buf[..vocab]);
        let log_sum_exp = sum_exp.ln();
        total -= l[targets[p]] - max_l - log_sum_exp;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// Allocating wrapper — prefer `masked_loss_into` in hot paths.
#[allow(dead_code)]
fn masked_loss(
    logits: &[f32],
    targets: &[usize],
    is_masked: &[bool],
    vocab: usize,
    averaging: LossAveraging,
) -> f32 {
    let mut exp_buf = vec![0.0f32; vocab];
    masked_loss_into(logits, targets, is_masked, vocab, averaging, &mut exp_buf)
}

/// Measure accuracy: fraction of correctly predicted masked tokens.
pub fn evaluate_accuracy(
    weights: &TransformerWeights,
    test_data: &[Vec<usize>],
    config: &Config,
    mask_ratio: f32,
    rng: &mut Rng,
) -> f32 {
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut corrupted_buf = Vec::with_capacity(config.block_size);
    let mut is_masked_buf = Vec::with_capacity(config.block_size);
    let mut positions_buf = Vec::with_capacity(config.block_size);
    // OPT: pre-allocate bidirectional context to avoid per-sample heap allocation
    let mut bctx = BidirectionalContext::new(config);
    for tokens in test_data {
        let n_mask = corrupt_block_into(
            tokens,
            mask_ratio,
            config.mask_token,
            rng,
            &mut corrupted_buf,
            &mut is_masked_buf,
            &mut positions_buf,
        );
        if n_mask == 0 {
            continue;
        }
        forward_bidirectional_positions_into(weights, &corrupted_buf, config, &mut bctx);
        let vocab = config.vocab_size;
        for (p, &masked) in is_masked_buf.iter().enumerate() {
            if !masked {
                continue;
            }
            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            // Single-pass argmax: fuses max-finding and index-recovery into one
            // traversal (vs the old two-pass simd_max_f32 + position scan).
            let (predicted, _) = katgpt_core::simd::simd_argmax_f32(logits_p);
            if predicted == tokens[p] {
                correct += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    }
}

/// Teacher-forced masked NLL (nats/token) on held-out blocks — the token-level
/// quality axis for the real-text corpus (Plan 601). Mirrors
/// [`evaluate_accuracy`]'s corruption loop but integrates
/// `-ln P(target | corrupted context)` over the masked positions instead of
/// counting argmax hits, so it has resolution even where accuracy saturates.
/// Deliberately re-implemented against the forward logits rather than reusing
/// [`masked_loss_into`]: the eval instrument must not silently inherit the
/// training loss's averaging/temperature conventions.
pub fn evaluate_masked_nll(
    weights: &TransformerWeights,
    test_data: &[Vec<usize>],
    config: &Config,
    mask_ratio: f32,
    rng: &mut Rng,
) -> f32 {
    let mut total_nll = 0.0f32;
    let mut n_masked = 0usize;
    let mut corrupted_buf = Vec::with_capacity(config.block_size);
    let mut is_masked_buf = Vec::with_capacity(config.block_size);
    let mut positions_buf = Vec::with_capacity(config.block_size);
    let mut bctx = BidirectionalContext::new(config);
    let vocab = config.vocab_size;
    for tokens in test_data {
        let n_mask = corrupt_block_into(
            tokens,
            mask_ratio,
            config.mask_token,
            rng,
            &mut corrupted_buf,
            &mut is_masked_buf,
            &mut positions_buf,
        );
        if n_mask == 0 {
            continue;
        }
        forward_bidirectional_positions_into(weights, &corrupted_buf, config, &mut bctx);
        for (p, &masked) in is_masked_buf.iter().enumerate() {
            if !masked {
                continue;
            }
            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let target = tokens[p];
            let mut m = f32::NEG_INFINITY;
            for &l in logits_p {
                m = m.max(l);
            }
            let mut sum = 0.0f32;
            for &l in logits_p {
                sum += (l - m).exp();
            }
            let lse = m + sum.ln();
            total_nll += lse - logits_p[target];
            n_masked += 1;
        }
    }
    if n_masked == 0 {
        0.0
    } else {
        total_nll / n_masked as f32
    }
}

/// Generate pattern-based dataset with learnable structure for dLLM training.
///
/// Each sequence follows an alternating pattern: [a, b, a, b, ...].
/// This gives bidirectional attention a clear signal — a masked position can
/// always be inferred from its partner at the same parity (position 0 ↔ 2,
/// position 1 ↔ 3, etc.).
///
/// The model learns the **structure** (alternating), not specific pairs,
/// so it generalizes to unseen (a, b) combinations at test time.
pub fn generate_pattern_dataset(
    rng: &mut Rng,
    n_sequences: usize,
    seq_len: usize,
    effective_vocab: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(n_sequences);
    let mut seq = Vec::with_capacity(seq_len);
    for _ in 0..n_sequences {
        let a = (rng.next() as usize) % effective_vocab;
        let mut b = (rng.next() as usize) % effective_vocab;
        // Reject a == b (constant sequences). Issue 049: a constant sequence
        // [c,c,...,c] teaches the model nothing about the alternating pattern
        // and corrupts FUNCATTN's learned basis with a degenerate direction.
        // Bump b to the next token; preserves the rest of the PRNG stream so
        // downstream RNG state is byte-identical to the pre-fix behavior.
        if effective_vocab > 1 && b == a {
            b = (b + 1) % effective_vocab;
        }
        seq.clear();
        seq.extend((0..seq_len).map(|i| if i % 2 == 0 { a } else { b }));
        // Clone seq into output and reuse the allocation for next iteration.
        // This avoids an extra allocation per iteration: the new Vec takes
        // ownership of seq's allocation via std::mem::take, and seq gets a
        // fresh (pre-reserved) empty Vec for the next loop iteration.
        out.push(std::mem::take(&mut seq));
        seq.reserve(seq_len);
    }
    out
}

/// Train mini dLLM and return (weights, loss_history).
/// Prints progress every 100 epochs.
pub fn train_mini_dllm(
    config: &Config,
    train_data: &[Vec<usize>],
    test_data: &[Vec<usize>],
    n_epochs: usize,
    lr: f32,
    mask_ratio: f32,
    seed: u64,
) -> (TransformerWeights, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let weights = TransformerWeights::new(config, &mut rng);
    train_mini_dllm_epochs(config, weights, train_data, test_data, n_epochs, lr, mask_ratio, rng)
}

/// Continue-train mini dLLM from EXISTING weights (riir-train Plan 416 T1.2:
/// the planted-forgetting arms' stage-2 entry — pretrain on a mix, then
/// re-enter the trainer with a different data mix; [`train_mini_dllm`] always
/// re-initializes). Identical epoch loop; `seed` seeds the shuffle/corruption
/// stream only (no weight-initialization draws are consumed from it).
pub fn train_mini_dllm_from(
    config: &Config,
    weights: TransformerWeights,
    train_data: &[Vec<usize>],
    test_data: &[Vec<usize>],
    n_epochs: usize,
    lr: f32,
    mask_ratio: f32,
    seed: u64,
) -> (TransformerWeights, Vec<f32>) {
    train_mini_dllm_epochs(
        config,
        weights,
        train_data,
        test_data,
        n_epochs,
        lr,
        mask_ratio,
        Rng::new(seed),
    )
}

/// The shared epoch loop behind [`train_mini_dllm`] and
/// [`train_mini_dllm_from`]. `weights` and `rng` enter by value — the rng
/// stream arrives positioned after any weight-initialization draws, so both
/// entry points keep their historical shuffle/corruption streams.
#[allow(clippy::too_many_arguments)]
fn train_mini_dllm_epochs(
    config: &Config,
    mut weights: TransformerWeights,
    train_data: &[Vec<usize>],
    test_data: &[Vec<usize>],
    n_epochs: usize,
    lr: f32,
    mask_ratio: f32,
    mut rng: Rng,
) -> (TransformerWeights, Vec<f32>) {
    let mut loss_history = Vec::with_capacity(n_epochs);
    let mut fwd_ctx = ForwardSaveContext::new(config);
    let mut bwd_ctx = BackwardContext::new(config);
    let mut corrupted_buf = Vec::with_capacity(config.block_size);
    let mut is_masked_buf = Vec::with_capacity(config.block_size);
    let mut positions_buf = Vec::with_capacity(config.block_size);

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_samples = 0usize;

        // Shuffle training data in-place
        for i in (1..indices.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let tokens = &train_data[idx];
            let n_mask = corrupt_block_into(
                tokens,
                mask_ratio,
                config.mask_token,
                &mut rng,
                &mut corrupted_buf,
                &mut is_masked_buf,
                &mut positions_buf,
            );

            // Skip if nothing masked
            if n_mask == 0 {
                continue;
            }

            let act = forward_save(&weights, &corrupted_buf, config, &mut fwd_ctx);
            let loss = masked_loss_into(
                act.logits,
                tokens,
                &is_masked_buf,
                config.vocab_size,
                LossAveraging::Global,
                &mut bwd_ctx.loss_exp_buf,
            );
            backward(&act, &weights, tokens, &is_masked_buf, config, &mut bwd_ctx);
            sgd_update(&mut weights, &bwd_ctx.grads, lr);

            epoch_loss += loss;
            n_samples += 1;
        }

        let avg_loss = if n_samples > 0 {
            epoch_loss / n_samples as f32
        } else {
            0.0
        };
        loss_history.push(avg_loss);

        if epoch % 100 == 0 || epoch == n_epochs - 1 {
            let acc = evaluate_accuracy(&weights, test_data, config, mask_ratio, &mut rng);
            eprintln!(
                "Epoch {:>4}/{}: loss={:.4} test_acc={:.1}%",
                epoch,
                n_epochs,
                avg_loss,
                acc * 100.0
            );
        }
    }

    (weights, loss_history)
}

// ═══════════════════════════════════════════════════════════════
// SW-SetDLM Training (set-causal attention) — moved to `set_causal`
// ═══════════════════════════════════════════════════════════════
//
// Issue 813 (2026-09-17): the set-causal train/eval block moved to
// [`set_causal`] — the custom-order reveal seam made it its own module
// (mod.rs was at the 2048-line soft limit). The re-exports below preserve
// every historical `crate::dllm::{train_mini_set_causal,
// evaluate_set_causal_nelbo}` path byte-identically (the Plan 402/403
// move-and-re-export precedent); the seam entry points live beside them.
#[cfg(feature = "set_diffusion")]
mod set_causal;

#[cfg(feature = "set_diffusion")]
pub use set_causal::{
    SetCausalGenStepsFn, evaluate_set_causal_denoiser_nll_with_gen_steps,
    evaluate_set_causal_nelbo, evaluate_set_causal_nelbo_with_gen_steps, train_mini_set_causal,
    train_mini_set_causal_denoiser_with_gen_steps, train_mini_set_causal_with_gen_steps,
};

// ═══════════════════════════════════════════════════════════════
// Plan 078 T3: Adaptive Noise Schedule Training
// ═══════════════════════════════════════════════════════════════

/// Train mini dLLM with adaptive noise schedule (RePlaid variance-minimized).
///
/// Identical to [`train_mini_dllm`] except:
/// - Per-block mask ratios come from [`AdaptiveNoiseSchedule::ratios`]
/// - Each sample cycles through blocks via modulo counter
/// - Losses are recorded per block via [`AdaptiveNoiseSchedule::record_step_loss`]
/// - Ratios are adapted at epoch boundaries via [`AdaptiveNoiseSchedule::adapt_ratios`]
#[cfg(feature = "replaid_schedules")]
pub fn train_mini_dllm_adaptive(
    config: &Config,
    train_data: &[Vec<usize>],
    test_data: &[Vec<usize>],
    n_epochs: usize,
    lr: f32,
    schedule: &mut AdaptiveNoiseSchedule,
    seed: u64,
) -> (TransformerWeights, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut weights = TransformerWeights::new(config, &mut rng);
    let mut loss_history = Vec::with_capacity(n_epochs);
    let n_blocks = schedule.ratios().len().max(1);
    let mut fwd_ctx = ForwardSaveContext::new(config);
    let mut bwd_ctx = BackwardContext::new(config);
    let mut corrupted_buf = Vec::with_capacity(config.block_size);
    let mut is_masked_buf = Vec::with_capacity(config.block_size);
    let mut positions_buf = Vec::with_capacity(config.block_size);

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_samples = 0usize;
        let mut sample_counter: usize = 0;

        // Shuffle training data in-place
        for i in (1..indices.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let tokens = &train_data[idx];

            // Cycle through schedule blocks using modulo counter
            let block_idx = sample_counter % n_blocks;
            let mask_ratio = schedule.ratios()[block_idx];

            let n_mask = corrupt_block_into(
                tokens,
                mask_ratio,
                config.mask_token,
                &mut rng,
                &mut corrupted_buf,
                &mut is_masked_buf,
                &mut positions_buf,
            );

            // Skip if nothing masked
            if n_mask == 0 {
                sample_counter += 1;
                continue;
            }

            let act = forward_save(&weights, &corrupted_buf, config, &mut fwd_ctx);
            let loss = masked_loss_into(
                act.logits,
                tokens,
                &is_masked_buf,
                config.vocab_size,
                LossAveraging::Global,
                &mut bwd_ctx.loss_exp_buf,
            );

            // Record per-block loss for adaptive schedule
            schedule.record_step_loss(block_idx, loss);

            backward(&act, &weights, tokens, &is_masked_buf, config, &mut bwd_ctx);
            sgd_update(&mut weights, &bwd_ctx.grads, lr);

            epoch_loss += loss;
            n_samples += 1;
            sample_counter += 1;
        }

        let avg_loss = if n_samples > 0 {
            epoch_loss / n_samples as f32
        } else {
            0.0
        };
        loss_history.push(avg_loss);

        // Adapt schedule ratios at epoch boundary
        schedule.adapt_ratios();

        if epoch % 100 == 0 || epoch == n_epochs - 1 {
            // Use the mean adapted ratio for evaluation
            let adapted = schedule.ratios();
            let eval_ratio = adapted.iter().copied().sum::<f32>() / adapted.len().max(1) as f32;
            let acc = evaluate_accuracy(&weights, test_data, config, eval_ratio, &mut rng);
            eprintln!(
                "Epoch {:>4}/{}: loss={:.4} test_acc={:.1}% schedule_adapt={} ratios=[{}]",
                epoch,
                n_epochs,
                avg_loss,
                acc * 100.0,
                schedule.adaptations(),
                adapted
                    .iter()
                    .map(|r| format!("{r:.3}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    (weights, loss_history)
}

// ═══════════════════════════════════════════════════════════════
// Block-Causal Forward — re-export from katgpt-forward (Plan 402)
// ═══════════════════════════════════════════════════════════════
//
// Plan 402 (2026-07-06): `forward_block_causal_positions` moved to
// `katgpt_forward::forward_positions`. Re-exported here so every historical
// `crate::dllm::forward_block_causal_positions` import path (notably the
// now-moved comparison tests + D2F training code) continues to resolve.
#[cfg(feature = "dllm")]
pub use katgpt_forward::forward_positions::forward_block_causal_positions;

// ═══════════════════════════════════════════════════════════════
// Set-Causal Attention Forward (Research 376 Phase 0 T0.2)
// ═══════════════════════════════════════════════════════════════
//
// Plan 401 (2026-07-06): The function body + 5 of 7 Research 376 T0.2 tests
// moved to `crates/katgpt-forward/src/forward_set_causal.rs` (the function is
// pure inference — no gradients/backprop/loss — despite the old "Root-resident
// by design (Issue 033 §C, Option C)" comment, which was obsolete; Issue 033
// does not exist and all cited blockers now resolve to leaf crates).
//
// The 2 comparison tests that additionally need `forward_block_causal_positions`
// / `forward_bidirectional_positions` stay here (those siblings are NOT yet
// extracted — deferred to Plan 402). They call this function via the re-export
// below, so their source is unchanged.
//
// Re-export preserves every historical `crate::dllm::forward_set_causal_positions`
// import path (notably `src/speculative/set_diffusion.rs` production + tests).

/// Re-export of the set-causal forward pass (moved to katgpt-forward).
/// See `katgpt_forward::forward_set_causal::forward_set_causal_positions` for docs.
#[cfg(feature = "set_diffusion")]
pub use katgpt_forward::forward_set_causal_positions;

// ═══════════════════════════════════════════════════════════════
// Zero-Alloc D2F Context + Forward — re-export from katgpt-forward
// ═══════════════════════════════════════════════════════════════
//
// Plan 398 (2026-07-05): `D2fContext`, `forward_block_causal_with`, and
// `denoising_accuracy` moved to `katgpt_forward::d2f_context`. This module
// re-exports them so every historical `katgpt_rs::dllm::D2fContext` /
// `katgpt_rs::dllm::forward_block_causal_with` /
// `katgpt_rs::dllm::denoising_accuracy` import path continues to resolve.
//
// The substrate is gated `dllm` in katgpt-forward (mirrors root's gate on
// the same name); we gate the re-export here with the same feature so the
// items disappear together when the feature is off.
//
// `attention_forward_safe_into` is re-exported separately near its original
// location (search above) because 4 stay-in-root training callers consume it.

#[cfg(feature = "dllm")]
pub use katgpt_forward::d2f_context::{D2fContext, forward_block_causal_with};

// ═══════════════════════════════════════════════════════════════
// Task 0.5: Denoising Loop with Constraint
// ═══════════════════════════════════════════════════════════════
//
// Plan 403 (2026-07-06): The `DenoiseConstraint` trait, `NoConstraint` /
// `NoRepeatConstraint` impls, and the four `denoise_loop*` variants moved to
// `katgpt-forward/src/denoise_loops.rs`. Root re-exports via the shims below
// so every historical `crate::dllm::{denoise_loop, denoise_loop_rcd,
// DenoiseConstraint, NoConstraint, NoRepeatConstraint, ...}` import path
// continues to resolve. The 9 denoise tests in `mod tests` exercise the
// public API via these re-exports (they depend on root-only training helpers
// `train_mini_dllm` / `generate_pattern_dataset`, so they stay in root).

#[cfg(all(feature = "dllm", feature = "rcd_residual"))]
pub use katgpt_forward::denoise_loops::denoise_loop_rcd;
#[cfg(all(feature = "dllm", feature = "d2f_3sr_warm_start"))]
pub use katgpt_forward::denoise_loops::denoise_loop_rcd_3sr;
#[cfg(feature = "dllm")]
pub use katgpt_forward::denoise_loops::{
    DenoiseConstraint, NoConstraint, NoRepeatConstraint, denoise_loop, denoise_loop_scheduled,
};

// ═══════════════════════════════════════════════════════════════
// Position-Offset Reveal-Time Schedule (Research 376, arXiv:2607.01775)
// ═══════════════════════════════════════════════════════════════
//
// DRY consolidation (2026-07-04): the canonical `PositionOffsetSchedule` now
// lives in `katgpt-core::set_diffusion_schedule`. Re-exported here so existing
// `katgpt_rs::dllm::PositionOffsetSchedule` paths continue to resolve.
//
// The katgpt-core version is RNG-agnostic via `sample_order_with(l, || ...)`.
// Call sites in this file that use `katgpt_types::Rng` pass `|| rng.uniform()`;
// consumers using `fastrand::Rng` (e.g. riir-train) use `|| rng.f32()` or the
// `sample_order(l, &mut fastrand::Rng)` convenience wrapper.
pub use katgpt_core::PositionOffsetSchedule;

/// Measure denoising accuracy: fraction of correctly recovered tokens.
///
/// Plan 398 (2026-07-05): Re-exported from `katgpt_forward::d2f_context`.
#[cfg(feature = "dllm")]
pub use katgpt_forward::d2f_context::denoising_accuracy;

// ═══════════════════════════════════════════════════════════════
// Tests — split into dllm/tests.rs (Issue 166, 2026-07-17)
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests;
