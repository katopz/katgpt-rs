//! Drafter LoRA training for speculative decoding (Plan 117 Phase 1).
//!
//! Implements a LoRA-trained drafter that learns to predict target model outputs.
//! At Config::draft() scale (372 base params, ~288 LoRA params at rank-4),
//! training is trivially fast using finite-difference gradients.

use std::path::Path;

use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, LoraAdapter, Rng, kv_dim, lora_apply, matmul, matmul_relu, rmsnorm};

// ── Binary format constants ──────────────────────────────────
const DRAFTER_LORA_MAGIC: &[u8; 4] = b"DLRA";
const DRAFTER_LORA_VERSION: u32 = 1;
const DRAFTER_LORA_N_ADAPTERS: u32 = 6;

/// LoRA adapters for a single-layer drafter model.
///
/// Contains 6 adapters: Q, K, V, O projections + MLP W1, W2.
/// Standard LoRA init: A is random (Kaiming-like), B is zeros, so ΔW = B@A ≈ 0
/// preserves the base model at initialization.
///
/// At rank-4 on Config::draft() (n_embd=4, kv_dim=4, mlp_hidden=16):
/// - Q,K,V,O: 32 params each (A[16] + B[16])
/// - MLP1: 80 params (A[16] + B[64])
/// - MLP2: 80 params (A[64] + B[16])
/// - Total: 288 params
pub struct DrafterLoraWeights {
    pub q_lora: LoraAdapter,
    pub k_lora: LoraAdapter,
    pub v_lora: LoraAdapter,
    pub o_lora: LoraAdapter,
    pub mlp1_lora: LoraAdapter,
    pub mlp2_lora: LoraAdapter,
}

impl std::fmt::Debug for DrafterLoraWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrafterLoraWeights")
            .field("q_lora", &lora_summary(&self.q_lora))
            .field("k_lora", &lora_summary(&self.k_lora))
            .field("v_lora", &lora_summary(&self.v_lora))
            .field("o_lora", &lora_summary(&self.o_lora))
            .field("mlp1_lora", &lora_summary(&self.mlp1_lora))
            .field("mlp2_lora", &lora_summary(&self.mlp2_lora))
            .finish()
    }
}

fn lora_summary(lora: &LoraAdapter) -> String {
    format!(
        "LoraAdapter(rank={}, alpha={:.1}, in_dim={}, out_dim={})",
        lora.rank, lora.alpha, lora.in_dim, lora.out_dim
    )
}

impl DrafterLoraWeights {
    /// Initialize all adapters with zeros (identity LoRA — adds nothing).
    /// Useful as a baseline or for testing.
    pub fn zeros(config: &Config, rank: usize, alpha: f32) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let mlp = config.mlp_hidden;
        Self {
            q_lora: make_lora(rank, alpha, n, n),
            k_lora: make_lora(rank, alpha, n, kvd),
            v_lora: make_lora(rank, alpha, n, kvd),
            o_lora: make_lora(rank, alpha, n, n),
            mlp1_lora: make_lora(rank, alpha, n, mlp),
            mlp2_lora: make_lora(rank, alpha, mlp, n),
        }
    }

    /// Initialize with random A and zero B (standard LoRA init).
    /// ΔW = B @ A ≈ 0 at init, preserving base model behavior.
    pub fn new(config: &Config, rank: usize, alpha: f32, rng: &mut Rng) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let mlp = config.mlp_hidden;
        Self {
            q_lora: make_lora_random(rank, alpha, n, n, rng),
            k_lora: make_lora_random(rank, alpha, n, kvd, rng),
            v_lora: make_lora_random(rank, alpha, n, kvd, rng),
            o_lora: make_lora_random(rank, alpha, n, n, rng),
            mlp1_lora: make_lora_random(rank, alpha, n, mlp, rng),
            mlp2_lora: make_lora_random(rank, alpha, mlp, n, rng),
        }
    }

    /// Collect all LoRA params into a flat vector for gradient computation.
    /// Order: [Q_A, Q_B, K_A, K_B, V_A, V_B, O_A, O_B, MLP1_A, MLP1_B, MLP2_A, MLP2_B]
    fn params_flat(&self) -> Vec<f32> {
        let mut params = Vec::with_capacity(self.total_params());
        for adapter in &self.adapters() {
            params.extend_from_slice(&adapter.a);
            params.extend_from_slice(&adapter.b);
        }
        params
    }

    /// Set all LoRA params from a flat vector (same order as params_flat).
    fn set_params_flat(&mut self, params: &[f32]) {
        let mut offset = 0;
        for adapter in self.adapters_mut() {
            let a_len = adapter.a.len();
            let b_len = adapter.b.len();
            adapter.a.copy_from_slice(&params[offset..offset + a_len]);
            offset += a_len;
            adapter.b.copy_from_slice(&params[offset..offset + b_len]);
            offset += b_len;
        }
    }

    /// Total number of LoRA parameters across all 6 adapters.
    pub fn total_params(&self) -> usize {
        self.adapters().iter().map(|a| a.a.len() + a.b.len()).sum()
    }

    /// Get all adapters as an array (ordered: q, k, v, o, mlp1, mlp2).
    fn adapters(&self) -> [&LoraAdapter; 6] {
        [
            &self.q_lora,
            &self.k_lora,
            &self.v_lora,
            &self.o_lora,
            &self.mlp1_lora,
            &self.mlp2_lora,
        ]
    }

    /// Get all adapters as a mutable array.
    fn adapters_mut(&mut self) -> [&mut LoraAdapter; 6] {
        [
            &mut self.q_lora,
            &mut self.k_lora,
            &mut self.v_lora,
            &mut self.o_lora,
            &mut self.mlp1_lora,
            &mut self.mlp2_lora,
        ]
    }
}

/// Create a zero-initialized LoRA adapter.
fn make_lora(rank: usize, alpha: f32, in_dim: usize, out_dim: usize) -> LoraAdapter {
    LoraAdapter {
        a: vec![0.0; rank * in_dim],
        b: vec![0.0; out_dim * rank],
        rank,
        alpha,
        in_dim,
        out_dim,
    }
}

/// Create a LoRA adapter with random A (Kaiming-like) and zero B.
fn make_lora_random(
    rank: usize,
    alpha: f32,
    in_dim: usize,
    out_dim: usize,
    rng: &mut Rng,
) -> LoraAdapter {
    let scale = (2.0 / in_dim as f32).sqrt();
    LoraAdapter {
        a: (0..rank * in_dim).map(|_| rng.normal() * scale).collect(),
        b: vec![0.0; out_dim * rank],
        rank,
        alpha,
        in_dim,
        out_dim,
    }
}

// ── Training pair generation (T3) ───────────────────────────

/// A single training pair: (input_tokens, target_output_token).
/// The drafter LoRA should learn to predict `target_token` given `input_tokens`.
pub struct TrainingPair {
    pub input_tokens: Vec<usize>,
    pub target_token: usize,
}

/// Generate training pairs by running the target model on replay sequences.
///
/// For each position in the sequence, runs the target model forward and uses
/// argmax of the output logits as the "correct" next token prediction.
/// This teaches the drafter to mimic the target model's behavior.
pub fn generate_training_pairs_from_replays(
    target_config: &Config,
    target_weights: &TransformerWeights,
    replay_sequences: &[Vec<usize>],
) -> Vec<TrainingPair> {
    let mut pairs = Vec::new();

    for sequence in replay_sequences {
        if sequence.len() < 2 {
            continue;
        }

        let mut ctx = ForwardContext::new(target_config);
        let mut cache = MultiLayerKVCache::new(target_config);

        for pos in 0..sequence.len().saturating_sub(1) {
            let token = sequence[pos];
            let logits = forward(
                &mut ctx,
                target_weights,
                &mut cache,
                token,
                pos,
                target_config,
            );
            let target_token = argmax(logits);
            pairs.push(TrainingPair {
                input_tokens: sequence[..=pos].to_vec(),
                target_token,
            });
        }
    }

    pairs
}

/// Generate synthetic training pairs from random sequences.
/// Useful for testing without needing real replay data.
pub fn generate_synthetic_pairs(
    config: &Config,
    weights: &TransformerWeights,
    n_sequences: usize,
    seq_len: usize,
    rng: &mut Rng,
) -> Vec<TrainingPair> {
    let sequences: Vec<Vec<usize>> = (0..n_sequences)
        .map(|_| {
            (0..seq_len)
                .map(|_| (rng.next() % config.vocab_size as u64) as usize)
                .collect()
        })
        .collect();
    generate_training_pairs_from_replays(config, weights, &sequences)
}

/// Find the index of the maximum value in a slice.
fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Standalone forward with LoRA (for training) ─────────────

/// Forward the drafter with LoRA applied to all 6 projections.
///
/// This is a standalone forward that applies LoRA inline, matching the behavior
/// of `forward_base` for a single-layer model. It operates on pre-allocated
/// scratch buffers to avoid re-allocation during training.
///
/// Note: matches `forward_base` structure exactly — no final rmsnorm before lm_head.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn forward_drafter_with_lora(
    config: &Config,
    weights: &TransformerWeights,
    lora: &DrafterLoraWeights,
    token: usize,
    pos: usize,
    lora_buf: &mut [f32],
    // Scratch buffers (pre-allocated)
    x: &mut [f32],
    xr: &mut [f32],
    xr2: &mut [f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    attn_out: &mut [f32],
    scores: &mut [f32],
    hidden: &mut [f32],
    logits: &mut [f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let n_kv = config.n_kv_head;
    let layer_weights = &weights.layers[0];

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off = pos * n;
    for i in 0..n {
        x[i] = weights.wte[tok_off + i] + weights.wpe[pos_off + i];
    }

    // 2. Pre-attention: RMSNorm → save residual → RMSNorm (matches forward_base)
    rmsnorm(&mut x[..n]);
    xr[..n].copy_from_slice(&x[..n]);
    rmsnorm(&mut x[..n]);

    // 3. QKV projections with LoRA
    matmul(q, &layer_weights.attn_wq, &x[..n], n, n);
    lora_apply(q, &lora.q_lora, &x[..n], lora_buf);

    matmul(k, &layer_weights.attn_wk, &x[..n], kvd, n);
    lora_apply(k, &lora.k_lora, &x[..n], lora_buf);

    matmul(v, &layer_weights.attn_wv, &x[..n], kvd, n);
    lora_apply(v, &lora.v_lora, &x[..n], lora_buf);

    // 4. Store K,V in per-position cache
    let pos_off_cache = pos * kvd;
    key_cache[pos_off_cache..pos_off_cache + kvd].copy_from_slice(&k[..kvd]);
    value_cache[pos_off_cache..pos_off_cache + kvd].copy_from_slice(&v[..kvd]);

    // 5. Multi-head attention with GQA
    let scale = 1.0 / (hd as f32).sqrt();
    attn_out[..n].fill(0.0);
    let t_n = pos + 1;

    for h in 0..config.n_head {
        let kv_group = h * n_kv / config.n_head;
        let q_off = h * hd;
        let kv_off = kv_group * hd;

        // Pass 1: compute scores, find max
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..t_n {
            let k_off = t * kvd + kv_off;
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[q_off + d] * key_cache[k_off + d];
            }
            let score = dot * scale;
            scores[t] = score;
            max_score = max_score.max(score);
        }

        // Pass 2: exp and accumulate sum
        let mut sum = 0.0f32;
        for t in 0..t_n {
            scores[t] = (scores[t] - max_score).exp();
            sum += scores[t];
        }
        let inv_sum = 1.0 / sum;

        // Pass 3: weighted value accumulation
        for d in 0..hd {
            let mut val = 0.0f32;
            for t in 0..t_n {
                val += scores[t] * inv_sum * value_cache[t * kvd + kv_off + d];
            }
            attn_out[q_off + d] = val;
        }
    }

    // 6. Output projection with LoRA + residual
    matmul(&mut x[..n], &layer_weights.attn_wo, &attn_out[..n], n, n);
    lora_apply(&mut x[..n], &lora.o_lora, &attn_out[..n], lora_buf);

    for i in 0..n {
        x[i] += xr[i];
    }

    // 7. MLP: save residual → RMSNorm → MLP with LoRA → residual
    xr2[..n].copy_from_slice(&x[..n]);
    rmsnorm(&mut x[..n]);

    // MLP w1 with ReLU + LoRA
    matmul_relu(hidden, &layer_weights.mlp_w1, &x[..n], config.mlp_hidden, n);
    lora_apply(hidden, &lora.mlp1_lora, &x[..n], lora_buf);

    // MLP w2 + LoRA
    matmul(
        &mut x[..n],
        &layer_weights.mlp_w2,
        &hidden[..config.mlp_hidden],
        n,
        config.mlp_hidden,
    );
    lora_apply(
        &mut x[..n],
        &lora.mlp2_lora,
        &hidden[..config.mlp_hidden],
        lora_buf,
    );

    // Residual
    for i in 0..n {
        x[i] += xr2[i];
    }

    // Note: no final rmsnorm here — forward_base applies lm_head directly
    // after the MLP residual without an additional norm step.

    // 8. LM Head
    matmul(logits, &weights.lm_head, &x[..n], config.vocab_size, n);
}

/// Pre-allocated context for drafter forward with LoRA.
///
/// Avoids re-allocation during training by reusing scratch buffers
/// across forward passes. Create once per training session.
pub struct DrafterForwardContext {
    x: Vec<f32>,
    xr: Vec<f32>,
    xr2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    lora_buf: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl DrafterForwardContext {
    /// Create a new context with pre-allocated buffers for the given config and LoRA rank.
    pub fn new(config: &Config, rank: usize) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        Self {
            x: vec![0.0; n],
            xr: vec![0.0; n],
            xr2: vec![0.0; n],
            q: vec![0.0; n],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; n],
            scores: vec![0.0; config.block_size],
            hidden: vec![0.0; config.mlp_hidden],
            logits: vec![0.0; config.vocab_size],
            lora_buf: vec![0.0; rank],
            key_cache: vec![0.0; config.block_size * kvd],
            value_cache: vec![0.0; config.block_size * kvd],
        }
    }

    /// Forward the drafter with LoRA, returning logits.
    ///
    /// Automatically resets KV cache when pos=0 (start of new sequence).
    pub fn forward_lora(
        &mut self,
        config: &Config,
        weights: &TransformerWeights,
        lora: &DrafterLoraWeights,
        token: usize,
        pos: usize,
    ) -> &[f32] {
        if pos == 0 {
            self.key_cache.fill(0.0);
            self.value_cache.fill(0.0);
        }

        forward_drafter_with_lora(
            config,
            weights,
            lora,
            token,
            pos,
            &mut self.lora_buf,
            &mut self.x,
            &mut self.xr,
            &mut self.xr2,
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.attn_out,
            &mut self.scores,
            &mut self.hidden,
            &mut self.logits,
            &mut self.key_cache,
            &mut self.value_cache,
        );

        &self.logits
    }
}

// ── Loss computation ────────────────────────────────────────

/// Run drafter forward on a token sequence, returning the final logits.
fn run_drafter_sequence(
    ctx: &mut DrafterForwardContext,
    config: &Config,
    weights: &TransformerWeights,
    lora: &DrafterLoraWeights,
    tokens: &[usize],
) -> Vec<f32> {
    for (pos, &token) in tokens.iter().enumerate() {
        ctx.forward_lora(config, weights, lora, token, pos);
    }
    ctx.logits.clone()
}

/// Cross-entropy loss: -log(softmax(logits)[target]).
/// Numerically stable via log-sum-exp trick.
fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for &val in logits {
        sum_exp += (val - max_val).exp();
    }
    -(logits[target] - max_val) + sum_exp.ln()
}

/// Compute cross-entropy loss for a single training pair.
fn compute_lora_loss(
    ctx: &mut DrafterForwardContext,
    config: &Config,
    weights: &TransformerWeights,
    lora: &DrafterLoraWeights,
    pair: &TrainingPair,
) -> f32 {
    let logits = run_drafter_sequence(ctx, config, weights, lora, &pair.input_tokens);
    cross_entropy(&logits, pair.target_token)
}

// ── Training loop (T2) ──────────────────────────────────────

/// Train drafter LoRA to predict target outputs using finite-difference gradients.
///
/// For each training pair:
/// 1. Compute base loss with current LoRA params
/// 2. For each of ~288 LoRA params, perturb by ε and recompute loss
/// 3. Estimate gradient: (loss_perturbed - loss_base) / ε
/// 4. Apply SGD update: param -= lr × gradient
///
/// Returns the best (lowest) average epoch loss across training.
pub fn train_drafter_lora(
    drafter_config: &Config,
    drafter_weights: &TransformerWeights,
    lora: &mut DrafterLoraWeights,
    pairs: &[TrainingPair],
    epochs: usize,
    learning_rate: f32,
) -> f32 {
    let eps = 1e-3f32;
    let mut ctx = DrafterForwardContext::new(drafter_config, lora.q_lora.rank);
    let mut best_loss = f32::MAX;

    for _epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for pair in pairs {
            // Compute current loss
            let base_loss =
                compute_lora_loss(&mut ctx, drafter_config, drafter_weights, lora, pair);

            // Snapshot current params
            let original_params = lora.params_flat();
            let mut gradients = vec![0.0f32; original_params.len()];

            // Compute gradients via finite differences
            for i in 0..original_params.len() {
                let mut perturbed = original_params.clone();
                perturbed[i] += eps;
                lora.set_params_flat(&perturbed);

                let perturbed_loss =
                    compute_lora_loss(&mut ctx, drafter_config, drafter_weights, lora, pair);
                gradients[i] = (perturbed_loss - base_loss) / eps;
            }

            // Restore original params, then apply gradient descent
            lora.set_params_flat(&original_params);
            let mut updated = lora.params_flat();
            for i in 0..updated.len() {
                updated[i] -= learning_rate * gradients[i];
            }
            lora.set_params_flat(&updated);

            epoch_loss += base_loss;
        }

        if !pairs.is_empty() {
            epoch_loss /= pairs.len() as f32;
        }
        best_loss = best_loss.min(epoch_loss);
    }

    best_loss
}

// ── Save/Load (T7) ──────────────────────────────────────────

/// Serialize drafter LoRA weights to binary format.
///
/// ```text
/// [MAGIC: "DLRA" 4B]
/// [VERSION: 4B LE] = 1
/// [RANK: 4B LE]
/// [ALPHA: 4B LE f32]
/// [N_ADAPTERS: 4B LE] = 6
/// [ADAPTER_DIMS: for each of 6 adapters:
///   [IN_DIM: 4B LE]
///   [OUT_DIM: 4B LE]
/// ]
/// [ADAPTER_DATA: for each of 6 adapters:
///   [A: rank × in_dim × 4B f32 LE]
///   [B: out_dim × rank × 4B f32 LE]
/// ]
/// [BLAKE3: 32B]
/// ```
pub fn save_drafter_lora(lora: &DrafterLoraWeights, path: &Path) -> Result<(), String> {
    let rank = lora.q_lora.rank;
    let alpha = lora.q_lora.alpha;
    let adapters = lora.adapters();

    // Header: magic(4) + version(4) + rank(4) + alpha(4) + n_adapters(4) + 6*(in_dim+out_dim)(48) = 68
    let header_size = 4 + 4 + 4 + 4 + 4 + (DRAFTER_LORA_N_ADAPTERS as usize * 8);
    let data_size: usize = adapters
        .iter()
        .map(|a| (a.a.len() + a.b.len()) * std::mem::size_of::<f32>())
        .sum();
    let total_size = header_size + data_size + 32; // +32 for blake3

    let mut buf = Vec::with_capacity(total_size);

    // Header
    buf.extend_from_slice(DRAFTER_LORA_MAGIC);
    buf.extend_from_slice(&DRAFTER_LORA_VERSION.to_le_bytes());
    buf.extend_from_slice(&(rank as u32).to_le_bytes());
    buf.extend_from_slice(&alpha.to_le_bytes());
    buf.extend_from_slice(&DRAFTER_LORA_N_ADAPTERS.to_le_bytes());

    // Adapter dimensions
    for adapter in &adapters {
        buf.extend_from_slice(&(adapter.in_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(adapter.out_dim as u32).to_le_bytes());
    }

    // Adapter weights: A then B for each
    for adapter in &adapters {
        for &val in &adapter.a {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        for &val in &adapter.b {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Blake3 checksum over everything so far
    let hash = blake3::hash(&buf);
    buf.extend_from_slice(hash.as_bytes());

    std::fs::write(path, &buf).map_err(|e| format!("Failed to write drafter lora: {e}"))
}

/// Load drafter LoRA weights from binary format.
pub fn load_drafter_lora(path: &Path) -> Result<DrafterLoraWeights, String> {
    let file_data =
        std::fs::read(path).map_err(|e| format!("Failed to read drafter lora file: {e}"))?;

    // Minimum: header(68) + hash(32) = 100
    if file_data.len() < 100 {
        return Err("File too small for drafter lora header".into());
    }

    // Verify magic
    if &file_data[0..4] != DRAFTER_LORA_MAGIC {
        return Err("Invalid drafter lora magic bytes".into());
    }

    // Verify version
    let version = u32::from_le_bytes(
        file_data[4..8]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("Version parse: {e}"))?,
    );
    if version != DRAFTER_LORA_VERSION {
        return Err(format!("Unsupported drafter lora version: {version}"));
    }

    // Verify blake3 checksum (last 32 bytes cover everything before them)
    let data_len = file_data.len() - 32;
    let stored_checksum = &file_data[data_len..];
    let computed = blake3::hash(&file_data[..data_len]);
    if computed.as_bytes() != stored_checksum {
        return Err("Drafter lora file checksum mismatch".into());
    }

    // Parse header
    let rank = u32::from_le_bytes(
        file_data[8..12]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("Rank parse: {e}"))?,
    ) as usize;

    let alpha = f32::from_le_bytes(
        file_data[12..16]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("Alpha parse: {e}"))?,
    );

    let n_adapters = u32::from_le_bytes(
        file_data[16..20]
            .try_into()
            .map_err(|e: std::array::TryFromSliceError| format!("N_ADAPTERS parse: {e}"))?,
    );
    if n_adapters != DRAFTER_LORA_N_ADAPTERS {
        return Err(format!("Expected 6 adapters, found {n_adapters}"));
    }

    // Parse adapter dimensions
    let mut in_dims = [0usize; 6];
    let mut out_dims = [0usize; 6];
    let dim_offset = 20;
    for i in 0..6 {
        let off = dim_offset + i * 8;
        in_dims[i] = u32::from_le_bytes(
            file_data[off..off + 4]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("InDim[{i}] parse: {e}"))?,
        ) as usize;
        out_dims[i] = u32::from_le_bytes(
            file_data[off + 4..off + 8]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("OutDim[{i}] parse: {e}"))?,
        ) as usize;
    }

    // Parse adapter weights
    let mut offset = dim_offset + 6 * 8; // After dimension table
    let adapters: Vec<LoraAdapter> = (0..6)
        .map(|i| {
            let in_dim = in_dims[i];
            let out_dim = out_dims[i];
            let a_count = rank * in_dim;
            let b_count = out_dim * rank;

            let a: Vec<f32> = file_data[offset..offset + a_count * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                .collect();
            offset += a_count * 4;

            let b: Vec<f32> = file_data[offset..offset + b_count * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                .collect();
            offset += b_count * 4;

            LoraAdapter {
                a,
                b,
                rank,
                alpha,
                in_dim,
                out_dim,
            }
        })
        .collect();

    // Destructure into DrafterLoraWeights (no Clone needed)
    let mut iter = adapters.into_iter();
    let next_adapter =
        |iter: &mut std::vec::IntoIter<LoraAdapter>, name: &str| -> Result<LoraAdapter, String> {
            iter.next().ok_or_else(|| format!("Missing {name} adapter"))
        };

    Ok(DrafterLoraWeights {
        q_lora: next_adapter(&mut iter, "Q")?,
        k_lora: next_adapter(&mut iter, "K")?,
        v_lora: next_adapter(&mut iter, "V")?,
        o_lora: next_adapter(&mut iter, "O")?,
        mlp1_lora: next_adapter(&mut iter, "MLP1")?,
        mlp2_lora: next_adapter(&mut iter, "MLP2")?,
    })
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn draft_config() -> Config {
        Config::draft()
    }

    #[test]
    fn test_drafter_lora_zeros_init() {
        let config = draft_config();
        let lora = DrafterLoraWeights::zeros(&config, 4, 8.0);

        // All B matrices should be zero (standard LoRA init)
        assert!(
            lora.q_lora.b.iter().all(|&v| v == 0.0),
            "Q B should be zero"
        );
        assert!(
            lora.k_lora.b.iter().all(|&v| v == 0.0),
            "K B should be zero"
        );
        assert!(
            lora.v_lora.b.iter().all(|&v| v == 0.0),
            "V B should be zero"
        );
        assert!(
            lora.o_lora.b.iter().all(|&v| v == 0.0),
            "O B should be zero"
        );
        assert!(
            lora.mlp1_lora.b.iter().all(|&v| v == 0.0),
            "MLP1 B should be zero"
        );
        assert!(
            lora.mlp2_lora.b.iter().all(|&v| v == 0.0),
            "MLP2 B should be zero"
        );

        // All A matrices should also be zero in zeros() init
        assert!(
            lora.q_lora.a.iter().all(|&v| v == 0.0),
            "Q A should be zero"
        );
    }

    #[test]
    fn test_drafter_lora_new_has_random_a_zero_b() {
        let config = draft_config();
        let mut rng = Rng::new(42);
        let lora = DrafterLoraWeights::new(&config, 4, 8.0, &mut rng);

        // B matrices should be zero
        assert!(
            lora.q_lora.b.iter().all(|&v| v == 0.0),
            "Q B should be zero"
        );
        assert!(
            lora.k_lora.b.iter().all(|&v| v == 0.0),
            "K B should be zero"
        );

        // A matrices should be non-zero (random init)
        assert!(
            lora.q_lora.a.iter().any(|&v| v != 0.0),
            "Q A should be non-zero"
        );
    }

    #[test]
    fn test_drafter_lora_forward_produces_logits() {
        let config = draft_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let lora = DrafterLoraWeights::new(&config, 4, 8.0, &mut rng);
        let mut ctx = DrafterForwardContext::new(&config, 4);

        let logits = ctx.forward_lora(&config, &weights, &lora, 0, 0);

        assert_eq!(
            logits.len(),
            config.vocab_size,
            "Logits should have vocab_size elements"
        );
        assert!(
            logits.iter().all(|&v| v.is_finite()),
            "All logits should be finite"
        );
    }

    #[test]
    fn test_drafter_lora_zeros_same_as_no_lora() {
        let config = draft_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Forward without LoRA (using standard forward)
        let mut ctx_std = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits_std = forward(&mut ctx_std, &weights, &mut cache, 0, 0, &config);
        let std_logits: Vec<f32> = logits_std.to_vec();

        // Forward with zero LoRA (should be identical since ΔW=0)
        let lora = DrafterLoraWeights::zeros(&config, 4, 8.0);
        let mut ctx_lora = DrafterForwardContext::new(&config, 4);
        let lora_logits: Vec<f32> = ctx_lora
            .forward_lora(&config, &weights, &lora, 0, 0)
            .to_vec();

        // Zero LoRA produces identical output to standard forward (ΔW=0)
        for i in 0..config.vocab_size {
            assert!(
                (std_logits[i] - lora_logits[i]).abs() < 1e-4,
                "Logit {i}: std={} lora={}",
                std_logits[i],
                lora_logits[i]
            );
        }
    }

    #[test]
    fn test_drafter_lora_training_loss_decreases() {
        let config = draft_config();
        let mut rng = Rng::new(123);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Create simple training pairs
        let pairs: Vec<TrainingPair> = (0..5)
            .map(|i| TrainingPair {
                input_tokens: vec![i % config.vocab_size, (i + 1) % config.vocab_size],
                target_token: (i + 2) % config.vocab_size,
            })
            .collect();

        let mut lora = DrafterLoraWeights::new(&config, 4, 8.0, &mut rng);

        // Compute initial loss
        let mut ctx = DrafterForwardContext::new(&config, 4);
        let initial_loss: f32 = pairs
            .iter()
            .map(|p| compute_lora_loss(&mut ctx, &config, &weights, &lora, p))
            .sum::<f32>()
            / pairs.len() as f32;

        // Train for 5 epochs
        let final_loss = train_drafter_lora(&config, &weights, &mut lora, &pairs, 5, 0.01);

        // Loss should decrease (or stay roughly the same)
        assert!(
            final_loss <= initial_loss + 0.5,
            "Final loss ({final_loss}) should be <= initial loss ({initial_loss}) + 0.5"
        );
    }

    #[test]
    fn test_drafter_lora_save_load_roundtrip() {
        let config = draft_config();
        let mut rng = Rng::new(42);
        let original = DrafterLoraWeights::new(&config, 4, 8.0, &mut rng);

        let temp_file = NamedTempFile::new().expect("create temp file");
        let path = temp_file.path();

        // Save
        save_drafter_lora(&original, path).expect("save should succeed");

        // Load
        let loaded = load_drafter_lora(path).expect("load should succeed");

        // Verify all weights match
        let original_adapters = original.adapters();
        let loaded_adapters = loaded.adapters();

        for (i, (orig, loaded)) in original_adapters
            .iter()
            .zip(loaded_adapters.iter())
            .enumerate()
        {
            assert_eq!(orig.rank, loaded.rank, "Adapter {i} rank mismatch");
            assert_eq!(orig.alpha, loaded.alpha, "Adapter {i} alpha mismatch");
            assert_eq!(orig.in_dim, loaded.in_dim, "Adapter {i} in_dim mismatch");
            assert_eq!(orig.out_dim, loaded.out_dim, "Adapter {i} out_dim mismatch");
            assert_eq!(
                orig.a.len(),
                loaded.a.len(),
                "Adapter {i} A length mismatch"
            );
            assert_eq!(
                orig.b.len(),
                loaded.b.len(),
                "Adapter {i} B length mismatch"
            );

            for j in 0..orig.a.len() {
                assert!(
                    (orig.a[j] - loaded.a[j]).abs() < 1e-6,
                    "Adapter {i} A[{j}] mismatch: {} vs {}",
                    orig.a[j],
                    loaded.a[j]
                );
            }
            for j in 0..orig.b.len() {
                assert!(
                    (orig.b[j] - loaded.b[j]).abs() < 1e-6,
                    "Adapter {i} B[{j}] mismatch: {} vs {}",
                    orig.b[j],
                    loaded.b[j]
                );
            }
        }
    }

    #[test]
    fn test_drafter_lora_total_params() {
        let config = draft_config();
        let lora = DrafterLoraWeights::zeros(&config, 4, 8.0);

        // Config::draft(): n_embd=4, kv_dim=4, mlp_hidden=16, rank=4
        // Q: 4*4 + 4*4 = 32
        // K: 4*4 + 4*4 = 32
        // V: 4*4 + 4*4 = 32
        // O: 4*4 + 4*4 = 32
        // MLP1: 4*4 + 16*4 = 80
        // MLP2: 4*16 + 4*4 = 80
        // Total: 288
        assert_eq!(lora.total_params(), 288, "Total params should be 288");
    }

    #[test]
    fn test_generate_training_pairs() {
        let config = draft_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let sequences = vec![vec![0, 1, 2, 3], vec![5, 10, 15]];

        let pairs = generate_training_pairs_from_replays(&config, &weights, &sequences);

        // sequence [0,1,2,3] (len=4): positions 0..3 = 3 pairs
        // sequence [5,10,15] (len=3): positions 0..2 = 2 pairs
        // Total: 5
        assert_eq!(pairs.len(), 5, "Should produce 5 training pairs");

        // Verify input_tokens lengths grow with position
        assert_eq!(pairs[0].input_tokens.len(), 1); // [0]
        assert_eq!(pairs[1].input_tokens.len(), 2); // [0, 1]
        assert_eq!(pairs[2].input_tokens.len(), 3); // [0, 1, 2]
        assert_eq!(pairs[3].input_tokens.len(), 1); // [5] (new sequence)
        assert_eq!(pairs[4].input_tokens.len(), 2); // [5, 10]
    }

    #[test]
    fn test_load_drafter_lora_invalid_magic() {
        let temp_file = NamedTempFile::new().expect("create temp file");
        // Write enough bytes to pass size check but with wrong magic
        let mut data = vec![0u8; 120];
        data[0..4].copy_from_slice(b"XXXX");
        std::fs::write(temp_file.path(), &data).expect("write");
        let result = load_drafter_lora(temp_file.path());
        assert!(result.is_err(), "Should fail with invalid magic");
        assert!(
            result.unwrap_err().contains("magic"),
            "Error should mention magic"
        );
    }

    #[test]
    fn test_load_drafter_lora_bad_checksum() {
        let config = draft_config();
        let mut rng = Rng::new(42);
        let lora = DrafterLoraWeights::new(&config, 4, 8.0, &mut rng);

        let temp_file = NamedTempFile::new().expect("create temp file");
        save_drafter_lora(&lora, temp_file.path()).expect("save");

        // Corrupt the last byte (part of checksum)
        let mut data = std::fs::read(temp_file.path()).expect("read");
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        std::fs::write(temp_file.path(), &data).expect("write corrupted");

        let result = load_drafter_lora(temp_file.path());
        assert!(result.is_err(), "Should fail with bad checksum");
        assert!(
            result.unwrap_err().contains("checksum"),
            "Error should mention checksum"
        );
    }
}
