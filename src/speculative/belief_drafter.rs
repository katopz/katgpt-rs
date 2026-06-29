//! NextLat Belief-State Speculative Drafter — lightweight MLP recursive hidden state prediction.
//!
//! Implements Plan 217: `LatentDynamicsMLP` (Phase 0) + `BeliefDrafter` with `draft()` (Plan 1).
//! The MLP predicts `h_{t+1}` from `(h_t, emb(x_{t+1}))` using a 3-layer residual architecture
//! inspired by arXiv:2511.05963 (NextLat). `BeliefDrafter` wraps the MLP with an output head
//! for recursive variable-length speculative drafting with entropy-gated stopping.
//!
//! Architecture: `h_{t+1} = h_t + FC3(GELU(FC2(GELU(FC1(LN(concat(h_t, next_emb)))))))`
//!
//! Self-conditioning (Plan 222 T11): additive SC injection into the embedding channel.
//! `forward_with_sc(h_t, next_emb, sc, scale)` blends SC into next_emb before the MLP,
//! keeping weight dimensions unchanged. Feature-gated behind `self_cond_draft`.
//!
//! Feature-gated behind `belief_drafter` — off by default until GOAT proof.
//!
//! # Attention-drift subject (Plan 306, Research 286, arXiv:2605.09992)
//!
//! This drafter is a **known subject** of the attention-drift failure mode diagnosed
//! by Eldenk et al. — the architecture (input LayerNorm + unnormalized residual) is
//! structurally identical to the pre-norm EAGLE-3 drafter that paper §3 shows classifies
//! as `DepthSpecificRefinement` beyond the TTT horizon. Run [`BeliefDrafter::audit_depth_invariance`]
//! (gated on `depth_invariance`) to classify a chain.
//!
//! **The fix (post-norm residual) is NOT applied here.** It requires MLP retraining —
//! training-side work lives in `riir-train`. Inference-time [`katgpt_core::MagnitudeRegularization`]
//! is **diagnostic-only** for this kernel: paper §4.4 Table 4 reports -56% acceptance
//! when applied to a frozen pre-norm model. For kernels we own (HLA, functor,
//! micro_belief, engram, Raven) `MagnitudeRegularization` is the modelless upstream fix;
//! for this frozen-pretrained MLP it is not.
//!
//! Disambiguation: this is the **drafter-side magnitude-accumulation** mechanism, distinct
//! from the **target-side sink classification** mechanism of Plan 287 / Research 258
//! (arXiv:2606.08105). Different paper, different mechanism.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use katgpt_core::{Config, SpeculativeGenerator};

use katgpt_core::simd::simd_dot_f32;

// ── Magic & Version ────────────────────────────────────────────
const MAGIC: &[u8; 4] = b"NLDM";
const VERSION: u32 = 1;

// ── GELU Approximation ────────────────────────────────────────

/// Standard GELU approximation: `0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ── LayerNorm ──────────────────────────────────────────────────

/// Standard LayerNorm: normalize to zero mean / unit variance, then apply affine transform.
fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(weight.len(), n);
    assert_eq!(bias.len(), n);
    assert_eq!(output.len(), n);

    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();

    for i in 0..n {
        output[i] = weight[i] * (input[i] - mean) * inv_std + bias[i];
    }
}

// ── Linear (Row-Major Matmul + Bias) ──────────────────────────

/// Row-major matmul + bias: `output[i] = dot(weight[i*in_dim..(i+1)*in_dim], input) + bias[i]`
/// Uses `simd_dot_f32` for each row.
fn linear(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize, output: &mut [f32]) {
    let in_dim = input.len();
    assert_eq!(weight.len(), out_dim * in_dim);
    assert_eq!(bias.len(), out_dim);
    assert_eq!(output.len(), out_dim);

    for i in 0..out_dim {
        let row_off = i * in_dim;
        let dot = simd_dot_f32(&weight[row_off..row_off + in_dim], input, in_dim);
        output[i] = dot + bias[i];
    }
}

// ── LatentDynamicsMLP ──────────────────────────────────────────

/// 3-layer residual MLP that predicts next hidden states from `(h_t, emb(x_{t+1}))`.
///
/// - Input: `LayerNorm(concat(h_t, next_emb))` — shape `[2 * n_embd]`
/// - FC1: `[2 * n_embd] → [n_embd]`, GELU
/// - FC2: `[n_embd] → [n_embd]`, GELU
/// - FC3: `[n_embd] → [n_embd]`
/// - Output: `h_{t+1} = h_t + FC3(...)` (residual connection)
///
/// For Config::micro (n_embd=16): ~1.5K params. For Config::bpe (n_embd=32): ~6K params.
#[derive(Debug)]
pub struct LatentDynamicsMLP {
    pub n_embd: usize,
    pub norm_weight: Vec<f32>, // [2 * n_embd]
    pub norm_bias: Vec<f32>,   // [2 * n_embd]
    pub fc1_weight: Vec<f32>,  // [n_embd, 2*n_embd] row-major
    pub fc1_bias: Vec<f32>,    // [n_embd]
    pub fc2_weight: Vec<f32>,  // [n_embd, n_embd] row-major
    pub fc2_bias: Vec<f32>,    // [n_embd]
    pub fc3_weight: Vec<f32>,  // [n_embd, n_embd] row-major
    pub fc3_bias: Vec<f32>,    // [n_embd]
}

/// Reusable scratch buffers for [`LatentDynamicsMLP::forward_into`].
///
/// Sized once at construction and reused across every draft step — eliminates
/// the 5 intermediate `vec!` allocations that the allocating `forward` would
/// otherwise do per token. None of these buffers are read between calls; they
/// are fully overwritten on each `forward_into`.
pub struct MlpForwardScratch {
    /// `[2 * n_embd]` — concat(h_t, next_emb).
    concat: Vec<f32>,
    /// `[2 * n_embd]` — LayerNorm output.
    normed: Vec<f32>,
    /// `[n_embd]` — FC1 output (post-GELU).
    fc1_out: Vec<f32>,
    /// `[n_embd]` — FC2 output (post-GELU).
    fc2_out: Vec<f32>,
    /// `[n_embd]` — FC3 output (pre-residual).
    fc3_out: Vec<f32>,
}

impl MlpForwardScratch {
    /// Allocate scratch sized for a model with embedding dim `n_embd`.
    #[inline]
    pub fn new(n_embd: usize) -> Self {
        let concat_dim = 2 * n_embd;
        Self {
            concat: vec![0.0; concat_dim],
            normed: vec![0.0; concat_dim],
            fc1_out: vec![0.0; n_embd],
            fc2_out: vec![0.0; n_embd],
            fc3_out: vec![0.0; n_embd],
        }
    }
}

impl LatentDynamicsMLP {
    /// Run the MLP forward pass: `h_{t+1} = h_t + FC3(GELU(FC2(GELU(FC1(LN(concat))))))`.
    ///
    /// - `h_t`: current hidden state `[n_embd]`
    /// - `next_emb`: embedding of next token `[n_embd]`
    /// - Returns: predicted next hidden state `[n_embd]`
    ///
    /// Allocating convenience wrapper around [`Self::forward_into`]; hot-path
    /// callers (e.g. [`BeliefDrafter::draft`]) should reuse an
    /// [`MlpForwardScratch`] via `forward_into` to avoid the 6 per-step
    /// allocations.
    pub fn forward(&self, h_t: &[f32], next_emb: &[f32]) -> Vec<f32> {
        let n = self.n_embd;
        let mut scratch = MlpForwardScratch::new(n);
        let mut out = vec![0.0f32; n];
        self.forward_into(h_t, next_emb, &mut scratch, &mut out);
        out
    }

    /// Zero-allocation forward pass writing into caller-provided scratch + output.
    ///
    /// All intermediate buffers (`concat`, `normed`, `fc1_out`, `fc2_out`,
    /// `fc3_out`) live in `scratch` and are reused across calls; the residual
    /// result is written into `out` (`h_{t+1} = h_t + FC3(...)`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into(
        &self,
        h_t: &[f32],
        next_emb: &[f32],
        scratch: &mut MlpForwardScratch,
        out: &mut [f32],
    ) {
        let n = self.n_embd;
        assert_eq!(h_t.len(), n, "h_t must have length n_embd");
        assert_eq!(next_emb.len(), n, "next_emb must have length n_embd");

        let concat_dim = 2 * n;

        // 1. Concatenate h_t and next_emb
        let concat = &mut scratch.concat[..concat_dim];
        concat[..n].copy_from_slice(h_t);
        concat[n..].copy_from_slice(next_emb);

        // 2. LayerNorm
        let normed = &mut scratch.normed[..concat_dim];
        layer_norm(concat, &self.norm_weight, &self.norm_bias, normed);

        // 3. FC1: [2*n_embd] → [n_embd] + GELU
        let fc1_out = &mut scratch.fc1_out[..n];
        linear(normed, &self.fc1_weight, &self.fc1_bias, n, fc1_out);
        for v in fc1_out.iter_mut() {
            *v = gelu(*v);
        }

        // 4. FC2: [n_embd] → [n_embd] + GELU
        let fc2_out = &mut scratch.fc2_out[..n];
        linear(fc1_out, &self.fc2_weight, &self.fc2_bias, n, fc2_out);
        for v in fc2_out.iter_mut() {
            *v = gelu(*v);
        }

        // 5. FC3: [n_embd] → [n_embd] (no activation)
        let fc3_out = &mut scratch.fc3_out[..n];
        linear(fc2_out, &self.fc3_weight, &self.fc3_bias, n, fc3_out);

        // 6. Residual: h_{t+1} = h_t + FC3(...)
        let out = &mut out[..n];
        for i in 0..n {
            out[i] = h_t[i] + fc3_out[i];
        }
    }

    /// Forward pass with self-conditioning: additive SC injection into embedding channel.
    ///
    /// Blends the SC signal (previous prediction) into `next_emb` before the standard
    /// MLP forward pass. This keeps weight dimensions unchanged — same weights work
    /// with or without SC.
    ///
    /// Formula: `next_emb' = next_emb + scale * sc`
    /// Then: `h_{t+1} = forward(h_t, next_emb')`
    ///
    /// - `h_t`: current hidden state `[n_embd]`
    /// - `next_emb`: embedding of next token `[n_embd]`
    /// - `sc`: self-conditioning signal from previous prediction `[n_embd]`
    /// - `sc_scale`: blending factor (typical: 0.1–0.5). 0.0 = no SC effect.
    /// - Returns: predicted next hidden state `[n_embd]`
    #[cfg(feature = "self_cond_draft")]
    pub fn forward_with_sc(
        &self,
        h_t: &[f32],
        next_emb: &[f32],
        sc: &[f32],
        sc_scale: f32,
    ) -> Vec<f32> {
        let n = self.n_embd;
        assert_eq!(h_t.len(), n, "h_t must have length n_embd");
        assert_eq!(next_emb.len(), n, "next_emb must have length n_embd");
        assert_eq!(sc.len(), n, "sc must have length n_embd");

        // Additive blend: next_emb' = next_emb + scale * sc
        let mut blended_emb = vec![0.0f32; n];
        for i in 0..n {
            blended_emb[i] = next_emb[i] + sc_scale * sc[i];
        }

        // Standard forward with blended embedding (reuses forward_into to
        // avoid a second allocation for the residual output buffer).
        let mut scratch = MlpForwardScratch::new(n);
        let mut out = vec![0.0f32; n];
        self.forward_into(h_t, &blended_emb, &mut scratch, &mut out);
        out
    }

    /// Load MLP weights from a binary file.
    ///
    /// Binary format:
    /// - 4 bytes: magic "NLDM"
    /// - u32: version (must be 1)
    /// - u32: n_embd
    /// - Raw f32 arrays in order: norm_weight, norm_bias, fc1_weight, fc1_bias,
    ///   fc2_weight, fc2_bias, fc3_weight, fc3_bias
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open error: {e}"))?;
        let mut rdr = BufReader::new(file);

        // Magic
        let mut magic = [0u8; 4];
        rdr.read_exact(&mut magic)
            .map_err(|e| format!("read magic: {e}"))?;
        if &magic != MAGIC {
            return Err(format!("bad magic: expected {:?}, got {:?}", MAGIC, &magic));
        }

        // Version
        let version = read_u32(&mut rdr)?;
        if version != VERSION {
            return Err(format!(
                "unsupported version: {version} (expected {VERSION})"
            ));
        }

        // n_embd
        let n_embd = read_u32(&mut rdr)? as usize;
        if n_embd == 0 {
            return Err("n_embd must be > 0".into());
        }

        let concat_dim = 2 * n_embd;
        let fc1_rows = n_embd * concat_dim;
        let fc2_rows = n_embd * n_embd;
        let fc3_rows = n_embd * n_embd;

        let norm_weight = read_f32_vec(&mut rdr, concat_dim, "norm_weight")?;
        let norm_bias = read_f32_vec(&mut rdr, concat_dim, "norm_bias")?;
        let fc1_weight = read_f32_vec(&mut rdr, fc1_rows, "fc1_weight")?;
        let fc1_bias = read_f32_vec(&mut rdr, n_embd, "fc1_bias")?;
        let fc2_weight = read_f32_vec(&mut rdr, fc2_rows, "fc2_weight")?;
        let fc2_bias = read_f32_vec(&mut rdr, n_embd, "fc2_bias")?;
        let fc3_weight = read_f32_vec(&mut rdr, fc3_rows, "fc3_weight")?;
        let fc3_bias = read_f32_vec(&mut rdr, n_embd, "fc3_bias")?;

        Ok(Self {
            n_embd,
            norm_weight,
            norm_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        })
    }

    /// Save MLP weights to a binary file (for roundtrip testing).
    pub fn save_to_bin(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("create error: {e}"))?;
        let mut wtr = BufWriter::new(file);

        wtr.write_all(MAGIC)
            .map_err(|e| format!("write magic: {e}"))?;
        write_u32(&mut wtr, VERSION)?;
        write_u32(&mut wtr, self.n_embd as u32)?;

        write_f32_slice(&mut wtr, &self.norm_weight)?;
        write_f32_slice(&mut wtr, &self.norm_bias)?;
        write_f32_slice(&mut wtr, &self.fc1_weight)?;
        write_f32_slice(&mut wtr, &self.fc1_bias)?;
        write_f32_slice(&mut wtr, &self.fc2_weight)?;
        write_f32_slice(&mut wtr, &self.fc2_bias)?;
        write_f32_slice(&mut wtr, &self.fc3_weight)?;
        write_f32_slice(&mut wtr, &self.fc3_bias)?;

        wtr.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }

    /// Initialize MLP with Xavier-like weights, zeros for biases, ones for norm_weight.
    ///
    /// Uses a simple seeded LCG RNG for reproducibility (no external rand dependency).
    pub fn random_init(n_embd: usize) -> Self {
        let concat_dim = 2 * n_embd;

        // Seeded LCG: x_{n+1} = (a * x_n + c) mod 2^32
        // Using Numerical Recipes constants
        let mut state: u32 = 42;

        let mut next_f32 = || -> f32 {
            state = state.wrapping_mul(1_106_351_524).wrapping_add(12_345);
            // Map to (-1, 1) uniformly
            let bits = state >> 1; // clear sign bit
            (bits as f32) / (u32::MAX as f32 * 0.5) - 1.0
        };

        // Xavier init: scale = sqrt(2 / fan_in)
        let xavier_fc1 = (2.0 / concat_dim as f32).sqrt();
        let xavier_fc2 = (2.0 / n_embd as f32).sqrt();
        let xavier_fc3 = (2.0 / n_embd as f32).sqrt();

        // LayerNorm: ones for weight, zeros for bias
        let norm_weight = vec![1.0f32; concat_dim];
        let norm_bias = vec![0.0f32; concat_dim];

        // FC1: [n_embd, 2*n_embd]
        let fc1_weight: Vec<f32> = (0..n_embd * concat_dim)
            .map(|_| next_f32() * xavier_fc1)
            .collect();
        let fc1_bias = vec![0.0f32; n_embd];

        // FC2: [n_embd, n_embd]
        let fc2_weight: Vec<f32> = (0..n_embd * n_embd)
            .map(|_| next_f32() * xavier_fc2)
            .collect();
        let fc2_bias = vec![0.0f32; n_embd];

        // FC3: [n_embd, n_embd]
        let fc3_weight: Vec<f32> = (0..n_embd * n_embd)
            .map(|_| next_f32() * xavier_fc3)
            .collect();
        let fc3_bias = vec![0.0f32; n_embd];

        Self {
            n_embd,
            norm_weight,
            norm_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        }
    }
}

// ── Entropy Computation ────────────────────────────────────────

/// Compute Shannon entropy from log-probabilities (natural log).
/// `log_probs` must contain log-softmax outputs.
/// Returns entropy in nats.
#[inline]
fn entropy_from_log_probs(log_probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &lp in log_probs {
        if lp > -50.0 {
            // skip near-zero probs to avoid -0 * inf
            let p = lp.exp();
            h -= p * lp;
        }
    }
    h
}

/// Convert logits to log-probabilities (log-softmax) in-place.
#[inline]
fn log_softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Shift by −max in place AND accumulate sum_exp in the same pass, then
    // subtract ln(sum_exp). Result: x_i − max − ln(sum_exp) = log_softmax(x_i).
    // This avoids the old exp→store→sum→per-element ln() undo path that
    // issued N ln() calls; now there is exactly one ln() call total.
    let mut sum_exp = 0.0f32;
    for v in logits.iter_mut() {
        *v -= max;
        sum_exp += v.exp();
    }
    let log_sum = sum_exp.ln();
    for v in logits.iter_mut() {
        *v -= log_sum;
    }
}

/// Sample greedily (argmax) from logits. Returns (token_idx, log_prob).
#[inline]
fn greedy_sample(logits: &[f32]) -> (usize, f32) {
    let (idx, &val) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0f32));
    (idx, val)
}

// ── BeliefDrafter ───────────────────────────────────────────────

/// A single drafted token from the belief drafter.
#[derive(Clone, Debug, PartialEq)]
pub struct BeliefDraftToken {
    /// Token index in the vocabulary.
    pub token_idx: usize,
    /// Log-probability (from log-softmax over logits).
    pub log_prob: f32,
    /// Entropy of the distribution at this draft step (in nats).
    pub entropy: f32,
}

/// Error type for `BeliefDrafter` operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BeliefDraftError {
    /// MLP not initialized.
    NotInitialized,
    /// Output head (lm_head) has wrong dimensions.
    OutputHeadDimensionMismatch,
}

/// Condition for `BeliefDrafter` as a `SpeculativeGenerator`.
#[derive(Clone, Debug)]
pub struct BeliefDraftCondition {
    /// Current hidden state `[n_embd]`.
    pub h_t: Vec<f32>,
    /// Maximum draft steps.
    pub max_steps: usize,
    /// Entropy threshold — stop drafting when entropy exceeds this.
    pub entropy_threshold: f32,
}

/// NextLat Belief-State Speculative Drafter.
///
/// Wraps a `LatentDynamicsMLP` with a shared output head (lm_head) from the target model.
/// The `draft()` method recursively predicts next hidden states, projects them through
/// the output head to get logits, and samples tokens with entropy-gated stopping.
///
/// ## Integration
///
/// - `SpeculativeGenerator` trait: produces `Vec<BeliefDraftToken>` candidates
/// - `DecodeStage::BeliefDraft`: pipeline integration point
/// - `Config::belief_drafter_path`: load MLP weights from `nextlat.bin`
///
/// ## Entropy-Gated Stopping
///
/// At each recursive step, entropy of the output distribution is computed.
/// If entropy exceeds `entropy_threshold`, drafting stops — high entropy means the
/// drafter is uncertain, and continuing would degrade acceptance rate.
#[derive(Debug)]
pub struct BeliefDrafter {
    /// The latent dynamics MLP.
    mlp: LatentDynamicsMLP,
    /// Output head weights: `[vocab_size, n_embd]` row-major (shared with target model).
    output_head: Vec<f32>,
    /// Vocabulary size.
    vocab_size: usize,
    /// Embedding table for token lookup: `[vocab_size, n_embd]` row-major.
    /// Can be a reference to the target model's `wte`.
    wte: Vec<f32>,
}

impl BeliefDrafter {
    /// Create a new `BeliefDrafter` with the given MLP, output head, and embedding table.
    ///
    /// - `mlp`: The latent dynamics MLP (loaded from bin or random init).
    /// - `output_head`: LM head weights `[vocab_size, n_embd]` (shared with target model).
    /// - `wte`: Token embedding table `[vocab_size, n_embd]` (shared with target model).
    pub fn new(
        mlp: LatentDynamicsMLP,
        output_head: Vec<f32>,
        wte: Vec<f32>,
    ) -> Result<Self, BeliefDraftError> {
        let vocab_size = output_head.len() / mlp.n_embd;
        if output_head.len() != vocab_size * mlp.n_embd {
            return Err(BeliefDraftError::OutputHeadDimensionMismatch);
        }
        if wte.len() != vocab_size * mlp.n_embd {
            return Err(BeliefDraftError::OutputHeadDimensionMismatch);
        }
        Ok(Self {
            mlp,
            output_head,
            vocab_size,
            wte,
        })
    }

    /// Create with random init MLP from config dimensions.
    pub fn random_init(config: &Config) -> Self {
        let n = config.n_embd;
        let mlp = LatentDynamicsMLP::random_init(n);
        // Dummy output head + wte — in production these come from the target model
        let vocab_size = config.vocab_size.max(1);
        let output_head = vec![0.0f32; vocab_size * n];
        let wte = vec![0.0f32; vocab_size * n];
        Self {
            mlp,
            output_head,
            vocab_size,
            wte,
        }
    }

    /// Project hidden state through output head to get logits.
    /// Returns logits `[vocab_size]`.
    #[allow(dead_code)]
    fn logits_from_hidden(&self, h: &[f32]) -> Vec<f32> {
        let n = self.mlp.n_embd;
        let vs = self.vocab_size;
        let mut logits = vec![0.0f32; vs];
        // output_head: [vocab_size, n_embd] row-major
        // logits[i] = dot(output_head[i*n..(i+1)*n], h)
        for (i, slot) in logits.iter_mut().enumerate().take(vs) {
            let row_off = i * n;
            *slot = simd_dot_f32(&self.output_head[row_off..row_off + n], h, n);
        }
        logits
    }

    /// Get the embedding for a token.
    #[inline]
    fn token_embedding(&self, token_idx: usize) -> &[f32] {
        let n = self.mlp.n_embd;
        let off = token_idx * n;
        if off + n <= self.wte.len() {
            &self.wte[off..off + n]
        } else {
            // Fallback: return zeros (shouldn't happen with valid token indices)
            &self.wte[..n] // safe because n > 0 and wte.len() >= n
        }
    }

    /// Draft variable-length token sequence from current hidden state.
    ///
    /// Recursively applies the MLP to predict next hidden states:
    /// 1. Project `h_t` through output head → logits
    /// 2. Greedy sample token, compute entropy
    /// 3. If entropy > threshold, stop
    /// 4. Otherwise: `h_{t+1} = mlp.forward(h_t, emb(token))`, go to 1
    ///
    /// Returns the drafted tokens (at least 1, at most `max_steps`).
    pub fn draft(
        &self,
        h_t: &[f32],
        max_steps: usize,
        entropy_threshold: f32,
    ) -> Vec<BeliefDraftToken> {
        let n = self.mlp.n_embd;
        assert_eq!(h_t.len(), n, "h_t must have length n_embd");

        let mut drafts = Vec::with_capacity(max_steps);
        // Double-buffered hidden state: forward_into writes h_{t+1} into the
        // inactive buffer while reading h_t from the active one, then we swap.
        // This removes the per-step `h_current = mlp.forward(...)` allocation
        // (the prior `h_t.to_vec()` + a fresh result Vec every step).
        let mut h_a = h_t.to_vec();
        let mut h_b = vec![0.0f32; n];

        // Pre-allocate scratch for log-softmax and for the MLP forward pass
        // (reused across all steps — see MlpForwardScratch).
        let mut logits_buf = vec![0.0f32; self.vocab_size];
        let mut mlp_scratch = MlpForwardScratch::new(n);

        for _ in 0..max_steps {
            // 1. Project hidden → logits
            self.logits_from_hidden_into(&h_a, &mut logits_buf);

            // 2. Convert to log-probs + compute entropy
            log_softmax_inplace(&mut logits_buf);
            let ent = entropy_from_log_probs(&logits_buf);

            // 3. Greedy sample
            let (token_idx, log_prob) = greedy_sample(&logits_buf);

            // 4. Record draft
            drafts.push(BeliefDraftToken {
                token_idx,
                log_prob,
                entropy: ent,
            });

            // 5. Entropy-gated stop (after first token — always draft at least 1)
            if drafts.len() > 1 && ent > entropy_threshold {
                break;
            }

            // 6. Get embedding for drafted token, advance hidden state.
            //    forward_into reads h_a, writes h_b; swap for next iteration.
            let emb = self.token_embedding(token_idx);
            self.mlp.forward_into(&h_a, emb, &mut mlp_scratch, &mut h_b);
            std::mem::swap(&mut h_a, &mut h_b);
        }

        drafts
    }

    /// Draft variable-length token sequence with self-conditioning from previous prediction.
    ///
    /// After each recursive step, the predicted hidden state is used as the SC signal
    /// for the *next* step. This creates a feedback loop where the drafter refines its
    /// own predictions, similar to Chen et al. (2022) self-conditioning for diffusion.
    ///
    /// - `h_t`: initial hidden state `[n_embd]`
    /// - `max_steps`: maximum draft steps
    /// - `entropy_threshold`: stop when entropy exceeds this
    /// - `sc_scale`: blending factor for SC injection (typical: 0.1–0.5). 0.0 = no SC.
    /// - `initial_sc`: optional initial SC signal `[n_embd]`. None = zeros (no initial SC).
    ///
    /// Returns drafted tokens (at least 1, at most `max_steps`).
    #[cfg(feature = "self_cond_draft")]
    pub fn draft_with_sc(
        &self,
        h_t: &[f32],
        max_steps: usize,
        entropy_threshold: f32,
        sc_scale: f32,
        initial_sc: Option<&[f32]>,
    ) -> Vec<BeliefDraftToken> {
        let n = self.mlp.n_embd;
        assert_eq!(h_t.len(), n, "h_t must have length n_embd");

        let mut drafts = Vec::with_capacity(max_steps);
        // Double-buffered hidden state (h_a = current input, h_b = next output).
        let mut h_a = h_t.to_vec();
        let mut h_b = vec![0.0f32; n];

        // SC buffer starts with initial signal or zeros.
        let mut sc_current: Vec<f32> = match initial_sc {
            Some(sc) => {
                assert_eq!(sc.len(), n, "initial_sc must have length n_embd");
                sc.to_vec()
            }
            None => vec![0.0f32; n],
        };
        // Previous hidden state used as the SC signal next iteration; reused
        // across steps to avoid the prior per-step `h_current.clone()`.
        let mut prev_h = vec![0.0f32; n];
        // Blended embedding scratch (next_emb + scale * sc).
        let mut blended_emb = vec![0.0f32; n];

        // Pre-allocate scratch for log-softmax and the MLP forward pass.
        let mut logits_buf = vec![0.0f32; self.vocab_size];
        let mut mlp_scratch = MlpForwardScratch::new(n);

        for _ in 0..max_steps {
            // 1. Project hidden → logits
            self.logits_from_hidden_into(&h_a, &mut logits_buf);

            // 2. Convert to log-probs + compute entropy
            log_softmax_inplace(&mut logits_buf);
            let ent = entropy_from_log_probs(&logits_buf);

            // 3. Greedy sample
            let (token_idx, log_prob) = greedy_sample(&logits_buf);

            // 4. Record draft
            drafts.push(BeliefDraftToken {
                token_idx,
                log_prob,
                entropy: ent,
            });

            // 5. Entropy-gated stop (after first token — always draft at least 1)
            if drafts.len() > 1 && ent > entropy_threshold {
                break;
            }

            // 6. Get embedding for drafted token
            let emb = self.token_embedding(token_idx);

            // 7. Snapshot current hidden state into prev_h (reused buffer) —
            //    becomes the SC signal next iteration.
            prev_h.copy_from_slice(&h_a);

            // 8. Blend SC into embedding: next_emb' = next_emb + scale * sc.
            for i in 0..n {
                blended_emb[i] = emb[i] + sc_scale * sc_current[i];
            }

            // 9. Advance: h_{t+1} = mlp.forward(h_t, blended_emb) via the
            //    zero-alloc forward_into (reads h_a, writes h_b); then swap.
            self.mlp
                .forward_into(&h_a, &blended_emb, &mut mlp_scratch, &mut h_b);
            std::mem::swap(&mut h_a, &mut h_b);

            // 10. Update SC signal for next iteration (previous hidden state).
            std::mem::swap(&mut sc_current, &mut prev_h);
        }

        drafts
    }

    /// Project hidden state through output head into pre-allocated buffer.
    fn logits_from_hidden_into(&self, h: &[f32], logits: &mut [f32]) {
        let n = self.mlp.n_embd;
        let vs = self.vocab_size;
        debug_assert_eq!(logits.len(), vs);
        for (i, slot) in logits.iter_mut().enumerate().take(vs) {
            let row_off = i * n;
            *slot = simd_dot_f32(&self.output_head[row_off..row_off + n], h, n);
        }
    }

    /// Get the MLP's n_embd dimension.
    #[inline]
    pub fn n_embd(&self) -> usize {
        self.mlp.n_embd
    }

    /// Get the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Load `BeliefDrafter` from a binary MLP file + target model weights.
    pub fn load_from_bin(
        path: &Path,
        output_head: Vec<f32>,
        wte: Vec<f32>,
    ) -> Result<Self, String> {
        let mlp = LatentDynamicsMLP::load_from_bin(path)?;
        let n = mlp.n_embd;
        let vocab_size = output_head.len() / n;
        if output_head.len() != vocab_size * n {
            return Err(format!(
                "output_head length {} not divisible by n_embd {}",
                output_head.len(),
                n
            ));
        }
        if wte.len() != vocab_size * n {
            return Err(format!(
                "wte length {} not divisible by n_embd {}",
                wte.len(),
                n
            ));
        }
        Ok(Self {
            mlp,
            output_head,
            vocab_size,
            wte,
        })
    }

    /// Recursively advance the MLP for `max_depth` steps and classify the hidden
    /// state chain with [`katgpt_core::classify_chain`].
    ///
    /// Drives the drafter with an external token sequence: at each step the
    /// next token's embedding is fed as `next_emb` to
    /// [`LatentDynamicsMLP::forward_into`], producing `h_{t+1} = h_t +
    /// FC3(GELU(FC2(GELU(FC1(LN(concat(h_t, next_emb)))))))`. The chain
    /// `h_0, h_1, …, h_k` (with `k = min(max_depth, token_seq.len())`) is
    /// captured into a flattened buffer and classified.
    ///
    /// Reuses the drafter's own `MlpForwardScratch` + double-buffered `h_a` /
    /// `h_b` pattern from [`Self::draft`] (zero per-step allocation; only one
    /// `Vec::with_capacity` for the chain buffer up-front). The depth-invariance
    /// `Scratch` is allocated inside this call — callers running many audits in
    /// a tight loop should re-use one themselves via the raw
    /// [`katgpt_core::classify_chain`] primitive.
    ///
    /// # Plan 306 Phase 3 (G2 — paper finding reproduction)
    ///
    /// Paper arXiv:2605.09992 §3 shows pre-norm EAGLE-3 drafters classify as
    /// [`DepthSpecificRefinement`] beyond TTT. Our `LatentDynamicsMLP` has the
    /// same structural shape; random-init results may differ from trained
    /// (Xavier init bounds FC3 output) — informative either way. See Plan 306
    /// §T3.2 doc for the random-init caveat.
    ///
    /// Returns [`DepthInvarianceDiagnostic::kind`] == [`Insufficient`] if
    /// `max_depth + 1 < cfg.min_samples`.
    ///
    /// [`DepthSpecificRefinement`]: katgpt_core::DepthInvarianceKind::DepthSpecificRefinement
    /// [`Insufficient`]: katgpt_core::DepthInvarianceKind::Insufficient
    #[cfg(feature = "depth_invariance")]
    pub fn audit_depth_invariance(
        &self,
        h_0: &[f32],
        token_seq: &[usize],
        max_depth: usize,
        cfg: &katgpt_core::DepthInvarianceConfig,
    ) -> katgpt_core::DepthInvarianceDiagnostic {
        // No regularization on the plain audit — the audit is a measurement
        // of the drafter as-shipped, not as-regularized.
        let chain = self.capture_chain(h_0, token_seq, max_depth, katgpt_core::MagnitudeRegularization::None);
        let k_plus_1 = chain.len() / self.mlp.n_embd;
        let mut scratch = katgpt_core::Scratch::with_capacity(k_plus_1, self.mlp.n_embd);
        katgpt_core::classify_chain(&chain, self.mlp.n_embd, cfg, &mut scratch)
    }

    /// Plan 306 Phase 3 G2c — capture the hidden-state chain with optional
    /// inference-time [`katgpt_core::MagnitudeRegularization`] applied to
    /// `h_{t+1}` after each forward step.
    ///
    /// Returns the flattened chain `[k+1][n_embd]` row-major. The caller is
    /// responsible for `classify_chain` on the result. Exposed publicly so
    /// tests can interleave custom regularization schedules without re-implementing
    /// the double-buffered forward loop.
    ///
    /// # Diagnostic intent (Plan 306 §T3.4)
    ///
    /// For our frozen-pretrained drafter, applying `MagnitudeRegularization`
    /// here is **diagnostic-only** — paper §4.4 Table 4 reports -56%
    /// acceptance on pre-norm models when applied at inference time. The
    /// shipped fix requires MLP retraining (→ riir-train). For kernels we own
    /// (HLA, functor, micro_belief, engram, Raven) this same primitive is the
    /// modelless upstream fix.
    #[cfg(feature = "depth_invariance")]
    pub fn capture_chain(
        &self,
        h_0: &[f32],
        token_seq: &[usize],
        max_depth: usize,
        regularization: katgpt_core::MagnitudeRegularization,
    ) -> Vec<f32> {
        let n = self.mlp.n_embd;
        assert_eq!(h_0.len(), n, "h_0 must have length n_embd");

        let k = max_depth.min(token_seq.len());
        let k_plus_1 = k + 1;

        let mut chain: Vec<f32> = Vec::with_capacity(k_plus_1 * n);
        chain.extend_from_slice(h_0);

        let mut h_a: Vec<f32> = h_0.to_vec();
        let mut h_b: Vec<f32> = vec![0.0f32; n];
        let mut mlp_scratch = MlpForwardScratch::new(n);
        // Scratch for the optional RmsNorm/ScalarPinch path. Length-d as per
        // the apply_magnitude_regularization contract (currently unused by
        // RmsNorm but required for API stability — see module doc).
        let mut reg_scratch: Vec<f32> = vec![0.0f32; n];

        for &tok in &token_seq[..k] {
            let emb = self.token_embedding(tok);
            self.mlp.forward_into(&h_a, emb, &mut mlp_scratch, &mut h_b);
            katgpt_core::apply_magnitude_regularization(&mut h_b, regularization, &mut reg_scratch);
            chain.extend_from_slice(&h_b[..n]);
            std::mem::swap(&mut h_a, &mut h_b);
        }

        chain
    }
}

// ── SpeculativeGenerator Integration ─────────────────────────────

impl SpeculativeGenerator for BeliefDrafter {
    type Condition = BeliefDraftCondition;
    type Output = BeliefDraftToken;
    type Error = BeliefDraftError;

    fn generate(
        &mut self,
        condition: &Self::Condition,
        _rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        if condition.h_t.len() != self.mlp.n_embd {
            return Err(BeliefDraftError::NotInitialized);
        }
        Ok(self.draft(
            &condition.h_t,
            condition.max_steps,
            condition.entropy_threshold,
        ))
    }
}

// ── Binary I/O Helpers ─────────────────────────────────────────

fn read_u32(rdr: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    rdr.read_exact(&mut buf)
        .map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u32(wtr: &mut impl Write, val: u32) -> Result<(), String> {
    wtr.write_all(&val.to_le_bytes())
        .map_err(|e| format!("write u32: {e}"))?;
    Ok(())
}

fn read_f32_vec(rdr: &mut impl Read, expected_len: usize, label: &str) -> Result<Vec<f32>, String> {
    // Allocate the f32 buffer directly and read bytes in place via bytemuck,
    // avoiding the old intermediate Vec<u8> + N `f32::from_le_bytes` calls.
    // Assumes a little-endian target (matches `write_f32_slice`'s `to_le_bytes`);
    // katgpt-rs targets LE (x86_64/aarch64 macOS).
    let mut vec: Vec<f32> = vec![0.0f32; expected_len];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(vec.as_mut_slice());
    rdr.read_exact(bytes)
        .map_err(|e| format!("read {label}: {e}"))?;
    debug_assert_eq!(vec.len(), expected_len);
    Ok(vec)
}

fn write_f32_slice(wtr: &mut impl Write, data: &[f32]) -> Result<(), String> {
    for &v in data {
        wtr.write_all(&v.to_le_bytes())
            .map_err(|e| format!("write f32: {e}"))?;
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_mlp_forward_shape_micro() {
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.5f32; n_embd];
        let next_emb = vec![0.3f32; n_embd];
        let output = mlp.forward(&h_t, &next_emb);
        assert_eq!(output.len(), n_embd, "output must have length n_embd=16");
    }

    #[test]
    fn test_mlp_forward_shape_bpe() {
        let n_embd = 32;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.5f32; n_embd];
        let next_emb = vec![0.3f32; n_embd];
        let output = mlp.forward(&h_t, &next_emb);
        assert_eq!(output.len(), n_embd, "output must have length n_embd=32");
    }

    #[test]
    fn test_mlp_residual_connection() {
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t: Vec<f32> = (0..n_embd).map(|i| i as f32 * 0.1).collect();
        let next_emb = vec![1.0f32; n_embd];

        let output = mlp.forward(&h_t, &next_emb);

        // The output should NOT equal h_t exactly (FC3 output is nonzero for non-zero input)
        // and should NOT equal just the FC3 output (it's h_t + FC3, not just FC3)
        // Verify residual: output[i] != h_t[i] for at least some i (FC3 is nonzero)
        let any_different = output
            .iter()
            .zip(h_t.iter())
            .any(|(&o, &h)| (o - h).abs() > 1e-6);
        assert!(
            any_different,
            "residual connection must produce output != h_t"
        );
    }

    #[test]
    fn test_random_init_produces_valid_mlp() {
        for &n_embd in &[16usize, 32] {
            let mlp = LatentDynamicsMLP::random_init(n_embd);
            let h_t = vec![1.0f32; n_embd];
            let next_emb = vec![-0.5f32; n_embd];
            let output = mlp.forward(&h_t, &next_emb);

            for (i, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "output[{i}] is not finite (n_embd={n_embd}): {v}"
                );
            }
        }
    }

    #[test]
    fn test_load_from_bin_roundtrip() {
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.7f32; n_embd];
        let next_emb = vec![0.2f32; n_embd];
        let expected = mlp.forward(&h_t, &next_emb);

        // Write to temp file
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nextlat_test.bin");
        mlp.save_to_bin(&path).expect("save");

        // Load back
        let loaded = LatentDynamicsMLP::load_from_bin(&path).expect("load");

        // Verify dimensions match
        assert_eq!(loaded.n_embd, n_embd);

        // Forward pass must produce identical output
        let actual = loaded.forward(&h_t, &next_emb);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "roundtrip mismatch at [{i}]: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn test_load_from_bin_bad_magic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad_magic.bin");
        let mut file = std::fs::File::create(&path).expect("create");
        file.write_all(b"XXXX").expect("write bad magic");
        file.write_all(&1u32.to_le_bytes()).expect("write version");
        file.write_all(&16u32.to_le_bytes()).expect("write n_embd");
        drop(file);

        let result = LatentDynamicsMLP::load_from_bin(&path);
        // Verify that loading with a bad magic byte sequence produces an error
        if let Err(msg) = &result {
            assert!(
                msg.contains("bad magic"),
                "expected bad magic error, got: {msg}"
            );
        } else {
            panic!("expected bad magic error, got: {result:?}");
        }
    }

    #[test]
    fn test_gelu_sanity() {
        // GELU(0) ≈ 0
        assert!(gelu(0.0).abs() < 1e-6, "gelu(0) should be ~0");

        // GELU(large positive) > 0
        assert!(gelu(10.0) > 0.0, "gelu(10) should be positive");
        assert!((gelu(10.0) - 10.0).abs() < 0.1, "gelu(10) should be ~10");

        // GELU(negative) < 0 (for moderately negative values)
        assert!(gelu(-1.0) < 0.0, "gelu(-1) should be negative");

        // GELU is approximately identity for large positive
        assert!(gelu(5.0) > 4.9, "gelu(5) should be ~5");

        // GELU approaches 0 for large negative
        assert!(gelu(-10.0).abs() < 0.01, "gelu(-10) should be ~0");
    }

    // ── Phase 1: BeliefDrafter Tests ─────────────────────────────

    /// Helper: create a simple BeliefDrafter with known weights.
    /// n_embd=4, vocab_size=3 for minimal test surface.
    fn make_test_drafter() -> BeliefDrafter {
        let n_embd = 4;
        let vocab_size = 3;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        // Non-zero output head so logits are meaningful
        let output_head: Vec<f32> = (0..vocab_size * n_embd)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let wte: Vec<f32> = (0..vocab_size * n_embd)
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();
        BeliefDrafter::new(mlp, output_head, wte).expect("valid drafter")
    }

    #[test]
    fn test_belief_drafter_new_valid() {
        let drafter = make_test_drafter();
        assert_eq!(drafter.n_embd(), 4);
        assert_eq!(drafter.vocab_size(), 3);
    }

    #[test]
    fn test_belief_drafter_new_dimension_mismatch() {
        let n_embd = 4;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let bad_output_head = vec![0.0f32; 7]; // not divisible by 4
        let wte = vec![0.0f32; 8];
        let result = BeliefDrafter::new(mlp, bad_output_head, wte);
        assert_eq!(
            result.unwrap_err(),
            BeliefDraftError::OutputHeadDimensionMismatch
        );
    }

    #[test]
    fn test_belief_drafter_draft_produces_tokens() {
        let drafter = make_test_drafter();
        let h_t = vec![1.0f32; 4];
        let drafts = drafter.draft(&h_t, 5, 10.0); // high entropy threshold → all 5 steps

        assert!(!drafts.is_empty(), "should produce at least 1 draft");
        assert!(drafts.len() <= 5, "should not exceed max_steps");

        for (i, token) in drafts.iter().enumerate() {
            assert!(
                token.token_idx < 3,
                "token {} index {} must be < vocab_size=3",
                i,
                token.token_idx
            );
            assert!(
                token.log_prob.is_finite(),
                "token {} log_prob must be finite",
                i
            );
            assert!(
                token.entropy >= 0.0,
                "token {} entropy must be non-negative",
                i
            );
        }
    }

    #[test]
    fn test_belief_drafter_draft_always_at_least_one() {
        let drafter = make_test_drafter();
        let h_t = vec![0.5f32; 4];
        // Very low entropy threshold — should still produce at least 1 token
        let drafts = drafter.draft(&h_t, 5, 0.0);
        assert!(!drafts.is_empty(), "must produce at least 1 draft token");
    }

    #[test]
    fn test_belief_drafter_draft_entropy_gating() {
        let drafter = make_test_drafter();
        let h_t = vec![1.0f32; 4];

        // With very high threshold, should draft max_steps tokens
        let drafts_full = drafter.draft(&h_t, 5, 100.0);
        assert_eq!(
            drafts_full.len(),
            5,
            "high threshold should draft all steps"
        );

        // With zero threshold, should draft 1-2 tokens (stops after 2nd if entropy > 0)
        let drafts_limited = drafter.draft(&h_t, 5, 0.0);
        assert!(
            !drafts_limited.is_empty() && drafts_limited.len() <= 5,
            "low threshold should produce 1-5 tokens, got {}",
            drafts_limited.len()
        );
    }

    #[test]
    fn test_belief_drafter_draft_token_indices_valid() {
        let drafter = make_test_drafter();
        let h_t = vec![0.3f32; 4];
        let drafts = drafter.draft(&h_t, 10, 100.0);

        for (step, token) in drafts.iter().enumerate() {
            assert!(
                token.token_idx < drafter.vocab_size(),
                "step {}: token_idx {} >= vocab_size {}",
                step,
                token.token_idx,
                drafter.vocab_size()
            );
        }
    }

    #[test]
    fn test_belief_drafter_speculative_generator() {
        use katgpt_core::SpeculativeGenerator;

        let mut drafter = make_test_drafter();
        let condition = BeliefDraftCondition {
            h_t: vec![1.0f32; 4],
            max_steps: 3,
            entropy_threshold: 10.0,
        };
        let mut rng = fastrand::Rng::new();
        let result = drafter.generate(&condition, &mut rng).expect("generate");

        assert!(!result.is_empty());
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_belief_drafter_speculative_generator_wrong_dimension() {
        use katgpt_core::SpeculativeGenerator;

        let mut drafter = make_test_drafter();
        let condition = BeliefDraftCondition {
            h_t: vec![1.0f32; 8], // wrong dimension
            max_steps: 3,
            entropy_threshold: 10.0,
        };
        let mut rng = fastrand::Rng::new();
        let result = drafter.generate(&condition, &mut rng);
        assert_eq!(result.unwrap_err(), BeliefDraftError::NotInitialized);
    }

    #[test]
    fn test_belief_drafter_load_from_bin() {
        let n_embd = 4;
        let vocab_size = 3;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let output_head: Vec<f32> = (0..vocab_size * n_embd).map(|i| (i as f32) * 0.1).collect();
        let wte: Vec<f32> = (0..vocab_size * n_embd)
            .map(|i| (i as f32) * 0.05)
            .collect();

        let h_t = vec![1.0f32; n_embd];

        // Create drafter, draft tokens
        let drafter = BeliefDrafter::new(mlp, output_head.clone(), wte.clone()).expect("new");
        let expected = drafter.draft(&h_t, 3, 100.0);

        // Save MLP, reload, recreate drafter
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nextlat.bin");
        drafter.mlp.save_to_bin(&path).expect("save");

        let reloaded = BeliefDrafter::load_from_bin(&path, output_head, wte).expect("load");
        let actual = reloaded.draft(&h_t, 3, 100.0);

        assert_eq!(expected.len(), actual.len());
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert_eq!(e.token_idx, a.token_idx, "token mismatch at step {i}");
            assert!(
                (e.log_prob - a.log_prob).abs() < 1e-5,
                "log_prob mismatch at step {i}"
            );
        }
    }

    #[test]
    fn test_log_softmax_sums_to_one() {
        let mut logits = [1.0f32, 2.0f32, 3.0f32];
        log_softmax_inplace(&mut logits);
        let sum_exp: f32 = logits.iter().map(|&lp| lp.exp()).sum();
        assert!(
            (sum_exp - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum_exp}"
        );
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform over 4 items: entropy = ln(4) ≈ 1.386
        let log_probs = vec![-4.0f32.ln(); 4];
        let ent = entropy_from_log_probs(&log_probs);
        assert!(
            (ent - 4.0f32.ln()).abs() < 1e-4,
            "expected ln(4), got {ent}"
        );
    }

    #[test]
    fn test_greedy_sample_argmax() {
        let logits = [0.1f32, 0.5f32, 0.3f32];
        let (idx, _) = greedy_sample(&logits);
        assert_eq!(idx, 1, "should pick index 1 (0.5)");
    }

    // ── Self-Conditioning Tests (Plan 222 T11, feature: self_cond_draft) ──────

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_forward_with_sc_zero_scale_is_noop() {
        // sc_scale=0.0 must produce identical output to forward() without SC
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.5f32; n_embd];
        let next_emb = vec![0.3f32; n_embd];
        let sc = vec![1.0f32; n_embd];

        let expected = mlp.forward(&h_t, &next_emb);
        let actual = mlp.forward_with_sc(&h_t, &next_emb, &sc, 0.0);

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "SC noop mismatch at [{i}]: got {a}, expected {e}"
            );
        }
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_forward_with_sc_nonzero_differs() {
        // sc_scale > 0 must produce different output from forward() without SC
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.5f32; n_embd];
        let next_emb = vec![0.3f32; n_embd];
        let sc = vec![1.0f32; n_embd];

        let baseline = mlp.forward(&h_t, &next_emb);
        let sc_output = mlp.forward_with_sc(&h_t, &next_emb, &sc, 0.5);

        let any_different = baseline
            .iter()
            .zip(sc_output.iter())
            .any(|(&b, &s)| (b - s).abs() > 1e-6);
        assert!(any_different, "SC with nonzero scale must change output");
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_forward_with_sc_preserves_shape() {
        let n_embd = 32;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.5f32; n_embd];
        let next_emb = vec![0.3f32; n_embd];
        let sc = vec![0.7f32; n_embd];

        let output = mlp.forward_with_sc(&h_t, &next_emb, &sc, 0.3);
        assert_eq!(output.len(), n_embd, "output must have length n_embd");

        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] must be finite: {v}");
        }
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_draft_with_sc_produces_valid_tokens() {
        let drafter = make_test_drafter();
        let h_t = vec![1.0f32; 4];
        let drafts = drafter.draft_with_sc(&h_t, 5, 100.0, 0.3, None);

        assert!(!drafts.is_empty(), "should produce at least 1 draft");
        assert!(drafts.len() <= 5, "should not exceed max_steps");

        for (i, token) in drafts.iter().enumerate() {
            assert!(
                token.token_idx < 3,
                "token {} index {} must be < vocab_size=3",
                i,
                token.token_idx
            );
            assert!(
                token.log_prob.is_finite(),
                "token {} log_prob must be finite",
                i
            );
            assert!(
                token.entropy >= 0.0,
                "token {} entropy must be non-negative",
                i
            );
        }
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_draft_with_sc_always_at_least_one() {
        let drafter = make_test_drafter();
        let h_t = vec![0.5f32; 4];
        // Very low entropy threshold — should still produce at least 1 token
        let drafts = drafter.draft_with_sc(&h_t, 5, 0.0, 0.5, None);
        assert!(!drafts.is_empty(), "must produce at least 1 draft token");
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_draft_with_sc_with_initial_signal() {
        let drafter = make_test_drafter();
        let h_t = vec![1.0f32; 4];
        let initial_sc = vec![0.8f32; 4];

        let drafts = drafter.draft_with_sc(&h_t, 5, 100.0, 0.3, Some(&initial_sc));

        assert!(
            !drafts.is_empty(),
            "should produce at least 1 draft with initial SC"
        );
        for token in &drafts {
            assert!(token.token_idx < 3, "token must be < vocab_size");
        }
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_draft_with_sc_zero_scale_matches_draft() {
        // sc_scale=0.0 should produce same tokens as draft() without SC
        let drafter = make_test_drafter();
        let h_t = vec![1.0f32; 4];

        let baseline = drafter.draft(&h_t, 5, 100.0);
        let sc_draft = drafter.draft_with_sc(&h_t, 5, 100.0, 0.0, None);

        assert_eq!(baseline.len(), sc_draft.len(), "same number of draft steps");
        for (i, (b, s)) in baseline.iter().zip(sc_draft.iter()).enumerate() {
            assert_eq!(
                b.token_idx, s.token_idx,
                "step {i}: zero-scale SC must match baseline"
            );
        }
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_draft_with_sc_entropy_gating() {
        let drafter = make_test_drafter();
        let h_t = vec![1.0f32; 4];

        // High threshold → draft all steps
        let drafts_full = drafter.draft_with_sc(&h_t, 5, 100.0, 0.3, None);
        assert_eq!(
            drafts_full.len(),
            5,
            "high threshold should draft all steps"
        );

        // Zero threshold → draft 1-2 tokens
        let drafts_limited = drafter.draft_with_sc(&h_t, 5, 0.0, 0.3, None);
        assert!(!drafts_limited.is_empty(), "must produce at least 1 token");
    }
}
