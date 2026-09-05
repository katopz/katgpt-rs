//! Weaver inference-only logit corrector (Issue 131).
//!
//! Mirrors riir-train-engine's Weaver adapter as a read-only inference module.
//! Loads trained weights from a safetensors checkpoint, runs the 7-step forward
//! pass (conditioning → causal attention → SwiGLU → top-K gather → residual add
//! → renormalize), and applies the residual correction to DFlash draft logits
//! over the top-K candidate set.
//!
//! ## Architecture
//!
//! ```text
//!  h_verifier ──► RMSNorm(norm_cond) ──► W_c ──┐
//!  h_dflash[D] ─► RMSNorm(norm_cond) ──► W_c ──┤ + pos_emb ──► u_seq[D+1]
//!                                                │
//!  u_seq ──► Wq/Wk/Wv ──► causal MHA ──► W_o + residual ──► RMSNorm(norm_attn)
//!                                                                       │
//!                    SwiGLU(W_gate, W_up) → W_down + residual ──► RMSNorm(norm_mlp)
//!                                                                       │
//!                  u_final[D+1] ──► top-K gather: residual_k = <h, emb[topk_k]>
//!                                                                       │
//!        corrected = dflash_logits + weaver_residual ──► softmax over K
//! ```
//!
//! ## No-harm contract
//!
//! Zero-init weights produce **zero residual** (all matmul outputs are zero,
//! all RMSNorm scales are zero → inv_rms is finite but the dot product with
//! zero embedding rows is still zero). This means the corrector is a safe
//! no-op when no trained checkpoint is available. The feature is opt-in
//! (`weaver_runtime`); when off, DFlash behavior is bit-identical.
//!
//! ## References
//!
//! - riir-train `crates/riir-train-engine/src/weaver.rs` — training-side model
//! - riir-train Plan 314 — training plan (synthetic GOAT gate DONE)
//! - katgpt-rs Issue 131 — this integration
//! - arXiv:2607.06763 §3.2 — "Trees from Marginals" (Oda et al.)

use core::f32;

// ── Config ───────────────────────────────────────────────────────────────

/// Weaver model hyperparameters. Recovered from safetensors metadata on load.
#[derive(Debug, Clone)]
pub struct WeaverConfig {
    /// Hidden dimension (d_model). Paper default 2048; the real-data precompute
    /// uses 2304 (Gemma2-2B).
    pub hidden_dim: usize,
    /// Number of attention heads. head_dim = hidden_dim / n_heads.
    pub n_heads: usize,
    /// Candidate token count K. Weaver only projects over these.
    pub k_candidates: usize,
    /// Transformer layers. Fixed at 1 per the paper.
    pub n_layer: usize,
    /// SwiGLU intermediate dimension.
    pub d_ff: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Maximum drafter depth D. Position embeddings allocated for [0, D).
    pub max_depth: usize,
}

impl Default for WeaverConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            n_heads: 16,
            k_candidates: 512,
            n_layer: 1,
            d_ff: 5824,
            rms_eps: 1e-6,
            max_depth: 8,
        }
    }
}

impl WeaverConfig {
    /// Per-head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }
}

// ── Weights ──────────────────────────────────────────────────────────────

/// Weaver learnable weights. Stored as flat `Vec<f32>` (row-major).
///
/// All matrices use `[in_dim, out_dim]` row-major layout:
/// `output[j] = Σ_i input[i] · weight[i · out_dim + j]`.
///
/// This is a read-only mirror of riir-train's `WeaverWeights` — no optimizer
/// state, no gradient buffers.
#[derive(Debug, Clone)]
pub struct WeaverWeights {
    /// Conditioning projection W_c `[hidden, hidden]`.
    pub w_c: Vec<f32>,
    /// Attention query projection `[hidden, hidden]`.
    pub w_q: Vec<f32>,
    /// Attention key projection `[hidden, hidden]`.
    pub w_k: Vec<f32>,
    /// Attention value projection `[hidden, hidden]`.
    pub w_v: Vec<f32>,
    /// Attention output projection `[hidden, hidden]`.
    pub w_o: Vec<f32>,
    /// SwiGLU gate projection `[hidden, d_ff]`.
    pub w_gate: Vec<f32>,
    /// SwiGLU up projection `[hidden, d_ff]`.
    pub w_up: Vec<f32>,
    /// SwiGLU down projection `[d_ff, hidden]`.
    pub w_down: Vec<f32>,
    /// RMSNorm scale (conditioning). `[hidden]`
    pub norm_cond: Vec<f32>,
    /// RMSNorm scale (post-attention). `[hidden]`
    pub norm_attn: Vec<f32>,
    /// RMSNorm scale (post-MLP). `[hidden]`
    pub norm_mlp: Vec<f32>,
    /// Learned position embeddings for drafter lookaheads. `[max_depth, hidden]`
    /// Position 0 (verifier) gets no position embedding.
    pub pos_emb: Vec<f32>,
    /// Config snapshot.
    pub config: WeaverConfig,
}

impl WeaverWeights {
    /// Create zero-initialized weights for the given config.
    ///
    /// Zero weights produce zero residuals — the corrector is a safe no-op
    /// before a trained checkpoint is loaded.
    pub fn zeros(config: WeaverConfig) -> Self {
        let h = config.hidden_dim;
        let ff = config.d_ff;
        let md = config.max_depth;
        Self {
            w_c: vec![0.0; h * h],
            w_q: vec![0.0; h * h],
            w_k: vec![0.0; h * h],
            w_v: vec![0.0; h * h],
            w_o: vec![0.0; h * h],
            w_gate: vec![0.0; h * ff],
            w_up: vec![0.0; h * ff],
            w_down: vec![0.0; ff * h],
            norm_cond: vec![0.0; h],
            norm_attn: vec![0.0; h],
            norm_mlp: vec![0.0; h],
            pos_emb: vec![0.0; md * h],
            config,
        }
    }

    /// Deserialize weights from safetensors bytes.
    ///
    /// The format mirrors riir-train's `weights_to_safetensors_bytes`:
    /// 12 tensor keys (`w_c`, `w_q`, …, `pos_emb`) stored as flat 1-D F32,
    /// plus metadata (`hidden_dim`, `n_heads`, `k_candidates`, `d_ff`,
    /// `max_depth`). `rms_eps` and `n_layer` are not stored — hardcoded
    /// to `1e-6` and `1` respectively.
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, WeaverLoadError> {
        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(WeaverLoadError::SafetensorsParse)?;

        // Read config from the safetensors JSON header metadata. The
        // safetensors 0.4 crate doesn't expose a public metadata() accessor,
        // so we parse the raw header bytes directly.
        let header = parse_safetensors_header(bytes)?;
        let hidden_dim = extract_meta(&header, "hidden_dim")?;
        let n_heads = extract_meta(&header, "n_heads")?;
        let k_candidates = extract_meta(&header, "k_candidates")?;
        let d_ff = extract_meta(&header, "d_ff")?;
        let max_depth = extract_meta(&header, "max_depth")?;
        let config = WeaverConfig {
            hidden_dim,
            n_heads,
            k_candidates,
            n_layer: 1,
            d_ff,
            rms_eps: 1e-6,
            max_depth,
        };

        // Read each tensor via TensorView::data().
        let read = |name: &str, expected: usize| -> Result<Vec<f32>, WeaverLoadError> {
            let tv = st
                .tensor(name)
                .map_err(|e| WeaverLoadError::TensorMissing {
                    name: name.to_string(),
                    source: e,
                })?;
            let raw = tv.data();
            let n = raw.len() / 4;
            if n != expected {
                return Err(WeaverLoadError::ShapeMismatch {
                    tensor: name.to_string(),
                    expected,
                    actual: n,
                });
            }
            // Safe little-endian f32 decode — no alignment assumption.
            Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };

        let h = hidden_dim;
        let ff = d_ff;
        let md = max_depth;
        Ok(Self {
            w_c: read("w_c", h * h)?,
            w_q: read("w_q", h * h)?,
            w_k: read("w_k", h * h)?,
            w_v: read("w_v", h * h)?,
            w_o: read("w_o", h * h)?,
            w_gate: read("w_gate", h * ff)?,
            w_up: read("w_up", h * ff)?,
            w_down: read("w_down", ff * h)?,
            norm_cond: read("norm_cond", h)?,
            norm_attn: read("norm_attn", h)?,
            norm_mlp: read("norm_mlp", h)?,
            pos_emb: read("pos_emb", md * h)?,
            config,
        })
    }
}

// ── Input / Output ───────────────────────────────────────────────────────

/// Borrowed input for the Weaver forward pass.
pub struct WeaverInput<'a> {
    /// Verifier hidden state at the prefix position. `[hidden_dim]`
    pub h_verifier: &'a [f32],
    /// Drafter lookahead hidden states. `D` slices, each `[hidden_dim]`.
    pub h_dflash: &'a [&'a [f32]],
    /// Top-K candidate token ids per draft depth. `D` slices, each `[K]`.
    pub topk_ids: &'a [&'a [u32]],
    /// DFlash draft logits over the K candidates per depth. `D` slices, each `[K]`.
    pub dflash_logits: &'a [&'a [f32]],
    /// Shared vocab embedding (verifier/drafter), row-major. `[V * hidden_dim]`
    pub embedding: &'a [f32],
    /// Vocabulary size V (for bounds checking on gathered ids).
    pub vocab_size: usize,
}

/// Output of the Weaver forward pass.
#[derive(Debug, Clone)]
pub struct WeaverOutput {
    /// Weaver residual logits per depth. `[D][K]`
    pub weaver_residual: Vec<Vec<f32>>,
    /// Corrected logits (dflash + weaver). `[D][K]`
    pub corrected_logits: Vec<Vec<f32>>,
    /// Corrected probabilities (softmax of corrected_logits over K). `[D][K]`
    pub corrected_probs: Vec<Vec<f32>>,
    /// Drafter depth D.
    pub depth: usize,
    /// Candidate count K.
    pub k: usize,
}

// ── Scratch buffers (zero-alloc hot path, Issue 131 G4) ──────────────────

/// Pre-allocated scratch buffers for the Weaver forward pass.
///
/// Eliminates ~20 `Vec` allocations per forward pass. Allocate once with
/// [`WeaverScratch::new`], then reuse across calls via
/// [`WeaverCorrector::correct_with_scratch`] or
/// [`WeaverCorrector::correct_marginals_with_scratch`].
///
/// The buffers are sized for the worst case (`max_depth` positions) and
/// reused across all depths — no per-call or per-depth allocation.
///
/// **Issue 131 G4 (latency):** the allocating `weaver_forward` allocates
/// and zero-fills ~20 buffers per call. This scratch struct hoists those
/// allocations to a one-time cost. Combined with the batched matmul (which
/// reads each weight matrix once instead of `seq_len` times), this is the
/// CPU-side G4 optimization path (Issue 131 G4 option 2: SIMD/BLAS-level
/// optimization via reduced memory traffic, not new SIMD intrinsics).
pub struct WeaverScratch {
    // ── Forward-pass buffers, sized for seq_len = max_depth + 1 ──
    /// Conditioning sequence `[seq_len * h]`.
    u_cond: Vec<f32>,
    /// Query projection `[seq_len * h]`.
    q: Vec<f32>,
    /// Key projection `[seq_len * h]`.
    kk: Vec<f32>,
    /// Value projection `[seq_len * h]`.
    v: Vec<f32>,
    /// Attention output `[seq_len * h]`.
    attn_out: Vec<f32>,
    /// Post-attention normed `[seq_len * h]`.
    u_attn_normed: Vec<f32>,
    /// Final (post-MLP normed) `[seq_len * h]`.
    u_final: Vec<f32>,

    // ── Per-position scratch (size h or d_ff) ──
    normed_buf: Vec<f32>, // [h]
    post_buf: Vec<f32>,   // [h]
    gate: Vec<f32>,       // [d_ff] — also reused as activation buffer
    up: Vec<f32>,         // [d_ff]
    down: Vec<f32>,       // [h]

    // ── Attention scratch ──
    scores: Vec<f32>, // [seq_len]

    // ── Top-K output (flat, not Vec<Vec<f32>>) ──
    /// Weaver residual logits `[max_depth * K]`.
    residual_flat: Vec<f32>,
    /// Corrected logits `[max_depth * K]`.
    corrected_logits_flat: Vec<f32>,
    /// Corrected probabilities `[max_depth * K]`.
    corrected_probs_flat: Vec<f32>,

    // ── correct_marginals_with_scratch scratch ──
    // Bounded at k+1 entries via partial insertion sort (never grows to
    // vocab_size — see correct_marginals_with_scratch).
    top_pairs: Vec<(usize, f32)>, // [≤ k+1]
}

impl WeaverScratch {
    /// Allocate scratch buffers for the given config. The buffers are sized
    /// for the worst case (`max_depth` positions) and are zero-initialized.
    pub fn new(config: &WeaverConfig) -> Self {
        let h = config.hidden_dim;
        let ff = config.d_ff;
        let k = config.k_candidates;
        let max_seq = config.max_depth + 1; // verifier + drafter lookaheads
        let max_dk = config.max_depth * k;

        Self {
            u_cond: vec![0.0; max_seq * h],
            q: vec![0.0; max_seq * h],
            kk: vec![0.0; max_seq * h],
            v: vec![0.0; max_seq * h],
            attn_out: vec![0.0; max_seq * h],
            u_attn_normed: vec![0.0; max_seq * h],
            u_final: vec![0.0; max_seq * h],
            normed_buf: vec![0.0; h],
            post_buf: vec![0.0; h],
            gate: vec![0.0; max_seq * ff],
            up: vec![0.0; max_seq * ff],
            down: vec![0.0; h],
            scores: vec![0.0; max_seq],
            residual_flat: vec![0.0; max_dk],
            corrected_logits_flat: vec![0.0; max_dk],
            corrected_probs_flat: vec![0.0; max_dk],
            top_pairs: Vec::new(), // grown on demand in correct_marginals
        }
    }

    /// Weaver residual logits after [`WeaverCorrector::correct_with_scratch`].
    /// Flat `[depth * k]`, row-major. Valid extent: `depth * k` where `depth`
    /// and `k` are the values returned by the forward call.
    pub fn residual_flat(&self) -> &[f32] {
        &self.residual_flat
    }

    /// Corrected logits after [`WeaverCorrector::correct_with_scratch`].
    /// Flat `[depth * k]`, row-major.
    pub fn corrected_logits_flat(&self) -> &[f32] {
        &self.corrected_logits_flat
    }

    /// Corrected probabilities after [`WeaverCorrector::correct_with_scratch`].
    /// Flat `[depth * k]`, row-major. Sums to 1.0 per depth (G1 invariant).
    pub fn corrected_probs_flat(&self) -> &[f32] {
        &self.corrected_probs_flat
    }
}

// ── High-level corrector ─────────────────────────────────────────────────

/// Convenience wrapper holding loaded weights.
///
/// Construct via [`WeaverCorrector::from_checkpoint`] (reads a safetensors file
/// from disk, optionally verifying a `.blake3` sidecar) or
/// [`WeaverCorrector::from_weights`] (from a pre-loaded `WeaverWeights`).
pub struct WeaverCorrector {
    weights: WeaverWeights,
}

impl WeaverCorrector {
    /// Wrap pre-loaded weights.
    pub fn from_weights(weights: WeaverWeights) -> Self {
        Self { weights }
    }

    /// Deserialize from safetensors bytes (e.g. from `include_bytes!` or mmap).
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, WeaverLoadError> {
        Ok(Self::from_weights(WeaverWeights::from_safetensors_bytes(
            bytes,
        )?))
    }

    /// Load from a file path. If a `<path>.blake3` sidecar exists, the file's
    /// BLAKE3 hash is verified before deserialization.
    pub fn from_checkpoint(path: impl AsRef<std::path::Path>) -> Result<Self, WeaverLoadError> {
        use std::fs;
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(WeaverLoadError::Io)?;

        // Optional BLAKE3 sidecar verification.
        let sidecar = path.with_extension("safetensors.blake3");
        if sidecar.exists() {
            let expected_hex = fs::read_to_string(&sidecar)
                .map_err(WeaverLoadError::Io)?
                .trim()
                .to_string();
            let actual = blake3::hash(&bytes).to_hex().to_string();
            if actual != expected_hex {
                return Err(WeaverLoadError::Blake3Mismatch {
                    expected: expected_hex,
                    actual,
                });
            }
        }

        Self::from_safetensors_bytes(&bytes)
    }

    /// Run the Weaver forward pass and produce corrected probabilities.
    pub fn correct(&self, input: &WeaverInput) -> WeaverOutput {
        weaver_forward(&self.weights, input)
    }

    /// Zero-alloc forward pass (Issue 131 G4). Writes results into `scratch`
    /// instead of allocating a `WeaverOutput`. Returns `(depth, k)`.
    ///
    /// After the call, read `scratch.residual_flat()`,
    /// `scratch.corrected_logits_flat()`, `scratch.corrected_probs_flat()`
    /// — each is `[depth * k]` flat row-major.
    ///
    /// Use this in hot paths (e.g. the speculative decode loop). Allocate the
    /// scratch once per corrector and reuse it across calls.
    pub fn correct_with_scratch(
        &self,
        input: &WeaverInput,
        scratch: &mut WeaverScratch,
    ) -> (usize, usize) {
        weaver_forward_into(&self.weights, input, scratch)
    }

    /// Parallel forward pass (Issue 131 G4). Same I/O as
    /// [`correct_with_scratch`] but uses rayon to parallelize the heavy
    /// matmuls across positions. ~3.2× faster on M3 Max (12 P-cores).
    ///
    /// Use this instead of `correct_with_scratch` when `depth ≥ 1` and the
    /// config is large enough that per-position matmul work exceeds rayon's
    /// thread-pool overhead (~5µs). At hidden=2304 this is always true.
    pub fn correct_parallel(
        &self,
        input: &WeaverInput,
        scratch: &mut WeaverScratch,
    ) -> (usize, usize) {
        weaver_forward_parallel(&self.weights, input, scratch)
    }

    /// Borrow the underlying weights.
    pub fn weights(&self) -> &WeaverWeights {
        &self.weights
    }

    /// Apply the Weaver correction to full-vocabulary DFlash marginals (T3).
    ///
    /// This is the **marginal corrector** integration point (Issue 131 option 1).
    /// After `dflash_predict_with` produces full-vocab marginals, the caller
    /// invokes this method to apply the Weaver residual correction over the
    /// top-K candidates at each draft depth.
    ///
    /// ## What it does
    ///
    /// For each draft depth `d ∈ [0, depth)`:
    /// 1. Select the top-K=32 token ids from `marginals[d*vocab..(d+1)*vocab]`
    ///    (by probability mass).
    /// 2. Build a `WeaverInput` from the top-K ids + drafter logits + hidden
    ///    states + shared embedding.
    /// 3. Run `weaver_forward` → corrected top-K probabilities.
    /// 4. Write the corrected probabilities back to `marginals`, zeroing all
    ///    non-top-K positions and renormalizing.
    ///
    /// ## No-harm contract
    ///
    /// When the corrector holds zero-init weights, the Weaver residual is zero,
    /// so the corrected top-K probabilities equal the drafter's top-K
    /// probabilities. After zeroing non-top-K and renormalizing over K, the
    /// result is a **truncated** version of the original marginal — the top-K
    /// mass is preserved but redistributed. This is a minor change (the top-K
    /// typically captures >99% of the probability mass), and the tree builder
    /// only consumes the top-K anyway.
    ///
    /// ## Arguments
    ///
    /// - `marginals` — full-vocab marginals, shape `[depth * vocab_size]`,
    ///   row-major (depth-outer). Modified in-place.
    /// - `h_verifier` — verifier hidden state at the prefix position, `[hidden_dim]`.
    /// - `h_dflash` — drafter hidden states per depth, `[depth][hidden_dim]`.
    /// - `embedding` — shared token embedding, `[vocab_size * hidden_dim]`.
    /// - `vocab_size` — vocabulary size.
    ///
    /// ## Returns
    ///
    /// `Ok(())` on success, or an error if the shapes are inconsistent.
    pub fn correct_marginals_inplace(
        &self,
        marginals: &mut [f32],
        h_verifier: &[f32],
        h_dflash: &[&[f32]],
        embedding: &[f32],
        vocab_size: usize,
    ) -> Result<(), WeaverCorrectError> {
        let cfg = &self.weights.config;
        let k = cfg.k_candidates;
        let h = cfg.hidden_dim;
        let depth = h_dflash.len();

        if marginals.len() != depth * vocab_size {
            return Err(WeaverCorrectError::MarginalsShape {
                expected: depth * vocab_size,
                actual: marginals.len(),
            });
        }
        if h_verifier.len() != h {
            return Err(WeaverCorrectError::HiddenShape {
                context: "h_verifier",
                expected: h,
                actual: h_verifier.len(),
            });
        }
        if embedding.len() < vocab_size * h {
            return Err(WeaverCorrectError::EmbeddingShape {
                expected: vocab_size * h,
                actual: embedding.len(),
            });
        }
        for (di, h_d) in h_dflash.iter().enumerate() {
            if h_d.len() != h {
                return Err(WeaverCorrectError::HiddenShape {
                    context: "h_dflash[di]",
                    expected: h,
                    actual: h_d.len(),
                });
            }
            let _ = di; // index not needed for the check
        }
        if depth > cfg.max_depth {
            return Err(WeaverCorrectError::DepthExceedsConfig {
                depth,
                max_depth: cfg.max_depth,
            });
        }
        if k > vocab_size {
            // Can't select K candidates from fewer than K tokens. Correct
            // nothing — leave marginals unchanged (safe no-op).
            return Ok(());
        }

        // ── Per-depth: select top-K, correct, write back ──
        // Reuse scratch buffers across depths to avoid per-depth allocation.
        // We process one depth at a time (depth is typically 4-8), building a
        // fresh single-depth WeaverInput per iteration.
        let mut topk_ids: Vec<u32> = vec![0; k];
        let mut topk_logits: Vec<f32> = vec![0.0; k];
        // Hoisted out of the depth loop — reused via clear() to avoid one
        // Vec allocation per depth.
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(k + 1);

        for di in 0..depth {
            let marg_row = &marginals[di * vocab_size..(di + 1) * vocab_size];

            // Select top-K token ids by probability mass.
            // Use partial selection sort (same as precompute_weaver_real_data).
            top.clear();
            for (vid, &p) in marg_row.iter().enumerate() {
                if !p.is_finite() {
                    continue;
                }
                // O(1) fast reject before the binary search. `top` is sorted
                // descending, so once it holds K entries any `p` STRICTLY below
                // the K-th best has `v > p` true for all K entries, which makes
                // `partition_point` return exactly `k` and the `pos < k` branch
                // below a no-op. `p == worst` deliberately falls through to the
                // original path (there `partition_point` returns `k-1` and the
                // entry does get inserted), so the selected set is unchanged.
                if top.len() == k
                    && let Some(&(_, worst)) = top.last()
                    && p < worst
                {
                    continue;
                }
                let pos = top.partition_point(|&(_, v)| v > p);
                if pos < k {
                    top.insert(pos, (vid, p));
                    if top.len() > k {
                        top.pop();
                    }
                }
            }
            // Fill topk_ids + topk_logits (log-space for the Weaver residual).
            for (ki, &(vid, p)) in top.iter().enumerate().take(k) {
                topk_ids[ki] = vid as u32;
                // Recover approximate logits from probs (Weaver adds residual
                // to logits, then softmaxes). Use log(p) clamped to avoid -inf.
                topk_logits[ki] = if p > 1e-30 { p.ln() } else { -69.07 }; // ln(1e-30)
            }
            // If fewer than K valid tokens, pad with zeros (id=0, logit=-69).
            for ki in top.len()..k {
                topk_ids[ki] = 0;
                topk_logits[ki] = -69.07;
            }

            // Build WeaverInput for this single depth.
            let h_dflash_slice: &[&[f32]] = &h_dflash[di..=di];
            let topk_ids_slice: &[&[u32]] = &[&topk_ids[..]];
            let topk_logits_slice: &[&[f32]] = &[&topk_logits[..]];

            let input = WeaverInput {
                h_verifier,
                h_dflash: h_dflash_slice,
                topk_ids: topk_ids_slice,
                dflash_logits: topk_logits_slice,
                embedding,
                vocab_size,
            };

            let out = weaver_forward(&self.weights, &input);

            // Write back: zero the full-vocab row, then write corrected top-K.
            let out_row = &out.corrected_probs[0]; // depth 0 of the single-depth input
            let marg_out = &mut marginals[di * vocab_size..(di + 1) * vocab_size];
            for v in marg_out.iter_mut() {
                *v = 0.0;
            }
            for (ki, &(vid, _)) in top.iter().enumerate().take(k) {
                marg_out[vid] = out_row[ki];
            }
            // marg_out already sums to ~1.0 (Weaver softmaxes over K), but
            // renormalize for safety (floating-point drift).
            let sum: f32 = marg_out.iter().sum();
            if sum > 1e-30 {
                let inv = 1.0 / sum;
                for v in marg_out.iter_mut() {
                    *v *= inv;
                }
            }
        }

        Ok(())
    }

    /// Zero-alloc variant of [`correct_marginals_inplace`] (Issue 131 G4).
    ///
    /// Same semantics: select top-K per depth, run the Weaver forward pass,
    /// write corrected probabilities back to `marginals` in-place. The
    /// difference is that the heavy forward-pass buffers live in `scratch`
    /// and are reused across calls — no per-call allocation of the ~20
    /// forward-pass buffers (the dominant cost in `weaver_forward`).
    ///
    /// Two small per-call allocations remain (`topk_ids`, `topk_logits` — K
    /// elements each); they cannot live in `scratch` because the
    /// `WeaverInput` borrows them immutably while `weaver_forward_into`
    /// borrows `scratch` mutably. The `top_pairs` sort buffer IS reused
    /// via `scratch`.
    ///
    /// Use this in the speculative decode hot path. Allocate `scratch` once
    /// via [`WeaverScratch::new`] and pass it to every call.
    pub fn correct_marginals_with_scratch(
        &self,
        marginals: &mut [f32],
        h_verifier: &[f32],
        h_dflash: &[&[f32]],
        embedding: &[f32],
        vocab_size: usize,
        scratch: &mut WeaverScratch,
    ) -> Result<(), WeaverCorrectError> {
        let cfg = &self.weights.config;
        let k = cfg.k_candidates;
        let h = cfg.hidden_dim;
        let depth = h_dflash.len();

        if marginals.len() != depth * vocab_size {
            return Err(WeaverCorrectError::MarginalsShape {
                expected: depth * vocab_size,
                actual: marginals.len(),
            });
        }
        if h_verifier.len() != h {
            return Err(WeaverCorrectError::HiddenShape {
                context: "h_verifier",
                expected: h,
                actual: h_verifier.len(),
            });
        }
        if embedding.len() < vocab_size * h {
            return Err(WeaverCorrectError::EmbeddingShape {
                expected: vocab_size * h,
                actual: embedding.len(),
            });
        }
        for h_d in h_dflash {
            if h_d.len() != h {
                return Err(WeaverCorrectError::HiddenShape {
                    context: "h_dflash[di]",
                    expected: h,
                    actual: h_d.len(),
                });
            }
        }
        if depth > cfg.max_depth {
            return Err(WeaverCorrectError::DepthExceedsConfig {
                depth,
                max_depth: cfg.max_depth,
            });
        }
        if k > vocab_size {
            return Ok(());
        }

        // Top-K buffers live outside scratch to avoid the borrow conflict
        // (WeaverInput borrows them immutably while forward_into borrows
        // scratch mutably). Allocated once here, reused across depths.
        let mut topk_ids: Vec<u32> = vec![0; k];
        let mut topk_logits: Vec<f32> = vec![0.0; k];

        // ── Per-depth: select top-K, correct, write back ──
        for di in 0..depth {
            let marg_row = &marginals[di * vocab_size..(di + 1) * vocab_size];

            // Select top-K token ids by probability mass using partial
            // insertion sort — O(vocab · log k) expected (most entries skip
            // the insert because partition_point returns k). Matches the
            // allocating sibling's algorithm; bit-identical output.
            // scratch.top_pairs stays bounded at k+1 entries (no vocab-sized
            // allocation, no full sort).
            scratch.top_pairs.clear();
            for (vid, &p) in marg_row.iter().enumerate() {
                if !p.is_finite() {
                    continue;
                }
                // O(1) fast reject — see `correct_marginals_inplace` for why
                // `p < worst` (strict) leaves the selected set unchanged.
                if scratch.top_pairs.len() == k
                    && let Some(&(_, worst)) = scratch.top_pairs.last()
                    && p < worst
                {
                    continue;
                }
                let pos = scratch.top_pairs.partition_point(|&(_, v)| v > p);
                if pos < k {
                    scratch.top_pairs.insert(pos, (vid, p));
                    if scratch.top_pairs.len() > k {
                        scratch.top_pairs.pop();
                    }
                }
            }

            // Fill topk_ids + topk_logits.
            let n_take = scratch.top_pairs.len().min(k);
            for ki in 0..n_take {
                let (vid, p) = scratch.top_pairs[ki];
                topk_ids[ki] = vid as u32;
                topk_logits[ki] = if p > 1e-30 { p.ln() } else { -69.07 };
            }
            for ki in n_take..k {
                topk_ids[ki] = 0;
                topk_logits[ki] = -69.07;
            }

            // Build WeaverInput for this single depth and run forward_into.
            let h_dflash_slice: &[&[f32]] = &h_dflash[di..=di];
            let topk_ids_slice: &[&[u32]] = &[&topk_ids[..]];
            let topk_logits_slice: &[&[f32]] = &[&topk_logits[..]];
            let input = WeaverInput {
                h_verifier,
                h_dflash: h_dflash_slice,
                topk_ids: topk_ids_slice,
                dflash_logits: topk_logits_slice,
                embedding,
                vocab_size,
            };
            weaver_forward_into(&self.weights, &input, scratch);

            // Write back: zero the full-vocab row, then write corrected top-K.
            // Read from scratch.corrected_probs_flat (depth 0 of single-depth input).
            let marg_out = &mut marginals[di * vocab_size..(di + 1) * vocab_size];
            for v in marg_out.iter_mut() {
                *v = 0.0;
            }
            for ki in 0..n_take {
                let vid = scratch.top_pairs[ki].0;
                marg_out[vid] = scratch.corrected_probs_flat[ki];
            }
            // Renormalize for safety (floating-point drift).
            let sum: f32 = marg_out.iter().sum();
            if sum > 1e-30 {
                let inv = 1.0 / sum;
                for v in marg_out.iter_mut() {
                    *v *= inv;
                }
            }
        }

        Ok(())
    }
}

// ── f16 Corrector (Issue 136) ───────────────────────────────────────────

/// f16-weight Weaver corrector. Mirrors [`WeaverCorrector`] but stores weight
/// matrices as `half::f16`, halving memory traffic on the hot path.
///
/// Convert from a loaded [`WeaverCorrector`] via [`WeaverCorrectorF16::from_f32`].
/// The conversion is a one-time cost (f32→f16 rounding). The forward pass then
/// uses `simd_fused_scale_acc_f16` which converts f16→f32 during the FMA loop
/// — halving weight-read bandwidth while maintaining f32 accumulation precision.
///
/// Only the parallel forward path is provided (the hot path). For testing /
/// validation, use the f32 [`WeaverCorrector`].
pub struct WeaverCorrectorF16 {
    weights: WeaverWeightsF16,
}

impl WeaverCorrectorF16 {
    /// Convert a loaded f32 corrector to f16.
    pub fn from_f32(src: &WeaverCorrector) -> Self {
        Self {
            weights: WeaverWeightsF16::from_f32(&src.weights),
        }
    }

    /// Parallel forward pass with f16 weights (Issue 136 G4 optimization).
    ///
    /// Same I/O contract as [`WeaverCorrector::correct_parallel`] — writes
    /// results into `scratch`. The only difference is the weight precision:
    /// f16 weights are converted to f32 inside the SIMD AXPY loop.
    pub fn correct_parallel(
        &self,
        input: &WeaverInput,
        scratch: &mut WeaverScratch,
    ) -> (usize, usize) {
        weaver_forward_parallel_f16(&self.weights, input, scratch)
    }

    /// Borrow the underlying f16 weights.
    pub fn weights(&self) -> &WeaverWeightsF16 {
        &self.weights
    }
}

/// Errors from `correct_marginals_inplace`.
#[derive(Debug)]
pub enum WeaverCorrectError {
    /// `marginals.len()` != `depth * vocab_size`.
    MarginalsShape { expected: usize, actual: usize },
    /// A hidden-state slice had the wrong length.
    HiddenShape {
        context: &'static str,
        expected: usize,
        actual: usize,
    },
    /// `embedding.len()` < `vocab_size * hidden_dim`.
    EmbeddingShape { expected: usize, actual: usize },
    /// Draft depth exceeds the Weaver config's `max_depth`.
    DepthExceedsConfig { depth: usize, max_depth: usize },
}

impl std::fmt::Display for WeaverCorrectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MarginalsShape { expected, actual } => write!(
                f,
                "marginals shape mismatch: expected {expected} (= depth*vocab), got {actual}"
            ),
            Self::HiddenShape {
                context,
                expected,
                actual,
            } => write!(
                f,
                "{context} shape mismatch: expected {expected}, got {actual}"
            ),
            Self::EmbeddingShape { expected, actual } => write!(
                f,
                "embedding shape mismatch: expected >={expected}, got {actual}"
            ),
            Self::DepthExceedsConfig { depth, max_depth } => write!(
                f,
                "draft depth {depth} exceeds Weaver max_depth {max_depth}"
            ),
        }
    }
}

impl std::error::Error for WeaverCorrectError {}

// ── Forward pass ─────────────────────────────────────────────────────────

/// The 7-step Weaver forward pass.
///
/// See the module-level doc for the architecture diagram. Returns corrected
/// logits + probabilities over the K candidates per draft depth.
pub fn weaver_forward(weights: &WeaverWeights, input: &WeaverInput) -> WeaverOutput {
    let cfg = &weights.config;
    let h = cfg.hidden_dim;
    let k = cfg.k_candidates;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let d_ff = cfg.d_ff;
    let eps = cfg.rms_eps;
    let d_depth = input.h_dflash.len();
    let seq_len = d_depth + 1; // verifier + drafter lookaheads

    debug_assert_eq!(input.h_verifier.len(), h);
    debug_assert_eq!(input.topk_ids.len(), d_depth);
    debug_assert_eq!(input.dflash_logits.len(), d_depth);
    for di in 0..d_depth {
        debug_assert_eq!(input.h_dflash[di].len(), h);
        debug_assert_eq!(input.topk_ids[di].len(), k);
        debug_assert_eq!(input.dflash_logits[di].len(), k);
    }

    // ── Step 1: Conditioning sequence u[0..seq_len] ──
    // Pre-allocate scratch buffers reused across all positions — avoids
    // per-position Vec allocations (5 * seq_len allocs eliminated).
    let mut u_cond = vec![0.0f32; seq_len * h];
    let mut normed_buf = vec![0.0f32; h];
    let mut post_buf = vec![0.0f32; h];
    for pos in 0..seq_len {
        let raw = if pos == 0 {
            input.h_verifier
        } else {
            input.h_dflash[pos - 1]
        };
        rmsnorm_into(raw, &weights.norm_cond, eps, &mut normed_buf);
        let u_row = &mut u_cond[pos * h..(pos + 1) * h];
        matmul_vec(&normed_buf, &weights.w_c, h, h, u_row);
        if pos > 0 {
            let pe = &weights.pos_emb[(pos - 1) * h..pos * h];
            for j in 0..h {
                u_row[j] += pe[j];
            }
        }
    }

    // ── Step 2: QKV projections ──
    let mut q = vec![0.0f32; seq_len * h];
    let mut kk = vec![0.0f32; seq_len * h];
    let mut v = vec![0.0f32; seq_len * h];
    for pos in 0..seq_len {
        let u_row = &u_cond[pos * h..(pos + 1) * h];
        matmul_vec(u_row, &weights.w_q, h, h, &mut q[pos * h..(pos + 1) * h]);
        matmul_vec(u_row, &weights.w_k, h, h, &mut kk[pos * h..(pos + 1) * h]);
        matmul_vec(u_row, &weights.w_v, h, h, &mut v[pos * h..(pos + 1) * h]);
    }

    // ── Step 3: Causal multi-head attention ──
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * h];
    let mut scores = vec![0.0f32; seq_len];
    for head in 0..n_heads {
        let ho = head * head_dim;
        for qi in 0..seq_len {
            let q_row = &q[qi * h + ho..qi * h + ho + head_dim];
            // Causal: attend to kj ∈ [0, qi]
            let mut max_s = f32::NEG_INFINITY;
            for kj in 0..=qi {
                let k_row = &kk[kj * h + ho..kj * h + ho + head_dim];
                let s = dot(q_row, k_row) * attn_scale;
                scores[kj] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            // Fused softmax: SIMD shift + exp-sum in one pass over `scores[..=qi]`.
            use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
            let s_row = &mut scores[..=qi];
            simd_add_scalar_inplace(s_row, -max_s);
            let sum_e = simd_exp_sum_inplace(s_row);
            let inv_sum = 1.0 / sum_e;
            let out_row = &mut attn_out[qi * h + ho..qi * h + ho + head_dim];
            for kj in 0..=qi {
                let w = scores[kj] * inv_sum;
                let v_row = &v[kj * h + ho..kj * h + ho + head_dim];
                katgpt_core::simd::simd_fused_scale_acc(out_row, v_row, w, head_dim);
            }
        }
    }

    // ── Step 4: Output projection + residual + post-attn RMSNorm ──
    let mut u_attn_normed = vec![0.0f32; seq_len * h];
    let mut tmp = vec![0.0f32; h];
    for pos in 0..seq_len {
        let o_row = &attn_out[pos * h..(pos + 1) * h];
        matmul_vec(o_row, &weights.w_o, h, h, &mut tmp);
        let u = &u_cond[pos * h..(pos + 1) * h];
        for j in 0..h {
            post_buf[j] = u[j] + tmp[j];
        }
        rmsnorm_into(
            &post_buf,
            &weights.norm_attn,
            eps,
            &mut u_attn_normed[pos * h..(pos + 1) * h],
        );
    }

    // ── Step 5: SwiGLU MLP + residual + post-MLP RMSNorm ──
    let mut u_final = vec![0.0f32; seq_len * h];
    let mut gate = vec![0.0f32; d_ff];
    let mut up = vec![0.0f32; d_ff];
    let mut act = vec![0.0f32; d_ff];
    let mut down = vec![0.0f32; h];
    for pos in 0..seq_len {
        let u_row = &u_attn_normed[pos * h..(pos + 1) * h];
        matmul_vec(u_row, &weights.w_gate, h, d_ff, &mut gate);
        matmul_vec(u_row, &weights.w_up, h, d_ff, &mut up);
        for j in 0..d_ff {
            act[j] = silu(gate[j]) * up[j];
        }
        matmul_vec(&act, &weights.w_down, d_ff, h, &mut down);
        for j in 0..h {
            post_buf[j] = u_row[j] + down[j];
        }
        rmsnorm_into(
            &post_buf,
            &weights.norm_mlp,
            eps,
            &mut u_final[pos * h..(pos + 1) * h],
        );
    }

    // ── Steps 6 + 7: Top-K residual add + softmax over K ──
    // The gather+dot is fused: each top-K embedding row is read directly from
    // `input.embedding` and dotted with `h_weaver` once. No scratch buffer is
    // needed — `h_weaver` is `h` floats (~9 KB at h=2304) and stays hot in L1
    // across the K reads, while each embedding row is touched exactly once
    // (versus gather-then-dot which would touch each row twice).
    let mut weaver_residual = vec![vec![0.0f32; k]; d_depth];
    let mut corrected_logits = vec![vec![0.0f32; k]; d_depth];
    let mut corrected_probs = vec![vec![0.0f32; k]; d_depth];

    for di in 0..d_depth {
        let pos = di + 1; // skip verifier position 0
        let h_weaver = &u_final[pos * h..(pos + 1) * h];
        let ids = input.topk_ids[di];
        let dfl = input.dflash_logits[di];

        // Compute residual logits directly from the vocab embedding.
        for (ki, &tid) in ids.iter().enumerate() {
            let tid = tid as usize;
            debug_assert!(
                tid < input.vocab_size,
                "topk id {} >= vocab {}",
                tid,
                input.vocab_size
            );
            let row = &input.embedding[tid * h..(tid + 1) * h];
            weaver_residual[di][ki] = dot(h_weaver, row);
        }

        // Corrected = dflash + weaver_residual.
        for ki in 0..k {
            corrected_logits[di][ki] = dfl[ki] + weaver_residual[di][ki];
        }

        // Softmax over K candidates.
        let mut max_c = f32::NEG_INFINITY;
        for cl in corrected_logits[di][..k].iter() {
            if *cl > max_c {
                max_c = *cl;
            }
        }
        // Softmax over K candidates (SIMD exp-sum into corrected_probs).
        use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
        let cl_row = &corrected_logits[di][..k];
        corrected_probs[di][..k].copy_from_slice(cl_row);
        let cp_row = &mut corrected_probs[di][..k];
        simd_add_scalar_inplace(cp_row, -max_c);
        let sum_e = simd_exp_sum_inplace(cp_row);
        let inv_sum = 1.0 / sum_e;
        for cp in corrected_probs[di][..k].iter_mut() {
            *cp *= inv_sum;
        }
    }

    WeaverOutput {
        weaver_residual,
        corrected_logits,
        corrected_probs,
        depth: d_depth,
        k,
    }
}

/// Zero-alloc Weaver forward pass (Issue 131 G4 optimization).
///
/// This is the hot-path variant of [`weaver_forward`]: it writes into
/// pre-allocated [`WeaverScratch`] buffers instead of allocating ~20 `Vec`s
/// per call, and it uses batched matmuls ([`matmul_vec_batched`]) that read
/// each weight matrix **once** instead of `seq_len` times.
///
/// The results live in `scratch` after the call:
/// - `scratch.residual_flat[di*K + ki]` — Weaver residual logits
/// - `scratch.corrected_logits_flat[di*K + ki]` — dflash + weaver
/// - `scratch.corrected_probs_flat[di*K + ki]` — softmax over K
///
/// Returns `(depth, k)` so the caller knows the valid extent of the flat
/// output buffers (always equal to the config's `max_depth` and
/// `k_candidates`).
///
/// # Correctness
///
/// Bit-identical to `weaver_forward` modulo floating-point reassociation
/// (the batched matmul accumulates in the same order as the per-position
/// matmul — both iterate `i` outer, SIMD-AXPY inner). Verified by
/// `g4_scratch_matches_allocating`.
pub fn weaver_forward_into(
    weights: &WeaverWeights,
    input: &WeaverInput,
    scratch: &mut WeaverScratch,
) -> (usize, usize) {
    let cfg = &weights.config;
    let h = cfg.hidden_dim;
    let k = cfg.k_candidates;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let d_ff = cfg.d_ff;
    let eps = cfg.rms_eps;
    let d_depth = input.h_dflash.len();
    let seq_len = d_depth + 1;

    debug_assert_eq!(input.h_verifier.len(), h);
    debug_assert_eq!(input.topk_ids.len(), d_depth);
    debug_assert_eq!(input.dflash_logits.len(), d_depth);
    for di in 0..d_depth {
        debug_assert_eq!(input.h_dflash[di].len(), h);
        debug_assert_eq!(input.topk_ids[di].len(), k);
        debug_assert_eq!(input.dflash_logits[di].len(), k);
    }
    debug_assert!(
        seq_len <= cfg.max_depth + 1,
        "seq_len exceeds scratch capacity"
    );

    // Borrow disjoint scratch slices once.
    let WeaverScratch {
        u_cond,
        q,
        kk,
        v,
        attn_out,
        u_attn_normed,
        u_final,
        normed_buf,
        post_buf,
        gate,
        up,
        down,
        scores,
        residual_flat,
        corrected_logits_flat,
        corrected_probs_flat,
        top_pairs: _,
    } = scratch;

    // ── Step 1: Conditioning sequence u[0..seq_len] ──
    // RMSNorm + W_c per position, + pos_emb for drafter positions.
    // (Not batched: pos 0 uses h_verifier, others use h_dflash[pos-1],
    //  and pos_emb is only added to drafter positions. The W_c matmul could
    //  be batched if we first build a [seq_len, h] input buffer, but the
    //  per-position variant keeps the code simple and W_c is only h×h.)
    for pos in 0..seq_len {
        let raw = if pos == 0 {
            input.h_verifier
        } else {
            input.h_dflash[pos - 1]
        };
        rmsnorm_into(raw, &weights.norm_cond, eps, normed_buf);
        let u_row = &mut u_cond[pos * h..(pos + 1) * h];
        matmul_vec(normed_buf, &weights.w_c, h, h, u_row);
        if pos > 0 {
            let pe = &weights.pos_emb[(pos - 1) * h..pos * h];
            for j in 0..h {
                u_row[j] += pe[j];
            }
        }
    }

    // ── Step 2: QKV projections (BATCHED — reads w_q, w_k, w_v once each) ──
    matmul_vec_batched(u_cond, &weights.w_q, h, h, seq_len, q);
    matmul_vec_batched(u_cond, &weights.w_k, h, h, seq_len, kk);
    matmul_vec_batched(u_cond, &weights.w_v, h, h, seq_len, v);

    // ── Step 3: Causal multi-head attention ──
    // (Not batched: causal masking means each position attends to a
    //  different key range.)
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    attn_out[..seq_len * h].fill(0.0);
    for head in 0..n_heads {
        let ho = head * head_dim;
        for qi in 0..seq_len {
            let q_row = &q[qi * h + ho..qi * h + ho + head_dim];
            let mut max_s = f32::NEG_INFINITY;
            for kj in 0..=qi {
                let k_row = &kk[kj * h + ho..kj * h + ho + head_dim];
                let s = dot(q_row, k_row) * attn_scale;
                scores[kj] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            // Fused softmax: SIMD shift + exp-sum in one pass over `scores[..=qi]`.
            use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
            let s_row = &mut scores[..=qi];
            simd_add_scalar_inplace(s_row, -max_s);
            let sum_e = simd_exp_sum_inplace(s_row);
            let inv_sum = 1.0 / sum_e;
            let out_row = &mut attn_out[qi * h + ho..qi * h + ho + head_dim];
            // attn_out was zeroed above; now AXPY in the weighted values.
            for kj in 0..=qi {
                let w = scores[kj] * inv_sum;
                let v_row = &v[kj * h + ho..kj * h + ho + head_dim];
                katgpt_core::simd::simd_fused_scale_acc(out_row, v_row, w, head_dim);
            }
        }
    }

    // ── Step 4: Output projection (BATCHED) + residual + post-attn RMSNorm ──
    matmul_vec_batched(attn_out, &weights.w_o, h, h, seq_len, u_attn_normed);
    // Add residual (u_cond) then RMSNorm into u_attn_normed (in place).
    for pos in 0..seq_len {
        let ua = &mut u_attn_normed[pos * h..(pos + 1) * h];
        let uc = &u_cond[pos * h..(pos + 1) * h];
        for j in 0..h {
            post_buf[j] = uc[j] + ua[j];
        }
        rmsnorm_into(post_buf, &weights.norm_attn, eps, ua);
    }

    // ── Step 5: SwiGLU MLP (partially batched) + residual + post-MLP RMSNorm ──
    // w_gate and w_up are batched (each reads h→d_ff once for all positions).
    // The SiLU elementwise + w_down matmul is per-position (d_ff→h).
    matmul_vec_batched(u_attn_normed, &weights.w_gate, h, d_ff, seq_len, gate);
    matmul_vec_batched(u_attn_normed, &weights.w_up, h, d_ff, seq_len, up);
    for pos in 0..seq_len {
        let u_row = &u_attn_normed[pos * h..(pos + 1) * h];
        let g_row = &mut gate[pos * d_ff..(pos + 1) * d_ff];
        let u_row_ff = &up[pos * d_ff..(pos + 1) * d_ff];
        // act = silu(gate) * up (in-place into g_row to reuse the buffer).
        for j in 0..d_ff {
            g_row[j] = silu(g_row[j]) * u_row_ff[j];
        }
        // down = w_down · act.
        matmul_vec(g_row, &weights.w_down, d_ff, h, down);
        // Residual + RMSNorm into u_final.
        let uf = &mut u_final[pos * h..(pos + 1) * h];
        for j in 0..h {
            post_buf[j] = u_row[j] + down[j];
        }
        rmsnorm_into(post_buf, &weights.norm_mlp, eps, uf);
    }

    // ── Steps 6 + 7: Top-K residual add + softmax over K ──
    // Gather+dot is fused: each top-K embedding row is read directly from
    // `input.embedding` and dotted with `h_weaver` once. This drops the
    // `gathered` scratch buffer entirely (was `[max_depth * K * h]` f32,
    // ~37 MB at production config) and halves embedding memory traffic.
    for di in 0..d_depth {
        let pos = di + 1; // skip verifier position 0
        let h_weaver = &u_final[pos * h..(pos + 1) * h];
        let ids = input.topk_ids[di];
        let dfl = input.dflash_logits[di];

        // Compute residual logits directly from the vocab embedding.
        let r_off = di * k;
        for (ki, &tid) in ids.iter().enumerate() {
            let tid = tid as usize;
            debug_assert!(
                tid < input.vocab_size,
                "topk id {} >= vocab {}",
                tid,
                input.vocab_size
            );
            let row = &input.embedding[tid * h..(tid + 1) * h];
            residual_flat[r_off + ki] = dot(h_weaver, row);
        }

        // Corrected = dflash + weaver_residual; softmax over K.
        let cl_off = di * k;
        let cp_off = di * k;
        let mut max_c = f32::NEG_INFINITY;
        for ki in 0..k {
            let cl = dfl[ki] + residual_flat[r_off + ki];
            corrected_logits_flat[cl_off + ki] = cl;
            if cl > max_c {
                max_c = cl;
            }
        }
        // Softmax over K (SIMD exp-sum into corrected_probs_flat).
        use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
        corrected_probs_flat[cp_off..cp_off + k]
            .copy_from_slice(&corrected_logits_flat[cl_off..cl_off + k]);
        let cp_row = &mut corrected_probs_flat[cp_off..cp_off + k];
        simd_add_scalar_inplace(cp_row, -max_c);
        let sum_e = simd_exp_sum_inplace(cp_row);
        let inv_sum = 1.0 / sum_e;
        for cp in corrected_probs_flat[cp_off..cp_off + k].iter_mut() {
            *cp *= inv_sum;
        }
    }

    (d_depth, k)
}

/// Parallel Weaver forward pass (Issue 131 G4 optimization — rayon).
///
/// Same I/O contract as [`weaver_forward_into`] (writes into `scratch`,
/// returns `(depth, k)`), but parallelizes the heavy matmuls across positions
/// using rayon. The attention step (step 3) remains sequential because of
/// causal masking; everything else is embarrassingly parallel across the
/// `seq_len = depth + 1` positions.
///
/// # Speedup
///
/// On M3 Max (12 P-cores, hidden=2304, seq_len=5): ~3.2× over sequential.
/// The theoretical max is `seq_len`× (one thread per position), but memory
/// bandwidth contention (all threads read the same weight matrices)
/// caps the speedup below the thread count.
///
/// # When to use
///
/// Use this when `seq_len ≥ 2` and each position's matmul work exceeds
/// rayon's thread-pool overhead (~5µs). At hidden=2304, a single position's
/// matmul is ~4 ms — well above the threshold. For tiny configs (test
/// config hidden=32), the sequential path is faster (overhead-dominated).
pub fn weaver_forward_parallel(
    weights: &WeaverWeights,
    input: &WeaverInput,
    scratch: &mut WeaverScratch,
) -> (usize, usize) {
    use rayon::prelude::*;

    let cfg = &weights.config;
    let h = cfg.hidden_dim;
    let k = cfg.k_candidates;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let d_ff = cfg.d_ff;
    let eps = cfg.rms_eps;
    let d_depth = input.h_dflash.len();
    let seq_len = d_depth + 1;

    debug_assert_eq!(input.h_verifier.len(), h);
    debug_assert!(
        seq_len <= cfg.max_depth + 1,
        "seq_len exceeds scratch capacity"
    );

    // Borrow scratch fields. We need mutable access to per-position rows.
    // Rayon requires Sync/Send on the closure captures. All our buffers are
    // `Vec<f32>` (Send + Sync), and we split them into disjoint per-position
    // slices — safe.
    let WeaverScratch {
        u_cond,
        q,
        kk,
        v,
        attn_out,
        u_attn_normed,
        u_final,
        normed_buf: _,
        post_buf: _,
        gate,
        up,
        down: _,
        scores,
        residual_flat,
        corrected_logits_flat,
        corrected_probs_flat,
        top_pairs: _,
    } = scratch;

    // ── Step 1+2 (PARALLEL): Conditioning + QKV per position ──
    // Each position independently: RMSNorm → W_c → (+ pos_emb) → W_q/W_k/W_v.
    // Reads 5 weight matrices (w_c, w_q, w_k, w_v) per position. With rayon,
    // all `seq_len` positions compute in parallel.
    //
    // Per-thread scratch avoids a heap allocation per rayon iteration.
    // Safety: thread_local guarantees exclusive per-thread access; rayon's
    // work-stealing ensures each closure runs on one thread at a time.
    thread_local! {
        static NORMED_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
    }
    u_cond
        .par_chunks_mut(h)
        .zip(q.par_chunks_mut(h))
        .zip(kk.par_chunks_mut(h))
        .zip(v.par_chunks_mut(h))
        .enumerate()
        .for_each(|(pos, (((u_row, q_row), k_row), v_row))| {
            let raw = if pos == 0 {
                input.h_verifier
            } else {
                input.h_dflash[pos - 1]
            };
            // RMSNorm + W_c into u_row.
            NORMED_BUF.with(|buf| {
                let normed = unsafe { &mut *buf.get() };
                if normed.len() < h {
                    normed.resize(h, 0.0);
                }
                let normed = &mut normed[..h];
                rmsnorm_into(raw, &weights.norm_cond, eps, normed);
                matmul_vec(normed, &weights.w_c, h, h, u_row);
            });
            if pos > 0 {
                let pe = &weights.pos_emb[(pos - 1) * h..pos * h];
                for j in 0..h {
                    u_row[j] += pe[j];
                }
            }
            // QKV from u_row.
            matmul_vec(u_row, &weights.w_q, h, h, q_row);
            matmul_vec(u_row, &weights.w_k, h, h, k_row);
            matmul_vec(u_row, &weights.w_v, h, h, v_row);
        });

    // ── Step 3 (SEQUENTIAL): Causal multi-head attention ──
    // Cannot parallelize across query positions (causal dependency: each
    // query attends to all previous keys). This step is ~3% of total FLOPs.
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    attn_out[..seq_len * h].fill(0.0);
    for head in 0..n_heads {
        let ho = head * head_dim;
        for qi in 0..seq_len {
            let q_row = &q[qi * h + ho..qi * h + ho + head_dim];
            let mut max_s = f32::NEG_INFINITY;
            for kj in 0..=qi {
                let k_row = &kk[kj * h + ho..kj * h + ho + head_dim];
                let s = dot(q_row, k_row) * attn_scale;
                scores[kj] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            // Fused softmax: SIMD shift + exp-sum in one pass over `scores[..=qi]`.
            use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
            let s_row = &mut scores[..=qi];
            simd_add_scalar_inplace(s_row, -max_s);
            let sum_e = simd_exp_sum_inplace(s_row);
            let inv_sum = 1.0 / sum_e;
            let out_row = &mut attn_out[qi * h + ho..qi * h + ho + head_dim];
            for kj in 0..=qi {
                let w = scores[kj] * inv_sum;
                let v_row = &v[kj * h + ho..kj * h + ho + head_dim];
                katgpt_core::simd::simd_fused_scale_acc(out_row, v_row, w, head_dim);
            }
        }
    }

    // ── Step 4+5 (PARALLEL): Output projection + MLP per position ──
    // Each position independently: W_o → (+ residual) → RMSNorm → SwiGLU → (+ residual) → RMSNorm.
    // Reads 4 weight matrices (w_o, w_gate, w_up, w_down) per position.
    // Per-thread scratch (3 buffers: tmp_o, post, down) — avoids 3 heap
    // allocations per rayon iteration.
    thread_local! {
        static TMP_O_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
        static POST_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
        static DOWN_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
    }
    u_attn_normed
        .par_chunks_mut(h)
        .zip(u_final.par_chunks_mut(h))
        .zip(u_cond.par_chunks(h))
        .zip(gate.par_chunks_mut(d_ff))
        .zip(up.par_chunks_mut(d_ff))
        .enumerate()
        .for_each(|(pos, ((((ua_norm, uf), uc), gate_row), up_row))| {
            TMP_O_BUF.with(|tmp_o_cell| {
                POST_BUF.with(|post_cell| {
                    DOWN_BUF.with(|down_cell| {
                        let tmp_o = unsafe { &mut *tmp_o_cell.get() };
                        let post = unsafe { &mut *post_cell.get() };
                        let down = unsafe { &mut *down_cell.get() };
                        if tmp_o.len() < h {
                            tmp_o.resize(h, 0.0);
                            post.resize(h, 0.0);
                            down.resize(h, 0.0);
                        }
                        let tmp_o = &mut tmp_o[..h];
                        let post = &mut post[..h];
                        let down = &mut down[..h];

                        // W_o into a per-position scratch, then add residual.
                        matmul_vec(&attn_out[pos * h..(pos + 1) * h], &weights.w_o, h, h, tmp_o);
                        // post = u_cond + tmp_o; RMSNorm → ua_norm.
                        for j in 0..h {
                            post[j] = uc[j] + tmp_o[j];
                        }
                        rmsnorm_into(post, &weights.norm_attn, eps, ua_norm);

                        // SwiGLU: gate = silu(W_gate · ua_norm) * (W_up · ua_norm).
                        matmul_vec(ua_norm, &weights.w_gate, h, d_ff, gate_row);
                        matmul_vec(ua_norm, &weights.w_up, h, d_ff, up_row);
                        for j in 0..d_ff {
                            gate_row[j] = silu(gate_row[j]) * up_row[j];
                        }
                        // W_down · gate.
                        matmul_vec(gate_row, &weights.w_down, d_ff, h, down);
                        // post = ua_norm + down; RMSNorm → uf.
                        for j in 0..h {
                            post[j] = ua_norm[j] + down[j];
                        }
                        rmsnorm_into(post, &weights.norm_mlp, eps, uf);
                    });
                });
            });
        });

    // ── Steps 6+7 (PARALLEL): Top-K residual + softmax per depth ──
    // Each depth is independent. The gather+dot is fused: each top-K
    // embedding row is read directly from `input.embedding` and dotted with
    // `h_weaver` once — no scratch buffer needed (h_weaver stays in L1 across
    // the K reads). This also drops the per-thread `GROW_BUF` thread_local
    // entirely.
    residual_flat
        .par_chunks_mut(k)
        .zip(corrected_logits_flat.par_chunks_mut(k))
        .zip(corrected_probs_flat.par_chunks_mut(k))
        .enumerate()
        .for_each(|(di, ((resid_row, cl_row), cp_row))| {
            let pos = di + 1; // skip verifier position 0
            let h_weaver = &u_final[pos * h..(pos + 1) * h];
            let ids = input.topk_ids[di];
            let dfl = input.dflash_logits[di];

            // Compute residual logits directly from the vocab embedding.
            for (ki, &tid) in ids.iter().enumerate() {
                let tid = tid as usize;
                debug_assert!(tid < input.vocab_size);
                let row = &input.embedding[tid * h..(tid + 1) * h];
                resid_row[ki] = dot(h_weaver, row);
            }

            // Corrected = dflash + residual; softmax over K.
            let mut max_c = f32::NEG_INFINITY;
            for ki in 0..k {
                let cl = dfl[ki] + resid_row[ki];
                cl_row[ki] = cl;
                if cl > max_c {
                    max_c = cl;
                }
            }
            // SIMD softmax over K: copy cl_row → cp_row, shift, exp-sum.
            use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
            cp_row[..k].copy_from_slice(&cl_row[..k]);
            simd_add_scalar_inplace(&mut cp_row[..k], -max_c);
            let sum_e = simd_exp_sum_inplace(&mut cp_row[..k]);
            let inv_sum = 1.0 / sum_e;
            for cp in cp_row.iter_mut().take(k) {
                *cp *= inv_sum;
            }
        });

    (d_depth, k)
}

/// f16-weight parallel Weaver forward (Issue 136).
///
/// Exact mirror of [`weaver_forward_parallel`] but uses f16 weight matrices
/// via `matmul_vec_f16` + `simd_fused_scale_acc_f16`. Steps 3 (attention)
/// and 6+7 (top-K gather) are identical — they operate on f32 activations
/// and the f32 embedding table, which is passed via `WeaverInput`.
///
/// The thread-local scratch buffers are separate from the f32 path's
/// (function-scope statics are unique per function body). Each buffer is
/// a small `Vec<f32>` that grows once per thread and is reused thereafter.
pub fn weaver_forward_parallel_f16(
    weights: &WeaverWeightsF16,
    input: &WeaverInput,
    scratch: &mut WeaverScratch,
) -> (usize, usize) {
    use rayon::prelude::*;

    let cfg = &weights.config;
    let h = cfg.hidden_dim;
    let k = cfg.k_candidates;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let d_ff = cfg.d_ff;
    let eps = cfg.rms_eps;
    let d_depth = input.h_dflash.len();
    let seq_len = d_depth + 1;

    debug_assert_eq!(input.h_verifier.len(), h);
    debug_assert!(
        seq_len <= cfg.max_depth + 1,
        "seq_len exceeds scratch capacity"
    );

    let WeaverScratch {
        u_cond,
        q,
        kk,
        v,
        attn_out,
        u_attn_normed,
        u_final,
        normed_buf: _,
        post_buf: _,
        gate,
        up,
        down: _,
        scores,
        residual_flat,
        corrected_logits_flat,
        corrected_probs_flat,
        top_pairs: _,
    } = scratch;

    // ── Step 1+2 (PARALLEL): Conditioning + QKV per position (f16 weights) ──
    thread_local! {
        static F16_NORMED_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
    }
    u_cond
        .par_chunks_mut(h)
        .zip(q.par_chunks_mut(h))
        .zip(kk.par_chunks_mut(h))
        .zip(v.par_chunks_mut(h))
        .enumerate()
        .for_each(|(pos, (((u_row, q_row), k_row), v_row))| {
            let raw = if pos == 0 {
                input.h_verifier
            } else {
                input.h_dflash[pos - 1]
            };
            F16_NORMED_BUF.with(|buf| {
                let normed = unsafe { &mut *buf.get() };
                if normed.len() < h {
                    normed.resize(h, 0.0);
                }
                let normed = &mut normed[..h];
                rmsnorm_into(raw, &weights.norm_cond, eps, normed);
                matmul_vec_f16(normed, &weights.w_c, h, h, u_row);
            });
            if pos > 0 {
                let pe = &weights.pos_emb[(pos - 1) * h..pos * h];
                for j in 0..h {
                    u_row[j] += pe[j];
                }
            }
            matmul_vec_f16(u_row, &weights.w_q, h, h, q_row);
            matmul_vec_f16(u_row, &weights.w_k, h, h, k_row);
            matmul_vec_f16(u_row, &weights.w_v, h, h, v_row);
        });

    // ── Step 3 (SEQUENTIAL): Causal multi-head attention (f32, unchanged) ──
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    attn_out[..seq_len * h].fill(0.0);
    for head in 0..n_heads {
        let ho = head * head_dim;
        for qi in 0..seq_len {
            let q_row = &q[qi * h + ho..qi * h + ho + head_dim];
            let mut max_s = f32::NEG_INFINITY;
            for kj in 0..=qi {
                let k_row = &kk[kj * h + ho..kj * h + ho + head_dim];
                let s = dot(q_row, k_row) * attn_scale;
                scores[kj] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            // Fused softmax: SIMD shift + exp-sum in one pass over `scores[..=qi]`.
            use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
            let s_row = &mut scores[..=qi];
            simd_add_scalar_inplace(s_row, -max_s);
            let sum_e = simd_exp_sum_inplace(s_row);
            let inv_sum = 1.0 / sum_e;
            let out_row = &mut attn_out[qi * h + ho..qi * h + ho + head_dim];
            for kj in 0..=qi {
                let w = scores[kj] * inv_sum;
                let v_row = &v[kj * h + ho..kj * h + ho + head_dim];
                katgpt_core::simd::simd_fused_scale_acc(out_row, v_row, w, head_dim);
            }
        }
    }

    // ── Step 4+5 (PARALLEL): Output projection + MLP per position (f16 weights) ──
    thread_local! {
        static F16_TMP_O_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
        static F16_POST_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
        static F16_DOWN_BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
    }
    u_attn_normed
        .par_chunks_mut(h)
        .zip(u_final.par_chunks_mut(h))
        .zip(u_cond.par_chunks(h))
        .zip(gate.par_chunks_mut(d_ff))
        .zip(up.par_chunks_mut(d_ff))
        .enumerate()
        .for_each(|(pos, ((((ua_norm, uf), uc), gate_row), up_row))| {
            F16_TMP_O_BUF.with(|tmp_o_cell| {
                F16_POST_BUF.with(|post_cell| {
                    F16_DOWN_BUF.with(|down_cell| {
                        let tmp_o = unsafe { &mut *tmp_o_cell.get() };
                        let post = unsafe { &mut *post_cell.get() };
                        let down = unsafe { &mut *down_cell.get() };
                        if tmp_o.len() < h {
                            tmp_o.resize(h, 0.0);
                            post.resize(h, 0.0);
                            down.resize(h, 0.0);
                        }
                        let tmp_o = &mut tmp_o[..h];
                        let post = &mut post[..h];
                        let down = &mut down[..h];

                        matmul_vec_f16(
                            &attn_out[pos * h..(pos + 1) * h],
                            &weights.w_o,
                            h,
                            h,
                            tmp_o,
                        );
                        for j in 0..h {
                            post[j] = uc[j] + tmp_o[j];
                        }
                        rmsnorm_into(post, &weights.norm_attn, eps, ua_norm);

                        matmul_vec_f16(ua_norm, &weights.w_gate, h, d_ff, gate_row);
                        matmul_vec_f16(ua_norm, &weights.w_up, h, d_ff, up_row);
                        for j in 0..d_ff {
                            gate_row[j] = silu(gate_row[j]) * up_row[j];
                        }
                        matmul_vec_f16(gate_row, &weights.w_down, d_ff, h, down);
                        for j in 0..h {
                            post[j] = ua_norm[j] + down[j];
                        }
                        rmsnorm_into(post, &weights.norm_mlp, eps, uf);
                    });
                });
            });
        });

    // ── Steps 6+7 (PARALLEL): Top-K residual + softmax (f32, unchanged) ──
    // Gather+dot fused — see `weaver_forward_parallel` for rationale.
    residual_flat
        .par_chunks_mut(k)
        .zip(corrected_logits_flat.par_chunks_mut(k))
        .zip(corrected_probs_flat.par_chunks_mut(k))
        .enumerate()
        .for_each(|(di, ((resid_row, cl_row), cp_row))| {
            let pos = di + 1;
            let h_weaver = &u_final[pos * h..(pos + 1) * h];
            let ids = input.topk_ids[di];
            let dfl = input.dflash_logits[di];

            for (ki, &tid) in ids.iter().enumerate() {
                let tid = tid as usize;
                debug_assert!(tid < input.vocab_size);
                let row = &input.embedding[tid * h..(tid + 1) * h];
                resid_row[ki] = dot(h_weaver, row);
            }

            let mut max_c = f32::NEG_INFINITY;
            for ki in 0..k {
                let cl = dfl[ki] + resid_row[ki];
                cl_row[ki] = cl;
                if cl > max_c {
                    max_c = cl;
                }
            }
            // SIMD softmax over K: copy cl_row → cp_row, shift, exp-sum.
            use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_sum_inplace};
            cp_row[..k].copy_from_slice(&cl_row[..k]);
            simd_add_scalar_inplace(&mut cp_row[..k], -max_c);
            let sum_e = simd_exp_sum_inplace(&mut cp_row[..k]);
            let inv_sum = 1.0 / sum_e;
            for cp in cp_row.iter_mut().take(k) {
                *cp *= inv_sum;
            }
        });

    (d_depth, k)
}

/// Matrix-vector multiply: `output[j] = Σ_i input[i] · weight[i · out_dim + j]`.
///
/// The weight matrix is `[in_dim, out_dim]` row-major. Uses AXPY iteration
/// (for each input element, scale-add the weight row into output) which is
/// cache-friendly for this layout. The inner AXPY delegates to
/// `simd_fused_scale_acc` for NEON/AVX2 dispatch.
#[inline]
fn matmul_vec(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize, output: &mut [f32]) {
    output[..out_dim].fill(0.0);
    for i in 0..in_dim {
        let xi = input[i];
        let row = &weight[i * out_dim..(i + 1) * out_dim];
        katgpt_core::simd::simd_fused_scale_acc(output, row, xi, out_dim);
    }
}

// ── f16 Weight Path (Issue 136) ─────────────────────────────────────────
//
// Stores weight matrices as `half::f16` to halve memory traffic. The f16→f32
// conversion happens inside `simd_fused_scale_acc_f16` during the AXPY loop,
// at 1 cycle per 4 elements on aarch64 NEON (hardware fcvt). The norm scales
// and position embeddings stay f32 (they are small: h + max_depth*h elements).
// The embedding table stays f32 because it is passed via `WeaverInput` by the
// caller and is not part of the corrector's weight budget.
//
// The f16 path is a sibling variant: `WeaverWeightsF16` mirrors `WeaverWeights`,
// `matmul_vec_f16` mirrors `matmul_vec`, and `weaver_forward_parallel_f16`
// mirrors `weaver_forward_parallel`. Callers explicitly opt in via
// `WeaverCorrectorF16`. The f32 path is preserved bit-identical (G3).

/// Compact f16 weight storage with **transposed layout** for dot-product GEMV.
///
/// Weight matrices are stored as `[out_dim, in_dim]` row-major (transposed
/// from `WeaverWeights`'s `[in_dim, out_dim]`). This enables the dot-product
/// GEMV pattern: `output[o] = simd_dot_f16_f32(&weight_t[o*in_dim..], input)`.
///
/// The dot-product pattern is critical for f16: the AXPY pattern re-reads/
/// re-writes the f32 output `in_dim` times, leaving only 17% theoretical
/// bandwidth reduction — not enough to overcome f16→f32 conversion overhead
/// (measured: 0.71× — a regression). The dot-product pattern reads each f16
/// weight row once and accumulates in a register, achieving the full 50%
/// weight-bandwidth reduction with minimal conversion overhead.
///
/// Norm scales and pos_emb stay f32 (small arrays).
#[derive(Debug, Clone)]
pub struct WeaverWeightsF16 {
    /// Conditioning projection `[hidden, hidden]` transposed (f16).
    pub w_c: Vec<half::f16>,
    /// Attention Q/K/V/O projections `[hidden, hidden]` transposed (f16).
    pub w_q: Vec<half::f16>,
    pub w_k: Vec<half::f16>,
    pub w_v: Vec<half::f16>,
    pub w_o: Vec<half::f16>,
    /// SwiGLU gate/up `[d_ff, hidden]` transposed, down `[hidden, d_ff]` transposed (f16).
    pub w_gate: Vec<half::f16>,
    pub w_up: Vec<half::f16>,
    pub w_down: Vec<half::f16>,
    /// RMSNorm scales stay f32 — only `[hidden]` each, negligible bandwidth.
    pub norm_cond: Vec<f32>,
    pub norm_attn: Vec<f32>,
    pub norm_mlp: Vec<f32>,
    /// Position embeddings stay f32 — only `[max_depth * hidden]`, small.
    pub pos_emb: Vec<f32>,
    /// Config snapshot.
    pub config: WeaverConfig,
}

impl WeaverWeightsF16 {
    /// Convert f32 weights to f16 with transposition. One-time cost at load time.
    ///
    /// Each weight matrix `[in_dim, out_dim]` is transposed to `[out_dim, in_dim]`
    /// and converted to f16. The transposition enables the dot-product GEMV
    /// pattern which is ~1.5-2× faster than the AXPY pattern for f16 weights.
    pub fn from_f32(src: &WeaverWeights) -> Self {
        // Transpose `[in_dim, out_dim]` → `[out_dim, in_dim]` and convert to f16.
        let cvt_t = |v: &[f32], in_dim: usize, out_dim: usize| -> Vec<half::f16> {
            let mut t = vec![half::f16::ZERO; in_dim * out_dim];
            for i in 0..in_dim {
                for o in 0..out_dim {
                    // src[i * out_dim + o] → dst[o * in_dim + i]
                    t[o * in_dim + i] = half::f16::from_f32(v[i * out_dim + o]);
                }
            }
            t
        };
        let h = src.config.hidden_dim;
        let ff = src.config.d_ff;
        Self {
            w_c: cvt_t(&src.w_c, h, h),
            w_q: cvt_t(&src.w_q, h, h),
            w_k: cvt_t(&src.w_k, h, h),
            w_v: cvt_t(&src.w_v, h, h),
            w_o: cvt_t(&src.w_o, h, h),
            w_gate: cvt_t(&src.w_gate, h, ff),
            w_up: cvt_t(&src.w_up, h, ff),
            w_down: cvt_t(&src.w_down, ff, h),
            norm_cond: src.norm_cond.clone(),
            norm_attn: src.norm_attn.clone(),
            norm_mlp: src.norm_mlp.clone(),
            pos_emb: src.pos_emb.clone(),
            config: src.config.clone(),
        }
    }
}

/// f16×f32 matrix-vector multiply using the **dot-product pattern**:
/// `output[o] = simd_dot_f16_f32(&weight_t[o*in_dim..], input)`.
///
/// The weight matrix is `[out_dim, in_dim]` row-major (transposed from the
/// f32 layout). Each output element is a single dot product of an f16 weight
/// row against the f32 input. The dot product accumulates in registers —
/// no per-element write-back like the AXPY pattern.
///
/// This pattern is critical for f16: it achieves the full 50% weight-bandwidth
/// reduction. The AXPY pattern would re-read/re-write the f32 output `in_dim`
/// times, leaving only 17% bandwidth reduction — not enough to overcome
/// f16→f32 conversion overhead (measured regression: 0.71×).
#[inline]
fn matmul_vec_f16(
    input: &[f32],
    weight_t_f16: &[half::f16],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    for o in 0..out_dim {
        let row = &weight_t_f16[o * in_dim..(o + 1) * in_dim];
        output[o] = katgpt_core::simd::simd_dot_f16_f32(row, input, in_dim);
    }
}

/// Batched matrix-vector multiply: for each batch `b`, computes
/// `output[b·out_dim + j] = Σ_i input[b·in_dim + i] · weight[i·out_dim + j]`.
///
/// **This is the Issue 131 G4 optimization:** the non-batched `matmul_vec`
/// reads the full weight matrix once per position. In the Weaver forward pass,
/// it is called `seq_len=5` times per weight matrix (once per position), so
/// each weight matrix is streamed from memory 5×. This batched variant reads
/// each weight row **once** and applies it to all `batch` positions — a
/// `seq_len`× reduction in weight-matrix memory traffic.
///
/// The weight matrix layout is `[in_dim, out_dim]` row-major (same as
/// `matmul_vec`). The input is `[batch, in_dim]` row-major, output is
/// `[batch, out_dim]` row-major.
///
/// # Memory traffic comparison (h=2304, d_ff=4096, seq_len=5)
///
/// - Non-batched (5 calls to `matmul_vec`): 5 × h × h = 26.5M weight reads
///   per matrix; 5 × (h×h + h×h) = 53M memory ops (read weight + RMW output).
/// - Batched (1 call): h × h = 5.3M weight reads; h × (h + batch×h) = 5.3M × 6
///   = 31.8M memory ops (read weight once + read input + RMW batch outputs).
///
/// Net: ~1.7× fewer memory ops for the h×h matrices, and the weight matrix
/// stays hot in L2/L3 cache across all batch positions instead of being
/// re-streamed from DRAM.
#[inline]
fn matmul_vec_batched(
    input: &[f32],
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    batch: usize,
    output: &mut [f32],
) {
    // Zero all batch outputs.
    output[..batch * out_dim].fill(0.0);
    // For each input dimension, read the weight row once and AXPY it into
    // every batch's output. This streams the weight matrix sequentially
    // (cache-friendly) and reuses each row across all batch positions.
    for i in 0..in_dim {
        let row = &weight[i * out_dim..(i + 1) * out_dim];
        for b in 0..batch {
            let xi = input[b * in_dim + i];
            let out_row = &mut output[b * out_dim..(b + 1) * out_dim];
            katgpt_core::simd::simd_fused_scale_acc(out_row, row, xi, out_dim);
        }
    }
}

/// RMSNorm: `output = x / sqrt(mean(x²) + eps) · scale`.
///
/// Writes into `output` — zero allocation. The caller must ensure `output`
/// has the same length as `x` and `scale`.
#[inline]
fn rmsnorm_into(x: &[f32], scale: &[f32], eps: f32, output: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for (o, (&v, &s)) in output.iter_mut().zip(x.iter().zip(scale.iter())) {
        *o = v * inv_rms * s;
    }
}

/// SiLU / Swish activation: `x * σ(x)`.
///
/// Delegates to `katgpt_core::simd::fast_sigmoid` (Cephes polynomial).
#[inline]
fn silu(x: f32) -> f32 {
    x * katgpt_core::simd::fast_sigmoid(x)
}

/// Dot product — delegates to `simd_dot_f32` for NEON/AVX2 dispatch.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    katgpt_core::simd::simd_dot_f32(a, b, a.len().min(b.len()))
}

// ── Error type ───────────────────────────────────────────────────────────

/// Errors that can occur during Weaver checkpoint loading.
#[derive(Debug)]
pub enum WeaverLoadError {
    /// safetensors format parse failure.
    SafetensorsParse(safetensors::SafeTensorError),
    /// A metadata key is missing or unparseable.
    MetadataParse { key: String, value: String },
    /// A tensor is missing from the checkpoint.
    TensorMissing {
        name: String,
        source: safetensors::SafeTensorError,
    },
    /// A tensor has the wrong number of elements.
    ShapeMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    /// BLAKE3 sidecar verification failed.
    Blake3Mismatch { expected: String, actual: String },
    /// Filesystem I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for WeaverLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafetensorsParse(e) => write!(f, "safetensors parse error: {e}"),
            Self::MetadataParse { key, value } => {
                write!(f, "cannot parse metadata '{key}' from value '{value}'")
            }
            Self::TensorMissing { name, .. } => {
                write!(f, "tensor '{name}' not found in checkpoint")
            }
            Self::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{tensor}' has {actual} elements, expected {expected}"
            ),
            Self::Blake3Mismatch { expected, actual } => {
                write!(f, "BLAKE3 mismatch: expected {expected}, got {actual}")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for WeaverLoadError {}

/// Extract the UTF-8 JSON header from raw safetensors bytes.
fn parse_safetensors_header(bytes: &[u8]) -> Result<String, WeaverLoadError> {
    if bytes.len() < 8 {
        return Err(WeaverLoadError::SafetensorsParse(
            safetensors::SafeTensorError::InvalidHeader,
        ));
    }
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    if bytes.len() < 8 + header_len {
        return Err(WeaverLoadError::SafetensorsParse(
            safetensors::SafeTensorError::InvalidHeader,
        ));
    }
    std::str::from_utf8(&bytes[8..8 + header_len])
        .map(String::from)
        .map_err(|_| WeaverLoadError::SafetensorsParse(safetensors::SafeTensorError::InvalidHeader))
}

/// Search the JSON header for a metadata value like `"hidden_dim":"2048"`.
/// This avoids needing a full JSON parser — the safetensors metadata keys
/// are simple string-to-string maps with numeric values.
fn extract_meta(header: &str, key: &str) -> Result<usize, WeaverLoadError> {
    // Look for the pattern: "key":"value"
    // The metadata section appears as: "__metadata__":{"hidden_dim":"2048",...}
    let needle = format!("\"{key}\":\"");
    let start = header
        .find(&needle)
        .ok_or_else(|| WeaverLoadError::MetadataParse {
            key: key.to_string(),
            value: "<missing>".to_string(),
        })?;
    let val_start = start + needle.len();
    let val_end = header[val_start..]
        .find('"')
        .ok_or_else(|| WeaverLoadError::MetadataParse {
            key: key.to_string(),
            value: "<unterminated>".to_string(),
        })?;
    let val_str = &header[val_start..val_start + val_end];
    val_str
        .parse::<usize>()
        .map_err(|_| WeaverLoadError::MetadataParse {
            key: key.to_string(),
            value: val_str.to_string(),
        })
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Small config for fast tests.
    fn test_config() -> WeaverConfig {
        WeaverConfig {
            hidden_dim: 32,
            n_heads: 4,
            k_candidates: 8,
            n_layer: 1,
            d_ff: 64,
            rms_eps: 1e-6,
            max_depth: 3,
        }
    }

    /// Build a dummy WeaverInput for the test config.
    fn test_input(cfg: &WeaverConfig, vocab_size: usize) -> WeaverInput<'static> {
        // We can't easily build 'static slices; leak the data.
        let h = cfg.hidden_dim;
        let k = cfg.k_candidates;
        let d = cfg.max_depth;

        let h_verifier: &'static [f32] = Box::leak(vec![0.5f32; h].into_boxed_slice());
        let mut h_dflash: Vec<&'static [f32]> = Vec::with_capacity(d);
        let mut topk_ids: Vec<&'static [u32]> = Vec::with_capacity(d);
        let mut dflash_logits: Vec<&'static [f32]> = Vec::with_capacity(d);
        for di in 0..d {
            h_dflash.push(Box::leak(
                (0..h)
                    .map(|i| 0.3 + 0.01 * (di + i) as f32)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            ));
            topk_ids.push(Box::leak(
                (0..k)
                    .map(|i| (i as u32) % vocab_size as u32)
                    .collect::<Vec<u32>>()
                    .into_boxed_slice(),
            ));
            dflash_logits.push(Box::leak(
                (0..k)
                    .map(|i| (i as f32) * 0.1)
                    .collect::<Vec<f32>>()
                    .into_boxed_slice(),
            ));
        }
        let emb: &'static [f32] = Box::leak(vec![0.1f32; vocab_size * h].into_boxed_slice());

        WeaverInput {
            h_verifier,
            h_dflash: Box::leak(h_dflash.into_boxed_slice()),
            topk_ids: Box::leak(topk_ids.into_boxed_slice()),
            dflash_logits: Box::leak(dflash_logits.into_boxed_slice()),
            embedding: emb,
            vocab_size,
        }
    }

    // ── G1: Correctness gates ──

    #[test]
    fn g1_zero_weights_produce_zero_residual() {
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        assert_eq!(out.depth, cfg.max_depth);
        assert_eq!(out.k, cfg.k_candidates);

        // Zero weights → zero attention, zero MLP, but RMSNorm of non-zero input
        // is non-zero. However, all weight matrices are zero, so every matmul
        // output is zero. The final u_final is RMSNorm(post_mlp, norm_mlp=0)
        // which is zero (scale=0). So residuals are dot(0_vector, embedding) = 0.
        for di in 0..out.depth {
            for ki in 0..out.k {
                assert!(
                    out.weaver_residual[di][ki].abs() < 1e-6,
                    "non-zero residual at di={di} ki={ki}: {}",
                    out.weaver_residual[di][ki]
                );
            }
        }
    }

    #[test]
    fn g1_corrected_probs_sum_to_one() {
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        for di in 0..out.depth {
            let sum: f32 = out.corrected_probs[di].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "probs at di={di} sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn g1_no_nan_or_inf_in_output() {
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        for di in 0..out.depth {
            for ki in 0..out.k {
                assert!(out.corrected_probs[di][ki].is_finite(), "NaN/Inf in probs");
                assert!(
                    out.corrected_logits[di][ki].is_finite(),
                    "NaN/Inf in logits"
                );
                assert!(
                    out.weaver_residual[di][ki].is_finite(),
                    "NaN/Inf in residual"
                );
            }
        }
    }

    #[test]
    fn g1_zero_weights_corrected_equals_dflash() {
        // With zero residual, corrected_logits == dflash_logits.
        let cfg = test_config();
        let weights = WeaverWeights::zeros(cfg.clone());
        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        for di in 0..out.depth {
            for ki in 0..out.k {
                let diff = (out.corrected_logits[di][ki] - input.dflash_logits[di][ki]).abs();
                assert!(
                    diff < 1e-6,
                    "corrected != dflash at di={di} ki={ki}: diff={diff}"
                );
            }
        }
    }

    // ── G3: No-regression gate (feature isolation) ──

    #[test]
    fn g3_nonzero_weights_change_logits() {
        // With non-zero weights, the corrected logits should differ from dflash.
        let cfg = test_config();
        let mut weights = WeaverWeights::zeros(cfg.clone());
        // Set non-zero RMSNorm scales so u_final is non-zero.
        weights.norm_cond.fill(1.0);
        weights.norm_attn.fill(1.0);
        weights.norm_mlp.fill(1.0);
        // Identity W_c (so conditioning preserves the hidden state direction).
        for i in 0..cfg.hidden_dim {
            weights.w_c[i * cfg.hidden_dim + i] = 1.0;
        }

        let input = test_input(&cfg, 16);
        let out = weaver_forward(&weights, &input);

        // With non-zero norm scales, u_final should be non-zero, so residuals
        // should be non-zero (dot of non-zero vector with non-zero embedding).
        let any_nonzero = (0..out.depth).flat_map(|_di| 0..out.k).any(|_| true);
        assert!(any_nonzero, "output should have entries");

        // At least some residuals should be non-zero with non-zero norms.
        let max_residual = (0..out.depth)
            .flat_map(|_di| 0..out.k)
            .map(|di_k| {
                let di = di_k / out.k;
                let ki = di_k % out.k;
                out.weaver_residual[di][ki].abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_residual > 1e-6,
            "expected non-zero residuals with non-zero norm scales, got max={max_residual}"
        );
    }

    // ── Safetensors round-trip ──

    #[test]
    fn safetensors_roundtrip() {
        use safetensors::Dtype;
        use safetensors::tensor::TensorView;

        let cfg = WeaverConfig {
            hidden_dim: 16,
            n_heads: 2,
            k_candidates: 4,
            n_layer: 1,
            d_ff: 32,
            rms_eps: 1e-6,
            max_depth: 2,
        };
        let original = WeaverWeights::zeros(cfg.clone());

        // Serialize to safetensors bytes.
        let h = cfg.hidden_dim;
        let ff = cfg.d_ff;
        let md = cfg.max_depth;

        let tensors: Vec<(String, safetensors::tensor::TensorView)> = vec![
            (
                "w_c".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_c.len()],
                    bytemuck::cast_slice(&original.w_c),
                )
                .unwrap(),
            ),
            (
                "w_q".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_q.len()],
                    bytemuck::cast_slice(&original.w_q),
                )
                .unwrap(),
            ),
            (
                "w_k".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_k.len()],
                    bytemuck::cast_slice(&original.w_k),
                )
                .unwrap(),
            ),
            (
                "w_v".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_v.len()],
                    bytemuck::cast_slice(&original.w_v),
                )
                .unwrap(),
            ),
            (
                "w_o".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_o.len()],
                    bytemuck::cast_slice(&original.w_o),
                )
                .unwrap(),
            ),
            (
                "w_gate".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_gate.len()],
                    bytemuck::cast_slice(&original.w_gate),
                )
                .unwrap(),
            ),
            (
                "w_up".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_up.len()],
                    bytemuck::cast_slice(&original.w_up),
                )
                .unwrap(),
            ),
            (
                "w_down".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.w_down.len()],
                    bytemuck::cast_slice(&original.w_down),
                )
                .unwrap(),
            ),
            (
                "norm_cond".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.norm_cond.len()],
                    bytemuck::cast_slice(&original.norm_cond),
                )
                .unwrap(),
            ),
            (
                "norm_attn".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.norm_attn.len()],
                    bytemuck::cast_slice(&original.norm_attn),
                )
                .unwrap(),
            ),
            (
                "norm_mlp".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.norm_mlp.len()],
                    bytemuck::cast_slice(&original.norm_mlp),
                )
                .unwrap(),
            ),
            (
                "pos_emb".to_string(),
                TensorView::new(
                    Dtype::F32,
                    vec![original.pos_emb.len()],
                    bytemuck::cast_slice(&original.pos_emb),
                )
                .unwrap(),
            ),
        ];

        let metadata = Some(std::collections::HashMap::from([
            ("format".to_string(), "weaver_v1".to_string()),
            ("hidden_dim".to_string(), h.to_string()),
            ("n_heads".to_string(), cfg.n_heads.to_string()),
            ("k_candidates".to_string(), cfg.k_candidates.to_string()),
            ("d_ff".to_string(), ff.to_string()),
            ("max_depth".to_string(), md.to_string()),
        ]));

        let bytes = safetensors::serialize(tensors, &metadata).expect("serialize");

        // Deserialize.
        let loaded = WeaverWeights::from_safetensors_bytes(&bytes).expect("deserialize");

        // Verify config.
        assert_eq!(loaded.config.hidden_dim, h);
        assert_eq!(loaded.config.n_heads, cfg.n_heads);
        assert_eq!(loaded.config.k_candidates, cfg.k_candidates);
        assert_eq!(loaded.config.d_ff, ff);
        assert_eq!(loaded.config.max_depth, md);

        // Verify weights bit-identically.
        assert_eq!(loaded.w_c, original.w_c);
        assert_eq!(loaded.w_q, original.w_q);
        assert_eq!(loaded.w_k, original.w_k);
        assert_eq!(loaded.w_v, original.w_v);
        assert_eq!(loaded.w_o, original.w_o);
        assert_eq!(loaded.w_gate, original.w_gate);
        assert_eq!(loaded.w_up, original.w_up);
        assert_eq!(loaded.w_down, original.w_down);
        assert_eq!(loaded.norm_cond, original.norm_cond);
        assert_eq!(loaded.norm_attn, original.norm_attn);
        assert_eq!(loaded.norm_mlp, original.norm_mlp);
        assert_eq!(loaded.pos_emb, original.pos_emb);
    }

    // ── Helper function tests ──

    #[test]
    fn matmul_vec_identity() {
        // Identity matrix: output should equal input.
        let n = 4;
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; n];
        matmul_vec(&input, &identity, n, n, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn silu_zero_is_zero() {
        assert!(silu(0.0).abs() < 1e-10);
    }

    #[test]
    fn silu_large_positive_approximates_x() {
        assert!((silu(10.0) - 10.0).abs() < 0.1);
    }

    #[test]
    fn rmsnorm_unit_scale_preserves_direction() {
        let x = vec![3.0, 4.0]; // ||x|| = 5, mean(x²) = 12.5
        let scale = vec![1.0, 1.0];
        let mut out = vec![0.0, 0.0];
        rmsnorm_into(&x, &scale, 1e-6, &mut out);
        // RMS = sqrt(12.5 + eps) ≈ 3.536
        let rms = (12.5f32 + 1e-6).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
        assert!((out[1] - 4.0 / rms).abs() < 1e-4);
    }

    // ── T3: correct_marginals_inplace (marginal corrector integration) ──

    /// Build a tiny WeaverConfig + WeaverCorrector for the marginal-corrector test.
    fn corrector_config() -> WeaverConfig {
        WeaverConfig {
            hidden_dim: 16,
            n_heads: 2,
            k_candidates: 4,
            n_layer: 1,
            d_ff: 32,
            rms_eps: 1e-6,
            max_depth: 2,
        }
    }

    #[test]
    fn t3_correct_marginals_zero_weights_preserves_topk() {
        // With zero-init weights, the Weaver residual is zero, so the corrected
        // top-K probabilities equal the drafter's top-K probabilities (after
        // renorm over K). Non-top-K positions are zeroed.
        let cfg = corrector_config();
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(cfg.clone()));

        let h = cfg.hidden_dim;
        let k = cfg.k_candidates;
        let depth = cfg.max_depth;
        let vocab = 32usize; // > k so top-K selection is meaningful

        // Build marginals: each depth has a peaked distribution with 5 peaks
        // (> k=4 so top-K selection is meaningful and pads correctly).
        let mut marginals = vec![0.0f32; depth * vocab];
        for di in 0..depth {
            marginals[di * vocab + di] = 0.5; // peak at token `di`
            marginals[di * vocab + (vocab - 1)] = 0.2; // second peak
            marginals[di * vocab + 10] = 0.15; // third peak
            marginals[di * vocab + 11] = 0.1; // fourth peak
            marginals[di * vocab + 12] = 0.05; // fifth peak (below K, gets zeroed)
            // rest is 0
        }

        // Save the original top-K for comparison.
        let mut orig_topk: Vec<(usize, f32)> = Vec::new();
        for di in 0..depth {
            let row = &marginals[di * vocab..(di + 1) * vocab];
            let mut sorted: Vec<(usize, f32)> = row
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, p)| *p > 0.0)
                .collect();
            sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
            orig_topk.extend(sorted.into_iter().take(k));
        }

        // Hidden states + embedding (arbitrary non-degenerate values).
        let h_verifier: Vec<f32> = vec![0.5; h];
        let h_dflash_owned: Vec<Vec<f32>> = (0..depth)
            .map(|di| (0..h).map(|j| 0.3 + 0.01 * (di * h + j) as f32).collect())
            .collect();
        let h_dflash: Vec<&[f32]> = h_dflash_owned.iter().map(|v| v.as_slice()).collect();
        let embedding: Vec<f32> = vec![0.1; vocab * h];

        corrector
            .correct_marginals_inplace(&mut marginals, &h_verifier, &h_dflash, &embedding, vocab)
            .expect("correction should succeed");

        // After correction: exactly K non-zero entries per depth (the top-K).
        for di in 0..depth {
            let row = &marginals[di * vocab..(di + 1) * vocab];
            let nonzero_count = row.iter().filter(|&&p| p > 1e-10).count();
            assert_eq!(
                nonzero_count, k,
                "depth {di}: expected exactly {k} non-zero entries, got {nonzero_count}"
            );
            // Probabilities must sum to 1.0 (renormalized over K).
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "depth {di}: probs sum to {sum}");
            // No NaN/Inf.
            for &p in row {
                assert!(p.is_finite(), "depth {di}: NaN/Inf in probs");
            }
        }

        // The top-K token ids should be preserved (the same tokens that had
        // mass before still have mass after, with zero weights). The top-4 by
        // mass are: di (0.5), vocab-1 (0.2), 10 (0.15), 11 (0.1). Token 12
        // (0.05) is rank 5 — below K, gets zeroed.
        for di in 0..depth {
            let row = &marginals[di * vocab..(di + 1) * vocab];
            assert!(
                row[di] > 1e-6,
                "depth {di}: original peak at {di} lost mass"
            );
            assert!(
                row[vocab - 1] > 1e-6,
                "depth {di}: original peak at {} lost mass",
                vocab - 1
            );
            assert!(row[10] > 1e-6, "depth {di}: original peak at 10 lost mass");
            assert!(row[11] > 1e-6, "depth {di}: original peak at 11 lost mass");
            // Token 12 was rank 5 — below K=4, should be zeroed.
            assert!(
                row[12] < 1e-10,
                "depth {di}: rank-5 token at 12 should be zeroed"
            );
        }
    }

    #[test]
    fn t3_correct_marginals_rejects_bad_shapes() {
        let cfg = corrector_config();
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(cfg.clone()));

        let h = cfg.hidden_dim;
        let depth = cfg.max_depth;
        let vocab = 32usize;

        let h_verifier = vec![0.5f32; h];
        let h_dflash_owned: Vec<Vec<f32>> = (0..depth).map(|_| vec![0.3; h]).collect();
        let h_dflash: Vec<&[f32]> = h_dflash_owned.iter().map(|v| v.as_slice()).collect();
        let embedding = vec![0.1f32; vocab * h];

        // Wrong marginals length (depth * vocab + 1).
        let mut bad_marginals = vec![0.0f32; depth * vocab + 1];
        let err = corrector
            .correct_marginals_inplace(
                &mut bad_marginals,
                &h_verifier,
                &h_dflash,
                &embedding,
                vocab,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            super::WeaverCorrectError::MarginalsShape { .. }
        ));

        // Wrong h_verifier length.
        let mut marginals = vec![0.0f32; depth * vocab];
        let bad_h = vec![0.5f32; h + 1];
        let err = corrector
            .correct_marginals_inplace(&mut marginals, &bad_h, &h_dflash, &embedding, vocab)
            .unwrap_err();
        assert!(matches!(err, super::WeaverCorrectError::HiddenShape { .. }));

        // Depth exceeds max_depth.
        let too_deep_owned: Vec<Vec<f32>> = (0..=depth).map(|_| vec![0.3; h]).collect();
        let too_deep: Vec<&[f32]> = too_deep_owned.iter().map(|v| v.as_slice()).collect();
        let mut deep_marginals = vec![0.0f32; (depth + 1) * vocab];
        let err = corrector
            .correct_marginals_inplace(
                &mut deep_marginals,
                &h_verifier,
                &too_deep,
                &embedding,
                vocab,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            super::WeaverCorrectError::DepthExceedsConfig { .. }
        ));
    }

    // ── G4: Scratch path equivalence + latency (Issue 131 G4) ──

    /// Build non-trivial weights (identity W_c, unit norm scales) so the
    /// forward pass produces non-zero, non-trivial residuals. This exercises
    /// the full matmul + attention + MLP + top-K path.
    fn nonzero_weights(cfg: &WeaverConfig) -> WeaverWeights {
        let mut w = WeaverWeights::zeros(cfg.clone());
        // Unit norm scales so RMSNorm preserves magnitude.
        w.norm_cond.fill(1.0);
        w.norm_attn.fill(1.0);
        w.norm_mlp.fill(1.0);
        // Identity W_c.
        for i in 0..cfg.hidden_dim {
            w.w_c[i * cfg.hidden_dim + i] = 1.0;
        }
        // Small random-ish weights for other matrices (deterministic).
        let mut seed = 42u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16777216.0 - 0.5
        };
        for w_val in &mut w.w_q {
            *w_val = rng() * 0.1;
        }
        for w_val in &mut w.w_k {
            *w_val = rng() * 0.1;
        }
        for w_val in &mut w.w_v {
            *w_val = rng() * 0.1;
        }
        for w_val in &mut w.w_o {
            *w_val = rng() * 0.1;
        }
        for w_val in &mut w.w_gate {
            *w_val = rng() * 0.1;
        }
        for w_val in &mut w.w_up {
            *w_val = rng() * 0.1;
        }
        for w_val in &mut w.w_down {
            *w_val = rng() * 0.1;
        }
        for w_val in &mut w.pos_emb {
            *w_val = rng() * 0.01;
        }
        w
    }

    /// The zero-alloc scratch path (`weaver_forward_into`) must produce the
    /// same output as the allocating path (`weaver_forward`). The batched
    /// matmul accumulates in the same order as the per-position matmul (both
    /// iterate `i` outer, SIMD-AXPY inner), so the results should be
    /// bit-identical or differ only by floating-point reassociation epsilon.
    #[test]
    fn g4_scratch_matches_allocating() {
        let cfg = test_config();
        let weights = nonzero_weights(&cfg);
        let input = test_input(&cfg, 16);

        // Allocating path.
        let out_alloc = weaver_forward(&weights, &input);

        // Scratch path.
        let mut scratch = WeaverScratch::new(&cfg);
        let (depth, k) = weaver_forward_into(&weights, &input, &mut scratch);

        assert_eq!(depth, out_alloc.depth);
        assert_eq!(k, out_alloc.k);

        // Compare residual, corrected_logits, corrected_probs.
        for di in 0..depth {
            for ki in 0..k {
                let alloc_r = out_alloc.weaver_residual[di][ki];
                let scratch_r = scratch.residual_flat[di * k + ki];
                assert!(
                    (alloc_r - scratch_r).abs() < 1e-4,
                    "residual mismatch at di={di} ki={ki}: alloc={alloc_r:.6} scratch={scratch_r:.6}"
                );

                let alloc_cl = out_alloc.corrected_logits[di][ki];
                let scratch_cl = scratch.corrected_logits_flat[di * k + ki];
                assert!(
                    (alloc_cl - scratch_cl).abs() < 1e-4,
                    "corrected_logits mismatch at di={di} ki={ki}: alloc={alloc_cl:.6} scratch={scratch_cl:.6}"
                );

                let alloc_cp = out_alloc.corrected_probs[di][ki];
                let scratch_cp = scratch.corrected_probs_flat[di * k + ki];
                assert!(
                    (alloc_cp - scratch_cp).abs() < 1e-4,
                    "corrected_probs mismatch at di={di} ki={ki}: alloc={alloc_cp:.6} scratch={scratch_cp:.6}"
                );
            }
        }
    }

    /// `correct_marginals_with_scratch` must produce the same marginals as
    /// `correct_marginals_inplace`. Both select top-K, run the forward pass,
    /// and write back — the scratch variant just avoids allocations.
    #[test]
    fn g4_correct_marginals_scratch_matches_allocating() {
        let cfg = corrector_config();
        let weights = nonzero_weights(&cfg);
        let corrector = WeaverCorrector::from_weights(weights);

        let h = cfg.hidden_dim;
        let depth = cfg.max_depth;
        let vocab = 32usize;

        // Build identical marginals for both paths.
        let mut marginals_alloc = vec![0.0f32; depth * vocab];
        let mut marginals_scratch = vec![0.0f32; depth * vocab];
        for di in 0..depth {
            marginals_alloc[di * vocab + di] = 0.4;
            marginals_alloc[di * vocab + (vocab - 1)] = 0.3;
            marginals_alloc[di * vocab + 10] = 0.2;
            marginals_alloc[di * vocab + 11] = 0.1;
        }
        marginals_scratch.copy_from_slice(&marginals_alloc);

        // Hidden states + embedding.
        let h_verifier: Vec<f32> = vec![0.5; h];
        let h_dflash_owned: Vec<Vec<f32>> = (0..depth)
            .map(|di| (0..h).map(|j| 0.3 + 0.01 * (di * h + j) as f32).collect())
            .collect();
        let h_dflash: Vec<&[f32]> = h_dflash_owned.iter().map(|v| v.as_slice()).collect();
        let embedding: Vec<f32> = vec![0.1; vocab * h];

        // Run both paths.
        corrector
            .correct_marginals_inplace(
                &mut marginals_alloc,
                &h_verifier,
                &h_dflash,
                &embedding,
                vocab,
            )
            .expect("allocating path should succeed");

        let mut scratch = WeaverScratch::new(&cfg);
        corrector
            .correct_marginals_with_scratch(
                &mut marginals_scratch,
                &h_verifier,
                &h_dflash,
                &embedding,
                vocab,
                &mut scratch,
            )
            .expect("scratch path should succeed");

        // Both should produce the same marginals (within float epsilon).
        for i in 0..marginals_alloc.len() {
            assert!(
                (marginals_alloc[i] - marginals_scratch[i]).abs() < 1e-4,
                "marginal mismatch at index {i}: alloc={:.6} scratch={:.6}",
                marginals_alloc[i],
                marginals_scratch[i]
            );
        }
    }

    /// G1 invariants must hold on the scratch path (sum to 1.0, no NaN/Inf).
    #[test]
    fn g4_scratch_g1_invariants_hold() {
        let cfg = test_config();
        let weights = nonzero_weights(&cfg);
        let input = test_input(&cfg, 16);

        let mut scratch = WeaverScratch::new(&cfg);
        let (depth, k) = weaver_forward_into(&weights, &input, &mut scratch);

        for di in 0..depth {
            let sum: f32 = (0..k)
                .map(|ki| scratch.corrected_probs_flat[di * k + ki])
                .sum();
            assert!((sum - 1.0).abs() < 1e-4, "probs at di={di} sum to {sum}");
            for ki in 0..k {
                let cp = scratch.corrected_probs_flat[di * k + ki];
                assert!(
                    cp.is_finite(),
                    "NaN/Inf in corrected_probs at di={di} ki={ki}"
                );
                let cl = scratch.corrected_logits_flat[di * k + ki];
                assert!(
                    cl.is_finite(),
                    "NaN/Inf in corrected_logits at di={di} ki={ki}"
                );
            }
        }
    }

    /// The parallel path (`weaver_forward_parallel`) must produce the same
    /// output as the sequential scratch path (`weaver_forward_into`). Both
    /// compute the same function; only the execution order differs (rayon
    /// parallelizes across positions). Floating-point results may differ
    /// slightly due to non-associativity, but should match within 1e-4.
    #[test]
    fn g4_parallel_matches_sequential() {
        let cfg = test_config();
        let weights = nonzero_weights(&cfg);
        let input = test_input(&cfg, 16);

        // Sequential path.
        let mut scratch_seq = WeaverScratch::new(&cfg);
        let (depth_seq, k_seq) = weaver_forward_into(&weights, &input, &mut scratch_seq);

        // Parallel path.
        let mut scratch_par = WeaverScratch::new(&cfg);
        let (depth_par, k_par) = weaver_forward_parallel(&weights, &input, &mut scratch_par);

        assert_eq!(depth_seq, depth_par);
        assert_eq!(k_seq, k_par);

        for di in 0..depth_seq {
            for ki in 0..k_seq {
                let idx = di * k_seq + ki;
                let diff_r =
                    (scratch_seq.residual_flat[idx] - scratch_par.residual_flat[idx]).abs();
                assert!(diff_r < 1e-4, "residual mismatch di={di} ki={ki}: {diff_r}");

                let diff_cl = (scratch_seq.corrected_logits_flat[idx]
                    - scratch_par.corrected_logits_flat[idx])
                    .abs();
                assert!(
                    diff_cl < 1e-4,
                    "corrected_logits mismatch di={di} ki={ki}: {diff_cl}"
                );

                let diff_cp = (scratch_seq.corrected_probs_flat[idx]
                    - scratch_par.corrected_probs_flat[idx])
                    .abs();
                assert!(
                    diff_cp < 1e-4,
                    "corrected_probs mismatch di={di} ki={ki}: {diff_cp}"
                );
            }
        }
    }

    // ── Issue 136: f16 weight path tests ──

    /// f16 forward with zero weights must produce zero residual (no-harm).
    /// Mirrors `g1_zero_weights_produce_zero_residual` for the f32 path.
    #[test]
    fn f16_zero_weights_produce_zero_residual() {
        let cfg = test_config();
        let weights_f32 = WeaverWeights::zeros(cfg.clone());
        let weights_f16 = WeaverWeightsF16::from_f32(&weights_f32);
        let input = test_input(&cfg, 16);

        let mut scratch = WeaverScratch::new(&cfg);
        let (depth, k) = weaver_forward_parallel_f16(&weights_f16, &input, &mut scratch);

        assert_eq!(depth, cfg.max_depth);
        assert_eq!(k, cfg.k_candidates);

        // Zero weights → zero residual.
        for di in 0..depth {
            for ki in 0..k {
                let idx = di * k + ki;
                assert!(
                    scratch.residual_flat[idx].abs() < 1e-6,
                    "f16 zero-weight residual should be ~0 at di={di} ki={ki}, got {}",
                    scratch.residual_flat[idx]
                );
            }
        }
    }

    /// f16 forward must produce valid probabilities (sum to 1, no NaN/Inf).
    #[test]
    fn f16_corrected_probs_sum_to_one_no_nan() {
        let cfg = test_config();
        let weights_f32 = nonzero_weights(&cfg);
        let weights_f16 = WeaverWeightsF16::from_f32(&weights_f32);
        let input = test_input(&cfg, 16);

        let mut scratch = WeaverScratch::new(&cfg);
        weaver_forward_parallel_f16(&weights_f16, &input, &mut scratch);

        let depth = cfg.max_depth;
        let k = cfg.k_candidates;
        for di in 0..depth {
            let row = &scratch.corrected_probs_flat[di * k..(di + 1) * k];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "f16 probs sum at di={di}: {sum}");
            for &p in row {
                assert!(p.is_finite(), "f16 prob is not finite at di={di}");
                assert!(p >= 0.0, "f16 prob is negative at di={di}: {p}");
            }
        }
    }

    /// f16 forward should produce results close to the f32 parallel path.
    /// The f16 rounding introduces ≤0.5 ULP per weight; with small test
    /// weights (~0.1 magnitude), the corrected_probs should match within ~5%.
    #[test]
    fn f16_matches_f32_within_precision() {
        let cfg = test_config();
        let weights_f32 = nonzero_weights(&cfg);
        let weights_f16 = WeaverWeightsF16::from_f32(&weights_f32);
        let input = test_input(&cfg, 16);

        // f32 parallel path.
        let mut scratch_f32 = WeaverScratch::new(&cfg);
        weaver_forward_parallel(&weights_f32, &input, &mut scratch_f32);

        // f16 parallel path.
        let mut scratch_f16 = WeaverScratch::new(&cfg);
        weaver_forward_parallel_f16(&weights_f16, &input, &mut scratch_f16);

        let depth = cfg.max_depth;
        let k = cfg.k_candidates;
        for di in 0..depth {
            for ki in 0..k {
                let idx = di * k + ki;
                let p_f32 = scratch_f32.corrected_probs_flat[idx];
                let p_f16 = scratch_f16.corrected_probs_flat[idx];
                // f16 has ~3 decimal digits of precision. For probabilities
                // in [0,1], the relative error should be < 10%.
                let diff = (p_f32 - p_f16).abs();
                assert!(
                    diff < 0.1,
                    "f16 vs f32 prob mismatch di={di} ki={ki}: f32={p_f32:.6} f16={p_f16:.6} diff={diff:.6}"
                );
            }
        }
    }

    /// WeaverCorrectorF16 wrapper must produce the same output as the raw
    /// f16 forward function.
    #[test]
    fn f16_corrector_wrapper_matches_forward() {
        let cfg = test_config();
        let weights_f32 = nonzero_weights(&cfg);
        let corrector_f32 = WeaverCorrector::from_weights(weights_f32);
        let corrector_f16 = WeaverCorrectorF16::from_f32(&corrector_f32);
        let input = test_input(&cfg, 16);

        let mut scratch_fn = WeaverScratch::new(&cfg);
        weaver_forward_parallel_f16(corrector_f16.weights(), &input, &mut scratch_fn);

        let mut scratch_wrap = WeaverScratch::new(&cfg);
        corrector_f16.correct_parallel(&input, &mut scratch_wrap);

        let depth = cfg.max_depth;
        let k = cfg.k_candidates;
        for idx in 0..depth * k {
            assert_eq!(
                scratch_fn.corrected_probs_flat[idx], scratch_wrap.corrected_probs_flat[idx],
                "corrector wrapper mismatch at idx={idx}"
            );
        }
    }
}
