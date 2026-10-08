//! MLA (Multi-head Latent Attention) — DeepSeek-V2 §2.1 + Appendix C.
//!
//! Implements the MLA mechanism with decoupled RoPE from DeepSeek-V2, plus the
//! Kimi-K3 output-gate extension (`mla_use_output_gate`) and the latent
//! RMSNorms (`q_a_layernorm`, `kv_a_layernorm`). See Research 327 for the
//! mathematical distillation, Research 330 §2 for the actual-model-code
//! divergence analysis, and Proposal 032 for the architectural context.
//!
//! # The mechanism (single-token decode)
//!
//! Matches the actual `modeling_kimi_k3_linear.py` forward path (Research 330
//! §2). The latent RMSNorms were MISSING in the original Phase 2 (which followed
//! the paper distillation); they are now applied in-place before up-projection.
//!
//! ```text
//! 1. Down-projections + latent RMSNorm (actual model: q_a_proj → q_a_layernorm
//!    → q_b_proj; kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj):
//!    c_q_raw  = W_DQ  · h         (query latent compression)
//!    c_q      = RMSNorm(c_q_raw, q_a_norm_weight, eps)    ← NORMED
//!    c_kv_raw = W_DKV · h         (KV latent compression)
//!    c_kv     = RMSNorm(c_kv_raw, kv_a_norm_weight, eps)  ← NORMED before cache
//!
//! 2. Up-projections + decoupled RoPE split (query path, from normed c_q):
//!    q_c     = W_UQ  · c_q   (content query — NO RoPE)
//!    q_r_raw = W_QR  · c_q   (rope query — RoPE applied per-head sub-vector)
//!    for each head i: q_r[i] = RoPE(q_r_raw[i], pos)
//!    q_i = [q_c[i] ; q_r[i]]  (concatenate content + rope per head)
//!
//! 3. Up-projections (key/value path, from normed c_kv) + shared RoPE key:
//!    k_c = W_UK · c_kv       (content key — NO RoPE, per head)
//!    v_c = W_UV · c_kv       (value, per head)
//!    k_r = RoPE(W_KR · h, pos)  (shared rope key — SAME for all heads — CACHED)
//!    k_i = [k_c[i] ; k_r]    (concatenate per head)
//!
//! 4. Attention (scale = 1/sqrt(d_h + d_R^h)):
//!    o_i = Σ_j softmax(q_i · k_j / scale) · v_c[j][i]
//!
//! 5. Output gate + projection (Kimi-K3 extension — gate BEFORE o_proj):
//!    if use_output_gate: attn_out *= sigmoid(W_g · h)   (W_g ∈ R^{v_h·n_h × d})
//!    u = W_O · gated_attn_out
//!
//! When `use_nope = true` (Kimi-K3-0.40B), no RoPE is applied — the decoupled
//! "rope" components are concatenated without rotation. The model relies on
//! KDA layers + attn-res for position information.
//! ```
//!
//! # Weight layout note (loader concern, Phase 5)
//!
//! The actual model stores fused projections (`q_b_proj`, `kv_a_proj_with_mqa`,
//! `kv_b_proj`) as single tensors. This implementation keeps them as logical
//! components (`w_uq`+`w_qr`, `w_dkv`+`w_kr`, `w_uk`+`w_uv`) because the math is
//! identical — the fused projection is just a row-stack of the components.
//! The Phase 5 safetensors loader will de-fuse (split) the fused tensors into
//! these fields at load time.
//!
//! # KV cache contents
//!
//! Only TWO vectors cached per token:
//! - `c_kv ∈ R^{d_c}` — the compressed KV latent (NORMED via `kv_a_layernorm`)
//! - `k_r  ∈ R^{d_R^h}` — the shared decoupled RoPE key
//!
//! For Kimi-K3-0.40B: `(128 + 32) = 160` elements/token/layer vs `1024` for MHA.

use katgpt_core::simd::{simd_dot_f32, simd_matmul_rows};
#[cfg(feature = "mla_absorbed")]
use katgpt_core::simd::simd_transpose_matvec_into;
use katgpt_kv::shard_kv::rope::RopeFreqs;

// ─── Config ─────────────────────────────────────────────────────────────────

/// MLA configuration parameters.
///
/// Mirrors the Kimi-K3 / DeepSeek-V2 config fields. See Research 327 §4 for the
/// 0.40B-specific values.
#[derive(Clone, Debug)]
pub struct MlaConfig {
    /// KV latent compression dim (`d_c` = `kv_lora_rank`). Kimi-K3-0.40B: 128.
    pub kv_lora_rank: usize,
    /// Query latent compression dim (`d'_c` = `q_lora_rank`). Kimi-K3-0.40B: 256.
    pub q_lora_rank: usize,
    /// Content per-head dim (`d_h` = `qk_nope_head_dim`). Kimi-K3-0.40B: 64.
    pub qk_nope_head_dim: usize,
    /// RoPE per-head dim (`d_R^h` = `qk_rope_head_dim`). Kimi-K3-0.40B: 32.
    pub qk_rope_head_dim: usize,
    /// Value per-head dim (`v_h` = `v_head_dim`). Kimi-K3-0.40B: 64.
    pub v_head_dim: usize,
    /// Number of attention heads (`n_h`). Kimi-K3-0.40B: 8.
    pub n_heads: usize,
    /// Hidden dim (`d`). Kimi-K3-0.40B: 1024.
    pub hidden_size: usize,
    /// Kimi-K3 output gate (`mla_use_output_gate`). When true, output is gated
    /// by `sigmoid(W_g · h)`. Kimi-K3-0.40B: true.
    pub use_output_gate: bool,
    /// Kimi-K3 no-RoPE mode (`mla_use_nope`). When true, the decoupled RoPE
    /// components (`qk_rope_head_dim`) are concatenated WITHOUT rotation — the
    /// model uses no positional encoding in MLA. Kimi-K3-0.40B: true.
    pub use_nope: bool,
    /// RoPE base frequency. Actual `config.json`: **10,000** (corrected from the
    /// prior 1,000,000 assumption — Research 330 §7).
    pub rope_theta: f32,
    /// RMSNorm epsilon. Actual `config.json`: **1e-5** (corrected from the prior
    /// 1e-6 assumption — Research 330 §7). Used by `q_a_layernorm` and
    /// `kv_a_layernorm`.
    pub rms_norm_eps: f32,
}

impl MlaConfig {
    /// Kimi-K3-0.40B MLA configuration (text path, `kimi_linear` model type).
    ///
    /// Values match the actual `config.json` (Research 330 §7). `rope_theta`
    /// is 10,000 (not 1,000,000 as the paper distillation suggested).
    pub fn kimi_k3_0_40b() -> Self {
        Self {
            kv_lora_rank: 128,
            q_lora_rank: 256,
            qk_nope_head_dim: 64,
            qk_rope_head_dim: 32,
            v_head_dim: 64,
            n_heads: 8,
            hidden_size: 1024,
            use_output_gate: true,
            use_nope: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Per-head content dim for queries/keys (`d_h`).
    #[inline]
    pub fn d_h(&self) -> usize {
        self.qk_nope_head_dim
    }

    /// Per-head RoPE dim (`d_R^h`).
    #[inline]
    pub fn d_r(&self) -> usize {
        self.qk_rope_head_dim
    }

    /// Per-head query/key dim after concatenation: `d_h + d_R^h`.
    #[inline]
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Attention scale: `1/sqrt(d_h + d_R^h)`.
    #[inline]
    pub fn attn_scale(&self) -> f32 {
        1.0 / ((self.qk_head_dim() as f32).sqrt())
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

/// MLA layer weight matrices (row-major `Vec<f32>`).
///
/// Naming follows DeepSeek-V2 / Kimi-K3 conventions. See Research 327 §5 for the
/// safetensors tensor-name mapping (confirmed in Phase 5 loader).
///
/// All matrices are stored as `[out_dim][in_dim]` (row-major), matching the
/// `simd_matmul_rows(output, weight, input, rows=out, cols=in)` convention.
#[derive(Clone)]
pub struct MlaWeights {
    /// `W_DKV ∈ R^{d_c × d}` — KV down-projection.
    pub w_dkv: Vec<f32>,
    /// `W_DQ ∈ R^{d'_c × d}` — query down-projection.
    pub w_dq: Vec<f32>,
    /// `W_UQ ∈ R^{d_h·n_h × d'_c}` — content query up-projection.
    pub w_uq: Vec<f32>,
    /// `W_QR ∈ R^{d_R^h·n_h × d'_c}` — rope query up-projection.
    pub w_qr: Vec<f32>,
    /// `W_UK ∈ R^{d_h·n_h × d_c}` — content key up-projection.
    pub w_uk: Vec<f32>,
    /// `W_UV ∈ R^{v_h·n_h × d_c}` — value up-projection.
    pub w_uv: Vec<f32>,
    /// `W_KR ∈ R^{d_R^h × d}` — shared rope key projection.
    pub w_kr: Vec<f32>,
    /// `W_O ∈ R^{d × v_h·n_h}` — output projection.
    pub w_o: Vec<f32>,
    /// `q_a_norm_weight ∈ R^{d'_c}` — RMSNorm gamma between query down/up
    /// projections (actual model: `q_a_layernorm`). Applied to `c_q` before
    /// `W_UQ`/`W_QR`. Added per Research 330 §2 (was missing in original Phase 2).
    pub q_a_norm_weight: Vec<f32>,
    /// `kv_a_norm_weight ∈ R^{d_c}` — RMSNorm gamma on the KV latent before
    /// up-projection (actual model: `kv_a_layernorm`). Applied to `c_kv` before
    /// caching + `W_UK`/`W_UV`. Added per Research 330 §2.
    pub kv_a_norm_weight: Vec<f32>,
    /// `W_g ∈ R^{v_h·n_h × d}` — output gate projection (Kimi-K3 extension).
    /// Applied to `attn_out` BEFORE `o_proj` (not after). Shape verified against
    /// actual safetensors: `[512, 1024]` = `[v_h·n_h, d]`.
    /// Only present when `config.use_output_gate` is true.
    pub w_g: Option<Vec<f32>>,
}

impl MlaWeights {
    /// Construct random weights from a seeded RNG (for G1 testing).
    ///
    /// Uses a simple LCG to avoid pulling a new RNG crate dep. Weights are drawn
    /// from `N(0, 1/sqrt(in_dim))`-ish (scaled uniform in `[-1/sqrt(in), 1/sqrt(in)]`).
    pub fn random(config: &MlaConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let d = config.hidden_size;

        let w_dkv = random_matrix(&mut rng, config.kv_lora_rank, d);
        let w_dq = random_matrix(&mut rng, config.q_lora_rank, d);
        let w_uq = random_matrix(&mut rng, config.d_h() * config.n_heads, config.q_lora_rank);
        let w_qr = random_matrix(&mut rng, config.d_r() * config.n_heads, config.q_lora_rank);
        let w_uk = random_matrix(&mut rng, config.d_h() * config.n_heads, config.kv_lora_rank);
        let w_uv = random_matrix(
            &mut rng,
            config.v_head_dim * config.n_heads,
            config.kv_lora_rank,
        );
        let w_kr = random_matrix(&mut rng, config.d_r(), d);
        let w_o = random_matrix(&mut rng, d, config.v_head_dim * config.n_heads);
        // Norm weights: initialized near 1.0 (typical RMSNorm gamma init) with
        // small random perturbation [0.9, 1.1] to exercise the scaling path in
        // G1 tests. The actual trained weights will have drifted from 1.0.
        let q_a_norm_weight = random_norm_weights(&mut rng, config.q_lora_rank);
        let kv_a_norm_weight = random_norm_weights(&mut rng, config.kv_lora_rank);
        let w_g = if config.use_output_gate {
            // g_proj shape: [v_h*n_h, d] — gate applied to attn_out BEFORE o_proj
            Some(random_matrix(
                &mut rng,
                config.v_head_dim * config.n_heads,
                d,
            ))
        } else {
            None
        };

        Self {
            w_dkv,
            w_dq,
            w_uq,
            w_qr,
            w_uk,
            w_uv,
            w_kr,
            w_o,
            q_a_norm_weight,
            kv_a_norm_weight,
            w_g,
        }
    }
}

/// Simple seeded LCG RNG for test weight generation (avoids new dep).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    /// xorshift64* — fast, decent distribution for test fixtures.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform float in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Generate a `[rows][cols]` row-major weight matrix with Xavier-ish scaling.
fn random_matrix(rng: &mut SimpleRng, rows: usize, cols: usize) -> Vec<f32> {
    let scale = 1.0 / (cols as f32).sqrt();
    (0..rows * cols)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
        .collect()
}

/// Generate RMSNorm gamma weights near 1.0: values in `[0.9, 1.1]`.
///
/// RMSNorm gammas are typically initialized to 1.0 (identity scaling) and
/// drift slightly during training. This range exercises the scaling path
/// without producing degenerate near-zero values.
fn random_norm_weights(rng: &mut SimpleRng, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| 1.0 + (rng.next_f32() * 2.0 - 1.0) * 0.1)
        .collect()
}

// ─── KV cache ───────────────────────────────────────────────────────────────

/// MLA KV cache — stores the compressed latent + shared RoPE key per token.
///
/// Only `(d_c + d_R^h)` elements per token are cached (vs `2·n_h·d_h` for MHA).
/// For Kimi-K3-0.40B: 160 elements/token vs 1024 for equivalent MHA.
pub struct MlaKVCache {
    /// `[max_seq][d_c]` — compressed KV latent per token (`c_kv`).
    pub latent_kv: Vec<f32>,
    /// `[max_seq][d_R^h]` — shared decoupled RoPE key per token (`k_r`).
    pub rope_key: Vec<f32>,
    /// Current sequence length (number of tokens cached).
    pub seq_len: usize,
    d_c: usize,
    d_r: usize,
    max_seq: usize,
}

impl MlaKVCache {
    /// Create a new cache with capacity for `max_seq` tokens.
    pub fn new(config: &MlaConfig, max_seq: usize) -> Self {
        Self {
            latent_kv: vec![0.0; max_seq * config.kv_lora_rank],
            rope_key: vec![0.0; max_seq * config.qk_rope_head_dim],
            seq_len: 0,
            d_c: config.kv_lora_rank,
            d_r: config.qk_rope_head_dim,
            max_seq,
        }
    }

    /// Reset the cache to empty (seq_len = 0) without deallocating.
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Rewind the cache's logical length to `seq_len` without deallocating.
    /// Used by the Ouro looped forward (Plan 324) to prevent KV cache expansion
    /// across loop iterations: each iteration re-processes the same token
    /// position, so it should overwrite the entry at `pos` rather than append.
    /// Before iteration t (t >= 1), rewind to `pos` so the next `append()` writes
    /// at position `pos` (overwriting iteration 0's entry) + restores seq_len
    /// to `pos + 1`. The previous entries (0..pos) are preserved.
    ///
    /// # Panics
    /// Panics if `seq_len` exceeds the current length (can't grow via rewind).
    pub fn rewind_to(&mut self, seq_len: usize) {
        debug_assert!(
            seq_len <= self.seq_len,
            "rewind_to({seq_len}) > current seq_len {} — rewind can only shrink",
            self.seq_len
        );
        self.seq_len = seq_len;
    }

    /// Append a token's compressed KV latent + shared RoPE key.
    ///
    /// Returns the position index of the newly cached token.
    /// Panics if the cache is full.
    pub fn append(&mut self, c_kv: &[f32], k_r: &[f32]) -> usize {
        debug_assert_eq!(c_kv.len(), self.d_c, "c_kv dim mismatch");
        debug_assert_eq!(k_r.len(), self.d_r, "k_r dim mismatch");
        let pos = self.seq_len;
        assert!(pos < self.max_seq, "MLA KV cache full");
        self.latent_kv[pos * self.d_c..(pos + 1) * self.d_c].copy_from_slice(c_kv);
        self.rope_key[pos * self.d_r..(pos + 1) * self.d_r].copy_from_slice(k_r);
        self.seq_len += 1;
        pos
    }

    /// Get the compressed KV latent for token at position `pos`.
    #[inline]
    pub fn latent_kv_at(&self, pos: usize) -> &[f32] {
        &self.latent_kv[pos * self.d_c..(pos + 1) * self.d_c]
    }

    /// Get the shared RoPE key for token at position `pos`.
    #[inline]
    pub fn rope_key_at(&self, pos: usize) -> &[f32] {
        &self.rope_key[pos * self.d_r..(pos + 1) * self.d_r]
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────────

/// Scratch buffers for the MLA forward pass — pre-allocated to be alloc-free
/// on the hot path.
///
/// Sized for a single token's decode. Reuse across tokens via `clear()`/overwrite.
pub struct MlaForwardScratch {
    // Down-projection outputs
    pub(crate) c_kv: Vec<f32>, // [d_c]
    pub(crate) c_q: Vec<f32>,  // [d'_c]
    // Query up-projections
    pub(crate) q_c: Vec<f32>, // [d_h * n_h]
    pub(crate) q_r: Vec<f32>, // [d_R^h * n_h] — RoPE applied in-place
    // Key/value up-projections
    pub(crate) k_c: Vec<f32>, // [d_h * n_h]
    pub(crate) v_c: Vec<f32>, // [v_h * n_h]
    // Shared RoPE key
    pub(crate) k_r: Vec<f32>, // [d_R^h]
    // Per-head attention output
    pub(crate) attn_out: Vec<f32>, // [v_h * n_h]
    // Attention scores scratch
    pub(crate) scores: Vec<f32>, // [seq_len]
    // Output gate scratch
    pub(crate) gate_buf: Vec<f32>, // [d]
    // Output
    pub(crate) output: Vec<f32>, // [d]
    // Issue 926 absorbed-decode scratch (compiled only under `mla_absorbed`):
    // per-head absorbed queries `[d_c * n_h]` + one latent accumulator `[d_c]`.
    // Pre-allocated here so the absorbed hot path allocates nothing per step.
    #[cfg(feature = "mla_absorbed")]
    pub(crate) q_abs: Vec<f32>,   // [d_c * n_h]
    #[cfg(feature = "mla_absorbed")]
    pub(crate) lat_acc: Vec<f32>, // [d_c]
}

impl MlaForwardScratch {
    /// Allocate scratch for the given config + max attention length.
    pub fn new(config: &MlaConfig, max_seq: usize) -> Self {
        let n_h = config.n_heads;
        Self {
            c_kv: vec![0.0; config.kv_lora_rank],
            c_q: vec![0.0; config.q_lora_rank],
            q_c: vec![0.0; config.d_h() * n_h],
            q_r: vec![0.0; config.d_r() * n_h],
            k_c: vec![0.0; config.d_h() * n_h],
            v_c: vec![0.0; config.v_head_dim * n_h],
            k_r: vec![0.0; config.d_r()],
            attn_out: vec![0.0; config.v_head_dim * n_h],
            scores: vec![0.0; max_seq],
            gate_buf: vec![0.0; config.hidden_size],
            output: vec![0.0; config.hidden_size],
            #[cfg(feature = "mla_absorbed")]
            q_abs: vec![0.0; config.kv_lora_rank * n_h],
            #[cfg(feature = "mla_absorbed")]
            lat_acc: vec![0.0; config.kv_lora_rank],
        }
    }

    /// Read-only access to the content query projections `[d_h * n_heads]`.
    ///
    /// Valid after a forward call. Used by diagnostic benches that compute
    /// per-block attention scores for retrieval-quality analysis (e.g. NIAH).
    pub fn q_c_view(&self) -> &[f32] {
        &self.q_c
    }

    /// Read-only access to the RoPE query projections `[d_r * n_heads]`.
    ///
    /// Valid after a forward call. Paired with `q_c_view` for full query access.
    pub fn q_r_view(&self) -> &[f32] {
        &self.q_r
    }
}

/// Apply RoPE to the decoupled query/key sub-vectors.
///
/// MLA applies RoPE ONLY to the `d_R^h`-dim rope sub-vectors (not the content).
/// For the query, each head has its own `d_R^h`-dim rope sub-vector; for the
/// key, the `d_R^h`-dim rope key is shared across all heads.
///
/// Uses `RopeFreqs` from `katgpt-kv` (the cached sin/cos RoPE substrate).
fn apply_decoupled_rope(
    rope_freqs: &mut RopeFreqs,
    x: &mut [f32],
    d_r: usize,
    n_heads: usize,
    pos: usize,
) {
    // For queries: each head's d_r sub-vector gets RoPE rotated at `pos`.
    // For the shared key: n_heads=1 (single d_r vector, shared).
    for h in 0..n_heads {
        let start = h * d_r;
        rope_freqs.apply(&mut x[start..start + d_r], pos, false);
    }
}

/// RMSNorm with learnable gamma, applied in-place.
///
/// `x[i] = gamma[i] * x[i] / sqrt(mean(x²) + eps)`
///
/// Mirrors the actual model's `KimiRMSNorm` (Research 330 §2): a standard
/// RMSNorm with per-channel gamma (weight) and configurable epsilon. Used for
/// `q_a_layernorm` (on the query latent) and `kv_a_layernorm` (on the KV latent
/// before up-projection/caching).
#[inline]
pub(crate) fn rmsnorm_inplace(x: &mut [f32], gamma: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), gamma.len(), "rmsnorm_inplace: dim mismatch");
    let n = x.len();
    let sum_sq = simd_dot_f32(x, x, n);
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    // zip instead of double indexing: same left-to-right multiply order
    // (`x * inv_rms * gamma`) → bit-identical, but the two bounds checks per
    // element hoist out of the loop.
    for (xi, &g) in x.iter_mut().zip(gamma.iter()) {
        *xi = *xi * inv_rms * g;
    }
}

/// Shared decode prologue (Steps 1–2): down-projections + latent RMSNorms +
/// query up-projections + RoPE on the per-head rope queries.
///
/// Used VERBATIM by both decode paths — the reconstruct reference
/// ([`mla_forward_token`]) and the weight-absorbed variant
/// ([`mla_forward_token_absorbed`], Issue 926) — so the pair cannot drift on
/// their shared prefix. Moved out of the reference fn byte-for-byte; the
/// spec-match + grad-check suites pin the reference.
fn mla_decode_prologue(
    config: &MlaConfig,
    weights: &MlaWeights,
    scratch: &mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
    pos: usize,
) {
    let d = config.hidden_size;
    let d_c = config.kv_lora_rank;
    let d_qc = config.q_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let n_h = config.n_heads;
    debug_assert_eq!(h.len(), d, "hidden state dim mismatch");

    // c_kv_raw = W_DKV · h   [d_c]  (KV latent compression)
    simd_matmul_rows(&mut scratch.c_kv, &weights.w_dkv, h, d_c, d);
    // c_q_raw  = W_DQ  · h   [d'_c] (query latent compression)
    simd_matmul_rows(&mut scratch.c_q, &weights.w_dq, h, d_qc, d);

    // Apply latent RMSNorms (actual model: q_a_layernorm, kv_a_layernorm).
    // The normed c_kv is what gets cached + up-projected.
    rmsnorm_inplace(
        &mut scratch.c_q,
        &weights.q_a_norm_weight,
        config.rms_norm_eps,
    );
    rmsnorm_inplace(
        &mut scratch.c_kv,
        &weights.kv_a_norm_weight,
        config.rms_norm_eps,
    );

    // q_c = W_UQ · c_q   [d_h * n_h]  (content query — NO RoPE)
    simd_matmul_rows(
        &mut scratch.q_c,
        &weights.w_uq,
        &scratch.c_q,
        d_h * n_h,
        d_qc,
    );
    // q_r_raw = W_QR · c_q   [d_r * n_h]  (rope query — RoPE applied next, unless use_nope)
    simd_matmul_rows(
        &mut scratch.q_r,
        &weights.w_qr,
        &scratch.c_q,
        d_r * n_h,
        d_qc,
    );
    // Apply decoupled RoPE to each head's d_r sub-vector (unless use_nope).
    if !config.use_nope {
        apply_decoupled_rope(rope_freqs, &mut scratch.q_r, d_r, n_h, pos);
    }
}

/// Shared decode midfix (Steps 4–5): shared rope key projection + RoPE, then
/// caching of the normed latent + rope key. Used verbatim by both decode
/// paths (see [`mla_decode_prologue`]).
fn mla_decode_rope_key_and_cache(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
    pos: usize,
) {
    let d = config.hidden_size;
    let d_r = config.d_r();

    // k_r_raw = W_KR · h   [d_r]  (shared across all heads — NOT normed;
    // the rope key is the tail of kv_a_proj_with_mqa output, outside kv_a_norm)
    simd_matmul_rows(&mut scratch.k_r, &weights.w_kr, h, d_r, d);
    // Apply RoPE to the shared key (n_heads=1 — it's shared), unless use_nope.
    if !config.use_nope {
        apply_decoupled_rope(rope_freqs, &mut scratch.k_r, d_r, 1, pos);
    }

    // ── Step 5: Cache the normed latent + shared rope key ───────────────
    // c_kv is ALREADY normed (Step 1); the attention loop up-projects directly.
    cache.append(&scratch.c_kv, &scratch.k_r);
}

/// Shared decode epilogue (Steps 7–8): the Kimi-K3 output gate (applied to
/// `attn_out` BEFORE the output projection) + `W_O`. Used verbatim by both
/// decode paths (see [`mla_decode_prologue`]).
fn mla_decode_epilogue<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    scratch: &'s mut MlaForwardScratch,
    h: &[f32],
) -> &'s mut [f32] {
    let d = config.hidden_size;
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;
    let proj_size = v_h * n_h;

    // ── Step 7: Output gate (Kimi-K3 extension) — BEFORE o_proj ────────
    // The actual model applies the gate to attn_output (shape [v_h*n_h]) BEFORE
    // the output projection, not after. g_proj has shape [v_h*n_h, d].
    //   g = sigmoid(g_proj · h)   [v_h*n_h]
    //   attn_output *= g
    // Then o_proj maps the gated attn_output back to [d].
    if config.use_output_gate
        && let Some(ref w_g) = weights.w_g
    {
        // gate = sigmoid(g_proj · h)  [v_h*n_h]
        simd_matmul_rows(&mut scratch.gate_buf, w_g, h, proj_size, d);
        // zip over the two pre-sliced prefixes: same element order and the same
        // `1/(1+exp(-x))` expression per element → bit-identical, minus the two
        // bounds checks per element.
        for (o, &gb) in scratch.attn_out[..proj_size]
            .iter_mut()
            .zip(scratch.gate_buf[..proj_size].iter())
        {
            let g = 1.0 / (1.0 + (-gb).exp());
            *o *= g;
        }
    }

    // ── Step 8: Output projection ──────────────────────────────────────
    // u = W_O · gated_attn_out   [d]
    simd_matmul_rows(
        &mut scratch.output,
        &weights.w_o,
        &scratch.attn_out,
        d,
        proj_size,
    );

    &mut scratch.output[..d]
}

/// Single-token MLA forward pass (decode path) — the RECONSTRUCT reference.
///
/// Computes one step of MLA attention:
/// 1. Down/up-projections on the current hidden state `h`
/// 2. Applies decoupled RoPE to the rope query + shared rope key
/// 3. Caches the compressed KV latent + shared rope key
/// 4. Attends to all cached tokens (content dot product + rope dot product)
/// 5. Output projection (with optional output gate)
///
/// This is the per-token up-projection form: every cached token's
/// `k_c_j = W_UK·c_kv_j` / `v_c_j = W_UV·c_kv_j` is rebuilt at attention time
/// (O(seq)·(d_h+v_h)·d_c FLOPs per step). It stays compiled in every feature
/// posture as the spec-match reference — [`mla_forward_token_absorbed`] is the
/// algebraically-identical absorbed variant, [`mla_forward_token_dispatched`]
/// the runtime router.
///
/// # Arguments
/// - `config` — MLA dimensions
/// - `weights` — weight matrices
/// - `cache` — KV cache (appended to; `cache.seq_len` is the new token's position)
/// - `scratch` — pre-allocated scratch buffers (reused across calls)
/// - `rope_freqs` — RoPE frequency table (must match `config.qk_rope_head_dim`)
/// - `h` — input hidden state `[hidden_size]`
///
/// # Returns
/// A slice into `scratch.output` of length `hidden_size`.
pub fn mla_forward_token<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &'s mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
) -> &'s mut [f32] {
    let d_c = config.kv_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;

    // The position of the token being processed = current cache length.
    let pos = cache.seq_len;

    // ── Steps 1–2 (shared prologue) ────────────────────────────────
    mla_decode_prologue(config, weights, scratch, rope_freqs, h, pos);

    // ── Step 3: Key/value up-projections (reconstruct-only) ─────────────
    // These compute the current token's k_c/v_c (overwritten in the attention
    // loop below — kept for structural clarity).
    simd_matmul_rows(
        &mut scratch.k_c,
        &weights.w_uk,
        &scratch.c_kv,
        d_h * n_h,
        d_c,
    );
    simd_matmul_rows(
        &mut scratch.v_c,
        &weights.w_uv,
        &scratch.c_kv,
        v_h * n_h,
        d_c,
    );

    // ── Steps 4–5: shared rope key + cache append (midfix) ─────────────
    mla_decode_rope_key_and_cache(config, weights, cache, scratch, rope_freqs, h, pos);

    // ── Step 6: Attention (per head) ───────────────────────────────────────
    // For each head h:
    //   q_i = [q_c[h*d_h..(h+1)*d_h] ; q_r[h*d_r..(h+1)*d_r]]  (concatenated)
    //   For each cached token j:
    //     k_j_i = [k_c[j,h*d_h..] ; k_r[j]]  — BUT k_c is the CURRENT token's
    //     up-projection, not cached! We need to up-project ALL cached latents.
    //
    // IMPORTANT: in MLA decode, we do NOT cache k_c/v_c — we cache c_kv (the
    // latent) and up-project at attention time. So for each cached token j, we
    // must compute k_c_j = W_UK · c_kv_j and v_c_j = W_UV · c_kv_j.
    //
    // For Phase 2 (no weight absorption), we recompute per-token. This is O(seq)
    // up-projections per decode step — correct but not optimal. Phase 6 may
    // cache the up-projected k_c/v_c or use weight absorption.
    let scale = config.attn_scale();
    let seq = cache.seq_len;

    for head in 0..n_h {
        let q_c_h = &scratch.q_c[head * d_h..(head + 1) * d_h];
        let q_r_h = &scratch.q_r[head * d_r..(head + 1) * d_r];

        // Compute attention scores for this head against all cached tokens.
        let scores = &mut scratch.scores[..seq];

        // We need per-token k_c, which requires up-projecting each cached c_kv.
        // Reuse scratch.k_c as a single-token buffer (overwrite per token).
        let mut max_score = f32::NEG_INFINITY;
        for (j, score_slot) in scores.iter_mut().enumerate().take(seq) {
            let c_kv_j = cache.latent_kv_at(j);
            let k_r_j = cache.rope_key_at(j);

            // k_c_j = W_UK · c_kv_j  (this token's content key)
            simd_matmul_rows(
                &mut scratch.k_c[..d_h],
                &weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                c_kv_j,
                d_h,
                d_c,
            );
            let k_c_j_h = &scratch.k_c[..d_h];

            // Score = (q_c · k_c + q_r · k_r) / scale
            //   content dot product (no RoPE) + rope dot product (with RoPE)
            let content_dot = simd_dot_f32(q_c_h, k_c_j_h, d_h);
            let rope_dot = simd_dot_f32(q_r_h, k_r_j, d_r);
            let score = (content_dot + rope_dot) * scale;
            *score_slot = score;
            if score > max_score {
                max_score = score;
            }
        }

        // Softmax (numerically stable: subtract max).
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut().take(seq) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;

        // Weighted sum of values: o_h = Σ_j softmax_j · v_c_j_h
        // v_c_j_h = W_UV[head slice] · c_kv_j
        let o_h = &mut scratch.attn_out[head * v_h..(head + 1) * v_h];
        o_h[..v_h].fill(0.0);
        // We need a scratch for the per-token v_c_h.
        // Reuse the gate_buf region (size d) which is unused during the attention
        // phase (it's only written in Step 8 below, after attention completes).
        let v_c_j_h_scratch = &mut scratch.gate_buf[..v_h];
        for (j, &weight_unnorm) in scores.iter().enumerate().take(seq) {
            let c_kv_j = cache.latent_kv_at(j);
            let weight = weight_unnorm * inv_sum;
            // v_c_j_h = W_UV[head slice] · c_kv_j
            simd_matmul_rows(
                v_c_j_h_scratch,
                &weights.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
                c_kv_j,
                v_h,
                d_c,
            );
            // zip drops the `o_h[vi]` bounds check; same j-then-vi visit order,
            // so each `o_h` accumulator sees the same addend sequence.
            for (o, &vc) in o_h.iter_mut().zip(v_c_j_h_scratch[..v_h].iter()) {
                *o += weight * vc;
            }
        }
    }

    // ── Steps 7–8: shared output gate + projection (epilogue) ───────
    mla_decode_epilogue(config, weights, scratch, h)
}

/// Weight-absorbed single-token MLA decode (Issue 926 — DeepSeek-V2 §2.1
/// absorption; external evidence arXiv:2610.07940 §5.2).
///
/// Algebraically IDENTICAL to [`mla_forward_token`], reassociated to remove
/// the per-token up-projections. Per step per head:
///
/// 1. `q_abs_h = W_UK[h]ᵀ · q_c_h` — once per step (8,192 FLOPs at the kimi
///    geometry), so the content score reads the latent directly:
///    `q_c·(W_UK·c_kv_j) ≡ (W_UKᵀ·q_c)·c_kv_j`.
/// 2. Scores: `content = q_abs_h·c_kv_j` (d_c dot) + the UNCHANGED rope dot
///    `q_r_h·k_r_j`.
/// 3. Softmax: same form as the reference (subtract max, exp, sum).
/// 4. Values: the latent-space accumulator `s_h = Σ_j w_j·c_kv_j` is ONE
///    transpose matvec over the cached `[seq × d_c]` latent matrix
///    ([`simd_transpose_matvec_into`] — NEON/AVX2/wasm32 paths), then
///    `o_h = W_UV[h]·s_h` once per step. The softmax weights are applied to
///    `s_h` AFTER the accumulation (d_c multiplies) instead of per token.
///
/// At the kimi_k3_0_40b geometry (d_h=64, d_c=128, v_h=64, n_h=16, seq=4096)
/// this drops the attention step from ~1.09 GFLOPs of up-projection work to
/// ~0.017 GFLOPs (~60×) per decode step; the k_r rope dot, the Kimi output
/// gate, and the `use_nope` structure are untouched, and the 6.4× cache
/// footprint win is preserved (the rejected alternative — caching
/// up-projected k_c/v_c — grows the cache to ≈MHA size and is recorded in
/// Issue 926 so nobody reopens it).
///
/// The reconstruct path is NOT changed by this fn and remains the
/// spec-match reference; G1 = the absorbed/reconstruct pair within the
/// existing 1e-4 (1e-3 full-dims) bar (`tests/mla_absorbed_spec_match.rs`).
#[cfg(feature = "mla_absorbed")]
pub fn mla_forward_token_absorbed<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &'s mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
) -> &'s mut [f32] {
    let d_c = config.kv_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;

    // The position of the token being processed = current cache length.
    let pos = cache.seq_len;

    // ── Steps 1–2 (shared prologue) ────────────────────────────────
    mla_decode_prologue(config, weights, scratch, rope_freqs, h, pos);

    // ── Step 3: SKIPPED — the reconstruct path's k_c/v_c up-projections are
    // dead work there (k_c is overwritten before read, v_c never read in the
    // attention loop) and the absorbed path never materializes k_c/v_c at all.

    // ── Steps 4–5: shared rope key + cache append (midfix) ─────────────
    mla_decode_rope_key_and_cache(config, weights, cache, scratch, rope_freqs, h, pos);

    // ── Step 6: ABSORBED attention (per head) ────────────────────────────
    let scale = config.attn_scale();
    let seq = cache.seq_len;

    for head in 0..n_h {
        // q_abs_h = W_UK[h]ᵀ · q_c_h — once per step per head. The head slice
        // is row-major [d_h × d_c]; the transpose matvec zeroes `q_abs[..d_c]`
        // then accumulates the row-major matrix transposed against q_c_h.
        simd_transpose_matvec_into(
            &mut scratch.q_abs[..d_c],
            &weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
            &scratch.q_c[head * d_h..(head + 1) * d_h],
            d_h,
            d_c,
        );
        let q_abs_h = &scratch.q_abs[..d_c];
        let q_r_h = &scratch.q_r[head * d_r..(head + 1) * d_r];

        // Scores: content dot over the latent (d_c) + the unchanged rope dot.
        let scores = &mut scratch.scores[..seq];
        let mut max_score = f32::NEG_INFINITY;
        for (j, score_slot) in scores.iter_mut().enumerate().take(seq) {
            let c_kv_j = cache.latent_kv_at(j);
            let k_r_j = cache.rope_key_at(j);

            let content_dot = simd_dot_f32(q_abs_h, c_kv_j, d_c);
            let rope_dot = simd_dot_f32(q_r_h, k_r_j, d_r);
            let score = (content_dot + rope_dot) * scale;
            *score_slot = score;
            if score > max_score {
                max_score = score;
            }
        }

        // Softmax (numerically stable: subtract max) — same form as the
        // reference. Weights stay UNNORMALIZED here; the single inv_sum
        // scaling happens on the latent accumulator below.
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut().take(seq) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;

        // Values: s_h = Σ_j exp_j · c_kv_j = Latents[seq × d_c]ᵀ · exp — ONE
        // SIMD pass over the cached latent matrix (the cache IS the matrix,
        // row-major by construction), then the single W_UV up-projection.
        {
            let s_acc = &mut scratch.lat_acc[..d_c];
            simd_transpose_matvec_into(s_acc, &cache.latent_kv[..seq * d_c], scores, seq, d_c);
            for v in s_acc.iter_mut() {
                *v *= inv_sum;
            }
            // o_h = W_UV[h] · s_h
            let o_h = &mut scratch.attn_out[head * v_h..(head + 1) * v_h];
            simd_matmul_rows(
                o_h,
                &weights.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
                s_acc,
                v_h,
                d_c,
            );
        }
    }

    // ── Steps 7–8: shared output gate + projection (epilogue) ───────
    mla_decode_epilogue(config, weights, scratch, h)
}

/// Runtime decode entry for model runners (Issue 926).
///
/// Routes to the weight-absorbed path ([`mla_forward_token_absorbed`]) when
/// compiled with the `mla_absorbed` feature, unless `KATGPT_MLA_RECONSTRUCT=1`
/// forces the reconstruct reference (read once, first call — the kill-switch
/// posture of the feature-flag discipline). WITHOUT the feature this is a
/// transparent forward to [`mla_forward_token`]: default builds are behavior-
/// and byte-identical, and the training/backward paths call the reference
/// directly regardless.
pub fn mla_forward_token_dispatched<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &'s mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
) -> &'s mut [f32] {
    #[cfg(feature = "mla_absorbed")]
    {
        static RECONSTRUCT_FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let forced = *RECONSTRUCT_FORCED.get_or_init(|| {
            std::env::var("KATGPT_MLA_RECONSTRUCT").is_ok_and(|v| v == "1")
        });
        if !forced {
            return mla_forward_token_absorbed(config, weights, cache, scratch, rope_freqs, h);
        }
    }
    mla_forward_token(config, weights, cache, scratch, rope_freqs, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run MLA forward and return the output Vec.
    fn run_mla(config: &MlaConfig, weights: &MlaWeights, tokens: &[Vec<f32>]) -> Vec<f32> {
        let max_seq = tokens.len();
        let mut cache = MlaKVCache::new(config, max_seq);
        let mut scratch = MlaForwardScratch::new(config, max_seq);
        let mut rope_freqs = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
        let mut last_out = Vec::new();
        for h in tokens {
            let out = mla_forward_token(
                config,
                weights,
                &mut cache,
                &mut scratch,
                &mut rope_freqs,
                h,
            );
            last_out = out.to_vec();
        }
        last_out
    }

    #[test]
    fn smoke_single_token_zero_position() {
        // At position 0, RoPE is identity (θ=0 for all pairs). So the output
        // should match a reference that skips RoPE entirely.
        let config = small_config();
        let weights = MlaWeights::random(&config, 42);
        let h = vec![0.5f32; config.hidden_size];
        let out = run_mla(&config, &weights, std::slice::from_ref(&h));
        assert_eq!(out.len(), config.hidden_size);
        // Sanity: output should be finite (no NaN/Inf from bad math).
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn smoke_two_token_sequence() {
        // Two tokens: first at pos=0, second at pos=1 attends to both.
        let config = small_config();
        let weights = MlaWeights::random(&config, 7);
        let h0 = vec![0.3f32; config.hidden_size];
        let h1 = vec![0.7f32; config.hidden_size];
        let out = run_mla(&config, &weights, &[h0, h1]);
        assert_eq!(out.len(), config.hidden_size);
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn smoke_kimi_k3_0_40b_dims() {
        // Verify the full 0.40B config dimensions work end-to-end.
        let config = MlaConfig::kimi_k3_0_40b();
        let weights = MlaWeights::random(&config, 99);
        let h = vec![0.1f32; config.hidden_size];
        let out = run_mla(&config, &weights, &[h.clone(), h.clone(), h.clone()]);
        assert_eq!(out.len(), 1024);
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn cache_appends_correctly() {
        let config = small_config();
        let mut cache = MlaKVCache::new(&config, 8);
        assert_eq!(cache.seq_len, 0);
        let c_kv = vec![1.0; config.kv_lora_rank];
        let k_r = vec![2.0; config.qk_rope_head_dim];
        cache.append(&c_kv, &k_r);
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.latent_kv_at(0), &c_kv[..]);
        assert_eq!(cache.rope_key_at(0), &k_r[..]);
    }

    #[test]
    fn cache_reset_clears_seq_len() {
        let config = small_config();
        let mut cache = MlaKVCache::new(&config, 8);
        cache.append(
            &vec![1.0; config.kv_lora_rank],
            &vec![2.0; config.qk_rope_head_dim],
        );
        assert_eq!(cache.seq_len, 1);
        cache.reset();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn output_gate_changes_output() {
        // When use_output_gate=true, the output should differ from the
        // ungated path (the gate multiplies element-wise by sigmoid(W_g·h)).
        let mut config = small_config();
        let weights_gated = MlaWeights::random(&config, 11);
        config.use_output_gate = false;
        let weights_ungated = {
            let mut w = MlaWeights::random(&config, 11);
            w.w_g = None;
            w
        };
        config.use_output_gate = true;

        let h = vec![0.5f32; config.hidden_size];
        let out_gated = run_mla(&config, &weights_gated, std::slice::from_ref(&h));
        let out_ungated = run_mla(&config, &weights_ungated, std::slice::from_ref(&h));
        // Outputs should differ (gate != 1.0 everywhere with random weights).
        let any_diff = out_gated
            .iter()
            .zip(out_ungated.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "output gate had no effect");
    }

    fn small_config() -> MlaConfig {
        // A small config for fast tests: 2 heads, tiny dims.
        MlaConfig {
            kv_lora_rank: 8,
            q_lora_rank: 12,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            n_heads: 2,
            hidden_size: 16,
            use_output_gate: true,
            use_nope: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }
}
