//! Parallax attention: streaming covariance correction on top of tiled attention.
//!
//! Implements the Parallax formula (Research 135, extended for sigmoid in Research 140):
//! ```text
//! o_PLX = o_SA − gate_scale · Σ_KV · ρ     where ρ = W_R · x
//! ```
//!
//! - `o_SA`  = attention output under chosen activation (softmax or sigmoid)
//! - `Σ_KV`  = KV cross-covariance under attention weights (streaming accumulator)
//! - `ρ`     = learned probe from layer input via extra projection `W_R`
//!
//! ## Sigmoid extension (Research 140)
//!
//! The Parallax correction is kernel-agnostic: any normalized attention weights
//! `p(i,j) ≥ 0, Σ_j p(i,j) = 1` define a Nadaraya-Watson estimator whose local-linear
//! upgrade is `o_LL = o_NW − Σ_KV · ρ`. Normalized sigmoid attention uses kernel
//! `K(x,y) = σ(x·y·s)` where `s = 1/√d`, giving:
//! ```text
//! p_σ(i,j) = σ(q_i · k_j · s) / Σ_k σ(q_i · k_k · s)
//! ```
//! Advantages over softmax: no attention sinks, better numerical stability, no exp
//! overflow risk. The column-sum factorization still applies identically.
//!
//! ## Training-time caller requirement: W_R gradient clipping (Issue 002)
//!
//! The Parallax correction `o_PLX = o_SA − gate_scale · Σ_KV · ρ` (where
//! `ρ = W_R · x`) has a positive-feedback loop under naive SGD on W_R:
//! as `|ρ|` grows, the correction grows, the gradient w.r.t. W_R grows
//! (proportional to `Σ_KV · x`), W_R amplifies `ρ` further, etc. Once the
//! loop runs away, attention-weight normalization overflows and the
//! forward emits NaN.
//!
//! Re-investigation against the current forward path (Issue 002,
//! 2026-06-19; see `tests/parallax_sigmoid_stability_grad_clip.rs`):
//! - **Sigmoid activation is stable** for ≥500 FD-SGD steps at LR=1.0 from
//!   zero W_R init at the canonical test setup (n=64, d=8).
//! - **Softmax activation diverges to NaN around step 325–350** under the
//!   same setup. This is the opposite of the original Issue 002 analysis,
//!   which expected sigmoid to be the worse case (Research 140).
//! - **Caller mitigation** for either activation when training W_R: apply
//!   global L2 gradient clipping on the W_R gradient only
//!   (`‖∇_W_R‖₂ ≤ 1.0` per step). This bounds the feedback loop's
//!   per-step amplification without altering the W_Q/W_K/W_V trajectories.
//!   For inference (frozen W_R), no mitigation is needed — the forward
//!   path is finite for any finite `ρ`.
//!
//! ## Optimization: column-sum factorization
//!
//! The covariance factorizes via column sums of the attention weight matrix:
//! ```text
//! Σ_KV = Σ_j c_j · v_j ⊗ k_j^T    where c_j = Σ_i p(i,j)
//! ```
//! This reduces outer products from O(N²) to O(N), bringing CPU overhead
//! from ~50× down to ~2× over base SDPA.
//!
//! ## Sink-Aware Composition (Plan 289)
//!
//! When both `parallax_attn` and `sink_aware_attn` features are enabled, this
//! module also exposes [`tiled_attention_parallax_forward_sink_aware`] — a
//! single entry point composing the parallax forward with the dual-policy
//! NOP/Broadcast classifier from [`crate::data_probe`] (Plan 287, Research 258).
//!
//! - [`SinkAwarePolicy::Uniform`] short-circuits to vanilla
//!   [`tiled_attention_parallax_forward`] (zero-cost contract: ≤5% overhead,
//!   measured within noise across n ∈ {64, 128, 256}).
//! - [`SinkAwarePolicy::DualPolicy`] runs the retained-attention forward into
//!   a caller-owned `o_temp`, then applies the flat dual-policy gate to produce
//!   the final output. Caller owns scratch via [`SinkAwareParallaxScratch`] —
//!   one struct bundles `attn_matrix`, `o_temp`, classifier scratch, and an
//!   optional [`crate::data_probe::CachedSinkClassification`] for audit-cadence
//!   amortization (Plan 287 Issue 001 mitigation).
//!
//! Design rationale (see `.plans/289_sink_aware_forward_path_wiring.md` §Scope
//! decisions A1–A5): separate entry point (not a `ParallaxConfig` field) so
//! `Default::default()` stays feature-independent; optional out-param on the
//! forward for attention matrix retention (zero overhead when `None`);
//! out-of-place gate with caller-provided temp buffer (no raw-pointer in-place
//! variants needed in `data_probe`).
//!
//! Feature-gated behind `#[cfg(feature = "parallax_attn")]` for the vanilla
//! forward and `#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]`
//! for the sink-aware composition.

use crate::simd;

// ── Config ────────────────────────────────────────────────────────

/// Activation function for attention weight normalization.
///
/// Both produce normalized weights `p(i,j) ≥ 0, Σ_j p(i,j) = 1` that define
/// a Nadaraya-Watson kernel regression estimator. The Parallax local-linear
/// correction `o_LL = o_NW − Σ_KV · ρ` applies identically to both.
#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParallaxActivation {
    /// Standard softmax: `p(i,j) = exp(q_i · k_j · s) / Σ_k exp(q_i · k_k · s)`.
    /// Gaussian-like kernel with attention sinks.
    Softmax,
    /// Normalized sigmoid: `p(i,j) = σ(q_i · k_j · s) / Σ_k σ(q_i · k_k · s)`.
    /// No attention sinks, better numerical stability, no exp overflow.
    /// Higher COR capacity than softmax on real LM data (Plan 161 T3).
    /// Kernel: `K(x,y) = σ(x · y · s)`.
    #[default]
    Sigmoid,
}

/// Configuration for the Parallax attention correction.
#[derive(Debug, Clone)]
pub struct ParallaxConfig {
    /// Scaling factor for the covariance correction. Default 1.0; can be
    /// annealed during training (set to 0.0 to recover pure attention).
    pub gate_scale: f32,
    /// Whether `W_R` starts zeroed. When `true` and weights are zero, the
    /// module is a no-op and recovers exact base attention.
    pub zero_init: bool,
    /// Activation function for attention weight normalization.
    /// Default: Sigmoid (sink-free, higher COR capacity per Plan 161).
    /// Set to Softmax for backward-compatible attention sinks.
    pub activation: ParallaxActivation,
}

impl Default for ParallaxConfig {
    fn default() -> Self {
        Self {
            gate_scale: 1.0,
            zero_init: true,
            activation: ParallaxActivation::default(),
        }
    }
}

// ── Weight normalization ───────────────────────────────────────────

/// Normalize attention scores in-place to produce valid probability weights.
///
/// - `Softmax`: `p(j) = exp(s_j − max) / Σ exp(s_k − max)`
/// - `Sigmoid`: `p(j) = σ(s_j) / Σ σ(s_k)` where `σ(x) = 1/(1+exp(−x))`
///
/// Both produce `Σ_j p(j) = 1` and `p(j) ≥ 0`.
#[inline]
fn normalize_attention_weights(row: &mut [f32], activation: ParallaxActivation) {
    match activation {
        ParallaxActivation::Softmax => {
            let max_score = simd::simd_max_f32(row);
            simd::simd_add_scalar_inplace(row, -max_score);
            simd::simd_exp_inplace(row);
            let rowsum = simd::simd_sum_f32(row);
            simd::simd_scale_inplace(row, 1.0 / rowsum);
        }
        ParallaxActivation::Sigmoid => {
            // σ(x) = 1/(1+exp(−x)), then normalize by sum
            simd::simd_scale_inplace(row, -1.0);
            simd::simd_exp_inplace(row);
            simd::simd_add_scalar_inplace(row, 1.0);
            // Invert elementwise: row = 1/(1+exp(−x)) = σ(x)
            simd::simd_reciprocal_inplace(row);
            // Normalize so Σ_j p(j) = 1 (Nadaraya-Watson requirement)
            let rowsum = simd::simd_sum_f32(row);
            simd::simd_scale_inplace(row, 1.0 / rowsum);
        }
    }
}

// ── R Projection ──────────────────────────────────────────────────

/// Compute ρ = R_proj · x (matrix–vector product, row-major weight).
///
/// - `r_proj`: R projection weight matrix `[head_dim × head_dim]`, row-major
/// - `x`:      layer input `[head_dim]`
/// - `out`:    output buffer `[head_dim]`
///
/// Uses [`simd::simd_dot_f32`] for the row–vector dot products.
#[inline]
pub fn compute_rho(r_proj: &[f32], x: &[f32], out: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(r_proj.len(), d * d, "r_proj must be head_dim × head_dim");
    debug_assert_eq!(x.len(), d, "x must have length head_dim");

    simd::simd_matmul_rows(out, r_proj, x, d, d);
}

// ── Streaming Covariance Correction ───────────────────────────────

/// Compute the Parallax correction: `out = Σ_KV · ρ`.
///
/// - `sigma_kv`: softmax-weighted KV cross-covariance `[head_dim × head_dim]`, row-major
/// - `rho`:      R projection output `[head_dim]`
/// - `out`:      correction output `[head_dim]`
///
/// Uses [`simd::simd_dot_f32`] for each row of the matrix–vector product.
#[inline]
pub fn parallax_correction(sigma_kv: &[f32], rho: &[f32], out: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(
        sigma_kv.len(),
        d * d,
        "sigma_kv must be head_dim × head_dim"
    );
    debug_assert_eq!(rho.len(), d, "rho must have length head_dim");

    simd::simd_matvec(out, sigma_kv, rho, d, d);
}

// ── Fused tiled attention + Parallax ──────────────────────────────

/// Scratch buffer sizes for [`tiled_attention_parallax_forward`].
///
/// Pre-compute once and reuse across calls to avoid per-call allocation.
/// All buffers are flat `Vec<f32>` — clear + reuse across calls.
pub struct ParallaxScratch {
    /// ρ = W_R · x, length `head_dim`
    pub rho: Vec<f32>,
    /// Column sums c[j] = Σ_i softmax(i,j), length `seq_len`
    pub col_sums: Vec<f32>,
    /// Per-row score buffer, length `seq_len`
    pub scores: Vec<f32>,
    /// Σ_KV cross-covariance, length `head_dim * head_dim`
    pub sigma_kv: Vec<f32>,
    /// Scaled v row for outer product, length `head_dim`
    pub pv_buf: Vec<f32>,
    /// Correction output, length `head_dim`
    pub correction: Vec<f32>,
    /// Cached dimensions from the last `ensure_capacity` call. Fast-path returns
    /// immediately when both match, skipping 6 length comparisons + branches.
    cached_seq_len: usize,
    cached_head_dim: usize,
}

impl ParallaxScratch {
    /// Create scratch buffers sized for the given dimensions.
    pub fn new(seq_len: usize, head_dim: usize) -> Self {
        Self {
            rho: vec![0.0; head_dim],
            col_sums: vec![0.0; seq_len],
            scores: vec![0.0; seq_len],
            sigma_kv: vec![0.0; head_dim * head_dim],
            pv_buf: vec![0.0; head_dim],
            correction: vec![0.0; head_dim],
            cached_seq_len: seq_len,
            cached_head_dim: head_dim,
        }
    }

    /// Reset all buffers for reuse (clear without deallocating).
    pub fn reset(&mut self) {
        self.rho.fill(0.0);
        self.col_sums.fill(0.0);
        self.scores.fill(0.0);
        self.sigma_kv.fill(0.0);
        self.pv_buf.fill(0.0);
        self.correction.fill(0.0);
    }

    /// Resize buffers if dimensions changed (avoids reallocation when sizes match).
    pub fn ensure_capacity(&mut self, seq_len: usize, head_dim: usize) {
        // Fast path: most calls reuse the same dimensions. Two comparisons
        // replace six length reads + branches in the steady state.
        if self.cached_seq_len == seq_len && self.cached_head_dim == head_dim {
            return;
        }
        let d = head_dim;
        let d2 = d * d;
        let mut changed = false;
        if self.rho.len() != d {
            self.rho.resize(d, 0.0);
            changed = true;
        }
        if self.col_sums.len() < seq_len {
            self.col_sums.resize(seq_len, 0.0);
            changed = true;
        }
        if self.scores.len() < seq_len {
            self.scores.resize(seq_len, 0.0);
            changed = true;
        }
        if self.sigma_kv.len() != d2 {
            self.sigma_kv.resize(d2, 0.0);
            changed = true;
        }
        if self.pv_buf.len() != d {
            self.pv_buf.resize(d, 0.0);
            changed = true;
        }
        if self.correction.len() != d {
            self.correction.resize(d, 0.0);
            changed = true;
        }
        // Only zero-fill when dimensions changed; caller resets specific buffers as needed.
        // The hot path (tiled_attention_parallax_forward) writes to all buffers before reading.
        if changed {
            self.reset();
        }
        self.cached_seq_len = seq_len;
        self.cached_head_dim = head_dim;
    }
}

/// Tiled online-softmax flash attention with Parallax covariance correction.
///
/// Uses column-sum factorization to compute the KV cross-covariance:
/// ```text
/// Σ_KV = Σ_j c_j · v_j ⊗ k_j^T    where c_j = Σ_i softmax(i,j)
/// ```
/// This reduces outer products from O(N²) to O(N), bringing CPU decode
/// overhead from ~50× down to ~2× over base SDPA.
///
/// # Algorithm
/// 1. For each query row i: compute scores, softmax, accumulate output + column sums
/// 2. Compute Σ_KV from column sums using N outer products
/// 3. Compute correction = Σ_KV · ρ and subtract from output
///
/// # Arguments
/// * `q`, `k`, `v` — Q, K, V tensors `[seq_len × head_dim]`, row-major
/// * `output`       — output tensor `[seq_len × head_dim]`, pre-allocated
/// * `seq_len`      — sequence length N
/// * `head_dim`     — dimension per head D
/// * `scale`        — softmax temperature (typically 1/√D)
/// * `r`            — R projection weights `[head_dim × head_dim]`, row-major
/// * `x`            — layer input `[head_dim]`
/// * `parallax_config` — gate scale and init config
/// * `scratch`      — pre-allocated scratch buffers (pass `None` for one-shot allocation)
///
/// See [`tiled_attention_parallax_forward_retaining`] for a variant that optionally
/// writes the full `n×n` normalized attention matrix into a caller-owned buffer
/// (used by [`tiled_attention_parallax_forward_sink_aware`] for sink classification).
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_parallax_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    r: &[f32],
    x: &[f32],
    parallax_config: &ParallaxConfig,
    scratch: Option<&mut ParallaxScratch>,
) {
    tiled_attention_parallax_forward_retaining(
        q, k, v, output, seq_len, head_dim, scale, r, x, parallax_config, None, scratch,
    );
}

/// Same as [`tiled_attention_parallax_forward`] but optionally retains the full
/// `n×n` normalized attention matrix.
///
/// When `attn_matrix` is `Some(buf)`, `buf` must have length `seq_len * seq_len`
/// and will be filled row-major with the post-normalization attention weights:
/// `attn_matrix[i * seq_len + j] = p(i, j)`. This is required by sink-aware
/// composition ([`tiled_attention_parallax_forward_sink_aware`]) because the
/// classifier scans attention *columns* (per-position sink strength), while
/// the parallax forward computes attention *row-by-row* and discards.
///
/// When `attn_matrix` is `None`, behavior is bit-identical to
/// [`tiled_attention_parallax_forward`] — the per-row `copy_from_slice` is
/// skipped (single hoisted branch outside the inner `j` loop).
///
/// # Arguments
/// * `attn_matrix`  — `None` for vanilla behavior, or a caller-owned
///   `&mut [f32]` of length `seq_len * seq_len` to retain the attention map.
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_parallax_forward_retaining(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    r: &[f32],
    x: &[f32],
    parallax_config: &ParallaxConfig,
    mut attn_matrix: Option<&mut [f32]>,
    scratch: Option<&mut ParallaxScratch>,
) {
    let expected = seq_len * head_dim;
    debug_assert_eq!(q.len(), expected, "Q slice length mismatch");
    debug_assert_eq!(k.len(), expected, "K slice length mismatch");
    debug_assert_eq!(v.len(), expected, "V slice length mismatch");
    debug_assert_eq!(output.len(), expected, "output slice length mismatch");
    debug_assert_eq!(
        r.len(),
        head_dim * head_dim,
        "R must be head_dim × head_dim"
    );
    debug_assert_eq!(x.len(), head_dim, "x must have length head_dim");
    if let Some(am) = attn_matrix.as_deref() {
        debug_assert_eq!(
            am.len(),
            seq_len * seq_len,
            "attn_matrix must be (seq_len, seq_len) row-major when Some"
        );
    }

    if seq_len == 0 {
        return;
    }

    // Allocate scratch on demand if caller didn't provide one
    let mut local_scratch;
    let scratch = match scratch {
        Some(s) => {
            s.ensure_capacity(seq_len, head_dim);
            s
        }
        None => {
            local_scratch = ParallaxScratch::new(seq_len, head_dim);
            &mut local_scratch
        }
    };

    let d = head_dim;
    let n = seq_len;

    // Compute ρ = W_R · x (reuse scratch buffer)
    compute_rho(r, x, &mut scratch.rho);

    // If gate_scale is zero, or ρ is all zeros (zero_init with zeroed W_R),
    // plain softmax attention is sufficient.
    // Perf: skip O(d) linear scan when gate_scale is already zero.
    let rho_is_zero = if parallax_config.gate_scale == 0.0 {
        true // gate_scale=0 makes correction zero regardless of ρ
    } else {
        // SIMD single-pass: sum of |values| is 0 iff all are zero.
        // Faster than scalar early-exit scan for typical non-zero ρ
        // (which must scan all d elements anyway to confirm all-zero).
        crate::simd::simd_sum_abs_f32(&scratch.rho[..d]) == 0.0
    };
    if rho_is_zero {
        tiled_attention_core(
            q,
            k,
            v,
            output,
            seq_len,
            head_dim,
            scale,
            Some(&mut scratch.scores),
            parallax_config.activation,
            attn_matrix.as_deref_mut(),
        );
        return;
    }

    // Zero column sums before accumulation — scratch may be reused across calls
    scratch.col_sums[..n].fill(0.0);
    scratch.sigma_kv[..d * d].fill(0.0);

    // Phase 1: Compute o_SA and accumulate column sums in one pass
    for i in 0..n {
        let row_off = i * d;
        output[row_off..row_off + d].fill(0.0);

        // Compute scores for row i: scores[j] = q_i · k_j * scale
        for j in 0..n {
            let k_off = j * d;
            scratch.scores[j] =
                simd::simd_dot_f32(&q[row_off..row_off + d], &k[k_off..k_off + d], d) * scale;
        }

        // Normalize attention weights (softmax or sigmoid)
        let row = &mut scratch.scores[..n];
        normalize_attention_weights(row, parallax_config.activation);

        // Retain normalized attention row if caller asked for the full matrix.
        // Branch is hoisted: either every row copies, or no row does.
        if let Some(am) = attn_matrix.as_deref_mut() {
            am[i * n..(i + 1) * n].copy_from_slice(&scratch.scores[..n]);
        }

        // Accumulate output: o_i = Σ_j p_ij · v_j
        for j in 0..n {
            let p = scratch.scores[j];
            let v_off = j * d;
            simd::simd_fused_scale_acc(
                &mut output[row_off..row_off + d],
                &v[v_off..v_off + d],
                p,
                d,
            );
        }

        // Accumulate column sums: c[j] += softmax(i,j)
        // (already in scratch.scores after softmax)
        simd::simd_add_inplace(&mut scratch.col_sums[..n], &scratch.scores[..n]);
    }

    // Phase 2: Compute Σ_KV = Σ_j c_j · v_j ⊗ k_j^T
    // Only N outer products instead of N² — the key optimization.
    //
    // Folds the per-column attention weight `c_j` directly into the outer
    // product's broadcast multiplier (`simd_outer_product_acc_scaled`),
    // eliminating the `pv_buf` materialization that the unscaled variant
    // required (saves 2·n·d memory ops: n·d writes + n·d reads of `pv_buf`).
    for j in 0..n {
        let c_j = scratch.col_sums[j];
        if c_j == 0.0 {
            continue;
        }
        let v_off = j * d;
        let k_off = j * d;
        simd::simd_outer_product_acc_scaled(
            scratch.sigma_kv.as_mut(),
            c_j,
            &v[v_off..v_off + d],
            &k[k_off..k_off + d],
            d,
            d,
        );
    }

    // Phase 3: Compute correction = Σ_KV · ρ
    parallax_correction(&scratch.sigma_kv, &scratch.rho, &mut scratch.correction);

    // Phase 4: Apply correction — output[i] -= gate_scale * correction for all i
    // Pre-scale correction by -gs once, then SIMD-add to each output row.
    let gs = parallax_config.gate_scale;
    simd::simd_scale_inplace(&mut scratch.correction, -gs);
    for i in 0..n {
        let off = i * d;
        simd::simd_add_inplace(&mut output[off..off + d], &scratch.correction[..d]);
    }
}

// ── Sink-Aware composition (Plan 289) ─────────────────────────────
// Requires both `parallax_attn` and `sink_aware_attn` features. Composes the
// retained-attention forward with the flat-layout dual-policy gate so callers
// get a single entry point that (a) is zero-cost when policy = Uniform, and
// (b) handles all the buffer plumbing (attn matrix, temp output, classifier
// scratch, optional cache) behind one owned scratch struct.
//
// Design rationale: see `.plans/289_sink_aware_forward_path_wiring.md`
// §Scope decisions A1–A5. Summary: we add a *new entry point* rather than a
// field on `ParallaxConfig` (preserves `Default::default()` + avoids
// feature-gated fields); we use an *optional out-param* on the forward to
// retain the n×n attention matrix (zero overhead when None); we forward into
// a *caller-owned temp buffer* rather than adding `_inplace` gate variants
// (keeps the gate API honest about its read/write split).

/// Scratch buffers for [`tiled_attention_parallax_forward_sink_aware`].
///
/// Bundles everything the sink-aware composition needs beyond a vanilla
/// [`ParallaxScratch`]:
///
/// * `attn_matrix` — `n×n` row-major buffer to retain the post-normalization
///   attention map. Required by the classifier (it scans attention columns).
/// * `o_temp` — `n×d` buffer that receives the parallax forward output before
///   the gate consumes it. The flat gate API is out-of-place (`o: &[f32]`,
///   `out: &mut [f32]`) — this buffer is the `o` side.
/// * `classifier` — [`StableRankScratch`] for the power-iteration kernel.
/// * `cached` — when `Some`, the wrapper uses [`apply_dual_policy_gate_cached_flat`]
///   to amortize classifier cost across calls (Plan 287 Issue 001 mitigation).
///   `None` runs the classifier every call.
///
/// Callers typically construct once via [`SinkAwareParallaxScratch::new`] and
/// reuse across calls; [`SinkAwareParallaxScratch::ensure_capacity`] is a
/// no-op when dimensions match (mirrors [`ParallaxScratch::ensure_capacity`]).
#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]
pub struct SinkAwareParallaxScratch {
    /// `n×n` row-major attention matrix. Written by the retained forward.
    pub attn_matrix: Vec<f32>,
    /// `n×d` temporary output. Written by the forward, read by the gate.
    pub o_temp: Vec<f32>,
    /// Classifier scratch (power iteration + col sums).
    pub classifier: crate::data_probe::StableRankScratch,
    /// Optional audit-cadence cache. When `None`, classifier runs every call.
    pub cached: Option<crate::data_probe::CachedSinkClassification>,
    cached_seq_len: usize,
    cached_head_dim: usize,
}

#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]
impl SinkAwareParallaxScratch {
    /// Allocate for the given dimensions. The `cached` field starts as `None`
    /// (classifier runs every call); set it via [`Self::with_cache`] or by
    /// direct field assignment to enable audit-cadence amortization.
    pub fn new(seq_len: usize, head_dim: usize) -> Self {
        Self {
            attn_matrix: vec![0.0; seq_len * seq_len],
            o_temp: vec![0.0; seq_len * head_dim],
            classifier: crate::data_probe::StableRankScratch::new(head_dim),
            cached: None,
            cached_seq_len: seq_len,
            cached_head_dim: head_dim,
        }
    }

    /// Enable the audit-cadence cache with default config (cadence 16).
    /// Convenience for callers who want the cached path without constructing
    /// a [`crate::data_probe::CachedSinkClassification`] by hand.
    pub fn with_cache(mut self) -> Self {
        self.cached = Some(crate::data_probe::CachedSinkClassification::new());
        self
    }

    /// Resize buffers if dimensions changed. No-op when both match the last
    /// call — mirrors [`ParallaxScratch::ensure_capacity`]'s fast path.
    pub fn ensure_capacity(&mut self, seq_len: usize, head_dim: usize) {
        if self.cached_seq_len == seq_len && self.cached_head_dim == head_dim {
            return;
        }
        if self.attn_matrix.len() != seq_len * seq_len {
            self.attn_matrix.resize(seq_len * seq_len, 0.0);
        }
        if self.o_temp.len() != seq_len * head_dim {
            self.o_temp.resize(seq_len * head_dim, 0.0);
        }
        self.classifier.ensure_capacity_dn(head_dim, seq_len);
        self.cached_seq_len = seq_len;
        self.cached_head_dim = head_dim;
    }
}

/// Sink-aware composition of parallax forward + dual-policy gate (Plan 289).
///
/// Single entry point for callers who want per-head NOP/Broadcast gating
/// applied to the parallax forward output, without manually plumbing the
/// attention matrix, temporary output, classifier scratch, and optional
/// audit cache.
///
/// # Behavior
///
/// * [`SinkAwarePolicy::Uniform`] — calls vanilla
///   [`tiled_attention_parallax_forward`] directly into `output`. **Zero
///   overhead** vs the vanilla path: no attention matrix is retained, no
///   temporary buffer is touched, no classifier runs. Returns
///   [`SinkKind::None`].
/// * [`SinkAwarePolicy::DualPolicy(_)`] — runs the retained forward into
///   `sink_scratch.o_temp` while writing the full `n×n` attention matrix into
///   `sink_scratch.attn_matrix`, then applies the flat dual-policy gate
///   (`o_temp → output`). When `sink_scratch.cached` is `Some`, uses the
///   cached variant (amortizes classifier cost; Plan 287 Issue 001 mitigation).
///
/// # Arguments
///
/// * `q`, `k`, `v`, `output`, `seq_len`, `head_dim`, `scale`, `r`, `x`,
///   `parallax_config`, `scratch` — see [`tiled_attention_parallax_forward`].
/// * `policy`        — [`SinkAwarePolicy::Uniform`] for the zero-cost path;
///   [`SinkAwarePolicy::DualPolicy`] to invoke the classifier + gate.
/// * `gate_scale`    — pre-sigmoid logit for the NOP gate. `σ(gate_scale)` is
///   the scale applied to NOP-classified output rows. Pass `0.0` for
///   σ(0)=0.5 (half-suppression) or a large negative for near-zero.
/// * `sink_scratch`  — owns `attn_matrix`, `o_temp`, classifier scratch, and
///   optional cache. Construct once via [`SinkAwareParallaxScratch::new`] and
///   reuse across calls.
///
/// # Returns
///
/// The dominant sink's [`SinkKind`] (`None` for Uniform path, or the
/// classifier verdict for DualPolicy path).
///
/// # Feature gates
///
/// Requires both `parallax_attn` and `sink_aware_attn` features at the crate
/// level. The vanilla forward is always available with just `parallax_attn`.
#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_parallax_forward_sink_aware(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    r: &[f32],
    x: &[f32],
    parallax_config: &ParallaxConfig,
    policy: &crate::data_probe::SinkAwarePolicy,
    gate_scale: f32,
    sink_scratch: &mut SinkAwareParallaxScratch,
    scratch: Option<&mut ParallaxScratch>,
) -> crate::data_probe::SinkKind {
    use crate::data_probe::SinkAwarePolicy;

    // Uniform short-circuit: zero-cost contract. Vanilla forward writes
    // directly into `output`; we touch nothing in `sink_scratch`.
    if matches!(policy, SinkAwarePolicy::Uniform) {
        tiled_attention_parallax_forward(
            q, k, v, output, seq_len, head_dim, scale, r, x, parallax_config, scratch,
        );
        return crate::data_probe::SinkKind::None;
    }

    // DualPolicy path: ensure scratch sized, forward into temp with attn
    // matrix retained, then apply the flat gate (cached or uncached).
    let n = seq_len;
    let d = head_dim;
    sink_scratch.ensure_capacity(n, d);

    tiled_attention_parallax_forward_retaining(
        q,
        k,
        v,
        &mut sink_scratch.o_temp,
        seq_len,
        head_dim,
        scale,
        r,
        x,
        parallax_config,
        Some(&mut sink_scratch.attn_matrix),
        scratch,
    );

    // Source of truth for classifier thresholds is the `policy` argument.
    // When the cache is enabled, sync its cfg from the policy so the cached
    // variant (which reads `cached.cfg` internally) stays consistent. Cost is
    // one 5-f32 struct copy per call — negligible vs the n×n classifier.
    let policy_cfg = match policy {
        SinkAwarePolicy::DualPolicy(c) => *c,
        // Unreachable: Uniform short-circuited above.
        SinkAwarePolicy::Uniform => unreachable!("Uniform handled by short-circuit"),
    };

    // Borrow classifier + cached disjointly from attn_matrix / o_temp. Split
    // the mutable borrow on `sink_scratch` by destructuring field-by-field:
    // Rust's borrow checker treats disjoint field borrows as independent.
    let SinkAwareParallaxScratch {
        attn_matrix,
        o_temp,
        classifier,
        cached,
        cached_seq_len: _,
        cached_head_dim: _,
    } = sink_scratch;

    match cached {
        Some(c) => {
            c.cfg = policy_cfg;
            crate::data_probe::apply_dual_policy_gate_cached_flat(
                attn_matrix,
                v,
                o_temp,
                n,
                d,
                gate_scale,
                classifier,
                c,
                output,
            )
        }
        None => {
            crate::data_probe::apply_dual_policy_gate_flat(
                attn_matrix,
                v,
                o_temp,
                n,
                d,
                policy,
                gate_scale,
                classifier,
                output,
            )
        }
    }
}

// ── Core attention (no feature-flag dependency) ───────────────────

/// Core attention, used when Parallax correction is not needed.
///
/// Accepts an optional pre-allocated `scores` scratch buffer (length >= seq_len)
/// to avoid per-call heap allocation. When `None`, allocates on demand.
#[allow(clippy::too_many_arguments)]
#[inline]
fn tiled_attention_core(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    scores: Option<&mut [f32]>,
    activation: ParallaxActivation,
    mut attn_matrix: Option<&mut [f32]>,
) {
    let d = head_dim;
    let n = seq_len;

    // Use caller-provided scratch or allocate on demand. No `fill(0.0)` needed —
    // the score loop below writes every `scores[0..n]` element before any read.
    let mut local_scores;
    let scores: &mut [f32] = match scores {
        Some(s) if s.len() >= n => s,
        _ => {
            local_scores = vec![0.0f32; n];
            &mut local_scores
        }
    };

    for i in 0..n {
        let q_off = i * d;
        let out_off = i * d;
        output[out_off..out_off + d].fill(0.0);

        // Compute scores for row i
        for (j, score_slot) in scores.iter_mut().enumerate().take(n) {
            let k_off = j * d;
            *score_slot = simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d) * scale;
        }

        // Normalize attention weights (softmax or sigmoid)
        let row = &mut scores[..n];
        normalize_attention_weights(row, activation);

        // Retain normalized attention row if caller asked for the full matrix.
        if let Some(am) = attn_matrix.as_deref_mut() {
            am[i * n..(i + 1) * n].copy_from_slice(&scores[..n]);
        }

        // Accumulate output: o_i = Σ_j p_ij · v_j
        for (j, &p) in scores.iter().enumerate().take(n) {
            let v_off = j * d;
            simd::simd_fused_scale_acc(
                &mut output[out_off..out_off + d],
                &v[v_off..v_off + d],
                p,
                d,
            );
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// With R_proj = zero matrix, rho should be all zeros.
    #[test]
    fn test_rho_zero_init() {
        let d = 8;
        let r_proj = vec![0.0f32; d * d];
        let x: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let mut rho = vec![0.0f32; d];

        compute_rho(&r_proj, &x, &mut rho);

        for (i, &v) in rho.iter().enumerate() {
            assert!(
                v == 0.0,
                "rho[{}] should be 0.0 with zero R_proj, got {}",
                i,
                v
            );
        }
    }

    /// With identity sigma_kv, correction should equal rho.
    #[test]
    fn test_correction_identity() {
        let d = 8;
        let mut sigma_kv = vec![0.0f32; d * d];
        // Identity matrix
        for i in 0..d {
            sigma_kv[i * d + i] = 1.0;
        }
        let rho: Vec<f32> = (1..=d).map(|i| i as f32 * 0.5).collect();
        let mut correction = vec![0.0f32; d];

        parallax_correction(&sigma_kv, &rho, &mut correction);

        for (i, (&c, &r)) in correction.iter().zip(rho.iter()).enumerate() {
            let expected = r;
            assert!(
                (c - expected).abs() < 1e-5,
                "correction[{}] should be {} (identity sigma), got {}",
                i,
                expected,
                c
            );
        }
    }

    /// With gate_scale=0, the output should equal standard softmax attention.
    #[test]
    fn test_parallax_recovers_softmax_gate_zero() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        // R projection — non-zero, but gate_scale=0 should cancel it
        let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: false,
            activation: ParallaxActivation::Softmax,
        };

        let mut output_parallax = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output_parallax,
            n,
            d,
            scale,
            &r,
            &x,
            &config,
            None,
        );

        // Compute reference: standard softmax attention
        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(
            &q,
            &k,
            &v,
            &mut output_ref,
            n,
            d,
            scale,
            None,
            ParallaxActivation::Softmax,
            None,
        );

        for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "output[{}]: parallax ({}) should match softmax ({}) with gate_scale=0",
                i,
                a,
                b
            );
        }
    }

    /// With zero R_proj, the output should equal standard softmax attention
    /// regardless of gate_scale (since rho = 0 implies correction = 0).
    #[test]
    fn test_parallax_recovers_softmax_zero_r() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        // Zero R projection weights
        let r = vec![0.0f32; d * d];
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: true,
            activation: ParallaxActivation::Softmax,
        };

        let mut output_parallax = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output_parallax,
            n,
            d,
            scale,
            &r,
            &x,
            &config,
            None,
        );

        // Compute reference: standard softmax attention
        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(
            &q,
            &k,
            &v,
            &mut output_ref,
            n,
            d,
            scale,
            None,
            ParallaxActivation::Softmax,
            None,
        );

        for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "output[{}]: parallax ({}) should match softmax ({}) with zero R",
                i,
                a,
                b
            );
        }
    }

    /// Verify that compute_rho produces correct matrix-vector product.
    #[test]
    fn test_compute_rho_correct() {
        let d = 4;
        // R = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
        let mut r_proj = vec![0.0f32; d * d];
        for i in 0..d {
            r_proj[i * d + i] = (i + 1) as f32;
        }
        let x = vec![1.0f32; d];
        let mut rho = vec![0.0f32; d];

        compute_rho(&r_proj, &x, &mut rho);

        let expected = [1.0f32, 2.0, 3.0, 4.0];
        for (i, (&r, &e)) in rho.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "rho[{}] should be {}, got {}",
                i,
                e,
                r
            );
        }
    }

    /// Verify that parallax_correction with a known sigma produces the right result.
    #[test]
    fn test_correction_known_sigma() {
        let d = 3;
        // sigma_kv = [[2, 0, 0], [0, 2, 0], [0, 0, 2]] (2 * identity)
        let mut sigma_kv = vec![0.0f32; d * d];
        for i in 0..d {
            sigma_kv[i * d + i] = 2.0;
        }
        let rho = vec![1.0f32, 2.0, 3.0];
        let mut correction = vec![0.0f32; d];

        parallax_correction(&sigma_kv, &rho, &mut correction);

        let expected = [2.0f32, 4.0, 6.0];
        for (i, (&c, &e)) in correction.iter().zip(expected.iter()).enumerate() {
            assert!(
                (c - e).abs() < 1e-5,
                "correction[{}] should be {}, got {}",
                i,
                e,
                c
            );
        }
    }

    // ── Sigmoid-specific tests ──────────────────────────────────────

    /// With gate_scale=0 and sigmoid activation, output should equal pure sigmoid attention.
    #[test]
    fn test_parallax_sigmoid_recovers_base() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };

        let mut output_parallax = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output_parallax,
            n,
            d,
            scale,
            &r,
            &x,
            &config,
            None,
        );

        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(
            &q,
            &k,
            &v,
            &mut output_ref,
            n,
            d,
            scale,
            None,
            ParallaxActivation::Sigmoid,
            None,
        );

        for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "output[{}]: sigmoid parallax ({}) should match base sigmoid ({}) with gate_scale=0",
                i,
                a,
                b
            );
        }
    }

    /// Sigmoid attention weights should be non-negative and sum to 1 per row.
    #[test]
    fn test_sigmoid_weights_normalized() {
        let d = 4;
        let n = 5;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.53).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.71).sin()).collect();

        // Run with gate_scale=0 so we get pure sigmoid attention
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let config = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: true,
            activation: ParallaxActivation::Sigmoid,
        };

        let mut output = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output,
            n,
            d,
            scale,
            &r,
            &x,
            &config,
            None,
        );

        // Output should be finite (no NaN/Inf from numerical issues)
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] should be finite, got {}", i, v);
        }
    }

    /// Sigmoid and softmax should produce different outputs (different kernels).
    #[test]
    fn test_sigmoid_differs_from_softmax() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let mut out_sm = vec![0.0f32; n * d];
        let mut out_sig = vec![0.0f32; n * d];

        let config_sm = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: true,
            activation: ParallaxActivation::Softmax,
        };
        let config_sig = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: true,
            activation: ParallaxActivation::Sigmoid,
        };

        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_sm,
            n,
            d,
            scale,
            &r,
            &x,
            &config_sm,
            None,
        );
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_sig,
            n,
            d,
            scale,
            &r,
            &x,
            &config_sig,
            None,
        );

        let any_differs = out_sm
            .iter()
            .zip(out_sig.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(
            any_differs,
            "sigmoid and softmax should produce different outputs"
        );
    }

    /// With non-zero R projection, sigmoid Parallax should modify the output.
    #[test]
    fn test_sigmoid_parallax_correction_applied() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config_no_corr = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };
        let config_with_corr = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };

        let mut out_no = vec![0.0f32; n * d];
        let mut out_yes = vec![0.0f32; n * d];

        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_no,
            n,
            d,
            scale,
            &r,
            &x,
            &config_no_corr,
            None,
        );
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_yes,
            n,
            d,
            scale,
            &r,
            &x,
            &config_with_corr,
            None,
        );

        let any_differs = out_no
            .iter()
            .zip(out_yes.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(
            any_differs,
            "sigmoid parallax correction should modify output vs base sigmoid"
        );
    }

    // ── Plan 289 tests ──────────────────────────────────────────────
    // Covers: retained-attention forward correctness (always-on, parallax_attn
    // only), and sink-aware composition parity (Uniform + DualPolicy) + G2.
    // The latency G3 microbench lives in benches/ (T3.5).

    /// Deterministic LCG for reproducible test inputs. Cheap, no deps.
    /// `pub(super)` so the sibling `sink_aware_tests` module can reuse it.
    pub(super) fn lcg_fill(seed: u64, buf: &mut [f32]) {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        for x in buf.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (((s >> 33) as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
    }

    /// Reference row-by-row attention matrix computation. Independent of the
    /// forward's internal accumulation — recomputes scores + normalization.
    fn reference_attn_matrix(
        q: &[f32],
        k: &[f32],
        n: usize,
        d: usize,
        scale: f32,
        activation: ParallaxActivation,
        am: &mut [f32],
    ) {
        let mut row = vec![0.0f32; n];
        for i in 0..n {
            let q_off = i * d;
            for (j, row_slot) in row.iter_mut().enumerate().take(n) {
                let k_off = j * d;
                *row_slot = crate::simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d) * scale;
            }
            normalize_attention_weights(&mut row, activation);
            am[i * n..(i + 1) * n].copy_from_slice(&row);
        }
    }

    /// Helper: build (q, k, v) where attention concentrates strongly on position
    /// `sink_pos` (mean column strength ≈ 0.94, well above τ_sink=0.5) but
    /// `v[sink_pos]` is optionally zero (NOP) or normal content (Broadcast).
    ///
    /// Construction:
    /// - q[i] = [i*0.5, 0, ...] for i in 0..n (varies across queries).
    /// - k[sink] = [+10, 0, ...] → σ(q·k) saturates to ≈1 for i≥1.
    /// - k[j≠sink] = [-10, 0, ...] → σ(q·k) ≈ 0 for i≥1.
    /// - v[j] = ones (or zeros at sink for NOP case).
    ///
    /// Result: column `sink_pos` receives mean strength ≈ 0.94 across rows,
    /// dominating all other columns. The AV update is rank-1 (output rows
    /// proportional to v[sink]) when v[sink] is non-zero.
    pub(super) fn build_sink_case(n: usize, d: usize, sink_pos: usize, sink_v_zero: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        for i in 0..n {
            q[i * d] = (i as f32) * 0.5;
        }
        // Sink column attracts strongly.
        k[sink_pos * d] = 10.0;
        // Other columns strongly repel.
        for j in 0..n {
            if j != sink_pos {
                k[j * d] = -10.0;
            }
            for c in 0..d {
                v[j * d + c] = if j == sink_pos && sink_v_zero { 0.0 } else { 1.0 };
            }
        }
        (q, k, v)
    }

    /// T1.3 — retained attention matrix matches row-by-row reference (Sigmoid).
    #[test]
    fn plan289_retained_attn_matches_per_row_sigmoid() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let v = vec![0.0f32; n * d];
        let mut output = vec![0.0f32; n * d];
        let mut am_actual = vec![0.0f32; n * n];
        let mut am_expected = vec![0.0f32; n * n];
        lcg_fill(0xC0DE, &mut q);
        lcg_fill(0xFEED, &mut k);

        let cfg = ParallaxConfig { gate_scale: 0.0, zero_init: true, activation: ParallaxActivation::Sigmoid };
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        tiled_attention_parallax_forward_retaining(
            &q, &k, &v, &mut output, n, d, scale, &r, &x, &cfg, Some(&mut am_actual), None,
        );
        reference_attn_matrix(&q, &k, n, d, scale, ParallaxActivation::Sigmoid, &mut am_expected);

        for i in 0..(n * n) {
            assert_eq!(am_actual[i], am_expected[i], "am[{}] mismatch (Sigmoid)", i);
        }
    }

    /// T1.3 — retained attention matrix matches row-by-row reference (Softmax).
    #[test]
    fn plan289_retained_attn_matches_per_row_softmax() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let v = vec![0.0f32; n * d];
        let mut output = vec![0.0f32; n * d];
        let mut am_actual = vec![0.0f32; n * n];
        let mut am_expected = vec![0.0f32; n * n];
        lcg_fill(0x1234, &mut q);
        lcg_fill(0x5678, &mut k);

        let cfg = ParallaxConfig { gate_scale: 0.0, zero_init: true, activation: ParallaxActivation::Softmax };
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        tiled_attention_parallax_forward_retaining(
            &q, &k, &v, &mut output, n, d, scale, &r, &x, &cfg, Some(&mut am_actual), None,
        );
        reference_attn_matrix(&q, &k, n, d, scale, ParallaxActivation::Softmax, &mut am_expected);

        for i in 0..(n * n) {
            assert_eq!(am_actual[i], am_expected[i], "am[{}] mismatch (Softmax)", i);
        }
    }
}

// ── Plan 289 sink-aware tests (require sink_aware_attn feature) ───

#[cfg(all(test, feature = "parallax_attn", feature = "sink_aware_attn"))]
mod sink_aware_tests {
    use super::*;
    use crate::data_probe::{
        SinkAwarePolicy, SinkClassifierConfig, SinkKind, StableRankScratch,
        apply_dual_policy_gate_flat,
    };

    fn parallax_zero_cfg(act: ParallaxActivation) -> ParallaxConfig {
        ParallaxConfig { gate_scale: 0.0, zero_init: true, activation: act }
    }

    /// T3.1 — Uniform policy path produces bit-identical output to vanilla forward.
    #[test]
    fn plan289_uniform_bit_identical_to_vanilla() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        super::tests::lcg_fill(0xA1B2, &mut q);
        super::tests::lcg_fill(0xC3D4, &mut k);
        super::tests::lcg_fill(0xE5F6, &mut v);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let mut out_vanilla = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q, &k, &v, &mut out_vanilla, n, d, scale, &r, &x, &cfg, None,
        );

        let mut out_uniform = vec![0.0f32; n * d];
        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d);
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_uniform, n, d, scale, &r, &x, &cfg,
            &SinkAwarePolicy::Uniform, -10.0, &mut sink_scratch, None,
        );
        assert!(matches!(kind, SinkKind::None), "Uniform must return SinkKind::None");

        for i in 0..(n * d) {
            assert_eq!(out_vanilla[i], out_uniform[i], "output[{}] differs (Uniform path)", i);
        }
    }

    /// T3.2 — DualPolicy path bit-identical to manual composition.
    #[test]
    fn plan289_dualpolicy_matches_manual_composition() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        super::tests::lcg_fill(0x11, &mut q);
        super::tests::lcg_fill(0x22, &mut k);
        super::tests::lcg_fill(0x33, &mut v);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy_cfg = SinkClassifierConfig::default();
        let policy = SinkAwarePolicy::DualPolicy(policy_cfg);
        let gate_scale = -2.0;

        // Wrapper path
        let mut out_wrapper = vec![0.0f32; n * d];
        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d);
        let kind_wrapper = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_wrapper, n, d, scale, &r, &x, &cfg,
            &policy, gate_scale, &mut sink_scratch, None,
        );

        // Manual composition
        let mut out_manual = vec![0.0f32; n * d];
        let mut o_temp = vec![0.0f32; n * d];
        let mut am = vec![0.0f32; n * n];
        let mut classifier = StableRankScratch::new(d);
        tiled_attention_parallax_forward_retaining(
            &q, &k, &v, &mut o_temp, n, d, scale, &r, &x, &cfg, Some(&mut am), None,
        );
        let kind_manual = apply_dual_policy_gate_flat(
            &am, &v, &o_temp, n, d, &policy, gate_scale, &mut classifier, &mut out_manual,
        );

        assert_eq!(kind_wrapper, kind_manual, "SinkKind mismatch");
        for i in 0..(n * d) {
            assert_eq!(out_wrapper[i], out_manual[i], "output[{}] differs (DualPolicy path)", i);
        }
    }

    /// T3.3 — synthetic NOP head: dominant sink has zero v, classifier must
    /// return Nop, and output must be scaled by σ(gate_scale).
    #[test]
    fn plan289_synthetic_nop_head_gated() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::tests::build_sink_case(n, d, 0, true);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());

        // Ungated reference via Uniform.
        let mut out_ungated = vec![0.0f32; n * d];
        let mut sa_scratch_un = SinkAwareParallaxScratch::new(n, d);
        tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_ungated, n, d, scale, &r, &x, &cfg,
            &SinkAwarePolicy::Uniform, 0.0, &mut sa_scratch_un, None,
        );

        // DualPolicy with strong suppression.
        let gate_scale = -10.0; // σ(-10) ≈ 4.5e-5
        let mut out_gated = vec![0.0f32; n * d];
        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d);
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_gated, n, d, scale, &r, &x, &cfg,
            &policy, gate_scale, &mut sink_scratch, None,
        );
        assert!(matches!(kind, SinkKind::Nop), "expected Nop, got {:?}", kind);

        // NOP-gated output must equal σ(gate_scale) × ungated output.
        let sigma = 1.0 / (1.0 + (-gate_scale).exp());
        for i in 0..(n * d) {
            let expected = out_ungated[i] * sigma;
            let delta = (out_gated[i] - expected).abs();
            assert!(delta < 1e-5, "gated[{}]={} != σ(gs)·ungated={} (delta {})",
                i, out_gated[i], expected, delta);
        }
    }

    /// T3.4 — synthetic Broadcast head: dominant sink carries content AND the
    /// AV update is rank-1. Classifier must return Broadcast, and output must
    /// be bit-identical to the Uniform (ungated) path.
    #[test]
    fn plan289_synthetic_broadcast_head_preserved() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::tests::build_sink_case(n, d, 0, false);

        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let mut out_uniform = vec![0.0f32; n * d];
        let mut sa_un = SinkAwareParallaxScratch::new(n, d);
        tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_uniform, n, d, scale, &r, &x, &cfg,
            &SinkAwarePolicy::Uniform, 0.0, &mut sa_un, None,
        );

        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());
        let gate_scale = -10.0;
        let mut out_dp = vec![0.0f32; n * d];
        let mut sa_dp = SinkAwareParallaxScratch::new(n, d);
        let kind = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_dp, n, d, scale, &r, &x, &cfg,
            &policy, gate_scale, &mut sa_dp, None,
        );
        assert!(matches!(kind, SinkKind::Broadcast), "expected Broadcast, got {:?}", kind);

        for i in 0..(n * d) {
            assert_eq!(out_uniform[i], out_dp[i], "Broadcast output[{}] must equal Uniform", i);
        }
    }

    /// Cached path: wrapper uses cached variant when `cached = Some`. Two
    /// consecutive DualPolicy calls → second reuses cached SinkKind.
    #[test]
    fn plan289_cached_path_audit_and_reuse() {
        let n = 16;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let (q, k, v) = super::tests::build_sink_case(n, d, 0, true);
        let cfg = parallax_zero_cfg(ParallaxActivation::Sigmoid);
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let policy = SinkAwarePolicy::DualPolicy(SinkClassifierConfig::default());

        let mut sink_scratch = SinkAwareParallaxScratch::new(n, d).with_cache();
        if let Some(c) = sink_scratch.cached.as_mut() {
            c.audit_every_n = 4;
        }

        let gate_scale = -5.0;
        let mut out_a = vec![0.0f32; n * d];
        let kind_a = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_a, n, d, scale, &r, &x, &cfg,
            &policy, gate_scale, &mut sink_scratch, None,
        );
        assert!(matches!(kind_a, SinkKind::Nop));
        assert_eq!(
            sink_scratch.cached.as_ref().unwrap().calls_since_audit, 1,
            "first call must reset cadence counter"
        );

        let mut out_b = vec![0.0f32; n * d];
        let kind_b = tiled_attention_parallax_forward_sink_aware(
            &q, &k, &v, &mut out_b, n, d, scale, &r, &x, &cfg,
            &policy, gate_scale, &mut sink_scratch, None,
        );
        assert!(matches!(kind_b, SinkKind::Nop));
        assert_eq!(
            sink_scratch.cached.as_ref().unwrap().calls_since_audit, 2,
            "second call must increment without re-audit"
        );

        for i in 0..(n * d) {
            assert_eq!(out_a[i], out_b[i], "cached NOP output[{}] must match audit output", i);
        }
    }
}
