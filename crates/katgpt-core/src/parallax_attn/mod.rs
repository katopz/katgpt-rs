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
//! ## Module layout
//!
//! Split from the historical monolithic `src/parallax_attn.rs` (Issue 167,
//! 2026-07-17). The implementation stays in `mod.rs`; the test suite moved
//! to [`tests`] (tests are exempt from the 2048-line soft limit per Issue 162).
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
//! ## Per-key prior-logit lane + the length-aware margin law (Issue 819, Research 566)
//!
//! Attention is the closed-form solution of a KL-regularized transport problem
//! against a prior π; the log-prior is the missing additive term in the
//! transport cost (arXiv:2601.15380, GOAT). For the softmax arm the paper's
//! form is `p* = softmax(s/τ + log π)`. Our default sigmoid arm solves the
//! same KL(g‖π) regularizer over independent gates on the hypercube to the
//! closed form `g* = σ(s + logit π)` — see Research 566 §2 for the derivation
//! (no published precedent, verified 2026-09-17).
//!
//! [`ParallaxConfig::prior_logits`] (gated `prior_logit_lane`) injects the
//! per-key log-prior `ℓ_j = logit π_j` additively into the pre-normalization
//! scores: one f32 add per key, applied AFTER SSMax's length rescale and
//! BEFORE normalization — the prior is not a content score and must not be
//! rescaled by the length temperature, mirroring GOAT's pre-scaling trick that
//! keeps prior lanes unscaled by the kernel temperature. `None` (the default)
//! is the bit-identical constant path: the lane-off branch is the literal
//! pre-lane code, and a lane table of all zeros is also output-identical (the
//! `±0.0` score perturbation dies at `exp(±0) = 1` in both activations).
//!
//! **Stability law (sigmoid margin bound).** With content-score dynamic range
//! ω and prior-logit margin δ = ℓ_sink − min_{k∈context} ℓ_k, every context
//! gate obeys `g_k ≤ e^{ω−δ}`, so context-noise sensitivity is bounded by
//! `‖Δo‖ ≲ ε·(L−1)·e^{ω−δ}` — finite protection requires **δ ≳ ω + ln L**
//! (the same logarithmic margin scaling as the paper's softmax Thm 5.4, and
//! strictly more load-bearing on the sigmoid arm: with a uniform prior the
//! context mass grows ≈ L·σ(ω) — linear and unbounded, where softmax
//! saturates at Ψ→1. Softmax redistributes noise; sigmoid accumulates it).
//! The shipped constant `logit_bias` (engram kernel, Plan 364) is the
//! constant-prior special case; the xHC "boot CLOSED at −4" init is the
//! maximum-entropy-style default prior this law says is load-bearing.
//! `forecast_stable_positions(δ) = e^δ` (in `data_probe`, gated
//! `sink_margin_forecast`) forecasts the context length a measured margin
//! protects at the ε = ½ sensitivity budget.
//!
//! ## Sink-Aware Composition (Plan 289)
//!
//! When both `parallax_attn` and `sink_aware_attn` features are enabled, this
//! module also exposes [`tiled_attention_parallax_forward_sink_aware`] — a
//! single entry point composing the parallax forward with the dual-policy
//! NOP/Broadcast classifier from [`crate::data_probe`] (Plan 287, Research 258).
//!
//! - [`SinkAwarePolicy::Uniform`](crate::data_probe::SinkAwarePolicy::Uniform) short-circuits to vanilla
//!   [`tiled_attention_parallax_forward`] (zero-cost contract: ≤5% overhead,
//!   measured within noise across n ∈ {64, 128, 256}).
//! - [`SinkAwarePolicy::DualPolicy`](crate::data_probe::SinkAwarePolicy::DualPolicy) runs the retained-attention forward into
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
    /// Activation function for attention weight normalization.
    /// Default: Sigmoid (sink-free, higher COR capacity per Plan 161).
    /// Set to Softmax for backward-compatible attention sinks.
    pub activation: ParallaxActivation,
    /// Optional SSMax (length-aware log-N attention temperature) mode.
    /// When `Some`, the forward path applies `s_L · log(N)` multiplicative
    /// rescaling to the pre-normalization attention scores, cancelling the
    /// attention dilution bound at large N (Plan 411, Research 392,
    /// arXiv:2607.01538). Default `None` — no change to existing behavior.
    /// Only present when the `ssmax_temperature` feature is enabled.
    #[cfg(feature = "ssmax_temperature")]
    pub ssmax: Option<crate::ssmax::SsmaxMode>,
    /// Optional per-key additive prior-logit lane `ℓ_j = logit π_j`
    /// (Issue 819 T2, Research 566 — the sigmoid/softmax counterpart of
    /// GOAT's rank-1 key-only prior lane u(j)). Owned `Arc<[f32]>` (not a
    /// borrow) so the config keeps a single lifetime-free shape across every
    /// feature combination — the table is built once per head/window and
    /// shared. When `Some`, the forward adds `ℓ_j` to key `j`'s
    /// pre-normalization score — after SSMax's length rescale (the prior
    /// enters unscaled) and before normalization — so softmax realizes
    /// `softmax(s/τ + log π)` and normalized sigmoid the analogous KL-prior
    /// injection. Cost: one f32 add per key per query row.
    ///
    /// Default `None` is the bit-identical constant path (the lane-off branch
    /// is the literal pre-lane loop; a table of all zeros is output-identical
    /// because the `±0.0` score perturbation dies at `exp(±0) = 1`).
    ///
    /// Length contract: the table MUST have length ≥ `seq_len` (debug_assert;
    /// slice indexing fails loud otherwise). The margin law the lane serves:
    /// a prior margin δ ≳ ω + ln L bounds context-noise sensitivity by
    /// `‖Δo‖ ≲ ε·(L−1)·e^{ω−δ}` — see the module-level doc above.
    /// Only present when the `prior_logit_lane` feature is enabled (and the
    /// `parallax_attn` feature that compiles this module).
    #[cfg(feature = "prior_logit_lane")]
    pub prior_logits: Option<std::sync::Arc<[f32]>>,
    /// Optional scale-invariant schedule LUT (Plan 622 Phase 2, Research 610
    /// — arXiv:2505.17083). When `Some`, the forward applies the per-key
    /// affine `s → a_{i−j}·s + m_{i−j}` before normalization — `a_t` rides a
    /// per-key MULTIPLICATIVE lane (sibling of the additive
    /// `prior_logit_lane` shape) and `m_t` reuses the additive shape, one
    /// fused affine per key, distance-indexed. BOTH arms (softmax and
    /// normalized sigmoid) consume the same LUT — the sigmoid tilt transfer
    /// gives the sigmoid arm its first length-generalization mechanism.
    ///
    /// ORDERING LAW: this field and [`ParallaxConfig::ssmax`] are BOTH length
    /// temperatures; arming both stacks two sharpeners and is a LOUD config
    /// error (asserted in the forward, pinned by test). The sink key (0) is
    /// excluded from the transform; future keys pass through bit-exact.
    /// Cost: one fused affine per causal key per row; zero alloc (the owned
    /// Arc LUT is built once per session and shared). Default `None` is the
    /// bit-identical constant path. Only present when the
    /// `scale_invariant_attn` feature is enabled (and the `parallax_attn`
    /// feature that compiles this module).
    #[cfg(feature = "scale_invariant_attn")]
    pub scale_invariant: Option<std::sync::Arc<crate::scale_invariant::ScaleInvariantLut>>,
}

impl Default for ParallaxConfig {
    fn default() -> Self {
        Self {
            gate_scale: 1.0,
            activation: ParallaxActivation::default(),
            #[cfg(feature = "ssmax_temperature")]
            ssmax: None,
            #[cfg(feature = "prior_logit_lane")]
            prior_logits: None,
            #[cfg(feature = "scale_invariant_attn")]
            scale_invariant: None,
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

/// Apply SSMax (length-aware log-N attention temperature) to a score row,
/// if the config requests it. No-op when `ssmax` is `None` or the feature
/// is off.
///
/// This is the SSMax intervention point in the parallax forward path:
/// scores are computed, then SSMax rescales them by `s_L · log(N)` before
/// normalization (sigmoid/softmax). Plan 411, Research 392.
#[cfg(feature = "ssmax_temperature")]
#[inline]
fn apply_ssmax_to_row(row: &mut [f32], ssmax: Option<&crate::ssmax::SsmaxMode>) {
    if let Some(mode) = ssmax {
        let n = row.len();
        if n > 1 {
            let log_n = (n as f32).ln();
            crate::ssmax::apply_ssmax_inplace(row, mode, log_n);
        }
    }
}

/// Apply the per-key additive prior-logit lane to a score row, if configured.
/// No-op when `prior_logits` is `None` (the bit-identical constant path).
///
/// This is the prior-lane intervention point in the parallax forward path
/// (Issue 819 T2, Research 566): the additive counterpart of SSMax — SSMax
/// rescales content scores multiplicatively, the lane shifts them
/// additively — applied AFTER SSMax so the prior enters unscaled by the
/// length temperature, and BEFORE normalization.
#[cfg(feature = "prior_logit_lane")]
#[inline]
fn apply_prior_lane_to_row(row: &mut [f32], prior_logits: Option<&[f32]>) {
    if let Some(lane) = prior_logits {
        let n = row.len();
        debug_assert!(
            lane.len() >= n,
            "prior_logits lane shorter than seq_len ({}/{}): extend the table or pad with 0.0",
            lane.len(),
            n
        );
        // Slice (not zip-with-rest): a short lane fails loud in both profiles.
        let lane = &lane[..n];
        for (s, &l) in row.iter_mut().zip(lane) {
            *s += l;
        }
    }
}

/// Apply the scale-invariant per-key affine schedule to a score row, if
/// configured. No-op when `si` is `None` (the bit-identical constant path).
///
/// Plan 622 Phase 2 (Research 610): `a_t` rides the per-key MULTIPLICATIVE
/// lane and `m_t` reuses the additive lane shape — one fused affine
/// `s → a_{i−j}·s + m_{i−j}` per key, distance-indexed by the query's global
/// row index. Applied BEFORE normalization, in place of (never beside) SSMax
/// — the ordering law. The sink key (0) and future keys pass through inside
/// [`crate::scale_invariant::ScaleInvariantLut::apply_inplace`].
#[cfg(feature = "scale_invariant_attn")]
#[inline]
fn apply_scale_invariant_to_row(
    row: &mut [f32],
    si: Option<&crate::scale_invariant::ScaleInvariantLut>,
    query_i: usize,
) {
    if let Some(lut) = si {
        lut.apply_inplace(row, query_i, 0, 1.0);
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
    /// Column sums `c[j] = Σ_i softmax(i,j)`, length `seq_len`
    pub col_sums: Vec<f32>,
    /// Per-row score buffer, length `seq_len`
    pub scores: Vec<f32>,
    /// Σ_KV cross-covariance, length `head_dim * head_dim`
    pub sigma_kv: Vec<f32>,
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
        q,
        k,
        v,
        output,
        seq_len,
        head_dim,
        scale,
        r,
        x,
        parallax_config,
        None,
        scratch,
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

    // ORDERING LAW (Plan 622, Research 610): scale_invariant and ssmax are
    // BOTH length temperatures — arming both stacks two sharpeners and
    // double-sharpens attention. Loud config error in both profiles.
    #[cfg(all(feature = "scale_invariant_attn", feature = "ssmax_temperature"))]
    assert!(
        parallax_config.scale_invariant.is_none() || parallax_config.ssmax.is_none(),
        "scale_invariant_attn and ssmax_temperature are both length temperatures — \
         arming both double-sharpens attention (Plan 622 ordering law); arm exactly one"
    );

    if seq_len == 0 {
        return;
    }

    // Allocate scratch on demand if caller didn't provide one
    let mut local_scratch;
    let scratch = if let Some(s) = scratch {
        s.ensure_capacity(seq_len, head_dim);
        s
    } else {
        local_scratch = ParallaxScratch::new(seq_len, head_dim);
        &mut local_scratch
    };

    let d = head_dim;
    let n = seq_len;

    // Compute ρ = W_R · x (reuse scratch buffer)
    compute_rho(r, x, &mut scratch.rho);

    // If gate_scale is zero, or ρ is all zeros (zeroed W_R — the documented
    // zero-init convention for callers constructing W_R),
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
            #[cfg(feature = "ssmax_temperature")]
            parallax_config.ssmax.as_ref(),
            #[cfg(feature = "prior_logit_lane")]
            parallax_config.prior_logits.as_deref(),
            #[cfg(feature = "scale_invariant_attn")]
            parallax_config.scale_invariant.as_deref(),
        );
        return;
    }

    // Zero column sums before accumulation — scratch may be reused across calls
    scratch.col_sums[..n].fill(0.0);
    scratch.sigma_kv[..d * d].fill(0.0);

    // Phase 1: Compute o_SA and accumulate column sums in one pass
    for i in 0..n {
        let row_off = i * d;
        // Hoist the Q row slice out of the inner `j` loop: one bounds-check per
        // query row instead of one per (i, j) pair (n² → n total).
        let q_row = &q[row_off..row_off + d];
        output[row_off..row_off + d].fill(0.0);

        // Compute scores for row i: scores[j] = q_i · k_j * scale
        for j in 0..n {
            let k_off = j * d;
            scratch.scores[j] = simd::simd_dot_f32(q_row, &k[k_off..k_off + d], d) * scale;
        }

        // Scale-invariant per-key affine schedule (Plan 622 Phase 2).
        // Applied FIRST among the length lanes — it REPLACES SSMax (the
        // ordering law asserts they are never both armed above) and before
        // the additive prior lane. No-op when None.
        #[cfg(feature = "scale_invariant_attn")]
        apply_scale_invariant_to_row(
            &mut scratch.scores[..n],
            parallax_config.scale_invariant.as_deref(),
            i,
        );

        // SSMax: rescale scores by s_L · log(N) before normalization.
        // No-op when ssmax is None (the default).
        #[cfg(feature = "ssmax_temperature")]
        apply_ssmax_to_row(&mut scratch.scores[..n], parallax_config.ssmax.as_ref());

        // Per-key additive prior-logit lane (Issue 819 T2). Applied AFTER the
        // SSMax rescale — the prior enters unscaled by the length temperature
        // — and BEFORE normalization. No-op when None.
        #[cfg(feature = "prior_logit_lane")]
        apply_prior_lane_to_row(
            &mut scratch.scores[..n],
            parallax_config.prior_logits.as_deref(),
        );

        // Normalize attention weights (softmax or sigmoid)
        let row = &mut scratch.scores[..n];
        normalize_attention_weights(row, parallax_config.activation);

        // Retain normalized attention row if caller asked for the full matrix.
        // Branch is hoisted: either every row copies, or no row does.
        if let Some(am) = attn_matrix.as_deref_mut() {
            am[i * n..(i + 1) * n].copy_from_slice(&scratch.scores[..n]);
        }

        // Accumulate output: o_i = Σ_j p_ij · v_j
        // Hoist the mutable output-row borrow out of the `j` loop — one
        // bounds-check + reborrow per query row instead of per (i, j) pair.
        let out_row = &mut output[row_off..row_off + d];
        for j in 0..n {
            let p = scratch.scores[j];
            let v_off = j * d;
            simd::simd_fused_scale_acc(out_row, &v[v_off..v_off + d], p, d);
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
/// * `classifier` — [`StableRankScratch`](crate::data_probe::StableRankScratch) for the power-iteration kernel.
/// * `cached` — when `Some`, the wrapper uses [`apply_dual_policy_gate_cached_flat`](crate::data_probe::apply_dual_policy_gate_cached_flat)
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
/// * [`SinkAwarePolicy::Uniform`](crate::data_probe::SinkAwarePolicy::Uniform) — calls vanilla
///   [`tiled_attention_parallax_forward`] directly into `output`. **Zero
///   overhead** vs the vanilla path: no attention matrix is retained, no
///   temporary buffer is touched, no classifier runs. Returns
///   [`SinkKind::None`](crate::data_probe::SinkKind::None).
/// * [`SinkAwarePolicy::DualPolicy(_)`](crate::data_probe::SinkAwarePolicy::DualPolicy) — runs the retained forward into
///   `sink_scratch.o_temp` while writing the full `n×n` attention matrix into
///   `sink_scratch.attn_matrix`, then applies the flat dual-policy gate
///   (`o_temp → output`). When `sink_scratch.cached` is `Some`, uses the
///   cached variant (amortizes classifier cost; Plan 287 Issue 001 mitigation).
///
/// # Arguments
///
/// * `q`, `k`, `v`, `output`, `seq_len`, `head_dim`, `scale`, `r`, `x`,
///   `parallax_config`, `scratch` — see [`tiled_attention_parallax_forward`].
/// * `policy`        — [`SinkAwarePolicy::Uniform`](crate::data_probe::SinkAwarePolicy::Uniform) for the zero-cost path;
///   [`SinkAwarePolicy::DualPolicy`](crate::data_probe::SinkAwarePolicy::DualPolicy) to invoke the classifier + gate.
/// * `gate_scale`    — pre-sigmoid logit for the NOP gate. `σ(gate_scale)` is
///   the scale applied to NOP-classified output rows. Pass `0.0` for
///   σ(0)=0.5 (half-suppression) or a large negative for near-zero.
/// * `sink_scratch`  — owns `attn_matrix`, `o_temp`, classifier scratch, and
///   optional cache. Construct once via [`SinkAwareParallaxScratch::new`] and
///   reuse across calls.
///
/// # Returns
///
/// The dominant sink's [`SinkKind`](crate::data_probe::SinkKind) (`None` for Uniform path, or the
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
            q,
            k,
            v,
            output,
            seq_len,
            head_dim,
            scale,
            r,
            x,
            parallax_config,
            scratch,
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
        None => crate::data_probe::apply_dual_policy_gate_flat(
            attn_matrix,
            v,
            o_temp,
            n,
            d,
            policy,
            gate_scale,
            classifier,
            output,
        ),
    }
}

// ── SSMax + Sink-Aware 3-way composition (Plan 411 T2.3) ────────
// Requires all three of: parallax_attn, sink_aware_attn, ssmax_temperature.
// SSMax applies at the LOGIT level (pre-normalization), inside the parallax
// forward; the sink-aware gate applies at the OUTPUT level (post-value-weighted
// sum). They compose cleanly because they operate at different stages — there
// is no interference and the composition order is forced by the data flow.

/// Sink-aware parallax forward with SSMax log-N temperature (Plan 411 T2.3).
///
/// Three-way composition:
/// 1. **SSMax** (logit level) — rescales pre-normalization scores by
///    `s_L · log(N)` via `parallax_config.ssmax` (set by T2.1) or the explicit
///    `ssmax_mode` override.
/// 2. **Parallax forward** — normalization + value-weighted sum + covariance
///    correction.
/// 3. **Sink-aware gate** (output level) — dual-policy NOP/Broadcast classifier
///    on the retained attention matrix.
///
/// SSMax and the sink-aware gate compose cleanly because they operate at
/// different stages of the forward (logits vs output); there is no interference.
///
/// When `ssmax_mode` is `Some`, it takes precedence over
/// `parallax_config.ssmax` (the explicit param wins). This lets callers reuse
/// a base `ParallaxConfig` across SSMax-on / SSMax-off calls without mutating
/// the shared config.
///
/// Requires all three features: `parallax_attn`, `sink_aware_attn`,
/// `ssmax_temperature`.
#[cfg(all(
    feature = "parallax_attn",
    feature = "sink_aware_attn",
    feature = "ssmax_temperature"
))]
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_parallax_forward_sink_aware_ssmax(
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
    ssmax_mode: Option<&crate::ssmax::SsmaxMode>,
    policy: &crate::data_probe::SinkAwarePolicy,
    gate_scale: f32,
    sink_scratch: &mut SinkAwareParallaxScratch,
    scratch: Option<&mut ParallaxScratch>,
) -> crate::data_probe::SinkKind {
    // Inject SSMax into a cloned config when the explicit override is provided.
    // The clone is a few f32 + a Copy enum + the lane Arc incref — negligible
    // vs the n×n classifier. When ssmax_mode is None, use the config as-is
    // (it may already have ssmax set).
    let owned_cfg;
    let cfg: &ParallaxConfig = match ssmax_mode {
        Some(mode) => {
            owned_cfg = {
                let mut c = parallax_config.clone();
                c.ssmax = Some(*mode);
                c
            };
            &owned_cfg
        }
        None => parallax_config,
    };
    tiled_attention_parallax_forward_sink_aware(
        q,
        k,
        v,
        output,
        seq_len,
        head_dim,
        scale,
        r,
        x,
        cfg,
        policy,
        gate_scale,
        sink_scratch,
        scratch,
    )
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
    #[cfg(feature = "ssmax_temperature")] ssmax: Option<&crate::ssmax::SsmaxMode>,
    #[cfg(feature = "prior_logit_lane")] prior_logits: Option<&[f32]>,
    #[cfg(feature = "scale_invariant_attn")]
    si: Option<&crate::scale_invariant::ScaleInvariantLut>,
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
        // Hoist the Q row slice out of the inner `j` loop: one bounds-check per
        // query row instead of one per (i, j) pair (n² → n total).
        let q_row = &q[q_off..q_off + d];
        output[out_off..out_off + d].fill(0.0);

        // Compute scores for row i
        for (j, score_slot) in scores.iter_mut().enumerate().take(n) {
            let k_off = j * d;
            *score_slot = simd::simd_dot_f32(q_row, &k[k_off..k_off + d], d) * scale;
        }

        // Scale-invariant per-key affine schedule (Plan 622 Phase 2).
        // Applied FIRST among the length lanes — it REPLACES SSMax (the
        // ordering law is asserted by the caller above) and before the
        // additive prior lane. No-op when None.
        #[cfg(feature = "scale_invariant_attn")]
        apply_scale_invariant_to_row(&mut scores[..n], si, i);

        // SSMax: rescale scores by s_L · log(N) before normalization.
        // No-op when ssmax is None (the default).
        #[cfg(feature = "ssmax_temperature")]
        apply_ssmax_to_row(&mut scores[..n], ssmax);

        // Per-key additive prior-logit lane (Issue 819 T2). Applied AFTER the
        // SSMax rescale (the prior enters unscaled) and BEFORE normalization.
        // No-op when None.
        #[cfg(feature = "prior_logit_lane")]
        apply_prior_lane_to_row(&mut scores[..n], prior_logits);

        // Normalize attention weights (softmax or sigmoid)
        let row = &mut scores[..n];
        normalize_attention_weights(row, activation);

        // Retain normalized attention row if caller asked for the full matrix.
        if let Some(am) = attn_matrix.as_deref_mut() {
            am[i * n..(i + 1) * n].copy_from_slice(&scores[..n]);
        }

        // Accumulate output: o_i = Σ_j p_ij · v_j
        // Hoist the mutable output-row borrow out of the `j` loop — one
        // bounds-check + reborrow per query row instead of per (i, j) pair.
        let out_row = &mut output[out_off..out_off + d];
        for (j, &p) in scores.iter().enumerate().take(n) {
            let v_off = j * d;
            simd::simd_fused_scale_acc(out_row, &v[v_off..v_off + d], p, d);
        }
    }
}

// ── Tests — split into parallax_attn/tests.rs (Issue 167, 2026-07-17) ───

#[cfg(test)]
mod tests;
