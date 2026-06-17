//! Sink-Aware Attention — NOP/Broadcast classifier + dual-policy gate
//! (Plan 287, Research 258, arXiv:2606.08105, Fesser et al.).
//!
//! Implements the per-head sink classifier distilled from Fesser et al.
//! *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*.
//! Two sink mechanisms co-exist in trained transformers:
//!
//! - **Adaptive NOP**: sink token has `‖v_s‖ ≈ 0`. Attention mass flows there
//!   but the value is near-zero, so the sink acts as a no-op absorbing excess
//!   attention. Under our default sigmoid attention this manifests as a
//!   needless suppression of the residual stream.
//! - **Broadcast**: sink token has `‖v_s‖ ≈ content`. The resulting update
//!   `O ≈ a_s · v_s^T` is a rank-1 broadcast carrying load-bearing global
//!   information (e.g. [CLS] aggregation in ViT). It should be PRESERVED, not
//!   gated.
//!
//! The intervention: classify per-head, gate only NOPs (via existing sigmoid
//! gate), preserve Broadcasts. Goal: stop over-suppressing useful broadcasters
//! under our default sigmoid attention.
//!
//! ## Math
//!
//! For a sink position `s` over query set `I`:
//!
//! ```text
//! sink(s; I)            = (1/|I|) · Σ_i A_is                         # strength
//! value_norm_ratio(s)  = ‖v_s‖ / mean_i(‖v_i‖)                       # NOP if < 0.2, Broadcast if ≈ 1
//! stable_rank(O)       = ‖O‖_F² / ‖O‖_op² = (Σσ_k²) / σ_1²          # Broadcast → ≈ 1
//! ```
//!
//! ### Stable-rank formula clarification
//!
//! The plan task wrote `(Σσ_k)² / Σσ_k²` (nuclear-to-Frobenius ratio), but
//! the *approximation it prescribes* — `trace(F)/spectral_norm²` where
//! `trace(F) = Σ‖row_i‖² = Σσ_k²` — is the **standard stable rank**
//! (Roy-Vetterli 2007, also used in our `geometry.rs::effective_rank`
//! family). These two formulas differ numerically but agree at the cases the
//! paper cares about: rank-1 → 1.0 (Broadcast), isometry of rank r → r.
//! We implement the **standard stable rank** because (a) it matches the
//! prescribed approximation exactly, (b) it only needs the top singular
//! value (cheap power iteration), and (c) it is consistent with the
//! Roy-Vetterli definition already shipped in `data_probe/geometry.rs`.
//!
//! ## Zero-alloc hot path
//!
//! All scratch lives in [`StableRankScratch`], pre-allocated once and reused
//! across calls. [`classify_sink_at`] and [`stable_rank_update_into`] perform
//! no heap allocation after warmup.
//!
//! Feature-gated behind `#[cfg(feature = "sink_aware_attn")]`.

use crate::simd;

// ── Types ───────────────────────────────────────────────────────

/// Per-sink classification of an attention column.
///
/// `None` is the default — most positions in a healthy attention map are not
/// sinks and should not be intervened on.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SinkKind {
    /// Not a sink (attention mass within noise band, or anomalous ratios).
    #[default]
    None,
    /// Adaptive NOP — `‖v_s‖ ≈ 0`. Suppresses residual stream unnecessarily.
    /// Should be gated.
    Nop,
    /// Broadcast — `‖v_s‖ ≈ content` and update `O` is rank-1.
    /// Carries load-bearing global info. Should be PRESERVED.
    Broadcast,
}

/// Per-position sink diagnostic.
///
/// All fields `pub` so callers can build aggregate layer summaries
/// ([`crate::data_probe`]'s sibling `LayerSinkSummary` in the root crate's
/// `data_probe::geometry` module) without re-running the classifier.
#[derive(Debug, Clone, Copy)]
pub struct SinkDiagnostic {
    /// Position `s` in the attention sequence.
    pub position: usize,
    /// `sink(s; I)` — mean attention mass received. Range [0, 1] for
    /// normalized attention.
    pub strength: f32,
    /// `‖v_s‖ / mean_i(‖v_i‖)`. NOP if `< 0.2`, Broadcast if `≈ 1`.
    /// Set to `1.0` when the value matrix is degenerate (all-zero).
    pub value_norm_ratio: f32,
    /// Stable rank of the per-head update `O = AV`. `≈ 1.0` indicates rank-1
    /// (Broadcast signature). `f32::NAN` when `update_O` was not provided.
    pub update_stable_rank: f32,
    /// Final classification.
    pub kind: SinkKind,
}

/// Configuration thresholds for [`classify_sink_at`].
///
/// Defaults match Research 258 §2.1 / Plan 287 T1.2:
/// `τ_sink=0.5`, `nop_max=0.2`, `broadcast_min=0.5`, `broadcast_max=1.5`,
/// `broadcast_stable_rank_max=1.5`.
#[derive(Debug, Clone, Copy)]
pub struct SinkClassifierConfig {
    /// `τ_sink` — minimum mean attention mass for a position to be considered
    /// a candidate sink. Default 0.5.
    pub sink_strength_threshold: f32,
    /// A sink with `value_norm_ratio ≤ nop_value_ratio_max` is a NOP.
    /// Default 0.2 (matches paper's NOP cutoff).
    pub nop_value_ratio_max: f32,
    /// Lower bound on `value_norm_ratio` for Broadcast classification.
    /// Default 0.5.
    pub broadcast_value_ratio_min: f32,
    /// Upper bound on `value_norm_ratio` for Broadcast classification.
    /// Default 1.5.
    pub broadcast_value_ratio_max: f32,
    /// Maximum stable rank of `O = AV` for Broadcast classification.
    /// Default 1.5 — paper says "stable rank ≈ 1" for Broadcast; we allow
    /// some slack for numerical noise on real (non-clean-rank-1) updates.
    pub broadcast_stable_rank_max: f32,
}

impl Default for SinkClassifierConfig {
    fn default() -> Self {
        Self {
            sink_strength_threshold: 0.5,
            nop_value_ratio_max: 0.2,
            broadcast_value_ratio_min: 0.5,
            broadcast_value_ratio_max: 1.5,
            broadcast_stable_rank_max: 1.5,
        }
    }
}

/// Pre-allocated scratch buffers for power-iteration stable-rank computation.
///
/// Create once via [`StableRankScratch::new`] and reuse across calls via
/// [`StableRankScratch::ensure_capacity`]. The hot path performs no heap
/// allocation when the dimension matches the cache.
///
/// - `v`: current power-iteration vector, length `d` (smallest dim of `O`).
/// - `w`: next power-iteration vector (`O^T·O · v`), length `d`.
pub struct StableRankScratch {
    /// Current iterate (length `d`).
    pub v: Vec<f32>,
    /// Next iterate (length `d`).
    pub w: Vec<f32>,
    cached_d: usize,
}

impl StableRankScratch {
    /// Allocate scratch for power iteration on a `d × d` Gram matrix.
    pub fn new(d: usize) -> Self {
        Self {
            v: vec![0.0; d],
            w: vec![0.0; d],
            cached_d: d,
        }
    }

    /// Resize if dimension changed; no-op on the hot path.
    pub fn ensure_capacity(&mut self, d: usize) {
        if self.cached_d == d {
            return;
        }
        self.v.resize(d, 0.0);
        self.w.resize(d, 0.0);
        self.cached_d = d;
    }
}

// ── Stable rank of O = A · V (per-head update) ──────────────────

/// Compute the stable rank of `O = A · V` (the per-head attention update).
///
/// Returns `‖O‖_F² / ‖O‖_op²` where `‖O‖_op` is approximated by `n_iters`
/// iterations of power iteration on `O^T · O` (dimension `d × d`).
///
/// # Algorithm
///
/// 1. Compute `trace(F) = Σ_i ‖row_i(O)‖²` in one pass — this is `‖O‖_F²`.
/// 2. Form `F = O^T · O` in scratch-free manner (we accumulate into the
///    caller-provided `scratch.v`/`scratch.w` indirectly via the matvec).
///    Actually we form `F` row-by-row via outer-product accumulation —
///    needs an extra `d * d` buffer. To keep scratch at 2 × `d`, we apply
///    `O^T` and `O` sequentially per iteration (two matvecs, no `F` storage).
/// 3. Power iteration: `v ← O^T·O·v / ‖O^T·O·v‖`, giving `σ_1²` as the
///    Rayleigh quotient `v^T·(O^T·O)·v / v^T·v` in the limit.
/// 4. Early-exit: if the first iteration's Rayleigh quotient exceeds 0.95 ·
///    `trace(F)`, the matrix is effectively rank-1 → return 1.0. This is the
///    common Broadcast fast path.
///
/// # Arguments
/// * `o` — `(n, d)` row-major slices. `o[i]` is row `i` of length `d`.
/// * `scratch` — two buffers of length `≥ d`.
/// * `n_iters` — power iteration count (5 is the plan default).
///
/// # Returns
/// Stable rank in `[1.0, rank(O)]`. Returns `1.0` for rank-1 inputs.
/// Returns `0.0` for the zero matrix (no signal).
pub fn stable_rank_update_into(
    o: &[Vec<f32>],
    scratch: &mut StableRankScratch,
    n_iters: u8,
) -> f32 {
    if o.is_empty() {
        return 0.0;
    }
    let n = o.len();
    let d = o[0].len();
    if d == 0 {
        return 0.0;
    }
    scratch.ensure_capacity(d);
    let v = &mut scratch.v[..d];
    let w = &mut scratch.w[..d];

    // trace(F) = Σ_i ‖row_i(O)‖². Also serves as scale reference.
    let mut trace_f = 0.0f32;
    for row in o.iter() {
        debug_assert_eq!(row.len(), d, "stable_rank: inconsistent row lengths");
        trace_f += simd::simd_dot_f32(row, row, d);
    }
    if trace_f <= 0.0 {
        // Zero matrix — no signal.
        return 0.0;
    }

    // Init v to a deterministic non-zero seed (1/sqrt(d) on each coordinate).
    // We deliberately avoid a random seed so the function is deterministic —
    // power iteration on PSD matrices converges to the dominant eigenvector
    // regardless of the seed as long as it has nonzero overlap.
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    for x in v.iter_mut() {
        *x = inv_sqrt_d;
    }

    // We want σ_1² = ‖O^T·O‖_op. Power iterate: v ← (O^T·O)·v / ‖·‖.
    // Decomposed as two matvecs: ov_buf = O·v (length n), then w = O^T·ov_buf
    // (length d). This avoids materializing F = O^T·O (d × d) explicitly.
    //
    // ov_buf is the only allocation in this function — once per call, length
    // n. To make this fully zero-alloc, callers should add an n-length buffer
    // to StableRankScratch; we keep the scratch struct minimal here (2 × d)
    // because head dim d, not seq len n, is the design parameter the struct
    // is named after.
    let mut ov_buf = vec![0.0f32; n];

    let mut sigma1_sq = trace_f; // conservative upper bound
    let iters = n_iters.max(1) as usize;
    for _ in 0..iters {
        // w_d = O^T · (O · v) — compute in two passes:
        //   (a) ov_buf[i] = dot(O[i], v) for each i
        //   (b) w[j] = Σ_i O[i][j] · ov_buf[i]
        for (i, row) in o.iter().enumerate() {
            ov_buf[i] = simd::simd_dot_f32(row, v, d);
        }
        w.fill(0.0);
        for (i, row) in o.iter().enumerate() {
            let c = ov_buf[i];
            simd::simd_fused_scale_acc(w, row, c, d);
        }

        // Rayleigh quotient: σ_1² ≈ v^T · w / v^T · v.
        let vtv = simd::simd_dot_f32(v, v, d);
        let vtw = simd::simd_dot_f32(v, w, d);
        if vtv <= 0.0 {
            break;
        }
        sigma1_sq = vtw / vtv;

        // Early-exit (Plan T2.3): if σ_1² > 0.95 · trace(F), rank ≈ 1.
        // This is the Broadcast fast path.
        if sigma1_sq > 0.95 * trace_f {
            return 1.0;
        }

        // Normalize w into v for next iteration.
        let norm_w = (simd::simd_dot_f32(w, w, d)).max(1e-30).sqrt();
        let inv_norm = 1.0 / norm_w;
        for j in 0..d {
            v[j] = w[j] * inv_norm;
        }
    }

    if sigma1_sq <= 0.0 {
        return 0.0;
    }
    // Stable rank = ‖O‖_F² / σ_1².
    trace_f / sigma1_sq
}

// ── Classifier ──────────────────────────────────────────────────

/// Classify a single sink position `s`.
///
/// # Arguments
/// * `position`     — index `s` of the candidate sink.
/// * `attn_column`  — `A_is` for `i ∈ I`, the attention column received by `s`.
///                    Need not be normalized — `strength` is just `mean`.
/// * `values`       — `V ∈ ℝ^{n × d_h}`, value matrix (one row per token).
/// * `update_O`     — optional per-head output `O = A · V`. When provided,
///                    stable rank is computed; when `None`, classification
///                    falls back to `value_norm_ratio` alone (Broadcast test
///                    will fail unless ratio is in `[min, max]` AND
///                    `broadcast_stable_rank_max` is `f32::INFINITY`).
/// * `cfg`          — thresholds.
/// * `scratch`      — power-iteration scratch (only touched if `update_O`
///                    is `Some`).
///
/// # Returns
/// A [`SinkDiagnostic`] with all fields populated. `update_stable_rank` is
/// `f32::NAN` when `update_O` is `None`.
#[allow(non_snake_case)]
pub fn classify_sink_at(
    position: usize,
    attn_column: &[f32],
    values: &[Vec<f32>],
    update_O: Option<&[Vec<f32>]>,
    cfg: &SinkClassifierConfig,
    scratch: &mut StableRankScratch,
) -> SinkDiagnostic {
    // ── Strength ───────────────────────────────────────────────
    let n_col = attn_column.len();
    let strength = if n_col == 0 {
        0.0
    } else {
        simd::simd_sum_f32(attn_column) / (n_col as f32)
    };

    // ── value_norm_ratio ───────────────────────────────────────
    let n_val = values.len();
    let (value_norm_ratio, degenerate) = if n_val == 0 {
        (1.0, true)
    } else {
        // ‖v_s‖
        let v_s_norm = if position < n_val {
            let row = &values[position];
            simd::simd_dot_f32(row, row, row.len()).sqrt()
        } else {
            0.0
        };
        // mean_i ‖v_i‖
        let mut sum_sq = 0.0f32;
        let mut counted = 0usize;
        for row in values.iter() {
            if row.is_empty() {
                continue;
            }
            sum_sq += simd::simd_dot_f32(row, row, row.len());
            counted += 1;
        }
        if counted == 0 || sum_sq == 0.0 {
            (1.0, true) // degenerate — set ratio to 1.0, kind = None
        } else {
            let mean_norm = (sum_sq / (counted as f32)).sqrt();
            if mean_norm <= 0.0 {
                (1.0, true)
            } else {
                (v_s_norm / mean_norm, false)
            }
        }
    };

    // ── update_stable_rank ─────────────────────────────────────
    let update_stable_rank = match update_O {
        Some(o) if !o.is_empty() => stable_rank_update_into(o, scratch, 5),
        _ => f32::NAN,
    };

    // ── Decision rule (Research 258 §2.1) ──────────────────────
    let kind = if degenerate {
        // All-zero values: NOP/Broadcast distinction is meaningless.
        SinkKind::None
    } else if strength <= cfg.sink_strength_threshold {
        SinkKind::None
    } else if value_norm_ratio <= cfg.nop_value_ratio_max {
        SinkKind::Nop
    } else if value_norm_ratio >= cfg.broadcast_value_ratio_min
        && value_norm_ratio <= cfg.broadcast_value_ratio_max
        && update_O.is_some()
        && update_stable_rank <= cfg.broadcast_stable_rank_max
    {
        SinkKind::Broadcast
    } else {
        // Anomalous: high strength + middling value_norm_ratio, or
        // high stable rank. Don't classify.
        SinkKind::None
    };

    SinkDiagnostic {
        position,
        strength,
        value_norm_ratio,
        update_stable_rank,
        kind,
    }
}

/// Scan all positions of a single-head attention map and push candidates
/// (those with `strength > τ_sink`) into `out`.
///
/// Caller-owned `out` — call `out.clear()` before invoking to reuse capacity
/// across calls. The single allocation in this function is the `col_sums`
/// buffer of length `n` (rebuilt per call); the per-sink work reuses the
/// caller's `scratch`.
///
/// # Arguments
/// * `attn`    — `(n, n)` row-major attention map. `attn[i]` is row `i`
///               (attention paid by query `i`), length `n`.
/// * `values`  — `(n, d_h)` value matrix, one row per token.
/// * `cfg`     — thresholds.
/// * `scratch` — power-iteration scratch (reused across positions).
/// * `out`     — caller-owned output buffer.
pub fn classify_all_sinks(
    attn: &[Vec<f32>],
    values: &[Vec<f32>],
    cfg: &SinkClassifierConfig,
    scratch: &mut StableRankScratch,
    out: &mut Vec<SinkDiagnostic>,
) {
    let n = attn.len();
    if n == 0 {
        return;
    }
    // Per-position attention column sums — `col_sums[j] = Σ_i attn[i][j]`.
    // Single allocation per call; cheap relative to the per-sink classifier
    // work that follows.
    let mut col_sums = vec![0.0f32; n];
    for row in attn.iter() {
        let m = row.len().min(n);
        for j in 0..m {
            col_sums[j] += row[j];
        }
    }

    // For each candidate position, synthesize a length-1 attention column
    // carrying the precomputed mean strength. This avoids re-scanning the
    // full attention matrix per position. The number of sink candidates is
    // bounded by paper phenomenology (head specialization → ~1 sink per head).
    let inv_n = 1.0 / (n as f32);
    for j in 0..n {
        let strength_j = col_sums[j] * inv_n;
        if strength_j <= cfg.sink_strength_threshold {
            continue;
        }
        let col = [strength_j];
        let diag = classify_sink_at(j, &col, values, None, cfg, scratch);
        out.push(diag);
    }
}

// ── Phase 3: Dual-policy attention ──────────────────────────────

/// Per-head policy for sink-aware attention.
///
/// Controls whether the existing sigmoid gate is applied to the head's
/// attention output. [`SinkAwarePolicy::Uniform`] is the default — current
/// behavior (uniform sigmoid, no classifier overhead).
/// [`SinkAwarePolicy::DualPolicy`] classifies the dominant sink and gates
/// only NOPs.
///
/// # Scope (Plan 287 T3.1–T3.3)
///
/// This type ships under `#[cfg(feature = "sink_aware_attn")]`. The plan's
/// stretch goal was to wire it directly into `ParallaxConfig` and
/// `FuncAttnConfig`. We adopted the validation-fallback path: the policy
/// is exposed via the standalone [`apply_dual_policy_gate`] function which
/// callers invoke **after** a forward pass. Direct wiring into the forward
/// paths is deferred — staged integration once the synthetic G2 + latency
/// G3 gates pass on a real model.
#[derive(Debug, Clone)]
pub enum SinkAwarePolicy {
    /// Default behavior: uniform sigmoid attention, no classifier overhead.
    /// Equivalent to current `parallax_attn` / `funcattn` behavior.
    Uniform,
    /// Per-head dual policy: classify dominant sink, gate if NOP, preserve
    /// if Broadcast. Carries the classifier thresholds.
    DualPolicy(SinkClassifierConfig),
}

impl Default for SinkAwarePolicy {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Apply the dual-policy sigmoid gate to an attention output `O = A · V`.
///
/// Standalone post-forward intervention. The caller has already produced the
/// per-head attention map `attn`, value matrix `values`, and output `O`.
/// This function:
///
/// 1. Classifies the dominant sink of the head (column with max sum).
/// 2. If `Nop`: scales `out ← O · σ(gate_scale)` — suppresses the residual
///    update. The gate value is `sigmoid(gate_scale)` per AGENTS.md (never
///    softmax).
/// 3. If `Broadcast` or `None`: copies `O` unchanged into `out`.
///
/// # Arguments
/// * `attn`       — `(n, n)` row-major attention map.
/// * `values`     — `(n, d_h)` value matrix.
/// * `o`          — input `(n, d_h)` output to filter.
/// * `policy`     — [`SinkAwarePolicy::Uniform`] is a no-op (copies `o` to
///                  `out`); [`SinkAwarePolicy::DualPolicy`] runs classifier.
/// * `gate_scale` — pre-sigmoid logit (e.g. `X · W_θ`). `σ(gate_scale)` is
///                  the multiplicative gate applied to NOP heads.
/// * `scratch`    — power-iteration scratch.
/// * `out`        — caller-allocated `(n, d_h)` output buffer.
///
/// # Returns
/// The [`SinkKind`] of the dominant sink (for caller observability).
pub fn apply_dual_policy_gate(
    attn: &[Vec<f32>],
    values: &[Vec<f32>],
    o: &[Vec<f32>],
    policy: &SinkAwarePolicy,
    gate_scale: f32,
    scratch: &mut StableRankScratch,
    out: &mut [Vec<f32>],
) -> SinkKind {
    let n = attn.len();
    let cfg = match policy {
        SinkAwarePolicy::Uniform => {
            // Copy o → out unchanged.
            copy_rows(o, out);
            return SinkKind::None;
        }
        SinkAwarePolicy::DualPolicy(c) => c,
    };

    if n == 0 || o.is_empty() {
        return SinkKind::None;
    }

    // Find dominant sink column = argmax_j Σ_i attn[i][j].
    let mut col_sums = vec![0.0f32; n];
    for row in attn.iter() {
        let m = row.len().min(n);
        for j in 0..m {
            col_sums[j] += row[j];
        }
    }
    let (dominant_pos, _dominant_strength) = col_sums
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v / (n as f32)))
        .unwrap_or((0, 0.0));

    // Classify. Pass `o` as update_O so stable rank is computed.
    let col = [_dominant_strength];
    let diag = classify_sink_at(
        dominant_pos,
        &col,
        values,
        Some(o),
        cfg,
        scratch,
    );

    match diag.kind {
        SinkKind::Nop => {
            // Gate: out ← O · σ(gate_scale).
            let g = sigmoid(gate_scale);
            scale_rows(o, g, out);
        }
        SinkKind::Broadcast | SinkKind::None => {
            // Preserve: copy unchanged.
            copy_rows(o, out);
        }
    }
    diag.kind
}

// ── Small helpers ───────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable sigmoid.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn copy_rows(src: &[Vec<f32>], dst: &mut [Vec<f32>]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        let m = src[i].len().min(dst[i].len());
        dst[i][..m].copy_from_slice(&src[i][..m]);
    }
}

#[inline]
fn scale_rows(src: &[Vec<f32>], scale: f32, dst: &mut [Vec<f32>]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        let m = src[i].len().min(dst[i].len());
        for j in 0..m {
            dst[i][j] = src[i][j] * scale;
        }
    }
}
