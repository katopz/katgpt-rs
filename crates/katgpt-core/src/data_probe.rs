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

/// Pre-allocated scratch buffers for power-iteration stable-rank computation
/// and sink classifier bookkeeping.
///
/// Create once via [`StableRankScratch::new`] and reuse across calls via
/// [`StableRankScratch::ensure_capacity`] (head-dim-only) or
/// [`StableRankScratch::ensure_capacity_dn`] (head-dim + seq-len). The hot
/// path performs no heap allocation when the dimensions match the cache.
///
/// - `v`: current power-iteration vector, length `d` (smallest dim of `O`).
/// - `w`: next power-iteration vector (`O^T·O · v`), length `d`.
/// - `ov_buf`: per-row `O · v` matvec intermediate, length `n`.
/// - `col_sums`: attention column accumulator, length `n`.
///
/// The `ov_buf`/`col_sums` buffers were added by Issue 001 to eliminate the
/// two per-call `vec![0.0; n]` allocations that dominated the G3 latency
/// benchmark (`benches/sink_aware_latency_bench.rs`).
pub struct StableRankScratch {
    /// Current iterate (length `d`).
    pub v: Vec<f32>,
    /// Next iterate (length `d`).
    pub w: Vec<f32>,
    /// Per-row `O · v` matvec intermediate (length `n`). Lazily grown on
    /// first call to [`stable_rank_update_into`] for a given `n`.
    pub ov_buf: Vec<f32>,
    /// Attention column sums (length `n`). Lazily grown on first call to
    /// [`classify_all_sinks`] / [`apply_dual_policy_gate`] for a given `n`.
    pub col_sums: Vec<f32>,
    cached_d: usize,
    cached_n: usize,
}

impl StableRankScratch {
    /// Allocate scratch for power iteration on a `d × d` Gram matrix.
    ///
    /// The two `n`-length buffers (`ov_buf`, `col_sums`) are allocated empty
    /// and lazily grown on first use — callers that only ever use `d` (the
    /// historical use case for `stable_rank_update_into` on small matrices)
    /// pay no `n`-length cost up front.
    pub fn new(d: usize) -> Self {
        Self {
            v: vec![0.0; d],
            w: vec![0.0; d],
            ov_buf: Vec::new(),
            col_sums: Vec::new(),
            cached_d: d,
            cached_n: 0,
        }
    }

    /// Resize if head dim changed; no-op on the hot path.
    ///
    /// Backward-compatible signature: callers that don't track `n` continue
    /// to work. The `n`-length buffers (`ov_buf`, `col_sums`) are grown
    /// lazily by the consumers that need them.
    pub fn ensure_capacity(&mut self, d: usize) {
        if self.cached_d == d {
            return;
        }
        self.v.resize(d, 0.0);
        self.w.resize(d, 0.0);
        self.cached_d = d;
    }

    /// Resize both head dim `d` and seq len `n` if either changed. Use this
    /// when you know `n` up front (e.g. `apply_dual_policy_gate`) to keep the
    /// hot path allocation-free.
    pub fn ensure_capacity_dn(&mut self, d: usize, n: usize) {
        if self.cached_d != d {
            self.v.resize(d, 0.0);
            self.w.resize(d, 0.0);
            self.cached_d = d;
        }
        if self.cached_n != n {
            // Resize, preserving any existing prefix (cheap when shrinking
            // and the cached_n was larger).
            self.ov_buf.resize(n, 0.0);
            self.col_sums.resize(n, 0.0);
            self.cached_n = n;
        }
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
    scratch.ensure_capacity_dn(d, n);
    let v = &mut scratch.v[..d];
    let w = &mut scratch.w[..d];
    let ov_buf = &mut scratch.ov_buf[..n];

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

    // Issue 001 T5: cheap rank-1 probe. Compare the first and last rows of O
    // by cosine similarity. If they're near-parallel (cos > 0.95), O is very
    // likely rank-1 (a Broadcast head where every row is `a_s · v_s^T`).
    // This is O(d) work that lets us skip the O(n·d) power iteration in the
    // common case where Broadcast sinks dominate.
    //
    // False-positive cost: a matrix that happens to have O[0] ∥ O[n-1] but is
    // not rank-1 would be misclassified as rank-1. We accept this risk because
    // (a) the caller has already gated on `value_norm_ratio ∈ [0.5, 1.5]`
    // (broadcast window) before invoking stable-rank, and (b) the paper's
    // Broadcast signature is exactly "rows parallel to v_s". False negatives
    // are not possible — if cosine is low, we fall through to power iteration.
    if n >= 2 {
        let first = &o[0];
        let last = &o[n - 1];
        let dot_fl = simd::simd_dot_f32(first, last, d);
        let nf_sq = simd::simd_dot_f32(first, first, d);
        let nl_sq = simd::simd_dot_f32(last, last, d);
        if nf_sq > 0.0 && nl_sq > 0.0 {
            let cos_fl = dot_fl / (nf_sq.sqrt() * nl_sq.sqrt());
            if cos_fl > 0.95 {
                // Strong rank-1 signature. Conservative: σ_1² ≈ trace_f / 1,
                // stable rank = 1.0.
                return 1.0;
            }
        }
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
    // ov_buf is now reused from scratch (Issue 001 T4). The first call for a
    // new `n` pays one resize; subsequent calls are allocation-free.
    ov_buf.fill(0.0);

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

    // ── update_stable_rank ─────────────────────────────────
    //
    // Issue 001 T2: skip stable-rank computation when `value_norm_ratio` is
    // already decisive. Power iteration is the most expensive part of the
    // classifier; if the position is clearly NOP (ratio ≤ nop_max) or clearly
    // out-of-range for Broadcast (ratio outside [min, max]), stable rank
    // would not change the final classification. Only compute it when the
    // Broadcast window is reachable.
    let stable_rank_reachable = !degenerate
        && strength > cfg.sink_strength_threshold
        && value_norm_ratio > cfg.nop_value_ratio_max
        && value_norm_ratio >= cfg.broadcast_value_ratio_min
        && value_norm_ratio <= cfg.broadcast_value_ratio_max;
    let update_stable_rank = if stable_rank_reachable {
        match update_O {
            Some(o) if !o.is_empty() => stable_rank_update_into(o, scratch, 5),
            _ => f32::NAN,
        }
    } else {
        f32::NAN
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
    // Issue 001 T3: reuse `scratch.col_sums` instead of allocating per call.
    // The first call for a new `n` pays one resize; subsequent calls are
    // allocation-free.
    scratch.ensure_capacity_dn(values.first().map(|r| r.len()).unwrap_or(0), n);
    let col_sums = &mut scratch.col_sums;
    col_sums[..n].fill(0.0);
    for row in attn.iter() {
        let m = row.len().min(n);
        for j in 0..m {
            col_sums[j] += row[j];
        }
    }
    let inv_n = 1.0 / (n as f32);

    // Collect (j, strength_j) pairs that clear the threshold into a stack-backed
    // buffer, mirroring `classify_all_sinks_flat`. Dominant sinks are rare
    // (typically only a handful of positions qualify), so this avoids the
    // per-call length-n `Vec<f32>::collect()` the previous code did just to
    // release the `&mut scratch.col_sums` borrow before the
    // `classify_sink_at(&mut scratch)` call below. CAP=32 covers realistic
    // attention sinks; on overflow we spill to heap rather than dropping diag.
    const CANDIDATE_CAP: usize = 32;
    let mut candidates_stack: [(usize, f32); CANDIDATE_CAP] = [(0, 0.0); CANDIDATE_CAP];
    let mut candidates_heap: Vec<(usize, f32)> = Vec::new();
    let mut n_candidates = 0usize;
    let mut overflow = false;
    for j in 0..n {
        let strength_j = col_sums[j] * inv_n;
        if strength_j <= cfg.sink_strength_threshold {
            continue;
        }
        if n_candidates < CANDIDATE_CAP {
            candidates_stack[n_candidates] = (j, strength_j);
            n_candidates += 1;
        } else {
            if !overflow {
                candidates_heap.extend_from_slice(&candidates_stack);
                overflow = true;
            }
            candidates_heap.push((j, strength_j));
        }
    }

    // `col_sums` borrow ends here — safe to hand `scratch` to the classifier.
    let process: &[(usize, f32)] = if overflow {
        &candidates_heap
    } else {
        &candidates_stack[..n_candidates]
    };
    for &(j, strength_j) in process {
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

    let d = o[0].len();
    if d == 0 {
        return SinkKind::None;
    }

    // Issue 001 T1+T3: reuse all scratch buffers — no allocations on the hot
    // path after warmup.
    scratch.ensure_capacity_dn(d, n);

    // Find dominant sink column = argmax_j Σ_i attn[i][j].
    let col_sums = &mut scratch.col_sums;
    col_sums[..n].fill(0.0);
    for row in attn.iter() {
        let m = row.len().min(n);
        for j in 0..m {
            col_sums[j] += row[j];
        }
    }
    let (dominant_pos, _dominant_strength) = {
        let mut best_i = 0usize;
        let mut best_v = col_sums[0];
        for (i, &v) in col_sums[..n].iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        (best_i, best_v / (n as f32))
    };

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

/// Cached classification state for [`apply_dual_policy_gate_cached`].
///
/// Sinks in trained transformers are **stable across forward calls** — the
/// same head tends to be NOP-dominant or Broadcast-dominant across the whole
/// sequence. This struct lets callers classify once and reuse the decision
/// for `audit_every_n` subsequent calls, dropping steady-state overhead to
/// the cost of a copy + (conditional) scale.
///
/// ## Why this exists
///
/// The per-call [`apply_dual_policy_gate`] cannot beat a memcpy: it has to
/// scan `attn` (n² values) and `values` (n·d values) to classify, while
/// [`SinkAwarePolicy::Uniform`] is just a copy. Memory-bandwidth-bound, the
/// gap is structural — see Issue 001 § Latency analysis. The cached variant
/// is the production-realistic path: amortize the classifier over `N` calls.
///
/// ## Cadence
///
/// `audit_every_n` controls how often the classifier re-runs. Default 16
/// (≈6% steady-state overhead in the worst case where classification costs
/// as much as the copy itself; in practice far less because the cached
/// classify path uses the rank-1 cosine probe).
///
/// Set to `1` to disable caching (equivalent to [`apply_dual_policy_gate`]).
/// Set to `usize::MAX` to classify exactly once and never re-classify
/// (useful for frozen-model analysis).
#[derive(Debug, Clone)]
pub struct CachedSinkClassification {
    /// Classifier config to use when re-classifying.
    pub cfg: SinkClassifierConfig,
    /// Re-classify every `audit_every_n` calls. `0` and `1` both mean "every
    /// call" (no caching).
    pub audit_every_n: usize,
    /// Last computed kind. `None` until first classification.
    pub cached_kind: Option<SinkKind>,
    /// Calls since last classification.
    pub calls_since_audit: usize,
}

impl CachedSinkClassification {
    /// Create with the default classifier config and audit cadence 16.
    pub fn new() -> Self {
        Self {
            cfg: SinkClassifierConfig::default(),
            audit_every_n: 16,
            cached_kind: None,
            calls_since_audit: 0,
        }
    }

    /// Create with a specific config and cadence.
    pub fn with_config(cfg: SinkClassifierConfig, audit_every_n: usize) -> Self {
        Self {
            cfg,
            audit_every_n,
            cached_kind: None,
            calls_since_audit: 0,
        }
    }

    /// Force re-classification on the next call (e.g. after a model swap).
    pub fn invalidate(&mut self) {
        self.cached_kind = None;
        self.calls_since_audit = 0;
    }
}

impl Default for CachedSinkClassification {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached variant of [`apply_dual_policy_gate`].
///
/// On the audit cadence (`cached.audit_every_n`), runs the full classifier
/// and stores the result. On other calls, applies the cached decision:
///
/// - `Nop` → `out ← O · σ(gate_scale)`
/// - `Broadcast` / `None` → `out ← O` (copy)
///
/// Steady-state cost is a copy (or copy + scale) — same as
/// [`SinkAwarePolicy::Uniform`]. The classifier cost is amortized over
/// `audit_every_n` calls.
///
/// # Returns
/// The [`SinkKind`] applied on this call (the cached value, except on audit
/// calls where it is freshly computed).
pub fn apply_dual_policy_gate_cached(
    attn: &[Vec<f32>],
    values: &[Vec<f32>],
    o: &[Vec<f32>],
    gate_scale: f32,
    scratch: &mut StableRankScratch,
    cached: &mut CachedSinkClassification,
    out: &mut [Vec<f32>],
) -> SinkKind {
    let cadence = cached.audit_every_n.max(1);
    let needs_audit = cached.cached_kind.is_none()
        || cached.calls_since_audit >= cadence;

    if needs_audit {
        let policy = SinkAwarePolicy::DualPolicy(cached.cfg);
        let kind = apply_dual_policy_gate(
            attn, values, o, &policy, gate_scale, scratch, out,
        );
        cached.cached_kind = Some(kind);
        cached.calls_since_audit = 1;
        return kind;
    }

    cached.calls_since_audit += 1;
    match cached.cached_kind {
        Some(SinkKind::Nop) => {
            let g = sigmoid(gate_scale);
            scale_rows(o, g, out);
            SinkKind::Nop
        }
        Some(SinkKind::Broadcast) => {
            copy_rows(o, out);
            SinkKind::Broadcast
        }
        Some(SinkKind::None) | None => {
            copy_rows(o, out);
            SinkKind::None
        }
    }
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

/// Scale each row of `src` by `scale`, writing into `dst`.
///
/// Routes through [`simd::simd_fused_decay_write`] per row so the scaling is
/// vectorized (NEON/AVX2). Equivalent to `dst[i] = 0.0 * dst[i] + scale * src[i]`.
#[inline]
fn scale_rows(src: &[Vec<f32>], scale: f32, dst: &mut [Vec<f32>]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        let m = src[i].len().min(dst[i].len());
        crate::simd::simd_fused_decay_write(&mut dst[i][..m], 0.0, &src[i][..m], scale);
    }
}

// ════════════════════════════════════════════════════════════════════
// ── Plan 288: Flat `&[f32]` row-major variants ──────────────────────
// ════════════════════════════════════════════════════════════════════
//
// These mirror the `&[Vec<f32>]` API above but take flat row-major
// slices (`o[i*d..(i+1)*d]` = row `i`). They exist so that callers with
// contiguous tensors — notably `parallax_attn::tiled_attention_parallax_forward`
// and `funcattn::funcattn_forward`, which produce `output: &mut [f32]` and
// `v: &[f32]` — can invoke the sink-aware gate without materializing a
// `Vec<Vec<f32>>` wrapper (which would be an O(n·d) per-call allocation,
// breaking the zero-alloc property verified by Plan 287 G5).
//
// ## Correctness parity
//
// The arithmetic is identical to the `&[Vec<f32>]` variants — only the
// slicing changes (`row = &o[i*d..(i+1)*d]` vs `row = &o[i]`). T7 parity
// tests in `src/data_probe/sink_classify.rs` assert bit-identical
// `SinkKind` decisions on identical inputs.
//
// ## Why keep both layouts
//
// `&[Vec<f32>]` is diagnostic-friendly: callers building attention maps
// row-by-row (e.g. `examples/sink_phase_plot.rs`) already have that
// layout. `&[f32]` is forward-path-friendly: matches parallax/funcattn.
// We do NOT deprecate either. Pick whichever matches your data.

/// Compute the stable rank of `O = A · V` from a **flat** row-major `O`.
///
/// Flat-layout variant of [`stable_rank_update_into`]. Identical
/// arithmetic; rows are sliced as `o[i*d..(i+1)*d]` instead of `o[i]`.
///
/// # Arguments
/// * `o` — `(n, d)` flat row-major, length `n * d`.
/// * `n` — number of rows.
/// * `d` — row stride / head dim.
/// * `scratch` — power-iteration scratch (reused across calls).
/// * `n_iters` — power iteration count (5 is the plan default).
///
/// # Returns
/// Stable rank in `[1.0, rank(O)]`. Same semantics as
/// [`stable_rank_update_into`] — see that function's docs for the
/// algorithm, early-exits, and edge cases.
pub fn stable_rank_update_into_flat(
    o: &[f32],
    n: usize,
    d: usize,
    scratch: &mut StableRankScratch,
    n_iters: u8,
) -> f32 {
    if n == 0 || d == 0 {
        return 0.0;
    }
    debug_assert_eq!(o.len(), n * d, "flat stable_rank: o must be (n={}, d={}) flat", n, d);
    scratch.ensure_capacity_dn(d, n);
    let v = &mut scratch.v[..d];
    let w = &mut scratch.w[..d];
    let ov_buf = &mut scratch.ov_buf[..n];

    // trace(F) = Σ_i ‖row_i(O)‖². Single pass, contiguous reads.
    let mut trace_f = 0.0f32;
    for i in 0..n {
        let row = &o[i * d..(i + 1) * d];
        trace_f += simd::simd_dot_f32(row, row, d);
    }
    if trace_f <= 0.0 {
        return 0.0;
    }

    // Issue 001 T5: rank-1 cosine probe on first/last row. Same as the
    // Vec<Vec<f32>> variant — O(d) work that often skips the full power
    // iteration when Broadcast heads dominate.
    if n >= 2 {
        let first = &o[0..d];
        let last = &o[(n - 1) * d..n * d];
        let dot_fl = simd::simd_dot_f32(first, last, d);
        let nf_sq = simd::simd_dot_f32(first, first, d);
        let nl_sq = simd::simd_dot_f32(last, last, d);
        if nf_sq > 0.0 && nl_sq > 0.0 {
            let cos_fl = dot_fl / (nf_sq.sqrt() * nl_sq.sqrt());
            if cos_fl > 0.95 {
                return 1.0;
            }
        }
    }

    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    for x in v.iter_mut() {
        *x = inv_sqrt_d;
    }

    ov_buf.fill(0.0);
    let mut sigma1_sq = trace_f;
    let iters = n_iters.max(1) as usize;
    for _ in 0..iters {
        // (a) ov_buf[i] = O[i] · v
        for i in 0..n {
            let row = &o[i * d..(i + 1) * d];
            ov_buf[i] = simd::simd_dot_f32(row, v, d);
        }
        // (b) w = O^T · ov_buf = Σ_i ov_buf[i] · O[i]
        w.fill(0.0);
        for i in 0..n {
            let row = &o[i * d..(i + 1) * d];
            let c = ov_buf[i];
            simd::simd_fused_scale_acc(w, row, c, d);
        }

        let vtv = simd::simd_dot_f32(v, v, d);
        let vtw = simd::simd_dot_f32(v, w, d);
        if vtv <= 0.0 {
            break;
        }
        sigma1_sq = vtw / vtv;
        if sigma1_sq > 0.95 * trace_f {
            return 1.0;
        }

        let norm_w = (simd::simd_dot_f32(w, w, d)).max(1e-30).sqrt();
        let inv_norm = 1.0 / norm_w;
        for j in 0..d {
            v[j] = w[j] * inv_norm;
        }
    }

    if sigma1_sq <= 0.0 {
        return 0.0;
    }
    trace_f / sigma1_sq
}

/// Flat-layout variant of [`classify_sink_at`]. Identical decision rule;
/// `values` is `(n, d)` flat row-major and `update_O` carries its own
/// `(flat_o, n, d)` triple.
///
/// See [`classify_sink_at`] for argument semantics. The flat layout adds
/// `n` and `d` parameters; `update_O` becomes `Option<(&[f32], usize, usize)>`
/// = `(flat_O, n_O, d_O)`.
#[allow(non_snake_case)]
pub fn classify_sink_at_flat(
    position: usize,
    attn_column: &[f32],
    values: &[f32],
    n: usize,
    d: usize,
    update_O: Option<(&[f32], usize, usize)>,
    cfg: &SinkClassifierConfig,
    scratch: &mut StableRankScratch,
) -> SinkDiagnostic {
    debug_assert_eq!(
        values.len(),
        n * d,
        "flat classify_sink_at: values must be (n={}, d={}) flat",
        n,
        d
    );

    // ── Strength ───────────────────────────────────────────────
    let n_col = attn_column.len();
    let strength = if n_col == 0 {
        0.0
    } else {
        simd::simd_sum_f32(attn_column) / (n_col as f32)
    };

    // ── value_norm_ratio ───────────────────────────────────────
    let (value_norm_ratio, degenerate) = if n == 0 || d == 0 {
        (1.0, true)
    } else {
        // ‖v_s‖
        let v_s_norm = if position < n {
            let row = &values[position * d..(position + 1) * d];
            simd::simd_dot_f32(row, row, d).sqrt()
        } else {
            0.0
        };
        // mean_i ‖v_i‖ = sqrt(Σ_i ‖v_i‖² / n). Single pass, contiguous.
        let mut sum_sq = 0.0f32;
        for i in 0..n {
            let row = &values[i * d..(i + 1) * d];
            sum_sq += simd::simd_dot_f32(row, row, d);
        }
        if sum_sq == 0.0 {
            (1.0, true)
        } else {
            let mean_norm = (sum_sq / (n as f32)).sqrt();
            if mean_norm <= 0.0 {
                (1.0, true)
            } else {
                (v_s_norm / mean_norm, false)
            }
        }
    };

    // ── update_stable_rank ────────────────────────────────
    // Issue 001 T2: skip stable rank when value_norm_ratio is already decisive.
    let stable_rank_reachable = !degenerate
        && strength > cfg.sink_strength_threshold
        && value_norm_ratio > cfg.nop_value_ratio_max
        && value_norm_ratio >= cfg.broadcast_value_ratio_min
        && value_norm_ratio <= cfg.broadcast_value_ratio_max;
    let update_stable_rank = if stable_rank_reachable {
        match update_O {
            Some((o_flat, n_o, d_o)) if n_o > 0 && d_o > 0 => {
                stable_rank_update_into_flat(o_flat, n_o, d_o, scratch, 5)
            }
            _ => f32::NAN,
        }
    } else {
        f32::NAN
    };

    // ── Decision rule (Research 258 §2.1) ──────────────────────
    let kind = if degenerate {
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

/// Flat-layout variant of [`classify_all_sinks`].
///
/// `attn` is `(n, n)` flat row-major; `values` is `(n, d)` flat row-major.
/// Same semantics as the `&[Vec<f32>]` variant — caller-owned `out`, single
/// `col_sums` scratch reused across calls.
pub fn classify_all_sinks_flat(
    attn: &[f32],
    n: usize,
    values: &[f32],
    d: usize,
    cfg: &SinkClassifierConfig,
    scratch: &mut StableRankScratch,
    out: &mut Vec<SinkDiagnostic>,
) {
    if n == 0 {
        return;
    }
    debug_assert_eq!(attn.len(), n * n, "flat classify_all_sinks: attn must be (n, n)");
    debug_assert_eq!(values.len(), n * d, "flat classify_all_sinks: values must be (n, d)");

    scratch.ensure_capacity_dn(d, n);
    let col_sums = &mut scratch.col_sums;
    col_sums[..n].fill(0.0);
    // Column sums: c[j] = Σ_i attn[i*n + j]. Stride-d-1 access on the
    // inner read — cache-unfriendly but unavoidable for column sums; the
    // row-major layout of `attn` means each row is contiguous, so we walk
    // row-by-row and accumulate into col_sums[j].
    for i in 0..n {
        let row = &attn[i * n..(i + 1) * n];
        for j in 0..n {
            col_sums[j] += row[j];
        }
    }
    let inv_n = 1.0 / (n as f32);

    // First pass: collect (j, strength_j) pairs that clear the threshold into
    // a tiny stack-backed buffer. Almost always only a handful of positions
    // qualify (dominant sinks are rare), so this avoids the per-call
    // `Vec<f32>::collect()` of length `n` the previous code did just to release
    // the `col_sums` borrow before the `classify_sink_at_flat(&mut scratch)`
    // call below.
    //
    // CAP=32 covers realistic attention sinks; if more positions clear the
    // threshold we fall back to a heap allocation rather than silently dropping
    // diagnostics.
    const CANDIDATE_CAP: usize = 32;
    let mut candidates_stack: [(usize, f32); CANDIDATE_CAP] = [(0, 0.0); CANDIDATE_CAP];
    let mut candidates_heap: Vec<(usize, f32)> = Vec::new();
    let mut n_candidates = 0usize;
    let mut overflow = false;
    for j in 0..n {
        let strength_j = col_sums[j] * inv_n;
        if strength_j <= cfg.sink_strength_threshold {
            continue;
        }
        if n_candidates < CANDIDATE_CAP {
            candidates_stack[n_candidates] = (j, strength_j);
            n_candidates += 1;
        } else {
            if !overflow {
                // Spill the stack contents into the heap vec on first overflow,
                // then continue appending.
                candidates_heap.extend_from_slice(&candidates_stack);
                overflow = true;
            }
            candidates_heap.push((j, strength_j));
        }
    }

    // `col_sums` borrow ends here — safe to hand `scratch` to the classifier.
    let process: &[(usize, f32)] = if overflow {
        &candidates_heap
    } else {
        &candidates_stack[..n_candidates]
    };
    for &(j, strength_j) in process {
        let col = [strength_j];
        let diag = classify_sink_at_flat(j, &col, values, n, d, None, cfg, scratch);
        out.push(diag);
    }
}

/// Flat-layout variant of [`apply_dual_policy_gate`].
///
/// `attn` is `(n, n)` flat, `values`/`o`/`out` are `(n, d)` flat. Same
/// decision rule as the `&[Vec<f32>]` variant — see that function's docs.
///
/// This is the variant intended for direct integration with
/// `parallax_attn::tiled_attention_parallax_forward` and
/// `funcattn::funcattn_forward`, which produce flat `output: &mut [f32]`.
pub fn apply_dual_policy_gate_flat(
    attn: &[f32],
    values: &[f32],
    o: &[f32],
    n: usize,
    d: usize,
    policy: &SinkAwarePolicy,
    gate_scale: f32,
    scratch: &mut StableRankScratch,
    out: &mut [f32],
) -> SinkKind {
    let cfg = match policy {
        SinkAwarePolicy::Uniform => {
            copy_rows_flat(o, out, n * d);
            return SinkKind::None;
        }
        SinkAwarePolicy::DualPolicy(c) => *c,
    };

    if n == 0 || d == 0 {
        return SinkKind::None;
    }

    debug_assert_eq!(attn.len(), n * n, "flat gate: attn must be (n, n)");
    debug_assert_eq!(values.len(), n * d, "flat gate: values must be (n, d)");
    debug_assert_eq!(o.len(), n * d, "flat gate: o must be (n, d)");
    debug_assert_eq!(out.len(), n * d, "flat gate: out must be (n, d)");

    scratch.ensure_capacity_dn(d, n);

    // Dominant sink column = argmax_j Σ_i attn[i*n + j].
    let col_sums = &mut scratch.col_sums;
    col_sums[..n].fill(0.0);
    for i in 0..n {
        let row = &attn[i * n..(i + 1) * n];
        for j in 0..n {
            col_sums[j] += row[j];
        }
    }
    let (dominant_pos, _dominant_strength) = {
        let mut best_i = 0usize;
        let mut best_v = col_sums[0];
        for (i, &v) in col_sums[..n].iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        (best_i, best_v / (n as f32))
    };

    let col = [_dominant_strength];
    let diag = classify_sink_at_flat(
        dominant_pos,
        &col,
        values,
        n,
        d,
        Some((o, n, d)),
        &cfg,
        scratch,
    );

    match diag.kind {
        SinkKind::Nop => {
            let g = sigmoid(gate_scale);
            scale_rows_flat(o, g, out, n * d);
        }
        SinkKind::Broadcast | SinkKind::None => {
            copy_rows_flat(o, out, n * d);
        }
    }
    diag.kind
}

/// Flat-layout cached variant of [`apply_dual_policy_gate_cached`].
///
/// Same audit-cadence logic as the `&[Vec<f32>]` variant — see that
/// function's docs for the caching contract. Flat layout enables direct
/// composition with parallax/funcattn forward paths without `Vec<Vec<f32>>`
/// materialization.
pub fn apply_dual_policy_gate_cached_flat(
    attn: &[f32],
    values: &[f32],
    o: &[f32],
    n: usize,
    d: usize,
    gate_scale: f32,
    scratch: &mut StableRankScratch,
    cached: &mut CachedSinkClassification,
    out: &mut [f32],
) -> SinkKind {
    let cadence = cached.audit_every_n.max(1);
    let needs_audit = cached.cached_kind.is_none() || cached.calls_since_audit >= cadence;

    if needs_audit {
        let policy = SinkAwarePolicy::DualPolicy(cached.cfg);
        let kind = apply_dual_policy_gate_flat(
            attn, values, o, n, d, &policy, gate_scale, scratch, out,
        );
        cached.cached_kind = Some(kind);
        cached.calls_since_audit = 1;
        return kind;
    }

    cached.calls_since_audit += 1;
    let total = n * d;
    match cached.cached_kind {
        Some(SinkKind::Nop) => {
            let g = sigmoid(gate_scale);
            scale_rows_flat(o, g, out, total);
            SinkKind::Nop
        }
        Some(SinkKind::Broadcast) => {
            copy_rows_flat(o, out, total);
            SinkKind::Broadcast
        }
        Some(SinkKind::None) | None => {
            copy_rows_flat(o, out, total);
            SinkKind::None
        }
    }
}

// ── Flat helpers ───────────────────────────────────────────────

#[inline]
fn copy_rows_flat(src: &[f32], dst: &mut [f32], total: usize) {
    let m = src.len().min(dst.len()).min(total);
    dst[..m].copy_from_slice(&src[..m]);
}

#[inline]
fn scale_rows_flat(src: &[f32], scale: f32, dst: &mut [f32], total: usize) {
    let m = src.len().min(dst.len()).min(total);
    // simd_fused_decay_write(dst, 0.0, src, scale) = dst = 0·dst + scale·src
    // = scale·src. Single SIMD pass, no scalar fallback needed (simd
    // module already handles scalar platforms).
    simd::simd_fused_decay_write(&mut dst[..m], 0.0, &src[..m], scale);
}
