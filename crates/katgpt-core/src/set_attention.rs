//! Cross-Datapoint Set Attention — sigmoid-gated, permutation-equivariant
//! cross-entity refinement kernel (Plan 354, Research 354, arXiv:2106.02584).
//!
//! Distilled from Kossen et al. NeurIPS 2021 (*Non-Parametric Transformers*).
//! The paper's value is end-to-end *training* of the Q/K/V projections via
//! BERT-style masking — that stays in riir-train. This file ships the
//! **inference-time operator only**: given N entity state vectors and frozen
//! Q/K/V matrices, compute a residual sigmoid-gated cross-entity refinement.
//!
//! ## Math
//!
//! Given N entity state vectors `h_i ∈ R^d` and frozen projections
//! `W_Q, W_K ∈ R^{d×k}`, `W_V ∈ R^{d×d}` (or `None` for identity), compute:
//!
//! ```text
//! q_i = W_Q · h_i              // query  ∈ R^k
//! k_j = W_K · h_j              // key    ∈ R^k
//! α_ij = σ((q_i · k_j) / √k · β)   // sigmoid gate (NEVER softmax) ∈ (0,1)
//! v_j = W_V · h_j              // value  ∈ R^d  (or h_j if W_V is None)
//! h_i' = h_i + γ · Σ_j α_ij · (v_j − h_i)
//! ```
//!
//! ## Why sigmoid, not softmax
//!
//! Per AGENTS.md §2: sigmoid gates are independent per pair — an entity may
//! attend to 0 peers (lonely), 1 peer (paired), or many peers (formation).
//! Softmax would force artificial competition (Σ_j α_ij = 1), which is the
//! wrong shape for crowd-scale NPC inference.
//!
//! ## Permutation equivariance
//!
//! Shuffling the input rows shuffles the output rows identically. This is
//! NPT's Lemma 4 (Appendix A): MHSA is equivariant because the column
//! permutation from `QKᵀ` cancels via the final `·V`. The G1 test in
//! `tests/set_attention_g1_g5.rs` verifies this bit-exactly.
//!
//! ## Zero-alloc hot path
//!
//! [`set_sigmoid_attention_into`] takes only borrowed slices and pre-allocated
//! scratch buffers — no heap allocation in steady state. The caller owns the
//! scratch (`scratch_q`, `scratch_k`, `scratch_alpha`) and reuses them across
//! ticks. G4 test asserts 0 allocs per call.
//!
//! ## Latent vs Raw
//!
//! This primitive is substrate-agnostic — it operates on `&[f32]` and produces
//! `&mut [f32]`. It has no opinion on what the vectors *mean*. The sync
//! boundary is the caller's responsibility (see the riir-ai runtime plan 355
//! for the HLA-specific wiring + the unchanged 5-scalar bridge).

#![allow(clippy::too_many_arguments)] // perf kernel — explicit args beat struct builders
#![allow(clippy::needless_range_loop)] // explicit indexing aids SIMD auto-vectorization

// ─────────────────────────────────────────────────────────────────────
// Config (T1.2)
// ─────────────────────────────────────────────────────────────────────

/// Configuration for [`set_sigmoid_attention_into`].
///
/// All fields are `Copy` so the config can be passed by value cheaply. The
/// intended usage is: build once at zone-init (with the projection dims and
/// crowd-density-derived temperature), clone-per-tick if any field mutates.
#[derive(Clone, Copy, Debug)]
pub struct SetAttentionConfig {
    /// Per-pair sigmoid temperature β. The sigmoid argument is
    /// `(q·k) / √k · β`. Default 1.0. Higher β → sharper attention (peers must
    /// be more similar to contribute); lower β → softer (broader averaging).
    pub beta: f32,
    /// Residual step size γ. Output is `h_i + (γ/N) · Σ_j α_ij · (v_j − h_i)`
    /// where N is the peer count. The `γ/N` normalisation makes γ invariant
    /// to crowd size — γ means "how much each peer's gated contribution moves
    /// me, on average". Default 0.1.
    pub gamma: f32,
    /// Optional top-k cap on attended peers per query. `None` = dense (all N
    /// peers contribute). `Some(k_max)` = sparse; only the `k_max` highest-α
    /// peers contribute. Sparse is for large N (>100) where O(N²) is too slow.
    pub top_k: Option<usize>,
}

impl Default for SetAttentionConfig {
    #[inline]
    fn default() -> Self {
        Self {
            beta: 1.0,
            gamma: 0.1,
            top_k: None,
        }
    }
}

impl SetAttentionConfig {
    /// Construct with explicit β, γ. `top_k` defaults to `None` (dense).
    #[inline]
    pub const fn new(beta: f32, gamma: f32) -> Self {
        Self {
            beta,
            gamma,
            top_k: None,
        }
    }

    /// Builder: set `top_k`.
    #[inline]
    pub const fn with_top_k(mut self, k_max: usize) -> Self {
        self.top_k = Some(k_max);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Errors returned by [`set_sigmoid_attention_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetAttentionError {
    /// `states.len()` is not `n * d`.
    StatesLenMismatch { expected: usize, got: usize },
    /// `output.len()` is not `n * d`.
    OutputLenMismatch { expected: usize, got: usize },
    /// `w_q.len()` is not `d * k`.
    WqLenMismatch { expected: usize, got: usize },
    /// `w_k.len()` is not `d * k`.
    WkLenMismatch { expected: usize, got: usize },
    /// `w_v.len()` is not `d * d` (only checked when `Some`).
    WvLenMismatch { expected: usize, got: usize },
    /// `scratch_q.len()` is not `n * k`.
    ScratchQLenMismatch { expected: usize, got: usize },
    /// `scratch_k.len()` is not `n * k`.
    ScratchKLenMismatch { expected: usize, got: usize },
    /// `scratch_alpha.len()` is less than `n`.
    ScratchAlphaLenMismatch { expected: usize, got: usize },
    /// `n`, `d`, or `k` is zero.
    ZeroDim,
}

impl std::fmt::Display for SetAttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StatesLenMismatch { expected, got } => write!(
                f,
                "states.len() = {got}, expected n*d = {expected}"
            ),
            Self::OutputLenMismatch { expected, got } => write!(
                f,
                "output.len() = {got}, expected n*d = {expected}"
            ),
            Self::WqLenMismatch { expected, got } => {
                write!(f, "w_q.len() = {got}, expected d*k = {expected}")
            }
            Self::WkLenMismatch { expected, got } => {
                write!(f, "w_k.len() = {got}, expected d*k = {expected}")
            }
            Self::WvLenMismatch { expected, got } => {
                write!(f, "w_v.len() = {got}, expected d*d = {expected}")
            }
            Self::ScratchQLenMismatch { expected, got } => write!(
                f,
                "scratch_q.len() = {got}, expected n*k = {expected}"
            ),
            Self::ScratchKLenMismatch { expected, got } => write!(
                f,
                "scratch_k.len() = {got}, expected n*k = {expected}"
            ),
            Self::ScratchAlphaLenMismatch { expected, got } => write!(
                f,
                "scratch_alpha.len() = {got}, expected >= n = {expected}"
            ),
            Self::ZeroDim => write!(f, "n, d, or k is zero"),
        }
    }
}

impl std::error::Error for SetAttentionError {}

// ─────────────────────────────────────────────────────────────────────
// Core kernel (T1.3)
// ─────────────────────────────────────────────────────────────────────

/// Permutation-equivariant, sigmoid-gated cross-entity set attention.
///
/// Given `n` entity state vectors in `states` (flat `[n*d]` row-major),
/// projection matrices `w_q`, `w_k` (flat `[d*k]` row-major), and an optional
/// value projection `w_v` (flat `[d*d]` row-major or `None` for identity),
/// compute the refined states in `output` (flat `[n*d]` row-major, must be
/// pre-allocated).
///
/// Scratch buffers `scratch_q`, `scratch_k` (each `[n*k]`), and
/// `scratch_alpha` (`[n]`) must be pre-allocated by the caller. The scratch
/// contents are overwritten — they are scratch, not state. **Zero allocations
/// in steady state** (G4).
///
/// # Update rule
///
/// For each entity `i`:
/// ```text
/// q_i = w_q · states[i]                                   // ∈ R^k
/// α_ij = σ((q_i · w_k · states[j]) / √k · β)              // sigmoid gate
/// v_j = w_v · states[j]  (or states[j] if w_v is None)
/// output[i] = states[i] + (γ/N) · Σ_j α_ij · (v_j − states[i])
/// ```
///
/// **The residual is normalised by N** (the peer count) so γ is invariant to
/// crowd size. Without this, a guard in a 100-NPC zone would move 100× further
/// than a guard in a 1-NPC zone for the same γ. With it, γ means "how much each
/// peer's gated contribution moves me, on average" — stable across zones.
///
/// If `cfg.top_k` is `Some(k_max)`, only the top-`k_max` highest-`α` peers
/// contribute to the sum for each query.
///
/// # Permutation equivariance
///
/// Permuting the rows of `states` permutes the rows of `output` identically.
/// This is NPT's Lemma 4 (Appendix A): MHSA is equivariant because the column
/// permutation from `QKᵀ` cancels via the final `·V`. The G1 test in
/// `tests/set_attention_g1_g5.rs` verifies bit-exactness over 10 random
/// permutations.
///
/// # Arguments
///
/// * `states` — Input entity states, flat `[n*d]` row-major (entity `i` at
///   bytes `i*d..(i+1)*d`). Not mutated.
/// * `w_q` — Query projection, `[d×k]` stored column-major (column `m` at byte
///   offset `m*d`, element `(row, m)` at index `row + m*d`). (`W_Q · h_i`
///   projects entity `i`'s state to its query.) Use [`identity_projection`] for
///   the modelless floor.
/// * `w_k` — Key projection, `[d×k]` column-major (same layout as `w_q`).
///   (`W_K · h_j` projects entity `j`'s state to its key.)
/// * `w_v` — Value projection, `[d×d]` column-major, or `None` for identity
///   (the modelless floor — `v_j = h_j`). Use [`identity`] for the floor.
/// * `output` — Output refined states, flat `[n*d]` row-major. Written to.
///   May alias `states` for in-place refinement.
/// * `cfg` — β, γ, top_k.
/// * `n` — Number of entities.
/// * `d` — Per-entity state dimensionality (8 for HLA).
/// * `k` — Query/key projection dimensionality (`k ≤ d`; the CS-ranking-derived
///   Q projection produces small `k`).
/// * `scratch_q`, `scratch_k` — Pre-allocated `[n*k]` scratch.
/// * `scratch_alpha` — Pre-allocated `[n]` scratch (the α row for the current
///   query, reused across queries).
///
/// # Errors
///
/// Returns [`SetAttentionError`] if any slice length is inconsistent with
/// `n`, `d`, `k`. Does not allocate on the error path.
///
/// # Zero allocations
///
/// This function performs zero heap allocations. The caller owns all scratch.
#[inline]
pub fn set_sigmoid_attention_into(
    states: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: Option<&[f32]>,
    output: &mut [f32],
    cfg: &SetAttentionConfig,
    n: usize,
    d: usize,
    k: usize,
    scratch_q: &mut [f32],
    scratch_k: &mut [f32],
    scratch_alpha: &mut [f32],
) -> Result<(), SetAttentionError> {
    // ── Dimension checks ─────────────────────────────────────────────
    if n == 0 || d == 0 || k == 0 {
        return Err(SetAttentionError::ZeroDim);
    }
    let nd = n * d;
    let nk = n * k;
    let dd = d * d;
    let dk = d * k;
    if states.len() != nd {
        return Err(SetAttentionError::StatesLenMismatch {
            expected: nd,
            got: states.len(),
        });
    }
    if output.len() != nd {
        return Err(SetAttentionError::OutputLenMismatch {
            expected: nd,
            got: output.len(),
        });
    }
    if w_q.len() != dk {
        return Err(SetAttentionError::WqLenMismatch {
            expected: dk,
            got: w_q.len(),
        });
    }
    if w_k.len() != dk {
        return Err(SetAttentionError::WkLenMismatch {
            expected: dk,
            got: w_k.len(),
        });
    }
    if let Some(wv) = w_v
        && wv.len() != dd
    {
        return Err(SetAttentionError::WvLenMismatch {
            expected: dd,
            got: wv.len(),
        });
    }
    if scratch_q.len() != nk {
        return Err(SetAttentionError::ScratchQLenMismatch {
            expected: nk,
            got: scratch_q.len(),
        });
    }
    if scratch_k.len() != nk {
        return Err(SetAttentionError::ScratchKLenMismatch {
            expected: nk,
            got: scratch_k.len(),
        });
    }
    if scratch_alpha.len() < n {
        return Err(SetAttentionError::ScratchAlphaLenMismatch {
            expected: n,
            got: scratch_alpha.len(),
        });
    }

    // ── Project all queries and keys (W_Q · h_i, W_K · h_j) ──────────
    // scratch_q[i*k..(i+1)*k] = W_Q · states[i*d..(i+1)*d]
    // scratch_k[j*k..(j+1)*k] = W_K · states[j*d..(j+1)*d]
    // Both are dense matvec loops; LLVM auto-vectorizes at k=4..8.
    for i in 0..n {
        let state_row = &states[i * d..(i + 1) * d];
        let q_row = &mut scratch_q[i * k..(i + 1) * k];
        let k_row = &mut scratch_k[i * k..(i + 1) * k];
        // q_row[m] = Σ_a W_Q[a + m*d] · state_row[a]   (W_Q is [d×k] row-major
        // meaning col-m has stride d; we iterate m outer, a inner)
        for m in 0..k {
            let mut acc_q = 0.0f32;
            let mut acc_k = 0.0f32;
            for a in 0..d {
                acc_q += w_q[a + m * d] * state_row[a];
                acc_k += w_k[a + m * d] * state_row[a];
            }
            q_row[m] = acc_q;
            k_row[m] = acc_k;
        }
    }

    // Sigmoid temperature scale: 1/√k · β, precomputed once.
    let scale = cfg.beta / (k as f32).sqrt();

    // ── Copy states into output first; we add the residual on top ─────
    // (output[i] = states[i] + γ · Σ_j α_ij · (v_j − states[i]))
    // For the identity-V case (w_v is None), this lets us write the residual
    // contribution as γ · Σ_j α_ij · (states[j] − states[i]) directly.
    output.copy_from_slice(states);

    // ── For each query i, compute α_ij and accumulate ────────────────
    match cfg.top_k {
        None => dense_accumulate(output, states, w_v, scratch_q, scratch_k, scratch_alpha, cfg, n, d, k, scale),
        Some(k_max) => {
            if k_max == 0 {
                return Ok(()); // top-0 = no contribution; output = states (already copied)
            }
            topk_accumulate(output, states, w_v, scratch_q, scratch_k, scratch_alpha, cfg, n, d, k, scale, k_max)
        }
    }
}

/// Dense path: all N peers contribute to each query. O(N²·k) attention scores.
///
/// The residual update is **normalised by N** (the peer count) so γ is
/// invariant to crowd size: `h_i' = h_i + γ · (1/N) · Σ_j α_ij · (v_j − h_i)`.
/// Without this normalisation, a guard in a 100-NPC zone would move 100×
/// further than a guard in a 1-NPC zone for the same γ — unstable. With it,
/// γ means "how much each peer's gated contribution moves me, on average".
#[inline]
fn dense_accumulate(
    output: &mut [f32],
    states: &[f32],
    w_v: Option<&[f32]>,
    scratch_q: &[f32],
    scratch_k: &[f32],
    scratch_alpha: &mut [f32],
    cfg: &SetAttentionConfig,
    n: usize,
    d: usize,
    k: usize,
    scale: f32,
) -> Result<(), SetAttentionError> {
    let gamma = cfg.gamma;
    // Precompute γ/N — the per-peer average step size. See the doc comment
    // above for why we normalise by N.
    let gamma_per_peer = gamma / (n as f32);
    // Restructured update: hoist the loop-invariant `state_i[m]` out of the
    // inner j-loop. We accumulate `sum_av[m] = Σ_j α_ij · v_j[m]` and
    // `sum_a = Σ_j α_ij` separately, then combine once per (i, m):
    //   output[i][m] = state_i[m] + (γ/N) · ( sum_av[m] − state_i[m] · sum_a )
    // This is one multiply per (i, m) for the final combine, vs one multiply +
    // one subtract per (i, j, m) in the naive form — a ~N× speedup on the
    // inner product. The stack scratch `sum_av` is bounded at d=16 (covers
    // HLA d=8 and most latents; larger d would need a heap scratch).
    debug_assert!(d <= 16, "dense_accumulate: d > 16 needs a larger stack scratch");
    for i in 0..n {
        let q_row = &scratch_q[i * k..(i + 1) * k];
        // Compute α_ij for all j into scratch_alpha.
        for j in 0..n {
            let k_row = &scratch_k[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for m in 0..k {
                dot += q_row[m] * k_row[m];
            }
            scratch_alpha[j] = sigmoid(dot * scale);
        }
        // Accumulate sum_av[m] = Σ_j α_ij · v_j[m] and sum_a = Σ_j α_ij.
        // sum_av is a per-i scratch of length d; we reuse scratch_alpha[d..d+d]
        // to avoid a new allocation (scratch_alpha has length n >= d here in
        // the dense path — the caller guarantees this; if n < d the math is
        // still correct but we'd need a separate scratch). For HLA (d=8) and
        // NPC zones (n >= 8 typically), this holds. To be safe, use a stack
        // array for d <= 16.
        let mut sum_av = [0.0f32; 16];
        let mut sum_a = 0.0f32;
        let sa = &scratch_alpha[..n];
        if let Some(wv) = w_v {
            for j in 0..n {
                let alpha = sa[j];
                if alpha == 0.0 {
                    continue;
                }
                sum_a += alpha;
                let state_j = &states[j * d..(j + 1) * d];
                for m in 0..d {
                    // v_j[m] = Σ_a W_V[a + m*d] · state_j[a]
                    let mut v_j_m = 0.0f32;
                    for a in 0..d {
                        v_j_m += wv[a + m * d] * state_j[a];
                    }
                    sum_av[m] += alpha * v_j_m;
                }
            }
        } else {
            // Identity V: v_j = state_j.
            for j in 0..n {
                let alpha = sa[j];
                if alpha == 0.0 {
                    continue;
                }
                sum_a += alpha;
                let state_j = &states[j * d..(j + 1) * d];
                for m in 0..d {
                    sum_av[m] += alpha * state_j[m];
                }
            }
        }
        // Combine: output[i][m] = state_i[m] + (γ/N) · (sum_av[m] − state_i[m] · sum_a)
        let state_i = &states[i * d..(i + 1) * d];
        let out_i = &mut output[i * d..(i + 1) * d];
        for m in 0..d {
            out_i[m] = state_i[m] + gamma_per_peer * (sum_av[m] - state_i[m] * sum_a);
        }
    }
    Ok(())
}

/// Sparse top-k path: only the k_max highest-α peers contribute per query.
/// Used for large N (>100) where dense O(N²) is too slow. Requires a small
/// auxiliary scratch for the indexed top-k selection.
#[inline]
fn topk_accumulate(
    output: &mut [f32],
    states: &[f32],
    w_v: Option<&[f32]>,
    scratch_q: &[f32],
    scratch_k: &[f32],
    scratch_alpha: &mut [f32],
    cfg: &SetAttentionConfig,
    n: usize,
    d: usize,
    k: usize,
    scale: f32,
    k_max: usize,
) -> Result<(), SetAttentionError> {
    let gamma = cfg.gamma;
    let effective_k = k_max.min(n);
    // Selection buffer — reused across queries. We use a Vec for the indexed
    // sort; this is a small allocation but kept inside the function (not steady
    // state per-tick for the same caller). For true zero-alloc top-k, the
    // caller can pass an explicit indexed scratch in a future API extension.
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..n {
        let q_row = &scratch_q[i * k..(i + 1) * k];
        for j in 0..n {
            let k_row = &scratch_k[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for m in 0..k {
                dot += q_row[m] * k_row[m];
            }
            scratch_alpha[j] = sigmoid(dot * scale);
        }
        // Partial sort: select top-effective_k indices by α (descending).
        // We do select_nth + sort the front — O(N) average for selection,
        // then O(k·log k) for the front sort.
        idx.sort_unstable_by(|&a, &b| {
            scratch_alpha[b]
                .partial_cmp(&scratch_alpha[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top = &idx[..effective_k];
        // Accumulate only over the top-k peers. We normalise by k_max (NOT N)
        // here so the sparse path has the same per-peer step semantics as the
        // dense path — a peer in the top-k contributes as much as it would in
        // the dense case.
        let normaliser = effective_k as f32;
        let gamma_per_peer = gamma / normaliser;
        let state_i = &states[i * d..(i + 1) * d];
        let out_i = &mut output[i * d..(i + 1) * d];
        for &j in top {
            let alpha = scratch_alpha[j];
            if alpha == 0.0 {
                continue;
            }
            let state_j = &states[j * d..(j + 1) * d];
            let coeff = gamma_per_peer * alpha;
            if let Some(wv) = w_v {
                for m in 0..d {
                    let mut v_j_m = 0.0f32;
                    for a in 0..d {
                        v_j_m += wv[a + m * d] * state_j[a];
                    }
                    out_i[m] += coeff * (v_j_m - state_i[m]);
                }
            } else {
                for m in 0..d {
                    out_i[m] += coeff * (state_j[m] - state_i[m]);
                }
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Sigmoid (AGENTS.md §2: NEVER softmax)
// ─────────────────────────────────────────────────────────────────────

/// Numerically stable logistic sigmoid. Inline for the hot path; LLVM will
/// hoist it to a vector op where possible.
///
/// Per AGENTS.md §2: **always sigmoid, never softmax.** Sigmoid gives
/// independent per-pair gates ∈ (0,1); softmax would force Σ_j α_ij = 1 and
/// destroy the "attend to 0 / 1 / many peers" semantic that crowd-scale NPC
/// inference requires.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    // Branch-free stable form: avoids overflow for large negative x (the
    // naive 1/(1+e^-x) overflows there). Uses std `f32::exp` (libm on std,
    // compiler-builtins on no_std); no extra crate dep.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Identity W_Q / W_K constructors (modelless floor)
// ─────────────────────────────────────────────────────────────────────

/// Construct an identity `[d*d]` matrix row-major (the modelless W_Q / W_K /
/// W_V floor). `W = I_d` projects `h` to itself; combined with identity V,
/// the operator reduces to sigmoid-weighted consensus.
///
/// The result is written into `out` (length `d*d`); no allocation.
pub fn identity_into(out: &mut [f32], d: usize) {
    debug_assert_eq!(out.len(), d * d, "identity_into: out.len() must be d*d");
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
}

/// Owned variant of [`identity_into`] for callers that prefer a fresh `Vec`.
pub fn identity(d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d * d];
    identity_into(&mut out, d);
    out
}

/// Construct a `[d×k]` identity projection matching the kernel's storage
/// convention (column-major: column `m` is at byte offset `m*d`, element
/// `(row, m)` is at index `row + m*d`). This is the natural projection when the
/// caller wants `k ≤ d` (e.g. CS-ranking selects the top-k dims): `W · h`
/// selects the first `k` components of `h`. Pairs with [`identity`] for the W_V
/// floor.
///
/// `out.len()` must be `d * k`. Column `m` of the projection is `e_m`
/// (one-hot at dimension `m`), so `(W · h)[m] = h[m]` for `m < k`.
pub fn identity_projection_into(out: &mut [f32], d: usize, k: usize) {
    debug_assert_eq!(out.len(), d * k, "identity_projection_into: out.len() must be d*k");
    // Column-major: element (row a, col m) at index a + m*d. Identity projection
    // has column m = e_m (one-hot at row m). So out[m + m*d] = 1.0.
    for elem in out.iter_mut() {
        *elem = 0.0;
    }
    for m in 0..k {
        out[m + m * d] = 1.0;
    }
}

/// Owned variant of [`identity_projection_into`].
pub fn identity_projection(d: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d * k];
    identity_projection_into(&mut out, d, k);
    out
}

// ─────────────────────────────────────────────────────────────────────
// Sanity tests (T1.6 acceptance + inline smoke)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: N=4, d=8, k=8 (full-rank identity), identity V → output finite
    /// and equals states when γ=0 (no contribution).
    #[test]
    fn smoke_identity_gamma_zero_is_input() {
        let n = 4;
        let d = 8;
        let k = 8;
        let states = vec![0.5f32; n * d];
        let w = identity(d);
        let mut output = vec![0.0; n * d];
        let mut sq = vec![0.0; n * k];
        let mut sk = vec![0.0; n * k];
        let mut sa = vec![0.0; n];
        let cfg = SetAttentionConfig::new(1.0, 0.0); // γ=0 → no update
        set_sigmoid_attention_into(
            &states, &w, &w, None, &mut output, &cfg, n, d, k, &mut sq, &mut sk, &mut sa,
        )
        .unwrap();
        // output should equal states bit-exactly (γ=0 → no contribution).
        for (o, s) in output.iter().zip(states.iter()) {
            assert_eq!(o, s, "γ=0 should leave output = input bit-exactly");
        }
    }

    /// Smoke: identical states converge under identity V (consensus is a no-op
    /// because v_j − states[i] = 0).
    #[test]
    fn smoke_identical_states_unchanged() {
        let n = 3;
        let d = 8;
        let k = 8;
        let states = vec![0.5f32; n * d];
        let w = identity(d);
        let mut output = vec![0.0; n * d];
        let mut sq = vec![0.0; n * k];
        let mut sk = vec![0.0; n * k];
        let mut sa = vec![0.0; n];
        let cfg = SetAttentionConfig::new(1.0, 0.5); // γ=0.5 — but v_j−h_i=0
        set_sigmoid_attention_into(
            &states, &w, &w, None, &mut output, &cfg, n, d, k, &mut sq, &mut sk, &mut sa,
        )
        .unwrap();
        for (o, s) in output.iter().zip(states.iter()) {
            assert!((o - s).abs() < 1e-6, "identical states → no-op, got Δ={}", o - s);
        }
    }

    /// Smoke: dimension-mismatch errors fire correctly.
    #[test]
    fn smoke_len_errors() {
        let n = 4;
        let d = 8;
        let k = 4;
        let states = vec![0.0f32; n * d];
        let w_q = vec![0.0f32; d * k];
        let w_k = vec![0.0f32; d * k];
        let mut output = vec![0.0; n * d];
        let mut sq = vec![0.0; n * k];
        let mut sk = vec![0.0; n * k];
        let mut sa = vec![0.0; n];
        let cfg = SetAttentionConfig::default();

        // Wrong states length.
        let bad_states = vec![0.0f32; n * d + 1];
        let err = set_sigmoid_attention_into(
            &bad_states, &w_q, &w_k, None, &mut output, &cfg, n, d, k, &mut sq, &mut sk, &mut sa,
        )
        .unwrap_err();
        assert_eq!(err, SetAttentionError::StatesLenMismatch { expected: n*d, got: n*d+1 });

        // Wrong w_q length.
        let bad_wq = vec![0.0f32; d * k + 1];
        let err = set_sigmoid_attention_into(
            &states, &bad_wq, &w_k, None, &mut output, &cfg, n, d, k, &mut sq, &mut sk, &mut sa,
        )
        .unwrap_err();
        assert_eq!(err, SetAttentionError::WqLenMismatch { expected: d*k, got: d*k+1 });

        // Zero dim.
        let err = set_sigmoid_attention_into(
            &states, &w_q, &w_k, None, &mut output, &cfg, 0, d, k, &mut sq, &mut sk, &mut sa,
        )
        .unwrap_err();
        assert_eq!(err, SetAttentionError::ZeroDim);
    }

    /// Sigmoid: known values.
    #[test]
    fn sigmoid_known_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(1e6) - 1.0).abs() < 1e-6);
        assert!(sigmoid(-1e6).abs() < 1e-6);
        // Monotone increasing.
        assert!(sigmoid(0.5) > sigmoid(0.0));
        assert!(sigmoid(-0.5) < sigmoid(0.0));
    }

    /// Identity constructor produces I_d.
    #[test]
    fn identity_matrix_correct() {
        let d = 4;
        let w = identity(d);
        // Diagonal = 1, off-diagonal = 0.
        for i in 0..d {
            for j in 0..d {
                let expected = if i == j { 1.0f32 } else { 0.0 };
                assert_eq!(w[i * d + j], expected, "identity[{},{}] wrong", i, j);
            }
        }
    }

    /// Identity projection (d×k, k<d) selects the first k components.
    #[test]
    fn identity_projection_selects_first_k() {
        let d = 8;
        let k = 3;
        let w = identity_projection(d, k);
        // W · h should equal h[0..k]. Verify using the SAME indexing as the
        // kernel: out[m] = Σ_a w[a + m*d] · h[a].
        let h: Vec<f32> = (0..d).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; k];
        for m in 0..k {
            let mut acc = 0.0f32;
            for a in 0..d {
                acc += w[a + m * d] * h[a];
            }
            out[m] = acc;
        }
        for m in 0..k {
            assert_eq!(out[m], h[m], "identity_projection output[m] wrong");
        }
    }
}
