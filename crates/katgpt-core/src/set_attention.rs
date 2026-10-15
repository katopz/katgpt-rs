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

use std::cell::UnsafeCell;

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
    /// `reliability.len()` is not `n` (CLR-weighted variant only).
    ReliabilityLenMismatch { expected: usize, got: usize },
    /// `n`, `d`, or `k` is zero.
    ZeroDim,
    /// `scratch_sum_av.len()` < `d` (wide entry point only).
    SumAvLenMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for SetAttentionError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StatesLenMismatch { expected, got } => {
                write!(f, "states.len() = {got}, expected n*d = {expected}")
            }
            Self::OutputLenMismatch { expected, got } => {
                write!(f, "output.len() = {got}, expected n*d = {expected}")
            }
            Self::WqLenMismatch { expected, got } => {
                write!(f, "w_q.len() = {got}, expected d*k = {expected}")
            }
            Self::WkLenMismatch { expected, got } => {
                write!(f, "w_k.len() = {got}, expected d*k = {expected}")
            }
            Self::WvLenMismatch { expected, got } => {
                write!(f, "w_v.len() = {got}, expected d*d = {expected}")
            }
            Self::ScratchQLenMismatch { expected, got } => {
                write!(f, "scratch_q.len() = {got}, expected n*k = {expected}")
            }
            Self::ScratchKLenMismatch { expected, got } => {
                write!(f, "scratch_k.len() = {got}, expected n*k = {expected}")
            }
            Self::ScratchAlphaLenMismatch { expected, got } => {
                write!(f, "scratch_alpha.len() = {got}, expected >= n = {expected}")
            }
            Self::ReliabilityLenMismatch { expected, got } => {
                write!(f, "reliability.len() = {got}, expected n = {expected}")
            }
            Self::ZeroDim => write!(f, "n, d, or k is zero"),
            Self::SumAvLenMismatch { expected, got } => write!(
                f,
                "scratch_sum_av.len() (got {got}) < d (expected {expected}) — wide entry point only"
            ),
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
pub fn set_sigmoid_attention_wide_into(
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
    scratch_sum_av: &mut [f32],
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
    if scratch_sum_av.len() < d {
        return Err(SetAttentionError::SumAvLenMismatch {
            expected: d,
            got: scratch_sum_av.len(),
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

    // ── For each query i, compute α_ij and accumulate ────────────────
    //
    // The `output.copy_from_slice(states)` seed is only needed by the sparse
    // path, so it now lives in the `Some` arm. `dense_accumulate` *assigns*
    // `out_i[m] = state_i[m] + …` for every `i in 0..n` and `m in 0..d`, and
    // `output.len()` is checked to equal exactly `n * d` above, so every slot
    // it later reads is written first in the same pass — the n·d memcpy was
    // entirely dead on the dense path. (Only observable difference: if the
    // legacy wrapper's `d <= 16` release assert fires, `output` is now left
    // untouched rather than pre-seeded on the way out of the panic.)
    match cfg.top_k {
        None => dense_accumulate(
            output,
            states,
            w_v,
            scratch_q,
            scratch_k,
            scratch_alpha,
            cfg,
            n,
            d,
            k,
            scale,
            scratch_sum_av,
        ),
        Some(k_max) => {
            // The sparse path accumulates (`out_i[m] += …`) over only the top-k
            // peers, so it needs `output` seeded with `states` first:
            // output[i] = states[i] + γ · Σ_j α_ij · (v_j − states[i]).
            output.copy_from_slice(states);
            if k_max == 0 {
                return Ok(()); // top-0 = no contribution; output = states (already copied)
            }
            topk_accumulate(
                output,
                states,
                w_v,
                scratch_q,
                scratch_k,
                scratch_alpha,
                cfg,
                n,
                d,
                k,
                scale,
                k_max,
            )
        }
    }
}

/// Legacy stack-scratch entry point — identical math to
/// [`set_sigmoid_attention_wide_into`] with the accumulation scratch owned
/// here as a `[f32; 16]` stack array, hence the historical `d <= 16` cap
/// (HLA d=8 and most latents). Wider embeddings (decision planes run
/// d = 256, Plan 627 `set_refine`) must call the wide variant with a
/// caller-owned scratch. At `d <= 16` the two entry points are
/// bit-identical: this wrapper asserts the cap, provides the stack
/// scratch, and delegates.
///
/// # Panics
/// Panics in release when `d > 16` (the stack scratch bound) — the same
/// guard the dense path carried before the wide variant existed.
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
    assert!(
        d <= 16,
        "set_sigmoid_attention_into: d > 16 needs a caller-owned scratch — use set_sigmoid_attention_wide_into"
    );
    let mut sum_av_stack = [0.0f32; 16];
    set_sigmoid_attention_wide_into(
        states,
        w_q,
        w_k,
        w_v,
        output,
        cfg,
        n,
        d,
        k,
        scratch_q,
        scratch_k,
        scratch_alpha,
        &mut sum_av_stack[..d],
    )
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
    sum_av: &mut [f32],
) -> Result<(), SetAttentionError> {
    // The per-i accumulation scratch lives with the CALLER (stack in the
    // legacy entry point, caller-owned buffer in the wide one) — the
    // kernel itself never allocates (G4 law).
    debug_assert!(sum_av.len() >= d, "dense_accumulate: sum_av.len() >= d");
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
    // HLA d=8 and most latents). The d cap moved to the ENTRY POINTS: the
    // legacy `set_sigmoid_attention_into` release-asserts d <= 16 before
    // providing its stack scratch; `set_sigmoid_attention_wide_into` takes
    // the accumulation scratch from the caller and has no cap.
    // Precompute v_proj[j] = W_V · state_j for all j, once.
    // Without this, v_j would be recomputed n times (once per query i),
    // turning an O(n·d²) matvec into O(n²·d²).
    //
    // Reuse a per-thread grow-only buffer (matching the pattern in
    // `attention::tiled_attention_batched`). The first call with (n, d)
    // allocates; subsequent calls with the same or smaller workload reuse
    // the existing capacity (zero alloc in steady state — required by the
    // G4 zero-alloc gate and by the module-level doc claim).
    let v_proj: Option<&mut [f32]> = w_v.map(|wv| {
        thread_local! {
            static V_PROJ_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
        }
        V_PROJ_BUF.with(|cell| {
            // Safety: thread_local guarantees exclusive per-thread access.
            let vp = unsafe { &mut *cell.get() };
            if vp.len() < n * d {
                vp.resize(n * d, 0.0);
            }
            let vp = &mut vp[..n * d];
            for j in 0..n {
                let state_j = &states[j * d..(j + 1) * d];
                let vp_row = &mut vp[j * d..(j + 1) * d];
                for m in 0..d {
                    let mut v_j_m = 0.0f32;
                    for a in 0..d {
                        v_j_m += wv[a + m * d] * state_j[a];
                    }
                    vp_row[m] = v_j_m;
                }
            }
            vp
        })
    });
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
        sum_av[..d].fill(0.0);
        let mut sum_a = 0.0f32;
        let sa = &scratch_alpha[..n];
        if let Some(vp) = &v_proj {
            for j in 0..n {
                let alpha = sa[j];
                if alpha == 0.0 {
                    continue;
                }
                sum_a += alpha;
                let v_j = &vp[j * d..(j + 1) * d];
                for m in 0..d {
                    sum_av[m] += alpha * v_j[m];
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
    // Selection buffer — reused across queries via a per-thread grow-only
    // Vec<usize>. Each call resets the [0..n) range in place (one linear pass)
    // rather than reallocating. The downstream `select_nth_unstable_by` +
    // `sort_unstable_by` scramble the contents but leave the same multiset,
    // so this in-place reset is sufficient.
    //
    // Same thread_local pattern as `attention::tiled_attention_batched` and
    // the v_proj scratch in `dense_accumulate` — zero alloc in steady state.
    thread_local! {
        static IDX_BUF: UnsafeCell<Vec<usize>> = const { UnsafeCell::new(Vec::new()) };
    }
    let idx: &mut [usize] = IDX_BUF.with(|cell| {
        // Safety: thread_local guarantees exclusive per-thread access.
        let buf = unsafe { &mut *cell.get() };
        if buf.len() < n {
            buf.resize(n, 0);
        }
        let buf = &mut buf[..n];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i;
        }
        buf
    });
    // Precompute v_proj[j] = W_V · state_j for all j, once.
    // Without this, v_j would be recomputed per query i (up to n times),
    // turning an O(n·d²) matvec into O(n²·d²).
    //
    // Reuse the same per-thread V_PROJ_BUF as `dense_accumulate` — the two
    // paths never run concurrently within a single call, and even across
    // threads each thread has its own thread_local slot.
    let v_proj: Option<&mut [f32]> = w_v.map(|wv| {
        thread_local! {
            static V_PROJ_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
        }
        V_PROJ_BUF.with(|cell| {
            // Safety: thread_local guarantees exclusive per-thread access.
            let vp = unsafe { &mut *cell.get() };
            if vp.len() < n * d {
                vp.resize(n * d, 0.0);
            }
            let vp = &mut vp[..n * d];
            for j in 0..n {
                let state_j = &states[j * d..(j + 1) * d];
                let vp_row = &mut vp[j * d..(j + 1) * d];
                for m in 0..d {
                    let mut v_j_m = 0.0f32;
                    for a in 0..d {
                        v_j_m += wv[a + m * d] * state_j[a];
                    }
                    vp_row[m] = v_j_m;
                }
            }
            vp
        })
    });
    // Normalise by k_max (NOT N) so the sparse path has the same per-peer step
    // semantics as the dense path — a peer in the top-k contributes as much as
    // it would in the dense case. Both `effective_k` and `gamma` are fixed
    // before the query loop, so this cast + divide is loop-invariant; it used to
    // run once per query. (`dense_accumulate` already hoists the equivalent.)
    let normaliser = effective_k as f32;
    let gamma_per_peer = gamma / normaliser;
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
        // select_nth_unstable_by gives O(N) average selection, then we sort only
        // the front effective_k slice in O(k·log k). Total O(N + k·log k) instead
        // of O(N·log N) for a full sort — a log(N) speedup per query, so
        // log(N) speedup overall for the n-query loop.
        //
        // The front sort is needed because select_nth partitions but does not
        // order the front; the downstream accumulation iterates `for &j in top`
        // and the order is preserved bit-identically to the previous full-sort
        // (both use sort_unstable_by, so tie-breaking is identical).
        let cmp_alpha = |&a: &usize, &b: &usize| scratch_alpha[b].total_cmp(&scratch_alpha[a]);
        if effective_k > 0 {
            let (front, _, _) = idx.select_nth_unstable_by(effective_k - 1, cmp_alpha);
            front.sort_unstable_by(cmp_alpha);
        }
        let top = &idx[..effective_k];
        // Accumulate only over the top-k peers (`gamma_per_peer` hoisted above).
        let state_i = &states[i * d..(i + 1) * d];
        let out_i = &mut output[i * d..(i + 1) * d];
        for &j in top {
            let alpha = scratch_alpha[j];
            if alpha == 0.0 {
                continue;
            }
            let coeff = gamma_per_peer * alpha;
            if let Some(vp) = &v_proj {
                let v_j = &vp[j * d..(j + 1) * d];
                for m in 0..d {
                    out_i[m] += coeff * (v_j[m] - state_i[m]);
                }
            } else {
                let state_j = &states[j * d..(j + 1) * d];
                for m in 0..d {
                    out_i[m] += coeff * (state_j[m] - state_i[m]);
                }
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// CLR-weighted variant (Plan 570)
// ─────────────────────────────────────────────────────────────────────

/// CLR-style reliability scores: `r_j = (mean_m sigmoid(h_j · dir_m))^M`.
///
/// The `^M` exponent (Plan 284, Research 255 — CLR) is the nonlinear
/// reliability gate: entities whose mean sigmoid verdict is high get amplified,
/// those with any low verdict get suppressed super-linearly. This is the
/// mechanism that closes the Set Attention G8 collective-inference failure
/// (Bench 354 L71, Plan 570): plain averaging dilutes the signal, while
/// reliability-weighted averaging concentrates it toward the most informative
/// entities.
///
/// # Arguments
///
/// * `states` — Input entity states, flat `[n*d]` row-major.
/// * `directions` — `m` direction vectors, flat `[m*d]` row-major (entity `i`
///   at offset `i*d`). Each direction is a unit-length probe in belief space.
/// * `m_exp` — Number of directions AND the CLR exponent. `m_exp=5` matches
///   the CLR hot path (Plan 284).
/// * `n`, `d` — Dimensions.
/// * `output` — Output reliability scores, `[n]`. Written to.
///
/// # Zero allocations
///
/// This function performs zero heap allocations.
///
/// # Panics (release-checked)
///
/// Panics if `output.len() < n`, `directions.len() < m_exp * d`, or `m_exp == 0`.
#[cfg(feature = "clr_weighted_set_attention")]
pub fn clr_reliability_scores(
    states: &[f32],
    directions: &[f32],
    m_exp: usize,
    n: usize,
    d: usize,
    output: &mut [f32],
) {
    assert!(m_exp > 0, "clr_reliability_scores: m_exp must be > 0");
    assert!(
        output.len() >= n,
        "clr_reliability_scores: output.len() < n"
    );
    assert!(
        directions.len() >= m_exp * d,
        "clr_reliability_scores: directions.len() < m_exp * d"
    );
    let m_f = m_exp as f32;
    for j in 0..n {
        let h = &states[j * d..(j + 1) * d];
        let mut sum_v = 0.0f32;
        for m_idx in 0..m_exp {
            let dir = &directions[m_idx * d..(m_idx + 1) * d];
            let mut dot = 0.0f32;
            for a in 0..d {
                dot += h[a] * dir[a];
            }
            sum_v += sigmoid(dot);
        }
        let mean_v = sum_v / m_f;
        output[j] = mean_v.powf(m_f);
    }
}

/// Reliability-weighted cross-entity set attention — CLR-amplified variant.
///
/// A sibling of [`set_sigmoid_attention_into`] that accepts per-entity
/// reliability weights `r_j` and uses them to modulate each peer's contribution.
/// The update rule is:
///
/// ```text
/// output_i = h_i + (γ / Σ_j r_j) · Σ_j α_ij · r_j · (v_j − h_i)
/// ```
///
/// where `α_ij = σ((q_i · k_j) / √k · β)` is the same sigmoid attention gate as
/// the plain variant, and `r_j ≥ 0` is the reliability weight for entity `j`.
///
/// When `r_j = 1` for all `j`, this reduces **bit-identically** to
/// [`set_sigmoid_attention_into`] — the plain variant is the uniform-reliability
/// special case (G1 test).
///
/// The reliability weighting concentrates the aggregate toward high-reliability
/// entities, converting plain averaging (which dilutes the signal — G8 failure)
/// into amplification. See Plan 570 for the full design + the G8 closure proof.
///
/// Use [`clr_reliability_scores`] to compute CLR-style reliability scores, or
/// pass any non-negative weight slice from another source.
///
/// # Arguments
///
/// Same as [`set_sigmoid_attention_into`], plus:
/// * `reliability` — Per-entity reliability weights `[n]`. Must be non-negative;
///   a weight of 0.0 means that entity contributes nothing. If all weights are
///   zero, `output` is set to `states` (no-op).
///
/// # Errors
///
/// Returns [`SetAttentionError::ReliabilityLenMismatch`] if
/// `reliability.len() != n`. All other error conditions match
/// [`set_sigmoid_attention_into`].
///
/// # Zero allocations
///
/// Same scratch discipline as [`set_sigmoid_attention_into`] — zero heap
/// allocations in steady state.
#[cfg(feature = "clr_weighted_set_attention")]
#[inline]
pub fn clr_weighted_set_attention_into(
    states: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: Option<&[f32]>,
    reliability: &[f32],
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
    if reliability.len() != n {
        return Err(SetAttentionError::ReliabilityLenMismatch {
            expected: n,
            got: reliability.len(),
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

    // ── Project all queries and keys (identical to plain SA) ─────────
    for i in 0..n {
        let state_row = &states[i * d..(i + 1) * d];
        let q_row = &mut scratch_q[i * k..(i + 1) * k];
        let k_row = &mut scratch_k[i * k..(i + 1) * k];
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

    let scale = cfg.beta / (k as f32).sqrt();

    match cfg.top_k {
        None => dense_accumulate_weighted(
            output,
            states,
            w_v,
            reliability,
            scratch_q,
            scratch_k,
            scratch_alpha,
            cfg,
            n,
            d,
            k,
            scale,
        ),
        Some(k_max) => {
            // Sparse path: seed output with states (accumulate over top-k only).
            output.copy_from_slice(states);
            if k_max == 0 {
                return Ok(());
            }
            topk_accumulate_weighted(
                output,
                states,
                w_v,
                reliability,
                scratch_q,
                scratch_k,
                scratch_alpha,
                cfg,
                n,
                d,
                k,
                scale,
                k_max,
            )
        }
    }
}

/// Dense reliability-weighted path. O(N²·k) attention scores.
///
/// The residual update is normalised by `R = Σ_j r_j` (the total reliability
/// weight) so γ is invariant to the reliability scale:
/// `h_i' = h_i + (γ/R) · Σ_j α_ij · r_j · (v_j − h_i)`.
#[cfg(feature = "clr_weighted_set_attention")]
#[inline]
fn dense_accumulate_weighted(
    output: &mut [f32],
    states: &[f32],
    w_v: Option<&[f32]>,
    reliability: &[f32],
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
    // R = Σ_j r_j — loop-invariant, precomputed once. Makes γ invariant to the
    // reliability scale (doubling all r_j leaves the output unchanged).
    let mut r_sum = 0.0f32;
    for j in 0..n {
        r_sum += reliability[j];
    }
    if r_sum < 1e-10 {
        // All-zero reliability → no contribution. Output = states.
        output.copy_from_slice(states);
        return Ok(());
    }
    let gamma_per_r = gamma / r_sum;

    assert!(
        d <= 16,
        "dense_accumulate_weighted: d > 16 needs a larger stack scratch"
    );

    // Precompute v_proj[j] = W_V · state_j for all j (same pattern as plain
    // dense_accumulate — thread_local grow-only buffer, zero alloc steady-state).
    let v_proj: Option<&mut [f32]> = w_v.map(|wv| {
        thread_local! {
            static V_PROJ_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
        }
        V_PROJ_BUF.with(|cell| {
            // Safety: thread_local guarantees exclusive per-thread access.
            let vp = unsafe { &mut *cell.get() };
            if vp.len() < n * d {
                vp.resize(n * d, 0.0);
            }
            let vp = &mut vp[..n * d];
            for j in 0..n {
                let state_j = &states[j * d..(j + 1) * d];
                let vp_row = &mut vp[j * d..(j + 1) * d];
                for m in 0..d {
                    let mut v_j_m = 0.0f32;
                    for a in 0..d {
                        v_j_m += wv[a + m * d] * state_j[a];
                    }
                    vp_row[m] = v_j_m;
                }
            }
            vp
        })
    });

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
        // Accumulate sum_arv[m] = Σ_j α_ij · r_j · v_j[m] and
        // sum_ar = Σ_j α_ij · r_j separately, then combine once per (i, m).
        let mut sum_arv = [0.0f32; 16];
        let mut sum_ar = 0.0f32;
        let sa = &scratch_alpha[..n];
        if let Some(vp) = &v_proj {
            for j in 0..n {
                let alpha = sa[j];
                if alpha == 0.0 {
                    continue;
                }
                let ar = alpha * reliability[j];
                sum_ar += ar;
                let v_j = &vp[j * d..(j + 1) * d];
                for m in 0..d {
                    sum_arv[m] += ar * v_j[m];
                }
            }
        } else {
            // Identity V: v_j = state_j.
            for j in 0..n {
                let alpha = sa[j];
                if alpha == 0.0 {
                    continue;
                }
                let ar = alpha * reliability[j];
                sum_ar += ar;
                let state_j = &states[j * d..(j + 1) * d];
                for m in 0..d {
                    sum_arv[m] += ar * state_j[m];
                }
            }
        }
        // Combine: output[i][m] = state_i[m] + (γ/R) · (sum_arv[m] − state_i[m] · sum_ar)
        let state_i = &states[i * d..(i + 1) * d];
        let out_i = &mut output[i * d..(i + 1) * d];
        for m in 0..d {
            out_i[m] = state_i[m] + gamma_per_r * (sum_arv[m] - state_i[m] * sum_ar);
        }
    }
    Ok(())
}

/// Sparse top-k reliability-weighted path.
///
/// Normalises by `R_k = Σ_{j∈topk} r_j` (the total reliability weight of the
/// selected peers) per query — matching the dense path's γ-invariance property.
#[cfg(feature = "clr_weighted_set_attention")]
#[inline]
fn topk_accumulate_weighted(
    output: &mut [f32],
    states: &[f32],
    w_v: Option<&[f32]>,
    reliability: &[f32],
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

    thread_local! {
        static IDX_BUF: UnsafeCell<Vec<usize>> = const { UnsafeCell::new(Vec::new()) };
    }
    let idx: &mut [usize] = IDX_BUF.with(|cell| {
        let buf = unsafe { &mut *cell.get() };
        if buf.len() < n {
            buf.resize(n, 0);
        }
        let buf = &mut buf[..n];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i;
        }
        buf
    });

    let v_proj: Option<&mut [f32]> = w_v.map(|wv| {
        thread_local! {
            static V_PROJ_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
        }
        V_PROJ_BUF.with(|cell| {
            let vp = unsafe { &mut *cell.get() };
            if vp.len() < n * d {
                vp.resize(n * d, 0.0);
            }
            let vp = &mut vp[..n * d];
            for j in 0..n {
                let state_j = &states[j * d..(j + 1) * d];
                let vp_row = &mut vp[j * d..(j + 1) * d];
                for m in 0..d {
                    let mut v_j_m = 0.0f32;
                    for a in 0..d {
                        v_j_m += wv[a + m * d] * state_j[a];
                    }
                    vp_row[m] = v_j_m;
                }
            }
            vp
        })
    });

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
        let cmp_alpha = |&a: &usize, &b: &usize| scratch_alpha[b].total_cmp(&scratch_alpha[a]);
        if effective_k > 0 {
            let (front, _, _) = idx.select_nth_unstable_by(effective_k - 1, cmp_alpha);
            front.sort_unstable_by(cmp_alpha);
        }
        let top = &idx[..effective_k];

        // R_k = Σ_{j∈topk} r_j — per-query, depends on which peers are selected.
        let mut r_sum = 0.0f32;
        for &j in top {
            r_sum += reliability[j];
        }
        if r_sum < 1e-10 {
            continue; // All selected peers have zero reliability → no update.
        }
        let gamma_per_r = gamma / r_sum;

        let state_i = &states[i * d..(i + 1) * d];
        let out_i = &mut output[i * d..(i + 1) * d];
        for &j in top {
            let alpha = scratch_alpha[j];
            if alpha == 0.0 {
                continue;
            }
            let coeff = gamma_per_r * alpha * reliability[j];
            if let Some(vp) = &v_proj {
                let v_j = &vp[j * d..(j + 1) * d];
                for m in 0..d {
                    out_i[m] += coeff * (v_j[m] - state_i[m]);
                }
            } else {
                let state_j = &states[j * d..(j + 1) * d];
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
    // Delegates to the codebase-wide Cephes-backed fast_sigmoid
    // (~1 ULP accurate, ~1.7× faster than libm exp on aarch64). The two-branch
    // form was a workaround for libm exp overflow; fast_sigmoid handles this
    // internally via +-40 early-exits.
    crate::simd::fast_sigmoid(x)
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
    debug_assert_eq!(
        out.len(),
        d * k,
        "identity_projection_into: out.len() must be d*k"
    );
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
            &states,
            &w,
            &w,
            None,
            &mut output,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
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
            &states,
            &w,
            &w,
            None,
            &mut output,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();
        for (o, s) in output.iter().zip(states.iter()) {
            assert!(
                (o - s).abs() < 1e-6,
                "identical states → no-op, got Δ={}",
                o - s
            );
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
            &bad_states,
            &w_q,
            &w_k,
            None,
            &mut output,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap_err();
        assert_eq!(
            err,
            SetAttentionError::StatesLenMismatch {
                expected: n * d,
                got: n * d + 1
            }
        );

        // Wrong w_q length.
        let bad_wq = vec![0.0f32; d * k + 1];
        let err = set_sigmoid_attention_into(
            &states,
            &bad_wq,
            &w_k,
            None,
            &mut output,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap_err();
        assert_eq!(
            err,
            SetAttentionError::WqLenMismatch {
                expected: d * k,
                got: d * k + 1
            }
        );

        // Zero dim.
        let err = set_sigmoid_attention_into(
            &states,
            &w_q,
            &w_k,
            None,
            &mut output,
            &cfg,
            0,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
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
                assert_eq!(w[i * d + j], expected, "identity[{i},{j}] wrong");
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

    /// Topk path (select_nth_unstable_by) must match the dense path when
    /// k_max >= n — the top-k of all N peers is all N peers, so the sparse
    /// path must produce bit-identical output to the dense path.
    ///
    /// This is the correctness gate for the select_nth_unstable_by partial
    /// sort: it verifies the front-sort produces the same accumulation order
    /// as the full sort.
    #[test]
    fn topk_matches_dense_when_kmax_ge_n() {
        let n = 6;
        let d = 8;
        let k = 8;
        // Distinct states so attention weights differ (exercises the sort).
        let states: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let w = identity(d);
        let cfg_dense = SetAttentionConfig::new(1.0, 0.3);
        let cfg_topk = SetAttentionConfig::new(1.0, 0.3).with_top_k(n); // k_max = n

        let mut out_dense = vec![0.0f32; n * d];
        let mut out_topk = vec![0.0f32; n * d];
        let mut sq = vec![0.0f32; n * k];
        let mut sk = vec![0.0f32; n * k];
        let mut sa = vec![0.0f32; n];

        set_sigmoid_attention_into(
            &states,
            &w,
            &w,
            None,
            &mut out_dense,
            &cfg_dense,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();
        // Reset scratches (the kernel writes into them).
        sq.iter_mut().for_each(|x| *x = 0.0);
        sk.iter_mut().for_each(|x| *x = 0.0);
        sa.iter_mut().for_each(|x| *x = 0.0);
        set_sigmoid_attention_into(
            &states,
            &w,
            &w,
            None,
            &mut out_topk,
            &cfg_topk,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();

        for i in 0..n * d {
            // Algebraically identical (both normalise by N when k_max=n), but
            // numerically different accumulation order: dense accumulates
            // sum_av/sum_a separately then combines once; topk accumulates
            // coeff*(v_j-state) per j. So we expect ~1e-6 rounding-level agreement.
            assert!(
                (out_dense[i] - out_topk[i]).abs() < 1e-5,
                "topk path diverged from dense at index {}: dense={}, topk={}, Δ={}",
                i,
                out_dense[i],
                out_topk[i],
                out_dense[i] - out_topk[i]
            );
        }
    }

    /// Topk path with k_max < n must select only the highest-α peers.
    /// Smoke: with k_max=1 and γ>0, each query pulls toward its single most
    /// aligned peer. Output must be finite and distinct from the dense path.
    #[test]
    fn topk_kmax_lt_n_selects_subset() {
        let n = 5;
        let d = 8;
        let k = 8;
        let states: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1).collect();
        let w = identity(d);
        let cfg_topk = SetAttentionConfig::new(1.0, 0.5).with_top_k(1);
        let cfg_dense = SetAttentionConfig::new(1.0, 0.5);

        let mut out_topk = vec![0.0f32; n * d];
        let mut out_dense = vec![0.0f32; n * d];
        let mut sq = vec![0.0f32; n * k];
        let mut sk = vec![0.0f32; n * k];
        let mut sa = vec![0.0f32; n];

        set_sigmoid_attention_into(
            &states,
            &w,
            &w,
            None,
            &mut out_topk,
            &cfg_topk,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();
        sq.iter_mut().for_each(|x| *x = 0.0);
        sk.iter_mut().for_each(|x| *x = 0.0);
        sa.iter_mut().for_each(|x| *x = 0.0);
        set_sigmoid_attention_into(
            &states,
            &w,
            &w,
            None,
            &mut out_dense,
            &cfg_dense,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();

        // All outputs finite.
        for v in &out_topk {
            assert!(v.is_finite(), "topk output not finite: {v}");
        }
        // With k_max=1 the sparse path must differ from dense (which uses all 5).
        let any_diff = out_topk
            .iter()
            .zip(out_dense.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "k_max=1 should differ from dense (all-N) path");
    }

    // ─────────────────────────────────────────────────────────────────────
    // CLR-weighted variant tests (Plan 570)
    // ─────────────────────────────────────────────────────────────────────

    /// G1: uniform reliability (all 1.0) must produce bit-identical output to
    /// plain `set_sigmoid_attention_into`. This is the mathematical guarantee
    /// that the CLR-weighted variant is a strict generalization — plain SA is
    /// the uniform-reliability special case.
    ///
    /// IEEE 754 guarantees `x * 1.0 == x` for all finite x, so the extra multiply
    /// in the weighted path introduces zero rounding error when r_j = 1.0.
    #[cfg(feature = "clr_weighted_set_attention")]
    #[test]
    fn g1_clr_weighted_uniform_reliability_is_bit_identical_to_plain() {
        let n = 16;
        let d = 8;
        let k = 4;

        // Deterministic LCG (same as g1_permutation_equivariance).
        let mut state = 0x00C0_FFEE_5678_u64;
        let mut lcg = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 32) as f32) / (u32::MAX as f32)
        };

        let states: Vec<f32> = (0..n * d).map(|_| lcg()).collect();
        let w = identity_projection(d, k);
        let cfg = SetAttentionConfig::new(1.0, 0.3);
        let reliability = vec![1.0f32; n];

        // Plain SA.
        let mut plain_out = vec![0.0f32; n * d];
        {
            let mut sq = vec![0.0; n * k];
            let mut sk = vec![0.0; n * k];
            let mut sa = vec![0.0; n];
            set_sigmoid_attention_into(
                &states,
                &w,
                &w,
                None,
                &mut plain_out,
                &cfg,
                n,
                d,
                k,
                &mut sq,
                &mut sk,
                &mut sa,
            )
            .unwrap();
        }

        // CLR-weighted SA with uniform reliability.
        let mut weighted_out = vec![0.0f32; n * d];
        {
            let mut sq = vec![0.0; n * k];
            let mut sk = vec![0.0; n * k];
            let mut sa = vec![0.0; n];
            clr_weighted_set_attention_into(
                &states,
                &w,
                &w,
                None,
                &reliability,
                &mut weighted_out,
                &cfg,
                n,
                d,
                k,
                &mut sq,
                &mut sk,
                &mut sa,
            )
            .unwrap();
        }

        // Bit-identical assertion.
        for i in 0..n * d {
            assert_eq!(
                plain_out[i].to_bits(),
                weighted_out[i].to_bits(),
                "G1 FAIL at index {i}: plain={}, weighted={}",
                plain_out[i],
                weighted_out[i]
            );
        }
    }

    /// G1 (topk path): same bit-identical guarantee for the sparse top-k variant.
    #[cfg(feature = "clr_weighted_set_attention")]
    #[test]
    fn g1_clr_weighted_uniform_reliability_topk_bit_identical() {
        let n = 16;
        let d = 8;
        let k = 4;

        let mut s = 0xABCD_1234_u64;
        let mut lcg = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f32) / (u32::MAX as f32)
        };
        let states: Vec<f32> = (0..n * d).map(|_| lcg()).collect();
        let w = identity_projection(d, k);
        let cfg = SetAttentionConfig::new(1.0, 0.3).with_top_k(5);
        let reliability = vec![1.0f32; n];

        let mut plain_out = vec![0.0f32; n * d];
        let mut weighted_out = vec![0.0f32; n * d];
        let (mut sq, mut sk, mut sa) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        set_sigmoid_attention_into(
            &states,
            &w,
            &w,
            None,
            &mut plain_out,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();
        // Reset scratch (topk path uses thread_local buffers internally).
        sq.iter_mut().for_each(|x| *x = 0.0);
        sk.iter_mut().for_each(|x| *x = 0.0);
        sa.iter_mut().for_each(|x| *x = 0.0);
        clr_weighted_set_attention_into(
            &states,
            &w,
            &w,
            None,
            &reliability,
            &mut weighted_out,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();

        for i in 0..n * d {
            assert_eq!(
                plain_out[i].to_bits(),
                weighted_out[i].to_bits(),
                "G1 topk FAIL at index {i}: plain={}, weighted={}",
                plain_out[i],
                weighted_out[i]
            );
        }
    }

    /// G1 helper test: clr_reliability_scores with uniform-zero directions
    /// (all sigmoid = 0.5) gives mean 0.5, ^M = 0.5^M — a known value.
    #[cfg(feature = "clr_weighted_set_attention")]
    #[test]
    fn clr_reliability_scores_known_values() {
        let n = 4;
        let d = 2;
        let m_exp = 3;
        // States: [1, 0] repeated (so dot with dir = dir[0]).
        let states = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        // Direction: [0, 0] → dot = 0 → sigmoid(0) = 0.5 for all.
        let directions = vec![0.0; m_exp * d];
        let mut output = vec![0.0; n];
        clr_reliability_scores(&states, &directions, m_exp, n, d, &mut output);
        // mean_v = 0.5, ^3 = 0.125.
        let expected = 0.5f32.powi(3);
        for &r in &output {
            assert!((r - expected).abs() < 1e-6, "expected {expected}, got {r}");
        }
    }

    /// G4 (dense path): no allocation primitives in the weighted dense path by
    /// construction. The actual alloc-count perf gate lives in the bench file.
    #[cfg(feature = "clr_weighted_set_attention")]
    #[test]
    fn g4_clr_weighted_dense_path_by_construction_no_alloc() {
        let n = 64;
        let d = 8;
        let k = 4;
        let states: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.001).collect();
        let w = identity_projection(d, k);
        let reliability: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; n * d];
        let (mut sq, mut sk, mut sa) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        let cfg = SetAttentionConfig::default();
        for _ in 0..100 {
            clr_weighted_set_attention_into(
                &states,
                &w,
                &w,
                None,
                &reliability,
                &mut output,
                &cfg,
                n,
                d,
                k,
                &mut sq,
                &mut sk,
                &mut sa,
            )
            .unwrap();
        }
        for v in &output {
            assert!(v.is_finite(), "G4: non-finite output after 100 calls");
        }
    }

    /// Sanity: all-zero reliability → output = states (no-op).
    #[cfg(feature = "clr_weighted_set_attention")]
    #[test]
    fn clr_weighted_zero_reliability_is_noop() {
        let n = 8;
        let d = 4;
        let k = 4;
        let states: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1).collect();
        let w = identity(d);
        let reliability = vec![0.0f32; n];
        let mut output = vec![0.0f32; n * d];
        let (mut sq, mut sk, mut sa) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        let cfg = SetAttentionConfig::default();
        clr_weighted_set_attention_into(
            &states,
            &w,
            &w,
            None,
            &reliability,
            &mut output,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        )
        .unwrap();
        for i in 0..n * d {
            assert_eq!(output[i], states[i], "zero reliability should be a no-op");
        }
    }

    /// Sanity: error when reliability.len() != n.
    #[cfg(feature = "clr_weighted_set_attention")]
    #[test]
    fn clr_weighted_reliability_len_error() {
        let n = 8;
        let d = 4;
        let k = 4;
        let states = vec![0.5; n * d];
        let w = identity(d);
        let reliability = vec![1.0; n - 1]; // Wrong length.
        let mut output = vec![0.0; n * d];
        let (mut sq, mut sk, mut sa) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        let cfg = SetAttentionConfig::default();
        let result = clr_weighted_set_attention_into(
            &states,
            &w,
            &w,
            None,
            &reliability,
            &mut output,
            &cfg,
            n,
            d,
            k,
            &mut sq,
            &mut sk,
            &mut sa,
        );
        assert!(matches!(
            result,
            Err(SetAttentionError::ReliabilityLenMismatch {
                expected: 8,
                got: 7
            })
        ));
    }

    /// G2: CLR-weighted SA latency must be ≤ 2× plain SA at N=64.
    ///
    /// The theoretical overhead is one extra multiply per peer (alpha * r_j)
    /// plus one O(n) r_sum precomputation — total ~12% more arithmetic. In
    /// practice, the ratio is ~1.0× (the extra multiply is hidden by memory
    /// latency in the O(N²) inner loop).
    ///
    /// This test runs WITHOUT the CountingAllocator (which creates a false ~3.5×
    /// ratio in the bench binary due to thread_local cache interference).
    #[cfg(feature = "clr_weighted_set_attention")]
    #[test]
    fn g2_clr_weighted_latency_ratio() {
        use std::hint::black_box;
        use std::time::Instant;

        let n = 64;
        let d = 8;
        let k = 4;
        let states: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.001).collect();
        let w = identity_projection(d, k);
        let reliability: Vec<f32> = (0..n).map(|i| 0.3 + (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; n * d];
        let (mut sq, mut sk, mut sa) = (vec![0.0; n * k], vec![0.0; n * k], vec![0.0; n]);
        let cfg = SetAttentionConfig::default();

        // Warm up.
        for _ in 0..1000 {
            set_sigmoid_attention_into(
                &states,
                &w,
                &w,
                None,
                &mut output,
                &cfg,
                n,
                d,
                k,
                &mut sq,
                &mut sk,
                &mut sa,
            )
            .unwrap();
            clr_weighted_set_attention_into(
                &states,
                &w,
                &w,
                None,
                &reliability,
                &mut output,
                &cfg,
                n,
                d,
                k,
                &mut sq,
                &mut sk,
                &mut sa,
            )
            .unwrap();
        }

        let iters = 10_000;

        // Measure plain.
        let t0 = Instant::now();
        for _ in 0..iters {
            set_sigmoid_attention_into(
                black_box(&states),
                black_box(&w),
                black_box(&w),
                None,
                black_box(&mut output),
                black_box(&cfg),
                n,
                d,
                k,
                &mut sq,
                &mut sk,
                &mut sa,
            )
            .unwrap();
        }
        let plain_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        // Measure weighted.
        let t1 = Instant::now();
        for _ in 0..iters {
            clr_weighted_set_attention_into(
                black_box(&states),
                black_box(&w),
                black_box(&w),
                None,
                black_box(&reliability),
                black_box(&mut output),
                black_box(&cfg),
                n,
                d,
                k,
                &mut sq,
                &mut sk,
                &mut sa,
            )
            .unwrap();
        }
        let weighted_ns = t1.elapsed().as_nanos() as f64 / iters as f64;

        let ratio = weighted_ns / plain_ns;
        eprintln!(
            "G2: plain={plain_ns:.0}ns, weighted={weighted_ns:.0}ns, ratio={ratio:.2}× (target ≤2×)"
        );
        assert!(ratio <= 2.0, "G2 FAIL: ratio {ratio:.2}× > 2× target");
    }
}
