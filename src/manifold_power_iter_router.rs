//! Manifold Power Iteration MoE Router — modelless one-shot router-row
//! conditioning at freeze/thaw snapshot swap.
//!
//! Distilled from **arXiv:2606.12397** — *"Redesign Mixture-of-Experts
//! Routers with Manifold Power Iteration"* (Wu/Lv/Xie/Lin, RUC/Tencent,
//! 10 Jun 2026). Research 246, Plan 279.
//!
//! # Principle (paper §2)
//!
//! A well-coupled MoE router row `R[i]` should align with the **principal
//! singular direction** of expert `i`'s gate-weight matrix `W_g[i]`, because
//! the top singular vector is the optimal 1-vector compression of a matrix
//! (Eckart–Young / Rayleigh–Ritz). MPI enforces this via one power-iteration
//! step against the expert Gram `M[i] = W_g[i] · W_g[i]^T`, followed by an
//! L2 retraction to a constant norm `C = C'/√N`:
//!
//! ```text
//! R̂[i]  = R[i] · M[i]                    (Eq. 4 — power iteration step)
//! R'[i] = C · R̂[i] / ‖R̂[i]‖₂             (Eq. 5 — L2 retraction)
//! ```
//!
//! Then inference is identical to vanilla top-k gating — only the router
//! rows change. This is **not** a new inference behavior class; it is
//! *conditioning*. The paper's §3.3 proves MPI ≈ steepest ascent on the
//! Rayleigh quotient `‖R[i]·W_g[i]‖²/‖R[i]‖²` on the spherical manifold.
//!
//! # Sigmoid, Not Softmax (§2.3 distillation — AGENTS.md constraint)
//!
//! The paper uses `Softmax(TopK(x·R'^T))` (Eq. 6). We distill with
//! **independent per-expert sigmoid** instead:
//!
//! ```text
//! gate_i(x) = σ(β · x · R'[i]^T)         (independent per-expert sigmoid)
//! select    = TopK_k(gate_1, …, gate_N)
//! ```
//!
//! The paper §6 explicitly tried sigmoid and it still improved over vanilla
//! (41.64 → 42.05 downstream). For our use case (NPC expert routing where
//! multiple experts can independently be "relevant") sigmoid is the
//! *correct* semantics — a combat expert and a movement expert can both
//! fire on the same frame. Softmax would force zero-sum competition.
//! [`gate_sigmoid_topk`] enforces this — see G7.
//!
//! # When It Fires
//!
//! **Once per freeze/thaw snapshot swap** (research note §2.2), never
//! per-token. The router `R'` is precomputed at snapshot boundary and reused
//! for all subsequent inference — zero per-token overhead (paper §4.2).
//! Deterministic given `(R, M, c_prime, iters, snapshot_version)` → safe
//! under `SyncBlock → ChainConsensus` quorum (G5).
//!
//! # Substrate Routing
//!
//! Pure CPU SIMD via the shared [`power_iter_retract`] helper
//! (`crate::spectral_retract`). Sub-μs per row for `D ≤ 256` (plasma tier),
//! sub-ms for game-scale pools `(N=8, D=256)` (G4).
//!
//! # Example
//!
//! See `examples/manifold_power_iter_router_basic.rs` for a runnable demo
//! showing λ alignment and MaxVio before → after conditioning.

#![allow(clippy::too_many_arguments)]

use crate::simd::{simd_dot_f32, simd_fused_scale_acc};
use crate::spectral_retract::{power_iter_retract, PowerRetractScratch};
use blake3::Hasher;

// ── Config / result types ────────────────────────────────────────────────

/// Configuration for MPI router conditioning (paper §1.4 defaults).
#[derive(Debug, Clone, Copy)]
pub struct MpiRouterConfig {
    /// Norm scale `C'`. Target row norm is `C = C'/√N` (paper Eq. 7:
    /// chosen so `‖x·R'^T‖_∞ = O(1)`). Typical: `C' = 1.0`.
    pub c_prime: f32,
    /// Power-iteration steps. **Paper default `iters=1`** (§1.4: a single
    /// step is more robust than fully-converged SVD; `iters=10` loses 5%
    /// throughput with no convergence gain). Enforced as default by G8.
    pub iters: u8,
    /// Sigmoid temperature `β` for [`gate_sigmoid_topk`]. Replaces the
    /// paper's `C = C'/√N` as the calibration knob (§2.3).
    pub beta_sigmoid: f32,
}

impl Default for MpiRouterConfig {
    fn default() -> Self {
        // Paper §1.4 defaults. `iters=1` is the robust choice (G8 enforces).
        Self {
            c_prime: 1.0,
            iters: 1,
            beta_sigmoid: 1.0,
        }
    }
}

/// Result of an MPI router conditioning pass.
///
/// `r_prime` is the reconditioned router (written in place into the caller's
/// buffer; this field mirrors it for convenience). Diagnostics follow paper
/// Eq. 11 (λ alignment) and §1.4 (MaxVio).
#[derive(Debug, Clone)]
pub struct MpiRouterResult {
    /// The MPI-conditioned router `R'` (N×D, row-major). Caller-owned
    /// reference copy — the canonical store is the `&mut r` passed to
    /// [`manifold_power_iter_router`].
    pub r_prime: Vec<f32>,
    /// Router–expert alignment metric λ (paper Eq. 11): mean over rows of
    /// `(R'[i]·M[i]·R'[i]^T) / (‖R'[i]·M[i]‖ · ‖R'[i]‖)`. Vanilla MoE
    /// λ ≈ 0.22–0.37; MPI λ ≈ 0.62–0.70 (paper §1.4). Higher = better.
    pub lambda_alignment: f32,
    /// Max row-norm deviation from target `C = C'/√N` (paper §1.4 MaxVio).
    /// Should be ≈ 0 after retraction (each row is exactly at norm `C`).
    /// Before MPI, MaxVio reflects router-norm disparity (paper 1.13 → 0.96).
    pub maxvio: f32,
}

/// Borrowed view of one expert's Gram matrix `M[i] = W_g[i]·W_g[i]^T` (D×D).
///
/// Allows the snapshot hook to serve cached grams (Owned) or freshly-built
/// grams (Borrowed) without forcing an allocation on the caller.
#[derive(Debug, Clone, Copy)]
pub enum ExpertGramView<'a> {
    Borrowed(&'a [f32]),
    Owned(&'a [f32]),
}

impl<'a> AsRef<[f32]> for ExpertGramView<'a> {
    fn as_ref(&self) -> &[f32] {
        match self {
            ExpertGramView::Borrowed(s) | ExpertGramView::Owned(s) => s,
        }
    }
}

// ── Gram construction ────────────────────────────────────────────────────

/// Compute the expert Gram `M = W_g · W_g^T` into a caller-owned buffer.
///
/// `W_g` is `d_model × d_model` row-major (gate weights for one expert).
/// Output `out` is `d_model × d_model` row-major PSD Gram.
///
/// Blocked matmul (D×D → D²/D_blocked sub-blocks) for cache friendliness.
/// Called once per snapshot swap (warm tier) — never per-token.
///
/// # Panics
///
/// Debug assert: `w_g.len() == d_model*d_model`, `out.len() == d_model*d_model`.
pub fn compute_expert_gram_into(w_g: &[f32], d_model: usize, out: &mut [f32]) {
    debug_assert_eq!(w_g.len(), d_model * d_model, "W_g size mismatch");
    debug_assert_eq!(out.len(), d_model * d_model, "out size mismatch");

    // Blocked matmul: M[i,j] = Σ_k W_g[i,k] · W_g[j,k].
    // Block size tuned for L1 (32KB → 8K f32 → ~90×90 block).
    const BLOCK: usize = 64;

    out.fill(0.0);
    let n = d_model;

    let n_block = n.div_ceil(BLOCK);
    for ii in 0..n_block {
        for jj in 0..n_block {
            let i0 = ii * BLOCK;
            let j0 = jj * BLOCK;
            let i1 = (i0 + BLOCK).min(n);
            let j1 = (j0 + BLOCK).min(n);
            for i in i0..i1 {
                let row_i = &w_g[i * n..(i + 1) * n];
                let out_row = &mut out[i * n..(i + 1) * n];
                for j in j0..j1 {
                    // SIMD-friendly dot product of two rows of W_g.
                    let s = simd_dot_f32(row_i, &w_g[j * n..(j + 1) * n], n);
                    out_row[j] = s;
                }
            }
        }
    }
}

// ── Main primitive: manifold_power_iter_router ───────────────────────────

/// One-shot Manifold Power Iteration on a router matrix (paper Eq. 4–5).
///
/// For each expert row `i`: `R̂[i] = R[i]·M[i]` (Eq. 4), then
/// `R'[i] = C·R̂[i]/‖R̂[i]‖₂` with `C = c_prime/√N` (Eq. 5, Eq. 7).
///
/// Diagnostics: `lambda_alignment` (paper Eq. 11, mean cosine of
/// `(R'[i]·M[i]·R'[i]^T) / (‖R'[i]·M[i]‖·‖R'[i]‖)`), `maxvio` (max
/// `|‖R'[i]‖₂ − C|`, should be ≈0 post-retraction).
///
/// Deterministic (G5): pure function of `(r, grams, c_prime, iters)` →
/// byte-identical `R'` across runs — sync/quorum-safe. Zero-alloc in the
/// retraction loop (caller-owned `scratch`); the `r_prime` field in the
/// returned [`MpiRouterResult`] clones once for diagnostic convenience. For
/// the true zero-alloc path (no `r_prime` clone), use
/// [`manifold_power_iter_router_inplace`].
///
/// # Panics: debug assert only (shape mismatches).
pub fn manifold_power_iter_router(
    r: &mut [f32],
    gram_per_expert: &[&[f32]],
    n_experts: usize,
    d_model: usize,
    c_prime: f32,
    iters: u8,
    scratch: &mut PowerRetractScratch,
) -> MpiRouterResult {
    debug_assert_eq!(r.len(), n_experts * d_model, "router shape mismatch");
    debug_assert_eq!(
        gram_per_expert.len(),
        n_experts,
        "gram_per_expert count mismatch"
    );

    let target_norm = c_prime / (n_experts as f32).sqrt();
    let (lambda, maxvio) =
        manifold_power_iter_router_inplace(r, gram_per_expert, n_experts, d_model, target_norm, iters, scratch);

    MpiRouterResult {
        r_prime: r.to_vec(),
        lambda_alignment: lambda,
        maxvio,
    }
}

/// **Zero-alloc variant** of [`manifold_power_iter_router`].
///
/// Performs the same in-place retraction and computes the same `(λ, MaxVio)`
/// diagnostics, but skips the convenience `r_prime: Vec<f32>` clone in
/// [`MpiRouterResult`] — `r` is the canonical conditioned router (mutated in
/// place). The scratch buffer used during the diagnostics pass is taken from
/// `scratch.mv_out` (resized internally if needed), so no per-call allocation
/// is performed.
///
/// Use this from snapshot-swap hot paths that already own `r` and only need
/// the scalar diagnostics. The public [`manifold_power_iter_router`] wraps
/// this with a one-shot `r.to_vec()` for callers that want a snapshot.
///
/// `target_norm` is `c_prime / √N` (passed precomputed to avoid a redundant
/// sqrt+div on every call).
pub fn manifold_power_iter_router_inplace(
    r: &mut [f32],
    gram_per_expert: &[&[f32]],
    n_experts: usize,
    d_model: usize,
    target_norm: f32,
    iters: u8,
    scratch: &mut PowerRetractScratch,
) -> (f32, f32) {
    debug_assert_eq!(r.len(), n_experts * d_model, "router shape mismatch");
    debug_assert_eq!(
        gram_per_expert.len(),
        n_experts,
        "gram_per_expert count mismatch"
    );

    // Phase 1: retract each row (paper Eq. 4–5).
    for i in 0..n_experts {
        let row = &mut r[i * d_model..(i + 1) * d_model];
        let gram = gram_per_expert[i];
        debug_assert_eq!(gram.len(), d_model * d_model, "gram[i] shape mismatch");
        power_iter_retract(row, gram, d_model, target_norm, iters, scratch);
    }

    // Phase 2: diagnostics. Reuse the existing scratch buffer for the
    // per-row `R'[i]·M[i]` matvec output (zero-alloc).
    scratch.ensure_dim(d_model);
    let (lambda, maxvio) = compute_diagnostics_with_scratch(
        r,
        gram_per_expert,
        n_experts,
        d_model,
        target_norm,
        &mut scratch.mv_out,
    );
    (lambda, maxvio)
}

/// Compute (lambda_alignment, maxvio) diagnostics for a router given grams.
///
/// Exposed separately so callers can measure "before" (unconditioned) and
/// "after" (conditioned) λ / MaxVio without re-running MPI.
pub fn compute_diagnostics(
    r: &[f32],
    gram_per_expert: &[&[f32]],
    n_experts: usize,
    d_model: usize,
    target_norm: f32,
) -> (f32, f32) {
    let mut rm_scratch = vec![0.0f32; d_model];
    compute_diagnostics_with_scratch(
        r,
        gram_per_expert,
        n_experts,
        d_model,
        target_norm,
        &mut rm_scratch,
    )
}

/// **Zero-alloc variant** of [`compute_diagnostics`].
///
/// Identical math, but the per-row `R'[i]·M[i]` matvec output is written into
/// the caller-owned `rm_scratch` buffer (length `>= d_model`). Lets callers
/// that already own a `d_model`-sized scratch (e.g. [`PowerRetractScratch::mv_out`])
/// reuse it across consecutive diagnostics calls.
pub fn compute_diagnostics_with_scratch(
    r: &[f32],
    gram_per_expert: &[&[f32]],
    n_experts: usize,
    d_model: usize,
    target_norm: f32,
    rm_scratch: &mut [f32],
) -> (f32, f32) {
    debug_assert_eq!(
        rm_scratch.len(),
        d_model,
        "rm_scratch must be exactly d_model long"
    );
    let mut lambda_sum = 0.0f32;
    let mut maxvio = 0.0f32;
    for i in 0..n_experts {
        let row = &r[i * d_model..(i + 1) * d_model];
        let gram = gram_per_expert[i];

        // ‖R'[i]‖
        let r_norm = simd_dot_f32(row, row, d_model).sqrt();
        // MaxVio = max |‖R'[i]‖ − C|.
        let vio = (r_norm - target_norm).abs();
        if vio > maxvio {
            maxvio = vio;
        }

        // rm = R'[i] · M[i] (row-vector × matrix, matches Eq. 4 matvec).
        // Rank-1 update form: rm += row[k] * gram[k,:]. Each inner j-loop is a
        // contiguous scaled-accumulate → SIMD FMA kernel (was scalar before).
        rm_scratch.fill(0.0);
        for k in 0..d_model {
            let rk = row[k];
            if rk == 0.0 {
                continue;
            }
            let grow = &gram[k * d_model..(k + 1) * d_model];
            simd_fused_scale_acc(rm_scratch, grow, rk, d_model);
        }
        let rm_norm = simd_dot_f32(rm_scratch, rm_scratch, d_model).sqrt();

        // Numerator: R'[i] · M[i] · R'[i]^T = rm · R'[i]^T = dot(rm, row).
        let num = simd_dot_f32(rm_scratch, row, d_model);
        let denom = rm_norm * r_norm;
        let cos = if denom < 1e-20 { 0.0 } else { num / denom };
        lambda_sum += cos;
    }
    let lambda = lambda_sum / (n_experts as f32).max(1.0);
    (lambda, maxvio)
}

// ── Sigmoid gate (NEVER softmax — AGENTS.md constraint) ──────────────────

/// Independent per-expert sigmoid top-k gating (research note §2.3).
/// `score_i = σ(β · x · R'[i]^T)`, then `TopK_k(scores)`. **Never softmax**
/// — each expert is independent (G7). Correct semantics for NPC routing
/// where multiple experts can fire on the same frame.
///
/// `out_scores` is caller-owned (zero-alloc); returns top-k indices (one Vec
/// alloc). `σ(z) = 1/(1+e^{-z})` via libm `exp`.
///
/// # Returns
///
/// `Vec<usize>` of length `k` — expert indices in descending sigmoid-score
/// order. Allocates once (the index vector); the scores buffer is caller-owned.
///
/// # Sigmoid
///
/// `σ(z) = 1/(1+e^{-z})` via libm `expf`. Independent per expert.
pub fn gate_sigmoid_topk(
    x: &[f32],
    r_prime: &[f32],
    n_experts: usize,
    d_model: usize,
    beta: f32,
    k: usize,
    out_scores: &mut [f32],
) -> Vec<usize> {
    let mut idx: Vec<usize> = vec![0usize; n_experts];
    let kk = gate_sigmoid_topk_into(x, r_prime, n_experts, d_model, beta, k, out_scores, &mut idx);
    idx.truncate(kk);
    idx
}

/// **Zero-alloc variant** of [`gate_sigmoid_topk`].
///
/// Identical math and ordering, but writes the selected expert indices into
/// the caller-owned `idx_buf` (length `>= n_experts`) and returns the truncation
/// length `kk = min(k, n_experts)`. The returned slice `&mut idx_buf[..kk]` is
/// valid until the buffer is next mutated.
///
/// Suitable for per-token NPC routing where the caller holds a pre-allocated
/// `idx_buf` across frames (skips the per-call `Vec<usize>` allocation in
/// [`gate_sigmoid_topk`]).
///
/// # Returns
///
/// Truncation length `kk`. On return, `idx_buf[0..kk]` holds the top-`kk`
/// expert indices in descending sigmoid-score order.
pub fn gate_sigmoid_topk_into(
    x: &[f32],
    r_prime: &[f32],
    n_experts: usize,
    d_model: usize,
    beta: f32,
    k: usize,
    out_scores: &mut [f32],
    idx_buf: &mut [usize],
) -> usize {
    debug_assert_eq!(r_prime.len(), n_experts * d_model, "r_prime shape mismatch");
    debug_assert_eq!(out_scores.len(), n_experts, "out_scores length mismatch");
    debug_assert_eq!(idx_buf.len(), n_experts, "idx_buf length mismatch");

    // Per-expert independent sigmoid.
    for i in 0..n_experts {
        let row = &r_prime[i * d_model..(i + 1) * d_model];
        let dot = simd_dot_f32(x, row, d_model);
        let z = beta * dot;
        // σ(z) = 1/(1+e^{-z}). Use libm exp for portability.
        out_scores[i] = 1.0 / (1.0 + (-z).exp());
    }

    // Initialize idx_buf to [0, 1, 2, ..., n_experts-1] in place.
    for (i, slot) in idx_buf.iter_mut().enumerate() {
        *slot = i;
    }

    // Top-k by score. For small k (typical game-scale N ≤ 256) selection
    // sort is cache-friendly; for large N the caller should use a priority
    // queue. Keep this path branch-free for the hot N≤64 case.
    let kk = k.min(n_experts);
    for i in 0..kk {
        // Find max in [i..n].
        let mut best = i;
        let mut best_score = out_scores[idx_buf[i]];
        for j in (i + 1)..n_experts {
            if out_scores[idx_buf[j]] > best_score {
                best = j;
                best_score = out_scores[idx_buf[j]];
            }
        }
        idx_buf.swap(i, best);
    }
    kk
}

// ── Snapshot-swap hook (Phase 2) ─────────────────────────────────────────

/// Hook called once per freeze/thaw snapshot swap to recondition a router.
/// Called from the **swap path** only (NOT per-token — freeze/thaw constraint).
/// riir-ai's `LoRAHotSwap` consumes this trait.
pub trait MpiRouterSnapshotHook {
    /// Recondition `router` against the current expert grams. Returns the
    /// MPI-conditioned router + diagnostics. Must be deterministic in
    /// `(router, expert_grams, snapshot_version)` (G5). MUST NOT be called
    /// from a per-token forward path — only the swap boundary may mutate
    /// router weights.
    fn recondition_at_swap(
        &mut self,
        router: &mut [f32],
        expert_grams: &[&[f32]],
        n_experts: usize,
        d_model: usize,
        snapshot_version: u64,
    ) -> MpiRouterResult;
}

/// Default snapshot hook — wraps [`manifold_power_iter_router`] with a
/// BLAKE3-tagged Gram cache keyed by snapshot version. Cache hit (same
/// version): skips gram recomputation. Cache miss (version bump): re-fills.
pub struct DefaultMpiRouterSnapshotHook {
    config: MpiRouterConfig,
    scratch: PowerRetractScratch,
    cache_version: u64,
    cache_blake3: [u8; 32],
    cached_grams: Vec<Vec<f32>>,
}

impl DefaultMpiRouterSnapshotHook {
    /// Create a hook with the given config and scratch sized for `d_model`.
    pub fn new(config: MpiRouterConfig, d_model: usize) -> Self {
        Self {
            config,
            scratch: PowerRetractScratch::new(d_model),
            cache_version: 0,
            cache_blake3: [0u8; 32],
            cached_grams: Vec::new(),
        }
    }

    /// Compute BLAKE3 of the concatenation of all expert gram slices.
    /// Used as the cache key (research note §2.2 — "BLAKE3-tagged with
    /// snapshot version").
    fn hash_grams(expert_grams: &[&[f32]]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        for g in expert_grams {
            // Hash the raw f32 bytes. Deterministic across runs (same arch).
            let bytes: &[u8] = bytemuck_cast_f32(g);
            hasher.update(bytes);
        }
        *hasher.finalize().as_bytes()
    }

    /// Check if the cache is valid for this (version, grams) pair.
    fn cache_valid(&self, snapshot_version: u64, expert_grams: &[&[f32]]) -> bool {
        if snapshot_version != self.cache_version || snapshot_version == 0 {
            return false;
        }
        // Defensive: also verify the BLAKE3 matches (catches caller bugs
        // where version was bumped but grams weren't updated).
        let h = Self::hash_grams(expert_grams);
        h == self.cache_blake3
    }

    /// Populate cache from the given grams (copies into owned storage).
    fn fill_cache(&mut self, expert_grams: &[&[f32]]) {
        self.cached_grams.clear();
        self.cached_grams.reserve(expert_grams.len());
        for g in expert_grams {
            self.cached_grams.push(g.to_vec());
        }
        self.cache_blake3 = Self::hash_grams(expert_grams);
    }
}

impl MpiRouterSnapshotHook for DefaultMpiRouterSnapshotHook {
    fn recondition_at_swap(
        &mut self,
        router: &mut [f32],
        expert_grams: &[&[f32]],
        n_experts: usize,
        d_model: usize,
        snapshot_version: u64,
    ) -> MpiRouterResult {
        // Cache check: skip re-fetch if version + BLAKE3 match.
        if !self.cache_valid(snapshot_version, expert_grams) {
            self.fill_cache(expert_grams);
            self.cache_version = snapshot_version;
        }

        // Borrow the cached grams as &[&[f32]] for the kernel call.
        // Lifetime: we hold &mut self, so the borrows are valid for this call.
        let gram_refs: Vec<&[f32]> =
            self.cached_grams.iter().map(|g| g.as_slice()).collect();

        manifold_power_iter_router(
            router,
            &gram_refs,
            n_experts,
            d_model,
            self.config.c_prime,
            self.config.iters,
            &mut self.scratch,
        )
    }
}

// Re-usable byte cast without pulling bytemuck as a hard dep (we transmute
// the f32 slice to bytes via a typed pointer cast — safe because f32 is
// `#[repr(transparent)]` over its bits and the slice length is preserved).
#[inline]
fn bytemuck_cast_f32(s: &[f32]) -> &[u8] {
    let len = core::mem::size_of_val(s);
    unsafe { core::slice::from_raw_parts(s.as_ptr() as *const u8, len) }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 PRNG (matches the rest of the crate).
    fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v.push(((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0);
        }
        v
    }

    fn seeded_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        seeded_vec(seed, rows * cols)
    }

    /// Frobenius / L2 norm.
    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Build Gram M = W·W^T for a square d×d W.
    fn gram_of(w: &[f32], d: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; d * d];
        compute_expert_gram_into(w, d, &mut m);
        m
    }

    /// Construct a d×d matrix whose dominant right-singular vector is `u`
    /// (length d): rank-1 outer product `u·u^T` scaled to a known σ_max.
    fn rank1_matrix(u: &[f32], d: usize, sigma: f32) -> Vec<f32> {
        let un = norm(u);
        let scale = sigma / (un * un);
        let mut w = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                w[i * d + j] = u[i] * u[j] * scale;
            }
        }
        w
    }

    // ── T1.10 unit tests ────────────────────────────────────────────────

    #[test]
    fn t01_synthetic_principal_direction_recovery_iters1() {
        // Construct W_g with known dominant right-singular vector u.
        // Random R[0]; after MPI(iters=1), R'[0] should align with u (cos>0.95).
        let d = 16usize;
        let n = 1usize;
        let u = {
            let mut v = seeded_vec(101, d);
            let nv = norm(&v);
            for x in &mut v {
                *x /= nv;
            }
            v
        };
        let w_g = rank1_matrix(&u, d, 5.0);
        let gram = gram_of(&w_g, d);

        let mut r = seeded_matrix(7, n, d);
        let r_before = r.clone();
        let grams: Vec<&[f32]> = vec![&gram];
        let mut scratch = PowerRetractScratch::new(d);
        let _res = manifold_power_iter_router(&mut r, &grams, n, d, 1.0, 1, &mut scratch);

        let cos_before = {
            let dot = simd_dot_f32(&r_before, &u, d);
            let nb = norm(&r_before);
            dot / (nb * norm(&u))
        };
        let cos_after = {
            let dot = simd_dot_f32(&r, &u, d);
            let na = norm(&r);
            dot / (na * norm(&u))
        };
        assert!(
            cos_after.abs() > 0.95,
            "cos(R', u) = {} should be > 0.95 for iters=1 (before={})",
            cos_after.abs(),
            cos_before.abs()
        );
    }

    #[test]
    fn t01b_synthetic_principal_direction_recovery_iters5() {
        // At iters=5, alignment should be even tighter (cos > 0.99).
        let d = 16usize;
        let n = 1usize;
        let u = {
            let mut v = seeded_vec(202, d);
            let nv = norm(&v);
            for x in &mut v {
                *x /= nv;
            }
            v
        };
        let w_g = rank1_matrix(&u, d, 5.0);
        let gram = gram_of(&w_g, d);

        let mut r = seeded_matrix(9, n, d);
        let grams: Vec<&[f32]> = vec![&gram];
        let mut scratch = PowerRetractScratch::new(d);
        let _res = manifold_power_iter_router(&mut r, &grams, n, d, 1.0, 5, &mut scratch);

        let cos = {
            let dot = simd_dot_f32(&r, &u, d);
            let na = norm(&r);
            dot / (na * norm(&u))
        };
        assert!(
            cos.abs() > 0.99,
            "cos(R', u) = {} should be > 0.99 for iters=5",
            cos.abs()
        );
    }

    #[test]
    fn t02_determinism_byte_identical() {
        // Same (R, M, c_prime, iters) → byte-identical R' across two runs.
        let d = 8usize;
        let n = 3usize;
        let mut r1 = seeded_matrix(5, n, d);
        let mut r2 = seeded_matrix(5, n, d);
        let grams: Vec<Vec<f32>> = (0..n).map(|i| gram_of(&seeded_matrix(100 + i as u64, d, d), d)).collect();
        let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

        let mut s1 = PowerRetractScratch::new(d);
        let mut s2 = PowerRetractScratch::new(d);
        let _ = manifold_power_iter_router(&mut r1, &grams_ref, n, d, 1.0, 1, &mut s1);
        let _ = manifold_power_iter_router(&mut r2, &grams_ref, n, d, 1.0, 1, &mut s2);

        assert_eq!(r1, r2, "R' must be byte-identical across runs (G5 determinism)");
    }

    #[test]
    fn t03_norm_invariant_after_retraction() {
        // ‖R'[i]‖₂ ≈ C = c_prime/√N for all i.
        let d = 12usize;
        let n = 4usize;
        let c_prime = 2.0f32;
        let target = c_prime / (n as f32).sqrt();
        let mut r = seeded_matrix(3, n, d);
        let grams: Vec<Vec<f32>> = (0..n).map(|i| gram_of(&seeded_matrix(50 + i as u64, d, d), d)).collect();
        let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();
        let mut scratch = PowerRetractScratch::new(d);
        let res = manifold_power_iter_router(&mut r, &grams_ref, n, d, c_prime, 2, &mut scratch);

        for i in 0..n {
            let row = &r[i * d..(i + 1) * d];
            let ni = norm(row);
            assert!(
                (ni - target).abs() < 1e-4,
                "row {} norm {} != target {}",
                i,
                ni,
                target
            );
        }
        // maxvio should be ~0.
        assert!(
            res.maxvio < 1e-4,
            "maxvio {} should be ~0 after retraction",
            res.maxvio
        );
    }

    #[test]
    fn t04_lambda_monotone_in_iters() {
        // lambda_alignment should (weakly) increase with iters on a fixed (R, M).
        // This is the Rayleigh-quotient ascent story (paper §3.3).
        let d = 16usize;
        let n = 2usize;
        let grams: Vec<Vec<f32>> = (0..n)
            .map(|i| gram_of(&seeded_matrix(70 + i as u64, d, d), d))
            .collect();
        let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

        let mut prev = -1.0f32;
        for iters in [1u8, 3, 7, 15] {
            let mut r = seeded_matrix(11, n, d);
            let mut scratch = PowerRetractScratch::new(d);
            let res = manifold_power_iter_router(&mut r, &grams_ref, n, d, 1.0, iters, &mut scratch);
            // λ is direction-only (R' is normalized to C each iter, so λ
            // reflects angular alignment, not magnitude). Monotone non-decreasing.
            assert!(
                res.lambda_alignment >= prev - 1e-5,
                "λ decreased at iters={}: {} < {} (Rayleigh ascent violated)",
                iters,
                res.lambda_alignment,
                prev
            );
            prev = res.lambda_alignment;
        }
    }

    #[test]
    fn t05_zero_row_safety() {
        // A zero Gram (all-zero expert) must not panic and the
        // corresponding router row is left at target_norm in its input
        // direction (or zero if input was zero). Mirror gauge zero-matrix test.
        let d = 4usize;
        let n = 2usize;
        let mut r = seeded_matrix(1, n, d);
        // Zero gram for expert 0; non-zero for expert 1.
        let gram0 = vec![0.0f32; d * d];
        let gram1 = gram_of(&seeded_matrix(2, d, d), d);
        let grams_ref: Vec<&[f32]> = vec![gram0.as_slice(), gram1.as_slice()];
        let mut scratch = PowerRetractScratch::new(d);
        // Must not panic.
        let res = manifold_power_iter_router(&mut r, &grams_ref, n, d, 1.0, 1, &mut scratch);
        // All entries finite.
        for x in &r {
            assert!(x.is_finite(), "non-finite after zero-gram pass: {}", x);
        }
        // λ is well-defined (0 for the zero-gram row).
        assert!(res.lambda_alignment.is_finite());
    }

    #[test]
    fn t06_sigmoid_gate_independence() {
        // Changing one expert's router row score MUST NOT change another's.
        // (Softmax would couple them; sigmoid does not — G7 runtime check.)
        let d = 4usize;
        let n = 3usize;
        let x = seeded_vec(42, d);
        let r = seeded_matrix(13, n, d);

        let mut scores_a = vec![0.0f32; n];
        let mut scores_b = vec![0.0f32; n];
        gate_sigmoid_topk(&x, &r, n, d, 1.0, n, &mut scores_a);

        // Perturb row 0 only — scale by 2x.
        let mut r_perturbed = r.clone();
        for v in r_perturbed.iter_mut().take(d) {
            *v *= 2.0;
        }
        gate_sigmoid_topk(&x, &r_perturbed, n, d, 1.0, n, &mut scores_b);

        // Experts 1, 2 scores must be byte-identical (sigmoid independence).
        for i in 1..n {
            assert!(
                (scores_a[i] - scores_b[i]).abs() < 1e-7,
                "expert {} score changed by {} — sigmoid independence violated",
                i,
                (scores_a[i] - scores_b[i]).abs()
            );
        }
        // Expert 0 should have changed (sanity).
        assert!(
            (scores_a[0] - scores_b[0]).abs() > 1e-6,
            "expert 0 should have changed after 2x scaling"
        );
    }

    #[test]
    fn t07_compute_expert_gram_matches_naive() {
        // Blocked matmul result must match naive triple-loop.
        let d = 12usize;
        let w = seeded_matrix(99, d, d);
        let mut gram = vec![0.0f32; d * d];
        compute_expert_gram_into(&w, d, &mut gram);

        // Naive reference.
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0f32;
                for k in 0..d {
                    s += w[i * d + k] * w[j * d + k];
                }
                let rel = (gram[i * d + j] - s).abs() / s.abs().max(1e-3);
                assert!(
                    rel < 1e-3,
                    "gram[{},{}] = {} vs naive {} (rel {})",
                    i,
                    j,
                    gram[i * d + j],
                    s,
                    rel
                );
            }
        }
    }

    #[test]
    fn t08_snapshot_hook_cache_hit_is_deterministic() {
        // Same snapshot_version → byte-identical R' (cache hit).
        let d = 8usize;
        let n = 2usize;
        let grams: Vec<Vec<f32>> = (0..n)
            .map(|i| gram_of(&seeded_matrix(80 + i as u64, d, d), d))
            .collect();
        let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

        let mut hook = DefaultMpiRouterSnapshotHook::new(MpiRouterConfig::default(), d);
        let mut r1 = seeded_matrix(17, n, d);
        let mut r2 = seeded_matrix(17, n, d);

        let _ = hook.recondition_at_swap(&mut r1, &grams_ref, n, d, 42);
        let _ = hook.recondition_at_swap(&mut r2, &grams_ref, n, d, 42);
        assert_eq!(r1, r2, "cache hit must be deterministic");
    }

    #[test]
    fn t09_snapshot_hook_version_bump_invalidates_cache() {
        // Different snapshot_version → cache miss → still correct (re-fills).
        let d = 4usize;
        let n = 1usize;
        let grams: Vec<Vec<f32>> = vec![gram_of(&seeded_matrix(33, d, d), d)];
        let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

        let mut hook = DefaultMpiRouterSnapshotHook::new(MpiRouterConfig::default(), d);
        let r_seed = seeded_matrix(19, n, d);
        // Version 0 → miss; Version 1 → miss; both should produce identical
        // output because grams are the same (reset r between calls since
        // MPI mutates in place).
        let mut r_a = r_seed.clone();
        let _ = hook.recondition_at_swap(&mut r_a, &grams_ref, n, d, 0);
        let mut r_b = r_seed.clone();
        let _ = hook.recondition_at_swap(&mut r_b, &grams_ref, n, d, 1);
        // Same grams → same R' regardless of cache state.
        assert_eq!(r_a, r_b, "same grams must give same R' across version bumps");
    }

    #[test]
    fn t10_gate_sigmoid_topk_into_matches_allocating() {
        // The zero-alloc `gate_sigmoid_topk_into` MUST yield the same indices
        // (in the same order) as the allocating `gate_sigmoid_topk`.
        let d = 6usize;
        let n = 5usize;
        let r = seeded_matrix(31, n, d);
        let x = seeded_vec(7, d);
        let beta = 1.3f32;
        let k = 3usize;

        // Allocating path.
        let mut scores_a = vec![0.0f32; n];
        let topk_vec = gate_sigmoid_topk(&x, &r, n, d, beta, k, &mut scores_a);

        // Zero-alloc path.
        let mut scores_b = vec![0.0f32; n];
        let mut idx_buf = vec![0usize; n];
        let kk = gate_sigmoid_topk_into(&x, &r, n, d, beta, k, &mut scores_b, &mut idx_buf);

        assert_eq!(kk, topk_vec.len(), "kk must equal length of allocating variant");
        assert_eq!(&idx_buf[..kk], topk_vec.as_slice(), "indices must match");
        assert_eq!(scores_a, scores_b, "score buffer contents must match");
    }

    #[test]
    fn t11_gate_sigmoid_topk_into_handles_k_larger_than_n() {
        // k > n_experts → kk clamps to n_experts; no out-of-bounds writes.
        let d = 4usize;
        let n = 3usize;
        let r = seeded_matrix(11, n, d);
        let x = seeded_vec(2, d);

        let mut scores = vec![0.0f32; n];
        let mut idx_buf = vec![0usize; n];
        let kk = gate_sigmoid_topk_into(&x, &r, n, d, 1.0, n + 5, &mut scores, &mut idx_buf);

        assert_eq!(kk, n, "kk must clamp to n_experts when k > n");
        // All indices present exactly once.
        let mut sorted = idx_buf[..kk].to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<usize>>());
    }

    #[test]
    fn t12_compute_diagnostics_with_scratch_matches_allocating() {
        // `compute_diagnostics_with_scratch` MUST yield identical (λ, MaxVio)
        // to `compute_diagnostics`.
        let d = 8usize;
        let n = 3usize;
        let r = seeded_matrix(51, n, d);
        let grams: Vec<Vec<f32>> = (0..n)
            .map(|i| gram_of(&seeded_matrix(200 + i as u64, d, d), d))
            .collect();
        let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();
        let target = 1.0f32 / (n as f32).sqrt();

        let (lambda_a, maxvio_a) = compute_diagnostics(&r, &grams_ref, n, d, target);

        let mut rm = vec![0.0f32; d];
        let (lambda_b, maxvio_b) =
            compute_diagnostics_with_scratch(&r, &grams_ref, n, d, target, &mut rm);

        let bits = |f: f32| f.to_bits();
        assert_eq!(bits(lambda_a), bits(lambda_b), "λ must be byte-identical");
        assert_eq!(bits(maxvio_a), bits(maxvio_b), "MaxVio must be byte-identical");
    }

    #[test]
    fn t13_router_inplace_matches_allocating() {
        // `manifold_power_iter_router_inplace` MUST yield:
        //  - identical mutated `r` (byte-for-byte)
        //  - identical (λ, MaxVio) diagnostics
        // as `manifold_power_iter_router`.
        let d = 8usize;
        let n = 3usize;
        let c_prime = 1.0f32;
        let target = c_prime / (n as f32).sqrt();

        let r_seed = seeded_matrix(5, n, d);
        let grams: Vec<Vec<f32>> = (0..n)
            .map(|i| gram_of(&seeded_matrix(100 + i as u64, d, d), d))
            .collect();
        let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

        // Allocating path.
        let mut r_a = r_seed.clone();
        let mut s_a = PowerRetractScratch::new(d);
        let res_a = manifold_power_iter_router(&mut r_a, &grams_ref, n, d, c_prime, 1, &mut s_a);

        // Zero-alloc path.
        let mut r_b = r_seed.clone();
        let mut s_b = PowerRetractScratch::new(d);
        let (lambda_b, maxvio_b) =
            manifold_power_iter_router_inplace(&mut r_b, &grams_ref, n, d, target, 1, &mut s_b);

        assert_eq!(r_a, r_b, "R' mutated buffer must be byte-identical");
        assert_eq!(res_a.r_prime, r_b, "r_prime field must equal mutated r_b");
        assert_eq!(
            res_a.lambda_alignment.to_bits(),
            lambda_b.to_bits(),
            "λ must match byte-for-byte"
        );
        assert_eq!(
            res_a.maxvio.to_bits(),
            maxvio_b.to_bits(),
            "MaxVio must match byte-for-byte"
        );
    }
}
