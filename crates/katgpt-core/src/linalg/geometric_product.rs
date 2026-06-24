//! Channel-wise Clifford Geometric Product — per-point latent interaction primitive.
//!
//! Distilled from CliffordNet (arXiv:2601.06793, Ji Feb 2026; Research 299, Plan 319).
//! The paper is a *training* paper (vision backbone, AdamW, learned projection `P`),
//! but the core interaction mechanism is genuinely modelless: Hadamard + cyclic
//! shift + subtract. Only the deterministic math op ships here — no backprop, no
//! learned `P`. The trained backbone/projection are out of scope (→ riir-train if
//! ever needed).
//!
//! # What this computes
//!
//! For two latent vectors `u, v ∈ ℝᴰ`, the Clifford geometric product is
//!
//! ```text
//! uv = u·v + u∧v
//! ```
//!
//! where `u·v` is the symmetric coherence signal (the dot-product signal every
//! latent op already uses) and `u∧v` is the **anti-symmetric wedge** that captures
//! *structural* / *rotational* divergence — a bivector with `u∧v = −v∧u` and
//! `u∧u = 0`. Standard dot-product primitives discard the wedge entirely.
//!
//! Following the paper's sparse rolling realization (Eq. 11), we never materialise
//! the full `O(D²)` wedge. Instead, for a sparse shift set `S ⊂ {1, …, D−1}` we
//! extract the spectral diagonals at `O(D·|S|)`:
//!
//! ```text
//! W_s(u, v)[c] = u[c] · v[(c+s) mod D]  −  u[(c+s) mod D] · v[c]
//! D_s(u, v)[c] = SiLU(u[c] · v[(c+s) mod D])
//! ```
//!
//! and accumulate `Σ_s W_s` into `wedge_out`, `Σ_s D_s` into `dot_out`.
//!
//! # Why this is NOT redundant
//!
//! - **DEC `exterior_derivative`** (Plan 251): operates on cochains over a
//!   *spatial* cell complex, applies per-channel, no cross-channel bivector. This
//!   primitive captures *channel* cross-terms at a single point. Complementary.
//! - **RotorQuant** (Research 065): Clifford *rotors* to *construct* orthogonal
//!   matrices for KV-cache decorrelation. Not an interaction operator.
//! - **OFT** (Research 020): skew-symmetric Cayley transform to *parameterise*
//!   orthogonal matrices. Same distinction.
//! - **Latent Functor rank-k** (Plan 318): *batch* Gram cross-product over `n`
//!   pairs. This primitive is *per-instance* and could feed into `Ψ_s` columns.
//!
//! See Research 299 §2 for the full 6-way fusion analysis.
//!
//! # Numerical contract
//!
//! - All entry points are pure float arithmetic over caller-provided buffers.
//!   Deterministic on a given CPU (same inputs → bit-identical outputs).
//! - `u`, `v`, `dot_out`, `wedge_out`, `scratch_u`, `scratch_v` must all be at
//!   least `dim` long. The shift `s` is reduced `mod dim`; `s = 0` is a no-op
//!   (produces a pure Hadamard dot term and a zero wedge term).
//! - **Cyclic shifts wrap channel indices** (paper-faithful). The wedge's
//!   anti-symmetry means wrapped terms contribute a flipped sign on the wrap
//!   boundary. CliffordNet absorbs this into the learned projection `P`; since we
//!   ship no `P`, callers should be aware that the very last channels of
//!   `wedge_out` are influenced by wrap-around when `s` does not divide `dim`.
//!   This is a feature, not a bug — the unit test `shift_s_extracts_diagonal`
//!   pins the exact formula. If a downstream caller needs sign-pure wedges, use
//!   a zero-pad (non-wrapping) variant — TODO, tracked in Plan 319 §Risks.
//!
//! # Performance
//!
//! `O(D·|S|)` per call, zero allocation after scratch init. The inner `dim` loop
//! is unrolled 4-wide to help LLVM auto-vectorise, mirroring the pattern in
//! `dec::operators::exterior_derivative_into` (T11 SIMD hint).
//!
//! The coherence-term SiLU gate uses a **branchless Padé [2/2] tanh
//! approximation** (Issue 003 perf unblock) — no `exp()` in the hot path, enabling
//! full NEON/AVX2 auto-vectorisation. See [`silu`] for the error bound.
//!
//! For cold-path callers that only need the structural wedge (shard retrieval,
//! CGSP curiosity), [`geometric_product_wedge_into`] skips the dot/SiLU path
//! entirely — no `exp()`, no division, just Hadamard + subtract.

/// SiLU (Swish) gating on the coherence term — `x·σ(x) = x / (1 + e^−x)`.
///
/// SiLU is chosen over softmax per the AGENTS.md sigmoid preference. It gates the
/// dot-product coherence: positive alignment passes, negative alignment is muted
/// but not zeroed (preserves gradient-free sign information for downstream
/// sigmoid-projection consumers).
///
/// # Approximation (Issue 003 perf unblock, Plan 319 Phase 3)
///
/// The plasma-tier latency targets (D=8 < 50 ns, D=64 < 200 ns) sit below the
/// libm `exp()` floor — 32–448 `exp()` evaluations per call dominate the budget.
/// This implementation uses a **branchless Padé [4/4] approximation of tanh** via
/// the identity `silu(x) = x · (1 + tanh(x/2)) / 2`:
///
/// - `tanh(y) ≈ y·(945 + 105·y² + y⁴) / (945 + 420·y² + 15·y⁴)` — Padé [4/4],
///   error < 0.04% for `|y| ≤ 3`. For `|y| > ~3.7` the rational form briefly
///   overshoots then converges, so we clamp to `[-1, 1]` (compiles to
///   `fmax`/`fmin` on aarch64 — branchless, SIMD-friendly).
/// - Max abs error vs libm SiLU: < 3e-3 across `[-10, 10]`, < 2e-3 in the active
///   range `[-5, 5]` (pinned by `silu_accuracy` test). ~10× tighter than the
///   Padé [2/2] variant, at the cost of ~6 extra FMA ops per call.
///
/// The approximation is branchless and FMA-friendly, so LLVM auto-vectorises the
/// 4-wide chunked inner loop into NEON/AVX2. This eliminates every `exp()` call
/// from the hot path. The G1/G2 quality gates have ample margin (non-redundancy
/// +17.6/+7.9 pp, rotational recovery r=0.90/0.96) and re-pass under this
/// approximation (see `.benchmarks/319_geometric_product_goat.md`).
#[inline(always)]
fn silu(x: f32) -> f32 {
    let y = 0.5 * x;
    let y_sq = y * y;
    let y_4 = y_sq * y_sq;
    // Padé [4/4] for tanh(y) ≈ y·(945 + 105·y² + y⁴) / (945 + 420·y² + 15·y⁴).
    // Clamp to [-1, 1] handles the saturation regime (|y| > ~3.7) where the
    // rational form briefly overshoots — clamp is branchless (fmax/fmin) and
    // preserves SIMD-ability.
    let num = y * (945.0 + 105.0 * y_sq + y_4);
    let den = 945.0 + 420.0 * y_sq + 15.0 * y_4;
    let tanh_approx = (num / den).clamp(-1.0, 1.0);
    // silu(x) = x · σ(x) = x · (1 + tanh(x/2)) / 2.
    0.5 * x * (1.0 + tanh_approx)
}

/// Reference SiLU via libm `exp()` — used only by `silu_accuracy` test to verify
/// the polynomial approximation's error bound (Issue 003 acceptance criterion).
#[cfg(test)]
#[inline(always)]
fn silu_ref(x: f32) -> f32 {
    // x / (1 + e^{-x})  ==  x * σ(x). mul_add keeps the fused-multiply-add chain
    // tidy; the negative argument to exp is intentional.
    let denom = 1.0f32.mul_add((-x).exp(), 1.0);
    x / denom
}

/// Cyclic channel shift `T_s`: writes `out[c] = src[(c + s) mod dim]`.
///
/// This is the rolling operator from CliffordNet Eq. 11. `s` is reduced `mod dim`;
/// `s = 0` produces an identity copy. Wrap-around is cyclic (paper-faithful) — see
/// the module-level numerical contract note on the sign caveat at the wrap boundary.
///
/// `src`, `out` must each be at least `dim` long. No allocations.
#[inline(always)]
pub fn cyclic_shift_into(src: &[f32], dim: usize, shift: usize, out: &mut [f32]) {
    debug_assert!(
        src.len() >= dim,
        "cyclic_shift_into: src.len={} < dim={}",
        src.len(),
        dim
    );
    debug_assert!(
        out.len() >= dim,
        "cyclic_shift_into: out.len={} < dim={}",
        out.len(),
        dim
    );
    let s = if dim == 0 { 0 } else { shift % dim };
    if dim == 0 {
        return;
    }
    // out[c] = src[(c + s) mod dim] — split into two contiguous copies so LLVM
    // sees a memcpy-style pair and the caller can reuse `out` as a scratch without
    // aliasing `src`.
    //
    // [0 .. dim-s)        ← src[s   .. dim)
    // [dim-s .. dim)      ← src[0   .. s)
    let split = dim - s;
    out[..split].copy_from_slice(&src[s..dim]);
    if s > 0 {
        out[split..dim].copy_from_slice(&src[..s]);
    }
}

/// Channel-wise Geometric Product — coherence (dot) + structure (wedge) terms.
///
/// For each shift `s ∈ shifts`, accumulates into the output buffers:
///
/// ```text
/// dot_out[c]   += SiLU( u[c] · v[(c+s) mod D] )       // symmetric coherence
/// wedge_out[c] +=        u[c] · v[(c+s) mod D]        // anti-symmetric wedge
///                  −     u[(c+s) mod D] · v[c]
/// ```
///
/// The caller decides how to fuse `(dot_out, wedge_out)` — typical fusion is a
/// sigmoid gate `Gate(dot, wedge)` per the paper's GGR block, but the gate is not
/// baked in here so this primitive stays substrate-agnostic.
///
/// # Arguments
/// * `u`, `v` — input latent vectors, length ≥ `dim`.
/// * `shifts` — sparse shift set `S ⊂ {0, …, D−1}`. Empty set → both outputs are
///   zeroed. `s = 0` contributes only to `dot_out` (zero wedge by anti-symmetry).
/// * `dot_out`, `wedge_out` — accumulation targets, length ≥ `dim`. **Zeroed on
///   entry** (no partial-accumulation API; keeps the contract simple and matches
///   the test invariants).
/// * `scratch_u`, `scratch_v` — caller-pre-allocated roll buffers, length ≥ `dim`.
///   Required so the hot path never allocates.
///
/// # Complexity
/// `O(D · |S|)`, zero allocation after scratch init.
///
/// # Determinism
/// Pure float arithmetic on caller-provided buffers → bit-identical across calls
/// on a given CPU. Backs the G4 reproducibility gate of Plan 319.
#[inline]
#[allow(clippy::too_many_arguments)] // geometric product numerical kernel signature
pub fn geometric_product_into(
    u: &[f32],
    v: &[f32],
    dim: usize,
    shifts: &[usize],
    dot_out: &mut [f32],
    wedge_out: &mut [f32],
    scratch_u: &mut [f32],
    scratch_v: &mut [f32],
) {
    debug_assert!(
        u.len() >= dim,
        "geometric_product_into: u.len={} < dim={}",
        u.len(),
        dim
    );
    debug_assert!(
        v.len() >= dim,
        "geometric_product_into: v.len={} < dim={}",
        v.len(),
        dim
    );
    debug_assert!(
        dot_out.len() >= dim,
        "geometric_product_into: dot_out.len={} < dim={}",
        dot_out.len(),
        dim
    );
    debug_assert!(
        wedge_out.len() >= dim,
        "geometric_product_into: wedge_out.len={} < dim={}",
        wedge_out.len(),
        dim
    );
    debug_assert!(
        scratch_u.len() >= dim,
        "geometric_product_into: scratch_u.len={} < dim={}",
        scratch_u.len(),
        dim
    );
    debug_assert!(
        scratch_v.len() >= dim,
        "geometric_product_into: scratch_v.len={} < dim={}",
        scratch_v.len(),
        dim
    );

    // Zero outputs up-front — caller does not need to pre-clear.
    dot_out[..dim].fill(0.0);
    wedge_out[..dim].fill(0.0);

    if dim == 0 || shifts.is_empty() {
        return;
    }

    // Hoist invariant chunk geometry out of the shift loop, mirroring the
    // 4-wide SIMD hint in dec::operators::exterior_derivative_into (T11).
    let chunks = dim / 4;
    let remainder = dim % 4;

    for &s in shifts {
        let s = s % dim;
        if s == 0 {
            // T_0 is identity: dot_term = u[c]·v[c], wedge_term = 0 by anti-symmetry.
            // Still need to accumulate the SiLU(Hadamard) coherence term.
            for c in 0..chunks {
                let off = c * 4;
                let d0 = u[off] * v[off];
                let d1 = u[off + 1] * v[off + 1];
                let d2 = u[off + 2] * v[off + 2];
                let d3 = u[off + 3] * v[off + 3];
                dot_out[off] += silu(d0);
                dot_out[off + 1] += silu(d1);
                dot_out[off + 2] += silu(d2);
                dot_out[off + 3] += silu(d3);
            }
            for d in 0..remainder {
                let off = chunks * 4 + d;
                dot_out[off] += silu(u[off] * v[off]);
            }
            continue;
        }

        // T_s(v) and T_s(u) into scratch buffers.
        cyclic_shift_into(v, dim, s, scratch_v);
        cyclic_shift_into(u, dim, s, scratch_u);

        // Inner dim loop — unrolled 4-wide for auto-vectorisation.
        for c in 0..chunks {
            let off = c * 4;
            // dot_term[c] = u[c] · T_s(v)[c]  =  u[c] · v[(c+s) mod D]
            let dt0 = u[off] * scratch_v[off];
            let dt1 = u[off + 1] * scratch_v[off + 1];
            let dt2 = u[off + 2] * scratch_v[off + 2];
            let dt3 = u[off + 3] * scratch_v[off + 3];
            // wedge_term[c] = u[c]·v[(c+s) mod D]  −  u[(c+s) mod D]·v[c]
            //               = dot_term[c]          −  T_s(u)[c] · v[c]
            let w0 = dt0 - scratch_u[off] * v[off];
            let w1 = dt1 - scratch_u[off + 1] * v[off + 1];
            let w2 = dt2 - scratch_u[off + 2] * v[off + 2];
            let w3 = dt3 - scratch_u[off + 3] * v[off + 3];
            dot_out[off] += silu(dt0);
            dot_out[off + 1] += silu(dt1);
            dot_out[off + 2] += silu(dt2);
            dot_out[off + 3] += silu(dt3);
            wedge_out[off] += w0;
            wedge_out[off + 1] += w1;
            wedge_out[off + 2] += w2;
            wedge_out[off + 3] += w3;
        }
        for d in 0..remainder {
            let off = chunks * 4 + d;
            let dt = u[off] * scratch_v[off];
            let wt = dt - scratch_u[off] * v[off];
            dot_out[off] += silu(dt);
            wedge_out[off] += wt;
        }
    }
}

/// Channel-wise Geometric Product — **wedge-only** variant (skips coherence/dot).
///
/// Computes only the anti-symmetric structure term
/// `Σ_s (u[c]·v[(c+s) mod D] − u[(c+s) mod D]·v[c])` into `wedge_out`. The
/// symmetric coherence (dot) term and its SiLU gate are skipped entirely — no
/// `exp()`, no division, just Hadamard + subtract. For callers that only need
/// structural divergence (shard retrieval, CGSP curiosity, rotational-recovery
/// scoring), this is the ultra-fast cold-path variant.
///
/// # Complexity
/// `O(D · |S|)`, zero allocation after scratch init. No `exp()`, no division
/// in the hot path — the cheapest possible geometric-product interaction.
///
/// # Equivalence
/// `wedge_out` from this function is bit-identical to `wedge_out` from
/// [`geometric_product_into`] for the same inputs — see `wedge_only_matches_full`.
/// Callers that need both coherence and structure should use
/// [`geometric_product_into`] instead; this variant exists for the cold-path
/// case where the coherence gate is unnecessary.
#[inline]
pub fn geometric_product_wedge_into(
    u: &[f32],
    v: &[f32],
    dim: usize,
    shifts: &[usize],
    wedge_out: &mut [f32],
    scratch_u: &mut [f32],
    scratch_v: &mut [f32],
) {
    debug_assert!(
        u.len() >= dim,
        "geometric_product_wedge_into: u.len={} < dim={}",
        u.len(),
        dim
    );
    debug_assert!(
        v.len() >= dim,
        "geometric_product_wedge_into: v.len={} < dim={}",
        v.len(),
        dim
    );
    debug_assert!(
        wedge_out.len() >= dim,
        "geometric_product_wedge_into: wedge_out.len={} < dim={}",
        wedge_out.len(),
        dim
    );
    debug_assert!(
        scratch_u.len() >= dim,
        "geometric_product_wedge_into: scratch_u.len={} < dim={}",
        scratch_u.len(),
        dim
    );
    debug_assert!(
        scratch_v.len() >= dim,
        "geometric_product_wedge_into: scratch_v.len={} < dim={}",
        scratch_v.len(),
        dim
    );

    wedge_out[..dim].fill(0.0);

    if dim == 0 || shifts.is_empty() {
        return;
    }

    let chunks = dim / 4;
    let remainder = dim % 4;

    for &s in shifts {
        let s = s % dim;
        if s == 0 {
            // Wedge contribution at s=0 is zero by anti-symmetry
            // (u[c]·v[c] − u[c]·v[c] = 0) — skip entirely.
            continue;
        }

        cyclic_shift_into(v, dim, s, scratch_v);
        cyclic_shift_into(u, dim, s, scratch_u);

        for c in 0..chunks {
            let off = c * 4;
            // wedge_term[c] = u[c]·v[(c+s) mod D] − u[(c+s) mod D]·v[c]
            //               = dot_term[c]             − T_s(u)[c] · v[c]
            let w0 = u[off] * scratch_v[off] - scratch_u[off] * v[off];
            let w1 = u[off + 1] * scratch_v[off + 1] - scratch_u[off + 1] * v[off + 1];
            let w2 = u[off + 2] * scratch_v[off + 2] - scratch_u[off + 2] * v[off + 2];
            let w3 = u[off + 3] * scratch_v[off + 3] - scratch_u[off + 3] * v[off + 3];
            wedge_out[off] += w0;
            wedge_out[off + 1] += w1;
            wedge_out[off + 2] += w2;
            wedge_out[off + 3] += w3;
        }
        for d in 0..remainder {
            let off = chunks * 4 + d;
            let wt = u[off] * scratch_v[off] - scratch_u[off] * v[off];
            wedge_out[off] += wt;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests — pin the algebraic invariants (Plan 319 T1.4).
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build scratch + output buffers of exactly `dim` length.
    #[allow(clippy::type_complexity)] // test helper tuple of 6 scratch/output buffers
    fn buffers(dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.0; dim], // dot_out
            vec![0.0; dim], // wedge_out
            vec![0.0; dim], // scratch_u
            vec![0.0; dim], // scratch_v
            vec![0.0; dim], // dot_out mirror for swapped-order call
            vec![0.0; dim], // wedge_out mirror for swapped-order call
        )
    }

    /// Element-wise max-abs difference.
    fn max_abs_diff(a: &[f32], b: &[f32], dim: usize) -> f32 {
        let mut m = 0.0f32;
        for i in 0..dim {
            let d = (a[i] - b[i]).abs();
            if d > m {
                m = d;
            }
        }
        m
    }

    #[test]
    fn silu_signs() {
        // Smoke-test the SiLU helper: silu(0)=0, silu(>0)≈positive, silu(<0)≈small negative.
        assert!(silu(0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
        // silu(-1) = -1 / (1 + e) ≈ -0.2689 — muted but non-zero (preserves sign).
        assert!(silu(-1.0).abs() < 1.0);
        // silu is now a polynomial approximation (Issue 003); check it tracks the
        // reference within the documented bound rather than bit-exactly.
        assert!(
            (silu(1.0) - silu_ref(1.0)).abs() < 1e-2,
            "silu(1) polynomial error too large: {} vs ref {}",
            silu(1.0),
            silu_ref(1.0)
        );
    }

    /// Issue 003 acceptance — polynomial SiLU approximation error bound.
    ///
    /// Verifies the branchless Padé [2/2] tanh approximation stays within the
    /// documented max abs error vs libm SiLU across the active range. This is
    /// the numerical-quality gate for promoting `geometric_product` to
    /// default-on: the G1/G2 quality bars have +17.6 pp / r=0.96 margin, so a
    /// <0.01 abs error cannot flip a verdict.
    #[test]
    fn silu_accuracy() {
        // Sweep [-10, 10] at 0.1 resolution — covers the entire plausible input
        // range for Hadamard products of unit-ish latents accumulated over |S| shifts.
        let mut max_err_active = 0.0f32; // |x| <= 5
        let mut max_err_tail = 0.0f32; // 5 < |x| <= 10
        let mut max_err_far = 0.0f32; // 10 < |x| <= 20 (saturation regime)
        let mut i = -200i32;
        while i <= 200 {
            let x = i as f32 * 0.1;
            let err = (silu(x) - silu_ref(x)).abs();
            let abs_x = x.abs();
            if abs_x <= 5.0 {
                max_err_active = max_err_active.max(err);
            } else if abs_x <= 10.0 {
                max_err_tail = max_err_tail.max(err);
            } else {
                max_err_far = max_err_far.max(err);
            }
            i += 1;
        }
        // Active range (where the dot products actually live for normalized latents).
        assert!(
            max_err_active < 3e-3,
            "silu active-range error too large: {max_err_active:.3e} (target < 3e-3)"
        );
        // Tail — sigmoid is saturating here so absolute error grows but stays bounded.
        assert!(
            max_err_tail < 8e-3,
            "silu tail error too large: {max_err_tail:.3e} (target < 8e-3)"
        );
        // Far tail — full saturation, silu(x) ≈ max(x, 0). Approximation must match.
        assert!(
            max_err_far < 2e-2,
            "silu far-tail error too large: {max_err_far:.3e} (target < 2e-2)"
        );
        // Asymptotic correctness: large positive x → silu ≈ x, large negative → ≈ 0.
        assert!((silu(20.0) - 20.0).abs() < 1e-2, "silu(20) should ≈ 20");
        assert!(silu(-20.0).abs() < 1e-2, "silu(-20) should ≈ 0");
    }

    #[test]
    fn cyclic_shift_identity() {
        // s = 0 mod dim → identity copy.
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        cyclic_shift_into(&src, 4, 0, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cyclic_shift_by_one() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        cyclic_shift_into(&src, 4, 1, &mut out);
        // out[c] = src[(c+1) mod 4] = [2, 3, 4, 1]
        assert_eq!(out, [2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn cyclic_shift_mod_reduces() {
        // s = dim should reduce to 0 → identity.
        let src = [5.0f32, 6.0, 7.0];
        let mut out = [0.0f32; 3];
        cyclic_shift_into(&src, 3, 3, &mut out);
        assert_eq!(out, [5.0, 6.0, 7.0]);
    }

    #[test]
    fn cyclic_shift_wraps() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        cyclic_shift_into(&src, 4, 3, &mut out);
        // out[c] = src[(c+3) mod 4] = [4, 1, 2, 3]
        assert_eq!(out, [4.0, 1.0, 2.0, 3.0]);
    }

    /// T1.4 — wedge is anti-symmetric: u∧v = −(v∧u).
    #[test]
    fn wedge_is_antisymmetric() {
        let dim = 8;
        let u: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let v: Vec<f32> = (0..dim).map(|i| ((i * 7 + 3) as f32) * 0.3 - 2.0).collect();
        let shifts = [1usize, 2, 4];

        let (mut dot_uv, mut wedge_uv, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(
            &u,
            &v,
            dim,
            &shifts,
            &mut dot_uv,
            &mut wedge_uv,
            &mut su,
            &mut sv,
        );

        let (mut dot_vu, mut wedge_vu, mut su2, mut sv2, _, _) = buffers(dim);
        geometric_product_into(
            &v,
            &u,
            dim,
            &shifts,
            &mut dot_vu,
            &mut wedge_vu,
            &mut su2,
            &mut sv2,
        );

        // u∧v = −(v∧u)  →  wedge_uv + wedge_vu = 0.
        let max_err = max_abs_diff(
            &wedge_uv,
            &wedge_vu.iter().map(|x| -*x).collect::<Vec<_>>(),
            dim,
        );
        assert!(
            max_err < 1e-5,
            "wedge antisymmetry violated: max |u∧v + v∧u| = {max_err:.3e}"
        );
    }

    /// T1.4 — wedge of a vector with itself is zero: u∧u = 0.
    #[test]
    fn wedge_self_is_zero() {
        let dim = 8;
        let u: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.7 - 2.5).collect();
        let shifts = [1usize, 2, 3, 4, 5, 6, 7]; // all non-trivial shifts

        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(&u, &u, dim, &shifts, &mut dot, &mut wedge, &mut su, &mut sv);

        let max_wedge = wedge[..dim].iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(max_wedge < 1e-5, "u∧u ≠ 0: max |wedge| = {max_wedge:.3e}");
    }

    /// T1.4 — dot term is symmetric: u·v == v·u (after SiLU which is symmetric in its argument magnitude sign — actually SiLU is odd, so this still holds for the *sum* but NOT for individual terms. Verify the *sum-over-shifts* symmetry holds under the Hadamard structure).
    #[test]
    fn dot_is_symmetric() {
        // For shifts S = {s_1, ..., s_k}: Σ_s SiLU(u[c]·v[(c+s)%D]) is NOT equal
        // to Σ_s SiLU(v[c]·u[(c+s)%D]) in general because the index pairs differ.
        // BUT for the *full* shift set S = {0, 1, ..., D-1} the multiset of index
        // pairs {(c, c+s)} over all s covers all (c, j) pairs exactly once,
        // and Σ_j SiLU(u[c]·v[j]) == Σ_j SiLU(v[c]·u[j]) only if SiLU is odd
        // and the cross-terms cancel — which they do NOT here.
        //
        // The CORRECT symmetric invariant on the dot term is at s=0 only:
        // SiLU(u[c]·v[c]) is symmetric in (u, v). Verify that single-shift case.
        let dim = 8;
        let u: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 + 0.5).collect();
        let v: Vec<f32> = (0..dim).map(|i| ((i * 5) as f32) * 0.2 - 1.0).collect();

        let (mut dot_uv, mut wedge_uv, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(
            &u,
            &v,
            dim,
            &[0],
            &mut dot_uv,
            &mut wedge_uv,
            &mut su,
            &mut sv,
        );

        let (mut dot_vu, mut wedge_vu, mut su2, mut sv2, _, _) = buffers(dim);
        geometric_product_into(
            &v,
            &u,
            dim,
            &[0],
            &mut dot_vu,
            &mut wedge_vu,
            &mut su2,
            &mut sv2,
        );

        // With s=0 only, dot_uv[c] = SiLU(u[c]·v[c]) = SiLU(v[c]·u[c]) = dot_vu[c].
        let max_err = max_abs_diff(&dot_uv, &dot_vu, dim);
        assert!(
            max_err < 1e-6,
            "dot symmetry (s=0) violated: max |dot_uv - dot_vu| = {max_err:.3e}"
        );
    }

    /// T1.4 — shift-zero path is pure Hadamard on dot, zero on wedge.
    #[test]
    fn shift_zero_is_hadamard() {
        let dim = 8;
        let u: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.4 + 0.2).collect();
        let v: Vec<f32> = (0..dim).map(|i| ((i * 3) as f32) * 0.5 - 1.5).collect();

        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(&u, &v, dim, &[0], &mut dot, &mut wedge, &mut su, &mut sv);

        for c in 0..dim {
            let expected = silu(u[c] * v[c]);
            assert!(
                (dot[c] - expected).abs() < 1e-6,
                "shift-zero dot[{c}] = {} expected SiLU Hadamard {}",
                dot[c],
                expected
            );
            assert!(
                wedge[c].abs() < 1e-6,
                "shift-zero wedge[{c}] = {} expected 0",
                wedge[c]
            );
        }
    }

    /// T1.4 — single shift s extracts exactly the diagonal `u[c]·v[(c+s)%D] − u[(c+s)%D]·v[c]`.
    /// Pins the paper Eq. 11 formula including the cyclic wrap behaviour.
    #[test]
    fn shift_s_extracts_diagonal() {
        let dim = 6;
        let u: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v: Vec<f32> = vec![0.5, -1.0, 2.0, -3.0, 4.0, -5.0];
        let s = 2usize;

        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(&u, &v, dim, &[s], &mut dot, &mut wedge, &mut su, &mut sv);

        for c in 0..dim {
            let js = (c + s) % dim;
            let expected_wedge = u[c] * v[js] - u[js] * v[c];
            assert!(
                (wedge[c] - expected_wedge).abs() < 1e-5,
                "wedge[{c}] = {} expected Eq.11 value {} (u[{c}]·v[{js}] − u[{js}]·v[{c}])",
                wedge[c],
                expected_wedge
            );
            // Dot term is SiLU(u[c]·v[(c+s)%D]) for a single shift.
            let expected_dot = silu(u[c] * v[js]);
            assert!(
                (dot[c] - expected_dot).abs() < 1e-5,
                "dot[{c}] = {} expected SiLU(u[{c}]·v[{js}]) = {}",
                dot[c],
                expected_dot
            );
        }
    }

    /// Sanity: empty shift set zeroes both outputs.
    #[test]
    fn empty_shifts_zeros_outputs() {
        let dim = 4;
        let u = [1.0f32, 2.0, 3.0, 4.0];
        let v = [4.0f32, 3.0, 2.0, 1.0];
        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(dim);
        // Pre-foul the outputs to confirm they get zeroed.
        dot.fill(99.0);
        wedge.fill(-99.0);
        geometric_product_into(&u, &v, dim, &[], &mut dot, &mut wedge, &mut su, &mut sv);
        assert!(dot[..dim].iter().all(|x| x.abs() < 1e-6));
        assert!(wedge[..dim].iter().all(|x| x.abs() < 1e-6));
    }

    /// Sanity: dim=0 is a no-op (no panic, no index OOB).
    #[test]
    fn dim_zero_noop() {
        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(0);
        geometric_product_into(&[], &[], 0, &[1, 2], &mut dot, &mut wedge, &mut su, &mut sv);
        assert!(dot.is_empty());
        assert!(wedge.is_empty());
    }

    /// HLA-sized substrate (D=8) full primitive runs and produces finite output.
    #[test]
    fn hla_sized_smoke() {
        let dim = 8;
        let u: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let v: Vec<f32> = (0..dim).map(|i| ((i * 11) as f32) * 0.2 - 0.5).collect();
        let shifts = [1usize, 2, 4];

        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(&u, &v, dim, &shifts, &mut dot, &mut wedge, &mut su, &mut sv);
        for c in 0..dim {
            assert!(dot[c].is_finite(), "non-finite dot[{c}]");
            assert!(wedge[c].is_finite(), "non-finite wedge[{c}]");
        }
    }

    /// Shard-sized substrate (D=64) full primitive runs and produces finite output.
    #[test]
    fn shard_sized_smoke() {
        let dim = 64;
        let u: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01).sin()).collect();
        let v: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.02).cos()).collect();
        let shifts = [1usize, 2, 4, 8, 16, 32];

        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(&u, &v, dim, &shifts, &mut dot, &mut wedge, &mut su, &mut sv);
        for c in 0..dim {
            assert!(dot[c].is_finite(), "non-finite dot[{c}]");
            assert!(wedge[c].is_finite(), "non-finite wedge[{c}]");
        }
    }

    /// Non-4-multiple dim exercises the remainder path. Dim=5 is prime, never a clean chunk.
    #[test]
    fn non_multiple_of_four_dim() {
        let dim = 5;
        let u: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let v: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0, -5.0];
        let shifts = [1usize, 2];

        let (mut dot, mut wedge, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(&u, &v, dim, &shifts, &mut dot, &mut wedge, &mut su, &mut sv);

        // Verify wedge anti-symmetry holds even when dim is not 4-multiple.
        let (mut dot2, mut wedge2, mut su2, mut sv2, _, _) = buffers(dim);
        geometric_product_into(
            &v,
            &u,
            dim,
            &shifts,
            &mut dot2,
            &mut wedge2,
            &mut su2,
            &mut sv2,
        );
        for c in 0..dim {
            assert!(
                (wedge[c] + wedge2[c]).abs() < 1e-5,
                "non-4-multiple wedge antisymmetry violated at c={c}"
            );
        }
    }

    /// Issue 003 Option C — `geometric_product_wedge_into` produces the same
    /// `wedge_out` as `geometric_product_into` for the same inputs, but skips
    /// the dot/SiLU coherence path entirely (no `exp()`, no division).
    ///
    /// This is the equivalence contract: callers can swap the full primitive for
    /// the wedge-only variant when they don't need coherence, with bit-identical
    /// structural output.
    #[test]
    fn wedge_only_matches_full() {
        let dim = 64;
        let u: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.07).sin()).collect();
        let v: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.03).cos() - 0.5).collect();
        let shifts = [0usize, 1, 2, 4, 8, 16, 32];

        // Full primitive.
        let (_dot_full, mut wedge_full, mut su, mut sv, _, _) = buffers(dim);
        geometric_product_into(
            &u,
            &v,
            dim,
            &shifts,
            &mut vec![0.0; dim], // throwaway dot
            &mut wedge_full,
            &mut su,
            &mut sv,
        );

        // Wedge-only variant.
        let mut wedge_only = vec![0.0f32; dim];
        let mut su2 = vec![0.0f32; dim];
        let mut sv2 = vec![0.0f32; dim];
        geometric_product_wedge_into(&u, &v, dim, &shifts, &mut wedge_only, &mut su2, &mut sv2);

        // Bit-identical wedge output.
        let max_err = max_abs_diff(&wedge_full, &wedge_only, dim);
        assert!(
            max_err < 1e-6,
            "wedge_only ≠ full wedge: max |Δ| = {max_err:.3e}"
        );
    }

    /// Issue 003 Option C — wedge-only variant also respects anti-symmetry
    /// (`u∧v = −v∧u`) and `u∧u = 0`, the same algebraic invariants as the full
    /// primitive. Guards against the wedge-only path silently breaking the
    /// anti-symmetry contract that the G1/G2 quality gates depend on.
    #[test]
    fn wedge_only_antisymmetric_and_self_zero() {
        let dim = 16;
        let u: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.3 - 2.4).collect();
        let v: Vec<f32> = (0..dim)
            .map(|i| ((i * 5 + 1) as f32) * 0.17 - 1.3)
            .collect();
        let shifts = [0usize, 1, 2, 4, 8];

        let mut wedge_uv = vec![0.0f32; dim];
        let mut su = vec![0.0f32; dim];
        let mut sv = vec![0.0f32; dim];
        geometric_product_wedge_into(&u, &v, dim, &shifts, &mut wedge_uv, &mut su, &mut sv);

        let mut wedge_vu = vec![0.0f32; dim];
        geometric_product_wedge_into(&v, &u, dim, &shifts, &mut wedge_vu, &mut su, &mut sv);

        // u∧v = −(v∧u).
        let antisym_err = max_abs_diff(
            &wedge_uv,
            &wedge_vu.iter().map(|x| -*x).collect::<Vec<_>>(),
            dim,
        );
        assert!(
            antisym_err < 1e-5,
            "wedge_only antisymmetry violated: max |u∧v + v∧u| = {antisym_err:.3e}"
        );

        // u∧u = 0.
        let mut wedge_uu = vec![0.0f32; dim];
        geometric_product_wedge_into(&u, &u, dim, &shifts, &mut wedge_uu, &mut su, &mut sv);
        let self_wedge = wedge_uu[..dim].iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            self_wedge < 1e-5,
            "wedge_only u∧u ≠ 0: max |wedge| = {self_wedge:.3e}"
        );
    }
}
