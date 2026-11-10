//! CDF-based sampling primitives for speculative decoding.
//!
//! Substrate extraction (Plan 008 Step 6, 2026-06-28): moved verbatim from
//! `katgpt-rs/src/speculative/sampling.rs`. Depends only on
//! [`crate::types::Rng`] and [`crate::simd::simd_scale_inplace`] — both
//! always-on core modules. Any crate can `cargo add katgpt-core` and use
//! these samplers.
//!
//! Composition that needs speculative-decoding context types
//! (`SpeculativeContext::residual_buf` pre-allocated scratch) stays in
//! the root crate as a thin re-export shim — the algorithms themselves
//! are pure substrate.

use crate::types::Rng;

// Plan 367 Phase 3 — QMC K-rollout sampler needs the QmcSource trait.
// Gated on `qmc_sampling` for feature isolation (G6). No circular dep:
// `qmc.rs` imports only `crate::types::Rng`, never anything from `sampling`.
use super::qmc::QmcSource;

// Uses strict `r < cdf` (not `<=`) so zero-probability leading bins are never selected.
// Additionally, `rng.uniform()` is documented to return [0, 1) and can yield exactly
// 0.0 (e.g. for low-entropy seeds the first draw is deterministically 0.0). A draw of
// exactly 0.0 sits on the left boundary of the inverse-CDF map and deterministically
// selects the first token with any nonzero mass, defeating per-seed variation. We
// therefore redraw on `r == 0.0` to obtain a strictly-positive variate. This only
// consumes extra RNG state for sampling calls — weight init (`normal()`) is untouched.
///
/// CDF-based sampling from a probability distribution.
#[inline]
pub fn sample_from_distribution(probs: &[f32], rng: &mut Rng) -> usize {
    let mut r = rng.uniform();
    while r == 0.0 {
        r = rng.uniform();
    }
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// QMC-aware inverse-CDF descend with coordinate carry (Plan 367 Phase 2,
/// Research 367 — QuasiMoTTo, arXiv:2607.01179).
///
/// Given a carried local coordinate `*u ∈ [0,1)` (marginally uniform), find
/// the bin `t` of the CDF partition containing `*u`, then **rescale** `*u`
/// into bin `t`'s local frame:
///
/// ```text
/// *u ← (*u − ℓ_t) / p_t   where ℓ_t = Σ_{j<t} p_j
/// ```
///
/// The rescaled `*u` is marginally `Unif[0,1)` again (arithmetic-coding
/// invariant), so the caller feeds it directly into the next position's
/// descend — producing a token sequence whose joint is the arithmetic-coding
/// image of a single uniform, with each per-position marginal matching the
/// LM exactly.
///
/// This is the per-step descend operator that turns any `QmcSource` into a
/// token-sequence sampler: draw one `u_i` per rollout, descend through the
/// per-position distributions carrying `u_i` forward. Zero allocation, one
/// divide per step.
///
/// # Numerical stability
///
/// - Never touches the raw sequence probability `π(x_<t)` — only the local
///   coordinate. Stable because `p_t > 0` for any selected bin (zero-prob
///   bins are skipped by the strict `*u < cdf` walk, matching
///   [`sample_from_distribution`]).
/// - The carried coordinate is clamped to `[0, 1 − ULP)` to guard against
///   f32 rounding pushing the rescale to exactly `1.0` (which would break
///   the next descend's `u < cdf` walk by sitting past the last bin).
/// - Falls back to `probs.len() − 1` if the CDF walk exhausts without a hit
///   (f32 rounding when `*u` is very close to `1.0` and the CDF sums to
///   slightly less than `1.0`). The carried `*u` is left unmodified in
///   this branch — matches [`sample_from_distribution`]'s fallback.
///
/// # G3 no-regression floor
///
/// When `*u` is a fresh i.i.d. draw with `*u > 0` (no carry), the returned
/// token is bit-identical to [`sample_from_distribution`] with `r = *u`.
/// The rescale is a pure write-back that does not affect token selection.
///
/// # `*u == 0.0`
///
/// A valid (measure-zero) input: selects the first bin with nonzero mass,
/// rescales to `0.0` again (deterministic walk). Unlike
/// [`sample_from_distribution`], there is no redraw — the caller provided
/// the coordinate and owns its distribution.
#[inline]
pub fn sample_from_distribution_qmc(probs: &[f32], u: &mut f32) -> usize {
    // Largest f32 < 1.0 (0x3F7FFFFF = 1.0 − 2^−24 ≈ 0.99999994). Upper bound
    // for the carried coordinate — guards against f32 rounding pushing the
    // rescale to exactly 1.0, which would break the next descend.
    const ONE_MINUS_ULP: f32 = f32::from_bits(0x3F7F_FFFF);

    let r = *u;
    let mut cdf = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        let lower = cdf;
        cdf += p;
        if r < cdf {
            // p > 0 is guaranteed here: for p == 0, cdf == lower, so
            // r < cdf ⟺ r < lower, already false at the prior iteration
            // (or r >= 0 = lower₀ for i = 0). The guard defends the divide
            // against denormal/zero p from malformed input.
            *u = if p > 0.0 {
                ((r - lower) / p).clamp(0.0, ONE_MINUS_ULP)
            } else {
                0.0
            };
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// K-rollout QMC sampler: produces K correlated-but-marginally-exact token
/// sequences from a sequence of per-position distributions (Plan 367 Phase 3,
/// Research 367 — QuasiMoTTo, arXiv:2607.01179).
///
/// For each of the K rollouts, draws one `u_i` from the `source`, then
/// descends through `probs[0], probs[1], ..., probs[T-1]` using
/// [`sample_from_distribution_qmc`] with the carried coordinate. Each
/// rollout is marginally exact (per-position marginal matches the input
/// distribution); the joint structure is controlled by the source
/// (Lattice/Stratified/Sobol) for higher coverage than i.i.d. → 25–47%
/// fewer rollouts for matched pass@k.
///
/// # Zero-allocation contract
///
/// `uniforms_scratch` must be `>= k` (caller-provided QMC draw buffer).
/// `out` must be `>= k` elements, each with capacity `>= probs.len()`
/// (the caller pre-allocates; `clear` + `resize` inside are no-ops given
/// sufficient capacity). The source's `draw` writes into `uniforms_scratch`
/// — also caller-pre-allocated.
///
/// # Panics
///
/// Panics if `uniforms_scratch.len() < k` or `out.len() < k`.
pub fn sample_k_from_distribution_qmc(
    probs: &[&[f32]],
    source: &mut dyn QmcSource,
    k: usize,
    uniforms_scratch: &mut [f32],
    out: &mut [Vec<usize>],
) {
    assert!(
        uniforms_scratch.len() >= k,
        "uniforms_scratch.len() {} < k {}",
        uniforms_scratch.len(),
        k,
    );
    assert!(out.len() >= k, "out.len() {} < k {}", out.len(), k);

    if k == 0 || probs.is_empty() {
        for rollout in &mut out[..k] {
            rollout.clear();
        }
        return;
    }

    // Draw K marginally-uniform points with QMC joint structure.
    source.draw(k, &mut uniforms_scratch[..k]);

    // For each rollout, descend through positions with coordinate carry.
    // Each rollout is an independent descend — embarrassingly parallel by
    // construction (no cross-rollout data dependency). The `u_i` is carried
    // across positions WITHIN a rollout via arithmetic coding rescale.
    let t_len = probs.len();
    for i in 0..k {
        let mut u = uniforms_scratch[i];
        let rollout = &mut out[i];
        rollout.clear();
        rollout.resize(t_len, 0);
        for (t, &probs_t) in probs.iter().enumerate() {
            rollout[t] = sample_from_distribution_qmc(probs_t, &mut u);
        }
    }
}

/// Residual distribution sampling into pre-allocated scratch buffer (zero-alloc).
///
/// `p'(x) = normalize(max(0, p(x) - q(x)))`
///
/// Samples from tokens the target model likes *more* than the draft model.
/// Falls back to `sample_from_distribution(p)` if distributions are identical.
///
/// `scratch` must be `>= p.len()`. Written to but contents not meaningful after return.
#[inline]
pub fn sample_residual_distribution_into(
    p: &[f32],
    q: &[f32],
    scratch: &mut [f32],
    rng: &mut Rng,
) -> usize {
    let len = p.len().min(scratch.len());

    // Chunked residual computation fused with sum accumulation.
    //
    // Fuses the prior two-pass (compute residual → scan-sum) into a single pass:
    // we write the residual into `scratch` AND accumulate the sum in the same
    // loop. Saves a full read pass over `scratch[..len]` before normalization.
    // 4-wide chunked body helps LLVM auto-vectorize the max+add chain.
    //
    // Ported from riir-engine (Plan 008 Phase 2.6, 2026-06-28): the prior
    // scalar form used `simd_scale_inplace` for normalization, but the
    // fused write+sum + 4-wide chunked normalize is strictly fewer passes
    // and auto-vectorizes without a SIMD dispatch — bit-identical output
    // (same `max(0.0)`, same `inv_sum` multiply, same accumulation order
    // within each 4-tile: `(r0+r1)+(r2+r3)`).
    let chunks = len / 4;
    let mut sum = 0.0f32;
    for c in 0..chunks {
        let i = c * 4;
        let r0 = (p[i] - q[i]).max(0.0);
        let r1 = (p[i + 1] - q[i + 1]).max(0.0);
        let r2 = (p[i + 2] - q[i + 2]).max(0.0);
        let r3 = (p[i + 3] - q[i + 3]).max(0.0);
        scratch[i] = r0;
        scratch[i + 1] = r1;
        scratch[i + 2] = r2;
        scratch[i + 3] = r3;
        sum += (r0 + r1) + (r2 + r3);
    }
    for i in (chunks * 4)..len {
        let r = (p[i] - q[i]).max(0.0);
        scratch[i] = r;
        sum += r;
    }

    if sum > 0.0 {
        // Chunked normalization (4-wide) for auto-vectorization.
        // Replaces the prior `simd_scale_inplace` call — same math
        // (multiply by `inv_sum`), but inlined and unrolled.
        let inv_sum = 1.0 / sum;
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            scratch[i] *= inv_sum;
            scratch[i + 1] *= inv_sum;
            scratch[i + 2] *= inv_sum;
            scratch[i + 3] *= inv_sum;
        }
        for val in &mut scratch[chunks * 4..len] {
            *val *= inv_sum;
        }
        sample_from_distribution(&scratch[..len], rng)
    } else {
        // Distributions identical — fallback to target distribution
        sample_from_distribution(p, rng)
    }
}

/// Residual distribution sampling (Equation 3 from Leviathan et al. 2022).
///
/// **Allocating convenience wrapper.** For hot paths (speculative decoding loop),
/// prefer [`sample_residual_distribution_into`] which reuses a pre-allocated
/// scratch buffer from `SpeculativeContext::residual_buf`.
#[deprecated(note = "Use sample_residual_distribution_into with pre-allocated buffer")]
pub fn sample_residual_distribution(p: &[f32], q: &[f32], rng: &mut Rng) -> usize {
    let mut scratch = vec![0.0f32; p.len()];
    sample_residual_distribution_into(p, q, &mut scratch, rng)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::types::Rng;

    #[test]
    fn test_sample_from_distribution() {
        let mut rng = Rng::new(42);
        let probs = vec![0.1, 0.2, 0.5, 0.2];
        for _ in 0..100 {
            let t = sample_from_distribution(&probs, &mut rng);
            assert!(t < 4, "token should be 0-3, got {t}");
        }
    }

    #[test]
    fn test_sample_from_distribution_degenerate() {
        let mut rng = Rng::new(42);
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        for _ in 0..50 {
            let t = sample_from_distribution(&probs, &mut rng);
            assert_eq!(t, 2, "should always sample token 2");
        }
    }

    #[test]
    fn test_residual_distribution_sums_to_one() {
        let mut rng = Rng::new(42);
        let p = vec![0.3, 0.5, 0.1, 0.1];
        let q = vec![0.1, 0.6, 0.2, 0.1];
        // residual = [0.2, 0.0, 0.0, 0.0] → normalized [1.0, 0.0, 0.0, 0.0]
        for _ in 0..50 {
            let token = sample_residual_distribution(&p, &q, &mut rng);
            assert_eq!(token, 0, "residual should only pick token 0");
        }
    }

    #[test]
    fn test_residual_distribution_fallback_on_identical() {
        let mut rng = Rng::new(42);
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let token = sample_residual_distribution(&p, &q, &mut rng);
        assert!(token < 4, "token should be valid, got {token}");
    }

    #[test]
    fn test_residual_distribution_multiple_positive() {
        let mut rng = Rng::new(42);
        let p = vec![0.5, 0.1, 0.3, 0.1];
        let q = vec![0.1, 0.5, 0.1, 0.3];
        // residual = [0.4, 0.0, 0.2, 0.0] → normalized [0.667, 0.0, 0.333, 0.0]
        let mut counts = [0usize; 4];
        for _ in 0..1000 {
            let token = sample_residual_distribution(&p, &q, &mut rng);
            counts[token] += 1;
        }
        assert!(counts[0] > counts[2], "token 0 should be more frequent");
        assert_eq!(counts[1], 0, "token 1 should never be picked");
        assert_eq!(counts[3], 0, "token 3 should never be picked");
    }

    // ── Plan 367 Phase 2: QMC descend (arithmetic-coding with carry) ──────

    /// T2.2 — G3 no-regression floor. For any `u ∈ (0, 1)`, the QMC descend
    /// returns the same token as `sample_from_distribution`'s CDF walk. The
    /// rescale write-back does not affect token selection.
    #[test]
    fn test_qmc_descend_matches_iid_cdf_walk() {
        let probs = [0.1f32, 0.2, 0.5, 0.2];
        let n = 10_000usize;
        for step in 1..=n {
            // u ∈ (0, n/(n+1)) ⊂ (0, 1) — skips 0.0 (sample_from_distribution
            // redraws on it) and ≥1.0 (past the last bin).
            let u = step as f32 / (n + 1) as f32;
            let mut u_qmc = u;
            let tok_qmc = sample_from_distribution_qmc(&probs, &mut u_qmc);

            // Reference: replicate sample_from_distribution's CDF walk verbatim
            // (minus the redraw-on-zero, which we sidestep by u > 0).
            let mut cdf = 0.0f32;
            let mut tok_ref = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                cdf += p;
                if u < cdf {
                    tok_ref = i;
                    break;
                }
            }
            assert_eq!(
                tok_qmc, tok_ref,
                "u={u}: QMC descend disagrees with i.i.d. CDF walk"
            );
        }
    }

    /// T2.3 — marginal exactness. Sweep `u` uniformly over `[0,1)` (fresh draw
    /// per call, no carry). The empirical token frequency must match `probs` —
    /// arithmetic coding maps a uniform to the target marginal.
    #[test]
    fn test_qmc_descend_marginal_exactness() {
        let probs = [0.1f32, 0.25, 0.15, 0.5];
        let n = 10_000usize;
        let mut counts = [0u32; 4];
        for step in 0..n {
            let mut u = step as f32 / n as f32; // [0, 1)
            let tok = sample_from_distribution_qmc(&probs, &mut u);
            assert!(tok < 4, "token {tok} out of range for probs of len 4");
            counts[tok] += 1;
        }
        for (i, &p) in probs.iter().enumerate() {
            let emp = counts[i] as f32 / n as f32;
            // Grid discretization error is O(1/n) = 1e-4; allow slack.
            let delta = (emp - p).abs();
            assert!(
                delta < 5e-3,
                "token {i}: emp_freq={emp:.4} vs p={p:.4} (Δ={delta:.4})"
            );
        }
    }

    /// T2.4 — coordinate-carry associativity. Two sequential descends with
    /// carry (P then Q) must produce the same `(t1, t2)` pair as a single
    /// descend on the product partition `P ⊗ Q`. This is the arithmetic-coding
    /// associativity that makes the carry algebra correct over a sequence.
    ///
    /// We test cell-interior points (at 1/4, 1/2, 3/4 of each cell) rather
    /// than a uniform grid: the associativity identity holds over ℝ, but f32
    /// rounding can flip the cell assignment at exact partition boundaries
    /// (the descend's strict `u < cdf` vs the reference's `u >= cell_lo` make
    /// different choices at the boundary). Interior points avoid this.
    #[test]
    fn test_qmc_descend_carry_associativity() {
        let p = [0.3f32, 0.7]; // 2 bins
        let q = [0.2f32, 0.5, 0.3]; // 3 bins

        // For each product cell (i, j), probe three interior fractions and
        // verify the two-step descend maps to (i, j). The fractions 1/4,
        // 1/2, 3/4 are safely away from cell edges (where f32 rounding could
        // flip the assignment).
        for &frac in &[0.25f32, 0.5, 0.75] {
            let mut p_lo = 0.0f32;
            for (i, &pi) in p.iter().enumerate() {
                let mut q_acc = 0.0f32;
                for (j, &qj) in q.iter().enumerate() {
                    // Interior point of cell (i, j):
                    //   p_lo + pi * (q_lo + frac * qj)
                    let u0 = p_lo + pi * (q_acc + qj * frac);

                    // Two-step descend with carry.
                    let mut u1 = u0;
                    let t1 = sample_from_distribution_qmc(&p, &mut u1);
                    let t2 = sample_from_distribution_qmc(&q, &mut u1);

                    assert_eq!(
                        (t1, t2),
                        (i, j),
                        "u0={u0:.6} (frac={frac}, cell ({i},{j})): \
                         two-step gave ({t1},{t2})"
                    );
                    assert!(
                        (0.0..1.0).contains(&u1),
                        "carried u1 out of [0,1) after two-step: {u1}"
                    );

                    q_acc += qj;
                }
                p_lo += pi;
            }
        }
    }

    /// The carried coordinate after any descend must stay in `[0, 1)` — the
    /// arithmetic-coding invariant that keeps the next descend valid.
    #[test]
    fn test_qmc_descend_carried_coordinate_in_unit_interval() {
        let probs = [0.1f32, 0.2, 0.3, 0.4];
        let n = 10_000usize;
        for step in 0..n {
            let mut u = step as f32 / n as f32;
            sample_from_distribution_qmc(&probs, &mut u);
            assert!(u >= 0.0, "carried u < 0: {u}");
            assert!(u < 1.0, "carried u >= 1: {u}");
        }
    }

    /// `u == 0.0` selects the first bin with nonzero mass and rescales to
    /// `0.0` again (deterministic walk). Measure-zero edge case.
    #[test]
    fn test_qmc_descend_zero_u_selects_first_nonzero_bin() {
        let probs = [0.0f32, 0.0, 1.0, 0.0];
        let mut u = 0.0;
        let tok = sample_from_distribution_qmc(&probs, &mut u);
        assert_eq!(tok, 2, "u=0 must skip zero-prob bins and select bin 2");
        assert_eq!(u, 0.0, "rescale of (0 − 0) / 1 = 0");
    }

    /// Empty distribution: consistent with `sample_from_distribution`
    /// (returns 0 via `saturating_sub`). Carried `u` is unmodified.
    #[test]
    fn test_qmc_descend_empty_probs_returns_zero() {
        let probs: [f32; 0] = [];
        let mut u = 0.42;
        let tok = sample_from_distribution_qmc(&probs, &mut u);
        assert_eq!(tok, 0, "empty probs → 0 (saturating_sub)");
        assert_eq!(u, 0.42, "empty probs should not modify u");
    }

    // ── Plan 367 Phase 3: K-rollout QMC sampler ───────────────────────────
    use crate::speculative::qmc::LatticeQmc;

    /// T3.1 basic — K rollouts, each of length T, tokens in valid range.
    #[test]
    fn test_sample_k_basic() {
        let p0 = [0.1f32, 0.2, 0.3, 0.4];
        let p1 = [0.25f32, 0.25, 0.25, 0.25];
        let p2 = [0.5f32, 0.5];
        let probs: &[&[f32]] = &[&p0, &p1, &p2];

        let k = 8;
        let mut source = LatticeQmc::new(42);
        let mut uniforms = vec![0.0f32; k];
        let mut out: Vec<Vec<usize>> = (0..k).map(|_| Vec::with_capacity(3)).collect();

        sample_k_from_distribution_qmc(probs, &mut source, k, &mut uniforms, &mut out);

        assert_eq!(out.len(), k);
        for rollout in &out {
            assert_eq!(rollout.len(), 3, "each rollout has T=3 tokens");
            assert!(rollout[0] < 4, "pos 0 token in [0,4)");
            assert!(rollout[1] < 4, "pos 1 token in [0,4)");
            assert!(rollout[2] < 2, "pos 2 token in [0,2)");
        }
    }

    /// T3.1 determinism — same source state → same output.
    #[test]
    fn test_sample_k_deterministic() {
        let p0 = [0.3f32, 0.7];
        let p1 = [0.4f32, 0.6];
        let probs: &[&[f32]] = &[&p0, &p1];
        let k = 16;

        // First run
        let mut src1 = LatticeQmc::new(123);
        let mut u1 = vec![0.0f32; k];
        let mut out1: Vec<Vec<usize>> = (0..k).map(|_| Vec::with_capacity(2)).collect();
        sample_k_from_distribution_qmc(probs, &mut src1, k, &mut u1, &mut out1);

        // Second run — fresh source with same seed
        let mut src2 = LatticeQmc::new(123);
        let mut u2 = vec![0.0f32; k];
        let mut out2: Vec<Vec<usize>> = (0..k).map(|_| Vec::with_capacity(2)).collect();
        sample_k_from_distribution_qmc(probs, &mut src2, k, &mut u2, &mut out2);

        assert_eq!(out1, out2, "same seed → same output");
    }

    /// T3.1 marginal exactness — per-position token frequencies match probs.
    /// The arithmetic-coding descend preserves marginal exactness at every
    /// position (linearity of expectation), so K=10000 lattice points should
    /// reproduce the target distribution at each position.
    #[test]
    fn test_sample_k_marginal_exactness() {
        let p0 = [0.1f32, 0.2, 0.3, 0.4];
        let p1 = [0.25f32, 0.75];
        let probs: &[&[f32]] = &[&p0, &p1];

        let k = 10_000;
        let mut source = LatticeQmc::new(7);
        let mut uniforms = vec![0.0f32; k];
        let mut out: Vec<Vec<usize>> = (0..k).map(|_| Vec::with_capacity(2)).collect();
        sample_k_from_distribution_qmc(probs, &mut source, k, &mut uniforms, &mut out);

        // Position 0 marginal
        let mut c0 = [0u32; 4];
        for r in &out {
            c0[r[0]] += 1;
        }
        for (j, &p) in p0.iter().enumerate() {
            let emp = c0[j] as f32 / k as f32;
            assert!(
                (emp - p).abs() < 0.02,
                "pos 0 tok {j}: emp={emp:.4} vs p={p:.4}"
            );
        }

        // Position 1 marginal (carried coordinate — arithmetic coding guarantee)
        let mut c1 = [0u32; 2];
        for r in &out {
            c1[r[1]] += 1;
        }
        for (j, &p) in p1.iter().enumerate() {
            let emp = c1[j] as f32 / k as f32;
            assert!(
                (emp - p).abs() < 0.02,
                "pos 1 tok {j}: emp={emp:.4} vs p={p:.4}"
            );
        }
    }

    /// T3.1 K=0 — no-op.
    #[test]
    fn test_sample_k_zero_is_noop() {
        let probs: &[&[f32]] = &[&[0.5f32, 0.5]];
        let mut source = LatticeQmc::new(0);
        let mut uniforms: [f32; 0] = [];
        let mut out: Vec<Vec<usize>> = Vec::new();
        sample_k_from_distribution_qmc(probs, &mut source, 0, &mut uniforms, &mut out);
        assert!(out.is_empty());
    }

    /// T3.1 K=1 — single rollout.
    #[test]
    fn test_sample_k_single_rollout() {
        let probs: &[&[f32]] = &[&[1.0f32]];
        let k = 1;
        let mut source = LatticeQmc::new(99);
        let mut uniforms = [0.0f32];
        let mut out: Vec<Vec<usize>> = vec![Vec::with_capacity(1)];
        sample_k_from_distribution_qmc(probs, &mut source, k, &mut uniforms, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0], 0, "single-token vocab → token 0");
    }
}
