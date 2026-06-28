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
}
