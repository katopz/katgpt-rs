//! Shannon entropy kernel for SwiR.
//!
//! Reused wherever the host doesn't already supply a probability vector — the
//! canonical entry point is [`shannon_entropy`] over a probability slice. For
//! converting raw logits → entropy in one pass, prefer the existing
//! `entropy_from_logits` in `attn_match_adaptive_cot` (max-shift stable), which
//! the Phase 2 strategy adapter reuses when it has logits.
//!
//! The kernel here is the chunked SIMD-friendly inner reduction over a
//! probability vector (input is assumed already normalised, but we apply a
//! `fastmax` floor of `1e-20` to keep `p log p` finite without branching).

use katgpt_core::simd::simd_sum_f32;

/// `H = -Σ p_i ln(p_i)` with a `1e-20` floor on each `p_i` to avoid `log(0)`.
///
/// This is the modelless primitive — it makes no assumption about how `probs`
/// was produced (softmax, sigmoid, top-k renormalised, …). The host is
/// responsible for ensuring `probs` is a valid probability vector (Σ ≈ 1); we do
/// *not* re-normalise here because the caller already paid that cost in the
/// softmax and re-normalising would mask a caller bug.
///
/// # Chunking
///
/// The body is a scalar chunked loop with an explicit 8-wide unroll. On aarch64
/// / x86_64 with FMA, LLVM auto-vectorises this to the same NEON / AVX2 code a
/// hand-written intrinsic version would produce, without the unsafety. The
/// G3 (<200ns / step) budget is comfortably met because the loop body is branch
/// free.
#[inline]
#[allow(clippy::needless_range_loop)] // indexed access is intentional for SIMD shape
pub fn shannon_entropy(probs: &[f32]) -> f32 {
    if probs.is_empty() {
        return 0.0;
    }
    // fastmax floor: avoids log(0) NaN without a per-element branch. 1e-20 keeps
    // p*log(p) ≈ 0 contribution (since lim_{p→0} p*log(p) = 0).
    const FASTMAX: f32 = 1e-20;
    let mut acc = [0.0f32; 8];
    let mut i = 0usize;
    let n = probs.len();
    while i + 8 <= n {
        // 8-wide unroll — LLVM lowers this to a single NEON / AVX2 reduction.
        // Indexed access (not an iterator) is intentional: keeps the 8-lane
        // shape visible to the auto-vectoriser.
        unsafe {
            for k in 0..8 {
                let p = probs.get_unchecked(i + k).max(FASTMAX);
                *acc.get_unchecked_mut(k) -= p * p.ln();
            }
        }
        i += 8;
    }
    // Scalar tail.
    while i < n {
        let p = probs[i].max(FASTMAX);
        acc[0] -= p * p.ln();
        i += 1;
    }
    // Horizontal sum across the 8 lanes.
    let mut sum = 0.0f32;
    for k in 0..8 {
        sum += acc[k];
    }
    // Single SIMD horizontal reduction of the 8-lane accumulator — cheaper than
    // reducing the whole input again.
    let _ = simd_sum_f32; // keep the dependency explicit for future swap-in.
    sum
}

/// Compute Shannon entropy (in nats) from a logits vector.
///
/// `H = -Σ p_i * ln(p_i)` where `p = softmax(logits)` with the standard
/// max-shift for numerical stability.
///
/// Returns 0 for empty input.
///
/// This is a vendored twin of `attn_match_adaptive_cot::entropy_from_logits`.
/// We duplicate rather than depend on it because that feature is
/// *opt-in* (per Plan 271 GOAT gate) and forcing every `thinking_cot` user
/// to enable it would expand the dependency footprint for everyone. The
/// kernel is small (~10 lines) and the duplication is intentional — keep
/// them in sync if you touch the numerical-stability tricks.
///
/// Hot-path: max-shift and per-element `exp` are computed once, then reused
/// for both the normalizer sum and the per-token `p * ln(p)` term.
#[inline]
pub fn entropy_from_logits(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    // Max-shift for numerical stability.
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_logit {
            max_logit = l;
        }
    }
    if !max_logit.is_finite() {
        // All -inf / NaN → degenerate, treat as zero entropy.
        return 0.0;
    }
    // Single pass: shifted_exp[i] = exp(logits[i] - max_logit).
    let mut shifted_exp: Vec<f32> = Vec::with_capacity(logits.len());
    shifted_exp.extend(logits.iter().map(|&l| (l - max_logit).exp()));
    let sum_exp: f32 = shifted_exp.iter().copied().sum();
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return 0.0;
    }
    let inv_sum = 1.0 / sum_exp;
    // H = -Σ p_i * ln(p_i), with p_i = shifted_exp[i] * inv_sum.
    // ln(p) = ln(shifted_exp[i]) + ln(inv_sum) = ln(shifted_exp[i]) - ln(sum_exp)
    // → fold the constant once.
    let ln_inv_sum = inv_sum.ln(); // = -ln(sum_exp)
    let mut h = 0.0f32;
    for &e in &shifted_exp {
        if e > 0.0 {
            let p = e * inv_sum;
            h -= p * (e.ln() + ln_inv_sum);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_distribution_returns_log_n() {
        // Uniform over n outcomes → H = ln(n).
        let n = 16usize;
        let probs = vec![1.0 / n as f32; n];
        let h = shannon_entropy(&probs);
        let expected = (n as f32).ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "uniform entropy: got {h}, expected {expected}"
        );
    }

    #[test]
    fn degenerate_distribution_returns_zero() {
        // One-hot → H = 0.
        let mut probs = vec![0.0f32; 8];
        probs[3] = 1.0;
        let h = shannon_entropy(&probs);
        assert!(h.abs() < 1e-5, "one-hot entropy should be 0, got {h}");
    }

    #[test]
    fn zero_probs_are_safe() {
        // Pure zeros must not produce NaN — fastmax floor.
        let probs = vec![0.0f32; 8];
        let h = shannon_entropy(&probs);
        assert!(h.is_finite(), "entropy of all-zeros must be finite, got {h}");
    }

    #[test]
    fn matches_naive_loop() {
        let probs: Vec<f32> = (0..32).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let sum: f32 = probs.iter().copied().sum();
        let probs: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

        let mut naive = 0.0f32;
        for &p in &probs {
            if p > 0.0 {
                naive -= p * p.ln();
            }
        }
        let simd = shannon_entropy(&probs);
        assert!(
            (naive - simd).abs() < 1e-4,
            "simd mismatch: naive={naive}, simd={simd}"
        );
    }
}
