//! T1/M1 — per-position conditional entropy of a categorical (Issue 740,
//! Research 541 M1; paper eq 13: `H(x_i | z_t)`).
//!
//! One max-shift + log-sum-exp pass over the position's logits, reusing the
//! shared kernel [`crate::simd::logsumexp_parts`] factored from
//! `breakeven/fidelity.rs::cross_entropy` (the issue's reuse mandate — one
//! kernel shape, no divergent copy). For a softmax categorical:
//!
//! ```text
//! ln p_i = x_i − max − ln Z
//! H      = −Σᵢ pᵢ ln pᵢ = ln Z − E_p[x − max] = ln_z − mean_shift
//! ```
//!
//! which is why the shared kernel returns the softmax mean shift as its
//! third part. Zero allocation everywhere (G4): batch callers own the output
//! buffer (the scratch-buffer protocol — no per-call `Vec`).

/// Per-position conditional entropy in **nats** for one position's logits.
///
/// Zero-alloc, one pass. An empty slice returns `f32::NEG_INFINITY`
/// (honest degenerate — there is no distribution); non-finite logits
/// propagate as non-finite outputs rather than being clamped away.
///
/// Known answers: a uniform categorical over `V` outcomes gives `ln V`;
/// a one-hot gives `0`.
#[inline]
pub fn conditional_entropy_nats(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return f32::NEG_INFINITY;
    }
    let (_, ln_z, mean_shift) = crate::simd::logsumexp_parts(logits);
    ln_z - mean_shift
}

/// Batch entropies for a row-major flat logits block: `positions` rows of
/// `vocab` logits each, written to `out` (len must equal `positions`).
///
/// Zero allocation — `out` is caller-owned scratch (the G4 protocol).
/// Rows are processed in index order so the result is bit-identical to
/// calling [`conditional_entropy_nats`] row by row.
pub fn conditional_entropies_into(
    logits: &[f32],
    positions: usize,
    vocab: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(logits.len(), positions * vocab, "flat logits shape");
    debug_assert_eq!(out.len(), positions, "out must hold one entropy per position");
    for (p, slot) in out.iter_mut().enumerate() {
        *slot = conditional_entropy_nats(&logits[p * vocab..(p + 1) * vocab]);
    }
}

/// Mean per-position conditional entropy over a row-major flat logits block.
///
/// Streaming accumulation in f32 (same summation order as
/// [`conditional_entropies_into`] followed by a mean, minus the buffer).
/// Zero allocation. `positions == 0` gives `0.0`.
pub fn mean_conditional_entropy(logits: &[f32], positions: usize, vocab: usize) -> f32 {
    if positions == 0 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for p in 0..positions {
        acc += conditional_entropy_nats(&logits[p * vocab..(p + 1) * vocab]);
    }
    acc / positions as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_entropy_f64(logits: &[f32]) -> f64 {
        let max = logits.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
        logits
            .iter()
            .map(|&x| {
                let p = (x as f64 - max).exp() / z;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    #[test]
    fn uniform_gives_ln_v() {
        let v = 8.0f32;
        let logits = vec![0.0f32; 8];
        let h = conditional_entropy_nats(&logits);
        assert!(
            (h - v.ln()).abs() < 1e-5,
            "uniform H must be ln V: got {h}, want {}",
            v.ln()
        );
    }

    #[test]
    fn one_hot_gives_zero() {
        let mut logits = vec![-50.0f32; 16];
        logits[3] = 50.0;
        let h = conditional_entropy_nats(&logits);
        assert!(h.abs() < 1e-4, "one-hot H must be 0: got {h}");
    }

    #[test]
    fn empty_slice_is_negative_infinity() {
        assert_eq!(conditional_entropy_nats(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn matches_direct_sum_p_ln_p() {
        // Fixed "random" logits (no RNG dependency in a unit test).
        let logits: Vec<f32> = (0..37)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761) % 97)
            .map(|h| h as f32 / 7.0 - 5.0)
            .collect();
        let got = conditional_entropy_nats(&logits) as f64;
        let want = ref_entropy_f64(&logits);
        assert!(
            (got - want).abs() < 1e-3,
            "kernel {got} vs direct {want}"
        );
    }

    #[test]
    fn batch_matches_scalar_bit_identically() {
        let positions = 5;
        let vocab = 6;
        let logits: Vec<f32> = (0..positions * vocab)
            .map(|i| ((i * 31) % 11) as f32 - 4.0)
            .collect();
        let mut out = vec![0.0f32; positions];
        conditional_entropies_into(&logits, positions, vocab, &mut out);
        for (p, slot) in out.iter().enumerate() {
            let scalar = conditional_entropy_nats(&logits[p * vocab..(p + 1) * vocab]);
            assert_eq!(*slot, scalar, "row {p} differs from scalar path");
        }
        let mean = mean_conditional_entropy(&logits, positions, vocab);
        let ref_mean: f32 = out.iter().sum::<f32>() / positions as f32;
        assert_eq!(mean, ref_mean, "mean must match the batch buffer mean");
    }

    #[test]
    fn sharpening_monotonically_reduces_entropy() {
        // Scaling logits by α > 1 sharpens the softmax; entropy must fall.
        let base: Vec<f32> = vec![0.2, -1.0, 0.7, 0.1, -0.3, 0.5];
        let h0 = conditional_entropy_nats(&base);
        let sharp: Vec<f32> = base.iter().map(|x| x * 4.0).collect();
        let h1 = conditional_entropy_nats(&sharp);
        assert!(h1 < h0, "sharper categorical must have lower entropy: {h1} !< {h0}");
    }
}
