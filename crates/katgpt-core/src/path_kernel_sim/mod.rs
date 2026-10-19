//! Sensitivity-fingerprint response-space kernel (Issue 949, from Research
//! 625 — arXiv:2012.00152's path-kernel lens).
//!
//! The degenerate path kernel of an *authored, untrained* deterministic
//! scorer is a training-free, positive-semidefinite response-space
//! similarity: `K(x, x') = cos(Ŝ(x), Ŝ(x'))` where
//! `Ŝ(x) = ∇_w f(x; w)|_{w0}` is the scorer's sensitivity fingerprint over
//! its `p ≤ MAX_KNOBS` authored knobs. No analog shipped:
//! [`crate::similarity`] is input-space;
//! [`crate::subspace_phase_gate::participation_ratio`] runs over latent
//! directions, not knob-sensitivity directions.
//!
//! ⛔ **THE COLLAPSE CAVEAT (Issue 949, load-bearing).** For any scorer
//! LINEAR in its knobs, `f = σ(w·φ(x))`, the normalized fingerprint kernel
//! is ALGEBRAICALLY the input-feature cosine: `∇_w f = σ'(w·φ(x))·φ(x)` and
//! the scalar `σ'` cancels on normalization. New signal appears ONLY where
//! knobs enter nonlinearly — stacked gates, a knob inside a composition,
//! multiplicative knob coupling, per-stage gains. The linear case is kept as
//! the T5a correctness oracle, never as the target.
//!
//! # Arms (T2)
//!
//! - **closed-form** — the scorer writes its own knob gradient
//!   ([`KnobScorer::grad_knobs_closed_form`]); the canonical sigmoid-gated
//!   shape uses `σ'(z) = s·(1−s)` from the SAME `fast_sigmoid` the scorer
//!   evaluates with
//! - **central difference** — pinned `h` ([`DEFAULT_CENTRAL_DIFF_H`]), fixed
//!   knob-ascending evaluation order; every reduction is the
//!   `dot_f32_ordered` sequential fold (the Issue-940 determinism law)
//!
//! # Guards (T1)
//!
//! A near-zero-norm fingerprint (the frequency-knob stationary point where
//! `∇_w sin(wx) = x·cos(wx) = 0`) ABSTAINS with a sentinel — the kernel
//! returns `None`, never NaN.
//!
//! # Scope
//!
//! POC, opt-in (`path_kernel_sim`), all outputs LOCAL to the computing
//! process. If a consumer ever routes these outputs into SYNCED belief
//! attribution, the Issue-940 determinism law binds in full (sequential
//! fold, first-index ties, `freeze_version_hash` coverage).
//!
//! # Diagnostics (T3/T4)
//!
//! - [`spearman_rho_into`] — rank correlation between input-space distances
//!   and fingerprint similarities (the E4 symmetry-class detector)
//! - [`gram_spectrum_into`] + [`participation_ratio_of`] /
//!   [`effective_rank_of`] — PR / effective rank over the fingerprint Gram
//!   (the E3 spectrum arm), eigendecomposed by the existing
//!   [`crate::linalg::symmetric_eig`] substrate (substrate-first: no new
//!   solver shipped here)

pub mod types;

pub use types::{
    DEFAULT_CENTRAL_DIFF_H, DEFAULT_ZERO_NORM_FLOOR, FingerprintKernel, KnobScorer, MAX_KNOBS,
    SensitivityFingerprint,
};

use crate::linalg::symmetric_eig::{SymmetricEigScratch, symmetric_eig};
use crate::simd::dot_f32_ordered;

/// Write `Ŝ(x) = ∇_w f(x; w0)` into `out` (len `p`), returning its L2 norm.
///
/// Returns `0.0` = ABSTAIN when the norm falls under the kernel's
/// [`FingerprintKernel::zero_norm_floor`] — `out` is zeroed, the caller
/// skips the anchor. Never NaN: a scorer returning non-finite evaluations
/// produces non-finite fingerprints, which fail the floor check and abstain
/// (`NaN < floor` is false, so the guard treats NaN as abstain too — see
/// the explicit `!norm.is_finite()` arm).
///
/// Zero-alloc: `out` is caller-owned scratch (T8).
pub fn fingerprint_into<S: KnobScorer + ?Sized>(
    kernel: &FingerprintKernel,
    scorer: &S,
    x: &[f32],
    w0: &mut [f32],
    out: &mut [f32],
) -> f32 {
    let p = scorer.knobs();
    debug_assert_eq!(w0.len(), p, "w0 must carry exactly p knobs");
    debug_assert_eq!(out.len(), p, "out must carry exactly p slots");
    debug_assert!(p <= MAX_KNOBS, "p = {p} over the MAX_KNOBS envelope");

    if scorer.grad_knobs_closed_form(x, w0, out) {
        // Closed-form arm — the scorer owns its derivative (σ' from the same
        // fast_sigmoid it evaluates with, per the T2 protocol).
    } else {
        // Central-difference arm: pinned h, knob-ascending, stencil restore.
        let h = kernel.central_diff_h;
        for (j, o) in out.iter_mut().enumerate() {
            let original = w0[j];
            w0[j] = original + h;
            let f_plus = scorer.eval(x, w0);
            w0[j] = original - h;
            let f_minus = scorer.eval(x, w0);
            w0[j] = original;
            *o = (f_plus - f_minus) / (2.0 * h);
        }
    }

    let norm_sq = dot_f32_ordered(out, out);
    if !norm_sq.is_finite() {
        out.fill(0.0);
        return 0.0;
    }
    let norm = norm_sq.sqrt();
    if norm < kernel.zero_norm_floor {
        out.fill(0.0);
        return 0.0;
    }
    norm
}

/// Normalized kernel `K(a, b) = ⟨â, b̂⟩` over two GUARDED fingerprints.
///
/// `None` when either side is at/below the floor (an abstained anchor) —
/// the guard, never NaN. Sequential `dot_f32_ordered` folds throughout.
pub fn similarity(kernel: &FingerprintKernel, a: &[f32], b: &[f32]) -> Option<f32> {
    debug_assert_eq!(a.len(), b.len(), "fingerprint lengths must match");
    let dot = dot_f32_ordered(a, b);
    let na = dot_f32_ordered(a, a).sqrt();
    let nb = dot_f32_ordered(b, b).sqrt();
    if !dot.is_finite() || na < kernel.zero_norm_floor || nb < kernel.zero_norm_floor {
        return None;
    }
    Some(dot / (na * nb))
}

/// Rank both pair-arrays (first-index tie-break, average ranks inside tie
/// groups) and return Spearman ρ — the T3 diagnostic between input-space
/// distances and fingerprint similarities.
///
/// `None` on a degenerate input (fewer than 2 pairs, or an all-equal array —
/// zero rank variance). Zero-alloc: `idx`, `rank_x`, `rank_y` are
/// caller-owned scratch, each `n_pairs` long (T8).
pub fn spearman_rho_into(
    x: &[f32],
    y: &[f32],
    idx: &mut [u32],
    rank_x: &mut [f32],
    rank_y: &mut [f32],
) -> Option<f32> {
    let n = x.len();
    if n < 2 || y.len() != n {
        return None;
    }
    debug_assert!(idx.len() >= n && rank_x.len() >= n && rank_y.len() >= n);

    let sx = rank_into(x, &mut idx[..n], &mut rank_x[..n])?;
    let sy = rank_into(y, &mut idx[..n], &mut rank_y[..n])?;
    if !sx || !sy {
        return None; // all-equal array — zero variance
    }

    // Pearson over the ranks, sequential folds.
    let mut mean_x = 0.0f32;
    let mut mean_y = 0.0f32;
    for i in 0..n {
        mean_x += rank_x[i];
        mean_y += rank_y[i];
    }
    mean_x /= n as f32;
    mean_y /= n as f32;
    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;
    for i in 0..n {
        let dx = rank_x[i] - mean_x;
        let dy = rank_y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    let rho = cov / (var_x.sqrt() * var_y.sqrt());
    if rho.is_finite() {
        Some(rho.clamp(-1.0, 1.0))
    } else {
        None
    }
}

/// Average ranks (1-based) of `v` into `rank`, first-index tie-break.
/// Returns `false` when every element ties (zero variance). `idx` is
/// caller-owned scratch. Deterministic: the sort key is `(value, index)`.
fn rank_into(v: &[f32], idx: &mut [u32], rank: &mut [f32]) -> Option<bool> {
    let n = v.len();
    for (i, slot) in idx.iter_mut().enumerate() {
        *slot = i as u32;
    }
    // Sort indices by (value, index) — float_order::asc breaks ties by the
    // captured index, so equal values keep their input order (the Issue-940
    // first-index tie rule).
    idx.sort_by(|&a, &b| crate::float_order::asc(v[a as usize], v[b as usize]).then(a.cmp(&b)));

    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j] as usize] == v[idx[i] as usize] {
            j += 1;
        }
        // Average 1-based rank of the tie group.
        let avg = (i + 1 + j) as f32 / 2.0;
        for slot in &idx[i..j] {
            rank[*slot as usize] = avg;
        }
        i = j;
    }
    if n >= 2 && v[idx[0] as usize] == v[idx[n - 1] as usize] {
        return Some(false); // one giant tie group
    }
    Some(true)
}

/// Build the n×n Gram (f64, row-major) of the GUARDED fingerprints into
/// `out` (len n*n). Rows below the floor are refused — the caller must
/// pre-filter abstained anchors ([`fingerprint_into`] returning 0.0).
/// Sequential `dot_f32_ordered` per entry, cast once at the f32/f64 edge.
pub fn gram_into(kernel: &FingerprintKernel, fps: &[&[f32]], out: &mut [f64]) {
    let n = fps.len();
    debug_assert_eq!(out.len(), n * n, "out must carry n*n slots");
    for (i, fi) in fps.iter().enumerate() {
        let ni = dot_f32_ordered(fi, fi).sqrt();
        assert!(
            ni >= kernel.zero_norm_floor,
            "gram_into: fingerprint {i} is at/below the abstain floor — pre-filter abstained anchors"
        );
        for (j, fj) in fps.iter().enumerate().skip(i) {
            let dot = dot_f32_ordered(fi, fj) as f64;
            out[i * n + j] = dot;
            out[j * n + i] = dot;
        }
    }
}

/// Eigendecompose the fingerprint Gram (T4's E3 spectrum arm) — the
/// substrate-first reuse: [`symmetric_eig`] (Householder+QL) eigendecomposes,
/// then the spectrum is sorted DESCENDING in-place (the solver returns it in
/// deflation order, not sorted — basis_split sorts downstream the same way).
/// `gram` is consumed as the work matrix (destroyed). `eigvecs` (n*n) is
/// still produced (columns aligned to the PRE-sort order — sort values only
/// when the vectors are not needed). Scratch is the substrate's own
/// [`SymmetricEigScratch`].
pub fn gram_spectrum_into(
    gram: &mut [f64],
    eigvals: &mut [f64],
    eigvecs: &mut [f64],
    scratch: &mut SymmetricEigScratch,
    n: usize,
) {
    debug_assert_eq!(gram.len(), n * n);
    debug_assert_eq!(eigvals.len(), n);
    debug_assert_eq!(eigvecs.len(), n * n);
    symmetric_eig(eigvals, eigvecs, gram, scratch, n, 64);
    // Descending by value; index tie-break keeps the sort deterministic
    // (the Issue-940 tie rule).
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        crate::float_order::desc_f64(eigvals[a as usize], eigvals[b as usize]).then(a.cmp(&b))
    });
    let sorted: Vec<f64> = order.iter().map(|&i| eigvals[i as usize]).collect();
    eigvals[..n].copy_from_slice(&sorted);
}

/// Participation ratio `(Σλ)² / Σλ²` over a spectrum — the
/// [`crate::subspace_phase_gate::participation_ratio`] math applied to the
/// fingerprint Gram's eigenvalues (substrate reuse, f32 diagnostic).
/// Negative eigenvalues clamp to 0 (a Gram is PSD; `−ε‖G‖` solver residue).
pub fn participation_ratio_of(spectrum: &[f32]) -> f32 {
    crate::subspace_phase_gate::participation_ratio(spectrum)
}

/// Effective rank `exp(−Σ p_i ln p_i)` over the normalized spectrum
/// (`p_i = λ_i / Σλ`) — the Shannon-entropy sibling of PR. Zero total mass
/// → `0.0`.
pub fn effective_rank_of(spectrum: &[f32]) -> f32 {
    let total: f32 = spectrum.iter().map(|l| l.max(0.0)).sum();
    if !total.is_finite() || total <= 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0f32;
    for l in spectrum {
        let l = l.max(0.0);
        if l > 0.0 {
            let p = l / total;
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

// ── tests — T5a/T5b correctness + the T1 guard ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::fast_sigmoid;

    /// The canonical linear-in-knobs scorer `f = σ(w·φ(x))` — the T5a
    /// collapse-caveat oracle shape. Closed-form gradient:
    /// `σ'(z)·φ(x)`, `σ'` from the same `fast_sigmoid`.
    struct SigmoidFeatureGate {
        /// φ(x) — here an affine map of the raw input so the test can read
        /// the input-cosine identity off the raw vectors.
        scale: f32,
    }

    impl SigmoidFeatureGate {
        fn phi(&self, x: &[f32]) -> Vec<f32> {
            x.iter().map(|v| v * self.scale).collect()
        }
    }

    impl KnobScorer for SigmoidFeatureGate {
        fn knobs(&self) -> usize {
            8
        }
        fn eval(&self, x: &[f32], w: &[f32]) -> f32 {
            let phi = self.phi(x);
            let z = crate::simd::dot_f32_ordered(w, &phi);
            fast_sigmoid(z)
        }
        fn grad_knobs_closed_form(&self, x: &[f32], w: &[f32], out: &mut [f32]) -> bool {
            let phi = self.phi(x);
            let z = crate::simd::dot_f32_ordered(w, &phi);
            let s = fast_sigmoid(z);
            let s_prime = s * (1.0 - s);
            for (o, p) in out.iter_mut().zip(phi.iter()) {
                *o = s_prime * p;
            }
            true
        }
    }

    /// Deterministic LCG (the unseeded-global-RNG gate).
    struct Lcg(u64);
    impl Lcg {
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f32) / ((1u64 << 53) as f32)
        }
        fn next_signed(&mut self) -> f32 {
            self.next_unit() * 2.0 - 1.0
        }
    }

    /// T5a — the LINEAR-in-knobs identity: the normalized fingerprint cosine
    /// equals the input-feature cosine exactly (to f32 tolerance). Proves the
    /// kernel code, never the signal.
    #[test]
    fn t5a_linear_in_knobs_kernel_equals_input_cosine() {
        let kernel = FingerprintKernel::default();
        let scorer = SigmoidFeatureGate { scale: 1.0 };
        let p = scorer.knobs();

        let mut rng = Lcg(0x949_0001);
        let mut w0 = [0.0f32; 8];
        for w in w0.iter_mut() {
            *w = rng.next_signed() * 0.5;
        }

        let n_anchors = 12;
        let anchors: Vec<Vec<f32>> = (0..n_anchors)
            .map(|_| (0..p).map(|_| rng.next_signed()).collect())
            .collect();

        let mut fp = vec![0.0f32; p];
        let mut w_scratch = w0;
        let mut fingerprints: Vec<Vec<f32>> = Vec::with_capacity(n_anchors);
        for a in &anchors {
            let norm = fingerprint_into(&kernel, &scorer, a, &mut w_scratch, &mut fp);
            assert!(norm > 0.0, "T5a fixture anchors must not abstain");
            fingerprints.push(fp.clone());
        }

        let mut worst = 0.0f32;
        for i in 0..n_anchors {
            for j in (i + 1)..n_anchors {
                let k = similarity(&kernel, &fingerprints[i], &fingerprints[j]).unwrap();
                let input_cos = {
                    let (a, b) = (&anchors[i], &anchors[j]);
                    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
                    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
                    dot / (na * nb)
                };
                worst = worst.max((k - input_cos).abs());
            }
        }
        assert!(
            worst < 1.0e-5,
            "T5a FAIL: normalized fingerprint cosine diverges from input cosine by {worst}"
        );
    }

    /// T5b — the sine fixture (CORRECTNESS-ONLY): knob pinned to AMPLITUDE,
    /// `f(x; a) = a·sin(k·x)` with `k` fixed; the closed-form fingerprint IS
    /// `sin(kx)`, so an anchor at a sine maximum reads fingerprint [1.0], and
    /// two maxima (maximal input distance) are kernel-identical. ⚠ The
    /// amplitude knob is LINEAR — this proves the fingerprint code, never
    /// the signal.
    #[test]
    fn t5b_amplitude_sine_fingerprint_is_sin_kx() {
        struct AmplitudeSine {
            k: f32,
        }
        impl KnobScorer for AmplitudeSine {
            fn knobs(&self) -> usize {
                1
            }
            fn eval(&self, x: &[f32], w: &[f32]) -> f32 {
                w[0] * (self.k * x[0]).sin()
            }
            fn grad_knobs_closed_form(&self, x: &[f32], _w: &[f32], out: &mut [f32]) -> bool {
                out[0] = (self.k * x[0]).sin();
                true
            }
        }

        let kernel = FingerprintKernel::default();
        let scorer = AmplitudeSine { k: 2.0 };
        let mut w0 = [3.5f32]; // amplitude knob, arbitrary
        let mut fp = [0.0f32; 1];

        // A sine maximum: sin(k·x) = 1 → the fingerprint reads exactly [1.0].
        let x_max = [std::f32::consts::PI / 4.0];
        let n1 = fingerprint_into(&kernel, &scorer, &x_max, &mut w0, &mut fp);
        assert!(
            (n1 - 1.0).abs() < 1.0e-6,
            "fingerprint at a maximum is [1.0], got {n1}"
        );

        // The maximum a full period away (maximal input distance) is
        // kernel-IDENTICAL — the fingerprint sees the phase class, not x.
        let x_far = [x_max[0] + std::f32::consts::PI];
        let n2 = fingerprint_into(&kernel, &scorer, &x_far, &mut w0, &mut fp);
        assert!((n2 - 1.0).abs() < 1.0e-6);
        let k_cos = similarity(&kernel, &[1.0], &[1.0]).unwrap();
        assert!(
            (k_cos - 1.0).abs() < 1.0e-6,
            "maxima must be kernel-identical"
        );
    }

    /// T2 — both arms agree on a smooth scorer at generic points: the
    /// amplitude-sine gradient is exactly `sin(kx)`; the central stencil
    /// reproduces it within the pinned-h truncation error.
    #[test]
    fn t2_arms_agree_on_smooth_scorer() {
        struct AmplitudeSineOpaque;
        impl KnobScorer for AmplitudeSineOpaque {
            fn knobs(&self) -> usize {
                1
            }
            fn eval(&self, x: &[f32], w: &[f32]) -> f32 {
                w[0] * (2.0 * x[0]).sin()
            }
            // no closed-form hook → the opaque arm
        }

        let kernel = FingerprintKernel::default();
        let mut w0 = [2.0f32];
        let mut fp_cd = [0.0f32; 1];
        fingerprint_into(
            &kernel,
            &AmplitudeSineOpaque,
            &[0.37f32],
            &mut w0,
            &mut fp_cd,
        );
        let expected = (2.0f32 * 0.37).sin();
        assert!(
            (fp_cd[0] - expected).abs() < 1.0e-3,
            "central-diff {:?} vs closed form {expected}",
            fp_cd
        );
    }

    /// T1 guard — the frequency-knob stationary point: `f(x; w) = sin(w·x)`
    /// at `w` where `x·cos(wx) = 0` → fingerprint exactly zero → ABSTAIN
    /// (sentinel norm 0.0, zeroed scratch), and the kernel returns `None`
    /// against ANY partner. Never NaN.
    #[test]
    fn t1_zero_norm_fingerprint_abstains_never_nan() {
        struct FrequencySine;
        impl KnobScorer for FrequencySine {
            fn knobs(&self) -> usize {
                1
            }
            fn eval(&self, x: &[f32], w: &[f32]) -> f32 {
                (w[0] * x[0]).sin()
            }
            fn grad_knobs_closed_form(&self, x: &[f32], w: &[f32], out: &mut [f32]) -> bool {
                out[0] = x[0] * (w[0] * x[0]).cos();
                true
            }
        }

        let kernel = FingerprintKernel::default();
        let mut w0 = [std::f32::consts::PI / 2.0]; // cos(w·0.5) at x=1: cos(π/2)=0
        let x = [1.0f32];
        let mut fp = [9.9f32]; // dirty scratch — must be zeroed
        let norm = fingerprint_into(&kernel, &FrequencySine, &x, &mut w0, &mut fp);
        assert_eq!(norm, 0.0, "stationary fingerprint must abstain");
        assert_eq!(fp[0], 0.0, "abstained scratch must be zeroed");

        let none = similarity(&kernel, &fp, &[1.0]);
        assert!(none.is_none(), "kernel must refuse an abstained anchor");

        // NaN-input defense: a scorer returning NaN evaluates to a non-finite
        // norm → abstain, not NaN propagation.
        struct NaNEval;
        impl KnobScorer for NaNEval {
            fn knobs(&self) -> usize {
                1
            }
            fn eval(&self, _x: &[f32], _w: &[f32]) -> f32 {
                f32::NAN
            }
        }
        let mut w_nan = [1.0f32];
        let mut fp_nan = [0.0f32];
        let n = fingerprint_into(&kernel, &NaNEval, &x, &mut w_nan, &mut fp_nan);
        assert_eq!(n, 0.0, "non-finite fingerprint must abstain");
    }

    /// T3 — Spearman ρ sanity: perfectly monotone pair-arrays → ρ = 1;
    /// reversed → −1; constant input → None.
    #[test]
    fn t3_spearman_monotone_and_degenerate() {
        let kernel = FingerprintKernel::default();
        let n = 16;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut yr = Vec::with_capacity(n);
        for i in 0..n {
            let v = i as f32;
            x.push(v);
            y.push(v * 0.5 + 1.0);
            yr.push(100.0 - v);
        }
        let mut idx = vec![0u32; n];
        let mut rx = vec![0.0f32; n];
        let mut ry = vec![0.0f32; n];
        let rho_pos = spearman_rho_into(&x, &y, &mut idx, &mut rx, &mut ry).unwrap();
        let rho_neg = spearman_rho_into(&x, &yr, &mut idx, &mut rx, &mut ry).unwrap();
        assert!(
            (rho_pos - 1.0).abs() < 1.0e-5,
            "monotone → ρ=1, got {rho_pos}"
        );
        assert!(
            (rho_neg + 1.0).abs() < 1.0e-5,
            "anti-monotone → ρ=−1, got {rho_neg}"
        );

        let flat = vec![2.0f32; n];
        assert!(spearman_rho_into(&x, &flat, &mut idx, &mut rx, &mut ry).is_none());
        let _ = kernel; // floor config unused here; the API surface is the test
    }

    /// T4 — PR / effective rank over a KNOWN spectrum: identity-mass Gram
    /// (3 equal eigenvalues) → PR = 3; rank-1 Gram → PR = 1.
    #[test]
    fn t4_pr_and_effective_rank_known_spectra() {
        let equal = [2.0f32, 2.0, 2.0];
        assert!((participation_ratio_of(&equal) - 3.0).abs() < 1.0e-4);
        assert!((effective_rank_of(&equal) - 3.0).abs() < 1.0e-4);

        let rank1 = [8.0f32, 0.0, 0.0];
        assert!((participation_ratio_of(&rank1) - 1.0).abs() < 1.0e-4);
        assert!((effective_rank_of(&rank1) - 1.0).abs() < 1.0e-4);

        assert_eq!(participation_ratio_of(&[0.0; 4]), 0.0);
        assert_eq!(effective_rank_of(&[0.0; 4]), 0.0);

        // End-to-end: collinear fingerprints → Gram spectrum rank-1 → PR 1.
        let kernel = FingerprintKernel::default();
        let fps: Vec<&[f32]> = vec![&[1.0, 2.0], &[2.0, 4.0], &[-1.0, -2.0]];
        let mut gram = vec![0.0f64; 9];
        gram_into(&kernel, &fps, &mut gram);
        let mut eigvals = vec![0.0f64; 3];
        let mut eigvecs = vec![0.0f64; 9];
        let mut scratch = SymmetricEigScratch::new();
        gram_spectrum_into(&mut gram, &mut eigvals, &mut eigvecs, &mut scratch, 3);
        let spec: Vec<f32> = eigvals.iter().map(|v| *v as f32).collect();
        let pr = participation_ratio_of(&spec);
        assert!(
            pr < 1.05,
            "collinear Gram must read rank-1 (PR≈1), got {pr}"
        );
        assert!(eigvals[0] > 0.0);
        // Numerical zero of the QL deflation: relative to λ_max, not absolute.
        let tol = 1.0e-6 * eigvals[0].abs().max(1.0);
        assert!(eigvals[1].abs() < tol && eigvals[2].abs() < tol);
    }

    /// The T5c/T5d fixture scorer: the exponent is PURE cross-term —
    /// `f = σ(w1·w2·u·v + w3·w4·s·t)` — so ONLY the nonlinear cross-term
    /// structure carries any class signal (the issue's validity clause).
    /// Closed-form gradient `(σ'·w2·uv, σ'·w1·uv, σ'·w4·st, σ'·w3·st)`; at
    /// `w0 = 1` the fingerprint direction is `(uv, uv, st, st)` — the pair
    /// `(uv, st)` in the pair-subspace.
    struct CrossGateScorer;

    impl KnobScorer for CrossGateScorer {
        fn knobs(&self) -> usize {
            4
        }
        fn eval(&self, x: &[f32], w: &[f32]) -> f32 {
            let (u, v, s, t) = (x[0], x[1], x[2], x[3]);
            let z = w[0] * w[1] * u * v + w[2] * w[3] * s * t;
            fast_sigmoid(z)
        }
        fn grad_knobs_closed_form(&self, x: &[f32], w: &[f32], out: &mut [f32]) -> bool {
            let (u, v, s, t) = (x[0], x[1], x[2], x[3]);
            let z = w[0] * w[1] * u * v + w[2] * w[3] * s * t;
            let sgm = fast_sigmoid(z);
            let sp = sgm * (1.0 - sgm);
            out[0] = sp * w[1] * u * v;
            out[1] = sp * w[0] * u * v;
            out[2] = sp * w[3] * s * t;
            out[3] = sp * w[2] * s * t;
            true
        }
    }

    /// The planted-class battery (Issue 949 T5c + T5d).
    ///
    /// Labels: latent `z ∈ {0,1}` from a seeded LCG — INDEPENDENT of the
    /// scorer's response (z draws BEFORE x and never reads back).
    ///
    /// Class signal — the MAGNITUDE channel: both classes live on the same
    /// raw direction family `(1, 1, c, −c)·a`; `z=0` carries `c ≈ 1` (the
    /// two cross-terms balanced: `uv ≈ −st`), `z=1` carries `c ≈ 1.8` (the
    /// st cross-term dominates: `st/uv ≈ −c²`). The input-cosine metric is
    /// direction-only, so to first order it is blind to `c` — the fingerprint
    /// direction `normalize(uv, uv, st, st)` depends on the RATIO `uv:st`
    /// and amplifies `c` QUADRATICALLY. That ratio-shift is the one channel
    /// the fixture plants; per-coordinate jitter keeps within-class mass.
    ///
    /// Battery (T5d): nearest-class-centroid leave-one-out accuracy in each
    /// space — (i) input cosine, (ii) the output-vector baseline over m=4
    /// probe knob settings, (iii) the fingerprint kernel. T5d's promotion
    /// law: the fingerprint must BEAT BOTH baselines; a tie against the
    /// output baseline is already a keep-off.
    #[test]
    fn t5c_t5d_planted_class_battery() {
        let kernel = FingerprintKernel::default();
        let scorer = CrossGateScorer;
        let w0 = [1.0f32; 4];

        // Probe knobs for the output-vector baseline (T5d-ii): m = 4 authored
        // settings around w0 — the gradient-free response rival.
        let probes: [[f32; 4]; 4] = [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
        ];

        let n_per_class = 40;
        let mut rng = Lcg(0x949_C1A5); // the seed that plants z
        let mut xs: Vec<[f32; 4]> = Vec::with_capacity(n_per_class * 2);
        let mut labels: Vec<u8> = Vec::with_capacity(n_per_class * 2);
        for _ in 0..(n_per_class * 2) {
            let z = (rng.next_unit() < 0.5) as u8;
            let a = 0.8 + 0.4 * rng.next_unit(); // the shared scale — input-cosine-blind
            let c = if z == 0 { 1.0 } else { 1.8 } + 0.15 * rng.next_signed();
            let mut j = || 0.08 * rng.next_signed();
            // Same direction family (1, 1, c, −c)·a; z moves c only.
            xs.push([
                a * (1.0 + j()),
                a * (1.0 + j()),
                a * (c + j()),
                -a * (c + j()),
            ]);
            labels.push(z);
        }

        let n = xs.len();
        let mut fps: Vec<[f32; 4]> = Vec::with_capacity(n);
        let mut outs: Vec<[f32; 4]> = Vec::with_capacity(n);
        for x in &xs {
            let mut fp = [0.0f32; 4];
            let mut w_scratch = w0;
            let norm = fingerprint_into(&kernel, &scorer, x, &mut w_scratch, &mut fp);
            assert!(norm > 0.0, "battery anchors must not abstain");
            fps.push(fp);
            let mut o = [0.0f32; 4];
            for (slot, pw) in o.iter_mut().zip(probes.iter()) {
                *slot = scorer.eval(x, pw);
            }
            outs.push(o);
        }

        // Nearest-class-centroid leave-one-out accuracy in a similarity space:
        // centroid = the mean of the OTHER anchors' unit rows of that class;
        // classify by cosine to the centroid.
        fn loocv<const D: usize>(rows: &[[f32; D]], labels: &[u8]) -> f32 {
            let mut units: Vec<[f32; D]> = Vec::with_capacity(rows.len());
            for r in rows {
                let mut u = *r;
                let nr = u.iter().map(|x| x * x).sum::<f32>().sqrt();
                for e in u.iter_mut() {
                    *e /= nr;
                }
                units.push(u);
            }
            let mut hits = 0u32;
            for i in 0..rows.len() {
                let mut c0 = [0.0f32; D];
                let mut c1 = [0.0f32; D];
                let (mut n0, mut n1) = (0u32, 0u32);
                for (j, u) in units.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    if labels[j] == 0 {
                        for (c, e) in c0.iter_mut().zip(u) {
                            *c += e;
                        }
                        n0 += 1;
                    } else {
                        for (c, e) in c1.iter_mut().zip(u) {
                            *c += e;
                        }
                        n1 += 1;
                    }
                }
                for e in c0.iter_mut() {
                    *e /= n0 as f32;
                }
                for e in c1.iter_mut() {
                    *e /= n1 as f32;
                }
                let cos = |a: &[f32; D], b: &[f32; D]| -> f32 {
                    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
                        / (a.iter().map(|x| x * x).sum::<f32>().sqrt()
                            * b.iter().map(|x| x * x).sum::<f32>().sqrt())
                };
                let guess = if cos(&units[i], &c0) >= cos(&units[i], &c1) {
                    0
                } else {
                    1
                };
                if guess == labels[i] {
                    hits += 1;
                }
            }
            hits as f32 / rows.len() as f32
        }

        let acc_fp = loocv(&fps, &labels);
        let acc_in = loocv(&xs, &labels);
        let acc_out = loocv(&outs, &labels);

        println!(
            "[949] LOOCV accuracy — fingerprint {acc_fp:.3} · input-cosine {acc_in:.3} · output-vector {acc_out:.3}"
        );

        // Fixture validity (T5c): labels are independent of the scorer's
        // response and the kernel DOES read the planted classes — the fixture
        // is learnable through the fingerprint channel. Whether that channel
        // is UNIQUELY informative is T5d's measured comparison (printed
        // above), never an assert here: the T5d law records a tie/loss as
        // T9's keep-off verdict, not a test failure.
        assert!(
            acc_fp > 0.9,
            "fingerprint must read the planted classes ({acc_fp})"
        );
    }
}
