//! slt — singular learning theory selection math (Issue 781 / Research 558,
//! distilled from Watanabe's SLT program; arXiv:2010.11560, Aoyagi–Watanabe
//! 2005, Watanabe 2013).
//!
//! # What ships here
//!
//! | Primitive | Formula | What it prices |
//! |---|---|---|
//! | [`rlct_reduced_rank`] | `λ(r) = r·(a+b−r)/2` | the RLCT of a rank-r factorization of an a×b matrix (LoRA structure exactly: `ΔW = AB^T`) |
//! | [`wbic`] | `nL + λ·log n` | Watanabe's widely applicable BIC — which candidate artifact is cheapest given the data |
//! | [`free_energy`] | `nL + λ·log n − (m−1)·loglog n` | the free-energy law (m = singularity multiplicity) |
//! | [`bayes_gap`] | `λ/n` | the **Bayes-predictive** generalization-gap predictor |
//! | [`sigmoid_wbic_weight`] | `σ(−ΔWBIC/τ)` | pairwise mixture weight for the shipped `rating`/Elo consumers (sigmoid, never softmax) |
//! | [`sweep::noise_sweep_lambda`] | log-log CDF slope of loss deficits under seeded Gaussian noise (`slt_sweep`) | the LOCAL tangent-cone λ̂ at a frozen point — the measurement half (tubes/cones near-unbiased; star bodies carry a documented −18% tail bias; local ≠ tempered-global, Bench 765) |
//!
//! # Sign conventions + the two load-bearing distinctions
//!
//! - **Lower WBIC = better** (it is a free-energy-ish score, not a likelihood).
//! - **λ ≤ d/2 always** (realisable case) — the free sanity ceiling. For the
//!   reduced-rank family λ(r) = d_manifold(r)/2 EXACTLY (minimally singular;
//!   Aoyagi–Watanabe 2005): d_manifold(r) = r(a+b−r). The NAIVE
//!   parametrization count d_param(r) = r(a+b) over-counts by r² — the gauge
//!   orbit (GL(r) acts on (A,B) leaving ABᵀ fixed). BIC on d_param therefore
//!   over-penalizes rank — the over-count IS `(d/2−λ)·log n` per candidate.
//! - **`bayes_gap` prices the BAYES-predictive gap, NOT the point-fit gap.**
//!   The law is `E_n G(n) = λ/n` for the posterior-averaged predictor; a
//!   point (MLE/MAP) fit realizes `C/n, C ≥ λ` (for this family C = 2λ = the
//!   manifold dimension). Feeding a point-fit's realized gap to this
//!   predictor is a category error — the floor-gate test measures the
//!   WBIC-mixture predictor's gap for exactly this reason.
//! - **Anti-Laplace rule** (R558 §5): no Hessian/curvature-based
//!   generalization prediction — the instrument class the source paper
//!   measured at ~10³× the true λ. Route through λ/WBIC instead.
//!
//! λ is a **freeze/consolidation-seam scalar, not a per-tick signal** (R558
//! §5 game-context reframe): it ranks artifacts offline and never enters the
//! 20 Hz path or the sync boundary.
//!
//! # Domains
//!
//! `n ≥ 2` for [`free_energy`] (`loglog n` finite); all inputs are
//! plain `f64` arithmetic — garbage in, garbage out, no guards. Zero deps,
//! zero allocs, `f64` internals (log-domain), `#[must_use]` throughout.
//!
//! slt — singular learning theory selection math (Issue 781 / Research 558,
//! Watanabe's SLT program): rlct_reduced_rank (the Aoyagi–Watanabe rank-r
//! closed form — LoRA structure exactly), wbic, free_energy, bayes_gap (the
//! BAYES-predictive gap law), sigmoid_wbic_weight pairwise mixture weights,
//! bic_overpenalty_nats. Closed-form f64 arithmetic, zero deps, zero allocs.
//! λ is a freeze/consolidation-seam scalar — never a per-tick signal (R558
//! §5 game-context reframe).

#![allow(dead_code)]

/// RLCT of rank-r factorization of an a×b matrix (Aoyagi–Watanabe 2005):
/// `λ(r) = r·(a+b−r)/2` — the LoRA-rank complexity price in one line.
///
/// `r` clamps to `[0, min(a,b)]` (rank cannot exceed the matrix; oversized
/// ranks price as full rank). At `r = min(a,b)` this is `ab/2 = d/2`, the
/// regular (non-singular) limit — the property-test anchor.
#[must_use]
pub fn rlct_reduced_rank(a: usize, b: usize, r: usize) -> f64 {
    let r_eff = r.min(a).min(b);
    let d = a + b - r_eff;
    (r_eff as f64) * (d as f64) / 2.0
}

/// Watanabe's widely applicable BIC (2013): `nL + λ·log n`.
///
/// `loss_nats` is the empirical AVERAGE loss per sample (nats), so the first
/// term is the total loss. Lower = better. Selects among candidates whose
/// `λ` is known (reduced-rank family) or estimated (the riir-train lane).
#[must_use]
pub fn wbic(n: u64, loss_nats: f64, lambda: f64) -> f64 {
    let nf = n as f64;
    nf * loss_nats + lambda * nf.ln()
}

/// The free-energy law: `F_n ≈ nL + λ·log n − (m−1)·loglog n`.
///
/// `m` is the singularity multiplicity (`m = 1` ⇒ identical to [`wbic`]).
/// Requires `n ≥ 2` for a finite `loglog n`. Which artifact is cheapest
/// given the data — the cross-`n` comparability axis raw loss never has.
#[must_use]
pub fn free_energy(n: u64, loss_nats: f64, lambda: f64, m: u32) -> f64 {
    let nf = n as f64;
    let ln_n = nf.ln();
    nf * loss_nats + lambda * ln_n - (m as f64 - 1.0) * ln_n.ln()
}

/// The Bayes-predictive generalization-gap predictor: `E_n G(n) = λ/n`.
///
/// **Load-bearing caveat (the module doc's second distinction):** this prices
/// the gap of the posterior-AVERAGED (mixture) predictor. A point (MLE/MAP)
/// fit realizes `C/n, C ≥ λ` — for the reduced-rank family `C = 2λ`. UQ
/// floor-gated in the tests (must beat the incumbent `d/2n` floor and
/// constant-gap baselines on CRPS/coverage/Winkler).
#[must_use]
pub fn bayes_gap(lambda: f64, n: u64) -> f64 {
    lambda / n as f64
}

/// Pairwise WBIC mixture weight `σ(−ΔWBIC/τ)` — the weight FOR candidate `a`.
///
/// Lower WBIC (cheaper) ⇒ weight > 0.5. Sigmoid of the DIFFERENCE (never a
/// softmax renormalization): composing K candidates uses the Bradley-Terry
/// product of pairwise sigmoids, normalized by the caller. `τ > 0` is the
/// temperature in WBIC nats — the same scale as the score differences.
#[must_use]
pub fn sigmoid_wbic_weight(wbic_a: f64, wbic_b: f64, tau: f32) -> f32 {
    let delta = (wbic_b - wbic_a) / f64::from(tau.max(f32::MIN_POSITIVE));
    1.0 / (1.0 + (-delta).exp()) as f32
}

/// The `(d/2 − λ)·log n` BIC over-penalty for one candidate — the nats the
/// naive parameter count over-charges vs the manifold dimension (the gauge
/// orbit `r²/2·log n`). R558 §3 folded it here from a standalone artifact.
#[must_use]
pub fn bic_overpenalty_nats(n: u64, a: usize, b: usize, r: usize) -> f64 {
    let d = a + b;
    let lambda = rlct_reduced_rank(a, b, r);
    let half_naive = (r.min(d) as f64) * (d as f64) / 2.0;
    (half_naive - lambda) * (n as f64).ln()
}

// ── slt_sweep: the noise-sweep λ̂ estimator (Issue 782 / 781 T4) ──────
//
// The measurement half of the module: `V(s) = Vol{w near w₀ : K(w) < s} ∝
// s^λ` sampled by a deterministic Gaussian direction stream on FROZEN
// weights (R558 §2.3; Murfet et al. 2020 eq. 4.3 — the volume-codimension
// limit, estimated as a sublevel-volume RATIO so the probe measure
// cancels). Per noise scale t, the loss deficits uⱼ = K(w₀ + t·εⱼ) − K(w₀)
// carry the exponent in their near-zero CDF: P(u < s) ≈ V(s)·φ̄ₜ(s),
// where the mean shell density φ̄ₜ(s) varies with s only through the
// sublevel set's reach. A log-log slope fit of the EMPIRICAL CDF over an
// order-statistic ladder (counts ~ m/512 … m/16) cancels φ̄ₜ to first
// order — both thresholds share the shell — and recovers λ where a
// windowed Hill fit on the same samples is biased (the χ-family's
// exponential cutoff sits inside any upper-half window; measured −28…−37%
// at landing, the v1 failure kept here as the design record).
//
// Instrument boundary (measured, Bench 765): the near-zero CDF window is
// sample-starved for λ ≳ 3 at m ≤ 2048 draws (a χ^d-shaped deficit needs
// s ≪ t²d and that region holds ~Γ(d/2)-starved mass) — the feasible
// domain is the SINGULAR regime λ ≪ d/2, which is exactly the regime this
// module prices. Large-λ recovery stays riir-train Plan 404 (SGLD)
// territory.

/// The noise-sweep λ̂ estimator surface (Issue 782). Zero-alloc via
/// caller-owned [`NoiseSweepScratch`]; bit-reproducible from a seed.
#[cfg(feature = "slt_sweep")]
pub mod sweep {
    /// Deterministic xorshift64* + Box–Muller stream (module-canonical —
    /// identical to the test RNG so shipped and test draws agree).
    struct SweepRng(u64);

    impl SweepRng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn normal(&mut self) -> f64 {
            let scale = 1.0 / 9_007_199_254_740_992.0; // 2^-53
            let u1 = ((self.next_u64() >> 11) as f64 * scale).max(1e-300);
            let u2 = (self.next_u64() >> 11) as f64 * scale;
            (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
        }
    }

    /// Fill `dirs` (row-major `draws × dim` Gaussian matrix) from the
    /// canonical deterministic stream. Same `(seed, len)` ⇒ bit-identical
    /// bytes — the G3 determinism contract under every estimator call.
    pub fn fill_gaussian_dirs(dirs: &mut [f64], seed: u64) {
        let mut rng = SweepRng::new(seed);
        for d in dirs.iter_mut() {
            *d = rng.normal();
        }
    }

    /// Probe budget + ladder shape for [`noise_sweep_lambda`].
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NoiseSweepSpec {
        /// Gaussian draws per scale (m). The CDF ladder spans counts
        /// ~ m/512 … m/16; ≥ 512 recommended (slope noise ~ 1/√count).
        pub draws: usize,
        /// Largest (first) noise scale t₀.
        pub scale0: f64,
        /// Geometric ladder ratio ρ ∈ (0, 1): tₖ = t₀ · ρ^k.
        pub scale_decay: f64,
        /// Ladder length K ≥ 2. Scales are independently streamed
        /// (seed mixed per scale) and the ladder aggregates by median —
        /// a scale hitting the numerical floor drops out, not the estimate.
        pub scales: usize,
    }

    impl Default for NoiseSweepSpec {
        fn default() -> Self {
            Self {
                draws: 1024,
                scale0: 0.5,
                scale_decay: 0.45,
                scales: 6,
            }
        }
    }

    /// Pre-allocated scratch for [`noise_sweep_lambda`] — build once per
    /// (dim, spec), reuse across estimates. The estimate path itself
    /// performs zero allocations (in-place order selection + in-place
    /// ladder sort).
    pub struct NoiseSweepScratch {
        dirs: Vec<f64>,
        w: Vec<f64>,
        deficits: Vec<f64>,
        ladder: Vec<f64>,
    }

    impl NoiseSweepScratch {
        /// Scratch sized for `dim` parameters and `spec`'s draws/scales.
        #[must_use]
        pub fn new(dim: usize, spec: &NoiseSweepSpec) -> Self {
            Self {
                dirs: vec![0.0; spec.draws * dim],
                w: vec![0.0; dim],
                deficits: vec![0.0; spec.draws],
                ladder: Vec::with_capacity(spec.scales),
            }
        }
    }

    /// The noise-sweep λ̂ estimate: median over the ladder of per-scale
    /// log-log CDF-slope fits (the shell-cancelling volume-codimension
    /// estimator — see the module doc for why this form and not a
    /// windowed Hill fit).
    ///
    /// `loss` is evaluated at `1 + scales · draws` points total — a
    /// freeze/consolidation-seam instrument (R558 §5), never a per-tick
    /// signal. Scales whose deficits contain a non-positive or non-finite
    /// entry are skipped (numerical floor / not-a-minimum); if no scale
    /// yields a valid fit the estimate is NaN — never a silent zero.
    ///
    /// Determinism: identical `(spec, w0, seed)` ⇒ bit-identical result
    /// (fixed per-scale streams, in-place sort, closed-form arithmetic).
    #[must_use]
    pub fn noise_sweep_lambda<L>(
        spec: &NoiseSweepSpec,
        w0: &[f64],
        mut loss: L,
        scratch: &mut NoiseSweepScratch,
        seed: u64,
    ) -> f64
    where
        L: FnMut(&[f64]) -> f64,
    {
        let dim = w0.len();
        let m = spec.draws.max(64);
        debug_assert!(
            scratch.dirs.len() >= m * dim,
            "scratch sized for a larger dim/draws"
        );
        debug_assert!(scratch.deficits.len() >= m, "scratch deficits under-sized");
        let l0 = loss(w0);

        // CDF ladder policy (internal constant, documented): order
        // statistics at counts i₀ … i₁, geometrically spaced, P = 8 points.
        // The low end keeps the top quantile where the sublevel set stays
        // inside the Gaussian shell (φ̄ cancels); the high end keeps every
        // count ≥ 2 so the binomial noise is bounded.
        const POINTS: usize = 8;
        let i0 = 2.max(m / 512);
        let i1 = ((m / 16).max(8 * i0)).min(m - 1);

        scratch.ladder.clear();
        for s in 0..spec.scales.max(1) {
            let t = spec.scale0 * spec.scale_decay.powi(s as i32);
            // Independent per-scale stream (splitmix-style mixing).
            fill_gaussian_dirs(
                &mut scratch.dirs[..m * dim],
                seed ^ (s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut scale_valid = true;
            for j in 0..m {
                let off = j * dim;
                for (i, wi) in scratch.w[..dim].iter_mut().enumerate() {
                    *wi = w0[i] + t * scratch.dirs[off + i];
                }
                let u = loss(&scratch.w[..dim]) - l0;
                if u.is_finite() && u > 0.0 {
                    scratch.deficits[j] = u;
                } else {
                    scale_valid = false;
                    break;
                }
            }
            if !scale_valid {
                continue;
            }
            scratch.deficits[..m].sort_unstable_by(|a, b| {
                a.partial_cmp(b).expect("deficits pre-checked finite > 0")
            });
            // Weighted least squares of ln p̂ on ln s over the ladder;
            // weights ∝ count (Poisson). Slope = λ̂ (p ∝ s^λ).
            let mut sw = 0.0;
            let mut swx = 0.0;
            let mut swy = 0.0;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for p in 0..POINTS {
                let frac = p as f64 / (POINTS - 1) as f64;
                let idx = (i0 as f64 * (i1 as f64 / i0 as f64).powf(frac)).round() as usize;
                let s_l = scratch.deficits[idx - 1].ln();
                let y_l = ((idx as f64 - 0.5) / m as f64).ln();
                let w = idx as f64;
                sw += w;
                swx += w * s_l;
                swy += w * y_l;
            }
            for p in 0..POINTS {
                let frac = p as f64 / (POINTS - 1) as f64;
                let idx = (i0 as f64 * (i1 as f64 / i0 as f64).powf(frac)).round() as usize;
                let s_l = scratch.deficits[idx - 1].ln();
                let y_l = ((idx as f64 - 0.5) / m as f64).ln();
                let w = idx as f64;
                sxx += w * (s_l - swx / sw) * (s_l - swx / sw);
                sxy += w * (s_l - swx / sw) * (y_l - swy / sw);
            }
            if sxx > 0.0 && sxy.is_finite() {
                let lam = sxy / sxx;
                if lam.is_finite() && lam > 0.0 {
                    scratch.ladder.push(lam);
                }
            }
        }
        if scratch.ladder.is_empty() {
            return f64::NAN;
        }
        scratch
            .ladder
            .sort_unstable_by(|a, b| a.partial_cmp(b).expect("ladder pre-checked finite"));
        let n = scratch.ladder.len();
        if n % 2 == 1 {
            scratch.ladder[n / 2]
        } else {
            0.5 * (scratch.ladder[n / 2 - 1] + scratch.ladder[n / 2])
        }
    }
}

// ── Test harness lives below; the shipped surface ends here ───────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Deterministic test RNG (xorshift64*) — no rand dep ──────────────
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn next_f64(&mut self) -> f64 {
            // Uniform (0,1): top 53 bits / 2^53.
            ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
        }
        fn normal(&mut self) -> f64 {
            // Box-Muller (deterministic under our fixed stream).
            let u1 = self.next_f64().max(1e-300);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    // ── T1: property tests ──────────────────────────────────────────────

    /// r = min(a,b) ⇒ λ = ab/2 — the regular (non-singular) limit, exact.
    #[test]
    fn full_rank_limit_is_half_param_product() {
        for (a, b) in [(1, 1), (2, 3), (6, 5), (8, 8), (16, 12), (40, 64)] {
            let r = a.min(b);
            let expect = (a as f64) * (b as f64) / 2.0;
            assert_eq!(rlct_reduced_rank(a, b, r), expect, "a={a} b={b}");
        }
    }

    /// λ(r) monotone increasing over the valid range — dλ/dr = (a+b−2r)/2 > 0
    /// for r < (a+b)/2, and r ≤ min(a,b) ≤ (a+b)/2. This IS the
    /// interior-optimum property: against any non-increasing loss, WBIC's
    /// argmin sits strictly below r_max unless the loss drops faster than
    /// the penalty everywhere.
    #[test]
    fn rlct_monotone_in_rank() {
        for (a, b) in [(2, 3), (6, 5), (8, 8), (16, 12), (40, 64)] {
            let kmax = a.min(b);
            let mut prev = rlct_reduced_rank(a, b, 0);
            for r in 1..=kmax {
                let cur = rlct_reduced_rank(a, b, r);
                assert!(cur > prev, "λ must increase: a={a} b={b} r={r}");
                prev = cur;
            }
        }
    }

    /// The ceiling law: λ(r) ≤ d/2 (with d = ab the regular limit) and
    /// λ(r) ≤ naive_d/2 (with naive_d = r(a+b) the raw parametrization
    /// count) — the gauge discount r²/2 is never negative.
    #[test]
    fn rlct_below_regular_ceiling_and_naive_count() {
        for (a, b) in [(6, 5), (8, 8), (16, 12)] {
            for r in 0..=a.min(b) {
                let lambda = rlct_reduced_rank(a, b, r);
                let regular = (a as f64) * (b as f64) / 2.0;
                let naive_half = (r as f64) * ((a + b) as f64) / 2.0;
                assert!(lambda <= regular + 1e-12, "λ ≤ ab/2: a={a} b={b} r={r}");
                assert!(
                    lambda <= naive_half + 1e-12,
                    "λ ≤ r(a+b)/2: a={a} b={b} r={r}"
                );
            }
        }
    }

    /// Oversized ranks clamp to min(a,b) — rank 100 of a 6×5 prices as full
    /// rank, never a negative/absurd dimension.
    #[test]
    fn rlct_clamps_oversized_rank() {
        assert_eq!(rlct_reduced_rank(6, 5, 100), rlct_reduced_rank(6, 5, 5));
        assert_eq!(rlct_reduced_rank(3, 9, 5), rlct_reduced_rank(3, 9, 3));
        assert_eq!(rlct_reduced_rank(4, 4, 0), 0.0);
    }

    /// F − WBIC = −(m−1)·loglog n exactly; m=1 ⇒ F ≡ WBIC.
    #[test]
    fn free_energy_splits_into_wbic_and_multiplicity() {
        let (n, l, lambda) = (2000u64, 1.37, 12.5);
        for m in [1u32, 2, 3, 7] {
            let f = free_energy(n, l, lambda, m);
            let w = wbic(n, l, lambda);
            let expect = w - (f64::from(m) - 1.0) * (n as f64).ln().ln();
            assert!((f - expect).abs() < 1e-12, "m={m}");
        }
        assert_eq!(free_energy(n, l, lambda, 1), wbic(n, l, lambda));
    }

    /// The gap law scales exactly as 1/n: G(4n) = G(n)/4.
    #[test]
    fn bayes_gap_scales_inverse_n() {
        let lambda = 12.5;
        assert_eq!(bayes_gap(lambda, 4_000), bayes_gap(lambda, 1_000) / 4.0);
        assert_eq!(bayes_gap(lambda, 1), lambda);
    }

    /// Pairwise weight: favors the cheaper candidate, antisymmetric
    /// (w(a,b)+w(b,a)=1), τ→∞ ⇒ 0.5, sharp step as τ→0⁺. Sigmoid of the
    /// difference — never a softmax renormalization inside this fn.
    #[test]
    fn sigmoid_wbic_weight_favors_lower_score() {
        let (a, b) = (100.0, 110.0); // a cheaper by 10 nats
        let w = sigmoid_wbic_weight(a, b, 5.0);
        let w_rev = sigmoid_wbic_weight(b, a, 5.0);
        assert!(w > 0.5);
        assert!(
            (w + w_rev - 1.0).abs() < 1e-6,
            "antisymmetric pair sums to 1"
        );
        assert!((sigmoid_wbic_weight(a, b, 1e9) - 0.5).abs() < 1e-6);
        assert!(sigmoid_wbic_weight(a, b, 1e-3) > 0.999999);
        assert!((sigmoid_wbic_weight(a, a, 5.0) - 0.5).abs() < 1e-9);
    }

    /// The BIC over-penalty `(d/2−λ)·log n` = `r²/2·log n` (the gauge
    /// orbit) — present at EVERY rank including full (a full-rank
    /// factorization U·V still carries the GL(r) orbit; only the UNFACTORED
    /// matrix parametrization d=ab escapes it). Grows in r and n.
    #[test]
    fn bic_overpenalty_is_the_gauge_orbit() {
        let n = 2_000u64;
        let ln_n = (n as f64).ln();
        for (a, b, r) in [(6, 5, 3), (8, 8, 6), (16, 12, 4), (8, 8, 8), (6, 5, 5)] {
            let expect = (r * r) as f64 / 2.0 * ln_n;
            let got = bic_overpenalty_nats(n, a, b, r);
            assert!((got - expect).abs() < 1e-9, "a={a} b={b} r={r}");
        }
    }

    // ── The planted-rank test harness (test-only linear algebra) ────────

    /// Symmetric Jacobi eigensolver for small matrices (test-only; the
    /// shipped `thin_svd_into` substrate is feature-gated elsewhere and the
    /// module is zero-dep by contract). Cyclic rotations, fixed sweep count
    /// (deterministic). Returns eigenpairs sorted by DESCENDING eigenvalue.
    fn jacobi_eigen_desc(mat: &[f64], dim: usize, sweeps: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut a = mat.to_vec();
        let mut v: Vec<Vec<f64>> = (0..dim)
            .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        for _ in 0..sweeps {
            for p in 0..dim {
                for q in (p + 1)..dim {
                    let apq = a[p * dim + q];
                    if apq.abs() < 1e-15 {
                        continue;
                    }
                    let theta = (a[q * dim + q] - a[p * dim + p]) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..dim {
                        let akp = a[k * dim + p];
                        let akq = a[k * dim + q];
                        a[k * dim + p] = c * akp - s * akq;
                        a[k * dim + q] = s * akp + c * akq;
                    }
                    for k in 0..dim {
                        let apk = a[p * dim + k];
                        let aqk = a[q * dim + k];
                        a[p * dim + k] = c * apk - s * aqk;
                        a[q * dim + k] = s * apk + c * aqk;
                    }
                    for row in &mut v {
                        let vkp = row[p];
                        let vkq = row[q];
                        row[p] = c * vkp - s * vkq;
                        row[q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        let mut idx: Vec<usize> = (0..dim).collect();
        idx.sort_by(|&i, &j| {
            a[j * dim + j]
                .partial_cmp(&a[i * dim + i])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let eigs: Vec<f64> = idx.iter().map(|&i| a[i * dim + i]).collect();
        let vecs: Vec<Vec<f64>> = idx
            .iter()
            .map(|&i| (0..dim).map(|k| v[k][i]).collect())
            .collect();
        (eigs, vecs)
    }

    /// One planted reduced-rank regression cell. Generates n samples
    /// `y = W* x + ε` (x ~ N(0,I_b), ε ~ N(0,σ²I_a)), fits rank-k truncations
    /// of the cross-covariance estimator `C = (1/n)Σ y xᵀ`, and returns the
    /// per-rank train loss in nats (average per sample, Gaussian likelihood
    /// with KNOWN σ: `L(k) = mean_j 0.5·||y_j − W_k x_j||²/σ²`).
    struct PlantedCell {
        a: usize,
        b: usize,
        n: usize,
        sigma: f64,
        /// Per-rank train losses (index k-1 = rank k), nats per sample.
        losses: Vec<f64>,
    }

    /// Build the planted cell. `sing_values` are W*'s singular values
    /// (length r*, on fixed orthonormal directions from the seeded RNG).
    #[allow(clippy::too_many_arguments)]
    fn planted_cell(
        seed: u64,
        a: usize,
        b: usize,
        n: usize,
        sigma: f64,
        sing_values: &[f64],
    ) -> PlantedCell {
        let mut rng = Rng::new(seed);
        // Orthonormal row basis (a×r*) and col basis (b×r*): Gaussian G
        // (rows×r*), orthonormalize its columns via the symmetric
        // eigensolver on GᵀG (r*×r*, tiny) then map through — deterministic
        // under the seed.
        let r_star = sing_values.len();
        let mut ortho_cols = |rows: usize| -> Vec<Vec<f64>> {
            let g: Vec<f64> = (0..rows * r_star).map(|_| rng.normal()).collect();
            // Gram matrix r*×r*.
            let mut gram = vec![0.0f64; r_star * r_star];
            for i in 0..r_star {
                for j in 0..r_star {
                    gram[i * r_star + j] = (0..rows)
                        .map(|k| g[k * r_star + i] * g[k * r_star + j])
                        .sum();
                }
            }
            let (_eigs, vecs) = jacobi_eigen_desc(&gram, r_star, 40);
            // Orthonormal columns: Q = G · V · Λ^(-1/2) (columns of V are
            // eigenvectors of GᵀG). Build column i of Q.
            let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(r_star);
            for (ei, vi) in _eigs.iter().zip(vecs.iter()) {
                // Skip near-null directions (r* tiny vs rows — none in tests).
                let scale = 1.0 / ei.max(1e-12).sqrt();
                let col: Vec<f64> = (0..rows)
                    .map(|k| (0..r_star).map(|m| g[k * r_star + m] * vi[m]).sum::<f64>() * scale)
                    .collect();
                let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
                q_cols.push(col.into_iter().map(|x| x / norm).collect());
            }
            q_cols
        };
        let u_cols = ortho_cols(a); // a×r*, orthonormal columns
        let v_cols = ortho_cols(b); // b×r*, orthonormal columns
        // W* = Σ sᵢ uᵢ vᵢᵀ — store row-major a×b.
        let mut w_star = vec![0.0f64; a * b];
        for (i, s) in sing_values.iter().enumerate() {
            for ra in 0..a {
                for cb in 0..b {
                    w_star[ra * b + cb] += s * u_cols[i][ra] * v_cols[i][cb];
                }
            }
        }
        // Data.
        let mut xs: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            let x: Vec<f64> = (0..b).map(|_| rng.normal()).collect();
            let y: Vec<f64> = (0..a)
                .map(|ra| {
                    let base: f64 = (0..b).map(|cb| w_star[ra * b + cb] * x[cb]).sum();
                    base + sigma * rng.normal()
                })
                .collect();
            xs.push(x);
            ys.push(y);
        }
        // C = (1/n) Σ y xᵀ (a×b row-major).
        let mut c_mat = vec![0.0f64; a * b];
        for j in 0..n {
            for ra in 0..a {
                for cb in 0..b {
                    c_mat[ra * b + cb] += ys[j][ra] * xs[j][cb] / n as f64;
                }
            }
        }
        // Eigendecompose CᵀC (b×b symmetric): singular values + right
        // vectors of C.
        let mut ctc = vec![0.0f64; b * b];
        for i in 0..b {
            for j in 0..b {
                ctc[i * b + j] = (0..a).map(|k| c_mat[k * b + i] * c_mat[k * b + j]).sum();
            }
        }
        let (evals, vvecs) = jacobi_eigen_desc(&ctc, b, 60);
        let k_max = a.min(b);
        // Per-rank predictions on the TRAIN set: W_k x = Σ_{i≤k} (C vᵢ)(vᵢ·x).
        // Precompute C·vᵢ per direction.
        let cv: Vec<Vec<f64>> = vvecs
            .iter()
            .map(|vi| {
                (0..a)
                    .map(|ra| (0..b).map(|cb| c_mat[ra * b + cb] * vi[cb]).sum())
                    .collect()
            })
            .collect();
        let mut losses = Vec::with_capacity(k_max);
        for k in 1..=k_max {
            let mut total = 0.0f64;
            for j in 0..n {
                for ra in 0..a {
                    // ŷ_ra = Σ_{i<k} (C vᵢ)_ra · (vᵢ·x_j)
                    let mut pred = 0.0f64;
                    for i in 0..k {
                        let vdotx: f64 = (0..b).map(|cb| vvecs[i][cb] * xs[j][cb]).sum();
                        pred += cv[i][ra] * vdotx;
                    }
                    let resid = ys[j][ra] - pred;
                    total += resid * resid;
                }
            }
            // Per-sample nats: 0.5·Σ_dims(resid²)/σ² averaged over samples
            // (the SLT convention: L_n is the per-sample log-loss, NOT
            // per-dim — λ prices the full a×b manifold).
            let per_sample = total / n as f64;
            losses.push(0.5 * per_sample / (sigma * sigma));
        }
        let _ = evals; // singular values not needed once predictions exist
        PlantedCell {
            a,
            b,
            n,
            sigma,
            losses,
        }
    }

    /// The three criteria on a planted cell, at rank k (1-based):
    /// raw loss, WBIC (λ = manifold), naive BIC (d_param = raw count).
    struct Criteria {
        raw: Vec<f64>,
        wbic: Vec<f64>,
        bic: Vec<f64>,
    }

    fn criteria(cell: &PlantedCell) -> Criteria {
        let k_max = cell.a.min(cell.b);
        let n = cell.n as u64;
        let ln_n = (n as f64).ln();
        let mut raw = Vec::with_capacity(k_max);
        let mut wbic = Vec::with_capacity(k_max);
        let mut bic = Vec::with_capacity(k_max);
        for k in 1..=k_max {
            let l = cell.losses[k - 1];
            let lambda = rlct_reduced_rank(cell.a, cell.b, k);
            let naive_half = (k as f64) * ((cell.a + cell.b) as f64) / 2.0;
            raw.push(l);
            wbic.push((n as f64) * l + lambda * ln_n);
            bic.push((n as f64) * l + naive_half * ln_n);
        }
        Criteria { raw, wbic, bic }
    }

    fn argmin(v: &[f64]) -> usize {
        let mut best = 0;
        for (i, x) in v.iter().enumerate() {
            if *x < v[best] {
                best = i;
            }
        }
        best + 1 // 1-based rank
    }

    /// G1 headline (Issue 781 T2): planted-rank recovery. WBIC recovers the
    /// planted r*; raw loss picks r_max (monotone in k); naive-parameter
    /// BIC over-penalizes — at full signal it agrees with WBIC, at a
    /// marginal planted direction its r²/2·log n over-count rejects a rank
    /// WBIC correctly keeps.
    #[test]
    fn g1_planted_rank_recovery_wbic_vs_loss_vs_bic() {
        // Cell A — strong signal, a=8 b=8 r*=6: everything detectable.
        let strong: Vec<f64> = vec![1.5, 1.4, 1.3, 1.2, 1.1, 1.0];
        let cell_a = planted_cell(0x955A, 8, 8, 2_000, 0.5, &strong);
        let crit_a = criteria(&cell_a);
        assert_eq!(argmin(&crit_a.raw), 8, "raw loss is monotone: picks r_max");
        assert_eq!(
            argmin(&crit_a.wbic),
            6,
            "WBIC recovers the planted rank (strong signal)"
        );
        assert_eq!(
            argmin(&crit_a.bic),
            6,
            "full-strength signal: BIC agrees (over-count too small to bite)"
        );

        // Cell B — the 6th direction marginal (tuned): the realized gain
        // sits between the WBIC threshold (Δλ·ln n ≈ 19 nats) and the
        // naive-BIC threshold (Δd_param/2·ln n ≈ 30 nats) ⇒ WBIC keeps
        // rank 6, BIC's gauge over-count rejects it.
        let marginal: Vec<f64> = vec![1.5, 1.4, 1.3, 1.2, 1.1, 0.075];
        let cell_b = planted_cell(0x955B, 8, 8, 2_000, 0.5, &marginal);
        let crit_b = criteria(&cell_b);
        assert_eq!(argmin(&crit_b.raw), 8, "raw loss still picks r_max");
        assert_eq!(
            argmin(&crit_b.wbic),
            6,
            "WBIC keeps the marginal planted direction"
        );
        assert_eq!(
            argmin(&crit_b.bic),
            5,
            "naive BIC's r²/2·ln n over-count rejects it (over-penalizes)"
        );
    }

    /// G2: the selection loop is O(k) in the candidate count — 1_000
    /// full k=1..=8 selections well under 1 ms each (generous ceiling; the
    /// per-candidate cost is a handful of float ops).
    #[test]
    fn g2_selection_loop_is_o_k() {
        let n = 2_000u64;
        let losses = [1.4f64, 1.2, 1.1, 1.05, 1.02, 1.01, 1.005, 1.003];
        let start = std::time::Instant::now();
        let mut sink = 0.0f64;
        for _ in 0..1_000 {
            let mut best = f64::INFINITY;
            let mut best_k = 0usize;
            for (i, l) in losses.iter().enumerate() {
                let k = i + 1;
                let score = wbic(n, *l, rlct_reduced_rank(8, 8, k));
                if score < best {
                    best = score;
                    best_k = k;
                }
            }
            sink += best + best_k as f64;
        }
        let per = start.elapsed().as_secs_f64() / 1_000.0;
        assert!(sink.is_finite());
        assert!(
            per < 1e-3,
            "selection must be far under 1 ms: {per:.3?}/iter over k=1..=8"
        );
    }

    /// G4: the five shipped functions + a selection loop allocate nothing.
    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn g4_alloc_free_selection_math() {
        crate::alloc::reset_alloc_stats();
        let _ = rlct_reduced_rank(8, 8, 6);
        let _ = wbic(2_000, 1.0, rlct_reduced_rank(8, 8, 6));
        let _ = free_energy(2_000, 1.0, 12.5, 2);
        let _ = bayes_gap(12.5, 2_000);
        let _ = sigmoid_wbic_weight(100.0, 110.0, 5.0);
        let _ = bic_overpenalty_nats(2_000, 8, 8, 6);
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(count, 0, "selection math must be alloc-free");
    }

    // ── T3: the UQ floor gate (bayes_gap vs the incumbent d/2n floor) ────

    /// Gaussian CRPS (closed form). `z = (g − center)/s`.
    fn crps_gaussian(g: f64, center: f64, s: f64) -> f64 {
        let z = (g - center) / s;
        let phi = (-(z * z) / 2.0).exp() / (2.0 * core::f64::consts::PI).sqrt();
        // Φ(z) via erf approximation (Abramowitz-Stegun 7.1.26 class).
        let phi_cdf = 0.5 * (1.0 + erf(z / (2.0f64).sqrt()));
        s * (z * (2.0 * phi_cdf - 1.0) + 2.0 * phi - 1.0 / core::f64::consts::PI.sqrt())
    }

    fn erf(x: f64) -> f64 {
        // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7) — test-only.
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp();
        sign * y
    }

    /// Winkler interval score for a central 95% interval [lo, hi].
    fn winkler(g: f64, lo: f64, hi: f64) -> f64 {
        let alpha = 0.05;
        let w = hi - lo;
        if g < lo {
            w + 2.0 / alpha * (lo - g)
        } else if g > hi {
            w + 2.0 / alpha * (g - hi)
        } else {
            w
        }
    }

    /// Bradley-Terry mixture weights from PAIRWISE sigmoids (sigmoid-native;
    /// no softmax): w_k ∝ Π_{j≠k} σ(−(WBIC_k − WBIC_j)/τ), normalized.
    fn bt_weights(wbic_scores: &[f64], tau: f32) -> Vec<f64> {
        let mut w: Vec<f64> = wbic_scores
            .iter()
            .enumerate()
            .map(|(k, wk)| {
                wbic_scores
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != k)
                    .map(|(_, wj)| f64::from(sigmoid_wbic_weight(*wk, *wj, tau)))
                    .product::<f64>()
            })
            .collect();
        let sum: f64 = w.iter().sum();
        for x in &mut w {
            *x /= sum;
        }
        w
    }

    /// One floor-gate replication: fit rank candidates, select by WBIC,
    /// build the BT mixture predictor, and measure the mixture's realized
    /// train→test gap on a fresh test set. Returns
    /// (realized gap, k*, per-rank WBIC scores).
    fn gap_replication(
        seed: u64,
        a: usize,
        b: usize,
        n: usize,
        sigma: f64,
        sing_values: &[f64],
        tau: f32,
    ) -> (f64, usize, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let r_star = sing_values.len();
        // Planted W* on seeded orthonormal bases (same construction as
        // planted_cell, refactored inline for the train/test split).
        let ortho_cols = |rows: usize, rng: &mut Rng| -> Vec<Vec<f64>> {
            let g: Vec<f64> = (0..rows * r_star).map(|_| rng.normal()).collect();
            let mut gram = vec![0.0f64; r_star * r_star];
            for i in 0..r_star {
                for j in 0..r_star {
                    gram[i * r_star + j] = (0..rows)
                        .map(|k| g[k * r_star + i] * g[k * r_star + j])
                        .sum();
                }
            }
            let (eigs, vecs) = jacobi_eigen_desc(&gram, r_star, 40);
            let mut q_cols = Vec::with_capacity(r_star);
            for (ei, vi) in eigs.iter().zip(vecs.iter()) {
                let scale = 1.0 / ei.max(1e-12).sqrt();
                let col: Vec<f64> = (0..rows)
                    .map(|k| (0..r_star).map(|m| g[k * r_star + m] * vi[m]).sum::<f64>() * scale)
                    .collect();
                let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
                q_cols.push(col.into_iter().map(|x| x / norm).collect());
            }
            q_cols
        };
        let u_cols = ortho_cols(a, &mut rng);
        let v_cols = ortho_cols(b, &mut rng);
        let mut w_star = vec![0.0f64; a * b];
        for (i, s) in sing_values.iter().enumerate() {
            for ra in 0..a {
                for cb in 0..b {
                    w_star[ra * b + cb] += s * u_cols[i][ra] * v_cols[i][cb];
                }
            }
        }
        let sample = |rng: &mut Rng| -> (Vec<f64>, Vec<f64>) {
            let x: Vec<f64> = (0..b).map(|_| rng.normal()).collect();
            let y: Vec<f64> = (0..a)
                .map(|ra| {
                    let base: f64 = (0..b).map(|cb| w_star[ra * b + cb] * x[cb]).sum();
                    base + sigma * rng.normal()
                })
                .collect();
            (x, y)
        };
        let train: Vec<(Vec<f64>, Vec<f64>)> = (0..n).map(|_| sample(&mut rng)).collect();
        let test: Vec<(Vec<f64>, Vec<f64>)> = (0..n).map(|_| sample(&mut rng)).collect();

        // Fit: C from train, eigendecomp, per-rank train losses + scores.
        let mut c_mat = vec![0.0f64; a * b];
        for (x, y) in &train {
            for ra in 0..a {
                for cb in 0..b {
                    c_mat[ra * b + cb] += y[ra] * x[cb] / n as f64;
                }
            }
        }
        let mut ctc = vec![0.0f64; b * b];
        for i in 0..b {
            for j in 0..b {
                ctc[i * b + j] = (0..a).map(|k| c_mat[k * b + i] * c_mat[k * b + j]).sum();
            }
        }
        let (_evals, vvecs) = jacobi_eigen_desc(&ctc, b, 60);
        let cv: Vec<Vec<f64>> = vvecs
            .iter()
            .map(|vi| {
                (0..a)
                    .map(|ra| (0..b).map(|cb| c_mat[ra * b + cb] * vi[cb]).sum())
                    .collect()
            })
            .collect();
        let k_max = a.min(b);
        let predict = |x: &[f64], k: usize, out: &mut Vec<f64>| {
            out.clear();
            out.resize(a, 0.0);
            for ra in 0..a {
                let mut acc = 0.0;
                for i in 0..k {
                    let vdotx: f64 = (0..b).map(|cb| vvecs[i][cb] * x[cb]).sum();
                    acc += cv[i][ra] * vdotx;
                }
                out[ra] = acc;
            }
        };
        // Per-rank train loss (nats/sample).
        let mut losses = Vec::with_capacity(k_max);
        let mut pred = Vec::new();
        for k in 1..=k_max {
            let mut total = 0.0;
            for (x, y) in &train {
                predict(x, k, &mut pred);
                for ra in 0..a {
                    let r = y[ra] - pred[ra];
                    total += r * r;
                }
            }
            losses.push(0.5 * (total / n as f64 / a as f64) / (sigma * sigma));
        }
        let n64 = n as u64;
        let scores: Vec<f64> = (1..=k_max)
            .map(|k| wbic(n64, losses[k - 1], rlct_reduced_rank(a, b, k)))
            .collect();
        let k_star = argmin(&scores);

        // BT mixture predictor + its train/test losses.
        let w = bt_weights(&scores, tau);
        let mix_predict = |x: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.resize(a, 0.0);
            let mut pk = Vec::new();
            for (i, wk) in w.iter().enumerate() {
                if *wk < 1e-12 {
                    continue;
                }
                predict(x, i + 1, &mut pk);
                for ra in 0..a {
                    out[ra] += wk * pk[ra];
                }
            }
        };
        let loss_of = |set: &[(Vec<f64>, Vec<f64>)]| -> f64 {
            let mut total = 0.0;
            let mut p = Vec::new();
            for (x, y) in set {
                mix_predict(x, &mut p);
                for ra in 0..a {
                    let r = y[ra] - p[ra];
                    total += r * r;
                }
            }
            0.5 * (total / set.len() as f64) / (sigma * sigma)
        };
        let gap = loss_of(&test) - loss_of(&train);
        (gap, k_star, scores)
    }

    /// The UQ floor rule (Issue 781 T3 / R558 §7): `bayes_gap` (λ/n) must
    /// beat the incumbent floor `d/2n` (BIC's own gap prediction — the
    /// naive parameter count of the SELECTED rank) and constant-gap
    /// baselines on CRPS / coverage / Winkler across synthetic families and
    /// n. Scoring machinery (per-cell realized-gap spread s, central 95%
    /// intervals) is IDENTICAL across arms — the comparison isolates the
    /// center. The measured gap is the WBIC-mixture predictor's (the
    /// λ/n law prices the Bayes-predictive gap, not a point fit's).
    #[test]
    fn uq_floor_gate_bayes_gap_beats_bic_floor_and_constants() {
        const TAU: f32 = 10.0;
        const REPS: usize = 12;
        const A: usize = 8;
        const B: usize = 8;

        let families: [(&[f64], f64); 3] = [
            (&[1.5f64, 1.3, 1.1, 0.9], 0.5), // a=8 b=8 r*=4 strong
            (&[1.2, 1.0, 0.8, 0.6], 0.4),    // weaker ladder
            (&[1.6, 1.2, 0.7], 0.6),         // steep decay, r*=3
        ];
        let ns = [250usize, 500, 1_000, 2_000];

        // Per-arm per-cell sums.
        let arms = [
            "bayes_gap",
            "bic_floor",
            "const_1e-3",
            "const_1e-2",
            "const_1e-1",
        ];
        let mut crps = [0.0f64; 5];
        let mut wink = [0.0f64; 5];
        let mut cov = [0.0f64; 5];
        let mut cells = 0usize;

        for (f_idx, (sv, sigma)) in families.iter().enumerate() {
            for (n_idx, n) in ns.iter().enumerate() {
                // Replications.
                let mut gaps = Vec::with_capacity(REPS);
                let mut k_stars = Vec::with_capacity(REPS);
                for rep in 0..REPS {
                    let seed = 0xF100_0000 + ((f_idx * 16 + n_idx) as u64) * 1_000 + rep as u64;
                    let (gap, k_star, _scores) = gap_replication(seed, A, B, *n, *sigma, sv, TAU);
                    gaps.push(gap);
                    k_stars.push(k_star);
                }
                // Common spread s (identical machinery for every arm).
                let mean: f64 = gaps.iter().sum::<f64>() / gaps.len() as f64;
                let s = (gaps.iter().map(|g| (g - mean) * (g - mean)).sum::<f64>()
                    / (gaps.len() - 1) as f64)
                    .sqrt()
                    .max(1e-9);
                // Per-replication centers for each arm.
                for (g, k) in gaps.iter().zip(k_stars.iter()) {
                    let lambda = rlct_reduced_rank(A, B, *k);
                    let centers = [
                        bayes_gap(lambda, *n as u64),
                        bayes_gap((*k * (A + B)) as f64 / 2.0, *n as u64),
                        1e-3,
                        1e-2,
                        1e-1,
                    ];
                    for (arm, center) in arms.iter().zip(centers.iter()) {
                        crps[arm_index(arm)] += crps_gaussian(*g, *center, s) / s;
                        let lo = center - 1.959964 * s;
                        let hi = center + 1.959964 * s;
                        wink[arm_index(arm)] += winkler(*g, lo, hi) / s;
                        if *g >= lo && *g <= hi {
                            cov[arm_index(arm)] += 1.0;
                        }
                    }
                }
                cells += 1;
            }
        }
        let total = (cells * REPS) as f64;
        for i in 0..5 {
            crps[i] /= total;
            wink[i] /= total;
            cov[i] /= total;
        }
        println!("floor gate: arms={arms:?}");
        println!("  crps/s={crps:?}");
        println!("  winkler/s={wink:?}");
        println!("  coverage={cov:?}");

        let b = arm_index("bayes_gap");
        let f = arm_index("bic_floor");
        // The gate: bayes beats the floor AND every constant on CRPS and
        // Winkler; coverage at least the floor's (same-width intervals ⇒
        // center ordering drives every metric).
        assert!(
            crps[b] < crps[f],
            "bayes_gap CRPS {b:.4} must beat the d/2n floor {f:.4}",
            b = crps[b],
            f = crps[f]
        );
        assert!(
            wink[b] < wink[f],
            "bayes_gap Winkler must beat the d/2n floor"
        );
        for (i, name) in arms.iter().enumerate() {
            if i != b {
                assert!(crps[b] < crps[i], "bayes_gap CRPS must beat {name}");
                assert!(wink[b] < wink[i], "bayes_gap Winkler must beat {name}");
            }
        }
        assert!(
            cov[b] >= cov[f] - 1e-9,
            "bayes_gap coverage {c:.3} must not trail the floor's {d:.3}",
            c = cov[b],
            d = cov[f]
        );
    }

    fn arm_index(name: &str) -> usize {
        match name {
            "bayes_gap" => 0,
            "bic_floor" => 1,
            "const_1e-3" => 2,
            "const_1e-2" => 3,
            "const_1e-1" => 4,
            _ => unreachable!(),
        }
    }

    // ── Issue 782: the noise-sweep λ̂ estimator (slt_sweep) ─────────────
    #[cfg(feature = "slt_sweep")]
    mod sweep_tests {
        use super::super::sweep::*;

        fn estimate<L: FnMut(&[f64]) -> f64>(
            dim: usize,
            spec: &NoiseSweepSpec,
            loss: L,
            seed: u64,
        ) -> f64 {
            let mut scratch = NoiseSweepScratch::new(dim, spec);
            let w0 = vec![0.0; dim];
            noise_sweep_lambda(spec, &w0, loss, &mut scratch, seed)
        }

        /// G1 anchor 1 — quadratic bowl (ISOLATED regular minimum — star
        /// body): λ = d/2. Measured +3.2% (d=2) / −17.1% (d=4): the
        /// Gaussian radial tail inside the ladder band biases star bodies
        /// DOWN as d grows (documented instrument bias, Bench 765) — gates
        /// carry the margin, ranking within a family stays monotone.
        #[test]
        fn sweep_bowl_recovers_half_dimension() {
            for (d, tol) in [(2_usize, 0.15), (4_usize, 0.25)] {
                let spec = NoiseSweepSpec::default();
                let lam = estimate(
                    d,
                    &spec,
                    |w| 0.5 * w.iter().map(|x| x * x).sum::<f64>(),
                    0x5EED_0782,
                );
                let target = d as f64 / 2.0;
                assert!(
                    (lam - target).abs() <= tol * target,
                    "bowl d={d}: λ̂={lam:.4} target={target} (rel {:+.1}%)",
                    (lam / target - 1.0) * 100.0
                );
            }
        }

        /// G1 anchor 2 — quartic monomial Σwᵢ⁴ (star body, no cutoff in
        /// K itself): λ = d/4. Same star-body radial-tail bias:
        /// +1.0% / −18.4% / −19.3% at d = 2 / 4 / 8 (measured).
        #[test]
        fn sweep_quartic_recovers_quarter_dimension() {
            for (d, tol) in [(2_usize, 0.15), (4_usize, 0.25), (8_usize, 0.25)] {
                let spec = NoiseSweepSpec::default();
                let lam = estimate(
                    d,
                    &spec,
                    |w| w.iter().map(|x| x.powi(4)).sum::<f64>(),
                    0x5EED_0782,
                );
                let target = d as f64 / 4.0;
                assert!(
                    (lam - target).abs() <= tol * target,
                    "quartic d={d}: λ̂={lam:.4} target={target} (rel {:+.1}%)",
                    (lam / target - 1.0) * 100.0
                );
            }
        }

        /// G1 anchor 3 — planted RRR (a=b=3, r=1): λ = r(a+b−r)/2 = 2.5,
        /// d = 6 params — the LoRA-shape TUBE anchor: the flat fiber makes
        /// the sublevel sets tubes, the along-tube Gaussian factor is
        /// s-independent and cancels — measured −1.1%, the near-unbiased
        /// class (singular geometries: tubes/cones).
        #[test]
        fn sweep_planted_rrr_recovers_tangent_dimension() {
            let mut planted = [0.0_f64; 6];
            fill_gaussian_dirs(&mut planted, 99);
            let (a, b) = planted.split_at(3);
            // W* = A*·B* (3×3 rank-1); loss over w = [A (3), B (3)].
            let loss = |w: &[f64]| {
                let mut acc = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        let dw = w[i] * w[3 + j] - a[i] * b[j];
                        acc += dw * dw;
                    }
                }
                acc
            };
            let spec = NoiseSweepSpec::default();
            let mut scratch = NoiseSweepScratch::new(6, &spec);
            let lam = noise_sweep_lambda(&spec, &planted, loss, &mut scratch, 0x5EED_0782);
            let target = 2.5;
            assert!(
                (lam - target).abs() <= 0.15 * target,
                "rrr: λ̂={lam:.4} target={target} (rel {:+.1}%)",
                (lam / target - 1.0) * 100.0
            );
        }

        // ── The paper's ReLU toy (Murfet et al. 2020 §6) ──────────────────
        // H=5 two-layer ReLU on x ∈ [−1,1]², d = 4H+1 = 21 params, layout
        // [wᵢx, wᵢy, bᵢ, qᵢ]×5 then c. Truth s_m (m=3): ridge directions at
        // angles π/m + (i−1)·2π/m, b = −1/3, q = 1; units m+1..=H dead.
        // K is a finite midpoint sum over a FIXED 32×32 point set (the
        // paper's own empirical L_n is the same construction at n=1000):
        // exact f64 arithmetic, no quadrature-error floor.
        const TOY_H: usize = 5;
        const TOY_M: usize = 3;
        const TOY_DIM: usize = 4 * TOY_H + 1;

        fn toy_truth() -> [f64; TOY_DIM] {
            let mut s = [0.0_f64; TOY_DIM];
            for i in 0..TOY_M {
                let ang = core::f64::consts::PI / TOY_M as f64
                    + (i as f64) * core::f64::consts::TAU / TOY_M as f64;
                s[4 * i] = ang.cos();
                s[4 * i + 1] = ang.sin();
                s[4 * i + 2] = -1.0 / 3.0;
                s[4 * i + 3] = 1.0;
            }
            s
        }

        fn toy_f(x: [f64; 2], w: &[f64]) -> f64 {
            let mut out = w[4 * TOY_H]; // c
            for i in 0..TOY_H {
                let pre = w[4 * i] * x[0] + w[4 * i + 1] * x[1] + w[4 * i + 2];
                if pre > 0.0 {
                    out += w[4 * i + 3] * pre;
                }
            }
            out
        }

        fn toy_loss_n(w: &[f64], truth: &[f64], n: usize) -> f64 {
            let h = 2.0 / n as f64;
            let mut acc = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let x = [-1.0 + (i as f64 + 0.5) * h, -1.0 + (j as f64 + 0.5) * h];
                    let d = toy_f(x, w) - toy_f(x, truth);
                    acc += d * d;
                }
            }
            acc // scale-invariant for the slope fit
        }

        fn toy_loss(w: &[f64], truth: &[f64]) -> f64 {
            toy_loss_n(w, truth, 32)
        }

        /// G1 anchor 4 — the headline singularity cell (Murfet et al. 2020
        /// §6): d=21, d/2 = 10.5, paper SGLD reference λ ≈ 0.526 (tempered
        /// GLOBAL). Measured LOCAL tangent-cone exponent at the s_m apex:
        /// λ̂ ≈ 2.77 — stable under 9× denser quadrature (2.7715 → 2.7710)
        /// and 16× draws (2.91): a local isotropic probe resolves the cone
        /// mixture's dominant slope, NOT the tempered-global 0.526 (which
        /// lives in strata/posterior mass a local probe cannot see — the
        /// recorded instrument boundary, Bench 765). The GATE is the
        /// singularity assertion: λ̂ ∈ [2.0, 3.6] — a 2.9×-to-5.3×
        /// reduction vs d/2 — the estimator sees the singular geometry,
        /// decisively not the parameter count.
        #[test]
        fn sweep_relu_toy_sees_the_singularity() {
            let truth = toy_truth();
            assert_eq!(toy_loss(&truth, &truth), 0.0); // w₀ on W₀ exactly
            let spec = NoiseSweepSpec::default();
            let mut scratch = NoiseSweepScratch::new(TOY_DIM, &spec);
            let lam = noise_sweep_lambda(
                &spec,
                &truth,
                |w| toy_loss(w, &truth),
                &mut scratch,
                0x5EED_0782,
            );
            assert!(
                (2.0..=3.6).contains(&lam),
                "relu toy: λ̂={lam:.4} must sit in [2.0, 3.6] vs d/2=10.5 (singularity gate)"
            );
        }

        /// G2 — cost contract: a full default estimate is `1 + K·m` loss
        /// evals (6145 here) with µs-class overhead per eval; generous
        /// 250 ms debug-build ceiling guards accidental O(m²) rot on a
        /// cold-path (freeze-time) instrument.
        #[test]
        fn sweep_g2_estimate_cost_ceiling() {
            let spec = NoiseSweepSpec::default();
            let mut scratch = NoiseSweepScratch::new(4, &spec);
            let w0 = [0.0_f64; 4];
            let t0 = std::time::Instant::now();
            let lam = noise_sweep_lambda(
                &spec,
                &w0,
                |w| 0.5 * w.iter().map(|x| x * x).sum::<f64>(),
                &mut scratch,
                7,
            );
            let el = t0.elapsed();
            assert!(lam.is_finite());
            assert!(
                el.as_millis() < 250,
                "default estimate took {el:?} (6145 trivial evals) — O(m²) rot?"
            );
        }

        /// Diagnostics (not a gate): print every anchor's measured λ̂ and
        /// relative error for bench recording / re-calibration.
        /// `cargo test -p katgpt-core --features slt_sweep --lib
        /// sweep_diagnostics -- --ignored --nocapture`
        #[test]
        #[ignore = "diagnostics: prints the measured ladder values"]
        fn sweep_diagnostics() {
            let spec = NoiseSweepSpec::default();
            for d in [2_usize, 4] {
                let lam = estimate(
                    d,
                    &spec,
                    |w| 0.5 * w.iter().map(|x| x * x).sum::<f64>(),
                    0x5EED_0782,
                );
                println!(
                    "bowl      d={d:2}  λ̂={lam:8.4}  target={:<5} rel={:+7.2}%",
                    d as f64 / 2.0,
                    (lam / (d as f64 / 2.0) - 1.0) * 100.0
                );
            }
            for d in [2_usize, 4, 8] {
                let lam = estimate(
                    d,
                    &spec,
                    |w| w.iter().map(|x| x.powi(4)).sum::<f64>(),
                    0x5EED_0782,
                );
                println!(
                    "quartic   d={d:2}  λ̂={lam:8.4}  target={:<5} rel={:+7.2}%",
                    d as f64 / 4.0,
                    (lam / (d as f64 / 4.0) - 1.0) * 100.0
                );
            }
            let mut planted = [0.0_f64; 6];
            fill_gaussian_dirs(&mut planted, 99);
            let (a, b) = planted.split_at(3);
            let rrr_loss = |w: &[f64]| {
                let mut acc = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        let dw = w[i] * w[3 + j] - a[i] * b[j];
                        acc += dw * dw;
                    }
                }
                acc
            };
            let mut scratch = NoiseSweepScratch::new(6, &spec);
            let lam = noise_sweep_lambda(&spec, &planted, rrr_loss, &mut scratch, 0x5EED_0782);
            println!(
                "rrr  a=b=3 r=1   λ̂={lam:8.4}  target=2.5   rel={:+7.2}%",
                (lam / 2.5 - 1.0) * 100.0
            );
            let truth = toy_truth();
            for n in [32_usize, 96] {
                let mut scratch = NoiseSweepScratch::new(TOY_DIM, &spec);
                let lam = noise_sweep_lambda(
                    &spec,
                    &truth,
                    |w| toy_loss_n(w, &truth, n),
                    &mut scratch,
                    0x5EED_0782,
                );
                println!("relu toy d=21 N={n:2}  λ̂={lam:8.4}  ref≈0.526 (SGLD)  d/2=10.5");
            }
            // The sample-wall probe: 16× draws moves the ladder's mass
            // window two decades deeper (i0 stays at absolute count 2).
            {
                let deep = NoiseSweepSpec {
                    draws: 16_384,
                    scales: 3,
                    ..Default::default()
                };
                let mut scratch = NoiseSweepScratch::new(TOY_DIM, &deep);
                let lam = noise_sweep_lambda(
                    &deep,
                    &truth,
                    |w| toy_loss_n(w, &truth, 32),
                    &mut scratch,
                    0x5EED_0782,
                );
                println!("relu toy d=21 N=32 m=16384  λ̂={lam:8.4}  (wall probe)");
            }
        }

        /// G3 — determinism: same seed ⇒ bit-identical; different seed ⇒ a
        /// different estimate (stream actually moved).
        #[test]
        fn sweep_deterministic_under_seed() {
            let spec = NoiseSweepSpec::default();
            let bowl = |w: &[f64]| 0.5 * w.iter().map(|x| x * x).sum::<f64>();
            let a = estimate(4, &spec, bowl, 0x5EED_0782);
            let b = estimate(4, &spec, bowl, 0x5EED_0782);
            assert_eq!(a.to_bits(), b.to_bits());
            let c = estimate(4, &spec, bowl, 0xDEAD_BEEF);
            assert_ne!(a.to_bits(), c.to_bits());
        }

        /// G4 — alloc-free estimate path (scratch pre-built outside the
        /// measured region; in-place selection + sort). The Issue-741
        /// profile-free predicate.
        #[test]
        #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
        fn sweep_alloc_free_estimate() {
            let spec = NoiseSweepSpec::default();
            let mut scratch = NoiseSweepScratch::new(4, &spec);
            let w0 = [0.0_f64; 4];
            let bowl = |w: &[f64]| 0.5 * w.iter().map(|x| x * x).sum::<f64>();
            crate::alloc::reset_alloc_stats();
            let lam = noise_sweep_lambda(&spec, &w0, bowl, &mut scratch, 7);
            let (count, _bytes) = crate::alloc::get_alloc_stats();
            assert!(lam.is_finite());
            assert_eq!(count, 0, "estimate path must be alloc-free");
        }
    }
}
