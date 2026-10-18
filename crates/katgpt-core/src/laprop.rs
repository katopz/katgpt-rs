//! LaProp-style normalize-before-accumulate momentum (Issue 689).
//!
//! Source: [riir-train Research 428](../../riir-train/.research/428_LaProp_Decoupled_Momentum_Adaptive_Optimizer.md)
//! — LaProp (arXiv:2002.04839) Path-0 extraction, components C2/C3 (the
//! modelless half; the optimizer itself lives in riir-train Plan 354).
//!
//! # The ordering law (C3)
//!
//! **Accumulate-then-normalize** makes historical evidence's weight depend on
//! the stale-to-fresh magnitude ratio — unbounded under heavy tails (LaProp
//! §3.1, infinite variance). **Normalize-then-accumulate** makes it a function
//! of μ alone.
//!
//! Concretely: this accumulator maintains an EMA `m` over RMS-normalized
//! intake `u_t = x_t / √(n̂_t + ε)`, where `n̂` is the bias-corrected EMA of
//! `x²`. Because each `u_s` is normalized by the second moment estimate *at
//! its own time step* (`n̂_s ≥ x_s²·(1−ν)/(1−ν^s)`), every intake satisfies
//! `|u_s| ≤ √((1−ν^s)/(1−ν)) ≤ 1/√(1−ν)`, and `m` — a convex combination of
//! intakes with weights `(1−μ)μ^k` summing to 1 — inherits the same bound:
//!
//! ```text
//! |m|_∞ ≤ 1/√(1−ν)          (Prop-1-style closed form, per component)
//! |m|_2  ≤ √D/√(1−ν)        (L2 form — pinned for precision honesty)
//! ```
//!
//! Downstream accumulators that adopt this ordering can DELETE their clamps
//! and get a theorem instead (the clamp treats the symptom — unbounded raw
//! intake; the ordering removes the cause).
//!
//! # What this is NOT
//!
//! * NOT an optimizer — no gradient/loss semantics (riir-train Plan 354).
//! * NOT UQ-bearing — a bound, not an interval; no conformal-floor extension.
//! * No novelty claim — normalize-before-accumulate is the normalized-LMS
//!   family's classic pattern; the claim is no-analog-in-our-repos + the
//!   closed-form bound replacing clamps.
//!
//! # Consumer demand (the anti-pattern this deletes)
//!
//! riir-clippy `src/evolve.rs::ema_step` (the self_evolve direction-learning
//! core) accumulates `rate·(fix_dir[i] − dir[i])` RAW into `dir` then clamps
//! to ±1 — a heavy-tailed-intake site where one pathological trajectory delta
//! jerks the direction and the clamp hides it asymmetrically. Migration is a
//! follow-up in that repo (behavior-change A/B behind its own gate).
//!
//! laprop — LaProp normalize-before-accumulate momentum (Issue 689, from
//! riir-train Research 428 C2/C3): EMA over RMS-normalized intake with the
//! closed-form accumulator bound |m| ≤ 1/√(1−ν) — downstream accumulators
//! delete their clamps and get a theorem instead. Vector + scalar twins;
//! opt-in until a consumer adopts.

/// Default numerical floor inside the normalization denominator.
pub const DEFAULT_EPS: f32 = 1e-8;

/// LaProp-ordered momentum accumulator over a fixed-dim vector stream:
/// `m ← μ·m + (1−μ)·x/√(n̂+ε)`; `n ← ν·n + (1−ν)·x²` (bias-corrected `n̂`).
///
/// Zero crate deps, pure f32, caller-owned arrays — G4 alloc-free by
/// construction (all state is inline `[f32; D]`).
///
/// # Panics
///
/// `new`/`with_epsilon` panic unless `0 ≤ μ < 1` and `0 ≤ ν < 1` — a decay
/// rate outside that range is a caller configuration error, not a runtime
/// condition to silently absorb.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedMomentumAccumulator<const D: usize> {
    /// Momentum over NORMALIZED intake (bias correction applied on read).
    m: [f32; D],
    /// Running second moment of RAW intake.
    n: [f32; D],
    /// Steps pushed.
    t: u32,
    /// Momentum decay rate μ ∈ [0, 1).
    mu: f32,
    /// Second-moment decay rate ν ∈ [0, 1).
    nu: f32,
    /// Numerical floor in the normalization denominator.
    eps: f32,
    /// μ^t — maintained incrementally for O(1) bias correction.
    mu_pow: f32,
    /// ν^t — maintained incrementally for O(1) bias correction.
    nu_pow: f32,
}

impl<const D: usize> NormalizedMomentumAccumulator<D> {
    /// Construct with the default ε ([`DEFAULT_EPS`]).
    pub fn new(mu: f32, nu: f32) -> Self {
        Self::with_epsilon(mu, nu, DEFAULT_EPS)
    }

    /// Construct with an explicit ε.
    pub fn with_epsilon(mu: f32, nu: f32, eps: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&mu) && (0.0..1.0).contains(&nu),
            "laprop: need 0 <= mu < 1 and 0 <= nu < 1, got mu={mu}, nu={nu}"
        );
        assert!(eps > 0.0, "laprop: eps must be positive, got {eps}");
        Self {
            m: [0.0; D],
            n: [0.0; D],
            t: 0,
            mu,
            nu,
            eps,
            mu_pow: 1.0,
            nu_pow: 1.0,
        }
    }

    /// Push one observation. `m ← μ·m + (1−μ)·x/√(n̂+ε)`; `n ← ν·n + (1−ν)·x²`.
    #[inline]
    pub fn push(&mut self, x: &[f32; D]) {
        debug_assert!(
            x.iter().all(|v| v.is_finite()),
            "laprop: non-finite intake poisons the accumulator permanently"
        );
        self.t += 1;
        self.mu_pow *= self.mu;
        self.nu_pow *= self.nu;
        let n_bias = 1.0 - self.nu_pow; // > 0: ν^t < 1 for t ≥ 1, ν < 1
        let m_iter = self.m.iter_mut();
        let n_iter = self.n.iter_mut();
        for (m, (n, &xi)) in m_iter.zip(n_iter.zip(x.iter())) {
            *n = self.nu * *n + (1.0 - self.nu) * xi * xi;
            let n_hat = *n / n_bias;
            let u = xi / (n_hat + self.eps).sqrt();
            *m = self.mu * *m + (1.0 - self.mu) * u;
        }
    }

    /// Bias-corrected momentum `m̂ = m/(1−μ^t)`, by value (stack array — the
    /// corrected form is computed, so it cannot be borrowed; see
    /// [`Self::momentum_uncorrected`] for the raw borrow). Zero steps → zeros.
    pub fn momentum(&self) -> [f32; D] {
        let mut out = [0.0f32; D];
        self.momentum_into(&mut out);
        out
    }

    /// Zero-alloc bias-corrected momentum into a caller-owned buffer.
    #[inline]
    pub fn momentum_into(&self, out: &mut [f32; D]) {
        if self.t == 0 {
            // μ^0 = 1 → division by zero; the uncorrected value IS zeros.
            *out = self.m;
            return;
        }
        let m_bias = 1.0 - self.mu_pow;
        for (o, &m) in out.iter_mut().zip(self.m.iter()) {
            *o = m / m_bias;
        }
    }

    /// The raw (uncorrected) momentum state, borrowed.
    pub fn momentum_uncorrected(&self) -> &[f32; D] {
        &self.m
    }

    /// Steps pushed so far.
    #[inline]
    pub const fn steps(&self) -> u32 {
        self.t
    }

    /// Closed-form Prop-1-style accumulator bound: `1/√(1−ν)` (L∞, per
    /// component). Holds with NO clamp anywhere — see the module doc proof.
    pub fn bound(&self) -> f32 {
        Self::bound_for(self.nu)
    }

    /// Bound at an arbitrary ν (associated form — no instance needed).
    pub fn bound_for(nu: f32) -> f32 {
        1.0 / (1.0 - nu).sqrt()
    }

    /// Single-observation influence bound: `(1−μ)/√(1−ν)` — the largest share
    /// of the accumulator one fresh intake can move.
    pub fn influence(&self) -> f32 {
        (1.0 - self.mu) / (1.0 - self.nu).sqrt()
    }

    /// L2 form of the bound: `√D/√(1−ν)` — the per-component bound is L∞;
    /// pin both so the norm choice is never implicit.
    pub fn l2_bound(&self) -> f32 {
        (D as f32).sqrt() / (1.0 - self.nu).sqrt()
    }

    /// C8 doc note: Adam's *accumulate-then-normalize* coupling cost
    /// `1/(1−μ/√ν)` — INFINITE once μ ≥ √ν (the divergence regime Adam's
    /// bound is famous for). LaProp's per-time normalization keeps the
    /// accumulator bound μ-independent (see [`Self::bound`]) — this function
    /// exists so consumers can print/compare the contrast honestly.
    pub fn coupling_cost(mu: f32, nu: f32) -> f32 {
        let ratio = mu / nu.sqrt();
        if ratio >= 1.0 {
            f32::INFINITY
        } else {
            1.0 / (1.0 - ratio)
        }
    }
}

/// Scalar twin for 1-D signals (reward EMAs, priorities) — identical
/// ordering law and bounds, one component.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedMomentumScalar {
    m: f32,
    n: f32,
    t: u32,
    mu: f32,
    nu: f32,
    eps: f32,
    mu_pow: f32,
    nu_pow: f32,
}

impl NormalizedMomentumScalar {
    /// Construct with the default ε ([`DEFAULT_EPS`]).
    pub fn new(mu: f32, nu: f32) -> Self {
        Self::with_epsilon(mu, nu, DEFAULT_EPS)
    }

    /// Construct with an explicit ε.
    pub fn with_epsilon(mu: f32, nu: f32, eps: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&mu) && (0.0..1.0).contains(&nu),
            "laprop: need 0 <= mu < 1 and 0 <= nu < 1, got mu={mu}, nu={nu}"
        );
        assert!(eps > 0.0, "laprop: eps must be positive, got {eps}");
        Self {
            m: 0.0,
            n: 0.0,
            t: 0,
            mu,
            nu,
            eps,
            mu_pow: 1.0,
            nu_pow: 1.0,
        }
    }

    /// Push one observation.
    #[inline]
    pub fn push(&mut self, x: f32) {
        debug_assert!(
            x.is_finite(),
            "laprop: non-finite intake poisons the accumulator permanently"
        );
        self.t += 1;
        self.mu_pow *= self.mu;
        self.nu_pow *= self.nu;
        self.n = self.nu * self.n + (1.0 - self.nu) * x * x;
        let n_hat = self.n / (1.0 - self.nu_pow);
        let u = x / (n_hat + self.eps).sqrt();
        self.m = self.mu * self.m + (1.0 - self.mu) * u;
    }

    /// Bias-corrected momentum (0 steps → 0.0).
    pub fn momentum(&self) -> f32 {
        if self.t == 0 {
            return self.m;
        }
        self.m / (1.0 - self.mu_pow)
    }

    /// Steps pushed so far.
    #[inline]
    pub const fn steps(&self) -> u32 {
        self.t
    }

    /// Closed-form bound `1/√(1−ν)` — same proof as the vector form.
    pub fn bound(&self) -> f32 {
        1.0 / (1.0 - self.nu).sqrt()
    }

    /// Single-observation influence bound `(1−μ)/√(1−ν)`.
    pub fn influence(&self) -> f32 {
        (1.0 - self.mu) / (1.0 - self.nu).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// G1a: planted outlier — one 1e6 delta into a unit-scale stream. Every
    /// bias-corrected momentum component stays within `bound()·(1+1ulp)` with
    /// NO clamp anywhere in the primitive.
    #[test]
    fn g1a_planted_outlier_respects_closed_form_bound() {
        const D: usize = 8;
        let mut acc = NormalizedMomentumAccumulator::<D>::new(0.9, 0.9);
        // Deterministic unit-scale stream (alternating signs, magnitudes 0.5..1.5).
        let mut seed = 0x5A5Eu32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            0.5 + (seed & 0xFF) as f32 / 255.0
        };
        for step in 0..10_000 {
            let mut x = [0.0f32; D];
            for v in &mut x {
                *v = if (step + *v as usize).is_multiple_of(2) {
                    next()
                } else {
                    -next()
                };
            }
            acc.push(&x);
        }
        // THE planted outlier: every component spikes to 1e6.
        acc.push(&[1e6; D]);
        // …then 1_000 more clean steps (heavy-tail recovery must also hold).
        for step in 0..1_000 {
            let mut x = [0.0f32; D];
            for (i, v) in x.iter_mut().enumerate() {
                *v = if (step + i) % 2 == 0 { 1.0 } else { -1.0 };
            }
            acc.push(&x);
        }
        let bound = acc.bound(); // 1/√0.1 ≈ 3.1623
        let m = acc.momentum();
        for (i, &v) in m.iter().enumerate() {
            let limit = bound * (1.0 + f32::EPSILON); // +1 ulp headroom
            assert!(
                v.abs() <= limit,
                "G1a violated at component {i}: |{v}| > bound*(1+ulp) = {limit}"
            );
        }
        // And the L2 form holds as well (it follows, but pin it).
        let l2: f32 = m.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(l2 <= acc.l2_bound() * (1.0 + f32::EPSILON));
    }

    /// G1b: after the spike, the outlier's residual influence decays as
    /// EXACTLY μ^k (bit-identical to the formula) — zero intake contributes
    /// `μ·m + (1−μ)·0 = μ·m` per step, so the RAW accumulator decays by pure
    /// multiplication. (The exact law is on `m`, not the bias-corrected
    /// read `m̂` — the correction denominator `(1−μ^t)` legitimately changes
    /// as t grows; it is a read-side transform, not accumulator dynamics.)
    #[test]
    fn g1b_outlier_residual_decays_exactly_mu_pow_k() {
        const D: usize = 4;
        let mu = 0.9f32;
        let mut acc = NormalizedMomentumAccumulator::<D>::new(mu, 0.9);
        acc.push(&[1.0; D]);
        acc.push(&[1e6; D]); // the spike
        let m_after_spike = *acc.momentum_uncorrected();
        // k clean ZERO steps.
        const K: usize = 6;
        for _ in 0..K {
            acc.push(&[0.0; D]);
        }
        // Expected: μ^K · m_after_spike, computed by the same repeated
        // multiplication (each step is exactly `m ← μ·m + 0`).
        let mut pow = 1.0f32;
        for _ in 0..K {
            pow *= mu;
        }
        let m = acc.momentum_uncorrected();
        for i in 0..D {
            let expected = pow * m_after_spike[i];
            let same = m[i].to_bits() == expected.to_bits() || (m[i] == 0.0 && expected == 0.0); // ±0 sign-flip via `+0.0`
            assert!(
                same,
                "G1b: residual[{i}] = {} (bits {:#x}) != μ^{}·spike = {} (bits {:#x})",
                m[i],
                m[i].to_bits(),
                K,
                expected,
                expected.to_bits()
            );
        }
    }

    /// G1c: the CURRENT clamped raw-EMA shape (riir-clippy `ema_step` —
    /// `dir += rate·(fix − dir)`, clamp removed) FAILS the same no-clamp
    /// assertion — the A/B proving the gain is real.
    #[test]
    fn g1c_clamped_raw_ema_shape_fails_the_same_bound() {
        const D: usize = 8;
        let mu_or_rate = 0.9f32;
        let nu = 0.9f32;
        let bound = NormalizedMomentumAccumulator::<D>::bound_for(nu);
        // The raw-EMA shape, clamp deleted (the exact anti-pattern the issue
        // documents — `dir[i] = (dir[i] + applied).clamp(±1)` sans clamp).
        let mut dir = [1.0f32; D];
        let fix_dir = [1e6f32; D]; // pathological trajectory delta
        for i in 0..D {
            let applied = mu_or_rate * (fix_dir[i] - dir[i]);
            dir[i] += applied;
        }
        // WITHOUT its clamp the raw shape blows far past the closed-form bound.
        let max_abs = dir.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(
            max_abs > bound * 10.0,
            "G1c: raw-EMA sans clamp should violate the bound by >10x, got {max_abs} vs bound {bound}"
        );
    }

    /// Bias correction: after one push of a constant c, m̂ ≈ c/√(c²+ε) ≈ 1
    /// for c = 1 (the first-step estimate is the normalized intake itself).
    #[test]
    fn bias_correction_first_step() {
        let mut acc = NormalizedMomentumAccumulator::<3>::new(0.9, 0.9);
        acc.push(&[1.0; 3]);
        let m = acc.momentum();
        let expected = 1.0f32 / (1.0 + DEFAULT_EPS).sqrt();
        for &v in &m {
            assert!(
                (v - expected).abs() < 1e-6,
                "got {v}, expected ≈ {expected}"
            );
        }
        // Zero steps → zeros.
        let fresh = NormalizedMomentumAccumulator::<3>::new(0.9, 0.9);
        assert_eq!(fresh.momentum(), [0.0; 3]);
    }

    /// G5 (ν-dial): at ν = 0 the normalized intake degenerates to sign(x) —
    /// the ternary-sign limit — and the bound stays finite (1.0).
    #[test]
    fn g5_nu_zero_sign_limit() {
        let mut acc = NormalizedMomentumScalar::new(0.5, 0.0);
        for _ in 0..200 {
            acc.push(2.0); // constant positive stream
        }
        assert!(
            (acc.momentum() - 1.0).abs() < 1e-5,
            "ν=0 positive stream should saturate to sign=+1, got {}",
            acc.momentum()
        );
        let mut neg = NormalizedMomentumScalar::new(0.5, 0.0);
        for _ in 0..200 {
            neg.push(-3.0);
        }
        assert!(
            (neg.momentum() + 1.0).abs() < 1e-5,
            "ν=0 negative stream should saturate to sign=−1, got {}",
            neg.momentum()
        );
        assert_eq!(acc.bound(), 1.0); // 1/√(1−0) — finite at ν = 0
    }

    /// G5: monotone interpolation of the bound over ν ∈ (0, 1).
    #[test]
    fn g5_bound_monotone_in_nu() {
        let mut prev = 0.0f32;
        let mut nu = 0.0f32;
        while nu < 0.99 {
            let b = NormalizedMomentumScalar::new(0.9, nu).bound();
            assert!(
                b >= prev,
                "bound must be monotone in ν: {b} < {prev} at ν={nu}"
            );
            prev = b;
            nu += 0.05;
        }
    }

    /// Precision honesty: the Prop-1 bound is per-component (L∞); the L2
    /// form is √D/√(1−ν). Pin both expressions against the formulas.
    #[test]
    fn linfty_and_l2_forms_pinned() {
        let acc = NormalizedMomentumAccumulator::<8>::new(0.9, 0.9);
        let expected_inf = 1.0f32 / (1.0f32 - 0.9).sqrt();
        assert!((acc.bound() - expected_inf).abs() < 1e-5);
        let expected_l2 = 8.0f32.sqrt() * expected_inf;
        assert!((acc.l2_bound() - expected_l2).abs() < 1e-4);
        assert!((acc.influence() - 0.1 * expected_inf).abs() < 1e-5);
    }

    /// C8 coupling-cost contrast: Adam's 1/(1−μ/√ν) diverges once μ ≥ √ν,
    /// while the LaProp accumulator bound stays finite and μ-independent.
    #[test]
    fn coupling_cost_contrast() {
        // μ < √ν: finite.
        let c = NormalizedMomentumAccumulator::<2>::coupling_cost(0.9, 0.9);
        let expected = 1.0f32 / (1.0 - 0.9f32 / 0.9f32.sqrt());
        assert!((c - expected).abs() < 1e-4);
        // μ ≥ √ν (0.99 ≥ √0.9801 = 0.99): the Adam divergence regime.
        assert!(NormalizedMomentumAccumulator::<2>::coupling_cost(0.99, 0.9801).is_infinite());
        // …and the LaProp bound at the SAME (μ, ν) is finite + μ-independent.
        let la = NormalizedMomentumAccumulator::<2>::new(0.99, 0.9801);
        assert!(la.bound().is_finite());
        let lb = NormalizedMomentumAccumulator::<2>::new(0.5, 0.9801);
        assert!((la.bound() - lb.bound()).abs() < 1e-6);
    }

    /// Constructor validation.
    #[test]
    #[should_panic(expected = "0 <= mu < 1")]
    fn constructor_rejects_bad_rates() {
        let _ = NormalizedMomentumAccumulator::<2>::new(1.0, 0.9);
    }

    /// G4: zero allocs per push (lib-internal TrackingAllocator, debug only —
    /// the sentinel-skip pattern from `latent_confounder_audit.rs`).
    #[cfg(all(test, debug_assertions))]
    #[test]
    fn g4_zero_alloc_per_push() {
        crate::alloc::reset_alloc_stats();
        let mut acc = NormalizedMomentumAccumulator::<8>::new(0.9, 0.9);
        let x = [1.0f32; 8];
        let mut out = [0.0f32; 8];
        for _ in 0..1_000 {
            acc.push(&x);
            acc.momentum_into(&mut out);
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(count, 0, "push/momentum_into must be allocation-free");
    }

    /// G2: per-push cost ≤ clamped-EMA + ~15 ns at D=8 (one extra mul + FMA
    /// per component). Release-only — debug builds carry the tracking
    /// allocator + no optimization, ~10× off.
    #[cfg_attr(debug_assertions, ignore)]
    #[test]
    fn g2_push_cost_within_15ns_of_raw_ema() {
        const D: usize = 8;
        const N: usize = 100_000;
        let mut acc = NormalizedMomentumAccumulator::<D>::new(0.9, 0.9);
        let x = [1.0f32; D];
        // Warm up.
        for _ in 0..1_000 {
            acc.push(&x);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            acc.push(&x);
        }
        let laprop_ns = t0.elapsed().as_nanos() as f64 / N as f64;

        // The comparison arm: the raw-EMA step (dir += rate·(fix−dir)).
        let mut dir = [0.0f32; D];
        let fix = [1.0f32; D];
        let rate = 0.9f32;
        let t1 = std::time::Instant::now();
        for _ in 0..N {
            for i in 0..D {
                dir[i] += rate * (fix[i] - dir[i]);
            }
        }
        let raw_ns = t1.elapsed().as_nanos() as f64 / N as f64;

        assert!(
            laprop_ns - raw_ns < 15.0,
            "G2: laprop push {laprop_ns:.2} ns exceeds raw EMA {raw_ns:.2} ns by more than 15 ns"
        );
        assert!(
            laprop_ns < 500.0,
            "G2: absolute per-push budget blown: {laprop_ns:.2} ns"
        );
    }
}
