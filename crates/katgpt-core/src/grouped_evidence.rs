//! grouped_evidence — noise-weighting primitives for grouped noisy evidence
//! (Issue 913, the MODELLESS half of riir-train Research 463 ← arXiv:2609.36802
//! "EasyPPO"; the training-side twin is riir-train Plan 431).
//!
//! The paper's noise-normalized critic regression decomposes into closed-form
//! statistics that know nothing about critics: the gradient second moment of
//! any squared-loss estimator on grouped noisy targets is
//! `‖h‖²·[pred_err² + Var(R | group)]`, so as the fit improves the NOISE term
//! dominates, and weighting each group by `1 / max(σ̂, ε)` equalises the
//! contributions. For EXOGENOUS weights — a function of the group, not of the
//! outcome — the reweighting preserves the estimation optimum. That invariance
//! law is the whole license for this module, and its scope is enforced in the
//! types below rather than remembered.
//!
//! | Primitive | Law |
//! |---|---|
//! | [`variance_floor`] (T1) | `ε = Δ / (2√n)` — Popoviciu (`σ ≤ Δ/2` on `[a, a+Δ]`) over an `n`-draw mean; tight at the two-point `{a, a+Δ}`, `p = ½` |
//! | [`filter_bias_bound`] (T2) | `|E[R | C] − E[R]| ≤ 2Γ·P(¬C)` for `|R| ≤ Γ` — the censoring decomposition; tight at `R = +Γ` on `C`, `−Γ` on `¬C` |
//! | [`best_belief_score_weighted`] (T3) | fractional-count Beta LCB with floored, mean-one weights and Kish `n_eff` disclosed |
//! | [`noise_scaled_k`] + the `update_*_noise_scaled` pair (T4) | `K_eff = K · ε / max(σ̂, ε)` — bit-identical to the fixed-K update at the floor |
//!
//! # The endogeneity law (load-bearing — Issue 913 verdict rounds 1 + 2)
//!
//! The invariance law holds only for weights independent of the outcome given
//! the group. CROSS-group pooling (T3) and rating updates (T4) therefore need
//! `σ̂` from a source INDEPENDENT of the counts being aggregated: a prior
//! epoch, a leave-one-out estimate, or a design-level variance. The counts'
//! own plug-in `σ̂_g = √(p̂_g(1−p̂_g))` is a deterministic function of `k_g`:
//! groups near the extremes get the smallest `σ̂`, are floored to the LARGEST
//! weight, and drag the pooled rate toward the extremes. The tests pin that
//! failure mode as a measured negative control, not a caveat.
//!
//! [`ExogenousSigma`] carries the provenance at the call site; there is no
//! constructor named "plug-in", on purpose. Under group-rate HETEROGENEITY
//! the weighted Beta estimates the `w`-weighted estimand
//! `Σ w_g n_g p_g / Σ w_g n_g`, not the unweighted one — every readout names
//! which estimand it carries ([`Estimand`]).
//!
//! # Allocation discipline (G4)
//!
//! Every function takes slices / scalars by value or reference and returns a
//! `Copy` value. No `Vec`, `Box`, `String` or collecting iterator.
//!
//! Grouped-evidence noise-weighting primitives (Issue 913, riir-train
//! Research 463 ← arXiv:2609.36802 EasyPPO, the modelless half): Popoviciu
//! variance floor, filter-bias bound, exogenous-σ̂ weighted Beta LCB with Kish
//! `n_eff` disclosed, and noise-scaled rating K. Opt-in
//! (`grouped_evidence_weighting`).

use crate::best_belief::{best_belief_score, beta_quantile};
use crate::rating;

// ──────────────────────────────────────────────────────────────────────────
// T1 — variance floor
// ──────────────────────────────────────────────────────────────────────────

/// The worst-case standard error of an `n`-draw mean of a quantity bounded in
/// an interval of width `delta`: `Δ / (2√n)`.
///
/// Popoviciu bounds any distribution on `[a, a+Δ]` by `σ ≤ Δ/2`, attained by
/// the two-point `{a, a+Δ}` at `p = ½`; the mean of `n` i.i.d. draws then has
/// `σ/√n ≤ Δ/(2√n)`. Use it as the `ε` floor in `1 / max(σ̂, ε)` — the
/// smallest noise a group of `n` bounded draws can credibly be assigned
/// without a variance source tighter than the bound itself — and as the
/// design-level `σ` of a bounded-outcome group when nothing better exists.
///
/// `n = 0` is read as `n = 1` (one draw carries the full `Δ/2`), and the
/// width is taken by magnitude.
#[inline]
#[must_use]
pub fn variance_floor(delta: f32, n: u32) -> f32 {
    delta.abs() / (2.0 * (n.max(1) as f32).sqrt())
}

// ──────────────────────────────────────────────────────────────────────────
// T2 — filter bias bound
// ──────────────────────────────────────────────────────────────────────────

/// The maximum bias any conditionally-filtered readout can carry:
/// `2Γ·P(¬C)` for a reward bounded by `|R| ≤ Γ` (EasyPPO Eq. 24).
///
/// From `E[R] = P(C)·E[R|C] + P(¬C)·E[R|¬C]`,
/// `E[R|C] − E[R] = P(¬C)·(E[R|C] − E[R|¬C])`, and the bracket is at most
/// `2Γ`. Tight at `R = +Γ` on `C`, `−Γ` on `¬C`. The DISCLOSURE half of the
/// Report-the-Floor law for filtered readouts: a readout that conditions on a
/// filter `C` prints this beside its number.
///
/// `p_not_c` is clamped to `[0, 1]`; `gamma` is taken by magnitude.
#[inline]
#[must_use]
pub fn filter_bias_bound(gamma: f32, p_not_c: f32) -> f32 {
    2.0 * gamma.abs() * p_not_c.clamp(0.0, 1.0)
}

// ──────────────────────────────────────────────────────────────────────────
// T3 — weighted best-belief score
// ──────────────────────────────────────────────────────────────────────────

/// Where a per-group `σ̂` came from. Every variant is a source INDEPENDENT of
/// the counts the weights are applied to — the endogeneity law (module doc).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigmaProvenance {
    /// Estimated on a previous epoch of the same groups.
    PriorEpoch,
    /// Estimated with the aggregated observation held out.
    LeaveOneOut,
    /// Known from the design (group size, outcome bound, instrument class).
    Design,
}

/// Per-group noise scales from an independent source, tagged with that source.
///
/// A newtype rather than a bare `&[f32]` so that the provenance is spelled at
/// every call site. There is deliberately no plug-in constructor: a `σ̂`
/// computed from the very counts being pooled is the failure mode the
/// negative control in this module measures.
#[derive(Clone, Copy, Debug)]
pub struct ExogenousSigma<'a> {
    sigma: &'a [f32],
    provenance: SigmaProvenance,
}

impl<'a> ExogenousSigma<'a> {
    /// `σ̂` from a previous epoch of the same groups.
    #[inline]
    #[must_use]
    pub fn prior_epoch(sigma: &'a [f32]) -> Self {
        Self {
            sigma,
            provenance: SigmaProvenance::PriorEpoch,
        }
    }

    /// `σ̂` with the aggregated observation held out.
    #[inline]
    #[must_use]
    pub fn leave_one_out(sigma: &'a [f32]) -> Self {
        Self {
            sigma,
            provenance: SigmaProvenance::LeaveOneOut,
        }
    }

    /// `σ̂` known from the design (e.g. [`variance_floor`] of the group size).
    #[inline]
    #[must_use]
    pub fn design(sigma: &'a [f32]) -> Self {
        Self {
            sigma,
            provenance: SigmaProvenance::Design,
        }
    }

    /// The per-group noise scales.
    #[inline]
    #[must_use]
    pub fn sigma(&self) -> &'a [f32] {
        self.sigma
    }

    /// The declared source.
    #[inline]
    #[must_use]
    pub fn provenance(&self) -> SigmaProvenance {
        self.provenance
    }
}

/// Which estimand a weighted readout carries.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimand {
    /// Every non-empty group carried the same weight — the readout IS the
    /// unweighted pooled Beta, bit-identical to [`best_belief_score`].
    Unweighted,
    /// The `w`-weighted estimand `Σ w_g n_g p_g / Σ w_g n_g`. Equal to the
    /// unweighted rate only when the groups' true rates are homogeneous.
    NoiseWeighted,
}

/// The readout of [`best_belief_score_weighted`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedBelief {
    /// ε-quantile of `Beta(1 + successes_w, 1 + failures_w)`.
    pub score: f32,
    /// Weighted success evidence `Σ w̃_g k_g` (mean-one weights).
    pub successes_w: f32,
    /// Weighted failure evidence `Σ w̃_g (n_g − k_g)`.
    pub failures_w: f32,
    /// Raw observation count `Σ n_g` (saturating at `u32::MAX`).
    pub n_raw: u32,
    /// Kish effective sample size `(Σ w_g n_g)² / Σ w_g² n_g` — `n_raw` under
    /// uniform weights, smaller otherwise. Disclosed on every readout: the
    /// Beta is built at `n_raw` total evidence, so a reader comparing interval
    /// widths needs the information the weighting spent.
    pub n_eff: f32,
    /// Which estimand the readout carries.
    pub estimand: Estimand,
    /// The declared `σ̂` source.
    pub provenance: SigmaProvenance,
}

impl WeightedBelief {
    /// The weighted point rate `successes_w / (successes_w + failures_w)` —
    /// no prior pseudocount. `0.0` for zero evidence.
    #[inline]
    #[must_use]
    pub fn rate(&self) -> f32 {
        let total = self.successes_w + self.failures_w;
        match total > 0.0 {
            true => self.successes_w / total,
            false => 0.0,
        }
    }
}

/// Conservative selection score over GROUPED Beta-Bernoulli evidence, each
/// group weighted by `1 / max(σ̂_g, floor)` from an independent source.
///
/// - `groups[g] = (k_g, n_g − k_g)` — successes and failures, the
///   [`best_belief_score`] convention.
/// - `sigma.sigma()[g]` — group `g`'s noise scale; NaN reads as the floor.
/// - `floor` — the `ε` in `max(σ̂, ε)`, typically [`variance_floor`]; must be
///   positive and finite.
///
/// The raw weights are rescaled to mean one PER OBSERVATION
/// (`Σ w̃_g n_g = Σ n_g`), so the posterior carries the same total evidence
/// as the unweighted pool and only its allocation across groups moves:
/// `α = 1 + Σ w̃_g k_g`, `β = 1 + Σ w̃_g (n_g − k_g)`, and the score is the
/// ε-quantile of `Beta(α, β)` through the shared Newton/Lentz solver.
///
/// **G3:** when every non-empty group carries the same weight (equal `σ̂`, or
/// all floored) the readout takes the integer path and is bit-identical to
/// `best_belief_score(Σk, Σ(n−k), ε)` — LUT included.
///
/// # Panics
///
/// If `groups` and `sigma` differ in length, or `floor` is not positive and
/// finite (caller bugs — a zero floor makes a zero-noise group's weight
/// infinite).
#[must_use]
pub fn best_belief_score_weighted(
    groups: &[(u32, u32)],
    sigma: ExogenousSigma<'_>,
    floor: f32,
    epsilon: f32,
) -> WeightedBelief {
    let sig = sigma.sigma();
    assert_eq!(
        groups.len(),
        sig.len(),
        "best_belief_score_weighted: groups/sigma length mismatch (caller bug)"
    );
    assert!(
        floor.is_finite() && floor > 0.0,
        "best_belief_score_weighted: floor must be positive and finite, got {floor}"
    );

    // One pass, f64 accumulators: the weighted sums feed a ratio and a Kish
    // square, where f32 accumulation over thousands of observations would
    // cost visible digits.
    let mut s_tot: u64 = 0;
    let mut f_tot: u64 = 0;
    let mut w1 = 0.0_f64; // Σ w n
    let mut w2 = 0.0_f64; // Σ w² n
    let mut ws = 0.0_f64; // Σ w k
    let mut wf = 0.0_f64; // Σ w (n − k)
    let mut first_w: Option<u32> = None;
    let mut uniform = true;
    for (&(s, f), &sg) in groups.iter().zip(sig.iter()) {
        let n = s as u64 + f as u64;
        if n == 0 {
            continue;
        }
        let w = 1.0 / sg.max(floor);
        match first_w {
            None => first_w = Some(w.to_bits()),
            Some(bits) => uniform &= bits == w.to_bits(),
        }
        let wd = w as f64;
        s_tot += s as u64;
        f_tot += f as u64;
        w1 += wd * n as f64;
        w2 += wd * wd * n as f64;
        ws += wd * s as f64;
        wf += wd * f as f64;
    }
    let n_tot = s_tot + f_tot;
    let n_raw = n_tot.min(u32::MAX as u64) as u32;

    match uniform {
        // Equal weights (or no evidence): the weighting is a no-op, so the
        // readout IS the unweighted pool — route through the integer path so
        // the result is bit-identical (the G3 contract), LUT included.
        true => {
            let s = s_tot.min(u32::MAX as u64) as u32;
            let f = f_tot.min(u32::MAX as u64) as u32;
            WeightedBelief {
                score: best_belief_score(s, f, epsilon),
                successes_w: s as f32,
                failures_w: f as f32,
                n_raw,
                n_eff: n_raw as f32,
                estimand: Estimand::Unweighted,
                provenance: sigma.provenance(),
            }
        }
        false => {
            let c = n_tot as f64 / w1; // mean-one per observation
            let successes_w = (c * ws) as f32;
            let failures_w = (c * wf) as f32;
            WeightedBelief {
                score: beta_quantile(1.0 + successes_w, 1.0 + failures_w, epsilon),
                successes_w,
                failures_w,
                n_raw,
                n_eff: (w1 * w1 / w2) as f32,
                estimand: Estimand::NoiseWeighted,
                provenance: sigma.provenance(),
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T4 — noise-scaled rating K
// ──────────────────────────────────────────────────────────────────────────

/// `K_eff = K · ε / max(σ̂, ε)` — the step size of a match whose outcome noise
/// is `σ̂`, relative to a reference noise `ε` at which the step is the full
/// `K`. Noisier outcomes move ratings less.
///
/// **`σ̂` from the group's history, a prior epoch or the design — never from
/// the outcome being rated** (the endogeneity law: a `σ̂` derived from the
/// current score biases the asymptote; pinned by this module's negative
/// control). NaN reads as the floor.
///
/// **G3:** for `σ̂ ≤ ε` the factor is `ε/ε`, which IEEE division makes exactly
/// `1.0`, so `K_eff == K` bit-for-bit and the update is the fixed-K update.
#[inline]
#[must_use]
pub fn noise_scaled_k(k: f64, sigma: f64, floor: f64) -> f64 {
    k * (floor / sigma.max(floor))
}

/// f32 twin of [`noise_scaled_k`].
#[inline]
#[must_use]
pub fn noise_scaled_k_f32(k: f32, sigma: f32, floor: f32) -> f32 {
    k * (floor / sigma.max(floor))
}

/// [`rating::update_scored`] at the noise-scaled step [`noise_scaled_k`].
/// Bit-identical to `update_scored(a, b, score_a, k, scale)` for `σ̂ ≤ ε`.
#[inline]
#[must_use]
pub fn update_scored_noise_scaled(
    a: f64,
    b: f64,
    score_a: f64,
    k: f64,
    scale: f64,
    sigma: f64,
    floor: f64,
) -> (f64, f64) {
    rating::update_scored(a, b, score_a, noise_scaled_k(k, sigma, floor), scale)
}

/// [`rating::update_f32`] at the noise-scaled step [`noise_scaled_k_f32`].
/// Bit-identical to `update_f32(a, b, a_won, k, scale)` for `σ̂ ≤ ε`.
#[inline]
#[must_use]
pub fn update_f32_noise_scaled(
    a: f32,
    b: f32,
    a_won: bool,
    k: f32,
    scale: f32,
    sigma: f32,
    floor: f32,
) -> (f32, f32) {
    rating::update_f32(a, b, a_won, noise_scaled_k_f32(k, sigma, floor), scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::best_belief::best_belief_score_cf;

    /// Binomial(n, q) by n Bernoulli draws — test fixture only.
    fn binomial(rng: &mut fastrand::Rng, n: u32, q: f32) -> u32 {
        (0..n).filter(|_| rng.f32() < q).count() as u32
    }

    /// Paired mean and 95% lower bound of a per-trial difference series.
    fn paired_lb95(diffs: &[f64]) -> (f64, f64) {
        let n = diffs.len() as f64;
        let mean = diffs.iter().sum::<f64>() / n;
        let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1.0);
        (mean, mean - 1.96 * (var / n).sqrt())
    }

    // ── T1: variance_floor ──

    #[test]
    fn t1_two_point_extremal_attains_the_floor() {
        // Analytic: {0, Δ} at p = ½ has σ = Δ/2, so the n-mean SE is Δ/(2√n).
        for &(delta, n) in &[(1.0_f32, 1_u32), (1.0, 8), (2.5, 50), (0.3, 400)] {
            let se = (0.25_f32).sqrt() * delta / (n as f32).sqrt();
            let fl = variance_floor(delta, n);
            assert!(
                (se - fl).abs() <= 1e-6 * fl.max(1.0),
                "Δ={delta} n={n}: {se} vs {fl}"
            );
        }
        // Empirical: the SD of simulated 16-draw means of the extremal law.
        let mut rng = fastrand::Rng::with_seed(0x0913_0001);
        let (delta, n, trials) = (2.0_f32, 16_u32, 40_000);
        let means: Vec<f64> = (0..trials)
            .map(|_| {
                let s: f64 = (0..n)
                    .map(|_| if rng.bool() { delta as f64 } else { 0.0 })
                    .sum();
                s / n as f64
            })
            .collect();
        let m = means.iter().sum::<f64>() / trials as f64;
        let sd = (means.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / trials as f64).sqrt();
        let fl = variance_floor(delta, n) as f64;
        assert!(
            (sd - fl).abs() < 0.02 * fl,
            "empirical SE {sd} vs floor {fl}"
        );
    }

    #[test]
    fn t1_every_bounded_law_sits_at_or_below_the_floor() {
        // Random discrete laws on [0, Δ]: σ/√n never exceeds Δ/(2√n).
        let mut rng = fastrand::Rng::with_seed(0x0913_0002);
        for _ in 0..2000 {
            let delta = 0.1 + 5.0 * rng.f32();
            let n = 1 + rng.u32(0..200);
            let mut support = [0.0_f32; 5];
            let mut prob = [0.0_f32; 5];
            let mut z = 0.0;
            for i in 0..5 {
                support[i] = delta * rng.f32();
                prob[i] = rng.f32() + 1e-3;
                z += prob[i];
            }
            let mean: f32 = (0..5).map(|i| prob[i] / z * support[i]).sum();
            let var: f32 = (0..5)
                .map(|i| prob[i] / z * (support[i] - mean).powi(2))
                .sum();
            let se = var.sqrt() / (n as f32).sqrt();
            assert!(
                se <= variance_floor(delta, n) * (1.0 + 1e-5),
                "law beat Popoviciu"
            );
        }
    }

    #[test]
    fn t1_monotone_in_n_and_n0_reads_as_one() {
        let mut prev = f32::INFINITY;
        for n in 1..=1000 {
            let v = variance_floor(1.0, n);
            assert!(v < prev, "not strictly decreasing at n={n}");
            prev = v;
        }
        assert_eq!(
            variance_floor(1.0, 0).to_bits(),
            variance_floor(1.0, 1).to_bits()
        );
        assert_eq!(
            variance_floor(-3.0, 9).to_bits(),
            variance_floor(3.0, 9).to_bits()
        );
        assert_eq!(variance_floor(1.0, 1), 0.5);
    }

    // ── T2: filter_bias_bound ──

    #[test]
    fn t2_bound_is_tight_at_the_adversarial_two_point() {
        // R = +Γ on C, −Γ on ¬C: E[R|C] = Γ, E[R] = Γ(1 − 2q), bias = 2Γq.
        for &gamma in &[0.5_f32, 1.0, 3.0] {
            for i in 0..=20 {
                let q = i as f32 / 20.0;
                let bias = gamma - gamma * (1.0 - 2.0 * q);
                let bound = filter_bias_bound(gamma, q);
                assert!((bias - bound).abs() <= 1e-6 * gamma, "Γ={gamma} q={q}");
            }
        }
    }

    #[test]
    fn t2_bound_dominates_every_bounded_filter() {
        let mut rng = fastrand::Rng::with_seed(0x0913_0003);
        for _ in 0..20_000 {
            let gamma = 0.1 + 4.0 * rng.f32();
            let q = rng.f32();
            let m_c = gamma * (2.0 * rng.f32() - 1.0);
            let m_nc = gamma * (2.0 * rng.f32() - 1.0);
            let e_r = (1.0 - q) * m_c + q * m_nc;
            let bias = (m_c - e_r).abs();
            assert!(bias <= filter_bias_bound(gamma, q) * (1.0 + 1e-5) + 1e-6);
        }
        assert_eq!(filter_bias_bound(1.0, -0.5), 0.0);
        assert_eq!(filter_bias_bound(1.0, 1.5), 2.0);
        assert_eq!(filter_bias_bound(-2.0, 0.25), 1.0);
    }

    // ── T3: best_belief_score_weighted ──

    #[test]
    fn t3_fractional_solver_is_bit_identical_on_integer_shapes() {
        // The extraction contract: beta_quantile(1+S, 1+F) IS the integer
        // closed form, so the LUT and every integer readout are unchanged.
        for s in [1_u32, 3, 17, 31, 40, 120] {
            for f in [0_u32, 2, 9, 31, 77] {
                for eps in [0.01_f32, 0.05, 0.2, 0.5] {
                    let a = beta_quantile(1.0 + s as f32, 1.0 + f as f32, eps);
                    let b = best_belief_score_cf(s, f, eps);
                    assert_eq!(a.to_bits(), b.to_bits(), "S={s} F={f} ε={eps}");
                }
            }
        }
    }

    #[test]
    fn g3_uniform_weights_are_bit_identical_to_the_unweighted_score() {
        let groups = [(3_u32, 5_u32), (7, 1), (0, 4), (2, 2), (0, 0)];
        let (s, f) = (12_u32, 12_u32);
        for eps in [0.01_f32, 0.05, 0.1, 0.25, 0.5, 0.07, 0.0, 1.0] {
            let want = best_belief_score(s, f, eps);
            // Equal σ̂ above the floor.
            let eq = [0.3_f32; 5];
            let r = best_belief_score_weighted(&groups, ExogenousSigma::design(&eq), 0.05, eps);
            assert_eq!(r.score.to_bits(), want.to_bits(), "equal σ̂, ε={eps}");
            assert_eq!(r.estimand, Estimand::Unweighted);
            assert_eq!(r.n_eff, 24.0);
            // All floored (distinct σ̂, every one below ε), plus a NaN.
            let fl = [0.001_f32, 0.02, f32::NAN, 0.0, 0.04];
            let r =
                best_belief_score_weighted(&groups, ExogenousSigma::prior_epoch(&fl), 0.05, eps);
            assert_eq!(r.score.to_bits(), want.to_bits(), "all floored, ε={eps}");
            assert_eq!(r.provenance, SigmaProvenance::PriorEpoch);
        }
        // An empty group's weight is irrelevant — it does not break uniformity.
        let odd = [0.3_f32, 0.3, 0.3, 0.3, 9.0];
        let r = best_belief_score_weighted(&groups, ExogenousSigma::design(&odd), 0.05, 0.05);
        assert_eq!(r.estimand, Estimand::Unweighted);
        // No evidence at all → the uniform prior's ε-quantile.
        let r = best_belief_score_weighted(&[], ExogenousSigma::design(&[]), 0.05, 0.05);
        assert_eq!(r.score.to_bits(), best_belief_score(0, 0, 0.05).to_bits());
    }

    #[test]
    fn g1_mean_one_normalisation_and_kish_disclosure() {
        let groups = [(4_u32, 6_u32), (9, 1), (2, 18), (5, 5)];
        let sigma = [0.1_f32, 0.4, 0.2, 0.05];
        let floor = 0.08;
        let r = best_belief_score_weighted(&groups, ExogenousSigma::design(&sigma), floor, 0.05);
        assert_eq!(r.estimand, Estimand::NoiseWeighted);
        assert_eq!(r.n_raw, 50);
        // Mean-one per observation: total weighted evidence == raw count.
        assert!((r.successes_w + r.failures_w - 50.0).abs() < 1e-4);
        // Kish by hand, and strictly below n_raw under unequal weights.
        let (mut a, mut b, mut ws, mut n_tot) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for (&(s, f), &sg) in groups.iter().zip(sigma.iter()) {
            let w = 1.0 / sg.max(floor) as f64;
            let n = (s + f) as f64;
            a += w * n;
            b += w * w * n;
            ws += w * s as f64;
            n_tot += n;
        }
        assert!((r.n_eff as f64 - a * a / b).abs() < 1e-3);
        assert!(r.n_eff < 50.0);
        assert!((r.rate() as f64 - ws / a).abs() < 1e-6);
        assert!((r.successes_w as f64 - ws * n_tot / a).abs() < 1e-3);
        // The score is the Beta quantile at the fractional shapes.
        let want = beta_quantile(1.0 + r.successes_w, 1.0 + r.failures_w, 0.05);
        assert_eq!(r.score.to_bits(), want.to_bits());
        // Lower ε is more conservative.
        let lo = best_belief_score_weighted(&groups, ExogenousSigma::design(&sigma), floor, 0.01);
        assert!(lo.score < r.score);
    }

    /// GOAT (a) — the NEGATIVE CONTROL: the counts' own plug-in σ̂ pulls the
    /// pooled rate toward the extreme; the same σ̂ taken from an independent
    /// prior epoch does not. Proves the endogeneity law bites.
    #[test]
    fn goat_a_plug_in_sigma_biases_toward_the_extreme() {
        let (p, groups_n, n, trials) = (0.15_f32, 40_usize, 8_u32, 3000);
        let floor = variance_floor(1.0, n);
        let mut rng = fastrand::Rng::with_seed(0x0913_000a);
        let mut groups = vec![(0_u32, 0_u32); groups_n];
        let mut plug = vec![0.0_f32; groups_n];
        let mut prior = vec![0.0_f32; groups_n];
        let (mut b_plug, mut b_prior, mut b_raw) = (0.0_f64, 0.0_f64, 0.0_f64);
        for _ in 0..trials {
            for g in 0..groups_n {
                let k = binomial(&mut rng, n, p);
                groups[g] = (k, n - k);
                let ph = k as f32 / n as f32;
                plug[g] = (ph * (1.0 - ph)).sqrt();
                let k0 = binomial(&mut rng, n, p); // independent epoch
                let ph0 = k0 as f32 / n as f32;
                prior[g] = (ph0 * (1.0 - ph0)).sqrt();
            }
            // The plug-in arm reaches the API through a mislabelled tag —
            // exactly the realistic mistake the provenance type cannot stop.
            let rp =
                best_belief_score_weighted(&groups, ExogenousSigma::design(&plug), floor, 0.05);
            let ri = best_belief_score_weighted(
                &groups,
                ExogenousSigma::prior_epoch(&prior),
                floor,
                0.05,
            );
            let s: u32 = groups.iter().map(|g| g.0).sum();
            b_plug += (rp.rate() - p) as f64;
            b_prior += (ri.rate() - p) as f64;
            b_raw += (s as f32 / (groups_n as u32 * n) as f32 - p) as f64;
        }
        let (b_plug, b_prior, b_raw) = (
            b_plug / trials as f64,
            b_prior / trials as f64,
            b_raw / trials as f64,
        );
        eprintln!(
            "GOAT(a) bias: plug-in {b_plug:+.5}  prior-epoch {b_prior:+.5}  unweighted {b_raw:+.5}"
        );
        assert!(b_plug < -0.01, "plug-in σ̂ must pull toward 0: {b_plug}");
        assert!(
            b_prior.abs() < 0.003,
            "independent σ̂ must be unbiased: {b_prior}"
        );
        assert!(b_raw.abs() < 0.003, "unweighted must be unbiased: {b_raw}");
    }

    /// Draw one arm's grouped evidence under heteroscedastic group noise:
    /// group rate `q = p + σ_g·√3·U(−1, 1)` (zero-mean, unclamped for the
    /// fixture's ranges), so `E[q] = p` and the same-rate condition holds.
    fn draw_arm(rng: &mut fastrand::Rng, p: f32, noise: &[f32], n: u32, groups: &mut [(u32, u32)]) {
        for (g, &sg) in groups.iter_mut().zip(noise.iter()) {
            let q = p + sg * 3.0_f32.sqrt() * (2.0 * rng.f32() - 1.0);
            let k = binomial(rng, n, q.clamp(0.0, 1.0));
            *g = (k, n - k);
        }
    }

    /// GOAT (b) — best-arm identification: weighting by a DESIGN-level σ̂
    /// (binomial worst case ⊕ known group noise) beats the unweighted pool,
    /// paired over the same draws, LB95 > 0. Also measures the LCB's
    /// parameter coverage for the record (Report-the-Floor disclosure).
    #[test]
    fn goat_b_independent_weights_beat_unweighted_at_best_arm_identification() {
        const ARMS: usize = 4;
        const G: usize = 12;
        let p = [0.40_f32, 0.43, 0.46, 0.50];
        let (n, trials, eps) = (20_u32, 4000, 0.05_f32);
        let floor = variance_floor(1.0, n);
        let mut rng = fastrand::Rng::with_seed(0x0913_000b);
        let mut groups = [[(0_u32, 0_u32); G]; ARMS];
        let mut noise = [[0.0_f32; G]; ARMS];
        let mut design = [[0.0_f32; G]; ARMS];
        let mut diffs = Vec::with_capacity(trials);
        let (mut hit_w, mut hit_u) = (0_u32, 0_u32);
        let (mut cov_w, mut cov_u, mut cov_k, mut cov_n) = (0_u32, 0_u32, 0_u32, 0_u32);
        for _ in 0..trials {
            let (mut best_w, mut best_u) = ((0_usize, f32::MIN), (0_usize, f32::MIN));
            for a in 0..ARMS {
                for g in 0..G {
                    noise[a][g] = if rng.bool() { 0.02 } else { 0.20 };
                    let fl = variance_floor(1.0, n);
                    design[a][g] = (fl * fl + noise[a][g] * noise[a][g]).sqrt();
                }
                draw_arm(&mut rng, p[a], &noise[a], n, &mut groups[a]);
                let w = best_belief_score_weighted(
                    &groups[a],
                    ExogenousSigma::design(&design[a]),
                    floor,
                    eps,
                );
                let (s, f) = groups[a]
                    .iter()
                    .fold((0, 0), |(s, f), g| (s + g.0, f + g.1));
                let u = best_belief_score(s, f, eps);
                if w.score > best_w.1 {
                    best_w = (a, w.score);
                }
                if u > best_u.1 {
                    best_u = (a, u);
                }
                cov_w += (p[a] >= w.score) as u32;
                // Kish-scaled variant: the same rate at n_eff total evidence.
                let kish = beta_quantile(
                    1.0 + w.rate() * w.n_eff,
                    1.0 + (1.0 - w.rate()) * w.n_eff,
                    eps,
                );
                cov_k += (p[a] >= kish) as u32;
                cov_u += (p[a] >= u) as u32;
                cov_n += 1;
            }
            let cw = (best_w.0 == ARMS - 1) as u32;
            let cu = (best_u.0 == ARMS - 1) as u32;
            hit_w += cw;
            hit_u += cu;
            diffs.push(cw as f64 - cu as f64);
        }
        let (mean, lb) = paired_lb95(&diffs);
        eprintln!(
            "GOAT(b) P(correct): weighted {:.4}  unweighted {:.4}  paired Δ {mean:+.4} LB95 {lb:+.4}",
            hit_w as f64 / trials as f64,
            hit_u as f64 / trials as f64,
        );
        eprintln!(
            "GOAT(b) LCB_0.05 parameter coverage (nominal 0.95): weighted {:.4}  \
             weighted@Kish {:.4}  unweighted {:.4}",
            cov_w as f64 / cov_n as f64,
            cov_k as f64 / cov_n as f64,
            cov_u as f64 / cov_n as f64,
        );
        assert!(
            lb > 0.0,
            "weighted must beat unweighted at paired LB95: Δ={mean} LB={lb}"
        );
    }

    #[test]
    fn g2_weighted_score_latency() {
        use std::hint::black_box;
        use std::time::Instant;
        let groups: [(u32, u32); 16] =
            core::array::from_fn(|g| (3 + g as u32 % 5, 11 - g as u32 % 7));
        let sigma: [f32; 16] = core::array::from_fn(|g| 0.05 + 0.03 * g as f32);
        let iters = 20_000_u32;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let mut acc = 0.0_f32;
            for i in 0..iters {
                let r = best_belief_score_weighted(
                    black_box(&groups),
                    ExogenousSigma::design(black_box(&sigma)),
                    black_box(0.08),
                    black_box(0.05 + (i & 1) as f32 * 1e-9),
                );
                acc += r.score;
            }
            black_box(acc);
            best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
        }
        eprintln!("G2 best_belief_score_weighted @ G=16: {best:.1} ns/call (best of 5)");
        assert!(
            best > 1.0,
            "loud zero: the timed loop was optimised away ({best} ns)"
        );
        #[cfg(not(debug_assertions))]
        assert!(best < 5_000.0, "G2 bar 5 µs: {best} ns");
    }

    // ── T4: noise-scaled rating K ──

    #[test]
    fn g3_noise_scaled_update_is_bit_identical_at_the_floor() {
        for &(a, b, s) in &[
            (1200.0_f64, 1000.0, 1.0),
            (900.0, 1300.0, 0.5),
            (1000.0, 1000.0, 0.0),
        ] {
            for &sigma in &[0.0_f64, 0.01, 0.0625, f64::NAN] {
                let want = rating::update_scored(a, b, s, 32.0, 400.0);
                let got = update_scored_noise_scaled(a, b, s, 32.0, 400.0, sigma, 0.0625);
                assert_eq!(want.0.to_bits(), got.0.to_bits());
                assert_eq!(want.1.to_bits(), got.1.to_bits());
            }
        }
        for &(a, b, won) in &[(1200.0_f32, 1000.0, true), (900.0, 1300.0, false)] {
            for &sigma in &[0.0_f32, 0.05, 0.1, f32::NAN] {
                let want = rating::update_f32(a, b, won, 32.0, 400.0);
                let got = update_f32_noise_scaled(a, b, won, 32.0, 400.0, sigma, 0.1);
                assert_eq!(want.0.to_bits(), got.0.to_bits());
                assert_eq!(want.1.to_bits(), got.1.to_bits());
            }
        }
        // Above the floor the step shrinks by exactly ε/σ̂.
        assert!((noise_scaled_k(32.0, 0.5, 0.0625) - 4.0).abs() < 1e-12);
        assert!((noise_scaled_k_f32(32.0, 0.25, 0.0625) - 8.0).abs() < 1e-6);
    }

    /// One scored match at true win probability `p_true` over `r` rounds.
    fn match_score(rng: &mut fastrand::Rng, p_true: f64, r: u32) -> f64 {
        (0..r).filter(|_| rng.f64() < p_true).count() as f64 / r as f64
    }

    /// GOAT — MSE dominance at equal update count. Mixed-noise matches (64
    /// rounds or 1 round, design σ̂ = `variance_floor(1, r)`), true gap 200.
    /// Adaptive K beats BOTH fixed K = 32 (its own maximum) and fixed K =
    /// its own MEAN step (so the win is the weighting, not a smaller step).
    #[test]
    fn goat_t4_adaptive_k_dominates_fixed_k_on_mixed_noise() {
        let (gap, scale, k) = (200.0_f64, 400.0_f64, 32.0_f64);
        let p_true = rating::expected(gap, 0.0, scale);
        let floor = variance_floor(1.0, 64) as f64;
        let rounds = [64_u32, 1];
        let k_mean = (rounds
            .iter()
            .map(|&r| noise_scaled_k(k, variance_floor(1.0, r) as f64, floor))
            .sum::<f64>())
            / rounds.len() as f64;
        let (seeds, updates, burn) = (400_u64, 600_usize, 200_usize);
        let (mut d_max, mut d_mean) = (Vec::new(), Vec::new());
        let (mut m_ad, mut m_fx, mut m_fm) = (0.0, 0.0, 0.0);
        for seed in 0..seeds {
            let mut rng = fastrand::Rng::with_seed(0x0913_0040 + seed);
            let (mut ad, mut fx, mut fm) = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0));
            let (mut se_ad, mut se_fx, mut se_fm) = (0.0, 0.0, 0.0);
            for t in 0..updates {
                let r = rounds[rng.usize(0..rounds.len())];
                let s = match_score(&mut rng, p_true, r);
                let sg = variance_floor(1.0, r) as f64;
                ad = update_scored_noise_scaled(ad.0, ad.1, s, k, scale, sg, floor);
                fx = rating::update_scored(fx.0, fx.1, s, k, scale);
                fm = rating::update_scored(fm.0, fm.1, s, k_mean, scale);
                if t >= burn {
                    se_ad += ((ad.0 - ad.1) - gap).powi(2);
                    se_fx += ((fx.0 - fx.1) - gap).powi(2);
                    se_fm += ((fm.0 - fm.1) - gap).powi(2);
                }
            }
            let w = (updates - burn) as f64;
            let (se_ad, se_fx, se_fm) = (se_ad / w, se_fx / w, se_fm / w);
            m_ad += se_ad;
            m_fx += se_fx;
            m_fm += se_fm;
            d_max.push(se_fx - se_ad);
            d_mean.push(se_fm - se_ad);
        }
        let n = seeds as f64;
        let (mx, lb_max) = paired_lb95(&d_max);
        let (mm, lb_mean) = paired_lb95(&d_mean);
        eprintln!(
            "GOAT(T4) time-avg MSE: adaptive {:.1}  fixed K=32 {:.1}  fixed K={k_mean:.1} {:.1}  \
             Δmax {mx:+.1} LB95 {lb_max:+.1}  Δmean {mm:+.1} LB95 {lb_mean:+.1}",
            m_ad / n,
            m_fx / n,
            m_fm / n,
        );
        assert!(
            lb_max > 0.0 && lb_mean > 0.0,
            "adaptive K must dominate both fixed arms"
        );
    }

    /// T4 negative control + unbiasedness pin: 4-round matches, gap 200.
    /// Design σ̂ (constant per match type) leaves the asymptote at the true
    /// gap; the match's OWN plug-in σ̂ `√(s(1−s)/r)` gives a clean sweep
    /// (s = 1) the full step and inflates the gap.
    #[test]
    fn goat_t4_plug_in_sigma_shifts_the_asymptote() {
        let (gap, scale, k, r) = (200.0_f64, 400.0_f64, 32.0_f64, 4_u32);
        let p_true = rating::expected(gap, 0.0, scale);
        let floor = variance_floor(1.0, 64) as f64;
        let (seeds, updates, burn) = (200_u64, 3000_usize, 1000_usize);
        let (mut g_design, mut g_plug) = (0.0, 0.0);
        for seed in 0..seeds {
            let mut rng = fastrand::Rng::with_seed(0x0913_004c + seed);
            let (mut de, mut pl) = ((0.0, 0.0), (0.0, 0.0));
            for t in 0..updates {
                let s = match_score(&mut rng, p_true, r);
                let sg_design = variance_floor(1.0, r) as f64;
                let sg_plug = (s * (1.0 - s) / r as f64).sqrt();
                de = update_scored_noise_scaled(de.0, de.1, s, k, scale, sg_design, floor);
                pl = update_scored_noise_scaled(pl.0, pl.1, s, k, scale, sg_plug, floor);
                if t >= burn {
                    g_design += de.0 - de.1;
                    g_plug += pl.0 - pl.1;
                }
            }
        }
        let w = (seeds as usize * (updates - burn)) as f64;
        let (g_design, g_plug) = (g_design / w, g_plug / w);
        eprintln!(
            "GOAT(T4-control) asymptote: design σ̂ {g_design:.1}  plug-in σ̂ {g_plug:.1}  (true {gap})"
        );
        assert!(
            (g_design - gap).abs() < 15.0,
            "design σ̂ must keep the asymptote: {g_design}"
        );
        assert!(
            g_plug > gap + 50.0,
            "plug-in σ̂ must inflate the gap: {g_plug}"
        );
    }

    // ── G4: allocation-free ──

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn goat_g4_alloc_free() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};
        let groups = [(4_u32, 6_u32), (9, 1), (2, 18), (5, 5)];
        let sigma = [0.1_f32, 0.4, 0.2, 0.05];
        reset_alloc_stats();
        let mut acc = variance_floor(1.0, 20) + filter_bias_bound(1.0, 0.2);
        acc +=
            best_belief_score_weighted(&groups, ExogenousSigma::design(&sigma), 0.08, 0.05).score;
        acc += best_belief_score_weighted(&groups, ExogenousSigma::design(&[0.3; 4]), 0.08, 0.05)
            .score;
        let (a, b) = update_scored_noise_scaled(1000.0, 1000.0, 0.5, 32.0, 400.0, 0.2, 0.0625);
        let (c, d) = update_f32_noise_scaled(1000.0, 1000.0, true, 32.0, 400.0, 0.2, 0.0625);
        std::hint::black_box((acc, a, b, c, d));
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "hot paths must be allocation-free, got {count} allocs"
        );
    }
}
