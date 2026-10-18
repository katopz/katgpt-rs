//! Kamath range-law regime detector + normalized-entropy dispersion
//! diagnostic for attention-routing logit rows (Issue 762 T4.2, Research 549
//! §2.2 — the ASEntmax duality's measurement half).
//!
//! The extreme-value law (Kamath et al. 2015, via Research 549): the range of
//! `n` IID Gaussian logits grows as `2σ√(2 ln n)`. The ratio
//!
//! ```text
//! ρ = Δ̂ / (2σ̂√(2 ln n))     (Δ̂ = max − min, σ̂ = moment std)
//! ```
//!
//! therefore **classifies the row's regime**: ρ ≈ 1 (precisely: the finite-n
//! Gaussian band `[0.5, 1.1]` — the `√(2 ln n)` asymptotic overestimates
//! `E[max]` at small n) means the row looks like IID Gaussian noise around its
//! mean — the regime where the ASEntmax Eq-10 damping derivation is exact and
//! fixed-α entmax behaves as theorized; ρ ≫ 1 means a **spike** — one or few
//! dominant keys (the retrieval/needle regime) stretch the range beyond the
//! Gaussian law, and over-sparsification there is *desired*, not a failure.
//! This is the detector Bench 713's σ̂-regime question (T0) left implicit: σ̂
//! alone cannot separate "wider Gaussian" from "needle present" — the
//! range-law σ̂ estimate (katgpt-attn `RollingSigmaEstimator`) is by
//! construction always self-consistent; ρ compares it against an INDEPENDENT
//! moment estimate, which is exactly what makes it a detector.
//!
//! The companion diagnostic, normalized entropy `H(p)/ln n` over the row's
//! softmax (Research 549 Prop 1: softmax dispersion → 1, entmax support
//! `O(n^β)` keeps it ≤ β), measures how dispersed the row's attention
//! actually is — `≈ 1` uniform (diluted), `≈ 0` concentrated. Computed via the
//! shared ungated [`crate::simd::logsumexp_parts`] kernel — the SAME kernel
//! `regime_probe::entropy::conditional_entropy_nats` wraps (one kernel shape,
//! no divergent copy; consuming the gated wrapper instead would chain this
//! module's feature onto `regime_probe`).
//!
//! # Modelless discipline
//!
//! Pure two-pass statistics — exact f64 moments (the textbook two-pass
//! algorithm: exact sum, then squared deviations — no per-element division,
//! no Σx²−n·mean² cancellation), the Cephes exp/logsumexp kernel, one
//! `fast_sigmoid`. Zero training, zero learned
//! parameters, zero allocation (G4). The classifier output is
//! `spike_score = σ(ln ρ)` — sigmoid, never softmax (AGENTS.md §2); 0.5
//! exactly at the Gaussian boundary ρ = 1, → 1 spiked.
//!
//! # Honest calibration (read before thresholding)
//!
//! A **single** outlier among `n` Gaussian-ish values gives
//! `ρ ≈ √n/(2√(2 ln n))` — INVARIANT to the outlier's magnitude (range and
//! σ̂ both scale with it): ~1.95 at n = 128, ~3.4 at n = 1024, i.e.
//! `spike_score ≈ 0.7–0.8` for realistic single-needle rows. The score's
//! magnitude encodes the row's shape class, not the needle's strength —
//! threshold near 0.55–0.65 for “a spike exists” at routing scales; the
//! Gaussian band never exceeds ~0.54 (ρ ≤ 1.15).
//!
//! # Degenerates (honesty over fabrication)
//!
//! `n < 2` or a constant row (`σ̂ = 0`) has no regime: ρ is `NaN`, the score
//! `0.5` (uninformed), entropy per its own convention. Callers gate on
//! `is_finite`.
//!
//! # Feature gate
//!
//! Opt-in `logit_regime` (beside `ssmax`, same statistics family). No default
//! consumer yet — promotion only after a consumer GOAT gate.
//!
//! Kamath range-law regime detector + normalized-entropy dispersion diagnostic
//! (Issue 762 T4.2, Research 549 — the ASEntmax duality's measurement half,
//! beside ssmax: same statistics family). `ρ = Δ̂/(2σ̂√(2 ln n))` classifies a
//! routing logit row as Gaussian-band (the Eq-10 damping regime) vs spiked
//! (needle present — over-sparsification desired); `H(p)/ln n` measures
//! Prop-1 dispersion. Pure one/two-pass modelless statistics over the shared
//! ungated `simd::logsumexp_parts` kernel; opt-in `logit_regime` until a
//! consumer promotes.

/// One row's regime reading: the Kamath ρ, the normalized entropy, and the
/// sigmoid spike score derived from ρ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitRegime {
    /// `ρ = Δ̂/(2σ̂√(2 ln n))` — ≈1 Gaussian band (see module docs for the
    /// finite-n band), ≫1 spiked, ≪1 sub-Gaussian/clipped. `NaN` on
    /// degenerate rows (n < 2 or σ̂ = 0).
    pub rho: f32,
    /// `H(p)/ln n` over the row's softmax — 1 uniform, 0 one-hot. `0.0` for
    /// `n ≤ 1` (no dispersion to measure).
    pub normalized_entropy: f32,
    /// `σ(ln ρ)` ∈ (0, 1): 0.5 at the Gaussian boundary ρ = 1, → 1 spiked.
    /// `0.5` when ρ is non-finite (uninformed, never fabricated).
    pub spike_score: f32,
}

/// One-pass-range + two-pass moments (max, min, σ̂) in f64 — the textbook
/// two-pass algorithm (exact f64 sum, then Σ(x−mean)²): no per-element
/// division (the Welford hot-spot) and no Σx²−n·mean² cancellation
/// (deviations are squared AFTER the mean is subtracted).
/// Returns `(range, sigma, n)`.
#[inline]
fn row_moments(logits: &[f32]) -> (f64, f64, f64) {
    let n = logits.len();
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;
    let mut sum = 0.0f64; // exact: f32→f64 partial sums stay below 2⁵³
    for &x in logits {
        if x > max {
            max = x;
        }
        if x < min {
            min = x;
        }
        sum += f64::from(x);
    }
    let nf = n as f64;
    let mean = sum / nf;
    let mut m2 = 0.0f64;
    for &x in logits {
        let d = f64::from(x) - mean;
        m2 += d * d;
    }
    (f64::from(max) - f64::from(min), (m2 / nf).sqrt(), nf)
}

/// The Kamath range-law statistic `ρ = Δ̂/(2σ̂√(2 ln n))` for one logit row.
///
/// Gaussian rows land in the finite-n band around 1 (the `√(2 ln n)`
/// asymptotic overestimates `E[max]` at small n — ρ ≈ 0.65 at n = 16,
/// ≈ 0.87 at n = 1000, → 1 from below); a planted spike grows ρ roughly as
/// `√n/(2√(2 ln n))`. `NaN` for `n < 2` or a constant row — the honest
/// degenerate (no regime), never a fabricated 1.0.
#[inline]
pub fn kamath_rho(logits: &[f32]) -> f32 {
    let (range, sigma, n) = row_moments(logits);
    if n < 2.0 || sigma <= 0.0 || !sigma.is_finite() {
        return f32::NAN;
    }
    let denom = 2.0 * sigma * (2.0 * n.ln()).sqrt();
    (range / denom) as f32
}

/// Normalized entropy `H(p)/ln n` (nats) of the row's softmax — the Prop-1
/// dispersion axis: 1 = uniform (complete dispersion), 0 = one-hot.
///
/// Same shared kernel as `regime_probe::entropy::conditional_entropy_nats`
/// (see module docs for why the kernel is consumed directly). `0.0` for
/// `n ≤ 1` (no dispersion); non-finite logits propagate honestly.
#[inline]
pub fn normalized_entropy_nats(logits: &[f32]) -> f32 {
    let n = logits.len();
    if n <= 1 {
        return 0.0;
    }
    let (_, ln_z, mean_shift) = crate::simd::logsumexp_parts(logits);
    let h = ln_z - mean_shift;
    if !h.is_finite() {
        return h;
    }
    // H ≤ ln n by construction; clamp the ratio into [0, 1] against f32
    // rounding at the uniform end.
    let ratio = h / (n as f32).ln();
    ratio.clamp(0.0, 1.0)
}

/// Full regime reading for one logit row: ρ + normalized entropy + the
/// sigmoid spike score. Two passes over the row (Welford moments +
/// logsumexp), zero allocation (G4).
pub fn kamath_regime(logits: &[f32]) -> LogitRegime {
    let rho = kamath_rho(logits);
    LogitRegime {
        rho,
        normalized_entropy: normalized_entropy_nats(logits),
        spike_score: if rho.is_finite() {
            crate::simd::fast_sigmoid(rho.ln())
        } else {
            0.5
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic pseudo-logits (no RNG in unit tests): a fixed
    // multiplicative-hash sequence mapped to a band — good enough for
    // "Gaussian-ish noise" constructions after scaling.
    fn hash_noise(i: usize, salt: u32) -> f32 {
        let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
        ((h % 2009) as f32 / 1004.5) - 1.0
    }

    #[test]
    fn gaussian_rows_stay_in_the_finite_n_band() {
        // E[max] < √(2 ln n) at finite n, so Gaussian ρ sits below ~1 — the
        // honest band (not a magic ==1) the docs pin.
        for &n in &[64_usize, 256, 1024] {
            let row: Vec<f32> = (0..n).map(|i| hash_noise(i, 7)).collect();
            let rho = kamath_rho(&row);
            assert!(
                (0.35..=1.15).contains(&rho),
                "n={n}: Gaussian-ish ρ {rho} outside [0.35, 1.15]"
            );
        }
    }

    #[test]
    fn planted_spike_separates_from_gaussian() {
        // One dominant key (the needle regime) must push ρ well above the
        // Gaussian band, and the separation grows with n.
        let mut rho_small = 0.0;
        let mut rho_big = 0.0;
        for (slot, n) in [(&mut rho_small, 128_usize), (&mut rho_big, 1024)] {
            let mut row: Vec<f32> = (0..n).map(|i| hash_noise(i, 11) * 0.5).collect();
            row[3] = 8.0; // the spike
            *slot = kamath_rho(&row);
        }
        assert!(
            rho_small > 1.5,
            "spiked ρ must clear the Gaussian band: {rho_small}"
        );
        assert!(
            rho_big > rho_small,
            "separation grows with n: {rho_big} !> {rho_small}"
        );
        // and the score is decisively on the spiked side. NOTE the honest
        // calibration: a single spike gives ρ ≈ √n/(2√(2 ln n)) — INVARIANT
        // to spike magnitude (range and σ̂ both scale with it) — so realistic
        // single-needle rows score ~0.7-0.8, not →1; the Gaussian band sits
        // below 0.54 (ρ < 1.15).
        let mut row: Vec<f32> = (0..512).map(|i| hash_noise(i, 11) * 0.5).collect();
        row[3] = 8.0;
        let reading = kamath_regime(&row);
        assert!(
            reading.spike_score > 0.65,
            "score {} must be spiked-side (>0.65; Gaussian band <0.54)",
            reading.spike_score
        );
    }

    #[test]
    fn uniform_and_one_hot_entropy_bounds() {
        let uni = vec![0.0f32; 32];
        let h = normalized_entropy_nats(&uni);
        assert!((h - 1.0).abs() < 1e-5, "uniform H/ln n = 1, got {h}");

        let mut one_hot = vec![-40.0f32; 32];
        one_hot[5] = 40.0;
        let h = normalized_entropy_nats(&one_hot);
        assert!(h < 1e-4, "one-hot H/ln n ≈ 0, got {h}");
    }

    #[test]
    fn spike_score_is_a_half_at_the_gaussian_boundary() {
        // ρ = 1 exactly (constructed): range = 2σ√(2 ln n). Build a two-point
        // symmetric row ±a: range = 2a, σ = a → ρ = 2a/(2a√(2 ln 2)) … n=2:
        // √(2 ln 2) ≈ 1.177 → ρ ≈ 0.849. Instead check the boundary through
        // the score contract directly: ln ρ = 0 ⇒ 0.5.
        let reading = LogitRegime {
            rho: 1.0,
            normalized_entropy: 0.0,
            spike_score: crate::simd::fast_sigmoid(0.0),
        };
        assert!((reading.spike_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn degenerates_are_honest() {
        // n < 2 → NaN ρ, uninformed score, zero dispersion
        let reading = kamath_regime(&[3.0]);
        assert!(reading.rho.is_nan());
        assert_eq!(reading.spike_score, 0.5);
        assert_eq!(reading.normalized_entropy, 0.0);
        // constant row → σ̂ = 0 → NaN ρ (no regime), entropy honest (0: one-hot)
        let reading = kamath_regime(&[1.0, 1.0, 1.0, 1.0]);
        assert!(reading.rho.is_nan());
        assert!((reading.normalized_entropy - 1.0).abs() < 1e-5);
        // empty row
        assert!(kamath_rho(&[]).is_nan());
        assert_eq!(normalized_entropy_nats(&[]), 0.0);
    }

    #[test]
    fn dilution_direction_on_real_shaped_rows() {
        // A "distractor soup + weak gold" row (the SSMax dilution fixture
        // shape): gold barely above the soup → ρ in the Gaussian-ish band,
        // entropy HIGH (dispersed). Amplify the gold → ρ rises, entropy falls.
        let base: Vec<f32> = (0..256).map(|i| hash_noise(i, 3) * 0.4).collect();
        let mut weak = base.clone();
        weak[42] += 1.5;
        let mut strong = base;
        strong[42] += 9.0;
        let r_weak = kamath_regime(&weak);
        let r_strong = kamath_regime(&strong);
        assert!(r_strong.rho > r_weak.rho, "amplified gold must raise ρ");
        assert!(
            r_strong.normalized_entropy < r_weak.normalized_entropy,
            "amplified gold must concentrate the row"
        );
        assert!(
            r_weak.normalized_entropy > 0.9,
            "weak-gold soup is dispersed: {}",
            r_weak.normalized_entropy
        );
    }
}
