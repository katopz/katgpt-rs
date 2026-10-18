//! KarcShard Differential-Privacy Output Perturbation (Issue 370 T4).
//!
//! Post-hoc Gaussian noise on a fitted ridge `Wout` matrix to provide formal
//! (ε, δ)-DP for the committed KarcShard parameters. This is the modelless
//! parameter-defense primitive — no training, no gradient descent, just a
//! one-shot noise injection at the commit boundary.
//!
//! # What this defends (and what it does NOT — Benchmark 399)
//!
//! Research 315 (and Benchmark 398) established that the committed KarcShard
//! is always MI-vulnerable (F1 ≥ 0.755 under the Yeom 2018 loss-threshold
//! attack) because the 3.4 params/sample ratio causes systematic overfitting.
//! Ridge regularization (`λ`) cannot fix this — it preserves the train/test
//! residual *ratio* that the MI attack exploits.
//!
//! This primitive provides formal (ε, δ)-DP for the Wout **parameters**
//! (Chaudhuri et al. 2011 output perturbation), defending
//! **parameter-inspection MI** (attacker reads Wout to detect memorized
//! patterns).
//!
//! **It does NOT defend Yeom 2018 loss-threshold MI** (Benchmark 399,
//! 2026-07-07). The noise adds `σ_dp²·‖x‖²` to BOTH member and non-member
//! expected forecast losses equally, preserving the train/test loss ordering
//! that Yeom exploits. Measured F1 stays ≥ 0.71 (noisy data) / ≥ 0.94 (clean
//! data) even at σ_dp=1.0 (ε≈4.8). The true defenses against loss-based MI
//! are operational (don't commit overfit models) or training-time (DP-SGD →
//! riir-train).
//!
//! # Theory
//!
//! Based on Chaudhuri et al. 2011 ("Differentially Private Empirical Risk
//! Minimization", §4.2 Output Perturbation), applied to the closed-form
//! ridge solve `Wout = Y·Hᵀ·(H·Hᵀ + λI)⁻¹`:
//!
//! 1. The ridge solution `w*` is the optimal point of a λ-strongly convex ERM.
//! 2. The ℓ₂ sensitivity of `w*` to a single-sample change is bounded by
//!    `Δ₂ ≤ 2L / (λn)`, where `L` is the per-sample loss Lipschitz constant
//!    and `n` is the sample count.
//! 3. The Gaussian mechanism gives (ε, δ)-DP with
//!    `σ ≥ Δ₂ · √(2·ln(1.25/δ)) / ε` (Dwork & Roth 2014, Theorem A.1).
//!
//! Adding iid `N(0, σ²)` noise to every entry of `Wout` before commit yields
//! (ε, δ)-DP for the committed **parameter artifact**. This guarantees that an
//! attacker examining the Wout values cannot distinguish (beyond the DP
//! budget) whether a specific sample was in the fit set BY INSPECTING THE
//! PARAMETERS. It does not prevent loss-based inference at inference time.
//!
//! # Privacy / utility tradeoff
//!
//! Larger `σ_dp` → more parameter privacy but worse forecast quality. The
//! loss-based MI F1 barely moves (Benchmark 399). See
//! `riir-ai/.benchmarks/399_karc_dp_noise_defense.md` for the full analysis.
//!
//! # Modelless compliance
//!
//! This is a post-hoc perturbation on a closed-form solve — no training, no
//! backprop. Fits the katgpt-rs modelless-first mandate (rule 1.c: latent-
//! space updates). No riir-train dependency.
//!
//! # Entropy caveat
//!
//! The unseeded path uses `std::time::SystemTime` nanoseconds as the PRNG
//! seed. This is NOT a CSPRNG — sufficient for NPC HLA trajectories (the
//! KarcShard use case) but inadequate for protecting user PII. For
//! high-stakes privacy, the caller should supply their own CSPRNG-derived
//! seed via [`KarcDpNoiseConfig::seed`].
//!
//! KarcShard DP Output Perturbation (Issue 370 T4) — post-hoc Gaussian noise
//! on a fitted ridge Wout matrix to provide formal (ε,δ)-DP for the committed
//! KarcShard parameters. Defends PARAMETER-INSPECTION MI (attacker reads Wout
//! to detect memorized patterns). Does NOT defend Yeom loss-threshold MI —
//! see karc_dp module docs and riir-ai/.benchmarks/399 for the structural
//! insufficiency analysis. Modelless (post-hoc noise on a closed-form solve).
//! Gated on karc_forecaster since it operates on the Wout produced by
//! KarcForecaster::fit_ridge.

/// Configuration for differential-privacy output perturbation on a fitted
/// ridge `Wout` matrix.
///
/// See the module docs for the Chaudhuri et al. 2011 theory and the
/// privacy/utility tradeoff.
#[derive(Clone, Copy, Debug)]
pub struct KarcDpNoiseConfig {
    /// Privacy parameter `ε > 0`. Smaller = more private = more noise.
    /// Typical range: `[0.1, 10.0]`. The DP guarantee degrades as `ε → 0`.
    pub epsilon: f32,
    /// Privacy parameter `δ ∈ (0, 1)`. Smaller = more private.
    /// Typically `δ ≤ 1/n` where `n` is the sample count; `1e-5` is standard.
    pub delta: f32,
    /// ℓ₂ sensitivity bound `Δ₂` of the ridge solution to a single-sample
    /// change. The caller must derive this from their data regime:
    ///
    /// `Δ₂ = 2L / (λn)` where `L` is the Lipschitz constant of the per-sample
    /// squared loss. For bounded `‖x‖ ≤ R_x`, `|y| ≤ R_y`, `‖w‖ ≤ R_w`:
    /// `L ≤ (R_w · R_x + R_y) · R_x`.
    ///
    /// If unknown, measure it empirically: fit on a dataset, drop one sample,
    /// refit, measure `‖Δw‖₂`, take the max over the dataset.
    pub sensitivity_l2: f32,
    /// Optional deterministic seed for the Gaussian noise.
    /// - `Some(seed)`: deterministic (tests, reproducibility).
    /// - `None`: seeded from system time (production — fresh noise per call).
    pub seed: Option<u64>,
}

impl KarcDpNoiseConfig {
    /// Calibrate the Gaussian noise stddev `σ_dp` from the privacy budget.
    ///
    /// `σ_dp = Δ₂ · √(2·ln(1.25/δ)) / ε`
    ///
    /// This is the standard Gaussian mechanism calibration (Dwork & Roth
    /// 2014, Theorem A.1). Returns `f32::INFINITY` if `epsilon ≤ 0` or
    /// `delta ∉ (0, 1)` (invalid budget — caller bug; the resulting noise
    /// will be ±inf, surfacing the error immediately).
    #[inline]
    pub fn sigma_dp(&self) -> f32 {
        if self.epsilon <= 0.0 || self.delta <= 0.0 || self.delta >= 1.0 {
            return f32::INFINITY;
        }
        let two_ln = 2.0f32 * (1.25f32 / self.delta).ln();
        self.sensitivity_l2 * two_ln.sqrt() / self.epsilon
    }
}

impl Default for KarcDpNoiseConfig {
    /// A conservative default budget: `ε=1.0, δ=1e-5, Δ₂=0.1, seed=None`.
    ///
    /// `Δ₂=0.1` is a placeholder for normalized-feature regimes. The caller
    /// SHOULD override `sensitivity_l2` with a value derived from their actual
    /// data bounds (see the field doc). Using the default sensitivity without
    /// measurement may under- or over-state the privacy guarantee.
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: None,
        }
    }
}

// ── PRNG (xorshift64 + Box-Muller) ─────────────────────────────────────────
//
// Matches the MI benchmark harness (`riir-engine/tests/karc_mi_bench.rs`)
// for consistency. xorshift64 is fast and deterministic; Box-Muller gives
// a good Gaussian approximation for the tail behavior DP needs. Neither is
// a CSPRNG — see the module entropy caveat.

struct DpRng {
    state: u64,
}

impl DpRng {
    #[inline]
    fn new(seed: u64) -> Self {
        // Avoid the xorshift64 degenerate seed 0.
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Seed from system time (production entropy — NOT a CSPRNG; see module
    /// entropy caveat). Falls back to a fixed constant if the clock is
    /// unavailable (e.g. on some embedded targets before UNIX_EPOCH).
    fn from_system_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0xDEAD_BEEF_CAFE_F00D, |d| d.as_nanos() as u64);
        Self::new(nanos)
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Top 24 bits → [0, 1) — enough mantissa precision for f32 Gaussian.
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Box-Muller Gaussian — mean 0, stddev `sigma`.
    #[inline]
    fn next_gaussian(&mut self, sigma: f32) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * core::f32::consts::PI * u2;
        sigma * r * theta.cos()
    }
}

/// Apply differential-privacy output perturbation to a fitted `Wout` matrix,
/// in-place.
///
/// Adds iid `N(0, σ_dp²)` noise to every entry, where `σ_dp` is calibrated
/// from the config's privacy budget via [`KarcDpNoiseConfig::sigma_dp`].
///
/// This is the modelless MI defense for the committed KarcShard (Issue 370
/// T4). Call this on the forecaster's `wout` field AFTER `fit_ridge` and
/// BEFORE committing (freeze/serialize). The noise destroys the systematic
/// member-vs-non-member residual gap that the Yeom 2018 MI attack exploits.
///
/// # Arguments
///
/// - `wout` — the fitted readout matrix (`D × d_h` row-major, mutable slice).
/// - `config` — the privacy budget configuration.
///
/// # Example
///
/// ```
/// use katgpt_core::karc_dp::{apply_dp_noise_to_wout, KarcDpNoiseConfig};
///
/// let mut wout = [1.0f32, 0.5, -0.3, 0.8];
/// let config = KarcDpNoiseConfig {
///     epsilon: 1.0,
///     delta: 1e-5,
///     sensitivity_l2: 0.1,
///     seed: Some(42),
/// };
/// apply_dp_noise_to_wout(&mut wout, &config);
/// // wout entries are now perturbed by ~N(0, sigma_dp²).
/// ```
pub fn apply_dp_noise_to_wout(wout: &mut [f32], config: &KarcDpNoiseConfig) {
    let sigma = config.sigma_dp();
    let mut rng = match config.seed {
        Some(seed) => DpRng::new(seed),
        None => DpRng::from_system_time(),
    };
    for w in wout.iter_mut() {
        *w += rng.next_gaussian(sigma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_dp_calibration_matches_formula() {
        // σ_dp = Δ₂ · √(2·ln(1.25/δ)) / ε
        // With ε=1, δ=1e-5, Δ₂=0.1:
        //   √(2·ln(1.25/1e-5)) = √(2·ln(125000)) = √(2·11.736) = √23.472 = 4.8449
        //   σ_dp = 0.1 · 4.8449 / 1.0 = 0.48449
        let cfg = KarcDpNoiseConfig {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(0),
        };
        let sigma = cfg.sigma_dp();
        let expected: f32 = 0.1 * (2.0f32 * (1.25f32 / 1e-5f32).ln()).sqrt();
        assert!(
            (sigma - expected).abs() < 1e-4,
            "sigma_dp {sigma:.6} != expected {expected:.6}"
        );
        // Spot-check against the hand-computed value.
        assert!(
            (sigma - 0.4845).abs() < 1e-2,
            "sigma_dp {sigma} should be ~0.4845, got {sigma}"
        );
    }

    #[test]
    fn sigma_dp_returns_infinity_for_invalid_budget() {
        let cases = [
            (0.0, 1e-5, 0.1),  // epsilon <= 0
            (-1.0, 1e-5, 0.1), // epsilon < 0
            (1.0, 0.0, 0.1),   // delta <= 0
            (1.0, 1.0, 0.1),   // delta >= 1
            (1.0, 1.5, 0.1),   // delta > 1
        ];
        for &(eps, delta, sens) in &cases {
            let cfg = KarcDpNoiseConfig {
                epsilon: eps,
                delta,
                sensitivity_l2: sens,
                seed: Some(0),
            };
            assert!(
                cfg.sigma_dp().is_infinite(),
                "epsilon={eps}, delta={delta} should give infinite sigma"
            );
        }
    }

    #[test]
    fn sigma_dp_scales_linearly_with_sensitivity() {
        let base = KarcDpNoiseConfig {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(0),
        };
        let double = KarcDpNoiseConfig {
            sensitivity_l2: 0.2,
            ..base
        };
        // Doubling Δ₂ doubles σ_dp.
        let ratio = double.sigma_dp() / base.sigma_dp();
        assert!(
            (ratio - 2.0).abs() < 1e-4,
            "doubling sensitivity should double sigma_dp, got ratio {ratio}"
        );
    }

    #[test]
    fn sigma_dp_scales_inversely_with_epsilon() {
        let base = KarcDpNoiseConfig {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(0),
        };
        let half_eps = KarcDpNoiseConfig {
            epsilon: 0.5,
            ..base
        };
        // Halving ε doubles σ_dp.
        let ratio = half_eps.sigma_dp() / base.sigma_dp();
        assert!(
            (ratio - 2.0).abs() < 1e-4,
            "halving epsilon should double sigma_dp, got ratio {ratio}"
        );
    }

    #[test]
    fn same_seed_gives_identical_noise() {
        let mut a = [0.0f32; 100];
        let mut b = [0.0f32; 100];
        let cfg = KarcDpNoiseConfig {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(12345),
        };
        apply_dp_noise_to_wout(&mut a, &cfg);
        apply_dp_noise_to_wout(&mut b, &cfg);
        assert_eq!(a, b, "same seed must give bit-identical noise");
    }

    #[test]
    fn different_seeds_give_different_noise() {
        let mut a = [0.0f32; 100];
        let mut b = [0.0f32; 100];
        let cfg_a = KarcDpNoiseConfig {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(1),
        };
        let cfg_b = KarcDpNoiseConfig {
            seed: Some(2),
            ..cfg_a
        };
        apply_dp_noise_to_wout(&mut a, &cfg_a);
        apply_dp_noise_to_wout(&mut b, &cfg_b);
        let diffs = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        assert!(
            diffs > 90,
            "different seeds should produce mostly-different noise ({diffs} of 100 differ)"
        );
    }

    #[test]
    fn noise_statistics_match_sigma_dp() {
        // Over a large sample, the noise mean should be ≈ 0 and stddev ≈ σ_dp.
        // We generate a large wout and check the empirical statistics.
        let n = 10_000;
        let mut wout = vec![0.0f32; n];
        let cfg = KarcDpNoiseConfig {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(42),
        };
        apply_dp_noise_to_wout(&mut wout, &cfg);
        let sigma = cfg.sigma_dp();

        let mean: f64 = wout.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var: f64 = wout.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = var.sqrt();

        // Box-Muller + xorshift64 is not perfectly calibrated; allow 10%
        // tolerance. Mean should be well within ±5% of σ_dp.
        assert!(
            mean.abs() < 0.05 * sigma as f64,
            "noise mean {} should be near 0 (tol {})",
            mean,
            0.05 * sigma as f64
        );
        assert!(
            (stddev - sigma as f64).abs() / (sigma as f64) < 0.10,
            "noise stddev {stddev} should be near sigma_dp {sigma} (10% tol)"
        );
    }

    #[test]
    fn apply_modifies_all_entries() {
        let mut wout = [1.0f32; 50];
        let original = wout;
        let cfg = KarcDpNoiseConfig {
            epsilon: 1.0,
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(7),
        };
        apply_dp_noise_to_wout(&mut wout, &cfg);
        let unchanged = wout
            .iter()
            .zip(original.iter())
            .filter(|(a, b)| a == b)
            .count();
        // With continuous noise, the probability of any entry being bit-
        // identical to the original is essentially zero. Allow a tiny margin
        // for floating-point coincidence.
        assert!(
            unchanged < 2,
            "{unchanged} of 50 entries unchanged after DP noise — noise not applied to all"
        );
    }

    #[test]
    fn larger_sigma_produces_larger_spread() {
        // Two configs with different ε → different σ_dp → the larger-σ one
        // should produce a larger empirical spread.
        let mut small = vec![0.0f32; 5000];
        let mut large = vec![0.0f32; 5000];
        let cfg_small = KarcDpNoiseConfig {
            epsilon: 10.0, // small σ_dp
            delta: 1e-5,
            sensitivity_l2: 0.1,
            seed: Some(99),
        };
        let cfg_large = KarcDpNoiseConfig {
            epsilon: 0.1, // large σ_dp (100× more noise)
            ..cfg_small
        };
        apply_dp_noise_to_wout(&mut small, &cfg_small);
        apply_dp_noise_to_wout(&mut large, &cfg_large);

        let var_small: f64 =
            small.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / small.len() as f64;
        let var_large: f64 =
            large.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / large.len() as f64;

        assert!(
            var_large > var_small * 10.0,
            "larger sigma should produce larger variance: large={} vs small={} (ratio {})",
            var_large,
            var_small,
            var_large / var_small
        );
    }

    #[test]
    fn default_config_is_valid() {
        // The Default impl should produce a finite, positive sigma_dp.
        let cfg = KarcDpNoiseConfig::default();
        let sigma = cfg.sigma_dp();
        assert!(sigma.is_finite(), "default sigma_dp should be finite");
        assert!(sigma > 0.0, "default sigma_dp should be positive");
    }
}
