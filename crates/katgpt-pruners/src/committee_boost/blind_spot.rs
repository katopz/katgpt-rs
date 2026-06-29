//! Blind-spot floor estimation for committee boost (Plan 132, Phase 4, T17–T21).
//!
//! Measures the proposer diversity ceiling: B = 1 - lim_{k→∞} p_oracle(k).
//! A high blind-spot floor means no amount of committee scaling will help —
//! the proposers themselves need diversification.
//!
//! ## Paper Reference
//!
//! The blind-spot floor quantifies the fraction of inputs where *no* proposer
//! in the pool produces a correct response. If B = 0.2, then 20% of problems
//! are fundamentally uncovered by the current proposer set.

/// Result of exponential convergence fit on oracle best-of-k rates.
#[derive(Debug, Clone)]
pub struct ConvergenceFit {
    /// Estimated asymptote: lim_{k→∞} p_oracle(k).
    pub asymptote: f64,
    /// Exponential decay rate of residuals (higher = faster convergence).
    pub rate: f64,
    /// Whether the curve has effectively converged (residual < threshold).
    pub is_converged: bool,
}

/// Recommended action based on blind-spot analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CoverageAction {
    /// Blind-spot floor too high → diversify proposer pool (add new strategies).
    DiversifyProposers,
    /// Blind-spot floor is low but current k is insufficient → increase committee width.
    IncreaseK,
    /// Coverage is adequate — committee scaling is effective.
    Adequate,
}

/// Full coverage diagnostic from oracle best-of-k curve analysis.
#[derive(Debug, Clone)]
pub struct CoverageDiagnostic {
    /// Blind-spot floor: B ≈ 1 - max(p_oracle).
    pub blind_spot_floor: f64,
    /// Oracle success rate at the largest k tested.
    pub oracle_at_max_k: f64,
    /// Largest committee width tested.
    pub max_k: usize,
    /// Exponential convergence fit.
    pub convergence: ConvergenceFit,
    /// Recommended action to improve coverage.
    pub action: CoverageAction,
}

/// Blind-spot floor estimate from oracle best-of-k curve.
#[derive(Debug, Clone)]
pub struct BlindSpotEstimate {
    /// Blind-spot floor: B ≈ 1 - max(p_oracle).
    pub blind_spot_floor: f64,
    /// Oracle success rate at the largest k tested.
    pub oracle_at_max_k: f64,
    /// Largest committee width tested.
    pub max_k: usize,
}

/// Estimate the blind-spot floor from oracle best-of-k rates.
///
/// `oracle_rates` is a slice of `(k, p_oracle_k)` pairs where k is the
/// committee width and p_oracle_k is the best-of-k success rate with a
/// perfect selector.
///
/// Returns `B ≈ 1 - max(p_oracle)` across all observed k values.
/// Returns `1.0` (maximum blind spot) for empty input.
pub fn estimate_blind_spot_floor(oracle_rates: &[(usize, f64)]) -> f64 {
    match oracle_rates.iter().map(|&(_, rate)| rate).reduce(f64::max) {
        Some(max_rate) => 1.0 - max_rate,
        None => 1.0,
    }
}

/// Fit an exponential convergence model to oracle best-of-k rates.
///
/// Model: `p(k) ≈ a - b * exp(-c * k)` where `a` is the asymptote.
/// Estimates the asymptote and convergence rate from the data.
///
/// Uses a simple heuristic:
/// 1. Estimate asymptote as max observed rate (or slightly above).
/// 2. Fit decay rate from log-transformed residuals.
/// 3. Check convergence: residual between last two rates < 0.02.
pub fn fit_convergence(oracle_rates: &[(usize, f64)]) -> ConvergenceFit {
    if oracle_rates.is_empty() {
        return ConvergenceFit {
            asymptote: 0.0,
            rate: 0.0,
            is_converged: false,
        };
    }

    if oracle_rates.len() == 1 {
        return ConvergenceFit {
            asymptote: oracle_rates[0].1,
            rate: 0.0,
            is_converged: false,
        };
    }

    // Sort by k to ensure monotonic order.
    let mut sorted: Vec<(usize, f64)> = oracle_rates.to_vec();
    sorted.sort_by_key(|&(k, _)| k);

    // Asymptote estimate: use the maximum observed rate, clamped to [0, 1].
    let max_rate = sorted
        .iter()
        .map(|&(_, r)| r)
        .reduce(f64::max)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let asymptote = max_rate;

    // Fit exponential decay rate from log-transformed residuals.
    // residual(k) = asymptote - p(k) ≈ b * exp(-c * k)
    // ln(residual) ≈ ln(b) - c * k
    // Linear regression on (k, ln(residual)).
    let residuals: Vec<(f64, f64)> = sorted
        .iter()
        .filter(|(_, rate)| *rate < asymptote)
        .map(|&(k, rate)| {
            let residual = (asymptote - rate).max(1e-10);
            (k as f64, residual.ln())
        })
        .collect();

    let rate = if residuals.len() >= 2 {
        // Simple least-squares slope: c ≈ -Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
        let n = residuals.len() as f64;
        let x_mean = residuals.iter().map(|(x, _)| *x).sum::<f64>() / n;
        let y_mean = residuals.iter().map(|(_, y)| *y).sum::<f64>() / n;

        let numerator: f64 = residuals
            .iter()
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();
        let denominator: f64 = residuals.iter().map(|(x, _)| (x - x_mean).powi(2)).sum();

        if denominator.abs() < 1e-12 {
            0.0
        } else {
            // Slope is negative (residuals decay), rate = -slope.
            (-numerator / denominator).max(0.0)
        }
    } else {
        0.0
    };

    // Check convergence: need both (a) small gap between last two points AND
    // (b) at least 2 points observed (single point can't prove convergence).
    // The max rate being the last point doesn't prove convergence — need a plateau.
    let last_rate = sorted.last().map(|&(_, r)| r).unwrap_or(0.0);
    let second_last_rate = sorted
        .get(sorted.len().saturating_sub(2))
        .map(|&(_, r)| r)
        .unwrap_or(0.0);
    let residual_gap = (last_rate - second_last_rate).abs();
    let has_plateau = sorted.len() >= 2 && residual_gap < 0.02;
    let near_ceiling = (asymptote - last_rate).abs() < 0.02;
    let is_converged = has_plateau && near_ceiling;

    ConvergenceFit {
        asymptote,
        rate,
        is_converged,
    }
}

/// Produce a full coverage diagnostic from oracle best-of-k rates.
///
/// Combines blind-spot floor estimation, convergence fitting, and
/// action recommendation into a single diagnostic report.
///
/// ## Decision Logic
///
/// | Condition | Action |
/// |-----------|--------|
/// | B > 0.3 | `DiversifyProposers` — 30%+ of problems are uncovered |
/// | B ≤ 0.3 and not converged | `IncreaseK` — scaling committee width will help |
/// | B ≤ 0.3 and converged | `Adequate` — current setup is effective |
pub fn coverage_diagnostic(oracle_rates: &[(usize, f64)]) -> CoverageDiagnostic {
    let blind_spot_floor = estimate_blind_spot_floor(oracle_rates);
    let convergence = fit_convergence(oracle_rates);

    let max_k = oracle_rates.iter().map(|&(k, _)| k).max().unwrap_or(0);
    let oracle_at_max_k = oracle_rates
        .iter()
        .filter(|(k, _)| *k == max_k)
        .map(|&(_, rate)| rate)
        .next()
        .unwrap_or(0.0);

    let action = match () {
        _ if blind_spot_floor > 0.3 => CoverageAction::DiversifyProposers,
        _ if !convergence.is_converged => CoverageAction::IncreaseK,
        _ => CoverageAction::Adequate,
    };

    CoverageDiagnostic {
        blind_spot_floor,
        oracle_at_max_k,
        max_k,
        convergence,
        action,
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    /// Assert `actual` is within `eps` of `expected`.
    fn approx_eq(actual: f64, expected: f64, eps: f64) {
        assert!(
            (actual - expected).abs() < eps,
            "expected ~{expected}, got {actual}, diff={}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn test_saturation_at_0_8_gives_b_0_2() {
        let rates = vec![(1, 0.5), (2, 0.65), (4, 0.75), (8, 0.8), (16, 0.8)];
        let b = estimate_blind_spot_floor(&rates);
        approx_eq(b, 0.2, EPS);
    }

    #[test]
    fn test_monotonic_increase_b_near_zero() {
        let rates = vec![
            (1, 0.6),
            (2, 0.75),
            (4, 0.85),
            (8, 0.92),
            (16, 0.97),
            (32, 0.995),
        ];
        let b = estimate_blind_spot_floor(&rates);
        approx_eq(b, 0.005, EPS);
    }

    #[test]
    fn test_single_point_b_is_one_minus_rate() {
        let rates = vec![(4, 0.65)];
        let b = estimate_blind_spot_floor(&rates);
        approx_eq(b, 0.35, EPS);
    }

    #[test]
    fn test_empty_input_b_is_one() {
        let b = estimate_blind_spot_floor(&[]);
        approx_eq(b, 1.0, EPS);
    }

    #[test]
    fn test_convergence_fit_empty() {
        let fit = fit_convergence(&[]);
        approx_eq(fit.asymptote, 0.0, EPS);
        assert!(!fit.is_converged);
    }

    #[test]
    fn test_convergence_fit_single_point() {
        let fit = fit_convergence(&[(4, 0.7)]);
        approx_eq(fit.asymptote, 0.7, EPS);
        assert!(!fit.is_converged);
    }

    #[test]
    fn test_convergence_fit_converged() {
        let rates = vec![(1, 0.5), (2, 0.65), (4, 0.78), (8, 0.79), (16, 0.79)];
        let fit = fit_convergence(&rates);
        approx_eq(fit.asymptote, 0.79, EPS);
        assert!(fit.is_converged);
        assert!(fit.rate > 0.0);
    }

    #[test]
    fn test_convergence_fit_not_converged() {
        let rates = vec![(1, 0.3), (2, 0.5), (4, 0.7), (8, 0.85)];
        let fit = fit_convergence(&rates);
        approx_eq(fit.asymptote, 0.85, EPS);
        assert!(!fit.is_converged);
    }

    #[test]
    fn test_convergence_rate_positive() {
        // Strictly increasing rates should give positive decay rate.
        let rates = vec![
            (1, 0.4),
            (2, 0.55),
            (4, 0.65),
            (8, 0.72),
            (16, 0.76),
            (32, 0.78),
        ];
        let fit = fit_convergence(&rates);
        assert!(
            fit.rate > 0.0,
            "decay rate should be positive, got {}",
            fit.rate
        );
    }

    #[test]
    fn test_diagnostic_diversify_proposers() {
        // High blind-spot floor (B=0.4) → need diversification.
        let rates = vec![(1, 0.4), (2, 0.5), (4, 0.58), (8, 0.6)];
        let diag = coverage_diagnostic(&rates);
        assert_eq!(diag.action, CoverageAction::DiversifyProposers);
        approx_eq(diag.blind_spot_floor, 0.4, EPS);
    }

    #[test]
    fn test_diagnostic_increase_k() {
        // Low blind-spot floor but not converged → increase committee width.
        let rates = vec![(1, 0.6), (2, 0.75), (4, 0.85), (8, 0.91)];
        let diag = coverage_diagnostic(&rates);
        assert_eq!(diag.action, CoverageAction::IncreaseK);
        approx_eq(diag.blind_spot_floor, 0.09, EPS);
    }

    #[test]
    fn test_diagnostic_adequate() {
        // Low blind-spot floor and converged → adequate.
        let rates = vec![
            (1, 0.7),
            (2, 0.82),
            (4, 0.89),
            (8, 0.92),
            (16, 0.93),
            (32, 0.93),
        ];
        let diag = coverage_diagnostic(&rates);
        assert_eq!(diag.action, CoverageAction::Adequate);
        approx_eq(diag.blind_spot_floor, 0.07, EPS);
    }

    #[test]
    fn test_diagnostic_max_k_and_oracle() {
        let rates = vec![(1, 0.5), (4, 0.7), (16, 0.85)];
        let diag = coverage_diagnostic(&rates);
        assert_eq!(diag.max_k, 16);
        approx_eq(diag.oracle_at_max_k, 0.85, EPS);
    }

    #[test]
    fn test_unsorted_input() {
        // Input is not sorted by k — should still work.
        let rates = vec![(8, 0.8), (2, 0.6), (16, 0.82), (1, 0.4), (4, 0.7)];
        let b = estimate_blind_spot_floor(&rates);
        approx_eq(b, 0.18, EPS);
    }

    #[test]
    fn test_flat_rates_converged() {
        // All rates are the same → converged.
        let rates = vec![(1, 0.7), (2, 0.7), (4, 0.7)];
        let fit = fit_convergence(&rates);
        approx_eq(fit.asymptote, 0.7, EPS);
        // No residuals to fit, rate should be 0.
        approx_eq(fit.rate, 0.0, EPS);
    }
}
