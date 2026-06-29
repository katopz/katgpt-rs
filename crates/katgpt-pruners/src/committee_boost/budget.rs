//! Committee budget sizing from theoretical bounds (Plan 132, Phase 3).
//!
//! Given committee protocol parameters (α₀, β₀, σ₀, L, δ), compute optimal
//! (k, m, r) per the paper's Theorem 3:
//!
//! - **k** (proposer width): `k ≥ |P_N| × ⌈ln(2L/δ) / α₀⌉`
//! - **m** (critic depth): `m ≥ ⌈(1/2β₀) × ln(2k²L/δ)⌉`
//! - **r** (comparator votes): `r ≥ ⌈(1/4σ₀²) × ln(2k²L/δ)⌉`
//!
//! These bounds guarantee that the committee protocol achieves success
//! probability ≥ 1 − δ across L rounds of selection.
//!
//! Reference: arXiv:2605.14163, Theorem 3

/// Committee budget: optimal (k, m, r) sizing from theoretical bounds.
///
/// - `k`: proposer width — number of candidates generated per round
/// - `m`: critic depth — number of critic evaluations per candidate
/// - `r`: comparator votes — number of pairwise comparisons per pair
///
/// Derived from Theorem 3 parameters (α₀, β₀, σ₀, L, δ, |P_N|).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitteeBudget {
    /// Proposer width: number of candidates generated per round.
    pub k: usize,
    /// Critic depth: number of ScreeningPruner evaluations per candidate.
    pub m: usize,
    /// Comparator votes: number of BtRank pairwise comparisons per pair.
    pub r: usize,
}

/// Validation error for committee budget parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetError {
    /// alpha (base proposer accuracy) must be in (0, 1].
    AlphaOutOfRange { alpha: f64 },
    /// beta (critic accuracy) must be in (0, 1].
    BetaOutOfRange { beta: f64 },
    /// sigma (comparator accuracy) must be in (0, 1].
    SigmaOutOfRange { sigma: f64 },
    /// delta (failure probability) must be in (0, 1).
    DeltaOutOfRange { delta: f64 },
    /// depth (L) must be ≥ 1.
    DepthTooSmall { depth: usize },
    /// portfolio_size (|P_N|) must be ≥ 1.
    PortfolioTooSmall { portfolio_size: usize },
    /// Computed k is zero.
    KIsZero,
    /// Computed m is zero.
    MIsZero,
    /// Computed r is zero.
    RIsZero,
}

impl core::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlphaOutOfRange { alpha } => {
                write!(f, "alpha must be in (0, 1], got {alpha}")
            }
            Self::BetaOutOfRange { beta } => {
                write!(f, "beta must be in (0, 1], got {beta}")
            }
            Self::SigmaOutOfRange { sigma } => {
                write!(f, "sigma must be in (0, 1], got {sigma}")
            }
            Self::DeltaOutOfRange { delta } => {
                write!(f, "delta must be in (0, 1), got {delta}")
            }
            Self::DepthTooSmall { depth } => {
                write!(f, "depth must be >= 1, got {depth}")
            }
            Self::PortfolioTooSmall { portfolio_size } => {
                write!(f, "portfolio_size must be >= 1, got {portfolio_size}")
            }
            Self::KIsZero => write!(f, "computed k is zero — alpha may be too large"),
            Self::MIsZero => write!(f, "computed m is zero — beta may be too large"),
            Self::RIsZero => write!(f, "computed r is zero — sigma may be too large"),
        }
    }
}

impl core::error::Error for BudgetError {}

impl CommitteeBudget {
    /// Total number of role calls for a full committee protocol execution.
    ///
    /// The committee protocol Π_{k,m,r} over L rounds makes:
    /// - **k** proposer calls per round (generate candidates)
    /// - **k × m** critic calls per round (evaluate each candidate m times)
    /// - **k² × r** comparator calls per round (compare all k×(k-1)/2 pairs, r votes each,
    ///   upper-bounded by k² × r for simplicity)
    ///
    /// Total: `L × (k + m·k + r·k²)` = `L × k × (1 + m + r·k)`
    pub fn total_role_calls(&self, depth: usize) -> usize {
        depth * self.k * (1 + self.m + self.r * self.k)
    }

    /// Validate that all computed budget values are sensible.
    ///
    /// Returns `Ok(())` if k ≥ 1, m ≥ 1, r ≥ 1.
    /// Returns `Err(BudgetError)` with the first violation found.
    pub fn validate(&self) -> Result<(), BudgetError> {
        if self.k == 0 {
            return Err(BudgetError::KIsZero);
        }
        if self.m == 0 {
            return Err(BudgetError::MIsZero);
        }
        if self.r == 0 {
            return Err(BudgetError::RIsZero);
        }
        Ok(())
    }
}

/// Compute committee budget from theoretical sizing rules (Theorem 3).
///
/// # Parameters
///
/// - `depth` (L): number of selection rounds (tree depth)
/// - `delta`: target failure probability (must be in (0, 1))
/// - `alpha` (α₀): base proposer accuracy per candidate (must be in (0, 1])
/// - `beta` (β₀): critic accuracy per evaluation (must be in (0, 1])
/// - `sigma` (σ₀): comparator accuracy per comparison (must be in (0, 1])
/// - `portfolio_size` (|P_N|): number of distinct proposal strategies
///
/// # Returns
///
/// `CommitteeBudget` with theoretically optimal (k, m, r).
///
/// # Errors
///
/// Returns `BudgetError` if any parameter is out of range or if any
/// computed value is zero.
pub fn committee_budget(
    depth: usize,
    delta: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    portfolio_size: usize,
) -> Result<CommitteeBudget, BudgetError> {
    // Validate inputs.
    if depth == 0 {
        return Err(BudgetError::DepthTooSmall { depth });
    }
    if portfolio_size == 0 {
        return Err(BudgetError::PortfolioTooSmall { portfolio_size });
    }
    if delta <= 0.0 || delta >= 1.0 {
        return Err(BudgetError::DeltaOutOfRange { delta });
    }
    if alpha <= 0.0 || alpha > 1.0 {
        return Err(BudgetError::AlphaOutOfRange { alpha });
    }
    if beta <= 0.0 || beta > 1.0 {
        return Err(BudgetError::BetaOutOfRange { beta });
    }
    if sigma <= 0.0 || sigma > 1.0 {
        return Err(BudgetError::SigmaOutOfRange { sigma });
    }

    let l = depth as f64;
    let ln_2l_delta = (2.0 * l / delta).ln();

    // k ≥ |P_N| × ⌈ln(2L/δ) / α₀⌉
    let k = (portfolio_size as f64 * (ln_2l_delta / alpha).ceil()).ceil() as usize;

    let k_f64 = k as f64;
    let ln_2k2l_delta = (2.0 * k_f64 * k_f64 * l / delta).ln();

    // m ≥ ⌈(1/2β₀) × ln(2k²L/δ)⌉
    let m = ((1.0 / (2.0 * beta)) * ln_2k2l_delta).ceil() as usize;

    // r ≥ ⌈(1/4σ₀²) × ln(2k²L/δ)⌉
    let r = ((1.0 / (4.0 * sigma * sigma)) * ln_2k2l_delta).ceil() as usize;

    let budget = CommitteeBudget { k, m, r };
    budget.validate()?;
    Ok(budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_budget_basic_sizing() {
        // Standard parameters: L=10, δ=0.05, α=0.3, β=0.2, σ=0.4, |P_N|=2
        let budget = committee_budget(10, 0.05, 0.3, 0.2, 0.4, 2).expect("valid budget");
        assert!(budget.k >= 2, "k should be >= portfolio_size");
        assert!(budget.m >= 1, "m should be >= 1");
        assert!(budget.r >= 1, "r should be >= 1");
    }

    #[test]
    fn test_budget_small_depth() {
        // L=1, δ=0.1, α=0.5, β=0.5, σ=0.5, |P_N|=1
        let budget = committee_budget(1, 0.1, 0.5, 0.5, 0.5, 1).expect("valid budget");
        assert!(budget.k >= 1);
        assert!(budget.m >= 1);
        assert!(budget.r >= 1);
    }

    #[test]
    fn test_budget_large_portfolio() {
        // Large portfolio means more proposers needed
        let small = committee_budget(5, 0.05, 0.3, 0.2, 0.4, 2).expect("valid");
        let large = committee_budget(5, 0.05, 0.3, 0.2, 0.4, 8).expect("valid");
        assert!(
            large.k >= small.k,
            "larger portfolio should need >= k: {} vs {}",
            large.k,
            small.k
        );
    }

    #[test]
    fn test_budget_tighter_delta_needs_more() {
        // Tighter delta (lower failure probability) needs more samples
        let loose = committee_budget(5, 0.1, 0.3, 0.2, 0.4, 2).expect("valid");
        let tight = committee_budget(5, 0.01, 0.3, 0.2, 0.4, 2).expect("valid");
        assert!(
            tight.k >= loose.k,
            "tighter delta should need >= k: {} vs {}",
            tight.k,
            loose.k
        );
    }

    #[test]
    fn test_budget_high_alpha_fewer_proposers() {
        // Higher α₀ means proposer is more accurate, needs fewer samples
        let low_alpha = committee_budget(5, 0.05, 0.2, 0.3, 0.4, 2).expect("valid");
        let high_alpha = committee_budget(5, 0.05, 0.8, 0.3, 0.4, 2).expect("valid");
        assert!(
            high_alpha.k <= low_alpha.k,
            "higher alpha should need <= k: {} vs {}",
            high_alpha.k,
            low_alpha.k
        );
    }

    #[test]
    fn test_budget_high_beta_fewer_critic() {
        // Higher β₀ means critic is more accurate, needs fewer evaluations
        let low_beta = committee_budget(5, 0.05, 0.3, 0.2, 0.4, 2).expect("valid");
        let high_beta = committee_budget(5, 0.05, 0.3, 0.8, 0.4, 2).expect("valid");
        assert!(
            high_beta.m <= low_beta.m,
            "higher beta should need <= m: {} vs {}",
            high_beta.m,
            low_beta.m
        );
    }

    #[test]
    fn test_budget_high_sigma_fewer_comparisons() {
        // Higher σ₀ means comparator is more accurate, needs fewer votes
        let low_sigma = committee_budget(5, 0.05, 0.3, 0.2, 0.2, 2).expect("valid");
        let high_sigma = committee_budget(5, 0.05, 0.3, 0.2, 0.8, 2).expect("valid");
        assert!(
            high_sigma.r <= low_sigma.r,
            "higher sigma should need <= r: {} vs {}",
            high_sigma.r,
            low_sigma.r
        );
    }

    #[test]
    fn test_total_role_calls_formula() {
        // L=3, k=2, m=3, r=4 → 3 × 2 × (1 + 3 + 4×2) = 6 × 12 = 72
        let budget = CommitteeBudget { k: 2, m: 3, r: 4 };
        assert_eq!(budget.total_role_calls(3), 72);
    }

    #[test]
    fn test_total_role_calls_single_round() {
        // L=1, k=3, m=2, r=1 → 1 × 3 × (1 + 2 + 1×3) = 3 × 6 = 18
        let budget = CommitteeBudget { k: 3, m: 2, r: 1 };
        assert_eq!(budget.total_role_calls(1), 18);
    }

    #[test]
    fn test_total_role_calls_zero_depth() {
        let budget = CommitteeBudget { k: 5, m: 3, r: 2 };
        assert_eq!(budget.total_role_calls(0), 0);
    }

    #[test]
    fn test_validate_ok() {
        let budget = CommitteeBudget { k: 1, m: 1, r: 1 };
        assert!(budget.validate().is_ok());
    }

    #[test]
    fn test_validate_k_zero() {
        let budget = CommitteeBudget { k: 0, m: 1, r: 1 };
        assert_eq!(budget.validate(), Err(BudgetError::KIsZero));
    }

    #[test]
    fn test_validate_m_zero() {
        let budget = CommitteeBudget { k: 1, m: 0, r: 1 };
        assert_eq!(budget.validate(), Err(BudgetError::MIsZero));
    }

    #[test]
    fn test_validate_r_zero() {
        let budget = CommitteeBudget { k: 1, m: 1, r: 0 };
        assert_eq!(budget.validate(), Err(BudgetError::RIsZero));
    }

    #[test]
    fn test_budget_error_display() {
        let err = BudgetError::AlphaOutOfRange { alpha: 1.5 };
        assert!(err.to_string().contains("1.5"));

        let err = BudgetError::KIsZero;
        assert!(err.to_string().contains("k is zero"));
    }

    #[test]
    fn test_reject_zero_depth() {
        let result = committee_budget(0, 0.05, 0.3, 0.2, 0.4, 2);
        assert_eq!(result, Err(BudgetError::DepthTooSmall { depth: 0 }));
    }

    #[test]
    fn test_reject_zero_portfolio() {
        let result = committee_budget(5, 0.05, 0.3, 0.2, 0.4, 0);
        assert_eq!(
            result,
            Err(BudgetError::PortfolioTooSmall { portfolio_size: 0 })
        );
    }

    #[test]
    fn test_reject_delta_out_of_range() {
        assert_eq!(
            committee_budget(5, 0.0, 0.3, 0.2, 0.4, 2),
            Err(BudgetError::DeltaOutOfRange { delta: 0.0 })
        );
        assert_eq!(
            committee_budget(5, 1.0, 0.3, 0.2, 0.4, 2),
            Err(BudgetError::DeltaOutOfRange { delta: 1.0 })
        );
        assert_eq!(
            committee_budget(5, -0.1, 0.3, 0.2, 0.4, 2),
            Err(BudgetError::DeltaOutOfRange { delta: -0.1 })
        );
    }

    #[test]
    fn test_reject_alpha_out_of_range() {
        assert_eq!(
            committee_budget(5, 0.05, 0.0, 0.2, 0.4, 2),
            Err(BudgetError::AlphaOutOfRange { alpha: 0.0 })
        );
        assert_eq!(
            committee_budget(5, 0.05, 1.5, 0.2, 0.4, 2),
            Err(BudgetError::AlphaOutOfRange { alpha: 1.5 })
        );
    }

    #[test]
    fn test_reject_beta_out_of_range() {
        assert_eq!(
            committee_budget(5, 0.05, 0.3, 0.0, 0.4, 2),
            Err(BudgetError::BetaOutOfRange { beta: 0.0 })
        );
        assert_eq!(
            committee_budget(5, 0.05, 0.3, 2.0, 0.4, 2),
            Err(BudgetError::BetaOutOfRange { beta: 2.0 })
        );
    }

    #[test]
    fn test_reject_sigma_out_of_range() {
        assert_eq!(
            committee_budget(5, 0.05, 0.3, 0.2, 0.0, 2),
            Err(BudgetError::SigmaOutOfRange { sigma: 0.0 })
        );
        assert_eq!(
            committee_budget(5, 0.05, 0.3, 0.2, -0.1, 2),
            Err(BudgetError::SigmaOutOfRange { sigma: -0.1 })
        );
    }

    #[test]
    fn test_alpha_one_is_valid() {
        // alpha=1.0 is the boundary — should succeed
        let budget =
            committee_budget(5, 0.05, 1.0, 0.5, 0.5, 2).expect("alpha=1.0 should be valid");
        assert!(budget.k >= 2);
    }

    #[test]
    fn test_budget_equality() {
        let a = CommitteeBudget { k: 3, m: 2, r: 1 };
        let b = CommitteeBudget { k: 3, m: 2, r: 1 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_budget_deterministic() {
        // Same inputs → same outputs
        let a = committee_budget(10, 0.05, 0.3, 0.2, 0.4, 4).unwrap();
        let b = committee_budget(10, 0.05, 0.3, 0.2, 0.4, 4).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_total_role_calls_scales_with_depth() {
        let budget = CommitteeBudget { k: 2, m: 3, r: 4 };
        let calls_d1 = budget.total_role_calls(1);
        let calls_d5 = budget.total_role_calls(5);
        assert!(
            (calls_d5 - calls_d1) as f64 > EPS,
            "deeper trees should need more calls"
        );
        assert_eq!(calls_d5, calls_d1 * 5, "should scale linearly with depth");
    }

    #[test]
    fn test_budget_reasonable_range() {
        // Paper-like parameters should give sensible (not astronomical) values
        let budget = committee_budget(10, 0.05, 0.3, 0.3, 0.3, 4).expect("valid");
        assert!(budget.k < 1000, "k should be reasonable, got {}", budget.k);
        assert!(budget.m < 1000, "m should be reasonable, got {}", budget.m);
        assert!(budget.r < 1000, "r should be reasonable, got {}", budget.r);
    }
}
