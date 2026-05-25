//! Committee Boost core types — oracle-gap recovery, failure modes, diagnostics.
//!
//! Types for measuring and diagnosing inference-time committee (boosting) performance:
//! - **OracleGapRecovery**: how much latent capability the selector recovers
//! - **FailureMode**: whether failures are selection-limited or coverage-limited
//! - **CommitteeBudget**: theoretical (k, m, r) sizing from paper Theorem 3
//! - **BlindSpotEstimate**: proposer diversity ceiling from oracle best-of-k curve
//!
//! Reference: arXiv:2605.14163 — Verifier-backed committee search as boosting.

use core::fmt;

// ── Failure Mode ──────────────────────────────────────────────

/// Failure mode diagnosis from oracle-gap recovery fraction.
///
/// Indicates whether the committee bottleneck is in the **selector**
/// (critic/comparator not finding the best candidate) or in the
/// **proposer** (not enough diverse candidates to cover the answer space).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FailureMode {
    /// Recovery > 0.7: selector works well, proposer needs more diversity.
    CoverageLimited,
    /// Recovery < 0.3: selector struggles, improve critic/comparator.
    SelectionLimited,
    /// 0.3 ≤ Recovery ≤ 0.7: both need improvement.
    Mixed,
}

impl fmt::Display for FailureMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoverageLimited => write!(f, "coverage-limited (diversify proposers)"),
            Self::SelectionLimited => write!(f, "selection-limited (improve critic/comparator)"),
            Self::Mixed => write!(f, "mixed (improve both proposer diversity and selector)"),
        }
    }
}

// ── Oracle-Gap Recovery ───────────────────────────────────────

/// Oracle-gap recovery metric: how much latent capability the selector recovers.
///
/// Given three accuracy measurements:
/// - `p1`: single-shot accuracy (Pass@1, no committee)
/// - `p_oracle`: best-of-k with a perfect selector (oracle ceiling)
/// - `p_system`: deployed harness accuracy (our DDTree + BtRank stack)
///
/// Recovery `Rec = (p_system - p1) / (p_oracle - p1)` tells us:
/// - `Rec → 1.0`: selector recovers nearly all latent capability
/// - `Rec → 0.0`: selector adds nothing beyond single-shot
/// - `Rec < 0.0`: selector is worse than single-shot (broken)
#[derive(Debug, Clone)]
pub struct OracleGapRecovery {
    /// Single-shot accuracy (Pass@1).
    pub p1: f64,
    /// Best-of-k with perfect selector (oracle ceiling).
    pub p_oracle: f64,
    /// Deployed system accuracy.
    pub p_system: f64,
}

impl OracleGapRecovery {
    /// Create a new oracle-gap recovery metric.
    ///
    /// Values are clamped to `[0.0, 1.0]`.
    pub fn new(p1: f64, p_oracle: f64, p_system: f64) -> Self {
        Self {
            p1: p1.clamp(0.0, 1.0),
            p_oracle: p_oracle.clamp(0.0, 1.0),
            p_system: p_system.clamp(0.0, 1.0),
        }
    }

    /// Compute recovery fraction: `(p_system - p1) / (p_oracle - p1)`.
    ///
    /// Returns `None` when the gap is zero (p_oracle == p1), meaning
    /// the oracle ceiling equals single-shot — committee cannot help.
    /// Returns `0.0` when p_system == p1 (no improvement).
    /// Returns `1.0` when p_system == p_oracle (perfect recovery).
    pub fn recovery(&self) -> Option<f64> {
        let gap = self.p_oracle - self.p1;
        if gap.abs() < f64::EPSILON {
            return None;
        }
        let rec = (self.p_system - self.p1) / gap;
        Some(rec.clamp(-1.0, 1.0))
    }

    /// Diagnose failure mode from recovery fraction.
    ///
    /// Thresholds from paper Section 4 analysis:
    /// - `Rec > 0.7` → `CoverageLimited` (selector works, need diverse proposers)
    /// - `Rec < 0.3` → `SelectionLimited` (selector struggles)
    /// - Otherwise → `Mixed`
    pub fn failure_mode(&self) -> FailureMode {
        match self.recovery() {
            None => FailureMode::CoverageLimited,
            Some(rec) if rec > 0.7 => FailureMode::CoverageLimited,
            Some(rec) if rec < 0.3 => FailureMode::SelectionLimited,
            Some(_) => FailureMode::Mixed,
        }
    }

    /// Human-readable diagnostic breakdown.
    ///
    /// Example output:
    /// ```text
    /// Recovery=80.0%: selection recovers most latent capability;
    ///   focus on proposer diversity for further gains
    /// ```
    pub fn diagnostic(&self) -> String {
        let mode = self.failure_mode();
        match self.recovery() {
            None => {
                format!(
                    "Recovery=N/A (oracle gap is zero: p1={:.3}, p_oracle={:.3}); \
                     committee cannot improve over single-shot",
                    self.p1, self.p_oracle
                )
            }
            Some(rec) => {
                let pct = rec * 100.0;
                let advice = match mode {
                    FailureMode::CoverageLimited => {
                        "selection recovers most latent capability; \
                         focus on proposer diversity for further gains"
                    }
                    FailureMode::SelectionLimited => {
                        "selector struggles to find the best candidate; \
                         improve critic/comparator quality"
                    }
                    FailureMode::Mixed => {
                        "both selection and coverage need improvement; \
                         increase k and improve comparator simultaneously"
                    }
                };
                format!("Recovery={pct:.1}% ({mode}); {advice}")
            }
        }
    }
}

// ── Committee Budget ──────────────────────────────────────────

/// Committee budget from theoretical sizing rules (paper Theorem 3).
///
/// The committee protocol Π_{k,m,r} uses:
/// - `k` proposer candidates (DDTree width)
/// - `m` critic evaluations per candidate (ScreeningPruner depth)
/// - `r` comparator votes per pair (BtRank repetitions)
///
/// Theoretical guarantees require (k, m, r) to satisfy bounds
/// parameterized by (α₀, β₀, σ₀, L, δ).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitteeBudget {
    /// Proposer width — number of candidates generated per query.
    pub k: usize,
    /// Critic depth — number of evaluations per candidate.
    pub m: usize,
    /// Comparator votes — number of pairwise comparison rounds.
    pub r: usize,
}

impl CommitteeBudget {
    /// Total role calls for a single committee round: `L × (k + m×k + r×k²)`.
    ///
    /// - `k` proposer calls (one per candidate)
    /// - `m × k` critic calls (m evaluations per candidate)
    /// - `r × k²` comparator calls (r votes for each of k×(k-1)/2 pairs, ≈ r×k²)
    pub fn total_role_calls(&self, depth: usize) -> usize {
        let k = self.k;
        let m = self.m;
        let r = self.r;
        depth * (k + m * k + r * k.saturating_mul(k))
    }

    /// Validate budget parameters are sane.
    ///
    /// Returns `Ok(())` if all parameters are ≥ 1.
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("k (proposer width) must be >= 1".to_string());
        }
        if self.m == 0 {
            return Err("m (critic depth) must be >= 1".to_string());
        }
        if self.r == 0 {
            return Err("r (comparator votes) must be >= 1".to_string());
        }
        Ok(())
    }
}

// ── Convergence Fit ───────────────────────────────────────────

/// Exponential convergence fit for oracle best-of-k curve.
///
/// Models `p_oracle(k) ≈ a - b × exp(-c × k)` where:
/// - `a` is the asymptotic ceiling
/// - `b` is the initial gap (a - p_oracle(1))
/// - `c` is the convergence rate
#[derive(Debug, Clone)]
pub struct ConvergenceFit {
    /// Asymptotic ceiling: `lim_{k→∞} p_oracle(k)`.
    pub ceiling: f64,
    /// Initial gap: `ceiling - p_oracle(1)`.
    pub initial_gap: f64,
    /// Convergence rate (higher = faster saturation).
    pub rate: f64,
    /// R² goodness of fit (1.0 = perfect).
    pub r_squared: f64,
}

// ── Blind-Spot Estimate ───────────────────────────────────────

/// Blind-spot floor estimate from oracle best-of-k curve.
///
/// The blind-spot floor `B ≈ 1 - max(p_oracle(k))` represents the
/// fraction of problems that NO committee member can solve —
/// the proposer diversity ceiling.
#[derive(Debug, Clone)]
pub struct BlindSpotEstimate {
    /// Blind-spot floor: fraction of problems no member solves.
    pub blind_spot_floor: f64,
    /// Oracle accuracy at the largest k tested.
    pub oracle_at_max_k: f64,
    /// Largest k tested.
    pub max_k: usize,
    /// Number of data points used.
    pub n_points: usize,
}

// ── Coverage Diagnostic ───────────────────────────────────────

/// Recommended action from coverage analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CoverageAction {
    /// Blind-spot floor is high → diversify proposers (new templates, domains).
    DiversifyProposers,
    /// Residual is large → increase k (more candidates per query).
    IncreaseK,
    /// Coverage is good → focus on selector quality instead.
    ImproveSelector,
}

impl fmt::Display for CoverageAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DiversifyProposers => write!(f, "diversify proposers"),
            Self::IncreaseK => write!(f, "increase k"),
            Self::ImproveSelector => write!(f, "improve selector"),
        }
    }
}

/// Full coverage diagnostic combining blind-spot floor, convergence fit, and recommendation.
#[derive(Debug, Clone)]
pub struct CoverageDiagnostic {
    /// Blind-spot floor estimate.
    pub blind_spot: BlindSpotEstimate,
    /// Exponential convergence fit (if enough data points).
    pub convergence: Option<ConvergenceFit>,
    /// Residual: `1 - blind_spot_floor - oracle_at_max_k` (should be ~0 if converged).
    pub residual: f64,
    /// Recommended action based on analysis.
    pub recommended_action: CoverageAction,
}

impl fmt::Display for CoverageDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "B={:.3} (max_k={}, oracle={:.3}), residual={:.3}, action={}",
            self.blind_spot.blind_spot_floor,
            self.blind_spot.max_k,
            self.blind_spot.oracle_at_max_k,
            self.residual,
            self.recommended_action
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_known_values() {
        // p1=0.5, p_oracle=0.8, p_system=0.74 → Rec = 0.24/0.30 = 0.8
        let r = OracleGapRecovery::new(0.5, 0.8, 0.74);
        let rec = r.recovery().expect("should compute recovery");
        assert!((rec - 0.8).abs() < 1e-9, "expected 0.8, got {rec}");
    }

    #[test]
    fn test_recovery_perfect() {
        let r = OracleGapRecovery::new(0.5, 0.8, 0.8);
        let rec = r.recovery().expect("should compute");
        assert!((rec - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_recovery_no_improvement() {
        let r = OracleGapRecovery::new(0.5, 0.8, 0.5);
        let rec = r.recovery().expect("should compute");
        assert!(rec.abs() < 1e-9);
    }

    #[test]
    fn test_recovery_zero_gap() {
        let r = OracleGapRecovery::new(0.5, 0.5, 0.7);
        assert!(r.recovery().is_none());
    }

    #[test]
    fn test_failure_mode_coverage_limited() {
        let r = OracleGapRecovery::new(0.5, 0.8, 0.75);
        assert_eq!(r.failure_mode(), FailureMode::CoverageLimited);
    }

    #[test]
    fn test_failure_mode_selection_limited() {
        let r = OracleGapRecovery::new(0.5, 0.8, 0.55);
        assert_eq!(r.failure_mode(), FailureMode::SelectionLimited);
    }

    #[test]
    fn test_failure_mode_mixed() {
        let r = OracleGapRecovery::new(0.5, 0.8, 0.65);
        assert_eq!(r.failure_mode(), FailureMode::Mixed);
    }

    #[test]
    fn test_diagnostic_contains_recovery_pct() {
        let r = OracleGapRecovery::new(0.5, 0.8, 0.74);
        let diag = r.diagnostic();
        assert!(diag.contains("80.0%"), "diagnostic: {diag}");
        assert!(diag.contains("coverage-limited"), "diagnostic: {diag}");
    }

    #[test]
    fn test_diagnostic_zero_gap() {
        let r = OracleGapRecovery::new(0.5, 0.5, 0.6);
        let diag = r.diagnostic();
        assert!(diag.contains("N/A"), "diagnostic: {diag}");
    }

    #[test]
    fn test_budget_validate_ok() {
        let b = CommitteeBudget { k: 4, m: 3, r: 2 };
        assert!(b.validate().is_ok());
    }

    #[test]
    fn test_budget_validate_zero_k() {
        let b = CommitteeBudget { k: 0, m: 3, r: 2 };
        assert!(b.validate().is_err());
    }

    #[test]
    fn test_budget_validate_zero_m() {
        let b = CommitteeBudget { k: 4, m: 0, r: 2 };
        assert!(b.validate().is_err());
    }

    #[test]
    fn test_budget_validate_zero_r() {
        let b = CommitteeBudget { k: 4, m: 3, r: 0 };
        assert!(b.validate().is_err());
    }

    #[test]
    fn test_total_role_calls_formula() {
        // k=4, m=3, r=2, depth=1 → 1 × (4 + 3×4 + 2×16) = 4 + 12 + 32 = 48
        let b = CommitteeBudget { k: 4, m: 3, r: 2 };
        assert_eq!(b.total_role_calls(1), 48);
    }

    #[test]
    fn test_total_role_calls_depth_2() {
        // k=2, m=1, r=1, depth=3 → 3 × (2 + 1×2 + 1×4) = 3 × 8 = 24
        let b = CommitteeBudget { k: 2, m: 1, r: 1 };
        assert_eq!(b.total_role_calls(3), 24);
    }

    #[test]
    fn test_failure_mode_display() {
        assert_eq!(
            FailureMode::CoverageLimited.to_string(),
            "coverage-limited (diversify proposers)"
        );
        assert_eq!(
            FailureMode::SelectionLimited.to_string(),
            "selection-limited (improve critic/comparator)"
        );
        assert_eq!(
            FailureMode::Mixed.to_string(),
            "mixed (improve both proposer diversity and selector)"
        );
    }

    #[test]
    fn test_coverage_action_display() {
        assert_eq!(
            CoverageAction::DiversifyProposers.to_string(),
            "diversify proposers"
        );
        assert_eq!(CoverageAction::IncreaseK.to_string(), "increase k");
        assert_eq!(
            CoverageAction::ImproveSelector.to_string(),
            "improve selector"
        );
    }
}
