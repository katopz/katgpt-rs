//! Committee Boost — Oracle-Gap Recovery, Debiasing, Budget Sizing (Plan 132).
//!
//! Four diagnostics from the boosting committee paper (arXiv:2605.14163):
//!
//! 1. **Oracle-gap recovery** — `Rec = (p_system - p1) / (p_oracle - p1)` measures
//!    whether failures are selection-limited or coverage-limited.
//! 2. **Position-swap debiasing** — Compare pairs in both A/B orders; count win
//!    only if both agree. Eliminates lead-position bias in BtRank.
//! 3. **Budget sizing** — Given (α₀, β₀, σ₀, L, δ), compute optimal (k, m, r)
//!    per paper Theorem 3.
//! 4. **Blind-spot floor** — `B = 1 - lim_{k→∞} p_oracle(k)` measures the
//!    proposer diversity ceiling.
//!
//! Our DDTree + BtRank + ScreeningPruner stack IS the committee protocol Π_{k,m,r}.
//! These additions make theoretical guarantees **measurable and actionable**.
//!
//! **Feature gate:** `committee_boost` (opt-in, requires `bt_rank` + `bandit`)

pub mod blind_spot;
pub mod budget;
pub mod debiased_compare;
pub mod types;

// ── Phase 1: Oracle-Gap Recovery ──────────────────────────────

pub use types::{FailureMode, OracleGapRecovery};

// ── Phase 2: Position-Swap Debiasing ──────────────────────────

pub use debiased_compare::{DebiasedComparator, debiased_compare};

// ── Phase 3: Budget Sizing ────────────────────────────────────

pub use budget::{BudgetError, CommitteeBudget, committee_budget};

// ── Phase 4: Blind-Spot Floor ─────────────────────────────────

pub use blind_spot::{
    BlindSpotEstimate, ConvergenceFit, CoverageAction, CoverageDiagnostic, coverage_diagnostic,
    estimate_blind_spot_floor, fit_convergence,
};
