//! Safe Exploration Budget — configurable verification tier limits (Plan 211 F3).
//!
//! Limits how many Tier 0 (DFA), Tier 1 (AST parse), and Tier 2 (cargo check)
//! verifications can run. When budget is exhausted, falls back to conservative mode
//! (Tier 0 DFA only, no speculative exploration).
//!
//! # Architecture
//!
//! ```text
//! ExplorationBudgetConfig ──► ExplorationBudget
//!                                    │
//!       verify(tier) ◄───────────────┤
//!            │                       │
//!      ┌─────┴──────┐               │
//!   Some(result)   None             │
//!      │            │               │
//!   continue    exhausted ──► conservative_mode = true
//! ```
//!
//! # Feature Gate
//!
//! `safe_exploration_budget` (depends on `three_mode_router`).

use std::env;

// ── Verification Tier ─────────────────────────────────────────

/// Verification tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VerificationTier {
    /// DFA bracket balance checks — O(1) per token.
    Tier0,
    /// AST parse checks — moderate cost.
    Tier1,
    /// Cargo check in sandbox — expensive.
    Tier2,
}

// ── Verification Result ───────────────────────────────────────

/// Result of a verification attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VerificationResult {
    /// Verification passed.
    Pass,
    /// Verification failed.
    Fail,
    /// Budget exhausted — conservative mode.
    BudgetExhausted,
}

// ── Exploration Budget ────────────────────────────────────────

/// Exploration budget with per-tier limits.
///
/// Tracks remaining verifications per tier. When Tier 2 is exhausted,
/// enters conservative mode (Tier 0 DFA only, no speculative exploration).
#[derive(Debug, Clone)]
#[repr(C)]
pub struct ExplorationBudget {
    /// DFA bracket balance checks remaining (default: u32::MAX = unlimited).
    pub tier0_remaining: u32,
    /// AST parse checks remaining (default: 1000).
    pub tier1_remaining: u32,
    /// Cargo check in sandbox remaining (default: 100).
    pub tier2_remaining: u32,
    /// Set when Tier 2 budget exhausted.
    pub conservative_mode: bool,
}

impl ExplorationBudget {
    /// Create a new budget from config.
    pub fn new(config: &ExplorationBudgetConfig) -> Self {
        Self {
            tier0_remaining: config.tier0_limit,
            tier1_remaining: config.tier1_limit,
            tier2_remaining: config.tier2_limit,
            conservative_mode: false,
        }
    }

    /// Attempt verification at the given tier.
    ///
    /// Decrement the appropriate tier counter. Returns `None` when the
    /// tier is exhausted, signalling conservative fallback.
    ///
    /// When `tier2_remaining` reaches 0, sets `conservative_mode = true`.
    pub fn verify(&mut self, tier: VerificationTier) -> Option<VerificationResult> {
        match tier {
            VerificationTier::Tier0 => {
                if self.tier0_remaining == 0 {
                    return None;
                }
                self.tier0_remaining -= 1;
                Some(VerificationResult::Pass)
            }
            VerificationTier::Tier1 => {
                if self.tier1_remaining == 0 {
                    return None;
                }
                self.tier1_remaining -= 1;
                // In conservative mode, skip Tier 1 entirely.
                if self.conservative_mode {
                    return Some(VerificationResult::BudgetExhausted);
                }
                Some(VerificationResult::Pass)
            }
            VerificationTier::Tier2 => {
                if self.tier2_remaining == 0 {
                    self.conservative_mode = true;
                    return None;
                }
                self.tier2_remaining -= 1;
                if self.tier2_remaining == 0 {
                    self.conservative_mode = true;
                }
                Some(VerificationResult::Pass)
            }
        }
    }

    /// Check whether a tier has remaining budget (no side effects).
    pub fn has_budget(&self, tier: VerificationTier) -> bool {
        match tier {
            VerificationTier::Tier0 => self.tier0_remaining > 0,
            VerificationTier::Tier1 => self.tier1_remaining > 0 && !self.conservative_mode,
            VerificationTier::Tier2 => self.tier2_remaining > 0,
        }
    }

    /// Reset budget to a new config.
    pub fn reset(&mut self, config: &ExplorationBudgetConfig) {
        self.tier0_remaining = config.tier0_limit;
        self.tier1_remaining = config.tier1_limit;
        self.tier2_remaining = config.tier2_limit;
        self.conservative_mode = false;
    }
}

impl Default for ExplorationBudget {
    fn default() -> Self {
        Self::new(&ExplorationBudgetConfig::default_limits())
    }
}

// ── Config ────────────────────────────────────────────────────

/// User-configurable limits for the exploration budget.
#[derive(Debug, Clone)]
pub struct ExplorationBudgetConfig {
    /// DFA bracket balance checks (default: u32::MAX = unlimited).
    pub tier0_limit: u32,
    /// AST parse checks (default: 1000).
    pub tier1_limit: u32,
    /// Cargo check in sandbox (default: 100).
    pub tier2_limit: u32,
}

impl ExplorationBudgetConfig {
    /// Sensible defaults: Tier 0 unlimited, Tier 1 moderate, Tier 2 limited.
    pub fn default_limits() -> Self {
        Self {
            tier0_limit: u32::MAX,
            tier1_limit: 1000,
            tier2_limit: 100,
        }
    }

    /// Read limits from environment variables.
    ///
    /// - `KATGPT_TIER0_LIMIT`: Tier 0 limit (default: u32::MAX)
    /// - `KATGPT_TIER1_LIMIT`: Tier 1 limit (default: 1000)
    /// - `KATGPT_TIER2_LIMIT`: Tier 2 limit (default: 100)
    pub fn from_env() -> Self {
        let tier0_limit = env::var("KATGPT_TIER0_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(u32::MAX);
        let tier1_limit = env::var("KATGPT_TIER1_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1000);
        let tier2_limit = env::var("KATGPT_TIER2_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(100);
        Self {
            tier0_limit,
            tier1_limit,
            tier2_limit,
        }
    }
}

impl Default for ExplorationBudgetConfig {
    fn default() -> Self {
        Self::default_limits()
    }
}

// ── Budget Check Helper ───────────────────────────────────────

/// Check if a verification can proceed. Logs warning on budget exhaustion.
///
/// Returns `true` if the tier has budget remaining, `false` if exhausted.
pub fn check_budget(budget: &mut ExplorationBudget, tier: VerificationTier) -> bool {
    match budget.verify(tier) {
        Some(_) => true,
        None => {
            log::warn!(
                "Exploration budget exhausted for {:?} — entering conservative mode",
                tier
            );
            false
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── F3.6: Budget limits respected ─────────────────────────

    #[test]
    fn tier2_budget_limits_respected() {
        let config = ExplorationBudgetConfig {
            tier0_limit: u32::MAX,
            tier1_limit: 1000,
            tier2_limit: 3,
        };
        let mut budget = ExplorationBudget::new(&config);

        let mut passes = 0u32;
        let mut exhausted = 0u32;

        for _ in 0..5 {
            match budget.verify(VerificationTier::Tier2) {
                Some(VerificationResult::Pass) => passes += 1,
                None => exhausted += 1,
                _ => {}
            }
        }

        assert_eq!(passes, 3, "Expected 3 passes, got {passes}");
        assert_eq!(exhausted, 2, "Expected 2 exhausted, got {exhausted}");
    }

    // ── F3.7: Conservative mode ───────────────────────────────

    #[test]
    fn conservative_mode_after_tier2_exhausted() {
        let config = ExplorationBudgetConfig {
            tier0_limit: u32::MAX,
            tier1_limit: 1000,
            tier2_limit: 2,
        };
        let mut budget = ExplorationBudget::new(&config);

        // Use up Tier 2 budget
        assert!(budget.verify(VerificationTier::Tier2).is_some());
        assert!(!budget.conservative_mode);
        assert!(budget.verify(VerificationTier::Tier2).is_some());
        assert!(
            budget.conservative_mode,
            "conservative_mode should be true after Tier 2 exhausted"
        );

        // Tier 0 still works in conservative mode
        let t0 = budget.verify(VerificationTier::Tier0);
        assert!(
            t0.is_some(),
            "Tier 0 should still work in conservative mode"
        );
        assert_eq!(t0, Some(VerificationResult::Pass));
    }

    #[test]
    fn tier1_skipped_in_conservative_mode() {
        let config = ExplorationBudgetConfig {
            tier0_limit: u32::MAX,
            tier1_limit: 1000,
            tier2_limit: 1,
        };
        let mut budget = ExplorationBudget::new(&config);

        // Exhaust Tier 2 to trigger conservative mode
        budget.verify(VerificationTier::Tier2);
        assert!(budget.conservative_mode);

        // Tier 1 returns BudgetExhausted in conservative mode
        let result = budget.verify(VerificationTier::Tier1);
        assert_eq!(
            result,
            Some(VerificationResult::BudgetExhausted),
            "Tier 1 should return BudgetExhausted in conservative mode"
        );
    }

    // ── Exhaustion returns None ────────────────────────────────

    #[test]
    fn exhausted_tier_returns_none() {
        let config = ExplorationBudgetConfig {
            tier0_limit: 1,
            tier1_limit: 1,
            tier2_limit: 1,
        };
        let mut budget = ExplorationBudget::new(&config);

        // Use up each tier
        assert!(budget.verify(VerificationTier::Tier0).is_some());
        assert!(budget.verify(VerificationTier::Tier1).is_some());
        assert!(budget.verify(VerificationTier::Tier2).is_some());

        // Now each should return None
        assert!(budget.verify(VerificationTier::Tier0).is_none());
        assert!(budget.verify(VerificationTier::Tier1).is_none());
        assert!(budget.verify(VerificationTier::Tier2).is_none());
    }

    // ── has_budget ────────────────────────────────────────────

    #[test]
    fn has_budget_reflects_remaining() {
        let config = ExplorationBudgetConfig {
            tier0_limit: 1,
            tier1_limit: 1,
            tier2_limit: 1,
        };
        let mut budget = ExplorationBudget::new(&config);

        assert!(budget.has_budget(VerificationTier::Tier0));
        assert!(budget.has_budget(VerificationTier::Tier1));
        assert!(budget.has_budget(VerificationTier::Tier2));

        budget.verify(VerificationTier::Tier0);
        assert!(!budget.has_budget(VerificationTier::Tier0));

        budget.verify(VerificationTier::Tier2);
        assert!(!budget.has_budget(VerificationTier::Tier2));
        assert!(budget.conservative_mode);
        // Tier 1 should report no budget in conservative mode
        assert!(!budget.has_budget(VerificationTier::Tier1));
    }

    // ── Reset ─────────────────────────────────────────────────

    #[test]
    fn reset_restores_budget() {
        let config = ExplorationBudgetConfig {
            tier0_limit: 5,
            tier1_limit: 5,
            tier2_limit: 5,
        };
        let mut budget = ExplorationBudget::new(&config);

        // Exhaust everything
        for _ in 0..5 {
            budget.verify(VerificationTier::Tier0);
            budget.verify(VerificationTier::Tier1);
            budget.verify(VerificationTier::Tier2);
        }
        assert!(budget.conservative_mode);

        let new_config = ExplorationBudgetConfig {
            tier0_limit: 10,
            tier1_limit: 10,
            tier2_limit: 10,
        };
        budget.reset(&new_config);

        assert_eq!(budget.tier0_remaining, 10);
        assert_eq!(budget.tier1_remaining, 10);
        assert_eq!(budget.tier2_remaining, 10);
        assert!(!budget.conservative_mode);
    }

    // ── Default config ────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let config = ExplorationBudgetConfig::default_limits();
        assert_eq!(config.tier0_limit, u32::MAX);
        assert_eq!(config.tier1_limit, 1000);
        assert_eq!(config.tier2_limit, 100);
    }

    // ── Default budget ────────────────────────────────────────

    #[test]
    fn default_budget_matches_config() {
        let budget = ExplorationBudget::default();
        assert_eq!(budget.tier0_remaining, u32::MAX);
        assert_eq!(budget.tier1_remaining, 1000);
        assert_eq!(budget.tier2_remaining, 100);
        assert!(!budget.conservative_mode);
    }

    // ── check_budget helper ───────────────────────────────────

    #[test]
    fn check_budget_returns_true_when_available() {
        let mut budget = ExplorationBudget::default();
        assert!(check_budget(&mut budget, VerificationTier::Tier0));
        assert!(check_budget(&mut budget, VerificationTier::Tier1));
        assert!(check_budget(&mut budget, VerificationTier::Tier2));
    }

    #[test]
    fn check_budget_returns_false_when_exhausted() {
        let config = ExplorationBudgetConfig {
            tier0_limit: 0,
            tier1_limit: 0,
            tier2_limit: 0,
        };
        let mut budget = ExplorationBudget::new(&config);
        assert!(!check_budget(&mut budget, VerificationTier::Tier0));
        assert!(!check_budget(&mut budget, VerificationTier::Tier1));
        assert!(!check_budget(&mut budget, VerificationTier::Tier2));
    }

    // ── F3.8: Benchmark — tier escalation overhead ────────────

    #[test]
    fn bench_tier0_overhead_o1() {
        let mut budget = ExplorationBudget::default();

        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            let _ = budget.verify(VerificationTier::Tier0);
        }
        let elapsed = start.elapsed();

        let ns_per_call = elapsed.as_nanos() as f64 / 100_000.0;
        // Tier 0 should be O(1) — well under 1μs per call
        assert!(
            ns_per_call < 1000.0,
            "Tier 0 verify should be < 1μs, took {ns_per_call:.1}ns"
        );
    }

    #[test]
    fn bench_tier_escalation_overhead() {
        let mut budget = ExplorationBudget::default();

        // Tier 0
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = budget.verify(VerificationTier::Tier0);
        }
        let t0 = start.elapsed();

        // Tier 1
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = budget.verify(VerificationTier::Tier1);
        }
        let t1 = start.elapsed();

        // Tier 2
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = budget.verify(VerificationTier::Tier2);
        }
        let t2 = start.elapsed();

        // All tiers should be sub-microsecond per call (just counter decrement)
        let ns_t0 = t0.as_nanos() as f64 / 10_000.0;
        let ns_t1 = t1.as_nanos() as f64 / 10_000.0;
        let ns_t2 = t2.as_nanos() as f64 / 10_000.0;

        assert!(ns_t0 < 1000.0, "Tier 0: {ns_t0:.1}ns/call");
        assert!(ns_t1 < 1000.0, "Tier 1: {ns_t1:.1}ns/call");
        assert!(ns_t2 < 1000.0, "Tier 2: {ns_t2:.1}ns/call");
    }
}
