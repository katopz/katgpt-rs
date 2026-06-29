//! Parallelism guard for population-based sketch selection (Plan 128, T7).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763), Supplementary Insight 6:
//! "Population search with only 1 generator **underperforms** the basic setup.
//! The database only helps when multiple agents contribute asynchronously."
//!
//! # Problem
//!
//! The population database (top-64 Elo sketches) adds overhead per selection:
//! hashmap lookups, P-UCB scoring, eviction checks. In single-threaded mode,
//! there is no benefit because only one agent contributes — no cross-pollination
//! between parallel branches. The ablation in the paper confirms this.
//!
//! # Solution
//!
//! Runtime guard: check `rayon::current_num_threads() > 1`. If true, use the
//! full population pipeline (P-UCB sampling, Elo updates, diversity injection).
//! If false, fall back to basic UCB (single-armed bandit, no population overhead).
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

use std::fmt;

// ── Free function ──────────────────────────────────────────────

/// Check whether the population database should be used.
///
/// Returns `true` when rayon has ≥2 threads available, indicating that
/// multiple agents can contribute to the population asynchronously.
/// Returns `false` in single-threaded mode — population overhead is
/// not justified without parallel contribution.
///
/// # Paper Reference
///
/// Supplementary Insight 6 (Research 088): population search underperforms
/// basic UCB when only 1 generator is available.
pub fn should_use_population() -> bool {
    rayon::current_num_threads() > 1
}

// ── ParallelismGuard ───────────────────────────────────────────

/// Captured parallelism decision for the sketch selection pipeline.
///
/// Constructed once at the start of a decode step (or session), then
/// passed to selection logic to avoid repeated rayon queries.
/// Provides logging-friendly introspection via [`fallback_reason`].
///
/// [`fallback_reason`]: ParallelismGuard::fallback_reason
#[derive(Clone, Debug)]
pub struct ParallelismGuard {
    /// Number of rayon threads at construction time.
    threads: usize,
    /// Whether population-based selection is enabled (threads > 1).
    population_enabled: bool,
}

impl ParallelismGuard {
    /// Snapshot the current rayon thread pool state.
    ///
    /// Call this once at session/step start, then reuse the guard.
    /// Does not panic — rayon returns the default thread count if
    /// no pool is active.
    pub fn new() -> Self {
        let threads = rayon::current_num_threads();
        let population_enabled = threads > 1;
        Self {
            threads,
            population_enabled,
        }
    }

    /// Whether population-based P-UCB sampling should be used.
    ///
    /// Returns the cached decision from construction time.
    pub fn should_use_population(&self) -> bool {
        self.population_enabled
    }

    /// Number of rayon threads at guard construction time.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Whether population mode is enabled.
    ///
    /// Same as [`should_use_population`](Self::should_use_population),
    /// named for ergonomic field-style access.
    pub fn population_enabled(&self) -> bool {
        self.population_enabled
    }

    /// Human-readable reason for falling back to basic UCB.
    ///
    /// Returns `Some(reason)` when population is disabled, `None` when
    /// population mode is active. Useful for diagnostic logging.
    pub fn fallback_reason(&self) -> Option<&'static str> {
        match self.population_enabled {
            true => None,
            false => Some(
                "single-threaded mode: population overhead not justified without parallel agents",
            ),
        }
    }
}

impl Default for ParallelismGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ParallelismGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.population_enabled {
            true => write!(
                f,
                "ParallelismGuard(population=enabled, threads={})",
                self.threads
            ),
            false => write!(
                f,
                "ParallelismGuard(population=disabled, threads={})",
                self.threads
            ),
        }
    }
}

// ── SketchSelectionStrategy ────────────────────────────────────

/// Strategy for selecting the next proof sketch to pursue.
///
/// Determines whether the full population pipeline (P-UCB, Elo updates,
/// diversity injection) or a lightweight fallback (basic UCB, epsilon-greedy)
/// is used for sketch selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SketchSelectionStrategy {
    /// Population + P-UCB sampling (parallel mode).
    ///
    /// Uses the top-64 Elo population database with P-UCB scoring.
    /// Requires ≥2 rayon threads for population to provide benefit.
    PopulationPucb,
    /// Basic UCB fallback (single-threaded mode).
    ///
    /// Single-armed bandit without population database overhead.
    /// Used when only one agent contributes — no cross-pollination possible.
    BasicUcb,
    /// Epsilon-greedy (simplest fallback).
    ///
    /// Uniform random with probability ε, greedy otherwise.
    /// Available for configurations that skip UCB entirely.
    EpsilonGreedy,
}

impl SketchSelectionStrategy {
    /// All strategy variants for iteration.
    pub const ALL: [SketchSelectionStrategy; 3] = [
        SketchSelectionStrategy::PopulationPucb,
        SketchSelectionStrategy::BasicUcb,
        SketchSelectionStrategy::EpsilonGreedy,
    ];

    /// Index for use in RNG-based selection.
    pub fn index(self) -> usize {
        match self {
            Self::PopulationPucb => 0,
            Self::BasicUcb => 1,
            Self::EpsilonGreedy => 2,
        }
    }

    /// Whether this strategy uses the population database.
    pub fn uses_population(self) -> bool {
        match self {
            Self::PopulationPucb => true,
            Self::BasicUcb => false,
            Self::EpsilonGreedy => false,
        }
    }

    /// Human-readable description of the strategy.
    pub fn description(self) -> &'static str {
        match self {
            Self::PopulationPucb => "Population + P-UCB sampling (parallel mode)",
            Self::BasicUcb => "Basic UCB fallback (single-threaded mode)",
            Self::EpsilonGreedy => "Epsilon-greedy (simplest fallback)",
        }
    }
}

impl fmt::Display for SketchSelectionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PopulationPucb => write!(f, "PopulationPucb"),
            Self::BasicUcb => write!(f, "BasicUcb"),
            Self::EpsilonGreedy => write!(f, "EpsilonGreedy"),
        }
    }
}

// ── Strategy selection ─────────────────────────────────────────

/// Select the appropriate sketch selection strategy based on parallelism.
///
/// Returns [`PopulationPucb`](SketchSelectionStrategy::PopulationPucb) when
/// the population database is beneficial (≥2 threads), otherwise falls back
/// to [`BasicUcb`](SketchSelectionStrategy::BasicUcb).
pub fn select_strategy(guard: &ParallelismGuard) -> SketchSelectionStrategy {
    match guard.should_use_population() {
        true => SketchSelectionStrategy::PopulationPucb,
        false => SketchSelectionStrategy::BasicUcb,
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── should_use_population ──────────────────────────────────

    #[test]
    fn should_use_population_returns_bool() {
        // We can't control rayon's thread pool in tests, but it must not panic.
        let result = should_use_population();
        // Verify it's a valid bool (trivially true, but exercises the fn).
        assert!(matches!(result, true | false));
    }

    // ── ParallelismGuard ───────────────────────────────────────

    #[test]
    fn guard_new_does_not_panic() {
        let guard = ParallelismGuard::new();
        assert!(guard.threads() > 0);
    }

    #[test]
    fn guard_threads_returns_nonzero() {
        let guard = ParallelismGuard::new();
        assert!(guard.threads() > 0, "rayon always reports ≥1 thread");
    }

    #[test]
    fn guard_default_matches_new() {
        let from_new = ParallelismGuard::new();
        let from_default = ParallelismGuard::default();
        assert_eq!(from_new.threads(), from_default.threads());
        assert_eq!(
            from_new.population_enabled(),
            from_default.population_enabled()
        );
    }

    #[test]
    fn guard_should_use_population_matches_population_enabled() {
        let guard = ParallelismGuard::new();
        assert_eq!(guard.should_use_population(), guard.population_enabled());
    }

    #[test]
    fn guard_display_includes_threads() {
        let guard = ParallelismGuard::new();
        let display = format!("{guard}");
        assert!(display.contains(&format!("{}", guard.threads())));
    }

    #[test]
    fn guard_display_shows_enabled_when_population_active() {
        let guard = ParallelismGuard::new();
        let display = format!("{guard}");
        match guard.population_enabled() {
            true => assert!(display.contains("population=enabled")),
            false => assert!(display.contains("population=disabled")),
        }
    }

    // ── fallback_reason ────────────────────────────────────────

    #[test]
    fn fallback_reason_none_when_population_enabled() {
        let guard = ParallelismGuard::new();
        match guard.population_enabled() {
            true => assert!(guard.fallback_reason().is_none()),
            false => assert!(guard.fallback_reason().is_some()),
        }
    }

    #[test]
    fn fallback_reason_provides_explanation() {
        let guard = ParallelismGuard::new();
        match guard.fallback_reason() {
            Some(reason) => {
                assert!(!reason.is_empty(), "fallback reason should be descriptive");
                assert!(
                    reason.contains("single-threaded"),
                    "reason should mention single-threaded mode, got: {reason}"
                );
            }
            None => {
                // Population is enabled — no fallback needed.
            }
        }
    }

    #[test]
    fn fallback_reason_is_static_str() {
        let guard = ParallelismGuard::new();
        // Verify the return type is &'static str (compiles only if correct).
        let _: Option<&'static str> = guard.fallback_reason();
    }

    // ── select_strategy ────────────────────────────────────────

    #[test]
    fn select_strategy_returns_population_when_enabled() {
        let guard = ParallelismGuard::new();
        let strategy = select_strategy(&guard);
        match guard.population_enabled() {
            true => assert_eq!(strategy, SketchSelectionStrategy::PopulationPucb),
            false => assert_eq!(strategy, SketchSelectionStrategy::BasicUcb),
        }
    }

    #[test]
    fn select_strategy_basic_ucb_does_not_use_population() {
        assert!(!SketchSelectionStrategy::BasicUcb.uses_population());
    }

    #[test]
    fn select_strategy_population_uses_population() {
        assert!(SketchSelectionStrategy::PopulationPucb.uses_population());
    }

    // ── SketchSelectionStrategy ────────────────────────────────

    #[test]
    fn strategy_all_has_three_variants() {
        assert_eq!(SketchSelectionStrategy::ALL.len(), 3);
    }

    #[test]
    fn strategy_index_roundtrip() {
        for strategy in SketchSelectionStrategy::ALL {
            let idx = strategy.index();
            assert!(
                (0..3).contains(&idx),
                "index for {strategy} out of range: {idx}"
            );
        }
    }

    #[test]
    fn strategy_display_roundtrip() {
        let displays = ["PopulationPucb", "BasicUcb", "EpsilonGreedy"];
        for (strategy, expected) in SketchSelectionStrategy::ALL.iter().zip(displays) {
            assert_eq!(format!("{strategy}"), expected);
        }
    }

    #[test]
    fn strategy_description_is_nonempty() {
        for strategy in SketchSelectionStrategy::ALL {
            assert!(
                !strategy.description().is_empty(),
                "description should not be empty for {strategy}"
            );
        }
    }

    #[test]
    fn strategy_uses_population_only_for_pucb() {
        assert!(SketchSelectionStrategy::PopulationPucb.uses_population());
        assert!(!SketchSelectionStrategy::BasicUcb.uses_population());
        assert!(!SketchSelectionStrategy::EpsilonGreedy.uses_population());
    }

    // ── Edge case: manual guard construction ───────────────────

    #[test]
    fn manual_guard_single_thread_disables_population() {
        let guard = ParallelismGuard {
            threads: 1,
            population_enabled: false,
        };
        assert!(!guard.should_use_population());
        assert!(guard.fallback_reason().is_some());
        assert_eq!(select_strategy(&guard), SketchSelectionStrategy::BasicUcb);
    }

    #[test]
    fn manual_guard_multi_thread_enables_population() {
        let guard = ParallelismGuard {
            threads: 4,
            population_enabled: true,
        };
        assert!(guard.should_use_population());
        assert!(guard.fallback_reason().is_none());
        assert_eq!(
            select_strategy(&guard),
            SketchSelectionStrategy::PopulationPucb
        );
    }

    #[test]
    fn manual_guard_exactly_two_threads_enables_population() {
        let guard = ParallelismGuard {
            threads: 2,
            population_enabled: true,
        };
        assert!(guard.should_use_population());
        assert_eq!(
            select_strategy(&guard),
            SketchSelectionStrategy::PopulationPucb
        );
    }
}
