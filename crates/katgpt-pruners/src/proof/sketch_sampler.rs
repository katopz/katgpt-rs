//! P-UCB sketch sampling + diversity injection (Plan 128, T5).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "P-UCB balances exploitation (high Elo) with exploration (low visits)
//! via an upper confidence bound applied to the normalized Elo rating."
//!
//! # Architecture
//!
//! ```text
//! SketchSampler
//! ├── population: SketchPopulation      // top-64 Elo database
//! ├── c: f64                            // exploration constant (0.2 per paper)
//! ├── epsilon: f64                      // ε-greedy fallback (0.1 per paper)
//! │
//! ├── sample(rng) -> Option<&SketchEntry>
//! │   ├── should_use_population()?  → parallelism guard (rayon threads > 1)
//! │   ├── true  → sample_p_ucb(rng)
//! │   └── false → sample_fallback_ucb(rng)   // basic UCB for single-thread
//! │
//! ├── sample_p_ucb(rng) -> Option<&SketchEntry>
//! │   ├── Compute scores: q + c * sqrt(total / (visits + 1))
//! │   │   where q = normalize(elo, min, max) ∈ [0, 1]
//! │   └── Return entry with highest P-UCB score
//! │
//! ├── sample_epsilon_greedy(rng) -> Option<&SketchEntry>
//! │   ├── ε chance → random entry (explore)
//! │   └── 1-ε chance → best Elo entry (exploit)
//! │
//! └── inject_diversity(rng) -> DiversityHint
//!     ├── 33% → Decompose
//!     ├── 33% → Combine
//!     └── 34% → NovelApproach
//! ```
//!
//! # P-UCB Formula
//!
//! ```text
//! score(s) = q(s) + c × √(N / (n_s + 1))
//!
//! q(s) = normalize_to_01(elo_s, elo_min, elo_max) ∈ [0, 1]
//! N    = Σ n_s (total visits across population)
//! n_s  = visits for sketch s
//! c    = exploration constant (0.2 per paper)
//! ```
//!
//! Unvisited entries get `n_s = 0`, yielding maximal exploration bonus.
//! As visits accumulate, the bonus shrinks, shifting toward exploitation.
//!
//! # Diversity Injection
//!
//! Supplementary Insight 7 (Research 088): The paper's controller
//! stochastically injects structured exploration hints to prevent
//! population collapse into a single lineage. The 33/33/34 split
//! ensures roughly equal coverage of all three strategies.
//!
//! # Parallelism Guard
//!
//! Supplementary Insight 6 (Research 088): Population search with
//! only 1 generator underperforms the basic setup. The database
//! only helps when multiple agents contribute asynchronously.
//! Runtime guard: `should_use_population()` from [`parallelism`] module.
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

use std::fmt;

use fastrand::Rng;

use super::parallelism::should_use_population;
use super::sketch_population::SketchPopulation;
use super::sketch_types::{DiversityHint, DiversityStrategy, SketchEntry, SketchId};

// ── Constants ─────────────────────────────────────────────────

/// Default exploration constant (paper: c = 0.2).
pub const DEFAULT_EXPLORATION_C: f64 = 0.2;

/// Default epsilon for ε-greedy fallback (paper: ε = 0.1).
pub const DEFAULT_EPSILON: f64 = 0.1;

/// Minimum visits required for P-UCB sampling eligibility.
const MIN_SAMPLE_VISITS: usize = 0;

// ── SketchSamplerConfig ───────────────────────────────────────

/// Configuration for [`SketchSampler`].
#[derive(Clone, Debug)]
pub struct SketchSamplerConfig {
    /// Exploration constant for P-UCB (paper: 0.2).
    pub c: f64,
    /// Epsilon for ε-greedy fallback (paper: 0.1).
    pub epsilon: f64,
}

impl SketchSamplerConfig {
    /// Create config with paper defaults (c=0.2, ε=0.1).
    pub const fn paper_defaults() -> Self {
        Self {
            c: DEFAULT_EXPLORATION_C,
            epsilon: DEFAULT_EPSILON,
        }
    }

    /// Create config with custom exploration constant.
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Create config with custom epsilon.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Default for SketchSamplerConfig {
    fn default() -> Self {
        Self::paper_defaults()
    }
}

impl fmt::Display for SketchSamplerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SketchSamplerConfig(c={:.3}, ε={:.3})",
            self.c, self.epsilon
        )
    }
}

// ── SketchSampler ─────────────────────────────────────────────

/// P-UCB sketch sampler with diversity injection.
///
/// Wraps [`SketchPopulation`] and provides structured exploration/exploitation
/// sampling for proof sketch evolution. Uses P-UCB (Predictor - Upper
/// Confidence Bound) to balance high-Elo exploitation with under-explored
/// entry exploration.
///
/// # Sampling Strategies
///
/// - **P-UCB**: Primary strategy. Normalized Elo + exploration bonus.
/// - **ε-greedy**: Fallback. Best Elo with ε random exploration.
/// - **Diversity injection**: Structured hints during explore arm.
///
/// # Thread Safety
///
/// The sampler borrows the population immutably during sampling.
/// Mutation (visit tracking, Elo updates) happens externally via
/// [`SketchPopulation::get_mut`].
pub struct SketchSampler {
    /// The sketch population to sample from.
    population: SketchPopulation,
    /// Sampler configuration.
    config: SketchSamplerConfig,
}

impl SketchSampler {
    /// Create a new sampler wrapping the given population with paper defaults.
    pub fn new(population: SketchPopulation) -> Self {
        Self {
            population,
            config: SketchSamplerConfig::paper_defaults(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(population: SketchPopulation, config: SketchSamplerConfig) -> Self {
        Self { population, config }
    }

    /// Exploration constant `c`.
    pub fn c(&self) -> f64 {
        self.config.c
    }

    /// Epsilon for ε-greedy fallback.
    pub fn epsilon(&self) -> f64 {
        self.config.epsilon
    }

    /// Reference to the underlying population.
    pub fn population(&self) -> &SketchPopulation {
        &self.population
    }

    /// Mutable reference to the underlying population.
    pub fn population_mut(&mut self) -> &mut SketchPopulation {
        &mut self.population
    }

    // ── Primary Sampling ──────────────────────────────────────

    /// Sample a sketch entry using P-UCB with parallelism guard.
    ///
    /// Checks [`should_use_population`] first:
    /// - **true** → P-UCB sampling (evolutionary path, multi-threaded)
    /// - **false** → fallback basic UCB on sorted Elo (single-threaded)
    ///
    /// Returns `None` if population is empty.
    pub fn sample(&self, rng: &mut Rng) -> Option<&SketchEntry> {
        match self.population.is_empty() {
            true => None,
            false => match should_use_population() {
                true => self.sample_p_ucb(rng),
                false => self.sample_fallback_ucb(rng),
            },
        }
    }

    /// Sample a sketch entry using P-UCB with parallelism guard, mutable.
    ///
    /// Same as [`sample`](Self::sample) but returns a mutable reference
    /// for visit tracking.
    pub fn sample_mut(&mut self, rng: &mut Rng) -> Option<&mut SketchEntry> {
        match self.population.is_empty() {
            true => None,
            false => {
                let id = match should_use_population() {
                    true => self.pick_p_ucb_id(rng),
                    false => self.pick_fallback_id(rng),
                };
                match id {
                    Some(id) => self.population.get_mut(&id),
                    None => None,
                }
            }
        }
    }

    // ── P-UCB Sampling ────────────────────────────────────────

    /// P-UCB sampling: returns entry with highest P-UCB score.
    ///
    /// ```text
    /// score = q + c * sqrt(total_visits / (visits + 1))
    /// q = normalize_to_01(elo, elo_min, elo_max)
    /// ```
    ///
    /// Unvisited entries (visits=0) get the maximum exploration bonus,
    /// ensuring they are selected before explored entries.
    pub fn sample_p_ucb(&self, rng: &mut Rng) -> Option<&SketchEntry> {
        // Prefer unvisited entries; fall back to all if none unvisited
        let eligible = self.population.sample_eligible(MIN_SAMPLE_VISITS);
        let candidates: Vec<&SketchEntry> = match eligible.is_empty() {
            true => self.population.sorted_by_elo(),
            false => eligible,
        };

        match candidates.is_empty() {
            true => None,
            false => {
                let total = self.population.total_visits();
                let (elo_min, elo_max) = match self.population.elo_range() {
                    Some(range) => range,
                    None => return candidates.first().copied(),
                };

                let best = candidates.iter().max_by(|a, b| {
                    let score_a = a.p_ucb_score(total, elo_min, elo_max, self.config.c);
                    let score_b = b.p_ucb_score(total, elo_min, elo_max, self.config.c);
                    score_a.total_cmp(&score_b)
                });

                match best {
                    Some(entry) => Some(*entry),
                    None => {
                        // Tiebreak: random selection from candidates
                        let idx = rng.usize(0..candidates.len());
                        Some(candidates[idx])
                    }
                }
            }
        }
    }

    /// Pick the ID of the best P-UCB entry (for mutable access).
    fn pick_p_ucb_id(&self, rng: &mut Rng) -> Option<SketchId> {
        self.sample_p_ucb(rng).map(|e| e.id)
    }

    // ── ε-Greedy Sampling ─────────────────────────────────────

    /// ε-greedy sampling: best Elo with ε probability of random exploration.
    ///
    /// With probability ε, returns a random entry from the population.
    /// Otherwise, returns the entry with the highest Elo rating.
    ///
    /// Returns `None` if population is empty.
    pub fn sample_epsilon_greedy(&self, rng: &mut Rng) -> Option<&SketchEntry> {
        match self.population.is_empty() {
            true => None,
            false => {
                let roll = rng.f64();
                match roll < self.config.epsilon {
                    true => self.sample_random(rng),
                    false => self.sample_best_elo(),
                }
            }
        }
    }

    /// Random entry from the population.
    ///
    /// Draws uniformly by HashMap iteration index — no sort, no allocation.
    /// Previously called `sorted_by_elo()` (O(N log N) sort + Vec alloc)
    /// purely to index by a random integer.
    fn sample_random(&self, rng: &mut Rng) -> Option<&SketchEntry> {
        let len = self.population.len();
        if len == 0 {
            return None;
        }
        let idx = rng.usize(0..len);
        self.population.nth_in_arbitrary_order(idx)
    }

    /// Best Elo entry from the population.
    ///
    /// O(N) single pass via `max_by` — replaces the previous
    /// `sorted_by_elo().first()` which sorted all entries (O(N log N))
    /// and allocated a Vec just to read the first element.
    fn sample_best_elo(&self) -> Option<&SketchEntry> {
        self.population.best_elo()
    }

    // ── Fallback UCB (single-threaded) ────────────────────────

    /// Fallback basic UCB for single-threaded execution.
    ///
    /// Single O(N) pass with a lightweight exploration bonus — no sort, no
    /// allocation. Previously called `sorted_by_elo()` (O(N log N) sort + Vec
    /// alloc) and then immediately `max_by`-ed over the sorted vec, which
    /// discarded the sort ordering anyway.
    fn sample_fallback_ucb(&self, rng: &mut Rng) -> Option<&SketchEntry> {
        let len = self.population.len();
        if len == 0 {
            return None;
        }
        // Simple fallback: prefer best Elo with small random perturbation
        let total = self.population.total_visits();
        let c = self.config.c * 0.5; // reduced exploration for single-thread

        let best = self.population.values_arbitrary().max_by(|a, b| {
            // Basic UCB without Elo normalization
            let score_a = a.elo_rating + c * (total as f64 / (a.visits + 1) as f64).sqrt();
            let score_b = b.elo_rating + c * (total as f64 / (b.visits + 1) as f64).sqrt();
            score_a.total_cmp(&score_b)
        });

        // `values_arbitrary` on a non-empty HashMap always yields at least
        // one element, so `best` is `Some` here. Keep the fallback for safety.
        best.or_else(|| {
            let idx = rng.usize(0..len);
            self.population.nth_in_arbitrary_order(idx)
        })
    }

    /// Pick the ID of the best fallback entry (for mutable access).
    fn pick_fallback_id(&self, rng: &mut Rng) -> Option<SketchId> {
        self.sample_fallback_ucb(rng).map(|e| e.id)
    }

    // ── Diversity Injection ───────────────────────────────────

    /// Inject a diversity hint for structured exploration.
    ///
    /// Randomly selects a [`DiversityStrategy`] with the paper's
    /// 33/33/34 split to prevent population collapse into a
    /// single lineage.
    ///
    /// Call during the explore arm of P-UCB or ε-greedy sampling.
    pub fn inject_diversity(&self, rng: &mut Rng) -> DiversityHint {
        let roll = rng.u32(0..100);
        let strategy = match roll {
            0..=32 => DiversityStrategy::Decompose, // 33%
            33..=65 => DiversityStrategy::Combine,  // 33%
            _ => DiversityStrategy::NovelApproach,  // 34%
        };
        DiversityHint::new(strategy)
    }

    /// Inject diversity with context derived from the selected entry.
    ///
    /// Provides richer hints by attaching entry-specific context
    /// (e.g., pending goal count, visit count).
    pub fn inject_diversity_with_context(
        &self,
        entry: &SketchEntry,
        rng: &mut Rng,
    ) -> DiversityHint {
        let hint = self.inject_diversity(rng);
        let context = format!(
            "entry={id}, goals={goals}, visits={visits}, elo={elo:.0}",
            id = entry.id,
            goals = entry.pending_goal_count(),
            visits = entry.visits,
            elo = entry.elo_rating,
        );
        DiversityHint::with_context(hint.strategy, context)
    }
}

impl fmt::Display for SketchSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SketchSampler(c={:.3}, ε={:.3}, pop={len}/{max})",
            self.config.c,
            self.config.epsilon,
            len = self.population.len(),
            max = self.population.config().max_population,
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::sketch_population::PopulationConfig;
    use crate::proof::sketch_types::{DEFAULT_ELO, ProofState, SketchEntry};

    // ── Helpers ────────────────────────────────────────────────

    fn make_entry(elo: f64) -> SketchEntry {
        let state = ProofState::new(format!("state_{elo}").into_bytes());
        let mut entry = SketchEntry::new(state, vec![]);
        entry.update_elo(elo);
        entry
    }

    fn make_entry_with_visits(elo: f64, visits: usize) -> SketchEntry {
        let mut entry = make_entry(elo);
        for _ in 0..visits {
            entry.record_visit();
        }
        entry
    }

    fn make_sampler_with_entries(entries: Vec<SketchEntry>) -> SketchSampler {
        let capacity = entries.len().max(1);
        let config = PopulationConfig::with_overshoot(capacity, capacity * 2);
        let mut pop = SketchPopulation::with_config(config);
        for entry in entries {
            pop.insert_no_evict(entry);
        }
        SketchSampler::new(pop)
    }

    fn make_empty_sampler() -> SketchSampler {
        SketchSampler::new(SketchPopulation::with_paper_defaults())
    }

    // ── Config Tests ──────────────────────────────────────────

    #[test]
    fn config_paper_defaults_matches_constants() {
        let cfg = SketchSamplerConfig::paper_defaults();
        assert!((cfg.c - DEFAULT_EXPLORATION_C).abs() < f64::EPSILON);
        assert!((cfg.epsilon - DEFAULT_EPSILON).abs() < f64::EPSILON);
    }

    #[test]
    fn config_with_c_custom() {
        let cfg = SketchSamplerConfig::paper_defaults().with_c(1.5);
        assert!((cfg.c - 1.5).abs() < f64::EPSILON);
        assert!((cfg.epsilon - DEFAULT_EPSILON).abs() < f64::EPSILON);
    }

    #[test]
    fn config_with_epsilon_custom() {
        let cfg = SketchSamplerConfig::paper_defaults().with_epsilon(0.3);
        assert!((cfg.c - DEFAULT_EXPLORATION_C).abs() < f64::EPSILON);
        assert!((cfg.epsilon - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn config_display() {
        let cfg = SketchSamplerConfig::paper_defaults();
        let s = format!("{cfg}");
        assert!(s.contains("c=0.200"));
        assert!(s.contains("ε=0.100"));
    }

    // ── Empty Population Tests ────────────────────────────────

    #[test]
    fn sample_empty_population_returns_none() {
        let sampler = make_empty_sampler();
        let mut rng = Rng::with_seed(42);
        assert!(sampler.sample(&mut rng).is_none());
    }

    #[test]
    fn sample_p_ucb_empty_returns_none() {
        let sampler = make_empty_sampler();
        let mut rng = Rng::with_seed(42);
        assert!(sampler.sample_p_ucb(&mut rng).is_none());
    }

    #[test]
    fn sample_epsilon_greedy_empty_returns_none() {
        let sampler = make_empty_sampler();
        let mut rng = Rng::with_seed(42);
        assert!(sampler.sample_epsilon_greedy(&mut rng).is_none());
    }

    #[test]
    fn sample_mut_empty_returns_none() {
        let mut sampler = make_empty_sampler();
        let mut rng = Rng::with_seed(42);
        assert!(sampler.sample_mut(&mut rng).is_none());
    }

    // ── P-UCB Sampling Tests ──────────────────────────────────

    #[test]
    fn p_ucb_selects_unvisited_entries_first() {
        // Create entries: one visited, one unvisited with lower Elo
        let visited = make_entry_with_visits(1500.0, 10); // high Elo, 10 visits
        let unvisited = make_entry(1200.0); // default Elo, 0 visits

        let sampler = make_sampler_with_entries(vec![visited, unvisited]);
        let mut rng = Rng::with_seed(42);

        let selected = sampler
            .sample_p_ucb(&mut rng)
            .expect("should select an entry");

        // Unvisited entry should be selected due to maximal exploration bonus
        assert_eq!(selected.visits, 0, "P-UCB should prefer unvisited entries");
        assert!((selected.elo_rating - 1200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn p_ucb_prefers_higher_elo_when_visits_equal() {
        let low_elo = make_entry_with_visits(1100.0, 5);
        let high_elo = make_entry_with_visits(1500.0, 5);

        let sampler = make_sampler_with_entries(vec![low_elo, high_elo]);
        let mut rng = Rng::with_seed(42);

        let selected = sampler
            .sample_p_ucb(&mut rng)
            .expect("should select an entry");

        // Same visits: higher Elo should win
        assert!(
            selected.elo_rating > 1400.0,
            "P-UCB should prefer higher Elo with equal visits"
        );
    }

    #[test]
    fn p_ucb_single_entry_always_selected() {
        let entry = make_entry(1300.0);
        let sampler = make_sampler_with_entries(vec![entry]);

        for seed in 0..10u64 {
            let mut rng = Rng::with_seed(seed);
            let selected = sampler.sample_p_ucb(&mut rng).expect("should select entry");
            assert!((selected.elo_rating - 1300.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn p_ucb_many_unvisited_prefers_higher_elo() {
        // When all are unvisited (visits=0), exploration bonus is identical
        // for all, so Elo becomes the tiebreaker
        let entries: Vec<SketchEntry> = (0..5)
            .map(|i| make_entry(1200.0 + i as f64 * 100.0))
            .collect();

        let sampler = make_sampler_with_entries(entries);
        let mut rng = Rng::with_seed(42);

        let selected = sampler.sample_p_ucb(&mut rng).expect("should select entry");

        // All unvisited: highest Elo should be selected
        assert!(
            selected.elo_rating > 1550.0,
            "All-unvisited should prefer highest Elo, got {elo}",
            elo = selected.elo_rating
        );
    }

    // ── Epsilon-Greedy Tests ──────────────────────────────────

    #[test]
    fn epsilon_greedy_explores_sometimes() {
        let best = make_entry(1600.0);
        let worst = make_entry(800.0);

        let sampler = make_sampler_with_entries(vec![best, worst]);

        // With ε=0.1 and enough samples, we should see both entries
        let mut explored_low_elo = false;
        let mut rng = Rng::with_seed(123);

        for _ in 0..200 {
            let selected = sampler
                .sample_epsilon_greedy(&mut rng)
                .expect("should select");
            if selected.elo_rating < 900.0 {
                explored_low_elo = true;
                break;
            }
        }

        assert!(
            explored_low_elo,
            "ε-greedy should explore low-Elo entry within 200 samples"
        );
    }

    #[test]
    fn epsilon_greedy_exploits_mostly() {
        let best = make_entry(1600.0);
        let worst = make_entry(800.0);

        let sampler = make_sampler_with_entries(vec![best, worst]);

        let mut exploit_count = 0;
        let mut rng = Rng::with_seed(456);

        for _ in 0..1000 {
            let selected = sampler
                .sample_epsilon_greedy(&mut rng)
                .expect("should select");
            if (selected.elo_rating - 1600.0).abs() < f64::EPSILON {
                exploit_count += 1;
            }
        }

        // With ε=0.1, expect ~90% exploitation
        assert!(
            exploit_count > 800,
            "Expected >800 exploits out of 1000, got {exploit_count}"
        );
    }

    #[test]
    fn epsilon_greedy_zero_epsilon_always_exploits() {
        let best = make_entry(1600.0);
        let worst = make_entry(800.0);

        let config = SketchSamplerConfig::paper_defaults().with_epsilon(0.0);
        let sampler = SketchSampler::with_config(
            make_sampler_with_entries(vec![best, worst])
                .population()
                .clone(),
            config,
        );

        let mut rng = Rng::with_seed(789);

        for _ in 0..100 {
            let selected = sampler
                .sample_epsilon_greedy(&mut rng)
                .expect("should select");
            assert!(
                (selected.elo_rating - 1600.0).abs() < f64::EPSILON,
                "ε=0 should always exploit"
            );
        }
    }

    #[test]
    fn epsilon_greedy_one_epsilon_always_explores() {
        let best = make_entry(1600.0);
        let worst = make_entry(800.0);

        let config = SketchSamplerConfig::paper_defaults().with_epsilon(1.0);
        let sampler = SketchSampler::with_config(
            make_sampler_with_entries(vec![best, worst])
                .population()
                .clone(),
            config,
        );

        let mut rng = Rng::with_seed(101);

        // With ε=1.0, every sample is random — both entries should appear
        let mut saw_best = false;
        let mut saw_worst = false;

        for _ in 0..100 {
            let selected = sampler
                .sample_epsilon_greedy(&mut rng)
                .expect("should select");
            if (selected.elo_rating - 1600.0).abs() < f64::EPSILON {
                saw_best = true;
            }
            if (selected.elo_rating - 800.0).abs() < f64::EPSILON {
                saw_worst = true;
            }
        }

        assert!(saw_best, "ε=1.0 should see best entry via random");
        assert!(saw_worst, "ε=1.0 should see worst entry via random");
    }

    // ── Diversity Injection Tests ─────────────────────────────

    #[test]
    fn diversity_injection_returns_all_three_strategies() {
        let sampler = make_empty_sampler();
        let mut rng = Rng::with_seed(42);

        let mut saw_decompose = false;
        let mut saw_combine = false;
        let mut saw_novel = false;

        for _ in 0..300 {
            let hint = sampler.inject_diversity(&mut rng);
            match hint.strategy {
                DiversityStrategy::Decompose => saw_decompose = true,
                DiversityStrategy::Combine => saw_combine = true,
                DiversityStrategy::NovelApproach => saw_novel = true,
            }

            if saw_decompose && saw_combine && saw_novel {
                return; // all seen, test passes
            }
        }

        assert!(saw_decompose, "should see Decompose strategy");
        assert!(saw_combine, "should see Combine strategy");
        assert!(saw_novel, "should see NovelApproach strategy");
    }

    #[test]
    fn diversity_injection_no_context_by_default() {
        let sampler = make_empty_sampler();
        let mut rng = Rng::with_seed(42);

        let hint = sampler.inject_diversity(&mut rng);
        assert!(
            hint.context.is_none(),
            "inject_diversity should not add context"
        );
    }

    #[test]
    fn diversity_injection_with_context() {
        let entry = make_entry(1400.0);
        let sampler = make_sampler_with_entries(vec![entry]);
        let mut rng = Rng::with_seed(42);

        let hint = sampler
            .inject_diversity_with_context(sampler.population().sorted_by_elo()[0], &mut rng);

        assert!(hint.context.is_some(), "should have context");
        let ctx = hint.context.expect("checked above");
        assert!(
            ctx.contains("elo=1400"),
            "context should mention elo, got: {ctx}"
        );
        assert!(
            ctx.contains("visits=0"),
            "context should mention visits, got: {ctx}"
        );
    }

    #[test]
    fn diversity_distribution_approximately_uniform() {
        let sampler = make_empty_sampler();
        let mut rng = Rng::with_seed(99);

        let mut counts = [0usize; 3];
        let trials = 3000;

        for _ in 0..trials {
            let hint = sampler.inject_diversity(&mut rng);
            counts[hint.strategy.index()] += 1;
        }

        // Each strategy should get ~33% (allow 25-42% range for RNG variance)
        for (i, &count) in counts.iter().enumerate() {
            let fraction = count as f64 / trials as f64;
            assert!(
                fraction > 0.25 && fraction < 0.42,
                "Strategy {i} has fraction {fraction:.3}, expected ~0.33"
            );
        }
    }

    // ── Parallelism Guard Tests ───────────────────────────────

    #[test]
    fn parallelism_guard_returns_bool() {
        // Just verify it doesn't panic — the actual value depends on runtime
        let result = should_use_population();
        // In test context, rayon usually initializes with 1 thread
        // so we just verify it's a valid bool (which it always is)
        let _ = result;
    }

    #[test]
    fn sample_works_regardless_of_parallelism() {
        let entry = make_entry(1300.0);
        let sampler = make_sampler_with_entries(vec![entry]);
        let mut rng = Rng::with_seed(42);

        // sample() should work whether parallelism guard is true or false
        let selected = sampler.sample(&mut rng);
        assert!(
            selected.is_some(),
            "sample should return Some for non-empty population"
        );
    }

    // ── Elo Ordering Tests ────────────────────────────────────

    #[test]
    fn sample_respects_elo_ordering_with_equal_visits() {
        let entries: Vec<SketchEntry> = [1000.0, 1200.0, 1400.0, 1600.0, 1800.0]
            .iter()
            .map(|&elo| make_entry_with_visits(elo, 10))
            .collect();

        let sampler = make_sampler_with_entries(entries);
        let mut rng = Rng::with_seed(42);

        // With equal visits, P-UCB should prefer higher Elo
        let selected = sampler.sample_p_ucb(&mut rng).expect("should select");
        assert!(
            selected.elo_rating > 1700.0,
            "Expected highest Elo with equal visits, got {elo}",
            elo = selected.elo_rating
        );
    }

    #[test]
    fn sample_best_elo_returns_highest() {
        let entries: Vec<SketchEntry> = [1000.0, 1400.0, 1200.0]
            .iter()
            .map(|&elo| make_entry(elo))
            .collect();

        let sampler = make_sampler_with_entries(entries);
        let best = sampler.sample_best_elo().expect("should return best");

        assert!(
            (best.elo_rating - 1400.0).abs() < f64::EPSILON,
            "Expected 1400 Elo, got {elo}",
            elo = best.elo_rating
        );
    }

    // ── Sampler Display Test ──────────────────────────────────

    #[test]
    fn sampler_display() {
        let entry = make_entry(1400.0);
        let sampler = make_sampler_with_entries(vec![entry]);
        let s = format!("{sampler}");
        assert!(s.contains("c=0.200"), "display should show c, got: {s}");
        assert!(s.contains("ε=0.100"), "display should show ε, got: {s}");
        assert!(
            s.contains("pop=1/"),
            "display should show population, got: {s}"
        );
    }

    // ── Accessor Tests ────────────────────────────────────────

    #[test]
    fn accessor_c_returns_configured_value() {
        let config = SketchSamplerConfig::paper_defaults().with_c(0.5);
        let sampler = SketchSampler::with_config(SketchPopulation::with_paper_defaults(), config);
        assert!((sampler.c() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn accessor_epsilon_returns_configured_value() {
        let config = SketchSamplerConfig::paper_defaults().with_epsilon(0.25);
        let sampler = SketchSampler::with_config(SketchPopulation::with_paper_defaults(), config);
        assert!((sampler.epsilon() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn population_mut_allows_modification() {
        let entry = make_entry(1200.0);
        let mut sampler = make_sampler_with_entries(vec![entry]);

        let id = {
            let pop = sampler.population();
            pop.sorted_by_elo()[0].id
        };

        let entry_mut = sampler.population_mut().get_mut(&id).expect("should exist");
        entry_mut.update_elo(1500.0);

        assert!(
            (sampler.population().get(&id).expect("exists").elo_rating - 1500.0).abs()
                < f64::EPSILON,
        );
    }

    // ── Integration: Sample + Diversity ───────────────────────

    #[test]
    fn sample_and_inject_diversity_integration() {
        let entries: Vec<SketchEntry> = [1200.0, 1400.0, 1600.0]
            .iter()
            .map(|&elo| make_entry_with_visits(elo, 3))
            .collect();

        let sampler = make_sampler_with_entries(entries);
        let mut rng = Rng::with_seed(42);

        // Sample an entry
        let selected = sampler.sample_p_ucb(&mut rng).expect("should select");
        assert!(selected.visits > 0 || selected.elo_rating > 0.0);

        // Inject diversity for the selected entry
        let hint = sampler.inject_diversity_with_context(selected, &mut rng);
        assert!(hint.context.is_some());
        assert!(matches!(
            hint.strategy,
            DiversityStrategy::Decompose
                | DiversityStrategy::Combine
                | DiversityStrategy::NovelApproach
        ));
    }

    #[test]
    fn sample_mut_tracks_visit() {
        let entry = make_entry(1200.0);
        let mut sampler = make_sampler_with_entries(vec![entry]);
        let mut rng = Rng::with_seed(42);

        {
            let selected = sampler.sample_mut(&mut rng).expect("should select");
            selected.record_visit();
        }

        let pop = sampler.population();
        let updated = pop.sorted_by_elo()[0];
        assert_eq!(updated.visits, 1, "visit should be recorded");
    }

    // ── Edge Cases ────────────────────────────────────────────

    #[test]
    fn large_population_sampling() {
        let entries: Vec<SketchEntry> = (0..64)
            .map(|i| {
                let elo = DEFAULT_ELO + i as f64 * 10.0;
                let visits = (i % 5) as usize; // varied visit counts
                make_entry_with_visits(elo, visits)
            })
            .collect();

        let sampler = make_sampler_with_entries(entries);
        let mut rng = Rng::with_seed(42);

        // Should handle full paper-size population without issues
        let selected = sampler.sample_p_ucb(&mut rng);
        assert!(selected.is_some());

        let entry = selected.expect("checked above");
        // Entry with visits=0 should be preferred
        assert!(entry.elo_rating > 0.0, "should select a valid entry");
    }

    #[test]
    fn all_same_elo_degenerate() {
        // All entries have identical Elo — P-UCB should still work
        let entries: Vec<SketchEntry> = (0..5).map(|_| make_entry_with_visits(1200.0, 0)).collect();

        let sampler = make_sampler_with_entries(entries);
        let mut rng = Rng::with_seed(42);

        let selected = sampler.sample_p_ucb(&mut rng).expect("should select");
        assert!((selected.elo_rating - 1200.0).abs() < f64::EPSILON);
    }
}
