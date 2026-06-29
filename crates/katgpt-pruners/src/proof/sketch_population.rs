//! Top-64 Elo-rated sketch population database (Plan 128, T3).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "The population database maintains a top-64 set of proof sketches
//! ranked by Elo, evicting the lowest-rated entries when full."
//!
//! # Architecture
//!
//! ```text
//! SketchPopulation
//! ├── sketches: HashMap<SketchId, SketchEntry>   // all entries
//! ├── top_k: usize                                // 64 per paper
//! ├── max_population: usize                       // hard cap
//! └── insert(entry) → EvictionReport
//!     1. Add entry to sketches
//!     2. If len > max_population → evict lowest Elo entries
//!     3. Return report of evicted entries
//! ```
//!
//! # Eviction Policy
//!
//! Keep top-K by Elo. For ties at the Elo boundary, use LRU
//! (least-recently-visited, i.e., lowest visit count) as tiebreaker.
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

use std::collections::HashMap;
use std::fmt;

use super::sketch_types::{DEFAULT_ELO, SketchEntry, SketchId};

// ── PopulationConfig ───────────────────────────────────────────

/// Configuration for the sketch population database.
#[derive(Clone, Debug, PartialEq)]
pub struct PopulationConfig {
    /// Maximum number of entries to keep (eviction threshold).
    ///
    /// Paper default: 64. Entries are evicted when population exceeds this.
    pub top_k: usize,

    /// Hard cap on population size.
    ///
    /// Allows temporary overshoot beyond `top_k` before eviction runs.
    /// Must be >= `top_k`. Set equal to `top_k` for strict enforcement.
    pub max_population: usize,
}

impl PopulationConfig {
    /// Paper defaults: top_k=64, max_population=64 (strict).
    pub const PAPER_DEFAULTS: Self = Self {
        top_k: 64,
        max_population: 64,
    };

    /// Create config with custom top_k and strict enforcement.
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            max_population: top_k,
        }
    }

    /// Create config with overshoot allowed (max_population > top_k).
    pub fn with_overshoot(top_k: usize, max_population: usize) -> Self {
        Self {
            top_k,
            max_population: max_population.max(top_k),
        }
    }

    /// Validate config consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.top_k == 0 {
            return Err("top_k must be > 0".to_string());
        }
        if self.max_population < self.top_k {
            return Err(format!(
                "max_population ({}) must be >= top_k ({})",
                self.max_population, self.top_k
            ));
        }
        Ok(())
    }
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self::PAPER_DEFAULTS
    }
}

// ── EvictionReport ─────────────────────────────────────────────

/// Report of entries evicted during population maintenance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvictionReport {
    /// IDs of evicted entries.
    pub evicted_ids: Vec<SketchId>,
    /// Number of entries before eviction.
    pub population_before: usize,
    /// Number of entries after eviction.
    pub population_after: usize,
}

impl EvictionReport {
    /// Was anything evicted?
    pub fn did_evict(&self) -> bool {
        !self.evicted_ids.is_empty()
    }

    /// Number of entries evicted.
    pub fn count(&self) -> usize {
        self.evicted_ids.len()
    }
}

impl fmt::Display for EvictionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.did_evict() {
            true => write!(
                f,
                "EvictionReport(evicted={}, {}→{})",
                self.count(),
                self.population_before,
                self.population_after
            ),
            false => write!(f, "EvictionReport(none)"),
        }
    }
}

// ── SketchPopulation ───────────────────────────────────────────

/// Top-64 Elo-rated sketch population database.
///
/// Maintains a bounded set of proof/game strategy sketches ranked by Elo.
/// When the population exceeds `max_population`, the lowest-rated entries
/// are evicted to maintain the top-K by Elo.
///
/// # Thread Safety
///
/// Not thread-safe by itself. Use `Mutex<SketchPopulation>` or
/// `papaya::HashMap` for concurrent access if needed.
///
/// # Example
///
/// ```rust,ignore
/// use katgpt::pruners::proof::{SketchPopulation, SketchEntry, ProofState, Goal};
///
/// let mut pop = SketchPopulation::new(64);
///
/// let state = ProofState::new(b"opening".to_vec());
/// let entry = SketchEntry::new(state, vec![Goal::from_label("win")]);
///
/// let report = pop.insert(entry);
/// assert!(!report.did_evict());
/// assert_eq!(pop.len(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct SketchPopulation {
    /// Sketch entries keyed by ID.
    sketches: HashMap<SketchId, SketchEntry>,
    /// Population configuration.
    config: PopulationConfig,
}

impl SketchPopulation {
    /// Create a new population with the given top_k capacity.
    pub fn new(top_k: usize) -> Self {
        Self {
            sketches: HashMap::with_capacity(top_k),
            config: PopulationConfig::new(top_k),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: PopulationConfig) -> Self {
        Self {
            sketches: HashMap::with_capacity(config.top_k),
            config,
        }
    }

    /// Create with paper defaults (top_k=64).
    pub fn with_paper_defaults() -> Self {
        Self::with_config(PopulationConfig::PAPER_DEFAULTS)
    }

    // ── Insertion ──────────────────────────────────────────────

    /// Insert a sketch entry into the population.
    ///
    /// If an entry with the same ID already exists, it is replaced.
    /// Triggers eviction if population exceeds `max_population`.
    ///
    /// Returns an eviction report describing what was removed.
    pub fn insert(&mut self, entry: SketchEntry) -> EvictionReport {
        let population_before = self.sketches.len();
        self.sketches.insert(entry.id, entry);

        match self.needs_eviction() {
            true => self.evict_to_capacity(),
            false => EvictionReport {
                evicted_ids: Vec::new(),
                population_before,
                population_after: self.sketches.len(),
            },
        }
    }

    /// Insert without eviction (allows temporary overshoot).
    ///
    /// Useful during batch operations where eviction is deferred.
    pub fn insert_no_evict(&mut self, entry: SketchEntry) {
        self.sketches.insert(entry.id, entry);
    }

    /// Run eviction to bring population down to `top_k`.
    ///
    /// Call this after a batch of `insert_no_evict` calls.
    pub fn finalize_batch(&mut self) -> EvictionReport {
        match self.needs_eviction() {
            true => self.evict_to_capacity(),
            false => EvictionReport::default(),
        }
    }

    // ── Lookup ─────────────────────────────────────────────────

    /// Get a sketch entry by ID.
    pub fn get(&self, id: &SketchId) -> Option<&SketchEntry> {
        self.sketches.get(id)
    }

    /// Get a mutable reference to a sketch entry by ID.
    pub fn get_mut(&mut self, id: &SketchId) -> Option<&mut SketchEntry> {
        self.sketches.get_mut(id)
    }

    /// Check if an entry exists.
    pub fn contains(&self, id: &SketchId) -> bool {
        self.sketches.contains_key(id)
    }

    /// Remove an entry by ID.
    pub fn remove(&mut self, id: &SketchId) -> Option<SketchEntry> {
        self.sketches.remove(id)
    }

    // ── Queries ────────────────────────────────────────────────

    /// Number of entries in the population.
    pub fn len(&self) -> usize {
        self.sketches.len()
    }

    /// Is the population empty?
    pub fn is_empty(&self) -> bool {
        self.sketches.is_empty()
    }

    /// Population configuration.
    pub fn config(&self) -> &PopulationConfig {
        &self.config
    }

    /// Is the population at or above capacity?
    pub fn is_full(&self) -> bool {
        self.sketches.len() >= self.config.max_population
    }

    /// Sum of visit counts across all entries.
    pub fn total_visits(&self) -> usize {
        self.sketches.values().map(|e| e.visits).sum()
    }

    /// Elo range (min, max) across all entries.
    ///
    /// Returns `None` if population is empty.
    /// Used for P-UCB normalization.
    pub fn elo_range(&self) -> Option<(f64, f64)> {
        // Single-pass min/max with no allocation (replaces the previous
        // `Vec<f64>::collect()` + 2-pass scan). Called per P-UCB sample.
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for e in self.sketches.values() {
            if e.elo_rating < min {
                min = e.elo_rating;
            }
            if e.elo_rating > max {
                max = e.elo_rating;
            }
        }
        if min.is_finite() {
            Some((min, max))
        } else {
            None
        }
    }

    /// Entry with the highest Elo rating.
    ///
    /// O(N) single pass, no allocation. Replaces the previous
    /// `sorted_by_elo().first()` which performed an O(N log N) sort + Vec
    /// allocation per call.
    pub fn best_elo(&self) -> Option<&SketchEntry> {
        self.sketches
            .values()
            .max_by(|a, b| a.elo_rating.total_cmp(&b.elo_rating))
    }

    /// Pick the entry at the `idx`-th position in HashMap iteration order.
    ///
    /// Used by `sample_random` to draw a uniform random entry without the
    /// O(N log N) sort + allocation that `sorted_by_elo()` imposed. HashMap
    /// iteration order is unspecified but well-defined for a given
    /// mutation history, so `idx ∈ [0, len)` is a uniform draw over the
    /// current entries. Returns `None` if `idx >= len`.
    pub fn nth_in_arbitrary_order(&self, idx: usize) -> Option<&SketchEntry> {
        if idx >= self.sketches.len() {
            return None;
        }
        // Iterator advance is O(idx); for small populations (≤64 per config)
        // this is faster than allocating + sorting.
        self.sketches.values().nth(idx)
    }

    /// Iterator over all entries in arbitrary (HashMap) order.
    ///
    /// Use this instead of `sorted_by_elo()` whenever the call site does not
    /// actually need entries sorted by Elo — e.g. when it will immediately
    /// `max_by` / `sum` / `filter` over them. Avoids the O(N log N) sort and
    /// the `Vec<&SketchEntry>` allocation.
    pub fn values_arbitrary(&self) -> impl Iterator<Item = &SketchEntry> {
        self.sketches.values()
    }

    /// Average Elo across all entries.
    ///
    /// Returns `DEFAULT_ELO` if population is empty.
    pub fn avg_elo(&self) -> f64 {
        match self.sketches.is_empty() {
            true => DEFAULT_ELO,
            false => {
                let total: f64 = self.sketches.values().map(|e| e.elo_rating).sum();
                total / self.sketches.len() as f64
            }
        }
    }

    /// Entries sorted by Elo descending (best first).
    ///
    /// Returns a Vec of references, highest Elo first.
    /// Allocates — use sparingly in hot paths.
    pub fn sorted_by_elo(&self) -> Vec<&SketchEntry> {
        let mut entries: Vec<&SketchEntry> = self.sketches.values().collect();
        entries.sort_by(|a, b| b.elo_rating.total_cmp(&a.elo_rating));
        entries
    }

    /// Top K entries by Elo.
    pub fn top_k(&self) -> Vec<&SketchEntry> {
        let mut sorted = self.sorted_by_elo();
        sorted.truncate(self.config.top_k);
        sorted
    }

    /// Entries eligible for exploration (visits < threshold).
    ///
    /// Used by P-UCB sampling to find under-explored entries.
    pub fn sample_eligible(&self, min_visits: usize) -> Vec<&SketchEntry> {
        self.sketches
            .values()
            .filter(|e| e.visits <= min_visits)
            .collect()
    }

    /// All entry IDs.
    pub fn ids(&self) -> Vec<SketchId> {
        self.sketches.keys().copied().collect()
    }

    // ── Eviction ───────────────────────────────────────────────

    /// Does the population need eviction?
    ///
    /// Triggers when population exceeds `max_population` (not `top_k`).
    /// This allows overshoot mode where `max_population > top_k`.
    fn needs_eviction(&self) -> bool {
        self.sketches.len() > self.config.max_population
    }

    /// Evict lowest-Elo entries until population is at `top_k`.
    ///
    /// Tiebreaker: lowest visit count (LRU proxy).
    /// Returns a report of what was evicted.
    fn evict_to_capacity(&mut self) -> EvictionReport {
        let population_before = self.sketches.len();

        if population_before <= self.config.top_k {
            return EvictionReport {
                evicted_ids: Vec::new(),
                population_before,
                population_after: population_before,
            };
        }

        let to_remove = population_before - self.config.top_k;

        // Sort by (elo ASC, visits ASC) to find worst entries
        let mut ranked: Vec<(SketchId, f64, usize)> = self
            .sketches
            .values()
            .map(|e| (e.id, e.elo_rating, e.visits))
            .collect();

        ranked.sort_by(|a, b| {
            // Primary: Elo ascending (worst first)
            match a.1.total_cmp(&b.1) {
                std::cmp::Ordering::Equal => {
                    // Tiebreaker: visits ascending (least explored first)
                    a.2.cmp(&b.2)
                }
                other => other,
            }
        });

        let evicted_ids: Vec<SketchId> = ranked
            .into_iter()
            .take(to_remove)
            .map(|(id, _, _)| id)
            .collect();

        for id in &evicted_ids {
            self.sketches.remove(id);
        }

        EvictionReport {
            population_after: self.sketches.len(),
            evicted_ids,
            population_before,
        }
    }

    /// Clear the entire population.
    pub fn clear(&mut self) {
        self.sketches.clear();
    }

    // ── Metrics ────────────────────────────────────────────────

    /// Estimate memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> usize {
        // SketchEntry is ~200-300 bytes (state + goals + lessons + metadata)
        let per_entry = 256;
        self.sketches.len() * per_entry
    }
}

impl Default for SketchPopulation {
    fn default() -> Self {
        Self::with_paper_defaults()
    }
}

impl fmt::Display for SketchPopulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (min_elo, max_elo) = match self.elo_range() {
            Some((min, max)) => (format!("{min:.0}"), format!("{max:.0}")),
            None => ("—".to_string(), "—".to_string()),
        };
        write!(
            f,
            "SketchPopulation(entries={}/{}, elo={}..{}, visits={})",
            self.len(),
            self.config.top_k,
            min_elo,
            max_elo,
            self.total_visits(),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::sketch_types::ProofState;

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

    // ── PopulationConfig Tests ─────────────────────────────────

    #[test]
    fn config_paper_defaults() {
        let cfg = PopulationConfig::PAPER_DEFAULTS;
        assert_eq!(cfg.top_k, 64);
        assert_eq!(cfg.max_population, 64);
    }

    #[test]
    fn config_new_strict() {
        let cfg = PopulationConfig::new(32);
        assert_eq!(cfg.top_k, 32);
        assert_eq!(cfg.max_population, 32);
    }

    #[test]
    fn config_with_overshoot() {
        let cfg = PopulationConfig::with_overshoot(32, 48);
        assert_eq!(cfg.top_k, 32);
        assert_eq!(cfg.max_population, 48);
    }

    #[test]
    fn config_overshoot_clamps_to_top_k() {
        let cfg = PopulationConfig::with_overshoot(32, 16);
        assert_eq!(cfg.max_population, 32, "max_population clamps to top_k");
    }

    #[test]
    fn config_validate_ok() {
        assert!(PopulationConfig::new(64).validate().is_ok());
    }

    #[test]
    fn config_validate_zero_top_k() {
        assert!(PopulationConfig::new(0).validate().is_err());
    }

    #[test]
    fn config_validate_inverted() {
        let cfg = PopulationConfig {
            top_k: 64,
            max_population: 32,
        };
        assert!(cfg.validate().is_err());
    }

    // ── EvictionReport Tests ───────────────────────────────────

    #[test]
    fn eviction_report_no_eviction() {
        let report = EvictionReport::default();
        assert!(!report.did_evict());
        assert_eq!(report.count(), 0);
    }

    #[test]
    fn eviction_report_with_eviction() {
        let report = EvictionReport {
            evicted_ids: vec![SketchId::new(), SketchId::new()],
            population_before: 10,
            population_after: 8,
        };
        assert!(report.did_evict());
        assert_eq!(report.count(), 2);
    }

    #[test]
    fn eviction_report_display() {
        let report = EvictionReport {
            evicted_ids: vec![SketchId::new()],
            population_before: 5,
            population_after: 4,
        };
        let display = format!("{report}");
        assert!(display.contains("evicted=1"));
        assert!(display.contains("5→4"));
    }

    // ── SketchPopulation Insertion Tests ───────────────────────

    #[test]
    fn population_new_empty() {
        let pop = SketchPopulation::new(64);
        assert!(pop.is_empty());
        assert_eq!(pop.len(), 0);
    }

    #[test]
    fn population_insert_single() {
        let mut pop = SketchPopulation::new(4);
        let entry = make_entry(1200.0);
        let id = entry.id;

        let report = pop.insert(entry);
        assert!(!report.did_evict());
        assert_eq!(pop.len(), 1);
        assert!(pop.contains(&id));
    }

    #[test]
    fn population_insert_replaces_same_id() {
        let mut pop = SketchPopulation::new(4);
        let entry = make_entry(1200.0);
        let id = entry.id;

        pop.insert(entry);
        assert_eq!(pop.len(), 1);

        // Re-insert with updated Elo
        let mut updated = make_entry(1200.0);
        updated.id = id;
        updated.update_elo(1400.0);
        pop.insert(updated);

        assert_eq!(pop.len(), 1, "same ID should replace, not add");
        assert_eq!(pop.get(&id).unwrap().elo_rating, 1400.0);
    }

    #[test]
    fn population_insert_triggers_eviction() {
        let mut pop = SketchPopulation::new(3);

        // Insert 4 entries — should evict lowest Elo
        let e1 = make_entry(1100.0);
        let e2 = make_entry(1200.0);
        let e3 = make_entry(1300.0);
        let e4 = make_entry(1400.0);

        let id_low = e1.id;

        pop.insert(e1);
        pop.insert(e2);
        pop.insert(e3);
        assert_eq!(pop.len(), 3);

        let report = pop.insert(e4);
        assert!(report.did_evict());
        assert_eq!(report.count(), 1);
        assert_eq!(pop.len(), 3);
        assert!(!pop.contains(&id_low), "lowest Elo entry should be evicted");
    }

    #[test]
    fn population_eviction_keeps_highest_elo() {
        let mut pop = SketchPopulation::new(3);

        let e1 = make_entry(1000.0);
        let e2 = make_entry(1100.0);
        let e3 = make_entry(1200.0);
        let e4 = make_entry(1300.0);
        let e5 = make_entry(1400.0);

        pop.insert(e1);
        pop.insert(e2);
        pop.insert(e3);
        pop.insert(e4);
        pop.insert(e5);

        assert_eq!(pop.len(), 3);

        // Should keep 1200, 1300, 1400
        let elos: Vec<f64> = pop.sorted_by_elo().iter().map(|e| e.elo_rating).collect();
        assert_eq!(elos, vec![1400.0, 1300.0, 1200.0]);
    }

    #[test]
    fn population_eviction_tiebreaks_by_visits() {
        let mut pop = SketchPopulation::new(2);

        // Two entries with same Elo but different visits
        let e1 = make_entry_with_visits(1200.0, 10); // more visits = keep
        let e2 = make_entry_with_visits(1200.0, 1); // fewer visits = evict
        let e3 = make_entry(1300.0); // highest Elo = keep

        let id_few_visits = e2.id;

        pop.insert(e1);
        pop.insert(e2);
        assert_eq!(pop.len(), 2);

        let report = pop.insert(e3);
        assert!(report.did_evict());
        assert_eq!(pop.len(), 2);
        assert!(
            !pop.contains(&id_few_visits),
            "fewer visits should lose tiebreak"
        );
    }

    // ── Batch Insertion Tests ──────────────────────────────────

    #[test]
    fn population_batch_insert_deferred_eviction() {
        let mut pop = SketchPopulation::new(2);

        // Insert 4 entries without eviction
        let e1 = make_entry(1000.0);
        let e2 = make_entry(1100.0);
        let e3 = make_entry(1200.0);
        let e4 = make_entry(1300.0);

        pop.insert_no_evict(e1);
        pop.insert_no_evict(e2);
        pop.insert_no_evict(e3);
        pop.insert_no_evict(e4);
        assert_eq!(pop.len(), 4, "no eviction during batch");

        let report = pop.finalize_batch();
        assert!(report.did_evict());
        assert_eq!(pop.len(), 2);
    }

    // ── Lookup Tests ───────────────────────────────────────────

    #[test]
    fn population_get_missing() {
        let pop = SketchPopulation::new(4);
        assert!(pop.get(&SketchId::new()).is_none());
    }

    #[test]
    fn population_get_mut_updates() {
        let mut pop = SketchPopulation::new(4);
        let entry = make_entry(1200.0);
        let id = entry.id;
        pop.insert(entry);

        if let Some(e) = pop.get_mut(&id) {
            e.update_elo(1500.0);
        }

        assert_eq!(pop.get(&id).unwrap().elo_rating, 1500.0);
    }

    #[test]
    fn population_remove() {
        let mut pop = SketchPopulation::new(4);
        let entry = make_entry(1200.0);
        let id = entry.id;
        pop.insert(entry);

        let removed = pop.remove(&id);
        assert!(removed.is_some());
        assert!(!pop.contains(&id));
        assert!(pop.is_empty());
    }

    // ── Query Tests ────────────────────────────────────────────

    #[test]
    fn population_elo_range_empty() {
        let pop = SketchPopulation::new(4);
        assert!(pop.elo_range().is_none());
    }

    #[test]
    fn population_elo_range() {
        let mut pop = SketchPopulation::new(4);
        pop.insert(make_entry(1100.0));
        pop.insert(make_entry(1300.0));
        pop.insert(make_entry(1200.0));

        let (min, max) = pop.elo_range().unwrap();
        assert!((min - 1100.0).abs() < 1e-9);
        assert!((max - 1300.0).abs() < 1e-9);
    }

    #[test]
    fn population_avg_elo_empty() {
        let pop = SketchPopulation::new(4);
        assert!((pop.avg_elo() - DEFAULT_ELO).abs() < 1e-9);
    }

    #[test]
    fn population_avg_elo() {
        let mut pop = SketchPopulation::new(4);
        pop.insert(make_entry(1200.0));
        pop.insert(make_entry(1400.0));

        let avg = pop.avg_elo();
        assert!((avg - 1300.0).abs() < 1e-9);
    }

    #[test]
    fn population_total_visits() {
        let mut pop = SketchPopulation::new(4);
        let e1 = make_entry_with_visits(1200.0, 5);
        let e2 = make_entry_with_visits(1300.0, 3);
        pop.insert(e1);
        pop.insert(e2);

        assert_eq!(pop.total_visits(), 8);
    }

    #[test]
    fn population_is_full() {
        let mut pop = SketchPopulation::new(2);
        assert!(!pop.is_full());

        pop.insert(make_entry(1200.0));
        assert!(!pop.is_full());

        pop.insert(make_entry(1300.0));
        assert!(pop.is_full());
    }

    #[test]
    fn population_sorted_by_elo() {
        let mut pop = SketchPopulation::new(4);
        pop.insert(make_entry(1100.0));
        pop.insert(make_entry(1300.0));
        pop.insert(make_entry(1200.0));

        let sorted = pop.sorted_by_elo();
        let elos: Vec<f64> = sorted.iter().map(|e| e.elo_rating).collect();
        assert_eq!(elos, vec![1300.0, 1200.0, 1100.0]);
    }

    #[test]
    fn population_top_k_truncates() {
        let mut pop = SketchPopulation::new(2);
        pop.insert(make_entry(1100.0));
        pop.insert(make_entry(1300.0));
        pop.insert(make_entry(1200.0));

        let top = pop.top_k();
        assert_eq!(top.len(), 2);
        assert!((top[0].elo_rating - 1300.0).abs() < 1e-9);
        assert!((top[1].elo_rating - 1200.0).abs() < 1e-9);
    }

    #[test]
    fn population_sample_eligible() {
        let mut pop = SketchPopulation::new(4);
        pop.insert(make_entry_with_visits(1200.0, 0));
        pop.insert(make_entry_with_visits(1300.0, 5));
        pop.insert(make_entry_with_visits(1400.0, 2));

        let eligible = pop.sample_eligible(2);
        assert_eq!(eligible.len(), 2, "visits 0 and 2 are <= 2");
    }

    #[test]
    fn population_ids() {
        let mut pop = SketchPopulation::new(4);
        let e1 = make_entry(1200.0);
        let id1 = e1.id;
        pop.insert(e1);

        let ids = pop.ids();
        assert_eq!(ids, vec![id1]);
    }

    // ── Clear Tests ────────────────────────────────────────────

    #[test]
    fn population_clear() {
        let mut pop = SketchPopulation::new(4);
        pop.insert(make_entry(1200.0));
        pop.insert(make_entry(1300.0));
        assert_eq!(pop.len(), 2);

        pop.clear();
        assert!(pop.is_empty());
        assert_eq!(pop.total_visits(), 0);
    }

    // ── Display Tests ──────────────────────────────────────────

    #[test]
    fn population_display_empty() {
        let pop = SketchPopulation::new(4);
        let display = format!("{pop}");
        assert!(display.contains("entries=0/4"));
        assert!(display.contains("elo=—..—"));
    }

    #[test]
    fn population_display_with_entries() {
        let mut pop = SketchPopulation::new(4);
        pop.insert(make_entry(1200.0));
        pop.insert(make_entry(1400.0));

        let display = format!("{pop}");
        assert!(display.contains("entries=2/4"));
        assert!(display.contains("elo=1200..1400"));
    }

    // ── Memory Tests ───────────────────────────────────────────

    #[test]
    fn population_estimated_memory() {
        let pop = SketchPopulation::new(4);
        assert_eq!(pop.estimated_memory_bytes(), 0);

        let mut pop = SketchPopulation::new(4);
        pop.insert(make_entry(1200.0));
        assert!(pop.estimated_memory_bytes() > 0);
    }

    // ── Large Population Tests ─────────────────────────────────

    #[test]
    fn population_paper_size_64() {
        let mut pop = SketchPopulation::with_paper_defaults();

        for i in 0..100 {
            pop.insert(make_entry(1000.0 + i as f64 * 10.0));
        }

        assert_eq!(pop.len(), 64, "should cap at 64");

        let (min, max) = pop.elo_range().unwrap();
        assert!(
            min >= 1360.0,
            "lowest Elo should be around 1360+ (top 64 of 100)"
        );
        assert!((max - 1990.0).abs() < 1e-9);
    }

    #[test]
    fn population_config_with_overshoot_delays_eviction() {
        let mut pop = SketchPopulation::with_config(PopulationConfig::with_overshoot(2, 4));

        pop.insert(make_entry(1000.0));
        pop.insert(make_entry(1100.0));
        assert_eq!(pop.len(), 2);

        // Still within max_population=4
        let report = pop.insert(make_entry(1200.0));
        assert!(
            !report.did_evict(),
            "should not evict within max_population"
        );
        assert_eq!(pop.len(), 3);

        // Exceeds top_k=2, but still <= max_population=4
        let report = pop.insert(make_entry(1300.0));
        assert!(!report.did_evict());
        assert_eq!(pop.len(), 4);

        // Now exceeds max_population=4 → eviction to top_k=2
        let report = pop.insert(make_entry(1400.0));
        assert!(report.did_evict());
        assert_eq!(pop.len(), 2);
    }
}
