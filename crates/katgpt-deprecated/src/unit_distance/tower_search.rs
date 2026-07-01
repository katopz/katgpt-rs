//! G-Zero modelless tower parameter search for optimal CM field constructions.
//!
//! Implements Plan 090 T5: InfiniteTowerSearch — a bandit-driven search over
//! tower parameters (field family, split primes, degree, denominator) to
//! maximize the exponent δ in the unit distance bound ν(n) ≥ n^(1+δ).
//!
//! # Architecture
//!
//! This is a **domain-specific application of G-Zero Phase 1 modelless search**.
//! The G-Zero infrastructure (UCB1 bandit, delta-gated promotion) is repurposed:
//!
//! ```text
//! G-Zero concept            → Tower search analog
//! ────────────────────────────────────────────────
//! Query (q)                 → Field family (qi, q_sqrt5_i, pro2, cyclotomic)
//! Hint (h)                  → Parameter configuration (split primes, denom)
//! HintDelta δ               → DeltaEstimate.delta (the actual δ exponent)
//! Arm                        → (family, num_primes, denominator) config
//! DeltaBanditPruner          → Bandit over field configurations
//! TemplateProposer           → Deterministic tower parameter generator
//! ```
//!
//! # Why Bandit Search?
//!
//! The parameter space has structure amenable to bandit optimization:
//! - δ depends on `t·ln(2) - ln(h)` (monotone in split prime count for h=1)
//! - But δ = γ / (4·log(4·R·D)) — more primes increase D, so the ratio is non-trivial
//! - Different field families have different root discriminants
//! - UCB1 naturally balances exploration (new families) vs exploitation (best-known configs)
//!
//! # Feature Gate
//!
//! Requires both `unit_distance` and `g_zero` features.

use super::cm_field::{CmField, class_number_bound, enumerate_split_primes};
use super::types::{CmFieldParams, DeltaEstimate};

// ── TowerArm ──────────────────────────────────────────────────

/// A search arm representing a specific field configuration to evaluate.
///
/// Each arm encapsulates enough information to construct a `CmField`
/// and compute its δ value. Arms are ordered by their index for bandit tracking.
#[derive(Clone, Debug)]
pub struct TowerArm {
    /// Arm index (for bandit bookkeeping).
    pub id: usize,
    /// Field family name.
    pub family: TowerFamily,
    /// Number of split primes to use.
    pub num_split_primes: usize,
    /// Denominator D for lattice embedding.
    pub denominator: u64,
    /// Degree f of the totally real subfield.
    pub degree: usize,
    /// Root discriminant of the field.
    pub root_discriminant: f64,
}

/// Field family for tower parameter search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TowerFamily {
    /// Q(i) — Gaussian integers, degree 1, h=1, rd=1.
    Qi,
    /// Q(√5, i) — degree 2, h=1, rd=√5.
    QSqrt5I,
    /// Pro-2 tower base — degree 6, from Remarks paper.
    Pro2Tower,
    /// Generic cyclotomic extension — configurable degree.
    Cyclotomic { degree: usize },
}

impl TowerFamily {
    /// Default root discriminant for this family.
    pub fn default_root_disc(&self) -> f64 {
        match self {
            Self::Qi => 1.0,
            Self::QSqrt5I => 5.0_f64.sqrt(),
            Self::Pro2Tower => 5.36, // From Remarks paper
            Self::Cyclotomic { degree } => {
                // Rough bound: cyclotomic discriminant grows with degree
                (1..=*degree).fold(1.0, |acc, k| acc * (k as f64).ln().max(1.0))
            }
        }
    }

    /// Default class number for this family.
    pub fn default_class_number(&self) -> u64 {
        match self {
            Self::Qi => 1,
            Self::QSqrt5I => 1,
            Self::Pro2Tower => 1, // Estimated
            Self::Cyclotomic { .. } => 1,
        }
    }

    /// Default degree f for this family.
    pub fn default_degree(&self) -> usize {
        match self {
            Self::Qi => 1,
            Self::QSqrt5I => 2,
            Self::Pro2Tower => 6,
            Self::Cyclotomic { degree } => *degree,
        }
    }
}

impl std::fmt::Display for TowerFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Qi => write!(f, "Q(i)"),
            Self::QSqrt5I => write!(f, "Q(√5,i)"),
            Self::Pro2Tower => write!(f, "Pro-2"),
            Self::Cyclotomic { degree } => write!(f, "Cyc({})", degree),
        }
    }
}

impl TowerArm {
    /// Build the CM field parameters for this arm.
    pub fn to_field_params(&self) -> CmFieldParams {
        let split_primes = enumerate_split_primes(200)
            .into_iter()
            .take(self.num_split_primes)
            .collect();

        CmFieldParams {
            degree: self.degree,
            split_primes,
            class_number: self.family.default_class_number(),
            root_discriminant: self.root_discriminant,
            denominator: self.denominator,
        }
    }

    /// Compute δ for this arm's configuration.
    ///
    /// Returns `None` if γ ≤ 0 (construction doesn't work for these parameters).
    pub fn compute_delta(&self) -> Option<DeltaEstimate> {
        let params = self.to_field_params();
        DeltaEstimate::from_field_params(&params)
    }

    /// Build the actual CmField for this arm.
    ///
    /// Uses the family-specific constructor when available,
    /// falls back to `CmField::from_params` for generic arms.
    pub fn build_field(&self) -> CmField {
        let split_primes = enumerate_split_primes(200)
            .into_iter()
            .take(self.num_split_primes)
            .collect();

        match self.family {
            TowerFamily::Qi => CmField::qi(split_primes),
            TowerFamily::QSqrt5I => CmField::q_sqrt5_i(split_primes, self.denominator),
            TowerFamily::Pro2Tower => CmField::pro2_tower_base(),
            TowerFamily::Cyclotomic { degree } => {
                let root_disc = self.root_discriminant;
                let class_num = class_number_bound(root_disc, degree);
                let params = CmFieldParams {
                    degree,
                    split_primes,
                    class_number: class_num,
                    root_discriminant: root_disc,
                    denominator: self.denominator,
                };
                let exponents = vec![1; params.split_primes.len()];
                CmField::from_params(&format!("Cyc({})", degree), params, exponents)
            }
        }
    }

    /// Human-readable label for logging.
    pub fn label(&self) -> String {
        format!(
            "{}:t={}:D={}",
            self.family, self.num_split_primes, self.denominator
        )
    }
}

// ── TowerBandit ───────────────────────────────────────────────

/// UCB1 bandit over tower parameter configurations.
///
/// Tracks accumulated δ rewards per arm and selects the next arm to
/// evaluate using the Upper Confidence Bound formula:
///
/// ```text
/// UCB(i) = μ(i) + c × sqrt(ln(N) / n(i))
/// ```
///
/// where μ(i) is the mean reward, n(i) is the pull count, N is total pulls,
/// and c is the exploration constant (default sqrt(2)).
#[derive(Clone, Debug)]
pub struct TowerBandit {
    /// Arms in the search space.
    arms: Vec<TowerArm>,
    /// Accumulated δ rewards per arm.
    rewards: Vec<f64>,
    /// Pull counts per arm.
    pulls: Vec<usize>,
    /// Total pulls across all arms.
    total_pulls: usize,
    /// Exploration constant (default: sqrt(2)).
    exploration_c: f64,
    /// Best δ found so far.
    best_delta: f64,
    /// Index of the best arm.
    best_arm: usize,
}

impl TowerBandit {
    /// Create a new bandit over the given arms.
    pub fn new(arms: Vec<TowerArm>) -> Self {
        let n = arms.len();
        Self {
            arms,
            rewards: vec![0.0; n],
            pulls: vec![0; n],
            total_pulls: 0,
            exploration_c: 2.0_f64.sqrt(),
            best_delta: 0.0,
            best_arm: 0,
        }
    }

    /// Set the exploration constant.
    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration_c = c;
        self
    }

    /// Number of arms.
    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    /// Evaluate the arm at the given index (computes its δ).
    pub fn evaluate_selected(&self, idx: usize) -> f64 {
        TowerSearch::evaluate_arm(&self.arms[idx])
    }

    /// Select the next arm to evaluate using UCB1.
    ///
    /// During warm-up (first N pulls), each arm is pulled once.
    /// After warm-up, UCB1 formula balances exploration vs exploitation.
    pub fn select(&mut self) -> usize {
        // Warm-up: pull each arm once
        for i in 0..self.arms.len() {
            if self.pulls[i] == 0 {
                return i;
            }
        }

        // UCB1 selection
        let mut best_idx = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        for i in 0..self.arms.len() {
            let mean = self.rewards[i] / self.pulls[i] as f64;
            let exploration = self.exploration_c * (self.total_pulls as f64).ln().sqrt()
                / (self.pulls[i] as f64).sqrt();
            let ucb = mean + exploration;

            if ucb > best_ucb {
                best_ucb = ucb;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Record the δ reward for a pulled arm.
    pub fn observe(&mut self, arm_idx: usize, delta: f64) {
        self.rewards[arm_idx] += delta;
        self.pulls[arm_idx] += 1;
        self.total_pulls += 1;

        let mean = self.rewards[arm_idx] / self.pulls[arm_idx] as f64;
        if mean > self.best_delta {
            self.best_delta = mean;
            self.best_arm = arm_idx;
        }
    }

    /// Best arm found so far.
    pub fn best_arm(&self) -> &TowerArm {
        &self.arms[self.best_arm]
    }

    /// Best mean δ found so far.
    pub fn best_delta(&self) -> f64 {
        self.best_delta
    }

    /// Get arm statistics (arm_id, mean_reward, pull_count).
    pub fn stats(&self) -> Vec<(usize, f64, usize)> {
        self.arms
            .iter()
            .enumerate()
            .map(|(i, arm)| {
                let mean = if self.pulls[i] > 0 {
                    self.rewards[i] / self.pulls[i] as f64
                } else {
                    0.0
                };
                (arm.id, mean, self.pulls[i])
            })
            .collect()
    }

    /// Top-K arms by mean reward.
    pub fn top_k(&self, k: usize) -> Vec<(&TowerArm, f64, usize)> {
        let mut stats: Vec<_> = self
            .arms
            .iter()
            .enumerate()
            .map(|(i, arm)| {
                let mean = if self.pulls[i] > 0 {
                    self.rewards[i] / self.pulls[i] as f64
                } else {
                    0.0
                };
                (arm, mean, self.pulls[i])
            })
            .collect();

        stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        stats.into_iter().take(k).collect()
    }
}

// ── TowerSearchConfig ─────────────────────────────────────────

/// Configuration for the infinite tower parameter search.
#[derive(Clone, Debug)]
pub struct TowerSearchConfig {
    /// Number of UCB1 rounds to run.
    pub num_rounds: usize,
    /// Split prime counts to search (e.g., [1, 2, 3, 4, 5, 8, 12]).
    pub prime_counts: Vec<usize>,
    /// Denominators to search (e.g., [1, 2, 4, 8]).
    pub denominators: Vec<u64>,
    /// Field families to include.
    pub families: Vec<TowerFamily>,
    /// Exploration constant for UCB1.
    pub exploration_c: f64,
    /// Minimum δ to count as a valid result.
    pub delta_floor: f64,
}

impl Default for TowerSearchConfig {
    fn default() -> Self {
        Self {
            num_rounds: 100,
            prime_counts: vec![1, 2, 3, 4, 5, 8, 12],
            denominators: vec![1, 2, 4],
            families: vec![TowerFamily::Qi, TowerFamily::QSqrt5I],
            exploration_c: 2.0_f64.sqrt(),
            delta_floor: 0.0,
        }
    }
}

// ── TowerSearchResult ─────────────────────────────────────────

/// Result of the tower parameter search.
#[derive(Clone, Debug)]
pub struct TowerSearchResult {
    /// Best arm found.
    pub best_arm: TowerArm,
    /// Best δ achieved.
    pub best_delta: f64,
    /// Best CmField constructed.
    pub best_field: CmField,
    /// Number of arms searched.
    pub arms_searched: usize,
    /// Total rounds executed.
    pub rounds: usize,
    /// All arm results sorted by δ (best first).
    pub rankings: Vec<(TowerArm, f64)>,
    /// Whether the search found a configuration with δ > delta_floor.
    pub success: bool,
}

// ── TowerSearch ───────────────────────────────────────────────

/// Infinite tower parameter search engine.
///
/// Generates a search space of `(family, num_primes, denominator)` configurations,
/// then uses UCB1 bandit to efficiently explore the space and find the configuration
/// maximizing δ in the unit distance bound ν(n) ≥ n^(1+δ).
///
/// # Usage
///
/// ```rust,ignore
/// let config = TowerSearchConfig::default();
/// let result = TowerSearch::run(&config);
/// println!("Best δ: {:.6} from {}", result.best_delta, result.best_arm.label());
/// ```
pub struct TowerSearch;

impl TowerSearch {
    /// Generate the search space from configuration.
    pub fn generate_arms(config: &TowerSearchConfig) -> Vec<TowerArm> {
        let mut arms = Vec::new();
        let mut id = 0;

        for &family in &config.families {
            let default_degree = family.default_degree();
            let default_rd = family.default_root_disc();

            for &num_primes in &config.prime_counts {
                for &denom in &config.denominators {
                    arms.push(TowerArm {
                        id,
                        family,
                        num_split_primes: num_primes,
                        denominator: denom,
                        degree: default_degree,
                        root_discriminant: default_rd,
                    });
                    id += 1;
                }
            }
        }

        arms
    }

    /// Evaluate a single arm — compute its δ.
    pub fn evaluate_arm(arm: &TowerArm) -> f64 {
        arm.compute_delta().map(|d| d.delta).unwrap_or(0.0)
    }

    /// Run the full search.
    ///
    /// 1. Generate arms from configuration
    /// 2. UCB1 bandit over arms, evaluating δ for each pull
    /// 3. Return the best configuration found
    pub fn run(config: &TowerSearchConfig) -> TowerSearchResult {
        let arms = Self::generate_arms(config);
        let mut bandit = TowerBandit::new(arms).with_exploration(config.exploration_c);

        // UCB1 search loop
        for _ in 0..config.num_rounds {
            let arm_idx = bandit.select();
            let delta = Self::evaluate_arm(&bandit.arms[arm_idx]);
            if delta > config.delta_floor {
                bandit.observe(arm_idx, delta);
            } else {
                bandit.observe(arm_idx, 0.0);
            }
        }

        // Build final results
        let best_arm = bandit.best_arm().clone();
        let best_delta = bandit.best_delta();
        let best_field = best_arm.build_field();

        let mut rankings: Vec<_> = bandit
            .arms
            .iter()
            .enumerate()
            .map(|(i, arm)| {
                let mean = if bandit.pulls()[i] > 0 {
                    bandit.rewards()[i] / bandit.pulls()[i] as f64
                } else {
                    0.0
                };
                (arm.clone(), mean)
            })
            .collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        TowerSearchResult {
            best_arm,
            best_delta,
            best_field,
            arms_searched: bandit.num_arms(),
            rounds: config.num_rounds,
            rankings,
            success: best_delta > config.delta_floor,
        }
    }
}

// ── Accessor methods for TowerBandit private fields ───────────

impl TowerBandit {
    /// Access pulls vector (for result computation).
    fn pulls(&self) -> &[usize] {
        &self.pulls
    }

    /// Access rewards vector (for result computation).
    fn rewards(&self) -> &[f64] {
        &self.rewards
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tower_arm_qi_delta_positive() {
        let arm = TowerArm {
            id: 0,
            family: TowerFamily::Qi,
            num_split_primes: 3,
            denominator: 1,
            degree: 1,
            root_discriminant: 1.0,
        };
        let delta = arm.compute_delta();
        assert!(
            delta.is_some(),
            "Q(i) with 3 split primes should yield δ > 0"
        );
        assert!(delta.unwrap().delta > 0.0);
    }

    #[test]
    fn tower_arm_qi_more_primes_larger_delta() {
        let arm3 = TowerArm {
            id: 0,
            family: TowerFamily::Qi,
            num_split_primes: 3,
            denominator: 1,
            degree: 1,
            root_discriminant: 1.0,
        };
        let arm8 = TowerArm {
            id: 1,
            family: TowerFamily::Qi,
            num_split_primes: 8,
            denominator: 1,
            degree: 1,
            root_discriminant: 1.0,
        };

        let d3 = arm3.compute_delta().unwrap().delta;
        let d8 = arm8.compute_delta().unwrap().delta;
        assert!(
            d8 > d3,
            "More split primes should give larger δ: {} vs {}",
            d8,
            d3
        );
    }

    #[test]
    fn tower_arm_qi_zero_primes_no_delta() {
        let arm = TowerArm {
            id: 0,
            family: TowerFamily::Qi,
            num_split_primes: 0,
            denominator: 1,
            degree: 1,
            root_discriminant: 1.0,
        };
        let delta = arm.compute_delta();
        assert!(delta.is_none(), "Zero split primes → γ ≤ 0 → no δ");
    }

    #[test]
    fn tower_bandit_warmup_pulls_all() {
        let arms = vec![
            TowerArm {
                id: 0,
                family: TowerFamily::Qi,
                num_split_primes: 2,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
            TowerArm {
                id: 1,
                family: TowerFamily::Qi,
                num_split_primes: 3,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
            TowerArm {
                id: 2,
                family: TowerFamily::QSqrt5I,
                num_split_primes: 2,
                denominator: 1,
                degree: 2,
                root_discriminant: 5.0_f64.sqrt(),
            },
        ];

        let mut bandit = TowerBandit::new(arms);

        // First 3 selections should be 0, 1, 2 (warm-up)
        assert_eq!(bandit.select(), 0);
        bandit.observe(0, 0.1);

        assert_eq!(bandit.select(), 1);
        bandit.observe(1, 0.2);

        assert_eq!(bandit.select(), 2);
        bandit.observe(2, 0.05);

        // After warm-up, should select arm 1 (highest reward)
        let next = bandit.select();
        assert_eq!(next, 1, "After warm-up, should prefer highest-reward arm");
    }

    #[test]
    fn tower_bandit_best_delta_tracking() {
        let arms = vec![
            TowerArm {
                id: 0,
                family: TowerFamily::Qi,
                num_split_primes: 1,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
            TowerArm {
                id: 1,
                family: TowerFamily::Qi,
                num_split_primes: 5,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
        ];

        let mut bandit = TowerBandit::new(arms);

        // Pull arm 0
        let idx = bandit.select();
        let delta = TowerSearch::evaluate_arm(&bandit.arms[idx]);
        bandit.observe(idx, delta);

        // Pull arm 1
        let idx = bandit.select();
        let delta = TowerSearch::evaluate_arm(&bandit.arms[idx]);
        bandit.observe(idx, delta);

        // Best should be arm 1 (more split primes = higher δ)
        assert!(bandit.best_delta() > 0.0, "Should find positive δ");
        assert_eq!(
            bandit.best_arm().id,
            1,
            "Arm with more split primes should win"
        );
    }

    #[test]
    fn tower_search_default_finds_positive_delta() {
        let config = TowerSearchConfig::default();
        let result = TowerSearch::run(&config);

        assert!(
            result.best_delta > 0.0,
            "Default search should find positive δ, got {}",
            result.best_delta
        );
        assert!(result.success, "Search should succeed");
        assert!(!result.rankings.is_empty(), "Should have rankings");
    }

    #[test]
    fn tower_search_more_primes_wins() {
        let config = TowerSearchConfig {
            num_rounds: 50,
            families: vec![TowerFamily::Qi],
            prime_counts: vec![1, 2, 4, 8, 12],
            denominators: vec![1],
            ..Default::default()
        };
        let result = TowerSearch::run(&config);

        // The best arm should have many split primes
        assert!(
            result.best_arm.num_split_primes >= 4,
            "Best arm should use at least 4 split primes, got {}",
            result.best_arm.num_split_primes
        );
    }

    #[test]
    fn tower_search_rankings_sorted() {
        let config = TowerSearchConfig {
            num_rounds: 50,
            ..Default::default()
        };
        let result = TowerSearch::run(&config);

        // Rankings should be sorted descending
        for window in result.rankings.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Rankings should be sorted descending: {} >= {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn tower_arm_build_field_qi() {
        let arm = TowerArm {
            id: 0,
            family: TowerFamily::Qi,
            num_split_primes: 3,
            denominator: 1,
            degree: 1,
            root_discriminant: 1.0,
        };
        let field = arm.build_field();
        assert_eq!(field.params.degree, 1);
        assert_eq!(field.params.split_primes.len(), 3);
        assert!(field.verify_split_primes());
    }

    #[test]
    fn tower_arm_build_field_pro2() {
        let arm = TowerArm {
            id: 0,
            family: TowerFamily::Pro2Tower,
            num_split_primes: 1,
            denominator: 1,
            degree: 6,
            root_discriminant: 5.36,
        };
        let field = arm.build_field();
        assert_eq!(field.params.degree, 6);
    }

    #[test]
    fn tower_family_display() {
        assert_eq!(format!("{}", TowerFamily::Qi), "Q(i)");
        assert_eq!(format!("{}", TowerFamily::QSqrt5I), "Q(√5,i)");
        assert_eq!(format!("{}", TowerFamily::Pro2Tower), "Pro-2");
        assert_eq!(
            format!("{}", TowerFamily::Cyclotomic { degree: 3 }),
            "Cyc(3)"
        );
    }

    #[test]
    fn tower_search_stats() {
        let arms = vec![
            TowerArm {
                id: 0,
                family: TowerFamily::Qi,
                num_split_primes: 2,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
            TowerArm {
                id: 1,
                family: TowerFamily::Qi,
                num_split_primes: 4,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
        ];
        let bandit = TowerBandit::new(arms);
        let stats = bandit.stats();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].0, 0);
        assert_eq!(stats[1].0, 1);
    }

    #[test]
    fn tower_search_top_k() {
        let mut bandit = TowerBandit::new(vec![
            TowerArm {
                id: 0,
                family: TowerFamily::Qi,
                num_split_primes: 2,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
            TowerArm {
                id: 1,
                family: TowerFamily::Qi,
                num_split_primes: 4,
                denominator: 1,
                degree: 1,
                root_discriminant: 1.0,
            },
            TowerArm {
                id: 2,
                family: TowerFamily::QSqrt5I,
                num_split_primes: 3,
                denominator: 1,
                degree: 2,
                root_discriminant: 5.0_f64.sqrt(),
            },
        ]);

        // Pull all arms
        for _ in 0..3 {
            let idx = bandit.select();
            let delta = TowerSearch::evaluate_arm(&bandit.arms[idx]);
            bandit.observe(idx, delta);
        }

        let top = bandit.top_k(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].1 >= top[1].1, "Top-K should be sorted");
    }
}
