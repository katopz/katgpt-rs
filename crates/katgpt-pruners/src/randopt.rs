//! RandOpt Weight-Space Perturbation Ensembling (Plan 121, Research 080).
//!
//! Implements random weight perturbation + top-K ensembling as a
//! BanditPruner-compatible protocol. Based on the RandOpt approach from
//! Neural Thickets: generate N perturbed copies of base weights (arms),
//! score each, ensemble the top-K for robust predictions.
//!
//! # Architecture
//!
//! - [`RandOptConfig`] — population size, ensemble size, sigma set, seed
//! - [`RandOptWeightSampler`] — generates θ' = θ + σ·ε(seed) perturbations
//! - [`RandOptScorer`] — trait for scoring perturbed weight vectors
//! - [`AccuracyScorer`] — simple accuracy scorer for discrete-answer tasks
//! - [`RandOptEnsemble`] — majority-vote + mean aggregation
//! - [`RandOptSession`] — orchestrates the full pipeline
//!
//! # Usage
//!
//! ```rust,ignore
//! let config = RandOptConfig::default();
//! let session = RandOptSession::new(config);
//! let scorer = AccuracyScorer { expected: &target, threshold: 0.5 };
//! let result = session.run(&base_weights, &scorer);
//! println!("Ensemble score: {:.4}", result.ensemble_score);
//! ```

// ── Helpers ─────────────────────────────────────────────────────

/// xorshift64-based hash for deterministic pseudo-random noise.
fn simple_hash(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Map a hash seed to a unit-scale value [0, 1].
fn hash_to_unit(seed: u64) -> f32 {
    (simple_hash(seed) as f32) / (u64::MAX as f32)
}

// ── Config ──────────────────────────────────────────────────────

/// RandOpt configuration.
#[derive(Clone, Debug)]
pub struct RandOptConfig {
    /// Number of random perturbation arms (paper: 100-1000).
    pub population_size: usize,
    /// Number of top-K arms to ensemble (paper: 10-50).
    pub ensemble_size: usize,
    /// Set of perturbation magnitudes to try (paper: {0.01, 0.02, 0.03}).
    pub sigma_set: Vec<f32>,
    /// Base seed for reproducibility.
    pub base_seed: u64,
}

impl Default for RandOptConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            ensemble_size: 10,
            sigma_set: vec![0.01, 0.02, 0.03],
            base_seed: 42,
        }
    }
}

// ── Weight Sampler ──────────────────────────────────────────────

/// Generates weight perturbations θ' = θ + σ·ε(seed).
pub struct RandOptWeightSampler {
    config: RandOptConfig,
}

impl RandOptWeightSampler {
    /// Create a new sampler with the given configuration.
    pub fn new(config: RandOptConfig) -> Self {
        Self { config }
    }

    /// Generate a perturbed weight vector for arm `i`.
    ///
    /// Uses deterministic seed = base_seed + i as u64.
    /// σ assigned round-robin from sigma_set.
    pub fn perturb(&self, base_weights: &[f32], arm_index: usize) -> Vec<f32> {
        let seed_i = self.config.base_seed.wrapping_add(arm_index as u64);
        let sigma = self.sigma_for_arm(arm_index);

        base_weights
            .iter()
            .enumerate()
            .map(|(j, &w)| {
                let noise_seed = seed_i.wrapping_add(j as u64);
                // Map to [-1, 1] via (hash_to_unit * 2.0 - 1.0)
                let noise = hash_to_unit(noise_seed) * 2.0 - 1.0;
                w + sigma * noise
            })
            .collect()
    }

    /// Get the sigma for a given arm index (round-robin).
    pub fn sigma_for_arm(&self, arm_index: usize) -> f32 {
        self.config.sigma_set[arm_index % self.config.sigma_set.len()]
    }
}

// ── Scorer ──────────────────────────────────────────────────────

/// Trait for scoring perturbed weights.
pub trait RandOptScorer: Send + Sync {
    /// Score a perturbed weight vector. Higher = better.
    fn score(&self, weights: &[f32]) -> f32;
}

/// Simple accuracy scorer for discrete-answer tasks.
pub struct AccuracyScorer<'a> {
    /// Expected outputs for each input.
    pub expected: &'a [f32],
    /// Threshold for correct prediction.
    pub threshold: f32,
}

impl<'a> RandOptScorer for AccuracyScorer<'a> {
    fn score(&self, weights: &[f32]) -> f32 {
        let len = match weights.len() {
            0 => return 0.0,
            n => n,
        };
        let correct = weights
            .iter()
            .zip(self.expected.iter())
            .filter(|(w, e)| (*w - *e).abs() < self.threshold)
            .count();
        correct as f32 / len as f32
    }
}

// ── Ensemble ────────────────────────────────────────────────────

/// Majority-vote + mean aggregation for ensembles.
pub struct RandOptEnsemble {
    /// Number of ensemble members.
    pub ensemble_size: usize,
}

impl RandOptEnsemble {
    /// Create a new ensemble with the given size.
    pub fn new(ensemble_size: usize) -> Self {
        Self { ensemble_size }
    }

    /// Aggregate discrete predictions via majority vote.
    pub fn aggregate_discrete(&self, predictions: &[u32]) -> u32 {
        match predictions.is_empty() {
            true => 0,
            false => {
                let mut counts: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::new();
                for &p in predictions {
                    *counts.entry(p).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(v, _)| v)
                    .unwrap_or(0)
            }
        }
    }

    /// Aggregate continuous predictions via mean.
    pub fn aggregate_continuous(&self, predictions: &[f32]) -> f32 {
        match predictions.is_empty() {
            true => 0.0,
            false => predictions.iter().sum::<f32>() / predictions.len() as f32,
        }
    }
}

// ── Result ──────────────────────────────────────────────────────

/// Result of a RandOpt session.
#[derive(Clone, Debug)]
pub struct RandOptResult {
    /// Indices of top-K arms.
    pub top_k_indices: Vec<usize>,
    /// Scores for all arms.
    pub scores: Vec<f32>,
    /// Seeds of best arms.
    pub best_seeds: Vec<u64>,
    /// Sigmas of best arms.
    pub best_sigmas: Vec<f32>,
    /// Base score (unperturbed).
    pub base_score: f32,
    /// Ensemble score (average of top-K).
    pub ensemble_score: f32,
    /// Solution density at margin=0.
    pub solution_density: f32,
}

// ── Session ─────────────────────────────────────────────────────

/// Orchestrates the full RandOpt pipeline.
pub struct RandOptSession {
    config: RandOptConfig,
    sampler: RandOptWeightSampler,
}

impl RandOptSession {
    /// Create a new RandOpt session with the given configuration.
    pub fn new(config: RandOptConfig) -> Self {
        let sampler = RandOptWeightSampler::new(config.clone());
        Self { config, sampler }
    }

    /// Run the full RandOpt pipeline on base weights.
    pub fn run(&self, base_weights: &[f32], scorer: &dyn RandOptScorer) -> RandOptResult {
        use super::bandit::solution_density;

        // 1. Score base weights
        let base_score = scorer.score(base_weights);

        // 2. For each arm in population: perturb + score
        let scores: Vec<f32> = (0..self.config.population_size)
            .map(|i| {
                let perturbed = self.sampler.perturb(base_weights, i);
                scorer.score(&perturbed)
            })
            .collect();

        // 3. Find top-K by score (stable sort descending)
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.config.ensemble_size.min(indexed.len());
        let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

        let top_k_indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
        let best_seeds: Vec<u64> = top_k_indices
            .iter()
            .map(|&i| self.config.base_seed.wrapping_add(i as u64))
            .collect();
        let best_sigmas: Vec<f32> = top_k_indices
            .iter()
            .map(|&i| self.sampler.sigma_for_arm(i))
            .collect();

        // 4. Compute ensemble score (average of top-K)
        let ensemble_score = match top_k.is_empty() {
            true => base_score,
            false => top_k.iter().map(|(_, s)| *s).sum::<f32>() / top_k.len() as f32,
        };

        // 5. Compute solution density at margin=0
        let sol_density = solution_density(&scores, base_score, 0.0);

        RandOptResult {
            top_k_indices,
            scores,
            best_seeds,
            best_sigmas,
            base_score,
            ensemble_score,
            solution_density: sol_density,
        }
    }
}
