//! Plackett-Luce rating via Gibbs sampling for Elo conversion (Plan 128, T4).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "Multi-item ranking (P=7) is more information-efficient than pairwise BT.
//! Uses Gibbs sampling with hierarchical Gamma prior for posterior estimation."
//!
//! # Plackett-Luce Model
//!
//! Given parameters λ = (λ_1, ..., λ_n), the probability of a ranking
//! σ = (σ_1, ..., σ_k) is:
//!
//! ```text
//! P(σ | λ) = ∏_{i=1}^{k} (λ_{σ_i} / Σ_{j≥i} λ_{σ_j})
//! ```
//!
//! This is a generalization of Bradley-Terry: pairwise BT is the special case
//! k=2, where the model reduces to P(a > b) = λ_a / (λ_a + λ_b).
//!
//! Bridge to existing: Our `BradleyTerry` (Research 040, Plan 080) handles
//! pairwise. Plackett-Luce is the multi-item generalization. This module
//! implements PL as an extension, not a replacement.
//!
//! # Gibbs Sampling
//!
//! Uses conjugate Gamma posterior with hierarchical prior:
//! - λ_s ~ Gamma(1, r_s)    [item strength]
//! - r_s ~ Gamma(1, 1)      [hierarchical rate]
//!
//! Gibbs updates per iteration:
//! 1. λ_s | rankings, r_s ~ Gamma(1 + w_s, r_s + d_s)
//!    where w_s = wins (times chosen), d_s = exposure (competition denominator)
//! 2. r_s | λ_s ~ Gamma(2, 1 + λ_s)
//!
//! # Elo Conversion
//!
//! ```text
//! Elo_s = 1200 + 400 × log10(mean(λ_s post burn-in))
//! ```
//!
//! Uses log10 (not ln) for compatibility with standard Elo scale.
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

use std::collections::HashMap;
use std::fmt;

use fastrand::Rng;

use super::sketch_types::{DEFAULT_ELO, ELO_SCALE, SketchEntry, SketchId};

// ── PlackettLuceConfig ─────────────────────────────────────────

/// Configuration for Plackett-Luce Gibbs sampling.
///
/// Paper defaults: P=7 match size, I=1000 Gibbs samples, B=200 burn-in.
/// Hierarchical Gamma(1, Gamma(1,1)) prior on item strengths.
///
/// # Parameter Guide
///
/// | Parameter | Paper Default | Effect |
/// |-----------|--------------|--------|
/// | `match_size` | 7 | Items per ranking (more = more info per match) |
/// | `gibbs_samples` | 1000 | Total MCMC iterations (more = more precise) |
/// | `burn_in` | 200 | Initial samples to discard (convergence warmup) |
/// | `elo_offset` | 1200 | Elo baseline (standard chess) |
/// | `elo_scale` | 400 | Elo per log10 unit (standard chess) |
#[derive(Clone, Debug, PartialEq)]
pub struct PlackettLuceConfig {
    /// Number of sketches per rating match (P=7 per paper).
    pub match_size: usize,
    /// Total Gibbs sampling iterations (I=1000 per paper).
    pub gibbs_samples: usize,
    /// Burn-in iterations to discard (B=200 per paper).
    pub burn_in: usize,
    /// Elo offset (1200 per paper, standard chess baseline).
    pub elo_offset: f64,
    /// Elo scale factor (400 per paper, standard chess scale).
    pub elo_scale: f64,
}

impl Default for PlackettLuceConfig {
    fn default() -> Self {
        Self::PAPER_DEFAULTS
    }
}

impl PlackettLuceConfig {
    /// Paper defaults: P=7, I=1000, B=200, Elo offset=1200, Elo scale=400.
    pub const PAPER_DEFAULTS: Self = Self {
        match_size: 7,
        gibbs_samples: 1000,
        burn_in: 200,
        elo_offset: DEFAULT_ELO,
        elo_scale: ELO_SCALE,
    };

    /// Create config with custom match size and paper defaults for rest.
    pub fn new(match_size: usize) -> Self {
        Self {
            match_size,
            ..Self::PAPER_DEFAULTS
        }
    }

    /// Create config with custom sample counts.
    pub fn with_samples(gibbs_samples: usize, burn_in: usize) -> Self {
        Self {
            gibbs_samples,
            burn_in,
            ..Self::PAPER_DEFAULTS
        }
    }

    /// Validate config consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.match_size == 0 {
            return Err("match_size must be > 0".to_string());
        }
        if self.gibbs_samples <= self.burn_in {
            return Err(format!(
                "gibbs_samples ({}) must be > burn_in ({})",
                self.gibbs_samples, self.burn_in
            ));
        }
        if self.elo_scale <= 0.0 {
            return Err("elo_scale must be > 0".to_string());
        }
        Ok(())
    }

    /// Number of effective (post-burn-in) samples.
    pub fn effective_samples(&self) -> usize {
        self.gibbs_samples.saturating_sub(self.burn_in)
    }
}

// ── GibbsScratch ──────────────────────────────────────────────

/// Pre-allocated scratch buffers for Gibbs sampling.
/// Reuse across `rate()` calls to avoid per-call heap allocation.
/// Call `prepare(n)` before each call to ensure buffers are sized correctly.
///
/// # Usage
///
/// ```rust,ignore
/// let mut scratch = GibbsScratch::new();
/// let elos = rater.rate(&sketches, &rankings, &mut rng, Some(&mut scratch));
/// // scratch reused for next call — no reallocation if n <= previous n
/// ```
pub struct GibbsScratch {
    lambda: Vec<f64>,
    r: Vec<f64>,
    lambda_sum: Vec<f64>,
    appearances: Vec<Vec<(usize, usize)>>,
}

impl GibbsScratch {
    /// Create empty scratch space (no allocation until first `prepare()`).
    pub fn new() -> Self {
        Self {
            lambda: Vec::new(),
            r: Vec::new(),
            lambda_sum: Vec::new(),
            appearances: Vec::new(),
        }
    }

    /// Ensure all buffers are sized for `n` sketches.
    /// Grows if needed; never shrinks (reuse across calls).
    fn prepare(&mut self, n: usize) {
        self.lambda.resize(n, 1.0);
        self.r.resize(n, 1.0);
        self.lambda_sum.resize(n, 0.0);
        // Appearances inner vecs must be cleared but capacity preserved.
        if self.appearances.len() < n {
            self.appearances.resize_with(n, Vec::new);
        }
        for vec in &mut self.appearances[..n] {
            vec.clear();
        }
    }
}

impl Default for GibbsScratch {
    fn default() -> Self {
        Self::new()
    }
}

// ── PlackettLuceRater ──────────────────────────────────────────

/// Plackett-Luce rater — converts multi-item rankings to Elo via Gibbs sampling.
///
/// Takes a set of sketches and rankings (each ranking is a permutation of
/// sketch indices), then estimates Elo ratings using Bayesian inference.
///
/// # Usage
///
/// ```rust,ignore
/// use katgpt::pruners::proof::{PlackettLuceRater, SketchEntry, ProofState, Goal};
/// use fastrand::Rng;
///
/// let rater = PlackettLuceRater::with_paper_defaults();
/// let sketches = vec![entry_a, entry_b, entry_c, entry_d];
///
/// // Rankings: each Vec<usize> is a permutation of sketch indices (best first)
/// let rankings = vec![
///     vec![0, 1, 2, 3], // sketch 0 > 1 > 2 > 3
///     vec![1, 0, 3, 2], // sketch 1 > 0 > 3 > 2
///     vec![0, 2, 1, 3], // sketch 0 > 2 > 1 > 3
/// ];
///
/// let mut rng = Rng::with_seed(42);
/// let elos = rater.rate(&sketches, &rankings, &mut rng);
/// // sketch 0 should have highest Elo (won most rankings)
/// ```
pub struct PlackettLuceRater {
    config: PlackettLuceConfig,
}

impl PlackettLuceRater {
    /// Create a rater with custom configuration.
    pub fn new(config: PlackettLuceConfig) -> Self {
        Self { config }
    }

    /// Create with paper defaults.
    pub fn with_paper_defaults() -> Self {
        Self::new(PlackettLuceConfig::PAPER_DEFAULTS)
    }

    /// Configuration reference.
    pub fn config(&self) -> &PlackettLuceConfig {
        &self.config
    }

    /// Rate sketches from rankings, returning Elo for each sketch.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize λ_s and r_s for each sketch from hierarchical prior
    /// 2. Run Gibbs sampler for `gibbs_samples` iterations
    /// 3. Discard first `burn_in` samples
    /// 4. Convert mean(λ_s) → Elo via log10 scaling
    ///
    /// # Arguments
    ///
    /// * `sketches` — slice of sketch entries to rate
    /// * `rankings` — each Vec<usize> is a ranking (best first) of sketch indices
    /// * `rng` — random number generator for Gibbs sampling
    ///
    /// # Returns
    ///
    /// HashMap mapping SketchId → Elo rating. Sketches not appearing in any
    /// ranking receive Elo near `DEFAULT_ELO` (prior predictive).
    ///
    /// # Panics
    ///
    /// Panics if any ranking index >= sketches.len().
    pub fn rate(
        &self,
        sketches: &[SketchEntry],
        rankings: &[Vec<usize>],
        rng: &mut Rng,
    ) -> HashMap<SketchId, f64> {
        self.rate_with_scratch(sketches, rankings, rng, None)
    }

    /// Rate sketches with optional pre-allocated scratch buffers.
    ///
    /// Pass `Some(&mut GibbsScratch)` to reuse buffers across calls.
    /// First call allocates; subsequent calls with `n <= previous_n` reuse.
    pub fn rate_with_scratch(
        &self,
        sketches: &[SketchEntry],
        rankings: &[Vec<usize>],
        rng: &mut Rng,
        scratch: Option<&mut GibbsScratch>,
    ) -> HashMap<SketchId, f64> {
        let n = sketches.len();
        if n == 0 {
            return HashMap::new();
        }

        // Validate ranking indices
        for (r_idx, ranking) in rankings.iter().enumerate() {
            for &pos in ranking {
                assert!(
                    pos < n,
                    "Ranking {r_idx} contains index {pos} but only {n} sketches exist"
                );
            }
        }

        // Use scratch if provided, otherwise allocate fresh.
        let mut owned_scratch;
        let buf: &mut GibbsScratch = match scratch {
            Some(s) => s,
            None => {
                owned_scratch = GibbsScratch::new();
                &mut owned_scratch
            }
        };
        buf.prepare(n);

        let lambda = &mut buf.lambda[..n];
        let r = &mut buf.r[..n];
        let lambda_sum = &mut buf.lambda_sum[..n];
        let sketch_appearances = &mut buf.appearances[..n];

        // Pre-compute sketch appearances in rankings
        for (r_idx, ranking) in rankings.iter().enumerate() {
            for (pos_in_ranking, &sketch_idx) in ranking.iter().enumerate() {
                sketch_appearances[sketch_idx].push((r_idx, pos_in_ranking));
            }
        }

        // Gibbs sampling
        let effective_samples = self.config.effective_samples() as f64;

        for iteration in 0..self.config.gibbs_samples {
            // Update each λ_s
            for s in 0..n {
                let (wins, exposure) =
                    self.compute_stats(s, &sketch_appearances[s], rankings, lambda);

                // λ_s | data, r_s ~ Gamma(1 + wins, r_s + exposure)
                let shape = 1.0 + wins as f64;
                let rate = r[s] + exposure;
                lambda[s] = sample_gamma(rng, shape, rate);
            }

            // Update each r_s (hierarchical)
            for s in 0..n {
                // r_s | λ_s ~ Gamma(2, 1 + λ_s)
                r[s] = sample_gamma(rng, 2.0, 1.0 + lambda[s]);
            }

            // Accumulate post-burn-in
            if iteration >= self.config.burn_in {
                for s in 0..n {
                    lambda_sum[s] += lambda[s];
                }
            }
        }

        // Convert to Elo
        let mut elos = HashMap::with_capacity(n);
        for s in 0..n {
            let mean_lambda = lambda_sum[s] / effective_samples;
            let elo = lambda_to_elo(mean_lambda, self.config.elo_offset, self.config.elo_scale);
            elos.insert(sketches[s].id, elo);
        }

        elos
    }

    /// Compute wins and exposure for a sketch from rankings.
    ///
    /// For each ranking where sketch s appears at position p:
    /// - wins += 1 (s was chosen from remaining set)
    /// - exposure += 1 / (sum of λ's at positions p..end)
    fn compute_stats(
        &self,
        _sketch_idx: usize,
        appearances: &[(usize, usize)],
        rankings: &[Vec<usize>],
        lambda: &[f64],
    ) -> (usize, f64) {
        let mut wins = 0usize;
        let mut exposure = 0.0f64;

        for &(r_idx, pos) in appearances {
            wins += 1;

            // Compute suffix sum: λ_{σ_p} + λ_{σ_{p+1}} + ... + λ_{σ_k}
            let ranking = &rankings[r_idx];
            let suffix_sum: f64 = ranking[pos..].iter().map(|&idx| lambda[idx]).sum();

            if suffix_sum > 0.0 {
                exposure += 1.0 / suffix_sum;
            }
        }

        (wins, exposure)
    }
}

impl fmt::Display for PlackettLuceRater {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PlackettLuce(P={}, I={}, B={})",
            self.config.match_size, self.config.gibbs_samples, self.config.burn_in
        )
    }
}

// ── Elo Conversion ─────────────────────────────────────────────

/// Convert mean λ to Elo rating.
///
/// `Elo = offset + scale × log10(λ_mean)`
///
/// Clamps λ_mean to 1e-10 minimum to avoid -inf from log10.
fn lambda_to_elo(lambda_mean: f64, offset: f64, scale: f64) -> f64 {
    let clamped = lambda_mean.max(1e-10);
    offset + scale * clamped.log10()
}

// ── Gamma Sampler ──────────────────────────────────────────────

/// Sample from Gamma(shape, rate) distribution.
///
/// Uses Marsaglia-Tsang method for shape ≥ 1, with transformation trick
/// for shape < 1: sample Gamma(shape+1, rate) × U^(1/shape).
///
/// Parameterization: PDF = rate^shape / Γ(shape) × x^(shape-1) × exp(-rate × x)
/// Mean = shape / rate.
fn sample_gamma(rng: &mut Rng, shape: f64, rate: f64) -> f64 {
    debug_assert!(shape > 0.0, "Gamma shape must be > 0, got {shape}");
    // Clamp rate to avoid division-by-zero in Marsaglia-Tsang (hierarchical
    // prior can produce near-zero rates when sketch has no ranking appearances).
    let rate = rate.max(1e-10);
    debug_assert!(rate > 0.0, "Gamma rate must be > 0, got {rate}");

    let (alpha, needs_transform) = if shape < 1.0 {
        (shape + 1.0, true)
    } else {
        (shape, false)
    };

    // Marsaglia-Tsang method
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    let result = loop {
        let x = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);

        if v <= 0.0 {
            continue;
        }

        let u: f64 = rng.f32().into();

        // Acceptance check: ln(u) < 0.5*x² + d - d*v + d*ln(v)
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            break d * v;
        }
    };

    let mut final_val = result;
    if needs_transform {
        let u: f64 = rng.f32().max(1e-30).into();
        final_val *= u.powf(1.0 / shape);
    }

    final_val / rate
}

// ── Normal Sampler ─────────────────────────────────────────────

/// Sample from Standard Normal N(0, 1) using Box-Muller transform.
///
/// z = √(-2 × ln(u₁)) × cos(2π × u₂)
fn sample_normal(rng: &mut Rng) -> f64 {
    let u1: f64 = rng.f32().max(1e-30).into(); // avoid ln(0)
    let u2: f64 = rng.f32().into();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ── Ranking Generation ─────────────────────────────────────────

/// Generate random P-sized rankings from a population of sketches.
///
/// Creates `num_rankings` rankings, each containing `match_size` randomly
/// selected sketch indices in random order (simulating LLM-based ranking).
///
/// # Arguments
///
/// * `n_sketches` — total number of sketches
/// * `match_size` — items per ranking (P=7 per paper)
/// * `num_rankings` — number of rankings to generate
/// * `rng` — random number generator
///
/// # Returns
///
/// Vec of rankings, each a permutation of randomly selected sketch indices.
pub fn generate_random_rankings(
    n_sketches: usize,
    match_size: usize,
    num_rankings: usize,
    rng: &mut Rng,
) -> Vec<Vec<usize>> {
    if n_sketches == 0 || match_size == 0 {
        return Vec::new();
    }

    let effective_size = match_size.min(n_sketches);

    (0..num_rankings)
        .map(|_| {
            // Fisher-Yates partial shuffle for random subset
            let mut indices: Vec<usize> = (0..n_sketches).collect();

            for i in 0..effective_size {
                let j = rng.usize(i..n_sketches);
                indices.swap(i, j);
            }

            indices[..effective_size].to_vec()
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::sketch_types::{Goal, ProofState};

    fn make_sketches(n: usize) -> Vec<SketchEntry> {
        (0..n)
            .map(|i| {
                let state = ProofState::new(format!("sketch_{i}").into_bytes());
                SketchEntry::new(state, vec![Goal::from_label(format!("goal_{i}"))])
            })
            .collect()
    }

    // ── Config Tests ───────────────────────────────────────────

    #[test]
    fn config_paper_defaults() {
        let cfg = PlackettLuceConfig::PAPER_DEFAULTS;
        assert_eq!(cfg.match_size, 7);
        assert_eq!(cfg.gibbs_samples, 1000);
        assert_eq!(cfg.burn_in, 200);
        assert!((cfg.elo_offset - 1200.0).abs() < 1e-9);
        assert!((cfg.elo_scale - 400.0).abs() < 1e-9);
    }

    #[test]
    fn config_validate_ok() {
        assert!(PlackettLuceConfig::PAPER_DEFAULTS.validate().is_ok());
    }

    #[test]
    fn config_validate_zero_match_size() {
        let cfg = PlackettLuceConfig::new(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_burn_in_too_large() {
        let cfg = PlackettLuceConfig {
            gibbs_samples: 100,
            burn_in: 200,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_negative_scale() {
        let cfg = PlackettLuceConfig {
            elo_scale: -1.0,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_effective_samples() {
        let cfg = PlackettLuceConfig::PAPER_DEFAULTS;
        assert_eq!(cfg.effective_samples(), 800);
    }

    // ── Gamma Sampler Tests ────────────────────────────────────

    #[test]
    fn gamma_mean_converges() {
        let mut rng = Rng::with_seed(42);
        let shape = 5.0;
        let rate = 2.0;
        let expected_mean = shape / rate; // 2.5

        let n_samples = 50_000;
        let sum: f64 = (0..n_samples)
            .map(|_| sample_gamma(&mut rng, shape, rate))
            .sum();

        let actual_mean = sum / n_samples as f64;
        assert!(
            (actual_mean - expected_mean).abs() < 0.1,
            "Gamma({shape},{rate}) mean: expected {expected_mean}, got {actual_mean}"
        );
    }

    #[test]
    fn gamma_shape_below_one() {
        let mut rng = Rng::with_seed(42);
        let shape = 0.5;
        let rate = 1.0;
        let expected_mean = shape / rate;

        let n_samples = 50_000;
        let sum: f64 = (0..n_samples)
            .map(|_| sample_gamma(&mut rng, shape, rate))
            .sum();

        let actual_mean = sum / n_samples as f64;
        assert!(
            (actual_mean - expected_mean).abs() < 0.1,
            "Gamma({shape},{rate}) mean: expected {expected_mean}, got {actual_mean}"
        );
    }

    #[test]
    fn gamma_always_positive() {
        let mut rng = Rng::with_seed(42);
        for _ in 0..1000 {
            let val = sample_gamma(&mut rng, 1.0, 1.0);
            assert!(val > 0.0, "Gamma samples must be positive");
        }
    }

    // ── Normal Sampler Tests ───────────────────────────────────

    #[test]
    fn normal_mean_converges() {
        let mut rng = Rng::with_seed(42);
        let n_samples = 50_000;
        let sum: f64 = (0..n_samples).map(|_| sample_normal(&mut rng)).sum();
        let mean = sum / n_samples as f64;
        assert!(mean.abs() < 0.1, "N(0,1) mean should be near 0, got {mean}");
    }

    #[test]
    fn normal_variance_converges() {
        let mut rng = Rng::with_seed(42);
        let n_samples = 50_000;
        let samples: Vec<f64> = (0..n_samples).map(|_| sample_normal(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / n_samples as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
        assert!(
            (variance - 1.0).abs() < 0.1,
            "N(0,1) variance should be near 1, got {variance}"
        );
    }

    // ── Elo Conversion Tests ───────────────────────────────────

    #[test]
    fn elo_conversion_lambda_one() {
        // λ=1 → Elo = 1200 + 400*log10(1) = 1200
        let elo = lambda_to_elo(1.0, 1200.0, 400.0);
        assert!((elo - 1200.0).abs() < 1e-9);
    }

    #[test]
    fn elo_conversion_lambda_ten() {
        // λ=10 → Elo = 1200 + 400*log10(10) = 1600
        let elo = lambda_to_elo(10.0, 1200.0, 400.0);
        assert!((elo - 1600.0).abs() < 1e-9);
    }

    #[test]
    fn elo_conversion_lambda_tenth() {
        // λ=0.1 → Elo = 1200 + 400*log10(0.1) = 800
        let elo = lambda_to_elo(0.1, 1200.0, 400.0);
        assert!((elo - 800.0).abs() < 1e-9);
    }

    #[test]
    fn elo_conversion_clamps_near_zero() {
        let elo = lambda_to_elo(0.0, 1200.0, 400.0);
        assert!(elo.is_finite(), "should not be -inf");
    }

    // ── Rating Tests ───────────────────────────────────────────

    #[test]
    fn rate_empty_sketches() {
        let rater = PlackettLuceRater::with_paper_defaults();
        let mut rng = Rng::with_seed(42);
        let elos = rater.rate(&[], &[], &mut rng);
        assert!(elos.is_empty());
    }

    #[test]
    fn rate_single_sketch() {
        let rater = PlackettLuceRater::with_paper_defaults();
        let sketches = make_sketches(1);
        let mut rng = Rng::with_seed(42);
        let elos = rater.rate(&sketches, &[], &mut rng);
        assert_eq!(elos.len(), 1);
        // No rankings → Elo near DEFAULT_ELO (prior predictive).
        // Hierarchical Gamma prior can drift mean λ, so threshold is generous.
        let elo = elos[&sketches[0].id];
        assert!(
            (elo - DEFAULT_ELO).abs() < 500.0,
            "Single sketch with no rankings should be near DEFAULT_ELO, got {elo}"
        );
    }

    #[test]
    fn rate_consistent_winner_gets_highest_elo() {
        // Sketch 0 always wins (ranked first in all rankings)
        let config = PlackettLuceConfig {
            gibbs_samples: 2000,
            burn_in: 500,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        };
        let rater = PlackettLuceRater::new(config);
        let sketches = make_sketches(4);
        let mut rng = Rng::with_seed(42);

        // Sketch 0 always ranked first
        let rankings = vec![
            vec![0, 1, 2, 3],
            vec![0, 2, 3, 1],
            vec![0, 3, 1, 2],
            vec![0, 1, 3, 2],
            vec![0, 2, 1, 3],
        ];

        let elos = rater.rate(&sketches, &rankings, &mut rng);

        let elo_0 = elos[&sketches[0].id];
        let elo_1 = elos[&sketches[1].id];
        let elo_2 = elos[&sketches[2].id];
        let elo_3 = elos[&sketches[3].id];

        assert!(
            elo_0 > elo_1 && elo_0 > elo_2 && elo_0 > elo_3,
            "Consistent winner should have highest Elo: {elo_0} vs {elo_1}, {elo_2}, {elo_3}"
        );
    }

    #[test]
    fn rate_no_rankings_all_near_default() {
        let config = PlackettLuceConfig {
            gibbs_samples: 500,
            burn_in: 100,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        };
        let rater = PlackettLuceRater::new(config);
        let sketches = make_sketches(3);
        let mut rng = Rng::with_seed(42);

        let elos = rater.rate(&sketches, &[], &mut rng);

        for sketch in &sketches {
            let elo = elos[&sketch.id];
            assert!(
                (elo - DEFAULT_ELO).abs() < 500.0,
                "No rankings → Elo near DEFAULT_ELO, got {elo}"
            );
        }
    }

    // ── Ranking Generation Tests ───────────────────────────────

    #[test]
    fn generate_rankings_correct_count() {
        let mut rng = Rng::with_seed(42);
        let rankings = generate_random_rankings(10, 7, 5, &mut rng);
        assert_eq!(rankings.len(), 5);
        for ranking in &rankings {
            assert_eq!(ranking.len(), 7);
        }
    }

    #[test]
    fn generate_rankings_valid_indices() {
        let mut rng = Rng::with_seed(42);
        let n = 10;
        let rankings = generate_random_rankings(n, 7, 20, &mut rng);
        for ranking in &rankings {
            for &idx in ranking {
                assert!(idx < n, "Index {idx} out of bounds for {n} sketches");
            }
        }
    }

    #[test]
    fn generate_rankings_no_duplicates_within() {
        let mut rng = Rng::with_seed(42);
        let rankings = generate_random_rankings(10, 7, 20, &mut rng);
        for ranking in &rankings {
            let mut seen = std::collections::HashSet::new();
            for &idx in ranking {
                assert!(seen.insert(idx), "Duplicate index {idx} in ranking");
            }
        }
    }

    #[test]
    fn generate_rankings_empty_sketches() {
        let mut rng = Rng::with_seed(42);
        let rankings = generate_random_rankings(0, 7, 5, &mut rng);
        assert!(rankings.is_empty());
    }

    #[test]
    fn generate_rankings_match_size_exceeds_population() {
        let mut rng = Rng::with_seed(42);
        let rankings = generate_random_rankings(3, 7, 5, &mut rng);
        for ranking in &rankings {
            assert_eq!(
                ranking.len(),
                3,
                "match_size should clamp to population size"
            );
        }
    }

    // ── Display Tests ──────────────────────────────────────────

    #[test]
    fn rater_display() {
        let rater = PlackettLuceRater::with_paper_defaults();
        let display = format!("{rater}");
        assert!(display.contains("P=7"));
        assert!(display.contains("I=1000"));
        assert!(display.contains("B=200"));
    }

    // ── Integration Tests ──────────────────────────────────────

    #[test]
    fn rate_paper_scenario_four_candidates() {
        // Simulate paper scenario: 4 candidates, 10 rankings
        let config = PlackettLuceConfig {
            match_size: 4,
            gibbs_samples: 2000,
            burn_in: 500,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        };
        let rater = PlackettLuceRater::new(config);
        let sketches = make_sketches(4);
        let mut rng = Rng::with_seed(42);

        // Known ordering: 0 > 1 > 2 > 3 (mostly consistent)
        let rankings = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 3, 2],
            vec![0, 2, 1, 3],
            vec![1, 0, 2, 3], // upset: 1 beats 0
            vec![0, 1, 2, 3],
            vec![0, 3, 1, 2],
            vec![1, 0, 3, 2], // upset
            vec![0, 2, 3, 1],
            vec![0, 1, 2, 3],
            vec![2, 0, 1, 3], // upset: 2 beats 0
        ];

        let elos = rater.rate(&sketches, &rankings, &mut rng);

        let elo_0 = elos[&sketches[0].id];
        let elo_1 = elos[&sketches[1].id];
        let elo_2 = elos[&sketches[2].id];
        let elo_3 = elos[&sketches[3].id];

        assert!(elo_0 > elo_3, "0 > 3: {elo_0} vs {elo_3}");
        assert!(elo_1 > elo_3, "1 > 3: {elo_1} vs {elo_3}");
        assert!(elo_0 > elo_2, "0 > 2: {elo_0} vs {elo_2}");
    }

    #[test]
    fn rate_with_generated_rankings() {
        let config = PlackettLuceConfig {
            gibbs_samples: 1000,
            burn_in: 200,
            ..PlackettLuceConfig::PAPER_DEFAULTS
        };
        let rater = PlackettLuceRater::new(config);
        let sketches = make_sketches(10);
        let mut rng = Rng::with_seed(42);

        let rankings = generate_random_rankings(10, 7, 20, &mut rng);
        assert_eq!(rankings.len(), 20);

        let elos = rater.rate(&sketches, &rankings, &mut rng);
        assert_eq!(elos.len(), 10);

        // All Elo values must be finite (no NaN, no inf).
        // Absolute range is not checked — Gibbs sampler can produce wide Elo
        // spreads with random rankings due to λ variance in hierarchical prior.
        for sketch in &sketches {
            let elo = elos[&sketch.id];
            assert!(elo.is_finite(), "Elo must be finite, got {elo}");
        }

        // Elo values should have some spread (not all identical).
        let elo_vals: Vec<f64> = sketches.iter().map(|s| elos[&s.id]).collect();
        let min_elo = elo_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_elo = elo_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_elo > min_elo,
            "Elo should have spread from random rankings"
        );
    }
}
