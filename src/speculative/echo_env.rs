//! ECHO Environment Predictor — inference-time prediction scoring (Plan 247).
//!
//! Distills arXiv:2605.24517 insight: policies that better predict environment
//! dynamics also better navigate those dynamics. Three modelless primitives
//! wire into existing DDTree + BanditPruner + ScreeningPruner pipeline.
//!
//! # Primitives
//!
//! - **`EnvPredictorPruner`** — `ScreeningPruner` that scores candidate actions by
//!   how "expected" their predicted outcomes are versus historical averages.
//!   Uses sigmoid(dot-product) — never softmax — per project rules.
//!
//! - **`PredictionVerifier`** — post-hoc verification that compares predicted
//!   features against actual outcomes, producing a bandit reward signal based
//!   on EMA-tracked prediction accuracy.
//!
//! - **`PredictionConsistencyGate`** — entropy-based confidence gate that
//!   adjusts budget allocation: low inter-branch entropy → contract budget,
//!   high entropy → expand budget for exploration.
//!
//! Feature-gated behind `echo_env_predictor` — off by default until GOAT proof.

use katgpt_core::traits::ScreeningPruner;

// ── Data types ──────────────────────────────────────────────────

/// Predicted outcome from running the game's forward model.
/// Zero-allocation: fixed-size feature vector.
#[derive(Debug, Clone)]
pub struct PredictedOutcome {
    /// State features after applying action (from game forward model).
    pub features: Vec<f32>,
    /// Confidence score [0, 1] — how "expected" this outcome is.
    pub confidence: f32,
    /// Shannon entropy of the feature distribution.
    pub entropy: f32,
}

/// Record of a prediction vs actual outcome for bandit reward.
#[derive(Debug, Clone)]
pub struct PredictionRecord {
    /// Cosine similarity between predicted and actual features.
    pub accuracy: f32,
    /// Whether the prediction was within the confidence band.
    pub correct: bool,
    /// Timestamp for EMA tracking.
    pub tick: u64,
}

/// Configuration for ECHO environment predictor.
#[derive(Debug, Clone)]
pub struct EnvPredictorConfig {
    /// Sigmoid temperature for confidence scoring. Default: 1.0.
    pub temperature: f32,
    /// Minimum accuracy to count as "correct" prediction. Default: 0.7.
    pub accuracy_threshold: f32,
    /// EMA decay for running accuracy tracking. Default: 0.95.
    pub ema_decay: f32,
    /// Entropy threshold for consistency gate activation. Default: 2.0.
    pub consistency_entropy_threshold: f32,
    /// Budget expansion factor when consistency is low. Default: 1.5.
    pub budget_expand_factor: f32,
    /// Budget contraction factor when consistency is high. Default: 0.8.
    pub budget_contract_factor: f32,
}

impl Default for EnvPredictorConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            accuracy_threshold: 0.7,
            ema_decay: 0.95,
            consistency_entropy_threshold: 2.0,
            budget_expand_factor: 1.5,
            budget_contract_factor: 0.8,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
    (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

// ── A) EnvPredictorPruner ───────────────────────────────────────

/// ScreeningPruner that scores actions by predicted outcome quality.
///
/// Uses a deterministic forward model (provided as a closure) to predict
/// the next state from (current_state, action), then scores how "expected"
/// the outcome is versus historical averages via sigmoid(dot product).
pub struct EnvPredictorPruner<F>
where
    F: Fn(usize, &[usize]) -> Vec<f32> + Send + Sync,
{
    /// Forward model: (action_token, parent_tokens) → predicted state features.
    pub forward_model: F,
    /// Historical average features (running mean).
    pub feature_avg: Vec<f32>,
    /// Configuration.
    pub config: EnvPredictorConfig,
    /// Number of observations for running average.
    n_observations: usize,
}

impl<F> EnvPredictorPruner<F>
where
    F: Fn(usize, &[usize]) -> Vec<f32> + Send + Sync,
{
    pub fn new(forward_model: F, feature_dim: usize, config: EnvPredictorConfig) -> Self {
        Self {
            forward_model,
            feature_avg: vec![0.0; feature_dim],
            config,
            n_observations: 0,
        }
    }

    /// Update historical average with new observation features.
    pub fn update_avg(&mut self, features: &[f32]) {
        let n = self.n_observations as f32;
        let alpha = 1.0 / (n + 1.0);
        for (avg, &f) in self.feature_avg.iter_mut().zip(features.iter()) {
            *avg = *avg * (1.0 - alpha) + f * alpha;
        }
        self.n_observations += 1;
    }
}

impl<F> ScreeningPruner for EnvPredictorPruner<F>
where
    F: Fn(usize, &[usize]) -> Vec<f32> + Send + Sync,
{
    fn relevance(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.n_observations == 0 {
            return 0.5; // No history yet — neutral score
        }

        // Run forward model to predict outcome features
        let predicted = (self.forward_model)(token_idx, parent_tokens);

        // Score = sigmoid(cosine_similarity(predicted, historical_avg) / temperature)
        let dot: f32 = predicted
            .iter()
            .zip(self.feature_avg.iter())
            .map(|(&p, &a)| p * a)
            .sum();

        let norm_p: f32 = predicted
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        let norm_a: f32 = self
            .feature_avg
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);

        let cosine = dot / (norm_p * norm_a);
        sigmoid(cosine / self.config.temperature)
    }
}

// ── B) PredictionVerifier ───────────────────────────────────────

/// Verifies predictions against actual outcomes.
/// Produces a bandit reward signal based on prediction accuracy.
pub struct PredictionVerifier {
    /// Configuration.
    pub config: EnvPredictorConfig,
    /// Running EMA of prediction accuracy.
    pub accuracy_ema: f32,
    /// Total predictions verified.
    pub total_verified: usize,
    /// Total correct predictions (above threshold).
    pub total_correct: usize,
}

impl PredictionVerifier {
    pub fn new(config: EnvPredictorConfig) -> Self {
        Self {
            config,
            accuracy_ema: 0.5,
            total_verified: 0,
            total_correct: 0,
        }
    }

    /// Compare predicted features against actual features.
    /// Returns a PredictionRecord with accuracy score.
    pub fn verify(&mut self, predicted: &[f32], actual: &[f32], tick: u64) -> PredictionRecord {
        let accuracy = cosine_similarity(predicted, actual);
        let correct = accuracy >= self.config.accuracy_threshold;

        // Update EMA
        let alpha = 1.0 - self.config.ema_decay;
        self.accuracy_ema = self.config.ema_decay * self.accuracy_ema + alpha * accuracy;

        self.total_verified += 1;
        if correct {
            self.total_correct += 1;
        }

        PredictionRecord {
            accuracy,
            correct,
            tick,
        }
    }

    /// Returns the bandit reward based on current prediction accuracy.
    /// Higher accuracy → higher reward → promotion via AbsorbCompress.
    pub fn bandit_reward(&self) -> f32 {
        self.accuracy_ema
    }

    /// Returns the fraction of correct predictions.
    pub fn correct_rate(&self) -> f32 {
        if self.total_verified == 0 {
            0.5
        } else {
            self.total_correct as f32 / self.total_verified as f32
        }
    }
}

// ── C) PredictionConsistencyGate ────────────────────────────────

/// Uses entropy across DDTree branch predictions to gate budget allocation.
///
/// Low inter-branch entropy → high confidence → contract budget.
/// High inter-branch entropy → low confidence → expand budget for exploration.
pub struct PredictionConsistencyGate {
    /// Configuration.
    pub config: EnvPredictorConfig,
    /// Running entropy history for trend detection.
    pub entropy_history: Vec<f32>,
}

impl PredictionConsistencyGate {
    pub fn new(config: EnvPredictorConfig) -> Self {
        Self {
            config,
            entropy_history: Vec::with_capacity(64),
        }
    }

    /// Compute Shannon entropy from a set of prediction feature vectors.
    /// Each row is a branch's predicted features. We compute per-feature
    /// variance across branches, then sum log-variances as entropy proxy.
    pub fn compute_branch_entropy(branch_features: &[Vec<f32>]) -> f32 {
        if branch_features.len() <= 1 {
            return 0.0; // Single branch = zero entropy
        }

        let n_features = branch_features[0].len();
        let n_branches = branch_features.len() as f32;

        let mut total_entropy = 0.0f32;
        for j in 0..n_features {
            // Compute mean
            let mean: f32 = branch_features.iter().map(|b| b[j]).sum::<f32>() / n_branches;
            // Compute variance
            let var: f32 = branch_features
                .iter()
                .map(|b| (b[j] - mean) * (b[j] - mean))
                .sum::<f32>()
                / n_branches;
            // Entropy proxy: log(1 + variance) — higher variance = higher entropy
            total_entropy += (1.0 + var).ln();
        }

        total_entropy
    }

    /// Get budget multiplier based on current entropy.
    /// High entropy → expand budget. Low entropy → contract.
    pub fn budget_multiplier(&mut self, entropy: f32) -> f32 {
        self.entropy_history.push(entropy);

        if entropy > self.config.consistency_entropy_threshold {
            // High entropy (inconsistent predictions) → expand
            self.config.budget_expand_factor
        } else {
            // Low entropy (consistent predictions) → contract
            self.config.budget_contract_factor
        }
    }

    /// Returns average entropy over last N observations.
    pub fn avg_entropy(&self, last_n: usize) -> f32 {
        let start = self.entropy_history.len().saturating_sub(last_n);
        let slice = &self.entropy_history[start..];
        if slice.is_empty() {
            0.0
        } else {
            slice.iter().sum::<f32>() / slice.len() as f32
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_default_config() {
        let config = EnvPredictorConfig::default();
        assert!((config.temperature - 1.0).abs() < 1e-6);
        assert!((config.accuracy_threshold - 0.7).abs() < 1e-6);
        assert!((config.ema_decay - 0.95).abs() < 1e-6);
        assert!((config.consistency_entropy_threshold - 2.0).abs() < 1e-6);
        assert!((config.budget_expand_factor - 1.5).abs() < 1e-6);
        assert!((config.budget_contract_factor - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_predictor_relevance_no_history() {
        let forward_model = |_: usize, _: &[usize]| vec![1.0_f32, 0.0, 0.5];
        let pruner = EnvPredictorPruner::new(forward_model, 3, EnvPredictorConfig::default());

        // No history → neutral 0.5
        let score = pruner.relevance(0, 0, &[]);
        assert!((score - 0.5).abs() < 1e-6, "expected 0.5, got {score}");
    }

    #[test]
    fn test_predictor_relevance_with_history() {
        let forward_model = |_: usize, _: &[usize]| vec![1.0_f32, 0.0, 0.5];
        let mut pruner = EnvPredictorPruner::new(forward_model, 3, EnvPredictorConfig::default());

        // Seed history with same direction as predictions
        pruner.update_avg(&[1.0, 0.0, 0.5]);
        pruner.update_avg(&[1.0, 0.0, 0.5]);

        let score = pruner.relevance(0, 0, &[]);
        // Predicted == avg → cosine ~1.0 → sigmoid(1.0/1.0) ≈ 0.731
        assert!(
            score > 0.5,
            "similar predictions should score > 0.5, got {score}"
        );

        // Now seed history with orthogonal direction
        let forward_model_2 = |_: usize, _: &[usize]| vec![0.0_f32, 1.0, 0.0];
        let mut pruner2 =
            EnvPredictorPruner::new(forward_model_2, 3, EnvPredictorConfig::default());
        pruner2.update_avg(&[1.0, 0.0, 0.0]);

        let score2 = pruner2.relevance(0, 0, &[]);
        // Orthogonal → cosine ~0.0 → sigmoid(0.0) = 0.5
        assert!(
            score2 < score,
            "orthogonal predictions should score lower than aligned, got {score2} vs {score}"
        );
    }

    #[test]
    fn test_verifier_accuracy() {
        let mut verifier = PredictionVerifier::new(EnvPredictorConfig::default());

        // Identical vectors → accuracy = 1.0
        let record = verifier.verify(&[1.0, 0.0, 0.5], &[1.0, 0.0, 0.5], 0);
        assert!((record.accuracy - 1.0).abs() < 1e-6);
        assert!(record.correct);

        // Orthogonal vectors → accuracy ~0.0
        let record2 = verifier.verify(&[1.0, 0.0], &[0.0, 1.0], 1);
        assert!(
            record2.accuracy < 0.01,
            "orthogonal should be ~0, got {}",
            record2.accuracy
        );
        assert!(!record2.correct);
    }

    #[test]
    fn test_verifier_ema() {
        let mut verifier = PredictionVerifier::new(EnvPredictorConfig::default());

        // First verify: identical → accuracy 1.0
        verifier.verify(&[1.0, 0.0], &[1.0, 0.0], 0);
        let ema_after_1 = verifier.accuracy_ema;
        // EMA = 0.95 * 0.5 + 0.05 * 1.0 = 0.525
        assert!((ema_after_1 - 0.525).abs() < 1e-6, "got {ema_after_1}");

        // Second verify: orthogonal → accuracy 0.0
        verifier.verify(&[1.0, 0.0], &[0.0, 1.0], 1);
        let ema_after_2 = verifier.accuracy_ema;
        // EMA should have decreased
        assert!(
            ema_after_2 < ema_after_1,
            "EMA should decrease after bad prediction"
        );
    }

    #[test]
    fn test_verifier_bandit_reward() {
        let mut verifier = PredictionVerifier::new(EnvPredictorConfig::default());

        // Initially 0.5
        assert!((verifier.bandit_reward() - 0.5).abs() < 1e-6);

        // After many correct predictions, reward should increase
        for i in 0..20 {
            verifier.verify(&[1.0, 0.0, 0.5], &[1.0, 0.0, 0.5], i);
        }
        assert!(
            verifier.bandit_reward() > 0.5,
            "reward should increase after correct predictions, got {}",
            verifier.bandit_reward()
        );
    }

    #[test]
    fn test_consistency_entropy_single_branch() {
        let features = vec![vec![1.0, 2.0, 3.0]];
        let entropy = PredictionConsistencyGate::compute_branch_entropy(&features);
        assert!(
            (entropy - 0.0).abs() < 1e-6,
            "single branch should have zero entropy"
        );
    }

    #[test]
    fn test_consistency_entropy_multiple_branches() {
        // Identical branches → zero variance → zero entropy
        let identical = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        let entropy_identical = PredictionConsistencyGate::compute_branch_entropy(&identical);
        assert!(
            entropy_identical.abs() < 1e-6,
            "identical branches should have ~0 entropy, got {entropy_identical}"
        );

        // Divergent branches → positive variance → positive entropy
        let divergent = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let entropy_divergent = PredictionConsistencyGate::compute_branch_entropy(&divergent);
        assert!(
            entropy_divergent > entropy_identical,
            "divergent branches should have higher entropy than identical"
        );
    }

    #[test]
    fn test_consistency_budget_multiplier() {
        let mut gate = PredictionConsistencyGate::new(EnvPredictorConfig::default());

        // Low entropy → contract
        let m1 = gate.budget_multiplier(0.5);
        assert!(
            (m1 - 0.8).abs() < 1e-6,
            "low entropy should contract, got {m1}"
        );

        // High entropy → expand
        let m2 = gate.budget_multiplier(5.0);
        assert!(
            (m2 - 1.5).abs() < 1e-6,
            "high entropy should expand, got {m2}"
        );

        // Check entropy history recorded
        assert_eq!(gate.entropy_history.len(), 2);
        assert!((gate.entropy_history[0] - 0.5).abs() < 1e-6);
        assert!((gate.entropy_history[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_prediction_record() {
        let record = PredictionRecord {
            accuracy: 0.85,
            correct: true,
            tick: 42,
        };
        assert!((record.accuracy - 0.85).abs() < 1e-6);
        assert!(record.correct);
        assert_eq!(record.tick, 42);
    }
}
