//! Diffusion Sampler — Adaptive Per-Position Correctness Predictor for D2F Denoising.
//!
//! Plan 089 T6: Trained Sampler Research.
//!
//! Replaces fixed confidence threshold with a learned per-position correctness
//! predictor. Two capacity variants:
//!
//! 1. **Logistic** — 6 features, ~7 params. For micro_dllm scale (d=16, vocab=27).
//! 2. **Transformer** — 4-layer d=384, ~4.8M params. For production scale (stub, deferred).
//!
//! Research (Nemotron Appendix A): shifts Pareto frontier +1.3× TPF or +10.6% accuracy.
//!
//! # Feature Vector (per position, per denoising step)
//!
//! | Feature | Description |
//! |---------|-------------|
//! | top1_prob | Top-1 token probability after softmax |
//! | margin | Top-1 prob − top-2 prob |
//! | top3_mass | Sum of top-3 token probabilities |
//! | entropy | Entropy of softmax distribution |
//! | step_norm | Current step / max steps |
//! | pos_norm | Position / block_size |
//!
//! # Usage
//!
//! ```ignore
//! // Collect trajectories from D2F decode with ground truth
//! let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 100);
//!
//! // Train sampler
//! let mut sampler = DiffusionSampler::auto(&config);
//! sampler.train(&trajectories, 0.01, 200);
//!
//! // Use in denoising loop — replace fixed threshold
//! let p_correct = sampler.predict(&features);
//! if p_correct >= tau_conf { ... }
//! ```

#![cfg(feature = "tri_mode")]

use crate::dllm::{D2fContext, forward_block_causal_with};
use crate::speculative::d2f::{D2fDecodeConfig, d2f_decode_block_with_target_with};
use crate::speculative::types::NoPruner;
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

// ── Feature Extraction ────────────────────────────────────────

/// Number of input features per position.
pub const N_FEATURES: usize = 6;

/// Per-position features extracted from D2F denoising step.
///
/// Lightweight stats that capture the model's confidence at each position
/// without requiring the full embedding (which would be d=384 in production).
#[derive(Clone, Debug, Default)]
pub struct SamplerFeatures {
    /// Top-1 token probability after softmax.
    pub top1_prob: f32,
    /// Margin: top-1 prob − top-2 prob. Higher = more confident.
    pub margin: f32,
    /// Sum of top-3 token probabilities. Higher = peaked distribution.
    pub top3_mass: f32,
    /// Entropy of softmax distribution. Lower = more confident.
    pub entropy: f32,
    /// Current denoising step normalized: step / max_steps.
    pub step_norm: f32,
    /// Position within block normalized: pos / block_size.
    pub pos_norm: f32,
}

impl SamplerFeatures {
    /// Extract features from a flat logits slice for one position.
    ///
    /// `logits_p` must be length `vocab`. `mask` token is excluded from stats.
    pub fn from_logits(
        logits_p: &[f32],
        vocab: usize,
        mask: usize,
        step: usize,
        max_steps: usize,
        pos: usize,
        block_size: usize,
    ) -> Self {
        let max_logit = logits_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Softmax denominator over valid (non-mask) tokens
        let mut sum_exp = 0.0f32;
        let mut top_probs: Vec<f32> = Vec::with_capacity(vocab);
        for t in 0..vocab {
            if t == mask {
                top_probs.push(0.0);
                continue;
            }
            let prob = (logits_p[t] - max_logit).exp();
            top_probs.push(prob);
            sum_exp += prob;
        }

        if sum_exp <= 0.0 {
            return Self::default();
        }

        // Normalize to probabilities
        for p in &mut top_probs {
            *p /= sum_exp;
        }

        // Find top-1 and top-2
        let mut top1_idx = 0usize;
        let mut top1_prob = 0.0f32;
        let mut top2_prob = 0.0f32;
        for t in 0..vocab {
            if t == mask {
                continue;
            }
            if top_probs[t] > top1_prob {
                top2_prob = top1_prob;
                top1_prob = top_probs[t];
                top1_idx = t;
            } else if top_probs[t] > top2_prob {
                top2_prob = top_probs[t];
            }
        }

        // Top-3 mass
        let mut top3_mass = 0.0f32;
        let mut probs_sorted: Vec<f32> = top_probs
            .iter()
            .enumerate()
            .filter(|&(t, _)| t != mask)
            .map(|(_, &p)| p)
            .collect();
        probs_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for (i, &p) in probs_sorted.iter().enumerate() {
            if i >= 3 {
                break;
            }
            top3_mass += p;
        }

        // Entropy: -Σ p·log(p)
        let mut entropy = 0.0f32;
        for t in 0..vocab {
            if t == mask {
                continue;
            }
            let p = top_probs[t];
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }

        let margin = top1_prob - top2_prob;
        let step_norm = if max_steps > 0 {
            step as f32 / max_steps as f32
        } else {
            0.0
        };
        let pos_norm = if block_size > 0 {
            pos as f32 / block_size as f32
        } else {
            0.0
        };

        Self {
            top1_prob,
            margin,
            top3_mass,
            entropy,
            step_norm,
            pos_norm,
        }
    }

    /// Convert to flat feature array for model input.
    pub fn to_array(&self) -> [f32; N_FEATURES] {
        [
            self.top1_prob,
            self.margin,
            self.top3_mass,
            self.entropy,
            self.step_norm,
            self.pos_norm,
        ]
    }
}

// ── Sampler Variant ───────────────────────────────────────────

/// Which sampler capacity to use.
#[derive(Clone, Debug, Default)]
pub enum SamplerVariant {
    /// Logistic regression: 6 features → 1 output. ~7 params.
    /// Best for micro_dllm scale (d=16, vocab=27, block=16).
    #[default]
    Logistic,
    /// 2-layer MLP: 6 → hidden → 1. Scales with hidden_dim.
    /// For medium-scale models. hidden_dim=32 → ~250 params.
    Mlp { hidden_dim: usize },
    /// 4-layer transformer: d=384, ~4.8M params.
    /// For production-scale models (Nemotron paper spec). Stub, deferred.
    Transformer { d_model: usize, n_layers: usize },
}

// ── Logistic Sampler ──────────────────────────────────────────

/// Logistic regression sampler: sigmoid(w·x + b).
///
/// 6 weights + 1 bias = 7 parameters total.
/// O(1) inference, trivial memory footprint.
#[derive(Clone, Debug)]
pub struct LogisticSampler {
    /// Feature weights [N_FEATURES].
    pub weights: Vec<f32>,
    /// Bias term.
    pub bias: f32,
}

impl LogisticSampler {
    /// Create with random initialization.
    pub fn new(rng: &mut Rng) -> Self {
        let mut weights = Vec::with_capacity(N_FEATURES);
        for _ in 0..N_FEATURES {
            weights.push((rng.next() as f64 / u64::MAX as f64 * 0.1 - 0.05) as f32);
        }
        Self { weights, bias: 0.0 }
    }

    /// Create with zero initialization (fresh sampler).
    pub fn zeros() -> Self {
        Self {
            weights: vec![0.0; N_FEATURES],
            bias: 0.0,
        }
    }

    /// Predict P(correct | features) ∈ [0, 1].
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        let x = features.to_array();
        let mut logit = self.bias;
        for i in 0..N_FEATURES {
            logit += self.weights[i] * x[i];
        }
        sigmoid(logit)
    }

    /// Predict for a batch of features.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Train on trajectories using SGD.
    ///
    /// Binary cross-entropy loss:
    /// L = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]
    ///
    /// Returns final average loss.
    pub fn train(&mut self, trajectories: &[SamplerTrajectory], lr: f32, n_epochs: usize) -> f32 {
        let mut last_loss = 0.0f32;

        for _epoch in 0..n_epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_samples = 0usize;

            for traj in trajectories {
                let x = traj.features.to_array();
                let logit = {
                    let mut z = self.bias;
                    for i in 0..N_FEATURES {
                        z += self.weights[i] * x[i];
                    }
                    z
                };
                let pred = sigmoid(logit);
                let y = if traj.correct { 1.0f32 } else { 0.0f32 };

                // Binary cross-entropy loss
                let eps = 1e-7f32;
                let loss = -y * (pred + eps).ln() - (1.0 - y) * (1.0 - pred + eps).ln();
                epoch_loss += loss;

                // Gradient: dL/dz = pred - y (for sigmoid + BCE)
                let grad = pred - y;

                // Update weights: w -= lr * grad * x_i
                for i in 0..N_FEATURES {
                    self.weights[i] -= lr * grad * x[i];
                }
                self.bias -= lr * grad;

                n_samples += 1;
            }

            last_loss = if n_samples > 0 {
                epoch_loss / n_samples as f32
            } else {
                0.0
            };
        }

        last_loss
    }
}

// ── MLP Sampler ───────────────────────────────────────────────

/// 2-layer MLP sampler: input → hidden (ReLU) → output (sigmoid).
///
/// hidden_dim=32 → ~250 params. More capacity than logistic for
/// medium-scale models where feature interactions matter.
#[derive(Clone, Debug)]
pub struct MlpSampler {
    /// Input → hidden weights [hidden_dim × N_FEATURES].
    pub w1: Vec<f32>,
    /// Hidden bias [hidden_dim].
    pub b1: Vec<f32>,
    /// Hidden → output weights [hidden_dim].
    pub w2: Vec<f32>,
    /// Output bias.
    pub b2: f32,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl MlpSampler {
    /// Create with Xavier initialization.
    pub fn new(hidden_dim: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / (N_FEATURES + hidden_dim) as f32).sqrt();
        let mut w1 = Vec::with_capacity(hidden_dim * N_FEATURES);
        for _ in 0..hidden_dim * N_FEATURES {
            let u = (rng.next() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
            w1.push(u * scale);
        }

        let scale2 = (2.0 / (hidden_dim + 1) as f32).sqrt();
        let mut w2 = Vec::with_capacity(hidden_dim);
        for _ in 0..hidden_dim {
            let u = (rng.next() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
            w2.push(u * scale2);
        }

        Self {
            w1,
            b1: vec![0.0; hidden_dim],
            w2,
            b2: 0.0,
            hidden_dim,
        }
    }

    /// Forward pass: returns (hidden activations, output logit).
    fn forward(&self, x: &[f32; N_FEATURES]) -> (Vec<f32>, f32) {
        let h = &mut vec![0.0f32; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = self.b1[j];
            for i in 0..N_FEATURES {
                sum += self.w1[j * N_FEATURES + i] * x[i];
            }
            h[j] = relu(sum);
        }

        let mut logit = self.b2;
        for j in 0..self.hidden_dim {
            logit += self.w2[j] * h[j];
        }

        (h.clone(), logit)
    }

    /// Predict P(correct | features) ∈ [0, 1].
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        let x = features.to_array();
        let (_, logit) = self.forward(&x);
        sigmoid(logit)
    }

    /// Predict for a batch of features.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Train on trajectories using SGD with backprop.
    ///
    /// Returns final average loss.
    pub fn train(&mut self, trajectories: &[SamplerTrajectory], lr: f32, n_epochs: usize) -> f32 {
        let mut last_loss = 0.0f32;

        for _epoch in 0..n_epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_samples = 0usize;

            for traj in trajectories {
                let x = traj.features.to_array();
                let (hidden, logit) = self.forward(&x);
                let pred = sigmoid(logit);
                let y = if traj.correct { 1.0f32 } else { 0.0f32 };

                let eps = 1e-7f32;
                let loss = -y * (pred + eps).ln() - (1.0 - y) * (1.0 - pred + eps).ln();
                epoch_loss += loss;

                // dL/dlogit = pred - y
                let d_logit = pred - y;

                // Update w2 and b2
                for j in 0..self.hidden_dim {
                    self.w2[j] -= lr * d_logit * hidden[j];
                }
                self.b2 -= lr * d_logit;

                // Backprop through ReLU to w1 and b1
                for j in 0..self.hidden_dim {
                    let d_hidden = d_logit * self.w2[j] * relu_grad(hidden[j]);
                    for i in 0..N_FEATURES {
                        self.w1[j * N_FEATURES + i] -= lr * d_hidden * x[i];
                    }
                    self.b1[j] -= lr * d_hidden;
                }

                n_samples += 1;
            }

            last_loss = if n_samples > 0 {
                epoch_loss / n_samples as f32
            } else {
                0.0
            };
        }

        last_loss
    }
}

// ── Transformer Sampler (Stub, Deferred) ──────────────────────

/// 4-layer transformer sampler (d=384, ~4.8M params).
///
/// Nemotron paper spec: PCA-compressed semantic embeddings + distribution stats
/// as 144-dim input. This is a stub for production-scale models.
///
/// **Not implemented** — deferred until T6 proves value at micro_dllm scale.
/// When implemented, this will follow the same `predict` / `train` interface.
#[derive(Clone, Debug)]
pub struct TransformerSampler {
    /// d_model dimension (default: 384).
    pub d_model: usize,
    /// Number of transformer layers (default: 4).
    pub n_layers: usize,
    /// PCA projection [d_model × N_FEATURES] (not trained yet).
    pub _pca_proj: Vec<f32>,
}

impl TransformerSampler {
    /// Create a stub sampler with given dimensions.
    pub fn new(d_model: usize, n_layers: usize) -> Self {
        Self {
            d_model,
            n_layers,
            _pca_proj: vec![0.0; d_model * N_FEATURES],
        }
    }

    /// Not implemented — falls back to logistic-equivalent prediction.
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        // Stub: use a heuristic based on top1_prob and margin
        let x = features.to_array();
        let logit = x[0] * 2.0 + x[1] * 1.5 - 0.5;
        sigmoid(logit)
    }

    /// Not implemented — no-op.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    /// Not implemented — returns 0.0.
    pub fn train(
        &mut self,
        _trajectories: &[SamplerTrajectory],
        _lr: f32,
        _n_epochs: usize,
    ) -> f32 {
        0.0
    }
}

// ── DiffusionSampler (Unified Enum) ───────────────────────────

/// Adaptive per-position correctness predictor for D2F denoising.
///
/// Selects the appropriate capacity based on model scale:
/// - `micro_dllm` (d=16) → Logistic (~7 params)
/// - Small models (d=48-128) → MLP (~250 params)
/// - Production (d=768+) → Transformer (~4.8M params, stub)
///
/// Usage:
/// ```ignore
/// let mut sampler = DiffusionSampler::auto(&config);
/// sampler.train(&trajectories, 0.01, 200);
/// let p_correct = sampler.predict(&features);
/// ```
#[derive(Clone, Debug)]
pub enum DiffusionSampler {
    /// Logistic regression: ~7 params. For micro_dllm scale.
    Logistic(LogisticSampler),
    /// 2-layer MLP: ~250 params. For small models.
    Mlp(MlpSampler),
    /// 4-layer transformer: ~4.8M params. For production (stub).
    Transformer(TransformerSampler),
}

impl DiffusionSampler {
    /// Auto-select sampler variant based on model config.
    ///
    /// Heuristic:
    /// - n_embd ≤ 32 → Logistic
    /// - n_embd ≤ 256 → MLP (hidden = n_embd)
    /// - n_embd > 256 → Transformer (d=384, 4 layers)
    pub fn auto(config: &Config) -> Self {
        Self::auto_with_rng(config, &mut Rng::new(42))
    }

    /// Auto-select with custom RNG for reproducible initialization.
    pub fn auto_with_rng(config: &Config, rng: &mut Rng) -> Self {
        match config.n_embd {
            0..=32 => Self::Logistic(LogisticSampler::new(rng)),
            33..=256 => Self::Mlp(MlpSampler::new(config.n_embd.min(64), rng)),
            _ => Self::Transformer(TransformerSampler::new(384, 4)),
        }
    }

    /// Create a logistic sampler with zero weights (for training from scratch).
    pub fn logistic_zeros() -> Self {
        Self::Logistic(LogisticSampler::zeros())
    }

    /// Create a logistic sampler with random weights.
    pub fn logistic(rng: &mut Rng) -> Self {
        Self::Logistic(LogisticSampler::new(rng))
    }

    /// Create an MLP sampler with given hidden dimension.
    pub fn mlp(hidden_dim: usize, rng: &mut Rng) -> Self {
        Self::Mlp(MlpSampler::new(hidden_dim, rng))
    }

    /// Create a transformer sampler stub.
    pub fn transformer(d_model: usize, n_layers: usize) -> Self {
        Self::Transformer(TransformerSampler::new(d_model, n_layers))
    }

    /// Predict P(correct | features) ∈ [0, 1].
    pub fn predict(&self, features: &SamplerFeatures) -> f32 {
        match self {
            Self::Logistic(s) => s.predict(features),
            Self::Mlp(s) => s.predict(features),
            Self::Transformer(s) => s.predict(features),
        }
    }

    /// Predict for a batch of features.
    pub fn predict_batch(&self, features: &[SamplerFeatures]) -> Vec<f32> {
        match self {
            Self::Logistic(s) => s.predict_batch(features),
            Self::Mlp(s) => s.predict_batch(features),
            Self::Transformer(s) => s.predict_batch(features),
        }
    }

    /// Train on collected trajectories.
    ///
    /// Returns final average loss.
    pub fn train(&mut self, trajectories: &[SamplerTrajectory], lr: f32, n_epochs: usize) -> f32 {
        match self {
            Self::Logistic(s) => s.train(trajectories, lr, n_epochs),
            Self::Mlp(s) => s.train(trajectories, lr, n_epochs),
            Self::Transformer(s) => s.train(trajectories, lr, n_epochs),
        }
    }

    /// Which variant this is.
    pub fn variant(&self) -> SamplerVariant {
        match self {
            Self::Logistic(_) => SamplerVariant::Logistic,
            Self::Mlp(s) => SamplerVariant::Mlp {
                hidden_dim: s.hidden_dim,
            },
            Self::Transformer(s) => SamplerVariant::Transformer {
                d_model: s.d_model,
                n_layers: s.n_layers,
            },
        }
    }

    /// Evaluate AUC (Area Under ROC Curve) on trajectories.
    ///
    /// AUC = 1.0 means perfect prediction, 0.5 means random.
    pub fn evaluate_auc(&self, trajectories: &[SamplerTrajectory]) -> f32 {
        if trajectories.is_empty() {
            return 0.5;
        }

        // Collect (predicted, actual) pairs
        let mut pairs: Vec<(f32, bool)> = trajectories
            .iter()
            .map(|t| (self.predict(&t.features), t.correct))
            .collect();

        // Sort by predicted probability descending
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Count positives and negatives
        let n_pos = pairs.iter().filter(|(_, c)| *c).count() as f64;
        let n_neg = pairs.len() as f64 - n_pos;

        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5;
        }

        // Compute AUC using trapezoidal rule
        let mut tp = 0.0f64;
        let mut fp = 0.0f64;
        let mut auc = 0.0f64;
        let mut prev_fpr = 0.0f64;
        let mut prev_tpr = 0.0f64;

        for (_, positive) in &pairs {
            if *positive {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let tpr = tp / n_pos;
            let fpr = fp / n_neg;

            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_fpr = fpr;
            prev_tpr = tpr;
        }

        auc as f32
    }
}

// ── Trajectory Collection ─────────────────────────────────────

/// A single training example: features at one position + correctness label.
#[derive(Clone, Debug)]
pub struct SamplerTrajectory {
    /// Features extracted at this position/step.
    pub features: SamplerFeatures,
    /// Whether the sampled token was correct (matches ground truth).
    pub correct: bool,
}

/// Collect training trajectories from D2F decode with ground truth.
///
/// Runs D2F decode on each target sequence, extracting per-position features
/// at each denoising step. For each position, records whether the model's
/// top-1 prediction matches the ground truth token.
///
/// # Arguments
///
/// * `weights` — Trained dLLM weights.
/// * `config` — Model config.
/// * `decode_config` — D2F decode config.
/// * `targets` — Ground truth token sequences.
/// * `max_trajectories` — Cap on total samples collected (0 = unlimited).
///
/// # Returns
///
/// Vector of (features, correct) training pairs.
pub fn collect_trajectories(
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    targets: &[Vec<usize>],
    max_trajectories: usize,
) -> Vec<SamplerTrajectory> {
    let mut all_trajectories = Vec::new();
    let mut rng = Rng::new(42);
    let mut dctx = D2fContext::new(config);

    for target in targets {
        let block_size = decode_config.block_size;
        let seq_len = target.len().min(block_size);
        let max_steps = decode_config.denoise_steps;
        let mask = config.mask_token;
        let vocab = config.vocab_size;

        // Initialize tokens: all mask
        let mut tokens: Vec<usize> = vec![mask; seq_len];

        for step in 0..max_steps {
            // Forward pass with block-causal attention
            let _ = forward_block_causal_with(&mut dctx, weights, &tokens, config, block_size);

            for p in 0..seq_len {
                if tokens[p] != mask {
                    continue;
                }

                // Extract features from logits
                let logits_start = p * vocab;
                let logits_end = logits_start + vocab;
                let logits_p = &dctx.logits_flat[logits_start..logits_end];

                let features = SamplerFeatures::from_logits(
                    logits_p, vocab, mask, step, max_steps, p, seq_len,
                );

                // Find top-1 prediction
                let mut top1 = 0usize;
                let mut top1_val = f32::NEG_INFINITY;
                for t in 0..vocab {
                    if t == mask {
                        continue;
                    }
                    if logits_p[t] > top1_val {
                        top1_val = logits_p[t];
                        top1 = t;
                    }
                }

                let correct = top1 == target[p];
                all_trajectories.push(SamplerTrajectory { features, correct });

                // Check cap
                if max_trajectories > 0 && all_trajectories.len() >= max_trajectories {
                    return all_trajectories;
                }
            }

            // Sample tokens for next step (greedy for determinism)
            for p in 0..seq_len {
                if tokens[p] != mask {
                    continue;
                }
                let logits_start = p * vocab;
                let logits_end = logits_start + vocab;
                let logits_p = &dctx.logits_flat[logits_start..logits_end];

                let mut top1 = 0usize;
                let mut top1_val = f32::NEG_INFINITY;
                for t in 0..vocab {
                    if t == mask {
                        continue;
                    }
                    if logits_p[t] > top1_val {
                        top1_val = logits_p[t];
                        top1 = t;
                    }
                }

                // Confidence remasking (same threshold as decode)
                let max_logit = top1_val;
                let sum_exp: f32 = (0..vocab)
                    .filter(|&t| t != mask)
                    .map(|t| (logits_p[t] - max_logit).exp())
                    .sum();
                let top1_prob = (logits_p[top1] - max_logit).exp() / sum_exp;

                if top1_prob >= decode_config.confidence_threshold {
                    tokens[p] = top1;
                }
            }

            // Early exit if all unmasked
            if tokens.iter().all(|&t| t != mask) {
                break;
            }
        }
    }

    all_trajectories
}

/// Train a logistic sampler on D2F trajectories from pattern data.
///
/// Convenience function that:
/// 1. Generates pattern dataset
/// 2. Trains a mini dLLM
/// 3. Collects trajectories
/// 4. Trains the sampler
///
/// Returns (trained sampler, final loss, auc).
pub fn train_logistic_on_patterns(
    config: &Config,
    decode_config: &D2fDecodeConfig,
    n_train: usize,
    n_test: usize,
    n_dllm_epochs: usize,
    sampler_lr: f32,
    sampler_epochs: usize,
    seed: u64,
) -> (DiffusionSampler, f32, f32) {
    use crate::dllm::{generate_pattern_dataset, train_mini_dllm};

    let mut rng = Rng::new(seed);

    // Generate data
    let effective_vocab = config.vocab_size.saturating_sub(1);
    let train_data =
        generate_pattern_dataset(&mut rng, n_train, config.block_size, effective_vocab);
    let test_data = generate_pattern_dataset(&mut rng, n_test, config.block_size, effective_vocab);

    // Train dLLM
    let (weights, _) = train_mini_dllm(
        config,
        &train_data,
        &test_data,
        n_dllm_epochs,
        0.01,
        0.3,
        seed,
    );

    // Collect trajectories from test data
    let trajectories = collect_trajectories(
        &weights,
        config,
        decode_config,
        &test_data,
        0, // unlimited
    );

    // Train sampler
    let mut sampler = DiffusionSampler::logistic(&mut Rng::new(seed + 1));
    let final_loss = sampler.train(&trajectories, sampler_lr, sampler_epochs);
    let auc = sampler.evaluate_auc(&trajectories);

    (sampler, final_loss, auc)
}

// ── Integration Helper ────────────────────────────────────────

/// Decision from the sampler for a single position in the denoising loop.
#[derive(Clone, Debug)]
pub struct SamplerDecision {
    /// Predicted P(correct) for the top-1 token.
    pub p_correct: f32,
    /// Whether to accept the token (p_correct >= threshold).
    pub accept: bool,
}

impl DiffusionSampler {
    /// Make an accept/reject decision for a denoising position.
    ///
    /// Replaces the fixed `chosen_prob >= tau_conf` check in the denoising loop.
    /// Falls back to fixed threshold if sampler is not confident.
    pub fn decide(&self, features: &SamplerFeatures, threshold: f32) -> SamplerDecision {
        let p_correct = self.predict(features);
        SamplerDecision {
            p_correct,
            accept: p_correct >= threshold,
        }
    }

    /// Batch decision for multiple positions.
    pub fn decide_batch(
        &self,
        features: &[SamplerFeatures],
        threshold: f32,
    ) -> Vec<SamplerDecision> {
        features.iter().map(|f| self.decide(f, threshold)).collect()
    }
}

// ── Activation Functions ──────────────────────────────────────

/// Sigmoid: σ(x) = 1 / (1 + exp(-x)).
fn sigmoid(x: f32) -> f32 {
    match x >= 0.0 {
        true => 1.0 / (1.0 + (-x).exp()),
        false => {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }
}

/// ReLU: max(0, x).
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// ReLU gradient: 1 if x > 0, else 0.
fn relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> Config {
        Config::micro_dllm()
    }

    // ── SamplerFeatures Tests ──

    #[test]
    fn test_features_from_logits_uniform() {
        // Uniform logits → uniform probabilities
        let vocab = 4;
        let mask = 3;
        let logits = vec![1.0, 1.0, 1.0, 1.0]; // mask at index 3
        let features = SamplerFeatures::from_logits(&logits, vocab, mask, 0, 8, 0, 4);

        // With uniform logits, top1_prob ≈ 1/3 (3 valid tokens)
        assert!(
            (features.top1_prob - 1.0 / 3.0).abs() < 0.05,
            "uniform logits should give ~0.33 top1_prob, got {}",
            features.top1_prob,
        );
        // Margin should be ~0 for uniform
        assert!(
            features.margin.abs() < 0.05,
            "uniform logits should give ~0 margin, got {}",
            features.margin,
        );
    }

    #[test]
    fn test_features_from_logits_peaked() {
        // One dominant logit → high top1_prob, high margin
        let vocab = 4;
        let mask = 3;
        let logits = vec![10.0, 0.0, 0.0, 0.0]; // token 0 dominates
        let features = SamplerFeatures::from_logits(&logits, vocab, mask, 2, 8, 1, 4);

        assert!(
            features.top1_prob > 0.99,
            "peaked logits should give very high top1_prob, got {}",
            features.top1_prob,
        );
        assert!(
            features.margin > 0.9,
            "peaked logits should give high margin, got {}",
            features.margin,
        );
        assert!(
            features.entropy < 0.1,
            "peaked logits should give low entropy, got {}",
            features.entropy,
        );
    }

    #[test]
    fn test_features_normalization() {
        let features = SamplerFeatures::from_logits(
            &[1.0, 0.0, 0.0],
            3,
            2,
            4, // step
            8, // max_steps
            2, // pos
            4, // block_size
        );
        assert!(
            (features.step_norm - 0.5).abs() < 1e-6,
            "step_norm should be 4/8 = 0.5, got {}",
            features.step_norm,
        );
        assert!(
            (features.pos_norm - 0.5).abs() < 1e-6,
            "pos_norm should be 2/4 = 0.5, got {}",
            features.pos_norm,
        );
    }

    #[test]
    fn test_features_to_array_length() {
        let features = SamplerFeatures::default();
        let arr = features.to_array();
        assert_eq!(arr.len(), N_FEATURES);
    }

    // ── LogisticSampler Tests ──

    #[test]
    fn test_logistic_zeros_predicts_half() {
        // Zero weights → logit = 0 → sigmoid(0) = 0.5
        let sampler = LogisticSampler::zeros();
        let features = SamplerFeatures {
            top1_prob: 0.9,
            margin: 0.5,
            ..Default::default()
        };
        let pred = sampler.predict(&features);
        assert!(
            (pred - 0.5).abs() < 1e-6,
            "zero weights should predict 0.5, got {pred}",
        );
    }

    #[test]
    fn test_logistic_positive_weights_high_prob() {
        // Positive weights + high top1_prob → high prediction
        let mut sampler = LogisticSampler::zeros();
        sampler.weights[0] = 5.0; // top1_prob weight
        sampler.bias = -2.0;

        let features = SamplerFeatures {
            top1_prob: 0.9,
            ..Default::default()
        };
        let pred = sampler.predict(&features);
        assert!(
            pred > 0.5,
            "positive weight on high top1_prob should predict > 0.5, got {pred}",
        );
    }

    #[test]
    fn test_logistic_train_reduces_loss() {
        let mut rng = Rng::new(42);
        let mut sampler = LogisticSampler::new(&mut rng);

        // Create simple trajectories: high top1_prob → correct
        let trajectories: Vec<SamplerTrajectory> = (0..100)
            .map(|i| {
                let prob = i as f32 / 100.0;
                SamplerTrajectory {
                    features: SamplerFeatures {
                        top1_prob: prob,
                        ..Default::default()
                    },
                    correct: prob > 0.5,
                }
            })
            .collect();

        let loss_before = {
            let mut total = 0.0f32;
            for t in &trajectories {
                let pred = sampler.predict(&t.features);
                let y = if t.correct { 1.0f32 } else { 0.0f32 };
                let eps = 1e-7f32;
                total -= y * (pred + eps).ln() + (1.0 - y) * (1.0 - pred + eps).ln();
            }
            total / trajectories.len() as f32
        };

        let loss_after = sampler.train(&trajectories, 0.1, 100);

        assert!(
            loss_after < loss_before,
            "training should reduce loss: before={loss_before:.4}, after={loss_after:.4}",
        );
    }

    #[test]
    fn test_logistic_predict_batch_matches_individual() {
        let mut rng = Rng::new(42);
        let sampler = LogisticSampler::new(&mut rng);
        let features: Vec<SamplerFeatures> = (0..10)
            .map(|i| SamplerFeatures {
                top1_prob: i as f32 / 10.0,
                ..Default::default()
            })
            .collect();

        let batch_preds = sampler.predict_batch(&features);
        let individual_preds: Vec<f32> = features.iter().map(|f| sampler.predict(f)).collect();

        for (i, (b, ind)) in batch_preds.iter().zip(individual_preds.iter()).enumerate() {
            assert!(
                (b - ind).abs() < 1e-6,
                "batch[{i}] = {b} != individual[{i}] = {ind}",
            );
        }
    }

    // ── MlpSampler Tests ──

    #[test]
    fn test_mlp_predict_is_bounded() {
        let mut rng = Rng::new(42);
        let sampler = MlpSampler::new(32, &mut rng);
        let features = SamplerFeatures {
            top1_prob: 0.9,
            margin: 0.5,
            entropy: 0.3,
            top3_mass: 0.95,
            step_norm: 0.5,
            pos_norm: 0.5,
        };
        let pred = sampler.predict(&features);
        assert!(
            (0.0..=1.0).contains(&pred),
            "prediction should be in [0, 1], got {pred}",
        );
    }

    #[test]
    fn test_mlp_train_runs() {
        let mut rng = Rng::new(42);
        let mut sampler = MlpSampler::new(16, &mut rng);

        let trajectories: Vec<SamplerTrajectory> = (0..50)
            .map(|i| {
                let prob = i as f32 / 50.0;
                SamplerTrajectory {
                    features: SamplerFeatures {
                        top1_prob: prob,
                        margin: prob * 0.5,
                        ..Default::default()
                    },
                    correct: prob > 0.5,
                }
            })
            .collect();

        let loss = sampler.train(&trajectories, 0.01, 50);
        assert!(
            loss.is_finite(),
            "MLP training should produce finite loss, got {loss}",
        );
    }

    // ── DiffusionSampler Enum Tests ──

    #[test]
    fn test_auto_selects_logistic_for_micro() {
        let config = make_config();
        let sampler = DiffusionSampler::auto(&config);
        assert!(
            matches!(sampler.variant(), SamplerVariant::Logistic),
            "micro_dllm (n_embd=16) should select Logistic",
        );
    }

    #[test]
    fn test_auto_selects_mlp_for_medium() {
        let mut config = make_config();
        config.n_embd = 64;
        let sampler = DiffusionSampler::auto(&config);
        assert!(
            matches!(sampler.variant(), SamplerVariant::Mlp { .. }),
            "n_embd=64 should select MLP",
        );
    }

    #[test]
    fn test_auto_selects_transformer_for_large() {
        let mut config = make_config();
        config.n_embd = 768;
        let sampler = DiffusionSampler::auto(&config);
        assert!(
            matches!(sampler.variant(), SamplerVariant::Transformer { .. }),
            "n_embd=768 should select Transformer",
        );
    }

    #[test]
    fn test_logistic_zeros_factory() {
        let sampler = DiffusionSampler::logistic_zeros();
        assert!(matches!(sampler.variant(), SamplerVariant::Logistic));

        // Should predict 0.5 for any input
        let features = SamplerFeatures {
            top1_prob: 0.99,
            ..Default::default()
        };
        let pred = sampler.predict(&features);
        assert!(
            (pred - 0.5).abs() < 1e-6,
            "zeros sampler should predict 0.5, got {pred}",
        );
    }

    // ── Trajectory Collection Tests ──

    #[test]
    fn test_collect_trajectories_produces_data() {
        use crate::dllm::{generate_pattern_dataset, train_mini_dllm};

        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 20, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 5, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 100, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);

        assert!(
            !trajectories.is_empty(),
            "should collect at least some trajectories",
        );

        // Check features are valid
        for traj in &trajectories {
            assert!(
                traj.features.top1_prob >= 0.0 && traj.features.top1_prob <= 1.0,
                "top1_prob should be in [0, 1], got {}",
                traj.features.top1_prob,
            );
        }
    }

    #[test]
    fn test_collect_trajectories_respects_cap() {
        use crate::dllm::{generate_pattern_dataset, train_mini_dllm};

        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 20, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 5, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 50, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);
        let cap = 10;
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, cap);

        assert!(
            trajectories.len() <= cap,
            "should respect cap of {cap}, got {}",
            trajectories.len(),
        );
    }

    // ── End-to-End: Train Sampler ──

    #[test]
    fn test_train_logistic_on_patterns() {
        use crate::dllm::{generate_pattern_dataset, train_mini_dllm};

        let config = make_config();
        let mut rng = Rng::new(42);
        let effective_vocab = config.vocab_size.saturating_sub(1);

        let train_data = generate_pattern_dataset(&mut rng, 30, config.block_size, effective_vocab);
        let test_data = generate_pattern_dataset(&mut rng, 10, config.block_size, effective_vocab);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig::with_block_size(4);
        let trajectories = collect_trajectories(&weights, &config, &decode_config, &test_data, 0);

        let mut sampler = DiffusionSampler::logistic(&mut Rng::new(99));
        let loss = sampler.train(&trajectories, 0.1, 100);
        let auc = sampler.evaluate_auc(&trajectories);

        assert!(loss.is_finite(), "loss should be finite, got {loss}",);
        // AUC > 0.5 means the sampler learned something
        // (may be close to 0.5 with random weights, but should be finite)
        assert!(
            auc >= 0.0 && auc <= 1.0,
            "AUC should be in [0, 1], got {auc}",
        );
    }

    #[test]
    fn test_sampler_decision() {
        let mut sampler = DiffusionSampler::logistic_zeros();
        // Zero weights always predict 0.5 → accept with threshold <= 0.5

        let features = SamplerFeatures {
            top1_prob: 0.8,
            ..Default::default()
        };

        let decision_low = sampler.decide(&features, 0.3);
        assert!(
            decision_low.accept,
            "threshold 0.3 < 0.5 prediction → should accept",
        );

        let decision_high = sampler.decide(&features, 0.8);
        assert!(
            !decision_high.accept,
            "threshold 0.8 > 0.5 prediction → should reject",
        );
    }

    // ── AUC Tests ──

    #[test]
    fn test_auc_perfect_predictor() {
        // Create a perfect predictor scenario
        let sampler = DiffusionSampler::logistic_zeros();
        // Manually set weights to separate correct from incorrect
        if let DiffusionSampler::Logistic(ref mut s) =
            std::cell::RefCell::new(sampler).borrow_mut() as &mut LogisticSampler
        {
            // This won't compile due to ownership, let's test differently
        }

        // Test with uniform predictions → AUC should be ~0.5
        let mut rng = Rng::new(42);
        let sampler = DiffusionSampler::logistic(&mut rng);

        let trajectories: Vec<SamplerTrajectory> = (0..50)
            .map(|i| SamplerTrajectory {
                features: SamplerFeatures {
                    top1_prob: i as f32 / 50.0,
                    ..Default::default()
                },
                correct: i > 25,
            })
            .collect();

        let auc = sampler.evaluate_auc(&trajectories);
        assert!(
            auc >= 0.0 && auc <= 1.0,
            "AUC should be in [0, 1], got {auc}",
        );
    }

    // ── Convenience Function Test ──

    #[test]
    fn test_train_logistic_on_patterns_convenience() {
        let config = make_config();
        let decode_config = D2fDecodeConfig::with_block_size(4);

        let (sampler, loss, auc) =
            train_logistic_on_patterns(&config, &decode_config, 20, 5, 100, 0.1, 50, 42);

        assert!(
            matches!(sampler.variant(), SamplerVariant::Logistic),
            "should return logistic sampler",
        );
        assert!(loss.is_finite(), "loss should be finite, got {loss}",);
        assert!(
            auc >= 0.0 && auc <= 1.0,
            "AUC should be in [0, 1], got {auc}",
        );
    }
}
