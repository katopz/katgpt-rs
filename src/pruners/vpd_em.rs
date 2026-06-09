//! VPD EM-Style Modelless Distillation — Co-evolutionary teacher-student loop.
//!
//! Distilled from VPD (arXiv:2605.15113, Salesforce AI Research, 2026):
//! - E-step: refine teacher via BCO unpaired preference on feedback signals
//! - M-step: distill refined teacher → student via KL-gated absorb-compress
//! - Dynamic prior: anchor E-step to current student Q (not frozen baseline)
//!
//! # Key Insight
//!
//! SDAR treats the feedback-conditioned "teacher" as a passive signal processor.
//! VPD proves this plateaus — the teacher must be **actively trained** to distinguish
//! success/failure given feedback. BCO (Binary Classifier Optimization) provides the
//! unpaired preference signal: `L = -E_{y+}[log σ(r̃ - δ)] - E_{y-}[log σ(-(r̃ - δ))]`.
//!
//! # Components
//!
//! - [`BcoSample`] — unpaired preference sample (action, outcome, implicit_reward, feedback)
//! - [`BcoOptimizer`] — BCO loss computation and reward shift δ tracking
//! - [`VpdConfig`] — EM cycle configuration (frequency, temperature, KL penalty)
//! - [`VpdEmCycle`] — the alternating E/M loop state machine
//!
//! # Architecture
//!
//! ```text
//! VpdEmCycle
//!   ├── BcoOptimizer          (E-step: unpaired preference teacher refinement)
//!   ├── SdarGatedAbsorbCompress (M-step: KL-gated student distillation)
//!   ├── student_q              (dynamic prior — current student Q-values)
//!   ├── teacher_q              (feedback-conditioned, updated in E-step)
//!   └── reference_q            (frozen baseline for fixed-prior ablation)
//! ```
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "vpd_em_distill")`.
//! Feature: `vpd_em_distill = ["sdar_gate", "bandit"]` in `Cargo.toml`.
//!
//! **Source:** [VPD: Learning from Language Feedback via Variational Policy Distillation](https://arxiv.org/abs/2605.15113) — Salesforce AI Research, 2026

use std::marker::PhantomData;

use crate::pruners::sdar::SdarGatedAbsorbCompress;
use crate::pruners::sdar_gate::{SDAR_BETA, sdar_gate};
use crate::speculative::types::ScreeningPruner;

// ── Numerics ──────────────────────────────────────────────────

/// Numerically stable log-sigmoid: `log σ(x) = -softplus(-x)`.
///
/// Uses two-branch formula to avoid overflow in either direction:
/// - `x >= 0`: `-(1 + exp(-x)).ln()` — `exp(-x)` bounded by 1
/// - `x < 0`:  `x - (1 + exp(x)).ln()` — `exp(x)` bounded by 1
#[inline]
fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// Softmax over a slice, returning log-probabilities.
///
/// Numerically stable: subtract max before exp to avoid overflow.
/// Returns `Vec<f32>` of same length where values sum to ~1.0 in probability space.
fn softmax(q: &[f32]) -> Vec<f32> {
    if q.is_empty() {
        return Vec::new();
    }
    let max_val = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = q.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let log_sum = sum.ln();
    // Return log-probabilities for KL computation
    exps.iter().map(|e| e.ln() - log_sum).collect()
}

/// In-place softmax that reuses a pre-allocated buffer.
///
/// Writes log-probabilities into `out`, avoiding allocation on repeated calls.
fn softmax_inplace(q: &[f32], out: &mut Vec<f32>) {
    out.clear();
    if q.is_empty() {
        return;
    }
    let max_val = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    out.extend(q.iter().map(|&v| (v - max_val).exp()));
    let sum: f32 = out.iter().sum();
    let log_sum = sum.ln();
    for e in out.iter_mut() {
        *e = e.ln() - log_sum;
    }
}

/// KL divergence: `sum(p * (p - q))` where p and q are log-probabilities.
///
/// Returns 0.0 if slices are empty or mismatched.
fn kl_divergence(log_p: &[f32], log_q: &[f32]) -> f32 {
    if log_p.len() != log_q.len() || log_p.is_empty() {
        return 0.0;
    }
    log_p
        .iter()
        .zip(log_q.iter())
        .map(|(&lp, &lq)| lp.exp() * (lp - lq))
        .sum()
}

// ── BCO Types ─────────────────────────────────────────────────

/// Implicit reward clamp bounds (prevents BCO loss overflow).
const IMPLICIT_REWARD_CLAMP: (f32, f32) = (-10.0, 10.0);

/// Unpaired preference sample for BCO (Binary Classifier Optimization).
///
/// Unlike [`BtComparison`](crate::pruners::bt_rank::BtComparison) which records
/// paired winner/loser, BCO operates on individual samples with implicit rewards
/// derived from the teacher-student quality gap.
///
/// # Fields
///
/// - `action_idx`: which action/template was evaluated
/// - `outcome`: binary success (1.0) or failure (0.0)
/// - `implicit_reward`: `β · log(q_φ / π_θ)` — teacher-student quality gap
/// - `feedback_signal`: scalar from sdar_gate or similar signal processor
#[derive(Debug, Clone, Copy)]
pub struct BcoSample {
    /// Index of the action/template being evaluated.
    pub action_idx: usize,
    /// Binary outcome reward: 1.0 for success, 0.0 for failure.
    pub outcome: f32,
    /// Implicit reward: teacher-student quality gap (clamped to [-10, 10]).
    pub implicit_reward: f32,
    /// Feedback signal (scalar, from sdar_gate or similar).
    pub feedback_signal: f32,
}

impl BcoSample {
    /// Create a new BCO sample, clamping implicit reward to valid range.
    pub fn new(
        action_idx: usize,
        outcome: f32,
        implicit_reward: f32,
        feedback_signal: f32,
    ) -> Self {
        Self {
            action_idx,
            outcome,
            implicit_reward: implicit_reward
                .clamp(IMPLICIT_REWARD_CLAMP.0, IMPLICIT_REWARD_CLAMP.1),
            feedback_signal,
        }
    }
}

/// BCO optimizer for unpaired preference learning.
///
/// Implements the BCO loss from VPD (arXiv:2605.15113, Eq. 4-5):
/// ```text
/// L = -E_{y+}[log σ(r̃ - δ)] - E_{y-}[log σ(-(r̃ - δ))]
/// ```
///
/// The reward shift δ is maintained as an EMA of the midpoint between
/// positive and negative sample averages, centering the BCO signal.
#[derive(Debug, Clone)]
pub struct BcoOptimizer {
    /// BCO temperature τ (paper: 0.1). Scales the reward signal before sigmoid.
    pub temperature: f32,
    /// Moving average reward shift δ. Centers positive/negative samples.
    pub reward_shift: f32,
    /// EMA momentum for δ update (paper: 0.9).
    pub shift_momentum: f32,
}

impl BcoOptimizer {
    /// Create a new BCO optimizer with the given temperature.
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            reward_shift: 0.0,
            shift_momentum: 0.9,
        }
    }

    /// Compute BCO loss for a batch of unpaired samples.
    ///
    /// ```text
    /// L = -(1/N) * Σ log σ(scaled_r̃ - δ)    for positive samples
    ///   - (1/N) * Σ log σ(-(scaled_r̃ - δ))   for negative samples
    /// ```
    ///
    /// Uses [`log_sigmoid`] for numerical stability.
    pub fn compute_loss(&self, samples: &[BcoSample]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let n = samples.len() as f32;
        let mut loss = 0.0f32;
        for s in samples {
            let r_tilde = (s.implicit_reward - self.reward_shift) / self.temperature;
            if s.outcome > 0.5 {
                // Positive sample: log σ(r̃ - δ)
                loss -= log_sigmoid(r_tilde);
            } else {
                // Negative sample: log σ(-(r̃ - δ))
                loss -= log_sigmoid(-r_tilde);
            }
        }
        loss / n
    }

    /// Update reward shift δ via EMA of midpoint between pos/neg averages.
    ///
    /// ```text
    /// target = 0.5 * (E[r̃(y+)] + E[r̃(y-)])
    /// δ ← momentum * δ + (1 - momentum) * target
    /// ```
    pub fn update_shift(&mut self, samples: &[BcoSample]) {
        if samples.is_empty() {
            return;
        }
        let (pos_sum, neg_sum, pos_n, neg_n) =
            samples
                .iter()
                .fold((0.0f32, 0.0f32, 0usize, 0usize), |(ps, ns, pn, nn), s| {
                    if s.outcome > 0.5 {
                        (ps + s.implicit_reward, ns, pn + 1, nn)
                    } else {
                        (ps, ns + s.implicit_reward, pn, nn + 1)
                    }
                });
        let pos_avg = if pos_n > 0 {
            pos_sum / pos_n as f32
        } else {
            0.0
        };
        let neg_avg = if neg_n > 0 {
            neg_sum / neg_n as f32
        } else {
            0.0
        };
        let target = 0.5 * (pos_avg + neg_avg);
        self.reward_shift =
            self.shift_momentum * self.reward_shift + (1.0 - self.shift_momentum) * target;
    }
}

// ── VPD Config ────────────────────────────────────────────────

/// VPD EM cycle configuration.
///
/// Defaults from VPD paper Table C.1 and C.2 (arXiv:2605.15113):
/// - `e_step_frequency = 5` — 1 E-step per 5 M-steps
/// - `bco_temperature = 0.1` — BCO reward scaling
/// - `kl_penalty = 0.1` — KL divergence gating strength
/// - `dynamic_prior = true` — use current student Q as anchor
#[derive(Debug, Clone)]
pub struct VpdConfig {
    /// E-step frequency: 1 E-step per F M-steps (paper: F=5).
    pub e_step_frequency: usize,
    /// BCO temperature β (paper: 0.1). Scales implicit reward before sigmoid.
    pub bco_temperature: f32,
    /// KL penalty strength for M-step gating (paper: 0.1).
    pub kl_penalty: f32,
    /// Use dynamic prior (π_θ) vs fixed prior (π_ref).
    /// Paper ablation: dynamic 74.34 vs fixed 67.84 on SciKnowEval.
    pub dynamic_prior: bool,
}

impl Default for VpdConfig {
    fn default() -> Self {
        Self {
            e_step_frequency: 5,
            bco_temperature: 0.1,
            kl_penalty: 0.1,
            dynamic_prior: true,
        }
    }
}

impl VpdConfig {
    /// Create config with custom E-step frequency.
    pub fn with_frequency(mut self, freq: usize) -> Self {
        self.e_step_frequency = freq.max(1);
        self
    }

    /// Create config with dynamic prior disabled (for ablation).
    pub fn with_fixed_prior(mut self) -> Self {
        self.dynamic_prior = false;
        self
    }
}

// ── VPD EM Cycle ──────────────────────────────────────────────

/// VPD EM cycle state machine.
///
/// Alternates between:
/// - **M-step** (every round): KL-gated distillation of teacher → student via
///   [`SdarGatedAbsorbCompress`]
/// - **E-step** (every F M-steps): BCO unpaired preference refinement of teacher
///
/// The dynamic prior anchors the E-step to current student Q-values rather than
/// a frozen baseline, preventing distribution shift as the student improves.
///
/// # Type Parameters
///
/// - `P`: The inner [`ScreeningPruner`] wrapped by the absorb-compress layer.
pub struct VpdEmCycle<P: ScreeningPruner> {
    /// EM cycle configuration.
    config: VpdConfig,
    /// BCO optimizer for E-step teacher refinement.
    bco: BcoOptimizer,
    /// M-step counter — triggers E-step every `config.e_step_frequency` M-steps.
    m_step_count: usize,
    /// Current student Q-values (dynamic prior for E-step anchoring).
    student_q: Vec<f32>,
    /// Teacher Q-values (feedback-conditioned, updated in E-step).
    teacher_q: Vec<f32>,
    /// Reference Q-values (frozen from initialization, for fixed-prior ablation).
    reference_q: Vec<f32>,
    /// Collected samples for next E-step batch.
    e_step_buffer: Vec<BcoSample>,
    /// Pre-allocated scratch buffer for student log-probabilities (avoids per-M-step allocation).
    student_log_p: Vec<f32>,
    /// Pre-allocated scratch buffer for teacher log-probabilities (avoids per-M-step allocation).
    teacher_log_p: Vec<f32>,
    /// Phantom data for the ScreeningPruner type parameter.
    _phantom: PhantomData<P>,
}

impl<P: ScreeningPruner> VpdEmCycle<P> {
    /// Create a new VPD EM cycle with the given config and number of actions.
    ///
    /// Initializes all Q-value vectors to zero. The `reference_q` is frozen
    /// at initialization for the fixed-prior ablation study.
    pub fn new(config: VpdConfig, n_actions: usize) -> Self {
        let bco = BcoOptimizer::new(config.bco_temperature);
        Self {
            config,
            bco,
            m_step_count: 0,
            student_q: vec![0.0; n_actions],
            teacher_q: vec![0.0; n_actions],
            reference_q: vec![0.0; n_actions],
            e_step_buffer: Vec::new(),
            student_log_p: Vec::with_capacity(n_actions),
            teacher_log_p: Vec::with_capacity(n_actions),
            _phantom: PhantomData,
        }
    }

    /// Collect a sample for the next E-step batch.
    ///
    /// Samples accumulate during M-steps and are flushed when the E-step fires.
    pub fn collect_sample(&mut self, action_idx: usize, outcome: f32, feedback_signal: f32) {
        // Compute implicit reward from teacher-student quality gap
        let student_val = self.student_q.get(action_idx).copied().unwrap_or(0.0);
        let teacher_val = self.teacher_q.get(action_idx).copied().unwrap_or(0.0);
        let implicit_reward = if self.config.dynamic_prior {
            // Dynamic prior: use current student Q
            teacher_val - student_val
        } else {
            // Fixed prior: use frozen reference Q
            teacher_val - self.reference_q.get(action_idx).copied().unwrap_or(0.0)
        };

        let sample = BcoSample::new(action_idx, outcome, implicit_reward, feedback_signal);
        self.e_step_buffer.push(sample);
    }

    /// Run E-step: refine teacher Q-values via BCO on collected samples.
    ///
    /// 1. Update reward shift δ via EMA
    /// 2. Compute BCO loss (for diagnostics/logging)
    /// 3. Update teacher Q-values: positive samples push up, negative push down
    /// 4. Clear the sample buffer
    ///
    /// Returns the BCO loss for this batch, or 0.0 if the buffer was empty.
    pub fn e_step(&mut self) -> f32 {
        if self.e_step_buffer.is_empty() {
            return 0.0;
        }

        // Update reward shift before loss computation
        self.bco.update_shift(&self.e_step_buffer);

        // Compute loss for diagnostics
        let loss = self.bco.compute_loss(&self.e_step_buffer);

        // Update teacher Q-values based on BCO signal
        for sample in &self.e_step_buffer {
            let r_tilde = (sample.implicit_reward - self.bco.reward_shift) / self.bco.temperature;
            if let Some(q) = self.teacher_q.get_mut(sample.action_idx) {
                if sample.outcome > 0.5 {
                    // Positive sample: nudge teacher Q up
                    *q += r_tilde * 0.1;
                } else {
                    // Negative sample: nudge teacher Q down
                    *q -= r_tilde.abs() * 0.1;
                }
            }
        }

        self.e_step_buffer.clear();
        loss
    }

    /// Run M-step: KL-gated distillation of teacher → student.
    ///
    /// 1. Compute action-level KL divergence (student || teacher)
    /// 2. Gate the reward signal using SDAR sigmoid
    /// 3. Feed gated reward to absorb-compress layer
    /// 4. Soft-update student Q towards teacher Q (dynamic prior)
    ///
    /// Returns true if an E-step should follow this M-step.
    pub fn m_step(
        &mut self,
        action_idx: usize,
        reward: f32,
        absorb: &mut SdarGatedAbsorbCompress<P>,
    ) -> bool {
        // Compute action-level KL divergence as gating signal
        softmax_inplace(&self.student_q, &mut self.student_log_p);
        softmax_inplace(&self.teacher_q, &mut self.teacher_log_p);
        let kl = kl_divergence(&self.student_log_p, &self.teacher_log_p);

        // Gate the distillation signal using SDAR sigmoid
        let gate = sdar_gate(kl * self.config.kl_penalty, SDAR_BETA);

        // Absorb with gated signal (teacher Q as target)
        let teacher_q_val = self.teacher_q.get(action_idx).copied().unwrap_or(0.0);
        absorb.observe_with_q(action_idx, reward * gate, teacher_q_val);

        // Soft-update student Q towards teacher (η=0.2)
        if let Some(sq) = self.student_q.get_mut(action_idx) {
            *sq = *sq + 0.2 * (teacher_q_val - *sq);
        }

        self.m_step_count += 1;

        // Check if E-step is due
        self.should_e_step()
    }

    /// Returns true when an E-step should fire (every `e_step_frequency` M-steps).
    pub fn should_e_step(&self) -> bool {
        self.m_step_count > 0
            && self
                .m_step_count
                .is_multiple_of(self.config.e_step_frequency)
    }

    /// Current student Q-values (dynamic prior).
    pub fn student_q(&self) -> &[f32] {
        &self.student_q
    }

    /// Current teacher Q-values (feedback-conditioned).
    pub fn teacher_q(&self) -> &[f32] {
        &self.teacher_q
    }

    /// Reference Q-values (frozen baseline for fixed-prior ablation).
    pub fn reference_q(&self) -> &[f32] {
        &self.reference_q
    }

    /// Set teacher Q-values (for testing / external initialization).
    pub fn set_teacher_q(&mut self, q: Vec<f32>) {
        debug_assert_eq!(q.len(), self.teacher_q.len());
        self.teacher_q = q;
    }

    /// Set student Q-values (for testing / external initialization).
    pub fn set_student_q(&mut self, q: Vec<f32>) {
        debug_assert_eq!(q.len(), self.student_q.len());
        self.student_q = q;
    }

    /// Number of M-steps completed.
    pub fn m_step_count(&self) -> usize {
        self.m_step_count
    }

    /// Current BCO reward shift δ.
    pub fn reward_shift(&self) -> f32 {
        self.bco.reward_shift
    }

    /// Number of samples in the E-step buffer.
    pub fn buffer_len(&self) -> usize {
        self.e_step_buffer.len()
    }

    /// Configuration reference.
    pub fn config(&self) -> &VpdConfig {
        &self.config
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── log_sigmoid tests ──────────────────────────────────────

    #[test]
    fn test_log_sigmoid_zero() {
        // log σ(0) = log(0.5) ≈ -0.6931
        let val = log_sigmoid(0.0);
        assert!(
            (val - (-0.6931f32)).abs() < 0.01,
            "log_sigmoid(0) ≈ -0.693, got {val}"
        );
    }

    #[test]
    fn test_log_sigmoid_large_positive() {
        // log σ(100) ≈ 0
        let val = log_sigmoid(100.0);
        assert!(val > -1e-3, "log_sigmoid(100) ≈ 0, got {val}");
    }

    #[test]
    fn test_log_sigmoid_large_negative() {
        // log σ(-100) ≈ -100
        let val = log_sigmoid(-100.0);
        assert!(
            (val - (-100.0f32)).abs() < 0.1,
            "log_sigmoid(-100) ≈ -100, got {val}"
        );
    }

    #[test]
    fn test_log_sigmoid_negative_input() {
        // Uses the x < 0 branch
        let val = log_sigmoid(-2.0);
        let expected = (-2.0f32) - (1.0 + (-2.0f32).exp()).ln();
        assert!(
            (val - expected).abs() < 1e-5,
            "log_sigmoid(-2) = {val}, expected {expected}"
        );
    }

    #[test]
    fn test_log_sigmoid_positive_input() {
        // Uses the x >= 0 branch
        let val = log_sigmoid(3.0);
        let expected = -(1.0 + (-3.0f32).exp()).ln();
        assert!(
            (val - expected).abs() < 1e-5,
            "log_sigmoid(3) = {val}, expected {expected}"
        );
    }

    // ── softmax tests ──────────────────────────────────────────

    #[test]
    fn test_softmax_normalizes() {
        let q = vec![1.0, 2.0, 3.0];
        let log_p = softmax(&q);
        // Probabilities should sum to 1
        let prob_sum: f32 = log_p.iter().map(|lp| lp.exp()).sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-5,
            "softmax probabilities sum to {prob_sum}, expected 1.0"
        );
    }

    #[test]
    fn test_softmax_empty() {
        let log_p = softmax(&[]);
        assert!(log_p.is_empty(), "softmax of empty should be empty");
    }

    #[test]
    fn test_softmax_uniform() {
        let q = vec![2.0, 2.0, 2.0];
        let log_p = softmax(&q);
        // All log-probs should be equal: log(1/3)
        let expected = (1.0f32 / 3.0).ln();
        for (i, &lp) in log_p.iter().enumerate() {
            assert!(
                (lp - expected).abs() < 1e-5,
                "softmax uniform [{i}] = {lp}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_softmax_peak() {
        let q = vec![0.0, 10.0, 0.0];
        let log_p = softmax(&q);
        let probs: Vec<f32> = log_p.iter().map(|lp| lp.exp()).collect();
        assert!(
            probs[1] > 0.99,
            "softmax peak at index 1: probs[1] = {}",
            probs[1]
        );
    }

    // ── KL divergence tests ────────────────────────────────────

    #[test]
    fn test_kl_divergence_identical() {
        let log_p = vec![-1.0f32, -2.0, -3.0];
        let kl = kl_divergence(&log_p, &log_p);
        assert!(kl.abs() < 1e-5, "KL(p||p) = 0, got {kl}");
    }

    #[test]
    fn test_kl_divergence_nonnegative() {
        let log_p = vec![-1.0f32, -2.0, -3.0];
        let log_q = vec![-2.0f32, -1.0, -0.5];
        let kl = kl_divergence(&log_p, &log_q);
        assert!(
            kl >= -1e-5,
            "KL divergence should be non-negative, got {kl}"
        );
    }

    #[test]
    fn test_kl_divergence_empty() {
        let kl = kl_divergence(&[], &[]);
        assert_eq!(kl, 0.0, "KL of empty should be 0");
    }

    #[test]
    fn test_kl_divergence_mismatched() {
        let log_p = vec![-1.0f32];
        let log_q = vec![-1.0f32, -2.0];
        let kl = kl_divergence(&log_p, &log_q);
        assert_eq!(kl, 0.0, "KL of mismatched lengths should be 0");
    }

    // ── BcoSample tests ────────────────────────────────────────

    #[test]
    fn test_bco_sample_clamps_implicit_reward() {
        let sample = BcoSample::new(0, 1.0, 100.0, 0.5);
        assert_eq!(sample.implicit_reward, 10.0, "clamped to 10");

        let sample = BcoSample::new(0, 1.0, -100.0, 0.5);
        assert_eq!(sample.implicit_reward, -10.0, "clamped to -10");
    }

    // ── BcoOptimizer tests ─────────────────────────────────────

    #[test]
    fn test_bco_loss_positive_samples() {
        let bco = BcoOptimizer::new(0.1);
        let samples = vec![
            BcoSample::new(0, 1.0, 1.0, 0.5),
            BcoSample::new(1, 1.0, 2.0, 0.5),
        ];
        let loss = bco.compute_loss(&samples);
        // Positive samples with positive implicit reward → log σ(positive) → near 0
        // Loss = -log σ(positive) → small positive
        assert!(
            loss >= 0.0,
            "BCO loss should be non-negative for positive samples, got {loss}"
        );
    }

    #[test]
    fn test_bco_loss_negative_samples() {
        let bco = BcoOptimizer::new(0.1);
        let samples = vec![
            BcoSample::new(0, 0.0, -1.0, 0.5),
            BcoSample::new(1, 0.0, -2.0, 0.5),
        ];
        let loss = bco.compute_loss(&samples);
        assert!(
            loss >= 0.0,
            "BCO loss should be non-negative for negative samples, got {loss}"
        );
    }

    #[test]
    fn test_bco_loss_low_implicit_reward_positive_sample() {
        let bco = BcoOptimizer::new(0.1);
        // Positive sample with very low implicit reward → high loss
        let samples = vec![BcoSample::new(0, 1.0, -5.0, 0.5)];
        let loss = bco.compute_loss(&samples);
        assert!(
            loss > 0.0,
            "Positive sample with low implicit reward should produce loss > 0, got {loss}"
        );
    }

    #[test]
    fn test_bco_loss_empty() {
        let bco = BcoOptimizer::new(0.1);
        let loss = bco.compute_loss(&[]);
        assert_eq!(loss, 0.0, "BCO loss of empty samples should be 0");
    }

    #[test]
    fn test_bco_shift_update() {
        let mut bco = BcoOptimizer::new(0.1);
        let samples = vec![
            BcoSample::new(0, 1.0, 4.0, 0.5), // positive avg → 4.0
            BcoSample::new(1, 0.0, 2.0, 0.5), // negative avg → 2.0
        ];
        bco.update_shift(&samples);
        // target = 0.5 * (4.0 + 2.0) = 3.0
        // shift = 0.9 * 0.0 + 0.1 * 3.0 = 0.3
        let expected = 0.3f32;
        assert!(
            (bco.reward_shift - expected).abs() < 1e-5,
            "reward shift = {}, expected {expected}",
            bco.reward_shift
        );
    }

    #[test]
    fn test_bco_shift_update_ema_convergence() {
        let mut bco = BcoOptimizer::new(0.1);
        let samples = vec![
            BcoSample::new(0, 1.0, 4.0, 0.5),
            BcoSample::new(1, 0.0, 2.0, 0.5),
        ];
        // Repeated updates should converge to target = 3.0
        for _ in 0..100 {
            bco.update_shift(&samples);
        }
        assert!(
            (bco.reward_shift - 3.0).abs() < 0.01,
            "reward shift should converge to 3.0, got {}",
            bco.reward_shift
        );
    }

    // ── VpdConfig tests ────────────────────────────────────────

    #[test]
    fn test_vpd_config_default() {
        let config = VpdConfig::default();
        assert_eq!(config.e_step_frequency, 5, "default e_step_frequency = 5");
        assert!(
            (config.bco_temperature - 0.1).abs() < 1e-5,
            "default bco_temperature = 0.1"
        );
        assert!(
            (config.kl_penalty - 0.1).abs() < 1e-5,
            "default kl_penalty = 0.1"
        );
        assert!(config.dynamic_prior, "default dynamic_prior = true");
    }

    #[test]
    fn test_vpd_config_with_frequency() {
        let config = VpdConfig::default().with_frequency(10);
        assert_eq!(config.e_step_frequency, 10);
    }

    #[test]
    fn test_vpd_config_with_frequency_clamps_to_1() {
        let config = VpdConfig::default().with_frequency(0);
        assert_eq!(config.e_step_frequency, 1);
    }

    #[test]
    fn test_vpd_config_with_fixed_prior() {
        let config = VpdConfig::default().with_fixed_prior();
        assert!(!config.dynamic_prior);
    }

    // ── VpdEmCycle tests ───────────────────────────────────────

    use crate::pruners::absorb_compress::{AbsorbCompressLayer, CompressConfig};
    use crate::pruners::sdar::SdarAbsorbConfig;
    use crate::speculative::types::NoScreeningPruner;

    fn make_absorb(n_actions: usize) -> SdarGatedAbsorbCompress<NoScreeningPruner> {
        let inner =
            AbsorbCompressLayer::new(NoScreeningPruner, n_actions, CompressConfig::default());
        SdarGatedAbsorbCompress::new(inner, n_actions, SdarAbsorbConfig::default())
    }

    #[test]
    fn test_em_cycle_new_initializes_zeros() {
        let cycle: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(VpdConfig::default(), 7);
        assert_eq!(cycle.student_q().len(), 7);
        assert_eq!(cycle.teacher_q().len(), 7);
        assert_eq!(cycle.m_step_count(), 0);
        assert_eq!(cycle.buffer_len(), 0);
        assert!(
            cycle.student_q().iter().all(|&q| q == 0.0),
            "student_q initialized to zeros"
        );
    }

    #[test]
    fn test_em_cycle_should_e_step_frequency() {
        let mut cycle: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(
            VpdConfig::default(), // e_step_frequency = 5
            7,
        );
        let mut absorb = make_absorb(7);

        // M-steps 1-4: should NOT trigger E-step
        for i in 0..4 {
            let should = cycle.m_step(0, 1.0, &mut absorb);
            assert!(!should, "M-step {} should not trigger E-step", i + 1);
        }

        // M-step 5: SHOULD trigger E-step
        let should = cycle.m_step(0, 1.0, &mut absorb);
        assert!(should, "M-step 5 should trigger E-step");
    }

    #[test]
    fn test_em_cycle_collect_sample() {
        let mut cycle: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(VpdConfig::default(), 7);
        cycle.collect_sample(0, 1.0, 0.5);
        cycle.collect_sample(1, 0.0, 0.3);
        assert_eq!(cycle.buffer_len(), 2);
    }

    #[test]
    fn test_em_cycle_e_step_clears_buffer() {
        let mut cycle: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(VpdConfig::default(), 7);
        cycle.collect_sample(0, 1.0, 0.5);
        cycle.collect_sample(1, 0.0, 0.3);
        assert_eq!(cycle.buffer_len(), 2);

        let loss = cycle.e_step();
        assert_eq!(cycle.buffer_len(), 0, "E-step should clear buffer");
        assert!(
            loss >= 0.0,
            "E-step loss should be non-negative, got {loss}"
        );
    }

    #[test]
    fn test_em_cycle_e_step_empty_buffer() {
        let mut cycle: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(VpdConfig::default(), 7);
        let loss = cycle.e_step();
        assert_eq!(loss, 0.0, "E-step on empty buffer returns 0 loss");
    }

    #[test]
    fn test_em_cycle_m_step_updates_student_q() {
        let mut cycle: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(VpdConfig::default(), 7);
        let mut absorb = make_absorb(7);

        // Set teacher Q for action 0 high
        cycle.teacher_q[0] = 5.0;

        // M-step with teacher Q = 5.0 → student Q moves towards 5.0
        let initial_student_q = cycle.student_q()[0];
        cycle.m_step(0, 1.0, &mut absorb);
        let updated_student_q = cycle.student_q()[0];

        assert!(
            updated_student_q > initial_student_q,
            "student_q should increase towards teacher_q: {initial_student_q} → {updated_student_q}"
        );
    }

    #[test]
    fn test_em_cycle_dynamic_prior_vs_fixed() {
        // Dynamic prior uses student_q, fixed uses reference_q
        let config_dynamic = VpdConfig::default();
        let config_fixed = VpdConfig::default().with_fixed_prior();

        let mut cycle_dynamic: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(config_dynamic, 3);
        let mut cycle_fixed: VpdEmCycle<NoScreeningPruner> = VpdEmCycle::new(config_fixed, 3);

        // Set different student and reference Q
        cycle_dynamic.student_q[0] = 2.0;
        cycle_dynamic.reference_q[0] = 0.0;
        cycle_fixed.student_q[0] = 2.0;
        cycle_fixed.reference_q[0] = 0.0;

        // Both have teacher Q = 5.0
        cycle_dynamic.teacher_q[0] = 5.0;
        cycle_fixed.teacher_q[0] = 5.0;

        // Collect samples — implicit reward differs based on prior mode
        cycle_dynamic.collect_sample(0, 1.0, 0.5);
        cycle_fixed.collect_sample(0, 1.0, 0.5);

        // Dynamic: implicit_reward = teacher - student = 5.0 - 2.0 = 3.0
        // Fixed: implicit_reward = teacher - reference = 5.0 - 0.0 = 5.0
        // Both should have samples but with different implicit rewards
        assert_eq!(cycle_dynamic.buffer_len(), 1);
        assert_eq!(cycle_fixed.buffer_len(), 1);
    }
}
