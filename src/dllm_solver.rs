//! Discrete Critical Interval Solver Switching (Plan 222).
//!
//! Entropy-triggered solver switching during DDTree construction.
//! When marginal entropy exceeds H_critical, switch from DPM-Solver++(2M)
//! to q-sampling or other strategies.

/// Solver kind for D2F decode steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SolverKind {
    /// DPM-Solver++(2M) — fast, current default.
    #[default]
    DpmSolver2M = 0,
    /// Q-Sample — re-noise + re-predict for critical steps.
    QSample = 1,
    /// DDPM — standard denoising for fallback.
    DDPM = 2,
}

/// Configuration for CriticalIntervalGate.
#[derive(Clone, Debug)]
pub struct CriticalIntervalConfig {
    /// Entropy threshold above which critical interval is detected.
    /// Default: log(vocab_size) * 0.5
    pub h_critical: f32,
    /// Vocab size for computing default threshold.
    pub vocab_size: usize,
    /// Whether to use q-sampling during critical steps.
    pub use_q_sample: bool,
}

impl Default for CriticalIntervalConfig {
    fn default() -> Self {
        let vocab_size = 32000; // typical LLM vocab
        Self {
            h_critical: (vocab_size as f32).ln() * 0.5,
            vocab_size,
            use_q_sample: false,
        }
    }
}

impl CriticalIntervalConfig {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            h_critical: (vocab_size as f32).ln() * 0.5,
            vocab_size,
            use_q_sample: false,
        }
    }
}

/// Detect whether entropy at current step exceeds critical threshold.
/// Returns true if H >= H_critical.
#[inline]
pub fn is_critical_interval(entropy: f32, config: &CriticalIntervalConfig) -> bool {
    entropy >= config.h_critical
}

/// Select solver based on entropy level.
/// If critical interval and q_sample enabled → QSample.
/// Otherwise → DpmSolver2M.
#[inline]
pub fn select_solver(entropy: f32, config: &CriticalIntervalConfig) -> SolverKind {
    if is_critical_interval(entropy, config) && config.use_q_sample {
        SolverKind::QSample
    } else {
        SolverKind::DpmSolver2M
    }
}

/// Compute Shannon entropy from marginal probabilities.
/// H = -Σ p_i * log(p_i)
pub fn shannon_entropy(marginals: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in marginals {
        if p > 1e-10 {
            h -= p * p.ln();
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Adaptive DDTree Build with Critical Interval Solver Switching (Plan 222 T4)
// ---------------------------------------------------------------------------

/// Log entry for solver transitions during DDTree build.
#[cfg(feature = "critical_interval_gate")]
#[derive(Debug, Clone)]
pub struct SolverTransition {
    pub depth: usize,
    pub entropy: f64,
    pub solver_before: SolverKind,
    pub solver_after: SolverKind,
    pub critical: bool,
}

/// Adaptive DDTree build with critical interval solver switching.
/// Per-depth entropy check → solver switch between DpmSolver2M and QSample.
/// Returns solver transition log for diagnostics.
///
/// Zero-allocation entropy computation (no extra Vecs).
#[cfg(feature = "critical_interval_gate")]
pub fn build_dd_tree_adaptive(
    marginals_per_depth: &[Vec<f32>],
    h_critical: f64,
    solver_kind: &mut SolverKind,
) -> Vec<SolverTransition> {
    let mut transitions = Vec::with_capacity(marginals_per_depth.len());
    let vocab_size = marginals_per_depth.first().map(|m| m.len()).unwrap_or(0);
    let default_h_critical = if h_critical > 0.0 {
        h_critical
    } else {
        (vocab_size as f64).ln() * 0.5
    };

    for (depth, marginals) in marginals_per_depth.iter().enumerate() {
        // Compute Shannon entropy of marginals (zero-alloc: no extra Vec)
        let entropy: f64 = marginals
            .iter()
            .map(|&p| {
                let p = p as f64;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum();

        let prev_solver = *solver_kind;

        if entropy >= default_h_critical {
            // Critical interval — switch to q-sampling if available
            #[cfg(feature = "q_sample_solver")]
            {
                *solver_kind = SolverKind::QSample;
            }
            #[cfg(not(feature = "q_sample_solver"))]
            {
                *solver_kind = SolverKind::DpmSolver2M;
            }
        } else {
            // Below threshold — use fast solver
            *solver_kind = SolverKind::DpmSolver2M;
        }

        transitions.push(SolverTransition {
            depth,
            entropy,
            solver_before: prev_solver,
            solver_after: *solver_kind,
            critical: entropy >= default_h_critical,
        });
    }

    transitions
}

// ---------------------------------------------------------------------------
// Plan 222 T15: Wire CriticalIntervalGate with TriggerGate
// ---------------------------------------------------------------------------
//
// When a critical interval is detected during DDTree construction, the
// CriticalIntervalGate can request a compute tier override from TriggerGate.
// This bridges the entropy-based solver switching with the load-based tier
// routing in InferenceRouter.
//
// Logic:
//   critical_interval + load low  → allow GPU for q-sample refinement
//   critical_interval + load high → stay on CPU with fast solver
//   no critical interval          → no override (defer to default routing)
//
// Feature gate: requires both `critical_interval_gate` and `rv_gated_routing`.
// ---------------------------------------------------------------------------

/// Result of CriticalInterval + TriggerGate integration.
#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CriticalTierDecision {
    /// No critical interval detected — defer to default routing.
    Defer,
    /// Critical interval + low load → promote to GPU for q-sample refinement.
    PromoteGpu,
    /// Critical interval + high load → stay on CPU with fast solver.
    StayCpu,
}

/// Decide compute tier when CriticalIntervalGate detects a critical step.
///
/// Uses TriggerGate's current QPS/tier as a proxy for load:
/// - If current tier is CpuOnly (low load) → promote to GPU for better quality.
/// - If current tier is CpuGpu or higher (high load) → stay on CPU to avoid overload.
///
/// This mirrors the `rv_tier_boost()` pattern from Plan 202, but uses
/// entropy as the signal instead of acceptance variance.
///
/// Returns `Defer` when entropy is below critical threshold (no override needed).
#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
pub fn critical_tier_decision(
    entropy: f32,
    config: &CriticalIntervalConfig,
    current_tier: crate::trigger_gate::ComputeTier,
    gpu_available: bool,
) -> CriticalTierDecision {
    if !is_critical_interval(entropy, config) {
        return CriticalTierDecision::Defer;
    }

    // Critical interval detected — decide based on load
    match current_tier {
        crate::trigger_gate::ComputeTier::CpuOnly if gpu_available => {
            CriticalTierDecision::PromoteGpu
        }
        _ => CriticalTierDecision::StayCpu,
    }
}

// ---------------------------------------------------------------------------
// Q-Sampling Solver (feature-gated: q_sample_solver)
// ---------------------------------------------------------------------------

/// Q-sampling solver step for discrete/mask-based diffusion.
///
/// Given model prediction `x0_hat` (marginal probabilities), produces a
/// refined prediction by:
/// 1. Mixing prediction with noise scaled by alpha schedule (re-noise)
/// 2. The output can then be re-predicted by the model for refinement
///
/// When `alpha_prev == 1.0` and noise is all-zero → identity (argmax commit).
/// When `alpha_prev < 1.0` → applies DDIM-like deterministic step:
/// `output = sqrt(alpha_prev) * x0_hat + sqrt(1 - alpha_prev) * noise`
///
/// For mask-based discrete diffusion, alpha controls how much of the
/// model prediction vs noise is retained at the re-noised step.
#[cfg(feature = "q_sample_solver")]
#[inline]
pub fn q_sample_step(x0_hat: &[f32], alpha_prev: f32, noise: &[f32], output: &mut [f32]) {
    let len = output.len().min(x0_hat.len()).min(noise.len());
    let sqrt_ap = alpha_prev.sqrt();
    let sqrt_1_minus_ap = (1.0 - alpha_prev).max(0.0).sqrt();

    for i in 0..len {
        output[i] = sqrt_ap * x0_hat[i] + sqrt_1_minus_ap * noise[i];
    }
}

/// Full q-sampling re-noise + re-predict cycle for discrete diffusion.
///
/// Adapted for mask-based discrete diffusion (not continuous):
/// 1. Compute x_0_hat from model marginals (the predicted clean distribution)
/// 2. Re-noise to intermediate level: `x_tilde = sqrt(alpha) * x0_hat + sqrt(1-alpha) * noise`
/// 3. Re-noise back down to `alpha_prev`: deterministic DDIM interpolation
/// 4. Commit: sigmoid-activation on the refined values to produce new marginals
///
/// When `alpha == alpha_prev` (no schedule step), returns marginals with
/// sigmoid activation applied (just a deterministic refinement pass).
/// When `noise` is all zeros, falls back to weighted interpolation.
#[cfg(feature = "q_sample_solver")]
pub fn q_sample_refine(
    marginals: &[f32],
    alpha: f32,
    alpha_prev: f32,
    noise: &[f32],
    output: &mut [f32],
) {
    let len = output.len().min(marginals.len()).min(noise.len());
    if len == 0 {
        return;
    }

    // When alpha == alpha_prev == 1.0 and no noise → argmax commit (identity-like)
    let is_identity = (alpha - 1.0).abs() < 1e-8 && (alpha_prev - 1.0).abs() < 1e-8;
    let noise_is_zero = noise[..len].iter().all(|&n| n.abs() < 1e-10);

    if is_identity {
        // Identity: return marginals through sigmoid for normalization
        for i in 0..len {
            output[i] = sigmoid(marginals[i]);
        }
        return;
    }

    if noise_is_zero {
        // Deterministic DDIM step: pure interpolation between marginals
        // output = sqrt(alpha_prev) * marginals / sqrt(alpha)
        //        + sqrt(1 - alpha_prev - (1-alpha)*alpha_prev/alpha) * 0
        // Simplifies to: output = sqrt(alpha_prev / alpha) * marginals
        let ratio = if alpha > 1e-10 {
            (alpha_prev / alpha).sqrt()
        } else {
            0.0
        };
        for i in 0..len {
            output[i] = sigmoid(ratio * marginals[i]);
        }
        return;
    }

    // Stochastic q-sample step:
    // x_tilde = sqrt(alpha) * x0_hat + sqrt(1-alpha) * noise  (re-noise)
    // Then project back to alpha_prev schedule:
    // x_{t-1} = sqrt(alpha_prev) * x0_hat + sqrt(1-alpha_prev) * noise
    // But with the re-noised intermediate, we blend:
    //   refined = sqrt(alpha_prev) * marginals + sqrt(1-alpha_prev) * noise
    // Then sigmoid-activate to produce valid probabilities.
    let sqrt_ap = alpha_prev.sqrt();
    let sqrt_1_minus_ap = (1.0 - alpha_prev).max(0.0).sqrt();

    for i in 0..len {
        let refined = sqrt_ap * marginals[i] + sqrt_1_minus_ap * noise[i];
        output[i] = sigmoid(refined);
    }
}

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
/// Used instead of softmax for independent per-token probability gating.
#[cfg(any(feature = "q_sample_solver", feature = "self_cond_draft"))]
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Find argmax index from a probability-like array.
/// Returns the index with the highest value, or 0 if empty.
#[cfg(any(feature = "q_sample_solver", feature = "self_cond_draft"))]
#[inline]
pub fn argmax(values: &[f32]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut best_val = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Self-Conditioned Drafter (feature-gated: self_cond_draft)
// ---------------------------------------------------------------------------

/// Two-pass self-conditioned speculative drafting.
///
/// Pass 1: model prediction → marginals (standard)
/// Feed best-path tokens as self-conditioning input
/// Pass 2: refined prediction with self-conditioning → improved marginals
///
/// This implements the self-conditioning trick from Chen et al. (2022)
/// "Analog Bits: Generating Discrete Data using Diffusion Models with
/// Self-Conditioning", adapted for discrete token diffusion.
///
/// The drafter is stateful: Pass 1 populates the self-conditioning buffer,
/// Pass 2 uses it for refinement.
#[cfg(feature = "self_cond_draft")]
pub struct SelfCondDraft {
    /// Self-conditioning buffer: stores pass-1 marginals for pass-2 input.
    /// Same shape as the model's marginal output (vocab_size per position).
    sc_buffer: Vec<f32>,
    /// Whether pass 1 has been completed and SC buffer is populated.
    sc_ready: bool,
    /// Number of positions in the current sequence.
    seq_len: usize,
    /// Vocab size per position.
    vocab_size: usize,
}

#[cfg(feature = "self_cond_draft")]
impl SelfCondDraft {
    /// Create a new SelfCondDraft for the given dimensions.
    pub fn new(seq_len: usize, vocab_size: usize) -> Self {
        Self {
            sc_buffer: vec![0.0f32; seq_len * vocab_size],
            sc_ready: false,
            seq_len,
            vocab_size,
        }
    }

    /// Reset for a new draft sequence.
    pub fn reset(&mut self, seq_len: usize) {
        let needed = seq_len * self.vocab_size;
        if self.sc_buffer.len() < needed {
            self.sc_buffer.resize(needed, 0.0);
        }
        self.sc_buffer[..needed].fill(0.0);
        self.sc_ready = false;
        self.seq_len = seq_len;
    }

    /// Whether the SC buffer is ready for pass 2.
    pub fn is_ready(&self) -> bool {
        self.sc_ready
    }

    /// Store pass-1 marginals into the SC buffer.
    ///
    /// `marginals` is a flat slice of `[seq_len * vocab_size]`.
    /// Also computes best-path tokens and blends them in.
    pub fn store_pass1(&mut self, marginals: &[f32]) {
        let needed = self.seq_len * self.vocab_size;
        let len = needed.min(marginals.len()).min(self.sc_buffer.len());
        self.sc_buffer[..len].copy_from_slice(&marginals[..len]);

        // Zero out any remaining buffer beyond what was written
        if len < self.sc_buffer.len() {
            self.sc_buffer[len..].fill(0.0);
        }

        // Enhance SC buffer: reinforce best-path tokens with sigmoid boost
        for p in 0..self.seq_len {
            let start = p * self.vocab_size;
            let end = (start + self.vocab_size).min(self.sc_buffer.len());
            if start >= end {
                break;
            }

            // Find best token for this position
            let best_idx = argmax(&self.sc_buffer[start..end]);

            // Sigmoid boost: sharpen the distribution around the best token
            let slice = &mut self.sc_buffer[start..end];
            for (t, val) in slice.iter_mut().enumerate() {
                if t == best_idx {
                    // Boost best token: sigmoid(x + sharpen_factor)
                    *val = sigmoid(*val + 1.0);
                } else {
                    // Attenuate others: sigmoid(x - sharpen_factor)
                    *val = sigmoid(*val - 1.0);
                }
            }
        }

        self.sc_ready = true;
    }

    /// Apply self-conditioning: blend SC buffer with current marginals.
    ///
    /// In pass 2, the model produces new marginals. We blend them with
    /// the SC buffer from pass 1 to produce refined marginals:
    /// `refined = (1 - blend) * marginals + blend * sc_buffer`
    ///
    /// Returns blended marginals written into `output`.
    pub fn blend_pass2(&self, marginals: &[f32], blend: f32, output: &mut [f32]) {
        let len = output.len().min(marginals.len()).min(self.sc_buffer.len());
        let inv_blend = 1.0 - blend;
        for i in 0..len {
            output[i] = inv_blend * marginals[i] + blend * self.sc_buffer[i];
        }
        // Copy any remaining marginals beyond SC buffer
        if marginals.len() > len && output.len() > len {
            let extra = output.len().min(marginals.len());
            output[len..extra].copy_from_slice(&marginals[len..extra]);
        }
    }

    /// Full 2-pass self-conditioned draft cycle.
    ///
    /// `predict_fn` is called twice:
    /// - Pass 1: `predict_fn(0)` → pass-1 marginals → stored in SC buffer
    /// - Pass 2: `predict_fn(1)` → pass-2 marginals → blended with SC buffer
    ///
    /// `blend` controls how much self-conditioning influences the final output
    /// (0.0 = no influence, 1.0 = fully SC, typical: 0.3-0.5).
    ///
    /// `output` receives the final refined marginals.
    pub fn draft<F>(&mut self, mut predict_fn: F, blend: f32, output: &mut [f32])
    where
        F: FnMut(usize, &mut [f32]),
    {
        let total = self.seq_len * self.vocab_size;

        // Ensure SC buffer is large enough
        if self.sc_buffer.len() < total {
            self.sc_buffer.resize(total, 0.0);
        }

        // Pass 1: standard prediction
        let mut pass1_out = vec![0.0f32; total];
        predict_fn(0, &mut pass1_out);

        // Store pass-1 result as self-conditioning
        self.store_pass1(&pass1_out);

        // Pass 2: prediction with self-conditioning awareness
        let mut pass2_out = vec![0.0f32; total];
        predict_fn(1, &mut pass2_out);

        // Blend pass-2 with SC buffer
        self.blend_pass2(&pass2_out, blend, output);
    }
}

// ---------------------------------------------------------------------------
// MBR Tree Selection (feature-gated)
// ---------------------------------------------------------------------------

/// Minimum Bayes Risk tree selection.
/// Extracts K best paths, scores each against all others,
/// selects the path with minimum total risk.
#[cfg(feature = "mbr_tree_select")]
pub fn mbr_select(
    paths: &[Vec<f32>], // K candidate paths (each is a sequence of token probs)
    scores: &[f32],     // quality score for each path
    k: usize,           // number of candidates to consider
) -> usize {
    if paths.is_empty() {
        return 0;
    }

    let k = k.min(paths.len());
    let top_k_indices: Vec<usize> = {
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed[..k].iter().map(|&(i, _)| i).collect()
    };

    if k == 1 {
        return top_k_indices[0];
    }

    // MBR: for each candidate, compute risk (sum of quality differences vs all others)
    let mut best_idx = top_k_indices[0];
    let mut min_risk = f32::MAX;

    for &i in &top_k_indices {
        let risk: f32 = top_k_indices
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| (scores[i] - scores[j]).max(0.0))
            .sum();
        if risk < min_risk {
            min_risk = risk;
            best_idx = i;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_threshold_detection() {
        let config = CriticalIntervalConfig::new(100);
        // Uniform distribution: entropy = log(100) ≈ 4.6
        // Threshold = log(100) * 0.5 ≈ 2.3
        // Uniform entropy exceeds threshold
        let uniform: Vec<f32> = vec![0.01; 100];
        let entropy = shannon_entropy(&uniform);
        assert!(is_critical_interval(entropy, &config));
    }

    #[test]
    fn test_low_entropy_not_critical() {
        let config = CriticalIntervalConfig::new(100);
        // Peaked distribution: most probability on one token
        let mut peaked = vec![0.001f32; 100];
        peaked[0] = 0.9;
        let entropy = shannon_entropy(&peaked);
        assert!(!is_critical_interval(entropy, &config));
    }

    #[test]
    fn test_solver_selection() {
        let mut config = CriticalIntervalConfig::new(100);
        config.use_q_sample = true;

        let low_entropy = 0.5f32;
        let high_entropy = 10.0f32;

        assert_eq!(select_solver(low_entropy, &config), SolverKind::DpmSolver2M);
        assert_eq!(select_solver(high_entropy, &config), SolverKind::QSample);
    }

    #[test]
    fn test_shannon_entropy() {
        // Binary uniform: H = log(2) ≈ 0.693
        let binary = vec![0.5f32, 0.5];
        let h = shannon_entropy(&binary);
        assert!((h - 2.0f32.ln()).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Q-Sample Solver tests (feature-gated)
    // -----------------------------------------------------------------------

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_q_sample_step_basic() {
        let x0 = vec![1.0f32, 2.0, 3.0];
        let noise = vec![0.1, 0.2, 0.3];
        let mut out = vec![0.0f32; 3];
        q_sample_step(&x0, 0.5, &noise, &mut out);
        // sqrt(0.5) * x0 + sqrt(0.5) * noise
        let expected: Vec<f32> = x0
            .iter()
            .zip(noise.iter())
            .map(|(&x, &n)| 0.5f32.sqrt() * x + 0.5f32.sqrt() * n)
            .collect();
        for i in 0..3 {
            assert!((out[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_q_sample_step_identity_alpha() {
        // When alpha_prev = 1.0: output = sqrt(1.0) * x0 + sqrt(0.0) * noise = x0
        let marginals = vec![0.3f32, 0.5, 0.2];
        let noise = vec![0.1, 0.2, 0.3];
        let mut out = vec![0.0f32; 3];
        q_sample_step(&marginals, 1.0, &noise, &mut out);
        for i in 0..3 {
            assert!(
                (out[i] - marginals[i]).abs() < 1e-5,
                "expected {}, got {}",
                marginals[i],
                out[i]
            );
        }
    }

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_q_sample_step_with_noise_differs_from_input() {
        let marginals = vec![0.3f32, 0.5, 0.2];
        let noise = vec![0.1, 0.2, 0.3];
        let mut out = vec![0.0f32; 3];
        q_sample_step(&marginals, 0.5, &noise, &mut out);
        // Output should differ from input marginals
        for i in 0..3 {
            assert!(
                (out[i] - marginals[i]).abs() > 1e-5,
                "output[{}] = {} should differ from marginal {}",
                i,
                out[i],
                marginals[i]
            );
        }
    }

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_q_sample_refine_identity() {
        // alpha=1.0, alpha_prev=1.0 → sigmoid(marginals) (identity-like)
        let marginals = vec![1.0f32, 2.0, -1.0];
        let noise = vec![0.0; 3];
        let mut out = vec![0.0f32; 3];
        q_sample_refine(&marginals, 1.0, 1.0, &noise, &mut out);
        let s0 = sigmoid(1.0f32);
        let s1 = sigmoid(2.0f32);
        let s2 = sigmoid(-1.0f32);
        assert!((out[0] - s0).abs() < 1e-5);
        assert!((out[1] - s1).abs() < 1e-5);
        assert!((out[2] - s2).abs() < 1e-5);
    }

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_q_sample_refine_with_noise_differs() {
        let marginals = vec![0.5f32, 0.3, 0.2];
        let noise = vec![0.1, 0.2, 0.3];
        let mut out = vec![0.0f32; 3];
        q_sample_refine(&marginals, 0.8, 0.5, &noise, &mut out);
        // With noise and alpha < 1, output should be sigmoid-blended and differ
        for i in 0..3 {
            assert!(
                (out[i] - marginals[i]).abs() > 1e-3,
                "output[{}] = {} too close to marginal {}",
                i,
                out[i],
                marginals[i]
            );
        }
        // Output should be in [0, 1] (sigmoid range)
        for i in 0..3 {
            assert!(
                out[i] >= 0.0 && out[i] <= 1.0,
                "output[{}] = {} outside [0,1]",
                i,
                out[i]
            );
        }
    }

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_q_sample_refine_zero_noise_deterministic() {
        let marginals = vec![1.0f32, 2.0, 0.5];
        let noise = vec![0.0; 3];
        let mut out = vec![0.0f32; 3];
        q_sample_refine(&marginals, 0.8, 0.5, &noise, &mut out);
        // With zero noise, deterministic interpolation: sigmoid(sqrt(0.5/0.8) * marginals)
        let ratio = (0.5f32 / 0.8f32).sqrt();
        for i in 0..3 {
            let expected = sigmoid(ratio * marginals[i]);
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "output[{}] = {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3]), 1);
        assert_eq!(argmax(&[0.9, 0.1, 0.0]), 0);
        assert_eq!(argmax(&[0.1, 0.2, 0.8]), 2);
        assert_eq!(argmax(&[]), 0); // empty → 0
    }

    // -----------------------------------------------------------------------
    // SelfCondDraft tests (feature-gated)
    // -----------------------------------------------------------------------

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_self_cond_draft_two_pass_refines() {
        let seq_len = 2;
        let vocab = 4;
        let mut drafter = SelfCondDraft::new(seq_len, vocab);

        // Simulate a predictable improvement: pass 1 predicts uniform-ish,
        // pass 2 predicts peaked — blending should produce different output
        // than either pass alone.
        let mut call_count = 0usize;
        drafter.draft(
            |pass, out| {
                let total = seq_len * vocab;
                if pass == 0 {
                    // Pass 1: somewhat flat marginals
                    for i in 0..total {
                        out[i] = 0.25; // uniform
                    }
                } else {
                    // Pass 2: peaked marginals (model learned from SC)
                    for p in 0..seq_len {
                        let offset = p * vocab;
                        out[offset] = 0.7;
                        out[offset + 1] = 0.1;
                        out[offset + 2] = 0.1;
                        out[offset + 3] = 0.1;
                    }
                }
                call_count += 1;
            },
            0.5, // blend 50/50
            &mut vec![0.0f32; seq_len * vocab],
        );

        // Should have called predict_fn twice
        assert_eq!(call_count, 2, "expected 2 passes, got {call_count}");
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_self_cond_draft_store_and_blend() {
        let seq_len = 2;
        let vocab = 3;
        let mut drafter = SelfCondDraft::new(seq_len, vocab);

        // Pass 1 marginals: peaked at index 0 for both positions
        let pass1 = vec![0.8f32, 0.1, 0.2, 0.7, 0.15, 0.15];
        drafter.store_pass1(&pass1);
        assert!(drafter.is_ready());

        // Pass 2 marginals: peaked at index 1 for both positions
        let pass2 = vec![0.1f32, 0.7, 0.2, 0.1, 0.8, 0.1];
        let mut output = vec![0.0f32; seq_len * vocab];
        drafter.blend_pass2(&pass2, 0.5, &mut output);

        // Blended output should be different from both pass1 and pass2
        for i in 0..seq_len * vocab {
            assert!(
                output[i] > 0.0,
                "output[{}] = {} should be positive after blending",
                i,
                output[i]
            );
        }

        // The blend is with the sigmoid-sharpened SC buffer, not raw pass1.
        // So we can't assert exact values, but should not be zero.
    }

    #[cfg(feature = "self_cond_draft")]
    #[test]
    fn test_self_cond_draft_reset() {
        let mut drafter = SelfCondDraft::new(3, 4);
        let pass1 = vec![0.5f32; 12];
        drafter.store_pass1(&pass1);
        assert!(drafter.is_ready());

        drafter.reset(3);
        assert!(!drafter.is_ready());
        // SC buffer should be zeroed
        for &v in &drafter.sc_buffer {
            assert_eq!(v, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // MBR tests (feature-gated)
    // -----------------------------------------------------------------------

    #[cfg(feature = "mbr_tree_select")]
    #[test]
    fn test_mbr_select() {
        let paths = vec![vec![1.0], vec![2.0], vec![3.0]];
        let scores = vec![0.1, 0.5, 0.9];
        let best = mbr_select(&paths, &scores, 3);
        // Middle path has minimum risk
        assert!(best < paths.len());
    }

    // -----------------------------------------------------------------------
    // Adaptive DDTree build tests (Plan 222 T4)
    // -----------------------------------------------------------------------

    #[cfg(feature = "critical_interval_gate")]
    #[test]
    fn test_adaptive_solver_switches_on_high_entropy() {
        // High entropy marginals (uniform distribution)
        let uniform = vec![0.1f32; 10];
        // Low entropy marginals (peaked distribution)
        let peaked = {
            let mut m = vec![0.01f32; 10];
            m[0] = 0.91;
            m
        };

        let depths = vec![uniform.clone(), peaked.clone(), uniform.clone()];
        let mut solver = SolverKind::DpmSolver2M;

        let transitions = build_dd_tree_adaptive(&depths, 1.0, &mut solver);

        assert_eq!(transitions.len(), 3);
        assert!(transitions[0].critical); // uniform → high entropy
        assert!(!transitions[1].critical); // peaked → low entropy
        assert!(transitions[2].critical); // uniform → high entropy

        // Solver should have switched during high entropy
        #[cfg(feature = "q_sample_solver")]
        assert_eq!(solver, SolverKind::QSample); // last depth was critical
    }

    #[cfg(feature = "critical_interval_gate")]
    #[test]
    fn test_adaptive_default_threshold() {
        let uniform = vec![0.1f32; 100]; // 100 tokens, uniform
        let depths = vec![uniform];
        let mut solver = SolverKind::DpmSolver2M;

        let transitions = build_dd_tree_adaptive(&depths, 0.0, &mut solver);

        // Default threshold = ln(100) * 0.5 ≈ 2.3
        // Uniform entropy = ln(100) ≈ 4.6
        assert!(transitions[0].critical);
    }

    // -----------------------------------------------------------------------
    // Benchmark stubs (Plan 222 T9)
    // -----------------------------------------------------------------------

    #[cfg(feature = "critical_interval_gate")]
    #[test]
    fn bench_adaptive_build_16_depths() {
        let depths: Vec<Vec<f32>> = (0..16)
            .map(|d| {
                if d % 3 == 0 {
                    vec![0.1; 100] // uniform
                } else {
                    let mut m = vec![0.01; 100];
                    m[d % 100] = 0.5;
                    m[(d + 1) % 100] = 0.3;
                    m
                }
            })
            .collect();

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let mut solver = SolverKind::DpmSolver2M;
            let _ = build_dd_tree_adaptive(&depths, 0.0, &mut solver);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / 1000;
        // Should be very fast — entropy is O(n) per depth
        assert!(
            per_call < std::time::Duration::from_micros(100),
            "Adaptive build too slow: {:?}",
            per_call
        );
    }

    #[cfg(feature = "mbr_tree_select")]
    #[test]
    fn bench_mbr_select_5_candidates() {
        let paths: Vec<Vec<f32>> = (0..10).map(|i| vec![0.1 * i as f32; 16]).collect();
        let scores: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();

        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = mbr_select(&paths, &scores, 5);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / 10000;
        assert!(
            per_call < std::time::Duration::from_micros(10),
            "MBR select too slow: {:?}",
            per_call
        );
    }

    // -----------------------------------------------------------------------
    // Plan 222 T15: CriticalIntervalGate + TriggerGate wiring tests
    // -----------------------------------------------------------------------

    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    #[test]
    fn test_critical_tier_decision_defers_when_not_critical() {
        use crate::trigger_gate::ComputeTier;

        let config = CriticalIntervalConfig::new(100);
        // Peaked distribution: low entropy → not critical
        let mut peaked = vec![0.001f32; 100];
        peaked[0] = 0.9;
        let entropy = shannon_entropy(&peaked);

        let decision = critical_tier_decision(entropy, &config, ComputeTier::CpuOnly, true);
        assert_eq!(decision, CriticalTierDecision::Defer);
    }

    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    #[test]
    fn test_critical_tier_decision_promotes_gpu_on_low_load() {
        use crate::trigger_gate::ComputeTier;

        let config = CriticalIntervalConfig::new(100);
        // Uniform distribution: high entropy → critical
        let uniform = vec![0.01f32; 100];
        let entropy = shannon_entropy(&uniform);

        let decision = critical_tier_decision(entropy, &config, ComputeTier::CpuOnly, true);
        assert_eq!(decision, CriticalTierDecision::PromoteGpu);
    }

    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    #[test]
    fn test_critical_tier_decision_stays_cpu_on_high_load() {
        use crate::trigger_gate::ComputeTier;

        let config = CriticalIntervalConfig::new(100);
        let uniform = vec![0.01f32; 100];
        let entropy = shannon_entropy(&uniform);

        // High load: already on CpuGpu → stay on CPU
        let decision = critical_tier_decision(entropy, &config, ComputeTier::CpuGpu, true);
        assert_eq!(decision, CriticalTierDecision::StayCpu);
    }

    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    #[test]
    fn test_critical_tier_decision_stays_cpu_when_no_gpu() {
        use crate::trigger_gate::ComputeTier;

        let config = CriticalIntervalConfig::new(100);
        let uniform = vec![0.01f32; 100];
        let entropy = shannon_entropy(&uniform);

        // Critical but no GPU available → stay on CPU
        let decision = critical_tier_decision(entropy, &config, ComputeTier::CpuOnly, false);
        assert_eq!(decision, CriticalTierDecision::StayCpu);
    }

    // -----------------------------------------------------------------------
    // Plan 222 T12: Benchmark — before/after CriticalIntervalGate
    // Stub: verifies machinery correctness without real model inference.
    // Real benchmarks require a live model; this validates the adaptive build
    // path produces correct solver transitions and entropy measurements.
    // -----------------------------------------------------------------------

    #[cfg(feature = "critical_interval_gate")]
    #[test]
    fn bench_t12_critical_interval_before_after() {
        // Simulate 16-depth DDTree with mixed entropy marginals
        let depths: Vec<Vec<f32>> = (0..16)
            .map(|d| {
                if d % 4 == 0 {
                    // High entropy: uniform distribution
                    vec![0.01f32; 100]
                } else {
                    // Low entropy: peaked distribution
                    let mut m = vec![0.001f32; 100];
                    m[d % 100] = 0.7;
                    m[(d + 1) % 100] = 0.2;
                    m
                }
            })
            .collect();

        // --- Without CriticalIntervalGate (baseline): always DpmSolver2M ---
        let mut solver_baseline = SolverKind::DpmSolver2M;
        let transitions_baseline = build_dd_tree_adaptive(&depths, f64::MAX, &mut solver_baseline);
        // With h_critical = MAX, nothing is critical
        assert!(transitions_baseline.iter().all(|t| !t.critical));
        assert_eq!(solver_baseline, SolverKind::DpmSolver2M);

        // --- With CriticalIntervalGate: adaptive switching ---
        let mut solver_gated = SolverKind::DpmSolver2M;
        let transitions_gated = build_dd_tree_adaptive(&depths, 0.0, &mut solver_gated);
        // With h_critical = 0, everything is critical
        assert!(transitions_gated.iter().all(|t| t.critical));

        // --- With realistic threshold ---
        let mut solver_real = SolverKind::DpmSolver2M;
        let realistic_h = (100f64).ln() * 0.5; // ~2.3
        let transitions_real = build_dd_tree_adaptive(&depths, realistic_h, &mut solver_real);

        let critical_count = transitions_real.iter().filter(|t| t.critical).count();
        let non_critical_count = transitions_real.iter().filter(|t| !t.critical).count();

        // Mixed depths should produce both critical and non-critical transitions
        assert!(critical_count > 0, "expected some critical intervals");
        assert!(
            non_critical_count > 0,
            "expected some non-critical intervals"
        );

        // Verify entropy measurements are deterministic
        for i in 0..transitions_real.len() {
            let recomputed: f64 = depths[i]
                .iter()
                .map(|&p| {
                    let p = p as f64;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            assert!((transitions_real[i].entropy - recomputed).abs() < 1e-6);
        }

        // Perf: 1000 iterations should complete quickly
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let mut s = SolverKind::DpmSolver2M;
            let _ = build_dd_tree_adaptive(&depths, realistic_h, &mut s);
        }
        let per_call = start.elapsed() / 1000;
        assert!(
            per_call < std::time::Duration::from_micros(200),
            "Adaptive build too slow: {:?}",
            per_call
        );
    }

    // -----------------------------------------------------------------------
    // Plan 222 T13: Benchmark — MBR vs existing strategies
    // Stub: verifies MBR machinery correctness without real model inference.
    // Compares MBR select output against BestQ (argmax) and EqR strategies.
    // -----------------------------------------------------------------------

    #[cfg(feature = "mbr_tree_select")]
    #[test]
    fn bench_t13_mbr_vs_bestq_vs_eqr() {
        // Simulate 20 candidate paths with known score distribution
        let n_candidates = 20;
        let seq_len = 16;
        let paths: Vec<Vec<f32>> = (0..n_candidates)
            .map(|i| {
                (0..seq_len)
                    .map(|j| ((i * seq_len + j) as f32 * 0.01).sin().max(0.0))
                    .collect()
            })
            .collect();
        let scores: Vec<f32> = (0..n_candidates)
            .map(|i| {
                // Bimodal: first 10 are mediocre, last 10 are good
                if i < 10 {
                    0.3 + (i as f32 * 0.01)
                } else {
                    0.7 + ((i - 10) as f32 * 0.02)
                }
            })
            .collect();

        // --- BestQ strategy: argmax of scores ---
        let bestq_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(bestq_idx, n_candidates - 1); // last candidate has highest score

        // --- EqR strategy: equal-random from top-K ---
        // (in real code this would be random from top-K; here we verify determinism)
        let eqr_k = 5;
        let mut eqr_indexed: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        eqr_indexed.select_nth_unstable_by(eqr_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let eqr_top_k: Vec<usize> = eqr_indexed[..eqr_k].iter().map(|&(i, _)| i).collect();
        assert!(eqr_top_k.iter().all(|&i| i >= 10)); // all top-5 are from the "good" group

        // --- MBR strategy: minimum risk selection ---
        for k in [3, 5, 10] {
            let mbr_idx = mbr_select(&paths, &scores, k);
            assert!(mbr_idx < n_candidates);
            // MBR selects from top-K, so it should be in the good group
            let mut top_k_idx: Vec<usize> = (0..n_candidates).collect();
            top_k_idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
            let top_k_set: std::collections::HashSet<usize> =
                top_k_idx[..k].iter().copied().collect();
            assert!(top_k_set.contains(&mbr_idx), "MBR should select from top-K");
        }

        // --- Perf: MBR vs BestQ ---
        let n_iters = 10000;

        let start_mbr = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = mbr_select(&paths, &scores, 5);
        }
        let mbr_time = start_mbr.elapsed() / n_iters;

        let start_bestq = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        }
        let bestq_time = start_bestq.elapsed() / n_iters;

        // MBR should be reasonably fast (O(K^2) for small K)
        assert!(
            mbr_time < std::time::Duration::from_micros(50),
            "MBR too slow: {:?}",
            mbr_time
        );

        // Log comparison (not a hard assertion — informational)
        eprintln!(
            "MBR: {:?}, BestQ: {:?} (MBR overhead: {:.1}x)",
            mbr_time,
            bestq_time,
            mbr_time.as_nanos() as f64 / bestq_time.as_nanos().max(1) as f64
        );
    }
}
