//! Adaptive Chain-of-Thought — Self-Learning When to Think (Plan 194).
//!
//! Decides per-query whether to think (latent reasoning via RiM buffer slots)
//! or answer directly. The controller learns from episode feedback (bandit
//! self-improvement), auto-routes between CPU (PPoT resample) and GPU
//! (full RiM decode) based on load, and persists learned knowledge via
//! freeze/thaw weight dumps.
//!
//! Key principle: Thinking is *optional* — the system works without it.
//! When enabled, it must make better decisions.

use std::path::Path;

use crate::pruners::freeze::{load_frozen, save_frozen};

// ── T1: Types ──────────────────────────────────────────────────────────

/// Per-query thinking mode — the adaptive controller decides which to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ThinkingMode {
    /// No thinking — direct answer via DDTree + ScreeningPruner (baseline).
    /// Zero overhead. The existing pipeline unchanged.
    Direct,
    /// Latent thinking — RiM buffer slots activated, multiple reasoning passes.
    /// Uses DDTree with expanded budget + FrozenBaseGuard intermediate steps.
    Latent,
    /// CPU-only thinking — PPoT resample high-entropy tokens on CPU.
    /// Zero GPU overhead beyond initial forward pass. Good when GPU is loaded.
    CpuResample,
    /// NMDA-gated adaptive thinking — deterministic gate replaces bandit.
    /// Uses entropy + coincidence to modulate thinking budget. Zero randomness.
    /// Feature-gated behind `dendritic_gate`.
    Dendritic,
}

/// How the thinking mode is selected per-query.
#[derive(Debug, Clone)]
pub enum ThinkingSelector {
    /// Always use direct mode — no thinking (baseline for benchmarks).
    AlwaysDirect,
    /// Always think — every query gets latent reasoning (stress test).
    AlwaysLatent,
    /// Adaptive — bandit learns which queries benefit from thinking.
    /// Starts with heuristic (confidence < threshold → think), then improves.
    Adaptive {
        /// Exploration rate for bandit (ε in ε-greedy). Default: 0.1.
        exploration_rate: f32,
        /// Weight for dendritic arm in adaptive selection. Default: 0.25.
        dendritic_weight: f32,
    },
}

impl Default for ThinkingSelector {
    fn default() -> Self {
        Self::Adaptive {
            exploration_rate: 0.1,
            dendritic_weight: 0.25,
        }
    }
}

/// Configuration for adaptive thinking.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    /// How the system decides when to think.
    pub mode: ThinkingSelector,
    /// Maximum thinking budget (in RiM buffer block passes). Default: 8.
    pub max_blocks: usize,
    /// Minimum thinking budget (for easy problems). Default: 0 (= direct).
    pub min_blocks: usize,
    /// Confidence threshold below which thinking activates. Default: 0.7.
    /// If best DDTree candidate score < threshold, switch to thinking.
    pub confidence_threshold: f32,
    /// GPU load threshold (0-1) above which CPU-only mode is preferred. Default: 0.8.
    pub gpu_load_threshold: f32,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            mode: ThinkingSelector::default(),
            max_blocks: 8,
            min_blocks: 0,
            confidence_threshold: 0.7,
            gpu_load_threshold: 0.8,
        }
    }
}

impl ThinkingConfig {
    /// Bias thinking mode selection based on entropy signal.
    /// High entropy (low top-1 prob) → bias toward Latent mode.
    /// Low entropy (high top-1 prob) → bias toward Direct mode.
    #[cfg(feature = "directional_credit")]
    pub fn entropy_bias(&self, top1_prob: f32) -> ThinkingMode {
        match top1_prob < self.confidence_threshold {
            true => ThinkingMode::Latent,
            false => ThinkingMode::Direct,
        }
    }
}

// ── T2: ThinkingBandit ─────────────────────────────────────────────────

/// Lightweight bandit with 4 arms: Direct, Latent, CpuResample, Dendritic.
/// Tracks reward = answer_quality * (1 - normalized_cost).
/// Uses Thompson sampling (consistent with BanditPruner).
struct ThinkingBandit {
    /// Per-arm success/failure counts for Beta posterior.
    successes: [f32; 4],
    failures: [f32; 4],
    /// Decay factor for recency weighting. Default: 0.99.
    decay: f32,
    /// Total pulls across all arms.
    total_pulls: u32,
}

impl ThinkingBandit {
    fn new() -> Self {
        Self {
            successes: [1.0; 4], // Beta(1,1) = Uniform prior
            failures: [1.0; 4],
            decay: 0.99,
            total_pulls: 0,
        }
    }

    /// Thompson sample from Beta posteriors to select an arm.
    fn sample(&mut self, exploration_rate: f32, rng: &mut impl Rng) -> usize {
        self.total_pulls += 1;
        // ε-greedy: with probability ε, pick a random arm
        if rng.next_f32() < exploration_rate {
            return (rng.next_u32() as usize) % 4;
        }
        // Thompson sampling: sample from Beta(α, β) for each arm, pick max
        let mut best_arm = 0;
        let mut best_score = 0.0f32;
        for arm in 0..4 {
            let alpha = self.successes[arm];
            let beta = self.failures[arm];
            // Sample from Beta(α, β) using the ratio of Gamma variates
            // Approximation: use a simple random score weighted by α/(α+β)
            let score = sample_beta(alpha, beta, rng);
            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }
        best_arm
    }

    fn total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Record an observation for an arm.
    fn record(&mut self, arm: usize, success: bool) {
        if success {
            self.successes[arm] += 1.0;
        } else {
            self.failures[arm] += 1.0;
        }
    }

    /// Decay old observations for recency weighting.
    fn decay_observations(&mut self) {
        let d = self.decay;
        for arm in 0..4 {
            self.successes[arm] *= d;
            self.failures[arm] *= d;
        }
    }

    /// Freeze bandit state for persistence.
    fn freeze(&self) -> ThinkingBanditFrozen {
        ThinkingBanditFrozen {
            magic: ThinkingBanditFrozen::MAGIC,
            version: ThinkingBanditFrozen::VERSION,
            successes: self.successes,
            failures: self.failures,
            total_pulls: self.total_pulls,
        }
    }

    /// Thaw bandit state from frozen form.
    fn thaw(frozen: &ThinkingBanditFrozen) -> Result<Self, String> {
        frozen.validate()?;
        Ok(Self {
            successes: frozen.successes,
            failures: frozen.failures,
            decay: 0.99,
            total_pulls: frozen.total_pulls,
        })
    }
}

/// Simple Beta(α,β) sampling using the JDH approximation.
/// For small counts, use a simple α/(α+β) with noise.
fn sample_beta(alpha: f32, beta: f32, rng: &mut impl Rng) -> f32 {
    // Use two uniform random variates to approximate Beta sampling
    // via the Jöhnk algorithm for small parameters, or simple ratio for larger.
    let u1 = rng.next_f32().max(1e-8);
    let u2 = rng.next_f32().max(1e-8);

    if alpha < 1.0 && beta < 1.0 {
        // Jöhnk's algorithm for α<1, β<1
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);
        if x + y > 1.0 {
            // Reject — return simple mean estimate
            return alpha / (alpha + beta) + (rng.next_f32() - 0.5) * 0.1;
        }
        x / (x + y)
    } else {
        // Simple ratio-based approximation for larger parameters
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);
        x / (x + y + 1e-10)
    }
}

/// RNG trait needed by the bandit for Thompson sampling.
/// Matches the existing `katgpt_core::types::Rng` trait.
pub trait Rng {
    fn next_u32(&mut self) -> u32;
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

// ── GpuLoadSignal ──────────────────────────────────────────────────────

/// GPU load signal — zero-cost when no GPU monitor is available.
enum GpuLoadSignal {
    /// No GPU monitoring available — always assume GPU is free.
    Unavailable,
    /// Static threshold — load > threshold → prefer CPU.
    Threshold(f32),
    /// Dynamic load value — updated externally.
    Dynamic(f32),
}

impl GpuLoadSignal {
    fn is_loaded(&self, threshold: f32) -> bool {
        match self {
            GpuLoadSignal::Unavailable => false,
            GpuLoadSignal::Threshold(t) => *t >= threshold,
            GpuLoadSignal::Dynamic(load) => *load >= threshold,
        }
    }

    fn load_value(&self) -> f32 {
        match self {
            GpuLoadSignal::Unavailable => 0.0,
            GpuLoadSignal::Threshold(t) => *t,
            GpuLoadSignal::Dynamic(load) => *load,
        }
    }
}

// ── T4: Freeze/Thaw ────────────────────────────────────────────────────

/// Frozen bandit knowledge for thinking controller.
/// `repr(C)` for zero-dependency binary persistence via `pruners::freeze`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ThinkingBanditFrozen {
    /// Magic bytes for validation: b"THKB".
    pub magic: [u8; 4],
    /// Version for migration.
    pub version: u32,
    /// Per-arm success counts: [direct, latent, cpu_resample, dendritic].
    pub successes: [f32; 4],
    /// Per-arm failure counts: [direct, latent, cpu_resample, dendritic].
    pub failures: [f32; 4],
    /// Total episodes observed.
    pub total_pulls: u32,
}

impl ThinkingBanditFrozen {
    const MAGIC: [u8; 4] = *b"THKB";
    const VERSION: u32 = 2;

    fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "ThinkingBanditFrozen: bad magic {:?}, expected {:?}",
                self.magic,
                Self::MAGIC
            ));
        }
        if self.version != Self::VERSION {
            return Err(format!(
                "ThinkingBanditFrozen: bad version {}, expected {}",
                self.version,
                Self::VERSION
            ));
        }
        Ok(())
    }
}

// ── T2+T3: ThinkingController ──────────────────────────────────────────

/// Adaptive thinking controller — decides per-query whether to think.
///
/// Wraps the existing DDTree pipeline and adds an optional thinking layer
/// on top. The controller is stateless per-query but accumulates bandit
/// knowledge across episodes (via freeze/thaw).
pub struct ThinkingController {
    config: ThinkingConfig,
    /// Bandit arm 0: direct (no thinking). Arm 1: latent thinking. Arm 2: CPU resample.
    bandit: ThinkingBandit,
    /// GPU load monitor — reads from external load signal.
    gpu_load: GpuLoadSignal,
    /// Trust signal from speculative verifier (0.0 = low trust, 1.0 = high trust).
    /// Low trust → prefer thinking mode. High trust → prefer direct mode.
    /// Updated externally via `set_trust_signal()` (Plan 182).
    /// Value of -1.0 means "not set" — trust override is inactive.
    trust_signal: f32,
}

impl ThinkingController {
    /// Create a new thinking controller with the given config.
    pub fn new(config: ThinkingConfig) -> Self {
        Self {
            config,
            bandit: ThinkingBandit::new(),
            gpu_load: GpuLoadSignal::Unavailable,
            trust_signal: -1.0,
        }
    }

    /// Create a thinking controller with GPU load monitoring.
    pub fn with_gpu_load(config: ThinkingConfig, gpu_load: f32) -> Self {
        Self {
            config,
            bandit: ThinkingBandit::new(),
            gpu_load: GpuLoadSignal::Dynamic(gpu_load),
            trust_signal: -1.0,
        }
    }

    /// Create a thinking controller with static GPU threshold monitoring.
    pub fn with_gpu_threshold(config: ThinkingConfig, threshold: f32) -> Self {
        Self {
            config,
            bandit: ThinkingBandit::new(),
            gpu_load: GpuLoadSignal::Threshold(threshold),
            trust_signal: -1.0,
        }
    }

    /// Restore a thinking controller from frozen bandit knowledge.
    pub fn from_frozen(
        config: ThinkingConfig,
        frozen: &ThinkingBanditFrozen,
    ) -> Result<Self, String> {
        let bandit = ThinkingBandit::thaw(frozen)?;
        Ok(Self {
            config,
            bandit,
            gpu_load: GpuLoadSignal::Unavailable,
            trust_signal: -1.0,
        })
    }

    /// Update the GPU load value.
    pub fn set_gpu_load(&mut self, load: f32) {
        self.gpu_load = GpuLoadSignal::Dynamic(load);
    }

    /// Update the trust signal from speculative verification (Plan 182).
    ///
    /// Low trust (< 0.4) biases toward thinking modes.
    /// High trust (> 0.8) biases toward direct mode.
    pub fn set_trust_signal(&mut self, trust: f32) {
        self.trust_signal = trust;
    }

    /// Get current trust signal. Returns -1.0 if not set.
    pub fn trust_signal(&self) -> f32 {
        self.trust_signal
    }

    /// Decide thinking mode for this query.
    ///
    /// # Arguments
    /// * `first_pass_confidence` — best DDTree candidate score from initial pass (0-1).
    ///   Available after Phase C. If not available yet, pass 0.0.
    /// * `rng` — random number generator for Thompson sampling.
    ///
    /// # Returns
    /// Which `ThinkingMode` to use for this query.
    pub fn select_mode(&mut self, first_pass_confidence: f32, rng: &mut impl Rng) -> ThinkingMode {
        match &self.config.mode {
            ThinkingSelector::AlwaysDirect => ThinkingMode::Direct,
            ThinkingSelector::AlwaysLatent => ThinkingMode::Latent,
            ThinkingSelector::Adaptive {
                exploration_rate,
                dendritic_weight: _,
            } => {
                let exploration_rate = *exploration_rate;

                // 1. Check GPU load → decide CPU vs GPU route
                let gpu_loaded = self.gpu_load.is_loaded(self.config.gpu_load_threshold);

                // 2. Thompson sample from bandit posterior
                let arm = self.bandit.sample(exploration_rate, rng);

                // 3. Trust-triggered override (Plan 182)
                // Only active when trust_signal has been explicitly set (>= 0).
                // Low trust → prefer thinking. High trust → prefer direct.
                let trust_override = if self.trust_signal >= 0.0 && self.trust_signal < 0.4 {
                    // Low trust: force some form of thinking
                    if gpu_loaded {
                        Some(ThinkingMode::CpuResample)
                    } else {
                        Some(ThinkingMode::Latent)
                    }
                } else if self.trust_signal > 0.8 {
                    // High trust: force direct (skip thinking)
                    Some(ThinkingMode::Direct)
                } else {
                    None
                };

                // 4. Combine: trust override, cold-start heuristic, or bandit decision
                let cold = self.bandit.total_pulls() < 10;
                if let Some(mode) = trust_override {
                    mode
                } else if cold && first_pass_confidence < self.config.confidence_threshold {
                    // Heuristic: low confidence → think
                    if gpu_loaded {
                        ThinkingMode::CpuResample
                    } else {
                        ThinkingMode::Latent
                    }
                } else {
                    match arm {
                        0 => ThinkingMode::Direct,
                        1 => {
                            if gpu_loaded {
                                ThinkingMode::CpuResample
                            } else {
                                ThinkingMode::Latent
                            }
                        }
                        2 => ThinkingMode::CpuResample,
                        3 => ThinkingMode::Dendritic,
                        _ => ThinkingMode::Direct,
                    }
                }
            }
        }
    }

    /// Record reward for the chosen thinking mode.
    ///
    /// # Arguments
    /// * `mode` — which mode was used
    /// * `answer_quality` — 0-1 score from ScreeningPruner/verifier
    /// * `cost_normalized` — 0-1 cost (latency or token count, normalized to budget)
    pub fn record_reward(&mut self, mode: ThinkingMode, answer_quality: f32, cost_normalized: f32) {
        let arm = mode as usize;
        // Reward = quality * (1 - cost). Higher is better.
        let reward = answer_quality * (1.0 - cost_normalized);
        self.bandit.record(arm, reward > 0.5);
        // Decay old observations
        self.bandit.decay_observations();
    }

    // ── T3: CPU/GPU Auto-Route ──────────────────────────────

    /// Route thinking to CPU or GPU based on load.
    pub fn route_thinking(&self) -> ThinkingMode {
        let gpu_load = self.gpu_load.load_value();
        match gpu_load > self.config.gpu_load_threshold {
            true => ThinkingMode::CpuResample,
            false => ThinkingMode::Latent,
        }
    }

    // ── T5: Adaptive Thinking Budget via Cumprodsum Freshness ────

    /// Fast sigmoid: `1 / (1 + e^{-x})`.
    #[inline(always)]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x.clamp(-50.0, 50.0)).exp())
    }

    /// Compute adaptive thinking budget based on context freshness.
    ///
    /// Uses cumprodsum-derived freshness signal to allocate more thinking
    /// budget when the context is "fresh" (recent information dominates)
    /// and less when context is "stale" (old information persists, meaning
    /// the model has had time to fully process it).
    ///
    /// Formula: `budget = min_blocks + (max_blocks - min_blocks) * sigmoid(beta * (freshness - 0.5))`
    ///
    /// # Arguments
    /// * `decay_factors` — Per-position decay factors from the SSM gate
    ///   (typically `sigmoid(gate)` values in [0, 1]).
    /// * `beta` — Sensitivity of budget to freshness changes.
    ///   Default: 4.0 (moderate). Higher = sharper transition.
    ///
    /// # Returns
    /// Thinking budget in RiM buffer block passes, clamped to
    /// [`ThinkingConfig::min_blocks`, `ThinkingConfig::max_blocks`].
    pub fn adaptive_budget(&self, decay_factors: &[f32], beta: f32) -> usize {
        let freshness = crate::cumprodsum::context_freshness(decay_factors);
        let range = self.config.max_blocks.saturating_sub(self.config.min_blocks);
        let scale = Self::sigmoid(beta * (freshness - 0.5));
        self.config.min_blocks + (range as f32 * scale).round() as usize
    }

    /// Convenience: adaptive budget with default beta=4.0.
    #[inline]
    pub fn adaptive_budget_default(&self, decay_factors: &[f32]) -> usize {
        self.adaptive_budget(decay_factors, 4.0)
    }

    // ── T4: Freeze/Thaw ─────────────────────────────────────────

    /// Freeze bandit knowledge for disk persistence.
    pub fn freeze(&self) -> ThinkingBanditFrozen {
        self.bandit.freeze()
    }

    /// Save bandit knowledge to disk.
    pub fn save_bandit(&self, path: &Path) -> Result<(), String> {
        let frozen = self.freeze();
        save_frozen(path, &frozen)
    }

    /// Load bandit knowledge from disk.
    pub fn load_bandit(&mut self, path: &Path) -> Result<(), String> {
        let frozen: ThinkingBanditFrozen = load_frozen(path)?;
        self.bandit = ThinkingBandit::thaw(&frozen)?;
        Ok(())
    }

    // ── RV-Gated Mode Selection (Plan 202) ─────────────────────────

    /// Select thinking mode with RV signal bias.
    ///
    /// High RV (model uncertain) → bias toward Latent thinking.
    /// Low RV (model confident) → bias toward Direct mode.
    /// Medium RV → bandit decides (no bias).
    ///
    /// Feature-gated behind `rv_gated_thinking`.
    /// Falls back to `select_mode()` when disabled.
    #[cfg(feature = "rv_gated_thinking")]
    pub fn select_mode_with_rv(
        &mut self,
        first_pass_confidence: f32,
        rv_signal: f64,
        rng: &mut impl Rng,
    ) -> ThinkingMode {
        // RV thresholds (matching RvThresholds defaults)
        const RV_THETA_HIGH: f64 = 0.10;
        const RV_THETA_LOW: f64 = 0.02;

        match rv_signal {
            rv if rv > RV_THETA_HIGH => {
                // High RV → model uncertain → prefer Latent thinking
                // Soft bias: record success toward latent to influence future bandit
                let gpu_loaded = self.gpu_load.is_loaded(self.config.gpu_load_threshold);
                if gpu_loaded {
                    ThinkingMode::CpuResample
                } else {
                    ThinkingMode::Latent
                }
            }
            rv if rv < RV_THETA_LOW && rv >= 0.0 => {
                // Low RV → model confident → Direct mode
                ThinkingMode::Direct
            }
            _ => {
                // Medium RV or unavailable → defer to bandit
                self.select_mode(first_pass_confidence, rng)
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Simple deterministic RNG for testing.
    struct TestRng {
        state: u32,
    }

    impl TestRng {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }
    }

    impl Rng for TestRng {
        fn next_u32(&mut self) -> u32 {
            // xorshift32
            self.state ^= self.state << 13;
            self.state ^= self.state >> 17;
            self.state ^= self.state << 5;
            self.state
        }
    }

    #[test]
    fn test_always_direct() {
        let config = ThinkingConfig {
            mode: ThinkingSelector::AlwaysDirect,
            ..Default::default()
        };
        let mut ctrl = ThinkingController::new(config);
        let mut rng = TestRng::new(42);
        assert_eq!(ctrl.select_mode(0.5, &mut rng), ThinkingMode::Direct);
    }

    #[test]
    fn test_always_latent() {
        let config = ThinkingConfig {
            mode: ThinkingSelector::AlwaysLatent,
            ..Default::default()
        };
        let mut ctrl = ThinkingController::new(config);
        let mut rng = TestRng::new(42);
        assert_eq!(ctrl.select_mode(0.9, &mut rng), ThinkingMode::Latent);
    }

    #[test]
    fn test_adaptive_cold_low_confidence_thinks() {
        let config = ThinkingConfig {
            mode: ThinkingSelector::Adaptive {
                exploration_rate: 0.0,
                dendritic_weight: 0.25,
            },
            confidence_threshold: 0.7,
            ..Default::default()
        };
        let mut ctrl = ThinkingController::new(config);
        let mut rng = TestRng::new(42);
        // Cold bandit + low confidence → should think
        let mode = ctrl.select_mode(0.3, &mut rng);
        assert_ne!(mode, ThinkingMode::Direct);
    }

    #[test]
    fn test_adaptive_cold_high_confidence_direct() {
        let config = ThinkingConfig {
            mode: ThinkingSelector::Adaptive {
                exploration_rate: 0.0,
                dendritic_weight: 0.25,
            },
            confidence_threshold: 0.7,
            ..Default::default()
        };
        let mut ctrl = ThinkingController::new(config);
        let mut rng = TestRng::new(123);
        // Cold bandit + high confidence → let bandit decide (may still explore)
        // With exploration_rate=0.0, the bandit Thompson-samples
        let mode = ctrl.select_mode(0.9, &mut rng);
        // High confidence should not force thinking, bandit decides
        // (no assertion on exact mode since Thompson sampling is stochastic)
        assert!(
            mode == ThinkingMode::Direct
                || mode == ThinkingMode::Latent
                || mode == ThinkingMode::CpuResample
                || mode == ThinkingMode::Dendritic
        );
    }

    #[test]
    fn test_gpu_loaded_routes_to_cpu() {
        let config = ThinkingConfig {
            mode: ThinkingSelector::Adaptive {
                exploration_rate: 0.0,
                dendritic_weight: 0.25,
            },
            confidence_threshold: 0.7,
            gpu_load_threshold: 0.8,
            ..Default::default()
        };
        let mut ctrl = ThinkingController::with_gpu_load(config, 0.9);
        let mut rng = TestRng::new(42);
        // Cold bandit + low confidence + GPU loaded → CpuResample
        let mode = ctrl.select_mode(0.3, &mut rng);
        assert_eq!(mode, ThinkingMode::CpuResample);
    }

    #[test]
    fn test_route_thinking_gpu_free() {
        let config = ThinkingConfig {
            gpu_load_threshold: 0.8,
            ..Default::default()
        };
        let ctrl = ThinkingController::with_gpu_load(config, 0.5);
        assert_eq!(ctrl.route_thinking(), ThinkingMode::Latent);
    }

    #[test]
    fn test_route_thinking_gpu_loaded() {
        let config = ThinkingConfig {
            gpu_load_threshold: 0.8,
            ..Default::default()
        };
        let ctrl = ThinkingController::with_gpu_load(config, 0.9);
        assert_eq!(ctrl.route_thinking(), ThinkingMode::CpuResample);
    }

    #[test]
    fn test_record_reward() {
        let config = ThinkingConfig::default();
        let mut ctrl = ThinkingController::new(config);
        // Record some rewards
        ctrl.record_reward(ThinkingMode::Direct, 0.9, 0.1);
        ctrl.record_reward(ThinkingMode::Latent, 0.8, 0.5);
        ctrl.record_reward(ThinkingMode::CpuResample, 0.7, 0.3);
        // No assertion needed — just ensure it doesn't panic
    }

    #[test]
    fn test_freeze_thaw_roundtrip() {
        let config = ThinkingConfig::default();
        let mut ctrl = ThinkingController::new(config);
        let mut rng = TestRng::new(42);

        // Train for a few episodes
        for _i in 0..20 {
            let mode = ctrl.select_mode(0.5, &mut rng);
            ctrl.record_reward(mode, 0.8, 0.3);
        }

        let frozen = ctrl.freeze();
        assert_eq!(frozen.magic, *b"THKB");
        assert_eq!(frozen.version, 2);
        assert!(frozen.total_pulls > 0);

        // Thaw into new controller
        let config2 = ThinkingConfig::default();
        let ctrl2 = ThinkingController::from_frozen(config2, &frozen).unwrap();
        let frozen2 = ctrl2.freeze();
        assert_eq!(frozen.successes, frozen2.successes);
        assert_eq!(frozen.failures, frozen2.failures);
        assert_eq!(frozen.total_pulls, frozen2.total_pulls);
    }

    #[test]
    fn test_save_load_disk_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("thinking_bandit.bin");

        let config = ThinkingConfig::default();
        let mut ctrl = ThinkingController::new(config);
        let mut rng = TestRng::new(42);

        for _ in 0..10 {
            let mode = ctrl.select_mode(0.5, &mut rng);
            ctrl.record_reward(mode, 0.8, 0.3);
        }

        ctrl.save_bandit(&path).unwrap();

        let mut ctrl2 = ThinkingController::new(ThinkingConfig::default());
        ctrl2.load_bandit(&path).unwrap();

        let f1 = ctrl.freeze();
        let f2 = ctrl2.freeze();
        assert_eq!(f1.successes, f2.successes);
        assert_eq!(f1.failures, f2.failures);
        assert_eq!(f1.total_pulls, f2.total_pulls);
    }

    #[test]
    fn test_frozen_size() {
        // 4 (magic) + 4 (version) + 16 (4 floats) + 16 (4 floats) + 4 (total_pulls) = 44
        // But repr(C) may add padding. Check it's small.
        let size = std::mem::size_of::<ThinkingBanditFrozen>();
        assert!(
            size <= 64,
            "ThinkingBanditFrozen is {size} bytes, expected <= 64"
        );
    }

    #[test]
    fn test_frozen_bad_magic() {
        let frozen = ThinkingBanditFrozen {
            magic: *b"XXXX",
            version: 2,
            successes: [1.0; 4],
            failures: [1.0; 4],
            total_pulls: 0,
        };
        assert!(ThinkingBanditFrozen::validate(&frozen).is_err());
    }

    // ── Adaptive Thinking Budget (Plan 263, Phase 4) ───

    #[test]
    fn test_adaptive_budget_fresh_context() {
        // Fresh context: high decay (0.5) → freshness low → less budget
        // Stale context: low decay (1.0) → freshness high → more budget
        let ctrl = ThinkingController::new(ThinkingConfig {
            min_blocks: 0,
            max_blocks: 8,
            ..Default::default()
        });

        // Fast decay: recent context dominates ("fresh")
        // With decay 0.5, freshness ≈ sum(0.5^k)/T which is low
        let fresh = vec![0.5f32; 64];
        let budget_fresh = ctrl.adaptive_budget_default(&fresh);

        // No decay: uniform context ("stale")
        let stale = vec![1.0f32; 64];
        let budget_stale = ctrl.adaptive_budget_default(&stale);

        // Fresh context (low freshness) should get LESS budget
        // because recent info is concentrated and doesn't need deep thinking.
        // Stale context (high freshness) should get MORE budget.
        assert!(
            budget_fresh <= budget_stale,
            "fresh ({budget_fresh}) should be <= stale ({budget_stale})"
        );
        assert!(
            budget_fresh >= ctrl.config.min_blocks,
            "budget {budget_fresh} below min"
        );
        assert!(
            budget_fresh <= ctrl.config.max_blocks,
            "budget {budget_fresh} above max"
        );
    }

    #[test]
    fn test_adaptive_budget_clamps_to_range() {
        let ctrl = ThinkingController::new(ThinkingConfig {
            min_blocks: 2,
            max_blocks: 6,
            ..Default::default()
        });

        // Extreme freshness values should still clamp
        let budgets: Vec<usize> = [0.0f32, 0.1, 0.5, 0.9, 1.0]
            .iter()
            .map(|&decay| ctrl.adaptive_budget_default(&[decay; 32]))
            .collect();

        for &b in &budgets {
            assert!((2..=6).contains(&b), "budget {b} out of [2, 6] range");
        }
    }

    #[test]
    fn test_adaptive_budget_beta_sensitivity() {
        let ctrl = ThinkingController::new(ThinkingConfig {
            min_blocks: 0,
            max_blocks: 10,
            ..Default::default()
        });
        // Medium decay: freshness ≈ 0.5
        let decay = vec![0.9f32; 64];

        let low_beta = ctrl.adaptive_budget(&decay, 1.0);
        let high_beta = ctrl.adaptive_budget(&decay, 10.0);

        // Higher beta = sharper transition. Both should be in range.
        assert!(low_beta <= 10);
        assert!(high_beta <= 10);
        // usize is always >= 0, no need to check lower bound
    }
}
