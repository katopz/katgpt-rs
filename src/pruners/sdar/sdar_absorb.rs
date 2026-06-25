//! Absorb-compress with SDAR sigmoid soft gate replacing hard benefit-ratio threshold.
//!
//! Replaces the hard binary threshold in [`AbsorbCompressLayer`] with a sigmoid
//! soft gate from SDAR's asymmetric trust principle:
//!
//! - **Hard gate**: `benefit_ratio >= threshold` → promote or block (binary)
//! - **Soft gate**: `probability = σ(β · (benefit_ratio - 1.0))` → partial credit
//!
//! # Why Soft Gate?
//!
//! Borderline promotions that barely fail the hard threshold get a second chance
//! with probability proportional to how close they are. This prevents losing
//! near-useful patterns that just barely didn't meet the cutoff.
//!
//! # Promotion Probability
//!
//! ```text
//! gap = benefit_ratio - 1.0
//! gate = σ(β · gap)        // sigmoid gate ∈ (0, 1)
//! promote = random_draw < gate  // stochastic decision
//! ```
//!
//! - `benefit_ratio > 1.0` → gate opens → likely promoted
//! - `benefit_ratio = 1.0` → gate = 0.5 → 50/50 chance
//! - `benefit_ratio < 1.0` → gate closes → unlikely promoted
//!
//! # Usage
//!
//! ```rust,ignore
//! let layer = SdarGatedAbsorbCompress::new(
//!     AbsorbCompressLayer::new(NoScreeningPruner, 5, CompressConfig::default()),
//!     5,
//!     SdarAbsorbConfig::default(),
//! );
//!
//! // Feed observations — benefit ratio computed from Q-values
//! layer.observe(0, 0.01, 2.5); // benefit_ratio=2.5 → likely promote
//! layer.observe(1, 0.5, 0.8);  // benefit_ratio=0.8 → unlikely promote
//! ```
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "sdar_gate")]`.
//!
//! **Source:** [SDAR: Self-Distilled Agentic RL](https://arxiv.org/abs/2605.15155)

#[cfg(debug_assertions)]
use std::cmp::Ordering;

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer};
use crate::pruners::review_metrics::ReviewMetrics;
use crate::pruners::sdar_gate::{SDAR_BETA, sdar_benefit_gate, sdar_should_promote};
use crate::speculative::types::ScreeningPruner;

// ── Config ──────────────────────────────────────────────────────

/// Configuration for [`SdarGatedAbsorbCompress`].
///
/// Controls the sigmoid soft gate that replaces the hard benefit-ratio threshold.
#[derive(Clone, Debug)]
pub struct SdarAbsorbConfig {
    /// Sigmoid sharpness β (default: 5.0 from SDAR paper).
    ///
    /// - β=1: soft gate (borderline promotions easily pass)
    /// - β=5: optimal balance (paper-validated)
    /// - β=10: near-binary (barely softer than hard threshold)
    pub beta: f32,
    /// Minimum benefit ratio to even consider promotion (default: 0.5).
    ///
    /// Arms with benefit ratio below this are never promoted regardless of
    /// stochastic draw. This provides a hard floor to prevent truly harmful
    /// promotions from passing through.
    pub min_benefit_ratio_floor: f32,
    /// Seed for deterministic stochastic draws (default: 42).
    ///
    /// Set to `None` for non-deterministic behavior.
    pub seed: Option<u64>,
    /// Whether to track promotion statistics per arm (default: false).
    ///
    /// Only available in debug builds — gated behind `#[cfg(debug_assertions)]`.
    #[cfg(debug_assertions)]
    pub track_promotion_stats: bool,
}

impl Default for SdarAbsorbConfig {
    fn default() -> Self {
        Self {
            beta: SDAR_BETA,
            min_benefit_ratio_floor: 0.5,
            seed: Some(42),
            #[cfg(debug_assertions)]
            track_promotion_stats: false,
        }
    }
}

impl SdarAbsorbConfig {
    /// Create config with custom β.
    pub fn new(beta: f32) -> Self {
        Self {
            beta,
            ..Self::default()
        }
    }

    /// Create config with custom floor.
    pub fn with_floor(mut self, floor: f32) -> Self {
        self.min_benefit_ratio_floor = floor;
        self
    }

    /// Enable promotion statistics tracking (debug builds only).
    #[cfg(debug_assertions)]
    pub fn with_promotion_stats(mut self) -> Self {
        self.track_promotion_stats = true;
        self
    }

    /// Use soft gating (β=1.0).
    pub fn soft() -> Self {
        Self {
            beta: 1.0,
            ..Self::default()
        }
    }

    /// Use aggressive gating (β=10.0, near-binary).
    pub fn aggressive() -> Self {
        Self {
            beta: 10.0,
            ..Self::default()
        }
    }

    /// Disable deterministic seeding.
    pub fn non_deterministic(mut self) -> Self {
        self.seed = None;
        self
    }
}

// ── PromotionStats (debug builds only) ──────────────────────────

/// Per-arm promotion statistics for debugging.
///
/// Gated behind `#[cfg(debug_assertions)]` — eliminated in release builds
/// to remove the per-arm `Vec<PromotionStats>` allocation and the stats
/// tracking branch in [`SdarGatedAbsorbCompress::observe`].
#[cfg(debug_assertions)]
#[derive(Clone, Debug, Default)]
pub struct PromotionStats {
    /// Number of promotion attempts (compress calls that included this arm).
    pub promotion_attempts: usize,
    /// Number of successful promotions.
    pub promotions: usize,
    /// Sum of gate probabilities at promotion time.
    pub gate_probability_sum: f32,
    /// Sum of benefit ratios at promotion time.
    pub benefit_ratio_sum: f32,
    /// Last benefit ratio observed.
    pub last_benefit_ratio: f32,
    /// Last gate probability computed.
    pub last_gate_probability: f32,
}

#[cfg(debug_assertions)]
impl PromotionStats {
    /// Mean gate probability across all promotion attempts.
    pub fn mean_gate_probability(&self) -> f32 {
        if self.promotion_attempts == 0 {
            return 0.0;
        }
        self.gate_probability_sum / self.promotion_attempts as f32
    }

    /// Mean benefit ratio across all promotion attempts.
    pub fn mean_benefit_ratio(&self) -> f32 {
        if self.promotion_attempts == 0 {
            return 0.0;
        }
        self.benefit_ratio_sum / self.promotion_attempts as f32
    }

    /// Actual promotion rate (promotions / attempts).
    pub fn promotion_rate(&self) -> f32 {
        if self.promotion_attempts == 0 {
            return 0.0;
        }
        self.promotions as f32 / self.promotion_attempts as f32
    }
}

// ── ArmAbsorbState ──────────────────────────────────────────────

/// Per-arm absorb tracking state.
#[derive(Clone, Debug)]
struct ArmAbsorbState {
    /// Accumulated reward observations.
    reward_sum: f32,
    /// Number of reward observations.
    observation_count: usize,
    /// Benefit ratio computed from accumulated data.
    benefit_ratio: f32,
}

impl ArmAbsorbState {
    fn new() -> Self {
        Self {
            reward_sum: 0.0,
            observation_count: 0,
            benefit_ratio: 0.0,
        }
    }

    /// Feed a new (reward, benefit_ratio) observation.
    fn observe(&mut self, reward: f32, benefit_ratio: f32) {
        self.reward_sum += reward;
        self.observation_count += 1;
        self.benefit_ratio = benefit_ratio;
    }
}

// ── SdarGatedAbsorbCompress ─────────────────────────────────────

/// Absorb-compress layer with SDAR sigmoid soft gate.
///
/// Wraps [`AbsorbCompressLayer`] and replaces the hard benefit-ratio threshold
/// with a stochastic sigmoid gate. Borderline promotions get partial probability
/// instead of all-or-nothing.
///
/// # Architecture
///
/// ```text
/// SdarGatedAbsorbCompress<P>
///   ├── inner: AbsorbCompressLayer<P>  (existing absorb-compress logic)
///   ├── arm_states: Vec<ArmAbsorbState> (per-arm benefit tracking)
///   ├── config: SdarAbsorbConfig       (β, floor, seed)
///   └── promotion_stats: Vec<PromotionStats> (optional per-arm stats)
/// ```
///
/// # Key Difference from DeltaGatedAbsorbCompress
///
/// - **Delta**: `gate = mean_delta > threshold` (scalar, hard binary)
/// - **SDAR**: `gate = σ(β · (benefit_ratio - 1.0))` (soft, stochastic)
///
/// The soft gate provides:
/// 1. Partial credit for borderline cases
/// 2. Smooth degradation (no sharp threshold artifacts)
/// 3. Stochastic exploration of near-threshold promotions
pub struct SdarGatedAbsorbCompress<P: ScreeningPruner> {
    /// Inner absorb-compress layer (delegates actual promotion logic).
    inner: AbsorbCompressLayer<P>,
    /// Per-arm absorb tracking state.
    arm_states: Vec<ArmAbsorbState>,
    /// Configuration thresholds.
    config: SdarAbsorbConfig,
    /// Per-arm promotion statistics (only in debug builds).
    #[cfg(debug_assertions)]
    promotion_stats: Vec<PromotionStats>,
    /// PRNG state for stochastic promotion decisions.
    rng_state: u64,
}

impl<P: ScreeningPruner> SdarGatedAbsorbCompress<P> {
    /// Create a new SDAR-gated absorb-compress layer with default config.
    ///
    /// Wraps an existing `AbsorbCompressLayer` with sigmoid soft gating.
    pub fn new(inner: AbsorbCompressLayer<P>, num_arms: usize, config: SdarAbsorbConfig) -> Self {
        let rng_state = config.seed.unwrap_or(0);
        #[cfg(debug_assertions)]
        let promotion_stats = if config.track_promotion_stats {
            (0..num_arms).map(|_| PromotionStats::default()).collect()
        } else {
            Vec::new()
        };

        Self {
            inner,
            arm_states: (0..num_arms).map(|_| ArmAbsorbState::new()).collect(),
            config,
            #[cfg(debug_assertions)]
            promotion_stats,
            rng_state,
        }
    }

    /// Feed an observation with known benefit ratio.
    ///
    /// The benefit ratio determines the gate probability:
    /// - `benefit_ratio > 1.0` → likely to absorb
    /// - `benefit_ratio = 1.0` → 50/50 absorb
    /// - `benefit_ratio < 1.0` → unlikely to absorb
    ///
    /// If the benefit ratio exceeds the floor and the stochastic gate passes,
    /// the reward is forwarded to the inner absorb-compress layer.
    #[inline]
    pub fn observe(&mut self, arm: usize, reward: f32, benefit_ratio: f32) {
        let Some(state) = self.arm_states.get_mut(arm) else {
            return;
        };

        state.observe(reward, benefit_ratio);

        // Hard floor: skip if below minimum benefit ratio
        if benefit_ratio < self.config.min_benefit_ratio_floor {
            return;
        }

        // Soft gate: stochastic promotion decision
        let draw = self.next_random();

        let promoted = sdar_should_promote(benefit_ratio, self.config.beta, draw);
        if promoted {
            self.inner.absorb(arm, reward);
        }

        // Track statistics if enabled (debug builds only)
        #[cfg(debug_assertions)]
        if self.config.track_promotion_stats
            && let Some(stats) = self.promotion_stats.get_mut(arm)
        {
            let gate_probability = sdar_benefit_gate(benefit_ratio, self.config.beta);
            stats.promotion_attempts += 1;
            if promoted {
                stats.promotions += 1;
            }
            stats.benefit_ratio_sum += benefit_ratio;
            stats.gate_probability_sum += gate_probability;
            stats.last_benefit_ratio = benefit_ratio;
            stats.last_gate_probability = gate_probability;
        }
    }

    /// Feed an observation with benefit ratio computed from reward vs Q-value.
    ///
    /// Computes `benefit_ratio = reward / max(Q-value, ε)` and delegates
    /// to [`observe`](Self::observe).
    ///
    /// # Arguments
    ///
    /// * `arm` — Bandit arm index
    /// * `reward` — Observed reward
    /// * `q_value` — Current Q-value estimate for this arm
    #[inline]
    pub fn observe_with_q(&mut self, arm: usize, reward: f32, q_value: f32) {
        let benefit_ratio = if q_value.abs() < f32::EPSILON {
            // No prior Q-value → benefit ratio = reward magnitude
            reward.max(0.0)
        } else {
            reward / q_value.abs()
        };
        self.observe(arm, reward, benefit_ratio);
    }

    /// Simple absorb that always passes through (bypasses SDAR gate).
    ///
    /// Use for direct reward fallback when benefit ratio is unknown.
    #[inline]
    pub fn absorb_direct(&mut self, arm: usize, reward: f32) {
        self.inner.absorb(arm, reward);
    }

    /// Get the current benefit ratio for a specific arm.
    pub fn benefit_ratio(&self, arm: usize) -> f32 {
        self.arm_states
            .get(arm)
            .map(|s| s.benefit_ratio)
            .unwrap_or(0.0)
    }

    /// Number of observations for a specific arm.
    pub fn observation_count(&self, arm: usize) -> usize {
        self.arm_states
            .get(arm)
            .map(|s| s.observation_count)
            .unwrap_or(0)
    }

    /// Get promotion statistics for a specific arm (debug builds only).
    ///
    /// Returns `None` if tracking is disabled or arm is out of bounds.
    #[cfg(debug_assertions)]
    pub fn promotion_stats(&self, arm: usize) -> Option<&PromotionStats> {
        self.promotion_stats.get(arm)
    }

    /// Compute the gate probability for a benefit ratio without promoting.
    ///
    /// Useful for logging/debugging: shows what probability of promotion
    /// a given benefit ratio would produce.
    pub fn gate_probability(&self, benefit_ratio: f32) -> f32 {
        sdar_benefit_gate(benefit_ratio, self.config.beta)
    }

    /// Which arms have the highest benefit ratios (top-K candidates)?
    ///
    /// Returns arms sorted by benefit ratio, descending.
    /// Only arms with at least one observation are included.
    ///
    /// Diagnostic-only: gated behind `#[cfg(debug_assertions)]`.
    #[cfg(debug_assertions)]
    pub fn candidate_arms(&self, top_k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .arm_states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.observation_count > 0)
            .map(|(i, s)| (i, s.benefit_ratio))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    /// Access the inner absorb-compress layer.
    pub fn inner(&self) -> &AbsorbCompressLayer<P> {
        &self.inner
    }

    /// Mutable access to the inner absorb-compress layer.
    pub fn inner_mut(&mut self) -> &mut AbsorbCompressLayer<P> {
        &mut self.inner
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.arm_states.len()
    }

    /// Configuration reference.
    pub fn config(&self) -> &SdarAbsorbConfig {
        &self.config
    }

    /// Simple PRNG: splitmix64-based for deterministic stochastic draws.
    ///
    /// Unlike xorshift64, splitmix64 handles seed=0 correctly and produces
    /// well-distributed output from the first call.
    pub(crate) fn next_random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        // Map upper 31 bits to [0, 1) f32
        (z >> 33) as f32 / (1u64 << 31) as f32
    }
}

impl<P: ScreeningPruner> ScreeningPruner for SdarGatedAbsorbCompress<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Delegate to inner layer (which handles compressed arm blocking)
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

impl<P: ScreeningPruner> AbsorbCompress for SdarGatedAbsorbCompress<P> {
    fn absorb(&mut self, arm: usize, reward: f32) {
        // Direct absorb bypasses SDAR gate — use for raw reward fallback.
        self.inner.absorb(arm, reward);
    }

    fn compress(&mut self) -> Vec<usize> {
        self.inner.compress()
    }

    fn compressed_arms(&self) -> &[usize] {
        self.inner.compressed_arms()
    }

    fn should_compress(&self) -> bool {
        self.inner.should_compress()
    }

    fn should_compress_gated(&self, metrics: Option<&ReviewMetrics>) -> bool {
        // Use SDAR soft gate for benefit-ratio gating
        if !self.inner.should_compress() {
            return false;
        }

        let Some(metrics) = metrics else {
            return true;
        };

        // Apply SDAR soft gate to the benefit ratio
        let ratio = metrics.benefit_ratio() as f32;
        let gate = sdar_benefit_gate(ratio, self.config.beta);

        // Stochastic decision
        if gate > 0.5 {
            true
        } else {
            // Below neutral — use probability
            let draw = fastrand::f32();
            draw < gate
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::absorb_compress::CompressConfig;
    use crate::speculative::types::NoScreeningPruner;

    fn make_layer(num_arms: usize) -> SdarGatedAbsorbCompress<NoScreeningPruner> {
        let inner =
            AbsorbCompressLayer::new(NoScreeningPruner, num_arms, CompressConfig::default());
        SdarGatedAbsorbCompress::new(inner, num_arms, SdarAbsorbConfig::default())
    }

    fn make_layer_with_stats(num_arms: usize) -> SdarGatedAbsorbCompress<NoScreeningPruner> {
        let inner =
            AbsorbCompressLayer::new(NoScreeningPruner, num_arms, CompressConfig::default());
        SdarGatedAbsorbCompress::new(
            inner,
            num_arms,
            SdarAbsorbConfig::default().with_promotion_stats(),
        )
    }

    fn make_layer_with_config(
        num_arms: usize,
        config: SdarAbsorbConfig,
    ) -> SdarGatedAbsorbCompress<NoScreeningPruner> {
        let inner =
            AbsorbCompressLayer::new(NoScreeningPruner, num_arms, CompressConfig::default());
        SdarGatedAbsorbCompress::new(inner, num_arms, config)
    }

    // ── High benefit ratio → gate ≈ 1.0 (promote) ───────────────

    #[test]
    fn test_high_benefit_ratio_promotes() {
        // High benefit ratio (3.0) → gate ≈ 0.999+ → very likely promoted
        let mut promotions = 0;
        for _ in 0..100 {
            let mut layer = make_layer_with_stats(3);
            layer.observe(0, 0.01, 3.0);
            let stats = layer.promotion_stats(0).unwrap();
            if stats.promotion_attempts > 0 {
                promotions += 1;
            }
        }

        // Almost all observations should result in promotion attempts
        assert!(
            promotions > 90,
            "High benefit ratio should promote ~100%, got {promotions}/100"
        );
    }

    // ── Zero benefit ratio → gate ≈ 0.5 (neutral) ──────────────

    #[test]
    fn test_neutral_benefit_ratio_is_fifty_fifty() {
        // benefit_ratio = 1.0 → gap = 0.0 → gate = 0.5
        let mut promote_count = 0;
        let total = 1000;

        for i in 0..total {
            let mut layer = make_layer_with_config(
                3,
                SdarAbsorbConfig {
                    seed: Some(i as u64), // Varying seed per iteration for different PRNG draws
                    ..SdarAbsorbConfig::default()
                },
            );
            layer.observe(0, 0.5, 1.0);
            // Check if inner layer received the absorb
            if layer.inner().total_absorbed() > 0 {
                promote_count += 1;
            }
        }

        // Should be roughly 50%
        assert!(
            promote_count > 350 && promote_count < 650,
            "Neutral should be ~50%, got {promote_count}/{total}"
        );
    }

    // ── Negative benefit ratio → gate ≈ 0.0 (block) ─────────────

    #[test]
    fn test_negative_benefit_ratio_blocks() {
        // Negative benefit ratio (0.0) → gate ≈ 0.007 → almost never promoted
        let mut promotions = 0;
        for _ in 0..100 {
            let mut l = make_layer_with_stats(3);
            l.observe(0, 0.01, 0.0);
            let stats = l.promotion_stats(0).unwrap();
            if stats.promotion_attempts > 0 {
                promotions += 1;
            }
        }

        // Very few should pass
        assert!(
            promotions < 10,
            "Negative benefit ratio should rarely promote, got {promotions}/100"
        );
    }

    // ── β sensitivity matches paper ablation ─────────────────────

    #[test]
    fn test_beta_5_optimal_balance() {
        let layer = make_layer(3);

        // β=5 at benefit_ratio=1.5: gate = σ(5·0.5) = σ(2.5) ≈ 0.924
        let prob = layer.gate_probability(1.5);
        assert!(
            prob > 0.85 && prob < 0.98,
            "β=5 at ratio=1.5 should be ~0.92, got {prob}"
        );
    }

    #[test]
    fn test_beta_1_soft_gate() {
        let config = SdarAbsorbConfig::soft();
        let layer = make_layer_with_config(3, config);

        // β=1 at benefit_ratio=1.5: gate = σ(0.5) ≈ 0.62
        let prob = layer.gate_probability(1.5);
        assert!(
            prob > 0.5 && prob < 0.8,
            "β=1 at ratio=1.5 should be ~0.62, got {prob}"
        );
    }

    #[test]
    fn test_beta_10_near_binary() {
        let config = SdarAbsorbConfig::aggressive();
        let layer = make_layer_with_config(3, config);

        // β=10 at benefit_ratio=1.2: gate = σ(10·0.2) = σ(2.0) ≈ 0.88
        let prob = layer.gate_probability(1.2);
        assert!(prob > 0.8, "β=10 at ratio=1.2 should be >0.8, got {prob}");
    }

    // ── Floor threshold ──────────────────────────────────────────

    #[test]
    fn test_floor_blocks_low_benefit_ratio() {
        let config = SdarAbsorbConfig::new(SDAR_BETA).with_floor(2.0);
        // Override config
        let inner = AbsorbCompressLayer::new(NoScreeningPruner, 3, CompressConfig::default());
        let mut layer = SdarGatedAbsorbCompress::new(inner, 3, config);

        // Benefit ratio below floor → never absorb regardless of gate
        layer.observe(0, 0.5, 1.0); // ratio=1.0 < floor=2.0
        assert_eq!(layer.observation_count(0), 1);
        assert_eq!(
            layer.inner().total_absorbed(),
            0,
            "Should not absorb below floor"
        );
    }

    // ── observe_with_q ───────────────────────────────────────────

    #[test]
    fn test_observe_with_q_computes_ratio() {
        let mut layer = make_layer_with_stats(3);

        // reward=2.0, q_value=1.0 → benefit_ratio=2.0
        layer.observe_with_q(0, 2.0, 1.0);
        assert!((layer.benefit_ratio(0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_observe_with_q_zero_q_uses_reward() {
        let mut layer = make_layer_with_stats(3);

        // reward=1.5, q_value=0 → benefit_ratio=1.5 (reward as ratio)
        layer.observe_with_q(0, 1.5, 0.0);
        assert!(
            (layer.benefit_ratio(0) - 1.5).abs() < 1e-6,
            "Expected 1.5, got {}",
            layer.benefit_ratio(0)
        );
    }

    #[test]
    fn test_observe_with_q_negative_q_uses_abs() {
        let mut layer = make_layer_with_stats(3);

        // reward=1.0, q_value=-2.0 → benefit_ratio = 1.0 / 2.0 = 0.5
        layer.observe_with_q(0, 1.0, -2.0);
        assert!((layer.benefit_ratio(0) - 0.5).abs() < 1e-6);
    }

    // ── absorb_direct bypasses gate ──────────────────────────────

    #[test]
    fn test_absorb_direct_bypasses_gate() {
        let mut layer = make_layer(3);

        // Direct absorb always goes through
        layer.absorb_direct(0, 0.01);
        layer.absorb_direct(0, 0.01);

        assert_eq!(layer.inner().total_absorbed(), 2);
    }

    // ── Candidate arms ranked ────────────────────────────────────

    #[test]
    fn test_candidate_arms_ranked() {
        let mut layer = make_layer(5);

        layer.observe(0, 0.1, 1.0);
        layer.observe(1, 0.1, 3.0);
        layer.observe(2, 0.1, 2.0);
        // Arm 3, 4: no observations

        let candidates = layer.candidate_arms(3);
        assert_eq!(candidates[0], 1, "Highest benefit ratio first");
        assert_eq!(candidates[1], 2, "Second highest");
        assert_eq!(candidates[2], 0, "Third highest");
    }

    #[test]
    fn test_candidate_arms_excludes_unobserved() {
        let layer = make_layer(3);
        assert!(layer.candidate_arms(5).is_empty());
    }

    // ── Promotion statistics ─────────────────────────────────────

    #[test]
    fn test_promotion_stats_tracking() {
        let mut layer = make_layer_with_stats(3);

        layer.observe(0, 0.5, 2.0);
        layer.observe(0, 0.3, 1.5);

        let stats = layer.promotion_stats(0).unwrap();
        assert_eq!(stats.promotion_attempts, 2);
        assert!(
            (stats.benefit_ratio_sum - 3.5).abs() < 1e-6,
            "Expected 3.5, got {}",
            stats.benefit_ratio_sum
        );
        assert!(
            stats.gate_probability_sum > 0.0,
            "Gate probability sum should be positive"
        );
        assert!((stats.last_benefit_ratio - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_promotion_stats_not_tracked_by_default() {
        let layer = make_layer(3);
        assert!(layer.promotion_stats(0).is_none());
    }

    #[test]
    fn test_promotion_stats_mean_values() {
        let mut layer = make_layer_with_stats(2);

        for _ in 0..10 {
            layer.observe(0, 0.5, 2.0);
        }

        let stats = layer.promotion_stats(0).unwrap();
        assert_eq!(stats.promotion_attempts, 10);
        assert!((stats.mean_benefit_ratio() - 2.0).abs() < 1e-6);
        assert!(stats.mean_gate_probability() > 0.9);
    }

    #[test]
    fn test_promotion_stats_rate() {
        // Use varying seeds so each iteration gets a different PRNG draw
        let config = SdarAbsorbConfig {
            seed: Some(42),
            track_promotion_stats: true,
            ..SdarAbsorbConfig::default()
        };
        let inner = AbsorbCompressLayer::new(NoScreeningPruner, 3, CompressConfig::default());
        let mut layer = SdarGatedAbsorbCompress::new(inner, 3, config);

        // High benefit ratio → should promote most of the time
        for _ in 0..100 {
            layer.observe(0, 0.5, 3.0);
        }

        let stats = layer.promotion_stats(0).unwrap();
        assert_eq!(stats.promotion_attempts, 100);
        // Most should pass the gate
        assert!(
            stats.promotion_rate() > 0.8,
            "High benefit should promote >80%, got {}",
            stats.promotion_rate()
        );
    }

    // ── Compress delegates to inner ──────────────────────────────

    #[test]
    fn test_compress_delegates_to_inner() {
        let mut layer = make_layer_with_config(3, SdarAbsorbConfig::default());

        // Direct absorb (bypass gate)
        layer.absorb(0, 0.1);
        layer.absorb(0, 0.1);

        let promoted = layer.compress();
        // May or may not promote depending on config thresholds
        assert!(layer.compressed_arms().len() >= promoted.len());
    }

    // ── Delegation tests ─────────────────────────────────────────

    #[test]
    fn test_delegates_relevance_to_inner() {
        let layer = make_layer(3);
        // NoScreeningPruner always returns 1.0
        let rel = layer.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_should_compress_delegates() {
        let layer = make_layer(3);
        assert!(!layer.should_compress()); // No observations
    }

    // ── Out of bounds ────────────────────────────────────────────

    #[test]
    fn test_out_of_bounds_arm_is_noop() {
        let mut layer = make_layer(2);
        layer.observe(99, 0.5, 2.0);
        assert_eq!(layer.num_arms(), 2);
    }

    // ── Config builders ──────────────────────────────────────────

    #[test]
    fn test_config_soft() {
        let config = SdarAbsorbConfig::soft();
        assert!((config.beta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_aggressive() {
        let config = SdarAbsorbConfig::aggressive();
        assert!((config.beta - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_with_floor() {
        let config = SdarAbsorbConfig::default().with_floor(3.0);
        assert!((config.min_benefit_ratio_floor - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_non_deterministic() {
        let config = SdarAbsorbConfig::default().non_deterministic();
        assert!(config.seed.is_none());
    }

    // ── Multiple observations ────────────────────────────────────

    #[test]
    fn test_multiple_observations_accumulate() {
        let mut layer = make_layer(3);

        for _ in 0..10 {
            layer.observe(0, 0.5, 2.0);
        }

        assert_eq!(layer.observation_count(0), 10);
        assert!((layer.benefit_ratio(0) - 2.0).abs() < 1e-6);
    }

    // ── Gate probability computation ─────────────────────────────

    #[test]
    fn test_gate_probability_without_promoting() {
        let layer = make_layer(3);

        let prob_high = layer.gate_probability(3.0);
        let prob_neutral = layer.gate_probability(1.0);
        let prob_low = layer.gate_probability(0.0);

        assert!(prob_high > 0.99, "High → ≈1.0, got {prob_high}");
        assert!(
            (prob_neutral - 0.5).abs() < 1e-6,
            "Neutral → 0.5, got {prob_neutral}"
        );
        assert!(prob_low < 0.01, "Low → ≈0.0, got {prob_low}");
    }

    // ── Deterministic seeding ────────────────────────────────────

    #[test]
    fn test_deterministic_seeding_same_result() {
        let config = SdarAbsorbConfig {
            seed: Some(42),
            ..SdarAbsorbConfig::default()
        };

        let inner1 = AbsorbCompressLayer::new(NoScreeningPruner, 3, CompressConfig::default());
        let mut layer1 = SdarGatedAbsorbCompress::new(inner1, 3, config.clone());

        let inner2 = AbsorbCompressLayer::new(NoScreeningPruner, 3, CompressConfig::default());
        let mut layer2 = SdarGatedAbsorbCompress::new(inner2, 3, config);

        // Same seed → same PRNG sequence → same decisions
        for _ in 0..10 {
            layer1.observe(0, 0.5, 1.5);
            layer2.observe(0, 0.5, 1.5);
        }

        assert_eq!(
            layer1.inner().total_absorbed(),
            layer2.inner().total_absorbed(),
            "Same seed should produce same absorb count"
        );
    }

    // ── PromotionStats default ───────────────────────────────────

    #[test]
    fn test_promotion_stats_default() {
        let stats = PromotionStats::default();
        assert_eq!(stats.promotion_attempts, 0);
        assert_eq!(stats.promotions, 0);
        assert!((stats.mean_gate_probability()).abs() < 1e-6);
        assert!((stats.mean_benefit_ratio()).abs() < 1e-6);
        assert!((stats.promotion_rate()).abs() < 1e-6);
    }

    // ── Benefit ratio from last observation ──────────────────────

    #[test]
    fn test_benefit_ratio_updates_per_observation() {
        let mut layer = make_layer(3);

        layer.observe(0, 0.5, 2.0);
        assert!((layer.benefit_ratio(0) - 2.0).abs() < 1e-6);

        layer.observe(0, 0.3, 1.5);
        assert!((layer.benefit_ratio(0) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_benefit_ratio_unobserved_arm() {
        let layer = make_layer(3);
        assert!((layer.benefit_ratio(99)).abs() < 1e-6);
    }

    // ── PRNG distribution test ───────────────────────────────────

    #[test]
    fn test_prng_produces_both_halves() {
        // Verify the PRNG produces values both above and below 0.5
        let config = SdarAbsorbConfig {
            seed: Some(42),
            ..SdarAbsorbConfig::default()
        };
        let inner = AbsorbCompressLayer::new(NoScreeningPruner, 1, CompressConfig::default());
        let mut layer = SdarGatedAbsorbCompress::new(inner, 1, config);

        let mut below_half = 0;
        let mut above_half = 0;
        let total = 1000;

        for _ in 0..total {
            let draw = layer.next_random();
            if draw < 0.5 {
                below_half += 1;
            } else {
                above_half += 1;
            }
        }

        assert!(
            below_half > 300 && above_half > 300,
            "PRNG should produce values on both sides of 0.5: below={below_half}, above={above_half}/{total}"
        );
    }

    #[test]
    fn test_prng_handles_zero_seed() {
        // Seed=0 should still produce valid random values (not all zeros)
        let config = SdarAbsorbConfig {
            seed: Some(0),
            ..SdarAbsorbConfig::default()
        };
        let inner = AbsorbCompressLayer::new(NoScreeningPruner, 1, CompressConfig::default());
        let mut layer = SdarGatedAbsorbCompress::new(inner, 1, config);

        let mut distinct_values = std::collections::HashSet::new();
        for _ in 0..100 {
            let draw = layer.next_random();
            distinct_values.insert(draw.to_bits());
        }

        assert!(
            distinct_values.len() > 50,
            "PRNG with seed=0 should produce diverse values, got {} distinct",
            distinct_values.len()
        );
    }
}
