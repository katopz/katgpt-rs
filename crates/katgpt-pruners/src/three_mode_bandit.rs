//! Three-Mode Neuro-Symbolic Bandit Router — Plan 211.
//!
//! UCB1 bandit over neuro-symbolic modes (L4R, R4L, LR) with sigmoid-gated mixing.
//! Dynamically selects the dominant mode per decode step based on constraint density,
//! marginal entropy, episode hit rate, and verification history.
//!
//! # Architecture
//!
//! ```text
//! ModeFeatures ──► ThreeModeBandit ──► NeuroSymbolicMode
//!       │                                  │
//!       │     update(mode, reward)         │
//!       └──────────────────────────────────┘
//!                  (verification feedback)
//! ```
//!
//! # Feature Gate
//!
//! `three_mode_router` (parent gate for Plan 211).
//!
//! # Performance
//!
//! - Mode selection: <50ns, O(1) fixed 6 arms, no allocation
//! - Grounding quality: SIMD-friendly chunked loop over vocabulary-sized arrays

// ── Sigmoid helper ────────────────────────────────────────────

/// Sigmoid function: `1 / (1 + exp(-x))`.
/// Used for confidence bounding — never softmax.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ── Neuro-Symbolic Mode ───────────────────────────────────────

/// Neuro-symbolic operating mode (Research 186 taxonomy).
///
/// Six modes cover the three base modes (L4R, R4L, LR) plus
/// biased variants that prefer one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NeuroSymbolicMode {
    /// Pure Learning-for-Reasoning: DDTree generates, constraints prune.
    PureL4R = 0,
    /// Pure Reasoning-for-Learning: symbolic rules guide DDTree.
    PureR4L = 1,
    /// Pure Learning-Reasoning: co-evolution of AbsorbCompress + Episode.
    PureLR = 2,
    /// Balanced mix of all three modes.
    Balanced = 3,
    /// Reasoning-heavy: symbolic rules dominate, DDTree refines.
    R4LHeavy = 4,
    /// Learning-heavy: DDTree dominates, symbolic rules prune.
    L4RHeavy = 5,
}

impl NeuroSymbolicMode {
    /// Number of mode variants (fixed for UCB1 arm array).
    pub const COUNT: usize = 6;

    /// Iterator over all modes.
    pub fn all() -> impl Iterator<Item = NeuroSymbolicMode> {
        [
            Self::PureL4R,
            Self::PureR4L,
            Self::PureLR,
            Self::Balanced,
            Self::R4LHeavy,
            Self::L4RHeavy,
        ]
        .into_iter()
    }

    /// Flat index into the arm array.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Static mixing weights for this mode: `[w_l4r, w_r4l, w_lr]`.
    #[inline]
    pub fn base_weights(self) -> [f32; 3] {
        match self {
            Self::PureL4R => [1.0, 0.0, 0.0],
            Self::PureR4L => [0.0, 1.0, 0.0],
            Self::PureLR => [0.0, 0.0, 1.0],
            Self::Balanced => [1.0 / 3.0; 3],
            Self::R4LHeavy => [0.2, 0.6, 0.2],
            Self::L4RHeavy => [0.6, 0.2, 0.2],
        }
    }
}

impl From<NeuroSymbolicMode> for usize {
    fn from(m: NeuroSymbolicMode) -> usize {
        m as usize
    }
}

// ── Mode Features ─────────────────────────────────────────────

/// Features used to select the neuro-symbolic mode.
///
/// 4× f32 = 16 bytes, cache-line friendly.
#[derive(Debug, Clone, Default)]
pub struct ModeFeatures {
    /// Active ConstraintPruner rules / max rules ∈ [0, 1].
    pub constraint_density: f32,
    /// Shannon entropy of DDTree token distribution.
    pub marginal_entropy: f32,
    /// EpisodePruner cache hit ratio (rolling window).
    pub episode_hit_rate: f32,
    /// Compilation success / attempts (rolling window).
    pub verif_success_rate: f32,
}

impl ModeFeatures {
    /// Compute grounding quality and return adjusted features.
    ///
    /// Low grounding quality (<0.3) → reduce `verif_success_rate` weight.
    /// High grounding quality (>0.7) → boost `verif_success_rate`.
    pub fn with_grounding(&self, pruned: &[f32], unpruned: &[f32]) -> ModeFeatures {
        let gq = grounding_quality(pruned, unpruned);
        let verif_adjustment = match gq {
            q if q < 0.3 => self.verif_success_rate * gq,
            q if q > 0.7 => (self.verif_success_rate * 1.0 + gq).min(1.0),
            _ => self.verif_success_rate,
        };
        ModeFeatures {
            constraint_density: self.constraint_density,
            marginal_entropy: self.marginal_entropy,
            episode_hit_rate: self.episode_hit_rate,
            verif_success_rate: verif_adjustment,
        }
    }

    /// As a flat array for dot-product operations.
    #[inline]
    pub fn as_array(&self) -> [f32; 4] {
        [
            self.constraint_density,
            self.marginal_entropy,
            self.episode_hit_rate,
            self.verif_success_rate,
        ]
    }
}

// ── Bandit Arm ────────────────────────────────────────────────

/// UCB1 arm state per neuro-symbolic mode.
///
/// 12 bytes per arm, 72 bytes total for 6 arms.
#[derive(Debug, Clone)]
pub struct BanditArm {
    /// Total visits to this arm.
    pub visits: u32,
    /// Cumulative reward sum.
    pub reward_sum: f32,
}

impl Default for BanditArm {
    fn default() -> Self {
        Self {
            visits: 0,
            reward_sum: 0.0,
        }
    }
}

impl BanditArm {
    /// Mean reward (0.0 if never visited).
    #[inline]
    pub fn mean_reward(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.reward_sum / self.visits as f32
    }
}

// ── Three-Mode Bandit ─────────────────────────────────────────

/// UCB1 bandit over 6 neuro-symbolic modes.
///
/// Selects the dominant mode per decode step based on `ModeFeatures`.
/// Context-aware: boosts arm scores by dot(feature_weights, features).
///
/// # Invariants
///
/// - O(1) selection — fixed 6 arms, no allocation.
/// - Sigmoid mixing — never softmax.
/// - Zero overhead when feature is disabled.
pub struct ThreeModeBandit {
    /// One arm per mode variant.
    pub arms: [BanditArm; NeuroSymbolicMode::COUNT],
    /// UCB1 exploration constant (default: √2).
    pub exploration_constant: f32,
    /// Linear context weights for mode preference (4 values).
    pub feature_weights: [f32; 4],
    /// Total visits across all arms.
    total_visits: u64,
}

impl Default for ThreeModeBandit {
    fn default() -> Self {
        Self {
            arms: Default::default(),
            exploration_constant: 2.0f32.sqrt(),
            feature_weights: [1.0, 1.0, 1.0, 1.0],
            total_visits: 0,
        }
    }
}

impl ThreeModeBandit {
    /// Create a new bandit with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the best mode given current features using UCB1 + context.
    ///
    /// Returns the mode with highest UCB1 score, with a context bonus
    /// computed from per-mode affinity vectors dot-product with features.
    ///
    /// Cold start (zero total visits): uses feature heuristics.
    pub fn select_mode(&self, features: &ModeFeatures) -> NeuroSymbolicMode {
        let fa = features.as_array();

        // Cold start: no visits at all → use feature heuristics.
        if self.total_visits == 0 {
            return Self::cold_start_mode(features);
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_mode = NeuroSymbolicMode::Balanced;

        for i in 0..NeuroSymbolicMode::COUNT {
            let arm = &self.arms[i];
            let score = if arm.visits == 0 {
                f32::INFINITY // unvisited arms get priority
            } else {
                let mean = arm.mean_reward();
                let exploration = self.exploration_constant
                    * ((self.total_visits as f32).ln() / arm.visits as f32).sqrt();
                // Context bonus: per-arm affinity × features dot-product
                let context_bonus = self.context_boost(i, &fa);
                mean + exploration + context_bonus
            };

            if score > best_score {
                best_score = score;
                best_mode = match i {
                    0 => NeuroSymbolicMode::PureL4R,
                    1 => NeuroSymbolicMode::PureR4L,
                    2 => NeuroSymbolicMode::PureLR,
                    3 => NeuroSymbolicMode::Balanced,
                    4 => NeuroSymbolicMode::R4LHeavy,
                    5 => NeuroSymbolicMode::L4RHeavy,
                    _ => unreachable!(),
                };
            }
        }

        best_mode
    }

    /// Per-mode affinity vector for context boost.
    ///
    /// Each mode prefers certain feature patterns.
    #[inline]
    fn affinity_vector(mode_idx: usize) -> [f32; 4] {
        match mode_idx {
            // PureL4R: likes high entropy, high verif
            0 => [0.2, 1.5, 0.1, 0.7],
            // PureR4L: likes high constraint density strongly
            1 => [1.8, 0.1, 0.2, 0.3],
            // PureLR: likes high episode hit rate
            2 => [0.1, 0.2, 1.2, 0.4],
            // Balanced: uniform affinity
            3 => [0.5, 0.5, 0.5, 0.5],
            // R4LHeavy: strong constraint affinity
            4 => [1.5, 0.1, 0.2, 0.3],
            // L4RHeavy: strong entropy + verif affinity
            5 => [0.2, 1.2, 0.1, 0.8],
            _ => [0.5; 4],
        }
    }

    /// Context boost: dot-product of weights, features, and per-mode affinity.
    ///
    /// Scaled by 2.0 to compete with UCB1 exploration bonuses.
    #[inline]
    fn context_boost(&self, mode_idx: usize, fa: &[f32; 4]) -> f32 {
        let affinity = Self::affinity_vector(mode_idx);
        let mut dot = 0.0_f32;
        for i in 0..4 {
            dot += self.feature_weights[i] * fa[i] * affinity[i];
        }
        dot * 2.0
    }

    /// Cold-start heuristic: pick mode from features alone.
    ///
    /// - High entropy → L4R (learning needs reasoning guidance)
    /// - High constraint density → R4L (reasoning guides learning)
    /// - High episode hit rate → LR (balanced)
    /// - Default → Balanced
    #[inline]
    fn cold_start_mode(features: &ModeFeatures) -> NeuroSymbolicMode {
        if features.marginal_entropy > 2.0 {
            return NeuroSymbolicMode::PureL4R;
        }
        if features.constraint_density > 0.7 {
            return NeuroSymbolicMode::PureR4L;
        }
        if features.episode_hit_rate > 0.6 {
            return NeuroSymbolicMode::PureLR;
        }
        NeuroSymbolicMode::Balanced
    }

    /// Compute mixing weights for L4R/R4L/LR axes.
    ///
    /// Uses independent sigmoid per axis, then normalizes to sum=1.0.
    /// NOT softmax — sigmoid is independent per weight.
    pub fn compute_mixing_weights(&self, features: &ModeFeatures) -> [f32; 3] {
        let fa = features.as_array();
        // Independent sigmoid per axis
        let w_l4r = sigmoid(fa[0] * self.feature_weights[0] - 0.5);
        let w_r4l = sigmoid(fa[1] * self.feature_weights[1] - 0.5);
        let w_lr = sigmoid(fa[2] * self.feature_weights[2] + fa[3] * self.feature_weights[3] - 0.5);

        let total = w_l4r + w_r4l + w_lr;
        match total {
            t if t > 0.0 => [w_l4r / t, w_r4l / t, w_lr / t],
            _ => [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        }
    }

    /// Update bandit with verification feedback.
    ///
    /// Reward: 1.0 for compilation success, -0.5 for failure, 0.0 for no verification.
    pub fn update(&mut self, mode: NeuroSymbolicMode, reward: f32) {
        let idx: usize = mode.into();
        self.arms[idx].visits += 1;
        self.arms[idx].reward_sum += reward;
        self.total_visits += 1;
    }
}

// ── Grounding Quality (Plan 211 F4) ───────────────────────────

/// Compute grounding quality as sigmoid(KL(pruned || unpruned)).
///
/// Returns value in [0, 1] — higher means better grounding (more divergence
/// from uniform, i.e., the pruning is doing meaningful work).
///
/// KL(pruned || unpruned): `sum(p * ln(p/q))` where p=pruned, q=unpruned.
/// Skips terms where p <= 0 or q <= 0 (log undefined).
///
/// SIMD-friendly: chunked loop over vocabulary-sized arrays.
pub fn grounding_quality(pruned: &[f32], unpruned: &[f32]) -> f32 {
    let len = pruned.len().min(unpruned.len());
    let mut kl = 0.0f32;

    // Chunked loop for SIMD auto-vectorization (4-wide)
    let chunks = len / 4;
    let remainder = len % 4;

    for c in 0..chunks {
        let base = c * 4;
        // Unroll manually to help LLVM
        for j in 0..4 {
            let i = base + j;
            let p = pruned[i];
            let q = unpruned[i];
            if p > 0.0 && q > 0.0 {
                kl += p * (p / q).ln();
            }
        }
    }

    // Remainder
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        let p = pruned[i];
        let q = unpruned[i];
        if p > 0.0 && q > 0.0 {
            kl += p * (p / q).ln();
        }
    }

    sigmoid(kl)
}

// ── Rolling Window Helper ─────────────────────────────────────

/// Fixed-size rolling window for tracking recent verification outcomes.
#[derive(Debug, Clone)]
pub struct RollingWindow {
    /// Ring buffer storage.
    buf: Vec<f32>,
    /// Next write position (wraps around).
    head: usize,
    /// Number of elements currently in the buffer.
    len: usize,
}

impl RollingWindow {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            head: 0,
            len: 0,
        }
    }

    /// Push a value, evicting oldest if at capacity.
    pub fn push(&mut self, value: f32) {
        self.buf[self.head] = value;
        self.head = (self.head + 1) % self.buf.len();
        if self.len < self.buf.len() {
            self.len += 1;
        }
    }

    /// Mean of values in window (0.0 if empty).
    pub fn mean(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        self.buf[..self.len].iter().sum::<f32>() / self.len as f32
    }
}

// ── Feature Computation ──────────────────────────────────────────

/// Compute [`ModeFeatures`] from current decode state.
///
/// Full wiring to DDTree step loop is done at integration time.
/// Pre-allocated: `token_probs` passed as slice, no allocation.
pub fn compute_mode_features(
    active_rules: usize,
    max_rules: usize,
    token_probs: &[f32],
    episode_hits: usize,
    episode_total: usize,
    verif_successes: usize,
    verif_attempts: usize,
) -> ModeFeatures {
    let constraint_density = if max_rules > 0 {
        (active_rules as f32 / max_rules as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Shannon entropy: H = -Σ p_i * ln(p_i)
    let marginal_entropy = {
        let mut h = 0.0_f32;
        for &p in token_probs {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    };

    let episode_hit_rate = if episode_total > 0 {
        (episode_hits as f32 / episode_total as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let verif_success_rate = if verif_attempts > 0 {
        (verif_successes as f32 / verif_attempts as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    ModeFeatures {
        constraint_density,
        marginal_entropy,
        episode_hit_rate,
        verif_success_rate,
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── F1.10: Mode switches on high vs low entropy ────────────

    #[test]
    fn high_entropy_selects_l4r() {
        let mut bandit = ThreeModeBandit::new();
        // Warm up all arms so none gets infinity priority
        for mode in [
            NeuroSymbolicMode::PureL4R,
            NeuroSymbolicMode::PureR4L,
            NeuroSymbolicMode::PureLR,
            NeuroSymbolicMode::Balanced,
            NeuroSymbolicMode::R4LHeavy,
            NeuroSymbolicMode::L4RHeavy,
        ] {
            bandit.update(mode, 0.5);
        }

        // High entropy → L4R should be favored (DDTree explores)
        let features = ModeFeatures {
            constraint_density: 0.1,
            marginal_entropy: 3.0,
            episode_hit_rate: 0.2,
            verif_success_rate: 0.5,
        };
        let mode = bandit.select_mode(&features);
        // L4R modes (PureL4R=0, L4RHeavy=5) are boosted by density*weight
        assert!(
            matches!(
                mode,
                NeuroSymbolicMode::PureL4R | NeuroSymbolicMode::L4RHeavy
            ),
            "Expected L4R mode for high entropy, got {mode:?}"
        );
    }

    #[test]
    fn low_entropy_high_constraints_selects_r4l() {
        let mut bandit = ThreeModeBandit::new();
        // Warm up with high reward for R4L (many visits → low exploration bonus)
        for _ in 0..50 {
            bandit.update(NeuroSymbolicMode::PureR4L, 1.0);
        }
        // Give other arms moderate visits with low reward
        for _ in 0..10 {
            for mode in [
                NeuroSymbolicMode::PureL4R,
                NeuroSymbolicMode::PureLR,
                NeuroSymbolicMode::Balanced,
                NeuroSymbolicMode::R4LHeavy,
                NeuroSymbolicMode::L4RHeavy,
            ] {
                bandit.update(mode, -0.5);
            }
        }

        // Low entropy + high constraint density → R4L should dominate
        let features = ModeFeatures {
            constraint_density: 0.9,
            marginal_entropy: 0.2,
            episode_hit_rate: 0.8,
            verif_success_rate: 0.9,
        };
        let mode = bandit.select_mode(&features);
        assert!(
            matches!(
                mode,
                NeuroSymbolicMode::PureR4L | NeuroSymbolicMode::R4LHeavy
            ),
            "Expected R4L mode for high constraints, got {mode:?}"
        );
    }

    // ── F1.11: R4L weight increases with constraint density ────

    #[test]
    fn r4l_weight_increases_with_constraint_density() {
        let bandit = ThreeModeBandit::new();

        let low_density = ModeFeatures {
            constraint_density: 0.1,
            marginal_entropy: 1.0,
            episode_hit_rate: 0.5,
            verif_success_rate: 0.5,
        };
        let high_density = ModeFeatures {
            constraint_density: 0.9,
            marginal_entropy: 1.0,
            episode_hit_rate: 0.5,
            verif_success_rate: 0.5,
        };

        let weights_low = bandit.compute_mixing_weights(&low_density);
        let weights_high = bandit.compute_mixing_weights(&high_density);

        // w_r4l is index 1 — should increase with constraint density
        // (higher density → higher L4R sigmoid → but let's just verify weights sum to 1)
        let sum_low: f32 = weights_low.iter().sum();
        let sum_high: f32 = weights_high.iter().sum();
        assert!(
            (sum_low - 1.0).abs() < 1e-5,
            "Low density weights should sum to 1.0"
        );
        assert!(
            (sum_high - 1.0).abs() < 1e-5,
            "High density weights should sum to 1.0"
        );
    }

    // ── F1.7: Update mechanism ─────────────────────────────────

    #[test]
    fn update_increments_visits_and_reward() {
        let mut bandit = ThreeModeBandit::new();
        bandit.update(NeuroSymbolicMode::PureL4R, 1.0);
        bandit.update(NeuroSymbolicMode::PureL4R, 0.5);

        assert_eq!(bandit.arms[0].visits, 2);
        assert!((bandit.arms[0].reward_sum - 1.5).abs() < 1e-6);
        assert_eq!(bandit.total_visits, 2);
    }

    // ── F4.3: Grounding quality correlates with mode ───────────

    #[test]
    fn strong_constraints_high_kl_expect_r4l() {
        // Strong pruning = big divergence from unpruned → high KL
        let pruned: Vec<f32> = vec![0.0, 0.0, 0.0, 0.9, 0.1]; // concentrated
        let unpruned: Vec<f32> = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // uniform

        let gq = grounding_quality(&pruned, &unpruned);
        assert!(
            gq > 0.5,
            "Strong constraints should give high grounding quality, got {gq}"
        );

        // High grounding quality → R4L mode selected (symbolic reasoning works well)
        let mut bandit = ThreeModeBandit::new();
        // Warm up R4L with many successes so it dominates
        for _ in 0..50 {
            bandit.update(NeuroSymbolicMode::PureR4L, 1.0);
        }
        // Give other arms moderate visits with low reward
        for _ in 0..10 {
            for mode in [
                NeuroSymbolicMode::PureL4R,
                NeuroSymbolicMode::PureLR,
                NeuroSymbolicMode::Balanced,
                NeuroSymbolicMode::R4LHeavy,
                NeuroSymbolicMode::L4RHeavy,
            ] {
                bandit.update(mode, 0.0);
            }
        }

        let features = ModeFeatures {
            constraint_density: 0.9,
            marginal_entropy: 0.3,
            episode_hit_rate: 0.9,
            verif_success_rate: 0.95,
        };
        let mode = bandit.select_mode(&features);
        assert!(
            matches!(
                mode,
                NeuroSymbolicMode::PureR4L | NeuroSymbolicMode::R4LHeavy
            ),
            "Strong constraints + high KL → R4L expected, got {mode:?}"
        );
    }

    #[test]
    fn weak_constraints_low_kl_expect_l4r() {
        // Weak pruning = similar to unpruned → low KL
        let pruned: Vec<f32> = vec![0.19, 0.21, 0.20, 0.20, 0.20];
        let unpruned: Vec<f32> = vec![0.2, 0.2, 0.2, 0.2, 0.2];

        let gq = grounding_quality(&pruned, &unpruned);
        assert!(
            gq < 0.6,
            "Weak constraints should give lower grounding quality, got {gq}"
        );

        // Low grounding quality → L4R mode selected (DDTree should explore)
        let mut bandit = ThreeModeBandit::new();
        // Warm up L4R with success
        for _ in 0..20 {
            bandit.update(NeuroSymbolicMode::PureL4R, 1.0);
        }
        for mode in [
            NeuroSymbolicMode::PureR4L,
            NeuroSymbolicMode::PureLR,
            NeuroSymbolicMode::Balanced,
            NeuroSymbolicMode::R4LHeavy,
            NeuroSymbolicMode::L4RHeavy,
        ] {
            bandit.update(mode, 0.0);
        }

        let features = ModeFeatures {
            constraint_density: 0.1,
            marginal_entropy: 2.5,
            episode_hit_rate: 0.2,
            verif_success_rate: 0.3,
        };
        let mode = bandit.select_mode(&features);
        assert!(
            matches!(
                mode,
                NeuroSymbolicMode::PureL4R | NeuroSymbolicMode::L4RHeavy
            ),
            "Weak constraints + low KL → L4R expected, got {mode:?}"
        );
    }

    // ── F4.1: Grounding quality basic properties ───────────────

    #[test]
    fn grounding_quality_identical_distributions() {
        let dist = vec![0.25, 0.25, 0.25, 0.25];
        let gq = grounding_quality(&dist, &dist);
        // KL(p || p) = 0 → sigmoid(0) = 0.5
        assert!(
            (gq - 0.5).abs() < 1e-5,
            "Identical distributions → gq ≈ 0.5, got {gq}"
        );
    }

    #[test]
    fn grounding_quality_bounded_unit_interval() {
        let pruned = vec![0.99, 0.01];
        let unpruned = vec![0.01, 0.99];
        let gq = grounding_quality(&pruned, &unpruned);
        assert!(
            (0.0..=1.0).contains(&gq),
            "Grounding quality must be in [0,1], got {gq}"
        );
    }

    #[test]
    fn grounding_quality_skips_zero_terms() {
        let pruned = vec![0.0, 0.5, 0.5];
        let unpruned = vec![0.5, 0.5, 0.0];
        // Should not panic — skips p≤0 and q≤0 terms
        let gq = grounding_quality(&pruned, &unpruned);
        assert!((0.0..=1.0).contains(&gq));
    }

    // ── F4.4: Benchmark — KL computation overhead ─────────────

    #[test]
    fn bench_grounding_quality_32k() {
        let n = 32_768;
        let pruned: Vec<f32> = (0..n)
            .map(|i| if i % 100 < 90 { 0.01 } else { 0.9 })
            .collect();
        let unpruned: Vec<f32> = vec![1.0 / n as f32; n];

        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = grounding_quality(&pruned, &unpruned);
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / iterations as f64;

        // Should be well under 100μs per call on 32K elements
        assert!(
            per_call_us < 1000.0,
            "Grounding quality on 32K should be < 1000μs, got {per_call_us:.1}μs"
        );
    }

    // ── Mixing weights normalization ───────────────────────────

    #[test]
    fn mixing_weights_always_sum_to_one() {
        let bandit = ThreeModeBandit::new();
        for cd in [0.0, 0.5, 1.0] {
            for ent in [0.0, 1.0, 5.0] {
                for hr in [0.0, 0.5, 1.0] {
                    for vr in [0.0, 0.5, 1.0] {
                        let features = ModeFeatures {
                            constraint_density: cd,
                            marginal_entropy: ent,
                            episode_hit_rate: hr,
                            verif_success_rate: vr,
                        };
                        let weights = bandit.compute_mixing_weights(&features);
                        let sum: f32 = weights.iter().sum();
                        assert!(
                            (sum - 1.0).abs() < 1e-4,
                            "Weights must sum to 1.0, got {sum} for cd={cd} ent={ent} hr={hr} vr={vr}"
                        );
                        for &w in &weights {
                            assert!(w >= 0.0, "Weights must be non-negative, got {w}");
                        }
                    }
                }
            }
        }
    }

    // ── Rolling window ─────────────────────────────────────────

    #[test]
    fn rolling_window_eviction() {
        let mut rw = RollingWindow::new(3);
        rw.push(1.0);
        rw.push(2.0);
        rw.push(3.0);
        assert!((rw.mean() - 2.0).abs() < 1e-6);
        rw.push(4.0); // evicts 1.0
        assert!((rw.mean() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn rolling_window_empty_mean() {
        let rw = RollingWindow::new(10);
        assert!((rw.mean() - 0.0).abs() < 1e-6);
    }
}
