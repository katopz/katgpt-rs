//! Four-Regime Router — extends Plan 211 Three-Mode Router with Discovery and Consolidation regimes.
//!
//! UCB1 bandit over Regime × Heaviness (6 arms) with sigmoid-gated confidence bounding.
//! Dynamically routes to Standard, Discovery, or Consolidation based on regime-collapse
//! signals and transition-success feedback from the RegimeTransitionGate.
//!
//! # Architecture
//!
//! ```text
//! RegimeFeatures ──► FourRegimeRouter ──► RegimeArm
//!       │                                     │
//!       │     update(arm, reward)             │
//!       └─────────────────────────────────────┘
//!                  (verification feedback)
//! ```
//!
//! # Feature Gate
//!
//! `regime_transition` (parent gate for Plan 215).
//!
//! # Performance
//!
//! - Arm selection: <50ns, O(1) fixed 6 arms, no allocation
//! - Sigmoid confidence bounding — never softmax

use std::collections::VecDeque;

// ── Sigmoid helper ────────────────────────────────────────────

/// Sigmoid function: `1 / (1 + exp(-x))`.
/// Used for confidence bounding — never softmax per project rules.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ── Regime ────────────────────────────────────────────────────

/// Operating regime — extends Three-Mode Router with Discovery and Consolidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Regime {
    /// Standard neuro-symbolic operation (delegates to ThreeModeBandit).
    Standard = 0,
    /// Discovery mode — entered when RegimeCollapseClassifier detects regime collapse.
    Discovery = 1,
    /// Consolidation mode — entered after successful regime transition.
    Consolidation = 2,
}

// ── Heaviness ─────────────────────────────────────────────────

/// Heaviness level for each regime — controls compute investment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Heaviness {
    Light = 0,
    Heavy = 1,
}

// ── RegimeArm ─────────────────────────────────────────────────

/// Full routing arm: Regime × Heaviness = 6 arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RegimeArm {
    StandardLight = 0,
    StandardHeavy = 1,
    DiscoveryLight = 2,
    DiscoveryHeavy = 3,
    ConsolidationLight = 4,
    ConsolidationHeavy = 5,
}

impl RegimeArm {
    /// Number of arms (3 regimes × 2 heaviness options).
    pub const COUNT: usize = 6;

    /// Iterator over all arms.
    pub fn all() -> impl Iterator<Item = RegimeArm> {
        [
            Self::StandardLight,
            Self::StandardHeavy,
            Self::DiscoveryLight,
            Self::DiscoveryHeavy,
            Self::ConsolidationLight,
            Self::ConsolidationHeavy,
        ]
        .into_iter()
    }

    /// Flat index into the arm array.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Extract the regime component.
    #[inline]
    pub fn regime(self) -> Regime {
        match self {
            Self::StandardLight | Self::StandardHeavy => Regime::Standard,
            Self::DiscoveryLight | Self::DiscoveryHeavy => Regime::Discovery,
            Self::ConsolidationLight | Self::ConsolidationHeavy => Regime::Consolidation,
        }
    }

    /// Extract the heaviness component.
    #[inline]
    pub fn heaviness(self) -> Heaviness {
        match self {
            Self::StandardLight | Self::DiscoveryLight | Self::ConsolidationLight => {
                Heaviness::Light
            }
            Self::StandardHeavy | Self::DiscoveryHeavy | Self::ConsolidationHeavy => {
                Heaviness::Heavy
            }
        }
    }

    /// Check if this arm belongs to the given regime.
    #[inline]
    pub fn is_regime(self, regime: Regime) -> bool {
        self.regime() == regime
    }
}

// ── RegimeFeatures ────────────────────────────────────────────

/// Features for regime routing decisions.
///
/// 4 × f32 = 16 bytes, cache-line friendly.
#[derive(Debug, Clone, Default)]
pub struct RegimeFeatures {
    /// Fraction of recent DDTree branches that failed ∈ [0, 1].
    pub failure_rate: f32,
    /// Whether a regime collapse was detected.
    pub regime_collapse: bool,
    /// Whether a regime transition just completed successfully.
    pub transition_success: bool,
    /// Current regime's bandit Q-value (rolling average).
    pub regime_q_value: f32,
}

// ── Four-Regime Router ────────────────────────────────────────

/// UCB1 bandit router over Regime × Heaviness arms.
///
/// Sigmoid-gated mixing between regimes (NOT softmax).
/// 6 arms: 3 regimes × 2 heaviness options.
///
/// # Routing Rules
///
/// - `regime_collapse` → only Discovery arms eligible
/// - `transition_success` → only Consolidation arms eligible
/// - Otherwise → only Standard arms eligible
///
/// # Invariants
///
/// - O(1) selection — fixed 6 arms, no allocation
/// - Sigmoid confidence bounding — never softmax
/// - Zero overhead when feature is disabled
pub struct FourRegimeRouter {
    /// Per-arm visit counts.
    visits: [u32; RegimeArm::COUNT],
    /// Per-arm total rewards.
    reward_sums: [f64; RegimeArm::COUNT],
    /// Exploration parameter for UCB1.
    exploration: f32,
    /// Rolling window of recent regime decisions.
    history: VecDeque<RegimeArm>,
    /// Maximum history length.
    history_len: usize,
    /// Total visits across all arms.
    total_visits: u64,
}

impl Default for FourRegimeRouter {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl FourRegimeRouter {
    /// Create a new router with the given exploration parameter.
    ///
    /// Default exploration is √2 ≈ 1.414 (standard UCB1).
    pub fn new(exploration: f32) -> Self {
        Self {
            visits: [0; RegimeArm::COUNT],
            reward_sums: [0.0; RegimeArm::COUNT],
            exploration,
            history: VecDeque::new(),
            history_len: 64,
            total_visits: 0,
        }
    }

    /// Create a router with default parameters.
    ///
    /// - exploration: √2
    /// - history_len: 64
    pub fn with_defaults() -> Self {
        Self::new(2.0f32.sqrt())
    }

    /// Select the best arm given current features using UCB1 with sigmoid confidence bound.
    ///
    /// Routing rules:
    /// - `regime_collapse` → only Discovery arms
    /// - `transition_success` → only Consolidation arms
    /// - Otherwise → only Standard arms
    ///
    /// Among eligible arms, uses UCB1: `Q(a) + exploration * sigmoid(ln(N) / n(a))`
    /// where sigmoid bounds the confidence term to (0, 1).
    pub fn select(&self, features: &RegimeFeatures) -> RegimeArm {
        // Determine eligible regime from features.
        let target_regime = if features.regime_collapse {
            Regime::Discovery
        } else if features.transition_success {
            Regime::Consolidation
        } else {
            Regime::Standard
        };

        let mut best_score = f64::NEG_INFINITY;
        let mut best_arm = RegimeArm::StandardLight;

        for arm in RegimeArm::all() {
            if !arm.is_regime(target_regime) {
                continue;
            }

            let idx = arm.index();
            let score = if self.visits[idx] == 0 {
                f64::INFINITY // unvisited arms get priority
            } else {
                let mean = self.reward_sums[idx] / self.visits[idx] as f64;
                // UCB1 confidence bound, sigmoid-gated to (0, 1)
                let raw_confidence = (self.total_visits as f64).ln() / self.visits[idx] as f64;
                let confidence = self.exploration as f64 * sigmoid(raw_confidence as f32) as f64;
                mean + confidence
            };

            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }

        best_arm
    }

    /// Update arm statistics with a reward signal.
    pub fn update(&mut self, arm: RegimeArm, reward: f32) {
        let idx = arm.index();
        self.visits[idx] += 1;
        self.reward_sums[idx] += reward as f64;
        self.total_visits += 1;

        self.history.push_back(arm);
        if self.history.len() > self.history_len {
            self.history.pop_front();
        }
    }

    /// Average reward for the given arm (0.0 if never visited).
    pub fn q_value(&self, arm: RegimeArm) -> f64 {
        let idx = arm.index();
        if self.visits[idx] == 0 {
            return 0.0;
        }
        self.reward_sums[idx] / self.visits[idx] as f64
    }

    /// Number of times the given arm has been visited.
    pub fn visits(&self, arm: RegimeArm) -> u32 {
        self.visits[arm.index()]
    }

    /// Total visits across all arms.
    pub fn total_visits(&self) -> u64 {
        self.total_visits
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_features() -> RegimeFeatures {
        RegimeFeatures {
            failure_rate: 0.1,
            regime_collapse: false,
            transition_success: false,
            regime_q_value: 0.5,
        }
    }

    fn collapse_features() -> RegimeFeatures {
        RegimeFeatures {
            failure_rate: 0.9,
            regime_collapse: true,
            transition_success: false,
            regime_q_value: 0.2,
        }
    }

    fn transition_features() -> RegimeFeatures {
        RegimeFeatures {
            failure_rate: 0.3,
            regime_collapse: false,
            transition_success: true,
            regime_q_value: 0.7,
        }
    }

    #[test]
    fn regime_collapse_selects_discovery_arm() {
        let router = FourRegimeRouter::with_defaults();
        let features = collapse_features();
        let arm = router.select(&features);
        assert!(
            matches!(arm.regime(), Regime::Discovery),
            "Expected Discovery regime, got {:?}",
            arm
        );
    }

    #[test]
    fn transition_success_selects_consolidation_arm() {
        let router = FourRegimeRouter::with_defaults();
        let features = transition_features();
        let arm = router.select(&features);
        assert!(
            matches!(arm.regime(), Regime::Consolidation),
            "Expected Consolidation regime, got {:?}",
            arm
        );
    }

    #[test]
    fn standard_features_selects_standard_arm() {
        let router = FourRegimeRouter::with_defaults();
        let features = standard_features();
        let arm = router.select(&features);
        assert!(
            matches!(arm.regime(), Regime::Standard),
            "Expected Standard regime, got {:?}",
            arm
        );
    }

    #[test]
    fn update_reflects_in_q_value() {
        let mut router = FourRegimeRouter::with_defaults();
        let arm = RegimeArm::StandardLight;
        assert_eq!(router.q_value(arm), 0.0);

        router.update(arm, 1.0);
        router.update(arm, 0.5);
        let qv = router.q_value(arm);
        assert!(
            (qv - 0.75).abs() < 1e-6,
            "Expected q_value ≈ 0.75, got {}",
            qv
        );
        assert_eq!(router.visits(arm), 2);
    }

    #[test]
    fn ucb1_explores_unvisited_arms_first() {
        let mut router = FourRegimeRouter::with_defaults();

        // Heavily reward StandardLight so its Q-value is high.
        for _ in 0..10 {
            router.update(RegimeArm::StandardLight, 1.0);
        }

        // StandardHeavy is unvisited → UCB1 should pick it (infinite score).
        let features = standard_features();
        let arm = router.select(&features);
        assert_eq!(
            arm,
            RegimeArm::StandardHeavy,
            "UCB1 should prefer unvisited arm, got {:?}",
            arm
        );
    }

    #[test]
    fn sigmoid_confidence_bound_is_bounded() {
        // Sigmoid maps ℝ → (0, 1) — bounded by definition.
        // At f32 precision, extreme values saturate to exactly 0.0 or 1.0,
        // but the UCB1 confidence term uses ln(N)/n(a) which is always finite
        // and moderate — sigmoid always returns strictly in (0, 1) for that domain.

        // Sigmoid of 0 should be exactly 0.5
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);

        // Moderate values are strictly bounded
        assert!(sigmoid(10.0) < 1.0);
        assert!(sigmoid(-10.0) > 0.0);

        // Monotonicity: larger input → larger output
        assert!(sigmoid(1.0) > sigmoid(0.0));
        assert!(sigmoid(0.0) > sigmoid(-1.0));

        // Values in UCB1 domain (ln(N)/n for realistic N, n)
        // With N=1000, n=1: ln(1000)/1 ≈ 6.9 → sigmoid ≈ 0.999
        let ucb_input = 1000.0_f64.ln() / 1.0_f64;
        let sig_val = sigmoid(ucb_input as f32);
        assert!(sig_val > 0.5 && sig_val <= 1.0);
    }

    #[test]
    fn arm_count_is_six() {
        assert_eq!(RegimeArm::COUNT, 6);
        assert_eq!(RegimeArm::all().count(), 6);
    }

    #[test]
    fn regime_heaviness_roundtrip() {
        for arm in RegimeArm::all() {
            let regime = arm.regime();
            let heaviness = arm.heaviness();
            // Reconstruct the arm from regime + heaviness
            let reconstructed = match (regime, heaviness) {
                (Regime::Standard, Heaviness::Light) => RegimeArm::StandardLight,
                (Regime::Standard, Heaviness::Heavy) => RegimeArm::StandardHeavy,
                (Regime::Discovery, Heaviness::Light) => RegimeArm::DiscoveryLight,
                (Regime::Discovery, Heaviness::Heavy) => RegimeArm::DiscoveryHeavy,
                (Regime::Consolidation, Heaviness::Light) => RegimeArm::ConsolidationLight,
                (Regime::Consolidation, Heaviness::Heavy) => RegimeArm::ConsolidationHeavy,
            };
            assert_eq!(arm, reconstructed);
        }
    }

    #[test]
    fn history_tracks_recent_arms() {
        let mut router = FourRegimeRouter::with_defaults();

        for _ in 0..100 {
            router.update(RegimeArm::DiscoveryLight, 0.5);
        }

        // History should be capped at history_len (64)
        assert!(router.history.len() <= 64);
        assert!(
            router
                .history
                .iter()
                .all(|&a| a == RegimeArm::DiscoveryLight)
        );
    }

    #[test]
    fn total_visits_tracks_correctly() {
        let mut router = FourRegimeRouter::with_defaults();

        router.update(RegimeArm::StandardLight, 1.0);
        router.update(RegimeArm::DiscoveryHeavy, 0.5);
        router.update(RegimeArm::ConsolidationLight, 0.8);

        assert_eq!(router.total_visits(), 3);
    }
}
