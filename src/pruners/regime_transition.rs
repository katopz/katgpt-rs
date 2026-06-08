//! Regime-Transition Inference — self-revising discovery (Plan 215, GOAT candidate).
//!
//! Detects when the DDTree exploration regime has exhausted its current pruner set
//! and needs to transition to a new regime (new pruners, new strategies). Provides:
//!
//! - **T1**: `CollapseClassifier` — classifies DDTree failures as Search (keep exploring)
//!   or Regime (need new pruners). Uses uniform-failure-depth heuristic.
//! - **T2**: `RegimeTransitionGate` — gates admission of candidate pruners via
//!   MDL (Minimum Description Length) check and correctness validation.
//! - **T3**: `ProvenanceChain` — tamper-evident audit trail for the AbsorbCompress cycle,
//!   with blake3 commitment hashes and schema-transport verification.
//!
//! # Feature Gate
//!
//! Entire module is behind `#[cfg(feature = "regime_transition")]`.
//! Depends on: `and_or_dtree`, `bandit`, `decision_trace`, `fol_constraints`, `rule_extraction`.

#![cfg(feature = "regime_transition")]

use blake3::Hasher;

use super::decision_trace::DecisionTrace;

// ── Sigmoid ────────────────────────────────────────────────────────

/// Sigmoid function: `1 / (1 + exp(-x))`. Bounded to (0, 1).
/// Used in tests and available for future gating logic.
#[cfg(test)]
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ════════════════════════════════════════════════════════════════════
// T1: CollapseClassifier
// ════════════════════════════════════════════════════════════════════

/// What to do when DDTree exploration encounters widespread failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CollapseType {
    /// Keep exploring — failures are scattered, no clear pattern.
    Search = 0,
    /// Need new pruners — failures cluster at the same depth (regime collapse).
    Regime = 1,
}

/// Lightweight stats from DDTree exploration, used for collapse classification.
#[derive(Clone, Debug, Default)]
pub struct DDTreeStats {
    /// Total number of branches explored.
    pub total_branches: u32,
    /// Number of branches that failed (pruned to dead-end).
    pub failed_branches: u32,
    /// Depth at which each failure occurred.
    pub failure_depths: Vec<u32>,
    /// Maximum depth reached across all branches.
    pub max_depth: u32,
}

/// Trait for classifying whether DDTree failures indicate a regime collapse.
///
/// Implementations examine failure patterns and decide whether the current
/// pruner set is exhausted (Regime) or exploration should continue (Search).
pub trait CollapseClassifier: Send + Sync {
    /// Classify the current DDTree exploration state.
    fn classify(&self, stats: &DDTreeStats) -> CollapseType;
}

/// Classifier that detects regime collapse when all failures cluster at the same depth.
///
/// If the standard deviation of failure depths is within `tolerance` of zero,
/// the failures are "uniform" and the current pruner set is likely exhausted.
/// Otherwise, failures are scattered and exploration should continue.
pub struct RegimeCollapseClassifier {
    /// Maximum allowed standard deviation in failure depths for Regime classification.
    /// Default: 1.0 (depths within ±1 of each other).
    pub tolerance: f64,
}

impl RegimeCollapseClassifier {
    /// Create a new classifier with the given tolerance.
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Compute the standard deviation of failure depths.
    /// Returns 0.0 for empty or single-element sets.
    fn failure_depth_std(&self, stats: &DDTreeStats) -> f64 {
        match stats.failure_depths.len() {
            0 | 1 => 0.0,
            n => {
                let mean = stats.failure_depths.iter().map(|&d| d as f64).sum::<f64>() / n as f64;
                let variance = stats
                    .failure_depths
                    .iter()
                    .map(|&d| {
                        let diff = d as f64 - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / n as f64;
                variance.sqrt()
            }
        }
    }
}

impl Default for RegimeCollapseClassifier {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl CollapseClassifier for RegimeCollapseClassifier {
    fn classify(&self, stats: &DDTreeStats) -> CollapseType {
        // No failures → keep searching
        if stats.failed_branches == 0 || stats.failure_depths.is_empty() {
            return CollapseType::Search;
        }

        // All failures at the same depth (within tolerance) → regime collapse
        let std = self.failure_depth_std(stats);
        if std <= self.tolerance {
            return CollapseType::Regime;
        }

        CollapseType::Search
    }
}

// ════════════════════════════════════════════════════════════════════
// T2: RegimeTransitionGate
// ════════════════════════════════════════════════════════════════════

/// Result of evaluating a candidate pruner through the regime transition gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GateResult {
    /// Candidate passed both correctness and information checks.
    Accept = 0,
    /// Candidate failed one or more checks.
    Reject = 1,
}

/// Gates admission of candidate pruners into a new regime.
///
/// A candidate must pass two checks:
/// 1. **Correctness**: can process basic input without panicking (sandbox check).
/// 2. **Information (MDL)**: reduces description length by more than `admission_cost_bits`.
///
/// The MDL gate uses a simplified description length model:
/// - `DL = rules_applied * 16.0 + alternatives_rejected * 8.0`
/// - A candidate that explains `rules_explained` rules reduces DL by `rules_explained * 16.0`.
/// - Accept iff `DL_old - DL_new > admission_cost_bits`.
pub struct RegimeTransitionGate {
    /// Minimum description-length reduction (in bits) to admit a candidate.
    /// Default: 32.0 bits.
    pub admission_cost_bits: f64,
}

impl RegimeTransitionGate {
    /// Create a new gate with the given admission cost threshold.
    pub fn new(admission_cost_bits: f64) -> Self {
        Self {
            admission_cost_bits,
        }
    }

    /// Evaluate a candidate pruner against a decision trace.
    ///
    /// `rules_explained` is the number of rules the candidate would explain
    /// (i.e., how many applied rules it can account for).
    pub fn evaluate(&self, trace: &DecisionTrace, rules_explained: usize) -> GateResult {
        // Correctness check: sandbox validation (placeholder — real impl would use wasmi).
        if !self.sandbox_check() {
            return GateResult::Reject;
        }

        // Information check: MDL gate.
        let dl_old = description_length(trace);
        let reduction = rules_explained as f64 * 16.0;
        let dl_new = dl_old - reduction;

        if dl_new < dl_old - self.admission_cost_bits {
            GateResult::Accept
        } else {
            GateResult::Reject
        }
    }

    /// Placeholder sandbox check — validates the candidate can process basic input.
    ///
    /// Real implementation would use wasmi (WASM sandboxing) for safe execution.
    /// For now, always returns `true` since we're feature-gated and don't have
    /// actual WASM sandboxing available.
    fn sandbox_check(&self) -> bool {
        true
    }
}

impl Default for RegimeTransitionGate {
    fn default() -> Self {
        Self::new(32.0)
    }
}

/// Compute a simplified MDL (Minimum Description Length) estimate for a decision trace.
///
/// Model: `DL = rules_applied * 16.0 + alternatives_rejected * 8.0`
///
/// Each applied rule costs 16 bits (condition + action encoding).
/// Each rejected alternative costs 8 bits (simpler encoding, just the score delta).
pub fn description_length(trace: &DecisionTrace) -> f64 {
    trace.rules_applied.len() as f64 * 16.0 + trace.alternatives_rejected.len() as f64 * 8.0
}

// ════════════════════════════════════════════════════════════════════
// T3: ProvenanceChain
// ════════════════════════════════════════════════════════════════════

/// Pruner type classification for schema transport validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrunerType {
    /// Constraint-based pruner.
    Constraint = 0,
    /// Screening pruner.
    Screening = 1,
    /// First-order logic pruner.
    Fol = 2,
    /// Expression-based pruner.
    Expression = 3,
    /// Custom pruner with sub-type identifier.
    Custom(u8),
}

/// A single step in the provenance chain — records an AbsorbCompress episode.
#[derive(Clone, Debug)]
pub struct ProvenanceStep {
    /// Episode identifier (monotonically increasing).
    pub episode_id: u64,
    /// Reward from this episode.
    pub reward: f32,
    /// Which bandit arm was pulled.
    pub bandit_pull: usize,
    /// blake3 hash of this step's data for tamper detection.
    pub blake3_hash: [u8; 32],
}

/// Result of transporting a provenance chain to a new schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransportResult {
    /// All steps are valid under the new schema.
    AllValid,
    /// Some steps are invalid; contains indices of invalid steps.
    SomeInvalid(Vec<usize>),
}

/// Tamper-evident audit trail for the AbsorbCompress cycle.
///
/// Each step records episode metadata and a blake3 hash of the step data.
/// The chain's commitment hash is the blake3 of all concatenated step hashes,
/// providing a single digest that covers the entire history.
#[derive(Clone, Debug, Default)]
pub struct ProvenanceChain {
    pub steps: Vec<ProvenanceStep>,
}

impl ProvenanceChain {
    /// Record a new step in the provenance chain.
    ///
    /// Creates a `ProvenanceStep` with a blake3 hash of the step data:
    /// `hash(episode_id || reward_bits || bandit_pull)`
    pub fn record(&mut self, episode_id: u64, reward: f32, bandit_pull: usize) {
        let mut hasher = Hasher::new();
        hasher.update(&episode_id.to_le_bytes());
        hasher.update(&reward.to_le_bytes());
        hasher.update(&(bandit_pull as u64).to_le_bytes());
        let hash = hasher.finalize();

        self.steps.push(ProvenanceStep {
            episode_id,
            reward,
            bandit_pull,
            blake3_hash: hash.into(),
        });
    }

    /// Compute the commitment hash of the entire chain.
    ///
    /// Concatenates all step hashes and returns the blake3 of the result.
    /// This provides a single digest covering the entire provenance history.
    pub fn commitment_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        for step in &self.steps {
            hasher.update(&step.blake3_hash);
        }
        hasher.finalize().into()
    }

    /// Verify all step hashes match their data.
    ///
    /// Returns `true` iff every step's blake3 hash is consistent with its
    /// episode_id, reward, and bandit_pull values.
    pub fn verify(&self) -> bool {
        for step in &self.steps {
            let mut hasher = Hasher::new();
            hasher.update(&step.episode_id.to_le_bytes());
            hasher.update(&step.reward.to_le_bytes());
            hasher.update(&(step.bandit_pull as u64).to_le_bytes());
            let expected: [u8; 32] = hasher.finalize().into();
            if expected != step.blake3_hash {
                return false;
            }
        }
        true
    }

    /// Transport this provenance chain to a new pruner schema.
    ///
    /// Placeholder implementation: marks all steps as valid.
    /// Real implementation would check parameter compatibility between
    /// the old schema and `new_schema`.
    pub fn transport(&self, _new_schema: &[PrunerType]) -> TransportResult {
        TransportResult::AllValid
    }
}

// ════════════════════════════════════════════════════════════════════
// T4: AdversarialBreaker
// ════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::Mutex;

use katgpt_core::traits::ConstraintPruner;

/// Token sequence that failed validation, recorded for adversarial analysis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FailurePattern {
    /// Token sequence that failed validation.
    pub tokens: Vec<usize>,
    /// Depth at which failure occurred.
    pub failure_depth: usize,
}

/// Wrapper around any [`ConstraintPruner`] that tracks failure patterns and
/// generates synthetic edge cases for adversarial testing.
///
/// When the same failure pattern recurs enough times (≥ `threshold`), it
/// is considered "hot" — a systematic weakness worth probing further via
/// [`AdversarialBreaker::generate_synthetic`] perturbations.
pub struct AdversarialBreaker<P: ConstraintPruner> {
    inner: P,
    failure_counts: Mutex<HashMap<FailurePattern, u32>>,
    threshold: u32,
}

impl<P: ConstraintPruner> AdversarialBreaker<P> {
    /// Create a new wrapper with the given inner pruner and failure threshold.
    pub fn new(inner: P, threshold: u32) -> Self {
        Self {
            inner,
            failure_counts: Mutex::new(HashMap::new()),
            threshold,
        }
    }

    /// Create with the default threshold of 5.
    pub fn with_default_threshold(inner: P) -> Self {
        Self::new(inner, 5)
    }

    /// Record a failure and check if it exceeds the pattern threshold.
    /// Returns `true` if the pattern count *just* reached the threshold.
    pub fn record_failure(&self, pattern: FailurePattern) -> bool {
        let mut map = self.failure_counts.lock().unwrap();
        let count = map.entry(pattern).or_insert(0);
        *count += 1;
        *count == self.threshold
    }

    /// Generate synthetic edge cases from a failure pattern.
    ///
    /// Perturbs the failing token sequence by ±1 at each position,
    /// producing `2 * tokens.len()` variants.
    pub fn generate_synthetic(&self, pattern: &FailurePattern) -> Vec<Vec<usize>> {
        let mut variants = Vec::with_capacity(pattern.tokens.len() * 2);
        for i in 0..pattern.tokens.len() {
            let mut plus = pattern.tokens.clone();
            plus[i] = plus[i].wrapping_add(1);
            variants.push(plus);
            let mut minus = pattern.tokens.clone();
            minus[i] = minus[i].wrapping_sub(1);
            variants.push(minus);
        }
        variants
    }

    /// Return all failure patterns whose count has reached the threshold.
    pub fn hot_patterns(&self) -> Vec<FailurePattern> {
        let map = self.failure_counts.lock().unwrap();
        map.iter()
            .filter(|&(_, &count)| count >= self.threshold)
            .map(|(pattern, _)| pattern.clone())
            .collect()
    }
}

impl<P: ConstraintPruner> ConstraintPruner for AdversarialBreaker<P> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let result = self.inner.is_valid(depth, token_idx, parent_tokens);
        if !result {
            let mut tokens = parent_tokens.to_vec();
            tokens.push(token_idx);
            self.record_failure(FailurePattern {
                tokens,
                failure_depth: depth,
            });
        }
        result
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::rule_extractor::ExtractedRule;
    use super::*;

    // ── T1: CollapseClassifier Tests ─────────────────────────────────

    #[test]
    fn regime_classifier_uniform_failure_depths() {
        let classifier = RegimeCollapseClassifier::default();

        // All failures at depth 3 → std = 0 → Regime
        let stats = DDTreeStats {
            total_branches: 10,
            failed_branches: 5,
            failure_depths: vec![3, 3, 3, 3, 3],
            max_depth: 5,
        };
        assert_eq!(
            classifier.classify(&stats),
            CollapseType::Regime,
            "Uniform failure depths should classify as Regime"
        );

        // Failures within ±1 → std ≈ 0.5 → Regime (tolerance=1.0)
        let stats = DDTreeStats {
            total_branches: 10,
            failed_branches: 4,
            failure_depths: vec![3, 3, 4, 4],
            max_depth: 5,
        };
        assert_eq!(
            classifier.classify(&stats),
            CollapseType::Regime,
            "Failures within ±1 should classify as Regime"
        );
    }

    #[test]
    fn regime_classifier_scattered_failure_depths() {
        let classifier = RegimeCollapseClassifier::default();

        // Failures scattered across depths → std > 1.0 → Search
        let stats = DDTreeStats {
            total_branches: 10,
            failed_branches: 5,
            failure_depths: vec![1, 2, 4, 6, 8],
            max_depth: 10,
        };
        assert_eq!(
            classifier.classify(&stats),
            CollapseType::Search,
            "Scattered failure depths should classify as Search"
        );
    }

    #[test]
    fn regime_classifier_no_failures() {
        let classifier = RegimeCollapseClassifier::default();

        let stats = DDTreeStats {
            total_branches: 10,
            failed_branches: 0,
            failure_depths: vec![],
            max_depth: 5,
        };
        assert_eq!(
            classifier.classify(&stats),
            CollapseType::Search,
            "No failures should classify as Search"
        );
    }

    #[test]
    fn regime_classifier_single_failure() {
        let classifier = RegimeCollapseClassifier::default();

        let stats = DDTreeStats {
            total_branches: 10,
            failed_branches: 1,
            failure_depths: vec![3],
            max_depth: 5,
        };
        assert_eq!(
            classifier.classify(&stats),
            CollapseType::Regime,
            "Single failure (std=0) should classify as Regime"
        );
    }

    // ── T2: RegimeTransitionGate Tests ───────────────────────────────

    fn sample_rule(depth: usize, tok: usize, score: f32, support: u32) -> ExtractedRule {
        ExtractedRule::new(vec![(0, 1), (1, 2)], (depth, tok), score, support)
    }

    fn make_trace(n_applied: usize, n_rejected: usize) -> DecisionTrace {
        DecisionTrace {
            rules_applied: (0..n_applied).map(|i| sample_rule(2, i, 0.8, 5)).collect(),
            alternatives_rejected: (0..n_rejected).map(|i| sample_rule(2, i, 0.3, 1)).collect(),
            confidence: 0.8,
        }
    }

    #[test]
    fn gate_accepts_when_dl_reduction_exceeds_cost() {
        let gate = RegimeTransitionGate::new(32.0);
        // DL_old = 5 * 16.0 + 2 * 8.0 = 96.0
        // Candidate explains 4 rules → reduction = 64.0
        // DL_new = 96.0 - 64.0 = 32.0
        // 32.0 < 96.0 - 32.0 = 64.0 → Accept
        let trace = make_trace(5, 2);
        assert_eq!(
            gate.evaluate(&trace, 4),
            GateResult::Accept,
            "Candidate reducing DL by 64 bits should be accepted (cost=32)"
        );
    }

    #[test]
    fn gate_rejects_when_dl_reduction_below_cost() {
        let gate = RegimeTransitionGate::new(32.0);
        // DL_old = 5 * 16.0 + 2 * 8.0 = 96.0
        // Candidate explains 1 rule → reduction = 16.0
        // DL_new = 96.0 - 16.0 = 80.0
        // 80.0 < 96.0 - 32.0 = 64.0? No → Reject
        let trace = make_trace(5, 2);
        assert_eq!(
            gate.evaluate(&trace, 1),
            GateResult::Reject,
            "Candidate reducing DL by only 16 bits should be rejected (cost=32)"
        );
    }

    #[test]
    fn gate_rejects_zero_explained() {
        let gate = RegimeTransitionGate::default();
        let trace = make_trace(3, 1);
        assert_eq!(
            gate.evaluate(&trace, 0),
            GateResult::Reject,
            "Candidate explaining zero rules should be rejected"
        );
    }

    #[test]
    fn description_length_computation() {
        let trace = make_trace(3, 2);
        let dl = description_length(&trace);
        assert_eq!(
            dl,
            3.0 * 16.0 + 2.0 * 8.0,
            "DL should be 3*16 + 2*8 = 64.0, got {}",
            dl
        );
    }

    // ── T3: ProvenanceChain Tests ────────────────────────────────────

    #[test]
    fn provenance_commitment_hash_is_deterministic() {
        let mut chain = ProvenanceChain::default();
        chain.record(1, 0.5, 0);
        chain.record(2, 0.3, 1);
        chain.record(3, 0.8, 2);

        let hash1 = chain.commitment_hash();

        // Same operations → same hash
        let mut chain2 = ProvenanceChain::default();
        chain2.record(1, 0.5, 0);
        chain2.record(2, 0.3, 1);
        chain2.record(3, 0.8, 2);

        let hash2 = chain2.commitment_hash();
        assert_eq!(
            hash1, hash2,
            "Same sequence of records must produce identical commitment hash"
        );
    }

    #[test]
    fn provenance_verify_returns_true_for_valid_chain() {
        let mut chain = ProvenanceChain::default();
        chain.record(1, 0.5, 0);
        chain.record(2, 0.3, 1);
        chain.record(3, 0.8, 2);

        assert!(
            chain.verify(),
            "Freshly recorded chain must verify successfully"
        );
    }

    #[test]
    fn provenance_verify_detects_tampering() {
        let mut chain = ProvenanceChain::default();
        chain.record(1, 0.5, 0);

        // Tamper with reward
        chain.steps[0].reward = 0.99;
        assert!(!chain.verify(), "Tampered chain must fail verification");
    }

    #[test]
    fn provenance_empty_chain_commitment_and_verify() {
        let chain = ProvenanceChain::default();

        // Empty chain should verify (vacuously true)
        assert!(chain.verify(), "Empty chain should verify");

        // Empty chain commitment hash is just blake3 of nothing
        let hash = chain.commitment_hash();
        let expected: [u8; 32] = blake3::hash(&[]).into();
        assert_eq!(
            hash, expected,
            "Empty chain commitment should be blake3 of empty input"
        );
    }

    #[test]
    fn provenance_transport_same_schema_returns_all_valid() {
        let mut chain = ProvenanceChain::default();
        chain.record(1, 0.5, 0);
        chain.record(2, 0.3, 1);

        let schema = vec![PrunerType::Constraint, PrunerType::Fol];
        let result = chain.transport(&schema);
        assert_eq!(
            result,
            TransportResult::AllValid,
            "Placeholder transport should return AllValid"
        );
    }

    #[test]
    fn sigmoid_bounded() {
        for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let s = sigmoid(x);
            assert!(
                (0.0..=1.0).contains(&s),
                "sigmoid({}) = {} not in [0,1]",
                x,
                s
            );
        }
    }

    // ── T4: AdversarialBreaker Tests ──────────────────────────────────

    /// A pruner that rejects token 3 at any depth/parents.
    struct RejectThree;

    impl ConstraintPruner for RejectThree {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx != 3
        }
    }

    #[test]
    fn adversarial_record_failure_increments_count() {
        let inner = RejectThree;
        let ab = AdversarialBreaker::new(inner, 5);
        let pattern = FailurePattern {
            tokens: vec![1, 2, 3],
            failure_depth: 2,
        };
        assert!(!ab.record_failure(pattern.clone()));
        assert!(!ab.record_failure(pattern.clone()));
        assert!(!ab.record_failure(pattern.clone()));
        assert!(!ab.record_failure(pattern.clone()));
        // 5th call should return true (just reached threshold)
        assert!(ab.record_failure(pattern.clone()));
        // 6th should return false (already exceeded)
        assert!(!ab.record_failure(pattern));
    }

    #[test]
    fn adversarial_threshold_detection() {
        let inner = RejectThree;
        let ab = AdversarialBreaker::new(inner, 3);
        let p1 = FailurePattern {
            tokens: vec![1, 2],
            failure_depth: 1,
        };
        let p2 = FailurePattern {
            tokens: vec![4, 5],
            failure_depth: 1,
        };
        // p1 hits threshold
        ab.record_failure(p1.clone());
        ab.record_failure(p1.clone());
        assert!(ab.record_failure(p1.clone()));
        // p2 below threshold
        ab.record_failure(p2.clone());

        let hot = ab.hot_patterns();
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0], p1);
    }

    #[test]
    fn adversarial_synthetic_generation() {
        let inner = RejectThree;
        let ab = AdversarialBreaker::with_default_threshold(inner);
        let pattern = FailurePattern {
            tokens: vec![10, 20],
            failure_depth: 1,
        };
        let variants = ab.generate_synthetic(&pattern);
        // 2 tokens * 2 perturbations = 4 variants
        assert_eq!(variants.len(), 4);
        assert_eq!(variants[0], vec![11, 20]); // +1 on pos 0
        assert_eq!(variants[1], vec![9, 20]); // -1 on pos 0
        assert_eq!(variants[2], vec![10, 21]); // +1 on pos 1
        assert_eq!(variants[3], vec![10, 19]); // -1 on pos 1
    }

    #[test]
    fn adversarial_is_valid_delegates_and_records() {
        let ab = AdversarialBreaker::new(RejectThree, 2);
        // token 3 is rejected
        assert!(!ab.is_valid(0, 3, &[]));
        assert!(!ab.is_valid(1, 3, &[1]));
        // token 1 is accepted
        assert!(ab.is_valid(0, 1, &[]));

        // Two different patterns, each recorded once
        let hot = ab.hot_patterns();
        assert!(hot.is_empty());

        // Same pattern again should make it hot
        assert!(!ab.is_valid(0, 3, &[]));
        let hot = ab.hot_patterns();
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].tokens, vec![3]);
        assert_eq!(hot[0].failure_depth, 0);
    }

    #[test]
    fn adversarial_hot_patterns_returns_only_at_threshold() {
        let ab = AdversarialBreaker::new(RejectThree, 3);
        let p1 = FailurePattern {
            tokens: vec![3],
            failure_depth: 0,
        };
        let p2 = FailurePattern {
            tokens: vec![1, 3],
            failure_depth: 1,
        };
        // p1 to threshold
        ab.record_failure(p1.clone());
        ab.record_failure(p1.clone());
        ab.record_failure(p1.clone());
        // p2 below threshold
        ab.record_failure(p2.clone());

        let hot = ab.hot_patterns();
        assert_eq!(hot.len(), 1);
        assert!(hot.contains(&p1));
        assert!(!hot.contains(&p2));
    }
}
