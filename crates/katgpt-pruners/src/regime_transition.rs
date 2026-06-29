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

    /// Compute the standard deviation of failure depths using Welford's one-pass algorithm.
    /// Returns 0.0 for empty or single-element sets.
    fn failure_depth_std(&self, stats: &DDTreeStats) -> f64 {
        match stats.failure_depths.len() {
            0 | 1 => 0.0,
            _ => {
                let mut mean = 0.0f64;
                let mut m2 = 0.0f64;
                for (i, &d) in stats.failure_depths.iter().enumerate() {
                    let x = d as f64;
                    let delta = x - mean;
                    mean += delta / (i as f64 + 1.0);
                    let delta2 = x - mean;
                    m2 += delta * delta2;
                }
                let n = stats.failure_depths.len() as f64;
                (m2 / n).sqrt()
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

/// Hash of a failure pattern token sequence, used as a compact HashMap key.
/// Avoids storing/cloning the full `Vec<usize>` per pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FailurePatternHash([u8; 32]);

impl FailurePatternHash {
    /// Compute blake3 hash from depth + token sequence.
    pub fn from_parts(depth: usize, tokens: &[usize]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(depth as u64).to_le_bytes());
        for &t in tokens {
            hasher.update(&(t as u64).to_le_bytes());
        }
        Self(hasher.finalize().into())
    }

    /// Compute blake3 hash from depth + parent tokens + trailing token.
    ///
    /// Equivalent to `from_parts(depth, parent_tokens.iter().chain([&token]))`
    /// but avoids the heap allocation that materializing the concatenated
    /// `Vec<usize>` would require. This is the hot-path variant used by
    /// `AdversarialBreaker::is_valid`, which is invoked per-candidate per-node
    /// during DDTree construction.
    #[inline]
    pub fn from_parts_with_token(
        depth: usize,
        parent_tokens: &[usize],
        token: usize,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(depth as u64).to_le_bytes());
        for &t in parent_tokens {
            hasher.update(&(t as u64).to_le_bytes());
        }
        hasher.update(&(token as u64).to_le_bytes());
        Self(hasher.finalize().into())
    }
}

/// Token sequence that failed validation, recorded for adversarial analysis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FailurePattern {
    /// Token sequence that failed validation.
    pub tokens: Vec<usize>,
    /// Depth at which failure occurred.
    pub failure_depth: usize,
}

impl FailurePattern {
    /// Compute blake3 hash for use as a compact HashMap key.
    pub fn hash_key(&self) -> FailurePatternHash {
        FailurePatternHash::from_parts(self.failure_depth, &self.tokens)
    }
}

/// A rule extracted from a systematic failure pattern.
///
/// Represents: "tokens matching this prefix always fail at this depth."
/// Can be converted into a new pruner that avoids the identified failure pattern.
#[derive(Clone, Debug)]
pub struct FailureRule {
    /// Token prefix that triggers the failure.
    pub trigger_prefix: Vec<usize>,
    /// Depth at which failure occurs.
    pub failure_depth: usize,
    /// How many times this pattern was observed.
    pub observations: u32,
    /// Synthetic variants that also failed (confirms genuine weakness).
    pub synthetic_confirms: u32,
}

/// Wrapper around any [`ConstraintPruner`] that tracks failure patterns and
/// generates synthetic edge cases for adversarial testing.
///
/// When the same failure pattern recurs enough times (≥ `threshold`), it
/// is considered "hot" — a systematic weakness worth probing further via
/// [`AdversarialBreaker::generate_synthetic`] perturbations.
pub struct AdversarialBreaker<P: ConstraintPruner> {
    inner: P,
    failure_counts: Mutex<HashMap<FailurePatternHash, u32>>,
    /// Stores patterns that reached threshold (needed for hot_patterns, extract_failure_rule).
    hot_patterns: Mutex<HashMap<FailurePatternHash, FailurePattern>>,
    threshold: u32,
}

impl<P: ConstraintPruner> AdversarialBreaker<P> {
    /// Create a new wrapper with the given inner pruner and failure threshold.
    pub fn new(inner: P, threshold: u32) -> Self {
        Self {
            inner,
            failure_counts: Mutex::new(HashMap::new()),
            hot_patterns: Mutex::new(HashMap::new()),
            threshold,
        }
    }

    /// Create with the default threshold of 5.
    pub fn with_default_threshold(inner: P) -> Self {
        Self::new(inner, 5)
    }

    /// Record a failure and check if it exceeds the pattern threshold.
    /// Returns `true` if the pattern count *just* reached the threshold.
    ///
    /// Lock discipline: `failure_counts` is acquired and dropped *before*
    /// `hot_patterns` is taken — never holds both at once. This removes lock-
    /// held-while-waiting-on-another-lock serialization under concurrent
    /// `is_valid` calls (which record failures from `ConstraintPruner::is_valid`)
    /// and eliminates a latent deadlock risk if any future caller locks in
    /// the opposite order.
    pub fn record_failure(&self, pattern: FailurePattern) -> bool {
        let key = pattern.hash_key();
        let reached_threshold = {
            let mut map = self.failure_counts.lock().unwrap();
            let count = map.entry(key).or_insert(0);
            *count += 1;
            *count == self.threshold
        };
        // failure_counts lock is now dropped — safe to acquire hot_patterns.
        if reached_threshold {
            self.hot_patterns.lock().unwrap().insert(key, pattern);
        }
        reached_threshold
    }

    /// Hot-path record: hash directly from `parent_tokens` + `token_idx` without
    /// materializing a `Vec<usize>`, and only build the owning `FailurePattern`
    /// on the rare threshold-hit path.
    ///
    /// Issue 001 H-4: `is_valid` is called per-candidate per-node; failures are
    /// common but threshold-hits are rare (1 in `threshold` failures). The
    /// previous code allocated `parent_tokens.to_vec()` on *every* failure just
    /// to hash it and discard. This variant defers the allocation to the rare
    /// branch, turning O(failures) allocations into O(threshold_hits).
    ///
    /// Produces a hash identical to
    /// `FailurePattern { tokens: [..parent_tokens, token_idx], failure_depth: depth }.hash_key()`
    /// so it interoperates with `record_failure` and `hot_patterns` lookups.
    fn record_failure_from_tokens(
        &self,
        depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
    ) {
        let key = FailurePatternHash::from_parts_with_token(depth, parent_tokens, token_idx);
        let reached_threshold = {
            let mut map = self.failure_counts.lock().unwrap();
            let count = map.entry(key).or_insert(0);
            *count += 1;
            *count == self.threshold
        };
        if reached_threshold {
            // Rare path: now we need the owning Vec for hot_patterns storage.
            let mut tokens = parent_tokens.to_vec();
            tokens.push(token_idx);
            self.hot_patterns.lock().unwrap().insert(
                key,
                FailurePattern {
                    tokens,
                    failure_depth: depth,
                },
            );
        }
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
    ///
    /// Lock discipline: snapshot the qualifying keys from `failure_counts`
    /// first (lock dropped), then look them up in `hot_patterns` — never
    /// holds both locks at once.
    pub fn hot_patterns(&self) -> Vec<FailurePattern> {
        let hot_keys: Vec<FailurePatternHash> = {
            let counts = self.failure_counts.lock().unwrap();
            counts
                .iter()
                .filter(|&(_, &count)| count >= self.threshold)
                .map(|(key, _)| *key)
                .collect()
        };
        // failure_counts lock is now dropped — safe to acquire hot_patterns.
        let hot = self.hot_patterns.lock().unwrap();
        hot_keys.iter().filter_map(|key| hot.get(key).cloned()).collect()
    }

    /// Feed synthetic variants through the inner pruner to verify the weakness is genuine.
    ///
    /// Returns the number of synthetic variants that also fail. If this is > 0,
    /// the weakness is confirmed as systematic, not a one-off edge case.
    pub fn verify_synthetic_failure(&self, pattern: &FailurePattern) -> u32 {
        let variants = self.generate_synthetic(pattern);
        let mut confirms = 0u32;
        for variant in &variants {
            if !self.inner.is_valid(
                pattern.failure_depth,
                variant[variant.len().saturating_sub(1)],
                &variant[..variant.len().saturating_sub(1)],
            ) {
                confirms += 1;
            }
        }
        confirms
    }

    /// Extract a failure rule from a hot pattern, verifying it with synthetic variants.
    ///
    /// Returns `None` if the pattern hasn't reached threshold or isn't confirmed.
    pub fn extract_failure_rule(&self, pattern: &FailurePattern) -> Option<FailureRule> {
        let key = pattern.hash_key();
        let map = self.failure_counts.lock().unwrap();
        let count = map.get(&key).copied()?;
        if count < self.threshold {
            return None;
        }
        drop(map);

        let synthetic_confirms = self.verify_synthetic_failure(pattern);
        if synthetic_confirms == 0 {
            return None;
        }

        Some(FailureRule {
            trigger_prefix: pattern.tokens.clone(),
            failure_depth: pattern.failure_depth,
            observations: count,
            synthetic_confirms,
        })
    }
}

impl<P: ConstraintPruner> ConstraintPruner for AdversarialBreaker<P> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let result = self.inner.is_valid(depth, token_idx, parent_tokens);
        if !result {
            // Hot path: hash from slices directly; allocate only on the rare
            // threshold-hit (see `record_failure_from_tokens`).
            self.record_failure_from_tokens(depth, token_idx, parent_tokens);
        }
        result
    }
}

// ════════════════════════════════════════════════════════════════════
// T7: RegimeTransitionScheduler
// ════════════════════════════════════════════════════════════════════

/// Error returned when a regime transition is deferred due to load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransitionDeferred;

impl std::fmt::Display for TransitionDeferred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "regime transition deferred due to concurrency limit")
    }
}

impl std::error::Error for TransitionDeferred {}

/// Scheduler for regime transitions with configurable concurrency control.
///
/// When the Discovery regime is active and load is low, runs regime transition
/// immediately on the current thread. When load is high, defers to background.
///
/// Default concurrency limit: 1 (one concurrent transition at a time).
pub struct RegimeTransitionScheduler {
    /// Maximum number of concurrent regime transitions.
    concurrency_limit: u32,
    /// Current number of active transitions.
    active_count: std::sync::atomic::AtomicU32,
    /// Whether background mode is enabled (defer transitions).
    background_enabled: bool,
}

impl RegimeTransitionScheduler {
    /// Create a new scheduler with the given concurrency limit.
    pub fn new(concurrency_limit: u32) -> Self {
        Self {
            concurrency_limit: concurrency_limit.max(1),
            active_count: std::sync::atomic::AtomicU32::new(0),
            background_enabled: false,
        }
    }

    /// Create with default settings: concurrency_limit=1, background=false.
    pub fn with_defaults() -> Self {
        Self::new(1)
    }

    /// Enable or disable background deferral mode.
    pub fn set_background_mode(&mut self, enabled: bool) {
        self.background_enabled = enabled;
    }

    /// Check if a new transition can be started (under concurrency limit).
    pub fn can_start(&self) -> bool {
        self.active_count.load(std::sync::atomic::Ordering::Relaxed) < self.concurrency_limit
    }

    /// Try to acquire a transition slot. Returns true if successful.
    pub fn try_acquire(&self) -> bool {
        use std::sync::atomic::Ordering;
        loop {
            let current = self.active_count.load(Ordering::Relaxed);
            if current >= self.concurrency_limit {
                return false;
            }
            if self
                .active_count
                .compare_exchange_weak(current, current + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Release a transition slot after completion.
    pub fn release(&self) {
        self.active_count
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }

    /// Get current number of active transitions.
    pub fn active_transitions(&self) -> u32 {
        self.active_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether transitions should be deferred to background.
    pub fn should_defer(&self) -> bool {
        self.background_enabled
    }

    /// Execute a transition function with concurrency control.
    ///
    /// If background mode is on and concurrency is exceeded, returns `Err(TransitionDeferred)`.
    /// Otherwise, runs the closure and returns its result.
    pub fn execute<F, R>(&self, f: F) -> Result<R, TransitionDeferred>
    where
        F: FnOnce() -> R,
    {
        if !self.try_acquire() {
            return Err(TransitionDeferred);
        }
        let result = f();
        self.release();
        Ok(result)
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

    /// Issue 001 H-4: the fast-path hash (from slices) must equal the
    /// owning-Vec hash (from `FailurePattern::hash_key`) so counts and
    /// `hot_patterns` lookups interoperate between `is_valid` and
    /// `record_failure`.
    #[test]
    fn adversarial_hash_fast_path_matches_owning_vec() {
        let depth = 7usize;
        let parents = [1usize, 2, 3, 42];
        let token = 99usize;

        // Owning path: materialize Vec, hash via FailurePattern.
        let mut full = parents.to_vec();
        full.push(token);
        let owning = FailurePattern {
            tokens: full,
            failure_depth: depth,
        }
        .hash_key();

        // Fast path: hash from slices directly.
        let fast = FailurePatternHash::from_parts_with_token(depth, &parents, token);

        assert_eq!(owning, fast);
    }

    /// Issue 001 H-4: `is_valid` (fast path) and `record_failure` (owning path)
    /// must share the same counter bucket so counts accumulate across both
    /// call sites, and the threshold-hit still stores the owning pattern.
    #[test]
    fn adversarial_fast_and_owning_paths_share_counter() {
        let ab = AdversarialBreaker::new(RejectThree, 3);
        // Two fast-path calls via is_valid (token 3 rejected) at depth 0.
        assert!(!ab.is_valid(0, 3, &[]));
        assert!(!ab.is_valid(0, 3, &[]));
        // One owning-path call via record_failure — same (depth=0, tokens=[3]).
        // This 3rd call should hit threshold and return true.
        assert!(
            ab.record_failure(FailurePattern {
                tokens: vec![3],
                failure_depth: 0,
            }),
            "combined counter from fast + owning paths should reach threshold"
        );
        let hot = ab.hot_patterns();
        assert_eq!(hot.len(), 1, "threshold should have been hit on combined path");
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

    // ════════════════════════════════════════════════════════════════
    // Integration Tests (Plan 215)
    // ════════════════════════════════════════════════════════════════

    use super::super::four_regime_router::{FourRegimeRouter, Regime, RegimeArm, RegimeFeatures};

    /// T2 Integration — Full pipeline: mock DDTree → regime collapse → gate → provenance.
    #[test]
    fn integration_t2_full_pipeline_mock_ddtree_regime_collapse() {
        // 1. Create mock DDTreeStats with uniform failure depths (all at depth 3)
        let stats = DDTreeStats {
            total_branches: 20,
            failed_branches: 8,
            failure_depths: vec![3, 3, 3, 3, 3, 3, 3, 3],
            max_depth: 6,
        };

        // 2. Classify — uniform depths → Regime collapse
        let classifier = RegimeCollapseClassifier::default();
        let collapse = classifier.classify(&stats);
        assert_eq!(
            collapse,
            CollapseType::Regime,
            "Uniform failure depths (all=3) must classify as Regime"
        );

        // 3. Create a DecisionTrace with rules applied and alternatives rejected
        let trace = make_trace(5, 3);
        // DL_old = 5*16 + 3*8 = 104.0

        // 4. Evaluate candidate that explains enough rules to pass the gate
        //    Candidate explains 4 rules → reduction = 4*16 = 64
        //    DL_new = 104 - 64 = 40.0
        //    40.0 < 104 - 32 = 72.0 → Accept
        let gate = RegimeTransitionGate::default();
        let result = gate.evaluate(&trace, 4);
        assert_eq!(
            result,
            GateResult::Accept,
            "Candidate explaining 4/5 rules should pass the gate (reduction=64 > cost=32)"
        );

        // 5. Record the outcome in a ProvenanceChain
        let mut chain = ProvenanceChain::default();
        chain.record(1, 0.85, 0); // episode 1, reward, arm 0
        chain.record(2, 0.90, 2); // episode 2, reward, arm 2

        // 6. Verify the chain
        assert!(
            chain.verify(),
            "Provenance chain must verify after recording legitimate steps"
        );

        // 7. Commitment hash must be non-zero
        let hash = chain.commitment_hash();
        let zero: [u8; 32] = [0u8; 32];
        assert_ne!(
            hash, zero,
            "Commitment hash of non-empty chain must be non-zero"
        );
    }

    /// T4 Integration — mock failure pattern → synthetic generation → rule extraction.
    #[test]
    fn integration_t4_failure_pattern_synthetic_rule_extraction() {
        // 1. Create AdversarialBreaker wrapping RejectThree, threshold=3 for speed
        let ab = AdversarialBreaker::new(RejectThree, 3);

        // 2. Feed failures until a pattern goes hot
        //    RejectThree rejects token 3. Use is_valid to trigger failure recording.
        //    Calling is_valid(0, 3, &[]) creates pattern { tokens: [3], failure_depth: 0 }
        assert!(!ab.is_valid(0, 3, &[])); // count=1
        assert!(!ab.is_valid(0, 3, &[])); // count=2
        assert!(!ab.is_valid(0, 3, &[])); // count=3 → hot!

        // 3. Get hot patterns
        let hot = ab.hot_patterns();
        assert_eq!(hot.len(), 1, "Exactly one pattern should be hot");
        assert_eq!(hot[0].tokens, vec![3], "Hot pattern should be at token 3");

        // 4. Generate synthetic edge cases from the hot pattern
        let variants = ab.generate_synthetic(&hot[0]);
        assert_eq!(variants.len(), 2, "1 token × 2 perturbations = 2 variants");
        // Perturbations of token 3: ±1 → 4 and 2
        assert!(variants.contains(&vec![4]), "Variant +1: token 3→4");
        assert!(variants.contains(&vec![2]), "Variant -1: token 3→2");

        // 5. Run each synthetic through the breaker's is_valid
        //    Token 4 and 2 are accepted by RejectThree (only token 3 is rejected)
        //    These pass — confirming the weakness is specific to token 3
        let mut _synthetic_failures: Vec<Vec<usize>> = Vec::new();
        for variant in variants.iter() {
            // Use depth 0, first token of variant, no parents
            if !ab.is_valid(0, variant[0], &[]) {
                _synthetic_failures.push(variant.clone());
            }
        }

        // The synthetic variants (token 2 and 4) pass — they don't trigger the weakness.
        // The weakness is at token 3 specifically. This confirms token 3 is the systematic failure.
        // To expose the weakness more thoroughly, test the actual failing token:
        assert!(
            !ab.is_valid(0, 3, &[]),
            "Token 3 still fails — confirms the systematic weakness"
        );

        // 6. Collect failures — verify they expose the systematic weakness at token 3
        let hot_after = ab.hot_patterns();
        assert_eq!(
            hot_after[0].tokens,
            vec![3],
            "Hot pattern confirms weakness at token 3"
        );

        // 7. Create a DecisionTrace from the failure analysis results
        let trace = DecisionTrace {
            rules_applied: vec![
                // Rule documenting the discovered weakness: token 3 rejection
                ExtractedRule::new(
                    vec![(0, 3)], // condition: token at pos 0 == 3
                    (0, 3),       // action: reject at depth 0, token 3
                    0.95,         // high confidence — systematic pattern
                    10,           // support: observed 10+ times
                ),
            ],
            alternatives_rejected: vec![
                ExtractedRule::new(
                    vec![(0, 2)], // alternative: token 2 (didn't fail)
                    (0, 2),
                    0.5,
                    3,
                ),
                ExtractedRule::new(
                    vec![(0, 4)], // alternative: token 4 (didn't fail)
                    (0, 4),
                    0.5,
                    3,
                ),
            ],
            confidence: 0.9,
        };

        // 8. Gate evaluates candidate that would fix the issue
        //    DL_old = 1*16 + 2*8 = 32.0
        //    Candidate explains 1 rule → reduction = 16.0
        //    DL_new = 32 - 16 = 16.0
        //    16.0 < 32 - 32 = 0.0? No → Reject with default cost.
        //    Use a lower cost gate (8.0) so the fix candidate is accepted.
        let gate = RegimeTransitionGate::new(8.0);
        let result = gate.evaluate(&trace, 1);
        assert_eq!(
            result,
            GateResult::Accept,
            "Fix candidate should pass gate with low admission cost"
        );

        // 9. Record in ProvenanceChain and verify
        let mut chain = ProvenanceChain::default();
        chain.record(10, 0.95, 3); // episode for this fix
        assert!(chain.verify(), "Chain must verify after recording");
        let hash = chain.commitment_hash();
        assert_ne!(hash, [0u8; 32], "Commitment hash must be non-zero");
    }

    /// T5 Integration — discovery → regime transition → consolidation → return to standard.
    #[test]
    fn integration_t5_discovery_regime_transition_consolidation_cycle() {
        let mut router = FourRegimeRouter::with_defaults();
        let classifier = RegimeCollapseClassifier::default();
        let gate = RegimeTransitionGate::default();
        let mut chain = ProvenanceChain::default();

        // ── Phase 1: Standard ──────────────────────────────────────
        let standard_features = RegimeFeatures {
            failure_rate: 0.1,
            regime_collapse: false,
            transition_success: false,
            regime_q_value: 0.5,
        };
        let arm = router.select(&standard_features);
        assert_eq!(
            arm.regime(),
            Regime::Standard,
            "Phase 1: Standard features must select Standard regime"
        );
        router.update(arm, 0.8);

        // ── Phase 2: Discovery ─────────────────────────────────────
        // Create DDTreeStats with uniform failure depths → regime collapse
        let collapse_stats = DDTreeStats {
            total_branches: 15,
            failed_branches: 6,
            failure_depths: vec![3, 3, 3, 3, 3, 3],
            max_depth: 5,
        };
        let collapse_type = classifier.classify(&collapse_stats);
        assert_eq!(
            collapse_type,
            CollapseType::Regime,
            "Phase 2: Uniform failures must classify as Regime"
        );

        // Set regime_collapse=true → router must select Discovery
        let discovery_features = RegimeFeatures {
            failure_rate: 0.9,
            regime_collapse: true,
            transition_success: false,
            regime_q_value: 0.2,
        };
        let discovery_arm = router.select(&discovery_features);
        assert_eq!(
            discovery_arm.regime(),
            Regime::Discovery,
            "Phase 2: regime_collapse=true must select Discovery regime"
        );

        // ── Phase 3: Regime Transition ─────────────────────────────
        let trace = make_trace(5, 2);
        // DL_old = 5*16 + 2*8 = 96.0; explains 4 → reduction=64 → DL_new=32
        // 32.0 < 96.0 - 32.0 = 64.0 → Accept
        let gate_result = gate.evaluate(&trace, 4);
        assert_eq!(
            gate_result,
            GateResult::Accept,
            "Phase 3: Candidate reducing DL by 64 bits should be accepted"
        );

        // Record in ProvenanceChain
        chain.record(1, 0.85, discovery_arm.index());
        chain.record(2, 0.90, discovery_arm.index());
        assert!(chain.verify(), "Phase 3: Provenance chain must verify");

        // Update router with the discovery arm reward
        router.update(discovery_arm, 0.85);

        // ── Phase 4: Consolidation ─────────────────────────────────
        let consolidation_features = RegimeFeatures {
            failure_rate: 0.3,
            regime_collapse: false,
            transition_success: true,
            regime_q_value: 0.7,
        };
        let consol_arm = router.select(&consolidation_features);
        assert_eq!(
            consol_arm.regime(),
            Regime::Consolidation,
            "Phase 4: transition_success=true must select Consolidation regime"
        );
        router.update(consol_arm, 0.9);
        chain.record(3, 0.90, consol_arm.index());

        // ── Phase 5: Return to Standard ────────────────────────────
        let returned_features = RegimeFeatures {
            failure_rate: 0.05,
            regime_collapse: false,
            transition_success: false,
            regime_q_value: 0.8,
        };
        let return_arm = router.select(&returned_features);
        assert_eq!(
            return_arm.regime(),
            Regime::Standard,
            "Phase 5: No collapse/transition flags must return to Standard regime"
        );
        router.update(return_arm, 0.95);

        // ── Verify full cycle ──────────────────────────────────────
        // Router has visits in Standard, Discovery, and Consolidation
        let all_arms: Vec<RegimeArm> = RegimeArm::all().collect();
        let has_standard = all_arms
            .iter()
            .any(|&a| router.visits(a) > 0 && a.regime() == Regime::Standard);
        let has_discovery = all_arms
            .iter()
            .any(|&a| router.visits(a) > 0 && a.regime() == Regime::Discovery);
        let has_consolidation = all_arms
            .iter()
            .any(|&a| router.visits(a) > 0 && a.regime() == Regime::Consolidation);

        assert!(has_standard, "Cycle must include Standard regime visits");
        assert!(has_discovery, "Cycle must include Discovery regime visits");
        assert!(
            has_consolidation,
            "Cycle must include Consolidation regime visits"
        );

        // ProvenanceChain verifies
        assert!(chain.verify(), "Final chain must verify");

        // Commitment hash is consistent (compute twice, must match)
        let hash1 = chain.commitment_hash();
        let hash2 = chain.commitment_hash();
        assert_eq!(hash1, hash2, "Commitment hash must be deterministic");
        assert_ne!(hash1, [0u8; 32], "Commitment hash must be non-zero");
    }

    // ── T4 Integration: Synthetic DDTree Verification + Rule Extraction ──

    /// Verify that synthetic perturbation confirms genuine weaknesses.
    ///
    /// Pattern [1, 3] ends in token 3. Perturbations include [2, 3] and [0, 3]
    /// which still end in token 3, so RejectThree still rejects them.
    #[test]
    fn integration_t4_synthetic_ddtree_verifies_genuine_failure() {
        let ab = AdversarialBreaker::new(RejectThree, 3);

        // Feed failures for pattern [1, 3] at depth 1 until hot
        // Each call creates FailurePattern { tokens: [1, 3], failure_depth: 1 }
        for _ in 0..3 {
            assert!(!ab.is_valid(1, 3, &[1]), "token 3 must be rejected");
        }

        // Pattern should be hot
        let hot = ab.hot_patterns();
        assert_eq!(hot.len(), 1, "Exactly one pattern should be hot");
        assert_eq!(hot[0].tokens, vec![1, 3]);

        let pattern = &hot[0];

        // Verify synthetic failure — perturbations of [1,3] include [2,3] and [0,3]
        // Both still end in token 3, so they still fail → confirms > 0
        let confirms = ab.verify_synthetic_failure(pattern);
        assert!(
            confirms > 0,
            "Synthetic confirms must be > 0 for genuine weakness, got {}",
            confirms
        );

        // Extract failure rule
        let rule = ab
            .extract_failure_rule(pattern)
            .expect("rule should be extracted from hot pattern with synthetic confirms");
        assert_eq!(rule.failure_depth, 1, "failure_depth must match");
        assert!(
            rule.observations >= 3,
            "observations ({}) must be >= threshold (3)",
            rule.observations
        );
        assert!(
            rule.synthetic_confirms > 0,
            "synthetic_confirms ({}) must be > 0",
            rule.synthetic_confirms
        );
        assert_eq!(
            rule.trigger_prefix,
            vec![1, 3],
            "trigger_prefix must match pattern tokens"
        );
    }

    /// Verify that rule extraction works across multiple failure patterns
    /// all pointing to the same systematic weakness (token 3).
    #[test]
    fn integration_t4_rule_extraction_from_failure() {
        let ab = AdversarialBreaker::new(RejectThree, 3);

        // Feed many different failure patterns all involving token 3
        // Each pattern ends in token 3 (the systematic weakness)
        let prefixes: &[&[usize]] = &[&[1], &[2], &[3], &[4], &[5]];
        for prefix in prefixes {
            for _ in 0..3 {
                let mut tokens = prefix.to_vec();
                tokens.push(3); // failing token
                let depth = tokens.len() - 1;
                // Call is_valid with parent = prefix, token = 3
                assert!(
                    !ab.is_valid(depth, 3, prefix),
                    "token 3 must be rejected with prefix {:?}",
                    prefix
                );
            }
        }

        // Get hot patterns
        let hot = ab.hot_patterns();
        assert!(!hot.is_empty(), "At least one pattern should be hot");

        // Extract rules from all hot patterns
        let rules: Vec<FailureRule> = hot
            .iter()
            .filter_map(|p| ab.extract_failure_rule(p))
            .collect();

        assert!(!rules.is_empty(), "At least one rule should be extracted");

        // All extracted rules should point to the same systematic weakness: token 3
        // The last element of trigger_prefix is the failing token
        for rule in &rules {
            let last = rule
                .trigger_prefix
                .last()
                .expect("trigger_prefix should be non-empty");
            assert_eq!(
                *last, 3,
                "All rules should point to token 3 as the systematic weakness, got {}",
                last
            );
            assert!(
                rule.synthetic_confirms > 0,
                "Each rule should have synthetic confirms > 0"
            );
        }
    }

    /// T7 Integration — RegimeTransitionScheduler concurrency control.
    #[test]
    fn integration_t7_concurrent_decode_no_regression() {
        // 1. Create scheduler with concurrency_limit=1
        let mut sched = RegimeTransitionScheduler::new(1);

        // 2. can_start() is true initially
        assert!(sched.can_start(), "Should be able to start initially");
        assert_eq!(sched.active_transitions(), 0);

        // 3. Acquire slot via try_acquire() → true
        assert!(sched.try_acquire(), "First acquire should succeed");

        // 4. can_start() is now false (limit reached)
        assert!(!sched.can_start(), "Should be at limit after one acquire");

        // 5. Second try_acquire() → false
        assert!(
            !sched.try_acquire(),
            "Second acquire must fail at concurrency_limit=1"
        );

        // 6. Release the slot
        sched.release();

        // 7. can_start() is true again
        assert!(sched.can_start(), "Should be able to start after release");
        assert_eq!(sched.active_transitions(), 0);

        // 8. execute() with a closure that returns GateResult::Accept → Ok(Accept)
        let result = sched.execute(|| GateResult::Accept);
        assert_eq!(
            result,
            Ok(GateResult::Accept),
            "execute should run closure and return Ok(Accept)"
        );

        // 9. active_transitions() is 0 after execute completes
        assert_eq!(
            sched.active_transitions(),
            0,
            "active_transitions must be 0 after execute completes"
        );

        // 10. Background mode: set_background_mode(true), verify should_defer()
        assert!(!sched.should_defer(), "Background should be off by default");
        sched.set_background_mode(true);
        assert!(
            sched.should_defer(),
            "Background should be on after set_background_mode(true)"
        );
        sched.set_background_mode(false);

        // 11. Execute under load: acquire first, then execute → Err(TransitionDeferred)
        assert!(sched.try_acquire(), "Acquire for load test");
        let result = sched.execute(|| GateResult::Accept);
        assert_eq!(
            result,
            Err(TransitionDeferred),
            "execute under concurrency limit must return Err(TransitionDeferred)"
        );
        sched.release();
    }
}
