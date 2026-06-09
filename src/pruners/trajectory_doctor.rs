//! TrajectoryDoctor — failure localization in DDTree paths (Plan 223, Phase 3).
//!
//! Traces back from rejected output to first violation, enabling
//! localized repair instead of full retry.
//!
//! ## Components
//!
//! - [`BracketTrajectoryDoctor`] — bracket-depth failure localization
//! - [`HoareTrajectoryDoctor`] — predicate-based failure localization via [`HoarePruner`]
//! - [`FailureEpisodeStore`] — persists failure sites indexed by prompt hash for flywheel learning
//!
//! [`HoarePruner`]: crate::pruners::hoare_pruner::HoarePruner

use std::collections::HashMap;

/// A specific failure site in a DDTree trajectory.
#[derive(Clone, Debug)]
pub struct FailureSite {
    /// Depth in DDTree where violation was detected.
    pub depth: usize,
    /// Token index where violation was detected.
    pub token_idx: usize,
    /// Description of violated predicate.
    pub violated_predicate: String,
    /// Alternative tokens that would have satisfied the predicate.
    pub alternatives: Vec<String>,
}

impl FailureSite {
    pub fn new(depth: usize, token_idx: usize, violated: &str, alts: Vec<String>) -> Self {
        Self {
            depth,
            token_idx,
            violated_predicate: violated.to_string(),
            alternatives: alts,
        }
    }
}

// ── FailureEpisodeStore ──────────────────────────────────────────────

/// Store for failure sites from TrajectoryDoctor analysis.
/// Enables Episode DB flywheel — past failures inform future constraint synthesis.
#[derive(Clone, Debug, Default)]
pub struct FailureEpisodeStore {
    /// Map from prompt_hash → list of failure sites.
    failures: HashMap<u64, Vec<FailureSite>>,
}

impl FailureEpisodeStore {
    /// Record a failure site for a given prompt hash.
    pub fn record(&mut self, prompt_hash: u64, site: FailureSite) {
        self.failures.entry(prompt_hash).or_default().push(site);
    }

    /// Look up past failure sites for a given prompt hash.
    pub fn lookup(&self, prompt_hash: u64) -> &[FailureSite] {
        self.failures.get(&prompt_hash).map_or(&[], Vec::as_slice)
    }

    /// Clear all stored failure sites.
    pub fn clear(&mut self) {
        self.failures.clear();
    }

    /// Number of distinct prompts with recorded failures.
    pub fn len(&self) -> usize {
        self.failures.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.failures.is_empty()
    }
}

// ── HoareTrajectoryDoctor ─────────────────────────────────────────────

/// TrajectoryDoctor that uses [`HoarePruner`] predicate tracking to localize failures.
///
/// Replays tokens through a HoarePruner; when a predicate is violated, records
/// the [`FailureSite`] with the violated predicate description.
///
/// [`HoarePruner`]: crate::pruners::hoare_pruner::HoarePruner
#[derive(Clone, Debug)]
pub struct HoareTrajectoryDoctor {
    /// Predicates to check during replay.
    predicates: Vec<crate::pruners::hoare_pruner::Predicate>,
}

impl HoareTrajectoryDoctor {
    /// Create a new HoareTrajectoryDoctor with the given predicates.
    pub fn new(predicates: Vec<crate::pruners::hoare_pruner::Predicate>) -> Self {
        Self { predicates }
    }
}

impl TrajectoryDoctor for HoareTrajectoryDoctor {
    fn localize_failure(&self, tokens: &[String], depth_limit: usize) -> Option<FailureSite> {
        let mut pruner = crate::pruners::hoare_pruner::HoarePruner::new(self.predicates.clone());

        for (idx, token) in tokens.iter().enumerate() {
            let depth = idx.min(depth_limit);
            if !pruner.propagate(token) {
                return Some(FailureSite::new(
                    depth,
                    idx,
                    &format!(
                        "predicate violated at depth {} (bracket_depth={})",
                        depth,
                        pruner.state().bracket_depth
                    ),
                    vec![],
                ));
            }
        }
        None
    }
}

/// TrajectoryDoctor trait for failure localization.
pub trait TrajectoryDoctor {
    /// Given a rejected trajectory, find the first failure site.
    fn localize_failure(&self, tokens: &[String], depth_limit: usize) -> Option<FailureSite>;
}

/// Simple bracket-tracking TrajectoryDoctor.
#[derive(Clone, Debug, Default)]
pub struct BracketTrajectoryDoctor {
    /// Maximum allowed bracket nesting depth.
    pub max_depth: u32,
}

impl BracketTrajectoryDoctor {
    pub fn new(max_depth: u32) -> Self {
        Self { max_depth }
    }
}

impl TrajectoryDoctor for BracketTrajectoryDoctor {
    fn localize_failure(&self, tokens: &[String], _depth_limit: usize) -> Option<FailureSite> {
        let mut depth = 0u32;
        for (idx, token) in tokens.iter().enumerate() {
            match token.as_str() {
                "(" | "[" | "{" => {
                    depth += 1;
                    if depth > self.max_depth {
                        return Some(FailureSite::new(
                            idx,
                            idx,
                            &format!("bracket depth {} exceeds max {}", depth, self.max_depth),
                            vec![")".to_string(), "]".to_string(), "}".to_string()],
                        ));
                    }
                }
                ")" | "]" | "}" => {
                    depth = depth.saturating_sub(1);
                }
                _ => {}
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_localize_bracket_failure() {
        let doctor = BracketTrajectoryDoctor::new(2);
        let tokens: Vec<String> = ["(", "(", "(", "x", ")"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let failure = doctor.localize_failure(&tokens, 10);
        assert!(failure.is_some());
        let f = failure.unwrap();
        assert_eq!(f.token_idx, 2);
        assert_eq!(f.depth, 2);
    }

    #[test]
    fn test_no_failure() {
        let doctor = BracketTrajectoryDoctor::new(3);
        let tokens: Vec<String> = ["(", "(", "x", ")", ")"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(doctor.localize_failure(&tokens, 10).is_none());
    }

    // ── Task 1: FailureEpisodeStore tests ─────────────────────────────

    #[test]
    fn test_failure_episode_store_record_and_lookup() {
        let mut store = FailureEpisodeStore::default();
        let hash = 0xDEAD_BEEF_u64;

        // Empty initially
        assert!(store.lookup(hash).is_empty());
        assert!(store.is_empty());

        // Record a failure
        store.record(
            hash,
            FailureSite::new(2, 5, "bracket overflow", vec![")".to_string()]),
        );
        assert_eq!(store.len(), 1);

        let sites = store.lookup(hash);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].depth, 2);
        assert_eq!(sites[0].token_idx, 5);

        // Different hash has no entries
        assert!(store.lookup(0xCAFE_F00D).is_empty());
    }

    #[test]
    fn test_failure_episode_store_multiple_sites() {
        let mut store = FailureEpisodeStore::default();
        let hash = 42_u64;

        store.record(hash, FailureSite::new(1, 3, "depth exceeded", vec![]));
        store.record(hash, FailureSite::new(4, 7, "unmatched close", vec![]));

        let sites = store.lookup(hash);
        assert_eq!(sites.len(), 2);
    }

    #[test]
    fn test_failure_episode_store_clear() {
        let mut store = FailureEpisodeStore::default();
        store.record(1, FailureSite::new(0, 0, "x", vec![]));
        store.record(2, FailureSite::new(0, 0, "y", vec![]));
        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
    }

    // ── Task 2: HoareTrajectoryDoctor tests ────────────────────────────

    #[test]
    fn test_hoare_trajectory_doctor_localizes_bracket_overflow() {
        use crate::pruners::hoare_pruner::Predicate;

        let doctor = HoareTrajectoryDoctor::new(vec![Predicate::BracketDepthLe(2)]);
        let tokens: Vec<String> = ["(", "(", "(", "x", ")"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let failure = doctor.localize_failure(&tokens, 10);
        assert!(failure.is_some());
        let f = failure.unwrap();
        // Third '(' pushes depth to 3, violating BracketDepthLe(2)
        assert_eq!(f.token_idx, 2);
    }

    #[test]
    fn test_hoare_trajectory_doctor_valid_trajectory() {
        use crate::pruners::hoare_pruner::Predicate;

        let doctor = HoareTrajectoryDoctor::new(vec![Predicate::BracketDepthLe(5)]);
        let tokens: Vec<String> = ["(", "(", "x", ")", ")"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        assert!(doctor.localize_failure(&tokens, 10).is_none());
    }

    #[test]
    fn test_hoare_trajectory_doctor_respects_depth_limit() {
        use crate::pruners::hoare_pruner::Predicate;

        let doctor = HoareTrajectoryDoctor::new(vec![Predicate::BracketDepthLe(1)]);
        let tokens: Vec<String> = ["(", "(", "x"].iter().map(|s| s.to_string()).collect();

        let f = doctor.localize_failure(&tokens, 3).unwrap();
        // idx=1 but depth_limit=3, so depth=min(1,3)=1
        assert_eq!(f.token_idx, 1);
        assert_eq!(f.depth, 1);
    }

    // ── Task 3: rejected trajectory localization ──────────────────────

    #[test]
    fn test_localize_rejected_trajectory_finds_correct_depth() {
        // max_depth=2, trajectory: ["(", "(", "(", "x", ")"]
        // Third '(' at index 2 exceeds max_depth=2 → depth=2, token_idx=2
        let doctor = BracketTrajectoryDoctor::new(2);
        let tokens: Vec<String> = ["(", "(", "(", "x", ")"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let failure = doctor.localize_failure(&tokens, 10);
        assert!(
            failure.is_some(),
            "rejected trajectory must produce a failure site"
        );
        let f = failure.unwrap();
        assert_eq!(
            f.depth, 2,
            "failure depth should be 2 (index where depth exceeds max)"
        );
        assert_eq!(f.token_idx, 2, "failure token_idx should be 2 (third '(')");
    }

    #[test]
    fn test_valid_trajectory_returns_none() {
        let doctor = BracketTrajectoryDoctor::new(2);
        // Properly nested, depth never exceeds 2
        let tokens: Vec<String> = ["(", "x", ")", "(", "x", ")"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        assert!(doctor.localize_failure(&tokens, 10).is_none());
    }

    // ── Task 4: before vs after localized repair ──────────────────────

    #[test]
    fn test_localized_repair_saves_regeneration_cost() {
        // Scenario: a rejected trajectory of 10 tokens where bracket depth
        // overflows at index 3. Without localization, you retry from scratch
        // (cost = full length). With localization, you only re-generate from
        // the failure point (cost = remaining tokens).
        let doctor = BracketTrajectoryDoctor::new(2);
        let tokens: Vec<String> = ["(", "(", "(", "x", ",", "y", ")", ")", ")", ";"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let failure = doctor.localize_failure(&tokens, 10);
        assert!(failure.is_some());
        let f = failure.unwrap();
        assert_eq!(f.token_idx, 2);

        // Before: full retry cost = tokens.len() = 10
        let full_retry_cost = tokens.len();

        // After: localized repair cost = tokens.len() - f.token_idx = 8
        // (only re-generate from the failure point onward)
        let localized_repair_cost = tokens.len() - f.token_idx;

        assert_eq!(full_retry_cost, 10);
        assert_eq!(localized_repair_cost, 8);
        assert!(
            localized_repair_cost < full_retry_cost,
            "localized repair must be cheaper than full retry"
        );

        // The prefix [0..token_idx) is preserved — no need to re-generate it.
        let preserved_prefix = &tokens[..f.token_idx];
        assert_eq!(preserved_prefix, &["(", "("]);
    }

    #[test]
    fn test_repair_with_episode_store_flywheel() {
        use crate::pruners::hoare_pruner::Predicate;

        // Round-trip: localize failure → store → lookup → verify site matches
        let mut store = FailureEpisodeStore::default();
        let prompt_hash = 12345_u64;

        // First run: localize and store
        let doctor = HoareTrajectoryDoctor::new(vec![Predicate::BracketDepthLe(1)]);
        let tokens: Vec<String> = ["(", "(", "x"].iter().map(|s| s.to_string()).collect();
        let failure = doctor.localize_failure(&tokens, 10).unwrap();
        store.record(prompt_hash, failure.clone());

        // Second run: lookup past failure for same prompt
        let past = store.lookup(prompt_hash);
        assert_eq!(past.len(), 1);
        assert_eq!(
            past[0].token_idx, 1,
            "past failure site should match original"
        );

        // We can use this to inform future constraint synthesis — only repair
        // from the known failure point instead of full retry.
        let repair_start = past[0].token_idx;
        assert_eq!(repair_start, 1);
        assert!(repair_start < tokens.len());
    }
}
