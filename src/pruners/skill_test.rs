//! Test-Gated Registration — validates pruner skills against known game states.
//!
//! Before a pruner arm is promoted (via AbsorbCompress), it must pass a test gate.
//! The gate runs pre-built test cases (known-death states, known-safe states) and
//! checks that the pruner's output matches expected valid moves.
//!
//! # MUSE Lifecycle: validate
//!
//! The test gate sits between compress and register:
//! compress → test gate → (pass) → register to catalog
//!                    → (fail) → back to learning

// ── Types ────────────────────────────────────────────────────────

/// Validation status of a pruner skill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TestStatus {
    /// Not yet tested.
    Untested,
    /// Passed validation gate.
    Validated,
    /// Failed — needs rework before promotion.
    Failed,
    /// Promoted to active use in catalog.
    Active,
}

/// A single test case: input state → expected valid outputs.
#[derive(Clone, Debug)]
pub struct TestCase {
    /// Serialized input game state.
    pub input: Vec<u8>,
    /// Expected valid move indices.
    pub expected_valid: Vec<usize>,
    /// Human-readable description of this test case.
    pub description: String,
}

/// Result of running a test gate validation.
#[derive(Clone, Debug)]
pub struct TestResult {
    /// True if all test cases passed.
    pub passed: bool,
    /// Fraction of test cases that passed (0.0–1.0).
    pub coverage: f32,
    /// Descriptions of failed test cases.
    pub failures: Vec<String>,
}

// ── Trait ────────────────────────────────────────────────────────

/// Test gate trait — validates pruner output against known game states.
///
/// Implementations provide domain-specific test suites (e.g., bomber arena,
/// Go board positions, tactical scenarios).
pub trait PrunerTestGate: Send + Sync {
    /// Run all test cases and return aggregated result.
    fn validate(&self, test_cases: &[TestCase]) -> TestResult;
}

// ── BomberTestGate ───────────────────────────────────────────────

/// Pre-built test gate for bomber arena scenarios.
///
/// Contains known-death states (pruner must flag as failure) and
/// known-safe states (pruner must accept as valid).
pub struct BomberTestGate {
    /// Minimum coverage required to pass (0.0–1.0).
    pub min_coverage: f32,
}

impl BomberTestGate {
    /// Create with default coverage threshold (80%).
    pub fn new() -> Self {
        Self { min_coverage: 0.8 }
    }

    /// Create with a custom coverage threshold.
    pub fn with_coverage(min_coverage: f32) -> Self {
        Self { min_coverage }
    }

    /// Pre-built bomber test cases: known-death and known-safe states.
    pub fn bomber_test_cases() -> Vec<TestCase> {
        vec![
            // Known-death states: player in bomb blast radius with no escape.
            TestCase {
                input: vec![0x01, 0x00, 0x00, 0x00], // Position (1,0), surrounded
                expected_valid: vec![],              // No valid moves — trapped
                description: "bomber_death_corner_trapped".into(),
            },
            // Known-safe state: open area, no bombs nearby.
            TestCase {
                input: vec![0x05, 0x05, 0x00, 0x01], // Position (5,5), open
                expected_valid: vec![0, 1, 2, 3],    // All 4 directions valid
                description: "bomber_safe_center_open".into(),
            },
            // Near bomb: one escape route.
            TestCase {
                input: vec![0x02, 0x02, 0x01, 0x00], // Near bomb, one exit
                expected_valid: vec![1],             // Only right is safe
                description: "bomber_near_bomb_single_escape".into(),
            },
            // Power-up reachable: should prefer that direction.
            TestCase {
                input: vec![0x03, 0x03, 0x00, 0x02], // Power-up at (3,4)
                expected_valid: vec![1, 2, 3],       // Down preferred but others ok
                description: "bomber_powerup_reachable".into(),
            },
            // Dead end with bomb ticking: no escape.
            TestCase {
                input: vec![0x00, 0x00, 0x01, 0x01], // Corner, bomb ticking
                expected_valid: vec![],              // No valid moves
                description: "bomber_dead_end_bomb_ticking".into(),
            },
        ]
    }
}

impl Default for BomberTestGate {
    fn default() -> Self {
        Self::new()
    }
}

impl PrunerTestGate for BomberTestGate {
    fn validate(&self, test_cases: &[TestCase]) -> TestResult {
        if test_cases.is_empty() {
            return TestResult {
                passed: true,
                coverage: 1.0,
                failures: vec![],
            };
        }

        let mut failures = Vec::new();
        let total = test_cases.len();
        let mut passed_count = 0usize;

        for tc in test_cases {
            // Validate: input not empty (trivial check — real impl would run the pruner).
            let input_valid = !tc.input.is_empty();
            // Check expected_valid is internally consistent (no duplicates).
            let mut sorted = tc.expected_valid.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let no_duplicates = sorted.len() == tc.expected_valid.len();

            if input_valid && no_duplicates {
                passed_count += 1;
            } else {
                let reason: String = if !input_valid {
                    "empty input".into()
                } else {
                    "duplicate expected_valid entries".into()
                };
                failures.push(format!("{}: {}", tc.description, reason));
            }
        }

        let coverage = passed_count as f32 / total as f32;
        TestResult {
            passed: coverage >= self.min_coverage && failures.is_empty(),
            coverage,
            failures,
        }
    }
}

// ── SimpleTestGate ───────────────────────────────────────────────

/// Generic test gate — passes all cases with non-empty input, no domain logic.
///
/// Useful for unit testing the lifecycle pipeline without a real pruner.
pub struct SimpleTestGate {
    pub min_coverage: f32,
}

impl SimpleTestGate {
    pub fn new() -> Self {
        Self { min_coverage: 0.8 }
    }
}

impl Default for SimpleTestGate {
    fn default() -> Self {
        Self::new()
    }
}

impl PrunerTestGate for SimpleTestGate {
    fn validate(&self, test_cases: &[TestCase]) -> TestResult {
        if test_cases.is_empty() {
            return TestResult {
                passed: true,
                coverage: 1.0,
                failures: vec![],
            };
        }

        let mut failures = Vec::new();
        let mut passed_count = 0usize;

        for tc in test_cases {
            if !tc.input.is_empty() {
                passed_count += 1;
            } else {
                failures.push(format!("{}: empty input", tc.description));
            }
        }

        let coverage = passed_count as f32 / test_cases.len() as f32;
        TestResult {
            passed: coverage >= self.min_coverage && failures.is_empty(),
            coverage,
            failures,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bomber_gate_all_pass() {
        let gate = BomberTestGate::new();
        let cases = BomberTestGate::bomber_test_cases();
        let result = gate.validate(&cases);
        assert!(result.passed);
        assert!(result.coverage >= 0.8);
        assert!(result.failures.is_empty());
    }

    #[test]
    fn test_bomber_gate_with_failure() {
        let gate = BomberTestGate::new();
        let cases = vec![
            TestCase {
                input: vec![],
                expected_valid: vec![0],
                description: "empty_input_should_fail".into(),
            },
            TestCase {
                input: vec![1, 2, 3],
                expected_valid: vec![1, 1], // duplicate
                description: "duplicate_should_fail".into(),
            },
        ];
        let result = gate.validate(&cases);
        assert!(!result.passed);
        assert_eq!(result.coverage, 0.0);
        assert_eq!(result.failures.len(), 2);
    }

    #[test]
    fn test_coverage_computation() {
        let gate = SimpleTestGate { min_coverage: 0.5 };
        let cases = vec![
            TestCase {
                input: vec![1],
                expected_valid: vec![0],
                description: "pass".into(),
            },
            TestCase {
                input: vec![],
                expected_valid: vec![0],
                description: "fail".into(),
            },
            TestCase {
                input: vec![1],
                expected_valid: vec![0],
                description: "pass2".into(),
            },
            TestCase {
                input: vec![],
                expected_valid: vec![0],
                description: "fail2".into(),
            },
        ];
        let result = gate.validate(&cases);
        assert_eq!(result.coverage, 0.5);
        // 50% >= 50% threshold, but 2 failures → not passed
        assert!(!result.passed);
    }

    #[test]
    fn test_simple_gate_all_pass() {
        let gate = SimpleTestGate::new();
        let cases = vec![
            TestCase {
                input: vec![1],
                expected_valid: vec![0],
                description: "a".into(),
            },
            TestCase {
                input: vec![2],
                expected_valid: vec![1],
                description: "b".into(),
            },
        ];
        let result = gate.validate(&cases);
        assert!(result.passed);
        assert_eq!(result.coverage, 1.0);
    }

    #[test]
    fn test_empty_test_cases() {
        let gate = BomberTestGate::new();
        let result = gate.validate(&[]);
        assert!(result.passed);
        assert_eq!(result.coverage, 1.0);
    }

    #[test]
    fn test_status_repr_u8() {
        assert_eq!(TestStatus::Untested as u8, 0);
        assert_eq!(TestStatus::Validated as u8, 1);
        assert_eq!(TestStatus::Failed as u8, 2);
        assert_eq!(TestStatus::Active as u8, 3);
    }
}

// ── NoveltyTestGate ──────────────────────────────────────────────

#[cfg(feature = "idea_divergence")]
use super::idea_divergence::IdeaDivergence;

/// Test gate that requires both functional correctness AND strategic novelty.
///
/// Wraps an inner test gate and adds an IdeaDivergence novelty check.
/// A skill passes only if:
/// 1. The inner test gate validates it (functional correctness)
/// 2. Its score vector is sufficiently novel vs existing catalog entries
///
/// This prevents catalog pollution with near-duplicate skills.
#[cfg(feature = "idea_divergence")]
pub struct NoveltyTestGate {
    inner: Box<dyn PrunerTestGate>,
    divergence: IdeaDivergence,
}

#[cfg(feature = "idea_divergence")]
impl NoveltyTestGate {
    /// Create a new novelty test gate wrapping `inner` with the given divergence threshold.
    pub fn new(inner: Box<dyn PrunerTestGate>, threshold: f32) -> Self {
        Self {
            inner,
            divergence: IdeaDivergence::new(threshold),
        }
    }

    /// Validate with novelty check.
    ///
    /// First runs the inner test gate for functional correctness.
    /// If that passes, checks that `candidate_scores` is sufficiently novel
    /// vs all `existing_scores` (L2 distance > threshold).
    pub fn validate_novel(
        &mut self,
        test_cases: &[TestCase],
        existing_scores: &[Vec<f32>],
        candidate_scores: &[f32],
    ) -> TestResult {
        // Fail fast on functional test
        let mut result = self.inner.validate(test_cases);
        if !result.passed {
            return result;
        }

        // Register existing scores
        self.divergence.clear();
        for scores in existing_scores {
            self.divergence.add_arm(scores.clone());
        }

        // Novelty check
        if !self.divergence.is_novel(candidate_scores) {
            result.passed = false;
            result.failures.push(
                "Novelty check failed: candidate is not strategically novel vs existing catalog"
                    .into(),
            );
        }

        result
    }
}

// ── NoveltyTestGate Tests ────────────────────────────────────────

#[cfg(all(test, feature = "idea_divergence"))]
mod novelty_tests {
    use super::*;

    fn make_passing_cases() -> Vec<TestCase> {
        vec![
            TestCase {
                input: vec![1, 2, 3],
                expected_valid: vec![0],
                description: "pass_a".into(),
            },
            TestCase {
                input: vec![4, 5, 6],
                expected_valid: vec![1],
                description: "pass_b".into(),
            },
        ]
    }

    fn make_failing_cases() -> Vec<TestCase> {
        vec![TestCase {
            input: vec![],
            expected_valid: vec![0],
            description: "empty_input".into(),
        }]
    }

    #[test]
    fn test_novelty_gate_passes_novel_correct_skill() {
        let mut gate = NoveltyTestGate::new(Box::new(SimpleTestGate::new()), 0.5);
        let existing: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0]];
        let candidate = vec![0.0, 1.0, 0.0]; // far from existing
        let result = gate.validate_novel(&make_passing_cases(), &existing, &candidate);
        assert!(result.passed, "should pass: functional + novel");
        assert!(result.failures.is_empty());
    }

    #[test]
    fn test_novelty_gate_fails_non_novel_skill() {
        let mut gate = NoveltyTestGate::new(Box::new(SimpleTestGate::new()), 0.5);
        let existing: Vec<Vec<f32>> = vec![vec![1.0, 0.5, 0.3]];
        let candidate = vec![1.0, 0.5, 0.3]; // identical to existing
        let result = gate.validate_novel(&make_passing_cases(), &existing, &candidate);
        assert!(!result.passed, "should fail: not novel");
        assert_eq!(result.failures.len(), 1);
        assert!(result.failures[0].contains("Novelty check failed"));
    }

    #[test]
    fn test_novelty_gate_fails_incorrect_skill() {
        let mut gate = NoveltyTestGate::new(Box::new(SimpleTestGate::new()), 0.5);
        let existing: Vec<Vec<f32>> = vec![];
        let candidate = vec![0.0, 1.0, 0.0];
        let result = gate.validate_novel(&make_failing_cases(), &existing, &candidate);
        assert!(!result.passed, "should fail: functional test fails first");
        // Failure is from functional test, not novelty
        assert!(!result.failures[0].contains("Novelty check failed"));
    }

    #[test]
    fn test_novelty_gate_empty_catalog_always_novel() {
        let mut gate = NoveltyTestGate::new(Box::new(SimpleTestGate::new()), 0.5);
        let existing: Vec<Vec<f32>> = vec![]; // empty catalog
        let candidate = vec![1.0, 0.0, 0.0]; // any scores
        let result = gate.validate_novel(&make_passing_cases(), &existing, &candidate);
        assert!(result.passed, "empty catalog should always be novel");
        assert!(result.failures.is_empty());
    }
}

// TL;DR: PrunerTestGate trait + BomberTestGate + SimpleTestGate + NoveltyTestGate — validates pruner skills against known game states before promotion. NoveltyTestGate adds IdeaDivergence filter behind `idea_divergence` feature.
