//! Verify-Refine (V-R) Loop — Plan 206.
//!
//! Iterative generate → verify → extract failures → re-generate loop.
//! Each round, the generator produces candidates, the verifier checks them,
//! and failure patterns are extracted into `SynthesizedConstraint`s for the next round.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ VrLoop                                       │
//! │                                              │
//! │  round 0: generate → verify                  │
//! │            ↓ accepted? → done                │
//! │            ↓ rejected  → extract failures    │
//! │                         → accumulate constraints
//! │  round 1: generate(constraints) → verify     │
//! │            ↓ ...repeat until accepted or     │
//! │              max_rounds exhausted             │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! Feature-gated behind `egcs`.

use super::episode_pruner::SynthesizedConstraint;

// ── VrLoopResult ──────────────────────────────────────────────────

/// Result of a V-R loop execution.
#[cfg(feature = "egcs")]
#[derive(Clone, Debug)]
pub struct VrLoopResult {
    /// Accepted candidates (passed all rounds).
    pub accepted: Vec<Vec<usize>>,
    /// Number of refinement rounds executed.
    pub rounds: usize,
    /// Whether the loop converged (found valid candidates).
    pub converged: bool,
    /// Rejection reasons per round.
    pub rejection_log: Vec<String>,
}

// ── VrRoundFeedback ───────────────────────────────────────────────

/// Feedback from a single verification round.
#[cfg(feature = "egcs")]
#[derive(Clone, Debug)]
pub struct VrRoundFeedback {
    /// Token positions that were rejected.
    pub rejected_positions: Vec<usize>,
    /// Token indices that failed verification.
    pub rejected_tokens: Vec<usize>,
    /// Human-readable failure description.
    pub failure_description: String,
}

// ── VrVerifier Trait ──────────────────────────────────────────────

/// Trait for verifying candidate sequences.
///
/// Abstracts over compiler feedback, test runners, formal verifiers, etc.
/// Returns `Ok(())` if the candidate is valid, `Err(feedback)` otherwise.
#[cfg(feature = "egcs")]
pub trait VrVerifier: Send + Sync {
    /// Verify a candidate sequence.
    fn verify(&self, candidate: &[usize]) -> Result<(), VrRoundFeedback>;
}

// ── VrGenerator Trait ─────────────────────────────────────────────

/// Generator trait for the V-R loop.
///
/// Produces candidate token sequences given accumulated constraints.
/// Decoupled from `SpeculativeGenerator` to keep the V-R loop self-contained
/// within the `egcs` feature without pulling in the full speculative machinery.
#[cfg(feature = "egcs")]
pub trait VrGenerator: Send + Sync {
    /// Generate candidates given accumulated constraints and vocab size.
    fn generate(
        &mut self,
        constraints: &[SynthesizedConstraint],
        vocab_size: usize,
        seq_len: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<Vec<usize>>;
}

// ── Failure Extraction ────────────────────────────────────────────

/// Extract `SynthesizedConstraint`s from verification feedback.
///
/// For each rejected position, creates a constraint that disallows the rejected
/// token at that position. This guides the generator away from known failures.
#[cfg(feature = "egcs")]
fn extract_constraints(feedback: &VrRoundFeedback) -> Vec<SynthesizedConstraint> {
    let mut constraints = Vec::with_capacity(feedback.rejected_positions.len());

    for (&pos, &token) in feedback
        .rejected_positions
        .iter()
        .zip(feedback.rejected_tokens.iter())
    {
        constraints.push(SynthesizedConstraint {
            position_range: (pos, pos + 1),
            allowed_tokens: Vec::new(),
            disallowed_tokens: vec![token],
        });
    }

    constraints
}

// ── VrLoop ────────────────────────────────────────────────────────

/// Verify-Refine loop: iteratively generate → verify → extract failures → re-generate.
///
/// Each round:
/// 1. Generator produces candidates
/// 2. Verifier checks them
/// 3. If accepted → return result (converged)
/// 4. If rejected → extract failure patterns, inject as constraints
/// 5. Re-generate with augmented constraints
/// 6. Repeat until accepted or `max_rounds` reached
///
/// Feature-gated behind `egcs`.
#[cfg(feature = "egcs")]
pub struct VrLoop<G, V>
where
    G: VrGenerator,
    V: VrVerifier,
{
    /// Generator for producing candidates.
    generator: G,
    /// Verifier for checking candidates.
    verifier: V,
    /// Maximum refinement rounds (default: 3).
    max_rounds: usize,
    /// Sequence length for generated candidates.
    seq_len: usize,
    /// Candidates per round.
    candidates_per_round: usize,
    /// Constraints accumulated across rounds.
    accumulated_constraints: Vec<SynthesizedConstraint>,
}

#[cfg(feature = "egcs")]
impl<G, V> VrLoop<G, V>
where
    G: VrGenerator,
    V: VrVerifier,
{
    /// Create a new V-R loop with the given generator and verifier.
    ///
    /// Defaults: `max_rounds = 3`, `seq_len = 8`, `candidates_per_round = 4`.
    pub fn new(generator: G, verifier: V) -> Self {
        Self {
            generator,
            verifier,
            max_rounds: 3,
            seq_len: 8,
            candidates_per_round: 4,
            accumulated_constraints: Vec::new(),
        }
    }

    /// Set maximum refinement rounds.
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds.max(1);
        self
    }

    /// Set sequence length for generated candidates.
    pub fn with_seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len.max(1);
        self
    }

    /// Set number of candidates generated per round.
    pub fn with_candidates_per_round(mut self, n: usize) -> Self {
        self.candidates_per_round = n.max(1);
        self
    }

    /// Run the V-R loop.
    ///
    /// Generates candidates, verifies them, and iteratively refines constraints
    /// until an acceptable candidate is found or `max_rounds` is exhausted.
    pub fn run(&mut self, vocab_size: usize, rng: &mut fastrand::Rng) -> VrLoopResult {
        let mut result = VrLoopResult {
            accepted: Vec::new(),
            rounds: 0,
            converged: false,
            rejection_log: Vec::new(),
        };

        for round in 0..self.max_rounds {
            result.rounds = round + 1;

            let candidates = self.generator.generate(
                &self.accumulated_constraints,
                vocab_size,
                self.seq_len,
                rng,
            );

            for candidate in &candidates {
                match self.verifier.verify(candidate) {
                    Ok(()) => {
                        result.accepted.push(candidate.clone());
                        result.converged = true;
                        return result;
                    }
                    Err(feedback) => {
                        // Extract failure constraints for next round
                        let new_constraints = extract_constraints(&feedback);
                        self.accumulated_constraints.extend(new_constraints);

                        result
                            .rejection_log
                            .push(feedback.failure_description.clone());
                    }
                }
            }
        }

        result
    }

    /// Reset accumulated constraints (e.g., between independent prompts).
    pub fn reset(&mut self) {
        self.accumulated_constraints.clear();
    }

    /// Number of accumulated constraints across all rounds.
    pub fn constraint_count(&self) -> usize {
        self.accumulated_constraints.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifier that accepts everything.
    struct AcceptAllVerifier;

    impl VrVerifier for AcceptAllVerifier {
        fn verify(&self, _candidate: &[usize]) -> Result<(), VrRoundFeedback> {
            Ok(())
        }
    }

    /// Verifier that accepts sequences where all tokens are even.
    struct EvenTokensVerifier;

    impl VrVerifier for EvenTokensVerifier {
        fn verify(&self, candidate: &[usize]) -> Result<(), VrRoundFeedback> {
            let mut rejected_positions = Vec::new();
            let mut rejected_tokens = Vec::new();

            for (pos, &token) in candidate.iter().enumerate() {
                match token % 2 == 0 {
                    true => continue,
                    false => {
                        rejected_positions.push(pos);
                        rejected_tokens.push(token);
                    }
                }
            }

            match rejected_positions.is_empty() {
                true => Ok(()),
                false => Err(VrRoundFeedback {
                    rejected_positions,
                    rejected_tokens,
                    failure_description: "odd tokens found".into(),
                }),
            }
        }
    }

    /// Verifier that always rejects.
    struct RejectAllVerifier;

    impl VrVerifier for RejectAllVerifier {
        fn verify(&self, candidate: &[usize]) -> Result<(), VrRoundFeedback> {
            let positions: Vec<usize> = (0..candidate.len()).collect();
            Err(VrRoundFeedback {
                rejected_positions: positions,
                rejected_tokens: candidate.to_vec(),
                failure_description: "always rejected".into(),
            })
        }
    }

    /// Generator that produces random token sequences, respecting disallowed constraints.
    struct RandomGenerator {
        /// Pre-seeded candidates for deterministic testing.
        /// Each call to `generate` pops from this queue.
        preloaded: std::sync::Mutex<Vec<Vec<Vec<usize>>>>,
    }

    impl RandomGenerator {
        fn new(rounds: Vec<Vec<Vec<usize>>>) -> Self {
            Self {
                preloaded: std::sync::Mutex::new(rounds),
            }
        }
    }

    impl VrGenerator for RandomGenerator {
        fn generate(
            &mut self,
            _constraints: &[SynthesizedConstraint],
            _vocab_size: usize,
            _seq_len: usize,
            _rng: &mut fastrand::Rng,
        ) -> Vec<Vec<usize>> {
            match self.preloaded.lock().unwrap().pop() {
                Some(candidates) => candidates,
                None => vec![vec![0; _seq_len]], // fallback
            }
        }
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_vr_loop_single_round() {
        // Verifier accepts everything → converges on round 1
        let generator = RandomGenerator::new(vec![vec![vec![1, 2, 3]]]);
        let mut vr = VrLoop::new(generator, AcceptAllVerifier).with_max_rounds(3);
        let mut rng = fastrand::Rng::new();

        let result = vr.run(10, &mut rng);

        assert!(result.converged);
        assert_eq!(result.rounds, 1);
        assert_eq!(result.accepted.len(), 1);
        assert!(result.rejection_log.is_empty());
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_vr_loop_multi_round() {
        // Round 0: all odd tokens → rejected
        // Round 1: all even tokens → accepted
        let generator = RandomGenerator::new(vec![
            vec![vec![2, 4, 6]], // round 1 (pop last)
            vec![vec![1, 3, 5]], // round 0
        ]);
        let mut vr = VrLoop::new(generator, EvenTokensVerifier).with_max_rounds(3);
        let mut rng = fastrand::Rng::new();

        let result = vr.run(10, &mut rng);

        assert!(result.converged);
        assert_eq!(result.rounds, 2);
        assert_eq!(result.accepted.len(), 1);
        assert_eq!(result.accepted[0], vec![2, 4, 6]);
        assert_eq!(result.rejection_log.len(), 1);
    }

    #[cfg(feature = "egcs")]
    #[test]
    fn test_vr_loop_exhausted() {
        // Verifier always rejects → all rounds exhausted
        let generator = RandomGenerator::new(vec![
            vec![vec![1, 2, 3]], // round 2
            vec![vec![4, 5, 6]], // round 1
            vec![vec![7, 8, 9]], // round 0
        ]);
        let mut vr = VrLoop::new(generator, RejectAllVerifier).with_max_rounds(3);
        let mut rng = fastrand::Rng::new();

        let result = vr.run(10, &mut rng);

        assert!(!result.converged);
        assert_eq!(result.rounds, 3);
        assert!(result.accepted.is_empty());
        assert_eq!(result.rejection_log.len(), 3);
        // Constraints should have accumulated across rounds
        assert!(vr.constraint_count() > 0);
    }
}
