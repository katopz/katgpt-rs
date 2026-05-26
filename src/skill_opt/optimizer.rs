//! Skill optimizer trait and supporting types.
//!
//! Game-specific implementations of [`SkillOptimizer`] and [`Benchmark`] live in
//! the `riir-ai` crate. This module defines the interface contract.

use super::edit::SkillEdit;
use super::gate::ValidationGate;

/// Minimal scored trajectory for text-space optimization.
#[derive(Debug, Clone)]
pub struct ScoredTrajectory {
    /// Identifier for the task/game instance.
    pub task_id: String,
    /// Numeric score (higher is better).
    pub score: f64,
    /// Execution trace — raw text for the optimizer to analyse.
    pub trace: String,
    /// Whether the task was completed successfully.
    pub is_success: bool,
}

/// Trait for skill optimizers.
///
/// Implementations analyse scored trajectories and propose edits to the skill document.
/// Game-specific implementations live in `riir-ai`.
pub trait SkillOptimizer {
    /// Propose a batch of edits given recent trajectories, the current skill, a budget,
    /// and a buffer of previously rejected edits (negative examples).
    fn propose_edits(
        &self,
        trajectories: &[ScoredTrajectory],
        current_skill: &str,
        edit_budget: usize,
        rejected_buffer: &[super::gate::RejectedEdit],
    ) -> Vec<SkillEdit>;

    /// Validate a candidate skill against a benchmark and return the gate result.
    fn validate(
        &self,
        candidate_skill: &str,
        current_score: f64,
        benchmark: &mut dyn Benchmark,
    ) -> ValidationGate;
}

/// Benchmark trait for validation.
///
/// Evaluates a skill document and returns a numeric score.
/// Game-specific implementations live in `riir-ai`.
pub trait Benchmark {
    /// Evaluate the given skill and return a score (higher is better).
    fn evaluate(&mut self, skill: &str) -> f64;
}
