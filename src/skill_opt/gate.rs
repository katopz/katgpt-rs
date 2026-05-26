//! Validation gate — accepts or rejects candidate skills based on score delta.

use serde::{Deserialize, Serialize};

use super::edit::SkillEdit;

/// Result of validating a candidate skill against the current skill.
#[derive(Debug, Clone)]
pub struct ValidationGate {
    /// Whether the candidate skill was accepted (score improved).
    pub accepted: bool,
    /// Score of the candidate skill.
    pub candidate_score: f64,
    /// Score of the current skill.
    pub current_score: f64,
    /// Score improvement (candidate - current). Positive means improvement.
    pub delta: f64,
}

impl ValidationGate {
    /// Create a new validation gate. Accepts if `candidate_score > current_score`.
    pub fn new(candidate_score: f64, current_score: f64) -> Self {
        let delta = candidate_score - current_score;
        Self {
            accepted: delta > 0.0,
            candidate_score,
            current_score,
            delta,
        }
    }
}

/// Record of an edit that was rejected by the validation gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedEdit {
    /// The edit that was proposed.
    pub edit: SkillEdit,
    /// How much the score changed (negative for rejections).
    pub score_delta: f64,
    /// Failure patterns detected in the resulting skill.
    pub failure_patterns: Vec<String>,
    /// Training epoch when the edit was proposed.
    pub epoch: usize,
    /// Step within the epoch when the edit was proposed.
    pub step: usize,
}
