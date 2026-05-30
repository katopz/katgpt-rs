//! Edit operations and skill edit struct for text-space skill optimization.

use serde::{Deserialize, Serialize};

/// Kind of text edit to apply to a skill document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EditOp {
    /// Append content to end of the skill document.
    Append,
    /// Insert content immediately after the target text.
    InsertAfter,
    /// Replace the target text with new content.
    Replace,
    /// Delete the target text.
    Delete,
}

/// Origin of an edit proposal — determines priority and trust level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EditSource {
    /// Proposed from a failed trajectory — low confidence.
    Failure,
    /// Proposed from a successful trajectory — high confidence.
    Success,
    /// Proposed from a slow update cycle — medium confidence.
    SlowUpdate,
    /// Proposed by a meta-skill that analyses other edits.
    MetaSkill,
}

/// A single proposed edit to a skill document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillEdit {
    /// Which edit operation to perform.
    pub op: EditOp,
    /// Target text to locate (None for Append).
    pub target: Option<String>,
    /// New content to insert/replace/append.
    pub content: String,
    /// Number of trajectories supporting this edit (higher = more confident).
    pub support_count: usize,
    /// Where this edit came from.
    pub source: EditSource,
}
