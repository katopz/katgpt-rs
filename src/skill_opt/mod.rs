//! SkillOpt — text-space skill optimization for RL-driven skill editing.
//!
//! Provides a deterministic edit/apply pipeline for optimizing text skill documents:
//!
//! - [`edit`] — edit operations and the [`SkillEdit`] struct
//! - [`apply`] — deterministic text patching engine with budget and protected sections
//! - [`gate`] — validation gate that accepts/rejects candidates by score delta
//! - [`schedule`] — edit budget schedules (constant, linear, cosine, autonomous)
//! - [`buffer`] — FIFO ring buffer for rejected edits (negative examples)
//! - [`optimizer`] — [`SkillOptimizer`] trait and [`Benchmark`] trait for game-specific impls
//!
//! # Module layout
//!
//! ```text
//! edit ─► apply ─► gate ─► buffer ─► optimizer
//!         ▲                      │
//!         └── protected section  └── JSONL persistence
//! ```

// ── Submodules ─────────────────────────────────────────────────

/// Edit operations (`EditOp`, `EditSource`) and the `SkillEdit` struct.
pub mod edit;

/// Deterministic text patching engine — `apply_edits`, `ApplyResult`.
pub mod apply;

/// Validation gate — `ValidationGate`, `RejectedEdit`.
pub mod gate;

/// Edit budget schedule — `EditBudgetSchedule`.
pub mod schedule;

/// Rejected edit buffer — `RejectedEditBuffer` with JSONL persistence.
pub mod buffer;

/// Skill optimizer trait — `SkillOptimizer`, `Benchmark`, `ScoredTrajectory`.
pub mod optimizer;

// ── Re-exports ─────────────────────────────────────────────────

pub use apply::{ApplyResult, apply_edits};
pub use buffer::RejectedEditBuffer;
pub use edit::{EditOp, EditSource, SkillEdit};
pub use gate::{RejectedEdit, ValidationGate};
pub use optimizer::{Benchmark, ScoredTrajectory, SkillOptimizer};
pub use schedule::EditBudgetSchedule;
