//! ROPD Rubric Modelless Distillation — structured multi-criteria reward without LLM.
//!
//! Distills ROPD's rubric-based scoring into our modelless stack. Replaces scalar
//! [`HintDelta`](super::g_zero::types::HintDelta) with structured [`RubricVector`]
//! — multi-criteria reward without LLM judges. Template rubrics + WASM validators
//! provide per-criterion scoring at inference speed (~µs).
//!
//! # Key Insight
//!
//! ROPD's rubric = (criterion, weight) pairs scored by binary pass/fail.
//! Our `Validator` trait already does binary + graded validation.
//! The gap: our reward is scalar δ, ROPD's is a weighted vector.
//! This module vectorizes the reward signal while keeping everything modelless.
//!
//! # Components
//!
//! - [`RubricCriterion`] / [`RubricTemplate`] — fixed criteria per domain (no LLM)
//! - [`RubricVector`] — structured multi-criteria score (replaces scalar δ)
//! - [`RubricScorer`] — trait for scoring responses against rubric templates
//! - [`RubricGatedAbsorbCompress`] — absorb-compress gated by rubric vector
//! - [`RubricBanditPruner`] — bandit using rubric-weighted scores as reward
//!
//! # Multi-Reference Requirement (from ablation)
//!
//! ROPD ablation (Table 6) shows m=4→m=1 costs **−17.94 pts** — the single biggest impact.
//! Single reference over-anchors rubric to one solution trajectory.
//! Always use M ≥ 2 references for gap computation.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "ropd_rubric")]`.
//! Feature: `ropd_rubric = ["bandit"]` in `Cargo.toml`.
//!
//! **Source:** [ROPD: Rubric-Guided On-Policy Distillation](https://arxiv.org/abs/2506.xxxxx)

pub mod rubric_absorb;
pub mod rubric_bandit;
pub mod scorer;
pub mod template;
pub mod types;

pub use rubric_absorb::{RubricGatedAbsorbCompress, RubricGatedConfig};
pub use rubric_bandit::{RubricBanditConfig, RubricBanditPruner};
pub use scorer::{
    PatternRule, PatternScorer, RubricScorer, ScoreResult, score_with_references,
    score_with_references_id,
};
pub use template::{RubricCriterion, RubricTemplate};
pub use types::RubricVector;
