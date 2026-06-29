//! SDAR Gated Distillation — Modelless Path.
//!
//! Adapts SDAR's token-level sigmoid gating pattern to our modelless distillation stack.
//! Applies asymmetric trust (endorse positive gaps, attenuate negative) to bandit updates
//! and absorb-compress promotions. No gradients — pure modelless signal gating.
//!
//! # Key Insight
//!
//! SDAR proves uniform distillation collapses in multi-turn settings. The fix is a sigmoid
//! gate: `gt = σ(β·Δt)` that modulates signal intensity per token. We adapt this pattern
//! to our modelless stack: gate bandit reward signals by teacher-student quality gap, gate
//! absorb-compress promotions by positive-gap-only criteria.
//!
//! # Components
//!
//! - [`SdarBanditPruner`] — bandit with sigmoid-gated reward updates
//! - [`SdarGatedAbsorbCompress`] — absorb-compress with soft sigmoid gate
//! - [`sdar_gate`] — core sigmoid gate function (re-exported from parent module)
//!
//! # Asymmetric Trust Principle
//!
//! - Positive gaps (endorsement) → gate opens → strong update signal
//! - Negative gaps (rejection) → gate closes → attenuated update signal
//! - Sigmoid provides smooth, bounded modulation (no gradient explosion)
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "sdar_gate")]`.
//! Feature: `sdar_gate = []` in `Cargo.toml`.
//!
//! **Source:** [SDAR: Self-Distilled Agentic RL](https://arxiv.org/abs/2605.15155) — ZJU-REAL, 2025

pub mod sdar_absorb;
pub mod sdar_bandit;

#[cfg(debug_assertions)]
pub use sdar_absorb::PromotionStats;
pub use sdar_absorb::{SdarAbsorbConfig, SdarGatedAbsorbCompress};
pub use sdar_bandit::{GateStats, SdarBanditConfig, SdarBanditPruner};
