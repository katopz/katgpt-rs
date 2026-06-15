//! `thinking_cot` — adaptive CoT thinking framework (Plan 194) and host of the
//! [`ThinkingStrategy`] integration point.
//!
//! Historically a *meta-feature* that pulled in the bandit, prune, and probe
//! machinery ([`ThinkingController`](crate::speculative)) without owning any
//! code itself. Plan 275 Phase 2 introduces this module as the home of the
//! shared strategy trait so that SwiR (Plan 275), CollapseAware (Plan 212),
//! ChainFold (Plan 195), and future strategies can plug into the decode loop
//! through a uniform interface.
//!
//! Only the strategy primitives live here — the bandit-driven
//! [`ThinkingController`](crate::speculative) stays where it is.

pub mod strategy;

pub use strategy::{ControlTokenIds, StepContext, StepDirective, ThinkingStrategy};
