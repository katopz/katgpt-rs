//! Per-head budget allocation (Plan 271 Phase 3, Algorithm 4).
//!
//! Given per-head sensitivity curves `ΔQ_h(r)` (quality drop when head `h`
//! keeps ratio `r` of its KV cache), solve for a nonuniform per-head budget
//! that minimizes total quality loss subject to a global ratio target.
//!
//! # Module layout
//! - `curve`   — `HeadSensitivityCurve` and linear interpolation.
//! - `solver`  — `HeadBudgetSolver` greedy-swap implementation.
//! - `schedule`— `HeadBudgetSchedule` with BLAKE3 + postcard ser/deser.
//! - `measure` — Offline measurement stub (real impl lives in riir-ai).

pub mod curve;
pub mod measure;
pub mod schedule;
pub mod solver;

pub use curve::HeadSensitivityCurve;
pub use measure::measure_sensitivity_stub;
pub use schedule::HeadBudgetSchedule;
pub use solver::HeadBudgetSolver;
