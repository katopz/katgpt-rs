//! QGF (Q-Guided Flow) — Test-Time Gradient Guidance for Modelless Inference.
//!
//! Distillation of arXiv:2606.11087 (Zhou et al., 2026) into katgpt-rs.
//!
//! # The QGF Principle
//!
//! Train a reference policy (BC) and a critic (Q-function) **separately**,
//! then at **test time** guide generation with the critic gradient — no
//! policy training required. The gradient is evaluated at a **first-order
//! Euler projection** of the final output, with the Jacobian intentionally
//! dropped (lower variance, lower cost).
//!
//! # Module Layout
//!
//! - [`projector`] — `FirstOrderProjector` (F2): one-step chain projection
//! - [`drafter`] — `QGuidedDrafter` (F1): wraps any `SpeculativeGenerator`
//!   with Q-gradient velocity bias
//! - [`adaptive`] — `VarianceAdaptiveGuidance` (F4): sigmoid-gated per-query
//!   guidance weight
//!
//! # Five-Tier Routing
//!
//! | Tier | Oracle | Latency | Use |
//! |------|--------|---------|-----|
//! | Plasma | `ActionBridge` i8 ternary directions | < 100ns | Game NPCs |
//! | Hot | `LeoHead` cached f32 Q-values | < 1μs | Active inference |
//! | Warm | GPU batched critic forward | ~1ms | Batch / training |
//! | Cold | Turso Q-table snapshot | ~10ms | Consolidation |
//! | Freeze | `NoGuidanceOracle` (zero gradient) | 0ns | Engine always boots |
//!
//! See `.research/268_QGF_Test_Time_Q_Guided_Flow.md` and
//! `.plans/268_qgf_test_time_q_guided_flow.md`.

#![cfg(feature = "qgf")]

pub mod projector;

pub use projector::{project_batch, project_one_step};

#[cfg(feature = "qgf_adaptive")]
pub mod adaptive;
#[cfg(feature = "qgf_adaptive")]
pub use adaptive::adaptive_guidance_weight;

// Re-export the trait and no-op oracle from `traits.rs`.
#[cfg(feature = "qgf_oracle")]
pub use crate::traits::{NoGuidanceOracle, QGradientOracle};
