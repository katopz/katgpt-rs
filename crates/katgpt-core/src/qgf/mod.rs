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
//! - [`oracles`] — concrete `QGradientOracle` impls for `LeoHead`,
//!   `FlowField`, `ActionBridge`, BFN proxy
//! - [`adaptive`] — `VarianceAdaptiveGuidance` (F4): sigmoid-gated per-query
//!   guidance weight
//! - [`route`] — `QgfComputeRoute` + `route_for`: CPU/GPU/ANE backend selection
//!
//! # Tier Mapping (Plasma/Hot/Warm/Cold/Freeze)
//!
//! | Tier | Implementation | Latency | Use Case |
//! |------|----------------|---------|----------|
//! | Plasma | `ActionBridge` ternary i8 directions + f32 Q dot | < 100ns | Game NPCs |
//! | Hot | `LeoHead::all_goals_q` cached f32 values | < 1μs | Active inference |
//! | Warm | GPU batched Q-critic forward | ~1ms | Training / large batch |
//! | Cold | Turso Q-table snapshots | ~10ms | Episode-end consolidation |
//! | Freeze | `NoGuidanceOracle` (zero gradient) | 0ns | Pure BC reference |
//!
//! Tier ↔ Oracle mapping (Plan 268 T9):
//!
//! | Tier | Oracle Struct | Feature | `confidence()` |
//! |------|---------------|---------|----------------|
//! | Plasma | [`ActionBridgeOracle`](oracles::ActionBridgeOracle) | `action_bridge` | 1.0 |
//! | Plasma/Hot | [`FlowFieldOracle`](oracles::FlowFieldOracle) | `flow_field_nav` | 1.0 |
//! | Hot | [`LeoHeadOracle`](oracles::LeoHeadOracle) | `leo_all_goals` | 1.0 |
//! | Hot | [`DualLeoOracle`](oracles::DualLeoOracle) | `dual_leo` | 1.0 |
//! | Freeze | [`BfnProxyOracle`](oracles::BfnProxyOracle) | (always) | 0.3 |
//! | Freeze | [`NoGuidanceOracle`] | `qgf_oracle` | 0.0 |
//!
//! See `.research/236_QGF_Test_Time_Q_Guided_Flow.md` and
//! `.plans/268_qgf_test_time_q_guided_flow.md`.

pub mod projector;

pub use projector::{project_batch, project_one_step};

#[cfg(feature = "qgf_oracle")]
pub mod oracles;

#[cfg(feature = "qgf")]
pub mod drafter;
#[cfg(feature = "qgf")]
pub use drafter::QGuidedDrafter;

#[cfg(feature = "qgf")]
pub mod adaptive;
#[cfg(feature = "qgf")]
pub use adaptive::{
    QgfVarianceSignal, adaptive_guidance_weight, adaptive_guidance_weight_from_signal,
    confidence_from_disagreement,
};

#[cfg(feature = "qgf")]
pub mod route;
#[cfg(feature = "qgf")]
pub use route::{QgfComputeRoute, route_for};

// Plan 268 T8: backend dispatch for batched Q-gradient queries.
// CPU SIMD path is concrete (reuses the oracle's `q_gradient_into`, which
// already calls `simd::dot_f32_i8` + `fast_sigmoid` for ActionBridge);
// GPU/ANE paths are trait delegates that upper layers implement.
#[cfg(feature = "qgf")]
pub mod dispatch;
#[cfg(feature = "qgf")]
pub use dispatch::{QgfAneDelegate, QgfBackendDispatch, QgfGpuDelegate};

// Re-export the trait and no-op oracle from `traits.rs`.
#[cfg(feature = "qgf_oracle")]
pub use crate::traits::{NoGuidanceOracle, QGradientOracle};
