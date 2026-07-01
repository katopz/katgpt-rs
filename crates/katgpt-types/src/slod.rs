//! SLoD scale-boundary POD type — shared between katgpt-core's SLoD operator
//! (Plan 235) and katgpt-sense's LOD router. Pure data, zero deps.
//!
//! Co-extracted from `katgpt-core/src/slod.rs` (Plan 338 Phase 1) so that the
//! promoted katgpt-sense crate can depend on the leaf crate only, breaking the
//! katgpt-core cycle. katgpt-core re-exports this via
//! `katgpt_core::slod::ScaleBoundary` (bit-for-bit path preserved).

/// A detected scale boundary from the spectral analysis.
#[derive(Debug, Clone)]
pub struct ScaleBoundary {
    /// Diffusion scale σ at which this boundary was detected.
    pub sigma: f32,
    /// Composite boundary score S(σ).
    pub score: f32,
    /// Effective rank K* (number of significant eigenmodes) at this scale.
    pub k_star: usize,
}
