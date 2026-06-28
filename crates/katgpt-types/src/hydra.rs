//! Hydra Adaptive Layer Budget types.

// ---------------------------------------------------------------------------
// Hydra Adaptive Layer Budget (Research 148, Plan 165)
// ---------------------------------------------------------------------------

/// Per-layer Hydra profile entry (modelless mode).
/// Pre-computed from calibration data, stored in config.
#[cfg(feature = "hydra_budget")]
#[derive(Clone, Debug)]
pub struct HydraLayerProfile {
    /// Mean absolute direct effect on top-token logit.
    pub mean_de: f32,
    /// Fraction of prompts where this layer is a Hydra backup.
    pub backup_frequency: f32,
    /// Whether this layer acts as erasure (mean DE < 0 for MLP).
    pub is_erasure: bool,
}

/// Hydra budget configuration.
#[cfg(feature = "hydra_budget")]
#[derive(Clone, Debug)]
pub struct HydraBudgetConfig {
    /// Skip layers with |DE| below this threshold.
    pub skip_threshold: f32,
    /// Early-terminate when cumulative DE reaches this fraction of total.
    pub cumulative_threshold: f32,
    /// Use modelless mode (lookup) vs model-based (logit lens).
    pub modelless: bool,
    /// Skip erasure MLPs during draft stage.
    pub skip_erasure_draft: bool,
}

#[cfg(feature = "hydra_budget")]
impl Default for HydraBudgetConfig {
    fn default() -> Self {
        Self {
            skip_threshold: 0.01,
            cumulative_threshold: 0.95,
            modelless: true,
            skip_erasure_draft: false,
        }
    }
}
