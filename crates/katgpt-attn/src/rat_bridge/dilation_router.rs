//! Adaptive dilation routing — selects dilation factor per-layer based on load and RV signal.
//!
//! Plan 225 Phase 4: Routes dilation selection through QPS thresholds and
//! per-layer River Valley scores. High RV (peaked attention) tolerates aggressive
//! dilation; low RV (flat attention) stays dense.

use katgpt_core::types::DilationConfig;

/// Per-layer dilation configuration router.
///
/// Selects dilation factor based on QPS load and optional per-layer River Valley scores.
/// RV > 0.8 → peaked attention, tolerates more dilation.
/// RV < 0.2 → flat attention, stays dense.
#[derive(Debug, Clone)]
pub struct DilationRouter {
    /// QPS-based thresholds for dilation selection (ascending).
    pub qps_thresholds: Vec<(f32, DilationConfig)>,
    /// Per-layer River Valley scores (if available).
    pub layer_rv_scores: Vec<f32>,
}

impl DilationRouter {
    /// Create a new router with default QPS thresholds.
    pub fn new() -> Self {
        Self {
            qps_thresholds: vec![
                (1.0, DilationConfig::D1),
                (5.0, DilationConfig::D4),
                (20.0, DilationConfig::D16),
                (f32::MAX, DilationConfig::D64),
            ],
            layer_rv_scores: Vec::new(),
        }
    }

    /// Select dilation for a given QPS load level.
    pub fn select_dilation(&self, qps: f32) -> DilationConfig {
        for (threshold, config) in &self.qps_thresholds {
            if qps < *threshold {
                return *config;
            }
        }
        DilationConfig::D64
    }

    /// Select dilation for a specific layer, considering River Valley score.
    ///
    /// High RV (peaked) → tolerate more dilation.
    /// Low RV (flat) → stay dense.
    pub fn select_layer_dilation(&self, qps: f32, layer_idx: usize) -> DilationConfig {
        let base = self.select_dilation(qps);

        let rv = self.layer_rv_scores.get(layer_idx).copied().unwrap_or(0.5);

        if rv > 0.8 {
            // Peaked attention → can dilate more aggressively
            match base {
                DilationConfig::D1 => DilationConfig::D4,
                DilationConfig::D4 => DilationConfig::D16,
                _ => DilationConfig::D64,
            }
        } else if rv < 0.2 {
            // Flat attention → stay dense
            DilationConfig::D1
        } else {
            base
        }
    }

    /// Update RV score for a layer.
    pub fn update_rv(&mut self, layer_idx: usize, rv: f32) {
        if layer_idx >= self.layer_rv_scores.len() {
            self.layer_rv_scores.resize(layer_idx + 1, 0.5);
        }
        self.layer_rv_scores[layer_idx] = rv;
    }
}

/// Standalone dilation selection based on QPS and optional per-layer RV.
///
/// Convenience function wrapping the same logic as `DilationRouter::select_layer_dilation`
/// without requiring a router instance.
pub fn select_dilation(qps: f32, per_layer_rv: Option<f32>) -> DilationConfig {
    let base = DilationConfig::from_qps(qps);

    if let Some(rv) = per_layer_rv {
        if rv > 0.8 {
            match base {
                DilationConfig::D1 => DilationConfig::D4,
                DilationConfig::D4 => DilationConfig::D16,
                _ => DilationConfig::D64,
            }
        } else if rv < 0.2 {
            DilationConfig::D1
        } else {
            base
        }
    } else {
        base
    }
}

impl Default for DilationRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_dilation_low_qps() {
        let router = DilationRouter::new();
        assert_eq!(router.select_dilation(0.5), DilationConfig::D1);
    }

    #[test]
    fn test_select_dilation_medium_qps() {
        let router = DilationRouter::new();
        assert_eq!(router.select_dilation(3.0), DilationConfig::D4);
    }

    #[test]
    fn test_select_dilation_high_qps() {
        let router = DilationRouter::new();
        assert_eq!(router.select_dilation(10.0), DilationConfig::D16);
    }

    #[test]
    fn test_select_dilation_very_high_qps() {
        let router = DilationRouter::new();
        assert_eq!(router.select_dilation(50.0), DilationConfig::D64);
    }

    #[test]
    fn test_layer_dilation_peaked() {
        let mut router = DilationRouter::new();
        router.update_rv(0, 0.9); // peaked layer
        // base would be D1 (qps=0.5), but RV>0.8 promotes to D4
        assert_eq!(router.select_layer_dilation(0.5, 0), DilationConfig::D4);
    }

    #[test]
    fn test_layer_dilation_flat() {
        let mut router = DilationRouter::new();
        router.update_rv(0, 0.1); // flat layer
        // base would be D16 (qps=10), but RV<0.2 forces D1
        assert_eq!(router.select_layer_dilation(10.0, 0), DilationConfig::D1);
    }

    #[test]
    fn test_layer_dilation_neutral_rv() {
        let mut router = DilationRouter::new();
        router.update_rv(0, 0.5); // neutral
        // base = D16, RV is neutral → stays D16
        assert_eq!(router.select_layer_dilation(10.0, 0), DilationConfig::D16);
    }

    #[test]
    fn test_layer_dilation_no_rv_score() {
        let router = DilationRouter::new();
        // No RV score for layer 0 → defaults to 0.5 (neutral)
        assert_eq!(router.select_layer_dilation(0.5, 0), DilationConfig::D1);
    }

    #[test]
    fn test_peaked_promotes_multiple_levels() {
        let mut router = DilationRouter::new();
        router.update_rv(0, 0.95);
        // qps=3.0 → base=D4, RV>0.8 → D16
        assert_eq!(router.select_layer_dilation(3.0, 0), DilationConfig::D16);
        // qps=10.0 → base=D16, RV>0.8 → D64
        assert_eq!(router.select_layer_dilation(10.0, 0), DilationConfig::D64);
    }

    #[test]
    fn test_standalone_select_dilation() {
        // Low QPS, no RV
        assert_eq!(select_dilation(0.5, None), DilationConfig::D1);
        // Low QPS, peaked RV
        assert_eq!(select_dilation(0.5, Some(0.9)), DilationConfig::D4);
        // High QPS, flat RV
        assert_eq!(select_dilation(10.0, Some(0.1)), DilationConfig::D1);
        // Medium QPS, neutral RV
        assert_eq!(select_dilation(3.0, Some(0.5)), DilationConfig::D4);
    }

    #[test]
    fn test_update_rv_resizes() {
        let mut router = DilationRouter::new();
        router.update_rv(5, 0.8);
        assert_eq!(router.layer_rv_scores.len(), 6);
        assert_eq!(router.layer_rv_scores[5], 0.8);
        // Uninitialized layers get default 0.5
        assert_eq!(router.layer_rv_scores[0], 0.5);
    }

    #[test]
    fn test_default_trait() {
        let router = DilationRouter::default();
        assert_eq!(router.select_dilation(0.5), DilationConfig::D1);
    }
}
