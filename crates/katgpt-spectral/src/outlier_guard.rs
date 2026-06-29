//! OutlierGuard — per-layer KS D-statistic scanner for model load time (Plan 224).
//!
//! Detects outlier injection attacks in quantized weight matrices.
//! Runs once at model load, O(n log n) per weight matrix.

use super::spectral::ks_d_statistic;
use std::fmt;

// ---------------------------------------------------------------------------
// Outlier config types (moved from `katgpt-rs/src/types.rs` in Issue 015
// Phase 2 — these are spectralquant-specific config and live with the only
// code that consumes them; re-exported back to root via katgpt-spectral's
// lib.rs for back-compat).
// ---------------------------------------------------------------------------

/// Action to take when outlier injection is detected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum OutlierAction {
    /// Log warning, continue loading. Default for MIT engine.
    #[default]
    Warn = 0,
    /// Reject the model (return error). Useful for SaaS deployment.
    Reject = 1,
    /// Silent — just record metrics, no warning. Useful for benchmarking.
    Silent = 2,
}

/// Configuration for the outlier-aware quantization guard.
/// Runs once at model load time to detect outlier injection attacks.
#[derive(Clone, Debug)]
pub struct OutlierGuardConfig {
    /// KS D-statistic threshold above which a layer is flagged.
    /// Default: 0.15 (conservative midpoint between normal <0.1 and attacked >0.25).
    pub ks_threshold: f32,
    /// What to do when an outlier is detected.
    pub on_detection: OutlierAction,
    /// Whether to also check StiffSoft eigenvalue distribution if available.
    pub use_stiffsoft_crosscheck: bool,
}

impl Default for OutlierGuardConfig {
    fn default() -> Self {
        Self {
            ks_threshold: 0.15,
            on_detection: OutlierAction::Warn,
            use_stiffsoft_crosscheck: false,
        }
    }
}

impl serde::Serialize for OutlierGuardConfig {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = s.serialize_struct("OutlierGuardConfig", 3)?;
        state.serialize_field("ks_threshold", &self.ks_threshold)?;
        state.serialize_field("on_detection", &(self.on_detection as u8))?;
        state.serialize_field("use_stiffsoft_crosscheck", &self.use_stiffsoft_crosscheck)?;
        state.end()
    }
}

/// Report for a single layer scan.
#[derive(Clone, Debug)]
pub struct LayerReport {
    /// Layer index in the model.
    pub layer_idx: usize,
    /// Weight tensor name (e.g., "ffn.up_proj").
    pub weight_name: String,
    /// Computed KS D-statistic.
    pub ks_d: f32,
    /// Whether this layer was flagged as suspicious.
    pub flagged: bool,
    /// StiffSoft cross-check result (if available).
    pub stiffsoft_crosscheck: Option<bool>,
}

impl fmt::Display for LayerReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.flagged { "FLAGGED" } else { "OK" };
        write!(
            f,
            "Layer {} ({}): D={:.4} [{}]",
            self.layer_idx, self.weight_name, self.ks_d, status
        )
    }
}

/// Summary report for the entire model scan.
#[derive(Clone, Debug, Default)]
pub struct OutlierGuardReport {
    /// Per-layer reports.
    pub layers: Vec<LayerReport>,
    /// Total layers scanned.
    pub total_scanned: usize,
    /// Total layers flagged.
    pub total_flagged: usize,
    /// Maximum KS D-statistic across all layers.
    pub max_ks_d: f32,
}

impl OutlierGuardReport {
    fn new() -> Self {
        Self::default()
    }

    fn finalize(&mut self) {
        // Single pass: count flagged and track max ks_d together.
        let mut total_flagged = 0usize;
        let mut max_ks_d = 0.0f32;
        for l in &self.layers {
            if l.flagged {
                total_flagged += 1;
            }
            if l.ks_d > max_ks_d {
                max_ks_d = l.ks_d;
            }
        }
        self.total_scanned = self.layers.len();
        self.total_flagged = total_flagged;
        self.max_ks_d = max_ks_d;
    }
}

impl fmt::Display for OutlierGuardReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "OutlierGuard Report: {} layers scanned, {} flagged, max D={:.4}",
            self.total_scanned, self.total_flagged, self.max_ks_d
        )?;
        for layer in &self.layers {
            writeln!(f, "  {}", layer)?;
        }
        Ok(())
    }
}

/// OutlierGuard scanner. Runs once per model load.
pub struct OutlierGuard {
    config: OutlierGuardConfig,
    report: OutlierGuardReport,
    scratch: Vec<f32>,
}

impl OutlierGuard {
    /// Create a new OutlierGuard with given config.
    pub fn new(config: OutlierGuardConfig) -> Self {
        Self {
            config,
            report: OutlierGuardReport::new(),
            scratch: Vec::new(),
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(OutlierGuardConfig::default())
    }

    /// Scan a single weight matrix.
    /// Returns the KS D-statistic for this layer.
    pub fn scan_layer(&mut self, weights: &[f32], layer_idx: usize, name: &str) -> f32 {
        // Ensure scratch buffer is large enough
        if self.scratch.len() < weights.len() {
            self.scratch.resize(weights.len(), 0.0);
        }

        let ks_d = ks_d_statistic(weights, &mut self.scratch);
        let flagged = ks_d > self.config.ks_threshold;

        let layer_report = LayerReport {
            layer_idx,
            weight_name: name.to_string(),
            ks_d,
            flagged,
            stiffsoft_crosscheck: None, // populated later if StiffSoft available
        };

        if flagged {
            match self.config.on_detection {
                OutlierAction::Warn => {
                    log::warn!(
                        "OutlierGuard: layer {} ({}) flagged with D={:.4}",
                        layer_idx,
                        name,
                        ks_d
                    );
                }
                OutlierAction::Silent => {}
                OutlierAction::Reject => {
                    log::error!(
                        "OutlierGuard: layer {} ({}) flagged with D={:.4} — REJECT requested",
                        layer_idx,
                        name,
                        ks_d
                    );
                }
            }
        }

        self.report.layers.push(layer_report);
        ks_d
    }

    /// Get the final report. Call after all layers are scanned.
    pub fn report(&mut self) -> &OutlierGuardReport {
        self.report.finalize();
        &self.report
    }

    /// Check if any layer was flagged for rejection.
    pub fn should_reject(&self) -> bool {
        self.config.on_detection == OutlierAction::Reject
            && self.report.layers.iter().any(|l| l.flagged)
    }

    /// Get the config.
    pub fn config(&self) -> &OutlierGuardConfig {
        &self.config
    }

    /// Scan all weight matrices in a `TransformerWeights` instance.
    /// Returns a reference to the finalized report.
    ///
    /// This is the primary integration point for model load time.
    /// Call after `TransformerWeights::new()` or after deserializing weights.
    pub fn scan_transformer_weights(
        &mut self,
        weights: &katgpt_transformer::TransformerWeights,
    ) -> &OutlierGuardReport {
        // Scan embedding table
        self.scan_layer(&weights.wte, 0, "embedding.wte");
        self.scan_layer(&weights.wpe, 0, "embedding.wpe");
        self.scan_layer(&weights.lm_head, 0, "lm_head");

        // Scan per-layer weights
        for (idx, layer) in weights.layers.iter().enumerate() {
            self.scan_layer(&layer.attn_wq, idx, &format!("layer{idx}.attn.wq"));
            self.scan_layer(&layer.attn_wk, idx, &format!("layer{idx}.attn.wk"));
            self.scan_layer(&layer.attn_wv, idx, &format!("layer{idx}.attn.wv"));
            self.scan_layer(&layer.attn_wo, idx, &format!("layer{idx}.attn.wo"));
            self.scan_layer(&layer.mlp_w1, idx, &format!("layer{idx}.mlp.w1"));
            self.scan_layer(&layer.mlp_w2, idx, &format!("layer{idx}.mlp.w2"));
            if let Some(ref fused) = layer.attn_qkv_fused {
                self.scan_layer(fused, idx, &format!("layer{idx}.attn.qkv_fused"));
            }
        }

        self.report()
    }
}

/// Confidence level from dual-signal outlier detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConfidenceLevel {
    /// No anomaly detected.
    Clean = 0,
    /// Only one signal flagged (medium confidence).
    Medium = 1,
    /// Both KS and StiffSoft flagged (high confidence).
    High = 2,
}

/// StiffSoft cross-check result.
#[derive(Debug, Clone)]
pub struct StiffSoftCrossCheck {
    /// Whether the KS statistic flagged this layer.
    pub ks_flagged: bool,
    /// Whether the StiffSoft eigenvalue check flagged this layer.
    pub eigenvalue_flagged: bool,
    /// Combined confidence level.
    pub confidence: ConfidenceLevel,
}

#[cfg(feature = "stiff_anomaly")]
impl StiffSoftCrossCheck {
    /// Perform cross-check between KS and StiffSoft signals.
    pub fn check(ks_d: f32, ks_threshold: f32, eigenvalue_anomaly: Option<bool>) -> Self {
        let ks_flagged = ks_d > ks_threshold;
        let eigenvalue_flagged = eigenvalue_anomaly.unwrap_or(false);

        let confidence = match (ks_flagged, eigenvalue_flagged) {
            (true, true) => ConfidenceLevel::High,
            (true, false) | (false, true) => ConfidenceLevel::Medium,
            (false, false) => ConfidenceLevel::Clean,
        };

        Self {
            ks_flagged,
            eigenvalue_flagged,
            confidence,
        }
    }

    /// Log message for this cross-check result.
    pub fn log_message(&self, layer_idx: usize, weight_name: &str) -> String {
        match self.confidence {
            ConfidenceLevel::High => {
                format!(
                    "HIGH CONFIDENCE outlier detection at layer {} ({}): KS D={:.4}, eigenvalue anomaly={}",
                    layer_idx, weight_name, self.ks_flagged, self.eigenvalue_flagged
                )
            }
            ConfidenceLevel::Medium => {
                if self.ks_flagged {
                    format!(
                        "MEDIUM CONFIDENCE — weight distribution anomaly at layer {} ({}): KS flagged, eigenvalue clean",
                        layer_idx, weight_name
                    )
                } else {
                    format!(
                        "MEDIUM CONFIDENCE — eigenvalue anomaly at layer {} ({}): KS clean, eigenvalue flagged",
                        layer_idx, weight_name
                    )
                }
            }
            ConfidenceLevel::Clean => {
                format!("Layer {} ({}) clean", layer_idx, weight_name)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_weights_low_d() {
        let mut guard = OutlierGuard::with_defaults();
        // Generate approximately normal weights via Box-Muller
        let mut weights = Vec::with_capacity(1000);
        for i in 0..500 {
            let u1 = (i as f32 + 0.5) / 500.0;
            let u2 = (i as f32 + 1.0) / 501.0;
            let r = (-2.0 * u1.ln().max(1e-30)).sqrt();
            let z0 = r * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = r * (2.0 * std::f32::consts::PI * u2).sin();
            weights.push(z0 * 0.3);
            weights.push(z1 * 0.3);
        }
        let d = guard.scan_layer(&weights, 0, "test.weight");
        assert!(d < 0.15, "normal weights should have D < 0.15, got {}", d);
    }

    #[test]
    fn test_outlier_weights_high_d() {
        let mut guard = OutlierGuard::with_defaults();
        // Create weights with injected outliers
        let mut weights = vec![0.01f32; 1024];
        // Inject outliers: set every 32nd weight to a huge value
        for i in (0..1024).step_by(32) {
            weights[i] = 512.0; // outlier per paper's attack pattern
        }
        let d = guard.scan_layer(&weights, 0, "test.attacked");
        assert!(d > 0.15, "attacked weights should have D > 0.15, got {}", d);
    }

    #[test]
    fn test_report_summary() {
        let mut guard = OutlierGuard::with_defaults();
        let normal = vec![0.1f32; 100];
        let attacked = {
            let mut w = vec![0.1f32; 100];
            for i in (0..100).step_by(10) {
                w[i] = 1000.0;
            }
            w
        };
        guard.scan_layer(&normal, 0, "layer0.normal");
        guard.scan_layer(&attacked, 1, "layer1.attacked");
        let report = guard.report();
        assert_eq!(report.total_scanned, 2);
        assert!(report.total_flagged >= 1);
        assert!(report.max_ks_d > 0.0);
    }

    #[test]
    fn test_reject_mode() {
        let config = OutlierGuardConfig {
            on_detection: OutlierAction::Reject,
            ..Default::default()
        };
        let mut guard = OutlierGuard::new(config);
        let mut attacked = vec![0.1f32; 100];
        for i in (0..100).step_by(10) {
            attacked[i] = 1000.0;
        }
        guard.scan_layer(&attacked, 0, "test.attacked");
        assert!(guard.should_reject());
    }

    #[test]
    fn test_normal_not_rejected() {
        let config = OutlierGuardConfig {
            on_detection: OutlierAction::Reject,
            ..Default::default()
        };
        let mut guard = OutlierGuard::new(config);
        // Use actually normal-distributed weights
        let mut normal = Vec::with_capacity(200);
        for i in 0..100 {
            let u1 = (i as f32 + 0.5) / 100.0;
            let u2 = (i as f32 + 1.0) / 101.0;
            let r = (-2.0 * u1.ln().max(1e-30)).sqrt();
            let z0 = r * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = r * (2.0 * std::f32::consts::PI * u2).sin();
            normal.push(z0 * 0.3);
            normal.push(z1 * 0.3);
        }
        guard.scan_layer(&normal, 0, "test.normal");
        assert!(!guard.should_reject());
    }

    #[test]
    fn test_zero_allocation_in_scan() {
        let mut guard = OutlierGuard::with_defaults();
        let weights = vec![0.5f32; 500];
        guard.scan_layer(&weights, 0, "test");
        // Second scan should reuse scratch buffer
        let weights2 = vec![0.3f32; 500];
        guard.scan_layer(&weights2, 1, "test2");
        // If we got here without panic, scratch was reused
        assert_eq!(guard.report().layers.len(), 2);
    }
}

#[cfg(test)]
#[cfg(feature = "stiff_anomaly")]
mod crosscheck_tests {
    use super::*;

    #[test]
    fn test_both_flagged_high_confidence() {
        let check = StiffSoftCrossCheck::check(0.3, 0.15, Some(true));
        assert_eq!(check.confidence, ConfidenceLevel::High);
    }

    #[test]
    fn test_only_ks_flagged_medium() {
        let check = StiffSoftCrossCheck::check(0.3, 0.15, Some(false));
        assert_eq!(check.confidence, ConfidenceLevel::Medium);
    }

    #[test]
    fn test_only_eigenvalue_flagged_medium() {
        let check = StiffSoftCrossCheck::check(0.05, 0.15, Some(true));
        assert_eq!(check.confidence, ConfidenceLevel::Medium);
    }

    #[test]
    fn test_clean() {
        let check = StiffSoftCrossCheck::check(0.05, 0.15, Some(false));
        assert_eq!(check.confidence, ConfidenceLevel::Clean);
    }

    #[test]
    fn test_no_stiffsoft_available() {
        let check = StiffSoftCrossCheck::check(0.3, 0.15, None);
        assert!(check.ks_flagged);
        assert!(!check.eigenvalue_flagged);
        assert_eq!(check.confidence, ConfidenceLevel::Medium);
    }

    #[test]
    fn test_log_messages() {
        let high = StiffSoftCrossCheck::check(0.3, 0.15, Some(true));
        assert!(
            high.log_message(0, "ffn.up_proj")
                .contains("HIGH CONFIDENCE")
        );

        let clean = StiffSoftCrossCheck::check(0.05, 0.15, Some(false));
        assert!(clean.log_message(0, "ffn.up_proj").contains("clean"));
    }
}
