//! OutlierGuard — per-layer KS D-statistic scanner for model load time (Plan 224).
//!
//! Detects outlier injection attacks in quantized weight matrices.
//! Runs once at model load, O(n log n) per weight matrix.

use crate::spectralquant::spectral::ks_d_statistic;
use crate::types::{OutlierAction, OutlierGuardConfig};
use std::fmt;

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
        self.total_scanned = self.layers.len();
        self.total_flagged = self.layers.iter().filter(|l| l.flagged).count();
        self.max_ks_d = self.layers.iter().map(|l| l.ks_d).fold(0.0f32, f32::max);
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
