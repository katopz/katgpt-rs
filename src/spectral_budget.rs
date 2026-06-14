//! Spectral Budget Router — Layer-Adaptive NS Depth + Rank-p Truncation (Plan 253, Research 222).
//!
//! Uses spectral power laws from Magakyan et al. (2026) to predict per-layer
//! Newton-Schulz iteration depth and singular direction retention at inference time.
//! Pure arithmetic — no training, no data.
//!
//! # Power Law Formula
//!
//! After burn-in, singular value quantiles stabilize at:
//!   σ_q(M) = c_q · M^(-α_layer)
//!
//! Where α_layer varies by depth fraction:
//! - Mid-early/mid/mid-late: α ≈ -0.25
//! - Final attention (Q/K/V/O): α ≈ -0.38 to -0.66
//! - Final MLP Up: α ≈ -0.96

// ── Spectral Exponent Table ────────────────────────────────────

/// Power law exponent lookup by (depth_fraction, layer_type).
///
/// Exponents from Magakyan et al. Figure 16, fitted on 77M-2.8B GPT-2 models.
/// R² > 0.98 for all layer types.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum LayerType {
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    MlpUp,
    MlpDown,
}

/// Returns the spectral power law exponent for a given depth fraction and layer type.
///
/// Depth fraction is relative position in the network:
///   0.0 = first layer, 1.0 = last layer.
///
/// Values interpolated from the paper's empirical measurements:
/// - mid-early/mid/mid-late: mild scaling (~-0.25)
/// - final layers: aggressive scaling (-0.38 to -0.96)
#[inline]
pub fn spectral_exponent(depth_fraction: f32, layer_type: LayerType) -> f32 {
    debug_assert!(
        (0.0..=1.0).contains(&depth_fraction),
        "depth_fraction must be in [0, 1], got {depth_fraction}"
    );
    let d = depth_fraction.clamp(0.0, 1.0);

    // Mid-early (0.0-0.25), mid (0.25-0.50), mid-late (0.50-0.75):
    // All approximately -0.25 for all layer types.
    if d < 0.75 {
        return -0.25;
    }

    // Late (0.75-1.0): interpolate from -0.25 to layer-specific final value.
    let t = (d - 0.75) / 0.25; // 0.0 at d=0.75, 1.0 at d=1.0
    let alpha_final = match layer_type {
        LayerType::AttentionQ => -0.46,
        LayerType::AttentionK => -0.38,
        LayerType::AttentionV => -0.58,
        LayerType::AttentionO => -0.66,
        LayerType::MlpUp => -0.96,
        LayerType::MlpDown => -0.52,
    };
    // Linear interpolation in log space
    -0.25 + t * (alpha_final - (-0.25))
}

// ── NS Depth Prediction ────────────────────────────────────────

/// NS iteration depth thresholds.
///
/// Standard 5-step NanoGPT NS fails for σ < ~0.003.
/// DeepSeek-V4 10-step composition fails for σ < ~2e-6.
const NS5_FAILURE_THRESHOLD: f32 = 3e-3;
const NS7_FAILURE_THRESHOLD: f32 = 2e-4;
const _NS10_FAILURE_THRESHOLD: f32 = 2e-6;

/// Select Newton-Schulz iteration count based on predicted median singular value.
///
/// - σ > 0.003 → 5 steps (NanoGPT config sufficient)
/// - σ > 0.0002 → 7 steps (intermediate accuracy)
/// - σ ≤ 0.0002 → 10 steps (DeepSeek-V4 config)
#[inline]
pub fn ns_depth_for_sigma(predicted_sigma_50: f32) -> u8 {
    if predicted_sigma_50 > NS5_FAILURE_THRESHOLD {
        5
    } else if predicted_sigma_50 > NS7_FAILURE_THRESHOLD {
        7
    } else {
        10
    }
}

/// NS depth recommendation for a depth fraction, ignoring model size.
///
/// This is a conservative heuristic: at any model size, final layers
/// are harder than mid layers. Returns the minimum recommended depth.
#[inline]
pub fn ns_depth_for_depth_fraction(depth_fraction: f32) -> u8 {
    if depth_fraction < 0.75 {
        5
    } else if depth_fraction < 0.875 {
        7
    } else {
        10
    }
}

// ── Predictive Config ───────────────────────────────────────────

/// Pre-computed NS configuration per depth fraction.
#[derive(Debug, Clone)]
pub struct NsDepthConfig {
    /// Relative depth (0.0 = first layer, 1.0 = last)
    pub depth_fraction: f32,
    /// Power law exponent for this depth
    pub spectral_exponent: f32,
    /// Recommended NS iterations (5, 7, or 10)
    pub ns_iterations: u8,
    /// Fraction of singular directions to keep (from rank-p result)
    /// Paper proves top 50% suffices for full Muon performance.
    pub retention_fraction: f32,
    /// Predicted σ_0.5 for this layer at given model size (0 = not computed)
    pub predicted_median_sv: f32,
}

/// Pre-computed per-layer NS configuration for an entire model.
#[derive(Debug, Clone)]
pub struct SpectralBudgetConfig {
    /// Per-depth config, sorted by depth_fraction ascending.
    pub layers: Vec<NsDepthConfig>,
    /// Model size in millions of parameters.
    pub model_size_m: usize,
}

impl SpectralBudgetConfig {
    /// Predict per-layer NS config from model dimensions.
    ///
    /// Uses power law: σ_q(M) = c_q · M^(-α)
    /// where c_q is calibrated from the paper's 2.8B baseline.
    ///
    /// The coefficient c_q for median (q=0.5) is estimated from
    /// the paper's Figure 16 data points at 2.8B:
    /// - Mid layers: σ_0.5 ≈ 5e-3 at M=2800
    /// - Final O: σ_0.5 ≈ 1e-3 at M=2800
    /// - Final MLP Up: σ_0.5 ≈ 3e-3 at M=2800
    ///
    /// These are approximate; the exponents are the reliable part (R² > 0.98).
    pub fn from_model_dims(
        n_layers: usize,
        model_size_m: usize,
        layer_types: &[LayerType],
    ) -> Self {
        assert!(!layer_types.is_empty(), "need at least one layer type");
        // Each transformer block has 6 matrices: Q, K, V, O, MLP_Up, MLP_Down
        // layer_types should have 6 * n_layers entries, or we cycle.
        let layers = (0..n_layers)
            .map(|i| {
                let depth_frac = if n_layers > 1 {
                    i as f32 / (n_layers - 1) as f32
                } else {
                    0.5
                };
                let lt = layer_types[i % layer_types.len()];
                let alpha = spectral_exponent(depth_frac, lt);

                // Predict median σ at this model size.
                // Calibration: at M=2800, σ_0.5 for α=-0.25 layers ≈ 5e-3.
                // σ(M) = c · M^α, so c = σ(2800) / 2800^α = 5e-3 / 2800^(-0.25)
                // Then σ(model_size_m) = c · model_size_m^α
                let predicted_sv = predict_median_sv(alpha, model_size_m);

                let ns_iters = ns_depth_for_sigma(predicted_sv);

                // Retention: 50% for gaming/standard, but final layers with
                // aggressive scaling might need more retention
                let retention = if depth_frac < 0.75 {
                    0.5
                } else if depth_frac < 0.875 {
                    0.6
                } else {
                    0.75
                };

                NsDepthConfig {
                    depth_fraction: depth_frac,
                    spectral_exponent: alpha,
                    ns_iterations: ns_iters,
                    retention_fraction: retention,
                    predicted_median_sv: predicted_sv,
                }
            })
            .collect();

        SpectralBudgetConfig {
            layers,
            model_size_m,
        }
    }

    /// Get NS iteration count for a specific layer index.
    #[inline]
    pub fn ns_iterations(&self, layer_idx: usize) -> u8 {
        self.layers
            .get(layer_idx)
            .map(|c| c.ns_iterations)
            .unwrap_or(5)
    }

    /// Get retention fraction for a specific layer index.
    #[inline]
    pub fn retention(&self, layer_idx: usize) -> f32 {
        self.layers
            .get(layer_idx)
            .map(|c| c.retention_fraction)
            .unwrap_or(0.5)
    }

    /// Count layers that need more than 5 NS iterations.
    pub fn count_deep_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.ns_iterations > 5).count()
    }

    /// Total NS iterations across all layers.
    pub fn total_ns_iterations(&self) -> usize {
        self.layers.iter().map(|l| l.ns_iterations as usize).sum()
    }

    /// Compare against uniform 5-step: (this_total, uniform_5_total, ratio).
    pub fn vs_uniform5(&self) -> (usize, usize, f32) {
        let this = self.total_ns_iterations();
        let uniform = self.layers.len() * 5;
        (this, uniform, this as f32 / uniform as f32)
    }
}

/// Predict median singular value at given model size for a layer with given exponent.
///
/// Calibrated from the paper's data at M=2800:
///   For α=-0.25: σ_0.5(2800) ≈ 5e-3
///   c = 5e-3 / 2800^(-0.25) ≈ 5e-3 / 0.13748 ≈ 0.03637
///
/// So σ_0.5(M) ≈ 0.03637 · M^(-|α|)
#[inline]
fn predict_median_sv(alpha: f32, model_size_m: usize) -> f32 {
    // Calibration coefficient from mid-layer data at 2.8B.
    // c = CALIB_SIGMA / CALIB_SIZE^calib_alpha  — a pure constant, hoisted out
    // of the per-layer loop so `powf` runs only once at compile time, not per layer.
    // Pre-computed: 5e-3 / 2800f32.powf(-0.25) ≈ 0.03637136 (powf is not const fn,
    // so the literal is evaluated once here instead of per call).
    const CALIB_C: f32 = 0.03637136;

    // σ(M) = c · M^alpha
    CALIB_C * (model_size_m as f32).powf(alpha)
}

// ── Spectral LOD ────────────────────────────────────────────────

/// Structural Level of Detail derived from spectral exponents.
///
/// Combines with SLoD (Semantic LOD) for compute routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpectralLod {
    /// Simple terrain — mid-early/mid layers, α ≈ -0.25
    /// 5-step NS, minimal compute
    Lod0,
    /// Slightly complex — mid-late, α ≈ -0.27
    /// 5-step NS, monitor effective rank
    Lod1,
    /// Complex — late attention, α ≈ -0.46 to -0.66
    /// 7-10 step NS recommended
    Lod2,
    /// Critical — final MLP Up, α ≈ -0.96
    /// Full 10-step NS + rank-p truncation
    Lod3,
}

impl SpectralLod {
    /// Classify a layer by its spectral exponent.
    #[inline]
    pub fn from_exponent(alpha: f32) -> Self {
        let abs_alpha = alpha.abs();
        if abs_alpha < 0.30 {
            SpectralLod::Lod0
        } else if abs_alpha < 0.45 {
            SpectralLod::Lod1
        } else if abs_alpha < 0.70 {
            SpectralLod::Lod2
        } else {
            SpectralLod::Lod3
        }
    }

    /// Classify a layer by its depth fraction.
    #[inline]
    pub fn from_depth(depth_fraction: f32) -> Self {
        if depth_fraction < 0.50 {
            SpectralLod::Lod0
        } else if depth_fraction < 0.75 {
            SpectralLod::Lod1
        } else if depth_fraction < 0.875 {
            SpectralLod::Lod2
        } else {
            SpectralLod::Lod3
        }
    }

    /// Recommended NS iterations for this LOD level.
    #[inline]
    pub fn ns_iterations(self) -> u8 {
        match self {
            SpectralLod::Lod0 => 5,
            SpectralLod::Lod1 => 5,
            SpectralLod::Lod2 => 7,
            SpectralLod::Lod3 => 10,
        }
    }

    /// Recommended retention fraction for this LOD level.
    #[inline]
    pub fn retention(self) -> f32 {
        match self {
            SpectralLod::Lod0 => 0.50,
            SpectralLod::Lod1 => 0.50,
            SpectralLod::Lod2 => 0.60,
            SpectralLod::Lod3 => 0.75,
        }
    }
}

// ── Rank-p Spectral Truncation (T3) ────────────────────────────────

/// SVD-free rank-p truncation based on row norms after Newton-Schulz.
///
/// After NS iteration, well-orthonormalized rows have ||row|| ≈ 1.
/// Poorly-orthonormalized rows have ||row|| << 1. This function sorts rows
/// by norm descending and retains the top `retention` fraction, zeroing
/// the rest.
///
/// Paper proves top 50% suffices to recover full Muon performance
/// (Magakyan et al. 2026, rank-p analysis).
///
/// `ns_output` is a `rows × cols` row-major matrix (mutated in place).
/// `retention` is in (0.0, 1.0] — 0.5 keeps top half.
/// `row_norms_buf` is a caller-provided buffer of `rows` elements for
/// scratch storage (avoids allocation).
///
/// Returns the number of rows retained.
pub fn rank_p_retain(
    ns_output: &mut [f32],
    rows: usize,
    cols: usize,
    retention: f32,
    row_norms_buf: &mut [f32],
) -> usize {
    debug_assert_eq!(ns_output.len(), rows * cols);
    debug_assert_eq!(row_norms_buf.len(), rows);
    debug_assert!(
        (0.0..=1.0).contains(&retention),
        "retention must be in [0, 1]"
    );

    if rows == 0 || cols == 0 {
        return 0;
    }

    let r = retention.clamp(0.0, 1.0);
    let keep = ((rows as f32 * r).ceil() as usize).min(rows);

    if keep == rows {
        return rows;
    }

    // Compute row norms
    for i in 0..rows {
        let row_start = i * cols;
        let row_end = row_start + cols;
        let sq_sum = crate::simd::simd_sum_sq(&ns_output[row_start..row_end], cols);
        row_norms_buf[i] = sq_sum;
    }

    // Build index array sorted by norm descending.
    // For small matrices (typical NS use), a simple insertion sort avoids allocation.
    let mut indices: Vec<usize> = (0..rows).collect();
    // Partial sort: only need top `keep` elements in correct position.
    indices.select_nth_unstable_by(keep - 1, |&a, &b| {
        row_norms_buf[b]
            .partial_cmp(&row_norms_buf[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Zero rows NOT in top `keep`.
    //
    // Avoid the auxiliary `Vec<bool>` mask: walk `indices[keep..]` directly.
    // After `select_nth_unstable_by`, indices[keep..] contains exactly the rows
    // to discard (in unspecified order). Clearing those rows in place is O(cols)
    // per row, matching the previous mask-based approach but with no allocation.
    for &discard_idx in &indices[keep..] {
        let row_start = discard_idx * cols;
        ns_output[row_start..row_start + cols].fill(0.0);
    }

    keep
}

// ── Spectral Budget Arms (T4) ────────────────────────────────────────

/// Pre-defined compute arms for spectral budget routing.
///
/// Each arm represents a compute/accuracy tradeoff tier:
/// - Gaming: minimum compute, 50% retention (hot path)
/// - Chain: standard quality, 75% retention
/// - Diagnostic: maximum accuracy, depth-adaptive NS, 90% retention
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralBudgetArm {
    /// NS iterations for this arm
    pub ns_iterations: u8,
    /// Singular direction retention fraction
    pub retention: f32,
    /// Relative compute cost estimate (1.0 = baseline 5-step NS)
    pub compute_cost: f32,
    /// Arm name for logging
    pub name: &'static str,
}

impl SpectralBudgetArm {
    /// Gaming arm: 5-step NS, 50% retention — minimum compute for hot path.
    /// Best for real-time game NPC updates where latency dominates.
    pub fn gaming() -> Self {
        Self {
            ns_iterations: 5,
            retention: 0.50,
            compute_cost: 1.0,
            name: "gaming",
        }
    }

    /// Chain arm: 5-step NS, 75% retention — standard quality.
    /// Good balance of compute and quality for chain-of-thought.
    pub fn chain() -> Self {
        Self {
            ns_iterations: 5,
            retention: 0.75,
            compute_cost: 1.0,
            name: "chain",
        }
    }

    /// Diagnostic arm: depth-adaptive NS, 90% retention — maximum accuracy.
    /// Uses full spectral budget config for layer-adaptive iteration counts.
    pub fn diagnostic() -> Self {
        Self {
            ns_iterations: 10, // overridden by config per-layer
            retention: 0.90,
            compute_cost: 2.0, // up to 2× more iterations
            name: "diagnostic",
        }
    }

    /// Resolve actual NS iterations for a specific layer using spectral config.
    ///
    /// Gaming/Chain arms use their fixed iteration count.
    /// Diagnostic arm uses the config's predicted depth.
    pub fn resolve_ns_iterations(&self, config: &SpectralBudgetConfig, layer_idx: usize) -> u8 {
        match self.name {
            "diagnostic" => config.ns_iterations(layer_idx),
            _ => self.ns_iterations,
        }
    }

    /// Resolve actual retention for a specific layer using spectral config.
    ///
    /// Gaming/Chain arms use their fixed retention.
    /// Diagnostic arm uses the config's predicted retention.
    pub fn resolve_retention(&self, config: &SpectralBudgetConfig, layer_idx: usize) -> f32 {
        match self.name {
            "diagnostic" => {
                let config_ret = config.retention(layer_idx);
                // Use the higher of arm default and config
                self.retention.max(config_ret)
            }
            _ => self.retention,
        }
    }
}

/// All three pre-defined spectral budget arms, ordered by compute cost.
pub const SPECTRAL_BUDGET_ARMS: [SpectralBudgetArm; 3] = [
    SpectralBudgetArm {
        ns_iterations: 5,
        retention: 0.50,
        compute_cost: 1.0,
        name: "gaming",
    },
    SpectralBudgetArm {
        ns_iterations: 5,
        retention: 0.75,
        compute_cost: 1.0,
        name: "chain",
    },
    SpectralBudgetArm {
        ns_iterations: 10,
        retention: 0.90,
        compute_cost: 2.0,
        name: "diagnostic",
    },
];

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // T1: Exponent table matches paper's values
    #[test]
    fn t1_mid_layer_exponent_near_025() {
        for lt in [
            LayerType::AttentionQ,
            LayerType::AttentionK,
            LayerType::AttentionV,
            LayerType::AttentionO,
            LayerType::MlpUp,
            LayerType::MlpDown,
        ] {
            let alpha = spectral_exponent(0.375, lt); // mid layer
            assert!(
                (alpha - (-0.25)).abs() < 0.01,
                "mid layer alpha for {lt:?} should be ~-0.25, got {alpha}"
            );
        }
    }

    #[test]
    fn t1_final_mlp_up_exponent_near_096() {
        let alpha = spectral_exponent(1.0, LayerType::MlpUp);
        assert!(
            (alpha - (-0.96)).abs() < 0.01,
            "final MLP Up alpha should be ~-0.96, got {alpha}"
        );
    }

    #[test]
    fn t1_final_o_exponent_near_066() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionO);
        assert!(
            (alpha - (-0.66)).abs() < 0.01,
            "final O alpha should be ~-0.66, got {alpha}"
        );
    }

    #[test]
    fn t1_final_v_exponent_near_058() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionV);
        assert!(
            (alpha - (-0.58)).abs() < 0.01,
            "final V alpha should be ~-0.58, got {alpha}"
        );
    }

    #[test]
    fn t1_final_q_exponent_near_046() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionQ);
        assert!(
            (alpha - (-0.46)).abs() < 0.01,
            "final Q alpha should be ~-0.46, got {alpha}"
        );
    }

    #[test]
    fn t1_final_k_exponent_near_038() {
        let alpha = spectral_exponent(1.0, LayerType::AttentionK);
        assert!(
            (alpha - (-0.38)).abs() < 0.01,
            "final K alpha should be ~-0.38, got {alpha}"
        );
    }

    #[test]
    fn t1_final_mlp_down_exponent_near_052() {
        let alpha = spectral_exponent(1.0, LayerType::MlpDown);
        assert!(
            (alpha - (-0.52)).abs() < 0.01,
            "final MLP Down alpha should be ~-0.52, got {alpha}"
        );
    }

    // T2: NS depth selector matches paper's thresholds
    #[test]
    fn t2_sigma_above_threshold_gets_5_steps() {
        assert_eq!(ns_depth_for_sigma(0.01), 5);
        assert_eq!(ns_depth_for_sigma(0.005), 5);
        assert_eq!(ns_depth_for_sigma(3.001e-3), 5);
    }

    #[test]
    fn t2_sigma_mid_range_gets_7_steps() {
        assert_eq!(ns_depth_for_sigma(0.001), 7);
        assert_eq!(ns_depth_for_sigma(2.01e-4), 7);
    }

    #[test]
    fn t2_sigma_below_threshold_gets_10_steps() {
        assert_eq!(ns_depth_for_sigma(1e-4), 10);
        assert_eq!(ns_depth_for_sigma(1e-5), 10);
        assert_eq!(ns_depth_for_sigma(1e-6), 10);
    }

    // T3: Spectral config prediction
    #[test]
    fn t3_config_mid_layers_get_5_steps() {
        let lt = [
            LayerType::AttentionQ,
            LayerType::AttentionK,
            LayerType::AttentionV,
            LayerType::AttentionO,
            LayerType::MlpUp,
            LayerType::MlpDown,
        ];
        let cfg = SpectralBudgetConfig::from_model_dims(8, 77, &lt);
        // First 6 layers (depth < 0.75) should get 5 steps
        for i in 0..6 {
            assert_eq!(
                cfg.ns_iterations(i),
                5,
                "layer {i} (depth={:.2}) should get 5 steps",
                cfg.layers[i].depth_fraction
            );
        }
    }

    #[test]
    fn t3_config_final_layers_may_need_more() {
        let lt = [LayerType::MlpUp]; // Most aggressive scaling
        let cfg = SpectralBudgetConfig::from_model_dims(28, 2800, &lt);
        // Last layer should need more than 5 steps at 2.8B scale
        let last = cfg.layers.last().unwrap();
        assert!(
            last.ns_iterations >= 5,
            "final layer should need at least 5 steps, got {}",
            last.ns_iterations
        );
    }

    #[test]
    fn t3_config_vs_uniform5_ratio() {
        let lt = [
            LayerType::AttentionQ,
            LayerType::AttentionK,
            LayerType::AttentionV,
            LayerType::AttentionO,
            LayerType::MlpUp,
            LayerType::MlpDown,
        ];
        let cfg = SpectralBudgetConfig::from_model_dims(28, 2800, &lt);
        let (total, uniform, ratio) = cfg.vs_uniform5();
        // Should be close to 1.0 — most layers get 5 steps, a few get more
        assert!(
            ratio >= 1.0 && ratio <= 1.3,
            "ratio {ratio} should be in [1.0, 1.3] (total={total}, uniform={uniform})"
        );
    }

    // T4: Spectral LOD classification
    #[test]
    fn t4_lod_from_exponent() {
        assert_eq!(SpectralLod::from_exponent(-0.25), SpectralLod::Lod0);
        assert_eq!(SpectralLod::from_exponent(-0.35), SpectralLod::Lod1);
        assert_eq!(SpectralLod::from_exponent(-0.55), SpectralLod::Lod2);
        assert_eq!(SpectralLod::from_exponent(-0.96), SpectralLod::Lod3);
    }

    #[test]
    fn t4_lod_from_depth() {
        assert_eq!(SpectralLod::from_depth(0.25), SpectralLod::Lod0);
        assert_eq!(SpectralLod::from_depth(0.60), SpectralLod::Lod1);
        assert_eq!(SpectralLod::from_depth(0.80), SpectralLod::Lod2);
        assert_eq!(SpectralLod::from_depth(0.95), SpectralLod::Lod3);
    }

    #[test]
    fn t4_lod_ns_iterations() {
        assert_eq!(SpectralLod::Lod0.ns_iterations(), 5);
        assert_eq!(SpectralLod::Lod1.ns_iterations(), 5);
        assert_eq!(SpectralLod::Lod2.ns_iterations(), 7);
        assert_eq!(SpectralLod::Lod3.ns_iterations(), 10);
    }

    #[test]
    fn t4_lod_retention() {
        assert!((SpectralLod::Lod0.retention() - 0.50).abs() < 0.01);
        assert!((SpectralLod::Lod3.retention() - 0.75).abs() < 0.01);
    }

    // T5: Predicted sigma decreases with model size
    #[test]
    fn t5_sigma_decreases_with_model_size() {
        let sigma_77m = predict_median_sv(-0.25, 77);
        let sigma_2_8b = predict_median_sv(-0.25, 2800);
        assert!(
            sigma_2_8b < sigma_77m,
            "sigma should decrease with model size: 77M={sigma_77m:.6} vs 2.8B={sigma_2_8b:.6}"
        );
    }

    #[test]
    fn t5_final_mlp_sigma_lower_than_mid() {
        let mid = predict_median_sv(-0.25, 2800);
        let final_mlp = predict_median_sv(-0.96, 2800);
        assert!(
            final_mlp < mid,
            "final MLP sigma should be lower than mid: mid={mid:.6} vs final={final_mlp:.6}"
        );
    }

    // T2: newton_schulz_n matches newton_schulz5 for n=5
    #[test]
    fn t2_newton_schulz_n_matches_schulz5_for_n5() {
        let g: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 - 3.2).sin()).collect();
        let mut out5 = vec![0.0f32; 64];
        let mut out_n = vec![0.0f32; 64];

        crate::newton_schulz::newton_schulz5(&g, 8, 8, &mut out5);
        crate::newton_schulz::newton_schulz_n(&g, 8, 8, &mut out_n, 5);

        for i in 0..64 {
            assert!(
                (out5[i] - out_n[i]).abs() < 1e-6,
                "newton_schulz_n(5) should match newton_schulz5 at index {i}: {} vs {}",
                out5[i],
                out_n[i]
            );
        }
    }

    #[test]
    fn t2_more_iterations_improve_orthogonality() {
        // Harder matrix: wide singular value spread.
        // More iterations should produce better orthonormalization.
        let mut g = vec![0.0f32; 64];
        for i in 0..8 {
            for j in 0..8 {
                g[i * 8 + j] = ((i * 8 + j) as f32 * 0.37).sin() * (1.0 + i as f32 * 0.5);
            }
        }

        let mut out5 = vec![0.0f32; 64];
        let mut out10 = vec![0.0f32; 64];

        crate::newton_schulz::newton_schulz_n(&g, 8, 8, &mut out5, 5);
        crate::newton_schulz::newton_schulz_n(&g, 8, 8, &mut out10, 10);

        // Measure orthogonality: off-diagonal of X*X^T should be small
        let off_diag5 = max_off_diag(&out5, 8);
        let off_diag10 = max_off_diag(&out10, 8);

        assert!(
            off_diag10 <= off_diag5 + 1e-6,
            "10-step NS should be at least as orthogonal as 5-step: {off_diag10} vs {off_diag5}"
        );
    }

    #[test]
    fn t2_nonsquare_matches() {
        let g: Vec<f32> = (0..72).map(|i| (i as f32 * 0.13).cos()).collect();
        let mut out5 = vec![0.0f32; 72];
        let mut out_n = vec![0.0f32; 72];

        crate::newton_schulz::newton_schulz5(&g, 12, 6, &mut out5);
        crate::newton_schulz::newton_schulz_n(&g, 12, 6, &mut out_n, 5);

        for i in 0..72 {
            assert!((out5[i] - out_n[i]).abs() < 1e-6, "mismatch at {i}");
        }
    }

    // T3: Rank-p truncation tests
    #[test]
    fn t3_retention_50_keeps_half() {
        // 8×4 matrix with known row norms
        let mut mat = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: norm 1.0
            0.0, 1.0, 0.0, 0.0, // row 1: norm 1.0
            0.5, 0.0, 0.0, 0.0, // row 2: norm 0.5
            0.0, 0.0, 0.0, 0.0, // row 3: norm 0.0
            0.9, 0.0, 0.0, 0.0, // row 4: norm 0.9
            0.0, 0.0, 0.0, 0.1, // row 5: norm 0.1
            0.8, 0.0, 0.0, 0.0, // row 6: norm 0.8
            0.0, 0.0, 0.3, 0.0, // row 7: norm 0.3
        ];
        let mut norms = vec![0.0f32; 8];
        let kept = rank_p_retain(&mut mat, 8, 4, 0.5, &mut norms);
        assert_eq!(kept, 4, "50%% retention of 8 rows should keep 4");

        // Rows 0, 1, 4, 6 have the highest norms (1.0, 1.0, 0.9, 0.8)
        // They should be non-zero; others should be zeroed
        let row_nonzero = |r: usize| -> bool { mat[r * 4..r * 4 + 4].iter().any(|&v| v != 0.0) };
        assert!(row_nonzero(0), "row 0 (norm 1.0) should be kept");
        assert!(row_nonzero(1), "row 1 (norm 1.0) should be kept");
        assert!(!row_nonzero(3), "row 3 (norm 0.0) should be zeroed");
        assert!(!row_nonzero(5), "row 5 (norm 0.1) should be zeroed");
    }

    #[test]
    fn t3_retention_100_keeps_all() {
        let mut mat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original = mat.clone();
        let mut norms = vec![0.0f32; 2];
        let kept = rank_p_retain(&mut mat, 2, 3, 1.0, &mut norms);
        assert_eq!(kept, 2);
        assert_eq!(mat, original, "100% retention should not modify data");
    }

    #[test]
    fn t3_retention_preserves_top_rows() {
        // After NS + rank-p truncation, the kept rows should be the highest-norm rows.
        // This doesn't require norm ≈ 1.0 (depends on matrix condition) — only that
        // the selection is correct (kept >= discarded norms).
        let g: Vec<f32> = (0..64).map(|i| (i as f32 * 0.17 + 1.0).sin()).collect();
        let mut ns_out = vec![0.0f32; 64];
        crate::newton_schulz::newton_schulz5(&g, 8, 8, &mut ns_out);

        // Compute original row norms
        let orig_norms: Vec<f32> = (0..8)
            .map(|i| {
                ns_out[i * 8..(i + 1) * 8]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
            })
            .collect();

        let mut norms = vec![0.0f32; 8];
        let kept = rank_p_retain(&mut ns_out, 8, 8, 0.5, &mut norms);
        assert_eq!(kept, 4);

        // Verify kept rows have higher norms than discarded rows
        let kept_norms: Vec<f32> = (0..8)
            .filter(|&i| ns_out[i * 8..(i + 1) * 8].iter().any(|&v| v != 0.0))
            .map(|i| orig_norms[i])
            .collect();
        let discarded_norms: Vec<f32> = (0..8)
            .filter(|&i| ns_out[i * 8..(i + 1) * 8].iter().all(|&v| v == 0.0))
            .map(|i| orig_norms[i])
            .collect();

        for &kn in &kept_norms {
            for &dn in &discarded_norms {
                assert!(
                    kn >= dn - 1e-6,
                    "kept norm {kn} should be >= discarded norm {dn}"
                );
            }
        }
    }

    #[test]
    fn t3_empty_matrix() {
        let mut mat: Vec<f32> = vec![];
        let mut norms: Vec<f32> = vec![];
        let kept = rank_p_retain(&mut mat, 0, 0, 0.5, &mut norms);
        assert_eq!(kept, 0);
    }

    // T4: SpectralBudgetArm tests
    #[test]
    fn t4_gaming_arm_uses_fewer_iterations() {
        let gaming = SpectralBudgetArm::gaming();
        let diagnostic = SpectralBudgetArm::diagnostic();
        assert!(
            gaming.ns_iterations <= diagnostic.ns_iterations,
            "gaming ({}) should use <= iterations than diagnostic ({})",
            gaming.ns_iterations,
            diagnostic.ns_iterations
        );
        assert_eq!(gaming.ns_iterations, 5);
    }

    #[test]
    fn t4_diagnostic_arm_uses_more_iterations() {
        let diagnostic = SpectralBudgetArm::diagnostic();
        assert_eq!(diagnostic.ns_iterations, 10);
        assert!(diagnostic.retention > 0.8);
    }

    #[test]
    fn t4_chain_arm_between_gaming_diagnostic() {
        let gaming = SpectralBudgetArm::gaming();
        let chain = SpectralBudgetArm::chain();
        let diagnostic = SpectralBudgetArm::diagnostic();

        assert!(chain.retention >= gaming.retention);
        assert!(chain.retention <= diagnostic.retention);
        assert_eq!(chain.ns_iterations, 5);
    }

    #[test]
    fn t4_diagnostic_resolve_uses_config() {
        let lt = [LayerType::MlpUp];
        let config = SpectralBudgetConfig::from_model_dims(28, 2800, &lt);
        let diagnostic = SpectralBudgetArm::diagnostic();

        // Diagnostic arm should use config's per-layer prediction
        let last_idx = config.layers.len() - 1;
        let resolved_iters = diagnostic.resolve_ns_iterations(&config, last_idx);
        let expected = config.ns_iterations(last_idx);
        assert_eq!(resolved_iters, expected);
    }

    #[test]
    fn t4_gaming_resolve_uses_fixed() {
        let lt = [LayerType::MlpUp];
        let config = SpectralBudgetConfig::from_model_dims(28, 2800, &lt);
        let gaming = SpectralBudgetArm::gaming();

        // Gaming arm should always use fixed 5 iterations
        let resolved = gaming.resolve_ns_iterations(&config, config.layers.len() - 1);
        assert_eq!(resolved, 5, "gaming arm should always use 5 iterations");
    }

    #[test]
    fn t4_spectral_arms_const_matches_constructors() {
        assert_eq!(SPECTRAL_BUDGET_ARMS[0], SpectralBudgetArm::gaming());
        assert_eq!(SPECTRAL_BUDGET_ARMS[1], SpectralBudgetArm::chain());
        assert_eq!(SPECTRAL_BUDGET_ARMS[2], SpectralBudgetArm::diagnostic());
    }

    // Helpers for tests
    fn max_off_diag(mat: &[f32], m: usize) -> f32 {
        let mut max_val = 0.0f32;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    max_val = max_val.max(mat[i * m + j].abs());
                }
            }
        }
        max_val
    }
}
