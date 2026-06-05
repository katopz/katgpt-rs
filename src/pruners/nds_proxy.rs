//! NDS Curvature Proxy — Modelless Inference-Time Budget Control (Plan 186).
//!
//! Paper (arXiv:2606.04662) proves Muon's advantage comes from lower NDS
//! (Normalized Directional Sharpness). At inference, we approximate from
//! marginal distributions using spectral flatness as a proxy.

/// Inference-time NDS (Normalized Directional Sharpness) proxy.
///
/// Returns value in [0, 1]: 1 = peaked/confident, 0 = flat/uncertain.
///
/// Uses spectral flatness (geometric mean / arithmetic mean) as a proxy:
/// - High NDS ≈ peaked distribution (few tokens dominate) → confident
/// - Low NDS ≈ flat distribution (many tokens compete) → uncertain
#[inline]
pub fn nds_proxy(top_k_probs: &[f32]) -> f32 {
    if top_k_probs.is_empty() {
        return 0.5;
    }
    let n = top_k_probs.len() as f32;
    let am = top_k_probs.iter().sum::<f32>() / n;
    if am <= 0.0 {
        return 0.5;
    }
    let ln_sum: f32 = top_k_probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|p| p.ln())
        .sum();
    let gm = (ln_sum / n).exp();
    // Spectral flatness = gm/am ∈ [0, 1]
    // NDS proxy = 1 - flatness
    (1.0 - gm / am).clamp(0.0, 1.0)
}

/// Spectral balance score for DDTree branch visit distribution.
/// Returns ∈ [0, 1]: 1.0 = perfectly balanced, 0.0 = all on one branch.
pub fn spectral_balance_score(visit_counts: &[u32]) -> f32 {
    let total: u32 = visit_counts.iter().sum();
    if total == 0 {
        return 1.0;
    }
    let n = visit_counts.len() as f32;
    if n <= 1.0 {
        return 1.0;
    }
    let entropy: f32 = visit_counts
        .iter()
        .filter(|&&v| v > 0)
        .map(|&v| {
            let p = v as f32 / total as f32;
            -p * p.log2()
        })
        .sum();
    (entropy / n.log2()).clamp(0.0, 1.0)
}

/// Layer depth classification for NDS-aware budget allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerDepth {
    Boundary = 3,
    Deep = 2,
    Middle = 1,
}

/// Classify layer depth for budget allocation decisions.
pub fn layer_nds_depth(layer_idx: usize, total_layers: usize) -> LayerDepth {
    let is_boundary = layer_idx == 0 || layer_idx == total_layers - 1;
    let is_deep = !is_boundary && layer_idx >= total_layers * 7 / 10;
    match (is_boundary, is_deep) {
        (true, _) => LayerDepth::Boundary,
        (false, true) => LayerDepth::Deep,
        _ => LayerDepth::Middle,
    }
}

/// Budget modifier based on NDS proxy.
pub trait NdsBudgetModifier: Send + Sync {
    fn budget_scale(&self, nds: f32) -> f32;
}

/// Spectral flatness budget modifier.
/// Scale = 1.0 + (1 - NDS) * max_boost ∈ [1.0, 1.0 + max_boost]
pub struct SpectralFlatnessBudget {
    pub max_boost: f32,
}

impl SpectralFlatnessBudget {
    pub fn new(max_boost: f32) -> Self {
        Self { max_boost }
    }
}

impl NdsBudgetModifier for SpectralFlatnessBudget {
    fn budget_scale(&self, nds: f32) -> f32 {
        1.0 + (1.0 - nds.clamp(0.0, 1.0)) * self.max_boost
    }
}

/// Compose NDS budget modifier with existing budget allocation.
/// Takes a base budget and scales it by NDS proxy value.
#[inline]
pub fn nds_scaled_budget(
    base_budget: usize,
    top_k_probs: &[f32],
    modifier: &dyn NdsBudgetModifier,
) -> usize {
    let nds = nds_proxy(top_k_probs);
    let scale = modifier.budget_scale(nds);
    ((base_budget as f32 * scale).ceil()) as usize
}

/// Apply spectral balance bonus to bandit arm scores.
/// Balanced exploration (high spectral balance) gets a bonus,
/// encouraging diverse arm selection in the DDTree.
#[inline]
pub fn spectral_balance_bonus(visit_counts: &[u32], weight: f32) -> f32 {
    let balance = spectral_balance_score(visit_counts);
    1.0 + balance * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inline unit tests ─────────────────────────────────────────

    #[test]
    fn test_nds_proxy_peaked() {
        let result = nds_proxy(&[0.9, 0.05, 0.05]);
        assert!(
            result > 0.5,
            "peaked distribution should have NDS > 0.5, got {result}"
        );
    }

    #[test]
    fn test_nds_proxy_flat() {
        let result = nds_proxy(&[0.33, 0.33, 0.34]);
        assert!(
            result < 0.5,
            "flat distribution should have NDS < 0.5, got {result}"
        );
    }

    #[test]
    fn test_nds_proxy_empty() {
        assert_eq!(nds_proxy(&[]), 0.5);
    }

    #[test]
    fn test_nds_proxy_single() {
        // Single element: geometric mean = arithmetic mean → NDS = 0
        let result = nds_proxy(&[0.5]);
        assert!(
            result.abs() < 1e-6,
            "single element should have NDS ≈ 0, got {result}"
        );
    }

    #[test]
    fn test_spectral_balance_balanced() {
        let result = spectral_balance_score(&[10, 10, 10]);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "balanced distribution should score ~1.0, got {result}"
        );
    }

    #[test]
    fn test_spectral_balance_imbalanced() {
        let result = spectral_balance_score(&[100, 1, 1]);
        assert!(
            result < 0.5,
            "imbalanced distribution should score low, got {result}"
        );
    }

    #[test]
    fn test_spectral_balance_empty() {
        assert_eq!(spectral_balance_score(&[]), 1.0);
    }

    #[test]
    fn test_layer_depth_boundary_first() {
        assert_eq!(layer_nds_depth(0, 12), LayerDepth::Boundary);
    }

    #[test]
    fn test_layer_depth_boundary_last() {
        assert_eq!(layer_nds_depth(11, 12), LayerDepth::Boundary);
    }

    #[test]
    fn test_layer_depth_middle() {
        assert_eq!(layer_nds_depth(3, 12), LayerDepth::Middle);
    }

    #[test]
    fn test_layer_depth_deep() {
        // 70% of 12 = 8.4 → layer_idx 9 >= 8
        assert_eq!(layer_nds_depth(9, 12), LayerDepth::Deep);
    }

    #[test]
    fn test_budget_scale_confident() {
        let modr = SpectralFlatnessBudget::new(0.5);
        let scale = modr.budget_scale(1.0);
        assert!(
            (scale - 1.0).abs() < 1e-6,
            "confident (NDS=1.0) should scale to 1.0, got {scale}"
        );
    }

    #[test]
    fn test_budget_scale_uncertain() {
        let max_boost = 0.5f32;
        let modr = SpectralFlatnessBudget::new(max_boost);
        let scale = modr.budget_scale(0.0);
        let expected = 1.0 + max_boost;
        assert!(
            (scale - expected).abs() < 1e-6,
            "uncertain (NDS=0.0) should scale to 1.0 + max_boost, got {scale}"
        );
    }
}
