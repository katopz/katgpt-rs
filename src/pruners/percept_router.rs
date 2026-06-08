//! Symbolic Percept Router — maps BFCP partition to routing decision (Plan 213 P4).
//!
//! Uses region count and label entropy (sigmoid, not softmax) for compute routing.
//! Low complexity → FastPath, high complexity → DeepThink, otherwise → Standard.

use super::bfcf_types::{BFCP, RegionLabel};

// ── sigmoid helper ──────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── ComputePath ─────────────────────────────────────────────────

/// Compute path decision from percept analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComputePath {
    /// Fast path — few regions, mostly accept/reject.
    FastPath,
    /// Standard path — moderate complexity.
    Standard,
    /// Deep think — many regions, high maybe ratio.
    DeepThink,
}

// ── PerceptRouterConfig ─────────────────────────────────────────

/// Symbolic percept router configuration.
#[derive(Debug, Clone)]
pub struct PerceptRouterConfig {
    /// Complexity threshold for fast → standard transition.
    pub fast_threshold: f32,
    /// Complexity threshold for standard → deep transition.
    pub deep_threshold: f32,
}

impl Default for PerceptRouterConfig {
    fn default() -> Self {
        Self {
            fast_threshold: 0.3,
            deep_threshold: 0.7,
        }
    }
}

impl PerceptRouterConfig {
    /// Create a new config with the given thresholds.
    pub fn new(fast_threshold: f32, deep_threshold: f32) -> Self {
        Self {
            fast_threshold,
            deep_threshold,
        }
    }
}

// ── PerceptRouter trait ─────────────────────────────────────────

/// Trait for routing based on BFCP partition.
pub trait PerceptRouter: Send + Sync {
    /// Route to a compute path based on BFCP partition complexity.
    fn route(&self, bfcp: &BFCP) -> ComputePath;
    /// Compute complexity measure for the partition (sigmoid-bounded [0, 1]).
    fn complexity(&self, bfcp: &BFCP) -> f32;
}

// ── SigmoidPerceptRouter ────────────────────────────────────────

/// Routes based on symbolic percept of BFCP partition.
///
/// Complexity = `sigmoid(region_count * entropy_of_labels)`.
/// Uses sigmoid (never softmax) — bounded [0, 1].
pub struct SigmoidPerceptRouter {
    config: PerceptRouterConfig,
}

impl SigmoidPerceptRouter {
    /// Create a new router with the given configuration.
    pub fn new(config: PerceptRouterConfig) -> Self {
        Self { config }
    }

    /// Create a router with default thresholds.
    pub fn default_router() -> Self {
        Self::new(PerceptRouterConfig::default())
    }
}

/// Compute Shannon entropy of the label distribution across regions.
///
/// Each region is one sample. Returns 0.0 for single-label partitions.
fn label_entropy(bfcp: &BFCP) -> f32 {
    let n = bfcp.region_count();
    if n <= 1 {
        return 0.0;
    }

    let mut counts = [0usize; 3]; // [accept, reject, maybe]
    for region in &bfcp.regions {
        match region.label {
            RegionLabel::Accept => counts[0] += 1,
            RegionLabel::Reject => counts[1] += 1,
            RegionLabel::Maybe => counts[2] += 1,
        }
    }

    let n_f = n as f32;
    let mut entropy = 0.0f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / n_f;
            entropy -= p * p.ln();
        }
    }

    entropy
}

impl PerceptRouter for SigmoidPerceptRouter {
    fn complexity(&self, bfcp: &BFCP) -> f32 {
        let region_count = bfcp.region_count();
        if region_count == 0 {
            return 0.0;
        }

        let entropy = label_entropy(bfcp);
        // sigmoid(region_count * entropy) — bounded [0, 1]
        sigmoid(region_count as f32 * entropy)
    }

    fn route(&self, bfcp: &BFCP) -> ComputePath {
        let c = self.complexity(bfcp);
        if c < self.config.fast_threshold {
            ComputePath::FastPath
        } else if c > self.config.deep_threshold {
            ComputePath::DeepThink
        } else {
            ComputePath::Standard
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bfcf_types::{BorelRegion, HalfSpace};

    /// Helper: create a BFCP with given token counts per label.
    fn make_partition(
        accept_regions: &[(usize, usize)],
        reject_regions: &[(usize, usize)],
        maybe_regions: &[(usize, usize)],
    ) -> BFCP {
        let mut regions = Vec::new();
        for &(tokens, constraints) in accept_regions {
            let hs: Vec<HalfSpace> = (0..constraints)
                .map(|dim| HalfSpace {
                    dim: dim as u16,
                    threshold: 0.5,
                    above: true,
                })
                .collect();
            regions.push(BorelRegion::new(RegionLabel::Accept, hs, tokens));
        }
        for &(tokens, constraints) in reject_regions {
            let hs: Vec<HalfSpace> = (0..constraints)
                .map(|dim| HalfSpace {
                    dim: dim as u16,
                    threshold: 0.5,
                    above: false,
                })
                .collect();
            regions.push(BorelRegion::new(RegionLabel::Reject, hs, tokens));
        }
        for &(tokens, constraints) in maybe_regions {
            let hs: Vec<HalfSpace> = (0..constraints)
                .map(|dim| HalfSpace {
                    dim: dim as u16,
                    threshold: 0.5,
                    above: true,
                })
                .collect();
            regions.push(BorelRegion::new(RegionLabel::Maybe, hs, tokens));
        }
        BFCP::from_regions(regions)
    }

    #[test]
    fn test_complexity_low_for_simple_partition() {
        // All accept, single region → entropy = 0, complexity ≈ sigmoid(0) = 0.5
        // Actually sigmoid(0) = 0.5, so let's check it's bounded low
        let bfcp = make_partition(&[(100, 0)], &[], &[]);
        let router = SigmoidPerceptRouter::default_router();
        let c = router.complexity(&bfcp);
        // Single region → entropy = 0 → sigmoid(1 * 0) = sigmoid(0) = 0.5
        assert!(
            (c - 0.5).abs() < 0.01,
            "single-region complexity should be ~0.5 (sigmoid(0)), got {}",
            c
        );
    }

    #[test]
    fn test_complexity_high_for_complex_partition() {
        // Many mixed regions → high entropy → high complexity
        let mut regions = Vec::new();
        for i in 0..30 {
            let label = match i % 3 {
                0 => RegionLabel::Accept,
                1 => RegionLabel::Reject,
                _ => RegionLabel::Maybe,
            };
            regions.push(BorelRegion::new(label, vec![], 10));
        }
        let bfcp = BFCP::from_regions(regions);

        let router = SigmoidPerceptRouter::default_router();
        let c = router.complexity(&bfcp);
        // 30 regions * ln(3) ≈ 30 * 1.099 ≈ 33 → sigmoid(33) ≈ 1.0
        assert!(
            c > 0.99,
            "complex partition should have high complexity, got {}",
            c
        );
    }

    #[test]
    fn test_route_fast_for_simple() {
        // Simple: mostly one label
        // Use custom config with higher fast_threshold so sigmoid(0)=0.5 < 0.7 triggers fast
        let bfcp = make_partition(&[(100, 0)], &[], &[]);
        let router = SigmoidPerceptRouter::new(PerceptRouterConfig::new(0.6, 0.8));
        assert_eq!(router.route(&bfcp), ComputePath::FastPath);
    }

    #[test]
    fn test_route_deep_for_complex() {
        // Many mixed regions
        let mut regions = Vec::new();
        for i in 0..30 {
            let label = match i % 3 {
                0 => RegionLabel::Accept,
                1 => RegionLabel::Reject,
                _ => RegionLabel::Maybe,
            };
            regions.push(BorelRegion::new(label, vec![], 10));
        }
        let bfcp = BFCP::from_regions(regions);

        let router = SigmoidPerceptRouter::default_router();
        assert_eq!(router.route(&bfcp), ComputePath::DeepThink);
    }

    #[test]
    fn test_route_standard_for_medium() {
        // 3 regions, all same label → entropy = 0 → sigmoid(3 * 0) = 0.5 → standard
        let bfcp = make_partition(&[(50, 0), (30, 0), (20, 0)], &[], &[]);
        let router = SigmoidPerceptRouter::default_router();
        // entropy=0 → sigmoid(3*0) = sigmoid(0) = 0.5 → between 0.3 and 0.7 → Standard
        assert_eq!(router.route(&bfcp), ComputePath::Standard);
    }

    #[test]
    fn test_complexity_bounded_unit_interval() {
        let router = SigmoidPerceptRouter::default_router();

        // Empty partition
        let empty = BFCP::empty();
        assert!(
            (0.0..=1.0).contains(&router.complexity(&empty)),
            "complexity should be in [0, 1]"
        );

        // Simple partition
        let simple = make_partition(&[(100, 0)], &[], &[]);
        assert!(
            (0.0..=1.0).contains(&router.complexity(&simple)),
            "complexity should be in [0, 1]"
        );

        // Complex partition
        let mut regions = Vec::new();
        for i in 0..50 {
            let label = match i % 3 {
                0 => RegionLabel::Accept,
                1 => RegionLabel::Reject,
                _ => RegionLabel::Maybe,
            };
            regions.push(BorelRegion::new(label, vec![], 10));
        }
        let complex = BFCP::from_regions(regions);
        assert!(
            (0.0..=1.0).contains(&router.complexity(&complex)),
            "complexity should be in [0, 1]"
        );
    }

    #[test]
    fn test_entropy_of_uniform_labels() {
        // Equal distribution: 10 accept, 10 reject, 10 maybe
        let mut regions = Vec::new();
        for _ in 0..10 {
            regions.push(BorelRegion::new(RegionLabel::Accept, vec![], 5));
        }
        for _ in 0..10 {
            regions.push(BorelRegion::new(RegionLabel::Reject, vec![], 5));
        }
        for _ in 0..10 {
            regions.push(BorelRegion::new(RegionLabel::Maybe, vec![], 5));
        }
        let bfcp = BFCP::from_regions(regions);

        let entropy = label_entropy(&bfcp);
        let expected = (3.0f32).ln(); // ln(3) ≈ 1.099
        assert!(
            (entropy - expected).abs() < 0.01,
            "uniform distribution entropy should be ln(3) ≈ 1.099, got {}",
            entropy
        );
    }
}
