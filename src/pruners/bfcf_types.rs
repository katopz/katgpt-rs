//! BFCF Tree core types — Borel Finite Connected Partition (Plan 213 Phase 1).
//!
//! Types for perceptual region folding: token space partitioned into convex BFCP regions
//! where all tokens are symbolically equivalent (same accept/reject/maybe label).

use std::fmt;

// ── RegionLabel ─────────────────────────────────────────────────

/// Label for a BFCP region — output of the perception function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RegionLabel {
    Accept,
    Reject,
    Maybe,
}

impl fmt::Display for RegionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionLabel::Accept => write!(f, "accept"),
            RegionLabel::Reject => write!(f, "reject"),
            RegionLabel::Maybe => write!(f, "maybe"),
        }
    }
}

// ── HalfSpace ───────────────────────────────────────────────────

/// Half-space constraint defining one face of the polytope.
#[derive(Clone, Debug)]
pub struct HalfSpace {
    /// Logit dimension index.
    pub dim: u16,
    /// Threshold value.
    pub threshold: f32,
    /// `true` = logit[dim] >= threshold, `false` = logit[dim] < threshold.
    pub above: bool,
}

impl HalfSpace {
    /// Check if a logit vector satisfies this half-space constraint.
    #[inline]
    pub fn contains(&self, logits: &[f32]) -> bool {
        let val = logits.get(self.dim as usize).copied().unwrap_or(0.0);
        match self.above {
            true => val >= self.threshold,
            false => val < self.threshold,
        }
    }
}

// ── BorelRegion ─────────────────────────────────────────────────

/// Contiguous region of logit space — convex polytope from ReLU thresholds.
#[derive(Clone, Debug)]
pub struct BorelRegion {
    /// Half-space constraints defining the polytope.
    pub constraints: Vec<HalfSpace>,
    /// Symbolic label from screening.
    pub label: RegionLabel,
    /// Number of tokens within this region.
    pub token_count: usize,
}

impl BorelRegion {
    /// Create a new BorelRegion with the given label, constraints, and token count.
    pub fn new(label: RegionLabel, constraints: Vec<HalfSpace>, token_count: usize) -> Self {
        Self {
            constraints,
            label,
            token_count,
        }
    }

    /// Check if a logit vector satisfies all constraints of this region.
    pub fn contains(&self, logits: &[f32]) -> bool {
        self.constraints.iter().all(|hs| hs.contains(logits))
    }

    /// Intersect this region with another, combining constraints.
    /// Returns `None` if the intersection is empty (contradictory constraints).
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let mut combined = self.constraints.clone();
        combined.extend_from_slice(&other.constraints);

        // Check for contradictions: same dim, same above flag, incompatible thresholds
        for a in &self.constraints {
            for b in &other.constraints {
                if a.dim == b.dim && a.above != b.above {
                    // One says >= t1, other says < t2. Empty if t1 >= t2.
                    if a.above && a.threshold >= b.threshold {
                        return None;
                    }
                    if b.above && b.threshold >= a.threshold {
                        return None;
                    }
                }
            }
        }

        // Label: intersect of accept+accept=accept, reject+anything=reject, else maybe
        let label = match (self.label, other.label) {
            (RegionLabel::Reject, _) | (_, RegionLabel::Reject) => RegionLabel::Reject,
            (RegionLabel::Accept, RegionLabel::Accept) => RegionLabel::Accept,
            _ => RegionLabel::Maybe,
        };

        Some(BorelRegion::new(
            label,
            combined,
            self.token_count.min(other.token_count),
        ))
    }
}

// ── BFCP ────────────────────────────────────────────────────────

/// BFCP — Borel Finite Connected Partition of logit space.
#[derive(Clone, Debug)]
pub struct BFCP {
    pub regions: Vec<BorelRegion>,
}

impl BFCP {
    /// Create an empty partition.
    pub fn empty() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Create a partition from a vector of regions.
    pub fn from_regions(regions: Vec<BorelRegion>) -> Self {
        Self { regions }
    }

    /// Number of regions in the partition.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Total tokens across all regions.
    pub fn total_tokens(&self) -> usize {
        self.regions.iter().map(|r| r.token_count).sum()
    }

    /// Check if the partition covers the full vocabulary (sum of token_counts matches).
    pub fn covers_all(&self, vocab_size: usize) -> bool {
        self.total_tokens() == vocab_size
    }

    /// Get all accept regions.
    pub fn accept_regions(&self) -> impl Iterator<Item = &BorelRegion> {
        self.regions
            .iter()
            .filter(|r| r.label == RegionLabel::Accept)
    }

    /// Get all reject regions.
    pub fn reject_regions(&self) -> impl Iterator<Item = &BorelRegion> {
        self.regions
            .iter()
            .filter(|r| r.label == RegionLabel::Reject)
    }

    /// Get all maybe regions.
    pub fn maybe_regions(&self) -> impl Iterator<Item = &BorelRegion> {
        self.regions
            .iter()
            .filter(|r| r.label == RegionLabel::Maybe)
    }

    /// Count of accept regions.
    pub fn accept_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| r.label == RegionLabel::Accept)
            .count()
    }

    /// Count of reject regions.
    pub fn reject_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| r.label == RegionLabel::Reject)
            .count()
    }

    /// Count of maybe regions.
    pub fn maybe_count(&self) -> usize {
        self.regions
            .iter()
            .filter(|r| r.label == RegionLabel::Maybe)
            .count()
    }

    /// Tokens in accept regions.
    pub fn accept_token_count(&self) -> usize {
        self.accept_regions().map(|r| r.token_count).sum()
    }

    /// Tokens in reject regions.
    pub fn reject_token_count(&self) -> usize {
        self.reject_regions().map(|r| r.token_count).sum()
    }

    /// Tokens in maybe regions.
    pub fn maybe_token_count(&self) -> usize {
        self.maybe_regions().map(|r| r.token_count).sum()
    }
}

// ── PWCValueFunction ────────────────────────────────────────────

/// Piecewise-constant value function over BFCP regions.
///
/// Each region maps to exactly one scalar value. Theorem 2 (NS-CSG):
/// after Bellman backup, values remain piecewise-constant — no leakage.
#[derive(Clone, Debug)]
pub struct PWCValueFunction {
    /// (region_index, value) — constant per region.
    pub region_values: Vec<(usize, f64)>,
}

impl PWCValueFunction {
    /// Create a new PWC value function with `region_count` regions, all initialized to `initial`.
    pub fn new(region_count: usize, initial: f64) -> Self {
        Self {
            region_values: (0..region_count).map(|i| (i, initial)).collect(),
        }
    }

    /// Get value for a specific region. Returns 0.0 if region not found.
    pub fn value(&self, region_idx: usize) -> f64 {
        self.region_values
            .iter()
            .find(|(idx, _)| *idx == region_idx)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }

    /// Update value for a specific region.
    pub fn update(&mut self, region_idx: usize, new_value: f64) {
        if let Some(entry) = self
            .region_values
            .iter_mut()
            .find(|(idx, _)| *idx == region_idx)
        {
            entry.1 = new_value;
        }
    }

    /// Number of regions.
    pub fn len(&self) -> usize {
        self.region_values.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.region_values.is_empty()
    }

    /// Verify PWC closure: each region has exactly one value (no duplicates).
    /// After updates, values haven't leaked between regions — structural invariant.
    pub fn verify_pwc_closure(&self) -> bool {
        // PWC closure: each region index appears exactly once
        let len = self.region_values.len();
        let unique_count: usize = {
            let mut indices: Vec<usize> = self.region_values.iter().map(|(i, _)| *i).collect();
            indices.sort();
            indices.dedup();
            indices.len()
        };
        unique_count == len
    }
}

// ── BfcpPartition Trait ────────────────────────────────────────

/// Extension trait for ScreeningPruner to produce BFCP partitions.
#[cfg(feature = "bfcf_tree")]
pub trait BfcpPartition: Send + Sync {
    /// Compute BFCP from current screening decisions.
    fn partition(&self, logits: &[f32]) -> BFCP;
    /// Refine a "maybe" region into sub-regions.
    fn refine(&self, region: &BorelRegion, prefix: &[usize]) -> Vec<BorelRegion>;
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_label_display() {
        assert_eq!(format!("{}", RegionLabel::Accept), "accept");
        assert_eq!(format!("{}", RegionLabel::Reject), "reject");
        assert_eq!(format!("{}", RegionLabel::Maybe), "maybe");
    }

    #[test]
    fn test_halfspace_contains() {
        let hs = HalfSpace {
            dim: 0,
            threshold: 0.5,
            above: true,
        };
        assert!(hs.contains(&[1.0, 0.0]));
        assert!(!hs.contains(&[0.3, 0.0]));
    }

    #[test]
    fn test_borel_region_contains() {
        let region = BorelRegion::new(
            RegionLabel::Accept,
            vec![
                HalfSpace {
                    dim: 0,
                    threshold: 0.5,
                    above: true,
                },
                HalfSpace {
                    dim: 1,
                    threshold: 0.3,
                    above: false,
                },
            ],
            10,
        );
        assert!(region.contains(&[0.8, 0.1]));
        assert!(!region.contains(&[0.3, 0.1])); // fails dim 0
        assert!(!region.contains(&[0.8, 0.5])); // fails dim 1
    }

    #[test]
    fn test_borel_region_intersect() {
        let r1 = BorelRegion::new(
            RegionLabel::Accept,
            vec![HalfSpace {
                dim: 0,
                threshold: 0.5,
                above: true,
            }],
            10,
        );
        let r2 = BorelRegion::new(
            RegionLabel::Accept,
            vec![HalfSpace {
                dim: 1,
                threshold: 0.3,
                above: false,
            }],
            8,
        );
        let intersection = r1.intersect(&r2).unwrap();
        assert_eq!(intersection.label, RegionLabel::Accept);
        assert_eq!(intersection.constraints.len(), 2);
        assert_eq!(intersection.token_count, 8); // min(10, 8)
    }

    #[test]
    fn test_borel_region_intersect_contradiction() {
        let r1 = BorelRegion::new(
            RegionLabel::Accept,
            vec![HalfSpace {
                dim: 0,
                threshold: 0.5,
                above: true,
            }],
            10,
        );
        let r2 = BorelRegion::new(
            RegionLabel::Reject,
            vec![HalfSpace {
                dim: 0,
                threshold: 0.5,
                above: false,
            }],
            8,
        );
        // >= 0.5 AND < 0.5 → empty
        assert!(r1.intersect(&r2).is_none());
    }

    #[test]
    fn test_bfcp_covers_all() {
        let bfcp = BFCP::from_regions(vec![
            BorelRegion::new(RegionLabel::Accept, vec![], 60),
            BorelRegion::new(RegionLabel::Reject, vec![], 30),
            BorelRegion::new(RegionLabel::Maybe, vec![], 10),
        ]);
        assert!(bfcp.covers_all(100));
        assert!(!bfcp.covers_all(99));
    }

    #[test]
    fn test_bfcp_region_counts() {
        let bfcp = BFCP::from_regions(vec![
            BorelRegion::new(RegionLabel::Accept, vec![], 60),
            BorelRegion::new(RegionLabel::Reject, vec![], 30),
            BorelRegion::new(RegionLabel::Maybe, vec![], 10),
        ]);
        assert_eq!(bfcp.accept_count(), 1);
        assert_eq!(bfcp.reject_count(), 1);
        assert_eq!(bfcp.maybe_count(), 1);
        assert_eq!(bfcp.accept_token_count(), 60);
        assert_eq!(bfcp.reject_token_count(), 30);
        assert_eq!(bfcp.maybe_token_count(), 10);
    }

    #[test]
    fn test_pwc_value_function_get_update() {
        let mut vf = PWCValueFunction::new(5, 0.0);
        assert_eq!(vf.value(0), 0.0);
        assert_eq!(vf.value(4), 0.0);

        vf.update(2, 1.5);
        assert_eq!(vf.value(2), 1.5);
        assert_eq!(vf.value(0), 0.0); // unchanged
    }

    #[test]
    fn test_pwc_closure_maintained() {
        let mut vf = PWCValueFunction::new(10, 0.5);
        assert!(vf.verify_pwc_closure());

        vf.update(3, 0.9);
        vf.update(7, 0.1);
        vf.update(0, 1.0);
        assert!(vf.verify_pwc_closure());
    }
}
