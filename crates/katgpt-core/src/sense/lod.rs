//! Spectral NPC Perception Compression (Plan 240).
//!
//! Level-of-detail routing for sense modules based on spectral scale boundaries.

use crate::slod::ScaleBoundary;
use crate::types::SenseKind;

/// Level-of-detail for sense module activation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SenseLodLevel {
    #[default]
    Full = 0,
    Compressed = 1,
    Minimal = 2,
}

impl SenseLodLevel {
    /// Returns the set of active sense modules for this LOD level.
    pub fn module_mask(self) -> &'static [SenseKind] {
        use SenseKind::*;
        match self {
            Self::Full => &[
                CommonSense,
                FighterSense,
                GameTheorySense,
                SpatialSense,
                SocialSense,
                SkillSense,
            ],
            Self::Compressed => &[CommonSense, FighterSense, SpatialSense],
            Self::Minimal => &[SpatialSense],
        }
    }
}

/// Routes distances to LOD levels using spectral scale boundaries.
#[derive(Clone, Debug)]
pub struct SenseLodRouter {
    boundaries: Vec<ScaleBoundary>,
    sigma1: f32,
    sigma2: f32,
}

impl SenseLodRouter {
    pub fn new(boundaries: Vec<ScaleBoundary>, sigma1: f32, sigma2: f32) -> Self {
        Self {
            boundaries,
            sigma1,
            sigma2,
        }
    }

    pub fn route(&self, distance: f32) -> SenseLodLevel {
        if distance <= self.sigma1 {
            return SenseLodLevel::Full;
        }
        if distance <= self.sigma2 {
            return SenseLodLevel::Compressed;
        }
        SenseLodLevel::Minimal
    }

    pub fn from_boundaries(boundaries: &[ScaleBoundary]) -> Option<Self> {
        if boundaries.len() < 2 {
            return None;
        }
        let sigma1 = boundaries[0].sigma;
        let sigma2 = boundaries[1].sigma;
        Some(Self {
            boundaries: boundaries.to_vec(),
            sigma1,
            sigma2,
        })
    }

    pub fn assign_lods(&self, distances: &[f32]) -> Vec<SenseLodLevel> {
        distances.iter().map(|&d| self.route(d)).collect()
    }
}

/// Fast boolean mask for sense module activation, indexed by `SenseKind` discriminant.
#[derive(Clone, Copy, Debug)]
pub struct SenseLodMask {
    mask: [bool; 6],
}

impl SenseLodMask {
    pub fn from_level(level: SenseLodLevel) -> Self {
        let mut mask = [false; 6];
        for &kind in level.module_mask() {
            let idx = kind as usize;
            if idx < 6 {
                mask[idx] = true;
            }
        }
        Self { mask }
    }

    pub fn is_active(&self, kind: SenseKind) -> bool {
        let idx = kind as usize;
        if idx < 6 {
            return self.mask[idx];
        }
        false
    }

    pub fn active_count(&self) -> usize {
        self.mask.iter().filter(|&&b| b).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slod::ScaleBoundary;

    fn router() -> SenseLodRouter {
        SenseLodRouter::new(vec![boundary(10.0), boundary(50.0)], 10.0, 50.0)
    }

    fn boundary(sigma: f32) -> ScaleBoundary {
        ScaleBoundary {
            sigma,
            k_star: 4,
            score: 1.0,
        }
    }

    #[test]
    fn test_lod_level_mask_correctness() {
        use SenseKind::*;
        let full = SenseLodLevel::Full.module_mask();
        assert!(full.contains(&CommonSense));
        assert!(full.contains(&FighterSense));
        assert!(full.contains(&GameTheorySense));
        assert!(full.contains(&SpatialSense));
        assert!(full.contains(&SocialSense));
        assert!(full.contains(&SkillSense));
        assert_eq!(full.len(), 6);

        let compressed = SenseLodLevel::Compressed.module_mask();
        assert!(compressed.contains(&CommonSense));
        assert!(compressed.contains(&FighterSense));
        assert!(compressed.contains(&SpatialSense));
        assert_eq!(compressed.len(), 3);

        let minimal = SenseLodLevel::Minimal.module_mask();
        assert!(minimal.contains(&SpatialSense));
        assert_eq!(minimal.len(), 1);
    }

    #[test]
    fn test_lod_router_within_sigma1() {
        assert_eq!(router().route(0.0), SenseLodLevel::Full);
        assert_eq!(router().route(5.0), SenseLodLevel::Full);
        assert_eq!(router().route(10.0), SenseLodLevel::Full);
    }

    #[test]
    fn test_lod_router_between_sigmas() {
        assert_eq!(router().route(10.1), SenseLodLevel::Compressed);
        assert_eq!(router().route(30.0), SenseLodLevel::Compressed);
        assert_eq!(router().route(50.0), SenseLodLevel::Compressed);
    }

    #[test]
    fn test_lod_router_beyond_sigma2() {
        assert_eq!(router().route(50.1), SenseLodLevel::Minimal);
        assert_eq!(router().route(1000.0), SenseLodLevel::Minimal);
    }

    #[test]
    fn test_lod_mask_active_count() {
        assert_eq!(
            SenseLodMask::from_level(SenseLodLevel::Full).active_count(),
            6
        );
        assert_eq!(
            SenseLodMask::from_level(SenseLodLevel::Compressed).active_count(),
            3
        );
        assert_eq!(
            SenseLodMask::from_level(SenseLodLevel::Minimal).active_count(),
            1
        );
    }

    #[test]
    fn test_from_boundaries_empty() {
        assert!(SenseLodRouter::from_boundaries(&[]).is_none());
        assert!(SenseLodRouter::from_boundaries(&[boundary(5.0)]).is_none());
    }

    #[test]
    fn test_from_boundaries_with_data() {
        let r = SenseLodRouter::from_boundaries(&[boundary(10.0), boundary(50.0)]).unwrap();
        assert_eq!(r.route(5.0), SenseLodLevel::Full);
        assert_eq!(r.route(30.0), SenseLodLevel::Compressed);
        assert_eq!(r.route(100.0), SenseLodLevel::Minimal);
    }

    #[test]
    fn test_assign_lods() {
        let r = router();
        let lods = r.assign_lods(&[5.0, 30.0, 100.0]);
        assert_eq!(
            lods,
            vec![
                SenseLodLevel::Full,
                SenseLodLevel::Compressed,
                SenseLodLevel::Minimal
            ]
        );
    }
}
