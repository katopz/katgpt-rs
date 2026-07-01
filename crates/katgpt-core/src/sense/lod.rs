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
    sigma1: f32,
    sigma2: f32,
}

impl SenseLodRouter {
    pub fn new(_boundaries: &[ScaleBoundary], sigma1: f32, sigma2: f32) -> Self {
        Self { sigma1, sigma2 }
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
        Some(Self {
            sigma1: boundaries[0].sigma,
            sigma2: boundaries[1].sigma,
        })
    }

    pub fn assign_lods(&self, distances: &[f32]) -> Vec<SenseLodLevel> {
        distances.iter().map(|&d| self.route(d)).collect()
    }

    /// Zero-alloc variant: writes LOD levels into a pre-allocated output buffer.
    ///
    /// `out` must have the same length as `distances`. Panics if lengths differ.
    pub fn assign_lods_into(&self, distances: &[f32], out: &mut [SenseLodLevel]) {
        assert_eq!(
            distances.len(),
            out.len(),
            "assign_lods_into: length mismatch"
        );
        for (out_val, &d) in out.iter_mut().zip(distances.iter()) {
            *out_val = self.route(d);
        }
    }
}

/// Fast boolean mask for sense module activation, indexed by `SenseKind` discriminant.
#[derive(Clone, Copy, Debug)]
pub struct SenseLodMask {
    mask: [bool; 8],
    active_count: u8,
}

impl SenseLodMask {
    /// Pre-computed masks — one per LOD level, built once.
    const MASKS: [Self; 3] = [
        // Full: all 6 active
        Self {
            mask: [true, true, true, true, true, true, false, false],
            active_count: 6,
        },
        // Compressed: Common(0), Fighter(1), Spatial(3)
        Self {
            mask: [true, true, false, true, false, false, false, false],
            active_count: 3,
        },
        // Minimal: Spatial(3) only
        Self {
            mask: [false, false, false, true, false, false, false, false],
            active_count: 1,
        },
    ];

    #[inline]
    pub fn from_level(level: SenseLodLevel) -> Self {
        Self::MASKS[level as usize]
    }

    #[inline]
    pub fn is_active(&self, kind: SenseKind) -> bool {
        let idx = kind as usize;
        // SenseKind is #[repr(u8)] with discriminants 0..7, mask covers all 8.
        unsafe { *self.mask.get_unchecked(idx) }
    }

    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_count as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slod::ScaleBoundary;

    fn router() -> SenseLodRouter {
        SenseLodRouter::new(&[boundary(10.0), boundary(50.0)], 10.0, 50.0)
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
