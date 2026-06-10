//! NpcBrain — composable sense modules with GM override.

use crate::types::{SenseKind, SenseModule};

#[cfg(feature = "sense_lod")]
use crate::sense::lod::SenseLodLevel;

/// Maximum number of per-sense overrides.
const MAX_OVERRIDES: usize = 8;

/// Number of SenseKind variants with valid discriminants 0..6.
const SENSE_KIND_COUNT: usize = 6;

/// Per-NPC sense override configuration. GM always wins.
#[derive(Clone, Debug, Default)]
pub struct SenseOverride {
    /// Pinned sense activations: (kind, value). If present, overrides autonomous.
    pub pinned: Vec<(SenseKind, f32)>,
    /// O(1) pin lookup indexed by SenseKind discriminant. Rebuilt on pin/unpin.
    pin_lookup: [Option<f32>; SENSE_KIND_COUNT],
    /// If true, all autonomous computation is disabled; only pinned values returned.
    pub autonomous_disabled: bool,
    /// Script ID if in scripted mode.
    pub script_id: Option<u64>,
}

impl SenseOverride {
    fn rebuild_pin_lookup(&mut self) {
        self.pin_lookup = [None; SENSE_KIND_COUNT];
        for &(kind, value) in &self.pinned {
            let idx = kind as usize;
            if idx < SENSE_KIND_COUNT {
                self.pin_lookup[idx] = Some(value);
            }
        }
    }

    #[inline]
    fn pinned_value(&self, kind: SenseKind) -> Option<f32> {
        let idx = kind as usize;
        if idx < SENSE_KIND_COUNT {
            self.pin_lookup[idx]
        } else {
            self.pinned
                .iter()
                .find(|(k, _)| *k == kind)
                .map(|(_, v)| *v)
        }
    }
}

/// NPC Brain — composes sense modules and projects HLA state.
#[derive(Clone, Debug)]
pub struct NpcBrain {
    /// Loaded sense modules.
    pub modules: Vec<SenseModule>,
    /// O(1) module lookup indexed by SenseKind discriminant. Rebuilt on compose.
    module_index: [Option<usize>; SENSE_KIND_COUNT],
    /// Current HLA state (8-dim).
    pub hla_state: [f32; 8],
    /// GM override mask.
    pub overrides: SenseOverride,
    /// Active LOD level — determines which modules to project.
    /// Default: Full (all modules). Only used with `sense_lod` feature.
    #[cfg(feature = "sense_lod")]
    pub active_lod: SenseLodLevel,
    /// Cached LOD mask — rebuilt when active_lod changes.
    #[cfg(feature = "sense_lod")]
    lod_mask: crate::sense::lod::SenseLodMask,
}

impl NpcBrain {
    /// Create a new brain with given modules.
    pub fn compose(modules: Vec<SenseModule>) -> Self {
        let mut module_index = [None; SENSE_KIND_COUNT];
        for (i, m) in modules.iter().enumerate() {
            let idx = m.kind as usize;
            if idx < SENSE_KIND_COUNT {
                module_index[idx] = Some(i);
            }
        }
        Self {
            modules,
            module_index,
            hla_state: [0.0; 8],
            overrides: SenseOverride::default(),
            #[cfg(feature = "sense_lod")]
            active_lod: SenseLodLevel::Full,
            #[cfg(feature = "sense_lod")]
            lod_mask: crate::sense::lod::SenseLodMask::from_level(SenseLodLevel::Full),
        }
    }

    /// Set active LOD level and rebuild cached mask.
    #[cfg(feature = "sense_lod")]
    pub fn set_lod(&mut self, level: SenseLodLevel) {
        self.active_lod = level;
        self.lod_mask = crate::sense::lod::SenseLodMask::from_level(level);
    }

    /// Project HLA state onto all loaded modules. GM override wins.
    /// Allocating version — see `project_all_into` for zero-alloc alternative.
    pub fn project_all(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.modules.len());
        self.project_all_into(&mut result);
        result
    }

    /// Zero-alloc projection into pre-allocated buffer.
    /// Clears `result` and fills with projected values for each module.
    /// Uses O(1) pin lookup and cached LOD mask — no linear scans, no per-call allocation.
    /// With `sense_lod` feature: skips modules not in active LOD level, pushes 0.0 for skipped.
    pub fn project_all_into(&self, result: &mut Vec<f32>) {
        result.clear();
        #[cfg(feature = "sense_lod")]
        let mask = self.lod_mask;
        for m in &self.modules {
            #[cfg(feature = "sense_lod")]
            if !mask.is_active(m.kind) {
                result.push(0.0);
                continue;
            }
            let val = match self.overrides.pinned_value(m.kind) {
                Some(v) => v,
                None if self.overrides.autonomous_disabled => 0.0,
                None => m.project(&self.hla_state),
            };
            result.push(val);
        }
    }

    /// Project a single sense kind, respecting GM override.
    /// Uses O(1) lookups — no linear scans.
    pub fn project_kind(&self, kind: SenseKind) -> Option<f32> {
        // Check pin first (O(1) lookup)
        if let Some(v) = self.overrides.pinned_value(kind) {
            return Some(v);
        }
        // Scripted mode with no pin → None
        if self.overrides.autonomous_disabled {
            return None;
        }
        // O(1) module lookup by index
        let idx = kind as usize;
        if idx < SENSE_KIND_COUNT {
            self.module_index[idx].map(|i| self.modules[i].project(&self.hla_state))
        } else {
            self.modules
                .iter()
                .find(|m| m.kind == kind)
                .map(|m| m.project(&self.hla_state))
        }
    }

    /// Update HLA state with delta.
    pub fn update_hla(&mut self, delta: &[f32]) {
        for (i, &d) in delta.iter().enumerate() {
            if i < self.hla_state.len() {
                self.hla_state[i] += d;
            }
        }
    }

    /// GM pins a sense activation. Rebuilds O(1) lookup table.
    pub fn pin_sense(&mut self, kind: SenseKind, value: f32) {
        if let Some(entry) = self.overrides.pinned.iter_mut().find(|(k, _)| *k == kind) {
            entry.1 = value;
        } else if self.overrides.pinned.len() < MAX_OVERRIDES {
            self.overrides.pinned.push((kind, value));
        }
        self.overrides.rebuild_pin_lookup();
    }

    /// Enter scripted mode — disable all autonomous behavior.
    pub fn disable_autonomous(&mut self, script_id: u64) {
        self.overrides.autonomous_disabled = true;
        self.overrides.script_id = Some(script_id);
    }

    /// Exit scripted mode — restore autonomous behavior.
    pub fn enable_autonomous(&mut self) {
        self.overrides.autonomous_disabled = false;
        self.overrides.script_id = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
    use crate::types::SenseKind;

    fn make_fighter_module() -> SenseModule {
        let builder = SenseOctreeBuilder::new(3);
        let emb = KgEmbedding {
            entity_hash: 1,
            relation_hash: 1,
            embedding: [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
            confidence: 1.0,
        };
        builder.build(SenseKind::FighterSense, &[emb])
    }

    fn make_spatial_module() -> SenseModule {
        let builder = SenseOctreeBuilder::new(3);
        let emb = KgEmbedding {
            entity_hash: 2,
            relation_hash: 2,
            embedding: [0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
            confidence: 1.0,
        };
        builder.build(SenseKind::SpatialSense, &[emb])
    }

    #[test]
    fn test_compose_and_project() {
        let brain = NpcBrain::compose(vec![make_fighter_module(), make_spatial_module()]);
        let results = brain.project_all();
        assert_eq!(results.len(), 2);
        // All results should be valid sigmoid outputs
        for r in &results {
            assert!(*r > 0.0 && *r < 1.0);
        }
    }

    #[test]
    fn test_project_all_into_matches_allocating() {
        let brain = NpcBrain::compose(vec![make_fighter_module(), make_spatial_module()]);
        let expected = brain.project_all();
        let mut buf = Vec::new();
        brain.project_all_into(&mut buf);
        assert_eq!(expected, buf);
    }

    #[test]
    fn test_pin_overrides_autonomous() {
        let mut brain = NpcBrain::compose(vec![make_fighter_module()]);
        brain.hla_state = [0.5; 8];

        let auto_val = brain.project_kind(SenseKind::FighterSense).unwrap();
        brain.pin_sense(SenseKind::FighterSense, 0.9);
        let pinned_val = brain.project_kind(SenseKind::FighterSense).unwrap();

        assert_eq!(pinned_val, 0.9);
        assert_ne!(pinned_val, auto_val);
    }

    #[test]
    fn test_disable_autonomous() {
        let mut brain = NpcBrain::compose(vec![make_fighter_module()]);
        brain.pin_sense(SenseKind::FighterSense, 0.9);
        brain.disable_autonomous(42);

        // Should return pinned value
        assert_eq!(brain.project_kind(SenseKind::FighterSense).unwrap(), 0.9);
        // Unpinned sense in scripted mode returns None
        assert!(brain.project_kind(SenseKind::SpatialSense).is_none());

        brain.enable_autonomous();
        assert!(!brain.overrides.autonomous_disabled);
    }
}

#[cfg(test)]
#[cfg(feature = "sense_lod")]
mod lod_tests {
    use super::*;
    use crate::sense::lod::SenseLodLevel;
    use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};

    fn make_brain_with_modules() -> NpcBrain {
        let builder = SenseOctreeBuilder::new(3);
        let kinds = [
            SenseKind::CommonSense,
            SenseKind::FighterSense,
            SenseKind::GameTheorySense,
            SenseKind::SpatialSense,
            SenseKind::SocialSense,
            SenseKind::SkillSense,
        ];
        let modules: Vec<SenseModule> = kinds
            .iter()
            .map(|&kind| {
                let emb = KgEmbedding {
                    entity_hash: kind as u64,
                    relation_hash: kind as u64,
                    embedding: [0.5; 8],
                    sign: true,
                    confidence: 1.0,
                };
                builder.build(kind, &[emb])
            })
            .collect();
        let mut brain = NpcBrain::compose(modules);
        brain.hla_state = [0.5; 8];
        brain
    }

    #[test]
    fn test_lod_full_all_modules() {
        let brain = make_brain_with_modules();
        let mut result = Vec::new();
        brain.project_all_into(&mut result);
        assert_eq!(result.len(), 6);
        // All should be non-zero
        assert!(result.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn test_lod_minimal_only_spatial() {
        let mut brain = make_brain_with_modules();
        brain.set_lod(SenseLodLevel::Minimal);
        let mut result = Vec::new();
        brain.project_all_into(&mut result);
        assert_eq!(result.len(), 6);
        // Only SpatialSense (index 3) should be non-zero
        for (i, v) in result.iter().enumerate() {
            if i == 3 {
                assert!(*v > 0.0, "SpatialSense should be non-zero");
            } else {
                assert_eq!(*v, 0.0, "Module {} should be skipped (0.0)", i);
            }
        }
    }

    #[test]
    fn test_lod_compressed_three_modules() {
        let mut brain = make_brain_with_modules();
        brain.set_lod(SenseLodLevel::Compressed);
        let mut result = Vec::new();
        brain.project_all_into(&mut result);
        assert_eq!(result.len(), 6);
        // Common (0), Fighter (1), Spatial (3) should be non-zero
        let active = [0, 1, 3];
        for (i, v) in result.iter().enumerate() {
            if active.contains(&i) {
                assert!(*v > 0.0, "Module {} should be active", i);
            } else {
                assert_eq!(*v, 0.0, "Module {} should be skipped", i);
            }
        }
    }

    #[test]
    fn test_lod_default_is_full() {
        let brain = make_brain_with_modules();
        assert_eq!(brain.active_lod, SenseLodLevel::Full);
    }
}
