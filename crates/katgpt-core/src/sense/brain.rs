//! NpcBrain — composable sense modules with GM override.

use crate::types::{SenseKind, SenseModule};

#[cfg(feature = "sense_lod")]
use crate::sense::lod::SenseLodLevel;

/// Maximum number of per-sense overrides.
const MAX_OVERRIDES: usize = 8;

/// Number of SenseKind variants with valid discriminants 0..6.
const SENSE_KIND_COUNT: usize = 6;

/// Per-NPC sense override configuration. GM always wins.
#[derive(Clone, Debug)]
pub struct SenseOverride {
    /// Pinned sense activations: (kind, value). If present, overrides autonomous.
    /// Fixed-size array avoids heap allocation — MAX_OVERRIDES slots.
    pub pinned: [(SenseKind, f32); MAX_OVERRIDES],
    /// Number of valid entries in `pinned`.
    pinned_count: usize,
    /// O(1) pin lookup indexed by SenseKind discriminant. Rebuilt on pin/unpin.
    pin_lookup: [Option<f32>; SENSE_KIND_COUNT],
    /// If true, all autonomous computation is disabled; only pinned values returned.
    pub autonomous_disabled: bool,
    /// Script ID if in scripted mode.
    pub script_id: Option<u64>,
}

impl Default for SenseOverride {
    fn default() -> Self {
        Self {
            pinned: [(SenseKind::CommonSense, 0.0); MAX_OVERRIDES],
            pinned_count: 0,
            pin_lookup: [None; SENSE_KIND_COUNT],
            autonomous_disabled: false,
            script_id: None,
        }
    }
}

impl SenseOverride {
    fn rebuild_pin_lookup(&mut self) {
        self.pin_lookup = [None; SENSE_KIND_COUNT];
        for i in 0..self.pinned_count {
            let (kind, value) = self.pinned[i];
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
            for i in 0..self.pinned_count {
                if self.pinned[i].0 == kind {
                    return Some(self.pinned[i].1);
                }
            }
            None
        }
    }

    /// Returns true if no pins are active.
    #[inline]
    fn is_empty(&self) -> bool {
        self.pinned_count == 0
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
        }
    }

    /// Set active LOD level and rebuild cached mask.
    #[cfg(feature = "sense_lod")]
    pub fn set_lod(&mut self, level: SenseLodLevel) {
        self.active_lod = level;
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
    ///
    /// LOD depth:
    /// - Full: iterate all modules, project all (6 dot-products)
    /// - Compressed: resize to zero, project only [Common, Fighter, Spatial] via O(1) index (3 dot-products)
    /// - Minimal: resize to zero, project only Spatial via O(1) index (1 dot-product)
    ///
    /// No linear scans, no per-module branch for LOD < Full.
    pub fn project_all_into(&self, result: &mut Vec<f32>) {
        let len = self.modules.len();
        result.clear();

        #[cfg(feature = "sense_lod")]
        match self.active_lod {
            SenseLodLevel::Full => {
                // Full writes every element — zero-fill overwritten but avoids uninit.
                result.resize(len, 0.0);
                self.project_full(result);
            }
            SenseLodLevel::Compressed | SenseLodLevel::Minimal => {
                // Compressed/Minimal only write specific indices — zero-fill first.
                result.resize(len, 0.0);
                match self.active_lod {
                    SenseLodLevel::Compressed => self.project_compressed(result),
                    SenseLodLevel::Minimal => self.project_minimal(result),
                    _ => unreachable!(),
                }
            }
        }
        #[cfg(not(feature = "sense_lod"))]
        {
            // Full writes every element — zero-fill overwritten but avoids uninit.
            result.resize(len, 0.0);
            self.project_full(result);
        }
    }

    /// Full projection: iterate all modules, project each.
    /// Fast-path skips per-module override checks when no overrides are active.
    #[inline]
    fn project_full(&self, result: &mut [f32]) {
        // Fast path: no overrides active
        if self.overrides.is_empty() && !self.overrides.autonomous_disabled {
            for (i, m) in self.modules.iter().enumerate() {
                result[i] = m.project(&self.hla_state);
            }
            return;
        }
        // Slow path: check overrides per module
        for (i, m) in self.modules.iter().enumerate() {
            result[i] = match self.overrides.pinned_value(m.kind) {
                Some(v) => v,
                None if self.overrides.autonomous_disabled => 0.0,
                None => m.project(&self.hla_state),
            };
        }
    }

    /// Compressed projection: only Common, Fighter, Spatial via O(1) index lookup.
    #[cfg(feature = "sense_lod")]
    #[inline]
    fn project_compressed(&self, result: &mut [f32]) {
        use SenseKind::*;
        for &kind in &[CommonSense, FighterSense, SpatialSense] {
            let kind_idx = kind as usize;
            if let Some(mod_idx) = self.module_index[kind_idx] {
                result[mod_idx] = match self.overrides.pinned_value(kind) {
                    Some(v) => v,
                    None if self.overrides.autonomous_disabled => 0.0,
                    None => self.modules[mod_idx].project(&self.hla_state),
                };
            }
        }
    }

    /// Minimal projection: only Spatial via O(1) index lookup.
    #[cfg(feature = "sense_lod")]
    #[inline]
    fn project_minimal(&self, result: &mut [f32]) {
        let spatial_idx = SenseKind::SpatialSense as usize;
        if let Some(mod_idx) = self.module_index[spatial_idx] {
            result[mod_idx] = match self.overrides.pinned_value(SenseKind::SpatialSense) {
                Some(v) => v,
                None if self.overrides.autonomous_disabled => 0.0,
                None => self.modules[mod_idx].project(&self.hla_state),
            };
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

    /// Update HLA state with delta (dynamic slice, bounds-checked).
    /// Direct indexing avoids zip().take() iterator overhead; LLVM unrolls better.
    pub fn update_hla(&mut self, delta: &[f32]) {
        let len = delta.len().min(8);
        for i in 0..len {
            self.hla_state[i] += delta[i];
        }
    }

    /// Update HLA state with fixed-size delta — fully unrolled for SIMD add.
    #[inline]
    pub fn update_hla_fixed(&mut self, delta: &[f32; 8]) {
        self.hla_state[0] += delta[0];
        self.hla_state[1] += delta[1];
        self.hla_state[2] += delta[2];
        self.hla_state[3] += delta[3];
        self.hla_state[4] += delta[4];
        self.hla_state[5] += delta[5];
        self.hla_state[6] += delta[6];
        self.hla_state[7] += delta[7];
    }

    /// GM pins a sense activation. Updates O(1) lookup directly — no full rebuild.
    pub fn pin_sense(&mut self, kind: SenseKind, value: f32) {
        let idx = kind as usize;
        if idx < SENSE_KIND_COUNT {
            // Fast path: O(1) check via pin_lookup
            if self.overrides.pin_lookup[idx].is_some() {
                // Already pinned — update array entry and lookup in-place
                for i in 0..self.overrides.pinned_count {
                    if self.overrides.pinned[i].0 == kind {
                        self.overrides.pinned[i].1 = value;
                        break;
                    }
                }
                self.overrides.pin_lookup[idx] = Some(value);
            } else if self.overrides.pinned_count < MAX_OVERRIDES {
                // New pin — append to array and update single lookup slot
                self.overrides.pinned[self.overrides.pinned_count] = (kind, value);
                self.overrides.pinned_count += 1;
                self.overrides.pin_lookup[idx] = Some(value);
            }
        } else {
            // Unknown discriminant — fall back to linear scan
            let mut found = false;
            for i in 0..self.overrides.pinned_count {
                if self.overrides.pinned[i].0 == kind {
                    self.overrides.pinned[i].1 = value;
                    found = true;
                    break;
                }
            }
            if !found && self.overrides.pinned_count < MAX_OVERRIDES {
                self.overrides.pinned[self.overrides.pinned_count] = (kind, value);
                self.overrides.pinned_count += 1;
            }
            self.overrides.rebuild_pin_lookup();
        }
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
