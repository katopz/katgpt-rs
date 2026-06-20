//! NpcBrain — composable sense modules with GM override.

use crate::sense::reconstruction::{
    ReconstructionConfig, ReconstructionResult, ReconstructionState,
};
use crate::types::{SenseKind, SenseModule};

#[cfg(feature = "sense_lod")]
use crate::sense::lod::SenseLodLevel;

/// Maximum number of per-sense overrides.
const MAX_OVERRIDES: usize = 8;

/// Number of SenseKind variants with valid discriminants 0..6.
const SENSE_KIND_COUNT: usize = 6;

/// Per-NPC sense override configuration. GM always wins.
///
/// # Layout
///
/// Fields ordered by alignment (u64 → f32 → u8) to eliminate padding — saves
/// 8 bytes per NPC vs the naïve declaration order. With thousands of NPCs per
/// shard this keeps the override struct at 136 bytes instead of 144.
#[derive(Clone, Debug)]
pub struct SenseOverride {
    /// Script ID if in scripted mode.
    pub script_id: Option<u64>,
    /// Pinned sense activations: (kind, value). If present, overrides autonomous.
    /// Fixed-size array avoids heap allocation — MAX_OVERRIDES slots.
    pub pinned: [(SenseKind, f32); MAX_OVERRIDES],
    /// O(1) pin lookup indexed by SenseKind discriminant. Rebuilt on pin/unpin.
    pin_lookup: [Option<f32>; SENSE_KIND_COUNT],
    /// Number of valid entries in `pinned`.
    pinned_count: u8,
    /// If true, all autonomous computation is disabled; only pinned values returned.
    pub autonomous_disabled: bool,
}

impl Default for SenseOverride {
    fn default() -> Self {
        Self {
            pinned: [(SenseKind::CommonSense, 0.0); MAX_OVERRIDES],
            pin_lookup: [None; SENSE_KIND_COUNT],
            script_id: None,
            pinned_count: 0,
            autonomous_disabled: false,
        }
    }
}

impl SenseOverride {
    fn rebuild_pin_lookup(&mut self) {
        self.pin_lookup = [None; SENSE_KIND_COUNT];
        for i in 0..self.pinned_count as usize {
            let (kind, value) = self.pinned[i];
            let idx = kind as usize;
            if idx < SENSE_KIND_COUNT {
                self.pin_lookup[idx] = Some(value);
            }
        }
    }

    /// Public accessor for pinned value by SenseKind.
    /// Used by NpcBrainInput::from_brain() to extract override state.
    #[inline]
    pub fn pinned_value_brain(&self, kind: SenseKind) -> Option<f32> {
        self.pinned_value(kind)
    }

    #[inline]
    fn pinned_value(&self, kind: SenseKind) -> Option<f32> {
        let idx = kind as usize;
        if idx < SENSE_KIND_COUNT {
            self.pin_lookup[idx]
        } else {
            for i in 0..self.pinned_count as usize {
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
    /// GM override mask.
    pub overrides: SenseOverride,
    /// Current HLA state (8-dim).
    pub hla_state: [f32; 8],
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
            overrides: SenseOverride::default(),
            hla_state: [0.0; 8],
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
    /// Direct indexing avoids iterator overhead; LLVM unrolls for fixed sizes.
    pub fn update_hla(&mut self, delta: &[f32]) {
        let len = delta.len().min(8);
        for (i, val) in delta.iter().copied().enumerate().take(len) {
            self.hla_state[i] += val;
        }
    }

    /// Update HLA state with fixed-size delta — delegates to SIMD add.
    #[inline]
    pub fn update_hla_fixed(&mut self, delta: &[f32; 8]) {
        crate::simd::simd_add_inplace(&mut self.hla_state, delta);
    }

    /// GM pins a sense activation. Updates O(1) lookup directly — no full rebuild.
    pub fn pin_sense(&mut self, kind: SenseKind, value: f32) {
        let idx = kind as usize;
        if idx < SENSE_KIND_COUNT {
            // Fast path: O(1) check via pin_lookup
            if self.overrides.pin_lookup[idx].is_some() {
                // Already pinned — update array entry and lookup in-place
                for i in 0..self.overrides.pinned_count as usize {
                    if self.overrides.pinned[i].0 == kind {
                        self.overrides.pinned[i].1 = value;
                        break;
                    }
                }
                self.overrides.pin_lookup[idx] = Some(value);
            } else if (self.overrides.pinned_count as usize) < MAX_OVERRIDES {
                // New pin — append to array and update single lookup slot
                self.overrides.pinned[self.overrides.pinned_count as usize] = (kind, value);
                self.overrides.pinned_count += 1;
                self.overrides.pin_lookup[idx] = Some(value);
            }
        } else {
            // Unknown discriminant — fall back to linear scan
            let mut found = false;
            for i in 0..self.overrides.pinned_count as usize {
                if self.overrides.pinned[i].0 == kind {
                    self.overrides.pinned[i].1 = value;
                    found = true;
                    break;
                }
            }
            if !found && (self.overrides.pinned_count as usize) < MAX_OVERRIDES {
                self.overrides.pinned[self.overrides.pinned_count as usize] = (kind, value);
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

    // -- Reconstruction integration (Phase 3) --

    /// Multi-step reconstructive projection — active retrieval with HLA evolution.
    ///
    /// Unlike `project_all()` which does single-shot passive retrieval,
    /// this uses iterative reconstruction: the HLA state evolves based on
    /// accumulated evidence across multiple traversal steps.
    ///
    /// Returns `ReconstructionResult` with passive baseline, active reconstructed
    /// activations, and the HLA delta from evolution.
    pub fn project_reconstruct(&self) -> ReconstructionResult {
        self.project_reconstruct_with_config(ReconstructionConfig::default())
    }

    /// Multi-step reconstructive projection with custom config.
    pub fn project_reconstruct_with_config(
        &self,
        config: ReconstructionConfig,
    ) -> ReconstructionResult {
        // Passive: single-shot projection for baseline comparison
        let passive = {
            let mut acts = [0.0f32; 6];
            for module in &self.modules {
                let idx = module.kind as usize;
                if idx < 6 {
                    acts[idx] = module.project(&self.hla_state);
                }
            }
            acts
        };

        // Active: multi-step reconstruction
        let mut state = ReconstructionState::with_config(self.hla_state, config);
        let active = state.reconstruct(self);

        // Compute HLA delta
        let mut hla_delta = [0.0f32; 8];
        for (i, (h, s)) in state.hla().iter().zip(self.hla_state.iter()).enumerate() {
            hla_delta[i] = h - s;
        }

        ReconstructionResult {
            passive,
            active,
            steps: state.step() as u32,
            evidence: state.evidence().clone(),
            hla_delta,
        }
    }

    /// Zero-alloc multi-step reconstruction into pre-allocated state.
    /// Reuses the `ReconstructionState` across calls to avoid allocation.
    pub fn reconstruct_into(&self, state: &mut ReconstructionState) {
        state.reconstruct(self);
    }

    /// Compare single-shot vs reconstructive projection for diagnostics.
    pub fn compare_projection_modes(&self) -> ReconstructionResult {
        self.project_reconstruct()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
    use crate::types::SenseKind;

    /// Layout regression guard: SenseOverride must stay at 136 bytes (no padding).
    /// Field reorder must keep this pinned — bump only if MAX_OVERRIDES or
    /// SENSE_KIND_COUNT changes, then re-audit the layout.
    #[test]
    fn sense_override_layout_is_compact() {
        assert_eq!(
            std::mem::size_of::<SenseOverride>(),
            136,
            "SenseOverride must stay at 136 bytes; padding regressed"
        );
    }

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

    #[test]
    fn test_project_reconstruct_returns_valid_result() {
        let brain = NpcBrain::compose(vec![make_fighter_module(), make_spatial_module()]);
        let result = brain.project_reconstruct();
        // Passive activations should be valid f32 values
        for &a in &result.passive {
            assert!(
                a.is_finite(),
                "Passive activation should be finite, got {a}"
            );
        }
        // Active activations should be valid f32 values
        for &a in &result.active {
            assert!(a.is_finite(), "Active activation should be finite, got {a}");
        }
        // Steps used should be <= default max_steps (3)
        assert!(
            result.steps <= 3,
            "Steps should be <= 3, got {}",
            result.steps
        );
        // HLA delta should be valid
        for &d in &result.hla_delta {
            assert!(d.is_finite(), "HLA delta should be finite, got {d}");
        }
    }

    #[test]
    fn test_project_reconstruct_with_config() {
        let config = ReconstructionConfig {
            max_steps: 1,
            ..Default::default()
        };
        let brain = NpcBrain::compose(vec![make_fighter_module()]);
        let result = brain.project_reconstruct_with_config(config);
        // With max_steps=1, should only do 1 step
        assert!(
            result.steps <= 1,
            "Steps should be <= 1, got {}",
            result.steps
        );
    }

    #[test]
    fn test_reconstruct_into_reuses_state() {
        let brain = NpcBrain::compose(vec![make_fighter_module()]);
        let mut state = ReconstructionState::new(brain.hla_state);
        brain.reconstruct_into(&mut state);
        assert!(state.step() > 0, "Should have taken at least 1 step");
        // Second call reuses the same state
        brain.reconstruct_into(&mut state);
        assert!(state.step() > 1, "Should have progressed further");
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
