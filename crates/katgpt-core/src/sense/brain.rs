//! NpcBrain — composable sense modules with GM override.

use crate::types::{SenseKind, SenseModule};

/// Maximum number of per-sense overrides.
const MAX_OVERRIDES: usize = 8;

/// Per-NPC sense override configuration. GM always wins.
#[derive(Clone, Debug, Default)]
pub struct SenseOverride {
    /// Pinned sense activations: (kind, value). If present, overrides autonomous.
    pub pinned: Vec<(SenseKind, f32)>,
    /// If true, all autonomous computation is disabled; only pinned values returned.
    pub autonomous_disabled: bool,
    /// Script ID if in scripted mode.
    pub script_id: Option<u64>,
}

/// NPC Brain — composes sense modules and projects HLA state.
#[derive(Clone, Debug)]
pub struct NpcBrain {
    /// Loaded sense modules.
    pub modules: Vec<SenseModule>,
    /// Current HLA state (8-dim).
    pub hla_state: [f32; 8],
    /// GM override mask.
    pub overrides: SenseOverride,
}

impl NpcBrain {
    /// Create a new brain with given modules.
    pub fn compose(modules: Vec<SenseModule>) -> Self {
        Self {
            modules,
            hla_state: [0.0; 8],
            overrides: SenseOverride::default(),
        }
    }

    /// Project HLA state onto all loaded modules. GM override wins.
    pub fn project_all(&self) -> Vec<f32> {
        self.modules
            .iter()
            .map(|m| {
                self.project_kind(m.kind)
                    .unwrap_or_else(|| m.project(&self.hla_state))
            })
            .collect()
    }

    /// Project a single sense kind, respecting GM override.
    pub fn project_kind(&self, kind: SenseKind) -> Option<f32> {
        // Check scripted mode first
        if self.overrides.autonomous_disabled {
            return self
                .overrides
                .pinned
                .iter()
                .find(|(k, _)| *k == kind)
                .map(|(_, v)| *v);
        }
        // Check per-sense pin
        if let Some((_, value)) = self.overrides.pinned.iter().find(|(k, _)| *k == kind) {
            return Some(*value);
        }
        // Autonomous projection
        self.modules
            .iter()
            .find(|m| m.kind == kind)
            .map(|m| m.project(&self.hla_state))
    }

    /// Update HLA state with delta.
    pub fn update_hla(&mut self, delta: &[f32]) {
        for (i, &d) in delta.iter().enumerate() {
            if i < self.hla_state.len() {
                self.hla_state[i] += d;
            }
        }
    }

    /// GM pins a sense activation.
    pub fn pin_sense(&mut self, kind: SenseKind, value: f32) {
        if let Some(entry) = self.overrides.pinned.iter_mut().find(|(k, _)| *k == kind) {
            entry.1 = value;
        } else if self.overrides.pinned.len() < MAX_OVERRIDES {
            self.overrides.pinned.push((kind, value));
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
