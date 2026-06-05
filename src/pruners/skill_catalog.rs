//! Progressive Disclosure Catalog — lightweight skill registry for lazy loading.
//!
//! The catalog holds lightweight `SkillDescriptor` entries (always in memory).
//! Full pruners are loaded on-demand when the bandit selects an arm.
//! This reduces memory pressure when many skills are registered but few are active.
//!
//! # MUSE Lifecycle: register + load
//!
//! After a pruner passes the test gate, its descriptor is registered here.
//! The bandit selects from descriptors (cheap), then `HotSwapPruner` loads
//! the full pruner for the selected arm.
//!
//! # Storage
//!
//! Uses `Vec` with linear scan by default (fine for <100 arms).
//! When `papaya` feature is enabled, uses lock-free `HashMap` for O(1) lookup.

use super::skill_test::TestStatus;

// ── SkillDescriptor ──────────────────────────────────────────────

/// Lightweight skill descriptor — always in memory.
///
/// Uses `u64` ID (blake3 hash truncated) instead of UUID to avoid extra dependency.
#[derive(Clone, Debug)]
pub struct SkillDescriptor {
    /// Unique identifier (blake3 hash of name, truncated to u64).
    pub id: u64,
    /// Short name for catalog lookup and debugging.
    pub name: String,
    /// Brief description of what this skill does.
    pub description: String,
    /// Maps to bandit arm index.
    pub arm_index: usize,
    /// Current validation status in the MUSE lifecycle.
    pub test_status: TestStatus,
}

impl SkillDescriptor {
    /// Create a new descriptor with auto-generated ID from name.
    pub fn new(name: &str, description: impl Into<String>, arm_index: usize) -> Self {
        let hash = blake3::hash(name.as_bytes());
        let id = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        Self {
            id,
            name: name.into(),
            description: description.into(),
            arm_index,
            test_status: TestStatus::Untested,
        }
    }
}

// ── SkillCatalog (std backend) ───────────────────────────────────

/// In-memory skill catalog.
///
/// Stores lightweight descriptors indexed by arm index.
/// Full pruner loaded on-demand by the bandit selection mechanism.
pub struct SkillCatalog {
    #[cfg(not(feature = "papaya"))]
    descriptors: Vec<SkillDescriptor>,
    #[cfg(feature = "papaya")]
    descriptors: papaya::HashMap<usize, SkillDescriptor>,
}

impl SkillCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            #[cfg(not(feature = "papaya"))]
            descriptors: Vec::new(),
            #[cfg(feature = "papaya")]
            descriptors: papaya::HashMap::new(),
        }
    }

    /// Register a new skill descriptor.
    ///
    /// If an arm with the same index already exists, it is replaced.
    pub fn register(&mut self, descriptor: SkillDescriptor) {
        #[cfg(not(feature = "papaya"))]
        {
            if let Some(existing) = self
                .descriptors
                .iter_mut()
                .find(|d| d.arm_index == descriptor.arm_index)
            {
                *existing = descriptor;
            } else {
                self.descriptors.push(descriptor);
            }
        }
        #[cfg(feature = "papaya")]
        {
            self.descriptors
                .pin()
                .insert(descriptor.arm_index, descriptor);
        }
    }

    /// Get a descriptor by arm index.
    pub fn get(&self, arm_index: usize) -> Option<SkillDescriptor> {
        #[cfg(not(feature = "papaya"))]
        {
            self.descriptors
                .iter()
                .find(|d| d.arm_index == arm_index)
                .cloned()
        }
        #[cfg(feature = "papaya")]
        {
            self.descriptors.pin().get(&arm_index).cloned()
        }
    }

    /// Update the test status of a skill by arm index.
    ///
    /// Returns `true` if the arm was found and updated.
    pub fn update_status(&mut self, arm_index: usize, status: TestStatus) -> bool {
        #[cfg(not(feature = "papaya"))]
        {
            if let Some(d) = self
                .descriptors
                .iter_mut()
                .find(|d| d.arm_index == arm_index)
            {
                d.test_status = status;
                true
            } else {
                false
            }
        }
        #[cfg(feature = "papaya")]
        {
            let mut map = self.descriptors.pin();
            if let Some(d) = map.get_mut(&arm_index) {
                d.test_status = status;
                true
            } else {
                false
            }
        }
    }

    /// Number of skills with `Active` status.
    pub fn active_count(&self) -> usize {
        #[cfg(not(feature = "papaya"))]
        {
            self.descriptors
                .iter()
                .filter(|d| d.test_status == TestStatus::Active)
                .count()
        }
        #[cfg(feature = "papaya")]
        {
            self.descriptors
                .pin()
                .values()
                .filter(|d| d.test_status == TestStatus::Active)
                .count()
        }
    }

    /// Total number of registered skills.
    pub fn len(&self) -> usize {
        #[cfg(not(feature = "papaya"))]
        {
            self.descriptors.len()
        }
        #[cfg(feature = "papaya")]
        {
            self.descriptors.pin().len()
        }
    }

    /// True if no skills registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all descriptors.
    #[cfg(not(feature = "papaya"))]
    pub fn iter(&self) -> impl Iterator<Item = &SkillDescriptor> {
        self.descriptors.iter()
    }

    /// Collect descriptors into a Vec for iteration (papaya backend).
    #[cfg(feature = "papaya")]
    pub fn iter(&self) -> impl Iterator<Item = SkillDescriptor> {
        self.descriptors
            .pin()
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl Default for SkillCatalog {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_descriptor(name: &str, arm: usize) -> SkillDescriptor {
        SkillDescriptor::new(name, format!("{name} skill"), arm)
    }

    #[test]
    fn test_register_and_get() {
        let mut catalog = SkillCatalog::new();
        let d = make_descriptor("ucb1_pruner", 0);
        catalog.register(d);
        assert_eq!(catalog.len(), 1);
        let got = catalog.get(0).unwrap();
        assert_eq!(got.name, "ucb1_pruner");
        assert_eq!(got.arm_index, 0);
    }

    #[test]
    fn test_get_missing() {
        let catalog = SkillCatalog::new();
        assert!(catalog.get(99).is_none());
    }

    #[test]
    fn test_register_replaces_existing_arm() {
        let mut catalog = SkillCatalog::new();
        catalog.register(make_descriptor("old", 0));
        catalog.register(make_descriptor("new", 0));
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog.get(0).unwrap().name, "new");
    }

    #[test]
    fn test_register_multiple_arms() {
        let mut catalog = SkillCatalog::new();
        catalog.register(make_descriptor("a", 0));
        catalog.register(make_descriptor("b", 1));
        catalog.register(make_descriptor("c", 2));
        assert_eq!(catalog.len(), 3);
    }

    #[test]
    fn test_update_status() {
        let mut catalog = SkillCatalog::new();
        catalog.register(make_descriptor("pruner", 5));
        assert_eq!(catalog.get(5).unwrap().test_status, TestStatus::Untested);

        assert!(catalog.update_status(5, TestStatus::Validated));
        assert_eq!(catalog.get(5).unwrap().test_status, TestStatus::Validated);

        assert!(catalog.update_status(5, TestStatus::Active));
        assert_eq!(catalog.active_count(), 1);
    }

    #[test]
    fn test_update_status_missing() {
        let mut catalog = SkillCatalog::new();
        assert!(!catalog.update_status(99, TestStatus::Active));
    }

    #[test]
    fn test_active_count() {
        let mut catalog = SkillCatalog::new();
        catalog.register(make_descriptor("a", 0));
        catalog.register(make_descriptor("b", 1));
        catalog.register(make_descriptor("c", 2));
        assert_eq!(catalog.active_count(), 0);

        catalog.update_status(0, TestStatus::Active);
        catalog.update_status(2, TestStatus::Active);
        assert_eq!(catalog.active_count(), 2);
    }

    #[test]
    fn test_is_empty() {
        let catalog = SkillCatalog::new();
        assert!(catalog.is_empty());
    }

    #[test]
    fn test_descriptor_id_deterministic() {
        let d1 = SkillDescriptor::new("test", "desc", 0);
        let d2 = SkillDescriptor::new("test", "desc", 0);
        let d3 = SkillDescriptor::new("other", "desc", 0);
        assert_eq!(d1.id, d2.id);
        assert_ne!(d1.id, d3.id);
    }

    #[test]
    fn test_default() {
        let catalog = SkillCatalog::default();
        assert!(catalog.is_empty());
    }
}

// TL;DR: SkillCatalog — lightweight skill registry with Vec (std) or papaya HashMap backends. Progressive disclosure: descriptors always in memory, full pruners loaded on demand. O(n) scan for <100 arms, O(1) with papaya.
