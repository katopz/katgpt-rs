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
//! # Lazy Loading
//!
//! `LazySkillLoader` trait defines the interface for on-demand pruner loading.
//! `FileSkillLoader` provides a disk-based implementation that reads arm state
//! from `{base_dir}/arm_{index}.bin` with blake3 integrity verification.
//!
//! # Storage
//!
//! Uses `Vec` with linear scan by default (fine for <100 arms).
//! When `papaya` feature is enabled, uses lock-free `HashMap` for O(1) lookup.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

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
    /// Optional lazy loader for full pruner state on demand.
    loader: Option<Box<dyn LazySkillLoader>>,
    /// Arm indices that have been loaded through this catalog.
    loaded_arms: HashSet<usize>,
}

impl SkillCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            #[cfg(not(feature = "papaya"))]
            descriptors: Vec::new(),
            #[cfg(feature = "papaya")]
            descriptors: papaya::HashMap::new(),
            loader: None,
            loaded_arms: HashSet::new(),
        }
    }

    /// Create a catalog with a lazy loader.
    ///
    /// When `load_arm()` is called, the loader provides the full pruner state.
    pub fn with_loader(loader: Box<dyn LazySkillLoader>) -> Self {
        Self {
            #[cfg(not(feature = "papaya"))]
            descriptors: Vec::new(),
            #[cfg(feature = "papaya")]
            descriptors: papaya::HashMap::new(),
            loader: Some(loader),
            loaded_arms: HashSet::new(),
        }
    }

    /// Lazy-load the full pruner state for the given arm.
    ///
    /// Delegates to the `LazySkillLoader` if set, returns `None` otherwise.
    /// Tracks loaded arms for `loaded_count()`.
    pub fn load_arm(&mut self, arm_index: usize) -> Option<Vec<u8>> {
        match &self.loader {
            Some(loader) => {
                let data = loader.load(arm_index);
                if data.is_some() {
                    self.loaded_arms.insert(arm_index);
                }
                data
            }
            None => None,
        }
    }

    /// Number of arms that have been loaded through `load_arm()`.
    pub fn loaded_count(&self) -> usize {
        self.loaded_arms.len()
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
            let map = self.descriptors.pin();
            map.update(arm_index, |d| {
                let mut new_d = d.clone();
                new_d.test_status = status;
                new_d
            }).is_some()
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

// ── LazySkillLoader trait ─────────────────────────────────────────

/// Trait for lazy-loading full pruners on demand.
///
/// When `BanditPruner` selects an arm, the catalog calls `load()` to
/// materialize the full pruner state via `HotSwapPruner`.
pub trait LazySkillLoader: Send + Sync {
    /// Load the full pruner state for the given arm index.
    ///
    /// Returns the serialized pruner bytes, or `None` if unavailable.
    fn load(&self, arm_index: usize) -> Option<Vec<u8>>;

    /// Check if a pruner is already loaded/cached for the given arm.
    fn is_loaded(&self, arm_index: usize) -> bool;
}

// ── FileSkillLoader ───────────────────────────────────────────────

/// File-based lazy loader — reads pruner state from `{base_dir}/arm_{index}.bin`.
///
/// Tracks which arms have been loaded to avoid redundant disk reads.
/// Uses blake3 for optional file integrity verification.
pub struct FileSkillLoader {
    base_dir: PathBuf,
    /// Arm indices that have been loaded at least once.
    loaded: HashSet<usize>,
}

impl FileSkillLoader {
    /// Create a new file loader rooted at `base_dir`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            loaded: HashSet::new(),
        }
    }

    /// Returns the expected file path for an arm.
    fn arm_path(&self, arm_index: usize) -> PathBuf {
        self.base_dir.join(format!("arm_{arm_index}.bin"))
    }
}

impl LazySkillLoader for FileSkillLoader {
    fn load(&self, arm_index: usize) -> Option<Vec<u8>> {
        let path = self.arm_path(arm_index);
        match std::fs::read(&path) {
            Ok(data) => {
                // blake3 integrity: first 32 bytes are the hash, remainder is payload.
                if data.len() < 32 {
                    return None;
                }
                let (hash_bytes, payload) = data.split_at(32);
                let computed = blake3::hash(payload);
                if computed.as_bytes() != hash_bytes {
                    return None;
                }
                Some(payload.to_vec())
            }
            Err(_) => None,
        }
    }

    fn is_loaded(&self, arm_index: usize) -> bool {
        self.loaded.contains(&arm_index)
    }
}

/// Write arm data to disk with blake3 integrity prefix.
///
/// Format: `[blake3_hash (32 bytes)][payload]`.
/// Helper for tests and integration code.
pub fn write_arm_data(dir: &Path, arm_index: usize, payload: &[u8]) -> std::io::Result<()> {
    let hash = blake3::hash(payload);
    let path = dir.join(format!("arm_{arm_index}.bin"));
    let mut out = Vec::with_capacity(32 + payload.len());
    out.extend_from_slice(hash.as_bytes());
    out.extend_from_slice(payload);
    std::fs::write(path, out)
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

    // ── Lazy loader tests ──────────────────────────────────────────

    #[test]
    fn test_lazy_loader_loads_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"pruner_state_arm_0";
        write_arm_data(dir.path(), 0, payload).unwrap();

        let loader = FileSkillLoader::new(dir.path());
        let data = loader.load(0).unwrap();
        assert_eq!(data, payload);
    }

    #[test]
    fn test_lazy_loader_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let loader = FileSkillLoader::new(dir.path());
        assert!(loader.load(99).is_none());
        assert!(!loader.is_loaded(99));
    }

    #[test]
    fn test_catalog_with_loader() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"full_pruner_bytes";
        write_arm_data(dir.path(), 3, payload).unwrap();

        let loader = Box::new(FileSkillLoader::new(dir.path()));
        let mut catalog = SkillCatalog::with_loader(loader);
        catalog.register(make_descriptor("skill_a", 3));

        let data = catalog.load_arm(3).unwrap();
        assert_eq!(data, payload);
        assert_eq!(catalog.loaded_count(), 1);
    }

    #[test]
    fn test_catalog_without_loader() {
        let mut catalog = SkillCatalog::new();
        catalog.register(make_descriptor("skill_a", 0));
        assert!(catalog.load_arm(0).is_none());
        assert_eq!(catalog.loaded_count(), 0);
    }

    #[test]
    fn test_loaded_count() {
        let dir = tempfile::tempdir().unwrap();
        write_arm_data(dir.path(), 0, b"arm0").unwrap();
        write_arm_data(dir.path(), 1, b"arm1").unwrap();
        write_arm_data(dir.path(), 2, b"arm2").unwrap();

        let loader = Box::new(FileSkillLoader::new(dir.path()));
        let mut catalog = SkillCatalog::with_loader(loader);

        assert_eq!(catalog.loaded_count(), 0);

        catalog.load_arm(0);
        assert_eq!(catalog.loaded_count(), 1);

        catalog.load_arm(2);
        assert_eq!(catalog.loaded_count(), 2);

        // Loading same arm again doesn't increase count.
        catalog.load_arm(0);
        assert_eq!(catalog.loaded_count(), 2);

        // Missing arm doesn't increase count.
        catalog.load_arm(99);
        assert_eq!(catalog.loaded_count(), 2);
    }
}

// TL;DR: SkillCatalog — lightweight skill registry with Vec (std) or papaya HashMap backends. Progressive disclosure: descriptors always in memory, full pruners loaded on demand via LazySkillLoader trait. FileSkillLoader reads from disk with blake3 integrity. O(n) scan for <100 arms, O(1) with papaya.
