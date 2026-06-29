//! Freeze/Thaw support for Dreamer consolidated banks.
//!
//! Persists dreamer pipeline state to disk using raw `repr(C)` binary format.
//! No serde/bincode needed — uses the same approach as `crate::freeze`.

use std::path::Path;

use crate::freeze::{load_frozen, save_frozen};

use super::pipeline::DreamerPipeline;

// ---------------------------------------------------------------------------
// DreamerFrozenBank
// ---------------------------------------------------------------------------

/// Frozen snapshot of a Dreamer pipeline state.
///
/// `repr(C)` for stable binary layout across compilations.
/// Use `save_frozen_dreamer` / `load_frozen_dreamer` for disk I/O.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DreamerFrozenBank {
    /// Magic bytes: `b"DRMR"`.
    pub magic: [u8; 4],
    /// Format version (currently 1).
    pub version: u32,
    /// Episode counter at save time.
    pub current_episode: u64,
    /// Number of consolidations performed.
    pub consolidation_count: u64,
    /// Cumulative arms before consolidations.
    pub total_arms_before: u64,
    /// Cumulative arms after consolidations.
    pub total_arms_after: u64,
    /// Cumulative arms forgotten.
    pub total_forgotten: u64,
    /// Config cadence at save time.
    pub cadence: u64,
    /// Config decay factor at save time.
    pub decay_factor: f32,
    /// Reserved for future use (explicit padding).
    pub _reserved: [u8; 4],
}

impl DreamerFrozenBank {
    pub const MAGIC: [u8; 4] = *b"DRMR";
    pub const VERSION: u32 = 1;

    /// Create a frozen snapshot from a live pipeline.
    pub fn from_pipeline(pipeline: &DreamerPipeline) -> Self {
        let config = pipeline.config();
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            current_episode: pipeline.episode() as u64,
            consolidation_count: pipeline.consolidation_count() as u64,
            total_arms_before: 0,
            total_arms_after: 0,
            total_forgotten: 0,
            cadence: config.cadence as u64,
            decay_factor: config.decay_factor,
            _reserved: [0u8; 4],
        }
    }

    /// Create a frozen snapshot with cumulative stats from a pipeline and result history.
    pub fn from_pipeline_with_stats(
        pipeline: &DreamerPipeline,
        total_arms_before: u64,
        total_arms_after: u64,
        total_forgotten: u64,
    ) -> Self {
        let mut bank = Self::from_pipeline(pipeline);
        bank.total_arms_before = total_arms_before;
        bank.total_arms_after = total_arms_after;
        bank.total_forgotten = total_forgotten;
        bank
    }

    /// Validate magic bytes and version.
    pub fn validate(&self) -> Result<(), String> {
        match self.magic {
            m if m == Self::MAGIC => {}
            m => {
                return Err(format!(
                    "DreamerFrozenBank: bad magic {:?}, expected {:?}",
                    m,
                    Self::MAGIC
                ));
            }
        }
        match self.version {
            v if v == Self::VERSION => {}
            v => {
                return Err(format!(
                    "DreamerFrozenBank: bad version {v}, expected {}",
                    Self::VERSION
                ));
            }
        }
        Ok(())
    }

    /// Human-readable summary of the frozen bank state.
    pub fn summary(&self) -> String {
        let arms_delta = self.total_arms_before.saturating_sub(self.total_arms_after);
        format!(
            "DreamerFrozenBank(episode={}, consolidations={}, arms_before={}, arms_after={}, forgotten={}, delta={arms_delta}, cadence={}, decay={})",
            self.current_episode,
            self.consolidation_count,
            self.total_arms_before,
            self.total_arms_after,
            self.total_forgotten,
            self.cadence,
            self.decay_factor,
        )
    }

    /// Arms saved through consolidation (before - after).
    pub fn arms_saved(&self) -> u64 {
        self.total_arms_before.saturating_sub(self.total_arms_after)
    }

    /// Compression ratio (arms_after / arms_before). Returns 1.0 if no arms.
    pub fn compression_ratio(&self) -> f32 {
        match self.total_arms_before {
            0 => 1.0,
            before => self.total_arms_after as f32 / before as f32,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Save a dreamer pipeline snapshot to disk.
pub fn save_frozen_dreamer(path: &Path, pipeline: &DreamerPipeline) -> Result<(), String> {
    let bank = DreamerFrozenBank::from_pipeline(pipeline);
    save_frozen(path, &bank)
}

/// Save a dreamer pipeline snapshot with cumulative stats to disk.
pub fn save_frozen_dreamer_with_stats(
    path: &Path,
    pipeline: &DreamerPipeline,
    total_arms_before: u64,
    total_arms_after: u64,
    total_forgotten: u64,
) -> Result<(), String> {
    let bank = DreamerFrozenBank::from_pipeline_with_stats(
        pipeline,
        total_arms_before,
        total_arms_after,
        total_forgotten,
    );
    save_frozen(path, &bank)
}

/// Load a dreamer frozen bank from disk.
///
/// Validates magic and version before returning.
pub fn load_frozen_dreamer(path: &Path) -> Result<DreamerFrozenBank, String> {
    let bank: DreamerFrozenBank = load_frozen(path)?;
    bank.validate()?;
    Ok(bank)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    use crate::dreamer::types::DreamerConfig;

    fn make_pipeline(config: DreamerConfig) -> DreamerPipeline {
        DreamerPipeline::new(config)
    }

    // ---- DreamerFrozenBank struct ----

    #[test]
    fn test_magic_and_version_constants() {
        assert_eq!(DreamerFrozenBank::MAGIC, *b"DRMR");
        assert_eq!(DreamerFrozenBank::VERSION, 1);
    }

    #[test]
    fn test_from_pipeline_snapshots_state() {
        let pipeline = make_pipeline(DreamerConfig {
            cadence: 7,
            decay_factor: 0.85,
            ..DreamerConfig::default()
        });
        let bank = DreamerFrozenBank::from_pipeline(&pipeline);

        assert_eq!(bank.magic, *b"DRMR");
        assert_eq!(bank.version, 1);
        assert_eq!(bank.current_episode, 0);
        assert_eq!(bank.consolidation_count, 0);
        assert_eq!(bank.cadence, 7);
        assert!((bank.decay_factor - 0.85).abs() < f32::EPSILON);
        assert_eq!(bank._reserved, [0u8; 4]);
    }

    #[test]
    fn test_from_pipeline_with_stats() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let bank = DreamerFrozenBank::from_pipeline_with_stats(&pipeline, 100, 60, 25);

        assert_eq!(bank.total_arms_before, 100);
        assert_eq!(bank.total_arms_after, 60);
        assert_eq!(bank.total_forgotten, 25);
    }

    // ---- validate ----

    #[test]
    fn test_validate_valid_bank() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let bank = DreamerFrozenBank::from_pipeline(&pipeline);
        assert!(bank.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_magic() {
        let mut bank = DreamerFrozenBank::from_pipeline(&make_pipeline(DreamerConfig::default()));
        bank.magic = *b"XXXX";
        let err = bank.validate().unwrap_err();
        assert!(err.contains("bad magic"));
    }

    #[test]
    fn test_validate_bad_version() {
        let mut bank = DreamerFrozenBank::from_pipeline(&make_pipeline(DreamerConfig::default()));
        bank.version = 99;
        let err = bank.validate().unwrap_err();
        assert!(err.contains("bad version"));
    }

    // ---- summary ----

    #[test]
    fn test_summary_contains_key_fields() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let bank = DreamerFrozenBank::from_pipeline_with_stats(&pipeline, 200, 120, 50);
        let s = bank.summary();
        assert!(s.contains("episode=0"));
        assert!(s.contains("consolidations=0"));
        assert!(s.contains("arms_before=200"));
        assert!(s.contains("arms_after=120"));
        assert!(s.contains("forgotten=50"));
        assert!(s.contains("delta=80"));
    }

    // ---- arms_saved / compression_ratio ----

    #[test]
    fn test_arms_saved() {
        let bank = DreamerFrozenBank {
            magic: DreamerFrozenBank::MAGIC,
            version: DreamerFrozenBank::VERSION,
            current_episode: 100,
            consolidation_count: 10,
            total_arms_before: 200,
            total_arms_after: 120,
            total_forgotten: 50,
            cadence: 10,
            decay_factor: 0.9,
            _reserved: [0; 4],
        };
        assert_eq!(bank.arms_saved(), 80);
    }

    #[test]
    fn test_arms_saved_zero() {
        let bank = DreamerFrozenBank {
            magic: DreamerFrozenBank::MAGIC,
            version: DreamerFrozenBank::VERSION,
            current_episode: 0,
            consolidation_count: 0,
            total_arms_before: 0,
            total_arms_after: 0,
            total_forgotten: 0,
            cadence: 10,
            decay_factor: 0.9,
            _reserved: [0; 4],
        };
        assert_eq!(bank.arms_saved(), 0);
    }

    #[test]
    fn test_compression_ratio_normal() {
        let bank = DreamerFrozenBank {
            magic: DreamerFrozenBank::MAGIC,
            version: DreamerFrozenBank::VERSION,
            current_episode: 0,
            consolidation_count: 0,
            total_arms_before: 100,
            total_arms_after: 50,
            total_forgotten: 0,
            cadence: 10,
            decay_factor: 0.9,
            _reserved: [0; 4],
        };
        assert!((bank.compression_ratio() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compression_ratio_zero_before() {
        let bank = DreamerFrozenBank {
            magic: DreamerFrozenBank::MAGIC,
            version: DreamerFrozenBank::VERSION,
            current_episode: 0,
            consolidation_count: 0,
            total_arms_before: 0,
            total_arms_after: 0,
            total_forgotten: 0,
            cadence: 10,
            decay_factor: 0.9,
            _reserved: [0; 4],
        };
        assert!((bank.compression_ratio() - 1.0).abs() < f32::EPSILON);
    }

    // ---- save/load roundtrip ----

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dreamer.bin");

        let pipeline = make_pipeline(DreamerConfig {
            cadence: 15,
            decay_factor: 0.88,
            ..DreamerConfig::default()
        });

        save_frozen_dreamer(&path, &pipeline).unwrap();

        let loaded = load_frozen_dreamer(&path).unwrap();

        assert_eq!(loaded.magic, *b"DRMR");
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.current_episode, 0);
        assert_eq!(loaded.cadence, 15);
        assert!((loaded.decay_factor - 0.88).abs() < f32::EPSILON);
    }

    #[test]
    fn test_save_load_with_stats_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dreamer_stats.bin");

        let pipeline = make_pipeline(DreamerConfig::default());

        save_frozen_dreamer_with_stats(&path, &pipeline, 500, 300, 120).unwrap();

        let loaded = load_frozen_dreamer(&path).unwrap();

        assert_eq!(loaded.total_arms_before, 500);
        assert_eq!(loaded.total_arms_after, 300);
        assert_eq!(loaded.total_forgotten, 120);
    }

    #[test]
    fn test_save_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/deep/dreamer.bin");

        let pipeline = make_pipeline(DreamerConfig::default());
        save_frozen_dreamer(&path, &pipeline).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_load_missing_file() {
        let path = Path::new("/nonexistent/dreamer.bin");
        let result = load_frozen_dreamer(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_corrupted_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt.bin");

        let pipeline = make_pipeline(DreamerConfig::default());
        save_frozen_dreamer(&path, &pipeline).unwrap();

        // Corrupt the magic bytes
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = b'X';
        bytes[1] = b'X';
        fs::write(&path, &bytes).unwrap();

        let result = load_frozen_dreamer(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("bad magic"));
    }

    #[test]
    fn test_load_size_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tooshort.bin");
        fs::write(&path, b"tiny").unwrap();

        let result = load_frozen_dreamer(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Size mismatch"));
    }

    // ---- copy semantics ----

    #[test]
    fn test_copy_is_independent() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let a = DreamerFrozenBank::from_pipeline(&pipeline);
        let mut b = a;
        b.current_episode = 999;
        assert_eq!(a.current_episode, 0);
        assert_eq!(b.current_episode, 999);
    }

    // ---- conservative / aggressive configs ----

    #[test]
    fn test_conservative_config_cadence_preserved() {
        let pipeline = make_pipeline(DreamerConfig::conservative());
        let bank = DreamerFrozenBank::from_pipeline(&pipeline);
        assert_eq!(bank.cadence, 20);
        assert!((bank.decay_factor - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aggressive_config_cadence_preserved() {
        let pipeline = make_pipeline(DreamerConfig::aggressive());
        let bank = DreamerFrozenBank::from_pipeline(&pipeline);
        assert_eq!(bank.cadence, 5);
        assert!((bank.decay_factor - 0.8).abs() < f32::EPSILON);
    }
}
