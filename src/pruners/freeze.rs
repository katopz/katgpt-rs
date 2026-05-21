//! Shared freeze/thaw disk I/O for `repr(C)` bandit knowledge structs.
//!
//! Zero-dependency binary persistence: raw `std::fs::write`/`read` on `repr(C)` data.
//! Magic bytes + version validation on load. No serde/bincode needed.

use std::path::Path;

/// Save a `repr(C)` struct to disk as raw bytes.
///
/// Creates parent directories if needed. Overwrites existing files.
pub fn save_frozen<T>(path: &Path, data: &T) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {:?}: {e}", parent))?;
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of::<T>())
    };
    std::fs::write(path, bytes).map_err(|e| format!("Failed to write {:?}: {e}", path))
}

/// Load a `repr(C)` struct from disk as raw bytes.
///
/// Validates file size matches expected struct size.
/// Caller should call `.validate()` on the result to check magic/version.
pub fn load_frozen<T>(path: &Path) -> Result<T, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read {:?}: {e}", path))?;
    let expected = std::mem::size_of::<T>();
    if bytes.len() != expected {
        return Err(format!(
            "Size mismatch: expected {expected} bytes, got {} bytes from {:?}",
            bytes.len(),
            path
        ));
    }
    // Read from bytes into T
    let mut result = std::mem::MaybeUninit::<T>::uninit();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), result.as_mut_ptr() as *mut u8, expected);
        Ok(result.assume_init())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct TestFrozen {
        magic: [u8; 4],
        version: u32,
        value: f32,
    }

    impl TestFrozen {
        const MAGIC: [u8; 4] = *b"TEST";
        const VERSION: u32 = 1;

        fn new(value: f32) -> Self {
            Self {
                magic: Self::MAGIC,
                version: Self::VERSION,
                value,
            }
        }

        #[allow(dead_code)]
        fn validate(&self) -> Result<(), String> {
            if self.magic != Self::MAGIC {
                return Err("bad magic".into());
            }
            if self.version != Self::VERSION {
                return Err("bad version".into());
            }
            Ok(())
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bin");
        let original = TestFrozen::new(42.5);
        save_frozen(&path, &original).unwrap();
        let loaded: TestFrozen = load_frozen(&path).unwrap();
        assert_eq!(original, loaded);
        assert_eq!(loaded.value, 42.5);
    }

    #[test]
    fn save_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/deep/test.bin");
        let data = TestFrozen::new(1.0);
        save_frozen(&path, &data).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn load_size_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        fs::write(&path, b"too short").unwrap();
        let result: Result<TestFrozen, String> = load_frozen(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Size mismatch"));
    }

    #[test]
    fn load_missing_file() {
        let path = Path::new("/nonexistent/path.bin");
        let result: Result<TestFrozen, String> = load_frozen(path);
        assert!(result.is_err());
    }
}
