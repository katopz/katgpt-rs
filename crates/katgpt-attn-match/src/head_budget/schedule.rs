//! `HeadBudgetSchedule` — serializable per-head budget plan (Plan 271 Phase 3).
//!
//! Wraps a solved per-head budget allocation with:
//! - A `model_id` identifying which model this schedule applies to.
//! - A `version` counter (bumped on re-solve).
//! - A BLAKE3 hash for tamper detection (recomputed on `verify()`).
//!
//! Serialization uses `postcard` (already a project dependency) for compact
//! binary representation. BLAKE3 is used per AGENTS.md ("Use blake3 as
//! possible instead of SHA1, SHA256").

#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

/// A solved per-head budget schedule, ready for serialization and deployment.
///
/// # Tamper detection
/// `blake3_hash` covers `model_id` (UTF-8 bytes) and `shares` (little-endian
/// f32 bytes), in that order. `verify()` recomputes the hash and compares;
/// any mismatch indicates corruption or tampering.
///
/// # Versioning
/// `version` starts at 1 and is bumped by callers when they re-solve with
/// new sensitivity data. The hash does not cover `version` (changing the
/// version alone is a metadata update, not a content change).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadBudgetSchedule {
    /// Model identifier this schedule applies to (e.g., "llama-3-8b").
    pub model_id: String,
    /// Per-head budget shares. Length = num_layers × num_heads. Each entry is
    /// the ratio (in `[0, 1]`) allocated to that head.
    pub shares: Vec<f32>,
    /// Schedule version (bumped on re-solve).
    pub version: u32,
    /// BLAKE3 hash of `(model_id, shares)` for tamper detection.
    pub blake3_hash: [u8; 32],
}

impl HeadBudgetSchedule {
    /// Construct a schedule from a solved shares vector. Computes the BLAKE3
    /// hash automatically and sets `version = 1`.
    pub fn new(model_id: String, shares: Vec<f32>) -> Self {
        let blake3_hash = compute_hash(&model_id, &shares);
        Self {
            model_id,
            shares,
            version: 1,
            blake3_hash,
        }
    }

    /// Construct a schedule with an explicit version. Recomputes the hash.
    pub fn with_version(model_id: String, shares: Vec<f32>, version: u32) -> Self {
        let blake3_hash = compute_hash(&model_id, &shares);
        Self {
            model_id,
            shares,
            version,
            blake3_hash,
        }
    }

    /// Recompute the BLAKE3 hash from `model_id` and `shares`, and verify it
    /// matches the stored `blake3_hash`. Returns `true` if the schedule is
    /// intact (not tampered with).
    pub fn verify(&self) -> bool {
        let recomputed = compute_hash(&self.model_id, &self.shares);
        // Constant-time comparison via BLAKE3's own equality (it's a fixed
        // 32-byte array; we use a simple loop to avoid early-exit timing
        // side channels, though for non-secret data this is overkill).
        let mut diff: u8 = 0;
        for i in 0..32 {
            diff |= recomputed[i] ^ self.blake3_hash[i];
        }
        diff == 0
    }

    /// Serialize to postcard bytes.
    pub fn to_postcard(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize from postcard bytes.
    pub fn from_postcard(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }
}

/// Compute the BLAKE3 hash of `(model_id, shares)`.
///
/// Hash covers:
/// 1. `model_id` as UTF-8 bytes.
/// 2. Each `shares[i]` as little-endian f32 bytes, in order.
///
/// `version` is deliberately not hashed — it's metadata about which solve
/// produced this schedule, not content to be tamper-protected.
#[inline]
pub fn compute_hash(model_id: &str, shares: &[f32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model_id.as_bytes());
    // We use a manual loop rather than `bytemuck::cast_slice` to keep the
    // hashing code dependency-light and to guarantee endianness explicitly.
    for s in shares {
        hasher.update(&s.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_deterministic() {
        let s1 = HeadBudgetSchedule::new("model-a".into(), vec![0.1, 0.2, 0.3, 0.4]);
        let s2 = HeadBudgetSchedule::new("model-a".into(), vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(
            s1.blake3_hash, s2.blake3_hash,
            "same inputs must produce same BLAKE3 hash"
        );
    }

    #[test]
    fn test_blake3_differs_on_model_id() {
        let s1 = HeadBudgetSchedule::new("model-a".into(), vec![0.5, 0.5]);
        let s2 = HeadBudgetSchedule::new("model-b".into(), vec![0.5, 0.5]);
        assert_ne!(
            s1.blake3_hash, s2.blake3_hash,
            "different model_id must produce different hash"
        );
    }

    #[test]
    fn test_blake3_differs_on_shares() {
        let s1 = HeadBudgetSchedule::new("model-a".into(), vec![0.1, 0.9]);
        let s2 = HeadBudgetSchedule::new("model-a".into(), vec![0.9, 0.1]);
        assert_ne!(
            s1.blake3_hash, s2.blake3_hash,
            "different shares must produce different hash"
        );
    }

    #[test]
    fn test_verify_intact() {
        let s = HeadBudgetSchedule::new("model-a".into(), vec![0.25, 0.25, 0.5]);
        assert!(s.verify(), "freshly-constructed schedule must verify");
    }

    #[test]
    fn test_verify_detects_tampering_shares() {
        let mut s = HeadBudgetSchedule::new("model-a".into(), vec![0.25, 0.75]);
        // Tamper: change shares without recomputing hash.
        s.shares[0] = 0.99;
        assert!(!s.verify(), "tampered shares must fail verification");
    }

    #[test]
    fn test_verify_detects_tampering_model_id() {
        let mut s = HeadBudgetSchedule::new("model-a".into(), vec![0.5, 0.5]);
        s.model_id = "model-b".into();
        assert!(!s.verify(), "tampered model_id must fail verification");
    }

    #[test]
    fn test_schedule_roundtrip() {
        let original = HeadBudgetSchedule::with_version(
            "test-model".into(),
            vec![0.1, 0.2, 0.3, 0.4, 0.0],
            7,
        );
        let bytes = original.to_postcard().expect("serialize");
        let recovered = HeadBudgetSchedule::from_postcard(&bytes).expect("deserialize");
        assert_eq!(recovered.model_id, original.model_id);
        assert_eq!(recovered.shares, original.shares);
        assert_eq!(recovered.version, original.version);
        assert_eq!(recovered.blake3_hash, original.blake3_hash);
        assert!(recovered.verify(), "recovered schedule must verify");
    }

    #[test]
    fn test_schedule_empty_shares() {
        // Edge case: zero shares. Should still hash and verify.
        let s = HeadBudgetSchedule::new("empty".into(), vec![]);
        assert!(s.verify());
        assert_eq!(s.shares.len(), 0);
    }

    #[test]
    fn test_deserialize_garbage_fails() {
        let result = HeadBudgetSchedule::from_postcard(&[0xff, 0xff, 0xff]);
        assert!(result.is_err(), "garbage input should fail to deserialize");
    }

    #[test]
    fn test_version_independent_of_hash() {
        // Two schedules with same model_id+shares but different versions
        // must have the same hash (version is metadata, not content).
        let s1 = HeadBudgetSchedule::with_version("m".into(), vec![0.5], 1);
        let s2 = HeadBudgetSchedule::with_version("m".into(), vec![0.5], 99);
        assert_eq!(
            s1.blake3_hash, s2.blake3_hash,
            "version must not affect hash"
        );
    }

    #[test]
    fn test_f32_endianness_in_hash() {
        // Verify that the hash is computed over little-endian f32 bytes by
        // checking against a manual computation.
        let model_id = "x";
        let shares = vec![1.0f32];
        let manual_hash = {
            let mut h = blake3::Hasher::new();
            h.update(model_id.as_bytes());
            h.update(&1.0f32.to_le_bytes());
            *h.finalize().as_bytes()
        };
        let schedule_hash = compute_hash(model_id, &shares);
        assert_eq!(manual_hash, schedule_hash);
    }
}
