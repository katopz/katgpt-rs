//! Core types for sleep-time compute (Plan 334 Phase 1 T1.2).
//!
//! Generic over `D` (latent dim) and `K` (catalog size). No game semantics —
//! the per-NPC direction-vector catalog lives in riir-ai Plan 341.
//!
//! # Latent vs Raw (AGENTS.md)
//!
//! - `AnticipatedQueryDir::direction` → latent, frozen, BLAKE3-committed.
//! - `AnticipatedQuerySet::blake3` / `version` → raw, syncable audit artifact
//!   (the commitment root + monotonic version that the chain quorum signs).
//! - `AnticipatedSlot::precomputed` → latent (the z_i sleep-time compute
//!   output). Stays latent at the sync boundary; only its scalar gate value
//!   crosses as a synced affect scalar.

use katgpt_types::simd::simd_dot_f32;

/// A frozen anticipated-query direction vector. One "slot key" in c'.
///
/// Generic over `D` (latent dim). No game semantics — the direction vectors
/// themselves are game IP and live in riir-ai Plan 341.
///
/// The `blake3` field commits the `direction` bytes so a tampered direction
/// is detectable without re-running sleep-time compute. `version` is a
/// monotonic counter used by freeze/thaw (Plan 025) — the chain quorum signs
/// `(blake3, version)` as the canonical artifact id.
///
/// # Layout
///
/// Fields are ordered by descending alignment (u64 → f32 → u8) per the
/// AGENTS.md struct-padding rule. This eliminates inter-field padding for odd
/// `D` (where `[f32; D]` is 4-aligned but not 8-aligned); Rust struct literals
/// are order-independent so this is API-neutral, and there is no `#[repr(C)]`
/// anywhere in the crate, so the on-disk codec in riir-ai (which uses its own
/// cursor-based byte layout, not `repr(C)`) is unaffected.
#[derive(Clone, Debug)]
pub struct AnticipatedQueryDir<const D: usize> {
    /// Monotonic version for freeze/thaw. Bumped on every swap.
    pub version: u64,
    /// The direction vector itself (unit-ish; the consumer normalizes if needed).
    pub direction: [f32; D],
    /// BLAKE3 of `direction` (little-endian f32 bytes). Computed once at
    /// construction; never recomputed in the hot path.
    pub blake3: [u8; 32],
}

impl<const D: usize> AnticipatedQueryDir<D> {
    /// Construct from a direction vector. Computes the BLAKE3 commitment.
    /// `version` starts at 0; the freeze/thaw layer bumps it on swap.
    #[inline]
    pub fn new(direction: [f32; D]) -> Self {
        let blake3 = commit_direction(&direction);
        Self {
            direction,
            blake3,
            version: 0,
        }
    }

    /// Construct with an explicit version (freeze/thaw reload path).
    #[inline]
    pub fn with_version(direction: [f32; D], version: u64) -> Self {
        let blake3 = commit_direction(&direction);
        Self {
            direction,
            blake3,
            version,
        }
    }

    /// Dot-product with a context/query vector. Convenience wrapper around
    /// `simd_dot_f32` so consumers don't pull in the simd module directly.
    #[inline]
    pub fn dot(&self, other: &[f32; D]) -> f32 {
        simd_dot_f32(&self.direction, other, D)
    }

    /// Verify the cached BLAKE3 matches the direction bytes. Cheap audit hook.
    #[inline]
    pub fn verify_commitment(&self) -> bool {
        commit_direction(&self.direction) == self.blake3
    }
}

/// BLAKE3 over a direction vector's little-endian f32 bytes.
///
/// Public so the anticipator can re-derive the slot commitment when slots
/// are mutated (e.g. during sleep-time compute that updates `precomputed`).
#[inline]
pub fn commit_direction<const D: usize>(direction: &[f32; D]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for &f in direction {
        hasher.update(&f.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// One slot in the anticipated-query projection set c'.
///
/// Output of sleep-time compute for one (`c`, `dir`) pair: the precomputed
/// latent answer `z_i` and its predictability score `p_i ∈ [0,1]`.
#[derive(Clone, Debug)]
pub struct AnticipatedSlot<const D: usize> {
    /// The anticipated-query direction this slot answers.
    pub dir: AnticipatedQueryDir<D>,
    /// The precomputed latent answer z_i (sleep-time compute output).
    pub precomputed: [f32; D],
    /// Predictability score p_i ∈ [0,1]. Higher = more predictable = more
    /// sleep-time compute was warranted for this direction.
    pub predictability: f32,
}

/// The full c' artifact — the output of sleep-time compute.
///
/// Reusable across consumers: one NPC's sleep-time compute amortizes over
/// all players who later query that NPC. BLAKE3 commits the whole set so a
/// single byte-tamper anywhere is detectable.
///
/// Generic over `D` (latent dim) and `K` (catalog size). The consumer's wake
/// path scans all `K` slots — `K` is bounded (paper uses K≤10; we expect
/// K≤8 per NPC), so the O(K) scan is hot-tier-cheap.
///
/// # Layout
///
/// Fields ordered by descending alignment (`slots` carries an 8-align `u64`
/// inside each `AnticipatedSlot`, then `version: u64`, then `blake3: [u8; 32]`)
/// per the AGENTS.md struct-padding rule — eliminates the padding that the
/// prior `slots → blake3 → version` order forced between `blake3` and `version`.
#[derive(Clone, Debug)]
pub struct AnticipatedQuerySet<const D: usize, const K: usize> {
    /// The K slots, one per anticipated-query direction.
    pub slots: [AnticipatedSlot<D>; K],
    /// Monotonic version. Bumped on every anticipate() call. The chain
    /// quorum signs `(blake3, version)` as the c' artifact id.
    pub version: u64,
    /// BLAKE3 over all slot bytes (dirs + precomputed + predictability).
    /// Recomputed by `SleepTimeAnticipator::anticipate` on every emission.
    pub blake3: [u8; 32],
}

impl<const D: usize, const K: usize> AnticipatedQuerySet<D, K> {
    /// Recompute the BLAKE3 over all slot bytes. Called by `anticipate()`
    /// after the slots are populated; also callable by audit paths.
    #[inline]
    pub fn commit_slots(slots: &[AnticipatedSlot<D>; K]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for slot in slots {
            hasher.update(&slot.dir.blake3);
            for &f in &slot.precomputed {
                hasher.update(&f.to_le_bytes());
            }
            hasher.update(&slot.predictability.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    /// Verify the cached BLAKE3 matches the current slot bytes.
    #[inline]
    pub fn verify_commitment(&self) -> bool {
        Self::commit_slots(&self.slots) == self.blake3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_commitment_is_deterministic() {
        let d1 = AnticipatedQueryDir::new([1.0, 2.0, 3.0]);
        let d2 = AnticipatedQueryDir::new([1.0, 2.0, 3.0]);
        assert_eq!(d1.blake3, d2.blake3, "same direction → same commitment");
    }

    #[test]
    fn direction_commitment_distinguishes_perturbations() {
        let d1 = AnticipatedQueryDir::new([1.0, 2.0, 3.0]);
        // Smallest possible perturbation: 1 ULP on the last element.
        let perturbed = [1.0, 2.0, f32::from_bits(3.0f32.to_bits() + 1)];
        let d2 = AnticipatedQueryDir::new(perturbed);
        assert_ne!(
            d1.blake3, d2.blake3,
            "1-ULP perturbation must change BLAKE3"
        );
    }

    #[test]
    fn direction_verify_commitment_roundtrip() {
        let d = AnticipatedQueryDir::new([0.5, -0.5, 0.0, 1.0]);
        assert!(d.verify_commitment());
    }

    #[test]
    fn direction_dot_uses_simd_kernel() {
        let d = AnticipatedQueryDir::new([1.0, 2.0, 3.0, 4.0]);
        let q = [1.0; 4];
        // 1+2+3+4 = 10
        assert!((d.dot(&q) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn direction_with_version_preserves_commitment() {
        let d0 = AnticipatedQueryDir::new([1.0, 2.0]);
        let d5 = AnticipatedQueryDir::with_version([1.0, 2.0], 5);
        assert_eq!(d0.blake3, d5.blake3, "version does not affect dir blake3");
        assert_eq!(d0.version, 0);
        assert_eq!(d5.version, 5);
    }

    #[test]
    fn slot_set_commitment_is_deterministic() {
        let mk_slots = || {
            [
                AnticipatedSlot {
                    dir: AnticipatedQueryDir::new([1.0, 0.0]),
                    precomputed: [2.0, 3.0],
                    predictability: 0.8,
                },
                AnticipatedSlot {
                    dir: AnticipatedQueryDir::new([0.0, 1.0]),
                    precomputed: [4.0, 5.0],
                    predictability: 0.6,
                },
            ]
        };
        let h1 = AnticipatedQuerySet::<2, 2>::commit_slots(&mk_slots());
        let h2 = AnticipatedQuerySet::<2, 2>::commit_slots(&mk_slots());
        assert_eq!(h1, h2, "same slots → same commitment");
    }

    #[test]
    fn slot_set_commitment_detects_tamper() {
        let mut slots = [
            AnticipatedSlot {
                dir: AnticipatedQueryDir::new([1.0, 0.0]),
                precomputed: [2.0, 3.0],
                predictability: 0.8,
            },
            AnticipatedSlot {
                dir: AnticipatedQueryDir::new([0.0, 1.0]),
                precomputed: [4.0, 5.0],
                predictability: 0.6,
            },
        ];
        let h_before = AnticipatedQuerySet::<2, 2>::commit_slots(&slots);
        // Tamper: bump predictability by 1 ULP on slot 0.
        slots[0].predictability = f32::from_bits(slots[0].predictability.to_bits() + 1);
        let h_after = AnticipatedQuerySet::<2, 2>::commit_slots(&slots);
        assert_ne!(h_before, h_after, "tamper must change commitment");
    }
}
