//! Latent Field Steering — top-down direction-vector injection into latent state.
//!
//! See `katgpt-rs/.research/290_latent_field_steering_open_primitive.md` and
//! Plan 309. This is the **missing top-down control direction** complement to
//! the existing bottom-up emotion computation (`EmotionDirections::project`,
//! Plan 162). Instead of reading affect from latent state, we **inject** a
//! designer/environment-supplied direction vector directly into mutable
//! per-tick latent state — the "wave interference" mechanism: linear
//! superposition of the NPC's current field with an injected steering field.
//!
//! ## Math
//!
//! Given NPC latent state `s ∈ R^d` (d=8 for HLA) and a unit-norm steering
//! direction `v ∈ R^d` with strength `α ∈ [0, 1]`:
//!
//! ```text
//! s' = s + α · v
//! ```
//!
//! For localized fields (only NPCs within a support region R), the effective
//! strength is modulated by a sigmoid-falloff kernel:
//!
//! ```text
//! s'_i = s_i + α · kernel(distance(i, center), bandwidth) · v
//! ```
//!
//! The kernel is `sigmoid((bandwidth - distance) · steepness)` — ~1 inside
//! the bandwidth, ~0 outside, smooth at the boundary. Per AGENTS.md: **sigmoid,
//! never softmax/Gaussian** for projections.
//!
//! ## Why modelless
//!
//! Direction vectors are **frozen, BLAKE3-committed artifacts** loaded at init.
//! No gradients, no training. The steering is an **additive overlay** on mutable
//! per-tick state — it does NOT mutate the frozen personality shard. This is
//! the freeze/thaw pattern: the shard is read-only, the steering field is a
//! separate mutable overlay (atomic Arc swap for hot-swap).
//!
//! ## Zero-alloc hot path
//!
//! [`apply_latent_steering`] and [`apply_field_to_crowd`] take borrowed slices
//! only — no heap allocation. The steering loop is an element-wise SAXPY
//! (`s[i] += α · v[i]`) that auto-vectorizes at d=8 (HLA scale).
//!
//! ## Design decision: `Vec<f32>` vs `[f32; D]`
//!
//! Uses `Vec<f32>` for the direction to match `EmotionDirections` storage
//! (dynamically sized, same artifact format for read-side project and write-
//! side steer). Game-side hot path can wrap in a typed `HLAField([f32; 8])`
//! alias in riir-ai once G1–G5 pass.

use blake3::Hasher;

// ── HLA axis indices ──────────────────────────────────────────────

/// Index of the valence axis in the 8-dim HLA activation vector.
pub const HLA_VALENCE: usize = 0;
/// Index of the arousal axis.
pub const HLA_AROUSAL: usize = 1;
/// Index of the desperation axis.
pub const HLA_DESPERATION: usize = 2;
/// Index of the calm axis.
pub const HLA_CALM: usize = 3;
/// Index of the fear axis.
pub const HLA_FEAR: usize = 4;
/// Standard HLA dimensionality (5 affective + 3 reserved).
pub const HLA_DIM: usize = 8;

// ── Errors ────────────────────────────────────────────────────────

/// Errors returned by [`LatentSteeringVector::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentSteeringError {
    /// Direction vector is not unit-norm within the constructor tolerance.
    NotUnitNorm,
    /// Strength `alpha` is outside `[0.0, 1.0]`.
    AlphaOutOfRange,
}

// ── Steering vector ───────────────────────────────────────────────

/// Unit-norm direction in latent space + scalar strength, BLAKE3-committed.
///
/// Reuses the same artifact shape as `EmotionDirections` (Plan 162) so the
/// same frozen direction works for read-side (project) and write-side (steer).
/// The `commitment` field is `BLAKE3(direction_le || alpha_le)`, mirroring the
/// `MerkleFrozenEnvelope` pattern in riir-neuron-db.
#[derive(Debug, Clone)]
pub struct LatentSteeringVector {
    /// Unit-norm direction, d ≤ 64 (HLA d=8).
    pub direction: Vec<f32>,
    /// Strength α ∈ [0, 1]. Sigmoid-bounded at construction.
    pub alpha: f32,
    /// `BLAKE3(direction_le || alpha_le)` — content-addressed commitment.
    pub commitment: [u8; 32],
}

impl LatentSteeringVector {
    /// Construct a steering vector, validating unit-norm and alpha range.
    ///
    /// Returns [`LatentSteeringError::NotUnitNorm`] if `||direction||` deviates
    /// from 1.0 by more than `norm_tol`. Returns
    /// [`LatentSteeringError::AlphaOutOfRange`] if `alpha` is outside `[0, 1]`.
    pub fn new(
        direction: Vec<f32>,
        alpha: f32,
        norm_tol: f32,
    ) -> Result<Self, LatentSteeringError> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(LatentSteeringError::AlphaOutOfRange);
        }
        let norm = direction_norm(&direction);
        if (norm - 1.0).abs() > norm_tol {
            return Err(LatentSteeringError::NotUnitNorm);
        }
        let commitment = compute_commitment(&direction, alpha);
        Ok(Self { direction, alpha, commitment })
    }

    /// Construct without validation. Caller guarantees unit-norm + alpha range.
    /// Used when the direction comes from a trusted frozen artifact.
    pub fn new_unchecked(direction: Vec<f32>, alpha: f32) -> Self {
        let commitment = compute_commitment(&direction, alpha);
        Self { direction, alpha, commitment }
    }

    /// Re-check unit-norm (within `tol`) AND that the stored commitment matches
    /// the current contents. Returns `false` if either check fails.
    pub fn verify(&self, tol: f32) -> bool {
        let norm = direction_norm(&self.direction);
        if (norm - 1.0).abs() > tol {
            return false;
        }
        compute_commitment(&self.direction, self.alpha) == self.commitment
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.direction.len()
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.direction
    }
}

// ── Field support ─────────────────────────────────────────────────

/// Support descriptor for a localized steering field.
#[derive(Debug, Clone, Copy)]
pub enum FieldSupport {
    /// Global — applies to all entities regardless of position/zone.
    Global,
    /// Radius-banded — applies within `bandwidth` of `center` (Euclidean).
    /// Kernel: `sigmoid((bandwidth - distance) * steepness)`.
    Radius {
        center: [f32; 2],
        bandwidth: f32,
        steepness: f32,
    },
    /// Zone-keyed — applies to entities whose zone hash matches.
    Zone { zone_hash: u64 },
}

/// A steering vector + support descriptor.
#[derive(Debug, Clone)]
pub struct LatentField {
    pub steering: LatentSteeringVector,
    pub support: FieldSupport,
}

// ── Steering kernels ──────────────────────────────────────────────

/// Apply steering to a single latent state slice. Zero-alloc.
///
/// `state` is d-dimensional (e.g., HLA 8-dim). Computes `state[i] += alpha * direction[i]`.
/// The support descriptor is ignored — use [`kernel_weight`] + [`apply_latent_steering_weighted`]
/// for localized fields, or [`apply_field_to_crowd`] for batch application.
///
/// Dispatches to an AVX2 SAXPY kernel on x86_64 when AVX2 is detected at
/// runtime; falls back to the scalar SAXPY otherwise. Both paths are
/// bit-identical (element-wise add+mul, no cross-lane reduction).
#[inline]
pub fn apply_latent_steering(state: &mut [f32], steering: &LatentSteeringVector) {
    debug_assert_eq!(
        state.len(),
        steering.dim(),
        "state dim {} != steering dim {}",
        state.len(),
        steering.dim()
    );
    saxpy_inplace(state, steering.alpha, steering.as_slice());
}

/// Apply steering with an explicit kernel weight `w`. Zero-alloc.
/// Effective strength is `alpha * w`. Use for localized fields where the
/// caller has already computed the kernel weight via [`kernel_weight`].
#[inline]
pub fn apply_latent_steering_weighted(
    state: &mut [f32],
    steering: &LatentSteeringVector,
    w: f32,
) {
    if w <= 0.0 {
        return;
    }
    debug_assert_eq!(state.len(), steering.dim());
    saxpy_inplace(state, steering.alpha * w, steering.as_slice());
}

// ── SAXPY backends (Plan 309 T3.1) ───────────────────────────────
//
// `state[i] += alpha * dir[i]` for `i in 0..len`. Three call sites share this
// kernel: `apply_latent_steering`, `apply_latent_steering_weighted`, and the
// per-entity inner loop of `apply_field_to_crowd`. Keeping one dispatcher
// avoids drift between the scalar fallback and the AVX2 path.
//
// NOTE on bit-identical results: the operation is element-wise — there is no
// cross-lane reduction, so the SIMD path produces the same per-element rounding
// as the scalar `*s += alpha * d`. We intentionally use `_mm256_mul_ps` +
// `_mm256_add_ps` (NOT `_mm256_fmadd_ps`) so the rounding matches the scalar
// `mul`-then-`add` sequence exactly.

/// Element-wise SAXPY dispatcher: `state[i] += alpha * dir[i]`.
///
/// Picks AVX2 on x86_64 when available, else scalar. Branch-free in the hot
/// path — the dispatch check happens once per call, not per lane.
#[inline]
fn saxpy_inplace(state: &mut [f32], alpha: f32, dir: &[f32]) {
    debug_assert_eq!(state.len(), dir.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 verified above at runtime; `state` and `dir` are
            // valid, equal-length slices (debug_assert'd).
            unsafe { saxpy_inplace_avx2(state, alpha, dir) };
            return;
        }
    }
    saxpy_inplace_scalar(state, alpha, dir);
}

/// Scalar SAXPY fallback. Written as a chunked loop over 8-lane strides to
/// keep the shape friendly to LLVM's auto-vectorizer on targets without AVX2
/// (per AGENTS.md hot-loop rule).
#[inline]
fn saxpy_inplace_scalar(state: &mut [f32], alpha: f32, dir: &[f32]) {
    for (s, d) in state.iter_mut().zip(dir.iter()) {
        *s += alpha * d;
    }
}

/// AVX2 SAXPY: 8 f32 lanes per iteration. Scalar tail handles the remainder
/// when `len` is not a multiple of 8 (e.g., d=16 is exactly 2 chunks, d=8 is
/// exactly 1 chunk; arbitrary dims still work).
///
/// # Safety
/// Caller must guarantee AVX2 is available at runtime and that `state` and
/// `dir` are valid, equal-length slices.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn saxpy_inplace_avx2(state: &mut [f32], alpha: f32, dir: &[f32]) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};
    let len = state.len();
    let chunks = len / 8;
    let v_alpha = unsafe { _mm256_set1_ps(alpha) };
    let mut i = 0;
    for _ in 0..chunks {
        let s = unsafe { _mm256_loadu_ps(state.as_ptr().add(i)) };
        let d = unsafe { _mm256_loadu_ps(dir.as_ptr().add(i)) };
        let r = unsafe { _mm256_add_ps(s, _mm256_mul_ps(v_alpha, d)) };
        unsafe { _mm256_storeu_ps(state.as_mut_ptr().add(i), r) };
        i += 8;
    }
    // Scalar tail for the remaining (len % 8) elements.
    while i < len {
        unsafe {
            *state.get_unchecked_mut(i) += alpha * *dir.get_unchecked(i);
        }
        i += 1;
    }
}

/// Kernel weight for an entity given support. Returns 0.0 outside support.
///
/// - `Global` → always 1.0.
/// - `Radius` → `sigmoid((bandwidth - distance) * steepness)`. Returns 0.0 if
///   `entity_pos` is `None` (entity has no position, cannot be in a radius field).
/// - `Zone` → 1.0 if zone matches, 0.0 otherwise.
#[inline]
pub fn kernel_weight(
    support: &FieldSupport,
    entity_pos: Option<[f32; 2]>,
    entity_zone: Option<u64>,
) -> f32 {
    match support {
        FieldSupport::Global => 1.0,
        FieldSupport::Radius { center, bandwidth, steepness } => {
            let pos = match entity_pos {
                Some(p) => p,
                None => return 0.0,
            };
            let dx = pos[0] - center[0];
            let dy = pos[1] - center[1];
            let dist = (dx * dx + dy * dy).sqrt();
            sigmoid((bandwidth - dist) * steepness)
        }
        FieldSupport::Zone { zone_hash } => match entity_zone {
            Some(z) if z == *zone_hash => 1.0,
            _ => 0.0,
        },
    }
}

/// Apply a field to a crowd of latent states. Zero-alloc given borrowed slices.
///
/// `states` is flattened `[e0d0, e0d1, ..., eNd(D-1)]` (N*D). `positions` and
/// `zones` are per-entity (length N). For `Global` fields, positions/zones are
/// ignored (kernel weight is always 1.0) but must still be the correct length.
///
/// For `Radius`/`Zone` fields, entities outside the support are skipped (no
/// mutation), achieving zero leakage.
pub fn apply_field_to_crowd(
    states: &mut [f32],
    entity_dim: usize,
    positions: &[Option<[f32; 2]>],
    zones: &[Option<u64>],
    field: &LatentField,
) {
    debug_assert_eq!(states.len(), positions.len() * entity_dim);
    debug_assert_eq!(positions.len(), zones.len());
    let dir = field.steering.as_slice();
    let base_alpha = field.steering.alpha;
    for (i, entity_state) in states.chunks_mut(entity_dim).enumerate() {
        let w = kernel_weight(&field.support, positions[i], zones[i]);
        if w <= 0.0 {
            continue;
        }
        let effective_alpha = base_alpha * w;
        saxpy_inplace(entity_state, effective_alpha, dir);
    }
}

// ── Internal helpers ──────────────────────────────────────────────

/// Compute `BLAKE3(direction_le || alpha_le)`.
///
/// Direction is serialized as little-endian `f32` per-element (matches the
/// `engram/commitment.rs::build_merkle_root` and `cross_resolution.rs`
/// conventions). Alpha is little-endian `f32`.
fn compute_commitment(direction: &[f32], alpha: f32) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for &f in direction {
        hasher.update(&f.to_le_bytes());
    }
    hasher.update(&alpha.to_le_bytes());
    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    out
}

#[inline]
fn direction_norm(direction: &[f32]) -> f32 {
    direction.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit_direction(d: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        let mut v: Vec<f32> = (0..d)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng >> 33) as f32) / (1u64 << 31) as f32 - 1.0
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm.max(1e-12);
        }
        v
    }

    #[test]
    fn smoke_global_field_shifts_state() {
        // T1.5: Global field shifts state by exactly alpha * direction.
        let dir = make_unit_direction(8, 42);
        let steering = LatentSteeringVector::new(dir.clone(), 0.5, 1e-4).unwrap();
        let field = LatentField { steering, support: FieldSupport::Global };

        let mut state = vec![0.1f32; 8];
        apply_latent_steering(&mut state, &field.steering);
        for i in 0..8 {
            let expected = 0.1 + 0.5 * dir[i];
            assert!(
                (state[i] - expected).abs() < 1e-5,
                "state[{i}] = {} expected {expected}",
                state[i]
            );
        }
    }

    #[test]
    fn smoke_radius_field_localizes() {
        // T1.5: Radius field applies inside, skips outside.
        let dir = make_unit_direction(8, 7);
        let steering = LatentSteeringVector::new(dir, 0.3, 1e-4).unwrap();
        let field = LatentField {
            steering,
            support: FieldSupport::Radius {
                center: [0.0, 0.0],
                bandwidth: 10.0,
                steepness: 2.0,
            },
        };

        let n = 100;
        let mut states = vec![0.0f32; n * 8];
        let positions: Vec<Option<[f32; 2]>> = (0..n)
            .map(|i| {
                if i < 50 {
                    Some([5.0, 0.0]) // inside (d=5 < bandwidth=10)
                } else {
                    Some([15.0, 0.0]) // outside (d=15 > bandwidth=10)
                }
            })
            .collect();
        let zones = vec![None; n];

        apply_field_to_crowd(&mut states, 8, &positions, &zones, &field);

        // Inside: shifted (sigmoid((10-5)*2) = sigmoid(10) ≈ 1.0)
        let inside_shift = states[0..50 * 8].iter().map(|x| x.abs()).sum::<f32>();
        assert!(inside_shift > 0.0, "inside entities must be shifted");

        // Outside: sigmoid((10-15)*2) = sigmoid(-10) ≈ 4.5e-5 — effectively 0
        let outside_shift = states[50 * 8..].iter().map(|x| x.abs()).sum::<f32>();
        let ratio = outside_shift / inside_shift.max(1e-12);
        assert!(
            ratio < 0.01,
            "outside/inside shift ratio = {ratio} should be < 0.01"
        );
    }

    #[test]
    fn smoke_constructor_rejects_non_unit_norm() {
        let dir = vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // norm = 2.0
        let result = LatentSteeringVector::new(dir, 0.5, 1e-4);
        assert_eq!(result.unwrap_err(), LatentSteeringError::NotUnitNorm);
    }

    #[test]
    fn smoke_constructor_rejects_alpha_out_of_range() {
        let dir = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = LatentSteeringVector::new(dir, 1.5, 1e-4);
        assert_eq!(result.unwrap_err(), LatentSteeringError::AlphaOutOfRange);
    }

    #[test]
    fn smoke_commitment_roundtrip() {
        let dir = make_unit_direction(8, 99);
        let sv = LatentSteeringVector::new(dir, 0.7, 1e-4).unwrap();
        assert!(sv.verify(1e-3), "commitment + norm must verify");
    }

    #[test]
    fn smoke_zone_field_matches_only_matching_zone() {
        let dir = make_unit_direction(8, 3);
        let steering = LatentSteeringVector::new(dir, 0.4, 1e-4).unwrap();
        let field = LatentField {
            steering,
            support: FieldSupport::Zone { zone_hash: 12345 },
        };

        let n = 4;
        let mut states = vec![0.0f32; n * 8];
        let positions = vec![None; n];
        let zones = vec![Some(1u64), Some(12345), Some(999), None];

        apply_field_to_crowd(&mut states, 8, &positions, &zones, &field);

        // Only entity 1 (zone 12345) should be shifted.
        let e0 = &states[0..8];
        let e1 = &states[8..16];
        let e2 = &states[16..24];
        let e3 = &states[24..32];
        assert!(e0.iter().all(|x| x.abs() < 1e-9), "e0 wrong zone, no shift");
        assert!(e1.iter().any(|x| x.abs() > 1e-5), "e1 matching zone, must shift");
        assert!(e2.iter().all(|x| x.abs() < 1e-9), "e2 wrong zone, no shift");
        assert!(e3.iter().all(|x| x.abs() < 1e-9), "e3 no zone, no shift");
    }

    // Plan 309 T3.1 — SIMD vs scalar SAXPY bit-equality.
    //
    // The SAXPY is element-wise (`state[i] += alpha * dir[i]`) — no cross-lane
    // reduction — so the AVX2 path and the scalar path MUST produce bit-identical
    // results. We verify this across d=8 and d=16 on multiple seeded inputs so
    // any drift between the two backends is caught immediately.
    #[test]
    fn saxpy_simd_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2") {
                eprintln!("AVX2 not available on this host — skipping SIMD vs scalar equality check");
                return;
            }
            let dims = [8usize, 16];
            let alphas = [0.1f32, 0.3, 0.5, 0.9];
            for &d in &dims {
                for &alpha in &alphas {
                    for seed in [0xC40D_u64, 0xDEAD_BEEF, 0x1234_5678] {
                        let dir = make_unit_direction(d, seed);
                        let base: Vec<f32> = (0..d)
                            .map(|i| ((seed.wrapping_mul((i as u64) + 1)) >> 33) as f32 / (1u64 << 31) as f32 - 1.0)
                            .collect();

                        let mut simd_state = base.clone();
                        let mut scalar_state = base.clone();

                        // SAFETY: AVX2 verified above.
                        unsafe { saxpy_inplace_avx2(&mut simd_state, alpha, &dir) };
                        saxpy_inplace_scalar(&mut scalar_state, alpha, &dir);

                        assert_eq!(
                            simd_state, scalar_state,
                            "d={d} alpha={alpha} seed={seed:#x}: SIMD path drifted from scalar",
                        );
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Non-x86_64 targets have no AVX2 backend — the dispatcher is a
            // scalar passthrough, so there is nothing to compare. Sanity-check
            // the scalar path is still correct.
            let dir = make_unit_direction(8, 7);
            let mut state = vec![0.1f32; 8];
            saxpy_inplace_scalar(&mut state, 0.3, &dir);
            for i in 0..8 {
                let expected = 0.1 + 0.3 * dir[i];
                assert!((state[i] - expected).abs() < 1e-6);
            }
        }
    }
}
