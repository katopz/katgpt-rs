//! The composition kernel — [`PersonalityWeightedComposition`] (Plan 297 T1.7,
//! Phase 2 T2.1–T2.3).
//!
//! This is the heart of the personality-weighted layer composition primitive:
//! a fixed-size `N × D` linear-algebra kernel with sigmoid gating, optional
//! belief-gating, and a reward-surprise drift rule. Zero-allocation on the
//! hot path.
//!
//! # Hot path
//!
//! [`compose_into`](PersonalityWeightedComposition::compose_into) is the
//! per-tick hot path. It costs `O(N · D)` multiplies — for `N=9, D=32` that's
//! 288 FMAs, trivially SIMD-able. The inner loop delegates to
//! [`simd_fused_scale_acc`] so NEON/AVX2/FMA
//! is used when available, with a scalar fallback otherwise.

use crate::sigmoid::sigmoid;
use crate::trait_def::LayerDirectionSource;
use crate::types::PersonalityConfig;
use katgpt_types::simd::{dot_f32_ordered, simd_fused_scale_acc};

/// Unit vector used by [`PersonalityWeightedComposition::drift`] to compute
/// `Σ_j d_recent[j]` as a dot-product (one NEON/AVX2 reduction instead of a
/// scalar add loop). Sized for the maximum D supported by the type aliases in
/// [`crate`] (production case is D=32; 64 gives headroom).
const ONES_MAX_D: usize = 64;
const ONES: [f32; ONES_MAX_D] = [1.0f32; ONES_MAX_D];

/// The personality-weighted composition kernel.
///
/// Generic over the number of layers `N` and the direction dimension `D`.
/// The host owns the layer sources and the reward signal; this kernel owns
/// the personality weights `w` and the per-layer reward EMA `r_expected`.
///
/// # Const-generic budget
///
/// Per AGENTS.md, `N` is pinned to `{1, 4, 7, 9}` via type aliases in
/// [`types`](crate::types) to keep monomorphisation
/// bounded. The production Entity Cognition Stack case is `N=9, D=32`.
///
/// # Layout
///
/// The struct is `N · 4 + sizeof::<PersonalityConfig>() + N · 4` bytes:
/// `w` (N f32), `config` (4 f32 = 16 B), `r_expected` (N f32). At `N=9`,
/// that's `9·4 + 16 + 9·4 = 88` bytes — fits comfortably in two cache lines.
pub struct PersonalityWeightedComposition<const N: usize, const D: usize> {
    /// Personality weights, one per layer. Signed; clamped to `[-w_max, +w_max]`.
    pub w: [f32; N],

    /// Kernel configuration (`tau`, `alpha`, `w_max`, `ema_decay`).
    config: PersonalityConfig,

    /// Per-layer EMA of observed reward. Drives the surprise signal in
    /// [`drift`](Self::drift).
    r_expected: [f32; N],
}

// SAFETY: `PersonalityWeightedComposition` contains only `f32` arrays and a
// `Copy` config — no interior mutability, no cell, no raw pointer. Safe to
// share across threads.
unsafe impl<const N: usize, const D: usize> Send for PersonalityWeightedComposition<N, D> {}
unsafe impl<const N: usize, const D: usize> Sync for PersonalityWeightedComposition<N, D> {}

impl<const N: usize, const D: usize> PersonalityWeightedComposition<N, D> {
    /// Construct a kernel with the given config and initial weights.
    ///
    /// `initial_w` is typically derived from the entity's archetype by the
    /// host (e.g. predator archetypes start with high `w_predator` and low
    /// `w_prey`). The kernel itself does NOT interpret `initial_w` — any
    /// values outside `[-w_max, +w_max]` will be clamped on the first
    /// [`drift`](Self::drift) call.
    ///
    /// `r_expected` is initialized to zero (no prior reward history).
    #[inline]
    pub fn new(config: PersonalityConfig, initial_w: [f32; N]) -> Self {
        Self {
            w: initial_w,
            config,
            r_expected: [0.0; N],
        }
    }

    /// Construct a kernel with default config and all-zero weights.
    ///
    /// Equivalent to `PersonalityWeightedComposition::new(PersonalityConfig::default(),
    /// [0.0; N])`. The zero-weight initial state gives "uniform 0.5
    /// personality" at `tau = 1.0` (see G1 test).
    #[inline]
    pub fn uniform() -> Self {
        Self::new(PersonalityConfig::default(), [0.0; N])
    }

    /// Read-only access to the config.
    #[inline]
    pub const fn config(&self) -> &PersonalityConfig {
        &self.config
    }

    /// Read-only access to the per-layer reward EMA.
    ///
    /// Useful for debugging / introspection: shows what reward level each
    /// layer currently "expects".
    #[inline]
    pub const fn r_expected(&self) -> &[f32; N] {
        &self.r_expected
    }

    // ─── Phase 1 T1.7: compose_into ────────────────────────────────────────

    /// Compose `N` layer direction vectors into a single behavior vector.
    ///
    /// Each layer's contribution is:
    ///
    /// ```text
    /// sigmoid(w_i / tau) · belief_confidence_i · d_i
    /// ```
    ///
    /// Writes into `out` (length `D`). Returns `&mut out` for chaining.
    ///
    /// # Zero-allocation (G5)
    ///
    /// The caller owns `scratch` and `out`. The kernel does not allocate.
    /// `scratch` is passed through to each layer's
    /// [`direction`](LayerDirectionSource::direction) call; it MUST be at
    /// least `D` elements. Layers MAY write into it or return an internal
    /// buffer — either way, no allocation happens inside the kernel.
    ///
    /// # SIMD
    ///
    /// The inner `out[j] += weight · d[j]` loop delegates to
    /// [`simd_fused_scale_acc`], which uses NEON/AVX2+FMA when available
    /// and falls back to scalar `f32::mul_add` otherwise. At `D=32`, this
    /// is 4 NEON iterations or 4 AVX2 iterations.
    ///
    /// # Panics (debug)
    ///
    /// In debug builds, panics if `out.len() != D` or `scratch.len() < D`.
    pub fn compose_into<'a>(
        &self,
        layers: &[&dyn LayerDirectionSource; N],
        scratch: &mut [f32],
        out: &'a mut [f32],
    ) -> &'a mut [f32] {
        debug_assert_eq!(out.len(), D, "out must be exactly D={D} elements");
        debug_assert!(
            scratch.len() >= D,
            "scratch must be at least D={D} elements, got {}",
            scratch.len()
        );

        // Zero the output. This is the only O(D) work outside the per-layer
        // loop — `.fill(0.0)` lowers to memset unconditionally.
        out[..D].fill(0.0);

        for (i, layer) in layers.iter().enumerate() {
            let d = layer.direction(scratch);
            debug_assert_eq!(
                d.len(),
                D,
                "layer {i} returned direction of length {}, expected D={D}",
                d.len()
            );

            // Per-layer gate: sigmoid(w_i / tau) · belief_confidence_i.
            // This is the personality expression — clamped to [0, belief_confidence].
            let gate = sigmoid(self.w[i] / self.config.tau) * layer.belief_confidence();

            // Inner loop: out[j] += gate · d[j]. Delegated to the SIMD fused
            // scale-accumulate kernel so NEON/AVX2+FMA is used when available.
            // `#[inline]` on `simd_fused_scale_acc` lets LLVM keep everything
            // in registers — no call overhead.
            simd_fused_scale_acc(out, d, gate, D);
        }

        out
    }

    // ─── Phase 2 T2.1–T2.3: drift + snapshot helpers ──────────────────────

    /// Update personality weights from observed reward.
    ///
    /// For each layer `i`:
    ///
    /// ```text
    /// surprise_i = r_observed - r_expected_i
    /// Δw_i       = alpha · surprise_i · Σ_j d_recent_i[j]
    /// w_i        ← clamp(w_i + Δw_i, -w_max, +w_max)
    /// r_expected_i ← ema_decay · r_expected_i + (1 - ema_decay) · r_observed
    /// ```
    ///
    /// Layers whose [`recent_direction`](LayerDirectionSource::recent_direction)
    /// returns an empty slice (the default) get zero `Δw_i` — they do not
    /// participate in drift, but their `r_expected_i` still tracks the reward
    /// signal. This lets a layer "listen" to rewards without influencing
    /// weights until the host starts maintaining its recent direction EMA.
    ///
    /// # Sign convention
    ///
    /// - Positive surprise (`r_observed > r_expected`) with positive
    ///   `d_recent_i[j]` → `w_i` increases (reinforces layer i).
    /// - Negative surprise with positive `d_recent_i[j]` → `w_i` decreases
    ///   (penalizes layer i).
    /// - The sign of `d_recent_i[j]` modulates which direction the weight
    ///   moves: a layer whose recent direction was "negative along axis j"
    ///   gets pushed the opposite way.
    ///
    /// # Zero-allocation
    ///
    /// No allocation. Operates in-place on `self.w` and `self.r_expected`.
    pub fn drift(&mut self, layers: &[&dyn LayerDirectionSource; N], r_observed: f32) {
        let alpha = self.config.alpha;
        let w_max = self.config.w_max;
        let ema_decay = self.config.ema_decay;
        let ema_innov = 1.0 - ema_decay;

        for (i, layer) in layers.iter().enumerate() {
            let d_recent = layer.recent_direction();
            let surprise = r_observed - self.r_expected[i];

            // Δw_i = alpha · surprise · Σ_j d_recent_i[j]
            // Issue 940 (PE-1): `w` feeds the synced BLAKE3 commitment
            // (`PersonalitySnapshot` — the commitment IS synced as an audit
            // event on hot-swap), so this fold is pinned to the sequential
            // order — `dot_f32_ordered` against the ONES unit vector (the
            // mul-by-1.0 is an exact f32 identity, so this IS Σ d_recent in
            // ascending index order), never a platform-lane SIMD kernel. The
            // host contract is `recent_direction()` returns either `&[]`
            // (layer opts out of drift → delta = 0) or a length-`D` slice; we
            // bound the fold to the actual slice length to stay safe against
            // hosts that return a short EMA buffer during warmup.
            let drift_len = d_recent.len().min(D);
            let delta = if drift_len == 0 {
                0.0
            } else {
                debug_assert!(
                    D <= ONES_MAX_D,
                    "D={D} exceeds ONES_MAX_D={ONES_MAX_D}; bump the constant"
                );
                dot_f32_ordered(&d_recent[..drift_len], &ONES[..drift_len])
            };
            let inc = alpha * surprise * delta;

            self.w[i] = (self.w[i] + inc).clamp(-w_max, w_max);

            // EMA update of expected reward.
            self.r_expected[i] = ema_decay * self.r_expected[i] + ema_innov * r_observed;
        }
    }

    /// Read-only access to the personality weights `w`.
    ///
    /// For snapshot integration ([`PersonalitySnapshot`](crate::PersonalitySnapshot)::from_composition)
    /// and host-side introspection (e.g. checking if `w_COMPANIONS` has risen
    /// above `τ_tame`).
    #[inline]
    pub fn w_snapshot(&self) -> &[f32; N] {
        &self.w
    }

    /// Restore personality weights from a snapshot.
    ///
    /// Overwrites `w` without touching `r_expected` or `config`. Used by the
    /// hot-swap layer to atomically thaw a personality — the reward EMA stays
    /// continuous across the swap so drift doesn't jump.
    ///
    /// Does NOT validate `w` against `w_max` — snapshots are trusted. A
    /// corrupted snapshot with out-of-range weights will produce out-of-range
    /// sigmoid inputs, but the kernel will still run (sigmoid saturates).
    #[inline]
    pub fn restore_w(&mut self, w: [f32; N]) {
        self.w = w;
    }

    /// Restore `r_expected` from a frozen snapshot.
    ///
    /// Counterpart to [`restore_w`](Self::restore_w) for the reward EMA.
    /// Used by the migration bundle thaw path (Issue 456 Phase 2) to fully
    /// reconstruct the composition from frozen bytes without losing reward
    /// history. Not used by the hot-swap layer (which keeps `r_expected`
    /// continuous across the swap).
    #[inline]
    pub fn restore_r_expected(&mut self, r_expected: [f32; N]) {
        self.r_expected = r_expected;
    }

    /// Reset `r_expected` to zero (clears reward memory).
    ///
    /// Used after a hot-swap if the host wants the new personality to start
    /// with a fresh surprise baseline. Typically not called —
    /// [`restore_w`](Self::restore_w) preserves `r_expected` for continuity.
    #[inline]
    pub fn reset_r_expected(&mut self) {
        self.r_expected = [0.0; N];
    }

    /// Serialize the composition to a flat byte buffer.
    ///
    /// Wire format (all little-endian f32):
    /// ```text
    /// w (N f32) || config.tau || config.alpha || config.w_max || config.ema_decay || r_expected (N f32)
    /// ```
    ///
    /// Total size: `8*N + 16` bytes (e.g. 88 bytes for N=9). The const
    /// generic `D` (direction dim) is NOT serialized — it's an output-sizing
    /// constant, not runtime state. The format is stable across `D` values.
    ///
    /// This is the freeze format for NPC migration (Issue 456 Phase 2) and
    /// cold-tier persistence. Deterministic: same state → same bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let total = N * 4 + 16 + N * 4;
        let mut buf = Vec::with_capacity(total);
        for &wi in &self.w {
            buf.extend_from_slice(&wi.to_le_bytes());
        }
        buf.extend_from_slice(&self.config.tau.to_le_bytes());
        buf.extend_from_slice(&self.config.alpha.to_le_bytes());
        buf.extend_from_slice(&self.config.w_max.to_le_bytes());
        buf.extend_from_slice(&self.config.ema_decay.to_le_bytes());
        for &r in &self.r_expected {
            buf.extend_from_slice(&r.to_le_bytes());
        }
        debug_assert_eq!(buf.len(), total);
        buf
    }

    /// Deserialize from a flat byte buffer produced by [`to_bytes`](Self::to_bytes).
    ///
    /// Returns `None` if the buffer is the wrong length. This is a structural
    /// check only — no cryptographic integrity. Callers needing tamper
    /// detection wrap the bytes in a `MerkleFrozenEnvelope` (the migration
    /// bundle path does this at the outer layer).
    ///
    /// Reconstructs the config from the buffer — the caller does NOT need to
    /// pass it in. This ensures the thawed composition matches the frozen one
    /// bit-for-bit (no config drift from caller defaults).
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        let expected_len = N * 4 + 16 + N * 4;
        if buf.len() != expected_len {
            return None;
        }
        let mut w = [0.0f32; N];
        let mut r_expected = [0.0f32; N];
        let read_f32 = |offset: &mut usize| -> f32 {
            let mut tmp = [0u8; 4];
            tmp.copy_from_slice(&buf[*offset..*offset + 4]);
            *offset += 4;
            f32::from_le_bytes(tmp)
        };
        let mut offset = 0;
        for slot in &mut w {
            *slot = read_f32(&mut offset);
        }
        let config = PersonalityConfig {
            tau: read_f32(&mut offset),
            alpha: read_f32(&mut offset),
            w_max: read_f32(&mut offset),
            ema_decay: read_f32(&mut offset),
        };
        for slot in &mut r_expected {
            *slot = read_f32(&mut offset);
        }
        debug_assert_eq!(offset, expected_len);
        Some(Self {
            w,
            config,
            r_expected,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // ─── Test layer impl ──────────────────────────────────────────────────

    /// A minimal `LayerDirectionSource` for testing: holds a fixed direction
    /// vector and an optional recent direction + belief confidence.
    pub(crate) struct StaticLayer {
        direction: [f32; 32],
        recent: [f32; 32],
        has_recent: bool,
        confidence: f32,
    }

    impl StaticLayer {
        pub(crate) fn new(direction: [f32; 32]) -> Self {
            Self {
                direction,
                recent: [0.0; 32],
                has_recent: false,
                confidence: 1.0,
            }
        }

        pub(crate) fn with_recent(mut self, recent: [f32; 32]) -> Self {
            self.recent = recent;
            self.has_recent = true;
            self
        }

        pub(crate) fn with_confidence(mut self, c: f32) -> Self {
            self.confidence = c;
            self
        }
    }

    impl LayerDirectionSource for StaticLayer {
        fn direction<'a>(&self, scratch: &'a mut [f32]) -> &'a [f32] {
            scratch[..self.direction.len()].copy_from_slice(&self.direction);
            &scratch[..self.direction.len()]
        }

        fn recent_direction(&self) -> &[f32] {
            if self.has_recent { &self.recent } else { &[] }
        }

        #[inline]
        fn belief_confidence(&self) -> f32 {
            self.confidence
        }
    }

    #[test]
    fn construct_and_access_fields() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        assert_eq!(k.w_snapshot(), &[0.1, -0.2, 0.3]);
        assert_eq!(k.config().tau, 1.0);
        assert_eq!(k.r_expected(), &[0.0; 3]);
    }

    #[test]
    fn to_bytes_from_bytes_roundtrip_preserves_all_fields() {
        let config = PersonalityConfig {
            tau: 0.75,
            alpha: 0.02,
            w_max: 4.0,
            ema_decay: 0.9,
        };
        let w = [0.5, -1.2, 2.3, 0.0, -0.7, 1.1, -2.0, 0.9, 0.3];
        let mut k = PersonalityWeightedComposition::<9, 32>::new(config, w);
        // Set non-zero r_expected to verify it round-trips.
        k.restore_r_expected([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);

        let bytes = k.to_bytes();
        assert_eq!(bytes.len(), 9 * 4 + 16 + 9 * 4, "wire size must be 8N+16");

        let thawed = PersonalityWeightedComposition::<9, 32>::from_bytes(&bytes)
            .expect("from_bytes must succeed on correct-length input");
        assert_eq!(thawed.w, k.w, "w must round-trip");
        assert_eq!(thawed.config(), &config, "config must round-trip");
        assert_eq!(
            thawed.r_expected(),
            k.r_expected(),
            "r_expected must round-trip"
        );
    }

    #[test]
    fn from_bytes_rejects_wrong_length() {
        let k = PersonalityWeightedComposition::<3, 32>::new(
            PersonalityConfig::default(),
            [0.1, -0.2, 0.3],
        );
        let bytes = k.to_bytes();
        // Truncate by one byte.
        assert!(
            PersonalityWeightedComposition::<3, 32>::from_bytes(&bytes[..bytes.len() - 1])
                .is_none()
        );
        // Expand by one byte.
        let mut expanded = bytes.clone();
        expanded.push(0);
        assert!(PersonalityWeightedComposition::<3, 32>::from_bytes(&expanded).is_none());
    }

    #[test]
    fn to_bytes_is_deterministic() {
        let config = PersonalityConfig {
            tau: 1.5,
            alpha: 0.03,
            w_max: 6.0,
            ema_decay: 0.88,
        };
        let w = [0.1, 0.2, 0.3];
        let k1 = PersonalityWeightedComposition::<3, 32>::new(config, w);
        let k2 = PersonalityWeightedComposition::<3, 32>::new(config, w);
        assert_eq!(
            k1.to_bytes(),
            k2.to_bytes(),
            "same state must produce identical bytes"
        );
    }
}
