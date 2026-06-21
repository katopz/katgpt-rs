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
//! [`simd_fused_scale_acc`](crate::simd::simd_fused_scale_acc) so NEON/AVX2/FMA
//! is used when available, with a scalar fallback otherwise.

use crate::personality_composition::sigmoid::sigmoid;
use crate::personality_composition::trait_def::LayerDirectionSource;
use crate::personality_composition::types::PersonalityConfig;
use crate::simd::simd_fused_scale_acc;

/// The personality-weighted composition kernel.
///
/// Generic over the number of layers `N` and the direction dimension `D`.
/// The host owns the layer sources and the reward signal; this kernel owns
/// the personality weights `w` and the per-layer reward EMA `r_expected`.
///
/// # Const-generic budget
///
/// Per AGENTS.md, `N` is pinned to `{1, 4, 7, 9}` via type aliases in
/// [`types`](crate::personality_composition::types) to keep monomorphisation
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
        // loop — LLVM elides it into a memset.
        for x in out[..D].iter_mut() {
            *x = 0.0;
        }

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
            let mut delta = 0.0f32;
            for &d in d_recent.iter().take(D) {
                delta += d;
            }
            let inc = alpha * surprise * delta;

            self.w[i] = (self.w[i] + inc).clamp(-w_max, w_max);

            // EMA update of expected reward.
            self.r_expected[i] = ema_decay * self.r_expected[i] + ema_innov * r_observed;
        }
    }

    /// Read-only access to the personality weights `w`.
    ///
    /// For snapshot integration ([`PersonalitySnapshot`](crate::personality_composition::PersonalitySnapshot)::from_composition)
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

    /// Reset `r_expected` to zero (clears reward memory).
    ///
    /// Used after a hot-swap if the host wants the new personality to start
    /// with a fresh surprise baseline. Typically not called —
    /// [`restore_w`](Self::restore_w) preserves `r_expected` for continuity.
    #[inline]
    pub fn reset_r_expected(&mut self) {
        self.r_expected = [0.0; N];
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
}
