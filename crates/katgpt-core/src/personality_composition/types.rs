//! Type definitions for the personality-weighted composition kernel.
//!
//! See `katgpt-rs/.plans/297_personality_weighted_composition.md` (Phase 1 T1.4)
//! and `katgpt-rs/.research/276_Personality_Weighted_Latent_Layer_Composition.md`
//! for the full design rationale.

/// Configuration for a [`PersonalityWeightedComposition`](crate::personality_composition::PersonalityWeightedComposition).
///
/// All fields are host-configured constants. The kernel holds a copy (16 bytes)
/// so it doesn't need an indirection on the hot path.
///
/// # Defaults
///
/// Per Plan 297 T1.4:
/// - `tau = 1.0` — moderate personality sharpness
/// - `alpha = 0.01` — slow plasticity (100 ticks to saturate under unit surprise)
/// - `w_max = 5.0` — `sigmoid(±5/1) ∈ {0.0067, 0.9933}` (near-binary extremes)
/// - `ema_decay = 0.95` — ~20-tick effective window on reward expectation
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PersonalityConfig {
    /// Personality-sharpness temperature `τ`.
    ///
    /// - `τ → ∞`: all weights contribute 0.5 (no personality — see G1 test).
    /// - `τ → 0`: weights become binary (extreme personality).
    /// - `τ = 1.0`: standard logistic sharpness.
    ///
    /// MUST be positive. The kernel divides `w_i / tau`; a zero or negative
    /// `tau` produces NaN/Inf.
    pub tau: f32,

    /// Plasticity (drift learning rate) `α ∈ (0, 1)`.
    ///
    /// Controls how fast `w` moves under reward surprise. Higher = faster
    /// adaptation but less stable; lower = slower but more robust.
    pub alpha: f32,

    /// Clamp bound on `w`. Weights are clamped to `[-w_max, +w_max]` after
    /// each drift step to prevent runaway.
    ///
    /// At `tau = 1.0`, `w_max = 5.0` gives `sigmoid(±5) ∈ {0.0067, 0.9933}`,
    /// which is near-binary but not exactly 0/1 (preserves a sliver of
    /// uncertainty for numerical stability).
    pub w_max: f32,

    /// EMA decay for `r_expected`. `r_expected_i ← decay · r_expected_i +
    /// (1 - decay) · r_observed`. Higher = longer memory.
    pub ema_decay: f32,
}

impl Default for PersonalityConfig {
    #[inline]
    fn default() -> Self {
        Self {
            tau: 1.0,
            alpha: 0.01,
            w_max: 5.0,
            ema_decay: 0.95,
        }
    }
}

impl PersonalityConfig {
    /// Validate config fields. Returns `false` if any field would produce
    /// NaN/Inf in the kernel (e.g. `tau <= 0`, `alpha < 0`, `w_max < 0`).
    ///
    /// The kernel does NOT call this on the hot path — callers should validate
    /// once at construction.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.tau > 0.0
            && self.tau.is_finite()
            && self.alpha >= 0.0
            && self.alpha.is_finite()
            && self.w_max > 0.0
            && self.w_max.is_finite()
            && self.ema_decay >= 0.0
            && self.ema_decay <= 1.0
    }
}

/// An opaque archetype label that seeds initial `w` and tags snapshots.
///
/// The kernel does NOT interpret this label — it's an opaque 16-byte blob
/// that the host uses to disambiguate "predator" vs "prey" vs "NPC" vs "robot"
/// personalities. It flows into [`PersonalitySnapshot`](crate::personality_composition::PersonalitySnapshot)
/// as part of the BLAKE3 commitment so two entities with identical weights but
/// different archetypes produce different hashes.
///
/// # Construction
///
/// - [`new`](Self::new) — from a raw `[u8; 16]`
/// - [`from_str`](Self::from_str) — from a `&str`, BLAKE3-hashed into 16 bytes
/// - [`empty`](Self::empty) — all-zeros (the "unlabelled" archetype)
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct ArchetypeLabel(pub [u8; 16]);

impl ArchetypeLabel {
    /// Construct from a raw 16-byte label.
    #[inline]
    pub fn new(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Construct from a string by BLAKE3-hashing it into 16 bytes.
    ///
    /// This gives a stable, deterministic mapping from archetype name to
    /// label — two entities with the same archetype name get the same label.
    #[allow(clippy::should_implement_trait)] // infallible hash constructor, not FromStr (which requires Result)
    pub fn from_str(s: &str) -> Self {
        let hash = blake3::hash(s.as_bytes());
        Self(
            hash.as_bytes()[..16]
                .try_into()
                .expect("blake3 is 32 bytes"),
        )
    }

    /// The "unlabelled" archetype (all-zeros). Use when the host does not
    /// distinguish archetypes.
    #[inline]
    pub fn empty() -> Self {
        Self([0u8; 16])
    }

    /// Raw bytes view (for hashing / serialization).
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl Default for ArchetypeLabel {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_plan_297() {
        let c = PersonalityConfig::default();
        assert_eq!(c.tau, 1.0);
        assert_eq!(c.alpha, 0.01);
        assert_eq!(c.w_max, 5.0);
        assert_eq!(c.ema_decay, 0.95);
    }

    #[test]
    fn default_config_is_valid() {
        assert!(PersonalityConfig::default().is_valid());
    }

    #[test]
    fn invalid_configs_rejected() {
        let mut c = PersonalityConfig {
            tau: 0.0,
            ..Default::default()
        };
        assert!(!c.is_valid());
        c.tau = -1.0;
        assert!(!c.is_valid());
        c.tau = f32::NAN;
        assert!(!c.is_valid());
    }

    #[test]
    fn archetype_from_str_is_deterministic() {
        let a1 = ArchetypeLabel::from_str("predator");
        let a2 = ArchetypeLabel::from_str("predator");
        let b = ArchetypeLabel::from_str("prey");
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn archetype_empty_is_all_zeros() {
        assert_eq!(ArchetypeLabel::empty(), ArchetypeLabel::new([0u8; 16]));
    }
}
