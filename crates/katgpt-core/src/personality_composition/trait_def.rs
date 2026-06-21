//! The [`LayerDirectionSource`] trait (Plan 297 T1.6).
//!
//! The host (game, robot, recommender) implements this trait per layer per
//! entity. The composition kernel calls [`direction`](LayerDirectionSource::direction)
//! once per tick per layer to obtain the layer's contribution direction, and
//! [`recent_direction`](LayerDirectionSource::recent_direction) during drift
//! to obtain the EMA-smoothed recent direction.

/// A host-supplied source of a latent direction vector for one layer.
///
/// The host implements this per layer per entity. The composition kernel calls
/// [`direction`](Self::direction) once per tick per layer.
///
/// # Entity-agnostic
///
/// The trait carries no game semantics. A wolf's KIN(pack) layer and a
/// shopkeeper's COMPANIONS(party) layer both implement this trait with
/// different internals. A recommender's "friend recommendation" layer also
/// implements it. The kernel is the same in all cases.
///
/// # Zero-allocation contract
///
/// Implementations of [`direction`](Self::direction) MUST be zero-allocation
/// on the hot path. The caller passes a scratch buffer; the implementation
/// writes its direction into it and returns a reference. Typically the
/// implementation writes into the scratch buffer directly, but it MAY return
/// a reference to an internal buffer if the direction is precomputed (the
/// scratch is then unused).
///
/// # `recent_direction` for drift
///
/// [`recent_direction`](Self::recent_direction) returns an EMA-smoothed recent
/// direction vector, used by
/// [`PersonalityWeightedComposition::drift`](crate::personality_composition::PersonalityWeightedComposition::drift)
/// to assign credit. The host maintains this EMA externally (typically a
/// rolling average updated each tick). The default returns an empty slice,
/// which disables drift contribution from that layer — override to enable.
pub trait LayerDirectionSource: Send + Sync {
    /// Returns the direction vector `d ∈ ℝ^D` for this layer at this tick.
    ///
    /// The implementation writes into `scratch` (length `D`) and returns a
    /// reference to the written region. The returned slice MAY be the scratch
    /// buffer itself, or an internal buffer if the direction is precomputed.
    ///
    /// # Lifetime
    ///
    /// The returned reference is tied to the lifetime of `scratch` (so impls
    /// can return `&scratch[..D]`) OR to `&self` (so impls can return a
    /// reference to an internal precomputed buffer). The `+'a` bound unifies
    /// both.
    ///
    /// # Zero-allocation
    ///
    /// MUST NOT allocate. Reuse the scratch buffer or an internal fixed-size
    /// buffer.
    fn direction<'a>(&self, scratch: &'a mut [f32]) -> &'a [f32];

    /// Returns the EMA-smoothed recent direction (for drift computation).
    ///
    /// This is the host-maintained rolling average of recent `direction()`
    /// outputs. The drift rule uses it to decide which way to push `w_i`.
    ///
    /// # Default
    ///
    /// Returns `&[]` (empty) — the layer does not maintain a recent EMA, so
    /// its drift contribution is zero (the kernel skips the `w_i` update for
    /// this layer but still updates `r_expected_i`). Override to return a
    /// length-`D` slice to enable drift for this layer.
    fn recent_direction(&self) -> &[f32] {
        &[]
    }

    /// Belief confidence in `(0, 1]` for this layer at this tick.
    ///
    /// For remote layers acting on stale think-brain belief (e.g. KIN/COMPANIONS
    /// when the target is not visible), the host decays this via
    /// `sigmoid(-λ · Δtick_since_last_observation)`. For plasma-tier immediate
    /// layers (SENSE/SITUATION), returns 1.0.
    ///
    /// The kernel multiplies `direction() * belief_confidence()` before
    /// applying the personality weight sigmoid.
    ///
    /// # Default
    ///
    /// Returns `1.0` — full confidence (plasma-tier immediate layer).
    fn belief_confidence(&self) -> f32 {
        1.0
    }
}
