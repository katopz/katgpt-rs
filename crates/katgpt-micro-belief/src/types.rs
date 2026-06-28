//! Type definitions for the `MicroRecurrentBeliefState` kernel family.
//!
//! See `katgpt-rs/.plans/276_micro_recurrent_belief_state.md` (Phase 1) and
//! `katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md`
//! for the full design rationale.
//!
//! # Summary
//!
//! A `MicroRecurrentBeliefState` is a small frozen kernel implementing one step
//! of `s_t = f(s_{t-1}, x_t)` over a fixed-size latent belief vector, applied
//! once per (entity, tick). Three recurrence families are anticipated by the
//! Mozer 2026 taxonomy:
//!
//! | Family | Update rule (one tick) | Plan slot |
//! |---|---|---|
//! | `Attractor`    | `s_t = σ(W_s·s + W_x·x + b)`    | Phase 1 (this file's `attractor.rs`) |
//! | `LatentThought`| K iters of the attractor rule   | Phase 3 (T3.1, not yet implemented) |
//! | `DeltaRule`    | leaky integrator / SSM          | Phase 2 (`leaky.rs`, standalone mirror) |
//!
//! The kernel weights are a freeze/thaw artifact — see `snapshot.rs`.

#![allow(clippy::needless_range_loop)]

use katgpt_types::simd::fast_sigmoid;

/// Recurrence family identifier.
///
/// Used for routing inside dispatch sites, for snapshot versioning, and for
/// choosing the right `step()` implementation. Matches the three slots in the
/// Mozer 2026 taxonomy that are relevant at inference time (see Research 242
/// §2.1).
///
/// `#[repr(u8)]` keeps the discriminant at 1 byte so it embeds cheaply into
/// snapshot headers and dispatch tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum RecurrenceFamily {
    /// Family A — attractor loop: `s_t = σ(W_s·s + W_x·x + b)`.
    ///
    /// Has fixed-point basins → beliefs exhibit hysteresis (stable opinions
    /// that resist noise until evidence accumulates). This is the GOAT
    /// candidate benchmarked in Plan 276 Phase 5 (G2.1).
    Attractor = 0,
    /// Family B — latent-thought loop: K iterations of Family A per tick.
    ///
    /// Opt-in "deliberation ticks" for negotiation / planning. Not yet
    /// implemented (Plan 276 Phase 3, T3.1).
    LatentThought = 1,
    /// Family C — delta-rule SSM / leaky integrator: `s_t = (1-α)·s + β·x`.
    ///
    /// Always-stable linear update. The existing `ReconstructionState::evolve_hla`
    /// shipped implementation is structurally a leaky integrator and is the
    /// battle-tested baseline. `leaky.rs` provides a standalone mirror of that
    /// math; Plan 276 Phase 2 (T2.1) will eventually make `evolve_hla` delegate
    /// to it (zero-behavior-change refactor, out of scope for Phase 1).
    DeltaRule = 2,
}

/// The core per-entity belief-state kernel trait.
///
/// Each NPC / agent holds one kernel (frozen at spawn, hot-swappable via
/// `MicroRecurrentKernelSnapshot`) plus its own belief vector `s_t`. The kernel
/// advances the belief one tick at a time via [`step`](Self::step), and bridges
/// the latent belief vector to bounded raw scalars via
/// [`project_to_scalars`](Self::project_to_scalars).
///
/// # Latent vs raw boundary (AGENTS.md)
///
/// - The belief vector `s_t` is **latent**, local to the entity, and **never
///   synced**. Syncing it would destroy emergent per-entity personality
///   divergence and waste bandwidth (32 floats × 1000 NPCs × 20 Hz ≈ 2.4 MB/s
///   of pure subjective state).
/// - The projected scalars cross the sync boundary **raw** — they drive
///   game-visible behavior and need bit-identical deterministic replay.
/// - The bridge is one-way: `s_t → scalars`. Never reconstruct `s_t` from the
///   synced scalars (5 equations, 32 unknowns — underdetermined and lossy).
///
/// # Zero-allocation contract
///
/// [`step`](Self::step) and [`project_to_scalars`](Self::project_to_scalars)
/// operate on caller-owned slices and MUST NOT allocate. The kernel's own
/// weights are allocated once at construction and read-only thereafter.
///
/// # Determinism contract (G1.1)
///
/// Given the same `(s_0, x_1..x_T)` sequence, [`step`](Self::step) MUST produce
/// bit-identical `s_T` across runs (no hidden RNG, no threading-dependent
/// reduction order). This is enforced by reusing `katgpt_types::simd::simd_dot_f32`
/// (deterministic SIMD reduction) and `katgpt_types::simd::fast_sigmoid` (exact libm
/// path, no polynomial approximation).
pub trait MicroRecurrentBeliefState: Send + Sync {
    /// Belief vector dimension (fixed at construction).
    fn dim(&self) -> usize;

    /// Advance one tick: `s_t = f(s_{t-1}, x_t)`. In-place update of `state`.
    ///
    /// # Zero-allocation
    ///
    /// Operates on the `&mut [f32]` slice directly — no `Vec` creation inside.
    /// The caller owns `state` and is responsible for its lifetime across
    /// ticks.
    ///
    /// # Panics (debug)
    ///
    /// Implementations MAY panic in debug builds if `state.len() != self.dim()`
    /// or `input.len() != self.dim()`. Release builds may use unchecked
    /// indexing on the hot path.
    fn step(&self, state: &mut [f32], input: &[f32]);

    /// Bridge: project the belief vector to K bounded scalars via
    /// `sigmoid(dot(state, direction_k))`.
    ///
    /// # Layout (R5 mitigation — generic const exprs not stable)
    ///
    /// `directions` is a **flattened** `[K * dim]` slice laid out row-major:
    /// `direction_k = directions[k*dim .. (k+1)*dim]`. The caller manages this
    /// layout; the kernel does not. `out` is a `&mut [f32]` of length K.
    ///
    /// This flattened layout avoids needing `#![feature(generic_const_exprs)]`
    /// to express `&[[f32; DIM]; K]`. The performance impact at `dim = 32` is
    /// negligible (one bounds check per row, elided by LLVM after inlining).
    ///
    /// # Zero-allocation
    ///
    /// Reads `state` / `directions`, writes `out` — no allocation.
    fn project_to_scalars(
        &self,
        state: &[f32],
        directions: &[f32],
        dim: usize,
        out: &mut [f32],
    );

    /// Family identifier (for routing, snapshot versioning).
    fn family(&self) -> RecurrenceFamily;
}

/// Default bridge implementation usable by any kernel whose family does not
/// require a custom projection.
///
/// Computes `out[k] = fast_sigmoid(dot(state, &directions[k*dim..(k+1)*dim]))`
/// for each k, reusing `katgpt_types::simd::simd_dot_f32` and `katgpt_types::simd::fast_sigmoid`.
///
/// Matches the existing `SenseModule::project` (dot + sigmoid) bridge pattern —
/// see Plan 276 T0.5. No duplication of bridge logic.
pub(crate) fn project_to_scalars_bridge(
    state: &[f32],
    directions: &[f32],
    dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(state.len(), dim, "state/dim mismatch in bridge");
    let k = out.len();
    debug_assert!(
        directions.len() >= k * dim,
        "directions slice too short: need {} have {}",
        k * dim,
        directions.len()
    );
    for k_idx in 0..k {
        let row_start = k_idx * dim;
        let dot = katgpt_types::simd::simd_dot_f32(state, &directions[row_start..row_start + dim], dim);
        out[k_idx] = fast_sigmoid(dot);
    }
}

/// Configuration for constructing a `MicroRecurrentBeliefState` kernel.
///
/// Used by the family-specific builders (`AttractorKernel::from_seed`,
/// `LeakyIntegrator::new`). Defaults target the Plan 255 plasma-tier budget:
/// `dim = 32` fits L1 and gives ~32 ns/NPC/tick on SIMD (G1.4 target).
#[derive(Clone, Debug)]
pub struct KernelConfig {
    /// Belief-vector dimension. Default `32` (Plan 255 budget, fits L1).
    pub dim: usize,
    /// Recurrence family to construct.
    pub family: RecurrenceFamily,
    /// Post-activation clamp magnitude (default `6.0`).
    ///
    /// For Family A the state is stored as `2·σ(·) − 1 ∈ (−1, 1)` (see
    /// `attractor.rs` for the range choice), so a clamp at ±6 is a no-op
    /// safety net for the rare case the post-sigmoid linear rescale is
    /// bypassed. Kept as a config field so future families with unbounded
    /// activations can clamp meaningfully.
    pub clamp: f32,
    /// Deterministic RNG seed for weight initialisation (Family A) or gate
    /// initialisation (future Family C builder). Default `42`.
    pub seed: u64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            dim: 32,
            family: RecurrenceFamily::Attractor,
            clamp: 6.0,
            seed: 42,
        }
    }
}

impl KernelConfig {
    /// Builder: set the belief-vector dimension.
    #[inline]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Builder: set the recurrence family.
    #[inline]
    pub fn with_family(mut self, family: RecurrenceFamily) -> Self {
        self.family = family;
        self
    }

    /// Builder: set the post-activation clamp magnitude.
    #[inline]
    pub fn with_clamp(mut self, clamp: f32) -> Self {
        self.clamp = clamp;
        self
    }

    /// Builder: set the deterministic RNG seed.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_plan() {
        let c = KernelConfig::default();
        assert_eq!(c.dim, 32, "Plan 276 T1.2: default dim=32");
        assert_eq!(c.family, RecurrenceFamily::Attractor);
        assert_eq!(c.clamp, 6.0, "Plan 276 T1.3: default clamp=6.0");
        assert_eq!(c.seed, 42);
    }

    #[test]
    fn config_builder_chains() {
        let c = KernelConfig::default()
            .with_dim(64)
            .with_family(RecurrenceFamily::DeltaRule)
            .with_clamp(3.0)
            .with_seed(7);
        assert_eq!(c.dim, 64);
        assert_eq!(c.family, RecurrenceFamily::DeltaRule);
        assert_eq!(c.clamp, 3.0);
        assert_eq!(c.seed, 7);
    }

    #[test]
    fn family_repr_u8() {
        // Snapshot headers depend on the exact discriminant values.
        assert_eq!(RecurrenceFamily::Attractor as u8, 0);
        assert_eq!(RecurrenceFamily::LatentThought as u8, 1);
        assert_eq!(RecurrenceFamily::DeltaRule as u8, 2);
    }

    #[test]
    fn bridge_is_monotone_in_dot() {
        // Sanity: bridge output tracks dot-product sign (G1.3 property at the
        // helper level — the full property test lives in tests.rs).
        let dim = 4usize;
        let state = [1.0f32, 0.0, 0.0, 0.0];
        // direction_0 aligned with state → large dot; direction_1 orthogonal.
        let directions: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 2];
        project_to_scalars_bridge(&state, &directions, dim, &mut out);
        assert!(out[0] > 0.5, "aligned direction should give sigmoid(1)>0.5");
        assert_eq!(out[1], fast_sigmoid(0.0), "orthogonal direction → sigmoid(0)=0.5");
    }
}
