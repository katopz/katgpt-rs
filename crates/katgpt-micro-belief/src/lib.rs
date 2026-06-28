//! `micro_belief` — implicit per-entity state tracking kernel (Plan 276).
//!
//! Small frozen recurrent kernels implementing `s_t = f(s_{t-1}, x_t)` over a
//! fixed-size latent belief vector, applied once per (entity, tick). The
//! belief vector is **latent and local** (never synced); a bridge projects it
//! to **bounded raw scalars** that cross the sync boundary.
//!
//! # Status (Phase 1)
//!
//! - ✅ [`MicroRecurrentBeliefState`] trait + [`RecurrenceFamily`] enum.
//! - ✅ [`AttractorKernel`] — Family A (the GOAT candidate).
//! - ✅ [`LeakyIntegrator`] — Family C (standalone mirror of `evolve_hla`;
//!   the refactor that wires `ReconstructionState::evolve_hla` to delegate to
//!   it is Plan 276 Phase 2 T2.1, out of scope for Phase 1).
//! - ✅ [`MicroRecurrentKernelSnapshot`] — BLAKE3-committed freeze/thaw artifact.
//! - ✅ Bridge: [`project_to_scalars`](bridge::project_to_scalars).
//! - ✅ G1.1–G1.5 GOAT-gate tests in [`tests`].
//! - ✅ G1.6 (K=1 reduces to Family A) test in [`latent_thought`].
//! - ✅ [`LatentThoughtKernel`] — Family B (Phase 3 T3.1).
//! - ✅ G2.1 coherence benchmark — Phase 5 T5.0, see [`coherence_bench`].
//! - ⏳ [`BoMSampler`] — K-hypothesis sampling (Plan 281, behind `bom_sampling`).
//! - ⏳ [`bom_arena`] — G2 arena harness (Plan 281 T2.3, behind `bom_sampling`).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? | Versioned? |
//! |---|---|---|---|
//! | `belief_vector s_t` (live state) | Latent | NO | NO (ephemeral) |
//! | Kernel weights (`W_s, W_x, b`)   | Latent | NO | YES (snapshot, BLAKE3) |
//! | Bridge direction vectors         | Latent | NO | YES (in snapshot) |
//! | Projected scalars (valence, …)   | Raw    | YES | NO (event stream) |
//! | `kernel_swap_event`              | Raw    | YES | YES (audit trail) |
//!
//! # Feature gate
//!
//! This module is gated behind the `micro_belief` Cargo feature (default-off
//! until G1 passes). The orchestrator wires it in `lib.rs` via
//! `#[cfg(feature = "micro_belief")] pub mod micro_belief;`.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/276_micro_recurrent_belief_state.md`]
//! - Research: [`katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md`]
//! - Private guide: [`riir-ai/.research/127_Implicit_Microcognition_Crowd_NPC_Guide.md`]
//! - Source paper: [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer et al., DeepMind, Jun 2026.

pub mod attractor;
#[cfg(feature = "bom_sampling")]
pub mod bom;
#[cfg(feature = "bom_sampling")]
pub mod bom_arena;
pub mod bridge;
pub mod coherence_bench;
pub mod latent_thought;
pub mod leaky;
pub mod snapshot;
pub mod types;

#[cfg(test)]
mod tests;

// ─── Uninitialized stack buffer helper ─────────────────────────────────
//
// Matches the pattern in `katgpt-rs/src/cumprodsum.rs`: an array of
// `MaybeUninit<f32>` initialized via the const block, reinterpreted as a
// `&mut [f32]` slice via `from_raw_parts_mut`. Skips the O(N) zero-init that
// `[0.0f32; N]` would impose — at N=1024 (4KB), that's ~40ns/call, which
// dominates the per-step cost at dim=32 (Issue 024).
//
// SAFETY: the caller MUST write to every accessed element before reading it.
// The `MaybeUninit<f32>` wrapper has no validity requirement, so creating
// the array is safe; only the `from_raw_parts_mut` cast is unsafe (it
// promises the memory is valid f32, which the caller upholds by writing first).

/// Allocate a fixed-size uninitialized f32 stack buffer and return it as a
/// `&mut [f32; N]` view.
///
/// # Safety
///
/// Caller MUST write to every element of the returned array before reading
/// any element. Reading uninitialized memory is UB even for `f32`.
#[inline]
pub(crate) const fn uninit_stack<const N: usize>() -> [core::mem::MaybeUninit<f32>; N] {
    [const { core::mem::MaybeUninit::uninit() }; N]
}

/// Reinterpret the first `len` elements of a `MaybeUninit<f32>` array as a
/// `&mut [f32]` slice. The returned slice borrows from `buf`.
///
/// # Safety
///
/// Caller MUST ensure elements `0..len` are written before any read.
#[inline]
pub(crate) unsafe fn assume_init_slice<const N: usize>(
    buf: &mut [core::mem::MaybeUninit<f32>; N],
    len: usize,
) -> &mut [f32] {
    // SAFETY: caller upholds that buf[0..len] is fully initialized before
    // any read. MaybeUninit<f32> has the same layout as f32.
    unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, len) }
}

// ── Public API re-exports ─────────────────────────────────────────────────
//
// Mirrors the idiom used by other katgpt-core feature modules (e.g.
// `spectral_hierarchy`, `dirichlet`, `questbench`): `pub use` the most common
// types at the module root so callers can write
// `katgpt_core::micro_belief::AttractorKernel` instead of
// `katgpt_core::micro_belief::attractor::AttractorKernel`.

pub use attractor::AttractorKernel;
#[cfg(feature = "bom_sampling")]
pub use bom::{BoMSampler, NoiseQueryConfig, SeedStrategy, dot_product_scorer};
#[cfg(feature = "bom_sampling")]
pub use bom_arena::{
    ArenaAction, ArenaEnvironment, BeliefPlanner, BoMMeanPlanner, BoMMinimaxPlanner,
    ComparisonResult, DeterministicPlanner, EnvHint, PlannerOutcome, SyntheticThreatArena,
    bom_mean_attractor, bom_minimax_attractor, bom_minimax_leaky, run_arena_comparison,
};
pub use bridge::project_to_scalars;
pub use latent_thought::LatentThoughtKernel;
pub use leaky::LeakyIntegrator;
pub use snapshot::{MicroRecurrentKernelSnapshot, SNAPSHOT_VERSION};
pub use types::{KernelConfig, MicroRecurrentBeliefState, RecurrenceFamily};
