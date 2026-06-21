//! `personality_composition` — Personality-Weighted Latent Layer Composition
//! (Plan 297, Research 276).
//!
//! A generic, modelless, MIT-licensed primitive: compose `N` latent direction
//! vectors `d_i ∈ ℝ^D` into a single behavior vector via a personality weight
//! vector `w ∈ ℝ^N` with sigmoid gating, and update `w` via an EMA on reward
//! prediction error.
//!
//! # The math
//!
//! ```text
//! behavior = Σ_i sigmoid(w_i / τ) · belief_confidence_i · d_i
//! ```
//!
//! Drift rule:
//!
//! ```text
//! surprise_i = R_observed - R_expected_i
//! Δw_i = α · surprise_i · d_recent_i
//! w_i ← clamp(w_i + Δw_i, -w_max, +w_max)
//! R_expected_i ← ema_decay · R_expected_i + (1 - ema_decay) · R_observed
//! ```
//!
//! # Why sigmoid, not softmax (AGENTS.md)
//!
//! Sigmoid is mandated for projections onto learned direction vectors. Softmax
//! would destroy the "negative weight = resistance" semantics — softmax always
//! assigns non-trivial probability to every layer. Sigmoid allows a layer to
//! contribute ~0 (the agent ignores it) or ~1 (the agent embodies it), with
//! signed resistance.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - `w` (personality weights) are **latent**, local to the entity, never
//!   synced. Syncing would destroy per-entity personality divergence.
//! - The composed behavior vector is **latent**, consumed downstream by the
//!   host's projection bridges.
//! - The [`PersonalitySnapshot`] commitment (BLAKE3 hash + version) IS synced
//!   as an audit event when a hot-swap occurs.
//!
//! # Entity-agnostic
//!
//! No game terms (no "faction", no "law", no "family"). The kernel is `N × D`
//! linear algebra + sigmoid + EMA. Applies to NPC, player, predator, prey,
//! robot, recommender user.
//!
//! # Feature gate
//!
//! Gated behind the `personality_composition` Cargo feature (default-off until
//! GOAT G4/G5 pass). See `katgpt-rs/.plans/297_personality_weighted_composition.md`.
//!
//! # Module layout
//!
//! - [`types`] — `PersonalityConfig`, `ArchetypeLabel`.
//! - [`sigmoid`] — numerically stable branching sigmoid.
//! - [`trait_def`] — `LayerDirectionSource` (file named to avoid the `trait`
//!   keyword, which Rust reserves as a module path component).
//! - [`kernel`] — `PersonalityWeightedComposition` (compose + drift + snapshot
//!   accessors).
//! - [`snapshot`] — `PersonalitySnapshot` with BLAKE3 commitment.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/297_personality_weighted_composition.md`]
//! - Research: [`katgpt-rs/.research/276_Personality_Weighted_Latent_Layer_Composition.md`]
//! - Private guide: [`riir-ai/.research/146_Entity_Cognition_Stack_Guide.md`]
//! - Companion plan (riir-ai runtime wiring): [`riir-ai/.plans/327_entity_cognition_stack_runtime.md`]

pub mod kernel;
pub mod sigmoid;
pub mod snapshot;
pub mod trait_def;
pub mod types;

pub use kernel::PersonalityWeightedComposition;
pub use sigmoid::{sigmoid, sigmoid_into};
pub use snapshot::PersonalitySnapshot;
pub use trait_def::LayerDirectionSource;
pub use types::{ArchetypeLabel, PersonalityConfig};

// ─── Pinned const-generic aliases (compile-time budget) ────────────────────
//
// Per AGENTS.md hard rule: pin N to {1, 4, 7, 9} via type aliases to keep
// monomorphisation bounded. The production Entity Cognition Stack case is
// N=9, D=32 (riir-ai Research 146). Hosts pick the alias that matches their
// layer count; new N values require adding an alias here.

/// Single-layer composition (degenerate: pure sigmoid gate on one direction).
pub type SingleLayerComposition<const D: usize> = PersonalityWeightedComposition<1, D>;

/// 4-layer composition.
pub type QuadLayerComposition<const D: usize> = PersonalityWeightedComposition<4, D>;

/// 7-layer composition.
pub type HeptaLayerComposition<const D: usize> = PersonalityWeightedComposition<7, D>;

/// 9-layer composition (the production Entity Cognition Stack case).
///
/// Fixed at D=32 to match the HLA belief-vector dimension used across
/// katgpt-rs / riir-ai (Research 146 / Research 242).
pub type EntityCognitionComposition = PersonalityWeightedComposition<9, 32>;

#[cfg(test)]
mod tests;
