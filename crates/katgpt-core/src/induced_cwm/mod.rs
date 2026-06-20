//! `induced_cwm` — Code World Model kernel primitive (Plan 296, Research 275).
//!
//! Open half of the CWM Super-GOAT distilled from arxiv 2510.04542
//! (Lehrach et al., DeepMind Oct 2025). The paper proposes using an LLM as
//! an induction engine to translate natural-language rules and observed
//! trajectories into a verifiable, executable Python forward model. MCTS /
//! ISMCTS then runs on the induced model — the LLM is never on the hot path.
//!
//! This module ships the **generic, IP-free half** of that idea:
//!
//! - [`InducedCwmKernel`] — marker trait over [`GameState`] for forward models
//!   that are verifiable (pass auto-generated transition tests), committable
//!   (BLAKE3), and hot-swappable (Phase 4 slot).
//! - [`CwmCommitment`] — the BLAKE3-committed induction-event artifact.
//! - [`BeliefInferenceFn`] — stochastic hidden-state sampler for IIGs
//!   (paired with the kernel for ISMCTS in Phase 2).
//! - [`TransitionUnitTest`] / [`verify_transition`] /
//!   [`make_transition_tests_from_trajectory`] — auto-generated regression
//!   tests for the G1 verifiability gate.
//!
//! # What stays OUT of katgpt-rs
//!
//! LLM synthesis, prompting, refinement tree, NPC integration, game-specific
//! code, chain bridging. Those are private → riir-ai Plan 326.
//!
//! # Hot-path vs cold-path
//!
//! | Layer | What | Tier |
//! |-------|------|------|
//! | Induction event | LLM call → kernel impl + canonical bytes | Cold (offline/background) |
//! | Commitment | BLAKE3 over canonical bytes | Cold (once per induction) |
//! | Hot swap | Atomic `Arc` swap of `(kernel, commitment)` | Cold (event) |
//! | Game tick | `kernel.advance(state, &action, pid)` | **Hot** (20Hz) |
//! | Commitment verify | `blake3 == kernel.commitment()` | Cold (audit cadence) |
//!
//! The hot path is pure Rust — no LLM, no JSON, no Python. Whatever induced
//! the impl is the integrator's problem.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | Kernel transition fn (`advance`) | Raw | YES (deterministic) |
//! | Kernel canonical bytes | Raw | YES (BLAKE3 commitment) |
//! | `CwmCommitment.blake3` | Raw | YES (audit event) |
//! | `BeliefInferenceFn::Sample` | Latent | NO (local to entity) |
//! | LLM prompts / refinement tree | Latent | NO (cold-tier private) |
//!
//! # Feature gate
//!
//! Gated behind the `induced_cwm` Cargo feature (default-off). Phase 2
//! (ISMCTS) adds `induced_cwm_ismcts = ["induced_cwm"]`. The orchestrator
//! wires this module in `lib.rs` via
//! `#[cfg(feature = "induced_cwm")] pub mod induced_cwm;`.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/296_induced_cwm_kernel_primitive.md`]
//! - Research: [`katgpt-rs/.research/275_Code_World_Model_Induced_Forward_Model.md`]
//! - Source paper: [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542)
//! - Direct ancestor: Plan 056 (`GameState` forward model + generic MCTS)
//! - Private Super-GOAT guide: `riir-ai/.research/145_CWM_Runtime_Induced_Game_Rules_Guide.md`
//! - Private runtime plan: `riir-ai/.plans/326_cwm_npc_runtime_integration.md`

pub mod belief;
pub mod commitment;
pub mod hot_swap;
pub mod kernel;
pub mod unit_test;

#[cfg(feature = "induced_cwm_ismcts")]
pub mod ismcts;

#[cfg(feature = "induced_cwm_tournament")]
pub mod tournament;

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "induced_cwm_ismcts"))]
mod ismcts_tests;

// ── Public API re-exports ─────────────────────────────────────────────────
//
// Mirrors the idiom used by other katgpt-core feature modules (e.g.
// `micro_belief`, `ict`): `pub use` the most common types at the module root
// so callers can write `katgpt_core::induced_cwm::InducedCwmKernel` instead
// of `katgpt_core::induced_cwm::kernel::InducedCwmKernel`.

pub use belief::BeliefInferenceFn;
pub use commitment::CwmCommitment;
pub use hot_swap::InducedCwmSlot;
pub use kernel::InducedCwmKernel;
pub use unit_test::{TransitionTestFailure, TransitionUnitTest, make_transition_tests_from_trajectory, verify_transition};

// ── Phase 2 (ISMCTS) re-exports ────────────────────────────────────────────
//
// Gated by `induced_cwm_ismcts` so callers can write
// `katgpt_core::induced_cwm::ismcts_search_with_inference` instead of
// `katgpt_core::induced_cwm::ismcts::ismcts_search_with_inference`.
#[cfg(feature = "induced_cwm_ismcts")]
pub use ismcts::{InformationSet, NodeStats, ismcts_search_with_inference};

// ── Phase 3 (Tournament) re-exports ────────────────────────────────────────
//
// Gated by `induced_cwm_tournament` so callers can write
// `katgpt_core::induced_cwm::ValueFnTournament` instead of
// `katgpt_core::induced_cwm::tournament::ValueFnTournament`.
#[cfg(feature = "induced_cwm_tournament")]
pub use tournament::{PlayerStats, TournamentWinner, ValueFnTournament};
