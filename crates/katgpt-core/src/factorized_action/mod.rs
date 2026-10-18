//! Factorized Transition Action Abstraction — modelless compositional
//! action latent primitive (Plan 375).
//!
//! Distilled from Nam et al., *Latent Actions from Factorized Transition
//! Effects under Agent Ambiguity*, [arXiv:2606.30544](https://arxiv.org/abs/2606.30544),
//! Brown, 2026-06-30. Research note:
//! [`katgpt-rs/.research/374_OTF_LAM_Factorized_Transition_Primitives.md`](../../../.research/374_OTF_LAM_Factorized_Transition_Primitives.md).
//!
//! # Mechanism
//!
//! Given a frozen codebook of K D-dim effect primitives, decompose an
//! observation transition into a sparse set of active primitives, score
//! each via a state-aware sigmoid relevance gate, and aggregate via a
//! normalized weighted average into a compact action latent:
//!
//! ```text
//! patches → Top-1 assign → finalize → factor_token (FiLM) → gate/mean → z
//! ```
//!
//! This is the **factorized/compositional cousin** of the shipped
//! monolithic `latent_functor` (`extract_functor` / `apply_functor`,
//! riir-ai Plan 273) — it enriches the action representation from "one
//! displacement vector" to "a mixture of K reusable effect primitives
//! gated by current state".
//!
//! # Modelless contract
//!
//! The codebook is constructed modellessly via **k-means clustering** on
//! observed transition patches (deterministic Lloyd's algorithm with
//! k-means++ init, no gradient descent — Path 2 of AGENTS.md §3.5). The
//! full inference path (patchify → assign → gate → aggregate) is
//! zero-allocation, sigmoid-gated (never softmax), feature-flagged.
//!
//! # Feature flag
//!
//! Opt-in via `factorized_action`. Promotion to default-on requires the
//! GOAT gate (Plan 375 Phase 3, `bench_375_factorized_action_goat`) to
//! pass G1–G6.
//!
//! # Module layout
//!
//! - [`types`] — pure data: `EffectCodebook`, `TransitionFactors`,
//!   `FactorizedActionLatent`, `AggregatorType`, `FilmProjectionBank`.
//! - [`kernel`] — inference hot path: `assign_patch_into`,
//!   `finalize_factors`, `factor_token_into`,
//!   `aggregate_action_latent_into`, `relevance_score`.
//! - [`codebook`] — offline construction: `fit_codebook_kmeans_into`,
//!   `from_observed_transitions`, `patchify_1d`,
//!   `motion_input_velocity_into`.
//!
//! # Sigmoid mandate
//!
//! The relevance gate uses `sigmoid(β·(relevance − τ))`, NEVER softmax.
//! Per AGENTS.md §2 and the paper's own design (sigmoid gating
//! throughout, verified in `otf_lam/model.py::GateNetwork.forward()`).
//!
//! Factorized Transition Action Abstraction — modelless compositional action
//! latent primitive distilled from Nam et al., *Latent Actions from Factorized
//! Transition Effects under Agent Ambiguity* (arXiv:2606.30544, Brown, 2026-06-30).
//! Research 374, Plan 375. The factorized/compositional cousin of the shipped
//! monolithic `latent_functor` (riir-ai Plan 273): frozen codebook of K D-dim
//! effect primitives + Top-1 patch assignment + sigmoid relevance gate +
//! normalized weighted average → compact action latent. Codebook constructed
//! modellessly via Lloyd's k-means (Path 2 of AGENTS.md §3.5 — deterministic,
//! no gradient descent). Sigmoid gating throughout (NEVER softmax per AGENTS.md
//! §2, verified in `otf_lam/model.py::GateNetwork.forward()`). Opt-in until the
//! G1–G6 GOAT gate (bench_375_factorized_action_goat) passes.

pub mod codebook;
pub mod kernel;
pub mod types;

pub use codebook::{fit_codebook_kmeans_into, motion_input_velocity_into, patchify_1d};
pub use kernel::{
    aggregate_action_latent_into, factor_token_into, finalize_factors, relevance_score,
};
pub use types::{
    AggregatorType, EffectCodebook, FactorizedActionLatent, FilmProjectionBank, MAX_K, MAX_PATCHES,
    TransitionFactors,
};
