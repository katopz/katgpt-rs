//! `sleep_time` — Sleep-Time Query Anticipator open primitive (Plan 334).
//!
//! Implements the **open half** of Sleep-Time Compute (arXiv:2504.13171 —
//! Lin et al. Letta/Berkeley): a generic, game-semantic-free math primitive
//! for offline query anticipation. The private runtime half (per-NPC HLA
//! wiring, chain commitment, NPC tiering) lives in riir-ai Plan 341.
//!
//! # The core idea (one paragraph)
//!
//! At **sleep-time** (offline, when no player is watching), pre-compute
//! answers for the queries an NPC is *likely* to be asked. Store them in an
//! [`AnticipatedQuerySet`] — the "c' artifact". At **wake-time** (online,
//! when a player asks), do a cheap dot-product + sigmoid-gated lookup into
//! c'; fall through to fresh compute only on unpredictable queries. The
//! amortization win: one sleep-time compute serves many wake-time consumers.
//!
//! # The pipeline
//!
//! ```text
//! Sleep-time (offline, once per c):
//!   for i in 0..K:
//!     z_i = sleep_compute(c, D_set[i], budgets[i])   // consumer-provided op
//!     p_i = predictability(c, D_set[i])              // sigmoid(dot(c, dir))
//!   c' = AnticipatedQuerySet { slots: [(D_i, z_i, p_i)], blake3, version }
//!
//! Wake-time (online, per query):
//!   i* = argmax_i dot(q, D_set[i])
//!   gate = sigmoid(beta * (p_{i*} − tau))
//!   out = gate * z_{i*} + (1 − gate) * fresh_think(q)
//! ```
//!
//! # Modelless (katgpt-rs mandate)
//!
//! Every step is closed-form algebra — no training, no backprop. The weight
//! mutations allowed are:
//! 1. **Freeze/thaw** of the direction set + c' artifact (atomic, BLAKE3-checked).
//! 2. **Latent-space updates** — the predictability scores and precomputed z_i
//!    are latent scalars/vectors, recomputed deterministically from `c`.
//!
//! # Sigmoid not softmax (AGENTS.md)
//!
//! Every gate in this module is a single-scalar `sigmoid`. There is no
//! `softmax` symbol anywhere — predictability is per-direction, not a
//! distribution over directions. The wake-time blend is a smooth
//! `gate * precomputed + (1 − gate) * fresh`, never a hard argmax switch.
//!
//! # Latent vs Raw (AGENTS.md)
//!
//! - `AnticipatedQueryDir::direction` / `AnticipatedSlot::precomputed` →
//!   latent, frozen, BLAKE3-committed.
//! - `AnticipatedQuerySet::blake3` / `version` → raw, syncable audit
//!   artifact (the commitment root the chain quorum signs).
//! - `consume()` output → latent (blended latent answer). The bridge to raw
//!   scalars happens in the consumer (e.g. HLA → 5 affect scalars).
//!
//! # Zero-allocation discipline
//!
//! - `anticipate()` allocates the output c' artifact (necessary — it's the
//!   output). Per-direction compute uses caller-provided scratch.
//! - `consume()` (the wake-time hot path) is **zero-allocation** in steady
//!   state. The `fresh_think` closure MAY allocate (fallback path).
//! - G5 gate (Phase 2 T2.3) verifies 0 allocs / 0 bytes per `consume()`.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/334_sleep_time_query_anticipator_primitive.md`]
//! - Research: [`katgpt-rs/.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md`]
//! - Source paper: [arXiv:2504.13171](https://arxiv.org/abs/2504.13171) —
//!   Lin et al. 2025, *Sleep-time Compute: Beyond Inference Scaling at Test-time*.
//! - Private runtime: `riir-ai/.plans/341_npc_sleep_time_anticipation_runtime.md`.

mod anticipator;
mod consume;
mod cost_model;
mod predictability;
mod types;

pub use anticipator::{
    IdentityFunctorOp, SleepTimeAnticipator, SleepTimeComputeOp, SleepTimeScratch,
};
pub use consume::{consume, consume_gate};
pub use cost_model::AmortizationCostModel;
pub use predictability::{DotPredictabilityScorer, PredictabilityScorer};
pub use types::{
    commit_direction, AnticipatedQueryDir, AnticipatedQuerySet, AnticipatedSlot,
};

// ── Constants ────────────────────────────────────────────────────────────────

/// Default latency premium (paper §5.3 uses t=10 — wake-time compute is 10×
/// more expensive per token than sleep-time compute, because it's on the
/// user's critical path).
pub const DEFAULT_LATENCY_PREMIUM: f32 = 10.0;

/// Default catalog size per NPC type (paper uses K≤10; we expect K≤8).
/// Prefixed `SLEEP_TIME_` to avoid collision with `cgsp::DEFAULT_K` when both
/// features are enabled (caught by `cargo check --all-features`).
pub const SLEEP_TIME_DEFAULT_K: usize = 8;
