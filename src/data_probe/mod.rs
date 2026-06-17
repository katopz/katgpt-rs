//! Data Probe Diagnostics — controlled information-theoretic validation.
//!
//! This module implements a "probe-LLM" framework: a Markov chain with known
//! ground-truth transition probabilities, entropy rate, and stationary distribution.
//! By generating sequences from a known source and comparing against model
//! estimates, we can formally validate information-theoretic claims (C1–C4).
//!
//! # Module layout
//!
//! - [`markov`]       — Dirichlet-sampled Markov chain generator
//! - [`nll`]          — NLL computation against known chain
//! - [`typical_set`]  — Three-way regime classification (Conservative/Typical/Uncertain)
//! - [`dirichlet_energy`] — Dirichlet Energy structural alignment diagnostic
//! - [`claim`]        — Claim card infrastructure for formal C1–C4 validation
//! - [`geometry`]     — Representation geometry diagnostics (Plan 151)
//! - [`sink_classify`] — Per-head NOP/Broadcast sink classifier (Plan 287)
//!
//! # Mechanism locator vs aggregate symptom
//!
//! [`sink_classify`] is the **mechanism locator**: it identifies *which*
//! sink columns in an attention map are Adaptive NOPs vs Broadcasts.
//! [`geometry::effective_rank`] is the **aggregate symptom**: it measures
//! how collapsed the resulting hidden states are across the whole layer.
//! Broadcast sinks reduce `effective_rank` across tokens (Fesser et al.
//! Lemma 4); the classifier tells you *why*. Phase 4's `LayerSinkSummary`
//! (in [`geometry`]) bridges the two.

// ── Submodules ─────────────────────────────────────────────────

/// Dirichlet-sampled Markov chain generator with entropy rate targeting.
pub mod markov;

/// NLL computation against a known Markov chain.
pub mod nll;

/// Three-way regime classification based on typical-set framework.
pub mod typical_set;

/// Dirichlet Energy structural alignment diagnostic.
pub mod dirichlet_energy;

/// Claim card infrastructure for formal C1–C4 validation.
pub mod claim;

/// Representation geometry diagnostics (Plan 151, Research 113).
pub mod geometry;

/// Sink-Aware Attention classifier — per-head NOP/Broadcast detection
/// (Plan 287, Research 258, arxiv 2606.08105). Re-exports the primitive
/// from `katgpt_core::data_probe`.
pub mod sink_classify;

// ── Re-exports ─────────────────────────────────────────────────

pub use claim::{ClaimCard, Intervention, ValidityVerdict};
pub use dirichlet_energy::{
    consecutive_adjacency, dirichlet_energy, functor_adjacency, kv_cache_dirichlet_energy,
};
pub use geometry::{
    GeometryReport, LayerSinkSummary, avg_cosine_similarity, effective_rank,
    representation_geometry_report, summarize_layer_sinks,
};
pub use markov::{MarkovChain, generate_markov_chain, sample_sequence};
pub use nll::{average_nll, nll_profile};
pub use sink_classify::{
    SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind, StableRankScratch,
    apply_dual_policy_gate, classify_all_sinks, classify_sink_at, stable_rank_update_into,
};
pub use typical_set::{Regime, RegimeDistribution, classify_regime, regime_distribution};
