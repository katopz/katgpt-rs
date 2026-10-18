//! Data Probe Diagnostics — controlled information-theoretic validation.
//!
//! This module implements a "probe-LLM" framework: a Markov chain with known
//! ground-truth transition probabilities, entropy rate, and stationary distribution.
//! By generating sequences from a known source and comparing against model
//! estimates, we can formally validate information-theoretic claims (C1–C4).
//!
//! # Module layout
//!
//! - [`markov`]           — Dirichlet-sampled Markov chain generator (always-on)
//! - [`nll`]              — NLL computation against known chain (always-on)
//! - [`typical_set`]      — Three-way regime classification (always-on)
//! - [`dirichlet_energy()`] — Dirichlet Energy structural alignment diagnostic (always-on)
//! - [`claim`]            — Claim card infrastructure for formal C1–C4 validation (always-on)
//! - [`geometry`]         — Representation geometry diagnostics (Plan 151) — gated `sink_aware_attn`
//! - `sink_classify`    — Per-head NOP/Broadcast sink classifier (Plan 287) — gated `sink_aware_attn`
//! - `gold_share`       — Content-specific output-fraction diagnostic (Plan 411) — gated `gold_share_probe`
//! - `gaussianity`      — Sketched projection-normality probe (Issue 681) — gated `gaussianity_probe`
//! - [`cca`]              — SVCCA SVD-denoised CCA subspace similarity (Issue 684) — gated `svd_cca`
//!
//! # Always-on vs feature-gated split
//!
//! The information-theoretic substrate (markov, nll, typical_set,
//! dirichlet_energy, claim) is always-on — it's pure math with no heavy deps.
//! The sink-aware classifier + its geometry bridge are gated
//! `sink_aware_attn` because they're an opt-in attention intervention
//! (Plan 287, default stays Uniform pending G2/G3 GOAT gate).
//!
//! # Mechanism locator vs aggregate symptom
//!
//! `sink_classify` is the **mechanism locator**: it identifies *which*
//! sink columns in an attention map are Adaptive NOPs vs Broadcasts.
//! `geometry::effective_rank` is the **aggregate symptom**: it measures
//! how collapsed the resulting hidden states are across the whole layer.
//! Broadcast sinks reduce `effective_rank` across tokens (Fesser et al.
//! Lemma 4); the classifier tells you *why*. Phase 4's `LayerSinkSummary`
//! (in `geometry`) bridges the two.
//!
//! `gold_share` (Plan 411, Research 392) adds the **content-specific**
//! view: it tells you whether the layer's output still carries the *gold*
//! signal or has been rewritten to carry aggregate noise — orthogonal to
//! both `effective_rank` (content-agnostic) and `stable_rank_update`
//! (per-sink degeneracy).
//!
//! # Plan 404 history (2026-07-06)
//!
//! The markov/nll/typical_set/dirichlet_energy/claim/geometry modules moved
//! here from root `src/data_probe/`. They're pure substrate (only deps:
//! katgpt-core + intra-module). Root keeps re-export shims so every
//! historical `katgpt_rs::data_probe::*` path resolves.
//!
//! Sink-Aware Attention — NOP/Broadcast classifier + dual-policy sigmoid gate
//! (Plan 287, Research 258, arxiv 2606.08105, Fesser et al.). Per-head
//! classifier (value-norm-ratio + stable-rank-of-update) decides whether a
//! sink is Adaptive NOP (gate it via sigmoid) or Broadcast (preserve it).
//! Staged integration: the policy enum + standalone apply_dual_policy_gate
//! ship here; direct wiring into parallax_attn / funcattn forward paths is
//! deferred until synthetic G2 + latency G3 gates pass on a real model
//! (validation fallback per Plan 287 §Validation).
//!
//! Plan 404 (2026-07-06): the parent module is now always-on. The pure
//! information-theoretic substrate (markov/nll/typical_set/dirichlet_energy/
//! claim) moved here from root `src/data_probe/`. The sink-aware classifier
//! (`sink_classify`) + `geometry` remain gated `sink_aware_attn` inside the
//! module. The gated re-exports below preserve `crate::data_probe::SinkKind`
//! etc. for internal consumers (notably `parallax_attn.rs`). The always-on
//! re-exports (markov/nll/typical_set/dirichlet_energy/claim) live in
//! `data_probe/mod.rs`.

// ── Always-on submodules (pure information-theoretic substrate) ─────────

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

// ── Feature-gated submodules (sink-aware attention intervention) ────────

/// Representation geometry diagnostics (Plan 151, Research 113).
/// Gated `sink_aware_attn` because it depends on `sink_classify` types.
#[cfg(feature = "sink_aware_attn")]
pub mod geometry;

/// Sink-Aware Attention classifier — per-head NOP/Broadcast detection
/// (Plan 287, Research 258, arxiv 2606.08105).
#[cfg(feature = "sink_aware_attn")]
pub mod sink_classify;

/// GoldShare content-specific output-fraction diagnostic (Plan 411,
/// Research 392, arxiv 2607.01538). `‖a^G_L‖ / ‖a_L‖` — the gold fraction
/// of a layer's attention output. Detects the paper's recall-generation
/// gap (signal in heads, lost in residual). Standalone norm computation
/// (no deps); gated `gold_share_probe`.
#[cfg(feature = "gold_share_probe")]
pub mod gold_share;

/// Sketched Gaussianity Probe — multi-direction projection-normality for
/// embedding populations (Issue 681, Research 498 — SIGReg distilled from
/// training loss to inference-time diagnostic). Catches the bimodal /
/// heavy-tail / discrete marginals that pass every second-moment metric
/// (erank et al.). Standalone: BLAKE3 direction table + self-contained KS;
/// gated `gaussianity_probe` (opt-in until a consumer promotes).
#[cfg(feature = "gaussianity_probe")]
pub mod gaussianity;

/// SVCCA — SVD-denoised CCA subspace similarity between two representation
/// snapshots (Issue 684, Research 501 — arXiv:1706.05806).
/// Affine-invariant ρ spectrum + mean over the η=0.99-energy subspaces;
/// answers "same function, up to invertible linear re-mixing?" where BLAKE3
/// only answers "same bytes". Composes `symmetric_eig` + `ns_inv_sqrt_psd_into`
/// + `numerical_rank`; gated `svd_cca` (opt-in until a consumer promotes).
#[cfg(feature = "svd_cca")]
pub mod cca;

/// kNN differential-entropy estimator (Kozachenko–Leonenko) — the
/// uncertainty/entropy axis beside the dispersion proxies above; the P2
/// imbalance-collapse monitor's channel A (Issue 708, Research 437).
/// Gated `knn_entropy` (opt-in — classic prior-art import, no consumer yet).
#[cfg(feature = "knn_entropy")]
pub mod entropy;

/// Two-channel imbalance collapse monitor (Issue 708 P2, Research 437 —
/// arXiv:2608.29335 GenFirst): channel A = the KL entropy above (spread),
/// channel B = strided-pair mean |cosine| (directional concentration);
/// conjunctive z-score alarm against a warm-up-frozen baseline — deviation
/// from baseline leads an absolute `h < τ_low` crossing (measured by
/// `bench_708_imbalance_goat` on the bench_681 fixtures), with the
/// isotropic-shrink scope boundary pinned in-module. Event-triggered
/// reporting only — never a controller. Gated `imbalance_monitor`
/// (implies `knn_entropy`); first consumer surface: the `population_collapse`
/// advisory latch on katgpt-pruners' `S2FCollapseDetector`.
#[cfg(feature = "imbalance_monitor")]
pub mod imbalance;

// ── Re-exports (always-on items) ────────────────────────────────────────

pub use claim::{ClaimCard, Intervention, ValidityVerdict};
#[cfg(feature = "dirichlet_energy")]
pub use dirichlet_energy::{
    consecutive_adjacency, dirichlet_energy, functor_adjacency, kv_cache_dirichlet_energy,
};
pub use markov::{MarkovChain, generate_markov_chain, sample_sequence};
pub use nll::{average_nll, nll_profile};
pub use typical_set::{Regime, RegimeDistribution, classify_regime, regime_distribution};

// ── Re-exports (sink-aware items, gated) ────────────────────────────────
//
// These mirror the historical `pub use data_probe::{...}` block that lived
// in katgpt-core's lib.rs. They're kept here so `crate::data_probe::SinkKind`
// etc. resolve for internal consumers (notably `parallax_attn.rs`) AND so
// root's `katgpt_rs::data_probe::sink_classify::*` re-export path works.

#[cfg(feature = "sink_aware_attn")]
pub use geometry::{
    GeometryReport, LayerSinkSummary, WithinClassGeometryReport, avg_cosine_similarity,
    effective_rank, representation_geometry_report, summarize_layer_sinks,
    within_class_effective_rank, within_class_effective_rank_owned, within_class_geometry_report,
};

#[cfg(feature = "sink_aware_attn")]
pub use sink_classify::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};

/// Sink stability forecast (Issue 819 T1, Research 566). The item lives in
/// `sink_classify` (itself gated `sink_aware_attn`), so both features must be
/// on for the re-export to resolve.
#[cfg(all(feature = "sink_aware_attn", feature = "sink_margin_forecast"))]
pub use sink_classify::forecast_stable_positions;

// ── Re-exports (gold_share items, gated) ────────────────────────────────

#[cfg(feature = "gold_share_probe")]
pub use gold_share::{GoldShareReport, GoldShareScratch, gold_share, gold_share_flat};

// ── Re-exports (gaussianity items, gated) ──────────────────────────

#[cfg(feature = "gaussianity_probe")]
pub use gaussianity::{
    GaussianityReport, GaussianityScratch, N_AXIS_ANCHORS, N_DIRECTIONS, ks_d_vs_fitted_gaussian,
    sketched_gaussianity,
};

// ── Re-exports (svd_cca items, gated) ─────────────────────────────

#[cfg(feature = "svd_cca")]
pub use cca::{CcaReport, CcaScratch, MAX_K, svcca_into};
