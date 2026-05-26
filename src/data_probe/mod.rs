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
//! - [`claim`]        — Claim card infrastructure for formal C1–C4 validation

// ── Submodules ─────────────────────────────────────────────────

/// Dirichlet-sampled Markov chain generator with entropy rate targeting.
pub mod markov;

/// NLL computation against a known Markov chain.
pub mod nll;

/// Three-way regime classification based on typical-set framework.
pub mod typical_set;

/// Claim card infrastructure for formal C1–C4 validation.
pub mod claim;

// ── Re-exports ─────────────────────────────────────────────────

pub use claim::{ClaimCard, Intervention, ValidityVerdict};
pub use markov::{MarkovChain, generate_markov_chain, sample_sequence};
pub use nll::{average_nll, nll_profile};
pub use typical_set::{Regime, RegimeDistribution, classify_regime, regime_distribution};
