//! Screening primitives — open, modelless inference-time biases distilled from
//! Dingle–Hutter 2026 (*Entropy* 28(2):226, "Simplicity and Complexity in
//! Combinatorial Optimization").
//!
//! - **Plan 305** — Algorithmic-Probability Sampler + Coincidence Gate.
//! - **Research 284** — distillation note (`.research/284_*`).
//!
//! Two open primitives:
//!
//! 1. [`complexity_prior::CompressionPriorSampler`] — replaces uniform candidate
//!    sampling with `sigmoid(-α·K̃(x) - β)`-weighted sampling (per-candidate
//!    sigmoid, **never softmax**). Pluggable `K̃` proxies: RLE ratio, Shannon
//!    entropy, L1 norm. Latent variant operates on `&[f32]` via byte-quantization.
//! 2. [`coincidence_gate::CoincidenceGate`] — theorem-backed cross-task transfer.
//!    Given a found optimum `x*` for one simple objective `f1`, probe `x*` against
//!    other simple objectives `f2_k`. Hit rate: `r / |X_O(1)|` per probe vs
//!    `r / |X|` from random candidates (exponential lift).
//!
//! **Open boundary:** these modules operate on `&[u8]` / `&[f32]` only — no
//! HLA / functor / shard types. riir-ai Plan 331 wires the latent variant into
//! the private runtime (HLA / functor / cgsp); that wiring is intentionally NOT
//! in katgpt-rs.
//!
//! Safety guarantee: never worse than uniform; exponentially faster when the
//! optimum is low-K (Levin-search variant).

pub mod coincidence_gate;
pub mod complexity_prior;

pub use coincidence_gate::CoincidenceGate;
pub use complexity_prior::{
    ComplexityProxy, CompressionPriorSampler, EntropyComplexity, L1Complexity,
    LatentCompressionPriorSampler, RleComplexity, quantize_latent,
};

// ── Plan 305 Phase 3 integration hooks ─────────────────────────────────────
//
// Each sub-feature implies `complexity_prior_sampler` (declared in Cargo.toml),
// so when any of these is on, the `crate::screening::complexity_prior` import
// inside the integration module resolves.
//
// All three modules are zero-cost when their sub-feature is off: the
// `#![cfg(...)]` at the top of each file drops the entire module, and the
// `#[cfg(...)]` here drops the `pub mod` + `pub use` lines.

#[cfg(feature = "mcts_k_prior")]
pub mod integration_mcts;
#[cfg(feature = "mcts_k_prior")]
pub use integration_mcts::{KPriorExpansion, MctsExpansionPrior, UniformExpansion};

#[cfg(feature = "bandit_k_prior")]
pub mod integration_bandit;
#[cfg(feature = "bandit_k_prior")]
pub use integration_bandit::KPriorBandit;

#[cfg(feature = "spec_k_prior")]
pub mod integration_spec;
#[cfg(feature = "spec_k_prior")]
pub use integration_spec::KPriorDrafter;
