//! Chiaroscuro Attention — Spectral-Entropy Operator Routing (Plan 269).
//!
//! Implements CHIAR-Former's three reusable inference-time primitives plus the
//! novel CHIAR-KV cache fusion. Pure inference-time — no gradients, no training,
//! no learned filter.
//!
//! # Architecture
//!
//! ```text
//! Token embedding x
//!      │
//!      ▼
//! ┌──────────────────────────┐
//! │ spectral_entropy_dct(x)  │  ← Fusion 0: per-token H(x) ∈ [0, 1]
//! └──────────────────────────┘
//!      │
//!      ├──────────────────────────────────────────┐
//!      ▼                                          ▼
//! ┌───────────────────┐                  ┌────────────────────┐
//! │  ChiaroscuroKv    │                  │  ChiaroscuroRouter │
//! │  (Fusion A)       │                  │  (Fusion B)        │
//! │                   │                  │                    │
//! │  H<τ_lo: DCT-trunc│                  │  Routes token to   │
//! │  H<τ_hi: Quantized│                  │  DctMix or FullAttn│
//! │  else:    Full    │                  │  op based on H(x)  │
//! └───────────────────┘                  └────────────────────┘
//!      │                                          │
//!      │           ┌──────────────────────┐        │
//!      └──────────►│  CollapseDiscovery   │◄───────┘
//!                  │  Harness (Fusion C)  │
//!                  │                      │
//!                  │  Detects U → 0       │
//!                  │  → OpPromotion       │
//!                  └──────────────────────┘
//!                            │
//!                            ▼
//!                  ┌────────────────────┐
//!                  │  ChiarRegimeGate   │
//!                  │  (Fusion D)        │
//!                  │                    │
//!                  │  Long+varied → on  │
//!                  │  Short/flat → off  │
//!                  └────────────────────┘
//! ```
//!
//! # Feature gate
//!
//! All CHIAR modules are behind the `chiaroscuro` feature flag (opt-in).
//! When disabled, zero impact on the rest of the crate.
//!
//! # Example
//!
//! ```no_run
//! use katgpt_rs::chiaroscuro::{
//!     kv::ChiaroscuroKvStrategy,
//!     tau::StreamingTauCalibrator,
//! };
//!
//! let mut calibrator = StreamingTauCalibrator::default();
//! let keys: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 64]).collect();
//! for k in &keys {
//!     calibrator.observe_embedding(k);
//! }
//! let tau_lo = calibrator.tau_lo();
//! let tau_hi = calibrator.tau_hi();
//! for k in &keys {
//!     let strategy = ChiaroscuroKvStrategy::decide_from_key(k, tau_lo, tau_hi);
//!     // Apply strategy to KV cache entry...
//! }
//! ```

pub mod collapse;
pub mod entropy;
pub mod kv;
pub mod op_trait;
pub mod regime;
pub mod tau;

// Convenience re-exports.
pub use collapse::{CollapseDiscoveryHarness, OpPromotion, DEFAULT_COLLAPSE_THRESHOLD};
pub use entropy::{sigmoid, spectral_entropy_dct, spectral_entropy_dct_into};
pub use kv::{
    ChiaroscuroKvDispatcher, ChiaroscuroKvStrategy, StrategyUtilization, DEFAULT_DCT_TRUNCATED_COEFFS,
};
pub use op_trait::{ChiaroscuroOp, ChiaroscuroRouter, DctMixOp, FullAttnOp};
pub use regime::{ChiarRegimeGate, WelfordVariance, DEFAULT_MIN_PROMPT_TOKENS, DEFAULT_NATURALISTIC_VARIANCE};
pub use tau::{StreamingTauCalibrator, DEFAULT_MIN_SAMPLES, DEFAULT_TAU_HI, DEFAULT_TAU_LO};

// TL;DR: Chiaroscuro Attention — per-token DCT spectral entropy drives
// (A) KV cache storage strategy, (B) operator routing, (C) collapse discovery,
// (D) operating regime gate. Pure inference-time, opt-in feature `chiaroscuro`.
