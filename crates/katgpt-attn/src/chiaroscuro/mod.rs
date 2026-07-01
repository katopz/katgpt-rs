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
//! use katgpt_attn::chiaroscuro::{
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

// ---------------------------------------------------------------------------
// ChiarRouterHook — InferenceRouter integration point (Plan 269 T15)
// ---------------------------------------------------------------------------
//
// Lightweight bridge between the CHIAR per-token primitives and the
// InferenceRouter's observation surface. The router does NOT make tier
// decisions based on CHIAR (CHIAR is a per-token attention operation, not a
// tier-routing signal). Instead, this hook exposes CHIAR utilization stats
// via RouterStats so callers can observe KV strategy distribution and regime.

/// Snapshot of CHIAR signals exposed via [`crate::inference_router::RouterStats`].
///
/// All fields are `None` when no keys have been observed yet (cold start).
#[derive(Clone, Debug, Default)]
pub struct ChiarRouterStats {
    /// Utilization entropy of the KV storage strategy dispatcher.
    /// `None` if no keys observed. Range `[0, 1]` — `1.0` = uniform mix,
    /// `0.0` = collapse (all tokens to one strategy).
    pub utilization_entropy: Option<f32>,
    /// Whether the regime gate currently recommends applying CHIAR.
    /// `None` if no prompt observed yet.
    pub should_apply_chiar: Option<bool>,
    /// Count of tokens observed by the KV dispatcher.
    pub tokens_observed: u64,
}

/// InferenceRouter hook for CHIAR observation (Plan 269 T15).
///
/// Wraps a [`ChiaroscuroKvDispatcher`] and [`ChiarRegimeGate`] so the router
/// can observe per-token spectral entropy signals without owning the full
/// CHIAR pipeline. Zero cost when `chiaroscuro` feature is disabled.
///
/// This is observation-only: it does NOT influence tier routing. CHIAR's
/// per-token DCT-mix vs full-attention routing happens inside the attention
/// layer, not at the InferenceRouter level.
pub struct ChiarRouterHook {
    dispatcher: ChiaroscuroKvDispatcher,
    regime_gate: ChiarRegimeGate,
    tau_calibrator: StreamingTauCalibrator,
}

impl ChiarRouterHook {
    /// Create a new hook with default configuration.
    pub fn new() -> Self {
        Self {
            dispatcher: ChiaroscuroKvDispatcher::new(DEFAULT_DCT_TRUNCATED_COEFFS),
            regime_gate: ChiarRegimeGate::default(),
            tau_calibrator: StreamingTauCalibrator::default(),
        }
    }

    /// Observe a key embedding for KV strategy classification.
    ///
    /// Updates the τ calibrator and dispatches the key to a storage strategy.
    /// Call this for each key entering the KV cache.
    pub fn observe_key(&mut self, key: &[f32]) {
        self.tau_calibrator.observe_embedding(key);
        let lo = self.tau_calibrator.tau_lo();
        let hi = self.tau_calibrator.tau_hi();
        self.dispatcher.dispatch(key, lo, hi);
    }

    /// Observe a prompt token's spectral entropy for regime classification.
    ///
    /// Updates the Welford variance tracker inside the regime gate.
    pub fn observe_prompt_token(&mut self, h: f32) {
        self.regime_gate.observe_h(h);
    }

    /// Snapshot the current CHIAR stats for RouterStats reporting.
    pub fn stats(&self) -> ChiarRouterStats {
        let total = self.dispatcher.utilization.total();
        ChiarRouterStats {
            utilization_entropy: if total > 0 {
                Some(self.dispatcher.utilization_entropy())
            } else {
                None
            },
            should_apply_chiar: if self.regime_gate.prompt_tokens() > 0 {
                Some(self.regime_gate.should_apply_chiar())
            } else {
                None
            },
            tokens_observed: total,
        }
    }
}

impl Default for ChiarRouterHook {
    fn default() -> Self {
        Self::new()
    }
}

// TL;DR: Chiaroscuro Attention — per-token DCT spectral entropy drives
// (A) KV cache storage strategy, (B) operator routing, (C) collapse discovery,
// (D) operating regime gate. Pure inference-time, opt-in feature `chiaroscuro`.
