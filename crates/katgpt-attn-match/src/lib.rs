//! Attention Matching (AM) KV cache compaction — modelless (Plan 271, Research 233).
// Allow unexpected cfgs — score_matrix_gpu.rs docs reference the root-only
// `gpu_inference` feature (the GPU dispatch is a stub that falls back to rayon).
#![allow(unexpected_cfgs)]
//!
//! Implements "Fast KV Compaction via Attention Matching"
//! (Zweiger, Fu, Guo, Kim — MIT, ICML 2026, arxiv 2602.16284).
//!
//! # The Idea
//!
//! When compacting a KV cache `(K, V)` to `(Ck, β, Cv)` with `t < T` tokens,
//! preserve two quantities on a set of reference queries `Qref`:
//!
//! 1. **Attention output** (Eq. 1):
//!    `softmax(qK^T/√d) V ≈ softmax((qCk^T + β)/√d) Cv`
//!
//! 2. **Attention mass** (Eq. 2):
//!    `Σ_j exp(qK_j^T/√d) ≈ Σ_j exp((qCk_j^T + β_j)/√d)`
//!
//! Together these guarantee that the compacted block's contribution under
//! concatenation with arbitrary future `(Kfixed, Vfixed)` is preserved,
//! because attention over concatenated blocks decomposes into a mixture
//! whose weights are determined by unnormalized attention mass.
//!
//! # Why β is Critical
//!
//! With `t < T` and no bias, `Mass(q; Ck) ≤ Mass(q; K)` always — compaction
//! systematically underestimates the compacted block's contribution.
//! β introduces per-token additive attention biases so each retained key can
//! account for the mass of many removed keys.
//!
//! Unlike `still_kv` (Plan 245, Research 213) which uses a **heuristic**
//! β (`log(T/t)` or vortex-flow weighting), AM computes **optimal** β via
//! nonnegative least squares (NNLS) — directly minimizing attention mass error.
//!
//! # The Pipeline (Closed-Form, No Gradient Descent)
//!
//! 1. Select compact keys `Ck` (subset of original `K`):
//!    - `HighestAttnKeys` — top-t by RMS attention score across `Qref`
//!    - `OMP` — greedy orthogonal matching pursuit on the mass feature matrix
//! 2. Fit `β` via NNLS (projected gradient descent)
//! 3. Fit `Cv` via ordinary least squares (normal equations + Cholesky)
//!
//! # Reference
//!
//! See `.research/233_Attention_Matching_KV_Compaction.md` for the full
//! distillation verdict, fusion ideas, and GOAT gate matrix.

// All modules are gated behind the `attn_match` feature (mirrors the root
// crate's historical `#[cfg(feature = "attn_match")] pub mod attn_match;`).
// Consumers enable `katgpt-attn-match/attn_match`; root forwards via
// `attn_match = ["katgpt-attn-match/attn_match"]`.

#[cfg(feature = "attn_match")]
pub mod beta_fitter;
#[cfg(feature = "attn_match")]
pub mod chunked;
#[cfg(feature = "attn_match")]
pub mod compact;
#[cfg(feature = "attn_match")]
pub mod compact_fixed_beta;
#[cfg(feature = "attn_match")]
pub mod head_budget;
#[cfg(feature = "attn_match")]
pub mod key_selection;
#[cfg(feature = "attn_match")]
pub mod online;
#[cfg(feature = "attn_match")]
pub mod router;
#[cfg(feature = "attn_match")]
pub mod score_matrix;
#[cfg(feature = "attn_match")]
pub mod score_matrix_gpu;
#[cfg(feature = "attn_match")]
pub mod score_matrix_rayon;
#[cfg(feature = "attn_match")]
pub mod score_matrix_simd;
#[cfg(feature = "attn_match")]
pub mod types;
#[cfg(feature = "attn_match")]
pub mod value_fitter;

#[cfg(feature = "attn_match")]
pub use beta_fitter::{fit_beta_nnls, BetaFitConfig, BetaFitResult};
#[cfg(feature = "attn_match")]
pub use chunked::{ChunkedCompactor, ChunkedCompactOutput, ChunkMeta, TextChunk};
#[cfg(feature = "attn_match")]
pub use compact::{compact, compact_with_router, CompactError, CompactOutput, RouterTrace};
#[cfg(feature = "attn_match")]
pub use head_budget::{HeadBudgetSchedule, HeadBudgetSolver, HeadSensitivityCurve};
#[cfg(feature = "attn_match")]
pub use key_selection::{
    highest_attn::select_highest_attn_keys, omp::select_omp_keys, KeySelection, KeySelectorKind,
};
#[cfg(feature = "attn_match")]
pub use online::{OnlineCompactResult, OnlineCompactor};
#[cfg(feature = "attn_match")]
pub use router::{pick_backend, SolverBackend, SolverRouter, SolverRouterConfig};
#[cfg(feature = "attn_match")]
pub use score_matrix::{compute_attention_output, compute_score_matrix, compute_softmax_attention};
#[cfg(feature = "attn_match")]
pub use score_matrix_rayon::{compute_rms_attention_rayon, compute_score_matrix_rayon};
#[cfg(feature = "attn_match")]
pub use score_matrix_simd::compute_score_matrix_simd;
#[cfg(feature = "attn_match")]
pub use types::{
    AmConfig, AmResult, KeySelector, ReconstructionReport, ScoreMethod, SolverChoice,
};
#[cfg(feature = "attn_match")]
pub use value_fitter::{fit_cv_least_squares, ValueFitConfig, ValueFitResult};

// NOTE: `adaptive_cot` (AdaptiveTraceCompactor / AdaptiveCompactResult) stays in the
// katgpt-rs ROOT crate at `src/attn_match_adaptive_cot.rs` because it composes
// `freq_bandit` + `trigger_gate` (root-only modules). See Issue 359. The root
// re-exports this leaf as `attn_match`, then adds the adaptive_cot glue on top.

// Simplified AM fast path (Issue 305): skip NNLS, use a fixed β (BETA_MID).
// Always-available — no feature gate. Replaces the removed `lora_beta_predictor`
// / `lora_beta_inference` modules, which proved mathematically moot under
// softmax invariance.
#[cfg(feature = "attn_match")]
pub use compact_fixed_beta::{compact_with_fixed_beta, BETA_MAX, BETA_MID, BETA_MIN};

#[cfg(all(test, feature = "attn_match"))]
mod tests;

/// Numerical stability epsilon for log/exp operations.
#[cfg(feature = "attn_match")]
pub const STABILITY_EPS: f32 = 1e-12;

/// Default diagonal jitter for Cholesky when rank-deficient.
#[cfg(feature = "attn_match")]
pub const DEFAULT_CHOLESKY_JITTER: f32 = 1e-6;

/// Default bound on `w = exp(β)` for HighestAttnKeys (paper Appendix C.2):
/// β ∈ [-3, 3] ⇒ w ∈ [e^-3, e^3].
#[cfg(feature = "attn_match")]
pub const DEFAULT_W_LOWER: f32 = 1e-3; // ~ e^-6.9, safe lower bound
#[cfg(feature = "attn_match")]
pub const DEFAULT_W_UPPER: f32 = 1_096.633; // e^7 ≈ 1096.63 (paper cap)

/// β lower bound for OMP key pruning (paper Appendix C.2).
#[cfg(feature = "attn_match")]
pub const OMP_BETA_PRUNE_THRESHOLD: f32 = -7.0;
