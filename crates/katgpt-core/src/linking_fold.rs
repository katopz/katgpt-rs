//! Linking-Number Detector + Fold Correction — modelless topological primitives.
//!
//! Distillation of Ren & Lim, *Low-dimensional topology of deep neural networks*,
//! ICML 2026 (arXiv:2606.31856). The paper proves that width-d feedforward nets
//! with coordinate-wise **monotonic** activations (ReLU, sigmoid, tanh) cannot
//! linearly separate two topologically **linked** class manifolds (link ≠ 0),
//! regardless of depth (Theorem 4.7). Only **folding** operations break the
//! constraint: ResNet skip (`|x| = x + 2·ReLU(−x)`, Eq. 1), attention, and
//! non-monotonic activations (GELU, Swish) — paper §5.
//!
//! This module ships two modelless primitives that close a gap the codebase has
//! implicitly: every sigmoid projection (HLA affect scalars, direction-vector
//! projections, `ItemEmbedIndex` cosine retrieval) is monotonic → provably
//! doomed on linked manifolds, but the codebase had no way to *detect* when.
//!
//! # Primitives
//!
//! 1. **[`linking_detector::detect_linking`]** — paper Algorithm 1: take two
//!    point clouds X, Y in R^d, PCA-project to R^3, build ε-filtered k-NN
//!    graphs, extract a fundamental cycle basis per graph via BFS spanning
//!    forest, compute the Gauss linking integral over O(β_X · β_Y) basis-cycle
//!    pairs. Returns [`linking_detector::LinkingVerdict`].
//! 2. **[`fold_projection_into`]** / **[`fold_gelu_into`]** — coordinate-wise
//!    `x ↦ c + |x − c|` fold (paper Eq. 1) applied in-place. The deterministic
//!    modelless correction when the detector fires.
//!
//! # Design
//!
//! - **Modelless**: detector is pure point-cloud geometry (PCA + k-NN + cycle
//!   basis + Gauss quadrature); fold is closed-form `|x − c|`. No weights, no
//!   training, no backprop.
//! - **Cold-path detector, hot-path fold**: the detector is brute-force O(n²)
//!   k-NN + cycle-basis construction — audit-cadence, may allocate. The fold
//!   is per-tick, zero-allocation, `#[inline]`.
//! - **Generic over `&[f32]`**: no game / chain / shard semantics. Operates on
//!   flat f32 slices; caller decides what the vectors mean (HLA states, shard
//!   style_weights, NPC behavior embeddings).
//! - **No new deps**: PCA is power-iteration on a 3×3 covariance (not an FFT);
//!   k-NN is brute-force (cold path); cycle basis is BFS; Gauss integral is
//!   midpoint quadrature. Pure stdlib.
//!
//! # Why this matters here
//!
//! Sigmoid is the codebase's mandated projection (AGENTS.md: sigmoid, never
//! softmax). By Theorem 4.7, sigmoid projections of linked manifolds cannot
//! achieve linear separability — the affect scalars are structurally
//! insufficient to distinguish two linked NPC behavior classes. The detector
//! says *when* the projection is doomed; the fold is the §3.5 path-3 latent
//! correction (deterministic, closed-form, no GD — exactly what the modelless
//! unblock protocol prefers over riir-train deferral).
//!
//! References:
//! - Plan 410 — open-primitive spec
//! - Research 391 — math + prior-art table + Q1–Q4 novelty gate
//! - arXiv:2606.31856 — Ren & Lim ICML 2026 (paper)
//!
//! linking_fold — Linking-Number Detector + Fold Correction (Plan 410,
//! Research 391, arXiv:2606.31856 Ren & Lim ICML 2026). SPLIT (Plan 410 T4.4
//! Option C, 2026-07-07) into two independently-gated sub-features:
//!   - linking_fold_fold     (hot-path |x−c| fold correction) — DEFAULT-ON
//!   - linking_fold_detector (cold-path Algorithm-1 linking detector) — opt-in
//!   - linking_fold          (umbrella = fold + detector) — opt-in
//!
//! The fold passes every GOAT gate modellessly and ships default-on; the
//! detector's G2 budget is the audit-cadence-appropriate 500 ms @ n=2×200
//! (Issue 050 Option A, resolved 2026-07-07) and it stays opt-in. The
//! module root exists when EITHER sub-feature is on; submodules gate their own
//! parts.

// ═══════════════════════════════════════════════════════════════════════════
// Module layout: fold (hot-path) first, then linking_detector (cold-path) as
// a submodule namespace. The fold is the cheap correction that consumers will
// call per-tick; the detector is the audit-cadence diagnostic that tells them
// when to call it.
//
// Feature split (Plan 410 T4.4 Option C, 2026-07-07): each submodule is gated
// on its own sub-feature so they ship independently.
//   - `linking_fold_fold`     → `fold` (hot-path, DEFAULT-ON)
//   - `linking_fold_detector` → `linking_detector` (cold-path, opt-in)
// The umbrella `linking_fold = [fold, detector]` enables both. The module root
// exists under either sub-feature (see lib.rs).
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "linking_fold_fold")]
pub mod fold;
#[cfg(feature = "linking_fold_detector")]
pub mod linking_detector;

#[cfg(feature = "linking_fold_fold")]
pub use fold::{fold_gelu_into, fold_projection_into, gelu_smoothed_abs};
#[cfg(feature = "linking_fold_detector")]
pub use linking_detector::{
    LinkingDetectorConfig, LinkingVerdict, detect_linking, detect_linking_into,
};
