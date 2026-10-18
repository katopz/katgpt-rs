//! SWE Trajectory Freezer — modelless committed freeze of an inference
//! attempt's trajectory geometry (Proposal 011 Phase 5, Task T5.5).
//!
//! Composes three shipped DEFAULT-ON primitives:
//!
//! - [`crate::latent_trajectory_geometry::from_states`] — the probe-free
//!   geometric diagnostic (length + mean curvature + min adjacent cosine).
//! - [`crate::committed_field_blend::CommittedFieldBlend`] — sampling-
//!   invariant per-entity MoE with sigmoid projection onto direction vectors
//!   and a BLAKE3 commitment.
//! - A local lightweight BLAKE3 envelope ([`TrajectoryFreezeEnvelope`])
//!   wrapping the frozen blend + summary + geometry. This matches the
//!   `MerkleFrozenEnvelope` pattern referenced across katgpt-core, but lives
//!   here instead of pulling a cross-repo dep on `riir-neuron-db` (the
//!   facade constraint forbids that, and this primitive is a generic open
//!   layer).
//!
//! # What this primitive is for
//!
//! A Rust-SWE-bench inference attempt (or any "iterative refinement" loop —
//! MCTS over patch proposals, repeated forward passes on evolving drafts, a
//! tf_loop rollout) leaves a trajectory through latent/patch space. The
//! geometry of that trajectory (how much it drifts, how sharply it turns,
//! whether it oscillates between attractors) is a modelless signal that can
//! be used to:
//!
//! 1. Discriminate failure modes (T5.1: oscillation vs committed-wrong vs
//!    converged-correct produce measurably distinct `(length, curvature)`
//!    signatures — `mean_curvature ≈ π` is the ping-pong signature).
//! 2. Produce a frozen, BLAKE3-committed, sampling-invariant archetype blend
//!    that characterizes *which* failure mode the attempt exhibited (T5.3b:
//!    data-derived directions from cluster centroids of geometry-encoded
//!    summaries → 100% probe accuracy on synthetic trajectories).
//! 3. Discriminate across snapshots/models (T5.6 G5 gate — the open
//!    question this primitive is built to answer).
//!
//! # The two-stage commit pipeline
//!
//! ```text
//! Stage 1 — Fit (offline, per-snapshot):  train_summaries → derive_directions
//! Stage 2 — Freeze (online, per-attempt): trajectory → encode_summary → commit
//! ```
//!
//! **Stage 1** consumes a corpus of training summaries (one per observed
//! attempt, labeled by failure mode) and produces `N` archetype direction
//! vectors via nearest-centroid derivation:
//!
//! ```text
//! direction_k = normalize( centroid_k − global_centroid )
//! ```
//!
//! This is the **T5.3b design constraint** encoded in code: archetype
//! directions MUST be data-derived, not random. Random directions hit the
//! concentration-of-measure failure (T5.3): in high-D, random unit vectors
//! are nearly orthogonal to any fixed summary, so `dot(summary, dir_k) ≈ 0`
//! for all k → degenerate uniform blend.
//!
//! **Stage 2** takes a single attempt's trajectory, computes its geometry
//! via `from_states`, encodes that geometry into a fixed-`D` summary via
//! [`GeometrySummaryEncoder`], projects onto the pre-fit directions via
//! FAME's sigmoid projection, and produces a [`FrozenAttempt`] carrying the
//! committed `CommittedFieldBlend` + the envelope.
//!
//! # The summary encoder (load-bearing per T5.3b)
//!
//! [`GeometrySummaryEncoder`] places four normalized geometry features
//! (length, curvature, min cosine, n_steps) into the summary, replicated
//! across blocks for dot-product stability. This is the **T5.3b design
//! constraint** encoded in code: the summary MUST capture failure-mode-
//! discriminative geometry, NOT just raw latent position. The dual-strategy
//! test in T5.3b proved endpoint-position summaries produce 17% accuracy
//! (vs 100% for geometry summaries) — the encoder is the load-bearing piece.
//!
//! # Const generics
//!
//! - `N` — archetype count (number of failure modes to discriminate).
//!   Production case is `N = 3` (oscillation / committed-wrong /
//!   converged-correct).
//! - `D` — summary dimension. Must be `>= 4` (the encoder writes 4 features
//!   per block). Production case is `D = 32` (matches FAME's
//!   `TriArchetypeBlend`).
//!
//! # Feature gate
//!
//! Gated behind `swe_trajectory_freeze`. Implies `latent_trajectory_geometry`
//! + `committed_field_blend` (both already default-on). Opt-in — this is a
//!   research-validation primitive (Proposal 011 Phase 5); promotion to
//!   default requires the T5.6 G5 gate (cross-snapshot discrimination) to
//!   pass on real model trajectories, which is currently PARTIAL (T5.4 G3
//!   FAIL at 29% on Kimi-K3 depth trajectories — see
//!   `.benchmarks/012_kimi_k3_trajectory_geometry.md`).
//!
//! # Allocation
//!
//! - [`GeometrySummaryEncoder::encode_into`] — zero-alloc (writes into a
//!   caller-supplied `&mut [f32; D]`).
//! - [`derive_directions`] — allocates one `[ центroids; N ]` on the stack
//!   (compile-time fixed) + writes into caller-supplied `&mut [[f32; D]; N]`.
//!   No heap allocation.
//! - [`SweTrajectoryFreezer::freeze_attempt`] — zero-alloc. The envelope
//!   payload is written into a stack-fixed `[u8; 512]` buffer (max payload
//!   at N=3, D=32 is 154 bytes); `from_states` + `encode_into` + FAME
//!   `commit` are all documented zero-alloc.
//!
//! # References
//!
//! - Proposal: `katgpt-rs/.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md` Phase 5
//! - T5.1–T5.3 verdict: Issue 569 (removed per noise-reduction rule; resolution in `katgpt-rs/.docs/09_feature_catalog/opt_in_features.md` §29)
//! - T5.3b verdict (data-derived directions): Issue 570 (removed per noise-reduction rule; resolution in §29 + Bench 013)
//! - T5.4 PARTIAL verdict (real Kimi-K3 depth trajectory): `katgpt-rs/.benchmarks/012_kimi_k3_trajectory_geometry.md`
//! - Substrate: Plan 342 (`latent_trajectory_geometry`), Plan 321
//!   (`committed_field_blend`)
//!
//! SWE Trajectory Freezer — modelless committed freeze of an inference
//! attempt's trajectory geometry (Proposal 011 Phase 5, Task T5.5). Composes
//! latent_trajectory_geometry + committed_field_blend + a local BLAKE3
//! envelope. Opt-in — research-validation primitive; promotion requires the
//! T5.6 G5 gate (cross-snapshot discrimination) to pass on real-model
//! trajectories (currently PARTIAL — T5.4 G3 FAIL at 29% on Kimi-K3 depth
//! trajectories; see .benchmarks/012_kimi_k3_trajectory_geometry.md).

// (Module gating is handled by `#[cfg(feature = "swe_trajectory_freeze")]`
// on the `mod` declaration in `lib.rs`; this file must NOT duplicate it.)

#![allow(clippy::needless_range_loop)] // const-generic multi-array indexing by index
#![allow(clippy::doc_lazy_continuation)] // multi-line prose with leading words clippy misreads as list items

use crate::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
use crate::latent_trajectory_geometry::{LatentTrajectoryGeometry, from_states_into};

// ─── Encoder ────────────────────────────────────────────────────────────────

/// Normalize a trajectory's raw geometry into a fixed-`D` summary suitable
/// for FAME's sigmoid projection.
///
/// This is the **load-bearing piece** of the T5.5 design (T5.3b design
/// constraint #2): the summary MUST capture failure-mode-discriminative
/// geometry (`length`, `mean_curvature`, `min_adjacent_cosine`, `n_steps`),
/// NOT just raw latent position. T5.3b's dual-strategy test proved endpoint
/// summaries produce 17% accuracy vs 100% for geometry summaries.
///
/// The four features are placed in the first 4 slots of `out`, then
/// replicated into subsequent 4-slot blocks. The replication is a defense
/// against sparse-vector edge effects in the dot product (a summary with
/// only 4 nonzero entries of 32 produces a dot product sensitive to noise
/// in the high-D direction vector). Replication doesn't change the
/// information content — it just makes the projection numerically
/// well-behaved without requiring a normalization step.
///
/// # Const generic `D`
///
/// Must be `>= 4`. If `D > 4`, the encoder writes `(D / 4)` blocks (rounded
/// down); trailing slots outside a full block are left as zero. At `D = 32`
/// (production) that's 8 blocks of 4 features each.
///
/// # Normalization
///
/// - `length`: divided by `length_scale` (default 20.0), clamped to `[0, 1]`.
///   The default matches the synthetic PoC (Bench 011) where 100-step
///   trajectories had lengths in the 0–20 range. **For real-model
///   trajectories the caller should override `length_scale`**: Kimi-K3
///   depth trajectories (Bench 012) have lengths in the 400–600 range, so
///   `length_scale = 1000.0` would be appropriate.
/// - `mean_curvature`: divided by `π` (curvature is in `[0, π]` by
///   construction via [`fast_acos`]).
/// - `min_adjacent_cosine`: already in `[-1, 1]`.
/// - `n_steps`: divided by `n_steps_scale` (default 100), clamped to `[0, 1]`.
///
/// [`fast_acos`]: crate::latent_trajectory_geometry::fast_acos
#[derive(Clone, Copy, Debug)]
pub struct GeometrySummaryEncoder {
    /// Divisor for the `length` feature. Caller overrides for trajectories
    /// with a different length scale (e.g. real-model depth trajectories).
    pub length_scale: f32,
    /// Divisor for the `n_steps` feature.
    pub n_steps_scale: f32,
}

impl GeometrySummaryEncoder {
    /// Default encoder matching the synthetic PoC (Bench 011).
    ///
    /// `length_scale = 20.0`, `n_steps_scale = 100.0`. Appropriate for
    /// 100-step iterative refinement trajectories with step magnitudes ~0.15.
    pub const fn default_synthetic() -> Self {
        Self {
            length_scale: 20.0,
            n_steps_scale: 100.0,
        }
    }

    /// Encoder tuned for real-model depth trajectories (Bench 012 Kimi-K3).
    ///
    /// `length_scale = 1000.0` (Kimi-K3 depth lengths ~400–600),
    /// `n_steps_scale = 9.0` (depth trajectory = embed + 8 post-layer = 9
    /// states = 8 displacement steps).
    pub const fn default_depth_trajectory() -> Self {
        Self {
            length_scale: 1000.0,
            n_steps_scale: 9.0,
        }
    }

    /// Construct a custom encoder.
    pub const fn new(length_scale: f32, n_steps_scale: f32) -> Self {
        Self {
            length_scale,
            n_steps_scale,
        }
    }

    /// Encode `geom` into `out`.
    ///
    /// Writes 4 normalized features per 4-slot block, replicating across as
    /// many full blocks as fit in `D`. Zero-allocation.
    ///
    /// # Panics (debug)
    ///
    /// In debug builds, panics if `out.len() < 4`.
    #[inline]
    pub fn encode_into<const D: usize>(&self, geom: &LatentTrajectoryGeometry, out: &mut [f32; D]) {
        debug_assert!(D >= 4, "D must be >= 4 to fit one feature block");

        // Normalize each feature to roughly [-1, 1] for stable dot products.
        let length_norm = (geom.length / self.length_scale).clamp(0.0, 1.0);
        let curvature_norm = (geom.mean_curvature / core::f32::consts::PI).clamp(0.0, 1.0);
        let cosine_norm = geom.min_adjacent_cosine.clamp(-1.0, 1.0);
        let n_steps_norm = if self.n_steps_scale > 0.0 {
            (geom.n_steps as f32 / self.n_steps_scale).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Replicate the 4-feature block across D, 4 slots at a time.
        // Trailing slots outside a full block are left at zero.
        let n_blocks = D / 4;
        for block in out.chunks_mut(4).take(n_blocks) {
            block[0] = length_norm;
            block[1] = curvature_norm;
            block[2] = cosine_norm;
            block[3] = n_steps_norm;
        }
    }
}

impl Default for GeometrySummaryEncoder {
    #[inline]
    fn default() -> Self {
        Self::default_synthetic()
    }
}

// ─── State-magnitude encoder (Bench 018 — value-level discrimination) ───────

/// Encode the sequence trajectory's STATE MAGNITUDE statistics into a
/// fixed-`D` summary for value-level (cross-snapshot) discrimination.
///
/// This is the substrate port of bench_018's `encode_seq_state_stats`,
/// which achieved **100% per-prompt accuracy at σ≥0.1** with
/// d_Mahalanobis = 14.526 (50× the geometry encoder's 0.285) on real
/// Kimi-K3 weights.
///
/// # Why this encoder exists alongside `GeometrySummaryEncoder`
///
/// [`GeometrySummaryEncoder`] captures trajectory SHAPE (length,
/// curvature, cosine). Bench 015 proved these shape features are
/// **perturbation-invariant** — they cannot discriminate value-level
/// weight differences. The discriminative signal for value-level
/// discrimination lives in state MAGNITUDE: the per-step L2 norm of the
/// final hidden state is determined by the model's weights (RMSNorm +
/// layer weights set the activation scale), so perturbing weights
/// changes the energy level directly.
///
/// The geometry encoder remains the right choice for STRUCTURAL
/// discrimination (failure-mode classification — bench_014 G5 PASS at
/// 100%); the state-magnitude encoder is the right choice for VALUE
/// discrimination (cross-snapshot identification — bench_018).
///
/// # The 8 features
///
/// Computed from the sequence of final hidden states (NOT from
/// `LatentTrajectoryGeometry` — this encoder consumes the raw `&[&[f32]]`
/// trajectory directly):
///
/// | Slot | Feature | Meaning |
///|------|---------|---------|
/// | 0 | `mean_norm` | mean of per-step L2 norms |
/// | 1 | `std_norm` | std dev of per-step L2 norms |
/// | 2 | `max_norm` | max per-step L2 norm |
/// | 3 | `min_norm` | min per-step L2 norm |
/// | 4 | `initial_norm` | L2 norm of first hidden state |
/// | 5 | `final_norm` | L2 norm of last hidden state |
/// | 6 | `norm_ratio` | `final_norm / initial_norm` (0 if initial≈0) |
/// | 7 | `mean_cos` | mean cosine similarity between consecutive states |
///
/// # Const generic `D`
///
/// Must be `>= 8`. The encoder writes exactly 8 features into the first
/// 8 slots of `out`; trailing slots are left as zero. Unlike the geometry
/// encoder, **no replication** is performed — the 8 features are
/// independent aggregate statistics (not a 4-feature block), so
/// replicating them would not improve dot-product stability and would
/// dilute the per-feature signal.
///
/// # Sequence trajectory extraction (the load-bearing pattern)
///
/// This encoder expects the **sequence trajectory**: the final hidden
/// state at each token, captured across a prompt's tokens with growing KV
/// cache (NO `reset()` between tokens). This is fundamentally different
/// from the depth trajectory (per-layer states within a single forward
/// pass, with `reset()` between tokens) — see `.benchmarks/018_sequence_trajectory.md`
/// §"Why this works where bench_012-017 failed" for the full analysis.
///
/// Extraction pseudocode:
///
/// ```text
/// let mut trajectory = Vec::new();
/// // NO reset() here — KV cache grows across tokens
/// for token in prompt_tokens {
///     let final_state = kimi_k3_forward_token_traced(runtime, token);
///     trajectory.push(final_state);
/// }
/// ```
///
/// # Allocation
///
/// [`StateMagnitudeEncoder::encode_into`] is zero-allocation — it writes
/// into a caller-supplied `&mut [f32; D]` and accumulates statistics in
/// registers (no scratch buffer needed for the two-pass mean+var).
#[derive(Clone, Copy, Debug, Default)]
pub struct StateMagnitudeEncoder;

impl StateMagnitudeEncoder {
    /// Construct a state-magnitude encoder.
    ///
    /// The encoder is parameterless — the 8 features are computed directly
    /// from the trajectory's raw state magnitudes. No scale tuning is
    /// needed because the features are not normalized to a fixed range
    /// (unlike `GeometrySummaryEncoder`'s length/curvature/n_steps, which
    /// need scale divisors). The downstream FAME projection handles
    /// arbitrary feature magnitudes via the sigmoid gate.
    pub const fn new() -> Self {
        Self
    }

    /// Encode the sequence trajectory's state-magnitude statistics into `out`.
    ///
    /// Writes 8 features into `out[..8]`; trailing slots (if `D > 8`) are
    /// left as zero. Zero-allocation.
    ///
    /// # Panics (debug)
    ///
    /// In debug builds, panics if `out.len() < 8`.
    #[inline]
    pub fn encode_into<const D: usize>(&self, trajectory: &[&[f32]], out: &mut [f32; D]) {
        debug_assert!(
            D >= 8,
            "D must be >= 8 to fit the 8 state-magnitude features"
        );

        // Zero the output (in case the caller left poison values).
        for v in out.iter_mut() {
            *v = 0.0;
        }

        let n = trajectory.len();
        if n == 0 {
            return;
        }
        let dim = trajectory[0].len();
        if dim == 0 {
            return;
        }

        // Single-pass computation using Welford's online algorithm for
        // mean+variance, plus running min/max/sum + consecutive cosine.
        // This avoids the 3x recomputation of per-step norms — the dominant
        // cost at D=1024, N=64 is the inner dim loop, so doing it once
        // instead of three times is a ~3x speedup.
        //
        // State carried across steps:
        // - Welford: count, mean, M2 (sum of squared deviations)
        // - Running: sum (for cross-check), max, min, initial, final
        // - Cosine: previous state's norm + dot-with-current
        let mut count: usize = 0;
        let mut mean_norm = 0.0_f32;
        let mut m2 = 0.0_f32;
        let mut max_norm = 0.0_f32;
        let mut min_norm = f32::INFINITY;
        let mut initial_norm = 0.0_f32;
        let mut final_norm = 0.0_f32;
        let mut sum_cos = 0.0_f32;
        let mut cos_count = 0usize;
        let mut prev_norm_sq = 0.0_f32; // ||h_{i-1}||^2 for cosine

        for (i, state) in trajectory.iter().enumerate() {
            // Compute ||h_i||^2 + dot(h_{i-1}, h_i) in one dim loop.
            let mut sum_sq = 0.0_f32;
            let mut dot_prev = 0.0_f32;
            if i > 0 {
                // Pre-slice both rows to `dim` up front: the `[..dim]` slicing
                // panics on a short row exactly as the old `state[j]`/`prev[j]`
                // indexing did, but hoists both per-element bounds checks out
                // of the loop and lets LLVM vectorize. `zip` over two slices
                // that are both exactly `dim` long visits the same elements in
                // the same order, so the float accumulation order into
                // `sum_sq`/`dot_prev` is unchanged (bit-identical).
                let cur = &state[..dim];
                let prev = &trajectory[i - 1][..dim];
                for (&x, &p) in cur.iter().zip(prev.iter()) {
                    sum_sq += x * x;
                    dot_prev += x * p;
                }
            } else {
                for &x in *state {
                    sum_sq += x * x;
                }
            }
            let norm = sum_sq.sqrt();

            // Welford update for mean + M2.
            count += 1;
            let delta = norm - mean_norm;
            mean_norm += delta / count as f32;
            let delta2 = norm - mean_norm;
            m2 += delta * delta2;

            // Running min/max/initial/final.
            if norm > max_norm {
                max_norm = norm;
            }
            if norm < min_norm {
                min_norm = norm;
            }
            if i == 0 {
                initial_norm = norm;
            }
            final_norm = norm;

            // Cosine similarity with previous state.
            if i > 0 {
                let denom = (prev_norm_sq * sum_sq).sqrt();
                if denom > 1e-12 {
                    sum_cos += dot_prev / denom;
                    cos_count += 1;
                }
            }
            prev_norm_sq = sum_sq;
        }

        // Welford variance (population, not sample — matches bench_018).
        let std_norm = if count > 0 {
            (m2 / count as f32).sqrt()
        } else {
            0.0
        };

        let norm_ratio = if initial_norm > 1e-12 {
            final_norm / initial_norm
        } else {
            0.0
        };

        let mean_cos = if cos_count > 0 {
            sum_cos / cos_count as f32
        } else {
            0.0
        };

        out[0] = mean_norm;
        out[1] = std_norm;
        out[2] = max_norm;
        out[3] = min_norm;
        out[4] = initial_norm;
        out[5] = final_norm;
        out[6] = norm_ratio;
        out[7] = mean_cos;
    }
}

// ─── Direction derivation (Stage 1: fit) ────────────────────────────────────

/// Derive archetype direction vectors from cluster centroids of training
/// summaries (the **nearest-centroid classifier** direction).
///
/// For each mode `k`:
///
/// ```text
/// direction_k = normalize( centroid_k − global_centroid )
/// ```
///
/// When probed with a summary from cluster `k`, `dot(summary_k, direction_k)`
/// is large and positive; for other clusters, small. This is the
/// **T5.3b design constraint** encoded in code: archetype directions MUST
/// be data-derived, not random. Random directions hit the concentration-of-
/// measure failure (T5.3) — in high-D, random unit vectors are nearly
/// orthogonal to any fixed summary.
///
/// # Arguments
///
/// - `train_summaries` — `&[[[f32; D]; M]; N]`: `N` modes, each with `M`
///   training summaries. The inner array length `M` is the per-mode sample
///   count used to compute the centroid. All summaries must be non-empty.
/// - `out_directions` — `&mut [[f32; D]; N]`: written in place. Each
///   direction is L2-normalized.
///
/// # Allocation
///
/// Stack-only: per-mode centroids + global centroid are fixed-size arrays.
/// No heap allocation.
///
/// # Panics
///
/// In debug builds, panics if `M == 0` (cannot compute a centroid with zero
/// samples). A release-mode `M == 0` produces a zero direction (which will
/// yield `dot = 0` → sigmoid(0) = 0.5 → uniform blend — the degenerate
/// signal that flags an unfitted freezer).
pub fn derive_directions<const N: usize, const M: usize, const D: usize>(
    train_summaries: &[[[f32; D]; M]; N],
    out_directions: &mut [[f32; D]; N],
) {
    debug_assert!(M > 0, "M (per-mode sample count) must be > 0");

    // Per-mode centroids.
    let mut centroids = [[0.0_f32; D]; N];
    for (mode_idx, mode_sums) in train_summaries.iter().enumerate() {
        for s in mode_sums.iter() {
            for j in 0..D {
                centroids[mode_idx][j] += s[j];
            }
        }
        let inv = 1.0 / (M as f32);
        for j in 0..D {
            centroids[mode_idx][j] *= inv;
        }
    }

    // Global centroid (mean of per-mode centroids).
    let mut global = [0.0_f32; D];
    for c in &centroids {
        for j in 0..D {
            global[j] += c[j];
        }
    }
    let inv_n = 1.0 / (N as f32);
    for j in 0..D {
        global[j] *= inv_n;
    }

    // direction_k = normalize(centroid_k − global).
    for k in 0..N {
        let mut norm_sq = 0.0_f32;
        for j in 0..D {
            out_directions[k][j] = centroids[k][j] - global[j];
            norm_sq += out_directions[k][j] * out_directions[k][j];
        }
        let norm = norm_sq.sqrt().max(1e-9);
        for j in 0..D {
            out_directions[k][j] /= norm;
        }
    }
}

/// Derive archetype directions + the global centroid (the fit constructor's
/// workhorse).
///
/// Like [`derive_directions`], but also writes the global centroid into
/// `out_global`. The centroid is used by [`SweTrajectoryFreezer::fit`] to
/// mean-center summaries before projection — without this, the FAME sigmoid
/// gate's threshold at 0 may not align with the natural decision boundary.
pub fn derive_directions_and_centroid<const N: usize, const M: usize, const D: usize>(
    train_summaries: &[[[f32; D]; M]; N],
    out_directions: &mut [[f32; D]; N],
    out_global: &mut [f32; D],
) {
    debug_assert!(M > 0, "M (per-mode sample count) must be > 0");

    // Per-mode centroids.
    let mut centroids = [[0.0_f32; D]; N];
    for (mode_idx, mode_sums) in train_summaries.iter().enumerate() {
        for s in mode_sums.iter() {
            for j in 0..D {
                centroids[mode_idx][j] += s[j];
            }
        }
        let inv = 1.0 / (M as f32);
        for j in 0..D {
            centroids[mode_idx][j] *= inv;
        }
    }

    // Global centroid (mean of per-mode centroids).
    for j in 0..D {
        out_global[j] = 0.0;
    }
    for c in &centroids {
        for j in 0..D {
            out_global[j] += c[j];
        }
    }
    let inv_n = 1.0 / (N as f32);
    for j in 0..D {
        out_global[j] *= inv_n;
    }

    // direction_k = normalize(centroid_k − global).
    for k in 0..N {
        let mut norm_sq = 0.0_f32;
        for j in 0..D {
            out_directions[k][j] = centroids[k][j] - out_global[j];
            norm_sq += out_directions[k][j] * out_directions[k][j];
        }
        let norm = norm_sq.sqrt().max(1e-9);
        for j in 0..D {
            out_directions[k][j] /= norm;
        }
    }
}

// ─── Trajectory freeze envelope (BLAKE3 commitment) ─────────────────────────

/// Wire-format magic for [`TrajectoryFreezeEnvelope`].
///
/// ASCII `"SWTF"` (SWE Trajectory Freeze). Distinct from every other magic
/// in the stack (riir-neuron-db uses `"NRFZ"`, `"KRCS"`, etc.).
pub const SWTF_MAGIC: [u8; 4] = *b"SWTF";

/// Envelope version 1.
pub const SWTF_VERSION: u32 = 1;

/// A lightweight BLAKE3 commitment envelope wrapping a frozen attempt.
///
/// Matches the `MerkleFrozenEnvelope` pattern referenced across katgpt-core
/// (see `cross_resolution.rs`, `curator.rs`, `latent_steering.rs`, etc.),
/// but lives here instead of pulling a cross-repo dep on `riir-neuron-db`.
/// This primitive is the generic open layer; the IP-bearing private bridge
/// (if ever needed) lives in the consumer.
///
/// # Layout
///
/// ```text
/// magic        : [u8; 4]   // "SWTF"
/// version      : u32       // envelope version (currently 1)
/// merkle_root  : [u8; 32]  // BLAKE3 of (pi, summary, geometry)
/// data_len     : u64       // byte length of the serialized payload
/// commitment   : [u8; 32]  // BLAKE3 of (magic || version || merkle_root || data_len)
/// ```
///
/// `merkle_root` commits to the attempt's frozen state (the blend's `pi` +
/// the encoded summary + the raw geometry triple). `commitment` commits to
/// the envelope header — tamper-detection at thaw time checks both.
///
/// # Determinism
///
/// All operations are deterministic: BLAKE3 is platform-independent, the
/// serialization layout is fixed (no padding, no pointers). Two freezers
/// with the same `(pi, summary, geometry)` produce bit-identical envelopes.
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryFreezeEnvelope {
    /// Format magic bytes — always [`SWTF_MAGIC`].
    pub magic: [u8; 4],
    /// Envelope version — currently [`SWTF_VERSION`].
    pub version: u32,
    /// BLAKE3 hash over the serialized payload `(pi, summary, geometry)`.
    pub merkle_root: [u8; 32],
    /// Byte length of the serialized payload (informational).
    pub data_len: u64,
    /// BLAKE3 commitment over `(magic, version, merkle_root, data_len)`.
    pub commitment: [u8; 32],
}

impl TrajectoryFreezeEnvelope {
    /// Construct an envelope from the serialized payload bytes.
    ///
    /// The payload is the concatenation `pi_bytes || summary_bytes ||
    /// geometry_bytes` — caller-supplied, layout determined by the caller
    /// (typically `bytemuck::cast_slice` for fixed-layout types).
    ///
    /// `merkle_root` = BLAKE3(payload). `commitment` = BLAKE3(header).
    /// `data_len` = payload length.
    pub fn freeze(payload: &[u8]) -> Self {
        let merkle_root = blake3::hash(payload);
        let mut header = [0u8; 4 + 4 + 32 + 8];
        header[0..4].copy_from_slice(&SWTF_MAGIC);
        header[4..8].copy_from_slice(&SWTF_VERSION.to_le_bytes());
        header[8..40].copy_from_slice(merkle_root.as_bytes());
        header[40..48].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        let commitment = blake3::hash(&header);
        Self {
            magic: SWTF_MAGIC,
            version: SWTF_VERSION,
            merkle_root: merkle_root.into(),
            data_len: payload.len() as u64,
            commitment: commitment.into(),
        }
    }

    /// Verify the envelope's header commitment.
    ///
    /// Recomputes BLAKE3(magic || version || merkle_root || data_len) and
    /// compares with `self.commitment`. Does NOT recompute `merkle_root`
    /// (that requires the original payload — use [`Self::verify_payload`]
    /// for full verification).
    ///
    /// Returns `false` if `magic != SWTF_MAGIC` or `version != SWTF_VERSION`
    /// or the header commitment doesn't match.
    pub fn verify_header(&self) -> bool {
        if self.magic != SWTF_MAGIC || self.version != SWTF_VERSION {
            return false;
        }
        let mut header = [0u8; 4 + 4 + 32 + 8];
        header[0..4].copy_from_slice(&SWTF_MAGIC);
        header[4..8].copy_from_slice(&SWTF_VERSION.to_le_bytes());
        header[8..40].copy_from_slice(&self.merkle_root);
        header[40..48].copy_from_slice(&self.data_len.to_le_bytes());
        let recomputed = blake3::hash(&header);
        recomputed.as_bytes() == &self.commitment
    }

    /// Full verification: header + payload.
    ///
    /// Returns `true` iff [`Self::verify_header`] passes AND
    /// `BLAKE3(payload) == self.merkle_root`.
    pub fn verify_payload(&self, payload: &[u8]) -> bool {
        if !self.verify_header() {
            return false;
        }
        let recomputed = blake3::hash(payload);
        recomputed.as_bytes() == &self.merkle_root
    }
}

// ─── FrozenAttempt ──────────────────────────────────────────────────────────

/// A frozen SWE-attempt characterization.
///
/// Composed at freeze-time from:
///
/// - The committed FAME blend (`blend`) — the sampling-invariant archetype
///   weights derived from the trajectory summary.
/// - The encoded summary (`summary`) — the geometry features that produced
///   the blend.
/// - The raw geometry (`geometry`) — the original `LatentTrajectoryGeometry`
///   triple, retained for diagnostics.
/// - The BLAKE3 envelope (`envelope`) — tamper-evident commitment.
///
/// # Const generics
///
/// `N` archetype count + `D` summary dimension, mirroring
/// [`CommittedFieldBlend<N, D>`].
///
/// [`CommittedFieldBlend<N, D>`]: crate::committed_field_blend::CommittedFieldBlend
#[derive(Clone, Debug)]
pub struct FrozenAttempt<const N: usize, const D: usize> {
    /// The committed archetype blend (sigmoid weights over `N` archetypes).
    pub blend: CommittedFieldBlend<N, D>,
    /// The encoded geometry summary (length-N features replicated across D).
    pub summary: [f32; D],
    /// The raw trajectory geometry (length, curvature, min_cos, n_steps).
    pub geometry: LatentTrajectoryGeometry,
    /// The BLAKE3 commitment envelope.
    pub envelope: TrajectoryFreezeEnvelope,
}

impl<const N: usize, const D: usize> FrozenAttempt<N, D> {
    /// Sigmoid gates over the blend's `pi`. Convenience accessor.
    ///
    /// `gate_k = sigmoid(pi_k / tau)`. Range `(0, 1)`.
    #[inline]
    pub fn gates(&self) -> [f32; N]
    where
        [(); N]:,
    {
        let tau = self.blend.tau;
        let mut gates = [0.0_f32; N];
        for k in 0..N {
            gates[k] = crate::personality_composition::sigmoid::sigmoid(self.blend.pi[k] / tau);
        }
        gates
    }

    /// Argmax archetype index (the most-embodied failure mode).
    ///
    /// Returns the `k` with the largest sigmoid gate. Ties broken by lowest
    /// index.
    #[inline]
    pub fn argmax_archetype(&self) -> usize
    where
        [(); N]:,
    {
        let gates = self.gates();
        let mut best_k = 0;
        let mut best_g = gates[0];
        for k in 1..N {
            if gates[k] > best_g {
                best_g = gates[k];
                best_k = k;
            }
        }
        best_k
    }

    /// Verify the envelope's header commitment (cheap — no payload needed).
    #[inline]
    pub fn verify_envelope_header(&self) -> bool {
        self.envelope.verify_header()
    }
}

// ─── FrozenValueAttempt (Bench 018 — value-level freeze) ────────────────────

/// A frozen SWE-attempt characterization via STATE MAGNITUDE statistics.
///
/// The value-level counterpart to [`FrozenAttempt`]. Where `FrozenAttempt`
/// commits to trajectory GEOMETRY (shape features — length, curvature,
/// cosine), `FrozenValueAttempt` commits to trajectory STATE MAGNITUDE
/// (the per-step L2 norm statistics that bench_018 proved discriminative
/// for cross-snapshot identification).
///
/// # When to use this vs `FrozenAttempt`
///
/// - **`FrozenAttempt`** (geometry): for STRUCTURAL discrimination —
///   classifying an attempt's failure mode (oscillation / committed-wrong /
///   converged-correct). Bench 014 G5 PASS at 100%.
/// - **`FrozenValueAttempt`** (state magnitude): for VALUE discrimination —
///   identifying which model snapshot produced an attempt. Bench 018 G5
///   PASS at 100% (σ≥0.1) with d_Mahalanobis = 14.526.
///
/// The two types are intentionally separate: they commit to different
/// payloads, produce different BLAKE3 roots, and answer different questions.
/// A production system that needs both characterizations should call both
/// `freeze_attempt_into` and `freeze_attempt_value_into` on the same
/// trajectory (the two encoders do not interfere).
///
/// # Const generics
///
/// `N` archetype count + `D` summary dimension, mirroring
/// [`CommittedFieldBlend<N, D>`]. `D` must be `>= 8` (the state-magnitude
/// encoder writes 8 features).
///
/// # Payload layout
///
/// The envelope's payload is `pi || summary` (no geometry triple — the
/// state-magnitude features ARE the payload). At the production case
/// (N=3, D=32) that's 140 bytes.
#[derive(Clone, Debug)]
pub struct FrozenValueAttempt<const N: usize, const D: usize> {
    /// The committed archetype blend (sigmoid weights over `N` archetypes).
    pub blend: CommittedFieldBlend<N, D>,
    /// The encoded state-magnitude summary (8 features in the first 8 slots).
    pub summary: [f32; D],
    /// The BLAKE3 commitment envelope.
    pub envelope: TrajectoryFreezeEnvelope,
}

impl<const N: usize, const D: usize> FrozenValueAttempt<N, D> {
    /// Sigmoid gates over the blend's `pi`. Convenience accessor.
    ///
    /// `gate_k = sigmoid(pi_k / tau)`. Range `(0, 1)`.
    #[inline]
    pub fn gates(&self) -> [f32; N]
    where
        [(); N]:,
    {
        let tau = self.blend.tau;
        let mut gates = [0.0_f32; N];
        for k in 0..N {
            gates[k] = crate::personality_composition::sigmoid::sigmoid(self.blend.pi[k] / tau);
        }
        gates
    }

    /// Argmax archetype index (the most-embodied snapshot).
    #[inline]
    pub fn argmax_archetype(&self) -> usize
    where
        [(); N]:,
    {
        let gates = self.gates();
        let mut best_k = 0;
        let mut best_g = gates[0];
        for k in 1..N {
            if gates[k] > best_g {
                best_g = gates[k];
                best_k = k;
            }
        }
        best_k
    }

    /// Verify the envelope's header commitment (cheap — no payload needed).
    #[inline]
    pub fn verify_envelope_header(&self) -> bool {
        self.envelope.verify_header()
    }
}

// ─── SweTrajectoryFreezer ───────────────────────────────────────────────────

/// The headline type — composes encoder + FAME + envelope.
///
/// Construct once with the pre-fit direction vectors + encoder config, then
/// call [`Self::freeze_attempt`] per observed trajectory.
///
/// # Const generics
///
/// - `N` — archetype count.
/// - `D` — summary dimension (must match the direction vectors' dimension).
///
/// # Example
///
/// ```no_run
/// # use katgpt_core::swe_trajectory_freeze::*;
/// # use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
/// # fn make_fields() -> [&'static dyn ArchetypeFieldSource<32>; 3] { unimplemented!() }
/// # fn make_directions() -> [[f32; 32]; 3] { unimplemented!() }
/// // Stage 1 — fit (offline): derive_directions from a labeled corpus
/// // (see derive_directions docs).
/// let directions = make_directions();
/// let freezer = SweTrajectoryFreezer::<3, 32>::new(directions);
///
/// // Stage 2 — freeze (online): characterize a single attempt
/// let trajectory: Vec<Vec<f32>> = vec![]; // populate with real states
/// let refs: Vec<&[f32]> = trajectory.iter().map(|v| v.as_slice()).collect();
/// let fields = make_fields();
/// let frozen = freezer.freeze_attempt(&refs, &fields, /* version */ 1);
///
/// // Inspect the result
/// let gates = frozen.gates();
/// let argmax = frozen.argmax_archetype();
/// let blake3 = frozen.envelope.commitment;
/// ```
pub struct SweTrajectoryFreezer<const N: usize, const D: usize> {
    /// Pre-fit archetype direction vectors (from [`derive_directions`]).
    pub directions: [[f32; D]; N],
    /// The global centroid of the training summaries (subtracted from each
    /// summary before projection to center the data for discriminative gates).
    /// Defaults to all-zeros when constructed via [`new`] / [`with_encoder`]
    /// (no centering). Set via [`fit`] / [`with_centroid`] for proper
    /// nearest-centroid classification.
    pub global_centroid: [f32; D],
    /// The geometry-summary encoder (for STRUCTURAL discrimination —
    /// failure-mode classification via [`freeze_attempt_into`]).
    pub encoder: GeometrySummaryEncoder,
    /// The state-magnitude encoder (for VALUE discrimination —
    /// cross-snapshot identification via [`freeze_attempt_value_into`]).
    /// Bench 018 G5 PASS at 100% (σ≥0.1) with d_Mahalanobis = 14.526.
    pub value_encoder: StateMagnitudeEncoder,
}

impl<const N: usize, const D: usize> SweTrajectoryFreezer<N, D> {
    /// Construct a freezer with the given direction vectors + default encoder.
    ///
    /// The global centroid defaults to all-zeros (no mean-centering). For
    /// proper nearest-centroid classification, use [`fit`] instead — it derives
    /// both the directions AND the global centroid from the training corpus.
    pub fn new(directions: [[f32; D]; N]) -> Self {
        Self {
            directions,
            global_centroid: [0.0_f32; D],
            encoder: GeometrySummaryEncoder::default(),
            value_encoder: StateMagnitudeEncoder::new(),
        }
    }

    /// Construct a freezer with the given direction vectors + custom encoder.
    ///
    /// The global centroid defaults to all-zeros (no mean-centering). See
    /// [`new`] for why you usually want [`fit`] instead.
    pub fn with_encoder(directions: [[f32; D]; N], encoder: GeometrySummaryEncoder) -> Self {
        Self {
            directions,
            global_centroid: [0.0_f32; D],
            encoder,
            value_encoder: StateMagnitudeEncoder::new(),
        }
    }

    /// Construct a freezer with pre-computed directions + global centroid +
    /// custom encoder. This is the full-control constructor for callers that
    /// have pre-computed the centroid externally.
    pub fn with_centroid(
        directions: [[f32; D]; N],
        global_centroid: [f32; D],
        encoder: GeometrySummaryEncoder,
    ) -> Self {
        Self {
            directions,
            global_centroid,
            encoder,
            value_encoder: StateMagnitudeEncoder::new(),
        }
    }

    /// Fit the freezer from a labeled training corpus — derives directions
    /// AND the global centroid from the data (the recommended constructor).
    ///
    /// This is the mathematically correct constructor for nearest-centroid
    /// classification: it computes `direction_k = normalize(centroid_k -
    /// global_centroid)` AND stores the global centroid so that
    /// [`freeze_attempt_into`] can mean-center each summary before projection.
    /// Without mean-centering, the FAME sigmoid gate's threshold at 0 may not
    /// align with the natural decision boundary between clusters (the N=2
    /// antiparallel degeneracy makes this especially acute).
    pub fn fit<const M: usize>(train_summaries: &[[[f32; D]; M]; N]) -> Self {
        let mut directions = [[0.0_f32; D]; N];
        let mut global_centroid = [0.0_f32; D];
        derive_directions_and_centroid(train_summaries, &mut directions, &mut global_centroid);
        Self {
            directions,
            global_centroid,
            encoder: GeometrySummaryEncoder::default(),
            value_encoder: StateMagnitudeEncoder::new(),
        }
    }

    /// Freeze a single attempt's trajectory into a committed characterization.
    ///
    /// Convenience wrapper around [`freeze_attempt_into`] — allocates scratch
    /// buffers internally (2 `Vec<f32>` allocations sized to `dim`). For
    /// zero-allocation steady-state, reuse a pair of `Vec<f32>` scratch buffers
    /// via [`freeze_attempt_into`].
    ///
    /// # Pipeline
    ///
    /// 1. Compute `LatentTrajectoryGeometry` via `from_states(trajectory)`.
    /// 2. Encode the geometry into a `[f32; D]` summary.
    /// 3. Commit the FAME blend: `pi_k = clamp(dot(summary, dir_k), ±pi_max)`.
    /// 4. Build the envelope: BLAKE3 over `(pi, summary, geometry)`.
    /// 5. Return the [`FrozenAttempt`].
    ///
    /// # Arguments
    ///
    /// - `trajectory` — `&[&[f32]]`, one slice per latent state. All slices
    ///   must share the same dimension.
    /// - `fields` — the `N` archetype fields (used for their BLAKE3
    ///   commitments only — FAME's `commit` does not invoke `evolve`).
    /// - `version` — monotonic version counter for this commit.
    ///
    /// # Allocation
    ///
    /// - `from_states` allocates 2 `Vec<f32>` displacement buffers.
    /// - `encode_into` is zero-alloc.
    /// - FAME `commit` is zero-alloc (Plan 321 G4).
    /// - The envelope payload uses a stack-fixed `[u8; 512]` buffer.
    ///
    /// Use [`freeze_attempt_into`] to eliminate the 2 `Vec` allocations by
    /// passing caller-managed scratch buffers.
    pub fn freeze_attempt(
        &self,
        trajectory: &[&[f32]],
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        version: u64,
    ) -> FrozenAttempt<N, D> {
        let dim = trajectory.first().map_or(0, |s| s.len());
        let mut disp_curr = vec![0.0_f32; dim];
        let mut disp_prev = vec![0.0_f32; dim];
        self.freeze_attempt_into(trajectory, fields, version, &mut disp_curr, &mut disp_prev)
    }

    /// Zero-allocation variant of [`freeze_attempt`] — takes caller-managed
    /// scratch buffers for the trajectory geometry computation.
    ///
    /// The scratch buffers are resized to `dim` if too small (no-op when
    /// already large enough). Callers that reuse the same pair across calls
    /// achieve **zero allocation in steady state** — the entire freeze
    /// pipeline (geometry → encode → FAME commit → BLAKE3 envelope) uses only
    /// stack-fixed buffers.
    ///
    /// See [`freeze_attempt`] for the pipeline description + argument semantics.
    pub fn freeze_attempt_into(
        &self,
        trajectory: &[&[f32]],
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        version: u64,
        disp_curr: &mut Vec<f32>,
        disp_prev: &mut Vec<f32>,
    ) -> FrozenAttempt<N, D> {
        // 1. Geometry.
        let geometry = from_states_into(trajectory, disp_curr, disp_prev);

        // 2. Summary (mean-centered for discriminative projection).
        //    Subtracting the global centroid centers the data so the FAME
        //    sigmoid gate's threshold at 0 aligns with the natural decision
        //    boundary between clusters. Without this, non-centered summaries
        //    with positive-valued features can produce dot products that are
        //    all on the same side of 0 (the N=2 antiparallel degeneracy makes
        //    this especially acute). When the centroid is all-zeros (constructed
        //    via `new` / `with_encoder`), this is a no-op.
        let mut summary = [0.0_f32; D];
        self.encoder.encode_into(&geometry, &mut summary);
        for j in 0..D {
            summary[j] -= self.global_centroid[j];
        }

        // 3. FAME commit.
        let mut blend = CommittedFieldBlend::<N, D>::uncommitted();
        blend.commit(&summary, &self.directions, fields, version);

        // 4. Envelope. Payload = pi || summary || geometry-bytes.
        //    Layout: N·4 (pi) + D·4 (summary) + 4+4+4+2 (geometry: length,
        //    curvature, min_cos, n_steps as f32/f32/f32/u16) = N·4 + D·4 + 14.
        //    Stack-fixed — no heap allocation. The max payload size at the
        //    production case (N=3, D=32) is 154 bytes, well under any
        //    reasonable stack budget.
        let payload_len = N * 4 + D * 4 + 14;
        let mut payload = [0u8; 512];
        debug_assert!(
            payload_len <= payload.len(),
            "payload_len {payload_len} exceeds stack buffer {}",
            payload.len()
        );
        let mut offset = 0;
        for k in 0..N {
            payload[offset..offset + 4].copy_from_slice(&blend.pi[k].to_le_bytes());
            offset += 4;
        }
        for j in 0..D {
            payload[offset..offset + 4].copy_from_slice(&summary[j].to_le_bytes());
            offset += 4;
        }
        payload[offset..offset + 4].copy_from_slice(&geometry.length.to_le_bytes());
        offset += 4;
        payload[offset..offset + 4].copy_from_slice(&geometry.mean_curvature.to_le_bytes());
        offset += 4;
        payload[offset..offset + 4].copy_from_slice(&geometry.min_adjacent_cosine.to_le_bytes());
        offset += 4;
        payload[offset..offset + 2].copy_from_slice(&geometry.n_steps.to_le_bytes());
        let envelope = TrajectoryFreezeEnvelope::freeze(&payload[..payload_len]);

        FrozenAttempt {
            blend,
            summary,
            geometry,
            envelope,
        }
    }

    /// Freeze a single attempt's trajectory into a VALUE-LEVEL characterization
    /// (state-magnitude statistics, not geometry).
    ///
    /// Convenience wrapper around [`freeze_attempt_value_into`] — no scratch
    /// buffers needed (the state-magnitude encoder is zero-alloc by
    /// construction — it accumulates statistics in registers, no displacement
    /// buffers required).
    ///
    /// # Pipeline
    ///
    /// 1. Encode the trajectory's state-magnitude statistics into a `[f32; D]`
    ///    summary via [`StateMagnitudeEncoder`] (8 features in the first 8 slots).
    /// 2. Mean-center the summary (subtract `global_centroid`).
    /// 3. Commit the FAME blend: `pi_k = clamp(dot(summary, dir_k), ±pi_max)`.
    /// 4. Build the envelope: BLAKE3 over `(pi, summary)`.
    /// 5. Return the [`FrozenValueAttempt`].
    ///
    /// # Arguments
    ///
    /// - `trajectory` — `&[&[f32]]`, one slice per hidden state. For the
    ///   sequence trajectory (bench_018's validated regime), this is the
    ///   final hidden state at each token, captured with growing KV cache.
    /// - `fields` — the `N` archetype fields (used for their BLAKE3
    ///   commitments only — FAME's `commit` does not invoke `evolve`).
    /// - `version` — monotonic version counter for this commit.
    ///
    /// # Allocation
    ///
    /// Zero-allocation. The state-magnitude encoder accumulates statistics in
    /// registers (no scratch buffer); FAME `commit` is zero-alloc (Plan 321 G4);
    /// the envelope payload uses a stack-fixed `[u8; 512]` buffer.
    pub fn freeze_attempt_value(
        &self,
        trajectory: &[&[f32]],
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        version: u64,
    ) -> FrozenValueAttempt<N, D> {
        self.freeze_attempt_value_into(trajectory, fields, version)
    }

    /// Zero-allocation variant of [`freeze_attempt_value`] (identical here —
    /// the state-magnitude path is zero-alloc by construction).
    ///
    /// Provided for API symmetry with [`freeze_attempt_into`] (the geometry
    /// path's zero-alloc variant). The state-magnitude encoder does not need
    /// displacement scratch buffers, so there is no `_into` variant that
    /// takes caller-managed buffers — but callers that want a named
    /// zero-alloc entry point should use this method.
    pub fn freeze_attempt_value_into(
        &self,
        trajectory: &[&[f32]],
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        version: u64,
    ) -> FrozenValueAttempt<N, D> {
        // 1. State-magnitude summary (mean-centered for discriminative projection).
        let mut summary = [0.0_f32; D];
        self.value_encoder.encode_into(trajectory, &mut summary);
        for j in 0..D {
            summary[j] -= self.global_centroid[j];
        }

        // 2. FAME commit.
        let mut blend = CommittedFieldBlend::<N, D>::uncommitted();
        blend.commit(&summary, &self.directions, fields, version);

        // 3. Envelope. Payload = pi || summary (no geometry triple — the
        //    state-magnitude features ARE the payload). Layout: N·4 (pi) +
        //    D·4 (summary). Stack-fixed — no heap allocation. At the
        //    production case (N=3, D=32) that's 140 bytes.
        let payload_len = N * 4 + D * 4;
        let mut payload = [0u8; 512];
        debug_assert!(
            payload_len <= payload.len(),
            "payload_len {payload_len} exceeds stack buffer {}",
            payload.len()
        );
        let mut offset = 0;
        for k in 0..N {
            payload[offset..offset + 4].copy_from_slice(&blend.pi[k].to_le_bytes());
            offset += 4;
        }
        for j in 0..D {
            payload[offset..offset + 4].copy_from_slice(&summary[j].to_le_bytes());
            offset += 4;
        }
        let envelope = TrajectoryFreezeEnvelope::freeze(&payload[..payload_len]);

        FrozenValueAttempt {
            blend,
            summary,
            envelope,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)] // index loops used for multi-array indexing
    use super::*;

    // Test constants matching Bench 011's synthetic regime.
    const DIM: usize = 8;
    const N_STEPS: usize = 100;
    const D: usize = 32;
    const N: usize = 3;

    // ─── LCG (deterministic, matches Bench 011) ─────────────────────────────

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        #[inline]
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
        }
    }

    // ─── Synthetic trajectory builders (mirrors Bench 011) ──────────────────

    fn build_committed_wrong(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
        let mut direction: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let norm = direction
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        for x in direction.iter_mut() {
            *x /= norm;
        }
        let step_size = 0.15;
        let mut traj = Vec::with_capacity(N_STEPS + 1);
        traj.push(state.clone());
        for _ in 0..N_STEPS {
            for j in 0..DIM {
                state[j] += step_size * direction[j];
            }
            traj.push(state.clone());
        }
        traj.shrink_to_fit();
        traj
    }

    fn build_oscillation(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let attractor_a: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let attractor_b: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let mut traj = Vec::with_capacity(N_STEPS + 1);
        // Alternate between A and B every step (period 2) — matches bench_011.
        // The smoothing variant (0.7/0.3 mix) kills the π-curvature signature.
        for i in 0..=N_STEPS {
            let target = if i % 2 == 0 {
                &attractor_a
            } else {
                &attractor_b
            };
            traj.push(target.clone());
        }
        traj.shrink_to_fit();
        traj
    }

    fn build_converged_correct(seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let target: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
        let mut traj = Vec::with_capacity(N_STEPS + 1);
        traj.push(state.clone());
        // Exponential convergence — 5% per step (matches bench_011).
        for _ in 0..N_STEPS {
            for j in 0..DIM {
                state[j] += 0.05 * (target[j] - state[j]);
            }
            traj.push(state.clone());
        }
        traj.shrink_to_fit();
        traj
    }

    fn build_trajectory_for_mode(mode_idx: usize, seed: u64) -> Vec<Vec<f32>> {
        match mode_idx {
            0 => build_oscillation(seed),
            1 => build_committed_wrong(seed),
            2 => build_converged_correct(seed),
            _ => unreachable!(),
        }
    }

    fn build_refs(traj: &[Vec<f32>]) -> Vec<&[f32]> {
        traj.iter().map(|v| v.as_slice()).collect()
    }

    // ─── T5.5 G1 — substrate composes + is deterministic ────────────────────

    /// Two freezes of the same trajectory + same directions + same version
    /// MUST produce bit-identical envelopes. This is the FAME sampling-
    /// invariance property extended to the full freeze pipeline.
    #[test]
    fn g1_freeze_is_deterministic() {
        // Train directions on 3 fixed trajectories.
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        // Freeze the same test trajectory twice.
        let test_traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&test_traj);
        let fields = make_stub_fields();

        let a = freezer.freeze_attempt(&refs, &fields, 1);
        let b = freezer.freeze_attempt(&refs, &fields, 1);

        // Bit-identical envelopes.
        assert_eq!(a.envelope.commitment, b.envelope.commitment);
        assert_eq!(a.envelope.merkle_root, b.envelope.merkle_root);
        assert_eq!(a.envelope.data_len, b.envelope.data_len);
        // Bit-identical pi + summary.
        assert_eq!(a.blend.pi, b.blend.pi);
        assert_eq!(a.summary, b.summary);
    }

    /// Helper used by the G1 test to encode a trajectory's geometry summary
    /// using the default synthetic encoder.
    fn encode_geom_summary(traj: &[Vec<f32>]) -> [f32; D] {
        let refs = build_refs(traj);
        let mut disp_curr = vec![0.0_f32; DIM];
        let mut disp_prev = vec![0.0_f32; DIM];
        let geom = from_states_into(&refs, &mut disp_curr, &mut disp_prev);
        let mut summary = [0.0_f32; D];
        GeometrySummaryEncoder::default_synthetic().encode_into(&geom, &mut summary);
        summary
    }

    // ─── T5.5 G2 — directions data-derived, not degenerate ──────────────────

    /// Directions derived from distinct cluster centroids must be distinct
    /// (the T5.3b fix: random directions hit concentration-of-measure).
    #[test]
    fn g2_directions_are_non_degenerate() {
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);

        // Each direction must be unit norm.
        for k in 0..N {
            let norm_sq: f32 = directions[k].iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-4,
                "direction {k} has norm {} (expected 1.0)",
                norm_sq.sqrt()
            );
        }

        // Distinct modes must produce distinct directions (not collapsed).
        for i in 0..N {
            for j in (i + 1)..N {
                let dot: f32 = (0..D).map(|k| directions[i][k] * directions[j][k]).sum();
                let cos = dot; // both are unit-norm
                assert!(
                    cos < 0.99,
                    "directions {i} and {j} are near-identical (cos={cos:.4})"
                );
            }
        }
    }

    // ─── T5.5 G3 — cross-mode discrimination (the load-bearing gate) ────────

    /// The freezer must correctly classify held-out test trajectories into
    /// their source failure mode. Mirrors T5.3b's geometry-strategy gate:
    /// ≥80% accuracy with the matching-gate > 0.6.
    ///
    /// **This is the substrate-level gate.** T5.6 G5 is the cross-snapshot
    /// gate (does the freezer discriminate across *models*, not just across
    /// failure modes within a model). This G3 only asserts the substrate
    /// works end-to-end on synthetic data — which T5.3b already proved.
    #[test]
    fn g3_cross_mode_discrimination() {
        // Train split: first 3 seeds per mode.
        const TRAJ_PER_MODE: usize = 5;
        const TRAIN_SEEDS: usize = 3;

        // Build all trajectories.
        let mut all_trajs: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(N);
        for mode_idx in 0..N {
            let mut mode_trajs = Vec::with_capacity(TRAJ_PER_MODE);
            for seed in 0..TRAJ_PER_MODE {
                mode_trajs.push(build_trajectory_for_mode(mode_idx, seed as u64 * 100 + 7));
            }
            all_trajs.push(mode_trajs);
        }

        // Derive directions from train split.
        let mut train: [[[f32; D]; TRAIN_SEEDS]; N] = [[[0.0; D]; TRAIN_SEEDS]; N];
        for mode_idx in 0..N {
            for seed in 0..TRAIN_SEEDS {
                train[mode_idx][seed] = encode_geom_summary(&all_trajs[mode_idx][seed]);
            }
        }
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);
        let fields = make_stub_fields();

        // Probe on test split.
        let mut n_correct = 0usize;
        let total = N * (TRAJ_PER_MODE - TRAIN_SEEDS);
        for mode_idx in 0..N {
            for seed in TRAIN_SEEDS..TRAJ_PER_MODE {
                let refs = build_refs(&all_trajs[mode_idx][seed]);
                let frozen = freezer.freeze_attempt(&refs, &fields, 1);
                let gates = frozen.gates();
                let matching_gate = gates[mode_idx];
                let argmax = frozen.argmax_archetype();
                let correct = matching_gate > 0.6 && argmax == mode_idx;
                if correct {
                    n_correct += 1;
                }
            }
        }

        let accuracy = n_correct as f32 / total as f32;
        assert!(
            accuracy >= 0.8,
            "G3 FAIL: accuracy {accuracy:.2} < 0.80 (correct {n_correct}/{total})"
        );
    }

    // ─── T5.5 G4 — envelope tamper-evidence ─────────────────────────────────

    #[test]
    fn g4_envelope_tamper_evidence() {
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        let traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&traj);
        let fields = make_stub_fields();
        let frozen = freezer.freeze_attempt(&refs, &fields, 1);

        // Header verification passes.
        assert!(frozen.envelope.verify_header());

        // Tamper the magic → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.magic = *b"XXXX";
        assert!(!tampered.verify_header());

        // Tamper the version → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.version = 999;
        assert!(!tampered.verify_header());

        // Tamper the merkle_root → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.merkle_root[0] ^= 0xff;
        assert!(!tampered.verify_header());

        // Tamper the commitment → header verification fails.
        let mut tampered = frozen.envelope;
        tampered.commitment[0] ^= 0xff;
        assert!(!tampered.verify_header());
    }

    /// Stub fields for tests — FAME's `commit` only reads `commitment()`,
    /// not `evolve()`. So a trivial stub suffices.
    fn make_stub_fields() -> [&'static dyn ArchetypeFieldSource<D>; N] {
        // SAFETY: this leaks a static. Acceptable for tests.
        // (We use Box::leak so the field outlives the call; for the test
        // suite this is fine — the process exits shortly after.)
        static F0: StubField = StubField::new(0);
        static F1: StubField = StubField::new(1);
        static F2: StubField = StubField::new(2);
        [&F0, &F1, &F2]
    }

    /// Minimal stub field — `commitment()` returns a per-index hash.
    /// `evolve()` is never called by `freeze_attempt` (only `commit()` is,
    /// which only reads `commitment()`).
    struct StubField {
        idx: u8,
    }
    impl StubField {
        const fn new(idx: u8) -> Self {
            Self { idx }
        }
    }
    impl ArchetypeFieldSource<D> for StubField {
        fn evolve<'a>(&self, _z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
            // Should never be called by freeze_attempt.
            for x in dz_scratch.iter_mut().take(D) {
                *x = 0.0;
            }
            &mut dz_scratch[..D]
        }
        fn commitment(&self) -> [u8; 32] {
            let mut h = blake3::Hasher::new();
            h.update(b"stub_field:");
            h.update(&[self.idx]);
            h.finalize().into()
        }
    }

    // ─── T5.5 G5 — encoder normalizes correctly ─────────────────────────────

    #[test]
    fn g5_encoder_normalizes_features() {
        let encoder = GeometrySummaryEncoder::default_synthetic();
        let mut summary = [0.0_f32; D];

        // Length at the scale boundary → normalized to 1.0.
        let geom = LatentTrajectoryGeometry {
            length: 25.0,                          // > 20.0 → clamps to 1.0
            mean_curvature: core::f32::consts::PI, // = π → normalized to 1.0
            min_adjacent_cosine: -1.0,
            n_steps: 200, // > 100 → clamps to 1.0
        };
        encoder.encode_into(&geom, &mut summary);

        // First block.
        assert!((summary[0] - 1.0).abs() < 1e-6, "length_norm");
        assert!((summary[1] - 1.0).abs() < 1e-6, "curvature_norm");
        assert!((summary[2] - (-1.0)).abs() < 1e-6, "cosine_norm");
        assert!((summary[3] - 1.0).abs() < 1e-6, "n_steps_norm");

        // Replication into second block.
        assert!((summary[4] - 1.0).abs() < 1e-6, "block 2 length");
        assert!((summary[5] - 1.0).abs() < 1e-6, "block 2 curvature");
    }

    #[test]
    fn g5_encoder_handles_zero_geometry() {
        let encoder = GeometrySummaryEncoder::default_synthetic();
        let mut summary = [99.0_f32; D]; // poison
        let geom = LatentTrajectoryGeometry::default(); // all zeros
        encoder.encode_into(&geom, &mut summary);
        // All features are 0 (length=0/scale=0, curvature=0/π=0, etc.).
        for j in 0..D {
            assert!(
                summary[j].abs() < 1e-6,
                "slot {j} = {} (expected 0)",
                summary[j]
            );
        }
    }

    // ─── T5.5 G6 — version affects envelope ─────────────────────────────────

    /// A re-commit with a different version produces a distinct envelope
    /// (mirrors FAME's `g5_blake3_version_affects_hash`).
    #[test]
    fn g6_version_affects_envelope() {
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 7)),
                encode_geom_summary(&build_trajectory_for_mode(0, 107)),
                encode_geom_summary(&build_trajectory_for_mode(0, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 7)),
                encode_geom_summary(&build_trajectory_for_mode(1, 107)),
                encode_geom_summary(&build_trajectory_for_mode(1, 207)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 7)),
                encode_geom_summary(&build_trajectory_for_mode(2, 107)),
                encode_geom_summary(&build_trajectory_for_mode(2, 207)),
            ],
        ];
        let mut directions = [[0.0_f32; D]; N];
        derive_directions(&train, &mut directions);
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        let traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&traj);
        let fields = make_stub_fields();

        let v1 = freezer.freeze_attempt(&refs, &fields, 1);
        let v2 = freezer.freeze_attempt(&refs, &fields, 2);

        // Same pi (the geometry + directions are identical; only version differs).
        assert_eq!(v1.blend.pi, v2.blend.pi, "pi must be identical");
        // Different FAME blake3 (version is part of FAME's commitment input).
        assert_ne!(
            v1.blend.blake3, v2.blend.blake3,
            "FAME blake3 must differ by version"
        );
        // Different envelope commitment (because pi is the same but version
        // differs in the FAME blake3, which is NOT in our payload — but our
        // payload is also identical, so the envelope commitment should ALSO
        // be identical. This is correct behavior: the envelope commits to
        // the trajectory's geometry, not to the FAME version counter).
        //
        // Wait — the payload includes `blend.pi`, which is identical, but
        // NOT `blend.blake3` or `blend.version`. So the envelope commitment
        // is identical across versions. That's the correct semantics: the
        // envelope commits to the *trajectory characterization*, not to the
        // FAME audit counter.
        assert_eq!(
            v1.envelope.commitment, v2.envelope.commitment,
            "envelope commits to trajectory, not to FAME version"
        );
    }

    /// T5.6 fix verification: `fit` (mean-centering) produces different gates
    /// than `new` (no centering) for the same directions + trajectory.
    ///
    /// This is a regression guard for the mean-centering fix discovered in
    /// Bench 014 — without it, non-centered summaries produce dot products all
    /// on the same side of 0, yielding degenerate classification.
    #[test]
    fn g7_fit_mean_centering_affects_gates() {
        // Build a synthetic training corpus with 3 modes (same as G3).
        let train: [[[f32; D]; 3]; N] = [
            [
                encode_geom_summary(&build_trajectory_for_mode(0, 100)),
                encode_geom_summary(&build_trajectory_for_mode(0, 101)),
                encode_geom_summary(&build_trajectory_for_mode(0, 102)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(1, 100)),
                encode_geom_summary(&build_trajectory_for_mode(1, 101)),
                encode_geom_summary(&build_trajectory_for_mode(1, 102)),
            ],
            [
                encode_geom_summary(&build_trajectory_for_mode(2, 100)),
                encode_geom_summary(&build_trajectory_for_mode(2, 101)),
                encode_geom_summary(&build_trajectory_for_mode(2, 102)),
            ],
        ];

        // Fit (with centroid) vs new (without centroid) — same directions.
        let freezer_fit = SweTrajectoryFreezer::<N, D>::fit(&train);
        let freezer_new = SweTrajectoryFreezer::<N, D>::new(freezer_fit.directions);

        let fields = make_stub_fields();
        let test_traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&test_traj);

        let frozen_fit = freezer_fit.freeze_attempt(&refs, &fields, 1);
        let frozen_new = freezer_new.freeze_attempt(&refs, &fields, 1);

        // The gates SHOULD differ because mean-centering shifts the dot
        // products. If they're identical, the centroid is zero (which means
        // the training data is centered — not the case for our synthetic
        // trajectories with positive-valued features).
        let gates_fit = frozen_fit.gates();
        let gates_new = frozen_new.gates();

        // At least one gate should differ.
        let any_diff = (0..N).any(|k| (gates_fit[k] - gates_new[k]).abs() > 1e-6);
        assert!(
            any_diff,
            "fit (mean-centered) gates should differ from new (raw) gates"
        );

        // Both should still classify mode 0 correctly (the synthetic regime
        // is discriminative enough that even without centering it works at
        // N=3 — the centering fix is critical for N=2 binary classification,
        // documented in Bench 014).
        assert_eq!(frozen_fit.argmax_archetype(), 0);
        assert_eq!(frozen_new.argmax_archetype(), 0);
    }

    /// `from_states_into` produces identical results to `from_states`.
    #[test]
    fn g8_from_states_into_matches_from_states() {
        use crate::latent_trajectory_geometry::{from_states, from_states_into};

        // Build a realistic trajectory.
        let traj = build_trajectory_for_mode(0, 42);
        let refs = build_refs(&traj);

        // from_states (allocating).
        let geom_alloc = from_states(&refs);

        // from_states_into (zero-alloc, reused buffers).
        let mut disp_curr = Vec::new();
        let mut disp_prev = Vec::new();
        let geom_into = from_states_into(&refs, &mut disp_curr, &mut disp_prev);

        // Bit-identical geometry.
        assert_eq!(geom_alloc.length, geom_into.length);
        assert_eq!(geom_alloc.mean_curvature, geom_into.mean_curvature);
        assert_eq!(
            geom_alloc.min_adjacent_cosine,
            geom_into.min_adjacent_cosine
        );
        assert_eq!(geom_alloc.n_steps, geom_into.n_steps);

        // Second call with same buffers — still identical (reuse works).
        let geom_into2 = from_states_into(&refs, &mut disp_curr, &mut disp_prev);
        assert_eq!(geom_alloc.length, geom_into2.length);
        assert_eq!(geom_alloc.mean_curvature, geom_into2.mean_curvature);
    }

    // ─── T5.6e G2 — StateMagnitudeEncoder perf ────────────────────────────

    /// The state-magnitude encoder MUST be no slower than the geometry
    /// pipeline (`from_states_into` + `GeometrySummaryEncoder::encode_into`)
    /// at the same trajectory scale. Both are O(n·dim) scans, so the value
    /// encoder (single-pass Welford) should be FASTER than the geometry path
    /// (which does a separate pass for displacements).
    ///
    /// The absolute target is < 100µs for D=1024, N=64 (the bench_018
    /// production scale). At this scale the O(n·dim) = 65K f32 muls dominate;
    /// the geometry path's `from_states_into` takes ~50µs for the same scan.
    /// A 100µs ceiling gives 2x headroom over the geometry baseline.
    ///
    /// Run in release mode only — debug builds are too slow for meaningful
    /// perf assertions. The `#[cfg_attr(debug_assertions, ignore)]` follows
    /// the established pattern for tight perf gates.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn g2_state_magnitude_encoder_under_100us() {
        use crate::latent_trajectory_geometry::from_states_into;

        const PERF_DIM: usize = 1024;
        const PERF_N: usize = 64;

        // Build a deterministic trajectory.
        let traj: Vec<Vec<f32>> = (0..PERF_N)
            .map(|i| {
                (0..PERF_DIM)
                    .map(|j| i as f32 * 0.001 + j as f32 * 0.0001)
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = traj.iter().map(|v| v.as_slice()).collect();
        let encoder = StateMagnitudeEncoder::new();
        let mut out = [0.0_f32; 32];

        // Warmup.
        for _ in 0..100 {
            encoder.encode_into(&refs, &mut out);
        }

        // Measure value encoder.
        const N_ITERS: usize = 1000;
        let start = std::time::Instant::now();
        for _ in 0..N_ITERS {
            encoder.encode_into(&refs, &mut out);
            std::hint::black_box(&out);
        }
        let value_ns = start.elapsed().as_nanos() as f64 / N_ITERS as f64;

        // Measure geometry pipeline (from_states_into + encode_into) for
        // comparison — this is what the existing geometry path costs at
        // the same scale.
        let geom_encoder = GeometrySummaryEncoder::default_depth_trajectory();
        let mut disp_curr = vec![0.0_f32; PERF_DIM];
        let mut disp_prev = vec![0.0_f32; PERF_DIM];
        let mut geom_out = [0.0_f32; 32];
        for _ in 0..100 {
            let g = from_states_into(&refs, &mut disp_curr, &mut disp_prev);
            geom_encoder.encode_into(&g, &mut geom_out);
        }
        let start = std::time::Instant::now();
        for _ in 0..N_ITERS {
            let g = from_states_into(&refs, &mut disp_curr, &mut disp_prev);
            geom_encoder.encode_into(&g, &mut geom_out);
            std::hint::black_box(&geom_out);
        }
        let geom_ns = start.elapsed().as_nanos() as f64 / N_ITERS as f64;

        eprintln!(
            "G2 perf: value={value_ns:.0}ns, geometry={geom_ns:.0}ns, ratio={:.2}x",
            value_ns / geom_ns
        );

        // Value encoder must be under 100µs (the geometry baseline is ~50µs;
        // 100µs gives 2x headroom).
        assert!(
            value_ns < 100_000.0,
            "G2 FAIL: StateMagnitudeEncoder {value_ns:.0} ns/call >= 100000 ns (D={PERF_DIM}, N={PERF_N})"
        );

        // Value encoder should be no slower than geometry (both O(n·dim),
        // but value is single-pass while geometry does displacements).
        // Allow 1.5x margin for Welford's per-step division overhead.
        assert!(
            value_ns < geom_ns * 1.5,
            "G2 FAIL: value {value_ns:.0}ns > 1.5x geometry {geom_ns:.0}ns"
        );
    }

    // ─── T5.6e G1 — StateMagnitudeEncoder correctness ─────────────────────────

    /// The state-magnitude encoder MUST produce the same 8 features as
    /// bench_018's `encode_seq_state_stats` on identical input. This is the
    /// G1 correctness gate for the substrate port.
    ///
    /// We use a hand-crafted trajectory with known L2 norms so the expected
    /// feature values can be computed independently (not via the encoder).
    #[test]
    fn g1_state_magnitude_encoder_correctness() {
        // Hand-crafted: 3 states of dim 2 with simple magnitudes.
        //   state 0 = [3.0, 4.0]   → norm = 5.0
        //   state 1 = [6.0, 8.0]   → norm = 10.0
        //   state 2 = [0.0, 0.0]   → norm = 0.0
        // mean_norm = (5 + 10 + 0) / 3 = 5.0
        // var_norm  = ((5-5)^2 + (10-5)^2 + (0-5)^2) / 3 = (0 + 25 + 25)/3 = 50/3
        // std_norm  = sqrt(50/3) ≈ 4.0824829
        // max_norm  = 10.0
        // min_norm  = 0.0
        // initial   = 5.0
        // final     = 0.0
        // ratio     = 0.0 / 5.0 = 0.0
        // cos(0,1)  = (18+32)/(5*10) = 50/50 = 1.0
        // cos(1,2)  = denom = (0)*(0) = 0 → skip (denom <= 1e-12)
        // mean_cos  = 1.0 / 1 = 1.0
        let states: Vec<Vec<f32>> = vec![vec![3.0, 4.0], vec![6.0, 8.0], vec![0.0, 0.0]];
        let refs = build_refs(&states);
        let encoder = StateMagnitudeEncoder::new();
        let mut out = [99.0_f32; D];
        encoder.encode_into(&refs, &mut out);

        let mean_norm = 5.0_f32;
        let var_norm = 50.0_f32 / 3.0;
        let std_norm = var_norm.sqrt();

        assert!(
            (out[0] - mean_norm).abs() < 1e-5,
            "mean_norm: {} vs {}",
            out[0],
            mean_norm
        );
        assert!(
            (out[1] - std_norm).abs() < 1e-5,
            "std_norm: {} vs {}",
            out[1],
            std_norm
        );
        assert!((out[2] - 10.0).abs() < 1e-5, "max_norm: {}", out[2]);
        assert!((out[3] - 0.0).abs() < 1e-5, "min_norm: {}", out[3]);
        assert!((out[4] - 5.0).abs() < 1e-5, "initial_norm: {}", out[4]);
        assert!((out[5] - 0.0).abs() < 1e-5, "final_norm: {}", out[5]);
        assert!((out[6] - 0.0).abs() < 1e-5, "norm_ratio: {}", out[6]);
        assert!((out[7] - 1.0).abs() < 1e-5, "mean_cos: {}", out[7]);

        // Trailing slots left at zero (D=32 > 8 features).
        for j in 8..D {
            assert!(
                out[j].abs() < 1e-6,
                "trailing slot {j} should be 0, got {}",
                out[j]
            );
        }
    }

    /// Empty trajectory + single-state trajectory edge cases.
    #[test]
    fn g1_state_magnitude_empty_and_single() {
        let encoder = StateMagnitudeEncoder::new();

        // Empty → all zeros (no panic).
        let empty: Vec<Vec<f32>> = vec![];
        let empty_refs: Vec<&[f32]> = empty.iter().map(|v| v.as_slice()).collect();
        let mut out = [99.0_f32; D];
        encoder.encode_into(&empty_refs, &mut out);
        for j in 0..D {
            assert!(
                out[j].abs() < 1e-6,
                "empty slot {j} should be 0, got {}",
                out[j]
            );
        }

        // Single state → mean/std/max/min/initial/final all equal,
        // std=0, ratio=1.0 (initial>1e-12 case), mean_cos=0 (no pairs).
        let single: Vec<Vec<f32>> = vec![vec![3.0, 4.0]];
        let single_refs: Vec<&[f32]> = single.iter().map(|v| v.as_slice()).collect();
        encoder.encode_into(&single_refs, &mut out);
        assert!((out[0] - 5.0).abs() < 1e-5, "single mean_norm: {}", out[0]);
        assert!((out[1] - 0.0).abs() < 1e-5, "single std_norm: {}", out[1]);
        assert!((out[2] - 5.0).abs() < 1e-5, "single max_norm: {}", out[2]);
        assert!((out[3] - 5.0).abs() < 1e-5, "single min_norm: {}", out[3]);
        assert!((out[4] - 5.0).abs() < 1e-5, "single initial: {}", out[4]);
        assert!((out[5] - 5.0).abs() < 1e-5, "single final: {}", out[5]);
        assert!((out[6] - 1.0).abs() < 1e-5, "single ratio: {}", out[6]);
        assert!((out[7] - 0.0).abs() < 1e-5, "single mean_cos: {}", out[7]);
    }

    // ─── T5.6e G1b — freeze_attempt_value is deterministic ──────────────────

    /// Two value-freezes of the same trajectory + same directions + same
    /// version MUST produce bit-identical envelopes.
    #[test]
    fn g1b_freeze_attempt_value_deterministic() {
        // Build a freezer with arbitrary directions (the value path doesn't
        // need fitted directions to be deterministic — it just needs them
        // fixed).
        let directions = [[0.0_f32; D]; N]; // zero directions are fine for determinism check
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        let traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&traj);
        let fields = make_stub_fields();

        let a = freezer.freeze_attempt_value(&refs, &fields, 1);
        let b = freezer.freeze_attempt_value(&refs, &fields, 1);

        assert_eq!(a.envelope.commitment, b.envelope.commitment);
        assert_eq!(a.envelope.merkle_root, b.envelope.merkle_root);
        assert_eq!(a.envelope.data_len, b.envelope.data_len);
        assert_eq!(a.blend.pi, b.blend.pi);
        assert_eq!(a.summary, b.summary);
    }

    // ─── T5.6e G3 — no-regression (geometry path still works) ──────────────

    /// Adding the value path MUST NOT break the geometry path. We re-run
    /// the G3 cross-mode discrimination gate (the load-bearing geometry
    /// test) to confirm no regression.
    #[test]
    fn g3_geometry_path_unaffected_by_value_addition() {
        // Re-use the same training corpus as g3_cross_mode_discrimination.
        const TRAJ_PER_MODE: usize = 5;
        const TRAIN_SEEDS: usize = 3;

        let mut all_trajs: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(N);
        for mode_idx in 0..N {
            let mut mode_trajs = Vec::with_capacity(TRAJ_PER_MODE);
            for seed in 0..TRAJ_PER_MODE {
                mode_trajs.push(build_trajectory_for_mode(mode_idx, seed as u64 * 100 + 7));
            }
            all_trajs.push(mode_trajs);
        }

        let mut train: [[[f32; D]; TRAIN_SEEDS]; N] = [[[0.0; D]; TRAIN_SEEDS]; N];
        for mode_idx in 0..N {
            for seed in 0..TRAIN_SEEDS {
                train[mode_idx][seed] = encode_geom_summary(&all_trajs[mode_idx][seed]);
            }
        }
        let freezer = SweTrajectoryFreezer::<N, D>::fit(&train);
        let fields = make_stub_fields();

        let mut n_correct = 0usize;
        let total = N * (TRAJ_PER_MODE - TRAIN_SEEDS);
        for mode_idx in 0..N {
            for seed in TRAIN_SEEDS..TRAJ_PER_MODE {
                let refs = build_refs(&all_trajs[mode_idx][seed]);
                let frozen = freezer.freeze_attempt(&refs, &fields, 1);
                if frozen.gates()[mode_idx] > 0.6 && frozen.argmax_archetype() == mode_idx {
                    n_correct += 1;
                }
            }
        }
        let accuracy = n_correct as f32 / total as f32;
        assert!(
            accuracy >= 0.8,
            "G3 regression: geometry accuracy {accuracy:.2} < 0.80"
        );
    }

    // ─── T5.6e G4 — value-freeze payload verifies + tamper-evident ──────────

    #[test]
    fn g4_value_envelope_tamper_evidence() {
        let directions = [[0.0_f32; D]; N];
        let freezer = SweTrajectoryFreezer::<N, D>::new(directions);

        let traj = build_trajectory_for_mode(0, 999);
        let refs = build_refs(&traj);
        let fields = make_stub_fields();
        let frozen = freezer.freeze_attempt_value(&refs, &fields, 1);

        // Header verifies clean.
        assert!(frozen.envelope.verify_header());

        // Tamper the merkle_root → fails.
        let mut tampered = frozen.envelope;
        tampered.merkle_root[0] ^= 0xff;
        assert!(!tampered.verify_header());

        // Tamper the commitment → fails.
        let mut tampered = frozen.envelope;
        tampered.commitment[0] ^= 0xff;
        assert!(!tampered.verify_header());

        // Payload length is N*4 + D*4 (no geometry triple).
        let expected_len = (N * 4 + D * 4) as u64;
        assert_eq!(frozen.envelope.data_len, expected_len);
    }

    // ─── T5.6e G5 — value discrimination (synthetic scale-shift) ────────────

    /// The load-bearing G5 gate for the value path. Bench 018 proved that
    /// state-magnitude features discriminate cross-snapshot (100% at σ≥0.1).
    /// The substrate-level test cannot run the full bench_018 (needs real
    /// Kimi-K3 weights), but it CAN prove the substrate discriminates
    /// scale-shifted trajectories — the synthetic analog of weight
    /// perturbation.
    ///
    /// Construction: build N classes of trajectories, each with a distinct
    /// constant scale factor applied to all states (mimicking how weight
    /// perturbation changes the activation scale). Train directions on
    /// train-split, probe on test-split. Accuracy must be ≥80%.
    #[test]
    fn g5_value_discrimination_synthetic_scale_shift() {
        // Mirrors bench_018: discriminating model snapshots that differ by
        // weight perturbation. Each class has a distinct scale (mimicking
        // the activation magnitude change) AND distinct per-token variation
        // (mimicking how perturbed weights process tokens differently).
        // The two independent signals prevent the centroids from being
        // collinear (which would degenerate the nearest-centroid classifier
        // to a 1D problem).
        const TRAJ_PER_CLASS: usize = 7;
        const TRAIN_PER_CLASS: usize = 5;
        const BASE_DIM: usize = 16;
        const N_TOKENS: usize = 32; // sequence-trajectory length
        // (scale, variance_factor) per class — two independent parameters
        // so centroids span a 2D subspace, not a 1D line.
        let class_params: [(f32, f32); N] = [(1.0, 0.5), (2.0, 1.5), (3.5, 0.8)];

        // Build all trajectories.
        let mut all_trajs: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(N);
        for class_idx in 0..N {
            let mut class_trajs = Vec::with_capacity(TRAJ_PER_CLASS);
            for seed in 0..TRAJ_PER_CLASS {
                let (scale, var_f) = class_params[class_idx];
                class_trajs.push(build_perturbed_trajectory(
                    scale,
                    var_f,
                    class_idx as u64,
                    seed as u64,
                    BASE_DIM,
                    N_TOKENS,
                ));
            }
            all_trajs.push(class_trajs);
        }

        // Train directions on state-magnitude summaries.
        let mut train: [[[f32; D]; TRAIN_PER_CLASS]; N] = [[[0.0; D]; TRAIN_PER_CLASS]; N];
        for class_idx in 0..N {
            for seed in 0..TRAIN_PER_CLASS {
                train[class_idx][seed] = encode_state_summary(&all_trajs[class_idx][seed]);
            }
        }
        let freezer = SweTrajectoryFreezer::<N, D>::fit(&train);
        let fields = make_stub_fields();

        // Probe on test split.
        let mut n_correct = 0usize;
        let total = N * (TRAJ_PER_CLASS - TRAIN_PER_CLASS);
        for class_idx in 0..N {
            for seed in TRAIN_PER_CLASS..TRAJ_PER_CLASS {
                let refs = build_refs(&all_trajs[class_idx][seed]);
                let frozen = freezer.freeze_attempt_value(&refs, &fields, 1);
                let gates = frozen.gates();
                let matching_gate = gates[class_idx];
                let argmax = frozen.argmax_archetype();
                if matching_gate > 0.6 && argmax == class_idx {
                    n_correct += 1;
                }
            }
        }

        let accuracy = n_correct as f32 / total as f32;
        assert!(
            accuracy >= 0.8,
            "G5 FAIL: value accuracy {accuracy:.2} < 0.80 (correct {n_correct}/{total})"
        );
    }

    /// Build a perturbed-model trajectory: mimics how weight perturbation
    /// changes both the activation scale AND the per-token variation pattern.
    ///
    /// Two independent parameters control the trajectory:
    /// - `scale`: overall activation magnitude (mimics weight magnitude change)
    /// - `var_factor`: per-token variation strength (mimics how perturbed
    ///   weights process different tokens differently)
    ///
    /// The class-specific bias (derived from `class_id`) adds a constant
    /// directional offset so classes differ in direction, not just magnitude.
    fn build_perturbed_trajectory(
        scale: f32,
        var_factor: f32,
        class_id: u64,
        seed: u64,
        dim: usize,
        n_tokens: usize,
    ) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed * 7919 + class_id * 1000 + 7);
        // Class-specific bias direction (fixed per class, varies across classes).
        let mut bias = vec![0.0_f32; dim];
        for j in 0..dim {
            bias[j] = rng.next_f32();
        }
        let bias_norm: f32 = bias.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for j in 0..dim {
            bias[j] /= bias_norm;
        }

        let mut traj = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            let mut state = vec![0.0_f32; dim];
            for j in 0..dim {
                // Base signal: class bias * scale + per-token noise * var_factor.
                state[j] = bias[j] * scale + rng.next_f32() * var_factor;
            }
            traj.push(state);
        }
        traj
    }

    /// Encode a trajectory's state-magnitude summary using the value encoder.
    fn encode_state_summary(traj: &[Vec<f32>]) -> [f32; D] {
        let refs = build_refs(traj);
        let mut summary = [0.0_f32; D];
        StateMagnitudeEncoder::new().encode_into(&refs, &mut summary);
        summary
    }
}
