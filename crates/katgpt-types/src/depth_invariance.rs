//! Depth-Invariance Diagnostic & Magnitude-Regularized Residual — root-cause
//! counterpart to four existing symptom-only detectors.
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/306_depth_invariance_diagnostic.md`
//! - **Research:** `katgpt-rs/.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md`
//! - **Source paper:** arXiv:2605.09992 — Eldenk et al., *Attention Drift: What
//!   Autoregressive Speculative Decoding Models Learn*
//! - **Companion private guide:** `riir-ai/.research/151_Recursive_Latent_State_Magnitude_Hygiene_Guide.md`
//!
//! # What this module is
//!
//! Root-cause counterpart to four existing symptom-only detectors
//! (BeliefRankPruner, GainCostLoopHalter, latent_functor/reestimation.rs,
//! micro_belief/coherence_bench.rs). Modelless math, no game semantics — the
//! diagnostic operates on raw `&[f32]` chains from any recursive latent-state
//! kernel.
//!
//! Given a chain `h_0, h_1, …, h_k ∈ ℝ^d`, three signals are extracted:
//! - **magnitude_slope** — least-squares slope of `‖h_t‖₂` vs `t`. Root-cause
//!   signal (the paper's primary insight: depth-specific refinement shows
//!   monotonically growing magnitude).
//! - **mean_cos_step** — mean `cos(h_t, h_{t-1})` for `t ∈ [1,k]`. Drift-lock
//!   signal (high cos + growing magnitude = "locked drift" sub-case).
//! - **effective_rank_slope** — least-squares slope of per-timestep flatness
//!   `(Σh²)² / (d·Σh⁴)` vs `t`. Collapse signal (negative slope = rank
//!   collapsing toward 1).
//!
//! The classifier then decides: [`DepthInvariant`], [`DepthSpecificRefinement`],
//! [`Collapsed`], or [`Insufficient`].
//!
//! # Magnitude regularization (the fix primitive)
//!
//! For kernels we own (HLA, functor, micro_belief, engram, Raven), the
//! [`apply_magnitude_regularization`] wrapper is the modelless upstream fix.
//!
//! > For frozen pretrained kernels (BeliefDrafter MLP), apply this only as a
//! > diagnostic — paper §4.4 Table 4 shows inference-time pin drops acceptance
//! > 56% on pre-norm models. The fix requires retraining. → riir-train. For
//! > kernels we own (HLA, functor, micro_belief, engram, Raven), this is the
//! > modelless upstream fix.

use crate::simd;

/// Epsilon for RMSNorm to avoid division by zero.
const RMS_EPS: f32 = 1e-8;

// ── Types ─────────────────────────────────────────────────────────────────

/// Classification of a recursive latent-state chain's depth behaviour.
///
/// Per Research 286 §2.1 / arXiv:2605.09992. `#[repr(u8)]` per AGENTS.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DepthInvarianceKind {
    /// `‖h_t‖` flat, cos step stable, effective rank flat.
    /// Kernel is a stable depth-invariant autoregressive predictor.
    DepthInvariant = 0,
    /// `‖h_t‖` monotonically growing. Kernel is doing depth-specific refinement
    /// (acting as N+1, N+2, … extra layers on whatever feeds it).
    DepthSpecificRefinement = 1,
    /// Effective rank collapsing toward 1. Kernel has collapsed onto a
    /// rank-1 attractor (mode collapse).
    Collapsed = 2,
    /// `k+1 < min_samples` — insufficient data to classify.
    Insufficient = 3,
}

/// Diagnostic output: the three raw signals plus the classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthInvarianceDiagnostic {
    /// Least-squares slope of `‖h_t‖₂` vs `t`. Root-cause signal.
    pub magnitude_slope: f32,
    /// Mean `cos(h_t, h_{t-1})` for `t ∈ [1,k]`. Drift direction signal.
    /// Callers can check `> cfg.cos_step_drift_lock` for the "locked drift"
    /// sub-case when `kind == DepthSpecificRefinement`.
    pub mean_cos_step: f32,
    /// Least-squares slope of effective rank vs `t`. Collapse signal.
    pub effective_rank_slope: f32,
    /// Classification.
    pub kind: DepthInvarianceKind,
}

/// Configuration thresholds for the depth-invariance classifier.
#[derive(Clone, Copy, Debug)]
pub struct DepthInvarianceConfig {
    /// Minimum number of samples `k+1` needed to fit a slope (default 4).
    pub min_samples: usize,
    /// `|magnitude_slope| > this` → `DepthSpecificRefinement` (default 0.05).
    pub magnitude_slope_drift: f32,
    /// **Reserved:** magnitude slope below this was originally part of the
    /// Collapsed AND-condition. Current decision rule uses
    /// [`Self::effective_rank_collapse`] as the primary collapse trigger
    /// (see module decision-rule note). Kept for API stability / future use.
    pub magnitude_slope_collapse: f32,
    /// `effective_rank_slope < this` → `Collapsed` (default -0.05).
    pub effective_rank_collapse: f32,
    /// Cos-step threshold above which a `DepthSpecificRefinement` chain is
    /// sub-classified as "locked drift" by the caller (default 0.95).
    pub cos_step_drift_lock: f32,
}

impl Default for DepthInvarianceConfig {
    fn default() -> Self {
        Self {
            min_samples: 4,
            magnitude_slope_drift: 0.05,
            magnitude_slope_collapse: -0.05,
            effective_rank_collapse: -0.05,
            cos_step_drift_lock: 0.95,
        }
    }
}

// ── Scratch (caller-owned, zero-alloc hot path) ───────────────────────────

/// Pre-allocated scratch buffers for [`classify_chain`] /
/// [`classify_chain_batched`]. Allocate once via [`Scratch::with_capacity`],
/// then `clear()` + reuse per call (per AGENTS.md hot-loop rules).
pub struct Scratch {
    /// `‖h_t‖₂` for each timestep. Length `k+1`.
    pub magnitude_series: Vec<f32>,
    /// Per-timestep flatness `(Σh²)²/(d·Σh⁴)`. Length `k+1`.
    pub rank_series: Vec<f32>,
    /// Length-`d` buffer reserved for in-place normalization / future SIMD
    /// cosine passes. Currently unused by Phase 1 (cosine computed directly
    /// from pre-computed norms + `simd_dot_f32`), but kept per Plan 306 T1.5
    /// for API compatibility with later SIMD-vectorized paths.
    pub h_tmp: Vec<f32>,
}

impl Scratch {
    /// Allocate scratch sized for chains up to `max_k_plus_1` timesteps at
    /// dimension `d`. All inner `Vec`s use `with_capacity` — no allocation
    /// happens on subsequent `clear()` + reuse cycles.
    pub fn with_capacity(max_k_plus_1: usize, d: usize) -> Self {
        Self {
            magnitude_series: Vec::with_capacity(max_k_plus_1),
            rank_series: Vec::with_capacity(max_k_plus_1),
            h_tmp: Vec::with_capacity(d),
        }
    }

    /// Reset all buffers to empty (length 0, capacity unchanged). Safe to
    /// call between chains in a batched sweep.
    pub fn clear(&mut self) {
        self.magnitude_series.clear();
        self.rank_series.clear();
        self.h_tmp.clear();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Least-squares slope of `ys` vs implicit `xs = [0, 1, …, n-1]` (timestep
/// indices). Zero-allocation — avoids materializing an `xs` slice.
/// Returns 0.0 if fewer than 2 points. Uses `mul_add` for accumulation
/// per AGENTS.md.
///
/// The general form `least_squares_slope(xs, ys)` is mathematically
/// equivalent but requires an `xs` allocation; since our only use case is
/// timestep-indexed xs, this specialized form is strictly better (Plan 306
/// T1.3 zero-allocation constraint).
#[inline]
fn least_squares_slope_vs_index(ys: &[f32]) -> f32 {
    let n = ys.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let x_mean = (nf - 1.0) * 0.5;
    let y_mean: f32 = ys.iter().copied().sum::<f32>() / nf;
    let mut num: f32 = 0.0;
    let mut den: f32 = 0.0;
    for (i, &y) in ys.iter().enumerate() {
        let dx = i as f32 - x_mean;
        num = dx.mul_add(y - y_mean, num);
        den = dx.mul_add(dx, den);
    }
    match den < 1e-30 {
        true => 0.0,
        false => num / den,
    }
}

// ── Public API: classify ──────────────────────────────────────────────────
//
// NOTE: the per-timestep participation-ratio flatness `(Σh²)² / (d · Σh⁴)`
// is computed inline in `classify_chain` (fused with the magnitude pass for
// a single sweep). The formula mirrors `BeliefRankPruner::flatness` (katgpt-rs
// root crate, separate feature); reimplemented inline because katgpt-core
// cannot depend on the root. Returns 0.0 for the zero vector (BeliefRankPruner
// convention). Range `[0, 1]`.

/// Classify a single recursive latent-state chain.
///
/// - `states`: flattened `[k+1][d]` row-major slice (`h_0, h_1, …, h_k`).
/// - `d`: per-timestep dimensionality.
/// - `cfg`: threshold config.
/// - `scratch`: caller-owned scratch (cleared + filled; not read).
///
/// **Zero allocation** in the hot path — all work uses `scratch` and stack
/// scalars. Complexity: O(k · d) — two passes (magnitude+flatness fused,
/// cosine via `simd_dot_f32`).
///
/// # Decision rule
///
/// Per Research 286 §2.1 and Plan 306 Phase 1 T1.3:
/// 1. `k+1 < cfg.min_samples` → [`Insufficient`].
/// 2. `effective_rank_slope < cfg.effective_rank_collapse` → [`Collapsed`]
///    (rank collapse is the defining feature; magnitude may be flat, growing,
///    or shrinking — see Research 286 enum doc "magnitude may be flat OR
///    growing").
/// 3. `|magnitude_slope| > cfg.magnitude_slope_drift` →
///    [`DepthSpecificRefinement`] (callers check `mean_cos_step >
///    cfg.cos_step_drift_lock` for the "locked drift" sub-case).
/// 4. Otherwise → [`DepthInvariant`].
///
/// **Deviation from Plan 306 T1.3 literal text:** the plan specified Collapsed
/// as `magnitude_slope < collapse AND effective_rank_slope < collapse`.
/// Plan 306 Phase 2 test T2.3 constructs a rank-collapse chain with *growing*
/// magnitude and expects [`Collapsed`] — the literal AND-rule would classify
/// it as [`DepthSpecificRefinement`]. Per the delegation instruction "fix the
/// rule, not the test", rank collapse is the sole Collapsed trigger.
/// `magnitude_slope_collapse` is retained in [`DepthInvarianceConfig`] for API
/// stability.
///
/// **SIMD inner loop (T7.4 revisit):** the per-timestep magnitude +
/// participation-ratio pass is now fused into a single
/// [`simd::simd_sum_sq_quartic`] sweep (NEON/AVX2/scalar dispatch). The cosine
/// step continues to use [`simd::simd_dot_f32`]. Zero allocation, same math,
/// same decision rule — pure perf change.
pub fn classify_chain(
    states: &[f32],
    d: usize,
    cfg: &DepthInvarianceConfig,
    scratch: &mut Scratch,
) -> DepthInvarianceDiagnostic {
    // ── Dimension validation ──
    if d == 0 || !states.len().is_multiple_of(d) {
        return DepthInvarianceDiagnostic {
            magnitude_slope: 0.0,
            mean_cos_step: 0.0,
            effective_rank_slope: 0.0,
            kind: DepthInvarianceKind::Insufficient,
        };
    }
    let k_plus_1 = states.len() / d;

    // ── Min-samples gate ──
    if k_plus_1 < cfg.min_samples {
        return DepthInvarianceDiagnostic {
            magnitude_slope: 0.0,
            mean_cos_step: 0.0,
            effective_rank_slope: 0.0,
            kind: DepthInvarianceKind::Insufficient,
        };
    }

    // ── Clear scratch for this chain ──
    scratch.clear();

    let d_f = d as f32;
    let mut cos_sum: f32 = 0.0;
    let mut cos_count: usize = 0;

    // ── Single sweep: magnitude + flatness + cosine step ──
    for t in 0..k_plus_1 {
        let h_t = &states[t * d..(t + 1) * d];

        // Fused Σx² + Σx⁴ in one SIMD sweep (Plan 306 T7.4). Σx² feeds the
        // magnitude; (Σx²)² / (d · Σx⁴) is the participation-ratio flatness.
        let (sum_sq, sum_quartic) = simd::simd_sum_sq_quartic(h_t);
        let magnitude = sum_sq.sqrt();
        scratch.magnitude_series.push(magnitude);

        let rank_t = match sum_quartic < 1e-12 {
            true => 0.0, // zero vector → peaked (BeliefRankPruner convention)
            false => {
                let pr = (sum_sq * sum_sq) / (d_f * sum_quartic);
                pr.clamp(0.0, 1.0)
            }
        };
        scratch.rank_series.push(rank_t);

        // Cosine step with previous timestep (simd_dot_f32 for the dot).
        if t > 0 {
            let h_prev = &states[(t - 1) * d..t * d];
            let mag_prev = scratch.magnitude_series[t - 1];
            if mag_prev > 0.0 && magnitude > 0.0 {
                let dot = simd::simd_dot_f32(h_t, h_prev, d);
                cos_sum += dot / (mag_prev * magnitude);
                cos_count += 1;
            } // else: degenerate pair — skip
        }
    }

    // ── Slopes ──
    let magnitude_slope = least_squares_slope_vs_index(&scratch.magnitude_series);
    let effective_rank_slope = least_squares_slope_vs_index(&scratch.rank_series);
    let mean_cos_step = match cos_count {
        0 => 0.0,
        c => cos_sum / c as f32,
    };

    // ── Decision rule (match on (collapsed, drifting) booleans per AGENTS.md) ──
    let collapsed = effective_rank_slope < cfg.effective_rank_collapse;
    let drifting = magnitude_slope.abs() > cfg.magnitude_slope_drift;
    let kind = match (collapsed, drifting) {
        (true, _) => DepthInvarianceKind::Collapsed,
        (false, true) => DepthInvarianceKind::DepthSpecificRefinement,
        (false, false) => DepthInvarianceKind::DepthInvariant,
    };

    DepthInvarianceDiagnostic {
        magnitude_slope,
        mean_cos_step,
        effective_rank_slope,
        kind,
    }
}

/// Batched classification — single sweep over multiple NPC chains.
///
/// - `states_per_kernel`: each entry is one kernel's flattened `[k+1][d]` chain.
/// - `scratch`: reused per chain (cleared before each).
/// - `out`: cleared, then one [`DepthInvarianceDiagnostic`] pushed per input
///   chain.
///
/// Complexity: O(N · k · d) total across `N` chains.
pub fn classify_chain_batched(
    states_per_kernel: &[&[f32]],
    d: usize,
    cfg: &DepthInvarianceConfig,
    scratch: &mut Scratch,
    out: &mut Vec<DepthInvarianceDiagnostic>,
) {
    out.clear();
    for &states in states_per_kernel {
        let diag = classify_chain(states, d, cfg, scratch);
        out.push(diag);
    }
}

// ── Magnitude Regularization (the fix primitive) ──────────────────────────

/// Post-residual magnitude regularization mode for recursive latent-state
/// kernels we own (HLA, functor, micro_belief, engram, Raven). For frozen
/// pretrained MLPs (BeliefDrafter), use diagnostic-only — see module doc.
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum MagnitudeRegularization {
    /// Identity: `h_{t+1} = h_t + Δ` (no regularization).
    None = 0,
    /// Pure RMSNorm (no learned γ/β): `h_{t+1} = (h_t + Δ) / rms(h_t + Δ)`.
    /// Paper's prescription; learned γ/β would require retraining → riir-train.
    RmsNorm = 1,
    /// Scalar pinch: if `‖h‖_rms > max_rms`, scale `h *= max_rms / ‖h‖_rms`.
    /// No-op if already below `max_rms`.
    ScalarPinch { max_rms: f32 } = 2,
}

/// Apply magnitude regularization in-place to a raw residual-updated hidden
/// state.
///
/// - `h_raw`: receives `h_t + Δ`, returns regularized `h_{t+1}` in-place.
/// - `mode`: regularization mode (see [`MagnitudeRegularization`]).
/// - `scratch`: length-`d` caller-owned slice, reserved for future learned
///   γ/β weight extensions (paper §4.4). Unused in Phase 1 pure RMSNorm —
///   the computation is scalar (running sum → rms → divide).
///
/// **Zero allocation** — all work is in-place on `h_raw`.
pub fn apply_magnitude_regularization(
    h_raw: &mut [f32],
    mode: MagnitudeRegularization,
    _scratch: &mut [f32],
) {
    match mode {
        MagnitudeRegularization::None => { /* identity — no-op */ }
        MagnitudeRegularization::RmsNorm => {
            let d_f = h_raw.len() as f32;
            let mut sum_sq: f32 = 0.0;
            for &x in h_raw.iter() {
                sum_sq += x * x;
            }
            // rms = sqrt(mean(h²) + eps); eps avoids div-by-zero on zero input.
            let rms = (sum_sq / d_f + RMS_EPS).sqrt();
            for x in h_raw.iter_mut() {
                *x /= rms;
            }
        }
        MagnitudeRegularization::ScalarPinch { max_rms } => {
            let d_f = h_raw.len() as f32;
            let mut sum_sq: f32 = 0.0;
            for &x in h_raw.iter() {
                sum_sq += x * x;
            }
            let current_rms = (sum_sq / d_f).sqrt();
            if current_rms > max_rms && current_rms > 0.0 {
                let scale = max_rms / current_rms;
                for x in h_raw.iter_mut() {
                    *x *= scale;
                }
            } // else: already bounded — no-op
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Root-mean-square of a slice (true RMS, no eps).
    fn rms(h: &[f32]) -> f32 {
        let n = h.len() as f32;
        let sum_sq: f32 = h.iter().map(|x| x * x).sum();
        (sum_sq / n).sqrt()
    }

    // ── T5.2: MagnitudeRegularization unit tests (4) ──

    #[test]
    fn none_is_identity() {
        let original = [0.1f32, -0.5, 0.8, 0.3, -0.2, 0.7];
        let mut h = original;
        let mut scratch = [0.0f32; 6];
        apply_magnitude_regularization(&mut h, MagnitudeRegularization::None, &mut scratch);
        assert_eq!(h, original, "None mode must be bit-identical identity");
    }

    #[test]
    fn rmsnorm_produces_unit_rms() {
        let mut h = [0.5f32, 1.2, -0.8, 0.3, 1.5, -0.4];
        let mut scratch = [0.0f32; 6];
        apply_magnitude_regularization(&mut h, MagnitudeRegularization::RmsNorm, &mut scratch);
        let post_rms = rms(&h);
        assert!(
            (post_rms - 1.0).abs() < 1e-5,
            "RmsNorm should produce unit RMS, got {post_rms}"
        );
    }

    #[test]
    fn scalar_pinch_caps_at_max_rms() {
        // Input rms = 2.0, max_rms = 1.0 → should scale to 1.0.
        let mut h = [2.0f32, 2.0, 2.0, 2.0]; // rms = 2.0
        let mut scratch = [0.0f32; 4];
        apply_magnitude_regularization(
            &mut h,
            MagnitudeRegularization::ScalarPinch { max_rms: 1.0 },
            &mut scratch,
        );
        let post_rms = rms(&h);
        assert!(
            post_rms <= 1.0 + 1e-5,
            "ScalarPinch should cap at max_rms=1.0, got {post_rms}"
        );
    }

    #[test]
    fn scalar_pinch_no_op_below_max_rms() {
        // Input rms = 0.5, max_rms = 1.0 → no-op (bit-identical).
        let original = [0.5f32, 0.5, 0.5, 0.5]; // rms = 0.5
        let mut h = original;
        let mut scratch = [0.0f32; 4];
        apply_magnitude_regularization(
            &mut h,
            MagnitudeRegularization::ScalarPinch { max_rms: 1.0 },
            &mut scratch,
        );
        assert_eq!(
            h, original,
            "ScalarPinch must be no-op (bit-identical) when rms < max_rms"
        );
    }

    // ── Phase 2 G1: classify_chain correctness gates (8) ──

    #[test]
    fn g1_flat_magnitude_is_depth_invariant() {
        // ‖h_t‖ = const (all identical, non-zero).
        let d = 4;
        let k_plus_1 = 6;
        let states: Vec<f32> = (0..k_plus_1)
            .flat_map(|_| [1.0f32, 1.0, 1.0, 1.0])
            .collect();
        let cfg = DepthInvarianceConfig::default();
        let mut scratch = Scratch::with_capacity(k_plus_1, d);
        let diag = classify_chain(&states, d, &cfg, &mut scratch);
        assert_eq!(
            diag.kind,
            DepthInvarianceKind::DepthInvariant,
            "flat magnitude → DepthInvariant, got {:?}",
            diag.kind
        );
        assert!(
            diag.magnitude_slope.abs() < 1e-5,
            "flat magnitude → slope ≈ 0, got {}",
            diag.magnitude_slope
        );
    }

    #[test]
    fn g1_linear_growth_is_depth_specific() {
        // h_t = t * v — linear growth. h_0 = 0 (zero vector, handled gracefully).
        let d = 4;
        let k_plus_1 = 6;
        let v = [1.0f32, 0.5, 0.3, 0.2];
        let states: Vec<f32> = (0..k_plus_1)
            .flat_map(|t| {
                let s = t as f32;
                [s * v[0], s * v[1], s * v[2], s * v[3]]
            })
            .collect();
        let cfg = DepthInvarianceConfig::default();
        let mut scratch = Scratch::with_capacity(k_plus_1, d);
        let diag = classify_chain(&states, d, &cfg, &mut scratch);
        assert_eq!(
            diag.kind,
            DepthInvarianceKind::DepthSpecificRefinement,
            "linear growth → DepthSpecificRefinement, got {:?}",
            diag.kind
        );
        assert!(
            diag.magnitude_slope > 0.0,
            "linear growth → positive slope, got {}",
            diag.magnitude_slope
        );
    }

    #[test]
    fn g1_rank_collapse_is_collapsed() {
        // h_t = [(t+1), 1, 1, 1] — growing magnitude in one component → rank
        // collapses (flatness trending down), magnitude trending up.
        let d = 4;
        let k_plus_1 = 6;
        let states: Vec<f32> = (0..k_plus_1)
            .flat_map(|t| {
                let s = (t + 1) as f32;
                [s, 1.0f32, 1.0, 1.0]
            })
            .collect();
        let cfg = DepthInvarianceConfig::default();
        let mut scratch = Scratch::with_capacity(k_plus_1, d);
        let diag = classify_chain(&states, d, &cfg, &mut scratch);
        assert_eq!(
            diag.kind,
            DepthInvarianceKind::Collapsed,
            "rank collapse → Collapsed, got {:?} (rank_slope={}, mag_slope={})",
            diag.kind,
            diag.effective_rank_slope,
            diag.magnitude_slope
        );
        assert!(
            diag.effective_rank_slope < cfg.effective_rank_collapse,
            "rank collapse → rank_slope < {}, got {}",
            cfg.effective_rank_collapse,
            diag.effective_rank_slope
        );
    }

    #[test]
    fn g1_insufficient_samples() {
        // k+1 = 3 < min_samples = 4 → Insufficient, slopes = 0.
        let d = 4;
        let k_plus_1 = 3;
        let states: Vec<f32> = (0..k_plus_1)
            .flat_map(|_| [1.0f32, 1.0, 1.0, 1.0])
            .collect();
        let cfg = DepthInvarianceConfig::default();
        let mut scratch = Scratch::with_capacity(8, d);
        let diag = classify_chain(&states, d, &cfg, &mut scratch);
        assert_eq!(diag.kind, DepthInvarianceKind::Insufficient);
        assert_eq!(diag.magnitude_slope, 0.0);
        assert_eq!(diag.effective_rank_slope, 0.0);
    }

    #[test]
    fn g1_oscillating_chain_is_depth_invariant() {
        // h_t = (-1)^t * [1, 1, 1, 1] — alternating sign, flat magnitude.
        let d = 4;
        let k_plus_1 = 6;
        let states: Vec<f32> = (0..k_plus_1)
            .flat_map(|t| {
                let s = if t % 2 == 0 { 1.0f32 } else { -1.0 };
                [s, s, s, s]
            })
            .collect();
        let cfg = DepthInvarianceConfig::default();
        let mut scratch = Scratch::with_capacity(k_plus_1, d);
        let diag = classify_chain(&states, d, &cfg, &mut scratch);
        assert_eq!(
            diag.kind,
            DepthInvarianceKind::DepthInvariant,
            "oscillating flat-magnitude → DepthInvariant, got {:?} (cos={})",
            diag.kind,
            diag.mean_cos_step
        );
        // Low cos (oscillating) but flat magnitude → still DepthInvariant.
        assert!(
            diag.mean_cos_step < 0.0,
            "oscillating → negative cos_step, got {}",
            diag.mean_cos_step
        );
    }

    #[test]
    fn g1_locked_drift_high_cos_growing_mag() {
        // h_t = (1 + 0.1*t) * v — collinear growth: high cos + growing mag.
        let d = 4;
        let k_plus_1 = 6;
        let v = [1.0f32, 0.5, 0.3, 0.2];
        let states: Vec<f32> = (0..k_plus_1)
            .flat_map(|t| {
                let s = 1.0 + 0.1 * t as f32;
                [s * v[0], s * v[1], s * v[2], s * v[3]]
            })
            .collect();
        let cfg = DepthInvarianceConfig::default();
        let mut scratch = Scratch::with_capacity(k_plus_1, d);
        let diag = classify_chain(&states, d, &cfg, &mut scratch);
        assert_eq!(
            diag.kind,
            DepthInvarianceKind::DepthSpecificRefinement,
            "locked drift → DepthSpecificRefinement, got {:?}",
            diag.kind
        );
        assert!(
            diag.mean_cos_step > cfg.cos_step_drift_lock,
            "locked drift → cos_step > {}, got {}",
            cfg.cos_step_drift_lock,
            diag.mean_cos_step
        );
    }

    #[test]
    fn g1_zero_chain_degenerate() {
        // All-zero h_t → flatness 0 (constant), magnitude 0 (constant) →
        // DepthInvariant (degenerate but stable).
        let d = 4;
        let k_plus_1 = 6;
        let states: Vec<f32> = (0..(k_plus_1 * d)).map(|_| 0.0f32).collect();
        let cfg = DepthInvarianceConfig::default();
        let mut scratch = Scratch::with_capacity(k_plus_1, d);
        let diag = classify_chain(&states, d, &cfg, &mut scratch);
        assert_eq!(
            diag.kind,
            DepthInvarianceKind::DepthInvariant,
            "zero chain → DepthInvariant (degenerate stable), got {:?}",
            diag.kind
        );
    }

    #[test]
    fn g1_batched_matches_single() {
        let d = 4;
        // Three distinct chains exercising different classifications.
        let chain_flat: Vec<f32> = (0..6).flat_map(|_| [1.0f32, 1.0, 1.0, 1.0]).collect();
        let chain_grow: Vec<f32> = (0..6)
            .flat_map(|t| {
                let s = 1.0 + 0.1 * t as f32;
                [s, s * 0.5, s * 0.3, s * 0.2]
            })
            .collect();
        let chain_collapse: Vec<f32> = (0..6)
            .flat_map(|t| {
                let s = (t + 1) as f32;
                [s, 1.0f32, 1.0, 1.0]
            })
            .collect();

        let chains: Vec<&[f32]> = vec![&chain_flat, &chain_grow, &chain_collapse];
        let cfg = DepthInvarianceConfig::default();

        // Per-chain results.
        let mut scratch_single = Scratch::with_capacity(8, d);
        let singles: Vec<_> = chains
            .iter()
            .map(|&s| classify_chain(s, d, &cfg, &mut scratch_single))
            .collect();

        // Batched results.
        let mut scratch_batch = Scratch::with_capacity(8, d);
        let mut batched = Vec::with_capacity(chains.len());
        classify_chain_batched(&chains, d, &cfg, &mut scratch_batch, &mut batched);

        assert_eq!(
            singles.len(),
            batched.len(),
            "batched should produce one diagnostic per chain"
        );
        for (i, (s, b)) in singles.iter().zip(batched.iter()).enumerate() {
            assert_eq!(
                s.kind, b.kind,
                "chain {i}: kind mismatch single={:?} batched={:?}",
                s.kind, b.kind
            );
            assert!(
                (s.magnitude_slope - b.magnitude_slope).abs() < 1e-6,
                "chain {i}: magnitude_slope mismatch single={} batched={}",
                s.magnitude_slope,
                b.magnitude_slope
            );
            assert!(
                (s.effective_rank_slope - b.effective_rank_slope).abs() < 1e-6,
                "chain {i}: effective_rank_slope mismatch single={} batched={}",
                s.effective_rank_slope,
                b.effective_rank_slope
            );
            assert!(
                (s.mean_cos_step - b.mean_cos_step).abs() < 1e-6,
                "chain {i}: mean_cos_step mismatch single={} batched={}",
                s.mean_cos_step,
                b.mean_cos_step
            );
        }
    }
}
