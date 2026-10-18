//! Position-Offset Reveal-Time Schedule for Set Diffusion (Research 376 Phase 1 T1.1).
//!
//! Implements Eq. 7 from Arriola & Kuleshov, "Set Diffusion: Interpolating
//! Token Orderings Between AR and Diffusion" (arXiv:2607.01775, ICML 2026).
//! Distilled in `riir-train/.research/376_Set_Diffusion_Flexible_Token_Sets.md`.
//!
//! # Core mechanism
//!
//! Each token position ℓ ∈ [0, L) gets a CDF for its "reveal time" R_ℓ ∈ [0, 1]:
//!
//! ```text
//! α^τ_ℓ = 0                        if τ ≤ a_ℓ            (not yet revealed)
//!       = ((τ - a_ℓ) / w)^k        if a_ℓ < τ < a_ℓ + w  (active interval)
//!       = 1                        if τ ≥ a_ℓ + w         (definitely revealed)
//! ```
//!
//! Where:
//! - **w** = active generation interval width ∈ (0, 1]. Controls left-to-right bias.
//! - **k** = shape parameter within the interval ∈ (0, ∞). k<1 front-loads reveal
//!   times toward interval start; k>1 back-loads toward interval end; k=1 is uniform.
//! - **a_ℓ = ℓ / (L-1) · (1-w)** = evenly spaced offset per position (0-indexed).
//!
//! # Interpolation property
//!
//! - **w → 0**: intervals shrink to points → strict left-to-right → **AR**.
//! - **w = 1**: all offsets collapse to 0, intervals fully overlap → **order-agnostic
//!   diffusion (MDLM)**.
//! - **Intermediate w**: tokens reveal in overlapping windows → **SW-SetDLM**
//!   (sliding-window set diffusion language model).
//!
//! # Sampling
//!
//! To sample a generation ordering σ (a permutation of [0, L)):
//! 1. For each position ℓ, sample R_ℓ via inverse-CDF: R_ℓ = a_ℓ + w · U^(1/k),
//!    where U ~ Uniform(0, 1).
//! 2. Sort positions by reveal time ascending; ties broken by position index
//!    (deterministic left-to-right preference on ties).
//!
//! # Expected inference budget (Eq. 52)
//!
//! C̄ = L · w · k / (k + 1) — the expected number of tokens eligible per
//! inference step. Matches: AR (w→0) → C̄→0 (singleton sets, many steps);
//! diffusion (w=1, k→∞) → C̄=L (all tokens, one step).
//!
//! # Consumers
//!
//! - **T1.2** (training loop): `sample_order` provides σ; `order_to_gen_steps`
//!   converts σ to the `position_order` buffer consumed by the set-causal
//!   attention kernel (`riir-gpu/src/kernels/attention_score_set_causal.wgsl`,
//!   shipped Plan 312 T2.2) and the CPU reference (`riir_engine::forward_set_causal`,
//!   shipped Research 376 T0.3).
//! - **T1.3** (A/B battle): `expected_budget` matches inference budget between
//!   block-causal and set-causal schedules.
//!
//! TL;DR: Eq. 7 CDF + inverse-CDF sampler for set-diffusion position-offset schedules.
//!
//! # Canonical source (DRY consolidation, 2026-07-04)
//!
//! This module is the **single source of truth** for `PositionOffsetSchedule`.
//! Consumers (`katgpt-rs`, `riir-train`) re-export from here rather than
//! maintaining their own copies. The RNG-agnostic core primitive is
//! [`PositionOffsetSchedule::sample_order_with`], which takes a closure
//! yielding `Uniform(0, 1)` floats — this lets each consumer use its preferred
//! RNG (`katgpt_types::Rng`, `fastrand::Rng`, etc.) without forcing a hard
//! dependency. A `fastrand::Rng` convenience wrapper is provided because
//! katgpt-core already depends on `fastrand`.
//!
//! Position-Offset Reveal-Time Schedule for Set Diffusion (Research 376).
//! Canonical source for `PositionOffsetSchedule` — pure math (CDF/inverse-CDF/
//! ordering), RNG-agnostic via closure-based sampling. No feature gate because
//! it's a zero-dep math substrate consumed by both katgpt-rs (runtime) and
//! riir-train (training). Eliminates the 3-way DRY violation that previously
//! had copies in katgpt-rs/src/dllm.rs, riir-train/.../set_diffusion_schedule.rs,
//! and riir-ai/crates/riir-poc/.
//!
//! Plan 602 T2.3: optional decode-order schedule variants (same feature;
//! defined in set_diffusion_schedule.rs per the plan target — that module
//! stays always-on, only these variants are gated; promote-only-on-G3,
//! demote silently if the Phase-3 G3 GOAT fails).

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════
// PositionOffsetSchedule
// ═══════════════════════════════════════════════════════════════════

/// Position-dependent reveal-time schedule (Eq. 7).
///
/// See module docs for the full math. The two knobs:
/// - `w` ∈ (0, 1]: interval width. w→0 is AR; w=1 is order-agnostic diffusion.
/// - `k` ∈ (0, ∞): intra-interval shape. k<1 front-loads; k=1 uniform; k>1 back-loads.
///
/// `Default` is the SW-SetDLM paper-aligned setting: w=0.5, k=1.0 (the config
/// that won the real-text battle in Plan 312 Phase 2 by 0.71 NLL over D2F).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PositionOffsetSchedule {
    /// Active generation interval width ∈ (0, 1].
    pub w: f32,
    /// Shape parameter within the interval ∈ (0, ∞).
    pub k: f32,
}

impl Default for PositionOffsetSchedule {
    fn default() -> Self {
        Self { w: 0.5, k: 1.0 }
    }
}

/// Minimum positive value for `w` and `k` to avoid degenerate math (div-by-zero,
/// INFINITY exponents). Anything below this is clamped on construction.
const MIN_POSITIVE: f32 = 1e-6;

impl PositionOffsetSchedule {
    /// Create a schedule with linear intra-interval shape (k=1).
    ///
    /// `w` is clamped to `[MIN_POSITIVE, 1.0]` to stay in the valid range.
    #[inline]
    pub fn new(w: f32) -> Self {
        Self {
            w: w.clamp(MIN_POSITIVE, 1.0),
            k: 1.0,
        }
    }

    /// Create a schedule with custom shape parameter.
    ///
    /// Both `w` and `k` are clamped to their valid ranges.
    #[inline]
    pub fn shaped(w: f32, k: f32) -> Self {
        Self {
            w: w.clamp(MIN_POSITIVE, 1.0),
            k: k.clamp(MIN_POSITIVE, f32::INFINITY),
        }
    }

    /// The AR endpoint schedule (w minimal → near-deterministic left-to-right).
    ///
    /// Note: this produces near-AR orderings, not exact AR. For exact AR
    /// (guaranteed [0, 1, ..., L-1]), use [`ar_order`]. The schedule is useful
    /// when you want the AR-like endpoint of the continuous w axis with a
    /// small stochastic perturbation.
    #[inline]
    pub fn ar() -> Self {
        Self {
            w: MIN_POSITIVE,
            k: 1.0,
        }
    }

    /// The order-agnostic diffusion endpoint (w=1, k=1) — uniform random orderings.
    #[inline]
    pub fn diffusion() -> Self {
        Self { w: 1.0, k: 1.0 }
    }

    // ── Core math (Eq. 7) ──────────────────────────────────────────

    /// Interval offset for position `ell` in a sequence of length `l`.
    ///
    /// `a_ℓ = ell / (L-1) · (1-w)` for `L > 1`; `0.0` for `L <= 1`.
    ///
    /// Returns the left endpoint of position ℓ's active reveal-time window.
    /// The window spans `[offset(ell), offset(ell) + w]`.
    #[inline]
    pub fn offset(&self, ell: usize, l: usize) -> f32 {
        if l <= 1 {
            0.0
        } else {
            (ell as f32 / (l - 1) as f32) * (1.0 - self.w)
        }
    }

    /// Check if position `ell` is eligible for unmasking at ordering time `tau`.
    ///
    /// Eligible if `tau` falls within the position's active generation interval
    /// [a_ℓ, a_ℓ + w] and the position has a non-zero reveal rate.
    ///
    /// At the boundary (tau >= a_ℓ + w), the position is always eligible
    /// (it must eventually be decoded).
    #[inline]
    pub fn is_eligible(&self, ell: usize, l: usize, tau: f32) -> bool {
        let a = self.offset(ell, l);
        // Position is eligible if we're past its interval start.
        // At tau >= a + w, it's past due — always eligible.
        tau >= a
    }

    /// Returns the set of positions eligible for unmasking at ordering time `tau`.
    ///
    /// A position is eligible if its active generation interval has started
    /// (tau >= a_ℓ). Positions whose intervals haven't started yet are
    /// blocked — they can't be committed even if confidence is high.
    pub fn eligible_positions(&self, l: usize, tau: f32) -> Vec<bool> {
        (0..l).map(|ell| self.is_eligible(ell, l, tau)).collect()
    }

    /// The CDF α^τ_ℓ = P(R_ℓ ≤ τ) at time `tau` for position `ell` (Eq. 7).
    ///
    /// Returns a value in `[0, 1]`. Monotonically non-decreasing in `tau`.
    #[inline]
    pub fn alpha(&self, tau: f32, ell: usize, l: usize) -> f32 {
        let a = self.offset(ell, l);
        if tau <= a {
            0.0
        } else if tau >= a + self.w {
            1.0
        } else {
            ((tau - a) / self.w).powf(self.k)
        }
    }

    /// Inverse-CDF: map a uniform `u ∈ [0, 1]` to a reveal time R_ℓ.
    ///
    /// `R_ℓ = a_ℓ + w · u^(1/k)`. This is the sampling primitive — call with
    /// `u ~ Uniform(0, 1)` to draw from the position-ℓ reveal-time distribution.
    #[inline]
    pub fn reveal_time_from_uniform(&self, u: f32, ell: usize, l: usize) -> f32 {
        let a = self.offset(ell, l);
        // u clamped to [0, 1] to handle fastrand::Rng::f32() which returns [0, 1).
        let u = u.clamp(0.0, 1.0);
        a + self.w * u.powf(1.0 / self.k)
    }

    /// Expected inference prediction budget C̄ (Eq. 52).
    ///
    /// `C̄ = L · w · k / (k + 1)` — the expected number of tokens eligible per
    /// inference step under this schedule.
    #[inline]
    pub fn expected_budget(&self, l: usize) -> f32 {
        l as f32 * self.w * self.k / (self.k + 1.0)
    }

    /// Find the width `w` that matches a target inference budget B for length L.
    ///
    /// Solves `L · w · k / (k+1) = B` for w (closed form):
    /// `w = B · (k+1) / (L · k)`, clamped to `[MIN_POSITIVE, 1.0]`.
    ///
    /// Used by T1.3 to match the inference budget between block-causal
    /// (budget = block_size) and set-causal schedules.
    ///
    /// Returns a new schedule with this instance's `k` and the solved `w`.
    #[inline]
    pub fn with_matched_budget(&self, target_budget: f32, l: usize) -> Self {
        let w = if self.k > MIN_POSITIVE && l > 0 {
            (target_budget * (self.k + 1.0) / (l as f32 * self.k)).clamp(MIN_POSITIVE, 1.0)
        } else {
            MIN_POSITIVE
        };
        Self { w, k: self.k }
    }

    // ── Sampling ───────────────────────────────────────────────────
    //
    // The core primitive is `sample_order_with` — RNG-agnostic via a closure
    // that yields `Uniform(0, 1)` floats. This lets katgpt-rs (which uses
    // `katgpt_types::Rng`) and riir-train (which uses `fastrand::Rng`) share
    // the same code without a hard RNG dependency. The `fastrand::Rng`
    // convenience wrappers below delegate to the core primitive.

    /// Sample a generation ordering σ — a permutation of `[0, L)`.
    ///
    /// **RNG-agnostic core primitive.** Pass a closure that yields
    /// `Uniform(0, 1)` floats — e.g. `|| rng.uniform()` for `katgpt_types::Rng`,
    /// or `|| rng.f32()` for `fastrand::Rng`.
    ///
    /// Each position gets an independent reveal time drawn via inverse-CDF;
    /// sorting ascending gives the order. Ties (measure-zero for continuous
    /// distributions, but possible with extreme k) are broken by position
    /// index (smaller index first) for deterministic left-to-right preference.
    ///
    /// Returns `vec![]` for `l == 0`, `vec![0]` for `l == 1`.
    pub fn sample_order_with(&self, l: usize, mut uniform: impl FnMut() -> f32) -> Vec<usize> {
        if l == 0 {
            return Vec::new();
        }
        if l == 1 {
            return vec![0];
        }
        // Draw reveal times, sort by (reveal_time, position) for stable tie-break.
        let mut indexed: Vec<(f32, usize)> = (0..l)
            .map(|ell| (self.reveal_time_from_uniform(uniform(), ell, l), ell))
            .collect();
        indexed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        indexed.into_iter().map(|(_, idx)| idx).collect()
    }

    /// Convenience wrapper: sample an ordering using `fastrand::Rng`.
    ///
    /// Delegates to [`sample_order_with`](Self::sample_order_with) with
    /// `|| rng.f32()`. Provided because katgpt-core already depends on
    /// `fastrand` and riir-train uses it as its primary RNG.
    pub fn sample_order(&self, l: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
        self.sample_order_with(l, || rng.f32())
    }

    /// Sample an ordering AND return the per-position reveal times (RNG-agnostic).
    ///
    /// **Core primitive** — pass a closure yielding `Uniform(0, 1)` floats.
    /// Returns `(order, reveal_times)` where `order` is the permutation and
    /// `reveal_times[i]` is the reveal time of position `i` (indexed by original
    /// position, not by order rank). Useful for debugging and for constructing
    /// weighted masks in T1.2.
    pub fn sample_order_with_reveal_times_with(
        &self,
        l: usize,
        mut uniform: impl FnMut() -> f32,
    ) -> (Vec<usize>, Vec<f32>) {
        if l == 0 {
            return (Vec::new(), Vec::new());
        }
        if l == 1 {
            return (
                vec![0],
                vec![self.reveal_time_from_uniform(uniform(), 0, 1)],
            );
        }
        let reveal_times: Vec<f32> = (0..l)
            .map(|ell| self.reveal_time_from_uniform(uniform(), ell, l))
            .collect();
        let mut indexed: Vec<(f32, usize)> = reveal_times.iter().copied().zip(0..l).collect();
        indexed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let order: Vec<usize> = indexed.into_iter().map(|(_, idx)| idx).collect();
        (order, reveal_times)
    }

    /// Convenience wrapper: sample ordering + reveal times using `fastrand::Rng`.
    pub fn sample_order_with_reveal_times(
        &self,
        l: usize,
        rng: &mut fastrand::Rng,
    ) -> (Vec<usize>, Vec<f32>) {
        self.sample_order_with_reveal_times_with(l, || rng.f32())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Static orderings (no schedule needed)
// ═══════════════════════════════════════════════════════════════════

/// Exact AR ordering: `[0, 1, ..., L-1]` (left-to-right, singleton sets).
pub fn ar_order(l: usize) -> Vec<usize> {
    (0..l).collect()
}

/// Uniform random ordering (MDLM / order-agnostic diffusion).
///
/// Fisher-Yates shuffle of `[0, L)`. **RNG-agnostic core primitive** —
/// pass a closure that yields `Uniform(0, n)` for an exclusive upper bound `n`.
pub fn uniform_order_with(l: usize, mut uniform_bounded: impl FnMut(u32) -> u32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..l).collect();
    // Fisher-Yates: for i from L-1 down to 1, swap with random j in [0, i].
    for i in (1..l).rev() {
        let j = uniform_bounded((i + 1) as u32) as usize;
        order.swap(i, j);
    }
    order
}

/// Convenience wrapper: uniform random ordering using `fastrand::Rng`.
pub fn uniform_order(l: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
    uniform_order_with(l, |n| rng.u32(0..n))
}

/// Confidence-descending ordering: positions revealed **most-confident
/// first** (DBTM attractor law, arXiv:2609.15903 §10.1.2 — Issue 811 /
/// Research 563).
///
/// The dominant token's attractor forms at t = 0 and **no non-dominant
/// attractor forms before t = 1/2** — so high-confidence positions are the
/// safe early reveals and low-confidence positions belong late in the
/// schedule (after the DBTM `commit_time_star` boundary when one is
/// available; same crate, `ignition_schedule` feature). Ties keep ascending
/// index order (deterministic; no RNG), and NaN confidences sort last.
///
/// Pairs with [`order_to_gen_steps`] for the set-causal kernel.
pub fn probability_order(confidence: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..confidence.len()).collect();
    order.sort_by(|&a, &b| {
        // Descending confidence; total order even under NaN (NaN → last).
        let (ca, cb) = (confidence[a], confidence[b]);
        match (ca.partial_cmp(&cb), ca.is_nan(), cb.is_nan()) {
            (Some(ord), false, false) => ord.reverse(),
            _ => {
                if ca.is_nan() && !cb.is_nan() {
                    std::cmp::Ordering::Greater
                } else if cb.is_nan() && !ca.is_nan() {
                    std::cmp::Ordering::Less
                } else {
                    // Both NaN (or incomparable): fall back to index order.
                    a.cmp(&b)
                }
            }
        }
    });
    order
}

// ═══════════════════════════════════════════════════════════════════
// Gen-step conversion (for the set-causal attention kernel)
// ═══════════════════════════════════════════════════════════════════

/// Convert an ordering σ to per-position generation steps (the kernel input).
///
/// Given `order = [σ_0, σ_1, ..., σ_{L-1}]` (σ_0 generated first), returns
/// `gen_step[p] = rank of position p in the ordering`. For SW-SetDLM with
/// singleton sets, this is a permutation of `[0, L)` as `u32`.
///
/// The returned buffer is what `riir_engine::forward_set_causal` and the
/// `attention_score_set_causal.wgsl` kernel consume as `position_order`.
pub fn order_to_gen_steps(order: &[usize]) -> Vec<u32> {
    if order.is_empty() {
        return Vec::new();
    }
    let l = order.len();
    let mut gen_steps = vec![0u32; l];
    for (step, &pos) in order.iter().enumerate() {
        debug_assert!(
            pos < l,
            "order contains position {pos} >= length {l} (not a valid permutation)"
        );
        gen_steps[pos] = step as u32;
    }
    gen_steps
}

/// Block-causal gen-steps: contiguous blocks of `block_size` share a step.
///
/// Positions `[0, block_size)` → step 0, `[block_size, 2·block_size)` → step 1, etc.
/// The final block may be shorter. This is what D2F/block-diffusion uses.
pub fn block_causal_gen_steps(l: usize, block_size: usize) -> Vec<u32> {
    assert!(block_size > 0, "block_size must be > 0");
    (0..l).map(|p| (p / block_size) as u32).collect()
}

/// MDLM gen-steps: all positions share step 0 (fully bidirectional).
pub fn mdlm_gen_steps(l: usize) -> Vec<u32> {
    vec![0u32; l]
}

// ═══════════════════════════════════════════════════════════════
// Plan 602 T2.3 — optional decode-order schedule variants
// (feature `decode_order_metrics`; promote-only-on-G3, demote silently
// if the Phase-3 G3 GOAT fails — the plan's standing rule)
// ═══════════════════════════════════════════════════════════════

/// Confidence-threshold τ eligibility (Plan 602 T2.3, arXiv:2609.20751's
/// parallel-decoding policy — the paper holds the quality–speed frontier
/// under τ unmasking to ~8×): the positions whose current confidence is
/// `≥ τ` are eligible to commit this pass.
///
/// The dynamic counterpart of [`probability_order`]'s static full sort
/// (same DBTM attractor law, Issue 811 / Research 563: confident positions
/// are the safe early reveals) — a decode loop calls this per pass with
/// the current confidences instead of pre-committing to one ordering.
/// Ascending indices; NaN confidences are never eligible (`NaN ≥ τ` is
/// false under IEEE — incomparable confidences stay masked, never a
/// false commit). The count the caller needs for NFE accounting is just
/// the length of the returned slice.
#[cfg(feature = "decode_order_metrics")]
pub fn confidence_threshold_eligible(confidence: &[f32], tau: f32) -> Vec<usize> {
    confidence
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c >= tau)
        .map(|(i, _)| i)
        .collect()
}

/// 1/λ_t-shaped unmask-slot allocation (Plan 602 T2.3): split `l` unmask
/// slots across `steps` decode passes with per-pass counts shaped by the
/// paper's time-reweighted objective `w_t = 1/λ_t` (equal weight per masked
/// target, linear schedule `α_t = 1−t`) transferred to inference — λ_t is
/// the remaining-masked count under the linear baseline, so early
/// (heavily-masked, hard) passes commit FEW tokens each and the compute
/// budget concentrates on the hard early region, while late passes take
/// larger, confident batches.
///
/// Contract: counts are **non-decreasing over the active prefix** (the
/// passes that carry work), every active pass commits ≥ 1 (surplus passes
/// when `steps > l` idle with 0 at the END — the hard early region keeps
/// its passes), and the total is exactly `l` (min-1 pre-assignment +
/// largest-remainder distribution, ties to the later pass). `steps == 0`
/// panics; `l == 0` returns all zeros.
#[cfg(feature = "decode_order_metrics")]
pub fn inverse_lambda_slot_counts(l: usize, steps: usize) -> Vec<usize> {
    assert!(steps > 0, "steps must be positive");
    let mut counts = vec![0usize; steps];
    if l == 0 {
        return counts;
    }
    // At most `l` passes carry work; surplus passes idle at the end.
    let active = steps.min(l);
    // Baseline remaining-masked at the START of active pass t (0-indexed):
    // r_t = ceil(l · (active − t) / active), ≥ 1 for every t < active.
    // Weight w_t = 1/r_t — increasing in t (later = fewer masked = larger
    // per-pass share), the transferred shape of the training objective.
    let weights: Vec<f64> = (0..active)
        .map(|t| {
            let r = (l * (active - t)).div_ceil(active);
            1.0 / r as f64
        })
        .collect();
    // Every active pass commits at least one token; distribute the
    // remaining l − active by largest remainder of the exact quotas
    // extra · w_t / Σw.
    for c in counts[..active].iter_mut() {
        *c = 1;
    }
    let extra = l - active;
    if extra == 0 {
        return counts;
    }
    let wsum: f64 = weights.iter().sum();
    let quota = |t: usize| extra as f64 * weights[t] / wsum;
    let mut floors: Vec<usize> = (0..active).map(|t| quota(t) as usize).collect();
    let mut assigned: usize = floors.iter().sum();
    // Largest fractional part first (descending); ties to the LATER pass
    // (keeps the profile non-decreasing — a tie broken earlier could
    // invert it).
    let mut order: Vec<usize> = (0..active).collect();
    order.sort_by(|&a, &b| {
        let (fa, fb) = (quota(a) - floors[a] as f64, quota(b) - floors[b] as f64);
        fa.partial_cmp(&fb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
            .then(b.cmp(&a))
    });
    let mut idx = 0;
    while assigned < extra {
        floors[order[idx % active]] += 1;
        assigned += 1;
        idx += 1;
    }
    for (t, &f) in floors.iter().enumerate() {
        counts[t] += f;
    }
    counts
}

// ── Plan 602 T3.2: the gap-predictor (Research 575 §distilled-2) ────────

/// The calibrated (ALR, AGR) signature table over the w-sweep —
/// `bench_ar_ness_w_sweep` (Plan 602 T3.1, micro set-causal fixture,
/// L=8/V=8, trained at w=0.5, 16 seeded greedy decodes per w). The
/// predictor inverts this table; re-calibrate by re-measuring and editing
/// these constants IN ONE PLACE (the table is the calibration).
#[cfg(feature = "decode_order_metrics")]
pub const ORDER_STATS_TO_W_TABLE: [(f32, f64, f64); 6] = [
    // (w, ALR, AGR) — measured 2026-09-20, this bench fixture.
    (0.1, 1.000, 1.000),
    (0.3, 0.804, 0.944),
    (0.5, 0.589, 0.797),
    (0.7, 0.580, 0.701),
    (0.9, 0.536, 0.562),
    (1.0, 0.518, 0.547),
];

/// The gap-predictor (Plan 602 T3.2, arXiv:2609.20751 §decode-order): from
/// a probe decode's measured (ALR, AGR), recommend the schedule width `w`
/// whose calibrated signature is nearest (L2) — the measured mechanism
/// behind the Plans 379–384 w=0.5 win. Signature-matching semantics:
/// decode with the schedule whose induced order statistics best match the
/// regime the probe measured — a strictly-AR signature (ALR ≈ AGR ≈ 1)
/// maps to the sequential end; a uniform-random signature (both ≈ 0.5) to
/// the parallel end; an off-curve hybrid signature (mid ALR + high AGR —
/// the paper's locally-out-of-order, globally-LTR band) maps to the
/// mid table (the w=0.5 class). NOTE this is deliberately NOT a
/// "high gap ⇒ crank parallelism" rule: in the w parameterization the
/// high-AGR signatures live at LOW w, so the recommendation follows the
/// measured regime, never a headroom extrapolation.
///
/// Nearest-row over [`ORDER_STATS_TO_W_TABLE`]; ties → the LOWER w (the
/// conservative/sequential side — never recommend parallelism off a tie).
/// Inputs are clamped into [0,1]; a NaN input returns the conservative
/// 0.1 endpoint (an unmeasurable probe must not look parallel-safe).
#[cfg(feature = "decode_order_metrics")]
pub fn predict_w_from_order_stats(alr: f32, agr: f32) -> f32 {
    if !alr.is_finite() || !agr.is_finite() {
        return ORDER_STATS_TO_W_TABLE[0].0;
    }
    let (a, g) = (alr.clamp(0.0, 1.0) as f64, agr.clamp(0.0, 1.0) as f64);
    let mut best_w = ORDER_STATS_TO_W_TABLE[0].0;
    let mut best_d = f64::INFINITY;
    for &(w, ca, cg) in ORDER_STATS_TO_W_TABLE.iter() {
        let d = (a - ca) * (a - ca) + (g - cg) * (g - cg);
        // Strict `<`: ties keep the earlier (lower-w) row.
        if d < best_d {
            best_d = d;
            best_w = w;
        }
    }
    best_w
}

/// The RESIDUAL gap-predictor (Plan 602 T3.2, the paper's actual measurement
/// posture): probe the model under a KNOWN schedule (`probe_w`), measure
/// the trajectory's (ALR, AGR), and shift the recommendation by the AR-DRAG
/// — the deviation of the measured ALR from the schedule's own calibrated
/// signature at `probe_w` (nearest row of [`ORDER_STATS_TO_W_TABLE`]):
/// ΔALR > 0 means the model defers commits until left context lands (drags
/// toward AR) → recommend a LOWER w; ΔALR ≤ 0 keeps `probe_w`.
///
/// Measured on the G3 fixture (real-text denoisers, probe w=0.5):
/// AR-trained ΔALR = +0.144, uniform-trained ΔALR = +0.015 — the
/// discrimination the mdlm-endpoint probe could not produce (a confident
/// model commits everything at once → all-ties → no signal).
///
/// `SHIFT = 2`: ΔALR of +0.2 moves w by −0.4 (the calibrated span from
/// w=0.5's 0.589 to w=0.1's 1.000 is ~0.4 ALR over ~0.4 w — unit slope,
/// doubled for the conservative direction). Output clamped to
/// [0.1, 1.0]; NaN measurements return `probe_w` clamped (no signal —
/// keep the incumbent posture, never extrapolate).
#[cfg(feature = "decode_order_metrics")]
pub fn predict_w_residual(alr: f32, agr: f32, probe_w: f32) -> f32 {
    const SHIFT: f32 = 2.0;
    let probe_w = probe_w.clamp(ORDER_STATS_TO_W_TABLE[0].0, 1.0);
    if !alr.is_finite() || !agr.is_finite() {
        return probe_w;
    }
    // Nearest calibrated row to probe_w (the schedule's own signature).
    let mut cal_alr = ORDER_STATS_TO_W_TABLE[0].1;
    let mut best_d = f64::INFINITY;
    for &(w, ca, _) in ORDER_STATS_TO_W_TABLE.iter() {
        let d = (probe_w as f64 - w as f64) * (probe_w as f64 - w as f64);
        if d < best_d {
            best_d = d;
            cal_alr = ca;
        }
    }
    let delta = alr as f64 - cal_alr;
    if delta <= 0.0 {
        probe_w
    } else {
        (probe_w - SHIFT * delta as f32).clamp(ORDER_STATS_TO_W_TABLE[0].0, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: verify a slice is a permutation of [0, n) ──
    fn assert_permutation(order: &[usize], n: usize) {
        assert_eq!(order.len(), n, "order length mismatch");
        let mut seen = vec![false; n];
        for &p in order {
            assert!(p < n, "position {p} out of range [0, {n})");
            assert!(!seen[p], "position {p} appears twice");
            seen[p] = true;
        }
    }

    // ── Helper: verify gen_steps is the inverse of order ──
    fn assert_gen_steps_inverse(order: &[usize], gen_steps: &[u32]) {
        assert_eq!(order.len(), gen_steps.len());
        for (step, &pos) in order.iter().enumerate() {
            assert_eq!(
                gen_steps[pos], step as u32,
                "gen_step[{pos}] should be {step}"
            );
        }
    }

    // ── Construction & validation ──

    #[test]
    fn test_new_clamps_w() {
        let s = PositionOffsetSchedule::new(2.0);
        assert!((s.w - 1.0).abs() < 1e-6, "w should clamp to 1.0");
        let s = PositionOffsetSchedule::new(-1.0);
        assert!(s.w > 0.0, "w should clamp to positive");
        assert!((s.k - 1.0).abs() < 1e-6, "new() should set k=1");
    }

    #[test]
    fn test_shaped_clamps_both() {
        let s = PositionOffsetSchedule::shaped(0.5, 0.0);
        assert!((s.w - 0.5).abs() < 1e-6);
        assert!(s.k > 0.0, "k should clamp to positive");
    }

    #[test]
    fn test_ar_endpoint() {
        let s = PositionOffsetSchedule::ar();
        assert!(s.w < 1e-3, "ar() should have tiny w");
    }

    #[test]
    fn test_diffusion_endpoint() {
        let s = PositionOffsetSchedule::diffusion();
        assert!((s.w - 1.0).abs() < 1e-6);
        assert!((s.k - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_is_paper_winner() {
        // Plan 312 Phase 2: w=0.5 won the real-text battle by 0.71 NLL.
        let s = PositionOffsetSchedule::default();
        assert!((s.w - 0.5).abs() < 1e-6);
        assert!((s.k - 1.0).abs() < 1e-6);
    }

    // ── Offset ──

    #[test]
    fn test_offset_zero_length() {
        let s = PositionOffsetSchedule::new(0.5);
        assert_eq!(s.offset(0, 0), 0.0);
        assert_eq!(s.offset(0, 1), 0.0);
    }

    #[test]
    fn test_offset_endpoints() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        // First position: offset 0.
        assert!((s.offset(0, l) - 0.0).abs() < 1e-6);
        // Last position: offset (1-w) = 0.5.
        assert!((s.offset(l - 1, l) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_offset_diffusion_all_zero() {
        let s = PositionOffsetSchedule::diffusion();
        let l = 8;
        for ell in 0..l {
            assert!(
                (s.offset(ell, l) - 0.0).abs() < 1e-6,
                "diffusion offset should be 0"
            );
        }
    }

    #[test]
    fn test_offset_monotone_increasing() {
        let s = PositionOffsetSchedule::new(0.3);
        let l = 16;
        for ell in 1..l {
            assert!(
                s.offset(ell, l) >= s.offset(ell - 1, l),
                "offset should be monotone non-decreasing"
            );
        }
    }

    // ── CDF (alpha) ──

    #[test]
    fn test_alpha_below_interval_is_zero() {
        let s = PositionOffsetSchedule::new(0.5);
        let a = s.offset(4, 8);
        assert_eq!(s.alpha(a - 0.01, 4, 8), 0.0);
        assert_eq!(s.alpha(a, 4, 8), 0.0);
    }

    #[test]
    fn test_alpha_above_interval_is_one() {
        let s = PositionOffsetSchedule::new(0.5);
        let a = s.offset(4, 8);
        assert_eq!(s.alpha(a + s.w, 4, 8), 1.0);
        assert_eq!(s.alpha(a + s.w + 0.01, 4, 8), 1.0);
    }

    #[test]
    fn test_alpha_midpoint_linear_is_half() {
        // k=1 → linear CDF. Midpoint of interval should give alpha=0.5.
        let s = PositionOffsetSchedule::new(0.5);
        let a = s.offset(4, 8);
        let mid = a + s.w * 0.5;
        assert!((s.alpha(mid, 4, 8) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_alpha_monotone_non_decreasing() {
        let s = PositionOffsetSchedule::shaped(0.4, 0.7);
        let l = 8;
        for ell in 0..l {
            let a = s.offset(ell, l);
            let mut prev = 0.0f32;
            let n_steps = 100;
            for i in 0..=n_steps {
                let tau = a + s.w * (i as f32 / n_steps as f32);
                let alpha = s.alpha(tau, ell, l);
                assert!(
                    alpha >= prev - 1e-6,
                    "alpha not monotone at ell={ell}, tau={tau}"
                );
                prev = alpha;
            }
        }
    }

    #[test]
    fn test_alpha_in_zero_one_range() {
        let s = PositionOffsetSchedule::shaped(0.6, 2.0);
        for ell in 0..8 {
            for i in 0..100 {
                let tau = i as f32 / 100.0;
                let a = s.alpha(tau, ell, 8);
                assert!((0.0..=1.0).contains(&a), "alpha out of [0,1]: {a}");
            }
        }
    }

    // ── Inverse CDF (reveal_time_from_uniform) ──

    #[test]
    fn test_reveal_time_endpoints() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        for ell in 0..l {
            let a = s.offset(ell, l);
            // u=0 → reveal time = a (interval start).
            assert!((s.reveal_time_from_uniform(0.0, ell, l) - a).abs() < 1e-6);
            // u=1 → reveal time = a + w (interval end).
            assert!((s.reveal_time_from_uniform(1.0, ell, l) - (a + s.w)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_reveal_time_in_interval() {
        let s = PositionOffsetSchedule::shaped(0.4, 0.5);
        let l = 16;
        for ell in 0..l {
            let a = s.offset(ell, l);
            for _ in 0..100 {
                let mut rng = fastrand::Rng::new();
                let r = s.reveal_time_from_uniform(rng.f32(), ell, l);
                assert!(
                    r >= a - 1e-6 && r <= a + s.w + 1e-6,
                    "reveal time {r} out of interval"
                );
            }
        }
    }

    #[test]
    fn test_inverse_cdf_roundtrip() {
        // alpha(reveal_time_from_uniform(u)) should recover u for k=1 (linear).
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        for ell in 0..l {
            for &u in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                let r = s.reveal_time_from_uniform(u, ell, l);
                let recovered = s.alpha(r, ell, l);
                assert!(
                    (recovered - u).abs() < 1e-4,
                    "roundtrip failed: u={u}, recovered={recovered}"
                );
            }
        }
    }

    // ── Expected budget ──

    #[test]
    fn test_budget_formula() {
        let s = PositionOffsetSchedule::shaped(0.5, 2.0);
        let l = 10;
        // C̄ = L · w · k / (k+1) = 10 · 0.5 · 2 / 3 = 10/3 ≈ 3.333
        let expected = 10.0 * 0.5 * 2.0 / 3.0;
        assert!((s.expected_budget(l) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_budget_ar_tends_to_zero() {
        // AR: w → 0, k=1. C̄ = L · w · 0.5 → 0.
        let s = PositionOffsetSchedule::ar();
        assert!(s.expected_budget(16) < 0.01, "AR budget should tend to 0");
    }

    #[test]
    fn test_budget_diffusion_is_l_half() {
        // Diffusion: w=1, k=1. C̄ = L · 1 · 0.5 = L/2.
        let s = PositionOffsetSchedule::diffusion();
        assert!((s.expected_budget(16) - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_with_matched_budget() {
        // Target budget B=4 for L=16, k=1.
        // w = B · (k+1) / (L · k) = 4 · 2 / (16 · 1) = 0.5.
        let base = PositionOffsetSchedule::new(0.3); // original w doesn't matter
        let matched = base.with_matched_budget(4.0, 16);
        assert!(
            (matched.w - 0.5).abs() < 1e-5,
            "w should be 0.5, got {}",
            matched.w
        );
        assert!((matched.k - 1.0).abs() < 1e-5, "k should be preserved");
        // Verify: expected_budget should now be ≈ 4.
        assert!((matched.expected_budget(16) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_with_matched_budget_clamps_to_one() {
        // If target budget > L·k/(k+1) (= L/2 for k=1), w should clamp to 1.
        let base = PositionOffsetSchedule::new(0.5);
        let matched = base.with_matched_budget(100.0, 16); // way over max
        assert!((matched.w - 1.0).abs() < 1e-6, "w should clamp to 1.0");
    }

    // ── sample_order ──

    #[test]
    fn test_sample_order_empty() {
        let s = PositionOffsetSchedule::new(0.5);
        let mut rng = fastrand::Rng::with_seed(42);
        assert!(s.sample_order(0, &mut rng).is_empty());
    }

    #[test]
    fn test_sample_order_single() {
        let s = PositionOffsetSchedule::new(0.5);
        let mut rng = fastrand::Rng::with_seed(42);
        assert_eq!(s.sample_order(1, &mut rng), vec![0]);
    }

    #[test]
    fn test_sample_order_is_permutation() {
        let s = PositionOffsetSchedule::new(0.5);
        let mut rng = fastrand::Rng::with_seed(42);
        for l in [2, 4, 8, 16, 32] {
            let order = s.sample_order(l, &mut rng);
            assert_permutation(&order, l);
        }
    }

    #[test]
    fn test_sample_order_deterministic_with_seed() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 16;
        let order1 = s.sample_order(l, &mut fastrand::Rng::with_seed(123));
        let order2 = s.sample_order(l, &mut fastrand::Rng::with_seed(123));
        assert_eq!(order1, order2, "same seed should give same order");
    }

    #[test]
    fn test_sample_order_different_seeds_differ() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 32;
        let order1 = s.sample_order(l, &mut fastrand::Rng::with_seed(1));
        let order2 = s.sample_order(l, &mut fastrand::Rng::with_seed(2));
        // Overwhelmingly likely to differ for l=32.
        assert_ne!(
            order1, order2,
            "different seeds should (almost certainly) differ"
        );
    }

    #[test]
    fn test_ar_schedule_near_left_to_right() {
        // w very small → near-AR ordering. The schedule with w=1e-6 should
        // almost always produce [0, 1, ..., L-1] because intervals barely overlap.
        let s = PositionOffsetSchedule::ar();
        let l = 16;
        let mut near_ar_count = 0;
        for seed in 0..100 {
            let order = s.sample_order(l, &mut fastrand::Rng::with_seed(seed));
            if order == (0..l).collect::<Vec<_>>() {
                near_ar_count += 1;
            }
        }
        // With w ≈ 1e-6 and L=16, overlap is negligible → expect ≥95% exact AR.
        assert!(
            near_ar_count >= 95,
            "AR schedule should produce near-AR orderings: {near_ar_count}/100"
        );
    }

    #[test]
    fn test_diffusion_schedule_is_uniform() {
        // w=1 → all offsets are 0, so reveal times are i.i.d. → uniform random order.
        // Check: position 0 is first roughly 1/L of the time.
        let s = PositionOffsetSchedule::diffusion();
        let l = 8;
        let mut first_is_zero = 0;
        let n_trials = 1000;
        for seed in 0..n_trials {
            let order = s.sample_order(l, &mut fastrand::Rng::with_seed(seed as u64));
            if order[0] == 0 {
                first_is_zero += 1;
            }
        }
        // Expected: 1000/8 = 125. Allow [80, 170] for randomness.
        assert!(
            (80..=170).contains(&first_is_zero),
            "diffusion should be ~uniform: position 0 first {first_is_zero}/{n_trials} (expected ~125)"
        );
    }

    #[test]
    fn test_sample_order_with_reveal_times() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        let (order, reveal_times) =
            s.sample_order_with_reveal_times(l, &mut fastrand::Rng::with_seed(42));
        assert_permutation(&order, l);
        assert_eq!(reveal_times.len(), l);
        // Reveal times should be sorted when read in order.
        for i in 1..l {
            let prev = reveal_times[order[i - 1]];
            let curr = reveal_times[order[i]];
            assert!(
                prev <= curr + 1e-6,
                "reveal times not sorted in order: {prev} > {curr}"
            );
        }
    }

    // ── Static orderings ──

    #[test]
    fn test_ar_order() {
        assert_eq!(ar_order(0), Vec::<usize>::new());
        assert_eq!(ar_order(1), vec![0]);
        assert_eq!(ar_order(5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_uniform_order_is_permutation() {
        let mut rng = fastrand::Rng::with_seed(42);
        for l in [2, 4, 8, 16] {
            let order = uniform_order(l, &mut rng);
            assert_permutation(&order, l);
        }
    }

    #[test]
    fn test_probability_order_descending_and_permutation() {
        let conf = [0.2f32, 0.9, 0.5, 0.9, 0.1];
        let order = probability_order(&conf);
        assert_permutation(&order, 5);
        // Descending confidence: ties (positions 1, 3) keep ascending index.
        assert_eq!(order, vec![1, 3, 2, 0, 4]);
        // Gen-steps round-trip: most-confident position gets step 0.
        let steps = order_to_gen_steps(&order);
        assert_eq!(steps[1], 0);
        assert_eq!(steps[4], 4);
    }

    #[test]
    fn test_probability_order_deterministic_and_degenerate() {
        // No RNG anywhere: two calls, one order.
        let conf = [0.3f32, 0.8, 0.3];
        assert_eq!(probability_order(&conf), probability_order(&conf));
        // Equal confidences ⟹ ascending index order.
        assert_eq!(probability_order(&[0.5f32; 4]), vec![0, 1, 2, 3]);
        // Empty and singleton.
        assert!(probability_order(&[]).is_empty());
        assert_eq!(probability_order(&[0.7f32]), vec![0]);
    }

    #[test]
    fn test_probability_order_nan_sorts_last() {
        let conf = [f32::NAN, 0.6f32, f32::NAN, 0.2];
        let order = probability_order(&conf);
        assert_permutation(&order, 4);
        assert_eq!(&order[..2], &[1, 3]); // finite first, descending
        assert!(order[2..].contains(&0) && order[2..].contains(&2)); // NaN last
    }

    #[test]
    fn test_uniform_order_deterministic() {
        let l = 16;
        let o1 = uniform_order(l, &mut fastrand::Rng::with_seed(7));
        let o2 = uniform_order(l, &mut fastrand::Rng::with_seed(7));
        assert_eq!(o1, o2);
    }

    // ── Gen-step conversion ──

    #[test]
    fn test_order_to_gen_steps_identity() {
        // AR order: gen_steps should be [0, 1, 2, ..., L-1].
        let order = ar_order(8);
        let gs = order_to_gen_steps(&order);
        assert_eq!(gs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_order_to_gen_steps_reversed() {
        // Reversed order [L-1, ..., 1, 0]: position 0 is last → gen_step[0] = L-1.
        let order: Vec<usize> = (0..8).rev().collect();
        let gs = order_to_gen_steps(&order);
        assert_eq!(gs[0], 7);
        assert_eq!(gs[7], 0);
    }

    #[test]
    fn test_order_to_gen_steps_inverse_property() {
        let s = PositionOffsetSchedule::new(0.5);
        let order = s.sample_order(16, &mut fastrand::Rng::with_seed(42));
        let gs = order_to_gen_steps(&order);
        assert_gen_steps_inverse(&order, &gs);
    }

    #[test]
    fn test_order_to_gen_steps_empty() {
        assert!(order_to_gen_steps(&[]).is_empty());
    }

    #[test]
    fn test_block_causal_gen_steps() {
        // L=8, block_size=4: positions 0-3 → step 0, positions 4-7 → step 1.
        let gs = block_causal_gen_steps(8, 4);
        assert_eq!(gs, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_block_causal_gen_steps_partial_last_block() {
        // L=10, block_size=4: [0,0,0,0, 1,1,1,1, 2,2].
        let gs = block_causal_gen_steps(10, 4);
        assert_eq!(gs, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn test_mdlm_gen_steps() {
        let gs = mdlm_gen_steps(8);
        assert_eq!(gs, vec![0u32; 8]);
    }

    // ── Cross-validation against the PoC reference ──

    #[test]
    fn test_matches_poc_reference_budget() {
        // The PoC at riir-ai/crates/riir-poc/src/set_diffusion_poc.rs implements
        // the same formula. Verify our production version matches at the paper-
        // winning config (w=0.5, k=1, L=16).
        let s = PositionOffsetSchedule::new(0.5);
        let l = 16;
        // PoC: expected_budget = L · w · k / (k+1) = 16 · 0.5 · 1 / 2 = 4.0.
        assert!((s.expected_budget(l) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_matches_poc_reference_offset() {
        // PoC offset for ell=4, L=8, w=0.5: (4/7) · 0.5 = 2/7 ≈ 0.2857.
        let s = PositionOffsetSchedule::new(0.5);
        let expected = (4.0f32 / 7.0) * 0.5;
        assert!((s.offset(4, 8) - expected).abs() < 1e-6);
    }

    // ── Sanity: schedule interpolates between AR and diffusion ──

    #[test]
    fn test_schedule_interpolation_property() {
        // As w increases from ~0 to 1, the average "left-to-right-ness" should
        // decrease monotonically. We measure this as the Kendall tau distance
        // from the identity (AR) ordering — more inversions = less AR-like.
        let l = 16;
        let n_trials = 200;

        let avg_inversions = |w: f32| -> f32 {
            let s = PositionOffsetSchedule::new(w);
            let mut total = 0usize;
            for seed in 0..n_trials {
                let order = s.sample_order(l, &mut fastrand::Rng::with_seed(seed as u64));
                total += count_inversions(&order);
            }
            total as f32 / n_trials as f32
        };

        let inv_ar = avg_inversions(0.01);
        let inv_mid = avg_inversions(0.5);
        let inv_diff = avg_inversions(1.0);

        // AR-like should have few inversions; diffusion should have ~L(L-1)/4.
        let max_inv = (l * (l - 1) / 2) as f32;
        assert!(
            inv_ar < max_inv * 0.1,
            "AR-like (w=0.01) should have <10% max inversions, got {inv_ar:.1}/{max_inv:.0}"
        );
        assert!(
            inv_diff > max_inv * 0.35,
            "diffusion (w=1.0) should have >35% max inversions, got {inv_diff:.1}/{max_inv:.0}"
        );
        assert!(
            inv_mid > inv_ar && inv_mid < inv_diff,
            "intermediate w should interpolate: inv_ar={inv_ar:.1}, inv_mid={inv_mid:.1}, inv_diff={inv_diff:.1}"
        );
    }

    /// Count the number of inversions in a permutation (pairs (i,j) with i<j but order[i]>order[j]).
    fn count_inversions(order: &[usize]) -> usize {
        let n = order.len();
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if order[i] > order[j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Plan 602 T2.3 schedule-variant tests (feature-gated with the fns).
    #[cfg(all(test, feature = "decode_order_metrics"))]
    mod decode_order_variants {
        use super::*;

        // ── confidence_threshold_eligible ─────────────────────────────

        #[test]
        fn threshold_picks_confident_positions_ascending() {
            let conf = [0.9f32, 0.2, 0.7, 0.95, 0.1];
            assert_eq!(confidence_threshold_eligible(&conf, 0.7), vec![0, 2, 3]);
            assert_eq!(confidence_threshold_eligible(&conf, 0.9), vec![0, 3]);
        }

        #[test]
        fn threshold_boundary_is_inclusive() {
            let conf = [0.5f32, 0.4999999, 0.5000001];
            // τ exactly at a confidence includes it (≥, not >).
            assert_eq!(confidence_threshold_eligible(&conf, 0.5), vec![0, 2]);
            // τ above everything / below everything.
            assert_eq!(
                confidence_threshold_eligible(&conf, 0.6),
                Vec::<usize>::new()
            );
            assert_eq!(confidence_threshold_eligible(&conf, 0.0), vec![0, 1, 2]);
        }

        #[test]
        fn threshold_nan_never_eligible_and_empty_is_empty() {
            let conf = [f32::NAN, 0.5, f32::NAN];
            assert_eq!(confidence_threshold_eligible(&conf, 0.3), vec![1]);
            assert_eq!(confidence_threshold_eligible(&conf, 0.0), vec![1]);
            let empty: [f32; 0] = [];
            assert!(confidence_threshold_eligible(&empty, 0.5).is_empty());
        }

        #[test]
        fn threshold_agrees_with_probability_order_prefix() {
            // With distinct confidences, the eligible set is exactly the
            // top-|eligible| prefix of probability_order — the dynamic τ
            // policy and the static confidence sort name the same tokens.
            let conf: Vec<f32> = (0..24).map(|i| ((i * 37) % 97) as f32 / 97.0).collect();
            for &tau in &[0.1f32, 0.3, 0.5, 0.7, 0.9] {
                let eligible = confidence_threshold_eligible(&conf, tau);
                let order = probability_order(&conf);
                let top: std::collections::HashSet<usize> =
                    order.iter().take(eligible.len()).copied().collect();
                let got: std::collections::HashSet<usize> = eligible.iter().copied().collect();
                assert_eq!(got, top, "tau={tau}");
            }
        }

        // ── inverse_lambda_slot_counts ────────────────────────────────

        #[test]
        fn slot_counts_sum_exact_and_are_monotone() {
            for &l in &[0usize, 1, 2, 3, 6, 10, 32, 100, 257] {
                for &steps in &[1usize, 2, 3, 4, 7, 16, 64] {
                    let c = inverse_lambda_slot_counts(l, steps);
                    assert_eq!(c.len(), steps, "l={l} steps={steps}");
                    assert_eq!(c.iter().sum::<usize>(), l, "l={l} steps={steps}: {c:?}");
                    // Non-decreasing over the active prefix (trailing zeros
                    // are idle surplus passes, not part of the profile).
                    let active = steps.min(l);
                    assert!(
                        c[..active].windows(2).all(|w| w[0] <= w[1]),
                        "non-monotone l={l} steps={steps}: {c:?}"
                    );
                    // Zeros only ever trail (surplus passes idle at the END).
                    if let Some(first_zero) = c.iter().position(|&x| x == 0) {
                        assert!(
                            c[first_zero..].iter().all(|&x| x == 0),
                            "interior zero l={l} steps={steps}: {c:?}"
                        );
                    }
                    // Active passes commit ≥ 1.
                    assert!(
                        c[..active].iter().all(|&x| x >= 1),
                        "l={l} steps={steps}: {c:?}"
                    );
                }
            }
        }

        #[test]
        fn slot_counts_hand_computed() {
            // l=6, steps=3: active=3, min-1 base [1,1,1], extra=3.
            // r_t = ceil(6·(3−t)/3) = [6, 4, 2] → w = [1/6, 1/4, 1/2],
            // Σw = 11/12. quotas = 3·w/Σw = [6/11, 9/11, 18/11]
            // → floors [0, 0, 1] (Σ=1), remainder 2 → fracs
            // [6/11, 9/11, 7/11] → t1 then t2 get +1.
            // Final: [1+0, 1+1, 1+1+1] = [1, 2, 3].
            assert_eq!(inverse_lambda_slot_counts(6, 3), vec![1, 2, 3]);
            // Degenerate shapes.
            assert_eq!(inverse_lambda_slot_counts(5, 1), vec![5]);
            assert_eq!(inverse_lambda_slot_counts(1, 4), vec![1, 0, 0, 0]);
            assert_eq!(inverse_lambda_slot_counts(0, 3), vec![0, 0, 0]);
            assert_eq!(inverse_lambda_slot_counts(4, 4), vec![1, 1, 1, 1]);
        }

        #[test]
        fn slot_counts_backload_batch_size() {
            // l=100, steps=10: quota_0 = 90·(1/100)/Σw ≈ 3.07 → the first
            // (hardest) pass commits ~4 tokens while the last commits ~32 —
            // the 1/λ_t shape concentrates per-pass batch size late, keeping
            // the early region's careful small batches.
            let c = inverse_lambda_slot_counts(100, 10);
            assert_eq!(c.iter().sum::<usize>(), 100);
            assert!(*c.last().unwrap() >= 4 * c[0], "profile must grow: {c:?}");
        }

        // ── predict_w_from_order_stats (T3.2 gap-predictor) ───────────

        #[test]
        fn predictor_recovers_calibrated_signatures() {
            // Exact table signatures invert exactly (ties → lower w).
            for &(w, ca, cg) in ORDER_STATS_TO_W_TABLE.iter() {
                assert_eq!(
                    predict_w_from_order_stats(ca as f32, cg as f32),
                    w,
                    "signature ({ca}, {cg})"
                );
            }
        }

        #[test]
        fn predictor_endpoints_and_conservatism() {
            // Strict-AR signature → the sequential endpoint.
            assert_eq!(predict_w_from_order_stats(1.0, 1.0), 0.1);
            // Uniform-random signature → the parallel endpoint (nearest
            // row is w=1.0: (0.518, 0.547) at d≈0.0025 vs w=0.9 at d≈0.0051).
            assert_eq!(predict_w_from_order_stats(0.5, 0.5), 1.0);
            // NaN → conservative 0.1 (an unmeasurable probe is never
            // parallel-safe).
            assert_eq!(predict_w_from_order_stats(f32::NAN, 0.7), 0.1);
            assert_eq!(predict_w_from_order_stats(0.6, f32::NAN), 0.1);
            // The paper's hybrid band (ALR ~0.6, AGR ~0.9 — locally
            // out-of-order, globally LTR) maps to the mid table, never the
            // sequential or fully-parallel endpoints.
            let w = predict_w_from_order_stats(0.60, 0.90);
            assert!(
                (0.3..=0.5).contains(&w),
                "hybrid-band signature mapped to w={w}"
            );
        }

        #[test]
        fn predictor_residual_shifts_against_ar_drag() {
            // No drag (measured == calibrated, or below): keep probe_w.
            assert_eq!(predict_w_residual(0.589, 0.797, 0.5), 0.5);
            assert_eq!(predict_w_residual(0.40, 0.70, 0.5), 0.5);
            // AR drag lowers w: Δ=+0.144 (the measured AR-trained fixture)
            // → 0.5 − 2·0.144 = 0.212.
            assert!((predict_w_residual(0.733, 0.9, 0.5) - 0.212).abs() < 1e-3);
            // Clamped at the sequential endpoint for huge drag.
            assert_eq!(predict_w_residual(1.0, 1.0, 0.5), 0.1);
            // NaN measurements keep the incumbent posture.
            assert_eq!(predict_w_residual(f32::NAN, 0.8, 0.7), 0.7);
            assert_eq!(predict_w_residual(0.6, f32::NAN, 0.7), 0.7);
        }
    }
}
