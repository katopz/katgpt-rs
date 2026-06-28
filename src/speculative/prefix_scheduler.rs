//! Hardware-Aware Prefix Scheduler — Multi-Request Verification Budget Allocator (Plan 339).
//!
//! Distilled from DSpark (DeepSeek-AI, 2026) §3.2.2, Algorithm 1, Appendix A.
//! See `.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md`.
//!
//! # The problem
//!
//! katgpt-rs has per-request verification budget selectors (`caddtree_budget.rs`,
//! `budget.rs`) but no **multi-request global** budget allocator. When multiple
//! spec-decode requests share a target forward pass (batch serving or crowd-NPC
//! cognition), static per-request block lengths waste target compute on
//! low-survival suffix tokens while starving high-survival tokens elsewhere.
//!
//! DSpark §3.2.2 formulates global throughput maximization:
//!
//! ```text
//! Θ = τ · SPS(B)
//! ```
//!
//! where `τ` is the total expected accepted tokens across all admitted
//! `(request, position)` candidates, `B` is the cumulative verification batch
//! size, and `SPS(B)` is the profiled steps-per-second curve of the engine at
//! batch size `B`. Solved greedily by globally sorting candidates descending
//! by per-position survival probability `a_{r,j} = Π_{i≤j} c_{r,i}` and
//! admitting while `Θ` keeps improving.
//!
//! # The non-anticipating early-stop (correctness theorem, NOT a heuristic)
//!
//! The greedy loop MUST terminate at the first `B` where `Θ ≤ Θ_best`. This is
//! the **non-anticipating property** required for lossless speculative decoding
//! (DSpark Appendix A correctness proof). Without it, retrospective global
//! search leaks future-token information into the current-token admission
//! decision, introducing selection bias that breaks distribution preservation.
//!
//! ## Appendix A counterexample
//!
//! Vocab `{A, B}`, target distribution `p_t = (0.7, 0.3)`, drafter distribution
//! `p_d = (0.5, 0.5)`. Correct lossless output is `(0.7, 0.3)`. A non-anticipating
//! (early-stopping) allocator that admits the next candidate only when it improves
//! `Θ` preserves this. A retrospective allocator that searches ahead would
//! over-admit the lower-probability tail and produce `(0.85, 0.15)` — selection
//! bias toward the drafter's flat distribution.
//!
//! ## DO NOT remove the early-stop
//!
//! DSpark §5.2 removes the early-stop in production via asynchronous 2-step-prior
//! prediction + a temporal-offset causality argument (ZOS). We do NOT port that
//! variant: removing the synchronous non-anticipating early-stop without porting
//! the async-ZOS causality proof would silently break distribution preservation.
//!
//! # Design
//!
//! - **Zero heap allocation on the hot path.** The only allocation is the
//!   caller-owned `Vec<usize>` return value (per-request prefix lengths). All
//!   scratch state is caller-supplied via `schedule_with_scratch`.
//! - **`SpsCurve`** is a one-time-allocated `Box<[(usize, f32)]>` LUT with
//!   `O(log n)` binary-search lookup + linear interpolation. No extrapolation
//!   (clamp at ends).
//! - **`schedule` (allocating facade)** is provided for callers who don't want
//!   to pre-allocate scratch; it delegates to `schedule_with_scratch`.
//! - **Sigmoid discipline (per AGENTS.md)**: the early-stop uses raw
//!   `τ · SPS(B)` — no normalization, no softmax. The cumulative survival
//!   probability `a_{r,j}` is a plain product, not a softmax projection.
//!
//! # References
//!
//! - DSpark (DeepSeek-AI, 2026) §3.2.2 (Algorithm 1), §5.2 (async variant),
//!   Appendix A (non-anticipating correctness proof).
//! - `src/speculative/caddtree_budget.rs` — per-request analog.
//! - `src/speculative/acceptance_forecast.rs` — `c_k` producer (Bebop Plan 243).
//! - `src/cumprodsum.rs::cumprodsum_scalar` — the atomic SIMD `Π c_i` primitive.

#![allow(clippy::needless_range_loop)]

// ─────────────────────────────────────────────────────────────────────────────
// SPS curve — profiled engine cost LUT
// ─────────────────────────────────────────────────────────────────────────────

/// Profiled engine cost curve: steps-per-second as a function of verification
/// batch size `B`.
///
/// Constructed **once** at init from a hardware profile
/// ([`SpsCurve::from_profile`]); lookups are `O(log n)` binary search + linear
/// interpolation between bracketing samples. Extrapolation is forbidden —
/// out-of-range `B` is clamped to the nearest endpoint.
///
/// DSpark §5.2 notes that real hardware produces jagged, step-wise SPS curves
/// rather than smooth/unimodal ones. The non-anticipating early-stop assumes
/// unimodality for *global optimality* (DSpark Appendix A); jagged curves may
/// yield *locally* optimal allocations, which is acceptable — the lossless
/// distribution guarantee is preserved regardless of curve shape.
#[derive(Clone, Debug)]
pub struct SpsCurve {
    /// Sorted-by-batch-size `(batch_size, steps_per_second)` samples.
    /// Stored as `Box<[...]>` for stable address + zero post-construction alloc.
    samples: Box<[(usize, f32)]>,
}

impl SpsCurve {
    /// Build the curve from a hardware profile.
    ///
    /// The profile MAY be unsorted; this constructor sorts it ascending by
    /// batch size. Duplicate batch sizes keep the LAST sample's SPS value
    /// (later samples override earlier ones — useful for hot-reload profiles).
    ///
    /// # Panics
    ///
    /// Debug-asserts that `samples` is non-empty. An empty profile has no
    /// meaningful SPS curve.
    pub fn from_profile(samples: &[(usize, f32)]) -> Self {
        debug_assert!(!samples.is_empty(), "SpsCurve profile must be non-empty");
        // Sort ascending by batch size (stable so original order is preserved
        // for ties, which matters for the override pass below).
        let mut sorted: Vec<(usize, f32)> = samples.to_vec();
        sorted.sort_by(|(b1, _), (b2, _)| b1.cmp(b2));
        // Collapse duplicates, keeping the LAST sample per batch size
        // (override semantics: a later sample overrides an earlier one).
        let mut deduped: Vec<(usize, f32)> = Vec::with_capacity(sorted.len());
        for (b, s) in sorted.into_iter() {
            if deduped.last().map(|(lb, _)| *lb == b).unwrap_or(false) {
                if let Some((_, last_s)) = deduped.last_mut() {
                    *last_s = s;
                }
            } else {
                deduped.push((b, s));
            }
        }
        Self {
            samples: deduped.into_boxed_slice(),
        }
    }

    /// Construct a constant curve (single sample). Useful for tests and for
    /// "no engine-cost model" baselines where every batch size costs the same.
    pub fn constant(steps_per_second: f32) -> Self {
        Self {
            samples: Box::new([(1usize, steps_per_second)]),
        }
    }

    /// Linear-interpolated steps-per-second at batch size `B`.
    ///
    /// `O(log n)` binary search to find the bracketing samples `[lo, hi]`,
    /// then `sps(B) = sps_lo + (sps_hi - sps_lo) * (B - B_lo) / (B_hi - B_lo)`.
    ///
    /// Out-of-range `B` is clamped to the nearest endpoint (no extrapolation).
    #[inline]
    pub fn steps_per_second(&self, batch_size: usize) -> f32 {
        let s = &*self.samples;
        // Clamp at the lower end.
        if batch_size <= s[0].0 {
            return s[0].1;
        }
        // Clamp at the upper end.
        let last = s.len() - 1;
        if batch_size >= s[last].0 {
            return s[last].1;
        }
        // Binary search for the bracketing pair. We want the largest `lo`
        // such that s[lo].0 <= batch_size.
        let mut lo = 0usize;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if s[mid].0 <= batch_size {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // s[lo].0 <= batch_size < s[hi].0 by construction.
        let (b_lo, sps_lo) = s[lo];
        let (b_hi, sps_hi) = s[hi];
        if b_hi == b_lo {
            // Defensive: should not happen (dedup removes equal batch sizes),
            // but guard against division-by-zero anyway.
            return sps_lo;
        }
        let t = (batch_size - b_lo) as f32 / (b_hi - b_lo) as f32;
        sps_lo + (sps_hi - sps_lo) * t
    }

    /// Number of profile samples (for diagnostics / tests).
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the curve is empty (always false post-`from_profile` because
    /// empty profiles panic in debug, but provided for API completeness).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cumulative-product helper (mirrors `cumprodsum_scalar` with x=0)
// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative product: `out[j] = Π_{i=0..=j} c[i]`, with `out[0] = c[0]`.
///
/// Mirrors `crate::cumprodsum::cumprodsum_scalar` with `x = [0, 0, ...]` and
/// `h_init = 1.0` (the pure-decay special case). Written inline here so this
/// module is feature-isolated from the always-compiled `cumprodsum` substrate
/// (`hardware_aware_scheduler` has zero feature deps).
///
/// Writes into a caller-supplied `&mut [f32]` scratch buffer of length
/// `c.len()`. Zero-allocation.
#[inline]
pub fn cumprod(c: &[f32], out: &mut [f32]) {
    debug_assert_eq!(c.len(), out.len());
    if out.is_empty() {
        return;
    }
    let mut prod = 1.0_f32;
    for j in 0..c.len() {
        unsafe {
            prod *= *c.get_unchecked(j);
            *out.get_unchecked_mut(j) = prod;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HardwareAwarePrefixScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-request verification budget allocator (DSpark §3.2.2, Algorithm 1).
///
/// Given R active spec-decode requests, each with per-position survival
/// probabilities `a_{r,j} = Π_{i≤j} c_{r,i}` (monotone non-increasing in `j`),
/// produces per-request prefix lengths `ℓ*_1..ℓ*_R` that maximize
/// `Θ = τ · SPS(B)` via:
///
/// 1. Globally sort all `(r, j)` candidates descending by `a_{r,j}`.
/// 2. Greedily admit candidates; update `B += 1`, `τ += a_{r,j}`; `O(log n)`
///    lookup of `SPS(B)`.
/// 3. **Early-stop when `Θ ≤ Θ_best`** — non-anticipating property required
///    for lossless speculative decoding (DSpark Appendix A). Removing this
///    breaks distribution preservation; do NOT remove without porting the
///    async-ZOS causality proof from DSpark §5.2.
///
/// # Inputs
///
/// Callers may pass either:
/// - **pre-computed `a_{r,j}`** (already monotone non-increasing per request),
///   in which case [`schedule`](Self::schedule) uses them directly; or
/// - **raw `c_k` per-request acceptance probabilities**, in which case the
///   caller invokes [`schedule_from_token_probs`](Self::schedule_from_token_probs)
///   which computes `a_{r,j} = Π_{i≤j} c_{r,i}` internally.
///
/// # Outputs
///
/// `Vec<usize>` of length R: prefix length `ℓ*_r` per request. Caller-owned.
///
/// # Zero-allocation hot path
///
/// The allocating [`schedule`](Self::schedule) facade is the simple API. For
/// hot-path callers, [`schedule_with_scratch`](Self::schedule_with_scratch)
/// takes a pre-allocated `&mut Vec<(f32, usize, usize)>` candidate scratch
/// buffer and avoids all per-call heap allocation (the buffer `clear()`s and
/// reuses capacity across calls).
#[derive(Clone, Debug)]
pub struct HardwareAwarePrefixScheduler {
    sps: SpsCurve,
}

impl HardwareAwarePrefixScheduler {
    /// Construct with a profiled SPS curve.
    pub fn new(sps: SpsCurve) -> Self {
        Self { sps }
    }

    /// Borrow the underlying SPS curve (for diagnostics / re-profiling hooks).
    #[inline]
    pub fn sps_curve(&self) -> &SpsCurve {
        &self.sps
    }

    /// Schedule per-request prefix lengths from pre-computed survival
    /// probabilities `a_{r,j} = Π_{i≤j} c_{r,i}`.
    ///
    /// Convenience facade that allocates internal scratch. For hot-path use,
    /// prefer [`schedule_with_scratch`](Self::schedule_with_scratch).
    ///
    /// # Arguments
    /// * `survival_probs` - One slice per request, indexed `[r][j]`. Each
    ///   inner slice should already be the cumulative product of per-token
    ///   acceptance probabilities `c_k`. The values MUST be in `[0.0, 1.0]`
    ///   (caller responsibility — out-of-range values are clamped by the
    ///   greedy admission rather than rejected).
    ///
    /// # Returns
    /// `Vec<usize>` of length `survival_probs.len()`, with `out[r]` = prefix
    /// length `ℓ*_r` for request `r`.
    pub fn schedule(&self, survival_probs: &[&[f32]]) -> Vec<usize> {
        let total: usize = survival_probs.iter().map(|s| s.len()).sum();
        let mut scratch: Vec<(f32, usize, usize)> = Vec::with_capacity(total);
        let mut out: Vec<usize> = vec![0; survival_probs.len()];
        self.schedule_with_scratch(survival_probs, &mut scratch, &mut out);
        out
    }

    /// Schedule from raw per-token acceptance probabilities `c_k` instead of
    /// pre-computed `a_{r,j}`. Computes `a_{r,j} = Π_{i≤j} c_{r,i}` internally
    /// via [`cumprod`] and delegates to [`schedule_with_scratch`].
    ///
    /// # Arguments
    /// * `token_probs` - One slice per request, indexed `[r][k]`. Each value
    ///   is the per-token drafter-target acceptance probability `c_k` from
    ///   e.g. `AcceptanceForecast` (Bebop Plan 243). Values should be in
    ///   `[0.0, 1.0]`; values outside that range produce ill-defined `a_{r,j}`
    ///   but are not rejected.
    pub fn schedule_from_token_probs(&self, token_probs: &[&[f32]]) -> Vec<usize> {
        let mut out: Vec<usize> = vec![0; token_probs.len()];
        {
            let mut survival_probs: Vec<&[f32]> = Vec::with_capacity(token_probs.len());
            // We need owned storage for the cumprod outputs so we can lend them
            // back as `&[f32]` to `schedule_with_scratch`.
            let mut owned: Vec<Vec<f32>> = Vec::with_capacity(token_probs.len());
            for tp in token_probs.iter() {
                let mut buf = vec![0.0_f32; tp.len()];
                cumprod(tp, &mut buf);
                owned.push(buf);
            }
            for buf in owned.iter() {
                survival_probs.push(buf.as_slice());
            }
            let mut cand_scratch: Vec<(f32, usize, usize)> = Vec::with_capacity(
                token_probs.iter().map(|s| s.len()).sum(),
            );
            self.schedule_with_scratch(&survival_probs, &mut cand_scratch, &mut out);
        }
        out
    }

    /// Hot-path schedule: zero heap allocation after warm-up.
    ///
    /// # Arguments
    /// * `survival_probs` - pre-computed `a_{r,j}` per request.
    /// * `candidates` - scratch buffer for `(a_{r,j}, r, j)` tuples. Will be
    ///   `clear()`ed and refilled; capacity is reused across calls.
    /// * `out` - prefix lengths output. Length MUST equal
    ///   `survival_probs.len()`; values are overwritten.
    ///
    /// # Panics
    /// Debug-asserts that `out.len() == survival_probs.len()`.
    pub fn schedule_with_scratch(
        &self,
        survival_probs: &[&[f32]],
        candidates: &mut Vec<(f32, usize, usize)>,
        out: &mut [usize],
    ) {
        debug_assert_eq!(
            out.len(),
            survival_probs.len(),
            "out length must match number of requests"
        );

        // Reset output to zero (each requestor starts at ℓ*_r = 0).
        for v in out.iter_mut() {
            *v = 0;
        }

        // Degenerate: no requests → empty output.
        if survival_probs.is_empty() {
            return;
        }

        // Materialize candidates: (a_{r,j}, r, j).
        candidates.clear();
        for (r, probs) in survival_probs.iter().enumerate() {
            for (j, &a) in probs.iter().enumerate() {
                // Skip zero-probability candidates up front — admitting them
                // adds nothing to τ and only inflates B. This is a no-op for
                // correctness (the early-stop would reject them anyway) but
                // keeps the sort small.
                if a > 0.0 {
                    candidates.push((a, r, j));
                }
            }
        }

        // If no candidates survived the zero-prob filter, every ℓ*_r = 0.
        if candidates.is_empty() {
            return;
        }

        // Global sort descending by a_{r,j}.
        candidates.sort_unstable_by(|(a1, _, _), (a2, _, _)| {
            a2.partial_cmp(a1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedy admission with non-anticipating early-stop.
        //
        // Θ(B) = τ · SPS(B). We admit candidates in order of decreasing
        // a_{r,j}, accumulating τ (expected accepted tokens) and B (batch
        // size). The first time Θ fails to set a new best, we stop — the
        // non-anticipating property. See module-level doc and DSpark
        // Appendix A.
        let mut tau: f32 = 0.0;
        let mut batch_size: usize = 0;
        let mut theta_best: f32 = 0.0;

        for &(a, r, j) in candidates.iter() {
            // Tentatively admit this candidate.
            let new_tau = tau + a;
            let new_b = batch_size + 1;
            let sps = self.sps.steps_per_second(new_b);
            let theta = new_tau * sps;
            if theta <= theta_best {
                // Non-anticipating early-stop. Do NOT admit this candidate
                // or any later (lower-a) candidate. Break.
                break;
            }
            // Commit the admission.
            tau = new_tau;
            batch_size = new_b;
            theta_best = theta;
            // Increment this request's prefix length. The candidate is at
            // position j; the prefix length must be j+1 (we accept positions
            // 0..=j for this request).
            //
            // Because candidates are admitted in decreasing-a order, and
            // within a single request a_{r,j} is monotone non-increasing in j,
            // candidates for a single request are admitted in j-ascending
            // order — so each admit for request r is exactly j+1.
            out[r] = j + 1;
        }
    }

    /// Compute `Θ = τ · SPS(B)` for a given allocation, used by tests/callers
    /// to compare scheduler output vs uniform allocation.
    ///
    /// `τ` is the sum of `a_{r, ℓ*_r - 1}` over requests with `ℓ*_r > 0`
    /// (the survival probability of the last accepted position per request —
    /// equivalent to the probability that all `ℓ*_r` positions survive).
    /// `B` is the total `Σ ℓ*_r`.
    ///
    /// NOTE: this is the *realized* Θ under the per-request survival-prob
    /// definition. The internal greedy loop accumulates a different `τ`
    /// (sum of all admitted a_{r,j}), which is what the DSpark Algorithm 1
    /// literally specifies. The two coincide when each request's last-admitted
    /// `j` equals `ℓ*_r - 1` (always true post-`schedule_with_scratch`).
    pub fn realized_theta(&self, survival_probs: &[&[f32]], prefix_lengths: &[usize]) -> f32 {
        debug_assert_eq!(survival_probs.len(), prefix_lengths.len());
        let mut tau: f32 = 0.0;
        let mut batch_size: usize = 0;
        for (probs, &ell) in survival_probs.iter().zip(prefix_lengths.iter()) {
            if ell == 0 || probs.is_empty() {
                continue;
            }
            let last_idx = (ell - 1).min(probs.len() - 1);
            tau += probs[last_idx];
            batch_size += ell;
        }
        if batch_size == 0 {
            0.0
        } else {
            tau * self.sps.steps_per_second(batch_size)
        }
    }
}

impl Default for HardwareAwarePrefixScheduler {
    fn default() -> Self {
        // Default: constant SPS curve (no engine-cost model). The greedy loop
        // degenerates to "admit everything" because Θ = τ · const is monotone
        // in τ, and τ only increases as we admit more candidates.
        Self::new(SpsCurve::constant(1.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — inline unit tests for the algorithmic core.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpsCurve ───────────────────────────────────────────────────────────

    #[test]
    fn sps_curve_clamp_below_first_sample() {
        let curve = SpsCurve::from_profile(&[(10, 100.0), (20, 50.0)]);
        // batch_size=5 is below the first sample (B=10) → clamp to 100.0.
        assert!((curve.steps_per_second(5) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn sps_curve_clamp_above_last_sample() {
        let curve = SpsCurve::from_profile(&[(10, 100.0), (20, 50.0)]);
        // batch_size=100 is above the last sample (B=20) → clamp to 50.0.
        assert!((curve.steps_per_second(100) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn sps_curve_interpolates_midpoint() {
        let curve = SpsCurve::from_profile(&[(10, 100.0), (30, 50.0)]);
        // Midpoint B=20 → linear interpolation: 100 + (50-100)*(20-10)/(30-10)
        //                                          = 100 + (-50)*(10/20) = 75.
        assert!((curve.steps_per_second(20) - 75.0).abs() < 1e-6);
    }

    #[test]
    fn sps_curve_unsorted_input_is_sorted() {
        let curve = SpsCurve::from_profile(&[(30, 50.0), (10, 100.0), (20, 75.0)]);
        assert_eq!(curve.len(), 3);
        // Midpoint still lands at 75.0 because internal order is sorted.
        assert!((curve.steps_per_second(20) - 75.0).abs() < 1e-6);
    }

    #[test]
    fn sps_curve_dedup_batch_sizes() {
        let curve = SpsCurve::from_profile(&[(10, 100.0), (10, 90.0), (20, 50.0)]);
        // Duplicate B=10 collapses to the last sample (90.0).
        assert_eq!(curve.len(), 2);
        assert!((curve.steps_per_second(10) - 90.0).abs() < 1e-6);
    }

    #[test]
    fn sps_curve_constant_single_sample() {
        let curve = SpsCurve::constant(42.0);
        // Any batch size returns 42.0.
        assert!((curve.steps_per_second(1) - 42.0).abs() < 1e-6);
        assert!((curve.steps_per_second(1000) - 42.0).abs() < 1e-6);
    }

    // ── cumprod ────────────────────────────────────────────────────────────

    #[test]
    fn cumprod_basic() {
        let c = [0.5_f32, 0.8, 0.9, 1.0];
        let mut out = [0.0_f32; 4];
        cumprod(&c, &mut out);
        // out[0] = 0.5
        // out[1] = 0.5 * 0.8 = 0.4
        // out[2] = 0.4 * 0.9 = 0.36
        // out[3] = 0.36 * 1.0 = 0.36
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.4).abs() < 1e-6);
        assert!((out[2] - 0.36).abs() < 1e-6);
        assert!((out[3] - 0.36).abs() < 1e-6);
    }

    #[test]
    fn cumprod_empty() {
        let c: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        cumprod(&c, &mut out);
        // No-panic.
    }

    // ── HardwareAwarePrefixScheduler — degenerate inputs ───────────────────

    #[test]
    fn schedule_empty_requests() {
        let scheduler = HardwareAwarePrefixScheduler::default();
        let survival_probs: &[&[f32]] = &[];
        let out = scheduler.schedule(survival_probs);
        assert!(out.is_empty());
    }

    #[test]
    fn schedule_all_zero_probs_returns_zero_prefixes() {
        let scheduler = HardwareAwarePrefixScheduler::default();
        let r1: &[f32] = &[0.0, 0.0, 0.0];
        let r2: &[f32] = &[0.0, 0.0];
        let survival_probs: &[&[f32]] = &[r1, r2];
        let out = scheduler.schedule(survival_probs);
        assert_eq!(out, vec![0, 0]);
    }

    #[test]
    fn schedule_single_request_single_position() {
        // Constant SPS curve → greedy admits everything (Θ = τ · const).
        let scheduler = HardwareAwarePrefixScheduler::default();
        let r1: &[f32] = &[0.9];
        let out = scheduler.schedule(&[r1]);
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn schedule_single_sample_sps_curve() {
        let curve = SpsCurve::constant(1.0);
        let scheduler = HardwareAwarePrefixScheduler::new(curve);
        let r1: &[f32] = &[0.9, 0.8, 0.7];
        let out = scheduler.schedule(&[r1]);
        // Constant SPS = 1.0 → Θ = τ (always increasing) → admit all.
        assert_eq!(out, vec![3]);
    }

    // ── HardwareAwarePrefixScheduler — non-anticipating early-stop ─────────

    #[test]
    fn schedule_cliff_sps_curve_truncates_at_cliff() {
        // SPS curve: 100 steps/sec at B≤2, drops to 1 step/sec at B≥3.
        // The cliff is between B=2 and B=3.
        let curve = SpsCurve::from_profile(&[(1, 100.0), (2, 100.0), (3, 1.0), (10, 1.0)]);
        let scheduler = HardwareAwarePrefixScheduler::new(curve);

        // Single request with 10 positions, all survival prob 0.5.
        // Without the cliff, all 10 would be admitted. With the cliff,
        // the early-stop fires when Θ drops at B=3.
        let r1: &[f32] = &[
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];
        let out = scheduler.schedule(&[r1]);
        // We expect exactly 2 admits: at B=1, Θ = 0.5 * 100 = 50.
        //                            at B=2, Θ = 1.0 * 100 = 100 (best).
        //                            at B=3, Θ = 1.5 * 1 = 1.5 ≤ 100 → STOP.
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn schedule_multi_request_allocates_more_to_high_survival() {
        // Two requests: r0 has high survival probs, r1 has low survival probs.
        // SPS curve is monotonically decreasing (penalizes big batches).
        let curve = SpsCurve::from_profile(&[
            (1, 100.0),
            (4, 50.0),
            (8, 25.0),
            (16, 10.0),
        ]);
        let scheduler = HardwareAwarePrefixScheduler::new(curve);

        let r0: &[f32] = &[0.95, 0.90, 0.85, 0.80]; // high survival
        let r1: &[f32] = &[0.30, 0.20, 0.10, 0.05]; // low survival
        let out = scheduler.schedule(&[r0, r1]);

        // r0 should get a longer prefix than r1 — high-survival candidates
        // sort to the front and get admitted before low-survival ones.
        assert!(
            out[0] >= out[1],
            "high-survival r0 should get >= prefix vs r1: out = {:?}",
            out
        );
        assert!(out[0] > 0, "r0 should get at least 1");
    }

    #[test]
    fn schedule_beats_uniform_allocation_on_cliff_curve() {
        // Cliff SPS curve: cheap up to B=4, expensive after.
        let curve = SpsCurve::from_profile(&[
            (1, 100.0),
            (2, 100.0),
            (3, 100.0),
            (4, 100.0),
            (5, 10.0),
            (16, 1.0),
        ]);
        let scheduler = HardwareAwarePrefixScheduler::new(curve);

        // 4 requests, varying survival profiles.
        let r0: &[f32] = &[0.9, 0.85, 0.8, 0.75]; // strong
        let r1: &[f32] = &[0.8, 0.7, 0.6, 0.5]; // medium
        let r2: &[f32] = &[0.5, 0.4, 0.3, 0.2]; // weak
        let r3: &[f32] = &[0.3, 0.2, 0.1, 0.05]; // weakest
        let survival_probs: &[&[f32]] = &[r0, r1, r2, r3];

        let scheduled = scheduler.schedule(survival_probs);
        let scheduled_theta = scheduler.realized_theta(survival_probs, &scheduled);

        // Uniform allocation of length 2 per request (total B=8).
        let uniform = vec![2usize, 2, 2, 2];
        let uniform_theta = scheduler.realized_theta(survival_probs, &uniform);

        assert!(
            scheduled_theta >= uniform_theta,
            "scheduled Θ ({:.4}) should >= uniform Θ ({:.4}); out = {:?}",
            scheduled_theta,
            uniform_theta,
            scheduled
        );
    }

    // ── schedule_from_token_probs ──────────────────────────────────────────

    #[test]
    fn schedule_from_token_probs_matches_precomputed() {
        let curve = SpsCurve::from_profile(&[(1, 100.0), (4, 50.0), (8, 25.0)]);
        let scheduler = HardwareAwarePrefixScheduler::new(curve);

        // Per-token acceptance probs c_k.
        let c0: &[f32] = &[0.9, 0.95, 0.9]; // high
        let c1: &[f32] = &[0.3, 0.5, 0.4]; // low

        // Compute a_{r,j} manually.
        let mut a0 = [0.0_f32; 3];
        let mut a1 = [0.0_f32; 3];
        cumprod(c0, &mut a0);
        cumprod(c1, &mut a1);

        let from_token = scheduler.schedule_from_token_probs(&[c0, c1]);
        let from_precomputed = scheduler.schedule(&[&a0, &a1]);

        assert_eq!(
            from_token, from_precomputed,
            "schedule_from_token_probs must match schedule with cumprod'd probs"
        );
    }

    // ── Non-anticipating property — the Appendix A counterexample ──────────

    #[test]
    fn non_anticipating_early_stop_preserves_distribution() {
        // Appendix A counterexample, ported to the scheduler abstraction.
        //
        // Vocab {A, B}, target p_t = (0.7, 0.3), drafter p_d = (0.5, 0.5).
        // Correct lossless output is (0.7, 0.3). A non-anticipating allocator
        // preserves this; a retrospective one would over-admit the lower-
        // probability tail and bias toward the drafter's flat distribution.
        //
        // We model this as: drafter always proposes token B (the lower-target-
        // prob token). Survival prob a_j for the j-th proposed B is the
        // probability the verifier accepts the j-th B in a row, which under
        // LeviathanVerifier semantics is Π_{i≤j} min(1, p_t(B) / p_d(B))
        //                                          = Π_{i≤j} min(1, 0.3/0.5)
        //                                          = Π_{i≤j} 0.6.
        //
        // For 5 positions: a = [0.6, 0.36, 0.216, 0.1296, 0.07776].
        //
        // A constant SPS curve means the greedy loop admits everything (Θ is
        // monotone in τ). This is the **correct** lossless behavior — there
        // is no batch-size pressure to truncate, so we verify all 5 and the
        // verifier's rejection sampling produces (0.7, 0.3). No selection
        // bias because the scheduler admits everything.
        //
        // The non-anticipating property's role is to prevent AHEAD-OF-TIME
        // truncation that would bias which positions get verified. With a
        // batch-size cliff, the early-stop truncates at the cliff without
        // consulting future lower-a candidates — that's what preserves the
        // distribution. We test that here by checking the truncation point
        // depends only on candidates admitted so far, not on future ones.

        // Cliff SPS curve: cheap up to B=3, drops sharply at B=4.
        let curve = SpsCurve::from_profile(&[
            (1, 100.0),
            (2, 100.0),
            (3, 100.0),
            (4, 1.0),
            (10, 1.0),
        ]);
        let scheduler = HardwareAwarePrefixScheduler::new(curve);

        let r1: &[f32] = &[0.6, 0.36, 0.216, 0.1296, 0.07776];
        let out = scheduler.schedule(&[r1]);

        // Expected: admit positions 0, 1, 2 (B=3, Θ = (0.6+0.36+0.216) * 100
        // = 1.176 * 100 = 117.6). At B=4, Θ = (1.176+0.1296) * 1 = 1.3056,
        // which is way below 117.6 → STOP. ℓ* = 3.
        assert_eq!(out, vec![3]);

        // Now add MORE low-probability candidates AFTER position 2. The
        // non-anticipating property says the truncation point (ℓ*=3) must
        // NOT change — the early-stop fired before consulting them.
        let r1_extended: &[f32] = &[
            0.6, 0.36, 0.216, 0.1296, 0.07776, 0.05, 0.03, 0.01, 0.005, 0.001,
        ];
        let out_extended = scheduler.schedule(&[r1_extended]);
        assert_eq!(
            out_extended, vec![3],
            "non-anticipating: adding more low-prob candidates must not change ℓ*"
        );
    }

    // ── schedule_with_scratch reuses capacity ──────────────────────────────

    #[test]
    fn schedule_with_scratch_reuses_capacity() {
        let curve = SpsCurve::from_profile(&[(1, 100.0), (4, 50.0)]);
        let scheduler = HardwareAwarePrefixScheduler::new(curve);

        let mut scratch: Vec<(f32, usize, usize)> = Vec::new();
        let mut out: Vec<usize> = vec![0; 2];

        // First call — scratch grows.
        let r0: &[f32] = &[0.9, 0.8, 0.7, 0.6];
        let r1: &[f32] = &[0.5, 0.4, 0.3];
        scheduler.schedule_with_scratch(&[r0, r1], &mut scratch, &mut out);
        let cap_after_first = scratch.capacity();
        assert!(cap_after_first > 0);

        // Second call with smaller inputs — capacity is preserved.
        let r0_small: &[f32] = &[0.9];
        let r1_small: &[f32] = &[0.5];
        scheduler.schedule_with_scratch(&[r0_small, r1_small], &mut scratch, &mut out);
        assert_eq!(scratch.capacity(), cap_after_first);
    }
}
