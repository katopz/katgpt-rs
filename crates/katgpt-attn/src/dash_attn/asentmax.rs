//! ASEntmax — length-adaptive damping schedule for α-entmax routing
//! (Issue 747 P0, Research 549; Vasylenko et al., *Long-Context
//! Generalization with Sparse Attention*, arXiv:2506.16640, ICLR 2026).
//!
//! α-entmax over-sparsifies as the scored candidate set grows: the logit
//! range of n IID scores scales as `E[Δ_n] = 2σ√(2 log n)` (Kamath 2015),
//! while the probability-mass budget the threshold competes with is the
//! fixed `1`. The paper's Eq 10 *derives* the counter-schedule: pre-scaling
//! scores by `(δ + β·(log n)^γ)` with **γ = −0.5, δ = 0** makes the scaled
//! range `β·(log n)^{−0.5}·Δ_n = 2σβ√2 = const` — the support size stops
//! being eaten by candidate-set growth. With `β = 1/(2σ̂√2)` (σ̂ a rolling
//! estimate of the score std) the scaled range is **exactly 1**, σ-free.
//!
//! # The duality (why this mirrors `katgpt_core::ssmax`)
//!
//! One extreme-value law, two signed arms, same socket shape:
//!
//! | Arm | Failure without correction | Correction |
//! |---|---|---|
//! | softmax (SSMax, Plan 411) | dilution — mass leaks to every token | sharpen UP ∝ `log n` (`s_L·log N`) |
//! | α-entmax (this module) | over-sparsification — threshold eats the support | damp DOWN ∝ `(log n)^{−0.5}` |
//!
//! SSMax ships γ=+1 sharpening for the softmax arm; this module ships the
//! γ=−0.5 damping arm for the entmax path (`entmax_1p5` in
//! [`super::entmax`], DashAttention routing, Plan 106 — which has no length
//! term at all today).
//!
//! # What scaling actually does (and why it is a real intervention)
//!
//! Multiplying every score by a constant `c` does NOT rescale entmax
//! outputs trivially: the support condition is `Σ_{j≤k}(z_j − z_(k)) < 1`
//! (excess-mass budget), so `c·z` gives `c·Σ(...)  < 1` — a sub-unit `c`
//! **enlarges the support**. Damping is exactly the knob that counteracts
//! over-sparsification; the `(log n)^{−0.5}` law makes it self-calibrating
//! in the candidate count.
//!
//! # Honest scope
//!
//! - γ=−0.5 is IID-Gaussian-optimal only (paper's own fitted per-head γ
//!   varies in sign). The derived default must beat the unscheduled
//!   baseline on OUR routing surfaces (Issue 747 G2); the [`AsentmaxSchedule::Generalized`]
//!   arm is the offline-sweep surface for per-head (δ, β, γ).
//! - No quality-parity claim is made for swapping trained softmax
//!   attention on pretrained weights — this is a routing-score rescaler,
//!   modelless by construction (runtime statistic, zero training).
//!
//! # Allocation discipline (G4)
//!
//! [`apply_asentmax_inplace`] is allocation-free by construction (single
//! in-place multiply pass, one `powf` + one mul per row — the multiplier is
//! computed once per head per step, not per score). The estimator's
//! `observe_row` is a single-pass max/min/scan with no allocation.
//!
//! References:
//! - Issue 747 — execution tracker (P0)
//! - Research 549 — distillation, novelty gate, fusion analysis
//! - arXiv:2506.16640 — the paper (Eq 10, Lemma 2, Prop 6)

use std::sync::atomic::{AtomicU64, Ordering};

/// √2, the constant in `β = 1/(2σ̂√2)` (clippy-clean sourced constant).
use std::f32::consts::SQRT_2;

/// ln floor for the `(log n)^γ` term: for n ≤ e the damping factor
/// `(log n)^{−0.5}` would exceed 1 (amplify). The extreme-value law is
/// asymptotic in n, so the factor is capped at 1 — the schedule is a
/// no-op-beyond-β for n ≤ e and monotonically damping thereafter.
const LOG_N_FLOOR: f32 = 1.0;

// ──────────────────────────────────────────────────────────────────────────
// Schedule
// ──────────────────────────────────────────────────────────────────────────

/// ASEntmax score-rescaling schedule: the entmax-side mirror of
/// `katgpt_core::ssmax::SsmaxMode`.
///
/// The multiplicative factor applied to pre-entmax scores is
/// `δ + β·(log n_c)^γ` where `n_c` is the scored candidate (chunk) count —
/// see [`AsentmaxSchedule::multiplier`].
///
/// # Modelless discipline
///
/// All variants are modelless:
/// - [`None`](AsentmaxSchedule::None) — the shipped Plan 106 behavior
///   (no length term). Identity multiplier.
/// - [`Derived`](AsentmaxSchedule::Derived) — the Eq 10 closed form with
///   `γ = −0.5, δ = 0` and `β = 1/(2σ̂√2)` resolved from a caller-managed
///   rolling σ̂. Zero training; one runtime statistic.
/// - [`Generalized`](AsentmaxSchedule::Generalized) — the full
///   per-head `(δ + β·(log n)^γ)` surface for offline sweeps (Issue 747
///   P4 T4.3; frozen-table freeze/thaw constants).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AsentmaxSchedule {
    /// No schedule — shipped Plan 106 routing behavior (identity scale).
    None,
    /// Derived damping (Eq 10 closed form): `γ = −0.5`, `δ = 0`,
    /// `β = 1/(2σ̂√2)`. The scaled logit range is pinned to 1 regardless
    /// of candidate count n_c or score scale σ̂.
    ///
    /// `sigma_hat` is a caller-managed rolling estimate of the score
    /// std — see [`RollingSigmaEstimator`] for the built-in, or construct
    /// directly. Should be positive; floored at `1e-3` on use.
    Derived {
        /// Rolling estimate of the pre-entmax score std σ̂.
        sigma_hat: f32,
    },
    /// Generalized per-head schedule `(δ + β·(log n_c)^γ)` — the offline
    /// sweep surface. `gamma` follows the paper's convention (γ = −0.5 is
    /// the derived damping; γ > 0 sharpens — the SSMax-like direction).
    Generalized {
        /// Additive term δ (paper default 0).
        delta: f32,
        /// Multiplicative coefficient β.
        beta: f32,
        /// Log-power γ (derived default −0.5).
        gamma: f32,
    },
}

impl Default for AsentmaxSchedule {
    /// Default: [`None`](AsentmaxSchedule::None) — the shipped Plan 106
    /// behavior. A default-on schedule requires the Issue 747 GOAT gate.
    fn default() -> Self {
        Self::None
    }
}

impl AsentmaxSchedule {
    /// The multiplicative factor applied to pre-entmax scores:
    /// `δ + β·(log n_c)^γ`.
    ///
    /// - [`None`](AsentmaxSchedule::None) → `1.0` (identity — zero
    ///   overhead, the off state).
    /// - [`Derived`](AsentmaxSchedule::Derived) →
    ///   `(2σ̂√2)^{−1} · (log n)^{−0.5}` with `log n` floored at 1
    ///   (no amplification for n ≤ e; monotonically damping above).
    /// - [`Generalized`](AsentmaxSchedule::Generalized) →
    ///   `δ + β·(log n)^γ` with the same floor applied when γ < 0.
    ///
    /// # Fallback contract
    ///
    /// If the resolved multiplier is non-finite or non-positive (ill-formed
    /// Generalized coefficients), this returns `1.0` — the identity, i.e.
    /// the schedule-off state. A debug assertion fires in debug builds so
    /// ill-formed coefficients surface in tests while the release hot path
    /// stays total (a negative scale would silently flip score order —
    /// the identity is the least-corrupting degradation).
    #[inline]
    pub fn multiplier(&self, log_n: f32) -> f32 {
        let mult = match self {
            Self::None => 1.0,
            Self::Derived { sigma_hat } => {
                let beta = 1.0 / (2.0 * sigma_hat.max(1e-3) * SQRT_2);
                beta * log_n.max(LOG_N_FLOOR).powf(-0.5)
            }
            Self::Generalized { delta, beta, gamma } => {
                let ln = if *gamma < 0.0 {
                    log_n.max(LOG_N_FLOOR)
                } else {
                    log_n
                };
                delta + beta * ln.powf(*gamma)
            }
        };
        debug_assert!(
            mult.is_finite() && mult > 0.0,
            "asentmax multiplier must be finite-positive, got {mult} for {self:?} at log_n={log_n}"
        );
        if mult.is_finite() && mult > 0.0 {
            mult
        } else {
            1.0
        }
    }

    /// Whether this schedule scales scores at all (fast-path check for
    /// callers that want to skip the multiply pass entirely).
    #[inline]
    pub fn is_identity(&self, log_n: f32) -> bool {
        self.multiplier(log_n) == 1.0
    }
}

/// Rescale pre-entmax routing scores in place by the schedule multiplier
/// `δ + β·(log n_c)^γ`.
///
/// This is the ASEntmax intervention, applied BEFORE the entmax threshold
/// pass ([`super::entmax::entmax_1p5_into`]). The scaled excess-mass budget
/// makes the support size stationary in `n_c` (see module docs).
///
/// # Arguments
///
/// - `scores` — per-chunk routing scores for one query head, modified in
///   place.
/// - `schedule` — the schedule source.
/// - `log_n` — `ln(n_c)`, the natural log of the scored candidate count.
///   Caller-supplied (the caller knows `n_c`; no `ln` in the hot loop).
///
/// # Allocation discipline (G4)
///
/// Allocation-free by construction: one multiplier computation (one
/// `powf`) then a single chunked in-place multiply pass. No `Vec`, `Box`,
/// `String`, or collecting iterator appears. Identity schedules
/// ([`AsentmaxSchedule::None`]) return without touching the slice.
#[inline]
pub fn apply_asentmax_inplace(scores: &mut [f32], schedule: &AsentmaxSchedule, log_n: f32) {
    let mult = schedule.multiplier(log_n);
    if mult == 1.0 {
        return;
    }
    // Chunked 8-wide loop to help LLVM auto-vectorize (hot-loop rule;
    // mirrors `apply_ssmax_inplace`).
    for chunk in scores.as_chunks_mut::<8>().0 {
        for x in chunk {
            *x *= mult;
        }
    }
    for x in scores.as_chunks_mut::<8>().1 {
        *x *= mult;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Rolling σ̂ estimator (mirror of `ssmax::RollingDeltaEstimator`)
// ──────────────────────────────────────────────────────────────────────────

/// Lock-free EMA estimator for the routing-score std σ̂.
///
/// Observes `max(row) − min(row)` and converts via the Kamath range law
/// `σ̂ = range / (2√(2 ln n))` — the same extreme-value identity the
/// Eq 10 derivation rests on, so the estimate is self-consistent with the
/// quantity the schedule cancels. Maintains an exponential moving average
/// via a lock-free `AtomicU64` CAS loop (mirrors
/// `katgpt_core::ssmax::RollingDeltaEstimator`, Plan 411 S2).
///
/// Produces an [`AsentmaxSchedule::Derived`] on demand.
///
/// # Warm-start
///
/// Before any observation the EMA holds `σ̂ = 1.0`. The estimator adapts
/// away from this only when observed ranges deviate.
///
/// # Thread safety
///
/// `Send + Sync` via `AtomicU64`; the CAS loop is lock-free. For per-head
/// estimators updated once per routing step, contention is negligible.
///
/// # Allocation discipline (G4)
///
/// [`observe_row`](Self::observe_row) is O(n) single-pass, zero
/// allocation. [`resolve_sigma`](Self::resolve_sigma) and
/// [`to_schedule`](Self::to_schedule) are O(1).
///
/// # Example
///
/// ```ignore
/// use katgpt_attn::dash_attn::asentmax::{
///     RollingSigmaEstimator, apply_asentmax_inplace,
/// };
///
/// let est = RollingSigmaEstimator::default(); // α = 0.99, warm-start σ̂ = 1.0
/// let mut scores = vec![2.0_f32, -1.0, 0.5, -0.3, 1.1];
/// est.observe_row(&scores);
/// let schedule = est.to_schedule();
/// let log_n = (scores.len() as f32).ln();
/// apply_asentmax_inplace(&mut scores, &schedule, log_n);
/// ```
#[derive(Debug)]
pub struct RollingSigmaEstimator {
    /// EMA of observed range-derived σ̂, stored as `f64::to_bits` in an
    /// `AtomicU64` for lock-free updates.
    ema_bits: AtomicU64,
    /// EMA decay factor in `(0, 1)`: `new = α·old + (1−α)·observed`.
    alpha: f64,
}

impl RollingSigmaEstimator {
    /// Construct with a custom EMA decay factor (clamped to `(0, 1)`).
    /// Higher α = slower adaptation. Default `0.99` gives ~100-step
    /// effective memory.
    #[inline]
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(1e-6, 1.0 - 1e-6);
        Self {
            ema_bits: AtomicU64::new(1.0_f64.to_bits()),
            alpha,
        }
    }

    /// Observe a row of pre-entmax scores and update the EMA.
    ///
    /// Computes the row range, converts to σ̂ via the Kamath law
    /// `range / (2√(2 ln n))`, and blends it into the EMA. Rows with
    /// fewer than 2 scores are no-ops.
    ///
    /// Zero allocation: one pass for min + max.
    #[inline]
    pub fn observe_row(&self, scores: &[f32]) {
        let n = scores.len();
        if n <= 1 {
            return;
        }
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        for &x in scores {
            if x > max {
                max = x;
            }
            if x < min {
                min = x;
            }
        }
        let range = (max - min) as f64;
        let sigma = range / (2.0 * (2.0 * (n as f64).ln()).sqrt());
        self.update_ema(sigma);
    }

    /// Lock-free EMA update via CAS loop. Skips non-finite or
    /// non-positive observations.
    #[inline]
    fn update_ema(&self, observed: f64) {
        if !observed.is_finite() || observed <= 0.0 {
            return;
        }
        loop {
            let old_bits = self.ema_bits.load(Ordering::Relaxed);
            let old_ema = f64::from_bits(old_bits);
            let new_ema = self.alpha * old_ema + (1.0 - self.alpha) * observed;
            let new_bits = new_ema.to_bits();
            match self.ema_bits.compare_exchange_weak(
                old_bits,
                new_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    /// Read the current EMA estimate of σ̂ (clamped to `[1e-3, 1e3]`).
    #[inline]
    pub fn resolve_sigma(&self) -> f32 {
        let bits = self.ema_bits.load(Ordering::Relaxed);
        (f64::from_bits(bits) as f32).clamp(1e-3, 1e3)
    }

    /// Produce an [`AsentmaxSchedule::Derived`] from the current EMA.
    #[inline]
    pub fn to_schedule(&self) -> AsentmaxSchedule {
        AsentmaxSchedule::Derived {
            sigma_hat: self.resolve_sigma(),
        }
    }
}

impl Default for RollingSigmaEstimator {
    /// Default: α = 0.99 (slow adaptation, ~100-step memory),
    /// warm-start σ̂ = 1.0.
    #[inline]
    fn default() -> Self {
        Self::new(0.99)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn none_is_identity_multiplier() {
        assert_eq!(AsentmaxSchedule::None.multiplier(9.0), 1.0);
        assert!(AsentmaxSchedule::None.is_identity(9.0));
        assert_eq!(AsentmaxSchedule::default(), AsentmaxSchedule::None);
    }

    #[test]
    fn derived_multiplier_matches_closed_form() {
        // σ̂ = 1, n = 512 → log n = ln 512 ≈ 6.2384.
        // mult = 1/(2·√2) · (ln 512)^(−1/2) ≈ 0.35355 · 0.40037 ≈ 0.14156.
        let log_n = 512_f32.ln();
        let sched = AsentmaxSchedule::Derived { sigma_hat: 1.0 };
        let expected = (1.0 / (2.0 * SQRT_2)) * log_n.powf(-0.5);
        assert!((sched.multiplier(log_n) - expected).abs() < TOL);
        assert!((sched.multiplier(log_n) - 0.14156).abs() < 1e-3);
    }

    #[test]
    fn derived_multiplier_is_monotone_damping() {
        let sched = AsentmaxSchedule::Derived { sigma_hat: 2.0 };
        let mut prev = f32::INFINITY;
        for &n in &[3_usize, 8, 64, 512, 4096, 524_288] {
            let m = sched.multiplier((n as f32).ln());
            assert!(m <= prev, "multiplier must be non-increasing in n: {m} > {prev}");
            prev = m;
        }
    }

    #[test]
    fn derived_multiplier_caps_amplification_below_e() {
        // For n ≤ e (log n ≤ 1) the (log n)^{−0.5} factor would exceed 1;
        // the floor pins it at 1 → mult = β exactly.
        let sched = AsentmaxSchedule::Derived { sigma_hat: 1.0 };
        let beta = 1.0 / (2.0 * SQRT_2);
        for log_n in [0.0_f32, 0.5, 1.0] {
            assert!((sched.multiplier(log_n) - beta).abs() < TOL);
        }
        // σ̂ flooring: a zero σ̂ must not blow up.
        let tiny = AsentmaxSchedule::Derived { sigma_hat: 0.0 };
        let m = tiny.multiplier(6.0);
        assert!(m.is_finite() && m > 0.0 && m <= 1.0 / (2.0 * 1e-3 * SQRT_2));
    }

    /// σ-invariance: σ·mult(n) is constant — the σ̂ in β cancels the row
    /// scale, which is the exact mechanism the G1 stationarity gate rests on.
    #[test]
    fn derived_sigma_invariance_of_effective_scale() {
        let log_n = 13.14_f32; // n = 512k
        let mut eff = None;
        for &sigma in &[0.5_f32, 1.0, 3.0, 10.0, 100.0] {
            let sched = AsentmaxSchedule::Derived { sigma_hat: sigma };
            let e = sigma * sched.multiplier(log_n);
            match eff {
                None => eff = Some(e),
                Some(prev) => {
                    assert!(
                        (e - prev).abs() / prev < 1e-3,
                        "σ·mult must be σ-invariant: {e} vs {prev}"
                    );
                }
            }
        }
    }

    #[test]
    fn generalized_known_values() {
        // γ = 0 → δ + β (constant).
        let g = AsentmaxSchedule::Generalized {
            delta: 0.1,
            beta: 0.5,
            gamma: 0.0,
        };
        assert!((g.multiplier(7.3) - 0.6).abs() < TOL);
        // γ = 1 → δ + β·log n.
        let g1 = AsentmaxSchedule::Generalized {
            delta: 0.0,
            beta: 2.0,
            gamma: 1.0,
        };
        assert!((g1.multiplier(5.0) - 10.0).abs() < TOL);
        // γ < 0 floors log n at 1 (no amplification below e).
        let gn = AsentmaxSchedule::Generalized {
            delta: 0.0,
            beta: 1.0,
            gamma: -0.5,
        };
        assert!((gn.multiplier(0.0) - 1.0).abs() < TOL);
    }

    /// Ill-formed coefficients (non-finite / non-positive multiplier) are a
    /// debug-build panic (surface in sweep tooling) and a release-build
    /// identity fallback (hot path stays total). Profile-split so both
    /// behaviors are pinned where they actually occur.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "asentmax multiplier")]
    fn generalized_ill_formed_panics_in_debug() {
        let g = AsentmaxSchedule::Generalized {
            delta: f32::INFINITY,
            beta: 1.0,
            gamma: 1.0,
        };
        let _ = g.multiplier(5.0);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn generalized_ill_formed_falls_back_to_identity_in_release() {
        // β huge → +inf multiplier → identity fallback (release path).
        let g = AsentmaxSchedule::Generalized {
            delta: f32::INFINITY,
            beta: 1.0,
            gamma: 1.0,
        };
        assert_eq!(g.multiplier(5.0), 1.0);
        // Zero multiplier (β=0, δ=0) is non-positive → identity.
        let z = AsentmaxSchedule::Generalized {
            delta: 0.0,
            beta: 0.0,
            gamma: 1.0,
        };
        assert_eq!(z.multiplier(5.0), 1.0);
    }

    #[test]
    fn apply_scales_every_score() {
        let mut scores = vec![1.0_f32, -2.0, 3.0, 0.5, -0.25];
        let sched = AsentmaxSchedule::Derived { sigma_hat: 1.0 };
        let log_n = 512_f32.ln();
        let mult = sched.multiplier(log_n);
        apply_asentmax_inplace(&mut scores, &sched, log_n);
        for (&s, &orig) in scores.iter().zip([1.0, -2.0, 3.0, 0.5, -0.25].iter()) {
            assert!((s - orig * mult).abs() < 1e-6 * orig.abs().max(1.0));
        }
    }

    #[test]
    fn apply_none_is_bit_identical_noop() {
        let mut scores = vec![1.0_f32, -2.0, 3.5, 0.125];
        let before = scores.clone();
        apply_asentmax_inplace(&mut scores, &AsentmaxSchedule::None, 9.0);
        assert_eq!(scores, before);
    }

    #[test]
    fn apply_empty_slice_is_noop() {
        let mut scores: Vec<f32> = vec![];
        apply_asentmax_inplace(&mut scores, &AsentmaxSchedule::Derived { sigma_hat: 1.0 }, 6.0);
        assert!(scores.is_empty());
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    #[test]
    fn estimator_warm_start_is_sigma_one() {
        let est = RollingSigmaEstimator::default();
        assert!((est.resolve_sigma() - 1.0).abs() < TOL);
        assert_eq!(
            est.to_schedule(),
            AsentmaxSchedule::Derived { sigma_hat: 1.0 }
        );
    }

    #[test]
    fn estimator_ignores_short_and_nan_rows() {
        let est = RollingSigmaEstimator::new(0.5);
        est.observe_row(&[]);
        est.observe_row(&[1.0]);
        assert!((est.resolve_sigma() - 1.0).abs() < TOL, "short rows are no-ops");
        // All-NaN row: max/min stay ±inf → range is NaN → skipped.
        est.observe_row(&[f32::NAN, f32::NAN]);
        assert!(
            (est.resolve_sigma() - 1.0).abs() < TOL,
            "NaN rows must not pollute the EMA"
        );
    }

    #[test]
    fn estimator_converges_to_range_law_sigma() {
        // A row with a known range: σ̂_row = range / (2√(2 ln n)).
        // With α = 0.5 and ~40 observations the EMA is within 1e-10 of the
        // fixed point (= σ̂_row, since every observation is identical).
        let n = 1024_usize;
        let mut row = vec![0.0_f32; n];
        let mut state = 0x243F_6A88_85A3_08D3_u64;
        for slot in row.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            *slot = (u as f32) * 8.0 - 4.0; // uniform [-4, 4)
        }
        let range = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - row.iter().cloned().fold(f32::INFINITY, f32::min);
        let sigma_row = range as f64 / (2.0 * (2.0 * (n as f64).ln()).sqrt());

        let est = RollingSigmaEstimator::new(0.5);
        for _ in 0..40 {
            est.observe_row(&row);
        }
        let got = est.resolve_sigma();
        assert!(
            (got as f64 - sigma_row).abs() / sigma_row < 1e-6,
            "EMA must converge to the range-law σ̂: {got} vs {sigma_row}"
        );
    }

    #[test]
    fn estimator_clamps_extreme_sigma() {
        // Huge-range rows push σ̂ up but resolve clamps at 1e3.
        let est = RollingSigmaEstimator::new(0.5);
        let huge: Vec<f32> = (0..2048).map(|i| if i % 2 == 0 { 1e9 } else { -1e9 }).collect();
        for _ in 0..80 {
            est.observe_row(&huge);
        }
        assert!(est.resolve_sigma() <= 1e3);
    }

    #[test]
    fn estimator_multi_threaded_no_panic() {
        let est = std::sync::Arc::new(RollingSigmaEstimator::new(0.9));
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let est = std::sync::Arc::clone(&est);
                std::thread::spawn(move || {
                    for i in 0..200 {
                        let row: Vec<f32> = (0..64).map(|j| ((i * 7 + t * 13 + j) % 17) as f32 - 8.0).collect();
                        est.observe_row(&row);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(est.resolve_sigma().is_finite());
    }

    // ── Entmax interaction (the mechanism the schedule exists for) ─────────

    /// Damping enlarges the support: the excess-mass budget condition
    /// `Σ_{j≤k}(z_j − z_(k)) < 1` scales with the row, so a sub-unit
    /// multiplier admits more chunks into the support.
    #[test]
    fn damping_enlarges_entmax_support() {
        use super::super::entmax::{entmax_1p5, entmax_support};

        // Bulk scores with σ ≈ 3 (a large-spread row — the
        // over-sparsification regime), n = 512.
        let n = 512_usize;
        let mut state = 0x13198A2E03707344_u64;
        let mut row = vec![0.0_f32; n];
        for slot in row.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Box-Muller-ish cheap normal: sum of 3 uniforms, centered.
            let u1 = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
            let u2 = ((state >> 21) as f64 / (1u64 << 53) as f64) as f32;
            let u3 = ((state >> 31) as f64 / (1u64 << 53) as f64) as f32;
            *slot = (u1 + u2 + u3 - 1.5) * 6.0; // ~N(0, 3²)-ish
        }

        let (raw_probs, _) = entmax_1p5(&row);
        let raw_support = entmax_support(&raw_probs).len();

        let sched = AsentmaxSchedule::Derived { sigma_hat: 3.0 };
        let mut damped = row.clone();
        apply_asentmax_inplace(&mut damped, &sched, (n as f32).ln());
        let (damped_probs, _) = entmax_1p5(&damped);
        let damped_support = entmax_support(&damped_probs).len();

        assert!(
            damped_support > raw_support,
            "damping must enlarge the support at large σ: {damped_support} vs {raw_support}"
        );
        // Simplex + exact-zero preservation under the schedule.
        let sum: f32 = damped_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs must stay on the simplex");
        for &p in damped_probs.iter() {
            assert!(p >= 0.0);
        }
        let zeros = damped_probs.iter().filter(|&&p| p == 0.0).count();
        assert!(
            zeros > 0,
            "off-support probs must be EXACT zeros ({zeros}/{n})"
        );
    }
}
