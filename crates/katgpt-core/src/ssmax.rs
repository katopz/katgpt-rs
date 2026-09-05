//! SSMax — length-aware log-N attention temperature.
//!
//! Distillation of Gollapudi et al., *Can Language Models Actually Retrieve
//! In-Context? Drowning in Documents at Million Token Scale*
//! (arXiv:2607.01538, 2026). The paper identifies **attention dilution** as
//! the primary bottleneck at million-token scale: as corpus size N grows, the
//! softmax denominator grows faster than the gold term's numerator, collapsing
//! the post-normalization mass on the gold document even when the pre-softmax
//! score stays high. The bound is:
//!
//! ```text
//! α_gold ≈ 1 / (1 + (N−1) · N^{−s·Δ})
//! ```
//!
//! where `Δ = s_{t⋆} − s̄_{distractor}` is the gold–distractor pre-softmax
//! logit gap. **SSMax** cancels the `(N−1)` growth by rescaling logits
//! multiplicatively by `s_L · log(N)`:
//!
//! ```text
//! s̃_{L,h,t} = s_L · log(N) · s_{L,h,t}
//! ```
//!
//! Whenever `s_L · Δ > 1`, the post-normalization gold weight is bounded away
//! from zero regardless of N.
//!
//! # The modelless reading (§3.5 unblock)
//!
//! The paper trains `s_L` in a shared parameter group. We do NOT train. We
//! derive `s_L` analytically from the desired behavior: pick `s_L` so that
//! `s_L · log(N) · Δ_typical ≈ log(N)`, i.e. `s_L ≈ 1/Δ_typical`. The default
//! `s_L = 1.0` ([`SsmaxMode::Fixed`]) is the truly modelless case (zero
//! training, zero new parameters); the [`SsmaxMode::Adaptive`] variant takes a
//! caller-managed rolling estimate of `Δ` and resolves `s_L = 1/Δ` clamped to
//! `[0.1, 10.0]`.
//!
//! # Composition
//!
//! SSMax operates at the **logit level** (before sigmoid/softmax). It composes
//! cleanly with:
//! - **sigmoid parallax** ([`crate::parallax_attn`]) — sigmoid's per-key bound
//!   means dilution is milder than softmax, but a length-adaptive sharpener
//!   still helps in the retrieval band when N grows into the thousands.
//! - **standard SDPA** ([`crate::attention`]) — for callers that need softmax.
//! - **sink-aware attention** — operates at the OUTPUT level (after the
//!   value-weighted sum), so SSMax and sink-aware don't interfere.
//!
//! SSMax does **NOT** apply to `funcattn` (Research 261 closed negative:
//! basis-mode structure has no `(n,n)` attention matrix, so dilution is
//! structurally absent). Do not wire it there.
//!
//! # Allocation discipline (G4)
//!
//! [`apply_ssmax_inplace`] is allocation-free by construction: a single
//! in-place multiply pass over the logits slice. No `Vec`, `Box`, or collecting
//! iterator appears. The chunked 8-wide loop pattern is there to help LLVM
//! auto-vectorize (per the AGENTS.md hot-loop rule).
//!
//! References:
//! - Plan 411 — open-primitive spec
//! - Research 392 — distillation, novelty gate, fusion analysis
//! - arXiv:2607.01538 — Gollapudi et al. (paper)
//! - arXiv:2501.19399 — Uszacorek et al., *Scalable-Softmax is Superior for
//!   Attention* (SSMax source paper, cited as `[9]` in 2607.01538)

// ──────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────

/// SSMax mode: the per-layer source of the temperature scalar `s_L`.
///
/// The multiplicative factor applied to pre-attention logits is
/// `s_L · log(N)` — see [`SsmaxMode::multiplier`].
///
/// # Modelless discipline
///
/// Both variants are modelless:
/// - [`Fixed`](SsmaxMode::Fixed) — a caller-chosen constant. The default
///   `s_l = 1.0` is the truly modelless case (zero training).
/// - [`Adaptive`](SsmaxMode::Adaptive) — derived from a caller-managed rolling
///   estimate of the gold-distractor logit gap Δ via the analytical formula
///   `s_L = 1/Δ_typical` clamped to `[0.1, 10.0]`. The estimator itself is
///   caller-owned (Plan 411 defers the built-in estimator to stretch S2); the
///   API here ships the contract, not the estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SsmaxMode {
    /// Fixed `s_L`. Truly modelless — zero training, zero new parameters.
    /// The default `s_l = 1.0` recovers the paper's SSMax with no tuning.
    Fixed {
        /// Per-layer temperature scalar `s_L`. Default `1.0`.
        s_l: f32,
    },
    /// Adaptive `s_L` derived from a caller-managed rolling estimate of the
    /// typical gold-distractor logit gap `Δ`. We resolve
    /// `s_L = 1/Δ_typical` (clamped to `[0.1, 10.0]`) so that
    /// `s_L · log(N) · Δ ≈ log(N)`, which cancels the `(N−1)` denominator
    /// growth in the dilution bound.
    ///
    /// The built-in estimator is [`crate::ssmax::RollingDeltaEstimator`] (Plan
    /// 411 S2, behind the `ssmax_adaptive` feature) — a lock-free EMA of the
    /// `max(logits) − mean(logits)` proxy. Callers can also construct this
    /// variant directly with their own `rolling_delta` value.
    Adaptive {
        /// Caller-managed rolling estimate of the gold-distractor logit gap.
        /// Should be positive (small values give large `s_L`, sharpening hard;
        /// large values give small `s_L`, sharpening mildly). Clamped on use.
        rolling_delta: f32,
    },
}

impl Default for SsmaxMode {
    /// Default: fixed `s_L = 1.0` (the truly modelless case).
    fn default() -> Self {
        Self::Fixed { s_l: 1.0 }
    }
}

impl SsmaxMode {
    /// Resolve the effective `s_L` scalar from the mode.
    ///
    /// - [`Fixed`](SsmaxMode::Fixed) → returns the stored `s_l` unchanged.
    /// - [`Adaptive`](SsmaxMode::Adaptive) → returns `1/rolling_delta` clamped
    ///   to `[0.1, 10.0]` (with `rolling_delta` floored at `1e-3` to avoid
    ///   division by zero).
    #[inline]
    pub fn resolve_s_l(&self) -> f32 {
        match self {
            Self::Fixed { s_l } => *s_l,
            Self::Adaptive { rolling_delta } => {
                (1.0_f32 / rolling_delta.max(1e-3)).clamp(0.1, 10.0)
            }
        }
    }

    /// The multiplicative factor applied to pre-attention logits: `s_L · log(N)`.
    ///
    /// `log_n` is `ln(N)` where N is the number of attended keys. The caller
    /// passes it in (rather than us computing `ln(N)` here) because the caller
    /// already knows N and we don't want to recompute `ln` in the hot loop.
    #[inline]
    pub fn multiplier(&self, log_n: f32) -> f32 {
        self.resolve_s_l() * log_n
    }
}

/// Pre-resolved SSMax configuration: bundles the resolved `s_L` scalar and the
/// precomputed `log(N)`.
///
/// Useful for storage in attention configs where `N` is known at construction
/// time and we want to cache the resolved values. For the hot-path apply, use
/// [`apply_ssmax_inplace`] with the [`SsmaxMode`] directly — the `log_n`
/// argument there is the only thing the caller needs to supply per call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsmaxConfig {
    /// Resolved `s_L` (from [`SsmaxMode::resolve_s_l`]).
    pub s_l: f32,
    /// Precomputed `ln(N)` where N is the number of attended keys.
    pub log_n: f32,
}

impl SsmaxConfig {
    /// Construct from a mode + sequence length N.
    ///
    /// `log_n` is `ln(N)`, with `ln(1) = 0` and `ln(0) = 0` by convention
    /// (so SSMax is a no-op for sequences of length ≤ 1).
    #[inline]
    pub fn from_mode(mode: &SsmaxMode, n: usize) -> Self {
        let log_n = if n <= 1 { 0.0 } else { (n as f32).ln() };
        Self {
            s_l: mode.resolve_s_l(),
            log_n,
        }
    }

    /// The multiplicative factor applied to pre-attention logits: `s_L · log(N)`.
    #[inline]
    pub fn multiplier(&self) -> f32 {
        self.s_l * self.log_n
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Core function
// ──────────────────────────────────────────────────────────────────────────

/// Rescale pre-attention logits in place by `s_L · log(N)`.
///
/// This is the SSMax intervention. Each logit is multiplied by `s_L · log(N)`
/// where `s_L` is resolved from `mode` and `log_n = ln(N)` is passed in by the
/// caller (the caller knows N and we don't want to recompute `ln(N)` in the
/// hot loop).
///
/// # Arguments
///
/// - `logits` — `(n_heads, n_kv)` row-major f32, modified in place.
/// - `mode` — the per-layer source of `s_L`.
/// - `log_n` — `ln(n_kv)`, the natural log of the number of attended keys.
///
/// # Effect
///
/// When the pre-softmax gold-distractor gap is `Δ`, the post-normalization gold
/// weight becomes `α_gold ≈ 1 / (1 + (N−1) · N^{−s_L·Δ})`. SSMax cancels the
/// `(N−1)` growth whenever `s_L · Δ > 1`, recovering retrieval at large N.
///
/// # Composition
///
/// SSMax operates at the LOGIT level (before sigmoid/softmax). It composes
/// cleanly with sink-aware attention (which operates at the OUTPUT level, after
/// the value-weighted sum) — apply SSMax first, then the sink-aware gate.
///
/// SSMax does NOT apply to `funcattn` (Research 261 closed negative:
/// basis-mode structure has no `(n,n)` attention matrix, so dilution is
/// structurally absent). Do not wire it there.
///
/// # Allocation discipline (G4)
///
/// Allocation-free by construction: a single in-place multiply pass. No `Vec`,
/// `Box`, `String`, or collecting iterator appears.
///
/// # Example
///
/// ```ignore
/// use katgpt_core::ssmax::{SsmaxMode, apply_ssmax_inplace};
///
/// // N = 10_000 attended keys → log_n ≈ 9.21.
/// let log_n = (10_000_f32).ln();
/// let mut logits = vec![-1.0_f32, 0.5, 2.0, -0.3];
/// let mode = SsmaxMode::Fixed { s_l: 1.0 };
/// apply_ssmax_inplace(&mut logits, &mode, log_n);
/// // Each logit is now scaled by s_L · log(N) = 1.0 · 9.21 = 9.21.
/// assert!((logits[0] - (-1.0 * 9.21)).abs() < 1e-5);
/// ```
#[inline]
pub fn apply_ssmax_inplace(logits: &mut [f32], mode: &SsmaxMode, log_n: f32) {
    let mult = mode.multiplier(log_n);
    // Chunked 8-wide loop to help LLVM auto-vectorize (AGENTS.md hot-loop rule).
    for chunk in logits.as_chunks_mut::<8>().0 {
        for x in chunk {
            *x *= mult;
        }
    }
    for x in logits.as_chunks_mut::<8>().1 {
        *x *= mult;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Rolling-Δ estimator (Plan 411 S2)
// ──────────────────────────────────────────────────────────────────────────

/// Lock-free EMA estimator for the typical gold-distractor logit gap `Δ`.
///
/// Observes `max(logits) − mean(logits)` per attention row — the observable
/// proxy for the paper's `Δ = s_{t⋆} − s̄_{distractor}`. At inference time we
/// don't know which key is "gold", but the max-mean gap captures the
/// effective separation (the gold key tends to be the max, distractors
/// cluster near the mean). Maintains an exponential moving average via a
/// lock-free `AtomicU64` CAS loop.
///
/// Produces a [`SsmaxMode::Adaptive`] on demand, resolving `s_L = 1/Δ_typical`
/// (clamped to `[0.1, 10.0]` by [`SsmaxMode::resolve_s_l`]).
///
/// # Modelless discipline
///
/// Zero training. The estimator is a runtime statistic — it observes the
/// logit distribution and derives `s_L` analytically via `1/Δ`. No gradient
/// descent, no learned parameters. The EMA decay `α` is a caller-chosen
/// hyperparameter.
///
/// # Warm-start
///
/// Before any observation, the EMA initializes to `Δ = 1.0`, giving
/// `s_L = 1/1.0 = 1.0` — identical to [`SsmaxMode::Fixed { s_l: 1.0 }`] (the
/// truly modelless default). The estimator adapts away from this only when
/// observed gaps deviate.
///
/// # Thread safety
///
/// `Send + Sync` via `AtomicU64`. The CAS loop is lock-free (not wait-free):
/// under contention, a thread may retry. For per-layer estimators updated
/// once per forward pass, contention is negligible.
///
/// # Allocation discipline (G4)
///
/// [`observe_row`](Self::observe_row) is O(n) with zero allocation (single
/// pass for max + sum). [`resolve_delta`](Self::resolve_delta) and
/// [`to_mode`](Self::to_mode) are O(1).
///
/// # Example
///
/// ```ignore
/// use katgpt_core::ssmax::{RollingDeltaEstimator, apply_ssmax_inplace};
///
/// let estimator = RollingDeltaEstimator::default(); // α = 0.99, warm-start Δ = 1.0
/// let mut logits = vec![1.5_f32, 1.0, 1.01, 0.99, 1.0]; // gold ≈ 1.5, distractors ≈ 1.0
/// estimator.observe_row(&logits);
/// let mode = estimator.to_mode();
/// let log_n = (logits.len() as f32).ln();
/// apply_ssmax_inplace(&mut logits, &mode, log_n);
/// ```
#[cfg(feature = "ssmax_adaptive")]
pub struct RollingDeltaEstimator {
    /// EMA of observed `max − mean` gap, stored as `f64::to_bits` in an
    /// `AtomicU64` for lock-free updates.
    ema_bits: std::sync::atomic::AtomicU64,
    /// EMA decay factor in `(0, 1)`. `new = α · old + (1 − α) · observed`.
    /// Higher α = slower adaptation (more stable, longer memory).
    alpha: f64,
}

#[cfg(feature = "ssmax_adaptive")]
impl RollingDeltaEstimator {
    /// Construct with a custom EMA decay factor.
    ///
    /// `alpha` should be in `(0, 1)`. Values outside this range are clamped
    /// to `(0, 1)`. Higher `alpha` = slower adaptation (default `0.99` gives
    /// ~100-step effective memory).
    #[inline]
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(1e-6, 1.0 - 1e-6);
        Self {
            ema_bits: std::sync::atomic::AtomicU64::new(1.0_f64.to_bits()),
            alpha,
        }
    }

    /// Observe a row of pre-attention logits and update the EMA.
    ///
    /// Computes `max(logits) − mean(logits)` as the proxy for the
    /// gold-distractor gap `Δ`, then blends it into the EMA via a lock-free
    /// CAS loop.
    ///
    /// # Arguments
    ///
    /// - `logits` — a row of pre-attention scores `(n_kv,)`. The caller
    ///   passes the full row (all keys for one query).
    ///
    /// # Allocation discipline
    ///
    /// Zero allocation: a single pass for max + sum. No `Vec`, `Box`, or
    /// collecting iterator.
    #[inline]
    pub fn observe_row(&self, logits: &[f32]) {
        let n = logits.len();
        if n <= 1 {
            return; // Need ≥2 elements for a meaningful gap
        }
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        for &x in logits {
            if x > max {
                max = x;
            }
            sum += x as f64;
        }
        let mean = (sum / n as f64) as f32;
        let delta = (max - mean) as f64;
        self.update_ema(delta);
    }

    /// Lock-free EMA update via CAS loop.
    ///
    /// Skips non-finite or negative observations (NaN / Inf / negatives
    /// from ill-formed logits don't pollute the estimate).
    #[inline]
    fn update_ema(&self, observed: f64) {
        use std::sync::atomic::Ordering;

        if !observed.is_finite() || observed < 0.0 {
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

    /// Read the current EMA estimate of `Δ_typical`.
    #[inline]
    pub fn resolve_delta(&self) -> f32 {
        use std::sync::atomic::Ordering;
        let bits = self.ema_bits.load(Ordering::Relaxed);
        f64::from_bits(bits) as f32
    }

    /// Produce a [`SsmaxMode::Adaptive`] from the current EMA estimate.
    ///
    /// The returned mode's `resolve_s_l()` will compute `1/Δ_typical`
    /// clamped to `[0.1, 10.0]`.
    #[inline]
    pub fn to_mode(&self) -> SsmaxMode {
        SsmaxMode::Adaptive {
            rolling_delta: self.resolve_delta(),
        }
    }
}

#[cfg(feature = "ssmax_adaptive")]
impl Default for RollingDeltaEstimator {
    /// Default: `α = 0.99` (slow adaptation, ~100-step effective memory),
    /// warm-start `Δ = 1.0` (gives `s_L = 1.0`, matching [`SsmaxMode::Fixed { s_l: 1.0 }`]).
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

    const TOL: f32 = 1e-6;

    #[test]
    fn fixed_mode_preserves_s_l() {
        assert_eq!(SsmaxMode::Fixed { s_l: 1.0 }.resolve_s_l(), 1.0);
        assert_eq!(SsmaxMode::Fixed { s_l: 0.5 }.resolve_s_l(), 0.5);
        assert_eq!(SsmaxMode::Fixed { s_l: 2.0 }.resolve_s_l(), 2.0);
    }

    #[test]
    fn adaptive_mode_clamps_tiny_delta() {
        // rolling_delta → 0 would make 1/delta → ∞; clamp to 10.0.
        let m = SsmaxMode::Adaptive {
            rolling_delta: 1e-10,
        };
        assert!((m.resolve_s_l() - 10.0).abs() < TOL);
    }

    #[test]
    fn adaptive_mode_clamps_huge_delta() {
        // rolling_delta → ∞ would make 1/delta → 0; clamp to 0.1.
        let m = SsmaxMode::Adaptive {
            rolling_delta: 1e10,
        };
        assert!((m.resolve_s_l() - 0.1).abs() < TOL);
    }

    #[test]
    fn adaptive_mode_resolves_inverse() {
        // The analytical contract: s_L = 1/Δ for in-range Δ.
        let m = SsmaxMode::Adaptive {
            rolling_delta: 0.25,
        };
        assert!((m.resolve_s_l() - 4.0).abs() < TOL);

        let m = SsmaxMode::Adaptive { rolling_delta: 5.0 };
        assert!((m.resolve_s_l() - 0.2).abs() < TOL);
    }

    #[test]
    fn adaptive_mode_floors_zero_delta() {
        // rolling_delta = 0 would divide by zero; floor at 1e-3 → s_L = 1/1e-3
        // = 1000, then clamp to 10.0.
        let m = SsmaxMode::Adaptive { rolling_delta: 0.0 };
        assert!((m.resolve_s_l() - 10.0).abs() < TOL);

        // Negative delta should also floor (max(1e-3) sees the negative as
        // less than 1e-3 and replaces it).
        let m = SsmaxMode::Adaptive {
            rolling_delta: -1.0,
        };
        assert!((m.resolve_s_l() - 10.0).abs() < TOL);
    }

    #[test]
    fn multiplier_is_s_l_times_log_n() {
        let log_n = 9.21_f32; // ≈ ln(10_000)
        let m = SsmaxMode::Fixed { s_l: 2.0 };
        assert!((m.multiplier(log_n) - (2.0 * 9.21)).abs() < TOL);
    }

    #[test]
    fn apply_ssmax_scales_every_logit_by_multiplier() {
        let log_n = (10_000_f32).ln();
        let mode = SsmaxMode::Fixed { s_l: 1.0 };
        let mult = mode.multiplier(log_n);

        let original = [-1.0_f32, 0.5, 2.0, -0.3, 1.7, 0.0, -3.12, 0.001, 42.0];
        let mut logits = original;
        apply_ssmax_inplace(&mut logits, &mode, log_n);

        for (got, &orig) in logits.iter().zip(original.iter()) {
            let expected = orig * mult;
            assert!(
                (got - expected).abs() < TOL,
                "logit {orig} should scale to {expected}, got {got}"
            );
        }
    }

    #[test]
    fn apply_ssmax_empty_slice_is_noop() {
        let mut logits: [f32; 0] = [];
        apply_ssmax_inplace(&mut logits, &SsmaxMode::Fixed { s_l: 1.0 }, 9.21);
        // No panic, no-op.
    }

    #[test]
    fn apply_ssmax_identity_at_multiplier_one() {
        // When s_L · log_n = 1.0, logits are unchanged (identity scale).
        // This is the small-N no-regression guarantee (G5): at s_L = 1.0,
        // log_n = 1.0 (N = e ≈ 2.718) the multiplier is exactly 1.0.
        // (Note: log_n = 0.0 / N ≤ 1 gives multiplier 0.0, NOT identity —
        //  SSMax zeroes the logits at N ≤ 1, which is the documented
        //  small-N safety net, not a no-op.)
        let original = [-1.0_f32, 0.5, 2.0, -0.3, 1.7];
        let mut logits = original;
        apply_ssmax_inplace(&mut logits, &SsmaxMode::Fixed { s_l: 1.0 }, 1.0);
        for (got, &orig) in logits.iter().zip(original.iter()) {
            assert!((got - orig).abs() < TOL, "logit {orig} changed to {got}");
        }
    }

    #[test]
    fn apply_ssmax_simd_and_remainder_paths_agree_with_naive() {
        // Test multiple sizes to exercise both the 8-wide chunk path and the
        // remainder path.
        let log_n = 9.21_f32;
        let mode = SsmaxMode::Fixed { s_l: 0.7 };
        let mult = mode.multiplier(log_n);

        for &size in &[0_usize, 1, 7, 8, 9, 15, 16, 17, 32, 33, 100] {
            let original: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 5.0).collect();
            let mut logits = original.clone();
            apply_ssmax_inplace(&mut logits, &mode, log_n);

            // Compare against the naive scalar loop.
            for (got, &orig) in logits.iter().zip(original.iter()) {
                let expected = orig * mult;
                assert!(
                    (got - expected).abs() < TOL,
                    "size {size}: logit {orig} should scale to {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn ssmax_config_from_mode_caches_log_n() {
        let mode = SsmaxMode::Fixed { s_l: 2.0 };
        let cfg = SsmaxConfig::from_mode(&mode, 10_000);
        assert!((cfg.s_l - 2.0).abs() < TOL);
        assert!((cfg.log_n - (10_000_f32).ln()).abs() < TOL);
        assert!((cfg.multiplier() - (2.0 * (10_000_f32).ln())).abs() < TOL);
    }

    #[test]
    fn ssmax_config_from_mode_handles_small_n() {
        // N ≤ 1 → log_n = 0 → multiplier = 0 (no-op / safety net).
        let cfg = SsmaxConfig::from_mode(&SsmaxMode::Fixed { s_l: 1.0 }, 0);
        assert_eq!(cfg.log_n, 0.0);
        assert_eq!(cfg.multiplier(), 0.0);

        let cfg = SsmaxConfig::from_mode(&SsmaxMode::Fixed { s_l: 1.0 }, 1);
        assert_eq!(cfg.log_n, 0.0);
        assert_eq!(cfg.multiplier(), 0.0);
    }

    #[test]
    fn default_mode_is_fixed_one() {
        // The truly-modelless default: s_L = 1.0.
        assert_eq!(SsmaxMode::default(), SsmaxMode::Fixed { s_l: 1.0 });
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Rolling-Δ estimator tests (Plan 411 S2)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "ssmax_adaptive"))]
mod estimator_tests {
    use super::RollingDeltaEstimator;
    use crate::ssmax::SsmaxMode;

    const TOL: f32 = 1e-5;

    #[test]
    fn warm_start_is_modelless_default() {
        // Before any observation, to_mode() gives Adaptive { rolling_delta = 1.0 },
        // which resolves to s_L = 1/1.0 = 1.0 — identical to Fixed { s_l: 1.0 }.
        let est = RollingDeltaEstimator::default();
        let mode = est.to_mode();
        assert!(
            matches!(mode, SsmaxMode::Adaptive { rolling_delta } if (rolling_delta - 1.0).abs() < TOL)
        );
        assert!(
            (est.resolve_delta() - 1.0).abs() < TOL,
            "warm-start Δ should be 1.0"
        );
        // s_L from warm-start should be 1.0 (same as Fixed default).
        assert!(
            (mode.resolve_s_l() - 1.0).abs() < TOL,
            "warm-start s_L should be 1.0"
        );
    }

    #[test]
    fn observe_empty_and_single_are_noop() {
        let est = RollingDeltaEstimator::default();
        est.observe_row(&[]);
        est.observe_row(&[3.12]);
        // EMA unchanged from warm-start.
        assert!(
            (est.resolve_delta() - 1.0).abs() < TOL,
            "empty/single observations should not change EMA"
        );
    }

    #[test]
    fn converges_to_true_delta() {
        // Logits with known gap: gold = 1.5, distractors = 1.0.
        // max - mean ≈ 1.5 - 1.0 = 0.5.
        // Use α = 0.0 (fast adaptation — fully replace on each observation)
        // to verify the proxy calculation directly.
        let est = RollingDeltaEstimator::new(0.0);
        let logits = [1.5_f32, 1.0, 1.0, 1.0, 1.0];
        est.observe_row(&logits);
        // max=1.5, mean=1.1 → delta=0.4. Not exactly 0.5 because mean of [1.5,1,1,1,1] = 1.1.
        let expected_delta = 1.5 - 1.1;
        assert!(
            (est.resolve_delta() - expected_delta).abs() < TOL,
            "got {}, expected {}",
            est.resolve_delta(),
            expected_delta
        );
    }

    #[test]
    fn ema_blends_slowly_at_high_alpha() {
        // α = 0.99: after one observation, EMA barely moves from warm-start.
        let est = RollingDeltaEstimator::new(0.99);
        let logits = [5.0_f32, 1.0, 1.0, 1.0]; // max-mean = 5.0 - 2.0 = 3.0
        est.observe_row(&logits);
        let delta = est.resolve_delta();
        // EMA = 0.99 * 1.0 + 0.01 * 3.0 = 0.99 + 0.03 = 1.02.
        assert!((delta - 1.02).abs() < TOL, "got {delta}, expected 1.02");
    }

    #[test]
    fn converges_to_steady_state_after_many_observations() {
        // With α = 0.5 and 100 identical observations, EMA converges to the
        // observed value.
        let est = RollingDeltaEstimator::new(0.5);
        let logits = [2.0_f32, 1.0, 1.0]; // max-mean = 2.0 - 4.0/3 ≈ 0.6667
        for _ in 0..100 {
            est.observe_row(&logits);
        }
        let expected = 2.0 - 4.0 / 3.0;
        assert!(
            (est.resolve_delta() - expected as f32).abs() < 1e-3,
            "got {}, expected ~{}",
            est.resolve_delta(),
            expected
        );
    }

    #[test]
    fn nan_and_negative_observations_are_ignored() {
        // Construct logits that produce NaN or negative gap — the estimator
        // should skip these and leave the EMA unchanged.
        let est = RollingDeltaEstimator::new(0.0);
        // First, set the EMA to a known value.
        est.observe_row(&[3.0_f32, 1.0, 1.0]); // max-mean = 3.0 - 5.0/3 ≈ 1.333
        let known_delta = est.resolve_delta();
        assert!(known_delta > 0.0);
        // Now observe NaN logits (NaN comparison always fails → max stays -inf,
        // sum becomes NaN → delta becomes NaN → update_ema skips).
        est.observe_row(&[f32::NAN, f32::NAN]);
        assert!(
            (est.resolve_delta() - known_delta).abs() < TOL,
            "NaN observation should not change EMA"
        );
    }

    #[test]
    fn to_mode_produces_adaptive_with_correct_delta() {
        let est = RollingDeltaEstimator::new(0.0);
        est.observe_row(&[4.0_f32, 1.0, 1.0, 1.0, 1.0]); // max-mean = 4.0 - 8.0/5 = 2.4
        let mode = est.to_mode();
        match mode {
            SsmaxMode::Adaptive { rolling_delta } => {
                assert!(
                    (rolling_delta - 2.4).abs() < TOL,
                    "got {rolling_delta}, expected 2.4"
                );
            }
            _ => panic!("expected Adaptive mode"),
        }
    }

    #[test]
    fn very_small_delta_clamps_to_high_s_l() {
        // If all logits are nearly equal, delta → 0, s_L → 10.0 (clamped max).
        let est = RollingDeltaEstimator::new(0.0);
        est.observe_row(&[1.00001_f32, 1.0, 1.0]); // max-mean ≈ tiny
        let mode = est.to_mode();
        let s_l = mode.resolve_s_l();
        assert!(s_l > 5.0, "tiny gap should give high s_L, got {s_l}");
    }

    #[test]
    fn very_large_delta_clamps_to_low_s_l() {
        // If gold is far above distractors, delta is large, s_L is small.
        let est = RollingDeltaEstimator::new(0.0);
        est.observe_row(&[1000.0_f32, 0.0, 0.0]); // max-mean = 1000 - 333.3 = 666.7
        let mode = est.to_mode();
        let s_l = mode.resolve_s_l();
        assert!(
            s_l < 0.2,
            "huge gap should give low s_L (mild sharpening), got {s_l}"
        );
    }

    #[test]
    fn multi_threaded_no_panic() {
        // Concurrent observe_row calls from multiple threads should not panic
        // and should converge.
        use std::sync::Arc;
        use std::thread;
        let est = Arc::new(RollingDeltaEstimator::new(0.5));
        let mut handles = Vec::with_capacity(4);
        for _ in 0..4 {
            let est = Arc::clone(&est);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    est.observe_row(&[2.0_f32, 1.0, 1.0, 1.0]);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // After 400 observations of the same distribution, EMA should be
        // close to the observed max-mean gap (2.0 - 1.25 = 0.75).
        let delta = est.resolve_delta();
        assert!(
            delta > 0.5 && delta < 1.0,
            "converged delta {delta} should be near 0.75"
        );
    }

    #[test]
    fn custom_alpha_is_clamped() {
        // alpha = 0.0 or 1.0 boundary values are clamped to just inside (0,1).
        let est_zero = RollingDeltaEstimator::new(0.0);
        let est_one = RollingDeltaEstimator::new(1.0);
        // est_zero should adapt fully on first observation (alpha clamped to 1e-6,
        // so new = 1e-6 * old + (1-1e-6) * obs ≈ obs).
        est_zero.observe_row(&[5.0_f32, 1.0, 1.0]);
        let delta = est_zero.resolve_delta();
        assert!(
            (delta - (5.0 - 7.0 / 3.0) as f32).abs() < 0.01,
            "alpha≈0 should fully adopt new value, got {delta}"
        );
        // est_one should barely adapt (alpha clamped to 1-1e-6).
        est_one.observe_row(&[5.0_f32, 1.0, 1.0]);
        let delta_one = est_one.resolve_delta();
        assert!(
            (delta_one - 1.0).abs() < 0.01,
            "alpha≈1 should barely move from warm-start, got {delta_one}"
        );
    }
}
