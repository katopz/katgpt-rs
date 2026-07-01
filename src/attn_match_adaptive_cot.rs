//! Adaptive CoT compaction — entropy-thresholded, bandit-tuned online
//! compaction for thinking traces (Plan 271 Phase 6).
//!
//! During long CoT reasoning, the model's per-token next-token distribution
//! carries a strong signal about whether the current step is "exploratory"
//! (high entropy — many plausible next tokens, lots of information) or
//! "predictable" (low entropy — one dominant next token, low information).
//!
//! # Heuristic
//!
//! - **Compact when entropy is LOW** (θ_low). Predictable stretches are
//!   compressible: the model is just walking through a deterministic chain
//!   whose individual tokens matter less than the conclusion.
//! - **Preserve when entropy is HIGH** (θ_high). Exploratory stretches carry
//!   information: alternative branches, key insights, branch-and-bound
//!   decisions. Compacting these risks losing the very tokens the model would
//!   attend back to.
//! - **In between (θ_low ≤ H ≤ θ_high)**: defer to the bandit. The UCB1
//!   FrequencyBandit (Plan 189) learns over time whether to be aggressive
//!   (Low band) or conservative (High band) in this ambiguous regime.
//!
//! # Self-Learning
//!
//! The `(θ_low, θ_high)` thresholds are *adjusted* by the bandit — not by LLM
//! training. After each trace, the caller reports a downstream reward
//! (`+1` correct, `-1` incorrect, `0` partial). The bandit updates its
//! Q-values, and on the next trace its selected band shifts the thresholds:
//!
//! - `FrequencyBand::Low` → lower θ_low by `BANDIT_ADJUST_STEP` (compact
//!   more aggressively in the ambiguous regime).
//! - `FrequencyBand::Mid` → keep thresholds.
//! - `FrequencyBand::High` → raise θ_low by `BANDIT_ADJUST_STEP` (compact
//!   more conservatively).
//!
//! Over many traces the bandit converges to the band whose reward is highest
//! for the actual workload.
//!
//! # Entropy Smoothing
//!
//! Per-token entropy is noisy (it spikes on every punctuation token). We EMA-
//! smooth with α=0.1 so decisions reflect the *recent trend*, not the latest
//! token.
//!
//! # TL;DR
//!
//! [`AdaptiveTraceCompactor`] wraps [`OnlineCompactor`] with an entropy gate
//! and a [`FrequencyBandit`] threshold tuner. Call [`observe_entropy`] every
//! token, [`maybe_compact_adaptive`] when you'd normally consider compaction,
//! and [`update_reward`] after each trace.

use crate::attn_match::compact::CompactError;
use crate::attn_match::online::{OnlineCompactResult, OnlineCompactor};
use crate::attn_match::types::AmConfig;
use crate::freq_bandit::{FrequencyBand, FrequencyBandit};
use crate::types::Rng;

/// Per-trace cap on compactions. Prevents a long trace from compacting
/// itself into oblivion — even if entropy stays low, we want a floor on
/// how often we touch the cache mid-trace.
pub const DEFAULT_MAX_COMPACTS: usize = 8;

/// EMA smoothing factor for per-token entropy. α=0.1 means the EMA tracks
/// the average over the last ~10 tokens — enough to absorb single-token
/// spikes without lagging the underlying trend.
pub const DEFAULT_EMA_ALPHA: f32 = 0.1;

/// Step size for bandit-driven threshold adjustments.
/// Each band decision shifts θ_low by ± this amount.
pub const BANDIT_ADJUST_STEP: f32 = 0.05;

/// Result of an adaptive compaction pass.
#[derive(Clone, Debug)]
pub struct AdaptiveCompactResult {
    /// The underlying online compaction result (compact prefix + recent).
    pub online: OnlineCompactResult,
    /// EMA entropy at the moment compaction was triggered.
    pub entropy_at_decision: f32,
    /// Remaining compactions allowed for this trace
    /// (`max_compacts - compacts_done` after this compaction).
    pub compacts_remaining: usize,
}

/// Entropy-thresholded, bandit-tuned online compactor for thinking traces.
///
/// See the [module docs](self) for the full design.
///
/// Note: does not derive `Clone`/`Debug` because [`FrequencyBandit`] (Plan 189)
/// doesn't either — use [`bandit`](Self::bandit) for inspection.
pub struct AdaptiveTraceCompactor {
    online: OnlineCompactor,
    bandit: FrequencyBandit,
    theta_low: f32,
    theta_high: f32,
    max_compacts: usize,
    compacts_done: usize,
    ema_entropy: f32,
    ema_alpha: f32,
}

impl std::fmt::Debug for AdaptiveTraceCompactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveTraceCompactor")
            .field("online", &self.online)
            .field("theta_low", &self.theta_low)
            .field("theta_high", &self.theta_high)
            .field("max_compacts", &self.max_compacts)
            .field("compacts_done", &self.compacts_done)
            .field("ema_entropy", &self.ema_entropy)
            .field("ema_alpha", &self.ema_alpha)
            .field("bandit_total_pulls", &self.bandit.total_pulls())
            .field("bandit_best_arm", &self.bandit.best_arm())
            .finish()
    }
}

impl AdaptiveTraceCompactor {
    /// Create a new adaptive trace compactor.
    ///
    /// # Arguments
    /// * `phys_budget` — forwarded to [`OnlineCompactor`].
    /// * `recent_window` — forwarded to [`OnlineCompactor`].
    /// * `theta_low` — compact when EMA entropy is below this.
    /// * `theta_high` — preserve when EMA entropy is above this.
    /// * `max_compacts` — per-trace cap on compactions.
    pub fn new(
        phys_budget: usize,
        recent_window: usize,
        theta_low: f32,
        theta_high: f32,
        max_compacts: usize,
    ) -> Self {
        assert!(theta_low >= 0.0, "theta_low must be >= 0");
        assert!(
            theta_high > theta_low,
            "theta_high ({theta_high}) must be > theta_low ({theta_low})"
        );
        assert!(max_compacts > 0, "max_compacts must be > 0");
        Self {
            online: OnlineCompactor::new(phys_budget, recent_window),
            bandit: FrequencyBandit::new(),
            theta_low,
            theta_high,
            max_compacts,
            compacts_done: 0,
            ema_entropy: 0.0,
            ema_alpha: DEFAULT_EMA_ALPHA,
        }
    }

    /// Compute next-token entropy from logits and update the EMA.
    ///
    /// `H = -Σ p_i * ln(p_i)` where `p = softmax(logits)`. Returns the raw
    /// (un-smoothed) entropy so callers can log it; decisions use the EMA.
    ///
    /// Softmax here is the math of entropy computation (not a routing
    /// decision), so it's correct per the project's "use sigmoid not
    /// softmax" rule — that rule applies to *routing*, not to computing
    /// probability distributions from logits.
    pub fn observe_entropy(&mut self, next_token_logits: &[f32]) -> f32 {
        let h = entropy_from_logits(next_token_logits);
        // EMA blend: ema = α * new + (1-α) * old.
        self.ema_entropy = self.ema_alpha * h + (1.0 - self.ema_alpha) * self.ema_entropy;
        h
    }

    /// Current EMA-smoothed entropy.
    #[inline]
    pub fn ema_entropy(&self) -> f32 {
        self.ema_entropy
    }

    /// Conditionally compact the KV cache.
    ///
    /// Returns `None` when any of:
    /// - `compacts_done >= max_compacts` (per-trace cap hit)
    /// - `ema_entropy > theta_high` (exploratory regime — preserve)
    /// - inner [`OnlineCompactor`] returns `None` (below phys budget)
    ///
    /// When `ema_entropy < theta_low`: triggers compaction eagerly (the bandit
    /// is consulted for threshold adjustment, but the decision is forced).
    /// When `theta_low <= ema_entropy <= theta_high`: bandit-selected band
    /// decides whether to compact at all in this ambiguous regime.
    ///
    /// On a successful compaction, `compacts_done` is incremented and the
    /// bandit's last-selected arm is updated with the trace's eventual reward
    /// via [`update_reward`].
    #[allow(clippy::too_many_arguments)] // hot-path: lane buffers bundled for zero-alloc inference
    pub fn maybe_compact_adaptive(
        &mut self,
        kv_keys: &[f32],
        kv_values: &[f32],
        queries: &[f32],
        current_pos: usize,
        d: usize,
        n: usize,
        config: &AmConfig,
    ) -> Result<Option<AdaptiveCompactResult>, CompactError> {
        if self.compacts_done >= self.max_compacts {
            return Ok(None);
        }
        if self.ema_entropy > self.theta_high {
            // Exploratory regime — preserve everything.
            return Ok(None);
        }

        // Consult bandit for threshold adjustment.
        // We use a deterministic seed-derived Rng; UCB1 selection is
        // deterministic once all arms are visited, so the seed only matters
        // for tie-breaking among unvisited arms (first call).
        let mut rng = Rng::new(current_pos as u64 | 1);
        let band = self.bandit.select_band(&mut rng);
        self.apply_bandit_adjustment(band);

        // Decide whether to actually compact.
        let should_compact = if self.ema_entropy < self.theta_low {
            // Low-entropy regime: compact eagerly.
            true
        } else {
            // Ambiguous regime: defer to bandit.
            // Low band → compact (aggressive), Mid/High band → skip this round.
            matches!(band, FrequencyBand::Low)
        };

        if !should_compact {
            return Ok(None);
        }

        let online_result = match self.online.maybe_compact(
            kv_keys,
            kv_values,
            queries,
            current_pos,
            d,
            n,
            config,
        )? {
            Some(r) => r,
            None => return Ok(None),
        };

        self.compacts_done += 1;
        let entropy_at_decision = self.ema_entropy;
        let compacts_remaining = self.max_compacts.saturating_sub(self.compacts_done);

        Ok(Some(AdaptiveCompactResult {
            online: online_result,
            entropy_at_decision,
            compacts_remaining,
        }))
    }

    /// Report downstream reward for the last bandit-selected band.
    ///
    /// Convention: `+1.0` correct, `-1.0` incorrect, `0.0` partial.
    /// Forwards to [`FrequencyBandit::update`] with the last-selected arm.
    /// No-op if the bandit has not yet selected an arm this trace.
    pub fn update_reward(&mut self, reward: f32) {
        if let Some(band) = self.bandit.last_selected() { self.bandit.update(band, reward as f64) }
    }

    /// Current `(theta_low, theta_high)`.
    #[inline]
    pub fn thresholds(&self) -> (f32, f32) {
        (self.theta_low, self.theta_high)
    }

    /// Set thresholds. Swaps `low`/`high` if they're reversed, and clamps
    /// both to `[0, +∞)`. Rejects equal thresholds (would create a
    /// zero-width ambiguous band).
    pub fn set_thresholds(&mut self, low: f32, high: f32) {
        let low = low.max(0.0);
        let high = high.max(0.0);
        let (low, high) = if low <= high {
            (low, high)
        } else {
            (high, low) // swap so low < high
        };
        // Guard against degenerate zero-width band.
        if high > low {
            self.theta_low = low;
            self.theta_high = high;
        }
        // else: silently reject — caller passed low == high (or both 0).
    }

    /// Compactions performed so far this trace.
    #[inline]
    pub fn compacts_done(&self) -> usize {
        self.compacts_done
    }

    /// Per-trace compaction cap.
    #[inline]
    pub fn max_compacts(&self) -> usize {
        self.max_compacts
    }

    /// Reference to the inner bandit (for inspection / serialization).
    pub fn bandit(&self) -> &FrequencyBandit {
        &self.bandit
    }

    /// Mutable reference to the inner bandit (for advanced tuning).
    pub fn bandit_mut(&mut self) -> &mut FrequencyBandit {
        &mut self.bandit
    }

    /// Reset per-trace state: clears `compacts_done` and `ema_entropy`.
    ///
    /// **Preserves**: bandit Q-values (so learning accumulates across
    /// traces), `theta_low` / `theta_high`, `max_compacts`, `ema_alpha`.
    pub fn reset(&mut self) {
        self.compacts_done = 0;
        self.ema_entropy = 0.0;
    }

    /// Apply the bandit's threshold adjustment for the selected band.
    ///
    /// - `Low` → θ_low -= step (compact more aggressively)
    /// - `Mid` → no change
    /// - `High` → θ_low += step (compact more conservatively)
    ///
    /// Clamped to `[0, theta_high - step]` so θ_low never crosses θ_high.
    fn apply_bandit_adjustment(&mut self, band: FrequencyBand) {
        match band {
            FrequencyBand::Low => {
                self.theta_low = (self.theta_low - BANDIT_ADJUST_STEP).max(0.0);
            }
            FrequencyBand::Mid => {}
            FrequencyBand::High => {
                self.theta_low = (self.theta_low + BANDIT_ADJUST_STEP)
                    .min(self.theta_high - BANDIT_ADJUST_STEP);
            }
        }
    }
}

impl Default for AdaptiveTraceCompactor {
    /// Sensible defaults for a 4k-token thinking trace:
    /// phys_budget=2048, recent_window=256, θ_low=0.5, θ_high=2.0, max=8.
    fn default() -> Self {
        Self::new(2048, 256, 0.5, 2.0, DEFAULT_MAX_COMPACTS)
    }
}

// ─── Entropy computation ───────────────────────────────────────────────────

/// Compute Shannon entropy (in nats) from a logits vector.
///
/// `H = -Σ p_i * ln(p_i)` where `p = softmax(logits)` with the standard
/// max-shift for numerical stability.
///
/// Returns 0 for empty input.
///
/// Hot-path: max-shift and per-element `exp` are computed once, then reused
/// for both the normalizer sum and the per-token `p * ln(p)` term.
pub fn entropy_from_logits(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    // Max-shift for numerical stability (SIMD-reduced).
    let max_logit = katgpt_core::simd::simd_max_f32(logits);
    // Single pass: shifted_exp[i] = exp(logits[i] - max_logit).
    let mut shifted_exp: Vec<f32> = Vec::with_capacity(logits.len());
    shifted_exp.extend(logits.iter().map(|&l| (l - max_logit).exp()));
    let sum_exp: f32 = shifted_exp.iter().copied().sum();
    if sum_exp <= 0.0 {
        return 0.0;
    }
    let inv_sum = 1.0 / sum_exp;
    // H = -Σ p_i * ln(p_i), with p_i = shifted_exp[i] * inv_sum.
    // Use ln(p) = ln(shifted_exp[i]) + ln(inv_sum) = ln(shifted_exp[i]) - ln(sum_exp)
    // → avoid one mul per iteration by folding the constant.
    let ln_inv_sum = inv_sum.ln(); // = -ln(sum_exp)
    let mut h = 0.0f32;
    for &e in &shifted_exp {
        if e > 0.0 {
            let p = e * inv_sum;
            // p * ln(p) = p * (ln(e) + ln(inv_sum)).
            h -= p * (e.ln() + ln_inv_sum);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_kv(t_len: usize, d: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
        let mut keys = vec![0.0f32; t_len * d];
        let mut values = vec![0.0f32; t_len * d];
        for i in 0..t_len {
            for k in 0..d {
                let x = ((i + seed) as f32) * 0.1 + (k as f32) * 0.01;
                keys[i * d + k] = x.sin() * 0.5;
                values[i * d + k] = x.cos() * 0.3;
            }
        }
        (keys, values)
    }

    fn synth_queries(n: usize, d: usize, seed: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; n * d];
        for i in 0..n {
            for k in 0..d {
                let x = ((i + seed + 100) as f32) * 0.2 + (k as f32) * 0.05;
                q[i * d + k] = x.sin() * 0.4;
            }
        }
        q
    }

    /// Peaked logits — low entropy (one dominant token).
    fn peaked_logits(n_classes: usize) -> Vec<f32> {
        let mut l = vec![-10.0; n_classes];
        l[0] = 10.0;
        l
    }

    /// Uniform logits — max entropy (= ln(n_classes)).
    fn uniform_logits(n_classes: usize) -> Vec<f32> {
        vec![1.0; n_classes]
    }

    #[test]
    fn test_observe_entropy_returns_correct_value() {
        // Uniform over N classes → H = ln(N).
        let n = 16usize;
        let uni = uniform_logits(n);
        let h = entropy_from_logits(&uni);
        let expected = (n as f32).ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "uniform entropy {h} should be ln({n}) = {expected}"
        );

        // Peaked → near-zero entropy.
        let peaked = peaked_logits(n);
        let h_peak = entropy_from_logits(&peaked);
        assert!(
            h_peak < 1e-3,
            "peaked entropy {h_peak} should be near 0"
        );

        // Empty → 0.
        assert_eq!(entropy_from_logits(&[]), 0.0);
    }

    #[test]
    fn test_observe_entropy_updates_ema() {
        let mut c = AdaptiveTraceCompactor::new(64, 8, 0.5, 2.0, 4);
        assert_eq!(c.ema_entropy(), 0.0);

        // First observation: ema = α * h.
        let h1 = c.observe_entropy(&uniform_logits(8));
        let expected_ema = 0.1 * h1; // default α = 0.1
        assert!(
            (c.ema_entropy() - expected_ema).abs() < 1e-6,
            "EMA after first obs: {} vs expected {}",
            c.ema_entropy(),
            expected_ema
        );

        // Second observation: ema = α*h2 + (1-α)*ema_old.
        let h2 = c.observe_entropy(&peaked_logits(8));
        let expected_ema2 = 0.1 * h2 + 0.9 * expected_ema;
        assert!(
            (c.ema_entropy() - expected_ema2).abs() < 1e-5,
            "EMA after second obs: {} vs expected {}",
            c.ema_entropy(),
            expected_ema2
        );
        // h1 was uniform (large), h2 was peaked (small) — ema dropped.
        assert!(c.ema_entropy() < expected_ema);
        let _ = h2; // suppress unused
    }

    #[test]
    fn test_low_entropy_triggers_compaction() {
        // Feed low-entropy logits until EMA settles, then verify compaction
        // triggers when phys budget is reached.
        let d = 8usize;
        let n = 4usize;
        let phys = 32usize;
        let window = 8usize;
        let pos = phys + window; // exactly at trigger threshold

        let (keys, values) = synth_kv(pos, d, 1);
        let queries = synth_queries(n, d, 1);
        let cfg = AmConfig::highest_attn(8);

        let mut c = AdaptiveTraceCompactor::new(phys, window, 0.5, 2.0, 4);
        // Warm up EMA with low-entropy observations.
        for _ in 0..50 {
            c.observe_entropy(&peaked_logits(32));
        }
        assert!(
            c.ema_entropy() < c.thresholds().0,
            "EMA {} should be below theta_low {} after warmup",
            c.ema_entropy(),
            c.thresholds().0
        );

        let r = c
            .maybe_compact_adaptive(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("compact ok");
        assert!(r.is_some(), "low entropy + at budget should trigger");
        let r = r.unwrap();
        assert_eq!(c.compacts_done(), 1);
        assert_eq!(r.compacts_remaining, 3);
        assert!(r.entropy_at_decision < c.thresholds().0);
    }

    #[test]
    fn test_high_entropy_prevents_compaction() {
        let d = 8usize;
        let n = 4usize;
        let phys = 32usize;
        let window = 8usize;
        let pos = phys + window;

        let (keys, values) = synth_kv(pos, d, 2);
        let queries = synth_queries(n, d, 2);
        let cfg = AmConfig::highest_attn(8);

        let mut c = AdaptiveTraceCompactor::new(phys, window, 0.5, 2.0, 4);
        // Warm up EMA with high-entropy observations.
        for _ in 0..50 {
            c.observe_entropy(&uniform_logits(32));
        }
        // Uniform over 32 classes → H ≈ ln(32) ≈ 3.47 > theta_high=2.0.
        assert!(
            c.ema_entropy() > c.thresholds().1,
            "EMA {} should be above theta_high {} after uniform warmup",
            c.ema_entropy(),
            c.thresholds().1
        );

        let r = c
            .maybe_compact_adaptive(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("compact ok");
        assert!(r.is_none(), "high entropy should prevent compaction");
        assert_eq!(c.compacts_done(), 0);
    }

    #[test]
    fn test_bandit_updates_thresholds_after_observations() {
        // Drive the bandit so that the Low arm is clearly preferred: reward
        // Low +1, Mid -1, High -1 repeatedly. After enough rounds the bandit
        // keeps picking Low → theta_low drifts downward monotonically.
        let mut c = AdaptiveTraceCompactor::new(64, 16, 1.0, 3.0, 16);
        let (low0, high0) = c.thresholds();

        // Manually exercise the bandit + threshold adjustment loop so the
        // test is deterministic (independent of UCB1's unvisited-arm order).
        // We bypass `maybe_compact_adaptive` here to directly verify the
        // threshold-adjustment wiring.
        for _ in 0..10 {
            // Pull each arm explicitly with controlled rewards.
            c.bandit_mut().update(FrequencyBand::Low, 1.0);
            c.bandit_mut().update(FrequencyBand::Mid, -1.0);
            c.bandit_mut().update(FrequencyBand::High, -1.0);
        }
        // Low has Q=+1, Mid/High have Q=-1 → best_arm is Low.
        assert_eq!(c.bandit().best_arm(), FrequencyBand::Low);

        // Now call apply_bandit_adjustment directly several times with Low.
        // theta_low should drop monotonically.
        for _ in 0..5 {
            c.apply_bandit_adjustment(FrequencyBand::Low);
        }
        let (low1, high1) = c.thresholds();
        assert!(
            low1 < low0,
            "theta_low should have dropped after 5 Low adjustments: {} -> {}",
            low0, low1
        );
        // theta_high is never touched by the bandit.
        assert!((high1 - high0).abs() < 1e-6);

        // Conversely, High should raise theta_low.
        let low_before_high = c.thresholds().0;
        for _ in 0..3 {
            c.apply_bandit_adjustment(FrequencyBand::High);
        }
        let low_after_high = c.thresholds().0;
        assert!(
            low_after_high > low_before_high,
            "theta_low should rise after 3 High adjustments: {} -> {}",
            low_before_high,
            low_after_high
        );
    }

    #[test]
    fn test_max_compacts_cap_respected() {
        let d = 8usize;
        let n = 4usize;
        let phys = 32usize;
        let window = 8usize;
        let pos = phys + window;
        let (keys, values) = synth_kv(pos, d, 4);
        let queries = synth_queries(n, d, 4);
        let cfg = AmConfig::highest_attn(8);

        let mut c = AdaptiveTraceCompactor::new(phys, window, 0.5, 2.0, 2);

        // Warm up with low entropy.
        for _ in 0..30 {
            c.observe_entropy(&peaked_logits(16));
        }

        // First two compactions should succeed.
        let r1 = c
            .maybe_compact_adaptive(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("ok");
        assert!(r1.is_some());
        let r2 = c
            .maybe_compact_adaptive(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("ok");
        assert!(r2.is_some());
        assert_eq!(c.compacts_done(), 2);

        // Third call: cap hit, returns None.
        let r3 = c
            .maybe_compact_adaptive(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("ok");
        assert!(r3.is_none(), "should hit max_compacts cap");
        assert_eq!(c.compacts_done(), 2, "compacts_done should not exceed cap");
    }

    #[test]
    fn test_reset_clears_trace_state() {
        let mut c = AdaptiveTraceCompactor::new(64, 16, 0.5, 2.0, 4);

        // Do some work.
        for _ in 0..10 {
            c.observe_entropy(&peaked_logits(8));
        }
        c.compacts_done = 3; // simulate work
        let pulls_before = c.bandit().total_pulls();
        let q_values_before: [f64; 3] = [
            c.bandit().q_value(FrequencyBand::Low),
            c.bandit().q_value(FrequencyBand::Mid),
            c.bandit().q_value(FrequencyBand::High),
        ];

        let ema_before = c.ema_entropy();
        assert!(ema_before > 0.0);
        assert_eq!(c.compacts_done(), 3);

        c.reset();

        // Per-trace state cleared.
        assert_eq!(c.compacts_done(), 0);
        assert_eq!(c.ema_entropy(), 0.0);

        // Bandit state preserved.
        assert_eq!(c.bandit().total_pulls(), pulls_before);
        let q_values_after: [f64; 3] = [
            c.bandit().q_value(FrequencyBand::Low),
            c.bandit().q_value(FrequencyBand::Mid),
            c.bandit().q_value(FrequencyBand::High),
        ];
        for i in 0..3 {
            assert!(
                (q_values_before[i] - q_values_after[i]).abs() < 1e-12,
                "bandit Q[{i}] changed across reset: {} vs {}",
                q_values_before[i],
                q_values_after[i]
            );
        }

        // Thresholds preserved.
        let (low, high) = c.thresholds();
        assert!((low - 0.5).abs() < 1e-6);
        assert!((high - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_thresholds_clamped_on_set() {
        let mut c = AdaptiveTraceCompactor::new(64, 16, 0.5, 2.0, 4);

        // Reversed order → should swap.
        c.set_thresholds(2.0, 0.5);
        let (low, high) = c.thresholds();
        assert!(low <= high, "low ({low}) should be <= high ({high}) after swap");
        assert!((low - 0.5).abs() < 1e-6);
        assert!((high - 2.0).abs() < 1e-6);

        // Negative → clamped to 0.
        c.set_thresholds(-1.0, 1.0);
        let (low, _high) = c.thresholds();
        assert!(low >= 0.0, "low should be clamped to >= 0, got {low}");
        assert!((low - 0.0).abs() < 1e-6);

        // Equal → rejected (no change from previous).
        let before_low = c.thresholds().0;
        let before_high = c.thresholds().1;
        c.set_thresholds(0.5, 0.5);
        assert!(
            (c.thresholds().0 - before_low).abs() < 1e-6,
            "equal thresholds should be rejected"
        );
        assert!(
            (c.thresholds().1 - before_high).abs() < 1e-6,
            "equal thresholds should be rejected"
        );
    }

    #[test]
    fn test_no_compaction_below_phys_budget() {
        let d = 8usize;
        let n = 4usize;
        let (keys, values) = synth_kv(16, d, 5); // way below budget
        let queries = synth_queries(n, d, 5);
        let cfg = AmConfig::highest_attn(8);

        let mut c = AdaptiveTraceCompactor::new(64, 16, 0.5, 2.0, 4);
        for _ in 0..30 {
            c.observe_entropy(&peaked_logits(16));
        }
        let r = c
            .maybe_compact_adaptive(&keys, &values, &queries, 16, d, n, &cfg)
            .expect("ok");
        assert!(r.is_none(), "should not compact below phys budget");
    }

    #[test]
    fn test_default_constructor() {
        let c = AdaptiveTraceCompactor::default();
        assert_eq!(c.max_compacts(), DEFAULT_MAX_COMPACTS);
        let (low, high) = c.thresholds();
        assert!((low - 0.5).abs() < 1e-6);
        assert!((high - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_bandit_adjustment_directions() {
        let mut c = AdaptiveTraceCompactor::new(64, 16, 1.0, 3.0, 4);
        let initial_low = c.theta_low;

        // Low band → decrease theta_low.
        c.apply_bandit_adjustment(FrequencyBand::Low);
        assert!(
            c.theta_low < initial_low,
            "Low band should decrease theta_low: {} vs {}",
            c.theta_low,
            initial_low
        );

        // Reset and test High band → increase theta_low.
        let mut c2 = AdaptiveTraceCompactor::new(64, 16, 1.0, 3.0, 4);
        let initial_low2 = c2.theta_low;
        c2.apply_bandit_adjustment(FrequencyBand::High);
        assert!(
            c2.theta_low > initial_low2,
            "High band should increase theta_low: {} vs {}",
            c2.theta_low,
            initial_low2
        );

        // Mid band → no change.
        let mut c3 = AdaptiveTraceCompactor::new(64, 16, 1.0, 3.0, 4);
        let initial_low3 = c3.theta_low;
        c3.apply_bandit_adjustment(FrequencyBand::Mid);
        assert!(
            (c3.theta_low - initial_low3).abs() < 1e-6,
            "Mid band should not change theta_low"
        );
    }

    #[test]
    fn test_update_reward_no_op_without_selection() {
        let mut c = AdaptiveTraceCompactor::new(64, 16, 0.5, 2.0, 4);
        // No bandit selection yet → update_reward should be a no-op.
        c.update_reward(1.0);
        assert_eq!(c.bandit().total_pulls(), 0);
    }

    #[test]
    fn test_update_reward_after_selection() {
        let mut c = AdaptiveTraceCompactor::new(64, 16, 0.5, 2.0, 4);
        // Force a selection by calling maybe_compact_adaptive (which calls
        // select_band internally).
        let d = 8usize;
        let n = 4usize;
        let pos = 80usize;
        let (keys, values) = synth_kv(pos, d, 6);
        let queries = synth_queries(n, d, 6);
        let cfg = AmConfig::highest_attn(8);
        for _ in 0..30 {
            c.observe_entropy(&peaked_logits(16));
        }
        let _ = c
            .maybe_compact_adaptive(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("ok");

        let pulls_before = c.bandit().total_pulls();
        // Note: select_band does NOT increment counts — only update does.
        // So pulls_before should be 0 here (selection happened, but no update yet).
        assert_eq!(pulls_before, 0);

        c.update_reward(1.0);
        assert_eq!(c.bandit().total_pulls(), 1);
    }
}
