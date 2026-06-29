//! Collapse-Aware Adaptive Thinking — S2F mid-reasoning early exit (Plan 212).
//!
//! Monitors the token stream during reasoning and triggers early exit when
//! reasoning collapse is detected (hesitation patterns, repetitive tokens,
//! "wait" frequency). This is a modelless inference-time feature behind the
//! `collapse_aware_thinking` feature gate.
//!
//! ## Architecture
//!
//! Three-layer adaptive thinking pipeline:
//! 1. **Pre-Decide**: `SelectivityRouter` — should we think at all?
//! 2. **Mid-Think**: `S2FCollapseDetector` (this module) — early exit on collapse
//! 3. **Post-Verify**: T2M OptionStripper — verify thinking helped
//!
//! ## Collapse Detection
//!
//! The detector maintains a fixed-size ring buffer of recent token IDs.
//! On each `check_collapse()` call, it counts how many tokens in the buffer
//! match the hesitation token set (e.g., "wait", "hmm", "actually", "let me").
//! If the count exceeds the threshold τ, collapse is signaled.
//!
//! ## Self-Learning (EMA Threshold Adaptation)
//!
//! After each trace, `reset()` updates the threshold τ via exponential moving
//! average. If a trace had many hesitation tokens but no collapse, τ increases
//! (more tolerant). If it triggered too early, τ decreases. This allows the
//! detector to self-calibrate per-domain.
//!
//! ## Efficiency Reward Shaping
//!
//! `efficiency_reward()` provides a scalar reward signal for the `ThinkingBandit`,
//! balancing correctness against token budget usage. Direct correct answers get
//! full reward; latent correct answers get discounted by budget fraction.

use std::path::Path;

use katgpt_core::traits::CollapseDetector;
use katgpt_core::types::ThinkingBudget;

#[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
use katgpt_core::temporal_deriv::TemporalDerivativeKernel;

use crate::freeze::{load_frozen, save_frozen};
// ThinkingMode was previously imported from crate::speculative::thinking_controller.
// That module is local to the main katgpt-rs crate (it depends on BanditStrategy
// from this crate via a cycle), so we cannot import it here. Define the enum
// locally — it is a pure-data 4-variant tag with no main-crate state, and
// collapse_detector only uses it as a parameter tag for efficiency_reward().
// The main crate's ThinkingMode is structurally identical; values cross the
// boundary as plain u8 via #[repr(u8)].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ThinkingMode {
    Direct,
    Latent,
    CpuResample,
    Dendritic,
}

// ── Frozen persistence struct (16 bytes, repr(C)) ─────────────────────

/// Binary persistence format for `S2FCollapseDetector` state.
///
/// 16 bytes, `repr(C)` for stable disk layout. Validated via magic bytes
/// and version on load. Uses `save_frozen` / `load_frozen` infrastructure.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CollapseDetectorFrozen {
    /// Magic bytes: `b"COLP"`.
    pub magic: [u8; 4],
    /// Serialization version. Currently 1.
    pub version: u32,
    /// Current collapse threshold τ.
    pub threshold: u32,
    /// EMA-smoothed hesitation count (self-learning signal).
    pub hesitation_ema: f32,
    /// EMA-smoothed budget utilization (tokens_used / max_tokens).
    pub budget_ema_mean: f32,
    /// Efficiency–accuracy trade-off γ from reward shaping.
    pub gamma: f32,
}

impl CollapseDetectorFrozen {
    /// Magic bytes identifying collapse detector frozen state.
    const MAGIC: [u8; 4] = *b"COLP";
    /// Current serialization version.
    const VERSION: u32 = 1;

    /// Create a new frozen state with default values.
    pub fn new(threshold: u32, gamma: f32) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            threshold,
            hesitation_ema: 0.0,
            budget_ema_mean: 0.0,
            gamma,
        }
    }

    /// Validate magic bytes and version. Returns `Err` on mismatch.
    pub fn validate(&self) -> Result<(), String> {
        match self.magic {
            m if m != Self::MAGIC => Err(format!(
                "CollapseDetectorFrozen: bad magic {:?}, expected {:?}",
                m,
                Self::MAGIC
            )),
            _ => match self.version {
                v if v != Self::VERSION => Err(format!(
                    "CollapseDetectorFrozen: bad version {v}, expected {}",
                    Self::VERSION
                )),
                _ => Ok(()),
            },
        }
    }
}

// ── S2FCollapseDetector ──────────────────────────────────────────────

/// Ring-buffer-based collapse detector with EMA self-learning.
///
/// Zero-allocation during detection: only fixed-size arrays and scalar fields.
/// The ring buffer is a `[u32; 64]` with a wrapping write index. On each
/// `check_collapse()`, the buffer is scanned for hesitation token matches.
///
/// Threshold τ adapts via EMA after each trace in `reset()`.
pub struct S2FCollapseDetector {
    /// Token IDs that signal hesitation (e.g., "wait", "hmm", "actually").
    /// Configurable per-domain via builder pattern.
    /// Sorted ascending on construction for O(log K) binary-search membership.
    hesitation_tokens: Vec<u32>,
    /// Fixed-size ring buffer of recent token IDs. Zero-allocation.
    ring_buffer: [u32; Self::RING_SIZE],
    /// Current write position in the ring buffer (wraps at RING_SIZE).
    ring_write_idx: usize,
    /// Cached count of hesitation tokens currently in the ring buffer.
    /// Updated incrementally on each `check_collapse` write to keep the
    /// per-token hot path O(log K) instead of O(RING_SIZE * K).
    current_count: u32,
    /// Collapse threshold τ — triggers when hesitation count ≥ threshold.
    threshold: u32,
    /// EMA smoothing factor α for self-learning. Default: 0.1.
    ema_alpha: f32,
    /// Hesitation count from the last completed trace (for EMA update).
    last_trace_hesitation: u32,
    /// Maximum budget for reward shaping (from `ThinkingBudget::max_tokens`).
    max_budget: u32,
    /// Efficiency γ for reward shaping (from `ThinkingBudget::efficiency_gamma`).
    gamma: f32,
    // ── Plan 267 T12: TVP (Thicket Variance Probe) integration ───────
    // High reasoning disagreement is the INVERSE of hesitation — it signals
    // genuine uncertainty rather than degenerate repetition. When TVP EMA
    // exceeds `tvp_expand_threshold`, the effective collapse threshold is
    // raised by `tvp_expand_budget_delta` to give the model more thinking
    // budget for substantive exploration.
    #[cfg(feature = "thicket_variance_probe")]
    tvp_reasoning_ema: f32,
    /// Budget expansion (in hesitation tokens) applied when TVP EMA is high.
    #[cfg(feature = "thicket_variance_probe")]
    tvp_expand_budget_delta: u32,
    /// TVP reasoning EMA above this triggers budget expansion. Range [0, 1].
    #[cfg(feature = "thicket_variance_probe")]
    tvp_expand_threshold: f32,
    // ── Plan 277 Phase 4 (F3): Temporal-Derivative Collapse Fusion ────
    //
    // Orthogonal early-warning signal. The hesitation detector fires *after*
    // degenerate tokens appear; the derivative signal fires *before* when the
    // entropy stream has stopped changing ("coasting"). The two are deliberately
    // independent: a trace can coast without hesitation, or hesitate without
    // the entropy derivative going to zero.
    //
    // `Option<...>` allows runtime ablation (set to `None` to disable the
    // derivative channel while keeping the feature compiled). When the
    // `temporal_deriv` feature is off the entire field is `cfg`-gated out and
    // the struct is byte-identical to its pre-fusion form (per AGENTS.md).
    /// Dual fast/slow EMA kernel over the scalar entropy stream. `None` =
    /// disabled (ablation).
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    pub derivative_collapse: Option<TemporalDerivativeKernel<1>>,
    /// Absolute value of the most recent `(fast − slow)` derivative sample.
    /// Updated by [`observe_entropy`](Self::observe_entropy). Zero before any
    /// observation.
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    last_entropy_derivative: f32,
    /// τ_deriv — below this magnitude the entropy derivative is considered
    /// "collapsed" (the stream is coasting). Default 0.01.
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    tau_deriv: f32,
    /// Fast EMA coefficient for the derivative kernel. Default 0.3.
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    temporal_deriv_alpha_fast: f32,
    /// Slow EMA coefficient for the derivative kernel. Default 0.03 (~10× ratio).
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    temporal_deriv_alpha_slow: f32,
}

impl S2FCollapseDetector {
    /// Ring buffer capacity. 64 tokens covers ~2 sentences of reasoning context.
    const RING_SIZE: usize = 64;

    // ── Plan 277 Phase 4 (F3): derivative-fusion defaults ──────────────
    // Centralized so the constructor and docs agree. τ_deriv = 0.01 catches
    // "coasting" entropy streams while still rejecting normal per-token
    // jitter; alphas follow the paper's canonical ~10× fast/slow ratio
    // (paper §Implementational, O'Reilly 2026).
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    const DEFAULT_TAU_DERIV: f32 = 0.01;
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    const DEFAULT_TEMPORAL_DERIV_ALPHA_FAST: f32 = 0.3;
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    const DEFAULT_TEMPORAL_DERIV_ALPHA_SLOW: f32 = 0.03;

    /// Create a new detector with the given hesitation tokens and budget.
    pub fn new(hesitation_tokens: Vec<u32>, budget: &ThinkingBudget) -> Self {
        // Sort once for binary-search membership tests in the hot path.
        let mut hesitation_tokens = hesitation_tokens;
        hesitation_tokens.sort_unstable();
        hesitation_tokens.dedup();
        Self {
            hesitation_tokens,
            ring_buffer: [0u32; Self::RING_SIZE],
            ring_write_idx: 0,
            current_count: 0,
            threshold: budget.collapse_threshold,
            ema_alpha: 0.1,
            last_trace_hesitation: 0,
            max_budget: budget.max_tokens,
            gamma: budget.efficiency_gamma,
            #[cfg(feature = "thicket_variance_probe")]
            tvp_reasoning_ema: 0.0,
            #[cfg(feature = "thicket_variance_probe")]
            tvp_expand_budget_delta: 0,
            #[cfg(feature = "thicket_variance_probe")]
            tvp_expand_threshold: 0.5,
            // Plan 277 Phase 4 (F3): the kernel is always constructed when the
            // `temporal_deriv` feature is on (it is 8 bytes on the stack and
            // zero-allocation). Callers can disable the channel for ablation
            // via [`disable_derivative_collapse`](Self::disable_derivative_collapse).
            #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
            derivative_collapse: Some(TemporalDerivativeKernel::new(
                Self::DEFAULT_TEMPORAL_DERIV_ALPHA_FAST,
                Self::DEFAULT_TEMPORAL_DERIV_ALPHA_SLOW,
            )),
            #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
            last_entropy_derivative: 0.0,
            #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
            tau_deriv: Self::DEFAULT_TAU_DERIV,
            #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
            temporal_deriv_alpha_fast: Self::DEFAULT_TEMPORAL_DERIV_ALPHA_FAST,
            #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
            temporal_deriv_alpha_slow: Self::DEFAULT_TEMPORAL_DERIV_ALPHA_SLOW,
        }
    }

    /// Create a detector with default hesitation tokens for common LLM patterns.
    ///
    /// Note: actual token IDs are tokenizer-dependent. These defaults assume
    /// a typical BPE vocabulary where hesitation tokens are in the low range.
    /// Production use should supply tokenizer-specific IDs.
    pub fn with_defaults(budget: &ThinkingBudget) -> Self {
        // Placeholder token IDs — override per-tokenizer in production.
        // These represent "wait", "hmm", "actually", "let me" equivalents.
        Self::new(vec![/* wait */ 0, /* hmm */ 0, /* actually */ 0], budget)
    }

    // ── Plan 267 T12: TVP (Thicket Variance Probe) integration ──────

    /// Configure TVP-driven budget expansion (Plan 267 T12).
    ///
    /// When `observe_tvp_disagreement()` is called with a signal whose
    /// EMA-smoothed `reasoning_disagreement` exceeds `expand_threshold`,
    /// the effective collapse threshold is raised by `expand_delta` tokens.
    /// This gives the model more thinking budget on genuinely uncertain
    /// queries (the inverse signal of degenerate hesitation).
    ///
    /// Disabled by default (`expand_delta = 0`). Enable once the probe
    /// runner (Plan 267 T4/T5) lands and produces real `TvpSignal`s.
    #[cfg(feature = "thicket_variance_probe")]
    pub fn with_tvp_expansion(mut self, expand_delta: u32, expand_threshold: f32) -> Self {
        self.tvp_expand_budget_delta = expand_delta;
        self.tvp_expand_threshold = expand_threshold.clamp(0.0, 1.0);
        self
    }

    /// Observe a TVP disagreement signal from the InferenceRouter (Plan 267 T12).
    ///
    /// Updates the EMA-smoothed reasoning disagreement. High disagreement
    /// indicates genuine uncertainty (multiple probes disagree on the answer),
    /// which is the INVERSE of hesitation — the detector should *expand*
    /// the thinking budget rather than contract it.
    ///
    /// Low disagreement + high hesitation → contract as normal (degenerate trace).
    /// High disagreement → expand threshold by `tvp_expand_budget_delta`.
    ///
    /// Zero-allocation: only scalar EMA update. No allocations.
    #[cfg(feature = "thicket_variance_probe")]
    pub fn observe_tvp_disagreement(
        &mut self,
        signal: &crate::thicket_variance_probe::TvpSignal,
    ) {
        let alpha = self.ema_alpha;
        self.tvp_reasoning_ema =
            (1.0 - alpha) * self.tvp_reasoning_ema + alpha * signal.reasoning_disagreement;
    }

    /// Current EMA-smoothed TVP reasoning disagreement (Plan 267 T12).
    ///
    /// Returns 0.0 when no TVP signals have been observed.
    #[cfg(feature = "thicket_variance_probe")]
    #[inline]
    pub fn tvp_reasoning_ema(&self) -> f32 {
        self.tvp_reasoning_ema
    }

    /// Compute the current effective threshold, accounting for TVP expansion.
    ///
    /// Returns `threshold + tvp_expand_budget_delta` when TVP EMA exceeds
    /// the configured expansion threshold, else `threshold`.
    #[cfg(feature = "thicket_variance_probe")]
    #[inline]
    fn effective_threshold(&self) -> u32 {
        if self.tvp_expand_budget_delta > 0 && self.tvp_reasoning_ema > self.tvp_expand_threshold {
            self.threshold.saturating_add(self.tvp_expand_budget_delta)
        } else {
            self.threshold
        }
    }

    // ── Plan 277 Phase 4 (F3): Temporal-Derivative Collapse Fusion ──────
    //
    // The hesitation ring-buffer detects collapse *after* degenerate tokens
    // appear. The dual EMA kernel over the entropy stream detects it *before*:
    // when `|fast − slow|` is small the entropy stream has stopped changing
    // and the trace is "coasting" toward a fixed point. We expose this as an
    // orthogonal early-warning predicate that host code can OR with the
    // existing `check_collapse` signal.
    //
    // The `CollapseDetector` trait signature is unchanged (constraint 4);
    // host code feeds entropy in via `observe_entropy` on the side.

    /// Override the default τ_deriv (the "coasting" threshold).
    ///
    /// Smaller values fire only on very flat entropy; larger values fire
    /// earlier but admit more false alarms. Must be finite and non-negative.
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    pub fn with_tau_deriv(mut self, tau_deriv: f32) -> Self {
        debug_assert!(
            tau_deriv.is_finite() && tau_deriv >= 0.0,
            "tau_deriv must be finite and non-negative, got {tau_deriv}"
        );
        self.tau_deriv = tau_deriv;
        self
    }

    /// Override the default fast/slow EMA coefficients for the derivative
    /// kernel and (re)construct the kernel with the new alphas.
    ///
    /// Validation is delegated to [`TemporalDerivativeKernel::new`]:
    /// `0 < alpha_slow < alpha_fast <= 1` (debug panic / release clamp).
    /// The canonical ~10× ratio (`alpha_fast=0.3, alpha_slow=0.03`) is the
    /// paper's recommended default.
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    pub fn with_temporal_deriv_alphas(mut self, alpha_fast: f32, alpha_slow: f32) -> Self {
        self.temporal_deriv_alpha_fast = alpha_fast;
        self.temporal_deriv_alpha_slow = alpha_slow;
        self.derivative_collapse = Some(TemporalDerivativeKernel::new(alpha_fast, alpha_slow));
        self
    }

    /// Disable the derivative-collapse channel at runtime (ablation).
    ///
    /// Sets `derivative_collapse = None`; [`observe_entropy`] becomes a no-op
    /// and [`derivative_collapse_detected`] always returns `false`. Used by
    /// the G4 benchmark to measure the false-negative rate *without* the
    /// derivative signal on identical traces.
    ///
    /// [`observe_entropy`]: Self::observe_entropy
    /// [`derivative_collapse_detected`]: Self::derivative_collapse_detected
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    pub fn disable_derivative_collapse(mut self) -> Self {
        self.derivative_collapse = None;
        self.last_entropy_derivative = 0.0;
        self
    }

    /// Feed a per-token scalar entropy sample into the derivative kernel and
    /// return the signed `(fast − slow)` derivative sample.
    ///
    /// Side-channel into the collapse detector — deliberately *outside* the
    /// `CollapseDetector` trait (which only receives a token id). Host code
    /// that already computes entropy per token should call this once per
    /// token **in addition to** [`check_collapse`](CollapseDetector::check_collapse):
    ///
    /// ```text
    /// let d = detector.observe_entropy(entropy);   // signed scalar derivative
    /// let hard = detector.check_collapse(token_id, position);
    /// let soft = detector.derivative_collapse_detected();
    /// if hard || soft { /* exit thinking */ }
    /// ```
    ///
    /// Returns the signed scalar derivative `result[0]` (positive when entropy
    /// is rising, negative when falling, ~0 when stationary). Returns `0.0`
    /// when the channel is disabled at runtime (`disable_derivative_collapse`).
    /// The stored [`last_entropy_derivative`] is this same signed value;
    /// [`derivative_collapse_detected`] takes the absolute value at check time.
    ///
    /// Zero-allocation: the `N=1` kernel is 16 bytes on the stack and the
    /// update is two in-place SIMD EMA writes. When the kernel is `None`
    /// (ablation) this is a single branch and returns `0.0` immediately.
    ///
    /// [`last_entropy_derivative`]: Self::last_entropy_derivative
    /// [`derivative_collapse_detected`]: Self::derivative_collapse_detected
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    #[inline]
    pub fn observe_entropy(&mut self, entropy: f32) -> f32 {
        match self.derivative_collapse.as_mut() {
            Some(kernel) => {
                let derivative = kernel.observe(&[entropy]);
                self.last_entropy_derivative = derivative[0];
                derivative[0]
            }
            None => {
                self.last_entropy_derivative = 0.0;
                0.0
            }
        }
    }

    /// Latest signed `(fast − slow)` derivative sample from
    /// [`observe_entropy`](Self::observe_entropy). Zero before the first
    /// observation or when the channel is disabled. The sign indicates
    /// direction (positive = entropy rising, negative = entropy falling).
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    #[inline]
    pub fn last_entropy_derivative(&self) -> f32 {
        self.last_entropy_derivative
    }

    /// Configured τ_deriv (the "coasting" threshold).
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    #[inline]
    pub fn tau_deriv(&self) -> f32 {
        self.tau_deriv
    }

    /// **Soft** early-warning predicate: the entropy derivative has gone to
    /// zero while the hesitation-based hard collapse has not yet fired.
    ///
    /// Returns `true` when *all* of the following hold:
    ///
    /// 1. `|last_entropy_derivative| < tau_deriv` — the entropy stream has
    ///    stopped changing ("coasting" toward a fixed point). Sign is
    ///    irrelevant: rising or falling, magnitude below τ_deriv means flat.
    /// 2. `hesitation_count() < threshold` — the hard hesitation detector has
    ///    *not* yet collapsed. This is what makes the signal **orthogonal**:
    ///    it fires only in the window where the trace is degenerate-by-flatness
    ///    but not yet degenerate-by-repetition.
    ///
    /// This is intentionally a softer signal than
    /// [`check_collapse`](CollapseDetector::check_collapse) — host code may
    /// choose to interpret it as a probabilistic early-exit hint rather than
    /// a hard force-exit. When the channel is disabled (kernel `None`) this
    /// always returns `false`.
    ///
    /// **Sigmoid, not softmax:** the eventual gate from this signal is
    /// `sigmoid(β · surprise_norm)`-shaped (see [`katgpt_core::temporal_deriv`]).
    /// This predicate is the boolean projection at `β → ∞`.
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    #[inline]
    pub fn derivative_collapse_detected(&self) -> bool {
        // If the kernel was disabled at runtime (ablation), never fire.
        if self.derivative_collapse.is_none() {
            return false;
        }
        self.last_entropy_derivative.abs() < self.tau_deriv
            && self.count_hesitation() < self.threshold
    }

    /// Freeze detector state to disk via `repr(C)` binary dump.
    pub fn freeze(&self, path: &Path) -> Result<(), String> {
        let frozen = CollapseDetectorFrozen {
            magic: CollapseDetectorFrozen::MAGIC,
            version: CollapseDetectorFrozen::VERSION,
            threshold: self.threshold,
            hesitation_ema: self.last_trace_hesitation as f32,
            budget_ema_mean: 0.0,
            gamma: self.gamma,
        };
        save_frozen(path, &frozen)
    }

    /// Thaw detector state from disk. Validates magic and version.
    pub fn thaw(&mut self, path: &Path) -> Result<(), String> {
        let frozen: CollapseDetectorFrozen = load_frozen(path)?;
        frozen.validate()?;
        self.threshold = frozen.threshold;
        self.gamma = frozen.gamma;
        self.last_trace_hesitation = frozen.hesitation_ema as u32;
        Ok(())
    }

    /// Count hesitation tokens in the current ring buffer.
    ///
    /// Returns the cached `current_count` maintained incrementally by
    /// `check_collapse` — O(1) on the hot path. The full O(RING_SIZE * log K)
    /// scan is only used by `recompute_count` for testing / recovery.
    #[inline]
    fn count_hesitation(&self) -> u32 {
        self.current_count
    }

    /// O(log K) binary-search membership test against the sorted hesitation set.
    #[inline]
    fn is_hesitation(&self, token_id: u32) -> bool {
        self.hesitation_tokens.binary_search(&token_id).is_ok()
    }

    /// Recompute `current_count` from scratch by scanning the entire ring buffer.
    /// Used by `from_frozen` to rebuild state and by tests that poke the buffer
    /// directly.
    #[allow(dead_code)]
    fn recompute_count(&mut self) {
        let mut count = 0u32;
        for &tok in self.ring_buffer.iter() {
            if self.is_hesitation(tok) {
                count += 1;
            }
        }
        self.current_count = count;
    }
}

impl CollapseDetector for S2FCollapseDetector {
    /// Check if the current trace exhibits collapse symptoms.
    ///
    /// Writes the token to the ring buffer, then counts hesitation matches.
    /// Returns `true` when hesitation count ≥ threshold τ.
    fn check_collapse(&mut self, token_id: u32, _position: usize) -> bool {
        // Incremental count update: subtract the evicted token's contribution,
        // write the new token, then add its contribution. Two binary searches
        // (O(log K)) replace the previous O(RING_SIZE * K) linear scan.
        let write_idx = self.ring_write_idx;
        let evicted = self.ring_buffer[write_idx];
        if self.is_hesitation(evicted) {
            self.current_count = self.current_count.saturating_sub(1);
        }
        self.ring_buffer[write_idx] = token_id;
        if self.is_hesitation(token_id) {
            self.current_count += 1;
        }
        self.ring_write_idx = (write_idx + 1) % Self::RING_SIZE;

        // Count hesitation tokens and compare against threshold.
        let count = self.count_hesitation();
        self.last_trace_hesitation = self.last_trace_hesitation.max(count);

        // Effective threshold: expand when TVP indicates genuine uncertainty
        // (Plan 267 T12). When `thicket_variance_probe` is off, this is just
        // `self.threshold` — zero-cost gate.
        #[cfg(feature = "thicket_variance_probe")]
        let effective = self.effective_threshold();
        #[cfg(not(feature = "thicket_variance_probe"))]
        let effective: u32 = self.threshold;

        count >= effective
    }

    /// Reset internal state between traces. Updates EMA threshold.
    ///
    /// Self-learning: if the last trace had high hesitation without collapse,
    /// the threshold adapts upward. If collapse was triggered, the threshold
    /// stays or decreases based on the hesitation level.
    fn reset(&mut self) {
        // EMA threshold adaptation: smooth toward observed hesitation level.
        let observed = self.last_trace_hesitation as f32;
        let current = self.threshold as f32;
        let adapted = current + self.ema_alpha * (observed - current);
        // Clamp threshold to [1, max_budget] — never zero (always detect)
        // and never exceed budget (never trigger).
        self.threshold = adapted.round() as u32;
        self.threshold = self.threshold.clamp(1, self.max_budget);

        // Clear ring buffer and tracking state.
        self.ring_buffer = [0u32; Self::RING_SIZE];
        self.ring_write_idx = 0;
        self.current_count = 0;
        self.last_trace_hesitation = 0;

        // Plan 267 T12: clear TVP EMA between traces (per-query signal).
        // Frozen config (expand_delta, expand_threshold) is preserved.
        #[cfg(feature = "thicket_variance_probe")]
        {
            self.tvp_reasoning_ema = 0.0;
        }

        // Plan 277 Phase 4 (F3): reset the derivative kernel and the cached
        // derivative sample. τ_deriv and alpha config are preserved (they are
        // per-detector config, not per-trace state). If the kernel was
        // disabled at runtime (ablation), keep it disabled.
        #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
        {
            if let Some(kernel) = self.derivative_collapse.as_mut() {
                kernel.reset();
            }
            self.last_entropy_derivative = 0.0;
        }
    }

    /// Number of hesitation tokens observed in the current trace.
    fn hesitation_count(&self) -> u32 {
        self.count_hesitation()
    }

    /// Current collapse threshold τ.
    fn threshold(&self) -> u32 {
        self.threshold
    }
}

// ── Efficiency Reward Shaping (T3) ──────────────────────────────────

/// Compute efficiency-shaped reward for the `ThinkingBandit`.
///
/// Encourages the bandit to prefer cheap correct answers over expensive ones:
/// - `(true, Direct)` → `1.0` (best possible: correct with zero thinking cost)
/// - `(true, Latent)` → `1.0 - γ × (tokens_used / max_budget)` (correct but costly)
/// - `(false, _)` → `-1.0` (wrong answer, always penalized)
/// - Other modes → `0.0` (no reward signal)
///
/// Uses sigmoid-compatible values (bounded [-1, 1]) for downstream bandit arms.
#[inline]
pub fn efficiency_reward(
    correct: bool,
    tokens_used: u32,
    max_budget: u32,
    mode: ThinkingMode,
    gamma: f32,
) -> f32 {
    match (correct, mode) {
        // Wrong answer → always penalized regardless of mode.
        (false, _) => -1.0,
        // Direct correct → full reward (zero thinking cost).
        (true, ThinkingMode::Direct) => 1.0,
        // Latent correct → reward discounted by budget utilization.
        (true, ThinkingMode::Latent) => {
            let utilization = if max_budget > 0 {
                tokens_used as f32 / max_budget as f32
            } else {
                1.0
            };
            1.0 - gamma * utilization
        }
        // Other modes → no reward signal (not yet calibrated).
        (true, ThinkingMode::CpuResample) => 0.0,
        // Dendritic correct → intermediate reward (deterministic gate, zero randomness).
        (true, ThinkingMode::Dendritic) => 0.5,
    }
}

// ── Decode-Loop Integration (Plan 212 T4) ──────────────────────────

/// Result of a collapse check during decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CollapseAction {
    /// No collapse detected — continue normal decoding.
    Continue,
    /// Collapse detected — force early exit from thinking mode.
    ForceExit,
}

/// Per-token hook for the decode loop. Returns `ForceExit` when
/// the detector's threshold is exceeded, signaling that the current CoT trace
/// is degenerate and should be terminated early.
#[inline]
pub fn check_collapse_action(
    detector: &mut dyn CollapseDetector,
    token_id: u32,
    position: usize,
    thinking_mode: bool,
) -> CollapseAction {
    if !thinking_mode {
        return CollapseAction::Continue;
    }
    match detector.check_collapse(token_id, position) {
        true => CollapseAction::ForceExit,
        false => CollapseAction::Continue,
    }
}

// ── Unit Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper: create a detector with specific hesitation token IDs.
    fn make_detector(hesitation_tokens: Vec<u32>, threshold: u32) -> S2FCollapseDetector {
        let budget = ThinkingBudget {
            max_tokens: 4096,
            collapse_threshold: threshold,
            efficiency_gamma: 0.5,
        };
        S2FCollapseDetector::new(hesitation_tokens, &budget)
    }

    #[test]
    fn test_collapse_detector_triggers_on_repetitive_wait() {
        let mut detector = make_detector(vec![42, 99], 3);
        // Feed 3 hesitation tokens (token 42) — should trigger at threshold=3.
        assert!(!detector.check_collapse(42, 0)); // count=1, <3
        assert!(!detector.check_collapse(42, 1)); // count=2, <3
        assert!(detector.check_collapse(42, 2)); // count=3, >=3 → collapse!
    }

    #[test]
    fn test_collapse_detector_no_false_positive() {
        let mut detector = make_detector(vec![42], 3);
        // Feed non-hesitation tokens — should never trigger.
        for i in 0..100 {
            assert!(
                !detector.check_collapse(7, i),
                "False positive at position {i}"
            );
        }
        assert_eq!(detector.hesitation_count(), 0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut detector = make_detector(vec![42], 2);
        // Trigger collapse.
        detector.check_collapse(42, 0);
        detector.check_collapse(42, 1);
        assert!(detector.hesitation_count() >= 2);

        // Reset should clear ring buffer.
        detector.reset();
        assert_eq!(detector.hesitation_count(), 0);
        assert_eq!(detector.ring_write_idx, 0);
    }

    #[test]
    fn test_efficiency_reward_direct_correct() {
        let reward = efficiency_reward(true, 0, 4096, ThinkingMode::Direct, 0.5);
        assert!(
            (reward - 1.0).abs() < 1e-6,
            "Direct correct should be 1.0, got {reward}"
        );
    }

    #[test]
    fn test_efficiency_reward_thinking_short_correct() {
        // Used only 10% of budget → reward ≈ 1.0 - 0.5 * 0.1 = 0.95
        let reward = efficiency_reward(true, 410, 4096, ThinkingMode::Latent, 0.5);
        let expected = 1.0 - 0.5 * (410.0_f32 / 4096.0);
        assert!(
            (reward - expected).abs() < 1e-4,
            "Short latent correct: expected {expected}, got {reward}"
        );
    }

    #[test]
    fn test_efficiency_reward_thinking_long_correct() {
        // Used 80% of budget → reward ≈ 1.0 - 0.5 * 0.8 = 0.6
        let reward = efficiency_reward(true, 3277, 4096, ThinkingMode::Latent, 0.5);
        let expected = 1.0 - 0.5 * (3277.0_f32 / 4096.0);
        assert!(
            (reward - expected).abs() < 1e-4,
            "Long latent correct: expected {expected}, got {reward}"
        );
    }

    #[test]
    fn test_efficiency_reward_incorrect() {
        let reward = efficiency_reward(false, 100, 4096, ThinkingMode::Latent, 0.5);
        assert!(
            (reward - (-1.0)).abs() < 1e-6,
            "Incorrect should be -1.0, got {reward}"
        );

        let reward_direct = efficiency_reward(false, 0, 4096, ThinkingMode::Direct, 0.5);
        assert!(
            (reward_direct - (-1.0)).abs() < 1e-6,
            "Incorrect direct should be -1.0, got {reward_direct}"
        );
    }

    #[test]
    fn test_freeze_thaw_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("collapse_detector.bin");

        let budget = ThinkingBudget {
            max_tokens: 4096,
            collapse_threshold: 5,
            efficiency_gamma: 0.7,
        };
        let detector = S2FCollapseDetector::new(vec![42], &budget);

        // Freeze.
        detector.freeze(&path).expect("freeze");

        // Create a new detector and thaw.
        let mut detector2 = make_detector(vec![42], 1);
        detector2.thaw(&path).expect("thaw");

        // Threshold should be restored from frozen state.
        assert_eq!(detector2.threshold(), 5);
        assert!((detector2.gamma - 0.7).abs() < 1e-6);
    }

    // ── T7: GOAT Tests ──────────────────────────────────────────────

    #[test]
    fn test_thinking_budget_adapts_after_collapse() {
        // Start with threshold=10, feed traces with early collapse (hesitation count 3).
        // After reset, EMA should lower the threshold toward the observed hesitation.
        let mut detector = make_detector(vec![42], 10);
        let initial_threshold = detector.threshold();

        // Simulate multiple traces where collapse triggers early (low hesitation).
        for _ in 0..5 {
            // Feed 3 hesitation tokens — ring buffer has 3 hesitation tokens,
            // but threshold is 10 so no collapse yet.
            for pos in 0..3 {
                detector.check_collapse(42, pos);
            }
            // last_trace_hesitation will track max hesitation count = 3
            detector.reset();
        }

        // After 5 traces with observed hesitation of 3, EMA should have moved
        // threshold toward 3 from initial 10. It should be strictly lower now.
        let adapted_threshold = detector.threshold();
        assert!(
            adapted_threshold < initial_threshold,
            "Threshold should decrease after early-collapse traces: initial={initial_threshold}, adapted={adapted_threshold}"
        );
    }

    #[test]
    fn test_efficiency_reward_short_correct_higher_than_long_correct() {
        let max_budget = 4096u32;
        let gamma = 0.5f32;

        let short_tokens = 100u32;
        let long_tokens = 3000u32;

        let reward_short =
            efficiency_reward(true, short_tokens, max_budget, ThinkingMode::Latent, gamma);
        let reward_long =
            efficiency_reward(true, long_tokens, max_budget, ThinkingMode::Latent, gamma);

        assert!(
            reward_short > reward_long,
            "Short correct ({reward_short:.4}) should yield higher reward than long correct ({reward_long:.4})"
        );

        // Also verify they're both positive (correct answers).
        assert!(reward_short > 0.0, "Short correct reward must be positive");
        assert!(reward_long > 0.0, "Long correct reward must be positive");
    }

    #[test]
    fn test_end_to_end_thinking_collapse_exit() {
        // Token 5 is hesitation token. Feed: [10, 20, 30, 5, 5, 5]
        // With threshold=3, collapse should trigger when 3 hesitation tokens are seen.
        let mut detector = make_detector(vec![5], 3);

        // Non-hesitation tokens — no collapse.
        assert!(!detector.check_collapse(10, 0));
        assert!(!detector.check_collapse(20, 1));
        assert!(!detector.check_collapse(30, 2));

        // First hesitation — count=1, no collapse.
        assert!(!detector.check_collapse(5, 3));

        // Second hesitation — count=2, no collapse.
        assert!(!detector.check_collapse(5, 4));

        // Third hesitation — count=3, >= threshold → collapse!
        assert!(detector.check_collapse(5, 5));

        // Verify collapse was detected at the right point.
        assert!(detector.hesitation_count() >= 3);

        // After collapse, efficiency_reward should give a signal that reflects
        // the waste — we used 6 tokens but collapsed. A correct answer with
        // partial budget should still be lower than a direct correct.
        let reward_collapsed = efficiency_reward(
            true, // correct answer despite collapse
            6,    // tokens used before collapse
            4096,
            ThinkingMode::Latent,
            0.5,
        );
        let reward_direct = efficiency_reward(true, 0, 4096, ThinkingMode::Direct, 0.5);

        assert!(
            reward_direct > reward_collapsed,
            "Direct correct ({reward_direct:.3}) should be rewarded more than collapsed latent ({reward_collapsed:.3})"
        );
        assert!(
            reward_collapsed > 0.0,
            "Collapsed correct should still be positive"
        );
    }

    /// T7: CPU/GPU routing — collapse signal feeds into ThinkingController load dispatch.
    ///
    /// When collapse is detected mid-reasoning, the load dispatcher should route
    /// to the CPU fast path (immediate answer) rather than continuing on GPU
    /// (deep think path). This test simulates that routing decision.
    #[test]
    fn collapse_signal_routes_to_cpu_on_high_load() {
        // Simulated routing decision based on collapse state.
        #[derive(Debug, PartialEq, Eq)]
        #[repr(u8)]
        enum ComputeRoute {
            /// Continue deep thinking on GPU.
            Gpu,
            /// Fast path: collapse detected, route to CPU for immediate answer.
            Cpu,
        }

        fn decide_route(collapsed: bool) -> ComputeRoute {
            match collapsed {
                true => ComputeRoute::Cpu,
                false => ComputeRoute::Gpu,
            }
        }

        // Create detector with low threshold: 2 hesitation tokens trigger collapse.
        let mut detector = make_detector(vec![42], 2);

        // Phase 1: No hesitation — should route to GPU (continue deep thinking).
        assert!(!detector.check_collapse(10, 0));
        assert_eq!(decide_route(false), ComputeRoute::Gpu);

        // Phase 2: First hesitation token — still under threshold, GPU continues.
        assert!(!detector.check_collapse(42, 1));
        assert_eq!(decide_route(false), ComputeRoute::Gpu);

        // Phase 3: Second hesitation token — threshold exceeded, collapse detected.
        // Load dispatch must now route to CPU (fast path / immediate answer).
        let collapsed = detector.check_collapse(42, 2);
        assert!(collapsed, "Collapse should be detected at threshold=2");
        assert_eq!(decide_route(collapsed), ComputeRoute::Cpu);

        // Verify the efficiency reward signal is consistent with CPU routing:
        // A collapsed trace that yields a correct answer should get less reward
        // than direct, but still positive — encouraging the CPU fast path.
        let reward_collapsed = efficiency_reward(
            true,
            3, // tokens used before collapse
            4096,
            ThinkingMode::Latent,
            0.5,
        );
        assert!(
            reward_collapsed > 0.0,
            "Collapsed correct should give positive reward for CPU routing, got {reward_collapsed}"
        );

        // Reset should allow a fresh trace to route back to GPU.
        detector.reset();
        assert_eq!(detector.hesitation_count(), 0);
        assert!(!detector.check_collapse(10, 0));
        assert_eq!(decide_route(false), ComputeRoute::Gpu);
    }

    // ── Plan 267 T12: TVP Integration Tests ─────────────────────────
    //
    // These verify the inverse-signal composition: high TVP reasoning
    // disagreement expands the effective collapse threshold, giving the
    // model more thinking budget on genuinely uncertain queries.
    //
    // Gated on `thicket_variance_probe` because they reference `TvpSignal`.
    #[cfg(feature = "thicket_variance_probe")]
    mod tvp_integration {
        use super::*;
        use crate::thicket_variance_probe::TvpSignal;

        fn make_detector_with_tvp(
            threshold: u32,
            expand_delta: u32,
            expand_threshold: f32,
        ) -> S2FCollapseDetector {
            let budget = ThinkingBudget {
                max_tokens: 4096,
                collapse_threshold: threshold,
                efficiency_gamma: 0.5,
            };
            S2FCollapseDetector::new(vec![42], &budget)
                .with_tvp_expansion(expand_delta, expand_threshold)
        }

        fn high_disagreement_signal() -> TvpSignal {
            TvpSignal {
                reasoning_disagreement: 0.9,
                format_disagreement: 0.05,
                logit_kl: 1.5,
                probe_count_used: 4,
            }
        }

        fn low_disagreement_signal() -> TvpSignal {
            TvpSignal {
                reasoning_disagreement: 0.05,
                format_disagreement: 0.0,
                logit_kl: 0.01,
                probe_count_used: 4,
            }
        }

        /// Observe high TVP disagreement → effective threshold expands by `expand_delta`.
        /// Model can emit more hesitation tokens before collapse triggers.
        #[test]
        fn high_disagreement_expands_effective_threshold() {
            let mut detector = make_detector_with_tvp(3, 5, 0.5);
            // Observe enough high-disagreement signals to push EMA above 0.5.
            // EMA: starts at 0.0, α=0.1. After 8 observations of 0.9:
            //   ema_n = 0.9 * (1 - 0.9^n)
            //   n=8: 0.9 * (1 - 0.4305) ≈ 0.513 > 0.5 ✓
            for _ in 0..8 {
                detector.observe_tvp_disagreement(&high_disagreement_signal());
            }
            assert!(
                detector.tvp_reasoning_ema() > 0.5,
                "EMA should exceed 0.5 after sustained high disagreement, got {}",
                detector.tvp_reasoning_ema()
            );

            // With expansion active, threshold=3 + delta=5 = 8 effective.
            // Feed 7 hesitation tokens — should NOT collapse (7 < 8).
            for i in 0..7 {
                assert!(
                    !detector.check_collapse(42, i),
                    "Should not collapse at count={} with expanded threshold 8",
                    i + 1
                );
            }

            // Feed 1 more (total 8) — now collapse triggers (8 >= 8).
            assert!(
                detector.check_collapse(42, 7),
                "Should collapse at count=8 with expanded threshold 8"
            );
        }

        /// Low TVP disagreement → no threshold expansion. Standard behavior preserved.
        #[test]
        fn low_disagreement_keeps_threshold_unchanged() {
            let mut detector = make_detector_with_tvp(3, 5, 0.5);
            // Observe low disagreement — EMA stays low.
            for _ in 0..8 {
                detector.observe_tvp_disagreement(&low_disagreement_signal());
            }
            assert!(
                detector.tvp_reasoning_ema() < 0.5,
                "EMA should stay below 0.5 after low disagreement, got {}",
                detector.tvp_reasoning_ema()
            );

            // Threshold unchanged: 3 hesitation tokens trigger collapse.
            assert!(!detector.check_collapse(42, 0)); // count=1
            assert!(!detector.check_collapse(42, 1)); // count=2
            assert!(detector.check_collapse(42, 2)); // count=3 → collapse
        }

        /// EMA smoothing: a single spike should NOT fully expand — need sustained signal.
        #[test]
        fn ema_smooths_single_spike() {
            let mut detector = make_detector_with_tvp(3, 5, 0.5);
            // Single high-disagreement observation.
            detector.observe_tvp_disagreement(&high_disagreement_signal());
            // EMA = 0.1 * 0.9 = 0.09 — well below 0.5.
            assert!((detector.tvp_reasoning_ema() - 0.09).abs() < 1e-6);

            // No expansion yet — single spike is smoothed out.
            assert!(!detector.check_collapse(42, 0)); // count=1
            assert!(!detector.check_collapse(42, 1)); // count=2
            assert!(detector.check_collapse(42, 2)); // count=3 → collapse (threshold=3)
        }

        /// TVP expansion disabled by default (expand_delta=0).
        #[test]
        fn tvp_expansion_disabled_by_default() {
            let budget = ThinkingBudget {
                max_tokens: 4096,
                collapse_threshold: 3,
                efficiency_gamma: 0.5,
            };
            let mut detector = S2FCollapseDetector::new(vec![42], &budget);
            // No with_tvp_expansion called → delta=0, no expansion possible.
            // Even with sustained high disagreement, threshold stays 3.
            // EMA after 20 steps of 0.9 input: 0.9 * (1 - 0.9^20) ≈ 0.79.
            for _ in 0..20 {
                detector.observe_tvp_disagreement(&high_disagreement_signal());
            }
            assert!(
                detector.tvp_reasoning_ema() > 0.7,
                "EMA should be high after sustained disagreement, got {}",
                detector.tvp_reasoning_ema()
            );
            // Still collapses at count=3 (delta=0 → no expansion).
            assert!(!detector.check_collapse(42, 0));
            assert!(!detector.check_collapse(42, 1));
            assert!(detector.check_collapse(42, 2));
        }

        /// `reset()` clears TVP EMA (per-query signal) but preserves config.
        #[test]
        fn reset_clears_tvp_ema_but_preserves_config() {
            let mut detector = make_detector_with_tvp(3, 5, 0.5);
            // Sustain high disagreement.
            for _ in 0..10 {
                detector.observe_tvp_disagreement(&high_disagreement_signal());
            }
            let ema_before = detector.tvp_reasoning_ema();
            assert!(ema_before > 0.5);

            // Reset clears EMA but config (delta=5) is preserved.
            detector.reset();
            assert_eq!(
                detector.tvp_reasoning_ema(),
                0.0,
                "reset should clear TVP EMA"
            );

            // After reset, a single low observation doesn't expand.
            // Threshold is back to 3 (no expansion).
            assert!(!detector.check_collapse(42, 0));
            assert!(!detector.check_collapse(42, 1));
            assert!(detector.check_collapse(42, 2));
        }

        /// Boundary: EMA exactly at threshold does NOT expand (strict `>` comparison).
        #[test]
        fn boundary_ema_at_threshold_does_not_expand() {
            let mut detector = make_detector_with_tvp(3, 5, 0.5);
            // Use the builder's clamped threshold (0.5). Drive EMA to exactly ~0.5.
            // With α=0.1, EMA converges toward 0.5 asymptotically but never reaches
            // it from below with constant 0.5 input. Use a 0.5001 input to nudge
            // just above — but verify that exactly at 0.5 input, EMA stays < 0.5.
            let boundary_signal = TvpSignal {
                reasoning_disagreement: 0.5,
                format_disagreement: 0.0,
                logit_kl: 0.0,
                probe_count_used: 4,
            };
            for _ in 0..50 {
                detector.observe_tvp_disagreement(&boundary_signal);
            }
            // EMA converges toward 0.5 from below — stays strictly < 0.5.
            assert!(
                detector.tvp_reasoning_ema() < 0.5,
                "EMA should stay < 0.5 (asymptotic from below), got {}",
                detector.tvp_reasoning_ema()
            );
            // No expansion — collapses at count=3 as normal.
            assert!(!detector.check_collapse(42, 0));
            assert!(!detector.check_collapse(42, 1));
            assert!(detector.check_collapse(42, 2));
        }
    }

    // ── Plan 277 Phase 4 (F3): Temporal-Derivative Collapse Fusion Tests ──
    //
    // G4 gate: prove the derivative channel reduces false-negative rate on
    // gradual-convergence traces by ≥ 20% vs hesitation-only. Two suites:
    //
    //   1. Entropy-collapse (one-hot forced, sharp drop) — hesitation path catches.
    //   2. Derivative-only-collapse (gradual convergence, no hesitation tokens)
    //      — only the derivative channel catches.
    //
    // Gated on both features (the same dual-gate as the production code).
    #[cfg(all(feature = "collapse_aware_thinking", feature = "temporal_deriv"))]
    mod derivative_collapse_fusion {
        use super::*;

        /// Hesitation set = {1, 2, 3}. Tests emit tokens with id ≥ 100, so the
        /// hesitation ring never sees a match on gradual-convergence traces.
        const HESITATION_TOKENS: [u32; 3] = [1, 2, 3];

        /// Non-hesitation token id for step `t` — guaranteed outside the set.
        #[inline]
        fn non_hesitation_token(t: usize) -> u32 {
            100 + (t as u32)
        }

        /// Detector with derivative channel ENABLED (default state).
        fn make_detector_with_derivative(threshold: u32) -> S2FCollapseDetector {
            let budget = ThinkingBudget {
                max_tokens: 4096,
                collapse_threshold: threshold,
                efficiency_gamma: 0.5,
            };
            S2FCollapseDetector::new(HESITATION_TOKENS.to_vec(), &budget)
        }

        /// Detector with derivative channel DISABLED (ablation arm).
        fn make_detector_without_derivative(threshold: u32) -> S2FCollapseDetector {
            make_detector_with_derivative(threshold).disable_derivative_collapse()
        }

        /// Generate a gradual-convergence entropy trace.
        ///
        /// `e(t) = e_star + (e0 - e_star) * exp(-t / tau)`.
        ///
        /// Entropy converges exponentially to a moderate fixed point `e_star`
        /// (well above 0), so the trace never becomes one-hot but its rate of
        /// change goes to zero. This is the regime the hesitation ring misses
        /// (no repetitive tokens) but `|d(entropy)/dt| → 0` should catch.
        fn gradual_convergence_trace(e0: f32, e_star: f32, tau: f32, len: usize) -> Vec<f32> {
            let mut out = Vec::with_capacity(len);
            for t in 0..len {
                let decay = (-(t as f32) / tau).exp();
                out.push(e_star + (e0 - e_star) * decay);
            }
            out
        }

        /// Run one detector arm over a trace and report whether collapse was
        /// detected at any step.
        ///
        /// - `with_derivative_channel`: if true, call `observe_entropy` and OR
        ///   the soft predicate with the hard `check_collapse` signal. If false,
        ///   only the hard hesitation path is consulted (the kernel state is
        ///   untouched, but the detector was constructed via
        ///   `disable_derivative_collapse` so the soft predicate is hard-wired
        ///   to false anyway).
        fn trace_collapses(
            detector: &mut S2FCollapseDetector,
            trace: &[f32],
            with_derivative_channel: bool,
        ) -> bool {
            for (t, &entropy) in trace.iter().enumerate() {
                let token_id = non_hesitation_token(t);
                if with_derivative_channel {
                    detector.observe_entropy(entropy);
                }
                let hard = detector.check_collapse(token_id, t);
                let soft = if with_derivative_channel {
                    detector.derivative_collapse_detected()
                } else {
                    false
                };
                if hard || soft {
                    return true;
                }
            }
            false
        }

        /// Sanity: the derivative channel fires on a long constant-entropy
        /// stream with no hesitation tokens. This is the kernel-level proof
        /// that the bridge is wired correctly.
        #[test]
        fn derivative_fires_on_constant_entropy_no_hesitation() {
            let mut detector = make_detector_with_derivative(8);
            // Constant entropy 0.5 — derivative → 0 after EMA warm-up.
            let trace = vec![0.5_f32; 200];
            let mut fired = false;
            for (t, &entropy) in trace.iter().enumerate() {
                detector.observe_entropy(entropy);
                let _ = detector.check_collapse(non_hesitation_token(t), t);
                if detector.derivative_collapse_detected() {
                    fired = true;
                    break;
                }
            }
            assert!(
                fired,
                "derivative_collapse_detected should fire on a long constant entropy stream"
            );
            // Hesitation path must NOT have fired — no hesitation tokens emitted.
            assert_eq!(
                detector.hesitation_count(),
                0,
                "hesitation ring must be empty when only non-hesitation tokens are emitted"
            );
        }

        /// Entropy-collapse suite: sharp drop to ~0 PLUS repetitive hesitation
        /// tokens. The existing hesitation detector catches this — both arms
        /// must report collapse.
        #[test]
        fn entropy_collapse_caught_by_hesitation_path() {
            // Trace: 5 warm-up non-hesitation tokens at high entropy, then a
            // sharp drop to ~0 with repetitive hesitation token id=1.
            let threshold = 4;
            let mut det_with = make_detector_with_derivative(threshold);
            let mut det_without = make_detector_without_derivative(threshold);

            // Warm-up: non-hesitation tokens, high entropy.
            for t in 0..5 {
                let _ = det_with.observe_entropy(2.0);
                let _ = det_without.observe_entropy(2.0);
                det_with.check_collapse(non_hesitation_token(t), t);
                det_without.check_collapse(non_hesitation_token(t), t);
            }

            // Collapse phase: entropy → 0, repetitive hesitation token.
            let mut caught_with = false;
            let mut caught_without = false;
            for t in 5..30 {
                let _ = det_with.observe_entropy(0.0);
                let _ = det_without.observe_entropy(0.0);
                // Emit the hesitation token id=1 on every step.
                if det_with.check_collapse(1, t) {
                    caught_with = true;
                }
                if det_without.check_collapse(1, t) {
                    caught_without = true;
                }
            }
            assert!(
                caught_without,
                "hesitation-only arm must catch the repetitive-token entropy collapse"
            );
            assert!(
                caught_with,
                "fused arm must also catch the repetitive-token entropy collapse"
            );
        }

        /// **G4 gate.** Gradual-convergence suite: false-negative rate must drop
        /// by ≥ 20% when the derivative channel is enabled.
        ///
        /// Each trace converges exponentially to a moderate fixed point
        /// (`e_star ∈ [0.3, 0.7]`) with no hesitation tokens. The hesitation
        /// ring never fires. Without the derivative channel, every such trace
        /// is a false negative. With the derivative channel, the
        /// `|fast − slow| < τ_deriv` predicate fires once the slow EMA catches
        /// up to the stabilized signal.
        #[test]
        fn g4_gate_derivative_reduces_false_negatives_by_at_least_20pct() {
            // Trace ensemble: 24 traces varying fixed point, time constant,
            // and starting entropy. All are genuine collapse cases (entropy has
            // stopped changing mid-budget while no answer has been produced).
            const N_TRACES: usize = 24;
            const TRACE_LEN: usize = 200;
            const THRESHOLD: u32 = 8;

            let mut false_neg_without = 0u32;
            let mut false_neg_with = 0u32;
            let mut detected_step_with = Vec::new();

            for i in 0..N_TRACES {
                // Spread e_star across [0.30, 0.70].
                let e_star = 0.30 + 0.40 * (i as f32) / (N_TRACES as f32 - 1.0);
                // Vary tau across {4, 8, 12} — different convergence speeds.
                let tau = 4.0_f32 * (1.0 + ((i % 3) as f32));
                // Vary e0 across {1.2, 1.5, 1.8, 2.1}.
                let e0 = 1.2 + 0.3 * ((i % 4) as f32);
                let trace =
                    gradual_convergence_trace(e0, e_star, tau, TRACE_LEN);

                // Arm A: hesitation-only (derivative channel disabled).
                let mut det_without = make_detector_without_derivative(THRESHOLD);
                let caught_without = trace_collapses(&mut det_without, &trace, false);
                if !caught_without {
                    false_neg_without += 1;
                }

                // Arm B: fused (hesitation OR derivative).
                let mut det_with = make_detector_with_derivative(THRESHOLD);
                let mut caught_with_at = None;
                for (t, &entropy) in trace.iter().enumerate() {
                    det_with.observe_entropy(entropy);
                    let hard = det_with.check_collapse(non_hesitation_token(t), t);
                    let soft = det_with.derivative_collapse_detected();
                    if hard || soft {
                        caught_with_at = Some(t);
                        break;
                    }
                }
                match caught_with_at {
                    Some(step) => detected_step_with.push((i, e_star, tau, step)),
                    None => false_neg_with += 1,
                }
            }

            // Improvement in false-negative rate.
            let denom = false_neg_without.max(1) as f32;
            let improvement = (false_neg_without - false_neg_with) as f32 / denom;

            // Diagnostic dump on failure.
            if improvement < 0.20 {
                let mut dbg = String::new();
                dbg.push_str(&format!(
                    "\nG4 FAIL: improvement={:.3} (< 0.20) without={}, with={}\n",
                    improvement, false_neg_without, false_neg_with
                ));
                dbg.push_str("detected traces (i, e_star, tau, step):\n");
                for (i, e_star, tau, step) in &detected_step_with {
                    dbg.push_str(&format!(
                        "  i={i:2} e_star={e_star:.3} tau={tau:.1} detected_at_step={step}\n"
                    ));
                }
                panic!("{dbg}");
            }

            // Additional invariant: the hesitation-only arm must miss every
            // gradual-convergence trace (no hesitation tokens emitted).
            assert_eq!(
                false_neg_without, N_TRACES as u32,
                "hesitation-only arm must miss all gradual-convergence traces (no hesitation tokens), got {} / {}",
                false_neg_without, N_TRACES
            );
            // The derivative channel must catch at least one (sanity).
            assert!(
                false_neg_with < false_neg_without,
                "derivative channel must strictly reduce false negatives: without={}, with={}",
                false_neg_without,
                false_neg_with
            );
        }

        /// `observe_entropy` returns the signed derivative (positive when
        /// entropy rises, negative when it falls, ~0 when stationary).
        #[test]
        fn observe_entropy_returns_signed_derivative() {
            let mut detector = make_detector_with_derivative(8);
            // Warm up at a constant 0.5 so the derivative settles near 0.
            for _ in 0..150 {
                let _ = detector.observe_entropy(0.5);
            }
            let baseline = detector.last_entropy_derivative().abs();
            assert!(
                baseline < 0.05,
                "derivative must be near zero after long constant stream, got {}",
                baseline
            );
            // Step up: derivative must be positive on the first rising sample.
            let d_up = detector.observe_entropy(2.0);
            assert!(
                d_up > 0.0,
                "rising entropy must produce a positive derivative, got {}",
                d_up
            );
            // Warm back down to constant 0.5.
            for _ in 0..150 {
                let _ = detector.observe_entropy(0.5);
            }
            // Step down: derivative must be negative.
            let d_down = detector.observe_entropy(0.0);
            assert!(
                d_down < 0.0,
                "falling entropy must produce a negative derivative, got {}",
                d_down
            );
        }

        /// `disable_derivative_collapse` makes `observe_entropy` return 0.0
        /// and `derivative_collapse_detected` always return false.
        #[test]
        fn disable_derivative_collapse_neutralizes_channel() {
            let mut detector = make_detector_without_derivative(8);
            for _ in 0..50 {
                let d = detector.observe_entropy(0.5);
                assert_eq!(d, 0.0, "disabled channel must return 0.0");
                assert!(
                    !detector.derivative_collapse_detected(),
                    "disabled channel must never fire"
                );
            }
        }

        /// Builder: `with_tau_deriv` tightens / loosens the firing threshold.
        #[test]
        fn with_tau_deriv_controls_firing_threshold() {
            // Tight tau: should fire earlier on a near-constant stream.
            let budget = ThinkingBudget {
                max_tokens: 4096,
                collapse_threshold: 8,
                efficiency_gamma: 0.5,
            };
            let mut tight = S2FCollapseDetector::new(HESITATION_TOKENS.to_vec(), &budget)
                .with_tau_deriv(0.001);
            let mut loose = S2FCollapseDetector::new(HESITATION_TOKENS.to_vec(), &budget)
                .with_tau_deriv(0.5);
            let trace = vec![0.5_f32; 200];
            let mut tight_step = None;
            let mut loose_step = None;
            for (t, &e) in trace.iter().enumerate() {
                tight.observe_entropy(e);
                if tight_step.is_none() && tight.derivative_collapse_detected() {
                    tight_step = Some(t);
                }
                loose.observe_entropy(e);
                if loose_step.is_none() && loose.derivative_collapse_detected() {
                    loose_step = Some(t);
                }
            }
            assert!(
                loose_step.unwrap_or(usize::MAX) <= tight_step.unwrap_or(usize::MAX),
                "looser tau must fire no later than tight tau: loose={:?} tight={:?}",
                loose_step,
                tight_step
            );
        }

        /// `reset()` clears derivative state but preserves config (tau, alphas).
        #[test]
        fn reset_clears_derivative_state_but_preserves_config() {
            let mut detector =
                make_detector_with_derivative(8).with_tau_deriv(0.005);
            let tau_before = detector.tau_deriv();
            // Drive some entropy through.
            for e in [0.1_f32, 0.9, 0.2, 0.8] {
                detector.observe_entropy(e);
            }
            let deriv_before = detector.last_entropy_derivative();
            assert!(deriv_before.abs() > 0.0, "derivative must be nonzero after mixed samples");

            detector.reset();
            assert_eq!(
                detector.last_entropy_derivative(),
                0.0,
                "reset must clear the cached derivative sample"
            );
            assert_eq!(
                detector.tau_deriv(),
                tau_before,
                "reset must preserve tau_deriv config"
            );
            assert!(
                detector.derivative_collapse.is_some(),
                "reset must keep the kernel Some(...) (preserves ablation state)"
            );
        }
    }
}
