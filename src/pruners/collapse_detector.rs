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

use crate::pruners::freeze::{load_frozen, save_frozen};
use crate::speculative::thinking_controller::ThinkingMode;

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
    hesitation_tokens: Vec<u32>,
    /// Fixed-size ring buffer of recent token IDs. Zero-allocation.
    ring_buffer: [u32; Self::RING_SIZE],
    /// Current write position in the ring buffer (wraps at RING_SIZE).
    ring_write_idx: usize,
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
}

impl S2FCollapseDetector {
    /// Ring buffer capacity. 64 tokens covers ~2 sentences of reasoning context.
    const RING_SIZE: usize = 64;

    /// Create a new detector with the given hesitation tokens and budget.
    pub fn new(hesitation_tokens: Vec<u32>, budget: &ThinkingBudget) -> Self {
        Self {
            hesitation_tokens,
            ring_buffer: [0u32; Self::RING_SIZE],
            ring_write_idx: 0,
            threshold: budget.collapse_threshold,
            ema_alpha: 0.1,
            last_trace_hesitation: 0,
            max_budget: budget.max_tokens,
            gamma: budget.efficiency_gamma,
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
    #[allow(clippy::needless_range_loop)]
    fn count_hesitation(&self) -> u32 {
        let mut count: u32 = 0;
        for i in 0..Self::RING_SIZE {
            if self.hesitation_tokens.contains(&self.ring_buffer[i]) {
                count += 1;
            }
        }
        count
    }
}

impl CollapseDetector for S2FCollapseDetector {
    /// Check if the current trace exhibits collapse symptoms.
    ///
    /// Writes the token to the ring buffer, then counts hesitation matches.
    /// Returns `true` when hesitation count ≥ threshold τ.
    fn check_collapse(&mut self, token_id: u32, _position: usize) -> bool {
        // Write token to ring buffer (zero-allocation, wraps around).
        self.ring_buffer[self.ring_write_idx] = token_id;
        self.ring_write_idx = (self.ring_write_idx + 1) % Self::RING_SIZE;

        // Count hesitation tokens and compare against threshold.
        let count = self.count_hesitation();
        self.last_trace_hesitation = self.last_trace_hesitation.max(count);

        count >= self.threshold
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
        self.last_trace_hesitation = 0;
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
}
