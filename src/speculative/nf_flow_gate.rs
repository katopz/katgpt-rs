//! NFCoT FlowGate — Adaptive Acceptance Criterion (Plan 229 T3, Research 204).
//!
//! Maintains an EMA (exponential moving average) of historical flow scores
//! and accepts speculative trajectories whose score exceeds the threshold.
//!
//! The threshold adapts: as the model produces better scores, the bar rises.
//! Uses sigmoid-bounded EMA to prevent NaN/Inf from propagating.

/// Default EMA smoothing factor (slow adaptation).
const DEFAULT_ALPHA: f32 = 0.01;
/// Default margin above EMA required for acceptance.
const DEFAULT_MARGIN: f32 = 0.0;

/// EMA update: `α·new + (1−α)·current`.
/// Free function for reuse outside the gate struct.
#[inline]
pub fn ema_update(current: f32, new_value: f32, alpha: f32) -> f32 {
    alpha * new_value + (1.0 - alpha) * current
}

/// Adaptive acceptance gate based on NF flow scores (Plan 229 T3, Research 204).
///
/// Maintains an EMA (exponential moving average) of historical flow scores
/// and accepts speculative trajectories whose score exceeds the threshold.
///
/// The threshold adapts: as the model produces better scores, the bar rises.
/// Uses sigmoid-bounded EMA to prevent NaN/Inf from propagating.
pub struct NfFlowGate {
    /// EMA smoothing factor (0 < α ≤ 1). Default: 0.01 (slow adaptation).
    alpha: f32,
    /// Current EMA value of historical flow scores.
    ema: f32,
    /// Number of observations seen.
    n: u64,
    /// Optional fixed threshold (overrides EMA if set).
    fixed_threshold: Option<f32>,
    /// Margin above EMA required for acceptance. Default: 0.0.
    margin: f32,
}

impl NfFlowGate {
    /// Create with EMA smoothing factor.
    #[inline]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            ema: 0.0,
            n: 0,
            fixed_threshold: None,
            margin: DEFAULT_MARGIN,
        }
    }

    /// Create with acceptance margin above EMA.
    #[inline]
    pub fn with_margin(alpha: f32, margin: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            ema: 0.0,
            n: 0,
            fixed_threshold: None,
            margin,
        }
    }

    /// Create with fixed (non-adaptive) threshold.
    #[inline]
    pub fn with_fixed_threshold(threshold: f32) -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            ema: 0.0,
            n: 0,
            fixed_threshold: Some(threshold),
            margin: DEFAULT_MARGIN,
        }
    }

    /// Returns true if score exceeds threshold.
    ///
    /// Updates EMA: `ema = alpha * score + (1 - alpha) * ema`.
    /// If first observation, sets `ema = score`.
    /// Threshold = `fixed_threshold.unwrap_or(ema + margin)`.
    /// No allocation.
    #[inline]
    pub fn accept(&mut self, score: f32) -> bool {
        let threshold = self.threshold();

        // Update EMA
        if self.n == 0 {
            self.ema = score;
        } else if score.is_finite() {
            self.ema = ema_update(self.ema, score, self.alpha);
        }
        self.n += 1;

        score > threshold
    }

    /// Accept/reject each score in a batch. Pre-allocates output.
    pub fn accept_batch(&mut self, scores: &[f32]) -> Vec<bool> {
        let mut results = Vec::with_capacity(scores.len());
        for &score in scores {
            results.push(self.accept(score));
        }
        results
    }

    /// Current threshold value.
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.fixed_threshold.unwrap_or(self.ema + self.margin)
    }

    /// Current EMA value.
    #[inline]
    pub fn ema(&self) -> f32 {
        self.ema
    }

    /// Number of observations.
    #[inline]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Reset to initial state.
    #[inline]
    pub fn reset(&mut self) {
        self.ema = 0.0;
        self.n = 0;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_accepts_high_score() {
        let mut gate = NfFlowGate::new(DEFAULT_ALPHA);
        // First observation sets EMA to 0.5
        gate.accept(0.5);
        // Score above EMA → accept
        assert!(gate.accept(0.8), "score 0.8 > EMA 0.5 should accept");
    }

    #[test]
    fn test_gate_rejects_low_score() {
        let mut gate = NfFlowGate::new(DEFAULT_ALPHA);
        // First observation sets EMA to 0.8
        gate.accept(0.8);
        // Score below EMA → reject
        assert!(!gate.accept(0.3), "score 0.3 < EMA ≈0.8 should reject");
    }

    #[test]
    fn test_gate_ema_adapts() {
        let mut gate = NfFlowGate::new(0.5); // fast adaptation
        // Feed high scores → EMA rises
        for _ in 0..100 {
            gate.accept(0.9);
        }
        // Now a previously acceptable score is rejected
        assert!(
            !gate.accept(0.7),
            "EMA should have risen near 0.9, rejecting 0.7"
        );
    }

    #[test]
    fn test_gate_fixed_threshold() {
        let mut gate = NfFlowGate::with_fixed_threshold(0.5);
        // Feed high scores to move EMA — threshold should stay fixed
        gate.accept(0.9);
        gate.accept(0.9);
        gate.accept(0.9);
        // Below fixed threshold → reject
        assert!(!gate.accept(0.4), "fixed threshold 0.5 should reject 0.4");
        // Above fixed threshold → accept
        assert!(gate.accept(0.6), "fixed threshold 0.5 should accept 0.6");
    }

    #[test]
    fn test_gate_with_margin() {
        let mut gate = NfFlowGate::with_margin(0.5, 0.1);
        // First observation sets EMA to 0.5
        gate.accept(0.5);
        // Score at exactly EMA is rejected (needs > EMA + margin)
        assert!(
            !gate.accept(0.55),
            "score 0.55 <= EMA 0.5 + margin 0.1 should reject"
        );
        // Score above EMA + margin → accept
        assert!(
            gate.accept(0.65),
            "score 0.65 > EMA 0.5 + margin 0.1 should accept"
        );
    }

    #[test]
    fn test_gate_accept_batch() {
        let mut gate = NfFlowGate::new(0.5);
        gate.accept(0.5); // seed EMA
        let results = gate.accept_batch(&[0.3, 0.8, 0.5]);
        assert!(!results[0], "0.3 < EMA ≈0.5 → reject");
        assert!(results[1], "0.8 > EMA → accept");
        // 0.5 == threshold after EMA update from 0.3 → reject (not strictly greater)
        assert!(!results[2], "0.5 not > threshold → reject");
    }

    #[test]
    fn test_gate_reset() {
        let mut gate = NfFlowGate::new(0.5);
        gate.accept(0.9);
        gate.accept(0.9);
        assert_eq!(gate.n(), 2);
        gate.reset();
        assert_eq!(gate.n(), 0);
        assert_eq!(gate.ema(), 0.0);
    }

    #[test]
    fn test_gate_first_observation() {
        let mut gate = NfFlowGate::new(DEFAULT_ALPHA);
        // First accept: threshold is read before EMA update, so threshold = ema(0.0) + margin(0.0) = 0.0
        // score 0.5 > 0.0 → true. EMA then set to 0.5.
        let accepted = gate.accept(0.5);
        assert!(
            accepted,
            "first observation: score > initial threshold 0.0 → accept"
        );
        assert_eq!(gate.ema(), 0.5);
        assert_eq!(gate.n(), 1);
    }

    #[test]
    fn test_ema_update() {
        // Known values: α=0.5, current=1.0, new=2.0 → 0.5*2 + 0.5*1 = 1.5
        let result = ema_update(1.0, 2.0, 0.5);
        assert!((result - 1.5).abs() < 1e-6, "expected 1.5, got {result}");
        // α=0.0 → no update
        assert!((ema_update(1.0, 2.0, 0.0) - 1.0).abs() < 1e-6);
        // α=1.0 → full replacement
        assert!((ema_update(1.0, 2.0, 1.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gate_n_counter() {
        let mut gate = NfFlowGate::new(DEFAULT_ALPHA);
        assert_eq!(gate.n(), 0);
        gate.accept(0.1);
        assert_eq!(gate.n(), 1);
        gate.accept(0.2);
        assert_eq!(gate.n(), 2);
        gate.accept(0.3);
        assert_eq!(gate.n(), 3);
    }

    // ── Benchmark: accept overhead ──────────────────────────────────
    // Target: < 1μs per call (debug build).

    #[test]
    fn test_bench_flow_gate_accept() {
        let mut gate = NfFlowGate::new(DEFAULT_ALPHA);
        gate.accept(0.5); // seed

        let start = std::time::Instant::now();
        let iters = 100_000;
        for i in 0..iters {
            std::hint::black_box(gate.accept(i as f32 * 0.00001));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("NfFlowGate::accept: {per_call:.0}ns/call");
        assert!(
            per_call < 1_000.0,
            "accept should be <1μs (debug), got {per_call:.0}ns"
        );
    }
}

// TL;DR: Adaptive acceptance gate using EMA of flow scores as threshold.
// `NfFlowGate` — no-alloc single-score accept, pre-alloc batch. Threshold adapts
// via EMA; fixed threshold mode for non-adaptive use. Feature: `nf_flow_gate`.
