//! TriggerGate — adaptive compute-tier escalation based on runtime load metrics.
//!
//! Monitors QPS, queue depth, and latency to decide when to promote from CPU-only
//! to CPU+GPU to CPU+GPU+ANE, with hysteresis to prevent thrashing.

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// ComputeTier
// ---------------------------------------------------------------------------

/// Available compute tiers, ordered by capability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ComputeTier {
    /// CPU only — idle, dev mode, low load.
    #[default]
    CpuOnly = 0,
    /// CPU + GPU — medium load, GPU handles forward pass.
    CpuGpu = 1,
    /// CPU + GPU + ANE — high load, ALL hardware engaged.
    CpuGpuAne = 2,
}

impl ComputeTier {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::CpuOnly,
            1 => Self::CpuGpu,
            2 => Self::CpuGpuAne,
            // Fall back to the safest tier on corruption.
            _ => Self::CpuOnly,
        }
    }
}

impl fmt::Display for ComputeTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuOnly => write!(f, "CPU_ONLY"),
            Self::CpuGpu => write!(f, "CPU+GPU"),
            Self::CpuGpuAne => write!(f, "CPU+GPU+ANE"),
        }
    }
}

// ---------------------------------------------------------------------------
// TriggerGateConfig
// ---------------------------------------------------------------------------

/// Configuration parameters that control tier promotion / demotion behaviour.
#[derive(Clone, Debug)]
pub struct TriggerGateConfig {
    /// Activate GPU when QPS exceeds this. Default: 10_000.0
    pub gpu_activate_qps: f64,
    /// Activate ANE when QPS exceeds this. Default: 100_000.0
    pub ane_activate_qps: f64,
    /// Deactivate tier at threshold * this factor. Default: 0.7
    pub hysteresis_factor: f64,
    /// Queue depth that triggers tier-up. Default: 100
    pub queue_depth_trigger: usize,
    /// Latency P99 that triggers tier-up (microseconds). Default: 5000
    pub latency_p99_trigger_us: u64,
    /// Minimum time between tier changes (milliseconds). Default: 500
    pub min_tier_change_interval_ms: u64,
}

impl Default for TriggerGateConfig {
    fn default() -> Self {
        Self {
            gpu_activate_qps: 10_000.0,
            ane_activate_qps: 100_000.0,
            hysteresis_factor: 0.7,
            queue_depth_trigger: 100,
            latency_p99_trigger_us: 5000,
            min_tier_change_interval_ms: 500,
        }
    }
}

impl fmt::Display for TriggerGateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TriggerGateConfig {{ gpu_activate_qps: {}, ane_activate_qps: {}, \
             hysteresis_factor: {}, queue_depth_trigger: {}, \
             latency_p99_trigger_us: {}, min_tier_change_interval_ms: {} }}",
            self.gpu_activate_qps,
            self.ane_activate_qps,
            self.hysteresis_factor,
            self.queue_depth_trigger,
            self.latency_p99_trigger_us,
            self.min_tier_change_interval_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// TriggerGate
// ---------------------------------------------------------------------------

/// Adaptive compute-tier gate that promotes/demotes based on live load metrics.
pub struct TriggerGate {
    config: TriggerGateConfig,
    /// Monotonically increasing inference counter.
    inference_count: AtomicU64,
    /// Sum of inference durations in microseconds.
    latency_sum_us: AtomicU64,
    /// Current queue depth (stored as `usize` bits).
    current_queue_depth: AtomicU64,
    /// Instant of last tier change.
    last_tier_change: Mutex<Instant>,
    /// Window start — used for QPS estimation. Reset on tier change.
    window_start: Mutex<Instant>,
    /// Current tier (stored as `ComputeTier` discriminant).
    current_tier: AtomicU8,
    /// Whether GPU is available (set once at construction).
    gpu_available: bool,
    /// Whether ANE is available (set once at construction).
    ane_available: bool,
}

impl TriggerGate {
    /// Create a new `TriggerGate`.
    ///
    /// Starts at [`ComputeTier::CpuOnly`].
    pub fn new(config: TriggerGateConfig, gpu_available: bool, ane_available: bool) -> Self {
        let now = Instant::now();
        Self {
            config,
            inference_count: AtomicU64::new(0),
            latency_sum_us: AtomicU64::new(0),
            current_queue_depth: AtomicU64::new(0),
            last_tier_change: Mutex::new(now),
            window_start: Mutex::new(now),
            current_tier: AtomicU8::new(ComputeTier::CpuOnly as u8),
            gpu_available,
            ane_available,
        }
    }

    /// Current compute tier.
    pub fn current_tier(&self) -> ComputeTier {
        ComputeTier::from_u8(self.current_tier.load(Ordering::Relaxed))
    }

    /// Record a completed inference.
    pub fn record_inference(&self, duration_us: u64) {
        self.inference_count.fetch_add(1, Ordering::Relaxed);
        self.latency_sum_us
            .fetch_add(duration_us, Ordering::Relaxed);
    }

    /// Update the observed queue depth.
    pub fn record_queue_depth(&self, depth: usize) {
        self.current_queue_depth
            .store(depth as u64, Ordering::Relaxed);
    }

    /// Estimated queries-per-second over the current measurement window.
    ///
    /// QPS = inference_count / elapsed_seconds.
    /// Returns 0.0 when fewer than two samples or the window is too short.
    pub fn estimated_qps(&self) -> f64 {
        let count = self.inference_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let start = self.window_start.lock().unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed < f64::EPSILON {
            return 0.0;
        }
        count as f64 / elapsed
    }

    /// Check whether load warrants promotion to a higher tier.
    pub fn should_promote(&self) -> Option<ComputeTier> {
        let qps = self.estimated_qps();
        let depth = self.current_queue_depth.load(Ordering::Relaxed) as usize;

        match self.current_tier() {
            ComputeTier::CpuOnly if self.gpu_available => {
                if qps >= self.config.gpu_activate_qps || depth >= self.config.queue_depth_trigger {
                    return Some(ComputeTier::CpuGpu);
                }
            }
            ComputeTier::CpuGpu if self.ane_available => {
                if qps >= self.config.ane_activate_qps
                    || depth >= self.config.queue_depth_trigger * 2
                {
                    return Some(ComputeTier::CpuGpuAne);
                }
            }
            _ => {}
        }
        None
    }

    /// Check whether load has dropped enough to justify demotion (with hysteresis).
    pub fn should_demote(&self) -> Option<ComputeTier> {
        let qps = self.estimated_qps();

        match self.current_tier() {
            ComputeTier::CpuGpuAne => {
                if qps < self.config.ane_activate_qps * self.config.hysteresis_factor {
                    return Some(ComputeTier::CpuGpu);
                }
            }
            ComputeTier::CpuGpu => {
                if qps < self.config.gpu_activate_qps * self.config.hysteresis_factor {
                    return Some(ComputeTier::CpuOnly);
                }
            }
            _ => {}
        }
        None
    }

    /// Evaluate whether a tier change is warranted, respecting the minimum interval.
    ///
    /// Returns `Some(new_tier)` if a change is recommended, `None` otherwise.
    /// On a recommended change the internal counters are reset and the tier is updated.
    pub fn evaluate(&self) -> Option<ComputeTier> {
        let mut last = self.last_tier_change.lock().unwrap();
        let min_interval =
            std::time::Duration::from_millis(self.config.min_tier_change_interval_ms);

        // Try promotion first — more conservative (prefer extra compute over dropped requests).
        let candidate = self.should_promote().or_else(|| self.should_demote())?;

        // Enforce minimum interval between tier changes.
        if last.elapsed() < min_interval {
            return None;
        }

        // Commit the tier change.
        self.current_tier.store(candidate as u8, Ordering::Relaxed);
        *last = Instant::now();

        // Reset measurement window.
        self.inference_count.store(0, Ordering::Relaxed);
        self.latency_sum_us.store(0, Ordering::Relaxed);
        let mut window = self.window_start.lock().unwrap();
        *window = Instant::now();

        Some(candidate)
    }

    /// Whether GPU was reported as available at construction time.
    pub fn gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Whether ANE was reported as available at construction time.
    pub fn ane_available(&self) -> bool {
        self.ane_available
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &TriggerGateConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    /// Helper: build a gate with tiny intervals so tests don't have to sleep long.
    fn fast_gate(gpu: bool, ane: bool) -> TriggerGate {
        let config = TriggerGateConfig {
            gpu_activate_qps: 10_000.0,
            ane_activate_qps: 100_000.0,
            hysteresis_factor: 0.7,
            queue_depth_trigger: 100,
            latency_p99_trigger_us: 5000,
            min_tier_change_interval_ms: 10, // 10 ms — fast for tests
        };
        TriggerGate::new(config, gpu, ane)
    }

    // 1. Starts at CpuOnly.
    #[test]
    fn test_trigger_gate_starts_cpu_only() {
        let gate = fast_gate(true, true);
        assert_eq!(gate.current_tier(), ComputeTier::CpuOnly);
    }

    // 2. Promotes to CpuGpu under high QPS.
    #[test]
    fn test_trigger_gate_promotes_to_gpu() {
        let gate = fast_gate(true, false);
        // Seed enough inferences to push QPS above 10 000.
        // We record 20 000 inferences, then wait a tiny bit so elapsed > 0.
        for _ in 0..20_000 {
            gate.record_inference(50);
        }
        thread::sleep(Duration::from_millis(20)); // past min_tier_change_interval

        let new_tier = gate.evaluate();
        assert_eq!(new_tier, Some(ComputeTier::CpuGpu));
        assert_eq!(gate.current_tier(), ComputeTier::CpuGpu);
    }

    // 3. Promotes to CpuGpuAne under very high QPS.
    #[test]
    fn test_trigger_gate_promotes_to_ane() {
        let gate = fast_gate(true, true);

        // First promote to CpuGpu.
        for _ in 0..20_000 {
            gate.record_inference(50);
        }
        thread::sleep(Duration::from_millis(20));
        let t = gate.evaluate();
        assert_eq!(t, Some(ComputeTier::CpuGpu));

        // Now push QPS above 100 000.
        // Record a large batch quickly so QPS is very high.
        for _ in 0..200_000 {
            gate.record_inference(10);
        }
        thread::sleep(Duration::from_millis(20));
        let t2 = gate.evaluate();
        assert_eq!(t2, Some(ComputeTier::CpuGpuAne));
        assert_eq!(gate.current_tier(), ComputeTier::CpuGpuAne);
    }

    // 4. Hysteresis prevents thrashing when QPS oscillates around the threshold.
    #[test]
    fn test_hysteresis_prevents_thrashing() {
        let gate = fast_gate(true, false);

        // Promote to CpuGpu.
        for _ in 0..20_000 {
            gate.record_inference(50);
        }
        thread::sleep(Duration::from_millis(20));
        let _ = gate.evaluate();
        assert_eq!(gate.current_tier(), ComputeTier::CpuGpu);

        // Drop QPS just below the activation threshold (not below threshold * hysteresis).
        // QPS should still be well above 10_000 * 0.7 = 7_000.
        // We simulate a burst of 9_500 inferences — QPS will be high initially but
        // we won't re-evaluate until the next cycle.
        // With the reset counters, we'll quickly record a modest batch.
        for _ in 0..9_500 {
            gate.record_inference(100);
        }
        thread::sleep(Duration::from_millis(20));
        // QPS is ~9_500 / elapsed_secs. elapsed is small so QPS will still be high.
        // The point: should_demote returns None because QPS >= threshold * hysteresis.
        // We force a scenario: record 5_000 inferences over 1 second => QPS ≈ 5_000.
        // 5_000 < 10_000 * 0.7 = 7_000? No: 5_000 < 7_000 → would demote.
        // Let's record 8_000 over 1 second: 8_000 >= 7_000 → no demotion.
        // But 8_000 < 10_000 → no promotion.
        // Actually we need to be more careful with the test.
        // Use queue depth = 0 and control QPS precisely.

        // Reset: create a fresh gate scenario.
        let gate2 = fast_gate(true, false);
        for _ in 0..20_000 {
            gate2.record_inference(50);
        }
        thread::sleep(Duration::from_millis(20));
        let _ = gate2.evaluate();
        assert_eq!(gate2.current_tier(), ComputeTier::CpuGpu);

        // After evaluate(), counters are reset. Now record exactly 8_000 inferences
        // and sleep ~1s so QPS ≈ 8_000 which is above 7_000 (hysteresis) but below 10_000 (activate).
        // This means: no demotion (QPS > hysteresis threshold) and no promotion (QPS < activate).
        for _ in 0..8_000 {
            gate2.record_inference(100);
        }
        thread::sleep(Duration::from_millis(1100)); // ~1 s

        let tier = gate2.evaluate();
        // No change recommended — tier stays at CpuGpu.
        assert_eq!(tier, None);
        assert_eq!(gate2.current_tier(), ComputeTier::CpuGpu);
    }

    // 5. Demotion only happens below hysteresis threshold.
    #[test]
    fn test_tier_down_requires_hysteresis() {
        let gate = fast_gate(true, false);

        // Promote to CpuGpu.
        for _ in 0..20_000 {
            gate.record_inference(50);
        }
        thread::sleep(Duration::from_millis(20));
        let _ = gate.evaluate();
        assert_eq!(gate.current_tier(), ComputeTier::CpuGpu);

        // Drop QPS to 8_000 — above 7_000 (hysteresis), so no demotion.
        for _ in 0..8_000 {
            gate.record_inference(100);
        }
        thread::sleep(Duration::from_millis(1100));
        assert_eq!(gate.evaluate(), None);

        // Now drop QPS below hysteresis threshold: 5_000 < 7_000.
        for _ in 0..5_000 {
            gate.record_inference(100);
        }
        thread::sleep(Duration::from_millis(1100));
        let demoted = gate.evaluate();
        assert_eq!(demoted, Some(ComputeTier::CpuOnly));
        assert_eq!(gate.current_tier(), ComputeTier::CpuOnly);
    }

    // 6. Minimum tier change interval blocks rapid changes.
    #[test]
    fn test_min_tier_change_interval() {
        let gate = fast_gate(true, false);

        // Promote to CpuGpu.
        for _ in 0..20_000 {
            gate.record_inference(50);
        }
        thread::sleep(Duration::from_millis(20));
        let _ = gate.evaluate();
        assert_eq!(gate.current_tier(), ComputeTier::CpuGpu);

        // Immediately try to demote — counters are reset so we can record low activity.
        // Record very few inferences over a tiny window.
        gate.record_inference(1);
        // Don't sleep — evaluate() immediately should be blocked by min interval.
        let blocked = gate.evaluate();
        assert_eq!(blocked, None);
        assert_eq!(gate.current_tier(), ComputeTier::CpuGpu);
    }

    // 7. ComputeTier ordering: CpuOnly < CpuGpu < CpuGpuAne.
    #[test]
    fn test_compute_tier_ordering() {
        assert!(ComputeTier::CpuOnly < ComputeTier::CpuGpu);
        assert!(ComputeTier::CpuGpu < ComputeTier::CpuGpuAne);
        assert!(ComputeTier::CpuOnly < ComputeTier::CpuGpuAne);
    }

    // 8. No GPU available → stays at CpuOnly even under high load.
    #[test]
    fn test_no_gpu_available_skips_gpu_tier() {
        let gate = fast_gate(false, false);

        // Hammer with inferences.
        for _ in 0..200_000 {
            gate.record_inference(10);
        }
        thread::sleep(Duration::from_millis(20));

        // No GPU available — cannot promote.
        assert_eq!(gate.evaluate(), None);
        assert_eq!(gate.current_tier(), ComputeTier::CpuOnly);
    }

    // 9. TriggerGateConfig Display shows all values.
    #[test]
    fn test_trigger_gate_config_display() {
        let cfg = TriggerGateConfig::default();
        let s = format!("{cfg}");
        assert!(s.contains("10000"), "should show gpu_activate_qps");
        assert!(s.contains("100000"), "should show ane_activate_qps");
        assert!(s.contains("0.7"), "should show hysteresis_factor");
        assert!(s.contains("100"), "should show queue_depth_trigger");
        assert!(s.contains("5000"), "should show latency_p99_trigger_us");
        assert!(s.contains("500"), "should show min_tier_change_interval_ms");
    }

    // 10. ComputeTier Display for all tiers.
    #[test]
    fn test_compute_tier_display() {
        assert_eq!(format!("{}", ComputeTier::CpuOnly), "CPU_ONLY");
        assert_eq!(format!("{}", ComputeTier::CpuGpu), "CPU+GPU");
        assert_eq!(format!("{}", ComputeTier::CpuGpuAne), "CPU+GPU+ANE");
    }
}
