//! Wired engram evidence tripwire — the DETECTOR half of the Issue-837
//! detector+repair split (riir-ai Bench 832 §Wiring; owner call executed
//! 2026-09-01).
//!
//! [`crate::evidence_tripwire`] ships the metrics + the split-conformal
//! threshold primitive and deliberately leaves calibration policy to the
//! consumer. This module IS that consumer-side policy, factored to the seam it
//! monitors so every engram-fusion consumer inherits it: a stateful
//! [`EngramTripwire`] that owns the benign pool, maintains the conformal
//! threshold, and returns a verdict WITH suspect attribution.
//!
//! # Composition (existing substrate only — no new kernel math)
//!
//! ```text
//!   per-source σ gates   ← kernel::sigmoid_fuse_scaled_into(.., 1.0)  (returns
//!                            the unscaled gate; bit-identical gate math to
//!                            sigmoid_fuse_into — pinned by the Issue-656 unit
//!                            test AND re-pinned by the wiring GOAT)
//!   metrics              ← evidence_tripwire::tripwire_metrics_into
//!   threshold            ← evidence_tripwire::conformal_threshold over the
//!                            benign ring (fixed capacity, evict-oldest)
//!   decision             ← TripwireMetrics::rank_inversion_fires (the D-SCAN
//!                            N=1 rank channel — the measured discriminator)
//! ```
//!
//! # The verdict + repair contract
//!
//! [`EngramTripwireVerdict::fired`] means the top-consumed source's retrieval
//! rank exceeds the benign conformal threshold — the adversarial-injection
//! signature (poison constructed for consumption, not retrieval). The response
//! surface is [`EngramTripwireVerdict::suspect_source`]: the first argmax-gate
//! source index, deterministic. Consumer policy: DROP that source from the
//! consumed set and re-fuse / re-check — measured in the wiring GOAT
//! (`tests/engram_tripwire_wiring_goat.rs`) to clear the inversion and restore
//! consumption-retrieval agreement (τ > 0).
//!
//! Scope limits carried from Bench 832, unchanged:
//! - The entropy (`h_norm`) and Kendall-τ metric axes stay TELEMETRY — the σ
//!   gate is non-competitive (sigmoid, not softmax) so entropy saturates in a
//!   ~0.977–1.000 band on every world shape.
//! - The dual-optimized adversary and the diffuse equal-cosine utility poison
//!   are invisible to the rank channel; that axis belongs to the engram
//!   privilege ledger ([`super::privilege`], feature `engram_privilege`) — the
//!   repair half of the split.
//!
//! # Cost
//!
//! `observe_benign` recomputes the threshold each call: O(capacity) copy +
//! O(capacity log capacity) sort — calibration work, off the hot path by
//! design. `check` is read-only: O(K²) metrics against a cached threshold.
//! Zero steady-state allocation: pool + sort scratch are pre-reserved at
//! [`EngramTripwireConfig::benign_pool_capacity`] and the capacity-bounded
//! ring never grows (pinned by the GOAT G4 gate).

use crate::evidence_tripwire::{TripwireMetrics, conformal_threshold, tripwire_metrics_into};

/// Default conformal level (α = 5% — the Bench-832 operating point).
pub const DEFAULT_ALPHA: f64 = 0.05;

/// Default benign pool capacity (900 calibration worlds — the Bench-832
/// calibration scale).
pub const DEFAULT_BENIGN_POOL_CAPACITY: usize = 900;

/// Configuration for [`EngramTripwire`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngramTripwireConfig {
    /// Conformal miscoverage level in (0, 1). A new BENIGN world's
    /// normalized top1-rank exceeds the maintained threshold with
    /// probability ≤ `alpha` under exchangeability of the benign pool.
    pub alpha: f64,
    /// Benign-pool ring capacity (worlds). Once full, the oldest score is
    /// evicted on every new observation (a sliding window — recent benign
    /// behavior stays the calibration reference). Pre-reserved: the pool
    /// never reallocates after construction.
    pub benign_pool_capacity: usize,
}

impl Default for EngramTripwireConfig {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            benign_pool_capacity: DEFAULT_BENIGN_POOL_CAPACITY,
        }
    }
}

/// The wired detector: benign-pool ring + split-conformal threshold + verdict
/// with suspect attribution.
///
/// Construct once per consumer (per seam / per NPC / per retrieval table),
/// [`EngramTripwire::observe_benign`] on worlds trusted benign,
/// [`EngramTripwire::check`] on every consumed world.
#[derive(Debug)]
pub struct EngramTripwire {
    config: EngramTripwireConfig,
    /// Benign normalized-top1-rank scores, oldest first, `len <= capacity`.
    pool: Vec<f64>,
    /// Sort scratch for threshold recomputation — pre-reserved, cleared+refilled.
    scratch: Vec<f64>,
    /// Cached conformal threshold (`f64::INFINITY` until the first benign
    /// observation — an uncalibrated tripwire never fires).
    threshold: f64,
    benign_worlds: u64,
}

/// Result of [`EngramTripwire::check`] — metrics land in the caller's scratch
/// [`TripwireMetrics`] (zero-alloc contract); this carries only the decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngramTripwireVerdict {
    /// `true` iff the rank-inversion statistic exceeded the maintained
    /// benign-conformal threshold. Never `true` before calibration.
    pub fired: bool,
    /// `false` while the benign pool is empty (threshold = +∞).
    pub calibrated: bool,
    /// The threshold this verdict was decided against.
    pub threshold: f64,
    /// Index (into the consumed set) of the first argmax-gate source — the
    /// drop target for the repair half. Deterministic (first maximum).
    pub suspect_source: usize,
}

/// First argmax of `gates` (strictly-greater scan — same deterministic rule
/// [`tripwire_metrics_into`] uses for the N=1 statistic).
#[inline]
pub fn suspect_gate_source(gates: &[f32]) -> usize {
    debug_assert!(!gates.is_empty(), "tripwire: consumed set must be non-empty");
    let mut am = 0usize;
    for (i, &g) in gates.iter().enumerate() {
        if g > gates[am] {
            am = i;
        }
    }
    am
}

impl EngramTripwire {
    /// Construct with pre-reserved pool + scratch capacity.
    ///
    /// # Panics (debug only)
    /// `alpha` outside (0, 1) — [`conformal_threshold`] would refuse it at
    /// calibration time anyway; failing at construction is louder.
    #[must_use]
    pub fn new(config: EngramTripwireConfig) -> Self {
        debug_assert!(
            config.alpha > 0.0 && config.alpha < 1.0,
            "engram_tripwire: alpha must be in (0, 1)"
        );
        let cap = config.benign_pool_capacity.max(1);
        Self {
            config,
            pool: Vec::with_capacity(cap),
            scratch: Vec::with_capacity(cap),
            threshold: f64::INFINITY,
            benign_worlds: 0,
        }
    }

    /// Observe a world trusted benign: compute metrics, slide the ring, and
    /// refresh the conformal threshold. O(capacity log capacity).
    ///
    /// Metrics land in `metrics` (caller-owned scratch, zero-alloc).
    /// `retrieval.len()` must equal `gates.len()` (debug-asserted by the
    /// primitive); gates must be strictly positive (σ gates are).
    pub fn observe_benign(&mut self, retrieval: &[f32], gates: &[f32], metrics: &mut TripwireMetrics) {
        tripwire_metrics_into(retrieval, gates, metrics);
        let score = metrics.normalized_top1_rank();
        let cap = self.config.benign_pool_capacity.max(1);
        if self.pool.len() >= cap {
            let excess = self.pool.len() + 1 - cap;
            self.pool.drain(0..excess);
        }
        self.pool.push(score);
        self.benign_worlds += 1;
        self.recompute_threshold();
    }

    /// Check a consumed world against the maintained threshold. READ-ONLY —
    /// the pool, the threshold, and the fusion outputs are untouched (G1
    /// observer purity is pinned by the wiring GOAT). O(K²).
    pub fn check(
        &self,
        retrieval: &[f32],
        gates: &[f32],
        metrics: &mut TripwireMetrics,
    ) -> EngramTripwireVerdict {
        tripwire_metrics_into(retrieval, gates, metrics);
        EngramTripwireVerdict {
            fired: metrics.rank_inversion_fires(self.threshold),
            calibrated: self.threshold.is_finite(),
            threshold: self.threshold,
            suspect_source: suspect_gate_source(gates),
        }
    }

    /// Current conformal threshold (`f64::INFINITY` before calibration).
    #[inline]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether at least one benign world has been observed.
    #[inline]
    pub fn is_calibrated(&self) -> bool {
        self.threshold.is_finite()
    }

    /// Total benign worlds observed (monotone; the ring is a window over the
    /// most recent `benign_pool_capacity` of them).
    #[inline]
    pub fn benign_worlds(&self) -> u64 {
        self.benign_worlds
    }

    /// Current ring occupancy (≤ [`EngramTripwireConfig::benign_pool_capacity`]).
    #[inline]
    pub fn pool_len(&self) -> usize {
        self.pool.len()
    }

    /// The detector's configuration.
    #[inline]
    pub fn config(&self) -> &EngramTripwireConfig {
        &self.config
    }

    fn recompute_threshold(&mut self) {
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.pool);
        self.threshold = conformal_threshold(&mut self.scratch, self.config.alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics_scratch() -> TripwireMetrics {
        TripwireMetrics {
            n: 0,
            h_norm: 0.0,
            top1_share: 0.0,
            tau: 0.0,
            top1_consumer_rank: 0.0,
        }
    }

    /// retrieval order == gate order (benign): rank 1, never fires.
    fn correlated(k: usize) -> (Vec<f32>, Vec<f32>) {
        let retrieval: Vec<f32> = (0..k).map(|i| 1.0 - i as f32 * 0.1).collect();
        let gates: Vec<f32> = (0..k).map(|i| 10.0 - i as f32).collect();
        (retrieval, gates)
    }

    /// top-gated source is retrieval-LAST (the adversarial signature).
    fn inverted(k: usize) -> (Vec<f32>, Vec<f32>) {
        let retrieval: Vec<f32> = (0..k).map(|i| 1.0 - i as f32 * 0.1).collect();
        let mut gates = vec![1.0f32; k];
        gates[k - 1] = 9.0;
        (retrieval, gates)
    }

    #[test]
    fn uncalibrated_never_fires() {
        let tw = EngramTripwire::new(EngramTripwireConfig::default());
        let (r, g) = inverted(8);
        let mut m = metrics_scratch();
        let v = tw.check(&r, &g, &mut m);
        assert!(!v.fired);
        assert!(!v.calibrated);
        assert_eq!(v.threshold, f64::INFINITY);
        assert_eq!(v.suspect_source, 7);
    }

    #[test]
    fn calibrates_and_fires_on_inversion() {
        let mut tw = EngramTripwire::new(EngramTripwireConfig::default());
        let mut m = metrics_scratch();
        for _ in 0..32 {
            let (r, g) = correlated(8);
            tw.observe_benign(&r, &g, &mut m);
        }
        assert!(tw.is_calibrated());
        assert_eq!(tw.benign_worlds(), 32);
        assert!(tw.threshold() < f64::INFINITY);

        let (r, g) = correlated(8);
        let v = tw.check(&r, &g, &mut m);
        assert!(!v.fired, "benign correlated world must not fire");

        let (r, g) = inverted(8);
        let v = tw.check(&r, &g, &mut m);
        assert!(v.fired, "rank-inverted world must fire after calibration");
        assert_eq!(v.suspect_source, 7);
    }

    #[test]
    fn pool_is_capacity_bounded_and_threshold_tracks_the_window() {
        let cfg = EngramTripwireConfig {
            alpha: 0.05,
            benign_pool_capacity: 8,
        };
        let mut tw = EngramTripwire::new(cfg);
        let mut m = metrics_scratch();
        for i in 0..20 {
            // Rank varies with i so evicted-oldest is observable.
            let k = 4;
            let mut gates = vec![1.0f32; k];
            let idx = i % k;
            gates[idx] = 9.0;
            let retrieval: Vec<f32> = (0..k).map(|j| 1.0 - j as f32 * 0.1).collect();
            tw.observe_benign(&retrieval, &gates, &mut m);
        }
        assert_eq!(tw.pool_len(), 8);
        // Threshold equals the conformal statistic of the LAST 8 ranks —
        // the window, not the full history.
        let expected_ranks: Vec<f64> = (12..20)
            .map(|i| {
                let k = 4;
                let idx = i % k;
                (idx as f64) / (k as f64 - 1.0)
            })
            .collect();
        let mut sorted = expected_ranks;
        assert!((tw.threshold() - conformal_threshold(&mut sorted, 0.05)).abs() < 1e-12);
    }

    #[test]
    fn check_is_read_only_and_deterministic() {
        let mut tw = EngramTripwire::new(EngramTripwireConfig::default());
        let mut m = metrics_scratch();
        for _ in 0..8 {
            let (r, g) = correlated(8);
            tw.observe_benign(&r, &g, &mut m);
        }
        let (worlds, len, threshold) = (tw.benign_worlds(), tw.pool_len(), tw.threshold());
        let (r, g) = inverted(8);
        let v1 = tw.check(&r, &g, &mut m);
        let v2 = tw.check(&r, &g, &mut m);
        assert_eq!(v1, v2);
        assert_eq!((tw.benign_worlds(), tw.pool_len(), tw.threshold()),
            (worlds, len, threshold), "check must not mutate detector state");
    }

    #[test]
    fn suspect_gate_source_first_max() {
        assert_eq!(suspect_gate_source(&[0.5, 0.7, 0.7]), 1);
        assert_eq!(suspect_gate_source(&[0.9, 0.1]), 0);
        assert_eq!(suspect_gate_source(&[0.1]), 0);
    }
}
