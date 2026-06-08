//! Auto-route integration for segment checkpoint.
//!
//! Adjusts checkpoint parameters based on inference load (QPS).
//! High QPS → SSC with lower k and shorter segments for throughput.
//! Low QPS → GRM with full retrieval for best accuracy.

/// Auto-route configuration for segment checkpointing.
///
/// Adjusts SSC k and segment_size based on current QPS to balance
/// throughput vs accuracy dynamically.
#[derive(Clone, Debug)]
pub struct SegmentAutoRoute {
    /// High QPS threshold for SSC mode.
    pub high_qps_threshold: f32,
    /// SSC k value for high QPS mode.
    pub ssc_k_high: usize,
    /// SSC k value for low QPS mode.
    pub ssc_k_low: usize,
    /// Segment size for low QPS.
    pub segment_size_low: usize,
    /// Segment size for high QPS.
    pub segment_size_high: usize,
}

impl Default for SegmentAutoRoute {
    fn default() -> Self {
        Self {
            high_qps_threshold: 10.0,
            ssc_k_high: 4,
            ssc_k_low: 8,
            segment_size_low: 128,
            segment_size_high: 64,
        }
    }
}

impl SegmentAutoRoute {
    /// Select SSC k based on current QPS.
    ///
    /// High QPS → lower k (less retrieval overhead).
    /// Low QPS → higher k (better accuracy).
    pub fn select_k(&self, qps: f32) -> usize {
        if qps > self.high_qps_threshold {
            self.ssc_k_high
        } else {
            self.ssc_k_low
        }
    }

    /// Select segment size based on current QPS.
    ///
    /// High QPS → shorter segments (more frequent checkpoints).
    /// Low QPS → longer segments (less overhead).
    pub fn select_segment_size(&self, qps: f32) -> usize {
        if qps > self.high_qps_threshold {
            self.segment_size_high
        } else {
            self.segment_size_low
        }
    }

    /// Select checkpoint policy based on QPS.
    ///
    /// Maps to CheckpointPolicy tiers:
    /// - High QPS → Lazy (every 4th boundary)
    /// - Medium QPS → Normal (every boundary)
    /// - Low QPS → Eager (every boundary + pre-compute summaries)
    pub fn select_policy(&self, qps: f32) -> crate::segment_checkpoint::CheckpointPolicy {
        crate::segment_checkpoint::CheckpointPolicy::from_tier(qps)
    }
}

#[cfg(test)]
mod auto_route_tests {
    use super::*;

    #[test]
    fn test_high_qps_lower_k() {
        let route = SegmentAutoRoute::default();
        assert!(
            route.select_k(20.0) < route.select_k(1.0),
            "high QPS should use lower k: {} vs {}",
            route.select_k(20.0),
            route.select_k(1.0)
        );
    }

    #[test]
    fn test_high_qps_shorter_segments() {
        let route = SegmentAutoRoute::default();
        assert!(
            route.select_segment_size(20.0) < route.select_segment_size(1.0),
            "high QPS should use shorter segments: {} vs {}",
            route.select_segment_size(20.0),
            route.select_segment_size(1.0)
        );
    }

    #[test]
    fn test_default_threshold() {
        let route = SegmentAutoRoute::default();
        assert!((route.high_qps_threshold - 10.0).abs() < 1e-6);
        assert_eq!(route.ssc_k_high, 4);
        assert_eq!(route.ssc_k_low, 8);
        assert_eq!(route.segment_size_low, 128);
        assert_eq!(route.segment_size_high, 64);
    }

    #[test]
    fn test_boundary_qps_uses_low() {
        let route = SegmentAutoRoute::default();
        // QPS == threshold is NOT high → uses low (default branch)
        assert_eq!(route.select_k(10.0), route.ssc_k_low);
        assert_eq!(route.select_segment_size(10.0), route.segment_size_low);
    }

    #[test]
    fn test_just_above_threshold_uses_high() {
        let route = SegmentAutoRoute::default();
        assert_eq!(route.select_k(10.01), route.ssc_k_high);
        assert_eq!(route.select_segment_size(10.01), route.segment_size_high);
    }

    #[test]
    fn test_custom_route() {
        let route = SegmentAutoRoute {
            high_qps_threshold: 50.0,
            ssc_k_high: 2,
            ssc_k_low: 16,
            segment_size_low: 256,
            segment_size_high: 32,
        };
        assert_eq!(route.select_k(100.0), 2);
        assert_eq!(route.select_k(1.0), 16);
        assert_eq!(route.select_segment_size(100.0), 32);
        assert_eq!(route.select_segment_size(1.0), 256);
    }

    #[test]
    fn test_segment_sizes_tile_aligned() {
        let route = SegmentAutoRoute::default();
        let tile_size = 64; // minimum tile alignment
        // Both sizes should be tile-aligned for zero-copy
        assert_eq!(route.segment_size_low % tile_size, 0);
        assert_eq!(route.segment_size_high % tile_size, 0);
    }

    #[test]
    fn test_select_policy_integration() {
        let route = SegmentAutoRoute::default();
        use crate::segment_checkpoint::CheckpointPolicy;

        // High QPS → Lazy
        assert_eq!(route.select_policy(50.0), CheckpointPolicy::Lazy);
        // Medium QPS → Normal
        assert_eq!(route.select_policy(10.0), CheckpointPolicy::Normal);
        // Low QPS → Eager
        assert_eq!(route.select_policy(1.0), CheckpointPolicy::Eager);
    }
}
