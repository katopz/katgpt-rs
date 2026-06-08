//! Benchmarks for Plan 218 — BFCF × LFU × Sharding Phase 1.
//!
//! B1: Cache hit rate on synthetic 100-step workload (target: ≥ 60%).

#![cfg(all(feature = "bfcf_lfu_shard", feature = "speculative_generator"))]

use katgpt_rs::pruners::{
    BFCP, BfcpRegionCache, BorelRegion, HalfSpace, RegionLabel, blake3_logit_hash,
};

/// Create a synthetic BFCP partition with `n_regions` regions.
fn make_partition(n_regions: usize, base_tokens: usize) -> BFCP {
    let regions: Vec<BorelRegion> = (0..n_regions)
        .map(|i| {
            BorelRegion::new(
                match i % 3 {
                    0 => RegionLabel::Accept,
                    1 => RegionLabel::Reject,
                    _ => RegionLabel::Maybe,
                },
                vec![HalfSpace {
                    dim: i as u16,
                    threshold: 0.5,
                    above: true,
                }],
                base_tokens + i,
            )
        })
        .collect();
    BFCP::from_regions(regions)
}

/// Generate logits that cycle through a small set of patterns.
/// This simulates a decode sequence where consecutive steps produce
/// similar partitions (high temporal locality).
fn cyclic_logits(step: usize, pattern_size: usize) -> Vec<f32> {
    let base = (step % pattern_size) as f32;
    (0..8).map(|i| base + i as f32 * 0.1).collect()
}

#[test]
fn b1_cache_hit_rate_synthetic_100_steps() {
    // Simulate 100 decode steps with 5 recurring logit patterns.
    // Steps 0,5,10,... produce same logits → cache hit.
    // Steps 1,6,11,... produce same logits → cache hit.
    // Total patterns: 5. Each appears 20 times.
    // Expected: 80 hits (steps 1-19 of each pattern) + 5 misses (first appearance).
    // Hit rate = 95/100 = 95% (well above 60% target).
    let pattern_size = 5;
    let n_steps = 100;
    let n_regions = 10;

    let mut cache = BfcpRegionCache::new(16);

    for step in 0..n_steps {
        let logits = cyclic_logits(step, pattern_size);
        let hash = blake3_logit_hash(&logits);

        match cache.lookup(&hash) {
            Some(_partition) => {
                // Cache hit — no recompute needed.
            }
            None => {
                // Cache miss — recompute partition and insert.
                let partition = make_partition(n_regions, step);
                cache.insert(hash, partition);
            }
        }
    }

    let hit_rate = cache.hit_rate();
    assert!(
        hit_rate >= 0.60,
        "cache hit rate should be >= 60%, got {:.1}%",
        hit_rate * 100.0,
    );

    // Verify cache contents are bounded.
    assert!(
        cache.len() <= 16,
        "cache should not exceed capacity, got {}",
        cache.len(),
    );
}
