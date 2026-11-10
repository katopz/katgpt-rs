//! Plan 576 Phase 4 G4 — zero-alloc steady state for the hint_regret
//! estimator + memory seam. Separate test binary (the house single-fn
//! convention: parallel tests share the global counting allocator —
//! `bench_655_propagation_alloc_check.rs` pattern).

use katgpt_core::hint_regret::{
    HintRegretEstimator, Regime, RegretMemory, RegretMemoryEntry, ReturnBounds, beta_lcb_order_into,
};

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

fn h(seed: u8) -> [u8; 32] {
    [seed; 32]
}

/// G4: zero allocations over 10⁴ estimator pairs (record_pair + amortized
/// estimate) AND zero allocations in the memory seam's steady state
/// (observe/refresh/evict/retire + salience reads + reused ordering
/// scratch). One function so the checks run serially against the shared
/// allocator.
#[test]
fn g4_zero_alloc_steady_state_estimator_and_memory() {
    // ── Estimator: 10^4 pairs + amortized estimates ──────────────────────
    {
        let (_, delta_allocs) = alloc_delta(|| {
            let mut est = HintRegretEstimator::new(ReturnBounds { lo: 0.0, hi: 1.0 });
            let mut sink = 0.0f32;
            for i in 0..10_000u32 {
                est.record_pair(0.5, 0.3 + (i % 7) as f32 * 0.01);
                if i % 16 == 0 {
                    sink += est.estimate(0.05).r_hat;
                }
            }
            assert!(est.n_pairs() == 10_000);
            sink
        });
        assert_eq!(
            delta_allocs, 0,
            "estimator steady state must be alloc-free (got {delta_allocs})"
        );
    }

    // ── Memory seam: observe / refresh / evict / retire / salience reads ──
    // Setup (allocations allowed) happens BEFORE the measured region; only
    // steady-state operations are inside.
    {
        let mut mem = RegretMemory::new(8, 4);
        for i in 0..12u64 {
            let entry = RegretMemoryEntry {
                content_hash: h(i as u8),
                r_hat: 0.1 * i as f32,
                ci: 0.05,
                skill_tag_bits: i as u32,
                last_seen_tick: i,
            };
            mem.observe(entry, Regime::Frontier);
        }
        let mut out: Vec<&RegretMemoryEntry> = Vec::with_capacity(8);
        let scores = [(3u32, 1u32), (10, 2), (0, 4), (7, 7), (2, 0)];
        let mut order = Vec::with_capacity(scores.len());
        let mut lcbs = Vec::with_capacity(scores.len());

        let (result_len, delta_allocs) = alloc_delta(|| {
            // Refresh every live entry (no growth).
            for i in 4..12u64 {
                let entry = RegretMemoryEntry {
                    content_hash: h(i as u8),
                    r_hat: 0.05 * i as f32,
                    ci: 0.02,
                    skill_tag_bits: i as u32,
                    last_seen_tick: 100 + i,
                };
                mem.observe(entry, Regime::Frontier);
            }
            // Retire two (absorbing eviction + tombstone ring push).
            for i in [4u8, 5].iter() {
                let entry = RegretMemoryEntry {
                    content_hash: h(*i),
                    r_hat: 0.0,
                    ci: 0.0,
                    skill_tag_bits: 0,
                    last_seen_tick: 200,
                };
                mem.observe(entry, Regime::Intractable);
            }
            // Refuse a retired hash (absorbing).
            let refused = RegretMemoryEntry {
                content_hash: h(4),
                r_hat: 1.0,
                ci: 0.0,
                skill_tag_bits: 0,
                last_seen_tick: 300,
            };
            mem.observe(refused, Regime::Frontier);
            // Salience reads (reused scratch).
            for now in [210u64, 220, 230, 240] {
                mem.most_salient_into(now, 0.01, &mut out);
                assert!(!out.is_empty());
            }
            // Beta-LCB ordering into reused scratch.
            for _ in 0..100 {
                beta_lcb_order_into(&scores, 0.05, &mut order, &mut lcbs);
            }
            assert_eq!(order.len(), scores.len());
            mem.len()
        });
        let _ = std::hint::black_box(result_len);
        assert_eq!(
            delta_allocs, 0,
            "memory seam steady state must be alloc-free (got {delta_allocs})"
        );
    }
}
