//! Bench — posterior (truncated-Gumbel) inverse sampler GOAT gate
//! (Issue 918; the posterior construction is Zhang et al. 2026 via
//! arXiv:2610.00497 App E).
//!
//! G1 correctness/determinism (live-build rerun of the lib gates' load-
//! bearing arms): the argmax identity over fixtures × winners × keys,
//! bit-determinism of the keyed draw, and the defensive no-write
//! conventions.
//!
//! G2 perf: `keyed_posterior_gumbel_noise` vs `keyed_gumbel_max_sample`
//! at V ∈ {256, 32_000} — the issue's bar is SAME ORDER, ns-class — as an
//! interleaved median-of-ratios over the shared `ab_timing` harness
//! (Issue 855's loud-zero defence: the harness PANICS when an arm
//! measures all-zero, which is exactly how this bench's first draft lost
//! its comparison arm to dead-code elimination). The posterior pays one
//! LSE pass + one exp/log pair per loser on top of the sampler's single
//! pass; the first-measurement bar asserts a ≤5× ratio (generous
//! headroom over the ~2-3× op count, honest about a shared box) and
//! prints both absolute figures with the per-round range.
//!
//! G4 alloc: counting-allocator canary over the hot path — ZERO
//! allocations after warmup (the Issue-741 predicate).
//!
//! Box state: latency numbers are REGRESSION CEILINGS on a shared box —
//! quote the box state beside any reuse (the AGENTS.md G2 box-state law).

#![cfg(feature = "ac_prefix")]

use katgpt_core::ac_prefix::{keyed_gumbel_max_sample, keyed_posterior_gumbel_noise};
use std::hint::black_box;

// Issue 855: the load-invariant timing treatment + the loud-zero defence
// (panics on an all-zero arm instead of letting a ratio be satisfied by a
// loop the optimiser deleted — this bench's first draft lost exactly that
// arm to `let _ =` dead-code elimination).
#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

fn fixture_logits(v: usize, parity: u64) -> Vec<f32> {
    // Deterministic, well-spread logits in a sane magnitude band (the
    // posterior's LSE assumes the verify-loop contract: finite logits).
    (0..v)
        .map(|i| {
            let z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ parity;
            let z = (z ^ (z >> 31)).wrapping_mul(0x853C_49E6_748F_EA9B);
            let frac = ((z >> 40) as f32) / (1u64 << 24) as f32;
            frac * 12.0 - 6.0
        })
        .collect()
}

fn main() {
    assert_counter_is_live();

    // ── G1: live-build rerun of the identity + determinism arms ─────────
    {
        let logits = fixture_logits(64, 7);
        let mut out = vec![0f32; 64];
        let mut again = vec![0f32; 64];
        for pos in 0..500u64 {
            for &winner in &[0u32, 1, 17, 40, 63] {
                keyed_posterior_gumbel_noise(&logits, winner, 0x918, pos, &mut out);
                keyed_posterior_gumbel_noise(&logits, winner, 0x918, pos, &mut again);
                assert_eq!(out, again, "keyed draw must be bit-identical");
                let mut best = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for (k, (&l, &x)) in logits.iter().zip(out.iter()).enumerate() {
                    let s = l + x;
                    if s > best_s {
                        best_s = s;
                        best = k;
                    }
                }
                assert_eq!(best as u32, winner, "pos {pos} winner {winner}");
            }
        }
        // Defensive: out-of-range winner and length mismatch write nothing.
        let mut guard = vec![f32::NAN; 64];
        keyed_posterior_gumbel_noise(&logits, 64, 1, 1, &mut guard);
        assert!(guard.iter().all(|x| x.is_nan()));
        keyed_posterior_gumbel_noise(&logits, 0, 1, 1, &mut guard[..32]);
        assert!(guard.iter().all(|x| x.is_nan()));
        println!("[G1] identity + determinism + defensive: PASS (2_500 draws)");
    }

    // ── G2: same-order latency vs the keyed sampler (interleaved A/B) ──
    for &v in &[256usize, 32_000] {
        let logits = fixture_logits(v, 11);
        let mut out = vec![0f32; v];
        let (rounds, iters, warmup) = if v < 1_000 {
            (7usize, 100usize, 50usize)
        } else {
            (5, 5, 3)
        };

        // Baseline (denominator) arm: the keyed sampler, sink-defended.
        let mut sink = 0u32;
        let a = |i: usize| {
            sink = sink.wrapping_add(keyed_gumbel_max_sample(
                black_box(&logits),
                black_box(5),
                black_box(i as u64),
            ));
        };
        // Candidate (numerator) arm: the posterior sampler.
        let b = |i: usize| {
            keyed_posterior_gumbel_noise(
                black_box(&logits),
                black_box((i * 7) as u32 % v as u32),
                black_box(5),
                black_box(i as u64),
                black_box(&mut out),
            );
        };

        let r = ab_median_ratio(rounds, iters, warmup, a, b);
        black_box(&sink);
        println!(
            "[G2] V={v}: posterior {:.0} ns vs keyed_max_sample {:.0} ns — b/a median {:.2}x (rounds {:.2}..{:.2})",
            r.b_ns_per_iter(),
            r.a_ns_per_iter(),
            r.median,
            r.min(),
            r.max()
        );
        assert!(
            r.median <= 5.0,
            "posterior/sampler median ratio {:.2}x exceeds the 5x same-order bar",
            r.median
        );
        // The absolute ns-class figure is printed, not asserted — it is a
        // shared-box regression ceiling, not a gate.
    }

    // ── G4: zero-alloc canary over the hot path ──────────────────────────
    {
        let logits = fixture_logits(1_024, 13);
        let mut out = vec![0f32; 1_024];
        for i in 0..5 {
            keyed_posterior_gumbel_noise(&logits, (i * 31) as u32 % 1_024, 9, i, &mut out);
        }
        let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        for i in 0..1_000 {
            keyed_posterior_gumbel_noise(
                black_box(&logits),
                black_box((i * 31) as u32 % 1_024),
                black_box(9),
                black_box(i),
                black_box(&mut out),
            );
        }
        let delta = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed) - before;
        assert_eq!(
            delta, 0,
            "posterior sampler allocated {delta}x in 1_000 calls"
        );
        println!("[G4] zero allocs over 1_000 calls: PASS");
    }

    println!("bench_posterior_gumbel: ALL GATES PASS");
}
