#![cfg(feature = "grid_reselect")]
//! Issue 966 G2 — grid re-selection latency vs the per-slot argmax it
//! replaces.
//!
//! The re-selection post-pass (`topk_pools_into` + `reselect_block_into`)
//! sits where the proposal path today takes per-slot argmax over the
//! drafter's step-major marginals. Both arms scan the same `steps × vocab`
//! buffer, so the question G2 answers is the OVERHEAD of keeping a top-K pool
//! and walking it, not the absolute scan cost (which the baseline already
//! pays).
//!
//! Pre-registered bars (written before the first run, 2026-10-27):
//!
//! - **G2 (gate):** paired interleaved median ratio `B / A ≤ 1.50` at
//!   `steps = 16`, `K = 16`, for BOTH `vocab = 32_000` and `vocab = 151_936`
//!   (Qwen-family). A = per-slot argmax; B = top-K extraction + the
//!   prefix-first walk at `λ = 0.5`. A = the stack's SIMD argmax reducer
//!   (`katgpt_types::simd::simd_argmax_f32`), never a scalar strawman.
//! - **Reported, not gated:** best-of µs per block for each arm at each vocab
//!   (the absolute cost the decode step pays; it is dominated by the scan).
//! - **Loud zero:** the shared harness refuses an all-zero arm; both arms are
//!   data-dependent (`black_box` on the marginals AND on `prev`, every output
//!   consumed through a checksum that is asserted non-trivial).
//!
//! Marginals are uniform random f32 — the insertion count into a top-16 pool
//! over V uniform draws is ~K·ln(V/K), the realistic regime for unsorted
//! drafter rows (an ascending-sorted row is the adversarial case and not a
//! drafter output).
//!
//! Run (release is mandatory — a debug latency gate measures nothing):
//!
//! ```sh
//! cargo test -p katgpt-speculative --release --features grid_reselect \
//!   --test bench_966_grid_reselect_g2 -- --nocapture
//! ```

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::{ab_median_ratio, best_of_us};
use katgpt_speculative::bigram_markov::{BigramMarkovBuilder, BigramMarkovTable};
use katgpt_speculative::grid_reselect::{DEFAULT_TOP_K, reselect_block_into, topk_pools_into};
use katgpt_types::simd::simd_argmax_f32;
use std::hint::black_box;
use std::time::Instant;

const STEPS: usize = 16;
const K: usize = DEFAULT_TOP_K;
const LAMBDA: f32 = 0.5;
const G2_MAX_RATIO: f64 = 1.50;

/// The baseline the post-pass replaces: per-slot argmax through the stack's
/// own token-sampling reducer (`simd_argmax_f32` — the NEON single-pass
/// kernel on aarch64), first strict max — the same tie rule the pool's
/// `(score desc, token asc)` head reproduces. A hand-rolled scalar loop is
/// NOT the baseline: measured 2026-10-27 it ran ~2.6× slower than the
/// re-selection arm, which would have made G2 a strawman PASS.
fn argmax_per_slot_into(m: &[f32], steps: usize, vocab: usize, out: &mut [u32]) {
    for (p, o) in out.iter_mut().enumerate().take(steps) {
        *o = simd_argmax_f32(&m[p * vocab..(p + 1) * vocab]).0 as u32;
    }
}

fn table(vocab: usize, rng: &mut fastrand::Rng) -> BigramMarkovTable {
    let mut b = BigramMarkovBuilder::with_capacity(1 << 16);
    for _ in 0..(1 << 16) {
        let p = rng.u32(0..vocab as u32);
        let n = rng.u32(0..vocab as u32);
        b.add_sequence(&[p, n]);
    }
    b.build(vocab, K)
}

fn checksum(out: &[u32]) -> u64 {
    out.iter().fold(0u64, |h, &t| {
        h.wrapping_mul(1_000_003).wrapping_add(u64::from(t))
    })
}

struct Row {
    vocab: usize,
    median: f64,
    min: f64,
    max: f64,
    a_us: f64,
    b_us: f64,
}

fn run_vocab(vocab: usize) -> Row {
    let mut rng = fastrand::Rng::with_seed(966 ^ vocab as u64);
    let t = table(vocab, &mut rng);
    let m: Vec<f32> = (0..STEPS * vocab).map(|_| rng.f32()).collect();
    let mut out_a = vec![0u32; STEPS];
    let mut out_b = vec![0u32; STEPS];
    let mut pools = vec![(0u32, 0f32); STEPS * K];
    let mut sink_a = 0u64;
    let mut sink_b = 0u64;

    // λ = 0 sanity on this exact shape: the walk's head IS argmax (G3 arm,
    // re-asserted here so a broken pool cannot pass G2 by doing less work).
    argmax_per_slot_into(&m, STEPS, vocab, &mut out_a);
    topk_pools_into(&m, STEPS, vocab, K, &mut pools);
    reselect_block_into(&t, 0, STEPS, K, &pools, 0.0, &mut out_b);
    assert_eq!(
        out_a, out_b,
        "λ=0 walk diverged from argmax at vocab {vocab}"
    );

    let iters = if vocab > 100_000 { 8 } else { 32 };
    let ab = ab_median_ratio(
        21,
        iters,
        3,
        |_| {
            argmax_per_slot_into(black_box(&m), STEPS, vocab, &mut out_a);
            sink_a = sink_a.wrapping_add(checksum(black_box(&out_a)));
        },
        |i| {
            let prev = black_box((i % vocab) as u32);
            let n = topk_pools_into(black_box(&m), STEPS, vocab, K, &mut pools);
            reselect_block_into(&t, prev, n, K, black_box(&pools), LAMBDA, &mut out_b);
            sink_b = sink_b.wrapping_add(checksum(black_box(&out_b)));
        },
    );
    ab.report(&format!("966 G2 vocab={vocab} (B/A)"));

    let a_us = best_of_us(3, 15, || {
        let t0 = Instant::now();
        argmax_per_slot_into(black_box(&m), STEPS, vocab, &mut out_a);
        sink_a = sink_a.wrapping_add(checksum(black_box(&out_a)));
        t0.elapsed()
    });
    let b_us = best_of_us(3, 15, || {
        let t0 = Instant::now();
        let n = topk_pools_into(black_box(&m), STEPS, vocab, K, &mut pools);
        reselect_block_into(
            &t,
            black_box(7),
            n,
            K,
            black_box(&pools),
            LAMBDA,
            &mut out_b,
        );
        sink_b = sink_b.wrapping_add(checksum(black_box(&out_b)));
        t0.elapsed()
    });
    assert!(
        sink_a != 0 && sink_b != 0,
        "a sink stayed zero — an arm's output was never produced"
    );
    Row {
        vocab,
        median: ab.median,
        min: ab.min(),
        max: ab.max(),
        a_us,
        b_us,
    }
}

#[test]
fn g2_reselect_overhead_vs_argmax() {
    if cfg!(debug_assertions) {
        eprintln!("966 G2: debug build — timing not read; run with --release");
        return;
    }
    let rows: Vec<Row> = [32_000usize, 151_936].into_iter().map(run_vocab).collect();
    eprintln!("\n966 G2 | vocab | median B/A | [min, max] | argmax µs | reselect µs | bar");
    for r in &rows {
        eprintln!(
            "966 G2 | {:>7} | {:.3} | [{:.3}, {:.3}] | {:.1} | {:.1} | ≤ {G2_MAX_RATIO}",
            r.vocab, r.median, r.min, r.max, r.a_us, r.b_us
        );
    }
    for r in &rows {
        assert!(
            r.median <= G2_MAX_RATIO,
            "G2 FAIL at vocab {}: median B/A {:.3} > {G2_MAX_RATIO}",
            r.vocab,
            r.median
        );
    }
}
