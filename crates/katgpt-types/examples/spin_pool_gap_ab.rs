//! Issue 936 — fork latency after a serial gap: rayon vs the decode spin pool.
//!
//! Each call is `GAP_US` of single-threaded busy work (the decode forward's
//! norms / conv1d / gates between forks) followed by the Bonsai-2-27B FFN
//! gate+up Q2 matvec (`simd_ternary_q2_matvec_q8_parallel_pair`, 2 × 17408 ×
//! 5120). On rayon the workers can park during the gap; on the spin pool they
//! keep spinning. Outputs are asserted bit-identical first. Interleaved rounds,
//! median of per-round ratios (spin / rayon), against rayon called from
//! the main thread AND from a pool worker (the in-model posture).
//!
//! ```bash
//! cargo run --release -p katgpt-types --features ternary_q2_q8,decode_spin_pool \
//!     --example spin_pool_gap_ab
//! ```
//! Env: `SPIN_WORKERS` (default `SpinPool::for_this_machine`: P-cores − 1 on macOS).

use std::hint::black_box;
use std::time::{Duration, Instant};

use katgpt_types::TernaryGroupWeights;
use katgpt_types::simd::{
    SpinPool, TernaryQ2Weights, simd_ternary_q2_matvec_q8_parallel_pair, with_spin_pool,
};

const ROUNDS: usize = 31;
const CALLS: usize = 40;

fn fixture(rows: usize, cols: usize, seed: u64) -> TernaryQ2Weights {
    let mut w = TernaryGroupWeights::new(rows, cols);
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 33
    };
    for r in 0..rows {
        for c in 0..cols {
            match next() % 3 {
                0 => w.set(r, c, -1),
                1 => {}
                _ => w.set(r, c, 1),
            }
        }
        for g in 0..w.groups_per_row {
            w.set_scale(r, g, 0.01 + (r % 7) as f32 * 0.03);
        }
    }
    TernaryQ2Weights::from_group(&w)
}

/// Single-threaded busy work for `d` (stands in for the serial decode sections).
fn busy(d: Duration) -> u64 {
    let t = Instant::now();
    let mut acc = 0u64;
    while t.elapsed() < d {
        acc = acc
            .wrapping_mul(6364136223846793005)
            .wrapping_add(black_box(1));
    }
    acc
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let (rows, cols) = (17408, 5120);
    let (wa, wb) = (fixture(rows, cols, 0x936), fixture(rows, cols, 0x937));
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5)
        .collect();
    // SPIN_WORKERS overrides the worker count (default: P-cores - 1 on macOS).
    let pool = match std::env::var("SPIN_WORKERS") {
        Ok(v) => SpinPool::new(v.parse().expect("SPIN_WORKERS: usize")),
        Err(_) => SpinPool::for_this_machine(),
    };
    println!(
        "rayon threads {} | spin pool workers {} (+ caller)",
        rayon::current_num_threads(),
        pool.workers()
    );

    let (mut ra, mut rb) = (vec![0.0f32; rows], vec![0.0f32; rows]);
    let (mut sa, mut sb) = (vec![0.0f32; rows], vec![0.0f32; rows]);
    simd_ternary_q2_matvec_q8_parallel_pair(&wa, &wb, &x, &mut ra, &mut rb);
    with_spin_pool(&pool, || {
        simd_ternary_q2_matvec_q8_parallel_pair(&wa, &wb, &x, &mut sa, &mut sb)
    });
    assert!(
        ra.iter()
            .chain(&rb)
            .zip(sa.iter().chain(&sb))
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "spin != rayon"
    );

    for gap_us in [0u64, 10, 30, 100] {
        let gap = Duration::from_micros(gap_us);
        // Arm 0: rayon, called from the main thread (a 17th thread).
        // Arm 1: rayon, the loop runs ON a pool worker (riir-infer's
        // `on_rayon_worker`, the in-model posture). Arm 2: the spin pool.
        let mut t = [
            Vec::with_capacity(ROUNDS),
            Vec::with_capacity(ROUNDS),
            Vec::with_capacity(ROUNDS),
        ];
        let mut sink = 0u64;
        for round in 0..ROUNDS {
            for k in 0..3 {
                let arm = (k + round) % 3;
                let t0 = Instant::now();
                let mut call = || {
                    for _ in 0..CALLS {
                        sink = sink.wrapping_add(busy(gap));
                        simd_ternary_q2_matvec_q8_parallel_pair(
                            &wa,
                            &wb,
                            black_box(&x),
                            &mut ra,
                            &mut rb,
                        );
                        black_box(&ra);
                    }
                };
                match arm {
                    0 => call(),
                    1 => rayon::scope(|_| call()),
                    _ => with_spin_pool(&pool, call),
                }
                t[arm].push(t0.elapsed().as_secs_f64() / CALLS as f64 - gap.as_secs_f64());
            }
        }
        black_box(sink);
        let mut vs_main: Vec<f64> = t[2].iter().zip(&t[0]).map(|(s, r)| s / r).collect();
        let mut vs_worker: Vec<f64> = t[2].iter().zip(&t[1]).map(|(s, r)| s / r).collect();
        let [mut a0, mut a1, mut a2] = t;
        println!(
            "gap {gap_us:>3} us: rayon(main) {:>6.1} | rayon(worker) {:>6.1} | spin {:>6.1} us | spin/rayon(worker) {:.3} | spin/rayon(main) {:.3}",
            median(&mut a0) * 1e6,
            median(&mut a1) * 1e6,
            median(&mut a2) * 1e6,
            median(&mut vs_worker),
            median(&mut vs_main)
        );
    }
}
