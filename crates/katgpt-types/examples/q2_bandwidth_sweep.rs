//! Streamed weight bandwidth of the Q2_0 int8-activation matvec vs rayon
//! thread count (Issue 937), on the Bonsai-2-27B FFN gate/up shape
//! (17408×5120).
//!
//! Each pass streams `MATS` distinct matrices (~24 MB each, ~142 MB total) so
//! the weights cannot sit in the system-level cache — the arm measures DRAM
//! streaming, which is what decode pays. One rayon pool per thread count,
//! all in this process; interleaved rounds (one pass per arm per round, arm
//! order rotated), median of per-round ratios vs the FIRST arm, so box load
//! lands on every arm alike. Outputs are asserted bit-identical across arms
//! before timing (fixed row partition ⇒ thread count cannot move numbers).
//!
//! Quote the box state with the numbers (AC, load, power mode).
//!
//! ```bash
//! cargo run --release -p katgpt-types --features ternary_q2_q8 \
//!     --example q2_bandwidth_sweep
//! Q2_BW_THREADS=16,8,10,12 cargo run --release ...   # first = baseline
//! ```

use std::hint::black_box;
use std::time::Instant;

use katgpt_types::simd::{Q2Block, TernaryQ2Weights, simd_ternary_q2_matvec_q8_parallel};

const ROWS: usize = 17408;
const COLS: usize = 5120;
const MATS: usize = 6;
const ROUNDS: usize = 31;

/// Random ternary codes (0..=2) and small f16 scales, built directly as
/// blocks — the bit-plane fixture is too slow at this size.
fn weights(seed: u64) -> TernaryQ2Weights {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let n = ROWS * COLS / 128;
    let blocks = (0..n)
        .map(|_| {
            let mut qs = [0u8; 32];
            for b in &mut qs {
                for k in 0..4 {
                    *b |= ((next() % 3) as u8) << (2 * k);
                }
            }
            Q2Block { d: half::f16::from_f32(0.01 + (next() % 7) as f32 * 0.03).to_bits(), qs }
        })
        .collect();
    TernaryQ2Weights::from_blocks(ROWS, COLS, blocks)
}

fn pass(pool: &rayon::ThreadPool, ws: &[TernaryQ2Weights], x: &[f32], y: &mut [f32]) -> f64 {
    pool.install(|| {
        let t = Instant::now();
        for w in ws {
            simd_ternary_q2_matvec_q8_parallel(black_box(w), black_box(x), y);
            black_box(&mut *y);
        }
        t.elapsed().as_secs_f64()
    })
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let threads: Vec<usize> = std::env::var("Q2_BW_THREADS")
        .ok()
        .map(|s| s.split(',').map(|v| v.trim().parse().expect("Q2_BW_THREADS: comma-separated counts")).collect())
        .unwrap_or_else(|| vec![rayon::current_num_threads(), 8, 10, 12]);
    let pools: Vec<rayon::ThreadPool> = threads
        .iter()
        .map(|&t| rayon::ThreadPoolBuilder::new().num_threads(t).build().expect("rayon pool"))
        .collect();
    println!("arms (threads) {threads:?} | {MATS} x {ROWS}x{COLS} Q2_0");

    let ws: Vec<TernaryQ2Weights> = (0..MATS as u64).map(|i| weights(0x937 + i)).collect();
    let bytes: usize = ws.iter().map(TernaryQ2Weights::storage_bytes).sum();
    let x: Vec<f32> = (0..COLS).map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5).collect();
    let mut y = vec![0.0f32; ROWS];

    // G1: every arm bit-identical to the first, on every matrix.
    let mut y_ref = vec![0.0f32; ROWS];
    for w in &ws {
        pools[0].install(|| simd_ternary_q2_matvec_q8_parallel(w, &x, &mut y_ref));
        for (pool, t) in pools.iter().zip(&threads).skip(1) {
            pool.install(|| simd_ternary_q2_matvec_q8_parallel(w, &x, &mut y));
            assert!(
                y.iter().zip(&y_ref).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{t} threads is not bit-identical"
            );
        }
    }
    println!("bit-identical: {} arms x {MATS} matrices", threads.len());

    for pool in &pools {
        pass(pool, &ws, &x, &mut y);
    }
    let mut secs = vec![Vec::with_capacity(ROUNDS); pools.len()];
    for round in 0..ROUNDS {
        for k in 0..pools.len() {
            let a = (k + round) % pools.len();
            secs[a].push(pass(&pools[a], &ws, &x, &mut y));
        }
    }
    for (a, &t) in threads.iter().enumerate() {
        let mut ratios: Vec<f64> = secs[a].iter().zip(&secs[0]).map(|(s, s0)| s / s0).collect();
        let ratio = median(&mut ratios);
        let s = median(&mut secs[a].clone());
        println!(
            "threads {t:>3}: {:>7.1} us/matrix | {:>6.1} GB/s | ratio vs {} threads {ratio:.3}",
            s / MATS as f64 * 1e6,
            bytes as f64 / s / 1e9,
            threads[0]
        );
    }
}
