//! Weight prefetch-distance sweep for the Q2_0 int8-activation matvec
//! (Issue 937 T3), on the Bonsai-2-27B FFN gate/up shape (17408×5120).
//!
//! Each pass streams `MATS` distinct matrices (~24 MB each, ~142 MB total) so
//! the weights cannot sit in the system-level cache — the arm measures DRAM
//! streaming, which is what decode pays. Outputs are asserted bit-identical
//! to the prefetch-off arm before timing. Interleaved rounds (one pass per arm
//! per round, arm order rotated), median of per-round ratios vs `pf = 0`.
//!
//! Quote the box state with the numbers (AC, load, power mode) — under load
//! this measures the scheduler, not the prefetcher.
//!
//! ```bash
//! cargo run --release -p katgpt-types --features q2_prefetch \
//!     --example q2_prefetch_sweep
//! Q2_PF_SWEEP=0,512,2048 cargo run --release ...   # custom distances
//! ```

use std::hint::black_box;
use std::time::Instant;

use katgpt_types::simd::{Q2Block, TernaryQ2Weights, set_q2_prefetch_bytes, simd_ternary_q2_matvec_q8_parallel};

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

fn pass(ws: &[TernaryQ2Weights], x: &[f32], y: &mut [f32]) -> f64 {
    let t = Instant::now();
    for w in ws {
        simd_ternary_q2_matvec_q8_parallel(black_box(w), black_box(x), y);
        black_box(&mut *y);
    }
    t.elapsed().as_secs_f64()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let dists: Vec<usize> = std::env::var("Q2_PF_SWEEP")
        .ok()
        .map(|s| s.split(',').map(|v| v.trim().parse().expect("Q2_PF_SWEEP: comma-separated bytes")).collect())
        .unwrap_or_else(|| vec![0, 256, 512, 1024, 2048, 4096]);
    assert_eq!(dists.first(), Some(&0), "the first distance is the baseline and must be 0");
    println!("rayon threads {} | {MATS} x {ROWS}x{COLS} Q2_0", rayon::current_num_threads());

    let ws: Vec<TernaryQ2Weights> = (0..MATS as u64).map(|i| weights(0x937 + i)).collect();
    let bytes: usize = ws.iter().map(TernaryQ2Weights::storage_bytes).sum();
    let x: Vec<f32> = (0..COLS).map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5).collect();
    let mut y = vec![0.0f32; ROWS];

    // G1: every distance bit-identical to prefetch-off, on every matrix.
    let mut y_ref = vec![0.0f32; ROWS];
    for w in &ws {
        set_q2_prefetch_bytes(0);
        simd_ternary_q2_matvec_q8_parallel(w, &x, &mut y_ref);
        for &pf in &dists[1..] {
            set_q2_prefetch_bytes(pf);
            simd_ternary_q2_matvec_q8_parallel(w, &x, &mut y);
            assert!(
                y.iter().zip(&y_ref).all(|(a, b)| a.to_bits() == b.to_bits()),
                "prefetch {pf} is not bit-identical"
            );
        }
    }
    println!("bit-identical: {} distances x {MATS} matrices", dists.len());

    // Warm every arm once.
    for &pf in &dists {
        set_q2_prefetch_bytes(pf);
        pass(&ws, &x, &mut y);
    }
    let mut secs = vec![Vec::with_capacity(ROUNDS); dists.len()];
    for round in 0..ROUNDS {
        for k in 0..dists.len() {
            let a = (k + round) % dists.len();
            set_q2_prefetch_bytes(dists[a]);
            secs[a].push(pass(&ws, &x, &mut y));
        }
    }
    for (a, &pf) in dists.iter().enumerate() {
        let mut ratios: Vec<f64> = secs[a].iter().zip(&secs[0]).map(|(t, t0)| t / t0).collect();
        let ratio = median(&mut ratios);
        let t = median(&mut secs[a].clone());
        println!(
            "pf {pf:>5} B: {:>7.1} us/matrix | {:>6.1} GB/s | ratio vs off {ratio:.3}",
            t / MATS as f64 * 1e6,
            bytes as f64 / t / 1e9
        );
    }
}
