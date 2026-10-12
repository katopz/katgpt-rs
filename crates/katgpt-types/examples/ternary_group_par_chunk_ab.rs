//! Rows-per-task A/B for `simd_ternary_group_matvec_parallel` on the
//! Bonsai-2-27B decode shapes (Issue 935).
//!
//! The shipped split is one chunk per rayon worker. On a heterogeneous box
//! (M3 Max: 12 P + 4 E cores) the E-core chunks finish last while the P-cores
//! idle; smaller chunks let work-stealing rebalance. Every chunk size is a pure
//! row partition, so the outputs are asserted bit-identical before timing.
//!
//! Interleaved rounds (one call per arm per round, arm order rotated), median
//! of per-round ratios vs the shipped arm.
//!
//! With `ternary_group_q8_act` the int8-activation SDOT kernel joins as an
//! extra arm (lossy — timed only, its quality gate lives with the consumer).
//!
//! ```bash
//! cargo run --release -p katgpt-types --features ternary_group_q8_act \
//!     --example ternary_group_par_chunk_ab
//! ```

use std::hint::black_box;
use std::time::Instant;

use katgpt_types::TernaryGroupWeights;
use katgpt_types::simd::{
    simd_ternary_group_matvec_parallel, simd_ternary_group_matvec_parallel_chunked,
};

fn fixture(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
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
    w
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("# rayon threads: {threads}");
    // (rows, cols): FFN gate/up, FFN down, GDN qkvz in-proj, GDN out-proj.
    let shapes = [(17408, 5120), (5120, 17408), (16384, 5120), (5120, 6144)];
    const ROUNDS: usize = 41;
    const CALLS: usize = 4;

    for &(rows, cols) in &shapes {
        let w = fixture(rows, cols, 0x935 + rows as u64);
        let x: Vec<f32> = (0..cols).map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5).collect();
        let shipped = rows.div_ceil(threads);
        let chunks = [shipped, 256, 128, 64, 32, 16];

        // G1: bit-identity of every arm vs the shipped entry point.
        let mut y_ref = vec![0.0f32; rows];
        simd_ternary_group_matvec_parallel(&w, &x, &mut y_ref);
        let mut y = vec![0.0f32; rows];
        for &c in &chunks {
            simd_ternary_group_matvec_parallel_chunked(&w, &x, &mut y, c);
            assert!(
                y.iter().zip(&y_ref).all(|(a, b)| a.to_bits() == b.to_bits()),
                "chunk {c} not bit-identical"
            );
        }

        // Arm index == chunks.len() is the q8 kernel when compiled in.
        let n_arms = chunks.len() + usize::from(cfg!(feature = "ternary_group_q8_act"));
        let mut times = vec![Vec::with_capacity(ROUNDS); n_arms];
        for round in 0..ROUNDS {
            for k in 0..n_arms {
                let a = (k + round) % n_arms;
                let t = Instant::now();
                for _ in 0..CALLS {
                    if a < chunks.len() {
                        simd_ternary_group_matvec_parallel_chunked(
                            black_box(&w),
                            black_box(&x),
                            black_box(&mut y),
                            chunks[a],
                        );
                    } else {
                        #[cfg(feature = "ternary_group_q8_act")]
                        katgpt_types::simd::simd_ternary_group_matvec_q8_parallel(
                            black_box(&w),
                            black_box(&x),
                            black_box(&mut y),
                        );
                    }
                }
                black_box(&y);
                times[a].push(t.elapsed().as_secs_f64() / CALLS as f64);
            }
        }
        println!("\n## {rows}x{cols}");
        let base = times[0].clone();
        for (a, arm_times) in times.iter().enumerate() {
            let label = chunks.get(a).map_or_else(|| "q8 sdot".to_owned(), |c| format!("chunk {c:>5}"));
            let mut ratios: Vec<f64> = arm_times.iter().zip(&base).map(|(t, b)| t / b).collect();
            let mut abs = arm_times.clone();
            println!(
                "{label:>11}: median {:>8.1} us | ratio vs shipped {:.3}",
                median(&mut abs) * 1e6,
                median(&mut ratios)
            );
        }
    }
}
