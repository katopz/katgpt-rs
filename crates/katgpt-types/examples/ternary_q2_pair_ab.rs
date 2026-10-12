//! Pair-vs-two-calls A/B for the Q2_0 int8-activation matvec on the
//! Bonsai-2-27B FFN gate/up shape (riir-infer Issue 043 T4 attribution).
//!
//! `simd_ternary_q2_matvec_q8_parallel_pair` saves one activation
//! quantize+permute and one rayon fork-join per FFN. Outputs are asserted
//! bit-identical before timing. A third arm times the pair's work as ONE
//! matrix of `2 * rows` rows (one prep, one fork-join), so `pair − merged`
//! isolates the cost of the two-iterator chain itself.
//!
//! Interleaved rounds (one call per arm per round, arm order rotated), median
//! of per-round ratios vs the two-call arm.
//!
//! ```bash
//! cargo run --release -p katgpt-types --features ternary_q2_q8 \
//!     --example ternary_q2_pair_ab
//! ```

use std::hint::black_box;
use std::time::Instant;

use katgpt_types::TernaryGroupWeights;
use katgpt_types::simd::{
    TernaryQ2Weights, simd_ternary_q2_matvec_q8_parallel, simd_ternary_q2_matvec_q8_parallel_pair,
};

const ROUNDS: usize = 41;
const CALLS: usize = 20;

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
    println!("rayon threads: {threads}");
    // (rows, cols): Bonsai-2-27B FFN gate/up.
    let (rows, cols) = (17408, 5120);
    let ga = fixture(rows, cols, 0x043);
    let gb = fixture(rows, cols, 0x044);
    let (wa, wb) = (
        TernaryQ2Weights::from_group(&ga),
        TernaryQ2Weights::from_group(&gb),
    );
    let mut merged_g = TernaryGroupWeights::new(2 * rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            merged_g.set(r, c, ga.get(r, c));
            merged_g.set(rows + r, c, gb.get(r, c));
        }
        for g in 0..ga.groups_per_row {
            merged_g.set_scale(r, g, ga.scale_at(r, g));
            merged_g.set_scale(rows + r, g, gb.scale_at(r, g));
        }
    }
    let merged = TernaryQ2Weights::from_group(&merged_g);
    drop((ga, gb, merged_g));
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5)
        .collect();

    let (mut ya, mut yb) = (vec![0.0f32; rows], vec![0.0f32; rows]);
    let (mut pa, mut pb) = (vec![0.0f32; rows], vec![0.0f32; rows]);
    let mut ym = vec![0.0f32; 2 * rows];
    simd_ternary_q2_matvec_q8_parallel(&wa, &x, &mut ya);
    simd_ternary_q2_matvec_q8_parallel(&wb, &x, &mut yb);
    simd_ternary_q2_matvec_q8_parallel_pair(&wa, &wb, &x, &mut pa, &mut pb);
    simd_ternary_q2_matvec_q8_parallel(&merged, &x, &mut ym);
    assert!(
        ya.iter().zip(&pa).all(|(a, b)| a.to_bits() == b.to_bits()),
        "pair a != single"
    );
    assert!(
        yb.iter().zip(&pb).all(|(a, b)| a.to_bits() == b.to_bits()),
        "pair b != single"
    );
    assert!(
        ya.iter()
            .chain(&yb)
            .zip(&ym)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "merged != singles"
    );

    const LABELS: [&str; 3] = ["two calls", "pair", "merged 2R"];
    let mut times = vec![Vec::with_capacity(ROUNDS); LABELS.len()];
    for round in 0..ROUNDS {
        for k in 0..LABELS.len() {
            let a = (k + round) % LABELS.len();
            let t = Instant::now();
            for _ in 0..CALLS {
                match a {
                    0 => {
                        simd_ternary_q2_matvec_q8_parallel(
                            black_box(&wa),
                            black_box(&x),
                            black_box(&mut ya),
                        );
                        simd_ternary_q2_matvec_q8_parallel(
                            black_box(&wb),
                            black_box(&x),
                            black_box(&mut yb),
                        );
                    }
                    1 => simd_ternary_q2_matvec_q8_parallel_pair(
                        black_box(&wa),
                        black_box(&wb),
                        black_box(&x),
                        black_box(&mut pa),
                        black_box(&mut pb),
                    ),
                    _ => simd_ternary_q2_matvec_q8_parallel(
                        black_box(&merged),
                        black_box(&x),
                        black_box(&mut ym),
                    ),
                }
            }
            black_box((&ya, &yb, &pa, &pb, &ym));
            times[a].push(t.elapsed().as_secs_f64() / CALLS as f64);
        }
    }
    println!("\n## gate+up {rows}x{cols} (x2)");
    let base = times[0].clone();
    for (label, arm_times) in LABELS.iter().zip(&times) {
        let mut ratios: Vec<f64> = arm_times.iter().zip(&base).map(|(t, b)| t / b).collect();
        let mut abs = arm_times.clone();
        println!(
            "{label:>10}: median {:>8.1} us | ratio vs two calls {:.3}",
            median(&mut abs) * 1e6,
            median(&mut ratios)
        );
    }
}
