//! Issue 991 T1 — the Family-B convergence census RUN instrument.
//!
//! Drives the existing micro-belief bench-family input sequences through the
//! REAL instrumented [`LatentThoughtKernel`] path (the census arms inside
//! `step()` via `KATGPT_BENIEF_CONVERGENCE_CENSUS=1`), one (family, K) cell
//! at a time, and prints the per-input converged-at distribution per cell.
//!
//! Run:
//!
//! ```sh
//! KATGPT_BELIEF_CONVERGENCE_CENSUS=1 \
//!   cargo run --release --features micro_belief --example convergence_census
//! ```
//!
//! Report-only: the census arm is bit-identical to the disarmed kernel path
//! (the residual reads copies); the census never changes what step() returns.
//!
//! K sweep: the stack's callers set K (benches run 1/3; the flow-scorer bench
//! scores per-K trajectories), so the census sweeps K until the converged-at
//! distribution stops changing — the mean exit point is K-independent, so a
//! stable mean across the sweep says the distribution has stopped moving.

use katgpt_core::micro_belief::{
    LatentThoughtKernel, MicroRecurrentBeliefState, coherence_bench, convergence_census_armed,
    convergence_census_reset, convergence_census_snapshot,
};

/// K postures swept (the bench trio is 1/3/5; the sweep continues until the
/// converged-at distribution stops changing).
const K_POSTURES: &[u8] = &[3, 5, 8, 16, 32, 64];

/// step() calls per cell (each call = one input = one census row).
const TICKS_PER_CELL: usize = 512;

/// Deterministic sine input generator — the `latent_thought.rs` tests'
/// helper, reused so the census sees the same input shapes the crate's own
/// tests exercise.
fn sine_inputs(dim: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|s| {
            let s = s as f32;
            (0..dim)
                .map(|i| {
                    let f = i as f32;
                    (f * 0.1 + s * 0.01).sin() * 0.5 * (s * 0.003).cos()
                })
                .collect()
        })
        .collect()
}

fn const_inputs(dim: usize, value: f32) -> Vec<Vec<f32>> {
    vec![vec![value; dim]]
}

fn rand_uniform_inputs(dim: usize, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.f32() * 2.0 - 1.0).collect())
        .collect()
}

/// Percentile over the crossed-input converged-at histogram (0-based inner
/// iteration index), linear-interpolated between slots.
fn hist_percentile(hist: &[u64; 64], crossed: u64, p: f64) -> f64 {
    if crossed == 0 {
        return f64::NAN;
    }
    let target = p * (crossed as f64 - 1.0);
    let mut acc = 0u64;
    for (slot, &count) in hist.iter().enumerate() {
        if acc as f64 + count as f64 > target {
            return slot as f64;
        }
        acc += count;
    }
    63.0
}

fn run_cell(label: &str, dim: usize, k: u8, seed: u64, inputs: &[Vec<f32>]) {
    let kernel = LatentThoughtKernel::from_seed(seed, dim, k);
    let mut state = vec![0.0f32; dim];
    convergence_census_reset();
    for tick in 0..TICKS_PER_CELL {
        let x = &inputs[tick % inputs.len()];
        kernel.step(&mut state, x);
    }
    let snap = convergence_census_snapshot()
        .unwrap_or_else(|| panic!("census must stay armed for cell {label}"));
    let crossed = snap.crossed();
    let total_iters = u64::from(k) * snap.inputs;
    let saved_share = if total_iters == 0 {
        0.0
    } else {
        snap.saved_iters as f64 / total_iters as f64
    };
    let exact_share = if snap.inputs == 0 {
        0.0
    } else {
        snap.exact_zero as f64 / snap.inputs as f64
    };
    println!(
        "{label:<24} K={k:<3} inputs={:<5} crossed={:<5.3} never={:<5} mean_exit={:<7.2} \
         p50={:<5.1} p90={:<5.1} max={:<4} exact0%={:<5.3} saved_iters%={:<5.3}",
        snap.inputs,
        snap.crossed_share(),
        snap.never,
        snap.mean_exit(),
        hist_percentile(&snap.hist, crossed, 0.50),
        hist_percentile(&snap.hist, crossed, 0.90),
        // max crossed slot
        snap.hist
            .iter()
            .rposition(|&c| c > 0)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "-".to_string()),
        exact_share,
        saved_share,
    );
}

fn main() {
    if !convergence_census_armed() {
        eprintln!("REFUSED: the census is not armed — run with KATGPT_BELIEF_CONVERGENCE_CENSUS=1");
        std::process::exit(2);
    }

    println!(
        "# Issue 991 T1 convergence census — K sweep until the converged-at \u{201c}\u{201d}distribution stops changing (mean_exit is K-independent)"
    );
    println!(
        "# exact0% = inputs whose residual hit exactly 0 within K (bit-identical successor \u{2014} the strongest early-exit class)"
    );
    println!();

    let coherence = coherence_bench::build_input_sequence(coherence_bench::BENCH_DIM);
    for k in K_POSTURES.iter().copied() {
        run_cell(
            "coherence_seq",
            coherence_bench::BENCH_DIM,
            k,
            42,
            &coherence,
        );
    }
    println!();

    for dim in [8usize, 32usize] {
        let sine = sine_inputs(dim, TICKS_PER_CELL);
        let bench_const = const_inputs(dim, 0.5);
        let rand = rand_uniform_inputs(dim, TICKS_PER_CELL, 0xBEEF_u64);
        for k in K_POSTURES.iter().copied() {
            run_cell(&format!("sine_dim{dim}"), dim, k, 42, &sine);
        }
        for k in K_POSTURES.iter().copied() {
            run_cell(&format!("const0.5_dim{dim}"), dim, k, 42, &bench_const);
        }
        for k in K_POSTURES.iter().copied() {
            run_cell(&format!("rand_uniform_dim{dim}"), dim, k, 42, &rand);
        }
        println!();
    }
}
